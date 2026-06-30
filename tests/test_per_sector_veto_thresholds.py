"""Tests for per-sector veto thresholds (config#921).

compute_per_sector_thresholds is pure compute over a DOWN-prediction DataFrame
(no S3). It reuses the global sweep + gate machinery per sector and emits a
sparse `overrides` map only for sectors whose optimum materially differs from
the global recommendation AND clears the lift/confidence gates.
"""

import pandas as pd
import pytest

from analysis.veto_analysis import compute_per_sector_thresholds, init_config


def _init_cfg(min_change=0.10):
    init_config({
        "veto_analysis": {
            "confidence_thresholds": [0.50, 0.60, 0.70, 0.80],
            "current_default_threshold": 0.60,
            "min_veto_decisions": 3,
            "cost_penalty_weight": 0.30,
            "min_threshold_change": min_change,
            "min_lift_over_base_rate": 0.05,
        }
    })


def _row(sector, conf, beat, ret):
    return {"sector": sector, "prediction_confidence": conf,
            "beat_spy_21d": beat, "return_21d": ret}


def _build_down_df():
    """Two sectors with different veto-precision profiles.

    - 'Energy': vetoing at 0.60/0.70 catches WINNERS (mid-confidence DOWN calls
      that actually beat SPY) → poor precision; only the top confidence band
      (>=0.80) is reliably wrong. So the sector's optimum is 0.80, materially
      higher than a 0.60 global → an override.
    - 'Technology': confidence uncorrelated with outcome → no clean threshold;
      should not earn an override.
    """
    rows = []
    # Energy: 0.70-band DOWN calls mostly BEAT SPY (vetoing them is a mistake)
    for _ in range(8):
        rows.append(_row("Energy", 0.72, 1, 0.09))   # winners — don't veto
    # Energy: 0.80-band DOWN calls reliably underperform (veto correctly)
    for _ in range(8):
        rows.append(_row("Energy", 0.82, 0, -0.10))  # losers — veto these
    # Technology: confidence uncorrelated with outcome → no clean threshold
    for i in range(12):
        rows.append(_row("Technology", 0.65 + (i % 3) * 0.05,
                         i % 2, 0.05 if i % 2 else -0.05))
    return pd.DataFrame(rows)


class TestComputePerSectorThresholds:
    def test_returns_no_sector_column(self):
        _init_cfg()
        df = pd.DataFrame([{"prediction_confidence": 0.7, "beat_spy_21d": 0,
                            "return_21d": -0.1}])
        out = compute_per_sector_thresholds(
            df, base_rate=0.5, thresholds=[0.6, 0.7],
            cost_weight=0.3, current_default=0.6, min_veto_dec=3,
            global_recommended=0.6,
        )
        assert out["status"] == "no_sector_column"
        assert out["overrides"] == {}

    def test_shape_and_keys(self):
        _init_cfg()
        df = _build_down_df()
        out = compute_per_sector_thresholds(
            df, base_rate=0.5, thresholds=[0.50, 0.60, 0.70, 0.80],
            cost_weight=0.3, current_default=0.6, min_veto_dec=3,
            global_recommended=0.60,
        )
        assert set(out) >= {"status", "global_recommended", "by_sector",
                            "overrides", "min_threshold_change"}
        assert "Energy" in out["by_sector"]
        assert "Technology" in out["by_sector"]
        for s in out["by_sector"].values():
            assert set(s) >= {"recommended_threshold", "status", "n_down",
                              "is_override", "delta_vs_global"}

    def test_energy_earns_override_tech_does_not(self):
        _init_cfg()
        df = _build_down_df()
        out = compute_per_sector_thresholds(
            df, base_rate=0.5, thresholds=[0.50, 0.60, 0.70, 0.80],
            cost_weight=0.3, current_default=0.6, min_veto_dec=3,
            global_recommended=0.60,
        )
        # Energy's clean high-confidence underperformers → 0.80 recommendation,
        # which is >= 0.10 from the 0.60 global → an override.
        assert out["by_sector"]["Energy"]["status"] == "ok"
        assert out["overrides"].get("Energy") == 0.80
        # Technology has no gate-clearing edge → not an override.
        assert "Technology" not in out["overrides"]

    def test_no_override_when_within_min_change(self):
        # Raise min_threshold_change so even Energy's 0.80-vs-0.60 (0.20) gap
        # would need >0.20 to override → no overrides emitted.
        _init_cfg(min_change=0.30)
        df = _build_down_df()
        out = compute_per_sector_thresholds(
            df, base_rate=0.5, thresholds=[0.50, 0.60, 0.70, 0.80],
            cost_weight=0.3, current_default=0.6, min_veto_dec=3,
            global_recommended=0.60,
        )
        assert out["overrides"] == {}

    def test_insufficient_sector_data(self):
        _init_cfg()
        # Each sector below min_veto_decisions → no sector fits.
        df = pd.DataFrame([
            _row("Energy", 0.82, 0, -0.1),
            _row("Technology", 0.82, 0, -0.1),
        ])
        out = compute_per_sector_thresholds(
            df, base_rate=0.5, thresholds=[0.80],
            cost_weight=0.3, current_default=0.6, min_veto_dec=3,
            global_recommended=0.60,
        )
        assert out["status"] == "insufficient_sector_data"
        assert out["overrides"] == {}
