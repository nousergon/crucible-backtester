"""Tests for analysis/factor_blend_sensitivity.py — PR 6 of the scanner-
placement arc (alpha-engine-docs/private/scanner-260514.md).

Pure-Python tests; no S3, no DB. The analyzer reads a DataFrame and emits
a DataFrame, so we exercise it with synthetic score_performance frames.
"""

import math

import numpy as np
import pandas as pd
import pytest

from analysis.factor_blend_sensitivity import (
    KNOWN_STANCES,
    MIN_TRUSTWORTHY_SAMPLES,
    _hit_rate,
    _sortino,
    build_report,
    compute_stance_outcomes,
    detect_mismatches,
)


@pytest.fixture
def regime_weights():
    """Mirrors alpha-engine-config research/scoring.yaml aggregator.factor_blend.
    BULL favors momentum; BEAR favors low_vol."""
    return {
        "bull": {
            "momentum_score": 0.40, "quality_score": 0.30,
            "value_score": 0.20, "low_vol_score": -0.10,
        },
        "bear": {
            "low_vol_score": 0.40, "quality_score": 0.30,
            "momentum_score": -0.20, "value_score": 0.10,
        },
        "neutral": {
            "momentum_score": 0.25, "quality_score": 0.25,
            "value_score": 0.25, "low_vol_score": 0.25,
        },
    }


# ── _sortino ────────────────────────────────────────────────────────────────


class TestSortino:
    def test_positive_returns_with_some_downside(self):
        s = pd.Series([0.05, -0.02, 0.03, -0.01, 0.04])
        result = _sortino(s)
        assert result is not None
        # mean = 0.018, downside dev = sqrt((0.02^2 + 0.01^2)/2) = ~0.0158
        assert result > 0

    def test_all_negative_returns(self):
        s = pd.Series([-0.05, -0.03, -0.02])
        result = _sortino(s)
        assert result is not None
        assert result < 0  # negative Sortino on losses

    def test_no_downside_returns_none(self):
        """All non-negative returns → Sortino undefined."""
        s = pd.Series([0.05, 0.03, 0.01, 0.00])
        assert _sortino(s) is None

    def test_insufficient_data_returns_none(self):
        assert _sortino(pd.Series([0.05])) is None
        assert _sortino(pd.Series([], dtype=float)) is None

    def test_drops_na(self):
        s = pd.Series([0.05, None, -0.02, None, 0.03])
        result = _sortino(s)
        assert result is not None


# ── _hit_rate ───────────────────────────────────────────────────────────────


class TestHitRate:
    def test_half_beats(self):
        s = pd.Series([1, 0, 1, 0])
        assert _hit_rate(s) == 0.5

    def test_all_beats(self):
        assert _hit_rate(pd.Series([1, 1, 1])) == 1.0

    def test_no_beats(self):
        assert _hit_rate(pd.Series([0, 0, 0])) == 0.0

    def test_empty_returns_none(self):
        assert _hit_rate(pd.Series([], dtype=int)) is None

    def test_handles_na(self):
        s = pd.Series([1, None, 0, 1])
        assert _hit_rate(s) == 2 / 3


# ── compute_stance_outcomes ─────────────────────────────────────────────────


def _seed_rows(regime, stance, n, return_distribution):
    """Helper: build n synthetic score_performance rows for one cell."""
    return [
        {
            "market_regime": regime,
            "stance": stance,
            "return_10d": r,
            "spy_10d_return": 0.01,  # constant SPY for simple alpha math
            "beat_spy_10d": int(r > 0.01),
            "return_30d": r * 2,
            "spy_30d_return": 0.02,
            "beat_spy_30d": int(r * 2 > 0.02),
        }
        for r in return_distribution
    ]


class TestComputeStanceOutcomes:
    def test_empty_returns_empty(self):
        result = compute_stance_outcomes(pd.DataFrame())
        assert result.empty

    def test_missing_columns_returns_empty(self):
        # No 'stance' column
        df = pd.DataFrame({"market_regime": ["bull"], "return_10d": [0.05]})
        result = compute_stance_outcomes(df)
        assert result.empty

    def test_basic_grouping(self):
        rows = (
            _seed_rows("bull", "momentum", 5, [0.05, 0.03, -0.01, 0.04, 0.02])
            + _seed_rows("bull", "low_vol", 5, [0.01, 0.00, -0.02, 0.01, 0.00])
        )
        result = compute_stance_outcomes(pd.DataFrame(rows))
        assert len(result) == 2
        assert set(result["stance"]) == {"momentum", "low_vol"}
        for _, row in result.iterrows():
            assert row["n_picks"] == 5
            assert row["market_regime"] == "bull"

    def test_alpha_is_return_minus_spy(self):
        """mean_alpha = mean(return_10d - spy_10d_return)."""
        rows = _seed_rows("bull", "momentum", 3, [0.05, 0.03, 0.01])
        # SPY = 0.01 constant → alphas: 0.04, 0.02, 0.00 → mean = 0.02
        result = compute_stance_outcomes(pd.DataFrame(rows))
        assert result.iloc[0]["mean_alpha"] == pytest.approx(0.02)

    def test_trustworthy_threshold(self):
        # Cell with < MIN_TRUSTWORTHY_SAMPLES → trustworthy=False
        # (compare with == not `is` — pandas stores np.bool_, not Python bool)
        rows = _seed_rows("bull", "momentum", 5, [0.05, 0.03, -0.01, 0.04, 0.02])
        result = compute_stance_outcomes(pd.DataFrame(rows))
        assert bool(result.iloc[0]["trustworthy"]) is False

        # Cell with >= MIN_TRUSTWORTHY_SAMPLES → trustworthy=True
        rng = np.random.default_rng(seed=42)
        returns = rng.normal(0.02, 0.03, size=MIN_TRUSTWORTHY_SAMPLES + 5)
        rows_big = _seed_rows("bull", "momentum", len(returns), returns.tolist())
        result_big = compute_stance_outcomes(pd.DataFrame(rows_big))
        assert bool(result_big.iloc[0]["trustworthy"]) is True

    def test_horizon_30d(self):
        rows = _seed_rows("bull", "momentum", 5, [0.05, 0.03, -0.01, 0.04, 0.02])
        result = compute_stance_outcomes(pd.DataFrame(rows), horizon="30d")
        assert not result.empty
        # 30d returns are 2x in fixture → mean_alpha = 2*(0.026) - 0.02 = 0.032
        assert result.iloc[0]["mean_alpha"] == pytest.approx(0.032)

    def test_rows_missing_data_dropped(self):
        rows = _seed_rows("bull", "momentum", 5, [0.05, 0.03, -0.01, 0.04, 0.02])
        rows.append({
            "market_regime": None, "stance": "momentum",  # missing regime
            "return_10d": 0.5, "spy_10d_return": 0.01, "beat_spy_10d": 1,
            "return_30d": 1.0, "spy_30d_return": 0.02, "beat_spy_30d": 1,
        })
        rows.append({
            "market_regime": "bull", "stance": None,  # missing stance
            "return_10d": -0.99, "spy_10d_return": 0.01, "beat_spy_10d": 0,
            "return_30d": -0.99, "spy_30d_return": 0.02, "beat_spy_30d": 0,
        })
        result = compute_stance_outcomes(pd.DataFrame(rows))
        # Only the 5 good rows contributed
        assert result.iloc[0]["n_picks"] == 5


# ── detect_mismatches ───────────────────────────────────────────────────────


class TestDetectMismatches:
    def test_empty_outcomes_returns_empty(self, regime_weights):
        result = detect_mismatches(pd.DataFrame(), regime_weights)
        assert result.empty

    def test_aligned_config_no_mismatch(self, regime_weights):
        """When configured top stance == realized top stance, mismatch=False."""
        # Seed enough rows for trustworthy=True; momentum beats other stances
        rng = np.random.default_rng(seed=1)
        rows = []
        # momentum: mean 0.04, vol 0.02 → high Sortino
        for r in rng.normal(0.04, 0.02, MIN_TRUSTWORTHY_SAMPLES + 5):
            rows.append({
                "market_regime": "bull", "stance": "momentum",
                "return_10d": r, "spy_10d_return": 0.01,
                "beat_spy_10d": int(r > 0.01),
                "return_30d": r * 2, "spy_30d_return": 0.02,
                "beat_spy_30d": int(r * 2 > 0.02),
            })
        # quality: mean 0.02, vol 0.02 → lower Sortino
        for r in rng.normal(0.02, 0.02, MIN_TRUSTWORTHY_SAMPLES + 5):
            rows.append({
                "market_regime": "bull", "stance": "quality",
                "return_10d": r, "spy_10d_return": 0.01,
                "beat_spy_10d": int(r > 0.01),
                "return_30d": r * 2, "spy_30d_return": 0.02,
                "beat_spy_30d": int(r * 2 > 0.02),
            })
        outcomes = compute_stance_outcomes(pd.DataFrame(rows))
        mismatches = detect_mismatches(outcomes, regime_weights)
        bull_row = mismatches[mismatches["market_regime"] == "bull"].iloc[0]
        assert bull_row["config_top_stance"] == "momentum"
        assert bull_row["realized_top_stance"] == "momentum"
        assert bool(bull_row["mismatch"]) is False

    def test_misaligned_config_detected(self, regime_weights):
        """When realized winner != config winner, mismatch=True flagged.
        Seeds: momentum has clearly negative alpha (mean below SPY);
        quality has positive alpha with some downside (so Sortino is
        defined for both)."""
        rng = np.random.default_rng(seed=2)
        rows = []
        # momentum (config #1 in BULL): mean -0.01, vol 0.03 → alpha mean
        # -0.02 vs SPY 0.01 → clearly negative Sortino
        for r in rng.normal(-0.01, 0.03, MIN_TRUSTWORTHY_SAMPLES + 5):
            rows.append({
                "market_regime": "bull", "stance": "momentum",
                "return_10d": r, "spy_10d_return": 0.01,
                "beat_spy_10d": int(r > 0.01),
                "return_30d": r * 2, "spy_30d_return": 0.02,
                "beat_spy_30d": int(r * 2 > 0.02),
            })
        # quality: mean 0.05, vol 0.025 → alpha mean 0.04 with some
        # downside (P(alpha < 0) ≈ 5%, ~1-2 obs in 25) → positive Sortino
        for r in rng.normal(0.05, 0.025, MIN_TRUSTWORTHY_SAMPLES + 5):
            rows.append({
                "market_regime": "bull", "stance": "quality",
                "return_10d": r, "spy_10d_return": 0.01,
                "beat_spy_10d": int(r > 0.01),
                "return_30d": r * 2, "spy_30d_return": 0.02,
                "beat_spy_30d": int(r * 2 > 0.02),
            })
        outcomes = compute_stance_outcomes(pd.DataFrame(rows))
        mismatches = detect_mismatches(outcomes, regime_weights)
        bull_row = mismatches[mismatches["market_regime"] == "bull"].iloc[0]
        assert bull_row["config_top_stance"] == "momentum"
        assert bull_row["realized_top_stance"] == "quality"
        assert bool(bull_row["mismatch"]) is True

    def test_untrustworthy_cells_excluded_from_realized_order(self, regime_weights):
        """Cells with trustworthy=False (< MIN_TRUSTWORTHY_SAMPLES) drop out
        of realized_order."""
        # Only 3 rows per stance — well below the threshold
        rows = (
            _seed_rows("bull", "momentum", 3, [0.05, 0.03, 0.01])
            + _seed_rows("bull", "quality", 3, [0.10, 0.08, 0.06])
        )
        outcomes = compute_stance_outcomes(pd.DataFrame(rows))
        mismatches = detect_mismatches(outcomes, regime_weights)
        bull_row = mismatches[mismatches["market_regime"] == "bull"].iloc[0]
        # No trustworthy cells → realized_order empty, mismatch=None
        assert bull_row["realized_top_stance"] is None
        assert bull_row["mismatch"] is None
        assert bull_row["n_trustworthy_cells"] == 0

    def test_regime_not_in_config_skipped(self, regime_weights):
        """A realized regime not present in the config (e.g. "caution") is
        skipped — we only check regimes we have config weights for."""
        rng = np.random.default_rng(seed=3)
        rows = []
        for r in rng.normal(0.03, 0.02, MIN_TRUSTWORTHY_SAMPLES + 5):
            rows.append({
                "market_regime": "caution",  # not in regime_weights fixture
                "stance": "low_vol",
                "return_10d": r, "spy_10d_return": 0.01,
                "beat_spy_10d": int(r > 0.01),
                "return_30d": r * 2, "spy_30d_return": 0.02,
                "beat_spy_30d": int(r * 2 > 0.02),
            })
        outcomes = compute_stance_outcomes(pd.DataFrame(rows))
        mismatches = detect_mismatches(outcomes, regime_weights)
        assert "caution" not in mismatches["market_regime"].tolist()


# ── build_report ────────────────────────────────────────────────────────────


class TestBuildReport:
    def test_empty_data_has_data_false(self, regime_weights):
        result = build_report(pd.DataFrame(), regime_weights)
        assert result["has_data"] is False
        assert result["n_total"] == 0
        assert result["outcomes"].empty
        assert result["mismatches"].empty

    def test_horizon_threaded_through(self, regime_weights):
        rows = _seed_rows("bull", "momentum", 5, [0.05, 0.03, -0.01, 0.04, 0.02])
        result = build_report(pd.DataFrame(rows), regime_weights, horizon="30d")
        assert result["horizon"] == "30d"
        assert result["has_data"] is True
        assert result["n_total"] == 5

    def test_known_stances_constant_complete(self):
        """KNOWN_STANCES should match factor_scoring's _STANCE_BY_FACTOR
        labels — pin so a drift triggers a test failure rather than a
        silent miscalibration."""
        assert set(KNOWN_STANCES) == {
            "momentum", "quality", "value", "low_vol",
        }
