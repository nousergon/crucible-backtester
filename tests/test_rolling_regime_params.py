"""Unit tests for analysis/rolling_regime_params.py (config#952).

Covers, with synthetic score_performance frames (no DB, no S3):
  - graceful degradation to a single expanding-span estimate when < 10y history
  - true rolling identification once >= 10y of history exists
  - the overfitting guardrails: min-sample trust floor + cross-window
    sign-stability (a sign-flipping stance is flagged unstable)
  - top-stance durability per regime
  - always-emit report section (thin data + ok + never-raise)
  - firewall: module exposes no apply/write surface
"""

from __future__ import annotations

import pandas as pd
import pytest

from analysis.rolling_regime_params import (
    WINDOW_DAYS,
    build_rolling_regime_params_section,
    identify_rolling_regime_params,
    summarize_parameter_stability,
    _window_bounds,
)

HORIZON = "21d"


def _rows(regime, stance, dates, returns, spy=0.01):
    """Synthetic score_performance rows for one (regime, stance) cell at the
    canonical 21d horizon."""
    return [
        {
            "score_date": pd.Timestamp(d),
            "market_regime": regime,
            "stance": stance,
            "return_21d": r,
            "spy_21d_return": spy,
            "beat_spy_21d": int(r > spy),
        }
        for d, r in zip(dates, returns)
    ]


def _dates(start, n, step_days=1):
    base = pd.Timestamp(start)
    return [base + pd.Timedelta(days=step_days * i) for i in range(n)]


# ── window bounds ───────────────────────────────────────────────────────────


class TestWindowBounds:
    def test_span_shorter_than_window_yields_no_windows(self):
        dates = pd.Series(_dates("2024-01-01", 100))  # ~100 days
        assert _window_bounds(dates, WINDOW_DAYS, 91) == []

    def test_span_longer_than_window_yields_multiple_windows(self):
        # 12 years of monthly stamps
        dates = pd.Series(_dates("2010-01-01", 12 * 12, step_days=30))
        bounds = _window_bounds(dates, WINDOW_DAYS, 91)
        assert len(bounds) >= 2
        # oldest first, each window is exactly WINDOW_DAYS wide
        assert bounds[0][0] <= bounds[-1][0]
        for start, end in bounds:
            assert (end - start).days == WINDOW_DAYS


# ── insufficient history → expanding-span estimate ──────────────────────────


class TestInsufficientHistory:
    def test_flags_and_degrades(self):
        rows = _rows("bull", "momentum", _dates("2024-01-01", 40), [0.05] * 40)
        result = identify_rolling_regime_params(pd.DataFrame(rows), horizon=HORIZON)
        assert result["insufficient_history_for_rolling"] is True
        assert result["mode"] == "expanding_single"
        assert result["n_windows"] == 1
        # 40 >= min_samples(20) → a trustworthy cell exists → status ok
        assert result["status"] == "ok"
        # No cross-window stability from a single window.
        assert result["stability"] == {}
        # But the single-window top stance IS identified.
        assert result["top_stance_stability"]["bull"]["modal_top_stance"] == "momentum"

    def test_thin_data_is_insufficient_not_crash(self):
        rows = _rows("bull", "momentum", _dates("2024-01-01", 5), [0.05] * 5)
        result = identify_rolling_regime_params(pd.DataFrame(rows), horizon=HORIZON)
        # 5 < min_samples → no trustworthy cell → insufficient
        assert result["status"] == "insufficient"

    def test_empty_frame(self):
        result = identify_rolling_regime_params(pd.DataFrame(), horizon=HORIZON)
        assert result["status"] == "insufficient"
        assert result["n_rows"] == 0


# ── true rolling identification (>= 10y history) ────────────────────────────


def _decade_frame():
    """~12 years of weekly picks. bull/momentum is a durable positive-alpha
    stance (mostly winning, occasional small loss so Sortino is defined);
    bull/value is a mediocre noisy-around-zero stance; bear/momentum is
    mildly negative. A 10y window over a 12y span overlaps ~8y between
    windows, so this frame deliberately tests the durable-signal path — the
    sign-instability guardrail is unit-tested separately at the
    summarize level (a mid-timeline flip is invisible at 10y-window
    resolution over only 12y, by construction)."""
    dates = _dates("2008-01-01", 12 * 52, step_days=7)
    rows: list[dict] = []
    for d in dates:
        # durable positive-alpha momentum in bull (one small loss for realism
        # so _sortino is defined and clearly high)
        rows += _rows("bull", "momentum", [d] * 4, [0.05, 0.04, 0.06, -0.005])
        # mediocre, noisy-around-zero value
        rows += _rows("bull", "value", [d] * 4, [0.015, -0.01, 0.005, -0.008])
        # a bear cell, mildly negative
        rows += _rows("bear", "momentum", [d] * 4, [-0.02, -0.03, -0.01, 0.005])
    return pd.DataFrame(rows)


class TestRollingIdentification:
    def test_rolling_mode_and_windows(self):
        result = identify_rolling_regime_params(_decade_frame(), horizon=HORIZON)
        assert result["mode"] == "rolling"
        assert result["insufficient_history_for_rolling"] is False
        assert result["n_windows"] >= 2
        assert result["status"] == "ok"

    def test_durable_stance_marked_stable(self):
        result = identify_rolling_regime_params(_decade_frame(), horizon=HORIZON)
        mom = result["stability"]["bull"]["momentum"]
        assert mom["stable"] is True
        assert mom["sign_consistency"] == pytest.approx(1.0)

    def test_top_stance_durability(self):
        result = identify_rolling_regime_params(_decade_frame(), horizon=HORIZON)
        # bull's durable positive momentum beats mediocre value every window
        tss = result["top_stance_stability"]["bull"]
        assert tss["modal_top_stance"] == "momentum"
        assert tss["top_stance_consistency"] == pytest.approx(1.0)


class TestSignInstabilityGuardrail:
    """A stance whose realized-alpha sign flips window-to-window is regime
    noise and must be flagged unstable — the core overfitting guardrail.
    Tested directly on summarize_parameter_stability with explicit windows
    (the only clean way to encode a cross-window sign flip)."""

    def _flipping_windows(self):
        # 4 windows: value flips + - + - ; momentum stays +
        signs = [0.03, -0.03, 0.03, -0.03]
        windows = []
        for a in signs:
            windows.append({"params": {"bull": {
                "value": {"mean_alpha": a, "sortino": 0.1, "n_picks": 50,
                          "trustworthy": True},
                "momentum": {"mean_alpha": 0.04, "sortino": 1.0, "n_picks": 50,
                             "trustworthy": True},
            }}})
        return windows

    def test_flipping_stance_unstable(self):
        stab = summarize_parameter_stability(self._flipping_windows(), min_samples=20)
        val = stab["bull"]["value"]
        assert val["n_windows_present"] == 4
        assert val["sign_consistency"] == pytest.approx(0.5)
        assert val["stable"] is False

    def test_consistent_stance_stable(self):
        stab = summarize_parameter_stability(self._flipping_windows(), min_samples=20)
        mom = stab["bull"]["momentum"]
        assert mom["sign_consistency"] == pytest.approx(1.0)
        assert mom["stable"] is True


# ── stability summary unit ──────────────────────────────────────────────────


class TestStabilitySummary:
    def test_untrustworthy_cells_excluded(self):
        windows = [
            {"params": {"bull": {"momentum": {
                "mean_alpha": 0.05, "sortino": 1.0, "n_picks": 5,
                "trustworthy": False}}}},
            {"params": {"bull": {"momentum": {
                "mean_alpha": 0.05, "sortino": 1.0, "n_picks": 50,
                "trustworthy": True}}}},
        ]
        stab = summarize_parameter_stability(windows, min_samples=20)
        # only the trustworthy window counts
        assert stab["bull"]["momentum"]["n_windows_present"] == 1

    def test_coef_var_none_on_zero_mean(self):
        windows = [
            {"params": {"bull": {"m": {"mean_alpha": 0.05, "sortino": None,
                                       "n_picks": 50, "trustworthy": True}}}},
            {"params": {"bull": {"m": {"mean_alpha": -0.05, "sortino": None,
                                       "n_picks": 50, "trustworthy": True}}}},
        ]
        stab = summarize_parameter_stability(windows, min_samples=20)
        assert stab["bull"]["m"]["coef_var"] is None  # mean ~ 0
        assert stab["bull"]["m"]["sign_consistency"] == pytest.approx(0.5)


# ── report section (always-emit) ────────────────────────────────────────────


class TestReportSection:
    def test_insufficient_section(self):
        md = build_rolling_regime_params_section(
            {"status": "insufficient", "n_rows": 5, "span_days": 4}
        )
        assert md.startswith("## 10y rolling regime parameters")
        assert "Insufficient data" in md

    def test_none_result_never_raises(self):
        md = build_rolling_regime_params_section(None)
        assert md.startswith("## 10y rolling regime parameters")

    def test_rolling_section_has_tables_and_firewall_note(self):
        result = identify_rolling_regime_params(_decade_frame(), horizon=HORIZON)
        md = build_rolling_regime_params_section(result)
        assert "Best stance per regime" in md
        assert "parameter stability" in md
        assert "Observability only" in md  # firewall note present
        assert "curve-fitting firewall" in md

    def test_expanding_section_flags_short_history(self):
        rows = _rows("bull", "momentum", _dates("2024-01-01", 40), [0.05] * 40)
        result = identify_rolling_regime_params(pd.DataFrame(rows), horizon=HORIZON)
        md = build_rolling_regime_params_section(result)
        assert "History < 10y" in md


# ── firewall: no apply/write surface ────────────────────────────────────────


def test_module_exposes_no_apply_surface():
    import analysis.rolling_regime_params as m

    for forbidden in ("apply", "write", "put_object", "save", "emit_to_s3"):
        assert not hasattr(m, forbidden), (
            f"{forbidden} must not exist — module is analysis-only (firewall)"
        )
