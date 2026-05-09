"""Tests for analyze_cio_performance — significance + sample-size gates.

These gates were tightened in the 2026-05-09 evaluator-email arc after
observing that the prior version flipped CIO to deterministic mode on
n_advance=31 and a -0.06% vs-ranking lift — well within sampling noise.
The new gates mirror the executor optimizer's PSR-confidence pattern.

Locked behavior:
  - Sample-size floor (n_advance ≥ MIN_ADVANCE_SAMPLES) blocks early flips
  - Symmetric thresholds (vs-ranking AND absolute-lift both must clear -0.5%)
  - Welch t-stat ≤ -1.96 required (one-sided significance check)
  - t-stat unavailable → decline to flip (legacy artifacts pre-stdev emission)
"""

from __future__ import annotations

import pytest

from optimizer.pipeline_optimizer import (
    _MIN_ADVANCE_SAMPLES,
    _MIN_T_FOR_DETERMINISTIC,
    _MIN_WEEKS,
    _NEGATIVE_LIFT_THRESHOLD,
    _CIO_MIN_LIFT_TO_KEEP,
    _welch_t_stat,
    analyze_cio_performance,
)


def _cio_lift_dict(
    *,
    advance_avg: float = -0.04,
    all_recs_avg: float = -0.01,
    advance_std: float = 0.05,
    all_recs_std: float = 0.05,
    n_advance: int = 200,
    n_recs: int = 400,
) -> dict:
    """Build a cio_lift dict shaped like _cio_lift's return value."""
    return {
        "advance_avg": advance_avg,
        "all_recs_avg": all_recs_avg,
        "lift": round(advance_avg - all_recs_avg, 4),
        "n_advance": n_advance,
        "n_recs": n_recs,
        "advance_std_5d": advance_std,
        "all_recs_std_5d": all_recs_std,
    }


def _e2e_dict(
    *,
    cio_lift: dict | float | None = None,
    cio_vs_ranking_lift: float = -0.01,
    n_dates: int = 8,
) -> dict:
    return {
        "status": "ok",
        "n_dates": n_dates,
        "cio_lift": cio_lift if cio_lift is not None else _cio_lift_dict(),
        "cio_vs_ranking": {"lift": cio_vs_ranking_lift, "n_dates": n_dates},
    }


# ── Welch t-stat helper ─────────────────────────────────────────────────────


class TestWelchTStat:
    def test_returns_none_on_missing_inputs(self):
        assert _welch_t_stat(None, 0, 1, 1, 10, 10) is None
        assert _welch_t_stat(0, None, 1, 1, 10, 10) is None
        assert _welch_t_stat(0, 0, None, 1, 10, 10) is None
        assert _welch_t_stat(0, 0, 1, 1, None, 10) is None

    def test_returns_none_when_n_below_two(self):
        assert _welch_t_stat(0, 0, 1, 1, 1, 10) is None
        assert _welch_t_stat(0, 0, 1, 1, 10, 1) is None

    def test_returns_none_when_se_zero(self):
        assert _welch_t_stat(0, 0, 0, 0, 10, 10) is None

    def test_negative_lift_yields_negative_t(self):
        # advance avg lower than all_recs avg → negative t (unfavorable)
        t = _welch_t_stat(-0.05, -0.01, 0.04, 0.04, 100, 200)
        assert t is not None and t < 0

    def test_positive_lift_yields_positive_t(self):
        t = _welch_t_stat(0.02, -0.01, 0.04, 0.04, 100, 200)
        assert t is not None and t > 0

    def test_known_value(self):
        # Hand-computed: SE = sqrt(0.04^2/100 + 0.04^2/200)
        #              = sqrt(0.000016 + 0.000008) = sqrt(0.000024) ≈ 0.0049
        # t = (-0.05 - 0.0)/0.0049 ≈ -10.21
        t = _welch_t_stat(-0.05, 0.0, 0.04, 0.04, 100, 200)
        assert t is not None
        assert -10.5 < t < -9.9


# ── analyze_cio_performance — gating ────────────────────────────────────────


class TestAnalyzeCioPerformance:
    def test_rejects_when_e2e_status_not_ok(self):
        result = analyze_cio_performance({"status": "skipped"})
        assert result["status"] == "insufficient_data"

    def test_rejects_when_n_dates_below_min_weeks(self):
        result = analyze_cio_performance(_e2e_dict(n_dates=_MIN_WEEKS - 1))
        assert result["status"] == "insufficient_data"
        assert result["n_weeks"] == _MIN_WEEKS - 1
        assert result["min_required"] == _MIN_WEEKS

    def test_rejects_when_n_advance_below_floor(self):
        # n_advance=31 like the 2026-05-09 email — must not flip
        e2e = _e2e_dict(
            cio_lift=_cio_lift_dict(n_advance=31, advance_avg=-0.04),
            cio_vs_ranking_lift=-0.01,
        )
        result = analyze_cio_performance(e2e)
        assert result["status"] == "insufficient_advance_samples"
        assert result["recommendation"] == "keep_llm"
        assert result["n_advance"] == 31
        assert result["min_required"] == _MIN_ADVANCE_SAMPLES

    def test_keeps_llm_when_lift_above_threshold(self):
        # Sample size + significance both fine, but absolute lift only
        # marginally negative (above the -0.5% effect-size floor).
        e2e = _e2e_dict(
            cio_lift=_cio_lift_dict(
                advance_avg=-0.0010, all_recs_avg=0.0,
                advance_std=0.001, all_recs_std=0.001,
                n_advance=200, n_recs=400,
            ),
            cio_vs_ranking_lift=-0.01,
        )
        result = analyze_cio_performance(e2e)
        assert result["status"] == "ok"
        assert result["recommendation"] == "keep_llm"

    def test_keeps_llm_when_vs_ranking_above_threshold(self):
        # Symmetric gate: vs-ranking must clear -0.5% just like absolute.
        e2e = _e2e_dict(
            cio_lift=_cio_lift_dict(
                advance_avg=-0.05, all_recs_avg=0.0,
                advance_std=0.04, all_recs_std=0.04,
                n_advance=200, n_recs=400,
            ),
            cio_vs_ranking_lift=-0.001,  # -0.1% — above -0.5% threshold
        )
        result = analyze_cio_performance(e2e)
        assert result["status"] == "ok"
        assert result["recommendation"] == "keep_llm"

    def test_keeps_llm_when_t_stat_above_significance(self):
        # Large effect size in the lift but huge stdev → |t| < 1.96
        e2e = _e2e_dict(
            cio_lift=_cio_lift_dict(
                advance_avg=-0.06, all_recs_avg=0.0,
                advance_std=0.30, all_recs_std=0.30,  # huge noise
                n_advance=120, n_recs=300,
            ),
            cio_vs_ranking_lift=-0.06,
        )
        result = analyze_cio_performance(e2e)
        assert result["status"] == "ok"
        assert result["recommendation"] == "keep_llm"
        assert result["t_stat"] is not None
        assert abs(result["t_stat"]) < _MIN_T_FOR_DETERMINISTIC

    def test_keeps_llm_when_t_stat_unavailable(self):
        # Legacy cio_lift dict pre-dating the stdev emission. The gate
        # must decline rather than divide by zero or treat as
        # significant — confidence_unavailable → keep_llm.
        legacy = {
            "advance_avg": -0.05, "all_recs_avg": 0.0,
            "lift": -0.05,
            "n_advance": 200, "n_recs": 400,
            # NO advance_std_5d / all_recs_std_5d
        }
        e2e = _e2e_dict(cio_lift=legacy, cio_vs_ranking_lift=-0.05)
        result = analyze_cio_performance(e2e)
        assert result["status"] == "ok"
        assert result["recommendation"] == "keep_llm"
        assert result["t_stat"] is None
        assert "t-stat unavailable" in result["reasoning"]

    def test_recommends_deterministic_only_when_all_gates_pass(self):
        # All four gates pass:
        #   - n_dates ≥ MIN_WEEKS
        #   - n_advance ≥ MIN_ADVANCE_SAMPLES
        #   - lift < -0.5% AND vs-ranking < -0.5%
        #   - |t| ≥ 1.96 with t negative
        e2e = _e2e_dict(
            cio_lift=_cio_lift_dict(
                advance_avg=-0.05, all_recs_avg=0.0,
                advance_std=0.05, all_recs_std=0.05,
                n_advance=200, n_recs=400,
            ),
            cio_vs_ranking_lift=-0.04,
        )
        result = analyze_cio_performance(e2e)
        assert result["status"] == "ok"
        assert result["recommendation"] == "deterministic"
        assert result["t_stat"] is not None
        assert result["t_stat"] <= -_MIN_T_FOR_DETERMINISTIC

    def test_positive_t_does_not_recommend_deterministic(self):
        # Effect goes the OTHER way — CIO outperforms — must not flip
        # just because |t| ≥ 1.96.
        e2e = _e2e_dict(
            cio_lift=_cio_lift_dict(
                advance_avg=0.05, all_recs_avg=0.0,
                advance_std=0.05, all_recs_std=0.05,
                n_advance=200, n_recs=400,
            ),
            cio_vs_ranking_lift=0.04,
        )
        result = analyze_cio_performance(e2e)
        assert result["status"] == "ok"
        assert result["recommendation"] == "keep_llm"
        assert result["t_stat"] > 0

    def test_2026_05_09_data_does_not_flip(self):
        """Replay of the 2026-05-09 evaluator email — the inputs that
        previously tripped the (broken) gate must NOT flip now.

        From research.db join: n_advance=31, advance_avg=-0.0111,
        all_recs_avg=-0.0086, advance_std≈0.0556, all_recs_std≈0.06,
        n_recs=65, cio_vs_ranking_lift=-0.0006.
        """
        e2e = _e2e_dict(
            cio_lift=_cio_lift_dict(
                advance_avg=-0.0111, all_recs_avg=-0.0086,
                advance_std=0.0556, all_recs_std=0.06,
                n_advance=31, n_recs=65,
            ),
            cio_vs_ranking_lift=-0.0006,
            n_dates=8,
        )
        result = analyze_cio_performance(e2e)
        # Sample-size gate fires first — n_advance=31 < 100
        assert result["status"] == "insufficient_advance_samples"
        assert result["recommendation"] == "keep_llm"


# ── Constant sanity ─────────────────────────────────────────────────────────


def test_threshold_symmetry():
    """The vs-ranking gate must mirror the absolute-lift gate to avoid
    the asymmetry that previously made vs-ranking the binding constraint."""
    assert _CIO_MIN_LIFT_TO_KEEP == _NEGATIVE_LIFT_THRESHOLD


def test_min_advance_samples_is_decision_grade():
    """Locked at 100. Drift would re-introduce small-sample false positives."""
    assert _MIN_ADVANCE_SAMPLES >= 100
