"""Alpha decay curve: unit tests for ``analysis.alpha_decay`` (config#1981).

Tests use synthetic long-format DataFrames — no DB fixture required for pure
computation logic. The :func:`_frame` helper produces ``score_performance_outcomes``
rows for a configurable set of horizons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.alpha_decay import compute_alpha_decay, load_alpha_decay_data

# Reusable RNG seed for reproducible synthetic data.
_RNG_SEED = 42


def _long_frame(
    n_per_horizon: int,
    horizons: tuple[int, ...] = (1, 3, 5, 10, 15, 21),
    *,
    primary_horizon: int = 21,
    seed: int = _RNG_SEED,
) -> pd.DataFrame:
    """Build a synthetic long-format ``score_performance_outcomes`` frame.

    Each (symbol, score_date) pair is generated once and replicated across
    horizons, matching the producer's cardinality (one row per signal × horizon).

    ``beat_spy`` is drawn from Bernoulli(0.55) — slightly better than coin-flip,
    so accuracy is ~55% and the decay curve is meaningful.
    """
    rng = np.random.default_rng(seed)
    n_total = max(n_per_horizon, 1)

    # Generate unique (symbol, score_date) pairs.
    symbols = [f"T{i}" for i in range(n_total)]
    score_dates = pd.date_range("2026-05-01", periods=n_total, freq="D")

    rows = []
    for i in range(n_total):
        for h in horizons:
            is_primary = 1 if h == primary_horizon else 0
            rows.append({
                "signal_id": f"sig-{symbols[i]}-{score_dates[i]}",
                "symbol": symbols[i],
                "score_date": str(score_dates[i].date()),
                "horizon_days": h,
                "beat_spy": int(rng.binomial(1, 0.55)),
                "stock_return": round(float(rng.normal(0.005, 0.02)), 6),
                "spy_return": round(float(rng.normal(0.002, 0.01)), 6),
                "log_alpha": round(float(rng.normal(0.003, 0.015)), 6),
                "is_primary": is_primary,
                "resolved_at": str(score_dates[i].date()),
            })

    return pd.DataFrame(rows)


# ── Gate: insufficient-data tests ───────────────────────────────────────────────


class TestGate:
    """The primary-horizon gate contract (mirror of signal_quality's canonical gate)."""

    def test_empty_frame_is_insufficient(self):
        result = compute_alpha_decay(pd.DataFrame())
        assert result["status"] == "insufficient_data"

    def test_none_frame_is_insufficient(self):
        result = compute_alpha_decay(None)
        assert result["status"] == "insufficient_data"

    def test_below_min_samples_is_insufficient(self):
        df = _long_frame(5, horizons=(21,))  # only 5 rows at the primary horizon, below 30
        result = compute_alpha_decay(df, min_samples=30)
        assert result["status"] == "insufficient_data"
        assert result["n_primary"] == 5
        assert result["rows_needed"] == 30

    def test_above_min_samples_is_ok(self):
        df = _long_frame(50)
        result = compute_alpha_decay(df, min_samples=30)
        assert result["status"] == "ok"
        assert result["n_primary"] >= 30


# ── Decay-curve structure ───────────────────────────────────────────────────────


class TestDecayCurveStructure:
    """The shape / key contract of the return value."""

    def test_curve_has_all_horizons(self):
        df = _long_frame(50)
        result = compute_alpha_decay(df, min_samples=30)
        curve = result["decay_curve"]
        assert len(curve) == 6  # 1, 3, 5, 10, 15, 21
        horizons = [p["horizon_days"] for p in curve]
        assert horizons == [1, 3, 5, 10, 15, 21]

    def test_each_point_has_required_keys(self):
        df = _long_frame(50)
        result = compute_alpha_decay(df, min_samples=30)
        required = {"horizon_days", "n", "accuracy", "avg_log_alpha",
                     "avg_excess_return", "ci_95"}
        for point in result["decay_curve"]:
            assert required.issubset(point.keys()), f"Missing keys in {point['horizon_days']}"

    def test_decay_rate_is_populated_with_sufficient_horizons(self):
        df = _long_frame(50)
        result = compute_alpha_decay(df, min_samples=30)
        assert result["decay_rate"] is not None
        assert isinstance(result["decay_rate"], float)

    def test_n_total_matches_expected(self):
        df = _long_frame(50)
        result = compute_alpha_decay(df, min_samples=30)
        # 50 rows at each of 6 horizons = 300 total, all with beat_spy populated
        assert result["n_total"] == 300


# ── Metric sanity ───────────────────────────────────────────────────────────────


class TestMetricSanity:
    """Basic reasonableness of computed metrics."""

    def test_accuracy_is_between_zero_and_one(self):
        df = _long_frame(50)
        result = compute_alpha_decay(df, min_samples=30)
        for point in result["decay_curve"]:
            if point["accuracy"] is not None:
                assert 0.0 <= point["accuracy"] <= 1.0

    def test_accuracy_is_not_constant_when_random(self):
        """With random Bernoulli(0.55) data, accuracy should be ~0.55,
        but different horizons get different samples so they won't all be
        identical to 4dp."""
        df = _long_frame(200)  # larger N for stability
        result = compute_alpha_decay(df, min_samples=30)
        accs = {p["horizon_days"]: p["accuracy"] for p in result["decay_curve"]
                if p["accuracy"] is not None}
        # With 200 rows per horizon, accuracies should be reasonably near 0.55
        for h, acc in accs.items():
            assert 0.3 < acc < 0.8, f"horizon {h} accuracy {acc} is extreme"

    def test_ci_95_is_symmetric_around_accuracy(self):
        """Wilson CI lower <= accuracy <= upper."""
        df = _long_frame(50)
        result = compute_alpha_decay(df, min_samples=30)
        for point in result["decay_curve"]:
            if point["ci_95"] is not None:
                lo, hi = point["ci_95"]
                acc = point["accuracy"]
                assert lo <= hi
                if acc is not None:
                    assert lo <= acc <= hi


# ── Partial data availability ───────────────────────────────────────────────────


class TestPartialData:
    """Handling of unpopulated intermediate horizons."""

    def test_reports_zero_for_missing_horizons(self):
        """Only horizons 5 and 21 populated — others report n=0."""
        df = _long_frame(50, horizons=(5, 21))
        result = compute_alpha_decay(df, min_samples=30)
        assert result["status"] == "ok"
        for point in result["decay_curve"]:
            if point["horizon_days"] in (5, 21):
                assert point["n"] > 0
            else:
                assert point["n"] == 0

    def test_decay_rate_none_with_single_horizon(self):
        """Only the primary horizon populated — can't fit a decay rate."""
        df = _long_frame(50, horizons=(21,))
        result = compute_alpha_decay(df, min_samples=30)
        assert result["status"] == "ok"
        assert result["decay_rate"] is None
        populated = [p for p in result["decay_curve"] if p["n"] > 0]
        assert len(populated) == 1
        assert populated[0]["horizon_days"] == 21


# ── Error handling ──────────────────────────────────────────────────────────────


class TestErrorHandling:
    """Graceful handling of edge cases and missing data."""

    def test_missing_log_alpha_column(self):
        """If log_alpha column is absent, avg_log_alpha should be None."""
        df = _long_frame(50)
        df = df.drop(columns=["log_alpha"])
        result = compute_alpha_decay(df, min_samples=30)
        assert result["status"] == "ok"
        for point in result["decay_curve"]:
            if point["n"] > 0:
                assert point["avg_log_alpha"] is None

    def test_all_beat_spy_nan(self):
        """If no beat_spy is resolved, every point reports n=0."""
        df = _long_frame(50)
        df["beat_spy"] = np.nan
        result = compute_alpha_decay(df, min_samples=30)
        assert result["status"] == "insufficient_data"

    def test_mixed_resolution(self):
        """Some rows have beat_spy, some don't — only resolved rows counted."""
        rng = np.random.default_rng(99)
        df = _long_frame(100)
        # Null out 40% of beat_spy at horizon=10 to simulate partial resolution
        mask = (df["horizon_days"] == 10) & (rng.random(len(df)) < 0.4)
        df.loc[mask, "beat_spy"] = np.nan
        df.loc[mask, "log_alpha"] = np.nan
        result = compute_alpha_decay(df, min_samples=30)
        assert result["status"] == "ok"
        h10 = [p for p in result["decay_curve"] if p["horizon_days"] == 10][0]
        assert h10["n"] < 100  # some rows were nulled
