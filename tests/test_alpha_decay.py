"""config#1981: alpha decay curve — compute_decay_curve with full ladder
horizons (1d, 3d, 5d, 10d, 15d, 21d).

Tests that:
1. Insufficient primary-horizon data → status "insufficient_data"
2. Sufficient data → status "ok" with overall decay curve
3. Overall decay has correct horizon_days ordering
4. Score-bucket stratification works
5. Missing horizon columns are gracefully skipped
6. Empty bucket does not crash
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.alpha_decay import (
    _LADDER,
    compute_decay_curve,
    _outcome_columns,
)


def _frame(
    n: int,
    *,
    seed: int = 0,
    with_ladder: bool = True,
    with_score: bool = True,
    nan_primary: bool = False,
    nan_some_horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Build a synthetic score_performance DataFrame with ladder horizon
    outcome columns matching what ``load_score_performance`` produces with
    ``policy=DECAY_POLICY``.

    Args:
        n: number of rows.
        with_ladder: if True, include ALL ladder horizon columns (1d/3d/5d
                     /10d/15d/21d).  If False, only 5d and 21d (the legacy
                     pair, no decay-curve support).
        with_score: if True, include a ``score`` column (needed for bucket
                    stratification).
        nan_primary: if True, set beat_spy_21d to NaN (simulates unresolved
                     outcomes).
        nan_some_horizons: list of horizon_days whose beat_spy column should
                           be NaN (simulates a horizon the producer hasn't
                           backfilled yet).
    """
    rng = np.random.default_rng(seed)
    d: dict[str, object] = {
        "symbol": [f"T{i}" for i in range(n)],
        "score_date": pd.date_range("2026-05-01", periods=n, freq="D").astype(str),
    }
    if with_score:
        d["score"] = rng.uniform(40, 90, n)

    horizons = _LADDER if with_ladder else [5, 21]
    for h in horizons:
        beat_col, ret_col, spy_col = _outcome_columns(h)
        d[beat_col] = rng.integers(0, 2, n).astype(float)
        d[ret_col] = rng.normal(0.02, 0.03, n)  # slight positive bias
        d[spy_col] = rng.normal(0.01, 0.02, n)

        if nan_some_horizons and h in nan_some_horizons:
            d[beat_col] = np.full(n, np.nan)

    if nan_primary:
        p_col, _, _ = _outcome_columns(21)
        d[p_col] = np.full(n, np.nan)

    return pd.DataFrame(d)


# ── Gate / insufficient-data tests ─────────────────────────────────────────────


def test_insufficient_primary_data():
    """Fewer than min_samples rows with resolved 21d → insufficient_data."""
    df = _frame(10, nan_primary=False)
    res = compute_decay_curve(df, min_samples=30)
    assert res["status"] == "insufficient_data"
    assert res["n_signals"] == 10
    assert res["n_needed"] == 30


def test_insufficient_rows():
    """Zero rows → insufficient_data."""
    df = _frame(0, with_ladder=True)
    res = compute_decay_curve(df, min_samples=30)
    assert res["status"] == "insufficient_data"
    assert res["n_signals"] == 0


# ── Happy path ─────────────────────────────────────────────────────────────────


def test_overall_decay_curve():
    """Sufficient data → status ok with overall decay curve."""
    df = _frame(50, with_ladder=True)
    res = compute_decay_curve(df, min_samples=30)
    assert res["status"] == "ok"
    assert res["n_signals"] == 50
    assert len(res["overall"]) == len(_LADDER)

    # Horizons should be in ascending order
    horizons = [p["horizon_days"] for p in res["overall"]]
    assert horizons == sorted(horizons)
    # All ladder horizons should be present
    assert horizons == list(_LADDER)

    # Each point should have the required fields
    for point in res["overall"]:
        assert "avg_alpha" in point
        assert "accuracy" in point
        assert "n" in point
        assert point["n"] <= 50
        assert point["n"] >= 0


def test_decay_curve_without_extra_horizons():
    """Data without the extra ladder horizons (only 5d+21d) should still
    produce a valid curve limited to the available horizons."""
    df = _frame(50, with_ladder=False)
    res = compute_decay_curve(df, min_samples=30)
    assert res["status"] == "ok"
    # Only two horizon points available
    horizons = [p["horizon_days"] for p in res["overall"]]
    assert set(horizons) == {5, 21}


# ── Score-bucket stratification ────────────────────────────────────────────────


def test_score_bucket_stratification():
    """Results include by_score_bucket with decay curves per bucket."""
    df = _frame(200, with_ladder=True)  # enough rows to fill buckets
    res = compute_decay_curve(df, min_samples=30)
    assert res["status"] == "ok"
    assert len(res["by_score_bucket"]) > 0

    for bucket in res["by_score_bucket"]:
        assert "bucket" in bucket
        assert "n" in bucket
        assert "decay_curve" in bucket
        assert len(bucket["decay_curve"]) > 0
        for point in bucket["decay_curve"]:
            assert "horizon_days" in point
            assert "avg_alpha" in point


def test_score_bucket_empty_buckets():
    """Empty buckets (no rows in a range) are skipped without error."""
    df = _frame(50, with_ladder=True)
    # Set all scores very low — only the bottom bucket should have data
    df["score"] = 5.0
    res = compute_decay_curve(df, min_samples=30)
    assert res["status"] == "ok"
    # Empty buckets gracefully excluded
    assert isinstance(res["by_score_bucket"], list)


def test_score_bucket_missing_score_column():
    """DataFrame without a score column → by_score_bucket is empty."""
    df = _frame(50, with_ladder=True, with_score=False)
    res = compute_decay_curve(df, min_samples=30)
    assert res["status"] == "ok"
    assert res["by_score_bucket"] == []


# ── Graceful degradation ───────────────────────────────────────────────────────


def test_missing_columns_skipped():
    """Missing horizon outcome columns are silently skipped (no crash)."""
    df = _frame(50, with_ladder=False)  # only 5d and 21d, no 1d/3d/10d/15d
    res = compute_decay_curve(df, min_samples=30)
    assert res["status"] == "ok"
    # Only available horizons should appear (5d, 21d)
    horizons = {p["horizon_days"] for p in res["overall"]}
    assert 1 not in horizons
    assert 3 not in horizons
    assert 10 not in horizons
    assert 15 not in horizons


def test_partial_nan_horizon_skipped():
    """A horizon column with all NaN values is skipped (insufficient resolved
    rows at that horizon)."""
    df = _frame(50, with_ladder=True, nan_some_horizons=[1])
    res = compute_decay_curve(df, min_samples=30)
    assert res["status"] == "ok"
    horizons = {p["horizon_days"] for p in res["overall"]}
    assert 1 not in horizons  # all NaN → skipped
    assert 21 in horizons  # primary should be fine


def test_partial_nan_primary_insufficient():
    """When the primary 21d horizon is partially NaN but still has enough
    resolved rows, the curve should produce a result (only the starved
    horizons are missing, not the whole result)."""
    df = _frame(50, with_ladder=True, nan_some_horizons=[1, 3])
    res = compute_decay_curve(df, min_samples=30)
    assert res["status"] == "ok"
    # 21d still has 50 resolved rows
    assert res["n_signals"] == 50
    horizons = {p["horizon_days"] for p in res["overall"]}
    assert 1 not in horizons
    assert 3 not in horizons
    assert 21 in horizons
