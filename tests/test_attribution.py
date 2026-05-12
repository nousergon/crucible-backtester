"""Tests for analysis.attribution — sub-score / predictor correlation against beat-SPY."""

import numpy as np
import pandas as pd
import pytest

from analysis.attribution import (
    SUB_SCORES,
    _resolve_sub_score_columns,
    compute_attribution,
)


def _make_df(
    n: int = 150,
    quant_signal_strength: float = 0.6,
    qual_signal_strength: float = 0.3,
    with_predictor: bool = False,
    with_correct: bool = False,
    seed: int = 7,
) -> pd.DataFrame:
    """Build a synthetic score_performance-shaped DataFrame.

    Sub-scores are noisy linear predictors of the beat_spy_10d / 30d outcome.
    quant_signal_strength > qual_signal_strength is the contract we
    assert on in the ranking tests.
    """
    rng = np.random.default_rng(seed)
    quant = rng.normal(50, 10, n)
    qual = rng.normal(50, 10, n)
    noise_10 = rng.normal(0, 1, n)
    noise_30 = rng.normal(0, 1, n)
    latent_10 = quant_signal_strength * (quant - 50) + qual_signal_strength * (qual - 50) + noise_10
    latent_30 = quant_signal_strength * (quant - 50) + qual_signal_strength * (qual - 50) + noise_30

    beat_10 = (latent_10 > 0).astype(int)
    beat_30 = (latent_30 > 0).astype(int)
    ret_10 = latent_10 / 10
    ret_30 = latent_30 / 10

    df = pd.DataFrame({
        "symbol": [f"T{i % 25}" for i in range(n)],
        "quant_score": quant,
        "qual_score": qual,
        "beat_spy_10d": beat_10,
        "beat_spy_30d": beat_30,
        "return_10d": ret_10,
        "return_30d": ret_30,
    })

    if with_predictor:
        df["p_up"] = rng.uniform(0.3, 0.8, n) + (latent_10 / 50)
        df["p_down"] = 1 - df["p_up"]
        df["prediction_confidence"] = rng.uniform(0.5, 0.9, n)
        df["predicted_direction"] = np.where(df["p_up"] > 0.5, "UP", "DOWN")

    if with_correct:
        df["correct"] = (df["beat_spy_10d"] == 1).astype(int)

    return df


def test_compute_attribution_ok_with_sufficient_rows():
    df = _make_df(n=200)
    result = compute_attribution(df)

    assert result["status"] == "ok"
    assert result["rows_analyzed"] == 200
    assert set(result["correlations"].keys()) == {"quant", "qual"}
    for label in SUB_SCORES:
        assert "beat_spy_10d" in result["correlations"][label]
        assert isinstance(result["correlations"][label]["beat_spy_10d"], float)
    assert result["ranking_10d"][0] == "quant"
    assert result["ranking_30d"][0] == "quant"
    assert result["p_values"]["quant"]["beat_spy_10d"] is not None


def test_compute_attribution_insufficient_rows():
    df = _make_df(n=50)
    result = compute_attribution(df)

    assert result["status"] == "insufficient_data"
    assert result["rows_populated"] == 50
    assert "Week 8+" in result["note"]


def test_compute_attribution_no_sub_score_columns():
    df = pd.DataFrame({
        "symbol": [f"T{i}" for i in range(150)],
        "beat_spy_10d": [1] * 150,
        "beat_spy_30d": [1] * 150,
        "return_10d": [0.01] * 150,
        "return_30d": [0.01] * 150,
    })

    result = compute_attribution(df)

    assert result["status"] == "no_sub_score_columns"
    assert "No sub-score columns" in result["note"]


def test_compute_attribution_filters_nulls_in_beat_spy_10d():
    df = _make_df(n=120).copy()
    df["beat_spy_10d"] = df["beat_spy_10d"].astype(float)
    null_rows = _make_df(n=40, seed=11).copy()
    null_rows["beat_spy_10d"] = float("nan")
    combined = pd.concat([df, null_rows], ignore_index=True)

    result = compute_attribution(combined)

    assert result["status"] == "ok"
    assert result["rows_analyzed"] == 120


def test_compute_attribution_with_predictor_columns():
    df = _make_df(n=200, with_predictor=True, with_correct=True)
    result = compute_attribution(df)

    assert result["status"] == "ok"
    assert "beat_spy_10d" in result["predictor_correlation"]
    assert result["predictor_correlation"]["beat_spy_10d"] is not None
    assert result["predictor_hit_rate"] is not None
    assert 0.0 <= result["predictor_hit_rate"] <= 1.0
    assert result["predictor_hit_rate_ci_95"] is not None


def test_compute_attribution_resolves_sub_scores_from_dict_column():
    n = 150
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "symbol": [f"T{i % 20}" for i in range(n)],
        "sub_scores": [{"quant": float(rng.uniform(40, 60)), "qual": float(rng.uniform(40, 60))} for _ in range(n)],
        "beat_spy_10d": rng.integers(0, 2, n),
        "beat_spy_30d": rng.integers(0, 2, n),
        "return_10d": rng.normal(0, 0.05, n),
        "return_30d": rng.normal(0, 0.05, n),
    })

    result = compute_attribution(df)

    assert result["status"] == "ok"
    assert set(result["correlations"].keys()) == {"quant", "qual"}


def test_compute_attribution_fdr_flag_present_on_correlations():
    df = _make_df(n=200)
    result = compute_attribution(df)

    assert result["status"] == "ok"
    for label in SUB_SCORES:
        for target in ["beat_spy_10d", "beat_spy_30d", "return_10d", "return_30d"]:
            key = f"{target}_fdr_significant"
            assert key in result["correlations"][label]
            assert isinstance(result["correlations"][label][key], (bool, np.bool_))


def test_compute_attribution_predictor_hit_rate_falls_back_to_correct_5d():
    """Pre-2026-05-09 schema only has correct_5d — verify fallback."""
    df = _make_df(n=200, with_predictor=True)
    df["correct_5d"] = (df["beat_spy_10d"] == 1).astype(int)
    result = compute_attribution(df)

    assert result["status"] == "ok"
    assert result["predictor_hit_rate"] is not None


def test_resolve_sub_score_columns_prefers_explicit():
    df = pd.DataFrame({"quant_score": [1, 2], "qual_score": [3, 4], "sub_scores": [{"quant": 10, "qual": 20}, {"quant": 30, "qual": 40}]})
    resolved = _resolve_sub_score_columns(df)
    # Explicit *_score columns win over the dict column.
    assert resolved == {"quant": "quant_score", "qual": "qual_score"}


def test_resolve_sub_score_columns_empty_when_neither_present():
    df = pd.DataFrame({"some_other_col": [1, 2, 3]})
    resolved = _resolve_sub_score_columns(df)
    assert resolved == {}
