"""Tests for analysis.attribution — sub-score / predictor correlation against beat-SPY."""

import numpy as np
import pandas as pd
import pytest

from analysis.attribution import (
    SUB_SCORES,
    _compute_multivariate_attribution,
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

    Sub-scores are noisy linear predictors of the canonical beat_spy_21d
    outcome (config#1456). quant_signal_strength > qual_signal_strength is
    the contract we assert on in the ranking tests.
    """
    rng = np.random.default_rng(seed)
    quant = rng.normal(50, 10, n)
    qual = rng.normal(50, 10, n)
    noise_21 = rng.normal(0, 1, n)
    latent_21 = quant_signal_strength * (quant - 50) + qual_signal_strength * (qual - 50) + noise_21

    beat_21 = (latent_21 > 0).astype(int)
    ret_21 = latent_21 / 10

    df = pd.DataFrame({
        "symbol": [f"T{i % 25}" for i in range(n)],
        "quant_score": quant,
        "qual_score": qual,
        "beat_spy_21d": beat_21,
        "return_21d": ret_21,
    })

    if with_predictor:
        df["p_up"] = rng.uniform(0.3, 0.8, n) + (latent_21 / 50)
        df["p_down"] = 1 - df["p_up"]
        df["prediction_confidence"] = rng.uniform(0.5, 0.9, n)
        df["predicted_direction"] = np.where(df["p_up"] > 0.5, "UP", "DOWN")

    if with_correct:
        df["correct"] = (df["beat_spy_21d"] == 1).astype(int)

    return df


def test_compute_attribution_ok_with_sufficient_rows():
    df = _make_df(n=200)
    result = compute_attribution(df)

    assert result["status"] == "ok"
    assert result["rows_analyzed"] == 200
    assert set(result["correlations"].keys()) == {"quant", "qual"}
    for label in SUB_SCORES:
        assert "beat_spy_21d" in result["correlations"][label]
        assert isinstance(result["correlations"][label]["beat_spy_21d"], float)
    assert result["ranking_21d"][0] == "quant"
    assert result["p_values"]["quant"]["beat_spy_21d"] is not None


def test_compute_attribution_insufficient_rows():
    df = _make_df(n=50)
    result = compute_attribution(df)

    assert result["status"] == "insufficient_data"
    assert result["rows_populated"] == 50
    assert "Week 8+" in result["note"]


def test_compute_attribution_no_sub_score_columns():
    df = pd.DataFrame({
        "symbol": [f"T{i}" for i in range(150)],
        "beat_spy_21d": [1] * 150,
        "return_21d": [0.01] * 150,
    })

    result = compute_attribution(df)

    assert result["status"] == "no_sub_score_columns"
    assert "No sub-score columns" in result["note"]


def test_compute_attribution_filters_nulls_in_beat_spy_21d():
    df = _make_df(n=120).copy()
    df["beat_spy_21d"] = df["beat_spy_21d"].astype(float)
    null_rows = _make_df(n=40, seed=11).copy()
    null_rows["beat_spy_21d"] = float("nan")
    combined = pd.concat([df, null_rows], ignore_index=True)

    result = compute_attribution(combined)

    assert result["status"] == "ok"
    assert result["rows_analyzed"] == 120


def test_compute_attribution_with_predictor_columns():
    df = _make_df(n=200, with_predictor=True, with_correct=True)
    result = compute_attribution(df)

    assert result["status"] == "ok"
    assert "beat_spy_21d" in result["predictor_correlation"]
    assert result["predictor_correlation"]["beat_spy_21d"] is not None
    assert result["predictor_hit_rate"] is not None
    assert 0.0 <= result["predictor_hit_rate"] <= 1.0
    assert result["predictor_hit_rate_ci_95"] is not None


def test_compute_attribution_predictor_merged_into_correlations(monkeypatch):
    """config#2305 fix: predictor's FDR-significance flag must be visible on
    `correlations` too — not just the separate `predictor_correlation` dict —
    so a consumer that counts significant flags by scanning
    `correlations.values()` (crucible-evaluator's fdr_surface_health tile)
    sees the FULL test surface that fed the shared BH-FDR correction, not an
    undercount missing the predictor's own test."""
    df = _make_df(n=200, with_predictor=True, with_correct=True)
    result = compute_attribution(df)

    assert "predictor" in result["correlations"]
    assert "beat_spy_21d" in result["correlations"]["predictor"]
    assert "beat_spy_21d_fdr_significant" in result["correlations"]["predictor"]
    assert isinstance(
        result["correlations"]["predictor"]["beat_spy_21d_fdr_significant"],
        (bool, np.bool_),
    )
    # And it must agree with the FDR flag the shared correction actually
    # computed for the predictor label (not a hardcoded True/False).
    from analysis.stats_utils import benjamini_hochberg

    all_pvals_ordered = (
        [result["p_values"]["quant"][t] for t in ("beat_spy_21d", "return_21d")]
        + [result["p_values"]["qual"][t] for t in ("beat_spy_21d", "return_21d")]
        + [result["predictor_p_values"]["beat_spy_21d"]]
    )
    expected_sig = benjamini_hochberg(all_pvals_ordered, alpha=0.05)[-1]
    assert result["correlations"]["predictor"]["beat_spy_21d_fdr_significant"] == expected_sig


def test_compute_attribution_no_predictor_columns_correlations_unaffected():
    """When predictor columns are absent (the common case — see the live
    2026-07-10 attribution.json), `correlations` must stay exactly
    {"quant", "qual"} — the merge is a no-op, not a KeyError/empty-dict leak."""
    df = _make_df(n=200)
    result = compute_attribution(df)

    assert set(result["correlations"].keys()) == {"quant", "qual"}
    assert result["predictor_correlation"] == {}


def test_compute_attribution_reports_n_fdr_tests():
    """config#2305: n_fdr_tests persists the size of the shared BH-FDR
    correction pool so a downstream consumer (the fdr_surface_health tile)
    can calibrate its significant-count band to the ACTUAL surface instead
    of a stale absolute constant calibrated for a since-shrunk surface."""
    df = _make_df(n=200)
    result = compute_attribution(df)
    # 2 sub-scores x 2 canonical horizons, no predictor columns here.
    assert result["n_fdr_tests"] == 4

    df_with_pred = _make_df(n=200, with_predictor=True, with_correct=True)
    result_with_pred = compute_attribution(df_with_pred)
    # +1 for the predictor's own beat_spy_21d test.
    assert result_with_pred["n_fdr_tests"] == 5


def test_compute_attribution_resolves_sub_scores_from_dict_column():
    n = 150
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "symbol": [f"T{i % 20}" for i in range(n)],
        "sub_scores": [{"quant": float(rng.uniform(40, 60)), "qual": float(rng.uniform(40, 60))} for _ in range(n)],
        "beat_spy_21d": rng.integers(0, 2, n),
        "return_21d": rng.normal(0, 0.05, n),
    })

    result = compute_attribution(df)

    assert result["status"] == "ok"
    assert set(result["correlations"].keys()) == {"quant", "qual"}


def test_compute_attribution_fdr_flag_present_on_correlations():
    df = _make_df(n=200)
    result = compute_attribution(df)

    assert result["status"] == "ok"
    for label in SUB_SCORES:
        for target in ["beat_spy_21d", "return_21d"]:
            key = f"{target}_fdr_significant"
            assert key in result["correlations"][label]
            assert isinstance(result["correlations"][label][key], (bool, np.bool_))


def test_compute_attribution_predictor_hit_rate_falls_back_to_correct_5d():
    """Pre-2026-05-09 schema only has correct_5d — verify fallback."""
    df = _make_df(n=200, with_predictor=True)
    df["correct_5d"] = (df["beat_spy_21d"] == 1).astype(int)
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


# ─────────────────────────────────────────────────────────────────────────
# Multivariate attribution (config#920)
# ─────────────────────────────────────────────────────────────────────────


def _make_mv_df(
    n: int = 400,
    quant_beta: float = 0.8,
    qual_beta: float = 0.2,
    regime_beta: float = 0.0,
    with_regime: bool = True,
    seed: int = 13,
) -> pd.DataFrame:
    """score_performance-shaped frame with a *known* linear data-generating
    process so the recovered standardized coefficients are checkable.

    return_21d = quant_beta*z(quant) + qual_beta*z(qual)
                 + regime_beta*1[bear] + small noise
    """
    rng = np.random.default_rng(seed)
    quant = rng.normal(50, 10, n)
    qual = rng.normal(50, 10, n)
    zq = (quant - quant.mean()) / quant.std()
    zl = (qual - qual.mean()) / qual.std()

    regimes = rng.choice(["bull", "neutral", "bear"], size=n)
    bear = (regimes == "bear").astype(float)

    noise = rng.normal(0, 0.05, n)
    latent = quant_beta * zq + qual_beta * zl + regime_beta * bear + noise

    df = pd.DataFrame({
        "symbol": [f"T{i % 30}" for i in range(n)],
        "score_date": pd.date_range("2026-01-01", periods=n, freq="h"),
        "quant_score": quant,
        "qual_score": qual,
        "return_21d": latent,
        "beat_spy_21d": (latent > latent.mean()).astype(int),
    })
    if with_regime:
        df["market_regime"] = regimes
    return df


def test_multivariate_recovers_known_coefficients():
    """Synthetic known-beta DGP → recovered standardized coefficients match,
    and quant (larger beta) outranks qual."""
    df = _make_mv_df(n=600, quant_beta=0.8, qual_beta=0.2)
    result = compute_attribution(df)

    assert result["status"] == "ok"
    mv = result["multivariate"]
    assert mv["status"] == "ok"
    fit = mv["targets"]["return_21d"]
    assert fit["method"] == "multivariate_ols"

    coefs = fit["coefficients"]
    # Standardized coefficients are the generating betas rescaled by
    # std(x_i)/std(y); with near-noiseless data the *ratio* of the two
    # coefficients recovers the ratio of the generating betas (0.8/0.2 = 4).
    assert coefs["quant"] / coefs["qual"] == pytest.approx(4.0, rel=0.2)
    # Quant is the stronger joint driver.
    assert abs(coefs["quant"]) > abs(coefs["qual"])
    assert fit["r_squared"] is not None and fit["r_squared"] > 0.8


def test_multivariate_includes_regime_dummies():
    """market_regime is one-hot encoded (drop-first) into the design."""
    df = _make_mv_df(n=500, with_regime=True)
    mv = compute_attribution(df)["multivariate"]

    assert mv["status"] == "ok"
    assert set(mv["regime_levels"]) == {"bull", "neutral", "bear"}
    # drop-first → two of the three levels appear as features.
    regime_feats = [f for f in mv["feature_labels"] if f.startswith("regime=")]
    assert len(regime_feats) == 2
    # A target fit jointly carries a regime coefficient.
    fit = mv["targets"]["return_21d"]
    assert any(k.startswith("regime=") for k in fit["coefficients"])


def test_multivariate_recovers_regime_effect():
    """A real bear-regime effect in the DGP is recovered as a non-trivial
    standardized regime coefficient. Levels sort alphabetically and
    drop-first drops 'bear', so the bear effect surfaces as a negative
    coefficient on the retained bull/neutral dummies (relative to bear)."""
    df = _make_mv_df(n=600, quant_beta=0.5, qual_beta=0.2, regime_beta=0.6)
    fit = compute_attribution(df)["multivariate"]["targets"]["return_21d"]
    regime_coefs = [v for k, v in fit["coefficients"].items() if k.startswith("regime=")]
    assert regime_coefs  # regime dummies present
    # At least one retained regime dummy carries a non-trivial effect.
    assert max(abs(c) for c in regime_coefs) > 0.05


def test_multivariate_falls_back_on_collinear_inputs():
    """Perfectly collinear sub-scores → rank-deficient design → safe
    fallback to univariate Pearson, not a crash."""
    df = _make_mv_df(n=500, with_regime=False)
    df["qual_score"] = df["quant_score"] * 2.0 + 1.0  # perfect collinearity

    result = compute_attribution(df)
    assert result["status"] == "ok"  # no crash

    mv = result["multivariate"]
    target = mv["targets"]["return_21d"]
    assert target["method"] == "univariate_fallback"
    assert "return_21d" in (mv.get("targets_fell_back_to_univariate") or [])
    # Fallback coefficients come from the retained univariate correlations.
    assert set(target["coefficients"].keys()) == {"quant", "qual"}


def test_multivariate_falls_back_on_low_n():
    """Below the rows-per-regressor floor → univariate fallback per target."""
    df = _make_mv_df(n=120, with_regime=True)  # 120 rows, several regressors
    # Force a regressor-heavy / low-N regime by keeping only ~ the floor.
    small = df.head(35)
    mv = _compute_multivariate_attribution(
        small,
        {"quant": "quant_score", "qual": "qual_score"},
        fallback_correlations={
            "quant": {"return_21d": 0.3},
            "qual": {"return_21d": 0.1},
        },
    )
    # With regime dummies the design has 4 regressors; 35 rows < 10*4 floor.
    fit = mv["targets"]["return_21d"]
    assert fit["method"] == "univariate_fallback"


def test_multivariate_works_without_regime_column():
    """No market_regime column → regression still runs on sub-scores alone."""
    df = _make_mv_df(n=500, with_regime=False)
    mv = compute_attribution(df)["multivariate"]
    assert mv["status"] == "ok"
    assert mv["regime_levels"] == []
    fit = mv["targets"]["return_21d"]
    assert fit["method"] == "multivariate_ols"
    assert set(fit["coefficients"].keys()) == {"quant", "qual"}


def test_legacy_univariate_schema_preserved():
    """The pre-existing univariate output (correlations / rankings / FDR)
    is unchanged so existing consumers keep working."""
    df = _make_df(n=200)
    result = compute_attribution(df)

    # Legacy keys still present and well-formed.
    assert set(result["correlations"].keys()) == {"quant", "qual"}
    assert result["ranking_21d"]
    assert "p_values" in result
    # New block added without removing the old contract.
    assert "multivariate" in result
