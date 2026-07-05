"""
attribution.py — which inputs (quant / qual sub-scores + market regime) drive beat-SPY?

Primary attribution is now **multivariate**: the target (beat_spy / forward
return) is regressed jointly on the full input set (quant_score, qual_score,
and the one-hot–encoded market_regime) so each input's contribution is
estimated *holding the others fixed* — standardized partial coefficients
rather than independent pairwise Pearson correlations. (config#920.)

The legacy univariate Pearson correlations are retained alongside as
contextual ranking and as the safe fallback when the multivariate fit is
not trustworthy (rank-deficient / collinear design, or too few samples).

This is the primary mechanism for improving research pipeline scoring weights.

Horizon separation: Research uses quant + qual only (6–12 month fundamental).
Technical analysis is handled by Predictor (GBM) and Executor (ATR/time exits).
"Macro" in this layer is the categorical Ang-Bekaert market_regime (bull /
neutral / bear) carried on score_performance — the raw macro series live in
the Predictor's feature space, not on the research evaluation rows.

Data availability: noisy with <200 rows; meaningful at Week 8+ (~500 rows).
"""

import logging

import numpy as np
import pandas as pd
from nousergon_lib.quant.horizons import DEFAULT_POLICY
from scipy.stats import pearsonr

from analysis.stats_utils import benjamini_hochberg

logger = logging.getLogger(__name__)

# Primary-horizon outcome column names resolve from the fleet HorizonPolicy
# rather than hardcoded `_Nd` literals (config#1483/#1529). The names are
# unchanged (beat_spy_21d / return_21d) — these are also the artifact dict keys
# reporter.py reads, so the identity is load-bearing — but the SOURCE df is now
# re-sourced from score_performance_outcomes upstream (signal_quality loader).
_PRIMARY_H = DEFAULT_POLICY.primary_horizon
_PRIMARY_COLS = DEFAULT_POLICY.outcome_columns(_PRIMARY_H)
_BEAT = _PRIMARY_COLS.beat_spy      # beat_spy_21d
_RET = _PRIMARY_COLS.stock_return   # return_21d

SUB_SCORES = ["quant", "qual"]
PREDICTOR_COLS = ["p_up", "p_down", "prediction_confidence", "predicted_direction"]
# config#1456: canonical 5d/21d horizons. The 10d/30d outcomes were retired in
# the canonical-alpha cutover — attribution now regresses on the 21d target only.
REGRESSION_TARGETS = [_BEAT, _RET]

# Multivariate-fit guards. A joint regression needs enough rows relative to
# the number of regressors before its partial coefficients are stable, and
# the (standardized) design matrix must be well-conditioned (no
# rank-deficiency / near-perfect collinearity among regressors).
_MV_MIN_ROWS_PER_REGRESSOR = 10  # rows-per-column floor for a stable fit
_MV_MIN_ROWS = 30                # absolute row floor for any multivariate fit
_MV_MAX_CONDITION_NUMBER = 1e3   # design-matrix condition-number ceiling


def compute_attribution(df: pd.DataFrame) -> dict:
    """
    Compute correlation between sub-scores and forward return outcomes.

    Expects score_performance rows joined with sub-score columns.
    Sub-scores are assumed to be in a 'sub_scores' JSON column or as separate
    columns named quant_score, qual_score.

    Returns:
        {
            "status": "ok" | "insufficient_data",
            "correlations": {
                # keys are the primary-horizon outcome column names
                # (HorizonPolicy.outcome_columns): beat-SPY flag + stock return
                "quant": {"<beat_spy>": 0.12, "<return>": 0.09, ...},
                "qual": {...},
            },
            "ranking_21d": ["qual", "quant"],  # descending by correlation
            "note": "..."
        }
    """
    populated = df[df[_BEAT].notna()].copy()

    if len(populated) < 100:
        return {
            "status": "insufficient_data",
            "rows_populated": len(populated),
            "note": (
                f"Attribution analysis requires at least 100 rows for FDR-robust "
                f"correlations (currently {len(populated)}). Meaningful results at "
                f"Week 8+ (~500 rows)."
            ),
        }

    # Resolve sub-score columns
    sub_score_cols = _resolve_sub_score_columns(populated)
    if not sub_score_cols:
        return {
            "status": "no_sub_score_columns",
            "note": (
                "No sub-score columns found. Expected 'sub_scores' JSON column or "
                "separate quant_score/qual_score columns."
            ),
        }

    correlations = {}
    p_values = {}
    for label, col in sub_score_cols.items():
        corr_row = {}
        pval_row = {}
        for target in [_BEAT, _RET]:
            valid = populated[[col, target]].dropna()
            if len(valid) >= 10:
                r, p = pearsonr(valid[col], valid[target])
                corr_row[target] = round(float(r), 4)
                pval_row[target] = round(float(p), 4)
            else:
                corr_row[target] = None
                pval_row[target] = None
        correlations[label] = corr_row
        p_values[label] = pval_row

    ranking_21d = sorted(
        correlations.keys(),
        key=lambda k: correlations[k].get(_BEAT) or 0,
        reverse=True,
    )

    # Predictor correlation (optional — only if predictor columns are present)
    predictor_corr = {}
    predictor_pvals = {}
    predictor_hit_rate = None
    predictor_hit_rate_ci = None
    if "p_up" in populated.columns and "p_down" in populated.columns:
        populated["_net_pred"] = (
            pd.to_numeric(populated["p_up"], errors="coerce").fillna(0)
            - pd.to_numeric(populated["p_down"], errors="coerce").fillna(0)
        )
        for outcome_col in [_BEAT]:
            if outcome_col in populated.columns:
                valid = populated[["_net_pred", outcome_col]].dropna()
                if len(valid) >= 10:
                    r, p = pearsonr(valid["_net_pred"], valid[outcome_col])
                    predictor_corr[outcome_col] = round(float(r), 4)
                    predictor_pvals[outcome_col] = round(float(p), 4)
    # Predictor hit rate: read horizon-agnostic `correct` (post 2026-05-09
    # migration) preferred, fall back to legacy `correct_5d`.
    correct_col = (
        "correct" if "correct" in populated.columns else
        ("correct_5d" if "correct_5d" in populated.columns else None)
    )
    if correct_col is not None:
        resolved = pd.to_numeric(populated[correct_col], errors="coerce").dropna()
        if len(resolved) >= 10:
            predictor_hit_rate = round(float(resolved.mean()), 4)
            from analysis.signal_quality import _wilson_ci
            predictor_hit_rate_ci = _wilson_ci(int(resolved.sum()), len(resolved))

    # Collect all p-values for FDR correction (Benjamini-Hochberg)
    all_pvals = []
    pval_keys = []  # track (label, target) for each p-value
    for label, pvals in {**p_values, "predictor": predictor_pvals}.items():
        for target, p in pvals.items():
            if p is not None:
                all_pvals.append(p)
                pval_keys.append((label, target))

    fdr_significant = benjamini_hochberg(all_pvals, alpha=0.05)

    # Build FDR significance map and flag non-significant correlations
    fdr_map = {}  # {(label, target): bool}
    fdr_non_significant = []
    for i, (label, target) in enumerate(pval_keys):
        fdr_map[(label, target)] = fdr_significant[i]
        if not fdr_significant[i]:
            p = all_pvals[i]
            fdr_non_significant.append(f"{label}.{target} (p={p:.3f})")

    # Snapshot keys before tagging — the loop body mutates the same dict,
    # which raises RuntimeError on Python 3.13+ and is undefined on older
    # versions.
    for label in correlations:
        for target in list(correlations[label].keys()):
            key = (label, target)
            correlations[label][f"{target}_fdr_significant"] = fdr_map.get(key, False)

    # Primary attribution (config#920): regress each target jointly on the
    # full input set so each input's contribution is estimated holding the
    # others fixed. Falls back to the univariate correlations above when the
    # joint fit is not trustworthy (collinear / rank-deficient / low-N).
    multivariate = _compute_multivariate_attribution(
        populated, sub_score_cols, fallback_correlations=correlations,
    )

    return {
        "status": "ok",
        "rows_analyzed": len(populated),
        "multivariate": multivariate,
        "correlations": correlations,
        "p_values": p_values,
        "ranking_21d": ranking_21d,
        "predictor_correlation": predictor_corr,
        "predictor_p_values": predictor_pvals,
        "predictor_hit_rate": predictor_hit_rate,
        "predictor_hit_rate_ci_95": predictor_hit_rate_ci,
        "fdr_non_significant": fdr_non_significant if fdr_non_significant else None,
        "note": (
            "Primary attribution is multivariate (standardized partial coefficients "
            "from a joint regression on sub-scores + market_regime); univariate "
            "Pearson correlations are retained as context and as the safe fallback. "
            "Univariate p-values adjusted for multiple comparisons "
            "(Benjamini-Hochberg, α=0.05). "
            "Automated weight optimization activates at Month 6+ (500+ rows)."
        ),
    }


def _build_design_matrix(
    df: pd.DataFrame, sub_score_cols: dict[str, str],
) -> tuple[np.ndarray | None, list[str], list[str]]:
    """Assemble the (unstandardized) multivariate design matrix.

    Columns: the numeric sub-scores (quant/qual) plus one-hot–encoded
    market_regime dummies (drop-first to avoid the dummy-variable trap /
    perfect collinearity with the intercept). Regime is the categorical
    "macro" state available on score_performance (config#920).

    Returns ``(matrix, feature_labels, regime_levels)``. ``matrix`` is None
    when no numeric regressor is available. Rows are not filtered here — the
    caller drops NaN rows per-target so each target uses its own complete
    cases.
    """
    parts: list[pd.Series] = []
    labels: list[str] = []
    for label, col in sub_score_cols.items():
        parts.append(pd.to_numeric(df[col], errors="coerce").rename(label))
        labels.append(label)

    regime_levels: list[str] = []
    if "market_regime" in df.columns:
        regime = df["market_regime"].astype("object").where(df["market_regime"].notna())
        # Only encode when the regime actually varies — a single level (or
        # all-NaN) carries no joint information and would be collinear with
        # the intercept.
        levels = sorted({str(v) for v in regime.dropna().unique()})
        if len(levels) >= 2:
            regime_levels = levels
            # drop-first: encode levels[1:] as 0/1 indicators.
            for lvl in levels[1:]:
                dummy = (regime == lvl).astype(float)
                # Rows where regime is NaN must not silently become 0 — mark
                # them NaN so the per-target dropna excludes them.
                dummy[regime.isna()] = np.nan
                parts.append(dummy.rename(f"regime={lvl}"))
                labels.append(f"regime={lvl}")

    if not parts:
        return None, [], regime_levels

    matrix = pd.concat(parts, axis=1)
    return matrix.to_numpy(dtype=float), labels, regime_levels


def _fit_one_target(
    design: np.ndarray, y: np.ndarray, feature_labels: list[str],
) -> dict | None:
    """Fit one standardized OLS of ``y`` on ``design``; return coefficients.

    Standardizes every regressor and the target to unit variance so the
    coefficients are directly comparable (standardized partial effects).
    Returns None — signalling "fall back" — when the fit is not trustworthy:
    too few rows, a zero-variance target, or a rank-deficient / ill-
    conditioned design (collinearity).
    """
    n, k = design.shape
    if n < _MV_MIN_ROWS or n < _MV_MIN_ROWS_PER_REGRESSOR * k:
        return None

    # Standardize regressors; drop columns with zero variance (constant in
    # this complete-case subset — e.g. a regime level absent for this target).
    col_std = design.std(axis=0)
    keep = col_std > 1e-10
    if not keep.any():
        return None
    X = (design[:, keep] - design[:, keep].mean(axis=0)) / col_std[keep]
    kept_labels = [lbl for lbl, k_ in zip(feature_labels, keep) if k_]

    y_std = float(y.std())
    if y_std <= 1e-10:
        return None
    y_z = (y - y.mean()) / y_std

    # Rank / conditioning guard: reject collinear or rank-deficient designs.
    # np.linalg.cond on the standardized matrix flags near-linear dependence
    # among regressors before the normal equations become unstable.
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None
    cond = float(np.linalg.cond(X))
    if not np.isfinite(cond) or cond > _MV_MAX_CONDITION_NUMBER:
        return None

    # Least-squares fit (no intercept needed — both sides are centered).
    coef, _residuals, _rank, _sv = np.linalg.lstsq(X, y_z, rcond=None)

    # R² from the centered fit.
    y_hat = X @ coef
    ss_res = float(((y_z - y_hat) ** 2).sum())
    ss_tot = float((y_z ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else None

    return {
        "n": int(n),
        "coefficients": {lbl: round(float(c), 4) for lbl, c in zip(kept_labels, coef)},
        "r_squared": round(r2, 4) if r2 is not None else None,
        "condition_number": round(cond, 1),
    }


def _compute_multivariate_attribution(
    df: pd.DataFrame,
    sub_score_cols: dict[str, str],
    fallback_correlations: dict,
) -> dict:
    """Joint multivariate attribution per target (config#920).

    For each forward-outcome target, regress it on the standardized full
    input set (sub-scores + regime dummies) and report standardized partial
    coefficients — each input's contribution holding the others fixed.

    Per-target safe fallback: when the joint fit is not trustworthy
    (rank-deficient / collinear / insufficient sample), the target's entry
    falls back to the prior univariate Pearson value with a logged note, so
    a degenerate input never crashes the evaluation run.
    """
    design, feature_labels, regime_levels = _build_design_matrix(df, sub_score_cols)
    if design is None:
        logger.info(
            "Multivariate attribution skipped: no numeric regressors available; "
            "univariate correlations retained."
        )
        return {
            "status": "fallback",
            "reason": "no_regressors",
            "feature_labels": [],
            "regime_levels": regime_levels,
            "targets": {},
            "fallback_to": "univariate_pearson",
        }

    targets: dict[str, dict] = {}
    fallbacks: list[str] = []
    fitted_any = False

    for target in REGRESSION_TARGETS:
        if target not in df.columns:
            continue
        y_series = pd.to_numeric(df[target], errors="coerce")
        mask = y_series.notna() & ~np.isnan(design).any(axis=1)
        if mask.sum() == 0:
            continue
        sub_design = design[mask.to_numpy()]
        y = y_series[mask].to_numpy(dtype=float)

        fit = _fit_one_target(sub_design, y, feature_labels)
        if fit is None:
            # Safe fallback: reuse the univariate Pearson coefficients for the
            # sub-scores, flagged so consumers know it is not the joint fit.
            uni = {
                label: fallback_correlations.get(label, {}).get(target)
                for label in sub_score_cols
            }
            targets[target] = {
                "method": "univariate_fallback",
                "coefficients": uni,
                "note": (
                    "Joint fit not trustworthy for this target "
                    "(insufficient sample, zero-variance, or collinear/"
                    "rank-deficient design); reusing univariate Pearson."
                ),
            }
            fallbacks.append(target)
            logger.info(
                "Multivariate attribution fell back to univariate for %s "
                "(degenerate/collinear/low-N design).",
                target,
            )
        else:
            fit["method"] = "multivariate_ols"
            targets[target] = fit
            fitted_any = True

    if not targets:
        return {
            "status": "fallback",
            "reason": "no_targets",
            "feature_labels": feature_labels,
            "regime_levels": regime_levels,
            "targets": {},
            "fallback_to": "univariate_pearson",
        }

    # Ranking: order inputs by |standardized coefficient| on beat_spy_21d
    # when that target was fit jointly (the headline horizon).
    ranking_21d: list[str] = []
    head = targets.get(_BEAT, {})
    if head.get("method") == "multivariate_ols":
        ranking_21d = sorted(
            head["coefficients"],
            key=lambda k: abs(head["coefficients"][k] or 0.0),
            reverse=True,
        )

    return {
        "status": "ok" if fitted_any else "fallback",
        "method": "joint_standardized_ols",
        "feature_labels": feature_labels,
        "regime_levels": regime_levels,
        "targets": targets,
        "ranking_21d": ranking_21d,
        "targets_fell_back_to_univariate": fallbacks or None,
        "note": (
            "Standardized partial coefficients from a joint OLS of each target "
            "on sub-scores + one-hot market_regime (drop-first). Coefficients "
            "are comparable across inputs (z-scored regressors and target). "
            "Per-target fallback to univariate Pearson on collinear / low-N fits."
        ),
    }


def _resolve_sub_score_columns(df: pd.DataFrame) -> dict[str, str]:
    """
    Find sub-score columns in the DataFrame.

    Checks for:
    1. Separate columns: quant_score, qual_score
    2. Falls back to flattening a 'sub_scores' JSON column if present

    Returns dict mapping label → column_name.
    """
    explicit = {}
    for name in SUB_SCORES:
        col = f"{name}_score"
        if col in df.columns:
            explicit[name] = col

    if explicit:
        return explicit

    # Try to expand a 'sub_scores' dict column if it was loaded as objects
    if "sub_scores" in df.columns:
        try:
            expanded = pd.json_normalize(df["sub_scores"])
            for name in SUB_SCORES:
                if name in expanded.columns:
                    df[f"_attr_{name}"] = expanded[name].values
                    explicit[name] = f"_attr_{name}"
            if explicit:
                return explicit
        except Exception as e:
            logger.debug("Could not expand sub_scores column: %s", e)

    return {}
