"""
attribution.py — which sub-score (quant / qual) drives beat-SPY?

Computes correlation between each sub-score and beat_spy_10d/30d.
This is the primary mechanism for improving research pipeline scoring weights.

Horizon separation: Research uses quant + qual only (6–12 month fundamental).
Technical analysis is handled by Predictor (GBM) and Executor (ATR/time exits).

Data availability: noisy with <200 rows; meaningful at Week 8+ (~500 rows).
"""

import logging

import pandas as pd
from scipy.stats import pearsonr

from analysis.stats_utils import benjamini_hochberg

logger = logging.getLogger(__name__)

SUB_SCORES = ["quant", "qual"]
PREDICTOR_COLS = ["p_up", "p_down", "prediction_confidence", "predicted_direction"]


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
                "quant": {"beat_spy_10d": 0.12, "beat_spy_30d": 0.09, ...},
                "qual": {...},
            },
            "ranking_10d": ["qual", "quant"],  # descending by correlation
            "ranking_30d": [...],
            "note": "..."
        }
    """
    populated = df[df["beat_spy_10d"].notna()].copy()

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
        for target in ["beat_spy_10d", "beat_spy_30d", "return_10d", "return_30d"]:
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

    ranking_10d = sorted(
        correlations.keys(),
        key=lambda k: correlations[k].get("beat_spy_10d") or 0,
        reverse=True,
    )
    ranking_30d = sorted(
        correlations.keys(),
        key=lambda k: correlations[k].get("beat_spy_30d") or 0,
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
        for outcome_col in ["beat_spy_10d", "beat_spy_30d"]:
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

    # Tag each correlation with FDR significance
    for label in correlations:
        for target in correlations[label]:
            key = (label, target)
            correlations[label][f"{target}_fdr_significant"] = fdr_map.get(key, False)

    return {
        "status": "ok",
        "rows_analyzed": len(populated),
        "correlations": correlations,
        "p_values": p_values,
        "ranking_10d": ranking_10d,
        "ranking_30d": ranking_30d,
        "predictor_correlation": predictor_corr,
        "predictor_p_values": predictor_pvals,
        "predictor_hit_rate": predictor_hit_rate,
        "predictor_hit_rate_ci_95": predictor_hit_rate_ci,
        "fdr_non_significant": fdr_non_significant if fdr_non_significant else None,
        "note": (
            "p-values adjusted for multiple comparisons (Benjamini-Hochberg, α=0.05). "
            "Automated weight optimization activates at Month 6+ (500+ rows)."
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
