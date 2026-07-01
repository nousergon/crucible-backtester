"""
score_analysis.py — accuracy vs. score threshold analysis.

Answers: "What is the optimal min_score cutoff?"

For each candidate threshold, compute the accuracy at that threshold and above.
Shows the tradeoff between signal count (universe size) and signal accuracy.

Data availability: requires Week 4+ (200+ populated score_performance rows).
"""

import logging

import pandas as pd

from analysis.signal_quality import _compute_slice_metrics, MIN_SAMPLES

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = [60, 65, 70, 72, 75, 78, 80, 85, 90]


def accuracy_by_threshold(
    df: pd.DataFrame,
    thresholds: list[int] = DEFAULT_THRESHOLDS,
    min_samples: int = MIN_SAMPLES,
) -> list[dict]:
    """
    For each threshold T in thresholds, compute accuracy for signals with score >= T.

    Returns list of dicts:
        [{"threshold": 70, "accuracy_21d": 0.58, "n_21d": 120, ...}, ...]

    Use this to find the score cutoff that maximises accuracy while maintaining
    a meaningful sample size.
    """
    populated_5d = df[df["beat_spy_5d"].notna()] if "beat_spy_5d" in df.columns else pd.DataFrame()
    # config#1456: canonical 21d horizon (10d/30d outcomes retired).
    populated_21d = df[df["beat_spy_21d"].notna()] if "beat_spy_21d" in df.columns else pd.DataFrame()

    results = []
    for t in sorted(thresholds):
        slice_5d = populated_5d[populated_5d["score"] >= t] if not populated_5d.empty else pd.DataFrame()
        slice_21d = populated_21d[populated_21d["score"] >= t]

        if len(slice_21d) < min_samples:
            logger.debug("Skipping threshold %d — only %d samples", t, len(slice_21d))
            continue

        metrics = _compute_slice_metrics(slice_5d, slice_21d)
        results.append({"threshold": t, **metrics})

    if not results:
        logger.warning(
            "No thresholds have %d+ samples. Score analysis deferred until Week 4.",
            min_samples,
        )

    return results


def optimal_threshold(
    df: pd.DataFrame,
    thresholds: list[int] = DEFAULT_THRESHOLDS,
    min_samples: int = MIN_SAMPLES,
    target: str = "accuracy_21d",
    min_n: int = 20,
) -> dict | None:
    """
    Return the threshold that maximises `target` metric while having at least min_n samples.

    Returns None if insufficient data.
    """
    rows = accuracy_by_threshold(df, thresholds, min_samples=1)
    eligible = [r for r in rows if r.get("n_21d", 0) >= min_n and r.get(target) is not None]

    if not eligible:
        return None

    return max(eligible, key=lambda r: r[target])
