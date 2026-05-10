"""
predictor_confusion.py — Confusion matrix for predictor directional predictions.

Computes a 3x3 confusion matrix (UP/FLAT/DOWN predicted vs actual) from
predictor_outcomes. Reveals whether the model confuses flat with directional
or reverses direction.

Actual direction is derived from canonical alpha (decimal scale —
`actual_log_alpha` for new rows post Track A cutover, or
`actual_5d_return / 100` for legacy rows; pipeline_common.ALPHA_COALESCE_SQL
normalizes both):
  UP:   canonical_actual > up_threshold (default 0.005, ≈ 0.5% decimal)
  DOWN: canonical_actual < -up_threshold
  FLAT: otherwise

Caller can override `up_threshold` to match a specific config (e.g.
predictor.yaml's `cfg.UP_THRESHOLD`) so the threshold is the source of
truth, not a hardcoded module constant.
"""

import logging
import sqlite3

import pandas as pd

from pipeline_common import (
    ALPHA_COALESCE_SQL,
    HORIZON_COALESCE_SQL,
    OUTCOMES_RESOLVED_SQL,
)

logger = logging.getLogger(__name__)

# Default magnitude classifying UP/DOWN. Decimal scale (0.005 = 0.5%).
# Caller can override via the `up_threshold` parameter on
# `compute_confusion_matrix` so the threshold lives in caller config
# (predictor.yaml's UP_THRESHOLD), not as a hardcoded module constant.
_DEFAULT_UP_THRESHOLD = 0.005

DIRECTIONS = ["UP", "FLAT", "DOWN"]


def _actual_direction(ret: float, up_threshold: float = _DEFAULT_UP_THRESHOLD) -> str:
    """Classify continuous decimal return as UP/FLAT/DOWN with a symmetric ±band."""
    if ret > up_threshold:
        return "UP"
    elif ret < -up_threshold:
        return "DOWN"
    return "FLAT"


def compute_confusion_matrix(
    db_path: str,
    min_samples: int = 30,
    up_threshold: float = _DEFAULT_UP_THRESHOLD,
) -> dict:
    """Compute a 3x3 confusion matrix from predictor_outcomes.

    Args:
        up_threshold: decimal magnitude classifying UP/DOWN. Defaults to
            0.005 (0.5%). Production callers should pass `cfg.UP_THRESHOLD`
            from predictor.yaml so the threshold is config-driven.

    Returns:
        status: "ok" | "insufficient_data" | "error"
        n: total resolved predictions
        matrix: {predicted: {actual: count}} e.g. {"UP": {"UP": 40, "FLAT": 15, "DOWN": 5}}
        accuracy: overall directional accuracy
        per_class: {direction: {precision, recall, f1, n_predicted, n_actual}}
        up_threshold: the threshold used (echoed back for forensic trail)
        horizons_days: list of horizons present in the underlying rows
    """
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            "SELECT predicted_direction, "
            f"{ALPHA_COALESCE_SQL} AS canonical_actual, "
            f"{HORIZON_COALESCE_SQL} AS horizon_days "
            "FROM predictor_outcomes "
            f"WHERE predicted_direction IS NOT NULL AND {OUTCOMES_RESOLVED_SQL}",
            conn,
        )
        conn.close()
    except Exception as e:
        return {"status": "error", "error": str(e)}

    if len(df) < min_samples:
        return {
            "status": "insufficient_data",
            "n": len(df),
            "min_required": min_samples,
        }

    df["actual_direction"] = df["canonical_actual"].apply(
        lambda r: _actual_direction(r, up_threshold=up_threshold)
    )

    # Build confusion matrix
    matrix = {}
    for pred in DIRECTIONS:
        matrix[pred] = {}
        for actual in DIRECTIONS:
            matrix[pred][actual] = int(
                ((df["predicted_direction"] == pred) & (df["actual_direction"] == actual)).sum()
            )

    n = len(df)
    correct = sum(matrix[d][d] for d in DIRECTIONS)
    accuracy = correct / n if n > 0 else None

    # Per-class precision, recall, F1
    per_class = {}
    for d in DIRECTIONS:
        n_predicted = sum(matrix[d][a] for a in DIRECTIONS)
        n_actual = sum(matrix[p][d] for p in DIRECTIONS)
        tp = matrix[d][d]

        precision = tp / n_predicted if n_predicted > 0 else None
        recall = tp / n_actual if n_actual > 0 else None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = None

        per_class[d] = {
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "n_predicted": n_predicted,
            "n_actual": n_actual,
            "tp": tp,
        }

    horizons_seen = sorted(df["horizon_days"].dropna().unique().tolist())

    return {
        "status": "ok",
        "n": n,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "matrix": matrix,
        "per_class": per_class,
        "up_threshold": up_threshold,
        "horizons_days": [int(h) for h in horizons_seen],
    }
