"""
veto_value.py — Net veto value in dollars.

For each veto decision, computes:
  - Correct vetoes: loss avoided = intended_position_dollars * |negative_return|
  - Incorrect vetoes: alpha foregone = intended_position_dollars * positive_return
  - Net veto value = losses_avoided - alpha_foregone

Requires: predictor_outcomes (predictions), score_performance or universe_returns
(actual returns), and executor context (position sizing).

Data sources:
  - predictor_outcomes in research.db (predictions with canonical alpha —
    `actual_log_alpha` for new rows, legacy `actual_5d_return` for old rows;
    pipeline_common.ALPHA_COALESCE_SQL normalizes both to decimal scale)
  - executor_shadow_book in trades.db (blocked entries with intended_dollars)
  - universe_returns in research.db (forward returns for all stocks)
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from pipeline_common import (
    ALPHA_COALESCE_SQL,
    CURRENT_HORIZON_FILTER_SQL,
    HORIZON_COALESCE_SQL,
    OUTCOMES_RESOLVED_SQL,
)

logger = logging.getLogger(__name__)

_DEFAULT_POSITION_SIZE = 50_000.0  # fallback if intended_dollars not available


def compute_veto_value(
    research_db_path: str,
    trades_db_path: str | None = None,
    default_position_size: float = _DEFAULT_POSITION_SIZE,
) -> dict:
    """
    Compute the dollar value of the predictor veto system.

    For each stock predicted DOWN (veto candidate), computes:
    - Actual forward return (from predictor_outcomes or universe_returns)
    - Position size (from shadow_book intended_dollars or default)
    - Dollar impact: position_size * actual_return

    Returns dict with:
        status: "ok" | "insufficient_data" | "error"
        n_vetoes: total veto decisions evaluated
        n_correct: vetoes where stock underperformed (loss avoided)
        n_incorrect: vetoes where stock outperformed (alpha foregone)
        total_losses_avoided: sum of dollar losses avoided by correct vetoes
        total_alpha_foregone: sum of dollar alpha foregone by incorrect vetoes
        net_veto_value: losses_avoided - alpha_foregone
        avg_loss_avoided: per correct veto
        avg_alpha_foregone: per incorrect veto
        by_confidence: breakdown by confidence bucket
    """
    if not Path(research_db_path).exists():
        return {"status": "error", "error": f"research.db not found at {research_db_path}"}

    try:
        conn = sqlite3.connect(research_db_path)

        po = pd.read_sql_query(
            "SELECT symbol, prediction_date, predicted_direction, "
            "prediction_confidence, "
            f"{ALPHA_COALESCE_SQL} AS actual_alpha, "
            f"{HORIZON_COALESCE_SQL} AS horizon_days, "
            "p_down "
            "FROM predictor_outcomes "
            f"WHERE predicted_direction = 'DOWN' "
            f"  AND {OUTCOMES_RESOLVED_SQL} "
            f"  AND {CURRENT_HORIZON_FILTER_SQL}",
            conn,
        )
        conn.close()
    except Exception as e:
        return {"status": "error", "error": str(e)}

    if po.empty or len(po) < 3:
        return {
            "status": "insufficient_data",
            "error": f"need >= 3 resolved DOWN predictions, have {len(po)}",
        }

    # Try to get position sizes from shadow book
    shadow_sizes = {}
    if trades_db_path and Path(trades_db_path).exists():
        try:
            tconn = sqlite3.connect(trades_db_path)
            shadow = pd.read_sql_query(
                "SELECT ticker, date, intended_dollars "
                "FROM executor_shadow_book "
                "WHERE block_reason LIKE '%veto%' OR predicted_direction = 'DOWN'",
                tconn,
            )
            tconn.close()
            for _, row in shadow.iterrows():
                if row["intended_dollars"]:
                    shadow_sizes[(row["ticker"], row["date"])] = float(row["intended_dollars"])
        except Exception:
            pass

    # `actual_alpha` is decimal alpha (decimal log-units for new rows post
    # canonical-21d cutover; arithmetic-decimal for old rows where the
    # SQL COALESCE divided actual_5d_return / 100). Dollar impact is the
    # naive linear product — accurate at small magnitudes where log(1+r) ≈ r.
    po["position_dollars"] = po.apply(
        lambda r: shadow_sizes.get(
            (r["symbol"], r["prediction_date"]),
            default_position_size,
        ),
        axis=1,
    )

    po["dollar_impact"] = po["position_dollars"] * po["actual_alpha"]

    correct = po[po["actual_alpha"] < 0]  # stock underperformed → veto was right
    incorrect = po[po["actual_alpha"] >= 0]  # stock outperformed → veto missed alpha

    total_losses_avoided = float(correct["dollar_impact"].abs().sum()) if not correct.empty else 0
    total_alpha_foregone = float(incorrect["dollar_impact"].sum()) if not incorrect.empty else 0
    net_value = total_losses_avoided - total_alpha_foregone

    # Confidence breakdown
    po["conf_bucket"] = pd.cut(
        po["prediction_confidence"].fillna(0.5),
        bins=[0, 0.55, 0.65, 0.75, 1.0],
        labels=["50-55%", "55-65%", "65-75%", "75%+"],
        include_lowest=True,
    )
    by_confidence = []
    for bucket in po["conf_bucket"].cat.categories:
        grp = po[po["conf_bucket"] == bucket]
        if grp.empty:
            continue
        grp_correct = grp[grp["actual_alpha"] < 0]
        grp_incorrect = grp[grp["actual_alpha"] >= 0]
        by_confidence.append({
            "confidence_range": str(bucket),
            "n_vetoes": len(grp),
            "precision": round(len(grp_correct) / len(grp), 4) if len(grp) > 0 else None,
            "losses_avoided": round(float(grp_correct["dollar_impact"].abs().sum()), 2) if not grp_correct.empty else 0,
            "alpha_foregone": round(float(grp_incorrect["dollar_impact"].sum()), 2) if not grp_incorrect.empty else 0,
            "net_value": round(
                float(grp_correct["dollar_impact"].abs().sum()) - float(grp_incorrect["dollar_impact"].sum()),
                2,
            ) if not grp.empty else 0,
        })

    # Single-horizon snapshot post 2026-05-10 filter: rolling-analytics reads
    # scope to ACTIVE_HORIZON_DAYS via CURRENT_HORIZON_FILTER_SQL so the
    # transition window doesn't mix 5d-arith and 21d-log distributions.
    # `horizons_seen` is retained for the forensic trail; expect a single
    # value during normal operation, multiple only if a horizon migration
    # is in progress and the filter constant was bumped mid-cycle.
    horizons_seen = sorted(po["horizon_days"].dropna().unique().tolist())

    return {
        "status": "ok",
        "n_vetoes": len(po),
        "n_correct": len(correct),
        "n_incorrect": len(incorrect),
        "precision": round(len(correct) / len(po), 4) if len(po) > 0 else None,
        "total_losses_avoided": round(total_losses_avoided, 2),
        "total_alpha_foregone": round(total_alpha_foregone, 2),
        "net_veto_value": round(net_value, 2),
        "avg_loss_avoided": round(total_losses_avoided / len(correct), 2) if len(correct) > 0 else 0,
        "avg_alpha_foregone": round(total_alpha_foregone / len(incorrect), 2) if len(incorrect) > 0 else 0,
        # avg_veto_alpha is now decimal (×100 to read as percentage)
        "avg_veto_alpha_pct": round(float(po["actual_alpha"].mean()) * 100.0, 2),
        "horizons_days": [int(h) for h in horizons_seen],
        "by_confidence": by_confidence,
    }
