"""
shadow_book.py — Risk guard shadow book analysis.

Compares forward returns of blocked entries vs. traded entries to evaluate
whether the risk guard is too conservative, appropriately calibrated, or
too loose.

Data sources:
  - executor_shadow_book in trades.db (blocked entries)
  - trades in trades.db (executed entries)
  - universe_returns in research.db (forward returns for blocked stocks)
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def compute_shadow_book_analysis(
    trades_db_path: str,
    research_db_path: str | None = None,
    min_blocks: int = 3,
) -> dict:
    """
    Compare blocked entries vs. traded entries.

    Uses universe_returns to get forward returns for blocked stocks (since they
    don't have realized PnL). Falls back to simple count/reason analysis if
    research.db isn't available.

    Returns dict with:
        status: "ok" | "insufficient_data" | "error"
        n_blocked: total blocked entries
        n_traded: total traded entries
        blocked_avg_return: avg 5d forward return of blocked stocks
        traded_avg_return: avg 5d forward return of traded stocks
        guard_lift: traded_avg - blocked_avg (positive = guard is helping)
        by_reason: breakdown by block_reason
        assessment: "too_tight" | "appropriate" | "too_loose"
    """
    if not Path(trades_db_path).exists():
        return {"status": "error", "error": f"trades.db not found at {trades_db_path}"}

    # Narrow scope of broad-except per backtester-audit-260415 Phase 1.2:
    # shadow_book is a required input to the replay parity test (Phase 1.1),
    # so silent-fail on schema corruption or permission errors would mask
    # parity-test regressions. Only "table missing" is legitimately recoverable
    # (fresh trades.db on first boot, pre-shadow-book-schema); everything else
    # propagates to surface as a loud pipeline failure.
    conn = sqlite3.connect(trades_db_path)
    try:
        try:
            shadow = pd.read_sql_query(
                "SELECT ticker, date, block_reason, research_score, "
                "prediction_confidence, predicted_direction, "
                "intended_position_pct, intended_dollars, "
                "current_price, market_regime "
                "FROM executor_shadow_book",
                conn,
            )
        except pd.errors.DatabaseError as exc:
            msg = str(exc).lower()
            if "no such table" in msg or "no such column" in msg:
                return {
                    "status": "insufficient_data",
                    "error": f"shadow_book schema not present: {exc}",
                }
            raise  # schema corruption / disk error / unexpected condition

        try:
            trades = pd.read_sql_query(
                "SELECT ticker, date, fill_price, "
                "realized_return_pct, realized_alpha_pct, "
                "trigger_type, days_held "
                "FROM trades WHERE action = 'ENTER'",
                conn,
            )
        except pd.errors.DatabaseError as exc:
            msg = str(exc).lower()
            if "no such table" in msg or "no such column" in msg:
                return {
                    "status": "insufficient_data",
                    "error": f"trades schema not present: {exc}",
                }
            raise
    finally:
        conn.close()

    if shadow.empty:
        return {"status": "insufficient_data", "error": "no blocked entries in shadow book"}

    if len(shadow) < min_blocks:
        return {
            "status": "insufficient_data",
            "error": f"need >= {min_blocks} blocked entries, have {len(shadow)}",
        }

    result: dict = {
        "status": "ok",
        "n_blocked": len(shadow),
        "n_traded": len(trades),
    }

    # Join with universe_returns if available to get forward returns for blocked stocks
    blocked_returns = None
    traded_returns = None
    if research_db_path and Path(research_db_path).exists():
        try:
            rconn = sqlite3.connect(research_db_path)
            ur = pd.read_sql_query(
                "SELECT ticker, eval_date, return_5d, return_10d, "
                "spy_return_5d, beat_spy_5d "
                "FROM universe_returns WHERE return_5d IS NOT NULL",
                rconn,
            )
            rconn.close()

            if not ur.empty:
                # Blocked stock returns
                blocked_merged = shadow.merge(
                    ur,
                    left_on=["ticker", "date"],
                    right_on=["ticker", "eval_date"],
                    how="inner",
                )
                if not blocked_merged.empty:
                    blocked_returns = blocked_merged

                # Traded stock returns (from universe, not realized PnL)
                traded_merged = trades.merge(
                    ur,
                    left_on=["ticker", "date"],
                    right_on=["ticker", "eval_date"],
                    how="inner",
                )
                if not traded_merged.empty:
                    traded_returns = traded_merged
        except (sqlite3.Error, pd.errors.DatabaseError, pd.errors.MergeError, KeyError, ValueError) as e:
            # Fail-soft: forward-return enrichment is optional. Narrowed to the
            # real failure surface here — sqlite/read_sql errors (missing
            # universe_returns table or unreadable research.db) and the
            # merge/key/value errors a schema mismatch in the joined columns
            # would raise. The analysis proceeds without blocked/traded
            # forward returns and reports "insufficient_return_data".
            logger.debug("Could not join universe_returns: %s", e)

    if blocked_returns is not None and not blocked_returns.empty:
        blocked_avg = round(float(blocked_returns["return_5d"].mean()), 4)
        blocked_beat_spy = round(float(blocked_returns["beat_spy_5d"].mean()), 4) if "beat_spy_5d" in blocked_returns else None
        result["blocked_avg_return_5d"] = blocked_avg
        result["blocked_beat_spy_pct"] = blocked_beat_spy
        result["blocked_with_returns"] = len(blocked_returns)
    else:
        blocked_avg = None

    if traded_returns is not None and not traded_returns.empty:
        traded_avg = round(float(traded_returns["return_5d"].mean()), 4)
        traded_beat_spy = round(float(traded_returns["beat_spy_5d"].mean()), 4) if "beat_spy_5d" in traded_returns else None
        result["traded_avg_return_5d"] = traded_avg
        result["traded_beat_spy_pct"] = traded_beat_spy
        result["traded_with_returns"] = len(traded_returns)
    elif not trades.empty and trades["realized_alpha_pct"].notna().any():
        traded_avg = round(float(trades["realized_alpha_pct"].dropna().mean()), 4)
        result["traded_avg_alpha"] = traded_avg
    else:
        traded_avg = None

    if blocked_avg is not None and traded_avg is not None:
        guard_lift = round(traded_avg - blocked_avg, 4)
        result["guard_lift"] = guard_lift

        if guard_lift > 0.5:
            result["assessment"] = "appropriate"
        elif guard_lift < -0.5:
            result["assessment"] = "too_tight"
        else:
            result["assessment"] = "neutral"
    else:
        result["assessment"] = "insufficient_return_data"

    # Classification: selected=blocked, positive=would have lost (didn't beat SPY)
    # TP = blocked AND didn't beat SPY (correct block)
    # FP = blocked AND beat SPY (incorrectly blocked a winner)
    # FN = traded AND didn't beat SPY (should have been blocked)
    # TN = traded AND beat SPY (correctly allowed)
    if (blocked_returns is not None and not blocked_returns.empty
            and traded_returns is not None and not traded_returns.empty
            and "beat_spy_5d" in blocked_returns.columns
            and "beat_spy_5d" in traded_returns.columns):
        from analysis.classification_metrics import compute_binary_metrics
        b = blocked_returns[blocked_returns["beat_spy_5d"].notna()]
        t = traded_returns[traded_returns["beat_spy_5d"].notna()]
        if not b.empty and not t.empty:
            tp = int((b["beat_spy_5d"] == 0).sum())
            fp = int((b["beat_spy_5d"] == 1).sum())
            fn = int((t["beat_spy_5d"] == 0).sum())
            tn = int((t["beat_spy_5d"] == 1).sum())
            result["classification"] = compute_binary_metrics(tp, fp, fn, tn)

    # Breakdown by block reason
    by_reason = []
    for reason in sorted(shadow["block_reason"].unique()):
        grp = shadow[shadow["block_reason"] == reason]
        reason_data = {
            "block_reason": reason,
            "count": len(grp),
            "pct_of_blocks": round(len(grp) / len(shadow), 4),
            "avg_score": round(float(grp["research_score"].dropna().mean()), 1)
            if grp["research_score"].notna().any() else None,
        }
        if blocked_returns is not None:
            reason_merged = blocked_returns[blocked_returns["block_reason"] == reason]
            if not reason_merged.empty:
                reason_data["avg_return_5d"] = round(float(reason_merged["return_5d"].mean()), 4)
                reason_data["n_with_returns"] = len(reason_merged)
        by_reason.append(reason_data)

    result["by_reason"] = by_reason
    return result
