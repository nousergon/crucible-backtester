"""
post_trade.py — Unified weekly post-trade analysis.

Aggregates entry trigger effectiveness, exit rule effectiveness, holding period
distribution, and time-of-day slippage into a single report. Designed to run
weekly as part of the backtester pipeline.

Data source: trades.db (downloaded from S3 at runtime).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def compute_post_trade_analysis(trades_db_path: str, min_trades: int = 3) -> dict:
    """
    Unified post-trade analysis covering triggers, exits, holding periods,
    and time-of-day slippage.

    Returns dict with:
        status: "ok" | "insufficient_data" | "error"
        trigger_effectiveness: per-trigger alpha and slippage
        exit_effectiveness: per-exit-reason capture ratio and hold time
        holding_period: alpha by days-held bucket
        time_of_day_slippage: slippage by fill-time bucket
        summary: top-level metrics
    """
    if not Path(trades_db_path).exists():
        return {"status": "error", "error": f"trades.db not found at {trades_db_path}"}

    try:
        conn = sqlite3.connect(trades_db_path)
        entries_df = pd.read_sql_query(
            "SELECT ticker, date, fill_price, price_at_order, signal_price, "
            "trigger_type, fill_time, realized_return_pct, realized_alpha_pct, "
            "spy_return_during_hold, slippage_vs_signal, days_held "
            "FROM trades WHERE action = 'ENTER'",
            conn,
        )
        exits_df = pd.read_sql_query(
            "SELECT ticker, date, exit_reason, realized_return_pct, "
            "realized_alpha_pct, days_held "
            "FROM trades WHERE action IN ('EXIT', 'REDUCE') "
            "AND entry_trade_id IS NOT NULL",
            conn,
        )
        conn.close()
    except Exception as e:
        return {"status": "error", "error": str(e)}

    if entries_df.empty:
        return {"status": "insufficient_data", "error": "no ENTER trades found"}

    result = {"status": "ok"}

    # ── Trigger effectiveness ──────────────────────────────────────────
    result["trigger_effectiveness"] = _trigger_analysis(entries_df, min_trades)

    # ── Exit effectiveness ─────────────────────────────────────────────
    result["exit_effectiveness"] = _exit_analysis(exits_df, min_trades)

    # ── Holding period analysis ────────────────────────────────────────
    result["holding_period"] = _holding_period_analysis(entries_df)

    # ── Time-of-day slippage ───────────────────────────────────────────
    result["time_of_day_slippage"] = _time_of_day_analysis(entries_df)

    # ── Summary ────────────────────────────────────────────────────────
    result["summary"] = {
        "n_entries": len(entries_df),
        "n_exits": len(exits_df),
        "n_roundtrips": int(entries_df["realized_alpha_pct"].notna().sum()),
        "avg_alpha_pct": _safe_mean(entries_df["realized_alpha_pct"]),
        "avg_slippage_pct": _safe_mean(entries_df["slippage_vs_signal"]),
        "avg_days_held": _safe_mean(entries_df["days_held"]),
        "win_rate_vs_spy": _win_rate(entries_df["realized_alpha_pct"]),
        "best_trigger": _best_by(result["trigger_effectiveness"], "avg_alpha_pct"),
        "best_exit_rule": _best_by(result["exit_effectiveness"], "avg_alpha_pct"),
    }

    return result


def _trigger_analysis(df: pd.DataFrame, min_trades: int) -> list[dict]:
    """Per-trigger-type metrics."""
    df = df.copy()
    df["trigger_cat"] = df["trigger_type"].apply(_categorize_trigger)

    results = []
    for cat in sorted(df["trigger_cat"].unique()):
        subset = df[df["trigger_cat"] == cat]
        n = len(subset)
        if n < min_trades:
            continue
        results.append({
            "trigger": cat,
            "n_trades": n,
            "avg_alpha_pct": _safe_mean(subset["realized_alpha_pct"]),
            "avg_slippage_pct": _safe_mean(subset["slippage_vs_signal"]),
            "avg_days_held": _safe_mean(subset["days_held"]),
            "win_rate_vs_spy": _win_rate(subset["realized_alpha_pct"]),
        })
    return results


def _exit_analysis(df: pd.DataFrame, min_trades: int) -> list[dict]:
    """Per-exit-reason metrics."""
    if df.empty:
        return []

    df = df.copy()
    df["exit_cat"] = df["exit_reason"].apply(_categorize_exit)

    results = []
    for cat in sorted(df["exit_cat"].unique()):
        subset = df[df["exit_cat"] == cat]
        n = len(subset)
        if n < min_trades:
            continue
        results.append({
            "exit_rule": cat,
            "n_trades": n,
            "avg_alpha_pct": _safe_mean(subset["realized_alpha_pct"]),
            "avg_return_pct": _safe_mean(subset["realized_return_pct"]),
            "avg_days_held": _safe_mean(subset["days_held"]),
            "win_rate_vs_spy": _win_rate(subset["realized_alpha_pct"]),
        })
    return results


def _holding_period_analysis(df: pd.DataFrame) -> list[dict]:
    """Alpha by days-held bucket."""
    roundtrips = df[df["realized_alpha_pct"].notna() & df["days_held"].notna()].copy()
    if roundtrips.empty:
        return []

    # Bucket: 1-2d, 3-5d, 6-10d, 11-20d, 21+d
    bins = [0, 2, 5, 10, 20, 999]
    labels = ["1-2d", "3-5d", "6-10d", "11-20d", "21+d"]
    roundtrips["bucket"] = pd.cut(roundtrips["days_held"], bins=bins, labels=labels)

    results = []
    for label in labels:
        subset = roundtrips[roundtrips["bucket"] == label]
        if subset.empty:
            continue
        results.append({
            "bucket": label,
            "n_trades": len(subset),
            "avg_alpha_pct": _safe_mean(subset["realized_alpha_pct"]),
            "avg_return_pct": _safe_mean(subset["realized_return_pct"]),
            "win_rate_vs_spy": _win_rate(subset["realized_alpha_pct"]),
        })
    return results


def _time_of_day_analysis(df: pd.DataFrame) -> list[dict]:
    """Slippage by fill-time bucket."""
    timed = df[df["fill_time"].notna()].copy()
    if timed.empty:
        return []

    def _time_bucket(fill_time: str) -> str:
        try:
            # fill_time is ISO format or HH:MM:SS
            parts = fill_time.split("T")[-1] if "T" in fill_time else fill_time
            hour = int(parts.split(":")[0])
            if hour < 10:
                return "early (pre-10)"
            elif hour < 12:
                return "morning (10-12)"
            elif hour < 14:
                return "midday (12-14)"
            else:
                return "afternoon (14+)"
        except (ValueError, IndexError):
            return "unknown"

    timed["time_bucket"] = timed["fill_time"].apply(_time_bucket)

    results = []
    for bucket in ["early (pre-10)", "morning (10-12)", "midday (12-14)", "afternoon (14+)"]:
        subset = timed[timed["time_bucket"] == bucket]
        if subset.empty:
            continue
        results.append({
            "time_bucket": bucket,
            "n_trades": len(subset),
            "avg_slippage_pct": _safe_mean(subset["slippage_vs_signal"]),
            "avg_alpha_pct": _safe_mean(subset["realized_alpha_pct"]),
        })
    return results


# ── Helpers ──────────────────────────────────────────────────────────────────


def _categorize_trigger(trigger_type: str | None) -> str:
    if not trigger_type:
        return "unknown"
    t = trigger_type.lower()
    # Both "time_expiry*" and bare "expiry*" trigger names canonicalize to
    # "time_expiry"; without an explicit map the old `.replace("expiry",
    # "time_expiry")` produced "time_time_expiry" for the first form.
    for keyword, canonical in (
        ("pullback", "pullback"),
        ("vwap", "vwap"),
        ("support", "support"),
        ("time_expiry", "time_expiry"),
        ("expiry", "time_expiry"),
    ):
        if keyword in t:
            return canonical
    return "other"


def _categorize_exit(exit_reason: str | None) -> str:
    if not exit_reason:
        return "unknown"
    r = exit_reason.lower()
    for keyword in ("trailing_stop", "profit_take", "time_decay", "collapse",
                    "signal_exit", "research_exit", "momentum"):
        if keyword in r:
            return keyword
    return "other"


def _safe_mean(series: pd.Series) -> float | None:
    valid = series.dropna()
    if valid.empty:
        return None
    return round(float(valid.mean()), 4)


def _win_rate(alpha_series: pd.Series) -> float | None:
    valid = alpha_series.dropna()
    if valid.empty:
        return None
    return round(float((valid > 0).mean()), 4)


def _best_by(items: list[dict], key: str) -> str | None:
    """Return the name of the item with the highest value for key."""
    valid = [i for i in items if i.get(key) is not None]
    if not valid:
        return None
    best = max(valid, key=lambda x: x[key])
    name_key = "trigger" if "trigger" in best else "exit_rule" if "exit_rule" in best else "bucket"
    return best.get(name_key)
