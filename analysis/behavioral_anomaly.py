"""
behavioral_anomaly.py — behavioral-anomaly eval layer (L4514 / config#698).

Computes four behavioral metrics off the EXISTING decision-capture
substrate (the "integrated decision log" is a derived query layer over
trades.db + research.db — no new hot-path producer; the fifth metric,
turnover, already ships via the executor's L4515 tripwire and is surfaced
by the evaluator tile directly from ``predictor/optimizer_shadow/``):

  1. decision_reversal — per-ticker EXIT -> re-ENTER inside a rolling
     window (default 10 trading days). Churn loops are the signature of
     the 2026-05/06 extreme-decision incidents the L4515 tripwire pages
     on; this measures the per-position frequency rather than the
     portfolio-level turnover band.
  2. conviction_stability — rolling std of the research composite score
     per ticker across consecutive score dates (research.db
     ``score_performance``), summarized over the trailing window. High
     variance = the system keeps changing its mind about a name (a
     future sizing gate input).
  3. cost_adjusted_quality — per completed roundtrip, realized alpha net
     of entry slippage (``realized_alpha_pct`` is percent; trades.db
     ``slippage_vs_signal`` is a FRACTION — converted here), plus the
     cost-drag fraction (roundtrips where slippage consumed more than
     ``cost_drag_threshold`` of gross positive alpha).
  4. portfolio_state_drift — day-over-day one-way L1 distance between
     consecutive ``eod_pnl.positions_snapshot`` weight vectors. Catches
     book-state divergence that per-trade views miss (e.g. the optimizer
     silently dropping a target).

Data sources: trades table + eod_pnl table in trades.db (downloaded from
S3 at runtime), score_performance in research.db — mirroring
``exit_timing.py`` / ``barrier_coherence.py``.

Output contract: dict with top-level ``status`` ("ok" | "insufficient_data"
| "error") and one sub-dict per component, each with its own status —
ALWAYS-EMIT in reporter.py (the evaluator distinguishes "didn't persist"
from "ran, no data").
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date as _date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Defaults — overridable via the backtester config dict.
DEFAULT_REVERSAL_WINDOW_DAYS = 10
DEFAULT_CONVICTION_WINDOW_DAYS = 90
DEFAULT_CONVICTION_MIN_SCORES = 3
DEFAULT_COST_DRAG_THRESHOLD = 0.25
DEFAULT_DRIFT_SPIKE_THRESHOLD = 0.25
_TOP_OFFENDERS = 5


def _days_between(d1: str, d2: str) -> int | None:
    try:
        return abs((_date.fromisoformat(d2[:10]) - _date.fromisoformat(d1[:10])).days)
    except (ValueError, TypeError):
        return None


# ── 1. decision reversal ────────────────────────────────────────────────────


def _compute_decision_reversal(conn: sqlite3.Connection, window_days: int) -> dict:
    trades = pd.read_sql_query(
        "SELECT date, ticker, action FROM trades "
        "WHERE action IN ('ENTER','EXIT') ORDER BY ticker, date",
        conn,
    )
    if trades.empty:
        return {"status": "insufficient_data", "n_trades": 0}

    n_reentries = 0
    reversals: dict[str, int] = {}
    for ticker, group in trades.groupby("ticker"):
        rows = group.reset_index(drop=True)
        for i in range(len(rows) - 1):
            if rows.loc[i, "action"] != "EXIT":
                continue
            nxt = rows.loc[i + 1]
            gap = _days_between(rows.loc[i, "date"], nxt["date"])
            if nxt["action"] == "ENTER" and gap is not None and gap <= window_days:
                n_reentries += 1
                reversals[ticker] = reversals.get(ticker, 0) + 1
    n_exits = int((trades["action"] == "EXIT").sum())

    if n_exits == 0:
        return {"status": "insufficient_data", "n_trades": int(len(trades)), "n_exits": 0}

    offenders = sorted(reversals.items(), key=lambda kv: -kv[1])[:_TOP_OFFENDERS]
    return {
        "status": "ok",
        "window_days": window_days,
        "n_exits": n_exits,
        "n_reversals": n_reentries,
        "reversal_rate": round(n_reentries / n_exits, 4),
        "offenders": [{"ticker": t, "n": n} for t, n in offenders],
    }


# ── 2. conviction stability ─────────────────────────────────────────────────


def _compute_conviction_stability(
    research_db_path: str | None, window_days: int, min_scores: int,
) -> dict:
    if not research_db_path or not Path(research_db_path).exists():
        return {"status": "insufficient_data", "reason": "research.db unavailable"}
    try:
        conn = sqlite3.connect(research_db_path)
        try:
            df = pd.read_sql_query(
                "SELECT symbol, score_date, score FROM score_performance "
                "WHERE score IS NOT NULL ORDER BY score_date",
                conn,
            )
        finally:
            conn.close()
    except (sqlite3.Error, pd.errors.DatabaseError) as e:
        logger.warning("behavioral_anomaly: score_performance query failed: %s", e)
        return {"status": "error", "error": str(e)}

    if df.empty:
        return {"status": "insufficient_data", "n_scores": 0}

    df["score_date"] = pd.to_datetime(df["score_date"])
    cutoff = df["score_date"].max() - pd.Timedelta(days=window_days)
    df = df[df["score_date"] >= cutoff]

    stds = (
        df.groupby("symbol")["score"]
        .agg(["std", "count", "mean"])
        .dropna(subset=["std"])
    )
    stds = stds[stds["count"] >= min_scores]
    if stds.empty:
        return {
            "status": "insufficient_data",
            "reason": f"no ticker has >= {min_scores} scores in trailing {window_days}d",
        }

    worst = stds.sort_values("std", ascending=False).head(_TOP_OFFENDERS)
    return {
        "status": "ok",
        "window_days": window_days,
        "n_tickers": int(len(stds)),
        "median_score_std": round(float(stds["std"].median()), 4),
        "p90_score_std": round(float(stds["std"].quantile(0.9)), 4),
        "high_variance": [
            {"ticker": t, "score_std": round(float(r["std"]), 2),
             "n_scores": int(r["count"]), "mean_score": round(float(r["mean"]), 1)}
            for t, r in worst.iterrows()
        ],
    }


# ── 3. cost-adjusted decision quality ───────────────────────────────────────


def _compute_cost_adjusted_quality(conn: sqlite3.Connection, drag_threshold: float) -> dict:
    rts = pd.read_sql_query(
        "SELECT e.ticker, e.realized_alpha_pct, en.slippage_vs_signal "
        "FROM trades e JOIN trades en ON e.entry_trade_id = en.trade_id "
        "WHERE e.entry_trade_id IS NOT NULL AND e.realized_alpha_pct IS NOT NULL",
        conn,
    )
    if rts.empty:
        return {"status": "insufficient_data", "n_roundtrips": 0}

    # slippage_vs_signal is a fraction (daemon: (fill-signal)/signal);
    # realized_alpha_pct is percent. Positive entry slippage = paid up = cost.
    rts["slippage_pct"] = rts["slippage_vs_signal"].fillna(0.0) * 100.0
    rts["net_alpha_pct"] = rts["realized_alpha_pct"] - rts["slippage_pct"]

    winners = rts[rts["realized_alpha_pct"] > 0]
    n_dragged = int(
        (winners["slippage_pct"] > drag_threshold * winners["realized_alpha_pct"]).sum()
    ) if not winners.empty else 0

    return {
        "status": "ok",
        "n_roundtrips": int(len(rts)),
        "median_gross_alpha_pct": round(float(rts["realized_alpha_pct"].median()), 3),
        "median_slippage_pct": round(float(rts["slippage_pct"].median()), 4),
        "median_net_alpha_pct": round(float(rts["net_alpha_pct"].median()), 3),
        "cost_drag_threshold": drag_threshold,
        "n_winners": int(len(winners)),
        "n_cost_dragged_winners": n_dragged,
        "cost_drag_fraction": round(n_dragged / len(winners), 4) if len(winners) else None,
    }


# ── 4. portfolio-state drift ────────────────────────────────────────────────


def _weights_from_snapshot(raw: str) -> dict[str, float] | None:
    try:
        positions = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(positions, list):
        return None
    mvs = {}
    for pos in positions:
        if not isinstance(pos, dict):
            continue
        ticker = pos.get("ticker") or pos.get("symbol")
        mv = pos.get("market_value")
        if ticker and isinstance(mv, (int, float)) and mv > 0:
            mvs[ticker] = mvs.get(ticker, 0.0) + float(mv)
    total = sum(mvs.values())
    if total <= 0:
        return None
    return {t: v / total for t, v in mvs.items()}


def _compute_state_drift(conn: sqlite3.Connection, spike_threshold: float) -> dict:
    snaps = pd.read_sql_query(
        "SELECT date, positions_snapshot FROM eod_pnl "
        "WHERE positions_snapshot IS NOT NULL ORDER BY date",
        conn,
    )
    if len(snaps) < 2:
        return {"status": "insufficient_data", "n_snapshots": int(len(snaps))}

    n_unparseable = 0
    series: list[tuple[str, dict[str, float]]] = []
    for _, row in snaps.iterrows():
        w = _weights_from_snapshot(row["positions_snapshot"])
        if w is None:
            n_unparseable += 1
            continue
        series.append((row["date"], w))
    if n_unparseable:
        logger.warning(
            "behavioral_anomaly: %d/%d positions_snapshot rows unparseable (skipped)",
            n_unparseable, len(snaps),
        )
    if len(series) < 2:
        return {"status": "insufficient_data", "n_snapshots": int(len(snaps)),
                "n_unparseable": n_unparseable}

    drifts = []
    for (d_prev, w_prev), (d_cur, w_cur) in zip(series, series[1:]):
        tickers = set(w_prev) | set(w_cur)
        l1 = sum(abs(w_cur.get(t, 0.0) - w_prev.get(t, 0.0)) for t in tickers)
        drifts.append({"date": d_cur, "drift": round(l1 / 2.0, 4)})  # one-way

    vals = pd.Series([d["drift"] for d in drifts])
    spikes = [d for d in drifts if d["drift"] > spike_threshold]
    return {
        "status": "ok",
        "n_days": int(len(drifts)),
        "n_unparseable": n_unparseable,
        "median_daily_drift": round(float(vals.median()), 4),
        "max_daily_drift": round(float(vals.max()), 4),
        "spike_threshold": spike_threshold,
        "n_spike_days": len(spikes),
        "spike_days": spikes[-_TOP_OFFENDERS:],
    }


# ── entry point ─────────────────────────────────────────────────────────────


def compute_behavioral_anomaly(
    trades_db_path: str,
    research_db_path: str | None = None,
    config: dict | None = None,
) -> dict:
    """
    Compute the behavioral-anomaly metric suite (L4514 components 2-5;
    component 1, turnover, is the executor's L4515 tripwire surfaced
    directly by the evaluator tile).

    Returns a dict with top-level status and per-component sub-dicts:
    decision_reversal, conviction_stability, cost_adjusted_quality,
    portfolio_state_drift.
    """
    cfg = config or {}
    if not Path(trades_db_path).exists():
        return {"status": "error", "error": f"trades.db not found at {trades_db_path}"}

    try:
        conn = sqlite3.connect(trades_db_path)
    except sqlite3.Error as e:
        return {"status": "error", "error": f"trades.db open failed: {e}"}

    try:
        components = {
            "decision_reversal": _compute_decision_reversal(
                conn, int(cfg.get("reversal_window_days", DEFAULT_REVERSAL_WINDOW_DAYS)),
            ),
            "conviction_stability": _compute_conviction_stability(
                research_db_path,
                int(cfg.get("conviction_window_days", DEFAULT_CONVICTION_WINDOW_DAYS)),
                int(cfg.get("conviction_min_scores", DEFAULT_CONVICTION_MIN_SCORES)),
            ),
            "cost_adjusted_quality": _compute_cost_adjusted_quality(
                conn, float(cfg.get("cost_drag_threshold", DEFAULT_COST_DRAG_THRESHOLD)),
            ),
            "portfolio_state_drift": _compute_state_drift(
                conn, float(cfg.get("drift_spike_threshold", DEFAULT_DRIFT_SPIKE_THRESHOLD)),
            ),
        }
    except (sqlite3.Error, pd.errors.DatabaseError) as e:
        # Schema-level failure (e.g. pre-migration trades.db without the
        # roundtrip columns) — surface as an explicit error artifact rather
        # than crashing the Saturday eval chain; the evaluator grades it.
        logger.warning("behavioral_anomaly: component computation failed: %s", e)
        return {"status": "error", "error": str(e)}
    finally:
        conn.close()

    statuses = [c.get("status") for c in components.values()]
    if any(s == "ok" for s in statuses):
        status = "ok"
    elif any(s == "error" for s in statuses):
        status = "error"
    else:
        status = "insufficient_data"

    return {"status": status, **components}
