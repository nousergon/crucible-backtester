"""
barrier_coherence.py — Predictor↔executor triple-barrier coherence diagnostic.

READ-ONLY. Quantifies the mismatch between the barriers the predictor is
*trained on* and the barriers the executor actually *realizes*.

  - The predictor trains on López de Prado triple-barrier LABELS
    (`alpha-engine-predictor/labeling/triple_barrier.py`): a fixed
    ``forward_window`` vertical (time) barrier and symmetric ±``vol_multiplier``·σ
    horizontal barriers, computed once at entry.
  - The executor realizes OCO bracket exits
    (`alpha-engine/executor/strategies/exit_manager.py`): an ATR *trailing*
    stop, a profit-take that only REDUCES, and a time-decay exit that fires
    only on a Research HOLD signal. Those three are sweep-tuned weekly by the
    backtester (``optimizer/executor_optimizer.py`` → ``config/executor_params.json``).

If the model is trained to predict outcomes under one barrier policy but the
executor realizes outcomes under a different one, the model's IC / calibration
measures a counterfactual that is never traded. This module measures that gap.

Three legs:
  (a) ``definition_divergence`` — static comparison of the label barriers vs the
      live execution barriers (vertical horizon + horizontal geometry). Always
      emitted; needs no trades.
  (b) ``horizon_coherence`` — realized holding-period distribution vs the label
      vertical barrier; headline scalar is the fraction of roundtrips that exit
      BEFORE the label horizon.
  (c) ``barrier_touch_mix`` — maps each realized ``exit_reason`` to its
      triple-barrier analog {upper / lower / vertical / non-barrier} and reports
      the realized mix, so the executor's realized barrier profile can be
      compared to the symmetric ±kσ / fixed-horizon label assumption.

Per-trade label-vs-realized concordance via offline price replay is the DEEPER
leg of Task A and is intentionally DEFERRED (it requires replaying the label
function over per-ticker OHLC for every pick). The aggregate mix comparison here
already answers the design question — "do we realize a different barrier profile
than we train on?" — without it. See the predictor↔executor triple-barrier
coherence arc on the ROADMAP (2026-05-29).

Data source: trades.db (downloaded from S3 at runtime), mirroring
``post_trade.py`` / ``exit_timing.py``.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# Predictor triple-barrier LABEL defaults. Source of truth: ``predictor.yaml``
# ``triple_barrier`` block (alpha-engine-predictor/config.py:199-209). The
# backtester is read-only wrt the predictor, so these documented defaults are
# the fallback when an explicit ``label_config`` is not injected.
_LABEL_DEFAULTS: dict = {
    "forward_window": 21,    # trading-day vertical (time) barrier
    "vol_window": 20,        # lookback for σ estimation
    "vol_multiplier": 2.0,   # horizontal barrier half-width = vol_multiplier·σ
}

# Executor execution-barrier defaults. Source of truth: the LIVE, sweep-tuned
# ``config/executor_params.json`` on S3; static fallbacks from
# alpha-engine/executor/strategies/config.py.
_EXEC_DEFAULTS: dict = {
    "atr_multiplier": 2.5,         # trailing-stop width (lower barrier)
    "profit_take_pct": 0.25,       # take-profit (upper barrier), reduce-only
    "time_decay_reduce_days": 7,   # reduce on HOLD
    "time_decay_exit_days": 14,    # full exit on HOLD (time barrier)
}

# Canonical ``exit_reason`` strings → triple-barrier analog. Strings verified
# against alpha-engine/executor/strategies/exit_manager.py. ``non_barrier``
# covers event/signal-driven exits that have no triple-barrier analog (they are
# neither a price nor a time barrier the label models).
_BARRIER_CLASS: dict = {
    "profit_take": "upper_barrier",
    "atr_trailing_stop": "lower_barrier",
    "fallback_stop": "lower_barrier",
    "momentum_exit": "lower_barrier",
    "time_decay_exit": "vertical_barrier",
    "time_decay_reduce": "vertical_barrier",
    "time_expiry": "vertical_barrier",
    "catalyst_hard_exit": "non_barrier",
    "signal_exit": "non_barrier",
    "research_exit": "non_barrier",
}

_CLASS_ORDER = [
    "upper_barrier",
    "lower_barrier",
    "vertical_barrier",
    "non_barrier",
    "other",
    "unknown",
]


def to_barrier_class(exit_reason: str | None) -> str:
    """Map a raw ``exit_reason`` to its triple-barrier analog.

    Exact match first, then substring (handles prefixed/suffixed variants),
    then generic price/time keywords. Returns ``"unknown"`` for empty input and
    ``"other"`` for an unrecognized non-empty reason — never raises, so a novel
    exit reason surfaces as a visible bucket rather than crashing the diagnostic.
    """
    if not exit_reason or not str(exit_reason).strip():
        return "unknown"
    r = str(exit_reason).strip().lower()
    if r in _BARRIER_CLASS:
        return _BARRIER_CLASS[r]
    for key, cls in _BARRIER_CLASS.items():
        if key in r:
            return cls
    if "stop" in r or "trail" in r:
        return "lower_barrier"
    if "profit" in r or "target" in r or "take" in r:
        return "upper_barrier"
    if "time" in r or "expiry" in r or "decay" in r:
        return "vertical_barrier"
    return "other"


def compute_barrier_coherence(
    trades_db_path: str,
    *,
    label_config: dict | None = None,
    exec_params: dict | None = None,
    exec_params_source: str = "defaults (executor/strategies/config.py)",
    min_trades: int = 3,
) -> dict:
    """Predictor↔executor triple-barrier coherence diagnostic.

    Pure-compute given an (optional) injected ``label_config`` / ``exec_params``;
    the S3 read of the live executor params lives in the ``evaluate.py`` wrapper
    so this function stays unit-testable without network I/O.

    Returns dict with:
        status: "ok"  (always — the definition leg needs no trades)
        label_config / exec_params / exec_params_source
        definition_divergence: {vertical, horizontal}   (always present)
        horizon_coherence: {status, ...}                (trade-based)
        barrier_touch_mix: {status, ...}                (trade-based)
        trades_status: "ok" | "error"
    """
    label_cfg = {**_LABEL_DEFAULTS, **(label_config or {})}
    exec_cfg = {**_EXEC_DEFAULTS, **(exec_params or {})}

    result: dict = {
        "status": "ok",
        "label_config": label_cfg,
        "exec_params": exec_cfg,
        "exec_params_source": exec_params_source,
        "definition_divergence": _definition_divergence(label_cfg, exec_cfg),
    }

    if not Path(trades_db_path).exists():
        err = f"trades.db not found at {trades_db_path}"
        logger.warning("barrier_coherence: %s — trade-based legs skipped", err)
        result["horizon_coherence"] = {"status": "error", "error": err}
        result["barrier_touch_mix"] = {"status": "error", "error": err}
        result["trades_status"] = "error"
        return result

    try:
        conn = sqlite3.connect(trades_db_path)
        exits_df = pd.read_sql_query(
            "SELECT ticker, date, exit_reason, realized_return_pct, "
            "realized_alpha_pct, days_held "
            "FROM trades WHERE action IN ('EXIT', 'REDUCE') "
            "AND entry_trade_id IS NOT NULL",
            conn,
        )
        conn.close()
    except Exception as e:  # noqa: BLE001 — diagnostic: record the failure surface, keep the definition leg.
        # NOT a silent swallow: the error is logged AND surfaced in the returned
        # artifact (rendered by reporter); the primary deliverable (the static
        # definition-divergence leg) still returns. Per the no-silent-fails rule
        # this is the "secondary observability that records its own failure" case.
        logger.warning("barrier_coherence: trades.db query failed: %s", e)
        result["horizon_coherence"] = {"status": "error", "error": str(e)}
        result["barrier_touch_mix"] = {"status": "error", "error": str(e)}
        result["trades_status"] = "error"
        return result

    result["horizon_coherence"] = _horizon_coherence(
        exits_df, int(label_cfg["forward_window"]), min_trades
    )
    result["barrier_touch_mix"] = _barrier_touch_mix(exits_df, min_trades)
    result["trades_status"] = "ok"
    return result


def _definition_divergence(label_cfg: dict, exec_cfg: dict) -> dict:
    """Static label-barrier vs execution-barrier comparison (no trades needed)."""
    fw = int(label_cfg["forward_window"])
    exit_days = int(exec_cfg["time_decay_exit_days"])
    return {
        "vertical": {
            "label_horizon_trading_days": fw,
            "exec_time_barrier_trading_days": exit_days,
            "exec_time_barrier_conditional": "fires only on a Research HOLD signal",
            "horizon_gap_days": fw - exit_days,
            "coherent": exit_days == fw,
            "note": (
                f"Label assumes a fixed {fw}-trading-day vertical barrier for "
                f"every name; the executor's time barrier is {exit_days}d AND "
                "conditional on a HOLD signal, so most names have no "
                "unconditional time barrier at the label horizon."
            ),
        },
        "horizontal": {
            "label_geometry": (
                f"symmetric ±{label_cfg['vol_multiplier']}·σ "
                f"(σ over {label_cfg['vol_window']}d), fixed at entry"
            ),
            "exec_upper": (
                f"profit-take at +{exec_cfg['profit_take_pct']:.0%} "
                "(REDUCES 50%, not a full exit)"
            ),
            "exec_lower": (
                f"ATR trailing stop at {exec_cfg['atr_multiplier']}×ATR "
                "(TRAILS the high-water mark, not fixed at entry)"
            ),
            "coherent": False,
            "note": (
                "Label barriers are symmetric, fixed-at-entry and vol-scaled. "
                "Execution barriers are asymmetric (profit-take only REDUCES; "
                "the stop TRAILS), so the realized exit geometry is not the "
                "symmetric ±kσ the model is trained to predict."
            ),
        },
    }


def _horizon_coherence(exits_df: pd.DataFrame, forward_window: int, min_trades: int) -> dict:
    """Realized holding period vs the label vertical (time) barrier."""
    rt = exits_df[exits_df["days_held"].notna()].copy()
    n = len(rt)
    if n < min_trades:
        return {"status": "insufficient_data", "n": n, "min_trades": min_trades}
    dh = rt["days_held"].astype(float)
    before = int((dh < forward_window).sum())
    return {
        "status": "ok",
        "n": n,
        "label_horizon_days": forward_window,
        "median_days_held": round(float(dh.median()), 2),
        "mean_days_held": round(float(dh.mean()), 2),
        "p25_days_held": round(float(dh.quantile(0.25)), 2),
        "p75_days_held": round(float(dh.quantile(0.75)), 2),
        "n_exit_before_label_horizon": before,
        "pct_exit_before_label_horizon": round(before / n, 4),
    }


def _barrier_touch_mix(exits_df: pd.DataFrame, min_trades: int) -> dict:
    """Realized first-barrier-touched mix, mapped to triple-barrier analogs."""
    n_total = len(exits_df)
    if n_total < min_trades:
        return {"status": "insufficient_data", "n": n_total, "min_trades": min_trades}

    df = exits_df.copy()
    df["barrier_class"] = df["exit_reason"].apply(to_barrier_class)

    rows = []
    for cls in _CLASS_ORDER:
        sub = df[df["barrier_class"] == cls]
        if sub.empty:
            continue
        rows.append({
            "barrier_class": cls,
            "n": len(sub),
            "pct": round(len(sub) / n_total, 4),
            "avg_days_held": _safe_mean(sub["days_held"]),
            "avg_alpha_pct": _safe_mean(sub["realized_alpha_pct"]),
        })

    by = {r["barrier_class"]: r["pct"] for r in rows}
    return {
        "status": "ok",
        "n": n_total,
        "by_class": rows,
        "pct_upper": by.get("upper_barrier", 0.0),
        "pct_lower": by.get("lower_barrier", 0.0),
        "pct_vertical": by.get("vertical_barrier", 0.0),
        "pct_non_barrier": by.get("non_barrier", 0.0),
    }


def _safe_mean(series: pd.Series) -> float | None:
    valid = series.dropna()
    if valid.empty:
        return None
    return round(float(valid.mean()), 4)
