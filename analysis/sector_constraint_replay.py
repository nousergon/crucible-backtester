"""sector_constraint_replay.py — Sharpe with vs without sector constraints (config#923).

Tests the sector-balancing hypothesis: does capping per-sector exposure help or
hurt risk-adjusted return? The executor enforces a ``max_sector_pct`` cap
(``synthetic/vectorized_entries.BLOCK_SECTOR_CAP``); this module measures its
effect by replaying the SAME per-cycle selections two ways —

  * **unconstrained**: take the top-N picks as ranked, and
  * **constrained**: take the top-N but drop any pick that would push a sector
    above ``max_sector_pct`` of the cycle's selection (next-best name fills in),

then runs each order list through the REAL simulation plumbing
(``vectorbt_bridge.orders_to_portfolio`` + ``portfolio_stats``) and compares
``sharpe_ratio`` (plus Sortino / alpha / maxDD) across the two arms.

Design mirrors ``analysis/factor_blend_counterfactual_replay.py``:
  * **Deterministic** — no RNG; ties broken by ticker; identical inputs →
    identical metrics.
  * **Config-driven, default OFF** — the live backtest never runs this unless
    explicitly invoked, so wiring it in changes no live path.
  * **Pure replay** — reuses ``vectorbt_bridge``; invents no new sim plumbing.

A "cycle" is ``{"date": <str|Timestamp>, "picks": [{"ticker", "sector",
"rank"?}, ...]}`` — selections already ranked by the live aggregator (rank
ascending = best, or input order when ``rank`` absent).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_PICKS_PER_CYCLE = 8
DEFAULT_HOLD_DAYS = 21
DEFAULT_MAX_SECTOR_PCT = 0.25


def _ranked_picks(picks: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Return picks sorted by ``rank`` asc (ties by ticker); preserves order when
    no rank is present."""
    indexed = list(enumerate(picks))
    indexed.sort(
        key=lambda it: (
            it[1].get("rank", it[0]),
            str(it[1].get("ticker", "")),
        )
    )
    return [dict(p) for _, p in indexed]


def select_with_sector_cap(
    picks: Sequence[Mapping[str, Any]],
    picks_per_cycle: int,
    max_sector_pct: float,
) -> list[dict]:
    """Greedily select up to ``picks_per_cycle`` names honoring a per-sector cap.

    Walks ranked picks best-first; admits a pick only while its sector's share of
    the *target* selection size stays <= ``max_sector_pct``. The cap is computed
    against ``picks_per_cycle`` (the intended sleeve size), so a sector can hold
    at most ``floor(max_sector_pct * picks_per_cycle)`` names — the same notion
    the executor's BLOCK_SECTOR_CAP gate enforces. Deterministic.
    """
    ranked = _ranked_picks(picks)
    if picks_per_cycle <= 0:
        return []
    max_per_sector = max(1, int(math.floor(max_sector_pct * picks_per_cycle)))
    chosen: list[dict] = []
    sector_counts: Counter = Counter()
    for p in ranked:
        if len(chosen) >= picks_per_cycle:
            break
        sector = p.get("sector") or "Unknown"
        if sector_counts[sector] >= max_per_sector:
            continue  # cap reached for this sector — skip, next-best fills in
        chosen.append(p)
        sector_counts[sector] += 1
    return chosen


def select_unconstrained(
    picks: Sequence[Mapping[str, Any]],
    picks_per_cycle: int,
) -> list[dict]:
    """Top-N ranked picks, no sector cap (deterministic)."""
    if picks_per_cycle <= 0:
        return []
    return _ranked_picks(picks)[:picks_per_cycle]


def build_orders(
    cycles: Sequence[Mapping[str, Any]],
    price_matrix: pd.DataFrame,
    *,
    constrained: bool,
    picks_per_cycle: int = DEFAULT_PICKS_PER_CYCLE,
    max_sector_pct: float = DEFAULT_MAX_SECTOR_PCT,
    hold_days: int = DEFAULT_HOLD_DAYS,
    init_cash: float = 1_000_000.0,
) -> list[dict]:
    """Build an ENTER/EXIT order list for one arm (constrained or not).

    Order shape matches what ``backtest._run_simulation_loop`` feeds
    ``orders_to_portfolio``: ``{date, ticker, action, shares, price_at_order}``.
    Equal-weight sizing from a fixed ``init_cash / picks_per_cycle`` budget so
    share counts are deterministic and comparable across the two arms.
    """
    dates = price_matrix.index
    orders: list[dict] = []
    if len(dates) == 0:
        return orders
    per_name_budget = float(init_cash) / max(1, picks_per_cycle)

    for cycle in cycles:
        picks = cycle.get("picks") or []
        if not picks:
            continue
        selected = (
            select_with_sector_cap(picks, picks_per_cycle, max_sector_pct)
            if constrained
            else select_unconstrained(picks, picks_per_cycle)
        )
        if not selected:
            continue

        cdate = pd.Timestamp(cycle.get("date"))
        if cdate not in dates:
            future = dates[dates >= cdate]
            if len(future) == 0:
                continue
            cdate = future[0]
        entry_pos = dates.get_loc(cdate)
        exit_pos = min(entry_pos + hold_days, len(dates) - 1)
        exit_date = dates[exit_pos]

        for p in selected:
            ticker = p.get("ticker")
            if ticker not in price_matrix.columns:
                continue
            entry_price = price_matrix.at[cdate, ticker]
            if pd.isna(entry_price) or float(entry_price) <= 0.0:
                continue
            shares = float(np.floor(per_name_budget / float(entry_price)))
            if shares <= 0.0:
                continue
            orders.append({
                "date": cdate.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "action": "ENTER",
                "shares": shares,
                "price_at_order": float(entry_price),
            })
            if exit_pos > entry_pos:
                orders.append({
                    "date": exit_date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "action": "EXIT",
                })
    return orders


def replay_arm(
    cycles: Sequence[Mapping[str, Any]],
    price_matrix: pd.DataFrame,
    *,
    constrained: bool,
    spy_prices: Optional[pd.Series] = None,
    picks_per_cycle: int = DEFAULT_PICKS_PER_CYCLE,
    max_sector_pct: float = DEFAULT_MAX_SECTOR_PCT,
    hold_days: int = DEFAULT_HOLD_DAYS,
    init_cash: float = 1_000_000.0,
    fees: float = 0.001,
) -> dict[str, Any]:
    """Replay ONE arm through the real simulation and return Sharpe + metrics."""
    from vectorbt_bridge import orders_to_portfolio
    from vectorbt_bridge import portfolio_stats as compute_portfolio_stats

    orders = build_orders(
        cycles, price_matrix, constrained=constrained,
        picks_per_cycle=picks_per_cycle, max_sector_pct=max_sector_pct,
        hold_days=hold_days, init_cash=init_cash,
    )
    arm = "constrained" if constrained else "unconstrained"
    if not orders:
        return {"status": "no_orders", "arm": arm, "n_orders": 0}

    pf = orders_to_portfolio(orders, price_matrix, init_cash=init_cash, fees=fees)
    stats = compute_portfolio_stats(pf, spy_prices=spy_prices)
    return {
        "status": "ok",
        "arm": arm,
        "n_orders": len(orders),
        "sharpe_ratio": _safe(stats.get("sharpe_ratio")),
        "sortino_ratio": _safe(stats.get("sortino_ratio")),
        "total_return": _safe(stats.get("total_return")),
        "total_alpha": _safe(stats.get("total_alpha")),
        "max_drawdown": _safe(stats.get("max_drawdown")),
        "psr": _safe(stats.get("psr")),
        "total_trades": int(stats.get("total_trades") or 0),
    }


def build_sector_balance_report(
    cycles: Sequence[Mapping[str, Any]],
    price_matrix: pd.DataFrame,
    *,
    spy_prices: Optional[pd.Series] = None,
    picks_per_cycle: int = DEFAULT_PICKS_PER_CYCLE,
    max_sector_pct: float = DEFAULT_MAX_SECTOR_PCT,
    hold_days: int = DEFAULT_HOLD_DAYS,
    init_cash: float = 1_000_000.0,
    fees: float = 0.001,
) -> dict[str, Any]:
    """Compare Sharpe (+ Sortino/alpha/maxDD) WITH vs WITHOUT the sector cap.

    Returns::

        {
          "status": "ok" | "skipped" | "no_data",
          "max_sector_pct": float,
          "unconstrained": {<replay_arm metrics>},
          "constrained": {<replay_arm metrics>},
          "sharpe_delta": constrained.sharpe - unconstrained.sharpe,  # >0 → cap helps
          "verdict": "cap_helps" | "cap_hurts" | "neutral" | "inconclusive",
        }
    """
    if not cycles or price_matrix is None or price_matrix.empty:
        return {"status": "skipped", "reason": "no cycles or empty price matrix"}

    common = dict(
        spy_prices=spy_prices, picks_per_cycle=picks_per_cycle,
        max_sector_pct=max_sector_pct, hold_days=hold_days,
        init_cash=init_cash, fees=fees,
    )
    unc = replay_arm(cycles, price_matrix, constrained=False, **common)
    con = replay_arm(cycles, price_matrix, constrained=True, **common)

    if unc.get("status") != "ok" or con.get("status") != "ok":
        return {
            "status": "no_data",
            "max_sector_pct": max_sector_pct,
            "unconstrained": unc,
            "constrained": con,
        }

    s_unc = unc.get("sharpe_ratio")
    s_con = con.get("sharpe_ratio")
    delta = (s_con - s_unc) if (s_unc is not None and s_con is not None) else None
    verdict = _verdict(delta)

    return {
        "status": "ok",
        "max_sector_pct": max_sector_pct,
        "unconstrained": unc,
        "constrained": con,
        "sharpe_delta": round(delta, 4) if delta is not None else None,
        "verdict": verdict,
    }


def _verdict(delta: Optional[float], eps: float = 0.05) -> str:
    if delta is None:
        return "inconclusive"
    if delta > eps:
        return "cap_helps"
    if delta < -eps:
        return "cap_hurts"
    return "neutral"


def _safe(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None
