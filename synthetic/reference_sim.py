"""
reference_sim.py — Independent NumPy/pandas portfolio oracle for leg (b) of the
backtester correctness battery (ROADMAP L4593).

WHY THIS EXISTS
---------------
vectorbt is the backtester's only simulation engine, and it auto-writes four live
config files every Saturday — so there is no second implementation to check it
against. This module IS that second implementation: a from-scratch portfolio
accountant that reads the same executor order list + price matrix and reproduces
the portfolio NAV path, total return, drawdown, trade tally, and alpha-vs-SPY —
WITHOUT importing vectorbt. Leg (b)'s test asserts the production path
(``vectorbt_bridge.orders_to_portfolio`` + ``portfolio_stats``) and this oracle
agree to near machine precision on hand-constructed scenarios where every fill,
fee, and return is independently computable.

This catches the bug class the null-calibration leg (a) cannot: a *systematic*
accounting error (wrong fill price, mis-applied fee side, NAV mark-to-market off
by a day, alpha computed over the wrong window) that biases EVERY backtest the
same way — invisible to a centered-on-zero null test, fatal to live tuning.

CONVENTIONS (empirically pinned against vectorbt 0.28.4 + vectorbt_bridge)
-------------------------------------------------------------------------
- Fills occur at the ORDER DATE's close from the price matrix (``assume_next_day_fill``
  is not modeled here — the bridge default is False).
- ``total_fees = fees + slippage_bps / 10_000``; a fee of ``trade_value * total_fees``
  is charged on BOTH entry and exit.
- NAV(t) = cash(t) + Σ shares_held(t) · close(t); cash is a single shared pool
  (matches the bridge's ``cash_sharing=True, group_by=True``).
- ``max_drawdown`` = min over t of (NAV(t) − running_peak(t)) / running_peak(t).
- Active window = [first date NAV departs from its initial value (atol 1e-9),
  last date]; ``spy_return`` compounds SPY over that window; ``total_alpha =
  total_return − spy_return``. Mirrors ``vectorbt_bridge._compute_active_window``.

SUPPORTED DOMAIN (fail-loud outside it — see [[feedback_no_silent_fails]])
--------------------------------------------------------------------------
At most one open position per ticker at a time: an ENTER is valid only when that
ticker is flat, and an EXIT closes the full position. This is vectorbt's default
``accumulate=False`` behavior; rather than silently replicate the quirk for
pyramiding/partial orders, the oracle RAISES so closed-form scenarios stay
unambiguous. Construct scenarios within this domain (use distinct tickers for
concurrent positions).

``total_trades`` / ``win_rate`` count CLOSED round trips only. A position still
open on the last day is correctly marked-to-market in ``nav`` (and therefore in
``total_return`` / ``max_drawdown`` / ``total_alpha``) but is NOT counted as a
trade — this differs from vectorbt's ``trades.count()``, which tallies open
trades too. So assert ``total_trades`` / ``win_rate`` agreement only on
fully-closed scenarios; the NAV-derived fields agree regardless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class UnsupportedOrderPattern(ValueError):
    """Raised when orders fall outside the oracle's closed-form domain."""


@dataclass
class ReferenceResult:
    """Mirror of the ``vectorbt_bridge.portfolio_stats`` fields the oracle owns."""
    nav: pd.Series = field(repr=False)
    total_return: float
    daily_returns: pd.Series = field(repr=False)
    max_drawdown: float
    total_trades: int
    win_rate: float
    active_window: tuple[pd.Timestamp, pd.Timestamp] | None
    spy_return: float | None
    total_alpha: float | None


def simulate_reference(
    orders: list[dict],
    prices: pd.DataFrame,
    *,
    init_cash: float = 1_000_000.0,
    fees: float = 0.0,
    slippage_bps: float = 0.0,
    spy_prices: pd.Series | None = None,
) -> ReferenceResult:
    """Independently simulate an executor order list against a price matrix.

    Args mirror :func:`vectorbt_bridge.orders_to_portfolio` /
    :func:`vectorbt_bridge.portfolio_stats`. Returns a :class:`ReferenceResult`
    whose fields can be asserted equal to the production path.

    Raises:
        UnsupportedOrderPattern: an order references an unknown ticker/date, an
            ENTER lands on a ticker that is already open, or an EXIT lands on a
            flat ticker.
    """
    total_fees = fees + slippage_bps / 10_000.0
    dates = prices.index
    date_set = set(dates)

    # Group orders by fill date for day-by-day processing.
    orders_by_date: dict[pd.Timestamp, list[dict]] = {}
    for o in orders:
        d = pd.Timestamp(o["date"])
        if d not in date_set:
            raise UnsupportedOrderPattern(f"order date {d.date()} not in price index")
        if o["ticker"] not in prices.columns:
            raise UnsupportedOrderPattern(f"order ticker {o['ticker']!r} not in price matrix")
        orders_by_date.setdefault(d, []).append(o)

    cash = float(init_cash)
    shares: dict[str, float] = {}      # ticker → open share count (>0 means open)
    entry_cost: dict[str, float] = {}  # ticker → cash paid to open (incl. entry fee)
    trade_pnls: list[float] = []
    nav_values: list[float] = []

    for d in dates:
        for o in orders_by_date.get(d, []):
            t = o["ticker"]
            fill_price = float(prices.loc[d, t])
            if not np.isfinite(fill_price) or fill_price <= 0:
                raise UnsupportedOrderPattern(f"non-positive fill price for {t} on {d.date()}")
            action = o["action"]
            if action == "ENTER":
                if shares.get(t, 0.0) > 0.0:
                    raise UnsupportedOrderPattern(
                        f"ENTER on already-open {t} on {d.date()} — outside closed-form domain"
                    )
                s = float(o.get("shares", 0.0))
                if s <= 0:
                    continue
                cost = s * fill_price * (1.0 + total_fees)
                cash -= cost
                shares[t] = s
                entry_cost[t] = cost
            elif action in ("EXIT", "REDUCE"):
                s = shares.get(t, 0.0)
                if s <= 0.0:
                    raise UnsupportedOrderPattern(
                        f"EXIT on flat {t} on {d.date()} — outside closed-form domain"
                    )
                proceeds = s * fill_price * (1.0 - total_fees)
                cash += proceeds
                trade_pnls.append(proceeds - entry_cost.get(t, 0.0))
                shares[t] = 0.0
                entry_cost.pop(t, None)
            else:
                raise UnsupportedOrderPattern(f"unknown action {action!r}")

        # Mark-to-market NAV at this day's close.
        held_value = sum(q * float(prices.loc[d, tk]) for tk, q in shares.items() if q > 0.0)
        nav_values.append(cash + held_value)

    nav = pd.Series(nav_values, index=dates, name="nav", dtype=np.float64)
    total_return = float(nav.iloc[-1] / init_cash - 1.0)

    daily_returns = nav.pct_change().dropna()
    daily_returns.name = "daily_return"

    peak = nav.cummax()
    drawdown = (nav - peak) / peak
    max_drawdown = float(drawdown.min())

    total_trades = len(trade_pnls)
    win_rate = float(np.mean([p > 0 for p in trade_pnls])) if total_trades else float("nan")

    # Active window — first NAV departure from its initial value through last date.
    active_window: tuple[pd.Timestamp, pd.Timestamp] | None = None
    spy_return: float | None = None
    total_alpha: float | None = None
    if len(nav) >= 2:
        initial = float(nav.iloc[0])
        changed = nav[~np.isclose(nav.to_numpy(dtype=np.float64), initial, atol=1e-9)]
        if len(changed) > 0:
            active_window = (changed.index[0], dates[-1])

    if spy_prices is not None and active_window is not None:
        a0, a1 = active_window
        spy_aligned = spy_prices.loc[a0:a1].dropna()
        if len(spy_aligned) >= 2:
            spy_return = float(spy_aligned.iloc[-1] / spy_aligned.iloc[0] - 1.0)
            total_alpha = total_return - spy_return

    return ReferenceResult(
        nav=nav,
        total_return=total_return,
        daily_returns=daily_returns,
        max_drawdown=max_drawdown,
        total_trades=total_trades,
        win_rate=win_rate,
        active_window=active_window,
        spy_return=spy_return,
        total_alpha=total_alpha,
    )
