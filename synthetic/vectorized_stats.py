"""Vectorized portfolio stats — numpy-direct, no vectorbt.

Background: Tier 4 Layer 3 v16 (2026-04-28) caught a hang in the
per-combo `vectorbt.Portfolio.from_orders` + `pf.sharpe_ratio()` chain
called from `vectorbt_bridge.portfolio_stats`. Watchdog tripped after
90 minutes with the stack pinned at vectorbt's accessor / config
machinery deep inside `get_returns_acc`. With 60 combos × ~26k orders
each, sequential vectorbt-portfolio instantiation + accessor builds
exceeded the predictor_pipeline cap.

This module replaces that path entirely for the vectorized sweep:
the sweep loop already tracks per-combo NAV trajectory in a
`[n_combos, n_dates]` matrix (mark-to-market each date). All stats
that vectorbt computes (total_return, sharpe, max_drawdown, calmar,
total_alpha) derive from that NAV trajectory in O(n_combos × n_dates)
vectorized numpy ops — under 100 ms for the v16 fixture vs >90 min
for the vectorbt path.

Trade counts (total_trades + win_rate) walk the columnar order
buffers per combo to count entry → EXIT round-trips. ~O(total_orders)
single pass per combo, no DataFrame construction.

Fee parity caveat: the vectorized sim is fee-free (`cash -= shares *
price`, no fee deduction). Scalar `predictor_single_run` runs orders
through `vectorbt.Portfolio.from_orders` with `fees=0.001` (10 bps
round-trip), so its NAV reflects fees while ours does not. For 9k
orders × $1k average × 10 bps = ~$9k fees per combo on a $1M portfolio
≈ ~0.9% NAV offset. Quantifiable; documented; addressed in a
follow-up that wires fee deduction into `vectorized_sim.apply_buy/sell`.
Until then, vectorized stats overstate `total_return` by the
fee-equivalent fraction. Does NOT affect relative ranking of combos
(fees apply uniformly), so optimizer config selection is unaffected.
"""
from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from synthetic.vectorized_orders import (
    ACTION_ENTER,
    VectorizedOrderStore,
    _OrderBuffer,
)
from synthetic.vectorized_exits import ACTION_EXIT, ACTION_REDUCE

logger = logging.getLogger(__name__)


# Annualization factor — standard "trading days per year" assumption
# used by vectorbt and the scalar reference path. Matches
# `vectorbt_bridge.portfolio_stats`'s implicit yearly axis.
_TRADING_DAYS_PER_YEAR = 252


# ── Per-combo stat primitives (vectorized over [n_combos, n_dates]) ─────────


def compute_total_return(
    nav_history: np.ndarray, init_cash: float,
) -> np.ndarray:
    """Per-combo total return vs initial cash.

    Returns
    -------
    np.ndarray, shape [n_combos]
    """
    final_nav = nav_history[:, -1]
    return final_nav / init_cash - 1.0


def compute_daily_returns(nav_history: np.ndarray) -> np.ndarray:
    """Per-combo daily simple returns.

    Returns
    -------
    np.ndarray, shape [n_combos, n_dates - 1]
        ``daily_returns[c, t]`` = (nav[c, t+1] / nav[c, t]) - 1.

    NaN-safe: divisor of 0 returns 0 (treated as flat, not infinite —
    a NAV of 0 means total wipeout, no further return movement
    measurable).
    """
    if nav_history.shape[1] < 2:
        return np.zeros((nav_history.shape[0], 0), dtype=np.float64)
    prev = nav_history[:, :-1]
    curr = nav_history[:, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(prev > 0, (curr / prev) - 1.0, 0.0)
    return out


def compute_sharpe_ratio(daily_returns: np.ndarray) -> np.ndarray:
    """Per-combo annualized Sharpe ratio (risk-free = 0).

    Returns
    -------
    np.ndarray, shape [n_combos]

    Matches vectorbt's default Sharpe: ``mean / std * sqrt(252)``,
    using sample std (ddof=1) for parity with `pandas.Series.std()`.
    Combos with zero variance return 0 (avoids division-by-zero noise).
    """
    if daily_returns.shape[1] < 2:
        return np.zeros(daily_returns.shape[0], dtype=np.float64)
    mean = daily_returns.mean(axis=1)
    std = daily_returns.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(
            std > 0,
            mean / std * np.sqrt(_TRADING_DAYS_PER_YEAR),
            0.0,
        )
    return sharpe


def compute_sortino_ratio(daily_returns: np.ndarray, target: float = 0.0) -> np.ndarray:
    """Per-combo annualized Sortino ratio (target = 0 by default).

    Sortino vs Sharpe: numerator is mean excess (same), denominator is
    downside RMS only — sqrt(mean(min(r - target, 0)**2)). Penalizes only
    below-target volatility, the right shape for long-only risk-seeking
    strategies where upside vol is the *goal*, not a cost.

    Returns
    -------
    np.ndarray, shape [n_combos]

    Combos with no below-target days return 0.0 (no downside, no
    risk-adjusted denominator). Combos with fewer than 2 dates return 0.0.
    """
    n_combos, n_steps = daily_returns.shape
    if n_steps < 2:
        return np.zeros(n_combos, dtype=np.float64)
    excess = daily_returns - target
    downside = np.minimum(excess, 0.0)
    downside_var = (downside * downside).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sortino = np.where(
            downside_var > 0,
            excess.mean(axis=1) / np.sqrt(downside_var) * np.sqrt(_TRADING_DAYS_PER_YEAR),
            0.0,
        )
    return sortino


def compute_cvar(daily_returns: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Per-combo Conditional Value at Risk at the q-quantile.

    CVaR_q = mean of the worst q-fraction of daily returns. Reported as
    the raw return value — negative numbers indicate tail losses.

    Returns
    -------
    np.ndarray, shape [n_combos]

    Combos with fewer than ceil(1/q) observations return 0.0
    (insufficient resolution to define the tail).
    """
    if not (0.0 < q < 1.0):
        raise ValueError(f"q must be in (0, 1), got {q}")
    n_combos, n_steps = daily_returns.shape
    min_n = int(np.ceil(1.0 / q))
    if n_steps < min_n:
        return np.zeros(n_combos, dtype=np.float64)
    n_tail = max(1, int(np.floor(n_steps * q)))
    sorted_returns = np.sort(daily_returns, axis=1)
    return sorted_returns[:, :n_tail].mean(axis=1)


def compute_max_drawdown(nav_history: np.ndarray) -> np.ndarray:
    """Per-combo max drawdown (negative number; 0 if monotonically
    non-decreasing).

    Returns
    -------
    np.ndarray, shape [n_combos]

    Definition matches vectorbt: drawdown = (nav - running_max) /
    running_max; max_drawdown = min over time. NAV that touches 0
    yields drawdown of -1.0. Constant-NAV combos return 0.
    """
    running_max = np.maximum.accumulate(nav_history, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.where(
            running_max > 0,
            (nav_history - running_max) / running_max,
            0.0,
        )
    return drawdown.min(axis=1)


def compute_calmar_ratio(
    total_return: np.ndarray,
    max_drawdown: np.ndarray,
    n_dates: int,
) -> np.ndarray:
    """Per-combo Calmar ratio: annualized return / |max drawdown|.

    Combos with zero drawdown return 0 (no risk-adjusted denom).
    """
    if n_dates < 2:
        return np.zeros_like(total_return)
    n_years = n_dates / _TRADING_DAYS_PER_YEAR
    # (1 + r)^(1/years) - 1, guarded for negative compounding.
    base = 1.0 + total_return
    with np.errstate(invalid="ignore"):
        annualized = np.where(
            base > 0,
            np.power(base, 1.0 / n_years) - 1.0,
            -1.0,  # full loss → -100% annualized
        )
    abs_dd = np.abs(max_drawdown)
    with np.errstate(divide="ignore", invalid="ignore"):
        calmar = np.where(abs_dd > 0, annualized / abs_dd, 0.0)
    return calmar


# ── SPY-relative stats ──────────────────────────────────────────────────────


def compute_spy_return(
    spy_prices: pd.Series | None,
    dates: pd.DatetimeIndex,
) -> float | None:
    """SPY total return over the portfolio's active date range.

    None if no SPY data or fewer than 2 valid prices.
    """
    if spy_prices is None:
        return None
    aligned = spy_prices.reindex(dates).dropna()
    if len(aligned) < 2:
        return None
    return float((aligned.iloc[-1] / aligned.iloc[0]) - 1.0)


# ── Trade counting (per-combo, columnar buffer walk) ────────────────────────


def count_trades_from_buffer(buf: _OrderBuffer | None) -> tuple[int, int]:
    """Count entry → EXIT round-trip trades + winning trades for one
    combo's order buffer.

    Returns
    -------
    (n_trades, n_winning_trades) : tuple[int, int]

    Semantics (matches vectorbt's "trade" definition for the common
    case of single-position-per-ticker):
      - ENTER opens a position; entry price is recorded per ticker.
      - EXIT closes the position; if exit_price > entry_price, it's a
        winning trade. Entry slot is freed.
      - REDUCE does NOT close the position — partial exits don't
        increment trade count (matches scalar's behavior where a
        position reduced 5× then fully closed = 1 trade, not 6).

    Edge case: if a ticker is re-entered after being fully exited, the
    new entry starts a new trade (per-ticker entry-price slot is reset
    on EXIT).

    None buffer → (0, 0). Used for combos that were released before
    stats compute, or for the empty-grid edge case.
    """
    if buf is None or len(buf) == 0:
        return 0, 0

    # Per-ticker slot for the most recent unclosed entry price.
    # int → float; default-absent means "no open position".
    open_entry_price: dict[int, float] = {}
    n_trades = 0
    n_wins = 0

    n = len(buf)
    for i in range(n):
        ac = buf.action_code[i]
        t_idx = buf.ticker_idx[i]
        price = buf.price[i]
        if ac == ACTION_ENTER:
            # Re-entry on an already-open position is unusual but
            # possible if the apply_buy semantics overwrite (per
            # vectorized_sim.py:284 docstring). Treat as new trade
            # opening — the prior unclosed entry is lost (no exit
            # event to score it).
            open_entry_price[t_idx] = price
        elif ac == ACTION_EXIT:
            entry_price = open_entry_price.pop(t_idx, None)
            if entry_price is not None:
                n_trades += 1
                if price > entry_price:
                    n_wins += 1
        # ACTION_REDUCE: skip — partial exits don't close the trade.

    return n_trades, n_wins


# ── Public entry point ──────────────────────────────────────────────────────


def compute_vectorized_stats(
    *,
    nav_history: np.ndarray,
    init_cash: float,
    spy_prices: pd.Series | None,
    dates: pd.DatetimeIndex,
    orders_per_combo: VectorizedOrderStore,
    combo_params: Iterable[dict],
) -> pd.DataFrame:
    """Compute per-combo stats from NAV trajectory + columnar orders.

    Replaces the per-combo `vectorbt.Portfolio.from_orders` +
    `compute_pf_stats` chain in the vectorized predictor sweep path.

    Parameters
    ----------
    nav_history : np.ndarray, shape [n_combos, n_dates]
        Per-combo mark-to-market NAV recorded each date by
        `run_vectorized_sweep`. NaN handling: this matrix should be
        clean (every (combo, date) has a real number); validated
        upstream by the sim's update_nav contract.
    init_cash : float
        Starting cash per combo. Used to compute total_return and to
        detect zero-activity combos (final_nav ≈ init_cash AND zero
        trades).
    spy_prices : pd.Series | None
        SPY close series indexed by date. None → spy_return /
        total_alpha set to None per combo.
    dates : pd.DatetimeIndex, length n_dates
        Date index aligned with nav_history's column axis.
    orders_per_combo : VectorizedOrderStore
        Columnar order buffers. Walked per combo to count trades +
        materialize per-combo `total_orders`. Buffers are inspected
        directly via `_buffers[i]` to avoid materializing dict-lists
        (the whole point of the columnar accumulator).
    combo_params : Iterable[dict]
        Per-combo input params (min_score, max_position_pct, etc.).
        Merged into each output row so callers can pivot on params.

    Returns
    -------
    pd.DataFrame, one row per combo, columns:
        - all keys from combo_params
        - status ("ok" | "no_orders")
        - total_orders, total_trades, win_rate
        - total_return, sharpe_ratio, sortino_ratio,
          max_drawdown, calmar_ratio, cvar_95
        - spy_return, total_alpha (None if no spy_prices)

    Schema is additive vs the prior vectorbt-path output: sortino_ratio
    + cvar_95 are new columns required by the evaluator-revamp grading
    rewire (see evaluator-revamp-260506.md). Existing consumers reading
    sharpe_ratio / max_drawdown / calmar_ratio see zero behavioral change.
    """
    n_combos, n_dates = nav_history.shape
    combo_params_list = list(combo_params)
    if len(combo_params_list) != n_combos:
        raise ValueError(
            f"combo_params has {len(combo_params_list)} entries but "
            f"nav_history is shape {nav_history.shape} — n_combos mismatch"
        )

    # All-combo vectorized stats (one numpy pass per metric).
    total_return = compute_total_return(nav_history, init_cash)
    daily_returns = compute_daily_returns(nav_history)
    sharpe = compute_sharpe_ratio(daily_returns)
    sortino = compute_sortino_ratio(daily_returns)
    cvar_95 = compute_cvar(daily_returns, q=0.05)
    max_drawdown = compute_max_drawdown(nav_history)
    calmar = compute_calmar_ratio(total_return, max_drawdown, n_dates)

    # SPY return over the portfolio's date range — single scalar shared
    # across all combos (every combo runs the same simulation window).
    spy_return = compute_spy_return(spy_prices, dates)
    if spy_return is not None:
        total_alpha_per_combo: list[float | None] = [
            float(tr - spy_return) for tr in total_return
        ]
        spy_value: float | None = float(spy_return)
    else:
        total_alpha_per_combo = [None] * n_combos
        spy_value = None

    # Per-combo trade counts + order counts. Walks columnar buffers
    # directly to avoid materializing dict-lists (60 combos × 26k orders
    # = 1.5M dicts would defeat the columnar refactor).
    rows: list[dict] = []
    for combo_idx, params in enumerate(combo_params_list):
        buf = orders_per_combo._buffers[combo_idx]
        n_orders = len(buf) if buf is not None else 0
        n_trades, n_wins = count_trades_from_buffer(buf)
        win_rate = (n_wins / n_trades) if n_trades > 0 else 0.0

        if n_orders == 0:
            # No-orders combo: zero out trade-derived metrics.
            # NAV-derived metrics are 0 already (NAV stays at init_cash).
            row = {
                **params,
                "status": "no_orders",
                "total_orders": 0,
                "total_trades": 0,
                "win_rate": 0.0,
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "calmar_ratio": 0.0,
                "cvar_95": 0.0,
                "spy_return": spy_value,
                "total_alpha": (
                    -spy_value if spy_value is not None else None
                ),
            }
        else:
            row = {
                **params,
                "status": "ok",
                "total_orders": n_orders,
                "total_trades": n_trades,
                "win_rate": win_rate,
                "total_return": float(total_return[combo_idx]),
                "sharpe_ratio": float(sharpe[combo_idx]),
                "sortino_ratio": float(sortino[combo_idx]),
                "max_drawdown": float(max_drawdown[combo_idx]),
                "calmar_ratio": float(calmar[combo_idx]),
                "cvar_95": float(cvar_95[combo_idx]),
                "spy_return": spy_value,
                "total_alpha": total_alpha_per_combo[combo_idx],
            }
        rows.append(row)

        # Release the buffer now that stats are computed for this combo.
        # Bounds peak memory to ~1 buffer at a time during the trade
        # counting walk above (each ~1.5 MB) instead of all 60 combos.
        orders_per_combo.release(combo_idx)

    return pd.DataFrame(rows)
