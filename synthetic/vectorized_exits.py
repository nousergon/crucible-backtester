"""Vectorized strategy exit decisions (Tier 4 PR 2, 2026-04-27).

Implements the 4 exit checks from
``executor.strategies.exit_manager.evaluate_exits`` as matrix ops on
``VectorizedSimulator`` state. All combos × all tickers evaluate
simultaneously per date — no per-combo Python loop.

Cascade (per ``evaluate_exits``):

  1. ATR trailing stop with sector-relative veto
       - Triggered when current price <= highest_high - ATR × multiplier
       - Vetoed if the stock's lookback return outperforms its sector
         ETF's by more than ``sector_relative_outperform_threshold``
         (the veto preserves momentum-positive holdings)
  2. Fallback fixed-percentage stop
       - Runs ONLY when ATR did NOT raise a raw signal (matches the
         scalar ``elif fallback_enabled`` branch — vetoed-ATR positions
         do NOT enter fallback, they fall through to profit/momentum/time)
  3. Profit-take (REDUCE)
       - Triggers when (price - avg_cost) / avg_cost >= profit_take_pct
  4. Momentum exit (EXIT)
       - 20d momentum < threshold AND RSI < oversold cutoff
  5. Time decay (REDUCE then EXIT)
       - Days held >= reduce → REDUCE
       - Days held >= exit → EXIT (overrides the REDUCE branch)
       - ONLY when research action is HOLD (not ENTER/EXIT/REDUCE)

Strategy exits are SKIPPED entirely when ``research_action`` is EXIT or
REDUCE — research is already exiting the position. Research-driven
exits flow through a separate path (``decide_exits_and_reduces`` →
PR 4 wiring).

Parity contract
---------------
For ``n_combos == 1``, this module's per-(combo, ticker) decision
matches the scalar ``evaluate_exits`` byte-for-byte (modulo float
precision in feature lookups, which are themselves byte-equal modulo
Wilder smoothing seed convergence).

Time-day approximation
----------------------
Scalar ``_approx_trading_days`` walks calendar days and counts
weekdays. The simulator's date axis is per-signal-date (typically
trading days), so we compute ``held_days = date_idx - entry_dates``
directly. For sims where the date axis IS trading days (the standard
predictor_param_sweep config), this is byte-equal to the scalar
weekday-walk. Holidays embedded in the scalar's calendar are not
modeled by the simulator's date axis either; both paths share the
same approximation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# Action codes (stored in exit_action matrix as int8).
ACTION_NONE = 0
ACTION_EXIT = 1
ACTION_REDUCE = 2

# Reason codes (stored in exit_reason matrix as int8).
REASON_NONE = 0
REASON_ATR = 1
REASON_FALLBACK = 2
REASON_PROFIT = 3
REASON_MOMENTUM = 4
REASON_TIME_EXIT = 5
REASON_TIME_REDUCE = 6
REASON_LOSS_FLOOR = 7  # MAE hard floor (L4549a #238) — stance-agnostic full EXIT

# Research action codes — caller translates string signals to these.
RA_HOLD = 0
RA_ENTER = 1
RA_EXIT = 2
RA_REDUCE = 3


@dataclass(frozen=True)
class VectorizedExitConfig:
    """Per-combo exit-strategy parameter arrays.

    Each field is shape ``[n_combos]``. Building one of these is the
    flatten step that runs ONCE at sweep setup, translating each combo's
    ``strategy_config`` dict to numpy arrays.

    Disabled checks: when an ``*_enabled`` array entry is False, that
    combo's check is no-op (no decision raised). The corresponding
    threshold values are still read but their result is masked out.
    """

    # Enables (bool[n_combos])
    atr_trailing_enabled: np.ndarray
    fallback_stop_enabled: np.ndarray
    profit_take_enabled: np.ndarray
    momentum_exit_enabled: np.ndarray
    time_decay_enabled: np.ndarray
    sector_relative_veto_enabled: np.ndarray
    position_loss_floor_enabled: np.ndarray  # MAE floor (L4549a)

    # MAE hard floor: full EXIT when price/avg_cost - 1 <= pct (a NEGATIVE
    # decimal, e.g. -0.15). Stance-agnostic, highest precedence (the
    # falling-knife backstop — see the override block in compute_*).
    position_loss_floor_pct: np.ndarray

    # ATR trailing stop multiplier (price - highest_high - ATR × mult)
    atr_multiplier: np.ndarray  # float64[n_combos]

    # Fallback fixed-percentage stop (entry × (1 - pct))
    fallback_stop_pct: np.ndarray

    # Profit-take threshold (unrealized gain %)
    profit_take_pct: np.ndarray

    # Momentum exit thresholds
    momentum_exit_threshold: np.ndarray  # 20d momentum % cutoff (e.g. -15.0)
    momentum_exit_rsi: np.ndarray         # RSI oversold cutoff (e.g. 30)

    # Time decay (in trading days)
    time_decay_reduce_days: np.ndarray    # int[n_combos]
    time_decay_exit_days: np.ndarray

    # Sector-relative veto threshold (outperformance fraction, e.g. 0.05)
    sector_relative_outperform_threshold: np.ndarray

    # REDUCE share fraction (e.g. 0.50 → sell 50% of held)
    reduce_fraction: np.ndarray

    @property
    def n_combos(self) -> int:
        return int(self.atr_multiplier.shape[0])

    @classmethod
    def from_uniform(cls, n_combos: int, **overrides) -> "VectorizedExitConfig":
        """Build a config with all combos sharing the same scalar values.

        Defaults match ``executor.strategies.config`` defaults. Pass
        ``**overrides`` to change any field; per-combo arrays via direct
        construction for actual sweeps.
        """
        defaults = {
            "atr_trailing_enabled": True,
            "fallback_stop_enabled": True,
            "profit_take_enabled": True,
            "momentum_exit_enabled": True,
            "time_decay_enabled": True,
            "sector_relative_veto_enabled": True,
            "position_loss_floor_enabled": True,
            "position_loss_floor_pct": -0.15,
            "atr_multiplier": 3.0,
            "fallback_stop_pct": 0.10,
            "profit_take_pct": 0.25,
            "momentum_exit_threshold": -15.0,
            "momentum_exit_rsi": 30.0,
            "time_decay_reduce_days": 5,
            "time_decay_exit_days": 10,
            "sector_relative_outperform_threshold": 0.05,
            "reduce_fraction": 0.50,
        }
        defaults.update(overrides)

        bool_fields = {
            "atr_trailing_enabled", "fallback_stop_enabled",
            "profit_take_enabled", "momentum_exit_enabled",
            "time_decay_enabled", "sector_relative_veto_enabled",
            "position_loss_floor_enabled",
        }
        int_fields = {"time_decay_reduce_days", "time_decay_exit_days"}

        kwargs = {}
        for k, v in defaults.items():
            if k in bool_fields:
                kwargs[k] = np.full(n_combos, bool(v), dtype=bool)
            elif k in int_fields:
                kwargs[k] = np.full(n_combos, int(v), dtype=np.int32)
            else:
                kwargs[k] = np.full(n_combos, float(v), dtype=np.float64)
        return cls(**kwargs)


@dataclass
class ExitDecisions:
    """Result of ``compute_vectorized_exits``.

    exit_action : int8[n_combos, n_tickers]
        ACTION_NONE / ACTION_EXIT / ACTION_REDUCE.
    exit_reason : int8[n_combos, n_tickers]
        REASON_* code corresponding to which gate fired.
    exit_shares : float64[n_combos, n_tickers]
        Number of shares to sell. Equals ``positions[c,t]`` for EXIT,
        ``floor(positions[c,t] * reduce_fraction[c])`` for REDUCE,
        0 otherwise.
    """

    exit_action: np.ndarray
    exit_reason: np.ndarray
    exit_shares: np.ndarray


def compute_vectorized_exits(
    sim,
    *,
    prices: np.ndarray,
    atr_dollar_at_date: np.ndarray,
    rsi_at_date: np.ndarray,
    momentum_at_date: np.ndarray,
    sector_lookback_return: np.ndarray,
    research_action_per_ticker: np.ndarray,
    sector_idx_per_ticker: np.ndarray,
    sector_etf_ticker_idx: np.ndarray,
    date_idx: int,
    config: VectorizedExitConfig,
) -> ExitDecisions:
    """Compute per-(combo, ticker) exit decisions as a matrix.

    Parameters
    ----------
    sim : VectorizedSimulator
        Reads ``positions``, ``avg_costs``, ``entry_dates``,
        ``highest_high``. Not mutated — apply via
        ``apply_vectorized_exits`` after.
    prices : float64[n_tickers]
        Current close prices, in the simulator's ticker-index order.
    atr_dollar_at_date : float64[n_tickers]
        Wilder ATR(14) at this date in dollar units. NaN where no data
        — those (combo, ticker) cells skip the ATR check and become
        eligible for the fallback stop branch.
    rsi_at_date : float64[n_tickers]
        Wilder RSI(14) at this date. NaN → momentum check skipped.
    momentum_at_date : float64[n_tickers]
        20-day percentage momentum at this date. NaN → momentum check
        skipped.
    sector_lookback_return : float64[n_tickers]
        Lookback return (default 20-bar) used by the sector-relative
        veto. NaN → veto cannot fire for that ticker. The caller must
        compute this against the same lookback as the scalar reference
        (``min(20, len(stock_history), len(etf_history))``); for the
        sweep we pin lookback=20 since all sweep dates have ≥20 bars
        of history available.
    research_action_per_ticker : int8/int32[n_tickers]
        Encoded research signal per ticker. ``RA_HOLD`` is the default
        (no signal); ``RA_EXIT`` and ``RA_REDUCE`` cause strategy
        checks to be skipped (research is already exiting).
    sector_idx_per_ticker : int32[n_tickers]
        Sector index per ticker. ``-1`` for unknown sector.
    sector_etf_ticker_idx : int32[n_sectors]
        For each sector index, the simulator-column-index of that
        sector's ETF (XLK for Tech, etc.). ``-1`` if no ETF mapped —
        veto cannot fire when this is the case.
    date_idx : int
        Current date index in the simulator's date axis (used for
        time-decay days-held arithmetic).
    config : VectorizedExitConfig
        Per-combo strategy parameters as numpy arrays.

    Returns
    -------
    ExitDecisions
    """
    held = sim.held_mask()  # bool[n_combos, n_tickers]
    n_combos, n_tickers = sim.positions.shape

    # Validate input shapes — fail loudly on wiring bugs rather than
    # silently broadcasting an off-shape vector.
    for name, arr in (
        ("prices", prices),
        ("atr_dollar_at_date", atr_dollar_at_date),
        ("rsi_at_date", rsi_at_date),
        ("momentum_at_date", momentum_at_date),
        ("sector_lookback_return", sector_lookback_return),
        ("research_action_per_ticker", research_action_per_ticker),
        ("sector_idx_per_ticker", sector_idx_per_ticker),
    ):
        if arr.shape != (n_tickers,):
            raise ValueError(
                f"{name} shape mismatch: expected ({n_tickers},), "
                f"got {arr.shape}"
            )

    # Eligibility: held AND research not in (EXIT, REDUCE).
    research_blocking = (
        (research_action_per_ticker == RA_EXIT)
        | (research_action_per_ticker == RA_REDUCE)
    )  # [n_tickers]
    eligible = held & ~research_blocking[None, :]  # [n_combos, n_tickers]

    exit_action = np.zeros((n_combos, n_tickers), dtype=np.int8)
    exit_reason = np.zeros((n_combos, n_tickers), dtype=np.int8)

    # ── 1. ATR trailing stop (with sector-relative veto) ────────────
    atr_data_ok = ~np.isnan(atr_dollar_at_date)  # [n_tickers]
    # stop_level[c, t] = highest_high[c, t] - atr_dollar[t] × multiplier[c]
    # NaN ATR will yield NaN stop_level, which yields False for the
    # `prices <= stop_level` comparison — masked out separately for
    # clarity via atr_data_ok.
    stop_level = (
        sim.highest_high
        - atr_dollar_at_date[None, :] * config.atr_multiplier[:, None]
    )  # [n_combos, n_tickers]
    atr_raw_triggered = (
        eligible
        & config.atr_trailing_enabled[:, None]
        & atr_data_ok[None, :]
        & (prices[None, :] <= stop_level)
    )  # [n_combos, n_tickers]

    # Sector-relative outperformance per ticker.
    # etf_idx_per_ticker[t] = sector_etf_ticker_idx[sector_idx_per_ticker[t]]
    # Where sector_idx == -1, etf_idx is undefined → treat as no veto possible.
    sector_known = sector_idx_per_ticker >= 0
    safe_sector_idx = np.where(sector_known, sector_idx_per_ticker, 0)
    etf_idx_per_ticker = sector_etf_ticker_idx[safe_sector_idx]
    etf_known = sector_known & (etf_idx_per_ticker >= 0)
    safe_etf_idx = np.where(etf_known, etf_idx_per_ticker, 0)
    sector_returns = sector_lookback_return[safe_etf_idx]
    outperformance = sector_lookback_return - sector_returns
    # Veto when outperformance > threshold[c]. Disabled / NaN / unknown
    # sector → no veto.
    outperf_finite = ~np.isnan(outperformance) & etf_known
    veto = (
        config.sector_relative_veto_enabled[:, None]
        & outperf_finite[None, :]
        & (outperformance[None, :] > config.sector_relative_outperform_threshold[:, None])
    )  # [n_combos, n_tickers]

    atr_after_veto = atr_raw_triggered & ~veto
    exit_action = np.where(atr_after_veto, ACTION_EXIT, exit_action).astype(np.int8)
    exit_reason = np.where(atr_after_veto, REASON_ATR, exit_reason).astype(np.int8)

    # ── 2. Fallback fixed-percentage stop ───────────────────────────
    # Scalar ``elif fallback_enabled`` runs ONLY when ATR raw didn't
    # trigger (vetoed positions skip fallback and fall through to
    # profit/momentum/time).
    fallback_eligible = (
        eligible
        & ~atr_raw_triggered
        & config.fallback_stop_enabled[:, None]
        & (sim.avg_costs > 0)
    )
    fallback_stop_level = sim.avg_costs * (1.0 - config.fallback_stop_pct[:, None])
    fallback_triggered = fallback_eligible & (prices[None, :] <= fallback_stop_level)
    # Only assign where exit_action is still NONE (defensive — atr_after_veto
    # and fallback_triggered are mutually exclusive by construction).
    fallback_assign = fallback_triggered & (exit_action == ACTION_NONE)
    exit_action = np.where(fallback_assign, ACTION_EXIT, exit_action).astype(np.int8)
    exit_reason = np.where(fallback_assign, REASON_FALLBACK, exit_reason).astype(np.int8)

    # ── 3. Profit-take (REDUCE) ─────────────────────────────────────
    cost_positive = sim.avg_costs > 0
    safe_cost = np.where(cost_positive, sim.avg_costs, 1.0)
    unrealized = (prices[None, :] - sim.avg_costs) / safe_cost
    profit_triggered = (
        eligible
        & (exit_action == ACTION_NONE)
        & config.profit_take_enabled[:, None]
        & cost_positive
        & (unrealized >= config.profit_take_pct[:, None])
    )
    exit_action = np.where(profit_triggered, ACTION_REDUCE, exit_action).astype(np.int8)
    exit_reason = np.where(profit_triggered, REASON_PROFIT, exit_reason).astype(np.int8)

    # ── 4. Momentum exit (EXIT) ─────────────────────────────────────
    mom_data_ok = ~np.isnan(momentum_at_date) & ~np.isnan(rsi_at_date)
    momentum_triggered = (
        eligible
        & (exit_action == ACTION_NONE)
        & config.momentum_exit_enabled[:, None]
        & mom_data_ok[None, :]
        & (momentum_at_date[None, :] < config.momentum_exit_threshold[:, None])
        & (rsi_at_date[None, :] < config.momentum_exit_rsi[:, None])
    )
    exit_action = np.where(momentum_triggered, ACTION_EXIT, exit_action).astype(np.int8)
    exit_reason = np.where(momentum_triggered, REASON_MOMENTUM, exit_reason).astype(np.int8)

    # ── 5. Time decay (REDUCE then EXIT) ────────────────────────────
    research_is_hold = (research_action_per_ticker == RA_HOLD)
    days_held = date_idx - sim.entry_dates  # [n_combos, n_tickers]
    time_eligible = (
        eligible
        & (exit_action == ACTION_NONE)
        & config.time_decay_enabled[:, None]
        & research_is_hold[None, :]
    )
    time_exit = (
        time_eligible
        & (days_held >= config.time_decay_exit_days[:, None])
    )
    time_reduce = (
        time_eligible
        & ~time_exit
        & (days_held >= config.time_decay_reduce_days[:, None])
    )
    exit_action = np.where(time_exit, ACTION_EXIT, exit_action).astype(np.int8)
    exit_reason = np.where(time_exit, REASON_TIME_EXIT, exit_reason).astype(np.int8)
    exit_action = np.where(time_reduce, ACTION_REDUCE, exit_action).astype(np.int8)
    exit_reason = np.where(time_reduce, REASON_TIME_REDUCE, exit_reason).astype(np.int8)

    # ── 0. Position loss floor (MAE) — HARD, stance-agnostic, HIGHEST ───
    # precedence. Scalar ``evaluate_exits`` runs this FIRST (step 0 of
    # ``_evaluate_single_position``): a held position whose loss from avg cost
    # breaches ``position_loss_floor_pct`` is a full EXIT regardless of
    # stance / catalyst / sector-veto / any price-based gate — the
    # falling-knife backstop (L4549a #238). Applied here as an UNCONDITIONAL
    # override (not gated on ``exit_action == NONE``) so it STOMPS whatever
    # ATR/fallback/profit/momentum/time decided for the cell — matching the
    # scalar precedence (floor wins) and reason. Gated on ``eligible`` (NOT
    # raw ``held``) for parity: scalar skips strategy checks, the floor
    # included, for research-EXIT/REDUCE names (``exit_manager.py`` L818).
    floor_cost_ok = sim.avg_costs > 0
    safe_cost_floor = np.where(floor_cost_ok, sim.avg_costs, 1.0)
    loss_frac = prices[None, :] / safe_cost_floor - 1.0  # [n_combos, n_tickers]
    loss_floor_triggered = (
        eligible
        & config.position_loss_floor_enabled[:, None]
        & floor_cost_ok
        & ~np.isnan(prices)[None, :]
        & (loss_frac <= config.position_loss_floor_pct[:, None])
    )
    exit_action = np.where(loss_floor_triggered, ACTION_EXIT, exit_action).astype(np.int8)
    exit_reason = np.where(loss_floor_triggered, REASON_LOSS_FLOOR, exit_reason).astype(np.int8)

    # ── Compute share counts ────────────────────────────────────────
    # EXIT: full position. REDUCE: floor(shares * reduce_frac[c]).
    reduce_shares = np.floor(sim.positions * config.reduce_fraction[:, None])
    exit_shares = np.where(
        exit_action == ACTION_EXIT,
        sim.positions,
        np.where(exit_action == ACTION_REDUCE, reduce_shares, 0.0),
    )

    # Demote REDUCE → NONE when shares round to 0 (matches scalar
    # "SKIP REDUCE — position too small to reduce").
    reduce_to_zero = (exit_action == ACTION_REDUCE) & (exit_shares == 0)
    exit_action = np.where(reduce_to_zero, ACTION_NONE, exit_action).astype(np.int8)
    exit_reason = np.where(reduce_to_zero, REASON_NONE, exit_reason).astype(np.int8)

    return ExitDecisions(
        exit_action=exit_action,
        exit_reason=exit_reason,
        exit_shares=exit_shares,
    )


def apply_vectorized_exits(
    sim,
    decisions: ExitDecisions,
    prices: np.ndarray,
) -> int:
    """Apply exit decisions to ``sim`` via ``sim.apply_sell``.

    Returns the number of (combo, ticker) cells where an exit was
    applied. Caller can use the count for telemetry / order-recording.

    Mutates ``sim.positions``, ``sim.cash``, ``sim.avg_costs``,
    ``sim.entry_dates``, ``sim.highest_high`` via apply_sell.
    """
    nonzero = decisions.exit_action != ACTION_NONE
    if not np.any(nonzero):
        return 0
    combo_idx, ticker_idx = np.nonzero(nonzero)
    shares = decisions.exit_shares[combo_idx, ticker_idx]
    px = prices[ticker_idx]
    sim.apply_sell(
        combo_idx.astype(np.int64),
        ticker_idx.astype(np.int64),
        shares,
        px,
    )
    return int(combo_idx.size)
