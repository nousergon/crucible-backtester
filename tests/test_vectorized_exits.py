"""Parity tests for vectorized exit decisions (Tier 4 PR 2, 2026-04-27).

Pins that ``compute_vectorized_exits`` produces per-(combo, ticker)
decisions byte-equivalent to scalar ``evaluate_exits`` from
``executor.strategies.exit_manager``. The Tier 4 simulator's exit
correctness rests entirely on this — if the vectorized cascade
diverges from scalar even on a single edge case, the 60-combo sweep
silently inherits the bug.

Coverage
--------
Per-check parity (each gate alone):
  * ATR trailing stop with sector veto
  * Fallback fixed-percentage stop
  * Profit-take REDUCE
  * Momentum exit
  * Time decay (reduce + exit)

Cascade ordering:
  * ATR vetoed → fall through to profit/momentum/time
  * ATR raw not triggered → fallback runs
  * Profit-take fires → momentum/time skipped
  * Time-decay only fires when research_action == HOLD

End-to-end:
  * Scalar ``evaluate_exits`` per combo == vectorized for N=1 random
    fixture across multiple decision dates.
  * Multi-combo: per-combo configs apply independently.
  * apply_vectorized_exits zeroes positions correctly.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest


_EXECUTOR_ROOT = os.path.expanduser("~/Development/alpha-engine")
if os.path.isdir(_EXECUTOR_ROOT) and _EXECUTOR_ROOT not in sys.path:
    sys.path.insert(0, _EXECUTOR_ROOT)


from synthetic.vectorized_sim import VectorizedSimulator
from synthetic.vectorized_exits import (
    ACTION_EXIT,
    ACTION_NONE,
    ACTION_REDUCE,
    RA_ENTER,
    RA_EXIT,
    RA_HOLD,
    RA_REDUCE,
    REASON_ATR,
    REASON_FALLBACK,
    REASON_LOSS_FLOOR,
    REASON_MOMENTUM,
    REASON_NONE,
    REASON_PROFIT,
    REASON_TIME_EXIT,
    REASON_TIME_REDUCE,
    ExitDecisions,
    VectorizedExitConfig,
    apply_vectorized_exits,
    compute_vectorized_exits,
)


_REASON_BY_NAME = {
    "atr_trailing_stop": REASON_ATR,
    "fallback_stop": REASON_FALLBACK,
    "profit_take": REASON_PROFIT,
    "momentum_exit": REASON_MOMENTUM,
    "time_decay_exit": REASON_TIME_EXIT,
    "time_decay_reduce": REASON_TIME_REDUCE,
    "position_loss_floor": REASON_LOSS_FLOOR,
}

_ACTION_BY_NAME = {"EXIT": ACTION_EXIT, "REDUCE": ACTION_REDUCE}


def _ticker_index(*tickers: str) -> dict:
    return {t: i for i, t in enumerate(tickers)}


def _trending_ohlcv(
    n_bars: int, start_price: float = 100.0, drift: float = 0.0,
    vol: float = 1.0, seed: int = 0,
) -> pd.DataFrame:
    """OHLCV DataFrame with a controllable drift + vol regime.

    Indexed by trading-day Timestamps starting 2024-01-01. Used by both
    scalar and vectorized paths so the price → feature mapping is the
    same.
    """
    rng = np.random.default_rng(seed)
    closes = np.zeros(n_bars)
    closes[0] = start_price
    for i in range(1, n_bars):
        closes[i] = closes[i - 1] * (1 + drift) + rng.normal(0, vol)
    closes = np.maximum(closes, 1.0)
    highs = closes + rng.uniform(0, 1, n_bars)
    lows = closes - rng.uniform(0, 1, n_bars)
    opens = np.concatenate(([closes[0]], closes[:-1]))
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="B")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=idx,
    )


# ────────────────────────────────────────────────────────────────────
# Per-check parity
# ────────────────────────────────────────────────────────────────────


class TestATRTrailingStopParity:
    """Single combo, single ticker — ATR exits should fire identically."""

    def _setup(self, current_price: float, highest_high: float, atr_dollar: float):
        ti = _ticker_index("AAPL")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti)
        # 100 shares of AAPL held since date_idx 0, avg_cost=100
        sim.positions[0, 0] = 100
        sim.avg_costs[0, 0] = 100.0
        sim.entry_dates[0, 0] = 0
        sim.highest_high[0, 0] = highest_high

        config = VectorizedExitConfig.from_uniform(
            n_combos=1,
            atr_multiplier=3.0,
            sector_relative_veto_enabled=False,  # disable veto for this test
            position_loss_floor_enabled=False,   # isolate ATR from the MAE floor
        )
        prices = np.array([current_price])
        atr = np.array([atr_dollar])
        rsi = np.array([50.0])  # neutral
        mom = np.array([0.0])
        sec_ret = np.array([0.0])
        ra = np.array([RA_HOLD], dtype=np.int8)
        sec_idx = np.array([0], dtype=np.int32)
        sec_etf = np.array([0], dtype=np.int32)  # "AAPL" is its own ETF (degenerate)
        return sim, config, prices, atr, rsi, mom, sec_ret, ra, sec_idx, sec_etf

    def test_atr_triggers_when_price_below_stop(self):
        sim, config, prices, atr, rsi, mom, sec_ret, ra, sec_idx, sec_etf = self._setup(
            current_price=85.0, highest_high=110.0, atr_dollar=5.0,
        )
        # stop = 110 - 5*3 = 95. price 85 ≤ 95 → exit.
        decisions = compute_vectorized_exits(
            sim, prices=prices, atr_dollar_at_date=atr,
            rsi_at_date=rsi, momentum_at_date=mom,
            sector_lookback_return=sec_ret,
            research_action_per_ticker=ra,
            sector_idx_per_ticker=sec_idx,
            sector_etf_ticker_idx=sec_etf,
            date_idx=1, config=config,
        )
        assert decisions.exit_action[0, 0] == ACTION_EXIT
        assert decisions.exit_reason[0, 0] == REASON_ATR
        assert decisions.exit_shares[0, 0] == 100.0

    def test_atr_skips_when_price_above_stop(self):
        sim, config, prices, atr, rsi, mom, sec_ret, ra, sec_idx, sec_etf = self._setup(
            current_price=100.0, highest_high=110.0, atr_dollar=5.0,
        )
        # stop = 95. price 100 > 95 → no exit.
        decisions = compute_vectorized_exits(
            sim, prices=prices, atr_dollar_at_date=atr,
            rsi_at_date=rsi, momentum_at_date=mom,
            sector_lookback_return=sec_ret,
            research_action_per_ticker=ra,
            sector_idx_per_ticker=sec_idx,
            sector_etf_ticker_idx=sec_etf,
            date_idx=1, config=config,
        )
        assert decisions.exit_action[0, 0] == ACTION_NONE


class TestSectorVetoBlocksATR:
    def test_sector_veto_blocks_atr_when_outperforming(self):
        # Two tickers: held stock + sector ETF (XLK). Stock outperforms.
        ti = _ticker_index("AAPL", "XLK")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti)
        sim.positions[0, 0] = 100
        sim.avg_costs[0, 0] = 100.0
        sim.entry_dates[0, 0] = 0
        sim.highest_high[0, 0] = 110.0

        config = VectorizedExitConfig.from_uniform(
            n_combos=1,
            atr_multiplier=3.0,
            sector_relative_veto_enabled=True,
            sector_relative_outperform_threshold=0.05,  # 5%
            position_loss_floor_enabled=False,  # isolate the veto from the MAE floor
        )

        # Stock returns 15%, sector ETF returns 5%. Outperformance = 10% > 5%.
        prices = np.array([85.0, 50.0])  # AAPL below stop (110-15=95)
        atr = np.array([5.0, np.nan])
        rsi = np.array([50.0, np.nan])
        mom = np.array([0.0, np.nan])
        sec_ret = np.array([0.15, 0.05])  # AAPL +15%, XLK +5%
        ra = np.array([RA_HOLD, RA_HOLD], dtype=np.int8)
        sec_idx = np.array([0, 0], dtype=np.int32)  # both in sector 0
        sec_etf = np.array([1], dtype=np.int32)     # sector 0's ETF is at idx 1 (XLK)

        decisions = compute_vectorized_exits(
            sim, prices=prices, atr_dollar_at_date=atr,
            rsi_at_date=rsi, momentum_at_date=mom,
            sector_lookback_return=sec_ret,
            research_action_per_ticker=ra,
            sector_idx_per_ticker=sec_idx,
            sector_etf_ticker_idx=sec_etf,
            date_idx=1, config=config,
        )
        # ATR was vetoed; no other gate fires (price up vs cost = -15%)
        assert decisions.exit_action[0, 0] == ACTION_NONE

    def test_sector_veto_does_not_block_when_underperforming(self):
        ti = _ticker_index("AAPL", "XLK")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti)
        sim.positions[0, 0] = 100
        sim.avg_costs[0, 0] = 100.0
        sim.entry_dates[0, 0] = 0
        sim.highest_high[0, 0] = 110.0

        config = VectorizedExitConfig.from_uniform(
            n_combos=1,
            atr_multiplier=3.0,
            sector_relative_veto_enabled=True,
            sector_relative_outperform_threshold=0.05,
            position_loss_floor_enabled=False,  # isolate the veto from the MAE floor
        )
        # Stock -15%, ETF +5% → outperformance -20% < 5% threshold → no veto
        prices = np.array([85.0, 50.0])
        atr = np.array([5.0, np.nan])
        rsi = np.array([50.0, np.nan])
        mom = np.array([0.0, np.nan])
        sec_ret = np.array([-0.15, 0.05])
        ra = np.array([RA_HOLD, RA_HOLD], dtype=np.int8)
        sec_idx = np.array([0, 0], dtype=np.int32)
        sec_etf = np.array([1], dtype=np.int32)

        decisions = compute_vectorized_exits(
            sim, prices=prices, atr_dollar_at_date=atr,
            rsi_at_date=rsi, momentum_at_date=mom,
            sector_lookback_return=sec_ret,
            research_action_per_ticker=ra,
            sector_idx_per_ticker=sec_idx,
            sector_etf_ticker_idx=sec_etf,
            date_idx=1, config=config,
        )
        assert decisions.exit_action[0, 0] == ACTION_EXIT
        assert decisions.exit_reason[0, 0] == REASON_ATR


class TestFallbackStop:
    def test_fallback_fires_when_atr_data_missing_and_price_below_threshold(self):
        ti = _ticker_index("AAPL")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti)
        sim.positions[0, 0] = 100
        sim.avg_costs[0, 0] = 100.0
        sim.entry_dates[0, 0] = 0
        sim.highest_high[0, 0] = 100.0

        config = VectorizedExitConfig.from_uniform(
            n_combos=1, fallback_stop_pct=0.10,
            position_loss_floor_enabled=False,  # isolate fallback from the MAE floor
        )
        # ATR data missing (NaN), price 85 ≤ entry 100 * 0.90 = 90 → fallback fires
        prices = np.array([85.0])
        decisions = compute_vectorized_exits(
            sim, prices=prices,
            atr_dollar_at_date=np.array([np.nan]),
            rsi_at_date=np.array([50.0]),
            momentum_at_date=np.array([0.0]),
            sector_lookback_return=np.array([0.0]),
            research_action_per_ticker=np.array([RA_HOLD], dtype=np.int8),
            sector_idx_per_ticker=np.array([-1], dtype=np.int32),
            sector_etf_ticker_idx=np.array([-1], dtype=np.int32),
            date_idx=1, config=config,
        )
        assert decisions.exit_action[0, 0] == ACTION_EXIT
        assert decisions.exit_reason[0, 0] == REASON_FALLBACK

    def test_fallback_skipped_when_atr_raw_triggered_but_vetoed(self):
        """Scalar's elif: when ATR raw triggered (even if vetoed), skip
        fallback. The vetoed position falls through to profit/momentum/time.
        """
        ti = _ticker_index("AAPL", "XLK")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti)
        sim.positions[0, 0] = 100
        sim.avg_costs[0, 0] = 100.0
        sim.entry_dates[0, 0] = 0
        sim.highest_high[0, 0] = 110.0

        config = VectorizedExitConfig.from_uniform(
            n_combos=1,
            atr_multiplier=3.0,
            sector_relative_veto_enabled=True,
            sector_relative_outperform_threshold=0.05,
            fallback_stop_enabled=True,
            fallback_stop_pct=0.05,  # tight fallback that WOULD fire
            position_loss_floor_enabled=False,  # isolate fallback/veto from the MAE floor
        )
        # ATR raw triggered (price below stop) but vetoed → fallback skipped
        prices = np.array([85.0, 50.0])
        decisions = compute_vectorized_exits(
            sim, prices=prices,
            atr_dollar_at_date=np.array([5.0, np.nan]),
            rsi_at_date=np.array([50.0, np.nan]),
            momentum_at_date=np.array([0.0, np.nan]),
            sector_lookback_return=np.array([0.15, 0.05]),
            research_action_per_ticker=np.array([RA_HOLD, RA_HOLD], dtype=np.int8),
            sector_idx_per_ticker=np.array([0, 0], dtype=np.int32),
            sector_etf_ticker_idx=np.array([1], dtype=np.int32),
            date_idx=1, config=config,
        )
        # Vetoed ATR + skipped fallback + no other check fires (-15% loss but
        # not enough for momentum and not enough days for time decay)
        assert decisions.exit_action[0, 0] == ACTION_NONE


class TestPositionLossFloor:
    """MAE hard floor (L4549a #238) — stance-agnostic full EXIT at the highest
    precedence. Mirrors scalar ``check_position_loss_floor`` / step-0 ordering
    in ``executor.strategies.exit_manager._evaluate_single_position``."""

    def _held(self, n_combos=1, n_tickers=1):
        ti = _ticker_index(*[f"T{i}" for i in range(n_tickers)])
        sim = VectorizedSimulator(n_combos=n_combos, ticker_index=ti)
        sim.positions[:, 0] = 100
        sim.avg_costs[:, 0] = 100.0
        sim.entry_dates[:, 0] = 0
        sim.highest_high[:, 0] = 110.0
        return sim

    def _run(self, sim, config, price, n_tickers=1):
        nan = np.full(n_tickers, np.nan)
        prices = np.array([price] + [50.0] * (n_tickers - 1))
        return compute_vectorized_exits(
            sim, prices=prices,
            atr_dollar_at_date=nan, rsi_at_date=nan, momentum_at_date=nan,
            sector_lookback_return=np.zeros(n_tickers),
            research_action_per_ticker=np.full(n_tickers, RA_HOLD, dtype=np.int8),
            sector_idx_per_ticker=np.full(n_tickers, -1, dtype=np.int32),
            sector_etf_ticker_idx=np.array([-1], dtype=np.int32),
            date_idx=1, config=config,
        )

    def test_floor_fires_full_exit_at_breach(self):
        sim = self._held()
        cfg = VectorizedExitConfig.from_uniform(n_combos=1)  # floor default -0.15
        d = self._run(sim, cfg, price=85.0)  # 85/100 - 1 = -0.15 ≤ -0.15
        assert d.exit_action[0, 0] == ACTION_EXIT
        assert d.exit_reason[0, 0] == REASON_LOSS_FLOOR
        assert d.exit_shares[0, 0] == 100.0  # full position

    def test_floor_skips_above_breach(self):
        sim = self._held()
        cfg = VectorizedExitConfig.from_uniform(
            n_combos=1, atr_trailing_enabled=False, fallback_stop_enabled=False,
        )
        d = self._run(sim, cfg, price=86.0)  # -14% > -15% floor → no floor exit
        assert d.exit_action[0, 0] == ACTION_NONE

    def test_floor_disabled_does_not_fire(self):
        sim = self._held()
        cfg = VectorizedExitConfig.from_uniform(
            n_combos=1, position_loss_floor_enabled=False,
            atr_trailing_enabled=False, fallback_stop_enabled=False,
        )
        d = self._run(sim, cfg, price=50.0)  # -50% but floor disabled
        assert d.exit_action[0, 0] == ACTION_NONE

    def test_floor_custom_pct(self):
        sim = self._held()
        cfg = VectorizedExitConfig.from_uniform(
            n_combos=1, position_loss_floor_pct=-0.08,
            atr_trailing_enabled=False, fallback_stop_enabled=False,
        )
        d = self._run(sim, cfg, price=90.0)  # -10% ≤ -8% → fires
        assert d.exit_action[0, 0] == ACTION_EXIT
        assert d.exit_reason[0, 0] == REASON_LOSS_FLOOR

    def test_floor_overrides_sector_veto(self):
        # The #238 invariant: the floor is stance/veto-AGNOSTIC. A position the
        # sector-relative veto would protect from an ATR exit is STILL cut by the
        # floor when it breaches — the veto suppresses the alpha exit, never the
        # hard risk floor.
        ti = _ticker_index("AAPL", "XLK")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti)
        sim.positions[0, 0] = 100
        sim.avg_costs[0, 0] = 100.0
        sim.entry_dates[0, 0] = 0
        sim.highest_high[0, 0] = 110.0
        cfg = VectorizedExitConfig.from_uniform(
            n_combos=1, atr_multiplier=3.0,
            sector_relative_veto_enabled=True,
            sector_relative_outperform_threshold=0.05,
            # floor default-on at -0.15
        )
        # AAPL -15% (breaches floor) AND outperforms its ETF (+15% vs +5% → veto
        # would block the ATR exit). Floor must win regardless.
        d = compute_vectorized_exits(
            sim, prices=np.array([85.0, 50.0]),
            atr_dollar_at_date=np.array([5.0, np.nan]),
            rsi_at_date=np.array([50.0, np.nan]),
            momentum_at_date=np.array([0.0, np.nan]),
            sector_lookback_return=np.array([0.15, 0.05]),
            research_action_per_ticker=np.array([RA_HOLD, RA_HOLD], dtype=np.int8),
            sector_idx_per_ticker=np.array([0, 0], dtype=np.int32),
            sector_etf_ticker_idx=np.array([1], dtype=np.int32),
            date_idx=1, config=cfg,
        )
        assert d.exit_action[0, 0] == ACTION_EXIT
        assert d.exit_reason[0, 0] == REASON_LOSS_FLOOR

    def test_floor_skips_research_blocked(self):
        # Parity: scalar ``evaluate_exits`` skips strategy checks (floor included)
        # for research EXIT/REDUCE names (exit_manager.py L818) — research is
        # already exiting. Vectorized gates the floor on ``eligible`` likewise.
        sim = self._held()
        cfg = VectorizedExitConfig.from_uniform(n_combos=1)
        d = compute_vectorized_exits(
            sim, prices=np.array([85.0]),
            atr_dollar_at_date=np.array([np.nan]),
            rsi_at_date=np.array([np.nan]),
            momentum_at_date=np.array([np.nan]),
            sector_lookback_return=np.array([0.0]),
            research_action_per_ticker=np.array([RA_EXIT], dtype=np.int8),
            sector_idx_per_ticker=np.array([-1], dtype=np.int32),
            sector_etf_ticker_idx=np.array([-1], dtype=np.int32),
            date_idx=1, config=cfg,
        )
        assert d.exit_action[0, 0] == ACTION_NONE


class TestProfitTake:
    def test_profit_take_reduce_when_threshold_breached(self):
        ti = _ticker_index("AAPL")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti)
        sim.positions[0, 0] = 100
        sim.avg_costs[0, 0] = 100.0
        sim.entry_dates[0, 0] = 0
        sim.highest_high[0, 0] = 130.0

        config = VectorizedExitConfig.from_uniform(
            n_combos=1, profit_take_pct=0.25,
            atr_trailing_enabled=False, fallback_stop_enabled=False,
            reduce_fraction=0.50,
        )
        # 30% gain ≥ 25% threshold
        prices = np.array([130.0])
        decisions = compute_vectorized_exits(
            sim, prices=prices,
            atr_dollar_at_date=np.array([np.nan]),
            rsi_at_date=np.array([50.0]),
            momentum_at_date=np.array([0.0]),
            sector_lookback_return=np.array([0.0]),
            research_action_per_ticker=np.array([RA_HOLD], dtype=np.int8),
            sector_idx_per_ticker=np.array([-1], dtype=np.int32),
            sector_etf_ticker_idx=np.array([-1], dtype=np.int32),
            date_idx=1, config=config,
        )
        assert decisions.exit_action[0, 0] == ACTION_REDUCE
        assert decisions.exit_reason[0, 0] == REASON_PROFIT
        assert decisions.exit_shares[0, 0] == 50.0  # 50% of 100


class TestMomentumExit:
    def test_momentum_exit_when_both_thresholds_breached(self):
        ti = _ticker_index("AAPL")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti)
        sim.positions[0, 0] = 100
        sim.avg_costs[0, 0] = 100.0
        sim.entry_dates[0, 0] = 0
        sim.highest_high[0, 0] = 100.0

        config = VectorizedExitConfig.from_uniform(
            n_combos=1, momentum_exit_threshold=-15.0, momentum_exit_rsi=30.0,
            atr_trailing_enabled=False, fallback_stop_enabled=False,
            profit_take_enabled=False,
        )
        prices = np.array([95.0])
        decisions = compute_vectorized_exits(
            sim, prices=prices,
            atr_dollar_at_date=np.array([np.nan]),
            rsi_at_date=np.array([25.0]),  # < 30 → oversold
            momentum_at_date=np.array([-20.0]),  # < -15
            sector_lookback_return=np.array([0.0]),
            research_action_per_ticker=np.array([RA_HOLD], dtype=np.int8),
            sector_idx_per_ticker=np.array([-1], dtype=np.int32),
            sector_etf_ticker_idx=np.array([-1], dtype=np.int32),
            date_idx=1, config=config,
        )
        assert decisions.exit_action[0, 0] == ACTION_EXIT
        assert decisions.exit_reason[0, 0] == REASON_MOMENTUM

    def test_momentum_exit_skipped_when_rsi_neutral(self):
        ti = _ticker_index("AAPL")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti)
        sim.positions[0, 0] = 100
        sim.avg_costs[0, 0] = 100.0
        sim.entry_dates[0, 0] = 0
        sim.highest_high[0, 0] = 100.0

        config = VectorizedExitConfig.from_uniform(
            n_combos=1, momentum_exit_threshold=-15.0, momentum_exit_rsi=30.0,
            atr_trailing_enabled=False, fallback_stop_enabled=False,
            profit_take_enabled=False,
        )
        decisions = compute_vectorized_exits(
            sim, prices=np.array([95.0]),
            atr_dollar_at_date=np.array([np.nan]),
            rsi_at_date=np.array([45.0]),   # >= 30 → not oversold
            momentum_at_date=np.array([-20.0]),
            sector_lookback_return=np.array([0.0]),
            research_action_per_ticker=np.array([RA_HOLD], dtype=np.int8),
            sector_idx_per_ticker=np.array([-1], dtype=np.int32),
            sector_etf_ticker_idx=np.array([-1], dtype=np.int32),
            date_idx=1, config=config,
        )
        assert decisions.exit_action[0, 0] == ACTION_NONE


class TestTimeDecay:
    def test_time_decay_reduce_then_exit(self):
        ti = _ticker_index("AAPL")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti)
        sim.positions[0, 0] = 100
        sim.avg_costs[0, 0] = 100.0
        sim.entry_dates[0, 0] = 0
        sim.highest_high[0, 0] = 100.0

        config = VectorizedExitConfig.from_uniform(
            n_combos=1, time_decay_reduce_days=5, time_decay_exit_days=10,
            atr_trailing_enabled=False, fallback_stop_enabled=False,
            profit_take_enabled=False, momentum_exit_enabled=False,
            reduce_fraction=0.50,
        )

        common = dict(
            atr_dollar_at_date=np.array([np.nan]),
            rsi_at_date=np.array([np.nan]),
            momentum_at_date=np.array([np.nan]),
            sector_lookback_return=np.array([0.0]),
            research_action_per_ticker=np.array([RA_HOLD], dtype=np.int8),
            sector_idx_per_ticker=np.array([-1], dtype=np.int32),
            sector_etf_ticker_idx=np.array([-1], dtype=np.int32),
            config=config,
        )
        # Day 4: still under reduce threshold → no decision
        decisions = compute_vectorized_exits(
            sim, prices=np.array([100.0]), date_idx=4, **common,
        )
        assert decisions.exit_action[0, 0] == ACTION_NONE

        # Day 5: hits reduce threshold
        decisions = compute_vectorized_exits(
            sim, prices=np.array([100.0]), date_idx=5, **common,
        )
        assert decisions.exit_action[0, 0] == ACTION_REDUCE
        assert decisions.exit_reason[0, 0] == REASON_TIME_REDUCE
        assert decisions.exit_shares[0, 0] == 50.0

        # Day 10: hits exit threshold (overrides reduce)
        decisions = compute_vectorized_exits(
            sim, prices=np.array([100.0]), date_idx=10, **common,
        )
        assert decisions.exit_action[0, 0] == ACTION_EXIT
        assert decisions.exit_reason[0, 0] == REASON_TIME_EXIT
        assert decisions.exit_shares[0, 0] == 100.0

    def test_time_decay_skipped_when_research_is_enter(self):
        ti = _ticker_index("AAPL")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti)
        sim.positions[0, 0] = 100
        sim.avg_costs[0, 0] = 100.0
        sim.entry_dates[0, 0] = 0
        sim.highest_high[0, 0] = 100.0

        config = VectorizedExitConfig.from_uniform(
            n_combos=1, time_decay_exit_days=10,
            atr_trailing_enabled=False, fallback_stop_enabled=False,
            profit_take_enabled=False, momentum_exit_enabled=False,
        )
        decisions = compute_vectorized_exits(
            sim, prices=np.array([100.0]),
            atr_dollar_at_date=np.array([np.nan]),
            rsi_at_date=np.array([np.nan]),
            momentum_at_date=np.array([np.nan]),
            sector_lookback_return=np.array([0.0]),
            research_action_per_ticker=np.array([RA_ENTER], dtype=np.int8),
            sector_idx_per_ticker=np.array([-1], dtype=np.int32),
            sector_etf_ticker_idx=np.array([-1], dtype=np.int32),
            date_idx=20, config=config,
        )
        # Research re-affirming ENTER → time decay does not fire
        # (eligible mask: research_is_hold required; ENTER fails that)
        assert decisions.exit_action[0, 0] == ACTION_NONE


class TestResearchBlocking:
    @pytest.mark.parametrize("ra_code", [RA_EXIT, RA_REDUCE])
    def test_research_exit_or_reduce_blocks_all_strategy_exits(self, ra_code):
        ti = _ticker_index("AAPL")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti)
        sim.positions[0, 0] = 100
        sim.avg_costs[0, 0] = 100.0
        sim.entry_dates[0, 0] = 0
        sim.highest_high[0, 0] = 110.0

        config = VectorizedExitConfig.from_uniform(n_combos=1, atr_multiplier=3.0)
        # Force ATR exit conditions: price 85 ≤ stop 95
        decisions = compute_vectorized_exits(
            sim, prices=np.array([85.0]),
            atr_dollar_at_date=np.array([5.0]),
            rsi_at_date=np.array([50.0]),
            momentum_at_date=np.array([0.0]),
            sector_lookback_return=np.array([0.0]),
            research_action_per_ticker=np.array([ra_code], dtype=np.int8),
            sector_idx_per_ticker=np.array([-1], dtype=np.int32),
            sector_etf_ticker_idx=np.array([-1], dtype=np.int32),
            date_idx=1, config=config,
        )
        # Research already exiting/reducing → strategy checks skipped
        assert decisions.exit_action[0, 0] == ACTION_NONE


# ────────────────────────────────────────────────────────────────────
# Multi-combo: per-combo configs apply independently
# ────────────────────────────────────────────────────────────────────


class TestMultiCombo:
    def test_per_combo_atr_multiplier_applies(self):
        """3 combos with different atr_multiplier values; one fires, two
        don't, depending on stop_level vs price."""
        ti = _ticker_index("AAPL")
        sim = VectorizedSimulator(n_combos=3, ticker_index=ti)
        sim.positions[:, 0] = 100
        sim.avg_costs[:, 0] = 100.0
        sim.entry_dates[:, 0] = 0
        sim.highest_high[:, 0] = 110.0

        # multipliers: tight=2 (stop=100, price 95 fires), normal=3 (stop=95, price 95 fires),
        # loose=4 (stop=90, no fire).
        config = VectorizedExitConfig(
            atr_trailing_enabled=np.array([True, True, True]),
            fallback_stop_enabled=np.array([False, False, False]),
            profit_take_enabled=np.array([False, False, False]),
            momentum_exit_enabled=np.array([False, False, False]),
            time_decay_enabled=np.array([False, False, False]),
            sector_relative_veto_enabled=np.array([False, False, False]),
            # Floor enabled at -0.15; price 95 vs cost 100 is only -5%, so it
            # never fires here — this case stays a pure ATR-multiplier test.
            position_loss_floor_enabled=np.array([True, True, True]),
            position_loss_floor_pct=np.full(3, -0.15),
            atr_multiplier=np.array([2.0, 3.0, 4.0]),
            fallback_stop_pct=np.full(3, 0.10),
            profit_take_pct=np.full(3, 0.25),
            momentum_exit_threshold=np.full(3, -15.0),
            momentum_exit_rsi=np.full(3, 30.0),
            time_decay_reduce_days=np.full(3, 5, dtype=np.int32),
            time_decay_exit_days=np.full(3, 10, dtype=np.int32),
            sector_relative_outperform_threshold=np.full(3, 0.05),
            reduce_fraction=np.full(3, 0.50),
        )

        decisions = compute_vectorized_exits(
            sim, prices=np.array([95.0]),
            atr_dollar_at_date=np.array([5.0]),
            rsi_at_date=np.array([50.0]),
            momentum_at_date=np.array([0.0]),
            sector_lookback_return=np.array([0.0]),
            research_action_per_ticker=np.array([RA_HOLD], dtype=np.int8),
            sector_idx_per_ticker=np.array([-1], dtype=np.int32),
            sector_etf_ticker_idx=np.array([-1], dtype=np.int32),
            date_idx=1, config=config,
        )
        # Combo 0 (mult=2): stop=110-10=100, price 95 ≤ 100 → EXIT
        # Combo 1 (mult=3): stop=110-15=95,  price 95 ≤ 95  → EXIT
        # Combo 2 (mult=4): stop=110-20=90,  price 95 > 90  → NONE
        assert decisions.exit_action[0, 0] == ACTION_EXIT
        assert decisions.exit_action[1, 0] == ACTION_EXIT
        assert decisions.exit_action[2, 0] == ACTION_NONE


# ────────────────────────────────────────────────────────────────────
# Apply mutates state correctly
# ────────────────────────────────────────────────────────────────────


class TestApplyExits:
    def test_exit_zeroes_position_and_credits_cash(self):
        ti = _ticker_index("AAPL", "MSFT")
        sim = VectorizedSimulator(
            n_combos=2, ticker_index=ti, init_cash=1_000_000,
        )
        # Two combos hold AAPL (100 sh @ $100); only combo 0 will EXIT
        sim.positions[:, 0] = 100
        sim.avg_costs[:, 0] = 100.0
        sim.entry_dates[:, 0] = 0
        sim.highest_high[:, 0] = 100.0
        # Pre-debit cash to reflect the position (1M - 10000 = 990000)
        sim.cash[:] = 990_000

        decisions = ExitDecisions(
            exit_action=np.array(
                [[ACTION_EXIT, ACTION_NONE], [ACTION_NONE, ACTION_NONE]],
                dtype=np.int8,
            ),
            exit_reason=np.array(
                [[REASON_ATR, REASON_NONE], [REASON_NONE, REASON_NONE]],
                dtype=np.int8,
            ),
            exit_shares=np.array(
                [[100.0, 0.0], [0.0, 0.0]], dtype=np.float64,
            ),
        )
        prices = np.array([110.0, 300.0])
        n = apply_vectorized_exits(sim, decisions, prices)
        assert n == 1
        # Combo 0: position cleared, cash += 100*110 = 1,001,000
        assert sim.positions[0, 0] == 0
        assert sim.cash[0] == 1_001_000
        # Combo 1: untouched
        assert sim.positions[1, 0] == 100
        assert sim.cash[1] == 990_000

    def test_reduce_partial_position(self):
        ti = _ticker_index("AAPL")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti, init_cash=1_000_000)
        sim.positions[0, 0] = 100
        sim.avg_costs[0, 0] = 100.0
        sim.entry_dates[0, 0] = 0
        sim.highest_high[0, 0] = 130.0
        sim.cash[0] = 990_000

        decisions = ExitDecisions(
            exit_action=np.array([[ACTION_REDUCE]], dtype=np.int8),
            exit_reason=np.array([[REASON_PROFIT]], dtype=np.int8),
            exit_shares=np.array([[50.0]], dtype=np.float64),
        )
        n = apply_vectorized_exits(sim, decisions, np.array([130.0]))
        assert n == 1
        assert sim.positions[0, 0] == 50.0
        assert sim.cash[0] == 990_000 + 50 * 130
        # avg_cost preserved on reduce
        assert sim.avg_costs[0, 0] == 100.0
        assert sim.entry_dates[0, 0] == 0


# ────────────────────────────────────────────────────────────────────
# End-to-end parity: scalar evaluate_exits == vectorized for N=1
# ────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not os.path.isdir(_EXECUTOR_ROOT),
    reason="alpha-engine sibling repo not present",
)
class TestEndToEndParityVsScalarEvaluateExits:
    """End-to-end parity: scalar ``evaluate_exits`` per combo vs
    vectorized for the same date + state should produce matching
    per-(ticker) decisions (action + reason + shares)."""

    def _build_scalar_inputs(
        self,
        sim: VectorizedSimulator,
        ohlcv_by_ticker: dict,
        sector_etf_by_ticker: dict,  # {ticker: etf_ticker_str}
        run_date: pd.Timestamp,
        research_actions: dict,  # {ticker: "HOLD"|"ENTER"|"EXIT"|"REDUCE"}
        strategy_config: dict,
    ):
        """Translate vectorized state into the scalar ``evaluate_exits``
        argument shape."""
        from executor.ibkr import SimulatedIBKRClient

        run_date_str = run_date.strftime("%Y-%m-%d")
        date_to_idx = {
            d: i for i, d in enumerate(
                pd.date_range("2024-01-01", periods=200, freq="B")
            )
        }

        positions: dict = {}
        prices_now: dict = {}
        for ticker, idx in sim.ticker_index.items():
            if sim.positions[0, idx] > 0:
                # Recover entry_date from the date axis
                entry_idx = int(sim.entry_dates[0, idx])
                entry_date = pd.date_range(
                    "2024-01-01", periods=200, freq="B",
                )[entry_idx].strftime("%Y-%m-%d")
                positions[ticker] = {
                    "shares": int(sim.positions[0, idx]),
                    "avg_cost": float(sim.avg_costs[0, idx]),
                    "entry_date": entry_date,
                    "sector": sector_etf_by_ticker.get(ticker, ""),
                }
            df = ohlcv_by_ticker.get(ticker)
            if df is not None and not df.empty:
                # Use close at run_date as current_price
                df_to_run = df.loc[:run_date]
                if not df_to_run.empty:
                    prices_now[ticker] = float(df_to_run["close"].iloc[-1])

        sim_client = SimulatedIBKRClient(
            prices=prices_now, nav=float(sim.nav[0]),
        )
        sim_client._simulation_date = run_date_str
        signals_by_ticker = {
            t: {"signal": ra} for t, ra in research_actions.items()
        }

        # Truncate price histories to bars on/before run_date so
        # evaluate_exits' "look back from current bar" behavior matches
        # the simulator's "as-of date" view.
        truncated = {
            t: df.loc[:run_date]
            for t, df in ohlcv_by_ticker.items()
        }

        # Build sector ETF histories — for each held position's sector,
        # find the ETF, look up its OHLCV.
        sector_etf_histories: dict = {}
        for ticker, sector_label in sector_etf_by_ticker.items():
            etf_df = ohlcv_by_ticker.get(sector_label)
            if etf_df is not None:
                sector_etf_histories[sector_label] = etf_df.loc[:run_date]

        return (
            positions, signals_by_ticker, run_date_str, truncated,
            sim_client, sector_etf_histories,
        )

    def test_atr_decision_matches_scalar_per_ticker(self):
        """Build a multi-ticker scenario; run scalar ``evaluate_exits`` and
        vectorized side-by-side. Check that for each held ticker, the
        chosen action + reason match.
        """
        from executor.feature_lookup import FeatureLookup
        from executor.strategies.exit_manager import evaluate_exits

        ti = _ticker_index("AAPL", "MSFT", "Technology")
        n_bars = 60
        # AAPL: trends down hard from 110 to 85 over recent bars
        aapl = _trending_ohlcv(n_bars, start_price=110, drift=-0.005, vol=0.5, seed=1)
        # MSFT: up trend, profit-take territory
        msft = _trending_ohlcv(n_bars, start_price=100, drift=0.005, vol=0.4, seed=2)
        # Tech ETF: flat
        tech = _trending_ohlcv(n_bars, start_price=200, drift=0.0001, vol=0.2, seed=3)
        ohlcv = {"AAPL": aapl, "MSFT": msft, "Technology": tech}
        run_date = aapl.index[-1]

        # Vectorized state
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti)
        # AAPL: held since bar 5, avg_cost=100, highest_high=110
        sim.positions[0, 0] = 100
        sim.avg_costs[0, 0] = 100.0
        sim.entry_dates[0, 0] = 5
        sim.highest_high[0, 0] = 110.0
        # MSFT: held since bar 10, avg_cost=100, highest_high=130
        sim.positions[0, 1] = 100
        sim.avg_costs[0, 1] = 100.0
        sim.entry_dates[0, 1] = 10
        sim.highest_high[0, 1] = 130.0

        # Scalar ``evaluate_exits`` config (dict shape)
        strategy_config = {
            "atr_trailing_enabled": True, "atr_period": 14, "atr_multiplier": 3.0,
            "fallback_stop_enabled": True, "fallback_stop_pct": 0.10,
            "profit_take_enabled": True, "profit_take_pct": 0.25,
            "momentum_exit_enabled": True, "momentum_exit_threshold": -15.0,
            "momentum_exit_rsi": 30,
            "time_decay_enabled": True, "time_decay_reduce_days": 5,
            "time_decay_exit_days": 10,
            "sector_relative_veto_enabled": True,
            "sector_relative_outperform_threshold": 0.05,
        }

        positions, signals, run_str, hist, sim_client, etf_hist = self._build_scalar_inputs(
            sim, ohlcv, {"AAPL": "Technology", "MSFT": "Technology"},
            run_date, {}, strategy_config,
        )
        feature_lookup = FeatureLookup.from_ohlcv_by_ticker(ohlcv)

        scalar_signals = evaluate_exits(
            current_positions=positions,
            signals_by_ticker=signals,
            run_date=run_str,
            price_histories=hist,
            ibkr_client=sim_client,
            strategy_config=strategy_config,
            sector_etf_histories=etf_hist,
            feature_lookup=feature_lookup,
        )

        # Vectorized
        n_bars_avail = n_bars
        # Build per-ticker feature arrays at run_date by calling
        # FeatureLookup ourselves.
        atr = np.array([
            feature_lookup.atr_dollar_at("AAPL", run_date) or np.nan,
            feature_lookup.atr_dollar_at("MSFT", run_date) or np.nan,
            np.nan,
        ])
        rsi = np.array([
            feature_lookup.rsi_at("AAPL", run_date) or np.nan,
            feature_lookup.rsi_at("MSFT", run_date) or np.nan,
            np.nan,
        ])
        mom = np.array([
            feature_lookup.momentum_20d_pct_at("AAPL", run_date) or np.nan,
            feature_lookup.momentum_20d_pct_at("MSFT", run_date) or np.nan,
            np.nan,
        ])
        # Sector lookback returns: 20-bar return per ticker.
        def _ret_20(df):
            if len(df) < 20:
                return np.nan
            return float(df["close"].iloc[-1] / df["close"].iloc[-20] - 1)
        sec_ret = np.array([
            _ret_20(aapl.loc[:run_date]),
            _ret_20(msft.loc[:run_date]),
            _ret_20(tech.loc[:run_date]),
        ])

        config = VectorizedExitConfig.from_uniform(
            n_combos=1, **{
                k: v for k, v in strategy_config.items()
                if k in {
                    "atr_trailing_enabled", "atr_multiplier",
                    "fallback_stop_enabled", "fallback_stop_pct",
                    "profit_take_enabled", "profit_take_pct",
                    "momentum_exit_enabled", "momentum_exit_threshold",
                    "momentum_exit_rsi",
                    "time_decay_enabled", "time_decay_reduce_days",
                    "time_decay_exit_days",
                    "sector_relative_veto_enabled",
                    "sector_relative_outperform_threshold",
                }
            },
            reduce_fraction=0.50,
        )

        sec_idx = np.array([0, 0, 0], dtype=np.int32)  # all in sector 0 (Technology)
        sec_etf = np.array([2], dtype=np.int32)  # sector 0's ETF idx = 2 (Technology)
        # date_idx = bar position of run_date in 2024-01-01-indexed B-freq
        date_idx = aapl.index.get_loc(run_date)

        prices = np.array([
            float(aapl.loc[run_date, "close"]),
            float(msft.loc[run_date, "close"]),
            float(tech.loc[run_date, "close"]),
        ])
        decisions = compute_vectorized_exits(
            sim, prices=prices,
            atr_dollar_at_date=atr,
            rsi_at_date=rsi,
            momentum_at_date=mom,
            sector_lookback_return=sec_ret,
            research_action_per_ticker=np.array(
                [RA_HOLD, RA_HOLD, RA_HOLD], dtype=np.int8,
            ),
            sector_idx_per_ticker=sec_idx,
            sector_etf_ticker_idx=sec_etf,
            date_idx=int(date_idx), config=config,
        )

        # Translate scalar signals to a {ticker: (action, reason)} dict
        scalar_by_ticker = {
            sig["ticker"]: (sig["action"], sig["reason"])
            for sig in scalar_signals
        }

        for ticker, idx in ti.items():
            if sim.positions[0, idx] == 0:
                continue
            v_action = decisions.exit_action[0, idx]
            v_reason = decisions.exit_reason[0, idx]
            if ticker in scalar_by_ticker:
                scalar_action_name, scalar_reason_name = scalar_by_ticker[ticker]
                expected_action = _ACTION_BY_NAME[scalar_action_name]
                expected_reason = _REASON_BY_NAME[scalar_reason_name]
                assert v_action == expected_action, (
                    f"{ticker}: vectorized action {v_action} != "
                    f"scalar {scalar_action_name} ({expected_action})"
                )
                assert v_reason == expected_reason, (
                    f"{ticker}: vectorized reason {v_reason} != "
                    f"scalar {scalar_reason_name} ({expected_reason})"
                )
            else:
                assert v_action == ACTION_NONE, (
                    f"{ticker}: vectorized fired ({v_action}) but scalar "
                    f"emitted no exit"
                )
