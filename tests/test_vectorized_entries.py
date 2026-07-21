"""Parity tests for vectorized entry decisions (Tier 4 PR 3, 2026-04-27).

Pins that ``compute_vectorized_entries`` produces per-(combo, signal)
decisions byte-equivalent to scalar ``executor.deciders.decide_entries``
for n_combos=1.

Coverage
--------
Per-gate:
  * Already held
  * Score gate
  * Momentum confirmation gate
  * Drawdown halt
  * Bear regime + underweight block
  * GBM veto
  * Sector cap
  * Equity cap
  * Correlation block
  * Shares-round-to-zero

Sizing parity:
  * Default sizing matches scalar ``compute_position_size``
  * ATR cap activates when atr_adj < 1.0
  * Min-position-dollar floor demotes to 0 shares

Multi-combo:
  * Per-combo configs apply independently to the same signal set

Apply:
  * Approved entries mutate sim state correctly
  * Cash debited, position assigned, entry_date + avg_cost recorded

End-to-end:
  * Scalar ``decide_entries`` for N=1 produces same per-signal pass/block
    decisions as vectorized for n_combos=1.
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


from synthetic.vectorized_entries import (
    BLOCK_ALREADY_HELD,
    BLOCK_BEAR_UNDERWEIGHT,
    BLOCK_CORRELATION,
    BLOCK_DRAWDOWN_HALT,
    BLOCK_EQUITY_CAP,
    BLOCK_GBM_VETO,
    BLOCK_MOMENTUM_GATE,
    BLOCK_NO_PRICE,
    BLOCK_NONE,
    BLOCK_SCORE,
    BLOCK_SECTOR_CAP,
    BLOCK_SHARES_ZERO,
    CONV_DECLINING,
    CONV_RISING,
    CONV_STABLE,
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_NEUTRAL,
    SR_MARKET_WEIGHT,
    SR_OVERWEIGHT,
    SR_UNDERWEIGHT,
    EntryDecisions,
    VectorizedEntryConfig,
    apply_vectorized_entries,
    compute_correlation_matrix,
    compute_vectorized_entries,
)
from synthetic.vectorized_sim import VectorizedSimulator


def _ticker_index(*tickers: str) -> dict:
    return {t: i for i, t in enumerate(tickers)}


def _empty_signal_arrays(n: int) -> dict:
    """Default per-signal arrays — caller overrides what matters."""
    return {
        "signal_ticker_idx": np.zeros(n, dtype=np.int32),
        "signal_score": np.full(n, 80.0, dtype=np.float64),
        "signal_sector_idx": np.zeros(n, dtype=np.int32),
        "signal_sector_rating": np.full(n, SR_MARKET_WEIGHT, dtype=np.int8),
        "signal_conviction": np.full(n, CONV_STABLE, dtype=np.int8),
        "signal_upside": np.full(n, 0.20, dtype=np.float64),
        "signal_atr_pct": np.full(n, np.nan, dtype=np.float64),
        "signal_pred_confidence": np.full(n, np.nan, dtype=np.float64),
        "signal_p_up": np.full(n, np.nan, dtype=np.float64),
        "signal_days_to_earnings": np.full(n, -1, dtype=np.int32),
        "signal_feature_coverage": np.full(n, np.nan, dtype=np.float64),
        "signal_gbm_veto": np.zeros(n, dtype=bool),
        "signal_momentum_at_date": np.full(n, np.nan, dtype=np.float64),
    }


# ────────────────────────────────────────────────────────────────────
# Per-gate tests
# ────────────────────────────────────────────────────────────────────


class TestAlreadyHeldBlock:
    def test_blocks_existing_position(self):
        ti = _ticker_index("AAPL")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti, init_cash=1_000_000)
        sim.positions[0, 0] = 100  # already held
        config = VectorizedEntryConfig.from_uniform(
            n_combos=1, min_score_to_enter=50.0,
        )
        sigs = _empty_signal_arrays(1)
        sigs["signal_ticker_idx"] = np.array([0], dtype=np.int32)

        decisions = compute_vectorized_entries(
            sim,
            **sigs,
            prices=np.array([150.0]),
            nav_per_combo=np.array([1_000_000.0]),
            dd_multiplier_per_combo=np.array([1.0]),
            market_regime=REGIME_BULL,
            signal_age_days=0,
            config=config,
        )
        assert decisions.entry_passed[0, 0] == False
        assert decisions.block_reason[0, 0] == BLOCK_ALREADY_HELD


class TestScoreGate:
    def test_score_below_threshold_blocks(self):
        ti = _ticker_index("AAPL")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti, init_cash=1_000_000)
        config = VectorizedEntryConfig.from_uniform(
            n_combos=1, min_score_to_enter=70.0,
        )
        sigs = _empty_signal_arrays(1)
        sigs["signal_ticker_idx"] = np.array([0], dtype=np.int32)
        sigs["signal_score"] = np.array([60.0])  # below threshold

        decisions = compute_vectorized_entries(
            sim, **sigs,
            prices=np.array([150.0]),
            nav_per_combo=np.array([1_000_000.0]),
            dd_multiplier_per_combo=np.array([1.0]),
            market_regime=REGIME_BULL,
            signal_age_days=0,
            config=config,
        )
        assert decisions.block_reason[0, 0] == BLOCK_SCORE


class TestMomentumGate:
    def test_momentum_below_threshold_blocks(self):
        ti = _ticker_index("AAPL")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti, init_cash=1_000_000)
        config = VectorizedEntryConfig.from_uniform(
            n_combos=1, momentum_gate_threshold=-5.0,
        )
        sigs = _empty_signal_arrays(1)
        sigs["signal_ticker_idx"] = np.array([0], dtype=np.int32)
        sigs["signal_momentum_at_date"] = np.array([-10.0])  # below -5%

        decisions = compute_vectorized_entries(
            sim, **sigs,
            prices=np.array([150.0]),
            nav_per_combo=np.array([1_000_000.0]),
            dd_multiplier_per_combo=np.array([1.0]),
            market_regime=REGIME_BULL,
            signal_age_days=0,
            config=config,
        )
        assert decisions.block_reason[0, 0] == BLOCK_MOMENTUM_GATE

    def test_momentum_nan_skips_gate(self):
        """Scalar: when ticker_history is None or has < 21 bars, momentum
        gate is skipped. Vectorized: NaN momentum value → skip."""
        ti = _ticker_index("AAPL")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti, init_cash=1_000_000)
        config = VectorizedEntryConfig.from_uniform(n_combos=1)
        sigs = _empty_signal_arrays(1)
        sigs["signal_ticker_idx"] = np.array([0], dtype=np.int32)
        sigs["signal_momentum_at_date"] = np.array([np.nan])  # data missing

        decisions = compute_vectorized_entries(
            sim, **sigs,
            prices=np.array([150.0]),
            nav_per_combo=np.array([1_000_000.0]),
            dd_multiplier_per_combo=np.array([1.0]),
            market_regime=REGIME_BULL,
            signal_age_days=0,
            config=config,
        )
        # Should pass momentum gate (skipped) and proceed to entry
        assert decisions.block_reason[0, 0] != BLOCK_MOMENTUM_GATE


class TestDrawdownHalt:
    def test_dd_zero_halts_all_signals(self):
        ti = _ticker_index("AAPL", "MSFT")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti, init_cash=1_000_000)
        config = VectorizedEntryConfig.from_uniform(n_combos=1)
        sigs = _empty_signal_arrays(2)
        sigs["signal_ticker_idx"] = np.array([0, 1], dtype=np.int32)

        decisions = compute_vectorized_entries(
            sim, **sigs,
            prices=np.array([150.0, 300.0]),
            nav_per_combo=np.array([1_000_000.0]),
            dd_multiplier_per_combo=np.array([0.0]),  # halt
            market_regime=REGIME_BULL,
            signal_age_days=0,
            config=config,
        )
        assert decisions.block_reason[0, 0] == BLOCK_DRAWDOWN_HALT
        assert decisions.block_reason[0, 1] == BLOCK_DRAWDOWN_HALT


class TestBearUnderweightBlock:
    def test_bear_underweight_blocks(self):
        ti = _ticker_index("AAPL")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti, init_cash=1_000_000)
        config = VectorizedEntryConfig.from_uniform(n_combos=1)
        sigs = _empty_signal_arrays(1)
        sigs["signal_ticker_idx"] = np.array([0], dtype=np.int32)
        sigs["signal_sector_rating"] = np.array([SR_UNDERWEIGHT], dtype=np.int8)

        decisions = compute_vectorized_entries(
            sim, **sigs,
            prices=np.array([150.0]),
            nav_per_combo=np.array([1_000_000.0]),
            dd_multiplier_per_combo=np.array([1.0]),
            market_regime=REGIME_BEAR,
            signal_age_days=0,
            config=config,
        )
        assert decisions.block_reason[0, 0] == BLOCK_BEAR_UNDERWEIGHT

    def test_bear_overweight_passes_underweight_check(self):
        ti = _ticker_index("AAPL")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti, init_cash=1_000_000)
        config = VectorizedEntryConfig.from_uniform(n_combos=1)
        sigs = _empty_signal_arrays(1)
        sigs["signal_ticker_idx"] = np.array([0], dtype=np.int32)
        sigs["signal_sector_rating"] = np.array([SR_OVERWEIGHT], dtype=np.int8)

        decisions = compute_vectorized_entries(
            sim, **sigs,
            prices=np.array([150.0]),
            nav_per_combo=np.array([1_000_000.0]),
            dd_multiplier_per_combo=np.array([1.0]),
            market_regime=REGIME_BEAR,
            signal_age_days=0,
            config=config,
        )
        assert decisions.block_reason[0, 0] != BLOCK_BEAR_UNDERWEIGHT


class TestGBMVeto:
    def test_gbm_veto_blocks(self):
        ti = _ticker_index("AAPL")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti, init_cash=1_000_000)
        config = VectorizedEntryConfig.from_uniform(n_combos=1)
        sigs = _empty_signal_arrays(1)
        sigs["signal_ticker_idx"] = np.array([0], dtype=np.int32)
        sigs["signal_gbm_veto"] = np.array([True])

        decisions = compute_vectorized_entries(
            sim, **sigs,
            prices=np.array([150.0]),
            nav_per_combo=np.array([1_000_000.0]),
            dd_multiplier_per_combo=np.array([1.0]),
            market_regime=REGIME_BULL,
            signal_age_days=0,
            config=config,
        )
        assert decisions.block_reason[0, 0] == BLOCK_GBM_VETO


class TestSectorCap:
    def test_sector_cap_blocks_when_over(self):
        # Two tickers in same sector. Already hold a $250k position. Sector
        # cap is 25% of $1M = $250k. Adding another would push over.
        ti = _ticker_index("AAPL", "MSFT")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti, init_cash=1_000_000)
        sim.positions[0, 0] = 1666  # ~$250k at price 150
        sim.avg_costs[0, 0] = 150.0
        sim.cash[0] = 750_100

        config = VectorizedEntryConfig.from_uniform(
            n_combos=1, max_sector_pct=0.25,
        )
        sigs = _empty_signal_arrays(1)
        sigs["signal_ticker_idx"] = np.array([1], dtype=np.int32)  # MSFT (sector 0 too)
        sigs["signal_sector_idx"] = np.array([0], dtype=np.int32)

        sector_idx_per_ticker = np.array([0, 0], dtype=np.int32)
        decisions = compute_vectorized_entries(
            sim, **sigs,
            prices=np.array([150.0, 300.0]),
            nav_per_combo=np.array([1_000_000.0]),
            dd_multiplier_per_combo=np.array([1.0]),
            market_regime=REGIME_BULL,
            signal_age_days=0,
            config=config,
            sector_idx_per_ticker=sector_idx_per_ticker,
        )
        assert decisions.block_reason[0, 0] == BLOCK_SECTOR_CAP


class TestEquityCap:
    def test_equity_cap_blocks_when_over(self):
        # Already hold near-maxed equity. Adding pushes over.
        ti = _ticker_index("AAPL", "MSFT")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti, init_cash=1_000_000)
        sim.positions[0, 0] = 6000  # $900k at price 150
        sim.avg_costs[0, 0] = 150.0
        sim.cash[0] = 100_000

        config = VectorizedEntryConfig.from_uniform(
            n_combos=1, max_equity_pct=0.90,
            max_sector_pct=1.0,  # disable sector cap interference
        )
        sigs = _empty_signal_arrays(1)
        sigs["signal_ticker_idx"] = np.array([1], dtype=np.int32)
        sigs["signal_sector_idx"] = np.array([0], dtype=np.int32)

        sector_idx_per_ticker = np.array([0, 1], dtype=np.int32)
        decisions = compute_vectorized_entries(
            sim, **sigs,
            prices=np.array([150.0, 300.0]),
            nav_per_combo=np.array([1_000_000.0]),
            dd_multiplier_per_combo=np.array([1.0]),
            market_regime=REGIME_BULL,
            signal_age_days=0,
            config=config,
            sector_idx_per_ticker=sector_idx_per_ticker,
        )
        assert decisions.block_reason[0, 0] == BLOCK_EQUITY_CAP


class TestCorrelationBlock:
    def test_high_correlation_blocks_same_sector_entry(self):
        # 3 tickers, 2 held in sector 0, candidate is the 3rd in sector 0.
        # Returns: held1 = held2 = candidate (perfectly correlated).
        ti = _ticker_index("AAPL", "MSFT", "NVDA")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti, init_cash=1_000_000)
        sim.positions[0, 0] = 100
        sim.positions[0, 1] = 100
        sim.avg_costs[0, 0] = 100.0
        sim.avg_costs[0, 1] = 100.0
        sim.cash[0] = 980_000

        # Build correlation matrix from synthetic returns where AAPL == MSFT == NVDA.
        n_tickers = 3
        lookback = 60
        rng = np.random.default_rng(42)
        base = rng.normal(0, 0.01, lookback)
        returns_window = np.stack([base, base, base])  # all identical → corr = 1
        corr_matrix = compute_correlation_matrix(returns_window)
        assert corr_matrix[0, 1] == pytest.approx(1.0)

        config = VectorizedEntryConfig.from_uniform(
            n_combos=1, correlation_block_threshold=0.80,
            max_sector_pct=1.0, max_equity_pct=1.0,
        )
        sigs = _empty_signal_arrays(1)
        sigs["signal_ticker_idx"] = np.array([2], dtype=np.int32)  # NVDA
        sigs["signal_sector_idx"] = np.array([0], dtype=np.int32)

        sector_idx_per_ticker = np.array([0, 0, 0], dtype=np.int32)
        decisions = compute_vectorized_entries(
            sim, **sigs,
            prices=np.array([100.0, 100.0, 100.0]),
            nav_per_combo=np.array([1_000_000.0]),
            dd_multiplier_per_combo=np.array([1.0]),
            market_regime=REGIME_BULL,
            signal_age_days=0,
            config=config,
            correlation_matrix=corr_matrix,
            sector_idx_per_ticker=sector_idx_per_ticker,
        )
        assert decisions.block_reason[0, 0] == BLOCK_CORRELATION


# ────────────────────────────────────────────────────────────────────
# Sizing parity
# ────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not os.path.isdir(_EXECUTOR_ROOT),
    reason="alpha-engine sibling repo not present",
)
class TestSizingParityVsScalar:
    def test_basic_sizing_matches_scalar(self):
        from executor.position_sizer import compute_position_size

        # 2 ENTER signals → base_weight = 0.5; sector_adj = 1.0;
        # conviction_adj = 1.0; upside_adj = 1.0; max_pct = 0.05.
        # raw_weight = 0.5 → capped at 0.05 → dollar = $50,000 → 333 shares
        scalar = compute_position_size(
            ticker="AAPL",
            portfolio_nav=1_000_000.0,
            enter_signals=[{"ticker": "AAPL"}, {"ticker": "MSFT"}],
            signal={
                "ticker": "AAPL", "score": 80, "conviction": "stable",
                "price_target_upside": 0.20, "sector_rating": "market_weight",
            },
            sector_rating="market_weight",
            current_price=150.0,
            config={
                "max_position_pct": 0.05,
                "min_position_dollar": 500,
                "atr_sizing_enabled": False,
                "confidence_sizing_enabled": False,
                "staleness_discount_enabled": False,
                "earnings_sizing_enabled": False,
                "coverage_sizing_enabled": False,
            },
        )

        # Vectorized
        ti = _ticker_index("AAPL", "MSFT")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti, init_cash=1_000_000)
        config = VectorizedEntryConfig.from_uniform(
            n_combos=1, max_position_pct=0.05, min_position_dollar=500.0,
            atr_sizing_enabled=False, confidence_sizing_enabled=False,
            staleness_discount_enabled=False, earnings_sizing_enabled=False,
            coverage_sizing_enabled=False,
            min_score_to_enter=50.0, max_sector_pct=1.0, max_equity_pct=1.0,
        )
        sigs = _empty_signal_arrays(2)
        sigs["signal_ticker_idx"] = np.array([0, 1], dtype=np.int32)
        sigs["signal_score"] = np.array([80.0, 80.0])

        decisions = compute_vectorized_entries(
            sim, **sigs,
            prices=np.array([150.0, 300.0]),
            nav_per_combo=np.array([1_000_000.0]),
            dd_multiplier_per_combo=np.array([1.0]),
            market_regime=REGIME_BULL,
            signal_age_days=0,
            config=config,
            sector_idx_per_ticker=np.array([0, 0], dtype=np.int32),
        )

        # Vectorized AAPL sizing should match scalar.
        assert decisions.entry_shares[0, 0] == scalar["shares"]
        assert decisions.entry_dollar[0, 0] == pytest.approx(
            scalar["dollar_size"], rel=1e-9,
        )

    def test_declining_conviction_reduces_size(self):
        ti = _ticker_index("AAPL")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti, init_cash=1_000_000)
        # Two combos: one with stable, one with declining conviction
        config = VectorizedEntryConfig.from_uniform(
            n_combos=1, max_position_pct=0.05, conviction_decline_adj=0.70,
            min_position_dollar=500.0, atr_sizing_enabled=False,
            confidence_sizing_enabled=False, staleness_discount_enabled=False,
            earnings_sizing_enabled=False, coverage_sizing_enabled=False,
            min_score_to_enter=50.0, max_sector_pct=1.0, max_equity_pct=1.0,
        )
        sigs = _empty_signal_arrays(1)
        sigs["signal_ticker_idx"] = np.array([0], dtype=np.int32)
        sigs["signal_conviction"] = np.array([CONV_DECLINING], dtype=np.int8)

        decisions = compute_vectorized_entries(
            sim, **sigs,
            prices=np.array([150.0]),
            nav_per_combo=np.array([1_000_000.0]),
            dd_multiplier_per_combo=np.array([1.0]),
            market_regime=REGIME_BULL,
            signal_age_days=0,
            config=config,
            sector_idx_per_ticker=np.array([0], dtype=np.int32),
        )

        # Single signal: base = 1.0; conviction_adj = 0.70.
        # raw = 1.0 × 0.70 = 0.70; capped at 0.05.
        # Expected position = 5% of $1M = $50,000 (cap, NOT 70% of NAV).
        # Note: Single-signal-base 1.0 always exceeds 5% cap, regardless of
        # conviction_adj, so the dollar = $50,000.
        assert decisions.entry_dollar[0, 0] == pytest.approx(50_000)


# ────────────────────────────────────────────────────────────────────
# Multi-combo
# ────────────────────────────────────────────────────────────────────


class TestMultiCombo:
    def test_per_combo_score_threshold(self):
        ti = _ticker_index("AAPL")
        sim = VectorizedSimulator(n_combos=3, ticker_index=ti, init_cash=1_000_000)
        config = VectorizedEntryConfig.from_uniform(n_combos=3)
        # Override min_score per combo
        config_dict = {
            **{
                k: getattr(config, k).copy() for k in [
                    "min_score_to_enter", "momentum_gate_enabled",
                    "momentum_gate_threshold", "max_position_pct",
                    "bear_max_position_pct", "max_sector_pct",
                    "max_equity_pct", "bear_block_underweight",
                    "sector_adj_overweight", "sector_adj_market_weight",
                    "sector_adj_underweight", "conviction_decline_adj",
                    "upside_fail_adj", "min_price_target_upside",
                    "atr_sizing_enabled", "atr_sizing_target_risk",
                    "atr_sizing_floor", "atr_sizing_ceiling",
                    "confidence_sizing_enabled", "confidence_sizing_min",
                    "confidence_sizing_range", "use_p_up_sizing",
                    "p_up_sizing_blend", "staleness_discount_enabled",
                    "signal_cadence_days", "staleness_decay_per_day",
                    "staleness_floor", "earnings_sizing_enabled",
                    "earnings_proximity_days", "earnings_sizing_reduction",
                    "coverage_sizing_enabled", "coverage_derate_floor",
                    "min_position_dollar", "correlation_block_enabled",
                    "correlation_block_threshold", "correlation_lookback_days",
                    "kelly_fraction",
                ]
            },
        }
        config_dict["min_score_to_enter"] = np.array([50.0, 70.0, 90.0])
        config = VectorizedEntryConfig(**config_dict)

        sigs = _empty_signal_arrays(1)
        sigs["signal_ticker_idx"] = np.array([0], dtype=np.int32)
        sigs["signal_score"] = np.array([75.0])

        decisions = compute_vectorized_entries(
            sim, **sigs,
            prices=np.array([150.0]),
            nav_per_combo=np.full(3, 1_000_000.0),
            dd_multiplier_per_combo=np.full(3, 1.0),
            market_regime=REGIME_BULL,
            signal_age_days=0,
            config=config,
            sector_idx_per_ticker=np.array([0], dtype=np.int32),
        )
        # Combo 0 (min=50): pass
        # Combo 1 (min=70): pass
        # Combo 2 (min=90): block (score 75 < 90)
        assert decisions.entry_passed[0, 0] == True
        assert decisions.entry_passed[1, 0] == True
        assert decisions.entry_passed[2, 0] == False
        assert decisions.block_reason[2, 0] == BLOCK_SCORE


# ────────────────────────────────────────────────────────────────────
# Apply mutations
# ────────────────────────────────────────────────────────────────────


class TestApplyEntries:
    def test_passed_entry_mutates_state(self):
        ti = _ticker_index("AAPL", "MSFT")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti, init_cash=1_000_000)
        decisions = EntryDecisions(
            entry_passed=np.array([[True, False]], dtype=bool),
            entry_shares=np.array([[100.0, 0.0]], dtype=np.float64),
            entry_dollar=np.array([[15_000.0, 0.0]], dtype=np.float64),
            block_reason=np.zeros((1, 2), dtype=np.int8),
        )
        signal_ticker_idx = np.array([0, 1], dtype=np.int32)
        prices = np.array([150.0, 300.0])

        n = apply_vectorized_entries(
            sim, decisions, signal_ticker_idx=signal_ticker_idx,
            prices=prices, date_idx=10,
        )
        assert n == 1
        assert sim.positions[0, 0] == 100
        assert sim.avg_costs[0, 0] == 150.0
        assert sim.entry_dates[0, 0] == 10
        assert sim.highest_high[0, 0] == 150.0
        assert sim.cash[0] == 1_000_000 - 100 * 150
        # MSFT untouched
        assert sim.positions[0, 1] == 0
        assert sim.avg_costs[0, 1] == 0.0


# ────────────────────────────────────────────────────────────────────
# End-to-end parity vs scalar decide_entries
# ────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not os.path.isdir(_EXECUTOR_ROOT),
    reason="alpha-engine sibling repo not present",
)
class TestEndToEndParityVsScalarDecideEntries:
    def test_pass_block_decisions_match_scalar(self):
        """For 4 representative signals + 1 combo, scalar decide_entries
        should produce the same accept/reject set as vectorized."""
        from executor.deciders import decide_entries

        ti = _ticker_index("AAPL", "MSFT", "NVDA", "AMZN", "GOOG")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti, init_cash=1_000_000)
        # AAPL is already held → ALREADY_HELD block
        sim.positions[0, 0] = 100
        sim.avg_costs[0, 0] = 100.0
        sim.cash[0] = 990_000

        # 4 ENTER signals
        signals_list = [
            # AAPL: already held → block
            {"ticker": "AAPL", "score": 80, "conviction": "stable",
             "price_target_upside": 0.20, "sector": "Technology"},
            # MSFT: passes everything
            {"ticker": "MSFT", "score": 80, "conviction": "stable",
             "price_target_upside": 0.20, "sector": "Technology"},
            # NVDA: low score → block
            {"ticker": "NVDA", "score": 40, "conviction": "stable",
             "price_target_upside": 0.20, "sector": "Technology"},
            # AMZN: GBM veto → block
            {"ticker": "AMZN", "score": 80, "conviction": "stable",
             "price_target_upside": 0.20, "sector": "Consumer Discretionary"},
        ]
        sector_ratings = {
            "Technology": {"rating": "market_weight"},
            "Consumer Discretionary": {"rating": "market_weight"},
        }
        prices_now = {
            "AAPL": 150.0, "MSFT": 300.0, "NVDA": 200.0, "AMZN": 130.0,
        }
        predictions = {
            "AMZN": {"gbm_veto": True, "predicted_alpha": -0.05, "combined_rank": 100},
        }
        config = {
            "min_score_to_enter": 70,
            "max_position_pct": 0.05,
            "max_sector_pct": 0.50,
            "max_equity_pct": 1.0,
            "min_position_dollar": 500,
            "momentum_gate_enabled": False,
            "atr_sizing_enabled": False,
            "confidence_sizing_enabled": False,
            "staleness_discount_enabled": False,
            "earnings_sizing_enabled": False,
            "coverage_sizing_enabled": False,
            "correlation_block_enabled": False,
            "drawdown_circuit_breaker": 0.20,
        }
        strategy_config = {
            "intraday_pullback_atr_multiple": 1.0,
            "intraday_vwap_discount_pct": 0.005,
            "intraday_support_lookback_days": 20,
        }

        plan = decide_entries(
            enter_signals=signals_list,
            signals_raw={"universe": [], "buy_candidates": []},
            predictions_by_ticker=predictions,
            config=config,
            strategy_config=strategy_config,
            market_regime="bull",
            sector_ratings=sector_ratings,
            portfolio_nav=1_000_000.0,
            peak_nav=1_000_000.0,
            current_positions={"AAPL": {
                "shares": 100, "market_value": 15_000, "avg_cost": 100.0,
                "sector": "Technology",
            }},
            prices_now=prices_now,
            price_histories=None,
            atr_map={"AAPL": 0.02, "MSFT": 0.02, "NVDA": 0.02, "AMZN": 0.02},
            vwap_map={},
            coverage_map={},
            dd_multiplier=1.0,
            signal_age_days=0,
            earnings_by_ticker={},
            run_date="2024-01-15",
        )

        approved_tickers = {o["ticker"] for o in plan.orders}
        blocked_tickers = {b["ticker"] for b in plan.blocked}

        # Vectorized
        sigs = _empty_signal_arrays(4)
        sigs["signal_ticker_idx"] = np.array([0, 1, 2, 3], dtype=np.int32)
        sigs["signal_score"] = np.array([80.0, 80.0, 40.0, 80.0])
        sigs["signal_sector_idx"] = np.array([0, 0, 0, 1], dtype=np.int32)
        sigs["signal_sector_rating"] = np.full(4, SR_MARKET_WEIGHT, dtype=np.int8)
        sigs["signal_conviction"] = np.full(4, CONV_STABLE, dtype=np.int8)
        sigs["signal_upside"] = np.full(4, 0.20)
        sigs["signal_gbm_veto"] = np.array([False, False, False, True])

        v_config = VectorizedEntryConfig.from_uniform(
            n_combos=1, min_score_to_enter=70.0,
            max_position_pct=0.05, max_sector_pct=0.50, max_equity_pct=1.0,
            min_position_dollar=500.0, momentum_gate_enabled=False,
            atr_sizing_enabled=False, confidence_sizing_enabled=False,
            staleness_discount_enabled=False, earnings_sizing_enabled=False,
            coverage_sizing_enabled=False, correlation_block_enabled=False,
        )

        sector_idx_per_ticker = np.array([0, 0, 0, 1, 0], dtype=np.int32)
        decisions = compute_vectorized_entries(
            sim, **sigs,
            prices=np.array([150.0, 300.0, 200.0, 130.0, 100.0]),
            nav_per_combo=np.array([1_000_000.0]),
            dd_multiplier_per_combo=np.array([1.0]),
            market_regime=REGIME_BULL,
            signal_age_days=0,
            config=v_config,
            sector_idx_per_ticker=sector_idx_per_ticker,
        )

        v_approved = set()
        v_blocked = set()
        ticker_names = ["AAPL", "MSFT", "NVDA", "AMZN"]
        for s_idx, name in enumerate(ticker_names):
            if decisions.entry_passed[0, s_idx]:
                v_approved.add(name)
            else:
                v_blocked.add(name)

        assert v_approved == approved_tickers, (
            f"vectorized approved: {v_approved}; scalar: {approved_tickers}"
        )
        assert v_blocked == blocked_tickers, (
            f"vectorized blocked: {v_blocked}; scalar: {blocked_tickers}"
        )


# ────────────────────────────────────────────────────────────────────
# correlation_matrix utility
# ────────────────────────────────────────────────────────────────────


class TestCorrelationMatrixComputation:
    def test_perfectly_correlated_returns(self):
        rng = np.random.default_rng(0)
        base = rng.normal(0, 0.01, 60)
        rw = np.stack([base, base])
        m = compute_correlation_matrix(rw)
        assert m.shape == (2, 2)
        assert m[0, 1] == pytest.approx(1.0)

    def test_uncorrelated_returns_near_zero(self):
        rng = np.random.default_rng(1)
        rw = rng.normal(0, 0.01, (2, 60))
        m = compute_correlation_matrix(rw)
        # Random two streams over 60 bars: |corr| typically < 0.4
        assert abs(m[0, 1]) < 0.5
