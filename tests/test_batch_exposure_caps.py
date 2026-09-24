"""Same-batch exposure caps (alpha-engine-config-I11504).

The 2026-09-23 rehearsal book sat at 96-101% of NAV against a 90% cap:
``decide_entries`` checks each candidate against the PRE-batch book, so a
batch whose orders individually fit is approved in full even when their sum
does not. These tests pin the fix at all three layers — the scalar helper,
the scalar simulator step that applies it, and the vectorized sweep gate —
each with a batch whose sum crosses the cap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthetic.batch_exposure import enforce_batch_exposure_caps
from tests._sibling_checkout import (
    ensure_executor_on_sys_path,
    executor_missing_reason,
    executor_root_missing_hard_fail_on_ci,
)

_EXECUTOR_ROOT = ensure_executor_on_sys_path()
_EXECUTOR_ROOT_MISSING = executor_root_missing_hard_fail_on_ci(_EXECUTOR_ROOT)

NAV = 1_000_000.0


def _order(ticker: str, pct: float, sector: str | None = "Technology") -> dict:
    return {
        "ticker": ticker, "action": "ENTER", "shares": int(pct * NAV / 100.0),
        "price_at_order": 100.0, "portfolio_nav_at_order": NAV,
        "position_pct": pct, "sector": sector,
    }


# ── scalar helper ─────────────────────────────────────────────────────────


_CFG = {"max_equity_pct": 0.90, "max_sector_pct": 1.0}


class TestEnforceBatchExposureCaps:
    CFG = _CFG

    def test_batch_summing_past_cap_is_trimmed_to_cap(self):
        # Twelve 10% orders on an empty book: each fits alone (10% < 90%),
        # the batch is 120%. Only nine may fill.
        orders = [_order(f"T{i:02d}", 0.10, sector=f"S{i}") for i in range(12)]
        kept, blocked = enforce_batch_exposure_caps(
            orders, current_positions={}, portfolio_nav=NAV, config=self.CFG,
        )
        assert [o["ticker"] for o in kept] == [f"T{i:02d}" for i in range(9)]
        assert [b["ticker"] for b in blocked] == ["T09", "T10", "T11"]
        assert all(b["rule"] == "max_equity" for b in blocked)
        total = sum(o["position_pct"] for o in kept)
        assert total <= 0.90 + 1e-12

    def test_existing_book_plus_batch_counts(self):
        # 71% already held (the rehearsal's day-1 book), then a 25% batch.
        held = {"OLD": {"market_value": 0.71 * NAV, "sector": "X"}}
        orders = [_order(f"N{i}", 0.05, sector=f"S{i}") for i in range(5)]
        kept, blocked = enforce_batch_exposure_caps(
            orders, current_positions=held, portfolio_nav=NAV, config=self.CFG,
        )
        # 71 + 5 + 5 + 5 = 86 fits; +5 = 91 does not.
        assert len(kept) == 3
        assert len(blocked) == 2
        assert blocked[0]["pending_dollars"] == pytest.approx(0.15 * NAV)

    def test_later_smaller_order_can_still_fit(self):
        # Approval order is preserved and a blocked order does not consume
        # headroom: a later, smaller order that fits is kept.
        orders = [_order("A", 0.60, "S1"), _order("B", 0.40, "S2"),
                  _order("C", 0.25, "S3")]
        kept, blocked = enforce_batch_exposure_caps(
            orders, current_positions={}, portfolio_nav=NAV, config=self.CFG,
        )
        assert [o["ticker"] for o in kept] == ["A", "C"]
        assert [b["ticker"] for b in blocked] == ["B"]

    def test_sector_cap_accumulates_within_batch(self):
        cfg = {"max_equity_pct": 0.90, "max_sector_pct": 0.25}
        orders = [_order(f"T{i}", 0.10, sector="Technology") for i in range(3)]
        kept, blocked = enforce_batch_exposure_caps(
            orders, current_positions={}, portfolio_nav=NAV, config=cfg,
        )
        assert len(kept) == 2
        assert blocked[0]["rule"] == "max_sector"

    def test_missing_sector_is_technology_like_decide_entries(self):
        cfg = {"max_equity_pct": 0.90, "max_sector_pct": 0.25}
        held = {"H": {"market_value": 0.20 * NAV, "sector": "Technology"}}
        kept, blocked = enforce_batch_exposure_caps(
            [_order("X", 0.10, sector=None)],
            current_positions=held, portfolio_nav=NAV, config=cfg,
        )
        assert kept == [] and blocked[0]["rule"] == "max_sector"

    def test_equality_is_allowed_like_check_order(self):
        # check_order blocks on strictly-greater; exactly-at-cap passes.
        orders = [_order("A", 0.50, "S1"), _order("B", 0.40, "S2")]
        kept, blocked = enforce_batch_exposure_caps(
            orders, current_positions={}, portfolio_nav=NAV, config=self.CFG,
        )
        assert len(kept) == 2 and blocked == []

    def test_non_enter_orders_pass_through(self):
        exit_order = {"ticker": "Z", "action": "EXIT", "shares": 10}
        kept, blocked = enforce_batch_exposure_caps(
            [exit_order], current_positions={}, portfolio_nav=NAV,
            config=self.CFG,
        )
        assert kept == [exit_order] and blocked == []

    def test_defaults_match_risk_guard(self):
        # No keys in config → max_equity 0.90, max_sector 0.25.
        orders = [_order(f"T{i}", 0.20, sector=f"S{i}") for i in range(5)]
        kept, blocked = enforce_batch_exposure_caps(
            orders, current_positions={}, portfolio_nav=NAV, config={},
        )
        assert len(kept) == 4 and len(blocked) == 1


# ── scalar simulator: the decider's plan is trimmed before it fills ───────


def _df_history(n_bars: int = 100, base: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [base + i * 0.1 for i in range(n_bars)],
            "high": [base + i * 0.1 + 0.5 for i in range(n_bars)],
            "low": [base + i * 0.1 - 0.5 for i in range(n_bars)],
            "close": [base + i * 0.1 + 0.2 for i in range(n_bars)],
        },
        index=pd.bdate_range("2024-01-01", periods=n_bars),
    )


_SECTORS = ["Technology", "Healthcare", "Financials", "Energy", "Utilities",
            "Materials", "Industrials", "Consumer Staples", "Real Estate",
            "Communication Services", "Consumer Discretionary", "Other"]


def _signals(n: int) -> dict:
    enter = [
        {
            "ticker": f"TKR{i:03d}", "signal": "ENTER", "score": 80,
            "conviction": "rising", "sector": _SECTORS[i % len(_SECTORS)],
            "rating": "BUY", "price_target_upside": 0.15,
            "thesis_summary": "test",
        }
        for i in range(n)
    ]
    return {
        "date": "2026-04-24", "market_regime": "neutral",
        "sector_ratings": {s: {"rating": "market_weight"} for s in _SECTORS},
        "enter": enter, "exit": [], "reduce": [], "hold": [],
        "universe": enter, "buy_candidates": enter,
    }


def _sim_config() -> dict:
    return {
        "init_cash": NAV,
        "signals_bucket": "alpha-engine-research",
        "min_score_to_enter": 70,
        "min_conviction_to_enter": ["rising", "stable"],
        # Large enough that 1/n sizing is not capped per-name: the batch
        # then sums to ~100%+ of NAV while every order fits on its own.
        "max_position_pct": 0.20,
        "bear_max_position_pct": 0.10,
        "max_sector_pct": 1.0,
        "max_equity_pct": 0.90,
        "drawdown_circuit_breaker": 0.08,
        "earnings_proximity_warning_days": 2,
        "momentum_gate_enabled": True,
        "momentum_gate_threshold": -50.0,
        "atr_sizing_enabled": True,
        "correlation_block_enabled": False,
        "coverage_sizing_enabled": False,
        "reduce_fraction": 0.50,
        "strategy": {
            "graduated_drawdown": {"enabled": False},
            "exit_manager": {
                "atr_trailing_enabled": False,
                "fallback_stop_enabled": False,
                "profit_take_enabled": False,
                "momentum_exit_enabled": False,
                "time_decay_enabled": False,
                "sector_relative_veto_enabled": False,
            },
        },
    }


@pytest.mark.skipif(
    _EXECUTOR_ROOT_MISSING, reason=executor_missing_reason(_EXECUTOR_ROOT),
)
def test_simulated_day_never_exceeds_max_equity():
    from executor.ibkr import SimulatedIBKRClient

    from backtest import _build_merged_simulate_config, _simulate_single_date

    n = 12
    tickers = [f"TKR{i:03d}" for i in range(n)]
    etfs = ["SPY", "XLK", "XLV", "XLF", "XLY", "XLP", "XLE", "XLU",
            "XLRE", "XLB", "XLI", "XLC"]
    ts = pd.Timestamp("2026-04-24")
    price_matrix = pd.DataFrame({t: [100.0] for t in tickers + etfs}, index=[ts])
    ohlcv = {t: _df_history(base=100 + i) for i, t in enumerate(price_matrix.columns)}
    atr = {t: 0.02 for t in price_matrix.columns}
    coverage = {t: 1.0 for t in price_matrix.columns}

    sim_client = SimulatedIBKRClient(prices={}, nav=NAV)
    merged_config, strategy_config = _build_merged_simulate_config(_sim_config())

    orders, skip = _simulate_single_date(
        sim_client=sim_client, signal_date="2026-04-24",
        price_matrix=price_matrix, ohlcv_by_ticker=ohlcv,
        bucket="test-bucket", merged_config=merged_config,
        strategy_config=strategy_config, signals_override=_signals(n),
        atr_by_ticker=atr, vwap_series_by_ticker=None,
        coverage_by_ticker=coverage,
    )
    assert skip is None
    enters = [o for o in orders if o["action"] == "ENTER"]
    # The fixture is only a test of the fix if the decider would have
    # approved more than the cap allows: 12 candidates sized at ~1/12 each.
    assert 0 < len(enters) < n, (
        f"expected the batch to be trimmed; {len(enters)} of {n} entered"
    )
    book = sum(
        p["shares"] * price_matrix.loc[ts, t]
        for t, p in sim_client.get_positions().items()
    )
    assert book / NAV <= 0.90 + 1e-9, f"book is {book / NAV:.1%} of NAV"


# ── vectorized sweep: gate 14 ─────────────────────────────────────────────


def _vec_signals(n: int) -> dict:
    from synthetic.vectorized_entries import CONV_STABLE, SR_MARKET_WEIGHT

    return {
        "signal_ticker_idx": np.arange(n, dtype=np.int32),
        "signal_score": np.full(n, 80.0),
        "signal_sector_idx": np.arange(n, dtype=np.int32),
        "signal_sector_rating": np.full(n, SR_MARKET_WEIGHT, dtype=np.int8),
        "signal_conviction": np.full(n, CONV_STABLE, dtype=np.int8),
        "signal_upside": np.full(n, 0.20),
        "signal_atr_pct": np.full(n, np.nan),
        "signal_pred_confidence": np.full(n, np.nan),
        "signal_p_up": np.full(n, np.nan),
        "signal_days_to_earnings": np.full(n, -1, dtype=np.int32),
        "signal_feature_coverage": np.full(n, np.nan),
        "signal_gbm_veto": np.zeros(n, dtype=bool),
        "signal_momentum_at_date": np.full(n, np.nan),
    }


def _vec_run(n: int, *, sector_idx: np.ndarray, max_sector: float,
             held_value: float = 0.0):
    from synthetic.vectorized_entries import (
        REGIME_BULL,
        VectorizedEntryConfig,
        compute_vectorized_entries,
    )
    from synthetic.vectorized_sim import VectorizedSimulator

    ti = {f"T{i}": i for i in range(n + 1)}  # last ticker = pre-held name
    sim = VectorizedSimulator(n_combos=2, ticker_index=ti, init_cash=NAV)
    if held_value:
        sim.positions[:, n] = held_value / 100.0
        sim.avg_costs[:, n] = 100.0
        sim.cash[:] = NAV - held_value
    config = VectorizedEntryConfig.from_uniform(
        n_combos=2, max_position_pct=0.20, max_equity_pct=0.90,
        max_sector_pct=max_sector, atr_sizing_enabled=False,
    )
    sigs = _vec_signals(n)
    sigs["signal_sector_idx"] = sector_idx
    return compute_vectorized_entries(
        sim, **sigs,
        prices=np.full(n + 1, 100.0),
        nav_per_combo=np.full(2, NAV),
        dd_multiplier_per_combo=np.ones(2),
        market_regime=REGIME_BULL,
        signal_age_days=0,
        config=config,
        sector_idx_per_ticker=np.append(sector_idx, n).astype(np.int32),
    )


def test_vectorized_batch_trimmed_to_equity_cap():
    from synthetic.vectorized_entries import BLOCK_EQUITY_CAP

    n = 12
    d = _vec_run(n, sector_idx=np.arange(n, dtype=np.int32), max_sector=1.0)
    for c in range(2):
        assert 0 < d.entry_passed[c].sum() < n
        assert d.entry_dollar[c].sum() <= 0.90 * NAV + 1e-6
        blocked = ~d.entry_passed[c]
        assert np.all(d.block_reason[c, blocked] == BLOCK_EQUITY_CAP)
        # Approval is in signal order: the tail is what is dropped.
        first_block = int(np.argmax(blocked))
        assert np.all(blocked[first_block:])


def test_vectorized_existing_book_plus_batch():
    n = 12
    d = _vec_run(n, sector_idx=np.arange(n, dtype=np.int32), max_sector=1.0,
                 held_value=0.71 * NAV)
    for c in range(2):
        assert 0.71 * NAV + d.entry_dollar[c].sum() <= 0.90 * NAV + 1e-6


def test_vectorized_sector_cap_accumulates_within_batch():
    from synthetic.vectorized_entries import BLOCK_SECTOR_CAP

    n = 8
    d = _vec_run(n, sector_idx=np.zeros(n, dtype=np.int32), max_sector=0.25)
    for c in range(2):
        assert d.entry_dollar[c].sum() <= 0.25 * NAV + 1e-6
        assert np.any(d.block_reason[c] == BLOCK_SECTOR_CAP)
