"""Tests for synthetic.vectorized_stats — numpy-direct portfolio stats.

Pins:
  1. Each per-combo stat primitive (total_return, sharpe, max_dd, calmar)
     produces the right value on hand-computed fixtures.
  2. Trade counting from columnar buffers matches the
     entry → EXIT round-trip semantics (REDUCE doesn't close).
  3. compute_vectorized_stats produces a DataFrame with the same
     columns the prior vectorbt path produced.
  4. Edge cases: zero orders, single-date series, all-flat NAV,
     wipeout NAV, missing SPY data.
  5. Performance regression guard — 60 combos × 2500 dates × 60k
     orders runs in <2s (vs >90 min for the vectorbt path).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from synthetic.vectorized_orders import (
    ACTION_ENTER,
    VectorizedOrderStore,
    _OrderBuffer,
)
from synthetic.vectorized_exits import (
    ACTION_EXIT, ACTION_REDUCE, REASON_ATR, REASON_PROFIT,
)
from synthetic.vectorized_stats import (
    compute_vectorized_stats,
    compute_total_return,
    compute_daily_returns,
    compute_sharpe_ratio,
    compute_max_drawdown,
    compute_calmar_ratio,
    compute_spy_return,
    compute_sortino_ratio,
    compute_cvar,
    count_trades_from_buffer,
)


# ── Per-combo stat primitives ───────────────────────────────────────────────


class TestTotalReturn:
    def test_positive_return(self):
        nav = np.array([[1_000_000.0, 1_100_000.0]])
        out = compute_total_return(nav, init_cash=1_000_000.0)
        assert out[0] == pytest.approx(0.10)

    def test_negative_return(self):
        nav = np.array([[1_000_000.0, 800_000.0]])
        out = compute_total_return(nav, init_cash=1_000_000.0)
        assert out[0] == pytest.approx(-0.20)

    def test_multi_combo_vectorized(self):
        nav = np.array([
            [1.0, 1.5],   # +50%
            [1.0, 0.8],   # -20%
            [1.0, 1.0],   # 0%
        ])
        out = compute_total_return(nav, init_cash=1.0)
        np.testing.assert_array_almost_equal(out, [0.5, -0.2, 0.0])


class TestDailyReturns:
    def test_simple_return_chain(self):
        nav = np.array([[100.0, 110.0, 99.0, 99.0]])
        # Daily: 110/100 - 1 = 0.10; 99/110 - 1 = -0.10; 99/99 - 1 = 0
        out = compute_daily_returns(nav)
        np.testing.assert_array_almost_equal(out[0], [0.10, -0.10, 0.0])

    def test_zero_nav_yields_zero_return(self):
        """Wipeout: NAV hits 0, subsequent returns must be 0 (not inf/nan)."""
        nav = np.array([[100.0, 0.0, 0.0]])
        out = compute_daily_returns(nav)
        np.testing.assert_array_almost_equal(out[0], [-1.0, 0.0])

    def test_single_date_yields_empty(self):
        """1-date series → no daily returns."""
        nav = np.array([[100.0]])
        out = compute_daily_returns(nav)
        assert out.shape == (1, 0)


class TestSharpeRatio:
    def test_constant_returns_yields_zero_sharpe(self):
        """Constant NAV → 0 std → Sharpe = 0 (no risk-free assumption)."""
        nav = np.full((1, 252), 1_000_000.0)
        daily = compute_daily_returns(nav)
        out = compute_sharpe_ratio(daily)
        assert out[0] == 0.0

    def test_known_sharpe_from_returns(self):
        """Daily returns 0.001, std≈0 → very high sharpe.
        Generate realistic noise to get a meaningful number."""
        rng = np.random.default_rng(42)
        # 1% daily mean, 2% std → annualized Sharpe ~7.9
        daily = rng.normal(0.01, 0.02, size=(1, 1000))
        out = compute_sharpe_ratio(daily)
        # Mean / std × sqrt(252)
        expected = 0.01 / 0.02 * np.sqrt(252)
        assert out[0] == pytest.approx(expected, rel=0.1)

    def test_negative_mean_yields_negative_sharpe(self):
        rng = np.random.default_rng(7)
        daily = rng.normal(-0.005, 0.015, size=(1, 500))
        out = compute_sharpe_ratio(daily)
        assert out[0] < 0


class TestSortinoVectorized:
    def test_hand_computed_negative_sortino(self):
        # Single combo: 5 small losses + 95 small gains.
        # mean = -0.00055, downside_var = 0.000055
        # sortino_daily = -0.00055 / sqrt(0.000055) ≈ -0.07416
        # annualized = sortino_daily * sqrt(252) ≈ -1.1772
        r = np.array([
            [-0.05, -0.04, -0.03, -0.02, -0.01] + [0.001] * 95
        ])
        out = compute_sortino_ratio(r, target=0.0)
        expected = (-0.00055 / np.sqrt(0.000055)) * np.sqrt(252)
        assert out[0] == pytest.approx(expected, rel=1e-6)

    def test_zero_downside_returns_zero(self):
        r = np.array([[0.01] * 50])
        assert compute_sortino_ratio(r, target=0.0)[0] == 0.0

    def test_short_series_returns_zero(self):
        r = np.array([[0.01]])  # n_steps = 1
        assert compute_sortino_ratio(r, target=0.0)[0] == 0.0

    def test_multi_combo_vectorized(self):
        # Combo 0: all positive (no downside) → 0
        # Combo 1: hand-computed (same as scalar test)
        r0 = [0.01] * 100
        r1 = [-0.05, -0.04, -0.03, -0.02, -0.01] + [0.001] * 95
        r = np.array([r0, r1])
        out = compute_sortino_ratio(r, target=0.0)
        expected_1 = (-0.00055 / np.sqrt(0.000055)) * np.sqrt(252)
        assert out[0] == 0.0
        assert out[1] == pytest.approx(expected_1, rel=1e-6)


class TestCVaRVectorized:
    def test_hand_computed(self):
        r = np.array([
            [-0.05, -0.04, -0.03, -0.02, -0.01] + [0.001] * 95
        ])
        out = compute_cvar(r, q=0.05)
        # Worst 5: mean = -0.03
        assert out[0] == pytest.approx(-0.03, rel=1e-9)

    def test_short_series_returns_zero(self):
        r = np.array([[-0.05, 0.01] * 5])  # n=10, min_n=20 for q=0.05
        assert compute_cvar(r, q=0.05)[0] == 0.0

    def test_invalid_q_raises(self):
        r = np.array([[0.01] * 100])
        with pytest.raises(ValueError):
            compute_cvar(r, q=0.0)
        with pytest.raises(ValueError):
            compute_cvar(r, q=1.0)

    def test_multi_combo_vectorized(self):
        r0 = [0.01] * 100  # CVaR(95) of constant 0.01 = 0.01
        r1 = [-0.05, -0.04, -0.03, -0.02, -0.01] + [0.001] * 95
        r = np.array([r0, r1])
        out = compute_cvar(r, q=0.05)
        assert out[0] == pytest.approx(0.01, rel=1e-9)
        assert out[1] == pytest.approx(-0.03, rel=1e-9)


class TestMaxDrawdown:
    def test_monotonically_increasing_zero_drawdown(self):
        nav = np.array([[100.0, 110.0, 120.0, 130.0]])
        out = compute_max_drawdown(nav)
        assert out[0] == 0.0

    def test_simple_drawdown(self):
        # 100 → 120 (peak) → 90 = drawdown of (90-120)/120 = -25%
        nav = np.array([[100.0, 120.0, 90.0]])
        out = compute_max_drawdown(nav)
        assert out[0] == pytest.approx(-0.25)

    def test_recovers_then_new_peak(self):
        # 100 → 120 → 90 (-25%) → 130 → 100 (-23%); first dd is bigger
        nav = np.array([[100.0, 120.0, 90.0, 130.0, 100.0]])
        out = compute_max_drawdown(nav)
        assert out[0] == pytest.approx(-0.25)


class TestCalmar:
    def test_positive_return_zero_drawdown_yields_zero(self):
        out = compute_calmar_ratio(
            total_return=np.array([0.10]),
            max_drawdown=np.array([0.0]),
            n_dates=252,
        )
        assert out[0] == 0.0

    def test_known_calmar(self):
        # 10% return over 252 days = 10% annualized; drawdown -5%
        # → calmar = 0.10 / 0.05 = 2.0
        out = compute_calmar_ratio(
            total_return=np.array([0.10]),
            max_drawdown=np.array([-0.05]),
            n_dates=252,
        )
        assert out[0] == pytest.approx(2.0, rel=0.01)


# ── Trade counting from columnar buffers ────────────────────────────────────


class TestCountTrades:
    def test_simple_round_trip_one_winner(self):
        buf = _OrderBuffer()
        buf.add_entry(0, 0, shares=100, price=100.0, nav=1e6, position_pct=0.05)
        buf.add_exit(1, 0, action_code=ACTION_EXIT, shares=100, price=110.0,
                     nav=1.01e6, reason_code=REASON_ATR)
        n_trades, n_wins = count_trades_from_buffer(buf)
        assert n_trades == 1
        assert n_wins == 1

    def test_round_trip_loser(self):
        buf = _OrderBuffer()
        buf.add_entry(0, 0, 100, 100.0, 1e6, 0.05)
        buf.add_exit(1, 0, ACTION_EXIT, 100, 90.0, 0.99e6, REASON_ATR)
        n_trades, n_wins = count_trades_from_buffer(buf)
        assert n_trades == 1
        assert n_wins == 0

    def test_reduce_does_not_close_trade(self):
        """REDUCE in scalar's vectorbt definition doesn't increment
        trade count — partial exits aren't separate trades. Final EXIT
        closes the trade. 5 reduces + 1 exit = 1 trade (not 6)."""
        buf = _OrderBuffer()
        buf.add_entry(0, 0, 100, 100.0, 1e6, 0.05)
        for k in range(5):
            buf.add_exit(k + 1, 0, ACTION_REDUCE, 10, 105.0, 1e6, REASON_PROFIT)
        buf.add_exit(6, 0, ACTION_EXIT, 50, 110.0, 1e6, REASON_ATR)
        n_trades, n_wins = count_trades_from_buffer(buf)
        assert n_trades == 1
        assert n_wins == 1  # exit price 110 > entry price 100

    def test_re_entry_after_full_exit_starts_new_trade(self):
        buf = _OrderBuffer()
        buf.add_entry(0, 0, 100, 100.0, 1e6, 0.05)
        buf.add_exit(1, 0, ACTION_EXIT, 100, 110.0, 1.01e6, REASON_ATR)
        # Re-enter same ticker
        buf.add_entry(2, 0, 100, 105.0, 1.01e6, 0.05)
        buf.add_exit(3, 0, ACTION_EXIT, 100, 100.0, 1e6, REASON_ATR)
        n_trades, n_wins = count_trades_from_buffer(buf)
        assert n_trades == 2
        assert n_wins == 1  # First trade (100→110) won; second (105→100) lost

    def test_independent_tickers_independent_trades(self):
        buf = _OrderBuffer()
        # Two tickers concurrently held
        buf.add_entry(0, 0, 100, 100.0, 1e6, 0.05)
        buf.add_entry(0, 1, 50, 200.0, 1e6, 0.05)
        buf.add_exit(1, 0, ACTION_EXIT, 100, 110.0, 1e6, REASON_ATR)
        buf.add_exit(2, 1, ACTION_EXIT, 50, 180.0, 1e6, REASON_ATR)
        n_trades, n_wins = count_trades_from_buffer(buf)
        assert n_trades == 2
        assert n_wins == 1  # Ticker 0 won (100→110), ticker 1 lost (200→180)

    def test_empty_buffer_yields_zero(self):
        buf = _OrderBuffer()
        assert count_trades_from_buffer(buf) == (0, 0)

    def test_none_buffer_yields_zero(self):
        """Released combo: buffer is None. Must not crash."""
        assert count_trades_from_buffer(None) == (0, 0)

    def test_dangling_entry_no_exit_yields_zero_trades(self):
        """An entry without a matching EXIT doesn't count as a trade
        (open position; not a completed round-trip)."""
        buf = _OrderBuffer()
        buf.add_entry(0, 0, 100, 100.0, 1e6, 0.05)
        # No exit recorded — position open at sweep end
        n_trades, n_wins = count_trades_from_buffer(buf)
        assert n_trades == 0


# ── compute_spy_return ──────────────────────────────────────────────────────


class TestSpyReturn:
    def test_simple_return(self):
        dates = pd.date_range("2026-01-01", periods=5, freq="B")
        spy = pd.Series([400.0, 402.0, 404.0, 406.0, 410.0], index=dates)
        out = compute_spy_return(spy, dates)
        assert out == pytest.approx(0.025)

    def test_none_spy_yields_none(self):
        dates = pd.date_range("2026-01-01", periods=5, freq="B")
        assert compute_spy_return(None, dates) is None

    def test_too_few_data_points_yields_none(self):
        dates = pd.date_range("2026-01-01", periods=5, freq="B")
        spy = pd.Series([np.nan, np.nan, 400.0, np.nan, np.nan], index=dates)
        assert compute_spy_return(spy, dates) is None


# ── Public entry point: compute_vectorized_stats ────────────────────────────


def _build_store(n_combos: int, dates, tickers) -> VectorizedOrderStore:
    s = VectorizedOrderStore(n_combos)
    s.finalize(dates, tickers)
    return s


class TestComputeVectorizedStats:
    def test_returns_canonical_columns(self):
        n_combos = 2
        dates = pd.date_range("2026-01-01", periods=10, freq="B")
        tickers = ["AAPL", "MSFT"]
        nav_history = np.full((n_combos, 10), 1_000_000.0)
        # Combo 0: NAV ramps to 1.1M; combo 1: drops to 0.9M.
        nav_history[0] = np.linspace(1_000_000, 1_100_000, 10)
        nav_history[1] = np.linspace(1_000_000, 900_000, 10)
        store = _build_store(n_combos, dates, tickers)
        # Combo 0 has 1 winning trade
        store.add_entry(0, 0, 0, 100, 100.0, 1e6, 0.05)
        store.add_exit(0, 5, 0, ACTION_EXIT, 100, 110.0, 1.01e6, REASON_ATR)

        df = compute_vectorized_stats(
            nav_history=nav_history,
            init_cash=1_000_000.0,
            spy_prices=None,
            dates=dates,
            orders_per_combo=store,
            combo_params=[{"min_score": 70}, {"min_score": 80}],
        )

        # Column shape — additive vs prior vectorbt path's output:
        # sortino_ratio + cvar_95 added by evaluator-revamp PR 1.
        expected_cols = {
            "min_score", "status", "total_orders", "total_trades",
            "win_rate", "total_return", "sharpe_ratio", "sortino_ratio",
            "max_drawdown", "calmar_ratio", "cvar_95",
            "spy_return", "total_alpha",
        }
        assert set(df.columns) == expected_cols
        assert len(df) == 2

    def test_no_orders_combo_marked_status_no_orders(self):
        dates = pd.date_range("2026-01-01", periods=5, freq="B")
        store = _build_store(2, dates, ["AAPL"])
        # Combo 0: orders. Combo 1: nothing.
        store.add_entry(0, 0, 0, 100, 100.0, 1e6, 0.05)
        store.add_exit(0, 1, 0, ACTION_EXIT, 100, 110.0, 1.01e6, REASON_ATR)
        nav = np.full((2, 5), 1_000_000.0)
        df = compute_vectorized_stats(
            nav_history=nav, init_cash=1_000_000.0, spy_prices=None,
            dates=dates, orders_per_combo=store,
            combo_params=[{"k": "a"}, {"k": "b"}],
        )
        assert df.loc[df["k"] == "a", "status"].iloc[0] == "ok"
        assert df.loc[df["k"] == "b", "status"].iloc[0] == "no_orders"
        assert df.loc[df["k"] == "b", "total_orders"].iloc[0] == 0

    def test_spy_alpha_computed_when_spy_provided(self):
        dates = pd.date_range("2026-01-01", periods=5, freq="B")
        spy = pd.Series([400.0, 401.0, 402.0, 403.0, 410.0], index=dates)
        nav = np.array([[1_000_000.0] * 4 + [1_100_000.0]])  # +10%
        store = _build_store(1, dates, ["AAPL"])
        store.add_entry(0, 0, 0, 100, 100.0, 1e6, 0.05)
        store.add_exit(0, 4, 0, ACTION_EXIT, 100, 110.0, 1.1e6, REASON_ATR)
        df = compute_vectorized_stats(
            nav_history=nav, init_cash=1_000_000.0, spy_prices=spy,
            dates=dates, orders_per_combo=store,
            combo_params=[{"k": 0}],
        )
        # SPY: 410/400 - 1 = 0.025
        assert df["spy_return"].iloc[0] == pytest.approx(0.025)
        # Alpha: 0.10 - 0.025 = 0.075
        assert df["total_alpha"].iloc[0] == pytest.approx(0.075)

    def test_releases_buffers_after_compute(self):
        """Memory invariant: after stats compute, all buffers are
        released so peak memory doesn't carry forward."""
        dates = pd.date_range("2026-01-01", periods=5, freq="B")
        store = _build_store(3, dates, ["AAPL"])
        for c in range(3):
            store.add_entry(c, 0, 0, 100, 100.0, 1e6, 0.05)
        nav = np.full((3, 5), 1_000_000.0)
        compute_vectorized_stats(
            nav_history=nav, init_cash=1_000_000.0, spy_prices=None,
            dates=dates, orders_per_combo=store,
            combo_params=[{}] * 3,
        )
        # All buffers released
        assert store.total_orders() == 0
        for c in range(3):
            assert store[c] == []

    def test_combo_params_count_mismatch_raises(self):
        nav = np.zeros((2, 5))
        store = _build_store(2, pd.date_range("2026-01-01", periods=5), [])
        with pytest.raises(ValueError, match="combo_params"):
            compute_vectorized_stats(
                nav_history=nav, init_cash=1.0, spy_prices=None,
                dates=pd.date_range("2026-01-01", periods=5),
                orders_per_combo=store,
                combo_params=[{}],  # only 1 vs n_combos=2
            )


# ── Performance regression guard ────────────────────────────────────────────


class TestPerformance:
    def test_sixty_combos_2500_dates_under_two_seconds(self):
        """v16 caught the failure mode where 60 combos × ~26k orders
        through vectorbt took >90 min and tripped the watchdog. Pin
        the new path's perf at production scale: 60 combos × 2500
        dates × ~1k orders each = ~60k orders total. Must complete
        in <2s."""
        n_combos = 60
        n_dates = 2500
        rng = np.random.default_rng(42)

        # Random-walk NAV per combo
        nav_history = 1_000_000.0 * np.cumprod(
            1.0 + rng.normal(0.0003, 0.01, size=(n_combos, n_dates)),
            axis=1,
        )
        # Synthetic orders: ~1k entries + 1k exits per combo
        dates = pd.date_range("2016-01-01", periods=n_dates, freq="B")
        tickers = [f"T{i:04d}" for i in range(100)]
        store = _build_store(n_combos, dates, tickers)
        for c in range(n_combos):
            for k in range(1000):
                d = (k * 2) % n_dates
                t = k % len(tickers)
                store.add_entry(c, d, t, 100, 50.0, 1e6, 0.05)
                store.add_exit(
                    c, (d + 5) % n_dates, t, ACTION_EXIT, 100, 51.0,
                    1.01e6, REASON_ATR,
                )

        params = [{"combo": i} for i in range(n_combos)]
        t0 = time.monotonic()
        df = compute_vectorized_stats(
            nav_history=nav_history,
            init_cash=1_000_000.0,
            spy_prices=None,
            dates=dates,
            orders_per_combo=store,
            combo_params=params,
        )
        elapsed = time.monotonic() - t0

        assert len(df) == n_combos
        assert elapsed < 2.0, (
            f"compute_vectorized_stats took {elapsed:.2f}s for "
            f"{n_combos} combos × {n_dates} dates. v16 vectorbt path "
            f"took >90 min on the same scale. If this hits 2s, "
            f"investigate before merging."
        )
