"""Tests for leg (b) of the L4593 backtester correctness battery — closed-form
planted-truth scenarios + independent NumPy reference oracle.

Two layers, both validating the PRODUCTION path
(``vectorbt_bridge.orders_to_portfolio`` + ``portfolio_stats``):

1. **Hand-computed literals** — single-asset scenarios on prices chosen so every
   fill, fee, NAV mark, drawdown, and alpha is computable by hand. The production
   path must equal those literals to ~machine precision. No oracle involved: pure
   arithmetic vs the engine.

2. **Independent oracle agreement** — richer multi-asset / multi-trade / fee /
   slippage scenarios run through BOTH the production path and
   ``synthetic.reference_sim.simulate_reference`` (a from-scratch accountant that
   never imports vectorbt). The two must agree on the full NAV path and every
   field the bridge owns (total_return, max_drawdown, trades, win_rate,
   spy_return, total_alpha).

This catches the systematic-accounting bug class leg (a) cannot: a wrong fill
price, mis-applied fee side, off-by-one NAV mark, or alpha-over-wrong-window error
that biases every backtest identically — invisible to a centered-on-zero null,
fatal to live config tuning. A FAILURE here is the oracle disagreeing with the
engine; record which leg is wrong in EXPERIMENTS.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthetic.reference_sim import (
    UnsupportedOrderPattern,
    simulate_reference,
)
from vectorbt_bridge import orders_to_portfolio, portfolio_stats

INIT_CASH = 1_000_000.0


def _bdays(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=n)


def _enter(date: str, ticker: str, shares: float, price: float) -> dict:
    return {"date": date, "ticker": ticker, "action": "ENTER",
            "shares": shares, "price_at_order": price}


def _exit(date: str, ticker: str, shares: float, price: float) -> dict:
    return {"date": date, "ticker": ticker, "action": "EXIT",
            "shares": shares, "price_at_order": price}


# ════════════════════════════════════════════════════════════════════════════
# Layer 1 — hand-computed literal scenarios
# ════════════════════════════════════════════════════════════════════════════

class TestHandComputedLiterals:
    def test_single_winner_no_fees(self):
        # Buy 10 @ 100 (day0), sell 10 @ 130 (day3). P&L = 10*(130-100) = +300.
        # NAV: 1e6 → 1,000,100 → 1,000,200 → 1,000,300 → 1,000,300.
        idx = _bdays(5)
        prices = pd.DataFrame({"AAA": [100., 110., 120., 130., 140.]}, index=idx)
        spy = pd.Series([400., 400., 400., 400., 440.], index=idx)  # +10% over active window
        orders = [_enter("2024-01-01", "AAA", 10, 100.),
                  _exit("2024-01-04", "AAA", 10, 130.)]
        pf = orders_to_portfolio(orders, prices, init_cash=INIT_CASH, fees=0.0)
        st = portfolio_stats(pf, spy_prices=spy)

        assert st["total_return"] == pytest.approx(300.0 / INIT_CASH, rel=1e-12)
        assert st["max_drawdown"] == pytest.approx(0.0, abs=1e-12)
        assert st["total_trades"] == 1
        assert st["win_rate"] == pytest.approx(1.0)
        # active window = first NAV change (day1) → last date (day4); SPY 400→440.
        assert st["spy_return"] == pytest.approx(0.10, rel=1e-12)
        assert st["total_alpha"] == pytest.approx(300.0 / INIT_CASH - 0.10, rel=1e-12)
        # full NAV path, computed by hand
        expected_nav = [1_000_000., 1_000_100., 1_000_200., 1_000_300., 1_000_300.]
        np.testing.assert_allclose(pf.value().to_numpy(), expected_nav, rtol=1e-12)

    def test_single_winner_with_fees(self):
        # Same trade, fees = 10 bps. Entry fee = 1000*0.001 = 1.0; exit fee =
        # 1300*0.001 = 1.3. Net P&L = 300 - 1.0 - 1.3 = 297.7.
        idx = _bdays(5)
        prices = pd.DataFrame({"AAA": [100., 110., 120., 130., 140.]}, index=idx)
        orders = [_enter("2024-01-01", "AAA", 10, 100.),
                  _exit("2024-01-04", "AAA", 10, 130.)]
        pf = orders_to_portfolio(orders, prices, init_cash=INIT_CASH, fees=0.001)
        st = portfolio_stats(pf)

        assert st["total_return"] == pytest.approx(297.7 / INIT_CASH, rel=1e-9)
        expected_nav = [999_999., 1_000_099., 1_000_199., 1_000_297.7, 1_000_297.7]
        np.testing.assert_allclose(pf.value().to_numpy(), expected_nav, rtol=1e-9)

    def test_single_loser_win_rate_zero(self):
        # Buy 10 @ 100, sell 10 @ 70. P&L = -300; a losing closed trade.
        idx = _bdays(5)
        prices = pd.DataFrame({"AAA": [100., 90., 80., 70., 60.]}, index=idx)
        orders = [_enter("2024-01-01", "AAA", 10, 100.),
                  _exit("2024-01-04", "AAA", 10, 70.)]
        pf = orders_to_portfolio(orders, prices, init_cash=INIT_CASH, fees=0.0)
        st = portfolio_stats(pf)

        assert st["total_return"] == pytest.approx(-300.0 / INIT_CASH, rel=1e-12)
        assert st["total_trades"] == 1
        assert st["win_rate"] == pytest.approx(0.0)
        # peak 1e6, trough 999700 → max drawdown -0.0003
        assert st["max_drawdown"] == pytest.approx(-300.0 / INIT_CASH, rel=1e-9)

    def test_drawdown_then_recovery(self):
        # Hold 1000 sh through 100→120→130→110→90→140. Peak NAV at price 130
        # (1,030,000); trough at price 90 (990,000). max_dd = (990000-1030000)/
        # 1030000 = -0.0388349514...
        idx = _bdays(6)
        prices = pd.DataFrame({"AAA": [100., 120., 130., 110., 90., 140.]}, index=idx)
        orders = [_enter("2024-01-01", "AAA", 1000, 100.),
                  _exit("2024-01-08", "AAA", 1000, 140.)]
        pf = orders_to_portfolio(orders, prices, init_cash=INIT_CASH, fees=0.0)
        st = portfolio_stats(pf)

        assert st["max_drawdown"] == pytest.approx(-40_000.0 / 1_030_000.0, rel=1e-12)
        assert st["total_return"] == pytest.approx(40_000.0 / INIT_CASH, rel=1e-12)


# ════════════════════════════════════════════════════════════════════════════
# Layer 2 — independent oracle agreement on richer scenarios
# ════════════════════════════════════════════════════════════════════════════

def _scenario_concurrent_two_asset():
    idx = _bdays(8)
    prices = pd.DataFrame({
        "AAA": [100., 105., 110., 108., 112., 120., 118., 125.],
        "BBB": [50., 48., 52., 55., 53., 51., 60., 58.],
    }, index=idx)
    spy = pd.Series([400., 402., 401., 405., 410., 408., 412., 415.], index=idx)
    orders = [
        _enter("2024-01-02", "AAA", 500, 105.),
        _enter("2024-01-03", "BBB", 1000, 52.),
        _exit("2024-01-05", "AAA", 500, 112.),
        _exit("2024-01-09", "BBB", 1000, 60.),
    ]
    return orders, prices, spy


def _scenario_sequential_same_ticker():
    # Two non-overlapping round trips on the same ticker (one win, one loss).
    idx = _bdays(9)
    prices = pd.DataFrame({
        "AAA": [100., 110., 120., 115., 105., 95., 100., 108., 112.],
    }, index=idx)
    spy = pd.Series([300., 301., 302., 303., 302., 300., 299., 301., 305.], index=idx)
    orders = [
        _enter("2024-01-01", "AAA", 200, 100.),
        _exit("2024-01-03", "AAA", 200, 120.),     # +win
        _enter("2024-01-05", "AAA", 300, 105.),
        _exit("2024-01-08", "AAA", 300, 95.),      # loss (sold below entry)
    ]
    return orders, prices, spy


def _scenario_position_open_at_end():
    # A position still open on the last day — NAV marks it, no closed trade for it.
    # Used only for NAV/return/alpha agreement (NOT trade tally: vectorbt counts
    # the open position as a trade, the oracle counts closed round trips only).
    idx = _bdays(6)
    prices = pd.DataFrame({
        "AAA": [100., 102., 104., 103., 106., 109.],
        "BBB": [200., 198., 205., 210., 207., 215.],
    }, index=idx)
    spy = pd.Series([500., 501., 503., 502., 506., 510.], index=idx)
    orders = [
        _enter("2024-01-01", "AAA", 400, 100.),
        _exit("2024-01-04", "AAA", 400, 103.),
        _enter("2024-01-03", "BBB", 100, 205.),    # never exited → open at end
    ]
    return orders, prices, spy


# Fully-closed scenarios — every position exited by the last day, so trade tally
# and win_rate are unambiguous and the oracle agrees on EVERY field.
_ORACLE_SCENARIOS = {
    "concurrent_two_asset": _scenario_concurrent_two_asset,
    "sequential_same_ticker": _scenario_sequential_same_ticker,
}

_FEE_GRID = [
    (0.0, 0.0),
    (0.001, 0.0),
    (0.0, 10.0),
    (0.0015, 5.0),
]


@pytest.mark.parametrize("scenario_name", list(_ORACLE_SCENARIOS))
@pytest.mark.parametrize("fees,slippage_bps", _FEE_GRID)
class TestOracleAgreement:
    """The production path and the independent oracle must agree to ~machine
    precision on every field the bridge owns, across scenarios × cost settings."""

    def _run_both(self, scenario_name, fees, slippage_bps):
        orders, prices, spy = _ORACLE_SCENARIOS[scenario_name]()
        pf = orders_to_portfolio(orders, prices, init_cash=INIT_CASH,
                                 fees=fees, slippage_bps=slippage_bps)
        st = portfolio_stats(pf, spy_prices=spy)
        ref = simulate_reference(orders, prices, init_cash=INIT_CASH,
                                 fees=fees, slippage_bps=slippage_bps, spy_prices=spy)
        return pf, st, ref

    def test_nav_path_matches(self, scenario_name, fees, slippage_bps):
        pf, _, ref = self._run_both(scenario_name, fees, slippage_bps)
        np.testing.assert_allclose(
            pf.value().to_numpy(), ref.nav.to_numpy(), rtol=1e-9, atol=1e-6
        )

    def test_scalar_fields_match(self, scenario_name, fees, slippage_bps):
        _, st, ref = self._run_both(scenario_name, fees, slippage_bps)
        assert st["total_return"] == pytest.approx(ref.total_return, rel=1e-9, abs=1e-12)
        assert st["max_drawdown"] == pytest.approx(ref.max_drawdown, rel=1e-9, abs=1e-12)
        assert st["total_trades"] == ref.total_trades
        assert st["win_rate"] == pytest.approx(ref.win_rate, rel=1e-9, nan_ok=True)

    def test_alpha_fields_match(self, scenario_name, fees, slippage_bps):
        _, st, ref = self._run_both(scenario_name, fees, slippage_bps)
        assert st["spy_return"] == pytest.approx(ref.spy_return, rel=1e-9, abs=1e-12)
        assert st["total_alpha"] == pytest.approx(ref.total_alpha, rel=1e-9, abs=1e-12)


# ════════════════════════════════════════════════════════════════════════════
# Open-position mark-to-market — NAV-derived fields agree even with a live
# position on the last day (trade tally intentionally excluded; see scenario note)
# ════════════════════════════════════════════════════════════════════════════

class TestOpenPositionMarking:
    @pytest.mark.parametrize("fees,slippage_bps", _FEE_GRID)
    def test_open_position_nav_and_alpha_match(self, fees, slippage_bps):
        orders, prices, spy = _scenario_position_open_at_end()
        pf = orders_to_portfolio(orders, prices, init_cash=INIT_CASH,
                                 fees=fees, slippage_bps=slippage_bps)
        st = portfolio_stats(pf, spy_prices=spy)
        ref = simulate_reference(orders, prices, init_cash=INIT_CASH,
                                 fees=fees, slippage_bps=slippage_bps, spy_prices=spy)
        # The open BBB position is marked-to-market in NAV — total_return,
        # drawdown, and alpha must still match the independent oracle exactly.
        np.testing.assert_allclose(
            pf.value().to_numpy(), ref.nav.to_numpy(), rtol=1e-9, atol=1e-6
        )
        assert st["total_return"] == pytest.approx(ref.total_return, rel=1e-9, abs=1e-12)
        assert st["max_drawdown"] == pytest.approx(ref.max_drawdown, rel=1e-9, abs=1e-12)
        assert st["total_alpha"] == pytest.approx(ref.total_alpha, rel=1e-9, abs=1e-12)


# ════════════════════════════════════════════════════════════════════════════
# Oracle domain guards — fail loud outside the closed-form domain
# ════════════════════════════════════════════════════════════════════════════

class TestOracleDomainGuards:
    def _prices(self):
        return pd.DataFrame({"AAA": [100., 101., 102., 103.]}, index=_bdays(4))

    def test_enter_on_open_position_raises(self):
        orders = [_enter("2024-01-01", "AAA", 10, 100.),
                  _enter("2024-01-02", "AAA", 10, 101.)]  # pyramiding — unsupported
        with pytest.raises(UnsupportedOrderPattern, match="already-open"):
            simulate_reference(orders, self._prices())

    def test_exit_on_flat_position_raises(self):
        orders = [_exit("2024-01-01", "AAA", 10, 100.)]
        with pytest.raises(UnsupportedOrderPattern, match="flat"):
            simulate_reference(orders, self._prices())

    def test_unknown_ticker_raises(self):
        orders = [_enter("2024-01-01", "ZZZ", 10, 100.)]
        with pytest.raises(UnsupportedOrderPattern, match="not in price matrix"):
            simulate_reference(orders, self._prices())

    def test_date_not_in_index_raises(self):
        orders = [_enter("2030-06-06", "AAA", 10, 100.)]
        with pytest.raises(UnsupportedOrderPattern, match="not in price index"):
            simulate_reference(orders, self._prices())

    def test_unknown_action_raises(self):
        orders = [{"date": "2024-01-01", "ticker": "AAA", "action": "HOLD",
                   "shares": 10, "price_at_order": 100.}]
        with pytest.raises(UnsupportedOrderPattern, match="unknown action"):
            simulate_reference(orders, self._prices())
