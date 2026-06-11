"""Tests for leg (d) of the L4593 backtester correctness battery — metamorphic
invariants via property-based testing (Hypothesis).

Legs (a)/(b)/(c) check the engine at fixed points (null, closed-form, golden).
Leg (d) checks RELATIONS that must hold for EVERY input: transform the scenario in
a known way and the output must change (or not) in a predictable way. Hypothesis
searches the scenario space — random-walk-ish prices, trade timings, share counts,
fee levels — and shrinks any counterexample to a minimal failing case.

Invariants pinned (production path = ``orders_to_portfolio`` + ``portfolio_stats``,
cross-checked against the leg-(b) ``reference_sim`` oracle):

1. **Price-scale invariance** — multiply every price by k and divide every share
   count by k (identical dollar exposure) ⇒ NAV path, total_return, max_drawdown,
   and total_alpha are UNCHANGED. A units/scale bug breaks this.
2. **Fee monotonicity** — raising fees can only lower (or hold) total_return.
3. **Fee conservation** — return(0 fees) − return(f) equals exactly the total fees
   paid as a fraction of capital. Ties fee accounting to no-money-created.
4. **No-op portfolio** — zero orders ⇒ total_return is exactly 0 and (by the
   active-window design) total_alpha is None.

A FAILURE is a metamorphic relation the engine violates on some input Hypothesis
found — record the shrunk counterexample in EXPERIMENTS.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from synthetic.reference_sim import simulate_reference
from vectorbt_bridge import orders_to_portfolio, portfolio_stats

# Large capital base so generated trades never hit a cash constraint (which
# vectorbt would resolve by rejecting/shrinking an order — a real behavior, but
# not what these invariants probe).
INIT_CASH = 1_000_000_000.0

_DEADLINE = None  # vectorbt construction time varies; correctness, not speed, here
_SETTINGS = settings(
    max_examples=60,
    deadline=_DEADLINE,
    suppress_health_check=[HealthCheck.too_slow],
)


@st.composite
def scenarios(draw, *, with_spy: bool = True):
    """Generate a fully-closed, in-domain portfolio scenario.

    Each ticker is traded at most once (one round trip) so every scenario stays
    inside the oracle's closed-form domain (one open position per ticker, full
    exits) without rejection. Returns (orders, prices_df, spy_series).
    """
    n_days = draw(st.integers(min_value=4, max_value=25))
    n_tickers = draw(st.integers(min_value=1, max_value=4))
    dates = pd.bdate_range("2024-01-01", periods=n_days)

    price_float = st.floats(min_value=20.0, max_value=500.0,
                            allow_nan=False, allow_infinity=False)

    cols: dict[str, list[float]] = {}
    for i in range(n_tickers):
        cols[f"T{i}"] = draw(st.lists(price_float, min_size=n_days, max_size=n_days))
    prices = pd.DataFrame(cols, index=dates)

    orders: list[dict] = []
    for i in range(n_tickers):
        entry = draw(st.integers(min_value=0, max_value=n_days - 2))
        exit_ = draw(st.integers(min_value=entry + 1, max_value=n_days - 1))
        shares = float(draw(st.integers(min_value=1, max_value=10_000)))
        t = f"T{i}"
        orders.append({"date": dates[entry].strftime("%Y-%m-%d"), "ticker": t,
                       "action": "ENTER", "shares": shares,
                       "price_at_order": float(prices.iloc[entry][t])})
        orders.append({"date": dates[exit_].strftime("%Y-%m-%d"), "ticker": t,
                       "action": "EXIT", "shares": shares,
                       "price_at_order": float(prices.iloc[exit_][t])})

    spy = None
    if with_spy:
        spy_vals = draw(st.lists(price_float, min_size=n_days, max_size=n_days))
        spy = pd.Series(spy_vals, index=dates, name="SPY")

    return orders, prices, spy


def _scale_orders(orders: list[dict], k: float) -> list[dict]:
    """Divide share counts by k and multiply order prices by k (same dollars)."""
    out = []
    for o in orders:
        out.append({**o, "shares": o["shares"] / k,
                    "price_at_order": o["price_at_order"] * k})
    return out


# ════════════════════════════════════════════════════════════════════════════
# 1. Price-scale invariance
# ════════════════════════════════════════════════════════════════════════════

class TestPriceScaleInvariance:
    @given(data=scenarios(), k=st.floats(min_value=0.1, max_value=10.0,
                                         allow_nan=False, allow_infinity=False))
    @_SETTINGS
    def test_returns_invariant_to_price_scaling(self, data, k):
        orders, prices, spy = data
        pf0 = orders_to_portfolio(orders, prices, init_cash=INIT_CASH, fees=0.0)
        st0 = portfolio_stats(pf0, spy_prices=spy)

        # prices × k, shares ÷ k ⇒ identical dollar position every day.
        orders_k = _scale_orders(orders, k)
        prices_k = prices * k
        spy_k = spy * k if spy is not None else None
        pfk = orders_to_portfolio(orders_k, prices_k, init_cash=INIT_CASH, fees=0.0)
        stk = portfolio_stats(pfk, spy_prices=spy_k)

        np.testing.assert_allclose(pf0.value().to_numpy(), pfk.value().to_numpy(), rtol=1e-9)
        assert stk["total_return"] == pytest.approx(st0["total_return"], rel=1e-9, abs=1e-12)
        assert stk["max_drawdown"] == pytest.approx(st0["max_drawdown"], rel=1e-9, abs=1e-12)
        if st0["total_alpha"] is not None and stk["total_alpha"] is not None:
            assert stk["total_alpha"] == pytest.approx(st0["total_alpha"], rel=1e-9, abs=1e-12)


# ════════════════════════════════════════════════════════════════════════════
# 2. Fee monotonicity
# ════════════════════════════════════════════════════════════════════════════

class TestFeeMonotonicity:
    @given(
        data=scenarios(with_spy=False),
        fees=st.lists(
            st.floats(min_value=0.0, max_value=0.02, allow_nan=False, allow_infinity=False),
            min_size=2, max_size=2,
        ),
    )
    @_SETTINGS
    def test_higher_fees_never_increase_return(self, data, fees):
        orders, prices, _ = data
        f_lo, f_hi = sorted(fees)
        r_lo = portfolio_stats(
            orders_to_portfolio(orders, prices, init_cash=INIT_CASH, fees=f_lo)
        )["total_return"]
        r_hi = portfolio_stats(
            orders_to_portfolio(orders, prices, init_cash=INIT_CASH, fees=f_hi)
        )["total_return"]
        # More cost can only erode return (tiny float slack).
        assert r_hi <= r_lo + 1e-12


# ════════════════════════════════════════════════════════════════════════════
# 3. Fee conservation
# ════════════════════════════════════════════════════════════════════════════

class TestFeeConservation:
    @given(
        data=scenarios(with_spy=False),
        f=st.floats(min_value=0.0, max_value=0.02, allow_nan=False, allow_infinity=False),
    )
    @_SETTINGS
    def test_fee_drag_equals_total_fees_paid(self, data, f):
        orders, prices, _ = data
        r0 = portfolio_stats(
            orders_to_portfolio(orders, prices, init_cash=INIT_CASH, fees=0.0)
        )["total_return"]
        rf = portfolio_stats(
            orders_to_portfolio(orders, prices, init_cash=INIT_CASH, fees=f)
        )["total_return"]

        # Total fee dollars = f × Σ (fill value) over every entry and exit fill;
        # the fill value uses the matrix close on the fill date (what vbt charges).
        total_fee_dollars = 0.0
        for o in orders:
            d = pd.Timestamp(o["date"])
            fill_value = o["shares"] * float(prices.loc[d, o["ticker"]])
            total_fee_dollars += fill_value * f
        expected_drag = total_fee_dollars / INIT_CASH

        assert (r0 - rf) == pytest.approx(expected_drag, rel=1e-7, abs=1e-12)


# ════════════════════════════════════════════════════════════════════════════
# 4. No-op portfolio
# ════════════════════════════════════════════════════════════════════════════

class TestNoOpPortfolio:
    @given(data=scenarios())
    @_SETTINGS
    def test_zero_orders_is_flat_and_benchmark_undefined(self, data):
        _, prices, spy = data
        pf = orders_to_portfolio([], prices, init_cash=INIT_CASH, fees=0.001)
        stx = portfolio_stats(pf, spy_prices=spy)
        # No trades ⇒ NAV never departs from init ⇒ zero return, no active window.
        assert stx["total_return"] == pytest.approx(0.0, abs=1e-12)
        assert stx["total_trades"] == 0
        assert stx["total_alpha"] is None  # active-window-None by design
        # The oracle agrees.
        ref = simulate_reference([], prices, init_cash=INIT_CASH, fees=0.001, spy_prices=spy)
        assert ref.total_return == pytest.approx(0.0, abs=1e-12)
        assert ref.total_alpha is None
