"""Tests for synthetic/null_calibration.py — leg (a) of the L4593 backtester
correctness battery.

These are the *leak detectors*. They drive the production simulation and
significance machinery over synthetic NULL inputs (zero-edge random-walk prices,
uninformative random signals) and assert the machinery invents no edge:

  Sim engine
    1. Realized alpha-vs-SPY centers on zero (0 inside the 95% CI of the mean).
    2. Sharpe centers on zero with no transaction costs.
    3. Transaction costs can only SUBTRACT — fees never push the null positive.
    4. PSR flags "skill" no more often than its (generous) nominal FP rate.

  Significance gate
    5. The Monte-Carlo permutation gate's empirical false-positive rate over many
       independent null datasets stays near its nominal alpha (no inflation).

All bounds are principled statistical tolerances, NOT values fitted to the seed:
"0 inside the CI", "fees ≤ no-fees", "FP rate ≤ nominal + finite-sample slack".
Everything is deterministic for the fixed default seeds. A FAILURE here means the
machinery is manufacturing alpha/significance from noise — exactly the bug class
this leg exists to catch (record it in EXPERIMENTS.md).
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from synthetic.null_calibration import (
    PSR_SIGNIFICANT_THRESHOLD,
    SPY_TICKER,
    build_null_research_db,
    generate_random_orders,
    generate_random_walk_prices,
    run_sim_null_calibration,
    run_significance_gate_calibration,
)

# Generous finite-sample slack above the nominal one-sided 5% rate. The leak
# detector fires only on gross inflation (e.g. a gate that "finds" alpha in
# noise ~half the time), not on ordinary sampling wobble at small N.
MAX_NULL_FALSE_POSITIVE_RATE = 0.25


# ── Module-scoped calibrations (computed once, shared across assertions) ──────

# Trial counts trimmed from module defaults for a lean default-suite footprint
# while keeping N large enough for a tight mean-CI / FP-rate granularity.
_SIM_TRIALS = 80
_GATE_DATASETS = 20
_GATE_PERMUTATIONS = 150


@pytest.fixture(scope="module")
def sim_zero_fees():
    return run_sim_null_calibration(fees=0.0, n_trials=_SIM_TRIALS)


@pytest.fixture(scope="module")
def sim_with_fees():
    return run_sim_null_calibration(fees=0.002, n_trials=_SIM_TRIALS)  # 20 bps


@pytest.fixture(scope="module")
def gate_report():
    return run_significance_gate_calibration(
        n_datasets=_GATE_DATASETS, n_permutations=_GATE_PERMUTATIONS
    )


# ════════════════════════════════════════════════════════════════════════════
# Generators — fast structural / determinism unit tests
# ════════════════════════════════════════════════════════════════════════════

class TestRandomWalkPrices:
    def test_shape_columns_and_anchor(self):
        df = generate_random_walk_prices(n_tickers=5, n_days=40, seed=1)
        assert df.shape == (40, 6)  # 5 tickers + SPY
        assert SPY_TICKER in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)
        # day 0 anchored at the start price for every column
        assert (df.iloc[0] == 100.0).all()
        assert (df > 0).all().all()

    def test_deterministic_by_seed(self):
        a = generate_random_walk_prices(4, 30, seed=7)
        b = generate_random_walk_prices(4, 30, seed=7)
        c = generate_random_walk_prices(4, 30, seed=8)
        pd.testing.assert_frame_equal(a, b)
        assert not a.equals(c)

    def test_zero_drift_returns_center_on_zero(self):
        # Mean daily simple return across a big panel should sit near 0.
        df = generate_random_walk_prices(200, 250, seed=3)
        daily = df.pct_change().dropna().to_numpy().ravel()
        se = daily.std(ddof=1) / np.sqrt(daily.size)
        assert abs(daily.mean()) <= 4 * se

    def test_can_omit_spy(self):
        df = generate_random_walk_prices(3, 10, seed=1, include_spy=False)
        assert SPY_TICKER not in df.columns
        assert df.shape == (10, 3)

    def test_rejects_degenerate_dims(self):
        with pytest.raises(ValueError):
            generate_random_walk_prices(0, 10, seed=1)
        with pytest.raises(ValueError):
            generate_random_walk_prices(3, 1, seed=1)


class TestRandomOrders:
    def test_contract_and_pairing(self):
        prices = generate_random_walk_prices(6, 60, seed=2, include_spy=False)
        orders = generate_random_orders(prices, seed=2, n_entries=20)
        assert orders, "expected some orders"
        enters = [o for o in orders if o["action"] == "ENTER"]
        exits = [o for o in orders if o["action"] == "EXIT"]
        # each surviving entry is paired with an exit
        assert len(enters) == len(exits)
        for o in orders:
            assert set(o) >= {"date", "ticker", "action", "shares", "price_at_order"}
            assert o["shares"] > 0
            assert o["ticker"] in prices.columns
            assert pd.Timestamp(o["date"]) in prices.index

    def test_exits_within_window(self):
        prices = generate_random_walk_prices(6, 60, seed=4, include_spy=False)
        orders = generate_random_orders(prices, seed=4, n_entries=30)
        last = prices.index[-1]
        for o in orders:
            assert pd.Timestamp(o["date"]) <= last

    def test_deterministic_by_seed(self):
        prices = generate_random_walk_prices(6, 60, seed=5, include_spy=False)
        assert generate_random_orders(prices, seed=9) == generate_random_orders(prices, seed=9)


class TestNullResearchDb:
    def test_schema_and_independence(self, tmp_path):
        db = str(tmp_path / "null.db")
        build_null_research_db(db, seed=1, n_dates=20, n_per_date=15)
        conn = sqlite3.connect(db)
        try:
            df = pd.read_sql_query("SELECT * FROM score_performance", conn)
        finally:
            conn.close()
        assert len(df) == 20 * 15
        for col in ("symbol", "score_date", "score", "return_5d", "spy_5d_return"):
            assert col in df.columns
        # score and forward return are independent by construction → near-zero corr
        corr = np.corrcoef(df["score"], df["return_5d"])[0, 1]
        assert abs(corr) < 0.15


# ════════════════════════════════════════════════════════════════════════════
# 1. Sim-engine null calibration
# ════════════════════════════════════════════════════════════════════════════

class TestSimEngineNullCalibration:
    def test_alpha_centers_on_zero(self, sim_zero_fees):
        # Random orders on zero-edge prices ⇒ no alpha vs SPY. 0 must lie inside
        # the 95% CI of the mean realized alpha.
        assert sim_zero_fees.alpha_centered_on_zero, sim_zero_fees.summary()

    def test_sharpe_centers_on_zero_without_costs(self, sim_zero_fees):
        assert sim_zero_fees.sharpe_centered_on_zero, sim_zero_fees.summary()

    def test_psr_false_positive_not_inflated(self, sim_zero_fees):
        fp = sim_zero_fees.psr_false_positive_rate
        assert fp <= MAX_NULL_FALSE_POSITIVE_RATE, (
            f"PSR flagged skill on {fp:.1%} of null trials "
            f"(threshold {PSR_SIGNIFICANT_THRESHOLD})"
        )

    def test_fees_only_subtract_alpha(self, sim_zero_fees, sim_with_fees):
        # Transaction costs can never manufacture positive edge: mean alpha with
        # fees must not exceed mean alpha without fees (allow a hair of MC noise).
        slack = 0.5 * sim_zero_fees.alpha_se
        assert sim_with_fees.alpha_mean <= sim_zero_fees.alpha_mean + slack, (
            f"fees raised mean alpha: {sim_with_fees.alpha_mean:.5f} vs "
            f"{sim_zero_fees.alpha_mean:.5f}"
        )

    def test_fees_drag_sharpe_down(self, sim_zero_fees, sim_with_fees):
        # The cost drag must show up as a lower Sharpe than the cost-free null.
        assert sim_with_fees.sharpe_mean < sim_zero_fees.sharpe_mean


# ════════════════════════════════════════════════════════════════════════════
# 2. Significance-gate null calibration
# ════════════════════════════════════════════════════════════════════════════

class TestSignificanceGateNullCalibration:
    def test_all_datasets_evaluated(self, gate_report):
        # Every synthetic null dataset must produce a usable Monte-Carlo verdict;
        # silent skips would hide an inflated FP rate.
        assert gate_report.n_evaluated == gate_report.n_datasets

    def test_false_positive_rate_near_nominal(self, gate_report):
        fp = gate_report.false_positive_rate
        assert fp <= MAX_NULL_FALSE_POSITIVE_RATE, (
            f"Monte-Carlo gate false-positive rate {fp:.1%} exceeds tolerance "
            f"(nominal {gate_report.nominal_alpha:.0%})"
        )

    def test_p_values_not_systematically_tiny(self, gate_report):
        # Under the null, p-values are ~Uniform(0,1); their mean should be near
        # 0.5. A mean crushed toward 0 signals a gate biased to "significant".
        assert gate_report.p_values.mean() > 0.30, gate_report.summary()
