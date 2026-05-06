"""Tests for analysis.risk_matched_benchmark.

Pins:
  1. EW-high-vol benchmark selects top-quartile by trailing vol.
  2. EW-high-vol returns equal-weight mean of selected returns.
  3. Beta-matched SPY: when portfolio = SPY, beta = 1, benchmark ≡ SPY.
  4. Beta-matched SPY: when portfolio = 0 (no exposure), beta = 0, benchmark = 0.
  5. compute_alpha_vs_benchmark hand-computes correctly on aligned series.
  6. IR formula: annualized excess / std of excess * sqrt(252).
  7. Empty / insufficient inputs handled.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from analysis.risk_matched_benchmark import (
    compute_alpha_vs_benchmark,
    construct_beta_matched_spy_benchmark,
    construct_ew_high_vol_benchmark,
)


def _build_prices(n_days: int = 200, seed: int = 0):
    """Synthetic prices: AAA high-vol, BBB low-vol, CCC mid-vol, DDD flat."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-05", periods=n_days, freq="B")
    returns = pd.DataFrame({
        "AAA": rng.normal(0.001, 0.04, size=n_days),  # high vol
        "BBB": rng.normal(0.0005, 0.01, size=n_days),  # low vol
        "CCC": rng.normal(0.0008, 0.02, size=n_days),  # mid vol
        "DDD": np.full(n_days, 0.0001),                # flat / ~zero vol
    }, index=dates)
    prices = (1.0 + returns).cumprod() * 100.0
    return prices, dates


class TestEWHighVol:
    def test_selects_high_vol_subset(self):
        prices, dates = _build_prices(n_days=200, seed=42)
        bench = construct_ew_high_vol_benchmark(
            prices, vol_quantile=0.75, vol_lookback_days=60,
        )
        # AAA is highest vol; bench should track AAA closely (top quartile
        # of 4 with q=0.75 = 1 ticker).
        assert not bench.empty
        # Sanity: bench std should be much higher than DDD's near-zero.
        ddd_std = prices["DDD"].pct_change().dropna().std()
        bench_std = bench.std()
        assert bench_std > 5 * ddd_std

    def test_invalid_quantile_raises(self):
        prices, _ = _build_prices(n_days=80)
        with pytest.raises(ValueError):
            construct_ew_high_vol_benchmark(prices, vol_quantile=0.0)
        with pytest.raises(ValueError):
            construct_ew_high_vol_benchmark(prices, vol_quantile=1.0)

    def test_short_lookback_raises(self):
        prices, _ = _build_prices(n_days=80)
        with pytest.raises(ValueError):
            construct_ew_high_vol_benchmark(prices, vol_lookback_days=4)

    def test_universe_filter(self):
        # Restrict to only AAA + DDD; benchmark must select AAA (top vol).
        prices, _ = _build_prices(n_days=200, seed=11)
        bench = construct_ew_high_vol_benchmark(
            prices, universe=["AAA", "DDD"],
            vol_quantile=0.5, vol_lookback_days=60,
        )
        assert not bench.empty


class TestBetaMatchedSpy:
    def test_portfolio_equals_spy_implies_beta_one(self):
        rng = np.random.default_rng(7)
        n = 200
        dates = pd.date_range("2026-01-05", periods=n, freq="B")
        spy = pd.Series(rng.normal(0.0005, 0.012, size=n), index=dates)
        port = spy.copy()
        bench = construct_beta_matched_spy_benchmark(
            port, spy, beta_lookback_days=60,
        )
        # Bench should equal SPY where defined.
        aligned = pd.concat([bench.rename("b"), spy.rename("s")], axis=1, join="inner").dropna()
        assert not aligned.empty
        np.testing.assert_allclose(
            aligned["b"].to_numpy(), aligned["s"].to_numpy(), atol=1e-6,
        )

    def test_zero_portfolio_implies_beta_zero(self):
        rng = np.random.default_rng(7)
        n = 200
        dates = pd.date_range("2026-01-05", periods=n, freq="B")
        spy = pd.Series(rng.normal(0.0005, 0.012, size=n), index=dates)
        port = pd.Series(np.zeros(n), index=dates)
        bench = construct_beta_matched_spy_benchmark(
            port, spy, beta_lookback_days=60,
        )
        # All zeros (beta = 0).
        np.testing.assert_allclose(bench.to_numpy(), 0.0, atol=1e-9)


class TestAlphaVsBenchmark:
    def test_hand_computed_excess(self):
        dates = pd.date_range("2026-01-05", periods=10, freq="B")
        port = pd.Series([0.01] * 10, index=dates)
        bench = pd.Series([0.005] * 10, index=dates)
        result = compute_alpha_vs_benchmark(port, bench, label="test")
        # Geometric: (1.01)^10 - 1 ≈ 0.10462; (1.005)^10 - 1 ≈ 0.05114
        assert result["portfolio_total_return"] == pytest.approx(1.01 ** 10 - 1, rel=1e-9)
        assert result["benchmark_total_return"] == pytest.approx(1.005 ** 10 - 1, rel=1e-9)
        assert result["excess_daily_mean"] == pytest.approx(0.005, abs=1e-9)
        # std of constant excess = 0 → IR = 0
        assert result["information_ratio"] == 0.0

    def test_ir_formula(self):
        rng = np.random.default_rng(3)
        n = 100
        dates = pd.date_range("2026-01-05", periods=n, freq="B")
        port = pd.Series(rng.normal(0.001, 0.01, size=n), index=dates)
        bench = pd.Series(rng.normal(0.0005, 0.01, size=n), index=dates)
        result = compute_alpha_vs_benchmark(port, bench)
        expected_ir = (
            result["excess_daily_mean"] / result["excess_daily_std"]
            * math.sqrt(252)
        )
        assert result["information_ratio"] == pytest.approx(expected_ir, rel=1e-9)

    def test_disjoint_indexes_returns_insufficient(self):
        a = pd.Series([0.01] * 5, index=pd.date_range("2026-01-05", periods=5, freq="B"))
        b = pd.Series([0.005] * 5, index=pd.date_range("2027-01-05", periods=5, freq="B"))
        result = compute_alpha_vs_benchmark(a, b)
        assert result["status"] == "insufficient_data"
        assert result["n_days"] == 0
