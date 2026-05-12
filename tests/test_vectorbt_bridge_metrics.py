"""Tests for vectorbt_bridge — Sortino, CVaR, daily returns extension.

Pins:
  1. Sortino primitive matches hand-computed value on a known fixture.
  2. CVaR primitive matches hand-computed value on a known fixture.
  3. Sortino returns 0 when there's no downside (all-positive series).
  4. CVaR returns 0 when sample size < ceil(1/q).
  5. portfolio_stats() emits the new fields (sortino_ratio, cvar_95,
     daily_returns, daily_log_returns) without breaking existing fields.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from vectorbt_bridge import _compute_cvar, _compute_sortino_ratio


class TestSortino:
    def test_hand_computed_negative_sortino(self):
        # 5 small losses + 95 small gains.
        # mean = (-0.15 + 0.095) / 100 = -0.00055
        # downside_var = (0.0025+0.0016+0.0009+0.0004+0.0001) / 100 = 0.000055
        # sortino_daily = -0.00055 / sqrt(0.000055) ≈ -0.07416
        # annualized = sortino_daily * sqrt(252) ≈ -1.1772
        r = pd.Series([-0.05, -0.04, -0.03, -0.02, -0.01] + [0.001] * 95)
        sortino = _compute_sortino_ratio(r, target=0.0)
        expected = (-0.00055 / math.sqrt(0.000055)) * math.sqrt(252)
        assert sortino == pytest.approx(expected, rel=1e-6)

    def test_zero_downside_returns_zero(self):
        # All-positive series — no below-target days, undefined Sortino.
        r = pd.Series([0.01] * 50)
        assert _compute_sortino_ratio(r, target=0.0) == 0.0

    def test_short_series_returns_zero(self):
        assert _compute_sortino_ratio(pd.Series([0.01]), target=0.0) == 0.0
        assert _compute_sortino_ratio(pd.Series([], dtype=float), target=0.0) == 0.0

    def test_drops_nan(self):
        r = pd.Series([np.nan, -0.01, 0.02, np.nan, -0.01])
        # 3 valid: mean = 0.0, downside_var = (0.0001+0.0001)/3 = 6.67e-5
        # sortino_daily = 0 / sqrt(6.67e-5) = 0
        assert _compute_sortino_ratio(r, target=0.0) == 0.0


class TestCVaR:
    def test_hand_computed(self):
        # 5 known losses + 95 small gains. q=0.05, n=100, n_tail = 5.
        # CVaR = mean of worst 5 = (-0.05-0.04-0.03-0.02-0.01)/5 = -0.03
        r = pd.Series([-0.05, -0.04, -0.03, -0.02, -0.01] + [0.001] * 95)
        cvar = _compute_cvar(r, q=0.05)
        assert cvar == pytest.approx(-0.03, rel=1e-9)

    def test_short_series_returns_zero(self):
        # n=10, q=0.05 → min_n=20, insufficient.
        r = pd.Series([-0.05, 0.01] * 5)
        assert _compute_cvar(r, q=0.05) == 0.0

    def test_n_tail_at_least_one(self):
        # n=20, q=0.05 → min_n=20, n_tail = floor(20*0.05) = 1.
        # Worst observation = -0.10.
        r = pd.Series([-0.10] + [0.01] * 19)
        cvar = _compute_cvar(r, q=0.05)
        assert cvar == pytest.approx(-0.10, rel=1e-9)

    def test_invalid_q_raises(self):
        r = pd.Series([0.01] * 100)
        with pytest.raises(ValueError):
            _compute_cvar(r, q=0.0)
        with pytest.raises(ValueError):
            _compute_cvar(r, q=1.0)
        with pytest.raises(ValueError):
            _compute_cvar(r, q=-0.05)

    def test_drops_nan(self):
        r = pd.Series([np.nan] * 5 + [-0.05, -0.04, -0.03, -0.02, -0.01]
                       + [0.001] * 95)
        # After dropna: 100 obs, same as hand-computed test.
        assert _compute_cvar(r, q=0.05) == pytest.approx(-0.03, rel=1e-9)


class TestPortfolioStatsExtensions:
    """Smoke test that portfolio_stats() emits the new fields end-to-end.

    Builds a minimal real vectorbt Portfolio so the test exercises the
    actual integration (pf.returns() shape, dict assembly) rather than
    mocking it. Keeps the fixture small to avoid long compute time.
    """

    def _build_portfolio(self):
        import vectorbt as vbt

        dates = pd.date_range("2026-01-02", periods=10, freq="B")
        prices = pd.DataFrame(
            {
                "AAA": np.linspace(100.0, 110.0, num=10),
                "BBB": np.linspace(50.0, 48.0, num=10),
            },
            index=dates,
        )
        entries = pd.DataFrame(False, index=dates, columns=["AAA", "BBB"])
        exits = pd.DataFrame(False, index=dates, columns=["AAA", "BBB"])
        sizes = pd.DataFrame(0.0, index=dates, columns=["AAA", "BBB"])
        entries.iloc[0] = True
        sizes.iloc[0] = [100.0, 100.0]
        exits.iloc[-1] = True

        return vbt.Portfolio.from_signals(
            close=prices,
            entries=entries,
            exits=exits,
            size=sizes,
            size_type="Amount",
            init_cash=100_000.0,
            cash_sharing=True,
            group_by=True,
            fees=0.0,
            freq="D",
        )

    def test_new_fields_present(self):
        try:
            from vectorbt_bridge import portfolio_stats
            pf = self._build_portfolio()
        except Exception:
            pytest.skip("vectorbt unavailable in this test env")
        stats = portfolio_stats(pf)

        # New fields
        assert "sortino_ratio" in stats
        assert "cvar_95" in stats
        assert "daily_returns" in stats
        assert "daily_log_returns" in stats

        assert isinstance(stats["sortino_ratio"], float)
        assert isinstance(stats["cvar_95"], float)
        assert isinstance(stats["daily_returns"], pd.Series)
        assert isinstance(stats["daily_log_returns"], pd.Series)

        # Pre-existing fields still present
        for key in (
            "total_return", "sharpe_ratio", "max_drawdown",
            "calmar_ratio", "total_trades", "win_rate",
            "spy_return", "total_alpha",
        ):
            assert key in stats, f"missing pre-existing field: {key}"

    def test_log_returns_match_log1p(self):
        try:
            from vectorbt_bridge import portfolio_stats
            pf = self._build_portfolio()
        except Exception:
            pytest.skip("vectorbt unavailable in this test env")
        stats = portfolio_stats(pf)
        r = stats["daily_returns"]
        log_r = stats["daily_log_returns"]
        # log(1 + r), guarded against r <= -1.
        np.testing.assert_allclose(
            log_r.to_numpy(),
            np.log1p(r.clip(lower=-0.999999).to_numpy()),
            rtol=1e-9,
        )

    def test_ew_high_vol_fields_default_none_when_kwarg_omitted(self):
        try:
            from vectorbt_bridge import portfolio_stats
            pf = self._build_portfolio()
        except Exception:
            pytest.skip("vectorbt unavailable in this test env")
        stats = portfolio_stats(pf)
        # Backward-compat: callers that don't pass the kwarg get None for
        # both new fields (parallel to spy_return / total_alpha pattern).
        assert stats["ew_high_vol_return"] is None
        assert stats["alpha_vs_ew_high_vol"] is None

    def test_ew_high_vol_basket_returns_emits_total_return_and_alpha(self):
        try:
            from vectorbt_bridge import portfolio_stats
            pf = self._build_portfolio()
        except Exception:
            pytest.skip("vectorbt unavailable in this test env")
        # Synthetic basket: constant +0.1% daily, indexed to the portfolio's
        # active dates. Compounded over 10 days → ~1.0045 → ~0.45% total.
        basket = pd.Series(
            0.001, index=pf.wrapper.index, name="ew_high_vol",
        )
        stats = portfolio_stats(pf, ew_high_vol_basket_returns=basket)
        assert stats["ew_high_vol_return"] is not None
        assert stats["alpha_vs_ew_high_vol"] is not None
        # Compounded basket return matches (1.001 ** 10) - 1 over the 10
        # active days.
        expected_basket_total = float((1.001 ** len(pf.wrapper.index)) - 1.0)
        assert stats["ew_high_vol_return"] == pytest.approx(
            expected_basket_total, rel=1e-6,
        )
        # Excess return shape parallels total_alpha (portfolio_return -
        # basket_return).
        assert stats["alpha_vs_ew_high_vol"] == pytest.approx(
            stats["total_return"] - expected_basket_total, rel=1e-6,
        )

    def test_ew_high_vol_aligns_only_overlapping_dates(self):
        try:
            from vectorbt_bridge import portfolio_stats
            pf = self._build_portfolio()
        except Exception:
            pytest.skip("vectorbt unavailable in this test env")
        # Basket extends beyond portfolio's window — reindex should
        # narrow to portfolio dates, NaN-extending dates dropped.
        extended_dates = pd.date_range(
            pf.wrapper.index[0] - pd.Timedelta(days=5),
            pf.wrapper.index[-1] + pd.Timedelta(days=5),
            freq="B",
        )
        basket = pd.Series(0.002, index=extended_dates, name="ew_high_vol")
        stats = portfolio_stats(pf, ew_high_vol_basket_returns=basket)
        # Compounded over only the portfolio's own dates (10 days), not
        # the extended 20-day series.
        expected = float((1.002 ** len(pf.wrapper.index)) - 1.0)
        assert stats["ew_high_vol_return"] == pytest.approx(expected, rel=1e-6)

    def test_ew_high_vol_insufficient_overlap_returns_none(self):
        try:
            from vectorbt_bridge import portfolio_stats
            pf = self._build_portfolio()
        except Exception:
            pytest.skip("vectorbt unavailable in this test env")
        # Basket dates don't overlap portfolio dates at all — fewer than 2
        # aligned points → None (graceful degrade, parallel to spy_return
        # short-history path).
        non_overlap_dates = pd.date_range(
            pf.wrapper.index[-1] + pd.Timedelta(days=30),
            periods=5, freq="B",
        )
        basket = pd.Series(0.001, index=non_overlap_dates, name="ew_high_vol")
        stats = portfolio_stats(pf, ew_high_vol_basket_returns=basket)
        assert stats["ew_high_vol_return"] is None
        assert stats["alpha_vs_ew_high_vol"] is None
