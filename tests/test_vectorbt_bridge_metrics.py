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
            from vectorbt_bridge import _compute_active_window, portfolio_stats
            pf = self._build_portfolio()
        except Exception:
            pytest.skip("vectorbt unavailable in this test env")
        # Synthetic basket: constant +0.1% daily, indexed to the portfolio's
        # active dates. Compounded over the active window only (post-fix
        # 2026-05-24) — the entry fills on day 0 but NAV stays flat until
        # day 1 when prices move, so the active window typically excludes
        # the entry-day itself.
        basket = pd.Series(
            0.001, index=pf.wrapper.index, name="ew_high_vol",
        )
        stats = portfolio_stats(pf, ew_high_vol_basket_returns=basket)
        assert stats["ew_high_vol_return"] is not None
        assert stats["alpha_vs_ew_high_vol"] is not None
        # Determine the actual active window length (varies by 1 with vbt's
        # fill semantics) instead of hardcoding `len(pf.wrapper.index)` —
        # the pre-fix test was pinning the buggy "compound over the full
        # wrapper" semantic.
        window = _compute_active_window(pf)
        assert window is not None
        active_dates = pf.wrapper.index[
            (pf.wrapper.index >= window[0]) & (pf.wrapper.index <= window[1])
        ]
        expected_basket_total = float((1.001 ** len(active_dates)) - 1.0)
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
            from vectorbt_bridge import _compute_active_window, portfolio_stats
            pf = self._build_portfolio()
        except Exception:
            pytest.skip("vectorbt unavailable in this test env")
        # Basket extends beyond portfolio's window — active-window narrowing
        # restricts the compound to the portfolio's active dates only.
        extended_dates = pd.date_range(
            pf.wrapper.index[0] - pd.Timedelta(days=5),
            pf.wrapper.index[-1] + pd.Timedelta(days=5),
            freq="B",
        )
        basket = pd.Series(0.002, index=extended_dates, name="ew_high_vol")
        stats = portfolio_stats(pf, ew_high_vol_basket_returns=basket)
        # Compounded over only the portfolio's ACTIVE window (post-fix
        # 2026-05-24) — same active-window-length adjustment as above.
        window = _compute_active_window(pf)
        assert window is not None
        active_dates = pf.wrapper.index[
            (pf.wrapper.index >= window[0]) & (pf.wrapper.index <= window[1])
        ]
        expected = float((1.002 ** len(active_dates)) - 1.0)
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


class TestActiveWindowAnchoring:
    """Pin the active-window-anchored benchmark comparison fix (2026-05-24).

    Pre-fix bug: ``portfolio_stats`` compared ``total_return`` (effectively
    the portfolio's active-window P&L) against ``spy_return`` /
    ``ew_high_vol_return`` compounded over the FULL ``pf.wrapper.index``
    (typically a 10-year price-matrix when the simulator only traded the
    final weeks). The 2026-05-24 backtester run surfaced this as
    ``ew_high_vol_return: 960%`` (10y basket compound) /
    ``alpha_vs_ew_high_vol: -954%`` against a portfolio that had a
    1.7% active-window return. The fix anchors both benchmark legs on
    the portfolio's active window (first non-flat NAV through last
    wrapper date) so the comparison is apples-to-apples.
    """

    def _build_flat_prefix_portfolio(self, n_flat: int = 20, n_active: int = 10):
        """Portfolio that holds no positions for ``n_flat`` days then trades
        for ``n_active`` days. NAV is constant at initial cash through the
        flat prefix, then changes once entries fire.
        """
        import vectorbt as vbt

        total = n_flat + n_active
        dates = pd.date_range("2026-01-02", periods=total, freq="B")
        prices = pd.DataFrame(
            {
                "AAA": np.concatenate([
                    np.full(n_flat, 100.0),
                    np.linspace(100.0, 110.0, num=n_active),
                ]),
            },
            index=dates,
        )
        entries = pd.DataFrame(False, index=dates, columns=["AAA"])
        exits = pd.DataFrame(False, index=dates, columns=["AAA"])
        sizes = pd.DataFrame(0.0, index=dates, columns=["AAA"])
        # Enter on first active day; exit on last day.
        entries.iloc[n_flat] = True
        sizes.iloc[n_flat] = [100.0]
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

    def test_compute_active_window_skips_flat_prefix(self):
        try:
            from vectorbt_bridge import _compute_active_window
            pf = self._build_flat_prefix_portfolio(n_flat=20, n_active=10)
        except Exception:
            pytest.skip("vectorbt unavailable in this test env")
        window = _compute_active_window(pf)
        assert window is not None, "active window should not be None for trading portfolio"
        active_start, active_end = window
        wrapper_index = pf.wrapper.index
        # active_start must be AFTER the 20-day flat prefix (the entry fires
        # at iloc[20]); active_end must be the final wrapper date.
        assert active_start > wrapper_index[19], (
            f"active_start={active_start} should be after the flat prefix's last "
            f"date {wrapper_index[19]}"
        )
        assert active_end == wrapper_index[-1]

    def test_compute_active_window_none_when_no_trades(self):
        try:
            import vectorbt as vbt
            from vectorbt_bridge import _compute_active_window
        except Exception:
            pytest.skip("vectorbt unavailable in this test env")
        # All-False entries — portfolio never trades; NAV stays constant.
        dates = pd.date_range("2026-01-02", periods=10, freq="B")
        prices = pd.DataFrame(
            {"AAA": np.linspace(100.0, 110.0, num=10)}, index=dates,
        )
        entries = pd.DataFrame(False, index=dates, columns=["AAA"])
        exits = pd.DataFrame(False, index=dates, columns=["AAA"])
        sizes = pd.DataFrame(0.0, index=dates, columns=["AAA"])
        pf = vbt.Portfolio.from_signals(
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
        assert _compute_active_window(pf) is None

    def test_ew_high_vol_compounds_over_active_window_not_full_wrapper(self):
        """The bug class: a 20-day-flat + 10-day-active portfolio compared
        against a 30-day basket should compound the basket over the 10-day
        active window, NOT the full 30-day wrapper.
        """
        try:
            from vectorbt_bridge import portfolio_stats
            pf = self._build_flat_prefix_portfolio(n_flat=20, n_active=10)
        except Exception:
            pytest.skip("vectorbt unavailable in this test env")
        # Basket: +0.1% daily for all 30 wrapper dates.
        basket = pd.Series(0.001, index=pf.wrapper.index, name="ew_high_vol")
        stats = portfolio_stats(pf, ew_high_vol_basket_returns=basket)

        # Pre-fix would compound (1.001 ** 30) - 1 ≈ 3.04%; post-fix compounds
        # ONLY over the active window starting at the first NAV change, which
        # is approximately 10 dates → (1.001 ** ~10) - 1 ≈ ~1.0%. Concretely
        # the basket return must be substantially SMALLER than the full-wrapper
        # compound. We assert the active-window result is closer in magnitude
        # to a 10-day compound than to a 30-day compound.
        ten_day_compound = (1.001 ** 10) - 1.0
        thirty_day_compound = (1.001 ** 30) - 1.0
        emitted = stats["ew_high_vol_return"]
        assert emitted is not None
        # Magnitude must be closer to the 10-day compound than the 30-day.
        # This is the load-bearing assertion that pins the active-window fix.
        assert abs(emitted - ten_day_compound) < abs(emitted - thirty_day_compound), (
            f"basket_return={emitted} is closer to the 30-day compound "
            f"{thirty_day_compound} than to the 10-day compound "
            f"{ten_day_compound} — active-window narrowing is not in effect"
        )

    def test_spy_return_anchored_on_active_window(self):
        """SPY-side mirror of the basket test: spy_return should reflect SPY's
        return over the portfolio's active window only, not the full 10y
        wrapper window.
        """
        try:
            from vectorbt_bridge import portfolio_stats
            pf = self._build_flat_prefix_portfolio(n_flat=20, n_active=10)
        except Exception:
            pytest.skip("vectorbt unavailable in this test env")
        # SPY: monotonically rising over the full 30-day wrapper from 100 → 130.
        # Full-wrapper return = (130/100) - 1 = 30%.
        # Active-window return must be SMALLER (SPY only rose from ~119 to
        # 130 across the active window, ≈ 9%).
        spy_prices = pd.Series(
            np.linspace(100.0, 130.0, num=len(pf.wrapper.index)),
            index=pf.wrapper.index,
            name="SPY",
        )
        stats = portfolio_stats(pf, spy_prices=spy_prices)
        assert stats["spy_return"] is not None
        full_wrapper_return = (130.0 / 100.0) - 1.0  # 30%
        # Active-window SPY return must be materially smaller than the full
        # 30% wrapper compound.
        assert stats["spy_return"] < full_wrapper_return * 0.6, (
            f"spy_return={stats['spy_return']} is not materially smaller than "
            f"the full-wrapper return {full_wrapper_return} — active-window "
            f"narrowing is not in effect"
        )

    def test_null_legs_listed_when_legs_missing(self):
        try:
            from vectorbt_bridge import portfolio_stats
            pf = self._build_flat_prefix_portfolio(n_flat=20, n_active=10)
        except Exception:
            pytest.skip("vectorbt unavailable in this test env")
        # No spy_prices, no basket — both legs degrade to None and the
        # surface ``null_legs`` lists every null field for caller alerting.
        stats = portfolio_stats(pf)
        assert stats.get("spy_return") is None
        assert stats.get("ew_high_vol_return") is None
        null_legs = stats.get("null_legs", [])
        assert "spy_return" in null_legs
        assert "total_alpha" in null_legs
        # basket-side legs were not requested (None passed), so they're
        # NOT in null_legs — the surface only fires for "requested but
        # could not compute" cases. spy_prices=None counts as "could not
        # compute" because spy_return is canonical / always-expected.

    def test_no_null_legs_when_legs_compute_cleanly(self):
        try:
            from vectorbt_bridge import portfolio_stats
            pf = self._build_flat_prefix_portfolio(n_flat=20, n_active=10)
        except Exception:
            pytest.skip("vectorbt unavailable in this test env")
        spy_prices = pd.Series(
            np.linspace(100.0, 130.0, num=len(pf.wrapper.index)),
            index=pf.wrapper.index,
            name="SPY",
        )
        basket = pd.Series(0.001, index=pf.wrapper.index, name="ew_high_vol")
        stats = portfolio_stats(
            pf, spy_prices=spy_prices, ew_high_vol_basket_returns=basket,
        )
        # All four legs compute cleanly — null_legs should be absent
        # (or empty) since no caller-action is owed.
        assert stats.get("null_legs", []) == []


class TestEwUniverseLeg:
    """config#834: full-universe equal-weight benchmark leg.

    Distinct from ``ew_high_vol_basket_returns`` (top vol-quartile only —
    see ``TestPortfolioStatsExtensions``): isolates stock-selection alpha
    from cap-weighted-tilt alpha. Mirrors the ``TestPortfolioStatsExtensions``
    shape (same fixtures, same default-None / active-window-anchoring /
    null_legs conventions) applied to the new
    ``ew_universe_basket_returns`` kwarg.
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

    def _build_flat_prefix_portfolio(self, n_flat: int = 20, n_active: int = 10):
        import vectorbt as vbt

        total = n_flat + n_active
        dates = pd.date_range("2026-01-02", periods=total, freq="B")
        prices = pd.DataFrame(
            {
                "AAA": np.concatenate([
                    np.full(n_flat, 100.0),
                    np.linspace(100.0, 110.0, num=n_active),
                ]),
            },
            index=dates,
        )
        entries = pd.DataFrame(False, index=dates, columns=["AAA"])
        exits = pd.DataFrame(False, index=dates, columns=["AAA"])
        sizes = pd.DataFrame(0.0, index=dates, columns=["AAA"])
        entries.iloc[n_flat] = True
        sizes.iloc[n_flat] = [100.0]
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

    def test_key_pair_present_and_distinct_from_ew_high_vol(self):
        """New ew_universe_return/alpha_vs_ew_universe keys exist and are
        independent of the pre-existing ew_high_vol_return/alpha_vs_ew_high_vol
        pair (config#834's core distinctness requirement)."""
        try:
            from vectorbt_bridge import portfolio_stats
            pf = self._build_portfolio()
        except Exception:
            pytest.skip("vectorbt unavailable in this test env")

        high_vol_basket = pd.Series(0.002, index=pf.wrapper.index, name="ew_high_vol")
        universe_basket = pd.Series(0.0005, index=pf.wrapper.index, name="ew_universe")
        stats = portfolio_stats(
            pf,
            ew_high_vol_basket_returns=high_vol_basket,
            ew_universe_basket_returns=universe_basket,
        )

        for key in ("ew_universe_return", "alpha_vs_ew_universe"):
            assert key in stats

        assert stats["ew_universe_return"] is not None
        assert stats["ew_high_vol_return"] is not None
        # Different input series (0.0005 vs 0.002 daily) must produce
        # different compounded totals — proves the two legs are wired to
        # independent kwargs, not aliases of the same computation.
        assert stats["ew_universe_return"] != pytest.approx(
            stats["ew_high_vol_return"], rel=1e-9,
        )
        assert stats["alpha_vs_ew_universe"] != pytest.approx(
            stats["alpha_vs_ew_high_vol"], rel=1e-9,
        )

    def test_default_none_when_kwarg_omitted(self):
        try:
            from vectorbt_bridge import portfolio_stats
            pf = self._build_portfolio()
        except Exception:
            pytest.skip("vectorbt unavailable in this test env")
        stats = portfolio_stats(pf)
        assert stats["ew_universe_return"] is None
        assert stats["alpha_vs_ew_universe"] is None
        # Opt-in leg (parallel to ew_high_vol): omitted kwarg is NOT a
        # null_legs entry — only "requested but failed to compute" is.
        assert "ew_universe_return" not in stats.get("null_legs", [])
        assert "alpha_vs_ew_universe" not in stats.get("null_legs", [])

    def test_emits_total_return_and_alpha(self):
        try:
            from vectorbt_bridge import _compute_active_window, portfolio_stats
            pf = self._build_portfolio()
        except Exception:
            pytest.skip("vectorbt unavailable in this test env")
        basket = pd.Series(0.001, index=pf.wrapper.index, name="ew_universe")
        stats = portfolio_stats(pf, ew_universe_basket_returns=basket)
        assert stats["ew_universe_return"] is not None
        assert stats["alpha_vs_ew_universe"] is not None

        window = _compute_active_window(pf)
        assert window is not None
        active_dates = pf.wrapper.index[
            (pf.wrapper.index >= window[0]) & (pf.wrapper.index <= window[1])
        ]
        expected_basket_total = float((1.001 ** len(active_dates)) - 1.0)
        assert stats["ew_universe_return"] == pytest.approx(
            expected_basket_total, rel=1e-6,
        )
        assert stats["alpha_vs_ew_universe"] == pytest.approx(
            stats["total_return"] - expected_basket_total, rel=1e-6,
        )

    def test_insufficient_overlap_returns_none_and_flags_null_legs(self):
        """Requested-but-uncomputable: basket dates don't overlap the
        portfolio's active window at all — degrades to None AND is flagged
        in null_legs (contrast with the omitted-kwarg case above)."""
        try:
            from vectorbt_bridge import portfolio_stats
            pf = self._build_portfolio()
        except Exception:
            pytest.skip("vectorbt unavailable in this test env")
        non_overlap_dates = pd.date_range(
            pf.wrapper.index[-1] + pd.Timedelta(days=30),
            periods=5, freq="B",
        )
        basket = pd.Series(0.001, index=non_overlap_dates, name="ew_universe")
        stats = portfolio_stats(pf, ew_universe_basket_returns=basket)
        assert stats["ew_universe_return"] is None
        assert stats["alpha_vs_ew_universe"] is None
        null_legs = stats.get("null_legs", [])
        assert "ew_universe_return" in null_legs
        assert "alpha_vs_ew_universe" in null_legs

    def test_compounds_over_active_window_not_full_wrapper(self):
        """Active-window-anchoring fix (2026-05-24) must cover the new leg
        too, not just SPY/ew_high_vol — this is the issue's explicit
        non-inferable gotcha."""
        try:
            from vectorbt_bridge import portfolio_stats
            pf = self._build_flat_prefix_portfolio(n_flat=20, n_active=10)
        except Exception:
            pytest.skip("vectorbt unavailable in this test env")
        basket = pd.Series(0.001, index=pf.wrapper.index, name="ew_universe")
        stats = portfolio_stats(pf, ew_universe_basket_returns=basket)

        ten_day_compound = (1.001 ** 10) - 1.0
        thirty_day_compound = (1.001 ** 30) - 1.0
        emitted = stats["ew_universe_return"]
        assert emitted is not None
        assert abs(emitted - ten_day_compound) < abs(emitted - thirty_day_compound), (
            f"ew_universe_return={emitted} is closer to the 30-day compound "
            f"{thirty_day_compound} than the 10-day compound {ten_day_compound} "
            "— active-window narrowing is not in effect for the new leg"
        )


class TestSectorEtfLegs:
    """config#834: per-sector-ETF benchmark legs.

    One ``alpha_vs_sector_<TICKER>`` (+ ``sector_<TICKER>_return``) pair per
    entry in the ``sector_etf_basket_returns`` dict — sector-relative alpha
    ("does the system's healthcare picks beat XLV?"). Same fixtures +
    conventions as ``TestPortfolioStatsExtensions``/``TestEwUniverseLeg``.
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

    def _build_flat_prefix_portfolio(self, n_flat: int = 20, n_active: int = 10):
        import vectorbt as vbt

        total = n_flat + n_active
        dates = pd.date_range("2026-01-02", periods=total, freq="B")
        prices = pd.DataFrame(
            {
                "AAA": np.concatenate([
                    np.full(n_flat, 100.0),
                    np.linspace(100.0, 110.0, num=n_active),
                ]),
            },
            index=dates,
        )
        entries = pd.DataFrame(False, index=dates, columns=["AAA"])
        exits = pd.DataFrame(False, index=dates, columns=["AAA"])
        sizes = pd.DataFrame(0.0, index=dates, columns=["AAA"])
        entries.iloc[n_flat] = True
        sizes.iloc[n_flat] = [100.0]
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

    def test_one_key_pair_per_sector_etf_present(self):
        """One alpha_vs_sector_<TICKER> key per ETF present in the run's
        universe — the acceptance criteria's headline assertion. Uses a
        3-sector subset of SECTOR_ETF_MAP/DEFAULT_SECTOR_ETF_MAP's ticker
        vocabulary (XLK/XLV/XLF) to prove the fan-out is driven by dict
        contents, not a hardcoded ticker list."""
        try:
            from vectorbt_bridge import portfolio_stats
            pf = self._build_portfolio()
        except Exception:
            pytest.skip("vectorbt unavailable in this test env")

        sector_baskets = {
            "XLK": pd.Series(0.001, index=pf.wrapper.index, name="XLK"),
            "XLV": pd.Series(0.0005, index=pf.wrapper.index, name="XLV"),
            "XLF": pd.Series(-0.0003, index=pf.wrapper.index, name="XLF"),
        }
        stats = portfolio_stats(pf, sector_etf_basket_returns=sector_baskets)

        for ticker in ("XLK", "XLV", "XLF"):
            assert f"alpha_vs_sector_{ticker}" in stats
            assert f"sector_{ticker}_return" in stats
            assert stats[f"sector_{ticker}_return"] is not None
            assert stats[f"alpha_vs_sector_{ticker}"] is not None

        # A sector ETF NOT in the run's universe must not appear at all.
        assert "alpha_vs_sector_XLE" not in stats
        assert "sector_XLE_return" not in stats

        # Distinct input series (0.001 vs 0.0005 vs -0.0003 daily) must
        # produce distinct compounded totals per ticker.
        returns = {t: stats[f"sector_{t}_return"] for t in ("XLK", "XLV", "XLF")}
        assert len(set(round(v, 8) for v in returns.values())) == 3

    def test_default_empty_when_kwarg_omitted(self):
        try:
            from vectorbt_bridge import portfolio_stats
            pf = self._build_portfolio()
        except Exception:
            pytest.skip("vectorbt unavailable in this test env")
        stats = portfolio_stats(pf)
        assert not any(k.startswith("alpha_vs_sector_") for k in stats)
        assert not any(
            k.startswith("sector_") and k.endswith("_return") for k in stats
        )
        assert stats.get("null_legs", []) == [] or all(
            not k.startswith(("alpha_vs_sector_", "sector_"))
            for k in stats.get("null_legs", [])
        )

    def test_empty_dict_same_as_omitted(self):
        try:
            from vectorbt_bridge import portfolio_stats
            pf = self._build_portfolio()
        except Exception:
            pytest.skip("vectorbt unavailable in this test env")
        stats = portfolio_stats(pf, sector_etf_basket_returns={})
        assert not any(k.startswith("alpha_vs_sector_") for k in stats)

    def test_per_ticker_emits_total_return_and_alpha(self):
        try:
            from vectorbt_bridge import _compute_active_window, portfolio_stats
            pf = self._build_portfolio()
        except Exception:
            pytest.skip("vectorbt unavailable in this test env")
        basket = pd.Series(0.0015, index=pf.wrapper.index, name="XLV")
        stats = portfolio_stats(pf, sector_etf_basket_returns={"XLV": basket})

        window = _compute_active_window(pf)
        assert window is not None
        active_dates = pf.wrapper.index[
            (pf.wrapper.index >= window[0]) & (pf.wrapper.index <= window[1])
        ]
        expected_total = float((1.0015 ** len(active_dates)) - 1.0)
        assert stats["sector_XLV_return"] == pytest.approx(expected_total, rel=1e-6)
        assert stats["alpha_vs_sector_XLV"] == pytest.approx(
            stats["total_return"] - expected_total, rel=1e-6,
        )

    def test_insufficient_overlap_returns_none_and_flags_null_legs(self):
        try:
            from vectorbt_bridge import portfolio_stats
            pf = self._build_portfolio()
        except Exception:
            pytest.skip("vectorbt unavailable in this test env")
        non_overlap_dates = pd.date_range(
            pf.wrapper.index[-1] + pd.Timedelta(days=30),
            periods=5, freq="B",
        )
        basket = pd.Series(0.001, index=non_overlap_dates, name="XLF")
        stats = portfolio_stats(pf, sector_etf_basket_returns={"XLF": basket})
        assert stats["sector_XLF_return"] is None
        assert stats["alpha_vs_sector_XLF"] is None
        null_legs = stats.get("null_legs", [])
        assert "sector_XLF_return" in null_legs
        assert "alpha_vs_sector_XLF" in null_legs

    def test_compounds_over_active_window_not_full_wrapper(self):
        """Active-window-anchoring fix must cover sector legs too."""
        try:
            from vectorbt_bridge import portfolio_stats
            pf = self._build_flat_prefix_portfolio(n_flat=20, n_active=10)
        except Exception:
            pytest.skip("vectorbt unavailable in this test env")
        basket = pd.Series(0.001, index=pf.wrapper.index, name="XLK")
        stats = portfolio_stats(pf, sector_etf_basket_returns={"XLK": basket})

        ten_day_compound = (1.001 ** 10) - 1.0
        thirty_day_compound = (1.001 ** 30) - 1.0
        emitted = stats["sector_XLK_return"]
        assert emitted is not None
        assert abs(emitted - ten_day_compound) < abs(emitted - thirty_day_compound), (
            f"sector_XLK_return={emitted} is closer to the 30-day compound "
            f"{thirty_day_compound} than the 10-day compound {ten_day_compound} "
            "— active-window narrowing is not in effect for sector legs"
        )

    def test_one_sector_null_does_not_null_another(self):
        """A sector ETF whose data doesn't overlap the active window emits
        None for its own pair without affecting a sibling sector ETF whose
        data does overlap — legs are independent, not all-or-nothing."""
        try:
            from vectorbt_bridge import portfolio_stats
            pf = self._build_portfolio()
        except Exception:
            pytest.skip("vectorbt unavailable in this test env")
        non_overlap_dates = pd.date_range(
            pf.wrapper.index[-1] + pd.Timedelta(days=30),
            periods=5, freq="B",
        )
        stats = portfolio_stats(
            pf,
            sector_etf_basket_returns={
                "XLF": pd.Series(0.001, index=non_overlap_dates, name="XLF"),
                "XLV": pd.Series(0.001, index=pf.wrapper.index, name="XLV"),
            },
        )
        assert stats["sector_XLF_return"] is None
        assert stats["sector_XLV_return"] is not None
        assert stats["alpha_vs_sector_XLV"] is not None


class TestComputeBenchmarkLegIndependentOfPortfolioGaps:
    """Regression pin (caught in review): a benchmark leg's total return
    must depend ONLY on the benchmark series' own aligned dates within the
    active window — never on the portfolio's own daily-returns coverage.

    ``compute_alpha_vs_benchmark`` inner-joins its two Series arguments
    before computing totals. An earlier version of ``_compute_benchmark_leg``
    passed the REAL portfolio daily-returns series as that function's
    ``portfolio_daily_returns`` argument; if the portfolio's returns had any
    gap inside the active window relative to the benchmark's dates, the join
    would silently truncate ``benchmark_total_return`` to the shorter
    overlap — even though the benchmark series itself had complete data for
    the window. Fixed by self-joining the benchmark against itself (the
    portfolio side of that inner call is irrelevant to `benchmark_total_return`,
    which is why ``_compute_benchmark_leg`` no longer accepts a
    ``portfolio_daily_returns`` argument at all).
    """

    def test_benchmark_total_return_unaffected_by_hypothetical_portfolio_gap(self):
        from vectorbt_bridge import _compute_benchmark_leg

        dates = pd.bdate_range("2026-01-01", periods=20)
        bench = pd.Series(0.001, index=dates, name="bench")
        active_window = (dates[0], dates[-1])

        bench_return, alpha = _compute_benchmark_leg(
            total_return=0.05,
            active_window=active_window,
            benchmark_daily_returns=bench,
            label="regression_pin",
        )
        expected = float((1.001 ** len(dates)) - 1.0)
        assert bench_return == pytest.approx(expected, rel=1e-9), (
            f"benchmark_total_return={bench_return} does not match the "
            f"benchmark-only compound {expected} — a portfolio-side "
            "dependency has crept back into the benchmark-total computation"
        )
        assert alpha == pytest.approx(0.05 - expected, rel=1e-9)

    def test_signature_has_no_portfolio_daily_returns_param(self):
        """Guards against silently reintroducing the portfolio-coupling: if
        a future refactor adds a `portfolio_daily_returns` parameter back to
        `_compute_benchmark_leg`, this test forces a conscious decision
        (update this test + re-verify the gap-independence property) rather
        than a silent behavior change."""
        import inspect

        from vectorbt_bridge import _compute_benchmark_leg

        params = set(inspect.signature(_compute_benchmark_leg).parameters)
        assert "portfolio_daily_returns" not in params, (
            "_compute_benchmark_leg regained a portfolio_daily_returns "
            "parameter — re-verify it can't truncate benchmark_total_return "
            "via compute_alpha_vs_benchmark's inner join (see "
            "test_benchmark_total_return_unaffected_by_hypothetical_portfolio_gap)"
        )
