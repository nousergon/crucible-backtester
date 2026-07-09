"""Tests for analysis.risk_matched_benchmark.construct_ew_universe_benchmark
(config#834 — full-universe equal-weight benchmark leg).

This is new local code (not just the re-export shim over nousergon-lib): a
standalone rebalance/hold construction (mirrors
construct_ew_high_vol_benchmark's bookkeeping, minus the vol-ranking step)
rather than a vol-quantile-based selection. An earlier version delegated to
construct_ew_high_vol_benchmark with an epsilon vol_quantile intended to
select "everyone" — that approach was reverted because pandas'
Series.quantile() uses linear interpolation by default, so even
vol.quantile(1e-15) lands measurably above vol.min() for non-tied floats,
silently excluding the single lowest-vol ticker every time (see
test_epsilon_quantile_delegation_would_have_excluded_min_vol_ticker below,
which pins the exact failure mode that was caught in review). Pins:
  1. Full-universe basket includes EVERY ticker, including the single
     lowest-vol one (the exact case the epsilon-quantile approach got wrong).
  2. Full-universe basket tracks the naive equal-weight mean almost exactly
     (no vol-lookback gating to introduce a startup lag or selection bias).
  3. Distinct from construct_ew_high_vol_benchmark's top-quartile output on
     a universe with dispersed per-ticker volatility.
  4. Respects an explicit `universe` subset argument (same contract as
     construct_ew_high_vol_benchmark).
  5. Output Series is named "ew_universe".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.risk_matched_benchmark import (
    construct_ew_high_vol_benchmark,
    construct_ew_universe_benchmark,
)


def _make_dispersed_vol_prices(n_tickers: int = 8, n_days: int = 200, seed: int = 7) -> pd.DataFrame:
    """Universe with clearly separated per-ticker volatility so a
    vol-quartile selection and a full-universe selection provably differ."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n_days)
    data = {}
    for i in range(n_tickers):
        vol = 0.002 * (i + 1)  # vol strictly increasing with ticker index
        data[f"T{i}"] = 100.0 * np.cumprod(1.0 + rng.normal(0.0, vol, size=n_days))
    return pd.DataFrame(data, index=dates)


class TestConstructEwUniverseBenchmark:
    def test_returns_named_series_no_nan(self):
        prices = _make_dispersed_vol_prices()
        out = construct_ew_universe_benchmark(prices)
        assert isinstance(out, pd.Series)
        assert out.name == "ew_universe"
        assert not out.isna().any()
        assert len(out) > 0

    def test_differs_from_top_quartile_high_vol_basket(self):
        """The whole point of config#834's new leg: full-universe EW must be
        a genuinely different series from the existing top-vol-quartile
        basket on a universe with dispersed volatility."""
        prices = _make_dispersed_vol_prices()
        full = construct_ew_universe_benchmark(prices)
        top_q = construct_ew_high_vol_benchmark(prices)
        # Same rebalance schedule (same index construction), but different
        # per-segment membership -> different daily values on at least one
        # overlapping date.
        common = full.index.intersection(top_q.index)
        assert len(common) > 0
        assert not np.allclose(
            full.loc[common].to_numpy(), top_q.loc[common].to_numpy(),
        )

    def test_approximates_hand_computed_equal_weight_mean(self):
        """Sanity-check the 'full universe' claim: since there's no vol-based
        selection step (unlike construct_ew_high_vol_benchmark, this
        constructor has no vol_lookback_days gating — every ticker
        qualifies from day one), the full-universe basket's return should
        equal the naive equal-weight mean of ALL tickers' returns almost
        exactly (small deviations only possible right at rebalance-segment
        boundaries where the "hold from rd+1" convention shifts which day
        contributes)."""
        prices = _make_dispersed_vol_prices(n_tickers=6, n_days=150)
        full = construct_ew_universe_benchmark(prices)
        hand_mean = prices.pct_change().mean(axis=1)
        aligned = pd.concat(
            [full.rename("full"), hand_mean.rename("hand")], axis=1, join="inner",
        ).dropna()
        assert len(aligned) > 100
        np.testing.assert_allclose(
            aligned["full"].to_numpy(), aligned["hand"].to_numpy(), rtol=1e-9,
        )

    def test_respects_explicit_universe_subset(self):
        prices = _make_dispersed_vol_prices(n_tickers=8)
        subset = ["T0", "T1", "T2"]
        out = construct_ew_universe_benchmark(prices, universe=subset)
        assert isinstance(out, pd.Series)
        assert not out.empty

    def test_rejects_universe_not_in_prices(self):
        prices = _make_dispersed_vol_prices(n_tickers=3)
        with pytest.raises(ValueError):
            construct_ew_universe_benchmark(prices, universe=["NOT_A_TICKER"])

    def test_min_vol_ticker_is_included(self):
        """Regression pin for the epsilon-vol_quantile bug caught in review:
        the lowest-vol ticker in the universe must contribute to every
        rebalance segment, not just the higher-vol ones."""
        prices = _make_dispersed_vol_prices(n_tickers=10)
        full = construct_ew_universe_benchmark(prices)
        # T0 has the lowest vol (0.002 * (0+1)) by construction. Drop T0 and
        # rebuild — if T0 were excluded from the real basket, dropping it
        # would produce an IDENTICAL series; it must not.
        without_min_vol = construct_ew_universe_benchmark(
            prices, universe=[c for c in prices.columns if c != "T0"],
        )
        common = full.index.intersection(without_min_vol.index)
        assert len(common) > 0
        assert not np.allclose(
            full.loc[common].to_numpy(), without_min_vol.loc[common].to_numpy(),
        ), "excluding the min-vol ticker changed nothing — it wasn't included to begin with"

    def test_epsilon_quantile_delegation_would_have_excluded_min_vol_ticker(self):
        """Documents WHY construct_ew_universe_benchmark is not implemented
        as construct_ew_high_vol_benchmark(vol_quantile=epsilon): pandas'
        default linear-interpolation quantile lands above the true minimum
        for non-tied floats, so an epsilon threshold silently drops exactly
        the lowest-vol ticker. This test pins that pandas behavior itself
        (not this module's code) so a future refactor back toward the
        delegation approach fails loudly instead of silently reintroducing
        the bug."""
        rng = np.random.default_rng(3)
        vol = pd.Series(rng.uniform(0.01, 0.05, size=20))
        threshold = vol.quantile(1e-9)
        assert threshold > vol.min(), (
            "if this ever becomes False, pandas' quantile() interpolation "
            "behavior changed and the epsilon-delegation approach may be "
            "safe again — until then, construct_ew_universe_benchmark must "
            "stay a standalone construction, not a delegation"
        )
