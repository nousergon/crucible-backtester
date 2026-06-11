"""Tests for leg (c) of the L4593 backtester correctness battery — golden
benchmarks against external ground truth + corporate-action correctness.

Legs (a) and (b) check the engine against *itself* (null distribution centers on
zero) and against an *independent oracle* (reference_sim). Leg (c) adds the third
anchor: **reality**. It pins the production path to numbers that exist outside this
codebase — a published index total return and a real stock split — so a systematic
compounding/scale bug that both the engine AND the oracle could share (e.g. a wrong
annualization or a units error) still gets caught, and so the split-adjustment data
contract the backtester depends on is enforced.

Fixtures (`tests/fixtures/golden/`) are FROZEN real adjusted-close series pulled
once from Yahoo Finance (auto_adjust=True → dividend+split adjusted). The tests are
deterministic and offline; re-pull only to extend coverage.

1. **SPY 2023 buy-and-hold** — the adjusted series reproduces the independently
   published SPY 2023 total return (~+26.3%), and the production sim holding SPY
   all year reproduces that return.
2. **NVDA 10:1 split (2024-06-10)** — the adjusted series is CONTINUOUS through the
   split (no phantom −90% return); the sim shows no phantom loss. The raw
   (unadjusted) series DOES show the −90% drop — proving both that the detector has
   teeth and that the sim faithfully propagates whatever data it is given, so split
   correctness lives in the data contract (adjusted OHLCV from ArcticDB).

A FAILURE here means the production path disagrees with reality, or unadjusted data
has leaked into the sim. Record which in EXPERIMENTS.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthetic.reference_sim import simulate_reference
from vectorbt_bridge import orders_to_portfolio, portfolio_stats

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"
INIT_CASH = 1_000_000.0

# ── External ground truth (documented, sourced outside this repo) ────────────
# SPY 2023 total return (dividends reinvested) was widely reported at ≈ +26.3%
# (sources range ~26.1–26.4%). The frozen adjusted fixture yields +26.71%; we
# require agreement to within 1 absolute pct-point of the headline figure.
SPY_2023_TOTAL_RETURN_PUBLISHED = 0.263
SPY_2023_PUBLISHED_TOLERANCE = 0.01

# NVDA executed a 10-for-1 forward split effective 2024-06-10.
NVDA_SPLIT_DATE = pd.Timestamp("2024-06-10")
NVDA_SPLIT_RATIO = 10.0

# A real single-day return never approaches the magnitude of an unadjusted split
# jump; this band cleanly separates "normal market move" from "phantom".
NORMAL_DAILY_RETURN_CEILING = 0.15


def _load_golden(name: str) -> pd.Series:
    df = pd.read_csv(GOLDEN_DIR / name, parse_dates=["date"])
    return pd.Series(
        df["adj_close"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(df["date"]),
        name="adj_close",
    )


def _buy_and_hold_orders(prices: pd.Series, ticker: str) -> tuple[list[dict], pd.DataFrame]:
    """Buy ~fully-invested at the first close, hold to the end (no exit)."""
    p0 = float(prices.iloc[0])
    shares = float(int(INIT_CASH // p0))
    matrix = prices.to_frame(name=ticker)
    orders = [{
        "date": prices.index[0].strftime("%Y-%m-%d"),
        "ticker": ticker,
        "action": "ENTER",
        "shares": shares,
        "price_at_order": p0,
    }]
    return orders, matrix


# ════════════════════════════════════════════════════════════════════════════
# 1. SPY 2023 buy-and-hold vs published total return
# ════════════════════════════════════════════════════════════════════════════

class TestSpyBuyAndHoldGolden:
    def test_fixture_matches_published_total_return(self):
        spy = _load_golden("spy_2023_adj_close.csv")
        assert len(spy) >= 240, "expected a full trading year of SPY closes"
        fixture_return = float(spy.iloc[-1] / spy.iloc[0] - 1.0)
        # External ground truth: the frozen data agrees with the published figure.
        assert fixture_return == pytest.approx(
            SPY_2023_TOTAL_RETURN_PUBLISHED, abs=SPY_2023_PUBLISHED_TOLERANCE
        ), f"fixture total return {fixture_return:.4f} drifted from published"

    def test_sim_reproduces_buy_and_hold_return(self):
        spy = _load_golden("spy_2023_adj_close.csv")
        asset_return = float(spy.iloc[-1] / spy.iloc[0] - 1.0)
        orders, matrix = _buy_and_hold_orders(spy, "SPY")

        pf = orders_to_portfolio(orders, matrix, init_cash=INIT_CASH, fees=0.0)
        st = portfolio_stats(pf)
        ref = simulate_reference(orders, matrix, init_cash=INIT_CASH, fees=0.0)

        # Fully-invested buy-and-hold reproduces the asset return (tiny dilution
        # from share flooring against a 365-dollar price + $1M capital).
        assert st["total_return"] == pytest.approx(asset_return, abs=2e-4)
        # And matches the independent oracle exactly.
        assert st["total_return"] == pytest.approx(ref.total_return, rel=1e-9, abs=1e-12)

    def test_sim_return_within_published_band(self):
        spy = _load_golden("spy_2023_adj_close.csv")
        orders, matrix = _buy_and_hold_orders(spy, "SPY")
        pf = orders_to_portfolio(orders, matrix, init_cash=INIT_CASH, fees=0.0)
        st = portfolio_stats(pf)
        # The full chain (orders → vectorbt → stats) lands within a point of the
        # published SPY 2023 total return — catches a systematic scale/compounding bug.
        assert abs(st["total_return"] - SPY_2023_TOTAL_RETURN_PUBLISHED) < SPY_2023_PUBLISHED_TOLERANCE


# ════════════════════════════════════════════════════════════════════════════
# 2. NVDA 10:1 split — no phantom return through the corporate action
# ════════════════════════════════════════════════════════════════════════════

class TestCorporateActionNoPhantom:
    def _adjusted(self) -> pd.Series:
        return _load_golden("nvda_2024_split_adj_close.csv")

    def _raw_unadjusted(self) -> pd.Series:
        # Reconstruct the unadjusted series: a 10:1 forward split divides all
        # PRE-split historical prices by the ratio, so raw = adjusted × ratio
        # before the split date, unchanged on/after it.
        adj = self._adjusted()
        raw = adj.copy()
        pre = raw.index < NVDA_SPLIT_DATE
        raw.loc[pre] = raw.loc[pre] * NVDA_SPLIT_RATIO
        return raw

    def test_split_window_present(self):
        adj = self._adjusted()
        assert NVDA_SPLIT_DATE in adj.index
        assert adj.index.min() < NVDA_SPLIT_DATE < adj.index.max()

    def test_adjusted_series_continuous_through_split(self):
        adj = self._adjusted()
        rets = adj.pct_change().dropna()
        # No day — least of all the split date — shows a phantom move.
        assert rets.abs().max() < NORMAL_DAILY_RETURN_CEILING
        split_ret = float(adj.loc[NVDA_SPLIT_DATE] / adj.loc[adj.index[adj.index.get_loc(NVDA_SPLIT_DATE) - 1]] - 1.0)
        assert abs(split_ret) < 0.05  # the real cross-split move was ~+0.7%

    def test_raw_series_exhibits_phantom_drop(self):
        # The detector has teeth: unadjusted data WOULD show the ~−90% phantom.
        raw = self._raw_unadjusted()
        split_pos = raw.index.get_loc(NVDA_SPLIT_DATE)
        phantom_ret = float(raw.iloc[split_pos] / raw.iloc[split_pos - 1] - 1.0)
        assert phantom_ret < -0.85, f"expected ~−90% phantom, got {phantom_ret:.3f}"

    def test_sim_buy_and_hold_no_phantom_loss(self):
        adj = self._adjusted()
        asset_return = float(adj.iloc[-1] / adj.iloc[0] - 1.0)
        orders, matrix = _buy_and_hold_orders(adj, "NVDA")

        pf = orders_to_portfolio(orders, matrix, init_cash=INIT_CASH, fees=0.0)
        st = portfolio_stats(pf)
        ref = simulate_reference(orders, matrix, init_cash=INIT_CASH, fees=0.0)

        # No single-day portfolio return craters through the split.
        daily = pf.returns().dropna()
        assert daily.min() > -NORMAL_DAILY_RETURN_CEILING
        # Total return tracks the (continuous) adjusted asset return, no phantom loss.
        assert st["total_return"] == pytest.approx(asset_return, abs=5e-4)
        assert st["total_return"] == pytest.approx(ref.total_return, rel=1e-9, abs=1e-12)
        assert st["total_return"] > 0  # NVDA rose ~13% across this window

    def test_sim_faithfully_propagates_unadjusted_phantom(self):
        # Contrast: feed the SAME sim the RAW series and the phantom appears —
        # proving the sim is a faithful propagator and split correctness lives in
        # the data contract (adjusted OHLCV), not the engine.
        raw = self._raw_unadjusted()
        orders, matrix = _buy_and_hold_orders(raw, "NVDA")
        pf = orders_to_portfolio(orders, matrix, init_cash=INIT_CASH, fees=0.0)
        daily = pf.returns().dropna()
        assert daily.min() < -0.85, "raw series should drive a phantom ~−90% sim day"
