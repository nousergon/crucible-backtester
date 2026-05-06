"""
risk_matched_benchmark.py — alpha vs comparable-risk baselines.

Replaces "beat SPY" as the headline alpha number. Per
evaluator-revamp-260506.md, the question being asked is *not* "did you
beat the market?" but "given how much risk you took, did you outperform
the dumb version of taking that risk?" Two complementary baselines:

1. **Equal-weight high-vol subset** — top vol-quartile of the population
   (~25 tracked stocks), equal-weight, rolled weekly. The within-universe
   selection grade. Asks: "given you're fishing in volatile waters, did
   you pick the good fish?"

2. **Beta-matched SPY** — synthetic position of size (portfolio_beta ×
   SPY) constructed each rebalance window. The pure-alpha grade. Asks:
   "did you outperform the dumb version of being this exposed?"

Both produce a daily return series for the benchmark, alongside the
input portfolio's daily return series; difference = excess return that
*can't* be explained by the risk you took. Information Ratio is
computed against each benchmark independently.

Pure-compute. Operates on price + portfolio-return series; no I/O.
"""

from __future__ import annotations

import logging
import math
from typing import TypedDict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252


class BenchmarkResult(TypedDict, total=False):
    status: str
    n_days: int
    benchmark_total_return: float
    portfolio_total_return: float
    excess_return: float                # portfolio - benchmark, total
    excess_daily_mean: float            # mean daily excess
    excess_daily_std: float
    information_ratio: float            # annualized excess / std of excess
    benchmark_daily_returns: pd.Series
    excess_daily_returns: pd.Series
    label: str                          # "ew_high_vol" | "beta_matched_spy"


# ── Equal-weight high-vol subset ────────────────────────────────────────────


def construct_ew_high_vol_benchmark(
    prices: pd.DataFrame,
    universe: list[str] | None = None,
    vol_quantile: float = 0.75,
    vol_lookback_days: int = 60,
    rebalance_freq: str = "W-MON",
) -> pd.Series:
    """Build daily return series for an equal-weight high-vol subset.

    Each rebalance date: rank tickers in ``universe`` by trailing
    realized vol over ``vol_lookback_days``, take the top quartile (or
    whatever ``vol_quantile`` selects), equal-weight them, hold until
    next rebalance.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily close prices. DatetimeIndex (trading days). Columns =
        tickers.
    universe : list[str] | None
        Tickers to select from. Default = all columns of ``prices``.
        Pass the agent's actual decision universe (the population the
        agents pick from) for a fair counterfactual.
    vol_quantile : float
        Vol-percentile threshold. Default 0.75 = top quartile. Use 0.5
        for top half, 0.9 for top decile, etc.
    vol_lookback_days : int
        Trailing window for realized vol estimation. Default 60 ≈ 3
        months — long enough to be stable, short enough to reflect
        regime shifts.
    rebalance_freq : str
        Pandas offset alias. Default ``"W-MON"`` (every Monday). Use
        ``"M"`` for monthly, ``"D"`` for daily (warning: high turnover).

    Returns
    -------
    pd.Series of daily simple returns indexed by trading day. NaN-free
    (rebalance days that produce zero-member subsets are dropped).
    """
    if not (0.0 < vol_quantile < 1.0):
        raise ValueError(f"vol_quantile must be in (0, 1), got {vol_quantile}")
    if vol_lookback_days < 5:
        raise ValueError(f"vol_lookback_days must be >= 5, got {vol_lookback_days}")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices.index must be a DatetimeIndex")

    cols = list(prices.columns) if universe is None else [
        t for t in universe if t in prices.columns
    ]
    if not cols:
        raise ValueError("universe is empty after intersecting with prices.columns")

    sub = prices[cols]
    daily_returns = sub.pct_change()

    # Rebalance dates: first trading day on or after each period boundary.
    # `to_period(rebalance_freq)` then `.start_time` gives the period boundary;
    # we snap to the next trading day in the price index.
    period_starts = pd.Series(sub.index).dt.to_period(rebalance_freq).dt.start_time.unique()
    rebalance_dates: list[pd.Timestamp] = []
    seen: set[pd.Timestamp] = set()
    for ps in period_starts:
        candidates = sub.index[sub.index >= ps]
        if len(candidates) > 0:
            d = candidates[0]
            if d not in seen and d in sub.index:
                rebalance_dates.append(d)
                seen.add(d)

    benchmark_segments: list[pd.Series] = []
    for i, rd in enumerate(rebalance_dates):
        # Compute trailing vol on returns up to rd.
        rd_pos = sub.index.get_loc(rd)
        if rd_pos < vol_lookback_days:
            continue  # not enough history yet
        lookback = daily_returns.iloc[rd_pos - vol_lookback_days : rd_pos]
        vol = lookback.std(ddof=1)  # per-ticker realized vol
        vol = vol.dropna()
        if vol.empty:
            continue
        threshold = vol.quantile(vol_quantile)
        selected = vol[vol >= threshold].index.tolist()
        if not selected:
            continue

        # Hold from rd until next rebalance (exclusive) or end of data.
        next_rd = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None
        end_pos = (
            sub.index.get_loc(next_rd)
            if next_rd is not None
            else len(sub.index)
        )
        # Daily returns over [rd+1, next_rd] — first day post-rebalance
        # is the first day the held basket starts compounding.
        segment = daily_returns[selected].iloc[rd_pos + 1 : end_pos].mean(axis=1)
        benchmark_segments.append(segment)

    if not benchmark_segments:
        return pd.Series(dtype=np.float64, name="ew_high_vol")
    out = pd.concat(benchmark_segments).sort_index()
    out.name = "ew_high_vol"
    return out.dropna()


# ── Beta-matched SPY ────────────────────────────────────────────────────────


def construct_beta_matched_spy_benchmark(
    portfolio_daily_returns: pd.Series,
    spy_daily_returns: pd.Series,
    beta_lookback_days: int = 60,
    rebalance_freq: str = "W-MON",
) -> pd.Series:
    """Build daily return series for a beta-matched SPY position.

    Each rebalance date: estimate the portfolio's trailing realized
    beta to SPY (OLS slope of portfolio returns regressed on SPY
    returns over ``beta_lookback_days``). The benchmark return on each
    subsequent day is ``beta * SPY_return``. Beta is held constant
    between rebalances.

    Parameters
    ----------
    portfolio_daily_returns : pd.Series
        Indexed by trading day. The portfolio whose alpha we're
        evaluating.
    spy_daily_returns : pd.Series
        SPY's daily simple returns aligned (or alignable) to the same
        trading-day index.
    beta_lookback_days : int
        Trailing window for beta estimation. Default 60 days.
    rebalance_freq : str
        Pandas offset alias for beta re-estimation cadence.

    Returns
    -------
    pd.Series of daily benchmark returns indexed by trading day. The
    first ``beta_lookback_days`` of the portfolio history are dropped
    (no trailing data to estimate beta on).
    """
    if beta_lookback_days < 5:
        raise ValueError(f"beta_lookback_days must be >= 5, got {beta_lookback_days}")

    aligned = pd.concat(
        [portfolio_daily_returns.rename("port"), spy_daily_returns.rename("spy")],
        axis=1, join="inner",
    ).dropna()
    if aligned.empty:
        return pd.Series(dtype=np.float64, name="beta_matched_spy")

    period_starts = pd.Series(aligned.index).dt.to_period(rebalance_freq).dt.start_time.unique()
    rebalance_dates: list[pd.Timestamp] = []
    seen: set[pd.Timestamp] = set()
    for ps in period_starts:
        candidates = aligned.index[aligned.index >= ps]
        if len(candidates) > 0:
            d = candidates[0]
            if d not in seen:
                rebalance_dates.append(d)
                seen.add(d)

    bench_segments: list[pd.Series] = []
    for i, rd in enumerate(rebalance_dates):
        rd_pos = aligned.index.get_loc(rd)
        if rd_pos < beta_lookback_days:
            continue
        window = aligned.iloc[rd_pos - beta_lookback_days : rd_pos]
        port = window["port"].to_numpy()
        spy = window["spy"].to_numpy()
        spy_var = float(np.var(spy, ddof=1))
        if spy_var == 0.0:
            beta = 0.0
        else:
            beta = float(np.cov(port, spy, ddof=1)[0, 1] / spy_var)

        next_rd = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None
        end_pos = (
            aligned.index.get_loc(next_rd)
            if next_rd is not None
            else len(aligned.index)
        )
        segment = aligned["spy"].iloc[rd_pos + 1 : end_pos] * beta
        bench_segments.append(segment)

    if not bench_segments:
        return pd.Series(dtype=np.float64, name="beta_matched_spy")
    out = pd.concat(bench_segments).sort_index()
    out.name = "beta_matched_spy"
    return out.dropna()


# ── Alpha + Information Ratio against any benchmark ─────────────────────────


def compute_alpha_vs_benchmark(
    portfolio_daily_returns: pd.Series,
    benchmark_daily_returns: pd.Series,
    label: str = "benchmark",
) -> BenchmarkResult:
    """Compute total + per-day excess return + Information Ratio.

    Parameters
    ----------
    portfolio_daily_returns : pd.Series
        Daily simple returns of the portfolio being evaluated.
    benchmark_daily_returns : pd.Series
        Daily simple returns of the comparison benchmark.
    label : str
        Stamped onto the result for downstream identification (e.g.
        "ew_high_vol", "beta_matched_spy", "spy", "qqq").

    Returns
    -------
    BenchmarkResult dict (see TypedDict at top).
    """
    aligned = pd.concat(
        [portfolio_daily_returns.rename("port"),
         benchmark_daily_returns.rename("bench")],
        axis=1, join="inner",
    ).dropna()
    if aligned.empty:
        return {"status": "insufficient_data", "n_days": 0, "label": label}

    port = aligned["port"]
    bench = aligned["bench"]
    excess = port - bench
    # Total returns via geometric compounding.
    port_total = float((1.0 + port).prod() - 1.0)
    bench_total = float((1.0 + bench).prod() - 1.0)

    excess_mean = float(excess.mean())
    excess_std = float(excess.std(ddof=1)) if len(excess) > 1 else 0.0
    # Use a small floor (1e-12) instead of `> 0.0` — a constant-excess
    # series has std ≈ 1e-17 due to float-mean residual, which would
    # otherwise produce a nonsense ~1e16 IR.
    if excess_std > 1e-12:
        ir = excess_mean / excess_std * math.sqrt(_TRADING_DAYS_PER_YEAR)
    else:
        ir = 0.0

    return {
        "status": "ok",
        "n_days": int(len(aligned)),
        "benchmark_total_return": bench_total,
        "portfolio_total_return": port_total,
        "excess_return": port_total - bench_total,
        "excess_daily_mean": excess_mean,
        "excess_daily_std": excess_std,
        "information_ratio": float(ir),
        "benchmark_daily_returns": bench,
        "excess_daily_returns": excess,
        "label": label,
    }
