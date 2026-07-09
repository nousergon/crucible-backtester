"""risk_matched_benchmark — re-export shim over
``nousergon_lib.quant.stats.risk_matched_benchmark``.

Lifted to the shared alpha-engine-lib (LV2-AE leverage arc, 2026-06-03). This
shim preserves the ``analysis.risk_matched_benchmark`` import surface; the
implementation + its unit tests now live in the lib.
"""

from __future__ import annotations

import pandas as pd

from nousergon_lib.quant.stats.risk_matched_benchmark import (
    BenchmarkResult,
    compute_alpha_vs_benchmark,
    construct_beta_matched_spy_benchmark,
    construct_ew_high_vol_benchmark,
)

__all__ = [
    "BenchmarkResult",
    "compute_alpha_vs_benchmark",
    "construct_beta_matched_spy_benchmark",
    "construct_ew_high_vol_benchmark",
    "construct_ew_universe_benchmark",
]


def construct_ew_universe_benchmark(
    prices: pd.DataFrame,
    universe: list[str] | None = None,
    rebalance_freq: str = "W-MON",
) -> pd.Series:
    """Build daily return series for the FULL decision universe, equal-weight.

    Sibling of ``construct_ew_high_vol_benchmark`` (config#834): where that
    constructor holds only the top vol-quartile of ``universe``, this one
    holds EVERY ticker in ``universe``, equal-weighted, rebalanced on the
    same cadence. Isolates stock-selection alpha from cap-weighted-tilt
    alpha — "did you beat picking everything, unweighted?" as opposed to
    "did you beat picking risky stuff?"

    Deliberately NOT implemented as a delegation to
    ``construct_ew_high_vol_benchmark`` with an extreme ``vol_quantile``: an
    attempt at that (epsilon ``vol_quantile``) was tried and reverted — pandas'
    ``Series.quantile()`` uses linear interpolation by default, so even
    ``vol.quantile(1e-15)`` lands measurably ABOVE ``vol.min()`` for
    non-tied floats, silently excluding the single lowest-vol ticker from
    every rebalance segment (verified: a 20-ticker synthetic universe drops
    exactly 1 ticker at every epsilon tried, down to 1e-15). "Select
    everyone" has no threshold to get subtly wrong, so this reimplements
    only the rebalance-date + trailing-window bookkeeping (identical to
    ``construct_ew_high_vol_benchmark``'s, so the two stay directly
    comparable) and skips the vol-ranking step entirely — not a parallel
    ALPHA computation (that stays in ``compute_alpha_vs_benchmark``), just a
    parallel BASKET construction with no selection logic to omit correctly.

    Parameters mirror ``construct_ew_high_vol_benchmark`` minus
    ``vol_quantile``/``vol_lookback_days`` (no vol ranking, so no vol
    lookback window is needed — every ticker qualifies from day one of
    ``prices``, unlike the vol-quartile basket which needs
    ``vol_lookback_days`` of trailing history before it can rank anyone).
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices.index must be a DatetimeIndex")

    cols = list(prices.columns) if universe is None else [
        t for t in universe if t in prices.columns
    ]
    if not cols:
        raise ValueError("universe is empty after intersecting with prices.columns")

    sub = prices[cols]
    daily_returns = sub.pct_change()

    # Rebalance dates: same construction as construct_ew_high_vol_benchmark
    # (first trading day on/after each period boundary) so the two baskets'
    # segment boundaries line up and are directly comparable.
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

    if not rebalance_dates:
        return pd.Series(dtype="float64", name="ew_universe")

    benchmark_segments: list[pd.Series] = []
    for i, rd in enumerate(rebalance_dates):
        rd_pos = sub.index.get_loc(rd)
        next_rd = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None
        end_pos = (
            sub.index.get_loc(next_rd) if next_rd is not None else len(sub.index)
        )
        # Daily returns over [rd+1, next_rd] — same "first day post-rebalance
        # starts compounding" convention as construct_ew_high_vol_benchmark.
        segment = daily_returns[cols].iloc[rd_pos + 1 : end_pos].mean(axis=1)
        benchmark_segments.append(segment)

    if not benchmark_segments:
        return pd.Series(dtype="float64", name="ew_universe")
    out = pd.concat(benchmark_segments).sort_index()
    out.name = "ew_universe"
    return out.dropna()
