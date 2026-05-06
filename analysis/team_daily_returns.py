"""
team_daily_returns.py — per-team daily return series for risk-adjusted metrics.

Sortino, Calmar, Information Ratio, CVaR, PSR, and DSR all need a daily return
*time series* to compute meaningfully. The existing `analysis/end_to_end._team_lift`
emits only horizon-point lift numbers (5d return averages). This module bridges
that gap: given the picks each team made over a window of eval_dates plus a daily
price matrix, simulate an equal-weight (or conviction-weighted) sleeve per team
and return its daily return series.

Pure-compute. No I/O. Caller supplies prices via
``loaders/price_loader.build_matrix`` or any equivalent producer.

Treatment of overlapping holding windows: a team picks 3 stocks on Mon, holds
them for 10 trading days. Next Mon it picks 3 more. On the day both are
"currently held" the team sleeve has 6 positions, equal-weight 1/6. The series
computed here is the team's daily portfolio return under that
"hold-each-pick-for-N-days, equal-weight-while-held" model.

Why not vectorbt: this is a synthetic per-team analysis, not a real portfolio
simulation. We don't need orders/fills/cash bookkeeping — just price-relative
daily returns aggregated across an opening/closing membership set. Doing it in
~80 lines of pandas keeps it dependency-light + testable from hand-computed
fixtures.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_team_daily_returns(
    picks: pd.DataFrame,
    prices: pd.DataFrame,
    horizon_days: int = 10,
) -> dict[str, pd.Series]:
    """Aggregate per-pick daily returns into per-team daily portfolio returns.

    Parameters
    ----------
    picks : pd.DataFrame
        Required columns: ``team_id`` (str), ``ticker`` (str), ``eval_date``
        (datetime-coercible). Optional column: ``weight`` (float; defaults
        to equal-weight = 1.0 if absent). Each row is one (team, ticker)
        pick on a given eval_date — i.e. the team selected this ticker on
        eval_date and held it for ``horizon_days`` trading days.
    prices : pd.DataFrame
        Daily close prices. Index = pd.DatetimeIndex of trading days,
        columns = ticker symbols, values = close. NaN tolerated; rows
        with NaN for a ticker on a given day mean "no fill data" → that
        ticker's contribution is dropped from the team sleeve for that
        day's return (consistent with how a missing close would be
        handled in any portfolio simulation).
    horizon_days : int
        Number of trading days each pick is held. Default 10 to match
        the predictor's primary 10d alpha horizon. The hold window for a
        pick on eval_date D is ``[D, D + horizon_days]`` in the price
        matrix's trading-day index — i.e. the first daily return contribution
        is from D → D+1 and the last from D+(horizon_days-1) → D+horizon_days.

    Returns
    -------
    dict[str, pd.Series]
        Mapping ``team_id`` → daily return series. Each series is indexed
        by trading day (subset of ``prices.index``) and dtype float64.
        Days when the team had no held picks are absent from the series
        (caller can reindex + fill if a contiguous index is needed).

    Notes
    -----
    - Each pick contributes its own daily simple return ``(p[t]/p[t-1]) - 1``
      while held. The team's daily return on day t is the weight-normalized
      mean over picks held on day t (using ``weight`` if present, else 1.0).
    - Picks whose eval_date is not in ``prices.index`` are dropped with a
      warning — the price matrix should cover the full pick history. The
      function does NOT forward-look for fills; if the eval_date isn't a
      trading day, that pick is skipped entirely.
    - Tickers absent from ``prices.columns`` are dropped silently per pick
      — a common case during smoke tests with restricted universes.

    Example
    -------
    >>> picks = pd.DataFrame({
    ...     "team_id": ["tech", "tech", "health"],
    ...     "ticker":  ["AAPL", "MSFT", "JNJ"],
    ...     "eval_date": pd.to_datetime(["2026-01-05", "2026-01-05", "2026-01-05"]),
    ... })
    >>> prices = ...  # 20-day price matrix for AAPL, MSFT, JNJ
    >>> returns = compute_team_daily_returns(picks, prices, horizon_days=5)
    >>> returns["tech"]  # pd.Series of 5 daily returns, equal-weight AAPL+MSFT
    """
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
    required_cols = {"team_id", "ticker", "eval_date"}
    missing = required_cols - set(picks.columns)
    if missing:
        raise ValueError(f"picks missing required columns: {sorted(missing)}")
    if picks.empty:
        return {}
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices.index must be a pd.DatetimeIndex")

    # Per-ticker daily simple returns. NaN rows propagate; downstream we mask.
    ticker_returns = prices.pct_change()

    eval_dates = pd.to_datetime(picks["eval_date"])
    weights = picks["weight"].astype(float) if "weight" in picks.columns else pd.Series(
        1.0, index=picks.index
    )

    out: dict[str, pd.Series] = {}
    for team_id, team_rows in picks.groupby("team_id"):
        per_pick_series: list[pd.Series] = []
        per_pick_weights: list[float] = []
        for idx, row in team_rows.iterrows():
            ticker = row["ticker"]
            if ticker not in ticker_returns.columns:
                continue
            ed = pd.Timestamp(eval_dates.loc[idx])
            if ed not in prices.index:
                logger.debug("pick eval_date %s not in price index; skipping %s",
                             ed.date(), ticker)
                continue
            start_pos = prices.index.get_loc(ed)
            end_pos = min(start_pos + horizon_days, len(prices.index) - 1)
            if end_pos <= start_pos:
                continue
            # Daily returns from D+1 through D+horizon_days, inclusive.
            window = ticker_returns[ticker].iloc[start_pos + 1 : end_pos + 1]
            window = window.dropna()
            if window.empty:
                continue
            per_pick_series.append(window)
            per_pick_weights.append(float(weights.loc[idx]))

        if not per_pick_series:
            continue

        # Stack returns into a wide DataFrame: rows = trading days,
        # columns = pick index. Each pick's Series carries the ticker
        # symbol as its name; rename to a positional integer per pick
        # so pd.concat(axis=1) aligns on the trading-day index without
        # collapsing duplicate ticker columns (a team can hold the same
        # ticker across overlapping eval_dates — each contributes
        # independently to the sleeve).
        renamed = [s.rename(i) for i, s in enumerate(per_pick_series)]
        wide = pd.concat(renamed, axis=1)

        w_array = np.array(per_pick_weights, dtype=np.float64)
        # Weighted-mean per row, ignoring NaN. Each row's effective weight
        # is the sum of weights of non-NaN entries — re-normalize so a row
        # with only 2 of 3 picks active doesn't get diluted to 2/3.
        mask = wide.notna().to_numpy()
        values = np.where(mask, wide.to_numpy(), 0.0)
        row_weights = mask * w_array[np.newaxis, :]
        row_total_weight = row_weights.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            weighted = np.where(
                row_total_weight > 0,
                (values * row_weights).sum(axis=1) / row_total_weight,
                np.nan,
            )
        series = pd.Series(weighted, index=wide.index, name=team_id, dtype=np.float64)
        series = series.dropna()
        if not series.empty:
            out[str(team_id)] = series

    return out


def stack_team_returns_to_long(
    returns_by_team: dict[str, pd.Series],
) -> pd.DataFrame:
    """Convert ``compute_team_daily_returns`` output to long-form DataFrame.

    Convenience for persistence: the long form is
    ``[team_id, trading_day, return]`` per row, easy to write to parquet
    or merge with other per-team metrics. Empty input → empty DataFrame
    with the expected schema.
    """
    if not returns_by_team:
        return pd.DataFrame(columns=["team_id", "trading_day", "return"])
    rows: list[pd.DataFrame] = []
    for team_id, series in returns_by_team.items():
        df = series.rename("return").reset_index().rename(
            columns={"index": "trading_day", series.index.name or "index": "trading_day"}
        )
        # The series's index name varies based on input; force-rename.
        df.columns = ["trading_day", "return"]
        df.insert(0, "team_id", team_id)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)
