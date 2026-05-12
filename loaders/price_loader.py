"""
price_loader.py — reads prices from ArcticDB (the only source).

Replaces the legacy S3 `prices/{date}/prices.json` → polygon → yfinance → IBKR
fallback chain with a single ArcticDB read. Matches predictor PR #38 rip-and-
replace pattern: hard-fail on unreachable, per-ticker misses logged + dropped,
>5% read error rate aborts the pipeline.

Folded in as Phase 0 of `backtester-audit-260415.md` — data-source parity is
the hard prerequisite to Phase 1 replay parity (running parity against
divergent data sources measures noise, not logic).
"""

from __future__ import annotations

import json
import logging

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from store.arctic_reader import load_universe_from_arctic

logger = logging.getLogger(__name__)

# Same tuning knobs as the prior implementation — ffill bounds + staleness gate
# survive the rip because they apply to the resulting matrix, not to the source.
_MAX_FFILL_DAYS = 5        # max forward-fill window (1 trading week)
_MAX_STALENESS_BDAYS = 5   # circuit breaker: halt simulation if data this stale
_GAP_WARNING_THRESHOLD = 5  # tickers with > this many genuine gap days get flagged

# Tickers that trade on a different calendar than the NYSE equity calendar
# (bond futures + options/volatility indexes) — excluded from gap warnings
# because their "missing" NYSE dates are calendar mismatches, not data
# quality issues. Pinned to the symbols actually persisted as ArcticDB
# universe entries: ``IRX`` (13-week T-bill), ``TNX`` (10-yr Treasury),
# ``VIX`` + ``VIX3M`` (CBOE volatility), plus the ``^`` Yahoo-prefixed
# aliases retained for legacy code paths.
_CALENDAR_MISMATCH_TICKERS = {
    "VIX", "VIX3M", "TNX", "IRX",
    "^VIX", "^TNX", "^IRX",
}


def build_matrix(
    dates: list[str],
    bucket: str,
    field: str = "close",
    signals_prefix: str = "signals",
    _ohlcv_out: dict | None = None,
    tickers_allowlist: set[str] | None = None,
) -> pd.DataFrame:
    """
    Build a price matrix for vectorbt: rows = dates, columns = tickers.

    Reads the full universe from ArcticDB once (bulk), then filters + pivots.
    Tickers are resolved from signals.json per date (unchanged semantic) so we
    only build columns for stocks that actually appeared in the signal history.

    Parameters
    ----------
    dates : List of date strings "YYYY-MM-DD".
    bucket : S3 bucket hosting the ArcticDB path prefix.
    field : OHLCV field to use ("open", "close", "high", "low"). Default "close".
    signals_prefix : S3 prefix for signal-set ticker resolution (default "signals").
    _ohlcv_out : Optional output dict — when provided, is populated with
                 {ticker: pd.DataFrame} (DatetimeIndex, lowercase
                 [open, high, low, close] columns, dtype float64) for
                 strategy-layer consumers (ATR trailing stops etc.).
                 Downstream consumers dispatch on shape — see
                 ``_simulate_single_date`` + ``precompute_indicator_series``
                 — so this is drop-in compatible with the legacy
                 list-of-dicts producers until step 9 cleanup lands
                 (pandas refactor, plan 2026-04-23).
    tickers_allowlist : Optional set of tickers to restrict the ArcticDB bulk
                 read. When provided, signal-resolved universe is intersected
                 with this allowlist BEFORE the ArcticDB read so we don't pay
                 full-universe cost for smoke fixtures. Production default
                 None (full universe). See smoke harness fixture wiring.

    Returns
    -------
    DataFrame indexed by datetime, columns by ticker. Missing values forward-
    filled up to `_MAX_FFILL_DAYS` then back-filled; tickers with unfilled gaps
    are dropped (VectorBT treats NaN as zero-return which distorts results).

    Attached to `df.attrs`:
      * `price_gap_warnings` — per-ticker count of business days where neither
          yfinance nor polygon had a row inside the per-ticker
          [first_listed_date, simulation_end] window. Excludes pre-IPO dates
          (derived from the ticker's own first non-NaN Close) and
          calendar-mismatch macros (VIX/VIX3M/TNX/IRX).
      * `unfilled_gaps` — per-ticker count of NaN rows that remained after ffill
      * `staleness_warning` — human-readable string or None
      * `stale_circuit_break` — True if last price date < today - 5 BDays
      * `no_data_dates` — date strings where signals existed but ArcticDB had nothing
    """
    # ── Resolve the signal-set ticker universe across all dates ─────────────
    tickers_by_date: dict[str, list[str]] = {}
    all_tickers: set[str] = set()
    no_signal_dates: list[str] = []
    for d in sorted(dates):
        tickers = _tickers_from_signals(bucket, d, signals_prefix)
        if tickers:
            tickers_by_date[d] = tickers
            all_tickers.update(tickers)
        else:
            no_signal_dates.append(d)

    if no_signal_dates:
        logger.warning("No signals found for %d dates — price rows will be empty: %s",
                       len(no_signal_dates), no_signal_dates[:10])

    # Smoke-fixture universe filter — intersect signal-resolved tickers
    # with the operator-provided allowlist so downstream ArcticDB bulk
    # reads only pay for tickers we actually need. Production default
    # None → no filter, full signal universe loads.
    if tickers_allowlist is not None:
        before = len(all_tickers)
        all_tickers = all_tickers & set(tickers_allowlist)
        logger.info(
            "price_loader: filtered to %d tickers via tickers_allowlist "
            "(from %d signal-resolved, allowlist=%d)",
            len(all_tickers), before, len(tickers_allowlist),
        )

    if not all_tickers:
        logger.warning("Zero tickers resolved from signals across %d dates — returning empty matrix", len(dates))
        df = pd.DataFrame()
        df.attrs["price_gap_warnings"] = {}
        df.attrs["unfilled_gaps"] = {}
        df.attrs["staleness_warning"] = None
        df.attrs["stale_circuit_break"] = False
        df.attrs["no_data_dates"] = no_signal_dates
        return df

    # ── Bulk read ArcticDB ─────────────────────────────────────────────────
    logger.info("Reading ArcticDB universe for %d tickers across %d dates", len(all_tickers), len(dates))
    # When the fixture specified an allowlist, also pass it to the
    # ArcticDB reader so it reads only those symbols. Without this the
    # reader still enumerates all 900+ symbols even though we'd drop
    # most of them in the filter above. Macro/ETF symbols are always
    # loaded downstream regardless — SPY required for benchmarking.
    price_data, _ = load_universe_from_arctic(
        bucket=bucket, tickers_allowlist=tickers_allowlist,
    )

    # ── Pivot price_data into matrix[date, ticker] ────────────────────────
    field_title = field.capitalize()  # "close" → "Close"
    series_by_ticker: dict[str, pd.Series] = {}
    for ticker in all_tickers:
        df_ticker = price_data.get(ticker)
        if df_ticker is None or df_ticker.empty:
            logger.debug("ArcticDB missing %s — excluded from price matrix", ticker)
            continue
        if field_title not in df_ticker.columns:
            logger.warning("ArcticDB %s missing '%s' column — excluded", ticker, field_title)
            continue
        series_by_ticker[ticker] = df_ticker[field_title]

    df = pd.DataFrame(series_by_ticker)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Restrict rows to requested dates (forward-fill handles any missing rows)
    date_index = pd.to_datetime(sorted(dates))
    df = df.reindex(df.index.union(date_index)).loc[:date_index.max()]

    # ── Capture per-ticker OHLCV for strategy layer consumers ──────────────
    #
    # 2026-04-23 (pandas refactor): emit per-ticker pd.DataFrame rather
    # than list-of-dicts. Shape matches ``build_ohlcv_df_by_ticker`` —
    # DatetimeIndex + lowercase [open, high, low, close], fallback-to-
    # close for missing OHL, sorted + dedup'd. Downstream consumers
    # dispatch on shape (see backtest._simulate_single_date and
    # synthetic.signal_generator.precompute_indicator_series) so this
    # drop-in replaces the prior list-of-dicts production without
    # coordinating producers and consumers across the migration.
    if _ohlcv_out is not None:
        for ticker in all_tickers:
            df_ticker = price_data.get(ticker)
            if df_ticker is None or df_ticker.empty:
                continue
            sliced = df_ticker.loc[df_ticker.index <= date_index.max()]
            if sliced.empty:
                continue
            cols: dict[str, pd.Series] = {}
            for title in ("Close", "Open", "High", "Low"):
                lower = title.lower()
                if title in sliced.columns:
                    cols[lower] = sliced[title]
                elif lower in sliced.columns:
                    cols[lower] = sliced[lower]
            if "close" not in cols:
                continue
            close = cols["close"]
            for key in ("open", "high", "low"):
                if key not in cols:
                    cols[key] = close
            frame = pd.DataFrame({
                "open":  cols["open"],
                "high":  cols["high"],
                "low":   cols["low"],
                "close": cols["close"],
            }).astype(float)
            frame.index = pd.to_datetime(frame.index)
            frame = frame[~frame.index.duplicated(keep="last")].sort_index()
            _ohlcv_out[ticker] = frame

    # ── Gap detection (source-aware, pre-IPO-aware) ────────────────────────
    #
    # A "gap" is a date inside the requested simulation window where the
    # ticker's per-row ``source`` provenance is NaN — i.e. neither yfinance
    # nor polygon landed a row for that (ticker, date) under the upstream
    # windowed-data-reconciliation arc (alpha-engine-data PRs #199-#201,
    # 2026-05-10). This replaces the prior matrix-wide NaN count, which
    # conflated three distinct phenomena:
    #   1. Front-of-history mismatches (e.g. ASGN/HOLX have 10 days of
    #      pre-2016-05-09 history that other tickers don't) — pivoting
    #      produced NaN cells on every other ticker for those 10 dates,
    #      surfacing as ~110 false-positive "gap" warnings.
    #   2. Calendar mismatches for VIX/TNX/IRX (bond/options calendars).
    #   3. Genuine ingestion gaps (~11 IPO/spinoff edge cases) — the
    #      population the metric is meant to surface.
    # Per-ticker first-listed-date is derived from the ticker's own
    # ``Close`` history (first non-NaN row) — no separate IPO metadata
    # registry exists, and the price panel is self-consistent for this
    # purpose. Calendar-mismatch macros are excluded via name set.
    universe_dates = pd.to_datetime(sorted(dates))
    sim_end = universe_dates.max() if len(universe_dates) else None
    price_gap_warnings: dict[str, int] = {}
    for ticker, df_ticker in price_data.items():
        if ticker in _CALENDAR_MISMATCH_TICKERS:
            continue
        if df_ticker is None or df_ticker.empty or "Close" not in df_ticker.columns:
            continue
        first_seen = df_ticker["Close"].first_valid_index()
        if first_seen is None or sim_end is None:
            continue
        window_lo = max(pd.Timestamp(first_seen), universe_dates.min())
        if window_lo > sim_end:
            continue
        window = pd.bdate_range(start=window_lo, end=sim_end)
        if len(window) == 0:
            continue
        # Source-aware path when provenance is present; legacy series
        # (pre-2026-05-09 daily_append rollout) fall back to a Close-NaN
        # check on the same windowed range.
        if "source" in df_ticker.columns:
            present = df_ticker["source"].reindex(window).notna().sum()
        else:
            present = df_ticker["Close"].reindex(window).notna().sum()
        gap_count = int(len(window) - present)
        if gap_count > _GAP_WARNING_THRESHOLD:
            price_gap_warnings[str(ticker)] = gap_count
    if price_gap_warnings:
        logger.warning("Price gaps detected (will be ffill'd up to %d days): %s",
                       _MAX_FFILL_DAYS, price_gap_warnings)

    df.ffill(limit=_MAX_FFILL_DAYS, inplace=True)
    df.bfill(inplace=True)

    remaining_nans = df.isna().sum(axis=0)
    unfilled = remaining_nans[remaining_nans > 0]
    unfilled_dict = {str(k): int(v) for k, v in unfilled.items()} if not unfilled.empty else {}
    if unfilled_dict:
        logger.warning("Dropping %d tickers with unfilled gaps after ffill(limit=%d): %s",
                       len(unfilled_dict), _MAX_FFILL_DAYS, unfilled_dict)
        df.drop(columns=unfilled.index, inplace=True)

    # Freshness validation (tiered)
    staleness_warning: str | None = None
    stale_circuit_break = False
    if not df.empty:
        last_price_date = df.index.max()
        expected_recent = pd.Timestamp.now() - pd.tseries.offsets.BDay(2)
        expected_max    = pd.Timestamp.now() - pd.tseries.offsets.BDay(_MAX_STALENESS_BDAYS)
        if last_price_date < expected_max:
            staleness_warning = (
                f"STALE price data: last date {last_price_date.date()}, "
                f"expected >= {expected_max.date()} — simulation results unreliable"
            )
            logger.error(staleness_warning)
            stale_circuit_break = True
        elif last_price_date < expected_recent:
            staleness_warning = (
                f"Stale price data: last date {last_price_date.date()}, "
                f"expected >= {expected_recent.date()}"
            )
            logger.warning(staleness_warning)

    df.attrs["price_gap_warnings"] = price_gap_warnings
    df.attrs["unfilled_gaps"] = unfilled_dict
    df.attrs["staleness_warning"] = staleness_warning
    df.attrs["stale_circuit_break"] = stale_circuit_break
    df.attrs["no_data_dates"] = no_signal_dates

    return df


# ── Private helpers ─────────────────────────────────────────────────────────

def _tickers_from_signals(bucket: str, signal_date: str, prefix: str = "signals") -> list[str]:
    """
    Extract the unique ticker list from signals.json for a given date.

    Retained from the legacy implementation — used to restrict the ArcticDB
    pivot to tickers that actually appeared in the signal history, rather than
    pulling every ~900 symbols into every backtest matrix.
    """
    key = f"{prefix}/{signal_date}/signals.json"
    s3 = boto3.client("s3")
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        data = json.loads(response["Body"].read())
        sigs = data.get("signals", {})
        if isinstance(sigs, dict):
            ticker_set = set(sigs.keys())
        else:
            ticker_set = {s["ticker"] for s in sigs if "ticker" in s}
        for k in ("universe", "buy_candidates"):
            for s in data.get(k, []):
                if isinstance(s, dict) and "ticker" in s:
                    ticker_set.add(s["ticker"])
        tickers = sorted(ticker_set)
        logger.debug("Resolved %d tickers from signals for %s", len(tickers), signal_date)
        return tickers
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("NoSuchKey", "AccessDenied", "403"):
            logger.warning("No signals.json found for %s (%s) — cannot resolve tickers", signal_date, code)
            return []
        raise
