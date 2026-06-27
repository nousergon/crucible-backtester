"""
store/arctic_reader.py — Read universe + macro data from ArcticDB for backtesting.

Connects to the shared ArcticDB instance on S3 and reads per-ticker DataFrames
containing OHLCV + 53 pre-computed features. Sole price source for the
backtester post-Phase-0 (backtester-audit-260415.md) — no S3 parquet cache,
no yfinance, no polygon, no IBKR fallback.

Usage:
    from store.arctic_reader import load_universe_from_arctic, _verify_arctic_fresh

    price_data, features_by_ticker = load_universe_from_arctic(bucket)

Failure semantics (matches predictor `load_price_data_from_arctic` pattern):
  * ArcticDB unreachable → RuntimeError (hard fail)
  * Per-ticker read error rate > 5% → RuntimeError
  * Individual ticker missing/empty → WARNING log, dropped
  * SPY missing from macro library (freshness gate) → RuntimeError
"""

from __future__ import annotations

import logging
import os
import time

import arcticdb as adb
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_BUCKET = "alpha-engine-research"
ARCTIC_PREFIX = "arcticdb"

# Macro/ETF tickers that are not stock symbols
_MACRO_TICKERS = {"SPY", "VIX", "VIX3M", "TNX", "IRX", "GLD", "USO", "^VIX", "^TNX", "^IRX"}
_SECTOR_ETFS = {"XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLC", "XLB"}
_SKIP_TICKERS = _MACRO_TICKERS | _SECTOR_ETFS

# OHLCV columns (kept separate for price_matrix / ohlcv_by_ticker)
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

# Per-ticker error rate threshold — matches predictor inference gate
_MAX_ERROR_RATE = 0.05


def _get_arctic(bucket: str) -> adb.Arctic:
    """Create ArcticDB connection via the lib chokepoint (L2771).

    Logs the resolved URI + library list at DEBUG level so subprocess-vs-
    parent Arctic-state divergence (e.g. 2026-04-24 parity incident:
    main backtest saw 910 universe symbols, fresh pytest subprocess on
    the same spot saw 0) can be diagnosed from the stream without a
    separate instrumentation pass.
    """
    from nousergon_lib.arcticdb import open_arctic
    try:
        arctic = open_arctic(bucket)
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc
    # Log connection details once per process so subprocess/parent
    # divergence is greppable. At INFO level on first call only — repeat
    # calls in the same process would spam.
    global _ARCTIC_LOGGED
    if not _ARCTIC_LOGGED:
        try:
            libs = sorted(arctic.list_libraries())
        except Exception as exc:
            libs = f"<list_libraries failed: {exc}>"
        # NOTE: the resolved URI + region are now owned by the lib chokepoint
        # (open_arctic) and are NOT in this scope — the L2771 migration replaced
        # the local `uri = ...; adb.Arctic(uri)` with `open_arctic(bucket)` but
        # left this log line referencing the removed `uri`/`region` locals,
        # raising `NameError: name 'uri' is not defined` on every fresh-process
        # connect. The main backtest reads prices from S3/yfinance so it never
        # hit this; pit_parity's replay subprocess connects to ArcticDB and so
        # failed with backtester_replay_error every cycle. The library list is
        # the load-bearing divergence signal (the 2026-04-24 incident was 910 vs
        # 0 symbols) — log bucket + libs, which ARE in scope.
        log.info(
            "ArcticDB connected: bucket=%s libraries=%s",
            bucket, libs,
        )
        _ARCTIC_LOGGED = True
    return arctic


_ARCTIC_LOGGED = False


def _safe_last_date(idx: pd.Index) -> pd.Timestamp | None:
    """Return the normalized last date from a DatetimeIndex, or None if empty/NaT."""
    if idx is None or idx.empty:
        return None
    last = idx.max()
    if pd.isna(last):
        return None
    return pd.Timestamp(last).normalize()


def _verify_arctic_fresh(bucket: str, min_date: str | None = None) -> None:
    """Assert ArcticDB's SPY close series has data through ``min_date``.

    Matches the predictor `_verify_arctic_fresh` pattern. If ``min_date`` is
    None, just asserts SPY exists and has at least one row.

    Raises RuntimeError on missing/stale SPY.
    """
    arctic = _get_arctic(bucket)
    try:
        macro_lib = arctic.get_library("macro")
    except Exception as exc:
        raise RuntimeError(f"ArcticDB macro library unreachable: {exc}") from exc

    try:
        df = macro_lib.read("SPY", columns=["Close"]).data
    except Exception as exc:
        raise RuntimeError(
            f"ArcticDB macro SPY unreadable: {exc} — DataPhase1 did not run "
            f"or the macro library is broken."
        ) from exc

    last_date = _safe_last_date(df.index)
    if last_date is None:
        raise RuntimeError("ArcticDB macro SPY has no rows — DataPhase1 has never written.")

    if min_date is not None:
        expected = pd.Timestamp(min_date).normalize()
        if last_date < expected:
            raise RuntimeError(
                f"ArcticDB macro SPY last_date={last_date.date()} is stale for "
                f"required date={expected.date()}."
            )


def load_universe_from_arctic(
    bucket: str = DEFAULT_BUCKET,
    tickers_allowlist: set[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Load universe symbols from ArcticDB.

    Returns two dicts, both keyed by ticker:
        price_data:         {ticker: DataFrame[OHLCV]}  — for price_matrix and ohlcv_by_ticker
        features_by_ticker: {ticker: DataFrame[OHLCV + features]}  — for GBM inference

    Macro/ETF tickers (SPY, VIX, sector ETFs) are included in price_data but
    excluded from features_by_ticker. The macro tickers are ALWAYS loaded
    regardless of `tickers_allowlist` — SPY in particular is required for
    benchmark comparisons and cannot be filtered out.

    Parameters
    ----------
    tickers_allowlist :
        If provided, only stocks in this set are read from the universe
        library. Intersection with the actual ArcticDB symbol list is
        applied. Macro/ETF symbols are ALWAYS loaded regardless. Intended
        for smoke-harness fixtures where reading the full ~900-ticker
        universe dominates wall-clock (e.g. 2026-04-23 dry-run surfaced
        ~196s of bulk read for a 5-date smoke). Defaults to None (full
        universe) so production code is unchanged.

    Raises
    ------
    RuntimeError : ArcticDB unreachable, or per-ticker read error rate > 5%.
    """
    t0 = time.time()
    arctic = _get_arctic(bucket)
    # Universe library opens via the shared lib helper (config#804). The
    # ``macro`` open and the #826 empty-universe diagnostic below still use
    # the local ``arctic`` handle and are intentionally left as-is.
    from nousergon_lib.arcticdb import open_universe_lib
    try:
        universe = open_universe_lib(bucket)
        macro_lib = arctic.get_library("macro")
    except Exception as exc:
        raise RuntimeError(
            f"ArcticDB library open failed on bucket {bucket}: {exc}"
        ) from exc

    symbols = universe.list_symbols()
    if tickers_allowlist is not None:
        # Intersect with the actual ArcticDB catalog so a bad allowlist
        # (typo, missing ticker) doesn't crash — we just read fewer
        # tickers. Loud log so operators see when the filter activates.
        requested = set(tickers_allowlist)
        symbols = [s for s in symbols if s in requested]
        log.info(
            "ArcticDB universe: filtered to %d symbols via tickers_allowlist "
            "(requested %d of available catalog)",
            len(symbols), len(requested),
        )
    else:
        log.info("ArcticDB universe: %d symbols", len(symbols))
        # No-silent-fails: an empty universe in a production path (no
        # allowlist) is never a legitimate outcome — it silently cascades
        # to empty price_matrix → empty orders → misleading "no divergence"
        # results in parity checks and silent zero-order outputs in real
        # backtests. Hard-fail here with the diagnostic context needed
        # to triage (URI, available libraries, a sample of what IS in
        # the library if anything).
        #
        # Motivated by 2026-04-24 parity incident: list_symbols returned
        # 0 in a fresh pytest subprocess on a spot where the main
        # backtest's Python process had just successfully read 910
        # symbols minutes earlier. Root cause still under investigation;
        # this guard ensures the next occurrence surfaces loud with
        # actionable context instead of silently producing zero-order
        # outputs downstream.
        if len(symbols) == 0:
            region = os.environ.get("AWS_REGION", "us-east-1")
            uri = (
                f"s3s://s3.{region}.amazonaws.com:{bucket}"
                f"?path_prefix={ARCTIC_PREFIX}&aws_auth=true"
            )
            try:
                all_libs = sorted(arctic.list_libraries())
            except Exception as exc:
                all_libs = f"<list_libraries failed: {exc}>"
            raise RuntimeError(
                f"ArcticDB universe library returned 0 symbols — refusing to "
                f"proceed with empty universe (no-silent-fails). "
                f"Context: uri={uri} bucket={bucket!r} "
                f"arctic.list_libraries()={all_libs} "
                f"process_pid={os.getpid()}. "
                f"If this is a fresh/new bucket, populate via "
                f"alpha-engine-data's weekly backfill first. If this was "
                f"working minutes ago in another process, look for env/auth "
                f"divergence between the invocations."
            )

    price_data: dict[str, pd.DataFrame] = {}
    features_by_ticker: dict[str, pd.DataFrame] = {}
    n_err = 0

    for i, ticker in enumerate(symbols):
        try:
            df = universe.read(ticker).data
        except Exception as exc:
            log.warning("ArcticDB universe read failed for %s: %s", ticker, exc)
            n_err += 1
            continue

        if df.empty:
            log.warning("ArcticDB universe returned empty frame for %s", ticker)
            n_err += 1
            continue

        # Defensive dedup per 2026-04-15 duplicate-row workaround window
        df = df[~df.index.duplicated(keep="last")].sort_index()

        # price_data gets OHLCV columns plus the row-provenance ``source``
        # column when present (alpha-engine-data wires it as of 2026-05-09).
        # Carrying it through lets price_loader's gap-warning metric
        # discriminate "neither yfinance nor polygon had data" cells from
        # pre-IPO matrix-pivot artifacts.
        price_cols = [c for c in OHLCV_COLS if c in df.columns]
        if "source" in df.columns:
            price_cols.append("source")
        price_data[ticker] = df[price_cols]

        # features_by_ticker gets the full DataFrame (OHLCV + features),
        # only for stock tickers (not macro/ETFs)
        if ticker not in _SKIP_TICKERS:
            features_by_ticker[ticker] = df

        if (i + 1) % 200 == 0:
            log.info("  Read %d/%d symbols from ArcticDB", i + 1, len(symbols))

    err_rate = n_err / max(len(symbols), 1)
    if err_rate > _MAX_ERROR_RATE:
        raise RuntimeError(
            f"ArcticDB per-ticker error rate {err_rate:.1%} exceeds "
            f"{_MAX_ERROR_RATE:.0%} threshold ({n_err} failed of {len(symbols)}) — "
            f"treating as pipeline failure."
        )

    # Macro + sector ETFs: try universe first (backfill writes full OHLCV),
    # fall back to macro library (daily_append writes Close only).
    all_macro_syms = ["SPY", "VIX", "VIX3M", "TNX", "IRX", "GLD", "USO"] + sorted(_SECTOR_ETFS)
    for sym in all_macro_syms:
        if sym in price_data:
            continue  # already loaded from universe
        try:
            mdf = macro_lib.read(sym).data
        except Exception as exc:
            log.warning("ArcticDB macro read failed for %s: %s", sym, exc)
            continue
        if mdf.empty:
            log.warning("ArcticDB macro returned empty frame for %s", sym)
            continue
        mdf = mdf[~mdf.index.duplicated(keep="last")].sort_index()
        price_data[sym] = mdf

    elapsed = time.time() - t0
    log.info(
        "[data_source=arcticdb] Load complete in %.1fs: %d price tickers, %d feature tickers",
        elapsed, len(price_data), len(features_by_ticker),
    )

    return price_data, features_by_ticker
