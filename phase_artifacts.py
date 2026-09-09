"""
phase_artifacts.py — Serialization helpers for PhaseRegistry artifacts.

Each `save_*` function uploads an artifact to S3 and returns the key so
callers can hand it to `ctx.record_artifact(key)`. Each `load_*` function
is the strict inverse.

Layout:
    s3://{bucket}/backtest/{date}/.phases/{phase}/{name}.{ext}

JSON is used for dicts-of-scalars + lists; parquet for DataFrames and
OHLCV data. Pickle is NOT used — it's fragile across Python versions
and hides artifact contents from non-Python consumers (the evaluator
and future dashboards read these).

Motivated by ROADMAP Backtester P0 "Phase-selective backtest execution —
skip already-successful phases on retry". Paired with PhaseRegistry in
pipeline_common.py.
"""

from __future__ import annotations

import io
import json
import logging

import boto3
import pandas as pd

logger = logging.getLogger(__name__)


def artifact_key(date: str, phase: str, name: str, ext: str) -> str:
    """Canonical S3 key for a phase artifact. Exposed for tests + callers
    that want to name a key without uploading (e.g. the `--dry-run` path).
    """
    return f"backtest/{date}/.phases/{phase}/{name}.{ext}"


def _client(s3_client=None):
    return s3_client if s3_client is not None else boto3.client("s3")


# ── JSON artifacts (dicts, lists, scalars) ───────────────────────────────────


def save_json(bucket: str, date: str, phase: str, name: str, obj, *, s3_client=None) -> str:
    """Serialize `obj` to JSON and upload. Returns the S3 key.

    Uses `default=str` so datetime / numpy scalars don't crash the encoder.
    Callers that need strict type preservation should use a DataFrame-
    backed artifact instead.
    """
    key = artifact_key(date, phase, name, "json")
    body = json.dumps(obj, indent=2, default=str).encode()
    _client(s3_client).put_object(
        Bucket=bucket, Key=key, Body=body, ContentType="application/json",
    )
    return key


def load_json(bucket: str, key: str, *, s3_client=None):
    """Load a JSON artifact. Raises if missing or corrupt."""
    obj = _client(s3_client).get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())


# ── DataFrame artifacts ──────────────────────────────────────────────────────


def save_dataframe(
    bucket: str, date: str, phase: str, name: str, df: pd.DataFrame,
    *, s3_client=None, preserve_index: bool = True,
) -> str:
    """Upload `df` as parquet. Returns the S3 key.

    `preserve_index=True` keeps the row index so DataFrames with a
    meaningful index (e.g. `price_matrix` indexed by date) round-trip
    correctly. Set False for row-only frames like sweep_df.
    """
    key = artifact_key(date, phase, name, "parquet")
    buf = io.BytesIO()
    df.to_parquet(buf, index=preserve_index)
    _client(s3_client).put_object(
        Bucket=bucket, Key=key, Body=buf.getvalue(),
        ContentType="application/vnd.apache.parquet",
    )
    return key


def load_dataframe(bucket: str, key: str, *, s3_client=None) -> pd.DataFrame:
    """Load a parquet DataFrame."""
    obj = _client(s3_client).get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


# ── OHLCV artifacts (dict[ticker → list[dict]]) ─────────────────────────────
#
# The backtester carries OHLCV as {ticker: [{date, open, high, low, close}, ...]}
# — that shape is not directly parquet-friendly because pandas would
# require one frame per ticker. Instead we stack all bars into a single
# frame with a `ticker` column. Load reconstructs the original dict.
# ~900 tickers × 2500 rows = ~2.3M rows, typical size ~50-100 MB on disk.


def save_ohlcv_by_ticker(
    bucket: str, date: str, phase: str, name: str,
    ohlcv_by_ticker: dict,
    *, s3_client=None,
) -> str:
    """Persist ``ohlcv_by_ticker`` (DataFrame shape only — Option A
    step 9 cleanup deleted the legacy list-of-dicts producer).

    Delegates to ``save_dict_of_dataframes`` which adds a ``ticker``
    column and an ``__idx__`` column carrying the DatetimeIndex.
    """
    if not ohlcv_by_ticker:
        return save_dict_of_dataframes(
            bucket, date, phase, name, ohlcv_by_ticker, s3_client=s3_client,
        )
    sample = next(iter(ohlcv_by_ticker.values()))
    if not isinstance(sample, pd.DataFrame):
        raise TypeError(
            f"save_ohlcv_by_ticker: expected dict[str, pd.DataFrame] post "
            f"Option A step 9 cleanup; got value of type "
            f"{type(sample).__name__}. Producer must use "
            f"build_ohlcv_df_by_ticker."
        )
    return save_dict_of_dataframes(
        bucket, date, phase, name, ohlcv_by_ticker, s3_client=s3_client,
    )


def load_ohlcv_by_ticker(
    bucket: str, key: str, *, s3_client=None,
) -> dict:
    """Inverse of ``save_ohlcv_by_ticker``. Returns
    ``dict[str, pd.DataFrame]`` (DatetimeIndex per ticker).

    Hard-fails on legacy artifacts (parquet without ``__idx__`` column)
    so a stale pre-Option-A artifact from a prior run can't silently
    feed list-of-dicts shape to the now-DataFrame-only consumers. The
    auto-skip layer treats this as a real failure and re-runs the
    upstream phase.
    """
    df = load_dataframe(bucket, key, s3_client=s3_client)
    if df.empty:
        return {}
    if "ticker" not in df.columns:
        raise ValueError(
            f"load_ohlcv_by_ticker: parquet at {key!r} missing 'ticker' column "
            f"(found: {list(df.columns)})"
        )
    if "__idx__" not in df.columns:
        raise ValueError(
            f"load_ohlcv_by_ticker: parquet at {key!r} is on the legacy "
            f"list-of-dicts schema (no '__idx__' column). Option A step 9 "
            f"cleanup removed legacy support — re-run the upstream "
            f"predictor_data_prep phase to regenerate the artifact in "
            f"DataFrame shape."
        )
    out_df: dict[str, pd.DataFrame] = {}
    for ticker, group in df.groupby("ticker", sort=False):
        frame = group.drop(columns=["ticker"]).copy()
        frame = frame.set_index("__idx__")
        frame.index.name = None
        out_df[str(ticker)] = frame
    return out_df


# ── Series artifacts (e.g. spy_prices) ───────────────────────────────────────


def save_series(
    bucket: str, date: str, phase: str, name: str, series: pd.Series,
    *, s3_client=None,
) -> str:
    """Persist a pd.Series as a single-column parquet. The Series name (if
    set) is preserved as the column name; index is kept."""
    df = series.to_frame(name=series.name or "value")
    return save_dataframe(
        bucket, date, phase, name, df, s3_client=s3_client, preserve_index=True,
    )


def load_series(bucket: str, key: str, *, s3_client=None) -> pd.Series:
    """Inverse of save_series — returns the single column as a Series."""
    df = load_dataframe(bucket, key, s3_client=s3_client)
    if df.shape[1] != 1:
        raise ValueError(
            f"load_series: parquet at {key!r} has {df.shape[1]} columns "
            f"(expected 1 for a Series round-trip)"
        )
    col = df.columns[0]
    return df[col]


# ── Dict-of-Series artifacts (e.g. vwap_series_by_ticker) ────────────────────


def save_dict_of_series(
    bucket: str, date: str, phase: str, name: str,
    data: dict[str, pd.Series],
    *, s3_client=None,
) -> str:
    """Stack per-ticker Series into (ticker, idx, value) rows + parquet upload.

    For ~900 tickers × ~2500 bars the compressed size is small (~20-40 MB).
    On load, rebuilt as dict[ticker → Series(index=idx)]. Series dtype +
    index dtype round-trip via parquet.
    """
    rows = []
    for ticker, s in data.items():
        if s is None:
            continue
        for idx, value in s.items():
            rows.append({"ticker": str(ticker), "idx": idx, "value": value})
    df = pd.DataFrame(rows)
    return save_dataframe(
        bucket, date, phase, name, df,
        s3_client=s3_client, preserve_index=False,
    )


def load_dict_of_series(
    bucket: str, key: str, *, s3_client=None,
) -> dict[str, pd.Series]:
    """Inverse of save_dict_of_series."""
    df = load_dataframe(bucket, key, s3_client=s3_client)
    if df.empty:
        return {}
    missing = {"ticker", "idx", "value"} - set(df.columns)
    if missing:
        raise ValueError(
            f"load_dict_of_series: parquet at {key!r} missing columns "
            f"{sorted(missing)} (found: {list(df.columns)})"
        )
    out: dict[str, pd.Series] = {}
    for ticker, group in df.groupby("ticker", sort=False):
        s = pd.Series(
            group["value"].values, index=group["idx"].values, name=str(ticker),
        )
        out[str(ticker)] = s
    return out


# ── Dict-of-DataFrames artifacts (e.g. features_by_ticker) ───────────────────


def _promote_numeric_dtype_drift(
    ticker_frames: list[tuple[str, pd.DataFrame]],
) -> None:
    """In-place: when a numeric column has different dtypes across ticker
    frames, cast every instance to ``float64`` so ``pa.unify_schemas``
    sees one common type. Handles all numeric drift cases:

    * ``int*`` vs ``float*`` (e.g. Volume int64 vs float64 — recent
      listings whose early bars had NaNs that pandas auto-cast to
      float64; observed 2026-04-26 across 107 of 911 tickers).
    * ``float32`` vs ``float64`` (e.g. rsi_14 float vs double —
      float-precision divergence between feature-write code paths;
      observed 2026-04-26 on rsi_14 after the Volume case was fixed).
    * Mixed int widths (int32 vs int64), uint vs int, etc.

    Float64 is the widest numeric type and lossless for the value
    ranges typical of OHLCV / feature data. Casting ``everything``
    in a drift column to float64 — not just the narrower instances —
    keeps the resulting parquet schema homogeneous on disk.

    Non-numeric drift (string vs numeric, datetime vs object, etc.)
    is left alone: those are real semantic divergences that should
    still surface as ``pa.unify_schemas`` errors so we investigate
    rather than silently coerce.

    No-op when zero columns drift, so the common case (all tickers
    homogeneous) pays only an O(N_frames × N_cols) dtype scan.

    Motivation: see 2026-04-26 c5.large validation arc — the optimizer
    arc's full-universe ``save_dict_of_dataframes`` call was the first
    code path to attempt schema unification across all 911 tickers,
    surfacing pre-existing dtype drift that was always on disk.
    """
    if not ticker_frames:
        return
    from collections import defaultdict
    import numpy as np

    col_dtypes: dict[str, set[str]] = defaultdict(set)
    for _, frame in ticker_frames:
        for col in frame.columns:
            col_dtypes[col].add(str(frame[col].dtype))

    drift_cols: list[str] = []
    for col, dtypes in col_dtypes.items():
        if len(dtypes) <= 1:
            continue
        # Numeric-only drift — handles int/uint/float of any width.
        # Object/string/datetime mixed with numeric still raises at
        # unify_schemas (intended).
        all_numeric = all(
            np.issubdtype(np.dtype(d), np.number) for d in dtypes
        )
        if all_numeric:
            drift_cols.append(col)

    if not drift_cols:
        return

    import logging
    logging.getLogger(__name__).info(
        "save_dict_of_dataframes: promoting %d numeric drift column(s) "
        "to float64: %s",
        len(drift_cols), drift_cols,
    )
    for _, frame in ticker_frames:
        for col in drift_cols:
            if col in frame.columns:
                # Cast all instances (even already-float64) so the
                # resulting tables share an exact dtype identity. A
                # float64 ``.astype('float64')`` is a no-op pass-through.
                if str(frame[col].dtype) != "float64":
                    frame[col] = frame[col].astype("float64")


def _unify_column_order(
    ticker_frames: list[tuple[str, pd.DataFrame]],
) -> None:
    """In-place: reindex every DataFrame to a common column order
    (the union of all observed columns, in stable first-seen order),
    NaN-filling any missing columns.

    Motivated by the 2026-04-26 v4 spot validation: the dtype-drift
    fix unblocked ``unify_schemas`` but exposed the next layer —
    ``pa.Table.cast(unified_schema)`` raises ``Target schema's field
    names are not matching the table's field names`` when the field
    names match by SET but differ in ORDER. Different ArcticDB ingest
    paths produced rotated views of the same column set across
    tickers; the legacy ``pd.concat`` producer sorted columns
    silently, but the streaming-write path doesn't.

    Algorithm: stable first-seen union. Walk each frame in order,
    append unseen columns to a master list. Then reindex every frame
    to that master list. Missing columns become NaN. Order is
    deterministic (depends only on dict-iter order) so the on-disk
    schema is reproducible run-to-run.

    Reindexing returns a new DataFrame — we replace ticker_frames
    entries in place via index assignment so the caller's list
    reflects the canonicalized frames.
    """
    if not ticker_frames:
        return
    seen: set = set()
    union_cols: list[str] = []
    for _, frame in ticker_frames:
        for col in frame.columns:
            if col not in seen:
                seen.add(col)
                union_cols.append(col)

    # Skip the reindex pass entirely if every frame already has the
    # same column ordering as the union — common case is homogeneous
    # tickers, no-op shouldn't allocate.
    needs_reindex = any(
        list(frame.columns) != union_cols
        for _, frame in ticker_frames
    )
    if not needs_reindex:
        return

    import logging
    logging.getLogger(__name__).info(
        "save_dict_of_dataframes: reindexing %d ticker frame(s) to a "
        "common %d-column order (column-order drift across ingest paths)",
        sum(1 for _, f in ticker_frames if list(f.columns) != union_cols),
        len(union_cols),
    )
    for i, (ticker, frame) in enumerate(ticker_frames):
        if list(frame.columns) != union_cols:
            ticker_frames[i] = (ticker, frame.reindex(columns=union_cols))


def save_dict_of_dataframes(
    bucket: str, date: str, phase: str, name: str,
    data: dict[str, pd.DataFrame],
    *, s3_client=None,
) -> str:
    """Stream per-ticker DataFrames into one parquet with a 'ticker' column.

    The original DataFrame index is materialized as a column named
    `__idx__` so it round-trips; on load the per-ticker frames set that
    column back as the index. Columns across tickers must be a compatible
    superset — a ticker missing column C will load with NaN for C.

    Used for features_by_ticker in predictor_data_prep. ~900 tickers
    × ~2500 rows × 59 feature cols compresses to ~150-250 MB parquet.

    Memory-efficient streaming write: uses pyarrow ParquetWriter to
    append one row group per ticker instead of building a stacked
    in-memory DataFrame via pd.concat. Motivated by the 2026-04-24
    dry-run's predictor_data_prep save block spending 20+ min
    thrashing swap on c5.large — the prior pd.concat approach held
    (dict + parts list + concat result + parquet buffer) = ~2.3 GB
    simultaneously. This path peaks at ~1 ticker (~1.4 MB) + the
    accumulating compressed parquet buffer (~250 MB final) = ~300 MB
    peak, ~8× reduction.

    Schema unification: when tickers have mismatched columns, pyarrow's
    default behavior rejects the second row group. We unify by building
    all pyarrow Tables first (cheap, zero-copy from pandas), then
    `pa.unify_schemas()` on their schemas and casting each table to the
    unified schema before appending. This preserves the legacy
    "compatible superset" semantic (missing column → NaN on load).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Filter + build the reset-and-annotated DataFrames, but do NOT
    # concat them. Each becomes a pyarrow Table below.
    ticker_frames: list[tuple[str, pd.DataFrame]] = []
    for ticker, df in data.items():
        if df is None or df.empty:
            continue
        reset = df.reset_index().rename(columns={df.index.name or "index": "__idx__"})
        reset.insert(0, "ticker", str(ticker))
        ticker_frames.append((str(ticker), reset))

    if not ticker_frames:
        # Empty-data fallback preserves the legacy shape: a parquet with
        # just the ticker + __idx__ column headers. Load path returns {}.
        empty = pd.DataFrame(columns=["ticker", "__idx__"])
        return save_dataframe(
            bucket, date, phase, name, empty,
            s3_client=s3_client, preserve_index=False,
        )

    # Type-drift unification (added 2026-04-26 after the c5.large
    # validation arc surfaced full-universe dtype drift on disk):
    # pyarrow's ``unify_schemas`` doesn't auto-promote across numeric
    # widths. Volume (int64 vs float64) and rsi_14 (float32 vs float64)
    # were both observed in production. ``_promote_numeric_dtype_drift``
    # casts every drift column to float64 (lossless for OHLCV/feature
    # ranges); non-numeric drift still raises at unify_schemas.
    _promote_numeric_dtype_drift(ticker_frames)

    # Column-order unification (added 2026-04-26 v4): reindex every
    # frame to a stable union order, NaN-filling missing columns, so
    # the schema is identical across frames before pyarrow conversion.
    _unify_column_order(ticker_frames)

    # After the two normalization passes above, every frame shares
    # the same column set, dtype, and order — so pa.Table.from_pandas
    # on any of them produces the same schema. Take the first frame's
    # schema as the unified schema and stream-write the rest.
    #
    # Memory: pre-2026-04-26 v5 this function held a list of 911
    # pa.Table objects (~1.1 GB raw data) + the original 911 DataFrames
    # (~1.1 GB) + dtype-promoted copies (cast int→float and float32→
    # float64 widen ~2×) + ParquetWriter buffer (grows to ~250 MB).
    # The 2026-04-26 v5 spot showed RSS jumping from 1331 MB at
    # post_inference to 2492 MB at post_feature_free — pyarrow's
    # internal allocator pool retained the table memory even after
    # the tables list went out of scope, contributing ~1 GB to the
    # final OOM at post_build_signals.
    #
    # Stream-write fix: build one pa.Table at a time, write it, free
    # immediately. Then ``pa.default_memory_pool().release_unused()``
    # forces the pool to return memory to the OS rather than retain
    # for reuse. Peak in this function drops from ~3.5 GB transient
    # to ~1.5 GB transient (just the caller's features + one in-flight
    # table + parquet buffer).
    sample_ticker, sample_frame = ticker_frames[0]
    sample_table = pa.Table.from_pandas(sample_frame, preserve_index=False)
    unified_schema = sample_table.schema
    del sample_table

    buf = io.BytesIO()
    writer = pq.ParquetWriter(buf, unified_schema, compression="snappy")
    try:
        for i in range(len(ticker_frames)):
            ticker, frame = ticker_frames[i]
            table = pa.Table.from_pandas(
                frame, preserve_index=False, schema=unified_schema,
            )
            writer.write_table(table)
            # Drop our reference to the frame so GC can reclaim it
            # before the next iteration's allocation. Caller still
            # holds the original via the input dict; we only release
            # the per-iteration reset+annotated copy.
            ticker_frames[i] = (ticker, None)
            del table, frame
    finally:
        writer.close()

    # Force pyarrow's allocator pool to return unused memory to the
    # OS. Without this, the 911-table allocation footprint stays
    # resident as "free but reusable" pool memory, contributing to
    # the post_feature_free RSS spike that took the 2026-04-26 v5
    # spot to OOM at post_build_signals=3039 MB.
    try:
        pa.default_memory_pool().release_unused()
    except Exception:
        # (a) release_unused() unsupported/rejected on this pyarrow
        # version. (c) no recording surface -- carve-out: best effort
        # memory-pool cleanup hint, older pyarrow versions may not expose
        # this API; the save must not fail on a rejected cleanup hint
        # (alpha-engine-config-I10226).
        pass

    key = artifact_key(date, phase, name, "parquet")
    _client(s3_client).put_object(
        Bucket=bucket, Key=key, Body=buf.getvalue(),
        ContentType="application/vnd.apache.parquet",
    )
    return key


def load_dict_of_dataframes(
    bucket: str, key: str, *, s3_client=None,
) -> dict[str, pd.DataFrame]:
    """Inverse of save_dict_of_dataframes."""
    stacked = load_dataframe(bucket, key, s3_client=s3_client)
    if stacked.empty or "ticker" not in stacked.columns:
        if stacked.empty:
            return {}
        raise ValueError(
            f"load_dict_of_dataframes: parquet at {key!r} missing 'ticker' "
            f"column (found: {list(stacked.columns)})"
        )
    out: dict[str, pd.DataFrame] = {}
    for ticker, group in stacked.groupby("ticker", sort=False):
        df = group.drop(columns=["ticker"]).copy()
        if "__idx__" in df.columns:
            df = df.set_index("__idx__")
            df.index.name = None
        out[str(ticker)] = df.reset_index(drop=True) if df.index.name == "index" else df
    return out


# ── signals_by_date — flat parquet persistence + lazy load ──────────────────
#
# Pre-2026-04-26: persisted as a single ``signals_by_date.json`` blob via
# ``save_json``. With ~38k buy_candidates + ~2M universe records across
# ~2,316 dates, the JSON serialization step took 27 minutes per Saturday
# SF run on c5.large — a single ``json.dumps`` on a ~1 GB dict-of-dicts
# without streaming. The 2026-04-26 v6 c5.large optimization-arc
# validation tripped the predictor_data_prep watchdog at 32 min, with
# all the slowness in this serialize step.
#
# Solution: split the envelope into two flat parquets — one row per
# ticker entry, one row per date for metadata. Stream-write per-date so
# memory peak stays bounded. Load as a ``LazySignalsByDate`` mapping
# wrapper that materializes per-date envelopes on demand from the
# in-memory DataFrames. Resident set drops from ~1 GB dict-of-dicts to
# ~150 MB DataFrame; serialize time drops from ~27 min to ~30-60s.
#
# Schema:
#   records.parquet  — columns: date, bucket, ticker, signal, score,
#                      conviction, sector, rating
#                      (one row per ticker entry, both buckets)
#   metadata.parquet — columns: date, market_regime, sector_ratings_json
#                      (one row per signal date)


def save_signals_by_date_flat(
    bucket: str, date: str, phase: str, name: str,
    signals_by_date,
    *, s3_client=None,
) -> tuple[str, str]:
    """Stream-write ``signals_by_date`` as two flat parquets.

    Returns ``(records_key, metadata_key)`` for the caller to register
    on the phase artifact list. Records and metadata are written to
    ``{name}_records.parquet`` and ``{name}_metadata.parquet``
    respectively under the canonical phase-artifact path.

    Stream-writes per-date so the in-memory peak during this function
    stays bounded by one date's records (~900 rows × 8 cols = ~50 KB).
    The records ParquetWriter buffer accumulates compressed output as
    rows append.

    Empty ``signals_by_date`` produces empty-header parquets so
    downstream load returns an empty Mapping.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    records_key = artifact_key(date, phase, f"{name}_records", "parquet")
    metadata_key = artifact_key(date, phase, f"{name}_metadata", "parquet")

    # Empty fallback — write empty headers so the load path's strict
    # column checks don't trip on a missing artifact.
    if not signals_by_date:
        empty_records = pd.DataFrame(columns=[
            "date", "bucket", "ticker", "signal", "score",
            "conviction", "sector", "rating",
        ])
        empty_metadata = pd.DataFrame(columns=[
            "date", "market_regime", "sector_ratings_json",
        ])
        save_dataframe(
            bucket, date, phase, f"{name}_records", empty_records,
            s3_client=s3_client, preserve_index=False,
        )
        save_dataframe(
            bucket, date, phase, f"{name}_metadata", empty_metadata,
            s3_client=s3_client, preserve_index=False,
        )
        return records_key, metadata_key

    records_buf = io.BytesIO()
    records_writer: pq.ParquetWriter | None = None
    metadata_rows: list[dict] = []

    for sig_date in signals_by_date:
        envelope = signals_by_date[sig_date]
        rows: list[dict] = []
        for entry in (envelope.get("buy_candidates") or []):
            rows.append({
                "date": sig_date,
                "bucket": "buy",
                "ticker": str(entry.get("ticker") or ""),
                "signal": str(entry.get("signal") or ""),
                "score": float(entry.get("score") or 0.0),
                "conviction": str(entry.get("conviction") or ""),
                "sector": str(entry.get("sector") or ""),
                "rating": str(entry.get("rating") or ""),
            })
        for entry in (envelope.get("universe") or []):
            rows.append({
                "date": sig_date,
                "bucket": "universe",
                "ticker": str(entry.get("ticker") or ""),
                "signal": str(entry.get("signal") or ""),
                "score": float(entry.get("score") or 0.0),
                "conviction": str(entry.get("conviction") or ""),
                "sector": str(entry.get("sector") or ""),
                "rating": str(entry.get("rating") or ""),
            })
        if rows:
            df = pd.DataFrame(rows)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if records_writer is None:
                records_writer = pq.ParquetWriter(
                    records_buf, table.schema, compression="snappy",
                )
            records_writer.write_table(table)
            del table, df, rows

        metadata_rows.append({
            "date": str(sig_date),
            "market_regime": str(envelope.get("market_regime") or "neutral"),
            "sector_ratings_json": json.dumps(envelope.get("sector_ratings") or {}),
        })

    if records_writer is not None:
        records_writer.close()

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_table = pa.Table.from_pandas(metadata_df, preserve_index=False)
    metadata_buf = io.BytesIO()
    metadata_writer = pq.ParquetWriter(
        metadata_buf, metadata_table.schema, compression="snappy",
    )
    metadata_writer.write_table(metadata_table)
    metadata_writer.close()

    try:
        pa.default_memory_pool().release_unused()
    except Exception:
        # (a) release_unused() unsupported/rejected on this pyarrow
        # version. (c) no recording surface -- carve-out, same class as
        # the identical guard above (alpha-engine-config-I10226).
        pass

    s3 = _client(s3_client)
    s3.put_object(
        Bucket=bucket, Key=records_key, Body=records_buf.getvalue(),
        ContentType="application/vnd.apache.parquet",
    )
    s3.put_object(
        Bucket=bucket, Key=metadata_key, Body=metadata_buf.getvalue(),
        ContentType="application/vnd.apache.parquet",
    )

    return records_key, metadata_key


class LazySignalsByDate:
    """Mapping-shaped lazy-load wrapper around the flat-parquet
    signals_by_date artifacts produced by ``save_signals_by_date_flat``.

    On first access (any of ``__getitem__``, ``__iter__``, ``__len__``,
    ``__contains__``, ``keys``, ``values``, ``items``, ``get``) reads
    both parquets into pandas DataFrames and groups records by date for
    O(N_dates) per-date envelope lookup. Per-date envelope
    materialization happens on each ``__getitem__`` call from the
    in-memory grouped DataFrame slice.

    Memory: 911-ticker × 2316-date corpus is ~150 MB resident as
    DataFrames + transient ~430 KB per envelope materialization,
    versus ~1 GB for the dict-of-dicts shape this replaces.

    Used by ``backtest._run_simulation_loop`` (single_run sim,
    param_sweep) which iterates as ``for d in sorted(signals_by_date.
    keys()): env = signals_by_date[d]`` — supported. Phase 4 evaluators
    rebuild their own signals via ``build_signals_by_date``; they don't
    consume this lazy handle.

    Pickle-friendly (lazy state is reinitialized via ``__getstate__``).
    """

    def __init__(
        self,
        bucket: str,
        records_key: str,
        metadata_key: str,
        s3_client=None,
    ):
        self._bucket = bucket
        self._records_key = records_key
        self._metadata_key = metadata_key
        self._s3 = s3_client
        self._records_by_date: dict | None = None
        self._metadata_by_date: dict | None = None

    def __getstate__(self):
        # Drop loaded state on pickle so the unpickled instance reloads
        # lazily in the new process. Spot-side multiprocessing in
        # param_sweep historically pickled the closure carrying
        # signals_by_date — keep that path working.
        return {
            "_bucket": self._bucket,
            "_records_key": self._records_key,
            "_metadata_key": self._metadata_key,
            "_s3": None,  # boto3 clients aren't picklable; reload uses default
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._records_by_date = None
        self._metadata_by_date = None

    def _load(self) -> None:
        if self._records_by_date is not None:
            return
        records_df = load_dataframe(self._bucket, self._records_key, s3_client=self._s3)
        metadata_df = load_dataframe(self._bucket, self._metadata_key, s3_client=self._s3)
        # Group records by date — empty groupby on empty DataFrame is
        # fine, returns an empty dict.
        if records_df.empty:
            self._records_by_date = {}
        else:
            self._records_by_date = {
                str(d): group for d, group in records_df.groupby("date", sort=False)
            }
        if metadata_df.empty:
            self._metadata_by_date = {}
        else:
            self._metadata_by_date = {
                str(row["date"]): row for _, row in metadata_df.iterrows()
            }

    def _materialize_envelope(self, date_str: str) -> dict:
        """Build the per-date envelope dict from the loaded DataFrames.
        Caller is responsible for ensuring ``date_str`` is in the
        metadata index (or accept the KeyError)."""
        meta = self._metadata_by_date[date_str]
        records = self._records_by_date.get(date_str)
        buy: list[dict] = []
        universe: list[dict] = []
        if records is not None:
            for _, row in records.iterrows():
                entry = {
                    "ticker": str(row["ticker"]),
                    "signal": str(row["signal"]),
                    "score": float(row["score"]),
                    "conviction": str(row["conviction"]),
                    "sector": str(row["sector"]),
                    "rating": str(row["rating"]),
                }
                if str(row["bucket"]) == "buy":
                    buy.append(entry)
                else:
                    universe.append(entry)
        sr_raw = meta["sector_ratings_json"]
        if sr_raw is None or (isinstance(sr_raw, float) and pd.isna(sr_raw)):
            sector_ratings = {}
        else:
            sector_ratings = json.loads(sr_raw or "{}")
        return {
            "date": date_str,
            "market_regime": str(meta["market_regime"]),
            "sector_ratings": sector_ratings,
            "buy_candidates": buy,
            "universe": universe,
        }

    # ── Mapping interface ──────────────────────────────────────────

    def __getitem__(self, date_str):
        self._load()
        if date_str not in self._metadata_by_date:
            raise KeyError(date_str)
        return self._materialize_envelope(str(date_str))

    def __contains__(self, date_str):
        self._load()
        return date_str in self._metadata_by_date

    def __iter__(self):
        self._load()
        return iter(self._metadata_by_date)

    def __len__(self):
        self._load()
        return len(self._metadata_by_date)

    def keys(self):
        self._load()
        return list(self._metadata_by_date.keys())

    def values(self):
        # Eagerly materializes all envelopes. Acceptable because the
        # only legitimate caller is the metadata-totals computation in
        # ``predictor_backtest.run`` which runs BEFORE the lazy handle
        # is created (operates on the in-memory dict). Downstream
        # consumers iterate via ``keys()`` + ``__getitem__`` instead.
        self._load()
        return [self._materialize_envelope(d) for d in self._metadata_by_date]

    def items(self):
        self._load()
        return [(d, self._materialize_envelope(d)) for d in self._metadata_by_date]

    def get(self, date_str, default=None):
        try:
            return self[date_str]
        except KeyError:
            return default


def load_signals_by_date_lazy(
    bucket: str, records_key: str, metadata_key: str,
    *, s3_client=None,
) -> LazySignalsByDate:
    """Construct a ``LazySignalsByDate`` from the artifact keys produced
    by ``save_signals_by_date_flat``. Symmetric helper for the
    auto-skip reload path."""
    return LazySignalsByDate(
        bucket=bucket,
        records_key=records_key,
        metadata_key=metadata_key,
        s3_client=s3_client,
    )
