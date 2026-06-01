"""
synthetic/predictor_backtest.py — predictor-only historical backtest pipeline.

Runs GBM inference on up to 10 years of OHLCV data to generate synthetic
signals, then feeds them through the full executor pipeline (risk guard,
position sizing, ATR stops, time decay, graduated drawdown).

This tests everything downstream of Research without any LLM API calls:
    1. Load OHLCV + pre-computed features from ArcticDB (sole source)
    2. Recompute features inline only when ArcticDB coverage is insufficient
    3. Run GBM inference in daily batches (up to ~2520 days × ~900 tickers)
    4. Convert alpha predictions to executor-compatible signals
    5. Build price matrix + OHLCV histories for simulation loop

The caller (backtest.py) then passes these to _run_simulation_loop() with
the existing executor pipeline.

Data source (Phase 0 of backtester-audit-260415.md):
    ArcticDB universe library — OHLCV + 53 features per ticker.
    Legacy S3 parquet cache (predictor/price_cache/*.parquet) and local
    slim-cache fallbacks were removed on 2026-04-16; ArcticDB is the
    unified source shared with predictor training + inference.

Performance notes (10y on c5.large spot):
    - Feature computation: ~900 calls to compute_features() (~3-5 min)
    - GBM inference: ~2500 batch calls (~2-3 min)
    - Total runtime: ~8-12 min
"""

from __future__ import annotations

import datetime as _dt
import gc
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import time

import pandas as pd

import boto3

from synthetic.signal_generator import predictions_to_signals

logger = logging.getLogger(__name__)

# Macro series tickers in the slim cache (used for feature computation)
_MACRO_TICKERS = {"SPY", "^VIX", "^TNX", "^IRX", "GLD", "USO"}

# Sector ETF tickers (present in slim cache)
_SECTOR_ETFS = {"XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLC", "XLB"}

# Minimum OHLCV rows required for feature computation (52-week rolling windows ≈ 260 trading days + buffer)
_MIN_ROWS_FOR_FEATURES = 265


def _log_rss(label: str) -> None:
    """Log process RSS (resident set size) at a named checkpoint.

    Noisy but invaluable for catching OOM-class issues like the 2026-04-23
    SF dry-run where predictor_data_prep blew past c5.large's 4 GB budget
    inside load_universe_from_arctic + build_ohlcv_df_by_ticker. Without
    these checkpoints we had to diagnose via CloudWatch CPU patterns +
    SSM-agent death instead of seeing the memory curve directly.

    Uses /proc/self/status on Linux (primary target — spot instances).
    Falls back to resource.getrusage.ru_maxrss elsewhere (Darwin for
    local tests) which reports in KB on Linux but bytes on Darwin —
    the distinction doesn't matter for a diagnostic log.

    Safe to call on any platform: any failure is swallowed since this
    is pure observability and must never fail the caller."""
    try:
        rss_bytes = None
        # Linux path — /proc/self/status line "VmRSS:  N kB"
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2 and parts[-1].lower() == "kb":
                            rss_bytes = int(parts[1]) * 1024
                        break
        except FileNotFoundError:
            pass
        if rss_bytes is None:
            # Darwin / non-Linux fallback
            import resource
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            # ru_maxrss is KB on Linux but BYTES on Darwin — we
            # don't know which, but this path is only hit in tests.
            rss_bytes = rusage.ru_maxrss * 1024
        rss_mb = rss_bytes / (1024 * 1024)
        logger.info("MEM %s: RSS=%.0f MB", label, rss_mb)
    except Exception as exc:
        logger.debug("MEM %s: failed to sample RSS: %s", label, exc)


def load_sector_map(predictor_path: str) -> dict[str, str]:
    """Load sector_map.json mapping tickers to sector ETF symbols."""
    map_path = Path(predictor_path) / "data" / "cache" / "sector_map.json"
    if not map_path.exists():
        logger.warning("sector_map.json not found at %s", map_path)
        return {}
    with open(map_path) as f:
        return json.load(f)


def compute_all_features(
    price_data: dict[str, pd.DataFrame],
    sector_map: dict[str, str],
    predictor_path: str,
) -> dict[str, pd.DataFrame]:
    """
    Compute 29 technical features for each stock ticker (not macro/ETF tickers).

    Features are computed once per ticker for the full 2y series, then indexed
    by date during inference. This avoids ~450K redundant compute_features()
    calls (900 tickers × 500 dates).

    Parameters
    ----------
    price_data : {ticker: OHLCV DataFrame} from load_universe_from_arctic()
    sector_map : {ticker: sector_etf} from sector_map.json
    predictor_path : path to predictor repo root (for importing compute_features)

    Returns
    -------
    {ticker: featured_df} — DataFrames with 29 feature columns + original OHLCV,
    rows with insufficient history already dropped.
    """
    # Import predictor's feature engineer
    if predictor_path not in sys.path:
        sys.path.insert(0, predictor_path)
    from data.feature_engineer import compute_features

    # Extract macro series from the cache
    spy_series = _extract_close(price_data, "SPY")
    vix_series = _extract_close(price_data, "^VIX")
    tnx_series = _extract_close(price_data, "^TNX")
    irx_series = _extract_close(price_data, "^IRX")
    gld_series = _extract_close(price_data, "GLD")
    uso_series = _extract_close(price_data, "USO")

    # Stock tickers only (exclude macro/ETF series from feature computation)
    skip_tickers = _MACRO_TICKERS | _SECTOR_ETFS
    stock_tickers = [t for t in price_data if t not in skip_tickers]

    logger.info("Computing features for %d stock tickers...", len(stock_tickers))
    features_by_ticker: dict[str, pd.DataFrame] = {}
    skip_reasons = {"too_short": 0, "empty_features": 0, "computation_error": 0}

    for i, ticker in enumerate(stock_tickers):
        df = price_data[ticker]

        if len(df) < _MIN_ROWS_FOR_FEATURES:
            skip_reasons["too_short"] += 1
            logger.debug("Skip %s: too_short (%d rows < %d)", ticker, len(df), _MIN_ROWS_FOR_FEATURES)
            continue

        # Get the sector ETF series for this ticker
        sector_etf = sector_map.get(ticker)
        sector_etf_series = _extract_close(price_data, sector_etf) if sector_etf else None

        try:
            featured = compute_features(
                df,
                spy_series=spy_series,
                vix_series=vix_series,
                sector_etf_series=sector_etf_series,
                tnx_series=tnx_series,
                irx_series=irx_series,
                gld_series=gld_series,
                uso_series=uso_series,
            )
            if not featured.empty:
                features_by_ticker[ticker] = featured
            else:
                skip_reasons["empty_features"] += 1
                logger.debug("Skip %s: empty_features after compute", ticker)
        except Exception as e:
            logger.warning("Feature computation failed for %s: %s", ticker, type(e).__name__)
            skip_reasons["computation_error"] += 1

        if (i + 1) % 100 == 0:
            logger.info("  Features computed: %d/%d tickers", i + 1, len(stock_tickers))

    skipped = sum(skip_reasons.values())
    reasons = {k: v for k, v in skip_reasons.items() if v > 0}
    logger.info(
        "Feature computation: %d tickers OK, %d skipped%s",
        len(features_by_ticker), skipped,
        f" ({reasons})" if reasons else "",
    )
    return features_by_ticker, skip_reasons


def download_gbm_model(bucket: str = "alpha-engine-research", region: str = "us-east-1") -> str:
    """
    Download the v3 Layer-1A momentum GBM from S3 to a temp file.

    Source: ``predictor/weights/meta/momentum_model.txt`` — the Layer-1A
    quant GBM that the v3 meta-model uses as an input, re-trained every
    Saturday alongside the rest of the meta stack. Saved by the current
    ``GBMScorer.save`` which persists ``feature_names`` metadata, so the
    backtester's feature-alignment check (``scorer.feature_names``) works
    cleanly.

    Why Layer-1A specifically (not the Ridge meta-model): the 10y synthetic
    backtest needs a pure quant scorer fed per-ticker features. The Ridge
    meta combines quant output with a Research calibrator whose input
    (Research composite score) only exists from ~March 2026 onward —
    replaying the full v3 stack over 10y would require fabricating research
    signals for 9.8 years of history. Scoping predictor-backtest to Layer
    1A measures the quant component in isolation, which per
    feedback_component_baseline_validation is the right standalone
    baseline for a stacked ensemble.

    Previously loaded ``predictor/weights/gbm_latest.txt`` — a v2 artifact
    last updated 2026-03-28, ripped from production 2026-04-13. Every
    Saturday since has been measuring a dead model. Cleanup of the stale
    v2 S3 artifacts is tracked in ROADMAP P2 "v2 legacy artifact cleanup".

    Returns the local path to the downloaded model.
    """
    s3 = boto3.client("s3", region_name=region)
    return _download_gbm_to_temp(
        s3,
        bucket,
        "predictor/weights/meta/momentum_model.txt",
        "predictor/weights/meta/momentum_model.txt.meta.json",
    )


def _download_gbm_to_temp(s3, bucket: str, model_key: str, meta_key: str) -> str:
    """Download a momentum-GBM booster + its meta.json to a temp file.

    Key-parameterized core shared by the live-weights path
    (:func:`download_gbm_model`) and the point-in-time walk-forward path
    (:func:`run_walk_forward_inference`, which passes an archived
    ``predictor/weights/meta/archive/{date}/momentum_model.txt`` key
    resolved by :mod:`synthetic.pit_weights`). Same hard-fail discipline
    on both legs regardless of which key was requested: a missing booster
    is a PredictorTraining-pipeline problem, and a missing meta.json would
    crash the downstream ``feature_names`` alignment in a less useful place
    — so both raise :class:`RuntimeError` rather than silently degrading.
    Returns the local booster path (``<path>.meta.json`` sits beside it).
    """
    model_tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    model_tmp.close()
    try:
        s3.download_file(bucket, model_key, model_tmp.name)
        logger.info("Downloaded Layer-1A momentum GBM from s3://%s/%s", bucket, model_key)
    except Exception as exc:
        raise RuntimeError(
            f"Layer-1A momentum GBM not found at s3://{bucket}/{model_key}. "
            "Saturday PredictorTraining step must populate "
            f"{model_key} on each run — investigate the training "
            f"pipeline if this key is missing. Underlying error: {exc}"
        ) from exc

    # Download metadata — hard-fail if missing. The backtester hard-requires
    # feature_names from the meta.json for input alignment; a successful
    # download of the booster with no meta.json would crash downstream in a
    # less useful place.
    meta_path = model_tmp.name + ".meta.json"
    try:
        s3.download_file(bucket, meta_key, meta_path)
    except Exception as exc:
        raise RuntimeError(
            f"Layer-1A momentum GBM metadata not found at s3://{bucket}/"
            f"{meta_key}. feature_names alignment will fail without it. "
            f"Underlying error: {exc}"
        ) from exc

    return model_tmp.name


def build_inference_tensor(
    features_by_ticker: dict[str, pd.DataFrame],
    feature_names: list[str],
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """Materialize per-ticker feature DataFrames into a dense 3D tensor.

    Replaces the O(n_dates × n_tickers) Python-loop vector-collection
    pattern (911 × 2500 ≈ 2.28M inner ticks per inference pass) with a
    vectorized per-date slice: ``tensor[date_idx]`` → (n_tickers, n_features).

    Callers downstream compute a per-date validity mask via
    ``~np.isnan(tensor[di]).any(axis=1)`` so feature zeroing (for
    Phase 4c pruning) can be applied to the tensor before the mask
    without paying a full DataFrame.copy() per ticker.

    Returns
    -------
    tensor : np.ndarray (n_dates, n_tickers, n_features), float32
        NaN in slots where a (date, ticker) has no feature row.
    tickers : list[str]
        Ticker ordering matching axis=1. Callers zip this with
        ``np.where(valid_mask_row)[0]`` to recover ticker labels per date.
    date_to_idx : dict[str, int]
        Map "YYYY-MM-DD" → axis=0 index.
    """
    # Skip tickers missing any required feature column — matches the
    # legacy ``try: df[feature_names] except KeyError: continue`` path
    usable: dict[str, pd.DataFrame] = {}
    for ticker, df in features_by_ticker.items():
        try:
            usable[ticker] = df[feature_names]
        except KeyError:
            continue

    if not usable:
        return (
            np.empty((0, 0, len(feature_names)), dtype=np.float32),
            [],
            {},
        )

    # Union of all dates across usable tickers, as ISO strings (the
    # contract downstream consumers rely on — signal_generator,
    # build_signals_by_date keyed on "YYYY-MM-DD")
    all_dates: set[str] = set()
    for df in usable.values():
        all_dates.update(df.index.strftime("%Y-%m-%d"))
    sorted_dates = sorted(all_dates)
    date_to_idx = {d: i for i, d in enumerate(sorted_dates)}

    tickers = list(usable.keys())
    tensor = np.full(
        (len(sorted_dates), len(tickers), len(feature_names)),
        np.nan, dtype=np.float32,
    )

    for ti, ticker in enumerate(tickers):
        df = usable[ticker]
        # Duplicate dates: keep last, matches legacy ``dict(zip(dates, arr))``
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="last")]
        arr = df.to_numpy(dtype=np.float32)
        date_indices = np.fromiter(
            (date_to_idx[d] for d in df.index.strftime("%Y-%m-%d")),
            dtype=np.int64, count=len(df),
        )
        tensor[date_indices, ti, :] = arr

    return tensor, tickers, date_to_idx


def _predict_from_tensor(
    tensor: np.ndarray,
    tickers: list[str],
    date_to_idx: dict[str, int],
    trading_dates: list[str],
    scorer,
    heartbeat_every: int,
    log_label: str,
) -> dict[str, dict[str, float]]:
    """Run batched scorer.predict() over a prebuilt inference tensor.

    Validity is checked per (date, ticker) via isnan — same contract as
    the legacy ``vec is not None and not np.any(np.isnan(vec))`` gate.
    """
    predictions_by_date: dict[str, dict[str, float]] = {}
    t0 = time.monotonic() if hasattr(time, "monotonic") else 0.0

    for i, date_str in enumerate(trading_dates):
        di = date_to_idx.get(date_str)
        if di is None:
            continue
        # Per-date slice: (n_tickers, n_features) view, no copy
        day_matrix = tensor[di]
        row_mask = ~np.isnan(day_matrix).any(axis=1)
        if not row_mask.any():
            continue
        X = day_matrix[row_mask]
        alphas = scorer.predict(X)
        ticker_idxs = np.flatnonzero(row_mask)
        predictions_by_date[date_str] = {
            tickers[ti]: float(a)
            for ti, a in zip(ticker_idxs, alphas)
        }

        if (i + 1) % heartbeat_every == 0:
            elapsed = (time.monotonic() - t0) if hasattr(time, "monotonic") else 0.0
            logger.info(
                "%s inference: %d/%d dates (%.1fs elapsed, last=%s)",
                log_label, i + 1, len(trading_dates), elapsed, date_str,
            )

    return predictions_by_date


def run_inference(
    features_by_ticker: dict[str, pd.DataFrame],
    model_path: str,
    predictor_path: str,
    trading_dates: list[str] | None = None,
    zero_features: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Run GBM inference for all tickers across all trading dates.

    For each trading date, stacks feature vectors for all tickers with valid
    features on that date, runs one batch GBMScorer.predict() call, and
    returns predictions indexed by date.

    Parameters
    ----------
    features_by_ticker : {ticker: featured_df} from compute_all_features()
    model_path : local path to GBM model file
    predictor_path : path to predictor repo root (for importing GBMScorer)
    trading_dates : optional list of dates to run inference on. If None,
        uses the union of all available feature dates.
    zero_features : optional list of feature names to zero in-place on
        the inference tensor before prediction. Used by
        ``evaluate_feature_pruning`` to test noise-feature removal
        without paying the ~1.1 GB DataFrame.copy() cost of building a
        separate ``features_by_ticker`` dict. Matches the legacy
        ``_zero_out_features`` contract: NaN entries in a zeroed column
        become 0.0 (which may re-admit previously-invalid rows, same
        as the legacy path).

    Returns
    -------
    {date_str: {ticker: alpha_score}} — predictions per date per ticker.
    """
    if predictor_path not in sys.path:
        sys.path.insert(0, predictor_path)
    from model.gbm_scorer import GBMScorer

    scorer = GBMScorer.load(model_path)

    # Use the model's own trained feature list rather than the current
    # GBM_FEATURES config — they drift when new features land in config.py
    # before a fresh training run promotes weights. Slicing by the model's
    # feature_names guarantees the input matrix matches regardless of config
    # drift; a fresh training run will update this list automatically.
    GBM_FEATURES = scorer.feature_names
    if not GBM_FEATURES:
        raise RuntimeError(
            "Loaded model has no feature_names metadata — cannot align "
            "input features. Retrain with a newer GBMScorer that persists "
            "feature_names in the metadata JSON."
        )
    logger.info("Predictor backtest using %d model features: %s",
                len(GBM_FEATURES), GBM_FEATURES)

    # Determine trading dates from feature data if not provided
    if trading_dates is None:
        all_dates = set()
        for df in features_by_ticker.values():
            all_dates.update(df.index.strftime("%Y-%m-%d"))
        trading_dates = sorted(all_dates)

    logger.info(
        "Running GBM inference: %d tickers × %d dates",
        len(features_by_ticker), len(trading_dates),
    )

    logger.info("Building inference tensor...")
    tensor, tickers, date_to_idx = build_inference_tensor(
        features_by_ticker, GBM_FEATURES,
    )
    logger.info(
        "Inference tensor: shape=%s usable_tickers=%d",
        tensor.shape, len(tickers),
    )

    if zero_features:
        # Zero requested feature columns across all (date, ticker)
        # slots in-place on the tensor. Skips unknown features silently
        # to match the legacy _zero_out_features behavior.
        zero_idx = [
            GBM_FEATURES.index(f) for f in zero_features if f in GBM_FEATURES
        ]
        if zero_idx:
            tensor[:, :, zero_idx] = 0.0
            logger.info(
                "Zeroed %d feature column(s) on inference tensor: %s",
                len(zero_idx), [GBM_FEATURES[i] for i in zero_idx],
            )

    predictions_by_date = _predict_from_tensor(
        tensor, tickers, date_to_idx, trading_dates,
        scorer=scorer, heartbeat_every=50, log_label=" ",
    )

    logger.info(
        "Inference complete: %d dates with predictions",
        len(predictions_by_date),
    )
    return predictions_by_date


def build_signals_by_date(
    predictions_by_date: dict[str, dict[str, float]],
    sector_map: dict[str, str],
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    top_n: int = 20,
    min_score: float = 60,
) -> dict[str, dict]:
    """
    Convert per-date predictions to executor signal envelopes using
    technical scoring from OHLCV data (not the broken alpha-to-score mapping).

    Parameters
    ----------
    predictions_by_date : {date: {ticker: alpha}} from run_inference()
    sector_map : {ticker: sector_etf} from sector_map.json
    ohlcv_by_ticker : {ticker: pd.DataFrame} (DatetimeIndex + lowercase
        open/high/low/close columns) per ``build_ohlcv_df_by_ticker``.
    top_n : max ENTER signals per day (prevents unrealistic portfolio churn)
    min_score : minimum trading score for ENTER signal

    Returns
    -------
    {date: signal_envelope} — each envelope is a full signals_override dict.

    Performance notes
    -----------------
    Pre-2026-04-21 implementation ran at ~2.2s per date × 2277 dates ≈ 75 min,
    which pushed the Saturday SF past its 7200s SSM ceiling. The bottleneck
    was the inner loop rebuilding ``ohlcv_up_to_date`` by scanning every
    ticker's full 10y bar list per date (~5B Python string comparisons
    total). The data already has a date axis; pandas can roll every
    indicator in one vectorized pass per ticker.

    This revision: one-shot ``precompute_indicator_series(ohlcv_by_ticker)``
    produces per-ticker date-indexed DataFrames of all 6 indicators. The
    per-date loop then does O(1) hashtable lookups via
    ``indicators_from_precomputed``. Expected speedup ~50-100x (verified
    on synthetic + production data).
    """
    from synthetic.signal_generator import (
        precompute_indicator_series,
        indicators_from_precomputed,
    )

    signals_by_date: dict[str, dict] = {}
    sorted_dates = sorted(predictions_by_date.keys())

    # One-shot vectorized indicator pass over the full history per ticker.
    logger.info(
        "  Precomputing indicator series for %d tickers (vectorized)...",
        len(ohlcv_by_ticker),
    )
    t_pre = time.time()
    precomputed = precompute_indicator_series(ohlcv_by_ticker)
    logger.info(
        "  Precompute complete: %d tickers indexed in %.1fs",
        len(precomputed), time.time() - t_pre,
    )

    for i, date_str in enumerate(sorted_dates):
        predictions = predictions_by_date[date_str]

        # O(1) hashtable lookup per ticker — the hot path that was an
        # O(bars) list-comp scan before.
        indicators_this_date = indicators_from_precomputed(
            precomputed, predictions.keys(), date_str,
        )

        envelope = predictions_to_signals(
            predictions=predictions,
            date=date_str,
            sector_map=sector_map,
            precomputed_indicators=indicators_this_date,
            top_n=top_n,
            min_score=min_score,
        )
        signals_by_date[date_str] = envelope

        if (i + 1) % 250 == 0:
            n_enter = len(envelope.get("buy_candidates", []))
            logger.info(
                "  Signal generation: %d/%d dates (ENTER=%d on %s)",
                i + 1, len(sorted_dates), n_enter, date_str,
            )

    return signals_by_date


def build_price_matrix(
    price_data: dict[str, pd.DataFrame],
    trading_dates: list[str],
) -> pd.DataFrame:
    """
    Build a price matrix from slim cache data (same format as price_loader.build_matrix).

    Returns DataFrame with DatetimeIndex (dates) and ticker columns,
    values are close prices.
    """
    # Only include stock tickers (not macro/ETF series)
    skip_tickers = _MACRO_TICKERS | _SECTOR_ETFS
    stock_tickers = [t for t in price_data if t not in skip_tickers]

    records = {}
    for ticker in stock_tickers:
        df = price_data[ticker]
        close_col = "Close" if "Close" in df.columns else "close"
        if close_col not in df.columns:
            continue
        close = df[close_col]
        records[ticker] = close

    matrix = pd.DataFrame(records)
    # Filter to trading dates only
    matrix.index = pd.to_datetime(matrix.index)
    date_index = pd.to_datetime(trading_dates)
    matrix = matrix.reindex(date_index)

    logger.info(
        "Price matrix: %d dates × %d tickers (%.1f%% fill)",
        len(matrix), len(matrix.columns),
        matrix.notna().sum().sum() / max(matrix.size, 1) * 100,
    )
    return matrix


def build_ohlcv_df_by_ticker(
    price_data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Produce the DataFrame-form ohlcv_by_ticker — the new shape for the
    backtester memory refactor (plan 2026-04-23). Each value is a
    pd.DataFrame with:

      - DatetimeIndex (sorted ascending, no duplicates)
      - Columns [open, high, low, close], dtype float64

    Normalizes ArcticDB's capitalized column names to lowercase at the
    producer boundary so downstream consumers (executor, indicator
    compute, artifact persistence) can rely on a single canonical
    column naming without per-site case handling.

    Missing OHL columns (frequent in thin-history tickers where only
    Close is populated) fall back to the Close series.

    Macro + sector ETFs are filtered at the producer boundary — they
    aren't used downstream in the list form either.

    This is the low-overhead shape (~91 MB for 911 tickers × 2500 bars
    vs. ~1.1 GB for the equivalent list-of-dicts, where Python dict
    header overhead dominates at ~240 B/row). Kills the backtester
    OOM-on-c5.large risk diagnosed in the 2026-04-23 root-cause
    analysis (see alpha-engine-backtester-pandas-refactor-plan-260423.md).
    """
    skip_tickers = _MACRO_TICKERS | _SECTOR_ETFS
    out: dict[str, pd.DataFrame] = {}
    for ticker, df in price_data.items():
        if ticker in skip_tickers:
            continue
        if df is None or df.empty:
            continue
        cols: dict[str, pd.Series] = {}
        for title in ("Close", "Open", "High", "Low"):
            lower = title.lower()
            if title in df.columns:
                cols[lower] = df[title]
            elif lower in df.columns:
                cols[lower] = df[lower]
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
        out[ticker] = frame
    return out


def _compute_adv_dollar(price_data: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Per-ticker average daily DOLLAR volume (median of close×volume over the
    full loaded history) — the ADV input to the W3.4 transaction-cost model's
    √-impact term (``analysis.horizon_net_alpha``). A single per-ticker liquidity
    scalar is sufficient: at our book size the impact term is near-negligible, so
    a representative median beats threading a full ADV series. Names without a
    Volume column are omitted (the cost model degrades to half-spread +
    commission for them — the conservative fallback, not a silent zero)."""
    out: dict[str, float] = {}
    for t, df in price_data.items():
        if df is None or getattr(df, "empty", True):
            continue
        close_col = "Close" if "Close" in df.columns else ("close" if "close" in df.columns else None)
        vol_col = "Volume" if "Volume" in df.columns else ("volume" if "volume" in df.columns else None)
        if not close_col or not vol_col:
            continue
        dv = (df[close_col].astype(float) * df[vol_col].astype(float)).dropna()
        if len(dv) > 0:
            v = float(dv.median())
            if np.isfinite(v) and v > 0:
                out[t] = v
    return out


def _extract_close(price_data: dict[str, pd.DataFrame], ticker: str | None) -> pd.Series | None:
    """Extract Close price series for a given ticker, or None if not found."""
    if ticker is None or ticker not in price_data:
        return None
    df = price_data[ticker]
    if "Close" in df.columns:
        return df["Close"]
    elif "close" in df.columns:
        return df["close"]
    return None


def _resolve_trading_dates(
    features_by_ticker: dict[str, pd.DataFrame],
    min_trading_days: int,
    max_trading_days: int,
) -> list[str] | dict:
    """Determine common trading dates from feature data.

    Returns sorted date list on success, or error dict if insufficient dates.
    """
    all_dates = set()
    for df in features_by_ticker.values():
        all_dates.update(df.index.strftime("%Y-%m-%d"))
    trading_dates = sorted(all_dates)

    if len(trading_dates) < min_trading_days:
        return {
            "status": "insufficient_data",
            "dates_available": len(trading_dates),
            "min_required": min_trading_days,
            "note": f"Only {len(trading_dates)} trading dates with features "
                    f"(need {min_trading_days})",
        }

    if len(trading_dates) > max_trading_days:
        trading_dates = trading_dates[-max_trading_days:]
        logger.info(
            "Trimmed to most recent %d trading dates (from %s to %s)",
            len(trading_dates), trading_dates[0], trading_dates[-1],
        )
    else:
        logger.info(
            "Trading dates: %d (from %s to %s)",
            len(trading_dates), trading_dates[0], trading_dates[-1],
        )

    return trading_dates


# Plan defaults for the purged + embargoed walk-forward fold scheme
# (alpha-engine-docs/private/pit-discipline-260515.md §D1). Every value is
# config-overridable via predictor_backtest.walk_forward_params so a later
# experiment can sweep the fold scheme without a code change.
#   test_window — ~21 trading days (~1mo), the weekly-Saturday weight-promotion
#                 cadence the simulation must respect (plan D1).
#   min_train   — 504 (~2y), mirrors the predictor's WF_MIN_TRAIN_DAYS for
#                 system-wide consistency (plan §3 scope discipline). The train
#                 block is not refit here — archived weights are resolved — so
#                 this only governs how much un-archived warmup history is
#                 skipped before the first scorable fold.
#   purge       — 21 = canonical label horizon (plan invariant 2; the
#                 predictor's own WF_PURGE_DAYS=5 is for its 5d-era labels,
#                 superseded by the 21d canonical-alpha cutover).
#   embargo     — 2 trading days, LdP Ch.7 lower bound (plan D1 / §7). This is
#                 the plan's addition over the predictor (purge but no embargo).
#   train_mode  — "expanding" genuinely matches the predictor
#                 (meta_trainer.py train_mask = d <= train_end, no lower bound;
#                 the plan doc's "rolling matches predictor" wording is a known
#                 doc error corrected in pit_folds.py).
_WF_DEFAULTS = {
    "test_window": 21,
    "min_train": 504,
    "purge": 21,
    "embargo": 2,
    "train_mode": "expanding",
}


def run_walk_forward_inference(
    features_by_ticker: dict[str, pd.DataFrame],
    trading_dates: list[str],
    predictor_path: str,
    *,
    bucket: str,
    region: str = "us-east-1",
    wf_params: dict | None = None,
    s3_client=None,
) -> tuple[dict[str, dict[str, float]], dict]:
    """Point-in-time-honest replacement for the single-pass ``run_inference``.

    Slice C of PR 1, ROADMAP L2371 / Backtester Phase 2 (plan
    ``alpha-engine-docs/private/pit-discipline-260515.md``). The single-pass
    path replays *all* history against the **current live** momentum GBM —
    look-ahead contamination, since the optimizer's param sweep then selects
    parameters on future-trained weights. This path instead, for each
    purged + embargoed walk-forward fold, resolves the latest archived
    momentum weights whose *knowledge time ≤ the fold's test-window start*
    (:func:`synthetic.pit_weights.resolve_momentum_weights`) and scores only
    that fold's test window with them.

    No-future-fallback (plan invariant 3 / the central trap): a fold with no
    archived weights on-or-before its decision date raises
    :class:`~synthetic.pit_weights.ColdStartExclusion`; the fold is **skipped
    and counted**, never resolved to the nearest / earliest-future snapshot.
    The cold-start-excluded count is a first-class run-quality output (a high
    count is itself a finding about archive coverage).

    Performance: the inference tensor is built once per *distinct scorer
    feature signature* (one in practice — the momentum feature set is stable
    across weekly retrains) and reused across every fold, so the cost stays
    ≈ the single-pass path rather than O(n_folds × full-tensor). Distinct
    archived boosters are downloaded once and cached by archive date.

    Returns ``(predictions_by_date, wf_stats)`` where ``predictions_by_date``
    is the same ``{date: {ticker: alpha}}`` contract the single-pass path
    returns (sparse: only scored test-window dates are present — warmup /
    cold-start dates are "not investable yet" and correctly absent), and
    ``wf_stats`` is the metadata block surfaced under
    ``metadata["walk_forward"]``.
    """
    from synthetic.pit_weights import (
        ColdStartExclusion,
        resolve_momentum_weights,
    )
    from synthetic.pit_folds import build_walk_forward_folds

    if predictor_path not in sys.path:
        sys.path.insert(0, predictor_path)
    from model.gbm_scorer import GBMScorer

    p = {**_WF_DEFAULTS, **(wf_params or {})}
    s3 = s3_client if s3_client is not None else boto3.client("s3", region_name=region)

    # trading_dates is the sorted-unique axis from _resolve_trading_dates;
    # fold indices index straight back into it so date<->index stays aligned.
    date_objs = [_dt.date.fromisoformat(d) for d in trading_dates]
    folds = build_walk_forward_folds(
        date_objs,
        test_window=p["test_window"],
        min_train=p["min_train"],
        purge=p["purge"],
        embargo=p["embargo"],
        train_mode=p["train_mode"],
    )
    logger.info(
        "[walk_forward] %d fold(s) over %d trading dates (%s..%s); "
        "test_window=%d min_train=%d purge=%d embargo=%d mode=%s",
        len(folds), len(trading_dates),
        trading_dates[0] if trading_dates else "-",
        trading_dates[-1] if trading_dates else "-",
        p["test_window"], p["min_train"], p["purge"], p["embargo"],
        p["train_mode"],
    )

    # Caches keyed so the expensive bits happen once: boosters by archive
    # date, tensors by the scorer's feature signature (tuple of names).
    scorer_by_archive: dict[_dt.date, object] = {}
    model_path_by_archive: dict[_dt.date, str] = {}
    tensor_by_sig: dict[tuple, tuple] = {}

    predictions_by_date: dict[str, dict[str, float]] = {}
    n_cold_start = 0
    cold_start_test_starts: list[str] = []
    archive_dates_used: set[_dt.date] = set()
    n_test_dates_scored = 0

    try:
        for fold in folds:
            decision_date = fold.test_start_date
            try:
                resolved = resolve_momentum_weights(s3, bucket, decision_date)
            except ColdStartExclusion as exc:
                n_cold_start += 1
                cold_start_test_starts.append(decision_date.isoformat())
                logger.info(
                    "[walk_forward] fold test_start=%s EXCLUDED (cold-start): %s",
                    decision_date.isoformat(), exc,
                )
                continue

            adate = resolved.archive_date
            archive_dates_used.add(adate)

            scorer = scorer_by_archive.get(adate)
            if scorer is None:
                mpath = _download_gbm_to_temp(
                    s3, bucket, resolved.model_key, resolved.meta_key,
                )
                model_path_by_archive[adate] = mpath
                scorer = GBMScorer.load(mpath)
                if not scorer.feature_names:
                    raise RuntimeError(
                        f"Archived momentum model {resolved.model_key} has no "
                        "feature_names metadata — cannot align input features. "
                        "The archived meta.json predates feature_names "
                        "persistence; exclude that archive date or refresh it."
                    )
                scorer_by_archive[adate] = scorer

            sig = tuple(scorer.feature_names)
            cached = tensor_by_sig.get(sig)
            if cached is None:
                logger.info(
                    "[walk_forward] building inference tensor for %d-feature "
                    "signature (archive %s)", len(sig), adate.isoformat(),
                )
                cached = build_inference_tensor(features_by_ticker, list(sig))
                tensor_by_sig[sig] = cached
            tensor, tickers, date_to_idx = cached

            fold_test_dates = trading_dates[
                fold.test_start_idx : fold.test_end_idx + 1
            ]
            fold_preds = _predict_from_tensor(
                tensor, tickers, date_to_idx, fold_test_dates,
                scorer=scorer, heartbeat_every=50,
                log_label=f"WF[{decision_date.isoformat()}<-{adate.isoformat()}]",
            )
            for d, row in fold_preds.items():
                if d in predictions_by_date:
                    # Non-overlapping test windows is a fold-splitter
                    # invariant; a collision means the splitter regressed.
                    raise RuntimeError(
                        f"[walk_forward] date {d} scored by >1 fold — "
                        "fold splitter produced overlapping test windows"
                    )
                predictions_by_date[d] = row
            n_test_dates_scored += len(fold_preds)
    finally:
        for mpath in model_path_by_archive.values():
            for path in (mpath, mpath + ".meta.json"):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    n_folds_scored = len(folds) - n_cold_start
    wf_stats = {
        "enabled": True,
        "n_folds": len(folds),
        "n_folds_scored": n_folds_scored,
        "n_cold_start_excluded": n_cold_start,
        "cold_start_test_starts": cold_start_test_starts,
        "archive_dates_used": sorted(d.isoformat() for d in archive_dates_used),
        "n_distinct_archives": len(archive_dates_used),
        "n_test_dates_scored": n_test_dates_scored,
        "params": p,
    }
    logger.info(
        "[walk_forward] complete: %d/%d folds scored, %d cold-start-excluded, "
        "%d distinct archive(s), %d test dates with predictions",
        n_folds_scored, len(folds), n_cold_start,
        len(archive_dates_used), n_test_dates_scored,
    )
    if n_cold_start and n_folds_scored == 0:
        # Every fold cold-start-excluded → the run produced zero PIT signals.
        # Loud, not silent (feedback_no_silent_fails): a parity run on this
        # would compare against nothing.
        logger.error(
            "[walk_forward] ALL %d folds cold-start-excluded — no archived "
            "weights on-or-before any fold decision date. Archive coverage "
            "(predictor/weights/meta/archive/) is the constraint, not a code "
            "bug; the PIT run has no signals to simulate.", len(folds),
        )
    return predictions_by_date, wf_stats


def run(
    config: dict,
    keep_features: bool = False,
    persist_features_callback=None,
    keep_predictions: bool = False,
) -> dict:
    """
    Full predictor-only backtest pipeline.

    Steps:
        1. Resolve predictor path and load slim cache
        2. Load sector map
        3. Compute features for all stock tickers
        4. Download GBM model from S3
        5. Run inference across all trading dates
        6. (NEW) Persist features via callback if provided, then drop
        7. Generate synthetic signals (without features in memory)
        8. Build price matrix and OHLCV histories

    Returns a dict with all data needed by backtest.py's simulation loop:
        - signals_by_date: {date: signal_envelope}
        - price_matrix: DataFrame
        - ohlcv_by_ticker: {ticker: pd.DataFrame}
        - metadata: {n_tickers, n_dates, date_range, ...}

    When ``keep_predictions=True``, the result also includes:
        - predictions_by_date: {date: {ticker: predicted_alpha}}
          The raw GBM alpha forecasts before signal envelopes are built.
          Cheap to keep (a few MB for 10y × 900 tickers); consumed by the
          portfolio-optimizer backtest harness (PR 3 of
          alpha-engine-docs/private/portfolio-optimizer-260511.md). Distinct
          from ``keep_features=True``, which keeps the ~1.1 GB features
          dict alive — that path was the cause of the 2026-04-26 c5.large
          OOM. ``keep_predictions`` is the lightweight alternative for
          downstream consumers that need alpha forecasts but not features.

    Parameters
    ----------
    config : pipeline config dict (signals_bucket, predictor_paths,
        predictor_backtest section, etc.)
    keep_features : if True, ``features_by_ticker`` survives into the
        returned dict (kept in memory through and beyond signal
        generation). Legacy behavior; mutually exclusive with
        ``persist_features_callback``.
    persist_features_callback : optional ``Callable[[dict], None]``.
        When provided, called with ``features_by_ticker`` right after
        inference and BEFORE ``build_signals_by_date``. Caller is
        responsible for durable persistence (typically S3 +
        ``ctx.record_artifact``). After the callback returns, run()
        drops the ~1.1 GB features dict and ``gc.collect()``s before
        signal generation.

        This is the Stage 4 c5.large fix: the 2026-04-26 OOM at
        ``post_build_signals=2768 MB`` was dominated by features
        coexisting with the just-built signals dict. Persisting +
        dropping features here saves ~1.1 GB at that checkpoint; the
        feature artifact is lazy-loaded by Phase 4a/4c via
        ``backtest._load_features_by_ticker_only``.
    """
    if keep_features and persist_features_callback is not None:
        raise ValueError(
            "run(): keep_features=True and persist_features_callback are "
            "mutually exclusive. The callback already persists + drops "
            "features; setting keep_features=True would un-drop them."
        )
    # Resolve predictor path
    predictor_paths = config.get("predictor_paths", [])
    if isinstance(predictor_paths, str):
        predictor_paths = [predictor_paths]
    predictor_path = next((p for p in predictor_paths if os.path.isdir(p)), None)
    if not predictor_path:
        raise ValueError(
            f"None of the predictor_paths exist: {predictor_paths}. "
            "Add the alpha-engine-predictor repo root to predictor_paths in config.yaml."
        )

    pb_config = config.get("predictor_backtest", {})
    min_trading_days = pb_config.get("min_trading_days", 252)
    max_trading_days = pb_config.get("max_trading_days", 500)
    top_n = pb_config.get("top_n_signals_per_day", 20)
    min_score = pb_config.get("min_score", 70)
    bucket = config.get("signals_bucket", "alpha-engine-research")

    # 1. Load price data + features from ArcticDB (sole source post-Phase-0).
    #    Hard-fail on unreachable per backtester-audit-260415.md: legacy S3
    #    parquet cache + inline slim-cache fallbacks have been removed.
    from store.arctic_reader import load_universe_from_arctic
    _log_rss("pre_arcticdb_load")
    logger.info("[data_source=arcticdb] Loading universe from ArcticDB...")
    # Smoke fixture universe filter — production default is None (full
    # universe load). When smoke_tickers is set, reader restricts the
    # stock-symbol read; macro/ETF symbols (SPY etc.) always load.
    _smoke_tickers = config.get("smoke_tickers")
    _allowlist = set(_smoke_tickers) if _smoke_tickers else None
    price_data, features_by_ticker = load_universe_from_arctic(
        bucket=bucket, tickers_allowlist=_allowlist,
    )
    data_source = "arcticdb"
    feature_skip_reasons: dict = {}
    logger.info("[data_source=arcticdb] %d tickers with pre-computed features", len(features_by_ticker))
    _log_rss("post_arcticdb_load")

    # 2. Load sector map
    sector_map = load_sector_map(predictor_path)

    # 3. Inline feature recompute is only hit when ArcticDB's feature coverage
    #    is insufficient for the requested backtest window (e.g., 10y synthetic
    #    backtest running before the feature schema was backfilled). In practice
    #    load_universe_from_arctic returns a non-empty dict for every stock
    #    ticker in the universe library; this branch is the safety net.
    if not features_by_ticker:
        logger.warning("ArcticDB returned no pre-computed features — recomputing inline from OHLCV")
        features_by_ticker, feature_skip_reasons = compute_all_features(price_data, sector_map, predictor_path)

    if not features_by_ticker:
        return {
            "status": "error",
            "error": "No tickers had sufficient data for feature computation",
            "tickers_loaded": len(price_data),
            "skip_reasons": feature_skip_reasons,
        }

    # 3b. Resolve trading dates
    trading_dates = _resolve_trading_dates(features_by_ticker, min_trading_days, max_trading_days)
    if isinstance(trading_dates, dict):
        return trading_dates  # early exit with error dict

    # 4. Build price matrix, extract SPY, build DataFrame-form ohlcv_by_ticker.
    #    2026-04-23 SF dry-run OOM'd on c5.large because price_data (~91 MB)
    #    and ohlcv_by_ticker (~1.1 GB, dominated by Python dict overhead in
    #    the list-of-dicts form) coexisted at peak. The pandas refactor
    #    (plan 2026-04-23) replaces list-of-dicts with per-ticker
    #    DataFrames, dropping ohlcv_by_ticker's resident size to ~91 MB
    #    (~12x reduction) and eliminating the concurrent-peak risk
    #    outright. Downstream consumers (simulate, precompute_indicator_series,
    #    artifact save/load) dispatch on shape until step 9 cleanup
    #    removes the legacy branches.
    price_matrix = build_price_matrix(price_data, trading_dates)
    _log_rss("post_price_matrix")
    spy_prices = _extract_close(price_data, "SPY")  # extracts a Series copy
    _log_rss("post_spy_extract")

    ohlcv_by_ticker = build_ohlcv_df_by_ticker(price_data)
    # Release price_data: its entries now live as normalized DataFrames
    # inside ohlcv_by_ticker. Holding both would re-introduce the
    # concurrent-peak the pandas refactor is designed to kill.
    del price_data
    gc.collect()
    _log_rss("post_ohlcv_build_and_drain")
    logger.info("Freed raw price data (memory optimization)")

    # 5 + 6. Inference. Two mutually-exclusive paths:
    #   - walk_forward OFF (default): single pass over all dates against the
    #     CURRENT LIVE momentum GBM. Look-ahead-contaminated but the
    #     historical behavior — preserved byte-for-byte until PR 3's parity
    #     report is reviewed and the default is flipped (plan §5 / S3-contract
    #     caution: a new code path never silently changes optimizer inputs).
    #   - walk_forward ON: per-fold point-in-time weight resolution
    #     (run_walk_forward_inference). The model temp files are managed
    #     inside that function; model_path stays None so the cleanup block
    #     below no-ops on this path.
    n_feature_tickers = len(features_by_ticker)
    wf_enabled = bool(config.get("walk_forward", False))
    wf_params = pb_config.get("walk_forward_params", {})
    if wf_enabled:
        logger.info(
            "[walk_forward] PIT-honest inference ON — archived weights "
            "resolved per fold (knowledge-time ≤ decision-date)"
        )
        model_path = None
        predictions_by_date, wf_stats = run_walk_forward_inference(
            features_by_ticker, trading_dates, predictor_path,
            bucket=bucket, wf_params=wf_params,
        )
    else:
        model_path = download_gbm_model(bucket=bucket)
        _log_rss("post_gbm_download")
        predictions_by_date = run_inference(
            features_by_ticker, model_path, predictor_path, trading_dates,
        )
        wf_stats = None
    _log_rss("post_inference")

    # 6b. Persist features via caller-provided callback BEFORE dropping.
    # Stage 4 fix for the post_build_signals OOM: without this, features
    # (~1.1 GB) coexist with the just-built signals dict (~700 MB-1 GB)
    # during the next phase, peaking RSS at ~2.7 GB on full universe.
    # When the callback is provided, the caller is responsible for durable
    # storage (typically S3) before run() drops the in-memory dict.
    if persist_features_callback is not None:
        persist_features_callback(features_by_ticker)
        logger.info("Features persisted via caller callback")

    # Free features and model. Drop unconditionally when keep_features
    # is False (the default): downstream needs predictions_by_date but
    # not features. The pre-Stage-4 ``keep_features=True`` path that
    # held features through build_signals is preserved for tests +
    # any caller that genuinely needs the in-memory dict, but the
    # production path (backtest.py) now uses the callback instead.
    if not keep_features:
        del features_by_ticker
        gc.collect()
        logger.info("Freed feature data (memory optimization)")
        _log_rss("post_feature_free")

    # Clean up temp model file. None on the walk_forward path —
    # run_walk_forward_inference already unlinked every archived booster
    # it downloaded (its own finally block).
    if model_path is not None:
        try:
            os.unlink(model_path)
            meta_path = model_path + ".meta.json"
            if os.path.exists(meta_path):
                os.unlink(meta_path)
        except OSError:
            pass

    # 7. Generate signals (using technical scoring from OHLCV, enriched by GBM alpha)
    signals_by_date = build_signals_by_date(
        predictions_by_date, sector_map, ohlcv_by_ticker,
        top_n=top_n, min_score=min_score,
    )
    _log_rss("post_build_signals")

    # Metadata for reporting
    n_enter_total = sum(
        len(env.get("buy_candidates", []))
        for env in signals_by_date.values()
    )

    metadata = {
        "data_source": data_source,
        "n_tickers": n_feature_tickers,
        "n_dates": len(trading_dates),
        "date_range_start": trading_dates[0],
        "date_range_end": trading_dates[-1],
        "n_enter_signals_total": n_enter_total,
        "top_n_per_day": top_n,
        "min_score": min_score,
        # None on the legacy single-pass path; the PIT run-quality block
        # (fold count, cold-start exclusions, archives used) when walk_forward
        # is on. Consumed by PR 3's --pit-parity contamination report.
        "walk_forward": wf_stats,
    }
    logger.info("Predictor backtest data ready: %s", metadata)

    result = {
        "status": "ok",
        "signals_by_date": signals_by_date,
        "price_matrix": price_matrix,
        "ohlcv_by_ticker": ohlcv_by_ticker,
        "spy_prices": spy_prices,
        "metadata": metadata,
    }

    if keep_features:
        result["features_by_ticker"] = features_by_ticker
        result["sector_map"] = sector_map
        result["trading_dates"] = trading_dates
        result["predictions_by_date"] = predictions_by_date
    elif keep_predictions:
        result["predictions_by_date"] = predictions_by_date
        result["sector_map"] = sector_map
        result["trading_dates"] = trading_dates

    # W3.4 (L4469): per-ticker ADV-dollar for the horizon net-alpha cost model.
    # Cheap (~one float per ticker); surfaced alongside predictions so the
    # consumer doesn't re-load price data. price_data carries OHLCV Volume.
    if keep_predictions or keep_features:
        result["adv_dollar_by_ticker"] = _compute_adv_dollar(price_data)

    return result
