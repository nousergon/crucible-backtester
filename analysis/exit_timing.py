"""
exit_timing.py — Exit timing analysis via MFE/MAE.

For each completed trade (with entry and exit), computes:
  - Max Favorable Excursion (MFE): best unrealized return during hold
  - Max Adverse Excursion (MAE): worst unrealized return during hold
  - Capture ratio: realized return / MFE (are we capturing gains?)
  - Stop efficiency: |realized loss| / MAE (are stops placed well?)

Requires daily OHLCV price data during the hold period. Reads from the
ArcticDB universe library (primary, via alpha_engine_lib), falling back to
the predictor/price_cache (10y) parquets in S3 (no external API calls).
Wave-4: the predictor/price_cache_slim leg was removed after the parity
observation confirmed slim<->ArcticDB equivalence.

Data source: trades table in trades.db (roundtrip trades with entry_trade_id).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from alpha_engine_lib.arcticdb import load_universe_ohlcv

logger = logging.getLogger(__name__)


def compute_exit_timing(
    trades_db_path: str,
    min_roundtrips: int = 5,
) -> dict:
    """
    Compute MFE/MAE analysis for completed roundtrip trades.

    A roundtrip is an EXIT trade linked to its ENTER via entry_trade_id.

    Returns dict with:
        status: "ok" | "insufficient_data" | "error"
        n_roundtrips: number of completed roundtrips analyzed
        summary: {avg_mfe, avg_mae, avg_capture_ratio, avg_realized_return}
        by_exit_type: [{exit_type, n, avg_mfe, avg_mae, avg_capture, avg_return}, ...]
        diagnosis: "exits_too_early" | "exits_well_timed" | "exits_too_late"
    """
    if not Path(trades_db_path).exists():
        return {"status": "error", "error": f"trades.db not found at {trades_db_path}"}

    try:
        conn = sqlite3.connect(trades_db_path)

        exits = pd.read_sql_query(
            "SELECT e.ticker, e.date AS exit_date, e.fill_price AS exit_price, "
            "e.trigger_type AS exit_type, e.realized_return_pct, "
            "e.realized_alpha_pct, e.days_held, "
            "en.date AS entry_date, en.fill_price AS entry_price, "
            "en.signal_price "
            "FROM trades e "
            "JOIN trades en ON e.entry_trade_id = en.trade_id "
            "WHERE e.action IN ('EXIT', 'REDUCE') "
            "AND en.action = 'ENTER' "
            "AND e.fill_price IS NOT NULL "
            "AND en.fill_price IS NOT NULL",
            conn,
        )
        conn.close()
    except Exception as e:
        return {"status": "error", "error": str(e)}

    if exits.empty or len(exits) < min_roundtrips:
        return {
            "status": "insufficient_data",
            "error": f"need >= {min_roundtrips} roundtrips, have {len(exits)}",
        }

    # Load price history from S3 price cache (no external API calls)
    tickers = exits["ticker"].unique().tolist()
    price_cache = _load_price_cache(tickers)
    if not price_cache:
        return {"status": "error", "error": "no price cache data available from S3"}

    results = []
    for _, trade in exits.iterrows():
        entry_ts = pd.Timestamp(trade["entry_date"])
        exit_ts = pd.Timestamp(trade["exit_date"])
        ticker_df = price_cache.get(trade["ticker"])
        if ticker_df is None:
            continue

        try:
            mask = (ticker_df.index >= entry_ts) & (ticker_df.index <= exit_ts)
            period = ticker_df.loc[mask]
            highs = period["High"] if "High" in period.columns else None
            lows = period["Low"] if "Low" in period.columns else None
        except (KeyError, TypeError):
            continue

        if highs is None or lows is None or highs.empty or lows.empty:
            continue

        entry_px = trade["entry_price"]
        if entry_px is None or entry_px <= 0:
            continue

        max_high = float(highs.max())
        min_low = float(lows.min())

        mfe_pct = ((max_high - entry_px) / entry_px) * 100
        mae_pct = ((min_low - entry_px) / entry_px) * 100

        realized = trade.get("realized_return_pct")
        if realized is None:
            if trade["exit_price"] and entry_px:
                realized = ((trade["exit_price"] - entry_px) / entry_px) * 100
            else:
                continue

        capture_ratio = (realized / mfe_pct) if mfe_pct > 0.01 else None

        results.append({
            "ticker": trade["ticker"],
            "entry_date": trade["entry_date"],
            "exit_date": trade["exit_date"],
            "exit_type": trade.get("exit_type", "unknown"),
            "entry_price": entry_px,
            "exit_price": trade["exit_price"],
            "mfe_pct": round(mfe_pct, 2),
            "mae_pct": round(mae_pct, 2),
            "realized_return_pct": round(realized, 2),
            "capture_ratio": round(capture_ratio, 2) if capture_ratio is not None else None,
            "days_held": trade.get("days_held"),
        })

    if len(results) < min_roundtrips:
        return {
            "status": "insufficient_data",
            "error": f"only {len(results)} roundtrips with price data (need {min_roundtrips})",
        }

    rdf = pd.DataFrame(results)

    # Capture ratio (realized / MFE) is a WINNER concept — "of the favorable
    # move that was available, how much did we keep?". Averaging it over ALL
    # trades is a defective metric (ROADMAP L4554): a stopped-out loser has a
    # tiny/near-zero MFE denominator and a negative numerator, so its per-trade
    # ratio explodes negative and a handful of stops drag the unbounded MEAN
    # below zero even when most exits capture their winners well. The graded
    # signal is therefore the MEDIAN capture over WINNERS (realized>0 & MFE>0);
    # the legacy all-trade mean is retained as a (misleading) diagnostic only,
    # and loser exit-quality is summarized separately via stop_efficiency
    # (|realized| / |MAE| — how much of the adverse move we absorbed before
    # cutting; lower = cut sooner).
    winners = rdf[(rdf["realized_return_pct"] > 0) & (rdf["mfe_pct"] > 0.01)
                  & rdf["capture_ratio"].notna()]
    losers = rdf[rdf["realized_return_pct"] < 0]
    capture_winners_median = (round(float(winners["capture_ratio"].median()), 2)
                              if not winners.empty else None)
    # Winsorize the all-trade capture to [-1, 1.5] before meaning, so the
    # outlier-driven legacy number has a bounded companion.
    _wins_cap = rdf["capture_ratio"].dropna().clip(lower=-1.0, upper=1.5)
    winsorized_capture = (round(float(_wins_cap.mean()), 2) if not _wins_cap.empty else None)

    # Stop efficiency on losers: |realized| / |MAE| (both negative). Near 1 means
    # we rode the loss most of the way to its worst point; small means we cut
    # early. Only meaningful when MAE < 0.
    loser_stop = losers[losers["mae_pct"] < -0.01]
    if not loser_stop.empty:
        stop_eff = (loser_stop["realized_return_pct"].abs()
                    / loser_stop["mae_pct"].abs())
        stop_efficiency_median = round(float(stop_eff.median()), 2)
    else:
        stop_efficiency_median = None

    summary = {
        "n_roundtrips": len(rdf),
        "avg_mfe": round(float(rdf["mfe_pct"].mean()), 2),
        "avg_mae": round(float(rdf["mae_pct"].mean()), 2),
        "avg_realized_return": round(float(rdf["realized_return_pct"].mean()), 2),
        "avg_capture_ratio": round(float(rdf["capture_ratio"].dropna().mean()), 2)
        if rdf["capture_ratio"].notna().any() else None,
        # Robust replacements (L4554) — graded in preference to avg_capture_ratio:
        "capture_winners_median": capture_winners_median,
        "n_winners": int(len(winners)),
        "winsorized_capture_ratio": winsorized_capture,
        "stop_efficiency_median": stop_efficiency_median,
        "n_losers": int(len(losers)),
        "win_rate": round(float((rdf["realized_return_pct"] > 0).mean()), 3),
        "median_mfe": round(float(rdf["mfe_pct"].median()), 2),
        "median_mae": round(float(rdf["mae_pct"].median()), 2),
    }

    # Diagnosis — robust, off WINNER capture (exit-timing quality on the trades
    # where there was a move to capture), not the outlier-polluted all-trade mean.
    # Cutpoints align with the report-card grading bands (red_line 0.40 / target
    # 0.70) so the diagnosis label and the graded status never disagree.
    cap_w = capture_winners_median
    if cap_w is None:
        diagnosis = "insufficient_winners"
    elif cap_w < 0.40:
        diagnosis = "exits_too_early"  # winners systematically under-captured
    elif cap_w >= 0.70:
        diagnosis = "exits_well_timed"
    else:
        diagnosis = "exits_could_improve"

    # By exit type
    by_exit_type = []
    for et in sorted(rdf["exit_type"].dropna().unique()):
        grp = rdf[rdf["exit_type"] == et]
        if len(grp) < 2:
            continue
        by_exit_type.append({
            "exit_type": et,
            "n": len(grp),
            "avg_mfe": round(float(grp["mfe_pct"].mean()), 2),
            "avg_mae": round(float(grp["mae_pct"].mean()), 2),
            "avg_realized": round(float(grp["realized_return_pct"].mean()), 2),
            "avg_capture": round(float(grp["capture_ratio"].dropna().mean()), 2)
            if grp["capture_ratio"].notna().any() else None,
        })

    return {
        "status": "ok",
        "n_roundtrips": len(rdf),
        "summary": summary,
        "by_exit_type": by_exit_type,
        "diagnosis": diagnosis,
    }


def _load_price_cache(tickers: list[str], bucket: str = "alpha-engine-research") -> dict[str, pd.DataFrame]:
    """Load OHLCV parquets from S3 price cache for the given tickers.

    Returns {ticker: DataFrame} with DatetimeIndex and OHLCV columns.
    Silently skips tickers that don't have cache files.
    """
    import io
    import boto3

    # Wave-4 terminal state (predictor/price_cache_slim deleted): the
    # ArcticDB universe lib is the source for traded tickers (all equities
    # + SPY, which are universe members — exit_timing never needs macro/
    # index symbols, so no macro-lib read). The 10y full per-ticker parquet
    # is the sole fallback for any ticker ArcticDB does not return. The slim
    # leg + parity dual-read were removed after the 5/23 observation
    # confirmed slim<->ArcticDB equivalence.
    #
    # Wave-3 PR4 cutover (#780, W3 slice): the 10y price_cache tree migrated
    # from predictor/price_cache/ to reference/price_cache/; the producer
    # (nousergon-data) now writes reference/ only and the legacy tree is
    # removed live via `aws s3 rm`. This fallback reads the reference home.
    tickers = list(tickers)
    s3 = boto3.client("s3")

    arctic: dict[str, pd.DataFrame] = {}
    try:
        arctic = load_universe_ohlcv(bucket, symbols=tickers)
    except Exception as exc:  # noqa: BLE001 - fall back to price_cache
        logger.warning(
            "ArcticDB universe read for exit_timing failed: %s", exc
        )

    def _read_parquet(prefix: str, ticker: str):
        key = f"{prefix}/{ticker}.parquet"
        try:
            resp = s3.get_object(Bucket=bucket, Key=key)
            df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
            if df.empty:
                return None
            if not isinstance(df.index, pd.DatetimeIndex):
                if "Date" in df.columns:
                    df = df.set_index("Date")
                df.index = pd.to_datetime(df.index)
            return df
        except Exception:
            return None

    cache = dict(arctic)
    # Fallback: reference/price_cache (10y) for any ticker ArcticDB missed.
    for ticker in tickers:
        if ticker in cache:
            continue
        df = _read_parquet("reference/price_cache", ticker)
        if df is not None:
            cache[ticker] = df
    return cache
