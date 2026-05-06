"""
excursion.py — per-trade Maximum Favorable / Adverse Excursion (MFE / MAE).

Process-quality per trade. For each pick, track how far the price went in
your favor (MFE) and against you (MAE) during the holding window. Skilled
risk-takers show MFE/MAE > 1.5 (cuts losers, rides winners). Agents
YOLOing into volatility show MFE ≈ MAE — wild swings both directions, no
selection edge.

Daily-bar fidelity: MFE = max(daily_high) over holding window minus entry
price; MAE = entry price minus min(daily_low). For the evaluator-revamp
scope this is sufficient — intraday tick fidelity is deferred to a
separate ROADMAP P3 item per the plan doc.

Pure-compute. Operates on entry records + a daily OHLC price source;
no I/O.
"""

from __future__ import annotations

import logging
from typing import TypedDict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExcursionRecord(TypedDict, total=False):
    ticker: str
    eval_date: str
    entry_price: float
    horizon_days: int
    mfe: float           # max favorable excursion (positive number, fraction)
    mae: float           # max adverse excursion (positive magnitude, fraction)
    mfe_mae_ratio: float | None  # None if mae == 0
    realized_return: float       # total return at horizon close (signed)


class ExcursionSummary(TypedDict, total=False):
    status: str
    n: int
    mean_mfe: float
    mean_mae: float
    mean_mfe_mae_ratio: float
    median_mfe_mae_ratio: float
    pct_mfe_gt_mae: float        # fraction of trades where MFE > MAE
    pct_high_quality: float      # fraction with mfe/mae > 1.5


def compute_per_pick_excursion(
    picks: pd.DataFrame,
    ohlc: dict[str, pd.DataFrame],
    horizon_days: int = 10,
) -> list[ExcursionRecord]:
    """Compute MFE/MAE per pick from daily OHLC data.

    Parameters
    ----------
    picks : pd.DataFrame
        Required columns: ``ticker``, ``eval_date``. Optional column:
        ``entry_price`` (override entry price; defaults to close on
        eval_date if absent).
    ohlc : dict[str, pd.DataFrame]
        ``{ticker: ohlc_df}`` where each DataFrame has a DatetimeIndex
        and lowercase columns ``high`` and ``low`` (and ``close`` for
        realized return + entry-price default). Matches the producer
        contract from ``loaders/price_loader.build_matrix(_ohlcv_out=...)``.
    horizon_days : int
        Holding window in trading days. Default 10. The window for a
        pick on eval_date D is ``[D, D + horizon_days]`` inclusive of
        endpoints — MFE/MAE scan covers all daily bars in that range.

    Returns
    -------
    list[ExcursionRecord]
        One record per pick. Picks whose ticker is missing from ``ohlc``
        or whose eval_date is outside the price index are skipped (logged).

    Notes
    -----
    - MFE = max(high) over window / entry_price - 1 (positive = favorable)
    - MAE = 1 - min(low) over window / entry_price (positive = adverse)
    - mfe_mae_ratio = MFE / MAE (None if MAE = 0, i.e. price never went
      below entry — degenerate case for risk analysis)
    """
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
    required = {"ticker", "eval_date"}
    missing = required - set(picks.columns)
    if missing:
        raise ValueError(f"picks missing required columns: {sorted(missing)}")

    out: list[ExcursionRecord] = []
    has_explicit_entry = "entry_price" in picks.columns

    for _, row in picks.iterrows():
        ticker = row["ticker"]
        eval_date = pd.Timestamp(row["eval_date"])
        df = ohlc.get(ticker)
        if df is None or df.empty:
            logger.debug("excursion: ticker %s missing from ohlc; skipping", ticker)
            continue
        if eval_date not in df.index:
            logger.debug("excursion: eval_date %s not in ohlc index for %s; skipping",
                         eval_date.date(), ticker)
            continue

        start_pos = df.index.get_loc(eval_date)
        end_pos = min(start_pos + horizon_days, len(df.index) - 1)
        if end_pos <= start_pos:
            continue

        window = df.iloc[start_pos : end_pos + 1]  # inclusive of endpoints
        if has_explicit_entry and pd.notna(row["entry_price"]):
            entry_price = float(row["entry_price"])
        elif "close" in window.columns:
            entry_price = float(window["close"].iloc[0])
        else:
            logger.debug("excursion: no entry price + no close col for %s; skipping", ticker)
            continue
        if entry_price <= 0:
            continue

        if "high" not in window.columns or "low" not in window.columns:
            logger.debug("excursion: ohlc for %s missing high/low; skipping", ticker)
            continue

        max_high = float(window["high"].max())
        min_low = float(window["low"].min())
        mfe = max_high / entry_price - 1.0
        mae = 1.0 - min_low / entry_price  # positive magnitude
        # Clamp negative MFE / MAE to 0 — high < entry or low > entry is
        # degenerate (the bar containing entry should at least equal entry
        # price). Treat as no-excursion-this-direction.
        mfe = max(mfe, 0.0)
        mae = max(mae, 0.0)
        ratio = mfe / mae if mae > 0 else None

        if "close" in window.columns:
            realized = float(window["close"].iloc[-1] / entry_price - 1.0)
        else:
            realized = float("nan")

        out.append({
            "ticker": ticker,
            "eval_date": str(row["eval_date"]),
            "entry_price": entry_price,
            "horizon_days": horizon_days,
            "mfe": mfe,
            "mae": mae,
            "mfe_mae_ratio": ratio,
            "realized_return": realized,
        })

    return out


def summarize_excursions(records: list[ExcursionRecord]) -> ExcursionSummary:
    """Aggregate MFE/MAE statistics across a set of picks.

    Returns a summary dict suitable for grading:
      - mean_mfe, mean_mae: average excursion magnitudes
      - mean_mfe_mae_ratio: simple mean of finite ratios (excludes
        records where mae == 0)
      - median_mfe_mae_ratio: more robust to outliers in tail trades
      - pct_mfe_gt_mae: skill marker — fraction where favorable excursion
        exceeded adverse
      - pct_high_quality: skilled-risk-taking marker — fraction with
        mfe/mae > 1.5
    """
    if not records:
        return {"status": "insufficient_data", "n": 0}
    n = len(records)
    mfe_arr = np.array([r["mfe"] for r in records], dtype=np.float64)
    mae_arr = np.array([r["mae"] for r in records], dtype=np.float64)
    ratios = np.array([
        r["mfe_mae_ratio"] for r in records if r.get("mfe_mae_ratio") is not None
    ], dtype=np.float64)
    finite_ratios = ratios[np.isfinite(ratios)]

    return {
        "status": "ok",
        "n": n,
        "mean_mfe": float(mfe_arr.mean()),
        "mean_mae": float(mae_arr.mean()),
        "mean_mfe_mae_ratio": float(finite_ratios.mean()) if finite_ratios.size else 0.0,
        "median_mfe_mae_ratio": float(np.median(finite_ratios)) if finite_ratios.size else 0.0,
        "pct_mfe_gt_mae": float((mfe_arr > mae_arr).mean()),
        "pct_high_quality": float((finite_ratios > 1.5).mean()) if finite_ratios.size else 0.0,
    }
