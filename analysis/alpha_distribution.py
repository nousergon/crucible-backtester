"""
alpha_distribution.py — Alpha magnitude distribution analysis.

Beyond binary beat_spy, buckets realized alpha into ranges to reveal
whether the system generates small consistent alpha or volatile guesses.

Buckets: <-5%, -5 to -2%, -2 to 0%, 0 to 2%, 2 to 5%, >5%

Data source: score_performance table in research.db.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_ALPHA_BUCKETS = [
    ("<-5%", -float("inf"), -5.0),
    ("-5% to -2%", -5.0, -2.0),
    ("-2% to 0%", -2.0, 0.0),
    ("0% to +2%", 0.0, 2.0),
    ("+2% to +5%", 2.0, 5.0),
    (">+5%", 5.0, float("inf")),
]


def compute_alpha_distribution(
    db_path: str,
    horizons: tuple[str, ...] = ("5d", "21d"),
    min_samples: int = 5,
) -> dict:
    """
    Compute alpha magnitude distribution from score_performance.

    Alpha is defined as stock return minus SPY return over the same window,
    expressed in percentage points (as stored in the DB).

    Returns dict with:
        status: "ok" | "insufficient_data" | "error"
        distributions: {horizon: [{bucket, count, pct, avg_alpha}, ...]}
        summary: {horizon: {n, avg_alpha, median_alpha, std_alpha, skew}}
    """
    if not Path(db_path).exists():
        return {"status": "error", "error": f"DB not found at {db_path}"}

    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM score_performance", conn)
        conn.close()
    except Exception as e:
        return {"status": "error", "error": str(e)}

    if df.empty:
        return {"status": "insufficient_data", "error": "score_performance is empty"}

    distributions: dict = {}
    summary: dict = {}

    horizon_cols = {
        "5d": ("return_5d", "spy_5d_return"),
        "21d": ("return_21d", "spy_21d_return"),
    }

    for h in horizons:
        if h not in horizon_cols:
            continue
        ret_col, spy_col = horizon_cols[h]
        if ret_col not in df.columns or spy_col not in df.columns:
            continue

        sub = df[[ret_col, spy_col]].dropna()
        if len(sub) < min_samples:
            continue

        # Alpha in percentage points (both columns are stored as pct)
        sub = sub.copy()
        sub["alpha"] = sub[ret_col] - sub[spy_col]

        buckets = []
        for label, low, high in _ALPHA_BUCKETS:
            mask = (sub["alpha"] >= low) & (sub["alpha"] < high)
            count = int(mask.sum())
            pct = round(count / len(sub), 4) if len(sub) > 0 else 0
            avg_a = round(float(sub.loc[mask, "alpha"].mean()), 2) if count > 0 else None
            buckets.append({
                "bucket": label,
                "count": count,
                "pct": pct,
                "avg_alpha": avg_a,
            })
        distributions[h] = buckets

        summary[h] = {
            "n": len(sub),
            "avg_alpha": round(float(sub["alpha"].mean()), 2),
            "median_alpha": round(float(sub["alpha"].median()), 2),
            "std_alpha": round(float(sub["alpha"].std()), 2),
            "skew": round(float(sub["alpha"].skew()), 2) if len(sub) >= 3 else None,
            "pct_positive": round(float((sub["alpha"] > 0).mean()), 4),
        }

    if not distributions:
        return {"status": "insufficient_data", "error": "no horizons with enough data"}

    return {
        "status": "ok",
        "distributions": distributions,
        "summary": summary,
    }


def compute_score_calibration(
    db_path: str,
    horizon: str = "21d",
    n_buckets: int = 5,
    min_per_bucket: int = 3,
) -> dict:
    """
    Score calibration curve: does higher score correlate with higher alpha?

    Groups scores into quantile buckets and computes average alpha per bucket.
    A well-calibrated system shows monotonically increasing alpha with score.

    Returns dict with:
        status: "ok" | "insufficient_data"
        calibration: [{score_range, n, avg_score, avg_alpha, beat_spy_pct}, ...]
        monotonic: bool (is the relationship monotonically increasing?)
    """
    if not Path(db_path).exists():
        return {"status": "error", "error": f"DB not found at {db_path}"}

    horizon_cols = {
        "5d": ("return_5d", "spy_5d_return", "beat_spy_5d"),
        "21d": ("return_21d", "spy_21d_return", "beat_spy_21d"),
    }
    if horizon not in horizon_cols:
        return {"status": "error", "error": f"unsupported horizon: {horizon}"}

    ret_col, spy_col, beat_col = horizon_cols[horizon]

    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM score_performance", conn)

        # Enrich with sector (universe_returns) + regime (macro_snapshots)
        # so non-monotonic calibration buckets can be diagnosed by
        # sector/regime concentration rather than sample noise alone.
        try:
            sectors = pd.read_sql_query(
                "SELECT DISTINCT ticker, sector FROM universe_returns "
                "WHERE sector IS NOT NULL AND sector != ''",
                conn,
            )
            if not sectors.empty:
                df = df.merge(
                    sectors, left_on="symbol", right_on="ticker", how="left",
                )
        except Exception as _e:
            logger.debug("universe_returns sector join skipped: %s", _e)

        try:
            regimes = pd.read_sql_query(
                "SELECT date, market_regime FROM macro_snapshots "
                "WHERE market_regime IS NOT NULL",
                conn,
            )
            if not regimes.empty:
                df = df.merge(
                    regimes, left_on="score_date", right_on="date", how="left",
                )
        except Exception as _e:
            logger.debug("macro_snapshots regime join skipped: %s", _e)

        conn.close()
    except Exception as e:
        return {"status": "error", "error": str(e)}

    required = ["score", ret_col, spy_col]
    for c in required:
        if c not in df.columns:
            return {"status": "insufficient_data", "error": f"column {c} not found"}

    keep_cols = ["score", ret_col, spy_col, "symbol", "score_date"]
    if "sector" in df.columns:
        keep_cols.append("sector")
    if "market_regime" in df.columns:
        keep_cols.append("market_regime")
    sub = df[keep_cols].dropna(subset=["score", ret_col, spy_col])
    if len(sub) < n_buckets * min_per_bucket:
        return {"status": "insufficient_data", "error": f"need {n_buckets * min_per_bucket} rows, have {len(sub)}"}

    sub = sub.copy()
    sub["alpha"] = sub[ret_col] - sub[spy_col]

    try:
        sub["bucket"] = pd.qcut(sub["score"], n_buckets, duplicates="drop")
    except ValueError:
        return {"status": "insufficient_data", "error": "not enough score variance for buckets"}

    calibration = []
    for bucket in sorted(sub["bucket"].unique()):
        group = sub[sub["bucket"] == bucket]
        if len(group) < min_per_bucket:
            continue

        # Per-bucket diagnostics: surface sector / regime / date concentration
        # so a non-monotonic pattern can be distinguished from small-sample
        # noise (e.g., one bad Healthcare week dominating the 59–65 bucket).
        top_sectors = []
        if "sector" in group.columns and group["sector"].notna().any():
            top_sectors = [
                {"sector": str(k), "n": int(v)}
                for k, v in group["sector"].value_counts().head(3).items()
            ]
        regime_counts = []
        if "market_regime" in group.columns and group["market_regime"].notna().any():
            regime_counts = [
                {"regime": str(k), "n": int(v)}
                for k, v in group["market_regime"].value_counts().items()
            ]
        dates = group["score_date"].dropna().unique() if "score_date" in group.columns else []
        tickers = group["symbol"].dropna().unique() if "symbol" in group.columns else []

        calibration.append({
            "score_range": str(bucket),
            "n": len(group),
            "avg_score": round(float(group["score"].mean()), 1),
            "avg_alpha": round(float(group["alpha"].mean()), 2),
            "beat_spy_pct": round(float((group["alpha"] > 0).mean()), 4),
            "top_sectors": top_sectors,
            "regime_breakdown": regime_counts,
            "n_unique_dates": len(dates),
            "n_unique_tickers": len(tickers),
            "date_range": (
                [str(min(dates)), str(max(dates))] if len(dates) else []
            ),
        })

    if len(calibration) < 2:
        return {"status": "insufficient_data", "error": "not enough buckets with data"}

    alphas = [c["avg_alpha"] for c in calibration]
    # Legacy bucket-based binary: strict non-decreasing avg_alpha across quantile
    # buckets. RETAINED for backward-compat + as a coarse diagnostic, but NO
    # LONGER the graded metric — it flips False on a single noisy bucket and
    # discards the per-bucket concentration diagnostics above. The graded signal
    # is now the robust row-level Spearman rank correlation below.
    monotonic = all(alphas[i] <= alphas[i + 1] for i in range(len(alphas) - 1))

    # Robust calibration: Spearman rank correlation of raw score vs realized
    # alpha across ALL rows (not bucket means). Measures monotonic association
    # continuously, is outlier-robust, and yields a significance test so a flat
    # calibration at low N reads as "no measurable signal" rather than RED.
    # Mirrors the rank-correlation + significance pattern already used in
    # analysis/attribution.py (scipy.stats). See ROADMAP L4550 (metric-quality
    # fix; the composite formula itself is provably monotonic in its inputs).
    spearman_rho: float | None = None
    spearman_p: float | None = None
    spearman_n = int(len(sub))
    if sub["score"].nunique() >= 2 and sub["alpha"].nunique() >= 2:
        from scipy.stats import spearmanr

        rho, pval = spearmanr(sub["score"], sub["alpha"])
        # spearmanr returns nan when a column is constant despite the nunique
        # guard (e.g. all-tied ranks); coerce to None so consumers see "unknown".
        if rho == rho:  # not NaN
            spearman_rho = round(float(rho), 4)
        if pval == pval:  # not NaN
            spearman_p = round(float(pval), 4)

    # Plain-language assessment for observability + grading fallback.
    if spearman_rho is None:
        calibration_assessment = "insufficient_data"
    elif spearman_p is not None and spearman_p >= 0.10:
        calibration_assessment = "flat"  # not statistically distinguishable from zero
    elif spearman_rho > 0:
        calibration_assessment = "positive"
    else:
        calibration_assessment = "negative"

    return {
        "status": "ok",
        "horizon": horizon,
        "calibration": calibration,
        "monotonic": monotonic,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "spearman_n": spearman_n,
        "calibration_assessment": calibration_assessment,
    }
