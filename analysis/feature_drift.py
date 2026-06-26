"""
analysis/feature_drift.py — Feature importance drift detection (Phase 3).

Detects when the features the predictor relies on diverge from what
actually predicts alpha in recent production data.

Reads from:
  - predictor_outcomes table (research.db) — resolved predictions + actual returns
  - ArcticDB universe library — per-ticker feature values
  - predictor/metrics/training_summary_latest.json (S3) — training-time feature ICs

Writes to:
  - predictor/metrics/feature_drift.json (S3)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta

import boto3
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pipeline_common import (
    ALPHA_COALESCE_SQL,
    CURRENT_HORIZON_FILTER_SQL,
    HORIZON_COALESCE_SQL,
    OUTCOMES_RESOLVED_SQL,
)

log = logging.getLogger(__name__)

_MIN_SAMPLES = 20  # minimum resolved outcomes to compute drift
_NOISE_IC_THRESHOLD = 0.005  # below this = decayed to noise
_SIGN_FLIP_THRESHOLD = -0.005  # production IC below this when training IC was positive = sign flip
_DRIFT_FRACTION_TRIGGER = 0.20  # recommend retrain if >20% of features drifted


def compute_feature_drift(
    db_path: str,
    bucket: str,
    run_date: str | None = None,
    lookback_days: int = 60,
) -> dict:
    """
    Compute per-feature IC on recent production data and compare to training ICs.

    Returns summary dict with drifted/stable features and recommendation.
    """
    run_date = run_date or datetime.utcnow().strftime("%Y-%m-%d")
    cutoff = (datetime.strptime(run_date, "%Y-%m-%d") - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    # ── Load resolved predictor outcomes ─────────────────────────────────────
    if not os.path.exists(db_path):
        log.warning("research.db not found at %s — skipping feature drift", db_path)
        return {"status": "skipped", "reason": "no_db"}

    conn = sqlite3.connect(db_path)
    outcomes = pd.read_sql_query(
        "SELECT symbol, prediction_date, "
        f"{ALPHA_COALESCE_SQL} AS canonical_actual, "
        f"{HORIZON_COALESCE_SQL} AS horizon_days "
        "FROM predictor_outcomes "
        f"WHERE {OUTCOMES_RESOLVED_SQL} "
        f"  AND {CURRENT_HORIZON_FILTER_SQL} "
        f"  AND prediction_date >= ?",
        conn,
        params=(cutoff,),
    )
    conn.close()

    if len(outcomes) < _MIN_SAMPLES:
        log.info("Feature drift: %d resolved outcomes (< %d minimum) — skipping", len(outcomes), _MIN_SAMPLES)
        return {"status": "skipped", "reason": "insufficient_samples", "n": len(outcomes)}

    outcomes["actual"] = pd.to_numeric(outcomes["canonical_actual"], errors="coerce")
    outcomes["prediction_date"] = pd.to_datetime(outcomes["prediction_date"])

    # ── Load feature values from ArcticDB ────────────────────────────────────
    features_by_ticker = _load_features_from_arctic(bucket, list(outcomes["symbol"].unique()))
    if not features_by_ticker:
        log.warning("Feature drift: no feature data from ArcticDB — skipping")
        return {"status": "skipped", "reason": "no_feature_data"}

    # ── Join outcomes with feature values ────────────────────────────────────
    joined = _join_outcomes_with_features(outcomes, features_by_ticker)
    if joined.empty or len(joined) < _MIN_SAMPLES:
        log.info("Feature drift: %d joined rows (< %d minimum) — skipping", len(joined), _MIN_SAMPLES)
        return {"status": "skipped", "reason": "insufficient_joined_data", "n": len(joined)}

    # ── Identify feature columns ────────────────────────────────────────────
    non_feature_cols = {
        "symbol", "prediction_date", "canonical_actual", "actual", "horizon_days",
    }
    feature_cols = [c for c in joined.columns if c not in non_feature_cols]

    if not feature_cols:
        return {"status": "skipped", "reason": "no_feature_columns"}

    # ── Compute production IC per feature (rank correlation with actual return)
    production_ics = {}
    for feat in feature_cols:
        valid = joined[["actual", feat]].dropna()
        if len(valid) < _MIN_SAMPLES:
            continue
        ic, _ = spearmanr(valid[feat], valid["actual"])
        production_ics[feat] = round(float(ic), 6)

    # ── Load training-time feature ICs ───────────────────────────────────────
    training_ics = _load_training_feature_ics(bucket)

    # ── Compare production vs training ICs ───────────────────────────────────
    drifted_features = []
    stable_count = 0
    evaluated_count = 0

    for feat, prod_ic in production_ics.items():
        train_ic = training_ics.get(feat)
        if train_ic is None:
            continue  # can't compare without training baseline

        evaluated_count += 1
        status = _classify_drift(train_ic, prod_ic)

        if status in ("sign_flip", "decayed"):
            drifted_features.append({
                "feature": feat,
                "training_ic": round(float(train_ic), 6),
                "production_ic": prod_ic,
                "status": status,
            })
        else:
            stable_count += 1

    # ── Recommendation ───────────────────────────────────────────────────────
    drift_fraction = len(drifted_features) / evaluated_count if evaluated_count > 0 else 0
    if drift_fraction > _DRIFT_FRACTION_TRIGGER:
        recommendation = "retrain_suggested"
    elif drifted_features:
        recommendation = "monitor"
    else:
        recommendation = "stable"

    # Sort drifted features by severity (largest IC drop first)
    drifted_features.sort(key=lambda x: x["training_ic"] - x["production_ic"], reverse=True)

    result = {
        "date": run_date,
        "lookback_days": lookback_days,
        "n_outcomes": len(joined),
        "drifted_features": drifted_features,
        "stable_features": stable_count,
        "total_features": evaluated_count,
        "drift_fraction": round(drift_fraction, 3),
        "recommendation": recommendation,
    }

    # ── Write to S3 ──────────────────────────────────────────────────────────
    try:
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=bucket,
            Key="predictor/metrics/feature_drift.json",
            Body=json.dumps(result, indent=2, default=str).encode(),
            ContentType="application/json",
        )
        log.info(
            "Feature drift: %d/%d drifted (%.0f%%)  recommendation=%s  n=%d",
            len(drifted_features), evaluated_count, drift_fraction * 100,
            recommendation, len(joined),
        )
    except Exception as exc:
        log.warning("Failed to write feature_drift.json to S3: %s", exc)

    return result


def _classify_drift(training_ic: float, production_ic: float) -> str:
    """Classify a feature's drift status based on training vs production IC."""
    if training_ic > _NOISE_IC_THRESHOLD and production_ic < _SIGN_FLIP_THRESHOLD:
        return "sign_flip"
    if training_ic < -_NOISE_IC_THRESHOLD and production_ic > -_SIGN_FLIP_THRESHOLD:
        return "sign_flip"
    if abs(training_ic) > _NOISE_IC_THRESHOLD and abs(production_ic) < _NOISE_IC_THRESHOLD:
        return "decayed"
    return "stable"


def _load_features_from_arctic(
    bucket: str,
    tickers: list[str],
) -> dict[str, pd.DataFrame]:
    """Load feature DataFrames for specific tickers from ArcticDB."""
    try:
        from store.arctic_reader import OHLCV_COLS
        from alpha_engine_lib.arcticdb import open_universe_lib
    except ImportError:
        log.warning("ArcticDB reader not available — cannot load features")
        return {}

    try:
        universe = open_universe_lib(bucket)
        available = set(universe.list_symbols())

        result = {}
        for ticker in tickers:
            if ticker not in available:
                continue
            try:
                df = universe.read(ticker).data
                if not df.empty:
                    # Drop OHLCV columns to keep only features
                    feat_cols = [c for c in df.columns if c not in OHLCV_COLS]
                    if feat_cols:
                        result[ticker] = df[feat_cols]
            except Exception:
                pass

        log.info("Feature drift: loaded features for %d/%d tickers from ArcticDB", len(result), len(tickers))
        return result

    except Exception as exc:
        log.warning("Feature drift: ArcticDB load failed: %s", exc)
        return {}


def _join_outcomes_with_features(
    outcomes: pd.DataFrame,
    features_by_ticker: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Join prediction outcomes with feature values on (ticker, date)."""
    rows = []
    for _, row in outcomes.iterrows():
        ticker = row["symbol"]
        pred_date = row["prediction_date"]

        if ticker not in features_by_ticker:
            continue

        feat_df = features_by_ticker[ticker]

        # Find the feature row matching prediction date (or nearest prior date)
        if pred_date in feat_df.index:
            feat_row = feat_df.loc[pred_date]
        else:
            # Use the most recent feature row on or before the prediction date
            prior = feat_df.index[feat_df.index <= pred_date]
            if len(prior) == 0:
                continue
            feat_row = feat_df.loc[prior[-1]]

        combined = {
            "symbol": ticker,
            "prediction_date": pred_date,
            "canonical_actual": row["canonical_actual"],
            "actual": row["actual"],
            "horizon_days": row.get("horizon_days"),
        }
        combined.update(feat_row.to_dict())
        rows.append(combined)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def _load_training_feature_ics(bucket: str) -> dict[str, float]:
    """Load per-feature ICs from the most recent training summary."""
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key="predictor/metrics/training_summary_latest.json")
        summary = json.loads(obj["Body"].read())
        return summary.get("feature_ics", {})
    except Exception as exc:
        log.debug("Failed to load training feature ICs: %s", exc)
        return {}
