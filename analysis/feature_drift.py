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
from botocore.exceptions import BotoCoreError, ClientError
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

        # Skip features with object dtype (e.g. nested dicts from non-unique
        # index data — should not happen after the _join_outcomes_with_features
        # fix, but guard defensively to avoid a confusing scipy error).
        if not np.issubdtype(valid[feat].dtype, np.number):
            log.warning("Feature drift: skipping feature %s — non-numeric dtype %s", feat, valid[feat].dtype)
            continue

        try:
            ic, _ = spearmanr(valid[feat], valid["actual"])
        except (ValueError, TypeError, AttributeError) as exc:
            # scipy can raise or return non-standard types for edge-case inputs
            # (uniform values, single-element arrays, mixed object dtypes).
            # log and skip this feature rather than crashing the whole module.
            log.warning(
                "Feature drift: skipping feature %s — spearmanr failed: %s: %s",
                feat, type(exc).__name__, exc,
            )
            continue

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
    except (ClientError, BotoCoreError, TypeError, ValueError) as exc:
        # Fail-soft: the drift result is already computed and returned to the
        # caller; persisting it to S3 is best-effort. Narrowed to S3/botocore
        # transport errors plus the TypeError/ValueError a non-serializable
        # value in ``result`` would raise in json.dumps.
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
        from nousergon_lib.arcticdb import open_universe_lib
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
            except Exception as exc:  # noqa: BLE001
                # Intentionally broad per-ticker resilience boundary: one
                # corrupt / unreadable symbol must not abort the whole feature
                # load. ArcticDB read raises library-specific exceptions whose
                # hierarchy is not statically importable here (arcticdb is an
                # optional, lazily-imported dep), so we cannot enumerate a
                # precise type set — skip the bad ticker and continue.
                # config#806 silent-fail hardening: log the skip so a corrupt
                # symbol is observable in run logs instead of vanishing.
                log.warning(
                    "Feature drift: skipping ticker %s — ArcticDB read failed: %s: %s",
                    ticker, type(exc).__name__, exc,
                )
                continue

        log.info("Feature drift: loaded features for %d/%d tickers from ArcticDB", len(result), len(tickers))
        return result

    except Exception as exc:  # noqa: BLE001
        # Intentionally broad ArcticDB-connection boundary: open_universe_lib /
        # list_symbols failures (network, missing library, auth) are all
        # arcticdb-internal exception types not importable without the optional
        # arcticdb dep. Feature drift degrades to "no_feature_data" rather than
        # crashing the diagnostics run.
        log.warning("Feature drift: ArcticDB load failed: %s", exc)
        return {}


def _join_outcomes_with_features(
    outcomes: pd.DataFrame,
    features_by_ticker: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Join prediction outcomes with feature values on (ticker, date).

    Defensively handles edge cases from ArcticDB data:
    - Non-unique index dates: ``feat_df.loc`` returns a DataFrame; the last
      row is taken (most recent data wins).
    - Single-column DataFrame returning a scalar in some pandas versions:
      wrapped back to a Series with the column name.

    Raises on unexpected types — a data-shape change should surface early.
    """
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

        # Normalize feat_row to a Series with scalar values.
        if isinstance(feat_row, pd.DataFrame):
            # Non-unique index: multiple rows matched — take the last one.
            feat_row = feat_row.iloc[-1]
        elif not isinstance(feat_row, pd.Series):
            # Scalar return (e.g. single-column DataFrame + unique index in
            # some pandas versions) — wrap to Series with the column name so
            # ``to_dict()`` below produces ``{col_name: value}``.
            col_name = feat_df.columns[0]
            feat_row = pd.Series({col_name: feat_row}, name=row.get("prediction_date"))

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
    except (ClientError, BotoCoreError, ValueError, KeyError) as exc:
        # Fail-soft: a missing/unreadable training summary just means we can't
        # compare production ICs to a training baseline (drift comparison
        # degrades to "no baseline"). Narrowed to S3/botocore transport errors,
        # the ValueError from json.loads on a malformed body, and the KeyError
        # from an unexpected get_object response shape.
        log.debug("Failed to load training feature ICs: %s", exc)
        return {}
