"""
analysis/production_health.py — Monitor predictor model quality in production.

Phase 2a: Rolling IC + hit rate stratified by market regime, training IC comparison,
          prediction distribution monitoring (mode collapse detection).
Phase 2b: Per-bin confidence calibration validation.

Reads from:
  - predictor_outcomes table (research.db) — predictions + resolved actual returns
  - signals/{date}/signals.json (S3) — market_regime per date
  - predictor/metrics/training_summary_latest.json (S3) — training IC for comparison

Writes to:
  - predictor/metrics/production_health.json (S3)
  - predictor/metrics/calibration_validation.json (S3)
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

from pipeline_common import (
    ALPHA_COALESCE_SQL,
    CORRECT_COALESCE_SQL,
    CURRENT_HORIZON_FILTER_SQL,
    HORIZON_COALESCE_SQL,
    OUTCOMES_GRADED_SQL,
)

log = logging.getLogger(__name__)

_MIN_SAMPLES = 10
_IC_STD_EPSILON = 1e-8
_MODE_COLLAPSE_THRESHOLD = 0.75  # if any direction > 75% of predictions → flag
_DEGRADATION_RATIO = 0.50  # flag if production IC < 50% of training IC


def compute_production_health(
    db_path: str,
    bucket: str,
    run_date: str | None = None,
    lookback_days: int = 30,
) -> dict:
    """
    Compute production model health metrics and write to S3.

    Returns summary dict with IC, hit rate, regime breakdown, and flags.
    """
    run_date = run_date or datetime.utcnow().strftime("%Y-%m-%d")
    cutoff = (datetime.strptime(run_date, "%Y-%m-%d") - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    # ── Load resolved predictor outcomes ─────────────────────────────────────
    if not os.path.exists(db_path):
        log.warning("research.db not found at %s — skipping production health", db_path)
        return {"status": "skipped", "reason": "no_db"}

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT symbol, prediction_date, predicted_direction, prediction_confidence, "
        "p_up, p_down, "
        f"{ALPHA_COALESCE_SQL} AS canonical_actual, "
        f"{CORRECT_COALESCE_SQL} AS canonical_correct, "
        f"{HORIZON_COALESCE_SQL} AS horizon_days "
        "FROM predictor_outcomes "
        f"WHERE {OUTCOMES_GRADED_SQL} "
        f"  AND {CURRENT_HORIZON_FILTER_SQL} "
        f"  AND prediction_date >= ?",
        conn,
        params=(cutoff,),
    )
    conn.close()

    if len(df) < _MIN_SAMPLES:
        log.info("Production health: %d resolved outcomes (< %d minimum) — skipping", len(df), _MIN_SAMPLES)
        return {"status": "skipped", "reason": "insufficient_samples", "n": len(df)}

    # ── Rolling IC & hit rate ────────────────────────────────────────────────
    df["net_signal"] = pd.to_numeric(df["p_up"], errors="coerce").fillna(0) - pd.to_numeric(df["p_down"], errors="coerce").fillna(0)
    df["actual"] = pd.to_numeric(df["canonical_actual"], errors="coerce")
    df["correct"] = pd.to_numeric(df["canonical_correct"], errors="coerce")

    hit_rate = float(df["correct"].mean())

    valid = df.dropna(subset=["net_signal", "actual"])
    ic_30d = None
    if len(valid) >= _MIN_SAMPLES:
        from scipy.stats import pearsonr
        ic_val, _ = pearsonr(valid["net_signal"], valid["actual"])
        ic_30d = round(float(ic_val), 4)

    # ── Regime-stratified IC ─────────────────────────────────────────────────
    regime_ic = _compute_regime_ic(df, bucket)

    # ── Training IC comparison ───────────────────────────────────────────────
    training_ic = _load_training_ic(bucket)
    ic_ratio = round(ic_30d / training_ic, 2) if ic_30d is not None and training_ic and training_ic > 0 else None
    degradation_flag = ic_ratio is not None and ic_ratio < _DEGRADATION_RATIO

    # ── Prediction distribution (mode collapse check) ────────────────────────
    direction_counts = df["predicted_direction"].value_counts(normalize=True).to_dict()
    prediction_distribution = {d: round(float(direction_counts.get(d, 0)), 3) for d in ["UP", "FLAT", "DOWN"]}
    mode_collapse_flag = any(v > _MODE_COLLAPSE_THRESHOLD for v in prediction_distribution.values())

    # ── Build result ─────────────────────────────────────────────────────────
    result = {
        "date": run_date,
        "lookback_days": lookback_days,
        "n_resolved": len(df),
        "rolling_30d_ic": ic_30d,
        "rolling_30d_hit_rate": round(hit_rate, 4),
        "regime_ic": regime_ic,
        "training_ic": training_ic,
        "ic_ratio": ic_ratio,
        "degradation_flag": degradation_flag,
        "prediction_distribution": prediction_distribution,
        "mode_collapse_flag": mode_collapse_flag,
    }

    # ── Write to S3 ──────────────────────────────────────────────────────────
    try:
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=bucket,
            Key="predictor/metrics/production_health.json",
            Body=json.dumps(result, indent=2, default=str).encode(),
            ContentType="application/json",
        )
        log.info(
            "Production health: IC=%.4f  hit_rate=%.3f  degradation=%s  mode_collapse=%s  n=%d",
            ic_30d or 0, hit_rate, degradation_flag, mode_collapse_flag, len(df),
        )
    except Exception as exc:
        log.warning("Failed to write production_health.json to S3: %s", exc)

    return result


def _compute_regime_ic(df: pd.DataFrame, bucket: str) -> dict[str, float | None]:
    """Compute IC per market regime by joining predictions with signals dates."""
    from scipy.stats import pearsonr

    # Load regime per date from signals
    regime_by_date = _load_regime_by_date(bucket)
    if not regime_by_date:
        log.debug("No regime data available — skipping regime IC")
        return {}

    df = df.copy()
    df["regime"] = df["prediction_date"].map(regime_by_date)
    df["net_signal"] = pd.to_numeric(df["p_up"], errors="coerce").fillna(0) - pd.to_numeric(df["p_down"], errors="coerce").fillna(0)
    # canonical_actual already computed by the SELECT in compute_production_health;
    # _compute_regime_ic receives the same DataFrame so the column is present.
    df["actual"] = pd.to_numeric(df["canonical_actual"], errors="coerce")

    regime_ic = {}
    for regime in ["bull", "neutral", "bear", "caution"]:
        subset = df[df["regime"] == regime].dropna(subset=["net_signal", "actual"])
        if len(subset) >= _MIN_SAMPLES:
            ic_val, _ = pearsonr(subset["net_signal"], subset["actual"])
            regime_ic[regime] = round(float(ic_val), 4)
        else:
            regime_ic[regime] = None

    return regime_ic


def _load_regime_by_date(bucket: str) -> dict[str, str]:
    """Load market_regime from signals.json files for recent dates."""
    s3 = boto3.client("s3")
    regime_map: dict[str, str] = {}

    try:
        # List recent signal dates
        paginator = s3.get_paginator("list_objects_v2")
        prefixes = []
        for page in paginator.paginate(Bucket=bucket, Prefix="signals/", Delimiter="/"):
            for p in page.get("CommonPrefixes", []):
                prefixes.append(p["Prefix"])

        # Take last 60 dates
        prefixes = sorted(prefixes)[-60:]

        for prefix in prefixes:
            date_str = prefix.rstrip("/").split("/")[-1]
            key = f"{prefix}signals.json"
            try:
                obj = s3.get_object(Bucket=bucket, Key=key)
                signals = json.loads(obj["Body"].read())
                regime = signals.get("market_regime", "neutral")
                regime_map[date_str] = regime
            except Exception:
                pass
    except Exception as exc:
        log.debug("Failed to load regime data from signals: %s", exc)

    return regime_map


def _load_training_ic(bucket: str) -> float | None:
    """Load the training walk-forward median IC from training_summary_latest.json."""
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key="predictor/metrics/training_summary_latest.json")
        summary = json.loads(obj["Body"].read())
        wf = summary.get("walk_forward", {})
        return float(wf["median_ic"]) if "median_ic" in wf else float(summary.get("test_ic", 0))
    except Exception as exc:
        log.debug("Failed to load training IC: %s", exc)
        return None


_MIN_BIN_N = 10  # skip bins with fewer samples — ECE is noise-dominated below this


def _load_calibrator_deployed_at(bucket: str) -> str | None:
    """Read the isotonic calibrator sidecar to get its deployment timestamp.

    PR 1 (predictor) writes predictor/weights/meta/isotonic_calibrator.meta.json
    alongside the pickle. The sidecar's ``deployed_at`` field is an ISO-8601
    UTC timestamp used by retrain_alert.py to grace-period the
    calibration_breakdown trigger after a fresh calibrator (ECE over a mixed
    calibrator-semantics window is noisy by construction).

    Returns None when the sidecar is absent (no calibrator yet) or unreadable.
    """
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key="predictor/weights/meta/isotonic_calibrator.meta.json")
        sidecar = json.loads(obj["Body"].read())
        value = sidecar.get("deployed_at")
        return str(value) if value else None
    except Exception as exc:
        log.debug("Calibrator sidecar not available: %s", exc)
        return None


def compute_calibration_validation(
    db_path: str,
    bucket: str,
    run_date: str | None = None,
    lookback_days: int = 60,
    min_bin_n: int = _MIN_BIN_N,
) -> dict:
    """
    Phase 2b: Per-bin confidence calibration validation.

    For each confidence bin with at least ``min_bin_n`` samples, compute the
    actual hit rate and compare it to the mean predicted confidence within
    that bin. ``expected`` is the mean of predicted confidences in the bin —
    this is the rigorous form of ECE. Using bin midpoints instead would
    systematically overstate miscalibration when predictions cluster at one
    end of a bin.

    Bins with fewer than ``min_bin_n`` samples are dropped from the ECE
    computation to avoid noise domination in sparse tails.

    Writes to predictor/metrics/calibration_validation.json.
    """
    run_date = run_date or datetime.utcnow().strftime("%Y-%m-%d")
    cutoff = (datetime.strptime(run_date, "%Y-%m-%d") - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    if not os.path.exists(db_path):
        return {"status": "skipped", "reason": "no_db"}

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        f"SELECT prediction_confidence, {CORRECT_COALESCE_SQL} AS canonical_correct "
        "FROM predictor_outcomes "
        f"WHERE {OUTCOMES_GRADED_SQL} "
        f"  AND {CURRENT_HORIZON_FILTER_SQL} "
        f"  AND prediction_date >= ?",
        conn,
        params=(cutoff,),
    )
    conn.close()

    if len(df) < _MIN_SAMPLES:
        return {"status": "skipped", "reason": "insufficient_samples", "n": len(df)}

    df["confidence"] = pd.to_numeric(df["prediction_confidence"], errors="coerce")
    df["correct"] = pd.to_numeric(df["canonical_correct"], errors="coerce")
    df = df.dropna(subset=["confidence", "correct"])

    # ── Bin by confidence ────────────────────────────────────────────────────
    bin_edges = [0.50, 0.60, 0.70, 0.80, 0.90, 1.01]
    bins = []
    dropped_bins = []
    total_ece = 0.0
    total_n = 0

    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (df["confidence"] >= lo) & (df["confidence"] < hi)
        subset = df[mask]
        n = len(subset)
        if n == 0:
            continue

        hit_rate = float(subset["correct"].mean())
        expected = float(subset["confidence"].mean())  # rigorous: mean predicted prob in bin

        bin_record = {
            "range": [round(lo, 2), round(min(hi, 1.0), 2)],
            "n": n,
            "hit_rate": round(hit_rate, 3),
            "expected": round(expected, 3),
        }

        if n < min_bin_n:
            bin_record["dropped_reason"] = f"n<{min_bin_n}"
            dropped_bins.append(bin_record)
            continue

        bins.append(bin_record)
        total_ece += abs(hit_rate - expected) * n
        total_n += n

    overall_ece = round(total_ece / total_n, 4) if total_n > 0 else None

    # Calibration quality label
    if overall_ece is None:
        quality = "unknown"
    elif overall_ece < 0.05:
        quality = "good"
    elif overall_ece < 0.10:
        quality = "acceptable"
    else:
        quality = "poor"

    result = {
        "date": run_date,
        "lookback_days": lookback_days,
        "min_bin_n": min_bin_n,
        "n_total": total_n,
        "bins": bins,
        "dropped_bins": dropped_bins,
        "overall_ece": overall_ece,
        "calibration_quality": quality,
    }

    # Propagate calibrator deployment timestamp when predictor writes it
    # alongside meta weights. retrain_alert.py uses this to apply a grace
    # period after a fresh calibrator (ECE is mixed-semantics during the
    # rollover window). Absent → no grace period, alerts fire normally.
    calibrator_deployed_at = _load_calibrator_deployed_at(bucket)
    if calibrator_deployed_at is not None:
        result["calibrator_deployed_at"] = calibrator_deployed_at

    # ── Write to S3 ──────────────────────────────────────────────────────────
    try:
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=bucket,
            Key="predictor/metrics/calibration_validation.json",
            Body=json.dumps(result, indent=2, default=str).encode(),
            ContentType="application/json",
        )
        log.info("Calibration validation: ECE=%.4f (%s)  bins=%d  n=%d", overall_ece or 0, quality, len(bins), total_n)
    except Exception as exc:
        log.warning("Failed to write calibration_validation.json to S3: %s", exc)

    return result
