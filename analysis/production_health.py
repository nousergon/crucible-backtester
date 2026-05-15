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
    ACTIVE_HORIZON_DAYS,
    ALPHA_COALESCE_SQL,
    CANONICAL_CUTOVER_DATE,
    CORRECT_COALESCE_SQL,
    CURRENT_HORIZON_FILTER_SQL,
    HORIZON_COALESCE_SQL,
    OUTCOMES_GRADED_SQL,
    POST_CUTOVER_FILTER_SQL,
)

log = logging.getLogger(__name__)

_MIN_SAMPLES = 10
_IC_STD_EPSILON = 1e-8
_MODE_COLLAPSE_THRESHOLD = 0.75  # if any direction > 75% of predictions → flag
_DEGRADATION_RATIO = 0.50  # flag if production IC < 50% of training IC


def _classify_skipped_window(conn, cutoff: str, n_current: int) -> dict:
    """Build the skip payload when the current-model window is under-sized.

    Distinguishes the expected post-cutover maturation blackout from a
    genuine data gap. `horizon_days = N` does NOT isolate the post-cutover
    model — the grader stamps `horizon_days` at grade time, so pre-cutover
    predictions whose 21d window closed post-migration also carry it. So
    production analytics additionally scope to `prediction_date >= the
    canonical cutover`. During the ~21-trading-day window after the cutover
    there are graded outcomes in the lookback but none from the post-cutover
    model. Re-count graded rows ignoring both the horizon and cutover
    filters: if such rows exist, this is the blackout (self-clears); only
    a truly empty graded window is a real `insufficient_samples` gap.

    Shared by compute_production_health + compute_calibration_validation so
    the IC and ECE paths classify the blackout identically.
    """
    n_any = int(
        pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM predictor_outcomes "
            f"WHERE {OUTCOMES_GRADED_SQL} AND prediction_date >= ?",
            conn,
            params=(cutoff,),
        )["n"].iloc[0]
    )
    if n_any >= _MIN_SAMPLES:
        msg = (
            f"Post-cutover blackout: {n_any} resolved outcomes in window but "
            f"{n_current} from the post-cutover model (prediction_date >= "
            f"{CANONICAL_CUTOVER_DATE}) at the active {ACTIVE_HORIZON_DAYS}d "
            f"horizon (< {_MIN_SAMPLES}). Pre-cutover-model rows are excluded "
            f"by design — their confidence/score semantics differ from the "
            f"current model. Current-model outcomes mature ~"
            f"{ACTIVE_HORIZON_DAYS} trading days after the "
            f"{CANONICAL_CUTOVER_DATE} cutover. Self-clears — not a "
            f"degradation and not a broken pipeline."
        )
        log.info("Production health: %s", msg)
        return {
            "status": "skipped",
            "reason": "post_cutover_ic_blackout",
            "n": n_current,
            "n_any_horizon": n_any,
            "active_horizon_days": ACTIVE_HORIZON_DAYS,
            "message": msg,
        }

    log.info(
        "Production health: %d post-cutover current-horizon outcomes (< %d) — skipping",
        n_current, _MIN_SAMPLES,
    )
    return {
        "status": "skipped",
        "reason": "insufficient_samples",
        "n": n_current,
        "n_any_horizon": n_any,
    }


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
        f"  AND {POST_CUTOVER_FILTER_SQL} "
        f"  AND prediction_date >= ?",
        conn,
        params=(cutoff,),
    )
    if len(df) < _MIN_SAMPLES:
        result = _classify_skipped_window(conn, cutoff, len(df))
        conn.close()
        return result

    conn.close()

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
    # Reference IC must be the model's OOS performance at the active horizon,
    # NOT in-sample fit. See _load_training_ic for the preference chain — the
    # 2026-05-11 false-positive retrain alert traced to this reference being
    # `meta_model_ic = 0.4634` (Ridge in-sample Pearson) instead of the
    # honest 21d OOS Spearman of 0.166.
    training_ic, training_ic_source = _load_training_ic(bucket)
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
        "training_ic_source": training_ic_source,
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


def _load_training_ic(
    bucket: str,
    active_horizon_days: int = ACTIVE_HORIZON_DAYS,
) -> tuple[float | None, str]:
    """Return (training_ic, source) from predictor/metrics/training_summary_latest.json.

    Preference chain — first available wins:

    1. ``meta_model_oos_ic`` (top-level). Published by alpha-engine-predictor
       after the rename arc (PR #2 of the 2026-05-11 follow-ups). Honest
       OOS measurement at the active horizon.
    2. ``horizon_diagnostic.curve.{H}d.spearman`` where H = active_horizon_days.
       The walk-forward Spearman IC at the production label horizon — already
       computed and persisted by the training pipeline today.
    3. ``walk_forward.median_ic`` (legacy v2 / pre-cutover). Logs a warning
       because the v3 meta-model writes its IN-SAMPLE Pearson fit here under
       certain code paths, which overstates OOS skill ~3× and produces
       false-positive degradation alerts.
    4. ``test_ic`` (deepest legacy).

    The returned source string is persisted in production_health.json so
    operators can see WHICH reference drove the degradation flag. If we
    silently fall back to a legacy/in-sample field, the source label makes
    that visible without needing to re-derive the chain by reading code.
    """
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key="predictor/metrics/training_summary_latest.json")
        summary = json.loads(obj["Body"].read())
    except Exception as exc:
        log.debug("Failed to load training summary: %s", exc)
        return None, "load_failed"

    if summary.get("meta_model_oos_ic") is not None:
        return float(summary["meta_model_oos_ic"]), "meta_model_oos_ic"

    horizon_key = f"{active_horizon_days}d"
    curve_entry = (
        summary.get("horizon_diagnostic", {})
        .get("curve", {})
        .get(horizon_key, {})
    )
    spearman = curve_entry.get("spearman")
    if spearman is not None:
        return float(spearman), f"horizon_diagnostic.curve.{horizon_key}.spearman"

    wf = summary.get("walk_forward", {})
    if "median_ic" in wf:
        log.warning(
            "production_health: training_ic reference falling back to legacy "
            "walk_forward.median_ic — under v3 meta-model this is the Ridge's "
            "in-sample Pearson fit and overstates OOS skill. Expect "
            "false-positive degradation alerts."
        )
        return float(wf["median_ic"]), "walk_forward.median_ic_legacy"

    if "test_ic" in summary:
        return float(summary["test_ic"]), "test_ic_legacy"

    return None, "absent"


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
        f"  AND {POST_CUTOVER_FILTER_SQL} "
        f"  AND prediction_date >= ?",
        conn,
        params=(cutoff,),
    )

    if len(df) < _MIN_SAMPLES:
        # Same blackout classification as the IC path: pre-cutover-model
        # rows graded at 21d would otherwise pool a stale, semantically
        # mismatched population into the ECE (the 2026-05-15 spurious
        # calibration_breakdown). During the post-cutover maturation
        # window this self-clears.
        result = _classify_skipped_window(conn, cutoff, len(df))
        conn.close()
        return result

    conn.close()

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
