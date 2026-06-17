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

from alpha_engine_lib.quant.stats.calibration import expected_calibration_error

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

# Calibration needs INDEPENDENT observations, not overlapping daily rows. At a
# 21d horizon, predictions made within ~horizon of each other share most of
# their forward window — their outcomes are correlated, so raw row count
# massively overstates the effective sample. ECE over 2-3 independent windows
# is noise. Require this many non-overlapping horizon-length cohorts before the
# ECE is trustworthy enough to drive a calibration_breakdown retrain alert.
_MIN_INDEPENDENT_WINDOWS = 3
# trading days → calendar days for cohort spacing (~7/5 + slack).
_TRADING_TO_CALENDAR = 1.45

_PROD_HEALTH_KEY = "predictor/metrics/production_health.json"
_CALIBRATION_KEY = "predictor/metrics/calibration_validation.json"


def _persist_metric(bucket: str, key: str, result: dict) -> None:
    """Best-effort write of a metrics artifact to S3.

    Called on EVERY non-error return path (skip/blackout AND full compute)
    so the standalone artifact always reflects the latest run. Before this,
    skip/blackout paths returned before the write, freezing
    production_health.json at the last full-compute run's
    `degradation_flag` — the 2026-05-15 forensic landmine that opened this
    investigation (`training_ic_source: None` was read off that stale file).
    """
    try:
        boto3.client("s3").put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(result, indent=2, default=str).encode(),
            ContentType="application/json",
        )
        log.info("Wrote %s (%s)", key, result.get("reason") or result.get("status") or "ok")
    except Exception as exc:
        log.warning("Failed to write %s to S3: %s", key, exc)


def _classify_skipped_window(conn, cutoff: str, n_current: int, run_date: str) -> dict:
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
            "date": run_date,
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
        "date": run_date,
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
        result = _classify_skipped_window(conn, cutoff, len(df), run_date)
        conn.close()
        _persist_metric(bucket, _PROD_HEALTH_KEY, result)
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

    # ── Per-L1 + L2 IC decomposition (ROADMAP L135) ──────────────────────────
    # Diagnostic for which ensemble component is contributing or drifting:
    # Spearman IC against canonical_actual for each L1 (momentum,
    # volatility, research_calibrator) plus L2 (predicted_alpha) separately,
    # so meta-learner-vs-L1-baseline is comparable. The per-L1 values aren't
    # persisted to ``predictor_outcomes`` — read from the per-date
    # predictions/{date}.json artifacts and merged in-memory.
    ic_decomposition = _compute_l1_l2_ic_decomposition(df, bucket)

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
        # ROADMAP L135 — per-L1 component IC + L2 stacker IC + L2 lift.
        # `None` for any component when the join with predictions/{date}.json
        # couldn't be performed (early-window, missing artifacts) or sample
        # count fell below ``_MIN_SAMPLES``.
        "l1_components": ic_decomposition["per_l1"],
        "l2_alpha_ic": ic_decomposition["l2_alpha"],
        "l2_lift_vs_l1_mean": ic_decomposition["l2_lift_vs_l1_mean"],
        "l1_l2_n_joined": ic_decomposition["n_joined"],
    }

    log.info(
        "Production health: IC=%.4f  hit_rate=%.3f  degradation=%s  mode_collapse=%s  n=%d",
        ic_30d or 0, hit_rate, degradation_flag, mode_collapse_flag, len(df),
    )
    _persist_metric(bucket, _PROD_HEALTH_KEY, result)
    return result


# ── L1 + L2 IC decomposition (ROADMAP L135) ──────────────────────────────────


_L1_COMPONENT_FIELDS = {
    # L1 component name on production_health.json → field in predictions/{date}.json
    "momentum": "momentum_confirmation",
    "volatility": "expected_move",
    "research_calibrator": "research_calibrator_prob",
}


def _load_l1_predictions_for_dates(
    bucket: str, dates: list[str], s3_client=None
) -> pd.DataFrame:
    """Return per-(symbol, date) L1 component values + L2 ``predicted_alpha``
    from ``predictor/predictions/{date}.json`` for the given dates.

    Empty DataFrame when no dates parse or every artifact fetch fails —
    callers degrade gracefully (decomposition values become ``None``).
    """
    if not dates:
        return pd.DataFrame()
    s3 = s3_client or boto3.client("s3")
    rows: list[dict] = []
    for d in sorted({str(d) for d in dates if d}):
        try:
            obj = s3.get_object(Bucket=bucket, Key=f"predictor/predictions/{d}.json")
            payload = json.loads(obj["Body"].read())
        except Exception as e:  # noqa: BLE001 — secondary observability; primary IC path unaffected
            log.debug("L1 decomposition: predictions/%s.json fetch failed: %s", d, e)
            continue
        for p in payload.get("predictions", []) or []:
            ticker = p.get("ticker")
            if not ticker:
                continue
            rows.append(
                {
                    "symbol": ticker,
                    "prediction_date": d,
                    "momentum_confirmation": p.get("momentum_confirmation"),
                    "expected_move": p.get("expected_move"),
                    "research_calibrator_prob": p.get("research_calibrator_prob"),
                    "predicted_alpha": p.get("predicted_alpha"),
                }
            )
    return pd.DataFrame(rows)


def _spearman_or_none(x: pd.Series, y: pd.Series) -> float | None:
    """Compute Spearman rank correlation; ``None`` on insufficient samples
    or zero-variance series (rank tie collapse).
    """
    if len(x) < _MIN_SAMPLES:
        return None
    if x.nunique() < 2 or y.nunique() < 2:
        return None
    from scipy.stats import spearmanr  # local — avoid eager import at module load

    rho, _ = spearmanr(x, y)
    if rho is None or pd.isna(rho):
        return None
    return round(float(rho), 4)


def _compute_l1_l2_ic_decomposition(df: pd.DataFrame, bucket: str) -> dict:
    """Per-L1 + L2 Spearman IC against ``canonical_actual``.

    Returns a dict with shape::

        {
          "per_l1": {"momentum": float|None, "volatility": float|None,
                     "research_calibrator": float|None},
          "l2_alpha": float|None,
          "l2_lift_vs_l1_mean": float|None,
          "n_joined": int,
        }

    ``l2_lift_vs_l1_mean`` is the load-bearing diagnostic per ROADMAP L135 —
    if the L2 stacker is not lifting above the L1-mean baseline, ensemble
    averaging would do as well as the Ridge meta-learner. ``None`` when
    fewer than 1 valid L1 IC was computable (e.g. early-window predictions
    artifacts haven't been written yet).
    """
    out: dict = {
        "per_l1": {name: None for name in _L1_COMPONENT_FIELDS},
        "l2_alpha": None,
        "l2_lift_vs_l1_mean": None,
        "n_joined": 0,
    }

    if df.empty:
        return out

    dates = df["prediction_date"].dropna().astype(str).unique().tolist()
    l1_df = _load_l1_predictions_for_dates(bucket, dates)
    if l1_df.empty:
        log.debug("L1 decomposition: zero predictions artifacts joined")
        return out

    merged = df.merge(l1_df, on=["symbol", "prediction_date"], how="left")
    out["n_joined"] = int(merged[[c for c in l1_df.columns if c not in {"symbol", "prediction_date"}]].notna().any(axis=1).sum())

    valid_actual = merged["actual"].notna()

    for component, field_name in _L1_COMPONENT_FIELDS.items():
        if field_name not in merged.columns:
            continue
        col_valid = valid_actual & merged[field_name].notna()
        subset = merged.loc[col_valid]
        out["per_l1"][component] = _spearman_or_none(subset[field_name], subset["actual"])

    if "predicted_alpha" in merged.columns:
        l2_valid = valid_actual & merged["predicted_alpha"].notna()
        l2_subset = merged.loc[l2_valid]
        out["l2_alpha"] = _spearman_or_none(l2_subset["predicted_alpha"], l2_subset["actual"])

    l1_ics = [v for v in out["per_l1"].values() if v is not None]
    if out["l2_alpha"] is not None and l1_ics:
        l1_mean = sum(l1_ics) / len(l1_ics)
        out["l2_lift_vs_l1_mean"] = round(out["l2_alpha"] - l1_mean, 4)

    log.info(
        "L1/L2 IC decomposition: per_l1=%s  l2=%s  l2_lift=%s  n_joined=%d",
        out["per_l1"], out["l2_alpha"], out["l2_lift_vs_l1_mean"], out["n_joined"],
    )
    return out


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

    # 3-class Ang-Bekaert macro taxonomy (v0.42.0 / 2026-05-28 —
    # caution-regime-retirement-260528.md). Iterate the legacy 4-class
    # set so pre-v0.42.0 score_performance rows whose market_regime
    # carries "caution" still surface a stratified IC for grandfather
    # attribution continuity. New (post-cutover) rows never populate
    # the caution bucket — its IC is None for those windows.
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


def _effective_independent_windows(
    prediction_dates: pd.Series | list,
    horizon_days: int = ACTIVE_HORIZON_DAYS,
) -> int:
    """Count non-overlapping horizon-length cohorts among the prediction dates.

    Predictions made within one horizon of each other share most of their
    forward window, so their outcomes are correlated — they are NOT independent
    calibration observations. Greedily walk the sorted distinct prediction
    dates, counting a new cohort only once we are at least one horizon (in
    calendar days) past the last counted date. This is the binding sample-size
    constraint for ECE at a 21d horizon: 60 daily rows over 3 weeks is ~1
    independent window, not 60.
    """
    parsed = sorted(
        {
            datetime.strptime(str(d)[:10], "%Y-%m-%d")
            for d in prediction_dates
            if d is not None and str(d)[:10]
        }
    )
    if not parsed:
        return 0
    horizon_cal = max(1, int(round(horizon_days * _TRADING_TO_CALENDAR)))
    count = 1
    anchor = parsed[0]
    for d in parsed[1:]:
        if (d - anchor).days >= horizon_cal:
            count += 1
            anchor = d
    return count


def compute_calibration_validation(
    db_path: str,
    bucket: str,
    run_date: str | None = None,
    lookback_days: int = 60,
    min_bin_n: int = _MIN_BIN_N,
) -> dict:
    """
    Phase 2b: Per-bin probability calibration validation.

    Measures ECE the rigorous way: bin the calibrated UP probability ``p_up``
    and, in each bin, compare its mean to the empirical UP frequency
    (``1[realized_log_alpha > 0]``). Both are probabilities on the same [0,1]
    scale, so the ECE is comparable to the predictor's own training-time ECE
    (both call ``alpha_engine_lib.quant.stats.calibration``).

    WHY NOT ``prediction_confidence``: that field is ``|p_up - 0.5| * 2`` — a
    *margin*, not a probability (since the 2026-05-12 convention flip,
    predictor #143). Binning the margin against the direction hit-rate compares
    two scales and manufactures a structural ECE (~0.2-0.25) on a perfectly
    calibrated model — the cause of the recurring false ``calibration_breakdown``
    retrain alerts. Fixed by measuring ``p_up`` vs the UP outcome here.

    Two guards keep the ECE honest:
      - raw-sample floor (``_MIN_SAMPLES``) + post-cutover blackout, as the IC
        path;
      - effective-independent-window floor (``_MIN_INDEPENDENT_WINDOWS``):
        overlapping 21d-horizon daily rows are correlated, so a handful of
        independent windows is too few to trust — skip rather than fire.

    Writes to predictor/metrics/calibration_validation.json.
    """
    run_date = run_date or datetime.utcnow().strftime("%Y-%m-%d")
    cutoff = (datetime.strptime(run_date, "%Y-%m-%d") - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    if not os.path.exists(db_path):
        return {"status": "skipped", "reason": "no_db"}

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT prediction_date, p_up, "
        f"{ALPHA_COALESCE_SQL} AS canonical_actual "
        "FROM predictor_outcomes "
        f"WHERE {OUTCOMES_GRADED_SQL} "
        "  AND p_up IS NOT NULL "
        f"  AND {CURRENT_HORIZON_FILTER_SQL} "
        f"  AND {POST_CUTOVER_FILTER_SQL} "
        f"  AND prediction_date >= ?",
        conn,
        params=(cutoff,),
    )

    df["p_up"] = pd.to_numeric(df["p_up"], errors="coerce")
    df["canonical_actual"] = pd.to_numeric(df["canonical_actual"], errors="coerce")
    df = df.dropna(subset=["p_up", "canonical_actual"])

    if len(df) < _MIN_SAMPLES:
        # Same blackout classification as the IC path: pre-cutover-model
        # rows graded at 21d would otherwise pool a stale, semantically
        # mismatched population into the ECE (the 2026-05-15 spurious
        # calibration_breakdown). During the post-cutover maturation
        # window this self-clears.
        result = _classify_skipped_window(conn, cutoff, len(df), run_date)
        conn.close()
        _persist_metric(bucket, _CALIBRATION_KEY, result)
        return result

    conn.close()

    # ── Effective-independent-window adequacy gate ────────────────────────────
    n_windows = _effective_independent_windows(df["prediction_date"])
    if n_windows < _MIN_INDEPENDENT_WINDOWS:
        msg = (
            f"Calibration skipped: {len(df)} post-cutover outcomes span only "
            f"{n_windows} independent {ACTIVE_HORIZON_DAYS}d window(s) "
            f"(< {_MIN_INDEPENDENT_WINDOWS}). Overlapping daily rows at this "
            f"horizon are correlated; ECE over this few independent windows is "
            f"noise-dominated and must not drive a retrain alert. Self-clears as "
            f"post-cutover history accumulates."
        )
        log.info("Calibration validation: %s", msg)
        result = {
            "date": run_date,
            "status": "skipped",
            "reason": "insufficient_independent_windows",
            "n": int(len(df)),
            "n_independent_windows": n_windows,
            "min_independent_windows": _MIN_INDEPENDENT_WINDOWS,
            "active_horizon_days": ACTIVE_HORIZON_DAYS,
            "message": msg,
        }
        _persist_metric(bucket, _CALIBRATION_KEY, result)
        return result

    # ── ECE: calibrated p_up vs the realized UP outcome (both probabilities) ──
    actual_up = (df["canonical_actual"].to_numpy() > 0).astype(float)
    ece_result = expected_calibration_error(
        df["p_up"].to_numpy(), actual_up, n_bins=10, min_bin_n=min_bin_n,
    )
    overall_ece = ece_result.get("ece")
    total_n = int(ece_result.get("n", 0))

    # Map the lib's per-bin records to the persisted artifact shape. The
    # persisted key stays ``expected`` (mean predicted prob in bin) for
    # display/dashboard continuity — additive ``mean_pred`` rides alongside.
    def _as_bin(rec: dict) -> dict:
        out = {
            "range": rec.get("range"),
            "n": rec.get("n"),
            "hit_rate": rec.get("hit_rate"),
            "expected": rec.get("mean_pred"),
            "mean_pred": rec.get("mean_pred"),
        }
        if rec.get("dropped_reason"):
            out["dropped_reason"] = rec["dropped_reason"]
        return out

    bins = [_as_bin(b) for b in ece_result.get("bins", [])]
    dropped_bins = [_as_bin(b) for b in ece_result.get("dropped_bins", [])]

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
        "n_independent_windows": n_windows,
        "measured_on": "p_up_vs_realized_up",
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

    log.info(
        "Calibration validation: ECE=%s (%s)  bins=%d  n=%d  windows=%d",
        f"{overall_ece:.4f}" if overall_ece is not None else "n/a",
        quality, len(bins), total_n, n_windows,
    )
    _persist_metric(bucket, _CALIBRATION_KEY, result)
    return result
