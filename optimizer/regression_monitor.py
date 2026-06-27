"""
regression_monitor.py — rolling metrics, promotion baselines, and auto-rollback.

After each weekly backtester run:
  1. save_rolling_metrics() persists current metrics to S3 history
  2. check_regression() compares current metrics to the promotion baseline
  3. If thresholds are breached, rollback_all() restores previous configs
  4. write_rollback_audit() captures a structured artifact at
     ``config/rollback_audit/{run_date}.json`` documenting the trigger +
     rolled-back configs + which per-optimizer recommendations were
     rejected. Closes the "rollback fired silently" surface from the
     2026-05-09 forensic.

The promotion baseline is saved automatically before any optimizer writes new
params to S3.  This module is fully automated — no human approval needed.

Rollback trigger (post 2026-05-16 spurious-rollback forensic):
  - PRIMARY risk-adjusted gate is **Sortino-%** (skilled-risk evaluator
    revamp), OR accuracy-pp. Sharpe-% is persisted for observability only
    and is NO LONGER a rollback trigger.
  - Rollback is SUPPRESSED (regression may still be reported) on a
    degraded / low-statistical-power week (below min-trades / min-signals
    floors), or when the saved baseline is stale / pre-cutover (in which
    case the baseline is skipped AND refreshed from current metrics so
    subsequent runs compare post-cutover).
"""

import json
import logging
from datetime import date, datetime, timezone

from nousergon_lib.eval_artifacts import (
    eval_artifact_key,
    eval_latest_key,
    new_eval_run_id,
)
from typing import Any

import boto3
from botocore.exceptions import ClientError

from optimizer.assembler import read_assembled
from optimizer.recommendation_artifact import read_all_artifacts_for_date
from optimizer.rollback import rollback_all

logger = logging.getLogger(__name__)

S3_METRICS_PREFIX = "config/metrics_history/"
S3_BASELINE_KEY = "config/promotion_baseline.json"
S3_ROLLBACK_AUDIT_PREFIX = "config/rollback_audit/"

# Config types whose per-optimizer recommendation artifacts can be captured
# in the rollback audit's `rejected_recommendations` section. Currently
# only `executor_params` has the artifact contract wired (PRs 1-3 of the
# optimizer-artifact-assembler arc); extend when the scoring_weights /
# predictor_params follow-up arc ships.
ARTIFACT_CONFIG_TYPES = ("executor_params",)

# Default thresholds (can be overridden via config.yaml regression_monitor section)
DEFAULT_ACCURACY_DROP_PP = 5.0     # rollback if accuracy drops > 5 percentage points
DEFAULT_SHARPE_DROP_PCT = 0.20     # rollback if Sharpe drops > 20% (OBSERVABILITY ONLY,
#                                    no longer a rollback trigger — see 2026-05-16 arc)
DEFAULT_SORTINO_DROP_PCT = 0.20    # rollback if Sortino drops > 20% (PRIMARY
#                                    risk-adjusted gate post skilled-risk evaluator revamp)

# Min-sample / degraded-week guard. A low-statistical-power week must not be
# allowed to auto-rollback live configs (mirrors executor_optimizer.min_valid_combos
# / veto_analysis.min_predictions conventions). The 2026-05-16 spurious rollback
# fired on a Mode-2 n=55-trade / signal n=42 recovery week — both below sane floors.
DEFAULT_MIN_TRADES_FOR_ROLLBACK = 30
DEFAULT_MIN_SIGNALS_FOR_ROLLBACK = 30

# Stale-baseline guard. A baseline must not be compared across a framework /
# regime cutover or persist stale forever. The 2026-05-16 rollback compared a
# 2026-05-02 baseline (predating the 2026-05-09 canonical-alpha cutover AND the
# 2026-05-13 portfolio-optimizer cutover) against a 2026-05-16 run — apples to
# oranges. Age-based staleness is used (no clean canonical cutover constant
# exists in-repo / lib; age is simpler and self-healing across any future
# cutover). When the baseline is older than this many days it is treated as
# not-comparable: regression/rollback is skipped this run AND the baseline is
# refreshed from current metrics so subsequent runs compare post-cutover.
DEFAULT_BASELINE_MAX_AGE_DAYS = 21


def extract_metrics(portfolio_stats: dict | None, signal_quality: dict | None) -> dict:
    """Extract the metrics used for regression comparison."""
    metrics = {}

    if portfolio_stats and isinstance(portfolio_stats, dict):
        # sortino_ratio is the PRIMARY risk-adjusted regression gate post the
        # skilled-risk evaluator revamp; total_trades drives the min-sample
        # guard. sharpe_ratio kept for continuity/observability only.
        for key in (
            "sharpe_ratio", "sortino_ratio", "total_alpha", "max_drawdown",
            "win_rate", "total_trades",
        ):
            if key in portfolio_stats:
                metrics[key] = portfolio_stats[key]

    if signal_quality and isinstance(signal_quality, dict):
        overall = signal_quality.get("overall", {})
        for key in ("accuracy_10d", "accuracy_30d"):
            if key in overall:
                metrics[key] = overall[key]
        # n_10d is the resolved-signal sample size backing accuracy_10d —
        # used by the min-sample guard. Persisted as n_signals for a stable
        # cross-version key independent of the accuracy horizon.
        if "n_10d" in overall:
            metrics["n_signals"] = overall["n_10d"]

    return metrics


def save_rolling_metrics(bucket: str, run_date: str, metrics: dict) -> None:
    """Persist rolling metrics snapshot to S3."""
    if not metrics:
        logger.info("No metrics to save for %s", run_date)
        return

    payload = {
        "run_date": run_date,
        "saved_at": date.today().isoformat(),
        **metrics,
    }
    # Canonical eval-style archive layout per lib v0.8.0 — flat
    # {prefix}/{run_id}.json + latest.json sidecar (YYMMDDHHMM run_id).
    # S3_METRICS_PREFIX has trailing slash; strip via lib's normalization.
    run_id = new_eval_run_id()
    key = eval_artifact_key(S3_METRICS_PREFIX, run_id)
    latest_key = eval_latest_key(S3_METRICS_PREFIX)
    body = json.dumps(payload, indent=2)
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket, Key=key, Body=body, ContentType="application/json",
    )
    s3.put_object(
        Bucket=bucket, Key=latest_key, Body=body, ContentType="application/json",
    )
    logger.info(
        "Rolling metrics saved to s3://%s/%s (+ latest.json sidecar; run_date=%s)",
        bucket, key, run_date,
    )


def save_promotion_baseline(bucket: str, metrics: dict, promoted_configs: list[str]) -> None:
    """
    Save current metrics as the pre-promotion baseline.
    Called immediately before any optimizer apply() succeeds.
    """
    payload = {
        "saved_at": date.today().isoformat(),
        "promoted_configs": promoted_configs,
        **metrics,
    }
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket, Key=S3_BASELINE_KEY,
        Body=json.dumps(payload, indent=2),
        ContentType="application/json",
    )
    logger.info("Promotion baseline saved to s3://%s/%s (configs: %s)",
                bucket, S3_BASELINE_KEY, promoted_configs)


def _load_baseline(bucket: str) -> dict | None:
    """Load the promotion baseline from S3. Returns None if not found."""
    s3 = boto3.client("s3")
    try:
        resp = s3.get_object(Bucket=bucket, Key=S3_BASELINE_KEY)
        return json.loads(resp["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return None
        raise


def _baseline_age_days(baseline: dict, run_date: str | None) -> int | None:
    """Age in days between the baseline's ``saved_at`` and the current run.

    Returns None if either date is unparseable (treated as "age unknown" —
    the caller decides the conservative action).
    """
    saved_at = baseline.get("saved_at")
    if not saved_at:
        return None
    try:
        base_dt = date.fromisoformat(str(saved_at)[:10])
    except (ValueError, TypeError):
        return None
    try:
        ref_dt = (
            date.fromisoformat(str(run_date)[:10]) if run_date else date.today()
        )
    except (ValueError, TypeError):
        ref_dt = date.today()
    return (ref_dt - base_dt).days


def _refresh_baseline_from_current(
    bucket: str, current_metrics: dict, run_date: str | None,
) -> None:
    """Overwrite the promotion baseline with the current run's metrics.

    Called when the existing baseline is not comparable (stale / pre-cutover)
    so subsequent runs compare against a post-cutover baseline rather than
    silently continuing to compare against the stale one forever.
    """
    payload = {
        "saved_at": (str(run_date)[:10] if run_date else date.today().isoformat()),
        "promoted_configs": [],
        "refreshed_reason": "baseline_stale_refreshed",
        **current_metrics,
    }
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket, Key=S3_BASELINE_KEY,
        Body=json.dumps(payload, indent=2),
        ContentType="application/json",
    )
    logger.warning(
        "Promotion baseline refreshed from current metrics (stale baseline "
        "skipped) → s3://%s/%s", bucket, S3_BASELINE_KEY,
    )


def _capture_rejected_recommendations(
    bucket: str, run_date: str, s3_client=None,
) -> dict:
    """Read the per-optimizer recommendation artifacts + assembled output
    for each config_type that has them wired, so the rollback audit can
    record what was thrown away.

    Returns ``{config_type: {from_optimizers: {name: artifact_dict},
    assembled: dict | None}}``. Skipped config_types are simply absent
    from the dict. Failure to read any specific config_type is logged but
    does not raise — partial capture is better than no audit.
    """
    s3 = s3_client or boto3.client("s3")
    rejected: dict = {}
    for config_type in ARTIFACT_CONFIG_TYPES:
        try:
            artifacts = read_all_artifacts_for_date(
                bucket, config_type, run_date, s3_client=s3,
            )
            assembled = read_assembled(
                bucket, config_type, run_date, s3_client=s3,
            )
            if artifacts or assembled:
                rejected[config_type] = {
                    "from_optimizers": {
                        name: a.to_dict() for name, a in artifacts.items()
                    },
                    "assembled": assembled,
                }
        except Exception as e:
            logger.warning(
                "Failed to capture rejected recommendations for %s: %s "
                "(rollback audit will record partial state)",
                config_type, e,
            )
    return rejected


def write_rollback_audit(
    bucket: str,
    run_date: str,
    regression_check_result: dict,
    rollback_results: list[dict],
    s3_client: Any = None,
) -> str:
    """Write a structured audit artifact documenting an auto-rollback event.

    Path: ``config/rollback_audit/{run_date}.json``. Captures:

    - **trigger** — ``regression_detected``, ``details`` (accuracy/Sharpe drop),
      thresholds applied
    - **baseline + current** — the metrics being compared
    - **rollback_results** — per-config-type outcome from ``rollback_all``
    - **rejected_recommendations** — for each config_type with per-optimizer
      artifacts wired, the artifacts + assembled output that the rollback
      threw away (so operators can audit *what* the system tried to land
      before regression vetoed it)
    - **audit_timestamp** — UTC ISO8601 wall-clock for the audit write

    Failure is non-fatal — audit-write failure logs warn but does not raise.
    """
    s3 = s3_client or boto3.client("s3")
    audit_key = f"{S3_ROLLBACK_AUDIT_PREFIX}{run_date}.json"

    rejected = _capture_rejected_recommendations(bucket, run_date, s3_client=s3)

    audit_body = {
        "schema_version": 1,
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "run_date": run_date,
        "trigger": {
            "regression_detected": regression_check_result.get("regression_detected"),
            "details": regression_check_result.get("details", {}),
            "thresholds": {
                "accuracy_drop_threshold_pp": DEFAULT_ACCURACY_DROP_PP,
                "sortino_drop_threshold_pct": DEFAULT_SORTINO_DROP_PCT,
                "sharpe_drop_threshold_pct": DEFAULT_SHARPE_DROP_PCT,
                "min_trades_for_rollback": DEFAULT_MIN_TRADES_FOR_ROLLBACK,
                "min_signals_for_rollback": DEFAULT_MIN_SIGNALS_FOR_ROLLBACK,
                "baseline_max_age_days": DEFAULT_BASELINE_MAX_AGE_DAYS,
            },
        },
        "baseline": regression_check_result.get("baseline"),
        "current": regression_check_result.get("current"),
        "rollback_results": rollback_results,
        "rejected_recommendations": rejected,
        "notes": (
            "Auto-rollback fired due to regression detection vs the saved "
            "promotion baseline. The configs listed in rollback_results "
            "have been restored from their _previous.json snapshots. The "
            "rejected_recommendations section captures what the optimizers "
            "tried to land on this run before regression vetoed it — useful "
            "for diagnosing whether the rollback was correct or whether the "
            "baseline itself is stale."
        ),
    }

    try:
        s3.put_object(
            Bucket=bucket, Key=audit_key,
            Body=json.dumps(audit_body, indent=2, sort_keys=True),
            ContentType="application/json",
        )
        logger.info(
            "Rollback audit written: s3://%s/%s (rejected_configs=%d)",
            bucket, audit_key, len(rejected),
        )
        return audit_key
    except Exception as e:
        logger.warning(
            "Failed to write rollback audit s3://%s/%s: %s "
            "(non-fatal — rollback itself already completed)",
            bucket, audit_key, e,
        )
        return ""


def check_regression(
    bucket: str,
    current_metrics: dict,
    config: dict | None = None,
    run_date: str | None = None,
) -> dict:
    """
    Compare current rolling metrics against the saved promotion baseline.

    Args:
        bucket: S3 bucket name.
        current_metrics: Metrics from this run's portfolio_stats +
            signal_quality.
        config: Backtester config dict (for threshold overrides).
        run_date: YYYY-MM-DD for the current run. Used to key the
            rollback audit + read per-optimizer artifacts. If None,
            defaults to today's date (back-compat for existing callers
            that don't yet pass it).

    Returns:
        {
            "checked": True,
            "regression_detected": bool,
            "rollback_triggered": bool,
            "rollback_audit_key": str | None,
            "details": {
                "accuracy_drop": float|None,
                "sortino_drop_pct": float|None,
                "sharpe_drop_pct": float|None,    # observability only
                "guard": str|None,                # why rollback was suppressed
            },
            "baseline": dict|None,
            "current": dict,
        }

    Rollback trigger is **accuracy-pp OR sortino-%** (Sharpe is observability
    only post the skilled-risk evaluator revamp). Rollback is suppressed —
    even when a regression is *detected* — when (a) the run is below the
    min-trades / min-signals floor (degraded/low-power week), or (b) the
    baseline is stale / pre-cutover (in which case it is refreshed), or
    (c) the baseline lacks ``sortino_ratio`` (older baseline → not
    comparable on the primary gate).
    """
    baseline = _load_baseline(bucket)
    if baseline is None:
        logger.info("No promotion baseline found — skipping regression check")
        return {"checked": False, "reason": "no baseline"}

    reg_config = (config or {}).get("regression_monitor", {})
    acc_threshold = reg_config.get("accuracy_drop_threshold_pp", DEFAULT_ACCURACY_DROP_PP)
    sortino_threshold = reg_config.get(
        "sortino_drop_threshold_pct", DEFAULT_SORTINO_DROP_PCT,
    )
    sharpe_threshold = reg_config.get("sharpe_drop_threshold_pct", DEFAULT_SHARPE_DROP_PCT)
    min_trades = reg_config.get(
        "min_trades_for_rollback", DEFAULT_MIN_TRADES_FOR_ROLLBACK,
    )
    min_signals = reg_config.get(
        "min_signals_for_rollback", DEFAULT_MIN_SIGNALS_FOR_ROLLBACK,
    )
    baseline_max_age_days = reg_config.get(
        "baseline_max_age_days", DEFAULT_BASELINE_MAX_AGE_DAYS,
    )

    # ── Guard 3: stale / pre-cutover baseline ────────────────────────────────
    # Checked FIRST: a stale baseline must never be compared (the 2026-05-16
    # spurious rollback compared a 2026-05-02 pre-cutover baseline). Skip the
    # check this run AND refresh the baseline so subsequent runs compare
    # against a post-cutover baseline. Never silently keep comparing.
    age_days = _baseline_age_days(baseline, run_date)
    if age_days is not None and age_days > baseline_max_age_days:
        logger.warning(
            "Baseline is %d days old (> %d-day max) — treating as not "
            "comparable (likely predates a framework/regime cutover). "
            "Skipping regression check and refreshing baseline from current "
            "metrics.",
            age_days, baseline_max_age_days,
        )
        try:
            _refresh_baseline_from_current(bucket, current_metrics, run_date)
        except Exception as e:  # refresh failure must not break the run
            logger.warning(
                "Baseline refresh failed (%s) — next run will re-detect "
                "staleness and retry.", e,
            )
        return {
            "checked": False,
            "reason": "baseline_stale_refreshed",
            "regression_detected": False,
            "rollback_triggered": False,
            "details": {
                "baseline_age_days": age_days,
                "baseline_max_age_days": baseline_max_age_days,
                "guard": "baseline_stale_refreshed",
            },
            "baseline": baseline,
            "current": current_metrics,
        }

    details: dict = {}
    regression_detected = False

    # Accuracy check (10d) — secondary gate (retained)
    base_acc = baseline.get("accuracy_10d")
    curr_acc = current_metrics.get("accuracy_10d")
    if base_acc is not None and curr_acc is not None:
        # Accuracy is 0-1 (proportion), threshold is in percentage points
        drop_pp = (base_acc - curr_acc) * 100
        details["accuracy_drop"] = drop_pp
        if drop_pp > acc_threshold:
            regression_detected = True
            logger.warning(
                "Regression: accuracy_10d dropped %.1fpp (%.1f%% -> %.1f%%), threshold=%.1fpp",
                drop_pp, base_acc * 100, curr_acc * 100, acc_threshold,
            )

    # Sortino check — PRIMARY risk-adjusted gate (skilled-risk evaluator revamp).
    base_sortino = baseline.get("sortino_ratio")
    curr_sortino = current_metrics.get("sortino_ratio")
    sortino_comparable = (
        base_sortino is not None and curr_sortino is not None and base_sortino > 0
    )
    if sortino_comparable:
        drop_pct = (base_sortino - curr_sortino) / abs(base_sortino)
        details["sortino_drop_pct"] = drop_pct
        if drop_pct > sortino_threshold:
            regression_detected = True
            logger.warning(
                "Regression: sortino dropped %.1f%% (%.4f -> %.4f), threshold=%.1f%%",
                drop_pct * 100, base_sortino, curr_sortino, sortino_threshold * 100,
            )
    elif base_sortino is None:
        # Older baseline with no Sortino → not comparable on the primary
        # risk-adjusted gate. Do NOT fall back to firing on Sharpe.
        details["sortino_not_comparable"] = True
        logger.info(
            "Baseline has no sortino_ratio — primary risk-adjusted gate not "
            "comparable; relying on accuracy gate only this run.",
        )

    # Sharpe check — OBSERVABILITY ONLY (no longer a rollback trigger). Persist
    # the drop for continuity / dashboards; never sets regression_detected.
    base_sharpe = baseline.get("sharpe_ratio")
    curr_sharpe = current_metrics.get("sharpe_ratio")
    if base_sharpe is not None and curr_sharpe is not None and base_sharpe > 0:
        sharpe_drop_pct = (base_sharpe - curr_sharpe) / abs(base_sharpe)
        details["sharpe_drop_pct"] = sharpe_drop_pct
        if sharpe_drop_pct > sharpe_threshold:
            logger.info(
                "Sharpe dropped %.1f%% (%.4f -> %.4f) — observability only, "
                "NOT a rollback trigger (primary gate is Sortino).",
                sharpe_drop_pct * 100, base_sharpe, curr_sharpe,
            )

    # ── Guard 2: min-sample / degraded-week ──────────────────────────────────
    # A low-statistical-power week may *detect* a regression but must NOT
    # auto-rollback live configs. (2026-05-16: Mode-2 n=55 trades / signal
    # n=42 recovery week.)
    n_trades = current_metrics.get("total_trades")
    n_signals = current_metrics.get("n_signals")
    low_power_reasons = []
    if n_trades is not None and n_trades < min_trades:
        low_power_reasons.append(
            f"total_trades={n_trades} < min_trades_for_rollback={min_trades}"
        )
    if n_signals is not None and n_signals < min_signals:
        low_power_reasons.append(
            f"n_signals={n_signals} < min_signals_for_rollback={min_signals}"
        )
    rollback_suppressed_low_power = bool(low_power_reasons)
    if rollback_suppressed_low_power:
        details["low_power"] = low_power_reasons
        details["guard"] = "min_sample"

    result = {
        "checked": True,
        "regression_detected": regression_detected,
        "rollback_triggered": False,
        "details": details,
        "baseline": baseline,
        "current": current_metrics,
    }

    if regression_detected and rollback_suppressed_low_power:
        logger.warning(
            "Regression DETECTED but rollback SUPPRESSED — low statistical "
            "power week (%s). Configs left in place.",
            "; ".join(low_power_reasons),
        )

    if regression_detected and not rollback_suppressed_low_power:
        logger.warning("Regression detected — triggering auto-rollback")
        rollback_results = rollback_all(bucket)
        result["rollback_triggered"] = True
        result["rollback_results"] = rollback_results

        # Emit structured rollback audit artifact so operators can answer
        # "what did the system try to land before regression vetoed it?"
        # without S3-timestamp-archaeology. Closes the silent-rollback
        # surface from the 2026-05-09 forensic.
        audit_run_date = run_date or str(date.today())
        audit_key = write_rollback_audit(
            bucket=bucket,
            run_date=audit_run_date,
            regression_check_result=result,
            rollback_results=rollback_results,
        )
        result["rollback_audit_key"] = audit_key or None

    return result
