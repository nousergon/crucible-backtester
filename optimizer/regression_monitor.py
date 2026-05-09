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
"""

import json
import logging
from datetime import date, datetime, timezone
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
DEFAULT_SHARPE_DROP_PCT = 0.20     # rollback if Sharpe drops > 20%


def extract_metrics(portfolio_stats: dict | None, signal_quality: dict | None) -> dict:
    """Extract the metrics used for regression comparison."""
    metrics = {}

    if portfolio_stats and isinstance(portfolio_stats, dict):
        for key in ("sharpe_ratio", "total_alpha", "max_drawdown", "win_rate"):
            if key in portfolio_stats:
                metrics[key] = portfolio_stats[key]

    if signal_quality and isinstance(signal_quality, dict):
        overall = signal_quality.get("overall", {})
        for key in ("accuracy_10d", "accuracy_30d"):
            if key in overall:
                metrics[key] = overall[key]

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
    key = f"{S3_METRICS_PREFIX}{run_date}.json"
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps(payload, indent=2),
        ContentType="application/json",
    )
    logger.info("Rolling metrics saved to s3://%s/%s", bucket, key)


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
                "sharpe_drop_threshold_pct": DEFAULT_SHARPE_DROP_PCT,
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
            "details": {"accuracy_drop": float|None, "sharpe_drop_pct": float|None},
            "baseline": dict|None,
            "current": dict,
        }
    """
    baseline = _load_baseline(bucket)
    if baseline is None:
        logger.info("No promotion baseline found — skipping regression check")
        return {"checked": False, "reason": "no baseline"}

    reg_config = (config or {}).get("regression_monitor", {})
    acc_threshold = reg_config.get("accuracy_drop_threshold_pp", DEFAULT_ACCURACY_DROP_PP)
    sharpe_threshold = reg_config.get("sharpe_drop_threshold_pct", DEFAULT_SHARPE_DROP_PCT)

    details = {}
    regression_detected = False

    # Accuracy check (10d)
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

    # Sharpe check
    base_sharpe = baseline.get("sharpe_ratio")
    curr_sharpe = current_metrics.get("sharpe_ratio")
    if base_sharpe is not None and curr_sharpe is not None and base_sharpe > 0:
        drop_pct = (base_sharpe - curr_sharpe) / abs(base_sharpe)
        details["sharpe_drop_pct"] = drop_pct
        if drop_pct > sharpe_threshold:
            regression_detected = True
            logger.warning(
                "Regression: sharpe dropped %.1f%% (%.4f -> %.4f), threshold=%.1f%%",
                drop_pct * 100, base_sharpe, curr_sharpe, sharpe_threshold * 100,
            )

    result = {
        "checked": True,
        "regression_detected": regression_detected,
        "rollback_triggered": False,
        "details": details,
        "baseline": baseline,
        "current": current_metrics,
    }

    if regression_detected:
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
