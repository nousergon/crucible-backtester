"""
lambda_health/handler.py — Daily predictor health check Lambda.

Runs lightweight model health monitoring (Phases 2a, 2b, 5) on weekdays
after predictor inference. Catches model degradation within 1-2 days
instead of waiting for the Saturday backtester.

Checks:
  - Production health: 30d rolling IC, hit rate, regime IC, mode collapse
  - Calibration validation: per-bin confidence vs actual hit rate
  - Retrain alert: evaluates 5 trigger conditions, sends email if fired

Feature drift (Phase 3) is NOT recomputed — it needs ArcticDB which is
too heavy for this Lambda. Instead, reads the last weekly feature_drift.json
from S3 as input to the retrain alert evaluator.

Lambda configuration:
  Memory: 512 MB  |  Timeout: 120s  |  Runtime: container (python:3.12)

Environment variables:
  S3_BUCKET          — default: alpha-engine-research
  EMAIL_SENDER       — from-address for retrain alerts
  EMAIL_RECIPIENTS   — comma-separated recipient list
  GMAIL_APP_PASSWORD — Gmail App Password (enables SMTP path)
  AWS_REGION         — SES fallback region (default: us-east-1)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

# Ensure the project root is on sys.path so sibling modules can be
# imported below. Cheap; safe at module-top.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Structured logging + flow-doctor singleton via alpha-engine-lib (shared
# pattern across all 5 entrypoints; see executor/main.py for reference).
# When FLOW_DOCTOR_ENABLED=1 the root logger gets a handler that
# captures ERROR+ records — every log.error() call in the handler
# below is automatically routed to flow-doctor without explicit
# fd.report() plumbing.
#
# Path resolution: LAMBDA_TASK_ROOT (=/var/task in the Lambda image,
# where lambda_health/Dockerfile COPYs flow-doctor.yaml) takes
# precedence; falls back to two-dirs-up from this file for local dev
# (lambda_health/handler.py → repo root). Mirrors alpha-engine-research
# / alpha-engine-predictor pattern.
#
# exclude_patterns starts empty by deliberate convention.
#
# Secrets load via nousergon_lib.secrets.get_secret() at use-site
# (per-process cached); flow-doctor.yaml only references EMAIL_SENDER /
# EMAIL_RECIPIENTS / GMAIL_APP_PASSWORD, all populated by Lambda's
# `--environment` block BEFORE the Python interpreter starts.
from nousergon_lib.logging import setup_logging, monitor_handler
from nousergon_lib.secrets import get_secret
_FLOW_DOCTOR_EXCLUDE_PATTERNS: list[str] = []
_FLOW_DOCTOR_YAML = os.path.join(
    os.environ.get(
        "LAMBDA_TASK_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ),
    "flow-doctor.yaml",
)
setup_logging(
    "lambda_health",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
)

log = logging.getLogger(__name__)

# Expensive init is deferred to the first handler invocation to keep
# Lambda's cold-start init phase under the 10-second hard timeout.
# Idempotent via the `_init_done` flag.
#
# Post-L2998-PR-9c (2026-05-14): secrets now load via
# nousergon_lib.secrets.get_secret() at use-site (lazy, per-process
# cached). No bulk SSM round-trip on cold-start. Stub retained to
# preserve the handler() call shape + idempotency flag in case future
# deferred init is needed.
_init_done = False


def _ensure_init() -> None:
    """Run deferred init once, on the first handler invocation."""
    global _init_done
    if _init_done:
        return
    _init_done = True


@monitor_handler
def handler(event: dict, context) -> dict:
    """Entry point. Runs the health report, then flushes cost telemetry.

    The `finally` is structural, not a prediction (alpha-engine-config-I7423).
    `krepis.cost_sink.S3JsonlCostSink` buffers to 200 records per
    `(date, callsite_id)` group and otherwise relies on an `atexit` hook —
    and **an AWS Lambda container is FROZEN between invocations, not exited,
    so `atexit` never runs.** A handler finishing below the threshold writes
    nothing at all.

    This handler makes no model call today. The flush is here anyway, for the
    same reason its two siblings in this repo and every handler in
    crucible-research and crucible-evaluator have it: the guard that keeps
    `lambda_concordance` honest is a glob over `lambda_*/handler.py`, and a
    guard that has to correctly predict which handlers reach a model is the
    guard that misses the next one. `flush_default_sink` returns 0 when no
    sink is configured and never raises, so the cost of being wrong in this
    direction is one function call.
    """
    try:
        return _run(event, context)
    finally:
        try:
            from krepis.cost_sink import flush_default_sink
            _n = flush_default_sink()
            if _n:
                log.info("[lambda_health] cost sink flushed: %d object(s)", _n)
        except ImportError as exc:
            # Loud, not silent: the image's krepis pin predates the function.
            # Any records held would be lost, and AggregateCosts' fan-in
            # coverage check is what would say so.
            log.error("cost-sink flush unavailable — records lost: %s", exc)


def _run(event: dict, context) -> dict:
    """
    AWS Lambda entry point.

    event may contain:
        date     (str)  : Override date YYYY-MM-DD (default: today UTC)
        dry_run  (bool) : If True, skip S3 writes and email (for canary tests)
    """
    # Run the deferred SSM secrets fetch on the first invocation. Warm
    # containers pay zero cost via the _init_done flag.
    _ensure_init()

    t0 = time.time()
    bucket = os.environ.get("S3_BUCKET", "alpha-engine-research")
    run_date = event.get("date")
    dry_run = event.get("dry_run", False)

    # Preflight: AWS_REGION + S3 bucket reachable. Fail fast so the
    # Lambda surfaces a 500 instead of a misleading 200 when the bucket
    # is unreachable mid-run. Mirrors the handler's own belt-and-braces
    # error handling below (phase_errors → 500).
    from preflight import BacktesterPreflight
    BacktesterPreflight(bucket=bucket, mode="lambda_health").run()

    log.info("Predictor health check starting: bucket=%s date=%s dry_run=%s", bucket, run_date, dry_run)

    # Track critical errors across phases. If any phase fails, return 500
    # so the Lambda caller (EventBridge scheduler / monitoring) sees the
    # failure instead of a misleading 200 OK. The previous behavior
    # returned 200 regardless of per-phase exceptions, which meant
    # degradations and calibration drift went unnoticed.
    phase_errors: list[str] = []

    # ── Download research.db from S3 ─────────────────────────────────────
    # alpha-engine-config-I8704 — the canary does NOT download it. Under
    # dry_run both compute phases short-circuit and the retrain-alert email is
    # skipped, so nothing reads the database: the download was 356 MB of pure
    # waste per deploy. Worse, it POISONED the container it warmed. `/tmp` is
    # 512 MB and research.db is 356 MB, so a warm container still holding the
    # canary's copy has no room for a second, and the next REAL invocation died
    # with `[Errno 28] No space left on device` — measured 2026-08-26T17:55Z,
    # four minutes after a canary at 17:51Z, on the first live run of the
    # freshly un-paused rule.
    db_path = None
    if dry_run:
        log.info(
            "[dry_run] Skipping research.db download — no phase reads it, and "
            "leaving a 356 MB file in a warm /tmp breaks the next real run "
            "(alpha-engine-config-I8704)"
        )
    else:
        db_path = _download_research_db(bucket)
        if not db_path:
            return _response(500, "Failed to download research.db from S3")

    results = {}

    # ── Phase 2a: Production health ──────────────────────────────────────
    production_health = None
    try:
        from analysis.production_health import compute_production_health
        if dry_run:
            log.info("[dry_run] Skipping production health S3 write")
            production_health = {"status": "dry_run"}
        else:
            production_health = compute_production_health(db_path, bucket, run_date)
        results["production_health"] = production_health
        log.info("Production health: %s", _summarize(production_health))
    except Exception as exc:
        log.error("Production health failed: %s", exc, exc_info=True)
        results["production_health"] = {"status": "error", "error": str(exc)}
        phase_errors.append(f"production_health: {exc}")

    # ── Phase 2b: Calibration validation ─────────────────────────────────
    calibration = None
    try:
        from analysis.production_health import compute_calibration_validation
        if dry_run:
            log.info("[dry_run] Skipping calibration S3 write")
            calibration = {"status": "dry_run"}
        else:
            calibration = compute_calibration_validation(db_path, bucket, run_date)
        results["calibration"] = calibration
        log.info("Calibration: %s", _summarize(calibration))
    except Exception as exc:
        log.error("Calibration validation failed: %s", exc, exc_info=True)
        results["calibration"] = {"status": "error", "error": str(exc)}
        phase_errors.append(f"calibration: {exc}")

    # ── Load last weekly feature drift (read-only, not recomputed) ───────
    feature_drift = _load_last_feature_drift(bucket)

    # ── Phase 5: Retrain alert evaluation ────────────────────────────────
    try:
        from analysis.retrain_alert import evaluate_retrain_triggers, send_retrain_alert
        alert = evaluate_retrain_triggers(production_health, feature_drift, calibration)
        results["retrain_alert"] = {
            "triggered": alert.get("triggered", False),
            "n_triggers": alert.get("n_triggers", 0),
            "summary": alert.get("summary", ""),
        }

        if alert.get("triggered") and not dry_run:
            email_config = _build_email_config()
            send_result = send_retrain_alert(alert, email_config, bucket)
            results["retrain_alert"]["email_sent"] = send_result.get("sent", False)
        elif dry_run:
            log.info("[dry_run] Skipping retrain alert email")
    except Exception as exc:
        log.error("Retrain alert evaluation failed: %s", exc, exc_info=True)
        results["retrain_alert"] = {"status": "error", "error": str(exc)}
        phase_errors.append(f"retrain_alert: {exc}")

    # ── Write health status to S3 ────────────────────────────────────────
    elapsed = time.time() - t0
    status = "ok"
    warnings = []

    if phase_errors:
        status = "error"
        warnings.extend(phase_errors)
    if production_health and production_health.get("degradation_flag"):
        status = "degraded"
        warnings.append("IC degradation detected")
    if production_health and production_health.get("mode_collapse_flag"):
        status = "degraded"
        warnings.append("Mode collapse detected")
    if results.get("retrain_alert", {}).get("triggered"):
        status = "degraded"

    # alpha-engine-config-I9708 — a REAL run can grade nothing too. When both
    # compute phases return `skipped` (too few post-cutover outcomes, or the
    # post-cutover IC blackout, or no calibration window yet), no metric was
    # evaluated, `degradation_flag` is absent, and every check above fell
    # through to the initial "ok". That is the dry-run defect below on the
    # live path: a health check that measured nothing reporting healthy. The
    # warning makes the health stamp derive `degraded`; the HTTP status stays
    # 200 because nothing failed.
    def _skipped(phase_result) -> bool:
        return isinstance(phase_result, dict) and phase_result.get("status") == "skipped"

    if status == "ok" and _skipped(production_health) and _skipped(calibration):
        status = "insufficient_data"
        warnings.append(
            "insufficient_data: NO metric was evaluated — production health "
            f"({production_health.get('reason')}) and calibration "
            f"({calibration.get('reason')}) were both skipped, so this run is "
            "not a health verdict."
        )

    # alpha-engine-config-I8704 — a dry run GRADED NOTHING, and must never say
    # `ok`. Under dry_run both compute phases short-circuit to
    # {"status": "dry_run"}, so `degradation_flag` is absent from
    # `production_health` and every check above falls through to the initial
    # "ok" — a canary that evaluated no metric reported
    # `status=ok warnings=[]`, which is champion-challenger-policy §7.2's
    # "well-formed artifact containing nothing".
    #
    # This is not hypothetical: the deploy canary
    # (infrastructure/deploy_health.sh, `{"dry_run": true}`) is currently the
    # ONLY thing invoking this Lambda — the scheduled EventBridge rule
    # `alpha-engine-predictor-health-check` is DISABLED under the 2026-08-07
    # automation pause. So every log line anyone has seen from this function
    # since then said `ok` while `degradation_flag` sat true in the artifact
    # the weekly backtester writes.
    #
    # The canary asserts on the Lambda's statusCode (200), never on this
    # field, so making it honest cannot fail a deploy.
    if dry_run:
        status = "dry_run"
        warnings.append(
            "dry_run: NO metric was evaluated — production health, calibration "
            "and the retrain triggers were all skipped. This is a canary "
            "invocation, not a health verdict."
        )

    if not dry_run:
        try:
            from nousergon_lib.health import Deliverable, write_health

            failed = status in ("failed", "error")
            write_health(
                module_name="predictor_health_check",
                deliverables=[
                    Deliverable(
                        name="health_check",
                        required=True,
                        produced=not failed,
                    ),
                ],
                run_date=run_date,
                duration_seconds=round(elapsed, 1),
                summary=results,
                warnings=warnings or None,
                error="; ".join(phase_errors) if failed and phase_errors else (
                    status if failed else None
                ),
                bucket=bucket,
            )
        except Exception as exc:
            log.error("Failed to write health status: %s", exc, exc_info=True)
            phase_errors.append(f"write_health: {exc}")

    log.info("Health check complete in %.1fs: status=%s warnings=%s", elapsed, status, warnings)

    # Return 500 when any critical phase failed, so callers see the
    # failure. Returning 200 here is what hid the 2026-04-10 predictor
    # degradation for 3 hours — Lambda dashboards showed green while the
    # underlying checks were throwing exceptions.
    if phase_errors:
        return _response(500, {
            "status": "error",
            "duration_seconds": round(elapsed, 1),
            "phase_errors": phase_errors,
            "results": results,
        })

    return _response(200, {
        "status": status,
        "duration_seconds": round(elapsed, 1),
        "results": results,
    })


#: Where research.db lands on the Lambda's ephemeral volume. Named so a test
#: can redirect it without patching `os.path` (alpha-engine-config-I8704).
_RESEARCH_DB_PATH = "/tmp/research.db"


def _download_research_db(bucket: str, db_path: str | None = None) -> str | None:
    """Download research.db from S3 to /tmp.

    alpha-engine-config-I8704: a warm container must never need 2x the file on
    disk. Lambda's `/tmp` is 512 MB and research.db was 356 MB on 2026-08-26 and
    grows weekly, so a stale copy from a previous invocation leaves no room for
    boto3's managed download and the invocation dies with
    `[Errno 28] No space left on device`. Removing the old file first is the
    difference between "warm container" and "warm container that cannot work".
    """
    import boto3
    db_path = db_path or _RESEARCH_DB_PATH
    try:
        if os.path.exists(db_path):
            stale_mb = os.path.getsize(db_path) / (1024 * 1024)
            os.remove(db_path)
            log.info(
                "Removed a stale /tmp/research.db (%.1f MB) from a warm container "
                "before re-downloading — /tmp cannot hold two copies "
                "(alpha-engine-config-I8704)", stale_mb,
            )
        s3 = boto3.client("s3")
        s3.download_file(bucket, "research.db", db_path)
        size_mb = os.path.getsize(db_path) / (1024 * 1024)
        log.info("Downloaded research.db (%.1f MB) to %s", size_mb, db_path)
        return db_path
    except Exception as exc:
        log.error("Failed to download research.db: %s", exc)
        return None


def _load_last_feature_drift(bucket: str) -> dict | None:
    """Read the most recent feature_drift.json from S3 (weekly output).

    Distinguishes NoSuchKey (expected — feature_drift.json hasn't been
    written yet or is produced elsewhere) from real errors (S3 permissions,
    network, corrupt JSON) so the latter surface via flow-doctor instead
    of silently returning None. Still returns None in both cases to
    preserve the retrain-alert caller's None-handling contract; the
    visibility improvement is the log level.
    """
    import boto3
    from botocore.exceptions import ClientError

    key = "predictor/metrics/feature_drift.json"
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            log.info(
                "No feature_drift.json at s3://%s/%s — retrain alert will "
                "run without drift input",
                bucket, key,
            )
            return None
        log.error(
            "Failed to load feature_drift from s3://%s/%s (ClientError): %s",
            bucket, key, e, exc_info=True,
        )
        return None
    except Exception as e:
        log.error(
            "Failed to parse feature_drift from s3://%s/%s: %s",
            bucket, key, e, exc_info=True,
        )
        return None


def _build_email_config() -> dict:
    """Build email config from environment variables."""
    recipients_str = get_secret("EMAIL_RECIPIENTS", required=False, default="") or ""
    recipients = [r.strip() for r in recipients_str.split(",") if r.strip()]
    return {
        "email_sender": get_secret("EMAIL_SENDER", required=False, default="") or "",
        "email_recipients": recipients,
        "aws_region": os.environ.get("AWS_REGION", "us-east-1"),
    }


def _summarize(result: dict | None) -> str:
    """One-line summary for logging."""
    if not result:
        return "None"
    if "status" in result and result["status"] in ("skipped", "error", "dry_run"):
        return f"status={result['status']}"
    parts = []
    for key in ("rolling_30d_ic", "rolling_30d_hit_rate", "degradation_flag",
                "mode_collapse_flag", "overall_ece", "calibration_quality"):
        if key in result:
            parts.append(f"{key}={result[key]}")
    return " ".join(parts) if parts else str(result.get("status", "ok"))


def _response(code: int, body) -> dict:
    """Standard Lambda response."""
    return {
        "statusCode": code,
        "body": json.dumps(body, default=str) if isinstance(body, dict) else body,
    }
