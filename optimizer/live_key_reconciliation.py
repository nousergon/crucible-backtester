"""live_key_reconciliation.py — post-optimizer live-key freshness check
(config#2332).

**The gap this retires:** the config#2054 orphaned-write class (the
assembler cutover silently dropped 3 of 4 config-promotion writes for ~6
weeks) had NO end-to-end detection — ``ARTIFACT_REGISTRY.yaml``'s
``config_*`` rows delegate liveness to ``optimizer_run_manifest``
(``evaluate.py::write_optimizer_run_manifest``), which proves the optimizer
STAGE ran, not that the live key actually landed on S3. A config type
absent from ``optimizer.assembler.DEFAULT_PRECEDENCE`` (or any future
config type added to ``apply_audit.LOOPS`` without a matching live-write
path) passes every existing guard as long as the stage itself didn't raise.

**This module closes the loop for the one signal that can actually tell:**
``apply_audit`` already classifies each loop's outcome every run (config
#1841 / config#2331). For every loop whose outcome THIS RUN is
``"promoted"``, HEAD the live ``config/{loop}.json`` key and compare its
``LastModified`` against the run's start time. A promoted-per-audit config
whose live key predates run start means the audit claimed success but the
write never happened — the exact orphaned-write shape from #2054, now
caught the same run it recurs in (not 6 weeks later).

**Absence-is-correct semantics (per the issue's own gotcha):** only loops
whose outcome is ``"promoted"`` THIS run are reconciled. A loop that has
never promoted (e.g. ``scoring_weights`` / ``predictor_params`` before
their first successful apply) legitimately has no live key yet — that is
not a defect and must not page. Likewise a loop classified ``blocked`` /
``insufficient_data`` / ``disabled`` / ``error`` this run is not expected
to have written this run, so it's skipped.

**Alerting:** ``severity="critical"`` via ``ops_alerts.publish_ops_alert``
— the only severities that page (``krepis.alerts.SEVERITY_PUSH ==
{"error", "critical"}``); "critical" is used here (rather than "error",
used elsewhere in this codebase for degraded-but-recoverable conditions)
because a promoted-per-audit config whose live key didn't move means the
executor may be trading on stale params RIGHT NOW — the same class of
defect config#2331 made fail-loud at the write site. This is the
detection-side complement: it catches any OTHER path (present or future)
that can produce the same orphaned-write shape, not just the assembler
cutover write config#2331 hardened.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# loop name (apply_audit.LOOPS key) -> live S3 key. Kept as an explicit map
# (not string-templated off the loop name) so a loop whose live key diverges
# from the `config/{loop}.json` convention doesn't silently mis-check —
# adding a new loop requires a deliberate entry here.
LIVE_KEYS: dict[str, str] = {
    "scoring_weights": "config/scoring_weights.json",
    "executor_params": "config/executor_params.json",
    "predictor_params": "config/predictor_params.json",
    "research_params": "config/research_params.json",
}


@dataclass
class ReconciliationFinding:
    """One loop whose apply_audit outcome claims "promoted" this run but
    whose live S3 key's LastModified predates the run start — i.e. the
    write that should have happened this run did not."""

    loop: str
    live_key: str
    last_modified: str | None  # ISO-8601, or None if the key is missing entirely
    run_start: str  # ISO-8601
    reason: str  # "missing" or "stale"

    def to_dict(self) -> dict:
        return asdict(self)


def reconcile_promoted_live_keys(
    bucket: str,
    audit: dict,
    run_start: datetime,
    s3_client=None,
) -> list[ReconciliationFinding]:
    """Check every loop whose ``audit["loops"][loop]["outcome"] ==
    "promoted"`` this run actually has a live key with ``LastModified >=
    run_start``.

    Args:
        bucket: S3 bucket name (same bucket apply_audit/assembler write to).
        audit: the ``build_audit()``-shaped dict from this run (schema v1;
            ``{"loops": {loop: {"outcome": ..., ...}, ...}}``).
        run_start: the run's start timestamp (timezone-aware UTC
            recommended — a naive datetime is treated as UTC).
        s3_client: optional boto3 client (test injection).

    Returns:
        A list of :class:`ReconciliationFinding`, one per loop that claims
        "promoted" but whose live key is missing or stale. Empty when every
        promoted loop's live key is fresh (the expected case every run).

    Only loops with outcome "promoted" THIS run are checked — legitimately
    never-promoted config types (their live key has never existed) and
    loops that were blocked/disabled/erroring this run are not expected to
    have written, so they're skipped rather than false-paged. Any
    non-404 ``head_object`` error (permission, network) RAISES — the
    reconciliation could not actually be performed, and a swallowed
    "couldn't check" must not read as "checked, all clear" (fail-loud,
    mirrors ``pipeline_common._first_missing_artifact``).
    """
    s3 = s3_client or boto3.client("s3")
    if run_start.tzinfo is None:
        run_start = run_start.replace(tzinfo=timezone.utc)

    loops = (audit or {}).get("loops", {}) or {}
    findings: list[ReconciliationFinding] = []

    for loop, record in loops.items():
        if not isinstance(record, dict) or record.get("outcome") != "promoted":
            continue
        live_key = LIVE_KEYS.get(loop)
        if live_key is None:
            logger.warning(
                "live_key_reconciliation: loop=%s classified promoted but "
                "has no entry in LIVE_KEYS — reconciliation gap, add the "
                "live key mapping (classifier/registry drift)",
                loop,
            )
            continue

        try:
            head = s3.head_object(Bucket=bucket, Key=live_key)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                findings.append(ReconciliationFinding(
                    loop=loop,
                    live_key=live_key,
                    last_modified=None,
                    run_start=run_start.isoformat(),
                    reason="missing",
                ))
                continue
            # Transient / permission error: we could not actually verify
            # this loop's live key — fail loud rather than silently
            # skipping a check that might have caught a real orphaned
            # write.
            raise

        last_modified = head["LastModified"]
        if last_modified.tzinfo is None:
            last_modified = last_modified.replace(tzinfo=timezone.utc)
        if last_modified < run_start:
            findings.append(ReconciliationFinding(
                loop=loop,
                live_key=live_key,
                last_modified=last_modified.isoformat(),
                run_start=run_start.isoformat(),
                reason="stale",
            ))

    return findings


def _format_alert_message(
    findings: list[ReconciliationFinding], run_date: str,
) -> str:
    lines = [
        f"Post-optimizer live-key reconciliation FAILED for run {run_date}: "
        f"{len(findings)} config type(s) claim outcome=promoted in "
        f"apply_audit but the live S3 key was NOT refreshed this run "
        f"(config#2054 orphaned-write class recurrence).",
    ]
    for f in findings:
        if f.reason == "missing":
            lines.append(
                f"  - {f.loop}: live key s3://.../{f.live_key} is MISSING "
                f"entirely despite outcome=promoted.",
            )
        else:
            lines.append(
                f"  - {f.loop}: live key s3://.../{f.live_key} last_modified="
                f"{f.last_modified} predates run_start={f.run_start}.",
            )
    lines.append(
        "The executor may be trading on stale params for the affected "
        "config type(s). Investigate the apply()/assembler write path for "
        "each loop listed above.",
    )
    return "\n".join(lines)


def run_reconciliation(
    bucket: str,
    audit: dict,
    run_start: datetime,
    run_date: str,
    *,
    s3_client=None,
    publish_alert=None,
) -> list[ReconciliationFinding]:
    """Run the reconciliation and page on any finding. Called once per run,
    after ``emit_apply_audit`` has produced this run's audit dict.

    Args:
        publish_alert: injected for testing; defaults to
            ``ops_alerts.publish_ops_alert``. Signature matches
            ``publish_ops_alert(message, *, severity, source, dedup_key=None,
            dedup_window_min=None)``.

    Returns the findings list (empty when clean). Does not raise on findings
    — paging is the intended signal, and the caller (the weekly evaluate
    run) must not abort the rest of the pipeline over a reconciliation
    alert. A failure to even PERFORM the check (e.g. head_object raising a
    non-404 error) does propagate, per ``reconcile_promoted_live_keys``'s
    fail-loud contract.
    """
    findings = reconcile_promoted_live_keys(
        bucket, audit, run_start, s3_client=s3_client,
    )
    if not findings:
        logger.info(
            "live_key_reconciliation: clean — every promoted loop's live "
            "key is fresh as of run_start=%s", run_start.isoformat(),
        )
        return findings

    logger.error(
        "live_key_reconciliation: %d promoted-per-audit config(s) have a "
        "stale/missing live key for run %s: %s",
        len(findings), run_date, [f.to_dict() for f in findings],
    )

    if publish_alert is None:
        from ops_alerts import publish_ops_alert as publish_alert

    message = _format_alert_message(findings, run_date)
    try:
        publish_alert(
            message,
            severity="critical",
            source="alpha-engine-backtester/optimizer/live_key_reconciliation.py",
            dedup_key=f"live_key_reconciliation_{run_date}_"
                       f"{'_'.join(sorted(f.loop for f in findings))}",
            dedup_window_min=1440,
        )
    except Exception as e:  # noqa: BLE001 — the page attempt itself must not
        # crash the pipeline; the ERROR log above is the durable trace even
        # if the alert channel is down.
        logger.error(
            "live_key_reconciliation: alert publish failed (findings are "
            "still logged above at ERROR): %s", e,
        )

    return findings
