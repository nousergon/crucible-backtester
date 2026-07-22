"""apply_audit.py — per-run outcome record for the four auto-apply loops
(config#1841).

**The defect this retires:** a loop that has silently never promoted is
indistinguishable from a broken one. Live S3 showed ``config/
scoring_weights.json`` and ``config/predictor_params.json`` had NEVER been
written and ``config/research_params.json`` had been stale for 2+ months —
and nothing graded that silence. Every Saturday evaluate run now emits ONE
audit artifact recording, for each auto-apply loop, whether it promoted, was
blocked (and by exactly which guardrail), was data-starved, was disabled
(shadow/freeze/flag), or errored — regardless of outcome.

**FROZEN cross-repo schema v1** (``nousergon_lib.contracts`` ``apply_audit``,
lifted from the repo-local copy on the second-adoption signal, config#1861):
the crucible-evaluator consumer is built in parallel against exactly this shape
(RED when a loop has been blocked N consecutive weeks with no human ack).
Evolution is additive-only; renames/removals require a schema_version bump
coordinated with the consumer.

**S3 layout:** ``config/apply_audit/{date}.json`` (dated, trading-day keyed)
plus a ``config/apply_audit/latest.json`` mirror. The write is gated on the
same ``args.upload``/``--freeze`` gate as sibling artifacts; the audit is
still BUILT and logged on local runs (emission of the record is
unconditional; only the S3 persistence is gated).

**Carry-forward counter:** ``consecutive_blocked_weeks`` is read from the
PRIOR ``latest.json`` and incremented on ``blocked`` / reset on ``promoted``
or ``insufficient_data`` / carried unchanged on ``error`` and ``disabled``
(an error or disabled week is evidence of neither blocking nor unblocking).
This is a carry-forward read-then-write of a SINGLE-OWNER artifact — only
this module (one Saturday evaluate run at a time) ever writes the
``config/apply_audit/`` prefix, so the rebuild-writer-clobbers-other-owner
bug class does not apply; the read-modify-write is safe by ownership, not by
locking.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
S3_AUDIT_PREFIX = "config/apply_audit"

# The four auto-apply loops (audit key → opt_results key in evaluate.py).
LOOPS: dict[str, str] = {
    "scoring_weights": "weight_result",
    "executor_params": "executor_rec",
    "predictor_params": "veto_result",
    "research_params": "research_params",
}

OUTCOMES = ("promoted", "blocked", "insufficient_data", "error", "disabled")

# ── Stable guardrail slugs ───────────────────────────────────────────────────
# Derived from the REAL guardrail code (not invented): each slug names the
# config knob / gate that refused the promotion. Enumerated (and documented)
# in the ``apply_audit`` schema in nousergon_lib.contracts — additions are
# additive schema evolution; renames are breaking.
BLOCKED_BY_SLUGS = (
    # weight_optimizer.apply_weights
    "oos_degradation",            # _validate_oos failed (degradation >= 20%)
    "confidence_below_medium",    # n_samples below confidence_medium/low bar
    "max_single_change",          # largest |Δweight| > max_single_change
    "min_meaningful_change",      # all changes below min_meaningful_change
                                  # (also research_optimizer status
                                  # "no_improvement": all param deltas < 0.001)
    "significance_floor",         # enforce_significance blocked (config#1426)
    # analysis/veto_analysis
    "min_lift_over_base_rate",    # status insufficient_lift (5pp point-lift gate)
    "precision_ci_below_base_rate",  # status insufficient_confidence (Wilson LB)
    "min_threshold_change",       # recommended too close to current threshold
    # optimizer/executor_optimizer.recommend
    "alpha_floor",                # status alpha_below_floor
    "min_trades_to_promote",      # status insufficient_trades
    "negative_rank_metric",       # status negative_sortino / negative_sharpe /
                                  # negative_alpha_vs_ew_high_vol
    "baseline_magnitude_floor",   # status baseline_insignificant
    "min_improvement",            # status no_improvement (executor)
    "min_psr",                    # status insufficient_psr_confidence
    # assembler cutover (executor_params live key)
    "assembler_skip",             # cutover on; assembler saw no promotable artifact
    # safety valve — an apply() rejection reason this classifier doesn't know.
    # A record carrying this slug is a classifier gap to fix, surfaced loudly
    # instead of mis-binned.
    "unclassified_guardrail",
)

# Pre-apply status → (outcome, blocked_by) maps, per loop. Statuses that
# terminate before any recommendation exists are honest data starvation;
# statuses where a recommendation existed but a NAMED gate refused are
# guardrail blocks.
_WEIGHT_STATUS = {
    "insufficient_data": ("insufficient_data", None),
    "no_subscores": ("insufficient_data", None),
}
_VETO_STATUS = {
    "insufficient_data": ("insufficient_data", None),
    "no_predictions": ("insufficient_data", None),
    "no_down_predictions": ("insufficient_data", None),
    "insufficient_vetoes": ("insufficient_data", None),
    "insufficient_lift": ("blocked", ["min_lift_over_base_rate"]),
    "insufficient_confidence": ("blocked", ["precision_ci_below_base_rate"]),
}
_RESEARCH_STATUS = {
    "insufficient_data": ("insufficient_data", None),
    "no_boost_data": ("insufficient_data", None),
    "no_improvement": ("blocked", ["min_meaningful_change"]),
    # boost-correlation signal retired (alpha-engine-config-I3246): a
    # by-design-off loop, not a data-starved or guardrail-blocked one — maps
    # to "disabled" like the existing freeze/shadow-mode paths so the
    # consecutive_blocked_weeks streak neither increments nor resets on it.
    "retired": ("disabled", None),
}
_EXECUTOR_STATUS = {
    "insufficient_data": ("insufficient_data", None),
    "no_params": ("insufficient_data", None),
    "degraded": ("insufficient_data", None),
    "alpha_below_floor": ("blocked", ["alpha_floor"]),
    "insufficient_trades": ("blocked", ["min_trades_to_promote"]),
    "negative_sortino": ("blocked", ["negative_rank_metric"]),
    "negative_sharpe": ("blocked", ["negative_rank_metric"]),
    "negative_alpha_vs_ew_high_vol": ("blocked", ["negative_rank_metric"]),
    "baseline_insignificant": ("blocked", ["baseline_magnitude_floor"]),
    "no_improvement": ("blocked", ["min_improvement"]),
    "insufficient_psr_confidence": ("blocked", ["min_psr"]),
}
_STATUS_MAPS = {
    "scoring_weights": _WEIGHT_STATUS,
    "predictor_params": _VETO_STATUS,
    "research_params": _RESEARCH_STATUS,
    "executor_params": _EXECUTOR_STATUS,
}

# proposed/current extractors per loop (small dicts for the audit record).
_PROPOSED_KEYS = {
    "scoring_weights": ("suggested_weights", "current_weights"),
    "executor_params": ("recommended_params", "baseline_params"),
    "research_params": ("recommended_params", "current_params"),
}


def _veto_proposed_current(result: dict) -> tuple[dict | None, dict | None]:
    rec = result.get("recommended_threshold")
    cur = result.get("current_threshold")
    proposed = {"veto_confidence": rec} if rec is not None else None
    current = {"veto_confidence": cur} if cur is not None else None
    return proposed, current


def _proposed_current(loop: str, result: dict) -> tuple[Any, Any]:
    if loop == "predictor_params":
        return _veto_proposed_current(result)
    proposed_key, current_key = _PROPOSED_KEYS[loop]
    return result.get(proposed_key), result.get(current_key)


def classify_loop(
    loop: str,
    result: dict | None,
    *,
    assembler_summary: dict | None = None,
    run_error: str | None = None,
) -> dict:
    """Classify one loop's evaluate-run result into an audit record.

    Args:
        loop: audit loop name (key of ``LOOPS``).
        result: the loop's result dict from ``_run_optimizers`` (already
            error-isolated by ``CompletenessTracker.run_module`` — a raising
            loop arrives here as ``{"status": "error", ...}``).
        assembler_summary: compact assembler outcome (executor_params only,
            under cutover): ``{"status", "cutover_status", "writers",
            "notes"}`` or None when the assembler did not complete.
            ``cutover_status`` (not ``status``) is authoritative for
            whether the live key was actually written — see
            ``optimizer.assembler.AssemblerResult.cutover_status``.
        run_error: when the optimizer stage itself aborted before this loop
            produced a result, the exception text (→ outcome "error").
    """
    record: dict[str, Any] = {
        "outcome": "error",
        "blocked_by": None,
        "consecutive_blocked_weeks": 0,  # rewritten by the carry-forward pass
        "detail": "",
        "proposed": None,
        "current": None,
    }

    if result is None:
        record["detail"] = (
            f"loop never produced a result — optimizer stage aborted: {run_error}"
            if run_error else "loop never produced a result"
        )
        return record

    status = result.get("status")
    apply_result = result.get("apply_result")

    if status == "error":
        record["detail"] = f"loop raised: {result.get('error', 'unknown error')}"
        return record

    if status == "skipped":
        record["outcome"] = "insufficient_data"
        record["detail"] = result.get("degradation_reason") or "skipped — required inputs unavailable"
        return record

    proposed, current = _proposed_current(loop, result)
    record["proposed"] = proposed
    record["current"] = current

    # Pre-apply statuses (no promotable recommendation reached apply()).
    status_map = _STATUS_MAPS[loop]
    if status != "ok":
        outcome, blocked_by = status_map.get(status, (None, None))
        if outcome is None:
            record["detail"] = f"unrecognized loop status {status!r}: {result.get('note') or result.get('reason') or ''}"
            return record  # outcome stays "error" — loud on classifier gaps
        record["outcome"] = outcome
        # Prefer the machine-readable blocked_by the producer stamped, else
        # the status-derived slug.
        record["blocked_by"] = result.get("blocked_by") or blocked_by
        record["detail"] = str(
            result.get("recommendation_reason")
            or result.get("note")
            or result.get("reason")
            or status
        )
        return record

    # status == "ok" → the apply() gate decided.
    if not isinstance(apply_result, dict):
        record["detail"] = "status ok but no apply_result recorded — apply() was never invoked"
        return record

    if apply_result.get("applied"):
        record["outcome"] = "promoted"
        record["detail"] = "promoted to live config"
        return record

    reason = str(apply_result.get("reason", ""))

    if reason.startswith("frozen"):
        record["outcome"] = "disabled"
        record["detail"] = "run invoked with --freeze — apply short-circuited"
        return record

    if reason.startswith("shadow mode"):
        record["outcome"] = "disabled"
        record["detail"] = f"live write disabled by flag: {reason}"
        return record

    if reason.startswith("cutover_mode"):
        # The assembler is the sole live writer for this config_type; the
        # loop's true live-key outcome is the assembler's — NOT its merge
        # status. ``cutover_status`` is keyed on whether the live-key
        # put_object actually succeeded (see assembler.CutoverApplyError);
        # ``status`` only says whether the merge produced promotable output
        # and must never by itself be read as "the live key changed".
        if assembler_summary is None:
            record["detail"] = (
                "cutover delegated the live write to the assembler, but the "
                "assembler did not complete this run — live key state unknown"
            )
            return record
        a_status = assembler_summary.get("status")
        cutover_status = assembler_summary.get("cutover_status")
        if cutover_status == "failed":
            # Live-key put_object raised. The live config is UNCHANGED —
            # this is an error, never "promoted", and must NOT reset
            # consecutive_blocked_weeks (config#2331).
            record["outcome"] = "error"
            record["detail"] = (
                "cutover assembler FAILED the live-key write — live config "
                f"unchanged this run: {assembler_summary.get('notes', '')}"
            )
            return record
        if cutover_status == "applied" and a_status == "ok":
            record["outcome"] = "promoted"
            record["detail"] = (
                "live key written by assembler cutover "
                f"(writers: {assembler_summary.get('writers')}; "
                f"{assembler_summary.get('notes', '')})".strip()
            )
            return record
        record["outcome"] = "blocked"
        record["blocked_by"] = ["assembler_skip"]
        record["detail"] = (
            f"cutover assembler made no live write (status={a_status}, "
            f"cutover_status={cutover_status}): "
            f"{assembler_summary.get('notes', '')}"
        )
        return record

    if "S3 write failed" in reason:
        record["detail"] = f"apply failed: {reason}"
        return record  # outcome "error"

    if reason in ("no recommended threshold", "no recommended params"):
        record["outcome"] = "insufficient_data"
        record["detail"] = reason
        return record

    # Guardrail rejection: prefer the machine-readable blocked_by stamped by
    # the apply() path; a rejection with no stamp is a classifier gap —
    # surface it as such rather than mis-binning.
    blocked_by = apply_result.get("blocked_by")
    record["outcome"] = "blocked"
    record["blocked_by"] = blocked_by or ["unclassified_guardrail"]
    record["detail"] = reason
    return record


def _carry_forward(outcome: str, prior_record: dict | None) -> int:
    """consecutive_blocked_weeks: +1 on blocked (absent prior ⇒ 1), reset on
    promoted/insufficient_data, carried unchanged on error/disabled."""
    prior_n = 0
    if isinstance(prior_record, dict):
        try:
            prior_n = int(prior_record.get("consecutive_blocked_weeks", 0))
        except (TypeError, ValueError):
            prior_n = 0
    if outcome == "blocked":
        return prior_n + 1
    if outcome in ("promoted", "insufficient_data"):
        return 0
    return prior_n  # error / disabled


def load_prior(bucket: str, s3_client=None) -> dict | None:
    """Read the prior audit artifact (latest.json) for the carry-forward
    counter. Absent artifact (first-ever run) → None. Any other read failure
    logs WARN and returns None (counters restart at 1 — degraded but honest;
    the write path below is where failures must raise)."""
    s3 = s3_client or boto3.client("s3")
    key = f"{S3_AUDIT_PREFIX}/latest.json"
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            logger.info("No prior apply-audit artifact at s3://%s/%s (first run)", bucket, key)
        else:
            logger.warning("Prior apply-audit read failed (%s) — counters restart", e)
        return None
    except Exception as e:  # noqa: BLE001 — degraded-read carve-out: the
        # counter restarts at 1 (recorded in the artifact itself); write-path
        # failures still raise below.
        logger.warning("Prior apply-audit read failed (%s) — counters restart", e)
        return None


def build_audit(
    as_of: str,
    opt_results: dict,
    *,
    assembler_summaries: dict[str, dict | None] | None = None,
    prior: dict | None = None,
    run_error: str | None = None,
) -> dict:
    """Build the audit artifact body (schema v1) from the run's results.

    Args:
        assembler_summaries: ``{config_type: compact_summary}`` — one entry
            per config_type that ran under cutover this run (config#2054
            extended cutover from ``executor_params`` alone to all four
            config types, so this is keyed per-loop rather than a single
            scalar). A loop absent from this dict gets ``None`` (matches
            "cutover was off or the assembler did not run for this loop").
    """
    assembler_summaries = assembler_summaries or {}
    prior_loops = (prior or {}).get("loops", {}) if isinstance(prior, dict) else {}
    is_idempotent_rerun = isinstance(prior, dict) and prior.get("as_of") == as_of
    loops: dict[str, dict] = {}
    for loop, results_key in LOOPS.items():
        record = classify_loop(
            loop,
            opt_results.get(results_key),
            assembler_summary=assembler_summaries.get(loop),
            run_error=run_error,
        )
        if is_idempotent_rerun:
            record["consecutive_blocked_weeks"] = prior_loops.get(loop, {}).get("consecutive_blocked_weeks", 0)
        else:
            record["consecutive_blocked_weeks"] = _carry_forward(
                record["outcome"], prior_loops.get(loop),
            )
        loops[loop] = record
    return {
        "schema_version": SCHEMA_VERSION,
        "as_of": as_of,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "loops": loops,
    }


def write_audit(bucket: str, run_date: str, audit: dict, s3_client=None) -> str:
    """Write dated + latest audit artifacts. RAISES on failure — this
    artifact is the load-bearing silence-breaker for the apply loops
    (config#1841); a swallowed write failure would recreate the exact
    invisible-silence defect it exists to retire."""
    s3 = s3_client or boto3.client("s3")
    body = json.dumps(audit, indent=2, allow_nan=False).encode("utf-8")
    dated_key = f"{S3_AUDIT_PREFIX}/{run_date}.json"
    latest_key = f"{S3_AUDIT_PREFIX}/latest.json"
    s3.put_object(Bucket=bucket, Key=dated_key, Body=body, ContentType="application/json")
    s3.put_object(Bucket=bucket, Key=latest_key, Body=body, ContentType="application/json")
    logger.info("Apply-audit written: s3://%s/%s (+ latest.json)", bucket, dated_key)
    return dated_key


def summarize_assembler(assemble_result) -> dict | None:
    """Compact, JSON-safe summary of an AssemblerResult for classification."""
    if assemble_result is None:
        return None
    try:
        merge_summary = getattr(assemble_result, "merge_summary", {}) or {}
        writers = sorted({
            v.get("writer")
            for v in merge_summary.values()
            if isinstance(v, dict) and v.get("writer")
        })
        return {
            "status": getattr(assemble_result, "status", None),
            "cutover_status": getattr(assemble_result, "cutover_status", "not_attempted"),
            "writers": writers,
            "notes": getattr(assemble_result, "notes", ""),
        }
    except Exception as e:  # noqa: BLE001 — summarization must not mask the
        # audit emission; an unreadable assembler result is recorded as such.
        logger.warning("Assembler summary extraction failed: %s", e)
        return {
            "status": None,
            "cutover_status": "failed",
            "writers": [],
            "notes": f"summary extraction failed: {e}",
        }


def emit_apply_audit(
    bucket: str,
    run_date: str,
    opt_results: dict,
    *,
    assembler_results: dict[str, Any] | None = None,
    upload: bool,
    run_error: BaseException | None = None,
    s3_client=None,
) -> dict:
    """Build + log the per-loop apply outcomes; persist to S3 when uploading.

    Called UNCONDITIONALLY at the end of the optimizer stage in evaluate.py —
    including when the stage raised (``run_error`` set): the audit records
    ``error`` outcomes for the affected loops, and the caller re-raises so the
    failure still surfaces (except-log-emit-reraise; no swallow).

    Args:
        assembler_results: ``{config_type: AssemblerResult | None}`` — one
            entry per config_type the assembler ran for this cycle
            (config#2054: all four config types run under cutover, not just
            ``executor_params``).
    """
    prior = load_prior(bucket, s3_client=s3_client) if upload else None
    assembler_summaries = {
        config_type: summarize_assembler(result)
        for config_type, result in (assembler_results or {}).items()
    }
    audit = build_audit(
        run_date,
        opt_results or {},
        assembler_summaries=assembler_summaries,
        prior=prior,
        run_error=str(run_error) if run_error else None,
    )
    for loop, rec in audit["loops"].items():
        logger.info(
            "apply_audit [%s]: outcome=%s blocked_by=%s consecutive_blocked_weeks=%d — %s",
            loop, rec["outcome"], rec["blocked_by"],
            rec["consecutive_blocked_weeks"], rec["detail"],
        )
    if upload:
        try:
            write_audit(bucket, run_date, audit, s3_client=s3_client)
        except Exception:
            if run_error is not None:
                # The stage's original failure is about to re-raise — don't
                # mask it with the audit-write failure; record and let the
                # primary error surface.
                logger.exception(
                    "Apply-audit S3 write failed while an optimizer-stage "
                    "error is pending — original error re-raises",
                )
            else:
                raise
    else:
        logger.info(
            "Apply-audit S3 write skipped (upload=%s) — audit logged only", upload,
        )
    return audit
