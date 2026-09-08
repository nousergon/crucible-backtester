"""lambda_concordance/handler.py — Weekly cheap-model concordance Lambda.

Wraps ``replay.batch.compute_and_emit_concordance`` for the Saturday SF
weekly run. Iterates the trailing-window decision_artifacts corpus,
replays each artifact under the configured target model(s) through the
krepis router edge + the canonical Pydantic schema, aggregates
agreement_score per (agent_id_base, target_model), emits the
``agent_cheap_model_concordance`` CloudWatch metric, persists per-target
summary JSON to S3.

**alpha-engine-config-I7878 (2026-08-20):** migrated off DIRECT provider
linkage onto ``krepis.router.resolve_model_spec`` — see
``replay/runner.py``'s module docstring for the full rationale, including
what the migration does to cross-run comparability of
``agent_cheap_model_concordance`` (it is a level shift, and the
CloudWatch ``target_model`` dimension changes so the break is visible).
``target_models`` is now a list of REGISTRY ENTRY IDS from
``alpha-engine-config/private-docs/LLM_MODEL_REGISTRY.yaml``, not
provider slugs.

This function is already provisioned for the router (measured
2026-08-20): ``KREPIS_EXEC_CONTEXT=lambda``,
``KREPIS_LITELLM_PROXY_URL`` naming the TLS edge, and its own
per-consumer credential via ``KREPIS_ROUTER_CREDENTIAL_SECRET``. No
provider key is read any more.

**alpha-engine-config-I2997 (2026-07-19):** migrated off direct Anthropic
(``langchain_anthropic.ChatAnthropic``).

**alpha-engine-config#3003 (2026-07-30):** migrated remaining plaintext
Lambda env-vars to SSM resolution via ``nousergon_lib.secrets.get_secret()``
at cold-start init. Secrets are loaded once and set in ``os.environ`` so
library consumers (krepis, flow-doctor, langchain) find them through
standard env-var paths. No PROVIDER key appears in ``_SECRET_NAMES`` —
after alpha-engine-config-I7878 this Lambda authenticates to the router
edge with its own per-consumer credential, which krepis resolves, and a
provider key present in the process would be a standing liability with
no call able to use it.

Per ROADMAP P0 "Replay harness + agent-justification gate" (Model-
Agnostic Capability Upgrade deliverable #7 — agent-justification gate
signal #3, cheap-model concordance).

Lambda configuration:
  Memory: 1024 MB  |  Timeout: 900s  |  Runtime: container (python:3.12)

Event shape (all fields optional):

    {
      "target_models": ["deepseek-v4-flash"],   # registry ids, default shown
      "end_time_iso":  "2026-05-09T00:00:00Z", # default: now UTC
      "window_days":   56,                      # default: 8 weeks
      "agents":        ["sector_quant", "ic_cio"],  # default: all 6 canonical
      "max_artifacts": 150,                     # default: cap fits 900s timeout
      "dry_run":       false                    # default: false
    }

Returns:

    {
      "status": "OK" | "PARTIAL" | "ERROR",
      "summary": <compute_and_emit_concordance result>
    }

Cost note: every replay invocation costs target-model tokens. DeepSeek V4
Flash is materially cheaper per-token than the pre-migration Haiku
baseline. The ``max_artifacts`` cap is also a runtime cap — at
~3-5 sec / replay call, 150 artifacts fits comfortably under the 900s
Lambda timeout.

Secrets (loaded from SSM ``/alpha-engine/<NAME>`` by _ensure_init at
cold-start unless already present — see ``_SECRET_NAMES``):
  All secrets previously set as plaintext Lambda env vars are now loaded
  from AWS SSM Parameter Store via ``nousergon_lib.secrets.get_secret()``.
  Non-secret env vars (S3_BUCKET, EMAIL_SENDER, EMAIL_RECIPIENTS) remain
  as configured on the Lambda.

Environment variables (set on the Lambda):
  S3_BUCKET             — default: alpha-engine-research
  EMAIL_SENDER          — flow-doctor wiring
  EMAIL_RECIPIENTS      — flow-doctor wiring
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone

# Project root on sys.path so ``from replay.batch import ...`` resolves
# in the Lambda task layout. Mirrors lambda_health/handler.py pattern.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Structured logging + flow-doctor singleton via alpha-engine-lib.
# LAMBDA_TASK_ROOT (=/var/task in the Lambda image) takes precedence;
# falls back to two-dirs-up for local dev. flow-doctor.yaml only
# references EMAIL_* env vars populated by Lambda's `--environment`
# block before the interpreter starts, so module-top init is safe.
# Secrets load via nousergon_lib.secrets.get_secret() at use-site.
from nousergon_lib.logging import setup_logging, monitor_handler
_FLOW_DOCTOR_EXCLUDE_PATTERNS: list[str] = []
_FLOW_DOCTOR_YAML = os.path.join(
    os.environ.get(
        "LAMBDA_TASK_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ),
    "flow-doctor.yaml",
)
setup_logging(
    "lambda_concordance",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
)

logger = logging.getLogger(__name__)


# ── Secrets migrated from plaintext Lambda env vars to SSM (config#3003) ────
# No PROVIDER key is listed. The direct-Anthropic key was dropped by I2997; the
# direct-provider key this module used after it was dropped by I7878, when
# the target-model call moved onto the router edge and started
# authenticating with this function's own per-consumer credential (krepis
# resolves it from KREPIS_ROUTER_CREDENTIAL_SECRET). Non-secret config
# (S3_BUCKET, EMAIL_SENDER, EMAIL_RECIPIENTS) stays as Lambda env vars.
_SECRET_NAMES: tuple[str, ...] = (
    "GITHUB_TOKEN",
    "LANGCHAIN_API_KEY",
    "GMAIL_APP_PASSWORD",
    "VOYAGE_API_KEY",
    "FMP_API_KEY",
    "POLYGON_API_KEY",
    "FRED_API_KEY",
    "RAG_DATABASE_URL",
)


def _load_secrets_from_ssm() -> None:
    """Load secrets from SSM into ``os.environ`` at cold-start init.

    Each secret name ``X`` maps to the SSM parameter
    ``/alpha-engine/<X>`` (the fleet standard prefix). Loaded
    only when NOT already present in the environment (during the
    transition period where the Lambda env vars are still configured;
    after the plaintext vars are removed from the Lambda configuration,
    the SSM values fill the same env var names so library consumers
    find them unchanged).

    ``get_secret(..., required=False, default=None)`` makes every secret
    optional: a parameter not yet created just logs a debug note and
    does not fail the invocation. This is the correct failure mode
    during the transition — missing SSM parameters are a provisioning
    gap, and a loud-env-var-error on the next run is the signal. The
    ``test_no_secret_environ_reads`` CI guard does NOT catch the
    ``os.environ.__setitem__`` calls below (it checks for
    ``os.environ.get`` / ``os.getenv`` calls in source — assignment
    from SSM is the intended path).
    """
    try:
        from nousergon_lib.secrets import get_secret  # noqa: PLC0415 — lazy import; Lambda runtime has it
    except ImportError:
        return  # Not in a Lambda / dev environment — skip gracefully

    for name in _SECRET_NAMES:
        if name in os.environ:
            # Already set (still configured on the Lambda during
            # transition; after the plaintext env vars are removed
            # this branch becomes dead and the SSM path below runs).
            continue
        value = get_secret(name, required=False, default=None)
        if value is not None:
            os.environ[name] = value
            logger.debug("Loaded secret from SSM")


_init_done = False


def _ensure_init() -> None:
    """Run deferred init once, on the first handler invocation.

    Post-L2998-PR-9c (2026-05-14): secrets load via
    nousergon_lib.secrets.get_secret() at use-site (per-process
    cached). Retained for the XDG_CACHE_HOME default needed for
    Lambda's read-only /var/task. config#3003 added the SSM bulk
    load above (``_SECRET_NAMES``) so library consumers find their
    secrets in ``os.environ`` without plaintext Lambda env vars."""
    global _init_done
    if _init_done:
        return
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    _load_secrets_from_ssm()
    _init_done = True


def _remaining_seconds(context):
    """Zero-arg callable giving the seconds left in this invocation.

    ``None`` when there is no Lambda context (local runs, tests), which the
    batch module treats as "no deadline" — identical behaviour to before
    config#6920.

    NOT decorated with ``@monitor_handler``: that decorator is flow-doctor's
    crash capture and belongs on the entry point. config#6920 inserted this
    helper directly above ``handler`` and the decorator stayed with the
    line above it rather than the function it named, so from 2026-08-11 the
    real handler ran unwrapped and an unhandled exception in it reached
    Lambda without ever reaching ``fd.report``.
    """
    getter = getattr(context, "get_remaining_time_in_millis", None)
    if not callable(getter):
        return None
    return lambda: getter() / 1000.0


@monitor_handler
def handler(event: dict, context) -> dict:
    """Entry point. Runs the concordance, then flushes cost telemetry.

    The `finally` is the whole point (alpha-engine-config-I7423).
    `krepis.cost_sink.S3JsonlCostSink` buffers to 200 records per
    `(date, callsite_id)` group and otherwise relies on an `atexit` hook —
    and **an AWS Lambda container is FROZEN between invocations, not
    exited, so `atexit` never runs.** A handler finishing below the
    threshold writes nothing at all, and the container may be reclaimed
    hours later without ever reaching interpreter shutdown.

    Measured 2026-08-15 on weekly-SF execution `watch-rerun-2026-08-15-2`:
    this Lambda ran 812 seconds of DeepSeek calls over 119 artifacts, and
    `AggregateCosts` then reported `replay-concordance` among `2 stage(s)
    ran and emitted no cost record ... Observed producers: (none)`. The env
    wiring was correct (config-I7179), the sink was constructed, the records
    were priced and accepted — and every one of them died in memory.

    It wraps `_run` rather than living inside it because `_run` has four
    return paths (dry-run, hard failure, PARTIAL, OK); a flush on the last
    one only would have kept exactly the failure this fixes on the others.

    `flush_default_sink` returns 0 when no sink is configured and never
    raises — telemetry must not take down the work it measures.
    """
    try:
        return _run(event, context)
    finally:
        try:
            from krepis.cost_sink import flush_default_sink
            _n = flush_default_sink()
            if _n:
                logger.info("[lambda_concordance] cost sink flushed: %d object(s)", _n)
        except ImportError as exc:
            # Loud, not silent: the image's krepis pin predates the function
            # (floor is >=0.59.8). Cost records for this run are lost, and
            # AggregateCosts' fan-in coverage check will say so by name.
            logger.error("cost-sink flush unavailable — records lost: %s", exc)


def _run(event: dict, context) -> dict:
    """Compute + emit per-(agent_id, target_model) cheap-model concordance.

    Returns a status envelope:

      OK     — replay succeeded for every (target, group); no failures.
      PARTIAL — at least one replay or metric-emission failure recorded;
               run completed but some signal is missing.
      ERROR  — compute_and_emit_concordance raised at the orchestration
               layer (S3 listing failure, deferred-import bust, etc.).
    """
    _ensure_init()

    # Captured at handler entry (config-I7214) — the window the per-stage
    # coverage assertion below uses to distinguish this run's artifact from
    # a leftover of a previous cycle.
    _started = datetime.now(timezone.utc)

    # Imports deferred until after _ensure_init so SSM-loaded secrets
    # are available for any module-level init that consults them.
    from replay import is_shell_run_dry, shell_run_dry_response
    from replay.batch import (
        DEFAULT_MAX_ARTIFACTS,
        compute_and_emit_concordance,
    )

    t0 = time.time()

    # Shell-run dry path (Saturday-SF keystone). Boot + module imports
    # above have already run for real (the keystone's whole point —
    # exercise bootstrap/import/lib-pin/transport). Return a benign
    # success BEFORE the replay.batch scan (decision_artifacts S3
    # discovery), BEFORE any router / target-model call, and BEFORE
    # any CloudWatch metric emit or S3 summary persist.
    if is_shell_run_dry(event):
        logger.info(
            "[lambda_concordance] shell-run dry path: boot+imports OK, "
            "skipping replay scan + router calls + S3/CW writes"
        )
        return shell_run_dry_response("lambda_concordance", t0)

    bucket = os.environ.get("S3_BUCKET", "alpha-engine-research")

    # A REGISTRY ENTRY ID, not a provider slug (alpha-engine-config-I7878).
    # `deepseek-v4-flash` is deliberately kept `active` and out of every
    # model_groups chain precisely so it stays callable by name; the router
    # refuses an unknown id and names what IS addressable.
    target_models = event.get("target_models") or ["deepseek-v4-flash"]
    if isinstance(target_models, str):
        # Convenience: accept comma-separated string from SF parameters.
        target_models = [m.strip() for m in target_models.split(",") if m.strip()]

    end_time_iso = event.get("end_time_iso")
    end_time = (
        datetime.fromisoformat(end_time_iso.replace("Z", "+00:00"))
        if end_time_iso else None
    )
    window_days = int(event.get("window_days", 56))
    agent_filter = event.get("agents") or None
    if isinstance(agent_filter, str):
        agent_filter = [a.strip() for a in agent_filter.split(",") if a.strip()]
    # Cap sized against the 900s Lambda timeout. The original rationale
    # here read "150 artifacts × ~3-5 sec/replay ≈ 450-750 sec"; measured
    # per-item latencies on 2026-08-11 were 6s-137s (p90 well above 20s),
    # so that estimate was wrong by 2-30× and the cap could never bind —
    # ReplayConcordance hit the wall at Status: timeout with the whole run's
    # aggregate discarded (config#6920). The cap stays as a COST guard; the
    # deadline below is what now bounds the time, measured from this run's
    # own item latencies rather than an assumed constant. The batch module's
    # DEFAULT_MAX_ARTIFACTS (500) remains right for spot runs with no
    # deadline. Override via event if the corpus is sparse.
    max_artifacts = int(event.get("max_artifacts", min(150, DEFAULT_MAX_ARTIFACTS)))
    dry_run = bool(event.get("dry_run", False))

    logger.info(
        "[lambda_concordance] start target_models=%s window_days=%d "
        "agents=%s max_artifacts=%d dry_run=%s end_time=%s",
        target_models, window_days, agent_filter, max_artifacts,
        dry_run, end_time_iso or "(now UTC)",
    )

    try:
        summary = compute_and_emit_concordance(
            target_models=target_models,
            end_time=end_time,
            window_days=window_days,
            agent_filter=agent_filter,
            bucket=bucket,
            max_artifacts=max_artifacts,
            emit_metrics=not dry_run,
            dry_run=dry_run,
            remaining_s=_remaining_seconds(context),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[lambda_concordance] computation failed hard")
        return {
            "status": "ERROR",
            "error": str(exc),
            "duration_seconds": round(time.time() - t0, 1),
        }

    elapsed = time.time() - t0

    # Status pattern matches the rolling-mean Lambda: OK when no
    # failures recorded, PARTIAL when any (replay error, metric emit
    # error, persist error) surface but the run completed. Eval is
    # observability — partial signal is preferable to abort.
    has_failures = False
    incomplete = False

    # A DRY RUN that could not resolve a target is a FAILED dry run.
    #
    # This is what makes the deploy canary cover the routed path. Every
    # `dry_run: true` invocation now resolves each target through the router
    # (replay.batch, alpha-engine-config-I7878), and `deploy_concordance.sh`
    # promotes the `live` alias only on an OK/PARTIAL status — so a mistyped
    # registry id, a retired registry entry, an unreachable router edge or a
    # missing per-consumer credential fails the deploy and leaves `live` on
    # the prior good version, instead of surfacing on Saturday as a lost
    # weekly-SF stage.
    unresolved = [
        r for r in summary.get("target_resolution", [])
        if not r.get("resolved")
    ]
    if unresolved:
        for r in unresolved:
            logger.error(
                "[lambda_concordance] target %s did not resolve: %s",
                r.get("target_model"), r.get("error"),
            )
        return {
            "status": "ERROR",
            "error": (
                "target model(s) did not resolve through the router: "
                + "; ".join(
                    f"{r.get('target_model')}: {r.get('error')}"
                    for r in unresolved
                )
            ),
            "duration_seconds": round(time.time() - t0, 1),
            "summary": summary,
        }

    if not dry_run:
        for target_summary in summary.get("per_target_model", []):
            if target_summary.get("replay_failures"):
                has_failures = True
            # config#6920: a run that stopped on budget covered less of the
            # corpus than it was asked to. That is partial signal, and saying
            # OK would make a truncated sweep read as a full one.
            if target_summary.get("budget_stopped"):
                incomplete = True
                logger.warning(
                    "[lambda_concordance] target=%s stopped on budget: %s of %s "
                    "artifacts not replayed",
                    target_summary.get("target_model"),
                    target_summary.get("n_artifacts_skipped_for_budget"),
                    target_summary.get("n_artifacts_candidate"),
                )
    status = "PARTIAL" if (has_failures or incomplete) else "OK"

    logger.info(
        "[lambda_concordance] done status=%s duration=%.1fs "
        "artifacts_discovered=%d targets=%d",
        status, elapsed, summary.get("artifacts_discovered", 0),
        len(summary.get("per_target_model", [])),
    )

    result = {
        "status": status,
        "duration_seconds": round(elapsed, 1),
        "summary": summary,
    }

    # Per-stage output assertion (config-I7214, sf-pipeline-policy.md §2.1).
    # OBSERVE MODE — never changes this handler's own outcome. run_date is
    # this Lambda's end_time (the SF's $$.Execution.StartTime), falling
    # back to "now" for a bare invocation with no end_time_iso.
    #
    # alpha-engine-config-I8206: NEVER asserted on a `dry_run=True`
    # invocation. `deploy_concordance.sh` invokes exactly this shape after
    # every publish (`{"dry_run": true, "window_days": 14}`) to validate
    # router resolution before promoting the `:live` alias — a deploy-time
    # canary, not a pipeline stage execution. Before this fix the assertion
    # ran unconditionally: the canary's own `window_start` is captured at
    # DEPLOY time, hours after the Saturday weekly run already wrote a
    # fresh, COVERED artifact, so `assert_stage_coverage` re-evaluated
    # freshness against the canary's later window, found the real artifact
    # "stale" relative to it, and OVERWROTE the correct COVERED verdict in
    # `_stage_coverage/{date}/ReplayConcordance.json` with a false STALE
    # one. Measured 2026-08-22: the real weekly Task invocation (05:21:24
    # PT) wrote `decision_artifacts/_replay_summary/2608220900_deepseek-v4-
    # flash.json` and recorded `status: COVERED`; a deploy canary for
    # Lambda version 240, published 18:00:55 UTC, ran `dry_run=True,
    # window_days=14` at 18:01:06 UTC and clobbered that same S3 object
    # with `status: STALE, covered: []` — a stage that ran and wrote its
    # artifact reported as having produced nothing, entirely from an
    # unrelated deploy-time probe never intended to represent the stage's
    # own execution. A dry run never asserts coverage for the same reason
    # it never persists a summary: it does not run the stage.
    if dry_run:
        result["stage_coverage"] = {
            "stage": "ReplayConcordance",
            "status": "SKIPPED",
            "reason": "dry_run=True (deploy canary) — not a stage execution, never asserted",
        }
        return result
    # alpha-engine-config-I10171 — CORRECTION. This keyed the verdict on
    # `end_time_iso` ($$.Execution.StartTime), the CALENDAR date, while
    # $.run_date is the cycle's TRADING day. On the 2026-09-05 Saturday
    # cycle for trading day 2026-09-04 the verdict landed in
    # `_stage_coverage/2026-09-05/`, a partition the reader stopped
    # consulting that same day (the dual-partition fallback expired, by
    # design) — after which this stage read `absent`, indistinguishable
    # from a stage that never ran.
    #
    # It also FABRICATED: `or _started` substituted this Lambda's own
    # wall-clock for a genuinely-absent execution identity, which is the
    # alpha-engine-config-I8155 forbidden class. Removed — an absent
    # identity now records UNMEASURED with a reason.
    from stage_coverage_run_date import resolve_stage_run_date

    _coverage_run_date, _run_date_provenance = resolve_stage_run_date(
        event,
        stage="ReplayConcordance",
        fallback=end_time.date().isoformat() if end_time else None,
        fallback_source="event.end_time_iso",
        logger=logger,
    )
    if not _coverage_run_date:
        logger.error(
            "stage-coverage assertion SKIPPED for ReplayConcordance: neither run_date "
            "nor end_time_iso on this event (execution identity absent) — "
            "never substituting wall-clock (alpha-engine-config-I8155)",
        )
        result["stage_coverage"] = {
            "stage": "ReplayConcordance",
            "status": "UNMEASURED",
            "reason": "execution run_date absent from event (no run_date, no end_time_iso)",
            **_run_date_provenance,
        }
        return result
    try:
        from krepis.stage_coverage import assert_stage_coverage
        result["stage_coverage"] = {
            **assert_stage_coverage(
                "ReplayConcordance", run_date=_coverage_run_date, window_start=_started,
            ),
            # alpha-engine-config-I10171: which field the partition key came
            # from travels WITH the verdict, into the SF execution history.
            # A fallback nobody can see afterwards reproduces the defect.
            **_run_date_provenance,
        }
    except ImportError as exc:
        # Loud, not silent: the lib pin predates the module. Observe mode —
        # the handler's own outcome is unchanged (config-I7214).
        logger.error("stage-coverage assertion unavailable: %s", exc)

    # alpha-engine-config-I10198 / sf-pipeline-policy §2.3b, clause
    # `SFP-2.3b-stage-status-is-the-worst-substatus` (nous-ergon-ops-PR1119).
    # A stage that fans out into named sub-results and returns ONE
    # enclosing status can lose arbitrary work while every §2.3
    # mechanism — this state's `Catch`, `MarkReplayConcordanceDegraded`,
    # the completion marker — reports health, because all of them key
    # off the STAGE's status. Structural, not a list of sub-result
    # names: the hand-kept list is how `EvalRollingMean` reported OK
    # over an errored `agent_quality` for two weeks. A no-op on a
    # payload whose sub-results all passed.
    from stage_substatus import enforce_worst_substatus

    enforce_worst_substatus(result, stage="ReplayConcordance", logger=logger)

    return result
