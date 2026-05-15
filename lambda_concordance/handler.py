"""lambda_concordance/handler.py — Weekly cheap-model concordance Lambda.

Wraps ``replay.batch.compute_and_emit_concordance`` for the Saturday SF
weekly run. Iterates the trailing-window decision_artifacts corpus,
replays each artifact under the configured target model(s) via
langchain_anthropic + the canonical Pydantic schema, aggregates
agreement_score per (agent_id_base, target_model), emits the
``agent_cheap_model_concordance`` CloudWatch metric, persists per-target
summary JSON to S3.

Per ROADMAP P0 "Replay harness + agent-justification gate" (Model-
Agnostic Capability Upgrade deliverable #7 — agent-justification gate
signal #3, cheap-model concordance).

Lambda configuration:
  Memory: 1024 MB  |  Timeout: 900s  |  Runtime: container (python:3.12)

Event shape (all fields optional):

    {
      "target_models": ["claude-haiku-4-5"],   # default: ["claude-haiku-4-5"]
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

Cost note: every replay invocation costs target-model tokens. At
production cadence (1 target × 150 artifacts × Haiku ≈ $0.08 / week).
The ``max_artifacts`` cap is also a runtime cap — at ~3-5 sec / replay
call, 150 artifacts fits comfortably under the 900s Lambda timeout.

Environment variables:
  S3_BUCKET             — default: alpha-engine-research
  ANTHROPIC_API_KEY     — pulled from SSM by ssm_secrets.load_secrets
  EMAIL_SENDER          — flow-doctor wiring
  EMAIL_RECIPIENTS      — flow-doctor wiring
  GMAIL_APP_PASSWORD    — flow-doctor wiring
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime

# Project root on sys.path so ``from replay.batch import ...`` resolves
# in the Lambda task layout. Mirrors lambda_health/handler.py pattern.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Structured logging + flow-doctor singleton via alpha-engine-lib.
# LAMBDA_TASK_ROOT (=/var/task in the Lambda image) takes precedence;
# falls back to two-dirs-up for local dev. flow-doctor.yaml only
# references EMAIL_* env vars populated by Lambda's `--environment`
# block before the interpreter starts, so module-top init is safe.
# Secrets load via alpha_engine_lib.secrets.get_secret() at use-site.
from alpha_engine_lib.logging import setup_logging
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


_init_done = False


def _ensure_init() -> None:
    """Run deferred init once, on the first handler invocation.

    Post-L2998-PR-9c (2026-05-14): secrets load via
    alpha_engine_lib.secrets.get_secret() at use-site (per-process
    cached). No bulk SSM fetch on cold-start. Retained for the
    XDG_CACHE_HOME default needed for Lambda's read-only /var/task."""
    global _init_done
    if _init_done:
        return
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    _init_done = True


def handler(event: dict, context) -> dict:
    """Compute + emit per-(agent_id, target_model) cheap-model concordance.

    Returns a status envelope:

      OK     — replay succeeded for every (target, group); no failures.
      PARTIAL — at least one replay or metric-emission failure recorded;
               run completed but some signal is missing.
      ERROR  — compute_and_emit_concordance raised at the orchestration
               layer (S3 listing failure, deferred-import bust, etc.).
    """
    _ensure_init()

    # Imports deferred until after _ensure_init so SSM-loaded secrets
    # are available for any module-level init that consults them.
    from replay.batch import (
        DEFAULT_MAX_ARTIFACTS,
        compute_and_emit_concordance,
    )

    t0 = time.time()
    bucket = os.environ.get("S3_BUCKET", "alpha-engine-research")

    target_models = event.get("target_models") or ["claude-haiku-4-5"]
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
    # Default cap chosen to fit comfortably within the 900s Lambda
    # timeout: 150 artifacts × ~3-5 sec/replay ≈ 450-750 sec. The
    # batch module's documented DEFAULT_MAX_ARTIFACTS (500) is
    # appropriate for spot-instance runs without a hard deadline; for
    # Lambda we tighten it. Override via event if the corpus is sparse.
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
    if not dry_run:
        for target_summary in summary.get("per_target_model", []):
            if target_summary.get("replay_failures"):
                has_failures = True
                break
    status = "PARTIAL" if has_failures else "OK"

    logger.info(
        "[lambda_concordance] done status=%s duration=%.1fs "
        "artifacts_discovered=%d targets=%d",
        status, elapsed, summary.get("artifacts_discovered", 0),
        len(summary.get("per_target_model", [])),
    )

    return {
        "status": status,
        "duration_seconds": round(elapsed, 1),
        "summary": summary,
    }
