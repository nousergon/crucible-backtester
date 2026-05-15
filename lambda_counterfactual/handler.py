"""lambda_counterfactual/handler.py — Weekly counterfactual rule fit Lambda.

Wraps ``replay.counterfactual.compute_and_emit`` for the Saturday SF
weekly run. Reads captured DecisionArtifacts from the trailing window,
fits a depth-≤3 ``DecisionTreeClassifier`` per supported agent on
(input → decision) pairs, emits the ``agent_counterfactual_rule_fit``
CloudWatch metric, persists per-agent analysis JSON to S3.

Per ROADMAP P0 "Replay harness + agent-justification gate" sub-bullet
#7c — third leg of the agent-justification triple alongside cross-week
clustering + cheap-model concordance.

Lambda configuration:
  Memory: 512 MB  |  Timeout: 600s  |  Runtime: container (python:3.12)

Lighter than the concordance Lambda — no LLM calls, no langchain. Pure
S3 + sklearn + CloudWatch.

Event shape (all fields optional):

    {
      "end_time_iso":  "2026-05-09T00:00:00Z",   # default: now UTC
      "window_days":   56,                        # default: 8 weeks
      "max_depth":     3,                         # default: 3 ("3-deep rule")
      "agents":        ["ic_cio","macro_economist"],  # default: all supported (v1)
      "dry_run":       false                      # default: false
    }

Returns:

    {
      "status": "OK" | "PARTIAL" | "ERROR",
      "summary": <compute_and_emit result>
    }

Cost: $0 ongoing. No LLM calls.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime

# Project root on sys.path so ``from replay.counterfactual import ...``
# resolves in the Lambda task layout. Mirrors lambda_health +
# lambda_concordance pattern.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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
    "lambda_counterfactual",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
)

logger = logging.getLogger(__name__)


_init_done = False


def _ensure_init() -> None:
    """Run deferred init once, on the first handler invocation.

    Post-L2998-PR-9c (2026-05-14): secrets load via
    alpha_engine_lib.secrets.get_secret() at use-site. No bulk SSM
    fetch here. Retained for parity with the Lambda fleet + the
    XDG_CACHE_HOME default needed for Lambda's read-only /var/task."""
    global _init_done
    if _init_done:
        return
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    _init_done = True


def handler(event: dict, context) -> dict:
    """Compute + emit per-agent counterfactual rule fit.

    Returns OK / PARTIAL / ERROR same as the concordance Lambda:
      OK      — every analyzed agent's analysis persisted; no failures.
      PARTIAL — load_failures or fit_failures non-empty; run completed.
      ERROR   — compute_and_emit raised at the orchestration layer.
    """
    _ensure_init()

    from replay.counterfactual import compute_and_emit

    t0 = time.time()
    bucket = os.environ.get("S3_BUCKET", "alpha-engine-research")

    end_time_iso = event.get("end_time_iso")
    end_time = (
        datetime.fromisoformat(end_time_iso.replace("Z", "+00:00"))
        if end_time_iso else None
    )
    window_days = int(event.get("window_days", 56))
    max_depth = int(event.get("max_depth", 3))
    agent_filter = event.get("agents") or None
    if isinstance(agent_filter, str):
        agent_filter = [a.strip() for a in agent_filter.split(",") if a.strip()]
    dry_run = bool(event.get("dry_run", False))

    logger.info(
        "[lambda_counterfactual] start window_days=%d max_depth=%d "
        "agents=%s dry_run=%s end_time=%s",
        window_days, max_depth, agent_filter, dry_run,
        end_time_iso or "(now UTC)",
    )

    try:
        summary = compute_and_emit(
            end_time=end_time,
            window_days=window_days,
            max_depth=max_depth,
            agent_filter=agent_filter,
            bucket=bucket,
            emit_metrics=not dry_run,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[lambda_counterfactual] computation failed hard")
        return {
            "status": "ERROR",
            "error": str(exc),
            "duration_seconds": round(time.time() - t0, 1),
        }

    elapsed = time.time() - t0

    has_failures = bool(summary.get("load_failures")) or bool(
        summary.get("fit_failures")
    )
    status = "PARTIAL" if has_failures else "OK"

    logger.info(
        "[lambda_counterfactual] done status=%s duration=%.1fs "
        "agents_analyzed=%d skipped_thin=%d unsupported=%d",
        status, elapsed,
        summary.get("agents_analyzed", 0),
        len(summary.get("agents_skipped_thin_sample", [])),
        len(summary.get("agents_unsupported", [])),
    )

    return {
        "status": status,
        "duration_seconds": round(elapsed, 1),
        "summary": summary,
    }
