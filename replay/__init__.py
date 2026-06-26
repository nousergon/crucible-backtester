"""Replay harness — re-run captured DecisionArtifacts under a different model.

Loads a stored ``DecisionArtifact`` from S3, reconstructs the original
prompt context (system + user + tool definitions), invokes a target
model via the Anthropic SDK, and persists a side-by-side comparison
artifact under the canonical eval_artifacts layout at
``decision_artifacts/_replay/{run_id}_{original_model}_vs_{target_model}.json``
(flat + ``latest.json`` sidecar; config#792).

Per ROADMAP P0 "Replay harness + agent-justification gate" (Model-
Agnostic Capability Upgrade deliverable #3):

  Tool that takes a stored decision artifact and re-runs it against a
  different model version, capturing the new decision for side-by-side
  comparison. Must support batch replay across historical date ranges
  for capability-delta measurement.

This package contains:

- ``runner.py`` — single-artifact replay (PR A)
- ``comparison.py`` — per-agent agreement scorers (PR B, follow-up)
- ``cli.py`` — CLI entry point (PR A single mode, PR C batch mode)

Composes with:

- ``decision_capture`` (alpha-engine-lib) — reads the captured artifact.
- Cost telemetry (closed 2026-05-01) — replay calls tagged
  ``run_type="replay"`` so judging cost is observable + bounded.
- Cross-week rationale clustering (closed 2026-05-05) — clustering
  measures *what* the agent emits; replay measures *whether a different
  model would emit the same*. Together they cover the agent-
  justification triple alongside the counterfactual-rule-fit signal.
"""

from __future__ import annotations

import time
from typing import Any

# Canonical Saturday-SF shell-run dry-path event key. Established
# verbatim by the shell-run keystone (alpha-engine-data
# step_function.json) for the Research Lambda
# (``"dry_run_llm.$": "$.research_dry"``); reused here so the
# ReplayConcordance + Counterfactual states can be routed dry (boot +
# imports for real, return a benign success before any scan / external
# call / S3 / CloudWatch write) instead of pure-skipped. Distinct from
# the handlers' pre-existing ``dry_run`` event key, which has a
# different (compute-but-do-not-emit-metrics) semantic and is left
# untouched for backward compatibility.
SHELL_RUN_DRY_EVENT_KEY = "dry_run_llm"


def is_shell_run_dry(event: dict | None) -> bool:
    """True when the SF shell-run keystone routed this Lambda dry.

    Reads the canonical ``dry_run_llm`` boolean off the invocation
    event. Tolerates a missing/None event and string ``"true"``/``"1"``
    forms (Step Functions string-parameter convenience), mirroring the
    coercion the handlers already apply to ``agents``/``target_models``.
    """
    if not event:
        return False
    raw = event.get(SHELL_RUN_DRY_EVENT_KEY, False)
    if isinstance(raw, str):
        return raw.strip().lower() in {"true", "1", "yes"}
    return bool(raw)


def shell_run_dry_response(handler_name: str, t0: float) -> dict:
    """Benign success envelope returned BEFORE the replay scan.

    Returned by both replay Lambdas when ``is_shell_run_dry`` is true.
    Hard invariant at the call site: zero external/LLM calls, zero
    S3/CloudWatch writes, no decision_artifacts discovery — boot +
    module imports have already run for real by the time this is
    called. ``status`` is a recognised value the SF (Catch-wrapped,
    non-blocking) treats as success.
    """
    return {
        "status": "DRY_RUN",
        "dry_run": True,
        "handler": handler_name,
        "note": (
            "shell-run dry path: boot + imports executed; replay scan, "
            "external/LLM calls, and all S3/CloudWatch writes skipped"
        ),
        "duration_seconds": round(time.time() - t0, 1),
    }


__all__ = [
    "SHELL_RUN_DRY_EVENT_KEY",
    "is_shell_run_dry",
    "shell_run_dry_response",
]
