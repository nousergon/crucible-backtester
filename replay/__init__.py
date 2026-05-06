"""Replay harness — re-run captured DecisionArtifacts under a different model.

Loads a stored ``DecisionArtifact`` from S3, reconstructs the original
prompt context (system + user + tool definitions), invokes a target
model via the Anthropic SDK, and persists a side-by-side comparison
artifact at ``decision_artifacts/_replay/{run_id}/{original_model}_vs_{target_model}.json``.

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
