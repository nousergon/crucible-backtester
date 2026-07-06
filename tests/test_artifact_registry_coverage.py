"""
Artifact-registry coverage CI guard.

Phase 4 PR 7 of the artifact-freshness-monitor arc (plan doc:
``~/Development/alpha-engine-docs/private/artifact-freshness-monitor-260527.md``).
Mirrors ``alpha-engine-data/tests/test_artifact_registry_coverage.py``
(PR 4, merged 2026-05-27); the cascade closes producer-side coverage
of the registry across all 4 producing repos (ae-data, ae-research,
ae-predictor, this repo).

**What this catches.** A new ``s3.put_object(...)`` or
``s3.upload_file(...)`` site in ae-backtester production code that
hasn't been registered in
``alpha-engine-config/private-docs/ARTIFACT_REGISTRY.yaml`` (or
explicitly grandfathered). Forces operator attention at every new
producer addition — the silent absence-of-artifact bug class
(e.g., 2026-05-17→27 ``pit_parity.json`` — which itself originates
in this repo, ``analysis/pit_parity.py``) can't slip past PR review
without an explicit register-or-grandfather decision.

**Design choice — per-file count rather than per-key-template
extraction.** Statically extracting key templates from f-string
``put_object(Key=...)`` calls is fragile (keys are often constructed
from surrounding context — backtester's optimizer sub-modules in
particular construct dynamic keys per param-sweep run). Per-file
count is stable across refactors and sufficient to force operator
review. See the ae-data PR 4 commit message for the full rationale.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Per-file PUT-site counts. Pinning enforces operator attention on
# every new producer addition. Captured 2026-05-27.
EXPECTED_PER_FILE_PUT_COUNTS: dict[str, int] = {
    # config#946 attribution-persistence monitor — 1 PUT of an append-only
    # history (decision_artifacts/_attribution_adequacy/history.jsonl), gated on
    # --upload (default off), idempotent per run_date. Explicitly mirrors
    # analysis/retrain_alert.py's _write_alert_to_s3 (an alert/monitor producer
    # grandfathered here, NOT freshness-SLA registered): a default-off diagnostic
    # whose absence does not break trading, so grandfathered rather than added to
    # ARTIFACT_REGISTRY.yaml (a freshness SLA on a default-off monitor would
    # false-alarm; and the producer is not yet live).
    "analysis/attribution_persistence.py": 1,
    "analysis/cost_report.py": 1,
    "analysis/feature_drift.py": 1,
    # L4471 L2 within-run sim checkpoint — transient (deleted on success), not a
    # freshness-tracked SLA artifact; the backtest/{trading_day}/_sim_checkpoint/
    # prefix is grandfathered in ARTIFACT_REGISTRY.yaml.
    "store/sim_checkpoint.py": 1,
    "analysis/grade_history.py": 1,
    "analysis/pit_parity.py": 1,
    "analysis/production_health.py": 1,
    "analysis/regime_stratified_sortino_runner.py": 2,
    "analysis/retrain_alert.py": 2,
    "analysis/veto_analysis.py": 5,
    # +1 (8→9) 2026-06-01: W3.4 horizon_net_alpha.json — an OBSERVE diagnostic
    # under backtest/{date}/ (sibling of cov_sweep/gamma_sweep), not a
    # freshness-SLA artifact; grandfathered here like the other per-run
    # diagnostics rather than added to the freshness registry.
    # +1 (9→10) 2026-06-02: L4488b model_version_net_alpha.json — per-version
    # net-of-cost OBSERVE diagnostic (same grandfather rationale).
    # +1 (10→11) 2026-06-14: config#1057 optimizer_param_sweep.json — the
    # risk_aversion × tcost_bps sweep verdict under backtest/{date}/ (sibling of
    # cov_sweep/gamma_sweep), OBSERVE-only, grandfathered like the other per-run
    # diagnostics rather than added to the freshness-SLA registry.
    "backtest.py": 11,
    # config#1726: optimizer_run/{trading_day}.json liveness proxy for event_driven
    # config_* rows — registered as optimizer_run_manifest in ARTIFACT_REGISTRY.yaml.
    "evaluate.py": 1,
    # config#1841 apply-audit — 2 PUTs (config/apply_audit/{date}.json + the
    # latest.json mirror): the per-run outcome record for the four auto-apply
    # loops (promoted / blocked-by-guardrail / insufficient_data / disabled /
    # error). A weekly per-run report-card INPUT read by crucible-evaluator
    # (absence ⇒ optimizer stage never ran), like optimizer_run/{date}.json
    # above — registering a freshness SLA row in ARTIFACT_REGISTRY.yaml is the
    # config-repo follow-up owned by the filing session (config#1841 parent).
    "optimizer/apply_audit.py": 2,
    "optimizer/assembler.py": 5,
    "optimizer/barrier_sizing_optimizer.py": 1,
    "optimizer/config_archive.py": 1,
    "optimizer/executor_optimizer.py": 5,
    # config#748 factor_blend_optimizer — 3 PUTs (live config/factor_blend_params.json
    # + shadow_history/{run_id}.json + latest.json sidecar), same config-write /
    # shadow-archive shape as optimizer/tech_weight_ablation.py; live write is
    # enforce-gated + reproduction-gated, default-off. Config writes, not
    # freshness-SLA artifacts: the live key is registered as config_factor_blend_params
    # and the shadow_history/ prefix is grandfathered in ARTIFACT_REGISTRY.yaml.
    "optimizer/factor_blend_optimizer.py": 3,
    # config#1719: apply_cio_mode retired (dead write into research_params.json,
    # no consumer) → 2 PUTs (team_slots + cio_mode) dropped to 1 (team_slots only).
    "optimizer/pipeline_optimizer.py": 1,
    # config#1057 inc 2: MVO optimizer-param auto-apply — 5 PUTs (live + shadow×2
    # + history×2), same config-write shape as optimizer/executor_optimizer.py;
    # config writes, not freshness-SLA artifacts, so pinned here (grandfathered).
    "optimizer/portfolio_optimizer_optimizer.py": 5,
    "optimizer/predictor_optimizer.py": 3,
    "optimizer/predictor_sizing_optimizer.py": 1,
    "optimizer/recommendation_artifact.py": 1,
    "optimizer/regression_monitor.py": 5,
    "optimizer/research_optimizer.py": 3,
    "optimizer/scanner_optimizer.py": 3,
    # L300 (2026-06-01): offline stance-sizing gate, field_overlay to
    # config/executor_params.json (1 PUT in apply()) — mirrors
    # barrier_sizing_optimizer; no new freshness-SLA artifact.
    "optimizer/stance_sizing_optimizer.py": 1,
    "optimizer/tech_weight_ablation.py": 3,
    "optimizer/trigger_optimizer.py": 1,
    "optimizer/weight_optimizer.py": 5,
    "phase_artifacts.py": 5,
    # +1 (2→3) 2026-06-23: config#1151 substrate_ops.json — the per-run watchdog
    # firing aggregate PhaseRegistry.phase() writes for the report card's
    # substrate tile (crucible-evaluator watchdog_firings). A per-run report-card
    # INPUT read windowed by the evaluator (absence → graceful N/A, like
    # sample_size.json), NOT a freshness-SLA artifact — grandfathered here rather
    # than added to the freshness registry, matching the per-run-diagnostic
    # precedent above.
    "pipeline_common.py": 3,
    # +1 (1→2) 2026-06-26: config#792 canonical eval_artifacts layout — each
    # replay producer now writes its dated eval artifact PLUS an operator-UX
    # latest.json sidecar that is a pure mirror of the dated key (last-writer-
    # wins "latest" semantic). The sidecar is not a distinct freshness-SLA
    # artifact — it mirrors the already-registered dated artifact — so it is
    # grandfathered here rather than added to the freshness registry.
    "replay/batch.py": 2,
    "replay/counterfactual.py": 2,
    "replay/runner.py": 2,
    "reporter.py": 1,
}


_SCAN_EXEMPT_PREFIXES: tuple[str, ...] = (
    "tests/",
    "infrastructure/lambdas/",
    ".claude/",
    ".venv/",
    "build/",
)


def _enumerate_put_sites() -> dict[str, int]:
    """Return ``{relative_path: count}`` of production files with PUT sites."""
    result = subprocess.run(
        [
            "git", "grep", "-l", "-E",
            r"(put_object|upload_file)\(",
            "--", "*.py",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    files = [
        line for line in result.stdout.splitlines()
        if line and not any(line.startswith(p) for p in _SCAN_EXEMPT_PREFIXES)
    ]
    counts: dict[str, int] = {}
    for rel in files:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        counts[rel] = len(re.findall(r"(?:put_object|upload_file)\(", text))
    return counts


def test_every_producer_file_is_pinned():
    actual = _enumerate_put_sites()
    unpinned = sorted(set(actual.keys()) - set(EXPECTED_PER_FILE_PUT_COUNTS.keys()))
    assert not unpinned, (
        "New producer file(s) with S3 PUT sites detected but not pinned:\n"
        + "\n".join(f"  - {f} ({actual[f]} PUT call(s))" for f in unpinned)
        + "\n\nResolution:\n"
        "  1. Register the new artifact(s) in alpha-engine-config/"
        "private-docs/ARTIFACT_REGISTRY.yaml (or add the prefix to "
        "grandfathered_paths with a one-line reason).\n"
        "  2. Add the file(s) to EXPECTED_PER_FILE_PUT_COUNTS in "
        "tests/test_artifact_registry_coverage.py with the per-file count.\n"
        "  3. Re-run this test."
    )


def test_every_pinned_file_still_exists():
    actual = _enumerate_put_sites()
    stale = sorted(set(EXPECTED_PER_FILE_PUT_COUNTS.keys()) - set(actual.keys()))
    assert not stale, (
        "Pinned file(s) no longer have PUT sites (or no longer exist):\n"
        + "\n".join(f"  - {f}" for f in stale)
        + "\n\nResolution: remove the file from EXPECTED_PER_FILE_PUT_COUNTS. "
        "If the artifact was retired, also retire its row in "
        "alpha-engine-config/private-docs/ARTIFACT_REGISTRY.yaml."
    )


def test_pinned_counts_match_actual():
    actual = _enumerate_put_sites()
    deltas = []
    for path, expected_count in sorted(EXPECTED_PER_FILE_PUT_COUNTS.items()):
        actual_count = actual.get(path, 0)
        if actual_count != expected_count:
            deltas.append(f"  - {path}: expected={expected_count}, actual={actual_count}")
    assert not deltas, (
        "PUT-site count drift detected:\n"
        + "\n".join(deltas)
        + "\n\nResolution: for each delta, either (a) the PUT count changed "
        "legitimately — register the new artifact in alpha-engine-config/"
        "private-docs/ARTIFACT_REGISTRY.yaml (or grandfather), then bump "
        "the pinned count; or (b) the change was inadvertent — revert."
    )
