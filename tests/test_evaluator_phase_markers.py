"""Pins evaluate.py's PhaseRegistry instrumentation (alpha-engine-config-I3112
step 1: phase markers for the bundled Evaluator SF state).

The weekly SF 'Evaluator' state runs this whole file as one opaque ~2h spot
task (evaluate.py --mode all --upload). The 2026-07-20 incident (watch-rerun-
2026-07-18-10/-11, a 3600s SIGKILL) showed two failure modes this file's
instrumentation fixes: (1) the killed phase wrote no duration marker, so
post-mortem attribution of the lost hour was impossible; (2) SF-level
progress was opaque — an hour of 'CheckEvaluatorStatus' gave zero visibility
into which internal phase was running.

This reuses backtest.py's existing PhaseRegistry (pipeline_common.py) rather
than inventing a parallel mechanism — markers land in the same
s3://{bucket}/backtest/{date}/.phases/ namespace, "evaluator_"-prefixed so
they never collide with backtest.py's own phase names.

Scope note (verified against live code, not the issue's paraphrase): the
"predictor-analysis phases" (double_sort/horizon_net_alpha/research-free-
backfill/portfolio-optimizer gate) named in #3112's body do NOT run inside
this state at all — they live in backtest.py's predictor_pipeline phase,
which the Evaluator's spot_backtest.sh invocation explicitly skips via
--skip-stages=backtest. ReportCard and Director are ALREADY separate,
already-non-blocking Step Functions states (nousergon-data/infrastructure/
step_function.json — Evaluator -> ... -> ReportCard -> Director, each with
its own Catch->PublishXDegraded advisory path). What genuinely remains
bundled into this one spot task is evaluate.py's own internal diagnostics/
optimizer/champion-promotion/report pipeline — this test pins markers on the
7 heaviest, least-nested of those phases (signal quality, diagnostics,
optimizers, assembler, apply-audit, champion promotion, regression). The
report/upload/email block is deliberately NOT wrapped in this pass — it
already tracks its own completion via `report_ok` + a `write_health` call in
its `finally`, and reindenting its ~400-line try body carried reindentation
risk out of proportion to the marginal observability gain here.

A static source-level check (AST-based, no import) — evaluate.py pulls in
vectorbt/cvxpy/arcticdb at import time, too heavy for a fast structural pin.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_EVALUATE_PATH = _REPO_ROOT / "evaluate.py"

EXPECTED_PHASES = {
    "evaluator_signal_quality",
    "evaluator_diagnostics",
    "evaluator_optimizers",
    "evaluator_assembler",
    "evaluator_apply_audit",
    "evaluator_champion_promotion",
    "evaluator_regression",
}


@pytest.fixture(scope="module")
def tree() -> ast.Module:
    return ast.parse(_EVALUATE_PATH.read_text())


@pytest.fixture(scope="module")
def main_impl(tree) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_main_impl":
            return node
    raise AssertionError("evaluate.py: _main_impl() not found")


def _registry_phase_names(node: ast.AST) -> set[str]:
    """Collect every string literal passed as the first positional arg to a
    `registry.phase(...)` call anywhere under `node` (any nesting depth —
    `with` statements and their `items` are walked like anything else)."""
    names: set[str] = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        if not (isinstance(func, ast.Attribute) and func.attr == "phase"):
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id == "registry"):
            continue
        if sub.args and isinstance(sub.args[0], ast.Constant) and isinstance(sub.args[0].value, str):
            names.add(sub.args[0].value)
    return names


class TestPhaseRegistryImported:
    def test_pipeline_common_imports_phase_registry(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "pipeline_common":
                imported = {alias.name for alias in node.names}
                if "PhaseRegistry" in imported:
                    return
        raise AssertionError("evaluate.py must import PhaseRegistry from pipeline_common")


class TestRegistryConstruction:
    def test_main_impl_constructs_a_registry(self, main_impl):
        assigns = [
            n for n in ast.walk(main_impl)
            if isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "registry"
        ]
        assert assigns, "_main_impl() must assign `registry = PhaseRegistry(...)`"
        call = assigns[0].value
        assert isinstance(call, ast.Call)
        assert isinstance(call.func, ast.Name) and call.func.id == "PhaseRegistry"
        kwarg_names = {kw.arg for kw in call.keywords}
        assert "date" in kwarg_names, "registry must be date-scoped (args.date)"
        assert "bucket" in kwarg_names, "registry must be bucket-scoped"


class TestExpectedPhasesPresent:
    def test_all_expected_evaluator_phases_wrapped(self, main_impl):
        found = _registry_phase_names(main_impl)
        missing = EXPECTED_PHASES - found
        assert not missing, f"missing registry.phase() wrap for: {sorted(missing)}"

    def test_phase_names_are_evaluator_prefixed(self, main_impl):
        # Shared marker namespace with backtest.py's own phases
        # (s3://{bucket}/backtest/{date}/.phases/{name}.json) — every
        # evaluate.py phase name must be "evaluator_"-prefixed so it can
        # never collide with a backtest.py phase name.
        found = _registry_phase_names(main_impl)
        assert found, "no registry.phase() calls found in _main_impl()"
        for name in found:
            assert name.startswith("evaluator_"), (
                f"phase name {name!r} must be 'evaluator_'-prefixed to avoid "
                "colliding with backtest.py's phase namespace"
            )


class TestOptimizerDeferredRaisePreserved:
    """The optimizer stage's error is deliberately swallowed-then-deferred
    (config#1841: apply-audit must still emit before the error re-raises).
    registry.phase() must sit INSIDE that try, not replace it — otherwise
    the deferred-raise contract silently breaks."""

    def test_evaluator_optimizers_phase_is_nested_inside_a_try_except(self, main_impl):
        for node in ast.walk(main_impl):
            if not isinstance(node, ast.Try):
                continue
            phases_in_try = set()
            for stmt in node.body:
                phases_in_try |= _registry_phase_names(stmt)
            if "evaluator_optimizers" in phases_in_try:
                assert node.handlers, (
                    "evaluator_optimizers phase must stay nested inside the "
                    "existing try/except (config#1841 deferred-raise contract)"
                )
                return
        raise AssertionError(
            "evaluator_optimizers phase not found nested inside a try/except"
        )


class TestChampionPromotionSwallowPreserved:
    def test_evaluator_champion_promotion_phase_is_nested_inside_a_try_except(self, main_impl):
        for node in ast.walk(main_impl):
            if not isinstance(node, ast.Try):
                continue
            phases_in_try = set()
            for stmt in node.body:
                phases_in_try |= _registry_phase_names(stmt)
            if "evaluator_champion_promotion" in phases_in_try:
                assert node.handlers, (
                    "evaluator_champion_promotion phase must stay nested inside "
                    "the existing try/except — champion-promotion failure must "
                    "never take down the whole evaluate run"
                )
                return
        raise AssertionError(
            "evaluator_champion_promotion phase not found nested inside a try/except"
        )
