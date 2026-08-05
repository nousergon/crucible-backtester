"""Pins evaluate.py's config-I3112 SF-split wiring (S3 snapshot handoff +
optimize-half-only outward artifacts).

The weekly SF 'Evaluator' state is decomposed into EvaluatorDiagnostics
(--mode diagnostics) and EvaluatorOptimize (--mode optimize). This test pins
the evaluate.py side of that contract:

  * --mode diagnostics --upload writes the S3 snapshot (diagnostics dict +
    signal-quality outputs + df_base) right after _run_diagnostics — the
    single-producer write the optimize half reads.
  * --mode optimize standalone loads that snapshot BEFORE the optimizers
    (replacing the pre-split empty-dict/None-df_base defaults that silently
    skipped the weight/veto/research/pillar optimizers).
  * The report/upload/email tail is the OPTIMIZE half's deliverable: the
    terminal S3 upload, grade-history append, and digest email are all gated
    on run_optimizers, so the diagnostics half never lands a partial report
    or a misleading digest that would win the run_date dedup race.

A static source-level check (AST-based, no import) — evaluate.py pulls in
vectorbt/cvxpy/arcticdb at import time, too heavy for a fast structural pin
(same rationale as test_evaluator_phase_markers.py).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_EVALUATE_PATH = _REPO_ROOT / "evaluate.py"
_HANDOFF_PATH = _REPO_ROOT / "evaluate_handoff.py"


@pytest.fixture(scope="module")
def tree() -> ast.Module:
    return ast.parse(_EVALUATE_PATH.read_text())


@pytest.fixture(scope="module")
def main_impl(tree) -> ast.FunctionDef:
    # main() is a thin guard wrapper; the pipeline lives in _main_impl
    # (same structure the phase-marker test pins against).
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_main_impl":
            return node
    raise AssertionError("_main_impl() not found")


class TestSnapshotWriteInDiagnosticsHalf:
    def test_handoff_module_imported(self, tree):
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        assert "evaluate_handoff" in imports, (
            "evaluate.py must import evaluate_handoff — the S3 snapshot "
            "handoff lives there"
        )

    def test_write_snapshot_called_inside_evaluator_diagnostics_phase(self, main_impl):
        """The write must sit inside the `evaluator_diagnostics` PhaseRegistry
        block so a FAIL-LOUD write error fails the diagnostics half loudly
        (the optimize half depends on the artifact existing)."""
        writes = [
            n
            for n in ast.walk(main_impl)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "write_snapshot"
        ]
        assert len(writes) == 1, (
            "exactly one write_snapshot call expected in main()"
        )
        write = writes[0]
        # The call must be within the `with registry.phase(...)` context of
        # the diagnostics block.
        phase_names = []
        for parent in _parents(main_impl, write):
            if isinstance(parent, ast.With):
                for item in parent.items:
                    phase_names.extend(_phase_names(item.context_expr))
        assert "evaluator_diagnostics" in phase_names, (
            "write_snapshot must be inside the evaluator_diagnostics phase block"
        )

    def test_write_rides_upload_gate(self, main_impl):
        """--freeze / local runs must not persist the snapshot (sibling
        artifact convention: build + log without persisting)."""
        writes = [
            n
            for n in ast.walk(main_impl)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "write_snapshot"
        ]
        assert len(writes) == 1
        gate = _nearest_if(writes[0], main_impl)
        assert gate is not None
        assert _has_upload_test(gate.test), (
            "write_snapshot must be gated on args.upload"
        )

    def test_write_passes_full_snapshot_payload(self, main_impl):
        """The optimize half needs the diagnostics dict AND the
        signal-quality outputs AND df_base — an empty-diagnostics-only
        handoff would still silently skip the df_base-dependent optimizers."""
        writes = [
            n
            for n in ast.walk(main_impl)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "write_snapshot"
        ]
        assert len(writes) == 1
        kwargs = {
            kw.arg: ast.dump(kw.value)
            for kw in writes[0].keywords
            if kw.arg is not None
        }
        for expected in (
            "diagnostics", "sq_result", "regime_rows", "score_rows",
            "attr_result", "df_base",
        ):
            assert expected in kwargs, f"write_snapshot missing {expected} kwarg"


class TestSnapshotReadInOptimizeHalf:
    def test_load_snapshot_called_before_optimizers(self, main_impl):
        loads = [
            n
            for n in ast.walk(main_impl)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "load_snapshot"
        ]
        assert len(loads) == 1, "exactly one load_snapshot call expected in main()"
        load = loads[0]
        # The load must precede the optimizer block's write to `diagnostics`.
        opt_stage_line = _line_of_phase(main_impl, "evaluator_optimizers")
        assert load.lineno < opt_stage_line, (
            "load_snapshot must run before the optimizer stage so the "
            "loaded diagnostics reach _run_optimizers"
        )

    def test_load_restores_signal_quality_outputs(self, main_impl):
        loads = [
            n
            for n in ast.walk(main_impl)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "load_snapshot"
        ]
        assert len(loads) == 1
        # After the load, the five downstream inputs must be re-bound from
        # the snapshot (not left at their empty defaults).
        after = _nodes_after(main_impl, loads[0])
        for name in ("sq_result", "regime_rows", "score_rows", "attr_result", "df_base"):
            assigns = [
                n for n in after
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
            ]
            assert any(
                isinstance(a.value, ast.Subscript)
                and isinstance(a.value.value, ast.Name)
                and a.value.value.id == "snapshot"
                for a in assigns
            ), f"after load_snapshot, {name} must be re-bound from the snapshot"


class TestReportTailIsOptimizeHalfDeliverable:
    def test_terminal_upload_gated_on_run_optimizers(self, main_impl):
        uploads = [
            n
            for n in ast.walk(main_impl)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "upload_to_s3"
        ]
        assert uploads, "upload_to_s3 call expected"
        assert _any_enclosing_if_has(main_impl, uploads[0], "run_optimizers"), (
            "the terminal report upload must be gated on run_optimizers — "
            "the diagnostics half must not upload a partial report"
        )

    def test_grade_history_gated_on_run_optimizers(self, main_impl):
        appends = [
            n
            for n in ast.walk(main_impl)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "append_grades"
        ]
        assert appends, "append_grades call expected"
        assert _any_enclosing_if_has(main_impl, appends[0], "run_optimizers"), (
            "append_grades must be gated on run_optimizers — the diagnostics "
            "half must not land a partial grade history entry"
        )

    def test_digest_email_gated_on_run_optimizers(self, main_impl):
        sends = [
            n
            for n in ast.walk(main_impl)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "send_digest_email"
        ]
        assert sends, "send_digest_email call expected"
        assert _any_enclosing_if_has(main_impl, sends[0], "run_optimizers"), (
            "the digest email must be gated on run_optimizers — an "
            "empty-opt_results digest from the diagnostics half would win "
            "the run_date dedup race against the optimize half's real one"
        )


class TestHandoffModuleContract:
    def test_handoff_module_loads_cleanly(self):
        """The module must import standalone (it is imported by evaluate.py
        at module top)."""
        import evaluate_handoff  # noqa: F401


# ── AST helpers ─────────────────────────────────────────────────────────────


def _parents(root: ast.AST, node: ast.AST) -> list[ast.AST]:
    """All ancestors of *node* within *root*, nearest first."""
    parent_map: dict[int, ast.AST] = {}
    for parent in ast.walk(root):
        for child in ast.iter_child_nodes(parent):
            parent_map[id(child)] = parent
    out = []
    cur = node
    while id(cur) in parent_map:
        cur = parent_map[id(cur)]
        out.append(cur)
    return out


def _phase_names(expr: ast.AST) -> list[str]:
    """Collect literal phase names from a `registry.phase("name")` call."""
    names = []
    for node in ast.walk(expr):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.append(node.value)
    return names


def _line_of_phase(main_impl: ast.FunctionDef, phase_name: str) -> int:
    for node in ast.walk(main_impl):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if getattr(node.func, "attr", None) == "phase":
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and arg.value == phase_name:
                        return node.lineno
    raise AssertionError(f"phase {phase_name} not found")


def _nearest_if(node: ast.AST, main_impl: ast.FunctionDef) -> ast.If | None:
    """Find the innermost enclosing `if` of *node*."""
    candidates = [
        n for n in ast.walk(main_impl)
        if isinstance(n, ast.If)
        and node.lineno >= n.lineno
        and (n.body or n.orelse)
        and _contains_line(n, node.lineno)
    ]
    if not candidates:
        return None
    # Innermost = the one whose block starts latest.
    return max(candidates, key=lambda n: n.lineno)


def _nodes_after(root: ast.AST, node: ast.AST) -> list[ast.AST]:
    """All nodes in *root* strictly after *node* (by source line)."""
    return [
        n for n in ast.walk(root)
        if getattr(n, "lineno", None) is not None and n.lineno > node.lineno
    ]


def _any_enclosing_if_has(
    main_impl: ast.FunctionDef, node: ast.AST, name: str
) -> bool:
    """True if ANY enclosing `if` of *node* tests *name* (e.g. the upload
    gate's outer `args.upload and run_optimizers` where the call sits under
    a nested `if grading_result...`)."""
    for parent in _parents(main_impl, node):
        if isinstance(parent, ast.If):
            if _has_name(parent.test, name):
                return True
    return False


def _contains_line(node: ast.AST, lineno: int) -> bool:
    for child in ast.walk(node):
        if getattr(child, "lineno", None) == lineno:
            return True
    return False


def _has_upload_test(test: ast.AST) -> bool:
    return _has_name_attr(test, "upload")


def _has_run_optimizers_test(test: ast.AST) -> bool:
    return _has_name(test, "run_optimizers")


def _has_name(test: ast.AST, name: str) -> bool:
    for node in ast.walk(test):
        if isinstance(node, ast.Name) and node.id == name:
            return True
    return False


def _has_name_attr(test: ast.AST, attr: str) -> bool:
    for node in ast.walk(test):
        if isinstance(node, ast.Attribute) and node.attr == attr:
            return True
    return False
