"""
tests/test_evaluate_run_module_signature.py — preflight for evaluate.py
diagnostic call shape.

Statically asserts every `tracker.run_module(...)` call site in evaluate.py
passes `required_inputs` as a keyword argument. CompletenessTracker.run_module()
makes `required_inputs` mandatory (no default); a missing kwarg raises TypeError
the first time the call site executes. The 2026-05-07 Sat-SF Evaluator failure
hit exactly this — `decision_capture_coverage` and `provenance_grounding` were
both added without `required_inputs={}`, neither had a unit test, and the bug
only surfaced when production tried to run the module.

Why static AST instead of a runtime mock: invoking _run_diagnostics requires
fixtures for research.db, S3, predictor metrics, etc. that drift over time. A
static check doesn't need any of that and runs in <50ms — ideal for catching
this exact regression class at PR time.
"""

import ast
import inspect
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATE_PY = REPO_ROOT / "evaluate.py"


def _collect_run_module_calls(tree: ast.AST) -> list[ast.Call]:
    """Find every `<obj>.run_module(...)` call regardless of receiver name.

    Captures both `tracker.run_module(...)` and any other receiver shape.
    """
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "run_module":
            calls.append(node)
    return calls


def test_evaluate_py_run_module_calls_pass_required_inputs():
    """Every tracker.run_module() call must supply required_inputs as kwarg.

    Regression target: 2026-05-07 Evaluator SF failure where two diagnostic
    modules added without `required_inputs={}` caused TypeError at runtime.
    Both call sites are inside _run_diagnostics so a deploy that skips that
    function (e.g. --mode optimize-only) wouldn't surface the bug — this
    static test catches it whether or not the path executes in CI fixtures.
    """
    source = EVALUATE_PY.read_text()
    tree = ast.parse(source)
    calls = _collect_run_module_calls(tree)

    # Sanity: evaluate.py has many diagnostic registrations; if this drops to
    # zero, the test is silently passing on a moved API.
    assert len(calls) >= 15, (
        f"Expected many run_module() call sites in evaluate.py; found "
        f"{len(calls)}. Did the API move?"
    )

    missing: list[tuple[int, str]] = []
    for call in calls:
        kwarg_names = {kw.arg for kw in call.keywords if kw.arg}
        if "required_inputs" not in kwarg_names:
            # Reconstruct the module name (the first positional arg, a string
            # literal in every existing call site) for a useful failure
            # message.
            module_name = "<unknown>"
            if call.args and isinstance(call.args[0], ast.Constant):
                module_name = repr(call.args[0].value)
            missing.append((call.lineno, module_name))

    assert not missing, (
        "evaluate.py has tracker.run_module() calls missing the required "
        "`required_inputs` kwarg — this raises TypeError at runtime. Add "
        "`required_inputs={}` for S3-only modules or "
        "`required_inputs={'<input>': avail['<input>']}` for DB-backed ones.\n"
        "Offending sites:\n  "
        + "\n  ".join(f"line {lineno}: {name}" for lineno, name in missing)
    )


def test_completeness_tracker_run_module_signature_is_stable():
    """If CompletenessTracker.run_module() ever gains a default for
    required_inputs, this preflight test becomes redundant — flag it so we
    can simplify."""
    from completeness import CompletenessTracker

    sig = inspect.signature(CompletenessTracker.run_module)
    param = sig.parameters["required_inputs"]
    assert param.default is inspect.Parameter.empty, (
        "CompletenessTracker.run_module() now defaults required_inputs — "
        "the static evaluate.py preflight test is redundant and can be "
        "removed (or kept as documentation-of-intent)."
    )
