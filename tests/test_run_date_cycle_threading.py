"""Every recommendation artifact and evaluator read uses the cycle's run_date.

alpha-engine-config-I11475. The 2026-09-23 weekly rehearsal crossed 00:00 UTC.
EvaluatorOptimize then wrote ``config/scoring_weights/recommendations/2026-09-24/``
but ``config/executor_params/recommendations/2026-09-23/``, so the assembler
reported ``scoring_weights: status=no_artifacts``. It was harmless that week
because the intent was skip, but a real promote filed under the UTC day would
have been silently dropped. config#1017 threaded ``run_date`` through the five
executor_params optimizers (``test_run_date_backfill_threading.py``). The
scoring_weights, research_params and predictor_params (veto) producers were
never threaded, so they fell back to ``today_iso()``, which is the box's wall
clock. The same rehearsal's smoke evaluator also probed
``backtest/2026-09-24/attestation.json``, because ``evaluate.py --smoke`` was
launched with no ``--date``.

These tests pin every link:
  * each decision path of the three apply() functions forwards run_date;
  * evaluate.py hands each of them ``config["_run_date"]``;
  * both evaluator-stage smoke launches pass ``--date``;
  * launcher banners print the cycle date, never ``$(date +%Y-%m-%d)``.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from optimizer.assembler import set_cutover_enabled

_ROOT = Path(__file__).resolve().parent.parent
_INFRA = _ROOT / "infrastructure"
CYCLE_DATE = "2026-09-23"

#: (module file, module import path, apply function name)
_APPLIES = (
    ("optimizer/weight_optimizer.py", "optimizer.weight_optimizer", "apply_weights"),
    ("optimizer/research_optimizer.py", "optimizer.research_optimizer", "apply"),
    ("analysis/veto_analysis.py", "analysis.veto_analysis", "apply"),
)


@pytest.fixture(autouse=True)
def _reset_cutover_flag():
    set_cutover_enabled(False)
    yield
    set_cutover_enabled(False)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


@pytest.mark.parametrize(("path", "_mod", "fn"), _APPLIES)
def test_every_produce_artifact_call_in_apply_forwards_run_date(path, _mod, fn):
    """Structural pin across every decision path: promote, skip, shadow and
    blocked. A new early-return that forgets run_date fails here, where a
    behavioural test would only catch the paths it happens to drive."""
    func = _function(ast.parse((_ROOT / path).read_text()), fn)
    assert "run_date" in [a.arg for a in func.args.args], f"{fn} takes no run_date"
    calls = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "produce_artifact"
    ]
    assert calls, f"{path}::{fn} calls produce_artifact nowhere"
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "run_date" in kw, (
            f"{path}:{call.lineno} produce_artifact() without run_date"
        )
        assert (
            isinstance(kw["run_date"], ast.Name) and kw["run_date"].id == "run_date"
        ), f"{path}:{call.lineno} forwards something other than apply()'s run_date"


@pytest.mark.parametrize(
    ("mod_name", "fn", "result"),
    [
        (
            "optimizer.weight_optimizer",
            "apply_weights",
            {"status": "insufficient_data"},
        ),
        (
            "optimizer.weight_optimizer",
            "apply_weights",
            {"status": "ok", "oos_passed": False},
        ),
        (
            "optimizer.weight_optimizer",
            "apply_weights",
            {"status": "ok", "confidence": "low"},
        ),
        ("optimizer.research_optimizer", "apply", {"status": "no_improvement"}),
        (
            "optimizer.research_optimizer",
            "apply",
            {"status": "ok", "recommended_params": {}},
        ),
        ("analysis.veto_analysis", "apply", {"status": "insufficient_data"}),
    ],
)
def test_apply_forwards_run_date_on_skip_paths(mod_name, fn, result):
    """The rehearsal's own path: a skip keyed under the UTC day is what the
    assembler failed to find."""
    mod = __import__(mod_name, fromlist=[fn])
    with patch(f"{mod_name}.produce_artifact") as produce:
        produce.return_value = {"written": True, "key": "k", "run_id": "r"}
        getattr(mod, fn)(result, "test-bucket", CYCLE_DATE)
    assert produce.call_args.kwargs.get("run_date") == CYCLE_DATE


@pytest.mark.parametrize(("mod_name", "fn"), [(m, f) for _p, m, f in _APPLIES])
def test_apply_run_date_defaults_to_none(mod_name, fn):
    """Backwards compatible: existing two-argument callers keep the
    today_iso() fallback inside produce_artifact."""
    mod = __import__(mod_name, fromlist=[fn])
    with patch(f"{mod_name}.produce_artifact") as produce:
        produce.return_value = {"written": True, "key": "k", "run_id": "r"}
        getattr(mod, fn)({"status": "insufficient_data"}, "test-bucket")
    assert produce.call_args.kwargs.get("run_date") is None


@pytest.mark.parametrize(
    "callee",
    [
        "weight_optimizer.apply_weights",
        "veto_analysis.apply",
        "research_optimizer.apply",
    ],
)
def test_evaluate_passes_the_cycle_run_date_to_each_apply(callee):
    tree = ast.parse((_ROOT / "evaluate.py").read_text())
    owner, attr = callee.split(".")
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == owner
    ]
    assert calls, f"evaluate.py never calls {callee}"
    for call in calls:
        kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
        assert kw.get("run_date") == "config.get('_run_date')", (
            f"evaluate.py:{call.lineno} {callee}() does not pass config['_run_date']"
        )


def test_weight_apply_end_to_end_keys_the_cycle_partition():
    """Through the real produce_artifact: the S3 key carries the cycle date."""
    from optimizer.weight_optimizer import apply_weights

    with (
        patch("optimizer.recommendation_artifact.boto3") as boto3_mock,
        patch("optimizer.recommendation_artifact.today_iso", return_value="2026-09-24"),
    ):
        s3 = MagicMock()
        boto3_mock.client.return_value = s3
        apply_weights({"status": "insufficient_data"}, "test-bucket", CYCLE_DATE)
    key = s3.put_object.call_args.kwargs["Key"]
    assert (
        key
        == f"config/scoring_weights/recommendations/{CYCLE_DATE}/from_weight_optimizer.json"
    )


# ── launchers ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("script", ["spot_evaluator.sh", "spot_backtest.sh"])
def test_every_smoke_evaluator_launch_passes_the_run_date(script):
    lines = [
        line
        for line in (_INFRA / script).read_text().splitlines()
        if "evaluate.py --smoke" in line
        and "echo" not in line
        and not line.lstrip().startswith("#")
    ]
    assert lines, f"{script}: no evaluate.py --smoke launch found"
    for line in lines:
        assert re.search(r'--date "\\?\$\{RUN_DATE\}"', line), line


def _banner_scripts() -> list[Path]:
    return sorted(
        p
        for p in _INFRA.glob("*.sh")
        if re.search(r'echo "  [^"\n]*Spot Run', p.read_text())
    )


def test_banner_scripts_are_enumerated():
    assert len(_banner_scripts()) >= 10


@pytest.mark.parametrize("script", _banner_scripts(), ids=lambda p: p.name)
def test_launcher_banner_never_prints_the_utc_day(script):
    for line in script.read_text().splitlines():
        if re.search(r'echo "  [^"\n]*Spot Run', line):
            assert "$(date" not in line, f"{script.name}: {line.strip()}"
            assert "$(spot_common_stage_run_date)" in line or "${RUN_DATE}" in line, (
                line
            )


def _stage_run_date(env_extra: dict[str, str], drop: tuple[str, ...] = ()) -> str:
    text = (_INFRA / "_spot_common.sh").read_text()
    m = re.search(r"^spot_common_stage_run_date\(\) \{\n.*?^\}\n", text, re.S | re.M)
    assert m, "spot_common_stage_run_date() not found"
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(env_extra)
    proc = subprocess.run(  # noqa: S603 -- fixed argv
        ["bash", "-c", m.group(0) + "spot_common_stage_run_date"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return proc.stdout.strip()


def test_stage_run_date_prefers_the_sf_run_date():
    assert _stage_run_date({"EXECUTION_RUN_DATE": CYCLE_DATE}) == CYCLE_DATE


def test_stage_run_date_falls_back_to_the_exchange_day():
    before = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    got = _stage_run_date({"TZ": "Pacific/Kiritimati"}, drop=("EXECUTION_RUN_DATE",))
    after = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    assert got in {before, after}
