"""Regression guard: the simulate auto-skip reload must pick
``portfolio_stats.json`` by name, never by position (alpha-engine-config-I11618).

The simulate phase records two artifacts, ``portfolio_daily_returns.parquet``
(config-I7616) and ``portfolio_stats.json``, and the phase marker stores its
``artifact_keys`` ``sorted()``, so the parquet always sorts first. The reload
took ``keys[0]`` and handed parquet bytes to ``json.loads``. Measured on the
2026-09-26 weekly retry: ``UnicodeDecodeError: 'utf-8' codec can't decode byte
0xd0 in position 7`` at ``backtest.py`` ``load_json(bucket, keys[0], ...)``.
The live ``s3://alpha-engine-research/backtest/2026-09-25/.phases/simulate.json``
marker lists the parquet first, and its first bytes are ``PAR1\\x15\\x04\\x15\\xd0``.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pandas as pd
import pytest

import backtest as bt
from phase_artifacts import load_json, save_json, save_series
from pipeline_common import PhaseRegistry
from tests.test_phase_registry import _FakeS3

DATE = "2026-09-25"


def _record_simulate_phase(s3) -> None:
    """Run the simulate phase's two writes through a real PhaseRegistry, in
    the same order backtest.py records them."""
    registry = PhaseRegistry(date=DATE, bucket="b", s3_client=s3)
    with registry.phase("simulate", supports_auto_skip=True) as ctx:
        idx = pd.date_range("2026-06-01", periods=30, freq="B")
        ctx.record_artifact(save_series(
            "b", DATE, "simulate", "portfolio_daily_returns",
            pd.Series(0.001, index=idx, name="daily_returns"), s3_client=s3,
        ))
        ctx.record_artifact(save_json(
            "b", DATE, "simulate", "portfolio_stats",
            {"status": "ok", "total_return": 0.05}, s3_client=s3,
        ))


def _retry_marker_keys(s3) -> list[str]:
    registry = PhaseRegistry(date=DATE, bucket="b", s3_client=s3)
    run, _reason = registry.should_run("simulate", supports_auto_skip=True)
    assert run is False, "a completed simulate phase must auto-skip on retry"
    return registry.load_marker("simulate")["artifact_keys"]


def test_marker_lists_the_parquet_first():
    """The precondition of the defect: positional reload reads the parquet."""
    s3 = _FakeS3()
    _record_simulate_phase(s3)
    keys = _retry_marker_keys(s3)
    assert keys[0].endswith("/portfolio_daily_returns.parquet")
    with pytest.raises(UnicodeDecodeError):
        load_json("b", keys[0], s3_client=s3)


def test_reload_selects_portfolio_stats_by_name():
    s3 = _FakeS3()
    _record_simulate_phase(s3)
    keys = _retry_marker_keys(s3)
    stats = load_json("b", bt._simulate_stats_key(keys), s3_client=s3)
    assert stats == {"status": "ok", "total_return": 0.05}


def test_reload_is_order_independent():
    keys = [
        f"backtest/{DATE}/.phases/simulate/portfolio_stats.json",
        f"backtest/{DATE}/.phases/simulate/portfolio_daily_returns.parquet",
    ]
    assert bt._simulate_stats_key(keys) == keys[0]
    assert bt._simulate_stats_key(list(reversed(keys))) == keys[0]


@pytest.mark.parametrize("keys", [
    [f"backtest/{DATE}/.phases/simulate/portfolio_daily_returns.parquet"],
    [
        f"backtest/{DATE}/.phases/simulate/portfolio_stats.json",
        f"backtest/{DATE}/.phases/.rehearsal/r1/simulate/portfolio_stats.json",
    ],
])
def test_reload_fails_loud_without_exactly_one_stats_artifact(keys):
    with pytest.raises(RuntimeError, match="cannot reload portfolio_stats"):
        bt._simulate_stats_key(keys)


def test_simulate_auto_skip_branch_uses_the_named_selector():
    """The pipeline's own auto-skip branch must go through the selector, not
    index the marker's key list."""
    src = textwrap.dedent(inspect.getsource(bt._run_simulation_pipeline))
    tree = ast.parse(src)
    stats_loads = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "portfolio_stats" for t in node.targets)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", None) == "load_json"
    ]
    assert len(stats_loads) == 1, "expected exactly one portfolio_stats reload"
    key_arg = stats_loads[0].value.args[1]
    assert isinstance(key_arg, ast.Call) and key_arg.func.id == "_simulate_stats_key"
