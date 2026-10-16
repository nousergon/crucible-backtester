"""Modules computed from stale score_performance rows are never graded ok.

alpha-engine-config-I11506: on the 2026-10-09 weekly run, signal_quality,
attribution, score_calibration and other diagnostics self-reported ``ok`` on
score_performance rows ending 2026-07-10 — 91 days before the trading day.
Those modules carry no freshness self-check, so the grader passed them. The
evaluator now makes one horizon-aware freshness verdict over score_performance
and grades every consumer ``degraded`` with that reason.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

import evaluate
from completeness import CompletenessTracker

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── the tracker grades a stale input ──────────────────────────────────────


def test_stale_input_grades_an_ok_module_degraded_and_names_it():
    tracker = CompletenessTracker()
    tracker.run_module(
        "m", lambda: {"status": "ok"}, required_inputs={"x": True},
        stale_inputs={"score_performance": "newest measured input is 2026-07-10"},
    )
    rec = tracker.results[0]
    assert rec.status == "degraded"
    assert "stale input score_performance" in rec.degradation_reason
    assert "2026-07-10" in rec.degradation_reason
    assert tracker.degraded_modules() == ["m"]


def test_fresh_input_leaves_the_grade_alone():
    tracker = CompletenessTracker()
    tracker.run_module(
        "m", lambda: {"status": "ok"}, required_inputs={"x": True},
        stale_inputs={"score_performance": None},
    )
    assert tracker.results[0].status == "ok"
    assert tracker.results[0].degradation_reason is None


def test_a_returned_error_outranks_a_stale_input():
    tracker = CompletenessTracker()
    tracker.run_module(
        "m", lambda: {"status": "error", "error": "boom"},
        required_inputs={"x": True},
        stale_inputs={"score_performance": "stale"},
    )
    assert tracker.results[0].status == "error"


# ── the freshness verdict over score_performance ──────────────────────────


def _frame(newest: str, n: int = 30) -> pd.DataFrame:
    dates = pd.date_range(end=newest, periods=n, freq="B")
    return pd.DataFrame({"symbol": ["AAA"] * n, "score_date": dates})


def test_rows_ending_months_before_the_trading_day_are_stale():
    reason = evaluate._score_performance_staleness(_frame("2026-07-10"), "2026-10-09")
    assert reason is not None
    assert "2026-07-10" in reason
    assert "2026-10-09" in reason


def test_rows_lagging_only_by_the_horizon_are_fresh():
    assert evaluate._score_performance_staleness(_frame("2026-10-08"), "2026-10-09") is None


def test_an_empty_frame_is_stale_not_fresh():
    empty = pd.DataFrame({"symbol": [], "score_date": []})
    assert evaluate._score_performance_staleness(empty, "2026-10-09") is not None


@pytest.mark.parametrize("df,run_date", [(None, "2026-10-09"), (_frame("2026-07-10"), None)])
def test_nothing_to_measure_against_is_not_a_staleness_verdict(df, run_date):
    assert evaluate._score_performance_staleness(df, run_date) is None


# ── the pipeline hands the verdict to every consumer ──────────────────────


def test_signal_quality_pipeline_grades_its_consumers_degraded(monkeypatch):
    df = _frame("2026-07-10")
    monkeypatch.setattr(evaluate, "_check_data_freshness", lambda _p: None)
    monkeypatch.setattr(evaluate.signal_quality, "load_score_performance", lambda _p: df)
    monkeypatch.setattr(evaluate, "_compute_signal_quality", lambda _c: {"status": "ok"})
    monkeypatch.setattr(evaluate, "_compute_regime", lambda _c: {"status": "ok", "rows": []})
    monkeypatch.setattr(evaluate.score_analysis, "accuracy_by_threshold", lambda *a, **k: [])
    monkeypatch.setattr(evaluate.attribution, "compute_attribution", lambda _d: {"status": "ok"})
    monkeypatch.setattr(evaluate, "push_predictor_rolling_metrics", lambda *a, **k: None)

    config = {"research_db": "/nonexistent/research.db", "_run_date": "2026-10-09"}
    tracker = CompletenessTracker()
    evaluate._run_signal_quality(config, tracker, {"research_db": True})

    grades = {r.name: r.status for r in tracker.results}
    assert grades == {
        "signal_quality": "degraded",
        "regime_analysis": "degraded",
        "attribution": "degraded",
    }
    assert "2026-07-10" in config["_score_performance_stale"]


def test_fresh_rows_keep_the_pipeline_ok(monkeypatch):
    df = _frame("2026-10-08")
    monkeypatch.setattr(evaluate, "_check_data_freshness", lambda _p: None)
    monkeypatch.setattr(evaluate.signal_quality, "load_score_performance", lambda _p: df)
    monkeypatch.setattr(evaluate, "_compute_signal_quality", lambda _c: {"status": "ok"})
    monkeypatch.setattr(evaluate, "_compute_regime", lambda _c: {"status": "ok", "rows": []})
    monkeypatch.setattr(evaluate.score_analysis, "accuracy_by_threshold", lambda *a, **k: [])
    monkeypatch.setattr(evaluate.attribution, "compute_attribution", lambda _d: {"status": "ok"})
    monkeypatch.setattr(evaluate, "push_predictor_rolling_metrics", lambda *a, **k: None)

    config = {"research_db": "/nonexistent/research.db", "_run_date": "2026-10-09"}
    tracker = CompletenessTracker()
    evaluate._run_signal_quality(config, tracker, {"research_db": True})

    assert {r.status for r in tracker.results} == {"ok"}
    assert config["_score_performance_stale"] is None


#: Diagnostics that read score_performance and carry no freshness self-check.
#: Each must hand the verdict to the grader.
_SCORE_PERFORMANCE_DIAGNOSTICS = {
    "alpha_distribution", "score_calibration", "monte_carlo",
    "factor_blend_sensitivity",
}


def test_score_performance_diagnostics_pass_the_staleness_verdict():
    tree = ast.parse((REPO_ROOT / "evaluate.py").read_text())
    seen: dict[str, bool] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            continue
        name = node.args[0].value
        if name in _SCORE_PERFORMANCE_DIAGNOSTICS:
            seen[name] = any(k.arg == "stale_inputs" for k in node.keywords)
    assert seen == {n: True for n in _SCORE_PERFORMANCE_DIAGNOSTICS}
