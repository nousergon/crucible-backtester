"""Silent no-ops are never graded ok (alpha-engine-config-I11506).

The 2026-09-23 rehearsal's EvaluatorDiagnostics logged ``[OK]`` for modules
that measured nothing: stale sources, 0/0 features compared, 136/136
unparseable snapshots, a regression check that did not run. The completeness
tracker graded every module that RETURNED as ok, whatever the module said
about itself. These tests pin the tracker, the frozen regression path, the
coverage funnel and the SPY warning.
"""
from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pandas as pd
import pytest

from completeness import CompletenessTracker, grade_self_reported

# ── the tracker reads the module's own verdict ────────────────────────────


@pytest.mark.parametrize("output,grade", [
    ({"status": "ok"}, "ok"),
    ({"no_status": 1}, "ok"),
    ("not a dict", "ok"),
    # computed decisions on real inputs stay ok
    ({"status": "no_change"}, "ok"),
    ({"status": "no_improvement"}, "ok"),
    ({"status": "blocked"}, "ok"),
    ({"status": "alpha_below_floor"}, "ok"),
    ({"status": "retired"}, "ok"),
    # ran on nothing / on stale inputs
    ({"status": "insufficient_data"}, "degraded"),
    ({"status": "insufficient_samples"}, "degraded"),
    ({"status": "skipped", "reason": "no_db"}, "degraded"),
    ({"status": "stale_sources"}, "degraded"),
    ({"status": "stale_baseline"}, "degraded"),
    ({"status": "no_data"}, "degraded"),
    ({"status": "no_signals"}, "degraded"),
    ({"status": "no_comparable_features"}, "degraded"),
    ({"status": "partial"}, "degraded"),
    ({"status": "degraded"}, "degraded"),
    ({"status": "alpha_unmeasured"}, "degraded"),
    ({"status": "snapshots_unparseable"}, "degraded"),
    ({"status": "stance_column_absent"}, "degraded"),
    ({"status": "unmeasurable"}, "degraded"),
    ({"status": "unservable"}, "degraded"),
    ({"status": "vacuous"}, "degraded"),
    ({"status": "not_run"}, "degraded"),
    # a failure returned rather than raised
    ({"status": "error", "error": "boom"}, "error"),
])
def test_run_module_grades_the_self_reported_status(output, grade):
    tracker = CompletenessTracker()
    tracker.run_module("m", lambda: output, required_inputs={"x": True})
    rec = tracker.results[0]
    assert rec.status == grade
    if grade != "ok":
        assert f"status={output['status']}" in rec.degradation_reason


def test_reason_carries_the_module_detail():
    grade, reason = grade_self_reported({"status": "skipped", "reason": "no_db"})
    assert grade == "degraded"
    assert "no_db" in reason


def test_regime_sortino_unmeasurable_is_degraded_with_its_reason():
    """The T2 runner's stale-input verdict (gap (a) on I11506): the lib's
    own constant, not a literal, so a renamed status fails here."""
    from analysis.regime_stratified_sortino import STATUS_OK, STATUS_UNMEASURABLE

    reason = "newest measured input is 2026-07-10 — 91 calendar days before"
    tracker = CompletenessTracker()
    tracker.run_module(
        "regime_stratified_sortino",
        lambda: {"status": STATUS_UNMEASURABLE, "status_reason": reason},
        required_inputs={"research_db": True},
    )
    rec = tracker.results[0]
    assert rec.status == "degraded"
    assert reason in rec.degradation_reason
    assert grade_self_reported({"status": STATUS_OK}) == (None, None)


def test_missing_input_and_self_report_both_named():
    tracker = CompletenessTracker()
    tracker.run_module(
        "m", lambda: {"status": "insufficient_data"},
        required_inputs={"trades_db": False},
    )
    rec = tracker.results[0]
    assert rec.status == "degraded"
    assert "ran without: trades_db" in rec.degradation_reason
    assert "status=insufficient_data" in rec.degradation_reason
    assert tracker.summary()["degraded"] == 1


# ── the evaluator's regression wrapper ────────────────────────────────────


def test_frozen_regression_is_skipped_not_ok(monkeypatch):
    import evaluate
    from optimizer import regression_monitor

    monkeypatch.setattr(regression_monitor, "save_rolling_metrics", lambda *a, **k: None)
    tracker = CompletenessTracker()
    out = evaluate._run_regression(
        {"signals_bucket": "b"}, tracker,
        sq_result={"status": "ok", "overall": {"accuracy_21d": 0.5, "n_21d": 40}},
        portfolio_stats={"sortino_ratio": 1.0, "total_trades": 40},
        weight_result=None, executor_rec=None, veto_result=None,
        freeze=True, run_date="2026-09-23",
    )
    assert out == {"status": "skipped", "reason": "frozen run"}
    assert tracker.results[0].status == "degraded"


# ── measurement coverage: an all-HOLD file is named, and not ok ────────────


def test_all_hold_signals_are_named_and_degraded(tmp_path):
    from analysis.measurement_coverage import compute_measurement_coverage

    signals = {"signals": {t: {"signal": "HOLD"} for t in ("AAA", "BBB", "CCC")}}
    s3 = MagicMock()
    s3.get_object.side_effect = lambda Bucket, Key: {
        "Body": MagicMock(read=lambda: json.dumps(
            signals if Key.startswith("signals/") else {"predictions": []}
        ).encode())
    }
    out = compute_measurement_coverage(
        run_date="2026-09-23", trades_db_path=None, s3_client=s3,
    )
    assert out["status"] == "no_signals"
    assert "3 signal(s), 0 ENTER (HOLD=3)" in out["reason"]
    assert grade_self_reported(out)[0] == "degraded"


# ── SPY: harnesses opt out of the warning, real consumers do not ──────────


def _pf():
    from vectorbt_bridge import orders_to_portfolio

    idx = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame({"AAA": [100., 110., 120., 130., 140.]}, index=idx)
    orders = [
        {"date": "2024-01-01", "ticker": "AAA", "action": "ENTER",
         "shares": 10, "price_at_order": 100.},
        {"date": "2024-01-04", "ticker": "AAA", "action": "EXIT",
         "shares": 10, "price_at_order": 130.},
    ]
    return orders_to_portfolio(orders, prices, init_cash=10_000, fees=0.0)


def test_missing_spy_still_warns_by_default(caplog):
    from vectorbt_bridge import portfolio_stats

    with caplog.at_level(logging.WARNING, logger="vectorbt_bridge"):
        stats = portfolio_stats(_pf())
    assert "spy_prices not provided" in caplog.text
    assert "total_alpha" in stats["null_legs"]


def test_harness_opt_out_quiets_the_warning_but_keeps_null_legs(caplog):
    from vectorbt_bridge import portfolio_stats

    with caplog.at_level(logging.WARNING, logger="vectorbt_bridge"):
        stats = portfolio_stats(_pf(), spy_expected=False)
    assert "spy_prices not provided" not in caplog.text
    assert stats["total_alpha"] is None
    assert "total_alpha" in stats["null_legs"]


def test_attestation_and_self_test_emit_no_spy_warning(caplog):
    """All 28 'spy_prices not provided' warnings in the rehearsal log came from
    these two known-answer harnesses, not from a real alpha computation."""
    from analysis import attestation, self_test

    with caplog.at_level(logging.WARNING, logger="vectorbt_bridge"):
        attestation._pnl_no_fees()
        attestation._fee_charged_both_sides()
        attestation._drawdown_peak_to_trough()
        self_test._run(
            pd.DataFrame({"AAA": [10., 11.]}, index=pd.bdate_range("2024-01-01", periods=2)),
            [{"date": "2024-01-01", "ticker": "AAA", "action": "ENTER",
              "shares": 1, "price_at_order": 10.}],
        )
    assert "spy_prices not provided" not in caplog.text


# ── a module that returned nothing did nothing (I11506) ───────────────────


@pytest.mark.parametrize("output", [None, {}, [], ""])
def test_empty_output_is_graded_degraded_not_ok(output):
    """``None`` / ``{}`` / ``[]`` used to grade ``[OK]``; a None was handed
    back to the caller as a fabricated ``{"status": "ok"}``."""
    tracker = CompletenessTracker()
    returned = tracker.run_module("m", lambda: output, required_inputs={"x": True})
    rec = tracker.results[0]
    assert rec.status == "degraded"
    assert "module returned no output" in rec.degradation_reason
    assert returned.get("status") != "ok"
    # whatever the caller gets back re-grades as degraded too
    if not isinstance(output, dict):
        assert grade_self_reported(returned)[0] == "degraded"


def test_empty_output_does_not_mask_an_error_or_a_missing_input():
    tracker = CompletenessTracker()
    tracker.run_module("m", lambda: None, required_inputs={"x": False})
    rec = tracker.results[0]
    assert rec.status == "degraded"
    assert "ran without: x" in rec.degradation_reason
    assert "module returned no output" in rec.degradation_reason


def test_non_empty_output_is_unchanged():
    tracker = CompletenessTracker()
    out = {"status": "ok", "rows": [1]}
    assert tracker.run_module("m", lambda: out, required_inputs={}) is out
    assert tracker.results[0].status == "ok"


def test_regime_analysis_with_zero_rows_is_not_ok(monkeypatch):
    """Zero regime buckets reaching min_samples is a measurement of nothing."""
    import evaluate

    monkeypatch.setattr(evaluate.regime_analysis, "load_with_regime", lambda _p: pd.DataFrame())
    monkeypatch.setattr(evaluate.regime_analysis, "accuracy_by_regime", lambda _d, min_samples=5: [])
    tracker = CompletenessTracker()
    out = tracker.run_module(
        "regime_analysis", lambda: evaluate._compute_regime({"research_db": "x.db"}),
        required_inputs={"research_db": True},
    )
    assert out["rows"] == []
    assert tracker.results[0].status == "degraded"
    assert "insufficient_samples" in tracker.results[0].degradation_reason

    monkeypatch.setattr(
        evaluate.regime_analysis, "accuracy_by_regime",
        lambda _d, min_samples=5: [{"regime": "bull", "n": 9}],
    )
    tracker = CompletenessTracker()
    tracker.run_module(
        "regime_analysis", lambda: evaluate._compute_regime({"research_db": "x.db"}),
        required_inputs={"research_db": True},
    )
    assert tracker.results[0].status == "ok"
