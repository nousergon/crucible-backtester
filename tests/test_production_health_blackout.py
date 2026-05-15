"""Pin the post-cutover blackout classification for BOTH the IC path
(compute_production_health) and the ECE path (compute_calibration_validation).

Two false-positive retrain alerts trace here:

  * 2026-05-15 HIGH ic_degradation — stale Lambda + in-sample denominator
    (fixed in #209/#210; the post-#180 strict horizon filter no-ops it).
  * 2026-05-15 MEDIUM calibration_breakdown — `horizon_days = 21` does NOT
    isolate the post-cutover model: the grader stamps it at grade time, so
    PRE-cutover-model predictions whose 21d window closed post-migration
    also carry it (415 such rows pooled a stale population into the ECE).

Fix: production analytics scope to `prediction_date >= CANONICAL_CUTOVER_DATE`
(POST_CUTOVER_FILTER_SQL), and both compute paths share
`_classify_skipped_window` so IC and ECE report the blackout identically.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from pipeline_common import CANONICAL_CUTOVER_DATE  # "2026-05-09"

_COLS = (
    "symbol, prediction_date, predicted_direction, prediction_confidence, "
    "p_up, p_flat, p_down, score_modifier_applied, actual_5d_return, "
    "correct_5d, actual_log_alpha, horizon_days, correct"
)
_PRE = "2026-05-05"   # < cutover (2026-05-09)
_POST = "2026-05-12"  # >= cutover


def _make_db(tmp_path, rows: list[tuple]) -> str:
    db = tmp_path / "research.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE predictor_outcomes ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, prediction_date TEXT, "
        "predicted_direction TEXT, prediction_confidence REAL, p_up REAL, "
        "p_flat REAL, p_down REAL, score_modifier_applied REAL, "
        "actual_5d_return REAL, correct_5d INTEGER, actual_log_alpha REAL, "
        "horizon_days REAL, correct INTEGER)"
    )
    conn.executemany(
        f"INSERT INTO predictor_outcomes ({_COLS}) VALUES ({','.join('?' * 13)})",
        rows,
    )
    conn.commit()
    conn.close()
    return str(db)


def _row(i: int, pred_date: str, horizon, *, conf=0.6) -> tuple:
    """A fully graded 21d-log row. `horizon` None = legacy/NULL-horizon."""
    return (
        f"T{i}", pred_date, "UP", conf, 0.6, 0.1, 0.3, 0.0,
        1.5, 1, 0.012, horizon, 1,
    )


def test_cutover_date_constant_is_canonical():
    assert CANONICAL_CUTOVER_DATE == "2026-05-09"


def test_pre_cutover_rows_graded_at_horizon_report_blackout(tmp_path):
    """The 2026-05-15 calibration_breakdown class: rows ARE horizon=21 and
    graded, but pre-cutover prediction dates → blackout, not a computation."""
    from analysis.production_health import compute_production_health

    db = _make_db(tmp_path, [_row(i, _PRE, 21) for i in range(15)])
    result = compute_production_health(db, bucket="b", run_date="2026-05-15")

    assert result["status"] == "skipped"
    assert result["reason"] == "post_cutover_ic_blackout"
    assert result["n"] == 0  # zero post-cutover current-horizon rows
    assert result["n_any_horizon"] == 15
    assert "degradation_flag" not in result
    assert CANONICAL_CUTOVER_DATE in result["message"]


def test_legacy_null_horizon_rows_also_blackout(tmp_path):
    from analysis.production_health import compute_production_health

    db = _make_db(tmp_path, [_row(i, _PRE, None) for i in range(12)])
    result = compute_production_health(db, bucket="b", run_date="2026-05-15")
    assert result["reason"] == "post_cutover_ic_blackout"


def test_genuinely_empty_window_reports_insufficient_samples(tmp_path):
    from analysis.production_health import compute_production_health

    db = _make_db(tmp_path, [_row(i, _PRE, 21) for i in range(3)])
    result = compute_production_health(db, bucket="b", run_date="2026-05-15")
    assert result["reason"] == "insufficient_samples"
    assert result["n_any_horizon"] == 3


def test_post_cutover_rows_compute_normally(tmp_path):
    """The cutover filter must not over-exclude: post-cutover horizon=21
    rows flow into the real IC/regime path (not skipped)."""
    from analysis.production_health import compute_production_health

    db = _make_db(tmp_path, [_row(i, _POST, 21) for i in range(15)])
    result = compute_production_health(db, bucket="b", run_date="2026-05-15")

    assert result.get("status") != "skipped"
    assert "rolling_30d_ic" in result
    assert result["n_resolved"] == 15


def test_calibration_pre_cutover_rows_report_blackout(tmp_path):
    """compute_calibration_validation must blackout on pre-cutover-model
    rows rather than emit a spurious ECE (the 2026-05-15 MEDIUM alert)."""
    from analysis.production_health import compute_calibration_validation

    # Overconfident pre-cutover model: conf 0.95, all wrong → huge ECE if
    # it ever reached the bin computation.
    rows = [
        (f"T{i}", _PRE, "UP", 0.95, 0.95, 0.0, 0.05, 0.0, -2.0, 0, 0.01, 21, 0)
        for i in range(40)
    ]
    db = _make_db(tmp_path, rows)
    result = compute_calibration_validation(db, bucket="b", run_date="2026-05-15")

    assert result["status"] == "skipped"
    assert result["reason"] == "post_cutover_ic_blackout"
    assert "overall_ece" not in result


def test_blackout_result_is_persisted_to_s3(tmp_path, monkeypatch):
    """Regression for the 2026-05-15 forensic landmine: the skip path
    returned before the S3 write, freezing production_health.json at a
    stale degradation_flag. The blackout result must be persisted."""
    from analysis import production_health as ph

    puts: dict[str, dict] = {}

    class _S3:
        def put_object(self, Bucket, Key, Body, ContentType):
            puts[Key] = json.loads(Body)

    monkeypatch.setattr(ph.boto3, "client", lambda svc, *a, **k: _S3())

    db = _make_db(tmp_path, [_row(i, _PRE, 21) for i in range(15)])
    ph.compute_production_health(db, bucket="b", run_date="2026-05-15")

    key = "predictor/metrics/production_health.json"
    assert key in puts, "blackout skip must still persist production_health.json"
    assert puts[key]["reason"] == "post_cutover_ic_blackout"
    assert puts[key]["date"] == "2026-05-15"
    assert "degradation_flag" not in puts[key]


def test_blackout_skip_does_not_trigger_retrain_alert(tmp_path):
    """End-to-end: blackout IC + blackout calibration → no triggers."""
    from analysis.production_health import (
        compute_calibration_validation,
        compute_production_health,
    )
    from analysis.retrain_alert import evaluate_retrain_triggers

    db = _make_db(tmp_path, [_row(i, _PRE, 21) for i in range(20)])
    ph = compute_production_health(db, bucket="b", run_date="2026-05-15")
    cal = compute_calibration_validation(db, bucket="b", run_date="2026-05-15")

    alert = evaluate_retrain_triggers(ph, feature_drift=None, calibration=cal)
    assert alert["triggered"] is False
    assert alert["n_triggers"] == 0
