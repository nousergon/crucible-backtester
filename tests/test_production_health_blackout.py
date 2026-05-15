"""Pin the post-cutover IC-blackout classification in compute_production_health.

Regression for the 2026-05-15 false-positive retrain alert. After the
2026-05-09 21d canonical-alpha cutover, the strict `horizon_days = 21`
filter correctly excludes pre-cutover NULL-horizon / legacy-5d rows. For
~21 trading days no graded current-horizon outcomes exist. That state
must be reported as an explicit, self-clearing `post_cutover_ic_blackout`
— NOT conflated with a genuine `insufficient_samples` data gap, and
never as a degradation (the skip dict has no `degradation_flag`, so
`evaluate_retrain_triggers` no-ops on it).
"""
from __future__ import annotations

import sqlite3

import pytest

_COLS = (
    "symbol, prediction_date, predicted_direction, prediction_confidence, "
    "p_up, p_flat, p_down, score_modifier_applied, actual_5d_return, "
    "correct_5d, actual_log_alpha, horizon_days, correct"
)


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
        f"INSERT INTO predictor_outcomes ({_COLS}) VALUES "
        f"({','.join('?' * 13)})",
        rows,
    )
    conn.commit()
    conn.close()
    return str(db)


def _pre_cutover_row(i: int) -> tuple:
    # Graded via correct_5d (legacy 5d path), horizon_days NULL, no log alpha.
    # Passes OUTCOMES_GRADED_SQL, excluded by `horizon_days = 21`.
    return (
        f"T{i}", "2026-05-10", "UP", 0.6, 0.6, 0.1, 0.3, 0.0,
        1.5, 1, None, None, None,
    )


def test_pre_cutover_only_window_reports_blackout(tmp_path):
    from analysis.production_health import compute_production_health

    db = _make_db(tmp_path, [_pre_cutover_row(i) for i in range(12)])
    result = compute_production_health(db, bucket="b", run_date="2026-05-15")

    assert result["status"] == "skipped"
    assert result["reason"] == "post_cutover_ic_blackout"
    assert result["n"] == 0  # zero rows at the active 21d horizon
    assert result["n_any_horizon"] == 12
    assert result["active_horizon_days"] == 21
    assert "blackout" in result["message"].lower()
    # Must NOT look like a degradation to the alert evaluator.
    assert "degradation_flag" not in result


def test_genuinely_empty_window_reports_insufficient_samples(tmp_path):
    from analysis.production_health import compute_production_health

    db = _make_db(tmp_path, [_pre_cutover_row(i) for i in range(3)])
    result = compute_production_health(db, bucket="b", run_date="2026-05-15")

    assert result["status"] == "skipped"
    assert result["reason"] == "insufficient_samples"
    assert result["n"] == 0
    assert result["n_any_horizon"] == 3


def test_blackout_skip_does_not_trigger_retrain_alert(tmp_path):
    """End-to-end: a blackout skip dict flows through the alert evaluator
    without firing ic_degradation / regime_negative_ic / mode_collapse."""
    from analysis.production_health import compute_production_health
    from analysis.retrain_alert import evaluate_retrain_triggers

    db = _make_db(tmp_path, [_pre_cutover_row(i) for i in range(12)])
    ph = compute_production_health(db, bucket="b", run_date="2026-05-15")

    alert = evaluate_retrain_triggers(ph, feature_drift=None, calibration=None)
    assert alert["triggered"] is False
    assert alert["n_triggers"] == 0
