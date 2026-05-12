"""Tests for analysis.veto_value — net dollar value of predictor DOWN vetoes."""

import sqlite3

import pytest

from analysis.veto_value import _DEFAULT_POSITION_SIZE, compute_veto_value
from pipeline_common import ACTIVE_HORIZON_DAYS


def _build_research_db(path, rows):
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE predictor_outcomes (
            symbol TEXT,
            prediction_date TEXT,
            predicted_direction TEXT,
            prediction_confidence REAL,
            actual_log_alpha REAL,
            actual_5d_return REAL,
            horizon_days INTEGER,
            p_down REAL,
            correct REAL,
            correct_5d REAL
        )
    """)
    conn.executemany(
        "INSERT INTO predictor_outcomes VALUES (:symbol,:prediction_date,:predicted_direction,"
        ":prediction_confidence,:actual_log_alpha,:actual_5d_return,:horizon_days,:p_down,"
        ":correct,:correct_5d)",
        rows,
    )
    conn.commit()
    conn.close()


def _build_trades_db(path, shadow_rows):
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE executor_shadow_book (
            ticker TEXT,
            date TEXT,
            intended_dollars REAL,
            block_reason TEXT,
            predicted_direction TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO executor_shadow_book VALUES (:ticker,:date,:intended_dollars,"
        ":block_reason,:predicted_direction)",
        shadow_rows,
    )
    conn.commit()
    conn.close()


def _down_row(symbol, date, alpha, confidence=0.70, horizon=None):
    return {
        "symbol": symbol,
        "prediction_date": date,
        "predicted_direction": "DOWN",
        "prediction_confidence": confidence,
        "actual_log_alpha": alpha,
        "actual_5d_return": None,
        "horizon_days": horizon if horizon is not None else ACTIVE_HORIZON_DAYS,
        "p_down": 1.0 - confidence + 0.2,
        "correct": (1.0 if alpha < 0 else 0.0) if alpha is not None else None,
        "correct_5d": None,
    }


def test_compute_veto_value_missing_db_returns_error(tmp_path):
    result = compute_veto_value(str(tmp_path / "no_such.db"))
    assert result["status"] == "error"
    assert "research.db not found" in result["error"]


def test_compute_veto_value_insufficient_rows(tmp_path):
    db = tmp_path / "research.db"
    _build_research_db(db, [_down_row("AAPL", "2026-04-01", -0.02)])
    result = compute_veto_value(str(db))
    assert result["status"] == "insufficient_data"
    assert "need >= 3" in result["error"]


def test_compute_veto_value_happy_path_default_position(tmp_path):
    db = tmp_path / "research.db"
    rows = [
        # 3 correct vetoes (negative alpha → losses avoided)
        _down_row("A", "2026-04-01", -0.05, confidence=0.80),
        _down_row("B", "2026-04-02", -0.03, confidence=0.70),
        _down_row("C", "2026-04-03", -0.02, confidence=0.60),
        # 2 incorrect vetoes (positive alpha → alpha foregone)
        _down_row("D", "2026-04-04", 0.04, confidence=0.65),
        _down_row("E", "2026-04-05", 0.06, confidence=0.55),
    ]
    _build_research_db(db, rows)

    result = compute_veto_value(str(db))

    assert result["status"] == "ok"
    assert result["n_vetoes"] == 5
    assert result["n_correct"] == 3
    assert result["n_incorrect"] == 2
    assert result["precision"] == pytest.approx(0.60)

    expected_losses_avoided = _DEFAULT_POSITION_SIZE * (0.05 + 0.03 + 0.02)
    expected_alpha_foregone = _DEFAULT_POSITION_SIZE * (0.04 + 0.06)
    assert result["total_losses_avoided"] == pytest.approx(expected_losses_avoided, abs=1.0)
    assert result["total_alpha_foregone"] == pytest.approx(expected_alpha_foregone, abs=1.0)
    assert result["net_veto_value"] == pytest.approx(expected_losses_avoided - expected_alpha_foregone, abs=1.0)
    assert result["horizons_days"] == [ACTIVE_HORIZON_DAYS]


def test_compute_veto_value_uses_shadow_book_sizing(tmp_path):
    research_db = tmp_path / "research.db"
    trades_db = tmp_path / "trades.db"

    rows = [
        _down_row("A", "2026-04-01", -0.05),
        _down_row("B", "2026-04-02", -0.03),
        _down_row("C", "2026-04-03", 0.04),
    ]
    _build_research_db(research_db, rows)

    # Shadow book sizes A and B; C falls back to default.
    shadow = [
        {"ticker": "A", "date": "2026-04-01", "intended_dollars": 100_000.0,
         "block_reason": "predictor_veto", "predicted_direction": "DOWN"},
        {"ticker": "B", "date": "2026-04-02", "intended_dollars": 25_000.0,
         "block_reason": "predictor_veto", "predicted_direction": "DOWN"},
    ]
    _build_trades_db(trades_db, shadow)

    result = compute_veto_value(str(research_db), str(trades_db))

    assert result["status"] == "ok"
    expected_losses = 100_000.0 * 0.05 + 25_000.0 * 0.03
    expected_foregone = _DEFAULT_POSITION_SIZE * 0.04
    assert result["total_losses_avoided"] == pytest.approx(expected_losses, abs=1.0)
    assert result["total_alpha_foregone"] == pytest.approx(expected_foregone, abs=1.0)


def test_compute_veto_value_confidence_buckets_partitioned(tmp_path):
    db = tmp_path / "research.db"
    rows = [
        _down_row("A", "2026-04-01", -0.02, confidence=0.51),  # 50-55%
        _down_row("B", "2026-04-02", -0.02, confidence=0.52),  # 50-55%
        _down_row("C", "2026-04-03", -0.02, confidence=0.60),  # 55-65%
        _down_row("D", "2026-04-04", -0.02, confidence=0.70),  # 65-75%
        _down_row("E", "2026-04-05", -0.02, confidence=0.80),  # 75%+
    ]
    _build_research_db(db, rows)

    result = compute_veto_value(str(db))

    assert result["status"] == "ok"
    by_confidence = {b["confidence_range"]: b for b in result["by_confidence"]}
    assert by_confidence["50-55%"]["n_vetoes"] == 2
    assert by_confidence["55-65%"]["n_vetoes"] == 1
    assert by_confidence["65-75%"]["n_vetoes"] == 1
    assert by_confidence["75%+"]["n_vetoes"] == 1
    # All five vetoes were correct → precision 1.0 per bucket
    for bucket in by_confidence.values():
        assert bucket["precision"] == pytest.approx(1.0)


def test_compute_veto_value_filters_out_other_directions(tmp_path):
    """Only predicted_direction='DOWN' rows count toward the analysis."""
    db = tmp_path / "research.db"
    down_rows = [_down_row(f"D{i}", f"2026-04-0{i+1}", -0.01) for i in range(3)]
    up_rows = []
    for i in range(5):
        r = _down_row(f"U{i}", f"2026-04-1{i+1}", -0.10)
        r["predicted_direction"] = "UP"
        up_rows.append(r)
    _build_research_db(db, down_rows + up_rows)

    result = compute_veto_value(str(db))

    assert result["status"] == "ok"
    assert result["n_vetoes"] == 3  # UP rows excluded


def test_compute_veto_value_excludes_wrong_horizon(tmp_path):
    """ACTIVE_HORIZON_DAYS filter must drop legacy horizon rows."""
    db = tmp_path / "research.db"
    wrong_horizon = ACTIVE_HORIZON_DAYS + 1 if ACTIVE_HORIZON_DAYS != 5 else 7
    rows = [
        _down_row("A", "2026-04-01", -0.02),
        _down_row("B", "2026-04-02", -0.02),
        _down_row("C", "2026-04-03", -0.02),
        _down_row("X", "2026-04-04", -0.02, horizon=wrong_horizon),
    ]
    _build_research_db(db, rows)

    result = compute_veto_value(str(db))

    assert result["status"] == "ok"
    assert result["n_vetoes"] == 3
    assert result["horizons_days"] == [ACTIVE_HORIZON_DAYS]


def test_compute_veto_value_unresolved_rows_excluded(tmp_path):
    """Rows where both actual_log_alpha and actual_5d_return are NULL are excluded
    by OUTCOMES_RESOLVED_SQL."""
    db = tmp_path / "research.db"
    rows = [
        _down_row("A", "2026-04-01", -0.02),
        _down_row("B", "2026-04-02", -0.02),
        _down_row("C", "2026-04-03", -0.02),
    ]
    unresolved = _down_row("X", "2026-04-04", None)
    unresolved["actual_5d_return"] = None
    rows.append(unresolved)
    _build_research_db(db, rows)

    result = compute_veto_value(str(db))

    assert result["status"] == "ok"
    assert result["n_vetoes"] == 3


def test_compute_veto_value_db_query_error_caught(tmp_path, monkeypatch):
    db = tmp_path / "research.db"
    db.touch()  # file exists so the Path.exists() guard passes

    def broken_connect(_path):
        raise sqlite3.OperationalError("simulated query failure")

    monkeypatch.setattr("analysis.veto_value.sqlite3.connect", broken_connect)
    result = compute_veto_value(str(db))
    assert result["status"] == "error"
    assert "simulated query failure" in result["error"]


def test_compute_veto_value_shadow_book_open_error_falls_through(tmp_path):
    """Shadow-book read errors should NOT abort — falls back to default position size."""
    research_db = tmp_path / "research.db"
    trades_db = tmp_path / "broken.db"  # invalid path → silently skipped

    rows = [
        _down_row("A", "2026-04-01", -0.05),
        _down_row("B", "2026-04-02", -0.03),
        _down_row("C", "2026-04-03", -0.02),
    ]
    _build_research_db(research_db, rows)
    # broken.db doesn't exist → branch falls through to default sizing
    result = compute_veto_value(str(research_db), str(trades_db))

    assert result["status"] == "ok"
    expected = _DEFAULT_POSITION_SIZE * (0.05 + 0.03 + 0.02)
    assert result["total_losses_avoided"] == pytest.approx(expected, abs=1.0)


def test_compute_veto_value_all_incorrect_zero_correct_branch(tmp_path):
    """When n_correct=0, avg_loss_avoided should be 0 (denominator guard)."""
    db = tmp_path / "research.db"
    rows = [_down_row(f"T{i}", f"2026-04-0{i+1}", 0.03) for i in range(3)]
    _build_research_db(db, rows)

    result = compute_veto_value(str(db))

    assert result["status"] == "ok"
    assert result["n_correct"] == 0
    assert result["n_incorrect"] == 3
    assert result["avg_loss_avoided"] == 0
    assert result["total_losses_avoided"] == 0
