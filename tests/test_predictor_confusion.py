"""Tests for analysis/predictor_confusion.py."""

import sqlite3
import tempfile

import pytest

from analysis.predictor_confusion import (
    DIRECTIONS,
    _actual_direction,
    compute_confusion_matrix,
)


class TestActualDirection:
    def test_up(self):
        assert _actual_direction(0.02) == "UP"

    def test_down(self):
        assert _actual_direction(-0.01) == "DOWN"

    def test_flat(self):
        assert _actual_direction(0.001) == "FLAT"

    def test_boundary_up(self):
        assert _actual_direction(0.005) == "FLAT"  # not strictly >

    def test_boundary_down(self):
        assert _actual_direction(-0.005) == "FLAT"  # not strictly <


class TestComputeConfusionMatrix:
    def _create_db(self, rows):
        """Create a temp DB with predictor_outcomes data.

        Rows tuple shape: (symbol, prediction_date, predicted_direction,
        prediction_confidence, actual_alpha_decimal). The 5th element is
        written into `actual_log_alpha` (canonical decimal log-units) so
        downstream COALESCE returns it as the canonical actual return.
        Schema includes both new (actual_log_alpha/horizon_days/correct)
        and legacy (actual_5d_return/correct_5d) columns post predictor
        21d migration (alpha-engine-research v13 + alpha-engine-data PR
        #198).
        """
        f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        conn = sqlite3.connect(f.name)
        conn.execute(
            "CREATE TABLE predictor_outcomes ("
            "id INTEGER PRIMARY KEY, symbol TEXT, prediction_date TEXT, "
            "predicted_direction TEXT, prediction_confidence REAL, "
            "p_up REAL, p_flat REAL, p_down REAL, "
            "actual_5d_return REAL, correct_5d INTEGER, "
            "actual_log_alpha REAL, horizon_days INTEGER, correct INTEGER)"
        )
        for r in rows:
            conn.execute(
                "INSERT INTO predictor_outcomes (symbol, prediction_date, predicted_direction, "
                "prediction_confidence, actual_log_alpha, horizon_days) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (*r, 21),
            )
        conn.commit()
        conn.close()
        return f.name

    def test_perfect_predictions(self):
        rows = [
            ("AAPL", "2026-04-01", "UP", 0.8, 0.03),
            ("MSFT", "2026-04-01", "UP", 0.7, 0.02),
            ("GOOG", "2026-04-01", "DOWN", 0.8, -0.04),
            ("AMZN", "2026-04-01", "DOWN", 0.7, -0.02),
        ] * 10  # 40 rows
        db = self._create_db(rows)
        result = compute_confusion_matrix(db, min_samples=5)
        assert result["status"] == "ok"
        assert result["accuracy"] == 1.0
        assert result["matrix"]["UP"]["UP"] == 20
        assert result["matrix"]["DOWN"]["DOWN"] == 20
        assert result["per_class"]["UP"]["precision"] == 1.0
        assert result["per_class"]["UP"]["recall"] == 1.0

    def test_insufficient_data(self):
        rows = [("AAPL", "2026-04-01", "UP", 0.8, 0.03)]
        db = self._create_db(rows)
        result = compute_confusion_matrix(db, min_samples=30)
        assert result["status"] == "insufficient_data"

    def test_mixed_predictions(self):
        rows = [
            ("A", "2026-04-01", "UP", 0.8, 0.03),    # correct UP
            ("B", "2026-04-01", "UP", 0.7, -0.02),   # wrong: predicted UP, actual DOWN
            ("C", "2026-04-01", "DOWN", 0.8, -0.04),  # correct DOWN
            ("D", "2026-04-01", "DOWN", 0.7, 0.03),   # wrong: predicted DOWN, actual UP
            ("E", "2026-04-01", "FLAT", 0.6, 0.001),  # correct FLAT
        ] * 10
        db = self._create_db(rows)
        result = compute_confusion_matrix(db, min_samples=5)
        assert result["status"] == "ok"
        assert result["n"] == 50
        assert result["accuracy"] == pytest.approx(0.6)  # 30/50
        assert result["matrix"]["UP"]["UP"] == 10
        assert result["matrix"]["UP"]["DOWN"] == 10  # UP predicted, actually DOWN
        assert result["matrix"]["DOWN"]["DOWN"] == 10
        assert result["matrix"]["DOWN"]["UP"] == 10  # DOWN predicted, actually UP

    def test_structure(self):
        rows = [("A", "2026-04-01", d, 0.7, r) for d, r in [("UP", 0.03), ("DOWN", -0.03), ("FLAT", 0.001)]] * 15
        db = self._create_db(rows)
        result = compute_confusion_matrix(db, min_samples=5)
        assert result["status"] == "ok"
        assert "matrix" in result
        assert "per_class" in result
        for d in DIRECTIONS:
            assert d in result["matrix"]
            assert d in result["per_class"]
            pc = result["per_class"][d]
            assert "precision" in pc
            assert "recall" in pc
            assert "f1" in pc

    def test_missing_db(self):
        result = compute_confusion_matrix("/nonexistent/path.db")
        assert result["status"] == "error"
