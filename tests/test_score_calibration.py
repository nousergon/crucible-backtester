"""Tests for compute_score_calibration's robust Spearman calibration fields.

The legacy ``monotonic`` binary (strict non-decreasing avg_alpha across quantile
buckets) flips False on a single noisy bucket and is no longer the graded
metric. These tests pin the row-level Spearman rank correlation that replaced
it as the load-bearing signal. See ROADMAP L4550.
"""

import sqlite3

import pytest

from analysis.alpha_distribution import compute_score_calibration


def _build_db(tmp_path, rows):
    """rows = list of (symbol, score, return_21d, spy_21d_return, score_date)."""
    db = tmp_path / "research.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE score_performance ("
        "symbol TEXT, score REAL, return_5d REAL, spy_5d_return REAL, "
        "return_21d REAL, spy_21d_return REAL, "
        "score_date TEXT)"
    )
    conn.executemany(
        "INSERT INTO score_performance "
        "(symbol, score, return_21d, spy_21d_return, score_date) VALUES (?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return str(db)


def test_positive_calibration(tmp_path):
    # alpha increases monotonically with score → strong positive rank corr.
    rows = [(f"T{i}", float(i), float(i) * 0.1, 0.0, "2026-06-01") for i in range(1, 51)]
    out = compute_score_calibration(_build_db(tmp_path, rows))
    assert out["status"] == "ok"
    assert out["spearman_rho"] == pytest.approx(1.0, abs=1e-6)
    assert out["spearman_p"] < 0.01
    assert out["spearman_n"] == 50
    assert out["calibration_assessment"] == "positive"


def test_negative_calibration(tmp_path):
    # alpha DECREASES with score → significant negative rank corr (worse signal
    # scored higher). This is the genuinely-miscalibrated case.
    rows = [(f"T{i}", float(i), -float(i) * 0.1, 0.0, "2026-06-01") for i in range(1, 51)]
    out = compute_score_calibration(_build_db(tmp_path, rows))
    assert out["spearman_rho"] == pytest.approx(-1.0, abs=1e-6)
    assert out["calibration_assessment"] == "negative"


def test_flat_calibration_reads_as_flat(tmp_path):
    # No relationship between score and alpha → insignificant rho → "flat",
    # which the grader treats as neutral rather than RED.
    alphas = [0.5, -0.5] * 25  # alternating, uncorrelated with the score ladder
    rows = [(f"T{i}", float(i), alphas[i - 1], 0.0, "2026-06-01") for i in range(1, 51)]
    out = compute_score_calibration(_build_db(tmp_path, rows))
    assert out["spearman_p"] >= 0.10
    assert out["calibration_assessment"] == "flat"


def test_legacy_monotonic_field_still_present(tmp_path):
    # Backward-compat: the legacy bucket binary is retained as a diagnostic.
    rows = [(f"T{i}", float(i), float(i) * 0.1, 0.0, "2026-06-01") for i in range(1, 51)]
    out = compute_score_calibration(_build_db(tmp_path, rows))
    assert "monotonic" in out
    assert isinstance(out["monotonic"], bool)
