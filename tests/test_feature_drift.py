"""Tests for analysis/feature_drift.py — Phase 3 feature drift detection."""

import json
import sqlite3
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from analysis.feature_drift import (
    compute_feature_drift,
    _classify_drift,
    _join_outcomes_with_features,
    _load_features_from_arctic,
)


# ── _classify_drift tests ───────────────────────────────────────────────────

def test_classify_drift_stable():
    assert _classify_drift(0.04, 0.03) == "stable"


def test_classify_drift_sign_flip_positive_to_negative():
    assert _classify_drift(0.04, -0.02) == "sign_flip"


def test_classify_drift_sign_flip_negative_to_positive():
    assert _classify_drift(-0.04, 0.02) == "sign_flip"


def test_classify_drift_decayed():
    assert _classify_drift(0.04, 0.002) == "decayed"


def test_classify_drift_noise_to_noise():
    """Both below noise threshold — stable."""
    assert _classify_drift(0.003, 0.002) == "stable"


# ── _join_outcomes_with_features tests ───────────────────────────────────────

def test_join_outcomes_with_features_exact_date_match():
    dates = pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-03"])
    features = {
        "AAPL": pd.DataFrame(
            {"rsi_14": [45.0, 50.0, 55.0], "momentum_20d": [0.1, 0.2, 0.3]},
            index=dates,
        )
    }
    outcomes = pd.DataFrame({
        "symbol": ["AAPL"],
        "prediction_date": pd.to_datetime(["2026-04-02"]),
        "canonical_actual": [1.5],
        "actual": [1.5],
        "horizon_days": [21],
    })

    joined = _join_outcomes_with_features(outcomes, features)
    assert len(joined) == 1
    assert joined.iloc[0]["rsi_14"] == 50.0
    assert joined.iloc[0]["momentum_20d"] == 0.2


def test_join_outcomes_with_features_nearest_prior_date():
    dates = pd.to_datetime(["2026-04-01", "2026-04-03"])
    features = {
        "AAPL": pd.DataFrame(
            {"rsi_14": [45.0, 55.0]},
            index=dates,
        )
    }
    outcomes = pd.DataFrame({
        "symbol": ["AAPL"],
        "prediction_date": pd.to_datetime(["2026-04-02"]),
        "canonical_actual": [1.5],
        "actual": [1.5],
        "horizon_days": [21],
    })

    joined = _join_outcomes_with_features(outcomes, features)
    assert len(joined) == 1
    assert joined.iloc[0]["rsi_14"] == 45.0  # falls back to Apr 1


def test_join_outcomes_missing_ticker():
    features = {
        "AAPL": pd.DataFrame(
            {"rsi_14": [45.0]},
            index=pd.to_datetime(["2026-04-01"]),
        )
    }
    outcomes = pd.DataFrame({
        "symbol": ["MSFT"],
        "prediction_date": pd.to_datetime(["2026-04-01"]),
        "actual_5d_return": [1.5],
        "actual": [1.5],
    })

    joined = _join_outcomes_with_features(outcomes, features)
    assert joined.empty


# ── _load_features_from_arctic uses the shared universe-lib helper (config#804) ─

def test_load_features_from_arctic_uses_open_universe_lib():
    """The universe library must be opened via the shared
    ``alpha_engine_lib.arcticdb.open_universe_lib`` helper, not a raw
    ``Arctic(...).get_library("universe")`` call (config#804 migration).

    ``open_universe_lib`` is imported inside the function at call time, so
    patching the lib attribute intercepts the migrated site.
    """
    dates = pd.to_datetime(["2026-04-01", "2026-04-02"])
    lib = MagicMock()
    lib.list_symbols.return_value = ["AAPL"]
    read_result = MagicMock()
    read_result.data = pd.DataFrame({"rsi_14": [50.0, 55.0]}, index=dates)
    lib.read.return_value = read_result

    with patch("nousergon_lib.arcticdb.open_universe_lib", return_value=lib) as helper:
        result = _load_features_from_arctic("alpha-engine-research", ["AAPL"])

    helper.assert_called_once_with("alpha-engine-research")
    assert "AAPL" in result


# ── compute_feature_drift integration tests ──────────────────────────────────

@pytest.fixture
def mock_db(tmp_path):
    """Create a temp SQLite DB with predictor_outcomes data (post-2026-05-09 schema)."""
    db_path = str(tmp_path / "research.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE predictor_outcomes (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            predicted_direction TEXT,
            prediction_confidence REAL,
            p_up REAL, p_flat REAL, p_down REAL,
            score_modifier_applied REAL DEFAULT 0.0,
            actual_5d_return REAL,
            correct_5d INTEGER,
            actual_log_alpha REAL,
            horizon_days INTEGER,
            correct INTEGER,
            UNIQUE(symbol, prediction_date)
        )
    """)

    # Insert 30 resolved outcomes for 3 tickers populated under canonical
    # post-cutover schema (actual_log_alpha set, horizon_days=21).
    np.random.seed(42)
    tickers = ["AAPL", "MSFT", "GOOGL"]
    dates = pd.bdate_range("2026-02-15", periods=10)
    for ticker in tickers:
        for d in dates:
            actual = round(np.random.normal(0, 2), 4)
            conn.execute(
                "INSERT INTO predictor_outcomes (symbol, prediction_date, predicted_direction, "
                "prediction_confidence, p_up, p_down, "
                "actual_log_alpha, horizon_days, correct) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ticker, d.strftime("%Y-%m-%d"), "UP", 0.65, 0.65, 0.15,
                 actual, 21, 1 if actual > 0 else 0),
            )
    conn.commit()
    conn.close()
    return db_path


def test_compute_feature_drift_skips_no_db():
    result = compute_feature_drift("/nonexistent/path.db", "bucket")
    assert result["status"] == "skipped"
    assert result["reason"] == "no_db"


def test_compute_feature_drift_skips_insufficient_outcomes(tmp_path):
    db_path = str(tmp_path / "research.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE predictor_outcomes (
            id INTEGER PRIMARY KEY, symbol TEXT, prediction_date TEXT,
            predicted_direction TEXT, prediction_confidence REAL,
            p_up REAL, p_flat REAL, p_down REAL,
            score_modifier_applied REAL, actual_5d_return REAL, correct_5d INTEGER,
            actual_log_alpha REAL, horizon_days INTEGER, correct INTEGER
        )
    """)
    conn.commit()
    conn.close()

    result = compute_feature_drift(db_path, "bucket", lookback_days=60)
    assert result["status"] == "skipped"
    assert result["reason"] == "insufficient_samples"


@patch("analysis.feature_drift._load_training_feature_ics")
@patch("analysis.feature_drift._load_features_from_arctic")
@patch("analysis.feature_drift.boto3")
def test_compute_feature_drift_detects_sign_flip(mock_boto, mock_arctic, mock_training_ics, mock_db):
    """End-to-end: features with flipped ICs are flagged."""
    # Build feature data that anti-correlates with actual returns for rsi_14
    conn = sqlite3.connect(mock_db)
    outcomes = pd.read_sql_query(
        "SELECT symbol, prediction_date, actual_log_alpha FROM predictor_outcomes",
        conn,
    )
    conn.close()

    features = {}
    for ticker in outcomes["symbol"].unique():
        ticker_rows = outcomes[outcomes["symbol"] == ticker].copy()
        dates = pd.to_datetime(ticker_rows["prediction_date"])
        actuals = ticker_rows["actual_log_alpha"].values

        features[ticker] = pd.DataFrame(
            {
                "rsi_14": -actuals + np.random.normal(0, 0.01, len(actuals)),  # anti-correlated
                "momentum_20d": actuals + np.random.normal(0, 0.01, len(actuals)),  # correlated
            },
            index=dates,
        )

    mock_arctic.return_value = features
    mock_training_ics.return_value = {"rsi_14": 0.045, "momentum_20d": 0.038}

    result = compute_feature_drift(mock_db, "alpha-engine-research", run_date="2026-04-07", lookback_days=120)

    assert "drifted_features" in result, f"Expected successful result, got: {result}"
    drifted_names = [f["feature"] for f in result["drifted_features"]]
    assert "rsi_14" in drifted_names  # should be flagged as sign_flip
    assert result["total_features"] == 2
    assert result["recommendation"] in ("retrain_suggested", "monitor")


@patch("analysis.feature_drift._load_training_feature_ics")
@patch("analysis.feature_drift._load_features_from_arctic")
@patch("analysis.feature_drift.boto3")
def test_compute_feature_drift_stable(mock_boto, mock_arctic, mock_training_ics, mock_db):
    """All features correlated with actuals — should be stable."""
    conn = sqlite3.connect(mock_db)
    outcomes = pd.read_sql_query(
        "SELECT symbol, prediction_date, actual_log_alpha FROM predictor_outcomes",
        conn,
    )
    conn.close()

    features = {}
    for ticker in outcomes["symbol"].unique():
        ticker_rows = outcomes[outcomes["symbol"] == ticker].copy()
        dates = pd.to_datetime(ticker_rows["prediction_date"])
        actuals = ticker_rows["actual_log_alpha"].values

        features[ticker] = pd.DataFrame(
            {
                "rsi_14": actuals + np.random.normal(0, 0.01, len(actuals)),
                "momentum_20d": actuals + np.random.normal(0, 0.01, len(actuals)),
            },
            index=dates,
        )

    mock_arctic.return_value = features
    mock_training_ics.return_value = {"rsi_14": 0.045, "momentum_20d": 0.038}

    result = compute_feature_drift(mock_db, "alpha-engine-research", run_date="2026-04-07", lookback_days=120)

    assert "drifted_features" in result, f"Expected successful result, got: {result}"
    assert len(result["drifted_features"]) == 0
    assert result["recommendation"] == "stable"


# ── config#806 silent-fail hardening ─────────────────────────────────────────


def test_load_features_from_arctic_logs_per_ticker_read_failure(caplog):
    """A corrupt / unreadable per-ticker ArcticDB read must NOT abort the whole
    feature load (fail-soft preserved) but MUST be logged loudly — the prior
    code skipped silently (config#806 silent-fail hardening)."""
    import logging

    dates = pd.to_datetime(["2026-04-01", "2026-04-02"])
    lib = MagicMock()
    lib.list_symbols.return_value = ["AAPL", "MSFT"]

    good = MagicMock()
    good.data = pd.DataFrame({"rsi_14": [50.0, 55.0]}, index=dates)

    def _read(symbol):
        if symbol == "MSFT":
            raise RuntimeError("arcticdb: corrupt segment")
        return good

    lib.read.side_effect = _read

    with patch("nousergon_lib.arcticdb.open_universe_lib", return_value=lib), \
         caplog.at_level(logging.WARNING, logger="analysis.feature_drift"):
        result = _load_features_from_arctic("alpha-engine-research", ["AAPL", "MSFT"])

    # Fail-soft: the good ticker still loads, the bad one is skipped.
    assert "AAPL" in result
    assert "MSFT" not in result
    # Loud: the skip is logged with the ticker + the failure type, not swallowed.
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "MSFT" in msgs
    assert "RuntimeError" in msgs
