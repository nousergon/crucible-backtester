"""Tests for analysis/retrain_alert.py — Phase 5 retrain triggers."""

import json
from unittest.mock import MagicMock, patch

import pytest

from analysis.retrain_alert import (
    evaluate_retrain_triggers,
    send_retrain_alert,
    _should_suppress,
    _build_subject,
    _build_plain_body,
)


# ── evaluate_retrain_triggers tests ──────────────────────────────────────────

def test_no_triggers_all_healthy():
    health = {
        "degradation_flag": False,
        "mode_collapse_flag": False,
        "regime_ic": {"bull": 0.05, "neutral": 0.03},
        "rolling_30d_ic": 0.04,
        "training_ic": 0.08,
        "ic_ratio": 0.50,
        "prediction_distribution": {"UP": 0.35, "FLAT": 0.40, "DOWN": 0.25},
    }
    drift = {"drift_fraction": 0.05, "drifted_features": []}
    calibration = {"overall_ece": 0.03, "calibration_quality": "good"}

    result = evaluate_retrain_triggers(health, drift, calibration)
    assert result["triggered"] is False
    assert result["n_triggers"] == 0


def test_trigger_ic_degradation():
    health = {
        "degradation_flag": True,
        "ic_ratio": 0.30,
        "rolling_30d_ic": 0.025,
        "training_ic": 0.08,
        "mode_collapse_flag": False,
        "regime_ic": {},
        "prediction_distribution": {},
    }

    result = evaluate_retrain_triggers(health, None, None)
    assert result["triggered"] is True
    triggers = [r["trigger"] for r in result["reasons"]]
    assert "ic_degradation" in triggers
    assert result["reasons"][0]["severity"] == "high"


def test_trigger_feature_drift():
    drift = {
        "drift_fraction": 0.25,
        "total_features": 34,
        "drifted_features": [
            {"feature": "rsi_14", "training_ic": 0.04, "production_ic": -0.01, "status": "sign_flip"},
            {"feature": "macd_cross", "training_ic": 0.03, "production_ic": 0.002, "status": "decayed"},
        ],
    }

    result = evaluate_retrain_triggers(None, drift, None)
    assert result["triggered"] is True
    triggers = [r["trigger"] for r in result["reasons"]]
    assert "feature_drift" in triggers


def test_trigger_calibration_breakdown():
    calibration = {"overall_ece": 0.15, "calibration_quality": "poor"}

    result = evaluate_retrain_triggers(None, None, calibration)
    assert result["triggered"] is True
    triggers = [r["trigger"] for r in result["reasons"]]
    assert "calibration_breakdown" in triggers
    assert result["reasons"][0]["severity"] == "medium"


def test_trigger_regime_negative_ic():
    health = {
        "degradation_flag": False,
        "mode_collapse_flag": False,
        "regime_ic": {"bull": 0.05, "bear": -0.03},
        "prediction_distribution": {},
    }

    result = evaluate_retrain_triggers(health, None, None)
    assert result["triggered"] is True
    triggers = [r["trigger"] for r in result["reasons"]]
    assert "regime_negative_ic" in triggers


def test_trigger_mode_collapse():
    health = {
        "degradation_flag": False,
        "mode_collapse_flag": True,
        "regime_ic": {},
        "prediction_distribution": {"UP": 0.05, "FLAT": 0.90, "DOWN": 0.05},
    }

    result = evaluate_retrain_triggers(health, None, None)
    assert result["triggered"] is True
    triggers = [r["trigger"] for r in result["reasons"]]
    assert "mode_collapse" in triggers


def test_multiple_triggers():
    health = {
        "degradation_flag": True,
        "ic_ratio": 0.20,
        "rolling_30d_ic": 0.015,
        "training_ic": 0.08,
        "mode_collapse_flag": True,
        "regime_ic": {"bear": -0.02},
        "prediction_distribution": {"UP": 0.02, "FLAT": 0.95, "DOWN": 0.03},
    }
    drift = {
        "drift_fraction": 0.30,
        "total_features": 34,
        "drifted_features": [{"feature": "x", "training_ic": 0.04, "production_ic": -0.01, "status": "sign_flip"}] * 11,
    }
    calibration = {"overall_ece": 0.15}

    result = evaluate_retrain_triggers(health, drift, calibration)
    assert result["triggered"] is True
    assert result["n_triggers"] >= 4  # ic_degradation, feature_drift, calibration, mode_collapse, regime


def test_none_inputs():
    result = evaluate_retrain_triggers(None, None, None)
    assert result["triggered"] is False
    assert result["n_triggers"] == 0


# ── calibrator grace window ──────────────────────────────────────────────────

def test_calibration_breakdown_suppressed_during_grace():
    """A calibrator deployed within the grace window should suppress the alert."""
    from datetime import datetime, timedelta
    recent = (datetime.utcnow() - timedelta(days=5)).isoformat()
    calibration = {
        "overall_ece": 0.25,  # above threshold
        "calibrator_deployed_at": recent,
    }
    result = evaluate_retrain_triggers(None, None, calibration)
    triggers = [r["trigger"] for r in result["reasons"]]
    assert "calibration_breakdown" not in triggers


def test_calibration_breakdown_fires_after_grace():
    """Calibrator older than the grace window should not suppress the alert."""
    from datetime import datetime, timedelta
    old = (datetime.utcnow() - timedelta(days=45)).isoformat()
    calibration = {
        "overall_ece": 0.25,
        "calibrator_deployed_at": old,
    }
    result = evaluate_retrain_triggers(None, None, calibration)
    triggers = [r["trigger"] for r in result["reasons"]]
    assert "calibration_breakdown" in triggers


def test_calibration_breakdown_fires_without_timestamp():
    """Absent calibrator_deployed_at → no grace, preserve legacy behavior."""
    calibration = {"overall_ece": 0.25}
    result = evaluate_retrain_triggers(None, None, calibration)
    triggers = [r["trigger"] for r in result["reasons"]]
    assert "calibration_breakdown" in triggers


def test_calibration_breakdown_malformed_timestamp_fires():
    """Unparseable timestamp should not silently swallow the alert."""
    calibration = {
        "overall_ece": 0.25,
        "calibrator_deployed_at": "not-a-date",
    }
    result = evaluate_retrain_triggers(None, None, calibration)
    triggers = [r["trigger"] for r in result["reasons"]]
    assert "calibration_breakdown" in triggers


def test_skipped_inputs():
    """Phase 2/3 returned skipped status — no crash."""
    health = {"status": "skipped", "reason": "insufficient_samples"}
    drift = {"status": "skipped", "reason": "no_db"}
    calibration = {"status": "skipped"}

    result = evaluate_retrain_triggers(health, drift, calibration)
    assert result["triggered"] is False


# ── send_retrain_alert tests ─────────────────────────────────────────────────

@patch("analysis.retrain_alert._write_alert_to_s3")
def test_send_alert_no_triggers(mock_write):
    alert = {"triggered": False, "reasons": [], "n_triggers": 0}
    result = send_retrain_alert(alert, {}, "bucket")
    assert result["sent"] is False
    assert result["reason"] == "no_triggers"
    mock_write.assert_not_called()


@patch("analysis.retrain_alert._write_alert_to_s3")
@patch("analysis.retrain_alert._should_suppress", return_value=True)
def test_send_alert_suppressed(mock_suppress, mock_write):
    alert = {"triggered": True, "reasons": [{"trigger": "test"}], "n_triggers": 1}
    result = send_retrain_alert(alert, {}, "bucket")
    assert result["sent"] is False
    assert result["reason"] == "suppressed"


@patch("analysis.retrain_alert.send_email")
@patch("analysis.retrain_alert._write_alert_to_s3")
@patch("analysis.retrain_alert._should_suppress", return_value=False)
def test_send_alert_email(mock_suppress, mock_write, mock_send_email):
    alert = {
        "triggered": True,
        "date": "2026-04-07",
        "n_triggers": 1,
        "summary": "RETRAIN RECOMMENDED",
        "reasons": [{"trigger": "ic_degradation", "detail": "IC dropped", "severity": "high"}],
    }
    config = {"email_sender": "test@test.com", "email_recipients": ["user@test.com"]}

    result = send_retrain_alert(alert, config, "bucket")

    assert result["sent"] is True
    mock_send_email.assert_called_once()
    mock_write.assert_called_once()


@patch("analysis.retrain_alert._write_alert_to_s3")
@patch("analysis.retrain_alert._should_suppress", return_value=False)
def test_send_alert_no_email_config(mock_suppress, mock_write):
    alert = {
        "triggered": True,
        "date": "2026-04-07",
        "n_triggers": 1,
        "summary": "test",
        "reasons": [{"trigger": "test", "detail": "test", "severity": "high"}],
    }
    result = send_retrain_alert(alert, {}, "bucket")
    assert result["sent"] is False
    assert result["s3_written"] is True


# ── Email formatting tests ───────────────────────────────────────────────────

def test_build_subject_high_severity():
    alert = {
        "date": "2026-04-07",
        "n_triggers": 2,
        "reasons": [
            {"trigger": "ic_degradation", "severity": "high"},
            {"trigger": "calibration_breakdown", "severity": "medium"},
        ],
    }
    subject = _build_subject(alert)
    assert "HIGH" in subject
    assert "2 trigger" in subject


def test_build_subject_medium_only():
    alert = {
        "date": "2026-04-07",
        "n_triggers": 1,
        "reasons": [{"trigger": "calibration_breakdown", "severity": "medium"}],
    }
    subject = _build_subject(alert)
    assert "MEDIUM" in subject


def test_build_plain_body():
    alert = {
        "summary": "RETRAIN RECOMMENDED: 1 trigger(s) fired",
        "reasons": [{"trigger": "ic_degradation", "detail": "IC dropped to 30%", "severity": "high"}],
    }
    body = _build_plain_body(alert)
    assert "RETRAIN" in body
    assert "ic_degradation" in body
    assert "IC dropped" in body


# ── _should_suppress tests ───────────────────────────────────────────────────

@patch("analysis.retrain_alert.boto3")
def test_suppress_yesterday_alert(mock_boto):
    """Alert from yesterday should be suppressed (within 2-day window)."""
    from datetime import date as d, timedelta
    s3 = MagicMock()
    mock_boto.client.return_value = s3
    yesterday = (d.today() - timedelta(days=1)).isoformat()
    s3.get_object.return_value = {
        "Body": MagicMock(read=lambda: json.dumps({"date": yesterday}).encode())
    }
    assert _should_suppress("bucket") is True


@patch("analysis.retrain_alert.boto3")
def test_no_suppress_old_alert(mock_boto):
    """Alert from 3+ days ago should NOT be suppressed."""
    s3 = MagicMock()
    mock_boto.client.return_value = s3
    s3.get_object.return_value = {
        "Body": MagicMock(read=lambda: json.dumps({"date": "2026-01-01"}).encode())
    }
    assert _should_suppress("bucket") is False


@patch("analysis.retrain_alert.boto3")
def test_no_suppress_no_previous(mock_boto):
    s3 = MagicMock()
    mock_boto.client.return_value = s3
    s3.get_object.side_effect = Exception("Not found")
    assert _should_suppress("bucket") is False
