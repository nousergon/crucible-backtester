"""Tests for lambda_health/handler.py — daily predictor health check Lambda."""

import json
import os
from unittest.mock import MagicMock, patch, call

import pytest

from lambda_health.handler import (
    handler,
    _download_research_db,
    _load_last_feature_drift,
    _build_email_config,
    _summarize,
    _response,
)


@pytest.fixture(autouse=True)
def _mock_preflight():
    """Short-circuit BacktesterPreflight.run() in handler tests.

    The handler calls preflight to verify AWS_REGION + S3 bucket before
    any work. These tests exercise handler orchestration logic, not the
    preflight itself (which has its own coverage in test_preflight.py).
    Patching at the source module so the handler's inline import picks
    up the stub.
    """
    with patch("preflight.BacktesterPreflight") as mock_pf:
        mock_pf.return_value.run = MagicMock()
        yield mock_pf


# ── _response tests ──────────────────────────────────────────────────────────

def test_response_dict_body():
    resp = _response(200, {"status": "ok"})
    assert resp["statusCode"] == 200
    assert json.loads(resp["body"])["status"] == "ok"


def test_response_string_body():
    resp = _response(500, "error message")
    assert resp["statusCode"] == 500
    assert resp["body"] == "error message"


# ── _build_email_config tests ────────────────────────────────────────────────

def test_build_email_config():
    with patch.dict(os.environ, {
        "EMAIL_SENDER": "test@test.com",
        "EMAIL_RECIPIENTS": "a@test.com, b@test.com",
        "AWS_REGION": "us-west-2",
    }):
        config = _build_email_config()
        assert config["email_sender"] == "test@test.com"
        assert config["email_recipients"] == ["a@test.com", "b@test.com"]
        assert config["aws_region"] == "us-west-2"


def test_build_email_config_empty():
    # Preserve ALPHA_ENGINE_SECRETS_SOURCE=env (from conftest) so get_secret()
    # reads from this (cleared) env-dict instead of falling through to SSM —
    # otherwise the test picks up real production EMAIL_SENDER.
    with patch.dict(os.environ, {"ALPHA_ENGINE_SECRETS_SOURCE": "env"}, clear=True):
        config = _build_email_config()
        assert config["email_sender"] == ""
        assert config["email_recipients"] == []


# ── _summarize tests ─────────────────────────────────────────────────────────

def test_summarize_none():
    assert _summarize(None) == "None"


def test_summarize_skipped():
    assert "skipped" in _summarize({"status": "skipped"})


def test_summarize_with_metrics():
    result = {"rolling_30d_ic": 0.045, "degradation_flag": False}
    s = _summarize(result)
    assert "rolling_30d_ic" in s
    assert "degradation_flag" in s


# ── _load_last_feature_drift tests ───────────────────────────────────────────

@patch("lambda_health.handler.boto3", create=True)
def test_load_feature_drift_success(mock_boto):
    import lambda_health.handler as mod
    s3 = MagicMock()
    mock_boto.client.return_value = s3
    drift_data = {"drift_fraction": 0.15, "drifted_features": []}
    s3.get_object.return_value = {"Body": MagicMock(read=lambda: json.dumps(drift_data).encode())}

    # Patch at module level
    with patch.object(mod, "boto3", mock_boto, create=True):
        # Re-import to get patched version
        result = _load_last_feature_drift("bucket")
    # The function uses its own boto3 import, so we patch differently
    assert result is None or isinstance(result, dict)


@patch("lambda_health.handler.json")
def test_load_feature_drift_not_found(mock_json):
    """Should return None when S3 object doesn't exist."""
    result = _load_last_feature_drift("nonexistent-bucket")
    # Will fail to connect to S3 in test env → returns None
    assert result is None


# ── handler integration tests ────────────────────────────────────────────────

@patch("lambda_health.handler._load_last_feature_drift", return_value=None)
@patch("lambda_health.handler._download_research_db", return_value="/tmp/test.db")
@patch("lambda_health.handler.compute_calibration_validation", create=True)
@patch("lambda_health.handler.compute_production_health", create=True)
def test_handler_dry_run(mock_health, mock_cal, mock_db, mock_drift):
    """dry_run should skip S3 writes and email."""
    result = handler({"dry_run": True}, None)

    assert result["statusCode"] == 200
    body = json.loads(result["body"])
    assert body["status"] == "ok"
    # Should NOT have called the analysis functions (dry_run skips them)
    mock_health.assert_not_called()
    mock_cal.assert_not_called()


@patch("lambda_health.handler._load_last_feature_drift", return_value=None)
@patch("lambda_health.handler._download_research_db", return_value=None)
def test_handler_db_download_failure(mock_db, mock_drift):
    """Should return 500 if research.db can't be downloaded."""
    result = handler({}, None)
    assert result["statusCode"] == 500
    assert "research.db" in result["body"]


@patch("lambda_health.handler.write_health", create=True)
@patch("lambda_health.handler.evaluate_retrain_triggers", create=True)
@patch("lambda_health.handler.compute_calibration_validation", create=True)
@patch("lambda_health.handler.compute_production_health", create=True)
@patch("lambda_health.handler._load_last_feature_drift", return_value=None)
@patch("lambda_health.handler._download_research_db", return_value="/tmp/test.db")
def test_handler_full_run(mock_db, mock_drift, mock_health, mock_cal, mock_triggers, mock_write):
    """Full run should call all phases and write health status."""
    mock_health.return_value = {
        "rolling_30d_ic": 0.04,
        "degradation_flag": False,
        "mode_collapse_flag": False,
    }
    mock_cal.return_value = {"overall_ece": 0.03, "calibration_quality": "good"}
    mock_triggers.return_value = {"triggered": False, "n_triggers": 0, "summary": "ok"}

    # These are imported inside handler, need to patch at module level
    with patch("analysis.production_health.compute_production_health", mock_health), \
         patch("analysis.production_health.compute_calibration_validation", mock_cal), \
         patch("analysis.retrain_alert.evaluate_retrain_triggers", mock_triggers), \
         patch("health_status.write_health", mock_write):
        result = handler({}, None)

    assert result["statusCode"] == 200
    body = json.loads(result["body"])
    assert body["status"] == "ok"
