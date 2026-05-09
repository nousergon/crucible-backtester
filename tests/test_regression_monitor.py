"""Unit tests for optimizer.regression_monitor — metrics extraction, regression detection."""
import json
from unittest.mock import patch, MagicMock

import pytest

from optimizer.recommendation_artifact import RecommendationArtifact
from optimizer.regression_monitor import (
    S3_ROLLBACK_AUDIT_PREFIX,
    _capture_rejected_recommendations,
    check_regression,
    extract_metrics,
    write_rollback_audit,
)


# ── extract_metrics ──────────────────────────────────────────────────────────


class TestExtractMetrics:

    def test_extracts_portfolio_fields(self):
        """Should extract sharpe_ratio, total_alpha, max_drawdown, win_rate."""
        stats = {
            "sharpe_ratio": 1.5,
            "total_alpha": 0.08,
            "max_drawdown": -0.12,
            "win_rate": 0.55,
            "irrelevant_key": 42,
        }
        metrics = extract_metrics(stats, None)
        assert metrics["sharpe_ratio"] == 1.5
        assert metrics["total_alpha"] == 0.08
        assert metrics["max_drawdown"] == -0.12
        assert metrics["win_rate"] == 0.55
        assert "irrelevant_key" not in metrics

    def test_extracts_signal_quality_fields(self):
        """Should extract accuracy_10d and accuracy_30d from overall dict."""
        sq = {
            "status": "ok",
            "overall": {
                "accuracy_10d": 0.62,
                "accuracy_30d": 0.58,
            },
        }
        metrics = extract_metrics(None, sq)
        assert metrics["accuracy_10d"] == 0.62
        assert metrics["accuracy_30d"] == 0.58

    def test_combines_both_sources(self):
        """Should merge fields from both portfolio stats and signal quality."""
        stats = {"sharpe_ratio": 1.2, "total_alpha": 0.05}
        sq = {"overall": {"accuracy_10d": 0.60}}
        metrics = extract_metrics(stats, sq)
        assert "sharpe_ratio" in metrics
        assert "accuracy_10d" in metrics

    def test_none_inputs_return_empty(self):
        """Both None inputs should return empty dict."""
        assert extract_metrics(None, None) == {}

    def test_empty_dicts_return_empty(self):
        """Empty dicts should return empty metrics."""
        assert extract_metrics({}, {}) == {}

    def test_missing_overall_key(self):
        """Signal quality without 'overall' key should not crash."""
        metrics = extract_metrics(None, {"status": "ok"})
        assert metrics == {}


# ── check_regression ─────────────────────────────────────────────────────────


class TestCheckRegression:

    @patch("optimizer.regression_monitor._load_baseline")
    def test_no_baseline_skips_check(self, mock_load):
        """No baseline → checked=False, no regression."""
        mock_load.return_value = None
        result = check_regression("test-bucket", {"sharpe_ratio": 1.0})
        assert result["checked"] is False
        assert "no baseline" in result.get("reason", "")

    @patch("optimizer.regression_monitor.write_rollback_audit", return_value="")
    @patch("optimizer.regression_monitor.rollback_all", return_value=[])
    @patch("optimizer.regression_monitor._load_baseline")
    def test_positive_sharpe_detects_large_drop(
        self, mock_load, mock_rollback, mock_audit,
    ):
        """Sharpe dropping >20% from positive baseline should trigger regression."""
        mock_load.return_value = {
            "sharpe_ratio": 2.0,
            "accuracy_10d": 0.60,
        }
        result = check_regression(
            "test-bucket",
            {"sharpe_ratio": 1.0, "accuracy_10d": 0.58},
        )
        assert result["checked"] is True
        assert result["regression_detected"] is True
        assert result["details"]["sharpe_drop_pct"] == pytest.approx(0.5, abs=0.01)

    @patch("optimizer.regression_monitor._load_baseline")
    def test_positive_sharpe_no_regression_when_stable(self, mock_load):
        """Sharpe within 20% of baseline should NOT trigger regression."""
        mock_load.return_value = {"sharpe_ratio": 2.0}
        result = check_regression(
            "test-bucket",
            {"sharpe_ratio": 1.8},
        )
        assert result["checked"] is True
        assert result["regression_detected"] is False

    @patch("optimizer.regression_monitor._load_baseline")
    def test_negative_sharpe_baseline_skips_sharpe_check(self, mock_load):
        """Negative baseline Sharpe should skip the Sharpe regression check."""
        mock_load.return_value = {
            "sharpe_ratio": -0.5,
            "accuracy_10d": 0.60,
        }
        result = check_regression(
            "test-bucket",
            {"sharpe_ratio": -1.0, "accuracy_10d": 0.58},
        )
        assert result["checked"] is True
        # Sharpe check skipped (base_sharpe <= 0), so no sharpe_drop_pct
        assert "sharpe_drop_pct" not in result["details"]
        # Accuracy drop is only 2pp (< 5pp threshold), so no regression
        assert result["regression_detected"] is False

    @patch("optimizer.regression_monitor.write_rollback_audit", return_value="")
    @patch("optimizer.regression_monitor.rollback_all", return_value=[])
    @patch("optimizer.regression_monitor._load_baseline")
    def test_accuracy_drop_triggers_regression(
        self, mock_load, mock_rollback, mock_audit,
    ):
        """Accuracy dropping >5pp should trigger regression."""
        mock_load.return_value = {
            "accuracy_10d": 0.65,
        }
        result = check_regression(
            "test-bucket",
            {"accuracy_10d": 0.55},
        )
        assert result["checked"] is True
        assert result["regression_detected"] is True
        assert result["details"]["accuracy_drop"] == pytest.approx(10.0, abs=0.1)

    @patch("optimizer.regression_monitor._load_baseline")
    def test_same_metrics_no_regression(self, mock_load):
        """Identical metrics should not trigger regression."""
        baseline = {"sharpe_ratio": 1.5, "accuracy_10d": 0.60}
        mock_load.return_value = baseline
        result = check_regression("test-bucket", baseline.copy())
        assert result["checked"] is True
        assert result["regression_detected"] is False

    @patch("optimizer.regression_monitor.write_rollback_audit", return_value="")
    @patch("optimizer.regression_monitor.rollback_all", return_value=[])
    @patch("optimizer.regression_monitor._load_baseline")
    def test_custom_thresholds(self, mock_load, mock_rollback, mock_audit):
        """Custom config thresholds should be respected."""
        mock_load.return_value = {"sharpe_ratio": 2.0}
        # 15% drop with a strict 10% threshold → should trigger
        result = check_regression(
            "test-bucket",
            {"sharpe_ratio": 1.7},
            config={"regression_monitor": {"sharpe_drop_threshold_pct": 0.10}},
        )
        assert result["regression_detected"] is True


# ── Rollback audit (PR 6 of optimizer-artifact-assembler arc) ───────────────


class TestCaptureRejectedRecommendations:
    """The audit's rejected_recommendations section reads per-optimizer
    artifacts + assembled output for each config_type with the artifact
    contract wired (currently only executor_params)."""

    def test_captures_executor_params_artifacts_and_assembled(self):
        s3 = MagicMock()
        executor = RecommendationArtifact(
            fit_target="skill_composite", optimizer_name="executor_optimizer",
            run_date="2026-05-09", recommendation_kind="full_replace",
            recommended_params={"atr_multiplier": 3.0, "min_score": 75},
            promotion_intent="promote",
        )
        sizing = RecommendationArtifact(
            fit_target="sizing_ic", optimizer_name="predictor_sizing_optimizer",
            run_date="2026-05-09", recommendation_kind="field_overlay",
            recommended_params={"use_p_up_sizing": True},
            overlay_keys=["use_p_up_sizing"], promotion_intent="promote",
        )
        assembled_body = {
            "status": "ok",
            "config_type": "executor_params",
            "run_date": "2026-05-09",
            "assembled_params": {"atr_multiplier": 3.0, "use_p_up_sizing": True},
        }
        s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "config/executor_params/recommendations/2026-05-09/from_executor_optimizer.json"},
                {"Key": "config/executor_params/recommendations/2026-05-09/from_predictor_sizing_optimizer.json"},
            ],
        }

        def get_side_effect(Bucket, Key):
            if "from_executor_optimizer" in Key:
                return {"Body": MagicMock(read=lambda: executor.to_json().encode())}
            if "from_predictor_sizing_optimizer" in Key:
                return {"Body": MagicMock(read=lambda: sizing.to_json().encode())}
            if "/assembled/" in Key:
                return {"Body": MagicMock(read=lambda: json.dumps(assembled_body).encode())}
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

        s3.get_object.side_effect = get_side_effect
        result = _capture_rejected_recommendations(
            "test-bucket", "2026-05-09", s3_client=s3,
        )
        assert "executor_params" in result
        assert set(result["executor_params"]["from_optimizers"].keys()) == {
            "executor_optimizer", "predictor_sizing_optimizer",
        }
        # Captured artifact's recommended_params must match what would have landed.
        captured = result["executor_params"]["from_optimizers"]["executor_optimizer"]
        assert captured["recommended_params"]["atr_multiplier"] == 3.0
        assert captured["promotion_intent"] == "promote"
        # Assembled section captures the merge result.
        assert result["executor_params"]["assembled"]["assembled_params"] == {
            "atr_multiplier": 3.0, "use_p_up_sizing": True,
        }

    def test_no_artifacts_returns_empty_dict(self):
        s3 = MagicMock()
        s3.list_objects_v2.return_value = {}
        from botocore.exceptions import ClientError
        s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject",
        )
        result = _capture_rejected_recommendations(
            "test-bucket", "2026-05-09", s3_client=s3,
        )
        assert result == {}

    def test_artifact_read_failure_partial_capture(self):
        # If reading one config_type's artifacts raises, the audit should
        # still capture what it can — partial state beats no audit.
        s3 = MagicMock()
        s3.list_objects_v2.side_effect = Exception("S3 disconnected mid-list")
        result = _capture_rejected_recommendations(
            "test-bucket", "2026-05-09", s3_client=s3,
        )
        # Partial: failed config_type absent but no exception raised.
        assert result == {}


class TestWriteRollbackAudit:

    @patch("optimizer.regression_monitor._capture_rejected_recommendations")
    def test_writes_audit_with_full_shape(self, mock_capture):
        mock_capture.return_value = {
            "executor_params": {
                "from_optimizers": {
                    "executor_optimizer": {
                        "recommended_params": {"atr_multiplier": 3.0},
                        "promotion_intent": "promote",
                    },
                },
                "assembled": {"assembled_params": {"atr_multiplier": 3.0}},
            },
        }
        s3 = MagicMock()
        regression_result = {
            "regression_detected": True,
            "details": {"sharpe_drop_pct": 0.30, "accuracy_drop": 2.5},
            "baseline": {"sharpe_ratio": 0.84, "accuracy_10d": 0.62},
            "current": {"sharpe_ratio": 0.59, "accuracy_10d": 0.60},
        }
        rollback_results = [
            {"rolled_back": True, "config_type": "executor_params",
             "key": "config/executor_params.json"},
        ]
        key = write_rollback_audit(
            "test-bucket", "2026-05-09", regression_result, rollback_results,
            s3_client=s3,
        )
        assert key == f"{S3_ROLLBACK_AUDIT_PREFIX}2026-05-09.json"
        s3.put_object.assert_called_once()
        body = json.loads(s3.put_object.call_args.kwargs["Body"])
        assert body["schema_version"] == 1
        assert body["run_date"] == "2026-05-09"
        assert body["trigger"]["regression_detected"] is True
        assert body["trigger"]["details"]["sharpe_drop_pct"] == 0.30
        assert body["baseline"]["sharpe_ratio"] == 0.84
        assert body["current"]["sharpe_ratio"] == 0.59
        assert body["rollback_results"] == rollback_results
        # Killer feature: rejected recommendations capture the alpha-misaligned
        # promotion that the rollback discarded.
        assert body["rejected_recommendations"]["executor_params"]["from_optimizers"][
            "executor_optimizer"
        ]["recommended_params"]["atr_multiplier"] == 3.0
        # Audit timestamp present + ISO8601-shaped.
        assert body["audit_timestamp"].endswith(("Z", "+00:00"))

    @patch("optimizer.regression_monitor._capture_rejected_recommendations",
           return_value={})
    def test_audit_write_failure_non_fatal(self, mock_capture):
        s3 = MagicMock()
        s3.put_object.side_effect = Exception("S3 disconnected on audit write")
        # Should NOT raise — audit failure must not break the rollback flow.
        key = write_rollback_audit(
            "test-bucket", "2026-05-09",
            {"regression_detected": True, "details": {}},
            [],
            s3_client=s3,
        )
        assert key == ""


class TestCheckRegressionInvokesAudit:
    """When regression fires, check_regression invokes write_rollback_audit
    AFTER rollback_all returns. When no regression, audit is not invoked."""

    @patch("optimizer.regression_monitor.write_rollback_audit",
           return_value="config/rollback_audit/2026-05-09.json")
    @patch("optimizer.regression_monitor.rollback_all",
           return_value=[{"rolled_back": True, "config_type": "executor_params",
                          "key": "config/executor_params.json"}])
    @patch("optimizer.regression_monitor._load_baseline")
    def test_audit_fired_on_regression(self, mock_load, mock_rb, mock_audit):
        mock_load.return_value = {"sharpe_ratio": 2.0}
        result = check_regression(
            "test-bucket", {"sharpe_ratio": 1.0}, run_date="2026-05-09",
        )
        assert result["regression_detected"] is True
        assert result["rollback_triggered"] is True
        assert result["rollback_audit_key"] == "config/rollback_audit/2026-05-09.json"
        # Audit was called once with run_date threaded through.
        mock_audit.assert_called_once()
        kwargs = mock_audit.call_args.kwargs
        assert kwargs["run_date"] == "2026-05-09"
        assert kwargs["regression_check_result"]["regression_detected"] is True
        assert kwargs["rollback_results"][0]["rolled_back"] is True

    @patch("optimizer.regression_monitor.write_rollback_audit")
    @patch("optimizer.regression_monitor._load_baseline")
    def test_audit_not_fired_when_no_regression(self, mock_load, mock_audit):
        mock_load.return_value = {"sharpe_ratio": 2.0}
        result = check_regression(
            "test-bucket", {"sharpe_ratio": 1.9}, run_date="2026-05-09",
        )
        assert result["regression_detected"] is False
        # No regression → no rollback → no audit.
        mock_audit.assert_not_called()
