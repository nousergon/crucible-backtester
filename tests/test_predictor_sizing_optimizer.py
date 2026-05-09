"""Unit tests for optimizer.predictor_sizing_optimizer — apply() + produce_artifact()
+ dual-write guarantee. All S3 calls mocked.

Part of the optimizer-artifact-assembler arc (PR 2).
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from optimizer.predictor_sizing_optimizer import (
    S3_PARAMS_KEY,
    _build_overlay_params,
    apply,
    produce_artifact,
)


def _set_module_cfg(extra: dict | None = None):
    """Set the module-level _cfg used by _build_overlay_params for blend_factor."""
    from optimizer import predictor_sizing_optimizer as mod
    mod._cfg = {"blend_factor": 0.3}
    if extra:
        mod._cfg.update(extra)


# ── _build_overlay_params ────────────────────────────────────────────────────


class TestBuildOverlayParams:

    def test_emits_4_overlay_fields(self):
        _set_module_cfg()
        result = {
            "status": "ok",
            "recommendation": "enable",
            "overall_rank_ic": 0.10,
        }
        params, keys = _build_overlay_params(result)
        assert set(keys) == {
            "use_p_up_sizing",
            "p_up_sizing_blend",
            "p_up_sizing_updated_at",
            "p_up_sizing_ic",
        }
        assert params["use_p_up_sizing"] is True
        assert params["p_up_sizing_blend"] == 0.3
        assert params["p_up_sizing_ic"] == 0.10

    def test_blend_factor_from_config(self):
        _set_module_cfg({"blend_factor": 0.5})
        result = {"status": "ok", "recommendation": "enable", "overall_rank_ic": 0.08}
        params, _ = _build_overlay_params(result)
        assert params["p_up_sizing_blend"] == 0.5


# ── produce_artifact ─────────────────────────────────────────────────────────


class TestProduceArtifact:

    @patch("optimizer.recommendation_artifact.boto3")
    def test_status_ok_recommendation_enable_promotes(self, mock_boto3):
        _set_module_cfg()
        s3 = MagicMock()
        mock_boto3.client.return_value = s3
        result = {
            "status": "ok",
            "recommendation": "enable",
            "overall_rank_ic": 0.10,
            "recent_mean_ic": 0.08,
            "n_samples": 60,
        }
        outcome = produce_artifact(result, bucket="test-bucket")
        assert outcome["written"] is True
        assert outcome["key"].endswith("/from_predictor_sizing_optimizer.json")
        body = json.loads(s3.put_object.call_args.kwargs["Body"])
        assert body["promotion_intent"] == "promote"
        assert body["recommendation_kind"] == "field_overlay"
        assert body["fit_target"] == "sizing_ic"
        assert body["recommended_params"]["use_p_up_sizing"] is True
        assert "use_p_up_sizing" in body["overlay_keys"]
        # diagnostic fields persisted
        assert body["diagnostic"]["overall_rank_ic"] == 0.10
        assert body["diagnostic"]["n_samples"] == 60

    @patch("optimizer.recommendation_artifact.boto3")
    def test_recommendation_keep_disabled_skips(self, mock_boto3):
        _set_module_cfg()
        s3 = MagicMock()
        mock_boto3.client.return_value = s3
        result = {
            "status": "ok",
            "recommendation": "keep_disabled",
            "overall_rank_ic": 0.02,  # below threshold
        }
        outcome = produce_artifact(result, bucket="test-bucket")
        assert outcome["written"] is True
        body = json.loads(s3.put_object.call_args.kwargs["Body"])
        # Intent is skip — gate didn't pass — but artifact still written for audit.
        assert body["promotion_intent"] == "skip"
        assert body["recommended_params"] == {}
        assert body["overlay_keys"] is None
        # Diagnostic still records what the optimizer found.
        assert body["diagnostic"]["overall_rank_ic"] == 0.02
        assert body["diagnostic"]["recommendation"] == "keep_disabled"

    @patch("optimizer.recommendation_artifact.boto3")
    def test_status_insufficient_data_skips(self, mock_boto3):
        _set_module_cfg()
        s3 = MagicMock()
        mock_boto3.client.return_value = s3
        result = {"status": "insufficient_data", "n_samples": 12}
        outcome = produce_artifact(result, bucket="test-bucket")
        assert outcome["written"] is True
        body = json.loads(s3.put_object.call_args.kwargs["Body"])
        assert body["promotion_intent"] == "skip"

    @patch("optimizer.recommendation_artifact.boto3")
    def test_swallows_s3_errors_non_fatal(self, mock_boto3):
        _set_module_cfg()
        s3 = MagicMock()
        s3.put_object.side_effect = Exception("S3 disconnected")
        mock_boto3.client.return_value = s3
        outcome = produce_artifact({
            "status": "ok",
            "recommendation": "enable",
            "overall_rank_ic": 0.10,
        }, bucket="test-bucket")
        assert outcome["written"] is False
        assert "S3 disconnected" in outcome["reason"]


# ── apply() dual-write contract ──────────────────────────────────────────────


class TestApplyDualWrite:

    @patch("optimizer.predictor_sizing_optimizer.boto3")
    @patch("optimizer.recommendation_artifact.boto3")
    def test_legacy_apply_path_also_writes_artifact(
        self, mock_artifact_boto3, mock_apply_boto3,
    ):
        _set_module_cfg()
        legacy_s3 = MagicMock()
        artifact_s3 = MagicMock()
        # Simulate empty current executor_params (NoSuchKey path).
        legacy_s3.get_object.side_effect = Exception("NoSuchKey")
        mock_apply_boto3.client.return_value = legacy_s3
        mock_artifact_boto3.client.return_value = artifact_s3

        result = {
            "status": "ok",
            "recommendation": "enable",
            "overall_rank_ic": 0.10,
        }
        outcome = apply(result, bucket="test-bucket")

        # Legacy live write happened.
        assert outcome["applied"] is True
        legacy_keys = [c.kwargs["Key"] for c in legacy_s3.put_object.call_args_list]
        assert S3_PARAMS_KEY in legacy_keys

        # Artifact write happened too.
        artifact_keys = [c.kwargs["Key"] for c in artifact_s3.put_object.call_args_list]
        assert any(
            k.endswith("/from_predictor_sizing_optimizer.json")
            for k in artifact_keys
        )

    @patch("optimizer.predictor_sizing_optimizer.boto3")
    @patch("optimizer.recommendation_artifact.boto3")
    def test_keep_disabled_still_writes_artifact(
        self, mock_artifact_boto3, mock_apply_boto3,
    ):
        # Even though apply() refuses to promote (recommendation=keep_disabled),
        # produce_artifact() still fires for audit. promotion_intent=skip.
        _set_module_cfg()
        legacy_s3 = MagicMock()
        artifact_s3 = MagicMock()
        mock_apply_boto3.client.return_value = legacy_s3
        mock_artifact_boto3.client.return_value = artifact_s3

        result = {
            "status": "ok",
            "recommendation": "keep_disabled",
            "overall_rank_ic": 0.02,
            "recent_mean_ic": 0.01,
        }
        outcome = apply(result, bucket="test-bucket")
        assert outcome["applied"] is False

        # Legacy NEVER wrote (apply returned early).
        legacy_writes = [c for c in legacy_s3.put_object.call_args_list]
        assert len(legacy_writes) == 0

        # Artifact STILL wrote.
        artifact_call = next(
            c for c in artifact_s3.put_object.call_args_list
            if c.kwargs["Key"].endswith("/from_predictor_sizing_optimizer.json")
        )
        body = json.loads(artifact_call.kwargs["Body"])
        assert body["promotion_intent"] == "skip"

    @patch("optimizer.predictor_sizing_optimizer.boto3")
    @patch("optimizer.recommendation_artifact.boto3")
    def test_already_enabled_idempotent_skip_still_writes_artifact(
        self, mock_artifact_boto3, mock_apply_boto3,
    ):
        _set_module_cfg()
        legacy_s3 = MagicMock()
        artifact_s3 = MagicMock()
        # Existing config already has use_p_up_sizing=True.
        legacy_s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps({
                "use_p_up_sizing": True,
                "atr_multiplier": 2.0,
            }).encode()),
        }
        mock_apply_boto3.client.return_value = legacy_s3
        mock_artifact_boto3.client.return_value = artifact_s3

        result = {
            "status": "ok",
            "recommendation": "enable",
            "overall_rank_ic": 0.10,
        }
        outcome = apply(result, bucket="test-bucket")
        assert outcome["applied"] is False
        assert "already enabled" in outcome["reason"]

        # Even on the idempotent-skip branch, artifact was still written
        # for this run (with promote intent — the optimizer's gate passed
        # even though apply ended up being a no-op).
        artifact_writes = [
            c for c in artifact_s3.put_object.call_args_list
            if c.kwargs["Key"].endswith("/from_predictor_sizing_optimizer.json")
        ]
        assert len(artifact_writes) == 1
