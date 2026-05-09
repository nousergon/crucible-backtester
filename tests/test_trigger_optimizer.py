"""Unit tests for optimizer.trigger_optimizer — apply() + produce_artifact()
+ dual-write guarantee. All S3 calls mocked.

Part of the optimizer-artifact-assembler arc (PR 2).
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from optimizer.trigger_optimizer import (
    S3_PARAMS_KEY,
    _build_overlay_params,
    apply,
    produce_artifact,
)


# ── _build_overlay_params ────────────────────────────────────────────────────


class TestBuildOverlayParams:

    def test_emits_disabled_triggers_field_overlay(self):
        result = {
            "status": "ok",
            "disabled_triggers": ["pullback", "vwap_discount"],
        }
        params, keys = _build_overlay_params(result)
        assert set(keys) == {"disabled_triggers", "disabled_triggers_updated_at"}
        assert params["disabled_triggers"] == ["pullback", "vwap_discount"]

    def test_empty_disabled_list_when_no_triggers_to_disable(self):
        result = {"status": "ok", "disabled_triggers": []}
        params, _ = _build_overlay_params(result)
        assert params["disabled_triggers"] == []


# ── produce_artifact ─────────────────────────────────────────────────────────


class TestProduceArtifact:

    @patch("optimizer.recommendation_artifact.boto3")
    def test_status_ok_promotes(self, mock_boto3):
        s3 = MagicMock()
        mock_boto3.client.return_value = s3
        result = {
            "status": "ok",
            "disabled_triggers": ["pullback"],
            "total_evaluated": 4,
            "min_trades_threshold": 50,
        }
        outcome = produce_artifact(result, bucket="test-bucket")
        assert outcome["written"] is True
        assert outcome["key"].endswith("/from_trigger_optimizer.json")
        body = json.loads(s3.put_object.call_args.kwargs["Body"])
        assert body["promotion_intent"] == "promote"
        assert body["recommendation_kind"] == "field_overlay"
        assert body["fit_target"] == "entry_timing_alpha"
        assert body["recommended_params"]["disabled_triggers"] == ["pullback"]
        assert "disabled_triggers" in body["overlay_keys"]
        assert body["diagnostic"]["total_evaluated"] == 4

    @patch("optimizer.recommendation_artifact.boto3")
    def test_status_insufficient_data_skips(self, mock_boto3):
        s3 = MagicMock()
        mock_boto3.client.return_value = s3
        result = {"status": "insufficient_data"}
        outcome = produce_artifact(result, bucket="test-bucket")
        assert outcome["written"] is True
        body = json.loads(s3.put_object.call_args.kwargs["Body"])
        assert body["promotion_intent"] == "skip"
        assert body["recommended_params"] == {}
        assert body["overlay_keys"] is None

    @patch("optimizer.recommendation_artifact.boto3")
    def test_swallows_s3_errors_non_fatal(self, mock_boto3):
        s3 = MagicMock()
        s3.put_object.side_effect = Exception("S3 disconnected")
        mock_boto3.client.return_value = s3
        outcome = produce_artifact({
            "status": "ok",
            "disabled_triggers": ["pullback"],
        }, bucket="test-bucket")
        assert outcome["written"] is False
        assert "S3 disconnected" in outcome["reason"]


# ── apply() dual-write contract ──────────────────────────────────────────────


class TestApplyDualWrite:

    @patch("optimizer.trigger_optimizer.boto3")
    @patch("optimizer.recommendation_artifact.boto3")
    def test_legacy_apply_path_also_writes_artifact(
        self, mock_artifact_boto3, mock_apply_boto3,
    ):
        legacy_s3 = MagicMock()
        artifact_s3 = MagicMock()
        # Existing config has empty disabled_triggers; new recommendation differs.
        legacy_s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps({
                "atr_multiplier": 2.0,
                "disabled_triggers": [],
            }).encode()),
        }
        mock_apply_boto3.client.return_value = legacy_s3
        mock_artifact_boto3.client.return_value = artifact_s3

        result = {"status": "ok", "disabled_triggers": ["pullback"]}
        outcome = apply(result, bucket="test-bucket")

        # Legacy live write happened.
        assert outcome["applied"] is True
        legacy_keys = [c.kwargs["Key"] for c in legacy_s3.put_object.call_args_list]
        assert S3_PARAMS_KEY in legacy_keys

        # Artifact write happened too.
        artifact_keys = [c.kwargs["Key"] for c in artifact_s3.put_object.call_args_list]
        assert any(
            k.endswith("/from_trigger_optimizer.json") for k in artifact_keys
        )

    @patch("optimizer.trigger_optimizer.boto3")
    @patch("optimizer.recommendation_artifact.boto3")
    def test_no_change_idempotent_skip_still_writes_artifact(
        self, mock_artifact_boto3, mock_apply_boto3,
    ):
        # Recommendation matches current → apply() returns early with
        # "no change" reason. Artifact is still written for audit.
        legacy_s3 = MagicMock()
        artifact_s3 = MagicMock()
        legacy_s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps({
                "disabled_triggers": ["pullback"],
            }).encode()),
        }
        mock_apply_boto3.client.return_value = legacy_s3
        mock_artifact_boto3.client.return_value = artifact_s3

        result = {"status": "ok", "disabled_triggers": ["pullback"]}
        outcome = apply(result, bucket="test-bucket")
        assert outcome["applied"] is False
        assert "no change" in outcome["reason"]

        # Legacy didn't write to live key.
        legacy_writes = [c for c in legacy_s3.put_object.call_args_list]
        assert len(legacy_writes) == 0

        # Artifact still written — the optimizer's gate passed (status=ok),
        # so promotion_intent=promote even though apply was a no-op.
        artifact_writes = [
            c for c in artifact_s3.put_object.call_args_list
            if c.kwargs["Key"].endswith("/from_trigger_optimizer.json")
        ]
        assert len(artifact_writes) == 1
        body = json.loads(artifact_writes[0].kwargs["Body"])
        assert body["promotion_intent"] == "promote"

    @patch("optimizer.trigger_optimizer.boto3")
    @patch("optimizer.recommendation_artifact.boto3")
    def test_status_not_ok_legacy_returns_early_artifact_still_writes(
        self, mock_artifact_boto3, mock_apply_boto3,
    ):
        legacy_s3 = MagicMock()
        artifact_s3 = MagicMock()
        mock_apply_boto3.client.return_value = legacy_s3
        mock_artifact_boto3.client.return_value = artifact_s3

        result = {"status": "insufficient_data"}
        outcome = apply(result, bucket="test-bucket")
        assert outcome["applied"] is False

        legacy_writes = [c for c in legacy_s3.put_object.call_args_list]
        assert len(legacy_writes) == 0

        # Artifact written with skip intent.
        artifact_call = next(
            c for c in artifact_s3.put_object.call_args_list
            if c.kwargs["Key"].endswith("/from_trigger_optimizer.json")
        )
        body = json.loads(artifact_call.kwargs["Body"])
        assert body["promotion_intent"] == "skip"
