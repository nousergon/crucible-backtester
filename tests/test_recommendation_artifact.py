"""Unit tests for optimizer.recommendation_artifact — Pydantic-equivalent
dataclass + S3 helpers, all S3 calls mocked.

Foundation for the optimizer-artifact-assembler arc (see
``alpha-engine-docs/private/optimizer-artifact-assembler-260509.md``).
"""
import json
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from optimizer.recommendation_artifact import (
    RecommendationArtifact,
    artifact_s3_key,
    derive_promotion_intent,
    read_all_artifacts_for_date,
    read_artifact,
    write_artifact,
)


class TestRecommendationArtifact:

    def test_required_fields_construct(self):
        a = RecommendationArtifact(
            fit_target="skill_composite",
            optimizer_name="executor_optimizer",
            run_date="2026-05-09",
            recommendation_kind="full_replace",
            recommended_params={"atr_multiplier": 2.0},
            promotion_intent="promote",
        )
        assert a.schema_version == 1
        assert a.run_id  # auto-generated UUID
        assert a.diagnostic == {}
        assert a.overlay_keys is None
        assert a.notes == ""

    def test_to_json_round_trips_via_from_dict(self):
        a = RecommendationArtifact(
            fit_target="skill_composite",
            optimizer_name="executor_optimizer",
            run_date="2026-05-09",
            recommendation_kind="full_replace",
            recommended_params={"atr_multiplier": 3.0, "min_score": 75},
            promotion_intent="promote",
            diagnostic={"best_sortino": 0.95},
            notes="test",
        )
        body = a.to_json()
        parsed = json.loads(body)
        # JSON should be human-readable — sorted keys, indented.
        assert "atr_multiplier" in body
        assert parsed["recommended_params"] == {"atr_multiplier": 3.0, "min_score": 75}
        b = RecommendationArtifact.from_dict(parsed)
        assert b.run_id == a.run_id
        assert b.diagnostic == a.diagnostic
        assert b.notes == "test"

    def test_s3_key_layout(self):
        a = RecommendationArtifact(
            fit_target="x", optimizer_name="executor_optimizer",
            run_date="2026-05-09", recommendation_kind="full_replace",
            recommended_params={}, promotion_intent="skip",
        )
        assert a.s3_key("executor_params") == (
            "config/executor_params/recommendations/2026-05-09/"
            "from_executor_optimizer.json"
        )

    def test_artifact_s3_key_helper_matches(self):
        # The free function produces the same key the dataclass method does.
        assert artifact_s3_key("executor_params", "2026-05-09", "executor_optimizer") == (
            "config/executor_params/recommendations/2026-05-09/"
            "from_executor_optimizer.json"
        )


class TestDerivePromotionIntent:
    """`derive_promotion_intent` translates the recommend()/apply() result
    convention (status + apply_result.applied) into the typed enum."""

    def test_status_ok_apply_applied_true_promotes(self):
        r = {"status": "ok", "apply_result": {"applied": True}}
        assert derive_promotion_intent(r) == "promote"

    def test_status_ok_apply_applied_false_shadows(self):
        r = {"status": "ok", "apply_result": {"applied": False, "reason": "shadow mode"}}
        assert derive_promotion_intent(r) == "shadow"

    def test_status_ok_no_apply_result_yet_promotes(self):
        # Artifact may be produced BEFORE apply() runs; status=ok with no
        # apply_result means the optimizer's own gates passed.
        r = {"status": "ok"}
        assert derive_promotion_intent(r) == "promote"

    def test_status_negative_sortino_skips(self):
        r = {"status": "negative_sortino"}
        assert derive_promotion_intent(r) == "skip"

    def test_status_no_improvement_skips(self):
        r = {"status": "no_improvement"}
        assert derive_promotion_intent(r) == "skip"


class TestWriteArtifact:

    def test_write_calls_put_object_with_canonical_key(self):
        s3 = MagicMock()
        a = RecommendationArtifact(
            fit_target="x", optimizer_name="executor_optimizer",
            run_date="2026-05-09", recommendation_kind="full_replace",
            recommended_params={"k": 1}, promotion_intent="promote",
        )
        key = write_artifact(a, "test-bucket", "executor_params", s3_client=s3)

        assert key == (
            "config/executor_params/recommendations/2026-05-09/"
            "from_executor_optimizer.json"
        )
        s3.put_object.assert_called_once()
        call = s3.put_object.call_args
        assert call.kwargs["Bucket"] == "test-bucket"
        assert call.kwargs["Key"] == key
        assert call.kwargs["ContentType"] == "application/json"
        body = json.loads(call.kwargs["Body"])
        assert body["recommended_params"] == {"k": 1}
        assert body["promotion_intent"] == "promote"


class TestReadArtifact:

    def test_returns_artifact_when_present(self):
        a = RecommendationArtifact(
            fit_target="skill_composite", optimizer_name="executor_optimizer",
            run_date="2026-05-09", recommendation_kind="full_replace",
            recommended_params={"atr_multiplier": 3.0},
            promotion_intent="promote",
            diagnostic={"best_sortino": 0.95},
        )
        s3 = MagicMock()
        s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: a.to_json().encode("utf-8"))
        }
        result = read_artifact(
            "test-bucket", "executor_params", "2026-05-09",
            "executor_optimizer", s3_client=s3,
        )
        assert result is not None
        assert result.recommended_params == {"atr_multiplier": 3.0}
        assert result.diagnostic == {"best_sortino": 0.95}

    def test_returns_none_on_no_such_key(self):
        s3 = MagicMock()
        s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject",
        )
        result = read_artifact(
            "test-bucket", "executor_params", "2026-05-09",
            "executor_optimizer", s3_client=s3,
        )
        assert result is None

    def test_other_client_errors_propagate(self):
        s3 = MagicMock()
        s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied"}}, "GetObject",
        )
        with pytest.raises(ClientError):
            read_artifact(
                "test-bucket", "executor_params", "2026-05-09",
                "executor_optimizer", s3_client=s3,
            )


class TestReadAllArtifactsForDate:

    def test_reads_multiple_optimizer_artifacts(self):
        a1 = RecommendationArtifact(
            fit_target="skill_composite", optimizer_name="executor_optimizer",
            run_date="2026-05-09", recommendation_kind="full_replace",
            recommended_params={"atr_multiplier": 3.0}, promotion_intent="promote",
        )
        a2 = RecommendationArtifact(
            fit_target="sizing_ic", optimizer_name="predictor_sizing_optimizer",
            run_date="2026-05-09", recommendation_kind="field_overlay",
            recommended_params={"use_p_up_sizing": True},
            promotion_intent="promote", overlay_keys=["use_p_up_sizing"],
        )
        s3 = MagicMock()
        s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "config/executor_params/recommendations/2026-05-09/from_executor_optimizer.json"},
                {"Key": "config/executor_params/recommendations/2026-05-09/from_predictor_sizing_optimizer.json"},
                # Garbage file in the same prefix — should be skipped.
                {"Key": "config/executor_params/recommendations/2026-05-09/random.txt"},
            ]
        }
        get_calls = {
            "config/executor_params/recommendations/2026-05-09/from_executor_optimizer.json": a1.to_json(),
            "config/executor_params/recommendations/2026-05-09/from_predictor_sizing_optimizer.json": a2.to_json(),
        }
        s3.get_object.side_effect = lambda Bucket, Key: {
            "Body": MagicMock(read=lambda body=get_calls[Key]: body.encode("utf-8")),
        }
        result = read_all_artifacts_for_date(
            "test-bucket", "executor_params", "2026-05-09", s3_client=s3,
        )
        assert set(result.keys()) == {"executor_optimizer", "predictor_sizing_optimizer"}
        assert result["executor_optimizer"].recommended_params == {"atr_multiplier": 3.0}
        assert result["predictor_sizing_optimizer"].recommendation_kind == "field_overlay"

    def test_malformed_artifact_is_skipped_not_fatal(self):
        s3 = MagicMock()
        s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "config/executor_params/recommendations/2026-05-09/from_executor_optimizer.json"},
                {"Key": "config/executor_params/recommendations/2026-05-09/from_broken.json"},
            ]
        }

        def get_side_effect(Bucket, Key):
            if "executor_optimizer" in Key:
                a = RecommendationArtifact(
                    fit_target="x", optimizer_name="executor_optimizer",
                    run_date="2026-05-09", recommendation_kind="full_replace",
                    recommended_params={}, promotion_intent="promote",
                )
                return {"Body": MagicMock(read=lambda: a.to_json().encode("utf-8"))}
            return {"Body": MagicMock(read=lambda: b"not json {{{")}

        s3.get_object.side_effect = get_side_effect
        result = read_all_artifacts_for_date(
            "test-bucket", "executor_params", "2026-05-09", s3_client=s3,
        )
        # Good artifact loaded; broken one skipped (logged warn).
        assert "executor_optimizer" in result
        assert "broken" not in result

    def test_empty_prefix_returns_empty_dict(self):
        s3 = MagicMock()
        s3.list_objects_v2.return_value = {}  # no Contents
        result = read_all_artifacts_for_date(
            "test-bucket", "executor_params", "2026-05-09", s3_client=s3,
        )
        assert result == {}
