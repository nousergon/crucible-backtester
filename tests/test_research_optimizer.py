"""Unit tests for optimizer.research_optimizer.apply — S3 fail-soft contract.

A failed S3 write in research_optimizer.apply() must NOT propagate and take
down the whole Saturday backtester run. It must mirror the sibling optimizers'
fail-soft contract (see executor_optimizer.apply): log loud and return the
not-applied shape ({"applied": False, "reason": ...}) so evaluate.py records
it as unapplied and the run continues (config#1238 item 2).
"""
import json

import pytest
from unittest.mock import MagicMock, patch

from optimizer.assembler import set_cutover_enabled
from optimizer.research_optimizer import (
    RESEARCH_GRAPH_RETIRED_DATE,
    S3_PARAMS_KEY,
    apply,
    compute_boost_correlations,
    produce_artifact,
)


@pytest.fixture(autouse=True)
def _reset_cutover_flag():
    """Always reset the assembler cutover flag around each test so
    set_cutover_enabled(True) in one test never leaks into the next."""
    set_cutover_enabled(False)
    yield
    set_cutover_enabled(False)


@pytest.fixture(autouse=True)
def _mock_recommendation_artifact_s3():
    """apply() now unconditionally calls produce_artifact() (config#2054),
    which writes to S3 via optimizer.recommendation_artifact. Autouse so
    every test is protected from making a real AWS call."""
    with patch("optimizer.recommendation_artifact.boto3") as mock_boto3:
        mock_boto3.client.return_value = MagicMock()
        yield mock_boto3


def _ok_result() -> dict:
    """A valid status=ok recommendation that triggers the live S3 write."""
    return {
        "status": "ok",
        "recommended_params": {
            "short_interest_buy_boost": 2.5,
            "institutional_boost": 3.0,
        },
        "n_samples": 1234,
        "correlations": {"short_interest_buy_boost": 0.12},
    }


@patch("optimizer.research_optimizer.boto3")
def test_apply_returns_failsoft_when_live_write_raises(mock_boto3):
    """S3 put_object raising on the live params key returns the not-applied
    shape instead of propagating and crashing the backtester run."""
    s3 = MagicMock()
    s3.put_object.side_effect = RuntimeError("S3 unavailable")
    mock_boto3.client.return_value = s3

    with patch("optimizer.rollback.save_previous"):
        # Must NOT raise.
        outcome = apply(_ok_result(), bucket="test-bucket")

    assert outcome["applied"] is False
    assert "S3 write failed" in outcome["reason"]
    assert "S3 unavailable" in outcome["reason"]


@patch("optimizer.research_optimizer.boto3")
def test_apply_history_archive_failure_is_non_fatal(mock_boto3):
    """If the live write succeeds but the history-archive write fails, the
    apply still reports applied=True — live params are already durable."""
    s3 = MagicMock()
    calls = {"n": 0}

    def _put(*args, **kwargs):
        calls["n"] += 1
        # First call is the live params write (succeeds); subsequent
        # history-archive writes fail.
        if kwargs.get("Key") != S3_PARAMS_KEY:
            raise RuntimeError("archive bucket unavailable")
        return MagicMock()

    s3.put_object.side_effect = _put
    mock_boto3.client.return_value = s3

    with patch("optimizer.rollback.save_previous"), \
            patch("optimizer.config_archive.record_apply"):
        outcome = apply(_ok_result(), bucket="test-bucket")

    assert outcome["applied"] is True
    assert "short_interest_buy_boost" in outcome["params"]


@patch("optimizer.research_optimizer.boto3")
def test_apply_happy_path_applies(mock_boto3):
    """Sanity: a successful write returns the applied shape and writes the
    live params key plus the history archive."""
    s3 = MagicMock()
    mock_boto3.client.return_value = s3

    with patch("optimizer.rollback.save_previous"), \
            patch("optimizer.config_archive.record_apply"):
        outcome = apply(_ok_result(), bucket="test-bucket")

    assert outcome["applied"] is True
    keys_written = [c.kwargs["Key"] for c in s3.put_object.call_args_list]
    assert S3_PARAMS_KEY in keys_written
    assert any("research_params_history" in k for k in keys_written)


# ═══════════════════════════════════════════════════════════════════════════════
# config#2054: optimizer-artifact-assembler arc extended to research_params
# ═══════════════════════════════════════════════════════════════════════════════


class TestProduceArtifact:

    def test_writes_to_canonical_key(self, _mock_recommendation_artifact_s3):
        outcome = produce_artifact(
            _ok_result(), bucket="test-bucket", promotion_intent="promote",
            recommended_params=_ok_result()["recommended_params"],
        )
        assert outcome["written"] is True
        assert outcome["key"].startswith("config/research_params/recommendations/")
        assert outcome["key"].endswith("/from_research_optimizer.json")
        s3 = _mock_recommendation_artifact_s3.client.return_value
        body = json.loads(s3.put_object.call_args.kwargs["Body"])
        assert body["promotion_intent"] == "promote"
        assert body["recommendation_kind"] == "full_replace"


class TestApplyCutoverGate:
    """When ``optimizer.assembler.is_cutover_enabled()`` returns True, the
    legacy live-key write path is skipped — the assembler is the sole writer
    of ``config/research_params.json``."""

    @patch("optimizer.research_optimizer.boto3")
    def test_cutover_enabled_skips_legacy_live_write(self, mock_boto3, _mock_recommendation_artifact_s3):
        set_cutover_enabled(True)
        legacy_s3 = MagicMock()
        mock_boto3.client.return_value = legacy_s3

        outcome = apply(_ok_result(), bucket="test-bucket")

        assert outcome["applied"] is False
        assert "cutover_mode" in outcome["reason"]
        assert legacy_s3.put_object.call_args_list == []
        artifact_s3 = _mock_recommendation_artifact_s3.client.return_value
        artifact_keys = [c.kwargs["Key"] for c in artifact_s3.put_object.call_args_list]
        assert any(k.endswith("/from_research_optimizer.json") for k in artifact_keys)

    @patch("optimizer.research_optimizer.boto3")
    def test_cutover_disabled_keeps_legacy_write(self, mock_boto3, _mock_recommendation_artifact_s3):
        legacy_s3 = MagicMock()
        mock_boto3.client.return_value = legacy_s3

        with patch("optimizer.rollback.save_previous"), \
                patch("optimizer.config_archive.record_apply"):
            outcome = apply(_ok_result(), bucket="test-bucket")

        assert outcome["applied"] is True
        legacy_keys = [c.kwargs["Key"] for c in legacy_s3.put_object.call_args_list]
        assert S3_PARAMS_KEY in legacy_keys

    def test_no_recommended_params_still_produces_skip_artifact(self, _mock_recommendation_artifact_s3):
        outcome = apply({"status": "ok", "recommended_params": {}}, bucket="test-bucket")
        assert outcome["applied"] is False
        artifact_s3 = _mock_recommendation_artifact_s3.client.return_value
        body = json.loads(artifact_s3.put_object.call_args.kwargs["Body"])
        assert body["promotion_intent"] == "skip"


class TestBoostCorrelationRetired:
    """alpha-engine-config-I3246: the live boost-correlation path is retired —
    it must never touch S3 and must always report status=retired, regardless
    of input, so the loop reads as by-design-off rather than data-starved."""

    @patch("optimizer.research_optimizer.boto3")
    def test_compute_boost_correlations_is_retired_and_never_touches_s3(self, mock_boto3):
        s3 = MagicMock()
        mock_boto3.client.return_value = s3

        import pandas as pd

        df = pd.DataFrame({"score_date": ["2026-07-01"], "symbol": ["AAA"]})
        result = compute_boost_correlations(df, bucket="test-bucket")

        assert result["status"] == "retired"
        assert result["retired_date"] == RESEARCH_GRAPH_RETIRED_DATE
        assert s3.get_object.call_args_list == []

    def test_retired_status_propagates_through_recommend_as_no_op(self):
        from optimizer.research_optimizer import recommend

        result = recommend(compute_boost_correlations(None, bucket="test-bucket"), current_params={})
        assert result["status"] == "retired"
