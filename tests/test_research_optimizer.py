"""Unit tests for optimizer.research_optimizer.apply — S3 fail-soft contract.

A failed S3 write in research_optimizer.apply() must NOT propagate and take
down the whole Saturday backtester run. It must mirror the sibling optimizers'
fail-soft contract (see executor_optimizer.apply): log loud and return the
not-applied shape ({"applied": False, "reason": ...}) so evaluate.py records
it as unapplied and the run continues (config#1238 item 2).
"""
from unittest.mock import MagicMock, patch

from optimizer.research_optimizer import S3_PARAMS_KEY, apply


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
