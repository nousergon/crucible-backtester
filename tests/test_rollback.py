"""Unit tests for optimizer.rollback — config#2448 (config#1958 Trust battery).

rollback.py had zero dedicated test coverage; a mutmut run scoped to it (per
setup.cfg's [mutmut] section, added by PR489) found 97/97 checked mutants
survived, meaning nothing exercised the module's logic at all. This is the
auto-apply rollback path for the backtester-promoted configs
(``scoring_weights``, ``executor_params``, ``predictor_params``,
``portfolio_optimizer``) — the last line of defense if an optimizer's
auto-applied change regresses live trading.

Coverage:
- ``save_previous``: successful snapshot copy; no-existing-config (404/NoSuchKey)
  returns False without raising; unknown config_type warns + returns False;
  non-404 ClientErrors propagate (fail loud, not silently).
- ``rollback``: successful restore from the ``_previous.json`` snapshot;
  missing/never-saved previous config fails loud via a structured
  ``{"rolled_back": False, "reason": ...}`` (never silently no-ops or raises
  past the caller); unknown config_type rejected before any S3 call;
  non-404 ClientErrors are caught and reported in ``reason`` rather than
  propagating (rollback is the last-resort path — regression_monitor.py's
  ``rollback_all`` sweep must not itself crash on one bad config type).
- ``rollback_all``: sweeps every registered CONFIG_KEYS entry and returns one
  result per type, independent of individual failures.
- All S3 access is mocked (no live S3).

Non-inferable note on the "guardrails" question raised in the issue: rollback.py
itself carries NO confidence-floor / max-single-change guardrail logic, and it
should not grow any — it doesn't re-derive new values the way
``weight_optimizer.py::apply_weights`` does, it byte-for-byte restores the prior
S3 object that ``save_previous`` snapshotted before the apply path overwrote it.
The apply-side guardrails already gate whether an apply happens at all; rollback
is the pure undo of that action. The guardrail-mirroring concern is what
``regression_monitor.py`` (the caller that decides *when* to invoke
``rollback_all``) already owns, not this module.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from optimizer.rollback import CONFIG_KEYS, rollback, rollback_all, save_previous


def _not_found_error(op_name="CopyObject"):
    return ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, op_name)


def _no_such_key_error(op_name="CopyObject"):
    return ClientError({"Error": {"Code": "NoSuchKey", "Message": "no key"}}, op_name)


def _access_denied_error(op_name="CopyObject"):
    return ClientError({"Error": {"Code": "AccessDenied", "Message": "nope"}}, op_name)


@pytest.fixture
def fake_s3():
    """A MagicMock standing in for boto3.client('s3'); copy_object is the
    only method rollback.py's save_previous/rollback call."""
    return MagicMock()


@pytest.fixture
def patched_boto(fake_s3):
    with patch("optimizer.rollback.boto3.client", return_value=fake_s3):
        yield fake_s3


class TestConfigKeys:
    def test_known_config_types_registered(self):
        # Pins the current registered set — a change here is a deliberate
        # CONFIG_KEYS edit, not an accidental drop of a config type.
        assert set(CONFIG_KEYS) == {
            "scoring_weights", "executor_params", "predictor_params",
            "portfolio_optimizer",
        }

    def test_active_keys_are_config_json_paths(self):
        for config_type, key in CONFIG_KEYS.items():
            assert key == f"config/{config_type}.json"


class TestSavePrevious:
    def test_successful_snapshot_copies_active_to_previous_key(self, patched_boto):
        result = save_previous("my-bucket", "scoring_weights")
        assert result is True
        patched_boto.copy_object.assert_called_once_with(
            Bucket="my-bucket",
            CopySource={"Bucket": "my-bucket", "Key": "config/scoring_weights.json"},
            Key="config/scoring_weights_previous.json",
        )

    @pytest.mark.parametrize("error", [_not_found_error(), _no_such_key_error()])
    def test_no_existing_active_config_returns_false_not_raise(self, patched_boto, error):
        patched_boto.copy_object.side_effect = error
        result = save_previous("my-bucket", "executor_params")
        assert result is False

    def test_non_404_client_error_propagates_fail_loud(self, patched_boto):
        patched_boto.copy_object.side_effect = _access_denied_error()
        with pytest.raises(ClientError):
            save_previous("my-bucket", "executor_params")

    def test_unknown_config_type_warns_and_returns_false_without_s3_call(self, patched_boto):
        result = save_previous("my-bucket", "not_a_real_config_type")
        assert result is False
        patched_boto.copy_object.assert_not_called()

    def test_research_params_is_not_a_registered_config_type(self, patched_boto):
        """Pins present behavior: optimizer/research_optimizer.py calls
        ``save_previous(bucket, "research_params")`` (research_optimizer.py:438)
        but "research_params" is NOT a key in CONFIG_KEYS, so this call is
        presently always a silent-to-the-caller no-op (returns False, logs a
        warning, does not raise) rather than actually snapshotting research_params
        for a future rollback. Flagged upstream on config#2448 as a possible
        follow-up rather than fixed here (out of this issue's stated scope,
        which is test coverage, not a CONFIG_KEYS behavior change)."""
        result = save_previous("my-bucket", "research_params")
        assert result is False
        patched_boto.copy_object.assert_not_called()


class TestRollback:
    def test_successful_rollback_restores_prior_version(self, patched_boto):
        result = rollback("my-bucket", "predictor_params")
        assert result == {
            "rolled_back": True,
            "config_type": "predictor_params",
            "key": "config/predictor_params.json",
        }
        patched_boto.copy_object.assert_called_once_with(
            Bucket="my-bucket",
            CopySource={"Bucket": "my-bucket", "Key": "config/predictor_params_previous.json"},
            Key="config/predictor_params.json",
        )

    @pytest.mark.parametrize("error", [_not_found_error(), _no_such_key_error()])
    def test_missing_previous_version_fails_loud_structured(self, patched_boto, error):
        """A missing/never-saved _previous.json must not silently succeed or
        silently no-op — it must come back as an explicit rolled_back: False
        with a reason a caller (regression_monitor.py) can log/alert on."""
        patched_boto.copy_object.side_effect = error
        result = rollback("my-bucket", "executor_params")
        assert result["rolled_back"] is False
        assert "predictor" not in result["reason"]  # sanity: right config_type
        assert "executor_params" in result["reason"]

    def test_corrupt_or_other_client_error_reports_reason_without_raising(self, patched_boto):
        """rollback() is the last-resort path invoked from a sweep
        (rollback_all / regression_monitor) — a single bad config type's S3
        error must be captured in the result, not raised past the caller, so
        one failure doesn't abort rolling back the other config types."""
        patched_boto.copy_object.side_effect = _access_denied_error()
        result = rollback("my-bucket", "executor_params")
        assert result["rolled_back"] is False
        assert "AccessDenied" in result["reason"] or "nope" in result["reason"]

    def test_unknown_config_type_rejected_before_any_s3_call(self, patched_boto):
        result = rollback("my-bucket", "not_a_real_config_type")
        assert result == {
            "rolled_back": False,
            "reason": "Unknown config type: not_a_real_config_type",
        }
        patched_boto.copy_object.assert_not_called()


class TestRollbackAll:
    def test_sweeps_every_registered_config_type(self, patched_boto):
        results = rollback_all("my-bucket")
        assert [r["config_type"] for r in results] == list(CONFIG_KEYS)
        assert all(r["rolled_back"] is True for r in results)
        assert patched_boto.copy_object.call_count == len(CONFIG_KEYS)

    def test_one_failure_does_not_abort_the_rest(self, patched_boto):
        # First config type's copy fails (no previous saved); the rest succeed.
        calls = {"n": 0}

        def _side_effect(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _no_such_key_error()
            return {}

        patched_boto.copy_object.side_effect = _side_effect
        results = rollback_all("my-bucket")
        assert len(results) == len(CONFIG_KEYS)
        assert results[0]["rolled_back"] is False
        assert all(r["rolled_back"] is True for r in results[1:])

    def test_returns_one_result_per_config_type_independent_of_order(self, patched_boto):
        results = rollback_all("my-bucket")
        assert len(results) == len(CONFIG_KEYS)
        assert {r["config_type"] for r in results} == set(CONFIG_KEYS)
