"""Unit tests for the shared shell-run dry helper in replay/__init__.py.

The helper is the single canonical (no-copy-paste) implementation used
by BOTH lambda_concordance/handler.py and lambda_counterfactual/
handler.py to short-circuit the Saturday-SF shell-run dry path before
any replay scan / external call / S3 / CloudWatch write.
"""

from __future__ import annotations

from replay import (
    SHELL_RUN_DRY_EVENT_KEY,
    is_shell_run_dry,
    shell_run_dry_response,
)


class TestEventKey:
    def test_canonical_key_is_dry_run_llm(self):
        # Verbatim match with the keystone's Research-Lambda key
        # (`"dry_run_llm.$": "$.research_dry"` in step_function.json).
        assert SHELL_RUN_DRY_EVENT_KEY == "dry_run_llm"


class TestIsShellRunDry:
    def test_true_bool(self):
        assert is_shell_run_dry({"dry_run_llm": True}) is True

    def test_false_bool(self):
        assert is_shell_run_dry({"dry_run_llm": False}) is False

    def test_absent_key(self):
        assert is_shell_run_dry({}) is False

    def test_none_event(self):
        assert is_shell_run_dry(None) is False

    def test_string_true_forms(self):
        for v in ("true", "True", "TRUE", "1", "yes", " true "):
            assert is_shell_run_dry({"dry_run_llm": v}) is True

    def test_string_false_forms(self):
        for v in ("false", "0", "no", ""):
            assert is_shell_run_dry({"dry_run_llm": v}) is False

    def test_legacy_dry_run_key_does_not_trigger(self):
        # The pre-existing `dry_run` (compute-but-don't-emit) key must
        # NOT be interpreted as the shell-run short-circuit signal.
        assert is_shell_run_dry({"dry_run": True}) is False


class TestShellRunDryResponse:
    def test_envelope_shape(self):
        resp = shell_run_dry_response("lambda_concordance", 0.0)
        assert resp["status"] == "DRY_RUN"
        assert resp["dry_run"] is True
        assert resp["handler"] == "lambda_concordance"
        assert "note" in resp
        assert isinstance(resp["duration_seconds"], float)
