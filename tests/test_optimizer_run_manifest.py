"""Tests for evaluate.py optimizer_run manifest (config#1726)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest


class TestSummarizeOptimizerModule:
    def test_skipped_module(self):
        from evaluate import _summarize_optimizer_module

        row = _summarize_optimizer_module(
            "trigger_optimizer",
            {"status": "skipped", "degradation_reason": "missing trigger_scorecard"},
        )
        assert row == {"ran": False, "applied": False, "reason": "missing trigger_scorecard"}

    def test_applied_module(self):
        from evaluate import _summarize_optimizer_module

        row = _summarize_optimizer_module(
            "weight_optimizer",
            {"status": "ok", "apply_result": {"applied": True, "reason": "promoted"}},
        )
        assert row == {"ran": True, "applied": True, "reason": "promoted"}

    def test_declined_apply(self):
        from evaluate import _summarize_optimizer_module

        row = _summarize_optimizer_module(
            "research_optimizer",
            {"status": "ok", "apply_result": {"applied": False, "reason": "gate_not_met"}},
        )
        assert row == {"ran": True, "applied": False, "reason": "gate_not_met"}


class TestWriteOptimizerRunManifest:
    def test_writes_expected_s3_key(self):
        from evaluate import write_optimizer_run_manifest

        s3 = MagicMock()
        key = write_optimizer_run_manifest(
            "alpha-engine-research",
            "2026-07-05",
            {"weight_result": {"status": "ok", "apply_result": {"applied": False, "reason": "no_change"}}},
            freeze=False,
            s3_client=s3,
        )
        assert key == "optimizer_run/2026-07-05.json"
        call = s3.put_object.call_args.kwargs
        assert call["Bucket"] == "alpha-engine-research"
        assert call["Key"] == key
        body = json.loads(call["Body"].decode())
        assert body["trading_day"] == "2026-07-05"
        assert "weight_result" in body["optimizers"]
        assert body["optimizers"]["weight_result"]["applied"] is False

    def test_freeze_skips_write(self):
        from evaluate import write_optimizer_run_manifest

        s3 = MagicMock()
        key = write_optimizer_run_manifest(
            "alpha-engine-research",
            "2026-07-05",
            {},
            freeze=True,
            s3_client=s3,
        )
        assert key == ""
        s3.put_object.assert_not_called()

    def test_s3_failure_raises(self):
        from evaluate import write_optimizer_run_manifest

        s3 = MagicMock()
        s3.put_object.side_effect = Exception("AccessDenied")
        with pytest.raises(Exception, match="AccessDenied"):
            write_optimizer_run_manifest(
                "alpha-engine-research",
                "2026-07-05",
                {"weight_result": {"status": "ok"}},
                freeze=False,
                s3_client=s3,
            )
