"""Unit tests for lambda_concordance/handler.py.

Mocks out compute_and_emit_concordance + ssm_secrets so the handler is
exercised in isolation. Mirrors test_eval_rolling_mean_handler.py /
test_rationale_clustering_handler.py shape.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_HANDLER_PATH = _REPO_ROOT / "lambda_concordance" / "handler.py"


def _load_handler_module():
    """Import lambda_concordance/handler.py without using the package
    name `lambda_concordance` (avoids confusion with the Python keyword
    `lambda` even though it's not actually a name collision)."""
    module_name = "lambda_concordance_handler_under_test"
    spec = importlib.util.spec_from_file_location(module_name, _HANDLER_PATH)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture
def handler_mod():
    mod = _load_handler_module()
    mod._init_done = False
    yield mod


def _ok_summary() -> dict:
    return {
        "window_start": "2026-03-14T00:00:00+00:00",
        "window_end": "2026-05-09T00:00:00+00:00",
        "agent_filter": ["sector_quant", "ic_cio"],
        "artifacts_discovered": 24,
        "per_target_model": [
            {
                "target_model": "claude-haiku-4-5",
                "n_artifacts_replayed": 24,
                "agents_analyzed": 2,
                "per_agent": [
                    {"agent_id_base": "sector_quant", "n": 16, "mean": 0.92},
                    {"agent_id_base": "ic_cio", "n": 8, "mean": 0.65},
                ],
                "agents_skipped_thin_sample": [],
                "replay_failures": [],
                "cost": {"input_tokens": 2000, "output_tokens": 1000},
                "summary_key": "decision_artifacts/_replay_summary/2026-05-09/claude-haiku-4-5.json",
            },
        ],
    }


def _partial_summary() -> dict:
    s = _ok_summary()
    s["per_target_model"][0]["replay_failures"] = [
        {"key": "x.json", "stage": "replay_artifact_call", "error": "Anthropic 500"},
    ]
    return s


def _dry_run_summary() -> dict:
    return {
        "dry_run": True,
        "window_start": "2026-03-14T00:00:00+00:00",
        "window_end": "2026-05-09T00:00:00+00:00",
        "target_models": ["claude-haiku-4-5"],
        "agent_filter": ["sector_quant", "ic_cio", "sector_qual",
                         "sector_peer_review", "macro_economist", "thesis_update"],
        "would_replay": 24,
        "would_replay_keys": [],
    }


# ── Status envelope ──────────────────────────────────────────────────────


class TestStatusEnvelope:
    def test_ok_when_no_failures(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.batch.compute_and_emit_concordance",
                   return_value=_ok_summary()):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "OK"
        assert result["summary"]["artifacts_discovered"] == 24

    def test_partial_when_replay_failures_present(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.batch.compute_and_emit_concordance",
                   return_value=_partial_summary()):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "PARTIAL"
        assert result["summary"]["per_target_model"][0]["replay_failures"]

    def test_error_when_compute_raises(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.batch.compute_and_emit_concordance",
                   side_effect=RuntimeError("S3 unreachable")):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "ERROR"
        assert "S3 unreachable" in result["error"]


# ── Event payload pass-through ───────────────────────────────────────────


class TestEventPayloadThreading:
    def test_default_target_models(self, handler_mod):
        captured = {}

        def fake_compute(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.batch.compute_and_emit_concordance",
                   side_effect=fake_compute):
            handler_mod.handler({}, context=None)

        # alpha-engine-config-I2997 (2026-07-19): default target model
        # migrated off direct Anthropic to OpenRouter/DeepSeek V4 Flash.
        assert captured["target_models"] == ["deepseek/deepseek-v4-flash"]

    def test_csv_string_target_models_split(self, handler_mod):
        captured = {}

        def fake_compute(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.batch.compute_and_emit_concordance",
                   side_effect=fake_compute):
            handler_mod.handler(
                {"target_models": "claude-haiku-4-5,claude-sonnet-4-6"},
                context=None,
            )
        assert captured["target_models"] == [
            "claude-haiku-4-5", "claude-sonnet-4-6",
        ]

    def test_list_target_models_pass_through(self, handler_mod):
        captured = {}

        def fake_compute(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.batch.compute_and_emit_concordance",
                   side_effect=fake_compute):
            handler_mod.handler(
                {"target_models": ["claude-haiku-4-5", "claude-sonnet-4-6"]},
                context=None,
            )
        assert captured["target_models"] == [
            "claude-haiku-4-5", "claude-sonnet-4-6",
        ]

    def test_end_time_iso_parsed(self, handler_mod):
        captured = {}

        def fake_compute(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.batch.compute_and_emit_concordance",
                   side_effect=fake_compute):
            handler_mod.handler(
                {"end_time_iso": "2026-05-09T00:00:00Z"},
                context=None,
            )
        assert captured["end_time"] == datetime(
            2026, 5, 9, 0, 0, tzinfo=timezone.utc,
        )

    def test_dry_run_disables_metric_emission(self, handler_mod):
        captured = {}

        def fake_compute(**kwargs):
            captured.update(kwargs)
            return _dry_run_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.batch.compute_and_emit_concordance",
                   side_effect=fake_compute):
            handler_mod.handler({"dry_run": True}, context=None)

        assert captured["emit_metrics"] is False
        assert captured["dry_run"] is True

    def test_max_artifacts_default_capped_to_lambda_budget(self, handler_mod):
        """Default cap is 150 (chosen to fit the 900s Lambda timeout
        at ~3-5 sec/replay). Mismatching this value with the timeout
        is a real failure mode — the Lambda will hard-kill mid-batch
        and lose partial signal."""
        captured = {}

        def fake_compute(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.batch.compute_and_emit_concordance",
                   side_effect=fake_compute):
            handler_mod.handler({}, context=None)

        assert captured["max_artifacts"] == 150

    def test_max_artifacts_event_override(self, handler_mod):
        captured = {}

        def fake_compute(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.batch.compute_and_emit_concordance",
                   side_effect=fake_compute):
            handler_mod.handler({"max_artifacts": 50}, context=None)

        assert captured["max_artifacts"] == 50

    def test_window_days_event_override(self, handler_mod):
        captured = {}

        def fake_compute(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.batch.compute_and_emit_concordance",
                   side_effect=fake_compute):
            handler_mod.handler({"window_days": 28}, context=None)

        assert captured["window_days"] == 28

    def test_agents_csv_string_split(self, handler_mod):
        captured = {}

        def fake_compute(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.batch.compute_and_emit_concordance",
                   side_effect=fake_compute):
            handler_mod.handler(
                {"agents": "sector_quant,ic_cio"},
                context=None,
            )
        assert captured["agent_filter"] == ["sector_quant", "ic_cio"]


# ── Shell-run dry path (Saturday-SF keystone) ────────────────────────────


class TestShellRunDryPath:
    """`dry_run_llm: true` (the canonical keystone shell-run key) must
    short-circuit BEFORE the replay scan: no compute_and_emit_concordance
    call (so no decision_artifacts S3 discovery, no langchain_anthropic /
    target-model call, no CloudWatch metric emit, no S3 summary
    persist), boot + module imports still run, and a benign success
    envelope is returned."""

    def test_dry_run_llm_short_circuits_before_scan(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init") as m_init, \
             patch("replay.batch.compute_and_emit_concordance") as m_compute:
            result = handler_mod.handler({"dry_run_llm": True}, context=None)

        # Boot/init still ran for real (the keystone's whole point).
        m_init.assert_called_once()
        # The replay scan / Anthropic / S3+CW path was never entered.
        m_compute.assert_not_called()
        # SF (Catch-wrapped, non-blocking) treats this as success.
        assert result["status"] == "DRY_RUN"
        assert result["dry_run"] is True
        assert result["handler"] == "lambda_concordance"
        assert "duration_seconds" in result

    def test_dry_run_llm_string_true_coerced(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.batch.compute_and_emit_concordance") as m_compute:
            result = handler_mod.handler({"dry_run_llm": "true"}, context=None)
        m_compute.assert_not_called()
        assert result["status"] == "DRY_RUN"

    def test_dry_run_llm_false_takes_real_path(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.batch.compute_and_emit_concordance",
                   return_value=_ok_summary()) as m_compute:
            result = handler_mod.handler({"dry_run_llm": False}, context=None)
        m_compute.assert_called_once()
        assert result["status"] == "OK"

    def test_absent_dry_run_llm_takes_real_path(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.batch.compute_and_emit_concordance",
                   return_value=_ok_summary()) as m_compute:
            result = handler_mod.handler({}, context=None)
        m_compute.assert_called_once()
        assert result["status"] == "OK"

    def test_legacy_dry_run_key_still_takes_real_path(self, handler_mod):
        """The pre-existing `dry_run` (compute-but-don't-emit-metrics)
        semantic is preserved — it must NOT short-circuit the scan."""
        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.batch.compute_and_emit_concordance",
                   return_value=_ok_summary()) as m_compute:
            result = handler_mod.handler({"dry_run": True}, context=None)
        m_compute.assert_called_once()
        assert m_compute.call_args.kwargs["emit_metrics"] is False
        assert result["status"] == "OK"
