"""Unit tests for lambda_counterfactual/handler.py.

Mocks compute_and_emit + ssm_secrets so the handler is exercised in
isolation. Mirrors test_lambda_concordance_handler.py shape.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_HANDLER_PATH = _REPO_ROOT / "lambda_counterfactual" / "handler.py"


def _load_handler_module():
    """Import lambda_counterfactual/handler.py without using
    `lambda_counterfactual` as a package name (avoid ambiguity even
    though it's not actually a Python keyword collision)."""
    module_name = "lambda_counterfactual_handler_under_test"
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
        "max_depth": 3,
        "artifacts_discovered": 24,
        "agents_analyzed": 2,
        "agents_skipped_thin_sample": [],
        "agents_unsupported": ["sector_quant", "sector_qual"],
        "load_failures": [],
        "fit_failures": [],
        "per_agent": [
            {
                "agent_id_base": "ic_cio",
                "n_samples": 80,
                "match_rate": 0.92,
                "n_classes": 2,
                "analysis_key": "decision_artifacts/_counterfactual/ic_cio/2026-W19.json",
            },
            {
                "agent_id_base": "macro_economist",
                "n_samples": 8,
                "match_rate": 0.5,
                "n_classes": 4,
                "analysis_key": "decision_artifacts/_counterfactual/macro_economist/2026-W19.json",
            },
        ],
    }


def _partial_summary() -> dict:
    s = _ok_summary()
    s["fit_failures"] = [{"agent_id_base": "ic_cio", "error": "sklearn fit raised"}]
    return s


# ── Status envelope ──────────────────────────────────────────────────────


class TestStatusEnvelope:
    def test_ok_when_no_failures(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.counterfactual.compute_and_emit",
                   return_value=_ok_summary()):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "OK"
        assert result["summary"]["agents_analyzed"] == 2

    def test_partial_when_fit_failures(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.counterfactual.compute_and_emit",
                   return_value=_partial_summary()):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "PARTIAL"

    def test_partial_when_load_failures(self, handler_mod):
        s = _ok_summary()
        s["load_failures"] = [{"key": "x.json", "error": "NoSuchKey"}]
        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.counterfactual.compute_and_emit", return_value=s):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "PARTIAL"

    def test_error_when_compute_raises(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.counterfactual.compute_and_emit",
                   side_effect=RuntimeError("S3 unreachable")):
            result = handler_mod.handler({}, context=None)
        assert result["status"] == "ERROR"
        assert "S3 unreachable" in result["error"]


# ── Event payload pass-through ───────────────────────────────────────────


class TestEventPayloadThreading:
    def test_default_window_days(self, handler_mod):
        """ROADMAP L293 (2026-05-19): default window dropped 56 → 28 days
        to keep the Saturday-SF Counterfactual Lambda under its 600s ceiling
        after the captured-artifact corpus crossed ~32k+ in the 56d window."""
        captured = {}

        def fake_compute(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.counterfactual.compute_and_emit",
                   side_effect=fake_compute):
            handler_mod.handler({}, context=None)

        assert captured["window_days"] == 28
        assert captured["max_depth"] == 3
        # ROADMAP L293 (2026-05-19): second-order bound — per-agent cap
        # defaults to 500 so a future heavy-population-agent backlog
        # can't blow the runtime past the ceiling even at the 28d default.
        assert captured["max_artifacts_per_agent"] == 500

    def test_max_artifacts_per_agent_event_override(self, handler_mod):
        """Explicit override threads through to compute_and_emit."""
        captured = {}

        def fake_compute(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.counterfactual.compute_and_emit",
                   side_effect=fake_compute):
            handler_mod.handler(
                {"max_artifacts_per_agent": 200}, context=None,
            )

        assert captured["max_artifacts_per_agent"] == 200

    def test_max_artifacts_per_agent_none_disables_cap(self, handler_mod):
        """Explicit None in the event payload disables the cap for
        ad-hoc deeper-corpus runs."""
        captured = {}

        def fake_compute(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.counterfactual.compute_and_emit",
                   side_effect=fake_compute):
            handler_mod.handler(
                {"max_artifacts_per_agent": None}, context=None,
            )

        assert captured["max_artifacts_per_agent"] is None

    def test_window_days_event_override(self, handler_mod):
        captured = {}

        def fake_compute(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.counterfactual.compute_and_emit",
                   side_effect=fake_compute):
            handler_mod.handler({"window_days": 28, "max_depth": 5}, context=None)

        assert captured["window_days"] == 28
        assert captured["max_depth"] == 5

    def test_end_time_iso_parsed(self, handler_mod):
        captured = {}

        def fake_compute(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.counterfactual.compute_and_emit",
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
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.counterfactual.compute_and_emit",
                   side_effect=fake_compute):
            handler_mod.handler({"dry_run": True}, context=None)

        assert captured["emit_metrics"] is False

    def test_agents_csv_string_split(self, handler_mod):
        captured = {}

        def fake_compute(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.counterfactual.compute_and_emit",
                   side_effect=fake_compute):
            handler_mod.handler(
                {"agents": "ic_cio,macro_economist"},
                context=None,
            )
        assert captured["agent_filter"] == ["ic_cio", "macro_economist"]

    def test_agents_list_pass_through(self, handler_mod):
        captured = {}

        def fake_compute(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.counterfactual.compute_and_emit",
                   side_effect=fake_compute):
            handler_mod.handler(
                {"agents": ["ic_cio"]},
                context=None,
            )
        assert captured["agent_filter"] == ["ic_cio"]


# ── Shell-run dry path (Saturday-SF keystone) ────────────────────────────


class TestShellRunDryPath:
    """`dry_run_llm: true` (the canonical keystone shell-run key) must
    short-circuit BEFORE the replay scan: no compute_and_emit call (so
    no decision_artifacts S3 discovery, no sklearn fit, no CloudWatch
    metric emit, no S3 per-agent persist), boot + module imports still
    run, and a benign success envelope is returned. No LLM calls exist
    on this handler's path regardless."""

    def test_dry_run_llm_short_circuits_before_scan(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init") as m_init, \
             patch("replay.counterfactual.compute_and_emit") as m_compute:
            result = handler_mod.handler({"dry_run_llm": True}, context=None)

        m_init.assert_called_once()
        m_compute.assert_not_called()
        assert result["status"] == "DRY_RUN"
        assert result["dry_run"] is True
        assert result["handler"] == "lambda_counterfactual"
        assert "duration_seconds" in result

    def test_dry_run_llm_string_true_coerced(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.counterfactual.compute_and_emit") as m_compute:
            result = handler_mod.handler({"dry_run_llm": "1"}, context=None)
        m_compute.assert_not_called()
        assert result["status"] == "DRY_RUN"

    def test_dry_run_llm_false_takes_real_path(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.counterfactual.compute_and_emit",
                   return_value=_ok_summary()) as m_compute:
            result = handler_mod.handler({"dry_run_llm": False}, context=None)
        m_compute.assert_called_once()
        assert result["status"] == "OK"

    def test_absent_dry_run_llm_takes_real_path(self, handler_mod):
        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.counterfactual.compute_and_emit",
                   return_value=_ok_summary()) as m_compute:
            result = handler_mod.handler({}, context=None)
        m_compute.assert_called_once()
        assert result["status"] == "OK"

    def test_legacy_dry_run_key_still_takes_real_path(self, handler_mod):
        """The pre-existing `dry_run` (compute-but-don't-emit-metrics)
        semantic is preserved — it must NOT short-circuit the scan."""
        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.counterfactual.compute_and_emit",
                   return_value=_ok_summary()) as m_compute:
            result = handler_mod.handler({"dry_run": True}, context=None)
        m_compute.assert_called_once()
        assert m_compute.call_args.kwargs["emit_metrics"] is False
        assert result["status"] == "OK"
