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
        captured = {}

        def fake_compute(**kwargs):
            captured.update(kwargs)
            return _ok_summary()

        with patch.object(handler_mod, "_ensure_init"), \
             patch("replay.counterfactual.compute_and_emit",
                   side_effect=fake_compute):
            handler_mod.handler({}, context=None)

        assert captured["window_days"] == 56
        assert captured["max_depth"] == 3

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
