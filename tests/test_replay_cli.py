"""Smoke tests for the replay CLI entry point.

Asserts argparse + happy-path JSON output without actually invoking
Anthropic — the underlying ``replay_artifact`` is patched.
"""

from __future__ import annotations

import io
import json
import sys
from unittest.mock import MagicMock, patch

import pytest


def _stub_replay_output(error: str | None = None):
    from replay.runner import ReplayOutput

    return ReplayOutput(
        original_run_id="r1",
        original_agent_id="sector_quant:tech",
        original_model="claude-sonnet-4-6",
        replay_model="claude-haiku-4-5",
        replay_output_kind="structured" if error is None else "error",
        replay_latency_ms=1234,
        replay_cost={"input_tokens": 100, "output_tokens": 50},
        replay_error=error,
    )


class TestCLI:
    def test_required_args_enforced(self):
        from replay.cli import main

        with pytest.raises(SystemExit):
            main([])  # missing both --artifact-key and --target-model

    def test_happy_path_returns_zero(self, capsys):
        from replay.cli import main

        with patch("replay.cli.replay_artifact",
                   return_value=_stub_replay_output()):
            rc = main([
                "--artifact-key", "k.json",
                "--target-model", "claude-haiku-4-5",
                "--no-persist",
            ])
        assert rc == 0
        out = capsys.readouterr().out
        summary = json.loads(out)
        assert summary["agent_id"] == "sector_quant:tech"
        assert summary["replay_model"] == "claude-haiku-4-5"
        assert summary["kind"] == "structured"
        assert summary["error"] is None

    def test_replay_error_returns_nonzero(self, capsys):
        from replay.cli import main

        with patch("replay.cli.replay_artifact",
                   return_value=_stub_replay_output(error="Anthropic 500")):
            rc = main([
                "--artifact-key", "k.json",
                "--target-model", "claude-haiku-4-5",
                "--no-persist",
            ])
        assert rc == 1
        summary = json.loads(capsys.readouterr().out)
        assert summary["error"] == "Anthropic 500"
        assert summary["kind"] == "error"

    def test_persist_flag_threads_through(self):
        from replay.cli import main

        captured = {}

        def fake_replay(**kwargs):
            captured.update(kwargs)
            return _stub_replay_output()

        # Default (persist).
        with patch("replay.cli.replay_artifact", side_effect=fake_replay):
            main(["--artifact-key", "k.json", "--target-model", "m"])
        assert captured["persist"] is True

        # --no-persist.
        captured.clear()
        with patch("replay.cli.replay_artifact", side_effect=fake_replay):
            main([
                "--artifact-key", "k.json", "--target-model", "m",
                "--no-persist",
            ])
        assert captured["persist"] is False
