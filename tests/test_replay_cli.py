"""Smoke tests for the replay CLI entry point.

Asserts argparse + happy-path JSON output without actually invoking
Anthropic — the underlying replay_artifact / compute_and_emit_concordance
calls are patched.

Two subcommands tested:

  * ``single`` — replay one captured artifact under a target model.
  * ``batch``  — date-range × target-models concordance pipeline.
"""

from __future__ import annotations

import json
from unittest.mock import patch

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
        comparison={
            "agreement_score": 0.85,
            "diff_summary": "top5_jaccard=0.85 |overlap|=4",
            "scorer": "sector_quant",
            "agent_id_base": "sector_quant",
        },
    )


# ── Argparse top-level ───────────────────────────────────────────────────


class TestArgparseTopLevel:
    def test_no_subcommand_errors(self):
        from replay.cli import main

        with pytest.raises(SystemExit):
            main([])

    def test_unknown_subcommand_errors(self):
        from replay.cli import main

        with pytest.raises(SystemExit):
            main(["walk"])  # not a subcommand


# ── single subcommand ────────────────────────────────────────────────────


class TestSingleMode:
    def test_required_args_enforced(self):
        from replay.cli import main

        with pytest.raises(SystemExit):
            main(["single"])  # missing both --artifact-key + --target-model

    def test_happy_path_returns_zero(self, capsys):
        from replay.cli import main

        with patch("replay.cli.replay_artifact",
                   return_value=_stub_replay_output()):
            rc = main([
                "single",
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
        assert summary["agreement_score"] == 0.85
        assert summary["error"] is None

    def test_replay_error_returns_nonzero(self, capsys):
        from replay.cli import main

        with patch("replay.cli.replay_artifact",
                   return_value=_stub_replay_output(error="Anthropic 500")):
            rc = main([
                "single",
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

        with patch("replay.cli.replay_artifact", side_effect=fake_replay):
            main(["single", "--artifact-key", "k.json", "--target-model", "m"])
        assert captured["persist"] is True

        captured.clear()
        with patch("replay.cli.replay_artifact", side_effect=fake_replay):
            main([
                "single", "--artifact-key", "k.json", "--target-model", "m",
                "--no-persist",
            ])
        assert captured["persist"] is False


# ── batch subcommand ─────────────────────────────────────────────────────


def _stub_batch_summary() -> dict:
    return {
        "window_start": "2026-03-14T00:00:00+00:00",
        "window_end": "2026-05-09T00:00:00+00:00",
        "agent_filter": ["sector_quant", "ic_cio"],
        "artifacts_discovered": 12,
        "per_target_model": [
            {
                "target_model": "claude-haiku-4-5",
                "n_artifacts_replayed": 12,
                "agents_analyzed": 2,
                "per_agent": [
                    {"agent_id_base": "sector_quant", "n": 8, "mean": 0.92},
                    {"agent_id_base": "ic_cio", "n": 4, "mean": 0.65},
                ],
                "agents_skipped_thin_sample": [],
                "replay_failures": [],
                "cost": {"input_tokens": 1000, "output_tokens": 500},
                "summary_key": "decision_artifacts/_replay_summary/2026-05-09/claude-haiku-4-5.json",
            },
        ],
    }


class TestBatchMode:
    def test_required_target_models(self):
        from replay.cli import main

        with pytest.raises(SystemExit):
            main(["batch"])  # missing --target-models

    def test_happy_path_prints_summary(self, capsys):
        from replay.cli import main

        with patch("replay.batch.compute_and_emit_concordance",
                   return_value=_stub_batch_summary()):
            rc = main([
                "batch",
                "--target-models", "claude-haiku-4-5",
            ])
        assert rc == 0
        summary = json.loads(capsys.readouterr().out)
        assert summary["artifacts_discovered"] == 12
        assert len(summary["per_target_model"]) == 1

    def test_target_models_csv_split(self):
        from replay.cli import main

        captured = {}

        def fake(**kwargs):
            captured.update(kwargs)
            return _stub_batch_summary()

        with patch("replay.batch.compute_and_emit_concordance", side_effect=fake):
            main([
                "batch",
                "--target-models", "claude-haiku-4-5,claude-sonnet-4-6",
            ])

        assert captured["target_models"] == [
            "claude-haiku-4-5", "claude-sonnet-4-6",
        ]

    def test_agents_csv_split(self):
        from replay.cli import main

        captured = {}

        def fake(**kwargs):
            captured.update(kwargs)
            return _stub_batch_summary()

        with patch("replay.batch.compute_and_emit_concordance", side_effect=fake):
            main([
                "batch",
                "--target-models", "claude-haiku-4-5",
                "--agents", "sector_quant,ic_cio",
            ])

        assert captured["agent_filter"] == ["sector_quant", "ic_cio"]

    def test_dry_run_threads_through(self):
        from replay.cli import main

        captured = {}

        def fake(**kwargs):
            captured.update(kwargs)
            return _stub_batch_summary()

        with patch("replay.batch.compute_and_emit_concordance", side_effect=fake):
            main([
                "batch",
                "--target-models", "claude-haiku-4-5",
                "--dry-run",
            ])
        assert captured["dry_run"] is True

    def test_no_emit_metrics_threads_through(self):
        from replay.cli import main

        captured = {}

        def fake(**kwargs):
            captured.update(kwargs)
            return _stub_batch_summary()

        with patch("replay.batch.compute_and_emit_concordance", side_effect=fake):
            main([
                "batch",
                "--target-models", "claude-haiku-4-5",
                "--no-emit-metrics",
            ])
        assert captured["emit_metrics"] is False
