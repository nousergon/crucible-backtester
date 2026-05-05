"""Unit tests for the single-artifact replay runner.

Covers:

- Happy path: structured (tool_use) replay extracts the input dict.
- Text fallback: replay without tool_definitions returns text under
  ``replay_output_kind="text"``.
- Error path: SDK exception is captured on the artifact rather than
  raising — replay never blows up the caller.
- S3 persistence: replay artifact lands at the documented prefix +
  filename shape.
- ``persist=False`` skips the S3 write but still returns ReplayOutput.
- max_tokens override is propagated to the SDK call.
- Token usage extraction handles None / missing fields gracefully.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ── Fixtures ─────────────────────────────────────────────────────────────


def _make_captured_artifact(
    *,
    run_id: str = "run-abc",
    agent_id: str = "sector_quant:technology",
    model_name: str = "claude-sonnet-4-6",
    user_prompt: str = "Pick top 5 tech names.",
    tool_definitions: list[dict] | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "timestamp": "2026-05-03T12:00:00Z",
        "agent_id": agent_id,
        "model_metadata": {
            "model_name": model_name,
            "input_tokens": 100,
            "output_tokens": 50,
            "cost_usd": 0.001,
        },
        "full_prompt_context": {
            "system_prompt": "You are a sector analyst.",
            "user_prompt": user_prompt,
            "tool_definitions": tool_definitions or [],
        },
        "input_data_snapshot": {"sector": "technology"},
        "agent_output": {
            "ranked_picks": [
                {"ticker": "NVDA", "rationale": "AI tailwind", "quant_score": 88},
            ],
        },
    }


def _make_s3_stub(artifact: dict) -> MagicMock:
    """Stub S3 client that returns the given artifact and records puts."""
    s3 = MagicMock()
    body = MagicMock()
    body.read.return_value = json.dumps(artifact).encode("utf-8")
    s3.get_object.return_value = {"Body": body}
    s3.put_object = MagicMock()
    return s3


def _make_anthropic_response(
    *,
    tool_use_input: dict | None = None,
    text: str | None = None,
    input_tokens: int = 200,
    output_tokens: int = 80,
) -> SimpleNamespace:
    """Stand-in for ``anthropic.types.Message``. SimpleNamespace mirrors
    the SDK's attribute-access shape (``.content``, ``.usage``)."""
    blocks: list[SimpleNamespace] = []
    if tool_use_input is not None:
        blocks.append(SimpleNamespace(type="tool_use", input=tool_use_input))
    if text is not None:
        blocks.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(
        content=blocks,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


# ── Happy path: structured replay ────────────────────────────────────────


class TestStructuredReplay:
    def test_extracts_tool_use_input(self):
        from replay.runner import replay_artifact

        tool_def = {
            "name": "emit_picks",
            "description": "Emit ranked picks.",
            "input_schema": {"type": "object"},
        }
        artifact = _make_captured_artifact(tool_definitions=[tool_def])
        s3 = _make_s3_stub(artifact)

        anth = MagicMock()
        anth.messages.create.return_value = _make_anthropic_response(
            tool_use_input={"ranked_picks": [{"ticker": "AAPL", "score": 75}]},
        )

        replay = replay_artifact(
            artifact_key="decision_artifacts/2026/05/03/x/run-abc.json",
            target_model="claude-haiku-4-5",
            s3_client=s3,
            anthropic_client=anth,
        )

        assert replay.replay_output_kind == "structured"
        assert replay.replay_output == {
            "ranked_picks": [{"ticker": "AAPL", "score": 75}]
        }
        assert replay.replay_error is None
        assert replay.original_model == "claude-sonnet-4-6"
        assert replay.replay_model == "claude-haiku-4-5"

    def test_invocation_carries_system_user_tools(self):
        from replay.runner import replay_artifact

        tool_def = {"name": "emit", "description": "Emit.", "input_schema": {}}
        artifact = _make_captured_artifact(
            user_prompt="Pick 5 tech names.", tool_definitions=[tool_def],
        )
        s3 = _make_s3_stub(artifact)
        anth = MagicMock()
        anth.messages.create.return_value = _make_anthropic_response(
            tool_use_input={"ok": True},
        )

        replay_artifact(
            artifact_key="k.json",
            target_model="claude-haiku-4-5",
            s3_client=s3, anthropic_client=anth,
        )

        kwargs = anth.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-haiku-4-5"
        assert kwargs["system"] == "You are a sector analyst."
        assert kwargs["messages"] == [
            {"role": "user", "content": "Pick 5 tech names."}
        ]
        assert kwargs["tools"] == [tool_def]
        # tool_choice must force structured output when tools are provided.
        assert kwargs["tool_choice"] == {"type": "any"}

    def test_no_tool_choice_when_no_tools(self):
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact(tool_definitions=[])
        s3 = _make_s3_stub(artifact)
        anth = MagicMock()
        anth.messages.create.return_value = _make_anthropic_response(
            text="some free-form text",
        )

        replay_artifact(
            artifact_key="k.json",
            target_model="claude-haiku-4-5",
            s3_client=s3, anthropic_client=anth,
        )

        kwargs = anth.messages.create.call_args.kwargs
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs

    def test_max_tokens_override(self):
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact()
        s3 = _make_s3_stub(artifact)
        anth = MagicMock()
        anth.messages.create.return_value = _make_anthropic_response(text="x")

        replay_artifact(
            artifact_key="k.json", target_model="claude-haiku-4-5",
            max_tokens=4096,
            s3_client=s3, anthropic_client=anth,
        )

        assert anth.messages.create.call_args.kwargs["max_tokens"] == 4096


# ── Text fallback ────────────────────────────────────────────────────────


class TestTextFallback:
    def test_text_only_response_marked_as_text(self):
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact(tool_definitions=[])
        s3 = _make_s3_stub(artifact)
        anth = MagicMock()
        anth.messages.create.return_value = _make_anthropic_response(
            text="Sonnet replied with prose.",
        )

        replay = replay_artifact(
            artifact_key="k.json", target_model="claude-haiku-4-5",
            s3_client=s3, anthropic_client=anth,
        )

        assert replay.replay_output_kind == "text"
        assert replay.replay_output == {"_text": "Sonnet replied with prose."}


# ── Error path ───────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_sdk_exception_captured_not_raised(self):
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact()
        s3 = _make_s3_stub(artifact)
        anth = MagicMock()
        anth.messages.create.side_effect = RuntimeError("Anthropic 500")

        replay = replay_artifact(
            artifact_key="k.json", target_model="claude-haiku-4-5",
            s3_client=s3, anthropic_client=anth,
        )

        assert replay.replay_output_kind == "error"
        assert replay.replay_error == "Anthropic 500"
        assert replay.replay_output == {}
        # Persisted even on error so operators can find the failure.
        assert s3.put_object.called

    def test_empty_content_blocks_marked_error(self):
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact()
        s3 = _make_s3_stub(artifact)
        anth = MagicMock()
        anth.messages.create.return_value = _make_anthropic_response()

        replay = replay_artifact(
            artifact_key="k.json", target_model="claude-haiku-4-5",
            s3_client=s3, anthropic_client=anth,
        )

        assert replay.replay_output_kind == "error"
        assert "no content blocks" in (replay.replay_error or "")


# ── S3 persistence ───────────────────────────────────────────────────────


class TestPersistence:
    def test_persists_to_canonical_key(self):
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact(
            run_id="run-xyz", model_name="claude-sonnet-4-6",
        )
        s3 = _make_s3_stub(artifact)
        anth = MagicMock()
        anth.messages.create.return_value = _make_anthropic_response(
            tool_use_input={"ranked_picks": []},
        )

        replay_artifact(
            artifact_key="src.json", target_model="claude-haiku-4-5",
            s3_client=s3, anthropic_client=anth,
        )

        put_call = s3.put_object.call_args
        assert put_call.kwargs["Bucket"] == "alpha-engine-research"
        # Key shape: {prefix}/{run_id}/{orig}_vs_{target}.json
        key = put_call.kwargs["Key"]
        assert key.startswith("decision_artifacts/_replay/run-xyz/")
        assert "claude-sonnet-4-6_vs_claude-haiku-4-5.json" in key

    def test_no_persist_skips_put_object(self):
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact()
        s3 = _make_s3_stub(artifact)
        anth = MagicMock()
        anth.messages.create.return_value = _make_anthropic_response(
            tool_use_input={"ok": True},
        )

        replay = replay_artifact(
            artifact_key="k.json", target_model="claude-haiku-4-5",
            s3_client=s3, anthropic_client=anth,
            persist=False,
        )

        s3.put_object.assert_not_called()
        # ReplayOutput still populated for the caller.
        assert replay.replay_output_kind == "structured"
        assert replay.replay_output == {"ok": True}

    def test_model_name_with_colon_sanitized_in_key(self):
        from replay.runner import replay_artifact

        # Some model identifiers carry colons (e.g. ":live" alias). Make
        # sure the S3 key path doesn't break.
        artifact = _make_captured_artifact(model_name="claude-sonnet-4-6:live")
        s3 = _make_s3_stub(artifact)
        anth = MagicMock()
        anth.messages.create.return_value = _make_anthropic_response(
            tool_use_input={"ok": True},
        )

        replay_artifact(
            artifact_key="k.json", target_model="claude-haiku-4-5",
            s3_client=s3, anthropic_client=anth,
        )

        key = s3.put_object.call_args.kwargs["Key"]
        # Colons replaced with hyphens in the filename portion.
        assert ":" not in key.rsplit("/", 1)[-1]


# ── Usage extraction ─────────────────────────────────────────────────────


class TestUsageExtraction:
    def test_token_counts_carry_through(self):
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact()
        s3 = _make_s3_stub(artifact)
        anth = MagicMock()
        anth.messages.create.return_value = _make_anthropic_response(
            tool_use_input={"ok": True},
            input_tokens=1234, output_tokens=567,
        )

        replay = replay_artifact(
            artifact_key="k.json", target_model="claude-haiku-4-5",
            s3_client=s3, anthropic_client=anth,
        )

        assert replay.replay_cost["input_tokens"] == 1234
        assert replay.replay_cost["output_tokens"] == 567

    def test_missing_usage_returns_empty_dict(self):
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact()
        s3 = _make_s3_stub(artifact)
        anth = MagicMock()
        # Response without a .usage attribute (older SDK fixtures).
        anth.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", input={"ok": True})],
            usage=None,
        )

        replay = replay_artifact(
            artifact_key="k.json", target_model="claude-haiku-4-5",
            s3_client=s3, anthropic_client=anth,
        )

        assert replay.replay_cost == {}
        # Replay still succeeded — missing usage is a non-fatal observation.
        assert replay.replay_output_kind == "structured"


# ── ReplayOutput dataclass ───────────────────────────────────────────────


class TestReplayOutputSerialization:
    def test_to_dict_contains_all_documented_fields(self):
        from replay.runner import ReplayOutput

        ro = ReplayOutput(
            original_run_id="r1",
            original_agent_id="a1",
            original_model="m1",
            replay_model="m2",
        )
        d = ro.to_dict()
        for field_name in (
            "schema_version",
            "original_run_id",
            "original_agent_id",
            "original_model",
            "original_artifact_key",
            "original_output",
            "replay_model",
            "replay_timestamp",
            "replay_output",
            "replay_output_kind",
            "replay_cost",
            "replay_latency_ms",
            "replay_error",
            "comparison",
        ):
            assert field_name in d
