"""Unit tests for the single-artifact replay runner.

Covers:

- Happy path: structured replay extracts the parsed Pydantic instance
  and dumps it for the comparison + persistence layers.
- Pydantic validation error path: target model emits structurally
  divergent output → captured on the artifact as replay_error.
- Generic SDK error path: langchain raises → captured, not propagated.
- S3 persistence: replay artifact lands at the documented prefix +
  filename shape.
- ``persist=False`` skips the S3 write but still returns ReplayOutput.
- Unknown agent_id family → skipped with marker.
- ``chat_anthropic_factory`` injection point exercised end-to-end.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock


# ── Fixtures ─────────────────────────────────────────────────────────────


def _make_captured_artifact(
    *,
    run_id: str = "run-abc",
    agent_id: str = "sector_quant:technology",
    model_name: str = "claude-sonnet-4-6",
    user_prompt: str = "Pick top 5 tech names.",
) -> dict:
    """Captured DecisionArtifact dict in the lib's schema shape."""
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
            "tool_definitions": [],
        },
        "input_data_snapshot": {"sector": "technology"},
        "agent_output": {
            "ranked_picks": [
                {"ticker": "NVDA", "rationale": "AI tailwind", "quant_score": 88},
            ],
        },
    }


def _make_s3_stub(artifact: dict) -> MagicMock:
    """Stub S3 client returning the given artifact and recording puts."""
    s3 = MagicMock()
    body = MagicMock()
    body.read.return_value = json.dumps(artifact).encode("utf-8")
    s3.get_object.return_value = {"Body": body}
    s3.put_object = MagicMock()
    return s3


def _make_chat_anthropic_factory(
    *,
    parsed: object | None = None,
    parsing_error: Exception | None = None,
    usage: dict | None = None,
    raise_on_invoke: Exception | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Build a fake ``ChatAnthropic`` factory whose ``with_structured_output``
    returns a runnable that responds to ``.invoke()`` with the
    langchain include_raw=True shape:

        {"raw": AIMessage-like, "parsed": Pydantic | None, "parsing_error": Exception | None}

    Returns ``(factory, structured_runnable)`` so tests can also assert
    on the call args.
    """
    structured = MagicMock()
    raw = SimpleNamespace(
        response_metadata={"usage": usage if usage is not None else {}}
    )
    if raise_on_invoke is not None:
        structured.invoke.side_effect = raise_on_invoke
    else:
        structured.invoke.return_value = {
            "raw": raw,
            "parsed": parsed,
            "parsing_error": parsing_error,
        }

    llm = MagicMock()
    llm.with_structured_output.return_value = structured

    factory = MagicMock(return_value=llm)
    return factory, structured


# ── Happy path: structured replay ────────────────────────────────────────


class TestStructuredReplay:
    def test_extracts_model_dump_from_parsed_pydantic(self):
        from alpha_engine_lib.agent_schemas import QuantAnalystOutput
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact()
        s3 = _make_s3_stub(artifact)

        parsed_instance = QuantAnalystOutput(
            ranked_picks=[
                {"ticker": "AAPL", "rationale": "FCF strong", "quant_score": 75},
            ],
        )
        factory, _ = _make_chat_anthropic_factory(
            parsed=parsed_instance,
            usage={"input_tokens": 200, "output_tokens": 80},
        )

        replay = replay_artifact(
            artifact_key="decision_artifacts/2026/05/03/x/run-abc.json",
            target_model="claude-haiku-4-5",
            s3_client=s3,
            chat_anthropic_factory=factory,
        )

        assert replay.replay_output_kind == "structured"
        assert replay.replay_output["ranked_picks"][0]["ticker"] == "AAPL"
        assert replay.replay_error is None
        assert replay.original_model == "claude-sonnet-4-6"
        assert replay.replay_model == "claude-haiku-4-5"

    def test_factory_called_with_target_model_and_max_tokens(self):
        from alpha_engine_lib.agent_schemas import QuantAnalystOutput
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact()
        s3 = _make_s3_stub(artifact)
        factory, _ = _make_chat_anthropic_factory(
            parsed=QuantAnalystOutput(ranked_picks=[]),
        )

        replay_artifact(
            artifact_key="k.json",
            target_model="claude-haiku-4-5",
            max_tokens=4096,
            s3_client=s3, chat_anthropic_factory=factory,
        )

        factory.assert_called_once()
        call_kwargs = factory.call_args.kwargs
        assert call_kwargs["model"] == "claude-haiku-4-5"
        assert call_kwargs["max_tokens"] == 4096

    def test_with_structured_output_resolves_canonical_schema(self):
        """Replay must call with_structured_output(SchemaClass, include_raw=True)
        with the schema RESOLVED FROM THE CAPTURED agent_id — confirming
        invocation isomorphism with how production agents call the model."""
        from alpha_engine_lib.agent_schemas import (
            QuantAnalystOutput, JointFinalizationOutput, CIORawOutput,
            MacroEconomistRawOutput, HeldThesisUpdateLLMOutput,
            QualAnalystOutput,
        )
        from replay.runner import replay_artifact

        cases = [
            ("sector_quant:tech", QuantAnalystOutput),
            ("sector_qual:healthcare", QualAnalystOutput),
            ("sector_peer_review:financials", JointFinalizationOutput),
            ("macro_economist", MacroEconomistRawOutput),
            ("ic_cio", CIORawOutput),
            ("thesis_update:AAPL", HeldThesisUpdateLLMOutput),
        ]
        for agent_id, expected_schema in cases:
            artifact = _make_captured_artifact(agent_id=agent_id)
            s3 = _make_s3_stub(artifact)
            factory, _ = _make_chat_anthropic_factory(
                # model_construct bypasses validation — fixture only needs
                # an instance, not a schema-conformant one.
                parsed=expected_schema.model_construct(),
            )

            replay_artifact(
                artifact_key="k.json",
                target_model="claude-haiku-4-5",
                s3_client=s3, chat_anthropic_factory=factory,
                persist=False,
            )

            llm = factory.return_value
            schema_arg = llm.with_structured_output.call_args.args[0]
            assert schema_arg is expected_schema, (
                f"agent_id={agent_id} resolved to {schema_arg}, "
                f"expected {expected_schema}"
            )
            # include_raw must be True so the runner can extract token usage.
            assert llm.with_structured_output.call_args.kwargs["include_raw"] is True

    def test_invoke_called_with_system_and_user_messages(self):
        from alpha_engine_lib.agent_schemas import QuantAnalystOutput
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact(user_prompt="Pick 5 tech names.")
        s3 = _make_s3_stub(artifact)
        factory, structured = _make_chat_anthropic_factory(
            parsed=QuantAnalystOutput(ranked_picks=[]),
        )

        replay_artifact(
            artifact_key="k.json",
            target_model="claude-haiku-4-5",
            s3_client=s3, chat_anthropic_factory=factory,
            persist=False,
        )

        messages = structured.invoke.call_args.args[0]
        assert messages == [
            {"role": "system", "content": "You are a sector analyst."},
            {"role": "user", "content": "Pick 5 tech names."},
        ]


# ── Pydantic validation error ────────────────────────────────────────────


class TestPydanticValidationError:
    def test_parsing_error_captured_on_artifact(self):
        """When the target model emits a structurally divergent output,
        with_structured_output(include_raw=True) populates parsing_error
        in the response dict. Replay surfaces that as replay_error
        rather than silently emitting a 0-agreement comparison — exactly
        the silent-drift signal we wanted to capture."""
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact()
        s3 = _make_s3_stub(artifact)
        validation_err = ValueError(
            "ranked_picks.0.quant_score Input should be ≤ 100"
        )
        factory, _ = _make_chat_anthropic_factory(
            parsed=None, parsing_error=validation_err,
        )

        replay = replay_artifact(
            artifact_key="k.json",
            target_model="claude-haiku-4-5",
            s3_client=s3, chat_anthropic_factory=factory,
            persist=False,
        )

        assert replay.replay_output_kind == "error"
        assert "pydantic validation failed" in (replay.replay_error or "")
        assert "quant_score" in (replay.replay_error or "")


# ── Generic SDK error path ───────────────────────────────────────────────


class TestErrorHandling:
    def test_sdk_exception_captured_not_raised(self):
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact()
        s3 = _make_s3_stub(artifact)
        factory, _ = _make_chat_anthropic_factory(
            raise_on_invoke=RuntimeError("Anthropic 500"),
        )

        replay = replay_artifact(
            artifact_key="k.json",
            target_model="claude-haiku-4-5",
            s3_client=s3, chat_anthropic_factory=factory,
        )

        assert replay.replay_output_kind == "error"
        assert "Anthropic 500" in (replay.replay_error or "")
        assert s3.put_object.called

    def test_no_parsed_object_marked_error(self):
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact()
        s3 = _make_s3_stub(artifact)
        factory, _ = _make_chat_anthropic_factory(
            parsed=None, parsing_error=None,
        )

        replay = replay_artifact(
            artifact_key="k.json",
            target_model="claude-haiku-4-5",
            s3_client=s3, chat_anthropic_factory=factory,
            persist=False,
        )

        assert replay.replay_output_kind == "error"
        assert "no parsed object" in (replay.replay_error or "")


# ── Unknown agent_id family ──────────────────────────────────────────────


class TestUnknownAgentSkip:
    def test_unknown_agent_id_skips_replay_with_marker(self):
        """Replay only runs against the 6 canonical agent families that
        have a registered schema in alpha_engine_lib.agent_schemas. An
        unknown agent_id (e.g. a future agent type) is skipped with a
        marker rather than attempting a free-form replay."""
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact(agent_id="brand_new_agent")
        s3 = _make_s3_stub(artifact)
        factory, _ = _make_chat_anthropic_factory(parsed=None)

        replay = replay_artifact(
            artifact_key="k.json",
            target_model="claude-haiku-4-5",
            s3_client=s3, chat_anthropic_factory=factory,
            persist=False,
        )

        assert replay.replay_output_kind == "error"
        assert "no canonical schema" in (replay.replay_error or "")
        assert replay.comparison["scorer"] == "skipped"
        # Factory not invoked — short-circuited before the LLM call.
        factory.assert_not_called()


class TestDeterministicArtifactSkip:
    """alpha-engine-lib v0.10.0 introduced ``DecisionArtifact`` schema_version=2
    with ``model_metadata=None`` for deterministic decisions (e.g.
    ``executor:entry_triggers`` algorithmic agents). Replay-as-model-
    substitution is meaningless for deterministic decisions — there's no
    LLM to swap. The runner must skip with an explicit marker rather
    than crash on ``None.get("model_name")``.
    """

    def _deterministic_artifact(self) -> dict:
        """v2 artifact shape with both LLM fields None — what executor
        captures will look like once L2308 ships."""
        return {
            "schema_version": 2,
            "run_id": "run-2026-05-15",
            "timestamp": "2026-05-15T13:25:00Z",
            "agent_id": "executor:entry_triggers",
            "model_metadata": None,
            "full_prompt_context": None,
            "input_data_snapshot": {
                "ticker": "AAPL",
                "current_price": 175.25,
                "day_high": 178.50,
                "thresholds": {"pullback_pct": 0.02},
            },
            "agent_output": {
                "fired_trigger": "pullback 1.8% from high $178.50",
                "trigger_kind": "pullback",
            },
        }

    def test_deterministic_v2_artifact_skipped_with_marker(self):
        """Critical: the prior code path read
        ``artifact.get("model_metadata", {}).get("model_name")`` which
        raises ``AttributeError`` on None.get(...). This regression test
        pins the explicit-skip behavior introduced for the L2308 arc.
        """
        from replay.runner import replay_artifact

        artifact = self._deterministic_artifact()
        s3 = _make_s3_stub(artifact)
        factory, _ = _make_chat_anthropic_factory(parsed=None)

        replay = replay_artifact(
            artifact_key="k.json",
            target_model="claude-haiku-4-5",
            s3_client=s3, chat_anthropic_factory=factory,
            persist=False,
        )

        assert replay.replay_output_kind == "skipped"
        assert "deterministic decision" in (replay.replay_error or "")
        assert replay.original_model == "deterministic"
        assert replay.original_agent_id == "executor:entry_triggers"
        assert replay.original_output["trigger_kind"] == "pullback"
        # Factory not invoked — no LLM call attempted.
        factory.assert_not_called()

    def test_deterministic_skip_does_not_crash_on_none_model_metadata(self):
        """Anti-regression: pin that the code path before this fix would
        have raised AttributeError on None.get(...). If a future refactor
        reintroduces the old code path, this test catches it.
        """
        from replay.runner import replay_artifact

        artifact = self._deterministic_artifact()
        s3 = _make_s3_stub(artifact)
        factory, _ = _make_chat_anthropic_factory(parsed=None)

        # Must not raise.
        replay = replay_artifact(
            artifact_key="k.json",
            target_model="claude-haiku-4-5",
            s3_client=s3, chat_anthropic_factory=factory,
            persist=False,
        )
        assert replay is not None


# ── S3 persistence ───────────────────────────────────────────────────────


class TestPersistence:
    def test_persists_to_canonical_key(self):
        from alpha_engine_lib.agent_schemas import QuantAnalystOutput
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact(
            run_id="run-xyz", model_name="claude-sonnet-4-6",
        )
        s3 = _make_s3_stub(artifact)
        factory, _ = _make_chat_anthropic_factory(
            parsed=QuantAnalystOutput(ranked_picks=[]),
        )

        replay_artifact(
            artifact_key="src.json", target_model="claude-haiku-4-5",
            s3_client=s3, chat_anthropic_factory=factory,
        )

        put_call = s3.put_object.call_args
        assert put_call.kwargs["Bucket"] == "alpha-engine-research"
        key = put_call.kwargs["Key"]
        assert key.startswith("decision_artifacts/_replay/run-xyz/")
        assert "claude-sonnet-4-6_vs_claude-haiku-4-5.json" in key

    def test_no_persist_skips_put_object(self):
        from alpha_engine_lib.agent_schemas import QuantAnalystOutput
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact()
        s3 = _make_s3_stub(artifact)
        factory, _ = _make_chat_anthropic_factory(
            parsed=QuantAnalystOutput(ranked_picks=[
                {"ticker": "X", "quant_score": 80, "rationale": "ok"},
            ]),
        )

        replay = replay_artifact(
            artifact_key="k.json", target_model="claude-haiku-4-5",
            s3_client=s3, chat_anthropic_factory=factory,
            persist=False,
        )

        s3.put_object.assert_not_called()
        assert replay.replay_output_kind == "structured"
        assert len(replay.replay_output["ranked_picks"]) == 1

    def test_model_name_with_colon_sanitized_in_key(self):
        from alpha_engine_lib.agent_schemas import QuantAnalystOutput
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact(model_name="claude-sonnet-4-6:live")
        s3 = _make_s3_stub(artifact)
        factory, _ = _make_chat_anthropic_factory(
            parsed=QuantAnalystOutput(ranked_picks=[]),
        )

        replay_artifact(
            artifact_key="k.json", target_model="claude-haiku-4-5",
            s3_client=s3, chat_anthropic_factory=factory,
        )

        key = s3.put_object.call_args.kwargs["Key"]
        assert ":" not in key.rsplit("/", 1)[-1]


# ── Usage extraction ─────────────────────────────────────────────────────


class TestUsageExtraction:
    def test_token_counts_carry_through(self):
        from alpha_engine_lib.agent_schemas import QuantAnalystOutput
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact()
        s3 = _make_s3_stub(artifact)
        factory, _ = _make_chat_anthropic_factory(
            parsed=QuantAnalystOutput(ranked_picks=[]),
            usage={
                "input_tokens": 1234, "output_tokens": 567,
                "cache_read_input_tokens": 100,
                "cache_creation_input_tokens": 50,
            },
        )

        replay = replay_artifact(
            artifact_key="k.json", target_model="claude-haiku-4-5",
            s3_client=s3, chat_anthropic_factory=factory,
        )

        assert replay.replay_cost["input_tokens"] == 1234
        assert replay.replay_cost["output_tokens"] == 567
        assert replay.replay_cost["cache_read_input_tokens"] == 100
        assert replay.replay_cost["cache_creation_input_tokens"] == 50

    def test_missing_usage_returns_empty_dict(self):
        from alpha_engine_lib.agent_schemas import QuantAnalystOutput
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact()
        s3 = _make_s3_stub(artifact)
        factory, _ = _make_chat_anthropic_factory(
            parsed=QuantAnalystOutput(ranked_picks=[]),
            usage={},
        )

        replay = replay_artifact(
            artifact_key="k.json", target_model="claude-haiku-4-5",
            s3_client=s3, chat_anthropic_factory=factory,
        )

        assert replay.replay_cost == {}
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
