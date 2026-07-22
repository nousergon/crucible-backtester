"""Unit tests for the single-artifact replay runner.

Covers:

- Happy path: structured replay extracts the parsed Pydantic instance
  and dumps it for the comparison + persistence layers.
- Schema-validation-failure path: target model emits structurally
  divergent output → captured on the artifact as replay_error.
- Generic transport error path: OpenRouter/SDK raises → captured, not
  propagated.
- S3 persistence: replay artifact lands at the documented prefix +
  filename shape.
- ``persist=False`` skips the S3 write but still returns ReplayOutput.
- Unknown agent_id family → skipped with marker.
- ``client_factory`` injection point exercised end-to-end.

alpha-engine-config-I2997 (2026-07-19): migrated off direct Anthropic
(``langchain_anthropic.ChatAnthropic``) to ``krepis.llm.LLMClient``'s
OpenRouter transport (see ``replay/runner.py``'s module docstring). Mocks
now build a fake ``openai``-shaped transport client via the
``client_factory`` seam (``(spec, api_key) -> client`` exposing
``chat.completions.create``) instead of a fake ``ChatAnthropic``.
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


def _make_krepis_factory(
    *,
    content: str | None = None,
    model: str = "deepseek/deepseek-v4-flash",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost: float | None = None,
    served_provider: str | None = None,
    raise_on_create: Exception | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Build a fake ``krepis.llm.LLMClient`` transport client — the
    ``client_factory`` test seam every alpha-engine-config-I2997 call site
    uses: a callable ``(spec, api_key) -> transport_client`` exposing
    ``chat.completions.create(**kwargs)`` (OpenAI-compatible shape).

    ``content`` is the raw text ``choices[0].message.content`` — under
    ``structured_outputs=False`` (REQUIRED here, see runner.py's module
    docstring) ``krepis.llm`` parses this as JSON (tolerating markdown
    fences) and validates it against the schema.

    ``served_provider`` sets a top-level ``.provider`` attribute on the
    fake response, mirroring OpenRouter's real (non-standard) response
    shape (config#3006) — ``krepis>=0.18.0`` reads this into
    ``LLMResult.served_provider``.

    Returns ``(factory, fake_client)`` so tests can also assert on the
    call args recorded by ``fake_client.chat.completions.create``.
    """
    fake_client = MagicMock()
    if raise_on_create is not None:
        fake_client.chat.completions.create.side_effect = raise_on_create
    else:
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=None,
            cost=cost,
        )
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message)
        resp = SimpleNamespace(
            choices=[choice], model=model, usage=usage,
        )
        if served_provider is not None:
            resp.provider = served_provider
        fake_client.chat.completions.create.return_value = resp
    factory = MagicMock(return_value=fake_client)
    return factory, fake_client


# ── Happy path: structured replay ────────────────────────────────────────


class TestStructuredReplay:
    def test_extracts_model_dump_from_parsed_pydantic(self):
        from nousergon_lib.agent_schemas import QuantAnalystOutput
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact()
        s3 = _make_s3_stub(artifact)

        parsed_instance = QuantAnalystOutput(
            ranked_picks=[
                {"ticker": "AAPL", "rationale": "FCF strong", "quant_score": 75},
            ],
        )
        factory, _ = _make_krepis_factory(
            content=json.dumps(parsed_instance.model_dump()),
            prompt_tokens=200, completion_tokens=80,
        )

        replay = replay_artifact(
            artifact_key="decision_artifacts/2026/05/03/x/run-abc.json",
            target_model="deepseek/deepseek-v4-flash",
            s3_client=s3,
            client_factory=factory,
            api_key="sk-or-test",
        )

        assert replay.replay_output_kind == "structured"
        assert replay.replay_output["ranked_picks"][0]["ticker"] == "AAPL"
        assert replay.replay_error is None
        assert replay.original_model == "claude-sonnet-4-6"
        assert replay.replay_model == "deepseek/deepseek-v4-flash"

    def test_factory_called_with_target_model_and_max_tokens(self):
        from nousergon_lib.agent_schemas import QuantAnalystOutput
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact()
        s3 = _make_s3_stub(artifact)
        factory, _ = _make_krepis_factory(
            content=json.dumps(QuantAnalystOutput(ranked_picks=[]).model_dump()),
        )

        replay_artifact(
            artifact_key="k.json",
            target_model="deepseek/deepseek-v4-flash",
            max_tokens=4096,
            s3_client=s3, client_factory=factory, api_key="sk-or-test",
        )

        # client_factory receives (spec, api_key) — spec carries the
        # resolved model/max_tokens/provider, not bare kwargs.
        factory.assert_called_once()
        spec = factory.call_args.args[0]
        assert spec.provider == "openrouter"
        assert spec.model == "deepseek/deepseek-v4-flash"
        assert spec.max_tokens == 4096
        assert factory.call_args.args[1] == "sk-or-test"

    def test_structured_outputs_false_and_reasoning_excluded(self):
        """REQUIRED, not incidental — see runner.py's module docstring:
        live-verified 2026-07-19 that strict response_format=json_schema
        is unreliable for DeepSeek-family models on OpenRouter."""
        from nousergon_lib.agent_schemas import QuantAnalystOutput
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact()
        s3 = _make_s3_stub(artifact)
        factory, _ = _make_krepis_factory(
            content=json.dumps(QuantAnalystOutput(ranked_picks=[]).model_dump()),
        )

        replay_artifact(
            artifact_key="k.json",
            target_model="deepseek/deepseek-v4-flash",
            s3_client=s3, client_factory=factory, api_key="sk-or-test",
            persist=False,
        )

        spec = factory.call_args.args[0]
        assert spec.structured_outputs is False
        assert spec.reasoning == {"exclude": True}

    def test_resolves_canonical_schema_and_round_trips_it(self):
        """Replay must validate against the schema RESOLVED FROM THE
        CAPTURED agent_id — confirming the canonical contract is enforced
        per agent family. Indirect check (no with_structured_output call
        to introspect under the new transport): feed back the SPECIFIC
        expected schema's own field shape and confirm it round-trips as
        "structured" (a wrong schema would fail validation against a
        mismatched field set)."""
        from nousergon_lib.agent_schemas import (
            QuantAnalystOutput, JointFinalizationOutput, CIORawOutput,
            CIORawDecision, MacroEconomistRawOutput, HeldThesisUpdateLLMOutput,
            QualAnalystOutput,
        )
        from replay.runner import replay_artifact

        cases = [
            ("sector_quant:tech", QuantAnalystOutput, {"ranked_picks": []}),
            ("sector_qual:healthcare", QualAnalystOutput, {}),
            ("sector_peer_review:financials", JointFinalizationOutput, {}),
            ("macro_economist", MacroEconomistRawOutput, {}),
            # CIORawOutput.decisions has min_length=1 — model_construct's
            # default empty list fails real validation on round-trip.
            ("ic_cio", CIORawOutput, {
                "decisions": [CIORawDecision(ticker="AAPL", decision="ADVANCE")],
            }),
            ("thesis_update:AAPL", HeldThesisUpdateLLMOutput, {}),
        ]
        for agent_id, expected_schema, minimal_payload in cases:
            artifact = _make_captured_artifact(agent_id=agent_id)
            s3 = _make_s3_stub(artifact)
            # model_construct + model_dump bypasses validation on the
            # BUILD side — fixture only needs a schema-shaped JSON string,
            # not a fully-conformant instance; extra=allow schemas accept
            # the minimal payload fine.
            payload = expected_schema.model_construct(**minimal_payload).model_dump()
            factory, _ = _make_krepis_factory(content=json.dumps(payload))

            replay = replay_artifact(
                artifact_key="k.json",
                target_model="deepseek/deepseek-v4-flash",
                s3_client=s3, client_factory=factory, api_key="sk-or-test",
                persist=False,
            )

            assert replay.replay_output_kind == "structured", (
                f"agent_id={agent_id} expected schema={expected_schema} "
                f"failed to validate: {replay.replay_error}"
            )

    def test_invoke_called_with_system_and_user_messages(self):
        from nousergon_lib.agent_schemas import QuantAnalystOutput
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact(user_prompt="Pick 5 tech names.")
        s3 = _make_s3_stub(artifact)
        factory, fake_client = _make_krepis_factory(
            content=json.dumps(QuantAnalystOutput(ranked_picks=[]).model_dump()),
        )

        replay_artifact(
            artifact_key="k.json",
            target_model="deepseek/deepseek-v4-flash",
            s3_client=s3, client_factory=factory, api_key="sk-or-test",
            persist=False,
        )

        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "You are a sector analyst."}
        # structured_outputs=False appends a JSON-schema instruction suffix
        # to the user turn (krepis's tolerant-extraction fallback) — the
        # captured user_prompt is unmodified as the PREFIX.
        assert messages[1]["content"].startswith("Pick 5 tech names.")
        assert messages[1]["role"] == "user"


# ── Schema-validation failure ────────────────────────────────────────────


class TestSchemaValidationError:
    def test_validation_failure_captured_on_artifact(self):
        """When the target model emits a structurally divergent output,
        krepis.llm.LLMClient.structured() raises LLMError after the
        (single, attempts=1 — see runner.py docstring) attempt fails
        schema validation. Replay surfaces that as replay_error rather
        than silently emitting a 0-agreement comparison — exactly the
        silent-drift signal we wanted to capture."""
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact()
        s3 = _make_s3_stub(artifact)
        # quant_score must be a number <= 100 (QuantAnalystOutput's
        # QuantPick) — this string value fails schema validation.
        bad_payload = json.dumps({
            "ranked_picks": [
                {"ticker": "AAPL", "quant_score": "not-a-number"},
            ],
        })
        factory, _ = _make_krepis_factory(content=bad_payload)

        replay = replay_artifact(
            artifact_key="k.json",
            target_model="deepseek/deepseek-v4-flash",
            s3_client=s3, client_factory=factory, api_key="sk-or-test",
            persist=False,
        )

        assert replay.replay_output_kind == "error"
        assert "validation failed" in (replay.replay_error or "").lower()
        assert "quant_score" in (replay.replay_error or "")

    def test_malformed_json_also_captured_as_validation_failure(self):
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact()
        s3 = _make_s3_stub(artifact)
        factory, _ = _make_krepis_factory(content="not json at all {{{")

        replay = replay_artifact(
            artifact_key="k.json",
            target_model="deepseek/deepseek-v4-flash",
            s3_client=s3, client_factory=factory, api_key="sk-or-test",
            persist=False,
        )

        assert replay.replay_output_kind == "error"
        assert replay.replay_error


# ── Generic transport error path ─────────────────────────────────────────


class TestErrorHandling:
    def test_transport_exception_captured_not_raised(self):
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact()
        s3 = _make_s3_stub(artifact)
        factory, _ = _make_krepis_factory(
            raise_on_create=RuntimeError("OpenRouter 500"),
        )

        replay = replay_artifact(
            artifact_key="k.json",
            target_model="deepseek/deepseek-v4-flash",
            s3_client=s3, client_factory=factory, api_key="sk-or-test",
        )

        assert replay.replay_output_kind == "error"
        assert "OpenRouter 500" in (replay.replay_error or "")
        assert s3.put_object.called

    def test_missing_api_key_captured_not_raised(self):
        """No api_key arg + no resolvable OPENROUTER_API_KEY (env-isolated
        by conftest's autouse secrets fixture) → captured as replay_error,
        never propagated. Replay is offline analysis; one bad config
        should never abort a batch."""
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact()
        s3 = _make_s3_stub(artifact)

        replay = replay_artifact(
            artifact_key="k.json",
            target_model="deepseek/deepseek-v4-flash",
            s3_client=s3, api_key=None,
            persist=False,
        )

        assert replay.replay_output_kind == "error"
        assert "OpenRouter API key" in (replay.replay_error or "")


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
        factory, _ = _make_krepis_factory()

        replay = replay_artifact(
            artifact_key="k.json",
            target_model="deepseek/deepseek-v4-flash",
            s3_client=s3, client_factory=factory, api_key="sk-or-test",
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
        factory, _ = _make_krepis_factory()

        replay = replay_artifact(
            artifact_key="k.json",
            target_model="deepseek/deepseek-v4-flash",
            s3_client=s3, client_factory=factory, api_key="sk-or-test",
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
        factory, _ = _make_krepis_factory()

        # Must not raise.
        replay = replay_artifact(
            artifact_key="k.json",
            target_model="deepseek/deepseek-v4-flash",
            s3_client=s3, client_factory=factory, api_key="sk-or-test",
            persist=False,
        )
        assert replay is not None


# ── S3 persistence ───────────────────────────────────────────────────────


class TestPersistence:
    def test_persists_to_canonical_key(self):
        from nousergon_lib.agent_schemas import QuantAnalystOutput
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact(
            run_id="run-xyz", model_name="claude-sonnet-4-6",
        )
        s3 = _make_s3_stub(artifact)
        factory, _ = _make_krepis_factory(
            content=json.dumps(QuantAnalystOutput(ranked_picks=[]).model_dump()),
        )

        replay_artifact(
            artifact_key="src.json", target_model="deepseek/deepseek-v4-flash",
            s3_client=s3, client_factory=factory, api_key="sk-or-test",
        )

        # Canonical eval_artifacts layout: a flat dated key
        # {run_id}_{orig}_vs_{target}.json + a latest.json sidecar. The
        # run_id is a fresh YYMMDDHHMM mint (NOT the original_run_id), so
        # the legacy nested {original_run_id}/ partition is gone.
        put_keys = [c.kwargs["Key"] for c in s3.put_object.call_args_list]
        assert all(
            c.kwargs["Bucket"] == "alpha-engine-research"
            for c in s3.put_object.call_args_list
        )
        dated = [k for k in put_keys if not k.endswith("/latest.json")]
        latest = [k for k in put_keys if k.endswith("/latest.json")]
        assert len(dated) == 1
        key = dated[0]
        # Flat — exactly one path segment after the prefix, no run-xyz dir.
        assert key.startswith("decision_artifacts/_replay/")
        assert "run-xyz/" not in key
        # target_model's "/" is sanitized to "-" the same as ":" — an
        # OpenRouter-shaped id (deepseek/deepseek-v4-flash) must not
        # fracture the S3 key into extra path segments.
        assert key.endswith("_claude-sonnet-4-6_vs_deepseek-deepseek-v4-flash.json")
        basename = key.rsplit("/", 1)[-1]
        assert basename.count("/") == 0
        run_id = basename.split("_", 1)[0]
        assert len(run_id) == 10 and run_id.isdigit()  # YYMMDDHHMM
        assert latest == ["decision_artifacts/_replay/latest.json"]

    def test_latest_sidecar_mirrors_dated_artifact(self):
        """The latest.json sidecar must be a byte-for-byte mirror of the
        dated forensic artifact, and the payload must carry the minted
        replay_run_id so the dated key is self-describing (config#792)."""
        from nousergon_lib.agent_schemas import QuantAnalystOutput
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact(
            run_id="run-xyz", model_name="claude-sonnet-4-6",
        )
        s3 = _make_s3_stub(artifact)
        factory, _ = _make_krepis_factory(
            content=json.dumps(QuantAnalystOutput(ranked_picks=[]).model_dump()),
        )

        replay_artifact(
            artifact_key="src.json", target_model="deepseek/deepseek-v4-flash",
            s3_client=s3, client_factory=factory, api_key="sk-or-test",
        )

        bodies_by_key = {
            c.kwargs["Key"]: c.kwargs["Body"]
            for c in s3.put_object.call_args_list
        }
        dated_key = next(k for k in bodies_by_key if not k.endswith("/latest.json"))
        latest_key = "decision_artifacts/_replay/latest.json"
        # Sidecar is a pure mirror of the dated artifact.
        assert bodies_by_key[dated_key] == bodies_by_key[latest_key]
        # Payload carries the minted run_id matching the dated basename.
        payload = json.loads(bodies_by_key[dated_key])
        run_id = dated_key.rsplit("/", 1)[-1].split("_", 1)[0]
        assert payload["replay_run_id"] == run_id

    def test_no_persist_skips_put_object(self):
        from nousergon_lib.agent_schemas import QuantAnalystOutput
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact()
        s3 = _make_s3_stub(artifact)
        factory, _ = _make_krepis_factory(
            content=json.dumps(QuantAnalystOutput(ranked_picks=[
                {"ticker": "X", "quant_score": 80, "rationale": "ok"},
            ]).model_dump()),
        )

        replay = replay_artifact(
            artifact_key="k.json", target_model="deepseek/deepseek-v4-flash",
            s3_client=s3, client_factory=factory, api_key="sk-or-test",
            persist=False,
        )

        s3.put_object.assert_not_called()
        assert replay.replay_output_kind == "structured"
        assert len(replay.replay_output["ranked_picks"]) == 1

    def test_model_name_with_colon_sanitized_in_key(self):
        from nousergon_lib.agent_schemas import QuantAnalystOutput
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact(model_name="claude-sonnet-4-6:live")
        s3 = _make_s3_stub(artifact)
        factory, _ = _make_krepis_factory(
            content=json.dumps(QuantAnalystOutput(ranked_picks=[]).model_dump()),
        )

        replay_artifact(
            artifact_key="k.json", target_model="deepseek/deepseek-v4-flash",
            s3_client=s3, client_factory=factory, api_key="sk-or-test",
        )

        dated = [
            c.kwargs["Key"] for c in s3.put_object.call_args_list
            if not c.kwargs["Key"].endswith("/latest.json")
        ]
        assert len(dated) == 1
        assert ":" not in dated[0].rsplit("/", 1)[-1]


# ── Usage extraction ─────────────────────────────────────────────────────


class TestUsageExtraction:
    def test_token_counts_carry_through(self):
        from nousergon_lib.agent_schemas import QuantAnalystOutput
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact()
        s3 = _make_s3_stub(artifact)
        factory, _ = _make_krepis_factory(
            content=json.dumps(QuantAnalystOutput(ranked_picks=[]).model_dump()),
            prompt_tokens=1234, completion_tokens=567, cost=0.00042,
        )

        replay = replay_artifact(
            artifact_key="k.json", target_model="deepseek/deepseek-v4-flash",
            s3_client=s3, client_factory=factory, api_key="sk-or-test",
        )

        assert replay.replay_cost["input_tokens"] == 1234
        assert replay.replay_cost["output_tokens"] == 567
        # OpenRouter's actually-billed cost — a capability the pre-migration
        # Anthropic SDK path never surfaced (purely additive; see runner.py).
        assert replay.replay_cost["provider_cost_usd"] == 0.00042

    def test_served_provider_carries_through(self):
        # config#3006 — jurisdiction/compliance check reads this off the
        # persisted artifact instead of re-deriving it from raw_response.
        from nousergon_lib.agent_schemas import QuantAnalystOutput
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact()
        s3 = _make_s3_stub(artifact)
        factory, _ = _make_krepis_factory(
            content=json.dumps(QuantAnalystOutput(ranked_picks=[]).model_dump()),
            served_provider="DeepInfra",
        )

        replay = replay_artifact(
            artifact_key="k.json", target_model="deepseek/deepseek-v4-flash",
            s3_client=s3, client_factory=factory, api_key="sk-or-test",
        )

        assert replay.replay_cost["served_provider"] == "DeepInfra"

    def test_served_provider_absent_when_not_reported(self):
        # Also covers the pre-v0.18.0 krepis pin case — the runner reads
        # this via getattr(result, "served_provider", None), so an older
        # LLMResult without the attribute degrades to None, not a crash.
        from nousergon_lib.agent_schemas import QuantAnalystOutput
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact()
        s3 = _make_s3_stub(artifact)
        factory, _ = _make_krepis_factory(
            content=json.dumps(QuantAnalystOutput(ranked_picks=[]).model_dump()),
        )

        replay = replay_artifact(
            artifact_key="k.json", target_model="deepseek/deepseek-v4-flash",
            s3_client=s3, client_factory=factory, api_key="sk-or-test",
        )

        assert replay.replay_cost.get("served_provider") is None

    def test_missing_usage_returns_zeroed_dict(self):
        from nousergon_lib.agent_schemas import QuantAnalystOutput
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact()
        s3 = _make_s3_stub(artifact)
        factory, _ = _make_krepis_factory(
            content=json.dumps(QuantAnalystOutput(ranked_picks=[]).model_dump()),
        )

        replay = replay_artifact(
            artifact_key="k.json", target_model="deepseek/deepseek-v4-flash",
            s3_client=s3, client_factory=factory, api_key="sk-or-test",
        )

        assert replay.replay_cost["input_tokens"] == 0
        assert replay.replay_cost["output_tokens"] == 0
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


# ── Placeholder-prompt skip (capture wiring gap, config#1035) ────────────


class TestPlaceholderPromptSkip:
    """Captures from agents not yet wired through track_llm_cost carry
    placeholder strings in full_prompt_context (research_graph.py
    fallback path). Replaying them burns spend on junk output — the
    2026-06-12 Friday shell run produced 31 '<UNKNOWN' int_parsing
    failures + flat-0.0 concordance this way. Pin the pre-LLM skip.
    """

    def _placeholder_artifact(self) -> dict:
        artifact = _make_captured_artifact(agent_id="thesis_update:consumer:GOOG")
        artifact["full_prompt_context"] = {
            "system_prompt": (
                "<see config/prompts/sector_team*.txt at run time; "
                "call site not yet wired through track_llm_cost>"
            ),
            "user_prompt": (
                "<rendered from input_data_snapshot at run time; "
                "call site not yet wired through track_llm_cost>"
            ),
            "tool_definitions": [],
        }
        return artifact

    def test_placeholder_prompt_artifact_skipped_before_llm_call(self):
        from replay.runner import replay_artifact

        artifact = self._placeholder_artifact()
        s3 = _make_s3_stub(artifact)
        factory, _ = _make_krepis_factory()

        replay = replay_artifact(
            artifact_key="k.json",
            target_model="deepseek/deepseek-v4-flash",
            s3_client=s3, client_factory=factory, api_key="sk-or-test",
            persist=False,
        )

        assert replay.replay_output_kind == "skipped"
        assert "placeholder prompt context" in (replay.replay_error or "")
        assert replay.comparison["agent_id_base"] == "thesis_update"
        # The load-bearing assertion: NO LLM call (no spend) attempted.
        factory.assert_not_called()

    def test_empty_prompts_also_skipped(self):
        from replay.runner import replay_artifact

        artifact = _make_captured_artifact()
        artifact["full_prompt_context"] = {
            "system_prompt": "", "user_prompt": "", "tool_definitions": [],
        }
        s3 = _make_s3_stub(artifact)
        factory, _ = _make_krepis_factory()

        replay = replay_artifact(
            artifact_key="k.json",
            target_model="deepseek/deepseek-v4-flash",
            s3_client=s3, client_factory=factory, api_key="sk-or-test",
            persist=False,
        )

        assert replay.replay_output_kind == "skipped"
        factory.assert_not_called()
