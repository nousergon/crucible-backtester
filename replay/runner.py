"""Single-artifact replay — re-run a captured DecisionArtifact under a
different model and persist a side-by-side comparison.

Pipeline:

  1. Load ``DecisionArtifact`` JSON from S3 by key.
  2. Resolve the canonical Pydantic schema for the ``agent_id`` via
     ``nousergon_lib.agent_schemas.resolve_schema_for_agent``.
     Skips replay for unknown agent families (no schema to validate
     against → no meaningful concordance signal).
  3. Invoke the target model via ``krepis.llm.LLMClient.structured()``
     against the krepis router edge (``resolve_target_spec`` below).
     Extracts the parsed
     Pydantic instance's ``model_dump()`` for the comparison +
     persistence layers. Schema-validation failures surface as
     ``replay_error`` on the artifact — they're the silent-drift signal
     we wanted to expose (target model emits a structurally divergent
     output that the canonical schema rejects).
  4. Persist side-by-side artifact under the canonical eval_artifacts
     layout: ``decision_artifacts/_replay/{run_id}_{original_model}_vs_{target_model}.json``
     (flat, YYMMDDHHMM run_id) + a ``decision_artifacts/_replay/latest.json``
     sidecar. Key format owned by ``nousergon_lib.eval_artifacts``.

**alpha-engine-config-I7878 (2026-08-20): migrated off DIRECT OpenRouter
onto the krepis router edge**, closing the last gap left by Brian's
2026-08-03 ruling that no agent may be directly linked to OpenRouter
(alpha-engine-config-I6367). This module and
``lambda_concordance/handler.py`` were the only live call sites in the
repo that still resolved a provider API key from SSM and constructed a
provider-pinned ``ModelSpec`` themselves; they were invisible
to the fleet guard until ``crucible-backtester`` was first scanned
(alpha-engine-config-I7864).

``target_model`` is now a **registry entry id** from
``alpha-engine-config/private-docs/LLM_MODEL_REGISTRY.yaml`` — e.g.
``"deepseek-v4-flash"`` — not an OpenRouter slug. That is what keeps a
PINNED model inside the router policy: a registry id is a layer-1 fact,
and provider, upstream model string, endpoint, credential and params are
all resolved above this module (``krepis.router.resolve_model_spec``,
krepis-PR172). The registry already models exactly this case — entries
kept ``active`` and out of every group are annotated "callable BY NAME
only" — because a concordance harness must measure a NAMED model: a
capability group would let the fallback chain silently vary the thing
being measured between runs.

**What this did to cross-run comparability — read before comparing a
score across 2026-08-20.** The pre-migration call addressed OpenRouter's
bare ``deepseek/deepseek-v4-flash`` slug; the registry's own note on
``deepseek-v4-flash-openrouter-max`` records that OpenRouter keeps that
bare slug on the ORIGINAL snapshot while the DeepSeek first-party API
serves the
2026-07-31 re-post-trained revision under the same bare name. Registry id
``deepseek-v4-flash`` routes to that first-party API. So the target model
genuinely changes weights, and no registry entry reproduces the old call
exactly (``deepseek-v4-flash-openrouter-max`` pins the 0731 slug but at
``reasoning: {effort: max}``). ``agent_cheap_model_concordance`` is
therefore a LEVEL SHIFT at this migration, not a continuous series.
It is not a silent one: the CloudWatch ``target_model`` dimension value
changes from ``deepseek/deepseek-v4-flash`` to ``deepseek-v4-flash``, so
the break appears as a new series rather than a step inside the old one.
Re-baseline rather than compare across it (alpha-engine-config-I7878).

**alpha-engine-config-I2997 (2026-07-19): migrated off direct Anthropic
(``langchain_anthropic.ChatAnthropic``) to the fleet-SOTA
``krepis.llm.LLMClient`` transport.** This dropped the "invocation isomorphism
with production agents" rationale the prior ``with_structured_output``
choice was built on (production agents still call Anthropic directly for
now; only THIS cheap-model-concordance measurement arm moved), but keeps
what actually matters for this module's purpose:

  - **Pydantic validation against the captured contract.** Catches the
    silent-drift class where a target model emits a slightly different
    structure that would otherwise wash through the comparison stage
    as an unexplained low concordance score. ``krepis.llm.LLMClient.
    structured()`` validates the SAME way (``schema.model_validate``),
    just over a different transport.
  - **Schema portability.** Schemas live in ``nousergon_lib.agent_schemas``
    (lifted 2026-05-05, lib v0.4.0) so backtester can validate against
    the canonical contract without a heavy cross-repo dep on research.

``ModelSpec.structured_outputs=False`` is REQUIRED, not incidental:
live-verified 2026-07-19 against ``nousergon_lib.agent_schemas.
QuantAnalystOutput`` (one of the six canonical schemas this module
resolves) — the JSON-instruction + tolerant-extraction fallback
(``structured_outputs=False``) round-tripped this schema correctly on
every live attempt via ``deepseek/deepseek-v4-flash``. Strict
``response_format=json_schema`` mode (``structured_outputs=True``) is
NOT used because the sibling alpha-engine-config-I2997 migration
(crucible-research's ``producers/single_agent.py``, same live-testing
session) found it UNRELIABLE for DeepSeek-family models on OpenRouter —
against a structurally similar schema, strict mode intermittently
renamed/dropped a REQUIRED field (e.g. the equivalent of ``ticker`` came
back as ``symbol``/``candidate``), failing schema validation on every
attempt, while the same prompt round-tripped correctly every time under
``structured_outputs=False``. Since this module measures exactly this
class of divergence (concordance/silent-drift), routing the measurement
itself through a transport mode with its OWN independent failure mode
would confound the signal — ``structured_outputs=False`` is the
verified-reliable choice, consistent across every DeepSeek+OpenRouter
call site this migration touched.
``attempts=1`` (no corrective retry) is DELIBERATE, not a missed
optimization: this module's whole purpose is measuring how often the
target model's raw output diverges from the canonical schema — a
corrective retry would suppress exactly the signal
(``agent_cheap_model_concordance``) it exists to produce.
``reasoning`` is no longer set here. It used to be hand-set to
``{"exclude": True}`` so a reasoning-capable model would not burn its
whole output budget on chain-of-thought and return empty content
(config#1659 / config#2575). That is now a REGISTRY fact —
``params.reasoning`` on the entry, which for ``deepseek-v4-flash`` is
exactly ``{exclude: true}``, so the served behaviour is unchanged — and a
second copy of it at the call site is the layer-5 duplication
model-router-policy §2 forbids.

The captured ``input_data_snapshot`` is intentionally NOT re-presented
to the model: the original ``user_prompt`` already contained the
relevant slice of the snapshot inlined (research's typed-state arc
canonicalized this — every agent's ``user_prompt`` is the load-bearing
input surface). Replay-time RAG re-execution would require the original
RAG corpus + ArcticDB + tools to be available, which is out of scope for
v1 (single-shot replay). When deeper replay is needed (full ReAct loop
with tool re-execution), wrap this module rather than fork it.

Cost attribution: every replay invocation records token counts +
derived cost in the persisted artifact's ``replay_cost`` block. The
existing cost telemetry pipeline (closed 2026-05-01) does NOT
auto-ingest replay calls — replay is offline analysis, not a
production run. To roll up replay spend, post-process the
``decision_artifacts/_replay/`` prefix.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import boto3

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────


DEFAULT_BUCKET = "alpha-engine-research"
DEFAULT_REPLAY_PREFIX = "decision_artifacts/_replay"
DEFAULT_MAX_TOKENS = 8192

# Research's decision-capture fallback path (research_graph.py
# _capture_agent_decision) stamps this marker into full_prompt_context
# when an agent's call site is not yet wired through track_llm_cost —
# the capture carries placeholder strings instead of the real prompts.
# Replaying a placeholder prompt is pure waste: the target model gets
# no actual content, so it emits junk (e.g. literal "<UNKNOWN>" into
# int fields) and the comparison stage scores a meaningless ~0.0
# concordance — while still paying full Anthropic spend. Found via the
# 2026-06-12 Friday shell run: 31/150 replay failures + flat-0.0
# concordance for every unwired agent family (config#1035).
PLACEHOLDER_PROMPT_MARKER = "not yet wired through track_llm_cost"


def _prompts_are_placeholder(system_prompt: str, user_prompt: str) -> bool:
    """True when the captured prompts can't drive a meaningful replay."""
    if not system_prompt.strip() and not user_prompt.strip():
        return True
    return (
        PLACEHOLDER_PROMPT_MARKER in system_prompt
        or PLACEHOLDER_PROMPT_MARKER in user_prompt
    )
"""Generous upper bound — the original agent's max_tokens is preserved
when present; this is the fallback for artifacts without an explicit
budget. 8192 covers all current rubric outputs (which are <2KB) plus
ample headroom for sector_quant ranked_picks (10 entries × ~500 chars =
~5KB)."""


# ── Replay output schema ─────────────────────────────────────────────────


@dataclass
class ReplayOutput:
    """Side-by-side replay artifact. Persisted to S3 as JSON.

    Schema is intentionally additive — comparison metrics (PR B) will
    extend ``comparison`` without touching this dataclass; per-agent
    agreement scorers attach their output as a sub-dict.
    """

    schema_version: int = 1
    original_run_id: str = ""
    original_agent_id: str = ""
    original_model: str = ""
    original_artifact_key: str = ""
    original_output: dict[str, Any] = field(default_factory=dict)

    replay_model: str = ""
    replay_timestamp: str = ""
    replay_output: dict[str, Any] = field(default_factory=dict)
    replay_output_kind: str = "structured"  # "structured" | "error"
    replay_cost: dict[str, Any] = field(default_factory=dict)
    replay_latency_ms: int = 0
    replay_error: str | None = None

    # Reserved for PR B's per-agent comparison scorers.
    comparison: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "original_run_id": self.original_run_id,
            "original_agent_id": self.original_agent_id,
            "original_model": self.original_model,
            "original_artifact_key": self.original_artifact_key,
            "original_output": self.original_output,
            "replay_model": self.replay_model,
            "replay_timestamp": self.replay_timestamp,
            "replay_output": self.replay_output,
            "replay_output_kind": self.replay_output_kind,
            "replay_cost": self.replay_cost,
            "replay_latency_ms": self.replay_latency_ms,
            "replay_error": self.replay_error,
            "comparison": self.comparison,
        }


# ── S3 IO ────────────────────────────────────────────────────────────────


def _load_artifact(s3: Any, *, bucket: str, key: str) -> dict[str, Any]:
    """Load + JSON-parse a captured DecisionArtifact. Returns the raw
    dict — we tolerate additive schema drift on the captured side."""
    raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(raw)


def _persist_replay(
    s3: Any,
    *,
    bucket: str,
    replay_prefix: str,
    replay: ReplayOutput,
) -> str:
    """Write the replay artifact under the canonical ``eval_artifacts``
    layout: a flat, structured-timestamp dated key
    ``{replay_prefix}/{run_id}_{original_model}_vs_{target_model}.json``
    plus a ``{replay_prefix}/latest.json`` operator-UX sidecar.

    Migrated from the legacy nested ``{replay_prefix}/{original_run_id}/
    {orig}_vs_{target}.json`` layout (backtester #179 deferred this site;
    config#792). The key format is owned by ``nousergon_lib.
    eval_artifacts`` — we mint a fresh ``run_id`` per replay invocation
    (``new_eval_run_id`` → ``YYMMDDHHMM``) and stamp it into the payload
    as ``replay_run_id`` so the dated artifact is self-describing, while
    the ``{orig}_vs_{target}`` discriminator survives as the canonical
    multi-file basename. Sanitize model names so colons or slashes don't
    break the S3 key.

    The dated key is the forensic source of truth (re-runs are preserved
    under distinct YYMMDDHHMM run_ids); the ``latest.json`` sidecar is a
    pure mirror of the most-recently-written replay for operator UX.
    """
    from nousergon_lib.eval_artifacts import (
        eval_artifact_key,
        eval_latest_key,
        new_eval_run_id,
    )

    safe_orig = replay.original_model.replace(":", "-").replace("/", "-")
    safe_target = replay.replay_model.replace(":", "-").replace("/", "-")
    run_id = new_eval_run_id()
    basename = f"{safe_orig}_vs_{safe_target}.json"
    key = eval_artifact_key(replay_prefix, run_id, basename=basename)

    payload = replay.to_dict()
    payload["replay_run_id"] = run_id
    body = json.dumps(payload, indent=2, default=str).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")

    # Operator-UX latest sidecar — pure mirror of the dated artifact.
    s3.put_object(
        Bucket=bucket,
        Key=eval_latest_key(replay_prefix),
        Body=body,
        ContentType="application/json",
    )
    return key


# ── Target-model invocation (krepis.llm / OpenRouter) ─────────────────────


def _build_messages(system_prompt: str, user_prompt: str) -> list[dict]:
    """Single-user-message replay: system + user, no chat history. The
    captured user_prompt already includes the relevant input snapshot
    inlined into the prompt body, so we don't need to re-present
    input_data_snapshot."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


#: Where this code runs — a FACT about the process, never a routing
#: preference (model-router-policy R28/R29). Concordance runs in the
#: ReplayConcordance Lambda by default (measured 2026-08-20: that function
#: sets ``KREPIS_EXEC_CONTEXT=lambda``, ``KREPIS_LITELLM_PROXY_URL`` and its
#: own per-consumer credential ``KREPIS_ROUTER_CREDENTIAL_SECRET=
#: ROUTER_CONSUMER_REPLAY``); a spot or laptop batch run exports the env var
#: for its own context. Mirrors ``crucible-evaluator/director/agent.py``.
#:
#: The registry declares ``lambda`` on NO model entry, deliberately: a Lambda
#: has no local egress proxy and no private-network peer, so the router edge
#: is its only path. That is what makes this call site FAIL CLOSED rather than
#: reaching for a direct provider endpoint whose traffic is DLP-unscanned. Do
#: not "fix" a router outage by widening what this context admits.
REPLAY_EXEC_CONTEXT = os.environ.get("KREPIS_EXEC_CONTEXT", "lambda")

#: The ``llm_trace.v1`` capture level the replay call sites DECLARE
#: (``replay-concordance`` and ``replay-batch-concordance``, which delegates
#: here). Brian's ruling C on alpha-engine-config-I12312 (2026-10-18):
#: ``hash_only`` — hashes, offsets, usage, route and timing, no prompt or
#: output bodies — because a replayed prompt carries news and filings
#: excerpts, and whether those vendors' terms allow storing derived excerpts
#: is unverified. The ruling promotes replay to ``full`` once a one-line
#: vendor-terms check confirms they may be stored. It must match the
#: ``trace_capture`` field of both LLM_CALLSITE_REGISTRY rows.
#:
#: Passed explicitly rather than left to krepis's default, which traces at
#: ``full`` as soon as a ``KREPIS_TRACE_SINK_*`` env reaches this process.
REPLAY_TRACE_CAPTURE = "hash_only"

#: The env pair krepis's process-default trace sink reads
#: (``krepis.trace_sink.BUCKET_ENV_VAR`` / ``PREFIX_ENV_VAR``).
TRACE_SINK_ENV_VARS = ("KREPIS_TRACE_SINK_BUCKET", "KREPIS_TRACE_SINK_PREFIX")


def replay_trace_capture() -> str:
    """The ``trace_capture=`` a replay ``LLMClient`` is constructed with.

    :data:`REPLAY_TRACE_CAPTURE` when a trace sink is configured on the
    process, ``off`` when none is: krepis refuses, at construction, a
    declared level that writes nowhere, and the replay Lambda has no trace
    env today. A HALF-configured env is not masked: krepis's own env check
    raises on it, and the invocation's existing error path records it.
    """
    if any(os.environ.get(name, "").strip() for name in TRACE_SINK_ENV_VARS):
        return REPLAY_TRACE_CAPTURE
    return "off"


def resolve_target_spec(
    target_model: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple:
    """Resolve *target_model* to a ``(ModelSpec, route)`` through the router.

    *target_model* is a **registry entry id** from
    ``alpha-engine-config/private-docs/LLM_MODEL_REGISTRY.yaml`` (e.g.
    ``"deepseek-v4-flash"``), NOT a provider slug — ``"deepseek/deepseek-v4-
    flash"`` is an OpenRouter slug and is refused with the addressable ids
    named. Provider, upstream model string, endpoint, credential and params
    are all registry decisions resolved above this module.

    ``structured_outputs=False`` is passed explicitly and is REQUIRED — see
    the module docstring. It is a statement about how this harness INVOKES,
    not about which model serves: strict ``response_format=json_schema`` is
    unreliable for the DeepSeek family, and this module measures exactly the
    divergence class a flaky strict mode would counterfeit. Several registry
    entries (``claude-haiku-4-5``) declare ``structured_outputs: true``, so
    the override has to be at the call site to hold across every target.

    ``reasoning`` is NOT passed. It used to be hand-set to
    ``{"exclude": True}`` here; it is a registry fact now
    (``params.reasoning`` on the entry) and a second copy at layer 5 is the
    duplication the router policy forbids.

    Raises loudly (never returns a degraded direct-provider spec) — see
    ``krepis.router.resolve_model_spec``: a pinned model has no registry-
    declared substitute, so the only available fallback would be the direct
    OpenRouter linkage Brian's 2026-08-03 ruling removed
    (alpha-engine-config-I6367 / I7878).
    """
    from krepis.router import resolve_model_spec

    return resolve_model_spec(
        target_model,
        exec_context=REPLAY_EXEC_CONTEXT,
        wire="openai",
        max_tokens=max_tokens,
        structured_outputs=False,
    )


def _invoke_target_with_schema(
    *,
    target_model: str,
    schema: type,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    client_factory: Any | None = None,
    model_spec: Any | None = None,
) -> tuple[Any, dict[str, Any], int, str | None]:
    """Invoke the target model through the krepis router edge.

    *model_spec* is the ``ModelSpec`` from :func:`resolve_target_spec`.
    Batch runs resolve ONCE per target model and pass it in — resolution is
    a precondition of the whole sweep, not a per-artifact outcome, and
    probing the edge 150 times to learn the same answer is waste. When it is
    ``None`` (the single-artifact CLI path) this resolves for itself.

    Returns ``(parsed_output_dict, usage_dict, latency_ms, error_or_none)``
    — same shape as the pre-migration bare-SDK helper. On exception:
    ``(None, usage_dict, latency, str(exc))`` — caller persists the error
    onto the replay artifact rather than raising. Replay is offline
    analysis; one failed re-invocation should never abort a batch.
    ``usage_dict`` is populated even on a validation-exhaustion failure
    when the underlying ``LLMError`` carries partial usage (tokens were
    still spent on the failed attempt).

    ``client_factory`` is the krepis.llm.LLMClient test seam (mirrors the
    Think Tank pattern, matches every other migrated call site): a callable
    ``(spec, api_key) -> transport_client``. Production leaves it unset —
    ``LLMClient`` lazily builds an ``openai.OpenAI`` client pointed at the
    router edge the resolved spec names.

    There is no ``api_key`` parameter any more. The credential is whatever
    the resolved route declares (this consumer's own router-edge credential,
    ``ROUTER_CONSUMER_REPLAY``), resolved by krepis — a provider key passed
    in here would be the direct linkage this module was migrated off
    (alpha-engine-config-I6367 / I7878).

    Schema-validation failure — the silent-drift signal this module
    exists to surface — and ordinary transport/SDK errors both funnel
    through ``krepis.llm.LLMError``/``LLMConfigError``/generic
    ``Exception`` here; the error string is not pattern-matched by any
    downstream consumer (verified: ``replay.batch``/``replay.cli`` only
    truncate + log it), so this function does not need to reproduce the
    exact pre-migration wording.
    """
    from krepis.llm import LLMClient, LLMError

    # Resolution happens OUTSIDE the try. A router that will not admit this
    # process is a precondition failure, not a per-artifact replay error: it
    # is identical for every artifact, it is not the divergence this module
    # measures, and recording it as one would report a router outage as 150
    # low-concordance observations. It raises, and the batch layer stops the
    # whole target rather than burning the corpus against a dead edge.
    spec = model_spec if model_spec is not None else resolve_target_spec(
        target_model, max_tokens=max_tokens
    )[0]

    start = time.monotonic()
    try:
        # krepis >=0.23.0 requires callsite_id as a keyword-only argument
        # (added as a breaking change — the param has no default). This id
        # is the row this call site already owns in alpha-engine-config's
        # LLM_CALLSITE_REGISTRY.yaml (`replay-concordance`), so spend lands
        # under the callsite the registry says produces it — inventing a
        # new string here would attribute it to nothing.
        client = LLMClient(
            spec,
            callsite_id="replay-concordance",
            # Declared `hash_only` (ruling C, alpha-engine-config-I12312), or
            # `off` when no trace sink is configured. See replay_trace_capture.
            trace_capture=replay_trace_capture(),
            client_factory=client_factory,
        )
        result = client.structured(
            system=system_prompt,
            user_content=user_prompt,
            schema=schema,
            schema_name=schema.__name__,
            # Deliberately no corrective retry — see module docstring:
            # this module MEASURES divergence, a retry would suppress it.
            attempts=1,
            max_tokens=max_tokens,
        )
    except LLMError as exc:
        latency_ms = int((time.monotonic() - start) * 1000)
        usage_dict = _usage_dict_from_llm_usage(exc.usage)
        return None, usage_dict, latency_ms, (
            f"structured output validation failed against the canonical "
            f"schema: {exc}"
        )
    except Exception as exc:  # noqa: BLE001 — covers LLMConfigError + transport
        # errors (network, edge 4xx/5xx) raised by the INVOCATION. Resolution
        # errors cannot land here: they are raised above the try, by design.
        # (a) swallowed: one artifact's invocation failing; (b) the sweep's
        # aggregate survives because the failure is counted, not averaged in;
        # (c) recorded on the artifact's replay_error and in the batch
        # summary's replay_failures, which turns the Lambda status PARTIAL.
        latency_ms = int((time.monotonic() - start) * 1000)
        return None, {}, latency_ms, str(exc)

    latency_ms = int((time.monotonic() - start) * 1000)
    # getattr, not attribute access — degrades to None gracefully against
    # a pre-v0.18.0 krepis pin (served_provider is new, config#3006) rather
    # than raising AttributeError.
    usage_dict = _usage_dict_from_llm_usage(
        result.usage, served_provider=getattr(result, "served_provider", None)
    )

    if result.data is None:
        return None, usage_dict, latency_ms, (
            "krepis.llm.LLMClient.structured() returned no parsed object"
        )

    return dict(result.data), usage_dict, latency_ms, None


def _usage_dict_from_llm_usage(
    usage: Any, *, served_provider: str | None = None
) -> dict[str, Any]:
    """Normalize a ``krepis.llm.LLMUsage`` into the persisted
    ``replay_cost`` dict shape. Keeps the two keys ``replay.batch``
    actually reads (``input_tokens``/``output_tokens``) plus the
    cache-token fields for parity with the pre-migration shape, and adds
    ``provider_cost_usd`` — OpenRouter's actually-billed USD cost when the
    request opts in (``usage.include: true``, set automatically by
    ``krepis.llm`` for the openrouter provider), a capability the prior
    Anthropic-SDK path never surfaced. Purely additive — no consumer reads
    a fixed key set (verified: ``replay.batch`` uses ``.get(k, 0)``).

    ``served_provider`` (config#3006) — the upstream backend OpenRouter
    actually routed to (e.g. "DeepInfra"), read off
    ``LLMResult.served_provider`` at the call site. ``None`` on the
    exhausted-retry error path (``LLMError`` carries no result object to
    read it from) — that's an accepted gap, not a bug: a failed call's
    provider identity isn't load-bearing for the jurisdiction check.

    **The DICT SHAPE is unchanged by the I7878 router migration** — same six
    keys, same types, same values for the same usage object, verified by
    running this function on both sides of the change. What DOES change is how
    often ``served_provider`` is populated: it is ``resp.provider``, a
    non-standard top-level field OpenRouter emits and nothing else does. The
    pre-migration call went to OpenRouter, so it arrived; a routed call to
    ``deepseek-v4-flash`` reaches the DeepSeek first-party API server-side of
    the edge, which does not emit it. Expect ``served_provider: null`` and an
    EMPTY ``served_providers_seen`` in the batch summary for that target —
    already handled as informational absence by ``replay.batch``, and covered
    by ``test_served_providers_seen_empty_when_none_reported``.

    That is a real loss of one observability field, and it is the right trade:
    config#3006 wanted this for a jurisdiction/compliance check, and the
    routed path answers that question at a better place — the registry's
    ``upstream_host`` for the addressed entry, plus the router edge's own
    per-request telemetry — rather than by trusting a field the response
    happens to carry. A pinned entry has no chain, so which upstream served it
    is a registry fact, not a discovery."""
    if usage is None:
        return {}
    return {
        "input_tokens": int(usage.input_tokens or 0),
        "output_tokens": int(usage.output_tokens or 0),
        "cache_read_input_tokens": int(usage.cache_read_tokens or 0),
        "cache_creation_input_tokens": int(
            (usage.cache_create_tokens or 0) + (usage.cache_create_1h_tokens or 0)
        ),
        "provider_cost_usd": usage.provider_cost_usd,
        "served_provider": served_provider,
    }


# ── Top-level entry ──────────────────────────────────────────────────────


def replay_artifact(
    *,
    artifact_key: str,
    target_model: str,
    bucket: str = DEFAULT_BUCKET,
    replay_prefix: str = DEFAULT_REPLAY_PREFIX,
    max_tokens: Optional[int] = None,
    s3_client: Optional[Any] = None,
    client_factory: Optional[Any] = None,
    model_spec: Optional[Any] = None,
    persist: bool = True,
) -> ReplayOutput:
    """Replay a single captured artifact under ``target_model``.

    Args:
        artifact_key: S3 key of the captured ``DecisionArtifact``.
        target_model: REGISTRY ENTRY ID to invoke (e.g.
            ``"deepseek-v4-flash"``) from
            ``alpha-engine-config/private-docs/LLM_MODEL_REGISTRY.yaml``.
            NOT a provider slug: ``"deepseek/deepseek-v4-flash"`` is an
            OpenRouter slug and is refused by the router with the
            addressable ids named (alpha-engine-config-I7878).
        bucket: S3 bucket; defaults to ``alpha-engine-research``.
        replay_prefix: S3 prefix for the replay output; defaults to
            ``decision_artifacts/_replay``.
        max_tokens: explicit max_tokens for the target call; defaults
            to ``DEFAULT_MAX_TOKENS``.
        s3_client: injected for tests.
        client_factory: krepis.llm.LLMClient test seam — a callable
            ``(spec, api_key) -> transport_client`` exposing
            ``chat.completions.create``. Production leaves it unset.
        model_spec: a ``ModelSpec`` already resolved by
            ``resolve_target_spec`` — batch runs resolve once per target
            model and pass it down. None resolves for this one call.
        persist: when False, returns the ``ReplayOutput`` without
            writing it to S3. Used by batch mode + tests.

    Returns:
        ``ReplayOutput`` populated with original + replay sides + per-
        agent comparison block.

    Schema resolution:
        Looks up the canonical Pydantic schema for the captured
        ``agent_id`` via ``nousergon_lib.agent_schemas.
        resolve_schema_for_agent``. Agents without a registered schema
        (or unknown families) are skipped — replay only runs against
        the 6 canonical agent types whose contracts live in the lib.
        This is intentional: replay-as-concordance-signal is meaningful
        only when the canonical schema enforces what "the same answer"
        means.
    """
    from nousergon_lib.agent_schemas import resolve_schema_for_agent

    s3 = s3_client or boto3.client("s3")

    artifact = _load_artifact(s3, bucket=bucket, key=artifact_key)

    # Skip deterministic v2 artifacts (e.g. ``executor:*`` algorithmic
    # agents). Per alpha-engine-lib v0.10.0, ``DecisionArtifact`` allows
    # ``model_metadata = None`` + ``full_prompt_context = None`` for
    # decisions produced without an LLM call. There's nothing to replay
    # under "rerun under a different model" framing — the decision is
    # deterministic given its inputs. Return a skip ReplayOutput so the
    # caller sees an explicit reason instead of a crash.
    if artifact.get("model_metadata") is None:
        agent_id = artifact.get("agent_id", "")
        return ReplayOutput(
            original_run_id=artifact.get("run_id", ""),
            original_agent_id=agent_id,
            original_model="deterministic",
            original_artifact_key=artifact_key,
            original_output=artifact.get("agent_output") or {},
            replay_model=target_model,
            replay_timestamp=datetime.now(timezone.utc).isoformat(),
            replay_output={},
            replay_output_kind="skipped",
            replay_cost={},
            replay_latency_ms=0,
            replay_error=(
                "deterministic decision (model_metadata=None) — no LLM to "
                "replay; deterministic captures don't go through "
                "model-substitution replay"
            ),
            comparison={
                "agreement_score": 0.0,
                "diff_summary": "skipped — deterministic decision",
            },
        )

    fpc = artifact.get("full_prompt_context") or {}
    system_prompt = fpc.get("system_prompt") or ""
    user_prompt = fpc.get("user_prompt") or ""

    agent_id = artifact.get("agent_id", "")
    original_model = (
        (artifact.get("model_metadata") or {}).get("model_name") or "unknown"
    )

    if _prompts_are_placeholder(system_prompt, user_prompt):
        # Capture wiring gap (see PLACEHOLDER_PROMPT_MARKER above) —
        # skip BEFORE the LLM call so no spend is burned replaying a
        # prompt with no content. Surfaced as kind="skipped" so batch
        # mode counts it separately from real replay errors; the fix
        # is research-side (wire the call site through track_llm_cost).
        return ReplayOutput(
            original_run_id=artifact.get("run_id", ""),
            original_agent_id=agent_id,
            original_model=original_model,
            original_artifact_key=artifact_key,
            original_output=artifact.get("agent_output") or {},
            replay_model=target_model,
            replay_timestamp=datetime.now(timezone.utc).isoformat(),
            replay_output={},
            replay_output_kind="skipped",
            replay_cost={},
            replay_latency_ms=0,
            replay_error=(
                "placeholder prompt context (capture wiring gap) — "
                "full_prompt_context carries the 'not yet wired through "
                "track_llm_cost' fallback stub instead of real prompts; "
                "nothing meaningful to replay. Fix is research-side: "
                "wire this agent's call site through track_llm_cost."
            ),
            comparison={
                "agreement_score": 0.0,
                "diff_summary": "skipped — placeholder prompt context",
                "scorer": "skipped",
                "agent_id_base": (agent_id or "").split(":", 1)[0],
            },
        )

    schema = resolve_schema_for_agent(agent_id)
    if schema is None:
        # Unknown agent family — no canonical schema to validate against.
        # Skip rather than try a free-form replay (which would produce a
        # noisy 0.0 concordance signal that pollutes downstream metrics).
        return ReplayOutput(
            original_run_id=artifact.get("run_id", ""),
            original_agent_id=agent_id,
            original_model=original_model,
            original_artifact_key=artifact_key,
            original_output=artifact.get("agent_output") or {},
            replay_model=target_model,
            replay_timestamp=datetime.now(timezone.utc).isoformat(),
            replay_output={},
            replay_output_kind="error",
            replay_cost={},
            replay_latency_ms=0,
            replay_error=(
                f"no canonical schema registered for agent_id={agent_id!r} — "
                "skipping replay (only the 6 canonical agent families have "
                "schemas in nousergon_lib.agent_schemas)"
            ),
            comparison={
                "agreement_score": 0.0,
                "diff_summary": "skipped — unknown agent_id family",
                "scorer": "skipped",
                "agent_id_base": (agent_id or "").split(":", 1)[0],
            },
        )

    parsed, usage, latency_ms, err = _invoke_target_with_schema(
        target_model=target_model,
        schema=schema,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
        client_factory=client_factory,
        model_spec=model_spec,
    )

    if err is not None:
        kind = "error"
        replay_output: dict[str, Any] = {}
    elif parsed is None:
        kind = "error"
        replay_output = {}
        err = "no parsed output returned by target model"
    else:
        kind = "structured"
        replay_output = parsed

    # Per-agent comparison (PR B). Only meaningful when the replay
    # actually produced structured output — error paths skip comparison
    # (they'd wash through the generic scorer with low agreement and
    # pollute downstream concordance metrics).
    original_output = artifact.get("agent_output") or {}
    if kind == "structured":
        from replay.comparison import compute_comparison
        comparison = compute_comparison(
            agent_id=agent_id,
            original_output=original_output,
            replay_output=replay_output,
        )
    else:
        comparison = {
            "agreement_score": 0.0,
            "diff_summary": f"replay produced no structured output (kind={kind})",
            "scorer": "skipped",
            "agent_id_base": (agent_id or "").split(":", 1)[0],
        }

    replay = ReplayOutput(
        original_run_id=artifact.get("run_id", ""),
        original_agent_id=agent_id,
        original_model=original_model,
        original_artifact_key=artifact_key,
        original_output=original_output,
        replay_model=target_model,
        replay_timestamp=datetime.now(timezone.utc).isoformat(),
        replay_output=replay_output,
        replay_output_kind=kind,
        replay_cost=usage,
        replay_latency_ms=latency_ms,
        replay_error=err,
        comparison=comparison,
    )

    if persist:
        replay_key = _persist_replay(
            s3, bucket=bucket, replay_prefix=replay_prefix, replay=replay,
        )
        logger.info(
            "[replay] persisted agent=%s original=%s target=%s kind=%s "
            "latency=%dms key=%s",
            replay.original_agent_id, original_model, target_model, kind,
            latency_ms, replay_key,
        )
    else:
        logger.info(
            "[replay] computed (no persist) agent=%s original=%s target=%s "
            "kind=%s latency=%dms",
            replay.original_agent_id, original_model, target_model, kind,
            latency_ms,
        )

    return replay
