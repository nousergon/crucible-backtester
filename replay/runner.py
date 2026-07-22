"""Single-artifact replay — re-run a captured DecisionArtifact under a
different model and persist a side-by-side comparison.

Pipeline:

  1. Load ``DecisionArtifact`` JSON from S3 by key.
  2. Resolve the canonical Pydantic schema for the ``agent_id`` via
     ``nousergon_lib.agent_schemas.resolve_schema_for_agent``.
     Skips replay for unknown agent families (no schema to validate
     against → no meaningful concordance signal).
  3. Invoke target model via ``krepis.llm.LLMClient.structured()``
     against OpenRouter (DeepSeek V4 Flash by default — see
     alpha-engine-config-I2997 note below). Extracts the parsed
     Pydantic instance's ``model_dump()`` for the comparison +
     persistence layers. Schema-validation failures surface as
     ``replay_error`` on the artifact — they're the silent-drift signal
     we wanted to expose (target model emits a structurally divergent
     output that the canonical schema rejects).
  4. Persist side-by-side artifact under the canonical eval_artifacts
     layout: ``decision_artifacts/_replay/{run_id}_{original_model}_vs_{target_model}.json``
     (flat, YYMMDDHHMM run_id) + a ``decision_artifacts/_replay/latest.json``
     sidecar. Key format owned by ``nousergon_lib.eval_artifacts``.

**alpha-engine-config-I2997 (2026-07-19): migrated off direct Anthropic
(``langchain_anthropic.ChatAnthropic``) to the fleet-SOTA
``krepis.llm.LLMClient`` OpenRouter transport** (``target_model`` is now
an OpenRouter model id, e.g. ``"deepseek/deepseek-v4-flash"`` — the
default ReplayConcordance dispatches, see ``lambda_concordance/handler.py``
— not an Anthropic model name). This drops the "invocation isomorphism
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
``reasoning={"exclude": True}`` mirrors the fleet's other live DeepSeek
V4 OpenRouter consumers (morning-signal's ``fallback_llm``,
crucible-research's ``evals/judge_models.py::OPENROUTER_SHADOW``) —
without it a reasoning-capable OpenRouter model can burn its entire
output budget on chain-of-thought and return empty content
(config#1659 / config#2575).

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


def _resolve_openrouter_api_key(api_key: str | None = None) -> str:
    """Resolve the OpenRouter API key: explicit ``api_key`` arg wins, else
    ``nousergon_lib.secrets.get_secret`` (SSM-first —
    ``/alpha-engine/OPENROUTER_API_KEY``, readable by this Lambda's
    existing ``alpha-engine-ssm-read`` instance-role policy, no new IAM —
    env fallback). Same fleet-standard convention every other
    alpha-engine-config-I2997 call site uses. Raises loudly rather than
    letting client construction fail with a less diagnosable error.
    """
    if api_key:
        return api_key
    from nousergon_lib.secrets import get_secret

    key = get_secret("OPENROUTER_API_KEY", required=False, default=None)
    if not key:
        raise RuntimeError(
            "replay target-model invocation requires an OpenRouter API "
            "key: pass api_key= explicitly, or ensure "
            "nousergon_lib.secrets.get_secret('OPENROUTER_API_KEY') "
            "resolves (SSM parameter /alpha-engine/OPENROUTER_API_KEY, or "
            "the OPENROUTER_API_KEY environment variable as a fallback)."
        )
    return key


def _invoke_target_with_schema(
    *,
    target_model: str,
    schema: type,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    client_factory: Any | None = None,
    api_key: str | None = None,
) -> tuple[Any, dict[str, Any], int, str | None]:
    """Invoke target model via ``krepis.llm.LLMClient.structured()``
    (OpenRouter — DeepSeek V4 Flash by default; see module docstring for
    the full alpha-engine-config-I2997 migration rationale).

    Returns ``(parsed_output_dict, usage_dict, latency_ms, error_or_none)``
    — same shape as the pre-migration bare-SDK helper. On exception:
    ``(None, usage_dict, latency, str(exc))`` — caller persists the error
    onto the replay artifact rather than raising. Replay is offline
    analysis; one failed re-invocation should never abort a batch.
    ``usage_dict`` is populated even on a validation-exhaustion failure
    when the underlying ``LLMError`` carries partial usage (tokens were
    still spent on the failed attempt).

    ``client_factory`` is the krepis.llm.LLMClient test seam (mirrors the
    Think Tank pattern, matches every other alpha-engine-config-I2997 call
    site): a callable ``(spec, api_key) -> transport_client``. Production
    leaves it unset — ``LLMClient`` lazily builds the real
    ``openai.OpenAI`` client pointed at OpenRouter's ``base_url``.

    Schema-validation failure — the silent-drift signal this module
    exists to surface — and ordinary transport/SDK errors both funnel
    through ``krepis.llm.LLMError``/``LLMConfigError``/generic
    ``Exception`` here; the error string is not pattern-matched by any
    downstream consumer (verified: ``replay.batch``/``replay.cli`` only
    truncate + log it), so this function does not need to reproduce the
    exact pre-migration wording.
    """
    from krepis.llm import LLMClient, LLMError
    from krepis.llm_config import ModelSpec

    start = time.monotonic()
    try:
        resolved_key = _resolve_openrouter_api_key(api_key)
        spec = ModelSpec(
            provider="openrouter",
            model=target_model,
            max_tokens=max_tokens,
            # REQUIRED — see module docstring (live-verified 2026-07-19:
            # strict response_format=json_schema is unreliable for
            # DeepSeek-family models on OpenRouter; the JSON-instruction +
            # tolerant-extraction fallback round-tripped correctly).
            structured_outputs=False,
            reasoning={"exclude": True},
        )
        client = LLMClient(spec, api_key=resolved_key, client_factory=client_factory)
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
        # errors (bad/missing api key, network, malformed base_url); replay
        # never raises.
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
    provider identity isn't load-bearing for the jurisdiction check."""
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
    api_key: Optional[str] = None,
    persist: bool = True,
) -> ReplayOutput:
    """Replay a single captured artifact under ``target_model``.

    Args:
        artifact_key: S3 key of the captured ``DecisionArtifact``.
        target_model: OpenRouter model id to invoke (e.g.
            ``"deepseek/deepseek-v4-flash"`` — alpha-engine-config-I2997;
            was an Anthropic model name pre-migration).
        bucket: S3 bucket; defaults to ``alpha-engine-research``.
        replay_prefix: S3 prefix for the replay output; defaults to
            ``decision_artifacts/_replay``.
        max_tokens: explicit max_tokens for the target call; defaults
            to ``DEFAULT_MAX_TOKENS``.
        s3_client: injected for tests.
        client_factory: krepis.llm.LLMClient test seam — a callable
            ``(spec, api_key) -> transport_client`` exposing
            ``chat.completions.create``. Production leaves it unset.
        api_key: explicit OpenRouter API key override; defaults to
            ``nousergon_lib.secrets.get_secret("OPENROUTER_API_KEY")``
            (see ``_resolve_openrouter_api_key``).
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
        api_key=api_key,
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
