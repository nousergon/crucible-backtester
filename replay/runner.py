"""Single-artifact replay — re-run a captured DecisionArtifact under a
different model and persist a side-by-side comparison.

Pipeline:

  1. Load ``DecisionArtifact`` JSON from S3 by key.
  2. Resolve the canonical Pydantic schema for the ``agent_id`` via
     ``alpha_engine_lib.agent_schemas.resolve_schema_for_agent``.
     Skips replay for unknown agent families (no schema to validate
     against → no meaningful concordance signal).
  3. Invoke target model via
     ``langchain_anthropic.ChatAnthropic.with_structured_output(SchemaClass,
     include_raw=True)`` — same invocation pattern as production
     agents, so any langchain change (model swap, prompt-caching
     defaults, retry posture) lands in both paths simultaneously.
  4. Extract the parsed Pydantic instance + ``model_dump`` it for the
     comparison + persistence layers. Pydantic validation errors
     surface as ``replay_error`` on the artifact — they're the
     silent-drift signal we wanted to expose (target model emits a
     structurally divergent output that the canonical schema rejects).
  5. Persist side-by-side artifact at
     ``decision_artifacts/_replay/{run_id}/{original_model}_vs_{target_model}.json``.

Why langchain_anthropic.with_structured_output (not bare SDK):

  - **Invocation isomorphism with production agents.** Production calls
    the model the same way; replay measures concordance against that
    invocation pattern, not against a divergent bare-SDK shim. When
    Claude 5 ships and langchain updates how it forces tool use, prod
    agents inherit the change automatically — replay would silently
    diverge if it stayed on bare SDK.
  - **Pydantic validation against the captured contract.** Catches the
    silent-drift class where a target model emits a slightly different
    structure that would otherwise wash through the comparison stage
    as an unexplained low concordance score.
  - **Schema portability.** Schemas live in ``alpha_engine_lib.agent_schemas``
    (lifted 2026-05-05, lib v0.4.0) so backtester can validate against
    the canonical contract without a heavy cross-repo dep on research.

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
    """Write the replay artifact to
    ``{replay_prefix}/{run_id}/{original_model}_vs_{target_model}.json``.
    Sanitize model names so colons or slashes don't break the S3 key."""
    safe_orig = replay.original_model.replace(":", "-").replace("/", "-")
    safe_target = replay.replay_model.replace(":", "-").replace("/", "-")
    key = (
        f"{replay_prefix}/{replay.original_run_id}/"
        f"{safe_orig}_vs_{safe_target}.json"
    )
    body = json.dumps(replay.to_dict(), indent=2, default=str).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    return key


# ── Target-model invocation (langchain_anthropic) ────────────────────────


def _build_messages(system_prompt: str, user_prompt: str) -> list[dict]:
    """Single-user-message replay: system + user, no chat history. The
    captured user_prompt already includes the relevant input snapshot
    inlined into the prompt body, so we don't need to re-present
    input_data_snapshot."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _invoke_target_with_schema(
    *,
    target_model: str,
    schema: type,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    chat_anthropic_factory: Any | None = None,
) -> tuple[Any, dict[str, Any], int, str | None]:
    """Invoke target model via langchain_anthropic.with_structured_output().

    Returns ``(parsed_output_dict, usage_dict, latency_ms, error_or_none)``.
    Same shape as the prior bare-SDK helper but with two SOTA upgrades:

    1. **Invocation isomorphism with production agents.** Production
       agents call the model via ``langchain_anthropic.ChatAnthropic.
       with_structured_output(SchemaClass)``. Replay uses the same path
       so any future langchain change (Claude 5 model swap, prompt-
       caching defaults, retry posture) lands in both prod + replay
       simultaneously.
    2. **Pydantic validation on the replay output.** with_structured_
       output validates the response against the captured schema and
       raises on drift. Catches the silent-drift class where a target
       model emits a slightly different structure that would otherwise
       wash through the comparison stage as a low concordance score.

    On exception: ``(None, {}, latency, str(exc))`` — caller persists
    the error onto the replay artifact rather than raising. Replay is
    offline analysis; one failed re-invocation should never abort a batch.

    The langchain ``include_raw=True`` mode is used to get both the
    parsed Pydantic instance AND the raw response (for usage extraction).
    Pydantic ValidationError surfaces as ``parsing_error`` in the raw
    output dict — caught + recorded on the replay artifact.
    """
    start = time.monotonic()
    try:
        # Lazy-import langchain to keep the module importable in tests
        # that don't need real LLM calls.
        if chat_anthropic_factory is None:
            from langchain_anthropic import ChatAnthropic
            chat_anthropic_factory = ChatAnthropic

        llm = chat_anthropic_factory(
            model=target_model,
            max_tokens=max_tokens,
        )
        structured_llm = llm.with_structured_output(schema, include_raw=True)
        response = structured_llm.invoke(_build_messages(system_prompt, user_prompt))
        latency_ms = int((time.monotonic() - start) * 1000)
    except Exception as exc:  # noqa: BLE001 — replay never raises
        latency_ms = int((time.monotonic() - start) * 1000)
        return None, {}, latency_ms, str(exc)

    # response shape with include_raw=True:
    #   {"raw": AIMessage, "parsed": Pydantic | None, "parsing_error": Exception | None}
    if not isinstance(response, dict):
        return None, {}, latency_ms, (
            f"unexpected response shape from with_structured_output: "
            f"{type(response).__name__}"
        )

    parsed_obj = response.get("parsed")
    parsing_error = response.get("parsing_error")
    raw = response.get("raw")

    # Token usage from the raw AIMessage (langchain populates
    # response_metadata['usage'] from the underlying SDK response).
    usage_dict: dict[str, Any] = {}
    if raw is not None:
        meta = getattr(raw, "response_metadata", {}) or {}
        usage = meta.get("usage") or {}
        if usage:
            usage_dict = {
                "input_tokens": int(usage.get("input_tokens", 0) or 0),
                "output_tokens": int(usage.get("output_tokens", 0) or 0),
                "cache_read_input_tokens": int(
                    usage.get("cache_read_input_tokens", 0) or 0,
                ),
                "cache_creation_input_tokens": int(
                    usage.get("cache_creation_input_tokens", 0) or 0,
                ),
            }

    if parsing_error is not None:
        # Pydantic validation failed against the captured schema. This
        # IS the silent-drift signal we wanted to surface — not an
        # error in the replay infrastructure but a real divergence
        # between the target model's output and the canonical contract.
        return None, usage_dict, latency_ms, (
            f"pydantic validation failed: {parsing_error}"
        )

    if parsed_obj is None:
        return None, usage_dict, latency_ms, (
            "with_structured_output returned no parsed object"
        )

    # Normalize to dict for the comparison + persistence layers.
    parsed_dict = (
        parsed_obj.model_dump() if hasattr(parsed_obj, "model_dump")
        else dict(parsed_obj) if isinstance(parsed_obj, dict)
        else {"_value": parsed_obj}
    )
    return parsed_dict, usage_dict, latency_ms, None


# ── Top-level entry ──────────────────────────────────────────────────────


def replay_artifact(
    *,
    artifact_key: str,
    target_model: str,
    bucket: str = DEFAULT_BUCKET,
    replay_prefix: str = DEFAULT_REPLAY_PREFIX,
    max_tokens: Optional[int] = None,
    s3_client: Optional[Any] = None,
    chat_anthropic_factory: Optional[Any] = None,
    persist: bool = True,
) -> ReplayOutput:
    """Replay a single captured artifact under ``target_model``.

    Args:
        artifact_key: S3 key of the captured ``DecisionArtifact``.
        target_model: Model identifier to invoke (e.g. ``"claude-haiku-4-5"``).
        bucket: S3 bucket; defaults to ``alpha-engine-research``.
        replay_prefix: S3 prefix for the replay output; defaults to
            ``decision_artifacts/_replay``.
        max_tokens: explicit max_tokens for the target call; defaults
            to ``DEFAULT_MAX_TOKENS``.
        s3_client / chat_anthropic_factory: injected for tests. The
            factory takes ``(model, max_tokens)`` and returns an object
            with ``with_structured_output(schema, include_raw=True)``;
            production passes ``langchain_anthropic.ChatAnthropic``.
        persist: when False, returns the ``ReplayOutput`` without
            writing it to S3. Used by batch mode + tests.

    Returns:
        ``ReplayOutput`` populated with original + replay sides + per-
        agent comparison block.

    Schema resolution:
        Looks up the canonical Pydantic schema for the captured
        ``agent_id`` via ``alpha_engine_lib.agent_schemas.
        resolve_schema_for_agent``. Agents without a registered schema
        (or unknown families) are skipped — replay only runs against
        the 6 canonical agent types whose contracts live in the lib.
        This is intentional: replay-as-concordance-signal is meaningful
        only when the canonical schema enforces what "the same answer"
        means.
    """
    from alpha_engine_lib.agent_schemas import resolve_schema_for_agent

    s3 = s3_client or boto3.client("s3")

    artifact = _load_artifact(s3, bucket=bucket, key=artifact_key)

    fpc = artifact.get("full_prompt_context") or {}
    system_prompt = fpc.get("system_prompt") or ""
    user_prompt = fpc.get("user_prompt") or ""

    agent_id = artifact.get("agent_id", "")
    original_model = (
        artifact.get("model_metadata", {}).get("model_name") or "unknown"
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
                "schemas in alpha_engine_lib.agent_schemas)"
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
        chat_anthropic_factory=chat_anthropic_factory,
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
