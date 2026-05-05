"""Single-artifact replay — re-run a captured DecisionArtifact under a
different model and persist a side-by-side comparison.

Pipeline:

  1. Load ``DecisionArtifact`` JSON from S3 by key.
  2. Reconstruct the LLM call:
     - ``system`` = ``full_prompt_context.system_prompt``
     - ``messages`` = single user message with ``user_prompt`` (the
       captured user message already contains the input snapshot
       inlined into the prompt body — we don't re-execute tools, we
       feed the same context the original agent saw).
     - ``tools`` = ``full_prompt_context.tool_definitions`` if the
       original used structured output (forces tool use to match the
       original's output shape).
  3. Invoke target model via ``anthropic.Anthropic().messages.create()``.
     If the original used a tool block, force ``tool_choice`` so the
     replay produces the same JSON shape.
  4. Extract structured output (tool_use input dict) or text fallback.
  5. Persist side-by-side artifact at
     ``decision_artifacts/_replay/{run_id}/{original_model}_vs_{target_model}.json``.

Why bare Anthropic SDK rather than langchain_anthropic:

  - Keeps backtester free of the langchain dependency tree.
  - Tool-use forcing is what ``langchain_anthropic.with_structured_output``
    does under the hood; we reproduce that exactly via ``tool_choice``.
  - Replay is a pure SDK operation — no graph orchestration, no node
    wrappers, no LangSmith tracing needed in the replay path.

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
    replay_output_kind: str = "structured"  # "structured" | "text" | "error"
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


# ── Anthropic invocation ─────────────────────────────────────────────────


def _build_messages(user_prompt: str) -> list[dict]:
    """Single-user-message replay: feed the captured user_prompt back
    to the target model. The captured user_prompt already includes the
    relevant input snapshot inlined into the prompt body, so we don't
    need to re-present input_data_snapshot."""
    return [{"role": "user", "content": user_prompt}]


def _invoke_target(
    *,
    client: Any,
    target_model: str,
    system_prompt: str,
    user_prompt: str,
    tool_definitions: list[dict],
    max_tokens: int,
) -> tuple[Any, dict[str, Any], int, str | None]:
    """Call ``client.messages.create`` against the target model.

    Returns ``(parsed_output, usage_dict, latency_ms, error_or_none)``.
    On exception: ``(None, {}, latency, str(exc))`` — caller persists
    the error onto the replay artifact rather than raising. Replay is
    offline analysis; one failed re-invocation should never abort a
    batch.

    Tool-use mechanics:

    - If ``tool_definitions`` is non-empty, we pass them through and
      set ``tool_choice={"type": "any"}`` to force structured output.
      The first ``tool_use`` content block's ``input`` dict is the
      structured replay output.
    - If empty, we fall back to free-form text and return that under
      ``replay_output_kind="text"``. Most current agents use structured
      output, so this branch is the safety net.
    """
    start = time.monotonic()
    try:
        kwargs: dict[str, Any] = {
            "model": target_model,
            "system": system_prompt,
            "messages": _build_messages(user_prompt),
            "max_tokens": max_tokens,
        }
        if tool_definitions:
            kwargs["tools"] = tool_definitions
            # Force tool use so the model produces structured output that
            # matches the original's shape — no free-form fallback when
            # tools are defined.
            kwargs["tool_choice"] = {"type": "any"}

        response = client.messages.create(**kwargs)
        latency_ms = int((time.monotonic() - start) * 1000)
    except Exception as exc:  # noqa: BLE001 — replay never raises
        latency_ms = int((time.monotonic() - start) * 1000)
        return None, {}, latency_ms, str(exc)

    # Token usage — present on every Anthropic response per SDK contract.
    usage = getattr(response, "usage", None)
    usage_dict: dict[str, Any] = {}
    if usage is not None:
        usage_dict = {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
            "cache_read_input_tokens": getattr(
                usage, "cache_read_input_tokens", 0,
            ) or 0,
            "cache_creation_input_tokens": getattr(
                usage, "cache_creation_input_tokens", 0,
            ) or 0,
        }

    # Extract structured output from the first tool_use block.
    parsed: Any = None
    for block in getattr(response, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type == "tool_use":
            parsed = getattr(block, "input", None)
            break
        if block_type == "text" and parsed is None:
            parsed = {"_text": getattr(block, "text", "")}

    return parsed, usage_dict, latency_ms, None


# ── Top-level entry ──────────────────────────────────────────────────────


def replay_artifact(
    *,
    artifact_key: str,
    target_model: str,
    bucket: str = DEFAULT_BUCKET,
    replay_prefix: str = DEFAULT_REPLAY_PREFIX,
    max_tokens: Optional[int] = None,
    s3_client: Optional[Any] = None,
    anthropic_client: Optional[Any] = None,
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
            to ``DEFAULT_MAX_TOKENS`` if not on the captured artifact.
        s3_client / anthropic_client: injected for tests.
        persist: when False, returns the ``ReplayOutput`` without
            writing it to S3. Used by batch mode + tests.

    Returns:
        ``ReplayOutput`` populated with original + replay sides; the
        ``comparison`` field is empty in PR A and populated by PR B's
        per-agent scorers.
    """
    s3 = s3_client or boto3.client("s3")

    # Lazy-import the Anthropic client so a missing API key only fails
    # the actual replay path, not the import-time test fixtures.
    if anthropic_client is None:
        import anthropic
        anthropic_client = anthropic.Anthropic()

    artifact = _load_artifact(s3, bucket=bucket, key=artifact_key)

    fpc = artifact.get("full_prompt_context") or {}
    system_prompt = fpc.get("system_prompt") or ""
    user_prompt = fpc.get("user_prompt") or ""
    tool_definitions = fpc.get("tool_definitions") or []

    original_model = (
        artifact.get("model_metadata", {}).get("model_name") or "unknown"
    )

    parsed, usage, latency_ms, err = _invoke_target(
        client=anthropic_client,
        target_model=target_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tool_definitions=tool_definitions,
        max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
    )

    if err is not None:
        kind = "error"
        replay_output: dict[str, Any] = {}
    elif parsed is None:
        kind = "error"
        replay_output = {}
        err = "no content blocks returned by target model"
    elif "_text" in parsed and len(parsed) == 1:
        kind = "text"
        replay_output = parsed
    else:
        kind = "structured"
        replay_output = parsed if isinstance(parsed, dict) else {"_value": parsed}

    replay = ReplayOutput(
        original_run_id=artifact.get("run_id", ""),
        original_agent_id=artifact.get("agent_id", ""),
        original_model=original_model,
        original_artifact_key=artifact_key,
        original_output=artifact.get("agent_output") or {},
        replay_model=target_model,
        replay_timestamp=datetime.now(timezone.utc).isoformat(),
        replay_output=replay_output,
        replay_output_kind=kind,
        replay_cost=usage,
        replay_latency_ms=latency_ms,
        replay_error=err,
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
