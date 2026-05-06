"""
provenance_grounding.py — Per-agent tool-call + input-trace metrics.

The fourth leg of the agent-justification stack:

  1. Cross-week rationale clustering (alpha-engine-research, 2026-05-05)
     — what the agent says over time
  2. Counterfactual rule fit (alpha-engine-backtester, 2026-05-06)
     — whether a depth-≤3 tree replicates agent decisions
  3. Cheap-model concordance (alpha-engine-backtester, 2026-05-06)
     — whether a smaller model produces the same answer
  4. Provenance grounding (this module)
     — what the agent looked at to produce the answer

Goal: detect agents emitting confident output without consulting tools or
inputs (hallucination signal) and degenerate behavior where an agent's
tool-call distribution collapses to a single tool over many runs (likely
rule-equivalence). Composes with the other three legs to give the per-
agent "is this agent earning its cost / should it be retired" verdict.

Design:

- Pure consumer-side. Reads ``decision_artifacts/{Y}/{M}/{D}/{agent_id}/
  {run_id}.json`` and computes metrics. No new schema, no prompt change,
  no LLM tax on the production path.
- Tool calls live in nested paths (sector_teams stash them under
  ``agent_output.quant_output.tool_calls`` and ``qual_output.tool_calls``
  because the team is a sub-graph). The extractor walks the agent_output
  dict and aggregates every ``tool_calls`` list it encounters.
- Tool-equipped agents (sector_quant, sector_qual, macro_economist) are
  flagged when a captured artifact has zero tool calls. CIO + peer_review
  legitimately make zero tool calls (synthesizers, no fetch tools), so
  they're excluded from the alarm denominator.

Returns the standard backtester-evaluator status dict so the existing
``CompletenessTracker.run_module`` pattern handles it without bespoke
wiring.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

import boto3

logger = logging.getLogger(__name__)


# ── Canonical agents (mirrors decision_capture_coverage) ────────────────────

CANONICAL_SECTORS = (
    "consumer", "defensives", "financials",
    "healthcare", "industrials", "technology",
)

CANONICAL_AGENTS: tuple[str, ...] = (
    "macro_economist",
    "ic_cio",
    *(f"sector_team:{s}" for s in CANONICAL_SECTORS),
)

# Agents that are EXPECTED to invoke tools. Zero tool calls on these is a
# hallucination signal. CIO + peer_review legitimately have zero (they're
# synthesizers operating on aggregated team output, no fetch tools).
TOOL_EQUIPPED_AGENTS: frozenset[str] = frozenset({
    "macro_economist",
    *(f"sector_team:{s}" for s in CANONICAL_SECTORS),
})

_META_PREFIXES = (
    "_eval", "_eval_judge_only", "_replay", "_replay_summary",
    "_cost", "_cost_raw", "_analysis", "_diagnostics",
)

DEFAULT_BUCKET = "alpha-engine-research"
DEFAULT_CAPTURE_PREFIX = "decision_artifacts"


# ── Tool-call walker ────────────────────────────────────────────────────────


def _walk_tool_calls(agent_output: dict[str, Any]) -> list[dict[str, Any]]:
    """Recursively collect every ``tool_calls`` list found anywhere in
    the agent_output dict tree.

    Sector teams nest tool calls under ``agent_output.quant_output.tool_calls``
    and ``agent_output.qual_output.tool_calls`` because the team is a
    sub-graph with quant + qual + peer_review sub-agents. Macro economist
    keeps them at the top-level ``agent_output.tool_calls``. CIO has no
    tool calls (synthesizer).

    Returns the merged list. Each entry is a dict matching the
    ``ToolCall`` schema (tool, ticker, args, result_summary).
    """
    collected: list[dict[str, Any]] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            tcs = node.get("tool_calls")
            if isinstance(tcs, list):
                for tc in tcs:
                    if isinstance(tc, dict):
                        collected.append(tc)
            for k, v in node.items():
                if k == "tool_calls":
                    continue  # already collected
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(agent_output)
    return collected


# ── Per-artifact metrics ────────────────────────────────────────────────────


def _artifact_metrics(artifact: dict[str, Any]) -> dict[str, Any]:
    """Compute per-artifact provenance metrics.

    Returns a dict with the load-bearing fields. Aggregator combines these
    across all artifacts for an agent on a Saturday.
    """
    agent_output = artifact.get("agent_output") or {}
    input_snapshot = artifact.get("input_data_snapshot") or {}

    tool_calls = _walk_tool_calls(agent_output)
    n_tool_calls = len(tool_calls)
    distinct_tools = sorted({
        tc.get("tool") for tc in tool_calls if tc.get("tool")
    })
    tool_distribution = Counter(
        tc.get("tool") for tc in tool_calls if tc.get("tool")
    )

    # input_consumption_ratio: fraction of input_data_snapshot top-level
    # field names that appear referenced (substring) in the serialized
    # agent_output prose. Rough but cheap; a non-zero score here means the
    # agent's output references the inputs it had access to. A zero or
    # near-zero score with non-empty input is a soft hallucination signal.
    snapshot_keys = [k for k in input_snapshot.keys() if isinstance(k, str)]
    if snapshot_keys:
        output_blob = json.dumps(agent_output, default=str)
        n_referenced = sum(1 for k in snapshot_keys if k in output_blob)
        input_consumption_ratio = round(n_referenced / len(snapshot_keys), 3)
    else:
        input_consumption_ratio = 0.0

    return {
        "n_tool_calls": n_tool_calls,
        "n_distinct_tools": len(distinct_tools),
        "distinct_tools": distinct_tools,
        "tool_distribution": dict(tool_distribution),
        "input_consumption_ratio": input_consumption_ratio,
        "input_snapshot_n_keys": len(snapshot_keys),
        "is_truncated": artifact.get("input_data_truncated_at") is not None,
    }


# ── Per-agent aggregation ───────────────────────────────────────────────────


def _agent_metrics(per_artifact: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-artifact metrics into a per-agent summary."""
    if not per_artifact:
        return {
            "n_artifacts": 0,
            "n_zero_call_artifacts": 0,
            "pct_zero_call_outputs": 0.0,
            "mean_n_tool_calls": 0.0,
            "mean_n_distinct_tools": 0.0,
            "tool_distribution": {},
            "mean_input_consumption_ratio": 0.0,
            "n_truncated": 0,
        }

    n = len(per_artifact)
    n_zero = sum(1 for m in per_artifact if m["n_tool_calls"] == 0)
    sum_calls = sum(m["n_tool_calls"] for m in per_artifact)
    sum_distinct = sum(m["n_distinct_tools"] for m in per_artifact)
    sum_consumption = sum(m["input_consumption_ratio"] for m in per_artifact)
    n_truncated = sum(1 for m in per_artifact if m["is_truncated"])

    merged_distribution: Counter[str] = Counter()
    for m in per_artifact:
        merged_distribution.update(m["tool_distribution"])

    return {
        "n_artifacts": n,
        "n_zero_call_artifacts": n_zero,
        "pct_zero_call_outputs": round((n_zero / n) * 100.0, 2),
        "mean_n_tool_calls": round(sum_calls / n, 2),
        "mean_n_distinct_tools": round(sum_distinct / n, 2),
        "tool_distribution": dict(merged_distribution),
        "mean_input_consumption_ratio": round(sum_consumption / n, 3),
        "n_truncated": n_truncated,
    }


# ── S3 listing + reading ────────────────────────────────────────────────────


def _list_artifact_keys(
    s3: Any,
    *,
    bucket: str,
    capture_prefix: str,
    date: datetime,
) -> dict[str, list[str]]:
    """List artifact keys under ``{capture_prefix}/{Y}/{M}/{D}/`` grouped
    by agent_id (the directory immediately below the date partition).

    Returns ``{agent_id: [key, ...]}``. Empty dict when no objects exist.
    """
    prefix = (
        f"{capture_prefix}/{date.strftime('%Y')}/"
        f"{date.strftime('%m')}/{date.strftime('%d')}/"
    )
    grouped: dict[str, list[str]] = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            if any(f"/{p}/" in key for p in _META_PREFIXES):
                continue
            relative = key[len(prefix):]
            parts = relative.split("/")
            if len(parts) < 2:
                continue
            agent_id = parts[0]
            grouped.setdefault(agent_id, []).append(key)
    return grouped


def _read_artifact(s3: Any, *, bucket: str, key: str) -> dict[str, Any] | None:
    """Read + parse one captured artifact. Returns None on read/parse error
    so the aggregator can continue past one bad artifact."""
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except Exception as e:
        logger.warning(
            "provenance_grounding: failed to read s3://%s/%s: %s",
            bucket, key, e,
        )
        return None


# ── Per-Saturday compute ────────────────────────────────────────────────────


def _saturday_provenance_for(
    s3: Any,
    *,
    bucket: str,
    capture_prefix: str,
    saturday: datetime,
) -> dict[str, Any]:
    """Compute per-agent provenance metrics for one Saturday SF date."""
    grouped = _list_artifact_keys(
        s3, bucket=bucket, capture_prefix=capture_prefix, date=saturday,
    )

    per_agent: dict[str, dict[str, Any]] = {}
    n_total_artifacts_read = 0
    for agent_id, keys in grouped.items():
        if agent_id.startswith("thesis_update:"):
            continue  # variable cardinality; not in canonical denominator
        per_artifact: list[dict[str, Any]] = []
        for key in keys:
            artifact = _read_artifact(s3, bucket=bucket, key=key)
            if artifact is None:
                continue
            per_artifact.append(_artifact_metrics(artifact))
            n_total_artifacts_read += 1
        per_agent[agent_id] = _agent_metrics(per_artifact)

    # Tool-equipped alarm: any tool-equipped agent with > 0 zero-call
    # artifacts on this Saturday. Listed for visibility; doesn't gate.
    tool_equipped_alarms = sorted(
        agent_id for agent_id, m in per_agent.items()
        if agent_id in TOOL_EQUIPPED_AGENTS and m["n_zero_call_artifacts"] > 0
    )

    return {
        "date": saturday.strftime("%Y-%m-%d"),
        "n_total_artifacts_read": n_total_artifacts_read,
        "per_agent": per_agent,
        "tool_equipped_alarms": tool_equipped_alarms,
    }


# ── Most-recent-Saturday resolution ─────────────────────────────────────────


def _most_recent_saturday_with_captures(
    s3: Any,
    *,
    bucket: str,
    capture_prefix: str,
    end_date: datetime,
    max_lookback_days: int = 7,
) -> datetime | None:
    """Walk back up to ``max_lookback_days`` from ``end_date`` and return
    the first Saturday whose date partition has any captures. None if
    no Saturday in the window has captures."""
    for offset in range(max_lookback_days + 1):
        d = end_date - timedelta(days=offset)
        if d.weekday() != 5:
            continue
        grouped = _list_artifact_keys(
            s3, bucket=bucket, capture_prefix=capture_prefix, date=d,
        )
        if grouped:
            return d
    return None


# ── Public entry point ──────────────────────────────────────────────────────


def compute_provenance_grounding(
    bucket: str = DEFAULT_BUCKET,
    run_date: str | None = None,
    lookback_weeks: int = 8,
    capture_prefix: str = DEFAULT_CAPTURE_PREFIX,
    s3_client: Any = None,
) -> dict[str, Any]:
    """Compute per-agent provenance grounding metrics for the most recent
    Saturday SF run on or before ``run_date``, plus an N-week rolling
    aggregate.

    Returns dict with keys:
        status: "ok" | "no_recent_sf_run" | "error"
        run_date: input run_date (echoed)
        most_recent_sf_date: ISO date of the Saturday computed
        per_agent: per-agent metrics on the most recent Saturday
        tool_equipped_alarms: list of tool-equipped agents with zero-call
            artifacts on the most recent Saturday
        rolling: N-week rolling aggregate per agent (mean of pct_zero,
            tool_distribution union, etc.)
    """
    run_date = run_date or datetime.utcnow().strftime("%Y-%m-%d")
    try:
        end_date = datetime.strptime(run_date, "%Y-%m-%d")
    except ValueError as e:
        return {"status": "error", "error": f"invalid run_date: {e}"}

    s3 = s3_client or boto3.client("s3")

    try:
        most_recent = _most_recent_saturday_with_captures(
            s3, bucket=bucket, capture_prefix=capture_prefix,
            end_date=end_date,
        )
    except Exception as e:
        logger.exception("provenance_grounding: S3 listing failed")
        return {"status": "error", "error": f"S3 listing failed: {e}"}

    if most_recent is None:
        return {
            "status": "no_recent_sf_run",
            "run_date": run_date,
            "reason": (
                f"no Saturday with captures within 7 days of {run_date} "
                f"under s3://{bucket}/{capture_prefix}/"
            ),
        }

    most_recent_summary = _saturday_provenance_for(
        s3, bucket=bucket, capture_prefix=capture_prefix,
        saturday=most_recent,
    )

    # Trailing window
    per_saturday: list[dict[str, Any]] = [most_recent_summary]
    for week in range(1, lookback_weeks):
        sat = most_recent - timedelta(days=7 * week)
        try:
            summary = _saturday_provenance_for(
                s3, bucket=bucket, capture_prefix=capture_prefix, saturday=sat,
            )
        except Exception as e:
            logger.warning(
                "provenance_grounding: per-Saturday lookup failed for %s: %s",
                sat.strftime("%Y-%m-%d"), e,
            )
            continue
        if summary["n_total_artifacts_read"] > 0:
            per_saturday.append(summary)

    # Rolling per-agent: mean pct_zero_call_outputs + union tool_distribution
    rolling_per_agent: dict[str, dict[str, Any]] = {}
    for agent_id in CANONICAL_AGENTS:
        zeros: list[float] = []
        consumptions: list[float] = []
        merged_dist: Counter[str] = Counter()
        n_artifacts_total = 0
        for sat in per_saturday:
            m = sat["per_agent"].get(agent_id)
            if not m or m["n_artifacts"] == 0:
                continue
            zeros.append(m["pct_zero_call_outputs"])
            consumptions.append(m["mean_input_consumption_ratio"])
            merged_dist.update(m["tool_distribution"])
            n_artifacts_total += m["n_artifacts"]
        if n_artifacts_total == 0:
            continue
        rolling_per_agent[agent_id] = {
            "n_saturdays": len(zeros),
            "n_artifacts_total": n_artifacts_total,
            "mean_pct_zero_call_outputs": (
                round(sum(zeros) / len(zeros), 2) if zeros else 0.0
            ),
            "mean_input_consumption_ratio": (
                round(sum(consumptions) / len(consumptions), 3)
                if consumptions else 0.0
            ),
            "tool_distribution": dict(merged_dist),
            "n_distinct_tools": len(merged_dist),
        }

    return {
        "status": "ok",
        "run_date": run_date,
        "most_recent_sf_date": most_recent.strftime("%Y-%m-%d"),
        "per_agent": most_recent_summary["per_agent"],
        "tool_equipped_alarms": most_recent_summary["tool_equipped_alarms"],
        "n_total_artifacts_read": most_recent_summary["n_total_artifacts_read"],
        "rolling": {
            "n_saturdays_with_data": len(per_saturday),
            "per_agent": rolling_per_agent,
        },
    }
