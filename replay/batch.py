"""Batch replay — iterate over a date range × target models × agents,
aggregate per-(agent_id_base, target_model) concordance, emit CloudWatch
metrics, and persist a summary artifact.

Per ROADMAP P0 "Replay harness + agent-justification gate" (Model-
Agnostic Capability Upgrade deliverable #7 — agent-justification gate
signal #3, cheap-model concordance):

  Replays each agent's recent corpus under a smaller/cheaper model;
  emits CW metric agent_cheap_model_concordance per (agent_id,
  target_model). >0.9 concordance → larger model isn't earning its
  cost.

Pipeline:

  1. List captured DecisionArtifacts under
     ``decision_artifacts/{YYYY}/{MM}/{DD}/`` for the date range.
     Filter to the 6 canonical agent families (others are skipped by
     replay_artifact anyway, but pre-filtering avoids wasted listing).
  2. For each (artifact × target_model): call replay_artifact with
     persist=False. The replay's comparison.agreement_score is the
     observation we aggregate.
  3. Group observations by (agent_id_base, target_model). Compute
     mean + count + min + max per group.
  4. Emit CloudWatch metric ``agent_cheap_model_concordance`` per
     group (Dimensions: judged_agent_id, target_model).
  5. Persist per-target-model analysis JSON under the canonical
     eval_artifacts layout: ``decision_artifacts/_replay_summary/{run_id}_{target_model}.json``
     (flat, YYMMDDHHMM run_id) + a ``decision_artifacts/_replay_summary/latest.json``
     sidecar. Key format owned by ``nousergon_lib.eval_artifacts``.

Cost discipline:

  - Each replayed artifact costs target-model tokens. A typical week
    has ~30 artifacts across the 6 agent families; an 8-week window
    × 1 target model ≈ 240 calls. At Haiku rates (~$0.0005/call) this
    is ~$0.12 per batch run. At Sonnet rates (~$0.005/call) ~$1.20.
    The ``max_artifacts`` cap (default 500) bounds blast radius.
  - ``dry_run=True`` skips replay entirely, listing what WOULD be
    replayed and persisting nothing. Use for smoke tests + cost
    estimation.

Composes with:

  - Cross-week rationale clustering — measures *what* an agent emits
    across weeks. Concordance measures *whether a different model
    would emit the same given the same input*. Together they're the
    triple alongside the counterfactual-rule-fit signal.
  - LLM-as-judge framework — judges measure output quality;
    concordance measures whether cheaper model = same output.
  - Cost telemetry — replay token counts roll into the persisted
    batch artifact's ``per_model_cost`` block.
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import boto3

from replay.runner import replay_artifact

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────


DEFAULT_NAMESPACE = "AlphaEngine/Eval"
"""Same namespace as the eval-judge + rationale-clustering metrics —
the agent-justification dashboard reads everything from one stream."""

DEFAULT_METRIC_NAME = "agent_cheap_model_concordance"
"""Mean agreement_score [0, 1] per (judged_agent_id, target_model).
>0.9 indicates the cheaper model produces equivalent output → demote
candidate."""

DEFAULT_BUCKET = "alpha-engine-research"
DEFAULT_CAPTURE_PREFIX = "decision_artifacts"
DEFAULT_SUMMARY_PREFIX = "decision_artifacts/_replay_summary"
DEFAULT_REPLAY_PREFIX = "decision_artifacts/_replay"

DEFAULT_MAX_ARTIFACTS = 500
"""Hard cap on artifacts replayed per batch run. Production weekly
cadence on the trailing 8-week window stays well under this; the cap
exists to bound cost on accidental wide-window runs (e.g. if the
caller passes ``window_days=365`` by mistake)."""

MIN_OBSERVATIONS_FOR_CONCORDANCE = 3
"""Below this count, the per-group mean is statistically meaningless —
emit metric as None (skip) rather than report a noisy value. Mirrors
the rationale-clustering thin-sample threshold."""


# ── Per-agent corpus listing ─────────────────────────────────────────────


def _list_artifact_keys_in_window(
    s3: Any,
    *,
    bucket: str,
    capture_prefix: str,
    end_date: datetime,
    window_days: int,
    agent_filter: list[str] | None = None,
) -> list[str]:
    """List captured-artifact keys under
    ``{capture_prefix}/{Y}/{M}/{D}/`` for each day in the trailing
    window. Excludes the ``_eval/``, ``_eval_judge_only/``,
    ``_analysis/``, ``_cost*/``, ``_replay/``, ``_replay_summary/``
    subtrees so we only ingest production captures.

    ``agent_filter`` is a list of base agent_ids (e.g. ``["sector_quant",
    "ic_cio"]``) — only artifacts whose path matches one of these
    families are kept. None / empty = include all canonical families.
    """
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []

    for day_offset in range(window_days):
        day = end_date - timedelta(days=day_offset)
        prefix = (
            f"{capture_prefix}/{day.strftime('%Y')}/"
            f"{day.strftime('%m')}/{day.strftime('%d')}/"
        )
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".json"):
                    continue
                # Skip every meta-prefix (eval, analysis, cost, replay).
                if any(
                    f"/_{p}/" in key
                    for p in (
                        "eval", "eval_judge_only", "analysis",
                        "cost", "cost_raw",
                        "replay", "replay_summary",
                    )
                ):
                    continue
                if agent_filter:
                    base_id = _agent_id_base_from_key(key)
                    if base_id is None or base_id not in agent_filter:
                        continue
                keys.append(key)

    return keys


def _agent_id_base_from_key(key: str) -> Optional[str]:
    """Extract the base agent_id from an S3 key shaped
    ``decision_artifacts/{Y}/{M}/{D}/{agent_id}/{run_id}.json``.
    The base is the part before the first colon (e.g.
    ``sector_quant`` for ``sector_quant:technology``)."""
    parts = key.split("/")
    if len(parts) < 2:
        return None
    full_id = parts[-2]
    return full_id.split(":", 1)[0]


# ── Per-group aggregation ────────────────────────────────────────────────


def _aggregate_group(observations: list[float]) -> dict[str, Any]:
    """Per-(agent_id_base, target_model) summary stats. Returns mean +
    count + min + max + stdev when n >= 2.

    Filters NaN / None from observations defensively before reducing —
    a comparison scorer should always emit a float, but a future bug
    where an unknown-agent path emitted None would propagate into the
    aggregate as a TypeError on min/max. Defensive filter keeps the
    aggregate robust."""
    obs = [o for o in observations if isinstance(o, (int, float))]
    if not obs:
        return {
            "n": 0, "mean": None, "min": None, "max": None, "stdev": None,
        }
    return {
        "n": len(obs),
        "mean": statistics.fmean(obs),
        "min": min(obs),
        "max": max(obs),
        "stdev": statistics.stdev(obs) if len(obs) >= 2 else 0.0,
    }


# ── CloudWatch metric emission ───────────────────────────────────────────


def _emit_concordance_metric(
    cw: Any,
    *,
    namespace: str,
    metric_name: str,
    agent_id_base: str,
    target_model: str,
    mean_agreement: float,
    n_observations: int,
    timestamp: datetime,
) -> None:
    """One datapoint per (agent_id_base, target_model) per batch run.
    Emit shape mirrors the rationale-clustering metric: a primary
    [0, 1] value + an _n_observations counter so the dashboard can
    flag thin-sample groups."""
    cw.put_metric_data(
        Namespace=namespace,
        MetricData=[
            {
                "MetricName": metric_name,
                "Dimensions": [
                    {"Name": "judged_agent_id", "Value": agent_id_base},
                    {"Name": "target_model", "Value": target_model},
                ],
                "Value": float(mean_agreement),
                "Unit": "None",
                "Timestamp": timestamp,
            },
            {
                "MetricName": f"{metric_name}_n_observations",
                "Dimensions": [
                    {"Name": "judged_agent_id", "Value": agent_id_base},
                    {"Name": "target_model", "Value": target_model},
                ],
                "Value": float(n_observations),
                "Unit": "Count",
                "Timestamp": timestamp,
            },
        ],
    )


# ── Persistence ──────────────────────────────────────────────────────────


def _persist_batch_summary(
    s3: Any,
    *,
    bucket: str,
    summary_prefix: str,
    target_model: str,
    end_date: datetime,
    payload: dict[str, Any],
) -> str:
    """Write the per-target-model batch summary JSON under the canonical
    ``eval_artifacts`` layout: a flat dated key
    ``{summary_prefix}/{run_id}_{target_model}.json`` plus a
    ``{summary_prefix}/latest.json`` operator-UX sidecar.

    Migrated from the legacy date-partitioned ``{summary_prefix}/
    {YYYY-MM-DD}/{target_model}.json`` layout (backtester #179 deferred
    this site; config#792). The ``run_id`` is minted from the batch
    ``end_date`` via ``new_eval_run_id`` so the YYMMDDHHMM timestamp
    encodes the run instant (replacing the ``{YYYY-MM-DD}/`` partition —
    the structured run_id already sorts chronologically across the flat
    prefix). The ``target_model`` survives as the canonical multi-file
    basename and is sanitized to drop colon-aliases (e.g. ``:live``) so
    the S3 key stays portable.

    The dated key is the forensic source of truth; the ``latest.json``
    sidecar is a pure mirror of the most-recently-written summary. Key
    format is owned by ``nousergon_lib.eval_artifacts``.

    Note: with multiple target_models in one batch run, each writes its
    own dated key (distinct basename) but they all mirror into the single
    shared ``latest.json`` — last writer wins, matching the per-pipeline
    "latest" semantic (operators inspect the dated keys for the full set).
    """
    from nousergon_lib.eval_artifacts import (
        eval_artifact_key,
        eval_latest_key,
        new_eval_run_id,
    )

    safe_target = target_model.replace(":", "-").replace("/", "-")
    run_id = new_eval_run_id(now=end_date)
    key = eval_artifact_key(
        summary_prefix, run_id, basename=f"{safe_target}.json",
    )
    enriched = {**payload, "run_id": run_id}
    body = json.dumps(enriched, indent=2, default=str).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")

    # Operator-UX latest sidecar — pure mirror of the dated artifact.
    s3.put_object(
        Bucket=bucket,
        Key=eval_latest_key(summary_prefix),
        Body=body,
        ContentType="application/json",
    )
    return key


# ── Top-level pipeline ───────────────────────────────────────────────────


def compute_and_emit_concordance(
    *,
    target_models: list[str],
    end_time: Optional[datetime] = None,
    window_days: int = 56,
    agent_filter: Optional[list[str]] = None,
    bucket: str = DEFAULT_BUCKET,
    capture_prefix: str = DEFAULT_CAPTURE_PREFIX,
    summary_prefix: str = DEFAULT_SUMMARY_PREFIX,
    replay_prefix: str = DEFAULT_REPLAY_PREFIX,
    namespace: str = DEFAULT_NAMESPACE,
    metric_name: str = DEFAULT_METRIC_NAME,
    max_artifacts: int = DEFAULT_MAX_ARTIFACTS,
    s3_client: Optional[Any] = None,
    cloudwatch_client: Optional[Any] = None,
    client_factory: Optional[Any] = None,
    api_key: Optional[str] = None,
    emit_metrics: bool = True,
    persist_per_replay: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Replay the trailing-window corpus under each target_model,
    aggregate concordance per (agent_id_base, target_model), emit
    CloudWatch metrics, and persist a per-target-model batch summary.

    Args:
        target_models: list of OpenRouter model ids to replay against
            (alpha-engine-config-I2997, 2026-07-19 — was an Anthropic
            model name pre-migration). Common shape:
            ``["deepseek/deepseek-v4-flash"]`` — the default
            ReplayConcordance dispatches (see
            ``lambda_concordance/handler.py``).
        end_time: window ends at this UTC instant (defaults to now).
        window_days: trailing window — default 8 weeks (56 days).
        agent_filter: list of base agent_ids to include. None = all
            canonical families (sector_quant, sector_qual,
            sector_peer_review, macro_economist, ic_cio, thesis_update).
        max_artifacts: hard cap on artifacts replayed per batch run
            (cost guard).
        emit_metrics: when False, skips CW emission. Used by tests +
            local smoke runs.
        persist_per_replay: when True, each individual replay is
            persisted to ``decision_artifacts/_replay/`` in addition
            to the batch summary. Default False — batch runs are
            usually about the aggregate, and per-replay persistence
            multiplies S3 PUTs ~N×.
        dry_run: when True, lists candidate artifacts + skips replay
            calls + persists nothing. Returns a summary with
            ``would_replay`` rather than ``per_agent``.

    Returns:
        Summary dict with per-target-model concordance + per-agent
        aggregates + run-level cost + failures. Same shape pattern as
        the eval orchestrator + rationale clustering — operators can
        pattern-match on ``failed`` to decide alarm severity.
    """
    s3 = s3_client or boto3.client("s3")
    cw = cloudwatch_client or (
        boto3.client("cloudwatch") if (emit_metrics and not dry_run) else None
    )
    end = end_time or datetime.now(timezone.utc)
    window_start = end - timedelta(days=window_days)

    # Default to the 6 canonical agent families if no filter supplied.
    if agent_filter is None:
        agent_filter = [
            "sector_quant",
            "sector_qual",
            "sector_peer_review",
            "macro_economist",
            "ic_cio",
            "thesis_update",
        ]

    keys = _list_artifact_keys_in_window(
        s3,
        bucket=bucket,
        capture_prefix=capture_prefix,
        end_date=end,
        window_days=window_days,
        agent_filter=agent_filter,
    )

    if len(keys) > max_artifacts:
        logger.warning(
            "[batch_replay] discovered %d artifacts; capping at "
            "max_artifacts=%d (use a higher cap if intended)",
            len(keys), max_artifacts,
        )
        keys = keys[:max_artifacts]

    logger.info(
        "[batch_replay] window=[%s, %s] target_models=%s "
        "artifact_count=%d agent_filter=%s",
        window_start.isoformat(), end.isoformat(),
        target_models, len(keys), agent_filter,
    )

    if dry_run:
        return {
            "dry_run": True,
            "window_start": window_start.isoformat(),
            "window_end": end.isoformat(),
            "target_models": target_models,
            "agent_filter": agent_filter,
            "would_replay": len(keys),
            "would_replay_keys": keys[:50],  # trim for log/return ergonomics
        }

    # Outer summary collects per-target-model results.
    per_target_summary: list[dict[str, Any]] = []

    for target_model in target_models:
        # Group observations by agent_id_base.
        observations_by_agent: dict[str, list[float]] = defaultdict(list)
        cost_total = {"input_tokens": 0, "output_tokens": 0}
        replay_failures: list[dict[str, str]] = []
        replay_skips: list[dict[str, str]] = []
        n_replayed = 0

        for key in keys:
            try:
                replay = replay_artifact(
                    artifact_key=key,
                    target_model=target_model,
                    bucket=bucket,
                    replay_prefix=replay_prefix,
                    s3_client=s3,
                    client_factory=client_factory,
                    api_key=api_key,
                    persist=persist_per_replay,
                )
            except Exception as exc:  # noqa: BLE001 — never abort a batch
                replay_failures.append({
                    "key": key, "stage": "replay_artifact_call", "error": str(exc),
                })
                logger.exception(
                    "[batch_replay] replay_artifact raised key=%s target=%s",
                    key, target_model,
                )
                continue

            n_replayed += 1

            # Roll up token cost (best-effort).
            for k in ("input_tokens", "output_tokens"):
                cost_total[k] += int(replay.replay_cost.get(k, 0) or 0)

            if replay.replay_output_kind == "skipped":
                # Deliberate non-replay (deterministic artifact or
                # placeholder prompt context from a capture wiring gap)
                # — counted separately so the failure list reflects
                # real replay errors only. Per no-silent-swallows:
                # every skip carries its reason and the count is in
                # the persisted summary.
                replay_skips.append({
                    "key": key,
                    "stage": "skipped",
                    "reason": (replay.replay_error or "")[:200],
                })
                continue

            if replay.replay_error:
                replay_failures.append({
                    "key": key,
                    "stage": "replay_error",
                    "error": replay.replay_error[:200],
                })
                continue

            agent_id_base = (
                replay.comparison.get("agent_id_base")
                or _agent_id_base_from_key(key)
                or "unknown"
            )
            agreement = replay.comparison.get("agreement_score")
            if isinstance(agreement, (int, float)):
                observations_by_agent[agent_id_base].append(float(agreement))

        # Aggregate + emit per agent.
        per_agent: list[dict[str, Any]] = []
        skipped_thin: list[dict[str, Any]] = []

        for agent_id_base in sorted(observations_by_agent.keys()):
            obs = observations_by_agent[agent_id_base]
            if len(obs) < MIN_OBSERVATIONS_FOR_CONCORDANCE:
                skipped_thin.append({
                    "agent_id_base": agent_id_base,
                    "n_observations": len(obs),
                })
                continue

            agg = _aggregate_group(obs)
            per_agent.append({
                "agent_id_base": agent_id_base,
                **agg,
            })

            if cw is not None:
                try:
                    _emit_concordance_metric(
                        cw,
                        namespace=namespace,
                        metric_name=metric_name,
                        agent_id_base=agent_id_base,
                        target_model=target_model,
                        mean_agreement=agg["mean"],
                        n_observations=agg["n"],
                        timestamp=end,
                    )
                except Exception as exc:  # noqa: BLE001 — observability of obs.
                    replay_failures.append({
                        "key": "(metric_emit)",
                        "stage": "metric_emit",
                        "agent_id_base": agent_id_base,
                        "error": str(exc),
                    })
                    logger.warning(
                        "[batch_replay] metric emission failed agent=%s err=%s",
                        agent_id_base, exc,
                    )

        target_summary = {
            "target_model": target_model,
            "n_artifacts_replayed": n_replayed,
            "agents_analyzed": len(per_agent),
            "per_agent": per_agent,
            "agents_skipped_thin_sample": skipped_thin,
            "replay_failures": replay_failures,
            "replay_skips": replay_skips,
            "cost": cost_total,
        }

        # Persist per-target-model summary.
        try:
            summary_key = _persist_batch_summary(
                s3,
                bucket=bucket,
                summary_prefix=summary_prefix,
                target_model=target_model,
                end_date=end,
                payload={
                    "schema_version": 1,
                    "window_start": window_start.isoformat(),
                    "window_end": end.isoformat(),
                    "agent_filter": agent_filter,
                    **target_summary,
                    "computed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            target_summary["summary_key"] = summary_key
        except Exception as exc:  # noqa: BLE001
            replay_failures.append({
                "key": "(persist_summary)",
                "stage": "persist",
                "error": str(exc),
            })
            logger.exception(
                "[batch_replay] persist_summary failed target=%s",
                target_model,
            )

        per_target_summary.append(target_summary)

        logger.info(
            "[batch_replay] target=%s replayed=%d agents_analyzed=%d "
            "thin=%d failures=%d skips=%d cost_in=%d cost_out=%d",
            target_model, n_replayed, len(per_agent),
            len(skipped_thin), len(replay_failures), len(replay_skips),
            cost_total["input_tokens"], cost_total["output_tokens"],
        )

    return {
        "window_start": window_start.isoformat(),
        "window_end": end.isoformat(),
        "agent_filter": agent_filter,
        "artifacts_discovered": len(keys),
        "per_target_model": per_target_summary,
    }
