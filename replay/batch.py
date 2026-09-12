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

from replay.runner import replay_artifact, resolve_target_spec

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

# Time held back from the replay loop for the aggregate, the CloudWatch emits
# and the summary PUT (config#6920). Without a reserve the loop can consume the
# whole budget and the run ends with nothing persisted — which is what made a
# 15-minute paid run indistinguishable from one that never happened.
CONCORDANCE_WRITE_RESERVE_S = 45.0

# Floor for the "can another item fit?" estimate before any item has completed.
# Measured per-item latencies on 2026-08-11 ranged 6s-137s; the handler's cap
# was sized on an assumed 3-5s, which is why it could never bind.
CONCORDANCE_MIN_ITEM_ESTIMATE_S = 30.0

# Bounds the backward S3 scan that names, in the empty-corpus alert, the last
# dated prefix that actually matched `agent_filter` — a diagnostic aid, never
# on the hot path (only runs when the configured window found zero
# candidates). ~13 months caps the worst case for a filter that has been
# stale a long time rather than scanning to the corpus's epoch.
EMPTY_CORPUS_ALERT_LOOKBACK_DAYS = 400


def _next_item_affordable(remaining_s, latencies_ms: list[int]) -> tuple[bool, float]:
    """Can another replay finish before the deadline?

    Estimates from the observed p90 of this run's own item latencies — the
    workload measures itself rather than trusting a literal. Returns
    (affordable, needed_seconds); always affordable when there is no deadline.
    """
    if remaining_s is None:
        return True, 0.0
    if latencies_ms:
        ordered = sorted(latencies_ms)
        p90 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))] / 1000.0
        estimate = max(CONCORDANCE_MIN_ITEM_ESTIMATE_S, p90)
    else:
        estimate = CONCORDANCE_MIN_ITEM_ESTIMATE_S
    needed = estimate + CONCORDANCE_WRITE_RESERVE_S
    return remaining_s() >= needed, needed
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


_META_SUBTREE_SKIP = (
    "eval", "eval_judge_only", "analysis", "cost", "cost_raw",
    "replay", "replay_summary",
)


def _day_matches_agent_filter(
    s3: Any, *, bucket: str, capture_prefix: str, day: datetime,
    agent_filter: list[str] | None,
) -> bool:
    """True if the dated prefix for `day` holds a production capture that
    matches `agent_filter`. Shared by the window lister's per-day skip
    logic and the empty-corpus lookback below, so the two never drift on
    what counts as a "production capture"."""
    prefix = (
        f"{capture_prefix}/{day.strftime('%Y')}/"
        f"{day.strftime('%m')}/{day.strftime('%d')}/"
    )
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            if any(f"/_{p}/" in key for p in _META_SUBTREE_SKIP):
                continue
            if agent_filter:
                base_id = _agent_id_base_from_key(key)
                if base_id is None or base_id not in agent_filter:
                    continue
            return True
    return False


def _find_last_matching_dated_prefix(
    s3: Any,
    *,
    bucket: str,
    capture_prefix: str,
    before_date: datetime,
    agent_filter: list[str] | None,
    max_lookback_days: int = EMPTY_CORPUS_ALERT_LOOKBACK_DAYS,
) -> Optional[str]:
    """Scan backward day-by-day from `before_date` for the most recent
    dated prefix under `capture_prefix` holding an artifact matching
    `agent_filter`.

    Diagnostic only, and called ONLY when the configured trailing window
    found zero candidates (`compute_and_emit_concordance`) — this is what
    lets the resulting alert name *which* filter went stale and *when* it
    last matched, rather than just asserting that it did. Bounded by
    `max_lookback_days` so a permanently-empty corpus costs a fixed number
    of List calls, not an unbounded scan back to the bucket's epoch.
    Returns ``None`` (not raise) on any listing failure or exhausted
    lookback — this must never be the reason the run fails, only the
    reason the alert is a little less specific.
    """
    for day_offset in range(max_lookback_days):
        day = before_date - timedelta(days=day_offset)
        try:
            if _day_matches_agent_filter(
                s3, bucket=bucket, capture_prefix=capture_prefix,
                day=day, agent_filter=agent_filter,
            ):
                return (
                    f"{capture_prefix}/{day.strftime('%Y')}/"
                    f"{day.strftime('%m')}/{day.strftime('%d')}/"
                )
        except Exception as exc:  # noqa: BLE001 — diagnostic scan, never fatal
            logger.warning(
                "[batch_replay] empty-corpus lookback: listing failed for "
                "day=%s: %s", day.date().isoformat(), exc,
            )
            return None
    return None


def _publish_empty_corpus_alert(
    *, agent_filter: list[str] | None, window_days: int,
    last_matching_prefix: Optional[str],
) -> None:
    """Ops alert: the configured trailing window matched zero candidate
    artifacts this run.

    Severity WARNING, not ERROR — a zero-candidate corpus is now a
    correctly-observed, non-failing run (see
    `_emit_zero_call_cost_record` — the stage still emits a cost record
    saying so, closing the fan-in coverage gap this exists alongside).
    But an `agent_filter` that has gone stale is a finding an operator
    should see once, not a fact buried in an INFO log line nobody reads
    until a downstream check trips on it two stages later — which is
    exactly the 2026-09-12 incident this guards against
    (alpha-engine-config-I7183).
    """
    try:
        from ops_alerts import publish_ops_alert
    except Exception as exc:  # noqa: BLE001 — alerting must not break the run
        logger.warning(
            "[batch_replay] could not import ops_alerts for empty-corpus "
            "alert: %s", exc,
        )
        return
    last = last_matching_prefix or (
        f"(none found in the trailing {EMPTY_CORPUS_ALERT_LOOKBACK_DAYS}-day "
        f"lookback)"
    )
    message = (
        "ReplayConcordance: zero-candidate corpus\n"
        f"agent_filter={agent_filter} matched no decision_artifacts in the "
        f"trailing {window_days}-day window. Last dated prefix with a "
        f"match: {last}. The replay stage now legitimately makes zero "
        "model calls every run until agent_filter is repointed at the "
        "current producer or the stage is retired."
    )
    try:
        publish_ops_alert(
            message,
            severity="warning",
            source="crucible-backtester/replay/batch.py::compute_and_emit_concordance",
            dedup_key="replay_concordance_empty_corpus",
            dedup_window_min=10080,  # once per weekly cadence
        )
    except Exception as exc:  # noqa: BLE001 — alerting must not break the run
        logger.warning(
            "[batch_replay] empty-corpus ops alert publish failed: %s", exc,
        )


def _emit_zero_call_cost_record(target_model: str, target_spec: Any) -> None:
    """Hand the cost sink one record saying this producer legitimately made
    zero calls this run.

    ``AggregateCosts``' fan-in check derives its observed-producers set
    from S3 KEY NAMES under the ``_cost_raw/`` prefix, never from row
    content (``crucible-research/scripts/cost_coverage.py::
    observed_producers``), so a record here — priced at zero, marked
    ``producer_ran_no_calls`` — makes ``replay-concordance`` observed the
    same way a real call would, without pretending any spend occurred.

    Root cause this exists for: the six-team ``agent_filter`` this module
    defaults to (sector_quant, sector_qual, sector_peer_review,
    macro_economist, ic_cio, thesis_update) matched its last artifact
    2026-07-11 (config-I1580 retired those agents 2026-07-20); every
    later run legitimately makes zero LLM calls, the sink never receives
    a real cost row, and the fan-in check read a correctly-silent stage
    identically to one that swallowed its records — the FAILED weekly SF
    of 2026-09-12 (alpha-engine-config-I7183).

    Built by hand rather than through ``krepis.cost.record_llm_call`` —
    that function prices an actual provider response object, and there is
    no call here to price. The record still carries the columns a real
    row carries (zeroed) plus an explicit ``event`` marker, so a reader
    can tell "ran, priced at zero" apart from "ran, lost its records" (the
    I7179/I7423 shape) without inferring it from an absent token count.

    Never raises — a telemetry fault must not take down the work it
    measures, the same contract every other cost-sink call site in krepis
    keeps.
    """
    try:
        from krepis.cost import COST_RECORD_SCHEMA_VERSION
        from krepis.cost_sink import default_sink_from_env

        sink = default_sink_from_env()
        if sink is None:
            return  # no sink configured (local/test/dev run) — nothing to emit
        provider = getattr(target_spec, "provider", None) or "unknown"
        model_name = getattr(target_spec, "model", None) or target_model
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "callsite_id": "replay-concordance",
            "agent_id": "replay-concordance",
            "provider": provider,
            "model": model_name,
            "model_name": model_name,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_create_tokens": 0,
            "cache_create_1h_tokens": 0,
            "cost_usd": 0.0,
            "cost_source": "producer_ran_no_calls",
            "event": "producer_ran_no_calls",
            "n_calls": 0,
            "schema_version": COST_RECORD_SCHEMA_VERSION,
        }
        sink(record)
        logger.info(
            "[batch_replay] emitted zero-call cost record target=%s "
            "provider=%s (producer_ran_no_calls)", target_model, provider,
        )
    except Exception as exc:  # noqa: BLE001 — telemetry must not break the run
        logger.error(
            "[batch_replay] failed to emit zero-call cost record for "
            "target=%s: %s", target_model, exc,
        )


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
    emit_metrics: bool = True,
    persist_per_replay: bool = False,
    dry_run: bool = False,
    remaining_s=None,
) -> dict[str, Any]:
    """Replay the trailing-window corpus under each target_model,
    aggregate concordance per (agent_id_base, target_model), emit
    CloudWatch metrics, and persist a per-target-model batch summary.

    Args:
        target_models: list of REGISTRY ENTRY IDS to replay against
            (alpha-engine-config-I7878, 2026-08-20 — was an OpenRouter
            slug pre-migration). Common shape: ``["deepseek-v4-flash"]``
            — the default ReplayConcordance dispatches (see
            ``lambda_concordance/handler.py``). Each is resolved through
            the krepis router ONCE, before any artifact is replayed; a
            router that will not admit this process raises here rather
            than producing one identical failure per artifact, and never
            falls back to a direct provider endpoint.
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
        remaining_s: zero-arg callable returning the seconds left before the
            caller's own deadline (in Lambda, ``context.get_remaining_time_in
            _millis() / 1000``). When supplied, the replay loop stops early
            rather than being killed mid-run, and the summary carries
            ``complete=False`` plus the skipped count (config#6920). ``None``
            (a spot run, a CLI run) means no deadline and no early stop.

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

    if not dry_run and len(keys) == 0:
        # Empty corpus is a finding about the filter, not a quiet cycle —
        # log + alert once here rather than let it surface only when the
        # fan-in check trips two stages later (alpha-engine-config-I7183).
        logger.error(
            "[batch_replay] zero-candidate corpus: agent_filter=%s matched "
            "no artifact in the trailing %d-day window ending %s. This "
            "stage will legitimately make zero model calls this run.",
            agent_filter, window_days, end.isoformat(),
        )
        try:
            last_prefix = _find_last_matching_dated_prefix(
                s3, bucket=bucket, capture_prefix=capture_prefix,
                before_date=window_start, agent_filter=agent_filter,
            )
        except Exception as exc:  # noqa: BLE001 — diagnostic only
            logger.warning(
                "[batch_replay] last-matching-prefix lookback raised: %s", exc,
            )
            last_prefix = None
        _publish_empty_corpus_alert(
            agent_filter=agent_filter, window_days=window_days,
            last_matching_prefix=last_prefix,
        )

    if dry_run:
        # A dry run RESOLVES every target, and never raises doing it.
        #
        # This is the deploy canary's only reachable path. `deploy_concordance.
        # sh` invokes `{"dry_run": true, "window_days": 14}` against the newly
        # published version and promotes the `live` alias only if the status
        # comes back OK/PARTIAL — so whatever the dry run does NOT exercise is
        # not covered by the one gate standing between a merge and the weekly
        # SF. Before alpha-engine-config-I7878 that was fine: the dry run's
        # only untested surface was the model call itself. It is not fine now.
        # The target is a REGISTRY ID resolved through the router, so a
        # mistyped id, a retired registry entry, an unreachable edge or a
        # missing per-consumer credential are all deploy-time facts that the
        # dry run was silently stepping over — and the first thing to notice
        # would have been Saturday's weekly run, with the whole stage lost.
        #
        # Resolution costs no tokens: a registry read plus one health probe.
        #
        # It records rather than raises, because the dry run has a SECOND
        # consumer with the opposite need — an operator listing the corpus
        # while diagnosing a router outage. Both are served: `would_replay` is
        # always present, and `target_resolution` says, per target, whether the
        # routed path is actually usable. The handler turns any failure here
        # into a non-OK status, which is what fails the canary and leaves the
        # `live` alias on the prior good version.
        target_resolution: list[dict[str, Any]] = []
        for target_model in target_models:
            try:
                _, route = resolve_target_spec(target_model)
            except Exception as exc:  # noqa: BLE001 — recorded, see above
                logger.error(
                    "[batch_replay] dry run: target=%s did NOT resolve (%s: %s)",
                    target_model, type(exc).__name__, exc,
                )
                target_resolution.append({
                    "target_model": target_model,
                    "resolved": False,
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                })
                continue
            logger.info(
                "[batch_replay] dry run: target=%s resolved model=%s route=%s "
                "exec_context=%s",
                target_model, route.get("deployment_id"), route.get("route"),
                route.get("exec_context"),
            )
            target_resolution.append({
                "target_model": target_model,
                "resolved": True,
                "deployment_id": route.get("deployment_id"),
                "route": route.get("route"),
                "exec_context": route.get("exec_context"),
            })

        return {
            "dry_run": True,
            "window_start": window_start.isoformat(),
            "window_end": end.isoformat(),
            "target_models": target_models,
            "agent_filter": agent_filter,
            "would_replay": len(keys),
            "would_replay_keys": keys[:50],  # trim for log/return ergonomics
            "target_resolution": target_resolution,
        }

    # Deferred like every other krepis import in this package — the Lambda
    # image imports this module during cold start before secrets load.
    from krepis.router import route_is_degraded

    # Outer summary collects per-target-model results.
    per_target_summary: list[dict[str, Any]] = []

    for target_model in target_models:
        # Resolve the target ONCE, before touching the corpus.
        #
        # Fail loud and fail early (model-router-policy R20, and the fleet's
        # no-silent-swallows rule). A router edge that will not admit this
        # process, or a target_model that is not a registry entry id, is a
        # PRECONDITION failure: it is identical for every artifact, it is not
        # the divergence this module measures, and letting it fall into the
        # per-item handler below would report a router outage as N
        # low-concordance observations and a PARTIAL status nobody reads.
        # Raising surfaces it as a failed ReplayConcordance stage on the
        # weekly SF, which is the honest shape.
        #
        # There is deliberately NO direct-provider fallback here. A pinned
        # model has no registry-declared substitute, so the only reachable
        # alternative would be a DLP-unscanned direct call — the linkage
        # Brian's 2026-08-03 ruling removed (alpha-engine-config-I6367).
        target_spec, target_route = resolve_target_spec(target_model)
        logger.info(
            "[batch_replay] target=%s resolved model=%s route=%s "
            "exec_context=%s degraded=%s",
            target_model, target_route.get("deployment_id"),
            target_route.get("route"), target_route.get("exec_context"),
            route_is_degraded(target_route),
        )

        if len(keys) == 0:
            # This target makes zero calls this run (see the empty-corpus
            # log+alert above). Emit the zero-call cost record now, per
            # target, so the fan-in check observes `replay-concordance`
            # even though the loop below is a no-op for an empty `keys`.
            _emit_zero_call_cost_record(target_model, target_spec)

        # Group observations by agent_id_base.
        observations_by_agent: dict[str, list[float]] = defaultdict(list)
        cost_total = {"input_tokens": 0, "output_tokens": 0}
        # config#3006 — jurisdiction/compliance observation: which
        # upstream OpenRouter backends actually served this batch's
        # requests. Empty when every replay pre-dates the krepis
        # served_provider field (v0.18.0+) or the provider didn't report
        # one — absence here is informational, not a failure signal.
        served_providers_seen: set[str] = set()
        replay_failures: list[dict[str, str]] = []
        replay_skips: list[dict[str, str]] = []
        n_replayed = 0

        item_latencies_ms: list[int] = []
        budget_stopped = False
        n_skipped_for_budget = 0

        for index, key in enumerate(keys):
            # config#6920: check BEFORE starting an item. Being killed
            # mid-replay discards the whole run's aggregate — nothing is
            # persisted until the loop finishes — so stopping early with a
            # partial, honestly-labelled result strictly dominates.
            affordable, needed = _next_item_affordable(remaining_s, item_latencies_ms)
            if not affordable:
                budget_stopped = True
                n_skipped_for_budget = len(keys) - index
                logger.warning(
                    "[batch_replay] stopping early on budget: %d of %d artifacts "
                    "not replayed for target=%s (needed ~%.0fs, %.0fs left). The "
                    "summary is persisted and marked incomplete.",
                    n_skipped_for_budget, len(keys), target_model, needed,
                    remaining_s(),
                )
                break
            try:
                replay = replay_artifact(
                    artifact_key=key,
                    target_model=target_model,
                    bucket=bucket,
                    replay_prefix=replay_prefix,
                    s3_client=s3,
                    client_factory=client_factory,
                    model_spec=target_spec,
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
            if isinstance(replay.replay_latency_ms, (int, float)):
                item_latencies_ms.append(int(replay.replay_latency_ms))

            # Roll up token cost (best-effort).
            for k in ("input_tokens", "output_tokens"):
                cost_total[k] += int(replay.replay_cost.get(k, 0) or 0)

            served_provider = replay.replay_cost.get("served_provider")
            if served_provider:
                served_providers_seen.add(served_provider)

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
            # config#6920: a consumer must be able to tell a full sweep from a
            # truncated one. `complete` is False whenever the loop stopped on
            # budget, so a partial concordance can never be read as a full one.
            "complete": not budget_stopped,
            "budget_stopped": budget_stopped,
            "n_artifacts_candidate": len(keys),
            "n_artifacts_skipped_for_budget": n_skipped_for_budget,
            "n_artifacts_replayed": n_replayed,
            "agents_analyzed": len(per_agent),
            "per_agent": per_agent,
            "agents_skipped_thin_sample": skipped_thin,
            "replay_failures": replay_failures,
            "replay_skips": replay_skips,
            "cost": cost_total,
            "served_providers_seen": sorted(served_providers_seen),
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
