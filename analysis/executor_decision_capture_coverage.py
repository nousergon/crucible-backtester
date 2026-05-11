"""
executor_decision_capture_coverage.py — Per-weekday executor decision-
capture coverage.

L2308 PR 5. Consumer-side counterpart of the producer-side wiring
shipped in L2308 PRs 1-4 (entry_triggers / position_sizer / risk_guard /
exit_rules captures emitted from the trading EC2's daemon + planner).
Closes the "Executor Components" insufficient-data gap in the Saturday
evaluator email by surfacing per-component artifact counts so the
operator can verify the producer-side wiring is firing.

This is the **substrate** PR: it produces a status dict reporting which
of the 4 canonical executor components emitted artifacts on the most
recent weekday's date partition. Grading dimensions that join artifacts
against trade outcomes (per-multiplier sizing decomposition,
counterfactual precision-of-refusal, exit-timing-vs-subsequent-N-day-
price) layer on in PR 5b once ≥2 weeks of artifacts have accumulated
and the substrate is proven.

Sibling module to ``decision_capture_coverage.py`` — same S3 listing
pattern + status-dict shape, but executor-specific cardinality:

  - Research side: Saturday SF, fixed 8 canonical agents
  - Executor side: weekday SF, 4 canonical components with variable
    cardinality (one artifact per fired event per ticker)

S3 contract:

  decision_artifacts/{YYYY}/{MM}/{DD}/executor:{component}/{run_id}.json

  component ∈ {entry_triggers, position_sizer, risk_guard, exit_rules}

Returns the standard backtester-evaluator status dict so
``CompletenessTracker.run_module`` handles it without bespoke wiring.

Plan doc: ``~/Development/alpha-engine-docs/private/executor-decision-capture-260511.md``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import boto3

logger = logging.getLogger(__name__)


# ── Canonical executor components ────────────────────────────────────────


EXECUTOR_COMPONENTS: tuple[str, ...] = (
    "executor:entry_triggers",
    "executor:position_sizer",
    "executor:risk_guard",
    "executor:exit_rules",
)
"""Per-weekday canonical executor capture set: 4 components from L2308
PRs 1-4. Coverage % is computed against this fixed set."""

N_CANONICAL = len(EXECUTOR_COMPONENTS)  # 4


_META_PREFIXES = (
    "_eval", "_eval_judge_only", "_replay", "_replay_summary",
    "_cost", "_cost_raw", "_analysis", "_diagnostics",
)

DEFAULT_BUCKET = "alpha-engine-research"
DEFAULT_CAPTURE_PREFIX = "decision_artifacts"


# ── S3 listing ───────────────────────────────────────────────────────────


def _list_executor_component_counts(
    s3: Any,
    *,
    bucket: str,
    capture_prefix: str,
    date: datetime,
) -> dict[str, int]:
    """List captures under ``{capture_prefix}/{Y}/{M}/{D}/executor:*/``
    and group by full agent_id (e.g. ``"executor:entry_triggers"``).

    Returns ``{agent_id: n_artifacts}``. Empty dict when no objects exist
    under the date prefix.
    """
    prefix = (
        f"{capture_prefix}/{date.strftime('%Y')}/"
        f"{date.strftime('%m')}/{date.strftime('%d')}/"
    )
    counts: dict[str, int] = {}
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
            # Only executor:* — research-side agents are owned by the
            # sibling decision_capture_coverage module.
            if not agent_id.startswith("executor:"):
                continue
            counts[agent_id] = counts.get(agent_id, 0) + 1
    return counts


# ── Per-date coverage ────────────────────────────────────────────────────


def _weekday_coverage_for(
    s3: Any,
    *,
    bucket: str,
    capture_prefix: str,
    weekday: datetime,
) -> dict[str, Any]:
    """Compute coverage for one weekday's date partition.

    Returns dict with: date, n_canonical_present, n_canonical_expected,
    coverage_pct, per_component, total_artifacts, uncategorized_executor_components.
    Empty listing → coverage_pct=0, n_canonical_present=0.
    """
    counts = _list_executor_component_counts(
        s3, bucket=bucket, capture_prefix=capture_prefix, date=weekday,
    )

    per_component: dict[str, dict[str, Any]] = {}
    n_present = 0
    for component in EXECUTOR_COMPONENTS:
        n = counts.get(component, 0)
        present = n >= 1
        per_component[component] = {"present": present, "n_artifacts": n}
        if present:
            n_present += 1

    total = sum(counts.values())
    # Anything else under executor:* that isn't in CANONICAL — flag for
    # visibility (a new producer rolling out, or a typo).
    uncategorized = sorted(
        agent for agent in counts if agent not in EXECUTOR_COMPONENTS
    )

    coverage_pct = (n_present / N_CANONICAL) * 100.0

    return {
        "date": weekday.strftime("%Y-%m-%d"),
        "n_canonical_present": n_present,
        "n_canonical_expected": N_CANONICAL,
        "coverage_pct": round(coverage_pct, 2),
        "per_component": per_component,
        "total_artifacts": total,
        "uncategorized_executor_components": uncategorized,
    }


# ── Most-recent-weekday resolution ───────────────────────────────────────


def _most_recent_weekday_with_executor_captures(
    s3: Any,
    *,
    bucket: str,
    capture_prefix: str,
    end_date: datetime,
    max_lookback_days: int = 7,
) -> datetime | None:
    """Walk back up to ``max_lookback_days`` from ``end_date`` and return
    the first weekday whose date partition has any executor:* captures.

    Unlike the research-side module (Saturday-only), executor artifacts
    flow on every weekday SF run (Mon-Fri). Returns None if no weekday
    in the window has captures — typical when the
    ``ALPHA_ENGINE_DECISION_CAPTURE_ENABLED`` env flag hasn't been
    enabled yet on the trading EC2 (default-off per producer convention).
    """
    for offset in range(max_lookback_days + 1):
        d = end_date - timedelta(days=offset)
        if d.weekday() >= 5:  # Saturday or Sunday
            continue
        counts = _list_executor_component_counts(
            s3, bucket=bucket, capture_prefix=capture_prefix, date=d,
        )
        if counts:
            return d
    return None


# ── Public API ───────────────────────────────────────────────────────────


def compute_executor_decision_capture_coverage(
    *,
    bucket: str = DEFAULT_BUCKET,
    capture_prefix: str = DEFAULT_CAPTURE_PREFIX,
    run_date: str | None = None,
    s3_client: Any | None = None,
    max_lookback_days: int = 7,
) -> dict[str, Any]:
    """Compute executor decision-capture coverage for the most recent
    weekday SF run within ``max_lookback_days`` of ``run_date``.

    Status semantics:

    - ``"ok"``: a weekday was found with ≥1 executor:* artifact.
      ``coverage_pct`` reports presence across the 4 canonical
      components.
    - ``"insufficient_data"``: no weekday in the lookback window
      produced any executor captures. Typical interpretation: the
      ``ALPHA_ENGINE_DECISION_CAPTURE_ENABLED`` env flag isn't enabled
      yet on the trading EC2, OR the trading instance didn't run on
      any weekday in the window (e.g. extended holiday period).

    Designed to be wired into ``CompletenessTracker.run_module`` —
    surfaces as a coverage row in the evaluator email alongside the
    research-side ``decision_capture_coverage``.
    """
    s3 = s3_client or boto3.client("s3")

    if run_date is not None:
        try:
            end_date = datetime.strptime(run_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc,
            )
        except ValueError:
            logger.warning(
                "Invalid run_date=%r — falling back to today (UTC)",
                run_date,
            )
            end_date = datetime.now(timezone.utc)
    else:
        end_date = datetime.now(timezone.utc)

    most_recent = _most_recent_weekday_with_executor_captures(
        s3, bucket=bucket, capture_prefix=capture_prefix,
        end_date=end_date, max_lookback_days=max_lookback_days,
    )

    if most_recent is None:
        return {
            "status": "insufficient_data",
            "reason": (
                f"no executor:* captures found in the trailing "
                f"{max_lookback_days} days; check that "
                "ALPHA_ENGINE_DECISION_CAPTURE_ENABLED=true is set on "
                "the trading EC2 .alpha-engine.env"
            ),
            "lookback_days": max_lookback_days,
            "end_date": end_date.strftime("%Y-%m-%d"),
        }

    coverage = _weekday_coverage_for(
        s3, bucket=bucket, capture_prefix=capture_prefix,
        weekday=most_recent,
    )

    return {
        "status": "ok",
        **coverage,
    }


__all__ = [
    "EXECUTOR_COMPONENTS",
    "N_CANONICAL",
    "compute_executor_decision_capture_coverage",
]
