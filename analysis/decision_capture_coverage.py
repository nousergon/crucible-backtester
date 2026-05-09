"""
decision_capture_coverage.py — Per-Saturday-SF agent capture coverage.

Phase 2 transparency-inventory: closes the *agent decisions* row in the
gate checklist. The inventory framing requires:

> Per-call artifact: prompt version + decision-capture record + cost
> telemetry + judge rubric scores; coverage ≥ 99%

Today the artifacts EXIST under
``decision_artifacts/{Y}/{M}/{D}/{agent_id}/{run_id}.json`` for every
Saturday SF run, but the *coverage %* metric — what fraction of the
canonical agent set produced ≥1 capture — wasn't a published number.
Without it, "every agent decision is captured" is an assertion, not a
queryable observation, and the inventory's 99%-coverage gate is
unverifiable.

This module reads the same S3 artifact tree that the replay batch
already iterates (see ``replay/batch.py:_list_artifact_keys_in_window``)
and groups by ``agent_id`` — the directory name immediately under the
date partition. Eight canonical agents are expected per Saturday SF
(1 macro_economist + 1 ic_cio + 6 sector_team). thesis_update is
captured per-held-stock and reported as a count alongside, not in the
coverage denominator (the held-stock cardinality varies week-to-week
with portfolio composition).

Sector-team virtualization: the research pipeline writes one artifact
per sub-stage (``sector_quant:{sector}``, ``sector_qual:{sector}``,
``sector_peer_review:{sector}``) rather than one composite
``sector_team:{sector}`` artifact. The canonical 8-agent denominator
stays semantically meaningful (one decision per team), but the
"present" check rolls up the 3 sub-stage artifacts: a sector counts as
captured iff all 3 sub-stages have ≥1 artifact. Sub-stage IDs are
classified, not surfaced as uncategorized.

S3 contract:

  decision_artifacts/{YYYY}/{MM}/{DD}/{agent_id}/{run_id}.json

  agent_id ∈ {macro_economist, ic_cio,
              sector_quant:<sector>, sector_qual:<sector>,
              sector_peer_review:<sector>,
              thesis_update:<sector>:<ticker>}

  Meta-prefixes (excluded from coverage): _eval/, _eval_judge_only/,
              _replay/, _replay_summary/, _cost/, _cost_raw/,
              _analysis/, _diagnostics/

Returns the standard backtester-evaluator status dict so the existing
``CompletenessTracker.run_module`` pattern handles it without bespoke
wiring (mirrors ``analysis/macro_eval.py`` shape).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import boto3

logger = logging.getLogger(__name__)


# ── Canonical agents ─────────────────────────────────────────────────────────

CANONICAL_SECTORS = (
    "consumer", "defensives", "financials",
    "healthcare", "industrials", "technology",
)

CANONICAL_AGENTS: tuple[str, ...] = (
    "macro_economist",
    "ic_cio",
    *(f"sector_team:{s}" for s in CANONICAL_SECTORS),
)
"""Per-Saturday-SF canonical agent set: 1 macro + 1 ic_cio + 6 sector_team.
Coverage % is computed against this fixed set. thesis_update varies with
portfolio composition and is reported separately."""

N_CANONICAL = len(CANONICAL_AGENTS)  # 8

# The research pipeline writes 3 artifacts per sector team — one per
# sub-stage. ``sector_team:{sector}`` is treated as a virtual aggregate
# in this module: present iff every sub-stage emitted ≥1 artifact.
SECTOR_SUB_STAGES: tuple[str, ...] = (
    "sector_quant", "sector_qual", "sector_peer_review",
)
_SECTOR_SUB_STAGE_PREFIXES: tuple[str, ...] = tuple(
    f"{stage}:" for stage in SECTOR_SUB_STAGES
)

# Meta-prefixes under decision_artifacts/{Y}/{M}/{D}/ that aren't agent
# captures — exclude when listing for coverage.
_META_PREFIXES = (
    "_eval", "_eval_judge_only", "_replay", "_replay_summary",
    "_cost", "_cost_raw", "_analysis", "_diagnostics",
)

DEFAULT_BUCKET = "alpha-engine-research"
DEFAULT_CAPTURE_PREFIX = "decision_artifacts"


# ── S3 listing ───────────────────────────────────────────────────────────────


def _list_agent_artifact_counts(
    s3: Any,
    *,
    bucket: str,
    capture_prefix: str,
    date: datetime,
) -> dict[str, int]:
    """List captures under ``{capture_prefix}/{Y}/{M}/{D}/`` and group
    by agent_id (the directory immediately below the date partition).

    Returns ``{agent_id: n_artifacts}``. Empty dict when no objects exist
    under the date prefix (typical for non-SF days).
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
            # Skip meta-prefixes — they aren't agent decisions.
            if any(f"/{p}/" in key for p in _META_PREFIXES):
                continue
            # Extract agent_id = the directory between the date partition
            # and the artifact filename.
            relative = key[len(prefix):]
            parts = relative.split("/")
            if len(parts) < 2:
                continue  # malformed — top-level json file under the date
            agent_id = parts[0]
            counts[agent_id] = counts.get(agent_id, 0) + 1
    return counts


# ── Per-date coverage ────────────────────────────────────────────────────────


def _saturday_coverage_for(
    s3: Any,
    *,
    bucket: str,
    capture_prefix: str,
    saturday: datetime,
) -> dict[str, Any]:
    """Compute coverage for one Saturday SF date.

    Returns dict with: date, n_canonical_present, n_canonical_expected,
    coverage_pct, per_agent, thesis_update_count, uncategorized_agents.
    Empty agent listing → coverage_pct=0, n_canonical_present=0.
    """
    counts = _list_agent_artifact_counts(
        s3, bucket=bucket, capture_prefix=capture_prefix, date=saturday,
    )

    per_agent: dict[str, dict[str, Any]] = {}
    n_present = 0
    for agent in CANONICAL_AGENTS:
        if agent.startswith("sector_team:"):
            sector = agent.split(":", 1)[1]
            sub_stage_counts = {
                stage: counts.get(f"{stage}:{sector}", 0)
                for stage in SECTOR_SUB_STAGES
            }
            # Virtual present: every sub-stage emitted ≥1 artifact.
            present = all(c >= 1 for c in sub_stage_counts.values())
            n = sum(sub_stage_counts.values())
            per_agent[agent] = {
                "present": present,
                "n_artifacts": n,
                "sub_stages": sub_stage_counts,
            }
        else:
            n = counts.get(agent, 0)
            present = n >= 1
            per_agent[agent] = {"present": present, "n_artifacts": n}
        if present:
            n_present += 1

    # thesis_update — variable cardinality, count separately.
    thesis_count = sum(
        n for agent, n in counts.items() if agent.startswith("thesis_update:")
    )

    # Anything else is uncategorized — flag for visibility (a new agent
    # type rolling out, or a typo in the agent_id). Sector sub-stage IDs
    # roll up into the virtual sector_team:{sector} entry, so they
    # don't count as uncategorized.
    uncategorized = sorted(
        agent for agent in counts
        if agent not in CANONICAL_AGENTS
        and not agent.startswith("thesis_update:")
        and not agent.startswith(_SECTOR_SUB_STAGE_PREFIXES)
    )

    coverage_pct = (n_present / N_CANONICAL) * 100.0

    return {
        "date": saturday.strftime("%Y-%m-%d"),
        "n_canonical_present": n_present,
        "n_canonical_expected": N_CANONICAL,
        "coverage_pct": round(coverage_pct, 2),
        "per_agent": per_agent,
        "thesis_update_count": thesis_count,
        "uncategorized_agents": uncategorized,
    }


# ── Most-recent-Saturday resolution ──────────────────────────────────────────


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
    no Saturday in the window has captures.

    Saturday-only because the SF runs Saturday cron(0 9 ? * SAT *); other
    days don't produce decision artifacts."""
    for offset in range(max_lookback_days + 1):
        d = end_date - timedelta(days=offset)
        if d.weekday() != 5:  # not Saturday
            continue
        counts = _list_agent_artifact_counts(
            s3, bucket=bucket, capture_prefix=capture_prefix, date=d,
        )
        if counts:
            return d
    return None


# ── Public entry point ───────────────────────────────────────────────────────


def compute_decision_capture_coverage(
    bucket: str = DEFAULT_BUCKET,
    run_date: str | None = None,
    lookback_weeks: int = 8,
    capture_prefix: str = DEFAULT_CAPTURE_PREFIX,
    s3_client: Any = None,
) -> dict[str, Any]:
    """Compute decision-capture coverage % for the most recent Saturday
    SF run on or before ``run_date``, plus an N-week rolling coverage
    average.

    Args:
        bucket: S3 bucket holding ``decision_artifacts/...``.
        run_date: ISO date string. Defaults to today (UTC). The function
            walks back up to 7 days from this date to find the most
            recent Saturday with captures.
        lookback_weeks: how many trailing Saturdays (inclusive of the
            most-recent one) to average for the rolling coverage trend.
            Default 8 mirrors the rolling-mean Lambda window.
        capture_prefix: S3 key prefix under ``bucket`` for the artifact
            tree. Override only for tests.
        s3_client: injected boto3 client (tests). None → ``boto3.client("s3")``.

    Returns:
        dict with keys
          status: "ok" | "no_recent_sf_run" | "error"
          run_date: the input run_date (echoed)
          most_recent_sf_date: ISO date of the Saturday graded
          coverage_pct: most-recent-Saturday coverage [0, 100]
          n_canonical_present, n_canonical_expected, per_agent,
            thesis_update_count, uncategorized_agents:
            (see _saturday_coverage_for)
          rolling: {n_saturdays_with_data, coverage_pct_mean,
                    coverage_pct_min, coverage_pct_max,
                    per_saturday: list[saturday-coverage dicts]}
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
        logger.exception("decision_capture_coverage: S3 listing failed")
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

    most_recent_summary = _saturday_coverage_for(
        s3, bucket=bucket, capture_prefix=capture_prefix,
        saturday=most_recent,
    )

    # Trailing window: walk back week-by-week from the most-recent Saturday.
    per_saturday: list[dict[str, Any]] = [most_recent_summary]
    for week in range(1, lookback_weeks):
        sat = most_recent - timedelta(days=7 * week)
        try:
            summary = _saturday_coverage_for(
                s3, bucket=bucket, capture_prefix=capture_prefix, saturday=sat,
            )
        except Exception as e:
            logger.warning(
                "decision_capture_coverage: per-Saturday lookup failed for %s: %s",
                sat.strftime("%Y-%m-%d"), e,
            )
            continue
        # Only include Saturdays that had any captures — empty Saturdays
        # mean the SF didn't run / failed before capture, not 0% coverage.
        # ``n_canonical_present`` only fires for fully-virtualized sector
        # teams (all 3 sub-stages), so partial-success Saturdays surface
        # via the per_agent n_artifacts check.
        any_per_agent_artifacts = any(
            entry.get("n_artifacts", 0) > 0
            for entry in summary["per_agent"].values()
        )
        if (
            any_per_agent_artifacts
            or summary["thesis_update_count"] > 0
            or summary["uncategorized_agents"]
        ):
            per_saturday.append(summary)

    coverages = [s["coverage_pct"] for s in per_saturday]
    rolling = {
        "n_saturdays_with_data": len(per_saturday),
        "coverage_pct_mean": round(sum(coverages) / len(coverages), 2)
            if coverages else 0.0,
        "coverage_pct_min": round(min(coverages), 2) if coverages else 0.0,
        "coverage_pct_max": round(max(coverages), 2) if coverages else 0.0,
        "per_saturday": per_saturday,
    }

    return {
        "status": "ok",
        "run_date": run_date,
        "most_recent_sf_date": most_recent.strftime("%Y-%m-%d"),
        "coverage_pct": most_recent_summary["coverage_pct"],
        "n_canonical_present": most_recent_summary["n_canonical_present"],
        "n_canonical_expected": most_recent_summary["n_canonical_expected"],
        "per_agent": most_recent_summary["per_agent"],
        "thesis_update_count": most_recent_summary["thesis_update_count"],
        "uncategorized_agents": most_recent_summary["uncategorized_agents"],
        "rolling": rolling,
    }
