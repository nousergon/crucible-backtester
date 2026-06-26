"""
analysis/agent_justification.py — summarize the eval-judge + agent-justification
triple from S3 for the evaluator email.

Pre-2026-05-07 reorder, the judge chain (EvalJudge / RationaleClustering /
ReplayConcordance / Counterfactual) ran AFTER Evaluator in the Saturday SF —
so by the time `evaluate.py` generated the weekly email, those Lambdas
hadn't written to S3 yet and their results were silently absent from the
operator's primary review surface. The reorder moves the chain upstream of
PredictorTraining so this loader can read fresh outputs each week.

Each summarizer is defensive — if its S3 prefix is missing or empty (e.g.
the corresponding Lambda hasn't run yet, or the run is mid-week without
fresh research), the loader returns a status dict rather than raising.
The renderer in reporter.py shows the section unconditionally so absent
data is visible (silent omission would mask a Lambda failure).

S3 layout this module reads (all rooted at
``s3://{bucket}/decision_artifacts/``):

  ``_eval/{date}/{agent_id}/{date}.{model}.json``   — judge rubric scores
  ``_analysis/{agent_id_base}/{YYYY-WWW}.json``     — clustering aggregates
  ``_counterfactual/{agent_id_base}/{run_id}.json`` — DT rule fits (canonical
      eval_artifacts layout post config#792; tolerant of legacy
      ``{YYYY-WWW}.json`` weekly files + the per-agent ``latest.json`` sidecar)
  ``_replay_summary/{run_id}_{target_model}.json``  — concordance summaries
      (canonical flat layout post config#792; tolerant of legacy
      ``{date}/{target_model}.json`` partitions + the ``latest.json`` sidecar)
"""

from __future__ import annotations

import json
import logging
from datetime import date as date_type, timedelta
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "alpha-engine-research"
_LOOKBACK_DAYS = 14


# ── helpers ──────────────────────────────────────────────────────────────


def _list_dirs(s3_client: Any, bucket: str, prefix: str) -> list[str]:
    """List 'subdirectories' (CommonPrefixes) under prefix. Each entry is
    the trailing path component (no leading/trailing slash)."""
    try:
        resp = s3_client.list_objects_v2(
            Bucket=bucket, Prefix=prefix, Delimiter="/",
        )
    except ClientError as exc:
        logger.warning("[agent_justification] list failed for %s: %s", prefix, exc)
        return []
    out = []
    for cp in resp.get("CommonPrefixes") or []:
        p = cp.get("Prefix", "").rstrip("/")
        out.append(p.rsplit("/", 1)[-1])
    return out


def _find_most_recent_date_subdir(
    s3_client: Any, *, bucket: str, prefix: str, run_date: str,
) -> Optional[str]:
    """Find the most recent ISO-date subdir under prefix that is <= run_date
    and within _LOOKBACK_DAYS. Returns the date string or None.
    """
    base = date_type.fromisoformat(run_date)
    candidates: list[str] = []
    for entry in _list_dirs(s3_client, bucket, prefix):
        try:
            d = date_type.fromisoformat(entry)
        except ValueError:
            continue
        if d > base:
            continue
        if (base - d).days > _LOOKBACK_DAYS:
            continue
        candidates.append(entry)
    return max(candidates) if candidates else None


def _get_json(s3_client: Any, bucket: str, key: str) -> Optional[dict]:
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
    except ClientError:
        return None
    try:
        return json.loads(obj["Body"].read())
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("[agent_justification] bad json at %s: %s", key, exc)
        return None


# ── per-source summarizers ────────────────────────────────────────────────


def summarize_judge(
    bucket: str, run_date: str, *, s3_client: Optional[Any] = None,
) -> dict:
    """Aggregate judge rubric scores across all agents for the most-recent
    `_eval/{date}/` subdir within the lookback window."""
    s3 = s3_client or boto3.client("s3")
    sf_date = _find_most_recent_date_subdir(
        s3, bucket=bucket, prefix="decision_artifacts/_eval/", run_date=run_date,
    )
    if sf_date is None:
        return {"status": "no_recent_sf_run", "run_date": run_date}

    agent_dirs = _list_dirs(
        s3, bucket, f"decision_artifacts/_eval/{sf_date}/",
    )
    n_agents = len(agent_dirs)
    if n_agents == 0:
        return {
            "status": "no_data",
            "run_date": run_date,
            "most_recent_sf_date": sf_date,
        }

    # Read one rubric file per agent and aggregate the per-agent overall
    # score. The actual rubric schema (verified against S3 2026-05-07)
    # carries `dimension_scores` — a list of {dimension, score, reasoning}
    # entries with per-dimension 1-5 ratings — NOT a top-level
    # `overall_score`. Compute each agent's overall as the mean across
    # its dimension scores; aggregate across agents.
    #
    # Skip rubrics where `judge_skip_reason` is non-null (judge bailed
    # before scoring) and rubrics where dimension_scores is missing or
    # empty. Counts these toward `n_agents` (presence) but not toward
    # `n_scored` (data) so the email surfaces both the judge attempt rate
    # and the successful scoring rate.
    per_agent_scores: list[float] = []
    for agent_dir in agent_dirs:
        # Rubric files: {date}.{model}.json — list and pick the lexically last
        try:
            resp = s3.list_objects_v2(
                Bucket=bucket,
                Prefix=f"decision_artifacts/_eval/{sf_date}/{agent_dir}/",
            )
        except ClientError:
            continue
        keys = [c["Key"] for c in resp.get("Contents") or [] if c["Key"].endswith(".json")]
        if not keys:
            continue
        latest_key = max(keys)
        rubric = _get_json(s3, bucket, latest_key)
        if rubric is None:
            continue
        if rubric.get("judge_skip_reason"):
            continue
        dim_scores = rubric.get("dimension_scores") or []
        per_dim = [
            float(d["score"]) for d in dim_scores
            if isinstance(d, dict) and isinstance(d.get("score"), (int, float))
        ]
        if not per_dim:
            continue
        per_agent_scores.append(sum(per_dim) / len(per_dim))
    scores = per_agent_scores

    if not scores:
        return {
            "status": "no_data",
            "run_date": run_date,
            "most_recent_sf_date": sf_date,
            "n_agents": n_agents,
        }

    return {
        "status": "ok",
        "run_date": run_date,
        "most_recent_sf_date": sf_date,
        "n_agents": n_agents,
        "n_scored": len(scores),
        "mean_score": round(sum(scores) / len(scores), 3),
        "min_score": round(min(scores), 3),
        "max_score": round(max(scores), 3),
    }


def _summarize_per_agent_weekly(
    s3_client: Any, *, bucket: str, prefix_root: str,
) -> dict:
    """Shared shape for clustering + counterfactual: per-agent JSONs under
    {prefix_root}/{agent_id_base}/{stem}.json. Returns the most-recent
    entry per agent across agents.

    Tolerant of BOTH per-agent file conventions (config#792 cutover for
    the counterfactual prefix; clustering's ``_analysis/`` prefix is
    unchanged and still ISO-week-keyed — both work here):

    - **Canonical (new):** ``{run_id}.json`` where run_id is a sortable
      ``YYMMDDHHMM`` timestamp, plus a ``latest.json`` operator sidecar.
    - **Legacy (old):** ``{YYYY-Www}.json`` ISO-week files (no sidecar).

    Both the YYMMDDHHMM run_ids and the YYYY-Www week labels sort
    chronologically under lexical ``max``, so the most-recent dated file
    is selected uniformly. The ``latest.json`` sidecar is explicitly
    excluded from the dated-key selection (it lexically outsorts the
    digit-prefixed run_ids but is a redundant mirror, not a dated
    forensic artifact) — reading it would double-count the newest run."""
    agent_bases = _list_dirs(s3_client, bucket, prefix_root)
    if not agent_bases:
        return {"status": "no_data"}

    per_agent: dict[str, dict] = {}
    most_recent_week: Optional[str] = None
    for agent in agent_bases:
        try:
            resp = s3_client.list_objects_v2(
                Bucket=bucket, Prefix=f"{prefix_root}{agent}/",
            )
        except ClientError:
            continue
        keys = [
            c["Key"] for c in resp.get("Contents") or []
            if c["Key"].endswith(".json")
            and c["Key"].rsplit("/", 1)[-1] != "latest.json"
        ]
        if not keys:
            continue
        latest_key = max(keys)  # lexical max on YYMMDDHHMM | YYYY-Www = chronological
        wk = latest_key.rsplit("/", 1)[-1].replace(".json", "")
        if most_recent_week is None or wk > most_recent_week:
            most_recent_week = wk
        body = _get_json(s3_client, bucket, latest_key)
        if body is not None:
            per_agent[agent] = {"week": wk, "body": body}

    if not per_agent:
        return {"status": "no_data"}
    return {
        "status": "ok",
        "n_agents": len(per_agent),
        "most_recent_week": most_recent_week,
        "per_agent": per_agent,
    }


def summarize_clustering(
    bucket: str, run_date: str, *, s3_client: Optional[Any] = None,
) -> dict:
    """Aggregate clustering metrics across all agents."""
    s3 = s3_client or boto3.client("s3")
    base = _summarize_per_agent_weekly(
        s3, bucket=bucket, prefix_root="decision_artifacts/_analysis/",
    )
    if base.get("status") != "ok":
        return {"status": "no_data", "run_date": run_date}

    concentrations: list[float] = []
    for agent, entry in base["per_agent"].items():
        c = entry["body"].get("top3_concentration")
        if isinstance(c, (int, float)):
            concentrations.append(float(c))

    return {
        "status": "ok",
        "run_date": run_date,
        "most_recent_week": base["most_recent_week"],
        "n_agents": base["n_agents"],
        "mean_top3_concentration": (
            round(sum(concentrations) / len(concentrations), 3)
            if concentrations else None
        ),
    }


def summarize_counterfactual(
    bucket: str, run_date: str, *, s3_client: Optional[Any] = None,
) -> dict:
    """Aggregate counterfactual decision-tree fit metrics across agents."""
    s3 = s3_client or boto3.client("s3")
    base = _summarize_per_agent_weekly(
        s3, bucket=bucket, prefix_root="decision_artifacts/_counterfactual/",
    )
    if base.get("status") != "ok":
        return {"status": "no_data", "run_date": run_date}

    match_rates: list[float] = []
    for entry in base["per_agent"].values():
        m = entry["body"].get("match_rate")
        if isinstance(m, (int, float)):
            match_rates.append(float(m))

    return {
        "status": "ok",
        "run_date": run_date,
        "most_recent_week": base["most_recent_week"],
        "n_agents": base["n_agents"],
        "mean_match_rate": (
            round(sum(match_rates) / len(match_rates), 3)
            if match_rates else None
        ),
        "agents": sorted(base["per_agent"].keys()),
    }


def _target_model_from_summary_basename(basename: str) -> str:
    """Recover the target_model label from a replay-summary file basename.

    Canonical layout (config#792): ``{run_id}_{target_model}.json`` where
    run_id is a 10-char ``YYMMDDHHMM`` structured timestamp. Strip the
    ``{run_id}_`` prefix to recover the target_model. Legacy layout had
    the bare ``{target_model}.json`` (under a ``{date}/`` partition) — no
    run_id prefix to strip, so fall through to the whole stem.
    """
    stem = basename[:-5] if basename.endswith(".json") else basename
    head, sep, tail = stem.partition("_")
    if sep and len(head) == 10 and head.isdigit():
        # ``YYMMDDHHMM_...`` canonical prefix — tail is the target_model.
        return tail
    return stem


def summarize_concordance(
    bucket: str, run_date: str, *, s3_client: Optional[Any] = None,
) -> dict:
    """Aggregate replay-concordance summaries (per-target-model).

    Tolerant of BOTH layouts (config#792 cutover):

    - **Canonical (new):** flat ``decision_artifacts/_replay_summary/
      {run_id}_{target_model}.json`` + a ``latest.json`` sidecar, where
      run_id is a sortable ``YYMMDDHHMM`` timestamp. Read-new-first: list
      the flat prefix, group by recovered target_model, keep the
      lexically-greatest (== most recent) dated key per target_model.
    - **Legacy (old):** date-partitioned ``_replay_summary/{YYYY-MM-DD}/
      {target_model}.json``. Used as a fallback only when no canonical
      flat keys are present, so the swap never strands pre-cutover data.
    """
    s3 = s3_client or boto3.client("s3")
    prefix = "decision_artifacts/_replay_summary/"

    # ── New canonical flat layout (read-new-first) ──
    try:
        resp = s3.list_objects_v2(
            Bucket=bucket, Prefix=prefix, Delimiter="/",
        )
    except ClientError:
        resp = {}
    # Flat per-run keys live directly under the prefix (Contents); the
    # legacy date partitions appear as CommonPrefixes and are ignored on
    # this path.
    flat_keys = [
        c["Key"] for c in resp.get("Contents") or []
        if c["Key"].endswith(".json")
        and not c["Key"].rsplit("/", 1)[-1].startswith("latest.")
    ]
    if flat_keys:
        # Group by target_model, keep the most-recent dated key per model
        # (lexical max on the YYMMDDHHMM-prefixed basename == chronological).
        latest_key_by_target: dict[str, str] = {}
        for k in flat_keys:
            basename = k.rsplit("/", 1)[-1]
            target = _target_model_from_summary_basename(basename)
            if target not in latest_key_by_target or k > latest_key_by_target[target]:
                latest_key_by_target[target] = k

        per_target: dict[str, dict] = {}
        for target, k in latest_key_by_target.items():
            body = _get_json(s3, bucket, k)
            if body is not None:
                per_target[target] = body

        if per_target:
            return {
                "status": "ok",
                "run_date": run_date,
                "layout": "canonical",
                "n_target_models": len(per_target),
                "per_target": per_target,
            }

    # ── Legacy date-partitioned layout (fallback) ──
    sf_date = _find_most_recent_date_subdir(
        s3, bucket=bucket, prefix=prefix, run_date=run_date,
    )
    if sf_date is None:
        return {"status": "no_recent_sf_run", "run_date": run_date}

    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}{sf_date}/")
    except ClientError:
        return {"status": "no_data", "run_date": run_date}
    keys = [c["Key"] for c in resp.get("Contents") or [] if c["Key"].endswith(".json")]
    if not keys:
        return {"status": "no_data", "run_date": run_date}

    per_target = {}
    for k in keys:
        body = _get_json(s3, bucket, k)
        if body is None:
            continue
        target = k.rsplit("/", 1)[-1].replace(".json", "")
        per_target[target] = body

    return {
        "status": "ok",
        "run_date": run_date,
        "layout": "legacy",
        "most_recent_sf_date": sf_date,
        "n_target_models": len(per_target),
        "per_target": per_target,
    }


# ── public composite ─────────────────────────────────────────────────────


def summarize_all(
    bucket: str, run_date: str, *, s3_client: Optional[Any] = None,
) -> dict:
    """Return all four summaries keyed by source name."""
    return {
        "judge": summarize_judge(bucket, run_date, s3_client=s3_client),
        "clustering": summarize_clustering(bucket, run_date, s3_client=s3_client),
        "concordance": summarize_concordance(bucket, run_date, s3_client=s3_client),
        "counterfactual": summarize_counterfactual(bucket, run_date, s3_client=s3_client),
    }
