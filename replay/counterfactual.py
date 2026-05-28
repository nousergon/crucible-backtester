"""Counterfactual rule fit — does the agent reduce to a 3-deep decision tree?

Per ROADMAP P0 "Replay harness + agent-justification gate" deliverable
#7 sub-bullet:

  Fit a shallow decision tree (depth ≤ 3) on (agent_input,
  agent_decision) pairs from the captured artifacts. If the tree
  achieves >85% match against the agent's actual decisions, the LLM
  is likely just running that tree at $X per call.

The third leg of the agent-justification triple alongside cross-week
rationale clustering (research/evals/rationale_clustering.py) and
cheap-model concordance (replay/batch.py). Each measures a different
flavor of "is this LLM doing real work":

  * Clustering        — does the agent emit varied rationales week-over-week?
  * Concordance       — would a cheaper model produce the same output?
  * Counterfactual    — would a 3-deep decision tree produce the same output?

All three reading the same decision_artifacts/ corpus.

Per-agent feature extraction is the hard part — each agent's
input_data_snapshot has a different shape, and what counts as a
"decision" varies (a regime literal for macro_economist; a per-
candidate ADVANCE/REJECT for ic_cio; a multi-pick set for
sector_quant). v1 ships full coverage for the two agents with the
cleanest (input, decision) shape and skip-markers for the others;
extending to remaining agents is incremental as their input contracts
soak.

v1 agent coverage:

  * ic_cio          — per-candidate (composite_score, conviction,
                       sector_modifier) → (ADVANCE / REJECT). One row
                       per candidate per run.
  * macro_economist — per-run (spy_trend, vix_level, yield_curve,
                       breadth) → regime literal. One row per run.

Future agents (skip-marker in v1):

  * sector_quant      — per-pick (technical sub-scores) → top-5 inclusion
  * sector_qual       — assessments require RAG-retrieval features not in snapshot
  * sector_peer_review — finalization decisions on quant + qual outputs
  * thesis_update     — held-stock score deltas
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import boto3

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────


DEFAULT_NAMESPACE = "AlphaEngine/Eval"
"""Same namespace as the eval-judge + clustering + concordance metrics —
the agent-justification dashboard reads everything from one stream."""

DEFAULT_METRIC_NAME = "agent_counterfactual_rule_fit"
"""Match rate [0, 1] of the fitted depth-≤3 decision tree against the
agent's actual decisions. >0.85 ⇒ the agent's decisions are well-
explained by a 3-deep rule → demote candidate."""

DEFAULT_BUCKET = "alpha-engine-research"
DEFAULT_CAPTURE_PREFIX = "decision_artifacts"
DEFAULT_ANALYSIS_PREFIX = "decision_artifacts/_counterfactual"

DEFAULT_WINDOW_DAYS = 28
"""Trailing scan window. Originally 56 days (8 weeks — same convention as
clustering); reduced 2026-05-19 to fit under the 600s Lambda ceiling once
captured-artifact count crossed ~32k+ in the 56-day window (ROADMAP L293).
28 days is still > the 30-day statistical-significance heuristic for the
LdP triple-barrier fits the counterfactual gate consumes, and at the
current ~585 artifacts/day rate keeps total per-run I/O bounded around
16k get_objects (vs 32k+ on the 56d default that timed out 5/16 + 5/17).

The per-agent cap below is the second-order bound — even if the window
default is reverted operationally, no single agent's artifact backlog
can stall the run."""

DEFAULT_MAX_ARTIFACTS_PER_AGENT = 500
"""Hard cap on per-agent_id_base artifact loads. Most agents stay well
below this (sector teams emit 1-2 artifacts/day, ic_cio 1/day, etc.);
the cap exists to bound the heavy population-wide agents (thesis_update
hit ~25 picks/day × 28 days = ~700 artifacts on its own at the
2026-05-13 substrate ramp-up rate). Most-recent-first ordering preserves
the recency-weighted tree fit. None or 0 disables the cap."""

DEFAULT_TREE_MAX_DEPTH = 3

MIN_SAMPLES_FOR_FIT = 10
"""Below this row count, decision-tree fit is statistically meaningless
(perfect fit on n=4 samples carries zero information). Mirrors the
thin-sample skip threshold pattern from clustering + concordance,
tightened upward because tree-fitting needs more samples than mean
estimation."""


# ── Per-agent feature + decision extraction ──────────────────────────────


# Sentinel for the "agent type not yet supported" path — distinct from
# "supported but no rows extractable from this artifact." Persisted to
# the analysis JSON so operators can see at a glance which agents the
# v1 framework covers.
UNSUPPORTED_AGENT = "UNSUPPORTED"


def extract_features_and_decision(
    agent_id: str,
    input_data_snapshot: dict[str, Any],
    agent_output: dict[str, Any],
) -> list[tuple[dict[str, float], Any]] | str:
    """Pull (feature_dict, decision) rows out of one captured artifact.

    Returns a list of rows when the agent is supported (multiple rows
    per artifact for agents that emit per-candidate decisions like
    ic_cio); empty list when no rows are extractable; the
    ``UNSUPPORTED_AGENT`` sentinel string when the agent_id family
    isn't covered by v1.

    Each row is ``(feature_dict, decision_label)``:
    - feature_dict: ``{feature_name: float}`` — values must be numeric
      so DictVectorizer can turn them into a dense feature matrix.
    - decision_label: hashable (str / int) — what the agent chose.

    Per-agent extraction logic lives here so the framework's tree-
    fitting + match-rate code stays agent-agnostic. Extending to a new
    agent = adding a branch here.
    """
    if not isinstance(agent_output, dict):
        return []

    base_id = (agent_id or "").split(":", 1)[0]

    if base_id == "ic_cio":
        return _extract_ic_cio(input_data_snapshot or {}, agent_output)

    if base_id == "macro_economist":
        return _extract_macro_economist(input_data_snapshot or {}, agent_output)

    # All other agents — sector_quant, sector_qual, sector_peer_review,
    # thesis_update — return UNSUPPORTED. Future PRs add coverage as
    # the input-snapshot contracts soak.
    return UNSUPPORTED_AGENT


def _extract_ic_cio(
    snapshot: dict[str, Any], output: dict[str, Any],
) -> list[tuple[dict[str, float], str]]:
    """ic_cio per-candidate (composite_score, conviction, ...) → ADVANCE/REJECT.

    The CIO agent receives a list of candidates (each with composite +
    sub-scores + sector_modifier) and emits an ic_decisions list with
    a literal decision per candidate. The interesting hypothesis: does
    the CIO's ADVANCE/REJECT call reduce to a simple threshold on
    composite_score + conviction?

    Snapshot shape (varies but typically):
        {"candidates": [{"ticker": ..., "composite_score": ...,
                         "sector_modifier": ..., ...}, ...]}
    Output shape:
        {"ic_decisions": [{"ticker": ..., "decision": "ADVANCE"|"REJECT",
                           "conviction": ...}, ...]}

    Match candidates to decisions on ticker. Skip rows where the
    snapshot lacks the candidate (e.g. captured artifact pre-dates a
    snapshot-schema change) or where the decision literal is the
    deadlock sentinel (NO_ADVANCE_DEADLOCK is a separate concern from
    the binary ADVANCE/REJECT call).
    """
    decisions = output.get("ic_decisions") or output.get("decisions") or []
    if not isinstance(decisions, list):
        return []

    # Build candidate-features index by ticker. Snapshot is captured
    # input — fall through if the field name varies.
    candidates_raw = (
        snapshot.get("candidates")
        or snapshot.get("entrant_candidates")
        or []
    )
    if not isinstance(candidates_raw, list):
        return []
    by_ticker: dict[str, dict] = {
        c["ticker"]: c
        for c in candidates_raw
        if isinstance(c, dict) and c.get("ticker")
    }

    rows: list[tuple[dict[str, float], str]] = []
    for d in decisions:
        if not isinstance(d, dict):
            continue
        ticker = d.get("ticker")
        decision_lit = d.get("decision")
        if ticker is None or decision_lit not in ("ADVANCE", "REJECT"):
            continue
        # NO_ADVANCE_DEADLOCK + ADVANCE_FORCED handled separately —
        # they're CIO's "I can't decide" / "post-processing forced this"
        # sentinels rather than the ADVANCE/REJECT binary call.

        cand = by_ticker.get(ticker, {})
        # Numeric-only features. Missing → 0.0 (sklearn DictVectorizer
        # tolerates this, and missing-feature variance becomes its own
        # signal — a 3-deep tree splitting on whether composite_score
        # is present is meaningful).
        features = {
            "composite_score": _safe_float(cand.get("composite_score")),
            "quant_score": _safe_float(cand.get("quant_score")),
            "qual_score": _safe_float(cand.get("qual_score")),
            "sector_modifier": _safe_float(cand.get("sector_modifier"), default=1.0),
            "conviction": _safe_float(cand.get("conviction")),
        }
        rows.append((features, decision_lit))

    return rows


def _extract_macro_economist(
    snapshot: dict[str, Any], output: dict[str, Any],
) -> list[tuple[dict[str, float], str]]:
    """macro_economist per-run (macro indicators) → regime literal.

    The macro agent receives macro indicators (SPY trend, VIX, yield
    curve, breadth) and emits a regime literal (bull / neutral / bear).
    One row per artifact. Legacy 4-class "caution" tolerated on the
    enum for grandfather replay over pre-v0.42.0 archives (the
    macro_agent's _validate_regime coerces raw LLM "caution" to
    "neutral" upstream post-cutover —
    caution-regime-retirement-260528.md).

    Snapshot shape (varies):
        {"macro_indicators": {"spy_20d_return": ...,
                              "vix_level": ...,
                              "yield_curve_slope": ...,
                              "market_breadth": ...,
                              ...}}
    Output shape:
        {"market_regime": "bull"|"neutral"|"bear" (post-v0.42.0)
                          | legacy "caution" (grandfather), ...}
    """
    regime = output.get("market_regime")
    if regime not in ("bull", "neutral", "bear", "caution"):
        return []

    indicators = (
        snapshot.get("macro_indicators")
        or snapshot.get("indicators")
        or {}
    )
    if not isinstance(indicators, dict):
        indicators = {}

    features = {
        "spy_20d_return": _safe_float(indicators.get("spy_20d_return")),
        "spy_20d_vol": _safe_float(indicators.get("spy_20d_vol")),
        "vix_level": _safe_float(indicators.get("vix_level")),
        "vix_term_slope": _safe_float(indicators.get("vix_term_slope")),
        "yield_curve_slope": _safe_float(indicators.get("yield_curve_slope")),
        "market_breadth": _safe_float(indicators.get("market_breadth")),
    }
    return [(features, regime)]


def _safe_float(v: Any, *, default: float = 0.0) -> float:
    """Cast to float, falling back to ``default`` for None / non-numeric.
    Used pervasively in feature extraction since captured snapshots
    have variable field presence (None on missing data, str on certain
    nullable scores, etc.)."""
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ── Tree fit + match rate ────────────────────────────────────────────────


def fit_counterfactual_tree(
    rows: list[tuple[dict[str, float], Any]],
    *,
    max_depth: int = DEFAULT_TREE_MAX_DEPTH,
) -> dict[str, Any]:
    """Train a shallow ``DecisionTreeClassifier`` on the rows and return
    a fit-result dict with match rate + tree structure.

    Sklearn imports are deferred to keep the module importable in
    environments without sklearn (test fixtures that don't exercise
    the fit path).

    Returns dict with:
      - ``n_samples``: total rows
      - ``match_rate``: [0, 1] — tree predictions matching actual
      - ``n_classes``: distinct decision labels
      - ``feature_names``: ordered list of features the tree used
      - ``tree_text``: human-readable tree structure (sklearn export_text)
      - ``feature_importances``: per-feature [0, 1] weight

    Caller decides whether match_rate > threshold ⇒ "replace candidate."
    """
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.tree import DecisionTreeClassifier, export_text

    if len(rows) < MIN_SAMPLES_FOR_FIT:
        return {
            "n_samples": len(rows),
            "match_rate": None,
            "skip_reason": (
                f"thin_sample (n={len(rows)} < "
                f"{MIN_SAMPLES_FOR_FIT})"
            ),
        }

    feature_dicts = [r[0] for r in rows]
    labels = [r[1] for r in rows]

    distinct_labels = set(labels)
    if len(distinct_labels) < 2:
        # Single-class corpus — tree fit is degenerate (always predict
        # the only class, match rate 1.0). Surfacing this as a
        # skip_reason is more honest than reporting 1.0 — a triple of
        # (n=200, match=1.0, single_class=True) means "the agent never
        # decided otherwise across this window," which is information
        # in itself but not what counterfactual rule fit measures.
        return {
            "n_samples": len(rows),
            "match_rate": None,
            "n_classes": 1,
            "skip_reason": (
                f"single_class (only one decision label observed: "
                f"{next(iter(distinct_labels))})"
            ),
        }

    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform(feature_dicts)
    y = labels

    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    tree.fit(X, y)
    predictions = tree.predict(X)
    match_rate = float(sum(1 for p, a in zip(predictions, y) if p == a) / len(y))

    feature_names = list(vec.get_feature_names_out())
    tree_text = export_text(
        tree, feature_names=feature_names, max_depth=max_depth,
    )

    return {
        "n_samples": len(rows),
        "match_rate": match_rate,
        "n_classes": len(distinct_labels),
        "feature_names": feature_names,
        "tree_text": tree_text,
        "feature_importances": dict(
            zip(feature_names, [float(w) for w in tree.feature_importances_])
        ),
    }


# ── S3 corpus reading ────────────────────────────────────────────────────


def _list_artifact_keys_in_window(
    s3: Any,
    *,
    bucket: str,
    capture_prefix: str,
    end_date: datetime,
    window_days: int,
    agent_filter: list[str] | None = None,
    max_artifacts_per_agent: int | None = DEFAULT_MAX_ARTIFACTS_PER_AGENT,
) -> list[str]:
    """List captured-artifact keys under the trailing window. Same
    meta-prefix exclusion + per-day pagination as the rationale-
    clustering + concordance pipelines.

    Iterates day-by-day from end_date BACKWARD so the keys list is
    already ordered most-recent-first. This lets the per-agent cap
    drop the oldest artifacts when an agent_id_base has more than
    ``max_artifacts_per_agent`` keys in the window.

    The cap is applied here (pre ``_load_artifact``) so the bounded
    list is what hits the expensive S3 get_object loop downstream —
    the bound translates directly into a wall-clock budget under the
    600s Lambda ceiling (ROADMAP L293).
    """
    paginator = s3.get_paginator("list_objects_v2")
    # day_offset iterates 0..N-1 from end_date (today) backward, so
    # appending in iteration order produces a most-recent-first list.
    keys_by_agent: dict[str, list[str]] = defaultdict(list)
    all_keys: list[str] = []

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
                if any(
                    f"/_{p}/" in key
                    for p in (
                        "eval", "eval_judge_only", "analysis",
                        "cost", "cost_raw",
                        "replay", "replay_summary",
                        "counterfactual",
                    )
                ):
                    continue
                base_id = _agent_id_base_from_key(key) or "unknown"
                if agent_filter and base_id not in agent_filter:
                    continue
                if (
                    max_artifacts_per_agent is not None
                    and max_artifacts_per_agent > 0
                    and len(keys_by_agent[base_id]) >= max_artifacts_per_agent
                ):
                    # Cap hit — drop older artifacts for this agent.
                    continue
                keys_by_agent[base_id].append(key)
                all_keys.append(key)
    return all_keys


def _agent_id_base_from_key(key: str) -> Optional[str]:
    parts = key.split("/")
    if len(parts) < 2:
        return None
    return parts[-2].split(":", 1)[0]


def _load_artifact(s3: Any, *, bucket: str, key: str) -> dict[str, Any]:
    raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(raw)


# ── CW emission + persistence ────────────────────────────────────────────


def _emit_match_rate_metric(
    cw: Any,
    *,
    namespace: str,
    metric_name: str,
    agent_id_base: str,
    match_rate: float,
    n_samples: int,
    timestamp: datetime,
) -> None:
    """One datapoint per agent_id_base. Mirrors the clustering +
    concordance metric shape: primary [0, 1] value + an _n_samples
    counter for thin-sample observability on the dashboard."""
    cw.put_metric_data(
        Namespace=namespace,
        MetricData=[
            {
                "MetricName": metric_name,
                "Dimensions": [
                    {"Name": "judged_agent_id", "Value": agent_id_base},
                ],
                "Value": float(match_rate),
                "Unit": "None",
                "Timestamp": timestamp,
            },
            {
                "MetricName": f"{metric_name}_n_samples",
                "Dimensions": [
                    {"Name": "judged_agent_id", "Value": agent_id_base},
                ],
                "Value": float(n_samples),
                "Unit": "Count",
                "Timestamp": timestamp,
            },
        ],
    )


def _persist_per_agent_analysis(
    s3: Any,
    *,
    bucket: str,
    analysis_prefix: str,
    agent_id_base: str,
    end_date: datetime,
    payload: dict[str, Any],
) -> str:
    """Write per-agent analysis JSON to
    ``{analysis_prefix}/{agent_id_base}/{YYYY-WW}.json``. ISO week is
    the natural cadence (one analysis per Saturday SF run)."""
    iso_year, iso_week, _ = end_date.isocalendar()
    key = f"{analysis_prefix}/{agent_id_base}/{iso_year}-W{iso_week:02d}.json"
    body = json.dumps(payload, indent=2, default=str).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    return key


# ── Top-level pipeline ───────────────────────────────────────────────────


def compute_and_emit(
    *,
    end_time: Optional[datetime] = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    bucket: str = DEFAULT_BUCKET,
    capture_prefix: str = DEFAULT_CAPTURE_PREFIX,
    analysis_prefix: str = DEFAULT_ANALYSIS_PREFIX,
    namespace: str = DEFAULT_NAMESPACE,
    metric_name: str = DEFAULT_METRIC_NAME,
    max_depth: int = DEFAULT_TREE_MAX_DEPTH,
    agent_filter: Optional[list[str]] = None,
    max_artifacts_per_agent: int | None = DEFAULT_MAX_ARTIFACTS_PER_AGENT,
    s3_client: Optional[Any] = None,
    cloudwatch_client: Optional[Any] = None,
    emit_metrics: bool = True,
) -> dict[str, Any]:
    """Read captured artifacts in the trailing window, fit a counterfactual
    tree per supported agent_id family, persist analysis JSON, emit
    CloudWatch match-rate metric.

    Returns summary dict with per-agent fit results + skipped-thin +
    unsupported-agent buckets + load failures.
    """
    s3 = s3_client or boto3.client("s3")
    cw = cloudwatch_client or (boto3.client("cloudwatch") if emit_metrics else None)
    end = end_time or datetime.now(timezone.utc)
    window_start = end - timedelta(days=window_days)

    keys = _list_artifact_keys_in_window(
        s3,
        bucket=bucket,
        capture_prefix=capture_prefix,
        end_date=end,
        window_days=window_days,
        agent_filter=agent_filter,
        max_artifacts_per_agent=max_artifacts_per_agent,
    )

    logger.info(
        "[counterfactual] discovered %d artifacts window=[%s, %s] "
        "(max_artifacts_per_agent=%s)",
        len(keys), window_start.isoformat(), end.isoformat(),
        max_artifacts_per_agent if max_artifacts_per_agent else "unbounded",
    )

    # Group rows by agent_id_base. Unsupported agents counted separately
    # so the summary is honest about what v1 covers.
    rows_by_agent: dict[str, list[tuple[dict[str, float], Any]]] = defaultdict(list)
    unsupported_seen: set[str] = set()
    load_failures: list[dict[str, str]] = []

    for key in keys:
        agent_id_base = _agent_id_base_from_key(key) or "unknown"
        try:
            artifact = _load_artifact(s3, bucket=bucket, key=key)
        except Exception as exc:  # noqa: BLE001
            load_failures.append({"key": key, "error": str(exc)})
            logger.warning(
                "[counterfactual] load failure key=%s err=%s", key, exc,
            )
            continue

        result = extract_features_and_decision(
            artifact.get("agent_id", agent_id_base),
            artifact.get("input_data_snapshot") or {},
            artifact.get("agent_output") or {},
        )
        if result == UNSUPPORTED_AGENT:
            unsupported_seen.add(agent_id_base)
            continue
        rows_by_agent[agent_id_base].extend(result)

    per_agent_summary: list[dict[str, Any]] = []
    skipped_thin: list[dict[str, Any]] = []
    fit_failures: list[dict[str, str]] = []

    for agent_id_base in sorted(rows_by_agent.keys()):
        rows = rows_by_agent[agent_id_base]
        try:
            fit = fit_counterfactual_tree(rows, max_depth=max_depth)
        except Exception as exc:  # noqa: BLE001
            fit_failures.append({"agent_id_base": agent_id_base, "error": str(exc)})
            logger.exception(
                "[counterfactual] fit failure agent=%s", agent_id_base,
            )
            continue

        if fit.get("match_rate") is None:
            skipped_thin.append({
                "agent_id_base": agent_id_base,
                "n_samples": fit["n_samples"],
                "skip_reason": fit.get("skip_reason"),
            })
            continue

        # Persist + emit.
        payload = {
            "schema_version": 1,
            "agent_id_base": agent_id_base,
            "window_start": window_start.isoformat(),
            "window_end": end.isoformat(),
            "max_depth": max_depth,
            **fit,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            analysis_key = _persist_per_agent_analysis(
                s3,
                bucket=bucket,
                analysis_prefix=analysis_prefix,
                agent_id_base=agent_id_base,
                end_date=end,
                payload=payload,
            )
        except Exception as exc:  # noqa: BLE001
            fit_failures.append({
                "agent_id_base": agent_id_base, "stage": "persist",
                "error": str(exc),
            })
            logger.exception(
                "[counterfactual] persist failure agent=%s", agent_id_base,
            )
            continue

        if cw is not None:
            try:
                _emit_match_rate_metric(
                    cw,
                    namespace=namespace,
                    metric_name=metric_name,
                    agent_id_base=agent_id_base,
                    match_rate=fit["match_rate"],
                    n_samples=fit["n_samples"],
                    timestamp=end,
                )
            except Exception as exc:  # noqa: BLE001
                fit_failures.append({
                    "agent_id_base": agent_id_base, "stage": "metric_emit",
                    "error": str(exc),
                })
                logger.warning(
                    "[counterfactual] metric emission failed agent=%s err=%s",
                    agent_id_base, exc,
                )

        per_agent_summary.append({
            "agent_id_base": agent_id_base,
            "n_samples": fit["n_samples"],
            "match_rate": fit["match_rate"],
            "n_classes": fit.get("n_classes"),
            "analysis_key": analysis_key,
        })

        logger.info(
            "[counterfactual] agent=%s n=%d classes=%d match_rate=%.3f",
            agent_id_base, fit["n_samples"], fit.get("n_classes", 0),
            fit["match_rate"],
        )

    return {
        "window_start": window_start.isoformat(),
        "window_end": end.isoformat(),
        "max_depth": max_depth,
        "artifacts_discovered": len(keys),
        "agents_analyzed": len(per_agent_summary),
        "agents_skipped_thin_sample": skipped_thin,
        "agents_unsupported": sorted(unsupported_seen),
        "load_failures": load_failures,
        "fit_failures": fit_failures,
        "per_agent": per_agent_summary,
    }
