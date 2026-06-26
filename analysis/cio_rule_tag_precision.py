"""
cio_rule_tag_precision.py — Per-rule-tag precision of the LLM CIO's gates.

The CIO (chief-investment-officer agent) emits a closed-vocabulary list of
``rule_tags`` alongside each ADVANCE / REJECT decision — the specific gates
that drove the call (e.g. ``rr_asymmetry``, ``qual_veto``,
``catalyst_specificity``). PRs alpha-engine-lib #35 (v0.7.0 schema) +
alpha-engine-config #101 (prompt v1.3.0) + crucible-research #152 (DB
persistence) shipped end-to-end producer-side persistence of those tags into
``cio_evaluations.rule_tags`` (research schema migration 14). This module is
the missing *consumer*: without it the column accumulates data that nobody
queries.

The point is per-gate precision tracking — "which CIO gates are systematically
over- or under-rejecting?" — so the deeper "drop the LLM CIO" decision is
defensible if it ever needs to be made.

Reads ``cio_evaluations`` joined to ``universe_returns`` from research.db on
``(ticker, eval_date)`` (both tables already exist; no schema change). The
realized outcome is ``universe_returns.beat_spy_5d`` (1 = the name beat SPY
over the next 5 trading days, 0 = it did not) — the same realized-outcome
column the sibling ``quant_rank_quality`` / ``end_to_end`` analyses join on.

For each rule tag it computes:

  * ``n_decisions``       — total tagged decisions (ADVANCE + REJECT) with a
                            realized 5d outcome.
  * ``advance_precision`` — of ADVANCE-class decisions carrying this tag,
                            the fraction that beat SPY at 5d. High = the gate
                            is admitting winners; low = the gate is waving
                            losers through.
  * ``reject_beat_rate``  — of REJECT decisions carrying this tag, the
                            fraction that *would have* beaten SPY at 5d (the
                            per-tag false-negative rate). High = the gate is
                            systematically rejecting names that would have
                            won — the canary for an over-rejecting gate.

A decision's ``cio_decision`` is classed ADVANCE when it is ``ADVANCE`` or
``ADVANCE_FORCED`` (both count as advances per crucible-research's ic_cio
post-processing), and REJECT when it is ``REJECT``. Other verdicts (HOLD,
NO_ADVANCE_DEADLOCK, UNKNOWN, …) are counted in ``n_decisions`` but excluded
from both rate numerators/denominators — they are neither an admit nor a
reject of the name.

``rule_tags`` is a JSON list[str] string. Rows where it ``IS NULL`` (legacy
artifacts from prompts < v1.3.0) are skipped — no false-positive risk, per the
issue's gate. A row with N tags contributes to all N tags' counts.

Gate: returns ``status="insufficient_data"`` until ``MIN_TAGGED_DECISIONS``
tagged rows with realized outcomes accumulate (≥4 weeks of rule_tags data,
going forward from research#152's deploy), so the evaluator surfaces the gap
rather than reporting noise off a handful of rows.

Returns the standard backtester-evaluator status dict so the existing
``CompletenessTracker.run_module`` pattern handles it without bespoke wiring
(mirrors ``analysis/quant_rank_quality.py`` shape). Emits per-tag metrics to
CloudWatch dimensioned by ``rule_tag`` so the substrate's alarm machinery can
fire when any tag's ``reject_beat_rate`` exceeds ``REJECT_BEAT_ALARM``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


# ── Configuration ───────────────────────────────────────────────────────────

DEFAULT_LOOKBACK_WEEKS = 8
DEFAULT_CW_NAMESPACE = "AlphaEngine/CioRuleTagPrecision"

# Verdicts that count as "the CIO admitted this name". Both literals are
# advances per crucible-research's _post_process_cio_decisions (ic_cio.py).
ADVANCE_DECISIONS = ("ADVANCE", "ADVANCE_FORCED")
REJECT_DECISIONS = ("REJECT",)

# Insufficient-data gate. The issue calls for ≥4 weeks of rule_tags data
# before the view is useful; floor the tagged-decision count so a thin DB
# surfaces as insufficient_data rather than reporting precision off noise.
MIN_TAGGED_DECISIONS = 20

# Alarm threshold. A tag whose REJECTs beat SPY more than half the time is
# systematically over-rejecting winners — the canary the whole arc exists to
# catch. Loose by design; the email/CW alarm flags any tag above it.
REJECT_BEAT_ALARM = 0.50


# ── rule_tags parsing ───────────────────────────────────────────────────────


def _parse_rule_tags(raw: Any) -> list[str]:
    """Parse the persisted ``rule_tags`` cell into a list of tag strings.

    Persisted as a JSON list[str] (``json.dumps`` in crucible-research's
    ``write_cio_evaluations``). Returns [] for NULL / blank / malformed /
    non-list payloads so a single bad row can never crash the weekly run.
    De-dupes within a row so one decision counts a tag at most once.
    """
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        text = str(raw).strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return []
        if not isinstance(parsed, list):
            return []
        items = parsed
    # Keep only non-empty strings; preserve first-seen order, de-dupe.
    seen: dict[str, None] = {}
    for it in items:
        if isinstance(it, str) and it.strip():
            seen.setdefault(it.strip(), None)
    return list(seen.keys())


# ── Per-tag aggregation ─────────────────────────────────────────────────────


def _accumulate(
    rows: list[tuple[Any, Any, Any]],
) -> dict[str, dict[str, int]]:
    """Fold ``(cio_decision, rule_tags, beat_spy_5d)`` rows into per-tag counts.

    Returns ``{tag: {n_decisions, n_advance, advance_beat, n_reject,
    reject_beat}}``. Only rows with a non-NULL realized ``beat_spy_5d`` and at
    least one parsed tag contribute. ``beat_spy_5d`` is coerced truthy: 1/True
    → beat, 0/False → not.
    """
    agg: dict[str, dict[str, int]] = {}
    for decision, rule_tags_raw, beat in rows:
        if beat is None:
            continue
        tags = _parse_rule_tags(rule_tags_raw)
        if not tags:
            continue
        beat_won = 1 if int(beat) == 1 else 0
        is_advance = decision in ADVANCE_DECISIONS
        is_reject = decision in REJECT_DECISIONS
        for tag in tags:
            slot = agg.setdefault(
                tag,
                {
                    "n_decisions": 0,
                    "n_advance": 0,
                    "advance_beat": 0,
                    "n_reject": 0,
                    "reject_beat": 0,
                },
            )
            slot["n_decisions"] += 1
            if is_advance:
                slot["n_advance"] += 1
                slot["advance_beat"] += beat_won
            elif is_reject:
                slot["n_reject"] += 1
                slot["reject_beat"] += beat_won
    return agg


def _finalize_tag(tag: str, c: dict[str, int]) -> dict[str, Any]:
    """Turn raw per-tag counts into the reported metric row.

    ``advance_precision`` / ``reject_beat_rate`` are None (not 0) when their
    denominator is empty, so the evaluator renders "—" rather than conflating
    "no ADVANCEs carried this tag" with "every ADVANCE lost".
    """
    n_advance = c["n_advance"]
    n_reject = c["n_reject"]
    advance_precision = (
        round(c["advance_beat"] / n_advance, 4) if n_advance else None
    )
    reject_beat_rate = (
        round(c["reject_beat"] / n_reject, 4) if n_reject else None
    )
    return {
        "rule_tag": tag,
        "n_decisions": c["n_decisions"],
        "n_advance": n_advance,
        "advance_precision": advance_precision,
        "n_reject": n_reject,
        "reject_beat_rate": reject_beat_rate,
    }


# ── CloudWatch emission ─────────────────────────────────────────────────────


def _emit_cw_metrics(
    cw: Any,
    *,
    namespace: str,
    run_date: datetime,
    per_tag: list[dict[str, Any]],
) -> None:
    """Emit per-tag precision metrics to CloudWatch, dimensioned by rule_tag.

    Lets the substrate's alarm machinery fire on a sustained
    ``reject_beat_rate`` above ``REJECT_BEAT_ALARM`` (over-rejecting gate) or a
    collapsing ``advance_precision`` (a gate admitting losers).
    """
    metric_data: list[dict[str, Any]] = []
    for entry in per_tag:
        dims = [{"Name": "rule_tag", "Value": entry["rule_tag"]}]
        metric_data.append({
            "MetricName": "n_decisions",
            "Dimensions": dims,
            "Value": float(entry["n_decisions"]),
            "Unit": "Count",
            "Timestamp": run_date,
        })
        if entry["advance_precision"] is not None:
            metric_data.append({
                "MetricName": "advance_precision",
                "Dimensions": dims,
                "Value": float(entry["advance_precision"]),
                "Unit": "None",
                "Timestamp": run_date,
            })
        if entry["reject_beat_rate"] is not None:
            metric_data.append({
                "MetricName": "reject_beat_rate",
                "Dimensions": dims,
                "Value": float(entry["reject_beat_rate"]),
                "Unit": "None",
                "Timestamp": run_date,
            })

    for i in range(0, len(metric_data), 20):
        cw.put_metric_data(Namespace=namespace, MetricData=metric_data[i: i + 20])


# ── Public entry point ──────────────────────────────────────────────────────


def compute_cio_rule_tag_precision(
    db_path: str | None = None,
    db_conn: sqlite3.Connection | None = None,
    run_date: str | None = None,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    min_tagged_decisions: int = MIN_TAGGED_DECISIONS,
    cloudwatch_client: Any = None,
    cw_namespace: str = DEFAULT_CW_NAMESPACE,
    emit_metrics: bool = True,
) -> dict[str, Any]:
    """Compute per-rule-tag CIO precision over a rolling window.

    Joins ``cio_evaluations`` → ``universe_returns`` on ``(ticker, eval_date)``
    and, per ``rule_tag``, reports ``n_decisions``, ADVANCE precision (fraction
    of ADVANCE-tagged decisions that beat SPY at 5d) and the REJECT-beat rate
    (per-tag false-negative rate).

    Args:
        db_path: path to research.db on disk. Either this or db_conn required.
        db_conn: already-open SQLite connection (tests + reusing the
            evaluator's already-pulled DB).
        run_date: ISO date (window end). Defaults to today (UTC).
        lookback_weeks: trailing N weeks to aggregate over (8 mirrors the
            other rolling-window diagnostics).
        min_tagged_decisions: insufficient-data floor; below it the result is
            status="insufficient_data" (the ≥4-weeks gate).
        cloudwatch_client: injected boto3 cloudwatch client (tests).
        cw_namespace: CW namespace.
        emit_metrics: if False, skip CW emission (--freeze runs / tests).

    Returns dict:
        status: "ok" | "insufficient_data" | "no_data" | "error"
        run_date, window_start, window_end, lookback_weeks
        per_tag: list[dict] sorted by n_decisions desc — rule_tag,
                 n_decisions, n_advance, advance_precision, n_reject,
                 reject_beat_rate
        n_tagged_decisions: total tagged rows with realized outcomes in window
        overall_advance_precision: pooled across all ADVANCE-class tagged rows
        alarm_tags: tags whose reject_beat_rate exceeds REJECT_BEAT_ALARM
        reject_beat_alarm: the threshold echoed for the renderer
    """
    if db_conn is None and db_path is None:
        return {"status": "error", "error": "must provide db_path or db_conn"}

    run_date = run_date or datetime.utcnow().strftime("%Y-%m-%d")
    try:
        end_dt = datetime.strptime(run_date, "%Y-%m-%d")
    except ValueError as e:
        return {"status": "error", "error": f"invalid run_date: {e}"}
    start_dt = end_dt - timedelta(weeks=lookback_weeks)
    start_iso = start_dt.strftime("%Y-%m-%d")
    end_iso = end_dt.strftime("%Y-%m-%d")

    own_conn = False
    conn = db_conn
    if conn is None:
        conn = sqlite3.connect(db_path)
        own_conn = True

    try:
        # Fast-fail when a required table is missing entirely (fresh DB /
        # smoke env). status=no_data lets the evaluator surface the gap
        # rather than crash.
        for table in ("cio_evaluations", "universe_returns"):
            try:
                conn.execute(f"SELECT 1 FROM {table} LIMIT 1")
            except sqlite3.OperationalError:
                return {
                    "status": "no_data",
                    "run_date": run_date,
                    "reason": f"{table} table missing",
                }

        # rule_tags arrived with migration 14; a DB predating it has no such
        # column. Surface that as no_data, not a crash.
        ce_cols = {r[1] for r in conn.execute("PRAGMA table_info(cio_evaluations)")}
        if "rule_tags" not in ce_cols:
            return {
                "status": "no_data",
                "run_date": run_date,
                "reason": "cio_evaluations.rule_tags column missing (pre-migration-14 DB)",
            }
        if "cio_decision" not in ce_cols:
            return {
                "status": "no_data",
                "run_date": run_date,
                "reason": "cio_evaluations.cio_decision column missing",
            }

        rows = conn.execute(
            """
            SELECT ce.cio_decision, ce.rule_tags, ur.beat_spy_5d
            FROM cio_evaluations ce
            INNER JOIN universe_returns ur
              ON ce.ticker = ur.ticker AND ce.eval_date = ur.eval_date
            WHERE ce.eval_date BETWEEN ? AND ?
              AND ce.rule_tags IS NOT NULL
              AND ur.beat_spy_5d IS NOT NULL
            """,
            (start_iso, end_iso),
        ).fetchall()

        agg = _accumulate(rows)
        per_tag = sorted(
            (_finalize_tag(tag, counts) for tag, counts in agg.items()),
            key=lambda e: (-e["n_decisions"], e["rule_tag"]),
        )

        n_tagged = sum(e["n_decisions"] for e in per_tag)

        if n_tagged == 0:
            return {
                "status": "no_data",
                "run_date": run_date,
                "window_start": start_iso,
                "window_end": end_iso,
                "reason": (
                    f"no cio_evaluations rows with non-NULL rule_tags joined "
                    f"to universe_returns in window {start_iso}..{end_iso}"
                ),
            }

        if n_tagged < min_tagged_decisions:
            return {
                "status": "insufficient_data",
                "run_date": run_date,
                "window_start": start_iso,
                "window_end": end_iso,
                "n_tagged_decisions": n_tagged,
                "min_tagged_decisions": min_tagged_decisions,
                "per_tag": per_tag,
                "reason": (
                    f"only {n_tagged} tagged decision(s) with realized outcomes "
                    f"in window (need {min_tagged_decisions}); "
                    f"≥4 weeks of rule_tags data must accumulate"
                ),
            }

        # Pooled ADVANCE precision across every ADVANCE-class tagged row.
        total_advance = sum(e["n_advance"] for e in per_tag)
        total_advance_beat = sum(agg[e["rule_tag"]]["advance_beat"] for e in per_tag)
        overall_advance_precision = (
            round(total_advance_beat / total_advance, 4) if total_advance else None
        )

        alarm_tags = sorted(
            e["rule_tag"] for e in per_tag
            if e["reject_beat_rate"] is not None
            and e["reject_beat_rate"] > REJECT_BEAT_ALARM
        )

        if emit_metrics:
            try:
                cw = cloudwatch_client
                if cw is None:
                    import boto3
                    cw = boto3.client("cloudwatch")
                _emit_cw_metrics(
                    cw, namespace=cw_namespace, run_date=end_dt, per_tag=per_tag,
                )
                logger.info(
                    "cio_rule_tag_precision: emitted metrics for %d tags to "
                    "CloudWatch namespace %s on %s",
                    len(per_tag), cw_namespace, run_date,
                )
            except Exception as e:
                logger.warning(
                    "cio_rule_tag_precision: CloudWatch emission failed: %s — "
                    "continuing with JSON artifact only", e,
                )

        return {
            "status": "ok",
            "run_date": run_date,
            "window_start": start_iso,
            "window_end": end_iso,
            "lookback_weeks": lookback_weeks,
            "per_tag": per_tag,
            "n_tagged_decisions": n_tagged,
            "overall_advance_precision": overall_advance_precision,
            "alarm_tags": alarm_tags,
            "reject_beat_alarm": REJECT_BEAT_ALARM,
        }
    finally:
        if own_conn:
            conn.close()
