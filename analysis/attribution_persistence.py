"""
attribution_persistence.py — config#946 part 2: alert when attribution stays
under-powered (`insufficient_data`) for too many *consecutive* evaluation cycles.

config#946 raised the attribution sample floor to 100 (sub-score correlation
needs the larger N for an FDR-robust multivariate fit) — `compute_attribution`
returns `status="insufficient_data"` below it (analysis/attribution.py:48). A
SINGLE under-powered cycle is normal early in a cohort's life (outcomes have not
realized yet); what matters is PERSISTENCE — if attribution never crosses the
floor over many weeks, the analysis is structurally starved (pick volume too
low, or the 21d outcome backfill is not landing) and the operator should either
widen the cohort or revisit the floor. Without this, an `insufficient_data`
attribution tile reads identically on week 1 and week 12.

`sample_size_adequacy.py` grades the per-cycle SNAPSHOT only (no history). This
module adds the cross-cycle PERSISTENCE half the issue asks for, mirroring the
established S3-history + consecutive-count + always-emit pattern of
`retrain_alert.py` and the report-card section builders (cost_report /
calibration_report): append this cycle's attribution status to an append-only
S3 history, count the trailing run of `insufficient_data`, and surface a warning
section in the evaluator report once it exceeds the threshold.

Threshold: ``ATTRIBUTION_INSUFFICIENT_PERSISTENCE_ALERT`` consecutive cycles.
Attribution is graded once per weekly Saturday evaluation, so the default (4)
fires after ~1 month of unbroken under-powering — long enough to rule out a
transient cohort gap, short enough to surface a structural starvation. Tune the
constant if the cadence or tolerance changes (documented-default convention,
same as retrain_alert's thresholds).
"""

from __future__ import annotations

import json
import logging

import boto3

logger = logging.getLogger(__name__)

# Consecutive `insufficient_data` attribution cycles before the persistence
# warning fires. Attribution is graded once per weekly evaluation cycle, so 4 ≈
# one month of unbroken under-powering. See module docstring for the rationale.
ATTRIBUTION_INSUFFICIENT_PERSISTENCE_ALERT = 4

# Append-only history of per-cycle attribution adequacy, on the research bucket
# alongside the other decision artifacts. One JSON object per line.
HISTORY_KEY = "decision_artifacts/_attribution_adequacy/history.jsonl"

_INSUFFICIENT = "insufficient_data"


# ── Pure logic ───────────────────────────────────────────────────────────────

def count_trailing_insufficient(statuses: list[str]) -> int:
    """Length of the trailing run of ``insufficient_data`` statuses.

    Only the most-recent unbroken streak counts — a single ``ok`` cycle resets
    the persistence clock, because it proves the analysis CAN power up at the
    current cohort size.
    """
    streak = 0
    for status in reversed(statuses):
        if status == _INSUFFICIENT:
            streak += 1
        else:
            break
    return streak


def evaluate_attribution_persistence(
    statuses: list[str],
    threshold: int = ATTRIBUTION_INSUFFICIENT_PERSISTENCE_ALERT,
) -> dict:
    """Grade the trailing `insufficient_data` streak against ``threshold``.

    Args:
        statuses: per-cycle attribution statuses in chronological order
            (oldest first); the last element is the current cycle.
        threshold: consecutive-cycle count at/above which the warning fires.

    Returns a dict with ``consecutive_insufficient``, ``threshold``,
    ``persistent`` (bool), ``latest_status``, and ``cycles_observed``.
    """
    consecutive = count_trailing_insufficient(statuses)
    return {
        "consecutive_insufficient": consecutive,
        "threshold": threshold,
        "persistent": consecutive >= threshold,
        "latest_status": statuses[-1] if statuses else None,
        "cycles_observed": len(statuses),
    }


# ── S3 history (mirrors retrain_alert._write_alert_to_s3) ─────────────────────

def _load_history(bucket: str) -> list[dict]:
    """Read the append-only adequacy history (chronological). [] if absent."""
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=HISTORY_KEY)
        text = obj["Body"].read().decode()
    except Exception:  # noqa: BLE001 — first run / missing object → empty history
        return []
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("attribution_persistence: skipping malformed history line")
    return rows


def _persist_history(bucket: str, rows: list[dict]) -> None:
    """Overwrite the history object with ``rows`` (chronological JSONL)."""
    s3 = boto3.client("s3")
    body = "".join(json.dumps(r, default=str) + "\n" for r in rows)
    s3.put_object(
        Bucket=bucket,
        Key=HISTORY_KEY,
        Body=body.encode(),
        ContentType="application/jsonlines",
    )


def _entry(attribution: dict | None, run_date: str) -> dict:
    """Build the history entry for this cycle from a ``compute_attribution`` result."""
    attr = attribution or {}
    status = attr.get("status") or "skipped"
    # `compute_attribution` emits its finalized-row count as `rows_analyzed` when
    # ok and `rows_populated` when insufficient; record whichever is present so
    # the history carries the N alongside the status.
    n = attr.get("rows_analyzed", attr.get("rows_populated"))
    return {"date": run_date, "status": status, "n": n}


# ── Orchestration ────────────────────────────────────────────────────────────

def record_and_evaluate(
    attribution: dict | None,
    run_date: str,
    bucket: str,
    *,
    upload: bool = False,
    threshold: int = ATTRIBUTION_INSUFFICIENT_PERSISTENCE_ALERT,
) -> dict:
    """Append this cycle's attribution status to history and grade persistence.

    The current cycle is ALWAYS included in the persistence evaluation. It is
    only written back to S3 when ``upload`` is true (matching the evaluator's
    write-as-you-compute gating, so local / dry runs incur no S3 cost and don't
    pollute history). Same-``run_date`` re-runs are idempotent — a prior entry
    for the date is replaced rather than duplicated, so a Saturday retry can't
    inflate the streak.

    Returns the ``evaluate_attribution_persistence`` dict plus ``latest_n``.
    """
    history = _load_history(bucket)
    entry = _entry(attribution, run_date)

    # Idempotent by date: drop any existing same-date row, then append.
    history = [r for r in history if r.get("date") != run_date]
    history.append(entry)

    if upload:
        try:
            _persist_history(bucket, history)
        except Exception as exc:  # noqa: BLE001 — don't fail the eval on a write error
            logger.warning("attribution_persistence: history write failed: %s", exc)

    statuses = [r.get("status") for r in history]
    result = evaluate_attribution_persistence(statuses, threshold)
    result["latest_n"] = entry["n"]
    return result


# ── Report section (mirrors calibration_report.build_calibration_section) ─────

def render_attribution_persistence_section(result: dict) -> str:
    """Always-emit markdown section for the evaluator report."""
    consecutive = result.get("consecutive_insufficient", 0)
    threshold = result.get("threshold", ATTRIBUTION_INSUFFICIENT_PERSISTENCE_ALERT)
    latest = result.get("latest_status")
    n = result.get("latest_n")
    n_str = "N/A" if n is None else str(n)

    lines = ["## Attribution sample adequacy (persistence)", ""]

    if result.get("persistent"):
        lines += [
            f"- ⚠️ **Attribution under-powered for {consecutive} consecutive "
            f"cycles** (threshold {threshold}; latest N={n_str} vs floor 100).",
            "  The 21d-realized cohort is structurally too small for an "
            "FDR-robust sub-score attribution fit. Widen the cohort (more "
            "finalized signals reaching 21d realization) or revisit the floor in "
            "`analysis/sample_size_adequacy.py` / `analysis/attribution.py`.",
            "",
        ]
    elif latest == _INSUFFICIENT:
        lines += [
            f"- Attribution under-powered this cycle (N={n_str} vs floor 100), "
            f"{consecutive}/{threshold} consecutive — within tolerance, "
            "monitoring.",
            "",
        ]
    elif latest == "ok":
        lines += [
            f"- Attribution well-powered this cycle (N={n_str} ≥ floor 100). "
            "Persistence streak reset.",
            "",
        ]
    else:
        lines += [
            f"- Attribution not computed this cycle (status: `{latest}`). "
            "Persistence streak unchanged.",
            "",
        ]

    return "\n".join(lines)
