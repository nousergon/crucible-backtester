"""stance_distribution.py — weekly stance-distribution drift detection.

Phase 5 acceptance check from
``alpha-engine-docs/private/attractiveness-pillars-260520.md``. When
predictor #183's pillar-aware ``classify_stance`` is live, the overall
stance distribution across predictions must stay within ±2σ of the prior
4-week mean for each stance. A regression (e.g. pillar path collapses
all picks into one stance) surfaces here within one Saturday cycle
instead of via NAV trajectory weeks later.

Stance vocabulary: ``("momentum", "value", "quality", "catalyst")`` — the
4-class ``StanceLiteral`` from the stance-taxonomy arc (2026-05-11). A
stance count of 0 is valid (e.g. a no-catalyst week).

Inputs: ``predictor/predictions/{date}.json`` for the current Saturday +
the most recent prediction in each of the prior 4 ISO weeks.

Outputs: structured report dict + optional Telegram + SNS alert via
``nousergon_lib.alerts.publish`` on ``status="fail"``. Mirrors the
canonical lift-to-lib pattern from
``analysis/cost_report._publish_anomaly_alert``.

Opt out of the alert publish (e.g. for tests / replay) with
``ALPHA_ENGINE_STANCE_DRIFT_ALERT_DISABLED=1``.

Composes with:
  - alpha-engine-predictor #183 — emits ``stance`` + ``stance_source`` on
    each ``predictions/{date}.json`` entry; this is the upstream signal.
  - alpha-engine-research _check_pillar_distribution_sanity — same
    defense-in-depth class, observes upstream pillar-emission failures
    in the score_aggregator node; this module observes the downstream
    expression in the predictor's stance assignment.
  - nousergon_lib.alerts (v0.21.0) — the canonical SNS+Telegram
    publish primitive; same lift-to-lib pattern as
    ``cost_report._publish_anomaly_alert``.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import math
import os
import statistics
from typing import Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


# 4-class stance vocabulary (predictor StanceLiteral). Count = 0 is valid.
KNOWN_STANCES: tuple[str, ...] = ("momentum", "value", "quality", "catalyst")

# Σ ± k×σ tolerance band — 2σ is the plan-doc acceptance criterion.
SIGMA_BAND: float = 2.0

# σ floor (in count units). Prevents alerts on tiny natural variation
# when prior weeks had identical stance counts (σ=0 + drift of 1 would
# otherwise fire). 1.0 = "one pick of slack" is the institutional default.
SIGMA_FLOOR: float = 1.0

# Default baseline window (ISO weeks before the current run).
DEFAULT_BASELINE_WEEKS: int = 4

# Minimum baseline weeks required to compute a meaningful σ.
MIN_BASELINE_WEEKS: int = 4

_ALERT_DISABLED_ENV_VAR: str = "ALPHA_ENGINE_STANCE_DRIFT_ALERT_DISABLED"


def _list_prediction_dates(bucket: str, s3_client=None) -> list[_dt.date]:
    """List all ``predictor/predictions/{YYYY-MM-DD}.json`` dates in the bucket."""
    s3 = s3_client or boto3.client("s3")
    dates: list[_dt.date] = []
    continuation: Optional[str] = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": "predictor/predictions/"}
        if continuation:
            kwargs["ContinuationToken"] = continuation
        try:
            resp = s3.list_objects_v2(**kwargs)
        except ClientError as e:
            logger.warning("Failed to list prediction files: %s", e)
            return []
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json") or "latest" in key:
                continue
            stem = key.split("/")[-1].replace(".json", "")
            try:
                dates.append(_dt.date.fromisoformat(stem))
            except ValueError:
                continue
        if not resp.get("IsTruncated"):
            break
        continuation = resp.get("NextContinuationToken")
    return sorted(dates)


def _select_baseline_dates(
    all_dates: list[_dt.date], current: _dt.date, n_weeks: int,
) -> list[_dt.date]:
    """Pick the most-recent prediction date in each of the prior n_weeks ISO weeks.

    Predictor runs daily Mon–Fri so the raw key list has up to 5 dates per
    week. Saturday SF reads the most recent prediction, so the comparable
    baseline is "one prediction per prior week" — taking the latest in
    each prior ISO week handles holiday weeks (Mon–Thu only) and missed
    Fridays naturally.

    Returns the picked dates sorted ascending.
    """
    by_week: dict[tuple[int, int], _dt.date] = {}
    for d in all_dates:
        if d >= current:
            continue
        iso = d.isocalendar()
        key = (iso.year, iso.week)
        if key not in by_week or d > by_week[key]:
            by_week[key] = d
    sorted_weeks = sorted(by_week.items(), key=lambda kv: kv[0], reverse=True)[:n_weeks]
    return sorted([d for _, d in sorted_weeks])


def _load_stance_counts(
    bucket: str, dates: list[_dt.date], s3_client=None,
) -> dict[_dt.date, dict[str, int]]:
    """Load ``predictor/predictions/{date}.json`` for each date and count stances.

    Returns ``{date: {stance: count, ...}}``. Stances missing from a
    prediction are not counted; an entirely-empty file or absent
    ``stance`` field yields a dict of zeros for KNOWN_STANCES.

    Defensive: dates whose S3 object is missing or unparseable are
    skipped with a WARN, not raised — a one-off corrupted file should
    not block the whole weekly check.
    """
    s3 = s3_client or boto3.client("s3")
    out: dict[_dt.date, dict[str, int]] = {}
    for d in dates:
        key = f"predictor/predictions/{d.isoformat()}.json"
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj["Body"].read())
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                logger.warning("predictions/%s.json absent — skipping", d.isoformat())
            else:
                logger.warning("S3 error loading %s: %s", key, e)
            continue
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Could not parse %s: %s", key, e)
            continue
        counts = {s: 0 for s in KNOWN_STANCES}
        for pred in data.get("predictions", []) or []:
            stance = pred.get("stance")
            if stance in counts:
                counts[stance] += 1
        out[d] = counts
    return out


def _check_within_band(
    current_counts: dict[str, int],
    baseline_counts: list[dict[str, int]],
    sigma_band: float = SIGMA_BAND,
    sigma_floor: float = SIGMA_FLOOR,
) -> dict[str, dict]:
    """Per-stance ±sigma_band check.

    Returns ``{stance: {current, baseline_mean, baseline_std, lower_bound,
    upper_bound, within_band, deviation}}``. ``deviation`` is signed
    z-score (current - mean) / max(std, sigma_floor); ``within_band`` is
    ``abs(deviation) <= sigma_band``.
    """
    result: dict[str, dict] = {}
    for stance in KNOWN_STANCES:
        baseline_series = [b.get(stance, 0) for b in baseline_counts]
        mean = statistics.fmean(baseline_series) if baseline_series else 0.0
        # Sample stddev needs ≥2 points; below that, treat as σ=0 and
        # let sigma_floor carry.
        if len(baseline_series) >= 2:
            std = statistics.stdev(baseline_series)
        else:
            std = 0.0
        effective_std = max(std, sigma_floor)
        lower = mean - sigma_band * effective_std
        upper = mean + sigma_band * effective_std
        current = float(current_counts.get(stance, 0))
        deviation = (current - mean) / effective_std
        result[stance] = {
            "current": int(current),
            "baseline_mean": round(mean, 3),
            "baseline_std": round(std, 3),
            "effective_std": round(effective_std, 3),
            "lower_bound": round(lower, 3),
            "upper_bound": round(upper, 3),
            "within_band": (lower <= current <= upper),
            "deviation": round(deviation, 3),
        }
    return result


def _publish_drift_alert(report: dict) -> None:
    """Fan a stance-drift FAIL out to SNS + Telegram via the canonical
    ``nousergon_lib.alerts`` lift-to-lib primitive (v0.21.0+).

    Mirrors ``cost_report._publish_anomaly_alert``. Best-effort —
    import + transport failures swallow at WARN; the load-bearing
    surface is the WARN log + the structured report dict returned to
    the evaluator caller.

    Opt-out via ``ALPHA_ENGINE_STANCE_DRIFT_ALERT_DISABLED=1`` (used by
    tests to avoid real boto3 reachability).
    """
    if os.environ.get(_ALERT_DISABLED_ENV_VAR, "").strip().lower() in (
        "1", "true", "yes", "on",
    ):
        return
    try:
        from nousergon_lib import alerts  # noqa: PLC0415 — lazy import
    except ImportError as e:
        logger.warning(
            "[stance_distribution] alerts publish skipped — nousergon_lib.alerts "
            "unavailable (lib pin <v0.21.0?): %s", e,
        )
        return
    failures = report.get("failures", []) or []
    current_date = report.get("current_date", "?")
    baseline_dates = report.get("baseline_dates", []) or []
    per_stance = report.get("per_stance", {}) or {}
    failure_lines = []
    for stance in failures:
        s = per_stance.get(stance, {})
        failure_lines.append(
            f"{stance} current={s.get('current')} "
            f"baseline_mean={s.get('baseline_mean')} "
            f"σ={s.get('baseline_std')} "
            f"band=[{s.get('lower_bound')},{s.get('upper_bound')}] "
            f"z={s.get('deviation')}"
        )
    message = (
        f"Stance-distribution drift on {current_date}: "
        f"{len(failures)} of {len(KNOWN_STANCES)} stances breached "
        f"±{SIGMA_BAND}σ vs {len(baseline_dates)}-week baseline "
        f"({', '.join(d for d in baseline_dates)}). "
        + " | ".join(failure_lines)
        + ". Phase 5 acceptance check; investigate "
        "classify_stance pillar-vs-heuristic path."
    )
    try:
        result = alerts.publish(
            message,
            severity="error",
            source="alpha-engine-backtester/analysis/stance_distribution.py",
        )
        logger.info(
            "[stance_distribution] drift alert publish: sns_ok=%s "
            "telegram_ok=%s any_ok=%s",
            result.sns.ok, result.telegram.ok, result.any_ok,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[stance_distribution] drift alert publish failed "
            "(best-effort, swallowed): %s", e,
        )


def compute_stance_distribution_drift(
    bucket: str,
    current_date: str,
    *,
    n_baseline_weeks: int = DEFAULT_BASELINE_WEEKS,
    sigma_band: float = SIGMA_BAND,
    sigma_floor: float = SIGMA_FLOOR,
    s3_client=None,
    publish_alert: bool = True,
) -> dict:
    """Run the Phase 5 stance-distribution drift check for ``current_date``.

    Loads the current Saturday SF's prediction file + the most recent
    prediction in each of the prior ``n_baseline_weeks`` ISO weeks,
    counts the 4-class stance distribution per file, computes the
    baseline mean + sample stddev, and asserts the current week's count
    is within ±``sigma_band`` × max(σ, ``sigma_floor``) for every
    stance.

    Returns ``{status, ...}`` with status ∈ ``{"ok", "fail",
    "insufficient_data", "error"}``. On ``"fail"`` and unless
    ``publish_alert=False``, fires a Telegram + SNS alert via
    ``nousergon_lib.alerts.publish``.
    """
    try:
        current = _dt.date.fromisoformat(current_date)
    except ValueError:
        return {
            "status": "error",
            "note": f"current_date {current_date!r} is not ISO YYYY-MM-DD",
        }

    all_dates = _list_prediction_dates(bucket, s3_client=s3_client)
    if not all_dates:
        return {
            "status": "insufficient_data",
            "note": "no predictions/{date}.json keys found in bucket",
            "current_date": current_date,
        }

    baseline_dates = _select_baseline_dates(all_dates, current, n_baseline_weeks)
    counts = _load_stance_counts(
        bucket, baseline_dates + [current], s3_client=s3_client,
    )

    current_counts = counts.get(current)
    baseline_counts = [counts[d] for d in baseline_dates if d in counts]

    if current_counts is None:
        return {
            "status": "insufficient_data",
            "note": f"predictions/{current_date}.json absent or unparseable",
            "current_date": current_date,
            "baseline_dates_found": [d.isoformat() for d in baseline_dates if d in counts],
        }

    if len(baseline_counts) < MIN_BASELINE_WEEKS:
        return {
            "status": "insufficient_data",
            "note": (
                f"only {len(baseline_counts)} baseline week(s) found "
                f"(need ≥{MIN_BASELINE_WEEKS}); cannot compute meaningful σ"
            ),
            "current_date": current_date,
            "baseline_dates_found": [d.isoformat() for d in baseline_dates if d in counts],
            "current_distribution": current_counts,
        }

    per_stance = _check_within_band(
        current_counts, baseline_counts, sigma_band=sigma_band, sigma_floor=sigma_floor,
    )
    failures = [s for s, info in per_stance.items() if not info["within_band"]]

    report = {
        "status": "fail" if failures else "ok",
        "current_date": current_date,
        "n_baseline_weeks": len(baseline_counts),
        "baseline_dates": [d.isoformat() for d in baseline_dates if d in counts],
        "current_distribution": current_counts,
        "sigma_band": sigma_band,
        "sigma_floor": sigma_floor,
        "per_stance": per_stance,
        "failures": failures,
    }

    if failures and publish_alert:
        _publish_drift_alert(report)

    return report
