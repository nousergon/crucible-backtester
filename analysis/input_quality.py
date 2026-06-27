"""
analysis/input_quality.py — pre-spend signal input-quality gate (L4525, plan
§6 Phase 3 / §2 P5).

Before the backtester burns ~121 min of spot on a param-sweep, assert the
signal INPUTS are sane. The failure this guards against: a `signals.json`
source that degenerates to a wall of `Score 0.0` (or to no signals at all)
produces an empty sweep — which the L4523 outcome taxonomy correctly treats as
a *valid no-op*, because a legitimately quiet market ALSO yields an empty
sweep. Without an input gate the two are indistinguishable, so a
garbage-input run silently no-ops and starves the Evaluator for weeks (the
L4521/L4529 silent-starvation failure mode) with no error.

This gate closes that gap by classifying the inputs as healthy or degenerate
BEFORE the spend:

  * UNAMBIGUOUS garbage — no usable signals across the sampled dates, signals
    present but none carry a numeric score (schema break), or effectively
    every score is exactly ``0.0`` — sets ``verdict.healthy = False``. When the
    caller is in *enforce* mode it raises ``InputQualityError`` (fail loud,
    pre-spend). These conditions cannot fire on a quiet-but-healthy week (whose
    scores still span ~45–90), so enforcing them carries ~zero false-positive
    risk.
  * SOFT degradations — an elevated-but-not-total zero-score fraction, or low
    universe coverage — are recorded as ``observations`` and logged/alerted,
    never raised. They are the tuning signal for the eventual Score-0.0 root
    fix and a future stricter enforce threshold.

The pure assessor (``assess_signal_quality``) is side-effect free so it
unit-tests directly; the loader-backed ``gate_signal_inputs`` adds the S3 read
+ logging + enforce decision. Per [[feedback_no_silent_fails]] +
[[feedback_sota_institutional_default_no_shortcuts]].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping

logger = logging.getLogger(__name__)


class InputQualityError(RuntimeError):
    """Raised by ``gate_signal_inputs`` (enforce mode) when the signal inputs
    are unambiguous garbage — no usable signals or a wall of Score 0.0 — so the
    pipeline fails loud BEFORE spending ~121 min on a doomed param-sweep."""


@dataclass(frozen=True)
class InputQualityVerdict:
    """Structured, JSON-serializable result of a signal input-quality
    assessment. ``healthy`` is the enforce-level verdict (False ⇒ unambiguous
    garbage ⇒ a caller in enforce mode raises); ``observations`` are soft
    degradations that are always logged but never raised."""

    healthy: bool
    reason: str
    metrics: dict = field(default_factory=dict)
    observations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "healthy": self.healthy,
            "reason": self.reason,
            "metrics": dict(self.metrics),
            "observations": list(self.observations),
        }


def _coerce_score(value) -> float | None:
    """Return ``value`` as a float, or None if it is not a finite number.

    Signal scores are emitted as ints (0–100) but tolerate float/str-numeric;
    a non-numeric or missing score means "this entry carries no score" — it is
    NOT silently treated as 0.0 (that would mask the schema-break case the
    gate exists to catch)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    # NaN / inf are not valid scores.
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def assess_signal_quality(
    per_date_signals: Mapping[str, dict],
    *,
    zero_score_garbage_fraction: float = 0.99,
    elevated_zero_observe_fraction: float = 0.10,
    low_coverage_observe_fraction: float = 0.25,
) -> InputQualityVerdict:
    """Classify a sample of loaded ``signals.json`` payloads as healthy or
    unambiguous-garbage, with soft observations.

    ``per_date_signals`` maps a date string → the parsed ``signals.json`` dict
    (as returned by ``loaders.signal_loader.load``). Pure: it reads only the
    provided dicts and returns a verdict — no I/O, no logging.

    Garbage (``healthy = False``), any of:
      1. No date in the sample has a non-empty ``signals`` mapping.
      2. Signals are present but NONE carry a numeric score (schema break).
      3. ``zero_score_fraction >= zero_score_garbage_fraction`` — effectively
         every scored entry is ``<= 0`` (the wall-of-Score-0.0 case).

    Observations (logged, not raised) when healthy:
      * ``elevated_zero_observe_fraction <= zero_score_fraction <
        zero_score_garbage_fraction`` — a non-total but elevated zero-score
        fraction (the Score-0.0 root tuning signal).
      * mean per-date coverage (scored entries / universe size) below
        ``low_coverage_observe_fraction``.
    """
    dates_checked = len(per_date_signals)
    dates_with_signals = 0
    total_entries = 0
    scored_entries = 0
    zero_score_entries = 0
    distinct_nonzero: set[float] = set()
    coverage_ratios: list[float] = []

    for _date, payload in per_date_signals.items():
        signals = (payload or {}).get("signals") or {}
        # signals.json uses a {ticker: {...}} mapping; tolerate a list too.
        if isinstance(signals, dict):
            entries = list(signals.values())
        elif isinstance(signals, list):
            entries = list(signals)
        else:
            entries = []
        if entries:
            dates_with_signals += 1
        total_entries += len(entries)

        date_scored = 0
        for entry in entries:
            score = _coerce_score((entry or {}).get("score"))
            if score is None:
                continue
            date_scored += 1
            scored_entries += 1
            if score <= 0:
                zero_score_entries += 1
            else:
                distinct_nonzero.add(score)

        universe = (payload or {}).get("universe") or []
        if isinstance(universe, (list, tuple, set)) and len(universe) > 0:
            coverage_ratios.append(date_scored / len(universe))

    zero_score_fraction = (
        zero_score_entries / scored_entries if scored_entries else 0.0
    )
    mean_coverage = (
        sum(coverage_ratios) / len(coverage_ratios) if coverage_ratios else None
    )

    metrics = {
        "dates_checked": dates_checked,
        "dates_with_signals": dates_with_signals,
        "total_entries": total_entries,
        "scored_entries": scored_entries,
        "zero_score_entries": zero_score_entries,
        "zero_score_fraction": round(zero_score_fraction, 4),
        "distinct_nonzero_scores": len(distinct_nonzero),
        "mean_coverage": round(mean_coverage, 4) if mean_coverage is not None else None,
    }

    # ── Unambiguous-garbage conditions (enforce-level) ──────────────────────
    if dates_checked == 0:
        return InputQualityVerdict(
            healthy=False,
            reason="no signal dates were sampled — cannot assess input quality",
            metrics=metrics,
        )
    if dates_with_signals == 0:
        return InputQualityVerdict(
            healthy=False,
            reason=(
                f"no signals present in any of the {dates_checked} sampled "
                f"signal dates — the signal source is empty"
            ),
            metrics=metrics,
        )
    if total_entries > 0 and scored_entries == 0:
        return InputQualityVerdict(
            healthy=False,
            reason=(
                f"{total_entries} signal entries present across "
                f"{dates_with_signals} dates but NONE carry a numeric score — "
                f"signals.json score field is missing/non-numeric (schema break)"
            ),
            metrics=metrics,
        )
    if scored_entries > 0 and zero_score_fraction >= zero_score_garbage_fraction:
        return InputQualityVerdict(
            healthy=False,
            reason=(
                f"wall of Score 0.0: {zero_score_entries}/{scored_entries} "
                f"({zero_score_fraction:.1%}) scored entries are <= 0 across "
                f"{dates_with_signals} dates (>= "
                f"{zero_score_garbage_fraction:.0%} threshold) — the param-sweep "
                f"would empty-out on garbage input, not a quiet market"
            ),
            metrics=metrics,
        )

    # ── Healthy, with soft observations ─────────────────────────────────────
    observations: list[str] = []
    if scored_entries > 0 and zero_score_fraction >= elevated_zero_observe_fraction:
        observations.append(
            f"elevated zero-score fraction {zero_score_fraction:.1%} "
            f"({zero_score_entries}/{scored_entries}) — above the "
            f"{elevated_zero_observe_fraction:.0%} observe threshold; possible "
            f"upstream score degradation (L4525 Score-0.0 root)"
        )
    if mean_coverage is not None and mean_coverage < low_coverage_observe_fraction:
        observations.append(
            f"low mean universe coverage {mean_coverage:.1%} "
            f"(< {low_coverage_observe_fraction:.0%} observe threshold) — "
            f"scored entries thin relative to the universe"
        )

    return InputQualityVerdict(
        healthy=True,
        reason=(
            f"signal inputs healthy: {scored_entries} scored entries across "
            f"{dates_with_signals}/{dates_checked} dates, zero-score fraction "
            f"{zero_score_fraction:.1%}, {len(distinct_nonzero)} distinct "
            f"non-zero scores"
        ),
        metrics=metrics,
        observations=observations,
    )


def gate_signal_inputs(
    bucket: str,
    dates,
    *,
    signal_loader,
    sample_recent: int = 20,
    enforce: bool = False,
    zero_score_garbage_fraction: float = 0.99,
    elevated_zero_observe_fraction: float = 0.10,
    low_coverage_observe_fraction: float = 0.25,
    alert_publisher=None,
    log=logger,
) -> InputQualityVerdict:
    """Load a sample of the most recent ``dates`` and assess signal-input
    quality before the expensive simulation/param-sweep spend.

    Always computes + logs the verdict (observe). When ``enforce`` is True and
    the verdict is unambiguous garbage, raises ``InputQualityError`` so the run
    fails loud in seconds instead of silently no-opping after ~121 min.

    ``signal_loader`` must expose ``load(bucket, date) -> dict`` (the repo's
    ``loaders.signal_loader``). ``alert_publisher`` is an optional
    ``callable(message, severity, source)`` (``nousergon_lib.alerts.publish``
    in production) — best-effort observability; a publish failure never alters
    the gate decision. Returns the verdict on pass.
    """
    all_dates = list(dates or [])
    sample = all_dates[-sample_recent:] if sample_recent and sample_recent > 0 else all_dates

    per_date: dict[str, dict] = {}
    load_failures: list[str] = []
    for d in sample:
        try:
            per_date[d] = signal_loader.load(bucket, d)
        except FileNotFoundError as e:
            # list_dates head-checks existence, so a miss here is unexpected —
            # record it (don't silently drop) and keep assessing what loaded.
            load_failures.append(str(d))
            log.warning("[input_quality] signals.json missing for sampled date %s: %s", d, e)

    verdict = assess_signal_quality(
        per_date,
        zero_score_garbage_fraction=zero_score_garbage_fraction,
        elevated_zero_observe_fraction=elevated_zero_observe_fraction,
        low_coverage_observe_fraction=low_coverage_observe_fraction,
    )

    # If EVERY sampled date failed to load, that is itself garbage (the sample
    # was non-empty but nothing came back) — override a "no dates sampled"
    # verdict with the more specific load-failure cause.
    if sample and not per_date:
        verdict = InputQualityVerdict(
            healthy=False,
            reason=(
                f"all {len(sample)} sampled signal dates failed to load "
                f"(missing signals.json) — cannot validate input quality"
            ),
            metrics={"dates_checked": 0, "load_failures": len(load_failures)},
        )

    log.info(
        "[input_quality] verdict=%s enforce=%s sampled=%d loaded=%d %s",
        "healthy" if verdict.healthy else "GARBAGE",
        enforce,
        len(sample),
        len(per_date),
        verdict.metrics,
    )
    for obs in verdict.observations:
        log.warning("[input_quality] observation: %s", obs)

    if not verdict.healthy:
        log.warning("[input_quality] DEGENERATE signal inputs: %s", verdict.reason)
        if alert_publisher is not None:
            try:
                alert_publisher(
                    message=(
                        f"Backtester input-quality gate flagged DEGENERATE signal "
                        f"inputs (enforce={enforce}): {verdict.reason}"
                    ),
                    severity="warning",
                    source="alpha-engine-backtester/analysis/input_quality.py",
                )
            except Exception:  # noqa: BLE001 — alert is best-effort observability
                pass
        if enforce:
            raise InputQualityError(verdict.reason)

    return verdict
