"""One resolver for the ``run_date`` every stage-coverage verdict is keyed by.

Root cause this exists (alpha-engine-config-I10171, I10194)
-----------------------------------------------------------
Five weekly-SF stages wrote their ``_stage_coverage`` verdict into the
CALENDAR-date partition rather than the cycle's TRADING-day partition. Each
handler derived its own key from ``event["end_time_iso"]`` — the raw
``$$.Execution.StartTime`` the Step Function forwards verbatim — because
its Payload carried no ``run_date`` at all. On a Saturday cycle the calendar
date is 2026-09-05 while the trading day is 2026-09-04, so the verdicts
landed under a partition no reader looks at. ``nousergon_lib.pipeline_status
.partition``'s dual-partition fallback expired on 2026-09-05 by design, and
on that day all five stages started reading **absent** — which is
byte-identical, on every console, to a stage that never ran.

The contract this module enforces
---------------------------------
``event["run_date"]`` — the cycle's trading day, threaded by the SF from
``NormalizeRunDates`` — is the ONLY correct grouping key, and it is PREFERRED
whenever present. A handler-local derivation (``end_time_iso``'s date part,
``$.eval_cadence.eval_date``, ...) is a FALLBACK for the off-cycle operator
invocation only, and using it is a **recorded** event, never a silent one:

  * a WARNING-or-worse log line naming the stage, the fallback source and the
    value it produced, and
  * a ``run_date_source`` / ``run_date_fallback_reason`` pair written into the
    handler's returned ``stage_coverage`` dict, which the Step Function's own
    execution history retains.

A handler that quietly used the calendar date where the trading date was
expected is the whole defect; a fallback nobody can see afterwards
reproduces it.

Never fabricates. With neither the preferred field nor a usable fallback the
resolver returns ``None`` and the caller records ``UNMEASURED``
(alpha-engine-config-I8155) — it does not substitute ``date.today()``.
"""

from __future__ import annotations

import logging
from typing import Any

# MIRRORED from `crucible-research/stage_coverage_run_date.py`, byte-identical
# below this comment. This is the SECOND adoption, which `policy-shared-code`
# makes the trigger to lift it into `nousergon-lib` — filed rather than done
# here: `nousergon-lib` was outside this change's blast radius, and a fork
# left undetected is how the two copies drift. Until the lift lands, any edit
# to one copy MUST be applied to the other.
_LOGGER = logging.getLogger(__name__)

#: Value of ``run_date_source`` when the SF-threaded trading day was used
#: under its default field name. Other preferred fields report ``event.<name>``.
SOURCE_EVENT_RUN_DATE = "event.run_date"
#: Value of ``run_date_source`` when nothing usable was on the event.
SOURCE_NONE = "none"


def _clean(value: Any) -> str | None:
    """A non-blank ISO-ish date string, or None. Never raises on a non-str."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    # `$$.Execution.StartTime` arrives as a full ISO timestamp; the partition
    # key is its date part. A bare `YYYY-MM-DD` passes through unchanged.
    return text[:10]


def resolve_stage_run_date(
    event: Any,
    *,
    stage: str,
    preferred_fields: tuple[str, ...] = ("run_date",),
    fallback: Any = None,
    fallback_source: str | None = None,
    logger: logging.Logger | None = None,
) -> tuple[str | None, dict[str, str]]:
    """Resolve the partition key for ``stage``'s coverage verdict.

    Args:
        event: the Lambda event (any mapping; a non-mapping is treated as
            carrying no fields rather than raising — the caller's own
            required-field guards own that failure).
        stage: the weekly-SF stage name, for the log line and the record.
        preferred_fields: event fields, in order, that the Step Function
            DECLARES carry the cycle's trading day. Defaults to
            ``("run_date",)``. Several states thread the same
            ``$.run_date`` under a differently-named key — ``ChallengerShadow``,
            ``ResearchSelfTest`` and ``AggregateCosts`` all use
            ``date.$: "$.run_date"`` — and those are PREFERRED values, not
            fallbacks. Naming them here is what keeps the fallback warning
            meaningful: a warning that fires on every healthy cycle is a
            detector nobody reads.
        fallback: the handler's own pre-existing derivation (e.g. the date
            part of ``end_time_iso``). Used ONLY when ``event["run_date"]``
            is absent, and recorded loudly when it is.
        fallback_source: a short name for where ``fallback`` came from, e.g.
            ``"event.end_time_iso"``. Required whenever ``fallback`` is
            given — an unnamed fallback is an unrecorded one.
        logger: the calling handler's logger, so the line lands in that
            handler's log stream. Defaults to this module's.

    Returns:
        ``(run_date, provenance)``. ``run_date`` is ``None`` when nothing
        usable was available. ``provenance`` is always a dict carrying
        ``run_date_source`` and, on the fallback path, a
        ``run_date_fallback_reason`` — merge it into the returned
        ``stage_coverage`` dict verbatim.

    Raises:
        ValueError: a ``fallback`` was supplied without a ``fallback_source``.
            Programmer error at the call site, caught by the contract test —
            raised rather than defaulted, because a fallback whose origin is
            unnamed defeats the entire point of this resolver.
    """
    log = logger or _LOGGER
    if fallback is not None and not fallback_source:
        raise ValueError(
            f"resolve_stage_run_date({stage!r}) was given a fallback with no "
            "fallback_source — an unnamed fallback is an unrecorded one "
            "(alpha-engine-config-I10171)."
        )

    if isinstance(event, dict):
        for field in preferred_fields:
            preferred = _clean(event.get(field))
            if preferred:
                return preferred, {"run_date_source": f"event.{field}"}

    resolved_fallback = _clean(fallback)
    if resolved_fallback:
        reason = (
            f"event carried no run_date; fell back to {fallback_source} "
            f"= {resolved_fallback!r}. On the live weekly-SF path this is a "
            "DEFECT — the SF Payload must thread run_date.$: '$.run_date' "
            "(the cycle's trading day). Off-cycle operator invocations are "
            "the only legitimate case (alpha-engine-config-I10171)."
        )
        log.warning("[stage_coverage] %s: %s", stage, reason)
        return resolved_fallback, {
            "run_date_source": f"fallback:{fallback_source}",
            "run_date_fallback_reason": reason,
        }

    log.error(
        "[stage_coverage] %s: no run_date and no usable fallback on this "
        "event — verdict will be UNMEASURED, never fabricated "
        "(alpha-engine-config-I8155)",
        stage,
    )
    return None, {"run_date_source": SOURCE_NONE}
