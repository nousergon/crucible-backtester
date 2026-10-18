"""
completeness.py — Per-module data completeness tracking for the evaluator.

Each evaluation module reports what data it had available and whether it ran
fully, in degraded mode (missing some inputs), was skipped, or errored.
The CompletenessTracker aggregates these results into a manifest that gets
written to S3 alongside the evaluation report.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ModuleResult:
    """Result of running a single evaluation module."""

    name: str
    status: str  # "ok" | "degraded" | "skipped" | "error"
    inputs_available: dict[str, bool] = field(default_factory=dict)
    inputs_missing: list[str] = field(default_factory=list)
    degradation_reason: str | None = None
    result: dict = field(default_factory=dict)
    duration_seconds: float = 0.0


#: Self-reported ``status`` values that mean "ran, but on no usable input".
#: Graded ``degraded``, never ``ok`` (alpha-engine-config-I11506).
_NO_INPUT_STATUSES = frozenset({
    "degraded", "partial", "incomplete", "skipped", "unreadable",
    "db_not_found", "alpha_unmeasured", "unmeasured",
    # A measurement the module itself declared impossible: the lib's
    # ``regime_sortino.STATUS_UNMEASURABLE`` (inputs past the horizon's
    # freshness allowance) was graded ``[OK]`` on the 2026-09-26 and
    # 2026-10-10 weekly runs because it was outside this vocabulary.
    "unmeasurable", "unservable", "vacuous", "not_run",
})
_NO_INPUT_PREFIXES = ("insufficient", "no_", "missing", "stale", "skipped_")
_NO_INPUT_SUFFIXES = ("_absent", "_unparseable")
#: ``no_*`` statuses that are a computed DECISION on real inputs, not an
#: absence of inputs (an optimizer that measured and chose not to move).
_DECISION_STATUSES = frozenset({"no_change", "no_improvement"})
#: Self-reported failure returned (not raised) by a module.
_ERROR_STATUSES = frozenset({"error", "failed", "fail"})


def grade_self_reported(output: Any) -> tuple[str | None, str | None]:
    """Grade a module's own ``status`` field.

    Returns ``(grade, reason)`` where ``grade`` is ``"degraded"`` or
    ``"error"``, or ``(None, None)`` when the output does not self-report a
    no-input / failure status. Statuses outside the vocabulary above
    (``ok``, optimizer decisions such as ``blocked``, ``retired``,
    ``accruing``) are left to the caller's grade.
    """
    if not isinstance(output, dict):
        return None, None
    raw = output.get("status")
    if not isinstance(raw, str):
        return None, None
    s = raw.strip().lower()
    if s in _ERROR_STATUSES:
        grade = "error"
    elif s in _DECISION_STATUSES:
        return None, None
    elif (
        s in _NO_INPUT_STATUSES
        or s.startswith(_NO_INPUT_PREFIXES)
        or s.endswith(_NO_INPUT_SUFFIXES)
    ):
        grade = "degraded"
    else:
        return None, None
    detail = (
        output.get("reason") or output.get("status_reason")
        or output.get("degraded_reason")
        or output.get("error") or output.get("note")
    )
    reason = f"module reported status={raw}"
    if detail:
        reason += f" ({str(detail)[:200]})"
    return grade, reason


#: Status handed back to the caller for a module that returned ``None`` (or
#: another empty non-dict). Matches ``_NO_INPUT_PREFIXES`` so anything that
#: re-grades the returned dict reads it as degraded too.
NO_OUTPUT_STATUS = "no_output"


def _is_empty_output(output: Any) -> bool:
    """True when a module returned nothing to grade: ``None``, or an empty
    dict / list / tuple / string. A non-empty non-dict (e.g. a row list) is
    output and is left alone."""
    if output is None:
        return True
    if isinstance(output, (dict, list, tuple, set, str, bytes)):
        return len(output) == 0
    return False


class CompletenessTracker:
    """Tracks completeness across all evaluation modules."""

    def __init__(self) -> None:
        self._results: list[ModuleResult] = []

    def record(self, result: ModuleResult) -> None:
        self._results.append(result)
        level = logging.INFO if result.status in ("ok", "degraded") else logging.WARNING
        msg = f"[{result.status.upper()}] {result.name} ({result.duration_seconds:.1f}s)"
        if result.inputs_missing:
            msg += f" — missing: {', '.join(result.inputs_missing)}"
        if result.degradation_reason:
            msg += f" — {result.degradation_reason}"
        logger.log(level, msg)

    def run_module(
        self,
        name: str,
        fn: Callable[..., dict],
        required_inputs: dict[str, bool],
        skip_if_missing: list[str] | None = None,
        stale_inputs: dict[str, str | None] | None = None,
    ) -> dict:
        """Run a module function with completeness tracking.

        Args:
            name: Module name for reporting.
            fn: Callable that returns a result dict. Called with no args.
            required_inputs: {input_name: is_available} map.
            skip_if_missing: Input names that cause a skip (not error) when missing.
            stale_inputs: {input_name: staleness_reason} for inputs that are
                PRESENT but whose newest row is too old for the run to mean
                anything. A truthy reason grades an otherwise-ok module
                ``degraded`` (alpha-engine-config-I11506): a module computing
                ``ok`` from months-old rows is a silent no-op, and most modules
                carry no freshness self-check of their own. ``None`` means the
                input was checked and is fresh.

        Returns:
            The module's result dict (or a status dict on skip/error).
        """
        missing = [k for k, v in required_inputs.items() if not v]

        # Skip if critical inputs are missing
        if skip_if_missing:
            skip_missing = [m for m in skip_if_missing if m in missing]
            if skip_missing:
                result = ModuleResult(
                    name=name,
                    status="skipped",
                    inputs_available=required_inputs,
                    inputs_missing=missing,
                    degradation_reason=f"required input(s) unavailable: {', '.join(skip_missing)}",
                    result={"status": "skipped"},
                )
                self.record(result)
                return result.result

        t0 = time.time()
        try:
            output = fn()
            duration = time.time() - t0

            reasons = [f"ran without: {', '.join(missing)}"] if missing else []
            status = "ok" if not missing else "degraded"
            stale = {k: v for k, v in (stale_inputs or {}).items() if v}
            if stale:
                reasons.extend(f"stale input {k}: {v}" for k, v in stale.items())
                status = "degraded"
            # The module's OWN verdict outranks "it returned without raising"
            # (alpha-engine-config-I11506): a module that ran on zero rows,
            # stale sources or a missing benchmark and said so must not be
            # graded ok.
            self_grade, self_reason = grade_self_reported(output)
            if self_grade is not None:
                reasons.append(self_reason)
                if self_grade == "error" or status == "ok":
                    status = self_grade
            # A module that returned NOTHING did nothing the grader can see
            # (alpha-engine-config-I11506). ``None`` / ``{}`` / ``[]`` used to
            # be graded ``ok`` and — for a non-dict — handed back to the
            # caller as a fabricated ``{"status": "ok"}``, so a module whose
            # body silently fell through to an implicit ``return None``
            # rendered as a pass in completeness.json and the report header.
            if _is_empty_output(output):
                reasons.append(f"module returned no output ({type(output).__name__})")
                if status == "ok":
                    status = "degraded"
                if not isinstance(output, dict):
                    output = {"status": NO_OUTPUT_STATUS,
                              "reason": "module returned no output"}
            returned = output if isinstance(output, dict) else {"status": "ok"}
            result = ModuleResult(
                name=name,
                status=status,
                inputs_available=required_inputs,
                inputs_missing=missing,
                degradation_reason="; ".join(reasons) or None,
                result=returned,
                duration_seconds=round(duration, 2),
            )
            self.record(result)
            return returned

        except Exception as e:
            duration = time.time() - t0
            result = ModuleResult(
                name=name,
                status="error",
                inputs_available=required_inputs,
                inputs_missing=missing,
                degradation_reason=str(e),
                result={"status": "error", "error": str(e)},
                duration_seconds=round(duration, 2),
            )
            self.record(result)
            logger.error("%s failed: %s", name, e)
            return result.result

    @property
    def results(self) -> list[ModuleResult]:
        return list(self._results)

    def summary(self) -> dict[str, Any]:
        """Aggregate completeness summary."""
        counts = {"ok": 0, "degraded": 0, "skipped": 0, "error": 0}
        for r in self._results:
            counts[r.status] = counts.get(r.status, 0) + 1
        return {
            "total": len(self._results),
            **counts,
            "total_duration_seconds": round(sum(r.duration_seconds for r in self._results), 2),
        }

    def degraded_modules(self) -> list[str]:
        return [r.name for r in self._results if r.status == "degraded"]

    def failed_modules(self) -> list[str]:
        return [r.name for r in self._results if r.status == "error"]

    def to_dict(self) -> dict:
        """Full completeness manifest as a dict."""
        return {
            "summary": self.summary(),
            "modules": [
                {
                    "name": r.name,
                    "status": r.status,
                    "inputs_available": r.inputs_available,
                    "inputs_missing": r.inputs_missing,
                    "degradation_reason": r.degradation_reason,
                    "duration_seconds": r.duration_seconds,
                }
                for r in self._results
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)
