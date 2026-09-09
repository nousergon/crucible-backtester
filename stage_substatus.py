"""A stage's status is no better than the worst of its own sub-results.

`alpha-engine-config-I10198`, enforcing `sf-pipeline-policy.md` §2.3b, clause
`SFP-2.3b-stage-status-is-the-worst-substatus` (`nous-ergon-ops-PR1119`).

Root cause this exists
----------------------
Measured on the 2026-08-15 scheduled weekly run (execution
`54acfc69-…_f1036888-…`, read live 2026-09-08)::

    EvalRollingMean -> eval_rolling_mean_result.Payload
        status:        "OK"
        agent_quality: {"status": "ERROR",
                        "error": "'str' object has no attribute 'isoformat'"}

Identical on 2026-08-22. Those are exactly the two weeks with no
`AlphaEngine/Eval/agent_quality_score` datapoint at all — and the control
bands went on reporting `IN_CONTROL` against a fortnight-old point, because a
band evaluated over an empty window has no breach to find.

**Nothing was broken.** Every §2.3 degradation mechanism — the SF `Catch`, the
`Mark*Degraded` Pass states, `$.research_degraded_local`, the completion
marker's `status: DEGRADED` — keys off the STAGE's status, and the stage said
`OK`. A nested failure was never a rule being broken; it was a rule that never
applied.

Why this is not a list of known sub-keys
----------------------------------------
A hand-written set of sub-result names to check is the shape that let this
survive: `agent_quality` would have been added to it only after the week it
cost. The scan is structural — any nested mapping carrying its own ``status``
string, at any depth — so a sub-result added tomorrow is covered without an
edit here.

Why the vocabularies are CLOSED
-------------------------------
A status in neither vocabulary is reported as UNCLASSIFIED, never defaulted to
a pass. A checker that reads an unrecognised status as healthy goes quiet the
first time a producer invents a word for failure (`principles.md` §2.7).

Relationship to the fleet copy
------------------------------
MIRRORED from `crucible-research/stage_substatus.py`, byte-identical below the
module docstring. Second adoption, which `policy-shared-code` makes the trigger
to lift into `nousergon-lib` — filed rather than done here because that repo
was outside this change's blast radius. Until the lift lands, an edit to one
copy MUST be applied to the other.

`nous-ergon-ops/scripts/sf_substatus_honesty.py` is the CONFORMANCE probe for
the same clause — it reads a finished execution's history and names violators
after the fact. This module is the PRODUCER side: it stops the stage from
emitting the dishonest status in the first place. The two vocabularies below
are deliberately identical to that script's, and `nousergon-lib` is the lift
target that would remove the duplication (`policy-shared-code`); filed rather
than done here because that repo was outside this change's blast radius.
"""

from __future__ import annotations

import logging
from typing import Any

_LOGGER = logging.getLogger(__name__)

#: A status meaning "this did what it was asked to do".
PASS_STATUSES: frozenset[str] = frozenset({
    "OK", "ok", "PASS", "PASSED", "SUCCESS", "SUCCEEDED", "Success",
    "COMPLETE", "COMPLETED", "CLEAN", "EMPTY", "COVERED",
    "COVERED_NO_OUTPUT", "SKIPPED", "skipped", "NOOP", "DRY_RUN",
})

#: A status meaning "this did not do what it was asked to do, and said so".
#: A stage may not report a pass over one of these.
ERROR_STATUSES: frozenset[str] = frozenset({
    "ERROR", "error", "ERRORED", "FAIL", "FAILED", "FAILURE", "EXCEPTION",
    "CRASHED", "TIMEOUT", "TIMED_OUT", "ABORTED", "DENIED", "UNAVAILABLE",
})

#: A status that is neither a clean pass nor a failure: the producer did part
#: of its job and SAID SO. Named here, per producer, so the check neither
#: raises over them (that would degrade healthy pipelines — the
#: chronic-false-positive class `alpha-engine-config-I10130`) nor reports them
#: as "a word nobody classified" on every healthy run, which is the same
#: silence by another route.
#:
#: This set is the ONE deliberate divergence from
#: `nous-ergon-ops/scripts/sf_substatus_honesty.py`, which lists these as
#: `unclassified`. That is right for an after-the-fact probe a human reads and
#: wrong for a producer-side check that runs on every invocation.
DEGRADED_STATUSES: dict[str, str] = {
    # eval_rolling_mean_handler: `control_bands` when some combos failed —
    # the count is carried in the same dict and the stage's OWN status is
    # already PARTIAL in that case.
    "PARTIAL": "a producer that completed part of its work and reported how much",
    "partial": "lowercase spelling of PARTIAL",
    # evals/calibration_kappa: no paired reviews in the window. A real,
    # expected state of a young corpus, not a failure.
    "empty": "a producer whose input window was legitimately empty",
    # evals/calibration_kappa: cells present but below the n floor.
    "insufficient": "a producer with input below its own sufficiency floor",
    "DEGRADED": "a producer explicitly declaring itself degraded",
    "degraded": "lowercase spelling of DEGRADED",
}

#: Sub-result keys this derivation does not read, each with its reason.
#: An exclusion list with reasons beside it is the only kind that cannot widen
#: silently — a test asserts every entry carries one.
EXCLUDED_SUBRESULT_KEYS: dict[str, str] = {
    "stage_coverage": (
        "The stage's own coverage verdict (MISSING/STALE/UNMEASURED). Already "
        "read by the weekly coverage sweep and alarmed by "
        "alpha-engine-stage-coverage-absent / -findings over its own artifact; "
        "a second reader would page twice for one fact "
        "(alpha-engine-config-I10130). It is also an OBSERVER of this stage, "
        "not a sub-result OF it — letting it decide the stage's status would "
        "let the observer fail the thing it observes."
    ),
}

#: The status a stage reports when a sub-result errored. Chosen, not invented:
#: it is the value the SF `Catch` on each of these states already routes to a
#: `Mark*Degraded` Pass over.
ERROR_STATUS = "ERROR"
#: Sub-statuses in neither vocabulary do NOT change the stage's status and do
#: NOT raise — they are recorded on the payload under ``substatus_unclassified``
#: and logged at ERROR. Raising over an unrecognised WORD would degrade healthy
#: pipelines (the fleet already emits "PARTIAL", "insufficient", "empty"), which
#: is the chronic-false-positive class `alpha-engine-config-I10130` was filed
#: for; staying silent about it is how a producer's new word for failure goes
#: unnoticed. Naming it is the honest middle, and matches what
#: `sf_substatus_honesty.py` does with the same input.


class StageSubResultError(RuntimeError):
    """A sub-result reported a failure the enclosing stage would have hidden.

    RAISED rather than returned. A returned ``{"status": "ERROR"}`` dict is a
    *successful* Task completion to Step Functions and would never route
    through the state's non-blocking `Catch` — which is precisely how the
    2026-08-15 failure stayed invisible. Raising is what reaches
    `Mark*Degraded`, sets `$.research_degraded_local`, and makes the run end
    as DEGRADED rather than clean.
    """

    def __init__(self, stage: str, findings: list[dict[str, str]]) -> None:
        self.stage = stage
        self.findings = findings
        detail = "; ".join(
            f"{f['path']}={f['status']}" + (f" ({f['detail']})" if f.get("detail") else "")
            for f in findings
        )
        super().__init__(
            f"{stage}: {len(findings)} sub-result(s) reported a failure the "
            f"stage would otherwise have reported a pass over "
            f"(alpha-engine-config-I10198, sf-pipeline-policy §2.3b): {detail}"
        )


def find_failed_substatuses(result: Any, *, path: str = "") -> list[dict[str, str]]:
    """Every nested sub-result reporting a failure or an unknown status.

    Structural, not name-based: any mapping value carrying a string ``status``
    is a sub-result. A sub-result that named itself the failure is NOT
    descended into — its own children are that failure's detail, not new
    findings (mirrors `sf_substatus_honesty.py`).
    """
    findings: list[dict[str, str]] = []
    if not isinstance(result, dict):
        return findings
    for key, value in result.items():
        if key in EXCLUDED_SUBRESULT_KEYS or not isinstance(value, dict):
            continue
        child = f"{path}.{key}" if path else key
        status = value.get("status")
        if not isinstance(status, str):
            findings += find_failed_substatuses(value, path=child)
            continue
        if status in ERROR_STATUSES:
            detail = value.get("error") or value.get("reason") or value.get("message")
            findings.append({
                "path": child,
                "status": status,
                "detail": str(detail) if detail is not None else "",
                "kind": "error",
            })
            continue
        if status in DEGRADED_STATUSES:
            findings.append({
                "path": child,
                "status": status,
                "detail": DEGRADED_STATUSES[status],
                "kind": "degraded",
            })
        elif status not in PASS_STATUSES:
            findings.append({
                "path": child,
                "status": status,
                "detail": (
                    "status in none of PASS_STATUSES, DEGRADED_STATUSES or "
                    "ERROR_STATUSES"
                ),
                "kind": "unclassified",
            })
        findings += find_failed_substatuses(value, path=child)
    return findings


def enforce_worst_substatus(
    result: dict,
    *,
    stage: str,
    logger: logging.Logger | None = None,
) -> dict:
    """Downgrade ``result["status"]`` to the worst of its sub-results, and raise.

    Returns ``result`` unchanged when every sub-result passed, so a stage with
    no sub-results is a no-op.

    Args:
        result: the handler's terminal payload, sub-results included.
        stage: the SF stage name, for the log line and the exception.
        logger: the calling handler's logger.

    Returns:
        ``result``, when there is nothing to report.

    Raises:
        StageSubResultError: a sub-result reported a FAILURE. An
            unclassifiable status is recorded and logged, never raised over.
            The mutated ``result`` (status downgraded,
            ``substatus_findings`` attached) is logged at ERROR immediately
            before the raise, so the work that DID succeed is not lost from
            the record — the SF keeps only the error under the state's Catch
            ResultPath.
    """
    log = logger or _LOGGER
    findings = find_failed_substatuses(result)
    if not findings:
        return result

    result["substatus_findings"] = findings

    degraded = [f for f in findings if f["kind"] == "degraded"]
    if degraded:
        result["substatus_degraded"] = True
        log.warning(
            "[%s] %d sub-result(s) reported a NAMED partial outcome — the "
            "stage's status is unchanged, but the fact rides in "
            "`substatus_findings` rather than being invisible "
            "(alpha-engine-config-I10198): %s",
            stage, len(degraded),
            ", ".join(f"{f['path']}={f['status']!r}" for f in degraded),
        )

    unclassified = [f for f in findings if f["kind"] == "unclassified"]
    if unclassified:
        # Reported, never defaulted to a pass — and never RAISED. "PARTIAL",
        # "insufficient", "empty" and friends are real vocabulary this fleet
        # already emits; raising on them would degrade a healthy pipeline over
        # a word, which is the chronic-false-positive class. The same split
        # `nous-ergon-ops/scripts/sf_substatus_honesty.py` makes: findings
        # alert, unclassified are named so someone classifies them.
        result["substatus_unclassified"] = True
        log.error(
            "[%s] %d sub-result status(es) in neither PASS_STATUSES nor "
            "DEGRADED_STATUSES nor ERROR_STATUSES — classify them or this "
            "check is silent about "
            "them (alpha-engine-config-I10198): %s",
            stage, len(unclassified),
            ", ".join(f"{f['path']}={f['status']!r}" for f in unclassified),
        )

    errored = [f for f in findings if f["kind"] == "error"]
    if not errored:
        return result

    result["status"] = ERROR_STATUS
    log.error(
        "[%s] stage status downgraded to %s: %d sub-result(s) reported a "
        "failure this stage would have reported a pass over "
        "(alpha-engine-config-I10198, sf-pipeline-policy §2.3b). Full "
        "payload: %s",
        stage, ERROR_STATUS, len(errored), result,
    )
    raise StageSubResultError(stage, errored)
