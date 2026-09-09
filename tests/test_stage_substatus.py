"""A stage's status is no better than the worst of its own sub-results.

`alpha-engine-config-I10198`, enforcing `sf-pipeline-policy.md` §2.3b, clause
`SFP-2.3b-stage-status-is-the-worst-substatus` (`nous-ergon-ops-PR1119`).

The measured instance was `EvalRollingMean` in `crucible-research`; this repo's
two replay Lambdas are the same SHAPE — a fan-out returning one enclosing
status — and are held to the clause here so the class does not reopen on the
next sub-result added to either.

Asserted against `tests/fixtures/sf_substatus/weekly_2026-08-15_scheduled.json`,
a verbatim copy of the 76 real stage results `nous-ergon-ops-PR1119` froze from
execution `54acfc69-…_f1036888-…`, so all three repos assert against ONE
artifact rather than three drifting reproductions.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "sf_substatus" / "weekly_2026-08-15_scheduled.json"


@pytest.fixture(scope="module")
def sub():
    spec = importlib.util.spec_from_file_location(
        "_i10198_stage_substatus", _REPO_ROOT / "stage_substatus.py"
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["_i10198_stage_substatus"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def frozen_history() -> list[dict]:
    return json.loads(_FIXTURE.read_text())


def test_the_fixture_carries_no_infra_identifier():
    """This repo is PUBLIC. The frozen history came from a live execution and
    carried the AWS account id and an IAM role name in an `AccessDenied`
    message; both are redacted here and nothing else is. `repository-tiering-
    policy` applies a PURPOSE test, not a secrecy test — an identifier has no
    public job in a test fixture, whether or not it is exploitable on its own.
    Asserted so a future refresh from the ops copy cannot quietly restore them.
    """
    raw = _FIXTURE.read_text()
    assert "REDACTED-ACCOUNT-ID" in raw
    assert "REDACTED-ROLE" in raw
    assert not re.search(r"\b\d{12}\b", raw), "a 12-digit AWS account id is in the fixture"


def test_the_fixture_is_the_shape_that_cost_two_weeks(frozen_history):
    payload = next(
        v["Payload"]
        for row in frozen_history if row["state"] == "EvalRollingMean"
        for v in row["result"].values()
        if isinstance(v, dict) and isinstance(v.get("Payload"), dict)
    )
    assert payload["status"] == "OK"
    assert payload["agent_quality"]["status"] == "ERROR"
    assert "isoformat" in payload["agent_quality"]["error"]


def test_that_payload_is_now_a_stage_failure(sub, frozen_history):
    payload = dict(next(
        v["Payload"]
        for row in frozen_history if row["state"] == "EvalRollingMean"
        for v in row["result"].values()
        if isinstance(v, dict) and isinstance(v.get("Payload"), dict)
    ))
    with pytest.raises(sub.StageSubResultError) as excinfo:
        sub.enforce_worst_substatus(payload, stage="EvalRollingMean")
    assert [f["path"] for f in excinfo.value.findings] == ["agent_quality"]


def test_only_the_errored_substatus_fails_the_stage(sub, frozen_history):
    """Precision over 76 real results carrying `insufficient`, `PARTIAL`,
    `MEASURED` and `STALE`. A check that fired on healthy neighbours would be
    suppressed within a week."""
    failed = []
    for row in frozen_history:
        for value in row["result"].values():
            payload = value.get("Payload") if isinstance(value, dict) else None
            if not isinstance(payload, dict) or payload.get("status") not in sub.PASS_STATUSES:
                continue
            try:
                sub.enforce_worst_substatus(dict(payload), stage=row["state"])
            except sub.StageSubResultError:
                failed.append(row["state"])
    assert failed == ["EvalRollingMean"], failed


def test_a_word_no_vocabulary_knows_is_reported_never_defaulted_to_pass(sub, caplog):
    with caplog.at_level(logging.ERROR):
        out = sub.enforce_worst_substatus({"status": "OK", "t": {"status": "wobbly"}}, stage="X")
    assert out["substatus_unclassified"] is True
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


def test_stage_coverage_is_excluded_with_its_reason(sub):
    assert sub.EXCLUDED_SUBRESULT_KEYS["stage_coverage"].strip()
    assert sub.find_failed_substatuses(
        {"status": "OK", "stage_coverage": {"status": "UNMEASURED"}}
    ) == []


# ── The class, derived from the repo rather than listed here ────────────────


def _lambda_handlers_returning_a_status() -> list[str]:
    found = []
    for directory in sorted(_REPO_ROOT.glob("lambda_*")):
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            if "assert_stage_coverage(" in path.read_text():
                found.append(str(path.relative_to(_REPO_ROOT)))
    return found


def test_the_scan_is_not_vacuous():
    assert len(_lambda_handlers_returning_a_status()) >= 2


@pytest.mark.parametrize("filename", _lambda_handlers_returning_a_status())
def test_every_lambda_handler_enforces_the_worst_substatus(filename):
    source = (_REPO_ROOT / filename).read_text()
    assert "enforce_worst_substatus" in source, (
        f"{filename} returns a status over sub-results it never checks — the "
        "alpha-engine-config-I10198 swallow (sf-pipeline-policy §2.3b)"
    )


def test_the_two_copies_of_the_derivation_have_not_drifted(sub):
    """`policy-shared-code`: this module is MIRRORED from
    `crucible-research/stage_substatus.py` until the `nousergon-lib` lift
    lands. The vocabularies are the part that must not drift — a status
    classified as a failure in one repo and a pass in the other is worse than
    either choice made consistently."""
    assert "PARTIAL" in sub.DEGRADED_STATUSES
    assert "ERROR" in sub.ERROR_STATUSES
    assert "OK" in sub.PASS_STATUSES
    assert not (sub.PASS_STATUSES & sub.ERROR_STATUSES)
    assert not (set(sub.DEGRADED_STATUSES) & sub.ERROR_STATUSES)
