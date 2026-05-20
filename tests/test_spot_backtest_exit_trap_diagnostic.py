"""Pins the EXIT-trap diagnostic in `infrastructure/spot_backtest.sh`.

L2246 regression guard. The dispatcher's `trap cleanup EXIT` terminates the
spot instance after every run; on failure the operator could only see "ssh
died abruptly" / "watchdog fired" / "clean exit but marker upload failed"
as indistinguishable single lines. The diagnostic adds three observability
fields before termination:
  * the dispatcher's exit code (``$?``)
  * the last `run_remote` command invoked (truncated to 200 chars to keep
    heredoc dumps bounded)
  * the spot instance's current state (via `describe-instances`)

Composes with the SSH-keepalive arc closed at PR #103 — that fixed the
silent-kill case at the connection layer; this names the failure mode in
the dispatcher log.
"""

from __future__ import annotations

import re
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "infrastructure" / "spot_backtest.sh"
)


def _read_script() -> str:
    return _SCRIPT.read_text()


def test_last_run_remote_cmd_declared():
    text = _read_script()
    assert 'LAST_RUN_REMOTE_CMD=""' in text, (
        "LAST_RUN_REMOTE_CMD must be declared before `trap cleanup EXIT` so "
        "the diagnostic can name the last remote call on failure (L2246)."
    )


def test_run_remote_records_last_cmd():
    text = _read_script()
    m = re.search(r"^run_remote\(\) \{.*?^\}", text, re.MULTILINE | re.DOTALL)
    assert m, "no run_remote() helper found — spot_backtest.sh structure changed"
    body = m.group(0)
    assert "LAST_RUN_REMOTE_CMD=" in body, (
        "run_remote() must record its invocation in LAST_RUN_REMOTE_CMD so the "
        "EXIT-trap diagnostic can name it on failure (L2246)."
    )


def test_cleanup_prints_exit_code_and_diagnostic_on_failure():
    text = _read_script()
    m = re.search(r"^cleanup\(\) \{.*?^\}", text, re.MULTILINE | re.DOTALL)
    assert m, "no cleanup() function found — spot_backtest.sh structure changed"
    body = m.group(0)
    assert "exit_code=$?" in body, (
        "cleanup() must capture $? on entry to report the dispatcher's exit "
        "status in the diagnostic (L2246)."
    )
    assert 'if [ "$exit_code" -ne 0 ]' in body, (
        "cleanup() must gate the diagnostic block on non-zero exit so "
        "successful runs stay quiet (L2246)."
    )
    assert "last run_remote" in body, (
        "cleanup() must print the last run_remote command on failure (L2246)."
    )
    assert "describe-instances" in body and "State.Name" in body, (
        "cleanup() must print the spot instance state on failure (L2246)."
    )


def test_cleanup_still_terminates():
    """The diagnostic must not break the original cleanup contract — terminate
    the spot instance even when the diagnostic block runs."""
    text = _read_script()
    m = re.search(r"^cleanup\(\) \{.*?^\}", text, re.MULTILINE | re.DOTALL)
    assert m
    body = m.group(0)
    assert "aws ec2 terminate-instances" in body, (
        "cleanup() must still terminate the instance after the diagnostic "
        "(don't trade cost-guard for observability)."
    )
