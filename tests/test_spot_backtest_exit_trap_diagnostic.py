"""Pins the EXIT-trap diagnostic in `infrastructure/spot_backtest.sh`.

L2246 regression guard. The dispatcher's `trap cleanup EXIT` terminates the
spot instance after every run; on failure the operator could only see "ssh
died abruptly" / "watchdog fired" / "clean exit but marker upload failed"
as indistinguishable single lines. The diagnostic adds three observability
fields before termination:
  * the dispatcher's exit code (``$?``)
  * the last remote command invoked (truncated to keep dumps bounded)
  * the spot instance's current state (via `describe-instances`)

Composes with the SSH-keepalive arc closed at PR #103 — that fixed the
silent-kill case at the connection layer; this names the failure mode in
the dispatcher log.

2026-05-27 SSH→SSM migration (ROADMAP L342 PR 3) renamed the diagnostic
variable from ``LAST_RUN_REMOTE_CMD`` to ``LAST_SSM_DESC`` and the
helper from ``run_remote()`` to ``run_ssm()``. The contract is the
same (record + print the last dispatched call on failure); only the
spelling changed. These tests now accept either shape so the
diagnostic invariant survives the transport flip.
"""

from __future__ import annotations

import re
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "infrastructure" / "spot_backtest.sh"
)


def _read_script() -> str:
    return _SCRIPT.read_text()


def test_last_dispatched_command_var_declared():
    """Accept either LAST_RUN_REMOTE_CMD (pre-SSM) or LAST_SSM_DESC
    (post-SSM, L342 PR 3) — both spellings of the same diagnostic
    invariant. Bare ``=""`` declaration before the cleanup trap."""
    text = _read_script()
    assert (
        'LAST_RUN_REMOTE_CMD=""' in text or 'LAST_SSM_DESC=""' in text
    ), (
        "Either LAST_RUN_REMOTE_CMD (SSH transport) or LAST_SSM_DESC (SSM "
        "transport) must be declared before `trap cleanup EXIT` so the "
        "diagnostic can name the last dispatched call on failure (L2246)."
    )


def test_dispatch_helper_records_last_call():
    """The dispatch helper — ``run_remote()`` (SSH) or ``run_ssm()``
    (SSM) — must record its invocation into the diagnostic variable so
    the EXIT-trap can surface it on failure."""
    text = _read_script()
    for helper, var in (
        ("run_remote", "LAST_RUN_REMOTE_CMD"),
        ("run_ssm", "LAST_SSM_DESC"),
    ):
        m = re.search(
            rf"^{helper}\(\) \{{.*?^\}}", text, re.MULTILINE | re.DOTALL
        )
        if m:
            body = m.group(0)
            assert f"{var}=" in body, (
                f"{helper}() must record its invocation in {var} so the "
                f"EXIT-trap diagnostic can name it on failure (L2246)."
            )
            return
    raise AssertionError(
        "neither run_remote() nor run_ssm() helper found in "
        "spot_backtest.sh — the dispatch surface is gone"
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
    # Accept either "last run_remote" (SSH transport) or "last run_ssm"
    # (SSM transport, L342 PR 3) — both name the same diagnostic surface.
    assert "last run_remote" in body or "last run_ssm" in body, (
        "cleanup() must print the last dispatched call on failure (L2246). "
        "Either 'last run_remote' (SSH) or 'last run_ssm' (SSM)."
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


def test_cleanup_fans_out_via_ops_alerts():
    """T3 config#1749: dispatcher cleanup publishes via ops_alerts.publish_ops_alert
    (SNS + flow-doctor forum topics) instead of raw krepis CLI telegram=True."""
    text = _read_script()
    m = re.search(r"^cleanup\(\) \{.*?^\}", text, re.MULTILINE | re.DOTALL)
    assert m, "no cleanup() function found — spot_backtest.sh structure changed"
    body = m.group(0)
    assert "from ops_alerts import publish_ops_alert" in body, (
        "cleanup() must fan out via ops_alerts.publish_ops_alert (config#1749 T3)."
    )
    assert "publish_ops_alert(" in body
    assert "cd \"$REPO_ROOT\"" in body, (
        "cleanup() must run the ops_alerts import from REPO_ROOT so ops_alerts.py resolves."
    )
    assert 'local _will_relaunch=0 _alert_sev="error"' in body, (
        "ops alert severity must default to error — downgraded to warning only on "
        "a recoverable spot reclaim about to relaunch (L4485-b)."
    )
    assert 'source=\'alpha-engine-backtester/spot_backtest.sh\'' in body, (
        "publish_ops_alert must identify itself via source for operator triage."
    )
    assert "ops alert fan-out failed" in body or "|| true" in body, (
        "ops alert must be best-effort — never block cleanup on secondary failure."
    )


def test_lib_pin_at_least_v0_21_0():
    """The dispatcher's ops_alerts fan-out requires nousergon-lib >= v0.21.0
    (ships the alerts module). Pin the floor; downgrades below v0.21.0 break SNS."""
    from pathlib import Path

    reqs = (
        Path(__file__).resolve().parent.parent / "requirements.txt"
    ).read_text()
    # Dist name renamed alpha-engine-lib -> nousergon-lib at lib 0.60.0
    # (config#1245). Accept either spelling.
    m = re.search(r"(?:alpha-engine-lib|nousergon-lib)\[[^\]]+\]\s*@\s*git\+https://[^@]+@v(\d+)\.(\d+)\.(\d+)", reqs)
    assert m, "no nousergon-lib version pin found in requirements.txt"
    major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
    assert (major, minor, patch) >= (0, 21, 0), (
        f"alpha-engine-lib pin v{major}.{minor}.{patch} is below the "
        f"v0.21.0 floor required by ops_alerts SNS fan-out. "
        f"Re-pin or remove the alert call."
    )


def test_krepis_pinned_for_alerts_transport():
    """krepis remains a direct pin — ops_alerts.publish_ops_alert calls
    krepis.alerts.publish internally (config#1749 T3)."""
    from pathlib import Path

    reqs = (
        Path(__file__).resolve().parent.parent / "requirements.txt"
    ).read_text()
    assert re.search(r"^\s*krepis\b", reqs, re.MULTILINE), (
        "requirements.txt must pin `krepis` directly — ops_alerts uses "
        "krepis.alerts.publish for the SNS leg (config#1749)."
    )
