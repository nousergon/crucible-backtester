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


def test_cleanup_fans_out_via_lib_alerts_cli():
    """L2246 SOTA upgrade per the CLAUDE.md sub-sub-rule (lift-to-lib for
    ≥2 consumers): the dispatcher cleanup must publish the diagnostic via
    `python -m krepis.alerts publish` so the operator gets an
    independent-channel alert (SNS + Telegram) regardless of whether the
    SF wrapper's own NotifyComplete/HandleFailure path fired. Pins the
    full invocation contract."""
    text = _read_script()
    m = re.search(r"^cleanup\(\) \{.*?^\}", text, re.MULTILINE | re.DOTALL)
    assert m, "no cleanup() function found — spot_backtest.sh structure changed"
    body = m.group(0)
    assert "krepis.alerts publish" in body, (
        "cleanup() must fan out the diagnostic via the krepis alerts CLI "
        "(L2246 SOTA upgrade — see CLAUDE.md sub-sub-rule). Mirrors the "
        "L117 'Lambda CI canary rollback should Telegram/email the "
        "operator on rollback' pattern via the canonical primitive. "
        "Target is krepis.alerts, NOT nousergon_lib.alerts (config#1339): "
        "the alerts module relocated to krepis (MIT) at nousergon-lib "
        "v0.66.0 and nousergon_lib.alerts is now a sys.modules re-export "
        "shim with no runpy __main__, so '-m nousergon_lib.alerts publish' "
        "is a SILENT no-op (exits 0, publishes nothing)."
    )
    # Guard the EXECUTED command (not comment text): the `python -m <mod>`
    # the dispatcher actually runs must be krepis.alerts, never the
    # runpy-silent nousergon_lib.alerts shim (config#1339).
    invoke = re.search(r'"\$_alert_python"\s+-m\s+(\S+)\s+publish', body)
    assert invoke, "no `\"$_alert_python\" -m <mod> publish` invocation in cleanup()"
    assert invoke.group(1) == "krepis.alerts", (
        f"cleanup() invokes `-m {invoke.group(1)}` — must be krepis.alerts; "
        f"'-m nousergon_lib.alerts' is a silent runpy no-op since the krepis "
        f"relocation at nousergon-lib v0.66.0 (config#1339)."
    )
    # L4485-b: severity is now variable — defaults to "error" (Telegram push),
    # downgraded to "warning" only on a recoverable spot reclaim about to relaunch.
    assert 'local _will_relaunch=0 _alert_sev="error"' in body, (
        "alerts.publish severity must default to error (Telegram push) — "
        "L4485-b made it a variable that only downgrades to warning on a "
        "recoverable spot reclaim."
    )
    assert '--severity "$_alert_sev"' in body, (
        "alerts.publish call must tag severity via $_alert_sev (error default, "
        "warning on a will-relaunch reclaim)."
    )
    assert "--source alpha-engine-backtester/spot_backtest.sh" in body, (
        "alerts.publish call must identify itself via --source so the "
        "operator can triage at a glance."
    )
    # Best-effort fallback — the alert is independent fan-out, not a
    # cleanup blocker. The wrapping `|| echo ...` keeps cleanup running
    # even when Python / lib / SNS / Telegram are all unreachable.
    assert "alerts.publish fan-out failed" in body or "|| true" in body, (
        "alerts.publish must be best-effort — never block cleanup on "
        "secondary surveillance failure."
    )


def test_lib_pin_at_least_v0_21_0():
    """The dispatcher's alerts.publish call requires alpha_engine_lib >=
    v0.21.0 (the version that ships the new `alerts` module + CLI). Pin
    the floor; lib bumps for unrelated reasons are fine but a downgrade
    below v0.21.0 would silently break the alert fan-out."""
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
        f"v0.21.0 floor required by the dispatcher's alerts.publish call. "
        f"Re-pin or remove the alerts call."
    )


def test_krepis_pinned_for_alerts_cli():
    """The alerts CLI the dispatcher invokes is now `python -m krepis.alerts`
    (config#1339): the module relocated to krepis (MIT) at nousergon-lib
    v0.66.0, and `nousergon_lib.alerts` is a runpy-silent re-export shim.
    krepis must therefore be a direct requirements pin so the CLI is present
    in the dispatcher venv — relying on the transitive nousergon-lib pull
    would leave the alert one un-pinned hop from silently disappearing."""
    from pathlib import Path

    reqs = (
        Path(__file__).resolve().parent.parent / "requirements.txt"
    ).read_text()
    assert re.search(r"^\s*krepis\b", reqs, re.MULTILINE), (
        "requirements.txt must pin `krepis` directly — it provides the "
        "`python -m krepis.alerts` CLI the dispatcher's cleanup fan-out "
        "invokes (config#1339)."
    )
