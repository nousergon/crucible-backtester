"""Pin the SSH→SSM transport migration in infrastructure/spot_backtest.sh.

Origin: ROADMAP L342 PR 3 — the 2026-05-27 SSH/SCP→SSM migration moved
all dispatcher→spot communication to the lib chokepoint
``python -m alpha_engine_lib.ssm_dispatcher`` (lib v0.35.0+). Mirrors
alpha-engine-data PR 2 (#330)'s test_spot_data_weekly_ssm_transport.py
1:1; without these chokepoint tests, a future refactor could silently
re-introduce SSH+SCP (the prior transport) and re-open the port-22
dependency the migration was designed to retire.

The shape of each test mirrors PR #322's
``TestDeployScriptsHaveNoEventBridgeWrites`` — a regex-based
"forbidden phrase" assertion on the deploy script's source. The lib
chokepoint is the canonical path; any reintroduction of SSH/SCP at the
top-level dispatch surface fails loud at PR time.

Closes the (i) alive-SSH-path finding from the 2026-05-24 audit for
the backtester repo (PR 3 of the 5-PR ROADMAP L342 arc). PR 4 will
follow this exact same pattern for predictor's ``spot_train.sh``
(retiring the inline ``run_ssm`` bash helper in favor of the lib
CLI); PR 5 will revoke the port-22 SG inbound rule once 1 clean
Saturday SF runs on the new transport across all three spots.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "infrastructure" / "spot_backtest.sh"


def _script_lines() -> list[tuple[int, str]]:
    """Return (line_no, line) tuples, comment lines stripped.

    Comment lines may legitimately reference SSH / SCP / port-22 in
    historical-context prose (e.g. "Replaces the pre-2026-05-27 SCP
    path"); only non-comment lines are subject to the forbidden-phrase
    chokepoint.
    """
    assert _SCRIPT.exists(), f"spot_backtest.sh missing at {_SCRIPT}"
    out: list[tuple[int, str]] = []
    for i, raw in enumerate(_SCRIPT.read_text().splitlines(), start=1):
        stripped = raw.strip()
        if stripped.startswith("#"):
            continue
        out.append((i, raw))
    return out


def test_spot_backtest_script_exists():
    """Guards against accidental script deletion. Without the script,
    the chokepoint assertions below silently no-op."""
    assert _SCRIPT.exists(), (
        f"infrastructure/spot_backtest.sh missing at {_SCRIPT}. "
        "This script drives the Saturday SF Backtester / Parity / "
        "Evaluator spots; CI cannot validate its SSM transport "
        "invariant without it."
    )


def test_no_top_level_ssh_invocation():
    """No ``ssh ...`` command at the top of any non-comment line.

    Replaces the pre-2026-05-27 SSH dispatch with ``python -m
    alpha_engine_lib.ssm_dispatcher`` (lib v0.35.0+). Any new ``ssh``
    invocation surfaces as an immediate red CI signal. Mirrors ae-data
    PR 2 (#330)'s chokepoint.
    """
    offenders = [
        (n, line)
        for n, line in _script_lines()
        if re.search(r"\bssh\s+-\w+", line) or re.search(r"^\s*ssh\s+", line)
    ]
    assert not offenders, (
        f"Found {len(offenders)} non-comment ``ssh`` invocations in "
        f"spot_backtest.sh:\n"
        + "\n".join(f"  line {n}: {line.strip()}" for n, line in offenders)
        + "\n\nThe 2026-05-27 SSH→SSM migration moved all dispatch to "
        "``python -m alpha_engine_lib.ssm_dispatcher``. Re-introducing "
        "ssh re-opens the port-22 dependency the migration retired. "
        "If the change is deliberate, update this test + ROADMAP L342 "
        "PR 5 (the planned port-22 SG revoke)."
    )


def test_no_top_level_scp_invocation():
    """No ``scp ...`` command at the top of any non-comment line.

    Replaces the pre-2026-05-27 SCP config upload (4 files: .env,
    config.yaml, risk.yaml, predictor.yaml) with the S3 staging pattern
    (dispatcher ``aws s3 cp`` to a temporary ``tmp/spot_backtest/``
    prefix, spot pulls via its existing ``alpha-engine-executor-profile``
    IAM role's ``s3:GetObject`` grant). Mirrors the
    alpha-engine-predictor #168 + alpha-engine-data #330 precedents.
    """
    offenders = [
        (n, line)
        for n, line in _script_lines()
        if re.search(r"\bscp\s+-\w+", line) or re.search(r"^\s*scp\s+", line)
    ]
    assert not offenders, (
        f"Found {len(offenders)} non-comment ``scp`` invocations in "
        f"spot_backtest.sh:\n"
        + "\n".join(f"  line {n}: {line.strip()}" for n, line in offenders)
        + "\n\nThe 2026-05-27 migration replaced SCP with S3 staging "
        "(4 files: .env + config.yaml + risk.yaml + predictor.yaml). "
        "Re-introducing scp re-opens the port-22 dependency."
    )


def test_no_ssh_keyscan_invocation():
    """No ``ssh-keyscan`` invocation — the pre-2026-05-27 bootstrap had
    ``ssh-keyscan github.com >> ~/.ssh/known_hosts`` to pre-seed the
    spot's known_hosts file. Post-migration the spot clones via HTTPS
    only (no host-key concern), and the dispatcher never SSHs in, so
    the keyscan step is dead code."""
    offenders = [
        (n, line)
        for n, line in _script_lines()
        if "ssh-keyscan" in line
    ]
    assert not offenders, (
        f"Found {len(offenders)} ``ssh-keyscan`` invocations in "
        f"spot_backtest.sh:\n"
        + "\n".join(f"  line {n}: {line.strip()}" for n, line in offenders)
    )


def test_uses_lib_ssm_dispatcher_chokepoint():
    """The migration's load-bearing surface: ``python -m
    alpha_engine_lib.ssm_dispatcher`` MUST appear in the script. Pinning
    this catches a regression where a future PR replaces the lib CLI
    with an inline ``aws ssm send-command`` bash helper (the pre-lift
    pattern L342 explicitly lifts to the lib chokepoint)."""
    body = _SCRIPT.read_text()
    assert "nousergon_lib.ssm_dispatcher" in body, (
        "spot_backtest.sh does not reference "
        "nousergon_lib.ssm_dispatcher. The 2026-05-27 migration uses "
        "the lib chokepoint as the SSM dispatch path (package renamed "
        "alpha_engine_lib -> nousergon_lib at lib 0.60.0; config#1245)."
    )
    assert "-m alpha_engine_lib." not in body, (
        "spot_backtest.sh still invokes 'python -m alpha_engine_lib.<mod>'. "
        "That deprecated meta-path alias shim lacks runpy's get_code, so "
        "'-m' dies under runpy on any box crossed to nousergon-lib "
        ">=0.60.x. Use '-m nousergon_lib.<mod>' (config#1245 / #1172)."
    )


def test_run_ssm_passes_diagnostics_flags():
    """L394 cascade — ``run_ssm`` MUST pass both ``--diagnostics-bucket``
    and ``--diagnostics-prefix`` so terminal non-Success in any spot
    SSM step writes a JSON failure record to
    ``s3://${S3_BUCKET}/_spot_diagnostics/ae-backtester/{date}.json`` per
    the lib v0.39.0 contract. Both flags must be present — lib's partial-
    config guard makes a missing flag a silent no-op."""
    body = _SCRIPT.read_text()
    assert "--diagnostics-bucket" in body, (
        "spot_backtest.sh does not pass --diagnostics-bucket to the "
        "lib CLI. L394 cascade requires both --diagnostics-bucket and "
        "--diagnostics-prefix together; without --diagnostics-bucket "
        "the lib's partial-config guard makes the diagnostics-write a "
        "silent no-op even on terminal non-Success."
    )
    assert "--diagnostics-prefix" in body, (
        "spot_backtest.sh does not pass --diagnostics-prefix to the lib CLI."
    )
    # Per-repo subprefix discriminates cascade A (ae-data) + cascade C
    # (ae-predictor) sibling writes — lib's {date}.json key shape would
    # otherwise clobber within a shared prefix.
    assert "_spot_diagnostics/ae-backtester" in body, (
        "spot_backtest.sh --diagnostics-prefix must scope to "
        "_spot_diagnostics/ae-backtester so ae-data + ae-predictor "
        "cascade siblings write to disjoint S3 namespaces."
    )


def test_no_inline_aws_ssm_send_command():
    """The script MUST NOT call ``aws ssm send-command`` directly — that
    bypasses the lib chokepoint and reverts to the pre-lift pattern.
    The lib CLI wraps that exact call with InvocationDoesNotExist
    registration grace, stdout streaming, and consistent S3 output-key
    layout; bypassing it loses those guarantees."""
    offenders = [
        (n, line)
        for n, line in _script_lines()
        if "aws ssm send-command" in line
    ]
    assert not offenders, (
        f"Found {len(offenders)} non-comment ``aws ssm send-command`` "
        f"invocations in spot_backtest.sh:\n"
        + "\n".join(f"  line {n}: {line.strip()}" for n, line in offenders)
        + "\n\nRoute through ``python -m alpha_engine_lib.ssm_dispatcher "
        "run`` instead."
    )


def test_stages_three_configs_via_s3():
    """The script MUST stage all 3 configs (config.yaml, risk.yaml,
    predictor.yaml) to S3 before dispatch. Without S3 staging, the spot
    has no path to read the dispatcher's private configs (no SCP, no
    shared filesystem). Pinning ALL 3 staging calls catches a regression
    that drops one but keeps the bootstrap fetch (which would then
    return NoSuchKey).

    Post-#890: ``.env`` (backtester.env) is no longer staged — its
    non-secret config (EMAIL_SENDER/EMAIL_RECIPIENTS/OUTPUT_BUCKET)
    moved into config.yaml and secrets load from SSM. The inverse guard
    below pins that the legacy ``.env`` staging stays gone."""
    body = _SCRIPT.read_text()
    for name in (
        "config.yaml",
        "risk.yaml",
        "predictor.yaml",
    ):
        # Each must appear in BOTH a dispatcher-side `aws s3 cp ... ${S3_STAGING}/<name>`
        # AND a bootstrap-side `aws s3 cp ${S3_STAGING}/<name> /home/...` fetch.
        # The simpler chokepoint: just assert the filename is in the script.
        assert name in body, (
            f"spot_backtest.sh does not reference {name!r}. The migration "
            f"requires staging 3 configs to S3; missing this name means "
            f"the bootstrap fetch will fail and the spot will run with a "
            f"missing config file."
        )
    # #890 cutover guard: the legacy .env staging must NOT reappear.
    assert "backtester.env" not in body, (
        "spot_backtest.sh still references 'backtester.env' — #890 "
        "deprecated the .env; its config moved to config.yaml and secrets "
        "to SSM. The .env must no longer be staged, fetched, or sourced."
    )


def test_no_residual_key_file_dispatch_use():
    """The pre-migration script referenced ``$KEY_FILE`` extensively for
    ssh + scp. Post-migration the SSH key file is no longer used for
    dispatch. Any remaining ``$KEY_FILE`` or ``$SSH_OPTS`` reference in
    a NON-COMMENT line means the migration is incomplete.

    Allow-list: the ``KEY_NAME`` variable for the ``lib.ec2_spot
    --key-name`` launch flag stays — that's a different concern
    (instance attribute, not dispatch transport).
    """
    forbidden = ["$KEY_FILE", "${KEY_FILE}", "$SSH_OPTS", "${SSH_OPTS}"]
    offenders: list[tuple[int, str]] = []
    for n, line in _script_lines():
        if any(token in line for token in forbidden):
            offenders.append((n, line))
    assert not offenders, (
        f"Found {len(offenders)} residual KEY_FILE / SSH_OPTS uses in "
        f"non-comment lines of spot_backtest.sh:\n"
        + "\n".join(f"  line {n}: {line.strip()}" for n, line in offenders)
    )


def test_exit_trap_tracks_last_ssm_description():
    """The pre-migration EXIT trap captured ``LAST_RUN_REMOTE_CMD`` for
    diagnostics. Post-migration the equivalent is ``LAST_SSM_DESC``
    (recorded by ``run_ssm`` each invocation). Pinning this catches a
    regression that drops the diagnostic surface (the L2246 EXIT-trap
    diagnostic + alpha_engine_lib.alerts fan-out depend on it)."""
    body = _SCRIPT.read_text()
    assert "LAST_SSM_DESC" in body, (
        "spot_backtest.sh does not track LAST_SSM_DESC — the EXIT trap "
        "loses its 'which step ran last' diagnostic surface. The L2246 "
        "diagnostic + alpha_engine_lib.alerts fan-out depend on it."
    )
