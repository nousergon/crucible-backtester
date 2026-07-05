"""Pin the #883 mid-run spot-reclaim relaunch adoption in spot_backtest.sh.

Origin: nousergon/alpha-engine-config#883 — the 2026-05-30 Saturday SF failed
in DataPhase1 when a nested data spot was reclaimed by AWS *mid-workload*.
The relaunch fix shipped first in alpha-engine-data #349, then a divergent
bespoke copy landed here (L4485-b, #283/#289: an inline
StateReason.Code/StateTransitionReason classifier + a decrementing
RECLAIM_RELAUNCH_MAX budget) while the predictor launcher had no relaunch at
all. Per #883, the classify→decide DECISION is lifted into the lib
chokepoint ``python -m krepis.ec2_spot relaunch-decision`` (lib v0.65.0+;
already satisfied by this repo's nousergon-lib@v0.78.0 / krepis>=0.4.0
pins). This PR migrates spot_backtest.sh off the inline classifier onto the
shared chokepoint, mirroring alpha-engine-predictor#308's adoption — but
invokes it via ``krepis.ec2_spot`` directly (NOT ``nousergon_lib.ec2_spot``,
which PR #308 used): config#1649/#1646 (2026-07-03) established that on lib
>=0.81.0 the ``nousergon_lib.*`` re-export shims are guard-less no-ops under
``python -m``, and this repo's existing
``test_spot_backtest_krepis_cli_executes.py`` already enforces the
``krepis.*`` convention for this launcher's other lib CLI callsites.

These tests pin that adoption so a future refactor can't silently revert it:
  * the lib ``relaunch-decision`` chokepoint is the DECISION surface (NOT a
    re-introduced inline StateReason.Code/StateTransitionReason classifier)
  * ``cleanup()`` captures the exit status FIRST and re-exits with it (so a
    recovered cleanup path never masks a real failure as rc=0)
  * the describe-instances call that remains is for stdout DIAGNOSTICS only
    (state/reason_code/state_reason are still logged + fanned out via
    krepis.alerts) — it must not feed a re-inlined relaunch decision
  * the relaunch ``exec``s a FRESH spot with the SAME argv, threading
    SPOT_ATTEMPT, bounded by MAX_SPOT_ATTEMPTS
  * the lib decision (which internally classifies via describe-instances)
    happens BEFORE terminate-instances, the relaunch exec AFTER teardown
  * only a confirmed reclaim relaunches — fail-loud is preserved because the
    lib classifies a genuine workload failure as "other"/"unknown" and exits
    NO_RELAUNCH_EXIT_CODE (no blind retry)
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "infrastructure" / "spot_backtest.sh"


def _cleanup_body() -> str:
    text = _SCRIPT.read_text()
    m = re.search(r"^cleanup\(\)\s*\{.*?^\}", text, re.MULTILINE | re.DOTALL)
    assert m, "no cleanup() function found in spot_backtest.sh"
    return m.group(0)


def test_script_exists():
    assert _SCRIPT.exists(), f"spot_backtest.sh missing at {_SCRIPT}"


def test_script_syntactically_valid():
    """``bash -n`` must accept the script — the relaunch edits add a command
    substitution + array re-exec that must parse under bash."""
    r = subprocess.run(["bash", "-n", str(_SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, f"bash -n rejected spot_backtest.sh:\n{r.stderr}"


def test_orig_args_captured_before_parse():
    """``_ORIG_ARGS`` must be captured BEFORE the flag-parse loop consumes
    "$@", so the relaunch re-execs with the IDENTICAL argv (--mode /
    --instance-type / --smoke-only etc.). Capturing after the parse loop
    (which shifts every arg away) would relaunch with an EMPTY argv."""
    text = _SCRIPT.read_text()
    cap = text.find('_ORIG_ARGS=("$@")')
    parse = text.find("while [[ $# -gt 0 ]]; do")
    assert cap != -1, (
        "spot_backtest.sh does not capture _ORIG_ARGS — the #883 relaunch "
        "cannot re-exec with the original argv."
    )
    assert parse != -1, "flag-parse loop (while [[ $# -gt 0 ]]) not found"
    assert cap < parse, (
        "_ORIG_ARGS must be captured BEFORE the flag-parse loop — capturing "
        "after the loop relaunches with an empty argv and silently drops "
        "--mode / --instance-type / other flags."
    )


def test_cleanup_uses_lib_relaunch_decision_chokepoint():
    """The relaunch DECISION MUST come from the lib chokepoint
    ``python -m krepis.ec2_spot relaunch-decision`` (lib v0.65.0+),
    NOT a re-introduced inline reclaim classifier. The lib owns the classify
    (describe-instances) + the bounded/SF-coupled decision; the launcher
    branches on the exit code."""
    body = _cleanup_body()
    assert "krepis.ec2_spot relaunch-decision" in body, (
        "cleanup() does not call the lib chokepoint "
        "`python -m krepis.ec2_spot relaunch-decision`. #883 lifts the "
        "classify->decide logic into the lib (krepis.ec2_spot); the launcher "
        "must consume it, not re-inline a StateReason.Code grep."
    )
    for flag in ("--instance-id", "--attempt", "--max-attempts"):
        assert flag in body, (
            f"cleanup() relaunch-decision call is missing {flag} — the lib "
            "decision needs the 1-based attempt + total-attempts budget."
        )
    assert "nousergon_lib.ec2_spot relaunch-decision" not in body, (
        "cleanup() calls `nousergon_lib.ec2_spot relaunch-decision` — on lib "
        ">=0.81.0 that module is a guard-less re-export shim that silently "
        "no-ops under `python -m` (config#1649/#1646). Must invoke "
        "`krepis.ec2_spot relaunch-decision` directly, matching this "
        "launcher's other lib CLI callsites."
    )


def test_cleanup_no_inline_reclaim_decision_classifier():
    """The launcher MUST NOT re-inline the spot-reclaim relaunch DECISION
    (that's exactly what #883 lifted to the lib). The old bespoke
    convention gated the relaunch on a hand-rolled `_is_reclaim` flag set by
    grepping Server.SpotInstanceTermination / a decrementing
    RECLAIM_RELAUNCH_MAX budget — that is the divergent-copy regression the
    chokepoint exists to prevent. A plain describe-instances call for stdout
    DIAGNOSTICS is fine and expected; a decision gate built from it is not."""
    forbidden = (
        "_is_reclaim",
        "RECLAIM_RELAUNCH_MAX",
    )
    offenders = []
    for i, raw in enumerate(_SCRIPT.read_text().splitlines(), start=1):
        stripped = raw.strip()
        if stripped.startswith("#"):
            continue
        for tok in forbidden:
            if tok in raw:
                offenders.append((i, tok, stripped))
    assert not offenders, (
        "cleanup() re-inlines the reclaim relaunch decision instead of using "
        "the lib chokepoint:\n"
        + "\n".join(f"  line {n}: {tok} :: {ln}" for n, tok, ln in offenders)
        + "\n\nRoute through `python -m krepis.ec2_spot relaunch-decision`."
    )


def test_cleanup_captures_and_reexits_status():
    """cleanup() must capture ``exit_code=$?`` as its FIRST statement and end
    on ``exit "$exit_code"`` so a recovered cleanup path (echos + `|| true`
    teardown all succeed) can never mask a real workload failure as rc=0 to
    the cron/orchestration wrapper (the L4485 status-masking class)."""
    body = _cleanup_body()
    first = body.split("{", 1)[1].lstrip().splitlines()
    first_stmt = next(
        ln.strip() for ln in first if ln.strip() and not ln.strip().startswith("#")
    )
    assert "exit_code=$?" in first_stmt, (
        f"cleanup()'s first statement must capture exit_code=$? — got "
        f"{first_stmt!r}. Any earlier command overwrites $? and the relaunch "
        "decision + final exit then run against the wrong status."
    )
    assert 'exit "$exit_code"' in body, (
        "cleanup() must end on `exit \"$exit_code\"` so the captured status is "
        "re-raised — otherwise a successful teardown command leaves the "
        "script exiting 0 and a failed run reads as success."
    )


def test_describe_instances_diagnostics_retained():
    """The pre-existing describe-instances diagnostic (state / reason_code /
    state_reason logged to stdout + fanned out via krepis.alerts) must
    survive the migration — it is orthogonal to the relaunch DECISION (now
    the lib's job) and remains the primary human-facing failure surface."""
    body = _cleanup_body()
    assert "StateReason.Code" in body and "StateTransitionReason" in body, (
        "cleanup() must still query StateReason.Code + StateTransitionReason "
        "for the stdout diagnostic — #883 only lifts the relaunch DECISION, "
        "not the human-facing failure diagnostics."
    )
    assert "krepis.alerts publish" in body, (
        "cleanup() must still fan out the failure diagnostic via "
        "`python -m krepis.alerts publish`."
    )


def test_relaunch_execs_fresh_spot_with_orig_args():
    """On a classified reclaim, cleanup() must `exec bash "$0"` with the
    captured _ORIG_ARGS and an incremented SPOT_ATTEMPT, after `trap - EXIT`.
    The exec re-runs the launcher in place (bounded by MAX_SPOT_ATTEMPTS)."""
    body = _cleanup_body()
    assert "trap - EXIT" in body, (
        "cleanup() must `trap - EXIT` before the relaunch exec so the "
        "exec'd process installs its own EXIT trap cleanly."
    )
    m = re.search(r'SPOT_ATTEMPT=\$\(\(SPOT_ATTEMPT \+ 1\)\) exec bash "\$0"', body)
    assert m, (
        "cleanup() must relaunch via "
        '`SPOT_ATTEMPT=$((SPOT_ATTEMPT + 1)) exec bash "$0" ...` threading the '
        "incremented attempt across the re-exec."
    )
    assert "_ORIG_ARGS" in body, (
        "the relaunch exec must forward _ORIG_ARGS so the fresh attempt "
        "re-runs under the same mode (--mode / --instance-type / etc.)."
    )


def test_decision_happens_before_terminate_relaunch_after():
    """The lib relaunch-decision call (which describe-instances the spot
    internally) must run while the instance still exists — BEFORE
    terminate-instances; the relaunch exec must run AFTER teardown so the
    dead worker + its S3 staging are already cleaned when the fresh attempt
    starts."""
    body = _cleanup_body()
    decide_idx = body.find("ec2_spot relaunch-decision")
    term_idx = body.find("aws ec2 terminate-instances")
    exec_idx = body.find("exec bash")
    assert decide_idx != -1 and term_idx != -1 and exec_idx != -1, (
        "cleanup() must contain the relaunch-decision call, the "
        "terminate-instances call, and the relaunch exec."
    )
    assert decide_idx < term_idx, (
        "the lib relaunch-decision must run BEFORE terminate-instances — "
        "after teardown the instance is gone and the reclaim is "
        "unclassifiable."
    )
    assert term_idx < exec_idx, (
        "the relaunch exec must run AFTER terminate-instances + staging "
        "cleanup so the fresh attempt does not race the dead worker's S3 "
        "staging."
    )


def test_relaunch_bounded_by_max_spot_attempts():
    """The relaunch must be gated on SPOT_ATTEMPT < MAX_SPOT_ATTEMPTS so a
    persistent-reclaim loop can't relaunch unboundedly. Both the env
    defaults and the guard must be present."""
    text = _SCRIPT.read_text()
    assert re.search(r'MAX_SPOT_ATTEMPTS="\$\{MAX_SPOT_ATTEMPTS:-\d+\}"', text), (
        "MAX_SPOT_ATTEMPTS default not declared."
    )
    assert re.search(r'SPOT_ATTEMPT="\$\{SPOT_ATTEMPT:-1\}"', text), (
        "SPOT_ATTEMPT must default to 1 (first run)."
    )
    body = _cleanup_body()
    assert "SPOT_ATTEMPT" in body and "MAX_SPOT_ATTEMPTS" in body, (
        "cleanup() must gate the relaunch on SPOT_ATTEMPT < MAX_SPOT_ATTEMPTS."
    )


def test_exit_trap_installed():
    """The `trap cleanup EXIT` installation must remain intact."""
    text = _SCRIPT.read_text()
    assert "trap cleanup EXIT" in text
