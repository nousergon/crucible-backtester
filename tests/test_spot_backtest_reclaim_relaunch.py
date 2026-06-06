"""Pins the L4485-b bounded spot-reclaim relaunch in `infrastructure/spot_backtest.sh`.

L4485-b (2026-06-05): a worker spot reclaimed mid-run by AWS
(Server.SpotInstanceTermination / instance-terminated-no-capacity) surfaced in
the validate-l4472 run as a generic SF poll-loop failure. The Saturday SF's
per-state Retry is on the `ssm:sendCommand` Task (which only SENDS and returns),
so it never fires on a mid-run reclaim. The dispatcher (spot_backtest.sh) is the
only layer that can see the reclaim reason — it already classifies it in
cleanup() — so it now re-execs itself on a fresh spot, bounded, gated STRICTLY
on the reclaim reason (no blind retry that could mask a real bug).

These tests pin the structure (regex-over-script — the spot lifecycle cannot run
locally). The happy path must remain byte-identical: the relaunch logic lives
ONLY inside the already-reclaim-aware cleanup() EXIT trap.
"""

from __future__ import annotations

import re
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "infrastructure" / "spot_backtest.sh"
)


def _read() -> str:
    return _SCRIPT.read_text()


def test_relaunch_budget_declared_bounded():
    """RECLAIM_RELAUNCH_MAX must be declared with a bounded default so the
    relaunch can never loop unboundedly."""
    s = _read()
    assert re.search(r'RECLAIM_RELAUNCH_MAX="?\$\{RECLAIM_RELAUNCH_MAX:-\d+\}', s), (
        "RECLAIM_RELAUNCH_MAX not declared with a bounded :-N default"
    )


def test_orig_args_captured_pre_parse():
    """The original argv must be captured BEFORE the arg-parse `while` loop
    consumes it, so the relaunch re-execs with identical flags."""
    s = _read()
    assert "_ORIG_ARGS=(\"$@\")" in s, "_ORIG_ARGS not captured for relaunch exec"
    assert s.index('_ORIG_ARGS=("$@")') < s.index("while [[ $# -gt 0 ]]; do"), (
        "_ORIG_ARGS must be captured before the parse loop shifts $@"
    )


def test_describe_captures_statereason_code():
    """The reclaim classifier's input bug (fixed 2026-06-06): the authoritative
    spot-reclaim signal is StateReason.Code == Server.SpotInstanceTermination.
    cleanup() previously queried StateTransitionReason ALONE (which only shows
    'Service initiated (<ts>)') and classified against Server.SpotInstanceTermination
    — a field/value mismatch that could never match, so two real reclaims on
    2026-06-06 hard-failed. The describe-instances query MUST capture
    StateReason.Code."""
    s = _read()
    assert "StateReason.Code" in s, (
        "cleanup() must query StateReason.Code — the authoritative reclaim signal "
        "(without it the classifier can never match a real reclaim)"
    )


def test_relaunch_gated_strictly_on_reclaim_reason():
    """The relaunch must be gated on a CLASSIFIED reclaim only — a generic
    failure must NOT trigger a relaunch (no blind retry). The reclaim is
    classified via _is_reclaim, set by the authoritative StateReason.Code case."""
    s = _read()
    # _is_reclaim is set inside the Server.SpotInstanceTermination (reason_code) case
    m = re.search(
        r'\*Server\.SpotInstanceTermination\*\)\s*_is_reclaim=1',
        s,
    )
    assert m, "_is_reclaim not set inside the Server.SpotInstanceTermination case"
    # _will_relaunch is gated on BOTH _is_reclaim AND a positive budget
    assert re.search(
        r'\[ "\$_is_reclaim" = "1" \] && \[ "\$\{RECLAIM_RELAUNCH_MAX:-0\}" -gt 0 \]',
        s,
    ), "relaunch must be gated on _is_reclaim AND a positive budget"


def test_service_initiated_shutting_down_classified_as_reclaim():
    """Belt-and-suspenders: a worker already shutting-down/terminated with
    StateTransitionReason 'Service initiated' (AWS reclaimed it out from under a
    still-failing dispatcher) must also classify as a reclaim — this is the exact
    2026-06-06 signature that the old classifier missed."""
    s = _read()
    assert re.search(
        r'shutting-down:\*Service\\? initiated\* \| terminated:\*Service\\? initiated\*\)\s*_is_reclaim=1',
        s,
    ), "the shutting-down/terminated + 'Service initiated' reclaim path is missing"


def test_relaunch_exec_decrements_budget():
    """The exec must decrement RECLAIM_RELAUNCH_MAX so the relaunch is bounded."""
    s = _read()
    assert re.search(
        r'exec env RECLAIM_RELAUNCH_MAX="\$\(\(RECLAIM_RELAUNCH_MAX - 1\)\)" bash "\$0" "\$\{_ORIG_ARGS\[@\]\}"',
        s,
    ), "bounded relaunch exec (decrementing budget, re-passing argv) not found"


def test_relaunch_happens_after_terminate_and_only_when_flagged():
    """The relaunch exec must be guarded by _will_relaunch and occur AFTER the
    dead worker is terminated + its S3 staging cleaned (so the fresh attempt is
    clean)."""
    s = _read()
    term = s.index("Instance terminated; S3 staging cleaned.")
    guard = s.index('if [ "$_will_relaunch" = "1" ]; then')
    exec_idx = s.index('exec env RECLAIM_RELAUNCH_MAX=')
    assert term < guard < exec_idx, (
        "relaunch must be guarded by _will_relaunch and run after terminate+clean"
    )


def test_exit_trap_and_status_preservation_intact():
    """The L4485 status-preserving EXIT trap must remain — the relaunch is
    additive, not a replacement (any non-reclaim failure still exits with its
    real code)."""
    s = _read()
    assert "trap cleanup EXIT" in s
    assert 'exit "$exit_code"' in s
    assert "terminate-instances" in s
