"""Pins the alpha-engine-config#3018 IAM-as-code-surface-4 provenance doc
block in spot_backtest.sh (fleet audit config#2340, dim-5/backtester).

`spot_backtest.sh` launches its spot box under `alpha-engine-executor-profile`,
whose backing role (`alpha-engine-executor-role`) is codified, applied, and
drift-checked in `crucible-executor/infrastructure/iam/` — NOT in this repo.
Backtester carries no IAM code of its own; this test only pins the doc
comments that make that cross-repo ownership legible from the consumer side,
plus the explicit accepted-by-design rationale for why SECURITY_GROUP/SUBNETS
are plain launch config rather than a tracked IAM/SG-as-code surface. Static-
analysis only (mirrors test_spot_backtest_config_drift_guard.py) — the actual
drift-check runs in crucible-executor's CI against live AWS, not here.
"""

from __future__ import annotations

from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "infrastructure" / "spot_backtest.sh"
)


def _text() -> str:
    return _SCRIPT.read_text()


def test_iam_profile_cites_tracked_source_of_truth():
    text = _text()
    assert "alpha-engine-config#3018" in text, (
        "IAM_PROFILE block must cite the issue that documented its "
        "cross-repo ownership"
    )
    idx = text.index('IAM_PROFILE="alpha-engine-executor-profile"')
    preceding = text[max(0, idx - 1500):idx]
    assert "crucible-executor/infrastructure/iam/" in preceding, (
        "must point at crucible-executor as the source of truth for the "
        "alpha-engine-executor-role policy backing this profile"
    )
    assert "check-drift.py" in preceding and "apply.sh" in preceding, (
        "must name the actual drift-check/apply mechanism rather than "
        "asserting governance without pointing at it"
    )


def test_subnets_and_sg_have_explicit_accepted_by_design_note():
    text = _text()
    idx = text.index('SECURITY_GROUP="sg-03cd3c4bd91e610b0"')
    preceding = text[max(0, idx - 1200):idx]
    assert "not an IAM policy" in preceding or "not scoped by subnet" in preceding, (
        "SECURITY_GROUP/SUBNETS must carry an explicit accepted-by-design "
        "rationale (per config#3018's escape hatch), not a silent literal"
    )
    assert 'Resource:"*"' in preceding or "Resource: \"*\"" in preceding, (
        "rationale must point at the actual IAM statement (RunInstances "
        "Resource:*) that makes subnet/SG values non-drift-relevant"
    )
