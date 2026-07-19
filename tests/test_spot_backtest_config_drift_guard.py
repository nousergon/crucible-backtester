"""Pins the config#2871 pre-launch drift guard in spot_backtest.sh:
config.yaml is commonly a symlink into the alpha-engine-config repo's
tracked backtester/config.yaml (operator flags like pit_parity_sweep live
there). If the symlink target is untracked, or has uncommitted drift
relative to git, the guard must WARN before launch — a hand-edited
operator flag would otherwise vanish silently on the next symlink/box
rebuild with no diff and no audit trail.

Static-analysis test (mirrors test_spot_backtest_preflight_only.py) — the
spot_backtest.sh SSM/EC2 path cannot be exercised in CI.
"""

from __future__ import annotations

from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "infrastructure" / "spot_backtest.sh"
)


def _text() -> str:
    return _SCRIPT.read_text()


def test_config_drift_guard_present_in_pre_launch_preflight():
    text = _text()
    start = text.index("pre_launch_preflight() {")
    end = text.index("\n}", start)
    body = text[start:end]
    assert "config#2871" in body, (
        "pre_launch_preflight must carry the config#2871 drift-guard block"
    )
    assert 'if [ -L "$REPO_ROOT/config.yaml" ]; then' in body, (
        "drift guard must detect whether config.yaml is a symlink"
    )


def test_config_drift_guard_checks_git_tracked_and_dirty():
    text = _text()
    start = text.index("config#2871")
    end = text.index("echo \"  pre-launch preflight OK.\"", start)
    block = text[start:end]
    assert "ls-files --error-unmatch" in block, (
        "guard must check the symlink target is git-tracked, not just present"
    )
    assert "status --porcelain -- \"$cfg_rel\"" in block, (
        "guard must check the symlink target for uncommitted (dirty) changes"
    )
    assert "will vanish on rebuild" in block or "will NOT survive a rebuild" in block, (
        "guard must WARN about the rebuild-silently-drops-flags risk, not just detect it silently"
    )


def test_config_drift_guard_is_soft_never_blocks_launch():
    """Matches the existing (3) dirty-.py/.sh-file check: this is a WARNING,
    not a launch blocker — config.yaml is often intentionally a plain file
    (not a symlink) in non-production/local runs."""
    text = _text()
    start = text.index("config#2871")
    end = text.index("echo \"  pre-launch preflight OK.\"", start)
    block = text[start:end]
    assert "exit 1" not in block, (
        "the config drift guard must be soft (WARNING only) — it must not "
        "call exit 1 and block a launch, since config.yaml is not always a "
        "symlink (e.g. local/non-EC2 runs)"
    )
