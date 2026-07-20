"""Pins the fail-loud ``pip check`` dependency-consistency gate in
``spot_backtest.sh`` deps (config#2973).

The numpy-2 guard (``test_spot_backtest_numpy2_guard.py``) only asserts ONE
specific import chain (numpy + scipy.sparse + lightgbm) stays consistent —
the exact chain that broke the 2026-07-19 weekly run. But `pip install`
reports ANY co-install/transitive-dependency conflict as a post-hoc "does not
take into account all installed packages" warning and still exits 0 — the
same silent-inconsistency CLASS, not unique to numpy. This guard runs
``pip check`` after all installs (backtester + predictor) and fails loud on
any conflict not explicitly allowlisted, turning the whole class into a
deps-time (seconds) failure instead of a per-import-chain guard that must be
hand-extended for every new breakage.

Regex-over-script — the spot deps step cannot run locally.
"""

from __future__ import annotations

from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "infrastructure" / "spot_backtest.sh"
)


def _read() -> str:
    return _SCRIPT.read_text()


def test_pip_check_gate_present_after_numpy2_guard():
    """The pip check gate must run AFTER the numpy-2 guard (so it sees the
    final resolved env, same rationale as the numpy-2 guard itself running
    after the predictor install)."""
    s = _read()
    numpy2_idx = s.find("import numpy, scipy.sparse, lightgbm")
    pip_check_idx = s.find("PIP_CHECK_OUT=")
    assert numpy2_idx != -1, "numpy-2 guard not found — test fixture assumption broken"
    assert pip_check_idx != -1, "no pip-check gate found in the deps step"
    assert pip_check_idx > numpy2_idx, (
        "the pip-check gate must run after the numpy-2 guard, at the end of the "
        "deps step, so it sees the fully resolved environment"
    )


def test_pip_check_gate_fails_loud_on_unallowlisted_conflict():
    """The gate must exit non-zero when pip check reports a conflict that
    survives the allowlist filter — not swallow it with `|| true` alone."""
    s = _read()
    gate_idx = s.find("PIP_CHECK_OUT=")
    seg = s[gate_idx: gate_idx + 900]
    assert "exit 1" in seg, "the pip-check gate must `exit 1` on an unallowlisted conflict"
    assert "PIP_CHECK_REMAIN" in seg, "the gate must filter pip check output through an allowlist"


def test_pip_check_itself_never_unconditionally_suppressed():
    """Binding constraint (config#2973): the raw `pip check` invocation must
    never be `|| true`'d directly — only the derived PIP_CHECK_OUT capture may
    tolerate its non-zero exit (pip check exits 1 merely to report conflicts
    that then go through the allowlist filter, not to signal an unrecoverable
    error)."""
    s = _read()
    idx = s.find("pip check")
    line = next(
        (ln for ln in s.splitlines() if "-m pip check" in ln),
        "",
    )
    assert line, "no `pip check` invocation found"
    assert "PIP_CHECK_OUT=" in line, (
        "the pip check invocation must be captured into PIP_CHECK_OUT for allowlist "
        "filtering, not run bare"
    )


def test_allowlist_is_explicit_not_broad_suppression():
    """The allowlist must be a data value the gate filters through, never a
    blanket `2>/dev/null` or `|| true` on the whole gate that would silence
    every future conflict, not just a curated set."""
    s = _read()
    gate_idx = s.find("PIP_CHECK_ALLOWLIST=")
    assert gate_idx != -1, "no PIP_CHECK_ALLOWLIST allowlist variable found"
    seg = s[gate_idx: gate_idx + 700]
    assert "grep -vFf" in seg or "grep -vF" in seg, (
        "the allowlist must be applied via a filter (grep -v) against the actual "
        "pip check output, not a bypass"
    )
