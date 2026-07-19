"""Pins the fail-loud numpy-2 consistency guard in spot_backtest.sh deps, and
guards against re-introducing the stale ``pip install 'numpy<2'`` downgrade.

2026-07-19 (first weekly SF spot run after the config#2815 numpy-2 migration):
the Backtester's ``runtime_smoke`` phase crashed at ``GBMScorer.load`` ->
``import scipy.sparse`` with ``AttributeError: module 'numpy' has no attribute
'long'``. Root cause: the deps step installs the backtester's + predictor's
requirements.txt (both pin numpy>=2 as of config#2815 / #536), which correctly
resolved numpy to 2.5.1 — then a stale ``$PIP install 'numpy<2'`` at the end of
the deps step (added 2026-03-24, commit 0534004, back when pyarrow wheels were
numpy-1 built) force-DOWNGRADED numpy to 1.26.4 AFTER the install. That left the
numpy-2-built scipy 1.18 / cvxpy 1.9.2 wheels referencing ``np.long`` (removed
in numpy 1.24-1.26) -> import crash, ~40 min into an otherwise-healthy weekly
run.

Root fix: complete the migration — REMOVE the caller-side downgrade so the env
stays on the numpy-2 stack requirements.txt already validates. CLASS fix (this
guard): after all installs, assert the exact import chain that broke (numpy>=2
+ scipy.sparse + lightgbm) so any future co-installed pin that downgrades numpy
breaks LOUD at deps time (seconds) instead of silently ~40 min into the run.

Regex-over-script — the spot deps step cannot run locally.
"""

from __future__ import annotations

from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "infrastructure" / "spot_backtest.sh"
)


def _read() -> str:
    return _SCRIPT.read_text()


def _executable_lines(s: str) -> str:
    """Executable (non-comment) lines only — comments may legitimately narrate
    the historical ``numpy<2`` downgrade; what must never recur is a live
    ``pip install 'numpy<2'`` invocation in the deps step."""
    return "\n".join(ln for ln in s.splitlines() if not ln.lstrip().startswith("#"))


def test_no_numpy_downgrade_in_deps_step():
    """The stale ``pip install 'numpy<2'`` downgrade must be gone. Re-adding it
    (or any ``numpy<2`` / ``numpy==1`` pin) downgrades numpy under the numpy-2
    scipy/cvxpy wheels requirements.txt installs -> the np.long crash."""
    code = _executable_lines(_read())
    assert "numpy<2" not in code, (
        "a `pip install 'numpy<2'` downgrade is back in the deps step; it "
        "force-downgrades numpy below the numpy-2 scipy/cvxpy build (config#2815) "
        "and reintroduces the `np.long` runtime_smoke crash"
    )


def test_numpy2_guard_present_and_after_predictor_install():
    """The numpy-2 guard must import numpy + scipy.sparse + lightgbm and assert
    numpy>=2, AFTER the predictor deps install (so it sees the final resolved
    env — the predictor install is what lands numpy>=2.5.1)."""
    s = _read()
    pred_idx = s.index("cd /home/ec2-user/alpha-engine-predictor")
    guard_idx = s.find("import numpy, scipy.sparse, lightgbm")
    assert guard_idx != -1, "no numpy-2 import guard found in the deps step"
    assert guard_idx > pred_idx, (
        "the numpy-2 guard must run AFTER the predictor deps install to catch a "
        "downgrade from the co-install"
    )
    seg = s[guard_idx: guard_idx + 400]
    assert ">= 2" in seg, (
        "the guard must assert numpy major version >= 2, not merely import it"
    )


def test_numpy2_guard_fails_loud():
    """The guard must exit non-zero on failure (fail-loud) — not swallow it."""
    s = _read()
    guard_idx = s.find("import numpy, scipy.sparse, lightgbm")
    seg = s[guard_idx: guard_idx + 600]
    assert "exit 1" in seg, (
        "the numpy-2 guard must `exit 1` on failure so a downgrade breaks the "
        "deps step loudly, not silently ~40 min into the run"
    )
    line = next((ln for ln in seg.splitlines() if "import numpy, scipy.sparse" in ln), "")
    assert "2>/dev/null" not in line, "the guard import must be loud (no 2>/dev/null)"


def test_requirements_still_pin_numpy2():
    """The guard is only correct while requirements.txt actually targets numpy 2.
    If a future edit re-caps numpy below 2, the guard (and this incident's fix)
    would wrongly fail every run — pin the intent so the two move in lockstep."""
    req = (Path(__file__).resolve().parent.parent / "requirements.txt").read_text()
    assert "numpy<3" in req.replace(" ", ""), (
        "backtester requirements.txt no longer targets numpy 2 (numpy<3); the "
        "numpy-2 deps guard assumes the config#2815 stack"
    )
