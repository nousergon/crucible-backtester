"""Pins the fail-loud nousergon-lib import guard in spot_backtest.sh deps.

2026-06-06 (L4513): the Evaluator failed every run with
``ModuleNotFoundError: No module named 'alpha_engine_lib.quant.stats'``.
Root cause: the deps step installs the backtester's requirements.txt then the
predictor's, and the predictor's older lib pin (v0.47.0, predating
quant.stats @v0.49.0) installed second and silently downgraded the lib. The
2>/dev/null on the predictor install hid pip's downgrade note, so the failure
only surfaced — silently — at evaluate.py's import, weeks later.

Root fix: fleet pin alignment (predictor + backtester both @v0.53.0). CLASS
fix (this guard): after all installs, assert the nousergon-lib modules the
Evaluator imports are actually present, and FAIL LOUD at deps time if a future
drift ever downgrades the lib below them — never silently again.

2026-06-26: the lib distribution/module was renamed alpha-engine-lib →
nousergon-lib (``alpha_engine_lib`` is now a deprecated import alias).
requirements.txt installs the ``nousergon-lib`` distribution, so the guard now
verifies via the real names — ``import nousergon_lib.quant.stats`` and
``pip show nousergon-lib``. The stale ``pip show alpha-engine-lib`` returned
nothing and exited 1 under ``set -eo pipefail``, failing the Saturday SF deps
step; these assertions pin the guard onto the non-deprecated names so it can
never silently regress onto the old distribution name again.

Regex-over-script — the spot deps step cannot run locally.
"""

from __future__ import annotations

from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "infrastructure" / "spot_backtest.sh"
)


def _read() -> str:
    return _SCRIPT.read_text()


def test_import_guard_present_and_after_predictor_install():
    """The guard must import alpha_engine_lib.quant.stats AFTER the predictor
    deps install (so it sees the final resolved env)."""
    s = _read()
    pred_idx = s.index("cd /home/ec2-user/alpha-engine-predictor")
    guard_idx = s.find("import nousergon_lib.quant.stats.multiple_testing")
    assert guard_idx != -1, "no quant.stats import guard found in deps step"
    assert guard_idx > pred_idx, (
        "the import guard must run AFTER the predictor deps install to catch a "
        "downgrade from the co-install"
    )


def test_import_guard_fails_loud():
    """The guard must exit non-zero on failure (fail-loud) — not swallow it."""
    s = _read()
    guard_idx = s.find("import nousergon_lib.quant.stats.multiple_testing")
    seg = s[guard_idx: guard_idx + 400]
    assert "exit 1" in seg, (
        "the import guard must `exit 1` on failure so a downgrade breaks the "
        "deps step loudly (per fail-loud), not silently at evaluate.py"
    )
    # the guard's own python invocation must NOT pipe stderr to /dev/null
    line = next((ln for ln in seg.splitlines() if "multiple_testing" in ln), "")
    assert "2>/dev/null" not in line, "the guard import must be loud (no 2>/dev/null)"


def test_no_silent_predictor_install_masks_the_guard():
    """config#2359: the predictor install used to be best-effort
    (``2>/dev/null || true``), which hid pip failures until the quant.stats
    guard tripped 60+ minutes later inside predictor_pipeline. The install
    itself must now fail loud (no ``2>/dev/null``, no ``|| true``) AND the
    downstream guard must still exist as a backstop."""
    s = _read()
    pred_install = s.index("cd /home/ec2-user/alpha-engine-predictor")
    guard_idx = s.index("import nousergon_lib.quant.stats.multiple_testing")
    pred_section = s[pred_install:guard_idx]
    assert "2>/dev/null || true" not in pred_section, (
        "predictor deps install must not swallow failures via 2>/dev/null || true"
    )
    assert "exit 1" in pred_section, (
        "no fail-loud backstop after the predictor install"
    )


def test_guard_verifies_the_renamed_nousergon_lib_distribution():
    """The lib was renamed alpha-engine-lib → nousergon-lib (2026-06-26).
    requirements.txt installs the ``nousergon-lib`` distribution, so the guard's
    ``pip show`` MUST query ``nousergon-lib`` — querying the old name returns
    'Package(s) not found' and exits 1 under ``set -eo pipefail``, which is the
    exact Saturday-SF deps-step failure this pins against. The distribution the
    guard verifies must match what requirements.txt actually installs."""
    s = _read()
    req = (
        Path(__file__).resolve().parent.parent / "requirements.txt"
    ).read_text()
    assert "nousergon-lib" in req, "requirements.txt no longer installs nousergon-lib"
    # Inspect EXECUTABLE lines only — comments may legitimately reference the
    # historical `alpha-engine-lib` name; what must never recur is a live
    # `pip show alpha-engine-lib` invocation in the deps step.
    code = "\n".join(
        ln for ln in s.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "show alpha-engine-lib" not in code, (
        "the deps guard still queries the stale `alpha-engine-lib` distribution "
        "name; `pip show alpha-engine-lib` exits 1 under pipefail and fails deps. "
        "Verify via the renamed `nousergon-lib` distribution instead."
    )
    assert "show nousergon-lib" in code, (
        "the deps guard must verify the resolved lib via `pip show nousergon-lib` "
        "(the distribution requirements.txt installs)"
    )
    # The guard import must also land on the real module, not the deprecated alias.
    assert "import nousergon_lib.quant.stats.multiple_testing" in s, (
        "the guard import must use the real `nousergon_lib` module, not the "
        "deprecated `alpha_engine_lib` alias"
    )


def test_guard_passes_on_successful_import():
    """§119 rule 1 success-path: the guard must use the ``|| { ... exit 1; }``
    pattern so a successful import passes silently (no exit), and the post-guard
    ``$PIP show nousergon-lib`` version line follows to confirm the resolved
    distribution on the success path."""
    s = _read()
    guard_idx = s.find("import nousergon_lib.quant.stats.multiple_testing")
    assert guard_idx != -1, "no quant.stats import guard found"
    seg = s[guard_idx: guard_idx + 800]
    # The guard must be `$PYBIN -c "import ..." || {` — the || ensures a
    # successful import continues past the guard without triggering exit 1.
    lines = seg.splitlines()
    guard_line = next((ln for ln in lines if "multiple_testing" in ln), "")
    assert "||" in guard_line, (
        "the import guard must use `|| {` so a successful import does NOT "
        "trigger exit 1 — a standalone import without the failover would be "
        "silent on failure but also provides no structural success-path signal"
    )
    assert "|| {" in guard_line or "||" + guard_line.split("||")[1].lstrip().startswith("{"), (
        "the import guard's `||` must be followed by a block containing "
        "exit 1 so the guard only fires on actual import failure"
    )
    # The guard line itself must not redirect stderr (would silence the error).
    assert "2>/dev/null" not in guard_line, (
        "the guard import must be loud (no 2>/dev/null) so a failure message "
        "reaches stderr before the exit"
    )
    # A $PIP show nousergon-lib version line should follow the guard on the
    # success path so the resolved version is visible in logs.
    post_guard = s[guard_idx + 300: guard_idx + 600]
    assert "show nousergon-lib" in post_guard, (
        "a `$PIP show nousergon-lib` version line should follow the import "
        "guard on the success path, confirming the resolved distribution"
    )
