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
    """The guard exists precisely because the predictor install uses
    2>/dev/null || true (best-effort). Confirm the guard is the loud backstop —
    i.e. a fail-loud exit exists somewhere after that best-effort install."""
    s = _read()
    pred_install = s.index("install -q -r requirements.txt 2>/dev/null || true")
    assert "exit 1" in s[pred_install:], (
        "no fail-loud backstop after the best-effort predictor install"
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
