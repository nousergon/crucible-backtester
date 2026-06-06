"""Pins the fail-loud alpha-engine-lib import guard in spot_backtest.sh deps.

2026-06-06 (L4513): the Evaluator failed every run with
``ModuleNotFoundError: No module named 'alpha_engine_lib.quant.stats'``.
Root cause: the deps step installs the backtester's requirements.txt then the
predictor's, and the predictor's older alpha-engine-lib pin (v0.47.0, predating
quant.stats @v0.49.0) installed second and silently downgraded the lib. The
2>/dev/null on the predictor install hid pip's downgrade note, so the failure
only surfaced — silently — at evaluate.py's import, weeks later.

Root fix: fleet pin alignment (predictor + backtester both @v0.53.0). CLASS
fix (this guard): after all installs, assert the alpha-engine-lib modules the
Evaluator imports are actually present, and FAIL LOUD at deps time if a future
drift ever downgrades the lib below them — never silently again.

Regex-over-script — the spot deps step cannot run locally.
"""

from __future__ import annotations

import re
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
    guard_idx = s.find("import alpha_engine_lib.quant.stats.multiple_testing")
    assert guard_idx != -1, "no quant.stats import guard found in deps step"
    assert guard_idx > pred_idx, (
        "the import guard must run AFTER the predictor deps install to catch a "
        "downgrade from the co-install"
    )


def test_import_guard_fails_loud():
    """The guard must exit non-zero on failure (fail-loud) — not swallow it."""
    s = _read()
    guard_idx = s.find("import alpha_engine_lib.quant.stats.multiple_testing")
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
