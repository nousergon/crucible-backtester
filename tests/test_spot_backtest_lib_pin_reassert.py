"""Guards the alpha-engine-lib pin re-assertion in spot_backtest.sh deps.

2026-06-06: the Evaluator stage failed every run with
``ModuleNotFoundError: No module named 'alpha_engine_lib.quant.stats'``.
Root cause: the deps step installs the backtester's requirements.txt
(alpha-engine-lib pinned with quant.stats) and THEN installs the
predictor's requirements.txt, whose OWN (older) alpha-engine-lib pin
(v0.47.0, predating quant.stats @v0.49.0) installed second and silently
DOWNGRADED the lib. evaluate.py -> analysis.stats_utils ->
quant.stats.multiple_testing then raised at import.

Fix: after the predictor install, re-install the backtester's
requirements.txt as the final word so a sibling's older pin can never
leave the env on a quant.stats-less lib.

This test pins that ordering invariant (regex-over-script — the spot
deps step cannot run locally).
"""

from __future__ import annotations

from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "infrastructure" / "spot_backtest.sh"
)


def _read() -> str:
    return _SCRIPT.read_text()


def test_predictor_reqs_installed_in_deps():
    s = _read()
    assert "cd /home/ec2-user/alpha-engine-predictor" in s, (
        "predictor deps install block not found — script structure changed"
    )


def test_backtester_lib_pin_reasserted_after_predictor_install():
    """The backtester's requirements.txt must be (re)installed AFTER the
    predictor's, so the predictor's older alpha-engine-lib pin can't win."""
    s = _read()
    pred_idx = s.index("cd /home/ec2-user/alpha-engine-predictor")
    # the re-assert: cd back to the backtester + reinstall its requirements
    reassert = s.index("cd /home/ec2-user/alpha-engine-backtester", pred_idx)
    assert reassert > pred_idx, (
        "backtester requirements.txt must be re-installed AFTER the predictor "
        "deps install to re-assert the lib pin (else the predictor's older pin "
        "downgrades alpha-engine-lib below quant.stats — the 2026-06-06 "
        "Evaluator failure)."
    )
    # and the reinstall line must follow that cd
    tail = s[reassert:]
    assert "install -q -r requirements.txt" in tail, (
        "no requirements.txt reinstall after cd back to the backtester"
    )


def test_reassert_install_is_loud_not_swallowed():
    """The re-assert install must NOT swallow stderr (no 2>/dev/null) — a
    genuine resolution failure should surface, per fail-loud."""
    s = _read()
    pred_idx = s.index("cd /home/ec2-user/alpha-engine-predictor")
    reassert = s.index("cd /home/ec2-user/alpha-engine-backtester", pred_idx)
    # the reinstall immediately after the re-assert cd should not pipe to /dev/null
    seg = s[reassert: reassert + 300]
    line = next(
        (ln for ln in seg.splitlines() if "install -q -r requirements.txt" in ln),
        "",
    )
    assert line and "2>/dev/null" not in line, (
        "the lib re-assert install must be loud (no 2>/dev/null) so a real "
        "failure isn't silently shipped"
    )
