"""Pins the fail-loud numba/vectorbt import guard in spot_backtest.sh deps.

2026-07-18 weekend (config-I3279): the config#2815 numpy-2 migration (#536)
lifted the requirements ceilings and pip resolved numpy 2.5.1 — above numba
0.66's declared ``numpy<2.5`` ceiling. Nothing imported numba at deps time, so
the inconsistent env shipped silently and only surfaced ~18 hours into the
weekly run, when the simulate phase's ``import vectorbt`` (vectorbt → numba)
raised "Numba needs NumPy 2.4 or less. Got NumPy 2.5". backtest.py's phase
handler degraded ``portfolio_stats.json`` to a ``status:error`` stub and the
LOAD-BEARING ``predictor/optimizer_gate/`` artifact went unwritten — stale for
10+ days before the freshness monitor's SLA backstop paged.

Resolution-layer fix: the ``numpy>=2.0,<2.5`` cap (config#2975 / PR #541) plus
the pip-check metadata gate (config#2973 / PRs #546/#551), which catches THIS
instance because numba declares its ceiling. CLASS fix (this guard): import
numba + vectorbt at deps time alongside the existing numpy/scipy/lightgbm
chain, so an ABI-level numba/numpy break that ships with self-consistent
metadata — which pip check cannot see — also fails LOUD at deps time (seconds)
instead of mid-run. vectorbt backs vectorbt_bridge.py → portfolio_stats and
the optimizer-gate arc, so this chain is load-bearing for the Saturday SF's
promotion artifacts.

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


def _guard_invocation(s: str) -> str:
    """Return the single $PYBIN -c guard invocation importing the chains."""
    matches = re.findall(r'\$PYBIN -c "import numpy[^"]*"', s)
    assert matches, "no numpy import guard invocation found in deps step"
    assert len(matches) == 1, (
        "expected exactly one combined import-chain guard invocation; a split "
        "would let one chain regress silently while the other still passes"
    )
    return matches[0]


def test_guard_imports_numba_and_vectorbt():
    """The deps-time import guard must exercise the numba + vectorbt chain."""
    guard = _guard_invocation(_read())
    assert "numba" in guard, (
        "deps import guard no longer imports numba — an ABI-level numba/numpy "
        "mismatch would ship silently and crash the simulate phase mid-run "
        "(config-I3279)"
    )
    assert "vectorbt" in guard, (
        "deps import guard no longer imports vectorbt — the vectorbt_bridge "
        "portfolio_stats/optimizer_gate arc would be unguarded (config-I3279)"
    )


def test_guard_runs_after_requirements_install():
    """The guard must run AFTER the requirements install so it sees the final
    resolved environment, not a partially-installed one."""
    s = _read()
    install_idx = s.index("install -q -r requirements.txt")
    guard_idx = s.index(_guard_invocation(s))
    assert guard_idx > install_idx, (
        "the numba/vectorbt import guard must run after the requirements "
        "install to validate the final resolved env"
    )


def test_guard_fails_loud():
    """The guard must exit non-zero on failure — never swallow the import
    error (feedback_no_silent_fails)."""
    s = _read()
    guard_idx = s.index(_guard_invocation(s))
    seg = s[guard_idx: guard_idx + 1000]
    assert "exit 1" in seg, (
        "the import-chain guard must exit 1 on failure so a broken env fails "
        "the deps step instead of shipping to the run"
    )
