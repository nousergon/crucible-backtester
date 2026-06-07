"""Cross-stage artifact contract: the SF Backtester mode MUST produce every
artifact the Evaluator treats as critical.

This is the preflight that would have caught L4513 at PR time. The 2026-05-16
SF backtester split (#249/#250) set the Backtester state to --mode=param-sweep,
but the `simulate` phase that produces portfolio_stats was gated to
("simulate","all") — so param-sweep silently stopped producing a CRITICAL
Evaluator input, and the Evaluator hard-failed for ~3 weeks with no signal.

The contract:
  for the mode the SF Backtester state runs (BACKTESTER_SF_MODE),
  every artifact in evaluate.py's `critical` set must have a producer phase in
  backtest.py whose mode-gate includes that mode.

A static source-level check (the full pipeline can't run in CI without S3) —
it locks the producer↔consumer artifact contract so a future mode change /
phase-gate edit / topology split can't silently orphan a critical artifact.
Runtime production of the declared artifacts is enforced separately by the
fail-loud export guard (L4518) in backtest.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_BACKTEST = _ROOT / "backtest.py"
_EVALUATE = _ROOT / "evaluate.py"

# The mode the Saturday SF "Backtester" state invokes (spot_backtest.sh
# --mode=param-sweep). If the SF topology changes this, update here AND ensure
# every critical artifact's producer gate below includes the new mode.
BACKTESTER_SF_MODE = "param-sweep"

# Which backtest.py phase produces each Evaluator-critical artifact, identified
# by the `if args.mode in (...)` gate that immediately precedes/guards it.
# artifact -> a regex matching that producer's mode-gate line.
_PRODUCER_GATE = {
    "portfolio_stats.json": re.compile(
        r'if args\.mode in \(([^)]*)\):\s*\n\s*from phase_artifacts import save_json, load_json'
    ),
    "sweep_df.parquet": re.compile(
        r'# ── Param sweep[^\n]*\n\s*if args\.mode in \(([^)]*)\):'
    ),
}


def _evaluator_critical() -> set[str]:
    txt = _EVALUATE.read_text()
    m = re.search(r'critical = \{([^}]*)\}', txt)
    assert m, "could not find evaluate.py `critical = {...}` set"
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def test_critical_set_is_what_the_contract_covers():
    """If evaluate.py adds a new critical artifact, this test must be extended
    with its producer gate — otherwise the contract silently misses it."""
    critical = _evaluator_critical()
    covered = set(_PRODUCER_GATE)
    assert critical == covered, (
        f"Evaluator critical set {critical} != contract-covered {covered}. "
        f"Add the new artifact + its backtest.py producer gate to _PRODUCER_GATE."
    )


@pytest.mark.parametrize("artifact", list(_PRODUCER_GATE))
def test_sf_mode_produces_each_critical_artifact(artifact):
    """The SF Backtester mode must be in each critical artifact's producer gate."""
    src = _BACKTEST.read_text()
    m = _PRODUCER_GATE[artifact].search(src)
    assert m, (
        f"could not locate the producer gate for {artifact} in backtest.py — "
        f"the producer phase/structure changed; update _PRODUCER_GATE."
    )
    modes = set(re.findall(r'"([^"]+)"', m.group(1)))
    assert BACKTESTER_SF_MODE in modes, (
        f"backtest.py does not produce {artifact} in --mode={BACKTESTER_SF_MODE} "
        f"(producer gate modes = {sorted(modes)}). The SF Backtester state runs "
        f"that mode and the Evaluator hard-requires {artifact} → it would starve "
        f"(this is the L4513 failure class). Add '{BACKTESTER_SF_MODE}' to the "
        f"producer gate, or change the SF mode."
    )
