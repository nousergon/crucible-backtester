"""Cross-stage artifact contract: the SF Backtester mode MUST produce every
artifact the Evaluator treats as critical — now driven by the declarative
``pipeline_manifest`` (L4526, plan §6 Phase 4 / §2 P2+P7).

This is the preflight that would have caught L4513 at PR time. The 2026-05-16
SF backtester split (#249/#250) set the Backtester state to --mode=param-sweep,
but the `simulate` phase that produces portfolio_stats was gated to
("simulate","all") — so param-sweep silently stopped producing a CRITICAL
Evaluator input and the Evaluator hard-failed for ~3 weeks with no signal.

The contract (L4520, first slice #293) used to hardcode the SF mode + an inline
regex map of producers. L4526 lifts that topology into ``pipeline_manifest`` as
the single source of truth; this test now:
  1. binds the manifest's Evaluator-critical set to evaluate.py's live
     ``critical`` set (the consumer can't drift from the manifest);
  2. binds each manifest producer stage's declared ``modes`` to its live
     ``if args.mode in (...)`` gate in backtest.py (the producer can't drift);
  3. asserts the manifest contract holds for the SF Backtester mode.

A static source-level check (the full pipeline can't run in CI without S3) — it
locks the producer↔consumer artifact contract so a future mode change /
phase-gate edit / topology split can't silently orphan a critical artifact.
Runtime production of the declared artifacts is enforced separately by the
fail-loud export guard (L4518) in backtest.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import pipeline_manifest as manifest

_ROOT = Path(__file__).resolve().parent.parent
_BACKTEST = _ROOT / "backtest.py"
_EVALUATE = _ROOT / "evaluate.py"


def _evaluator_critical_in_evaluate_py() -> set[str]:
    txt = _EVALUATE.read_text()
    m = re.search(r"critical = \{([^}]*)\}", txt)
    assert m, "could not find evaluate.py `critical = {...}` set"
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def test_manifest_critical_matches_evaluate_py():
    """The manifest's Evaluator-critical set (the `evaluator` stage's requires)
    must equal evaluate.py's live `critical` set — the single binding that keeps
    the manifest honest about what the consumer actually hard-requires."""
    assert manifest.evaluator_critical() == _evaluator_critical_in_evaluate_py(), (
        f"manifest evaluator_critical {set(manifest.evaluator_critical())} != "
        f"evaluate.py critical {_evaluator_critical_in_evaluate_py()}. Update the "
        f"`evaluator` stage's requires in pipeline_manifest.py (or evaluate.py)."
    )


@pytest.mark.parametrize(
    "stage",
    [s for s in manifest.STAGES if s.gate_pattern],
    ids=lambda s: s.name,
)
def test_producer_gate_matches_backtest_py(stage):
    """Each manifest producer's declared `modes` must equal the live
    `if args.mode in (...)` gate it points at in backtest.py."""
    src = _BACKTEST.read_text()
    m = re.search(stage.gate_pattern, src)
    assert m, (
        f"could not locate the producer gate for stage {stage.name!r} in "
        f"backtest.py — the producer phase/structure changed; update the "
        f"stage's gate_pattern in pipeline_manifest.py."
    )
    live_modes = set(re.findall(r'"([^"]+)"', m.group(1)))
    assert set(stage.modes) == live_modes, (
        f"stage {stage.name!r} manifest modes {set(stage.modes)} != live "
        f"backtest.py gate modes {live_modes}. The manifest drifted from the "
        f"code — reconcile pipeline_manifest.py with the actual mode-gate."
    )


def test_sf_backtester_mode_satisfies_contract():
    """For the mode the SF Backtester state runs, every Evaluator-critical
    artifact must be produced by a stage that runs in that mode (L4513 guard)."""
    violations = manifest.contract_violations(manifest.SF_BACKTESTER_MODE)
    assert violations == [], "\n".join(violations)


def test_sf_backtester_mode_is_known():
    assert manifest.SF_BACKTESTER_MODE in manifest.ALL_MODES
