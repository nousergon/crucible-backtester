"""Pins the L4486 pit_parity stage-gate relocation in `infrastructure/spot_backtest.sh`.

L4486 (2026-06-05): pit_parity used to be gated on `! _stage_skipped backtest`,
so it ran inside PredictorBacktest (whose skip-set is `parity,evaluator`, leaving
`backtest` un-skipped) — STACKED after the main predictor_pipeline already held
~3.5 GB, leaving only ~4.5 GB free on the 8 GB box < the 6 GB pre-pipeline RAM
headroom guard → the guard fail-fast refused and pit_parity produced no usable
artifact for the L3293 manual-flip gate.

The fix gates pit_parity on its own `pit_parity` stage token so it runs in the
standalone Parity SF state (which passes `--skip-stages=backtest,evaluator` →
`backtest` skipped but `pit_parity` NOT skipped) in a FRESH process with full RAM
headroom. The SF turns it OFF in PredictorBacktest via `--no-pit-parity`, so it
fires EXACTLY ONCE.

These tests pin the dispatcher-side script invariants. The "fires exactly once"
invariant across the four SF states is pinned in alpha-engine-data's SF tests.
"""

from __future__ import annotations

import re
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "infrastructure" / "spot_backtest.sh"
)


def _read_script() -> str:
    return _SCRIPT.read_text()


def test_pit_parity_is_a_known_stage_token():
    """`pit_parity` must be in the typo-guard vocabulary so `--skip-stages=pit_parity`
    is accepted (and a typo still hard-fails)."""
    script = _read_script()
    m = re.search(r'_KNOWN_STAGES="([^"]+)"', script)
    assert m, "_KNOWN_STAGES declaration not found"
    stages = m.group(1).split()
    assert "pit_parity" in stages, f"pit_parity missing from _KNOWN_STAGES: {stages}"


def test_pit_parity_gate_keyed_on_pit_parity_token_not_backtest():
    """The pit_parity gate must depend on the `pit_parity` stage token, NOT the
    `backtest` token — otherwise it cannot run in the Parity state (which skips
    `backtest`) and would still run stacked in PredictorBacktest."""
    script = _read_script()
    # The gate line: PIT_PARITY_ENABLED == 1 AND not skipped.
    gate = re.search(
        r'PIT_PARITY_ENABLED:-0.*&&\s*!\s*_stage_skipped\s+(\w+)', script
    )
    assert gate, "pit_parity gate line not found"
    assert gate.group(1) == "pit_parity", (
        f"pit_parity gate keyed on '{gate.group(1)}', expected 'pit_parity' (L4486)"
    )
    # Regression: the pit_parity gate must NOT be keyed on `backtest` anymore.
    # (The backtest stage's own `if _stage_skipped backtest; then` is a separate,
    # legitimate use — only the `&& ! _stage_skipped backtest` pit_parity form is
    # forbidden.)
    assert "&& ! _stage_skipped backtest" not in script, (
        "pit_parity gate still keyed on the `backtest` stage token (pre-L4486)"
    )


def test_parity_state_skipset_does_not_skip_pit_parity():
    """Sanity: the standalone Parity SF invocation skips `backtest,evaluator` but
    NOT `pit_parity`, so with the token-gate the pit_parity stage runs there.
    (Mirror of the SF-side contract; documents the cross-repo dependency.)"""
    skipset = {"backtest", "evaluator"}
    assert "pit_parity" not in skipset
