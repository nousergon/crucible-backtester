"""Pins that --mode=param-sweep produces the Evaluator's critical artifacts.

L4513 (2026-06-06): the Saturday SF's Backtester state runs --mode=param-sweep,
but the `simulate` phase that produces `portfolio_stats` was gated to
("simulate", "all") only — so param-sweep never produced portfolio_stats.json.
The Evaluator hard-requires portfolio_stats.json + sweep_df.parquet, so it
silently starved (config frozen 2026-05-20) until recovery chased it down.

Root timeline: the 2026-05-16 SF backtester split (#249/#250) moved the main
Backtester state to --mode=param-sweep + split predictor/portfolio-opt into
their own states, leaving NO state running simulate/all — orphaning the
simulate phase.

Fix: include "param-sweep" in the simulate-phase mode gate, and a mode-aware
FAIL-LOUD guard (L4518) that raises if a mode expected to produce the critical
artifacts didn't (instead of silently skipping the uploads).

Source-text invariants — the full sim pipeline can't run locally without S3.
"""

from __future__ import annotations

import re
from pathlib import Path

_BACKTEST = Path(__file__).resolve().parent.parent / "backtest.py"


def _src() -> str:
    return _BACKTEST.read_text()


def test_simulate_phase_gate_includes_param_sweep():
    """The simulate phase (producing portfolio_stats) must run in param-sweep
    mode — it's the mode the SF Backtester state runs."""
    s = _src()
    # the simulate-mode block guard
    assert 'if args.mode in ("simulate", "param-sweep", "all"):' in s, (
        "the simulate phase mode-gate must include 'param-sweep' — otherwise "
        "the SF Backtester state (--mode=param-sweep) never produces "
        "portfolio_stats.json and the Evaluator starves (L4513)."
    )
    # guard against a regression back to the broken 2-tuple gate on the
    # simulate block specifically (preceded by the 'Simulate mode' comment)
    assert 'if args.mode in ("simulate", "all"):\n        from phase_artifacts import save_json' not in s, (
        "the simulate phase gate reverted to ('simulate','all') — param-sweep "
        "would stop producing portfolio_stats again."
    )


def test_export_failloud_guard_present():
    """A mode-aware fail-loud guard must raise when a mode expected to produce
    the Evaluator's critical artifacts didn't (L4518) — no silent starve."""
    s = _src()
    # the guard checks portfolio_stats + sweep_df and raises in sim modes
    m = re.search(
        r'if args\.mode in \("simulate", "param-sweep", "all"\):\s*\n\s*_missing',
        s,
    )
    assert m, "the post-export fail-loud guard (mode-aware _missing check) is absent"
    assert "did not produce required" in s and "raise RuntimeError(" in s, (
        "the guard must raise RuntimeError naming the missing critical artifacts"
    )


def test_predictor_backtest_exempt_from_guard():
    """predictor-backtest legitimately produces neither portfolio_stats nor
    sweep_df — the guard must not require them for that mode."""
    s = _src()
    # the guard's mode set must NOT include predictor-backtest
    guard_block = s[s.index("FAIL-LOUD guard (L4518)"):]
    guard_block = guard_block[: guard_block.index("raise RuntimeError(") + 50]
    assert '"predictor-backtest"' not in guard_block, (
        "predictor-backtest must be exempt from the critical-artifact guard"
    )
