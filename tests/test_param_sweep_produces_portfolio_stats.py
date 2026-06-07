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
    """Outcome taxonomy (L4523): the guard must FAIL LOUD on ABSENT critical
    artifacts (portfolio_stats absent, sweep_df None) but treat an EMPTY sweep
    as a valid no-op (not a crash) — a risk/score gate must never kill the
    process."""
    s = _src()
    # absent portfolio_stats → raise
    assert "did not produce portfolio_stats" in s and "raise RuntimeError(" in s, (
        "guard must raise on ABSENT portfolio_stats"
    )
    # sweep_df None (phase didn't run) → raise; distinct from empty
    assert "sweep_df is ABSENT" in s, "guard must raise on sweep_df is None (absent)"
    # EMPTY sweep_df → no-op (WARN + alert), NOT raise
    assert 'getattr(sweep_df, "empty", True)' in s, "empty-sweep branch missing"
    assert "valid no-op" in s and "[outcome] sweep_df is EMPTY" in s, (
        "an empty sweep must be a logged valid no-op, not a fatal error "
        "(the 'risk gate killed the process' symptom — L4523)"
    )


def test_empty_sweep_is_exported_present_not_skipped():
    """An empty sweep_df must still be written (present-but-empty) so the
    Evaluator finds the artifact and no-ops, rather than seeing it absent."""
    s = _src()
    assert "if sweep_df is not None:" in s, (
        "export must write sweep_df whenever the frame exists (incl. empty), "
        "skipping only a None frame"
    )


def test_predictor_backtest_exempt_from_guard():
    """predictor-backtest legitimately produces neither portfolio_stats nor
    sweep_df — the guard must not require them for that mode."""
    s = _src()
    guard_block = s[s.index("Outcome taxonomy (L4523)"):]
    guard_block = guard_block[: guard_block.index("logger.warning")]
    assert '"predictor-backtest"' not in guard_block, (
        "predictor-backtest must be exempt from the critical-artifact guard"
    )
