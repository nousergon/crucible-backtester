"""Behavioral tests for the 3-way phase outcome taxonomy (L4523).

ARCHITECTURE.md §22: the backtester delivers config ACTIONS + a report card,
and a 0-result is legitimate only when the inputs were validated first. The
orchestration encoding is a 3-way ``PhaseStatus`` (SUCCESS | EMPTY | FAILURE),
NOT binary ok/fail — because binary forces the EMPTY case into one of two
harmful answers (silently drop the config action, or crash the whole SF when a
risk/score gate merely did its job — the 2026-06-06 symptom).

These exercise the real classifier + dataclass (not source-grep), so a refactor
that preserves behavior keeps them green while a behavior change is caught.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backtest import classify_simulation_outcome
from pipeline_common import PhaseOutcome, PhaseStatus


# ── FAILURE: infra/contract break — fail loud ────────────────────────────────


@pytest.mark.parametrize("mode", ["simulate", "param-sweep", "all"])
def test_absent_portfolio_stats_is_failure(mode):
    """portfolio_stats absent = the phase didn't run / errored → FAILURE."""
    out = classify_simulation_outcome(mode, portfolio_stats=None, sweep_df=pd.DataFrame({"sharpe_ratio": [1.0]}))
    assert out.is_failure
    assert out.status is PhaseStatus.FAILURE
    assert "did not produce portfolio_stats" in out.reason


def test_empty_portfolio_stats_dict_is_failure():
    """An empty/falsey portfolio_stats is treated as absent → FAILURE."""
    out = classify_simulation_outcome("all", portfolio_stats={}, sweep_df=pd.DataFrame({"sharpe_ratio": [1.0]}))
    assert out.is_failure


def test_none_sweep_df_is_failure():
    """sweep_df is None (param_sweep phase didn't run) → FAILURE, distinct
    from an empty frame."""
    out = classify_simulation_outcome("param-sweep", portfolio_stats={"sharpe_ratio": 1.0}, sweep_df=None)
    assert out.is_failure
    assert "sweep_df is ABSENT" in out.reason


# ── EMPTY: ran, no admissible result — valid no-op (do NOT crash) ─────────────


def test_empty_sweep_df_is_empty_not_failure():
    """An EMPTY frame = ran, no admissible combo (all gated by score/risk) →
    EMPTY (valid no-op), NEVER FAILURE. This is the 2026-06-06 fix: a risk/
    score gate doing its job must not kill the process."""
    out = classify_simulation_outcome("param-sweep", portfolio_stats={"sharpe_ratio": 1.0}, sweep_df=pd.DataFrame())
    assert out.is_empty
    assert not out.is_failure
    assert out.status is PhaseStatus.EMPTY
    assert out.n_admissible == 0
    assert out.degeneracy_reason  # names WHY it was empty
    assert "valid no-op" in out.reason
    assert "[outcome] sweep_df is EMPTY" in out.reason


# ── SUCCESS: admissible result present ────────────────────────────────────────


def test_nonempty_sweep_with_stats_is_success():
    df = pd.DataFrame({"sharpe_ratio": [0.5, 1.2, 0.9]})
    out = classify_simulation_outcome("all", portfolio_stats={"sharpe_ratio": 1.0}, sweep_df=df)
    assert out.is_success
    assert out.status is PhaseStatus.SUCCESS
    assert out.n_admissible == 3


# ── PhaseOutcome record shape ────────────────────────────────────────────────


def test_phase_outcome_to_dict_is_json_shaped():
    out = PhaseOutcome(
        status=PhaseStatus.EMPTY,
        phase="export_artifacts",
        reason="r",
        n_admissible=0,
        degeneracy_reason="all gated",
    )
    d = out.to_dict()
    assert d["status"] == "empty"  # enum serialized to its value
    assert d["phase"] == "export_artifacts"
    assert d["n_admissible"] == 0
    assert d["degeneracy_reason"] == "all gated"
    assert isinstance(d["artifacts_written"], list)


def test_status_predicates_are_mutually_exclusive():
    for status in PhaseStatus:
        out = PhaseOutcome(status=status, phase="p")
        flags = [out.is_success, out.is_empty, out.is_failure]
        assert sum(flags) == 1, f"{status} set more than one predicate"
