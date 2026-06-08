"""
tests/test_pipeline_manifest.py — the declarative pipeline manifest's query +
contract logic (L4526, plan §6 Phase 4). The manifest↔code/consumer bindings
live in test_evaluator_artifact_contract.py; this file covers the pure helpers.
"""

from __future__ import annotations

import pytest

import pipeline_manifest as manifest
from pipeline_manifest import Stage, contract_violations


def test_stage_by_name_and_missing():
    assert manifest.stage_by_name("simulate").name == "simulate"
    with pytest.raises(KeyError):
        manifest.stage_by_name("does-not-exist")


def test_producers_of_critical_artifacts():
    assert [s.name for s in manifest.producers_of("portfolio_stats.json")] == ["simulate"]
    assert [s.name for s in manifest.producers_of("sweep_df.parquet")] == ["param_sweep"]
    assert manifest.producers_of("nope") == []


def test_stages_for_mode_excludes_consumer_only():
    names = {s.name for s in manifest.stages_for_mode("param-sweep")}
    assert names == {"simulate", "param_sweep"}
    # evaluator is consumer-only → never a producer for any mode
    assert "evaluator" not in {s.name for s in manifest.stages_for_mode("all")}


def test_simulate_mode_produces_portfolio_stats_not_sweep():
    """--mode=simulate runs `simulate` (portfolio_stats) but NOT `param_sweep`."""
    names = {s.name for s in manifest.stages_for_mode("simulate")}
    assert "simulate" in names
    assert "param_sweep" not in names


def test_evaluator_critical_is_the_evaluator_requires():
    assert manifest.evaluator_critical() == frozenset(
        {"sweep_df.parquet", "portfolio_stats.json"}
    )


def test_contract_satisfied_for_sf_mode():
    assert contract_violations(manifest.SF_BACKTESTER_MODE) == []


def test_contract_satisfied_for_all_mode():
    assert contract_violations("all") == []


def test_contract_violation_when_mode_orphans_a_critical_artifact():
    """--mode=simulate runs no param_sweep producer → sweep_df.parquet is
    orphaned → the Evaluator would starve. This is exactly the L4513 class the
    contract exists to catch."""
    violations = contract_violations("simulate")
    assert len(violations) == 1
    assert "sweep_df.parquet" in violations[0]
    assert "L4513" in violations[0]


def test_contract_violation_for_nonproducing_mode():
    """A mode that runs no producers at all orphans BOTH critical artifacts."""
    violations = contract_violations("signal-quality")
    assert len(violations) == 2
    arts = {a for a in ("sweep_df.parquet", "portfolio_stats.json")
            if any(a in v for v in violations)}
    assert arts == {"sweep_df.parquet", "portfolio_stats.json"}


def test_contract_violations_is_pure():
    """contract_violations must not mutate the manifest (frozen dataclasses)."""
    before = list(manifest.STAGES)
    contract_violations("simulate")
    assert list(manifest.STAGES) == before
    with pytest.raises(Exception):
        Stage(name="x").modes.append("y")  # type: ignore[attr-defined]  # tuple is immutable
