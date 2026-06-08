"""
pipeline_manifest.py — declarative single source of truth for the backtester
pipeline's stage × mode × requires × produces contract (L4526, plan §6 Phase 4
/ §2 P2 + P7).

The Saturday SF, ``backtest.py``'s runtime mode-gates, and the cross-stage
contract test historically each encoded the pipeline topology independently —
which drifted. The 2026-05-16 SF split set the Backtester state to
``--mode=param-sweep`` while the ``simulate`` phase that produces
``portfolio_stats.json`` was gated to ``simulate``/``all`` — so param-sweep
silently stopped producing a CRITICAL Evaluator input and the Evaluator
hard-failed for ~3 weeks with no signal (L4513).

This manifest declares the producer/consumer contract ONCE. The contract test
(``tests/test_evaluator_artifact_contract.py``) consumes it AND binds it back to
the live producer (``backtest.py`` mode-gates) and consumer (``evaluate.py``'s
``critical`` set) via the per-stage ``gate_pattern`` regex + a critical-set
cross-check — so the manifest cannot silently drift from reality (a manifest
nobody verifies is just another place to drift). Follow-on Phase-4 slices wire
the pre-spend preflight gate + the SF generator to the same manifest, and extend
it to every stage with skip-flags + cost.

Per [[feedback_sota_institutional_default_no_shortcuts]] + the feature-store
``test_schema_contract.py`` precedent (a declarative contract enforced at CI).
"""

from __future__ import annotations

from dataclasses import dataclass

# The --mode the Saturday SF "Backtester" state invokes (spot_backtest.sh
# --mode=param-sweep). SINGLE SOURCE OF TRUTH — the contract test reads it from
# here; if the SF topology changes the mode, change it here and the contract
# re-validates against the manifest. (Was hardcoded in the contract test as
# BACKTESTER_SF_MODE — lifted here per plan P7.)
SF_BACKTESTER_MODE = "param-sweep"

# Every --mode backtest.py accepts. The contract is validated for the SF mode;
# this tuple lets the contract test sanity-check the whole mode space.
ALL_MODES: tuple[str, ...] = (
    "signal-quality",
    "simulate",
    "param-sweep",
    "all",
    "predictor-backtest",
    "portfolio-optimizer-backtest",
    "smoke",
)


@dataclass(frozen=True)
class Stage:
    """A backtester pipeline stage's contract surface.

    ``modes`` are the ``--mode`` values that RUN this stage as a producer (its
    ``if args.mode in (...)`` gate in backtest.py). ``produces`` / ``requires``
    are artifact basenames (the Evaluator-critical names, e.g.
    ``portfolio_stats.json``). ``gate_pattern`` is a regex whose first capture
    group is the comma-quoted mode list of the stage's live gate in
    ``backtest.py`` — the anti-drift binding the contract test asserts against.
    """

    name: str
    produces: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    modes: tuple[str, ...] = ()
    auto_skippable: bool = False
    consumer_only: bool = False  # consumes but is not a mode-gated producer
    gate_pattern: str | None = None


# Contract-relevant stages. Intentionally scoped to the producer/consumer
# contract axis for the two Evaluator-critical artifacts (the L4513 failure
# surface); each producer carries a gate_pattern so the test binds it to
# backtest.py. Extending the manifest to ALL stages (+ skip flags + cost) is the
# next Phase-4 slice — stages are added here only WITH a verification binding so
# the manifest never becomes unverified documentation that can drift.
STAGES: tuple[Stage, ...] = (
    Stage(
        name="simulate",
        produces=("portfolio_stats.json",),
        modes=("simulate", "param-sweep", "all"),
        auto_skippable=True,
        gate_pattern=(
            r"if args\.mode in \(([^)]*)\):\s*\n"
            r"\s*from phase_artifacts import save_json, load_json"
        ),
    ),
    Stage(
        name="param_sweep",
        produces=("sweep_df.parquet",),
        modes=("param-sweep", "all"),
        auto_skippable=True,
        gate_pattern=r"# ── Param sweep[^\n]*\n\s*if args\.mode in \(([^)]*)\):",
    ),
    Stage(
        name="evaluator",
        requires=("sweep_df.parquet", "portfolio_stats.json"),
        consumer_only=True,
    ),
)


def stage_by_name(name: str) -> Stage:
    for s in STAGES:
        if s.name == name:
            return s
    raise KeyError(f"no stage named {name!r} in the pipeline manifest")


def producer_stages() -> list[Stage]:
    """Stages that produce at least one artifact (exclude consumer-only)."""
    return [s for s in STAGES if s.produces and not s.consumer_only]


def producers_of(artifact: str) -> list[Stage]:
    """Every stage whose ``produces`` includes ``artifact`` (any mode)."""
    return [s for s in STAGES if artifact in s.produces]


def stages_for_mode(mode: str) -> list[Stage]:
    """Producer stages that run in ``mode`` (its mode-gate includes ``mode``)."""
    return [s for s in producer_stages() if mode in s.modes]


def evaluator_critical() -> frozenset[str]:
    """The Evaluator's critical-input set — the artifacts it hard-requires.

    Declared as the ``evaluator`` stage's ``requires``. The contract test
    cross-checks this against ``evaluate.py``'s actual ``critical`` set so the
    two can't drift.
    """
    return frozenset(stage_by_name("evaluator").requires)


def contract_violations(mode: str) -> list[str]:
    """Return human-readable contract violations for a run ``mode`` (empty ==
    satisfied).

    The contract: for the given mode, every Evaluator-critical artifact must be
    produced by some stage whose mode-gate includes that mode. A violation is
    exactly the L4513 failure class — a mode that runs the Evaluator but doesn't
    run a producer for one of its critical inputs, silently starving it.
    """
    runnable = {s.name for s in stages_for_mode(mode)}
    violations: list[str] = []
    for artifact in sorted(evaluator_critical()):
        in_mode = [s.name for s in producers_of(artifact) if s.name in runnable]
        if not in_mode:
            all_producers = [s.name for s in producers_of(artifact)]
            violations.append(
                f"--mode={mode}: no stage producing Evaluator-critical "
                f"{artifact!r} runs in this mode (producers={all_producers}, "
                f"none mode-gated for {mode!r}) — the Evaluator hard-requires it "
                f"and would starve (the L4513 silent-starvation class)."
            )
    return violations
