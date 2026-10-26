"""report.py — assembles the ``contribution_lift.json`` artifact.

Contract v1 (``contribution_lift_contract.md``), consumed by
``crucible-evaluator/grading/tiles/contribution_lift.py`` via the
``"contribution_lift": "contribution_lift.json"`` entry in
``grading/artifacts.py::ARTIFACT_MAP``.

Producer-side rules this module owns:

* ``red_line`` is 0.0 for every record and ``target`` is None — the producer
  never hand-sets a target; the evaluator resolves it through the T3
  thresholds registry (spec §3: an objective-CI-derived target, never a bare
  constant). Both live on the consumer side, so neither appears here.
* Status is never derived from ``sharpe_ratio``/Sortino while I7236 / I7237 /
  I7271 are open. ``dsr``/``psr`` ride along as diagnostics only.
* Always-emit: a ``skipped`` or ``error`` report is still a report. Absence of
  the S3 object must unambiguously mean "the producer never ran".
"""

from __future__ import annotations

import logging
from typing import Sequence

from analysis.contribution_lift.harness import (
    HORIZON_DAYS,
    N_FLOOR,
    TRIAL_PRODUCER,
    ReplayInputs,
    ReplaySpec,
    run_spec,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION: int = 1

#: The objective's name, carried on the artifact so a future second objective
#: can never be silently compared against records computed under this one.
OBJECTIVE_NAME: str = "log_alpha_21d_net_of_cost_vs_spy"


def _objective_block(inputs: ReplayInputs) -> dict:
    return {
        "name": OBJECTIVE_NAME,
        "horizon_days": inputs.horizon_days,
        "fees": inputs.fees,
        "slippage_bps": inputs.slippage_bps,
        "init_cash": inputs.init_cash,
    }


def _window_block(inputs: ReplayInputs) -> dict:
    return {
        "start": inputs.dates[0] if inputs.dates else None,
        "end": inputs.dates[-1] if inputs.dates else None,
        "n_cycles": len(inputs.dates),
        "n_floor": N_FLOOR,
    }


def _inputs_block(inputs: ReplayInputs) -> dict:
    return {
        "price_matrix_shape": [
            int(len(inputs.price_matrix)),
            int(len(inputs.price_matrix.columns)),
        ],
        "n_signal_dates": len(inputs.dates),
        "source_paths": list(inputs.source_paths),
    }


def _account_trials(inputs: ReplayInputs, n_arms: int) -> int:
    """Increment the fleet trial counter once, return the cumulative total.

    Every arm this harness simulates IS a trial in the DSR sense — each is a
    configuration whose objective was evaluated and could have been selected.
    Not counting them would make every DSR on this artifact (and on every
    OTHER producer reading the same shared counter) optimistically biased.

    Failure here is recorded and the report still emits with the last readable
    count: the counter is a cross-producer diagnostic, not the measurement.
    Failure mode swallowed: an S3 read/write error on
    ``backtest/cumulative_trial_count.json``. Recording surface: this WARNING
    in the evaluator's CloudWatch log group, plus ``dsr.n_trials`` on the
    artifact carrying the stale count rather than a fabricated one.
    """
    from nousergon_lib.quant.stats.trial_accumulator import (
        increment_trial_count,
        read_cumulative_trial_count,
    )

    try:
        result = increment_trial_count(
            TRIAL_PRODUCER,
            n_arms,
            inputs.run_date,
            bucket=inputs.bucket,
            s3_client=inputs.s3_client,
        )
        return int(result.get("total", 0) or 0)
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.warning(
            "contribution_lift: trial_accumulator increment failed (%s: %s) — "
            "falling back to the last readable cumulative count; dsr.n_trials "
            "on this artifact is stale, not fabricated",
            type(exc).__name__, exc,
        )
    try:
        current = read_cumulative_trial_count(
            bucket=inputs.bucket, s3_client=inputs.s3_client
        )
        return int(current.get("total", 0) or 0)
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.warning(
            "contribution_lift: trial_accumulator read also failed (%s: %s) — "
            "dsr.n_trials emitted as 0",
            type(exc).__name__, exc,
        )
        return 0


def _build_all(
    specs: Sequence[ReplaySpec], inputs: ReplayInputs
) -> list[tuple[ReplaySpec, object]]:
    """Build every spec's arms ONCE.

    ``build_arms`` is contractually pure, but calling it twice (once to count
    arms for the trial accumulator, once to run) would make a spec that
    accidentally is not pure produce an arm set the trial count does not
    describe. Built once, counted and run from the same object.
    """
    return [(spec, spec.build_arms(inputs)) for spec in specs]


def _n_arms(built: Sequence[tuple[ReplaySpec, object]]) -> int:
    """Arms this run will actually simulate — N/A specs simulate nothing."""
    from analysis.contribution_lift.harness import ArmSet

    return sum(
        len(b.all_arms()) for _spec, b in built if isinstance(b, ArmSet)
    )


def build_contribution_lift_report(
    inputs: ReplayInputs,
    specs: Sequence[ReplaySpec] | None = None,
    *,
    include_shadow: bool | None = None,
) -> dict:
    """Run every spec and return the contract-shaped artifact body.

    ``include_shadow`` (default: on for the registry run, off when a caller
    passes its own ``specs``) adds the top-level ``shadow`` block — ungraded
    diagnostics beside ``components``, see :func:`_shadow_block`.
    """
    if include_shadow is None:
        include_shadow = specs is None
    if specs is None:
        from analysis.contribution_lift.registry import SPECS

        specs = SPECS

    base = {
        "schema_version": SCHEMA_VERSION,
        "run_date": inputs.run_date,
        "objective": _objective_block(inputs),
        "window": _window_block(inputs),
        "inputs": _inputs_block(inputs),
    }

    if inputs.status != "ok":
        # Observational artifact: still emitted, with the loader's own reason.
        return base | {
            "status": inputs.status,
            "reason": inputs.reason or "loader reported a non-ok status",
            "n_trials_cumulative": 0,
            "components": [],
        }

    built = _build_all(specs, inputs)
    n_arms = _n_arms(built)
    n_trials = _account_trials(inputs, n_arms)

    components = [
        run_spec(spec, inputs, n_trials=n_trials, built=arms)
        for spec, arms in built
    ]

    logger.info(
        "contribution_lift: %d component(s) over %d cycles (%s -> %s), "
        "%d arms simulated, n_trials_cumulative=%d",
        len(components), len(inputs.dates),
        inputs.dates[0] if inputs.dates else "-",
        inputs.dates[-1] if inputs.dates else "-",
        n_arms, n_trials,
    )
    body = base | {
        "status": "ok",
        "n_trials_cumulative": n_trials,
        "components": components,
    }
    if include_shadow:
        body["shadow"] = _shadow_block(inputs, n_trials)
    return body


def _shadow_block(inputs: ReplayInputs, n_trials: int) -> dict:
    """Shadow diagnostics written beside ``components`` (alpha-engine-config-I12624).

    The evaluator's tile reads ``components`` only, so nothing here is graded.
    A failure here must never cost the live components their artifact: it is
    recorded in-band as ``status: error`` and logged at WARNING.
    """
    from analysis.contribution_lift.groups import executor_rules

    try:
        decomposition = executor_rules.exit_family_decomposition(
            inputs, n_trials=n_trials
        )
    except Exception as exc:  # noqa: BLE001 — shadow must not fail the live report
        logger.warning(
            "contribution_lift shadow exit_rules_decomposition failed: %s: %s",
            type(exc).__name__, exc,
        )
        decomposition = {
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}",
            "issue": executor_rules.ISSUE_12624,
        }
    return {"exit_rules_decomposition": decomposition}


def run_contribution_lift(config: dict, *, run_date: str | None = None) -> dict:
    """Live entry point called from ``evaluate.py``.

    Loads its own inputs — no caller injects cycles or a price matrix, which
    is the whole reason the two prior replay precedents never ran.
    """
    from analysis.contribution_lift.inputs import load_replay_inputs

    inputs = load_replay_inputs(config, run_date=run_date)
    return build_contribution_lift_report(inputs)


__all__ = [
    "HORIZON_DAYS",
    "OBJECTIVE_NAME",
    "SCHEMA_VERSION",
    "build_contribution_lift_report",
    "run_contribution_lift",
]
