"""L4520 — producer contract for config/executor_params.json (cross-repo).

Pins the VOCABULARY of the auto-tuned live executor config: every key any of
the five executor_params-feeding optimizers can emit (merged into the live key
by optimizer/assembler.py, the sole writer under cutover) must equal the set
declared in alpha-engine-config/private-docs/PIPELINE_CONTRACT.yaml
(boundary_id: executor_params). The executor's loader silently EXCLUDES
non-understood keys from the applied set — so an undeclared new key here is a
tuned param that never takes effect in live trading (the avg_volume_20d /
portfolio_stats silent-drop class, at the trading-config boundary).

This file hard-codes the declared set (per-repo CI can't import the config
repo's YAML — the test_scanner_consumer_contract.py precedent); the YAML is
the human SoT. If this test fails you are either (a) adding a producer key —
declare it in PIPELINE_CONTRACT.yaml AND confirm the executor understands it
(its consumer test + _PARAM_MAP/advisory list), or (b) removing one — update
the YAML + consumer the same way.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer import (
    barrier_sizing_optimizer,
    executor_optimizer,
    predictor_sizing_optimizer,
    stance_sizing_optimizer,
    trigger_optimizer,
)

# ── The declared vocabulary (mirror of PIPELINE_CONTRACT.yaml executor_params
#    required_top_level_fields) ────────────────────────────────────────────────
DECLARED = {
    # applied by the executor (_PARAM_MAP)
    "atr_multiplier", "time_decay_reduce_days", "time_decay_exit_days",
    "min_score", "max_position_pct", "reduce_fraction",
    "atr_sizing_target_risk", "confidence_sizing_min",
    "confidence_sizing_range", "staleness_decay_per_day",
    "earnings_sizing_reduction", "earnings_proximity_days",
    "momentum_gate_threshold", "correlation_block_threshold",
    "profit_take_pct", "momentum_exit_threshold",
    "barrier_win_prob_sizing_min", "barrier_win_prob_sizing_range",
    # applied special (non-numeric) keys
    "disabled_triggers", "use_p_up_sizing", "p_up_sizing_blend",
    "barrier_win_prob_sizing_enabled",
    # emitted but NOT applied (named gap — stance overlay application unwired)
    "stance_size_momentum", "stance_size_value", "stance_size_quality",
    "stance_size_catalyst",
    # provenance / metadata
    "updated_at", "assembled_by", "fit_target", "best_sharpe", "best_alpha",
    "best_sortino", "improvement_pct", "n_combos_tested", "manual_override",
    "disabled_triggers_updated_at", "barrier_win_prob_sizing_updated_at",
    "barrier_win_prob_sizing_ic", "p_up_sizing_updated_at", "p_up_sizing_ic",
    "stance_sizing_updated_at", "stance_sizing_alpha_spread",
}

# Keys that appear in the live artifact without a CURRENT producer code path:
# - manual_override: operator-written (executor advisory list documents it).
# - confidence_sizing_min/range: retired from SAFE_PARAMS (L300 2026-06-01 —
#   the sweep over them was a silent no-op) but still executor-understood
#   (_PARAM_MAP) and may persist in the live key via the assembler's merge
#   base (the current live config), so they stay declared.
_DECLARED_NOT_EMITTED = {
    "manual_override", "confidence_sizing_min", "confidence_sizing_range",
}

# executor_optimizer.apply()'s legacy payload envelope (the assembler's merge
# base inherits these), + the assembler's own stamp (assembler.py write path:
# payload["updated_at"] / payload["assembled_by"]).
_ENVELOPE = {
    "updated_at", "fit_target", "best_sharpe", "best_alpha", "best_sortino",
    "improvement_pct", "n_combos_tested", "assembled_by",
}


def _emittable() -> set[str]:
    keys: set[str] = set(executor_optimizer.SAFE_PARAMS)
    keys |= _ENVELOPE
    keys |= set(trigger_optimizer._build_overlay_params({})[0])
    keys |= set(barrier_sizing_optimizer._build_overlay_params({})[0])
    keys |= set(predictor_sizing_optimizer._build_overlay_params({})[0])
    keys |= set(
        stance_sizing_optimizer._build_overlay_params({
            "recommended_multipliers": {s: 1.0 for s in stance_sizing_optimizer._STANCES},
            "stance_alpha_spread": 0.0,
        })[0]
    )
    return keys


def test_every_emittable_key_is_declared():
    undeclared = _emittable() - DECLARED
    assert not undeclared, (
        f"executor_params producer emits key(s) the cross-repo contract does "
        f"not declare: {sorted(undeclared)}. Declare them in alpha-engine-"
        f"config/private-docs/PIPELINE_CONTRACT.yaml (executor_params) AND "
        f"confirm the executor loader understands them — an undeclared key is "
        f"silently dropped from the applied set in live trading."
    )


def test_declared_set_has_no_orphans():
    # Symmetric leg: a declared key with no producer path (and not operator-
    # written) is contract rot — remove it from the YAML + this mirror.
    orphans = DECLARED - _emittable() - _DECLARED_NOT_EMITTED
    assert not orphans, (
        f"PIPELINE_CONTRACT.yaml declares executor_params key(s) no optimizer "
        f"can emit: {sorted(orphans)} — stale contract, prune both sides."
    )


def test_safe_params_exclude_dangerous_keys():
    # The auto-tune allowlist must never grow the operator-only risk keys
    # (drawdown_circuit_breaker / max_sector_pct / max_equity_pct are excluded
    # by design — executor_optimizer.SAFE_PARAMS header).
    dangerous = {"drawdown_circuit_breaker", "max_sector_pct", "max_equity_pct"}
    assert not dangerous & set(executor_optimizer.SAFE_PARAMS)
    assert not dangerous & DECLARED
