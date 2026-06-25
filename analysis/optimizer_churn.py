"""
optimizer_churn.py — report-card producer (config#1151 Batch C).

The System Report Card grades a *critical* ``optimizer_churn`` component: is the
weekly weight optimizer making STABLE, incremental param moves, or is it thrashing
the live config cycle-over-cycle (a sign of an over-fit / noise-chasing fit
target)? An optimizer that proposes a +9% quant-weight swing one Saturday and a
-9% swing the next is destabilising the very config it's meant to tune, even if
each individual move clears the guardrail cap. Without this, the report card
can't tell the Director "the tuner is converging" from "the tuner is oscillating
against the guardrails."

Pure-compute over the already-computed weight-optimizer result (no new data
read): it reads the per-param ``changes`` the optimizer proposed this cycle and
measures churn as the largest single proposed move RELATIVE TO the guardrail cap
the optimizer enforces (``max_single_change``). churn_ratio < 1.0 means every
move sat inside the cap (healthy, incremental); >= 1.0 means a proposed move hit
or exceeded the cap and was clamped/blocked — the tuner WANTED to thrash and the
guardrail had to step in. Always-emit (even insufficient_data) so the evaluator
distinguishes "producer didn't run" from "ran, optimizer had no usable
recommendation this cycle".
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def compute_optimizer_churn(
    weight_result: dict | None,
    *,
    guardrail_cap: float | None = None,
) -> dict:
    """Per-cycle weight-optimizer param churn vs the guardrail cap (config#1151).

    Args:
        weight_result: the ``weight_optimizer.compute_weights`` result. The
            load-bearing fields are ``changes`` (``{sub_score: signed_delta}``,
            the proposed per-param weight move this cycle) and ``status``. A
            ``None`` / non-``ok`` / change-less result → ``insufficient_data``.
        guardrail_cap: the max-single-change guardrail the optimizer enforces
            (fraction, e.g. 0.10 = a 10-percentage-point cap on any one weight).
            Defaults to the optimizer's own ``_MAX_SINGLE_CHANGE`` so the producer
            and the optimizer can never silently disagree on the cap.

    Returns a dict with the per-param absolute deltas, the headline
    ``churn_ratio`` (max |delta| / cap — how close the largest proposed move came
    to the guardrail), and ``within_guardrails`` (churn_ratio < 1.0). Status
    ``insufficient_data`` when the optimizer produced no usable recommendation.
    """
    if guardrail_cap is None:
        # Import lazily + defensively: the producer must degrade, not crash, if
        # the optimizer module isn't importable in a given context.
        try:
            from optimizer.weight_optimizer import _MAX_SINGLE_CHANGE
            guardrail_cap = float(_MAX_SINGLE_CHANGE)
        except Exception:  # noqa: BLE001 — degrade to a documented default
            guardrail_cap = 0.10

    wr = weight_result or {}
    if wr.get("status") != "ok":
        return {
            "status": "insufficient_data",
            "reason": f"weight_optimizer status={wr.get('status')!r} — no usable recommendation this cycle",
            "guardrail_cap": guardrail_cap,
        }

    changes = wr.get("changes") or {}
    abs_deltas = {
        k: round(abs(float(v)), 6)
        for k, v in changes.items()
        if v is not None
    }
    if not abs_deltas:
        return {
            "status": "insufficient_data",
            "reason": "weight_optimizer reported no per-param changes this cycle",
            "guardrail_cap": guardrail_cap,
        }

    max_param, max_delta = max(abs_deltas.items(), key=lambda kv: kv[1])
    # Churn relative to the guardrail cap: the report card grades how hard the
    # tuner pushed against the cap, not the raw magnitude (which is config-unit
    # dependent). cap <= 0 is degenerate (no guardrail) → ratio is undefined.
    churn_ratio = round(max_delta / guardrail_cap, 4) if guardrail_cap > 0 else None
    n_meaningful = sum(1 for d in abs_deltas.values() if d > 0)

    return {
        "status": "ok",
        "churn_ratio": churn_ratio,
        "max_abs_change": round(max_delta, 6),
        "max_change_param": max_param,
        "guardrail_cap": guardrail_cap,
        "within_guardrails": bool(churn_ratio is not None and churn_ratio < 1.0),
        "n_params_changed": n_meaningful,
        "per_param_abs_change": abs_deltas,
    }
