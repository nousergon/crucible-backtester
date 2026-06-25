"""
walk_forward_stability.py — report-card producer (config#1151 Batch C).

The System Report Card grades a *diagnostic* ``walk_forward_stability`` component:
across the rolling weekly fit windows, are the weight optimizer's
recommendations DRIFTING in a stable direction, or are they reversing sign
week-over-week (the classic over-fit-to-noise signature — the tuner chases the
last window's idiosyncrasies and undoes itself on the next)? "Weekly parameter
drift" is the issue's framing: a healthy walk-forward tuner converges; an
unhealthy one oscillates. Standard fit metrics (OOS degradation on a single
split) don't catch slow oscillation across many weeks; the reversal series does.

Pure-compute over the weight optimizer's already-computed cross-week stability
block (no new data read): the optimizer's ``_check_stability`` already loads the
prior weeks' weight recommendations from ``config/scoring_weights_history/`` and
flags direction REVERSALS per sub-score across the window. This producer turns
that reversal series into a headline ``stability_ratio`` = 1 - reversals /
max_possible_reversals over the loaded window — 1.0 = no reversal (monotone
drift, converging), → 0.0 = every consecutive step reversed (pure oscillation).
Always-emit (even insufficient_data) so the evaluator distinguishes "producer
didn't run" from "ran, too few weeks of history to judge stability yet".
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Sub-scores the weight optimizer tunes (mirrors weight_optimizer.SUB_SCORES).
# Each contributes (n_steps - 1) possible consecutive-delta reversals over an
# n-point window; the producer reads the actual count from the optimizer's
# stability block rather than re-deriving it, so this is only used to bound the
# denominator when the optimizer reports a window but no per-param breakdown.
_N_SUB_SCORES_DEFAULT = 2


def compute_walk_forward_stability(
    weight_result: dict | None,
    *,
    n_sub_scores: int = _N_SUB_SCORES_DEFAULT,
) -> dict:
    """Cross-week weight-recommendation stability (config#1151 — weekly drift).

    Args:
        weight_result: the ``weight_optimizer.compute_weights`` result. The
            load-bearing field is ``stability`` (the ``_check_stability`` block:
            ``{weeks_loaded, reversals: [...], stable: bool}``) — the prior-weeks
            history diff the optimizer already computes. ``None`` / non-``ok`` /
            missing-stability → ``insufficient_data``.
        n_sub_scores: number of tuned params (each contributes (steps-1)
            reversal opportunities). Defaults to the optimizer's 2-weight scheme.

    Returns a dict with the headline ``stability_ratio`` (1 - reversals /
    max_possible_reversals over the loaded window; 1.0 = monotone/converging,
    0.0 = fully oscillating), ``n_reversals``, ``weeks_loaded``, and ``stable``.
    Status ``insufficient_data`` when fewer than 2 prior weeks were loaded (no
    consecutive-step series to judge drift on — a brand-new or backfilled run).
    """
    wr = weight_result or {}
    if wr.get("status") != "ok":
        return {
            "status": "insufficient_data",
            "reason": f"weight_optimizer status={wr.get('status')!r} — no recommendation to assess drift on",
        }

    stab = wr.get("stability") or {}
    weeks_loaded = int(stab.get("weeks_loaded") or 0)
    reversals = stab.get("reversals") or []
    n_reversals = len(reversals)

    # The optimizer appends the current suggestion to the loaded history before
    # diffing, so the consecutive-delta series spans (weeks_loaded + 1) points →
    # (weeks_loaded) deltas → (weeks_loaded - 1) consecutive-pair comparisons per
    # sub-score. With < 2 weeks loaded there are < 1 comparisons → drift is not
    # yet judgeable (a freshly-seeded or backfilled history).
    n_steps = weeks_loaded + 1
    max_reversals = max(0, n_steps - 2) * max(1, n_sub_scores)
    if weeks_loaded < 2 or max_reversals <= 0:
        return {
            "status": "insufficient_data",
            "reason": f"only {weeks_loaded} prior week(s) of weight history loaded — need >=2 to judge drift",
            "weeks_loaded": weeks_loaded,
        }

    stability_ratio = round(1.0 - (n_reversals / max_reversals), 4)
    # Clamp: reversal counting is per-consecutive-pair so it cannot exceed
    # max_reversals, but guard against history-shape surprises.
    stability_ratio = max(0.0, min(1.0, stability_ratio))

    return {
        "status": "ok",
        "stability_ratio": stability_ratio,
        "n_reversals": n_reversals,
        "max_possible_reversals": max_reversals,
        "weeks_loaded": weeks_loaded,
        "stable": bool(stab.get("stable", n_reversals == 0)),
        "reversals": list(reversals),
    }
