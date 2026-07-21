"""sizing_shootout.py — observe-only S-slot sizing-arm comparison (config#3081).

Compares three position-sizing algorithms — the incumbent conviction-weighted
sizer, a risk-parity (inverse-realized-vol) sizer, and a fractional-Kelly
sizer — on the SAME historical signal stream, in the SAME vectorized synthetic
backtest window, with the SAME entry gates, universe/exposure constraints
(sector caps, position caps, cash policy), and transaction-cost treatment
(``fee_rate``). Only the raw position-SIZE formula differs between arms; see
``synthetic.vectorized_entries.compute_vectorized_entries`` (the
``sizing_arm`` parameter) for the shared-gates / arm-specific-sizing pipeline,
and ``synthetic.vectorized_sweep.run_sizing_shootout`` for the fan-out
orchestrator that runs the same date loop once per arm.

OBSERVE-only: this module GATES NOTHING and changes no serving path, same
posture as ``analysis/double_sort.py`` (ARCHITECTURE.md §14(e)). It reports,
per arm, the realized Sharpe / max-drawdown / turnover / realized-alpha of a
representative combo (or combos) so a human can judge whether risk-parity or
fractional-Kelly clears the promotion bar — beating the incumbent on BOTH
Sharpe AND max-drawdown, after cost. Promotion off that bar is a manual
Decision Queue ruling (config#3081); this module only measures and reports.

This module contains only the comparison / promotion-candidate logic; the
per-combo Sharpe / max-drawdown / turnover / total_alpha accounting reuses
``synthetic.vectorized_stats.compute_vectorized_stats`` (NAV-trajectory-based,
already validated against the vectorbt path) so all three arms are scored
identically and nothing is reimplemented here.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from synthetic.vectorized_stats import compute_vectorized_stats

logger = logging.getLogger(__name__)

# Metrics pulled from each arm's compute_vectorized_stats row into the
# per-arm summary. "sharpe"/"max_drawdown"/"turnover"/"realized_alpha" are
# the issue's requested top-line names; they map onto the vectorized-stats
# DataFrame's own column names (turnover isn't a vectorized_stats column —
# approximated via total_orders, see _summarize_arm below).
_INCUMBENT_ARM_LABEL = "conviction"


def _summarize_arm(
    stats_df: pd.DataFrame, *, combo_idx: int = 0,
) -> dict:
    """Reduce one arm's per-combo stats DataFrame to the shootout's top-line
    metrics for a single representative combo (row ``combo_idx``).

    A single representative combo is sufficient for a demo shootout (config
    #3081 doesn't require a full grid sweep per arm — "same historical signal
    stream, same window" is the bar). ``turnover`` is approximated as
    ``total_orders / n_combos`` count (vectorized_stats doesn't compute a
    dollar-turnover column) — documented as an ORDER-COUNT proxy, not a
    dollar-turnover ratio; a caller wanting true dollar turnover can derive
    it from the same orders_per_combo store this module already has access
    to via a follow-up.
    """
    if stats_df.empty or combo_idx >= len(stats_df):
        return {"status": "no_data"}
    row = stats_df.iloc[combo_idx]
    return {
        "status": str(row.get("status", "unknown")),
        "sharpe": _safe_float(row.get("sharpe_ratio")),
        "max_drawdown": _safe_float(row.get("max_drawdown")),
        "turnover": _safe_float(row.get("total_orders")),
        "realized_alpha": _safe_float(row.get("total_alpha")),
        "total_return": _safe_float(row.get("total_return")),
        "sortino": _safe_float(row.get("sortino_ratio")),
        "cvar_95": _safe_float(row.get("cvar_95")),
        "total_orders": int(row.get("total_orders", 0)),
        "total_trades": int(row.get("total_trades", 0)),
        "win_rate": _safe_float(row.get("win_rate")),
        "n_combos": int(len(stats_df)),
    }


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _beats_incumbent(arm_summary: dict, incumbent_summary: dict) -> bool:
    """Promotion bar (config#3081): beats incumbent on BOTH Sharpe AND
    max_drawdown, after cost. REPORTING ONLY — never touches live sizing
    config; documents which arm(s) clear the bar for a manual Decision
    Queue ruling.

    max_drawdown is stored as a negative-or-zero fraction (more negative =
    worse), so "beats" means "less negative" (closer to zero) i.e.
    ``arm_dd > incumbent_dd``.
    """
    a_sharpe = arm_summary.get("sharpe")
    i_sharpe = incumbent_summary.get("sharpe")
    a_dd = arm_summary.get("max_drawdown")
    i_dd = incumbent_summary.get("max_drawdown")
    if None in (a_sharpe, i_sharpe, a_dd, i_dd):
        return False
    return (a_sharpe > i_sharpe) and (a_dd > i_dd)


def compute_sizing_shootout(
    shootout_results: dict[str, tuple],
    *,
    run_date: str | None,
    init_cash: float,
    spy_prices: pd.Series | None,
    dates: pd.DatetimeIndex,
    combo_configs: list,
    fee_rate: float,
    incumbent_arm: str = _INCUMBENT_ARM_LABEL,
    representative_combo_idx: int = 0,
) -> dict:
    """Observe-only S-slot sizing-arm comparison (config#3081).

    ``shootout_results`` is the dict ``synthetic.vectorized_sweep.
    run_sizing_shootout`` returns: ``{arm_label: (orders_per_combo,
    diagnostics)}``, one entry per arm (conviction / risk_parity /
    fractional_kelly_<fraction>...). Each arm's ``diagnostics["nav_history"]``
    and ``orders_per_combo`` feed ``compute_vectorized_stats`` — the SAME
    Sharpe/max-drawdown/turnover/total_alpha accounting the rest of the
    vectorized sweep path uses, so all arms are scored identically and
    nothing is reimplemented here.

    ``fee_rate`` is the SAME fee rate every arm's underlying
    ``run_vectorized_sweep`` call was invoked with (verified by the caller —
    ``synthetic.vectorized_sweep.run_sizing_shootout`` forwards ``fee_rate``
    identically to every arm's call, so a cost-free Kelly "win" against a
    fee-aware incumbent cannot happen by construction). Recorded here purely
    for the artifact's cost-transparency field.

    Returns ``{status, run_date, arms: {label: {sharpe, max_drawdown,
    turnover, realized_alpha, ...}}, incumbent_arm, promotion_candidates,
    cost_model, note}``. GATES NOTHING; observe-only per ARCHITECTURE.md
    §14(e) (see ``analysis/double_sort.py`` for the same framing/citation).
    """
    if not shootout_results:
        return {"status": "no_arms", "run_date": run_date}

    arms: dict[str, dict] = {}
    for label, (orders_per_combo, diagnostics) in shootout_results.items():
        try:
            nav_history = diagnostics["nav_history"]
            stats_df = compute_vectorized_stats(
                nav_history=nav_history,
                init_cash=init_cash,
                spy_prices=spy_prices,
                dates=dates,
                orders_per_combo=orders_per_combo,
                combo_params=combo_configs,
            )
            arms[label] = _summarize_arm(
                stats_df, combo_idx=representative_combo_idx,
            )
        except Exception as exc:
            # Per-arm fail-soft: one arm's stats blowing up must not
            # take down the whole comparison — surfaces as a status
            # field on just that arm rather than aborting compute.
            logger.warning(
                "sizing_shootout: arm %s stats computation failed: %s",
                label, exc,
            )
            arms[label] = {"status": "error", "error": str(exc)}

    incumbent_summary = arms.get(incumbent_arm, {})
    promotion_candidates = [
        label for label, summary in arms.items()
        if label != incumbent_arm
        and summary.get("status") == "ok"
        and incumbent_summary.get("status") == "ok"
        and _beats_incumbent(summary, incumbent_summary)
    ]

    logger.info(
        "sizing_shootout (OBSERVE, NOT gated): arms=%s incumbent=%s "
        "promotion_candidates=%s. Promotion is a manual Decision Queue "
        "ruling (config#3081).",
        list(arms), incumbent_arm, promotion_candidates,
    )

    return {
        "status": "ok",
        "run_date": run_date,
        "arms": arms,
        "incumbent_arm": incumbent_arm,
        "promotion_candidates": promotion_candidates,
        "cost_model": {
            "fee_rate": fee_rate,
            "note": (
                "fee_rate flows into VectorizedSimulator.apply_buy/apply_sell "
                "identically for every arm (config#3081 run_sizing_shootout "
                "forwards fee_rate verbatim to every arm's run_vectorized_sweep "
                "call) — costs are baked into each arm's NAV trajectory, not "
                "applied post-hoc, so a cost-free Kelly 'win' cannot occur by "
                "construction."
            ),
        },
        "note": (
            "OBSERVE-only (config#3081, ARCHITECTURE.md §14(e)); gates "
            "nothing. Sizing-arm comparison shares identical entry gates / "
            "universe / exposure constraints across arms (synthetic."
            "vectorized_entries.compute_vectorized_entries sizing_arm param); "
            "only the raw position-weight formula differs. Promotion "
            "candidates are REPORTING ONLY — this never touches live sizing "
            "config; promotion is a manual Decision Queue ruling."
        ),
    }
