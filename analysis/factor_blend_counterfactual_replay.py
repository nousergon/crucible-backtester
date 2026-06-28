"""factor_blend_counterfactual_replay.py — the REAL factor-blend optimizer signal.

The sibling ``analysis/factor_blend_sensitivity.py`` is the *cheap proxy*: it
reads realized ``score_performance`` outcomes and can detect that the configured
per-regime stance ordering doesn't match the realized Sortino ordering. What it
CANNOT do is answer the counterfactual "Sortino *would have been* higher if we
had used factor-blend weights X" — because it never re-scores the universe under
alternate weights and never runs the resulting portfolio through the simulator.

This module closes that gap (config#749 / alpha-engine-backtester deferred
follow-up #3 from PR #206). Given:

  * a candidate universe per cycle with its raw sub-factor scores
    (momentum / quality / value / low_vol), and
  * a price matrix + SPY series for the simulation window,

it replays the score aggregator's factor blend under a GRID of alternate
``factor_blend`` weight tuples. For each weight tuple it:

  1. recomputes each candidate's composite score per cycle as the signed,
     weighted sum of the (cross-sectionally z-scored) sub-factors — the same
     blend formula the live aggregator applies (mirrors
     ``analysis/end_to_end._scanner_factor_counterfactual``'s per-cycle z-score
     + sign-oriented sleeve construction),
  2. selects the top-N candidates per cycle (count-matched to ``picks_per_cycle``),
  3. turns the selections into an ENTER/EXIT order list (fixed holding horizon),
  4. runs that order list through the REAL simulation plumbing —
     ``vectorbt_bridge.orders_to_portfolio`` + ``vectorbt_bridge.portfolio_stats``
     (the same two calls ``backtest._run_simulation_loop`` makes), and
  5. reports per-weighting ``total_return / total_alpha / sortino_ratio /
     max_drawdown / psr``.

This is the "real" optimizer signal the ROADMAP asked for; the sensitivity
analyzer is the proxy that runs every week without the heavy sim.

Design constraints (mirroring the sibling counterfactuals):
  * **Deterministic** — no RNG; ties broken by ticker so a fixed universe always
    yields identical per-weighting metrics.
  * **Config-driven + default OFF** — the live backtest/evaluator does NOT run
    this unless ``factor_blend_counterfactual.enabled`` is truthy in config, so
    wiring it in never changes the live backtest path (same opt-in convention as
    the other heavy counterfactual analyses).
  * **Pure replay** — reuses ``vectorbt_bridge`` simulation; no new simulation
    plumbing is invented here.

Plan doc: alpha-engine-docs/private/scanner-260514.md PR 6 (follow-up #3).
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# The four within-sector factor sleeves the aggregator's ``factor_blend`` mixes.
# Sub-factor score columns are already sign-oriented (higher = better) by the
# research scoring layer, matching ``score_performance``'s ``*_score`` columns
# and ``factor_blend_sensitivity.DEFAULT_REGIME_WEIGHTS`` keys.
FACTOR_SCORE_COLUMNS: tuple[str, ...] = (
    "momentum_score",
    "quality_score",
    "value_score",
    "low_vol_score",
)

# The equal-weight blend — used as the baseline reference so the report can
# express each variant's lift relative to "blend everything equally." The
# acceptance test pins that supplying this exact tuple reproduces the baseline.
EQUAL_WEIGHTS: dict[str, float] = {c: 0.25 for c in FACTOR_SCORE_COLUMNS}

# Default holding horizon (trading days in the price matrix) for a selection
# before it is exited. Kept conservative + config-overridable.
DEFAULT_HOLD_DAYS: int = 10

# Default per-cycle pick count when the caller doesn't pin one.
DEFAULT_PICKS_PER_CYCLE: int = 5


def _normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Coerce a partial/absolute weight mapping to a full sub-factor mapping.

    Missing sub-factors default to 0.0 (the sleeve is simply not blended in).
    Unknown keys are dropped. Weights are used *as-is* (signed); negative weights
    penalize the sleeve, exactly like the live aggregator's per-regime blend
    (e.g. BULL low_vol = -0.10). They are intentionally NOT renormalized to sum
    to 1 — the composite is a cross-sectional ranking signal, so a uniform scale
    factor leaves the top-N selection unchanged, and preserving the raw signed
    weights keeps the variant labels readable.
    """
    return {c: float(weights.get(c, 0.0)) for c in FACTOR_SCORE_COLUMNS}


def _zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score within one cycle. Degenerate (zero/near-zero
    std) → all zeros, so a flat sub-factor contributes nothing rather than
    NaN-poisoning the composite. Mirrors ``end_to_end._scanner_factor_
    counterfactual``'s ddof=0 z-score convention."""
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std(ddof=0)
    if not sd or sd <= 1e-12:
        return pd.Series(0.0, index=s.index)
    return (x - x.mean()) / sd


def compute_composite_scores(
    cycle_df: pd.DataFrame,
    weights: Mapping[str, float],
) -> pd.Series:
    """Recompute composite scores for one cycle under ``weights``.

    Args:
        cycle_df: one cycle's candidates, indexed by ticker, with the
            ``FACTOR_SCORE_COLUMNS`` present (missing columns treated as absent
            sleeves). Raw sub-factor scores; cross-sectionally z-scored here.
        weights: signed per-sub-factor weights (partial allowed).

    Returns:
        Series of composite scores indexed by ticker. ``composite = Σ_f w_f *
        z(sub_factor_f)`` over the sleeves present in the frame.
    """
    w = _normalize_weights(weights)
    composite = pd.Series(0.0, index=cycle_df.index)
    for col in FACTOR_SCORE_COLUMNS:
        if col in cycle_df.columns and w[col] != 0.0:
            composite = composite + w[col] * _zscore(cycle_df[col])
    return composite


def _select_topn(scores: pd.Series, n: int) -> list[str]:
    """Top-``n`` tickers by score desc, ties broken by ticker asc for
    determinism. NaN scores drop out."""
    valid = scores.dropna()
    if valid.empty:
        return []
    ordered = valid.sort_index().sort_values(ascending=False, kind="stable")
    return list(ordered.head(n).index)


def build_orders_for_weights(
    cycles: Sequence[Mapping[str, Any]],
    weights: Mapping[str, float],
    price_matrix: pd.DataFrame,
    *,
    picks_per_cycle: int = DEFAULT_PICKS_PER_CYCLE,
    hold_days: int = DEFAULT_HOLD_DAYS,
    init_cash: float = 1_000_000.0,
) -> list[dict]:
    """Re-score each cycle under ``weights`` and emit an ENTER/EXIT order list.

    Each cycle is ``{"date": <str|Timestamp>, "candidates": <DataFrame indexed
    by ticker with FACTOR_SCORE_COLUMNS>}``. The selected top-N are entered on
    the cycle date and exited ``hold_days`` trading rows later (or at the last
    available price row). Order shape matches what
    ``backtest._run_simulation_loop`` feeds ``orders_to_portfolio``:
    ``{"date", "ticker", "action", "shares", "price_at_order"}``.

    Capital per name is equal-weighted across the cycle's selection from a fixed
    ``init_cash / picks_per_cycle`` budget, so share counts are deterministic
    and comparable across weight variants. This is a deliberately simple,
    deterministic sizing rule — the goal is to isolate the *blend weighting*'s
    effect on selection, not to reproduce the executor's full risk sizing.
    """
    dates = price_matrix.index
    orders: list[dict] = []
    if len(dates) == 0:
        return orders
    per_name_budget = float(init_cash) / max(1, picks_per_cycle)

    for cycle in cycles:
        raw_date = cycle.get("date")
        cand = cycle.get("candidates")
        if cand is None or len(cand) == 0:
            continue
        cdate = pd.Timestamp(raw_date)
        if cdate not in dates:
            # Snap to the first price row on/after the cycle date; skip if the
            # cycle falls entirely after the price matrix.
            future = dates[dates >= cdate]
            if len(future) == 0:
                continue
            cdate = future[0]
        entry_pos = dates.get_loc(cdate)
        exit_pos = min(entry_pos + hold_days, len(dates) - 1)
        exit_date = dates[exit_pos]

        scores = compute_composite_scores(cand, weights)
        for ticker in _select_topn(scores, picks_per_cycle):
            if ticker not in price_matrix.columns:
                continue
            entry_price = price_matrix.at[cdate, ticker]
            if pd.isna(entry_price) or float(entry_price) <= 0.0:
                continue
            shares = float(np.floor(per_name_budget / float(entry_price)))
            if shares <= 0.0:
                continue
            orders.append({
                "date": cdate.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "action": "ENTER",
                "shares": shares,
                "price_at_order": float(entry_price),
            })
            if exit_pos > entry_pos:
                orders.append({
                    "date": exit_date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "action": "EXIT",
                })
    return orders


def replay_weight_variant(
    cycles: Sequence[Mapping[str, Any]],
    weights: Mapping[str, float],
    price_matrix: pd.DataFrame,
    *,
    spy_prices: Optional[pd.Series] = None,
    picks_per_cycle: int = DEFAULT_PICKS_PER_CYCLE,
    hold_days: int = DEFAULT_HOLD_DAYS,
    init_cash: float = 1_000_000.0,
    fees: float = 0.001,
) -> dict[str, Any]:
    """Replay ONE weight tuple through the real simulation and return metrics.

    Builds the order list for ``weights`` then runs it through the SAME plumbing
    the production backtest uses (``vectorbt_bridge.orders_to_portfolio`` +
    ``portfolio_stats``). Returns a metrics dict with ``status`` plus the headline
    fields the ROADMAP acceptance asks for (total_return / alpha / Sortino /
    maxDD) and ``psr`` when computable.

    ``status`` is ``"ok"`` with metrics, or ``"no_orders"`` when the weighting
    produced no entries (e.g. all sub-factors flat → empty selection).
    """
    # Deferred import: vectorbt is a heavy optional dep (only needed when this
    # opt-in analysis actually runs), same lazy-import convention as
    # backtest._run_simulation_loop.
    from vectorbt_bridge import orders_to_portfolio
    from vectorbt_bridge import portfolio_stats as compute_portfolio_stats

    orders = build_orders_for_weights(
        cycles, weights, price_matrix,
        picks_per_cycle=picks_per_cycle, hold_days=hold_days, init_cash=init_cash,
    )
    norm = _normalize_weights(weights)
    if not orders:
        return {"status": "no_orders", "weights": norm, "n_orders": 0}

    pf = orders_to_portfolio(orders, price_matrix, init_cash=init_cash, fees=fees)
    stats = compute_portfolio_stats(pf, spy_prices=spy_prices)

    return {
        "status": "ok",
        "weights": norm,
        "n_orders": len(orders),
        "total_return": _safe_metric(stats.get("total_return")),
        "total_alpha": _safe_metric(stats.get("total_alpha")),
        "sortino_ratio": _safe_metric(stats.get("sortino_ratio")),
        "max_drawdown": _safe_metric(stats.get("max_drawdown")),
        "sharpe_ratio": _safe_metric(stats.get("sharpe_ratio")),
        "psr": _safe_metric(stats.get("psr")),
        "total_trades": int(stats.get("total_trades") or 0),
    }


def _safe_metric(v: Any) -> Optional[float]:
    """Coerce a metric to a plain float; None / NaN pass through as None so the
    report + downstream JSON round-trip stay clean."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # drop NaN


def _weight_label(weights: Mapping[str, float]) -> str:
    """Compact deterministic label for a weight tuple, e.g.
    ``mom=0.40,qual=0.30,val=0.20,lv=-0.10``."""
    norm = _normalize_weights(weights)
    short = {
        "momentum_score": "mom",
        "quality_score": "qual",
        "value_score": "val",
        "low_vol_score": "lv",
    }
    return ",".join(f"{short[c]}={norm[c]:g}" for c in FACTOR_SCORE_COLUMNS)


def build_counterfactual_replay_report(
    cycles: Sequence[Mapping[str, Any]],
    weight_variants: Iterable[Mapping[str, float]],
    price_matrix: pd.DataFrame,
    *,
    baseline_weights: Optional[Mapping[str, float]] = None,
    spy_prices: Optional[pd.Series] = None,
    picks_per_cycle: int = DEFAULT_PICKS_PER_CYCLE,
    hold_days: int = DEFAULT_HOLD_DAYS,
    init_cash: float = 1_000_000.0,
    fees: float = 0.001,
) -> dict[str, Any]:
    """Replay a grid of factor-blend weight tuples and assemble the report.

    The baseline (``baseline_weights``, default equal-weight) is always replayed
    first so each variant's ``sortino_lift`` / ``alpha_lift`` is expressed
    relative to it. Variants are reported sorted by ``sortino_ratio`` desc
    (Sortino is the gate the optimizer anchors on; see
    ``analysis.param_sweep._sort_sweep_df_skilled_risk``), with NaN/None last.

    Returns dict with:
      - ``status``    — "ok" / "skipped" (no cycles or empty price matrix) /
                        "no_data" (no variant produced orders)
      - ``baseline``  — the baseline variant's metrics dict
      - ``variants``  — list of per-weighting metrics dicts (incl. ``label`` +
                        ``sortino_lift`` / ``alpha_lift`` vs baseline), sorted
      - ``best``      — the top variant by Sortino lift (None if none beat NaN)
      - ``n_cycles`` / ``picks_per_cycle`` / ``hold_days``
    """
    if not cycles or price_matrix is None or len(price_matrix.index) == 0:
        return {"status": "skipped", "reason": "no cycles or empty price matrix"}

    base_w = _normalize_weights(baseline_weights or EQUAL_WEIGHTS)
    baseline = replay_weight_variant(
        cycles, base_w, price_matrix, spy_prices=spy_prices,
        picks_per_cycle=picks_per_cycle, hold_days=hold_days,
        init_cash=init_cash, fees=fees,
    )
    baseline["label"] = _weight_label(base_w)
    base_sortino = baseline.get("sortino_ratio")
    base_alpha = baseline.get("total_alpha")

    variants: list[dict[str, Any]] = []
    for w in weight_variants:
        res = replay_weight_variant(
            cycles, w, price_matrix, spy_prices=spy_prices,
            picks_per_cycle=picks_per_cycle, hold_days=hold_days,
            init_cash=init_cash, fees=fees,
        )
        res["label"] = _weight_label(w)
        res["sortino_lift"] = _lift(res.get("sortino_ratio"), base_sortino)
        res["alpha_lift"] = _lift(res.get("total_alpha"), base_alpha)
        variants.append(res)

    if not variants and baseline.get("status") != "ok":
        return {"status": "no_data", "reason": "no variant produced orders"}

    # Sort by Sortino desc, None/NaN last (mirrors the skilled-risk sort).
    variants.sort(
        key=lambda r: (r.get("sortino_ratio") is not None, r.get("sortino_ratio") or 0.0),
        reverse=True,
    )

    best = None
    for r in variants:
        if r.get("status") == "ok" and r.get("sortino_lift") is not None and r["sortino_lift"] > 0:
            best = r
            break

    return {
        "status": "ok",
        "n_cycles": len(cycles),
        "picks_per_cycle": picks_per_cycle,
        "hold_days": hold_days,
        "baseline": baseline,
        "variants": variants,
        "best": best,
        "any_variant_beats_baseline": best is not None,
    }


def _lift(value: Optional[float], baseline: Optional[float]) -> Optional[float]:
    """Variant minus baseline; None if either side is missing."""
    if value is None or baseline is None:
        return None
    return round(value - baseline, 6)
