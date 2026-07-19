"""double_sort.py — observe-only 2-horizon double-sort study (W3.3, config#1993).

Tests whether combining two prediction horizons via a **top-quantile
intersection** book beats either single horizon alone, after transaction cost.
For a horizon pair ``(h1, h2)`` and each rebalance date, the cross-section is
ranked independently on each horizon's predicted alpha; the top-quantile set of
each is taken; the **intersection** of the two is the double-sort book (long-
only, equal-weight). It is backtested against each single-horizon top-quantile
book as the baseline, through the same √-impact transaction-cost model the rest
of the backtester uses (``analysis.transaction_cost.TransactionCostModel``).

Operator-ratified scope (2026-07-10, config#1993 GO ruling; shares the
config#937 horizon ladder 10/21/42/63/126d and one config#624 BH-FDR family):
binding pair set ``10×21``, ``21×63``, ``21×126``; observe-only (ARCHITECTURE.md
§14(e)); CSCV/PBO (config#1995) applied to the **pair selection** step — picking
the winning pair post hoc is the data-mining trap this guards.

OBSERVE-only: this module GATES NOTHING and changes no serving path. It consumes
per-horizon leak-free OOS predicted-alpha panels (produced upstream by the
predictor's horizon battery — ``leakfree_horizon_ic_curve`` /
``_DIAGNOSTIC_HORIZONS``, the config#937 Phase-1 substrate) and reports, per
pair, the double-sort book's net alpha / net Sharpe / turnover vs the single-
horizon baselines, plus the CSCV-PBO of selecting the best pair. The per-horizon
gross-IC curve itself is the substrate's diagnostic, not recomputed here.

This module contains only the intersection / book-construction / pair-selection
logic; the per-period return, turnover, and √-impact cost accounting reuse the
exact conventions validated in ``analysis.horizon_net_alpha._one_horizon`` (from
which ``_ann_return`` / ``_SPY`` / the annualization constant are imported), so
the double-sort book and the single-horizon baselines are scored identically.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from analysis.horizon_net_alpha import _SPY, _TRADING_DAYS_PER_YEAR, _ann_return
from analysis.pbo import cscv_pbo
from analysis.portfolio_optimizer_backtest import _select_rebalance_dates
from analysis.transaction_cost import TransactionCostModel

logger = logging.getLogger(__name__)

_DEFAULT_TOP_QUANTILE = 0.20  # top quintile, per the operator-ratified plan
# Operator-ratified 2026-07-10 (config#1993), anchored on the config#937 ladder
# 10/21/42/63/126d: fast-adjacent×canonical, canonical×low-turnover-long,
# canonical×fundamentals-decay-long.
DEFAULT_PAIRS: list[tuple[int, int]] = [(10, 21), (21, 63), (21, 126)]
_DEFAULT_PBO_BLOCKS = 8  # date-aligned CSCV blocks for the pair-selection PBO
_PBO_MIN_BLOCKS = 4      # matches nousergon_lib cscv_pbo min_splits


def _top_quantile_book(
    predictions: dict, price_row: pd.Series, top_quantile: float
) -> list[str]:
    """Top-``top_quantile`` fraction of the cross-section by predicted alpha.

    Long-only, SPY excluded, restricted to names with a finite predicted alpha
    AND a finite price at the rebalance instant (same eligibility filter as
    ``horizon_net_alpha._book_at``). Returns names highest-alpha first; the
    quantile count is ``ceil(top_quantile * n_eligible)`` (≥1).
    """
    cand = [
        (t, a)
        for t, a in predictions.items()
        if t != _SPY
        and a is not None
        and np.isfinite(a)
        and t in price_row.index
        and np.isfinite(price_row.get(t, np.nan))
    ]
    if not cand:
        return []
    cand.sort(key=lambda kv: kv[1], reverse=True)
    k = max(1, math.ceil(top_quantile * len(cand)))
    return [t for t, _ in cand[:k]]


def _intersection_book(
    preds_h1: dict, preds_h2: dict, price_row: pd.Series, top_quantile: float
) -> list[str]:
    """Double-sort book = intersection of the two horizons' top-quantile sets.

    Order follows h1's alpha ranking (deterministic; the book is equal-weight so
    order affects nothing but reproducibility). Empty when the two top-quantiles
    do not overlap on this date — a legitimate outcome (no double-sorted names).
    """
    b1 = _top_quantile_book(preds_h1, price_row, top_quantile)
    s2 = set(_top_quantile_book(preds_h2, price_row, top_quantile))
    return [t for t in b1 if t in s2]


def _period_stats(
    book: list[str],
    prev_book: set[str],
    p0: pd.Series,
    p1: pd.Series,
    cost_model: TransactionCostModel,
    init_cash: float,
    adv_dollar_by_ticker: dict | None,
) -> tuple[float, float, float, set[str]]:
    """One holding period's (gross_return, one-way_turnover, cost_fraction,
    new_book). Equal-weight 1/N long book; identical accounting to
    ``horizon_net_alpha._one_horizon`` (target-to-target turnover, per-name
    √-impact cost on entering/leaving legs, renormalized realized return over
    names priced at both ends)."""
    w = 1.0 / len(book)
    rets = [
        float(p1[t] / p0[t] - 1.0)
        for t in book
        if t in p1.index and np.isfinite(p1.get(t, np.nan)) and p0[t] > 0
    ]
    gross = float(np.mean(rets)) if rets else 0.0

    new_book = set(book)
    added = new_book - prev_book
    dropped = prev_book - new_book
    oneway = (len(added) + len(dropped)) * w
    cost_fraction = 0.0
    for t in added | dropped:
        adv = (adv_dollar_by_ticker or {}).get(t)
        cost_fraction += cost_model.cost_for_turnover(w * init_cash, adv) / init_cash
    return gross, oneway, cost_fraction, new_book


def _backtest_books(
    book_by_rebal: dict[str, list[str]],
    rebal: list[str],
    price_matrix: pd.DataFrame,
    spy_prices: pd.Series,
    cost_model: TransactionCostModel,
    horizon: int,
    init_cash: float,
    adv_dollar_by_ticker: dict | None,
) -> dict:
    """Walk-forward backtest of a pre-selected book sequence.

    ``book_by_rebal`` maps each rebalance date (in ``rebal``) to that period's
    long book; the book is held to the next rebalance. Returns the same summary
    shape as ``horizon_net_alpha._one_horizon`` plus ``net_periods`` /
    ``period_start_dates`` (consumed by the pair-selection PBO). A period whose
    book is empty flattens to cash (no position, no cost) for that leg.
    """
    if len(rebal) < 2:
        return {"status": "insufficient_rebalances", "n_rebalances": len(rebal),
                "forward_days": horizon}
    rebal_ts = [pd.Timestamp(d) for d in rebal]
    spy = spy_prices.reindex(price_matrix.index)

    prev_book: set[str] = set()
    gross_periods: list[float] = []
    net_periods: list[float] = []
    spy_periods: list[float] = []
    period_starts: list[str] = []
    total_oneway = 0.0
    total_cost = 0.0

    for j in range(len(rebal_ts) - 1):
        t0, t1 = rebal_ts[j], rebal_ts[j + 1]
        book = book_by_rebal.get(rebal[j], [])
        if not book:
            prev_book = set()
            continue
        p0, p1 = price_matrix.loc[t0], price_matrix.loc[t1]
        gross, oneway, cost, new_book = _period_stats(
            book, prev_book, p0, p1, cost_model, init_cash, adv_dollar_by_ticker
        )
        spy_ret = (
            float(spy.loc[t1] / spy.loc[t0] - 1.0)
            if np.isfinite(spy.get(t0, np.nan))
            and np.isfinite(spy.get(t1, np.nan))
            and spy.loc[t0] > 0
            else 0.0
        )
        gross_periods.append(gross)
        net_periods.append(gross - cost)
        spy_periods.append(spy_ret)
        period_starts.append(rebal[j])
        total_oneway += oneway
        total_cost += cost
        prev_book = new_book

    if len(net_periods) < 2:
        return {"status": "insufficient_periods", "n_rebalances": len(rebal),
                "forward_days": horizon}

    n_days = int(price_matrix.index.get_loc(rebal_ts[len(net_periods)])
                 - price_matrix.index.get_loc(rebal_ts[0]))
    years = max(n_days / _TRADING_DAYS_PER_YEAR, 1e-9)
    gross_total = float(np.prod([1.0 + r for r in gross_periods]) - 1.0)
    net_total = float(np.prod([1.0 + r for r in net_periods]) - 1.0)
    spy_total = float(np.prod([1.0 + r for r in spy_periods]) - 1.0)

    net_arr = np.asarray(net_periods, dtype=float)
    periods_per_year = _TRADING_DAYS_PER_YEAR / horizon
    net_sharpe = (
        float(net_arr.mean() / net_arr.std(ddof=1) * np.sqrt(periods_per_year))
        if net_arr.std(ddof=1) > 1e-12 else None
    )
    return {
        "status": "ok",
        "forward_days": horizon,
        "n_rebalances": len(net_periods),
        "gross_alpha_ann": round(_ann_return(gross_total, n_days)
                                 - _ann_return(spy_total, n_days), 6),
        "net_alpha_ann": round(_ann_return(net_total, n_days)
                               - _ann_return(spy_total, n_days), 6),
        "turnover_oneway_ann": round(total_oneway / years, 6),
        "cost_drag_bps_ann": round(total_cost / years * 1e4, 4),
        "net_sharpe": round(net_sharpe, 4) if net_sharpe is not None else None,
        "net_periods": net_periods,
        "period_start_dates": period_starts,
    }


def _block_sharpes(
    net_periods: list[float],
    period_starts: list[str],
    block_edges: list[pd.Timestamp],
    horizon: int,
) -> list[float]:
    """Per-block annualized net Sharpe, blocks defined by shared date edges.

    ``block_edges`` are ``n_blocks + 1`` ascending timestamps; a period is
    assigned to the block whose ``[edge_i, edge_{i+1})`` contains its start
    date (last block right-inclusive). A block with <2 periods or ~zero variance
    yields ``nan`` (an honest hole — the PBO layer drops non-finite rows rather
    than fabricating a value).
    """
    starts = [pd.Timestamp(d) for d in period_starts]
    ppy = _TRADING_DAYS_PER_YEAR / horizon
    out: list[float] = []
    n_blocks = len(block_edges) - 1
    for i in range(n_blocks):
        lo, hi = block_edges[i], block_edges[i + 1]
        if i == n_blocks - 1:
            vals = [r for r, s in zip(net_periods, starts) if lo <= s <= hi]
        else:
            vals = [r for r, s in zip(net_periods, starts) if lo <= s < hi]
        arr = np.asarray(vals, dtype=float)
        if arr.size < 2 or arr.std(ddof=1) <= 1e-12:
            out.append(float("nan"))
        else:
            out.append(float(arr.mean() / arr.std(ddof=1) * np.sqrt(ppy)))
    return out


def _selection_pbo(
    per_pair: dict[str, dict], n_blocks: int
) -> dict:
    """CSCV-PBO over pair selection: trials = pairs, rows = date-aligned blocks,
    cell = the pair's double-sort net Sharpe in that block.

    Aligns blocks by a shared date span (the union of all pairs' period start
    dates) so rows are comparable across pairs even though each pair rebalances
    at its own cadence. Delegates the leave-one-block-out overfitting test to
    the shared ``nousergon_lib`` engine.
    """
    usable = {
        pid: e for pid, e in per_pair.items()
        if e.get("status") == "ok" and e.get("net_periods")
    }
    if len(usable) < 2:
        return {"status": "insufficient", "reason": "need ≥2 scored pairs"}
    all_starts = [
        pd.Timestamp(d) for e in usable.values() for d in e["period_start_dates"]
    ]
    lo, hi = min(all_starts), max(all_starts)
    if lo == hi:
        return {"status": "insufficient", "reason": "degenerate date span"}
    edges = list(pd.date_range(lo, hi, periods=n_blocks + 1))
    spec_ids = sorted(usable)
    columns = [
        _block_sharpes(usable[pid]["net_periods"],
                       usable[pid]["period_start_dates"], edges,
                       int(usable[pid]["forward_days"]))
        for pid in spec_ids
    ]
    ic_matrix = [list(row) for row in zip(*columns)]  # (n_blocks, n_pairs)
    return cscv_pbo(ic_matrix, spec_ids=spec_ids, min_splits=_PBO_MIN_BLOCKS)


def compute_double_sort(
    predictions_by_horizon: dict,
    price_matrix: pd.DataFrame,
    spy_prices: pd.Series,
    *,
    pairs: list[tuple[int, int]] | None = None,
    cost_model: TransactionCostModel | None = None,
    top_quantile: float = _DEFAULT_TOP_QUANTILE,
    init_cash: float = 1_000_000.0,
    adv_dollar_by_ticker: dict | None = None,
    pbo_blocks: int = _DEFAULT_PBO_BLOCKS,
) -> dict:
    """Observe-only 2-horizon double-sort study (config#1993, W3.3).

    ``predictions_by_horizon`` maps each horizon (int trading days) to that
    horizon's ``{date: {ticker: leak-free OOS predicted alpha}}`` panel — the
    upstream per-horizon substrate (config#937 Phase 1). For each pair
    ``(h1, h2)`` the double-sort book is rebalanced at ``min(h1, h2)`` (as fast
    as the faster signal), formed from the top-quantile intersection, and scored
    against each single-horizon top-quantile baseline (rebalanced at its own
    horizon). CSCV/PBO is computed over the pair-selection step (config#1995).

    Returns ``{status, pairs: {"h1xh2": {double_sort, baseline_h1, baseline_h2,
    net_alpha_uplift_vs_best_baseline, ...}}, selection_pbo, ...}``. GATES
    NOTHING; observe-only per ARCHITECTURE.md §14(e).
    """
    pairs = pairs or DEFAULT_PAIRS
    cost_model = cost_model or TransactionCostModel()

    def _baseline(h: int) -> dict:
        panel = predictions_by_horizon.get(h)
        if not panel:
            return {"status": "missing_horizon_panel", "forward_days": h}
        dates = sorted(panel)
        rebal = _select_rebalance_dates(dates, price_matrix.index, h)
        books = {
            d: _top_quantile_book(panel[d], price_matrix.loc[pd.Timestamp(d)],
                                  top_quantile)
            for d in rebal
        }
        return _backtest_books(books, rebal, price_matrix, spy_prices,
                               cost_model, h, init_cash, adv_dollar_by_ticker)

    baseline_cache: dict[int, dict] = {}
    per_pair: dict[str, dict] = {}
    for h1, h2 in pairs:
        pid = f"{h1}x{h2}"
        panel1 = predictions_by_horizon.get(h1)
        panel2 = predictions_by_horizon.get(h2)
        if not panel1 or not panel2:
            per_pair[pid] = {"status": "missing_horizon_panel",
                             "missing": [h for h, p in ((h1, panel1), (h2, panel2))
                                         if not p]}
            continue
        cadence = min(h1, h2)
        common_dates = sorted(set(panel1) & set(panel2))
        rebal = _select_rebalance_dates(common_dates, price_matrix.index, cadence)
        books = {
            d: _intersection_book(panel1[d], panel2[d],
                                  price_matrix.loc[pd.Timestamp(d)], top_quantile)
            for d in rebal
        }
        ds = _backtest_books(books, rebal, price_matrix, spy_prices,
                             cost_model, cadence, init_cash, adv_dollar_by_ticker)
        b1 = baseline_cache.setdefault(h1, _baseline(h1))
        b2 = baseline_cache.setdefault(h2, _baseline(h2))
        entry = {
            "status": ds.get("status"),
            "forward_days_cadence": cadence,
            "double_sort": ds,
            "baseline_h1": b1,
            "baseline_h2": b2,
        }
        ds_na = ds.get("net_alpha_ann")
        base_nas = [b.get("net_alpha_ann") for b in (b1, b2)
                    if isinstance(b.get("net_alpha_ann"), (int, float))]
        if isinstance(ds_na, (int, float)) and base_nas:
            entry["net_alpha_uplift_vs_best_baseline"] = round(
                ds_na - max(base_nas), 6)
        # strip the heavy per-period arrays from the double_sort payload the
        # caller serializes; the selection PBO already consumed them in-process.
        per_pair[pid] = entry

    selection_pbo = _selection_pbo(
        {pid: e.get("double_sort", {}) for pid, e in per_pair.items()},
        pbo_blocks,
    )
    for e in per_pair.values():
        e.get("double_sort", {}).pop("net_periods", None)
        e.get("double_sort", {}).pop("period_start_dates", None)

    finite = {
        pid: e["net_alpha_uplift_vs_best_baseline"]
        for pid, e in per_pair.items()
        if isinstance(e.get("net_alpha_uplift_vs_best_baseline"), (int, float))
        and np.isfinite(e["net_alpha_uplift_vs_best_baseline"])
    }
    best_pair = max(finite, key=finite.get) if finite else None
    logger.info(
        "double_sort (OBSERVE, NOT gated): pairs=%s best-uplift-pair=%s "
        "selection_pbo=%s. Pre-registered as ONE config#624 BH-FDR family; "
        "PBO on pair SELECTION guards the post-hoc pick.",
        list(per_pair), best_pair, selection_pbo.get("pbo", selection_pbo.get("status")),
    )
    return {
        "status": "ok",
        "pairs": per_pair,
        "best_uplift_pair": best_pair,
        "selection_pbo": selection_pbo,
        "top_quantile": top_quantile,
        "cost_model": {
            "half_spread_bps": cost_model.half_spread_bps,
            "impact_coef_bps": cost_model.impact_coef_bps,
            "commission_bps": cost_model.commission_bps,
        },
        "note": "OBSERVE-only (config#1993, ARCHITECTURE.md §14(e)); gates "
                "nothing. Cross-horizon double-sorts are the decay-prone class "
                "(McLean–Pontiff); the CSCV/PBO on pair selection and the "
                "config#624 BH-FDR family charge are the overfit guards.",
    }
