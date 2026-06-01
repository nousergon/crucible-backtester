"""horizon_net_alpha.py — turnover-adjusted NET alpha per horizon (W3.4, L4469).

The predictor emits a leak-free GROSS per-horizon IC curve
(``horizon_diagnostic.curve_leakfree`` in its manifest), but gross IC alone
does NOT settle "should we target 21d or 60/90d?": longer horizons rebalance
less often (turnover ~1/h), so they can win NET even at similar gross IC. This
module is the **net-of-cost judge** — for each candidate horizon it forms the
production-style book, rebalances at that cadence, and subtracts realized
transaction cost (``analysis.transaction_cost.TransactionCostModel``, a
√-impact + half-spread + commission model) to report NET alpha.

Construction (matches how the executor actually trades — long-only, top-N
equal-weight — not an academic decile long-short):
  For each horizon h ∈ config:
    * rebalance every h trading days (``_select_rebalance_dates``);
    * at each rebalance, rank the cross-section by the predictor's continuous
      alpha forecast and take the top-N equal-weight (1/N) long book;
    * hold to the next rebalance, accumulate the realized book return vs SPY;
    * turnover per rebalance = Σ|Δweight| (target-to-target, the repo
      convention in portfolio_optimizer_backtest); cost via the model
      (per-name ADV → participation → √-impact); NET = gross − cost drag.

OBSERVE-only: emitted to ``backtest/{date}/horizon_net_alpha.json``; gates
nothing and does NOT change the canonical 21d target. A horizon cutover is a
separate later decision informed by this NET read together with the
predictor's gross-IC curve. Net-of-cost is the judge gross IC is not.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from analysis.portfolio_optimizer_backtest import _select_rebalance_dates
from analysis.transaction_cost import TransactionCostModel

logger = logging.getLogger(__name__)

_SPY = "SPY"
_TRADING_DAYS_PER_YEAR = 252
_DEFAULT_HORIZONS = [5, 10, 21, 60, 90]
_DEFAULT_TOP_N = 20


def _ann_return(total_return: float, n_days: int) -> float:
    """Annualize a cumulative simple return over ``n_days`` trading days."""
    if n_days <= 0:
        return float("nan")
    base = 1.0 + total_return
    if base <= 0:  # total wipeout guard — annualization undefined
        return -1.0
    return float(base ** (_TRADING_DAYS_PER_YEAR / n_days) - 1.0)


def _book_at(predictions: dict, price_row: pd.Series, top_n: int) -> list[str]:
    """Top-N tickers by predicted alpha that also have a finite price now
    (long-only, SPY excluded)."""
    cand = [
        (t, a) for t, a in predictions.items()
        if t != _SPY and a is not None and np.isfinite(a)
        and t in price_row.index and np.isfinite(price_row.get(t, np.nan))
    ]
    cand.sort(key=lambda kv: kv[1], reverse=True)
    return [t for t, _ in cand[:top_n]]


def _one_horizon(
    horizon: int,
    predictions_by_date: dict,
    price_matrix: pd.DataFrame,
    spy_prices: pd.Series,
    cost_model: TransactionCostModel,
    top_n: int,
    init_cash: float,
    adv_dollar_by_ticker: dict | None,
) -> dict:
    pred_dates = sorted(predictions_by_date.keys())
    rebal = _select_rebalance_dates(pred_dates, price_matrix.index, horizon)
    if len(rebal) < 2:
        return {"status": "insufficient_rebalances", "n_rebalances": len(rebal),
                "forward_days": horizon}

    rebal_ts = [pd.Timestamp(d) for d in rebal]
    spy = spy_prices.reindex(price_matrix.index)

    prev_book: set[str] = set()
    gross_periods: list[float] = []
    net_periods: list[float] = []
    spy_periods: list[float] = []
    total_oneway_turnover = 0.0
    total_cost_fraction = 0.0

    for j in range(len(rebal_ts) - 1):
        t0, t1 = rebal_ts[j], rebal_ts[j + 1]
        p0 = price_matrix.loc[t0]
        p1 = price_matrix.loc[t1]
        book = _book_at(predictions_by_date[rebal[j]], p0, top_n)
        if not book:
            prev_book = set()
            continue
        w = 1.0 / len(book)

        # Period gross return: equal-weight realized return of names with a
        # finite price at both ends (renormalize over the valid subset).
        rets = [
            float(p1[t] / p0[t] - 1.0)
            for t in book
            if t in p1.index and np.isfinite(p1.get(t, np.nan)) and p0[t] > 0
        ]
        gross = float(np.mean(rets)) if rets else 0.0

        # Turnover (target-to-target, 1/N equal weight): added + dropped names.
        new_book = set(book)
        added = new_book - prev_book
        dropped = prev_book - new_book
        # Per-name |Δw|: entering/leaving names move 1/N; held names net ~0 when
        # both books are equal-weight 1/N (book size constant by construction).
        oneway = (len(added) + len(dropped)) * w
        # Cost: charge each per-name leg via the √-impact model at this rebalance.
        cost_fraction = 0.0
        for t in (added | dropped):
            adv = (adv_dollar_by_ticker or {}).get(t)
            cost_fraction += cost_model.cost_for_turnover(w * init_cash, adv) / init_cash

        net = gross - cost_fraction
        days = max(1, int(price_matrix.index.get_loc(t1) - price_matrix.index.get_loc(t0)))
        spy_ret = (
            float(spy.loc[t1] / spy.loc[t0] - 1.0)
            if np.isfinite(spy.get(t0, np.nan)) and np.isfinite(spy.get(t1, np.nan)) and spy.loc[t0] > 0
            else 0.0
        )
        gross_periods.append(gross)
        net_periods.append(net)
        spy_periods.append(spy_ret)
        total_oneway_turnover += oneway
        total_cost_fraction += cost_fraction
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
        "top_n": top_n,
        "gross_alpha_ann": round(_ann_return(gross_total, n_days) - _ann_return(spy_total, n_days), 6),
        "net_alpha_ann": round(_ann_return(net_total, n_days) - _ann_return(spy_total, n_days), 6),
        "gross_return_ann": round(_ann_return(gross_total, n_days), 6),
        "net_return_ann": round(_ann_return(net_total, n_days), 6),
        "spy_return_ann": round(_ann_return(spy_total, n_days), 6),
        "turnover_oneway_ann": round(total_oneway_turnover / years, 6),
        "cost_drag_bps_ann": round(total_cost_fraction / years * 1e4, 4),
        "net_sharpe": round(net_sharpe, 4) if net_sharpe is not None else None,
    }


def compute_horizon_net_alpha(
    predictions_by_date: dict,
    price_matrix: pd.DataFrame,
    spy_prices: pd.Series,
    *,
    cost_model: TransactionCostModel | None = None,
    horizons: list[int] | None = None,
    top_n: int = _DEFAULT_TOP_N,
    init_cash: float = 1_000_000.0,
    adv_dollar_by_ticker: dict | None = None,
) -> dict:
    """Per-horizon turnover-adjusted net-alpha study (W3.4, OBSERVE).

    Parameters mirror ``portfolio_optimizer_backtest.run_optimizer_backtest``:
    ``predictions_by_date`` ({date: {ticker: alpha}}), ``price_matrix`` (date ×
    ticker close), ``spy_prices`` (date-indexed). ``adv_dollar_by_ticker`` is an
    optional {ticker: avg-daily-dollar-volume} map; absent names degrade the
    cost model to half-spread + commission (impact term drops). Returns a dict
    with per-horizon NET/gross alpha, turnover, cost drag, and the
    net-alpha-maximizing horizon.
    """
    cost_model = cost_model or TransactionCostModel()
    horizons = horizons or _DEFAULT_HORIZONS
    n_adv = len(adv_dollar_by_ticker or {})
    logger.info(
        "horizon_net_alpha: horizons=%s top_n=%d cost=(spread=%.2f impact=%.2f "
        "comm=%.2f)bps ADV-coverage=%d names%s",
        horizons, top_n, cost_model.half_spread_bps, cost_model.impact_coef_bps,
        cost_model.commission_bps, n_adv,
        "" if n_adv else " (√-impact term inactive — half-spread+commission only)",
    )

    per_horizon: dict[str, dict] = {}
    for h in horizons:
        try:
            per_horizon[f"{int(h)}d"] = _one_horizon(
                int(h), predictions_by_date, price_matrix, spy_prices,
                cost_model, top_n, init_cash, adv_dollar_by_ticker,
            )
        except Exception as exc:  # observe-only — never fail the backtest run
            logger.warning("horizon_net_alpha: horizon %sd failed (non-fatal): %s", h, exc)
            per_horizon[f"{int(h)}d"] = {"status": "error", "error": str(exc),
                                         "forward_days": int(h)}

    finite = {
        k: v["net_alpha_ann"] for k, v in per_horizon.items()
        if isinstance(v.get("net_alpha_ann"), (int, float)) and np.isfinite(v["net_alpha_ann"])
    }
    net_peak = max(finite, key=finite.get) if finite else None
    if net_peak:
        logger.info(
            "horizon_net_alpha (OBSERVE, NOT gated): net-alpha by horizon=%s | "
            "net-alpha peak=%s. NET (not gross IC) settles a horizon cutover; "
            "compare against the predictor manifest's gross-IC curve.",
            {k: round(float(v), 4) for k, v in finite.items()}, net_peak,
        )

    return {
        "status": "ok",
        "horizons": per_horizon,
        "net_alpha_max_horizon": net_peak,
        "top_n": top_n,
        "cost_model": {
            "half_spread_bps": cost_model.half_spread_bps,
            "impact_coef_bps": cost_model.impact_coef_bps,
            "commission_bps": cost_model.commission_bps,
            "min_cost_bps": cost_model.min_cost_bps,
            "adv_coverage_names": n_adv,
        },
        "note": "OBSERVE-only; gates nothing. NET-of-cost is the horizon-cutover "
                "judge — gross IC (predictor manifest curve_leakfree) is not.",
    }
