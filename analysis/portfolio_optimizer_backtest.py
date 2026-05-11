"""
Portfolio-optimizer backtest — PR 3 of portfolio-optimizer-260511 arc.

Replays the constrained MVO optimizer (alpha-engine PR #157,
executor/portfolio_optimizer.py) over historical synthetic predictions from
synthetic/predictor_backtest.py, then simulates the resulting target-weight
trajectory through vectorbt. Produces the metric set required by the
cutover gate validator (PR 4): Sharpe, alpha vs SPY, max DD, turnover,
tracking error vs SPY, active share, mean SPY weight, mean cash weight.

Side-by-side comparison with the legacy 1/n bottom-up backtest's metrics is
the substantive cutover decision input.

Design notes:
  - Rebalance frequency defaults to weekly (5 trading days) to match the
    Saturday research cadence and to bound per-rebalance turnover.
  - Universe per rebalance is capped at top-N by |alpha_hat| to keep the
    cvxpy solve time bounded (real production universe is ~25-60 names).
  - SPY is added as the benchmark fill ticker (alpha_hat = 0). Its price
    series is sourced from the predictor_backtest `spy_prices` output.
  - CASH is included in the optimizer universe (pinned to 3% by the kernel)
    but excluded from the vectorbt price matrix; vbt handles cash natively
    via the unallocated portion of the size vector.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252
_DEFAULT_REBALANCE_FREQ = 5
_DEFAULT_UNIVERSE_CAP = 30
_DEFAULT_RETURNS_LOOKBACK = 252
_DEFAULT_MIN_RETURNS = 60
_SPY = "SPY"
_CASH = "CASH"
_BENCH_SECTOR = "__benchmark__"
_CASH_SECTOR = "__cash__"
_CASH_ALPHA_HINT = -1e-6


@dataclass
class OptimizerBacktestResult:
    target_weights: pd.DataFrame
    metrics: dict
    rebalance_dates: list[str]
    n_rebalances: int
    n_solver_failures: int = 0
    diagnostics_per_rebalance: list[dict] = field(default_factory=list)


def run_optimizer_backtest(
    predictions_by_date: dict[str, dict[str, float]],
    price_matrix: pd.DataFrame,
    spy_prices: pd.Series,
    sector_map: dict[str, str],
    executor_path: str,
    rebalance_freq_days: int = _DEFAULT_REBALANCE_FREQ,
    universe_cap: int = _DEFAULT_UNIVERSE_CAP,
    init_cash: float = 1_000_000.0,
    fees: float = 0.001,
    optimizer_cfg: dict | None = None,
    max_position_pct: float = 0.08,
    min_score_proxy: float | None = None,
) -> OptimizerBacktestResult:
    """
    Backtest the constrained MVO optimizer over historical synthetic predictions.

    Args:
        predictions_by_date: {date_str: {ticker: predicted_alpha}} from
            synthetic.predictor_backtest.run() (with keep_features=True or
            equivalent exposure).
        price_matrix: DataFrame, DatetimeIndex (trading dates) × columns (tickers).
        spy_prices: Series, DatetimeIndex (same as price_matrix), SPY close prices.
        sector_map: {ticker: sector_etf_or_sector_name} — used to assign sectors
            to the optimizer's sector caps. SPY and CASH get sentinel sectors.
        executor_path: filesystem path to alpha-engine repo root. Used to
            sys.path-insert so we can import executor.portfolio_optimizer.
        rebalance_freq_days: trading-day stride between rebalances (default 5
            = weekly Saturday-research cadence).
        universe_cap: max number of tickers (excluding SPY/CASH) per rebalance,
            ranked by |predicted_alpha|. Bounds cvxpy solve time.
        init_cash: starting portfolio NAV for vectorbt simulation.
        fees: per-trade fee fraction (round-trip). Same default as
            vectorbt_bridge.orders_to_portfolio.
        optimizer_cfg: passthrough to solve_target_weights via cfg kwarg.
        max_position_pct: per-name cap (uniform — synthetic predictions don't
            carry stance, so all caps default to this value).
        min_score_proxy: optional |alpha_hat| threshold below which a ticker
            is marked ineligible. None disables (all alpha-bearing tickers
            eligible).

    Returns:
        OptimizerBacktestResult with target_weights DataFrame (date × ticker),
        metrics dict, and per-rebalance diagnostics.
    """
    if executor_path not in sys.path:
        sys.path.insert(0, executor_path)
    try:
        from executor.portfolio_optimizer import (  # type: ignore[import-not-found]
            solve_target_weights,
            OPTIMIZER_CONFIG_DEFAULTS,
        )
    except ImportError as e:
        raise ImportError(
            f"Could not import executor.portfolio_optimizer from {executor_path}. "
            "Ensure executor_paths in config.yaml points to the alpha-engine repo "
            "root and that cvxpy + scikit-learn are installed."
        ) from e

    if optimizer_cfg is None:
        optimizer_cfg = {}
    effective_cfg = {**OPTIMIZER_CONFIG_DEFAULTS, **optimizer_cfg}

    price_matrix = _ensure_spy_column(price_matrix, spy_prices)
    rebalance_dates = _select_rebalance_dates(
        sorted(predictions_by_date.keys()), price_matrix.index, rebalance_freq_days,
    )
    if not rebalance_dates:
        raise ValueError(
            "No rebalance dates resolved. Check predictions_by_date and price_matrix "
            "have overlapping date coverage."
        )

    logger.info(
        f"Optimizer backtest: {len(rebalance_dates)} rebalances "
        f"({rebalance_dates[0]} → {rebalance_dates[-1]}), "
        f"universe_cap={universe_cap}, freq={rebalance_freq_days}d"
    )

    target_weights = pd.DataFrame(
        np.nan, index=price_matrix.index, columns=price_matrix.columns,
    )
    diagnostics: list[dict] = []
    n_solver_failures = 0

    for rebal_date in rebalance_dates:
        try:
            kwargs = _build_optimizer_inputs(
                rebal_date=rebal_date,
                predictions=predictions_by_date.get(rebal_date, {}),
                price_matrix=price_matrix,
                sector_map=sector_map,
                universe_cap=universe_cap,
                max_position_pct=max_position_pct,
                min_score_proxy=min_score_proxy,
                cfg=effective_cfg,
            )
        except _InsufficientHistoryError as e:
            logger.debug(f"{rebal_date}: skipped — {e}")
            diagnostics.append({"date": rebal_date, "status": "skipped_insufficient_history"})
            continue

        result = solve_target_weights(**kwargs)
        if result.diagnostics["status"] == "infeasible_fallback":
            n_solver_failures += 1
        diagnostics.append({
            "date": rebal_date,
            "status": result.diagnostics["status"],
            "n_active": result.diagnostics["n_active_positions"],
            "expected_alpha": result.diagnostics["expected_alpha"],
        })

        _populate_target_weights_row(
            target_weights, rebal_date, kwargs["tickers"], result.weights,
        )

    metrics = _simulate_and_measure(
        target_weights=target_weights,
        price_matrix=price_matrix,
        spy_prices=spy_prices,
        init_cash=init_cash,
        fees=fees,
    )
    metrics["n_rebalances"] = len(rebalance_dates)
    metrics["n_solver_failures"] = n_solver_failures
    metrics["rebalance_freq_days"] = rebalance_freq_days
    metrics["universe_cap"] = universe_cap

    return OptimizerBacktestResult(
        target_weights=target_weights,
        metrics=metrics,
        rebalance_dates=rebalance_dates,
        n_rebalances=len(rebalance_dates),
        n_solver_failures=n_solver_failures,
        diagnostics_per_rebalance=diagnostics,
    )


class _InsufficientHistoryError(Exception):
    pass


def _ensure_spy_column(price_matrix: pd.DataFrame, spy_prices: pd.Series) -> pd.DataFrame:
    if _SPY in price_matrix.columns:
        return price_matrix
    pm = price_matrix.copy()
    pm[_SPY] = spy_prices.reindex(pm.index)
    return pm


def _select_rebalance_dates(
    prediction_dates: list[str],
    price_index: pd.DatetimeIndex,
    freq_days: int,
) -> list[str]:
    valid = [d for d in prediction_dates if pd.Timestamp(d) in price_index]
    if not valid:
        return []
    rebalance_dates: list[str] = []
    last_idx = -freq_days
    for i, d in enumerate(valid):
        if i - last_idx >= freq_days:
            rebalance_dates.append(d)
            last_idx = i
    return rebalance_dates


def _build_optimizer_inputs(
    rebal_date: str,
    predictions: dict[str, float],
    price_matrix: pd.DataFrame,
    sector_map: dict[str, str],
    universe_cap: int,
    max_position_pct: float,
    min_score_proxy: float | None,
    cfg: dict,
) -> dict:
    rebal_ts = pd.Timestamp(rebal_date)
    if rebal_ts not in price_matrix.index:
        raise _InsufficientHistoryError(f"{rebal_date} not in price_matrix index")

    pos = price_matrix.index.get_loc(rebal_ts)
    lookback_start = max(0, pos - _DEFAULT_RETURNS_LOOKBACK)
    history = price_matrix.iloc[lookback_start:pos]
    if len(history) < _DEFAULT_MIN_RETURNS:
        raise _InsufficientHistoryError(
            f"Only {len(history)} prior trading days at {rebal_date}, need ≥{_DEFAULT_MIN_RETURNS}"
        )

    tickers_with_alpha = [t for t, a in predictions.items() if t in history.columns and pd.notna(a)]
    if not tickers_with_alpha:
        raise _InsufficientHistoryError("No tickers with usable predictions at this date")

    ranked = sorted(tickers_with_alpha, key=lambda t: abs(predictions[t]), reverse=True)
    selected = ranked[:universe_cap]

    universe = list(selected) + [_SPY, _CASH]
    spy_idx = universe.index(_SPY)
    cash_idx = universe.index(_CASH)
    N = len(universe)

    history_subset = history[selected + [_SPY]].dropna(axis=0, how="any")
    if len(history_subset) < _DEFAULT_MIN_RETURNS:
        raise _InsufficientHistoryError(
            f"Aligned returns have only {len(history_subset)} rows after dropna"
        )

    returns = history_subset.pct_change().dropna().values
    if returns.shape[0] < _DEFAULT_MIN_RETURNS:
        raise _InsufficientHistoryError(
            f"Returns panel has only {returns.shape[0]} rows after pct_change.dropna"
        )

    returns_panel = np.zeros((returns.shape[0], N))
    for i, t in enumerate(selected):
        col_idx = list(history_subset.columns).index(t)
        returns_panel[:, i] = returns[:, col_idx]
    spy_col_idx = list(history_subset.columns).index(_SPY)
    returns_panel[:, spy_idx] = returns[:, spy_col_idx]
    returns_panel[:, cash_idx] = 0.0

    alpha_hat = np.zeros(N)
    for i, t in enumerate(selected):
        alpha_hat[i] = float(predictions[t])
    alpha_hat[spy_idx] = 0.0
    alpha_hat[cash_idx] = _CASH_ALPHA_HINT

    sectors: list[str] = []
    for t in selected:
        sectors.append(str(sector_map.get(t, "Unknown")))
    sectors.append(_BENCH_SECTOR)
    sectors.append(_CASH_SECTOR)

    stance_caps = np.full(N, max_position_pct)
    stance_caps[spy_idx] = 1.0
    stance_caps[cash_idx] = 1.0

    eligibility = np.ones(N, dtype=bool)
    if min_score_proxy is not None:
        for i, t in enumerate(selected):
            if abs(predictions[t]) < min_score_proxy:
                eligibility[i] = False

    w_prev = np.zeros(N)
    w_prev[cash_idx] = 1.0

    return {
        "tickers": universe,
        "alpha_hat": alpha_hat,
        "returns_panel": returns_panel,
        "w_prev": w_prev,
        "sectors": sectors,
        "stance_caps": stance_caps,
        "eligibility": eligibility,
        "spy_idx": spy_idx,
        "cash_idx": cash_idx,
        "cfg": cfg,
    }


def _populate_target_weights_row(
    target_weights: pd.DataFrame,
    rebal_date: str,
    universe: list[str],
    weights: np.ndarray,
) -> None:
    rebal_ts = pd.Timestamp(rebal_date)
    for i, t in enumerate(universe):
        if t == _CASH:
            continue
        if t in target_weights.columns:
            target_weights.at[rebal_ts, t] = float(weights[i])


def _simulate_and_measure(
    target_weights: pd.DataFrame,
    price_matrix: pd.DataFrame,
    spy_prices: pd.Series,
    init_cash: float,
    fees: float,
) -> dict:
    import vectorbt as vbt

    aligned_prices = price_matrix.reindex(target_weights.index)
    size = target_weights.copy()
    size[size.isna()] = np.nan

    pf = vbt.Portfolio.from_orders(
        close=aligned_prices,
        size=size,
        size_type="targetpercent",
        init_cash=init_cash,
        cash_sharing=True,
        group_by=True,
        fees=fees,
        freq="D",
    )

    total_return = float(pf.total_return())
    sharpe = float(pf.sharpe_ratio())
    max_drawdown = float(pf.max_drawdown())
    calmar = float(pf.calmar_ratio())

    daily_returns = pd.Series(pf.returns(), name="daily_return").astype(np.float64).dropna()

    spy_aligned = spy_prices.reindex(target_weights.index).dropna()
    spy_daily = spy_aligned.pct_change().dropna()
    spy_aligned_to_pf = spy_daily.reindex(daily_returns.index).dropna()
    pf_aligned_to_spy = daily_returns.reindex(spy_aligned_to_pf.index)

    if len(spy_aligned) >= 2:
        spy_return = float((spy_aligned.iloc[-1] / spy_aligned.iloc[0]) - 1.0)
        total_alpha = total_return - spy_return
    else:
        spy_return = None
        total_alpha = None

    if len(pf_aligned_to_spy) > 1:
        active_returns = pf_aligned_to_spy.values - spy_aligned_to_pf.values
        tracking_error_ann = float(np.std(active_returns, ddof=1) * np.sqrt(_TRADING_DAYS_PER_YEAR))
    else:
        tracking_error_ann = None

    valid_weights = target_weights.dropna(how="all")
    if len(valid_weights) > 0:
        per_rebal_active = (valid_weights.abs().sum(axis=1) - valid_weights.get(_SPY, pd.Series(0)).abs())
        mean_active_share = float(per_rebal_active.mean())
        mean_spy_weight = float(valid_weights.get(_SPY, pd.Series(np.nan)).mean()) if _SPY in valid_weights.columns else None
        diffs = valid_weights.diff().abs().sum(axis=1).dropna()
        annual_factor = _TRADING_DAYS_PER_YEAR / max(1, (len(valid_weights) - 1))
        turnover_one_way_ann = float(diffs.mean() * annual_factor)
    else:
        mean_active_share = None
        mean_spy_weight = None
        turnover_one_way_ann = None

    return {
        "total_return": total_return,
        "spy_return": spy_return,
        "total_alpha": total_alpha,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "tracking_error_ann": tracking_error_ann,
        "mean_active_share": mean_active_share,
        "mean_spy_weight": mean_spy_weight,
        "turnover_one_way_ann": turnover_one_way_ann,
    }


def compare_to_legacy(
    optimizer_metrics: dict,
    legacy_metrics: dict | None,
) -> dict:
    """
    Build a side-by-side dict matching the cutover gate validator's input
    shape (PR 4). When legacy_metrics is None, the deltas are None — useful
    for first-run reports where the legacy run hasn't been recorded yet.

    Gate thresholds (per plan doc):
        Sharpe_opt   >= Sharpe_leg × 0.9
        alpha_opt    >= alpha_leg + 0.005   (50 bps annualized)
        max_dd_opt   >= -|max_dd_leg| × 1.2 (less-negative = better)
        turnover_opt <= turnover_leg × 2.5
        tracking_err in [0.02, 0.06]
        active_share in [0.08, 0.25]
    """
    out: dict = {"optimizer": dict(optimizer_metrics)}
    if legacy_metrics is None:
        out["legacy"] = None
        out["deltas"] = None
        out["gate_thresholds"] = _gate_thresholds(optimizer_metrics, None)
        return out

    out["legacy"] = dict(legacy_metrics)
    out["deltas"] = {
        "sharpe_delta": (optimizer_metrics.get("sharpe_ratio") or 0.0)
                        - (legacy_metrics.get("sharpe_ratio") or 0.0),
        "alpha_delta": (optimizer_metrics.get("total_alpha") or 0.0)
                       - (legacy_metrics.get("total_alpha") or 0.0),
        "max_drawdown_delta": (optimizer_metrics.get("max_drawdown") or 0.0)
                              - (legacy_metrics.get("max_drawdown") or 0.0),
        "turnover_ratio": _safe_ratio(
            optimizer_metrics.get("turnover_one_way_ann"),
            legacy_metrics.get("turnover_one_way_ann"),
        ),
    }
    out["gate_thresholds"] = _gate_thresholds(optimizer_metrics, legacy_metrics)
    return out


def _gate_thresholds(
    optimizer_metrics: dict, legacy_metrics: dict | None,
) -> dict:
    if legacy_metrics is None:
        return {
            "sharpe_min": None,
            "alpha_min": None,
            "max_drawdown_floor": None,
            "turnover_max": None,
            "tracking_error_range": [0.02, 0.06],
            "active_share_range": [0.08, 0.25],
        }
    sharpe_leg = legacy_metrics.get("sharpe_ratio") or 0.0
    alpha_leg = legacy_metrics.get("total_alpha") or 0.0
    dd_leg = legacy_metrics.get("max_drawdown") or 0.0
    turnover_leg = legacy_metrics.get("turnover_one_way_ann") or 0.0
    return {
        "sharpe_min": sharpe_leg * 0.9,
        "alpha_min": alpha_leg + 0.005,
        "max_drawdown_floor": dd_leg * 1.2,
        "turnover_max": turnover_leg * 2.5,
        "tracking_error_range": [0.02, 0.06],
        "active_share_range": [0.08, 0.25],
    }


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator) / float(denominator)
