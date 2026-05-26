"""
Portfolio-optimizer backtest — PR 3 of portfolio-optimizer-260511 arc.

Replays the constrained MVO optimizer (alpha-engine PR #157,
executor/portfolio_optimizer.py) over historical synthetic predictions from
synthetic/predictor_backtest.py, then simulates the resulting target-weight
trajectory through vectorbt. Produces the skilled-risk metric basket
required by the cutover gate validator (PR 4): Sortino, PSR (confidence
Sharpe > 0), CVaR(95), max DD, plus optimizer-construction metrics
(turnover, tracking error, active share, mean SPY weight).

Anchor — Sortino is the primary risk-adjusted measure, PSR is the
confidence-significance gate, CVaR + max DD cover tail / peak-to-trough
risk. Raw Sharpe + alpha vs SPY are emitted for observability/presentation
only — they are NOT gate inputs. Mirrors the evaluator-revamp framework
established in optimizer/executor_optimizer.py (Sortino-primary,
PSR ≥ 0.95 confidence gate) — see evaluator-revamp-260506.md.

Unit conventions (canonical-alpha framework — see triple-barrier-260510.md
+ the 2026-05-09 21d log-domain cutover, alpha-engine-predictor PRs A-E):
  * alpha_hat: log-domain decimal at 21d horizon (matches predictor's
    canonical_predicted_alpha)
  * returns_panel: daily LOG returns (ln(P_t / P_{t-1})), Ledoit-Wolf
    shrunk in the kernel. Daily log variance compounds linearly to
    higher horizons (Var_T = T · Var_daily for iid log returns), so the
    21d alpha_hat and daily Σ live in the same log-units family — the
    MVO objective wᵀα̂ − λ·wᵀΣw is dimensionally consistent up to the
    horizon ratio absorbed by λ.

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
from typing import Any, Callable

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

    returns = np.log(history_subset).diff().dropna().values
    if returns.shape[0] < _DEFAULT_MIN_RETURNS:
        raise _InsufficientHistoryError(
            f"Returns panel has only {returns.shape[0]} rows after log-diff.dropna"
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
    """
    Simulate target-weight trajectory through vectorbt and emit the
    skilled-risk metric basket (Sortino / PSR / CVaR / max DD), with raw
    Sharpe + alpha-vs-SPY available for observability/presentation only.

    Anchor follows the evaluator-revamp framework (workstream D of
    evaluator-revamp-260506.md): Sortino is the primary risk-adjusted
    measure, PSR (confidence Sharpe > 0) is the statistical-significance
    gate, CVaR(95) captures tail risk, max_drawdown captures peak-to-trough.
    Raw Sharpe is computed for legacy comparison but is NOT a gating metric
    — see optimizer/executor_optimizer.py:300-325 + tests/test_executor_optimizer.py
    for the established pattern.
    """
    import vectorbt as vbt
    from vectorbt_bridge import portfolio_stats

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

    spy_aligned = spy_prices.reindex(target_weights.index).dropna()
    stats = portfolio_stats(pf, spy_prices=spy_aligned)

    daily_returns = stats.pop("daily_returns", None)
    stats.pop("daily_log_returns", None)

    if daily_returns is not None and len(daily_returns) > 1:
        spy_daily = spy_aligned.pct_change().dropna()
        spy_aligned_to_pf = spy_daily.reindex(daily_returns.index).dropna()
        pf_aligned_to_spy = daily_returns.reindex(spy_aligned_to_pf.index)
        if len(pf_aligned_to_spy) > 1:
            active_returns = pf_aligned_to_spy.values - spy_aligned_to_pf.values
            tracking_error_ann = float(np.std(active_returns, ddof=1) * np.sqrt(_TRADING_DAYS_PER_YEAR))
        else:
            tracking_error_ann = None
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
        "sortino_ratio": stats.get("sortino_ratio"),
        "psr": stats.get("psr"),
        "cvar_95": stats.get("cvar_95"),
        "max_drawdown": stats.get("max_drawdown"),
        "calmar_ratio": stats.get("calmar_ratio"),
        "tracking_error_ann": tracking_error_ann,
        "mean_active_share": mean_active_share,
        "mean_spy_weight": mean_spy_weight,
        "turnover_one_way_ann": turnover_one_way_ann,
        "sharpe_ratio": stats.get("sharpe_ratio"),
        "total_return": stats.get("total_return"),
        "spy_return": stats.get("spy_return"),
        "total_alpha": stats.get("total_alpha"),
        "total_trades": stats.get("total_trades"),
        "win_rate": stats.get("win_rate"),
    }


_DEFAULT_MIN_PSR = 0.95
_DEFAULT_SORTINO_DEGRADE_RATIO = 0.9

# Absolute risk-tolerance floors (ROADMAP L124, enhanced-index design
# intent decided 2026-05-16). Previously max_drawdown_floor / cvar_95_floor
# were legacy-scaled (dd_leg × 1.2), which made the gate circular: on the
# synthetic replay legacy barely trades (50 trades / 10y, -1.1% max DD), so
# "must not be riskier than a portfolio that barely trades" is unfalsifiable
# noise — exactly why the 2026-05-13 cutover gate FAILed 5/7 and was
# manually overridden. These are now absolute, anchored to an enhanced-index
# (SPY core + tilt) risk appetite: an enhanced-index book tracks SPY closely
# so its worst-case peak-to-trough should be ~SPY-like, not unbounded.
#   - max_drawdown_floor -0.35 ≈ SPY's deepest 10y drawdown (COVID 2020 ~-34%)
#   - cvar_95_floor      -0.05 ≈ a generous outer bound on the worst-5%-day
#                                 expected loss (SPY daily CVaR95 ≈ -2.8%)
# These MAGNITUDES are operator-tunable risk-appetite parameters, not settled
# doctrine — the structural fix is "absolute, always-applied" vs "circular,
# skipped-without-legacy"; the specific numbers are a defensible starting
# point Brian can tighten/loosen.
_ABS_MAX_DRAWDOWN_FLOOR = -0.35
_ABS_CVAR_95_FLOOR = -0.05


def compare_to_legacy(
    optimizer_metrics: dict,
    legacy_metrics: dict | None,
    *,
    signal_source: str = "synthetic",
) -> dict:
    """
    Build a side-by-side dict matching the cutover gate validator's input
    shape (PR 4). Anchored on the skilled-risk basket — Sortino is the
    primary risk-adjusted measure, PSR is the statistical-significance
    confidence gate, CVaR(95) + max_drawdown cover tail / peak-to-trough
    risk. Raw Sharpe and alpha-vs-SPY are emitted as observability /
    presentation metrics, NOT as gate inputs.

    Mirrors the established executor_optimizer skill-composite pattern
    (optimizer/executor_optimizer.py:300-325,446-472). When legacy_metrics
    is None, deltas are None — useful for first-run reports where the
    legacy baseline hasn't been recorded yet.

    Gate thresholds (per evaluator-revamp-260506.md / [[evaluator_revamp_skilled_risk]]):
        sortino_opt  >= sortino_leg × 0.9        (primary risk-adjusted)
        psr_opt      >= 0.95                     (confidence Sharpe > 0)
        max_dd_opt   >= max_dd_leg × 1.2         (less-negative = better)
        cvar_95_opt  >= cvar_95_leg × 1.2        (less-negative = better)
        turnover_opt <= turnover_leg × 2.5
        tracking_err in [0.02, 0.06]
        active_share in [0.08, 0.25]

    ``signal_source`` ("synthetic" | "production") is threaded into the
    output so the operator can interpret the verdict against the right
    input distribution (ROADMAP L124). The synthetic predictor-GBM replay
    runs at ~96.6% active-share vs ~15.5% on production research signals —
    a FAIL on synthetic is not the same statement as a FAIL on production.
    """
    out: dict = {"optimizer": dict(optimizer_metrics), "signal_source": signal_source}
    if legacy_metrics is None:
        out["legacy"] = None
        out["deltas"] = None
        out["gate_thresholds"] = _gate_thresholds(optimizer_metrics, None)
        return out

    out["legacy"] = dict(legacy_metrics)
    out["deltas"] = {
        "sortino_delta": (optimizer_metrics.get("sortino_ratio") or 0.0)
                         - (legacy_metrics.get("sortino_ratio") or 0.0),
        "psr_delta": (optimizer_metrics.get("psr") or 0.0)
                     - (legacy_metrics.get("psr") or 0.0),
        "cvar_95_delta": (optimizer_metrics.get("cvar_95") or 0.0)
                         - (legacy_metrics.get("cvar_95") or 0.0),
        "max_drawdown_delta": (optimizer_metrics.get("max_drawdown") or 0.0)
                              - (legacy_metrics.get("max_drawdown") or 0.0),
        "turnover_ratio": _safe_ratio(
            optimizer_metrics.get("turnover_one_way_ann"),
            legacy_metrics.get("turnover_one_way_ann"),
        ),
        "alpha_delta_presentation": (optimizer_metrics.get("total_alpha") or 0.0)
                                    - (legacy_metrics.get("total_alpha") or 0.0),
    }
    out["gate_thresholds"] = _gate_thresholds(optimizer_metrics, legacy_metrics)
    return out


def _gate_thresholds(
    optimizer_metrics: dict, legacy_metrics: dict | None,
) -> dict:
    """
    Skilled-risk gate thresholds. Sortino is primary; PSR is the
    confidence floor; CVaR + max_drawdown cap tail risk.

    ``max_drawdown_floor`` and ``cvar_95_floor`` are ABSOLUTE
    risk-tolerance values (``_ABS_*`` constants) applied regardless of
    whether a legacy baseline exists — ROADMAP L124. They were previously
    ``legacy × 1.2``, which made them circular (and ``None``/skipped with
    no legacy): the gate could only fail risk relative to a barely-trading
    synthetic-replay legacy. ``sortino_min`` / ``turnover_max`` remain
    legacy-relative (relative-performance / behavior comparisons, not
    absolute risk floors) and still skip cleanly without a baseline.

    Raw Sharpe and alpha-vs-SPY are intentionally absent — see
    [[alpha_vs_spy_is_presentation_not_gating]] and the
    [[evaluator_revamp_skilled_risk]] basket.
    """
    if legacy_metrics is None:
        return {
            "sortino_min": None,
            "psr_min": _DEFAULT_MIN_PSR,
            "max_drawdown_floor": _ABS_MAX_DRAWDOWN_FLOOR,
            "cvar_95_floor": _ABS_CVAR_95_FLOOR,
            "turnover_max": None,
            "tracking_error_range": [0.02, 0.06],
            "active_share_range": [0.08, 0.25],
        }
    sortino_leg = legacy_metrics.get("sortino_ratio") or 0.0
    turnover_leg = legacy_metrics.get("turnover_one_way_ann") or 0.0
    return {
        "sortino_min": sortino_leg * _DEFAULT_SORTINO_DEGRADE_RATIO,
        "psr_min": _DEFAULT_MIN_PSR,
        "max_drawdown_floor": _ABS_MAX_DRAWDOWN_FLOOR,
        "cvar_95_floor": _ABS_CVAR_95_FLOOR,
        "turnover_max": turnover_leg * 2.5,
        "tracking_error_range": [0.02, 0.06],
        "active_share_range": [0.08, 0.25],
    }


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator) / float(denominator)


# ─── A.4: Covariance-estimator sweep harness ────────────────────────────────
# Plan: alpha-engine-docs/private/optimizer-sota-upgrades-260526.md §A.4
#
# Sweeps the covariance-estimator dimension introduced by alpha-engine A.1
# (sigma_horizon_days), A.2 (ewma), A.3 (oas). Per-cell metrics use the same
# skilled-risk basket as the legacy comparator (Sortino primary, PSR
# confidence gate, CVaR + max_dd risk floors); winner is the Sortino-max
# cell that clears the absolute risk floors.
#
# Compensating λ rescale: when sigma_horizon_days=H≠1, λ is divided by H to
# keep the effective risk-aversion-against-21d-variance constant across
# cells (scaling-invariance proof from alpha-engine A.1). Without this, the
# H=21 cells would silently behave like λ=5/21 vs the H=1 cells at λ=5 — an
# unfair comparison that would always crown H=21 the winner.

_DEFAULT_BASE_RISK_AVERSION = 5.0


def _build_cell_cfg(
    estimator: str,
    sigma_horizon_days: int,
    *,
    ewma_lambda_decay: float | None = None,
    base_risk_aversion: float = _DEFAULT_BASE_RISK_AVERSION,
) -> dict:
    """Build an optimizer cfg for one sweep cell with compensating λ rescale."""
    cfg: dict[str, Any] = {
        "covariance_shrinkage": estimator,
        "sigma_horizon_days": sigma_horizon_days,
        # Keep effective risk-aversion constant: solving (Σ_H, λ/H) yields
        # the same optimum as (Σ_1, λ) per the A.1 scaling-invariance proof.
        "risk_aversion": base_risk_aversion / sigma_horizon_days,
    }
    if estimator == "ewma":
        if ewma_lambda_decay is None:
            raise ValueError("ewma cells must specify ewma_lambda_decay")
        cfg["ewma_lambda_decay"] = ewma_lambda_decay
    return cfg


def default_cov_sweep_cells(
    base_risk_aversion: float = _DEFAULT_BASE_RISK_AVERSION,
) -> list[tuple[str, dict]]:
    """Eight-cell default sweep covering LW × OAS × EWMA × H ∈ {1, 21}
    × λ_decay ∈ {0.94, 0.97} (EWMA only). Baseline is the first cell."""
    return [
        ("ledoit_wolf_h1",     _build_cell_cfg("ledoit_wolf", 1,  base_risk_aversion=base_risk_aversion)),
        ("ledoit_wolf_h21",    _build_cell_cfg("ledoit_wolf", 21, base_risk_aversion=base_risk_aversion)),
        ("oas_h1",             _build_cell_cfg("oas",         1,  base_risk_aversion=base_risk_aversion)),
        ("oas_h21",            _build_cell_cfg("oas",         21, base_risk_aversion=base_risk_aversion)),
        ("ewma_lambda094_h1",  _build_cell_cfg("ewma",        1,  ewma_lambda_decay=0.94, base_risk_aversion=base_risk_aversion)),
        ("ewma_lambda097_h1",  _build_cell_cfg("ewma",        1,  ewma_lambda_decay=0.97, base_risk_aversion=base_risk_aversion)),
        ("ewma_lambda094_h21", _build_cell_cfg("ewma",        21, ewma_lambda_decay=0.94, base_risk_aversion=base_risk_aversion)),
        ("ewma_lambda097_h21", _build_cell_cfg("ewma",        21, ewma_lambda_decay=0.97, base_risk_aversion=base_risk_aversion)),
    ]


def run_cov_estimator_sweep(
    predictions_by_date: dict[str, dict[str, float]],
    price_matrix: pd.DataFrame,
    spy_prices: pd.Series,
    sector_map: dict[str, str],
    executor_path: str,
    *,
    cells: list[tuple[str, dict]] | None = None,
    rebalance_freq_days: int = _DEFAULT_REBALANCE_FREQ,
    universe_cap: int = _DEFAULT_UNIVERSE_CAP,
    init_cash: float = 1_000_000.0,
    fees: float = 0.001,
    max_position_pct: float = 0.08,
    min_score_proxy: float | None = None,
    backtest_runner: Callable | None = None,
) -> dict:
    """Run the same backtest over multiple covariance-estimator configurations.

    Returns a structured report dict shaped for the cutover-gate validator:

      {
        "cells": {name: metrics_dict, ...},
        "baseline_name": str,                # first cell, conventionally LW H=1
        "winner_name": str | None,           # Sortino-max cell that clears gate
        "gate_passes_per_cell": {name: bool}, # per-cell verdict
        "ranking": [(name, sortino), ...],   # desc by Sortino, None last
        "cells_with_solver_failures": {name: count},
      }

    Args:
        predictions_by_date, price_matrix, spy_prices, sector_map, executor_path,
        rebalance_freq_days, universe_cap, init_cash, fees, max_position_pct,
        min_score_proxy: passed through to ``run_optimizer_backtest`` for each cell.
        cells: list of ``(name, optimizer_cfg)`` pairs. None → ``default_cov_sweep_cells()``.
        backtest_runner: injected for testing. Default uses ``run_optimizer_backtest``.
    """
    if cells is None:
        cells = default_cov_sweep_cells()
    if not cells:
        raise ValueError("Empty cell list — sweep must have at least one cell")

    runner = backtest_runner if backtest_runner is not None else run_optimizer_backtest

    metrics_by_cell: dict[str, dict] = {}
    failures_by_cell: dict[str, int] = {}
    for name, cfg in cells:
        logger.info(f"Running cov-sweep cell {name!r}: cfg={cfg}")
        result = runner(
            predictions_by_date=predictions_by_date,
            price_matrix=price_matrix,
            spy_prices=spy_prices,
            sector_map=sector_map,
            executor_path=executor_path,
            rebalance_freq_days=rebalance_freq_days,
            universe_cap=universe_cap,
            init_cash=init_cash,
            fees=fees,
            optimizer_cfg=cfg,
            max_position_pct=max_position_pct,
            min_score_proxy=min_score_proxy,
        )
        metrics_by_cell[name] = dict(result.metrics)
        metrics_by_cell[name]["cell_cfg"] = dict(cfg)
        failures_by_cell[name] = int(result.n_solver_failures)

    baseline_name = cells[0][0]
    baseline_metrics = metrics_by_cell.get(baseline_name)

    ranking = sorted(
        ((name, m.get("sortino_ratio")) for name, m in metrics_by_cell.items()),
        key=lambda kv: (kv[1] is None, -(kv[1] or 0.0)),
    )

    gate_passes_per_cell = {
        name: _cell_passes_gate(metrics, baseline_metrics)
        for name, metrics in metrics_by_cell.items()
    }
    winner_name = next(
        (name for name, _ in ranking if gate_passes_per_cell.get(name, False)),
        None,
    )

    return {
        "cells": metrics_by_cell,
        "baseline_name": baseline_name,
        "winner_name": winner_name,
        "gate_passes_per_cell": gate_passes_per_cell,
        "ranking": [(name, sortino) for name, sortino in ranking],
        "cells_with_solver_failures": failures_by_cell,
        "gate_thresholds": _gate_thresholds(
            metrics_by_cell.get(winner_name, baseline_metrics or {}),
            baseline_metrics,
        ),
    }


def _cell_passes_gate(cell_metrics: dict, baseline_metrics: dict | None) -> bool:
    """Apply the skilled-risk gate to a single sweep cell.

    Same rules as ``_gate_thresholds`` / ``compare_to_legacy``: PSR ≥ 0.95
    confidence floor, absolute max_drawdown / CVaR risk floors, and (when
    baseline exists) Sortino ≥ baseline × 0.9. No baseline → only the
    absolute floors apply, mirroring ``compare_to_legacy`` semantics.
    """
    psr = cell_metrics.get("psr")
    if psr is None or psr < _DEFAULT_MIN_PSR:
        return False
    max_dd = cell_metrics.get("max_drawdown")
    if max_dd is None or max_dd < _ABS_MAX_DRAWDOWN_FLOOR:
        return False
    cvar_95 = cell_metrics.get("cvar_95")
    if cvar_95 is None or cvar_95 < _ABS_CVAR_95_FLOOR:
        return False
    if baseline_metrics is not None:
        cell_sortino = cell_metrics.get("sortino_ratio") or 0.0
        baseline_sortino = baseline_metrics.get("sortino_ratio") or 0.0
        if cell_sortino < baseline_sortino * _DEFAULT_SORTINO_DEGRADE_RATIO:
            return False
    return True


def compare_cov_sweep_to_baseline(sweep_report: dict) -> dict:
    """Reshape a ``run_cov_estimator_sweep`` report into the cutover-gate
    validator input shape (one ``compare_to_legacy``-style entry per cell)."""
    cells = sweep_report["cells"]
    baseline_name = sweep_report["baseline_name"]
    baseline_metrics = cells.get(baseline_name)
    return {
        "baseline_name": baseline_name,
        "winner_name": sweep_report.get("winner_name"),
        "comparisons": {
            name: compare_to_legacy(metrics, baseline_metrics, signal_source="synthetic")
            for name, metrics in cells.items()
        },
        "ranking": sweep_report["ranking"],
        "gate_passes_per_cell": sweep_report["gate_passes_per_cell"],
    }
