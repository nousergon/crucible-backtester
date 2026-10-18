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

import contextlib
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

#: How ``_build_optimizer_inputs`` sets the solver's prior book. Stated on the
#: metrics so a reader of the artifact knows whether the turnover figures
#: describe a from-cash rebuild at every rebalance or a carried book.
#:
#: ``W_PREV_BASIS_CARRIED`` (alpha-engine-config-I11507, Brian 2026-10-15):
#: the prior book is the previous rebalance's target drifted by realised
#: returns to the new date and renormalised, and every held name stays in the
#: solver's universe (exempt from the score proxy, as live held names are
#: exempt from ``min_score_to_enter``). This mirrors the executor's
#: ``optimizer_shadow._build_w_prev`` / ``_build_universe``. The first
#: rebalance has no prior book and starts from cash.
W_PREV_BASIS_ALL_CASH = "all_cash_each_rebalance"
W_PREV_BASIS_CARRIED = "carried_drifted"
W_PREV_BASES = (W_PREV_BASIS_ALL_CASH, W_PREV_BASIS_CARRIED)
#: Default for every caller: the gate, the cov/gamma/param sweeps and the
#: deployed-strategy report. Flipped to carried on 2026-10-17 after the
#: 10-17 weekly's shadow (10 years, 399 rebalances): forced turnover over
#: budget 100% -> 0.5%, one-way turnover 31.1 -> 10.3/yr. The all-cash basis
#: now runs as the gate shadow for one cycle so the old numbers stay readable.
W_PREV_BASIS = W_PREV_BASIS_CARRIED

#: Post-solve churn controls (crucible-executor decision-precedence note,
#: 2026-10-15). ``none`` trades the solve as the live executor does;
#: ``partial_adjustment`` trades a fraction ``gp_k`` of each name's move
#: toward the solved target (Garleanu-Pedersen); ``reversal_guard`` holds a
#: name whose move would reverse its last trade within ``guard_sessions``
#: rebalances. Neither touches an ineligible name (tiers 0/1 force it), and
#: SPY absorbs the freed weight, as the executor's no-conviction fill.
TURNOVER_CONTROL_NONE = "none"
TURNOVER_CONTROL_PARTIAL = "partial_adjustment"
TURNOVER_CONTROL_GUARD = "reversal_guard"
TURNOVER_CONTROLS = (TURNOVER_CONTROL_NONE, TURNOVER_CONTROL_PARTIAL, TURNOVER_CONTROL_GUARD)
#: Window for the ``reversed_share`` churn metric, in rebalances.
_REVERSAL_WINDOW = 3


@dataclass
class OptimizerBacktestResult:
    target_weights: pd.DataFrame
    metrics: dict
    rebalance_dates: list[str]
    n_rebalances: int
    n_solver_failures: int = 0
    diagnostics_per_rebalance: list[dict] = field(default_factory=list)
    # Daily simple-return series of the simulated portfolio and the SPY
    # benchmark, aligned. Kept OFF the ``metrics`` dict (which JSON/parquet
    # round-trips through the sweep reports) so they never break
    # serialization — populated only when the caller passes
    # ``keep_daily_returns=True``. The weekly report's deployed-strategy
    # section (config#1053) needs them to build the beta-matched SPY
    # benchmark (the risk-matched headline metric). ``None`` otherwise.
    portfolio_daily_returns: "pd.Series | None" = None
    spy_daily_returns: "pd.Series | None" = None


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
    alpha_uncertainty_by_date: dict[str, dict[str, float]] | None = None,
    keep_daily_returns: bool = False,
    w_prev_basis: str = W_PREV_BASIS,
    turnover_control: str = TURNOVER_CONTROL_NONE,
    gp_k: float = 0.5,
    guard_sessions: int = 3,
    rebalance_band: float = 0.0,
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
        turnover_control / gp_k / guard_sessions: post-solve churn control
            (``TURNOVER_CONTROLS``). Anything but ``none`` needs the carried
            basis: on the all-cash basis there is no prior book to damp.
        rebalance_band: a name whose move is smaller than this stays put
            (the live executor's ``rebalance_band_pct``). 0 disables.

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

    if w_prev_basis not in W_PREV_BASES:
        raise ValueError(f"w_prev_basis must be one of {W_PREV_BASES}, got {w_prev_basis!r}")
    if turnover_control not in TURNOVER_CONTROLS:
        raise ValueError(
            f"turnover_control must be one of {TURNOVER_CONTROLS}, got {turnover_control!r}"
        )
    if turnover_control != TURNOVER_CONTROL_NONE and w_prev_basis != W_PREV_BASIS_CARRIED:
        raise ValueError("a turnover_control needs w_prev_basis='carried_drifted'")
    if not 0.0 < gp_k <= 1.0:
        raise ValueError(f"gp_k must be in (0, 1], got {gp_k}")
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
        f"universe_cap={universe_cap}, freq={rebalance_freq_days}d, "
        f"w_prev_basis={w_prev_basis}"
    )

    target_weights = pd.DataFrame(
        np.nan, index=price_matrix.index, columns=price_matrix.columns,
    )
    diagnostics: list[dict] = []
    n_solver_failures = 0
    # alpha-engine-config-I12117: per-name last written target (for the
    # unfillable-exit count) and per-rebalance solved universe (for the
    # stale-holding diagnostic).
    last_target = pd.Series(0.0, index=target_weights.columns)
    universe_by_date: dict[pd.Timestamp, frozenset] = {}
    n_unfillable_exits = 0
    carry = w_prev_basis == W_PREV_BASIS_CARRIED
    last_rebal_ts: pd.Timestamp | None = None
    churn = _ChurnTracker(turnover_control, gp_k, guard_sessions, rebalance_band)

    for rebal_n, rebal_date in enumerate(rebalance_dates):
        held_weights: dict[str, float] | None = None
        if carry and last_rebal_ts is not None:
            held_weights = _drift_book(
                last_target.drop(labels=[_CASH], errors="ignore"),
                price_matrix, last_rebal_ts, pd.Timestamp(rebal_date),
            )
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
                alpha_uncertainty_for_date=(
                    alpha_uncertainty_by_date.get(rebal_date, {})
                    if alpha_uncertainty_by_date is not None
                    else None
                ),
                held_weights=held_weights,
            )
        except _InsufficientHistoryError as e:
            logger.debug(f"{rebal_date}: skipped — {e}")
            diagnostics.append({"date": rebal_date, "status": "skipped_insufficient_history"})
            continue

        alpha_uncertainty = kwargs.pop("alpha_uncertainty", None)
        carry_diag = kwargs.pop("carry")
        with _simulated_gate_off_is_not_an_incident():
            result = solve_target_weights(**kwargs, alpha_uncertainty=alpha_uncertainty)
        if result.diagnostics["status"] == "infeasible_fallback":
            n_solver_failures += 1
        weights = churn.apply(
            rebal_n, kwargs["tickers"], result.weights, kwargs["w_prev"],
            kwargs["eligibility"], kwargs["spy_idx"], kwargs["cash_idx"],
        )
        diagnostics.append({
            "date": rebal_date,
            "status": result.diagnostics["status"],
            "n_active": result.diagnostics["n_active_positions"],
            "expected_alpha": result.diagnostics["expected_alpha"],
            # alpha-engine-config-I11507: the one-way turnover the solver's
            # pins FORCE before alpha is consulted, and what forced it.
            **_forced_turnover_fields(result.diagnostics),
            **({"carry": carry_diag} if carry else {}),
        })

        rebal_ts = pd.Timestamp(rebal_date)
        n_unfillable_exits += _populate_target_weights_row(
            target_weights, rebal_date, kwargs["tickers"], weights,
            close_row=price_matrix.loc[rebal_ts],
            held=last_target,
        )
        # Last WRITTEN target per name: an unfillable (NaN) cell keeps the
        # prior target, mirroring vectorbt holding the position unchanged.
        row = target_weights.loc[rebal_ts]
        last_target = last_target.where(row.isna(), row)
        universe_by_date[rebal_ts] = frozenset(
            t for t in kwargs["tickers"] if t != _CASH
        )
        last_rebal_ts = rebal_ts

    metrics, daily_returns, spy_daily_returns = _simulate_and_measure(
        target_weights=target_weights,
        price_matrix=price_matrix,
        spy_prices=spy_prices,
        init_cash=init_cash,
        fees=fees,
        rebalance_freq_days=rebalance_freq_days,
        universe_by_date=universe_by_date,
    )
    metrics["n_unfillable_exits"] = n_unfillable_exits
    metrics["n_rebalances"] = len(rebalance_dates)
    metrics["n_solver_failures"] = n_solver_failures
    metrics.update(summarize_forced_turnover(diagnostics, w_prev_basis=w_prev_basis))
    if carry:
        metrics.update(_summarize_carry(diagnostics))
        metrics.update(churn.summary())
    _log_forced_turnover(metrics)
    metrics["rebalance_freq_days"] = rebalance_freq_days
    metrics["universe_cap"] = universe_cap

    return OptimizerBacktestResult(
        target_weights=target_weights,
        metrics=metrics,
        rebalance_dates=rebalance_dates,
        n_rebalances=len(rebalance_dates),
        n_solver_failures=n_solver_failures,
        diagnostics_per_rebalance=diagnostics,
        # Series are heavy + non-serializable, so only retain them when the
        # caller (the deployed-strategy report headline) explicitly asks.
        portfolio_daily_returns=daily_returns if keep_daily_returns else None,
        spy_daily_returns=spy_daily_returns if keep_daily_returns else None,
    )


class _InsufficientHistoryError(Exception):
    pass


# ── Forced turnover (alpha-engine-config-I11507) ────────────────────────────
#
# The executor's optimizer raises its turnover budget to whatever its pins
# MANDATE (`executor/portfolio_optimizer.py::_mandatory_turnover_floor`), so
# that a forced exit is never starved of budget. On this backtest the floor
# is ~0.97 on every rebalance, so the configured budget (0.20) never binds
# and every gate/cov/gamma sweep cell measures a full-rebuild strategy.
#
# The cause is THIS module, not eligibility churn: `_build_optimizer_inputs`
# hands the solver ``w_prev = 100% cash`` on every rebalance instead of the
# book the previous rebalance built. The cash-sleeve equality pin
# (``cash_sleeve_pct``, 0.03) then forces 1 − 0.03 = 0.97 of NAV out of cash
# before alpha is consulted — exactly the 0.9700 in the log.
#
# What changes that is a modelling decision (carry the prior book forward,
# keep held names in the universe, optionally hysteresis) and is NOT made
# here. This only makes the forced fraction a first-class number on the
# stage, so the decision is taken on a measurement.

#: ``W_PREV_BASIS*`` constants: see the module top.


#: The live executor's "turnover is unthrottled" alarm (crucible-executor
#: ``_alert_gate_off_without_sigma``, alpha-engine-config-I11791).
_GATE_OFF_ALERT_PREFIX = "Conviction gate OFF ("


class _DemoteSimulatedGateOff(logging.Filter):
    """Demote the executor's gate-off ERROR to INFO inside a simulation.

    That ERROR means "today's LIVE discretionary turnover is unthrottled".
    Here the executor's solver runs over historical predictions that mostly
    carry no sigma, so the same line fired from every weekly backtest that
    reached a turnover-budgeted solve and paged as a live-trading fault
    (three times from rehearsal-2026-10-16-1, one per spot retry). The
    simulated gate state is not lost: it is in every rebalance's
    ``conviction_gate_reason`` diagnostic. Any other ERROR passes untouched.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR and str(record.msg).startswith(_GATE_OFF_ALERT_PREFIX):
            record.levelno = logging.INFO
            record.levelname = "INFO"
            record.msg = "[simulated solve, not the live book] " + str(record.msg)
        return True


@contextlib.contextmanager
def _simulated_gate_off_is_not_an_incident():
    target = logging.getLogger("executor.portfolio_optimizer")
    f = _DemoteSimulatedGateOff()
    target.addFilter(f)
    try:
        yield
    finally:
        target.removeFilter(f)


def _forced_turnover_fields(solver_diagnostics: dict) -> dict:
    """Per-rebalance forced-turnover fields from ``solve_target_weights``."""
    d = solver_diagnostics or {}
    return {
        "forced_turnover": d.get("turnover_mandatory_floor"),
        "forced_turnover_by_cause": d.get("turnover_mandatory_floor_by_cause"),
        "turnover_budget_configured": d.get("turnover_budget_configured"),
        "turnover_constraint_cap": d.get("turnover_constraint_cap"),
        "turnover_cap_source": d.get("turnover_cap_source"),
    }


def summarize_forced_turnover(
    diagnostics: list[dict], w_prev_basis: str = W_PREV_BASIS,
) -> dict:
    """Stage-level summary of the forced (mandatory) one-way turnover.

    ``forced_turnover_share_over_budget`` is the fraction of measured
    rebalances whose forced floor exceeded the configured budget, i.e. on
    which the budget could not bind at all. ``None`` everywhere (never 0)
    when no rebalance reported a floor, so "not measured" cannot read as
    "no forced turnover".
    """
    floors: list[float] = []
    over = 0
    causes: dict[str, list[float]] = {}
    for row in diagnostics:
        f = row.get("forced_turnover")
        if not isinstance(f, (int, float)) or not np.isfinite(f):
            continue
        floors.append(float(f))
        budget = row.get("turnover_budget_configured")
        if isinstance(budget, (int, float)) and f > float(budget) + 1e-9:
            over += 1
        for cause, v in (row.get("forced_turnover_by_cause") or {}).items():
            if cause.startswith("n_"):
                continue  # name counts, not turnover
            if isinstance(v, (int, float)) and np.isfinite(v):
                causes.setdefault(cause, []).append(float(v))
    n = len(floors)
    return {
        "w_prev_basis": w_prev_basis,
        "forced_turnover_n_rebalances": n,
        "forced_turnover_mean": float(np.mean(floors)) if n else None,
        "forced_turnover_median": float(np.median(floors)) if n else None,
        "forced_turnover_max": float(np.max(floors)) if n else None,
        "forced_turnover_share_over_budget": (over / n) if n else None,
        "forced_turnover_by_cause_mean": (
            {c: float(np.mean(v)) for c, v in sorted(causes.items())} if causes else None
        ),
    }


def _summarize_carry(diagnostics: list[dict]) -> dict:
    """Carried-basis bookkeeping: how much of the book sat outside the
    top-``universe_cap`` cut (kept in the universe by the carry), and how
    much could not be carried for lack of history (sold by the simulation)."""
    rows = [r["carry"] for r in diagnostics if isinstance(r.get("carry"), dict)]
    if not rows:
        return {"carry_n_rebalances": 0, "held_outside_cut_mean": None,
                "held_weight_not_carried_mean": None, "held_weight_not_carried_max": None}
    not_carried = [float(r["held_weight_not_carried"]) for r in rows]
    return {
        "carry_n_rebalances": len(rows),
        "held_outside_cut_mean": float(np.mean([r["n_held_outside_cut"] for r in rows])),
        "held_weight_not_carried_mean": float(np.mean(not_carried)),
        "held_weight_not_carried_max": float(np.max(not_carried)),
    }


class _ChurnTracker:
    """Applies the post-solve churn control and the band, and measures the
    churn the book actually trades (one-way, ex SPY and cash)."""

    def __init__(self, control: str, gp_k: float, guard_sessions: int, band: float):
        self.control = control
        self.gp_k = float(gp_k)
        self.guard_sessions = int(guard_sessions)
        self.band = float(band)
        self.last_dir: dict[str, tuple[int, int]] = {}
        self.traded = 0.0
        self.reversed = 0.0
        self.n_guard_holds = 0

    def apply(
        self, n: int, tickers: list[str], solved: np.ndarray, w_prev: np.ndarray,
        eligibility: np.ndarray, spy_idx: int, cash_idx: int,
    ) -> np.ndarray:
        tgt = np.asarray(solved, dtype=float).copy()
        for i, t in enumerate(tickers):
            if i in (spy_idx, cash_idx):
                continue
            forced = not bool(eligibility[i])
            delta = tgt[i] - w_prev[i]
            if forced:
                continue  # tiers 0/1: the solve's exit stands, untouched
            if self.control == TURNOVER_CONTROL_PARTIAL:
                tgt[i] = w_prev[i] + self.gp_k * delta
                delta = tgt[i] - w_prev[i]
            elif self.control == TURNOVER_CONTROL_GUARD and t in self.last_dir:
                last_n, sign = self.last_dir[t]
                if n - last_n <= self.guard_sessions and np.sign(delta) == -sign:
                    tgt[i] = w_prev[i]
                    self.n_guard_holds += 1
                    continue
            if abs(delta) < self.band:
                tgt[i] = w_prev[i]
        if self.control != TURNOVER_CONTROL_NONE or self.band > 0.0:
            tgt[spy_idx] += 1.0 - float(tgt.sum())  # SPY: the no-conviction fill
            if tgt[spy_idx] < 0.0:
                tgt[cash_idx] += tgt[spy_idx]
                tgt[spy_idx] = 0.0
        for i, t in enumerate(tickers):
            if i in (spy_idx, cash_idx):
                continue
            delta = tgt[i] - w_prev[i]
            if abs(delta) < 1e-6:
                continue
            self.traded += abs(delta)
            prior = self.last_dir.get(t)
            if prior is not None and n - prior[0] <= _REVERSAL_WINDOW and np.sign(delta) == -prior[1]:
                self.reversed += abs(delta)
            self.last_dir[t] = (n, int(np.sign(delta)))
        return tgt

    def summary(self) -> dict:
        return {
            "turnover_control": self.control,
            "gp_k": self.gp_k if self.control == TURNOVER_CONTROL_PARTIAL else None,
            "guard_sessions": self.guard_sessions if self.control == TURNOVER_CONTROL_GUARD else None,
            "rebalance_band": self.band,
            "names_traded_one_way_sum": self.traded,
            "reversed_share": self.reversed / self.traded if self.traded else 0.0,
            "n_guard_holds": self.n_guard_holds,
        }


def _log_forced_turnover(metrics: dict) -> None:
    n = metrics.get("forced_turnover_n_rebalances") or 0
    if not n:
        logger.warning(
            "Optimizer backtest: no rebalance reported a forced-turnover floor — "
            "forced turnover is UNMEASURED this run (alpha-engine-config-I11507)"
        )
        return
    logger.warning(
        "Optimizer backtest forced turnover (alpha-engine-config-I11507): "
        "median %.4f, max %.4f one-way per rebalance; the forced floor exceeded "
        "the configured budget on %.0f%% of %d rebalances; by cause (mean) %s; "
        "w_prev_basis=%s",
        metrics["forced_turnover_median"], metrics["forced_turnover_max"],
        100.0 * (metrics["forced_turnover_share_over_budget"] or 0.0), n,
        metrics.get("forced_turnover_by_cause_mean"), metrics["w_prev_basis"],
    )


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
    alpha_uncertainty_for_date: dict[str, float] | None = None,
    held_weights: dict[str, float] | None = None,
) -> dict:
    """Solver kwargs for one rebalance.

    ``held_weights`` is the carried prior book ({ticker: NAV weight}, cash
    implied as the remainder; SPY allowed). ``None`` keeps the all-cash
    basis. When given, every held name with a full return history in the
    lookback window joins the universe beside the top-``universe_cap`` cut,
    with today's predicted alpha (0 when it has none), and ``w_prev`` is the
    held book. A held name that cannot join (no finite history) is sold by
    the simulation; its weight is reported as ``held_weight_not_carried``
    and handed to the solver as cash.
    """
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

    held = {
        t: float(w) for t, w in (held_weights or {}).items()
        if t not in (_SPY, _CASH) and np.isfinite(w) and w > 0.0
    }
    held_not_carried = 0.0
    for t in sorted(held):
        if t in selected:
            continue
        if (
            t in history.columns
            and t in price_matrix.columns
            and np.isfinite(price_matrix.at[rebal_ts, t])
            and history[t].notna().all()
        ):
            selected.append(t)
        else:
            held_not_carried += held[t]

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
        a = predictions.get(t)
        alpha_hat[i] = float(a) if a is not None and pd.notna(a) else 0.0
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
            if t in held:
                continue  # live: held names are exempt from the entry score gate
            if abs(alpha_hat[i]) < min_score_proxy:
                eligibility[i] = False

    # All-cash basis: every rebalance starts from 100% cash — the source of
    # the ~0.97 forced turnover per rebalance (alpha-engine-config-I11507).
    # Carried basis: the held book, as the executor's ``_build_w_prev``.
    w_prev = np.zeros(N)
    if held_weights is None:
        w_prev[cash_idx] = 1.0
    else:
        for i, t in enumerate(selected):
            w_prev[i] = held.get(t, 0.0)
        spy_w = (held_weights or {}).get(_SPY, 0.0)
        w_prev[spy_idx] = float(spy_w) if np.isfinite(spy_w) and spy_w > 0 else 0.0
        w_prev[cash_idx] = max(0.0, 1.0 - float(w_prev.sum()))

    # B.4 (optimizer-sota-upgrades-260526.md §B.4b): per-ticker σ_α̂ from
    # the BayesianRidge posterior. None when caller didn't pass uncertainty
    # → solve_target_weights treats as absent (penalty term skipped).
    # SPY/CASH always 0; missing entries → NaN, downstream B.3 path treats
    # as zero penalty per partial-rollout contract.
    alpha_uncertainty: np.ndarray | None = None
    if alpha_uncertainty_for_date is not None:
        alpha_uncertainty = np.full(N, np.nan)
        for i, t in enumerate(selected):
            v = alpha_uncertainty_for_date.get(t)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv) and fv >= 0.0:
                alpha_uncertainty[i] = fv
        alpha_uncertainty[spy_idx] = 0.0
        alpha_uncertainty[cash_idx] = 0.0

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
        "alpha_uncertainty": alpha_uncertainty,
        "carry": {
            "n_held": len(held),
            "n_held_outside_cut": sum(1 for t in held if t not in ranked[:universe_cap]),
            "held_weight_not_carried": held_not_carried,
        },
    }


def _drift_book(
    book: pd.Series,
    price_matrix: pd.DataFrame,
    from_ts: pd.Timestamp,
    to_ts: pd.Timestamp,
) -> dict[str, float]:
    """Drift a target book ({ticker: NAV weight}, cash = 1 − Σ) by realised
    price moves from ``from_ts`` to ``to_ts`` and renormalise.

    A name without a finite price on either date is held at its last price
    (ratio 1), matching vectorbt, which cannot trade or mark it.
    """
    w = book[book > 0.0]
    if w.empty:
        return {}
    cash = max(0.0, 1.0 - float(w.sum()))
    p0 = pd.to_numeric(price_matrix.loc[from_ts].reindex(w.index), errors="coerce")
    p1 = pd.to_numeric(price_matrix.loc[to_ts].reindex(w.index), errors="coerce")
    ratio = (p1 / p0).where(np.isfinite(p1 / p0) & (p0 > 0), 1.0)
    grown = w * ratio
    total = float(grown.sum()) + cash
    if not np.isfinite(total) or total <= 0.0:
        return {}
    return {t: float(v) for t, v in (grown / total).items()}


def _populate_target_weights_row(
    target_weights: pd.DataFrame,
    rebal_date: str,
    universe: list[str],
    weights: np.ndarray,
    close_row: "pd.Series | None" = None,
    held: "pd.Series | None" = None,
) -> int:
    """Write one rebalance row of the target-weight frame.

    Every non-cash column OUTSIDE the solved universe is set to an explicit
    ``0.0`` before the solved weights are written (alpha-engine-config-I12117,
    defect A). vectorbt's ``from_orders(size_type="targetpercent")`` reads a
    ``NaN`` size as *no order*, so a name that left the universe used to keep
    its position — the simulated book was "optimizer target + stale
    holdings", and with ``cash_sharing=True`` the stale position also starved
    new targets of cash.

    Guard: a column whose close is not finite and positive on ``rebal_date``
    (a delisted or not-yet-listed name), solved or not, is left ``NaN`` — an
    order vectorbt cannot fill. When
    such a column is still HELD (``held`` carries a positive last target), it
    is counted and the count is returned, so the stuck exit is visible as
    ``n_unfillable_exits`` rather than silent.

    ``close_row`` / ``held`` default to ``None`` (no price guard, nothing held)
    for callers that only need the row written.
    """
    rebal_ts = pd.Timestamp(rebal_date)
    cols = [c for c in target_weights.columns if c != _CASH]
    if close_row is not None:
        close = pd.to_numeric(close_row.reindex(cols), errors="coerce").to_numpy(dtype=float)
        # vectorbt refuses any order whose price is not finite AND > 0
        # ("order.price must be finite and greater than 0").
        with np.errstate(invalid="ignore"):
            fillable = np.isfinite(close) & (close > 0)
    else:
        fillable = np.ones(len(cols), dtype=bool)
    fill_cols = [c for c, ok in zip(cols, fillable) if ok]
    unfillable_cols = [c for c, ok in zip(cols, fillable) if not ok]
    target_weights.loc[rebal_ts, fill_cols] = 0.0

    unfillable = set(unfillable_cols)
    for i, t in enumerate(universe):
        if t == _CASH:
            continue
        # A SOLVED name can be unfillable too: the optimizer reads history up
        # to the rebalance date, so a name whose close is bad ON that date
        # (missing, or a zero / negative vendor print) still gets a weight.
        # Writing it handed vectorbt an order at a non-positive price, which
        # raised "order.price must be finite and greater than 0" and lost the
        # whole optimizer backtest (twice in rehearsal-2026-10-16-1).
        if t in target_weights.columns and t not in unfillable:
            target_weights.at[rebal_ts, t] = float(weights[i])

    n_unfillable_exits = 0
    if held is not None and unfillable_cols:
        held_w = pd.to_numeric(held.reindex(unfillable_cols), errors="coerce").fillna(0.0)
        n_unfillable_exits = int((held_w > 0).sum())
    return n_unfillable_exits


def _simulate_and_measure(
    target_weights: pd.DataFrame,
    price_matrix: pd.DataFrame,
    spy_prices: pd.Series,
    init_cash: float,
    fees: float,
    rebalance_freq_days: int = _DEFAULT_REBALANCE_FREQ,
    universe_by_date: "dict[pd.Timestamp, frozenset] | None" = None,
) -> tuple[dict, "pd.Series | None", "pd.Series | None"]:
    """
    Simulate target-weight trajectory through vectorbt and emit the
    skilled-risk metric basket (Sortino / PSR / CVaR / max DD), with raw
    Sharpe + alpha-vs-SPY available for observability/presentation only.

    Returns ``(metrics, portfolio_daily_returns, spy_daily_returns)`` where
    the two return series are aligned on the portfolio's active window (or
    ``None`` when fewer than 2 aligned days). The series feed the
    beta-matched SPY benchmark used as the deployed-strategy headline
    (config#1053); they are returned separately rather than stuffed into
    ``metrics`` because ``metrics`` JSON/parquet round-trips downstream.

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
    # NaN = "no order" in vectorbt. Non-rebalance rows are all-NaN (hold);
    # on a rebalance row only an unfillable (non-finite close) name is NaN —
    # every other name outside the solved universe carries an explicit 0.0
    # (alpha-engine-config-I12117, see _populate_target_weights_row).
    size = target_weights.copy()

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

    # Aligned (portfolio, SPY) daily simple returns over the overlap of the
    # two series — the basis for both the tracking-error number and the
    # beta-matched SPY benchmark (config#1053 risk-matched headline). None
    # when fewer than 2 aligned days.
    pf_returns_aligned: "pd.Series | None" = None
    spy_returns_aligned: "pd.Series | None" = None
    if daily_returns is not None and len(daily_returns) > 1:
        spy_daily = spy_aligned.pct_change().dropna()
        spy_aligned_to_pf = spy_daily.reindex(daily_returns.index).dropna()
        pf_aligned_to_spy = daily_returns.reindex(spy_aligned_to_pf.index)
        if len(pf_aligned_to_spy) > 1:
            pf_returns_aligned = pf_aligned_to_spy
            spy_returns_aligned = spy_aligned_to_pf
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
        turnover_one_way_ann = _turnover_one_way_ann(valid_weights, rebalance_freq_days)
    else:
        mean_active_share = None
        mean_spy_weight = None
        turnover_one_way_ann = None

    stale_mean, stale_max = _stale_holding_weight(pf, universe_by_date)

    metrics = {
        "sortino_ratio": stats.get("sortino_ratio"),
        "psr": stats.get("psr"),
        "cvar_95": stats.get("cvar_95"),
        "max_drawdown": stats.get("max_drawdown"),
        "calmar_ratio": stats.get("calmar_ratio"),
        "tracking_error_ann": tracking_error_ann,
        "mean_active_share": mean_active_share,
        "mean_spy_weight": mean_spy_weight,
        "turnover_one_way_ann": turnover_one_way_ann,
        "turnover_basis": _TURNOVER_BASIS,
        "stale_holding_weight_mean": stale_mean,
        "stale_holding_weight_max": stale_max,
        "sharpe_ratio": stats.get("sharpe_ratio"),
        "total_return": stats.get("total_return"),
        "spy_return": stats.get("spy_return"),
        "total_alpha": stats.get("total_alpha"),
        "total_trades": stats.get("total_trades"),
        "win_rate": stats.get("win_rate"),
    }
    return metrics, pf_returns_aligned, spy_returns_aligned


#: Declares how ``turnover_one_way_ann`` is computed, so an artifact written
#: before alpha-engine-config-I12117 (two-way Σ|Δw| on overlapping names,
#: × 252 / (n_rebalance_rows − 1)) is distinguishable from one written after.
_TURNOVER_BASIS = "half_sum_abs_dw_incl_cash_x_252_over_freq"


def _turnover_one_way_ann(
    rebalance_rows: pd.DataFrame, rebalance_freq_days: int,
) -> float | None:
    """One-way turnover, annualised (alpha-engine-config-I12117, defect B).

    ½·Σ|Δw| between consecutive rebalance target rows, cash included as
    ``1 − Σ non-cash``, averaged over rebalances (the first row has no prior
    target and is dropped, not averaged in as a zero), × rebalances per year
    = ``252 / rebalance_freq_days``.

    A ``NaN`` cell on a rebalance row is an unfillable order (no finite
    close); vectorbt leaves that position unchanged, so it carries the prior
    row's target here. A name never targeted counts as 0.

    Returns ``None`` (unmeasured, not zero) with fewer than two rebalance rows.
    """
    if len(rebalance_rows) < 2:
        return None
    filled = rebalance_rows.drop(columns=[_CASH], errors="ignore").ffill().fillna(0.0)
    filled = filled.assign(**{_CASH: 1.0 - filled.sum(axis=1)})
    per_rebalance = 0.5 * filled.diff().abs().sum(axis=1).iloc[1:]
    rebalances_per_year = _TRADING_DAYS_PER_YEAR / max(1, int(rebalance_freq_days))
    return float(per_rebalance.mean() * rebalances_per_year)


def _stale_holding_weight(
    pf: Any, universe_by_date: "dict[pd.Timestamp, frozenset] | None",
) -> tuple[float | None, float | None]:
    """Simulated NAV weight held in names OUTSIDE each rebalance's solved
    universe, read from ``pf`` at the close of that rebalance bar.

    Measures alpha-engine-config-I12117 defect A directly: on the pre-fix
    code it is the size of the stale book; after the fix it must read 0
    apart from counted unfillable exits (``n_unfillable_exits``).
    """
    if not universe_by_date:
        return None, None
    asset_w = pf.asset_value(group_by=False).div(pf.value(), axis=0)
    stale: list[float] = []
    for ts, universe in universe_by_date.items():
        if ts not in asset_w.index:
            continue
        row = asset_w.loc[ts]
        outside = [c for c in row.index if c not in universe]
        stale.append(float(row[outside].abs().sum()) if outside else 0.0)
    if not stale:
        return None, None
    return float(np.mean(stale)), float(np.max(stale))


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


# ─── B.4b: α̂-uncertainty γ-sweep harness ───────────────────────────────────
# Plan: alpha-engine-docs/private/optimizer-sota-upgrades-260526.md §B.4b
#
# Sweeps cfg["alpha_uncertainty_penalty"] = γ over a log-spaced grid to
# find the γ that maximizes Sortino on the synthetic-replay backtest.
# Mirror of A.4 cov-sweep architecture: per-cell run_optimizer_backtest,
# Sortino ranking, absolute-floor gate, baseline = γ=0 (legacy MVO).
#
# Compensating note: γ=0 produces the SAME result regardless of whether
# alpha_uncertainty is provided (per the B.3 contract: γ=0 → penalty
# skipped). So the γ=0 cell is the legacy baseline by construction —
# perfect for the cutover decision.


def default_gamma_sweep_cells() -> list[tuple[str, dict]]:
    """Log-spaced γ grid covering the realistic operating range for
    a 21d-horizon canonical-alpha book.

    γ semantics: the penalty is γ · sum_i(σ_α̂_i² · w_i²). With typical
    σ_α̂ ≈ 0.02 (2% log-domain 21d std) and w_i² ≈ 0.08² ≈ 6e-3, one
    pick's penalty contribution is γ · 4e-4 · 6e-3 ≈ γ · 2.4e-6.

    The α̂ gain per pick is roughly α̂ · w ≈ 0.05 · 0.08 = 4e-3. So:
      • γ = 10 → penalty ≈ 2.4e-5 per pick (∼0.6% of α̂ gain — negligible)
      • γ = 100 → penalty ≈ 2.4e-4 per pick (∼6% of α̂ gain — small tilt)
      • γ = 1000 → penalty ≈ 2.4e-3 per pick (∼60% of α̂ gain — strong)
      • γ = 10000 → penalty dominates the α̂ gain → conviction picks shrink
        substantially

    Baseline is γ=0 (legacy MVO behavior, no penalty).
    """
    return [
        ("baseline_gamma_0",   {"alpha_uncertainty_penalty": 0.0}),
        ("gamma_10",           {"alpha_uncertainty_penalty": 10.0}),
        ("gamma_100",          {"alpha_uncertainty_penalty": 100.0}),
        ("gamma_1000",         {"alpha_uncertainty_penalty": 1000.0}),
        ("gamma_10000",        {"alpha_uncertainty_penalty": 10000.0}),
    ]


def run_gamma_sweep(
    predictions_by_date: dict[str, dict[str, float]],
    price_matrix: pd.DataFrame,
    spy_prices: pd.Series,
    sector_map: dict[str, str],
    executor_path: str,
    alpha_uncertainty_by_date: dict[str, dict[str, float]],
    *,
    cells: list[tuple[str, dict]] | None = None,
    base_optimizer_cfg: dict | None = None,
    rebalance_freq_days: int = _DEFAULT_REBALANCE_FREQ,
    universe_cap: int = _DEFAULT_UNIVERSE_CAP,
    init_cash: float = 1_000_000.0,
    fees: float = 0.001,
    max_position_pct: float = 0.08,
    min_score_proxy: float | None = None,
    backtest_runner: Callable | None = None,
) -> dict:
    """Sweep alpha_uncertainty_penalty γ to find the value that maximizes
    Sortino on the synthetic-replay backtest.

    ``alpha_uncertainty_by_date`` is required: a γ-sweep against missing
    σ_α̂ would have every cell behave identically (B.3 skips the penalty
    when alpha_uncertainty has no signal) — so caller MUST provide it.

    Report shape matches ``run_cov_estimator_sweep`` so the same downstream
    operator tooling can read both verdicts uniformly.
    """
    if alpha_uncertainty_by_date is None:
        raise ValueError(
            "alpha_uncertainty_by_date is required for γ-sweep — without "
            "uncertainty input every γ cell collapses to the baseline solve. "
            "Generate σ_α̂ inputs (BayesianRidge posterior — predictor B.1) "
            "before running the sweep."
        )
    if cells is None:
        cells = default_gamma_sweep_cells()
    if not cells:
        raise ValueError("Empty cell list — sweep must have at least one cell")

    runner = backtest_runner if backtest_runner is not None else run_optimizer_backtest
    base_cfg = dict(base_optimizer_cfg or {})

    metrics_by_cell: dict[str, dict] = {}
    failures_by_cell: dict[str, int] = {}
    for name, gamma_overrides in cells:
        cfg = {**base_cfg, **gamma_overrides}
        logger.info(f"Running γ-sweep cell {name!r}: cfg={cfg}")
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
            alpha_uncertainty_by_date=alpha_uncertainty_by_date,
        )
        metrics_by_cell[name] = dict(result.metrics)
        metrics_by_cell[name]["cell_cfg"] = dict(cfg)
        failures_by_cell[name] = int(result.n_solver_failures)

    baseline_name = cells[0][0]  # γ=0 by convention
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


def default_optimizer_param_sweep_cells() -> list[tuple[str, dict]]:
    """Grid over the two PRIMARY MVO knobs the cov-/γ-sweeps don't cover
    (config#1057): ``risk_aversion`` (λ — variance-penalty strength) ×
    ``tcost_bps`` (turnover penalty, bps per $1 traded).

    Defaults are ``risk_aversion=5.0`` / ``tcost_bps=5.0`` (OPTIMIZER_CONFIG_
    DEFAULTS); the baseline cell pins those so the gate measures every
    challenger against the live behavior. Grid kept modest (9 cells) — each cell
    is a full optimizer backtest over the predictions-archive window, and this is
    a weekly batch stage.
    """
    cells: list[tuple[str, dict]] = []
    for ra in (5.0, 3.0, 7.0):          # baseline (5.0) FIRST — cells[0] is baseline
        for tc in (5.0, 2.0, 10.0):     # baseline tcost (5.0) first within ra
            name = (
                "baseline_ra5_tc5"
                if (ra == 5.0 and tc == 5.0)
                else f"ra{ra:g}_tc{tc:g}"
            )
            cells.append((name, {"risk_aversion": ra, "tcost_bps": tc}))
    return cells


def run_optimizer_param_sweep(
    predictions_by_date: dict[str, dict[str, float]],
    price_matrix: pd.DataFrame,
    spy_prices: pd.Series,
    sector_map: dict[str, str],
    executor_path: str,
    *,
    cells: list[tuple[str, dict]] | None = None,
    base_optimizer_cfg: dict | None = None,
    rebalance_freq_days: int = _DEFAULT_REBALANCE_FREQ,
    universe_cap: int = _DEFAULT_UNIVERSE_CAP,
    init_cash: float = 1_000_000.0,
    fees: float = 0.001,
    max_position_pct: float = 0.08,
    min_score_proxy: float | None = None,
    alpha_uncertainty_by_date: dict[str, dict[str, float]] | None = None,
    backtest_runner: Callable | None = None,
) -> dict:
    """Sweep the MVO optimizer's PRIMARY params (``risk_aversion`` ×
    ``tcost_bps``) to find the cell that maximizes Sortino on the backtest
    (config#1057). Observe-only — returns a verdict report; no live write.

    Report shape matches ``run_cov_estimator_sweep`` / ``run_gamma_sweep`` so the
    same downstream operator tooling + report section reads all three uniformly.
    Unlike the γ-sweep, no σ_α̂ input is required (these knobs always have
    signal); ``alpha_uncertainty_by_date`` is threaded through when available."""
    if cells is None:
        cells = default_optimizer_param_sweep_cells()
    if not cells:
        raise ValueError("Empty cell list — sweep must have at least one cell")

    runner = backtest_runner if backtest_runner is not None else run_optimizer_backtest
    base_cfg = dict(base_optimizer_cfg or {})

    metrics_by_cell: dict[str, dict] = {}
    failures_by_cell: dict[str, int] = {}
    for name, overrides in cells:
        cfg = {**base_cfg, **overrides}
        logger.info(f"Running optimizer-param sweep cell {name!r}: cfg={cfg}")
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
            alpha_uncertainty_by_date=alpha_uncertainty_by_date,
        )
        metrics_by_cell[name] = dict(result.metrics)
        metrics_by_cell[name]["cell_cfg"] = dict(cfg)
        failures_by_cell[name] = int(result.n_solver_failures)

    baseline_name = cells[0][0]  # baseline (live defaults) by convention
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
