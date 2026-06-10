"""
null_calibration.py — Leg (a) of the backtester correctness battery (ROADMAP L4593).

WHY THIS EXISTS
---------------
The backtester auto-writes four live config files every Saturday
(`scoring_weights` / `executor_params` / `predictor_params` / `research_params`)
that production consumes. vectorbt is the *only* simulation engine — there is no
independent oracle — so a correctness bug in the sim/sweep/significance machinery
would silently tune the live system on fiction. The predictor already carries an
institutional validation battery (CPCV purge/embargo, noise->IC≈0 nulls, planted
signal recovery, DSR/density-floor gates); the backtester had none.

This module is the *leak detector*: it drives the REAL machinery
(``vectorbt_bridge.orders_to_portfolio`` + ``vectorbt_bridge.portfolio_stats`` and
``analysis.monte_carlo.run_monte_carlo``) over synthetic NULL inputs — random-walk
prices with no drift edge, and uninformative random signals — and measures whether
the machinery manufactures alpha or declares significance where there is none.

Two calibration surfaces:

1. **Sim-engine null calibration** (:func:`run_sim_null_calibration`). Random
   orders on zero-edge random-walk prices, run through the production vectorbt
   path. Under the null the realized alpha-vs-SPY and Sharpe distributions must
   CENTER ON ZERO (the engine invents no edge), and per-trial PSR must flag
   "skill" no more often than its nominal false-positive rate (random luck is not
   skill). Transaction costs may only *subtract* — fees can never push mean alpha
   positive.

2. **Significance-gate null calibration** (:func:`run_significance_gate_calibration`).
   Builds independent synthetic ``score_performance`` tables where score and
   forward return are independent by construction, runs the production Monte-Carlo
   permutation gate on each, and measures the empirical false-positive rate. A
   correctly-calibrated gate rejects the null at ≈ its nominal alpha; an inflated
   rate means the gate "finds" alpha in noise.

All randomness is seeded; every result is deterministic for a given seed. The
battery PASSING is the expected outcome — a FAILURE is the battery doing its job
(record it in EXPERIMENTS.md). See [[feedback_sota_institutional_default_no_shortcuts]]
and [[feedback_no_silent_fails]].

Units: alpha/return fields are fractions of initial capital (0.01 = 1%), matching
``vectorbt_bridge.portfolio_stats``. ``run_monte_carlo`` works in percentage units
internally; that is opaque to the calibration (we only read its ``conclusion``).
"""

from __future__ import annotations

import logging
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Defaults — sized so a full calibration runs in seconds, while N is large
# enough that the sampling distribution of the mean is tight (CLT). ──────────
DEFAULT_N_TRIALS = 120
DEFAULT_N_DAYS = 160
DEFAULT_N_TICKERS = 24
DEFAULT_DAILY_VOL = 0.015  # ≈ 24% annualized — representative single-name vol
SPY_TICKER = "SPY"

# PSR nominal one-sided false-positive rate is 0.05 (confidence the true Sharpe
# exceeds 0 at the 95% level). With finite samples + the discreteness of a small
# trial count we allow generous slack before calling the rate "inflated".
PSR_SIGNIFICANT_THRESHOLD = 0.95


# ════════════════════════════════════════════════════════════════════════════
# Synthetic null generators
# ════════════════════════════════════════════════════════════════════════════

def generate_random_walk_prices(
    n_tickers: int,
    n_days: int,
    seed: int,
    *,
    start_price: float = 100.0,
    daily_vol: float = DEFAULT_DAILY_VOL,
    drift: float = 0.0,
    include_spy: bool = True,
) -> pd.DataFrame:
    """Generate a zero-edge price matrix: each column is an independent random
    walk with i.i.d. daily *simple* returns ~ Normal(``drift``, ``daily_vol``).

    With ``drift=0`` the expected daily return of every asset is exactly zero, so
    no asset has a systematic edge over any other (including SPY). Returns a
    DataFrame indexed by a business-day ``DatetimeIndex`` with ticker columns
    (``T0000 … T{n-1}`` plus ``SPY`` when ``include_spy``), values = close prices.
    """
    if n_tickers < 1 or n_days < 2:
        raise ValueError(f"need n_tickers>=1 and n_days>=2, got {n_tickers}, {n_days}")

    rng = np.random.default_rng(seed)
    cols = [f"T{i:04d}" for i in range(n_tickers)]
    if include_spy:
        cols.append(SPY_TICKER)

    # simple daily returns; price path via cumulative product of (1 + r).
    rets = rng.normal(loc=drift, scale=daily_vol, size=(n_days, len(cols)))
    prices = start_price * np.cumprod(1.0 + rets, axis=0)
    # anchor day 0 at start_price (the first return moves off the anchor)
    prices = np.vstack([np.full(len(cols), start_price), prices[:-1]])

    index = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(prices, index=index, columns=cols)


def generate_random_orders(
    prices: pd.DataFrame,
    seed: int,
    *,
    n_entries: int = 40,
    mean_hold_days: int = 15,
    dollar_per_entry: float = 25_000.0,
) -> list[dict]:
    """Generate uninformative ENTER/EXIT orders against ``prices``.

    Entry dates, tickers, and hold durations are all random — the signal carries
    no information about future returns. Each ENTER is paired with an EXIT
    ``~mean_hold_days`` later (clamped to the price window). Order dicts match the
    ``vectorbt_bridge.orders_to_portfolio`` contract
    (``date`` / ``ticker`` / ``action`` / ``shares`` / ``price_at_order``).
    """
    rng = np.random.default_rng(seed)
    dates = prices.index
    tickers = [c for c in prices.columns if c != SPY_TICKER]
    n_days = len(dates)

    orders: list[dict] = []
    # leave room so every entry can be exited within the window
    last_entry_pos = max(1, n_days - 2)
    entry_positions = rng.integers(0, last_entry_pos, size=n_entries)
    chosen = rng.integers(0, len(tickers), size=n_entries)
    holds = np.maximum(1, rng.poisson(mean_hold_days, size=n_entries))

    for pos, tk_idx, hold in zip(entry_positions, chosen, holds):
        ticker = tickers[int(tk_idx)]
        entry_date = dates[int(pos)]
        entry_price = float(prices.iloc[int(pos)][ticker])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue
        shares = float(int(dollar_per_entry // entry_price))
        if shares <= 0:
            continue
        orders.append({
            "date": entry_date.strftime("%Y-%m-%d"),
            "ticker": ticker,
            "action": "ENTER",
            "shares": shares,
            "price_at_order": entry_price,
        })
        exit_pos = min(n_days - 1, int(pos) + int(hold))
        orders.append({
            "date": dates[exit_pos].strftime("%Y-%m-%d"),
            "ticker": ticker,
            "action": "EXIT",
            "shares": shares,
            "price_at_order": float(prices.iloc[exit_pos][ticker]),
        })

    return orders


# ════════════════════════════════════════════════════════════════════════════
# 1. Sim-engine null calibration
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SimNullReport:
    """Aggregate of :func:`run_sim_null_calibration` across trials."""
    n_trials: int
    fees: float
    alphas: np.ndarray = field(repr=False)
    sharpes: np.ndarray = field(repr=False)
    psrs: np.ndarray = field(repr=False)

    @property
    def alpha_mean(self) -> float:
        return float(np.nanmean(self.alphas))

    @property
    def alpha_se(self) -> float:
        n = int(np.sum(np.isfinite(self.alphas)))
        return float(np.nanstd(self.alphas, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")

    @property
    def sharpe_mean(self) -> float:
        return float(np.nanmean(self.sharpes))

    @property
    def sharpe_se(self) -> float:
        n = int(np.sum(np.isfinite(self.sharpes)))
        return float(np.nanstd(self.sharpes, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")

    def zero_in_ci(self, values_mean: float, values_se: float, z: float = 1.96) -> bool:
        """True iff 0 lies within the mean ± z·SE confidence band."""
        if not np.isfinite(values_se) or values_se == 0:
            return abs(values_mean) < 1e-12
        return abs(values_mean) <= z * values_se

    @property
    def alpha_centered_on_zero(self) -> bool:
        return self.zero_in_ci(self.alpha_mean, self.alpha_se)

    @property
    def sharpe_centered_on_zero(self) -> bool:
        return self.zero_in_ci(self.sharpe_mean, self.sharpe_se)

    @property
    def psr_false_positive_rate(self) -> float:
        finite = self.psrs[np.isfinite(self.psrs)]
        if finite.size == 0:
            return float("nan")
        return float(np.mean(finite >= PSR_SIGNIFICANT_THRESHOLD))

    def summary(self) -> dict:
        return {
            "n_trials": self.n_trials,
            "fees": self.fees,
            "alpha_mean": self.alpha_mean,
            "alpha_se": self.alpha_se,
            "alpha_centered_on_zero": self.alpha_centered_on_zero,
            "sharpe_mean": self.sharpe_mean,
            "sharpe_se": self.sharpe_se,
            "sharpe_centered_on_zero": self.sharpe_centered_on_zero,
            "psr_false_positive_rate": self.psr_false_positive_rate,
        }


def run_sim_null_trial(seed: int, *, fees: float, n_tickers: int, n_days: int) -> dict:
    """One null trial: random-walk prices + random orders → real vectorbt path.

    Returns the subset of ``portfolio_stats`` the calibration reads:
    ``total_alpha`` (vs SPY), ``sharpe_ratio``, ``psr``, ``total_return``,
    ``total_trades``.
    """
    # Imported lazily so importing this module never forces a vectorbt import
    # (keeps it usable in contexts where vbt is unavailable).
    from vectorbt_bridge import orders_to_portfolio, portfolio_stats

    prices = generate_random_walk_prices(n_tickers, n_days, seed)
    spy = prices[SPY_TICKER]
    tradeable = prices.drop(columns=[SPY_TICKER])
    orders = generate_random_orders(tradeable, seed)

    pf = orders_to_portfolio(orders, tradeable, fees=fees)
    stats = portfolio_stats(pf, spy_prices=spy)
    return {
        "total_alpha": stats.get("total_alpha"),
        "sharpe_ratio": stats.get("sharpe_ratio"),
        "psr": stats.get("psr"),
        "total_return": stats.get("total_return"),
        "total_trades": stats.get("total_trades"),
    }


def run_sim_null_calibration(
    *,
    n_trials: int = DEFAULT_N_TRIALS,
    seed: int = 20260610,
    fees: float = 0.0,
    n_tickers: int = DEFAULT_N_TICKERS,
    n_days: int = DEFAULT_N_DAYS,
) -> SimNullReport:
    """Run ``n_trials`` independent null trials and aggregate.

    Each trial gets a distinct derived seed so price paths AND orders differ
    across trials while the whole calibration stays reproducible from ``seed``.
    """
    seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=n_trials)
    alphas, sharpes, psrs = [], [], []
    for s in seeds:
        r = run_sim_null_trial(int(s), fees=fees, n_tickers=n_tickers, n_days=n_days)
        a = r["total_alpha"]
        alphas.append(np.nan if a is None else float(a))
        sh = r["sharpe_ratio"]
        sharpes.append(np.nan if sh is None else float(sh))
        p = r["psr"]
        psrs.append(np.nan if p is None else float(p))

    return SimNullReport(
        n_trials=n_trials,
        fees=fees,
        alphas=np.asarray(alphas, dtype=float),
        sharpes=np.asarray(sharpes, dtype=float),
        psrs=np.asarray(psrs, dtype=float),
    )


# ════════════════════════════════════════════════════════════════════════════
# 2. Significance-gate null calibration
# ════════════════════════════════════════════════════════════════════════════

_HORIZONS = ("5d", "10d", "30d")


def build_null_research_db(
    path: str,
    seed: int,
    *,
    n_dates: int = 30,
    n_per_date: int = 18,
    score_mean: float = 60.0,
    score_std: float = 12.0,
    ret_std: float = 4.0,
) -> None:
    """Write a synthetic ``score_performance`` SQLite table where score and
    forward return are INDEPENDENT by construction.

    Schema matches the columns ``analysis.monte_carlo.run_monte_carlo`` reads:
    ``symbol, score_date, score, return_{h}, spy_{h}_return`` for each horizon.
    Returns are in percentage units (matching production). Under this null, any
    selection rule keyed on score has zero expected alpha.
    """
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2025-01-06")
    dates = [(base + pd.tseries.offsets.BDay(7 * i)).strftime("%Y-%m-%d") for i in range(n_dates)]

    rows = []
    sym_i = 0
    for d in dates:
        for _ in range(n_per_date):
            row = {
                "symbol": f"S{sym_i:05d}",
                "score_date": d,
                "score": float(rng.normal(score_mean, score_std)),
            }
            # returns drawn independently of score AND of the SPY leg
            for h in _HORIZONS:
                row[f"return_{h}"] = float(rng.normal(0.0, ret_std))
                row[f"spy_{h}_return"] = float(rng.normal(0.0, ret_std))
            rows.append(row)
            sym_i += 1

    df = pd.DataFrame(rows)
    conn = sqlite3.connect(path)
    try:
        df.to_sql("score_performance", conn, if_exists="replace", index=False)
    finally:
        conn.close()


@dataclass
class GateNullReport:
    """Aggregate of :func:`run_significance_gate_calibration` across datasets."""
    n_datasets: int
    nominal_alpha: float
    p_values: np.ndarray = field(repr=False)
    n_significant: int = 0
    n_evaluated: int = 0

    @property
    def false_positive_rate(self) -> float:
        return self.n_significant / self.n_evaluated if self.n_evaluated else float("nan")

    def summary(self) -> dict:
        return {
            "n_datasets": self.n_datasets,
            "n_evaluated": self.n_evaluated,
            "nominal_alpha": self.nominal_alpha,
            "false_positive_rate": self.false_positive_rate,
            "n_significant": self.n_significant,
        }


def run_significance_gate_calibration(
    *,
    n_datasets: int = 24,
    seed: int = 20260610,
    n_permutations: int = 200,
    horizon: str = "5d",
    min_score: float = 55.0,
    top_n: int = 5,
    nominal_alpha: float = 0.05,
) -> GateNullReport:
    """Run the production Monte-Carlo gate on ``n_datasets`` independent null
    ``score_performance`` tables and measure the empirical false-positive rate.

    A well-calibrated gate declares "significant" on ≈ ``nominal_alpha`` of null
    datasets; an inflated rate means the gate finds alpha in noise.
    """
    from analysis.monte_carlo import run_monte_carlo

    seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=n_datasets)
    p_values: list[float] = []
    n_significant = 0
    n_evaluated = 0

    for i, s in enumerate(seeds):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / f"null_research_{i}.db")
            build_null_research_db(db_path, int(s))
            res = run_monte_carlo(
                db_path,
                n_permutations=n_permutations,
                top_n=top_n,
                min_score=min_score,
                horizon=horizon,
                seed=int(s) % (2**31 - 1),
            )
        if res.get("status") != "ok":
            logger.warning("null gate dataset %d returned status=%s", i, res.get("status"))
            continue
        n_evaluated += 1
        p_values.append(float(res["p_value"]))
        if res.get("conclusion") == "significant":
            n_significant += 1

    return GateNullReport(
        n_datasets=n_datasets,
        nominal_alpha=nominal_alpha,
        p_values=np.asarray(p_values, dtype=float),
        n_significant=n_significant,
        n_evaluated=n_evaluated,
    )


if __name__ == "__main__":  # pragma: no cover — manual calibration run
    logging.basicConfig(level=logging.INFO)
    sim_zero = run_sim_null_calibration(fees=0.0)
    sim_fees = run_sim_null_calibration(fees=0.002)
    gate = run_significance_gate_calibration()
    print("sim (fees=0):   ", sim_zero.summary())
    print("sim (fees=20bp):", sim_fees.summary())
    print("gate:           ", gate.summary())
