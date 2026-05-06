"""
dsr.py — Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR).

Confidence-adjusted Sharpe per López de Prado:
  - PSR (Bailey & López de Prado 2012): probability that the *true* Sharpe
    is above a benchmark, given the observed sample size + skew + kurtosis.
    Answers "is this Sharpe distinguishable from the benchmark, given how
    little data we have?"
  - DSR (Bailey & López de Prado 2014): PSR with a multiple-testing
    correction. The benchmark is set to the expected maximum Sharpe under
    N independent trials, so DSR > 0.95 means "even after accounting for
    cherry-picking from N candidates, this Sharpe is significant."

Why this matters here: the auto-applying optimizers (`weight_optimizer`,
`executor_optimizer`, `veto_optimizer`) run weekly param sweeps and
auto-promote the top-Sharpe combo to S3 configs. That's a multiple-testing
factory. With ~6 months of paper trading, point-estimate Sharpe has a CI
of ±~1.5. DSR is the gate that prevents promoting noise winners.

Mathematical reference:
  Bailey & López de Prado (2012) "The Sharpe Ratio Efficient Frontier"
  Bailey & López de Prado (2014) "The Deflated Sharpe Ratio: Correcting
  for Selection Bias, Backtest Overfitting, and Non-Normality"

Pure-compute. Operates on a daily return series + sample-size metadata;
no I/O.
"""

from __future__ import annotations

import logging
import math
from typing import TypedDict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252


class PSRResult(TypedDict, total=False):
    status: str
    n: int
    sharpe: float           # observed annualized Sharpe
    sharpe_benchmark: float # benchmark Sharpe being tested against
    psr: float              # probability in [0, 1] that true SR > benchmark
    skew: float
    kurtosis: float


class DSRResult(TypedDict, total=False):
    status: str
    n: int
    sharpe: float
    n_trials: int           # number of candidates considered (multiple-testing N)
    sharpe_benchmark: float # implied benchmark from N_trials under H0: SR=0
    dsr: float              # probability that the true Sharpe survives selection bias
    skew: float
    kurtosis: float


def _normal_cdf(x: float) -> float:
    """Standard normal CDF — pure-Python, no scipy dependency."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _annualized_sharpe(returns: np.ndarray) -> float:
    """Annualized Sharpe (risk-free = 0), sample-std (ddof=1)."""
    if returns.size < 2:
        return 0.0
    mean = float(returns.mean())
    std = float(returns.std(ddof=1))
    if std == 0.0:
        return 0.0
    return mean / std * math.sqrt(_TRADING_DAYS_PER_YEAR)


def _sample_skew_kurtosis(returns: np.ndarray) -> tuple[float, float]:
    """Sample skewness and excess kurtosis. Pearson-style; scipy-equivalent.

    Excess kurtosis = K - 3 (so a normal has 0 excess kurtosis).
    Returns (0, 0) on insufficient sample.
    """
    n = returns.size
    if n < 4:
        return 0.0, 0.0
    mean = returns.mean()
    centered = returns - mean
    var = float((centered * centered).mean())
    if var == 0.0:
        return 0.0, 0.0
    std = math.sqrt(var)
    skew = float((centered ** 3).mean() / (std ** 3))
    kurt_excess = float((centered ** 4).mean() / (var * var)) - 3.0
    return skew, kurt_excess


def compute_psr(
    daily_returns: pd.Series | np.ndarray,
    sharpe_benchmark: float = 0.0,
) -> PSRResult:
    """Probabilistic Sharpe Ratio.

    Parameters
    ----------
    daily_returns : array-like
        Daily simple returns. NaN dropped.
    sharpe_benchmark : float
        Annualized Sharpe to test against (default 0.0, i.e. "is the
        true SR positive?").

    Returns
    -------
    PSRResult dict with:
        status: "ok" | "insufficient_data"
        n: sample size
        sharpe: observed annualized SR
        sharpe_benchmark: as input
        psr: probability that true SR > benchmark
        skew, kurtosis: moments of the return series

    Formula (Bailey & López de Prado 2012):
        PSR(SR*) = Phi(  (SR_hat - SR*) * sqrt(n - 1)
                        / sqrt(1 - skew * SR_hat + (kurtosis - 1)/4 * SR_hat^2) )

    where SR_hat is the *non-annualized* observed Sharpe and SR* is the
    benchmark on the same scale. We compute on daily Sharpe internally
    and convert benchmarks accordingly.
    """
    r = np.asarray(daily_returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 30:  # PSR is asymptotic; small samples produce nonsense
        return {"status": "insufficient_data", "n": n}

    sr_annualized = _annualized_sharpe(r)
    # PSR formula uses the daily SR. Convert annualized benchmark back to daily.
    sr_daily = sr_annualized / math.sqrt(_TRADING_DAYS_PER_YEAR)
    sr_bench_daily = sharpe_benchmark / math.sqrt(_TRADING_DAYS_PER_YEAR)

    skew, kurt_excess = _sample_skew_kurtosis(r)
    # The "kurtosis" term in López de Prado's formula is the raw 4th
    # moment / variance^2 (so 3.0 for a normal); we have excess kurtosis.
    kurt_raw = kurt_excess + 3.0

    denom_sq = 1.0 - skew * sr_daily + (kurt_raw - 1.0) / 4.0 * sr_daily ** 2
    if denom_sq <= 0.0:
        # Pathological skew/kurtosis combo; PSR formula breaks down.
        return {
            "status": "ok",
            "n": n,
            "sharpe": sr_annualized,
            "sharpe_benchmark": sharpe_benchmark,
            "psr": 0.5,  # max-uncertainty fallback
            "skew": skew,
            "kurtosis": kurt_excess,
        }
    z = (sr_daily - sr_bench_daily) * math.sqrt(n - 1) / math.sqrt(denom_sq)
    psr = _normal_cdf(z)

    return {
        "status": "ok",
        "n": n,
        "sharpe": sr_annualized,
        "sharpe_benchmark": sharpe_benchmark,
        "psr": float(psr),
        "skew": skew,
        "kurtosis": kurt_excess,
    }


_EULER_MASCHERONI = 0.5772156649015329


def compute_dsr(
    daily_returns: pd.Series | np.ndarray,
    n_trials: int,
) -> DSRResult:
    """Deflated Sharpe Ratio.

    Corrects PSR for the selection bias of choosing the maximum Sharpe
    from ``n_trials`` candidates. The benchmark Sharpe is set to the
    expected maximum SR under the null hypothesis (true SR = 0 for all
    candidates), accounting for sample size + sample moments.

    Parameters
    ----------
    daily_returns : array-like
        Daily returns of the *winner* (the candidate selected as best).
    n_trials : int
        Number of candidates considered when selecting this winner. For
        a 60-combo param sweep, n_trials = 60. Must be >= 1.

    Returns
    -------
    DSRResult dict with:
        status, n, sharpe, n_trials, sharpe_benchmark, dsr, skew, kurtosis

    Formula (Bailey & López de Prado 2014, Theorem 1):
        E[max(SR)] ≈ V * (sqrt(2 ln N) - (gamma + ln ln N) / (2 sqrt(2 ln N)))
    where V is the standard deviation of estimated SRs across trials and
    gamma is Euler-Mascheroni. We approximate V with the sampling std of
    SR_hat = sqrt((1 - skew*SR + (k-1)/4 * SR^2) / (n - 1)) on the winner.

    DSR = PSR(SR_hat | benchmark = E[max(SR_null)]).

    Notes
    -----
    - n_trials = 1 reduces to PSR(0) — no selection correction needed.
    - For very high n_trials (>1000) the asymptotic expansion above is
      adequate; for small n (< 5) it overstates the threshold slightly,
      which is the conservative direction (harder to clear) — fine for
      a promotion gate.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    r = np.asarray(daily_returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 30:
        return {"status": "insufficient_data", "n": n, "n_trials": n_trials}

    if n_trials == 1:
        # No selection bias correction needed; reduce to PSR(0).
        psr_result = compute_psr(r, sharpe_benchmark=0.0)
        return {
            "status": psr_result["status"],
            "n": n,
            "sharpe": psr_result.get("sharpe", 0.0),
            "n_trials": 1,
            "sharpe_benchmark": 0.0,
            "dsr": psr_result.get("psr", 0.5),
            "skew": psr_result.get("skew", 0.0),
            "kurtosis": psr_result.get("kurtosis", 0.0),
        }

    sr_annualized = _annualized_sharpe(r)
    sr_daily = sr_annualized / math.sqrt(_TRADING_DAYS_PER_YEAR)
    skew, kurt_excess = _sample_skew_kurtosis(r)
    kurt_raw = kurt_excess + 3.0

    # Sampling std of SR_hat (per López de Prado eq. 5).
    var_sr_sq = (1.0 - skew * sr_daily + (kurt_raw - 1.0) / 4.0 * sr_daily ** 2) / (n - 1)
    if var_sr_sq <= 0.0:
        return {
            "status": "ok",
            "n": n,
            "sharpe": sr_annualized,
            "n_trials": n_trials,
            "sharpe_benchmark": 0.0,
            "dsr": 0.5,
            "skew": skew,
            "kurtosis": kurt_excess,
        }
    v = math.sqrt(var_sr_sq)

    # Expected max SR under the null, in daily SR units.
    ln_n = math.log(n_trials)
    sqrt_2_ln_n = math.sqrt(2.0 * ln_n)
    if n_trials > 1:
        ln_ln_n = math.log(ln_n) if ln_n > 0 else 0.0
    else:
        ln_ln_n = 0.0
    expected_max_sr_daily = v * (sqrt_2_ln_n - (_EULER_MASCHERONI + ln_ln_n) / (2.0 * sqrt_2_ln_n))
    expected_max_sr_annualized = expected_max_sr_daily * math.sqrt(_TRADING_DAYS_PER_YEAR)

    psr_result = compute_psr(r, sharpe_benchmark=expected_max_sr_annualized)

    return {
        "status": psr_result["status"],
        "n": n,
        "sharpe": sr_annualized,
        "n_trials": n_trials,
        "sharpe_benchmark": expected_max_sr_annualized,
        "dsr": psr_result.get("psr", 0.5),
        "skew": skew,
        "kurtosis": kurt_excess,
    }
