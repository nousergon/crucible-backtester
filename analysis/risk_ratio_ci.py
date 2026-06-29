"""
risk_ratio_ci.py — block-bootstrap confidence intervals + magnitude-certainty
monitor for the risk-adjusted ratios (Sharpe / Sortino / Information Ratio).

Director proposal L4558 (config#976, id=widen-sample-before-magnitude-claims):
at small N the *direction* of a risk-adjusted ratio is robust across tiles but
the *point-estimate magnitude* is not — e.g. information_ratio CI
[-8.086, 0.02211] (upper bound touches zero), sharpe CI [-4.112, 3.828] and
sortino CI [-4.643, 7.427] both straddling zero at N=63. The Director's
instruction is to "size remediation by direction, not by the exact -0.36/-3.9
numbers" until the sample widens.

This module is the *no-action monitor* that operationalizes that discipline: it
re-estimates each ratio's sampling distribution by **moving-block bootstrap** of
the daily-return series (blocks preserve autocorrelation, which IID bootstrap
would destroy and thereby understate the CI width), reports a 95% CI per ratio,
and emits a per-ratio ``magnitude_certain`` flag:

  * ``magnitude_certain = False`` when the CI **straddles zero** (the sign — and
    therefore the magnitude — is not yet resolved by the data) OR when N is below
    the floor. Consumers (the report card / Director) should then size any
    remediation by the point-estimate *direction* only, not its magnitude.
  * ``magnitude_certain = True`` when the whole CI lies on one side of zero AND
    N ≥ floor — the sign is resolved and the magnitude can be quoted.

Pure-compute over an already-computed daily-return series (no new data read),
mirroring ``sample_size_adequacy`` / ``regime_stratified_sortino``. Deterministic:
the bootstrap is seeded so the same returns always yield the same CI (report-card
producers must be reproducible cycle-to-cycle). Always-emit (even
``insufficient_data``) so the evaluator can distinguish "producer didn't run"
from "ran, genuinely too few samples".

Closes config#976 when wired into the report card and verified surfacing the CIs.
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252

# Below this many daily observations the bootstrap CI is itself too noisy to
# trust; the Director cited N=63 as the regime where magnitude is uncertain, so
# the floor is set so that ~63 obs lands firmly in the "uncertain" band.
RISK_RATIO_SAMPLE_FLOOR = 126  # ~6 months of trading days

# Moving-block length. ~1 trading week preserves the short-horizon
# autocorrelation of daily portfolio returns without shrinking the number of
# resamplable blocks too far.
_BLOCK_LEN = 5
_N_BOOTSTRAP = 2000
_SEED = 976  # deterministic — same returns ⇒ same CI every cycle (issue id)
_Z = 1.96  # nominal, unused (we use empirical percentiles), kept for reference


def _sharpe(returns: np.ndarray) -> float | None:
    """Annualized Sharpe — mean / std of daily returns, scaled by √252.

    None on < 2 obs or zero volatility (degenerate — no risk to scale by).
    """
    if returns.size < 2:
        return None
    sd = returns.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return None
    return float(returns.mean() / sd * math.sqrt(_TRADING_DAYS_PER_YEAR))


def _sortino(returns: np.ndarray, target: float = 0.0) -> float | None:
    """Annualized Sortino — mean excess / downside deviation, scaled by √252.

    None on < 2 obs, no downside (Sortino undefined), or zero downside dev.
    Mirrors ``analysis.factor_blend_sensitivity._sortino`` (annualized here).
    """
    if returns.size < 2:
        return None
    excess = returns - target
    downside = excess[excess < 0]
    if downside.size == 0:
        return None
    downside_dev = math.sqrt((downside ** 2).mean())
    if downside_dev == 0:
        return None
    return float(excess.mean() / downside_dev * math.sqrt(_TRADING_DAYS_PER_YEAR))


def _information_ratio(active: np.ndarray) -> float | None:
    """Annualized Information Ratio — mean active return / tracking error.

    ``active`` is the per-day (portfolio − benchmark) return series. None on
    < 2 obs or zero tracking error.
    """
    if active.size < 2:
        return None
    te = active.std(ddof=1)
    if te == 0 or not np.isfinite(te):
        return None
    return float(active.mean() / te * math.sqrt(_TRADING_DAYS_PER_YEAR))


def _moving_block_resample(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One moving-block bootstrap resample of ``x`` to (approx) original length.

    Draws ceil(n / block) blocks of length ``_BLOCK_LEN`` from uniformly random
    start indices (blocks may overlap — standard moving-block bootstrap), then
    truncates to exactly ``n`` so every ratio is computed on a same-length draw.
    Preserves within-block autocorrelation that an IID resample would destroy.
    """
    n = x.size
    block = min(_BLOCK_LEN, n)
    n_blocks = math.ceil(n / block)
    max_start = n - block  # inclusive; >= 0 since block <= n
    if max_start <= 0:
        starts = np.zeros(n_blocks, dtype=int)
    else:
        starts = rng.integers(0, max_start + 1, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]).reshape(-1)[:n]
    return x[idx]


def _bootstrap_ci(x: np.ndarray, stat_fn) -> dict:
    """95% percentile CI for ``stat_fn`` over moving-block resamples of ``x``.

    Returns ``{point, ci_95:[lo,hi], straddles_zero, n_valid_resamples}``.
    Point estimate is on the original (un-resampled) series. The CI is the
    [2.5, 97.5] percentiles of the bootstrap distribution, ignoring resamples
    where the statistic is undefined (None). ``straddles_zero`` is True when the
    CI brackets 0 (sign — hence magnitude — unresolved) or could not be formed.
    """
    point = stat_fn(x)
    rng = np.random.default_rng(_SEED)
    draws: list[float] = []
    for _ in range(_N_BOOTSTRAP):
        val = stat_fn(_moving_block_resample(x, rng))
        if val is not None and np.isfinite(val):
            draws.append(val)
    if len(draws) < 2:
        return {
            "point": point,
            "ci_95": None,
            "straddles_zero": True,
            "n_valid_resamples": len(draws),
        }
    lo, hi = (float(v) for v in np.percentile(draws, [2.5, 97.5]))
    return {
        "point": round(point, 4) if point is not None else None,
        "ci_95": [round(lo, 4), round(hi, 4)],
        "straddles_zero": bool(lo <= 0.0 <= hi),
        "n_valid_resamples": len(draws),
    }


def compute_risk_ratio_ci(
    portfolio_daily_returns,
    spy_daily_returns=None,
) -> dict:
    """Block-bootstrap 95% CIs + magnitude-certainty flags for Sharpe / Sortino
    / Information Ratio (config#976 — the no-action magnitude-uncertainty monitor).

    Args:
        portfolio_daily_returns: per-day simple returns of the deployed strategy
            (the ``pf_returns_aligned`` series emitted by
            ``portfolio_optimizer_backtest._simulate_and_measure``). Accepts a
            pandas Series, list, or 1-D array.
        spy_daily_returns: optional aligned benchmark daily returns. When given,
            the Information Ratio is computed on the active (portfolio − SPY)
            series; when absent, IR is reported as N/A.

    Returns a dict::

        {
          "status": "ok" | "insufficient_data",
          "n_samples": int,
          "sample_floor": RISK_RATIO_SAMPLE_FLOOR,
          "n_adequate": bool,                 # N >= floor
          "ratios": {
             "sharpe_ratio":      {point, ci_95, straddles_zero, magnitude_certain, ...},
             "sortino_ratio":     {...},
             "information_ratio": {... or status: "no_benchmark"},
          },
          "all_magnitude_certain": bool,
          "note": "...",                       # direction-vs-magnitude guidance
        }

    ``magnitude_certain`` is True only when the ratio's CI is entirely one side of
    zero AND N >= floor. Otherwise size remediation by the point-estimate
    *direction*, not its magnitude (Director L4558).
    """
    pr = _as_1d(portfolio_daily_returns)
    n = int(pr.size)
    sample_floor = RISK_RATIO_SAMPLE_FLOOR
    n_adequate = n >= sample_floor

    if n < 2:
        return {
            "status": "insufficient_data",
            "n_samples": n,
            "sample_floor": sample_floor,
            "n_adequate": False,
            "ratios": {},
            "all_magnitude_certain": False,
            "note": "Fewer than 2 aligned daily returns — no ratio estimable.",
        }

    ratios: dict[str, dict] = {}
    for name, x in (("sharpe_ratio", pr), ("sortino_ratio", pr)):
        res = _bootstrap_ci(x, _sharpe if name == "sharpe_ratio" else _sortino)
        res["magnitude_certain"] = bool(
            n_adequate and res["ci_95"] is not None and not res["straddles_zero"]
        )
        ratios[name] = res

    sr = _as_1d(spy_daily_returns) if spy_daily_returns is not None else None
    if sr is not None and sr.size == n and n >= 2:
        active = pr - sr
        res = _bootstrap_ci(active, _information_ratio)
        res["magnitude_certain"] = bool(
            n_adequate and res["ci_95"] is not None and not res["straddles_zero"]
        )
        ratios["information_ratio"] = res
    else:
        ratios["information_ratio"] = {
            "status": "no_benchmark",
            "magnitude_certain": False,
            "note": "Aligned SPY daily returns absent — Information Ratio not computed.",
        }

    all_certain = all(
        r.get("magnitude_certain", False)
        for r in ratios.values()
        if r.get("status") != "no_benchmark"
    )

    return {
        "status": "ok",
        "n_samples": n,
        "sample_floor": sample_floor,
        "n_adequate": n_adequate,
        "ratios": ratios,
        "all_magnitude_certain": bool(all_certain),
        "note": (
            "Director L4558: when a ratio's CI straddles zero (magnitude_certain="
            "False), size remediation by the point-estimate DIRECTION only, not "
            "its magnitude."
        ),
    }


def _as_1d(series) -> np.ndarray:
    """Coerce a pandas Series / list / array to a finite 1-D float array."""
    arr = np.asarray(getattr(series, "values", series), dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]
