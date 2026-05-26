"""
predictive_interval_calibration.py — Gaussian predictive-interval calibration.

Closes the loop on the BayesianRidge cutover (alpha-engine-predictor B.1).
The meta-stacker now emits a posterior std alongside each predicted_alpha;
this module verifies the emitted intervals are well-calibrated against
realized outcomes BEFORE the executor's α̂-uncertainty penalty (alpha-engine
B.3) consumes them — a miscalibrated std propagated unchecked would size
positions on noise.

Why a new module: ``calibration_diagnostics.py`` handles binary-classification
calibration (probability of an event vs realized outcome — Brier score,
ECE, reliability bins on conviction tiers). That primitive answers
"is the agent's 70% conviction calibrated?" — a different question from
"does the predictor's 90% predictive interval cover 90% of realized?"
The former bins probabilities and checks hit rates; the latter computes
PIT z-scores and counts coverage at named confidence levels. Same
calibration umbrella, distinct primitive.

References:
  - Gneiting et al. 2007 "Probabilistic forecasts, calibration and sharpness"
    (JRSS B 69(2)) — foundational PIT calibration framework
  - Gneiting & Raftery 2007 "Strictly proper scoring rules, prediction,
    and estimation" (JASA 102) — Gaussian CRPS closed form
  - Kuleshov, Fenner, Ermon 2018 "Accurate uncertainties for deep learning
    using calibrated regression" — modern coverage-vs-confidence diagnostic

Plan: alpha-engine-docs/private/optimizer-sota-upgrades-260526.md §B.2

Pure-compute. Operates on parallel arrays; no I/O.
"""

from __future__ import annotations

import logging
import math
from typing import TypedDict

import numpy as np

logger = logging.getLogger(__name__)


class IntervalCoverageRecord(TypedDict, total=False):
    confidence: float          # nominal level, e.g. 0.90
    empirical: float           # fraction of realized inside the predicted interval
    deviation: float           # empirical - nominal (negative = overconfident)
    expected_lower_band: float  # nominal - tolerance
    expected_upper_band: float  # nominal + tolerance


class IntervalCalibrationResult(TypedDict, total=False):
    status: str                # "ok" | "insufficient_data" | "no_variance"
    n: int
    coverage: list[IntervalCoverageRecord]
    crps_mean: float           # Gaussian CRPS averaged over points
    pit_mean: float            # mean PIT z-score (≈ 0 when well-calibrated)
    pit_std: float             # std of PIT z-score (≈ 1 when well-calibrated)
    primary_gate_passes: bool  # whether 90% empirical ∈ [0.88, 0.92] per plan
    primary_gate_level: float  # 0.90
    quality: str               # "good" | "acceptable" | "poor"


# Plan §B.2 explicit gate: 90% predicted CI must cover 88-92% of realized.
# A ±2% tolerance corresponds to ≈ 2 standard errors at n=1000 under a
# Wald-binomial approximation: SE = √(0.9 · 0.1 / 1000) ≈ 0.0095 →
# 2·SE ≈ 0.019. So the band is statistically defensible for typical
# synthetic-backtest sizes; for smaller n the gate is conservative and
# may flag false negatives.
_PRIMARY_GATE_LEVEL = 0.90
_PRIMARY_GATE_TOLERANCE = 0.02

_DEFAULT_LEVELS = [0.50, 0.80, 0.90, 0.95, 0.99]
_DEFAULT_MIN_N = 30


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF via math.erf — avoids the scipy dep that the
    backtester repo otherwise doesn't carry."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def _normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)


def _gaussian_crps(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form CRPS for a Gaussian forecast (Gneiting & Raftery 2007).

        CRPS(N(μ, σ²), y) = σ · [ z · (2Φ(z) − 1) + 2φ(z) − 1/√π ]

    where z = (y − μ) / σ, Φ is the standard-normal CDF, φ is its PDF.
    """
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * _normal_cdf(z) - 1.0) + 2.0 * _normal_pdf(z) - 1.0 / math.sqrt(math.pi))


def compute_predictive_interval_calibration(
    predicted_mean: np.ndarray,
    predicted_std: np.ndarray,
    realized: np.ndarray,
    confidence_levels: list[float] | None = None,
    min_n: int = _DEFAULT_MIN_N,
) -> IntervalCalibrationResult:
    """Compute Gaussian predictive-interval calibration diagnostics.

    Parameters
    ----------
    predicted_mean : array-like (N,)
        Posterior mean prediction per observation (predicted_alpha from
        the BayesianRidge meta-stacker).
    predicted_std : array-like (N,)
        Posterior std per observation (predicted_alpha_std from B.1). Must
        be strictly positive — zero or negative std is a meaningless
        interval and an upstream contract violation.
    realized : array-like (N,)
        Realized outcome per observation (canonical_alpha realized over
        the prediction horizon). Same length as predicted_mean.
    confidence_levels : list[float] | None
        Confidence levels to report empirical coverage at. Default
        [0.50, 0.80, 0.90, 0.95, 0.99]. 0.90 is always the primary gate
        per the plan.
    min_n : int
        Minimum valid pairs after NaN filtering. Below floor returns
        status=insufficient_data — coverage at named levels is noisy on
        small N.

    Returns
    -------
    IntervalCalibrationResult dict with:
        status: "ok" | "insufficient_data" | "no_variance"
        n: total valid triples
        coverage: per-level empirical coverage records with deviation
        crps_mean: mean Gaussian CRPS (lower is better)
        pit_mean: mean of (y - μ) / σ — should be ≈ 0 under calibration
        pit_std: std of (y - μ) / σ — should be ≈ 1 under calibration
        primary_gate_passes: empirical_90 ∈ [0.88, 0.92] per plan §B.2
        primary_gate_level: 0.90
        quality: "good" | "acceptable" | "poor"

    Notes
    -----
    - PIT (probability integral transform) z-scores test the *whole*
      distributional fit; coverage at named levels is the operational
      diagnostic. We report both.
    - Negative predicted_std values raise ValueError — caller is
      responsible for upstream invariant. Predictor's BayesianRidge
      always emits std > 0; a negative value is a wiring bug to surface
      loudly per [[feedback_no_silent_fails]].
    - "no_variance" return is for the degenerate case where all
      predicted_std are identical (homoskedastic forecast); coverage
      is still computable but PIT-std loses meaning.
    """
    mu = np.asarray(predicted_mean, dtype=np.float64).ravel()
    sigma = np.asarray(predicted_std, dtype=np.float64).ravel()
    y = np.asarray(realized, dtype=np.float64).ravel()

    if not (mu.size == sigma.size == y.size):
        raise ValueError(
            f"Array length mismatch: mean={mu.size}, std={sigma.size}, "
            f"realized={y.size}"
        )
    # Strict invariant: predicted_std MUST be positive. Zero or negative
    # is an upstream contract violation (BR posterior always > 0). Raise
    # loud per [[feedback_no_silent_fails]].
    finite_sigma = np.isfinite(sigma)
    if np.any(finite_sigma & (sigma <= 0.0)):
        bad = int(np.sum(finite_sigma & (sigma <= 0.0)))
        raise ValueError(
            f"predicted_std has {bad} non-positive entries — must be > 0 "
            "(upstream contract violation; BayesianRidge posterior is "
            "always positive)"
        )

    valid = np.isfinite(mu) & np.isfinite(sigma) & np.isfinite(y) & (sigma > 0.0)
    mu = mu[valid]
    sigma = sigma[valid]
    y = y[valid]
    n = int(mu.size)

    if n < min_n:
        return {"status": "insufficient_data", "n": n}

    levels = confidence_levels if confidence_levels is not None else list(_DEFAULT_LEVELS)
    # Always include the primary gate level so it can be looked up later.
    if _PRIMARY_GATE_LEVEL not in levels:
        levels = sorted(set(levels) | {_PRIMARY_GATE_LEVEL})

    # PIT z-scores: (y - μ) / σ. Under calibration these are N(0, 1).
    pit_z = (y - mu) / sigma
    pit_mean = float(pit_z.mean())
    pit_std = float(pit_z.std(ddof=1)) if n > 1 else float("nan")

    coverage: list[IntervalCoverageRecord] = []
    primary_gate_passes = False
    for level in levels:
        # Symmetric two-sided interval at confidence `level`:
        # μ ± z_{(1+level)/2} · σ. z_quantile from the inverse normal CDF.
        tail = (1.0 + level) / 2.0
        z_q = _normal_quantile(tail)
        within = (np.abs(pit_z) <= z_q)
        empirical = float(within.mean())
        deviation = empirical - level
        record: IntervalCoverageRecord = {
            "confidence": round(level, 4),
            "empirical": round(empirical, 4),
            "deviation": round(deviation, 4),
            "expected_lower_band": round(level - _PRIMARY_GATE_TOLERANCE, 4),
            "expected_upper_band": round(level + _PRIMARY_GATE_TOLERANCE, 4),
        }
        coverage.append(record)
        if level == _PRIMARY_GATE_LEVEL:
            lo = _PRIMARY_GATE_LEVEL - _PRIMARY_GATE_TOLERANCE
            hi = _PRIMARY_GATE_LEVEL + _PRIMARY_GATE_TOLERANCE
            primary_gate_passes = (lo <= empirical <= hi)

    crps = _gaussian_crps(mu, sigma, y)
    crps_mean = float(crps.mean())

    # Quality grade from max |deviation| across reported levels (matches
    # ECE convention in calibration_diagnostics.py — single scalar grade,
    # interpretable across calibration primitives).
    max_dev = max(abs(rec["deviation"]) for rec in coverage)
    if max_dev < 0.03:
        quality = "good"
    elif max_dev < 0.06:
        quality = "acceptable"
    else:
        quality = "poor"

    return {
        "status": "ok",
        "n": n,
        "coverage": coverage,
        "crps_mean": round(crps_mean, 6),
        "pit_mean": round(pit_mean, 4),
        "pit_std": round(pit_std, 4),
        "primary_gate_passes": primary_gate_passes,
        "primary_gate_level": _PRIMARY_GATE_LEVEL,
        "quality": quality,
    }


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF via Beasley-Springer-Moro approximation.

    Avoids the scipy dep otherwise. Accurate to ~7 decimal places — plenty
    for coverage thresholds. Domain p ∈ (0, 1).
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"Quantile probability must be in (0, 1); got {p}")
    # Coefficients from Moro 1995, used in standard Black-Scholes pricers.
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
