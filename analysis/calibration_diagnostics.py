"""
calibration_diagnostics.py — generic conviction-vs-outcome calibration.

Catches the "decision quality vs outcome quality" failure mode (per
evaluator-revamp-260506.md): if the agent says "70% conviction" and
those picks actually win 50% of the time, the agent is miscalibrated —
even if its overall hit rate looks fine. Reliability diagrams + Brier
score + Expected Calibration Error (ECE) make this visible.

Why a new module: the existing
``analysis/production_health.compute_calibration_validation`` is bound
to the ``predictor_outcomes`` SQLite schema (predictor's per-pick
confidence on UP/DOWN classifications). The evaluator-revamp grading
needs a *generic* calibrator that takes any (predicted_probability,
realized_outcome) arrays — research conviction, sector-team picks,
CIO advance/reject decisions, etc. This module provides that primitive;
production_health can later delegate to it.

Pure-compute. Operates on parallel arrays; no I/O.

The headline ECE scalar is **not** computed locally — it delegates to
``nousergon_lib.quant.stats.calibration.expected_calibration_error``, the
one canonical ECE implementation for the fleet (per the lift-to-chokepoint
rule). A train-time ECE (predictor) and a production-time ECE (backtester) are
only comparable if both bin the *same quantity the same way*; keeping a second
ECE loop here would let the two drift. This module still owns the reliability
diagram records and the Brier score (the lib primitive doesn't compute Brier),
but the |hit_rate - expected| sum that defines ECE comes from the lib.
"""

from __future__ import annotations

import logging
from typing import TypedDict

import numpy as np
import pandas as pd
from nousergon_lib.quant.stats.calibration import expected_calibration_error

logger = logging.getLogger(__name__)


class ReliabilityBin(TypedDict, total=False):
    range: tuple[float, float]
    n: int
    hit_rate: float
    expected: float        # mean predicted probability in this bin
    gap: float             # hit_rate - expected (negative = overconfident)


class CalibrationResult(TypedDict, total=False):
    status: str
    n: int
    bins: list[ReliabilityBin]
    dropped_bins: list[ReliabilityBin]
    ece: float             # weighted mean |gap|
    brier_score: float     # mean squared error of probability vs outcome
    quality: str           # "good" | "acceptable" | "poor"


_DEFAULT_BIN_EDGES = [0.0, 0.20, 0.40, 0.60, 0.80, 1.01]
_DEFAULT_MIN_BIN_N = 10


def compute_calibration(
    predicted_probability: pd.Series | np.ndarray,
    realized_outcome: pd.Series | np.ndarray,
    bin_edges: list[float] | None = None,
    min_bin_n: int = _DEFAULT_MIN_BIN_N,
    min_total_samples: int = 30,
) -> CalibrationResult:
    """Compute reliability diagram + Brier score + ECE.

    Parameters
    ----------
    predicted_probability : array-like
        Probability or normalized conviction in [0, 1] per pick.
    realized_outcome : array-like
        Binary outcome per pick: 1 = the predicted event happened
        (e.g. beat SPY at 10d), 0 = it didn't. Same length as
        ``predicted_probability``. NaN in either drops that pair.
    bin_edges : list[float] | None
        Bin edges for the reliability diagram. Default = quintiles
        ``[0.0, 0.2, 0.4, 0.6, 0.8, 1.01]`` matching common conviction
        bucketization. Pass custom edges to align with the upstream
        conviction tier scheme.
    min_bin_n : int
        Bins below this sample size are excluded from ECE (noise
        dominates sparse-tail bins) but reported in ``dropped_bins``.
    min_total_samples : int
        Minimum total valid pairs required. Below floor returns
        status=insufficient_data.

    Returns
    -------
    CalibrationResult dict with:
        status: "ok" | "insufficient_data" | "no_variance"
        n: total valid pairs
        bins: list of reliability records (one per bin meeting min_bin_n)
        dropped_bins: bins below min_bin_n (for diagnostic visibility)
        ece: weighted mean of |hit_rate - expected| over kept bins
        brier_score: mean((p - y)^2) — single-shot scalar quality
        quality: "good" (ECE < 0.05) | "acceptable" (< 0.10) | "poor" (>= 0.10)

    Notes
    -----
    - ``expected`` is the mean predicted probability in each bin
      (rigorous ECE), not the bin midpoint. Bin midpoints
      systematically over- or under-state miscalibration when
      predictions cluster at one end of a bin.
    - Brier score is decomposable into reliability, resolution, and
      uncertainty components; we report the aggregate scalar here. The
      grading layer reads ECE for the quality grade (matches the
      production_health convention).
    """
    p = np.asarray(predicted_probability, dtype=np.float64)
    y = np.asarray(realized_outcome, dtype=np.float64)
    if p.size != y.size:
        raise ValueError(
            f"predicted_probability (n={p.size}) and realized_outcome "
            f"(n={y.size}) must be same length"
        )
    valid = np.isfinite(p) & np.isfinite(y)
    p = p[valid]
    y = y[valid]
    n = p.size
    if n < min_total_samples:
        return {"status": "insufficient_data", "n": n}

    # Use min == max for constant-array detection — exact comparison vs
    # std() can return a tiny non-zero residual on float64 arrays of
    # identical values (mean subtraction not bitwise-exact).
    if p.size > 0 and p.min() == p.max():
        return {
            "status": "no_variance",
            "n": n,
            "bins": [],
            "dropped_bins": [],
            "ece": 0.0,
            "brier_score": float(((p - y) ** 2).mean()),
            "quality": "unknown",
        }

    edges = bin_edges if bin_edges is not None else list(_DEFAULT_BIN_EDGES)

    # Headline ECE comes from the ONE fleet-wide primitive — do not re-derive
    # the |hit_rate - expected| sum here. Same edges + min_bin_n as the local
    # reliability diagram below, so the bins shown match the bins the lib summed.
    # NaN-filtering already happened above, so min_samples=1 here just guards the
    # empty case (the min_total_samples floor is enforced earlier).
    lib_result = expected_calibration_error(
        p, y, bin_edges=edges, min_bin_n=min_bin_n, min_samples=1,
    )

    bins: list[ReliabilityBin] = []
    dropped: list[ReliabilityBin] = []

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi)
        bin_n = int(mask.sum())
        if bin_n == 0:
            continue
        hit_rate = float(y[mask].mean())
        expected = float(p[mask].mean())
        gap = hit_rate - expected
        record: ReliabilityBin = {
            "range": (round(lo, 3), round(min(hi, 1.0), 3)),
            "n": bin_n,
            "hit_rate": round(hit_rate, 4),
            "expected": round(expected, 4),
            "gap": round(gap, 4),
        }
        if bin_n < min_bin_n:
            dropped.append(record)
            continue
        bins.append(record)

    # ``ece`` is None from the lib only when every bin was dropped by min_bin_n;
    # preserve the prior 0.0-when-no-kept-bins contract for the grading layer.
    ece = lib_result.get("ece")
    ece = ece if ece is not None else 0.0
    brier = float(((p - y) ** 2).mean())

    if ece < 0.05:
        quality = "good"
    elif ece < 0.10:
        quality = "acceptable"
    else:
        quality = "poor"

    return {
        "status": "ok",
        "n": n,
        "bins": bins,
        "dropped_bins": dropped,
        "ece": round(ece, 4),
        "brier_score": round(brier, 4),
        "quality": quality,
    }
