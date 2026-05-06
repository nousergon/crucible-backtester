"""
information_coefficient.py — Spearman rank correlation between agent
conviction and forward returns.

The risk-invariant quality metric. IC of 0.05 on conservative picks vs
0.02 on aggressive picks tells you the agent is *worse* at the harder
task even if absolute returns went up — exactly the diagnostic we need
when pushing agents toward more risk (per evaluator-revamp-260506.md).

Why Spearman over Pearson: rank correlation is invariant to the scale
and distribution of conviction scores. The agent's "conviction = 80"
and "conviction = 60" only need to *order* picks correctly; absolute
calibration is a separate concern (handled by calibration_diagnostics).

Pure-compute. Operates on parallel arrays of (conviction, return);
no I/O.
"""

from __future__ import annotations

import logging
from typing import TypedDict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ICResult(TypedDict, total=False):
    status: str
    n: int
    ic: float            # Spearman rank correlation
    p_value: float       # two-sided p-value vs null hypothesis IC = 0
    n_buckets: int       # number of distinct conviction levels


def compute_ic(
    conviction: pd.Series | np.ndarray,
    forward_return: pd.Series | np.ndarray,
    min_samples: int = 20,
) -> ICResult:
    """Compute Spearman rank correlation between conviction + forward return.

    Parameters
    ----------
    conviction : array-like
        Stated conviction or composite score per pick. Higher = more
        confident. Numeric or ordinal (string ranks need to be encoded
        upstream — Spearman is on the rank, but the array must sort
        meaningfully).
    forward_return : array-like
        Realized forward return per pick over the prediction horizon.
        Same length as ``conviction``. NaN in either column → that pair
        is dropped before computing.
    min_samples : int
        Minimum valid pairs required. Below this floor returns
        status=insufficient_data. Default 20 (Spearman p-values are
        unreliable on small samples).

    Returns
    -------
    ICResult dict with:
        status: "ok" | "insufficient_data" | "no_variance"
        n: number of (conviction, return) pairs after NaN filtering
        ic: Spearman rank correlation in [-1, 1]
        p_value: two-sided p-value
        n_buckets: number of distinct conviction levels (low = collapsed
                   conviction signal — the agent isn't differentiating
                   even if IC happens to look fine)

    Notes
    -----
    - Uses scipy.stats.spearmanr if available; falls back to numpy
      rank-then-pearson which gives identical IC but no p-value.
    - n_buckets is reported separately so callers can flag the
      degenerate case where the agent emits conviction = constant
      (IC undefined; correlation of constant with anything is 0).
    """
    c = np.asarray(conviction, dtype=np.float64)
    r = np.asarray(forward_return, dtype=np.float64)
    if c.size != r.size:
        raise ValueError(
            f"conviction (n={c.size}) and forward_return (n={r.size}) "
            "must be same length"
        )
    valid = np.isfinite(c) & np.isfinite(r)
    c = c[valid]
    r = r[valid]
    n = c.size
    if n < min_samples:
        return {"status": "insufficient_data", "n": n}

    n_buckets = int(np.unique(c).size)
    # min == max is the exact constancy test (avoids float64 std residual).
    c_const = c.size > 0 and c.min() == c.max()
    r_const = r.size > 0 and r.min() == r.max()
    if n_buckets < 2 or c_const or r_const:
        return {
            "status": "no_variance",
            "n": n,
            "n_buckets": n_buckets,
            "ic": 0.0,
            "p_value": 1.0,
        }

    try:
        from scipy.stats import spearmanr  # type: ignore[import-not-found]

        result = spearmanr(c, r)
        ic = float(result.statistic)
        p_value = float(result.pvalue)
    except Exception:
        # Fallback: rank-then-pearson. Identical IC value; no p-value.
        c_rank = pd.Series(c).rank().to_numpy()
        r_rank = pd.Series(r).rank().to_numpy()
        ic = float(np.corrcoef(c_rank, r_rank)[0, 1])
        p_value = float("nan")

    return {
        "status": "ok",
        "n": n,
        "n_buckets": n_buckets,
        "ic": ic,
        "p_value": p_value,
    }


def compute_ic_by_bucket(
    df: pd.DataFrame,
    conviction_col: str,
    return_col: str,
    bucket_col: str,
    min_samples: int = 20,
) -> dict[str, ICResult]:
    """IC stratified by a bucket column (sector, conviction tier, regime, ...).

    The "is the agent good at the harder task" cut: split by conviction
    decile or sector and compute IC within each. An agent whose IC drops
    on its highest-conviction picks is failing exactly where it claims
    to be most confident.
    """
    for col in (conviction_col, return_col, bucket_col):
        if col not in df.columns:
            raise KeyError(f"column {col!r} not in dataframe")
    out: dict[str, ICResult] = {}
    for bucket_value, sub in df.groupby(bucket_col):
        out[str(bucket_value)] = compute_ic(
            sub[conviction_col], sub[return_col], min_samples=min_samples,
        )
    return out
