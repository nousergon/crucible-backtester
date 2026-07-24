"""
alpha_decay.py — Alpha decay curve analysis (config#1981, narrowed scope).

Reads ``score_performance_outcomes`` from ``research.db`` via
:func:`analysis.outcome_store.load_outcomes` and computes per-horizon accuracy
/ alpha metrics to characterise how signal performance decays as the holding
horizon lengthens.

Uses a custom :class:`~nousergon_lib.quant.horizons.HorizonPolicy` that extends
beyond the fleet ``DEFAULT_POLICY`` (5d diagnostic, 21d primary) to include
intermediate horizons (1d, 3d, 10d, 15d). These were added to the producer-side
pipeline by nousergon-data#963 and materialise in
``score_performance_outcomes`` after the next ``weekly_collector`` backfill
cycle completes.

Data availability: meaningful results require at least ``MIN_SAMPLES`` resolved
rows at the primary horizon (21d). A subset of intermediate horizons may be
unpopulated — those entries are reported with ``n = 0`` rather than failing the
entire report.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from nousergon_lib.quant.horizons import HorizonPolicy

from analysis.outcome_store import load_outcomes

logger = logging.getLogger(__name__)

# ── Policy ──────────────────────────────────────────────────────────────────────

# Minimum resolved rows at the primary horizon before reporting — avoids
# misleading metrics on tiny samples, mirroring signal_quality.py's convention.
MIN_SAMPLES = 30

# Custom policy for the decay curve: adds intermediate horizons (1d, 3d, 10d,
# 15d) beyond the fleet DEFAULT_POLICY (5d diagnostic, 21d primary). The
# producer-side change (nousergon-data#963) added these to universe_returns and
# wired them into signal_returns.collect(); they populate
# score_performance_outcomes after the weekly_collector backfill runs.
_DECAY_CURVE_POLICY = HorizonPolicy(
    primary_horizon=21,
    diagnostic_horizons=(1, 3, 5, 10, 15),
    label="decay_curve",
)
_ALL_H = _DECAY_CURVE_POLICY.all_horizons  # (21, 1, 3, 5, 10, 15)


# ── Data loading ────────────────────────────────────────────────────────────────


def load_alpha_decay_data(db_path: str) -> pd.DataFrame:
    """Load long-format outcome rows for all decay-curve horizons.

    Returns a DataFrame with columns ``signal_id``, ``symbol``, ``score_date``,
    ``horizon_days``, ``beat_spy``, ``stock_return``, ``spy_return``,
    ``log_alpha``, ``is_primary``, ``resolved_at`` — one row per signal ×
    horizon. Unpopulated horizons produce no rows (graceful-empty).
    """
    path = Path(db_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"research.db not found at {path}")

    return load_outcomes(str(path), policy=_DECAY_CURVE_POLICY)


# ── Computation ─────────────────────────────────────────────────────────────────


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.
    Mirror of ``signal_quality._wilson_ci`` — kept separate to avoid a cross-module
    dependency on that module's internal constant layout.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (round(max(0.0, centre - spread), 4), round(min(1.0, centre + spread), 4))


def compute_alpha_decay(
    long_df: pd.DataFrame,
    min_samples: int = MIN_SAMPLES,
) -> dict:
    """Compute the alpha decay curve from long-format outcome data.

    Parameters
    ----------
    long_df:
        Long-format outcome DataFrame from :func:`load_alpha_decay_data` (or a
        synthetic frame with the same schema in tests).
    min_samples:
        Minimum resolved primary-horizon rows needed before reporting.

    Returns
    -------
    dict with keys:
        - ``status``: ``"ok"`` | ``"insufficient_data"``
        - ``n_primary``: resolved primary-horizon row count
        - ``rows_needed``: ``min_samples`` (present when status insufficient)
        - ``decay_curve``: list of per-horizon metric dicts, sorted by horizon_days
        - ``decay_rate``: float (linear slope of accuracy vs log-horizon) or
          ``None`` when fewer than 2 horizons have data
        - ``n_total``: total resolved outcome rows across all horizons
    """
    if long_df is None or long_df.empty:
        return {
            "status": "insufficient_data",
            "n_primary": 0,
            "rows_needed": min_samples,
        }

    # Separate primary and diagnostic cohorts — the gating is on the primary
    # horizon (21d), mirroring signal_quality.py's gate-on-canonical contract.
    primary_mask = long_df["is_primary"] == 1
    primary_resolved = long_df[primary_mask & long_df["beat_spy"].notna()]
    n_primary = len(primary_resolved)

    if n_primary < min_samples:
        logger.warning(
            "Only %d resolved primary-horizon rows (need %d)",
            n_primary, min_samples,
        )
        return {
            "status": "insufficient_data",
            "n_primary": n_primary,
            "rows_needed": min_samples,
        }

    # Per-horizon metrics — group by horizon_days and compute aggregates.
    horizon_groups = long_df[long_df["beat_spy"].notna()].groupby("horizon_days")
    curve = []
    for h in sorted(_ALL_H):
        group = horizon_groups.get_group(h) if h in horizon_groups.groups else pd.DataFrame()
        if group.empty:
            curve.append({
                "horizon_days": h,
                "n": 0,
                "accuracy": None,
                "avg_log_alpha": None,
                "avg_excess_return": None,
                "ci_95": None,
            })
            continue

        n = len(group)
        successes = int(group["beat_spy"].sum())
        accuracy = round(successes / n, 4) if n > 0 else None
        ci = _wilson_ci(successes, n)

        # Log-domain alpha — the canonical alpha measure in the store.
        log_alpha_col = "log_alpha"
        avg_log_alpha = (
            float(group[log_alpha_col].mean())
            if log_alpha_col in group.columns and group[log_alpha_col].notna().any()
            else None
        )

        # Excess return (stock_return - spy_return) — mean excess per horizon.
        avg_excess = float(
            (group["stock_return"] - group["spy_return"]).mean()
        ) if "stock_return" in group.columns and "spy_return" in group.columns else None

        curve.append({
            "horizon_days": h,
            "n": n,
            "accuracy": accuracy,
            "avg_log_alpha": avg_log_alpha,
            "avg_excess_return": avg_excess,
            "ci_95": ci,
        })

    # Decay rate: linear slope of accuracy vs log(horizon_days). A negative
    # slope means accuracy declines as the horizon lengthens — the canonical
    # "alpha decay" signal. Requires at least 2 populated points.
    decay_rate = None
    populated = [p for p in curve if p["n"] > 0 and p["accuracy"] is not None]
    if len(populated) >= 2:
        xs = np.log([p["horizon_days"] for p in populated])
        ys = [p["accuracy"] for p in populated]
        # Simple OLS slope: cov(x, y) / var(x)
        x_mean = np.mean(xs)
        y_mean = np.mean(ys)
        cov = np.sum((xs - x_mean) * (ys - y_mean))
        var_x = np.sum((xs - x_mean) ** 2)
        decay_rate = round(float(cov / var_x), 6) if var_x > 0 else None

    n_total = int(long_df["beat_spy"].notna().sum())

    return {
        "status": "ok",
        "n_primary": n_primary,
        "decay_curve": curve,
        "decay_rate": decay_rate,
        "n_total": n_total,
    }
