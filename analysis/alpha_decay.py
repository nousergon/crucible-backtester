"""alpha_decay.py — alpha decay curve after signal entry (config#1981).

Reads the score_performance_outcomes store (EPIC config#1483) using a
HorizonPolicy that spans all ladder horizons produced by the decay-curve
collector in nousergon-data (1d, 3d, 5d, 10d, 15d + primary 21d) and
computes the average alpha (stock_return - spy_return) per horizon, both
overall and stratified by score bucket.

WHY THIS EXISTS
---------------
The re-evaluation-cadence question: does signal alpha peak early (1–5d)
and fade by 21d, or continue to accumulate? A decay curve that approaches
zero before the canonical 21d primary horizon suggests the holding period
or exit rule should be re-evaluated.  A curve that continues to rise
suggests the signal has durable alpha and the long hold is warranted.

DATA AVAILABILITY
-----------------
Requires the decay-curve producer (nousergon-data#963 — merged 2026-07-20)
to have run its backfill and populated the extra intermediate horizons
(1d, 3d, 15d) into ``score_performance_outcomes``. The existing 5d
diagnostic and 21d primary are always expected. 10d is deliberately
included as an intermediate point (the decay-curve collector added it
back for ladder continuity even though the original 10d/30d were retired
by config#1456 — the 10d-wide columns are dead but the long-format store
relies on the ``horizon_days`` value, which is independent).

Missing horizons are silently skipped (graceful degrade).  The result
has ``status: insufficient_data`` until at least ``min_samples`` signals
carry resolved outcomes at the primary (21d) horizon.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from nousergon_lib.quant.horizons import HorizonPolicy

logger = logging.getLogger(__name__)

# The full ladder of horizons for the alpha decay curve.
# 1d / 3d / 15d — added by nousergon-data#963 (decay-curve collector).
# 5d  — canonical diagnostic horizon.
# 10d — intermediate point (re-added by the decay-curve collector for
#       ladder continuity; the retired wide 10d columns do not affect
#       the long-format store which keys on ``horizon_days``).
# 21d — canonical primary horizon (system prediction target).
_LADDER = (1, 3, 5, 10, 15, 21)

# HorizonPolicy that treats all ladder horizons as diagnostic except 21d.
# Mirrors the producer's ``_DECAY_CURVE_POLICY`` in
# ``nousergon-data/collectors/signal_returns.py`` (added by nousergon-data#963).
DECAY_POLICY = HorizonPolicy(primary_horizon=21, diagnostic_horizons=(1, 3, 5, 10, 15))

# Minimum signals with resolved primary-horizon outcomes before reporting.
MIN_SAMPLES = 30

# Known score-bucket boundaries (mirrors signal_quality._accuracy_by_score_bucket).
_BUCKETS = [(60, 70), (70, 80), (80, 90), (90, 101)]


# ── Public API ──────────────────────────────────────────────────────────────────


def compute_decay_curve(
    df: pd.DataFrame,
    *,
    min_samples: int = MIN_SAMPLES,
) -> dict:
    """Compute the alpha decay curve from a score_performance DataFrame with
    decay-curve outcome columns attached.

    The input DataFrame must carry the wide-format outcome columns produced
    by ``signal_quality.load_score_performance(db_path, policy=DECAY_POLICY)``
    (or equivalently, by ``analysis.outcome_store.attach_outcomes`` with the
    decay-curve policy).  At minimum the primary (21d) horizon outcome columns
    must be resolved — the function gates on ``n >= min_samples`` with a
    resolved ``beat_spy_21d``.

    Returns a dict:
        status: "ok" or "insufficient_data"
        n_signals: int — count of signals with resolved primary-horizon outcomes
        n_needed: int — only present when status is insufficient_data
        overall: list[dict] — one entry per resolved horizon:
            {horizon_days, n, avg_alpha, accuracy}
        by_score_bucket: list[dict] — one entry per score bucket with data:
            {bucket, n, decay_curve: [{horizon_days, avg_alpha, n, accuracy}]}
    """
    # Gate on the primary (21d) horizon, matching signal_quality's pattern.
    resolved_21d = df[df["beat_spy_21d"].notna()] if "beat_spy_21d" in df.columns else pd.DataFrame()
    n_signals = len(resolved_21d)

    if n_signals < min_samples:
        logger.warning(
            "alpha_decay: only %d rows with resolved 21d outcome (need %d)",
            n_signals, min_samples,
        )
        return {
            "status": "insufficient_data",
            "n_signals": n_signals,
            "n_needed": min_samples,
        }

    # Overall decay curve — alpha at every resolved ladder horizon.
    overall = _decay_points(df)

    # Score-bucket stratified decay.
    by_score_bucket = _decay_by_score_bucket(df)

    return {
        "status": "ok",
        "n_signals": n_signals,
        "overall": overall,
        "by_score_bucket": by_score_bucket,
    }


def run_decay_curve(
    db_path: str | Path,
    *,
    min_samples: int = MIN_SAMPLES,
) -> dict:
    """Load score_performance with decay-curve outcome columns and compute the
    alpha decay curve in one call.

    Convenience wrapper for callers (evaluate.py) that need the artifact
    without loading separately.
    """
    # Late import to avoid circular dependency at module level.
    from analysis.signal_quality import load_score_performance

    df = load_score_performance(str(db_path), policy=DECAY_POLICY)
    return compute_decay_curve(df, min_samples=min_samples)


# ── Internal helpers ───────────────────────────────────────────────────────────


def _outcome_columns(horizon: int) -> tuple[str, str, str]:
    """Return ``(beat_spy_col, stock_return_col, spy_return_col)`` for
    a given horizon, matching the wide column names produced by
    ``HorizonPolicy.outcome_columns(horizon)``."""
    cols = DECAY_POLICY.outcome_columns(horizon)
    return cols.beat_spy, cols.stock_return, cols.spy_return


def _decay_points(df: pd.DataFrame) -> list[dict]:
    """For each ladder horizon with sufficient resolved rows, compute
    ``avg_alpha = mean(stock_return - spy_return)`` and accuracy."""
    points: list[dict] = []
    for h in _LADDER:
        beat_col, ret_col, spy_col = _outcome_columns(h)
        if beat_col not in df.columns:
            continue
        resolved = df[df[beat_col].notna()]
        n = len(resolved)
        if n < 5:
            continue
        alpha = (resolved[ret_col] - resolved[spy_col]).mean()
        points.append({
            "horizon_days": h,
            "n": n,
            "avg_alpha": round(float(alpha), 6),
            "accuracy": round(float(resolved[beat_col].mean()), 4),
        })
    return points


def _decay_by_score_bucket(df: pd.DataFrame) -> list[dict]:
    """Stratify the decay curve by score bucket using the signal's ``score``
    column.  Only buckets with sufficient resolved primary-horizon data are
    included."""
    if "score" not in df.columns:
        return []

    buckets: list[dict] = []
    for lo, hi in _BUCKETS:
        label = f"{lo}-{min(hi, 100)}" if hi <= 100 else f"{lo}+"
        mask = (df["score"] >= lo) & (df["score"] < hi)
        slice_df = df[mask]
        if slice_df.empty or "beat_spy_21d" not in slice_df.columns:
            continue
        n_bucket = slice_df["beat_spy_21d"].notna().sum()
        if n_bucket < 5:
            continue
        curve = _decay_points(slice_df)

        # Compute primary-horizon accuracy for the bucket summary.
        resolved_21d = slice_df[slice_df["beat_spy_21d"].notna()]
        accuracy_21d = round(float(resolved_21d["beat_spy_21d"].mean()), 4) if len(resolved_21d) > 0 else None

        buckets.append({
            "bucket": label,
            "n": n_bucket,
            "accuracy_21d": accuracy_21d,
            "decay_curve": curve,
        })

    return buckets
