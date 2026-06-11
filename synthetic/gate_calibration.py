"""
gate_calibration.py — Leg (f) of the backtester correctness battery (ROADMAP
L4593): false-positive calibration of the auto-promotion GATE STACK.

WHY THIS EXISTS
---------------
Leg (a) calibrated single significance gates (a permutation test, the sim's PSR).
Leg (f) calibrates the *composite* promotion machinery that actually mutates live
production config: ``optimizer.weight_optimizer.compute_weights`` →
``apply_weights`` writes ``config/scoring_weights.json``, which the research Lambda
reads at cold start. That decision is a CONJUNCTION of guardrails (status ok + OOS
validation + confidence floor + bounded change + meaningful change). A conjunction
can still fire too often on noise if any leg is mis-calibrated.

This module drives the real gate stack over many NULL datasets — sub-scores drawn
independently of beat-SPY outcomes, so there is no real weight signal — and
measures the empirical PROMOTE rate (fraction of null runs that reach
``applied=True``). On pure noise the system should almost never change live
weights; a high rate means the auto-tuner re-weights production on nothing.

The caller is responsible for ensuring ``apply_weights`` does not touch S3 (mock
``boto3`` + ``optimizer.rollback.save_previous``) — on the rare null promote the
real code path would otherwise PUT to the bucket. The tests do this; see
``tests/test_gate_stack_calibration.py``. Recording surface: a high measured rate
is a finding for EXPERIMENTS.md, not a silent pass. See
[[feedback_sota_institutional_default_no_shortcuts]] + [[feedback_no_silent_fails]].
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from optimizer.weight_optimizer import apply_weights, compute_weights

logger = logging.getLogger(__name__)

# n >= 100 yields "medium" confidence (below the gate's low-confidence reject),
# so the stack reaches its substantive OOS / change-magnitude guardrails rather
# than bailing early on sample size.
DEFAULT_N_ROWS = 160


def build_null_subscore_df(
    seed: int,
    *,
    n_rows: int = DEFAULT_N_ROWS,
    n_dates: int = 24,
) -> pd.DataFrame:
    """Build a ``score_performance``-shaped frame where sub-scores are INDEPENDENT
    of the beat-SPY outcomes (and of the continuous returns).

    Columns match what ``compute_weights`` / ``_validate_and_split`` read:
    ``score_date``, ``quant_score``, ``qual_score``, ``beat_spy_10d``,
    ``beat_spy_30d`` (binary), plus ``return_10d`` / ``return_30d`` (continuous,
    used only if the skill-composite fit target is enabled). Under this null any
    weight shift the optimizer derives is sampling noise.
    """
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2025-01-06")
    dates = [(base + pd.tseries.offsets.BDay(i % n_dates)).strftime("%Y-%m-%d")
             for i in range(n_rows)]
    return pd.DataFrame({
        "score_date": dates,
        "quant_score": rng.uniform(0.0, 100.0, n_rows),
        "qual_score": rng.uniform(0.0, 100.0, n_rows),
        "beat_spy_10d": rng.integers(0, 2, n_rows).astype(float),
        "beat_spy_30d": rng.integers(0, 2, n_rows).astype(float),
        "return_10d": rng.normal(0.0, 4.0, n_rows),
        "return_30d": rng.normal(0.0, 4.0, n_rows),
    })


@dataclass
class GateStackNullReport:
    """Aggregate of :func:`run_weight_gate_null_calibration`."""
    n_datasets: int
    n_applied: int
    statuses: Counter = field(default_factory=Counter)
    reject_reasons: Counter = field(default_factory=Counter)

    @property
    def promote_rate(self) -> float:
        return self.n_applied / self.n_datasets if self.n_datasets else float("nan")

    def summary(self) -> dict:
        return {
            "n_datasets": self.n_datasets,
            "n_applied": self.n_applied,
            "promote_rate": self.promote_rate,
            "statuses": dict(self.statuses),
            "reject_reasons": dict(self.reject_reasons),
        }


def run_weight_gate_null_calibration(
    *,
    n_datasets: int = 40,
    seed: int = 20260610,
    n_rows: int = DEFAULT_N_ROWS,
    bucket: str = "null-calibration-bucket",
) -> GateStackNullReport:
    """Run ``compute_weights`` → ``apply_weights`` over ``n_datasets`` independent
    null sub-score frames and tally how often the gate stack promotes.

    NOTE: the caller MUST neutralize S3 side effects (mock ``boto3`` +
    ``optimizer.rollback.save_previous``) before calling — on a null promote the
    real ``apply_weights`` path PUTs to the bucket.
    """
    seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=n_datasets)
    report = GateStackNullReport(n_datasets=n_datasets, n_applied=0)

    for s in seeds:
        df = build_null_subscore_df(int(s), n_rows=n_rows)
        result = compute_weights(df)
        report.statuses[result.get("status", "unknown")] += 1
        decision = apply_weights(result, bucket)
        if decision.get("applied"):
            report.n_applied += 1
        else:
            # Coarsen the reason to a stable category for tallying.
            reason = decision.get("reason", "unknown")
            category = (
                "oos_failed" if "OOS" in reason
                else "not_meaningful" if "not worth" in reason
                else "change_too_large" if "exceeds" in reason
                else "low_confidence" if "confidence" in reason
                else "bad_status" if reason.startswith("status=")
                else "other"
            )
            report.reject_reasons[category] += 1

    return report
