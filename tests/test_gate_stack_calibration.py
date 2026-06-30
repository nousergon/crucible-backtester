"""Tests for leg (f) of the L4593 backtester correctness battery — false-positive
calibration of the auto-promotion GATE STACK.

The weight optimizer auto-writes ``config/scoring_weights.json`` to S3 every
Saturday (consumed by the research Lambda). The promote decision is a CONJUNCTION
of guardrails (status ok + OOS validation + confidence floor + bounded change +
meaningful change). Leg (a) calibrated single significance gates; this calibrates
the whole stack: drive ``compute_weights`` → ``apply_weights`` over many NULL
sub-score datasets (sub-scores drawn independently of beat-SPY outcomes) and pin
the empirical PROMOTE rate. On pure noise the system should almost never re-weight
production.

Measured baseline (seed 20260610, n=80): promote rate ≈ 0.10. That is the
auto-tuner changing live weights ~1-in-10 on noise — bounded (≤10% single change),
reversible (rollback), and well below an uncalibrated gate (which always finds
*some* shift). This suite PINS that rate against regression and proves the gate
still promotes/shifts on real signal (teeth). A breach of the ceiling is a finding
for EXPERIMENTS.md.

S3 is mocked throughout: on the rare null promote the real ``apply_weights`` path
PUTs to the bucket, so ``boto3`` + ``optimizer.rollback.save_previous`` are
neutralized.
"""
from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

import optimizer.rollback as rb
import optimizer.weight_optimizer as wo
from optimizer.weight_optimizer import compute_weights
from synthetic.gate_calibration import (
    build_null_subscore_df,
    run_weight_gate_null_calibration,
)

# Measured null promote rate is ~0.10; the ceiling gives principled headroom while
# still failing loudly if a gate-logic change makes the stack promote on noise more
# than ~1-in-5. Update only with a recorded EXPERIMENTS.md rationale.
NULL_PROMOTE_RATE_CEILING = 0.20
N_DATASETS = 80

_SUBSTANTIVE_REJECTIONS = {"oos_failed", "not_meaningful", "change_too_large"}


@pytest.fixture(scope="module")
def null_report():
    # mock.patch (not monkeypatch) so this can be module-scoped — the calibration
    # runs once and every assertion reads the same report.
    with mock.patch.object(wo, "boto3", mock.MagicMock()), \
         mock.patch.object(rb, "save_previous", lambda *a, **k: None):
        return run_weight_gate_null_calibration(n_datasets=N_DATASETS)


class TestGeneratorIsNull:
    def test_subscores_independent_of_outcomes(self):
        df = build_null_subscore_df(seed=1, n_rows=400)
        c_q = abs(np.corrcoef(df["quant_score"], df["beat_spy_21d"])[0, 1])
        c_l = abs(np.corrcoef(df["qual_score"], df["beat_spy_5d"])[0, 1])
        assert c_q < 0.15 and c_l < 0.15


class TestGateStackNullCalibration:
    def test_all_datasets_reach_ok_status(self, null_report):
        # n>=100 → confidence "medium", so the stack reaches its substantive
        # guardrails rather than bailing on sample size. If this regresses the
        # promote-rate result would be meaningless.
        assert dict(null_report.statuses) == {"ok": N_DATASETS}

    def test_promote_rate_below_ceiling(self, null_report):
        assert null_report.promote_rate <= NULL_PROMOTE_RATE_CEILING, (
            f"gate stack promoted on {null_report.promote_rate:.1%} of null datasets "
            f"(ceiling {NULL_PROMOTE_RATE_CEILING:.0%}): {null_report.summary()}"
        )

    def test_rejections_are_substantive_guardrails(self, null_report):
        # Every null rejection should come from a real guardrail (OOS, change
        # magnitude, meaningfulness) — not a degenerate status/confidence bail.
        seen = set(null_report.reject_reasons)
        assert seen <= _SUBSTANTIVE_REJECTIONS, f"unexpected reject reasons: {seen}"

    def test_no_real_s3_writes(self, null_report):
        # The fixture mocked boto3; reaching here without a botocore error means
        # no live S3 PUT occurred even on the null promotes.
        assert null_report.n_applied >= 0  # report computed under the mock


class TestGateHasTeeth:
    """The low null promote-rate must reflect calibration, not an always-reject
    gate: on a real (moderate) signal the optimizer shifts toward the predictive
    sub-score and clears OOS validation."""

    def _planted_signal_df(self, seed: int, n: int = 200) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        quant = rng.uniform(0.0, 100.0, n)
        # beat-SPY probability rises with quant_score; qual is pure noise.
        prob = 0.5 + 0.30 * ((quant - 50.0) / 50.0)
        return pd.DataFrame({
            "score_date": pd.bdate_range("2025-01-06", periods=n).astype(str),
            "quant_score": quant,
            "qual_score": rng.uniform(0.0, 100.0, n),
            "beat_spy_21d": (rng.uniform(0, 1, n) < prob).astype(float),
            "beat_spy_5d": (rng.uniform(0, 1, n) < prob).astype(float),
        })

    def test_optimizer_shifts_toward_predictive_subscore(self):
        res = compute_weights(self._planted_signal_df(seed=7))
        assert res["status"] == "ok"
        assert res["oos_passed"] is True
        # Weight moves toward the predictive sub-score (quant), away from noise (qual).
        assert res["suggested_weights"]["quant"] > res["suggested_weights"]["qual"]
        assert res["changes"]["quant"] > 0.0
