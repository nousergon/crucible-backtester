"""Unit tests for optimizer.pillar_weight_optimizer (config#789 Phase 6).

SHADOW-ONLY per-pillar weight optimizer. These tests pin the design intent:
  - Σ pillar weights == 1.0 sampling/normalization constraint.
  - alpha_floor hard constraint rejects alpha-negative candidates BEFORE ranking.
  - ranking/recommend selects the best candidate by Sortino-on-skilled-risk.
  - apply() writes ONLY to the shadow-history prefix (the previously-dead
    S3_SHADOW_WEIGHTS_PREFIX) and NEVER the live scoring-weights key.

S3 is mocked via MagicMock patched onto the module's boto3 (the repo's
executor_optimizer / weight_optimizer test pattern), so no network is touched.
"""
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from optimizer import pillar_weight_optimizer as pwo
from optimizer.pillar_weight_optimizer import (
    LEGACY_BLEND_KEY,
    PILLAR_WEIGHT_KEYS,
    S3_SHADOW_WEIGHTS_PREFIX,
    S3_WEIGHTS_KEY,
    WITHIN_PILLAR_KEY,
    apply,
    init_config,
    normalize_pillar_weights,
    recommend,
    sample_weight_vector,
    _sum_to_one_ok,
)

# The two canonical constants Phase 6 wires: shadow prefix reused, live key never written.
_SKILL = pwo._SKILL_TARGET
_RESOLVED = pwo._RESOLVED_OUTCOME


# ── Fixtures / helpers ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _default_config():
    init_config(
        {
            "pillar_weight_optimizer": {
                "min_valid_candidates": 3,
                "min_sortino_improvement": 0.05,
                "max_trials": 60,
                "seed": 7,
            }
        }
    )
    yield


def _make_df(n_dates: int = 25, n_names: int = 40, seed: int = 0,
             predictive_pillar: str = "quality", noise: float = 0.005) -> pd.DataFrame:
    """Per-name pillar-breakdown df where ``predictive_pillar``'s quant component
    drives realized log-alpha, so a Sortino-ranked sweep should up-weight it."""
    rng = np.random.RandomState(seed)
    rows = []
    for di in range(n_dates):
        d = f"2026-01-{di + 1:02d}"
        for ni in range(n_names):
            quant = {f"{p}_quant": rng.rand() * 100 for p in PILLAR_WEIGHT_KEYS}
            qual = {f"{p}_qual": rng.rand() * 100 for p in PILLAR_WEIGHT_KEYS}
            alpha = 0.04 * (quant[f"{predictive_pillar}_quant"] - 50) / 50 + rng.randn() * noise
            rows.append(
                {
                    "symbol": f"S{ni}",
                    "score_date": d,
                    "legacy_blend_score": rng.rand() * 100,
                    _SKILL: alpha,
                    _RESOLVED: 1,
                    **quant,
                    **qual,
                }
            )
    return pd.DataFrame(rows)


def _mock_s3():
    s3 = MagicMock()
    return s3


# ── Σ weights == 1 sampling / normalization constraint ────────────────────────


class TestSumToOneConstraint:
    def test_sampled_vectors_sum_to_one(self):
        import random

        rng = random.Random(123)
        for _ in range(500):
            vec = sample_weight_vector(rng)
            assert _sum_to_one_ok(vec)
            assert abs(sum(vec[k] for k in PILLAR_WEIGHT_KEYS) - 1.0) <= 1e-6
            # mix terms live in [0, 1], NOT on the simplex
            assert 0.0 <= vec[WITHIN_PILLAR_KEY] <= 1.0
            assert 0.0 <= vec[LEGACY_BLEND_KEY] <= 1.0

    def test_normalize_projects_onto_simplex(self):
        skewed = {k: 3.0 for k in PILLAR_WEIGHT_KEYS}
        skewed[WITHIN_PILLAR_KEY] = 5.0  # out of range → clamped
        skewed[LEGACY_BLEND_KEY] = -2.0  # out of range → clamped
        out = normalize_pillar_weights(skewed)
        assert abs(sum(out[k] for k in PILLAR_WEIGHT_KEYS) - 1.0) <= 1e-6
        assert out[WITHIN_PILLAR_KEY] == 1.0
        assert out[LEGACY_BLEND_KEY] == 0.0

    def test_recommended_weights_sum_to_one(self):
        res = recommend(_make_df(seed=1))
        assert res["status"] == "ok"
        rec = res["recommended_weights"]
        assert abs(sum(rec[k] for k in PILLAR_WEIGHT_KEYS) - 1.0) <= 1e-6
        assert res["constraint"] == "sum_pillar_weights_eq_1"


# ── Ranking / recommend selects best by Sortino ───────────────────────────────


class TestRankingBySortino:
    def test_recommend_ok_and_beats_baseline(self):
        res = recommend(_make_df(seed=2))
        assert res["status"] == "ok"
        assert res["rank_metric"] == "sortino_ratio"
        # winner ranks first → best >= baseline Sortino
        assert res["best_sortino"] >= res["baseline_sortino"]
        assert res["improvement_pct"] >= 0.05

    def test_upweights_the_predictive_pillar(self):
        # value pillar drives realized alpha here → should get above-equal weight.
        res = recommend(_make_df(seed=3, predictive_pillar="value"))
        assert res["status"] == "ok"
        equal = 1.0 / len(PILLAR_WEIGHT_KEYS)
        assert res["recommended_weights"]["value"] > equal

    def test_no_improvement_when_signal_is_noise(self):
        # Pure-noise realized alpha → best candidate barely beats baseline.
        df = _make_df(seed=4, noise=0.05)
        df[_SKILL] = np.random.RandomState(99).randn(len(df)) * 0.02
        init_config(
            {"pillar_weight_optimizer": {"min_valid_candidates": 3,
                                         "min_sortino_improvement": 0.99,  # unreachable
                                         "max_trials": 40, "seed": 7}}
        )
        res = recommend(df)
        assert res["status"] in ("no_improvement", "baseline_insignificant", "negative_sortino")

    def test_insufficient_dates_returns_early(self):
        df = _make_df(n_dates=3)  # < _MIN_RESOLVED_DATES
        res = recommend(df)
        assert res["status"] == "insufficient_data"

    def test_empty_df_returns_early(self):
        assert recommend(pd.DataFrame())["status"] == "insufficient_data"

    def test_missing_resolved_column_returns_early(self):
        df = _make_df().drop(columns=[_RESOLVED])
        assert recommend(df)["status"] == "insufficient_data"


# ── alpha_floor hard constraint ───────────────────────────────────────────────


class TestAlphaFloor:
    def test_alpha_floor_rejects_all_when_unreachable(self):
        init_config(
            {"pillar_weight_optimizer": {"min_valid_candidates": 3, "max_trials": 40,
                                         "seed": 7, "alpha_floor": 1.0}}  # 100% alpha — impossible
        )
        res = recommend(_make_df(seed=5))
        assert res["status"] == "alpha_below_floor"
        assert res["alpha_floor"] == 1.0
        assert "best_alpha_in_sweep" in res

    def test_alpha_floor_inactive_by_default(self):
        # No alpha_floor in config → gate never fires, normal ranking applies.
        res = recommend(_make_df(seed=6))
        assert res["status"] == "ok"

    def test_alpha_floor_zero_still_recommends_when_positive(self):
        init_config(
            {"pillar_weight_optimizer": {"min_valid_candidates": 3, "max_trials": 40,
                                         "seed": 7, "alpha_floor": 0.0}}
        )
        res = recommend(_make_df(seed=2))
        # predictive pillar → top-decile realized alpha is positive → survives floor
        assert res["status"] == "ok"
        assert res["best_alpha"] >= 0.0


# ── Shadow S3 write goes to the right prefix; live key NEVER written ───────────


class TestShadowApply:
    @patch("optimizer.pillar_weight_optimizer.boto3")
    def test_apply_writes_only_to_shadow_prefix(self, mock_boto3):
        s3 = _mock_s3()
        mock_boto3.client.return_value = s3
        res = recommend(_make_df(seed=2))
        assert res["status"] == "ok"
        out = apply(res, bucket="test-bucket")

        # shadow archival is NEVER a live apply
        assert out["applied"] is False
        assert "shadow" in out["reason"].lower()

        keys = [c.kwargs["Key"] for c in s3.put_object.call_args_list]
        assert keys, "apply should have written to S3"
        # every written key is under the shadow prefix (artifact + latest sidecar)
        assert all(k.startswith(S3_SHADOW_WEIGHTS_PREFIX) for k in keys)
        assert any(k.endswith("/latest.json") for k in keys)

    @patch("optimizer.pillar_weight_optimizer.boto3")
    def test_apply_never_writes_live_scoring_weights_key(self, mock_boto3):
        s3 = _mock_s3()
        mock_boto3.client.return_value = s3
        res = recommend(_make_df(seed=2))
        out = apply(res, bucket="test-bucket")

        keys = [c.kwargs["Key"] for c in s3.put_object.call_args_list]
        # HARD INVARIANT: the live scoring-weights config key is never touched.
        assert S3_WEIGHTS_KEY == "config/scoring_weights.json"
        assert S3_WEIGHTS_KEY not in keys
        assert not any(k == S3_WEIGHTS_KEY for k in keys)
        assert out.get("shadow_key", "").startswith(S3_SHADOW_WEIGHTS_PREFIX)

    @patch("optimizer.pillar_weight_optimizer.boto3")
    def test_apply_payload_marks_shadow_mode(self, mock_boto3):
        s3 = _mock_s3()
        mock_boto3.client.return_value = s3
        res = recommend(_make_df(seed=2))
        apply(res, bucket="test-bucket")
        # inspect the artifact body (first put_object)
        body = json.loads(s3.put_object.call_args_list[0].kwargs["Body"])
        assert body["mode"] == "shadow"
        assert body["constraint"] == "sum_pillar_weights_eq_1"
        assert abs(sum(body[k] for k in PILLAR_WEIGHT_KEYS) - 1.0) <= 1e-6

    @patch("optimizer.pillar_weight_optimizer.boto3")
    def test_apply_noops_on_non_ok_status(self, mock_boto3):
        s3 = _mock_s3()
        mock_boto3.client.return_value = s3
        out = apply({"status": "no_improvement"}, bucket="test-bucket")
        assert out["applied"] is False
        assert s3.put_object.call_args_list == []

    @patch("optimizer.pillar_weight_optimizer.boto3")
    def test_shadow_prefix_is_the_wired_dead_constant(self, mock_boto3):
        # Phase 6 wires the previously-dead weight_optimizer constant.
        from optimizer.weight_optimizer import S3_SHADOW_WEIGHTS_PREFIX as WOP_PREFIX

        assert S3_SHADOW_WEIGHTS_PREFIX == WOP_PREFIX
        assert S3_SHADOW_WEIGHTS_PREFIX == "config/scoring_weights_shadow_history"
