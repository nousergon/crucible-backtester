"""Tests for optimizer/factor_blend_optimizer.py — auto-apply companion to
the observability-only factor_blend_sensitivity diagnostic (config#748).

Pure-Python tests for the recommendation logic + the shadow/reproduction
gating (S3 is faked with a tiny in-memory stub — no boto3, no real bucket).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.factor_blend_sensitivity import (
    MIN_TRUSTWORTHY_SAMPLES,
    build_sensitivity_report,
)
from optimizer import factor_blend_optimizer as fbo


@pytest.fixture
def regime_weights():
    """Mirrors alpha-engine-config research/scoring.yaml aggregator.factor_blend."""
    return {
        "bull": {
            "momentum_score": 0.40, "quality_score": 0.30,
            "value_score": 0.20, "low_vol_score": -0.10,
        },
        "bear": {
            "low_vol_score": 0.40, "quality_score": 0.30,
            "momentum_score": -0.20, "value_score": 0.10,
        },
        "neutral": {
            "momentum_score": 0.25, "quality_score": 0.25,
            "value_score": 0.25, "low_vol_score": 0.25,
        },
    }


def _seed(regime, stance, mean, vol, n, seed):
    rng = np.random.default_rng(seed=seed)
    rows = []
    for r in rng.normal(mean, vol, n):
        rows.append({
            "market_regime": regime, "stance": stance,
            "return_10d": r, "spy_10d_return": 0.01,
            "beat_spy_10d": int(r > 0.01),
            "return_30d": r * 2, "spy_30d_return": 0.02,
            "beat_spy_30d": int(r * 2 > 0.02),
        })
    return rows


@pytest.fixture
def mismatch_report(regime_weights):
    """BULL config-top=momentum but realized-top=quality (a real mismatch,
    same shape as test_factor_blend_sensitivity.test_misaligned_config_detected)."""
    n = MIN_TRUSTWORTHY_SAMPLES + 5
    rows = (
        _seed("bull", "momentum", -0.01, 0.03, n, 2)   # negative Sortino
        + _seed("bull", "quality", 0.05, 0.025, n, 3)  # positive Sortino
    )
    return build_sensitivity_report(pd.DataFrame(rows), regime_weights, horizon="10d")


# ── recommend() ──────────────────────────────────────────────────────────────


class TestRecommend:
    def test_no_data(self, regime_weights):
        empty = build_sensitivity_report(pd.DataFrame(), regime_weights)
        out = fbo.recommend(empty, regime_weights)
        assert out["status"] in ("no_data", "insufficient_data")

    def test_aligned_no_recommendation(self, regime_weights):
        """Config-top == realized-top → no recommendation."""
        n = MIN_TRUSTWORTHY_SAMPLES + 5
        rows = (
            _seed("bull", "momentum", 0.05, 0.02, n, 1)  # high Sortino, is config top
            + _seed("bull", "quality", 0.02, 0.02, n, 4)
        )
        report = build_sensitivity_report(pd.DataFrame(rows), regime_weights)
        out = fbo.recommend(report, regime_weights)
        assert out["status"] == "insufficient_data"
        assert not out.get("recommendations")

    def test_material_mismatch_recommends_reorder(self, mismatch_report, regime_weights):
        out = fbo.recommend(mismatch_report, regime_weights)
        assert out["status"] == "ok"
        rec = out["recommendations"]["bull"]
        # quality should now hold the slot momentum used to (0.40); momentum
        # takes quality's old weight (0.30). Other weights unchanged.
        assert rec["quality_score"] == pytest.approx(0.40)
        assert rec["momentum_score"] == pytest.approx(0.30)
        assert rec["value_score"] == pytest.approx(0.20)
        assert rec["low_vol_score"] == pytest.approx(-0.10)

    def test_margin_floor_blocks_thin_mismatch(self, mismatch_report, regime_weights, monkeypatch):
        """A mismatch whose Sortino margin is below the floor is not acted on."""
        monkeypatch.setattr(fbo, "_MIN_SORTINO_MARGIN", 1e9)
        out = fbo.recommend(mismatch_report, regime_weights)
        assert out["status"] == "insufficient_data"
        assert not out.get("recommendations")


# ── _reweight_regime() ───────────────────────────────────────────────────────


class TestReweight:
    def test_swaps_into_top_slot(self):
        cur = {"momentum_score": 0.40, "quality_score": 0.30,
               "value_score": 0.20, "low_vol_score": -0.10}
        out = fbo._reweight_regime(cur, "quality")
        assert out["quality_score"] == 0.40
        assert out["momentum_score"] == 0.30

    def test_already_top_is_noop(self):
        cur = {"momentum_score": 0.40, "quality_score": 0.30}
        assert fbo._reweight_regime(cur, "momentum") is None

    def test_absent_stance_is_noop(self):
        cur = {"momentum_score": 0.40, "quality_score": 0.30}
        assert fbo._reweight_regime(cur, "value") is None


# ── apply() — shadow / reproduction gate (faked S3) ──────────────────────────


class _FakeS3:
    """Minimal in-memory S3 stub supporting put/list/get for the gate tests."""

    def __init__(self):
        self.store: dict[str, bytes] = {}

    def put_object(self, Bucket, Key, Body, ContentType=None):  # noqa: N803
        self.store[Key] = Body.encode() if isinstance(Body, str) else Body

    def list_objects_v2(self, Bucket, Prefix):  # noqa: N803
        contents = [{"Key": k} for k in self.store if k.startswith(Prefix)]
        return {"Contents": contents}

    def get_object(self, Bucket, Key):  # noqa: N803
        import io
        return {"Body": io.BytesIO(self.store[Key])}


@pytest.fixture
def fake_s3(monkeypatch):
    s3 = _FakeS3()
    import boto3
    monkeypatch.setattr(boto3, "client", lambda svc, *a, **k: s3)
    # Deterministic, monotonically-increasing run ids so the shadow archive
    # keys sort in time order (newest last → reversed = newest first).
    seq = {"n": 0}

    def _next_run_id():
        seq["n"] += 1
        return f"{seq['n']:010d}"

    import nousergon_lib.eval_artifacts as ea
    monkeypatch.setattr(ea, "new_eval_run_id", _next_run_id)
    return s3


def test_apply_off_by_default(mismatch_report, regime_weights, fake_s3):
    fbo.init_config({})  # no flags → use_factor_blend_target False
    rec = fbo.recommend(mismatch_report, regime_weights)
    res = fbo.apply(rec, "test-bucket")
    assert res["applied"] is False
    assert res["reason"] == "use_factor_blend_target=False"
    assert not fake_s3.store  # nothing written


def test_apply_shadow_writes_archive_not_live(mismatch_report, regime_weights, fake_s3):
    fbo.init_config({"factor_blend_optimizer": {"use_factor_blend_target": True}})
    rec = fbo.recommend(mismatch_report, regime_weights)
    res = fbo.apply(rec, "test-bucket")
    assert res["applied"] is False
    assert "shadow mode" in res["reason"]
    # shadow archive + latest sidecar written; live key NOT written
    assert any(fbo.S3_SHADOW_PREFIX in k for k in fake_s3.store)
    assert fbo.S3_LIVE_KEY not in fake_s3.store


def test_apply_live_blocked_until_reproduction(mismatch_report, regime_weights, fake_s3):
    fbo.init_config({"factor_blend_optimizer": {
        "use_factor_blend_target": True, "enforce_factor_blend": True,
    }})
    rec = fbo.recommend(mismatch_report, regime_weights)
    # First cycle: only one shadow archive exists → reproduction gate fails.
    res1 = fbo.apply(rec, "test-bucket")
    assert res1["applied"] is False
    assert "reproduction gate" in res1["reason"]
    assert fbo.S3_LIVE_KEY not in fake_s3.store
    # Second identical cycle: now _MIN_CONSECUTIVE_WEEKS (=2) identical
    # archives exist → gate passes, live write fires.
    res2 = fbo.apply(rec, "test-bucket")
    assert res2["applied"] is True
    assert fbo.S3_LIVE_KEY in fake_s3.store
