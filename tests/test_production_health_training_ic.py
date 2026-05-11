"""Pin behavior of `_load_training_ic` preference chain.

Regression for the 2026-05-11 false-positive retrain alert: the prior
implementation read `walk_forward.median_ic` or `test_ic` — both legacy
v2 fields that on the v3 meta-model can contain the Ridge's in-sample
fit (0.4634 today). The reference IC for degradation comparisons must
be the model's OOS performance at the active production horizon.

Preference chain:
  1. meta_model_oos_ic (post-PR-#2 predictor field)
  2. horizon_diagnostic.curve.{H}d.spearman
  3. walk_forward.median_ic (legacy, warning logged)
  4. test_ic (deepest legacy)
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest


@pytest.fixture
def s3_summary():
    """Patch boto3.client('s3').get_object to return a configurable summary dict."""
    def _factory(summary: dict | None):
        class _StubBody:
            def __init__(self, payload: bytes):
                self._payload = payload
            def read(self):
                return self._payload

        class _StubS3:
            def __init__(self, payload: dict | None):
                self._payload = payload
            def get_object(self, Bucket, Key):
                if self._payload is None:
                    raise RuntimeError(f"NoSuchKey: {Bucket}/{Key}")
                body = json.dumps(self._payload).encode()
                return {"Body": _StubBody(body)}

        return patch(
            "analysis.production_health.boto3.client",
            lambda svc: _StubS3(summary),
        )
    return _factory


def test_prefers_meta_model_oos_ic(s3_summary):
    from analysis.production_health import _load_training_ic
    summary = {
        "meta_model_oos_ic": 0.166,
        "horizon_diagnostic": {"curve": {"21d": {"spearman": 0.200}}},
        "walk_forward": {"median_ic": 0.4634},
        "test_ic": 0.999,
    }
    with s3_summary(summary):
        ic, src = _load_training_ic(bucket="b")
    assert ic == pytest.approx(0.166)
    assert src == "meta_model_oos_ic"


def test_falls_back_to_horizon_diagnostic_at_active_horizon(s3_summary):
    from analysis.production_health import _load_training_ic
    summary = {
        # meta_model_oos_ic absent (PR #2 not yet shipped)
        "horizon_diagnostic": {
            "curve": {
                "5d": {"spearman": 0.067},
                "21d": {"spearman": 0.166},
            }
        },
        "walk_forward": {"median_ic": 0.4634},  # inflated v3 in-sample fit
    }
    with s3_summary(summary):
        ic, src = _load_training_ic(bucket="b", active_horizon_days=21)
    assert ic == pytest.approx(0.166)
    assert src == "horizon_diagnostic.curve.21d.spearman"


def test_horizon_diagnostic_respects_active_horizon_arg(s3_summary):
    """If forward_days changes, the reference reads from the matching curve key."""
    from analysis.production_health import _load_training_ic
    summary = {
        "horizon_diagnostic": {
            "curve": {
                "5d": {"spearman": 0.067},
                "10d": {"spearman": 0.129},
                "21d": {"spearman": 0.166},
            }
        },
    }
    with s3_summary(summary):
        ic_5, src_5 = _load_training_ic(bucket="b", active_horizon_days=5)
        ic_10, src_10 = _load_training_ic(bucket="b", active_horizon_days=10)
    assert ic_5 == pytest.approx(0.067)
    assert "5d" in src_5
    assert ic_10 == pytest.approx(0.129)
    assert "10d" in src_10


def test_falls_back_to_walk_forward_median_ic_with_warning(s3_summary, caplog):
    """Legacy summary (no meta_model_oos_ic, no horizon_diagnostic) still
    works but logs a warning so the false-alarm risk is visible."""
    from analysis.production_health import _load_training_ic
    summary = {"walk_forward": {"median_ic": 0.10}}
    with caplog.at_level("WARNING", logger="analysis.production_health"):
        with s3_summary(summary):
            ic, src = _load_training_ic(bucket="b")
    assert ic == pytest.approx(0.10)
    assert src == "walk_forward.median_ic_legacy"
    assert any("in-sample" in r.message for r in caplog.records), (
        "Legacy fallback must warn about in-sample inflation risk"
    )


def test_falls_back_to_test_ic(s3_summary):
    from analysis.production_health import _load_training_ic
    summary = {"test_ic": 0.05}
    with s3_summary(summary):
        ic, src = _load_training_ic(bucket="b")
    assert ic == pytest.approx(0.05)
    assert src == "test_ic_legacy"


def test_absent_when_no_recognized_field(s3_summary):
    from analysis.production_health import _load_training_ic
    summary = {"unrelated": "data"}
    with s3_summary(summary):
        ic, src = _load_training_ic(bucket="b")
    assert ic is None
    assert src == "absent"


def test_load_failure_returns_load_failed(s3_summary):
    from analysis.production_health import _load_training_ic
    with s3_summary(None):  # raises in get_object
        ic, src = _load_training_ic(bucket="b")
    assert ic is None
    assert src == "load_failed"


def test_meta_model_oos_ic_null_skips_to_horizon_diagnostic(s3_summary):
    """If meta_model_oos_ic is present but None (e.g. nested-CV failed),
    fall through to horizon_diagnostic instead of returning None."""
    from analysis.production_health import _load_training_ic
    summary = {
        "meta_model_oos_ic": None,
        "horizon_diagnostic": {"curve": {"21d": {"spearman": 0.18}}},
    }
    with s3_summary(summary):
        ic, src = _load_training_ic(bucket="b", active_horizon_days=21)
    assert ic == pytest.approx(0.18)
    assert "horizon_diagnostic" in src


def test_horizon_curve_with_null_spearman_skips_to_legacy(s3_summary):
    """If the active-horizon curve entry exists but spearman is null
    (e.g. insufficient samples), fall through to legacy fields."""
    from analysis.production_health import _load_training_ic
    summary = {
        "horizon_diagnostic": {"curve": {"21d": {"spearman": None}}},
        "walk_forward": {"median_ic": 0.10},
    }
    with s3_summary(summary):
        ic, src = _load_training_ic(bucket="b", active_horizon_days=21)
    assert ic == pytest.approx(0.10)
    assert src == "walk_forward.median_ic_legacy"
