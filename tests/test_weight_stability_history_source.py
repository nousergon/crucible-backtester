"""alpha-engine-config-I12627: the weight-drift check read a key nothing writes.

`_check_stability` loaded ``config/scoring_weights_history/{calendar date}.json``.
`apply_weights` archives history only on an APPLIED change, under a run-id
layout, and the loop had been blocked for 20 straight weeks. So the 10-23
weekly wrote ``walk_forward_stability: insufficient_data, weeks_loaded 0``,
while ``config/scoring_weights/recommendations/{date}/`` held 20 weekly
recommendations. These tests pin the reader to the archive every run writes.
"""

from __future__ import annotations

import io
import json

from optimizer import weight_optimizer as wo
from analysis.walk_forward_stability import compute_walk_forward_stability


class _FakeS3:
    def __init__(self, recs: dict[str, dict]):
        self._objs = {
            f"config/scoring_weights/recommendations/{d}/from_weight_optimizer.json": {
                "recommended_params": params
            }
            for d, params in recs.items()
        }

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        objs = self._objs

        class _P:
            def paginate(self, Bucket, Prefix, Delimiter):
                prefixes = sorted({k[: len(Prefix) + 11] for k in objs if k.startswith(Prefix)})
                yield {"CommonPrefixes": [{"Prefix": p} for p in prefixes]}

        return _P()

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(json.dumps(self._objs[Key]).encode())}


_FLAT = {"quant": 0.5, "qual": 0.5}


def test_reads_the_three_prior_weekly_recommendations():
    s3 = _FakeS3({
        "2026-09-25": _FLAT, "2026-10-02": _FLAT, "2026-10-09": _FLAT,
        "2026-10-16": _FLAT, "2026-10-23": _FLAT,
    })
    out = wo._check_stability(_FLAT, bucket="b", s3=s3, run_date="2026-10-23")
    assert out["weeks_loaded"] == 3
    assert out["reversals"] == [] and out["stable"] is True


def test_the_current_runs_own_partition_is_never_its_own_history():
    """A rerun finds today's recommendation already written; it must not count."""
    s3 = _FakeS3({"2026-10-16": _FLAT, "2026-10-23": _FLAT})
    out = wo._check_stability(_FLAT, bucket="b", s3=s3, run_date="2026-10-23")
    assert out["weeks_loaded"] == 1


def test_oscillation_across_the_archive_is_a_reversal():
    s3 = _FakeS3({
        "2026-10-02": {"quant": 0.50, "qual": 0.50},
        "2026-10-09": {"quant": 0.60, "qual": 0.40},
        "2026-10-16": {"quant": 0.50, "qual": 0.50},
    })
    out = wo._check_stability({"quant": 0.60, "qual": 0.40}, bucket="b", s3=s3, run_date="2026-10-23")
    assert out["weeks_loaded"] == 3
    assert out["stable"] is False and len(out["reversals"]) >= 2


def test_the_producer_now_measures_on_the_live_shape():
    """End to end through the report-card producer: the 10-23 live archive
    (identical 0.5/0.5 recommendations) is a measured, stable series, not
    insufficient_data."""
    s3 = _FakeS3({"2026-10-02": _FLAT, "2026-10-09": _FLAT, "2026-10-16": _FLAT})
    stab = wo._check_stability(_FLAT, bucket="b", s3=s3, run_date="2026-10-23")
    res = compute_walk_forward_stability({"status": "ok", "stability": stab})
    assert res["status"] != "insufficient_data", res
    assert res["weeks_loaded"] == 3


def test_an_unlistable_archive_degrades_to_zero_weeks_not_a_raise():
    class _Denied(_FakeS3):
        def get_paginator(self, name):
            raise RuntimeError("AccessDenied")

    out = wo._check_stability(_FLAT, bucket="b", s3=_Denied({}), run_date="2026-10-23")
    assert out["weeks_loaded"] == 0
