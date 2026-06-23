"""Tests for analysis/sample_size_adequacy.py (config#1151 Batch C)."""

from __future__ import annotations

from analysis.sample_size_adequacy import (
    ATTRIBUTION_SAMPLE_FLOOR,
    SIGNAL_QUALITY_SAMPLE_FLOOR,
    compute_sample_size_adequacy,
)


def _sq(n_30d=None, n_10d=None, status="ok"):
    return {"status": status, "overall": {"n_30d": n_30d, "n_10d": n_10d}}


def test_adequate_when_above_floor():
    r = compute_sample_size_adequacy(_sq(n_30d=SIGNAL_QUALITY_SAMPLE_FLOOR * 2))
    assert r["status"] == "ok"
    assert r["adequate"] is True
    assert r["adequacy_ratio"] == 2.0
    assert r["per_analysis"]["signal_quality"]["n"] == SIGNAL_QUALITY_SAMPLE_FLOOR * 2


def test_inadequate_below_floor():
    r = compute_sample_size_adequacy(_sq(n_30d=15))
    assert r["status"] == "ok"
    assert r["adequate"] is False
    assert r["adequacy_ratio"] == round(15 / SIGNAL_QUALITY_SAMPLE_FLOOR, 4)


def test_prefers_30d_over_10d():
    r = compute_sample_size_adequacy(_sq(n_30d=90, n_10d=200))
    assert r["per_analysis"]["signal_quality"]["n"] == 90  # 30d realized slice wins


def test_falls_back_to_10d_when_no_30d():
    r = compute_sample_size_adequacy(_sq(n_30d=None, n_10d=80))
    assert r["per_analysis"]["signal_quality"]["n"] == 80


def test_weakest_link_headline_across_analyses():
    # signal_quality well above its floor; attribution below its (larger) floor →
    # the headline adequacy is the WEAKEST link (attribution).
    sq = _sq(n_30d=120)  # ratio 2.0 vs floor 60
    attr = {"status": "ok", "n": 50}  # ratio 0.5 vs floor 100
    r = compute_sample_size_adequacy(sq, attr)
    assert r["weakest_analysis"] == "attribution"
    assert r["adequacy_ratio"] == round(50 / ATTRIBUTION_SAMPLE_FLOOR, 4)
    assert r["adequate"] is False


def test_insufficient_when_no_counts():
    assert compute_sample_size_adequacy(None)["status"] == "insufficient_data"
    assert compute_sample_size_adequacy({"status": "error"})["status"] == "insufficient_data"
    # signal_quality ok but no n at all → still insufficient.
    assert compute_sample_size_adequacy(_sq())["status"] == "insufficient_data"
