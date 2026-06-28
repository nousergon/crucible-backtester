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


# ── attribution sample-count keying (config#946) ────────────────────────────


def test_attribution_count_read_from_rows_analyzed():
    """The real ``compute_attribution`` output keys its count as ``rows_analyzed``;
    the adequacy breakdown must read it (the prior ``n``/``n_samples``-only read
    silently dropped attribution every cycle — config#946)."""
    sq = _sq(n_30d=120)  # ratio 2.0 vs its floor
    attr = {"status": "ok", "rows_analyzed": 50}  # ratio 0.5 vs ATTRIBUTION_SAMPLE_FLOOR
    r = compute_sample_size_adequacy(sq, attr)
    assert "attribution" in r["per_analysis"]
    assert r["per_analysis"]["attribution"]["n"] == 50
    assert r["weakest_analysis"] == "attribution"
    assert r["adequacy_ratio"] == round(50 / ATTRIBUTION_SAMPLE_FLOOR, 4)


def test_attribution_count_fallback_keys_still_work():
    """`n` and `n_samples` remain accepted fallbacks for any alternate caller."""
    sq = _sq(n_30d=120)
    for key in ("n", "n_samples"):
        r = compute_sample_size_adequacy(sq, {"status": "ok", key: 50})
        assert r["per_analysis"]["attribution"]["n"] == 50


def test_attribution_breakdown_from_real_compute_attribution():
    """Integration: a real ``compute_attribution(df)`` result feeds the adequacy
    breakdown (guards the producer↔consumer contract that config#946 broke).

    Reuses the attribution suite's own ``_make_df`` fixture so the input matches
    the producer's real column contract."""
    from analysis.attribution import compute_attribution
    from tests.test_attribution import _make_df

    attr = compute_attribution(_make_df(n=150))
    assert attr.get("status") == "ok"
    assert "rows_analyzed" in attr  # the producer key the consumer must read
    assert attr["rows_analyzed"] > 0
    r = compute_sample_size_adequacy(_sq(n_30d=120), attr)
    assert "attribution" in r["per_analysis"]
    assert r["per_analysis"]["attribution"]["n"] == attr["rows_analyzed"]
