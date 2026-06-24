"""Tests for analysis/walk_forward_stability.py (config#1151 Batch C)."""

from __future__ import annotations

from analysis.walk_forward_stability import compute_walk_forward_stability


def _wr(stability, status="ok"):
    return {"status": status, "stability": stability}


def test_fully_stable_no_reversals():
    # 3 weeks loaded → 4 points → 3 deltas → 2 consecutive-pair comparisons per
    # sub-score → max 4 reversals (2 sub-scores). Zero reversals → ratio 1.0.
    r = compute_walk_forward_stability(
        _wr({"weeks_loaded": 3, "reversals": [], "stable": True}), n_sub_scores=2,
    )
    assert r["status"] == "ok"
    assert r["stability_ratio"] == 1.0
    assert r["n_reversals"] == 0
    assert r["max_possible_reversals"] == 4
    assert r["stable"] is True


def test_some_reversals_lower_ratio():
    r = compute_walk_forward_stability(
        _wr({"weeks_loaded": 3, "reversals": ["quant: ↑ → ↓ (weeks 1 → 2)"], "stable": False}),
        n_sub_scores=2,
    )
    assert r["n_reversals"] == 1
    assert r["max_possible_reversals"] == 4
    assert r["stability_ratio"] == 0.75
    assert r["stable"] is False


def test_full_oscillation_floor_at_zero():
    revs = ["r1", "r2", "r3", "r4"]
    r = compute_walk_forward_stability(
        _wr({"weeks_loaded": 3, "reversals": revs, "stable": False}), n_sub_scores=2,
    )
    assert r["stability_ratio"] == 0.0


def test_ratio_clamped_when_reversals_exceed_bound():
    # Defensive: more reversals than the theoretical max → clamp to 0, not negative.
    r = compute_walk_forward_stability(
        _wr({"weeks_loaded": 3, "reversals": ["a"] * 10, "stable": False}), n_sub_scores=2,
    )
    assert r["stability_ratio"] == 0.0


def test_insufficient_when_fewer_than_two_weeks():
    r = compute_walk_forward_stability(
        _wr({"weeks_loaded": 1, "reversals": [], "stable": True}),
    )
    assert r["status"] == "insufficient_data"
    assert r["weeks_loaded"] == 1


def test_insufficient_when_no_history():
    r = compute_walk_forward_stability(
        _wr({"weeks_loaded": 0, "reversals": [], "stable": True}),
    )
    assert r["status"] == "insufficient_data"


def test_insufficient_when_weight_result_not_ok():
    assert compute_walk_forward_stability(None)["status"] == "insufficient_data"
    assert (
        compute_walk_forward_stability({"status": "insufficient_data"})["status"]
        == "insufficient_data"
    )


def test_missing_stability_block_is_insufficient():
    r = compute_walk_forward_stability({"status": "ok"})
    assert r["status"] == "insufficient_data"


def test_reversals_passed_through():
    revs = ["quant: ↑ → ↓ (weeks 2 → 3)"]
    r = compute_walk_forward_stability(
        _wr({"weeks_loaded": 3, "reversals": revs, "stable": False}),
    )
    assert r["reversals"] == revs
