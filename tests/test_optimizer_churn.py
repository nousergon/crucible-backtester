"""Tests for analysis/optimizer_churn.py (config#1151 Batch C)."""

from __future__ import annotations

from analysis.optimizer_churn import compute_optimizer_churn


def _wr(changes, status="ok"):
    return {"status": status, "changes": changes}


def test_within_guardrails_small_moves():
    # cap 0.10; largest move 0.02 → ratio 0.2, well within.
    r = compute_optimizer_churn(_wr({"quant": -0.02, "qual": 0.02}), guardrail_cap=0.10)
    assert r["status"] == "ok"
    assert r["churn_ratio"] == 0.2
    assert r["within_guardrails"] is True
    assert r["max_change_param"] in ("quant", "qual")
    assert r["max_abs_change"] == 0.02


def test_churn_at_cap_is_not_within_guardrails():
    # largest move equals the cap → ratio 1.0 → NOT < 1.0.
    r = compute_optimizer_churn(_wr({"quant": 0.10, "qual": -0.10}), guardrail_cap=0.10)
    assert r["churn_ratio"] == 1.0
    assert r["within_guardrails"] is False


def test_churn_above_cap_flags_thrash():
    r = compute_optimizer_churn(_wr({"quant": 0.15, "qual": -0.15}), guardrail_cap=0.10)
    assert r["churn_ratio"] == 1.5
    assert r["within_guardrails"] is False
    assert r["max_abs_change"] == 0.15


def test_headline_is_largest_abs_move():
    r = compute_optimizer_churn(_wr({"quant": 0.01, "qual": -0.07}), guardrail_cap=0.10)
    assert r["max_change_param"] == "qual"
    assert r["max_abs_change"] == 0.07
    assert r["churn_ratio"] == 0.7


def test_n_params_changed_counts_nonzero():
    r = compute_optimizer_churn(_wr({"quant": 0.0, "qual": 0.05}), guardrail_cap=0.10)
    assert r["n_params_changed"] == 1
    assert r["per_param_abs_change"] == {"quant": 0.0, "qual": 0.05}


def test_default_cap_from_optimizer_module():
    # No explicit cap → producer pulls the optimizer's own _MAX_SINGLE_CHANGE.
    from optimizer.weight_optimizer import _MAX_SINGLE_CHANGE
    r = compute_optimizer_churn(_wr({"quant": _MAX_SINGLE_CHANGE, "qual": -_MAX_SINGLE_CHANGE}))
    assert r["guardrail_cap"] == _MAX_SINGLE_CHANGE
    assert r["churn_ratio"] == 1.0


def test_none_changes_ignored():
    r = compute_optimizer_churn(_wr({"quant": None, "qual": 0.04}), guardrail_cap=0.10)
    assert r["status"] == "ok"
    assert "quant" not in r["per_param_abs_change"]
    assert r["max_change_param"] == "qual"


def test_insufficient_when_not_ok():
    assert compute_optimizer_churn(None)["status"] == "insufficient_data"
    assert compute_optimizer_churn(_wr({}, status="insufficient_data"))["status"] == "insufficient_data"


def test_insufficient_when_no_changes():
    r = compute_optimizer_churn(_wr({}))
    assert r["status"] == "insufficient_data"
    # all-None changes → also insufficient (no usable delta).
    assert compute_optimizer_churn(_wr({"quant": None}))["status"] == "insufficient_data"


def test_degenerate_cap_gives_no_ratio():
    r = compute_optimizer_churn(_wr({"quant": 0.05}), guardrail_cap=0.0)
    assert r["status"] == "ok"
    assert r["churn_ratio"] is None
    assert r["within_guardrails"] is False
