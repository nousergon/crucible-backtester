"""
Unit tests for analysis/portfolio_optimizer_gate.py — PR 4 of the
portfolio-optimizer-260511 arc.

The gate evaluator is a pure function: input is the side-by-side dict from
compare_to_legacy(); output is a pass/fail report. Tests exercise each
criterion with synthetic compare dicts; the orchestrator is left for
Saturday SF integration testing (heavy fixture).
"""

from __future__ import annotations

import pytest

from analysis.portfolio_optimizer_gate import (
    evaluate_gate,
    gate_passed,
)


def _all_pass_comparison() -> dict:
    """Construct a comparison dict where every criterion should PASS."""
    return {
        "optimizer": {
            "sortino_ratio": 1.5,
            "psr": 0.97,
            "max_drawdown": -0.10,
            "cvar_95": -0.015,
            "turnover_one_way_ann": 1.5,
            "tracking_error_ann": 0.04,
            "mean_active_share": 0.15,
        },
        "legacy": {
            "sortino_ratio": 1.4,
            "psr": 0.95,
            "max_drawdown": -0.15,
            "cvar_95": -0.020,
            "turnover_one_way_ann": 1.0,
        },
        "gate_thresholds": {
            "sortino_min": 1.4 * 0.9,
            "psr_min": 0.95,
            "max_drawdown_floor": -0.15 * 1.2,
            "cvar_95_floor": -0.020 * 1.2,
            "turnover_max": 1.0 * 2.5,
            "tracking_error_range": [0.02, 0.06],
            "active_share_range": [0.08, 0.25],
        },
    }


class TestEvaluateGateAllPass:
    def test_verdict_is_pass(self):
        report = evaluate_gate(_all_pass_comparison())
        assert report["verdict"] == "pass"
        assert report["n_fail"] == 0
        assert report["n_pass"] >= 5

    def test_gate_passed_helper(self):
        assert gate_passed(evaluate_gate(_all_pass_comparison())) is True


class TestSortinoCriterion:
    def test_sortino_below_threshold_fails(self):
        comp = _all_pass_comparison()
        comp["optimizer"]["sortino_ratio"] = 1.0
        report = evaluate_gate(comp)
        sortino = next(c for c in report["criteria"] if c["name"] == "sortino_min")
        assert sortino["status"] == "fail"
        assert report["verdict"] == "fail"

    def test_sortino_skipped_when_no_legacy(self):
        comp = _all_pass_comparison()
        comp["legacy"] = None
        comp["gate_thresholds"]["sortino_min"] = None
        report = evaluate_gate(comp)
        sortino = next(c for c in report["criteria"] if c["name"] == "sortino_min")
        assert sortino["status"] == "skipped_no_legacy"


class TestPsrCriterion:
    def test_psr_below_0_95_fails(self):
        comp = _all_pass_comparison()
        comp["optimizer"]["psr"] = 0.80
        report = evaluate_gate(comp)
        psr = next(c for c in report["criteria"] if c["name"] == "psr_min")
        assert psr["status"] == "fail"
        assert report["verdict"] == "fail"

    def test_psr_none_skips_gate_not_fails(self):
        """PSR=None means 'insufficient daily returns to compute' — that's a
        data-coverage signal, not a failure. Mirrors executor_optimizer's
        precision_ci_95 pattern in the veto skill-composite cutover."""
        comp = _all_pass_comparison()
        comp["optimizer"]["psr"] = None
        report = evaluate_gate(comp)
        psr = next(c for c in report["criteria"] if c["name"] == "psr_min")
        assert psr["status"] == "skipped_no_legacy"

    def test_psr_threshold_is_absolute_not_legacy_relative(self):
        """PSR floor is 0.95 regardless of legacy PSR."""
        comp = _all_pass_comparison()
        comp["legacy"] = None
        comp["gate_thresholds"]["psr_min"] = 0.95
        report = evaluate_gate(comp)
        psr = next(c for c in report["criteria"] if c["name"] == "psr_min")
        assert psr["status"] == "pass", \
            "PSR threshold is absolute (0.95) and applies even without legacy"


class TestDrawdownAndCvar:
    def test_max_dd_more_negative_than_floor_fails(self):
        comp = _all_pass_comparison()
        comp["optimizer"]["max_drawdown"] = -0.25
        report = evaluate_gate(comp)
        dd = next(c for c in report["criteria"] if c["name"] == "max_drawdown_floor")
        assert dd["status"] == "fail"

    def test_cvar_more_negative_than_floor_fails(self):
        comp = _all_pass_comparison()
        comp["optimizer"]["cvar_95"] = -0.05
        report = evaluate_gate(comp)
        cvar = next(c for c in report["criteria"] if c["name"] == "cvar_95_floor")
        assert cvar["status"] == "fail"


class TestTurnover:
    def test_turnover_above_max_fails(self):
        comp = _all_pass_comparison()
        comp["optimizer"]["turnover_one_way_ann"] = 5.0
        report = evaluate_gate(comp)
        t = next(c for c in report["criteria"] if c["name"] == "turnover_max")
        assert t["status"] == "fail"


class TestRangeGates:
    def test_tracking_error_outside_range_fails(self):
        comp = _all_pass_comparison()
        comp["optimizer"]["tracking_error_ann"] = 0.15
        report = evaluate_gate(comp)
        te = next(c for c in report["criteria"] if c["name"] == "tracking_error_range")
        assert te["status"] == "fail"

    def test_tracking_error_under_range_fails(self):
        comp = _all_pass_comparison()
        comp["optimizer"]["tracking_error_ann"] = 0.005
        report = evaluate_gate(comp)
        te = next(c for c in report["criteria"] if c["name"] == "tracking_error_range")
        assert te["status"] == "fail"

    def test_active_share_outside_range_fails(self):
        comp = _all_pass_comparison()
        comp["optimizer"]["mean_active_share"] = 0.50
        report = evaluate_gate(comp)
        a_s = next(c for c in report["criteria"] if c["name"] == "active_share_range")
        assert a_s["status"] == "fail"


class TestNoLegacyBaseline:
    def test_first_run_path_skips_legacy_gates_keeps_absolute_gates(self):
        """When legacy is None, the gate should still check absolute criteria
        (PSR, tracking-error range, active-share range) but skip legacy-
        relative ones. Verdict can still be 'pass' if all checkable gates pass."""
        comp = {
            "optimizer": {
                "sortino_ratio": 1.5,
                "psr": 0.97,
                "max_drawdown": -0.10,
                "cvar_95": -0.015,
                "turnover_one_way_ann": 1.5,
                "tracking_error_ann": 0.04,
                "mean_active_share": 0.15,
            },
            "legacy": None,
            "gate_thresholds": {
                "sortino_min": None,
                "psr_min": 0.95,
                "max_drawdown_floor": None,
                "cvar_95_floor": None,
                "turnover_max": None,
                "tracking_error_range": [0.02, 0.06],
                "active_share_range": [0.08, 0.25],
            },
        }
        report = evaluate_gate(comp)
        assert report["n_skipped"] == 4, \
            f"sortino/max_dd/cvar/turnover should be skipped; got {report['n_skipped']}"
        assert report["n_pass"] == 3, \
            f"psr/TE/AS should pass; got {report['n_pass']}"
        assert report["verdict"] == "pass", \
            "All checkable gates pass → overall verdict pass"


class TestInputValidation:
    def test_non_dict_raises_typeerror(self):
        with pytest.raises(TypeError):
            evaluate_gate("not a dict")  # type: ignore[arg-type]

    def test_missing_optimizer_key_raises(self):
        with pytest.raises(KeyError):
            evaluate_gate({"gate_thresholds": {}})

    def test_missing_gate_thresholds_raises(self):
        with pytest.raises(KeyError):
            evaluate_gate({"optimizer": {}})


class TestSummaryRendering:
    def test_summary_contains_verdict_and_each_criterion(self):
        report = evaluate_gate(_all_pass_comparison())
        s = report["summary"]
        assert "PASS" in s.upper()
        for name in ("sortino_min", "psr_min", "max_drawdown_floor",
                     "cvar_95_floor", "turnover_max",
                     "tracking_error_range", "active_share_range"):
            assert name in s
