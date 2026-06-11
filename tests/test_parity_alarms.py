"""Tests for leg (g) of the L4593 backtester correctness battery — tolerance-band
+ step-change alarms on the parity deltas (analysis/parity_alarms.py) and their
observe-mode wiring into the pit_parity contamination report.

Coverage:
- tolerance bands: in-band passes, out-of-band breaches, None deltas skipped
- step-change: jump vs prior run breaches; no prior ⇒ no step results
- observe vs paging: observe never pages; paging publishes on breach; env opt-out
  + injected publish_fn for deterministic, transport-free assertions
- report integration: build_contamination_report carries an observe-mode alarms
  block and never pages from the pure builder
"""
from __future__ import annotations

import pytest

from analysis.parity_alarms import (
    DEFAULT_TOLERANCE_BANDS,
    _ALERT_DISABLED_ENV_VAR,
    evaluate_parity_alarms,
    evaluate_step_change,
    evaluate_tolerance_bands,
)


class TestToleranceBands:
    def test_in_band_no_breach(self):
        delta = {"sortino_ratio": 0.05, "psr": -0.04, "total_alpha": 0.01}
        res = evaluate_tolerance_bands(delta)
        assert all(not r["breach"] for r in res.values())

    def test_out_of_band_breaches(self):
        delta = {"sortino_ratio": 0.25, "total_alpha": -0.05}
        res = evaluate_tolerance_bands(delta)
        assert res["sortino_ratio"]["breach"] is True
        assert res["total_alpha"]["breach"] is True

    def test_none_deltas_skipped(self):
        delta = {"sortino_ratio": None, "psr": 0.02}
        res = evaluate_tolerance_bands(delta)
        assert "sortino_ratio" not in res
        assert "psr" in res

    def test_boundary_is_not_a_breach(self):
        # |delta| == band is within tolerance (strict >).
        band = DEFAULT_TOLERANCE_BANDS["sortino_ratio"]
        res = evaluate_tolerance_bands({"sortino_ratio": band})
        assert res["sortino_ratio"]["breach"] is False


class TestStepChange:
    def test_no_prior_yields_empty(self):
        assert evaluate_step_change({"sortino_ratio": 0.5}, None) == {}

    def test_large_jump_breaches(self):
        # Level can stay in-band while the run-over-run jump is large.
        prior = {"sortino_ratio": 0.0}
        cur = {"sortino_ratio": 0.30}  # step 0.30 > step band (0.20)
        res = evaluate_step_change(cur, prior)
        assert res["sortino_ratio"]["breach"] is True

    def test_small_step_no_breach(self):
        res = evaluate_step_change({"psr": 0.03}, {"psr": 0.01})
        assert res["psr"]["breach"] is False

    def test_metric_missing_on_one_side_skipped(self):
        res = evaluate_step_change({"psr": 0.5}, {"sortino_ratio": 0.5})
        assert res == {}


class TestObserveMode:
    def test_observe_never_pages_even_on_breach(self):
        calls = []
        verdict = evaluate_parity_alarms(
            {"sortino_ratio": 0.5},  # clear breach
            paging_enabled=False,
            publish_fn=lambda *a, **k: calls.append((a, k)),
        )
        assert verdict["status"] == "breach"
        assert verdict["mode"] == "observe"
        assert verdict["paged"] is False
        assert calls == []  # nobody paged

    def test_ok_status_when_in_band(self):
        verdict = evaluate_parity_alarms({"sortino_ratio": 0.01, "psr": 0.02})
        assert verdict["status"] == "ok"
        assert verdict["n_breaches"] == 0
        assert verdict["breached_metrics"] == []


class TestPagingMode:
    def test_paging_publishes_on_breach(self):
        calls = []

        def fake_publish(message, severity=None, source=None):
            calls.append({"message": message, "severity": severity, "source": source})

        verdict = evaluate_parity_alarms(
            {"sortino_ratio": 0.4, "total_alpha": -0.10},
            paging_enabled=True,
            publish_fn=fake_publish,
            run_date="2026-06-13",
        )
        assert verdict["status"] == "breach"
        assert verdict["mode"] == "paging"
        assert verdict["paged"] is True
        assert len(calls) == 1
        assert calls[0]["severity"] == "error"
        assert "sortino_ratio" in calls[0]["message"]

    def test_paging_no_publish_when_ok(self):
        calls = []
        verdict = evaluate_parity_alarms(
            {"sortino_ratio": 0.0},
            paging_enabled=True,
            publish_fn=lambda *a, **k: calls.append(a),
        )
        assert verdict["status"] == "ok"
        assert verdict["paged"] is False
        assert calls == []

    def test_env_opt_out_suppresses_paging(self, monkeypatch):
        monkeypatch.setenv(_ALERT_DISABLED_ENV_VAR, "1")
        calls = []
        verdict = evaluate_parity_alarms(
            {"sortino_ratio": 0.5},
            paging_enabled=True,
            publish_fn=lambda *a, **k: calls.append(a),
        )
        assert verdict["status"] == "breach"
        assert verdict["paged"] is False
        assert calls == []

    def test_publish_failure_is_swallowed(self):
        def boom(*a, **k):
            raise RuntimeError("transport down")

        verdict = evaluate_parity_alarms(
            {"sortino_ratio": 0.5}, paging_enabled=True, publish_fn=boom,
        )
        # Breach still reported; paging best-effort, swallowed → paged False.
        assert verdict["status"] == "breach"
        assert verdict["paged"] is False


class TestStepChangeInVerdict:
    def test_step_breach_alone_flags_breach(self):
        # Level in-band, but a large jump vs prior ⇒ breach via step channel.
        verdict = evaluate_parity_alarms(
            {"sortino_ratio": 0.08},                 # within 0.10 band
            prior_delta={"sortino_ratio": -0.30},    # step 0.38 > 0.20 step band
        )
        assert verdict["status"] == "breach"
        assert "sortino_ratio" in verdict["step_breaches"]
        assert "sortino_ratio" not in verdict["band_breaches"]


class TestReportIntegration:
    def test_contamination_report_carries_observe_alarms(self):
        from analysis.pit_parity import build_contamination_report

        # current vs pit stats with a large Sortino gap ⇒ band breach in observe.
        cur = {"sortino_ratio": 1.0, "psr": 0.6, "cvar_95": -0.02,
               "max_drawdown": -0.10, "total_return": 0.05, "total_alpha": 0.01}
        pit = {"sortino_ratio": 0.5, "psr": 0.55, "cvar_95": -0.03,
               "max_drawdown": -0.12, "total_return": 0.04, "total_alpha": 0.00}
        report = build_contamination_report(cur, pit, run_date="2026-06-13")

        assert "alarms" in report
        alarms = report["alarms"]
        assert alarms["mode"] == "observe"
        assert alarms["paged"] is False
        # ΔSortino = 0.5 − 1.0 = −0.5, outside the 0.10 band.
        assert alarms["status"] == "breach"
        assert "sortino_ratio" in alarms["band_breaches"]
