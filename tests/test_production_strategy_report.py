"""Tests for backtest.run_production_strategy_backtest (config#1053).

The deployed-strategy backtest that headlines the weekly report. Must be
FAIL-LOUD (returns a status dict the reporter banners) and NEVER crash the run.
"""

from __future__ import annotations

import types

import pytest

import backtest as B


class _FakeOpt:
    def __init__(self):
        self.metrics = {"total_alpha": 0.01, "sortino_ratio": 1.2, "total_return": 0.03}
        self.n_rebalances = 13
        self.n_solver_failures = 0


def _patch_inputs(monkeypatch, status="ok", **extra):
    payload = {"status": status, **extra}
    if status == "ok":
        payload.update({
            "predictions_by_date": {"2026-06-12": {"AAA": 0.02}},
            "price_matrix": object(),
            "spy_prices": object(),
            "sector_map": {"AAA": "tech"},
            "production_window": "2026-03-13 → 2026-06-12",
            "n_production_dates": 62,
        })
    monkeypatch.setattr(
        "synthetic.production_signal_backtest.build_production_signal_inputs",
        lambda config, s3_client=None: payload,
    )


def test_happy_path_returns_metrics(monkeypatch, tmp_path):
    _patch_inputs(monkeypatch)
    monkeypatch.setattr(
        "analysis.portfolio_optimizer_backtest.run_optimizer_backtest",
        lambda **kw: _FakeOpt(),
    )
    cfg = {"executor_paths": [str(tmp_path)]}  # tmp_path exists on disk
    out = B.run_production_strategy_backtest(cfg)
    assert out["status"] == "ok"
    assert out["metrics"]["total_alpha"] == 0.01
    assert out["n_rebalances"] == 13
    assert out["production_window"] == "2026-03-13 → 2026-06-12"


def test_no_production_data_surfaces_status(monkeypatch, tmp_path):
    _patch_inputs(monkeypatch, status="no_production_data", error="no overlapping dates")
    out = B.run_production_strategy_backtest({"executor_paths": [str(tmp_path)]})
    assert out["status"] == "no_production_data"
    assert "no overlapping dates" in out["error"]


def test_missing_executor_path_is_error(monkeypatch):
    _patch_inputs(monkeypatch)
    out = B.run_production_strategy_backtest({"executor_paths": ["/does/not/exist"]})
    assert out["status"] == "error"
    assert "executor_paths" in out["error"]


def test_exception_is_caught_not_raised(monkeypatch, tmp_path):
    _patch_inputs(monkeypatch)

    def _boom(**kw):
        raise RuntimeError("cvxpy exploded")

    monkeypatch.setattr(
        "analysis.portfolio_optimizer_backtest.run_optimizer_backtest", _boom,
    )
    out = B.run_production_strategy_backtest({"executor_paths": [str(tmp_path)]})
    assert out["status"] == "error"
    assert "cvxpy exploded" in out["error"]
