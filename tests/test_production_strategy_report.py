"""Tests for backtest.run_production_strategy_backtest (config#1053).

The deployed-strategy backtest that headlines the weekly report. Must be
FAIL-LOUD (returns a status dict the reporter banners) and NEVER crash the run.
"""

from __future__ import annotations

import types

import pytest

import backtest as B


class _FakeOpt:
    def __init__(self, daily=False):
        self.metrics = {"total_alpha": 0.01, "sortino_ratio": 1.2, "total_return": 0.03}
        self.n_rebalances = 13
        self.n_solver_failures = 0
        # run_production_strategy_backtest reads these to build the
        # risk-matched headline. Default None (the fixtures don't exercise
        # the beta-matched math — that's covered by the dedicated tests
        # against real Series below).
        self.portfolio_daily_returns = None
        self.spy_daily_returns = None


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
    # Risk-matched headline slot always present; insufficient_data here since
    # the fake opt returns no daily-return series.
    assert out["risk_matched"]["status"] == "insufficient_data"


class TestComputeRiskMatchedHeadline:
    """config#1053 part 2 — beta-matched SPY risk-matched headline."""

    def _series(self, n, seed):
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2026-03-13", periods=n)
        return pd.Series(rng.normal(0.0005, 0.01, size=n), index=idx)

    def test_none_inputs_insufficient(self):
        out = B._compute_risk_matched_headline(None, None)
        assert out["status"] == "insufficient_data"

    def test_short_window_insufficient(self):
        # Fewer days than the beta lookback + 1 → insufficient, fail soft.
        port = self._series(10, 1)
        spy = self._series(10, 2)
        out = B._compute_risk_matched_headline(port, spy)
        assert out["status"] == "insufficient_data"
        assert "beta lookback" in out["note"]

    def test_sufficient_window_computes_ir(self):
        # A window comfortably longer than the 20d lookback yields an "ok"
        # result with an information ratio + excess return.
        spy = self._series(80, 3)
        # Portfolio = 0.5*SPY beta + idiosyncratic noise → real residual alpha.
        import numpy as np
        port = 0.5 * spy + self._series(80, 4) * 0.5
        out = B._compute_risk_matched_headline(port, spy)
        assert out["status"] == "ok"
        assert out["beta_lookback_days"] == 20
        assert out["information_ratio"] is not None
        assert out["excess_return"] is not None
        assert isinstance(out["n_days"], int) and out["n_days"] > 0


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
