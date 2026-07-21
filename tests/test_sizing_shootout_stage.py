"""tests/test_sizing_shootout_stage.py — config#3081 pipeline-wiring test.

Mirrors ``tests/test_double_sort_stage.py``'s approach: ``run_predictor_backtest``
stubs the heavy GBM/executor layers and this file verifies only the
sizing_shootout STAGE WIRING contract:

1. enabled (default) + a price_matrix present -> run_sizing_shootout +
   compute_sizing_shootout are exercised and the result lands in
   stats["sizing_shootout"] and is persisted to
   backtest/{run_date}/sizing_shootout.json.
2. config["sizing_shootout"]["enabled"] = False -> stage skipped entirely.
3. no price_matrix -> stage skipped.
4. a failure anywhere in the stage is swallowed -- the backtest run's
   stats/status must not be affected (OBSERVE-only, ARCHITECTURE.md
   Section 14(e)).
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _install_fake_executor_modules():
    """Stub the executor package the same way test_double_sort_stage.py
    does, PLUS ``executor.feature_lookup.FeatureLookup`` which the
    sizing_shootout stage additionally imports (no real executor
    checkout exists in this sandbox)."""
    main_mod = types.ModuleType("executor.main")
    main_mod.run = MagicMock(name="executor_run")
    ibkr_mod = types.ModuleType("executor.ibkr")
    ibkr_mod.SimulatedIBKRClient = MagicMock(name="SimulatedIBKRClient")

    feature_lookup_mod = types.ModuleType("executor.feature_lookup")

    class _FakeFeatureLookup:
        """Minimal FeatureLookup stand-in — the four dict-attributes
        ``build_feature_matrices`` reads (see
        tests/test_vectorized_sweep.py FakeFeatureLookup)."""
        def __init__(self, atr_dollar, rsi, momentum_20d_pct, returns):
            self.atr_dollar = atr_dollar
            self.rsi = rsi
            self.momentum_20d_pct = momentum_20d_pct
            self.returns = returns

        @classmethod
        def from_ohlcv_by_ticker(cls, ohlcv_by_ticker, **kwargs):
            atr, rsi, mom, rets = {}, {}, {}, {}
            for ticker, df in (ohlcv_by_ticker or {}).items():
                close = df["close"]
                atr[ticker] = pd.Series(np.full(len(close), 1.5), index=close.index)
                rsi[ticker] = pd.Series(np.full(len(close), 50.0), index=close.index)
                mom[ticker] = (close.pct_change(periods=20) * 100.0)
                rets[ticker] = close.pct_change()
            return cls(atr, rsi, mom, rets)

    feature_lookup_mod.FeatureLookup = _FakeFeatureLookup

    executor_pkg = types.ModuleType("executor")
    sys.modules["executor"] = executor_pkg
    sys.modules["executor.main"] = main_mod
    sys.modules["executor.ibkr"] = ibkr_mod
    sys.modules["executor.feature_lookup"] = feature_lookup_mod


def _base_config(tmp_path, **extra):
    executor_dir = tmp_path / "executor_repo"
    executor_dir.mkdir()
    cfg = {
        "executor_paths": [str(executor_dir)],
        "signals_bucket": "test-bucket",
        "_run_date": "2026-07-20",
    }
    cfg.update(extra)
    return cfg


def _fake_predictor_result(n=60):
    idx = pd.bdate_range("2026-01-01", periods=n)
    cols = ["AAPL", "MSFT", "SPY"]
    price_matrix = pd.DataFrame(100.0, index=idx, columns=cols)
    ohlcv_by_ticker = {
        t: pd.DataFrame(
            {
                "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
            },
            index=idx,
        )
        for t in cols
    }
    signals_by_date = {}
    for i, d in enumerate(idx):
        ds = d.strftime("%Y-%m-%d")
        # Emit an ENTER signal for AAPL every 5th date (past the 20d
        # warmup so realized-vol has history) so the vectorized sweep
        # actually produces entries for the shootout to score.
        if i >= 22 and i % 5 == 0:
            signals_by_date[ds] = {
                "universe": [],
                "buy_candidates": [
                    {
                        "ticker": "AAPL", "signal": "ENTER", "score": 80,
                        "sector": "Technology", "sector_rating": "market_weight",
                        "conviction": "stable", "price_target_upside": 0.20,
                    }
                ],
                "market_regime": "neutral",
            }
        else:
            signals_by_date[ds] = {"universe": [], "buy_candidates": []}
    return {
        "status": "ok",
        "signals_by_date": signals_by_date,
        "price_matrix": price_matrix,
        "ohlcv_by_ticker": ohlcv_by_ticker,
        "spy_prices": price_matrix["SPY"],
        "metadata": {},
        "sector_map": {"AAPL": "Technology", "MSFT": "Technology"},
        "predictions_by_date": {ds: {"AAPL": 0.01} for ds in signals_by_date},
    }


@pytest.fixture(autouse=True)
def _fake_executor():
    _install_fake_executor_modules()
    yield
    sys.modules.pop("executor", None)
    sys.modules.pop("executor.main", None)
    sys.modules.pop("executor.ibkr", None)
    sys.modules.pop("executor.feature_lookup", None)


def _run_with_stubs(config, *, s3_client=None):
    fake_result = _fake_predictor_result()
    with (
        patch("synthetic.predictor_backtest.run", return_value=fake_result),
        patch("backtest._run_simulation_loop", return_value={"status": "ok"}),
        patch("os.path.isdir", return_value=True),
        patch(
            "analysis.double_sort.load_predictions_by_horizon_panel",
            return_value={},
        ),
        patch("boto3.client") as mock_boto_client,
    ):
        mock_s3 = s3_client or MagicMock()
        mock_boto_client.return_value = mock_s3
        import backtest
        stats = backtest.run_predictor_backtest(config)
    return stats, mock_s3


class TestSizingShootoutStageWiring:
    def test_enabled_computes_and_persists(self, tmp_path):
        config = _base_config(tmp_path)
        stats, mock_s3 = _run_with_stubs(config)

        assert "sizing_shootout" in stats
        assert stats["sizing_shootout"]["status"] == "ok"
        assert "arms" in stats["sizing_shootout"]
        assert "conviction" in stats["sizing_shootout"]["arms"]
        assert "risk_parity" in stats["sizing_shootout"]["arms"]
        assert any(
            k.startswith("fractional_kelly")
            for k in stats["sizing_shootout"]["arms"]
        )

        put_calls = list(mock_s3.put_object.call_args_list)
        keys = [c.kwargs.get("Key") for c in put_calls]
        assert "backtest/2026-07-20/sizing_shootout.json" in keys

    def test_disabled_flag_skips_stage(self, tmp_path):
        config = _base_config(tmp_path, sizing_shootout={"enabled": False})
        stats, mock_s3 = _run_with_stubs(config)

        assert "sizing_shootout" not in stats
        keys = [c.kwargs.get("Key") for c in mock_s3.put_object.call_args_list]
        assert "backtest/2026-07-20/sizing_shootout.json" not in keys

    def test_missing_price_matrix_skips_stage(self, tmp_path):
        config = _base_config(tmp_path)
        fake_result = _fake_predictor_result()
        fake_result["price_matrix"] = None
        with (
            patch("synthetic.predictor_backtest.run", return_value=fake_result),
            patch("backtest._run_simulation_loop", return_value={"status": "ok"}),
            patch("os.path.isdir", return_value=True),
            patch("analysis.double_sort.load_predictions_by_horizon_panel", return_value={}),
            patch("boto3.client", return_value=MagicMock()),
        ):
            import backtest
            stats = backtest.run_predictor_backtest(config)

        assert "sizing_shootout" not in stats

    def test_stage_failure_is_swallowed(self, tmp_path):
        """A broken shootout run must not abort the backtest -- OBSERVE-only,
        fail-soft (ARCHITECTURE.md Section 14(e))."""
        config = _base_config(tmp_path)
        with (
            patch("synthetic.predictor_backtest.run", return_value=_fake_predictor_result()),
            patch("backtest._run_simulation_loop", return_value={"status": "ok"}),
            patch("os.path.isdir", return_value=True),
            patch("analysis.double_sort.load_predictions_by_horizon_panel", return_value={}),
            patch("boto3.client", return_value=MagicMock()),
            patch(
                "synthetic.vectorized_sweep.run_sizing_shootout",
                side_effect=RuntimeError("boom"),
            ),
        ):
            import backtest
            stats = backtest.run_predictor_backtest(config)

        assert "sizing_shootout" not in stats
        assert stats.get("status") != "error"

    def test_s3_put_failure_is_swallowed_but_stats_populated(self, tmp_path):
        config = _base_config(tmp_path)
        mock_s3 = MagicMock()
        mock_s3.put_object.side_effect = RuntimeError("S3 PutObject failed")

        stats, _ = _run_with_stubs(config, s3_client=mock_s3)

        assert "sizing_shootout" in stats
        assert stats["sizing_shootout"]["status"] == "ok"
