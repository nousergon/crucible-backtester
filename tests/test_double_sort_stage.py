"""tests/test_double_sort_stage.py — W3.3 (config#1993) pipeline-wiring test.

``run_predictor_backtest`` in backtest.py is the orchestration entry point
that stitches GBM inference -> executor simulation -> the OBSERVE-only
diagnostic stages (horizon_net_alpha, model_version_net_alpha, and now
double_sort). Exercising it end-to-end needs a real executor checkout on
``executor_paths`` and ArcticDB-backed price history, neither of which is
available in a unit-test sandbox — so, matching the existing precedent in
tests/test_predictor_run_callback.py, this stubs the heavy GBM/executor
steps and verifies only the double_sort STAGE WIRING contract:

1. enabled (default) + a price_matrix present -> compute_double_sort is
   called with the S3-loaded/reshaped panel and the run's already-loaded
   price_matrix/spy_prices; the result lands in stats["double_sort"] and is
   persisted to backtest/{run_date}/double_sort.json.
2. config["double_sort"]["enabled"] = False -> stage skipped entirely.
3. no price_matrix -> stage skipped (matches the model_version_net_alpha
   guard style).
4. a failure anywhere in the stage (panel load, compute, or the S3 PUT) is
   swallowed -- the backtest run's stats/status must not be affected
   (OBSERVE-only, ARCHITECTURE.md Section 14(e)).
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _install_fake_executor_modules():
    """``run_predictor_backtest`` does ``from executor.main import run`` /
    ``from executor.ibkr import SimulatedIBKRClient`` after inserting
    ``executor_paths`` onto sys.path. No real executor checkout exists in
    this sandbox, so inject minimal stub modules directly into
    sys.modules -- the import machinery finds them there before ever
    touching the filesystem."""
    main_mod = types.ModuleType("executor.main")
    main_mod.run = MagicMock(name="executor_run")
    ibkr_mod = types.ModuleType("executor.ibkr")
    ibkr_mod.SimulatedIBKRClient = MagicMock(name="SimulatedIBKRClient")
    executor_pkg = types.ModuleType("executor")
    sys.modules["executor"] = executor_pkg
    sys.modules["executor.main"] = main_mod
    sys.modules["executor.ibkr"] = ibkr_mod


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
    return {
        "status": "ok",
        "signals_by_date": {d.strftime("%Y-%m-%d"): {"buy_candidates": []} for d in idx},
        "price_matrix": price_matrix,
        "ohlcv_by_ticker": {},
        "spy_prices": price_matrix["SPY"],
        "metadata": {},
        "predictions_by_date": {d.strftime("%Y-%m-%d"): {"AAPL": 0.01} for d in idx},
    }


@pytest.fixture(autouse=True)
def _fake_executor():
    _install_fake_executor_modules()
    yield
    sys.modules.pop("executor", None)
    sys.modules.pop("executor.main", None)
    sys.modules.pop("executor.ibkr", None)


def _run_with_stubs(config, *, panel=None, panel_side_effect=None, s3_client=None):
    """Drive run_predictor_backtest with the GBM/executor layers stubbed
    out, patching only the double_sort stage's two collaborators."""
    fake_result = _fake_predictor_result()
    with (
        patch("synthetic.predictor_backtest.run", return_value=fake_result),
        patch("backtest._run_simulation_loop", return_value={"status": "ok"}),
        patch("os.path.isdir", return_value=True),
        patch(
            "analysis.double_sort.load_predictions_by_horizon_panel",
            return_value=panel if panel is not None else {},
            side_effect=panel_side_effect,
        ) as mock_load,
        patch("boto3.client") as mock_boto_client,
    ):
        mock_s3 = s3_client or MagicMock()
        mock_boto_client.return_value = mock_s3
        import backtest
        stats = backtest.run_predictor_backtest(config)
    return stats, mock_load, mock_s3


class TestDoubleSortStageWiring:
    def test_enabled_calls_compute_and_persists(self, tmp_path):
        config = _base_config(tmp_path)
        panel = {21: {"2026-01-02": {"AAPL": 1.0, "MSFT": 0.0}}}
        stats, mock_load, mock_s3 = _run_with_stubs(config, panel=panel)

        mock_load.assert_called_once()
        called_bucket, called_key = mock_load.call_args[0]
        assert called_bucket == "test-bucket"
        assert called_key == "predictor/diagnostics/horizon_predictions/latest.parquet"

        assert "double_sort" in stats
        assert stats["double_sort"]["status"] == "ok"

        # persisted to the expected S3 key
        put_calls = [c for c in mock_s3.put_object.call_args_list]
        assert len(put_calls) >= 1
        keys = [c.kwargs.get("Key") for c in put_calls]
        assert "backtest/2026-07-20/double_sort.json" in keys

    def test_disabled_flag_skips_stage(self, tmp_path):
        config = _base_config(tmp_path, double_sort={"enabled": False})
        stats, mock_load, mock_s3 = _run_with_stubs(config, panel={21: {}})

        mock_load.assert_not_called()
        assert "double_sort" not in stats

    def test_missing_price_matrix_skips_stage(self, tmp_path):
        config = _base_config(tmp_path)
        fake_result = _fake_predictor_result()
        fake_result["price_matrix"] = None
        with (
            patch("synthetic.predictor_backtest.run", return_value=fake_result),
            patch("backtest._run_simulation_loop", return_value={"status": "ok"}),
            patch("os.path.isdir", return_value=True),
            patch("analysis.double_sort.load_predictions_by_horizon_panel") as mock_load,
            patch("boto3.client", return_value=MagicMock()),
        ):
            import backtest
            stats = backtest.run_predictor_backtest(config)

        mock_load.assert_not_called()
        assert "double_sort" not in stats

    def test_panel_load_failure_is_swallowed(self, tmp_path):
        """A broken/missing S3 panel must not abort the backtest run --
        OBSERVE-only, fail-soft (ARCHITECTURE.md Section 14(e))."""
        config = _base_config(tmp_path)
        stats, mock_load, mock_s3 = _run_with_stubs(
            config, panel_side_effect=RuntimeError("S3 GetObject failed: NoSuchKey"),
        )

        assert "double_sort" not in stats
        # the run itself still completed and returned normal stats
        assert stats.get("status") != "error"

    def test_s3_put_failure_is_swallowed_but_stats_populated(self, tmp_path):
        """The compute succeeds but the S3 PUT fails -- stats["double_sort"]
        must still be populated (in-memory result is not lost), matching
        the horizon_net_alpha / model_version_net_alpha dual-layer
        fail-soft pattern (inner try/except around just the PUT)."""
        config = _base_config(tmp_path)
        panel = {21: {"2026-01-02": {"AAPL": 1.0, "MSFT": 0.0}}}
        mock_s3 = MagicMock()
        mock_s3.put_object.side_effect = RuntimeError("S3 PutObject failed")

        stats, mock_load, _ = _run_with_stubs(config, panel=panel, s3_client=mock_s3)

        assert "double_sort" in stats
        assert stats["double_sort"]["status"] == "ok"
