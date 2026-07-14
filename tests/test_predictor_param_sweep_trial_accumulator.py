"""Tests for the cumulative trial-count accumulator gating inside
``backtest.run_predictor_param_sweep`` (config#2454, producer #4).

This producer is the one with the "gotcha" the issue calls out: it runs
inside a ``registry.phase("predictor_param_sweep", supports_auto_skip=True)``
block, and CAN auto-skip — reusing a prior marker's ``sweep_df`` instead of
running new trials — when the phase was already successful for this
``run_date`` on a previous attempt (retry-safe resumability, ROADMAP
Backtester P0). A skipped/reused cycle generates ZERO new trials, so the
accumulator must NOT be incremented on that branch, or the cumulative count
would double-count the same combos every time the phase auto-skips.

Rather than driving the ~400-line ``run_predictor_param_sweep`` through
real marker-JSON skip detection end-to-end (which is fragile to set up
correctly and, per test_skip_phase4_flag.py's own comment, the existing
suite avoids), these tests force ``PhaseRegistry.should_run`` per-phase so
every phase BEFORE the target one always "runs" (real code path, all
compute-heavy collaborators mocked cheap) and only the target
``predictor_param_sweep`` phase's skip/no-skip is toggled per test. This
exercises the REAL ``with registry.phase("predictor_param_sweep", ...)``
context manager and the REAL ``if ps_ctx.skipped: ... else: ...`` branch in
backtest.py — not a re-implementation of the gating logic — while sidestepping
the brittleness of hand-crafting marker JSON bodies for every upstream phase.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pipeline_common import PhaseRegistry
from tests.test_phase_registry import _FakeS3

import backtest as bt


@pytest.fixture
def fake_executor_module(tmp_path, monkeypatch):
    """Same pattern as test_simulation_setup_auto_skip.py's fixture: a
    stub `executor` package on sys.path so backtest.py's `from
    executor.main import run` / `from executor.ibkr import
    SimulatedIBKRClient` / `from executor.feature_lookup import
    FeatureLookup` succeed without the real alpha-engine repo checked out.
    """
    pkg_dir = tmp_path / "executor"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "main.py").write_text("def run(*a, **kw): return []\n")
    (pkg_dir / "ibkr.py").write_text(
        "class SimulatedIBKRClient:\n    def __init__(self, **kw): pass\n"
    )
    (pkg_dir / "feature_lookup.py").write_text(
        "class FeatureLookup:\n"
        "    atr_dollar = {}\n"
        "    rsi = {}\n"
        "    momentum_20d_pct = {}\n"
        "    returns = {}\n"
        "    support_20_low = {}\n"
        "    @classmethod\n"
        "    def from_ohlcv_by_ticker(cls, *a, **kw): return cls()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    for mod in list(sys.modules):
        if mod.startswith("executor"):
            del sys.modules[mod]
    yield str(tmp_path)
    for mod in list(sys.modules):
        if mod.startswith("executor"):
            del sys.modules[mod]


def _predictor_pipeline_result():
    dates = ["2026-01-01", "2026-01-02"]
    price_matrix = pd.DataFrame(
        {"AAPL": [100.0, 101.0]}, index=pd.to_datetime(dates),
    )
    return {
        "status": "ok",
        "signals_by_date": {},
        "price_matrix": price_matrix,
        "ohlcv_by_ticker": {},
        "spy_prices": None,
        "metadata": {"n_tickers": 1, "n_dates": 2},
        "sector_map": {},
        "trading_dates": [],
        "predictions_by_date": {},
    }


def _drive_run_predictor_param_sweep(fake_executor_module, *, sweep_phase_skipped: bool):
    """Shared driver: forces should_run() to make every phase EXCEPT
    predictor_param_sweep always run (real code path, cheap mocks), and
    forces predictor_param_sweep itself to the requested skip state.
    Returns the mock_incr Mock so the caller can assert on it.
    """
    s3 = _FakeS3()
    run_date = "2026-04-23"
    bucket = "test-bucket"
    registry = PhaseRegistry(date=run_date, bucket=bucket, s3_client=s3)

    real_should_run = registry.should_run

    def _forced_should_run(phase_name, supports_auto_skip=False):
        if phase_name == "predictor_param_sweep":
            if sweep_phase_skipped:
                return False, "auto_skip_marker_ok"
            return True, "default_run"
        # Every other phase: always run for real (no skip-load branch to
        # fake), regardless of supports_auto_skip.
        return True, "default_run"

    pred_result = _predictor_pipeline_result()
    config = {
        "executor_paths": [fake_executor_module],
        "predictor_paths": [],
        "signals_bucket": bucket,
        "skip_phase4_evaluations": True,
        "param_sweep": {"x": [1, 2, 3]},
        "_phase_registry": registry,
    }

    # When predictor_param_sweep is forced to skip, backtest.py's skip
    # branch calls registry.load_marker(...) then, if artifact_keys are
    # present, phase_artifacts.load_dataframe(...) to reconstruct sweep_df.
    # Pre-seed a marker so that path resolves instead of falling through to
    # the pd.DataFrame() empty fallback (either way len is 0 trials, but we
    # want a >0-row df on the skip path specifically to prove the assertion
    # isn't vacuously true from an always-empty df).
    if sweep_phase_skipped:
        import json
        marker_key = f"backtest/{run_date}/.phases/predictor_param_sweep.json"
        artifact_key = f"backtest/{run_date}/.phases/predictor_param_sweep/sweep_df.parquet"
        s3.store[(bucket, marker_key)] = json.dumps({
            "schema_version": 1, "status": "ok",
            "artifact_keys": [artifact_key],
        }).encode()

    with patch.object(registry, "should_run", side_effect=_forced_should_run), \
         patch("synthetic.predictor_backtest.run", return_value=pred_result), \
         patch.object(bt, "_load_predictor_data_prep", return_value=pred_result), \
         patch.object(bt, "_save_predictor_data_prep", MagicMock()), \
         patch.object(bt, "_load_predictor_feature_maps", return_value=({}, {}, {})), \
         patch.object(bt, "_save_predictor_feature_maps", MagicMock()), \
         patch.object(bt, "_run_simulation_loop", MagicMock(return_value={"trades": []})), \
         patch.object(bt, "_precompute_signal_lookups", MagicMock(return_value={})), \
         patch.object(bt, "_build_pit_universe_resolver", MagicMock(return_value=None)), \
         patch.object(bt, "_try_construct_ew_high_vol_basket", MagicMock(return_value=None)), \
         patch.object(bt, "_seed_grid_with_current", lambda grid, current: grid), \
         patch.object(bt, "read_params_pit_or_current", MagicMock(return_value={})), \
         patch("store.feature_maps.load_precomputed_feature_maps", MagicMock(return_value=({}, {}, {}))), \
         patch("nousergon_lib.arcticdb.get_universe_symbols", MagicMock(return_value=["AAPL"])), \
         patch("phase_artifacts.load_dataframe", return_value=pd.DataFrame({"combo": [1, 2, 3]})), \
         patch("phase_artifacts.save_dataframe", return_value="fake-key"), \
         patch.object(
             bt.param_sweep, "sweep",
             return_value=pd.DataFrame({"combo": [1, 2, 3], "sortino_ratio": [0.1, 0.2, 0.3]}),
         ), \
         patch(
             "nousergon_lib.quant.stats.trial_accumulator.increment_trial_count"
         ) as mock_incr:
        single_stats, sweep_df = bt.run_predictor_param_sweep(config)

    return mock_incr, sweep_df


class TestSkippedCycleDoesNotIncrement:
    def test_skipped_sweep_phase_never_calls_increment(self, fake_executor_module):
        """ps_ctx.skipped=True (phase auto-skip, reusing a prior marker's
        sweep_df) must NOT call increment_trial_count — a reused cycle
        generated zero new trials."""
        mock_incr, sweep_df = _drive_run_predictor_param_sweep(
            fake_executor_module, sweep_phase_skipped=True,
        )
        # Sanity: the skip path actually loaded a non-empty prior sweep_df
        # (proves this isn't vacuously passing off an always-empty frame).
        assert len(sweep_df) == 3
        mock_incr.assert_not_called()


class TestRealRunIncrementsWithRowCount:
    def test_non_skipped_sweep_phase_increments_with_len_sweep_df(self, fake_executor_module):
        """ps_ctx.skipped=False (a real sweep ran) MUST call
        increment_trial_count with producer='predictor_param_sweep' and
        n_trials=len(sweep_df) — one row per evaluated combo."""
        mock_incr, sweep_df = _drive_run_predictor_param_sweep(
            fake_executor_module, sweep_phase_skipped=False,
        )
        assert len(sweep_df) == 3
        mock_incr.assert_called_once()
        args, kwargs = mock_incr.call_args
        assert args[0] == "predictor_param_sweep"
        assert args[1] == 3
        assert args[2] == "2026-04-23"
