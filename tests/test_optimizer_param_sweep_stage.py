"""Tests for ``backtest.run_optimizer_param_sweep_stage`` (config#1057 inc 1).

Mirrors the shape of ``test_cov_estimator_sweep_stage.py`` /
``test_gamma_sweep_stage.py``: production inputs → sweep harness → S3
persist (+ best-effort recommend/apply). Tests here focus on the new
config#2454 surface — the ``n_trials`` field and the cumulative
trial-count accumulator increment — since the pre-existing wiring
(recommend/apply, S3 persist) predates this issue and isn't otherwise
covered by a dedicated file.
"""

from __future__ import annotations

import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def stub_config():
    tmp = tempfile.mkdtemp(prefix="optimizer_param_sweep_stage_test_")
    return {
        "signals_bucket": "alpha-engine-research",
        "executor_paths": [tmp],
    }


@pytest.fixture
def stub_inputs():
    return {
        "status": "ok",
        "predictions_by_date": {"2026-05-27": {"AAPL": 0.01}},
        "price_matrix": MagicMock(),
        "spy_prices": MagicMock(),
        "sector_map": {"AAPL": "Technology"},
        "production_window": "2026-01-01:2026-05-27",
        "n_production_dates": 90,
    }


@pytest.fixture
def stub_sweep_report():
    """9-cell default risk_aversion x tcost_bps grid shape."""
    return {
        "cells": {f"cell_{i}": {"sortino_ratio": 0.1 * i} for i in range(9)},
        "baseline_name": "baseline_ra5_tc5",
        "winner_name": None,
        "gate_passes_per_cell": {f"cell_{i}": False for i in range(9)},
        "ranking": [("baseline_ra5_tc5", 0.5)],
        "cells_with_solver_failures": {},
    }


def _patched(stub_inputs, stub_sweep_report, s3_client, extra=None):
    patches = [
        patch(
            "synthetic.production_signal_backtest.build_production_signal_inputs",
            return_value=stub_inputs,
        ),
        patch(
            "analysis.portfolio_optimizer_backtest.run_optimizer_param_sweep",
            return_value=stub_sweep_report,
        ),
        # recommend/apply are best-effort advisory calls unrelated to this
        # issue's scope — stub them out so the happy path doesn't depend on
        # optimizer.portfolio_optimizer_optimizer internals.
        patch(
            "optimizer.portfolio_optimizer_optimizer.recommend",
            return_value={"status": "ok"},
        ),
        patch(
            "optimizer.portfolio_optimizer_optimizer.apply",
            return_value={"applied": False},
        ),
    ]
    return patches


class TestHappyPathTrialCount:
    def test_n_trials_persisted_and_matches_cell_count(
        self, stub_config, stub_inputs, stub_sweep_report
    ):
        from backtest import run_optimizer_param_sweep_stage

        s3_client = MagicMock()
        with patch(
            "synthetic.production_signal_backtest.build_production_signal_inputs",
            return_value=stub_inputs,
        ), patch(
            "analysis.portfolio_optimizer_backtest.run_optimizer_param_sweep",
            return_value=stub_sweep_report,
        ), patch(
            "optimizer.portfolio_optimizer_optimizer.recommend",
            return_value={"status": "ok"},
        ), patch(
            "optimizer.portfolio_optimizer_optimizer.apply",
            return_value={"applied": False},
        ):
            payload = run_optimizer_param_sweep_stage(
                config=stub_config, run_date="2026-05-27", s3_client=s3_client,
            )

        assert payload["status"] == "ok"
        assert payload["n_trials"] == 9

        put_calls = [
            c for c in s3_client.put_object.call_args_list
            if c.kwargs.get("Key") == "backtest/2026-05-27/optimizer_param_sweep.json"
        ]
        assert put_calls
        body_json = json.loads(put_calls[0].kwargs["Body"].decode("utf-8"))
        assert body_json["n_trials"] == 9


class TestTrialCountAccumulator:
    def test_increments_on_success(self, stub_config, stub_inputs, stub_sweep_report):
        from backtest import run_optimizer_param_sweep_stage

        with patch(
            "synthetic.production_signal_backtest.build_production_signal_inputs",
            return_value=stub_inputs,
        ), patch(
            "analysis.portfolio_optimizer_backtest.run_optimizer_param_sweep",
            return_value=stub_sweep_report,
        ), patch(
            "optimizer.portfolio_optimizer_optimizer.recommend",
            return_value={"status": "ok"},
        ), patch(
            "optimizer.portfolio_optimizer_optimizer.apply",
            return_value={"applied": False},
        ), patch(
            "nousergon_lib.quant.stats.trial_accumulator.increment_trial_count"
        ) as mock_incr:
            run_optimizer_param_sweep_stage(
                config=stub_config, run_date="2026-05-27", s3_client=MagicMock(),
            )

        mock_incr.assert_called_once()
        args, _ = mock_incr.call_args
        assert args[0] == "optimizer_param_sweep"
        assert args[1] == 9
        assert args[2] == "2026-05-27"

    def test_skipped_cycle_does_not_increment(self, stub_config, stub_sweep_report):
        from backtest import run_optimizer_param_sweep_stage

        with patch(
            "synthetic.production_signal_backtest.build_production_signal_inputs",
            return_value={"status": "no_data"},
        ), patch(
            "nousergon_lib.quant.stats.trial_accumulator.increment_trial_count"
        ) as mock_incr:
            payload = run_optimizer_param_sweep_stage(
                config=stub_config, run_date="2026-05-27", s3_client=MagicMock(),
            )

        assert payload["status"] == "skipped"
        mock_incr.assert_not_called()

    def test_accumulator_failure_does_not_raise(
        self, stub_config, stub_inputs, stub_sweep_report, caplog
    ):
        from backtest import run_optimizer_param_sweep_stage

        with patch(
            "synthetic.production_signal_backtest.build_production_signal_inputs",
            return_value=stub_inputs,
        ), patch(
            "analysis.portfolio_optimizer_backtest.run_optimizer_param_sweep",
            return_value=stub_sweep_report,
        ), patch(
            "optimizer.portfolio_optimizer_optimizer.recommend",
            return_value={"status": "ok"},
        ), patch(
            "optimizer.portfolio_optimizer_optimizer.apply",
            return_value={"applied": False},
        ), patch(
            "nousergon_lib.quant.stats.trial_accumulator.increment_trial_count",
            side_effect=RuntimeError("contention"),
        ):
            payload = run_optimizer_param_sweep_stage(
                config=stub_config, run_date="2026-05-27", s3_client=MagicMock(),
            )

        assert payload["status"] == "ok"
        assert any(
            "trial-count increment failed" in record.message
            for record in caplog.records
        )
