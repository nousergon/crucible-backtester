"""Tests for ``backtest.run_cov_estimator_sweep_stage``.

ROADMAP A.4b — wires the covariance-estimator sweep harness shipped in
PR #248 (``analysis.portfolio_optimizer_backtest.run_cov_estimator_sweep``)
into ``backtest.py --mode all`` and persists the verdict to
``s3://{bucket}/backtest/{date}/cov_sweep.json``. Without this stage
the sweep verdict is operator-on-demand.

The stage mirrors the shape of ``run_portfolio_optimizer_gate``:
predictor backtest → sweep harness → S3 persist. Tests pin:

  * predictor-backtest failure path returns ``status=skipped`` (sweep
    cannot run without prediction inputs)
  * happy path persists to ``backtest/{date}/cov_sweep.json`` with the
    expected key shape
  * S3 persist failure is non-fatal (warns + returns payload)
  * the sweep harness is invoked with the predictor pipeline's outputs
    (predictions_by_date, price_matrix, spy_prices, sector_map,
    executor_path)
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def stub_config():
    """Config dict with at least one valid executor_paths entry (the
    sweep stage's path resolution requires the path to exist on disk)."""
    tmp = tempfile.mkdtemp(prefix="cov_sweep_stage_test_")
    return {
        "signals_bucket": "alpha-engine-research",
        "executor_paths": [tmp],
    }


@pytest.fixture
def stub_pred_inputs():
    """Minimal shape matching what ``synthetic.predictor_backtest.run``
    returns on ``status=ok``. The sweep harness receives these via
    keyword args; their internal types don't matter for the wiring
    test (the harness call itself is mocked)."""
    return {
        "status": "ok",
        "predictions_by_date": {"2026-05-27": {"AAPL": 0.01}},
        "price_matrix": MagicMock(),
        "spy_prices": MagicMock(),
        "sector_map": {"AAPL": "Technology"},
    }


@pytest.fixture
def stub_sweep_report():
    """Minimal shape matching ``run_cov_estimator_sweep``'s return."""
    return {
        "cells": {"LW_H1": {"sortino_ratio": 0.5}},
        "baseline_name": "LW_H1",
        "winner_name": None,
        "gate_passes_per_cell": {"LW_H1": False},
        "ranking": [("LW_H1", 0.5)],
        "cells_with_solver_failures": {"LW_H1": 0},
    }


# ── Happy path ──────────────────────────────────────────────────────────


class TestHappyPath:
    def test_persists_to_cov_sweep_key(
        self, stub_config, stub_pred_inputs, stub_sweep_report
    ):
        """Stage writes to ``s3://{bucket}/backtest/{date}/cov_sweep.json``."""
        from backtest import run_cov_estimator_sweep_stage

        s3_client = MagicMock()
        with patch(
            "synthetic.predictor_backtest.run", return_value=stub_pred_inputs
        ), patch(
            "analysis.portfolio_optimizer_backtest.run_cov_estimator_sweep",
            return_value=stub_sweep_report,
        ):
            payload = run_cov_estimator_sweep_stage(
                config=stub_config,
                run_date="2026-05-27",
                s3_client=s3_client,
            )

        assert payload["status"] == "ok"
        assert payload["run_date"] == "2026-05-27"
        # baseline + ranking carried through
        assert payload["baseline_name"] == "LW_H1"
        assert payload["ranking"] == [("LW_H1", 0.5)]

        # S3 put_object called with the expected key
        s3_client.put_object.assert_called_once()
        kwargs = s3_client.put_object.call_args.kwargs
        assert kwargs["Bucket"] == "alpha-engine-research"
        assert kwargs["Key"] == "backtest/2026-05-27/cov_sweep.json"
        assert kwargs["ContentType"] == "application/json"
        # Body decodes as JSON containing the report fields
        body_json = json.loads(kwargs["Body"].decode("utf-8"))
        assert body_json["baseline_name"] == "LW_H1"

    def test_harness_invoked_with_predictor_outputs(
        self, stub_config, stub_pred_inputs, stub_sweep_report
    ):
        """The sweep harness receives the predictor pipeline's outputs
        verbatim — predictions_by_date / price_matrix / spy_prices /
        sector_map all forwarded, plus executor_path resolved from
        config.executor_paths."""
        from backtest import run_cov_estimator_sweep_stage

        with patch(
            "synthetic.predictor_backtest.run", return_value=stub_pred_inputs
        ), patch(
            "analysis.portfolio_optimizer_backtest.run_cov_estimator_sweep",
            return_value=stub_sweep_report,
        ) as mock_sweep:
            run_cov_estimator_sweep_stage(
                config=stub_config,
                run_date="2026-05-27",
                s3_client=MagicMock(),
            )

        mock_sweep.assert_called_once()
        kwargs = mock_sweep.call_args.kwargs
        assert kwargs["predictions_by_date"] == stub_pred_inputs[
            "predictions_by_date"
        ]
        assert kwargs["price_matrix"] is stub_pred_inputs["price_matrix"]
        assert kwargs["spy_prices"] is stub_pred_inputs["spy_prices"]
        assert kwargs["sector_map"] == stub_pred_inputs["sector_map"]
        assert kwargs["executor_path"] == stub_config["executor_paths"][0]


# ── Predictor backtest skip ─────────────────────────────────────────────


class TestPredictorSkipPath:
    def test_predictor_failure_returns_skipped(
        self, stub_config, stub_sweep_report
    ):
        """If predictor backtest doesn't return status=ok the sweep
        cannot run — stage returns ``status=skipped`` with a reason,
        does not call the sweep harness, does not write to S3."""
        from backtest import run_cov_estimator_sweep_stage

        s3_client = MagicMock()
        with patch(
            "synthetic.predictor_backtest.run",
            return_value={"status": "error", "error": "boom"},
        ), patch(
            "analysis.portfolio_optimizer_backtest.run_cov_estimator_sweep",
            return_value=stub_sweep_report,
        ) as mock_sweep:
            payload = run_cov_estimator_sweep_stage(
                config=stub_config,
                run_date="2026-05-27",
                s3_client=s3_client,
            )

        assert payload["status"] == "skipped"
        assert "error" in payload["reason"]
        mock_sweep.assert_not_called()
        s3_client.put_object.assert_not_called()


# ── S3 persist failure ─────────────────────────────────────────────────


class TestS3PersistFailure:
    def test_s3_failure_does_not_raise(
        self, stub_config, stub_pred_inputs, stub_sweep_report, caplog
    ):
        """S3 persist is best-effort — failure warns and returns the
        payload anyway. Mirrors the run_portfolio_optimizer_gate
        try/except pattern. Important for spot-side runs where transient
        S3 throttling shouldn't abort the whole Backtester stage."""
        from backtest import run_cov_estimator_sweep_stage

        s3_client = MagicMock()
        s3_client.put_object.side_effect = RuntimeError("S3 outage")
        with patch(
            "synthetic.predictor_backtest.run", return_value=stub_pred_inputs
        ), patch(
            "analysis.portfolio_optimizer_backtest.run_cov_estimator_sweep",
            return_value=stub_sweep_report,
        ):
            payload = run_cov_estimator_sweep_stage(
                config=stub_config,
                run_date="2026-05-27",
                s3_client=s3_client,
            )
        # Stage still returned a payload (no crash)
        assert payload["status"] == "ok"
        # Warning emitted
        assert any(
            "S3 persist failed" in record.message
            for record in caplog.records
        )


# ── Config validation ───────────────────────────────────────────────────


class TestConfigValidation:
    def test_missing_executor_path_raises(self, stub_pred_inputs):
        """The sweep needs a valid executor_path on disk to import the
        optimizer kernel. Raise loud at construction time per
        no-silent-fails."""
        from backtest import run_cov_estimator_sweep_stage

        config = {
            "signals_bucket": "alpha-engine-research",
            "executor_paths": ["/nonexistent/path"],
        }
        with patch(
            "synthetic.predictor_backtest.run", return_value=stub_pred_inputs
        ):
            with pytest.raises(ValueError, match="executor_paths not found"):
                run_cov_estimator_sweep_stage(
                    config=config,
                    run_date="2026-05-27",
                    s3_client=MagicMock(),
                )
