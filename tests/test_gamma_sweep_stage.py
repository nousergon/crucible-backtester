"""Tests for ``backtest.run_gamma_sweep_stage``.

ROADMAP B.4b — wires the α̂-uncertainty γ-sweep harness shipped in
PR #250 (``analysis.portfolio_optimizer_backtest.run_gamma_sweep``)
into ``backtest.py --mode all`` and persists the verdict to
``s3://{bucket}/backtest/{date}/gamma_sweep.json``.

The stage handles the σ_α̂-coverage gating: γ-sweep against missing
uncertainty collapses every cell to baseline, so the stage auto-skips
when production-predictions archive coverage is sparse. Activates
automatically once predictor B.1 (BayesianRidge) has accumulated
enough Saturday cycles emitting non-None ``predicted_alpha_std``.

Tests pin:
  * predictor-backtest failure path returns ``status=skipped``
  * insufficient σ_α̂ coverage returns ``status=skipped`` with the
    coverage threshold reason (auto-activates once threshold cleared)
  * happy path persists to ``backtest/{date}/gamma_sweep.json``
  * S3 persist failure is non-fatal
  * the sweep harness receives both the predictor outputs AND the
    α-uncertainty mapping
  * the uncertainty loader skips dates without the archive file +
    skips entries with ``predicted_alpha_std=None``
"""

from __future__ import annotations

import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError


@pytest.fixture
def stub_config():
    tmp = tempfile.mkdtemp(prefix="gamma_sweep_stage_test_")
    return {
        "signals_bucket": "alpha-engine-research",
        "executor_paths": [tmp],
    }


@pytest.fixture
def stub_pred_inputs():
    """11 dates so default coverage threshold of 10 can be both met
    and missed deliberately in different tests."""
    return {
        "status": "ok",
        "predictions_by_date": {
            f"2026-05-{d:02d}": {"AAPL": 0.01} for d in range(1, 12)
        },
        "price_matrix": MagicMock(),
        "spy_prices": MagicMock(),
        "sector_map": {"AAPL": "Technology"},
    }


@pytest.fixture
def stub_sweep_report():
    return {
        "cells": {"gamma_0": {"sortino_ratio": 0.5}},
        "baseline_name": "gamma_0",
        "winner_name": None,
        "gate_passes_per_cell": {"gamma_0": False},
        "ranking": [("gamma_0", 0.5)],
        "cells_with_solver_failures": {"gamma_0": 0},
    }


def _make_s3_with_uncertainty(coverage_dates: list[str]) -> MagicMock:
    """Build an s3_client mock whose get_object returns a predictions
    archive doc with non-None predicted_alpha_std for the given dates.
    All other keys return a NoSuchKey ClientError so the loader treats
    them as uncovered."""
    s3 = MagicMock()
    by_key = {
        f"predictor/predictions/{d}.json": json.dumps(
            {"predictions": {"AAPL": {"predicted_alpha_std": 0.02}}}
        ).encode()
        for d in coverage_dates
    }

    def _get(Bucket, Key):
        if Key in by_key:
            body = MagicMock()
            body.read.return_value = by_key[Key]
            return {"Body": body}
        raise ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "..."}, "ResponseMetadata": {"HTTPStatusCode": 404}},
            "GetObject",
        )

    s3.get_object.side_effect = _get
    return s3


# ── Insufficient coverage skip ──────────────────────────────────────────


class TestInsufficientCoverageSkip:
    def test_zero_coverage_returns_skipped(
        self, stub_config, stub_pred_inputs, stub_sweep_report
    ):
        """No predicted_alpha_std in any archive file → skip the sweep
        entirely. The threshold today is 10 dates."""
        from backtest import run_gamma_sweep_stage

        s3_client = _make_s3_with_uncertainty(coverage_dates=[])
        with patch(
            "synthetic.predictor_backtest.run", return_value=stub_pred_inputs
        ), patch(
            "analysis.portfolio_optimizer_backtest.run_gamma_sweep",
            return_value=stub_sweep_report,
        ) as mock_sweep:
            payload = run_gamma_sweep_stage(
                config=stub_config,
                run_date="2026-05-27",
                s3_client=s3_client,
            )
        assert payload["status"] == "skipped"
        assert payload["n_dates_with_uncertainty"] == 0
        assert payload["n_target_dates"] == 11
        assert "insufficient σ_α̂ coverage" in payload["reason"]
        mock_sweep.assert_not_called()

    def test_below_threshold_returns_skipped(
        self, stub_config, stub_pred_inputs, stub_sweep_report
    ):
        """5 dates of coverage (below the 10-date threshold) → skip."""
        from backtest import run_gamma_sweep_stage

        coverage = [f"2026-05-{d:02d}" for d in range(1, 6)]
        s3_client = _make_s3_with_uncertainty(coverage_dates=coverage)
        with patch(
            "synthetic.predictor_backtest.run", return_value=stub_pred_inputs
        ), patch(
            "analysis.portfolio_optimizer_backtest.run_gamma_sweep",
            return_value=stub_sweep_report,
        ) as mock_sweep:
            payload = run_gamma_sweep_stage(
                config=stub_config,
                run_date="2026-05-27",
                s3_client=s3_client,
            )
        assert payload["status"] == "skipped"
        assert payload["n_dates_with_uncertainty"] == 5
        mock_sweep.assert_not_called()


# ── Happy path (coverage met) ──────────────────────────────────────────


class TestHappyPath:
    def test_threshold_met_runs_sweep(
        self, stub_config, stub_pred_inputs, stub_sweep_report
    ):
        """All 11 target dates covered → sweep runs, verdict persists."""
        from backtest import run_gamma_sweep_stage

        coverage = list(stub_pred_inputs["predictions_by_date"].keys())
        s3_client = _make_s3_with_uncertainty(coverage_dates=coverage)
        with patch(
            "synthetic.predictor_backtest.run", return_value=stub_pred_inputs
        ), patch(
            "analysis.portfolio_optimizer_backtest.run_gamma_sweep",
            return_value=stub_sweep_report,
        ) as mock_sweep:
            payload = run_gamma_sweep_stage(
                config=stub_config,
                run_date="2026-05-27",
                s3_client=s3_client,
            )
        assert payload["status"] == "ok"
        assert payload["n_dates_with_uncertainty"] == 11
        assert payload["n_target_dates"] == 11

        put_calls = [
            c for c in s3_client.put_object.call_args_list
            if c.kwargs.get("Key") == "backtest/2026-05-27/gamma_sweep.json"
        ]
        assert put_calls, "expected put_object on gamma_sweep.json key"
        body_json = json.loads(put_calls[0].kwargs["Body"].decode("utf-8"))
        assert body_json["baseline_name"] == "gamma_0"

        mock_sweep.assert_called_once()
        kwargs = mock_sweep.call_args.kwargs
        assert kwargs["predictions_by_date"] == stub_pred_inputs[
            "predictions_by_date"
        ]
        assert kwargs["alpha_uncertainty_by_date"]
        assert all(
            "AAPL" in v
            for v in kwargs["alpha_uncertainty_by_date"].values()
        )


# ── Predictor backtest skip ────────────────────────────────────────────


class TestPredictorSkipPath:
    def test_predictor_failure_returns_skipped(self, stub_config):
        from backtest import run_gamma_sweep_stage

        with patch(
            "synthetic.predictor_backtest.run",
            return_value={"status": "error", "error": "boom"},
        ), patch(
            "analysis.portfolio_optimizer_backtest.run_gamma_sweep",
        ) as mock_sweep:
            payload = run_gamma_sweep_stage(
                config=stub_config,
                run_date="2026-05-27",
                s3_client=MagicMock(),
            )
        assert payload["status"] == "skipped"
        assert "error" in payload["reason"]
        mock_sweep.assert_not_called()


# ── S3 persist failure ─────────────────────────────────────────────────


class TestS3PersistFailure:
    def test_s3_failure_does_not_raise(
        self, stub_config, stub_pred_inputs, stub_sweep_report, caplog
    ):
        from backtest import run_gamma_sweep_stage

        coverage = list(stub_pred_inputs["predictions_by_date"].keys())
        s3_client = _make_s3_with_uncertainty(coverage_dates=coverage)
        s3_client.put_object.side_effect = RuntimeError("S3 outage")
        with patch(
            "synthetic.predictor_backtest.run", return_value=stub_pred_inputs
        ), patch(
            "analysis.portfolio_optimizer_backtest.run_gamma_sweep",
            return_value=stub_sweep_report,
        ):
            payload = run_gamma_sweep_stage(
                config=stub_config,
                run_date="2026-05-27",
                s3_client=s3_client,
            )
        assert payload["status"] == "ok"
        assert any(
            "S3 persist failed" in record.message
            for record in caplog.records
        )


# ── Uncertainty loader behavior ────────────────────────────────────────


class TestUncertaintyLoader:
    def test_loader_skips_none_std_entries(self):
        from backtest import _load_alpha_uncertainty_from_predictions_archive

        s3 = MagicMock()
        doc = {
            "predictions": {
                "AAPL": {"predicted_alpha_std": 0.02},
                "MSFT": {"predicted_alpha_std": None},
                "GOOG": {"predicted_alpha_std": 0.015},
            }
        }
        body = MagicMock()
        body.read.return_value = json.dumps(doc).encode()
        s3.get_object.return_value = {"Body": body}

        result = _load_alpha_uncertainty_from_predictions_archive(
            bucket="bucket",
            target_dates=["2026-05-27"],
            s3_client=s3,
        )
        assert result == {"2026-05-27": {"AAPL": 0.02, "GOOG": 0.015}}

    def test_loader_skips_missing_archive_files(self):
        from backtest import _load_alpha_uncertainty_from_predictions_archive

        s3 = MagicMock()
        s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "..."}, "ResponseMetadata": {"HTTPStatusCode": 404}},
            "GetObject",
        )
        result = _load_alpha_uncertainty_from_predictions_archive(
            bucket="bucket",
            target_dates=["2026-05-27", "2026-05-28"],
            s3_client=s3,
        )
        assert result == {}

    def test_loader_skips_dates_with_only_none_std(self):
        from backtest import _load_alpha_uncertainty_from_predictions_archive

        s3 = MagicMock()
        doc = {
            "predictions": {
                "AAPL": {"predicted_alpha_std": None},
                "MSFT": {"predicted_alpha_std": None},
            }
        }
        body = MagicMock()
        body.read.return_value = json.dumps(doc).encode()
        s3.get_object.return_value = {"Body": body}

        result = _load_alpha_uncertainty_from_predictions_archive(
            bucket="bucket",
            target_dates=["2026-05-27"],
            s3_client=s3,
        )
        assert result == {}
