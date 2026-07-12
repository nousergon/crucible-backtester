"""Tests for backtest.run_portfolio_optimizer_gate.

ROADMAP L2222 PR 4.5. Pins the persistence contract:
- writes per-date + latest JSON to s3://{bucket}/predictor/optimizer_gate/
- adds a top-level ``passed`` boolean derived from the gate verdict
- forwards ``legacy_metrics`` to the underlying gate runner
- FATAL on S3 write failure: raises OptimizerGatePersistError (config#1234).
  This artifact is the Saturday SF's SOLE predictor-weight promotion lever, so
  a swallowed persist failure is the silent absence-of-artifact bug the
  fleet-wide producer-write-swallow audit targets. Persist failure must fail
  LOUD; only a gate-RUN/verdict failure remains observability-only.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import backtest as bt


@pytest.fixture
def fake_gate_result_pass():
    """A gate-runner result with all-pass verdict."""
    return {
        "comparison": {"optimizer": {"sortino": 1.2}, "legacy": None},
        "gate_report": {
            "verdict": "pass",
            "summary": "all criteria pass",
            "criteria": [
                {"name": "psr_min", "passed": True, "value": 0.96},
            ],
            "n_pass": 1,
            "n_fail": 0,
            "n_skipped": 4,
        },
        "optimizer_diagnostics": [{"date": "2026-01-05", "status": "optimal"}],
        "n_rebalances": 100,
        "n_solver_failures": 0,
    }


@pytest.fixture
def fake_gate_result_fail():
    """A gate-runner result with fail verdict."""
    return {
        "comparison": {"optimizer": {"sortino": 0.3}, "legacy": None},
        "gate_report": {
            "verdict": "fail",
            "summary": "psr_min failed",
            "criteria": [{"name": "psr_min", "passed": False, "value": 0.42}],
            "n_pass": 0,
            "n_fail": 1,
            "n_skipped": 4,
        },
        "optimizer_diagnostics": [],
        "n_rebalances": 100,
        "n_solver_failures": 0,
    }


class TestPersistenceContract:
    @patch("analysis.portfolio_optimizer_gate.run_gate_against_predictor_backtest")
    def test_writes_dated_and_latest_keys(
        self, mock_run_gate, fake_gate_result_pass,
    ):
        mock_run_gate.return_value = fake_gate_result_pass
        s3 = MagicMock()
        result = bt.run_portfolio_optimizer_gate(
            config={"signals_bucket": "my-bucket"},
            run_date="2026-05-12",
            s3_client=s3,
        )
        # Exactly two S3 writes: dated + latest pointer
        assert s3.put_object.call_count == 2
        keys = {call.kwargs["Key"] for call in s3.put_object.call_args_list}
        assert keys == {
            "predictor/optimizer_gate/2026-05-12.json",
            "predictor/optimizer_gate/latest.json",
        }
        # Same body on both writes
        bodies = {call.kwargs["Body"] for call in s3.put_object.call_args_list}
        assert len(bodies) == 1
        # Body parses + carries the verdict + top-level passed flag
        payload = json.loads(next(iter(bodies)).decode("utf-8"))
        assert payload["run_date"] == "2026-05-12"
        assert payload["passed"] is True
        assert payload["gate_report"]["verdict"] == "pass"

    @patch("analysis.portfolio_optimizer_gate.run_gate_against_predictor_backtest")
    def test_passed_flag_reflects_verdict(
        self, mock_run_gate, fake_gate_result_fail,
    ):
        mock_run_gate.return_value = fake_gate_result_fail
        result = bt.run_portfolio_optimizer_gate(
            config={"signals_bucket": "b"},
            run_date="2026-05-12",
            s3_client=MagicMock(),
        )
        assert result["passed"] is False
        assert result["gate_report"]["verdict"] == "fail"

    @patch("analysis.portfolio_optimizer_gate.run_gate_against_predictor_backtest")
    def test_default_bucket_alpha_engine_research(
        self, mock_run_gate, fake_gate_result_pass,
    ):
        """When signals_bucket is absent from config, default to the canonical bucket."""
        mock_run_gate.return_value = fake_gate_result_pass
        s3 = MagicMock()
        bt.run_portfolio_optimizer_gate(
            config={}, run_date="2026-05-12", s3_client=s3,
        )
        buckets = {call.kwargs["Bucket"] for call in s3.put_object.call_args_list}
        assert buckets == {"alpha-engine-research"}


class TestLegacyMetricsForwarding:
    @patch("analysis.portfolio_optimizer_gate.run_gate_against_predictor_backtest")
    def test_legacy_metrics_forwarded(
        self, mock_run_gate, fake_gate_result_pass,
    ):
        mock_run_gate.return_value = fake_gate_result_pass
        legacy = {"sortino": 0.9, "max_drawdown": -0.12}
        bt.run_portfolio_optimizer_gate(
            config={"signals_bucket": "b"},
            run_date="2026-05-12",
            legacy_metrics=legacy,
            s3_client=MagicMock(),
        )
        # Underlying gate runner received the legacy_metrics passthrough
        kwargs = mock_run_gate.call_args.kwargs
        assert kwargs["legacy_metrics"] is legacy

    @patch("analysis.portfolio_optimizer_gate.run_gate_against_predictor_backtest")
    def test_legacy_metrics_none_is_accepted(
        self, mock_run_gate, fake_gate_result_pass,
    ):
        """Standalone --mode portfolio-optimizer-backtest passes None — the
        gate then reports skipped verdicts for legacy-relative criteria but
        the absolute ones still run."""
        mock_run_gate.return_value = fake_gate_result_pass
        bt.run_portfolio_optimizer_gate(
            config={"signals_bucket": "b"},
            run_date="2026-05-12",
            s3_client=MagicMock(),
        )
        assert mock_run_gate.call_args.kwargs["legacy_metrics"] is None


class TestS3FailureIsFatal:
    """config#1234 root-cause fix. A PERSIST failure of the load-bearing
    optimizer-gate report must fail LOUD (previously swallowed to a WARNING —
    the silent absence-of-artifact bug the audit targets). The report is the
    Saturday SF's SOLE predictor-weight promotion lever, so a stale/absent write
    that the backtester reports as success is exactly the failure mode to
    prevent. Contrast: a gate-RUN/verdict failure remains observability-only,
    handled by the outer phase block's generic except."""

    @patch("analysis.portfolio_optimizer_gate.run_gate_against_predictor_backtest")
    def test_s3_put_failure_raises_persist_error(
        self, mock_run_gate, fake_gate_result_pass,
    ):
        mock_run_gate.return_value = fake_gate_result_pass
        s3 = MagicMock()
        s3.put_object.side_effect = Exception("simulated S3 error")
        with pytest.raises(bt.OptimizerGatePersistError):
            bt.run_portfolio_optimizer_gate(
                config={"signals_bucket": "b"},
                run_date="2026-05-12",
                s3_client=s3,
            )

    @patch("analysis.portfolio_optimizer_gate.run_gate_against_predictor_backtest")
    def test_persist_error_chains_original_cause(
        self, mock_run_gate, fake_gate_result_pass,
    ):
        """The raised error preserves the underlying S3 exception as __cause__
        so operators see the root cause, not just the wrapper."""
        mock_run_gate.return_value = fake_gate_result_pass
        s3 = MagicMock()
        original = Exception("simulated S3 error")
        s3.put_object.side_effect = original
        with pytest.raises(bt.OptimizerGatePersistError) as excinfo:
            bt.run_portfolio_optimizer_gate(
                config={"signals_bucket": "b"},
                run_date="2026-05-12",
                s3_client=s3,
            )
        assert excinfo.value.__cause__ is original


class TestModeArgparseChoice:
    def test_portfolio_optimizer_backtest_is_valid_mode(self):
        """Pins that --mode portfolio-optimizer-backtest is accepted by argparse."""
        # Calling parse_args() on a list directly avoids invoking main().
        import sys
        from unittest.mock import patch as _patch

        with _patch.object(
            sys, "argv",
            ["backtest.py", "--mode", "portfolio-optimizer-backtest"],
        ):
            args = bt._parse_args()
        assert args.mode == "portfolio-optimizer-backtest"
