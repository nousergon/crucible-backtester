"""Tests for the signal_source branch in the portfolio-optimizer gate
(ROADMAP L124 PR 2).

- run_gate_against_predictor_backtest(signal_source="production") uses the
  production producer and threads signal_source="production" into the verdict
- invalid signal_source raises
- production-producer failure → FAIL verdict carrying signal_source
- run_portfolio_optimizer_gate persists production runs to the additive
  predictor/optimizer_gate/production/ namespace; synthetic default is
  byte-for-byte the legacy key contract (unchanged consumer)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import backtest as bt
from analysis.portfolio_optimizer_gate import run_gate_against_predictor_backtest


@pytest.fixture
def cfg(tmp_path):
    # executor_paths must resolve to a real dir for the isdir() guard.
    return {"executor_paths": [str(tmp_path)], "signals_bucket": "b"}


def _opt_result():
    return SimpleNamespace(
        metrics={"sortino": 1.0},
        diagnostics_per_rebalance=[{"date": "2026-05-15", "status": "optimal"}],
        n_rebalances=10,
        n_solver_failures=0,
    )


class TestRunnerSignalSourceBranch:
    def test_invalid_signal_source_raises(self, cfg):
        with pytest.raises(ValueError, match="signal_source must be"):
            run_gate_against_predictor_backtest(cfg, signal_source="bogus")

    @patch("analysis.portfolio_optimizer_backtest.run_optimizer_backtest")
    @patch("synthetic.production_signal_backtest.build_production_signal_inputs")
    @patch("synthetic.predictor_backtest.run")
    def test_production_uses_production_producer(
        self, mock_synth, mock_prod, mock_opt, cfg,
    ):
        mock_prod.return_value = {
            "status": "ok",
            "predictions_by_date": {"2026-05-15": {"AAPL": 0.1}},
            "price_matrix": MagicMock(),
            "spy_prices": MagicMock(),
            "sector_map": {},
            "production_window": ["2026-05-13", "2026-05-15"],
        }
        mock_opt.return_value = _opt_result()

        # compare_to_legacy / evaluate_gate run for real (pure, tolerant of
        # thin metrics + legacy_metrics=None) so the signal_source thread is
        # exercised end-to-end, not mocked away.
        out = run_gate_against_predictor_backtest(cfg, signal_source="production")

        mock_prod.assert_called_once()
        mock_synth.assert_not_called()
        assert out["comparison"]["signal_source"] == "production"
        assert out["gate_report"]["signal_source"] == "production"
        assert out["signal_source"] == "production"
        assert out["production_window"] == ["2026-05-13", "2026-05-15"]

    @patch("analysis.portfolio_optimizer_backtest.run_optimizer_backtest")
    @patch("synthetic.predictor_backtest.run")
    def test_synthetic_default_unchanged(self, mock_synth, mock_opt, cfg):
        mock_synth.return_value = {
            "status": "ok",
            "predictions_by_date": {"2026-01-05": {"AAPL": 0.1}},
            "price_matrix": MagicMock(),
            "spy_prices": MagicMock(),
            "sector_map": {},
        }
        mock_opt.return_value = _opt_result()

        out = run_gate_against_predictor_backtest(cfg)  # default

        mock_synth.assert_called_once()
        assert out["comparison"]["signal_source"] == "synthetic"
        assert out["signal_source"] == "synthetic"

    @patch("synthetic.production_signal_backtest.build_production_signal_inputs")
    def test_production_producer_failure_is_fail_verdict(self, mock_prod, cfg):
        mock_prod.return_value = {
            "status": "no_production_data", "error": "no overlap",
        }
        out = run_gate_against_predictor_backtest(cfg, signal_source="production")
        assert out["gate_report"]["verdict"] == "fail"
        assert out["signal_source"] == "production"
        assert "no_production_data" in out["gate_report"]["summary"]


class TestWrapperPersistenceNamespace:
    @patch("analysis.portfolio_optimizer_gate.run_gate_against_predictor_backtest")
    def test_production_writes_to_production_namespace(self, mock_gate):
        mock_gate.return_value = {
            "gate_report": {"verdict": "pass"}, "signal_source": "production",
        }
        s3 = MagicMock()
        bt.run_portfolio_optimizer_gate(
            config={"signals_bucket": "b"}, run_date="2026-05-15",
            s3_client=s3, signal_source="production",
        )
        keys = {c.kwargs["Key"] for c in s3.put_object.call_args_list}
        assert keys == {
            "predictor/optimizer_gate/production/2026-05-15.json",
            "predictor/optimizer_gate/production/latest.json",
        }
        # signal_source forwarded to the runner
        assert mock_gate.call_args.kwargs["signal_source"] == "production"

    @patch("analysis.portfolio_optimizer_gate.run_gate_against_predictor_backtest")
    def test_synthetic_default_keeps_legacy_keys(self, mock_gate):
        mock_gate.return_value = {
            "gate_report": {"verdict": "pass"}, "signal_source": "synthetic",
        }
        s3 = MagicMock()
        bt.run_portfolio_optimizer_gate(
            config={"signals_bucket": "b"}, run_date="2026-05-15", s3_client=s3,
        )
        keys = {c.kwargs["Key"] for c in s3.put_object.call_args_list}
        assert keys == {
            "predictor/optimizer_gate/2026-05-15.json",
            "predictor/optimizer_gate/latest.json",
        }
        assert mock_gate.call_args.kwargs["signal_source"] == "synthetic"


class TestCarriedBasisShadow:
    """alpha-engine-config-I11507: the retired all-cash basis runs beside the
    gate on the same inputs and never reaches latest.json."""

    @patch("analysis.portfolio_optimizer_backtest.run_optimizer_backtest")
    @patch("synthetic.predictor_backtest.run")
    def test_shadow_runs_the_all_cash_basis_by_default(self, mock_synth, mock_opt, cfg):
        mock_synth.return_value = {
            "status": "ok", "predictions_by_date": {"2026-01-05": {"AAPL": 0.1}},
            "price_matrix": MagicMock(), "spy_prices": MagicMock(), "sector_map": {},
        }
        mock_opt.return_value = _opt_result()
        out = run_gate_against_predictor_backtest(cfg)
        bases = [c.kwargs.get("w_prev_basis") for c in mock_opt.call_args_list]
        assert bases[:2] == [None, "all_cash_each_rebalance"]
        assert out["shadow"]["status"] == "ok"
        assert out["shadow"]["w_prev_basis"] == "all_cash_each_rebalance"
        assert "gate_report" in out["shadow"]

    @patch("analysis.portfolio_optimizer_backtest.run_optimizer_backtest")
    @patch("synthetic.predictor_backtest.run")
    def test_shadow_disabled_by_null(self, mock_synth, mock_opt, cfg):
        mock_synth.return_value = {
            "status": "ok", "predictions_by_date": {"2026-01-05": {"AAPL": 0.1}},
            "price_matrix": MagicMock(), "spy_prices": MagicMock(), "sector_map": {},
        }
        mock_opt.return_value = _opt_result()
        cfg["portfolio_optimizer_gate"] = {"shadow_w_prev_basis": None, "churn_arms": None}
        out = run_gate_against_predictor_backtest(cfg)
        assert mock_opt.call_count == 1
        assert "shadow" not in out

    @patch("analysis.portfolio_optimizer_backtest.run_optimizer_backtest")
    @patch("synthetic.predictor_backtest.run")
    def test_shadow_failure_does_not_sink_the_gate(self, mock_synth, mock_opt, cfg):
        mock_synth.return_value = {
            "status": "ok", "predictions_by_date": {"2026-01-05": {"AAPL": 0.1}},
            "price_matrix": MagicMock(), "spy_prices": MagicMock(), "sector_map": {},
        }
        cfg["portfolio_optimizer_gate"] = {"churn_arms": None}
        mock_opt.side_effect = [_opt_result(), RuntimeError("boom")]
        out = run_gate_against_predictor_backtest(cfg)
        assert out["gate_report"]["verdict"]
        assert out["shadow"]["status"] == "error"
        assert "boom" in out["shadow"]["error"]

    @patch("analysis.portfolio_optimizer_gate.run_gate_against_predictor_backtest")
    def test_shadow_persists_to_a_sibling_key_not_latest(self, mock_gate):
        shadow = {"status": "ok", "w_prev_basis": "carried_drifted",
                  "gate_report": {"verdict": "fail"}}
        mock_gate.return_value = {
            "gate_report": {"verdict": "pass"}, "signal_source": "synthetic",
            "shadow": shadow,
        }
        s3 = MagicMock()
        payload = bt.run_portfolio_optimizer_gate(
            config={"signals_bucket": "b"}, run_date="2026-05-15", s3_client=s3,
        )
        bodies = {c.kwargs["Key"]: c.kwargs["Body"] for c in s3.put_object.call_args_list}
        assert set(bodies) == {
            "predictor/optimizer_gate/2026-05-15.json",
            "predictor/optimizer_gate/latest.json",
            "predictor/optimizer_gate/shadow_carried_drifted/2026-05-15.json",
        }
        assert b"shadow" not in bodies["predictor/optimizer_gate/latest.json"]
        assert payload["passed"] is True
        assert payload["shadow_key"].endswith("shadow_carried_drifted/2026-05-15.json")


class TestChurnArms:
    """Decision-precedence note 2026-10-15: the churn controls run as
    challenger arms beside a same-stride champion, never as the verdict."""

    _PRED = {
        "status": "ok", "predictions_by_date": {"2026-01-05": {"AAPL": 0.1}},
        "price_matrix": MagicMock(), "spy_prices": MagicMock(), "sector_map": {},
    }

    @patch("analysis.portfolio_optimizer_backtest.run_optimizer_backtest")
    @patch("synthetic.predictor_backtest.run")
    def test_default_arms_run_daily_on_the_carried_book(self, mock_synth, mock_opt, cfg):
        mock_synth.return_value = dict(self._PRED)
        mock_opt.return_value = _opt_result()
        out = run_gate_against_predictor_backtest(cfg)
        arm_calls = mock_opt.call_args_list[2:]
        assert [c.kwargs["turnover_control"] for c in arm_calls] == [
            "none", "partial_adjustment", "reversal_guard",
        ]
        for c in arm_calls:
            assert c.kwargs["rebalance_freq_days"] == 1
            assert c.kwargs["w_prev_basis"] == "carried_drifted"
            assert c.kwargs["rebalance_band"] == 0.005
        assert [a["status"] for a in out["churn_arms"]["arms"]] == ["ok"] * 3
        assert "verdict" in out["churn_arms"]["arms"][0]

    @patch("analysis.portfolio_optimizer_backtest.run_optimizer_backtest")
    @patch("synthetic.predictor_backtest.run")
    def test_failed_arm_is_recorded_and_the_rest_still_run(self, mock_synth, mock_opt, cfg):
        mock_synth.return_value = dict(self._PRED)
        cfg["portfolio_optimizer_gate"] = {"shadow_w_prev_basis": None}
        mock_opt.side_effect = [_opt_result(), _opt_result(), RuntimeError("boom"), _opt_result()]
        out = run_gate_against_predictor_backtest(cfg)
        assert out["gate_report"]["verdict"]
        assert [a["status"] for a in out["churn_arms"]["arms"]] == ["ok", "error", "ok"]

    def test_arms_past_the_budget_are_skipped(self):
        from analysis.portfolio_optimizer_gate import run_churn_arms

        ticks = iter([0.0, 0.0, 0.0, 10.0, 10.0, 999.0, 999.0])
        cfg = {"arms": [{"name": "a", "turnover_control": "none"},
                        {"name": "b", "turnover_control": "none"}],
               "rebalance_freq_days": 1, "rebalance_band": 0.0, "budget_minutes": 1}
        with patch("analysis.portfolio_optimizer_backtest.run_optimizer_backtest",
                   return_value=_opt_result()):
            out = run_churn_arms(cfg, dict(self._PRED), "/x", 30, None, "synthetic",
                                 clock=lambda: next(ticks))
        assert [a["status"] for a in out["arms"]] == ["ok", "skipped_budget"]

    @patch("analysis.portfolio_optimizer_backtest.run_optimizer_backtest")
    @patch("synthetic.predictor_backtest.run")
    def test_disabled_by_null(self, mock_synth, mock_opt, cfg):
        mock_synth.return_value = dict(self._PRED)
        mock_opt.return_value = _opt_result()
        cfg["portfolio_optimizer_gate"] = {"churn_arms": None}
        out = run_gate_against_predictor_backtest(cfg)
        assert "churn_arms" not in out

    @patch("analysis.portfolio_optimizer_gate.run_gate_against_predictor_backtest")
    def test_arms_persist_to_their_own_key_not_latest(self, mock_gate):
        mock_gate.return_value = {
            "gate_report": {"verdict": "pass"}, "signal_source": "synthetic",
            "churn_arms": {"rebalance_freq_days": 1, "arms": [{"name": "champion", "status": "ok"}]},
        }
        s3 = MagicMock()
        payload = bt.run_portfolio_optimizer_gate(
            config={"signals_bucket": "b"}, run_date="2026-05-15", s3_client=s3,
        )
        bodies = {c.kwargs["Key"]: c.kwargs["Body"] for c in s3.put_object.call_args_list}
        assert "predictor/optimizer_gate/churn_arms/2026-05-15.json" in bodies
        assert b"churn_arms" not in bodies["predictor/optimizer_gate/latest.json"]
        assert payload["churn_arms_key"].endswith("churn_arms/2026-05-15.json")
