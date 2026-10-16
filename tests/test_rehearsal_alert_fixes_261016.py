"""Regression tests for the ten alerts rehearsal-2026-10-16-1 sent Brian.

Every alert came from a weekly-SF rehearsal running on deliberately stale
data. Each test pins one cause:

* a rehearsal reports through a flow-doctor config with no paging channel,
  and its ops alerts skip SNS;
* the stale-price check compares dates, not instants, and a rehearsal
  measures freshness against the day it simulates;
* a solved optimizer name with no usable close (NaN, zero or negative) is
  left unordered instead of handing vectorbt an order it rejects;
* the live executor's "conviction gate OFF" alarm is demoted inside a
  simulated solve;
* T2 sortino's declared score_performance freeze logs WARNING, not ERROR;
* a simulate refusal is an upstream gap, not a missing-key defect.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from optimizer import run_role

REHEARSAL_ENV = {"RUN_TOKEN": "rehearsal-2026-10-16-1"}
LIVE_ENV = {"RUN_TOKEN": "weekly-2026-10-17"}


# ── rehearsal routing ───────────────────────────────────────────────────────

class TestRehearsalFlowDoctorRouting:
    def test_rehearsal_reports_through_the_rehearsal_yaml(self):
        path = run_role.flow_doctor_yaml("/repo", env=REHEARSAL_ENV)
        assert path == "/repo/flow-doctor.rehearsal.yaml"

    def test_a_real_run_keeps_the_production_yaml(self):
        assert run_role.flow_doctor_yaml("/repo", env=LIVE_ENV) == "/repo/flow-doctor.yaml"
        assert run_role.flow_doctor_yaml("/repo", env={}) == "/repo/flow-doctor.yaml"

    def test_rehearsal_yaml_has_no_paging_channel_but_still_files_a_finding(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        rehearsal = yaml.safe_load((root / "flow-doctor.rehearsal.yaml").read_text())
        production = yaml.safe_load((root / "flow-doctor.yaml").read_text())

        types = [n["type"] for n in rehearsal["notify"]]
        assert "email" not in types and "telegram" not in types
        github = [n for n in rehearsal["notify"] if n["type"] == "github"]
        assert github and "rehearsal-finding" in github[0]["labels"]
        # Same store as production, so the finding is still recorded.
        assert rehearsal["store"] == production["store"]


class TestRehearsalOpsAlert:
    def _publish(self, monkeypatch, env):
        for k in ("RUN_TOKEN", "AE_PIPELINE_ROLE"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        import ops_alerts

        with patch("nousergon_lib.alerts.publish") as pub, \
                patch.object(ops_alerts, "get_flow_doctor", return_value=None):
            ops_alerts.publish_ops_alert("boom", severity="error", source="src")
        return pub.call_args.kwargs

    def test_rehearsal_skips_sns_and_tags_the_message(self, monkeypatch):
        kwargs = self._publish(monkeypatch, REHEARSAL_ENV)
        assert kwargs["sns"] is False
        assert kwargs["message"].startswith("[REHEARSAL RUN_TOKEN=rehearsal-2026-10-16-1]")

    def test_a_real_run_still_pages(self, monkeypatch):
        kwargs = self._publish(monkeypatch, LIVE_ENV)
        assert kwargs["sns"] is True
        assert kwargs["message"] == "boom"


# ── stale-price boundary ────────────────────────────────────────────────────

def _matrix_ending(last: str, monkeypatch, now: str, as_of: str | None = None):
    from loaders import price_loader

    idx = pd.bdate_range(end=last, periods=10)
    dates = [d.strftime("%Y-%m-%d") for d in idx]
    price_data = {
        "AAPL": pd.DataFrame(
            {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": np.arange(10, 20, dtype=float),
             "Volume": 1_000_000}, index=idx,
        ),
    }
    monkeypatch.setattr(price_loader.pd.Timestamp, "now", classmethod(lambda cls: pd.Timestamp(now)))
    with patch("loaders.price_loader._tickers_from_signals", return_value=["AAPL"]), \
            patch("loaders.price_loader.load_universe_from_arctic", return_value=(price_data, {})):
        return price_loader.build_matrix(dates, bucket="test", as_of=as_of)


class TestStalenessBoundary:
    def test_exactly_at_the_threshold_date_is_not_stale(self, monkeypatch):
        # 2026-10-16 14:58 minus 5 BDays is 2026-10-09: the rehearsal's
        # "last date 2026-10-09, expected >= 2026-10-09" contradiction.
        df = _matrix_ending("2026-10-09", monkeypatch, now="2026-10-16 14:58")
        assert df.attrs["stale_circuit_break"] is False

    def test_a_day_past_the_threshold_still_breaks(self, monkeypatch):
        df = _matrix_ending("2026-10-08", monkeypatch, now="2026-10-16 14:58")
        assert df.attrs["stale_circuit_break"] is True
        assert "STALE price data" in df.attrs["staleness_warning"]

    def test_as_of_measures_against_the_simulated_day(self, monkeypatch):
        # Weeks-old data is fresh for a rehearsal simulating its own date.
        df = _matrix_ending("2026-10-09", monkeypatch, now="2026-11-20 12:00", as_of="2026-10-12")
        assert df.attrs["stale_circuit_break"] is False


# ── vectorbt NaN price ──────────────────────────────────────────────────────

class TestSolvedNameWithoutAClose:
    def test_solved_name_with_nan_close_is_left_unordered(self):
        from analysis.portfolio_optimizer_backtest import _populate_target_weights_row

        idx = pd.to_datetime(["2026-10-15"])
        tw = pd.DataFrame(np.nan, index=idx, columns=["AAPL", "MSFT", "SPY"])
        close = pd.Series({"AAPL": 100.0, "MSFT": np.nan, "SPY": 0.0})
        _populate_target_weights_row(
            tw, "2026-10-15", ["AAPL", "MSFT", "SPY", "CASH"],
            np.array([0.4, 0.3, 0.27, 0.03]), close_row=close,
        )
        row = tw.loc[idx[0]]
        assert row["AAPL"] == pytest.approx(0.4)
        # vectorbt raises on any order whose price is not finite and > 0.
        assert np.isnan(row["MSFT"])
        assert np.isnan(row["SPY"])

    def test_vectorbt_accepts_the_resulting_frame(self):
        vbt = pytest.importorskip("vectorbt")
        from analysis.portfolio_optimizer_backtest import _populate_target_weights_row

        idx = pd.bdate_range("2026-10-12", periods=4)
        # A zero print is what vectorbt rejects; a NaN close it skips.
        prices = pd.DataFrame({"AAPL": [100.0, 101, 102, 103], "MSFT": [50.0, 51, 0.0, 53]}, index=idx)
        tw = pd.DataFrame(np.nan, index=idx, columns=prices.columns)
        _populate_target_weights_row(
            tw, str(idx[2].date()), ["AAPL", "MSFT", "CASH"], np.array([0.5, 0.47, 0.03]),
            close_row=prices.loc[idx[2]],
        )
        vbt.Portfolio.from_orders(
            close=prices, size=tw, size_type="targetpercent", init_cash=1e6,
            cash_sharing=True, group_by=True, freq="D",
        )


# ── conviction gate alarm inside a simulation ───────────────────────────────

class TestSimulatedGateOff:
    def test_gate_off_error_is_demoted_inside_a_simulated_solve(self, caplog):
        from analysis.portfolio_optimizer_backtest import _simulated_gate_off_is_not_an_incident

        log = logging.getLogger("executor.portfolio_optimizer")
        with caplog.at_level(logging.DEBUG), _simulated_gate_off_is_not_an_incident():
            log.error("Conviction gate OFF (%s): the predictions carry no usable sigma", "no_alpha_uncertainty_vector")
            log.error("solver exploded")
        levels = {r.getMessage(): r.levelno for r in caplog.records}
        gate = [m for m in levels if "Conviction gate OFF" in m]
        assert gate and levels[gate[0]] == logging.INFO
        assert gate[0].startswith("[simulated solve, not the live book]")
        assert levels["solver exploded"] == logging.ERROR

    def test_filter_is_removed_after_the_solve(self, caplog):
        from analysis.portfolio_optimizer_backtest import _simulated_gate_off_is_not_an_incident

        log = logging.getLogger("executor.portfolio_optimizer")
        with _simulated_gate_off_is_not_an_incident():
            pass
        with caplog.at_level(logging.DEBUG):
            log.error("Conviction gate OFF (x): live")
        assert caplog.records[-1].levelno == logging.ERROR


# ── T2 declared freeze ──────────────────────────────────────────────────────

class TestT2DeclaredFreeze:
    def _run(self, caplog, max_score_date):
        from analysis import regime_stratified_sortino_runner as r

        window = r.InputWindow("2026-03-04", max_score_date, 534)
        df = pd.DataFrame({"x": [1]})
        with patch.object(r, "load_with_subscores_and_regime", return_value=df), \
                patch.object(r, "stratified_sortino_by_regime", return_value=[]), \
                patch.object(r, "input_window", return_value=window), \
                caplog.at_level(logging.WARNING):
            out = r.run_regime_stratified_sortino(db_path="x", s3_bucket=None, write=False)
        return out, [rec for rec in caplog.records if "[T2]" in rec.getMessage()]

    def test_the_declared_freeze_logs_warning_and_says_why(self, caplog):
        out, recs = self._run(caplog, "2026-07-10")
        assert out["status"] == "unmeasurable"
        assert "I11609" in out["status_reason"]
        assert recs and all(r.levelno == logging.WARNING for r in recs)

    def test_an_undeclared_stop_still_errors(self, caplog):
        out, recs = self._run(caplog, "2026-06-01")
        assert out["status"] == "unmeasurable"
        assert recs and recs[0].levelno == logging.ERROR

    def test_the_constant_matches_the_producer_declaration(self):
        from analysis.regime_stratified_sortino_runner import SCORE_PERFORMANCE_V1_FROZEN_THROUGH

        # nousergon-data collectors/signal_returns.py::_SCORE_PERFORMANCE_V1_FROZEN_THROUGH
        assert SCORE_PERFORMANCE_V1_FROZEN_THROUGH == "2026-07-10"


# ── simulate refusal is upstream, not a defect ─────────────────────────────

class TestSimulateRefusal:
    def _registry(self, keys):
        reg = MagicMock()
        reg.load_marker.return_value = {"artifact_keys": keys}
        reg.s3_client = MagicMock()
        return reg

    def test_stale_price_refusal_is_insufficient_data(self):
        import evaluate

        reg = self._registry(["backtest/2026-10-15/.phases/simulate/portfolio_stats.json"])
        with patch("phase_artifacts.load_json", return_value={"status": "stale_prices"}):
            obj, reason = evaluate._load_portfolio_daily_returns({}, reg)
        assert obj is None
        assert reason.startswith("insufficient_data:portfolio_daily_returns_simulation_stale_prices")

    def test_an_ok_simulate_without_the_key_is_still_a_defect(self):
        import evaluate

        reg = self._registry(["backtest/2026-10-15/.phases/simulate/portfolio_stats.json"])
        with patch("phase_artifacts.load_json", return_value={"status": "ok"}):
            obj, reason = evaluate._load_portfolio_daily_returns({}, reg)
        assert obj is None
        assert reason.startswith("defect:portfolio_daily_returns_artifact_key_missing")
