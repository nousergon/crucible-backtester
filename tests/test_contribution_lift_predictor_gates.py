"""tests/test_contribution_lift_predictor_gates.py — RC v3 T5 group
`predictor_gates` (alpha-engine-config-I7480).

Covers ``analysis.contribution_lift.groups.predictor_gates``:
  * veto_gate_precision   — null_arm, picks-at-matched-width on gbm_veto
  * output_distribution_gate — N/A-NOT-LIFT-SHAPED (whole-distribution gate)
  * direction_accuracy_vs_majority_baseline — N/A-NOT-LIFT-SHAPED (already a lift)

Runs on synthetic inputs only — no AWS, no ArcticDB, no vectorbt (mirrors
tests/test_contribution_lift.py's own fixture style).
"""

from __future__ import annotations

import dataclasses
import sqlite3
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.contribution_lift import harness, objective  # noqa: E402
from analysis.contribution_lift.groups import predictor_gates  # noqa: E402
from analysis.contribution_lift.harness import NotAvailable, ReplayInputs  # noqa: E402
from analysis.contribution_lift.registry import SPECS  # noqa: E402

H = harness.HORIZON_DAYS


def _axis(n: int = 200) -> pd.DatetimeIndex:
    return pd.bdate_range("2026-01-01", periods=n)


def _flat_prices(axis: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(100.0, index=axis, columns=tickers)


def _make_trades_db(tmp_path: Path, enter_rows: list[tuple[str, str]]) -> str:
    """``enter_rows``: [(date, ticker), ...] — one row per executed ENTER."""
    db_path = tmp_path / "trades.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE trades (ticker TEXT, date TEXT, action TEXT, fill_price REAL)"
    )
    for date, ticker in enter_rows:
        conn.execute(
            "INSERT INTO trades (ticker, date, action, fill_price) VALUES (?, ?, 'ENTER', 100.0)",
            (ticker, date),
        )
    conn.commit()
    conn.close()
    return str(db_path)


def _signals(dates: list[str], tickers: list[str]) -> dict:
    return {
        d: {
            "date": d,
            "signals": {
                t: {"ticker": t, "signal": "ENTER", "score": 50} for t in tickers
            },
        }
        for d in dates
    }


def _predictions(dates: list[str], per_ticker: dict[str, dict]) -> dict:
    """``per_ticker``: {ticker: {"predicted_alpha": x, "gbm_veto": bool}}."""
    return {d: dict(per_ticker) for d in dates}


def _inputs(
    *,
    axis: pd.DatetimeIndex,
    dates: list[str],
    signals_by_date: dict,
    predictions_by_date: dict | None = None,
    trades_db_path: str | None = None,
) -> ReplayInputs:
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    return ReplayInputs(
        run_date="2026-08-17",
        dates=dates,
        signals_by_date=signals_by_date,
        predictions_by_date=predictions_by_date or {},
        pillar_profiles_by_date={},
        price_matrix=_flat_prices(axis, tickers),
        spy_prices=pd.Series(100.0, index=axis),
        bucket="test-bucket",
        fees=0.001,
        slippage_bps=10.0,
        init_cash=1_000_000.0,
        trades_db_path=trades_db_path,
        source_paths=["s3://test-bucket/signals/{date}/signals.json"],
    )


_PER_TICKER = {
    "AAA": {"predicted_alpha": 0.05, "gbm_veto": False},
    "BBB": {"predicted_alpha": 0.03, "gbm_veto": True},
    "CCC": {"predicted_alpha": 0.02, "gbm_veto": False},
    "DDD": {"predicted_alpha": 0.01, "gbm_veto": False},
}


# --------------------------------------------------------------------------
# veto_gate_precision
# --------------------------------------------------------------------------


def test_veto_gate_precision_na_when_trades_db_path_missing():
    axis = _axis(200)
    dates = [d.strftime("%Y-%m-%d") for d in axis[:5]]
    inputs = _inputs(
        axis=axis, dates=dates,
        signals_by_date=_signals(dates, list(_PER_TICKER)),
        predictions_by_date=_predictions(dates, _PER_TICKER),
        trades_db_path=None,
    )

    result = predictor_gates.build_arms(inputs)

    assert isinstance(result, NotAvailable)
    assert result.status == "N/A-MISSING-INPUT"
    assert "trades_db_path is None" in result.reason
    assert "I7480" in result.reason


def test_veto_gate_precision_na_when_trades_db_has_no_enter_rows(tmp_path):
    axis = _axis(200)
    dates = [d.strftime("%Y-%m-%d") for d in axis[:5]]
    db_path = _make_trades_db(tmp_path, [])
    inputs = _inputs(
        axis=axis, dates=dates,
        signals_by_date=_signals(dates, list(_PER_TICKER)),
        predictions_by_date=_predictions(dates, _PER_TICKER),
        trades_db_path=db_path,
    )

    result = predictor_gates.build_arms(inputs)

    assert isinstance(result, NotAvailable)
    assert result.status == "N/A-MISSING-INPUT"
    assert "no executed ENTER rows" in result.reason


def test_veto_gate_precision_arms_are_count_matched_and_differ_on_the_vetoed_name():
    axis = _axis(200)
    dates = [d.strftime("%Y-%m-%d") for d in axis[:5]]
    # Live book each day: 2 ENTER orders (AAA, CCC — the two highest-alpha
    # non-vetoed names). BBB is the gbm_veto'd name.
    enter_rows = [(d, t) for d in dates for t in ("AAA", "CCC")]
    with tempfile.TemporaryDirectory() as td:
        db_path = _make_trades_db(Path(td), enter_rows)
        inputs = _inputs(
            axis=axis, dates=dates,
            signals_by_date=_signals(dates, list(_PER_TICKER)),
            predictions_by_date=_predictions(dates, _PER_TICKER),
            trades_db_path=db_path,
        )

        arms = predictor_gates.build_arms(inputs)

    assert isinstance(arms, harness.ArmSet)
    # baseline: top-2 by alpha among non-vetoed (AAA=.05, CCC=.02, DDD=.01) -> AAA, CCC
    assert arms.baseline.picks[0][1] == ("AAA", "CCC")
    # ablated: top-2 by alpha among ALL (AAA=.05, BBB=.03 vetoed, CCC=.02) -> AAA, BBB
    assert arms.ablated.picks[0][1] == ("AAA", "BBB")

    match = objective.check_count_match(
        harness.picks_per_cycle(arms.baseline),
        harness.picks_per_cycle(arms.ablated),
    )
    assert match.matched is True


def test_veto_gate_precision_width_caps_at_the_live_enter_count():
    axis = _axis(200)
    dates = [d.strftime("%Y-%m-%d") for d in axis[:4]]
    # Live book has only 1 ENTER on the first three days even though 3
    # non-vetoed candidates exist. The fourth day opens 2, so the veto binds
    # somewhere in the window and the replay is a measurement, not the
    # all-identical no-op covered below (alpha-engine-config-I12390).
    enter_rows = [(d, "AAA") for d in dates[:3]] + [
        (dates[3], "AAA"), (dates[3], "CCC"),
    ]
    with tempfile.TemporaryDirectory() as td:
        db_path = _make_trades_db(Path(td), enter_rows)
        inputs = _inputs(
            axis=axis, dates=dates,
            signals_by_date=_signals(dates, list(_PER_TICKER)),
            predictions_by_date=_predictions(dates, _PER_TICKER),
            trades_db_path=db_path,
        )
        arms = predictor_gates.build_arms(inputs)

    assert arms.baseline.picks[0][1] == ("AAA",)
    assert arms.ablated.picks[0][1] == ("AAA",)
    assert arms.baseline.picks[3][1] == ("AAA", "CCC")
    assert arms.ablated.picks[3][1] == ("AAA", "BBB")


# Shape of the live record (backtest/2026-10-16): gbm_veto fires only on
# "predicted DOWN + high confidence", so every vetoed name has a negative
# predicted_alpha and ranks below the non-vetoed UP names that fill the width.
_PER_TICKER_VETO_BELOW_WIDTH = {
    "AAA": {"predicted_alpha": 0.05, "gbm_veto": False},
    "BBB": {"predicted_alpha": 0.03, "gbm_veto": False},
    "CCC": {"predicted_alpha": -0.01, "gbm_veto": False},
    "DDD": {"predicted_alpha": -0.04, "gbm_veto": True},
}


def _veto_below_width_inputs(td: str, dates: list[str], axis) -> ReplayInputs:
    enter_rows = [(d, t) for d in dates for t in ("AAA", "BBB")]
    db_path = _make_trades_db(Path(td), enter_rows)
    return _inputs(
        axis=axis, dates=dates,
        signals_by_date=_signals(dates, list(_PER_TICKER_VETO_BELOW_WIDTH)),
        predictions_by_date=_predictions(dates, _PER_TICKER_VETO_BELOW_WIDTH),
        trades_db_path=db_path,
    )


def test_veto_gate_precision_is_na_when_the_veto_never_binds():
    """I12390: identical arms on every cycle are a no-op, not a 0.0 lift."""
    axis = _axis(200)
    dates = [d.strftime("%Y-%m-%d") for d in axis[:5]]
    with tempfile.TemporaryDirectory() as td:
        result = predictor_gates.build_arms(_veto_below_width_inputs(td, dates, axis))

    assert isinstance(result, NotAvailable)
    assert result.status == "N/A-MISSING-INPUT"
    assert "never changed the top-N on any of the 5 replayable cycle(s)" in result.reason
    assert "5 vetoed name-date(s) on 5 cycle(s)" in result.reason
    assert "provable no-op" in result.reason
    assert "I12390" in result.reason


def test_run_spec_reports_na_not_a_zero_width_zero_when_the_veto_never_binds(
    monkeypatch,
):
    """The degenerate read the Director flagged: value 0.0, CI [0, 0], ok."""
    axis = _axis(200)
    dates = [d.strftime("%Y-%m-%d") for d in axis[:60]]

    def _never_called(*_a, **_k):
        raise AssertionError("an inert ablation must not be simulated")

    monkeypatch.setattr(harness, "simulate_arm", _never_called)
    with tempfile.TemporaryDirectory() as td:
        component = harness.run_spec(
            predictor_gates.SPEC, _veto_below_width_inputs(td, dates, axis), n_trials=1
        )

    assert component["status"] == "N/A-MISSING-INPUT"
    assert component["value"] is None
    assert component["ci_low"] is None and component["ci_high"] is None


def test_veto_gate_precision_still_measures_when_the_veto_binds_on_one_cycle():
    """One binding cycle is enough; the non-binding ones stay as honest zeros."""
    axis = _axis(200)
    dates = [d.strftime("%Y-%m-%d") for d in axis[:5]]
    binding = dict(_PER_TICKER_VETO_BELOW_WIDTH)
    binding["DDD"] = {"predicted_alpha": 0.04, "gbm_veto": True}
    predictions = _predictions(dates, _PER_TICKER_VETO_BELOW_WIDTH)
    predictions[dates[2]] = dict(binding)
    with tempfile.TemporaryDirectory() as td:
        inputs = _veto_below_width_inputs(td, dates, axis)
        inputs = dataclasses.replace(inputs, predictions_by_date=predictions)
        arms = predictor_gates.build_arms(inputs)

    assert isinstance(arms, harness.ArmSet)
    assert len(arms.baseline.picks) == len(arms.ablated.picks) == 5
    differing = [
        b[0] for b, a in zip(arms.baseline.picks, arms.ablated.picks) if b[1] != a[1]
    ]
    assert differing == [dates[2]]


def test_veto_gate_precision_measures_a_lift_end_to_end(monkeypatch):
    axis = _axis(200)
    dates = [d.strftime("%Y-%m-%d") for d in axis[:60]]
    enter_rows = [(d, t) for d in dates for t in ("AAA", "CCC")]
    with tempfile.TemporaryDirectory() as td:
        db_path = _make_trades_db(Path(td), enter_rows)
        inputs = _inputs(
            axis=axis, dates=dates,
            signals_by_date=_signals(dates, list(_PER_TICKER)),
            predictions_by_date=_predictions(dates, _PER_TICKER),
            trades_db_path=db_path,
        )

        class _FakeSim:
            def __call__(self, arm, arm_set, inp):
                base_r = 0.0011 if "as-configured" in arm.label else 0.0009
                log_returns = pd.Series(base_r, index=axis)
                return {
                    "total_return": 0.0, "total_alpha": 0.0,
                    "sortino_ratio": 1.0, "sharpe_ratio": 0.9,
                    "max_drawdown": -0.05, "psr": 0.6,
                    "daily_returns": log_returns, "daily_log_returns": log_returns,
                    "n_orders": len(harness.arm_orders(arm, arm_set, inp)),
                }

        monkeypatch.setattr(harness, "simulate_arm", _FakeSim())
        component = harness.run_spec(predictor_gates.SPEC, inputs, n_trials=1)

    assert component["status"] == "ok"
    assert component["module"] == "predictor"
    assert component["criticality"] == "supporting"
    assert component["pattern"] == "null_arm"
    assert component["count_matched"] is True
    assert component["value"] > 0.0  # as-configured (veto applied) beats disabled


# --------------------------------------------------------------------------
# output_distribution_gate / direction_accuracy_vs_majority_baseline — N/A
# --------------------------------------------------------------------------


def test_output_distribution_gate_is_not_lift_shaped():
    axis = _axis(50)
    dates = [d.strftime("%Y-%m-%d") for d in axis[:5]]
    inputs = _inputs(axis=axis, dates=dates, signals_by_date=_signals(dates, ["AAA"]))

    result = predictor_gates.OUTPUT_DISTRIBUTION_GATE_SPEC.build_arms(inputs)

    assert isinstance(result, NotAvailable)
    assert result.status == "N/A-NOT-LIFT-SHAPED"
    assert "WHOLE per-cycle output distribution" in result.reason
    assert "I7480" in result.reason


def test_direction_accuracy_vs_majority_baseline_is_not_lift_shaped():
    axis = _axis(50)
    dates = [d.strftime("%Y-%m-%d") for d in axis[:5]]
    inputs = _inputs(axis=axis, dates=dates, signals_by_date=_signals(dates, ["AAA"]))

    result = predictor_gates.DIRECTION_ACCURACY_SPEC.build_arms(inputs)

    assert isinstance(result, NotAvailable)
    assert result.status == "N/A-NOT-LIFT-SHAPED"
    assert "already an in-tile lift" in result.reason
    assert "I7480" in result.reason


def test_run_spec_passes_both_na_specs_through():
    axis = _axis(50)
    dates = [d.strftime("%Y-%m-%d") for d in axis[:5]]
    inputs = _inputs(axis=axis, dates=dates, signals_by_date=_signals(dates, ["AAA"]))

    odg = harness.run_spec(
        predictor_gates.OUTPUT_DISTRIBUTION_GATE_SPEC, inputs, n_trials=1
    )
    dab = harness.run_spec(
        predictor_gates.DIRECTION_ACCURACY_SPEC, inputs, n_trials=1
    )

    assert odg["status"] == "N/A-NOT-LIFT-SHAPED"
    assert odg["criticality"] == "critical"
    assert dab["status"] == "N/A-NOT-LIFT-SHAPED"
    assert dab["criticality"] == "supporting"


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_predictor_gates_specs_are_registered():
    names = {s.name for s in SPECS}
    assert "veto_gate_precision" in names
    assert "output_distribution_gate" in names
    assert "direction_accuracy_vs_majority_baseline" in names
    for s in SPECS:
        if s.name in (
            "veto_gate_precision",
            "output_distribution_gate",
            "direction_accuracy_vs_majority_baseline",
        ):
            assert s.module == "predictor"
            assert s.issue == "alpha-engine-config-I7480"
