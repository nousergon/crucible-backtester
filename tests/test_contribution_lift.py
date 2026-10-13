"""tests/test_contribution_lift.py — the RC v3 T5 contribution-lift harness.

Everything here runs on a SYNTHETIC price matrix and synthetic signals: no
AWS, no ArcticDB, no vectorbt. The objective/CI/count-match/N-A/determinism
tests exercise the maths directly; the harness-level tests inject a fake
simulator so the wiring is covered without pulling the heavy sim into CI.

Anchors:
  * contract: contribution_lift.json v1 (crucible-evaluator consumer)
  * spec: alpha-engine-docs/private/report-card-v3-objective-and-attribution-260816.md §1, §3
  * epic alpha-engine-config-I7473; harness I7475; seed spec I7484
"""

from __future__ import annotations

import math
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.contribution_lift import harness, objective, report  # noqa: E402
from analysis.contribution_lift.groups import cost_adjusted_quality  # noqa: E402
from analysis.contribution_lift.harness import (  # noqa: E402
    HORIZON_DAYS,
    ArmSet,
    NotAvailable,
    ReplayInputs,
    ReplaySpec,
    picks_arm,
)
from analysis.contribution_lift.registry import SPECS  # noqa: E402


H = HORIZON_DAYS


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _axis(n: int = 200) -> pd.DatetimeIndex:
    """A synthetic trading-day axis (business days, deterministic)."""
    return pd.bdate_range("2026-01-01", periods=n)


def _flat_prices(axis: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(100.0, index=axis, columns=tickers)


def _trades_db(path: Path, enters: dict[str, list[str]]) -> str:
    """A synthetic trades.db carrying the executed ENTER rows of ``enters``.

    Same table and columns ``analysis/post_trade.py`` reads, so a fixture that
    passes here is a fixture the production SQL can read.
    """
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE trades (date TEXT, ticker TEXT, action TEXT, "
            "shares REAL, price_at_order REAL)"
        )
        conn.executemany(
            "INSERT INTO trades VALUES (?, ?, 'ENTER', 10.0, 100.0)",
            [(d, t) for d, tickers in enters.items() for t in tickers],
        )
        conn.commit()
    finally:
        conn.close()
    return str(path)


def _inputs(
    *,
    axis: pd.DatetimeIndex | None = None,
    dates: list[str] | None = None,
    signals_by_date: dict | None = None,
    trades_db_path: str | None = None,
    fees: float = 0.001,
    slippage_bps: float = 10.0,
) -> ReplayInputs:
    axis = axis if axis is not None else _axis()
    tickers = ["AAA", "BBB", "CCC"]
    if dates is None:
        dates = [d.strftime("%Y-%m-%d") for d in axis[:60]]
    return ReplayInputs(
        run_date="2026-08-17",
        dates=dates,
        signals_by_date=signals_by_date or {},
        predictions_by_date={},
        pillar_profiles_by_date={},
        price_matrix=_flat_prices(axis, tickers),
        spy_prices=pd.Series(100.0, index=axis),
        bucket="test-bucket",
        fees=fees,
        slippage_bps=slippage_bps,
        init_cash=1_000_000.0,
        trades_db_path=trades_db_path,
        source_paths=["s3://test-bucket/signals/{date}/signals.json"],
    )


class _FakeSim:
    """Stands in for vectorbt: maps an arm label to a canned stats dict."""

    def __init__(self, by_label: dict[str, pd.Series]):
        self.by_label = by_label
        self.calls: list[tuple[str, float, float]] = []

    def __call__(self, arm, arm_set, inputs):
        fees = inputs.fees if arm.fees is None else arm.fees
        slip = inputs.slippage_bps if arm.slippage_bps is None else arm.slippage_bps
        self.calls.append((arm.label, fees, slip))
        log_returns = self.by_label[arm.label]
        return {
            "total_return": float(np.expm1(log_returns.sum())),
            "total_alpha": 0.0,
            "sortino_ratio": 1.0,
            "sharpe_ratio": 0.9,
            "max_drawdown": -0.05,
            "psr": 0.6,
            "daily_returns": np.expm1(log_returns),
            "daily_log_returns": log_returns,
            "n_orders": len(harness.arm_orders(arm, arm_set, inputs)),
        }


# --------------------------------------------------------------------------
# Objective maths
# --------------------------------------------------------------------------


def test_per_cycle_log_alpha_known_returns():
    """A constant daily log return over a flat SPY gives H x r per cycle."""
    axis = _axis(100)
    r = 0.001
    arm = pd.Series(r, index=axis)
    spy = pd.Series(100.0, index=axis)
    cycles = [axis[0].strftime("%Y-%m-%d"), axis[5].strftime("%Y-%m-%d")]

    out = objective.per_cycle_log_alpha(arm, spy, axis, cycles, horizon_days=H)

    assert set(out) == set(cycles)
    for value in out.values():
        assert value == pytest.approx(H * r, abs=1e-12)


def test_per_cycle_log_alpha_subtracts_spy_over_the_same_window():
    """SPY compounding at a known rate is removed exactly, not approximately."""
    axis = _axis(100)
    arm_r, spy_r = 0.002, 0.0005
    arm = pd.Series(arm_r, index=axis)
    spy = pd.Series(100.0 * np.exp(spy_r * np.arange(len(axis))), index=axis)
    cycle = axis[3].strftime("%Y-%m-%d")

    out = objective.per_cycle_log_alpha(arm, spy, axis, [cycle], horizon_days=H)

    assert out[cycle] == pytest.approx(H * (arm_r - spy_r), abs=1e-10)


def test_per_cycle_log_alpha_omits_cycles_without_a_full_horizon():
    """A cycle inside the horizon tail is dropped, never truncated."""
    axis = _axis(30)
    arm = pd.Series(0.001, index=axis)
    spy = pd.Series(100.0, index=axis)
    ok_cycle = axis[0].strftime("%Y-%m-%d")          # 29 rows after -> fits
    short_cycle = axis[20].strftime("%Y-%m-%d")      # 9 rows after -> does not

    out = objective.per_cycle_log_alpha(
        arm, spy, axis, [ok_cycle, short_cycle], horizon_days=H
    )

    assert ok_cycle in out
    assert short_cycle not in out


def test_per_cycle_log_alpha_fills_inactive_rows_with_zero():
    """Rows outside the portfolio's active window contribute 0, not NaN."""
    axis = _axis(100)
    arm = pd.Series(0.001, index=axis[:10])  # active for 10 rows only
    spy = pd.Series(100.0, index=axis)
    cycle = axis[0].strftime("%Y-%m-%d")

    out = objective.per_cycle_log_alpha(arm, spy, axis, [cycle], horizon_days=H)

    # rows 1..9 are active (row 0 is the cycle date itself, excluded)
    assert out[cycle] == pytest.approx(9 * 0.001, abs=1e-12)


def test_per_cycle_log_alpha_rejects_a_non_series():
    with pytest.raises(TypeError, match="daily_log_returns"):
        objective.per_cycle_log_alpha(
            [0.1, 0.2], pd.Series(dtype=float), _axis(30), [], horizon_days=H
        )


# --------------------------------------------------------------------------
# Paired bootstrap CI
# --------------------------------------------------------------------------


def test_paired_diffs_only_pairs_shared_cycles():
    baseline = {"2026-01-01": 0.03, "2026-01-02": 0.01, "2026-01-05": 0.02}
    ablated = {"2026-01-02": 0.005, "2026-01-05": 0.02, "2026-01-09": 0.4}

    assert objective.paired_diffs(baseline, ablated) == [0.005, 0.0]


def test_paired_bootstrap_ci_brackets_a_constant_lift():
    """A constant paired difference has zero spread — CI collapses onto it."""
    result = objective.paired_bootstrap_ci([0.01] * 40)

    assert result["estimate"] == pytest.approx(0.01)
    assert result["ci_low"] == pytest.approx(0.01)
    assert result["ci_high"] == pytest.approx(0.01)
    assert result["ci_method"] == "bootstrap"


def test_paired_bootstrap_ci_is_deterministic():
    """seed=0 — two calls over identical data agree to the bit."""
    rs = np.random.RandomState(7)
    diffs = list(rs.normal(0.002, 0.01, 80))

    first = objective.paired_bootstrap_ci(diffs)
    second = objective.paired_bootstrap_ci(diffs)

    assert first == second
    assert first["ci_low"] < first["estimate"] < first["ci_high"]


# --------------------------------------------------------------------------
# Count matching
# --------------------------------------------------------------------------


def test_check_count_match_accepts_identical_widths():
    widths = {"2026-01-01": 5, "2026-01-02": 5}

    assert objective.check_count_match(widths, dict(widths)).matched is True


def test_check_count_match_names_the_mismatched_cycle():
    result = objective.check_count_match(
        {"2026-01-01": 8, "2026-01-02": 5},
        {"2026-01-01": 14, "2026-01-02": 5},
    )

    assert result.matched is False
    assert "2026-01-01: baseline=8 vs ablated=14" in result.reason
    assert "1/2 shared cycles" in result.reason


def test_check_count_match_rejects_disjoint_arms():
    result = objective.check_count_match({"2026-01-01": 5}, {"2026-02-01": 5})

    assert result.matched is False
    assert "no cycle is present in both arms" in result.reason


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------


def test_arm_requires_exactly_one_shape():
    with pytest.raises(ValueError, match="exactly one of picks/orders"):
        harness.Arm(label="bad")
    with pytest.raises(ValueError, match="exactly one of picks/orders"):
        harness.Arm(label="bad", picks=(), orders=())


def test_picks_arm_normalizes_deterministically():
    a = picks_arm("x", [{"date": "2026-01-02", "picks": ["BBB", "AAA", "AAA"]},
                        {"date": "2026-01-01", "picks": ["CCC"]}])
    b = picks_arm("x", [{"date": "2026-01-01", "picks": ["CCC"]},
                        {"date": "2026-01-02", "picks": ["AAA", "BBB"]}])

    assert a == b
    assert a.picks == (("2026-01-01", ("CCC",)), ("2026-01-02", ("AAA", "BBB")))


def test_picks_to_orders_sizes_off_the_shared_slot_count():
    """Both arms share one budget base, so a narrower arm cannot lever up."""
    axis = _axis(60)
    prices = _flat_prices(axis, ["AAA", "BBB"])
    wide = picks_arm("wide", [{"date": axis[0].strftime("%Y-%m-%d"),
                              "picks": ["AAA", "BBB"]}])
    narrow = picks_arm("narrow", [{"date": axis[0].strftime("%Y-%m-%d"),
                                  "picks": ["AAA"]}])
    arm_set = ArmSet(baseline=wide, ablated=narrow)

    orders = harness.picks_to_orders(
        narrow.picks, prices, init_cash=1_000_000.0,
        per_cycle_slots=harness._slot_count(arm_set), hold_days=H,
    )

    enters = [o for o in orders if o["action"] == "ENTER"]
    assert len(enters) == 1
    # 1_000_000 / 2 slots / 100.0 == 5000 shares, NOT 10000.
    assert enters[0]["shares"] == 5000.0


def test_picks_per_cycle_counts_distinct_enters_for_an_orders_arm():
    arm = harness.orders_arm("ex", [
        {"date": "2026-01-05", "ticker": "AAA", "action": "ENTER", "shares": 10},
        {"date": "2026-01-05", "ticker": "BBB", "action": "ENTER", "shares": 10},
        {"date": "2026-01-05", "ticker": "AAA", "action": "EXIT"},
        {"date": "2026-01-06", "ticker": "CCC", "action": "ENTER", "shares": 10},
    ])

    assert harness.picks_per_cycle(arm) == {"2026-01-05": 2, "2026-01-06": 1}


# --------------------------------------------------------------------------
# Spec validation
# --------------------------------------------------------------------------


def test_replay_spec_rejects_an_unknown_criticality():
    with pytest.raises(ValueError, match="criticality"):
        ReplaySpec(name="x", module="m", criticality="vital",
                   pattern="null_arm", issue="I1", build_arms=lambda i: None)


def test_replay_spec_rejects_an_unknown_pattern():
    with pytest.raises(ValueError, match="pattern"):
        ReplaySpec(name="x", module="m", criticality="critical",
                   pattern="ablation", issue="I1", build_arms=lambda i: None)


def test_not_available_rejects_an_off_taxonomy_status():
    with pytest.raises(ValueError, match="NotAvailable.status"):
        NotAvailable(status="MISSING", reason="nope")


# --------------------------------------------------------------------------
# run_spec
# --------------------------------------------------------------------------


def _spec(build_arms) -> ReplaySpec:
    return ReplaySpec(
        name="risk_guard",
        module="executor",
        criticality="critical",
        pattern="null_arm",
        issue="alpha-engine-config-I7482",
        build_arms=build_arms,
    )


def _two_arm_set(dates: list[str], *, widths=(2, 2)) -> ArmSet:
    baseline = picks_arm(
        "as-configured",
        [{"date": d, "picks": ["AAA", "BBB"][: widths[0]]} for d in dates],
    )
    ablated = picks_arm(
        "guard disabled",
        [{"date": d, "picks": ["AAA", "BBB"][: widths[1]]} for d in dates],
    )
    return ArmSet(baseline=baseline, ablated=ablated)


def test_run_spec_measures_a_known_lift(monkeypatch):
    axis = _axis(200)
    inputs = _inputs(axis=axis)
    fake = _FakeSim({
        "as-configured": pd.Series(0.002, index=axis),
        "guard disabled": pd.Series(0.001, index=axis),
    })
    monkeypatch.setattr(harness, "simulate_arm", fake)

    component = harness.run_spec(
        _spec(lambda i: _two_arm_set(i.dates)), inputs, n_trials=1234
    )

    assert component["status"] == "ok"
    assert component["count_matched"] is True
    assert component["n_samples"] == len(inputs.dates)
    # Each arm's per-cycle alpha is H x r (SPY is flat), so the lift is
    # H x (0.002 - 0.001) on every cycle and the CI collapses onto it.
    assert component["value"] == pytest.approx(H * 0.001, abs=1e-9)
    assert component["ci_low"] == pytest.approx(H * 0.001, abs=1e-9)
    assert component["ci_high"] == pytest.approx(H * 0.001, abs=1e-9)
    assert component["unit"] == "log_alpha_21d"
    assert component["dsr"]["n_trials"] == 1234
    assert component["pbo"] is None


def test_run_spec_emits_a_gap_on_a_width_mismatch(monkeypatch):
    axis = _axis(200)
    inputs = _inputs(axis=axis)
    fake = _FakeSim({
        "as-configured": pd.Series(0.002, index=axis),
        "guard disabled": pd.Series(0.001, index=axis),
    })
    monkeypatch.setattr(harness, "simulate_arm", fake)

    component = harness.run_spec(
        _spec(lambda i: _two_arm_set(i.dates, widths=(2, 1))),
        inputs, n_trials=10,
    )

    assert component["status"] == "gap"
    assert component["value"] is None
    assert component["ci_low"] is None
    assert component["count_matched"] is False
    assert "width mismatch" in component["status_reason"]
    assert "baseline=2 vs ablated=1" in component["status_reason"]


def test_run_spec_emits_low_n_below_thirty_cycles(monkeypatch):
    axis = _axis(200)
    dates = [d.strftime("%Y-%m-%d") for d in axis[:12]]
    inputs = _inputs(axis=axis, dates=dates)
    fake = _FakeSim({
        "as-configured": pd.Series(0.002, index=axis),
        "guard disabled": pd.Series(0.001, index=axis),
    })
    monkeypatch.setattr(harness, "simulate_arm", fake)

    component = harness.run_spec(
        _spec(lambda i: _two_arm_set(i.dates)), inputs, n_trials=10
    )

    assert component["status"] == "N/A-LOW-N"
    assert component["value"] is None
    assert component["n_samples"] == 12
    assert component["n_floor"] == 60


def test_run_spec_passes_not_available_through():
    inputs = _inputs()
    na = NotAvailable(
        status="N/A-RETIRED",
        reason="team_candidates retired 2026-07-12 (alpha-engine-config-I7476)",
    )

    component = harness.run_spec(_spec(lambda i: na), inputs, n_trials=5)

    assert component["status"] == "N/A-RETIRED"
    assert component["status_reason"] == na.reason
    assert component["value"] is None
    assert component["arms"] == {}
    assert component["dsr"] is None
    assert component["pbo"] is None
    # Identity fields survive an N/A so the evaluator can still render the row.
    assert component["name"] == "risk_guard"
    assert component["criticality"] == "critical"
    assert component["issue"] == "alpha-engine-config-I7482"


def test_run_spec_rejects_a_build_arms_returning_junk():
    with pytest.raises(TypeError, match="expected ArmSet or NotAvailable"):
        harness.run_spec(_spec(lambda i: {"arms": []}), _inputs(), n_trials=1)


def test_run_spec_is_deterministic(monkeypatch):
    axis = _axis(200)
    inputs = _inputs(axis=axis)
    rs = np.random.RandomState(3)
    base = pd.Series(rs.normal(0.001, 0.005, len(axis)), index=axis)
    abl = pd.Series(rs.normal(0.0005, 0.005, len(axis)), index=axis)
    monkeypatch.setattr(
        harness, "simulate_arm",
        _FakeSim({"as-configured": base, "guard disabled": abl}),
    )
    spec = _spec(lambda i: _two_arm_set(i.dates))

    first = harness.run_spec(spec, inputs, n_trials=99)
    second = harness.run_spec(spec, inputs, n_trials=99)

    assert first == second


def test_run_spec_reports_pbo_only_for_a_swept_grid(monkeypatch):
    axis = _axis(200)
    inputs = _inputs(axis=axis)
    labels = {
        "as-configured": pd.Series(0.002, index=axis),
        "guard disabled": pd.Series(0.001, index=axis),
    }
    sweep = []
    for i in range(3):
        label = f"arm-{i}"
        rs = np.random.RandomState(i)
        labels[label] = pd.Series(rs.normal(0.001, 0.004, len(axis)), index=axis)
        sweep.append(
            picks_arm(label, [{"date": d, "picks": ["AAA", "BBB"]}
                              for d in inputs.dates])
        )
    monkeypatch.setattr(harness, "simulate_arm", _FakeSim(labels))

    def build(i):
        pair = _two_arm_set(i.dates)
        return ArmSet(baseline=pair.baseline, ablated=pair.ablated,
                      sweep=tuple(sweep))

    component = harness.run_spec(_spec(build), inputs, n_trials=7)

    assert component["pbo"] is not None
    assert component["pbo"]["n_trials"] == 5
    assert 0.0 <= component["pbo"]["pbo"] <= 1.0


def test_finite_coerces_inf_and_nan_to_null():
    assert harness._finite(float("inf")) is None
    assert harness._finite(float("nan")) is None
    assert harness._finite("x") is None
    assert harness._finite(0.5) == 0.5


# --------------------------------------------------------------------------
# The seed spec: behavioral.cost_adjusted_quality
# --------------------------------------------------------------------------


def _signals(dates: list[str]) -> dict:
    return {
        d: {
            "date": d,
            "signals": {
                "AAA": {"ticker": "AAA", "signal": "ENTER", "score": 80},
                "BBB": {"ticker": "BBB", "signal": "ENTER", "score": 75},
                "CCC": {"ticker": "CCC", "signal": "HOLD", "score": 40},
            },
        }
        for d in dates
    }


def test_cost_adjusted_quality_arms_differ_only_in_cost():
    axis = _axis(200)
    dates = [d.strftime("%Y-%m-%d") for d in axis[:60]]
    inputs = _inputs(axis=axis, dates=dates, signals_by_date=_signals(dates))

    arms = cost_adjusted_quality.build_arms(inputs)

    assert isinstance(arms, ArmSet)
    assert arms.baseline.picks == arms.ablated.picks
    assert arms.baseline.fees is None and arms.baseline.slippage_bps is None
    assert arms.ablated.fees == 0.0 and arms.ablated.slippage_bps == 0.0
    # HOLD names are not entered.
    assert arms.baseline.picks[0][1] == ("AAA", "BBB")


def test_cost_adjusted_quality_is_count_matched_by_construction():
    axis = _axis(200)
    dates = [d.strftime("%Y-%m-%d") for d in axis[:60]]
    inputs = _inputs(axis=axis, dates=dates, signals_by_date=_signals(dates))

    arms = cost_adjusted_quality.build_arms(inputs)

    match = objective.check_count_match(
        harness.picks_per_cycle(arms.baseline),
        harness.picks_per_cycle(arms.ablated),
    )
    assert match.matched is True


def test_cost_adjusted_quality_na_when_no_enter_signal():
    axis = _axis(200)
    dates = [d.strftime("%Y-%m-%d") for d in axis[:60]]
    signals = {d: {"signals": {"CCC": {"ticker": "CCC", "signal": "HOLD"}}}
               for d in dates}
    inputs = _inputs(axis=axis, dates=dates, signals_by_date=signals)

    result = cost_adjusted_quality.build_arms(inputs)

    assert isinstance(result, NotAvailable)
    assert result.status == "N/A-MISSING-INPUT"
    assert "no traded book to price" in result.reason
    assert "signals.json" in result.reason
    assert "I7484" in result.reason


def test_cost_adjusted_quality_na_when_the_cost_model_is_already_zero():
    axis = _axis(200)
    dates = [d.strftime("%Y-%m-%d") for d in axis[:60]]
    inputs = _inputs(axis=axis, dates=dates, signals_by_date=_signals(dates),
                     fees=0.0, slippage_bps=0.0)

    result = cost_adjusted_quality.build_arms(inputs)

    assert isinstance(result, NotAvailable)
    assert result.status == "N/A-MISSING-INPUT"
    assert "zero-cost arm is identical" in result.reason


def test_cost_adjusted_quality_measures_the_cost_drag_end_to_end(monkeypatch):
    """The zero-cost arm outperforms; the lift is the (negative) cost drag."""
    axis = _axis(200)
    dates = [d.strftime("%Y-%m-%d") for d in axis[:60]]
    inputs = _inputs(axis=axis, dates=dates, signals_by_date=_signals(dates))
    monkeypatch.setattr(harness, "simulate_arm", _FakeSim({
        f"as-configured — {harness.live_selection_label(inputs)}":
            pd.Series(0.0010, index=axis),
        "zero-cost (fees=0, slippage=0)": pd.Series(0.0012, index=axis),
    }))

    component = harness.run_spec(
        cost_adjusted_quality.SPEC, inputs, n_trials=42
    )

    assert component["status"] == "ok"
    assert component["module"] == "behavioral"
    assert component["criticality"] == "supporting"
    assert component["pattern"] == "null_arm"
    assert component["value"] == pytest.approx(H * -0.0002, abs=1e-9)
    assert component["value"] < 0.0  # trading costs money


def test_registry_specs_are_unique_and_valid():
    assert SPECS, "registry.SPECS must not be empty — the harness would be inert"
    names = [s.name for s in SPECS]
    assert len(names) == len(set(names)), f"duplicate spec names: {names}"
    for spec in SPECS:
        assert spec.issue.startswith("alpha-engine-config-I")
        assert callable(spec.build_arms)


# --------------------------------------------------------------------------
# Report shape — the contract
# --------------------------------------------------------------------------


def _report(monkeypatch, **kwargs) -> dict:
    axis = _axis(200)
    dates = [d.strftime("%Y-%m-%d") for d in axis[:60]]
    inputs = _inputs(axis=axis, dates=dates, signals_by_date=_signals(dates),
                     **kwargs)
    monkeypatch.setattr(harness, "simulate_arm", _FakeSim({
        f"as-configured — {harness.live_selection_label(inputs)}":
            pd.Series(0.0010, index=axis),
        "zero-cost (fees=0, slippage=0)": pd.Series(0.0012, index=axis),
    }))
    monkeypatch.setattr(report, "_account_trials", lambda inputs, n: 1234)
    return report.build_contribution_lift_report(inputs, SPECS)


#: Every top-level key the contract document specifies.
_TOP_LEVEL_KEYS = {
    "schema_version", "status", "run_date", "objective", "window",
    "inputs", "n_trials_cumulative", "components",
}

#: Every component key the contract document specifies.
_COMPONENT_KEYS = {
    "name", "module", "criticality", "pattern", "issue", "status",
    "status_reason", "value", "unit", "ci_low", "ci_high", "ci_method",
    "n_samples", "n_floor", "count_matched", "arms", "dsr", "pbo",
    "source_path", "support",
}

#: Every per-arm key the contract document specifies.
_ARM_KEYS = {
    "label", "n_orders", "picks_per_cycle", "total_return", "total_alpha",
    "sortino_ratio", "sharpe_ratio", "max_drawdown", "psr", "dsr",
    "per_cycle_log_alpha_21d",
}


def test_report_matches_the_contract_shape(monkeypatch):
    body = _report(monkeypatch)

    assert _TOP_LEVEL_KEYS <= set(body)
    assert body["schema_version"] == 1
    assert body["status"] == "ok"
    assert body["run_date"] == "2026-08-17"
    assert body["n_trials_cumulative"] == 1234

    assert body["objective"] == {
        "name": "log_alpha_21d_net_of_cost_vs_spy",
        "horizon_days": H,
        "fees": 0.001,
        "slippage_bps": 10.0,
        "init_cash": 1_000_000.0,
    }
    assert set(body["window"]) == {"start", "end", "n_cycles", "n_floor"}
    assert body["window"]["n_cycles"] == 60
    assert body["window"]["n_floor"] == 60
    assert set(body["inputs"]) == {
        "price_matrix_shape", "n_signal_dates", "source_paths"
    }
    assert body["inputs"]["price_matrix_shape"] == [200, 3]

    assert body["components"]
    for component in body["components"]:
        assert _COMPONENT_KEYS <= set(component)
        assert component["unit"] == "log_alpha_21d"
        assert component["ci_method"] == "bootstrap"
        assert component["n_floor"] == 60
        assert component["source_path"].startswith("s3://test-bucket/backtest/")
        assert "#components/" in component["source_path"]
        # An N/A component measured no arm, so it carries no DSR block
        # (`_na_component` emits null) — only a measured one must.
        if component["status"] == "ok":
            assert set(component["dsr"]) == {"baseline", "ablated", "n_trials"}
        else:
            assert component["dsr"] is None
        # alpha-engine-config-I12243: every replayed component carries the
        # versioned support census; a component that replayed nothing, null.
        if component["arms"]:
            assert component["support"]["schema_version"] == 1
            assert component["support"]["n_sampled_windows"] == component["n_samples"]
        else:
            assert component["support"] is None
        for arm in component["arms"].values():
            assert _ARM_KEYS <= set(arm)
            assert isinstance(arm["per_cycle_log_alpha_21d"], dict)


def test_report_is_strict_json_serializable(monkeypatch):
    import json

    body = _report(monkeypatch)
    text = json.dumps(body, allow_nan=False)

    assert "Infinity" not in text and "NaN" not in text


def test_report_horizon_comes_from_the_policy(monkeypatch):
    """21 is never hardcoded — the artifact reports what DEFAULT_POLICY says."""
    from nousergon_lib.quant.horizons import DEFAULT_POLICY

    body = _report(monkeypatch)

    assert body["objective"]["horizon_days"] == DEFAULT_POLICY.primary_horizon


def test_report_still_emits_when_the_loader_skips():
    """Always-emit: absence of the artifact must mean 'never ran', only."""
    inputs = _inputs(dates=[])
    inputs.status = "skipped"
    inputs.reason = "no signals/{date}/signals.json partitions"

    body = report.build_contribution_lift_report(inputs, SPECS)

    assert body["status"] == "skipped"
    assert body["reason"] == "no signals/{date}/signals.json partitions"
    assert body["components"] == []
    assert body["window"]["n_cycles"] == 0
    assert body["schema_version"] == 1


def test_report_is_deterministic(monkeypatch):
    assert _report(monkeypatch) == _report(monkeypatch)


# --------------------------------------------------------------------------
# Wire-in guards
# --------------------------------------------------------------------------


def test_evaluate_registers_contribution_lift_live():
    """The module must be registered AND must not be gated on an enabled flag.

    Regression target: the two prior replay precedents in this repo are wired
    but default-OFF, so neither has ever produced a number.
    """
    source = (REPO_ROOT / "evaluate.py").read_text()

    assert 'tracker.run_module(\n        "contribution_lift"' in source
    assert "run_contribution_lift(config" in source
    assert "from analysis.contribution_lift.report import run_contribution_lift" in source


def test_reporter_persists_contribution_lift_json():
    source = (REPO_ROOT / "reporter.py").read_text()

    assert "contribution_lift: dict | None = None," in source
    assert '("contribution_lift.json", contribution_lift),' in source


def _code_only(path: Path) -> str:
    """Source with every comment and string literal stripped.

    The prose in this package NAMES the retired tables and the number 21 (both
    deliberately, to say why they are not used). A guard matching raw text
    would fire on the explanation instead of on a real use.
    """
    import io
    import tokenize

    out: list[str] = []
    with open(path, "rb") as handle:
        for tok in tokenize.tokenize(io.BytesIO(handle.read()).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    return " ".join(out)


def _package_sources() -> list[tuple[Path, str]]:
    package = REPO_ROOT / "analysis" / "contribution_lift"
    return [(path, _code_only(path)) for path in sorted(package.rglob("*.py"))]


def test_no_retired_research_graph_table_is_read():
    """The six-team/CIO graph was retired 2026-07-12 — never an input here."""
    for path, code in _package_sources():
        assert "team_candidates" not in code, path
        assert "cio_evaluations" not in code, path


def test_horizon_is_not_hardcoded():
    """No literal 21 in executable code — it comes from DEFAULT_POLICY."""
    import re

    for path, code in _package_sources():
        assert not re.search(r"(?<![\w.])21(?![\w.])", code), (
            f"{path} hardcodes 21; use HORIZON_DAYS "
            "(nousergon_lib.quant.horizons.DEFAULT_POLICY.primary_horizon)"
        )


def test_no_group_reads_the_signals_enter_feed_directly():
    """config-I7501: one champion-aware live-selection reader, in the harness.

    Entry selection left ``signals.json`` on 2026-07-13 (the
    ``scanner_predictor_direct`` champion cutover), and every group that had
    re-implemented its own ENTER-picks reader over that feed silently began
    measuring an empty book. The rule that survives the class is structural:
    a group module may not read the ``signal`` discriminator at all, and may
    not define a private ENTER reader — it calls the harness helper.
    """
    groups = REPO_ROOT / "analysis" / "contribution_lift" / "groups"
    for path in sorted(groups.rglob("*.py")):
        code = _code_only(path)
        # The discriminator is only ever spelled as this module-level constant
        # (string literals are stripped by _code_only), so its presence in a
        # group IS a private read of the signals feed. Verified against the
        # pre-fix sources: it fires on all five groups that had one.
        assert not re.search(r"(?<![\w.])_ENTER(?![\w])", code), (
            f"{path.name} reads the signals.json ENTER discriminator directly; "
            "use harness.live_picks_by_cycle / live_picks_by_date / live_widths "
            "(config-I7501)"
        )
        for private in ("_enter_picks", "_enter_picks_by_date", "_enter_tickers",
                        "_enter_candidates"):
            assert f"def {private}" not in code, (
                f"{path.name} defines a private live-selection reader "
                f"({private}); the harness owns that reader (config-I7501)"
            )


def test_live_selection_prefers_the_executed_trades_over_signals(tmp_path):
    """trades.db is the ground truth in BOTH eras; signals is only a fallback."""
    axis = _axis(200)
    dates = [d.strftime("%Y-%m-%d") for d in axis[:60]]
    db = _trades_db(tmp_path / "trades.db", {d: ["AAA", "CCC"] for d in dates})
    inputs = _inputs(
        axis=axis, dates=dates, signals_by_date=_signals(dates), trades_db_path=db
    )

    selection = harness.live_selection_of(inputs)

    assert selection.source == harness.SOURCE_TRADES
    # signals.json says AAA+BBB; the executed record says AAA+CCC and wins.
    assert harness.live_picks_by_date(inputs)[dates[0]] == ("AAA", "CCC")
    assert "trades.db" in harness.live_selection_label(inputs)


def test_live_selection_falls_back_to_signals_without_a_trades_db():
    axis = _axis(200)
    dates = [d.strftime("%Y-%m-%d") for d in axis[:60]]
    inputs = _inputs(axis=axis, dates=dates, signals_by_date=_signals(dates))

    selection = harness.live_selection_of(inputs)

    assert selection.source == harness.SOURCE_SIGNALS
    assert harness.live_picks_by_date(inputs)[dates[0]] == ("AAA", "BBB")


def test_live_selection_survives_the_champion_era_empty_signals_feed(tmp_path):
    """The exact I7501 failure: every signals row HOLD, a real executed book.

    Before the fix this window produced zero picks and every picks-based
    component graded ``N/A-MISSING-INPUT`` while the system was trading.
    """
    axis = _axis(200)
    dates = [d.strftime("%Y-%m-%d") for d in axis[:60]]
    all_hold = {
        d: {"signals": {t: {"ticker": t, "signal": "HOLD"} for t in ("AAA", "BBB")}}
        for d in dates
    }
    db = _trades_db(tmp_path / "trades.db", {d: ["AAA", "BBB"] for d in dates})
    inputs = _inputs(
        axis=axis, dates=dates, signals_by_date=all_hold, trades_db_path=db
    )

    arms = cost_adjusted_quality.build_arms(inputs)

    assert isinstance(arms, ArmSet)
    assert len(arms.baseline.picks) == len(dates)
    assert "trades.db" in arms.baseline.label


def test_live_picks_drop_unpriceable_names_and_out_of_window_cycles(tmp_path):
    """Declared width must equal traded width, or count-matching breaks."""
    axis = _axis(200)
    dates = [d.strftime("%Y-%m-%d") for d in axis[:60]]
    outside = axis[80].strftime("%Y-%m-%d")
    db = _trades_db(
        tmp_path / "trades.db",
        {dates[0]: ["AAA", "ZZZ"], outside: ["AAA"]},
    )
    inputs = _inputs(axis=axis, dates=dates, trades_db_path=db)

    picks = harness.live_picks_by_date(inputs)

    assert picks == {dates[0]: ("AAA",)}  # ZZZ unpriceable, `outside` off-window


def test_read_trades_enters_truncates_a_timestamped_date(tmp_path):
    db = _trades_db(tmp_path / "trades.db", {"2026-08-17T14:30:00Z": ["AAA"]})

    assert harness.read_trades_enters(db) == {"2026-08-17": ("AAA",)}


def test_read_trades_enters_tolerates_a_fresh_database(tmp_path):
    """A trades.db with no trades table yet is a real state, not a crash."""
    path = tmp_path / "empty.db"
    sqlite3.connect(path).close()

    assert harness.read_trades_enters(str(path)) == {}
    assert harness.read_trades_enters(str(tmp_path / "absent.db")) == {}


def test_unit_literal_matches_the_krepis_metric_unit():
    """krepis#158 widened MetricTypeLiteral with `contribution_lift` + `unit`."""
    assert harness.UNIT == "log_alpha_21d"
    assert harness.TRIAL_PRODUCER == "contribution_lift"


def test_min_samples_is_half_the_floor():
    assert harness.MIN_SAMPLES == harness.N_FLOOR // 2
    assert harness.N_FLOOR == 60


def test_log_alpha_sum_telescopes_to_the_nav_ratio():
    """Independent oracle: Sigma of daily log returns == log(NAV_end / NAV_start)."""
    axis = _axis(60)
    rs = np.random.RandomState(11)
    simple = pd.Series(rs.normal(0.0005, 0.01, len(axis)), index=axis)
    nav = 1_000_000.0 * (1 + simple).cumprod()
    log_returns = np.log1p(simple)
    spy = pd.Series(100.0, index=axis)
    cycle = axis[0].strftime("%Y-%m-%d")

    out = objective.per_cycle_log_alpha(
        log_returns, spy, axis, [cycle], horizon_days=H
    )

    assert out[cycle] == pytest.approx(
        math.log(float(nav.iloc[H]) / float(nav.iloc[0])), abs=1e-10
    )


# --------------------------------------------------------------------------
# The REAL simulator — no fake
# --------------------------------------------------------------------------
#
# Every test above injects a fake simulator so the objective maths is isolated.
# That leaves exactly the gap this repo has been burned by before: a
# well-tested unit whose real call site is never exercised. These two run the
# genuine `vectorbt_bridge.orders_to_portfolio` + `portfolio_stats` path on a
# small synthetic price matrix — still hermetic (no AWS, no ArcticDB).


def _trending_inputs(n: int = 140, drift: float = 0.001) -> ReplayInputs:
    axis = _axis(n)
    steps = np.arange(len(axis))
    prices = pd.DataFrame(
        {
            "AAA": 100.0 * np.exp(drift * steps),
            "BBB": 100.0 * np.exp(drift * steps),
        },
        index=axis,
    )
    dates = [d.strftime("%Y-%m-%d") for d in axis[: n - H - 5]]
    return ReplayInputs(
        run_date="2026-08-17",
        dates=dates,
        signals_by_date=_signals(dates),
        predictions_by_date={},
        pillar_profiles_by_date={},
        price_matrix=prices,
        spy_prices=pd.Series(100.0 * np.exp(0.0002 * steps), index=axis),
        bucket="test-bucket",
        fees=0.001,
        slippage_bps=10.0,
        init_cash=1_000_000.0,
        source_paths=[],
    )


def test_simulate_arm_runs_the_real_vectorbt_path():
    inputs = _trending_inputs()
    arms = cost_adjusted_quality.build_arms(inputs)

    stats = harness.simulate_arm(arms.baseline, arms, inputs)

    assert isinstance(stats["daily_log_returns"], pd.Series)
    assert len(stats["daily_log_returns"]) > 0
    assert stats["n_orders"] > 0
    assert np.isfinite(stats["total_return"])


def test_cost_drag_is_negative_through_the_real_simulator():
    """End to end, no fake: the zero-cost arm must beat the as-configured arm.

    This is the seed spec's whole claim — 10bps fees + 10bps slippage cost the
    book something — and it is checked against the production cost model, not
    an assertion about it.
    """
    inputs = _trending_inputs()

    component = harness.run_spec(
        cost_adjusted_quality.SPEC, inputs, n_trials=1
    )

    assert component["status"] == "ok"
    assert component["count_matched"] is True
    assert component["n_samples"] >= harness.MIN_SAMPLES
    assert component["value"] < 0.0, "trading costs must reduce net-of-cost alpha"
    assert component["arms"]["baseline"]["n_orders"] == (
        component["arms"]["ablated"]["n_orders"]
    )


# --------------------------------------------------------------------------
# The live loader (hermetic: stubbed S3, never boto3)
# --------------------------------------------------------------------------


def test_loader_reads_the_production_cost_keys():
    """Same keys `backtest._run_simulation_loop` reads — not a second default set."""
    from analysis.contribution_lift import inputs as inputs_mod

    fees, slippage_bps, init_cash = inputs_mod._cost_model({
        "simulation_fees": 0.002,
        "simulation": {"slippage_bps": 15},
        "init_cash": 250_000.0,
    })

    assert (fees, slippage_bps, init_cash) == (0.002, 15.0, 250_000.0)


def test_loader_window_excludes_the_horizon_tail():
    """Every returned cycle has a FULL horizon of price rows after it."""
    from analysis.contribution_lift import inputs as inputs_mod

    axis = _axis(50)
    signal_dates = [d.strftime("%Y-%m-%d") for d in axis]

    window = inputs_mod._select_window(signal_dates, axis, lookback=90)

    assert window
    assert window[-1] == axis[50 - H - 1].strftime("%Y-%m-%d")
    assert len(window) == 50 - H


def test_loader_window_honours_the_lookback():
    from analysis.contribution_lift import inputs as inputs_mod

    axis = _axis(200)
    signal_dates = [d.strftime("%Y-%m-%d") for d in axis]

    window = inputs_mod._select_window(signal_dates, axis, lookback=90)

    assert len(window) == 90


class _EmptyS3:
    """S3 stub with no signals/ partitions at all."""

    def get_paginator(self, _op):
        class _P:
            def paginate(self, **_kwargs):
                return iter([{}])
        return _P()


def test_loader_skips_when_there_are_no_signal_dates():
    """Zero signal dates is the ONLY degrade — and it still emits a report."""
    from analysis.contribution_lift.inputs import load_replay_inputs

    result = load_replay_inputs(
        {"signals_bucket": "test-bucket"},
        run_date="2026-08-17",
        s3_client=_EmptyS3(),
    )

    assert result.status == "skipped"
    assert "signals.json" in (result.reason or "")
    assert result.dates == []

    body = report.build_contribution_lift_report(result, SPECS)
    assert body["status"] == "skipped"
    assert body["components"] == []
