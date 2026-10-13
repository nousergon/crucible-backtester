"""tests/test_contribution_lift_executor_rules.py — RC v3 T5 executor group.

Covers ``analysis/contribution_lift/groups/executor_rules.py``: the four
null-arm specs built from a ``trades.db`` sqlite fixture —
``exit_rules``/``position_sizing``/``entry_triggers`` (alpha-engine-config-
I7481) and ``risk_guard`` (alpha-engine-config-I7482).

Structural assertions (labels, order shapes, N/A branches) run against a
tiny in-memory-backed sqlite fixture with ``harness.simulate_arm``
monkeypatched, mirroring ``tests/test_contribution_lift.py``'s own pattern.
One end-to-end test per issue runs the REAL ``vectorbt_bridge`` path (no
mocking) on a small synthetic price matrix, so the trades.db -> orders ->
simulator wiring is proven, not just the arm-construction maths.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.contribution_lift import harness  # noqa: E402
from analysis.contribution_lift.groups import executor_rules  # noqa: E402
from analysis.contribution_lift.harness import HORIZON_DAYS, NotAvailable, ReplayInputs  # noqa: E402
from analysis.contribution_lift.registry import SPECS  # noqa: E402

H = HORIZON_DAYS


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _axis(n: int = 200) -> pd.DatetimeIndex:
    return pd.bdate_range("2026-01-01", periods=n)


def _make_trades_db(
    tmp_path: Path,
    roundtrips: list[dict],
    shadow_rows: list[dict] | None = None,
) -> str:
    """A minimal sqlite trades.db: just the columns this module reads.

    ``roundtrips``: ``{"date", "ticker", "shares", "fill_price",
    "signal_date", "exit_date", "exit_days_held"}``.
    ``shadow_rows``: ``{"date", "ticker", "block_reason", "intended_shares",
    "intended_dollars", "current_price"}``.
    """
    db_path = tmp_path / "trades.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE trades (trade_id TEXT, date TEXT, ticker TEXT, "
        "action TEXT, shares REAL, fill_price REAL, signal_date TEXT, "
        "entry_trade_id TEXT, days_held REAL)"
    )
    conn.execute(
        "CREATE TABLE executor_shadow_book (date TEXT, ticker TEXT, "
        "block_reason TEXT, intended_shares REAL, intended_dollars REAL, "
        "current_price REAL)"
    )
    for i, rt in enumerate(roundtrips):
        trade_id = f"t{i}"
        conn.execute(
            "INSERT INTO trades (trade_id, date, ticker, action, shares, "
            "fill_price, signal_date, entry_trade_id, days_held) "
            "VALUES (?, ?, ?, 'ENTER', ?, ?, ?, NULL, NULL)",
            (trade_id, rt["date"], rt["ticker"], rt["shares"],
             rt["fill_price"], rt.get("signal_date")),
        )
        conn.execute(
            "INSERT INTO trades (trade_id, date, ticker, action, shares, "
            "fill_price, signal_date, entry_trade_id, days_held) "
            "VALUES (?, ?, ?, 'EXIT', NULL, NULL, NULL, ?, ?)",
            (f"{trade_id}-exit", rt["exit_date"], rt["ticker"], trade_id,
             rt.get("exit_days_held")),
        )
    for row in shadow_rows or []:
        conn.execute(
            "INSERT INTO executor_shadow_book (date, ticker, block_reason, "
            "intended_shares, intended_dollars, current_price) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (row["date"], row["ticker"], row["block_reason"],
             row.get("intended_shares"), row.get("intended_dollars"),
             row.get("current_price")),
        )
    conn.commit()
    conn.close()
    return str(db_path)


def _inputs(*, trades_db_path: str | None, axis: pd.DatetimeIndex | None = None) -> ReplayInputs:
    axis = axis if axis is not None else _axis()
    tickers = ["AAA", "BBB", "CCC"]
    prices = pd.DataFrame(100.0, index=axis, columns=tickers)
    dates = [d.strftime("%Y-%m-%d") for d in axis[:60]]
    return ReplayInputs(
        run_date="2026-08-17",
        dates=dates,
        signals_by_date={},
        predictions_by_date={},
        pillar_profiles_by_date={},
        price_matrix=prices,
        spy_prices=pd.Series(100.0, index=axis),
        bucket="test-bucket",
        fees=0.001,
        slippage_bps=10.0,
        init_cash=1_000_000.0,
        trades_db_path=trades_db_path,
        source_paths=[],
    )


_ROUNDTRIPS = [
    {"date": "2026-01-05", "ticker": "AAA", "shares": 100.0, "fill_price": 100.0,
     "signal_date": "2026-01-02", "exit_date": "2026-01-20", "exit_days_held": 10},
    {"date": "2026-01-05", "ticker": "BBB", "shares": 200.0, "fill_price": 100.0,
     "signal_date": "2026-01-05", "exit_date": "2026-01-15", "exit_days_held": 7},
    {"date": "2026-01-12", "ticker": "AAA", "shares": 50.0, "fill_price": 100.0,
     "signal_date": "2026-01-08", "exit_date": "2026-01-22", "exit_days_held": 8},
]


class _FakeSim:
    """Same shape as the harness suite's fake: label -> canned log-return series."""

    def __init__(self, by_label: dict[str, pd.Series]):
        self.by_label = by_label

    def __call__(self, arm, arm_set, inputs):
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
# N/A-MISSING-INPUT — trades_db_path absent (the live steady state today)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("build_arms,name", [
    (executor_rules._exit_rules_build_arms, "exit_rules"),
    (executor_rules._position_sizing_build_arms, "position_sizing"),
    (executor_rules._entry_triggers_build_arms, "entry_triggers"),
    (executor_rules._risk_guard_build_arms, "risk_guard"),
])
def test_missing_trades_db_path_is_na_for_every_component(build_arms, name):
    result = build_arms(_inputs(trades_db_path=None))

    assert isinstance(result, NotAvailable)
    assert result.status == "N/A-MISSING-INPUT"
    assert "trades_db_path is None" in result.reason


@pytest.mark.parametrize("build_arms", [
    executor_rules._exit_rules_build_arms,
    executor_rules._position_sizing_build_arms,
    executor_rules._entry_triggers_build_arms,
    executor_rules._risk_guard_build_arms,
])
def test_no_roundtrips_is_na(build_arms, tmp_path):
    db_path = _make_trades_db(tmp_path, [])
    result = build_arms(_inputs(trades_db_path=db_path))

    assert isinstance(result, NotAvailable)
    assert result.status == "N/A-MISSING-INPUT"


def test_risk_guard_na_when_shadow_book_is_empty(tmp_path):
    db_path = _make_trades_db(tmp_path, _ROUNDTRIPS, shadow_rows=[])

    result = executor_rules._risk_guard_build_arms(_inputs(trades_db_path=db_path))

    assert isinstance(result, NotAvailable)
    assert "executor_shadow_book has no blocked-entry rows" in result.reason


def test_entry_triggers_na_when_signal_date_never_populated(tmp_path):
    roundtrips = [dict(rt, signal_date=None) for rt in _ROUNDTRIPS]
    db_path = _make_trades_db(tmp_path, roundtrips)

    result = executor_rules._entry_triggers_build_arms(_inputs(trades_db_path=db_path))

    assert isinstance(result, NotAvailable)
    assert "signal_date is NULL" in result.reason


# --------------------------------------------------------------------------
# Arm construction — exit_rules
# --------------------------------------------------------------------------


def test_exit_rules_baseline_uses_actual_exit_dates(tmp_path):
    db_path = _make_trades_db(tmp_path, _ROUNDTRIPS)

    arms = executor_rules._exit_rules_build_arms(_inputs(trades_db_path=db_path))

    exits = [o for o in arms.baseline.orders if o["action"] == "EXIT"]
    assert {o["date"] for o in exits} == {"2026-01-20", "2026-01-15", "2026-01-22"}


def test_exit_rules_ablated_exits_30_trading_days_after_entry(tmp_path):
    db_path = _make_trades_db(tmp_path, _ROUNDTRIPS)

    arms = executor_rules._exit_rules_build_arms(_inputs(trades_db_path=db_path))

    axis = pd.DatetimeIndex(_inputs(trades_db_path=db_path).price_matrix.index).sort_values()
    for entry_date_str in {"2026-01-05", "2026-01-12"}:
        entry = pd.Timestamp(entry_date_str)
        expected_pos = min(axis.get_loc(entry) + 30, len(axis) - 1)
        expected = axis[expected_pos].strftime("%Y-%m-%d")
        matching_enters = [
            o for o in arms.ablated.orders
            if o["action"] == "ENTER" and o["date"] == entry_date_str
        ]
        assert matching_enters, f"no ablated ENTER on {entry_date_str}"
        for enter in matching_enters:
            exits_for_ticker = [
                o for o in arms.ablated.orders
                if o["action"] == "EXIT" and o["ticker"] == enter["ticker"]
                and o["date"] != "2026-01-20" and o["date"] != "2026-01-15"
                and o["date"] != "2026-01-22"
            ]
            assert any(o["date"] == expected for o in exits_for_ticker)


def test_exit_rules_same_entry_shares_and_tickers_both_arms(tmp_path):
    db_path = _make_trades_db(tmp_path, _ROUNDTRIPS)

    arms = executor_rules._exit_rules_build_arms(_inputs(trades_db_path=db_path))

    base_enters = sorted(
        (o["date"], o["ticker"], o["shares"])
        for o in arms.baseline.orders if o["action"] == "ENTER"
    )
    abl_enters = sorted(
        (o["date"], o["ticker"], o["shares"])
        for o in arms.ablated.orders if o["action"] == "ENTER"
    )
    assert base_enters == abl_enters


# --------------------------------------------------------------------------
# Arm construction — position_sizing
# --------------------------------------------------------------------------


def test_position_sizing_ablated_is_equal_weight(tmp_path):
    db_path = _make_trades_db(tmp_path, _ROUNDTRIPS)
    inputs = _inputs(trades_db_path=db_path)

    arms = executor_rules._position_sizing_build_arms(inputs)

    # 2026-01-05 has 2 names (AAA, BBB) -> init_cash/2/price=100 -> 5000 shares each.
    jan5_enters = [
        o for o in arms.ablated.orders
        if o["action"] == "ENTER" and o["date"] == "2026-01-05"
    ]
    assert {o["shares"] for o in jan5_enters} == {5000.0}
    # 2026-01-12 has 1 name (AAA) -> init_cash/1/100 -> 10000 shares.
    jan12_enters = [
        o for o in arms.ablated.orders
        if o["action"] == "ENTER" and o["date"] == "2026-01-12"
    ]
    assert {o["shares"] for o in jan12_enters} == {10000.0}


def test_position_sizing_keeps_baseline_entry_and_exit_dates(tmp_path):
    db_path = _make_trades_db(tmp_path, _ROUNDTRIPS)

    arms = executor_rules._position_sizing_build_arms(_inputs(trades_db_path=db_path))

    base_dates = sorted((o["date"], o["ticker"]) for o in arms.baseline.orders)
    abl_dates = sorted((o["date"], o["ticker"]) for o in arms.ablated.orders)
    assert base_dates == abl_dates


# --------------------------------------------------------------------------
# Arm construction — entry_triggers
# --------------------------------------------------------------------------


def test_entry_triggers_ablated_uses_signal_date(tmp_path):
    db_path = _make_trades_db(tmp_path, _ROUNDTRIPS)

    arms = executor_rules._entry_triggers_build_arms(_inputs(trades_db_path=db_path))

    abl_enter_dates = sorted(
        o["date"] for o in arms.ablated.orders if o["action"] == "ENTER"
    )
    # AAA/2026-01-05 -> signal_date 2026-01-02; BBB/2026-01-05 -> no delay
    # (signal_date == date); AAA/2026-01-12 -> signal_date 2026-01-08.
    assert abl_enter_dates == ["2026-01-02", "2026-01-05", "2026-01-08"]


def test_entry_triggers_ablated_keeps_baseline_shares_and_exit(tmp_path):
    db_path = _make_trades_db(tmp_path, _ROUNDTRIPS)

    arms = executor_rules._entry_triggers_build_arms(_inputs(trades_db_path=db_path))

    base_by_ticker_date = {
        (o["ticker"], o["date"]): o["shares"]
        for o in arms.baseline.orders if o["action"] == "ENTER"
    }
    for order in arms.ablated.orders:
        if order["action"] != "ENTER":
            continue
        # AAA appears twice in the baseline (two roundtrips); shares must
        # match one of them by ticker (dates differ by construction here).
        assert order["shares"] in {
            v for (t, _d), v in base_by_ticker_date.items() if t == order["ticker"]
        }
    abl_exits = sorted(o["date"] for o in arms.ablated.orders if o["action"] == "EXIT")
    base_exits = sorted(o["date"] for o in arms.baseline.orders if o["action"] == "EXIT")
    assert abl_exits == base_exits


# --------------------------------------------------------------------------
# Arm construction — risk_guard
# --------------------------------------------------------------------------


# Shadow blocks land on the SAME dates as baseline entries (the executor
# evaluates a day's whole candidate list together, so a block and an
# executed entry are typically same-day siblings) — this is what makes the
# ablated arm's per-date width diverge from the baseline's on a SHARED
# cycle. Since alpha-engine-config-I12243 `check_count_match` classes the
# UNION of both arms' dates, so a block on a day with no executed entry is
# named as ablated-only support too (tests/test_contribution_lift_support.py).
_SHADOW_ROWS = [
    {"date": "2026-01-05", "ticker": "CCC", "block_reason": "score 40.0 < min 55.0",
     "intended_shares": 300.0, "intended_dollars": None, "current_price": 100.0},
    {"date": "2026-01-12", "ticker": "CCC", "block_reason": "stance gate: DOWN veto",
     "intended_shares": None, "intended_dollars": 15000.0, "current_price": 100.0},
]


def test_risk_guard_ablated_is_wider_than_baseline(tmp_path):
    db_path = _make_trades_db(tmp_path, _ROUNDTRIPS, shadow_rows=_SHADOW_ROWS)

    arms = executor_rules._risk_guard_build_arms(_inputs(trades_db_path=db_path))

    base_tickers = {o["ticker"] for o in arms.baseline.orders if o["action"] == "ENTER"}
    abl_tickers = {o["ticker"] for o in arms.ablated.orders if o["action"] == "ENTER"}
    assert base_tickers == {"AAA", "BBB"}
    assert abl_tickers == {"AAA", "BBB", "CCC"}
    ccc_enters = [
        o for o in arms.ablated.orders
        if o["ticker"] == "CCC" and o["action"] == "ENTER"
    ]
    assert {o["shares"] for o in ccc_enters} == {300.0, 150.0}  # 15000/100=150


def test_risk_guard_ablated_exits_ccc_at_baseline_median_hold(tmp_path):
    db_path = _make_trades_db(tmp_path, _ROUNDTRIPS, shadow_rows=_SHADOW_ROWS)
    inputs = _inputs(trades_db_path=db_path)

    arms = executor_rules._risk_guard_build_arms(inputs)

    axis = pd.DatetimeIndex(inputs.price_matrix.index).sort_values()
    # median days_held over the 3 baseline roundtrips (10, 7, 8) -> 8.
    entry = pd.Timestamp("2026-01-05")
    expected_pos = min(axis.get_loc(entry) + 8, len(axis) - 1)
    expected_exit = axis[expected_pos].strftime("%Y-%m-%d")
    ccc_exits = [
        o["date"] for o in arms.ablated.orders
        if o["ticker"] == "CCC" and o["action"] == "EXIT" and o["date"] == expected_exit
    ]
    assert ccc_exits


# --------------------------------------------------------------------------
# run_spec through the harness (mocked simulator, real objective/CI maths)
# --------------------------------------------------------------------------


def test_run_spec_reports_gap_for_risk_guard_width_mismatch(tmp_path, monkeypatch):
    db_path = _make_trades_db(tmp_path, _ROUNDTRIPS, shadow_rows=_SHADOW_ROWS)
    axis = _axis(200)
    inputs = _inputs(trades_db_path=db_path, axis=axis)
    arms = executor_rules._risk_guard_build_arms(inputs)

    monkeypatch.setattr(harness, "simulate_arm", _FakeSim({
        arms.baseline.label: pd.Series(0.001, index=axis),
        arms.ablated.label: pd.Series(0.001, index=axis),
    }))

    component = harness.run_spec(executor_rules.SPEC_RISK_GUARD, inputs, n_trials=1)

    assert component["status"] == "gap"
    assert "width mismatch" in component["status_reason"]
    assert component["name"] == "risk_guard"
    assert component["issue"] == "alpha-engine-config-I7482"


def test_run_spec_measures_exit_rules_lift_with_mocked_sim(tmp_path, monkeypatch):
    db_path = _make_trades_db(tmp_path, _ROUNDTRIPS)
    axis = _axis(200)
    inputs = _inputs(trades_db_path=db_path, axis=axis)
    arms = executor_rules._exit_rules_build_arms(inputs)

    monkeypatch.setattr(harness, "simulate_arm", _FakeSim({
        arms.baseline.label: pd.Series(0.002, index=axis),
        arms.ablated.label: pd.Series(0.001, index=axis),
    }))

    component = harness.run_spec(executor_rules.SPEC_EXIT_RULES, inputs, n_trials=1)

    assert component["status"] == "ok"
    assert component["count_matched"] is True
    assert component["value"] == pytest.approx(H * 0.001, abs=1e-9)
    assert component["name"] == "exit_rules"
    assert component["issue"] == "alpha-engine-config-I7481"


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_registry_carries_all_four_executor_rules_components():
    names = {s.name for s in SPECS}
    assert {"exit_rules", "position_sizing", "entry_triggers", "risk_guard"} <= names


def test_specs_have_the_evaluator_tile_criticalities():
    by_name = {s.name: s for s in SPECS}
    assert by_name["entry_triggers"].criticality == "critical"
    assert by_name["risk_guard"].criticality == "critical"
    assert by_name["exit_rules"].criticality == "critical"
    assert by_name["position_sizing"].criticality == "supporting"
    for name in ("entry_triggers", "risk_guard", "exit_rules", "position_sizing"):
        assert by_name[name].module == "executor"
        assert by_name[name].pattern == "null_arm"


# --------------------------------------------------------------------------
# Real vectorbt path (no mocking) — one per issue
# --------------------------------------------------------------------------


def _trending_inputs(trades_db_path: str, n: int = 140, drift: float = 0.001) -> ReplayInputs:
    axis = _axis(n)
    steps = np.arange(len(axis))
    prices = pd.DataFrame(
        {"AAA": 100.0 * np.exp(drift * steps), "BBB": 100.0 * np.exp(drift * steps),
         "CCC": 100.0 * np.exp(drift * steps)},
        index=axis,
    )
    dates = [d.strftime("%Y-%m-%d") for d in axis[: n - H - 5]]
    return ReplayInputs(
        run_date="2026-08-17",
        dates=dates,
        signals_by_date={},
        predictions_by_date={},
        pillar_profiles_by_date={},
        price_matrix=prices,
        spy_prices=pd.Series(100.0 * np.exp(0.0002 * steps), index=axis),
        bucket="test-bucket",
        fees=0.001,
        slippage_bps=10.0,
        init_cash=1_000_000.0,
        trades_db_path=trades_db_path,
        source_paths=[],
    )


def test_position_sizing_runs_the_real_vectorbt_path(tmp_path):
    roundtrips = [
        {"date": "2026-01-05", "ticker": "AAA", "shares": 100.0, "fill_price": 100.0,
         "signal_date": "2026-01-05", "exit_date": "2026-02-02", "exit_days_held": 20},
        {"date": "2026-01-05", "ticker": "BBB", "shares": 100.0, "fill_price": 100.0,
         "signal_date": "2026-01-05", "exit_date": "2026-02-02", "exit_days_held": 20},
    ]
    db_path = _make_trades_db(tmp_path, roundtrips)
    inputs = _trending_inputs(db_path)

    component = harness.run_spec(executor_rules.SPEC_POSITION_SIZING, inputs, n_trials=1)

    assert component["status"] in ("ok", "N/A-LOW-N")
    assert component["arms"]["baseline"]["n_orders"] > 0
    assert component["arms"]["ablated"]["n_orders"] > 0


def test_risk_guard_runs_the_real_vectorbt_path_and_reports_gap(tmp_path):
    roundtrips = [
        {"date": "2026-01-05", "ticker": "AAA", "shares": 100.0, "fill_price": 100.0,
         "signal_date": "2026-01-05", "exit_date": "2026-02-02", "exit_days_held": 20},
    ]
    shadow_rows = [
        {"date": "2026-01-05", "ticker": "BBB", "block_reason": "GBM veto",
         "intended_shares": 50.0, "intended_dollars": None, "current_price": 100.0},
    ]
    db_path = _make_trades_db(tmp_path, roundtrips, shadow_rows=shadow_rows)
    inputs = _trending_inputs(db_path)

    component = harness.run_spec(executor_rules.SPEC_RISK_GUARD, inputs, n_trials=1)

    # Wider by construction on the shadow-blocked cycle -> gap, per the
    # issue's own "all flagged trades allowed to execute" framing (spec §3).
    assert component["status"] == "gap"
    assert "width mismatch" in component["status_reason"]
    assert component["value"] is None
