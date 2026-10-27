"""tests/test_contribution_lift_reduce_legs.py — alpha-engine-config-I12683.

The executor null arms model each partial ``REDUCE`` leg: the baseline sells
each leg's own shares on its own date and the terminal EXIT sells the rest,
so the simulated book is the as-executed one. ``reduce_*`` families ablate
those legs one family at a time (the reduced shares are held to the family's
fixed hold instead), entries untouched, so both arms count-match.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.contribution_lift import report  # noqa: E402
from analysis.contribution_lift.groups import executor_rules as er  # noqa: E402
from analysis.contribution_lift.harness import (  # noqa: E402
    NotAvailable,
    ReplayInputs,
    has_partial_exits,
)
from vectorbt_bridge import orders_to_portfolio  # noqa: E402


def _axis(n: int = 200) -> pd.DatetimeIndex:
    return pd.bdate_range("2026-01-01", periods=n)


# (trade_id, entry date, ticker, shares, terminal exit date or None, exit reason)
_ENTRIES = [
    ("t0", "2026-01-05", "AAA", 100.0, "2026-01-20", "optimizer_target_zero"),
    ("t1", "2026-01-05", "BBB", 200.0, "2026-01-15", "time_decay_exit"),
    ("t2", "2026-01-12", "CCC", 50.0, "2026-01-14", "intraday_trailing_stop"),
    ("t3", "2026-01-12", "DDD", 80.0, None, None),  # still open
]
# (entry trade_id, date, shares, reason)
_REDUCES = [
    ("t0", "2026-01-08", 40.0, "optimizer_scale_down"),
    ("t0", "2026-01-12", 20.0, "time_decay_reduce"),
    ("t1", "2026-01-09", 50.0, "intraday_profit_take"),
    ("t3", "2026-01-13", 30.0, "optimizer_scale_down"),  # open roundtrip
    ("t2", "2026-01-16", 10.0, "optimizer_scale_down"),  # after its terminal exit
]
_TICKERS = ["AAA", "BBB", "CCC", "DDD"]


def _make_db(tmp_path: Path, *, with_reason: bool = True, reduces=_REDUCES) -> str:
    db_path = tmp_path / "trades.db"
    conn = sqlite3.connect(str(db_path))
    reason_col = ", exit_reason TEXT" if with_reason else ""
    conn.execute(
        "CREATE TABLE trades (trade_id TEXT, date TEXT, ticker TEXT, action TEXT, "
        "shares REAL, fill_price REAL, signal_date TEXT, entry_trade_id TEXT, "
        f"days_held REAL{reason_col})"
    )
    tickers = {}
    for tid, d, t, sh, xd, why in _ENTRIES:
        tickers[tid] = t
        conn.execute(
            "INSERT INTO trades (trade_id, date, ticker, action, shares, fill_price, "
            "signal_date) VALUES (?, ?, ?, 'ENTER', ?, 100.0, ?)",
            (tid, d, t, sh, d),
        )
        if xd is None:
            continue
        cols, vals = "trade_id, date, ticker, action, entry_trade_id", [f"{tid}-x", xd, t, "EXIT", tid]
        if with_reason:
            cols, vals = cols + ", exit_reason", vals + [why]
        conn.execute(f"INSERT INTO trades ({cols}) VALUES ({', '.join('?' * len(vals))})", vals)
    for i, (tid, d, sh, why) in enumerate(reduces):
        cols = "trade_id, date, ticker, action, shares, entry_trade_id"
        vals = [f"{tid}-r{i}", d, tickers[tid], "REDUCE", sh, tid]
        if with_reason:
            cols, vals = cols + ", exit_reason", vals + [why]
        conn.execute(f"INSERT INTO trades ({cols}) VALUES ({', '.join('?' * len(vals))})", vals)
    conn.commit()
    conn.close()
    return str(db_path)


def _inputs(trades_db_path: str | None, *, prices: pd.DataFrame | None = None) -> ReplayInputs:
    axis = _axis()
    if prices is None:
        prices = pd.DataFrame(100.0, index=axis, columns=_TICKERS)
    return ReplayInputs(
        run_date="2026-10-27",
        dates=[d.strftime("%Y-%m-%d") for d in axis[:60]],
        signals_by_date={},
        predictions_by_date={},
        pillar_profiles_by_date={},
        price_matrix=prices,
        spy_prices=pd.Series(100.0, index=axis),
        bucket="test-bucket",
        fees=0.0,
        slippage_bps=0.0,
        init_cash=1_000_000.0,
        trades_db_path=trades_db_path,
        source_paths=[],
    )


def _sells(arm, ticker: str) -> list[tuple[str, float | None]]:
    return [
        (o["date"], o.get("shares"))
        for o in arm.orders
        if o["ticker"] == ticker and o["action"] in ("EXIT", "REDUCE")
    ]


def _holdings(orders: list[dict], prices: pd.DataFrame) -> pd.DataFrame:
    pf = orders_to_portfolio(
        orders, prices, init_cash=1_000_000.0, fees=0.0,
        partial_exits=has_partial_exits(orders),
    )
    assets = pf.assets()
    assets.columns = prices.columns
    return assets


# --------------------------------------------------------------------------
# the simulator's opt-in partial-exit path
# --------------------------------------------------------------------------


def test_partial_exit_sells_only_its_shares():
    axis = _axis(10)
    prices = pd.DataFrame(10.0, index=axis, columns=["AAA"])
    d = [x.strftime("%Y-%m-%d") for x in axis]
    orders = [
        {"date": d[0], "ticker": "AAA", "action": "ENTER", "shares": 100.0},
        {"date": d[2], "ticker": "AAA", "action": "REDUCE", "shares": 40.0},
        {"date": d[4], "ticker": "AAA", "action": "EXIT"},
    ]
    held = _holdings(orders, prices)["AAA"].tolist()
    assert held[:6] == [100.0, 100.0, 60.0, 60.0, 0.0, 0.0]


def test_default_path_still_closes_on_any_exit():
    axis = _axis(10)
    prices = pd.DataFrame(10.0, index=axis, columns=["AAA"])
    d = [x.strftime("%Y-%m-%d") for x in axis]
    orders = [
        {"date": d[0], "ticker": "AAA", "action": "ENTER", "shares": 100.0},
        {"date": d[2], "ticker": "AAA", "action": "REDUCE", "shares": 40.0},
    ]
    pf = orders_to_portfolio(orders, prices, init_cash=1_000_000.0, fees=0.0)
    assert pf.assets().iloc[:, 0].tolist()[:4] == [100.0, 100.0, 0.0, 0.0]


def test_partial_path_matches_default_without_partial_legs():
    axis = _axis(30)
    rng = np.random.default_rng(1)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(axis), 2)), axis=0)),
        index=axis, columns=["AAA", "BBB"],
    )
    d = [x.strftime("%Y-%m-%d") for x in axis]
    orders = [
        {"date": d[0], "ticker": "AAA", "action": "ENTER", "shares": 100.0},
        {"date": d[3], "ticker": "AAA", "action": "ENTER", "shares": 50.0},  # held: ignored
        {"date": d[1], "ticker": "BBB", "action": "ENTER", "shares": 70.0},
        {"date": d[8], "ticker": "AAA", "action": "EXIT"},
        {"date": d[9], "ticker": "BBB", "action": "REDUCE"},
        # Same-bar exit + re-entry: a signal conflict on either path.
        {"date": d[12], "ticker": "AAA", "action": "ENTER", "shares": 40.0},
        {"date": d[15], "ticker": "AAA", "action": "EXIT"},
        {"date": d[15], "ticker": "AAA", "action": "ENTER", "shares": 30.0},
        {"date": d[20], "ticker": "AAA", "action": "EXIT"},
    ]
    a = orders_to_portfolio(orders, prices, fees=0.001)
    b = orders_to_portfolio(orders, prices, fees=0.001, partial_exits=True)
    pd.testing.assert_series_equal(a.value(), b.value())


def test_has_partial_exits():
    assert not has_partial_exits([{"action": "ENTER", "shares": 1.0}, {"action": "EXIT"}])
    assert has_partial_exits([{"action": "REDUCE", "shares": 1.0}])


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def test_roundtrips_carry_their_reduce_legs(tmp_path):
    rt = er._load_roundtrips(_make_db(tmp_path)).set_index("trade_id")
    assert list(rt.loc["t0", "reduce_legs"]) == [
        ("2026-01-08", 40.0, "optimizer_scale_down"),
        ("2026-01-12", 20.0, "time_decay_reduce"),
    ]
    assert list(rt.loc["t1", "reduce_legs"]) == [("2026-01-09", 50.0, "intraday_profit_take")]
    # A leg dated after the terminal exit is outside the replay.
    assert list(rt.loc["t2", "reduce_legs"]) == []
    # The open roundtrip has no terminal exit and is not a roundtrip.
    assert "t3" not in rt.index


def test_reduce_leg_census_counts_modelled_and_unmodelled(tmp_path):
    census = er.reduce_leg_census(_make_db(tmp_path))
    assert census["modelled"] == {
        "reduce_optimizer": 1, "reduce_time": 1, "reduce_profit_take": 1, "reduce_other": 0,
    }
    assert census["unmodelled"] == {
        "reduce_optimizer": 2, "reduce_time": 0, "reduce_profit_take": 0, "reduce_other": 0,
    }


def test_db_without_reason_column_reads_reduces_as_other(tmp_path):
    db = _make_db(tmp_path, with_reason=False)
    assert er.reduce_leg_census(db)["modelled"]["reduce_other"] == 3


# --------------------------------------------------------------------------
# baseline = the as-executed book
# --------------------------------------------------------------------------


def test_baseline_emits_each_reduce_then_closes(tmp_path):
    arms = er.build_exit_family_arms(_inputs(_make_db(tmp_path)), "reduce_optimizer")
    assert _sells(arms.baseline, "AAA") == [
        ("2026-01-08", 40.0), ("2026-01-12", 20.0), ("2026-01-20", None),
    ]
    assert _sells(arms.baseline, "CCC") == [("2026-01-14", None)]


def test_baseline_reproduces_the_executed_share_path(tmp_path):
    inputs = _inputs(_make_db(tmp_path))
    arms = er.build_exit_family_arms(inputs, "reduce_optimizer")
    held = _holdings(arms.baseline.orders, inputs.price_matrix)
    path = held["AAA"]
    assert path[pd.Timestamp("2026-01-07")] == 100.0
    assert path[pd.Timestamp("2026-01-08")] == 60.0
    assert path[pd.Timestamp("2026-01-12")] == 40.0
    assert path[pd.Timestamp("2026-01-20")] == 0.0
    assert held["BBB"][pd.Timestamp("2026-01-09")] == 150.0
    assert held["BBB"][pd.Timestamp("2026-01-15")] == 0.0


def test_graded_exit_rules_ablated_arm_has_no_reduce_leg(tmp_path):
    arms = er._exit_rules_build_arms(_inputs(_make_db(tmp_path)))
    assert not has_partial_exits(arms.ablated.orders)
    assert has_partial_exits(arms.baseline.orders)


def test_position_sizing_scales_reduce_legs(tmp_path):
    arms = er._position_sizing_build_arms(_inputs(_make_db(tmp_path)))
    enter = next(o for o in arms.ablated.orders if o["ticker"] == "AAA" and o["action"] == "ENTER")
    sells = _sells(arms.ablated, "AAA")
    assert sells[0] == ("2026-01-08", float(int(40.0 * enter["shares"] / 100.0)))
    assert sells[-1] == ("2026-01-20", None)


# --------------------------------------------------------------------------
# reduce_* families
# --------------------------------------------------------------------------


def test_reduce_family_holds_only_its_legs(tmp_path):
    inputs = _inputs(_make_db(tmp_path))
    arms = er.build_exit_family_arms(inputs, "reduce_optimizer", hold_days=30)
    axis = inputs.price_matrix.index
    held_to = axis[axis.get_loc(pd.Timestamp("2026-01-05")) + 30].strftime("%Y-%m-%d")
    # The 40 optimizer_scale_down shares now leave at the hold, after the
    # terminal exit: the terminal sells only the 40 executed-remainder shares.
    assert _sells(arms.ablated, "AAA") == [
        ("2026-01-12", 20.0), ("2026-01-20", 40.0), (held_to, None),
    ]
    held = _holdings(arms.ablated.orders, inputs.price_matrix)["AAA"]
    assert held[pd.Timestamp("2026-01-08")] == 100.0
    assert held[pd.Timestamp("2026-01-20")] == 40.0
    assert held[pd.Timestamp(held_to)] == 0.0
    # Every other ticker is as executed.
    for t in ("BBB", "CCC"):
        assert _sells(arms.ablated, t) == _sells(arms.baseline, t)


def test_reduce_family_leaves_entries_identical(tmp_path):
    arms = er.build_exit_family_arms(_inputs(_make_db(tmp_path)), "reduce_profit_take")
    enters = lambda arm: [o for o in arm.orders if o["action"] == "ENTER"]  # noqa: E731
    assert enters(arms.baseline) == enters(arms.ablated)


def test_terminal_family_keeps_reduce_legs_as_executed(tmp_path):
    inputs = _inputs(_make_db(tmp_path))
    arms = er.build_exit_family_arms(inputs, "optimizer", hold_days=21)
    axis = inputs.price_matrix.index
    held_to = axis[axis.get_loc(pd.Timestamp("2026-01-05")) + 21].strftime("%Y-%m-%d")
    assert _sells(arms.ablated, "AAA") == [
        ("2026-01-08", 40.0), ("2026-01-12", 20.0), (held_to, None),
    ]


def test_reduce_family_with_no_leg_is_na(tmp_path):
    result = er.build_exit_family_arms(_inputs(_make_db(tmp_path)), "reduce_other")
    assert isinstance(result, NotAvailable)
    assert er.ISSUE_12683 in result.reason


@pytest.mark.parametrize("reason,family", [
    ("optimizer_scale_down", "reduce_optimizer"),
    ("time_decay_reduce", "reduce_time"),
    ("intraday_profit_take", "reduce_profit_take"),
    ("a_reduce_added_next_month", "reduce_other"),
    (None, "reduce_other"),
])
def test_reduce_family_of(reason, family):
    assert er.reduce_family_of(reason) == family


def test_decomposition_carries_reduce_rows(tmp_path, monkeypatch):
    axis = _axis()
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(axis), 4)), axis=0)),
        index=axis, columns=_TICKERS,
    )
    monkeypatch.setattr(report, "_account_trials", lambda inputs, n: 7)
    body = report.build_contribution_lift_report(
        _inputs(_make_db(tmp_path), prices=prices), [er.SPEC_EXIT_RULES], include_shadow=True,
    )
    block = body["shadow"]["exit_rules_decomposition"]
    rows = {r["name"]: r for r in block["families"]}
    for fam in ("reduce_optimizer", "reduce_time", "reduce_profit_take"):
        row = rows[f"exit_rules.{fam}.hold30"]
        assert row["leg_kind"] == "reduce"
        assert row["status"] in {"ok", "N/A-LOW-N"} and row["count_matched"] is True
    assert rows["exit_rules.reduce_other.hold30"]["status"] == "N/A-MISSING-INPUT"
    assert block["census_reduce_legs"]["modelled"]["reduce_optimizer"] == 1
    json.dumps(body, allow_nan=False)
