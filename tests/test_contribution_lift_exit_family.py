"""tests/test_contribution_lift_exit_family.py — alpha-engine-config-I12624.

The per-family ``exit_rules`` decomposition is a SHADOW diagnostic: one family
of terminal exits is replaced by a fixed hold at a time, every other order is
as executed. These tests pin the three properties that make it shadow-only and
correct:

* only the ablated family's EXIT dates move — entries, shares and every other
  exit are byte-identical to the baseline (so each cycle count-matches);
* the arms are NOT registered components (the evaluator grades every entry in
  ``registry.SPECS``), and the report writes them under a top-level ``shadow``
  key the tile does not read;
* a shadow failure is recorded in-band and never costs the live components
  their artifact.
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
from analysis.contribution_lift.harness import NotAvailable, ReplayInputs  # noqa: E402
from analysis.contribution_lift.registry import SPECS  # noqa: E402


def _axis(n: int = 200) -> pd.DatetimeIndex:
    return pd.bdate_range("2026-01-01", periods=n)


_ROUNDTRIPS = [
    # (entry date, ticker, shares, exit date, terminal exit_reason)
    ("2026-01-05", "AAA", 100.0, "2026-01-08", "optimizer_target_zero"),
    ("2026-01-05", "BBB", 200.0, "2026-01-15", "time_decay_exit"),
    ("2026-01-12", "CCC", 50.0, "2026-01-14", "intraday_trailing_stop"),
    ("2026-01-12", "AAA", 70.0, "2026-01-22", "some_new_reason"),
]


def _make_db(tmp_path: Path, roundtrips=_ROUNDTRIPS, *, with_reason: bool = True) -> str:
    db_path = tmp_path / "trades.db"
    conn = sqlite3.connect(str(db_path))
    reason_col = ", exit_reason TEXT" if with_reason else ""
    conn.execute(
        "CREATE TABLE trades (trade_id TEXT, date TEXT, ticker TEXT, action TEXT, "
        "shares REAL, fill_price REAL, signal_date TEXT, entry_trade_id TEXT, "
        f"days_held REAL{reason_col})"
    )
    for i, (d, t, sh, xd, why) in enumerate(roundtrips):
        conn.execute(
            "INSERT INTO trades (trade_id, date, ticker, action, shares, fill_price, "
            "signal_date, entry_trade_id, days_held) VALUES (?, ?, ?, 'ENTER', ?, 100.0, ?, NULL, NULL)",
            (f"t{i}", d, t, sh, d),
        )
        if with_reason:
            conn.execute(
                "INSERT INTO trades (trade_id, date, ticker, action, entry_trade_id, exit_reason) "
                "VALUES (?, ?, ?, 'EXIT', ?, ?)",
                (f"t{i}-x", xd, t, f"t{i}", why),
            )
        else:
            conn.execute(
                "INSERT INTO trades (trade_id, date, ticker, action, entry_trade_id) "
                "VALUES (?, ?, ?, 'EXIT', ?)",
                (f"t{i}-x", xd, t, f"t{i}"),
            )
    conn.commit()
    conn.close()
    return str(db_path)


def _inputs(trades_db_path: str | None, *, prices: pd.DataFrame | None = None) -> ReplayInputs:
    axis = _axis()
    if prices is None:
        prices = pd.DataFrame(100.0, index=axis, columns=["AAA", "BBB", "CCC"])
    return ReplayInputs(
        run_date="2026-10-23",
        dates=[d.strftime("%Y-%m-%d") for d in axis[:60]],
        signals_by_date={},
        predictions_by_date={},
        pillar_profiles_by_date={},
        price_matrix=prices,
        spy_prices=pd.Series(100.0, index=axis),
        bucket="test-bucket",
        fees=0.001,
        slippage_bps=0.0,
        init_cash=1_000_000.0,
        trades_db_path=trades_db_path,
        source_paths=[],
    )


def _exits(arm) -> list[tuple[str, str]]:
    """Sorted ``(ticker, exit_date)`` of every EXIT order (arms may reorder)."""
    return sorted((o["ticker"], o["date"]) for o in arm.orders if o["action"] == "EXIT")


# --------------------------------------------------------------------------
# family taxonomy
# --------------------------------------------------------------------------


def test_every_mapped_family_is_an_emitted_family():
    assert set(er.EXIT_REASON_FAMILY.values()) <= set(er.EXIT_FAMILIES)
    assert "other" in er.EXIT_FAMILIES


@pytest.mark.parametrize("reason,family", [
    ("optimizer_target_zero", "optimizer"),
    ("time_decay_exit", "time"),
    ("atr_trailing_stop", "stop"),
    ("position_loss_floor", "stop"),
    ("research_signal", "signal"),
    ("momentum_exit", "momentum"),
    ("a_reason_added_next_month", "other"),
    (None, "other"),
    ("", "other"),
])
def test_exit_family_of(reason, family):
    assert er.exit_family_of(reason) == family


def test_census_counts_terminal_exits_per_family(tmp_path):
    census = er.exit_family_census(_make_db(tmp_path))
    assert census == {"optimizer": 1, "time": 1, "stop": 1, "signal": 0, "momentum": 0, "other": 1}


# --------------------------------------------------------------------------
# arm construction
# --------------------------------------------------------------------------


def test_only_the_ablated_family_exit_moves(tmp_path):
    inputs = _inputs(_make_db(tmp_path))
    arms = er.build_exit_family_arms(inputs, "optimizer", hold_days=30)
    axis = inputs.price_matrix.index
    held_to = axis[axis.get_loc(pd.Timestamp("2026-01-05")) + 30].strftime("%Y-%m-%d")

    assert _exits(arms.baseline) == sorted([
        ("AAA", "2026-01-08"), ("BBB", "2026-01-15"), ("CCC", "2026-01-14"), ("AAA", "2026-01-22"),
    ])
    assert _exits(arms.ablated) == sorted([
        ("AAA", held_to), ("BBB", "2026-01-15"), ("CCC", "2026-01-14"), ("AAA", "2026-01-22"),
    ])


def test_entries_and_shares_are_identical_in_both_arms(tmp_path):
    arms = er.build_exit_family_arms(_inputs(_make_db(tmp_path)), "stop", hold_days=21)
    enters = lambda arm: [o for o in arm.orders if o["action"] == "ENTER"]  # noqa: E731
    assert enters(arms.baseline) == enters(arms.ablated)


def test_unknown_reason_is_ablated_as_other(tmp_path):
    arms = er.build_exit_family_arms(_inputs(_make_db(tmp_path)), "other", hold_days=21)
    moved = set(_exits(arms.baseline)) - set(_exits(arms.ablated))
    assert moved == {("AAA", "2026-01-22")}


def test_family_with_no_roundtrip_is_na(tmp_path):
    result = er.build_exit_family_arms(_inputs(_make_db(tmp_path)), "signal")
    assert isinstance(result, NotAvailable)
    assert result.status == "N/A-MISSING-INPUT"


def test_missing_db_is_na():
    result = er.build_exit_family_arms(_inputs(None), "optimizer")
    assert isinstance(result, NotAvailable)
    assert "trades_db_path is None" in result.reason


def test_db_without_exit_reason_column_puts_everything_in_other(tmp_path):
    db = _make_db(tmp_path, with_reason=False)
    assert isinstance(er.build_exit_family_arms(_inputs(db), "optimizer"), NotAvailable)
    assert er.exit_family_census(db)["other"] == len(_ROUNDTRIPS)


def test_unknown_family_raises(tmp_path):
    with pytest.raises(ValueError):
        er.build_exit_family_arms(_inputs(_make_db(tmp_path)), "target")


# --------------------------------------------------------------------------
# shadow-only: never a graded component
# --------------------------------------------------------------------------


def test_family_specs_are_not_registered_components():
    names = {s.name for s in SPECS}
    assert not any(n.startswith("exit_rules.") for n in names)
    assert "exit_rules" in names


def _report_with_stub_specs(inputs, monkeypatch, *, include_shadow):
    monkeypatch.setattr(report, "_account_trials", lambda inputs, n: 7)
    return report.build_contribution_lift_report(
        inputs, [er.SPEC_EXIT_RULES], include_shadow=include_shadow
    )


def test_report_writes_decomposition_under_shadow_not_components(tmp_path, monkeypatch):
    axis = _axis()
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(axis), 3)), axis=0)),
        index=axis, columns=["AAA", "BBB", "CCC"],
    )
    body = _report_with_stub_specs(
        _inputs(_make_db(tmp_path), prices=prices), monkeypatch, include_shadow=True
    )

    assert [c["name"] for c in body["components"]] == ["exit_rules"]
    block = body["shadow"]["exit_rules_decomposition"]
    assert block["mode"] == "shadow" and block["graded"] is False
    assert block["shadow_arm"]["name"] == "exit_rules.optimizer.hold30"
    names = {r["name"] for r in block["families"]}
    assert names == {
        f"exit_rules.{f}.hold{h}" for f in er.ALL_EXIT_FAMILIES for h in er.EXIT_FAMILY_HOLD_DAYS
    }
    assert block["schema_version"] == 2
    assert {r["leg_kind"] for r in block["families"] if r["family"].startswith("reduce_")} == {"reduce"}
    opt = block["shadow_arm"]["row"]
    assert opt["status"] in {"ok", "N/A-LOW-N"} and opt["count_matched"] is True
    json.dumps(body, allow_nan=False)


def test_report_without_shadow_has_no_shadow_key(tmp_path, monkeypatch):
    body = _report_with_stub_specs(_inputs(_make_db(tmp_path)), monkeypatch, include_shadow=False)
    assert "shadow" not in body


def test_explicit_specs_default_to_no_shadow(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "_account_trials", lambda inputs, n: 7)
    body = report.build_contribution_lift_report(_inputs(_make_db(tmp_path)), [er.SPEC_EXIT_RULES])
    assert "shadow" not in body


def test_shadow_failure_is_in_band_and_keeps_components(tmp_path, monkeypatch):
    def boom(inputs, *, n_trials):
        raise RuntimeError("simulated shadow failure")

    monkeypatch.setattr(er, "exit_family_decomposition", boom)
    body = _report_with_stub_specs(_inputs(_make_db(tmp_path)), monkeypatch, include_shadow=True)

    assert [c["name"] for c in body["components"]] == ["exit_rules"]
    block = body["shadow"]["exit_rules_decomposition"]
    assert block["status"] == "error"
    assert "simulated shadow failure" in block["reason"]


def test_window_summary_splits_on_cut_dates():
    diffs = {f"2026-0{m}-01": float(m) for m in range(1, 10)}
    w = er._window_summary(diffs, "2026-05-01", None)
    assert w["n_samples"] == 5 and w["status"] == "N/A-LOW-N"
    diffs = {d.strftime("%Y-%m-%d"): -0.01 for d in _axis(40)}
    w = er._window_summary(diffs, None, None)
    assert w["status"] == "ok" and w["value"] == pytest.approx(-0.01)
