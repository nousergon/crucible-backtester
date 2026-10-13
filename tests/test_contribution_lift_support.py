"""tests/test_contribution_lift_support.py — paired-support census (I12243).

alpha-engine-config-I12243 (parent I11988). ``objective.check_count_match``
compared only the INTERSECTION of the two arms' date keys, so a date one arm
traded and the other did not vanished from the assurance:
``baseline={2026-08-03: 2, 2026-08-10: 2}`` vs ``ablated={2026-08-03: 2}``
returned ``matched=True`` with an empty reason. These fixtures pin the union
census and the additive ``support`` diagnostics block the harness now emits:

* same width but a one-sided missing date      -> gap, named
* absent / invalid price                        -> ``unpriceable``, named
* generated but unfilled ENTER (real vectorbt)  -> ``unfilled_order_dates``
* wholly disjoint dates                         -> gap, census counts
* a shared valid cohort                         -> matched, empty census
* the intentionally wider risk_guard arm        -> stays a gap
* a sampled window with no component entry      -> counted, named
* overlapping 21-row windows                    -> inference UNASSURED

Diagnostics only: none of these change ``value``/CI for a matched replay.
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

from analysis.contribution_lift import harness, objective  # noqa: E402
from analysis.contribution_lift.groups import executor_rules  # noqa: E402
from analysis.contribution_lift.harness import (  # noqa: E402
    HORIZON_DAYS,
    ArmSet,
    ReplayInputs,
    ReplaySpec,
    orders_arm,
    picks_arm,
)

H = HORIZON_DAYS


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _axis(n: int = 200) -> pd.DatetimeIndex:
    return pd.bdate_range("2026-01-01", periods=n)


def _inputs(
    *,
    axis: pd.DatetimeIndex | None = None,
    dates: list[str] | None = None,
    prices: pd.DataFrame | None = None,
    trades_db_path: str | None = None,
) -> ReplayInputs:
    axis = axis if axis is not None else _axis()
    if prices is None:
        prices = pd.DataFrame(100.0, index=axis, columns=["AAA", "BBB", "CCC"])
    if dates is None:
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


class _FakeSim:
    """label -> canned log-return series; reports no fills (like any fake)."""

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


def _spec(build_arms, name: str = "meta_l2_ic") -> ReplaySpec:
    return ReplaySpec(
        name=name,
        module="predictor",
        criticality="critical",
        pattern="substitution",
        issue="alpha-engine-config-I12243",
        build_arms=build_arms,
    )


def _fake(axis, base_label="baseline", abl_label="ablated"):
    return _FakeSim({
        base_label: pd.Series(0.002, index=axis),
        abl_label: pd.Series(0.001, index=axis),
    })


# --------------------------------------------------------------------------
# check_count_match — the union census
# --------------------------------------------------------------------------


def test_one_sided_missing_date_is_a_named_gap_not_a_match():
    """The issue's own read-only reproduction, verbatim."""
    result = objective.check_count_match(
        {"2026-08-03": 2, "2026-08-10": 2}, {"2026-08-03": 2}
    )

    assert result.matched is False
    assert result.reason, "a one-sided date must name its gap"
    assert "1 baseline-only date(s) (2026-08-10)" in result.reason
    assert result.baseline_only_dates == ("2026-08-10",)
    assert result.matched_dates == ("2026-08-03",)


def test_ablated_only_date_is_a_named_gap():
    result = objective.check_count_match(
        {"2026-08-03": 2}, {"2026-08-03": 2, "2026-08-17": 3}
    )

    assert result.matched is False
    assert "1 ablated-only date(s) (2026-08-17)" in result.reason
    assert result.census()["n_ablated_only"] == 1


def test_shared_valid_cohort_matches_with_an_empty_census():
    widths = {"2026-08-03": 2, "2026-08-10": 4}

    result = objective.check_count_match(widths, dict(widths))

    assert result.matched is True
    assert result.reason == ""
    census = result.census()
    assert census["n_matched"] == 2
    assert census["n_union_dates"] == 2
    for key in ("n_mismatched", "n_baseline_only", "n_ablated_only", "n_unpriceable"):
        assert census[key] == 0


def test_wholly_disjoint_dates_report_both_supports():
    result = objective.check_count_match(
        {"2026-08-03": 2, "2026-08-10": 2}, {"2026-09-01": 2}
    )

    assert result.matched is False
    assert "no cycle is present in both arms" in result.reason
    census = result.census()
    assert census["baseline_only_dates"] == ["2026-08-03", "2026-08-10"]
    assert census["ablated_only_dates"] == ["2026-09-01"]


def test_unpriceable_one_sided_date_is_named_not_a_gap():
    result = objective.check_count_match(
        {"2026-08-03": 2, "2026-08-10": 2},
        {"2026-08-03": 2},
        unpriceable={"2026-08-10"},
    )

    assert result.matched is True
    assert result.unpriceable_dates == ("2026-08-10",)
    assert result.census()["n_unpriceable"] == 1


def test_zero_width_is_no_support():
    """Listing a date with no names trades exactly like omitting it."""
    result = objective.check_count_match(
        {"2026-08-03": 2, "2026-08-10": 0}, {"2026-08-03": 2}
    )

    assert result.matched is True
    assert result.census()["n_union_dates"] == 1


def test_width_mismatch_and_one_sided_support_are_both_named():
    result = objective.check_count_match(
        {"2026-08-03": 2, "2026-08-10": 2},
        {"2026-08-03": 3, "2026-08-17": 2},
    )

    assert result.matched is False
    assert "width mismatch on 1/1 shared cycles" in result.reason
    assert "baseline-only" in result.reason and "ablated-only" in result.reason


# --------------------------------------------------------------------------
# run_spec — end to end through the harness
# --------------------------------------------------------------------------


def test_run_spec_reports_a_gap_for_one_sided_support(monkeypatch):
    axis = _axis()
    inputs = _inputs(axis=axis)
    monkeypatch.setattr(harness, "simulate_arm", _fake(axis))

    def build(i):
        return ArmSet(
            baseline=picks_arm(
                "baseline", [{"date": d, "picks": ["AAA", "BBB"]} for d in i.dates]
            ),
            # The ablated arm silently lacks the last cycle at the SAME width.
            ablated=picks_arm(
                "ablated", [{"date": d, "picks": ["AAA", "CCC"]} for d in i.dates[:-1]]
            ),
        )

    component = harness.run_spec(_spec(build), inputs, n_trials=1)

    assert component["status"] == "gap"
    assert component["count_matched"] is False
    assert component["value"] is None
    assert "baseline-only" in component["status_reason"]
    census = component["support"]["census"]
    assert census["baseline_only_dates"] == [inputs.dates[-1]]
    assert census["n_matched"] == len(inputs.dates) - 1


def test_run_spec_shared_cohort_keeps_its_value_and_carries_support(monkeypatch):
    axis = _axis()
    inputs = _inputs(axis=axis)
    monkeypatch.setattr(harness, "simulate_arm", _fake(axis))

    def build(i):
        return ArmSet(
            baseline=picks_arm("baseline", [{"date": d, "picks": ["AAA"]} for d in i.dates]),
            ablated=picks_arm("ablated", [{"date": d, "picks": ["BBB"]} for d in i.dates]),
        )

    component = harness.run_spec(_spec(build), inputs, n_trials=1)

    assert component["status"] == "ok"
    assert component["value"] == pytest.approx(H * 0.001, abs=1e-9)
    support = component["support"]
    assert support["schema_version"] == harness.SUPPORT_SCHEMA_VERSION
    assert support["estimand"]["per_entry_cohort"] is False
    assert support["n_eligible_dates"] == len(inputs.dates)
    assert support["n_sampled_windows"] == component["n_samples"]
    assert support["census"]["n_matched"] == len(inputs.dates)
    assert support["effective_width_mismatch_dates"] == []
    for arm in support["arms"].values():
        assert arm["requested_picks"] == len(inputs.dates)
        assert arm["generated_enters"] == len(inputs.dates)
        assert arm["priceable_enters"] == len(inputs.dates)
        # A fake simulator observed nothing: explicit null, never zero.
        assert arm["fills_observed"] is False
        assert arm["observed_buy_fills"] is None
    assert "point_in_time_coefficient_provenance" in support["not_covered"]


def test_absent_price_is_named_on_the_arm_and_as_effective_mismatch(monkeypatch):
    """Same declared width, but one arm's name cannot be priced on one date.

    This is what meta_l2_ic's 74-vs-70 generated orders looked like on the
    2026-10-09 artifact while its declared widths matched: the status stays
    the count-match verdict on DECLARED support, and the shortfall is named.
    """
    axis = _axis()
    prices = pd.DataFrame(100.0, index=axis, columns=["AAA", "BBB", "CCC"])
    inputs = _inputs(axis=axis, prices=prices)
    bad = inputs.dates[5]
    prices.loc[pd.Timestamp(bad), "CCC"] = np.nan
    monkeypatch.setattr(harness, "simulate_arm", _fake(axis))

    def build(i):
        return ArmSet(
            baseline=picks_arm("baseline", [{"date": d, "picks": ["AAA"]} for d in i.dates]),
            ablated=picks_arm("ablated", [{"date": d, "picks": ["CCC"]} for d in i.dates]),
        )

    component = harness.run_spec(_spec(build), inputs, n_trials=1)

    support = component["support"]
    assert component["count_matched"] is True
    assert support["effective_width_mismatch_dates"] == [bad]
    assert support["arms"]["ablated"]["unpriced_declared_dates"] == [bad]
    assert support["arms"]["ablated"]["priceable_enters"] == len(inputs.dates) - 1
    assert support["arms"]["baseline"]["unpriced_declared_dates"] == []


def test_one_sided_date_that_no_arm_can_price_is_unpriceable_not_a_gap(monkeypatch):
    axis = _axis()
    inputs = _inputs(axis=axis)
    monkeypatch.setattr(harness, "simulate_arm", _fake(axis))
    beyond = "2027-06-01"  # after the last price row: nothing can trade it

    def build(i):
        return ArmSet(
            baseline=orders_arm("baseline", [
                {"date": i.dates[0], "ticker": "AAA", "action": "ENTER", "shares": 10.0},
                {"date": beyond, "ticker": "BBB", "action": "ENTER", "shares": 10.0},
            ]),
            ablated=orders_arm("ablated", [
                {"date": i.dates[0], "ticker": "CCC", "action": "ENTER", "shares": 10.0},
            ]),
        )

    component = harness.run_spec(_spec(build), inputs, n_trials=1)

    assert component["count_matched"] is True
    assert component["support"]["census"]["unpriceable_dates"] == [beyond]


def test_sampled_windows_without_a_component_entry_are_counted(monkeypatch):
    """The whole-book window is sampled on every eligible date, entry or not."""
    axis = _axis()
    inputs = _inputs(axis=axis)
    monkeypatch.setattr(harness, "simulate_arm", _fake(axis))
    entered = inputs.dates[::10]

    def build(i):
        return ArmSet(
            baseline=picks_arm("baseline", [{"date": d, "picks": ["AAA"]} for d in entered]),
            ablated=picks_arm("ablated", [{"date": d, "picks": ["BBB"]} for d in entered]),
        )

    component = harness.run_spec(_spec(build), inputs, n_trials=1)

    support = component["support"]
    assert component["n_samples"] == len(inputs.dates)
    assert support["n_sampled_windows_without_entry"] == len(inputs.dates) - len(entered)
    assert entered[0] not in support["sampled_windows_without_entry"]
    assert inputs.dates[1] in support["sampled_windows_without_entry"]


def test_weekly_windows_against_a_21_row_horizon_are_unassured(monkeypatch):
    axis = _axis()
    weekly = [d.strftime("%Y-%m-%d") for d in axis[:150:5]]
    inputs = _inputs(axis=axis, dates=weekly)
    monkeypatch.setattr(harness, "simulate_arm", _fake(axis))

    def build(i):
        return ArmSet(
            baseline=picks_arm("baseline", [{"date": d, "picks": ["AAA"]} for d in i.dates]),
            ablated=picks_arm("ablated", [{"date": d, "picks": ["BBB"]} for d in i.dates]),
        )

    component = harness.run_spec(_spec(build), inputs, n_trials=1)

    overlap = component["support"]["window_overlap"]
    n = component["n_samples"]
    assert overlap["n_windows"] == n
    assert overlap["n_adjacent_pairs"] == n - 1
    assert overlap["n_overlapping_adjacent_pairs"] == n - 1
    assert overlap["max_gap_trading_rows"] == 5
    assert overlap["mean_overlap_fraction"] == pytest.approx((H - 5) / H)
    # Greedy disjoint windows: one per ceil(21 / 5) = 5 sampled starts.
    assert overlap["n_non_overlapping_windows"] == -(-n // 5)
    assurance = component["support"]["inference_assurance"]
    assert assurance["status"] == "UNASSURED"
    assert assurance["overlap_aware"] is False
    # Diagnostics only: the estimate itself did not move.
    assert component["status"] == "ok"
    assert component["value"] == pytest.approx(H * 0.001, abs=1e-9)


def test_disjoint_windows_report_no_overlap():
    axis = _axis()
    dates = [d.strftime("%Y-%m-%d") for d in axis[:150:H]]

    overlap = objective.window_overlap(dates, axis, horizon_days=H)

    assert overlap["n_overlapping_adjacent_pairs"] == 0
    assert overlap["n_non_overlapping_windows"] == len(dates)
    assert overlap["mean_overlap_fraction"] == 0.0


def test_na_component_carries_a_null_support_block():
    component = harness.run_spec(
        _spec(lambda i: harness.NotAvailable(status="N/A-RETIRED", reason="retired")),
        _inputs(), n_trials=1,
    )

    assert component["support"] is None


# --------------------------------------------------------------------------
# Real simulator: the executed-fill witness
# --------------------------------------------------------------------------


def test_generated_but_unfilled_enter_is_named_through_real_vectorbt():
    """An ENTER for a name already held is generated and priceable, not filled."""
    axis = _axis()
    inputs = _inputs(axis=axis)
    d0, d1 = inputs.dates[0], inputs.dates[1]
    orders = [
        {"date": d0, "ticker": "AAA", "action": "ENTER", "shares": 10.0},
        {"date": d1, "ticker": "AAA", "action": "ENTER", "shares": 10.0},
        {"date": inputs.dates[30], "ticker": "AAA", "action": "EXIT"},
    ]
    arm = orders_arm("baseline", orders)
    arm_set = ArmSet(baseline=arm, ablated=arm)

    stats = harness.simulate_arm(arm, arm_set, inputs)
    census = harness.entry_census(arm, arm_set, inputs)
    support = harness._arm_support(census, stats)

    assert stats["buy_fills_by_date"] == {d0: 1}
    assert support["fills_observed"] is True
    assert support["priceable_enters"] == 2
    assert support["observed_buy_fills"] == 1
    assert support["unfilled_order_dates"] == [d1]


# --------------------------------------------------------------------------
# risk_guard: the intentionally wider arm stays a gap
# --------------------------------------------------------------------------


def _trades_db(tmp_path: Path) -> str:
    db = tmp_path / "trades.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE trades (trade_id TEXT, date TEXT, ticker TEXT, action TEXT, "
        "shares REAL, fill_price REAL, signal_date TEXT, entry_trade_id TEXT, "
        "days_held REAL)"
    )
    conn.execute(
        "CREATE TABLE executor_shadow_book (date TEXT, ticker TEXT, block_reason "
        "TEXT, intended_shares REAL, intended_dollars REAL, current_price REAL)"
    )
    conn.executemany(
        "INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("t0", "2026-01-05", "AAA", "ENTER", 100.0, 100.0, "2026-01-05", None, None),
            ("t0x", "2026-01-20", "AAA", "EXIT", None, None, None, "t0", 10),
        ],
    )
    conn.executemany(
        "INSERT INTO executor_shadow_book VALUES (?, ?, ?, ?, ?, ?)",
        [
            # Same-day sibling block: widens a shared cycle.
            ("2026-01-05", "BBB", "score below min", 100.0, None, 100.0),
            # Block on a day with NO executed entry: support only the
            # guard-off arm has — invisible to the old intersection check.
            ("2026-01-14", "CCC", "stance gate: DOWN veto", 100.0, None, 100.0),
        ],
    )
    conn.commit()
    conn.close()
    return str(db)


def test_risk_guard_stays_a_gap_and_names_its_guard_off_only_dates(tmp_path, monkeypatch):
    axis = _axis()
    inputs = _inputs(axis=axis, trades_db_path=_trades_db(tmp_path))
    arms = executor_rules._risk_guard_build_arms(inputs)
    monkeypatch.setattr(harness, "simulate_arm", _fake(
        axis, arms.baseline.label, arms.ablated.label
    ))

    component = harness.run_spec(executor_rules.SPEC_RISK_GUARD, inputs, n_trials=1)

    assert component["status"] == "gap"
    assert "width mismatch" in component["status_reason"]
    census = component["support"]["census"]
    assert census["mismatched_dates"] == ["2026-01-05"]
    assert census["ablated_only_dates"] == ["2026-01-14"]


# --------------------------------------------------------------------------
# Contract: the support block is strict JSON and versioned
# --------------------------------------------------------------------------


def test_support_block_is_strict_json(monkeypatch):
    axis = _axis()
    inputs = _inputs(axis=axis)
    monkeypatch.setattr(harness, "simulate_arm", _fake(axis))

    def build(i):
        return ArmSet(
            baseline=picks_arm("baseline", [{"date": d, "picks": ["AAA"]} for d in i.dates]),
            ablated=picks_arm("ablated", [{"date": d, "picks": ["BBB"]} for d in i.dates[1:]]),
        )

    component = harness.run_spec(_spec(build), inputs, n_trials=1)

    text = json.dumps(component["support"], allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
    assert set(component["support"]) == {
        "schema_version", "estimand", "n_eligible_dates", "n_sampled_windows",
        "census", "effective_width_mismatch_dates",
        "n_sampled_windows_without_entry", "sampled_windows_without_entry",
        "arms", "window_overlap", "inference_assurance", "not_covered",
    }
