"""Tests for analysis.team_daily_returns.

Pins:
  1. Single-pick → daily returns match (price[t]/price[t-1])-1 over the window.
  2. Two picks → equal-weight mean of per-pick returns when both held.
  3. Overlapping holding windows → mean over only the picks held on day t.
  4. Pick whose eval_date falls outside the price index is skipped.
  5. Pick whose ticker is absent from prices is silently dropped.
  6. horizon_days controls the number of daily-return contributions per pick.
  7. stack_team_returns_to_long round-trips the schema.
  8. Empty input → empty output (not an exception).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.team_daily_returns import (
    compute_team_daily_returns,
    stack_team_returns_to_long,
)


def _make_prices(n_days: int = 12) -> pd.DataFrame:
    """Synthetic price matrix: AAA up 1%/day, BBB flat, CCC down 0.5%/day."""
    dates = pd.date_range("2026-01-05", periods=n_days, freq="B")
    return pd.DataFrame(
        {
            "AAA": 100.0 * np.power(1.01, np.arange(n_days)),
            "BBB": np.full(n_days, 50.0),
            "CCC": 200.0 * np.power(0.995, np.arange(n_days)),
        },
        index=dates,
    )


class TestSinglePick:
    def test_returns_match_per_day_pct_change(self):
        prices = _make_prices(n_days=8)
        eval_date = prices.index[0]
        picks = pd.DataFrame([{
            "team_id": "tech", "ticker": "AAA", "eval_date": eval_date,
        }])
        out = compute_team_daily_returns(picks, prices, horizon_days=5)
        assert "tech" in out
        series = out["tech"]
        # 5 daily returns: AAA goes up 1%/day, so all returns ≈ 0.01.
        assert len(series) == 5
        np.testing.assert_allclose(series.to_numpy(), [0.01] * 5, rtol=1e-9)

    def test_horizon_days_controls_length(self):
        prices = _make_prices(n_days=15)
        eval_date = prices.index[0]
        picks = pd.DataFrame([{
            "team_id": "tech", "ticker": "AAA", "eval_date": eval_date,
        }])
        for h in (1, 3, 10):
            out = compute_team_daily_returns(picks, prices, horizon_days=h)
            assert len(out["tech"]) == h


class TestEqualWeightAggregation:
    def test_two_picks_same_eval_date(self):
        prices = _make_prices(n_days=8)
        eval_date = prices.index[0]
        picks = pd.DataFrame([
            {"team_id": "tech", "ticker": "AAA", "eval_date": eval_date},
            {"team_id": "tech", "ticker": "BBB", "eval_date": eval_date},
        ])
        out = compute_team_daily_returns(picks, prices, horizon_days=5)
        # AAA = +1%/day, BBB = 0%/day. Equal-weight mean = +0.5%/day.
        np.testing.assert_allclose(out["tech"].to_numpy(), [0.005] * 5, rtol=1e-9)

    def test_overlapping_holding_windows_remix_membership(self):
        # Pick AAA on day 0 (held days 1..3, horizon=3),
        # pick BBB on day 2 (held days 3..5, horizon=3).
        # Day 1: only AAA held → return = 0.01
        # Day 2: only AAA held → return = 0.01
        # Day 3: AAA + BBB held → mean(0.01, 0.0) = 0.005
        # Day 4: only BBB held → return = 0.0
        # Day 5: only BBB held → return = 0.0
        prices = _make_prices(n_days=10)
        picks = pd.DataFrame([
            {"team_id": "tech", "ticker": "AAA", "eval_date": prices.index[0]},
            {"team_id": "tech", "ticker": "BBB", "eval_date": prices.index[2]},
        ])
        out = compute_team_daily_returns(picks, prices, horizon_days=3)
        series = out["tech"]
        # 5 distinct trading days have at least one pick held.
        assert len(series) == 5
        np.testing.assert_allclose(
            series.to_numpy(),
            [0.01, 0.01, 0.005, 0.0, 0.0],
            atol=1e-9,
        )

    def test_per_team_isolation(self):
        prices = _make_prices(n_days=8)
        eval_date = prices.index[0]
        picks = pd.DataFrame([
            {"team_id": "tech", "ticker": "AAA", "eval_date": eval_date},
            {"team_id": "health", "ticker": "BBB", "eval_date": eval_date},
        ])
        out = compute_team_daily_returns(picks, prices, horizon_days=5)
        assert set(out.keys()) == {"tech", "health"}
        np.testing.assert_allclose(out["tech"].to_numpy(), [0.01] * 5, rtol=1e-9)
        np.testing.assert_allclose(out["health"].to_numpy(), [0.0] * 5, rtol=1e-9)


class TestEdgeCases:
    def test_eval_date_outside_price_index_is_skipped(self):
        prices = _make_prices(n_days=8)
        # Date that's before the price index.
        bad_date = prices.index[0] - pd.Timedelta(days=30)
        picks = pd.DataFrame([{
            "team_id": "tech", "ticker": "AAA", "eval_date": bad_date,
        }])
        out = compute_team_daily_returns(picks, prices, horizon_days=5)
        # No picks survived → no team in output.
        assert out == {}

    def test_ticker_absent_from_prices_is_dropped(self):
        prices = _make_prices(n_days=8)
        eval_date = prices.index[0]
        picks = pd.DataFrame([
            {"team_id": "tech", "ticker": "AAA", "eval_date": eval_date},
            {"team_id": "tech", "ticker": "ZZZ", "eval_date": eval_date},  # absent
        ])
        out = compute_team_daily_returns(picks, prices, horizon_days=5)
        # ZZZ silently dropped; AAA's series remains.
        np.testing.assert_allclose(out["tech"].to_numpy(), [0.01] * 5, rtol=1e-9)

    def test_empty_picks_returns_empty_dict(self):
        prices = _make_prices(n_days=8)
        empty = pd.DataFrame(columns=["team_id", "ticker", "eval_date"])
        assert compute_team_daily_returns(empty, prices, horizon_days=5) == {}

    def test_invalid_horizon_raises(self):
        prices = _make_prices(n_days=8)
        picks = pd.DataFrame([{
            "team_id": "tech", "ticker": "AAA", "eval_date": prices.index[0],
        }])
        with pytest.raises(ValueError):
            compute_team_daily_returns(picks, prices, horizon_days=0)
        with pytest.raises(ValueError):
            compute_team_daily_returns(picks, prices, horizon_days=-1)

    def test_missing_required_column_raises(self):
        prices = _make_prices(n_days=8)
        bad = pd.DataFrame([{"team_id": "tech", "eval_date": prices.index[0]}])
        with pytest.raises(ValueError, match="missing required columns"):
            compute_team_daily_returns(bad, prices, horizon_days=5)

    def test_non_datetime_index_raises(self):
        eval_date = pd.Timestamp("2026-01-05")
        picks = pd.DataFrame([{
            "team_id": "tech", "ticker": "AAA", "eval_date": eval_date,
        }])
        prices = pd.DataFrame({"AAA": [100, 101, 102]}, index=[0, 1, 2])
        with pytest.raises(TypeError, match="DatetimeIndex"):
            compute_team_daily_returns(picks, prices, horizon_days=2)


class TestConvictionWeights:
    def test_weight_column_used_when_present(self):
        # AAA weight 3, BBB weight 1 → weighted mean = (3*0.01 + 1*0.0)/4 = 0.0075.
        prices = _make_prices(n_days=8)
        eval_date = prices.index[0]
        picks = pd.DataFrame([
            {"team_id": "tech", "ticker": "AAA", "eval_date": eval_date, "weight": 3.0},
            {"team_id": "tech", "ticker": "BBB", "eval_date": eval_date, "weight": 1.0},
        ])
        out = compute_team_daily_returns(picks, prices, horizon_days=5)
        np.testing.assert_allclose(out["tech"].to_numpy(), [0.0075] * 5, rtol=1e-9)


class TestStackToLong:
    def test_round_trip(self):
        prices = _make_prices(n_days=8)
        eval_date = prices.index[0]
        picks = pd.DataFrame([
            {"team_id": "tech", "ticker": "AAA", "eval_date": eval_date},
            {"team_id": "health", "ticker": "BBB", "eval_date": eval_date},
        ])
        out = compute_team_daily_returns(picks, prices, horizon_days=5)
        long = stack_team_returns_to_long(out)
        assert list(long.columns) == ["team_id", "trading_day", "return"]
        # 5 rows × 2 teams = 10 rows.
        assert len(long) == 10
        assert set(long["team_id"].unique()) == {"tech", "health"}

    def test_empty_input_returns_empty_dataframe(self):
        long = stack_team_returns_to_long({})
        assert list(long.columns) == ["team_id", "trading_day", "return"]
        assert len(long) == 0
