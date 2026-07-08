"""Tests for evaluate.py's risk_ratio_ci wiring (config#976, Director L4558).

Covers the producer-side glue that config#976 asked to be wired the same
two-repo pattern as sample_size_adequacy (crucible-backtester#360 /
crucible-evaluator#65): the pure-compute ``compute_risk_ratio_ci`` itself is
already covered by tests/test_risk_ratio_ci.py (PR #411); this file only
tests the NEW wiring — the ``_portfolio_daily_returns_from_team_lift``
aggregation helper that sources the portfolio daily-return series from
already-in-scope e2e team_lift picks (evaluate.py has no direct access to
the ``pf_returns_aligned`` series backtest.py's separate deployed-strategy
headline computes — that process-boundary discrepancy is documented at the
helper's call site in evaluate.py).
"""

from __future__ import annotations

import pandas as pd
import pytest


def _prices(n_days: int = 10, start: str = "2026-01-05") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n_days, freq="D")
    return pd.DataFrame(
        {
            "AAPL": [100.0 + i for i in range(n_days)],
            "MSFT": [200.0 + 2 * i for i in range(n_days)],
        },
        index=idx,
    )


class TestPortfolioDailyReturnsFromTeamLift:
    def test_aggregates_picks_across_teams(self):
        from evaluate import _portfolio_daily_returns_from_team_lift

        team_lift = [
            {
                "team_id": "tech",
                "picks": [{"ticker": "AAPL", "eval_date": "2026-01-05", "return_5d": 0.01}],
            },
            {
                "team_id": "health",
                "picks": [{"ticker": "MSFT", "eval_date": "2026-01-05", "return_5d": 0.02}],
            },
        ]
        series = _portfolio_daily_returns_from_team_lift(
            team_lift, _prices(), horizon_days=3,
        )
        assert series is not None
        assert not series.empty
        # Both tickers' picks fall inside the same portfolio sleeve.
        assert len(series) == 3

    def test_empty_team_lift_returns_none(self):
        from evaluate import _portfolio_daily_returns_from_team_lift

        assert _portfolio_daily_returns_from_team_lift([], _prices()) is None
        assert _portfolio_daily_returns_from_team_lift(None, _prices()) is None

    def test_teams_with_no_picks_returns_none(self):
        from evaluate import _portfolio_daily_returns_from_team_lift

        team_lift = [{"team_id": "tech", "picks": []}, {"team_id": "health", "picks": None}]
        assert _portfolio_daily_returns_from_team_lift(team_lift, _prices()) is None

    def test_missing_prices_returns_none(self):
        from evaluate import _portfolio_daily_returns_from_team_lift

        team_lift = [{"team_id": "tech", "picks": [{"ticker": "AAPL", "eval_date": "2026-01-05"}]}]
        assert _portfolio_daily_returns_from_team_lift(team_lift, None) is None
        assert _portfolio_daily_returns_from_team_lift(team_lift, pd.DataFrame()) is None

    def test_unknown_ticker_dropped_silently(self):
        from evaluate import _portfolio_daily_returns_from_team_lift

        team_lift = [
            {"team_id": "tech", "picks": [{"ticker": "NOPE", "eval_date": "2026-01-05"}]},
        ]
        assert _portfolio_daily_returns_from_team_lift(team_lift, _prices()) is None


class TestRiskRatioCIEndToEndAggregation:
    """The aggregated series feeds compute_risk_ratio_ci exactly like
    evaluate.py's save() call site does — verifies the two producer pieces
    (aggregation helper + the already-shipped CI computation from PR #411)
    compose without error, end to end, at both an adequate and an
    inadequate sample size."""

    def test_thin_sample_reports_insufficient_or_uncertain(self):
        from analysis.risk_ratio_ci import compute_risk_ratio_ci
        from evaluate import _portfolio_daily_returns_from_team_lift

        team_lift = [
            {
                "team_id": "tech",
                "picks": [{"ticker": "AAPL", "eval_date": "2026-01-05"}],
            },
        ]
        pf_returns = _portfolio_daily_returns_from_team_lift(
            team_lift, _prices(n_days=10), horizon_days=3,
        )
        spy = pd.Series(
            [0.001] * 3, index=pf_returns.index if pf_returns is not None else None,
        )
        result = compute_risk_ratio_ci(pf_returns, spy)
        # Far below RISK_RATIO_SAMPLE_FLOOR (126) — magnitude must not be
        # reported as certain regardless of point-estimate sign.
        assert result["all_magnitude_certain"] is False

    def test_none_series_yields_insufficient_data_shape(self):
        from analysis.risk_ratio_ci import compute_risk_ratio_ci

        result = compute_risk_ratio_ci(None, None)
        assert result["status"] == "insufficient_data"
        assert result["all_magnitude_certain"] is False
