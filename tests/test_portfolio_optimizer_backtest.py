"""
Unit + integration tests for analysis/portfolio_optimizer_backtest.py — PR 3
of the portfolio-optimizer-260511 arc.

Helper-level unit tests run with synthetic inputs and no external deps.
The end-to-end integration test imports the real solve_target_weights
kernel from the sibling alpha-engine repo; if the repo isn't checked out
at the expected dev location, that test is skipped (CI without the sibling
checkout reaches all the unit tests).
"""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from analysis.portfolio_optimizer_backtest import (
    _ABS_CVAR_95_FLOOR,
    _ABS_MAX_DRAWDOWN_FLOOR,
    OptimizerBacktestResult,
    _ensure_spy_column,
    _gate_thresholds,
    _select_rebalance_dates,
    compare_to_legacy,
    run_optimizer_backtest,
)


_ALPHA_ENGINE_PATH = os.path.expanduser("~/Development/alpha-engine")


def _trading_dates(start: str = "2024-01-02", n_days: int = 260) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n_days, freq="B")


def _synthetic_price_matrix(tickers: list[str], n_days: int = 260, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = _trading_dates(n_days=n_days)
    data = {}
    for i, t in enumerate(tickers):
        returns = rng.normal(0.0005, 0.012, n_days)
        data[t] = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame(data, index=dates)


def _synthetic_spy_series(n_days: int = 260, seed: int = 99) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = _trading_dates(n_days=n_days)
    returns = rng.normal(0.0004, 0.010, n_days)
    return pd.Series(100 * np.exp(np.cumsum(returns)), index=dates, name="SPY")


def _synthetic_predictions(
    tickers: list[str], dates: pd.DatetimeIndex, seed: int = 42,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, float]] = {}
    for d in dates:
        out[d.strftime("%Y-%m-%d")] = {
            t: float(rng.normal(0.02, 0.03)) for t in tickers
        }
    return out


# ── Unit tests — pure helpers ───────────────────────────────────────────────


class TestSelectRebalanceDates:
    def test_weekly_cadence_picks_every_fifth(self):
        dates = [f"2024-01-{i:02d}" for i in range(2, 30)]
        idx = pd.DatetimeIndex(pd.to_datetime(dates))
        picks = _select_rebalance_dates(dates, idx, freq_days=5)
        assert picks[0] == "2024-01-02"
        for i in range(1, len(picks)):
            prev = dates.index(picks[i - 1])
            curr = dates.index(picks[i])
            assert curr - prev >= 5, f"Cadence violated at {picks[i]}"

    def test_filters_dates_not_in_price_index(self):
        prediction_dates = ["2024-01-02", "2024-01-03", "2099-12-31"]
        idx = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03"]))
        picks = _select_rebalance_dates(prediction_dates, idx, freq_days=1)
        assert "2099-12-31" not in picks

    def test_empty_predictions_returns_empty(self):
        idx = pd.DatetimeIndex([])
        assert _select_rebalance_dates([], idx, freq_days=5) == []


class TestEnsureSpyColumn:
    def test_adds_spy_when_missing(self):
        pm = _synthetic_price_matrix(["AAPL", "MSFT"])
        spy = _synthetic_spy_series().reindex(pm.index)
        out = _ensure_spy_column(pm, spy)
        assert "SPY" in out.columns
        assert (out["SPY"].dropna() == spy.dropna()).all()

    def test_preserves_spy_when_already_present(self):
        pm = _synthetic_price_matrix(["AAPL", "SPY"])
        original_spy = pm["SPY"].copy()
        out = _ensure_spy_column(pm, _synthetic_spy_series())
        assert (out["SPY"] == original_spy).all()


class TestGateThresholds:
    def test_with_legacy_metrics_computes_skilled_risk_thresholds(self):
        legacy = {
            "sortino_ratio": 1.4,
            "psr": 0.97,
            "cvar_95": -0.02,
            "max_drawdown": -0.15,
            "turnover_one_way_ann": 2.0,
        }
        out = _gate_thresholds({}, legacy)
        assert out["sortino_min"] == pytest.approx(1.4 * 0.9), \
            "Sortino stays legacy-relative (primary risk-adjusted gate)"
        assert out["psr_min"] == pytest.approx(0.95), \
            "PSR confidence floor matches executor_optimizer's _MIN_PSR"
        # ROADMAP L124: risk floors are now ABSOLUTE, not legacy × 1.2.
        assert out["max_drawdown_floor"] == pytest.approx(_ABS_MAX_DRAWDOWN_FLOOR)
        assert out["cvar_95_floor"] == pytest.approx(_ABS_CVAR_95_FLOOR)
        assert out["turnover_max"] == pytest.approx(2.0 * 2.5), \
            "Turnover stays legacy-relative (behavior comparison, not a risk floor)"
        assert out["tracking_error_range"] == [0.02, 0.06]
        assert out["active_share_range"] == [0.08, 0.25]
        assert "sharpe_min" not in out, \
            "Raw Sharpe is not a gate — observability only per evaluator-revamp"
        assert "alpha_min" not in out, \
            "alpha vs SPY is presentation-only, not a gate"

    def test_without_legacy_keeps_absolute_risk_floors_and_psr(self):
        """ROADMAP L124: without a legacy baseline the absolute risk floors
        still apply (previously None → skipped → circular gate). Only the
        genuinely legacy-relative thresholds are None."""
        out = _gate_thresholds({"sortino_ratio": 1.0}, None)
        assert out["sortino_min"] is None, "legacy-relative → None without baseline"
        assert out["turnover_max"] is None, "legacy-relative → None without baseline"
        assert out["psr_min"] == pytest.approx(0.95), \
            "PSR floor is absolute (95% confidence), not legacy-relative"
        assert out["max_drawdown_floor"] == pytest.approx(_ABS_MAX_DRAWDOWN_FLOOR), \
            "max-drawdown floor is now absolute — applies with no legacy"
        assert out["cvar_95_floor"] == pytest.approx(_ABS_CVAR_95_FLOOR), \
            "CVaR(95) floor is now absolute — applies with no legacy"


class TestCompareToLegacy:
    def test_with_both_sides_emits_skilled_risk_deltas(self):
        opt = {"sortino_ratio": 1.5, "psr": 0.96, "cvar_95": -0.018,
               "max_drawdown": -0.12, "turnover_one_way_ann": 3.0,
               "total_alpha": 0.05, "sharpe_ratio": 1.1}
        leg = {"sortino_ratio": 1.4, "psr": 0.92, "cvar_95": -0.020,
               "max_drawdown": -0.15, "turnover_one_way_ann": 2.0,
               "total_alpha": 0.04, "sharpe_ratio": 1.0}
        out = compare_to_legacy(opt, leg)
        assert out["optimizer"] == opt
        assert out["legacy"] == leg
        d = out["deltas"]
        assert d["sortino_delta"] == pytest.approx(0.1)
        assert d["psr_delta"] == pytest.approx(0.04)
        assert d["cvar_95_delta"] == pytest.approx(0.002)
        assert d["max_drawdown_delta"] == pytest.approx(0.03)
        assert d["turnover_ratio"] == pytest.approx(1.5)
        assert d["alpha_delta_presentation"] == pytest.approx(0.01), \
            "alpha delta is preserved but explicitly labeled as presentation-only"
        assert "sharpe_delta" not in d, "Raw Sharpe delta is not a gate input"
        assert "gate_thresholds" in out

    def test_with_none_legacy_emits_null_section(self):
        opt = {"sortino_ratio": 1.0, "psr": 0.95}
        out = compare_to_legacy(opt, None)
        assert out["legacy"] is None
        assert out["deltas"] is None
        assert out["gate_thresholds"]["sortino_min"] is None
        assert out["gate_thresholds"]["psr_min"] == pytest.approx(0.95)


# ── Integration test — exercises the real kernel ────────────────────────────


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(_ALPHA_ENGINE_PATH, "executor")),
    reason="alpha-engine sibling checkout not present at ~/Development/alpha-engine",
)
class TestEndToEndIntegration:
    def test_run_optimizer_backtest_produces_metrics_and_weights(self):
        tickers = ["AAPL", "MSFT", "GOOG", "JNJ", "PG"]
        sector_map = {
            "AAPL": "Technology", "MSFT": "Technology", "GOOG": "Technology",
            "JNJ": "Healthcare",  "PG": "Consumer Staples",
        }
        price_matrix = _synthetic_price_matrix(tickers, n_days=300, seed=0)
        spy_prices = _synthetic_spy_series(n_days=300, seed=99).reindex(price_matrix.index)
        prediction_dates = price_matrix.index[260:295]
        predictions = _synthetic_predictions(tickers, prediction_dates, seed=42)

        result = run_optimizer_backtest(
            predictions_by_date=predictions,
            price_matrix=price_matrix,
            spy_prices=spy_prices,
            sector_map=sector_map,
            executor_path=_ALPHA_ENGINE_PATH,
            rebalance_freq_days=5,
            universe_cap=10,
        )

        assert isinstance(result, OptimizerBacktestResult)
        assert result.n_rebalances >= 1
        assert "SPY" in result.target_weights.columns
        for t in tickers:
            assert t in result.target_weights.columns

        valid_rows = result.target_weights.dropna(how="all")
        assert len(valid_rows) == result.n_rebalances
        for _, row in valid_rows.iterrows():
            assert row.sum() <= 1.0 + 1e-3, f"Weights exceed 100%: {row.sum()}"
            assert (row >= -1e-6).all(), "Negative weights present"

        m = result.metrics
        for key in ("sortino_ratio", "psr", "cvar_95",
                    "max_drawdown", "calmar_ratio",
                    "tracking_error_ann", "mean_active_share",
                    "mean_spy_weight", "turnover_one_way_ann",
                    "n_rebalances",
                    "sharpe_ratio", "total_return", "total_alpha"):
            assert key in m, f"Missing metric: {key}"
        assert m["n_rebalances"] == result.n_rebalances
        assert m["n_solver_failures"] >= 0
        assert m["mean_spy_weight"] is not None
        assert 0.0 < m["mean_spy_weight"] < 1.0, \
            f"SPY weight should be inside (0,1); got {m['mean_spy_weight']}"

    def test_universe_cap_is_enforced(self):
        tickers = [f"T{i:02d}" for i in range(50)]
        sector_map = {t: "Technology" for t in tickers}
        price_matrix = _synthetic_price_matrix(tickers, n_days=300, seed=1)
        spy_prices = _synthetic_spy_series(n_days=300, seed=99).reindex(price_matrix.index)
        prediction_dates = price_matrix.index[260:280]
        predictions = _synthetic_predictions(tickers, prediction_dates, seed=7)

        cap = 8
        result = run_optimizer_backtest(
            predictions_by_date=predictions,
            price_matrix=price_matrix,
            spy_prices=spy_prices,
            sector_map=sector_map,
            executor_path=_ALPHA_ENGINE_PATH,
            rebalance_freq_days=5,
            universe_cap=cap,
        )

        valid_rows = result.target_weights.dropna(how="all")
        for date, row in valid_rows.iterrows():
            non_spy = row.drop("SPY", errors="ignore").dropna()
            non_spy_nonzero = (non_spy.abs() > 1e-9).sum()
            assert non_spy_nonzero <= cap, (
                f"At {date}, {non_spy_nonzero} non-SPY tickers have nonzero weight; cap={cap}"
            )
