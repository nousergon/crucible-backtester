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
    _build_cell_cfg,
    _cell_passes_gate,
    _ensure_spy_column,
    _gate_thresholds,
    _select_rebalance_dates,
    compare_cov_sweep_to_baseline,
    compare_to_legacy,
    default_cov_sweep_cells,
    default_gamma_sweep_cells,
    run_cov_estimator_sweep,
    run_gamma_sweep,
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


# ─── A.4 covariance-estimator sweep tests ───────────────────────────────────
# Plan: alpha-engine-docs/private/optimizer-sota-upgrades-260526.md §A.4
#
# Sweep harness is tested with an injected stub backtest runner so unit
# tests don't require vectorbt + a synthetic-data pipeline. End-to-end is
# covered by TestEndToEndIntegration (single-cell) above; sweep semantics
# (ranking, gate, baseline) are isolated here.


def _stub_runner_factory(metrics_by_cfg: dict[tuple, dict]):
    """Build a stub backtest_runner that returns metrics keyed by a
    sortable signature of the cell cfg."""
    def _key(cfg: dict) -> tuple:
        return (
            cfg.get("covariance_shrinkage"),
            cfg.get("sigma_horizon_days"),
            cfg.get("ewma_lambda_decay"),
        )

    def _runner(*, optimizer_cfg: dict, **_kwargs) -> OptimizerBacktestResult:
        metrics = dict(metrics_by_cfg[_key(optimizer_cfg)])
        n_failures = int(metrics.pop("_n_solver_failures", 0))
        return OptimizerBacktestResult(
            target_weights=pd.DataFrame(),
            metrics=metrics,
            rebalance_dates=[],
            n_rebalances=0,
            n_solver_failures=n_failures,
            diagnostics_per_rebalance=[],
        )
    return _runner


def _ok_metrics(sortino: float | None, **overrides) -> dict:
    """Build a metrics dict that passes the absolute risk floors so the
    sortino comparison is the load-bearing piece. Override individual
    fields to force gate failures in specific tests."""
    base = {
        "sortino_ratio": sortino,
        "psr": 0.97,
        "cvar_95": -0.02,
        "max_drawdown": -0.10,
        "calmar_ratio": 1.5,
        "tracking_error_ann": 0.04,
        "mean_active_share": 0.15,
        "mean_spy_weight": 0.85,
        "turnover_one_way_ann": 2.0,
        "sharpe_ratio": (sortino * 0.7) if sortino is not None else None,
        "total_return": 0.20,
        "spy_return": 0.10,
        "total_alpha": 0.10,
        "total_trades": 200,
        "win_rate": 0.55,
        "n_rebalances": 50,
        "n_solver_failures": 0,
        "rebalance_freq_days": 5,
        "universe_cap": 30,
    }
    base.update(overrides)
    return base


class TestBuildCellCfg:
    def test_lw_h1_baseline_cell(self):
        cfg = _build_cell_cfg("ledoit_wolf", 1)
        assert cfg["covariance_shrinkage"] == "ledoit_wolf"
        assert cfg["sigma_horizon_days"] == 1
        assert cfg["risk_aversion"] == 5.0
        assert "ewma_lambda_decay" not in cfg

    def test_h21_applies_compensating_lambda_rescale(self):
        cfg_h1 = _build_cell_cfg("ledoit_wolf", 1)
        cfg_h21 = _build_cell_cfg("ledoit_wolf", 21)
        assert cfg_h21["risk_aversion"] == pytest.approx(cfg_h1["risk_aversion"] / 21.0)

    def test_ewma_cell_requires_lambda_decay(self):
        with pytest.raises(ValueError, match="ewma cells must specify ewma_lambda_decay"):
            _build_cell_cfg("ewma", 1)

    def test_ewma_cell_propagates_lambda(self):
        cfg = _build_cell_cfg("ewma", 1, ewma_lambda_decay=0.97)
        assert cfg["ewma_lambda_decay"] == 0.97


class TestDefaultCovSweepCells:
    def test_eight_default_cells_unique_names(self):
        cells = default_cov_sweep_cells()
        names = [n for n, _ in cells]
        assert len(names) == 8
        assert len(set(names)) == 8

    def test_baseline_is_ledoit_wolf_h1(self):
        cells = default_cov_sweep_cells()
        baseline_name, baseline_cfg = cells[0]
        assert baseline_name == "ledoit_wolf_h1"
        assert baseline_cfg["covariance_shrinkage"] == "ledoit_wolf"
        assert baseline_cfg["sigma_horizon_days"] == 1


class TestCellPassesGate:
    def test_passes_when_metrics_clear_all_floors(self):
        baseline = _ok_metrics(sortino=1.0)
        cell = _ok_metrics(sortino=1.1)
        assert _cell_passes_gate(cell, baseline) is True

    def test_fails_on_low_psr(self):
        baseline = _ok_metrics(sortino=1.0)
        cell = _ok_metrics(sortino=1.5, psr=0.5)
        assert _cell_passes_gate(cell, baseline) is False

    def test_fails_on_drawdown_floor_violation(self):
        baseline = _ok_metrics(sortino=1.0)
        cell = _ok_metrics(sortino=1.5, max_drawdown=-0.50)  # worse than -0.35 floor
        assert _cell_passes_gate(cell, baseline) is False

    def test_fails_on_cvar_floor_violation(self):
        baseline = _ok_metrics(sortino=1.0)
        cell = _ok_metrics(sortino=1.5, cvar_95=-0.10)  # worse than -0.05 floor
        assert _cell_passes_gate(cell, baseline) is False

    def test_fails_when_sortino_below_baseline_x_0_9(self):
        baseline = _ok_metrics(sortino=2.0)
        cell = _ok_metrics(sortino=1.0)  # < 2.0 × 0.9 = 1.8
        assert _cell_passes_gate(cell, baseline) is False

    def test_no_baseline_only_absolute_floors_apply(self):
        cell = _ok_metrics(sortino=0.1)  # very low sortino
        assert _cell_passes_gate(cell, None) is True  # only PSR/dd/CVaR matter without baseline


class TestRunCovEstimatorSweep:
    def test_sweep_runs_each_cell_and_returns_report_shape(self):
        cells = [
            ("lw_h1", _build_cell_cfg("ledoit_wolf", 1)),
            ("lw_h21", _build_cell_cfg("ledoit_wolf", 21)),
        ]
        metrics_by_cfg = {
            ("ledoit_wolf", 1, None): _ok_metrics(sortino=1.0),
            ("ledoit_wolf", 21, None): _ok_metrics(sortino=1.3),
        }
        report = run_cov_estimator_sweep(
            predictions_by_date={},
            price_matrix=pd.DataFrame(),
            spy_prices=pd.Series(dtype=float),
            sector_map={},
            executor_path="/nonexistent",
            cells=cells,
            backtest_runner=_stub_runner_factory(metrics_by_cfg),
        )
        assert set(report["cells"].keys()) == {"lw_h1", "lw_h21"}
        assert report["baseline_name"] == "lw_h1"
        # lw_h21 has higher sortino (1.3 vs 1.0) → ranks first
        assert report["ranking"][0][0] == "lw_h21"
        assert report["ranking"][0][1] == pytest.approx(1.3)
        # Both cells pass the gate; winner is highest-sortino passing cell
        assert report["winner_name"] == "lw_h21"
        assert report["gate_passes_per_cell"] == {"lw_h1": True, "lw_h21": True}

    def test_empty_cell_list_raises(self):
        with pytest.raises(ValueError, match="Empty cell list"):
            run_cov_estimator_sweep(
                predictions_by_date={},
                price_matrix=pd.DataFrame(),
                spy_prices=pd.Series(dtype=float),
                sector_map={},
                executor_path="/nonexistent",
                cells=[],
                backtest_runner=_stub_runner_factory({}),
            )

    def test_winner_is_none_when_no_cell_passes_gate(self):
        cells = [
            ("lw_h1", _build_cell_cfg("ledoit_wolf", 1)),
            ("oas_h1", _build_cell_cfg("oas", 1)),
        ]
        # Both cells violate the drawdown floor
        metrics_by_cfg = {
            ("ledoit_wolf", 1, None): _ok_metrics(sortino=1.0, max_drawdown=-0.50),
            ("oas", 1, None): _ok_metrics(sortino=1.5, max_drawdown=-0.45),
        }
        report = run_cov_estimator_sweep(
            predictions_by_date={},
            price_matrix=pd.DataFrame(),
            spy_prices=pd.Series(dtype=float),
            sector_map={},
            executor_path="/nonexistent",
            cells=cells,
            backtest_runner=_stub_runner_factory(metrics_by_cfg),
        )
        assert report["winner_name"] is None
        assert all(v is False for v in report["gate_passes_per_cell"].values())

    def test_sortino_max_cell_winning_gate_is_selected_even_when_lower_sharpe(self):
        """Confirms Sortino — NOT raw Sharpe — is the ranking metric per the
        skilled-risk framework (evaluator-revamp-260506.md)."""
        cells = [
            ("a", _build_cell_cfg("ledoit_wolf", 1)),
            ("b", _build_cell_cfg("oas", 1)),
        ]
        metrics_by_cfg = {
            # Cell A: high Sharpe (3.0), low Sortino (1.0)
            ("ledoit_wolf", 1, None): _ok_metrics(sortino=1.0, sharpe_ratio=3.0),
            # Cell B: low Sharpe (0.5), high Sortino (2.0)
            ("oas", 1, None): _ok_metrics(sortino=2.0, sharpe_ratio=0.5),
        }
        report = run_cov_estimator_sweep(
            predictions_by_date={},
            price_matrix=pd.DataFrame(),
            spy_prices=pd.Series(dtype=float),
            sector_map={},
            executor_path="/nonexistent",
            cells=cells,
            backtest_runner=_stub_runner_factory(metrics_by_cfg),
        )
        assert report["winner_name"] == "b"  # Sortino-max, not Sharpe-max

    def test_cells_with_none_sortino_rank_last(self):
        """Failed cells (None sortino) must not crash the ranking and must
        rank after any cell with a real sortino."""
        cells = [
            ("a", _build_cell_cfg("ledoit_wolf", 1)),
            ("b", _build_cell_cfg("oas", 1)),
        ]
        metrics_by_cfg = {
            ("ledoit_wolf", 1, None): _ok_metrics(sortino=1.0),
            ("oas", 1, None): _ok_metrics(sortino=None),  # broken cell
        }
        report = run_cov_estimator_sweep(
            predictions_by_date={},
            price_matrix=pd.DataFrame(),
            spy_prices=pd.Series(dtype=float),
            sector_map={},
            executor_path="/nonexistent",
            cells=cells,
            backtest_runner=_stub_runner_factory(metrics_by_cfg),
        )
        ranking_names = [name for name, _ in report["ranking"]]
        assert ranking_names == ["a", "b"]  # a (1.0) before b (None)


class TestCompareCovSweepToBaseline:
    def test_produces_per_cell_compare_to_legacy_entries(self):
        cells = [
            ("baseline", _build_cell_cfg("ledoit_wolf", 1)),
            ("challenger", _build_cell_cfg("oas", 1)),
        ]
        metrics_by_cfg = {
            ("ledoit_wolf", 1, None): _ok_metrics(sortino=1.0),
            ("oas", 1, None): _ok_metrics(sortino=1.5),
        }
        sweep = run_cov_estimator_sweep(
            predictions_by_date={},
            price_matrix=pd.DataFrame(),
            spy_prices=pd.Series(dtype=float),
            sector_map={},
            executor_path="/nonexistent",
            cells=cells,
            backtest_runner=_stub_runner_factory(metrics_by_cfg),
        )
        verdict = compare_cov_sweep_to_baseline(sweep)
        assert verdict["baseline_name"] == "baseline"
        assert verdict["winner_name"] == "challenger"
        assert set(verdict["comparisons"].keys()) == {"baseline", "challenger"}
        # Challenger's compare_to_legacy entry has the Sortino delta we'd expect
        delta = verdict["comparisons"]["challenger"]["deltas"]["sortino_delta"]
        assert delta == pytest.approx(0.5)


# ─── B.4b γ-sweep tests ─────────────────────────────────────────────────────
# Plan: alpha-engine-docs/private/optimizer-sota-upgrades-260526.md §B.4b
#
# Sweeps cfg["alpha_uncertainty_penalty"] over a log-spaced grid. Mirror
# of A.4 cov-sweep — stub backtest_runner so tests don't need vectorbt.


def _gamma_stub_runner_factory(metrics_by_gamma: dict[float, dict]):
    """Stub runner keyed by the γ value in optimizer_cfg.
    Captures alpha_uncertainty_by_date so tests can assert it was threaded."""
    captured: dict[str, object] = {}

    def _runner(*, optimizer_cfg: dict, alpha_uncertainty_by_date=None, **_kwargs):
        gamma = float(optimizer_cfg.get("alpha_uncertainty_penalty", 0.0))
        captured["last_alpha_uncertainty_by_date"] = alpha_uncertainty_by_date
        metrics = dict(metrics_by_gamma[gamma])
        n_failures = int(metrics.pop("_n_solver_failures", 0))
        return OptimizerBacktestResult(
            target_weights=pd.DataFrame(),
            metrics=metrics,
            rebalance_dates=[],
            n_rebalances=0,
            n_solver_failures=n_failures,
            diagnostics_per_rebalance=[],
        )
    _runner.captured = captured  # type: ignore[attr-defined]
    return _runner


class TestDefaultGammaSweepCells:
    def test_five_cells_with_baseline_first(self):
        cells = default_gamma_sweep_cells()
        assert len(cells) == 5
        names = [n for n, _ in cells]
        assert names[0] == "baseline_gamma_0"
        assert len(set(names)) == 5

    def test_baseline_is_gamma_zero(self):
        cells = default_gamma_sweep_cells()
        baseline_name, baseline_cfg = cells[0]
        assert baseline_cfg["alpha_uncertainty_penalty"] == 0.0

    def test_grid_is_log_spaced(self):
        cells = default_gamma_sweep_cells()
        gammas = [c["alpha_uncertainty_penalty"] for _, c in cells]
        # 0, 10, 100, 1000, 10000
        assert gammas == [0.0, 10.0, 100.0, 1000.0, 10000.0]


class TestRunGammaSweep:
    def _stub_alpha_uncertainty_by_date(self):
        return {"2024-01-08": {"AAPL": 0.025, "MSFT": 0.015}}

    def test_sweep_runs_each_cell_and_threads_uncertainty(self):
        cells = [
            ("g0",   {"alpha_uncertainty_penalty": 0.0}),
            ("g100", {"alpha_uncertainty_penalty": 100.0}),
        ]
        metrics_by_gamma = {
            0.0: _ok_metrics(sortino=1.0),
            100.0: _ok_metrics(sortino=1.4),
        }
        runner = _gamma_stub_runner_factory(metrics_by_gamma)
        unc = self._stub_alpha_uncertainty_by_date()
        report = run_gamma_sweep(
            predictions_by_date={},
            price_matrix=pd.DataFrame(),
            spy_prices=pd.Series(dtype=float),
            sector_map={},
            executor_path="/nonexistent",
            alpha_uncertainty_by_date=unc,
            cells=cells,
            backtest_runner=runner,
        )
        # Both cells ran
        assert set(report["cells"].keys()) == {"g0", "g100"}
        # g100 wins on Sortino
        assert report["ranking"][0][0] == "g100"
        assert report["winner_name"] == "g100"
        assert report["baseline_name"] == "g0"
        # uncertainty was threaded into each runner call
        assert runner.captured["last_alpha_uncertainty_by_date"] is unc

    def test_alpha_uncertainty_by_date_required_raises(self):
        """γ-sweep without σ_α̂ inputs has every cell behave identically.
        Plan policy: raise rather than emit a meaningless sweep report."""
        with pytest.raises(ValueError, match="required for γ-sweep"):
            run_gamma_sweep(
                predictions_by_date={},
                price_matrix=pd.DataFrame(),
                spy_prices=pd.Series(dtype=float),
                sector_map={},
                executor_path="/nonexistent",
                alpha_uncertainty_by_date=None,
                cells=default_gamma_sweep_cells(),
                backtest_runner=lambda **_: None,
            )

    def test_empty_cell_list_raises(self):
        with pytest.raises(ValueError, match="Empty cell list"):
            run_gamma_sweep(
                predictions_by_date={},
                price_matrix=pd.DataFrame(),
                spy_prices=pd.Series(dtype=float),
                sector_map={},
                executor_path="/nonexistent",
                alpha_uncertainty_by_date={},
                cells=[],
                backtest_runner=lambda **_: None,
            )

    def test_baseline_wins_when_uncertainty_signal_is_noise(self):
        """When σ_α̂ is uninformative, higher γ doesn't help — the baseline
        γ=0 wins on Sortino. Verifies the sweep doesn't artificially favor
        the new term."""
        cells = default_gamma_sweep_cells()
        # All cells perform identically except baseline marginally better
        metrics_by_gamma = {
            0.0:     _ok_metrics(sortino=1.2),
            10.0:    _ok_metrics(sortino=1.15),
            100.0:   _ok_metrics(sortino=1.10),
            1000.0:  _ok_metrics(sortino=1.05),
            10000.0: _ok_metrics(sortino=0.95),  # too much penalty hurts
        }
        runner = _gamma_stub_runner_factory(metrics_by_gamma)
        report = run_gamma_sweep(
            predictions_by_date={},
            price_matrix=pd.DataFrame(),
            spy_prices=pd.Series(dtype=float),
            sector_map={},
            executor_path="/nonexistent",
            alpha_uncertainty_by_date={"d": {}},
            cells=cells,
            backtest_runner=runner,
        )
        # Baseline wins on Sortino
        assert report["winner_name"] == "baseline_gamma_0"

    def test_high_gamma_wins_when_uncertainty_signal_is_informative(self):
        """When the uncertainty signal is genuinely informative, the
        optimal γ is non-zero — the sweep identifies it."""
        cells = default_gamma_sweep_cells()
        # Inverted-U: γ=100 optimal
        metrics_by_gamma = {
            0.0:     _ok_metrics(sortino=1.0),
            10.0:    _ok_metrics(sortino=1.2),
            100.0:   _ok_metrics(sortino=1.6),  # peak
            1000.0:  _ok_metrics(sortino=1.3),
            10000.0: _ok_metrics(sortino=0.7),
        }
        runner = _gamma_stub_runner_factory(metrics_by_gamma)
        report = run_gamma_sweep(
            predictions_by_date={},
            price_matrix=pd.DataFrame(),
            spy_prices=pd.Series(dtype=float),
            sector_map={},
            executor_path="/nonexistent",
            alpha_uncertainty_by_date={"d": {}},
            cells=cells,
            backtest_runner=runner,
        )
        assert report["winner_name"] == "gamma_100"
        # Ranking reflects the inverted-U
        ranked_names = [n for n, _ in report["ranking"]]
        assert ranked_names[0] == "gamma_100"

    def test_winner_none_when_all_cells_fail_gate(self):
        """If every γ produces metrics that violate absolute risk floors,
        no winner is named — surfaces the failure rather than picking the
        least-bad cell."""
        cells = default_gamma_sweep_cells()
        # Every cell violates the drawdown floor (-0.35 absolute)
        metrics_by_gamma = {
            g: _ok_metrics(sortino=1.0 + i * 0.1, max_drawdown=-0.50)
            for i, g in enumerate([0.0, 10.0, 100.0, 1000.0, 10000.0])
        }
        runner = _gamma_stub_runner_factory(metrics_by_gamma)
        report = run_gamma_sweep(
            predictions_by_date={},
            price_matrix=pd.DataFrame(),
            spy_prices=pd.Series(dtype=float),
            sector_map={},
            executor_path="/nonexistent",
            alpha_uncertainty_by_date={"d": {}},
            cells=cells,
            backtest_runner=runner,
        )
        assert report["winner_name"] is None
        assert all(v is False for v in report["gate_passes_per_cell"].values())


class TestBuildOptimizerInputsAlphaUncertainty:
    """The new alpha_uncertainty_for_date passthrough on _build_optimizer_inputs
    populates the optimizer kwargs cleanly."""

    def test_alpha_uncertainty_populated_when_provided(self):
        from analysis.portfolio_optimizer_backtest import _build_optimizer_inputs

        tickers = ["AAPL", "MSFT"]
        sector_map = {t: "Technology" for t in tickers}
        price_matrix = _synthetic_price_matrix(tickers, n_days=200, seed=0)
        spy = _synthetic_spy_series(n_days=200, seed=99).reindex(price_matrix.index)
        price_matrix["SPY"] = spy
        predictions = {"AAPL": 0.04, "MSFT": 0.02}
        rebal_date = str(price_matrix.index[150].date())
        kwargs = _build_optimizer_inputs(
            rebal_date=rebal_date,
            predictions=predictions,
            price_matrix=price_matrix,
            sector_map=sector_map,
            universe_cap=30,
            max_position_pct=0.08,
            min_score_proxy=None,
            cfg={},
            alpha_uncertainty_for_date={"AAPL": 0.025, "MSFT": 0.015},
        )
        unc = kwargs["alpha_uncertainty"]
        assert unc is not None
        # Tickers order: selected (AAPL, MSFT) + SPY + CASH
        idx_aapl = kwargs["tickers"].index("AAPL")
        idx_msft = kwargs["tickers"].index("MSFT")
        idx_spy = kwargs["tickers"].index("SPY")
        idx_cash = kwargs["tickers"].index("CASH")
        assert unc[idx_aapl] == pytest.approx(0.025)
        assert unc[idx_msft] == pytest.approx(0.015)
        assert unc[idx_spy] == 0.0
        assert unc[idx_cash] == 0.0

    def test_alpha_uncertainty_is_none_when_not_provided(self):
        """Backwards-compat: when caller doesn't pass uncertainty (legacy
        callsite or pre-B.4b), kwargs["alpha_uncertainty"] is None and
        solve_target_weights' B.3 path treats it as 'no penalty.'"""
        from analysis.portfolio_optimizer_backtest import _build_optimizer_inputs

        tickers = ["AAPL", "MSFT"]
        sector_map = {t: "Technology" for t in tickers}
        price_matrix = _synthetic_price_matrix(tickers, n_days=200, seed=0)
        spy = _synthetic_spy_series(n_days=200, seed=99).reindex(price_matrix.index)
        price_matrix["SPY"] = spy
        predictions = {"AAPL": 0.04, "MSFT": 0.02}
        rebal_date = str(price_matrix.index[150].date())
        kwargs = _build_optimizer_inputs(
            rebal_date=rebal_date,
            predictions=predictions,
            price_matrix=price_matrix,
            sector_map=sector_map,
            universe_cap=30,
            max_position_pct=0.08,
            min_score_proxy=None,
            cfg={},
        )
        assert kwargs["alpha_uncertainty"] is None

    def test_missing_ticker_std_yields_nan(self):
        """Per-ticker missing entry → NaN in the array (B.3 path skips penalty)."""
        from analysis.portfolio_optimizer_backtest import _build_optimizer_inputs

        tickers = ["AAPL", "MSFT"]
        sector_map = {t: "Technology" for t in tickers}
        price_matrix = _synthetic_price_matrix(tickers, n_days=200, seed=0)
        spy = _synthetic_spy_series(n_days=200, seed=99).reindex(price_matrix.index)
        price_matrix["SPY"] = spy
        predictions = {"AAPL": 0.04, "MSFT": 0.02}
        rebal_date = str(price_matrix.index[150].date())
        kwargs = _build_optimizer_inputs(
            rebal_date=rebal_date,
            predictions=predictions,
            price_matrix=price_matrix,
            sector_map=sector_map,
            universe_cap=30,
            max_position_pct=0.08,
            min_score_proxy=None,
            cfg={},
            alpha_uncertainty_for_date={"AAPL": 0.025},  # MSFT missing
        )
        unc = kwargs["alpha_uncertainty"]
        idx_msft = kwargs["tickers"].index("MSFT")
        assert np.isnan(unc[idx_msft])
