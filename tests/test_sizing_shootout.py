"""tests/test_sizing_shootout.py — config#3081 S-slot sizing shootout.

Planted-truth unit tests for the two new sizing arms
(``synthetic.vectorized_entries``'s ``sizing_arm="risk_parity"`` /
``"fractional_kelly"``), an end-to-end integration test of
``synthetic.vectorized_sweep.run_sizing_shootout`` proving the
shared-gates / arm-specific-sizing design holds, and an emission-wiring
test analogous to ``tests/test_double_sort_stage.py`` for
``analysis.sizing_shootout.compute_sizing_shootout``.

Coverage
--------
  * Risk-parity: raw weights EXACTLY inverse-proportional to vol
    (weight[i] * vol[i] constant), correct ordering.
  * Fractional-Kelly: raw weight == fraction * alpha / variance for 2+
    fractions, max_position_pct cap clips an oversized raw Kelly
    weight, non-positive alpha/variance -> zero (not negative) weight.
  * compute_realized_vol_20d: known-input parity + NaN fallback.
  * run_sizing_shootout: all arms produce well-formed results with the
    SAME gating (same number of eligible signals pre-sizing-cap).
  * compute_sizing_shootout: status/shape + promotion-candidate logic.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest


_EXECUTOR_ROOT = os.path.expanduser("~/Development/alpha-engine")
if os.path.isdir(_EXECUTOR_ROOT) and _EXECUTOR_ROOT not in sys.path:
    sys.path.insert(0, _EXECUTOR_ROOT)


from synthetic.vectorized_entries import (
    SIZING_ARM_CONVICTION,
    SIZING_ARM_FRACTIONAL_KELLY,
    SIZING_ARM_RISK_PARITY,
    SR_MARKET_WEIGHT,
    CONV_STABLE,
    REGIME_BULL,
    VectorizedEntryConfig,
    compute_realized_vol_20d,
    compute_vectorized_entries,
)
from synthetic.vectorized_sim import VectorizedSimulator
from synthetic.vectorized_sweep import run_sizing_shootout


def _ticker_index(*tickers: str) -> dict:
    return {t: i for i, t in enumerate(tickers)}


def _empty_signal_arrays(n: int) -> dict:
    return {
        "signal_ticker_idx": np.arange(n, dtype=np.int32),
        "signal_score": np.full(n, 80.0, dtype=np.float64),
        "signal_sector_idx": np.zeros(n, dtype=np.int32),
        "signal_sector_rating": np.full(n, SR_MARKET_WEIGHT, dtype=np.int8),
        "signal_conviction": np.full(n, CONV_STABLE, dtype=np.int8),
        "signal_upside": np.full(n, 0.20, dtype=np.float64),
        "signal_atr_pct": np.full(n, np.nan, dtype=np.float64),
        "signal_pred_confidence": np.full(n, np.nan, dtype=np.float64),
        "signal_p_up": np.full(n, np.nan, dtype=np.float64),
        "signal_days_to_earnings": np.full(n, -1, dtype=np.int32),
        "signal_feature_coverage": np.full(n, np.nan, dtype=np.float64),
        "signal_gbm_veto": np.zeros(n, dtype=bool),
        "signal_momentum_at_date": np.full(n, np.nan, dtype=np.float64),
    }


# ────────────────────────────────────────────────────────────────────
# compute_realized_vol_20d
# ────────────────────────────────────────────────────────────────────


class TestComputeRealizedVol20d:
    def test_known_std_matches_annualized(self):
        # Constant-magnitude alternating returns -> known sample std.
        rets = np.array([0.01, -0.01] * 10).reshape(1, 20)
        vol = compute_realized_vol_20d(rets, annualize=True)
        expected = float(np.std(rets, ddof=1) * np.sqrt(252.0))
        assert vol[0] == pytest.approx(expected, rel=1e-9)

    def test_raw_daily_std_when_not_annualized(self):
        rets = np.array([0.02, -0.01, 0.015, -0.02] * 5).reshape(1, 20)
        vol_annual = compute_realized_vol_20d(rets, annualize=True)
        vol_daily = compute_realized_vol_20d(rets, annualize=False)
        assert vol_annual[0] == pytest.approx(vol_daily[0] * np.sqrt(252.0))

    def test_insufficient_history_yields_nan(self):
        rets = np.array([[0.01, np.nan] + [np.nan] * 18])
        vol = compute_realized_vol_20d(rets)
        assert np.isnan(vol[0])

    def test_uses_only_trailing_20_columns(self):
        # First 10 columns are huge outliers; only the trailing 20
        # (all small, constant) should be used if a wider window is
        # passed in.
        wide = np.concatenate([np.full(10, 100.0), np.array([0.01, -0.01] * 10)])
        vol = compute_realized_vol_20d(wide.reshape(1, -1), annualize=False)
        trailing_only = compute_realized_vol_20d(
            wide[-20:].reshape(1, -1), annualize=False,
        )
        assert vol[0] == pytest.approx(trailing_only[0])


# ────────────────────────────────────────────────────────────────────
# Risk-parity planted truth
# ────────────────────────────────────────────────────────────────────


class TestRiskParityPlantedTruth:
    def test_weights_exactly_inverse_proportional_to_vol(self):
        ti = _ticker_index("A", "B", "C", "D")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti, init_cash=1_000_000)
        # High caps + all conviction-style adjustments disabled so the
        # RAW risk-parity weight survives uncapped through to dollar_size.
        config = VectorizedEntryConfig.from_uniform(
            n_combos=1,
            max_position_pct=1.0, max_sector_pct=1.0, max_equity_pct=1.0,
            atr_sizing_enabled=False, confidence_sizing_enabled=False,
            staleness_discount_enabled=False, earnings_sizing_enabled=False,
            coverage_sizing_enabled=False, correlation_block_enabled=False,
            momentum_gate_enabled=False, min_score_to_enter=0.0,
            min_position_dollar=0.0,
        )
        n = 4
        sigs = _empty_signal_arrays(n)
        vols = np.array([0.10, 0.20, 0.40, 0.05])

        decisions = compute_vectorized_entries(
            sim, **sigs,
            prices=np.full(n, 100.0),
            nav_per_combo=np.array([1_000_000.0]),
            dd_multiplier_per_combo=np.array([1.0]),
            market_regime=REGIME_BULL,
            signal_age_days=0,
            config=config,
            sizing_arm=SIZING_ARM_RISK_PARITY,
            signal_realized_vol_20d=vols,
        )
        # Recover pre-share-rounding dollar weight via entry_dollar / nav.
        weight = decisions.entry_dollar[0] / 1_000_000.0
        assert np.all(decisions.entry_passed[0]), "all 4 signals should pass with caps disabled"

        products = weight * vols
        # weight[i] * vol[i] constant across all i (inverse proportionality).
        assert products == pytest.approx(products[0], rel=1e-6)

        # Ordering: lowest vol -> highest weight.
        order = np.argsort(vols)  # ascending vol
        assert np.all(np.diff(weight[order]) < 0), "weight should strictly decrease as vol increases"

    def test_unknown_vol_falls_back_not_nan_or_inf(self):
        ti = _ticker_index("A", "B")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti, init_cash=1_000_000)
        config = VectorizedEntryConfig.from_uniform(
            n_combos=1,
            max_position_pct=1.0, max_sector_pct=1.0, max_equity_pct=1.0,
            atr_sizing_enabled=False, confidence_sizing_enabled=False,
            staleness_discount_enabled=False, earnings_sizing_enabled=False,
            coverage_sizing_enabled=False, correlation_block_enabled=False,
            momentum_gate_enabled=False, min_score_to_enter=0.0,
            min_position_dollar=0.0,
        )
        sigs = _empty_signal_arrays(2)
        vols = np.array([0.20, np.nan])  # second signal has unknown vol

        decisions = compute_vectorized_entries(
            sim, **sigs,
            prices=np.full(2, 100.0),
            nav_per_combo=np.array([1_000_000.0]),
            dd_multiplier_per_combo=np.array([1.0]),
            market_regime=REGIME_BULL,
            signal_age_days=0,
            config=config,
            sizing_arm=SIZING_ARM_RISK_PARITY,
            signal_realized_vol_20d=vols,
        )
        weight = decisions.entry_dollar[0] / 1_000_000.0
        assert np.all(np.isfinite(weight))
        assert np.all(weight > 0)


# ────────────────────────────────────────────────────────────────────
# Fractional-Kelly planted truth
# ────────────────────────────────────────────────────────────────────


class TestFractionalKellyPlantedTruth:
    def _uncapped_config(self, n_combos: int, kelly_fraction) -> VectorizedEntryConfig:
        return VectorizedEntryConfig.from_uniform(
            n_combos=n_combos,
            max_position_pct=100.0, max_sector_pct=100.0, max_equity_pct=100.0,
            atr_sizing_enabled=False, confidence_sizing_enabled=False,
            staleness_discount_enabled=False, earnings_sizing_enabled=False,
            coverage_sizing_enabled=False, correlation_block_enabled=False,
            momentum_gate_enabled=False, min_score_to_enter=0.0,
            min_position_dollar=0.0,
            kelly_fraction=kelly_fraction,
        )

    def test_raw_weight_matches_fraction_times_alpha_over_variance(self):
        ti = _ticker_index("A", "B", "C")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti, init_cash=1_000_000)
        n = 3
        alpha = np.array([0.10, 0.20, 0.05])
        variance = np.array([0.04, 0.01, 0.09])  # = vol^2 for vol [0.2, 0.1, 0.3]
        vol = np.sqrt(variance)

        for fraction in (0.25, 0.5, 0.375):
            config = self._uncapped_config(1, fraction)
            sigs = _empty_signal_arrays(n)
            decisions = compute_vectorized_entries(
                sim, **sigs,
                prices=np.full(n, 100.0),
                nav_per_combo=np.array([1_000_000.0]),
                dd_multiplier_per_combo=np.array([1.0]),
                market_regime=REGIME_BULL,
                signal_age_days=0,
                config=config,
                sizing_arm=SIZING_ARM_FRACTIONAL_KELLY,
                signal_realized_vol_20d=vol,
                signal_alpha=alpha,
            )
            weight = decisions.entry_dollar[0] / 1_000_000.0
            expected = fraction * alpha / variance
            assert weight == pytest.approx(expected, rel=1e-6), f"fraction={fraction}"

    def test_max_position_pct_cap_clips_oversized_kelly_weight(self):
        ti = _ticker_index("A")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti, init_cash=1_000_000)
        # alpha/variance = 0.5/0.01 = 50 -> fraction 0.5 -> raw weight 25.0,
        # WAY beyond any sane cap. max_position_pct=0.05 should clip it.
        config = VectorizedEntryConfig.from_uniform(
            n_combos=1,
            max_position_pct=0.05, max_sector_pct=1.0, max_equity_pct=1.0,
            atr_sizing_enabled=False, confidence_sizing_enabled=False,
            staleness_discount_enabled=False, earnings_sizing_enabled=False,
            coverage_sizing_enabled=False, correlation_block_enabled=False,
            momentum_gate_enabled=False, min_score_to_enter=0.0,
            min_position_dollar=0.0,
            kelly_fraction=0.5,
        )
        sigs = _empty_signal_arrays(1)
        decisions = compute_vectorized_entries(
            sim, **sigs,
            prices=np.array([100.0]),
            nav_per_combo=np.array([1_000_000.0]),
            dd_multiplier_per_combo=np.array([1.0]),
            market_regime=REGIME_BULL,
            signal_age_days=0,
            config=config,
            sizing_arm=SIZING_ARM_FRACTIONAL_KELLY,
            signal_realized_vol_20d=np.array([0.1]),  # variance = 0.01
            signal_alpha=np.array([0.5]),
        )
        weight = decisions.entry_dollar[0, 0] / 1_000_000.0
        assert weight == pytest.approx(0.05, rel=1e-6)

    def test_nonpositive_alpha_yields_zero_not_negative(self):
        ti = _ticker_index("A", "B")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti, init_cash=1_000_000)
        config = self._uncapped_config(1, 0.25)
        sigs = _empty_signal_arrays(2)
        decisions = compute_vectorized_entries(
            sim, **sigs,
            prices=np.full(2, 100.0),
            nav_per_combo=np.array([1_000_000.0]),
            dd_multiplier_per_combo=np.array([1.0]),
            market_regime=REGIME_BULL,
            signal_age_days=0,
            config=config,
            sizing_arm=SIZING_ARM_FRACTIONAL_KELLY,
            signal_realized_vol_20d=np.array([0.2, 0.2]),
            signal_alpha=np.array([-0.10, 0.0]),  # negative and zero alpha
        )
        weight = decisions.entry_dollar[0] / 1_000_000.0
        assert np.all(weight >= 0.0)
        assert weight[0] == pytest.approx(0.0)
        assert weight[1] == pytest.approx(0.0)
        # Both signals size to $0 -> blocked (shares round to zero), not
        # a crash or negative dollar_size.
        assert not decisions.entry_passed[0, 0]
        assert not decisions.entry_passed[0, 1]

    def test_zero_or_negative_variance_yields_zero_weight(self):
        ti = _ticker_index("A")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti, init_cash=1_000_000)
        config = self._uncapped_config(1, 0.25)
        sigs = _empty_signal_arrays(1)
        decisions = compute_vectorized_entries(
            sim, **sigs,
            prices=np.array([100.0]),
            nav_per_combo=np.array([1_000_000.0]),
            dd_multiplier_per_combo=np.array([1.0]),
            market_regime=REGIME_BULL,
            signal_age_days=0,
            config=config,
            sizing_arm=SIZING_ARM_FRACTIONAL_KELLY,
            signal_realized_vol_20d=np.array([0.0]),  # variance = 0
            signal_alpha=np.array([0.10]),
        )
        weight = decisions.entry_dollar[0, 0] / 1_000_000.0
        assert weight == pytest.approx(0.0)
        assert np.isfinite(weight)


# ────────────────────────────────────────────────────────────────────
# Missing-input errors
# ────────────────────────────────────────────────────────────────────


class TestSizingArmInputValidation:
    def test_risk_parity_requires_vol_array(self):
        ti = _ticker_index("A")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti, init_cash=1_000_000)
        config = VectorizedEntryConfig.from_uniform(n_combos=1)
        sigs = _empty_signal_arrays(1)
        with pytest.raises(ValueError, match="risk_parity"):
            compute_vectorized_entries(
                sim, **sigs,
                prices=np.array([100.0]),
                nav_per_combo=np.array([1_000_000.0]),
                dd_multiplier_per_combo=np.array([1.0]),
                market_regime=REGIME_BULL,
                signal_age_days=0,
                config=config,
                sizing_arm=SIZING_ARM_RISK_PARITY,
            )

    def test_fractional_kelly_requires_alpha_and_vol(self):
        ti = _ticker_index("A")
        sim = VectorizedSimulator(n_combos=1, ticker_index=ti, init_cash=1_000_000)
        config = VectorizedEntryConfig.from_uniform(n_combos=1)
        sigs = _empty_signal_arrays(1)
        with pytest.raises(ValueError, match="fractional_kelly"):
            compute_vectorized_entries(
                sim, **sigs,
                prices=np.array([100.0]),
                nav_per_combo=np.array([1_000_000.0]),
                dd_multiplier_per_combo=np.array([1.0]),
                market_regime=REGIME_BULL,
                signal_age_days=0,
                config=config,
                sizing_arm=SIZING_ARM_FRACTIONAL_KELLY,
                signal_realized_vol_20d=np.array([0.2]),
                # signal_alpha missing
            )

    def test_default_sizing_arm_is_conviction(self):
        """Default sizing_arm value is the incumbent — a caller that
        doesn't pass sizing_arm at all gets today's exact behavior."""
        import inspect
        sig = inspect.signature(compute_vectorized_entries)
        assert sig.parameters["sizing_arm"].default == SIZING_ARM_CONVICTION


# ────────────────────────────────────────────────────────────────────
# End-to-end: run_sizing_shootout shared-gates integration test
# ────────────────────────────────────────────────────────────────────


def _make_price_matrix(n_dates: int, tickers: list) -> pd.DataFrame:
    rng = np.random.default_rng(13)
    idx = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    data = {}
    for i, t in enumerate(tickers):
        base = 100 + rng.normal(0, 3, n_dates).cumsum() * 0.3 + i * 5
        data[t] = np.maximum(base, 5.0)
    return pd.DataFrame(data, index=idx)


def _make_ohlcv(price_matrix: pd.DataFrame) -> dict:
    rng = np.random.default_rng(17)
    out = {}
    for t in price_matrix.columns:
        closes = price_matrix[t].to_numpy()
        highs = closes + rng.uniform(0, 1, len(closes))
        lows = closes - rng.uniform(0, 1, len(closes))
        opens = np.concatenate(([closes[0]], closes[:-1]))
        out[t] = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes},
            index=price_matrix.index,
        )
    return out


class _FakeFeatureLookup:
    def __init__(self, atr_dollar, rsi, momentum_20d_pct, returns):
        self.atr_dollar = atr_dollar
        self.rsi = rsi
        self.momentum_20d_pct = momentum_20d_pct
        self.returns = returns


def _make_feature_lookup(ohlcv: dict, lookback: int = 20) -> _FakeFeatureLookup:
    atr, rsi, mom, rets = {}, {}, {}, {}
    for ticker, df in ohlcv.items():
        close = df["close"]
        atr[ticker] = pd.Series(np.full(len(close), 1.5), index=close.index)
        rsi[ticker] = pd.Series(np.full(len(close), 50.0), index=close.index)
        mom[ticker] = (close.pct_change(periods=lookback) * 100.0)
        rets[ticker] = close.pct_change()
    return _FakeFeatureLookup(atr, rsi, mom, rets)


from dataclasses import dataclass, field


@dataclass
class _FakeSignalLookup:
    signals_raw_filtered: dict
    signals_by_ticker: dict
    universe_sectors: dict
    actionable: dict = field(default_factory=dict)


class TestRunSizingShootoutIntegration:
    def test_all_arms_share_gating_on_small_fixture(self):
        tickers = ["AAPL", "MSFT", "JNJ", "XOM"]
        n_dates = 80
        pm = _make_price_matrix(n_dates, tickers)
        ohlcv = _make_ohlcv(pm)
        fl = _make_feature_lookup(ohlcv)
        sector_map = {
            "AAPL": "Technology", "MSFT": "Technology",
            "JNJ": "Healthcare", "XOM": "Energy",
        }

        signal_lookups = {}
        for i, date in enumerate(pm.index):
            ds = date.strftime("%Y-%m-%d")
            enter = []
            if i >= 30 and i % 4 == 0:
                # Multiple simultaneous candidates so risk-parity/Kelly
                # weights can genuinely diverge from equal-weight.
                enter = [
                    {"ticker": "AAPL", "score": 80, "sector": "Technology",
                     "sector_rating": "market_weight", "conviction": "stable",
                     "price_target_upside": 0.15},
                    {"ticker": "MSFT", "score": 82, "sector": "Technology",
                     "sector_rating": "market_weight", "conviction": "stable",
                     "price_target_upside": 0.25},
                    {"ticker": "JNJ", "score": 75, "sector": "Healthcare",
                     "sector_rating": "market_weight", "conviction": "stable",
                     "price_target_upside": 0.10},
                ]
            signal_lookups[ds] = _FakeSignalLookup(
                signals_raw_filtered={"universe": [], "buy_candidates": [], "date": ds},
                signals_by_ticker={},
                universe_sectors=sector_map,
                actionable={"enter": enter, "exit": [], "reduce": [], "hold": []},
            )

        combos = [{
            "min_score": 70, "max_position_pct": 0.10,
            "atr_sizing_enabled": False, "confidence_sizing_enabled": False,
            "staleness_discount_enabled": False, "earnings_sizing_enabled": False,
            "coverage_sizing_enabled": False, "correlation_block_enabled": False,
            "momentum_gate_enabled": False, "max_sector_pct": 1.0,
            "max_equity_pct": 1.0,
        }]

        results = run_sizing_shootout(
            combo_configs=combos,
            kelly_fractions=(0.25, 0.5),
            price_matrix=pm,
            ohlcv_by_ticker=ohlcv,
            signal_lookups=signal_lookups,
            feature_lookup=fl,
            spy_prices=None,
            sector_map=sector_map,
            fee_rate=0.001,
        )

        expected_labels = {
            "conviction", "risk_parity",
            "fractional_kelly_0.25", "fractional_kelly_0.5",
        }
        assert set(results) == expected_labels

        # Shared-gates invariant: every arm sees the same candidate
        # signals and the same non-sizing gates, so entries_applied
        # should be IDENTICAL across arms on this fixture (caps are
        # generous enough that sizing never zeroes out a whole entry
        # via min_position_dollar in this fixture).
        entries_applied = {
            label: diag["entries_applied"] for label, (_, diag) in results.items()
        }
        assert len(set(entries_applied.values())) == 1, (
            f"entries_applied diverged across arms: {entries_applied}"
        )

        # Well-formed: every arm produced a real (non-crashing) result
        # with the expected diagnostics keys and orders_per_combo shape.
        for label, (orders_per_combo, diagnostics) in results.items():
            assert len(orders_per_combo) == 1  # 1 combo
            assert diagnostics["n_combos"] == 1
            assert diagnostics["n_dates"] == n_dates
            assert "nav_history" in diagnostics
            assert diagnostics["nav_history"].shape == (1, n_dates)

    def test_conviction_arm_unaffected_by_default(self):
        """run_vectorized_sweep called without sizing_arm at all still
        defaults to conviction — no signature-breaking regression for
        existing callers."""
        import inspect
        from synthetic.vectorized_sweep import run_vectorized_sweep
        sig = inspect.signature(run_vectorized_sweep)
        assert sig.parameters["sizing_arm"].default == SIZING_ARM_CONVICTION


# ────────────────────────────────────────────────────────────────────
# analysis.sizing_shootout.compute_sizing_shootout
# ────────────────────────────────────────────────────────────────────


class TestComputeSizingShootout:
    def _shootout_results_fixture(self):
        tickers = ["AAPL", "MSFT", "JNJ"]
        n_dates = 60
        pm = _make_price_matrix(n_dates, tickers)
        ohlcv = _make_ohlcv(pm)
        fl = _make_feature_lookup(ohlcv)
        sector_map = {"AAPL": "Technology", "MSFT": "Technology", "JNJ": "Healthcare"}

        signal_lookups = {}
        for i, date in enumerate(pm.index):
            ds = date.strftime("%Y-%m-%d")
            enter = []
            if i >= 25 and i % 4 == 0:
                enter = [
                    {"ticker": "AAPL", "score": 80, "sector": "Technology",
                     "sector_rating": "market_weight", "conviction": "stable",
                     "price_target_upside": 0.15},
                    {"ticker": "MSFT", "score": 82, "sector": "Technology",
                     "sector_rating": "market_weight", "conviction": "stable",
                     "price_target_upside": 0.25},
                ]
            signal_lookups[ds] = _FakeSignalLookup(
                signals_raw_filtered={"universe": [], "buy_candidates": [], "date": ds},
                signals_by_ticker={}, universe_sectors=sector_map,
                actionable={"enter": enter, "exit": [], "reduce": [], "hold": []},
            )

        combos = [{
            "min_score": 70, "max_position_pct": 0.10,
            "atr_sizing_enabled": False, "confidence_sizing_enabled": False,
            "staleness_discount_enabled": False, "earnings_sizing_enabled": False,
            "coverage_sizing_enabled": False, "correlation_block_enabled": False,
            "momentum_gate_enabled": False, "max_sector_pct": 1.0,
            "max_equity_pct": 1.0,
        }]
        results = run_sizing_shootout(
            combo_configs=combos,
            price_matrix=pm, ohlcv_by_ticker=ohlcv,
            signal_lookups=signal_lookups, feature_lookup=fl,
            spy_prices=None, sector_map=sector_map, fee_rate=0.001,
        )
        return results, pm, combos

    def test_produces_well_formed_artifact(self):
        from analysis.sizing_shootout import compute_sizing_shootout
        results, pm, combos = self._shootout_results_fixture()

        artifact = compute_sizing_shootout(
            results,
            run_date="2026-07-21",
            init_cash=1_000_000.0,
            spy_prices=None,
            dates=pm.index,
            combo_configs=combos,
            fee_rate=0.001,
        )
        assert artifact["status"] == "ok"
        assert artifact["run_date"] == "2026-07-21"
        assert set(artifact["arms"]) >= {"conviction", "risk_parity"}
        assert artifact["incumbent_arm"] == "conviction"
        assert isinstance(artifact["promotion_candidates"], list)
        assert artifact["cost_model"]["fee_rate"] == pytest.approx(0.001)
        assert "OBSERVE" in artifact["note"]
        for label, summary in artifact["arms"].items():
            assert "sharpe" in summary
            assert "max_drawdown" in summary
            assert "turnover" in summary
            assert "realized_alpha" in summary

    def test_empty_results_yields_no_arms_status(self):
        from analysis.sizing_shootout import compute_sizing_shootout
        artifact = compute_sizing_shootout(
            {}, run_date="2026-07-21", init_cash=1_000_000.0,
            spy_prices=None, dates=pd.date_range("2024-01-01", periods=5),
            combo_configs=[{}], fee_rate=0.001,
        )
        assert artifact["status"] == "no_arms"

    def test_promotion_candidate_requires_both_sharpe_and_drawdown_beat(self):
        from analysis.sizing_shootout import _beats_incumbent

        incumbent = {"sharpe": 1.0, "max_drawdown": -0.10}
        # Better sharpe, worse drawdown -> should NOT count as beating.
        mixed = {"sharpe": 1.5, "max_drawdown": -0.20}
        assert _beats_incumbent(mixed, incumbent) is False

        # Better on both -> counts.
        better = {"sharpe": 1.5, "max_drawdown": -0.05}
        assert _beats_incumbent(better, incumbent) is True

        # Missing data -> not a candidate (honest false, no crash).
        missing = {"sharpe": None, "max_drawdown": -0.05}
        assert _beats_incumbent(missing, incumbent) is False
