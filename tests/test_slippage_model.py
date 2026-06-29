"""Tests for the realistic slippage model (config#919).

The model is log-linear OLS (numpy-only). Tests pin: under-sample fallback,
coefficient recovery on synthetic data generated from a known model, prediction
monotonicity (size ↑ → slippage ↑; liquidity ↑ → slippage ↓), per-trigger
intercepts, and graceful handling of degenerate inputs.
"""

import math

import numpy as np
import pytest

from analysis.slippage_model import (
    DEFAULT_FLAT_SLIPPAGE_BPS,
    SlippageModel,
    build_observations_from_trades,
    fit_slippage_model,
    normalize_trigger,
    predict_slippage_bps,
)


def _synthetic_obs(n=400, seed=7):
    """Generate observations from a KNOWN log-linear slippage model + noise."""
    rng = np.random.default_rng(seed)
    triggers = ["pullback", "vwap", "market"]
    trig_intercept = {"pullback": 5.0, "vwap": 12.0, "market": 20.0}
    # true continuous coefs (applied to log1p of each feature)
    true = {
        "market_cap": -1.5,    # bigger cap → less slippage
        "dollar_volume": -2.0,  # more liquidity → less slippage
        "order_notional": 3.0,  # bigger order → more slippage
        "volatility": 8.0,      # more vol → more slippage
    }
    obs = []
    for _ in range(n):
        mc = rng.uniform(1e9, 5e11)
        dv = rng.uniform(1e6, 5e9)
        notional = rng.uniform(1e3, 5e5)
        vol = rng.uniform(0.005, 0.06)
        trig = triggers[rng.integers(0, 3)]
        slip = (
            trig_intercept[trig]
            + true["market_cap"] * math.log1p(mc)
            + true["dollar_volume"] * math.log1p(dv)
            + true["order_notional"] * math.log1p(notional)
            + true["volatility"] * math.log1p(vol)
            + rng.normal(0, 0.5)
        )
        obs.append({
            "slippage_bps": slip,
            "market_cap": mc, "dollar_volume": dv,
            "order_notional": notional, "volatility": vol,
            "trigger_type": trig,
        })
    return obs, true, trig_intercept


class TestFit:
    def test_under_sample_returns_none(self):
        obs, _, _ = _synthetic_obs(n=10)
        assert fit_slippage_model(obs, min_observations=30) is None

    def test_recovers_continuous_coef_signs(self):
        obs, true, _ = _synthetic_obs(n=600)
        model = fit_slippage_model(obs, ridge_lambda=0.01)
        assert model is not None
        # Signs must match the data-generating process.
        assert model.continuous_coefs["market_cap"] < 0
        assert model.continuous_coefs["dollar_volume"] < 0
        assert model.continuous_coefs["order_notional"] > 0
        assert model.continuous_coefs["volatility"] > 0
        # Magnitudes roughly recovered (loose — ridge + noise).
        assert model.continuous_coefs["order_notional"] == pytest.approx(
            true["order_notional"], abs=2.0
        )

    def test_good_fit_quality(self):
        obs, _, _ = _synthetic_obs(n=600)
        model = fit_slippage_model(obs, ridge_lambda=0.01)
        assert model.r_squared > 0.9
        assert model.n_obs == 600

    def test_trigger_intercepts_ordered(self):
        obs, _, trig_intercept = _synthetic_obs(n=600)
        model = fit_slippage_model(obs, ridge_lambda=0.01)
        # market (20) > vwap (12) > pullback (5) intercepts preserved in order.
        assert (
            model.trigger_intercepts["market"]
            > model.trigger_intercepts["vwap"]
            > model.trigger_intercepts["pullback"]
        )

    def test_drops_unusable_rows(self):
        obs, _, _ = _synthetic_obs(n=100)
        obs.append({"market_cap": 1e9})  # no target → dropped
        obs.append({"slippage_bps": float("nan"), "trigger_type": "x"})  # nan → dropped
        model = fit_slippage_model(obs)
        assert model.n_obs == 100


class TestPredict:
    def _model(self):
        # Hand-built model with realistic, well-scaled coefficients so raw
        # predictions stay positive across the test inputs (the synthetic-fit
        # DGP in TestFit is for sign/quality recovery, not absolute levels).
        return SlippageModel(
            continuous_coefs={
                "market_cap": -0.4,
                "dollar_volume": -0.3,
                "order_notional": 1.2,
                "volatility": 30.0,
            },
            trigger_intercepts={"pullback": 5.0, "vwap": 30.0, "market": 40.0},
            default_intercept=30.0,
            n_obs=600, r_squared=0.95, rmse_bps=0.5,
        )

    def test_monotonic_in_order_size(self):
        m = self._model()
        small = m.predict(market_cap=1e11, dollar_volume=1e9,
                          order_notional=1e3, volatility=0.02, trigger_type="vwap")
        large = m.predict(market_cap=1e11, dollar_volume=1e9,
                          order_notional=5e5, volatility=0.02, trigger_type="vwap")
        assert large > small > 0

    def test_monotonic_in_liquidity(self):
        m = self._model()
        thin = m.predict(market_cap=1e11, dollar_volume=1e6,
                         order_notional=1e5, volatility=0.02, trigger_type="vwap")
        liquid = m.predict(market_cap=1e11, dollar_volume=5e9,
                           order_notional=1e5, volatility=0.02, trigger_type="vwap")
        assert liquid < thin

    def test_monotonic_in_volatility(self):
        m = self._model()
        calm = m.predict(market_cap=1e11, dollar_volume=1e9,
                         order_notional=1e5, volatility=0.005, trigger_type="vwap")
        wild = m.predict(market_cap=1e11, dollar_volume=1e9,
                         order_notional=1e5, volatility=0.06, trigger_type="vwap")
        assert wild > calm

    def test_floored_at_zero(self):
        # A model whose only term is a large negative liquidity coef → the raw
        # linear prediction goes negative and predict() floors it at 0.
        m = SlippageModel(
            continuous_coefs={"market_cap": -5.0, "dollar_volume": 0.0,
                              "order_notional": 0.0, "volatility": 0.0},
            trigger_intercepts={"vwap": 1.0}, default_intercept=1.0,
            n_obs=50, r_squared=0.5, rmse_bps=1.0,
        )
        p = m.predict(market_cap=1e15, dollar_volume=1.0,
                      order_notional=1.0, volatility=0.0001, trigger_type="vwap")
        assert p == 0.0

    def test_unseen_trigger_uses_default(self):
        m = self._model()
        p = m.predict(market_cap=1e11, dollar_volume=1e9,
                      order_notional=1e5, volatility=0.02, trigger_type="never_seen")
        assert math.isfinite(p) and p >= 0


class TestHelpers:
    def test_predict_fallback_when_no_model(self):
        assert predict_slippage_bps(
            None, market_cap=1, dollar_volume=1, order_notional=1,
            volatility=0.01, trigger_type="x",
        ) == DEFAULT_FLAT_SLIPPAGE_BPS

    def test_predict_custom_fallback(self):
        assert predict_slippage_bps(
            None, market_cap=1, dollar_volume=1, order_notional=1,
            volatility=0.01, trigger_type="x", fallback_bps=7.5,
        ) == 7.5

    def test_normalize_trigger(self):
        assert normalize_trigger("  Pullback ") == "pullback"
        assert normalize_trigger(None) == "unspecified"
        assert normalize_trigger("") == "unspecified"

    def test_to_dict_roundtrips_keys(self):
        obs, _, _ = _synthetic_obs(n=100)
        d = fit_slippage_model(obs).to_dict()
        assert {"continuous_coefs", "trigger_intercepts", "default_intercept",
                "n_obs", "r_squared", "rmse_bps"} <= set(d)


class TestBuildObservations:
    def _lookup(self, ticker, date):
        return {"market_cap": 1e11, "dollar_volume": 1e9,
                "order_notional": 1e5, "volatility": 0.02}

    def test_converts_slippage_vs_signal_to_bps(self):
        entries = [{"ticker": "AAA", "date": "2026-06-01",
                    "trigger_type": "vwap", "slippage_vs_signal": 0.001}]
        obs = build_observations_from_trades(entries, self._lookup)
        assert len(obs) == 1
        assert obs[0]["slippage_bps"] == pytest.approx(10.0)  # 0.001 → 10 bps
        assert obs[0]["market_cap"] == 1e11

    def test_prefers_explicit_slippage_bps(self):
        entries = [{"ticker": "AAA", "date": "2026-06-01",
                    "trigger_type": "vwap", "slippage_bps": 7.0,
                    "slippage_vs_signal": 0.001}]
        obs = build_observations_from_trades(entries, self._lookup)
        assert obs[0]["slippage_bps"] == 7.0

    def test_skips_rows_without_features(self):
        entries = [{"ticker": "AAA", "date": "d", "slippage_bps": 5.0}]
        obs = build_observations_from_trades(entries, lambda t, d: None)
        assert obs == []

    def test_skips_rows_without_slippage(self):
        entries = [{"ticker": "AAA", "date": "d", "trigger_type": "vwap"}]
        obs = build_observations_from_trades(entries, self._lookup)
        assert obs == []
