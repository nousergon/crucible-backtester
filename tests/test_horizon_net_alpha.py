"""Tests for the W3.4 (L4469) turnover-adjusted net-alpha-per-horizon study.

Pins: zero-cost ⇒ net==gross; a real ranking signal ⇒ positive net alpha;
turnover (and therefore cost drag) scales ~1/h (shorter horizons rebalance more,
the whole point of the net-of-cost horizon judge); graceful status on thin data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.horizon_net_alpha import compute_horizon_net_alpha
from analysis.transaction_cost import TransactionCostModel

_ZERO_COST = TransactionCostModel(half_spread_bps=0.0, impact_coef_bps=0.0,
                                  commission_bps=0.0, min_cost_bps=0.0)


def _dates(n=400):
    return pd.bdate_range("2020-01-01", periods=n)


def _static_signal_panel(n_dates=400, n_good=20, n_bad=30, good_drift=0.0010, seed=0):
    """Good names rise steadily and are always predicted top; bad names flat.
    A top-20 book selects the good set → positive gross alpha vs a flat SPY."""
    rng = np.random.default_rng(seed)
    idx = _dates(n_dates)
    good = [f"G{i:02d}" for i in range(n_good)]
    bad = [f"B{i:02d}" for i in range(n_bad)]
    cols = good + bad + ["SPY"]
    prices = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for t in cols:
        drift = good_drift if t in good else 0.0
        noise = rng.normal(0, 0.002, n_dates)
        prices[t] = 100.0 * np.cumprod(1.0 + drift + noise)
    spy = prices["SPY"]
    preds = {
        d.strftime("%Y-%m-%d"): {**{t: 1.0 for t in good}, **{t: 0.0 for t in bad}}
        for d in idx
    }
    return preds, prices, spy


def _noise_panel(n_dates=400, n_names=60, seed=1):
    """Random predictions every date → the top-N book fully rotates each
    rebalance → turnover ~constant per rebalance → turnover_ann ~ 1/h."""
    rng = np.random.default_rng(seed)
    idx = _dates(n_dates)
    names = [f"N{i:02d}" for i in range(n_names)]
    cols = names + ["SPY"]
    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rng.normal(0, 0.003, (n_dates, len(cols))), axis=0),
        index=idx, columns=cols,
    )
    preds = {
        d.strftime("%Y-%m-%d"): {t: float(rng.normal()) for t in names}
        for d in idx
    }
    return preds, prices, prices["SPY"]


class TestZeroCostInvariant:
    def test_net_equals_gross_when_cost_zero(self):
        preds, prices, spy = _static_signal_panel()
        out = compute_horizon_net_alpha(preds, prices, spy, cost_model=_ZERO_COST,
                                        horizons=[5, 21], top_n=20)
        for h in ("5d", "21d"):
            e = out["horizons"][h]
            assert e["status"] == "ok"
            assert e["net_alpha_ann"] == e["gross_alpha_ann"]
            assert e["cost_drag_bps_ann"] == 0.0


class TestSignalRecovered:
    def test_positive_net_alpha_for_real_signal(self):
        preds, prices, spy = _static_signal_panel(good_drift=0.0015)
        out = compute_horizon_net_alpha(preds, prices, spy,
                                        cost_model=TransactionCostModel(),
                                        horizons=[21], top_n=20)
        e = out["horizons"]["21d"]
        assert e["status"] == "ok"
        assert e["gross_alpha_ann"] > 0
        assert e["net_alpha_ann"] > 0          # cost doesn't erase a strong signal
        assert e["net_alpha_ann"] <= e["gross_alpha_ann"]  # cost is a drag
        assert out["net_alpha_max_horizon"] == "21d"


class TestTurnoverScalesWithHorizon:
    def test_shorter_horizon_has_more_turnover_and_cost(self):
        preds, prices, spy = _noise_panel()
        out = compute_horizon_net_alpha(preds, prices, spy,
                                        cost_model=TransactionCostModel(),
                                        horizons=[5, 60], top_n=20)
        short, long = out["horizons"]["5d"], out["horizons"]["60d"]
        assert short["turnover_oneway_ann"] > long["turnover_oneway_ann"]
        assert short["cost_drag_bps_ann"] > long["cost_drag_bps_ann"]


class TestRobustness:
    def test_thin_data_returns_status_not_crash(self):
        preds, prices, spy = _static_signal_panel(n_dates=30)
        out = compute_horizon_net_alpha(preds, prices, spy, horizons=[90], top_n=20)
        assert out["horizons"]["90d"]["status"] in (
            "insufficient_rebalances", "insufficient_periods")

    def test_adv_coverage_reported(self):
        preds, prices, spy = _static_signal_panel()
        adv = {t: 5e8 for t in prices.columns}
        out = compute_horizon_net_alpha(preds, prices, spy, horizons=[21],
                                        top_n=20, adv_dollar_by_ticker=adv)
        assert out["cost_model"]["adv_coverage_names"] == len(prices.columns)
