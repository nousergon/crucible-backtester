"""Tests for the W3.3 (config#1993) observe-only 2-horizon double-sort study.

Pins the pure book-construction logic (top-quantile count, intersection
semantics), the zero-cost net==gross invariant, signal recovery, graceful
degradation on missing panels / thin data, and that the pair-selection CSCV-PBO
is wired to the shared engine with date-aligned blocks.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from analysis.double_sort import (
    DEFAULT_PAIRS,
    _intersection_book,
    _top_quantile_book,
    compute_double_sort,
)
from analysis.transaction_cost import TransactionCostModel

_ZERO_COST = TransactionCostModel(half_spread_bps=0.0, impact_coef_bps=0.0,
                                  commission_bps=0.0, min_cost_bps=0.0)


def _dates(n=500):
    return pd.bdate_range("2020-01-01", periods=n)


def _agreeing_panels(n_dates=500, n_good=12, n_bad=48, good_drift=0.0015, seed=0):
    """Good names drift up and are ranked top by BOTH horizons every date →
    the top-quantile intersection is exactly the good set → positive net alpha.
    Returns ``(predictions_by_horizon, price_matrix, spy)`` for horizons 10/21/63/126.
    """
    rng = np.random.default_rng(seed)
    idx = _dates(n_dates)
    good = [f"G{i:02d}" for i in range(n_good)]
    bad = [f"B{i:02d}" for i in range(n_bad)]
    cols = good + bad + ["SPY"]
    prices = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for t in cols:
        drift = good_drift if t in good else 0.0
        prices[t] = 100.0 * np.cumprod(1.0 + drift + rng.normal(0, 0.002, n_dates))
    # every horizon agrees: good names alpha 1.0, bad names 0.0
    panel = {
        d.strftime("%Y-%m-%d"): {**{t: 1.0 for t in good}, **{t: 0.0 for t in bad}}
        for d in idx
    }
    pbh = {h: panel for h in (10, 21, 63, 126)}
    return pbh, prices, prices["SPY"]


class TestTopQuantile:
    def test_count_is_ceil_quantile_times_n(self):
        preds = {f"T{i}": float(i) for i in range(10)}
        row = pd.Series({f"T{i}": 100.0 for i in range(10)})
        book = _top_quantile_book(preds, row, 0.2)
        assert len(book) == math.ceil(0.2 * 10) == 2
        # highest alpha first
        assert book == ["T9", "T8"]

    def test_spy_and_nonfinite_excluded(self):
        preds = {"A": 5.0, "SPY": 9.0, "B": float("nan"), "C": 3.0}
        row = pd.Series({"A": 10.0, "SPY": 10.0, "B": 10.0, "C": 10.0})
        book = _top_quantile_book(preds, row, 1.0)
        assert set(book) == {"A", "C"}  # SPY excluded, NaN-alpha dropped

    def test_unpriced_names_excluded(self):
        preds = {"A": 5.0, "B": 4.0}
        row = pd.Series({"A": 10.0, "B": float("nan")})
        assert _top_quantile_book(preds, row, 1.0) == ["A"]


class TestIntersection:
    def test_intersection_is_overlap_of_top_quantiles(self):
        row = pd.Series({t: 100.0 for t in ["A", "B", "C", "D", "E"]})
        # h1 ranks A,B,C top; h2 ranks B,C,D top (quantile 0.6 → ceil=3 each)
        h1 = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
        h2 = {"D": 5, "C": 4, "B": 3, "A": 2, "E": 1}
        book = _intersection_book(h1, h2, row, 0.6)
        assert set(book) == {"B", "C"}  # A only in h1-top, D only in h2-top

    def test_disjoint_top_quantiles_give_empty_book(self):
        row = pd.Series({t: 100.0 for t in ["A", "B", "C", "D"]})
        h1 = {"A": 9, "B": 8, "C": 1, "D": 0}
        h2 = {"A": 0, "B": 1, "C": 8, "D": 9}
        assert _intersection_book(h1, h2, row, 0.5) == []


class TestZeroCostInvariant:
    def test_double_sort_net_equals_gross_when_cost_zero(self):
        pbh, prices, spy = _agreeing_panels()
        out = compute_double_sort(pbh, prices, spy, cost_model=_ZERO_COST,
                                  pairs=[(21, 63)])
        ds = out["pairs"]["21x63"]["double_sort"]
        assert ds["status"] == "ok"
        assert ds["net_alpha_ann"] == ds["gross_alpha_ann"]
        assert ds["cost_drag_bps_ann"] == 0.0


class TestSignalRecovered:
    def test_agreeing_signal_gives_positive_net_alpha(self):
        pbh, prices, spy = _agreeing_panels(good_drift=0.0020)
        out = compute_double_sort(pbh, prices, spy,
                                  cost_model=TransactionCostModel(),
                                  pairs=DEFAULT_PAIRS)
        assert out["status"] == "ok"
        for pid in ("10x21", "21x63", "21x126"):
            ds = out["pairs"][pid]["double_sort"]
            assert ds["status"] == "ok", pid
            assert ds["gross_alpha_ann"] > 0, pid
            assert ds["net_alpha_ann"] > 0, pid                 # cost doesn't erase it
            assert ds["net_alpha_ann"] <= ds["gross_alpha_ann"] # cost is a drag

    def test_baselines_are_scored(self):
        pbh, prices, spy = _agreeing_panels()
        out = compute_double_sort(pbh, prices, spy, pairs=[(21, 63)])
        entry = out["pairs"]["21x63"]
        assert entry["baseline_h1"]["status"] == "ok"
        assert entry["baseline_h1"]["forward_days"] == 21
        assert entry["baseline_h2"]["forward_days"] == 63
        assert "net_alpha_uplift_vs_best_baseline" in entry


class TestSelectionPBO:
    def test_selection_pbo_present_and_bounded(self):
        pbh, prices, spy = _agreeing_panels()
        out = compute_double_sort(pbh, prices, spy, pairs=DEFAULT_PAIRS)
        pbo = out["selection_pbo"]
        # 3 pairs, long panel → PBO computed (not insufficient)
        assert pbo.get("status") == "ok"
        assert 0.0 <= pbo["pbo"] <= 1.0

    def test_heavy_period_arrays_stripped_from_payload(self):
        pbh, prices, spy = _agreeing_panels()
        out = compute_double_sort(pbh, prices, spy, pairs=[(21, 63)])
        ds = out["pairs"]["21x63"]["double_sort"]
        assert "net_periods" not in ds
        assert "period_start_dates" not in ds


class TestRobustness:
    def test_missing_horizon_panel_is_graceful(self):
        pbh, prices, spy = _agreeing_panels()
        del pbh[126]
        out = compute_double_sort(pbh, prices, spy, pairs=[(21, 126)])
        assert out["pairs"]["21x126"]["status"] == "missing_horizon_panel"
        assert 126 in out["pairs"]["21x126"]["missing"]

    def test_thin_data_returns_status_not_crash(self):
        pbh, prices, spy = _agreeing_panels(n_dates=40)
        out = compute_double_sort(pbh, prices, spy, pairs=[(21, 126)])
        ds = out["pairs"]["21x126"]["double_sort"]
        assert ds["status"] in ("insufficient_rebalances", "insufficient_periods")
        # PBO cannot be formed from a single thin pair
        assert out["selection_pbo"]["status"] == "insufficient"
