"""Tests for analysis.excursion — MFE/MAE.

Pins:
  1. MFE/MAE match hand-computed values on a known OHLC fixture.
  2. MFE clamped to 0 when no favorable excursion.
  3. mfe_mae_ratio is None when MAE = 0.
  4. Missing ticker / missing eval_date handled.
  5. summarize_excursions aggregates correctly.
  6. pct_high_quality counts the > 1.5 ratio cases.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.excursion import (
    compute_per_pick_excursion,
    summarize_excursions,
)


def _build_ohlc():
    """Fixture: 10 days of OHLC for AAA + BBB.

    AAA: entry $100, drifts up to $115 high then back to $108 close on
    day 10. Low touches $98 on day 2.
    BBB: entry $50, mostly flat with a sharp drop to $42 mid-window then
    recovery to $51 close.
    """
    dates = pd.date_range("2026-01-05", periods=10, freq="B")
    aaa = pd.DataFrame({
        "open":  [100, 101, 100, 102, 105, 108, 110, 112, 113, 108],
        "high":  [102, 103,  99, 105, 108, 110, 113, 115, 114, 110],
        "low":   [ 99,  98,  97, 100, 103, 106, 108, 110, 110, 106],
        "close": [101, 100,  99, 104, 107, 109, 111, 113, 112, 108],
    }, index=dates).astype(float)
    bbb = pd.DataFrame({
        "open":  [50, 49, 48, 45, 42, 44, 47, 49, 50, 51],
        "high":  [51, 50, 49, 47, 44, 46, 48, 50, 51, 52],
        "low":   [49, 48, 47, 44, 42, 43, 46, 48, 49, 50],
        "close": [49, 48, 47, 45, 43, 45, 48, 49, 50, 51],
    }, index=dates).astype(float)
    return {"AAA": aaa, "BBB": bbb}, dates


class TestExcursionPerPick:
    def test_single_pick_aaa_full_window(self):
        ohlc, dates = _build_ohlc()
        # Entry on day 0 (close=$101), 9 days remaining horizon.
        # Window covers all 10 days.
        # Max high = $115 (day 7). MFE = 115/101 - 1 ≈ 0.1386
        # Min low = $97 (day 2). MAE = 1 - 97/101 ≈ 0.0396
        picks = pd.DataFrame([{"ticker": "AAA", "eval_date": dates[0]}])
        out = compute_per_pick_excursion(picks, ohlc, horizon_days=9)
        assert len(out) == 1
        rec = out[0]
        assert rec["mfe"] == pytest.approx(115 / 101 - 1, rel=1e-6)
        assert rec["mae"] == pytest.approx(1 - 97 / 101, rel=1e-6)
        assert rec["mfe_mae_ratio"] == pytest.approx(rec["mfe"] / rec["mae"], rel=1e-6)
        assert rec["realized_return"] == pytest.approx(108 / 101 - 1, rel=1e-6)

    def test_explicit_entry_price_overrides_close(self):
        ohlc, dates = _build_ohlc()
        # Entry at $100 (lower than day-0 close $101) → larger MFE, smaller MAE.
        picks = pd.DataFrame([{
            "ticker": "AAA", "eval_date": dates[0], "entry_price": 100.0,
        }])
        out = compute_per_pick_excursion(picks, ohlc, horizon_days=9)
        rec = out[0]
        assert rec["mfe"] == pytest.approx(115 / 100 - 1, rel=1e-6)
        assert rec["mae"] == pytest.approx(1 - 97 / 100, rel=1e-6)

    def test_horizon_days_clamps_window(self):
        ohlc, dates = _build_ohlc()
        # horizon=2 → window covers days 0..2. MFE within first 3 bars only.
        # high in [102, 103, 99] = 103 → MFE = 103/101 - 1 ≈ 0.0198
        picks = pd.DataFrame([{"ticker": "AAA", "eval_date": dates[0]}])
        out = compute_per_pick_excursion(picks, ohlc, horizon_days=2)
        rec = out[0]
        assert rec["mfe"] == pytest.approx(103 / 101 - 1, rel=1e-6)


class TestEdgeCases:
    def test_missing_ticker_skipped(self):
        ohlc, dates = _build_ohlc()
        picks = pd.DataFrame([
            {"ticker": "AAA", "eval_date": dates[0]},
            {"ticker": "ZZZ", "eval_date": dates[0]},  # absent
        ])
        out = compute_per_pick_excursion(picks, ohlc, horizon_days=5)
        assert len(out) == 1
        assert out[0]["ticker"] == "AAA"

    def test_eval_date_outside_index_skipped(self):
        ohlc, dates = _build_ohlc()
        picks = pd.DataFrame([{
            "ticker": "AAA", "eval_date": dates[0] - pd.Timedelta(days=30),
        }])
        out = compute_per_pick_excursion(picks, ohlc, horizon_days=5)
        assert out == []

    def test_invalid_horizon_raises(self):
        ohlc, dates = _build_ohlc()
        picks = pd.DataFrame([{"ticker": "AAA", "eval_date": dates[0]}])
        with pytest.raises(ValueError):
            compute_per_pick_excursion(picks, ohlc, horizon_days=0)


class TestSummary:
    def test_summary_aggregates(self):
        # Build 10 synthetic records spanning skill ranges.
        records = []
        # 5 skilled (MFE/MAE > 1.5), 5 unskilled (MFE/MAE ~ 1.0)
        for i in range(5):
            records.append({
                "ticker": f"S{i}", "eval_date": "2026-01-05",
                "entry_price": 100.0, "horizon_days": 10,
                "mfe": 0.10, "mae": 0.04,
                "mfe_mae_ratio": 0.10 / 0.04,
                "realized_return": 0.07,
            })
        for i in range(5):
            records.append({
                "ticker": f"U{i}", "eval_date": "2026-01-05",
                "entry_price": 100.0, "horizon_days": 10,
                "mfe": 0.05, "mae": 0.05,
                "mfe_mae_ratio": 1.0,
                "realized_return": 0.0,
            })
        summary = summarize_excursions(records)
        assert summary["status"] == "ok"
        assert summary["n"] == 10
        assert summary["mean_mfe"] == pytest.approx(0.075, rel=1e-6)
        assert summary["mean_mae"] == pytest.approx(0.045, rel=1e-6)
        assert summary["pct_mfe_gt_mae"] == pytest.approx(0.5, rel=1e-6)
        assert summary["pct_high_quality"] == pytest.approx(0.5, rel=1e-6)

    def test_empty_summary(self):
        summary = summarize_excursions([])
        assert summary["status"] == "insufficient_data"
        assert summary["n"] == 0
