"""Tests for analysis.expectancy.

Pins:
  1. expectancy = hit_rate * avg_win - (1-hit_rate) * avg_loss on hand fixtures.
  2. Win/loss ratio + R-multiple expectancy.
  3. All-wins / all-losses edge cases marked status="no_losses" / "no_wins".
  4. Insufficient samples → status="insufficient_data".
  5. Threshold parameter shifts the win/loss boundary correctly.
  6. compute_expectancy_by_group stratifies correctly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.expectancy import compute_expectancy, compute_expectancy_by_group


class TestCoreExpectancy:
    def test_balanced_winners_and_losers(self):
        # 10 trades: 6 wins of +5%, 4 losses of -3%.
        # hit_rate = 0.6, avg_win = 0.05, avg_loss = 0.03
        # expectancy = 0.6 * 0.05 - 0.4 * 0.03 = 0.03 - 0.012 = 0.018
        # win_loss_ratio = 0.05 / 0.03 = 1.667
        returns = np.array([0.05] * 6 + [-0.03] * 4)
        result = compute_expectancy(returns)
        assert result["status"] == "ok"
        assert result["hit_rate"] == pytest.approx(0.6)
        assert result["avg_win"] == pytest.approx(0.05)
        assert result["avg_loss"] == pytest.approx(0.03)
        assert result["win_loss_ratio"] == pytest.approx(0.05 / 0.03, rel=1e-6)
        assert result["expectancy"] == pytest.approx(0.018, rel=1e-6)
        assert result["expectancy_per_unit_loss"] == pytest.approx(0.018 / 0.03, rel=1e-6)

    def test_skilled_convexity(self):
        # 30% hit rate but big winners — classic convexity skill.
        # hit_rate = 0.3, avg_win = 0.20, avg_loss = 0.04
        # expectancy = 0.3 * 0.20 - 0.7 * 0.04 = 0.06 - 0.028 = 0.032
        # W/L ratio = 5.0
        returns = np.array([0.20] * 3 + [-0.04] * 7)
        result = compute_expectancy(returns)
        assert result["expectancy"] == pytest.approx(0.032, rel=1e-6)
        assert result["win_loss_ratio"] == pytest.approx(5.0, rel=1e-6)
        # R-multiple: expectancy is +0.8 per unit of risk taken.
        assert result["expectancy_per_unit_loss"] == pytest.approx(0.032 / 0.04, rel=1e-6)


class TestEdgeCases:
    def test_all_winners_status(self):
        returns = np.array([0.01] * 20)
        result = compute_expectancy(returns)
        assert result["status"] == "no_losses"
        assert result["hit_rate"] == 1.0

    def test_all_losers_status(self):
        returns = np.array([-0.01] * 20)
        result = compute_expectancy(returns)
        assert result["status"] == "no_wins"
        assert result["hit_rate"] == 0.0

    def test_insufficient_samples(self):
        result = compute_expectancy(np.array([0.01, -0.01]), min_samples=10)
        assert result["status"] == "insufficient_data"
        assert result["n"] == 2

    def test_drops_nan(self):
        returns = np.array([np.nan] * 5 + [0.05] * 6 + [-0.03] * 4)
        result = compute_expectancy(returns)
        assert result["status"] == "ok"
        assert result["n"] == 10  # NaN dropped


class TestThreshold:
    def test_threshold_shifts_boundary(self):
        # 10 trades around +1%. threshold=0 → all winners.
        # threshold=0.02 → all losers (none beat 2%).
        returns = np.array([0.01] * 10)
        r0 = compute_expectancy(returns, threshold=0.0)
        r2 = compute_expectancy(returns, threshold=0.02)
        assert r0["status"] == "no_losses"  # all > 0
        assert r2["status"] == "no_wins"    # none > 0.02


class TestByGroup:
    def test_stratifies_correctly(self):
        df = pd.DataFrame({
            "team": ["a"] * 10 + ["b"] * 10,
            "return": [0.05] * 6 + [-0.03] * 4 + [0.02] * 7 + [-0.05] * 3,
        })
        out = compute_expectancy_by_group(df, "return", "team", min_samples=5)
        assert set(out.keys()) == {"a", "b"}
        assert out["a"]["expectancy"] == pytest.approx(0.018, rel=1e-6)
        # b: hit=0.7, win=0.02, loss=0.05; expectancy = 0.014 - 0.015 = -0.001
        assert out["b"]["expectancy"] == pytest.approx(-0.001, rel=1e-6)

    def test_missing_column_raises(self):
        df = pd.DataFrame({"return": [0.01]})
        with pytest.raises(KeyError):
            compute_expectancy_by_group(df, "return", "missing")
