"""Tests for analysis.information_coefficient.

Pins:
  1. Perfect rank correlation → IC = 1.0 (exact).
  2. Perfect inverse correlation → IC = -1.0.
  3. Random / no signal → IC near 0.
  4. Constant conviction → status="no_variance".
  5. Mismatched lengths → ValueError.
  6. compute_ic_by_bucket stratifies correctly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.information_coefficient import compute_ic, compute_ic_by_bucket


class TestCoreIC:
    def test_perfect_rank_correlation_ic_one(self):
        # Conviction matches return rank exactly → Spearman = 1.
        conviction = np.arange(50, dtype=float)
        forward_return = conviction * 0.001 + 0.005  # monotonic transform
        result = compute_ic(conviction, forward_return)
        assert result["status"] == "ok"
        assert result["ic"] == pytest.approx(1.0, abs=1e-9)

    def test_perfect_inverse_correlation(self):
        conviction = np.arange(50, dtype=float)
        forward_return = -conviction
        result = compute_ic(conviction, forward_return)
        assert result["ic"] == pytest.approx(-1.0, abs=1e-9)

    def test_zero_signal_ic_near_zero(self):
        rng = np.random.default_rng(42)
        conviction = rng.normal(0, 1, size=200)
        forward_return = rng.normal(0, 1, size=200)  # independent
        result = compute_ic(conviction, forward_return)
        # n=200 random pairs: |IC| should be < ~0.2 with high prob.
        assert result["status"] == "ok"
        assert abs(result["ic"]) < 0.2


class TestEdgeCases:
    def test_constant_conviction_no_variance(self):
        conviction = np.ones(50)
        forward_return = np.arange(50, dtype=float)
        result = compute_ic(conviction, forward_return)
        assert result["status"] == "no_variance"
        assert result["ic"] == 0.0
        assert result["n_buckets"] == 1

    def test_insufficient_samples(self):
        conviction = np.arange(10, dtype=float)
        forward_return = conviction
        result = compute_ic(conviction, forward_return, min_samples=20)
        assert result["status"] == "insufficient_data"
        assert result["n"] == 10

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="must be same length"):
            compute_ic(np.arange(10), np.arange(20))

    def test_drops_nan(self):
        conviction = np.array([1.0, 2.0, 3.0, np.nan, 5.0] * 10)
        forward_return = np.array([0.01, 0.02, 0.03, 0.04, 0.05] * 10)
        result = compute_ic(conviction, forward_return)
        # 40 valid pairs after NaN drop.
        assert result["n"] == 40
        assert result["status"] == "ok"


class TestByBucket:
    def test_stratification(self):
        # Two sectors: tech has perfect rank corr; health has no signal.
        rng = np.random.default_rng(0)
        rows = []
        for sector, has_signal in [("tech", True), ("health", False)]:
            n = 30
            conviction = np.arange(n, dtype=float)
            if has_signal:
                forward = conviction * 0.001
            else:
                forward = rng.normal(0, 1, size=n)
            for c, r in zip(conviction, forward):
                rows.append({"sector": sector, "conviction": c, "return": r})
        df = pd.DataFrame(rows)
        out = compute_ic_by_bucket(df, "conviction", "return", "sector")
        assert set(out.keys()) == {"tech", "health"}
        assert out["tech"]["ic"] == pytest.approx(1.0, abs=1e-9)
        assert abs(out["health"]["ic"]) < 0.5  # noise

    def test_missing_column_raises(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(KeyError):
            compute_ic_by_bucket(df, "a", "b", "c")
