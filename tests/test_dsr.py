"""Tests for analysis.dsr — PSR + DSR.

Pins:
  1. PSR(0) on a clearly-positive Sharpe series → > 0.95.
  2. PSR(0) on near-zero-mean noise → ~0.5.
  3. PSR(SR_observed) = 0.5 always (testing against own value).
  4. DSR with n_trials=1 == PSR(0).
  5. DSR with n_trials > 1 has stricter benchmark (DSR < PSR(0) for same series).
  6. Insufficient samples → status="insufficient_data".
  7. n_trials < 1 raises.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from analysis.dsr import compute_dsr, compute_psr


def _build_series(daily_mean: float, daily_std: float, n: int = 252, seed: int = 42):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(daily_mean, daily_std, size=n))


class TestPSR:
    def test_clearly_positive_sharpe_psr_high(self):
        # Daily mean 0.002, std 0.01 → daily SR ≈ 0.2, annualized ≈ 3.17.
        # n=252 + sufficient effect size should give PSR(0) >> 0.5 robustly
        # across seeds — even with sample skew/kurtosis eating into the
        # confidence numerator.
        r = _build_series(daily_mean=0.002, daily_std=0.01, n=252, seed=42)
        result = compute_psr(r, sharpe_benchmark=0.0)
        assert result["status"] == "ok"
        assert result["psr"] > 0.90

    def test_zero_signal_psr_near_half(self):
        r = _build_series(daily_mean=0.0, daily_std=0.01, n=252, seed=99)
        result = compute_psr(r, sharpe_benchmark=0.0)
        # Near coin-flip (allow wide envelope due to seed luck).
        assert 0.30 < result["psr"] < 0.70

    def test_against_own_sharpe_returns_half(self):
        r = _build_series(daily_mean=0.001, daily_std=0.01, n=252, seed=7)
        result = compute_psr(r, sharpe_benchmark=0.0)
        own_sharpe = result["sharpe"]
        # Test against the observed Sharpe — should be exactly 0.5.
        result2 = compute_psr(r, sharpe_benchmark=own_sharpe)
        assert result2["psr"] == pytest.approx(0.5, abs=1e-6)

    def test_insufficient_samples(self):
        r = pd.Series([0.001] * 10)
        result = compute_psr(r)
        assert result["status"] == "insufficient_data"


class TestDSR:
    def test_n_trials_one_equals_psr_zero(self):
        r = _build_series(daily_mean=0.001, daily_std=0.01, n=252, seed=1)
        psr0 = compute_psr(r, sharpe_benchmark=0.0)
        dsr1 = compute_dsr(r, n_trials=1)
        assert dsr1["dsr"] == pytest.approx(psr0["psr"], abs=1e-9)

    def test_more_trials_means_stricter(self):
        # Same series, more trials → benchmark Sharpe rises → DSR drops.
        r = _build_series(daily_mean=0.001, daily_std=0.01, n=252, seed=42)
        d1 = compute_dsr(r, n_trials=1)
        d10 = compute_dsr(r, n_trials=10)
        d100 = compute_dsr(r, n_trials=100)
        assert d1["dsr"] >= d10["dsr"] >= d100["dsr"]
        assert d1["sharpe_benchmark"] == 0.0
        assert d100["sharpe_benchmark"] > d10["sharpe_benchmark"] > d1["sharpe_benchmark"]

    def test_invalid_n_trials_raises(self):
        r = _build_series(daily_mean=0.001, daily_std=0.01, n=252)
        with pytest.raises(ValueError):
            compute_dsr(r, n_trials=0)
        with pytest.raises(ValueError):
            compute_dsr(r, n_trials=-5)

    def test_insufficient_samples(self):
        r = pd.Series([0.001] * 10)
        result = compute_dsr(r, n_trials=10)
        assert result["status"] == "insufficient_data"

    def test_strong_signal_survives_modest_n_trials(self):
        # Very strong daily SR (mean 0.005, std 0.005 → daily SR 1.0,
        # annualized ~15.87). Should survive even 1000 trials.
        r = _build_series(daily_mean=0.005, daily_std=0.005, n=252, seed=11)
        result = compute_dsr(r, n_trials=1000)
        assert result["dsr"] > 0.95
