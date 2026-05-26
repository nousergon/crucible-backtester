"""B.2 — Predictive-interval calibration diagnostics.

Plan: alpha-engine-docs/private/optimizer-sota-upgrades-260526.md §B.2

Tests the new ``predictive_interval_calibration`` analysis primitive that
validates the predictor's BayesianRidge posterior std (B.1) against
realized outcomes. The plan's load-bearing gate: 90% predicted CI must
cover 88-92% of realized values — verified on a well-calibrated
synthetic panel here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from analysis.predictive_interval_calibration import (
    _PRIMARY_GATE_LEVEL,
    _PRIMARY_GATE_TOLERANCE,
    _normal_quantile,
    compute_predictive_interval_calibration,
)


class TestNormalQuantile:
    """The inverse-normal-CDF helper is load-bearing for coverage
    thresholds; verify accuracy against standard tabulated values."""

    @pytest.mark.parametrize("p, expected, tol", [
        (0.5, 0.0, 1e-6),
        (0.95, 1.6448536, 1e-4),
        (0.975, 1.959964, 1e-4),
        (0.99, 2.3263479, 1e-4),
        (0.05, -1.6448536, 1e-4),
        (0.01, -2.3263479, 1e-4),
    ])
    def test_known_quantiles(self, p, expected, tol):
        assert abs(_normal_quantile(p) - expected) < tol

    def test_domain_boundary_raises(self):
        with pytest.raises(ValueError):
            _normal_quantile(0.0)
        with pytest.raises(ValueError):
            _normal_quantile(1.0)


class TestComputePredictiveIntervalCalibration:
    """Behavior on synthetic predict-vs-realize panels."""

    def _well_calibrated_panel(self, n=1000, seed=42, sigma=0.02):
        """y_i ~ N(μ_i, σ²) where the predictor emits the SAME (μ, σ).
        Empirical coverage at any level should match the nominal level."""
        rng = np.random.default_rng(seed)
        mu = rng.normal(0, 0.01, n)
        std = np.full(n, sigma)
        realized = mu + rng.normal(0, sigma, n)
        return mu, std, realized

    def test_well_calibrated_panel_passes_90_gate(self):
        """The plan's primary acceptance gate: 90% CI covers 88-92%."""
        mu, std, y = self._well_calibrated_panel(n=2000, seed=0)
        result = compute_predictive_interval_calibration(mu, std, y)
        assert result["status"] == "ok"
        assert result["primary_gate_passes"] is True
        # Empirical coverage at 90% should be within tolerance band
        rec_90 = next(r for r in result["coverage"] if r["confidence"] == 0.90)
        lo = _PRIMARY_GATE_LEVEL - _PRIMARY_GATE_TOLERANCE
        hi = _PRIMARY_GATE_LEVEL + _PRIMARY_GATE_TOLERANCE
        assert lo <= rec_90["empirical"] <= hi, (
            f"Empirical 90% coverage {rec_90['empirical']} outside [{lo}, {hi}]"
        )

    def test_well_calibrated_panel_yields_pit_zscores_near_n01(self):
        """Under calibration, (y - μ) / σ ~ N(0, 1) → mean ≈ 0, std ≈ 1."""
        mu, std, y = self._well_calibrated_panel(n=5000, seed=1)
        result = compute_predictive_interval_calibration(mu, std, y)
        # On n=5000 the sample mean is within ~2 SE of 0; SE ≈ 1/√5000 ≈ 0.014
        assert abs(result["pit_mean"]) < 0.05
        # Sample std of N(0,1) is approximately 1 ± 1/√(2(n-1)) ≈ 0.01
        assert abs(result["pit_std"] - 1.0) < 0.05

    def test_overconfident_predictor_fails_gate(self):
        """If the predictor's σ is HALF the true noise scale, intervals
        are too narrow → empirical coverage at 90% < 88%."""
        rng = np.random.default_rng(7)
        n = 2000
        true_sigma = 0.04
        mu = rng.normal(0, 0.01, n)
        predicted_std = np.full(n, true_sigma / 2.0)  # halved → overconfident
        realized = mu + rng.normal(0, true_sigma, n)
        result = compute_predictive_interval_calibration(mu, predicted_std, realized)
        assert result["status"] == "ok"
        assert result["primary_gate_passes"] is False
        rec_90 = next(r for r in result["coverage"] if r["confidence"] == 0.90)
        assert rec_90["empirical"] < 0.88
        # PIT std should be ~2 (the doubled true scale relative to predicted)
        assert result["pit_std"] > 1.5
        assert result["quality"] == "poor"

    def test_underconfident_predictor_fails_gate(self):
        """If σ is DOUBLE the true noise, intervals are too wide → coverage
        > 92% at the 90% nominal level."""
        rng = np.random.default_rng(13)
        n = 2000
        true_sigma = 0.02
        mu = rng.normal(0, 0.01, n)
        predicted_std = np.full(n, true_sigma * 2.0)
        realized = mu + rng.normal(0, true_sigma, n)
        result = compute_predictive_interval_calibration(mu, predicted_std, realized)
        assert result["status"] == "ok"
        assert result["primary_gate_passes"] is False
        rec_90 = next(r for r in result["coverage"] if r["confidence"] == 0.90)
        assert rec_90["empirical"] > 0.92
        # PIT std should be ~0.5 (true scale half of predicted)
        assert result["pit_std"] < 0.7

    def test_crps_lower_for_more_accurate_predictor(self):
        """A predictor with the right σ gets lower CRPS than one with
        σ inflated 5×. CRPS is the strictly-proper-scoring-rule check."""
        rng = np.random.default_rng(21)
        n = 1000
        true_sigma = 0.02
        mu = rng.normal(0, 0.01, n)
        realized = mu + rng.normal(0, true_sigma, n)

        correct = compute_predictive_interval_calibration(
            mu, np.full(n, true_sigma), realized,
        )
        bloated = compute_predictive_interval_calibration(
            mu, np.full(n, true_sigma * 5.0), realized,
        )
        assert correct["crps_mean"] < bloated["crps_mean"]

    def test_insufficient_data(self):
        """Fewer than min_n valid pairs returns status=insufficient_data."""
        result = compute_predictive_interval_calibration(
            np.array([0.0, 0.01, 0.02]),
            np.array([0.01, 0.01, 0.01]),
            np.array([0.0, 0.01, 0.02]),
        )
        assert result["status"] == "insufficient_data"
        assert result["n"] == 3

    def test_negative_std_raises(self):
        """Negative predicted_std is an upstream contract violation —
        raise loud per no-silent-fails."""
        mu = np.zeros(50)
        bad_std = np.full(50, 0.01)
        bad_std[0] = -0.005
        realized = np.zeros(50)
        with pytest.raises(ValueError, match="non-positive"):
            compute_predictive_interval_calibration(mu, bad_std, realized)

    def test_zero_std_raises(self):
        """Zero std is also invalid (would divide by zero for PIT)."""
        mu = np.zeros(50)
        bad_std = np.full(50, 0.01)
        bad_std[5] = 0.0
        realized = np.zeros(50)
        with pytest.raises(ValueError, match="non-positive"):
            compute_predictive_interval_calibration(mu, bad_std, realized)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            compute_predictive_interval_calibration(
                np.zeros(10), np.ones(10), np.zeros(11),
            )

    def test_nan_pairs_are_dropped(self):
        """NaN in any of (μ, σ, y) drops that row — calibration of valid
        rows is preserved without crashing on partial data."""
        rng = np.random.default_rng(0)
        n = 500
        mu = rng.normal(0, 0.01, n)
        std = np.full(n, 0.02)
        realized = mu + rng.normal(0, 0.02, n)
        # Inject NaN in random spots
        mu[10:15] = np.nan
        realized[100:110] = np.nan
        result = compute_predictive_interval_calibration(mu, std, realized)
        assert result["status"] == "ok"
        # ~15 dropped of 500 → 485 valid
        assert 480 <= result["n"] <= 490

    def test_heteroskedastic_well_calibrated(self):
        """Realistic case: per-prediction σ varies (heteroskedastic).
        If σ_i correctly tracks the true noise scale per-observation,
        coverage at named levels still holds."""
        rng = np.random.default_rng(99)
        n = 2000
        # Per-obs true noise scale varies 0.005 to 0.05
        true_sigma = rng.uniform(0.005, 0.05, n)
        mu = rng.normal(0, 0.01, n)
        predicted_std = true_sigma.copy()  # predictor knows the truth
        realized = mu + rng.normal(0, true_sigma)
        result = compute_predictive_interval_calibration(mu, predicted_std, realized)
        assert result["status"] == "ok"
        assert result["primary_gate_passes"] is True
        assert result["quality"] in ("good", "acceptable")

    def test_quality_grade_correlates_with_max_deviation(self):
        """Quality flag tracks the worst-coverage-level deviation:
        well-calibrated → good; bloated 5× → poor."""
        rng = np.random.default_rng(101)
        n = 1000
        true_sigma = 0.02
        mu = rng.normal(0, 0.01, n)
        realized = mu + rng.normal(0, true_sigma, n)
        good = compute_predictive_interval_calibration(
            mu, np.full(n, true_sigma), realized,
        )
        poor = compute_predictive_interval_calibration(
            mu, np.full(n, true_sigma * 5.0), realized,
        )
        assert good["quality"] == "good"
        assert poor["quality"] == "poor"
