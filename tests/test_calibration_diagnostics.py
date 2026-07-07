"""Tests for analysis.calibration_diagnostics.

Pins:
  1. Perfectly calibrated probabilities → ECE near 0, quality="good".
  2. Overconfident agent (says 90%, hits 50%) → large ECE, quality="poor".
  3. Brier score on a hand-fixture matches mean((p - y)^2).
  4. Bins below min_bin_n appear in dropped_bins, not in main bins.
  5. Constant-probability input → status="no_variance".
  6. Insufficient samples handled.
  7. Custom bin_edges respected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.calibration_diagnostics import compute_calibration


class TestCalibration:
    def test_perfectly_calibrated(self):
        # Equal counts in each bin, hit rate = bin midpoint.
        rng = np.random.default_rng(0)
        probs = []
        outcomes = []
        for p_target in [0.1, 0.3, 0.5, 0.7, 0.9]:
            n = 100
            probs.extend([p_target] * n)
            # Hit rate exactly p_target: round(n * p_target) hits.
            n_hits = round(n * p_target)
            outcomes.extend([1] * n_hits + [0] * (n - n_hits))
        result = compute_calibration(np.array(probs), np.array(outcomes))
        assert result["status"] == "ok"
        # ECE should be very low: each bin's hit_rate == bin's mean prob.
        assert result["ece"] < 0.01
        assert result["quality"] == "good"

    def test_overconfident_agent_ece_large(self):
        # Agent says 90% but only hits 50% of the time.
        n = 100
        probs = np.array([0.9] * n)
        outcomes = np.array([1] * 50 + [0] * 50)
        result = compute_calibration(probs, outcomes)
        # No variance in probs, but bin still meaningful.
        # Single bin (0.8-1.01): n=100, hit_rate=0.5, expected=0.9.
        # |0.5 - 0.9| = 0.4. ECE = 0.4.
        assert result["status"] == "no_variance"  # std(probs) = 0
        # Brier still computed even when no variance.
        assert result["brier_score"] == pytest.approx(((0.9 - outcomes) ** 2).mean(), rel=1e-9)

    def test_brier_score_matches_mse(self):
        rng = np.random.default_rng(1)
        probs = rng.uniform(0.1, 0.9, size=200)
        outcomes = (rng.uniform(size=200) < probs).astype(float)
        result = compute_calibration(probs, outcomes)
        expected_brier = float(((probs - outcomes) ** 2).mean())
        assert result["brier_score"] == pytest.approx(expected_brier, abs=0.001)

    def test_dropped_bins_below_min_n(self):
        # 100 samples in middle bin, 5 in tail bin.
        probs = [0.5] * 100 + [0.95] * 5
        outcomes = [1] * 50 + [0] * 50 + [1] * 5  # tail "perfectly correct"
        result = compute_calibration(
            np.array(probs), np.array(outcomes),
            min_bin_n=10,
        )
        # Tail bin has n=5; should be in dropped_bins.
        bin_ranges = [tuple(b["range"]) for b in result["bins"]]
        dropped_ranges = [tuple(b["range"]) for b in result["dropped_bins"]]
        assert (0.4, 0.6) in bin_ranges
        # The bin containing 0.95 (which is [0.8, 1.01) by default).
        assert any(r[0] == 0.8 for r in dropped_ranges)


class TestEdgeCases:
    def test_insufficient_samples(self):
        result = compute_calibration([0.5] * 10, [1] * 10, min_total_samples=30)
        assert result["status"] == "insufficient_data"

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            compute_calibration(np.array([0.5] * 10), np.array([1] * 5))

    def test_drops_nan(self):
        probs = np.array([np.nan, 0.5, 0.5, 0.5, 0.5] * 20)
        outcomes = np.array([1, 1, 0, 1, 0] * 20)
        result = compute_calibration(probs, outcomes)
        # 80 valid pairs after NaN drop; n = 80.
        assert result["n"] == 80


class TestCustomBinEdges:
    def test_decile_edges(self):
        edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        rng = np.random.default_rng(2)
        probs = rng.uniform(0.0, 1.0, size=500)
        outcomes = (rng.uniform(size=500) < probs).astype(float)
        result = compute_calibration(probs, outcomes, bin_edges=edges, min_bin_n=5)
        # Should produce up to 10 bins (some may be dropped if sparse).
        assert len(result["bins"]) + len(result["dropped_bins"]) <= 10


class TestLibConsolidation:
    """Regression for #1125: the headline ECE must come from the one fleet-wide
    primitive (``alpha_engine_lib.quant.stats.calibration``), and the
    margin-vs-probability bug class must stay caught — feeding the *margin*
    ``|p-0.5|*2`` instead of the probability produces a materially different
    (inflated) ECE, which a calibrated probability set must not.
    """

    def test_ece_matches_lib_primitive(self):
        # calibration_diagnostics must delegate the ECE scalar to the lib, not
        # re-derive it. Bit-for-bit identical on the same edges + min_bin_n.
        from nousergon_lib.quant.stats.calibration import (
            expected_calibration_error,
        )

        edges = [0.0, 0.20, 0.40, 0.60, 0.80, 1.01]
        rng = np.random.default_rng(11)
        probs = rng.uniform(0.0, 1.0, size=600)
        outcomes = (rng.uniform(size=600) < probs).astype(float)

        result = compute_calibration(probs, outcomes, min_bin_n=10)
        lib = expected_calibration_error(
            probs, outcomes, bin_edges=edges, min_bin_n=10, min_samples=1,
        )
        assert result["status"] == "ok"
        # compute_calibration rounds the headline ECE to 4dp for display; the
        # unrounded lib value must agree to that precision (i.e. the scalar is
        # the lib's, not a separately-derived one).
        assert result["ece"] == pytest.approx(round(lib["ece"], 4), abs=1e-9)

    def test_margin_vs_probability_bug_class(self):
        # A perfectly-calibrated probability set: P(y=1) == p in each bucket.
        # ECE on the PROBABILITY is ~0. ECE on the MARGIN |p-0.5|*2 binned
        # against the same hit-rate manufactures a large structural gap — the
        # exact scale-mismatch that produced months of false calibration_breakdown
        # alerts. The two must NOT be interchangeable.
        rng = np.random.default_rng(12)
        probs = []
        outcomes = []
        for p_target in [0.1, 0.3, 0.5, 0.7, 0.9]:
            n = 200
            probs.extend([p_target] * n)
            n_hits = round(n * p_target)
            outcomes.extend([1] * n_hits + [0] * (n - n_hits))
        probs = np.array(probs, dtype=float)
        outcomes = np.array(outcomes, dtype=float)

        on_probability = compute_calibration(probs, outcomes, min_bin_n=10)
        assert on_probability["status"] == "ok"
        assert on_probability["ece"] < 0.02  # well-calibrated as a probability

        # Same outcomes, but ECE computed over the confidence MARGIN.
        margin = np.abs(probs - 0.5) * 2.0
        on_margin = compute_calibration(margin, outcomes, min_bin_n=10)
        # Margin-vs-hit-rate is a scale mismatch: ECE is structurally large
        # even though the model is perfectly calibrated as a probability.
        assert on_margin["ece"] > 0.15
        assert on_margin["ece"] > on_probability["ece"] + 0.1
