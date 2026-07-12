"""Tests for analysis.score_calibrator + the calibrated compute_portfolio_calibration.

Pins the config#2304 Option-A fix:
  1. Isotonic fit maps composite score → P(beat SPY) monotonically.
  2. The config#2304 scenario — a high-but-uncalibrated score (60-82, ~50% hit
     rate) — reads as a spurious RED under raw score/100 but calibrates clean.
  3. Circularity guard: out-of-fold ECE is honest, NOT the trivially-~0 in-sample
     ECE a full-corpus fit would produce.
  4. Degenerate corpora (constant score / single-class outcome) degrade to the
     base rate without raising.
  5. save/load round-trips the mapping; OOF is deterministic under a fixed seed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.calibration_diagnostics import compute_calibration
from analysis.score_calibrator import (
    ScoreProbabilityCalibrator,
    out_of_fold_calibrated_probabilities,
)
from analysis.team_skill_metrics import compute_portfolio_calibration


def _miscalibrated_corpus(n: int = 402, seed: int = 7):
    """Reproduce the config#2304 shape: composite scores in the BUY-ish 60-85
    band whose *actual* beat-SPY rate is ~50% and only weakly rank-ordered by
    score. Raw score/100 (0.60-0.85) badly over-predicts → high ECE; the true
    monotone signal is mild, so an honest calibrator maps everything near ~0.5.
    """
    rng = np.random.default_rng(seed)
    scores = rng.uniform(60, 85, size=n)
    # Mild real signal: hit prob drifts 0.45 → 0.58 across the score band.
    true_p = 0.45 + (scores - 60.0) / (85.0 - 60.0) * 0.13
    outcomes = (rng.uniform(size=n) < true_p).astype(float)
    return scores, outcomes


class TestScoreProbabilityCalibrator:
    def test_isotonic_fit_is_monotone(self):
        scores, outcomes = _miscalibrated_corpus()
        cal = ScoreProbabilityCalibrator(method="isotonic").fit(scores, outcomes)
        assert cal.is_fitted
        grid = np.arange(60, 86, dtype=float)
        preds = cal.predict_batch(grid)
        assert np.all(np.diff(preds) >= -1e-9)  # non-decreasing
        assert np.all((preds >= 0.0) & (preds <= 1.0))

    def test_predict_single_matches_batch(self):
        scores, outcomes = _miscalibrated_corpus()
        cal = ScoreProbabilityCalibrator().fit(scores, outcomes)
        assert cal.predict(72.0) == pytest.approx(cal.predict_batch([72.0])[0])

    def test_insufficient_samples_stays_at_prior(self):
        cal = ScoreProbabilityCalibrator().fit(np.array([70.0, 72.0]), np.array([1.0, 0.0]))
        assert not cal.is_fitted
        assert cal.predict(70.0) == pytest.approx(0.50)

    def test_degenerate_constant_score(self):
        scores = np.full(50, 70.0)
        outcomes = np.array([1.0, 0.0] * 25)
        cal = ScoreProbabilityCalibrator().fit(scores, outcomes)
        assert cal.is_fitted
        assert cal.predict(70.0) == pytest.approx(0.5, abs=0.05)

    def test_degenerate_single_class_outcome(self):
        scores = np.linspace(60, 90, 40)
        outcomes = np.ones(40)  # everyone beat SPY
        cal = ScoreProbabilityCalibrator().fit(scores, outcomes)
        assert cal.is_fitted
        assert cal.predict(75.0) == pytest.approx(1.0, abs=1e-9)

    def test_platt_method(self):
        scores, outcomes = _miscalibrated_corpus()
        cal = ScoreProbabilityCalibrator(method="platt").fit(scores, outcomes)
        assert cal.is_fitted
        preds = cal.predict_batch(np.arange(60, 86, dtype=float))
        assert np.all(np.diff(preds) >= -1e-9)  # logistic is monotone in score

    def test_save_load_roundtrip(self, tmp_path):
        scores, outcomes = _miscalibrated_corpus()
        cal = ScoreProbabilityCalibrator().fit(scores, outcomes)
        p = tmp_path / "cal.json"
        cal.save(p)
        loaded = ScoreProbabilityCalibrator.load(p)
        grid = np.arange(60, 86, dtype=float)
        np.testing.assert_allclose(cal.predict_batch(grid), loaded.predict_batch(grid), atol=1e-5)

    def test_bad_method_raises(self):
        with pytest.raises(ValueError):
            ScoreProbabilityCalibrator(method="banana")


class TestOutOfFold:
    def test_oof_is_deterministic(self):
        scores, outcomes = _miscalibrated_corpus()
        a = out_of_fold_calibrated_probabilities(scores, outcomes)
        b = out_of_fold_calibrated_probabilities(scores, outcomes)
        np.testing.assert_array_equal(a, b)

    def test_oof_scores_every_sample(self):
        scores, outcomes = _miscalibrated_corpus()
        probs = out_of_fold_calibrated_probabilities(scores, outcomes)
        assert np.isfinite(probs).all()

    def test_oof_below_min_samples_is_nan(self):
        probs = out_of_fold_calibrated_probabilities(
            np.linspace(60, 80, 5), np.array([1, 0, 1, 0, 1.0])
        )
        assert np.isnan(probs).all()

    def test_oof_not_the_circular_zero(self):
        """A full-corpus (in-sample) isotonic fit is calibrated by construction —
        its ECE collapses to ~0, which is the circular measurement config#2304
        warns against. OOF must give an HONEST, materially non-zero-vs-in-sample
        signal, proving it isn't just re-scoring the training data.
        """
        scores, outcomes = _miscalibrated_corpus(seed=11)
        cal = ScoreProbabilityCalibrator().fit(scores, outcomes)
        in_sample = compute_calibration(cal.predict_batch(scores), outcomes, min_total_samples=30)
        oof_probs = out_of_fold_calibrated_probabilities(scores, outcomes)
        oof = compute_calibration(oof_probs, outcomes, min_total_samples=30)
        assert in_sample["status"] == "ok" and oof["status"] == "ok"
        assert in_sample["ece"] <= oof["ece"] + 1e-9  # in-sample never worse (it's circular)


class TestCalibratedPortfolioGate:
    def test_config2304_scenario_raw_red_calibrates_clean(self):
        """The headline config#2304 result: an uncalibrated high-score corpus
        reads RED (ECE past 0.15) on raw score/100 but PASSES once the isotonic
        layer maps the score to a legitimate probability.
        """
        scores, outcomes = _miscalibrated_corpus()
        df = pd.DataFrame({"score": scores, "beat_spy_21d": outcomes})

        raw = compute_portfolio_calibration(df, calibrate=False)
        calibrated = compute_portfolio_calibration(df, calibrate=True)

        assert raw["status"] == "ok" and calibrated["status"] == "ok"
        assert raw["ece"] > 0.15                       # spurious RED under raw score
        assert calibrated["ece"] < raw["ece"]          # calibration materially improves it
        assert calibrated["ece"] < 0.15                # clears the red-line honestly
        assert calibrated["calibration_method"] == "isotonic_oof"
        assert calibrated["raw_score_ece"] == pytest.approx(raw["ece"])

    def test_shape_stays_grader_compatible(self):
        scores, outcomes = _miscalibrated_corpus()
        df = pd.DataFrame({"score": scores, "beat_spy_21d": outcomes})
        result = compute_portfolio_calibration(df)
        # Keys the grading layer (_grade_calibration_diagnostics) reads.
        for key in ("status", "n", "ece", "brier_score", "quality", "bins"):
            assert key in result
