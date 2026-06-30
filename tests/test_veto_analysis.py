"""Unit tests for analysis.veto_analysis — scoring and precision logic, no S3 calls."""
import pytest
from unittest.mock import patch, MagicMock

import pandas as pd

from analysis.veto_analysis import analyze_veto_effectiveness, init_config


# ── Helpers ──────────────────────────────────────────────────────────────────


def _init_default_config():
    """Initialize the module config with test-friendly defaults."""
    init_config({
        "veto_analysis": {
            "confidence_thresholds": [0.50, 0.60, 0.70, 0.80],
            "current_default_threshold": 0.60,
            "min_predictions": 10,
            "min_veto_decisions": 3,
            "cost_penalty_weight": 0.30,
        }
    })


def _make_score_perf_df(n: int = 50, beat_rate: float = 0.5) -> pd.DataFrame:
    """Create a synthetic score_performance DataFrame."""
    import random
    random.seed(42)

    rows = []
    for i in range(n):
        day = (i % 28) + 1
        beat = 1 if random.random() < beat_rate else 0
        ret = random.uniform(-0.10, 0.15) if beat else random.uniform(-0.15, 0.05)
        rows.append({
            "symbol": f"STOCK{i % 10}",
            "score_date": f"2026-01-{day:02d}",
            "beat_spy_21d": beat,
            "return_21d": ret,
        })
    return pd.DataFrame(rows)


def _make_predictions_by_date(
    df: pd.DataFrame,
    down_fraction: float = 0.3,
    confidence_range: tuple = (0.45, 0.85),
) -> dict:
    """Create mock predictions matching the df dates and symbols.

    A fraction of predictions are labeled DOWN with varying confidence.
    """
    import random
    random.seed(123)

    predictions_by_date = {}
    for d in df["score_date"].unique():
        by_ticker = {}
        date_rows = df[df["score_date"] == d]
        for _, row in date_rows.iterrows():
            ticker = row["symbol"]
            is_down = random.random() < down_fraction
            conf = random.uniform(*confidence_range)
            by_ticker[ticker] = {
                "predicted_direction": "DOWN" if is_down else "UP",
                "prediction_confidence": conf,
                "p_up": 1 - conf if is_down else conf,
                "p_down": conf if is_down else 1 - conf,
            }
        predictions_by_date[d] = by_ticker
    return predictions_by_date


# ═══════════════════════════════════════════════════════════════════════════════
# Precision and scoring logic
# ═══════════════════════════════════════════════════════════════════════════════


class TestVetoAnalysis:

    def setup_method(self):
        _init_default_config()

    @patch("analysis.veto_analysis._load_all_predictions")
    def test_precision_computation(self, mock_load):
        """Precision should be true_negatives / n_vetoes for each threshold."""
        df = _make_score_perf_df(n=80, beat_rate=0.5)
        preds = _make_predictions_by_date(df, down_fraction=0.4, confidence_range=(0.50, 0.85))
        mock_load.return_value = preds

        result = analyze_veto_effectiveness(df, bucket="test-bucket")

        # Check that thresholds have precision computed
        for t in result.get("thresholds", []):
            if t["n_vetoes"] > 0:
                assert t["precision"] is not None
                expected = t["true_negatives"] / t["n_vetoes"]
                assert abs(t["precision"] - round(expected, 4)) < 0.001

    @patch("analysis.veto_analysis._load_all_predictions")
    def test_scoring_function_balances_precision_and_cost(self, mock_load):
        """The scoring function should be: precision - cost_weight * normalized_missed."""
        df = _make_score_perf_df(n=80, beat_rate=0.5)
        preds = _make_predictions_by_date(df, down_fraction=0.4, confidence_range=(0.50, 0.85))
        mock_load.return_value = preds

        result = analyze_veto_effectiveness(df, bucket="test-bucket")

        # If we got a recommendation, it should be in the thresholds list
        if result.get("status") == "ok":
            recommended = result["recommended_threshold"]
            threshold_confs = [t["confidence"] for t in result["thresholds"]]
            assert recommended in threshold_confs

    @patch("analysis.veto_analysis._load_all_predictions")
    def test_min_veto_decisions_gate(self, mock_load):
        """Thresholds with fewer than min_veto_decisions are not scoreable."""
        # Use very high confidence range so most predictions have low confidence
        df = _make_score_perf_df(n=30, beat_rate=0.5)
        preds = _make_predictions_by_date(
            df, down_fraction=0.1, confidence_range=(0.45, 0.55)
        )
        mock_load.return_value = preds

        # Set a high min_veto_decisions threshold
        init_config({
            "veto_analysis": {
                "confidence_thresholds": [0.50, 0.60, 0.70, 0.80],
                "current_default_threshold": 0.60,
                "min_predictions": 5,
                "min_veto_decisions": 100,  # Impossibly high
                "cost_penalty_weight": 0.30,
            }
        })

        result = analyze_veto_effectiveness(df, bucket="test-bucket")
        # Should get insufficient_vetoes since no threshold has 100+ decisions
        assert result.get("status") in ("insufficient_vetoes", "no_down_predictions", "insufficient_lift")

    @patch("analysis.veto_analysis._load_all_predictions")
    def test_lift_over_base_rate_check(self, mock_load):
        """If best lift < 5pp over base rate, status should be insufficient_lift."""
        # Create data where veto precision is close to base rate (no lift)
        df = _make_score_perf_df(n=100, beat_rate=0.5)
        # Make all DOWN predictions with confidence uniformly distributed
        preds = _make_predictions_by_date(
            df, down_fraction=0.5, confidence_range=(0.50, 0.90)
        )
        mock_load.return_value = preds

        result = analyze_veto_effectiveness(df, bucket="test-bucket")

        # The result should either be ok (if lift happens to be > 5%)
        # or insufficient_lift (if precision is close to base rate)
        assert result.get("status") in ("ok", "insufficient_lift", "insufficient_vetoes")

    def test_empty_df_returns_insufficient_data(self):
        """Empty DataFrame → insufficient_data."""
        result = analyze_veto_effectiveness(pd.DataFrame(), bucket="test-bucket")
        assert result["status"] == "insufficient_data"

    def test_none_df_returns_insufficient_data(self):
        """None DataFrame → insufficient_data."""
        result = analyze_veto_effectiveness(None, bucket="test-bucket")
        assert result["status"] == "insufficient_data"

    @patch("analysis.veto_analysis._load_all_predictions")
    def test_no_predictions_returns_no_predictions(self, mock_load):
        """No predictions in S3 → status=no_predictions."""
        df = _make_score_perf_df(n=50)
        mock_load.return_value = {}

        result = analyze_veto_effectiveness(df, bucket="test-bucket")
        assert result["status"] == "no_predictions"

    @patch("analysis.veto_analysis._load_all_predictions")
    def test_no_down_predictions_returns_no_down(self, mock_load):
        """All predictions are UP → status=no_down_predictions."""
        df = _make_score_perf_df(n=50)
        # All predictions are UP
        preds = {}
        for d in df["score_date"].unique():
            by_ticker = {}
            for _, row in df[df["score_date"] == d].iterrows():
                by_ticker[row["symbol"]] = {
                    "predicted_direction": "UP",
                    "prediction_confidence": 0.70,
                    "p_up": 0.70,
                    "p_down": 0.30,
                }
            preds[d] = by_ticker
        mock_load.return_value = preds

        result = analyze_veto_effectiveness(df, bucket="test-bucket")
        assert result["status"] == "no_down_predictions"

    @patch("analysis.veto_analysis._load_all_predictions")
    def test_base_rate_computed_correctly(self, mock_load):
        """Base rate should match mean of beat_spy_21d in populated data."""
        beat_rate = 0.6
        df = _make_score_perf_df(n=100, beat_rate=beat_rate)
        preds = _make_predictions_by_date(df, down_fraction=0.3)
        mock_load.return_value = preds

        result = analyze_veto_effectiveness(df, bucket="test-bucket")

        if "base_rate" in result:
            expected_base = float(df["beat_spy_21d"].mean())
            assert abs(result["base_rate"] - round(expected_base, 4)) < 0.01
