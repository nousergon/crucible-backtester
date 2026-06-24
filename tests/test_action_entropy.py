"""Tests for analysis.action_entropy.

Pins:
  1. Uniform 3-action distribution → entropy_normalized = 1.0.
  2. Single-action stream → entropy = 0, alarm = True.
  3. Hand-computed entropy for a known distribution.
  4. Alarm threshold sensitivity.
  5. Rolling entropy produces correct windowed values.
  6. Insufficient samples handled.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from analysis.action_entropy import (
    compute_action_entropy,
    compute_action_entropy_artifact,
    compute_rolling_entropy,
    shannon_entropy,
)


class TestShannonEntropyHelper:
    def test_uniform_distribution(self):
        # 3 equal-prob actions → H = ln(3)
        h = shannon_entropy({"BUY": 1, "HOLD": 1, "SELL": 1})
        assert h == pytest.approx(math.log(3), abs=1e-9)

    def test_concentrated_distribution(self):
        # 90/5/5 — close to 1 action.
        h = shannon_entropy({"HOLD": 0.9, "BUY": 0.05, "SELL": 0.05})
        # Hand-compute: -(0.9*ln(0.9) + 0.05*ln(0.05)*2) ≈ 0.394
        expected = -(0.9 * math.log(0.9) + 0.05 * math.log(0.05) * 2)
        assert h == pytest.approx(expected, abs=1e-9)


class TestComputeActionEntropy:
    def test_uniform_normalized_one(self):
        actions = ["BUY", "HOLD", "SELL"] * 20  # 60 obs, perfectly uniform
        result = compute_action_entropy(actions)
        assert result["status"] == "ok"
        assert result["entropy_normalized"] == pytest.approx(1.0, abs=1e-9)
        assert result["alarm"] is False

    def test_single_action_collapse(self):
        actions = ["HOLD"] * 50
        result = compute_action_entropy(actions)
        assert result["entropy"] == 0.0
        assert result["entropy_normalized"] == 0.0
        assert result["alarm"] is True
        assert result["most_common_fraction"] == 1.0

    def test_hand_computed_entropy(self):
        # 9 HOLD + 1 BUY = 10 obs → distribution [0.9, 0.1]
        # H = -(0.9*ln 0.9 + 0.1*ln 0.1) ≈ 0.325
        # H_max = ln(2) ≈ 0.693, so H_norm ≈ 0.469
        actions = ["HOLD"] * 9 + ["BUY"]
        result = compute_action_entropy(actions, alarm_threshold=0.3)
        expected_h = -(0.9 * math.log(0.9) + 0.1 * math.log(0.1))
        expected_h_norm = expected_h / math.log(2)
        assert result["entropy"] == pytest.approx(expected_h, abs=1e-9)
        assert result["entropy_normalized"] == pytest.approx(expected_h_norm, abs=1e-9)
        assert result["alarm"] is False  # 0.469 > 0.3

    def test_alarm_fires_below_threshold(self):
        # 95/5 split: H_norm ≈ 0.286, below default threshold 0.3.
        actions = ["HOLD"] * 19 + ["BUY"]
        result = compute_action_entropy(actions, alarm_threshold=0.3)
        assert result["alarm"] is True

    def test_insufficient_samples(self):
        result = compute_action_entropy(["BUY", "HOLD"], min_samples=10)
        assert result["status"] == "insufficient_data"


class TestRollingEntropy:
    def test_rolling_window_length(self):
        # 150 obs, window=10 → 141 valid rolling rows (n - window + 1).
        actions = pd.Series(["BUY", "HOLD", "SELL"] * 50)  # 150 elements
        result = compute_rolling_entropy(actions, window=10)
        assert len(result) == 141

    def test_rolling_alarm_on_collapse_segment(self):
        # First 50: uniform, second 50: all HOLD.
        actions = pd.Series(
            ["BUY", "HOLD", "SELL"] * 17 + ["HOLD"] * 50  # 51 + 50 = 101
        )
        result = compute_rolling_entropy(actions, window=20, alarm_threshold=0.3)
        # Earliest windows: uniform → no alarm. Latest: all HOLD → alarm.
        assert result["alarm"].iloc[0] is np.bool_(False) or result["alarm"].iloc[0] == False
        assert result["alarm"].iloc[-1] is np.bool_(True) or result["alarm"].iloc[-1] == True

    def test_invalid_window_raises(self):
        actions = pd.Series(["BUY", "HOLD"] * 10)
        with pytest.raises(ValueError):
            compute_rolling_entropy(actions, window=1)


class TestComputeActionEntropyArtifact:
    """Report-card producer (config#1151 Batch C) — decision-stream extraction."""

    def test_diverse_stance_stream_is_graded(self):
        df = pd.DataFrame({
            "stance": (["momentum", "quality", "value", "low_vol"] * 15),  # 60 obs
            "score": range(60),
        })
        r = compute_action_entropy_artifact(df)
        assert r["status"] == "ok"
        assert r["action_field"] == "stance"
        assert r["n_signals"] == 60
        assert r["entropy_normalized"] == pytest.approx(1.0, abs=1e-9)
        assert r["alarm"] is False

    def test_collapsed_stance_stream_alarms(self):
        df = pd.DataFrame({"stance": ["momentum"] * 40})
        r = compute_action_entropy_artifact(df)
        assert r["status"] == "ok"
        assert r["entropy_normalized"] == 0.0
        assert r["alarm"] is True
        assert r["action_field"] == "stance"

    def test_falls_back_to_conviction_when_no_stance(self):
        df = pd.DataFrame({
            "conviction": (["rising", "stable", "declining"] * 10),  # 30 obs
        })
        r = compute_action_entropy_artifact(df)
        assert r["status"] == "ok"
        assert r["action_field"] == "conviction"

    def test_prefers_stance_over_conviction(self):
        df = pd.DataFrame({
            "stance": ["momentum", "quality"] * 15,
            "conviction": ["rising"] * 30,
        })
        r = compute_action_entropy_artifact(df)
        assert r["action_field"] == "stance"

    def test_no_decision_stream_when_no_label_column(self):
        df = pd.DataFrame({"score": range(50)})
        r = compute_action_entropy_artifact(df)
        assert r["status"] == "no_decision_stream"
        assert r["n_signals"] == 50

    def test_all_null_label_column_is_no_decision_stream(self):
        df = pd.DataFrame({"stance": [None] * 30})
        r = compute_action_entropy_artifact(df)
        assert r["status"] == "no_decision_stream"

    def test_too_few_signals_is_insufficient(self):
        df = pd.DataFrame({"stance": ["momentum", "quality", "value"]})  # 3 < 20
        r = compute_action_entropy_artifact(df)
        assert r["status"] == "insufficient_data"
        assert r["n_signals"] == 3

    def test_none_or_empty_frame_is_insufficient(self):
        assert compute_action_entropy_artifact(None)["status"] == "insufficient_data"
        assert compute_action_entropy_artifact(pd.DataFrame())["status"] == "insufficient_data"

    def test_nulls_dropped_before_entropy(self):
        # 40 momentum + 40 None → after dropna, single-action collapse.
        df = pd.DataFrame({"stance": ["momentum"] * 40 + [None] * 40})
        r = compute_action_entropy_artifact(df)
        assert r["status"] == "ok"
        assert r["n"] == 40  # nulls excluded from the stream
        assert r["n_signals"] == 80  # but counted as rows seen
        assert r["alarm"] is True
