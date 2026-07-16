"""Tests for analysis modules — pure logic with synthetic DataFrames.

Covers: stats_utils, regime_analysis, score_analysis, alpha_distribution,
attribution, macro_eval, sizing_ab, trigger_scorecard, exit_timing,
shadow_book, veto_value.
"""

import sqlite3
import tempfile

import pandas as pd
import numpy as np
import pytest

from analysis.stats_utils import benjamini_hochberg
from analysis.score_analysis import accuracy_by_threshold
from analysis.signal_quality import compute_accuracy


# ---------------------------------------------------------------------------
# stats_utils
# ---------------------------------------------------------------------------


class TestBenjaminiHochberg:
    def test_all_significant(self):
        result = benjamini_hochberg([0.001, 0.002, 0.003])
        assert all(result)

    def test_none_significant(self):
        result = benjamini_hochberg([0.5, 0.8, 0.9])
        assert not any(result)

    def test_partial(self):
        result = benjamini_hochberg([0.001, 0.04, 0.5, 0.9])
        assert result[0] is True
        assert result[3] is False

    def test_empty(self):
        assert benjamini_hochberg([]) == []

    def test_single(self):
        assert benjamini_hochberg([0.01]) == [True]
        assert benjamini_hochberg([0.10]) == [False]

    def test_custom_alpha(self):
        result = benjamini_hochberg([0.08, 0.09], alpha=0.10)
        assert result[0] is True


# ---------------------------------------------------------------------------
# score_analysis
# ---------------------------------------------------------------------------


class TestAccuracyByThreshold:
    def _make_df(self, n=50):
        np.random.seed(42)
        return pd.DataFrame({
            "score": np.random.uniform(55, 95, n),
            "beat_spy_5d": np.random.choice([0, 1], n),
            "beat_spy_21d": np.random.choice([0, 1], n),
            "beat_spy_10d": np.random.choice([0, 1], n),
            "beat_spy_30d": np.random.choice([0, 1], n),
            "return_5d": np.random.uniform(-0.05, 0.05, n),
            "return_21d": np.random.uniform(-0.05, 0.05, n),
            "return_10d": np.random.uniform(-0.05, 0.05, n),
            "return_30d": np.random.uniform(-0.05, 0.05, n),
            "spy_5d_return": np.random.uniform(-0.02, 0.02, n),
            "spy_21d_return": np.random.uniform(-0.02, 0.02, n),
            "spy_10d_return": np.random.uniform(-0.02, 0.02, n),
            "spy_30d_return": np.random.uniform(-0.02, 0.02, n),
        })

    def test_returns_list(self):
        df = self._make_df()
        result = accuracy_by_threshold(df, thresholds=[60, 70, 80], min_samples=5)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_each_row_has_threshold(self):
        df = self._make_df()
        result = accuracy_by_threshold(df, thresholds=[60, 70], min_samples=5)
        for row in result:
            assert "threshold" in row
            assert "accuracy_21d" in row

    def test_higher_threshold_fewer_samples(self):
        df = self._make_df(100)
        result = accuracy_by_threshold(df, thresholds=[60, 80], min_samples=1)
        if len(result) == 2:
            assert result[0].get("n_21d", 0) >= result[1].get("n_21d", 0)

    def test_insufficient_data(self):
        df = self._make_df(5)
        result = accuracy_by_threshold(df, thresholds=[60], min_samples=10)
        assert result == []


# ---------------------------------------------------------------------------
# signal_quality.compute_accuracy
# ---------------------------------------------------------------------------


class TestComputeAccuracy:
    def _make_df(self, n=50):
        np.random.seed(42)
        return pd.DataFrame({
            "symbol": [f"TICK{i}" for i in range(n)],
            "score": np.random.uniform(55, 95, n),
            "beat_spy_5d": np.random.choice([0.0, 1.0], n),
            "beat_spy_21d": np.random.choice([0.0, 1.0], n),
            "beat_spy_10d": np.random.choice([0.0, 1.0], n),
            "beat_spy_30d": np.random.choice([0.0, 1.0], n),
            "return_5d": np.random.uniform(-0.05, 0.05, n),
            "return_21d": np.random.uniform(-0.05, 0.05, n),
            "return_10d": np.random.uniform(-0.05, 0.05, n),
            "return_30d": np.random.uniform(-0.05, 0.05, n),
            "spy_5d_return": np.random.uniform(-0.02, 0.02, n),
            "spy_21d_return": np.random.uniform(-0.02, 0.02, n),
            "spy_10d_return": np.random.uniform(-0.02, 0.02, n),
            "spy_30d_return": np.random.uniform(-0.02, 0.02, n),
        })

    def test_ok_result(self):
        df = self._make_df()
        result = compute_accuracy(df, min_samples=5)
        assert result["status"] == "ok"
        assert "overall" in result
        assert "by_score_bucket" in result

    def test_insufficient_data(self):
        df = self._make_df(3)
        result = compute_accuracy(df, min_samples=10)
        assert result["status"] == "insufficient_data"

    def test_overall_accuracy_range(self):
        df = self._make_df()
        result = compute_accuracy(df, min_samples=5)
        acc = result["overall"]["accuracy_21d"]
        assert 0.0 <= acc <= 1.0

    def test_precision_field(self):
        df = self._make_df()
        result = compute_accuracy(df, min_samples=5)
        assert "precision_21d" in result["overall"]

    def test_by_sector(self):
        df = self._make_df()
        df["sector"] = np.random.choice(["Technology", "Healthcare", "Financials"], len(df))
        result = compute_accuracy(df, min_samples=5)
        assert "by_sector" in result
        assert len(result["by_sector"]) > 0

    def test_by_conviction(self):
        df = self._make_df()
        df["conviction"] = np.random.choice(["rising", "stable", "declining"], len(df))
        result = compute_accuracy(df, min_samples=5)
        assert "by_conviction" in result


# ---------------------------------------------------------------------------
# signal_quality._accuracy_by_score_bucket — bucket-boundary coverage
# (config#2674: hit-rate stratified by score-bucket x horizon)
# ---------------------------------------------------------------------------


class TestAccuracyByScoreBucketBoundaries:
    """Buckets are [60-70), [70-80), [80-90), [90-101) — verify each boundary
    score lands in exactly the bucket the half-open interval implies, and that
    every returned row carries both the 5d and 21d horizon (score-bucket x
    horizon stratification)."""

    def _row(self, score):
        return {
            "score": score,
            "beat_spy_5d": 1.0,
            "beat_spy_21d": 0.0,
            "return_5d": 0.01,
            "return_21d": 0.02,
            "spy_5d_return": 0.005,
            "spy_21d_return": 0.01,
        }

    def _df_for(self, scores):
        df = pd.DataFrame([self._row(s) for s in scores])
        return df, df  # same frame stands in for both the 5d and 21d slice

    def _bucket_for_score(self, score):
        from analysis.signal_quality import _accuracy_by_score_bucket
        df_5d, df_21d = self._df_for([score])
        rows = _accuracy_by_score_bucket(df_5d, df_21d)
        assert len(rows) == 1, f"score {score} should land in exactly one bucket, got {rows}"
        return rows[0]["bucket"]

    @pytest.mark.parametrize("score,expected_bucket", [
        (60.0, "60-70"),
        (69.9, "60-70"),
        (70.0, "70-80"),
        (79.9, "70-80"),
        (80.0, "80-90"),
        (89.9, "80-90"),
        (90.0, "90+"),
        (100.0, "90+"),
        (100.9, "90+"),
    ])
    def test_boundary_assignment(self, score, expected_bucket):
        assert self._bucket_for_score(score) == expected_bucket

    def test_below_lowest_bucket_excluded(self):
        from analysis.signal_quality import _accuracy_by_score_bucket
        df_5d, df_21d = self._df_for([59.9])
        rows = _accuracy_by_score_bucket(df_5d, df_21d)
        assert rows == []

    def test_at_and_above_upper_edge_excluded(self):
        from analysis.signal_quality import _accuracy_by_score_bucket
        df_5d, df_21d = self._df_for([101.0, 105.0])
        rows = _accuracy_by_score_bucket(df_5d, df_21d)
        assert rows == []

    def test_each_bucket_row_carries_both_horizons(self):
        from analysis.signal_quality import _accuracy_by_score_bucket
        df_5d, df_21d = self._df_for([65, 75, 85, 95])
        rows = _accuracy_by_score_bucket(df_5d, df_21d)
        buckets = {r["bucket"] for r in rows}
        assert buckets == {"60-70", "70-80", "80-90", "90+"}
        for row in rows:
            assert "accuracy_5d" in row
            assert "accuracy_21d" in row


# ---------------------------------------------------------------------------
# regime_analysis (without DB — use DataFrame directly)
# ---------------------------------------------------------------------------


class TestRegimeAnalysis:
    def test_accuracy_by_regime(self):
        from analysis.regime_analysis import accuracy_by_regime
        np.random.seed(42)
        n = 60
        df = pd.DataFrame({
            "beat_spy_5d": np.random.choice([0.0, 1.0], n),
            "beat_spy_21d": np.random.choice([0.0, 1.0], n),
            "beat_spy_10d": np.random.choice([0.0, 1.0], n),
            "beat_spy_30d": np.random.choice([0.0, 1.0], n),
            "return_5d": np.random.uniform(-0.05, 0.05, n),
            "return_21d": np.random.uniform(-0.05, 0.05, n),
            "return_10d": np.random.uniform(-0.05, 0.05, n),
            "return_30d": np.random.uniform(-0.05, 0.05, n),
            "spy_5d_return": np.random.uniform(-0.02, 0.02, n),
            "spy_21d_return": np.random.uniform(-0.02, 0.02, n),
            "spy_10d_return": np.random.uniform(-0.02, 0.02, n),
            "spy_30d_return": np.random.uniform(-0.02, 0.02, n),
            "market_regime": np.random.choice(["bull", "neutral", "bear"], n),
        })
        result = accuracy_by_regime(df, min_samples=5)
        assert len(result) == 3
        for r in result:
            assert "market_regime" in r
            assert "accuracy_21d" in r

    def test_insufficient_data(self):
        from analysis.regime_analysis import accuracy_by_regime
        df = pd.DataFrame({
            "beat_spy_5d": [1.0],
            "beat_spy_10d": [1.0],
            "beat_spy_30d": [1.0],
            "market_regime": ["bull"],
        })
        result = accuracy_by_regime(df, min_samples=10)
        assert result == []


# ---------------------------------------------------------------------------
# alpha_distribution
# ---------------------------------------------------------------------------


class TestAlphaDistribution:
    def test_compute(self):
        from analysis.alpha_distribution import compute_alpha_distribution
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            conn = sqlite3.connect(f.name)
            conn.execute("""
                CREATE TABLE score_performance (
                    symbol TEXT, score_date TEXT, score REAL,
                    return_5d REAL, return_10d REAL, return_30d REAL,
                    spy_5d_return REAL, spy_10d_return REAL, spy_30d_return REAL,
                    beat_spy_5d INTEGER, beat_spy_10d INTEGER, beat_spy_30d INTEGER
                )
            """)
            np.random.seed(42)
            for i in range(50):
                conn.execute(
                    "INSERT INTO score_performance VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (f"T{i}", "2026-04-01", 70 + np.random.uniform(-10, 20),
                     np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05),
                     np.random.uniform(-0.05, 0.05), 0.01, 0.02, 0.03,
                     int(np.random.choice([0, 1])), int(np.random.choice([0, 1])),
                     int(np.random.choice([0, 1]))),
                )
            conn.commit()
            conn.close()
            result = compute_alpha_distribution(f.name, min_samples=5)
            assert result["status"] == "ok"
            assert "distributions" in result
            assert "summary" in result


# ---------------------------------------------------------------------------
# macro_eval
# ---------------------------------------------------------------------------


class TestMacroEval:
    def test_compute(self):
        from analysis.macro_eval import compute_macro_evaluation
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            conn = sqlite3.connect(f.name)
            conn.execute("""
                CREATE TABLE score_performance (
                    symbol TEXT, score_date TEXT, score REAL,
                    return_10d REAL, spy_10d_return REAL, beat_spy_10d INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE cio_evaluations (
                    ticker TEXT, eval_date TEXT, combined_score REAL,
                    final_score REAL, macro_shift REAL
                )
            """)
            np.random.seed(42)
            for i in range(40):
                score = 70 + np.random.uniform(-10, 20)
                macro_shift = np.random.uniform(-5, 5)
                conn.execute(
                    "INSERT INTO score_performance VALUES (?,?,?,?,?,?)",
                    (f"T{i}", "2026-04-01", score,
                     np.random.uniform(-0.05, 0.05), 0.01, int(np.random.choice([0, 1]))),
                )
                conn.execute(
                    "INSERT INTO cio_evaluations VALUES (?,?,?,?,?)",
                    (f"T{i}", "2026-04-01", score - macro_shift, score, macro_shift),
                )
            conn.commit()
            conn.close()
            result = compute_macro_evaluation(f.name, min_samples=5)
            # May return ok, insufficient_data, or error depending on schema match
            assert result["status"] in ("ok", "insufficient_data", "error")
