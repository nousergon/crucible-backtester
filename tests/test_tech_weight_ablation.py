"""Tests for optimizer.tech_weight_ablation.

PR-C of the 2026-05-09 sector-team diagnostic arc. Given persisted
sub-scores (PR-B's research v15 migration), find the per-sector weight
config that minimizes (most-negative) corr(rank, return_5d). Surfaced
from the post-mortem on quant rank inversion in healthcare/industrials/
tech.

Locked behavior:

- WeightConfig validates weights sum to 1.0
- Synthetic score = weighted sum of sub-scores
- Re-rank within (team, eval_date), then corr(rank, ret) across team
- Recommendation gates: ≥30 rows/team, best must beat current by 0.10+
- Recommendation-only — applied=False with explanatory note
- Schema-missing graceful degradation (status=no_data, no crash)
- min_weeks gate produces insufficient_data status
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

import pytest

from optimizer.tech_weight_ablation import (
    DEFAULT_GRID,
    WeightConfig,
    _MIN_IMPROVEMENT,
    _MIN_ROWS_PER_TEAM,
    _evaluate_team_under_config,
    compute_tech_weight_ablation,
)


@pytest.fixture
def conn():
    """Build an in-memory DB with the v15 schema (sub-score columns)."""
    c = sqlite3.connect(":memory:")
    c.executescript("""
        CREATE TABLE team_candidates (
            id INTEGER PRIMARY KEY,
            ticker TEXT, eval_date TEXT, team_id TEXT,
            quant_rank INTEGER, quant_score REAL, qual_score REAL,
            team_recommended INTEGER DEFAULT 0,
            rsi_sub_score REAL, macd_sub_score REAL,
            ma50_sub_score REAL, ma200_sub_score REAL,
            momentum_sub_score REAL
        );
        CREATE TABLE universe_returns (
            id INTEGER PRIMARY KEY,
            ticker TEXT, eval_date TEXT, return_5d REAL, beat_spy_5d INTEGER
        );
    """)
    yield c
    c.close()


def _seed(conn, team_id: str, eval_date: str, picks: list[tuple]):
    """Insert (ticker, rsi, macd, ma50, ma200, momentum, return_5d) rows."""
    for i, (ticker, rsi, macd, ma50, ma200, mom, ret) in enumerate(picks, 1):
        conn.execute(
            "INSERT INTO team_candidates "
            "(ticker, eval_date, team_id, quant_rank, quant_score, "
            "rsi_sub_score, macd_sub_score, ma50_sub_score, "
            "ma200_sub_score, momentum_sub_score) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (ticker, eval_date, team_id, i, 0.0,
             rsi, macd, ma50, ma200, mom),
        )
        conn.execute(
            "INSERT INTO universe_returns (ticker, eval_date, return_5d) "
            "VALUES (?,?,?)",
            (ticker, eval_date, ret),
        )


# ── WeightConfig ────────────────────────────────────────────────────────────


class TestWeightConfig:
    def test_validates_sum_to_one(self):
        with pytest.raises(ValueError, match="weights sum"):
            WeightConfig("bad", rsi=0.5, macd=0.5, ma50=0.5,
                         ma200=0.0, momentum=0.0)

    def test_synthetic_score_weighted_sum(self):
        c = WeightConfig("test", rsi=0.5, macd=0.5,
                         ma50=0.0, ma200=0.0, momentum=0.0)
        # 0.5*100 + 0.5*60 = 80
        assert c.synthetic_score(100, 60, 0, 0, 0) == 80.0


def test_default_grid_includes_current():
    names = {c.name for c in DEFAULT_GRID}
    assert "current_default" in names
    assert "rsi_only" in names
    assert "momentum_only" in names


def test_default_grid_all_valid():
    """Each named config must sum to 1.0 (validated at construction)."""
    for c in DEFAULT_GRID:
        s = c.rsi + c.macd + c.ma50 + c.ma200 + c.momentum
        assert abs(s - 1.0) < 1e-6


# ── _evaluate_team_under_config ─────────────────────────────────────────────


class TestEvaluateTeamUnderConfig:
    def test_returns_none_with_single_row_per_date(self):
        # Each date has only 1 row → rank is undefined
        rows = [
            ("2026-04-25", 50, 50, 50, 50, 50, 0.05),
            ("2026-05-02", 50, 50, 50, 50, 50, 0.03),
        ]
        cfg = next(c for c in DEFAULT_GRID if c.name == "current_default")
        assert _evaluate_team_under_config(rows, cfg) is None

    def test_perfect_inverse_rank_correlation(self):
        """If high synthetic-score rows have HIGHEST returns, the
        re-rank will produce strongly negative correlation (rank 1 →
        best return = skilled scorer)."""
        # All on one date, 5 picks. Assign sub-scores so that rsi_only
        # config produces a ranking that perfectly matches return
        # ordering (highest rsi → highest return).
        rows = [
            ("2026-04-25", 90, 50, 50, 50, 50, 0.05),  # high rsi → high return
            ("2026-04-25", 70, 50, 50, 50, 50, 0.03),
            ("2026-04-25", 50, 50, 50, 50, 50, 0.01),
            ("2026-04-25", 30, 50, 50, 50, 50, -0.02),
            ("2026-04-25", 10, 50, 50, 50, 50, -0.04),  # low rsi → low return
        ]
        rsi_only = next(c for c in DEFAULT_GRID if c.name == "rsi_only")
        corr = _evaluate_team_under_config(rows, rsi_only)
        assert corr is not None
        assert corr < -0.9  # near-perfect negative — skilled

    def test_anti_skilled_under_wrong_weight(self):
        """If we re-rank by momentum_only when momentum is anti-correlated
        with returns, we should see strongly POSITIVE rank correlation."""
        rows = [
            ("2026-04-25", 50, 50, 50, 50, 90, -0.04),  # high momentum → loss
            ("2026-04-25", 50, 50, 50, 50, 70, -0.02),
            ("2026-04-25", 50, 50, 50, 50, 50, 0.01),
            ("2026-04-25", 50, 50, 50, 50, 30, 0.03),
            ("2026-04-25", 50, 50, 50, 50, 10, 0.05),  # low momentum → win
        ]
        mom_only = next(c for c in DEFAULT_GRID if c.name == "momentum_only")
        corr = _evaluate_team_under_config(rows, mom_only)
        assert corr is not None
        assert corr > 0.9  # near-perfect positive — anti-skill


# ── compute_tech_weight_ablation ────────────────────────────────────────────


class TestComputeTechWeightAblation:
    def test_must_provide_db(self):
        result = compute_tech_weight_ablation()
        assert result["status"] == "error"

    def test_invalid_run_date(self, conn):
        result = compute_tech_weight_ablation(
            db_conn=conn, run_date="not-a-date"
        )
        assert result["status"] == "error"

    def test_missing_team_candidates_table(self):
        c = sqlite3.connect(":memory:")
        result = compute_tech_weight_ablation(
            db_conn=c, run_date="2026-05-09",
        )
        assert result["status"] == "no_data"
        assert "team_candidates" in result["reason"]
        c.close()

    def test_missing_sub_score_columns(self):
        """Pre-v15 schema (just quant_rank/quant_score) must surface
        clearly so operator knows the producer-side migration hasn't
        rolled out yet, not crash."""
        c = sqlite3.connect(":memory:")
        c.executescript("""
            CREATE TABLE team_candidates (
                id INTEGER PRIMARY KEY, ticker TEXT, eval_date TEXT,
                team_id TEXT, quant_rank INTEGER, quant_score REAL
            );
            CREATE TABLE universe_returns (
                id INTEGER PRIMARY KEY, ticker TEXT, eval_date TEXT,
                return_5d REAL
            );
        """)
        result = compute_tech_weight_ablation(
            db_conn=c, run_date="2026-05-09",
        )
        assert result["status"] == "no_data"
        assert "sub-score columns" in result["reason"]
        c.close()

    def test_insufficient_rows_per_team(self, conn):
        """Each team needs ≥ _MIN_ROWS_PER_TEAM. With 5 rows for one
        team and nothing else, status must be insufficient_data."""
        _seed(conn, "technology", "2026-05-02", [
            ("A", 80, 80, 80, 80, 80, 0.05),
            ("B", 70, 70, 70, 70, 70, 0.03),
            ("C", 60, 60, 60, 60, 60, 0.01),
            ("D", 50, 50, 50, 50, 50, -0.02),
            ("E", 40, 40, 40, 40, 40, -0.04),
        ])
        result = compute_tech_weight_ablation(
            db_conn=conn, run_date="2026-05-09",
        )
        assert result["status"] == "insufficient_data"
        # Per-team status reflects the floor
        tech = next(t for t in result["per_team"]
                    if t["team_id"] == "technology")
        assert tech["status"] == "insufficient_data"
        assert tech["min_required"] == _MIN_ROWS_PER_TEAM

    def test_recommendation_keep_current_when_close(self, conn):
        """If best ablation config beats current_default by less than
        _MIN_IMPROVEMENT, recommendation must be 'keep_current'."""
        # Seed exactly 30 rows where current_default is already
        # near-optimal (no big improvement possible).
        for i in range(30):
            date = f"2026-04-{(i % 8) + 1:02d}"  # 8 dates, ~4 picks each
            score = 50 + (i % 10)
            ret = score / 1000.0  # weak positive correlation with all sub-scores
            _seed(conn, "financials", date, [
                (f"T{i}", score, score, score, score, score, ret),
            ])
        # Need ≥2 rows per date for ranking — re-seed with 4 per date
        conn.execute("DELETE FROM team_candidates")
        conn.execute("DELETE FROM universe_returns")
        for d in range(8):
            date = f"2026-04-{d+1:02d}"
            picks = [
                (f"T{d}{i}", 50 + i*5, 50 + i*5, 50 + i*5, 50 + i*5, 50 + i*5,
                 (i+1) * 0.005)
                for i in range(4)
            ]
            _seed(conn, "financials", date, picks)
        result = compute_tech_weight_ablation(
            db_conn=conn, run_date="2026-04-30",
        )
        assert result["status"] == "ok"
        fin = next(t for t in result["per_team"]
                   if t["team_id"] == "financials")
        assert fin["status"] == "ok"
        # All sub-scores identical per ticker (i=0..3), so all configs
        # produce identical ranking → 0 improvement → keep_current.
        assert fin["recommendation"] == "keep_current"

    def test_recommendation_switch_when_alternative_clears_gate(self, conn):
        """Anti-skill with current_default but clean signal under
        rsi_only: recommend the switch."""
        # 8 dates × 5 picks each. current_default weights have momentum
        # at 0.25, but momentum is anti-correlated with returns.
        # rsi_only picks the right names.
        for d in range(8):
            date = f"2026-04-{d+1:02d}"
            # rsi-skilled order: rsi descending matches return descending
            # momentum-anti-skilled: momentum descending matches return ASCENDING
            picks = [
                (f"H{d}A", 90, 50, 50, 50, 10, 0.05),
                (f"H{d}B", 80, 50, 50, 50, 30, 0.03),
                (f"H{d}C", 60, 50, 50, 50, 50, 0.01),
                (f"H{d}D", 40, 50, 50, 50, 70, -0.02),
                (f"H{d}E", 20, 50, 50, 50, 90, -0.04),
            ]
            _seed(conn, "healthcare", date, picks)
        result = compute_tech_weight_ablation(
            db_conn=conn, run_date="2026-04-30",
        )
        assert result["status"] == "ok"
        hc = next(t for t in result["per_team"]
                  if t["team_id"] == "healthcare")
        assert hc["status"] == "ok"
        assert hc["recommendation"].startswith("switch_to_")
        # rsi_only should win (or at least beat current_default by gate)
        assert hc["best_corr"] < hc["current_corr"]
        assert (hc["current_corr"] - hc["best_corr"]) >= _MIN_IMPROVEMENT

    def test_recommendation_only_no_apply(self, conn):
        for d in range(8):
            date = f"2026-04-{d+1:02d}"
            picks = [
                (f"T{d}A", 90, 50, 50, 50, 10, 0.05),
                (f"T{d}B", 80, 50, 50, 50, 30, 0.03),
                (f"T{d}C", 60, 50, 50, 50, 50, 0.01),
                (f"T{d}D", 40, 50, 50, 50, 70, -0.02),
                (f"T{d}E", 20, 50, 50, 50, 90, -0.04),
            ]
            _seed(conn, "technology", date, picks)
        result = compute_tech_weight_ablation(
            db_conn=conn, run_date="2026-04-30",
        )
        assert result["status"] == "ok"
        # Must NEVER auto-apply in this PR — that's the parallel-
        # observation cutover follow-up.
        assert result["applied"] is False
        assert "recommendation-only" in result["apply_note"]

    def test_window_filtering(self, conn):
        # Old rows outside window must be excluded
        for d in range(8):
            date = f"2026-04-{d+1:02d}"
            picks = [
                (f"T{d}A", 90, 50, 50, 50, 10, 0.05),
                (f"T{d}B", 80, 50, 50, 50, 30, 0.03),
                (f"T{d}C", 60, 50, 50, 50, 50, 0.01),
                (f"T{d}D", 40, 50, 50, 50, 70, -0.02),
                (f"T{d}E", 20, 50, 50, 50, 90, -0.04),
            ]
            _seed(conn, "technology", date, picks)
        # Add ancient rows that should be excluded
        _seed(conn, "technology", "2024-01-01", [
            ("ANCIENT", 99, 99, 99, 99, 99, 0.99),
        ])
        result = compute_tech_weight_ablation(
            db_conn=conn, run_date="2026-04-30", lookback_weeks=8,
        )
        tech = next(t for t in result["per_team"]
                    if t["team_id"] == "technology")
        # 8 dates × 5 picks = 40 rows in window; ancient must not appear
        assert tech["n_rows"] == 40
