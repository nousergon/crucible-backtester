"""Tests for analysis.quant_rank_quality.

Diagnostic measures per-sector ``corr(quant_rank, realized return)`` over a
rolling window, at BOTH fleet-policy horizons (diagnostic 5d under the
legacy unsuffixed keys + canonical primary 21d under suffixed keys,
config#1529). The 2026-05-09 evaluator-email post-mortem found
healthcare/industrials/tech rank-correlations at +0.33-0.36 — anti-skill
territory. This module catches that drift weekly so it can't recur in
silence.

Locked behavior:
- Negative correlation in a skilled team (rank #1 → highest return)
- Positive correlation > ANTI_SKILL_THRESHOLD trips the anti-skill flag
- Empty data → status="no_data", no crash
- Per-team breakdown + pooled overall metric, per horizon
- Legacy (5d) keys keep their exact pre-config#1529 names and values —
  the 21d channel is strictly additive
- Primary-horizon (21d) metrics computed from the 21d outcome columns,
  including the resolution-lag case (5d resolved, 21d still NULL)
- Missing 21d columns in universe_returns degrade the 21d channel to
  None (loud WARN) without breaking the 5d channel
- CW emission injectable for tests; emits per-horizon metric names
- Top-3 hit rate computed alongside rank correlation, per horizon
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from analysis.quant_rank_quality import (
    ANTI_SKILL_THRESHOLD,
    CANONICAL_SECTORS,
    _safe_pearson,
    _team_rank_quality,
    compute_quant_rank_quality,
)


# ── Test fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def conn():
    """Build an in-memory DB with the minimal schema needed (both horizons,
    mirroring the live universe_returns post canonical-alpha cutover)."""
    c = sqlite3.connect(":memory:")
    c.executescript("""
        CREATE TABLE team_candidates (
            id INTEGER PRIMARY KEY,
            ticker TEXT, eval_date TEXT, team_id TEXT,
            quant_rank INTEGER, quant_score REAL, qual_score REAL,
            team_recommended INTEGER DEFAULT 0
        );
        CREATE TABLE universe_returns (
            id INTEGER PRIMARY KEY,
            ticker TEXT, eval_date TEXT, sector TEXT,
            close_price REAL, return_5d REAL, return_10d REAL,
            spy_return_5d REAL, spy_return_10d REAL,
            beat_spy_5d INTEGER, beat_spy_10d INTEGER,
            return_21d REAL, spy_return_21d REAL, beat_spy_21d INTEGER,
            sector_etf TEXT
        );
    """)
    yield c
    c.close()


@pytest.fixture
def conn_legacy_schema():
    """A pre-canonical-alpha universe_returns WITHOUT the 21d columns —
    the degrade path (primary-horizon metrics must go None, not crash)."""
    c = sqlite3.connect(":memory:")
    c.executescript("""
        CREATE TABLE team_candidates (
            id INTEGER PRIMARY KEY,
            ticker TEXT, eval_date TEXT, team_id TEXT,
            quant_rank INTEGER, quant_score REAL, qual_score REAL,
            team_recommended INTEGER DEFAULT 0
        );
        CREATE TABLE universe_returns (
            id INTEGER PRIMARY KEY,
            ticker TEXT, eval_date TEXT, sector TEXT,
            close_price REAL, return_5d REAL,
            spy_return_5d REAL, beat_spy_5d INTEGER,
            sector_etf TEXT
        );
    """)
    yield c
    c.close()


def _seed(conn, *, team_id: str, eval_date: str, picks: list[tuple]):
    """Insert picks into both tables.

    Each pick is (rank, ticker, score, return_5d, beat_spy_5d) — the 21d
    outcomes then mirror the 5d ones — or the 7-tuple
    (rank, ticker, score, return_5d, beat_spy_5d, return_21d, beat_spy_21d)
    to seed a diverging / unresolved primary horizon (None,None = 21d
    resolution lag).
    """
    for pick in picks:
        if len(pick) == 5:
            rank, ticker, score, ret, beat = pick
            ret21, beat21 = ret, beat
        else:
            rank, ticker, score, ret, beat, ret21, beat21 = pick
        conn.execute(
            "INSERT INTO team_candidates "
            "(ticker, eval_date, team_id, quant_rank, quant_score) "
            "VALUES (?,?,?,?,?)",
            (ticker, eval_date, team_id, rank, score),
        )
        conn.execute(
            "INSERT INTO universe_returns "
            "(ticker, eval_date, return_5d, beat_spy_5d, return_21d, beat_spy_21d) "
            "VALUES (?,?,?,?,?,?)",
            (ticker, eval_date, ret, beat, ret21, beat21),
        )


def _seed_legacy(conn, *, team_id: str, eval_date: str, picks: list[tuple]):
    """Seed the pre-canonical schema (5d columns only)."""
    for rank, ticker, score, ret, beat in picks:
        conn.execute(
            "INSERT INTO team_candidates "
            "(ticker, eval_date, team_id, quant_rank, quant_score) "
            "VALUES (?,?,?,?,?)",
            (ticker, eval_date, team_id, rank, score),
        )
        conn.execute(
            "INSERT INTO universe_returns "
            "(ticker, eval_date, return_5d, beat_spy_5d) VALUES (?,?,?,?)",
            (ticker, eval_date, ret, beat),
        )


# ── Pearson helper ──────────────────────────────────────────────────────────


class TestSafePearson:
    def test_returns_none_below_min_n(self):
        assert _safe_pearson([1, 2], [1, 2]) is None
        assert _safe_pearson([], []) is None

    def test_returns_none_zero_variance(self):
        # All x identical → den_x=0
        assert _safe_pearson([5, 5, 5], [1, 2, 3]) is None
        # All y identical → den_y=0
        assert _safe_pearson([1, 2, 3], [7, 7, 7]) is None

    def test_perfect_positive_correlation(self):
        c = _safe_pearson([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        assert c == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        c = _safe_pearson([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
        assert c == pytest.approx(-1.0)

    def test_zero_correlation(self):
        # Symmetric scatter
        x = [1, 2, 3, 4, 5]
        y = [3, 1, 4, 1, 5]
        c = _safe_pearson(x, y)
        assert c is not None
        assert -0.5 < c < 0.5


# ── Per-team aggregator ─────────────────────────────────────────────────────


class TestTeamRankQuality:
    def test_no_data_returns_zeros(self, conn):
        result = _team_rank_quality(
            conn, team_id="technology",
            start_date="2026-04-01", end_date="2026-05-01",
        )
        assert result["n_obs"] == 0
        assert result["rank_corr"] is None
        assert result["score_corr"] is None
        assert result["n_obs_21d"] == 0
        assert result["rank_corr_21d"] is None

    def test_skilled_team_negative_correlation(self, conn):
        # Rank 1 → best return; rank 5 → worst — skilled scorer.
        _seed(conn, team_id="financials", eval_date="2026-04-25", picks=[
            (1, "A", 80, 0.05, 1),
            (2, "B", 75, 0.03, 1),
            (3, "C", 70, 0.01, 0),
            (4, "D", 65, -0.02, 0),
            (5, "E", 60, -0.04, 0),
        ])
        result = _team_rank_quality(
            conn, team_id="financials",
            start_date="2026-04-01", end_date="2026-05-01",
        )
        assert result["rank_corr"] is not None
        assert result["rank_corr"] < -0.9  # near-perfect negative

    def test_anti_skill_team_positive_correlation(self, conn):
        # Rank 1 → worst return; rank 5 → best — anti-skill (the
        # 2026-05-09 healthcare/tech case).
        _seed(conn, team_id="technology", eval_date="2026-04-25", picks=[
            (1, "A", 80, -0.04, 0),
            (2, "B", 75, -0.02, 0),
            (3, "C", 70, 0.01, 0),
            (4, "D", 65, 0.03, 1),
            (5, "E", 60, 0.05, 1),
        ])
        result = _team_rank_quality(
            conn, team_id="technology",
            start_date="2026-04-01", end_date="2026-05-01",
        )
        assert result["rank_corr"] is not None
        assert result["rank_corr"] > 0.9  # near-perfect positive

    def test_top3_hit_rate_computed(self, conn):
        _seed(conn, team_id="financials", eval_date="2026-04-25", picks=[
            (1, "A", 80, 0.05, 1),
            (2, "B", 75, 0.03, 1),
            (3, "C", 70, 0.01, 0),
            (4, "D", 65, -0.02, 0),
            (5, "E", 60, -0.04, 0),
        ])
        result = _team_rank_quality(
            conn, team_id="financials",
            start_date="2026-04-01", end_date="2026-05-01",
        )
        # Top-3: 2 of 3 beat SPY
        assert result["hit_rate_top3"] == pytest.approx(66.67, abs=0.5)
        assert result["n_top3"] == 3

    def test_window_filtering(self, conn):
        # Rows outside window must be excluded
        _seed(conn, team_id="financials", eval_date="2026-03-01", picks=[
            (1, "OLD", 80, 0.10, 1),
        ])
        _seed(conn, team_id="financials", eval_date="2026-04-25", picks=[
            (1, "A", 80, 0.05, 1),
            (2, "B", 75, 0.03, 1),
            (3, "C", 70, 0.01, 0),
        ])
        result = _team_rank_quality(
            conn, team_id="financials",
            start_date="2026-04-01", end_date="2026-05-01",
        )
        assert result["n_obs"] == 3  # OLD excluded
        assert result["n_dates"] == 1

    def test_primary_horizon_diverges_from_diagnostic(self, conn):
        """Seed a team skilled at 5d but anti-skilled at 21d — each
        horizon's correlation must reflect ITS OWN outcome column."""
        _seed(conn, team_id="technology", eval_date="2026-04-25", picks=[
            # (rank, ticker, score, ret5, beat5, ret21, beat21):
            # 5d: rank 1 best → skilled; 21d: rank 1 worst → anti-skilled.
            (1, "A", 80, 0.05, 1, -0.08, 0),
            (2, "B", 75, 0.03, 1, -0.04, 0),
            (3, "C", 70, 0.01, 0, 0.01, 0),
            (4, "D", 65, -0.02, 0, 0.05, 1),
            (5, "E", 60, -0.04, 0, 0.09, 1),
        ])
        result = _team_rank_quality(
            conn, team_id="technology",
            start_date="2026-04-01", end_date="2026-05-01",
        )
        assert result["rank_corr"] < -0.9       # 5d skilled
        assert result["rank_corr_21d"] > 0.9    # 21d anti-skilled
        # Top-3 hit rates likewise per-horizon: 5d 2/3, 21d 0/3.
        assert result["hit_rate_top3"] == pytest.approx(66.67, abs=0.5)
        assert result["hit_rate_top3_21d"] == pytest.approx(0.0)

    def test_primary_horizon_resolution_lag(self, conn):
        """Rows whose 21d outcome is still NULL (resolution lag) count for
        the 5d channel but not the 21d one."""
        _seed(conn, team_id="financials", eval_date="2026-04-25", picks=[
            (1, "A", 80, 0.05, 1, 0.06, 1),
            (2, "B", 75, 0.03, 1, 0.02, 0),
            (3, "C", 70, 0.01, 0, 0.01, 0),
            (4, "D", 65, -0.02, 0, None, None),  # 21d unresolved
            (5, "E", 60, -0.04, 0, None, None),  # 21d unresolved
        ])
        result = _team_rank_quality(
            conn, team_id="financials",
            start_date="2026-04-01", end_date="2026-05-01",
        )
        assert result["n_obs"] == 5
        assert result["n_obs_21d"] == 3
        assert result["rank_corr"] is not None
        assert result["rank_corr_21d"] is not None


# ── Top-level entry point ───────────────────────────────────────────────────


class TestComputeQuantRankQuality:
    def test_no_team_candidates_table(self):
        c = sqlite3.connect(":memory:")  # empty schema
        result = compute_quant_rank_quality(
            db_conn=c, run_date="2026-05-09", emit_metrics=False,
        )
        assert result["status"] == "no_data"
        assert "team_candidates table missing" in result["reason"]

    def test_invalid_run_date(self, conn):
        result = compute_quant_rank_quality(
            db_conn=conn, run_date="not-a-date", emit_metrics=False,
        )
        assert result["status"] == "error"

    def test_must_provide_db(self):
        result = compute_quant_rank_quality(emit_metrics=False)
        assert result["status"] == "error"
        assert "db_path or db_conn" in result["error"]

    def test_empty_window_returns_no_data(self, conn):
        # Schema exists but no rows
        result = compute_quant_rank_quality(
            db_conn=conn, run_date="2026-05-09", emit_metrics=False,
        )
        assert result["status"] == "no_data"

    def test_anti_skill_team_flagged(self, conn):
        # Tech anti-skill (rank correlates positive with return)
        _seed(conn, team_id="technology", eval_date="2026-05-02", picks=[
            (1, "A", 80, -0.04, 0),
            (2, "B", 75, -0.02, 0),
            (3, "C", 70, 0.01, 0),
            (4, "D", 65, 0.03, 1),
            (5, "E", 60, 0.05, 1),
        ])
        # Financials skilled (negative correlation)
        _seed(conn, team_id="financials", eval_date="2026-05-02", picks=[
            (1, "F", 80, 0.05, 1),
            (2, "G", 75, 0.03, 1),
            (3, "H", 70, 0.01, 0),
            (4, "I", 65, -0.02, 0),
            (5, "J", 60, -0.04, 0),
        ])
        result = compute_quant_rank_quality(
            db_conn=conn, run_date="2026-05-09", emit_metrics=False,
        )
        assert result["status"] == "ok"
        assert "technology" in result["anti_skill_teams"]
        assert "financials" not in result["anti_skill_teams"]

    def test_anti_skill_flagged_per_horizon(self, conn):
        """A team anti-skilled ONLY at the canonical horizon lands in
        anti_skill_teams_21d but not the diagnostic list."""
        _seed(conn, team_id="technology", eval_date="2026-05-02", picks=[
            # 5d skilled, 21d anti-skilled.
            (1, "A", 80, 0.05, 1, -0.08, 0),
            (2, "B", 75, 0.03, 1, -0.04, 0),
            (3, "C", 70, 0.01, 0, 0.01, 0),
            (4, "D", 65, -0.02, 0, 0.05, 1),
            (5, "E", 60, -0.04, 0, 0.09, 1),
        ])
        result = compute_quant_rank_quality(
            db_conn=conn, run_date="2026-05-09", emit_metrics=False,
        )
        assert result["status"] == "ok"
        assert "technology" not in result["anti_skill_teams"]
        assert "technology" in result["anti_skill_teams_21d"]

    def test_horizon_metadata_declared(self, conn):
        _seed(conn, team_id="technology", eval_date="2026-05-02", picks=[
            (1, "A", 80, 0.01, 0), (2, "B", 75, 0.02, 1),
            (3, "C", 70, 0.03, 1),
        ])
        result = compute_quant_rank_quality(
            db_conn=conn, run_date="2026-05-09", emit_metrics=False,
        )
        assert result["status"] == "ok"
        assert result["diagnostic_horizon_days"] == 5
        assert result["primary_horizon_days"] == 21
        assert "overall_rank_corr_21d" in result
        assert "n_total_obs_21d" in result

    def test_per_team_dict_includes_all_canonical_sectors(self, conn):
        _seed(conn, team_id="technology", eval_date="2026-05-02", picks=[
            (1, "A", 80, 0.01, 0), (2, "B", 75, 0.02, 1),
            (3, "C", 70, 0.03, 1),
        ])
        result = compute_quant_rank_quality(
            db_conn=conn, run_date="2026-05-09", emit_metrics=False,
        )
        assert result["status"] == "ok"
        per_team_ids = {e["team_id"] for e in result["per_team"]}
        assert per_team_ids == set(CANONICAL_SECTORS)

    def test_missing_primary_columns_degrade_not_crash(self, conn_legacy_schema):
        """Pre-canonical universe_returns (no 21d columns): 5d channel keeps
        working; 21d fields degrade to None (with a WARN, not a crash)."""
        _seed_legacy(
            conn_legacy_schema, team_id="financials", eval_date="2026-05-02",
            picks=[
                (1, "A", 80, 0.05, 1), (2, "B", 75, 0.03, 1),
                (3, "C", 70, 0.01, 0), (4, "D", 65, -0.02, 0),
                (5, "E", 60, -0.04, 0),
            ],
        )
        result = compute_quant_rank_quality(
            db_conn=conn_legacy_schema, run_date="2026-05-09",
            emit_metrics=False,
        )
        assert result["status"] == "ok"
        per_team = {e["team_id"]: e for e in result["per_team"]}
        assert per_team["financials"]["rank_corr"] < -0.9
        assert per_team["financials"]["rank_corr_21d"] is None
        assert per_team["financials"]["n_obs_21d"] == 0
        assert result["overall_rank_corr_21d"] is None
        assert result["anti_skill_teams_21d"] == []

    def test_emits_cw_metrics_per_team(self, conn):
        _seed(conn, team_id="technology", eval_date="2026-05-02", picks=[
            (1, "A", 80, -0.04, 0), (2, "B", 75, 0.02, 1),
            (3, "C", 70, 0.05, 1),
        ])
        cw = MagicMock()
        result = compute_quant_rank_quality(
            db_conn=conn, run_date="2026-05-09",
            cloudwatch_client=cw,
        )
        assert result["status"] == "ok"
        assert cw.put_metric_data.called
        # Expected metric names: the legacy diagnostic trio plus the
        # primary-horizon (_21d-suffixed) variants (config#1529).
        all_metrics = []
        for call in cw.put_metric_data.call_args_list:
            all_metrics.extend(call.kwargs["MetricData"])
        names = {m["MetricName"] for m in all_metrics}
        assert "rank_corr_5d" in names
        assert "score_corr_5d" in names
        assert "hit_rate_top3" in names
        assert "rank_corr_21d" in names
        assert "score_corr_21d" in names
        assert "hit_rate_top3_21d" in names

    def test_cw_emission_failure_does_not_break(self, conn):
        _seed(conn, team_id="technology", eval_date="2026-05-02", picks=[
            (1, "A", 80, -0.04, 0), (2, "B", 75, 0.02, 1),
            (3, "C", 70, 0.05, 1),
        ])
        cw = MagicMock()
        cw.put_metric_data.side_effect = RuntimeError("CW unreachable")
        result = compute_quant_rank_quality(
            db_conn=conn, run_date="2026-05-09",
            cloudwatch_client=cw,
        )
        assert result["status"] == "ok"  # JSON path still completes

    def test_emit_metrics_false_skips_cw(self, conn):
        _seed(conn, team_id="technology", eval_date="2026-05-02", picks=[
            (1, "A", 80, -0.04, 0), (2, "B", 75, 0.02, 1),
            (3, "C", 70, 0.05, 1),
        ])
        cw = MagicMock()
        compute_quant_rank_quality(
            db_conn=conn, run_date="2026-05-09",
            cloudwatch_client=cw, emit_metrics=False,
        )
        cw.put_metric_data.assert_not_called()

    def test_replay_2026_05_09_pattern(self, conn):
        """Replay of the 2026-05-09 evaluator-email pattern: tech +0.33+
        rank_corr (anti-skill), financials -0.09 (mildly skilled)."""
        # Tech anti-skill — bottom ranks beat top ranks
        _seed(conn, team_id="technology", eval_date="2026-04-25", picks=[
            (1, "T1", 80, -0.01, 0), (2, "T2", 75, 0.00, 0),
            (3, "T3", 70, 0.02, 1), (4, "T4", 65, 0.04, 1),
            (5, "T5", 60, 0.17, 1),
        ])
        # Financials marginally skilled
        _seed(conn, team_id="financials", eval_date="2026-04-25", picks=[
            (1, "F1", 80, 0.03, 1), (2, "F2", 75, 0.01, 0),
            (3, "F3", 70, 0.02, 1), (4, "F4", 65, -0.01, 0),
            (5, "F5", 60, -0.02, 0),
        ])
        result = compute_quant_rank_quality(
            db_conn=conn, run_date="2026-05-02", emit_metrics=False,
        )
        per_team = {e["team_id"]: e for e in result["per_team"]}
        assert per_team["technology"]["rank_corr"] > ANTI_SKILL_THRESHOLD
        assert per_team["financials"]["rank_corr"] < 0
        assert "technology" in result["anti_skill_teams"]


# ── Reporter section renders both horizons ──────────────────────────────────


class TestReporterSection:
    def test_section_renders_both_horizons(self, conn):
        from reporter import _section_quant_rank_quality
        _seed(conn, team_id="financials", eval_date="2026-05-02", picks=[
            (1, "A", 80, 0.05, 1, -0.08, 0),
            (2, "B", 75, 0.03, 1, -0.04, 0),
            (3, "C", 70, 0.01, 0, 0.01, 0),
            (4, "D", 65, -0.02, 0, 0.05, 1),
            (5, "E", 60, -0.04, 0, 0.09, 1),
        ])
        result = compute_quant_rank_quality(
            db_conn=conn, run_date="2026-05-09", emit_metrics=False,
        )
        lines = _section_quant_rank_quality(result)
        text = "\n".join(lines)
        assert "Rank corr 5d" in text
        assert "Rank corr 21d" in text
        assert "21d, canonical" in text
        # 21d anti-skill line surfaces (financials anti-skilled at 21d only)
        assert "Anti-skill teams (21d corr" in text
