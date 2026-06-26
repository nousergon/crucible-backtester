"""Tests for analysis.cio_rule_tag_precision.

Per-rule-tag precision of the LLM CIO's gates. Consumes
``cio_evaluations.rule_tags`` (research schema migration 14, persisted from
crucible-research#152) joined to ``universe_returns.beat_spy_5d`` on
``(ticker, eval_date)``. Per tag: n_decisions, ADVANCE precision (% of a
tag's ADVANCEs that beat SPY at 5d), REJECT-beat rate (per-tag false-negative
rate).

Locked behavior:
- rule_tags parsed from the persisted JSON list[str]; NULL rows skipped
- ADVANCE and ADVANCE_FORCED both count as advances
- advance_precision / reject_beat_rate are None when their denominator is 0
  (a tag with zero ADVANCEs, a tag with zero REJECTs)
- a tag whose ADVANCEs all beat SPY → precision 1.0
- per-tag counts fold across rows; one decision counts a tag once
- window filtering on eval_date
- insufficient_data gate below MIN_TAGGED_DECISIONS
- missing tables / pre-migration-14 column → no_data, no crash
- alarm fires when reject_beat_rate exceeds REJECT_BEAT_ALARM
- CW emission injectable for tests
"""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import MagicMock

import pytest

from analysis.cio_rule_tag_precision import (
    MIN_TAGGED_DECISIONS,
    REJECT_BEAT_ALARM,
    _accumulate,
    _parse_rule_tags,
    compute_cio_rule_tag_precision,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def conn():
    """In-memory DB with the minimal cio_evaluations + universe_returns schema."""
    c = sqlite3.connect(":memory:")
    c.executescript(
        """
        CREATE TABLE cio_evaluations (
            id INTEGER PRIMARY KEY,
            ticker TEXT, eval_date TEXT, team_id TEXT,
            cio_decision TEXT, rule_tags TEXT,
            UNIQUE(ticker, eval_date)
        );
        CREATE TABLE universe_returns (
            id INTEGER PRIMARY KEY,
            ticker TEXT, eval_date TEXT,
            return_5d REAL, beat_spy_5d INTEGER,
            UNIQUE(ticker, eval_date)
        );
        """
    )
    yield c
    c.close()


def _seed(conn, *, ticker, eval_date, decision, tags, beat):
    """Insert one matched cio_evaluations + universe_returns row.

    tags: list[str] | None (None → persisted NULL rule_tags). beat: 1/0/None.
    """
    conn.execute(
        "INSERT INTO cio_evaluations (ticker, eval_date, cio_decision, rule_tags) "
        "VALUES (?,?,?,?)",
        (ticker, eval_date, decision, json.dumps(tags) if tags is not None else None),
    )
    conn.execute(
        "INSERT INTO universe_returns (ticker, eval_date, beat_spy_5d) VALUES (?,?,?)",
        (ticker, eval_date, beat),
    )


def _seed_many(conn, *, eval_date, decision, tags, beat, n, prefix):
    """Seed n distinct matched rows sharing decision/tags/beat (count padding)."""
    for i in range(n):
        _seed(
            conn, ticker=f"{prefix}{i}", eval_date=eval_date,
            decision=decision, tags=tags, beat=beat,
        )


# ── rule_tags parsing ───────────────────────────────────────────────────────


class TestParseRuleTags:
    def test_null_returns_empty(self):
        assert _parse_rule_tags(None) == []

    def test_blank_returns_empty(self):
        assert _parse_rule_tags("") == []
        assert _parse_rule_tags("   ") == []

    def test_json_list(self):
        assert _parse_rule_tags('["rr_asymmetry", "qual_veto"]') == [
            "rr_asymmetry", "qual_veto",
        ]

    def test_malformed_json_returns_empty(self):
        assert _parse_rule_tags("not json") == []
        assert _parse_rule_tags("{not: a list}") == []

    def test_non_list_payload_returns_empty(self):
        assert _parse_rule_tags('"just_a_string"') == []
        assert _parse_rule_tags("42") == []

    def test_dedupes_within_row(self):
        assert _parse_rule_tags('["a", "a", "b"]') == ["a", "b"]

    def test_drops_non_string_and_blank_items(self):
        assert _parse_rule_tags('["a", 7, "", "  ", "b"]') == ["a", "b"]

    def test_accepts_native_list(self):
        assert _parse_rule_tags(["a", "b"]) == ["a", "b"]


# ── Accumulator ─────────────────────────────────────────────────────────────


class TestAccumulate:
    def test_skips_null_outcome(self):
        rows = [("ADVANCE", '["t"]', None)]
        assert _accumulate(rows) == {}

    def test_skips_null_tags(self):
        rows = [("ADVANCE", None, 1)]
        assert _accumulate(rows) == {}

    def test_advance_forced_counts_as_advance(self):
        rows = [("ADVANCE_FORCED", '["t"]', 1)]
        agg = _accumulate(rows)
        assert agg["t"]["n_advance"] == 1
        assert agg["t"]["advance_beat"] == 1

    def test_other_verdict_counted_but_not_in_rates(self):
        # HOLD is neither advance nor reject: in n_decisions, not in either rate.
        rows = [("HOLD", '["t"]', 1)]
        agg = _accumulate(rows)
        assert agg["t"]["n_decisions"] == 1
        assert agg["t"]["n_advance"] == 0
        assert agg["t"]["n_reject"] == 0


# ── Top-level entry point ───────────────────────────────────────────────────


class TestComputeCioRuleTagPrecision:
    def test_missing_tables(self):
        c = sqlite3.connect(":memory:")
        res = compute_cio_rule_tag_precision(
            db_conn=c, run_date="2026-06-06", emit_metrics=False,
        )
        assert res["status"] == "no_data"
        assert "cio_evaluations table missing" in res["reason"]
        c.close()

    def test_pre_migration_14_column_missing(self):
        c = sqlite3.connect(":memory:")
        c.executescript(
            "CREATE TABLE cio_evaluations (ticker TEXT, eval_date TEXT, cio_decision TEXT);"
            "CREATE TABLE universe_returns (ticker TEXT, eval_date TEXT, beat_spy_5d INTEGER);"
        )
        res = compute_cio_rule_tag_precision(
            db_conn=c, run_date="2026-06-06", emit_metrics=False,
        )
        assert res["status"] == "no_data"
        assert "rule_tags column missing" in res["reason"]
        c.close()

    def test_null_rule_tags_rows_skipped(self, conn):
        # All rows untagged (legacy) → no_data, not a crash / false positive.
        _seed_many(
            conn, eval_date="2026-06-01", decision="ADVANCE",
            tags=None, beat=1, n=30, prefix="L",
        )
        res = compute_cio_rule_tag_precision(
            db_conn=conn, run_date="2026-06-06", emit_metrics=False,
        )
        assert res["status"] == "no_data"

    def test_insufficient_data_gate(self, conn):
        # A handful of tagged rows, below the floor → insufficient_data.
        _seed_many(
            conn, eval_date="2026-06-01", decision="ADVANCE",
            tags=["rr_asymmetry"], beat=1, n=3, prefix="A",
        )
        res = compute_cio_rule_tag_precision(
            db_conn=conn, run_date="2026-06-06", emit_metrics=False,
        )
        assert res["status"] == "insufficient_data"
        assert res["n_tagged_decisions"] == 3
        assert res["min_tagged_decisions"] == MIN_TAGGED_DECISIONS

    def test_precision_and_n_decisions(self, conn):
        # Build a clean scenario with min_tagged_decisions lowered so the
        # gate doesn't swallow it. Two tags:
        #   rr_asymmetry: 4 ADVANCEs, 3 beat SPY → precision 0.75
        #   qual_veto:    2 ADVANCEs, both beat   → precision 1.0 (all-correct)
        _seed(conn, ticker="A1", eval_date="2026-06-01", decision="ADVANCE",
              tags=["rr_asymmetry"], beat=1)
        _seed(conn, ticker="A2", eval_date="2026-06-01", decision="ADVANCE",
              tags=["rr_asymmetry"], beat=1)
        _seed(conn, ticker="A3", eval_date="2026-06-01", decision="ADVANCE",
              tags=["rr_asymmetry"], beat=1)
        _seed(conn, ticker="A4", eval_date="2026-06-01", decision="ADVANCE",
              tags=["rr_asymmetry"], beat=0)
        _seed(conn, ticker="Q1", eval_date="2026-06-01", decision="ADVANCE_FORCED",
              tags=["qual_veto"], beat=1)
        _seed(conn, ticker="Q2", eval_date="2026-06-01", decision="ADVANCE",
              tags=["qual_veto"], beat=1)

        res = compute_cio_rule_tag_precision(
            db_conn=conn, run_date="2026-06-06",
            min_tagged_decisions=1, emit_metrics=False,
        )
        assert res["status"] == "ok"
        by_tag = {e["rule_tag"]: e for e in res["per_tag"]}

        assert by_tag["rr_asymmetry"]["n_decisions"] == 4
        assert by_tag["rr_asymmetry"]["n_advance"] == 4
        assert by_tag["rr_asymmetry"]["advance_precision"] == pytest.approx(0.75)

        assert by_tag["qual_veto"]["n_decisions"] == 2
        assert by_tag["qual_veto"]["advance_precision"] == pytest.approx(1.0)

        # Pooled: 5 of 6 ADVANCE-class beat SPY.
        assert res["overall_advance_precision"] == pytest.approx(5 / 6, abs=1e-4)
        # Per-tag sort: rr_asymmetry (4) before qual_veto (2).
        assert res["per_tag"][0]["rule_tag"] == "rr_asymmetry"

    def test_tag_with_zero_advances(self, conn):
        # A tag that only ever appears on REJECTs → advance_precision is None,
        # not 0 (the edge the issue calls out).
        _seed_many(
            conn, eval_date="2026-06-01", decision="REJECT",
            tags=["macro_alignment"], beat=0, n=5, prefix="R",
        )
        res = compute_cio_rule_tag_precision(
            db_conn=conn, run_date="2026-06-06",
            min_tagged_decisions=1, emit_metrics=False,
        )
        assert res["status"] == "ok"
        tag = res["per_tag"][0]
        assert tag["rule_tag"] == "macro_alignment"
        assert tag["n_advance"] == 0
        assert tag["advance_precision"] is None
        assert tag["n_reject"] == 5
        # All 5 REJECTs did NOT beat SPY → reject_beat_rate 0.0 (a correct gate).
        assert tag["reject_beat_rate"] == pytest.approx(0.0)

    def test_reject_beat_alarm_fires(self, conn):
        # A gate that rejected names that mostly went on to beat SPY:
        # 3 of 4 REJECTs beat → reject_beat_rate 0.75 > 0.50 alarm.
        _seed(conn, ticker="R1", eval_date="2026-06-01", decision="REJECT",
              tags=["qual_veto"], beat=1)
        _seed(conn, ticker="R2", eval_date="2026-06-01", decision="REJECT",
              tags=["qual_veto"], beat=1)
        _seed(conn, ticker="R3", eval_date="2026-06-01", decision="REJECT",
              tags=["qual_veto"], beat=1)
        _seed(conn, ticker="R4", eval_date="2026-06-01", decision="REJECT",
              tags=["qual_veto"], beat=0)
        res = compute_cio_rule_tag_precision(
            db_conn=conn, run_date="2026-06-06",
            min_tagged_decisions=1, emit_metrics=False,
        )
        assert res["status"] == "ok"
        tag = res["per_tag"][0]
        assert tag["reject_beat_rate"] == pytest.approx(0.75)
        assert "qual_veto" in res["alarm_tags"]
        assert res["reject_beat_alarm"] == REJECT_BEAT_ALARM

    def test_window_filtering(self, conn):
        # Rows outside the lookback window are excluded.
        _seed_many(
            conn, eval_date="2026-01-01", decision="ADVANCE",
            tags=["rr_asymmetry"], beat=1, n=10, prefix="OLD",
        )
        _seed_many(
            conn, eval_date="2026-06-01", decision="ADVANCE",
            tags=["rr_asymmetry"], beat=1, n=2, prefix="NEW",
        )
        res = compute_cio_rule_tag_precision(
            db_conn=conn, run_date="2026-06-06",
            lookback_weeks=8, min_tagged_decisions=1, emit_metrics=False,
        )
        assert res["status"] == "ok"
        # Only the 2 in-window rows counted; the Jan rows fall before the window.
        assert res["n_tagged_decisions"] == 2

    def test_cloudwatch_emission_injectable(self, conn):
        _seed_many(
            conn, eval_date="2026-06-01", decision="ADVANCE",
            tags=["rr_asymmetry"], beat=1, n=5, prefix="A",
        )
        cw = MagicMock()
        res = compute_cio_rule_tag_precision(
            db_conn=conn, run_date="2026-06-06", min_tagged_decisions=1,
            cloudwatch_client=cw, emit_metrics=True,
        )
        assert res["status"] == "ok"
        assert cw.put_metric_data.called

    def test_missing_db_args(self):
        res = compute_cio_rule_tag_precision(run_date="2026-06-06")
        assert res["status"] == "error"
