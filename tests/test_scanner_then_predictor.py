"""Tests for the scanner -> research-free predictor direct counterfactual
(config#1405, arm 4 of the agentic-ablation ladder).

The fixture makes realized 21d alpha selection-driven (winners = low ``i``),
has the research-free predictor's ``predicted_alpha`` favour the winners, and
has the live agentic CIO ADVANCE the losers — so the count-matched
``scanner_then_predictor_topN`` must beat BOTH the actual scanner pass pool and
the agentic CIO selection. Mirrors ``test_scanner_factor_counterfactual.py``.

The meta-ensemble backfill that populates ``predictor_outcomes_research_free``
runs only on the Saturday spot box (ArcticDB-gated); this exercises the
analysis/consumer layer with a synthetic fixture, no ArcticDB needed.
"""

from __future__ import annotations

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.end_to_end import _scanner_then_predictor_topN  # noqa: E402


def _db(tmp_path, *, cio_decision_for=lambda i: "ADVANCE" if i >= 7 else "REJECT"):
    conn = sqlite3.connect(str(tmp_path / "r.db"))
    conn.execute(
        "CREATE TABLE scanner_evaluations (ticker TEXT, eval_date TEXT, quant_filter_pass INTEGER)"
    )
    conn.execute(
        "CREATE TABLE universe_returns (ticker TEXT, eval_date TEXT, sector TEXT, "
        "log_return_21d REAL, log_spy_return_21d REAL)"
    )
    conn.execute(
        "CREATE TABLE predictor_outcomes_research_free (ticker TEXT, prediction_date TEXT, "
        "predicted_alpha REAL, n_research_features_missing INTEGER)"
    )
    conn.execute("CREATE TABLE cio_evaluations (ticker TEXT, eval_date TEXT, cio_decision TEXT)")
    dates = ["2026-01-02", "2026-01-09", "2026-01-16", "2026-01-23"]
    for d in dates:
        for i in range(10):
            alpha = 0.03 if i < 5 else -0.03  # winners = low i
            # the live scanner passes the WHOLE pool (10/cycle) -> the predictor re-ranks it
            conn.execute("INSERT INTO scanner_evaluations VALUES (?,?,?)", (f"T{i}", d, 1))
            conn.execute(
                "INSERT INTO universe_returns VALUES (?,?,?,?,?)", (f"T{i}", d, "Tech", alpha, 0.0)
            )
            # research-free predicted_alpha favours the winners (low i -> high score);
            # 4 research meta-features omitted -> n_research_features_missing = 4
            conn.execute(
                "INSERT INTO predictor_outcomes_research_free VALUES (?,?,?,?)",
                (f"T{i}", d, float(10 - i), 4),
            )
            # the live agentic CIO advances the LOSERS (high i) -> agentic underperforms
            conn.execute(
                "INSERT INTO cio_evaluations VALUES (?,?,?)", (f"T{i}", d, cio_decision_for(i))
            )
    conn.commit()
    return conn


def test_predictor_beats_agentic_and_scanner(tmp_path):
    conn = _db(tmp_path)
    r = _scanner_then_predictor_topN(conn)
    assert r["status"] == "ok", r
    assert r["n_cycles"] == 4, r
    m = r["methods"]
    # research-free predictor picks the winners -> positive
    assert m["scanner_then_predictor_topN"]["mean_alpha_21d"] > 0, r
    # the live agentic CIO advanced the losers -> negative (the path being replaced)
    assert m["agentic_cio_advance"]["mean_alpha_21d"] < 0, r
    # the scanner pass pool is a 50/50 mix -> ~0
    assert abs(m["actual_scanner_pass"]["mean_alpha_21d"]) < 1e-6, r
    # both lifts positive
    assert m["scanner_then_predictor_topN"]["lift_vs_actual_scanner"] > 0, r
    assert m["scanner_then_predictor_topN"]["lift_vs_agentic_cio"] > 0, r
    assert m["scanner_then_predictor_topN"]["sn_lift_vs_actual_scanner"] is not None, r
    assert r["predictor_beats_agentic_cio"] is True, r
    assert r["predictor_beats_actual_scanner"] is True, r
    # count-match: 3 advance/cycle x 4 = 12 predictor picks; agentic 12; scanner pool 40
    assert m["scanner_then_predictor_topN"]["n_picks"] == 12, r
    assert m["agentic_cio_advance"]["n_picks"] == 12, r
    assert m["actual_scanner_pass"]["n_picks"] == 40, r
    # research-free guard: every prediction omitted the 4 research meta-features
    assert r["research_features_missing_mode"] == 4, r


def test_advance_forced_counts_as_agentic(tmp_path):
    """``ADVANCE_FORCED`` (the force-fill path) is part of the agentic selection."""
    conn = _db(tmp_path, cio_decision_for=lambda i: "ADVANCE_FORCED" if i >= 7 else "REJECT")
    r = _scanner_then_predictor_topN(conn)
    assert r["status"] == "ok", r
    assert r["methods"]["agentic_cio_advance"]["n_picks"] == 12, r
    assert r["predictor_beats_agentic_cio"] is True, r


def test_skipped_without_predictions_table(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "e.db"))
    conn.execute(
        "CREATE TABLE scanner_evaluations (ticker TEXT, eval_date TEXT, quant_filter_pass INTEGER)"
    )
    conn.execute(
        "CREATE TABLE universe_returns (ticker TEXT, eval_date TEXT, sector TEXT, "
        "log_return_21d REAL, log_spy_return_21d REAL)"
    )
    conn.execute("CREATE TABLE cio_evaluations (ticker TEXT, eval_date TEXT, cio_decision TEXT)")
    conn.commit()
    r = _scanner_then_predictor_topN(conn)
    assert r["status"] == "skipped", r
    assert "predictor_outcomes_research_free" in r["reason"], r


def test_skipped_without_predicted_alpha_column(tmp_path):
    conn = _db(tmp_path)
    conn.execute("DROP TABLE predictor_outcomes_research_free")
    conn.execute("CREATE TABLE predictor_outcomes_research_free (ticker TEXT, prediction_date TEXT)")
    conn.commit()
    r = _scanner_then_predictor_topN(conn)
    assert r["status"] == "skipped", r
    assert "predicted_alpha" in r["reason"], r


def test_skipped_when_no_predictions_match(tmp_path):
    """Table exists but the backfill hasn't populated rows yet -> honest skip."""
    conn = _db(tmp_path)
    conn.execute("DELETE FROM predictor_outcomes_research_free")
    conn.commit()
    r = _scanner_then_predictor_topN(conn)
    assert r["status"] == "skipped", r


def test_insufficient_without_cio_advance(tmp_path):
    """No agentic ADVANCE anywhere -> no count-match basis -> insufficient_data."""
    conn = _db(tmp_path, cio_decision_for=lambda i: "REJECT")
    r = _scanner_then_predictor_topN(conn)
    assert r["status"] == "insufficient_data", r
