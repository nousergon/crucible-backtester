"""Tests for the canonical 21d-horizon research-edge lift (ROADMAP L4551).

The selectors (scanner / sector teams / CIO) pick 21-day theses but were graded
on a 5-day window, collapsing precision toward the base rate. These tests build
a research.db where the 5d outcome is uncorrelated with selection while the 21d
outcome cleanly rewards the selected names, and assert the additive
``classification_21d`` + ``lift_21d_log`` blocks reflect the 21d edge.
"""

import sqlite3

import pytest

from analysis.end_to_end import compute_lift_metrics

DATE = "2026-05-01"


def _build_research_db(tmp_path):
    db = tmp_path / "research.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE universe_returns ("
        "ticker TEXT, eval_date TEXT, sector TEXT, "
        "return_5d REAL, spy_return_5d REAL, beat_spy_5d INTEGER, "
        "return_21d REAL, spy_return_21d REAL, beat_spy_21d INTEGER, "
        "log_return_21d REAL, log_spy_return_21d REAL)"
    )
    conn.execute("CREATE TABLE scanner_evaluations (ticker TEXT, eval_date TEXT, quant_filter_pass INTEGER)")
    conn.execute("CREATE TABLE team_candidates (ticker TEXT, eval_date TEXT, team_id TEXT, team_recommended INTEGER)")
    conn.execute("CREATE TABLE cio_evaluations (ticker TEXT, eval_date TEXT, cio_decision TEXT, final_score REAL, cio_conviction REAL)")
    # Empty — _predictor_lift queries it; the read must not error on a missing table.
    conn.execute("CREATE TABLE predictor_outcomes (symbol TEXT, prediction_date TEXT, "
                 "predicted_direction TEXT, prediction_confidence REAL)")

    # 20 names. Selected = first 10. 5d beat alternates (uncorrelated w/ select);
    # 21d beat = 1 iff selected (clean 21d edge). 21d log alpha +0.05 selected,
    # -0.02 unselected.
    for i in range(20):
        t = f"T{i:02d}"
        selected = i < 10
        beat_5d = i % 2  # 0/1 alternating, independent of `selected`
        beat_21d = 1 if selected else 0
        log_ret = (0.05 if selected else -0.02)
        conn.execute(
            "INSERT INTO universe_returns VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (t, DATE, "Tech", 1.0, 0.5, beat_5d, 2.0, 1.0, beat_21d, log_ret, 0.0),
        )
        conn.execute("INSERT INTO scanner_evaluations VALUES (?,?,?)", (t, DATE, 1 if selected else 0))
        conn.execute("INSERT INTO team_candidates VALUES (?,?,?,?)", (t, DATE, "tech", 1 if selected else 0))
        conn.execute(
            "INSERT INTO cio_evaluations VALUES (?,?,?,?,?)",
            (t, DATE, "ADVANCE" if selected else "REJECT", 70.0 if selected else 40.0,
             75.0 if selected else 45.0),
        )
    conn.commit()
    conn.close()
    return str(db)


def test_scanner_21d_block_present_and_perfect(tmp_path):
    out = compute_lift_metrics(_build_research_db(tmp_path))
    sl = out["scanner_lift"]
    # Legacy 5d precision is at the base rate (selection ⊥ 5d outcome) ~0.5...
    assert sl["classification"]["precision"] == 0.5
    # ...but the 21d classification reflects the real edge: every selected name
    # beat at 21d → precision 1.0.
    assert sl["classification_21d"]["precision"] == 1.0
    assert sl["lift_21d_log"]["lift"] > 0  # selected 21d alpha > universe


def test_cio_21d_block_present(tmp_path):
    out = compute_lift_metrics(_build_research_db(tmp_path))
    cl = out["cio_lift"]
    assert cl["classification"]["precision"] == 0.5
    assert cl["classification_21d"]["precision"] == 1.0


def test_cio_selection_skill_block(tmp_path):
    # Fixture: ADVANCE names have +0.05 21d log-alpha, REJECT -0.02 → positive
    # selection gap; conviction (75 vs 45) tracks alpha → positive IC. (L4561)
    out = compute_lift_metrics(_build_research_db(tmp_path))
    sel = out["cio_lift"]["selection_skill_21d"]
    assert sel is not None
    assert sel["advance_alpha_21d"] == pytest.approx(0.05)
    assert sel["reject_alpha_21d"] == pytest.approx(-0.02)
    assert sel["selection_gap_21d"] == pytest.approx(0.07)
    assert sel["n_advance"] == 10 and sel["n_reject"] == 10
    assert sel["conviction_ic_21d"] is not None and sel["conviction_ic_21d"] > 0


def test_team_21d_block_present(tmp_path):
    out = compute_lift_metrics(_build_research_db(tmp_path))
    team = out["team_lift"][0]
    assert team["classification_21d"]["precision"] == 1.0
    assert "lift_21d_log" in team


def test_21d_absent_when_columns_missing(tmp_path):
    # Older DB without 21d columns → 21d blocks are None, 5d still computes.
    db = tmp_path / "research.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE universe_returns (ticker TEXT, eval_date TEXT, sector TEXT, "
        "return_5d REAL, spy_return_5d REAL, beat_spy_5d INTEGER)"
    )
    conn.execute("CREATE TABLE scanner_evaluations (ticker TEXT, eval_date TEXT, quant_filter_pass INTEGER)")
    conn.execute("CREATE TABLE team_candidates (ticker TEXT, eval_date TEXT, team_id TEXT, team_recommended INTEGER)")
    conn.execute("CREATE TABLE cio_evaluations (ticker TEXT, eval_date TEXT, cio_decision TEXT, final_score REAL, cio_conviction REAL)")
    conn.execute("CREATE TABLE predictor_outcomes (symbol TEXT, prediction_date TEXT, "
                 "predicted_direction TEXT, prediction_confidence REAL)")
    for i in range(20):
        conn.execute("INSERT INTO universe_returns VALUES (?,?,?,?,?,?)",
                     (f"T{i:02d}", DATE, "Tech", 1.0, 0.5, i % 2))
        conn.execute("INSERT INTO scanner_evaluations VALUES (?,?,?)", (f"T{i:02d}", DATE, 1 if i < 10 else 0))
    conn.commit()
    conn.close()
    out = compute_lift_metrics(str(db))
    sl = out["scanner_lift"]
    assert sl["classification"] is not None
    assert sl["classification_21d"] is None
    assert sl["lift_21d_log"] is None
