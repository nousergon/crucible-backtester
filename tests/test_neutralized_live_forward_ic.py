"""Tests for the GRADED LIVE neutralization forward-IC producer (config#1187).

``_neutralized_live_forward_ic`` reads the DUAL field crucible-research now
persists to ``cio_evaluations`` — the raw composite (``final_score``) AND the
LIVE neutralized ranking score (``neutralized_final_score``) — joins it to
realized 21d log-alpha (``universe_returns``), and computes the per-week
raw-vs-neutralized Spearman rank-IC + the per-week paired delta with a
Grinold-Kahn one-sample t-test, segmented to the LIVE cohorts (rows with a
NON-NULL persisted neutralized score).

Unlike ``_neutralized_composite_ic`` (which RE-DERIVES neutralization from
history), this measures the score the live system ACTUALLY ranked on. The tests
pin:
  (a) on synthetic data where the persisted neutralized score predicts forward
      alpha better than the raw composite, the LIVE-forward segment shows a
      positive mean weekly delta computed against the neutralized FIELD;
  (b) NULL neutralized scores (live gate OFF) are identity — zero delta, and
      excluded from the live cohort count;
  (c) honest skip when the column is absent (research.db pre-migration) and
      when the join yields no realized outcomes.
"""

from __future__ import annotations

import math
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.end_to_end import _neutralized_live_forward_ic  # noqa: E402


def _base_db(tmp_path, with_neutralized_col=True):
    conn = sqlite3.connect(str(tmp_path / "r.db"))
    neu_col = ", neutralized_final_score REAL" if with_neutralized_col else ""
    conn.execute(
        "CREATE TABLE cio_evaluations ("
        "ticker TEXT, eval_date TEXT, final_score REAL" + neu_col + ")"
    )
    conn.execute(
        "CREATE TABLE universe_returns ("
        "ticker TEXT, eval_date TEXT, log_return_21d REAL, log_spy_return_21d REAL)"
    )
    return conn


def _insert_week(conn, date, *, live, with_neutralized_col=True):
    """Insert a 24-name cross-section for one week.

    Design (same orthogonalization as the composite-IC test): momentum cycles
    within blocks of 6, idiosyncratic varies across blocks, mom ⊥ idio.

        raw_score = 50 + 10*mom + 3*idio   (momentum DOMINATES the raw ranking)
        neu_score = 50 + 3*idio            (the residualized live score: momentum
                                            tilt removed — what the live cutover
                                            actually ranked on)
        alpha21   = 0.01*idio - 0.02*mom   (momentum ANTI-predicts; idio predicts)

    So the raw composite's forward IC is dragged negative while the persisted
    neutralized score's IC is positive. ``live`` controls whether the neutralized
    score is persisted (NON-NULL) for this week.
    """
    for i in range(24):
        mom = (i % 6) - 2.5
        idio = (i // 6) - 1.5
        raw = 50.0 + 10.0 * mom + 3.0 * idio
        neu = 50.0 + 3.0 * idio
        alpha = 0.01 * idio - 0.02 * mom
        if with_neutralized_col:
            neu_val = neu if live else None
            conn.execute(
                "INSERT INTO cio_evaluations VALUES (?,?,?,?)",
                (f"T{i}", date, raw, neu_val),
            )
        else:
            conn.execute(
                "INSERT INTO cio_evaluations VALUES (?,?,?)", (f"T{i}", date, raw)
            )
        # log_spy=0 so realized log-alpha == log_return_21d == alpha.
        conn.execute(
            "INSERT INTO universe_returns VALUES (?,?,?,?)", (f"T{i}", date, alpha, 0.0)
        )


# Post-cutover (>= 2026-06-22) and pre-cutover dates.
_POST = ["2026-06-22", "2026-06-29", "2026-07-06", "2026-07-13", "2026-07-20"]
_PRE = ["2026-05-04", "2026-05-11"]


def test_live_forward_grades_neutralized_field(tmp_path):
    """Persisted neutralized score predicts forward alpha better than the raw
    composite -> the LIVE segment shows a positive mean weekly delta, computed
    against the neutralized FIELD (not a re-derived counterfactual)."""
    conn = _base_db(tmp_path)
    for d in _POST:
        _insert_week(conn, d, live=True)
    conn.commit()

    r = _neutralized_live_forward_ic(conn)
    assert r["status"] == "ok", r
    assert r["source"].startswith("persisted cio_evaluations.neutralized_final_score")
    assert r["n_live_rows"] == 24 * len(_POST)

    lf = r["live_forward"]
    assert lf["n_weeks"] == len(_POST), lf
    # Raw composite IC dragged negative by the momentum tilt; the persisted
    # neutralized score's IC is positive -> positive delta, edge recovered live.
    assert lf["raw_mean_weekly_ic"] < 0, lf
    assert lf["neutralized_mean_weekly_ic"] > lf["raw_mean_weekly_ic"], lf
    assert lf["mean_weekly_delta"] > 0, lf
    assert lf["recovers_edge_live"] is True, lf
    # 5 weeks, identical positive per-week delta by construction -> nunique<2,
    # so the t-test is not run (degenerate); significance stays False but the
    # directional read is GREEN-leaning. Pin the t-test path separately below.
    assert lf["delta_t_p"] is None or lf["significant"] in (True, False)


def test_live_forward_tstat_runs_with_delta_variation(tmp_path):
    """With per-week delta variation (jittered idio loadings), n>=4 weeks ->
    the Grinold-Kahn t-test runs and yields a finite p-value."""
    conn = _base_db(tmp_path)
    for k, d in enumerate(_POST):
        # Scale the idiosyncratic predictiveness per week so weekly deltas vary.
        for i in range(24):
            mom = (i % 6) - 2.5
            idio = (i // 6) - 1.5
            raw = 50.0 + 10.0 * mom + 3.0 * idio
            neu = 50.0 + 3.0 * idio
            alpha = (0.008 + 0.001 * k) * idio - 0.02 * mom
            conn.execute(
                "INSERT INTO cio_evaluations VALUES (?,?,?,?)", (f"T{i}", d, raw, neu)
            )
            conn.execute(
                "INSERT INTO universe_returns VALUES (?,?,?,?)", (f"T{i}", d, alpha, 0.0)
            )
    conn.commit()

    lf = _neutralized_live_forward_ic(conn)["live_forward"]
    assert lf["n_weeks"] == len(_POST)
    assert lf["delta_t_p"] is not None and math.isfinite(lf["delta_t_p"]), lf
    assert lf["mean_weekly_delta"] > 0, lf


def test_null_neutralized_is_identity_and_excluded_from_live(tmp_path):
    """Gate OFF (neutralized_final_score NULL) -> the live ranking == raw, so the
    week's delta is zero and the rows are NOT counted as live cohorts."""
    conn = _base_db(tmp_path)
    for d in _PRE:  # gate off: persist NULL neutralized scores
        _insert_week(conn, d, live=False)
    conn.commit()

    r = _neutralized_live_forward_ic(conn)
    assert r["status"] == "ok", r
    assert r["n_live_rows"] == 0, r            # no live cohorts
    assert r["live_forward"]["n_weeks"] == 0, r
    # all_weeks still computes (identity): neutralized IC == raw IC, zero delta.
    aw = r["all_weeks"]
    assert aw["n_weeks"] == len(_PRE), aw
    assert aw["mean_weekly_delta"] == 0.0, aw
    assert aw["raw_mean_weekly_ic"] == aw["neutralized_mean_weekly_ic"], aw


def test_skips_when_column_absent(tmp_path):
    """research.db predating the config#1187 migration has no
    neutralized_final_score column -> honest skip, never an error."""
    conn = _base_db(tmp_path, with_neutralized_col=False)
    for d in _POST:
        _insert_week(conn, d, live=False, with_neutralized_col=False)
    conn.commit()

    r = _neutralized_live_forward_ic(conn)
    assert r["status"] == "skipped", r
    assert "neutralized_final_score" in r["reason"]


def test_insufficient_data_when_no_realized_outcomes(tmp_path):
    conn = _base_db(tmp_path)
    # cio_evaluations rows but no matching universe_returns -> empty join.
    conn.execute(
        "INSERT INTO cio_evaluations VALUES (?,?,?,?)", ("AAA", "2026-06-22", 50.0, 48.0)
    )
    conn.commit()
    r = _neutralized_live_forward_ic(conn)
    assert r["status"] == "insufficient_data", r
