"""Tests for the historical Barra-neutralized composite counterfactual
(config#1142/#1060).

``_neutralized_composite_ic`` answers the cutover-gate question on HISTORY: does
residualizing the wide research composite (``team_candidates.quant_score``,
joined to ``universe_returns`` for realized 21d alpha) against the
momentum/beta/size factor exposures recover forward 21d-alpha skill? These tests
construct a cross-section where the raw composite is dominated by an
anti-predictive momentum tilt (so its IC vs realized alpha is negative) while the
idiosyncratic residual predicts — the neutralized IC must then beat the raw IC.
The pure ``_xs_neutralize`` and the fail-soft paths are also pinned.
"""

from __future__ import annotations

import os
import sqlite3
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.end_to_end import (  # noqa: E402
    _neutralized_composite_ic,
    _xs_neutralize,
)


def _conn_loadings(tmp_path):
    """Build a research.db (``team_candidates`` + ``universe_returns``) + loadings
    on a BALANCED design where momentum is orthogonal to the idiosyncratic part.

    For i in 0..23:  mom = (i % 6) - 2.5   (cycles within each block of 6)
                     idio = (i // 6) - 1.5  (constant within each block)
    so across the cross-section mom ⊥ idio. Then::

        quant_score = 50 + 10*mom + 3*idio    (momentum DOMINATES the ranking)
        alpha21     = 0.01*idio - 0.02*mom     (momentum ANTI-predicts; idio predicts)

    Raw composite IC is dragged negative by the dominant momentum tilt;
    residualizing momentum out exposes the positive idiosyncratic signal ->
    neutralized IC > raw IC. (The 4-factor production path's rank/residual
    behaviour is covered by the ``_xs_neutralize`` unit tests; here we neutralize
    the single momentum factor so the mechanism is isolated.)
    """
    conn = sqlite3.connect(str(tmp_path / "r.db"))
    conn.execute(
        "CREATE TABLE team_candidates (ticker TEXT, eval_date TEXT, quant_score REAL)"
    )
    conn.execute(
        "CREATE TABLE universe_returns ("
        "ticker TEXT, eval_date TEXT, log_return_21d REAL, log_spy_return_21d REAL)"
    )
    loadings: dict = {}
    dates = [
        "2026-01-02", "2026-01-09", "2026-01-16",
        "2026-02-06", "2026-02-13", "2026-02-20",
    ]
    for d in dates:
        for i in range(24):
            mom = (i % 6) - 2.5
            idio = (i // 6) - 1.5
            score = 50.0 + 10.0 * mom + 3.0 * idio
            alpha = 0.01 * idio - 0.02 * mom
            conn.execute(
                "INSERT INTO team_candidates VALUES (?,?,?)", (f"T{i}", d, score)
            )
            # log_spy=0 so realized log-alpha == log_return_21d == alpha.
            conn.execute(
                "INSERT INTO universe_returns VALUES (?,?,?,?)", (f"T{i}", d, alpha, 0.0)
            )
            loadings[(d, f"T{i}")] = {"momentum_20d": mom}
    conn.commit()
    return conn, loadings


def test_neutralization_beats_raw_on_history(tmp_path):
    conn, loadings = _conn_loadings(tmp_path)
    r = _neutralized_composite_ic(conn, loadings, factors=("momentum_20d",))
    assert r["status"] == "ok", r
    assert r["n_weeks"] == 6
    # Raw composite IC is dragged negative by the dominant momentum tilt...
    assert r["raw_mean_weekly_ic"] < 0, r
    # ...and neutralizing momentum out lifts the IC above the raw value, here
    # all the way positive (the idiosyncratic part predicts).
    assert r["neutralized_mean_weekly_ic"] > r["raw_mean_weekly_ic"], r
    assert r["ic_improvement"] > 0, r
    assert r["neutralization_recovers_edge"] is True, r
    assert r["factor_coverage_frac"] == 1.0, r
    assert r["n_neutralized_weeks"] == 6, r


def test_live_forward_segments_at_cutover(tmp_path):
    # config#1187 — segment the per-week raw/neutralized IC at the 2026-06-22
    # live cutover. Dates straddle it (2 pre, 5 post); neutralizing the dominant
    # anti-predictive momentum tilt beats raw EVERY week, with small per-week
    # jitter so the post-cutover per-week delta has variance for the t-test.
    conn = sqlite3.connect(str(tmp_path / "r.db"))
    conn.execute("CREATE TABLE team_candidates (ticker TEXT, eval_date TEXT, quant_score REAL)")
    conn.execute("CREATE TABLE universe_returns ("
                 "ticker TEXT, eval_date TEXT, log_return_21d REAL, log_spy_return_21d REAL)")
    loadings: dict = {}
    dates = ["2026-05-01", "2026-05-08",                                   # pre-cutover
             "2026-06-26", "2026-07-03", "2026-07-10", "2026-07-17", "2026-07-24"]  # post
    for di, d in enumerate(dates):
        jitter = 0.001 * di  # tiny per-week tilt → non-degenerate weekly deltas
        for i in range(24):
            mom = (i % 6) - 2.5
            idio = (i // 6) - 1.5
            score = 50.0 + 10.0 * mom + 3.0 * idio
            alpha = (0.01 + jitter) * idio - 0.02 * mom
            conn.execute("INSERT INTO team_candidates VALUES (?,?,?)", (f"T{i}", d, score))
            conn.execute("INSERT INTO universe_returns VALUES (?,?,?,?)", (f"T{i}", d, alpha, 0.0))
            loadings[(d, f"T{i}")] = {"momentum_20d": mom}
    conn.commit()
    r = _neutralized_composite_ic(conn, loadings, factors=("momentum_20d",))
    assert r["status"] == "ok", r
    assert r["cutover_date"] == "2026-06-22"
    lf, pre = r["live_forward"], r["pre_cutover"]
    assert lf["n_weeks"] == 5 and pre["n_weeks"] == 2, (lf, pre)
    assert lf["recovers_edge_live"] is True, lf
    assert lf["mean_weekly_delta"] > 0, lf
    assert lf["neutralized_mean_weekly_ic"] > lf["raw_mean_weekly_ic"], lf
    assert lf["delta_t_p"] is not None, lf  # >=3 post weeks → t-test runs
    assert lf["significant"] is True, lf    # 5 wks, consistent +delta → significant


def test_live_forward_underpowered_when_few_post_weeks(tmp_path):
    # The original all-pre-cutover fixture → live_forward is empty (0 weeks),
    # under-powered (p=None, not significant) — the accumulating state.
    conn, loadings = _conn_loadings(tmp_path)
    r = _neutralized_composite_ic(conn, loadings, factors=("momentum_20d",))
    lf = r["live_forward"]
    assert lf["n_weeks"] == 0
    assert lf["delta_t_p"] is None and lf["significant"] is False
    assert r["pre_cutover"]["n_weeks"] == 6


def test_skipped_without_loadings(tmp_path):
    conn, _ = _conn_loadings(tmp_path)
    r = _neutralized_composite_ic(conn, {})
    assert r["status"] == "skipped", r
    assert "loadings" in r["reason"]


def test_skipped_when_no_team_candidates(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "r2.db"))
    conn.execute("CREATE TABLE other (x INTEGER)")
    conn.commit()
    r = _neutralized_composite_ic(conn, {("2026-01-02", "A"): {"momentum_20d": 1.0}})
    assert r["status"] == "skipped", r


def test_xs_neutralize_identity_when_no_factors():
    scores = {"A": 1.0, "B": 2.0, "C": 3.0}
    out = _xs_neutralize(scores, {}, [])
    assert out == scores


def test_xs_neutralize_identity_below_min_names():
    # 3 names < min_names=20 -> identity passthrough (fail-soft), never drops.
    scores = {f"T{i}": float(i) for i in range(3)}
    exposures = {f"T{i}": {"momentum_20d": float(i)} for i in range(3)}
    out = _xs_neutralize(scores, exposures, ["momentum_20d"])
    assert out == scores


def test_xs_neutralize_residualizes_and_preserves_names():
    # 24 names, score perfectly collinear with the single factor -> the residual
    # (rescaled) collapses the factor's ranking but returns every name.
    n = 24
    scores = {f"T{i}": 50.0 + 2.0 * i for i in range(n)}
    exposures = {f"T{i}": {"momentum_20d": float(i)} for i in range(n)}
    out = _xs_neutralize(scores, exposures, ["momentum_20d"])
    assert set(out) == set(scores)  # no name dropped
    # Original score ranks strictly by i; after removing the (collinear) factor
    # the residual carries essentially no monotone ranking in i.
    resid_vals = [out[f"T{i}"] for i in range(n)]
    assert max(resid_vals) - min(resid_vals) < (scores["T23"] - scores["T0"])
