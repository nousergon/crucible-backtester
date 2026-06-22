"""Tests for the historical Barra-neutralized composite counterfactual
(config#1142/#1060).

``_neutralized_composite_ic`` answers the cutover-gate question on HISTORY: does
residualizing the research composite against the momentum/beta/size factor
loadings recover forward 21d-alpha skill? These tests construct a cross-section
where the raw composite is dominated by an anti-predictive momentum tilt (so its
IC vs realized alpha is negative) while the idiosyncratic residual predicts —
the neutralized IC must then beat the raw IC. The pure ``_xs_neutralize`` and
the fail-soft paths are also pinned.
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


def _conn_ur_loadings(tmp_path):
    """Build a research.db + universe_returns + loadings on a BALANCED design
    where momentum is orthogonal to the idiosyncratic part by construction.

    For i in 0..23:  mom = (i % 6) - 2.5   (cycles within each block of 6)
                     idio = (i // 6) - 1.5  (constant within each block)
    so across the cross-section mom ⊥ idio (every idio block sees the full mom
    range). Then::

        score   = 50 + 10*mom + 3*idio        (momentum DOMINATES the ranking)
        alpha21 = 0.01*idio - 0.02*mom         (momentum ANTI-predicts; idio predicts)

    Raw-score IC is dragged negative by the dominant momentum tilt; residualizing
    momentum out exposes the positive idiosyncratic signal -> neutralized IC >
    raw IC. (The 4-factor production path's rank/residual behaviour is covered by
    the ``_xs_neutralize`` unit tests; here we neutralize the single momentum
    factor so the mechanism is isolated.)
    """
    conn = sqlite3.connect(str(tmp_path / "r.db"))
    conn.execute(
        "CREATE TABLE cio_evaluations (ticker TEXT, eval_date TEXT, combined_score REAL)"
    )
    ur_rows = []
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
                "INSERT INTO cio_evaluations VALUES (?,?,?)", (f"T{i}", d, score)
            )
            ur_rows.append({
                "ticker": f"T{i}", "eval_date": d, "return_5d": 0.01,
                "log_return_21d": alpha, "log_spy_return_21d": 0.0,
            })
            loadings[(d, f"T{i}")] = {"momentum_20d_zscore": mom}
    conn.commit()
    return conn, pd.DataFrame(ur_rows), loadings


def test_neutralization_beats_raw_on_history(tmp_path):
    conn, ur, loadings = _conn_ur_loadings(tmp_path)
    r = _neutralized_composite_ic(
        conn, ur, "", [], loadings, factors=("momentum_20d_zscore",)
    )
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


def test_skipped_without_loadings(tmp_path):
    conn, ur, _ = _conn_ur_loadings(tmp_path)
    r = _neutralized_composite_ic(conn, ur, "", [], {})
    assert r["status"] == "skipped", r
    assert "loadings" in r["reason"]


def test_skipped_when_no_combined_score_column(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "r2.db"))
    conn.execute("CREATE TABLE cio_evaluations (ticker TEXT, eval_date TEXT)")
    conn.commit()
    ur = pd.DataFrame([{
        "ticker": "A", "eval_date": "2026-01-02", "return_5d": 0.01,
        "log_return_21d": 0.01, "log_spy_return_21d": 0.0,
    }])
    r = _neutralized_composite_ic(conn, ur, "", [], {("2026-01-02", "A"): {"x": 1.0}})
    assert r["status"] == "skipped", r


def test_xs_neutralize_identity_when_no_factors():
    scores = {"A": 1.0, "B": 2.0, "C": 3.0}
    out = _xs_neutralize(scores, {}, [])
    assert out == scores


def test_xs_neutralize_identity_below_min_names():
    # 3 names < min_names=20 -> identity passthrough (fail-soft), never drops.
    scores = {f"T{i}": float(i) for i in range(3)}
    exposures = {f"T{i}": {"momentum_20d_zscore": float(i)} for i in range(3)}
    out = _xs_neutralize(scores, exposures, ["momentum_20d_zscore"])
    assert out == scores


def test_xs_neutralize_residualizes_and_preserves_names():
    # 24 names, score perfectly collinear with the single factor -> the residual
    # (rescaled) collapses the factor's ranking but returns every name.
    n = 24
    scores = {f"T{i}": 50.0 + 2.0 * i for i in range(n)}
    exposures = {f"T{i}": {"momentum_20d_zscore": float(i)} for i in range(n)}
    out = _xs_neutralize(scores, exposures, ["momentum_20d_zscore"])
    assert set(out) == set(scores)  # no name dropped
    # Original score ranks strictly by i; after removing the (collinear) factor
    # the residual carries essentially no monotone ranking in i.
    resid_vals = [out[f"T{i}"] for i in range(n)]
    assert max(resid_vals) - min(resid_vals) < (scores["T23"] - scores["T0"])
