"""Tests for the breadth-conditioned momentum IC producer (config#1140).

Validates that ``_momentum_regime_ic`` recovers the regime-dependence behind the
negative research edge: short-horizon momentum (tech_score) IC flips sign with
universe breadth. Synthetic data constructs low-breadth weeks where tech_score
anti-predicts realized 21d log-alpha and high-breadth weeks where it predicts.
"""

from __future__ import annotations

import os
import sqlite3
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.end_to_end import _momentum_regime_ic  # noqa: E402


def _conn_ur(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "r.db"))
    conn.execute(
        "CREATE TABLE scanner_evaluations (ticker TEXT, eval_date TEXT, tech_score REAL)"
    )
    ur = []
    # 3 low-breadth weeks (momentum inverted) + 3 high-breadth weeks (momentum
    # predictive). log_spy=0 so realized log-alpha == log_return_21d.
    weeks = [
        ("2026-01-02", -1), ("2026-01-09", -1), ("2026-01-16", -1),
        ("2026-02-06", 1), ("2026-02-13", 1), ("2026-02-20", 1),
    ]
    for date, sign in weeks:
        for i in range(20):
            tech = float(i)
            # high-breadth/positive-IC: alpha = 0.002*tech - 0.01 (breadth 0.70, IC +1)
            # low-breadth/negative-IC:  alpha = -0.002*tech + 0.01 (breadth 0.25, IC -1)
            alpha = (0.002 * tech - 0.01) if sign > 0 else (-0.002 * tech + 0.01)
            conn.execute(
                "INSERT INTO scanner_evaluations VALUES (?,?,?)", (f"T{i}", date, tech)
            )
            ur.append({
                "ticker": f"T{i}", "eval_date": date, "return_5d": 0.01,
                "log_return_21d": alpha, "log_spy_return_21d": 0.0,
            })
    conn.commit()
    return conn, pd.DataFrame(ur)


def test_momentum_regime_ic_flips_with_breadth(tmp_path):
    conn, ur = _conn_ur(tmp_path)
    r = _momentum_regime_ic(conn, ur, "", [])
    assert r["status"] == "ok", r
    assert r["n_weeks"] == 6
    # The whole point: momentum IC is negative in low-breadth, positive in high.
    assert r["low_breadth_ic"] < 0 < r["high_breadth_ic"], r
    # Breadth and weekly IC move together.
    assert r["breadth_ic_corr"] > 0.5, r


def test_momentum_regime_ic_failsoft_without_tech_score(tmp_path):
    # The legacy scanner_evaluations schema (no tech_score) must skip, not raise.
    conn = sqlite3.connect(str(tmp_path / "r2.db"))
    conn.execute(
        "CREATE TABLE scanner_evaluations (ticker TEXT, eval_date TEXT, quant_filter_pass INTEGER)"
    )
    conn.commit()
    ur = pd.DataFrame([{
        "ticker": "A", "eval_date": "2026-01-02", "return_5d": 0.01,
        "log_return_21d": 0.0, "log_spy_return_21d": 0.0,
    }])
    r = _momentum_regime_ic(conn, ur, "", [])
    assert r["status"] == "skipped", r


def test_momentum_regime_ic_insufficient_weeks(tmp_path):
    # Fewer than 4 realized weekly cohorts → insufficient_data, never ok.
    conn = sqlite3.connect(str(tmp_path / "r3.db"))
    conn.execute(
        "CREATE TABLE scanner_evaluations (ticker TEXT, eval_date TEXT, tech_score REAL)"
    )
    ur = []
    for i in range(20):
        conn.execute("INSERT INTO scanner_evaluations VALUES (?,?,?)", (f"T{i}", "2026-01-02", float(i)))
        ur.append({
            "ticker": f"T{i}", "eval_date": "2026-01-02", "return_5d": 0.01,
            "log_return_21d": 0.001 * i, "log_spy_return_21d": 0.0,
        })
    conn.commit()
    r = _momentum_regime_ic(conn, pd.DataFrame(ur), "", [])
    assert r["status"] == "insufficient_data", r
