"""Tests for the scanner multi-factor counterfactual (config#967, 2026-06-22).

Would a multi-factor (or single-sleeve) candidate generation beat the
momentum-only scanner? The fixture makes realized alpha VALUE-driven (cheap
names win) while the live scanner passes the high-momentum/expensive names — so
the value sleeve (and the composite) must beat the actual scanner pass.
"""

from __future__ import annotations

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.end_to_end import (  # noqa: E402
    SCANNER_RAW_FACTORS,
    _scanner_factor_counterfactual,
)


def _db_and_loadings(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "r.db"))
    conn.execute(
        "CREATE TABLE scanner_evaluations (ticker TEXT, eval_date TEXT, quant_filter_pass INTEGER)"
    )
    conn.execute(
        "CREATE TABLE universe_returns (ticker TEXT, eval_date TEXT, sector TEXT, "
        "log_return_21d REAL, log_spy_return_21d REAL)"
    )
    dates = ["2026-01-02", "2026-01-09", "2026-01-16", "2026-01-23"]
    loadings: dict = {}
    for d in dates:
        for i in range(10):
            # winners = low i (alpha +0.03); losers = high i (-0.03).
            alpha = 0.03 if i < 5 else -0.03
            # the live scanner PASSES the losers (5..9) — independent of factors.
            qpass = 1 if i >= 5 else 0
            conn.execute(
                "INSERT INTO scanner_evaluations VALUES (?,?,?)", (f"T{i}", d, qpass)
            )
            conn.execute(
                "INSERT INTO universe_returns VALUES (?,?,?,?,?)",
                (f"T{i}", d, "Tech", alpha, 0.0),
            )
            # ALL sleeves point at the winners (low i): cheap value, strong
            # momentum/quality, low vol — so the multi-factor composite and every
            # sleeve favour the names that actually realize +alpha.
            loadings[(d, f"T{i}")] = {
                "pe_ratio": float(i), "pb_ratio": float(i),          # value: cheap = low i
                "momentum_20d": float(10 - i), "return_60d": float(10 - i),  # momentum high = low i
                "roe": float(10 - i), "fcf_yield": float(10 - i),    # quality high = low i
                "realized_vol_63d": float(i), "idio_vol_60d": float(i),      # low vol = low i
            }
    conn.commit()
    return conn, loadings


def test_value_sleeve_beats_momentum_scanner(tmp_path):
    conn, loadings = _db_and_loadings(tmp_path)
    r = _scanner_factor_counterfactual(conn, loadings)
    assert r["status"] == "ok", r
    assert r["n_cycles"] == 4
    mset = r["methods"]
    # the live scanner passed the losers -> negative
    assert mset["actual_scanner_pass"]["mean_alpha_21d"] < 0, r
    # the multi-factor composite picks the winners -> positive, beats the scanner
    assert mset["multifactor_topN"]["mean_alpha_21d"] > 0, r
    assert mset["multifactor_topN"]["lift_vs_actual_scanner"] > 0, r
    # the value sleeve alone also beats the scanner (all sleeves favour winners)
    assert mset["value_sleeve_topN"]["lift_vs_actual_scanner"] > 0, r
    assert r["any_factor_beats_actual_scanner"] is True, r
    assert r["best_sleeve"] is not None, r
    # count-matched: 5 passed/cycle x 4 = 20
    assert mset["actual_scanner_pass"]["n_picks"] == 20, r


def test_skipped_without_loadings(tmp_path):
    conn, _ = _db_and_loadings(tmp_path)
    r = _scanner_factor_counterfactual(conn, None)
    assert r["status"] == "skipped", r


def test_skipped_without_scanner_table(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "e.db"))
    conn.execute("CREATE TABLE other (x INTEGER)")
    conn.commit()
    r = _scanner_factor_counterfactual(conn, {("d", "T"): {"pe_ratio": 1.0}})
    assert r["status"] == "skipped", r
