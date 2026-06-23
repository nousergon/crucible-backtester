"""Tests for the CIO consolidation counterfactual (config#967/#968, 2026-06-22).

Does a deterministic top-N selection beat the LLM CIO's ADVANCE gate, on the same
candidate pool with the entrant count held fixed at the CIO's own ADVANCE count?
The fixture constructs an ANTI-selecting CIO (advances the lowest-alpha names)
while `quant_score` ranks the winners, so `quant_score_topN` must beat the CIO —
exercising the count-matching, the lift_vs_cio sign, and best_method.
"""

from __future__ import annotations

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.end_to_end import _cio_consolidation_counterfactual  # noqa: E402


def _db(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "r.db"))
    conn.execute(
        "CREATE TABLE cio_evaluations (ticker TEXT, eval_date TEXT, cio_decision TEXT, "
        "quant_score REAL, combined_score REAL)"
    )
    conn.execute(
        "CREATE TABLE universe_returns (ticker TEXT, eval_date TEXT, sector TEXT, "
        "log_return_21d REAL, log_spy_return_21d REAL)"
    )
    # 4 cycles x 6 names. alpha = [.05,.04,.03,-.03,-.04,-.05]. The CIO ADVANCES
    # the 3 LOWEST-alpha names (anti-selection); quant_score == alpha so a
    # quant top-3 picks the winners.
    alphas = [0.05, 0.04, 0.03, -0.03, -0.04, -0.05]
    for d in ("2026-01-02", "2026-01-09", "2026-01-16", "2026-01-23"):
        for i, a in enumerate(alphas):
            decision = "ADVANCE" if a < 0 else "REJECT"  # advance the losers
            conn.execute(
                "INSERT INTO cio_evaluations VALUES (?,?,?,?,?)",
                # quant_score = alpha (ranks winners); combined_score = -alpha
                # (inverted, ranks losers) so quant is the UNIQUE best method.
                (f"T{i}", d, decision, a, -a),
            )
            conn.execute(
                "INSERT INTO universe_returns VALUES (?,?,?,?,?)",
                (f"T{i}", d, "Tech", a, 0.0),  # log_spy=0 -> alpha == log_return_21d
            )
    conn.commit()
    return conn


def test_quant_beats_an_anti_selecting_cio(tmp_path):
    conn = _db(tmp_path)
    r = _cio_consolidation_counterfactual(conn, None)
    assert r["status"] == "ok", r
    assert r["n_cycles"] == 4
    m = r["methods"]
    # CIO advanced the losers -> strongly negative.
    assert m["cio_advance"]["mean_alpha_21d"] < 0, r
    # quant top-N picks the winners -> positive and beats the CIO.
    assert m["quant_score_topN"]["mean_alpha_21d"] > 0, r
    assert m["quant_score_topN"]["lift_vs_cio"] > 0, r
    assert r["best_method"] == "quant_score_topN", r
    assert r["any_deterministic_beats_cio"] is True, r
    # count-matched: CIO advanced 3/cycle x 4 = 12; quant top-N also 12.
    assert m["cio_advance"]["n_picks"] == 12 and m["quant_score_topN"]["n_picks"] == 12, r


def test_factor_neutral_method_present_only_with_loadings(tmp_path):
    conn = _db(tmp_path)
    # No loadings -> factor_neutral method has no picks (None selection each cycle).
    r = _cio_consolidation_counterfactual(conn, None, loadings=None)
    assert r["methods"]["factor_neutral_quant_topN"]["n_picks"] == 0, r
    # With loadings present for every (date, ticker), it engages.
    dates = ["2026-01-02", "2026-01-09", "2026-01-16", "2026-01-23"]
    loadings = {
        (d, f"T{i}"): {"momentum_20d": float(i), "return_60d": float(i),
                       "beta_60d": 0.0, "size_log": 1.0}
        for d in dates for i in range(6)
    }
    r2 = _cio_consolidation_counterfactual(conn, None, loadings=loadings)
    assert r2["methods"]["factor_neutral_quant_topN"]["n_picks"] > 0, r2


def test_skipped_without_tables(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "empty.db"))
    conn.execute("CREATE TABLE other (x INTEGER)")
    conn.commit()
    r = _cio_consolidation_counterfactual(conn, None)
    assert r["status"] == "skipped", r
