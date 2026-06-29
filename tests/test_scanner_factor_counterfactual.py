"""Tests for the scanner multi-factor counterfactual (config#967, 2026-06-22).

Would a multi-factor (or single-sleeve) candidate generation beat the
momentum-only scanner? The fixture makes realized alpha factor-driven (the
named winners win) while the live scanner passes the losers — so the
multi-factor composite and the sleeves must beat the actual scanner pass.
"""

from __future__ import annotations

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.end_to_end import (  # noqa: E402
    _ATTRACTIVENESS_PILLARS,
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


def test_multifactor_beats_momentum_scanner(tmp_path):
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


def _db_with_tech_and_gradient(tmp_path):
    """Fixture exercising BOTH reconciliation axes (config#1186): a continuous
    alpha gradient (so cross-sectional rank-IC is well-defined) + a ``tech_score``
    column for the live scanner (anti-correlated with alpha — it passes losers),
    while momentum loads on the winners. 5 cycles with mild per-cycle jitter so
    the date-clustered t-tests are non-degenerate.
    """
    conn = sqlite3.connect(str(tmp_path / "rt.db"))
    conn.execute(
        "CREATE TABLE scanner_evaluations (ticker TEXT, eval_date TEXT, "
        "quant_filter_pass INTEGER, tech_score REAL)"
    )
    conn.execute(
        "CREATE TABLE universe_returns (ticker TEXT, eval_date TEXT, sector TEXT, "
        "log_return_21d REAL, log_spy_return_21d REAL)"
    )
    dates = ["2026-01-02", "2026-01-09", "2026-01-16", "2026-01-23", "2026-01-30"]
    loadings: dict = {}
    for di, d in enumerate(dates):
        jitter = (di - 2) * 0.002  # mild per-cycle shift, keeps cycles distinct
        for i in range(12):
            alpha = (6 - i) * 0.01 + jitter          # monotonic: low i = winner
            qpass = 1 if i >= 6 else 0                 # live scanner passes losers
            conn.execute(
                "INSERT INTO scanner_evaluations VALUES (?,?,?,?)",
                (f"T{i}", d, qpass, float(i)),         # tech_score = i (anti-corr w/ alpha)
            )
            conn.execute(
                "INSERT INTO universe_returns VALUES (?,?,?,?,?)",
                (f"T{i}", d, "Tech", alpha, 0.0),
            )
            loadings[(d, f"T{i}")] = {
                "pe_ratio": float(i), "pb_ratio": float(i),
                "momentum_20d": float(12 - i), "return_60d": float(12 - i),
                "roe": float(12 - i), "fcf_yield": float(12 - i),
                "realized_vol_63d": float(i), "idio_vol_60d": float(i),
            }
    conn.commit()
    return conn, loadings


def test_reconciliation_two_axes_present(tmp_path):
    conn, loadings = _db_with_tech_and_gradient(tmp_path)
    r = _scanner_factor_counterfactual(conn, loadings)
    assert r["status"] == "ok", r
    assert r["n_cycles"] == 5, r

    # objective_axes: every method carries BOTH axes, each well-formed.
    axes = r["objective_axes"]
    for k, v in axes.items():
        for ax in ("longonly_topn_alpha", "xs_rank_ic"):
            assert set(v[ax]) == {"mean", "p", "n"}, (k, ax, v)
    # momentum long-only axis is fully populated (one obs per cycle).
    assert axes["momentum_sleeve_topN"]["longonly_topn_alpha"]["n"] == 5, axes

    # With tech_score + a 12-point alpha gradient, the live scanner's cross-
    # sectional rank-IC is computable and NEGATIVE (tech_score passes losers).
    live_ic = axes["actual_scanner_pass"]["xs_rank_ic"]
    assert live_ic["n"] >= 3 and live_ic["mean"] is not None, live_ic
    assert live_ic["mean"] < 0, live_ic

    # reconciliation block contract + the momentum sleeve beats live long-only.
    rec = r["reconciliation"]
    assert isinstance(rec["verdict"], str) and rec["verdict"], rec
    assert isinstance(rec["sleeve_beats_live_longonly_significant"], bool), rec
    assert isinstance(rec["consistent_with_1142_neutralization"], bool), rec
    assert rec["momentum_sleeve_longonly_lift_vs_live"]["mean"] > 0, rec

    # breadth stratification present with >= 4 cycles.
    bs = r["breadth_stratified"]
    assert bs is not None and "median_breadth" in bs, bs
    assert "momentum_sleeve_topN" in bs and "actual_scanner_pass" in bs, bs


def _db_and_pillar_profiles(tmp_path):
    """Fixture for the EXACT live-attractiveness counterfactual (config#1398).
    The 6 sector-neutral pillar percentiles all favour the winners (low i) while
    the live scanner passes the losers — so ``attractiveness_topN`` must beat the
    actual scanner pass. Returns (conn, pillar_profiles)."""
    conn = sqlite3.connect(str(tmp_path / "ra.db"))
    conn.execute(
        "CREATE TABLE scanner_evaluations (ticker TEXT, eval_date TEXT, quant_filter_pass INTEGER)"
    )
    conn.execute(
        "CREATE TABLE universe_returns (ticker TEXT, eval_date TEXT, sector TEXT, "
        "log_return_21d REAL, log_spy_return_21d REAL)"
    )
    dates = ["2026-01-02", "2026-01-09", "2026-01-16", "2026-01-23"]
    profiles: dict = {}
    for d in dates:
        for i in range(10):
            alpha = 0.03 if i < 5 else -0.03
            qpass = 1 if i >= 5 else 0  # live scanner passes the losers
            conn.execute("INSERT INTO scanner_evaluations VALUES (?,?,?)", (f"T{i}", d, qpass))
            conn.execute(
                "INSERT INTO universe_returns VALUES (?,?,?,?,?)", (f"T{i}", d, "Tech", alpha, 0.0)
            )
            # every pillar percentile higher for winners (low i) -> 100 - i*10
            profiles[(d, f"T{i}")] = {p: float(100 - i * 10) for p in _ATTRACTIVENESS_PILLARS}
    conn.commit()
    return conn, profiles


def test_attractiveness_beats_scanner(tmp_path):
    conn, profiles = _db_and_pillar_profiles(tmp_path)
    r = _scanner_factor_counterfactual(conn, loadings=None, pillar_profiles=profiles)
    assert r["status"] == "ok", r
    assert r["n_cycles"] == 4, r
    assert r["attractiveness_profile_cohorts"] == 4, r
    mset = r["methods"]
    assert mset["actual_scanner_pass"]["mean_alpha_21d"] < 0, r
    # the live attractiveness composite picks the winners -> positive, beats scanner
    assert mset["attractiveness_topN"]["mean_alpha_21d"] > 0, r
    assert mset["attractiveness_topN"]["lift_vs_actual_scanner"] > 0, r
    # count-matched to the live pass count (5/cycle x 4 = 20)
    assert mset["attractiveness_topN"]["n_picks"] == 20, r
    # objective axes carry both axes for the attractiveness method
    assert set(r["objective_axes"]["attractiveness_topN"]["longonly_topn_alpha"]) == {"mean", "p", "n"}, r


def test_attractiveness_runs_without_arctic_loadings(tmp_path):
    """Pillar profiles alone (no ArcticDB loadings) still produce the
    attractiveness method; the raw-factor multifactor method is simply empty."""
    conn, profiles = _db_and_pillar_profiles(tmp_path)
    r = _scanner_factor_counterfactual(conn, loadings=None, pillar_profiles=profiles)
    assert r["status"] == "ok", r
    assert "attractiveness_topN" in r["methods"], r
    assert r["methods"]["attractiveness_topN"]["n_picks"] > 0, r
    # multifactor has no ArcticDB input -> no picks (but must not error)
    assert r["methods"]["multifactor_topN"]["n_picks"] == 0, r


def test_attractiveness_partial_profile_coverage(tmp_path):
    """When profiles exist for only some matured cohorts, the cohort count
    reflects the real overlap (the live data-maturation reality, config#1398)."""
    conn, profiles = _db_and_pillar_profiles(tmp_path)
    # drop two of the four cohorts' profiles
    profiles = {k: v for k, v in profiles.items() if k[0] in ("2026-01-02", "2026-01-09")}
    r = _scanner_factor_counterfactual(conn, loadings=None, pillar_profiles=profiles)
    assert r["status"] == "ok", r
    assert r["n_cycles"] == 4, r  # scanner cycles unchanged
    assert r["attractiveness_profile_cohorts"] == 2, r  # only 2 have profiles


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
