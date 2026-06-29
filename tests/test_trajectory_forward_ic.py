"""Tests for the OBSERVE-mode attractiveness-trajectory forward-IC producer
(crucible-research #337 / config#1392).

``_trajectory_forward_ic`` joins the PERSISTED per-name trajectory scores
(``pre_repricing_score`` and ``attr_slope_z``, as written weekly to
``s3://{bucket}/scanner/universe/trajectory/{date}/trajectory.json`` by
crucible-research and read back via ``load_historical_trajectory_scores``) to
realized 21d log market-relative alpha (``universe_returns.log_return_21d -
log_spy_return_21d``) on ``(eval_date, ticker)`` — the same realized-return
source/join ``_neutralized_live_forward_ic`` uses — and computes the per-week
cross-sectional Spearman rank-IC of each score via the shared quant engine
(``analysis.information_coefficient.compute_ic``).

It is pure OBSERVE-mode: it measures + surfaces the rolling IC and ``n_cohorts``
so the observe->cutover gate is decidable, but never auto-promotes. The tests
pin:
  (a) a positive-signal fixture (pre_repricing_score predicts forward alpha)
      over >= the maturity floor -> status 'ok' with a POSITIVE mean weekly IC
      and the expected shape/keys;
  (b) too few mature cohorts -> status 'accruing' with provisional_ic None (NOT
      a crash) — the value the console header replaces 'accruing' with;
  (c) no injected artifacts -> honest 'accruing' (loader returned {});
  (d) no realized outcomes in the join -> 'accruing';
  (e) the negative/zero-signal case reads non-positive.
"""

from __future__ import annotations

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.end_to_end import (  # noqa: E402
    TRAJECTORY_FORWARD_IC_MIN_COHORTS,
    _trajectory_forward_ic,
)


def _base_db(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "r.db"))
    conn.execute(
        "CREATE TABLE universe_returns ("
        "ticker TEXT, eval_date TEXT, log_return_21d REAL, log_spy_return_21d REAL)"
    )
    return conn


def _fixture(conn, dates, *, sign=1.0, n=24):
    """Insert ``n``-name cross-sections for each date and return the matching
    injected trajectory-scores dict.

    Construction: pre_repricing_score = ``i`` (monotone), attr_slope_z = ``i``,
    and realized alpha = ``sign * 0.001 * i`` with log_spy=0 so log-alpha ==
    log_return_21d. With ``sign=+1`` higher trajectory score -> higher forward
    alpha => Spearman IC ~ +1 per week; ``sign=-1`` flips it negative.
    """
    scores: dict = {}
    for d in dates:
        for i in range(n):
            tkr = f"T{i}"
            alpha = sign * 0.001 * i
            conn.execute(
                "INSERT INTO universe_returns VALUES (?,?,?,?)", (tkr, d, alpha, 0.0)
            )
            scores[(d, tkr)] = {
                "pre_repricing_score": float(i),
                "attr_slope_z": float(i),
            }
    conn.commit()
    return scores


_MATURE = ["2026-05-21", "2026-05-28", "2026-06-04", "2026-06-11", "2026-06-18"]
_IMMATURE = ["2026-05-21", "2026-05-28"]  # below the 4-cohort floor


def test_positive_signal_yields_positive_ic_and_shape():
    """Positive-signal fixture over >= the maturity floor -> status 'ok',
    positive mean weekly IC for both signals, correct keys/shape."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE universe_returns ("
        "ticker TEXT, eval_date TEXT, log_return_21d REAL, log_spy_return_21d REAL)"
    )
    scores = _fixture(conn, _MATURE, sign=1.0)

    r = _trajectory_forward_ic(conn, scores)
    assert r["status"] == "ok", r
    assert r["horizon"] == "21d"
    assert r["source"].startswith("persisted scanner/universe/trajectory")
    # headline surface that replaces the console's hard-coded 'accruing'
    assert r["n_cohorts"] == len(_MATURE), r
    assert r["provisional_ic"] is not None and r["provisional_ic"] > 0, r
    # both signals reported, both positive + mature
    for s in ("pre_repricing_score", "attr_slope_z"):
        sig = r[s]
        assert sig["status"] == "ok", sig
        assert sig["n_cohorts"] == len(_MATURE), sig
        assert sig["mean_weekly_ic"] is not None and sig["mean_weekly_ic"] > 0, sig
        assert sig["positive"] is True, sig
    assert r["n_mature_weeks"] == len(_MATURE), r
    assert r["n_join_rows"] == 24 * len(_MATURE), r


def test_insufficient_cohorts_is_accruing_not_crash(tmp_path):
    """Fewer than the maturity floor of mature weekly cohorts -> status
    'accruing', provisional_ic None (honest, NOT a crash)."""
    conn = _base_db(tmp_path)
    scores = _fixture(conn, _IMMATURE, sign=1.0)
    assert len(_IMMATURE) < TRAJECTORY_FORWARD_IC_MIN_COHORTS

    r = _trajectory_forward_ic(conn, scores)
    assert r["status"] == "accruing", r
    assert r["provisional_ic"] is None, r
    assert r["n_cohorts"] == len(_IMMATURE), r
    # per-signal blocks still present + honest
    assert r["pre_repricing_score"]["status"] == "accruing", r
    assert r["pre_repricing_score"]["mean_weekly_ic"] is None, r
    assert r["min_cohorts"] == TRAJECTORY_FORWARD_IC_MIN_COHORTS, r


def test_no_injected_artifacts_is_accruing(tmp_path):
    """No trajectory artifacts (loader returned {}) -> honest accruing, never
    an error — the warm-up state before artifacts accrue."""
    conn = _base_db(tmp_path)
    r = _trajectory_forward_ic(conn, None)
    assert r["status"] == "accruing", r
    assert r["n_cohorts"] == 0, r
    assert r["min_cohorts"] == TRAJECTORY_FORWARD_IC_MIN_COHORTS, r


def test_no_universe_returns_rows_is_insufficient_data(tmp_path):
    """universe_returns has no rows with realized 21d outcomes at all ->
    insufficient_data (honest, not a crash)."""
    conn = _base_db(tmp_path)
    scores = {("2026-05-21", "AAA"): {"pre_repricing_score": 1.0, "attr_slope_z": 1.0}}
    r = _trajectory_forward_ic(conn, scores)
    assert r["status"] == "insufficient_data", r


def test_no_overlapping_names_is_accruing(tmp_path):
    """Realized outcomes exist but NO trajectory name joins to one ->
    accruing (immature join), not a crash."""
    conn = _base_db(tmp_path)
    # realized outcomes for a disjoint ticker/date set
    conn.execute(
        "INSERT INTO universe_returns VALUES (?,?,?,?)", ("ZZZ", "2026-05-21", 0.01, 0.0)
    )
    conn.commit()
    scores = {("2026-05-21", "AAA"): {"pre_repricing_score": 1.0, "attr_slope_z": 1.0}}
    r = _trajectory_forward_ic(conn, scores)
    assert r["status"] == "accruing", r
    assert r["n_cohorts"] == 0, r


def test_negative_signal_reads_non_positive(tmp_path):
    """When the trajectory score ANTI-predicts forward alpha, the mean weekly IC
    is negative and 'positive' is False — the gate stays observe-only."""
    conn = _base_db(tmp_path)
    scores = _fixture(conn, _MATURE, sign=-1.0)
    r = _trajectory_forward_ic(conn, scores)
    assert r["status"] == "ok", r
    assert r["provisional_ic"] is not None and r["provisional_ic"] < 0, r
    assert r["pre_repricing_score"]["positive"] is False, r


def test_skipped_when_universe_returns_absent(tmp_path):
    """No universe_returns table -> honest skip (not error)."""
    conn = sqlite3.connect(str(tmp_path / "empty.db"))
    scores = {("2026-05-21", "AAA"): {"pre_repricing_score": 1.0, "attr_slope_z": 1.0}}
    r = _trajectory_forward_ic(conn, scores)
    assert r["status"] == "skipped", r
    assert "universe_returns" in r["reason"]
