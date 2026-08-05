"""Tests for analysis/wide_feed_counterfactual_review.py — the config#1398
review CLI's own additions (cohort sizing + the no-skill capture-rate
reference). Does NOT re-test ``attractiveness_eval._counterfactual`` itself
(covered by tests/test_attractiveness_eval.py) — only the two functions this
module adds on top of it. Fully hermetic: no S3, tmp sqlite research.db.
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from analysis.wide_feed_counterfactual_review import (
    build_review,
    cohort_sizes,
    format_report,
    null_capture_rate,
)
from nousergon_lib.quant.horizons import DEFAULT_POLICY

HORIZON = int(DEFAULT_POLICY.primary_horizon)
RET_COL = f"log_return_{HORIZON}d"
SPY_COL = f"log_spy_return_{HORIZON}d"

WEEKLY_DATES = ["2026-05-22", "2026-05-29", "2026-06-05"]


def _tickers(n: int) -> list[str]:
    return [f"T{i:03d}" for i in range(n)]


def _make_research_db(tmp_path, dates, n_names=100, n_pass=10, n_scored=40):
    """research.db with universe_returns (all resolved) + scanner_evaluations
    (first n_pass tickers pass quant_filter_pass each cycle). Returns the db
    path; ``n_scored`` is only used by the caller to build a matching history
    frame (not persisted here)."""
    db = tmp_path / "research.db"
    conn = sqlite3.connect(db)
    conn.execute(
        f"CREATE TABLE universe_returns (ticker TEXT, eval_date TEXT, "
        f"sector TEXT, {RET_COL} REAL, {SPY_COL} REAL)"
    )
    conn.execute(
        "CREATE TABLE scanner_evaluations (ticker TEXT, eval_date TEXT, "
        "quant_filter_pass INTEGER)"
    )
    for d in dates:
        for i, t in enumerate(_tickers(n_names)):
            conn.execute(
                "INSERT INTO universe_returns VALUES (?,?,?,?,?)",
                (t, d, "Tech", 0.30 - i * 0.002, 0.0),
            )
            conn.execute(
                "INSERT INTO scanner_evaluations VALUES (?,?,?)",
                (t, d, 1 if i < n_pass else 0),
            )
    conn.commit()
    conn.close()
    return str(db)


def _make_history(dates, n_names=100, n_scored=40):
    """attractiveness history covering only the first ``n_scored`` tickers
    per date (mirrors real partial coverage of the ~900-name universe)."""
    rows = []
    for d in dates:
        for i, t in enumerate(_tickers(n_scored)):
            rows.append({
                "as_of": d, "ticker": t,
                "attractiveness_raw": (n_scored - i) / n_scored,
                "attractiveness_score": float(n_scored - i),
                "sector": "Tech", "industry": None,
            })
    return pd.DataFrame(rows)


# ── null_capture_rate ────────────────────────────────────────────────────────


def test_null_capture_rate_is_n_over_universe():
    assert null_capture_rate(60, 900) == pytest.approx(round(60 / 900, 5), rel=1e-6)


def test_null_capture_rate_none_when_universe_unknown():
    assert null_capture_rate(60, None) is None
    assert null_capture_rate(60, 0) is None


def test_null_capture_rate_none_when_n_exceeds_universe():
    # A truncated top-N would not be the named variant (mirrors the
    # producer's own "a truncated top-N wouldn't be the named variant" guard).
    assert null_capture_rate(200, 150) is None


# ── cohort_sizes ─────────────────────────────────────────────────────────────


def test_cohort_sizes_counts_universe_survivors_and_scored(tmp_path):
    db = _make_research_db(tmp_path, WEEKLY_DATES, n_names=100, n_pass=10)
    history = _make_history(WEEKLY_DATES, n_names=100, n_scored=40)
    conn = sqlite3.connect(db)
    try:
        sizes = cohort_sizes(conn, history)
    finally:
        conn.close()
    assert set(sizes) == set(WEEKLY_DATES)
    for d in WEEKLY_DATES:
        assert sizes[d]["n_universe"] == 100
        assert sizes[d]["n_survivors"] == 10
        assert sizes[d]["n_scored"] == 40


def test_cohort_sizes_empty_history_gives_zero_scored(tmp_path):
    db = _make_research_db(tmp_path, WEEKLY_DATES)
    conn = sqlite3.connect(db)
    try:
        sizes = cohort_sizes(conn, None)
    finally:
        conn.close()
    assert all(v["n_scored"] == 0 for v in sizes.values())
    assert all(v["n_universe"] == 100 for v in sizes.values())


def test_cohort_sizes_empty_dict_when_tables_absent(tmp_path):
    db = tmp_path / "empty.db"
    conn = sqlite3.connect(db)
    conn.commit()
    try:
        assert cohort_sizes(conn, None) == {}
    finally:
        conn.close()


# ── build_review / format_report (integration, no S3) ──────────────────────


def test_build_review_annotates_top_n_with_null_capture(tmp_path, monkeypatch):
    # n_names/n_scored=300 clears every COUNTERFACTUAL_TOP_NS variant
    # (60/120/200) so the producer actually emits top_n rows to annotate.
    db = _make_research_db(tmp_path, WEEKLY_DATES, n_names=300, n_pass=10)
    history = _make_history(WEEKLY_DATES, n_names=300, n_scored=300)

    monkeypatch.setattr(
        "analysis.wide_feed_counterfactual_review.load_attractiveness_history",
        lambda bucket, s3_client=None: history,
    )

    review = build_review(db, bucket="unused-in-test", as_of="2026-06-10")
    assert review["cohort"]["n_cycles_covered"] == 3
    assert review["cohort"]["n_universe_avg"] == pytest.approx(300.0)

    top_n = review["artifact"]["counterfactual"]["top_n"]
    assert top_n, "n_scored=300 clears every N=60/120/200 variant — rows must be emitted"
    for row in top_n:
        expected_null = row["n"] / 300.0
        assert row["null_capture_rate"] == pytest.approx(expected_null, rel=1e-4)
        if row.get("capture_rate") is not None:
            assert row["capture_rate_vs_null"] == pytest.approx(
                row["capture_rate"] - expected_null, rel=1e-4, abs=1e-6
            )


def test_build_review_null_capture_none_when_history_thinner_than_every_variant(tmp_path, monkeypatch):
    # attractiveness_eval._counterfactual always emits one row per
    # COUNTERFACTUAL_TOP_NS variant (6 rows), but with capture_rate/mean_alpha
    # None when no cycle ever had len(scored) >= n — here n_scored=40 is below
    # every variant (60/120/200), so every row is the empty shape and our
    # null-capture annotation must not fabricate a number against it either.
    db = _make_research_db(tmp_path, WEEKLY_DATES, n_names=100, n_pass=10)
    history = _make_history(WEEKLY_DATES, n_names=100, n_scored=40)
    monkeypatch.setattr(
        "analysis.wide_feed_counterfactual_review.load_attractiveness_history",
        lambda bucket, s3_client=None: history,
    )
    review = build_review(db, bucket="unused-in-test", as_of="2026-06-10")
    top_n = review["artifact"]["counterfactual"]["top_n"]
    assert len(top_n) == 6
    for row in top_n:
        assert row["capture_rate"] is None
        assert row["n_cycles"] == 0
        # cohort n_universe_avg is real (100): null_capture_rate is still
        # computable for N=60 (<= 100), but None for N=120/200 (> universe,
        # same truncation guard null_capture_rate applies elsewhere).
        # capture_rate_vs_null needs a real capture_rate, so it's None either way.
        if row["n"] <= 100:
            assert row["null_capture_rate"] == pytest.approx(row["n"] / 100.0)
        else:
            assert row["null_capture_rate"] is None
        assert row["capture_rate_vs_null"] is None


def test_format_report_renders_without_error(tmp_path, monkeypatch):
    db = _make_research_db(tmp_path, WEEKLY_DATES, n_names=100, n_pass=10)
    history = _make_history(WEEKLY_DATES, n_names=100, n_scored=40)
    monkeypatch.setattr(
        "analysis.wide_feed_counterfactual_review.load_attractiveness_history",
        lambda bucket, s3_client=None: history,
    )
    review = build_review(db, bucket="unused-in-test", as_of="2026-06-10")
    report = format_report(review)
    assert "config#1398" in report
    assert "live_gate" in report
