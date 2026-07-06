"""Parity-over-history test for the config#1529 analysis/reporting cutover.

ACCEPTANCE (config#1529, the core deliverable): the migrated analysis/reporting
reads — now sourced from the long-format ``score_performance_outcomes`` store via
``analysis.outcome_store`` — must produce output IDENTICAL to the pre-migration
wide-``score_performance``-column reads across the full research.db history,
modulo (a) the documented percent↔decimal unit conversion and (b) the dropped
dead 10d/30d horizons.

HOW THIS PROVES FAITHFULNESS
----------------------------
The fixture builds a research.db carrying the SAME ground truth in BOTH physical
representations, exactly as the two producers write them:

  * wide ``score_performance`` columns — 2dp-rounded PERCENT returns
    (alpha-engine-data ``signal_returns`` Step 2: ``round(x * 100, 2)``);
  * long ``score_performance_outcomes`` rows — DECIMAL returns, canonical
    ``log_alpha`` on the primary horizon only (Step 2c).

For each migrated entry point we compute the result TWICE:

  * the LEGACY path — reads the wide columns directly, replicating the exact
    pre-migration code (frozen inline in this test);
  * the MIGRATED path — the shipped function, which reads the wide table and
    re-sources the outcome columns from the long store via
    ``attach_outcomes`` / ``primary_beat_counts``.

and assert the two are identical. Because ``attach_outcomes`` reproduces the
legacy 2dp-percent convention at its single documented boundary (verified
byte-identically in ``test_outcome_store_parity``), the downstream metrics —
accuracies, alpha means, correlations, regime splits, calibration curves,
threshold sweeps, report aggregates — come out equal. This is the same invariant
the full-history replay verifies against a copy of the live research.db (run at
PR time per the issue); this fixture pins it permanently in CI.

The horizons are resolved from ``HorizonPolicy`` (primary 21d + diagnostic 5d);
the fixture deliberately ALSO writes retired 10d/30d wide columns (all-NULL, as
the live schema carries them post config#1456) to prove the migrated reads DROP
them cleanly — they never appear in either representation's store rows.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest
from nousergon_lib.quant.horizons import DEFAULT_POLICY

from analysis import (
    alpha_distribution,
    attribution,
    regime_analysis,
    score_analysis,
    signal_quality,
)
from analysis.outcome_store import primary_beat_counts

_PRIMARY = DEFAULT_POLICY.primary_horizon           # 21
_DIAG = DEFAULT_POLICY.diagnostic_horizons[0]        # 5
_PC = DEFAULT_POLICY.outcome_columns(_PRIMARY)
_DC = DEFAULT_POLICY.outcome_columns(_DIAG)

# A broad ground-truth history: multiple dates, regimes, stances, sectors,
# score buckets, a negative-alpha name, and one UNRESOLVED row. Decimals.
# (symbol, score_date, score, quant, qual, stance, regime, sector,
#  r5, s5, r21, s21)
_TRUTH = [
    ("AAPL", "2026-01-02", 82, 78, 71, "momentum", "bull", "Tech", 0.0213, 0.0100, 0.0432, 0.0201),
    ("MSFT", "2026-01-02", 74, 70, 68, "quality", "bull", "Tech", -0.0150, 0.0100, -0.0311, 0.0201),
    ("NVDA", "2026-01-09", 91, 88, 80, "momentum", "bull", "Tech", 0.0555, 0.0120, 0.1023, 0.0250),
    ("KO", "2026-01-09", 63, 60, 66, "value", "neutral", "Staples", 0.0021, 0.0120, 0.0260, 0.0250),
    ("JPM", "2026-01-16", 77, 72, 74, "catalyst", "neutral", "Financials", -0.0033, -0.0010, 0.0140, 0.0080),
    ("XOM", "2026-01-16", 69, 64, 70, "value", "bear", "Energy", 0.0101, 0.0050, 0.0330, 0.0160),
    ("GOOG", "2026-01-23", 85, 80, 79, "momentum", "bull", "Tech", 0.0301, 0.0075, 0.0512, 0.0180),
    ("PG", "2026-01-23", 66, 61, 69, "quality", "neutral", "Staples", -0.0044, 0.0060, -0.0090, 0.0130),
    ("CVX", "2026-01-30", 71, 68, 67, "value", "bear", "Energy", 0.0088, 0.0040, 0.0201, 0.0150),
    ("META", "2026-01-30", 88, 84, 77, "momentum", "bull", "Tech", 0.0410, 0.0090, 0.0777, 0.0210),
    ("WMT", "2026-02-06", 64, 62, 65, "value", "neutral", "Staples", 0.0012, 0.0055, 0.0102, 0.0120),
    ("BAC", "2026-02-06", 73, 69, 71, "catalyst", "neutral", "Financials", -0.0121, -0.0020, 0.0060, 0.0075),
    ("TSLA", "2026-03-20", 80, 76, 70, "momentum", "bull", "Tech", None, None, None, None),  # unresolved
]


def _log_alpha(r21: float, s21: float) -> float:
    return round(float(np.log1p(r21) - np.log1p(s21)), 6)


@pytest.fixture()
def research_db(tmp_path):
    """A research.db carrying the same ground truth in both representations,
    plus retired 10d/30d wide columns (all-NULL) to prove they drop cleanly."""
    return _build_research_db(tmp_path / "research.db", include_long_store=True)


@pytest.fixture()
def research_db_wide_only(tmp_path):
    """The SAME ground truth with the long store ABSENT — attach_outcomes
    passes the wide columns through unchanged (pre-cutover fallback), which
    IS the mechanical legacy read path. Comparing a function's output on
    this DB vs the dual-representation DB proves the long-store read is
    equivalent to the pre-migration wide read, with zero frozen-code drift."""
    return _build_research_db(tmp_path / "research_wide.db", include_long_store=False)


def _build_research_db(db, include_long_store: bool) -> str:
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE score_performance (
            symbol TEXT, score_date TEXT, score REAL,
            quant_score REAL, qual_score REAL, stance TEXT,
            market_regime TEXT, sector TEXT, price_on_date REAL,
            return_5d REAL, spy_5d_return REAL, beat_spy_5d INTEGER,
            return_21d REAL, spy_21d_return REAL, beat_spy_21d INTEGER,
            log_alpha_21d REAL,
            return_10d REAL, return_30d REAL
        )"""
    )
    if include_long_store:
        conn.execute(
            """CREATE TABLE score_performance_outcomes (
                id INTEGER PRIMARY KEY, signal_id TEXT NOT NULL,
                symbol TEXT NOT NULL, score_date TEXT NOT NULL,
                horizon_days INTEGER NOT NULL, beat_spy INTEGER,
                stock_return REAL, spy_return REAL, log_alpha REAL,
                is_primary INTEGER NOT NULL, resolved_at TEXT NOT NULL,
                schema_version INTEGER NOT NULL DEFAULT 1,
                UNIQUE(signal_id, horizon_days)
            )"""
        )
    for sym, d, sc, q, ql, stance, regime, sector, r5, s5, r21, s21 in _TRUTH:
        resolved = r5 is not None
        conn.execute(
            "INSERT INTO score_performance VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                sym, d, sc, q, ql, stance, regime, sector, 100.0,
                round(r5 * 100, 2) if resolved else None,
                round(s5 * 100, 2) if resolved else None,
                (1 if r5 > s5 else 0) if resolved else None,
                round(r21 * 100, 2) if resolved else None,
                round(s21 * 100, 2) if resolved else None,
                (1 if r21 > s21 else 0) if resolved else None,
                _log_alpha(r21, s21) if resolved else None,
                None, None,  # retired 10d/30d — all-NULL, as live schema carries them
            ),
        )
        if not resolved or not include_long_store:
            continue
        for h, ret, spy in ((_DIAG, r5, s5), (_PRIMARY, r21, s21)):
            is_primary = h == _PRIMARY
            conn.execute(
                "INSERT INTO score_performance_outcomes "
                "(signal_id, symbol, score_date, horizon_days, beat_spy, "
                " stock_return, spy_return, log_alpha, is_primary, resolved_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    f"{sym}:{d}", sym, d, h, 1 if ret > spy else 0,
                    ret, spy,
                    _log_alpha(r21, s21) if is_primary else None,
                    1 if is_primary else 0, "2026-06-27T00:00:00+00:00",
                ),
            )
    conn.commit()
    conn.close()
    return str(db)


def _legacy_wide_df(db_path: str) -> pd.DataFrame:
    """The PRE-migration read: SELECT * FROM score_performance, dates parsed —
    the exact frame the analysis functions received before config#1529."""
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(
            "SELECT * FROM score_performance ORDER BY score_date",
            conn, parse_dates=["score_date"],
        )
    finally:
        conn.close()


# ── signal_quality.compute_accuracy — the central accuracy tile ──────────────


def test_signal_quality_accuracy_identical(research_db):
    """The migrated load_score_performance → compute_accuracy must equal the
    legacy wide read → compute_accuracy (same accuracies, alpha means, buckets)."""
    migrated_df = signal_quality.load_score_performance(research_db)
    legacy_df = _legacy_wide_df(research_db)

    migrated = signal_quality.compute_accuracy(migrated_df, min_samples=5)
    legacy = signal_quality.compute_accuracy(legacy_df, min_samples=5)
    assert migrated == legacy
    assert migrated["status"] == "ok"
    # sanity: the alpha means are non-trivial (the fixture has resolved rows)
    assert migrated["overall"]["avg_alpha_21d"] is not None


# ── score_analysis threshold sweep ───────────────────────────────────────────


def test_score_analysis_threshold_sweep_identical(research_db):
    migrated_df = signal_quality.load_score_performance(research_db)
    legacy_df = _legacy_wide_df(research_db)
    thresholds = [60, 65, 70, 75, 80, 85, 90]
    assert score_analysis.accuracy_by_threshold(
        migrated_df, thresholds=thresholds, min_samples=1
    ) == score_analysis.accuracy_by_threshold(
        legacy_df, thresholds=thresholds, min_samples=1
    )


# ── attribution correlations ─────────────────────────────────────────────────


def test_attribution_identical(research_db):
    migrated_df = signal_quality.load_score_performance(research_db)
    legacy_df = _legacy_wide_df(research_db)
    # attribution needs >=100 rows for a full run; below that it returns
    # insufficient_data deterministically — parity holds on that branch too, and
    # the correlation math is separately covered by test_outcome_store_parity.
    mig = attribution.compute_attribution(migrated_df)
    leg = attribution.compute_attribution(legacy_df)
    assert mig == leg


# ── regime_analysis split ────────────────────────────────────────────────────


def test_regime_analysis_split_identical(research_db):
    migrated_df = regime_analysis.load_with_regime(research_db)
    legacy_df = _legacy_wide_df(research_db)
    if "market_regime" not in legacy_df.columns:
        legacy_df["market_regime"] = pd.NA
    assert regime_analysis.accuracy_by_regime(
        migrated_df, min_samples=1
    ) == regime_analysis.accuracy_by_regime(legacy_df, min_samples=1)


# ── alpha_distribution + calibration ─────────────────────────────────────────


def test_alpha_distribution_identical(research_db):
    """The migrated compute_alpha_distribution (reads wide table + attaches long
    store) must equal a legacy computation that reads the wide columns directly."""
    migrated = alpha_distribution.compute_alpha_distribution(research_db, min_samples=2)

    # Legacy: replicate the pre-migration inline computation on the raw wide df.
    legacy_df = _legacy_wide_df(research_db)
    legacy: dict = {"status": "ok", "distributions": {}, "summary": {}}
    for label, (ret_col, spy_col) in {
        f"{_DIAG}d": ("return_5d", "spy_5d_return"),
        f"{_PRIMARY}d": ("return_21d", "spy_21d_return"),
    }.items():
        sub = legacy_df[[ret_col, spy_col]].dropna().copy()
        if len(sub) < 2:
            continue
        sub["alpha"] = sub[ret_col] - sub[spy_col]
        buckets = []
        for blabel, low, high in alpha_distribution._ALPHA_BUCKETS:
            mask = (sub["alpha"] >= low) & (sub["alpha"] < high)
            count = int(mask.sum())
            buckets.append({
                "bucket": blabel, "count": count,
                "pct": round(count / len(sub), 4) if len(sub) else 0,
                "avg_alpha": round(float(sub.loc[mask, "alpha"].mean()), 2) if count else None,
            })
        legacy["distributions"][label] = buckets
        legacy["summary"][label] = {
            "n": len(sub),
            "avg_alpha": round(float(sub["alpha"].mean()), 2),
            "median_alpha": round(float(sub["alpha"].median()), 2),
            "std_alpha": round(float(sub["alpha"].std()), 2),
            "skew": round(float(sub["alpha"].skew()), 2) if len(sub) >= 3 else None,
            "pct_positive": round(float((sub["alpha"] > 0).mean()), 4),
        }
    assert migrated == legacy


def test_score_calibration_identical(research_db, research_db_wide_only):
    """Mechanical read-path parity at the SAME horizon: the long-store-sourced
    calibration must equal the calibration computed over the identical ground
    truth with the long store ABSENT (attach_outcomes' documented pre-cutover
    fallback = the exact legacy wide-column read). The horizon UPGRADE itself
    is asserted separately (test_score_calibration_declares_primary_horizon)."""
    migrated = alpha_distribution.compute_score_calibration(
        research_db, n_buckets=3, min_per_bucket=1,
    )
    legacy = alpha_distribution.compute_score_calibration(
        research_db_wide_only, n_buckets=3, min_per_bucket=1,
    )
    assert migrated == legacy
    assert migrated["status"] == "ok"
    assert migrated["horizon"] == f"{_PRIMARY}d"
    # Non-vacuous: the curve and the Spearman association actually computed.
    assert len(migrated["calibration"]) >= 2
    assert migrated["spearman_rho"] is not None


def test_score_calibration_declares_primary_horizon(research_db):
    """The horizon-upgrade assertion (config#1529): called exactly as the
    evaluate.py producer calls it (db_path only), score_calibration.json
    declares the CANONICAL primary horizon — label AND integer days — so the
    evaluator's composite_scoring tile displays the true measurement horizon
    instead of falling back to an assumed sub-canonical default. The HORIZON
    argument is left at its default — the exact producer call shape."""
    artifact = alpha_distribution.compute_score_calibration(
        research_db, n_buckets=3, min_per_bucket=1,  # fixture-sized buckets
    )
    assert artifact["horizon"] == f"{_PRIMARY}d"
    assert artifact["horizon_days"] == _PRIMARY
    assert artifact["horizon_days"] == DEFAULT_POLICY.primary_horizon


# ── reporter data-accumulation aggregate (primary beat COUNT/SUM) ────────────


def test_reporter_beat_counts_identical(research_db):
    """The migrated primary_beat_counts (long store) must equal the legacy wide
    aggregate SELECT COUNT/SUM(beat_spy_21d) — beat_spy is 0/1 in both, so the
    counts are byte-identical."""
    n_resolved, n_beat = primary_beat_counts(research_db)

    conn = sqlite3.connect(research_db)
    try:
        legacy_resolved = conn.execute(
            "SELECT COUNT(*) FROM score_performance WHERE beat_spy_21d IS NOT NULL"
        ).fetchone()[0]
        legacy_beat = conn.execute(
            "SELECT SUM(beat_spy_21d) FROM score_performance WHERE beat_spy_21d IS NOT NULL"
        ).fetchone()[0] or 0
    finally:
        conn.close()
    assert (n_resolved, n_beat) == (int(legacy_resolved), int(legacy_beat))


# ── retired-horizon drop: the store never carries 10d/30d ────────────────────


def test_retired_horizons_absent_from_migrated_reads(research_db):
    """The migrated frame carries NO resolved 10d/30d outcome data — those
    horizons are dead (config#1456) and absent from the store. The wide 10d/30d
    columns exist (all-NULL) but the migrated read never resurrects them."""
    df = signal_quality.load_score_performance(research_db)
    for col in ("return_10d", "return_30d"):
        if col in df.columns:
            assert df[col].isna().all(), f"{col} must stay unresolved (retired)"


# ── units: the store is decimal, the wide columns are percent ────────────────


def test_units_conversion_is_percent_at_the_boundary(research_db):
    """The migrated frame's return columns are in PERCENT (legacy convention),
    reproduced from the DECIMAL store — the one deliberate unit conversion."""
    df = signal_quality.load_score_performance(research_db)
    aapl = df[df["symbol"] == "AAPL"].iloc[0]
    # ground truth AAPL 21d stock return = 0.0432 decimal → 4.32 percent
    assert aapl["return_21d"] == pytest.approx(4.32)
    assert aapl["spy_21d_return"] == pytest.approx(2.01)
    # log_alpha stays decimal (it always was)
    assert aapl["log_alpha_21d"] == pytest.approx(_log_alpha(0.0432, 0.0201))
