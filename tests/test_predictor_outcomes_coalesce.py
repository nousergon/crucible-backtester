"""Backtester COALESCE pattern over predictor_outcomes during the
2026-05-09 21d canonical-alpha transition window.

Validates that analytics readers consume both legacy (`actual_5d_return`,
`correct_5d`) and new horizon-agnostic (`actual_log_alpha`, `correct`)
columns, with the canonical (new) column preferred when both are
populated. The legacy column's pct-points scale is normalized to decimal
via `/100.0` in the SQL fragment.

These tests pin behavior for the transition window — they will need
trivial updates when PR F drops the legacy fallback (~2026-06-06).
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from pipeline_common import (
    ACTIVE_HORIZON_DAYS,
    ALPHA_COALESCE_SQL,
    CORRECT_COALESCE_SQL,
    CURRENT_HORIZON_FILTER_SQL,
    HORIZON_COALESCE_SQL,
    OUTCOMES_GRADED_SQL,
    OUTCOMES_RESOLVED_SQL,
)


def _create_db():
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    conn = sqlite3.connect(f.name)
    conn.execute(
        "CREATE TABLE predictor_outcomes ("
        "id INTEGER PRIMARY KEY, symbol TEXT, prediction_date TEXT, "
        "predicted_direction TEXT, prediction_confidence REAL, "
        "p_up REAL, p_flat REAL, p_down REAL, "
        "actual_5d_return REAL, correct_5d INTEGER, "
        "actual_log_alpha REAL, horizon_days INTEGER, correct INTEGER)"
    )
    conn.commit()
    conn.close()
    return f.name


def _insert(db, **fields):
    keys = ", ".join(fields.keys())
    placeholders = ", ".join("?" for _ in fields)
    with sqlite3.connect(db) as conn:
        conn.execute(
            f"INSERT INTO predictor_outcomes ({keys}) VALUES ({placeholders})",
            tuple(fields.values()),
        )
        conn.commit()


def _read(db: str) -> pd.DataFrame:
    with sqlite3.connect(db) as conn:
        return pd.read_sql_query(
            f"SELECT *, "
            f"{ALPHA_COALESCE_SQL} AS canonical_actual, "
            f"{CORRECT_COALESCE_SQL} AS canonical_correct, "
            f"{HORIZON_COALESCE_SQL} AS canonical_horizon "
            f"FROM predictor_outcomes",
            conn,
        )


# -- Canonical column preferred when both populated ---------------------------


def test_alpha_coalesce_prefers_new_column():
    """When actual_log_alpha is set, COALESCE returns it (decimal log-units),
    NOT the legacy actual_5d_return."""
    db = _create_db()
    _insert(
        db, symbol="AAPL", prediction_date="2026-04-01",
        predicted_direction="UP",
        actual_log_alpha=0.04, horizon_days=21, correct=1,
        actual_5d_return=2.5, correct_5d=0,  # legacy values disagree
    )
    df = _read(db)
    assert df.iloc[0]["canonical_actual"] == pytest.approx(0.04)
    assert df.iloc[0]["canonical_correct"] == 1
    assert df.iloc[0]["canonical_horizon"] == 21


def test_alpha_coalesce_falls_back_to_legacy_when_new_null():
    """Pre-PR-C rows have actual_log_alpha=NULL, actual_5d_return populated.
    COALESCE returns actual_5d_return / 100 (legacy pct points → decimal)."""
    db = _create_db()
    _insert(
        db, symbol="AAPL", prediction_date="2026-03-01",
        predicted_direction="UP",
        actual_5d_return=2.5, correct_5d=1,  # 2.5pp = 0.025 decimal
    )
    df = _read(db)
    assert df.iloc[0]["canonical_actual"] == pytest.approx(0.025)
    assert df.iloc[0]["canonical_correct"] == 1
    # horizon_days NULL → default 5d (legacy default)
    assert df.iloc[0]["canonical_horizon"] == 5


def test_resolved_predicate_matches_either_column():
    """OUTCOMES_RESOLVED_SQL filter accepts rows with either column populated."""
    db = _create_db()
    _insert(
        db, symbol="A", prediction_date="2026-04-01",
        actual_log_alpha=0.03, horizon_days=21,
    )
    _insert(
        db, symbol="B", prediction_date="2026-03-01",
        actual_5d_return=1.5,
    )
    _insert(
        db, symbol="C", prediction_date="2026-02-01",
        # both NULL → unresolved
    )

    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            f"SELECT symbol FROM predictor_outcomes WHERE {OUTCOMES_RESOLVED_SQL}"
        ).fetchall()
    found = {r[0] for r in rows}
    assert found == {"A", "B"}


def test_graded_predicate_matches_either_column():
    """OUTCOMES_GRADED_SQL filter accepts rows with either correct column populated."""
    db = _create_db()
    _insert(
        db, symbol="A", prediction_date="2026-04-01",
        actual_log_alpha=0.03, correct=1, horizon_days=21,
    )
    _insert(
        db, symbol="B", prediction_date="2026-03-01",
        actual_5d_return=1.5, correct_5d=0,
    )
    _insert(
        db, symbol="C", prediction_date="2026-02-01",
        actual_log_alpha=0.02, horizon_days=21,
        # `correct` NULL even though resolved
    )

    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            f"SELECT symbol FROM predictor_outcomes WHERE {OUTCOMES_GRADED_SQL}"
        ).fetchall()
    found = {r[0] for r in rows}
    assert found == {"A", "B"}


# -- Current-horizon filter for rolling analytics -----------------------------


def test_current_horizon_filter_excludes_legacy_5d_rows():
    """CURRENT_HORIZON_FILTER_SQL scopes rolling analytics to the active
    production horizon so the 21d-log distribution isn't mixed with the
    legacy 5d-arith distribution during the transition window."""
    db = _create_db()
    # Active horizon row (21d log canonical)
    _insert(
        db, symbol="NEW", prediction_date="2026-05-10",
        actual_log_alpha=0.03, horizon_days=ACTIVE_HORIZON_DAYS, correct=1,
    )
    # Legacy 5d row (NULL horizon_days → COALESCE → 5 → filtered out)
    _insert(
        db, symbol="OLD", prediction_date="2026-04-01",
        actual_5d_return=1.5, correct_5d=1,
    )
    # Pathological row with explicit non-active horizon (e.g. 10d) → filtered
    _insert(
        db, symbol="MID", prediction_date="2026-04-15",
        actual_log_alpha=0.02, horizon_days=10, correct=1,
    )

    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            f"SELECT symbol FROM predictor_outcomes "
            f"WHERE {OUTCOMES_RESOLVED_SQL} AND {CURRENT_HORIZON_FILTER_SQL}"
        ).fetchall()
    found = {r[0] for r in rows}
    assert found == {"NEW"}


def test_active_horizon_days_matches_production_config():
    """Smoke gate: ACTIVE_HORIZON_DAYS must match
    alpha-engine-config/predictor/predictor.yaml `labeling.forward_days`.
    If this fails, the constant was bumped without updating production
    config (or vice versa) — fix the mismatch before rolling out."""
    assert ACTIVE_HORIZON_DAYS == 21, (
        "ACTIVE_HORIZON_DAYS drifted from production forward_days=21. "
        "Bump in lockstep with predictor.yaml when migrating horizons."
    )
