"""Tests for analysis.regime_analysis.

Regression-locks the 2026-05-09 P1 evaluator ERROR: research migration #12
added market_regime as a column on score_performance, so the prior
``SELECT sp.*, ms.market_regime`` produced duplicate column names and
pandas surfaced df["market_regime"] as a DataFrame (not a Series). The
downstream logger / accuracy split crashed on the resulting Series.

Fix: drop the macro_snapshots join entirely. score_performance owns the
fact; pre-migration rows with NULL market_regime are silently excluded
from regime-split metrics.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from analysis.regime_analysis import accuracy_by_regime, load_with_regime


@pytest.fixture
def fresh_db(tmp_path: Path) -> str:
    """research.db with the post-migration #12 score_performance schema."""
    db = tmp_path / "research.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE score_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                score_date TEXT NOT NULL,
                score REAL,
                price_on_date REAL,
                price_5d REAL, return_5d REAL, spy_5d_return REAL,
                beat_spy_5d INTEGER, eval_date_5d TEXT,
                return_21d REAL, spy_21d_return REAL,
                beat_spy_21d INTEGER, eval_date_21d TEXT,
                quant_score REAL, qual_score REAL,
                conviction TEXT, sector_modifier REAL, market_regime TEXT,
                UNIQUE(symbol, score_date)
            )
            """
        )
        # macro_snapshots intentionally also has market_regime — this is what
        # caused the duplicate-column crash on Sat 2026-05-09. Including it
        # in the fixture proves the new query is collision-immune.
        conn.execute(
            """
            CREATE TABLE macro_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                market_regime TEXT
            )
            """
        )
        conn.commit()
    return str(db)


def _seed(db: str, rows: list[dict]) -> None:
    with sqlite3.connect(db) as conn:
        for r in rows:
            cols = list(r.keys())
            placeholders = ", ".join("?" for _ in cols)
            conn.execute(
                f"INSERT INTO score_performance ({', '.join(cols)}) "
                f"VALUES ({placeholders})",
                list(r.values()),
            )
        conn.commit()


# ── load_with_regime — duplicate-column regression ───────────────────────────


class TestLoadWithRegime:

    def test_returns_single_market_regime_series_not_dataframe(self, fresh_db):
        """Sat 2026-05-09 crash repro: df['market_regime'] must be a
        Series. Prior SQL emitted two columns named market_regime; this
        test would fail with the old query even before the logger blew
        up."""
        _seed(fresh_db, [
            {"symbol": "AAPL", "score_date": "2026-05-01",
             "score": 78.0, "price_on_date": 200.0,
             "beat_spy_21d": 1, "market_regime": "bull"},
            {"symbol": "MSFT", "score_date": "2026-05-01",
             "score": 70.0, "price_on_date": 430.0,
             "beat_spy_21d": 0, "market_regime": "bull"},
        ])
        df = load_with_regime(fresh_db)
        assert isinstance(df["market_regime"], pd.Series)
        # Column appears exactly once.
        assert list(df.columns).count("market_regime") == 1
        assert df["market_regime"].notna().sum() == 2

    def test_no_keyerror_on_pre_migration_schema(self, tmp_path):
        """Pre-migration #12 score_performance lacked market_regime entirely.
        Loader must inject it as all-NULL rather than KeyError."""
        db = tmp_path / "legacy.db"
        with sqlite3.connect(db) as conn:
            conn.execute(
                """
                CREATE TABLE score_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    score_date TEXT NOT NULL,
                    score REAL,
                    price_on_date REAL,
                    beat_spy_10d INTEGER,
                    beat_spy_30d INTEGER,
                    eval_date_10d TEXT,
                    eval_date_30d TEXT,
                    UNIQUE(symbol, score_date)
                )
                """
            )
            conn.execute(
                "INSERT INTO score_performance (symbol, score_date, score, price_on_date, beat_spy_10d) "
                "VALUES ('AAPL', '2026-03-01', 75.0, 200.0, 1)"
            )
            conn.commit()
        df = load_with_regime(str(db))
        assert "market_regime" in df.columns
        assert df["market_regime"].isna().all()


# ── accuracy_by_regime — splits correctly across regimes ─────────────────────


class TestAccuracyByRegime:

    def _seed_rows_across_regimes(self, db: str, min_per_regime: int):
        """Seed enough beat_spy_21d-populated rows in two regimes to clear
        signal_quality.MIN_SAMPLES."""
        rows = []
        for i in range(min_per_regime):
            rows.append({
                "symbol": f"BULL{i}", "score_date": "2026-05-01",
                "score": 75.0, "price_on_date": 100.0,
                "beat_spy_21d": i % 2,
                "market_regime": "bull",
            })
        for i in range(min_per_regime):
            rows.append({
                "symbol": f"BEAR{i}", "score_date": "2026-04-15",
                "score": 75.0, "price_on_date": 100.0,
                "beat_spy_21d": (i + 1) % 2,
                "market_regime": "bear",
            })
        _seed(db, rows)

    def test_splits_by_regime(self, fresh_db):
        from analysis.signal_quality import MIN_SAMPLES
        self._seed_rows_across_regimes(fresh_db, MIN_SAMPLES)
        df = load_with_regime(fresh_db)
        results = accuracy_by_regime(df)
        regimes = {r["market_regime"] for r in results}
        assert regimes == {"bull", "bear"}

    def test_returns_empty_when_no_market_regime_column(self):
        """Defensive: if a future schema rev drops market_regime, return
        [] rather than KeyError."""
        df = pd.DataFrame({"beat_spy_10d": [1, 0, 1]})
        assert accuracy_by_regime(df) == []

    def test_below_min_samples_returns_empty(self, fresh_db):
        _seed(fresh_db, [
            {"symbol": "AAPL", "score_date": "2026-05-01",
             "score": 75.0, "price_on_date": 200.0,
             "beat_spy_21d": 1, "market_regime": "bull"},
        ])
        df = load_with_regime(fresh_db)
        assert accuracy_by_regime(df) == []
