"""Tests for analysis.shadow_book — blocked-vs-traded entry forward-return diff."""

import sqlite3

import pytest

from analysis.shadow_book import compute_shadow_book_analysis


# ── DB seeders ─────────────────────────────────────────────────────────────


def _build_trades_db(path, shadow_rows, trade_rows):
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE executor_shadow_book (
            ticker TEXT, date TEXT, block_reason TEXT, research_score REAL,
            prediction_confidence REAL, predicted_direction TEXT,
            intended_position_pct REAL, intended_dollars REAL,
            current_price REAL, market_regime TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO executor_shadow_book VALUES (:ticker,:date,:block_reason,:research_score,"
        ":prediction_confidence,:predicted_direction,:intended_position_pct,:intended_dollars,"
        ":current_price,:market_regime)",
        shadow_rows,
    )
    conn.execute("""
        CREATE TABLE trades (
            ticker TEXT, date TEXT, action TEXT, fill_price REAL,
            realized_return_pct REAL, realized_alpha_pct REAL,
            trigger_type TEXT, days_held REAL
        )
    """)
    conn.executemany(
        "INSERT INTO trades VALUES (:ticker,:date,:action,:fill_price,:realized_return_pct,"
        ":realized_alpha_pct,:trigger_type,:days_held)",
        trade_rows,
    )
    conn.commit()
    conn.close()


def _build_research_db(path, universe_rows):
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE universe_returns (
            ticker TEXT, eval_date TEXT, return_5d REAL, return_10d REAL,
            spy_return_5d REAL, beat_spy_5d INTEGER
        )
    """)
    conn.executemany(
        "INSERT INTO universe_returns VALUES (:ticker,:eval_date,:return_5d,:return_10d,"
        ":spy_return_5d,:beat_spy_5d)",
        universe_rows,
    )
    conn.commit()
    conn.close()


def _shadow_row(ticker, date, reason, score=60.0):
    return {
        "ticker": ticker, "date": date, "block_reason": reason,
        "research_score": score, "prediction_confidence": 0.6,
        "predicted_direction": "UP", "intended_position_pct": 0.05,
        "intended_dollars": 50_000.0, "current_price": 100.0,
        "market_regime": "neutral",
    }


def _trade_row(ticker, date, alpha=0.5):
    return {
        "ticker": ticker, "date": date, "action": "ENTER", "fill_price": 100.0,
        "realized_return_pct": alpha + 0.5, "realized_alpha_pct": alpha,
        "trigger_type": "pullback", "days_held": 5,
    }


def _univ(ticker, date, return_5d, beat=None):
    return {
        "ticker": ticker, "eval_date": date, "return_5d": return_5d,
        "return_10d": return_5d * 1.5, "spy_return_5d": 0.5,
        "beat_spy_5d": int(beat) if beat is not None else (1 if return_5d > 0.5 else 0),
    }


# ── Tests ──────────────────────────────────────────────────────────────────


def test_compute_shadow_book_missing_db_returns_error(tmp_path):
    result = compute_shadow_book_analysis(str(tmp_path / "no_such.db"))
    assert result["status"] == "error"
    assert "trades.db not found" in result["error"]


def test_compute_shadow_book_no_shadow_table_returns_insufficient(tmp_path):
    db = tmp_path / "trades.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE unrelated (x INTEGER)")
    conn.commit()
    conn.close()

    result = compute_shadow_book_analysis(str(db))

    assert result["status"] == "insufficient_data"
    assert "shadow_book schema not present" in result["error"]


def test_compute_shadow_book_empty_shadow(tmp_path):
    db = tmp_path / "trades.db"
    _build_trades_db(db, [], [])
    result = compute_shadow_book_analysis(str(db))
    assert result["status"] == "insufficient_data"
    assert "no blocked entries" in result["error"]


def test_compute_shadow_book_below_min_blocks(tmp_path):
    db = tmp_path / "trades.db"
    shadow = [_shadow_row(f"T{i}", "2026-04-01", "low_score") for i in range(2)]
    _build_trades_db(db, shadow, [])
    result = compute_shadow_book_analysis(str(db), min_blocks=3)
    assert result["status"] == "insufficient_data"
    assert "have 2" in result["error"]


def test_compute_shadow_book_no_research_db_falls_back_to_alpha(tmp_path):
    db = tmp_path / "trades.db"
    shadow = [_shadow_row(f"B{i}", "2026-04-01", "low_score") for i in range(4)]
    trades = [_trade_row(f"T{i}", "2026-04-01", alpha=0.3 + i * 0.1) for i in range(3)]
    _build_trades_db(db, shadow, trades)

    result = compute_shadow_book_analysis(str(db))

    assert result["status"] == "ok"
    assert result["n_blocked"] == 4
    assert result["n_traded"] == 3
    assert result["assessment"] == "insufficient_return_data"
    # No research.db → no blocked_avg_return_5d in result
    assert "blocked_avg_return_5d" not in result
    # But traded alpha fallback should kick in (>=1 non-null realized_alpha)
    assert "traded_avg_alpha" in result


def test_compute_shadow_book_classifies_appropriate_when_guard_helps(tmp_path):
    db = tmp_path / "trades.db"
    research = tmp_path / "research.db"

    # 4 blocked stocks underperform; 4 traded stocks outperform → guard helps
    shadow = [_shadow_row(f"B{i}", "2026-04-01", "low_score") for i in range(4)]
    trades = [_trade_row(f"T{i}", "2026-04-01", alpha=2.0) for i in range(4)]
    universe = (
        [_univ(f"B{i}", "2026-04-01", return_5d=-1.0, beat=0) for i in range(4)] +
        [_univ(f"T{i}", "2026-04-01", return_5d=2.0, beat=1) for i in range(4)]
    )

    _build_trades_db(db, shadow, trades)
    _build_research_db(research, universe)

    result = compute_shadow_book_analysis(str(db), str(research))

    assert result["status"] == "ok"
    assert result["blocked_avg_return_5d"] == pytest.approx(-1.0)
    assert result["traded_avg_return_5d"] == pytest.approx(2.0)
    assert result["guard_lift"] == pytest.approx(3.0)
    assert result["assessment"] == "appropriate"
    # classification dict should be populated when beat_spy_5d is non-null on both sides
    assert "classification" in result
    cls = result["classification"]
    # TP = 4 (correctly blocked underperformers), FP = 0, FN = 0, TN = 4
    assert cls["tp"] == 4
    assert cls["fp"] == 0


def test_compute_shadow_book_classifies_too_tight_when_blocked_outperformed(tmp_path):
    db = tmp_path / "trades.db"
    research = tmp_path / "research.db"

    shadow = [_shadow_row(f"B{i}", "2026-04-01", "low_score") for i in range(4)]
    trades = [_trade_row(f"T{i}", "2026-04-01") for i in range(4)]
    universe = (
        [_univ(f"B{i}", "2026-04-01", return_5d=2.0, beat=1) for i in range(4)] +
        [_univ(f"T{i}", "2026-04-01", return_5d=-1.0, beat=0) for i in range(4)]
    )

    _build_trades_db(db, shadow, trades)
    _build_research_db(research, universe)

    result = compute_shadow_book_analysis(str(db), str(research))

    assert result["status"] == "ok"
    assert result["guard_lift"] == pytest.approx(-3.0)
    assert result["assessment"] == "too_tight"


def test_compute_shadow_book_neutral_when_diff_small(tmp_path):
    db = tmp_path / "trades.db"
    research = tmp_path / "research.db"

    shadow = [_shadow_row(f"B{i}", "2026-04-01", "low_score") for i in range(4)]
    trades = [_trade_row(f"T{i}", "2026-04-01") for i in range(4)]
    universe = (
        [_univ(f"B{i}", "2026-04-01", return_5d=0.4, beat=0) for i in range(4)] +
        [_univ(f"T{i}", "2026-04-01", return_5d=0.5, beat=1) for i in range(4)]
    )

    _build_trades_db(db, shadow, trades)
    _build_research_db(research, universe)

    result = compute_shadow_book_analysis(str(db), str(research))

    assert result["status"] == "ok"
    assert abs(result["guard_lift"]) < 0.5
    assert result["assessment"] == "neutral"


def test_compute_shadow_book_by_reason_breakdown(tmp_path):
    db = tmp_path / "trades.db"

    shadow = (
        [_shadow_row(f"S{i}", "2026-04-01", "low_score", score=45.0) for i in range(3)] +
        [_shadow_row(f"E{i}", "2026-04-02", "veto_predictor", score=70.0) for i in range(2)]
    )
    _build_trades_db(db, shadow, [])

    result = compute_shadow_book_analysis(str(db))

    assert result["status"] == "ok"
    by_reason = {r["block_reason"]: r for r in result["by_reason"]}
    assert by_reason["low_score"]["count"] == 3
    assert by_reason["low_score"]["pct_of_blocks"] == pytest.approx(0.6)
    assert by_reason["low_score"]["avg_score"] == pytest.approx(45.0)
    assert by_reason["veto_predictor"]["count"] == 2
    assert by_reason["veto_predictor"]["pct_of_blocks"] == pytest.approx(0.4)


def test_compute_shadow_book_handles_research_db_join_failure_gracefully(tmp_path):
    """If research.db opens but has no universe_returns table, return falls back."""
    db = tmp_path / "trades.db"
    research = tmp_path / "research.db"

    shadow = [_shadow_row(f"B{i}", "2026-04-01", "low_score") for i in range(3)]
    _build_trades_db(db, shadow, [])

    # Build research.db without universe_returns table
    rconn = sqlite3.connect(research)
    rconn.execute("CREATE TABLE unrelated (x INTEGER)")
    rconn.commit()
    rconn.close()

    result = compute_shadow_book_analysis(str(db), str(research))

    assert result["status"] == "ok"
    assert result["n_blocked"] == 3
    assert result["assessment"] == "insufficient_return_data"


def test_compute_shadow_book_research_db_missing_path_skipped(tmp_path):
    db = tmp_path / "trades.db"
    shadow = [_shadow_row(f"B{i}", "2026-04-01", "low_score") for i in range(3)]
    _build_trades_db(db, shadow, [])

    result = compute_shadow_book_analysis(str(db), str(tmp_path / "ghost.db"))

    assert result["status"] == "ok"
    assert result["assessment"] == "insufficient_return_data"
