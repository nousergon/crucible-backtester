"""Tests for analysis.trigger_scorecard — entry-trigger effectiveness from trades.db."""

import sqlite3

import pytest

from analysis.trigger_scorecard import (
    _categorize_trigger,
    _safe_mean,
    _win_rate,
    compute_trigger_scorecard,
)


# ── Categorization helpers ──────────────────────────────────────────────────


@pytest.mark.parametrize("raw,expected", [
    ("pullback", "pullback"),
    ("Pullback-1.5x", "pullback"),
    ("VWAP_discount", "vwap"),
    ("support_bounce", "support"),
    ("time_expiry_355pm", "time_expiry"),
    ("custom_thing", "other"),
    (None, "unknown"),
    ("", "unknown"),
])
def test_categorize_trigger_known_and_edge(raw, expected):
    assert _categorize_trigger(raw) == expected


def test_safe_mean_empty_series_returns_none():
    import pandas as pd
    assert _safe_mean(pd.Series([], dtype=float)) is None


def test_safe_mean_with_nans():
    import pandas as pd
    s = pd.Series([1.0, 2.0, None, 3.0])
    assert _safe_mean(s) == pytest.approx(2.0)


def test_win_rate_empty_returns_none():
    import pandas as pd
    assert _win_rate(pd.Series([], dtype=float)) is None


def test_win_rate_majority_wins():
    import pandas as pd
    s = pd.Series([0.05, 0.02, -0.01, 0.03])
    assert _win_rate(s) == pytest.approx(0.75)


# ── compute_trigger_scorecard (sqlite-backed) ───────────────────────────────


def _build_trades_db(path, rows):
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE trades (
            ticker TEXT,
            date TEXT,
            action TEXT,
            fill_price REAL,
            price_at_order REAL,
            signal_price REAL,
            trigger_type TEXT,
            trigger_price REAL,
            realized_return_pct REAL,
            realized_alpha_pct REAL,
            spy_return_during_hold REAL,
            slippage_vs_signal REAL,
            days_held REAL
        )
    """)
    conn.executemany(
        "INSERT INTO trades VALUES (:ticker,:date,:action,:fill_price,:price_at_order,"
        ":signal_price,:trigger_type,:trigger_price,:realized_return_pct,"
        ":realized_alpha_pct,:spy_return_during_hold,:slippage_vs_signal,:days_held)",
        rows,
    )
    conn.commit()
    conn.close()


def _row(ticker, trigger_type, alpha, return_pct=None, fill=100.0, signal=99.0, open_=99.5, days=10):
    return {
        "ticker": ticker,
        "date": "2026-04-01",
        "action": "ENTER",
        "fill_price": fill,
        "price_at_order": open_,
        "signal_price": signal,
        "trigger_type": trigger_type,
        "trigger_price": signal,
        "realized_return_pct": return_pct if return_pct is not None else alpha + 0.5,
        "realized_alpha_pct": alpha,
        "spy_return_during_hold": 0.5,
        "slippage_vs_signal": fill - signal,
        "days_held": days,
    }


def test_compute_trigger_scorecard_missing_db_returns_error(tmp_path):
    result = compute_trigger_scorecard(str(tmp_path / "no_such.db"))
    assert result["status"] == "error"
    assert "trades.db not found" in result["error"]


def test_compute_trigger_scorecard_empty_trades(tmp_path):
    db = tmp_path / "trades.db"
    _build_trades_db(db, [])
    result = compute_trigger_scorecard(str(db))
    assert result["status"] == "insufficient_data"
    assert "no ENTER trades found" in result["error"]


def test_compute_trigger_scorecard_below_min_trades(tmp_path):
    db = tmp_path / "trades.db"
    rows = [_row(f"T{i}", "pullback", 0.5) for i in range(2)]
    _build_trades_db(db, rows)
    result = compute_trigger_scorecard(str(db), min_trades=3)
    assert result["status"] == "insufficient_data"
    assert result["total_entries"] == 2


def test_compute_trigger_scorecard_happy_path_multiple_triggers(tmp_path):
    db = tmp_path / "trades.db"
    rows = []
    # 4 pullback entries — 3 winners (alpha > 0), 1 loser
    rows += [_row(f"P{i}", "pullback", 1.5, fill=101.0, signal=100.0, open_=100.5) for i in range(3)]
    rows += [_row("P3", "pullback", -0.8, fill=101.0, signal=100.0, open_=100.5)]
    # 3 vwap entries — 1 winner, 2 losers
    rows += [_row("V0", "vwap_discount", 0.4, fill=99.0, signal=100.0, open_=99.5)]
    rows += [_row(f"V{i+1}", "vwap_discount", -0.6, fill=99.0, signal=100.0, open_=99.5) for i in range(2)]
    # 2 time_expiry (below default min_trades=3, should be excluded)
    rows += [_row(f"E{i}", "time_expiry", 0.1) for i in range(2)]

    _build_trades_db(db, rows)
    result = compute_trigger_scorecard(str(db))

    assert result["status"] == "ok"
    triggers = {t["trigger"]: t for t in result["triggers"]}
    assert set(triggers) == {"pullback", "vwap"}  # time_expiry below threshold
    assert triggers["pullback"]["n_trades"] == 4
    assert triggers["pullback"]["tp"] == 3
    assert triggers["pullback"]["fp"] == 1
    assert triggers["pullback"]["precision"] == pytest.approx(0.75)
    assert triggers["pullback"]["win_rate_vs_spy"] == pytest.approx(0.75)
    assert triggers["vwap"]["n_trades"] == 3
    assert triggers["vwap"]["tp"] == 1
    assert triggers["vwap"]["fp"] == 2

    # Slippage signed correctly: pullback fill 101 vs signal 100 → +1.0%
    assert triggers["pullback"]["avg_slippage_vs_signal"] == pytest.approx(1.0)
    # vwap fill 99 vs signal 100 → -1.0%
    assert triggers["vwap"]["avg_slippage_vs_signal"] == pytest.approx(-1.0)

    summary = result["summary"]
    assert summary["total_entries"] == 9
    # Overall TP/FP from realized_alpha_pct: 4 positive (3 pullback + 1 vwap + 2 time_expiry above zero)
    # time_expiry alpha = 0.1 > 0, so total TP = 3 + 1 + 2 = 6, FP = 1 + 2 = 3
    assert summary["tp"] == 6
    assert summary["fp"] == 3


def test_compute_trigger_scorecard_skips_invalid_slippage_when_signal_missing(tmp_path):
    db = tmp_path / "trades.db"
    rows = []
    for i in range(3):
        r = _row(f"T{i}", "pullback", 0.5)
        r["signal_price"] = None  # invalid → should skip slippage
        rows.append(r)
    _build_trades_db(db, rows)
    result = compute_trigger_scorecard(str(db))
    assert result["status"] == "ok"
    assert result["triggers"][0]["avg_slippage_vs_signal"] is None


def test_compute_trigger_scorecard_db_open_error_propagates_safely(tmp_path, monkeypatch):
    db = tmp_path / "trades.db"
    _build_trades_db(db, [_row("X", "pullback", 0.5) for _ in range(3)])

    def broken_connect(_path):
        raise sqlite3.OperationalError("simulated open failure")

    monkeypatch.setattr("analysis.trigger_scorecard.sqlite3.connect", broken_connect)
    result = compute_trigger_scorecard(str(db))
    assert result["status"] == "error"
    assert "simulated open failure" in result["error"]


def test_compute_trigger_scorecard_unknown_trigger_excluded_below_min(tmp_path):
    db = tmp_path / "trades.db"
    rows = [_row(f"U{i}", None, 0.3) for i in range(2)]
    _build_trades_db(db, rows)
    result = compute_trigger_scorecard(str(db), min_trades=3)
    assert result["status"] == "insufficient_data"
    assert result["total_entries"] == 2
