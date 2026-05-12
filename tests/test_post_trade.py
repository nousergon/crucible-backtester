"""Tests for analysis.post_trade — unified post-trade weekly analysis."""

import sqlite3

import pandas as pd
import pytest

from analysis.post_trade import (
    _best_by,
    _categorize_exit,
    _categorize_trigger,
    _safe_mean,
    _time_of_day_analysis,
    _win_rate,
    compute_post_trade_analysis,
)


# ── Categorization helpers ──────────────────────────────────────────────────


@pytest.mark.parametrize("raw,expected", [
    ("pullback", "pullback"),
    ("VWAP_discount", "vwap"),
    ("support_bounce", "support"),
    ("time_expiry", "time_expiry"),
    ("expiry_3pm", "time_expiry"),
    ("random_thing", "other"),
    (None, "unknown"),
    ("", "unknown"),
])
def test_categorize_trigger(raw, expected):
    assert _categorize_trigger(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("trailing_stop", "trailing_stop"),
    ("profit_take_2pct", "profit_take"),
    ("time_decay_expiry", "time_decay"),
    ("collapse", "collapse"),
    ("signal_exit", "signal_exit"),
    ("research_exit", "research_exit"),
    ("momentum_break", "momentum"),
    ("custom_thing", "other"),
    (None, "unknown"),
    ("", "unknown"),
])
def test_categorize_exit(raw, expected):
    assert _categorize_exit(raw) == expected


def test_safe_mean_helpers():
    assert _safe_mean(pd.Series([], dtype=float)) is None
    assert _safe_mean(pd.Series([1.0, 2.0, None, 3.0])) == pytest.approx(2.0)


def test_win_rate_helpers():
    assert _win_rate(pd.Series([], dtype=float)) is None
    assert _win_rate(pd.Series([1.0, -0.5, 0.3])) == pytest.approx(2 / 3, abs=0.001)


def test_best_by_returns_none_for_empty():
    assert _best_by([], "avg_alpha_pct") is None
    assert _best_by([{"trigger": "x", "avg_alpha_pct": None}], "avg_alpha_pct") is None


def test_best_by_picks_max_by_trigger():
    items = [
        {"trigger": "a", "avg_alpha_pct": 0.5},
        {"trigger": "b", "avg_alpha_pct": 2.0},
        {"trigger": "c", "avg_alpha_pct": 1.0},
    ]
    assert _best_by(items, "avg_alpha_pct") == "b"


def test_best_by_picks_max_by_exit_rule():
    items = [
        {"exit_rule": "trailing", "avg_alpha_pct": 0.3},
        {"exit_rule": "profit", "avg_alpha_pct": 0.9},
    ]
    assert _best_by(items, "avg_alpha_pct") == "profit"


# ── _time_of_day_analysis helper ────────────────────────────────────────────


def test_time_of_day_analysis_buckets_by_hour():
    df = pd.DataFrame({
        "fill_time": ["2026-04-01T09:30:00", "2026-04-01T10:30:00",
                      "2026-04-01T13:00:00", "2026-04-01T15:00:00",
                      "2026-04-01T15:30:00"],
        "slippage_vs_signal": [0.5, 0.3, 0.1, -0.2, -0.1],
        "realized_alpha_pct": [1.0, 1.5, 0.5, -0.5, -0.3],
    })
    result = _time_of_day_analysis(df)

    bucket_names = {r["time_bucket"] for r in result}
    assert "early (pre-10)" in bucket_names
    assert "morning (10-12)" in bucket_names
    assert "midday (12-14)" in bucket_names
    assert "afternoon (14+)" in bucket_names


def test_time_of_day_analysis_empty_when_no_fill_time():
    df = pd.DataFrame({
        "fill_time": [None, None],
        "slippage_vs_signal": [0.5, 0.3],
        "realized_alpha_pct": [1.0, 1.5],
    })
    assert _time_of_day_analysis(df) == []


def test_time_of_day_analysis_handles_malformed_fill_time():
    df = pd.DataFrame({
        "fill_time": ["garbage", "no-time-here"],
        "slippage_vs_signal": [0.5, 0.3],
        "realized_alpha_pct": [1.0, 1.5],
    })
    # malformed times all fall into "unknown" bucket which is NOT in the
    # canonical bucket order → filtered out, empty result
    assert _time_of_day_analysis(df) == []


# ── compute_post_trade_analysis (sqlite-backed) ─────────────────────────────


def _build_trades_db(path, entry_rows, exit_rows):
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE trades (
            trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_trade_id INTEGER,
            ticker TEXT,
            date TEXT,
            action TEXT,
            fill_price REAL,
            price_at_order REAL,
            signal_price REAL,
            trigger_type TEXT,
            fill_time TEXT,
            realized_return_pct REAL,
            realized_alpha_pct REAL,
            spy_return_during_hold REAL,
            slippage_vs_signal REAL,
            days_held REAL,
            exit_reason TEXT
        )
    """)
    for r in entry_rows:
        conn.execute(
            "INSERT INTO trades(ticker, date, action, fill_price, price_at_order, signal_price, "
            "trigger_type, fill_time, realized_return_pct, realized_alpha_pct, "
            "spy_return_during_hold, slippage_vs_signal, days_held) "
            "VALUES (?,?,'ENTER',?,?,?,?,?,?,?,?,?,?)",
            (r["ticker"], r["date"], r["fill_price"], r.get("price_at_order"),
             r.get("signal_price"), r.get("trigger_type"), r.get("fill_time"),
             r.get("realized_return_pct"), r.get("realized_alpha_pct"),
             r.get("spy_return_during_hold"), r.get("slippage_vs_signal"),
             r.get("days_held")),
        )
    for r in exit_rows:
        conn.execute(
            "INSERT INTO trades(entry_trade_id, ticker, date, action, fill_price, "
            "realized_return_pct, realized_alpha_pct, days_held, exit_reason) "
            "VALUES (?,?,?,'EXIT',?,?,?,?,?)",
            (r.get("entry_trade_id", 1), r["ticker"], r["date"], r["fill_price"],
             r.get("realized_return_pct"), r.get("realized_alpha_pct"),
             r.get("days_held"), r.get("exit_reason")),
        )
    conn.commit()
    conn.close()


def _entry(ticker, alpha=1.0, trigger="pullback", days=5, fill_time="2026-04-01T10:30:00"):
    return {
        "ticker": ticker, "date": "2026-04-01", "fill_price": 100.0,
        "price_at_order": 99.5, "signal_price": 99.0, "trigger_type": trigger,
        "fill_time": fill_time, "realized_return_pct": alpha + 0.5,
        "realized_alpha_pct": alpha, "spy_return_during_hold": 0.5,
        "slippage_vs_signal": 1.0, "days_held": days,
    }


def _exit(ticker, alpha=1.0, reason="trailing_stop", days=5):
    return {
        "ticker": ticker, "date": "2026-04-08", "fill_price": 102.0,
        "realized_return_pct": alpha + 0.5, "realized_alpha_pct": alpha,
        "days_held": days, "exit_reason": reason,
    }


def test_compute_post_trade_missing_db_returns_error(tmp_path):
    result = compute_post_trade_analysis(str(tmp_path / "no_such.db"))
    assert result["status"] == "error"
    assert "trades.db not found" in result["error"]


def test_compute_post_trade_empty_entries(tmp_path):
    db = tmp_path / "trades.db"
    _build_trades_db(db, [], [])
    result = compute_post_trade_analysis(str(db))
    assert result["status"] == "insufficient_data"


def test_compute_post_trade_happy_path(tmp_path):
    db = tmp_path / "trades.db"
    entries = (
        [_entry(f"P{i}", alpha=1.5, trigger="pullback") for i in range(4)] +
        [_entry(f"V{i}", alpha=0.5, trigger="vwap_discount") for i in range(4)]
    )
    exits = (
        [_exit(f"P{i}", alpha=1.5, reason="trailing_stop") for i in range(3)] +
        [_exit(f"V{i}", alpha=0.5, reason="profit_take_5pct") for i in range(3)]
    )
    _build_trades_db(db, entries, exits)

    result = compute_post_trade_analysis(str(db))

    assert result["status"] == "ok"
    triggers = {t["trigger"]: t for t in result["trigger_effectiveness"]}
    assert {"pullback", "vwap"}.issubset(set(triggers))
    assert triggers["pullback"]["n_trades"] == 4

    exits_out = {e["exit_rule"]: e for e in result["exit_effectiveness"]}
    assert {"trailing_stop", "profit_take"}.issubset(set(exits_out))

    assert any(b["bucket"] == "3-5d" for b in result["holding_period"])

    tod = {r["time_bucket"]: r for r in result["time_of_day_slippage"]}
    assert "morning (10-12)" in tod

    summary = result["summary"]
    assert summary["n_entries"] == 8
    assert summary["n_exits"] == 6
    assert summary["best_trigger"] == "pullback"  # higher alpha
    assert summary["best_exit_rule"] == "trailing_stop"  # higher alpha


def test_compute_post_trade_db_error_caught(tmp_path, monkeypatch):
    db = tmp_path / "trades.db"
    _build_trades_db(db, [], [])

    def broken_connect(_path):
        raise sqlite3.OperationalError("simulated failure")

    monkeypatch.setattr("analysis.post_trade.sqlite3.connect", broken_connect)
    result = compute_post_trade_analysis(str(db))
    assert result["status"] == "error"
    assert "simulated failure" in result["error"]


def test_compute_post_trade_no_exits_handled(tmp_path):
    """exit_analysis must return [] when no EXIT rows exist."""
    db = tmp_path / "trades.db"
    entries = [_entry(f"P{i}", alpha=1.0) for i in range(5)]
    _build_trades_db(db, entries, [])

    result = compute_post_trade_analysis(str(db))

    assert result["status"] == "ok"
    assert result["exit_effectiveness"] == []
    assert result["summary"]["n_exits"] == 0
