"""Tests for analysis.behavioral_anomaly — L4514/config#698 metric suite.

Mirrors the synthetic-sqlite fixture pattern of test_exit_timing.py.
"""

import json
import sqlite3

import pytest

from analysis.behavioral_anomaly import compute_behavioral_anomaly


# ── DB seeding ──────────────────────────────────────────────────────────────


def _build_trades_db(path, trades=(), eod_rows=()):
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE trades (
            trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_trade_id INTEGER,
            ticker TEXT,
            date TEXT,
            action TEXT,
            realized_alpha_pct REAL,
            slippage_vs_signal REAL
        )
    """)
    conn.execute("""
        CREATE TABLE eod_pnl (
            date TEXT PRIMARY KEY,
            positions_snapshot TEXT
        )
    """)
    id_by_key = {}
    for t in trades:
        cur = conn.execute(
            "INSERT INTO trades(entry_trade_id, ticker, date, action, "
            "realized_alpha_pct, slippage_vs_signal) VALUES (?,?,?,?,?,?)",
            (id_by_key.get(t.get("entry_key")), t["ticker"], t["date"], t["action"],
             t.get("realized_alpha_pct"), t.get("slippage_vs_signal")),
        )
        if t.get("key"):
            id_by_key[t["key"]] = cur.lastrowid
    for d, snapshot in eod_rows:
        conn.execute(
            "INSERT INTO eod_pnl(date, positions_snapshot) VALUES (?,?)",
            (d, json.dumps(snapshot) if snapshot is not None else None),
        )
    conn.commit()
    conn.close()


def _build_research_db(path, scores=()):
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE score_performance (
            symbol TEXT, score_date TEXT, score REAL
        )
    """)
    conn.executemany(
        "INSERT INTO score_performance(symbol, score_date, score) VALUES (?,?,?)",
        scores,
    )
    conn.commit()
    conn.close()


def _snap(*pairs):
    return [{"ticker": t, "market_value": mv} for t, mv in pairs]


# ── top-level contract ──────────────────────────────────────────────────────


def test_missing_trades_db_is_error(tmp_path):
    out = compute_behavioral_anomaly(str(tmp_path / "nope.db"))
    assert out["status"] == "error"


def test_empty_db_is_insufficient(tmp_path):
    db = str(tmp_path / "trades.db")
    _build_trades_db(db)
    out = compute_behavioral_anomaly(db)
    assert out["status"] == "insufficient_data"
    for comp in ("decision_reversal", "conviction_stability",
                 "cost_adjusted_quality", "portfolio_state_drift"):
        assert out[comp]["status"] in ("insufficient_data", "error")


def test_pre_migration_schema_is_error_not_crash(tmp_path):
    db = str(tmp_path / "trades.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE trades (trade_id INTEGER PRIMARY KEY, ticker TEXT)")
    conn.commit()
    conn.close()
    out = compute_behavioral_anomaly(db)
    assert out["status"] == "error"


# ── decision reversal ───────────────────────────────────────────────────────


def test_reversal_detected_inside_window(tmp_path):
    db = str(tmp_path / "trades.db")
    _build_trades_db(db, trades=[
        {"ticker": "AAPL", "date": "2026-06-01", "action": "ENTER"},
        {"ticker": "AAPL", "date": "2026-06-03", "action": "EXIT"},
        {"ticker": "AAPL", "date": "2026-06-08", "action": "ENTER"},  # 5d re-entry
        {"ticker": "MSFT", "date": "2026-06-01", "action": "ENTER"},
        {"ticker": "MSFT", "date": "2026-06-03", "action": "EXIT"},   # never re-entered
    ])
    out = compute_behavioral_anomaly(db)["decision_reversal"]
    assert out["status"] == "ok"
    assert out["n_exits"] == 2
    assert out["n_reversals"] == 1
    assert out["reversal_rate"] == 0.5
    assert out["offenders"][0]["ticker"] == "AAPL"


def test_reentry_outside_window_not_a_reversal(tmp_path):
    db = str(tmp_path / "trades.db")
    _build_trades_db(db, trades=[
        {"ticker": "AAPL", "date": "2026-05-01", "action": "ENTER"},
        {"ticker": "AAPL", "date": "2026-05-05", "action": "EXIT"},
        {"ticker": "AAPL", "date": "2026-06-09", "action": "ENTER"},  # 35d later
    ])
    out = compute_behavioral_anomaly(db)["decision_reversal"]
    assert out["n_reversals"] == 0


# ── conviction stability ────────────────────────────────────────────────────


def test_conviction_stability_flags_high_variance(tmp_path):
    tdb, rdb = str(tmp_path / "trades.db"), str(tmp_path / "research.db")
    _build_trades_db(tdb)
    _build_research_db(rdb, scores=[
        ("AAPL", "2026-06-01", 70), ("AAPL", "2026-06-05", 71), ("AAPL", "2026-06-09", 69),
        ("NVDA", "2026-06-01", 40), ("NVDA", "2026-06-05", 80), ("NVDA", "2026-06-09", 45),
    ])
    out = compute_behavioral_anomaly(tdb, research_db_path=rdb)["conviction_stability"]
    assert out["status"] == "ok"
    assert out["n_tickers"] == 2
    assert out["high_variance"][0]["ticker"] == "NVDA"
    assert out["high_variance"][0]["score_std"] > out["median_score_std"]


def test_conviction_stability_without_research_db(tmp_path):
    db = str(tmp_path / "trades.db")
    _build_trades_db(db)
    out = compute_behavioral_anomaly(db, research_db_path=None)["conviction_stability"]
    assert out["status"] == "insufficient_data"


def test_conviction_min_scores_gate(tmp_path):
    tdb, rdb = str(tmp_path / "trades.db"), str(tmp_path / "research.db")
    _build_trades_db(tdb)
    _build_research_db(rdb, scores=[("AAPL", "2026-06-01", 70), ("AAPL", "2026-06-05", 75)])
    out = compute_behavioral_anomaly(tdb, research_db_path=rdb)["conviction_stability"]
    assert out["status"] == "insufficient_data"  # < 3 scores per ticker


# ── cost-adjusted quality ───────────────────────────────────────────────────


def test_cost_adjusted_quality_units_and_drag(tmp_path):
    db = str(tmp_path / "trades.db")
    _build_trades_db(db, trades=[
        # winner with heavy slippage: gross +2%, slippage fraction 0.01 -> 1%
        {"ticker": "AAPL", "date": "2026-06-01", "action": "ENTER", "key": "a",
         "slippage_vs_signal": 0.01},
        {"ticker": "AAPL", "date": "2026-06-05", "action": "EXIT", "entry_key": "a",
         "realized_alpha_pct": 2.0},
        # winner with negligible slippage
        {"ticker": "MSFT", "date": "2026-06-01", "action": "ENTER", "key": "b",
         "slippage_vs_signal": 0.0001},
        {"ticker": "MSFT", "date": "2026-06-06", "action": "EXIT", "entry_key": "b",
         "realized_alpha_pct": 3.0},
    ])
    out = compute_behavioral_anomaly(db)["cost_adjusted_quality"]
    assert out["status"] == "ok"
    assert out["n_roundtrips"] == 2
    # slippage fraction converted to percent: median of (1.0, 0.01) = 0.505
    assert out["median_slippage_pct"] == pytest.approx(0.505, abs=1e-3)
    # net medians subtract converted slippage: (2-1=1, 3-0.01=2.99) -> median 1.995
    assert out["median_net_alpha_pct"] == pytest.approx(1.995, abs=1e-3)
    # AAPL slippage 1% > 25% of 2% gross -> dragged; MSFT not
    assert out["n_cost_dragged_winners"] == 1
    assert out["cost_drag_fraction"] == 0.5


# ── portfolio-state drift ───────────────────────────────────────────────────


def test_state_drift_l1_one_way(tmp_path):
    db = str(tmp_path / "trades.db")
    _build_trades_db(db, eod_rows=[
        ("2026-06-01", _snap(("AAPL", 50.0), ("MSFT", 50.0))),
        # full swap of half the book: one-way L1 = 0.5
        ("2026-06-02", _snap(("AAPL", 50.0), ("NVDA", 50.0))),
        ("2026-06-03", _snap(("AAPL", 50.0), ("NVDA", 50.0))),  # no change
    ])
    out = compute_behavioral_anomaly(db)["portfolio_state_drift"]
    assert out["status"] == "ok"
    assert out["n_days"] == 2
    assert out["max_daily_drift"] == pytest.approx(0.5)
    assert out["median_daily_drift"] == pytest.approx(0.25)
    assert out["n_spike_days"] == 1  # 0.5 > default 0.25 threshold


def test_state_drift_skips_unparseable_snapshots(tmp_path):
    db = str(tmp_path / "trades.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE trades (trade_id INTEGER PRIMARY KEY, ticker TEXT, date TEXT, action TEXT, entry_trade_id INTEGER, realized_alpha_pct REAL, slippage_vs_signal REAL)")
    conn.execute("CREATE TABLE eod_pnl (date TEXT PRIMARY KEY, positions_snapshot TEXT)")
    conn.execute("INSERT INTO eod_pnl VALUES ('2026-06-01', 'not json')")
    conn.execute("INSERT INTO eod_pnl VALUES ('2026-06-02', ?)",
                 (json.dumps(_snap(("AAPL", 100.0))),))
    conn.commit()
    conn.close()
    out = compute_behavioral_anomaly(db)["portfolio_state_drift"]
    assert out["status"] == "insufficient_data"
    assert out["n_unparseable"] == 1


def test_config_overrides(tmp_path):
    db = str(tmp_path / "trades.db")
    _build_trades_db(db, trades=[
        {"ticker": "AAPL", "date": "2026-06-01", "action": "ENTER"},
        {"ticker": "AAPL", "date": "2026-06-03", "action": "EXIT"},
        {"ticker": "AAPL", "date": "2026-06-08", "action": "ENTER"},  # 5d gap
    ])
    tight = compute_behavioral_anomaly(db, config={"reversal_window_days": 3})
    assert tight["decision_reversal"]["n_reversals"] == 0
    loose = compute_behavioral_anomaly(db, config={"reversal_window_days": 7})
    assert loose["decision_reversal"]["n_reversals"] == 1
