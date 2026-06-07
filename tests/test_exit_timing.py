"""Tests for analysis.exit_timing — MFE/MAE on completed roundtrip trades."""

import sqlite3
from unittest.mock import patch

import pandas as pd
import pytest

from analysis.exit_timing import compute_exit_timing


# ── DB seeding ──────────────────────────────────────────────────────────────


def _build_trades_db(path, roundtrips):
    """roundtrips: list of dicts with keys ticker, entry_date, exit_date, entry_price, exit_price, realized_return_pct, exit_type, days_held."""
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE trades (
            trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_trade_id INTEGER,
            ticker TEXT,
            date TEXT,
            action TEXT,
            fill_price REAL,
            signal_price REAL,
            trigger_type TEXT,
            realized_return_pct REAL,
            realized_alpha_pct REAL,
            days_held REAL
        )
    """)
    for rt in roundtrips:
        cur = conn.execute(
            "INSERT INTO trades(ticker, date, action, fill_price, signal_price, trigger_type, "
            "realized_return_pct, realized_alpha_pct, days_held) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (rt["ticker"], rt["entry_date"], "ENTER", rt["entry_price"],
             rt.get("signal_price", rt["entry_price"]), "pullback",
             None, None, None),
        )
        entry_id = cur.lastrowid
        conn.execute(
            "INSERT INTO trades(entry_trade_id, ticker, date, action, fill_price, signal_price, "
            "trigger_type, realized_return_pct, realized_alpha_pct, days_held) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (entry_id, rt["ticker"], rt["exit_date"], "EXIT", rt["exit_price"],
             rt.get("signal_price", rt["entry_price"]), rt.get("exit_type", "atr_stop"),
             rt.get("realized_return_pct"), rt.get("realized_alpha_pct"),
             rt.get("days_held", 5)),
        )
    conn.commit()
    conn.close()


def _price_df(highs, lows, dates=None):
    if dates is None:
        dates = pd.date_range("2026-04-01", periods=len(highs), freq="D")
    return pd.DataFrame({"High": highs, "Low": lows}, index=pd.DatetimeIndex(dates))


# ── Tests ──────────────────────────────────────────────────────────────────


def test_compute_exit_timing_missing_db_returns_error(tmp_path):
    result = compute_exit_timing(str(tmp_path / "no_such.db"))
    assert result["status"] == "error"
    assert "trades.db not found" in result["error"]


def test_compute_exit_timing_insufficient_roundtrips(tmp_path):
    db = tmp_path / "trades.db"
    _build_trades_db(db, [{
        "ticker": "A", "entry_date": "2026-04-01", "exit_date": "2026-04-05",
        "entry_price": 100.0, "exit_price": 102.0, "realized_return_pct": 2.0,
    }])
    result = compute_exit_timing(str(db), min_roundtrips=5)
    assert result["status"] == "insufficient_data"
    assert "need >= 5" in result["error"]


def test_compute_exit_timing_no_price_cache_returns_error(tmp_path):
    db = tmp_path / "trades.db"
    roundtrips = [{
        "ticker": f"T{i}", "entry_date": "2026-04-01", "exit_date": "2026-04-05",
        "entry_price": 100.0, "exit_price": 105.0, "realized_return_pct": 5.0,
    } for i in range(5)]
    _build_trades_db(db, roundtrips)

    with patch("analysis.exit_timing._load_price_cache", return_value={}):
        result = compute_exit_timing(str(db))

    assert result["status"] == "error"
    assert "no price cache" in result["error"]


def test_compute_exit_timing_happy_path_well_timed(tmp_path):
    db = tmp_path / "trades.db"
    roundtrips = []
    for i in range(6):
        roundtrips.append({
            "ticker": f"T{i}",
            "entry_date": "2026-04-01",
            "exit_date": "2026-04-05",
            "entry_price": 100.0,
            "exit_price": 105.0,
            "realized_return_pct": 5.0,
            "exit_type": "atr_stop",
        })
    _build_trades_db(db, roundtrips)

    # MFE = 6%, MAE = -2%, realized 5% → capture 0.83 (well_timed)
    price_cache = {f"T{i}": _price_df([100, 104, 106, 105, 104], [100, 99, 102, 98, 103])
                   for i in range(6)}

    with patch("analysis.exit_timing._load_price_cache", return_value=price_cache):
        result = compute_exit_timing(str(db))

    assert result["status"] == "ok"
    assert result["n_roundtrips"] == 6
    assert result["summary"]["avg_mfe"] == pytest.approx(6.0)
    assert result["summary"]["avg_mae"] == pytest.approx(-2.0)
    assert result["summary"]["avg_realized_return"] == pytest.approx(5.0)
    assert result["diagnosis"] in ("exits_well_timed", "exits_could_improve")
    assert len(result["by_exit_type"]) == 1
    assert result["by_exit_type"][0]["exit_type"] == "atr_stop"
    assert result["by_exit_type"][0]["n"] == 6


def test_compute_exit_timing_diagnoses_exits_too_early(tmp_path):
    db = tmp_path / "trades.db"
    roundtrips = []
    for i in range(6):
        roundtrips.append({
            "ticker": f"T{i}",
            "entry_date": "2026-04-01",
            "exit_date": "2026-04-05",
            "entry_price": 100.0,
            "exit_price": 101.0,
            "realized_return_pct": 1.0,  # took only 1% out of 10% MFE
            "exit_type": "atr_stop",
        })
    _build_trades_db(db, roundtrips)

    price_cache = {f"T{i}": _price_df([100, 105, 110, 108, 101], [100, 99, 100, 98, 99])
                   for i in range(6)}

    with patch("analysis.exit_timing._load_price_cache", return_value=price_cache):
        result = compute_exit_timing(str(db))

    assert result["status"] == "ok"
    # MFE=10, realized=1 → winner-capture 0.1 < 0.40 → exits_too_early
    assert result["summary"]["capture_winners_median"] == pytest.approx(0.10)
    assert result["diagnosis"] == "exits_too_early"


def test_compute_exit_timing_skips_tickers_missing_from_cache(tmp_path):
    db = tmp_path / "trades.db"
    roundtrips = []
    for i in range(6):
        roundtrips.append({
            "ticker": f"T{i}",
            "entry_date": "2026-04-01",
            "exit_date": "2026-04-05",
            "entry_price": 100.0,
            "exit_price": 103.0,
            "realized_return_pct": 3.0,
        })
    _build_trades_db(db, roundtrips)

    # Only 3 tickers have cache data — under min_roundtrips=5 → insufficient
    price_cache = {f"T{i}": _price_df([102, 104, 105], [99, 100, 101]) for i in range(3)}

    with patch("analysis.exit_timing._load_price_cache", return_value=price_cache):
        result = compute_exit_timing(str(db), min_roundtrips=5)

    assert result["status"] == "insufficient_data"
    assert "with price data" in result["error"]


def test_compute_exit_timing_skips_zero_entry_price(tmp_path):
    db = tmp_path / "trades.db"
    roundtrips = []
    for i in range(5):
        roundtrips.append({
            "ticker": f"T{i}",
            "entry_date": "2026-04-01",
            "exit_date": "2026-04-05",
            "entry_price": 0.0 if i == 0 else 100.0,
            "exit_price": 105.0,
            "realized_return_pct": 5.0,
        })
    _build_trades_db(db, roundtrips)

    price_cache = {f"T{i}": _price_df([102, 105], [98, 100]) for i in range(5)}

    with patch("analysis.exit_timing._load_price_cache", return_value=price_cache):
        # 4 valid + 1 skipped (zero entry) → under default min_roundtrips=5
        result = compute_exit_timing(str(db))

    assert result["status"] == "insufficient_data"


def test_compute_exit_timing_falls_back_to_computed_realized_return(tmp_path):
    """When realized_return_pct is NULL, recompute from entry/exit prices."""
    db = tmp_path / "trades.db"
    roundtrips = []
    for i in range(5):
        roundtrips.append({
            "ticker": f"T{i}",
            "entry_date": "2026-04-01",
            "exit_date": "2026-04-05",
            "entry_price": 100.0,
            "exit_price": 104.0,
            "realized_return_pct": None,  # forces fallback
        })
    _build_trades_db(db, roundtrips)

    price_cache = {f"T{i}": _price_df([102, 105], [99, 100]) for i in range(5)}

    with patch("analysis.exit_timing._load_price_cache", return_value=price_cache):
        result = compute_exit_timing(str(db))

    assert result["status"] == "ok"
    assert result["summary"]["avg_realized_return"] == pytest.approx(4.0)


def test_compute_exit_timing_diagnoses_exits_could_improve(tmp_path):
    """Winner-capture median in [0.40, 0.70) → 'exits_could_improve' band."""
    db = tmp_path / "trades.db"
    roundtrips = []
    for i in range(6):
        roundtrips.append({
            "ticker": f"T{i}",
            "entry_date": "2026-04-01",
            "exit_date": "2026-04-05",
            "entry_price": 100.0,
            "exit_price": 105.0,
            "realized_return_pct": 5.0,  # 50% of 10% MFE → in [0.40, 0.70)
            "exit_type": "atr_stop",
        })
    _build_trades_db(db, roundtrips)

    # MFE 10%, MAE -2% → winner-capture 0.50 → exits_could_improve
    price_cache = {f"T{i}": _price_df([102, 105, 110, 108, 103], [100, 99, 100, 98, 99])
                   for i in range(6)}

    with patch("analysis.exit_timing._load_price_cache", return_value=price_cache):
        result = compute_exit_timing(str(db))

    assert result["status"] == "ok"
    assert result["summary"]["capture_winners_median"] == pytest.approx(0.50)
    assert result["diagnosis"] == "exits_could_improve"


def test_compute_exit_timing_diagnoses_exits_well_timed_high_capture(tmp_path):
    """Winner-capture median >= 0.70 → 'exits_well_timed'."""
    db = tmp_path / "trades.db"
    # MFE tiny but realized close to MFE → capture ~0.83
    roundtrips = []
    for i in range(6):
        roundtrips.append({
            "ticker": f"T{i}",
            "entry_date": "2026-04-01",
            "exit_date": "2026-04-05",
            "entry_price": 100.0,
            "exit_price": 100.5,
            "realized_return_pct": 0.5,
            "exit_type": "atr_stop",
        })
    _build_trades_db(db, roundtrips)

    # MFE = 0.6% (60 bps), realized 0.5 → winner-capture 0.83 >= 0.70
    price_cache = {f"T{i}": _price_df([100.6, 100.7, 100.5, 100.6], [99.5, 99.8, 99.9, 99.8])
                   for i in range(6)}

    with patch("analysis.exit_timing._load_price_cache", return_value=price_cache):
        result = compute_exit_timing(str(db))

    assert result["status"] == "ok"
    assert result["diagnosis"] == "exits_well_timed"


def test_capture_robust_to_stopped_out_losers(tmp_path):
    """THE L4554 regression: a few stopped-out losers tank the legacy all-trade
    avg_capture_ratio negative, but the robust winner-capture median stays
    healthy and the diagnosis is NOT a false 'exits_too_early'."""
    db = tmp_path / "trades.db"
    roundtrips = []
    # 8 well-captured winners: MFE ~6%, realized 5% → capture ~0.83.
    for i in range(8):
        roundtrips.append({
            "ticker": f"W{i}", "entry_date": "2026-04-01", "exit_date": "2026-04-05",
            "entry_price": 100.0, "exit_price": 105.0, "realized_return_pct": 5.0,
            "exit_type": "intraday_profit_take",
        })
    # 4 stopped-out losers: briefly up ~0.3% (tiny MFE) then exit at -4%.
    for i in range(4):
        roundtrips.append({
            "ticker": f"L{i}", "entry_date": "2026-04-01", "exit_date": "2026-04-05",
            "entry_price": 100.0, "exit_price": 96.0, "realized_return_pct": -4.0,
            "exit_type": "intraday_trailing_stop",
        })
    _build_trades_db(db, roundtrips)

    cache = {}
    for i in range(8):
        cache[f"W{i}"] = _price_df([103, 105, 106, 105, 104], [100, 99, 102, 98, 103])  # MFE ~6
    for i in range(4):
        cache[f"L{i}"] = _price_df([100.3, 99, 97, 96, 95], [100, 96, 94, 93, 95])  # MFE ~0.3, MAE ~-7

    with patch("analysis.exit_timing._load_price_cache", return_value=cache):
        result = compute_exit_timing(str(db))

    s = result["summary"]
    # Legacy mean is dragged negative by the loser ratios (-4 / 0.3 ≈ -13 each)...
    assert s["avg_capture_ratio"] < 0
    # ...but the robust winner-capture median is healthy and the diagnosis holds.
    assert s["capture_winners_median"] >= 0.70
    assert s["n_winners"] == 8
    assert s["n_losers"] == 4
    assert result["diagnosis"] == "exits_well_timed"


def test_compute_exit_timing_by_exit_type_skips_singletons(tmp_path):
    """exit_type breakdown filters groups with <2 trades."""
    db = tmp_path / "trades.db"
    roundtrips = []
    # 5 of the same exit type (passes overall min_roundtrips) + 1 singleton
    for i in range(5):
        roundtrips.append({
            "ticker": f"T{i}", "entry_date": "2026-04-01", "exit_date": "2026-04-05",
            "entry_price": 100.0, "exit_price": 105.0, "realized_return_pct": 5.0,
            "exit_type": "atr_stop",
        })
    roundtrips.append({
        "ticker": "T_solo", "entry_date": "2026-04-01", "exit_date": "2026-04-05",
        "entry_price": 100.0, "exit_price": 102.0, "realized_return_pct": 2.0,
        "exit_type": "time_exit",
    })
    _build_trades_db(db, roundtrips)
    price_cache = {rt["ticker"]: _price_df([102, 106], [99, 100]) for rt in roundtrips}

    with patch("analysis.exit_timing._load_price_cache", return_value=price_cache):
        result = compute_exit_timing(str(db))

    assert result["status"] == "ok"
    exit_types = {b["exit_type"] for b in result["by_exit_type"]}
    assert "atr_stop" in exit_types
    assert "time_exit" not in exit_types  # singleton filtered


def test_compute_exit_timing_db_query_error_caught(tmp_path, monkeypatch):
    db = tmp_path / "trades.db"
    _build_trades_db(db, [])

    def broken_connect(_path):
        raise sqlite3.OperationalError("simulated query failure")

    monkeypatch.setattr("analysis.exit_timing.sqlite3.connect", broken_connect)
    result = compute_exit_timing(str(db))
    assert result["status"] == "error"
    assert "simulated query failure" in result["error"]


# ── Wave-4: _load_price_cache ArcticDB primary / parquet fallback / parity ────

import io as _io  # noqa: E402

from analysis.exit_timing import _load_price_cache  # noqa: E402


def _pf(n=8, start=100.0):
    idx = pd.date_range("2026-03-01", periods=n, freq="D")
    return pd.DataFrame(
        {"Open": [start] * n, "High": [start] * n, "Low": [start] * n,
         "Close": [float(start + i) for i in range(n)], "Volume": [1] * n},
        index=idx,
    )


class _FakeS3:
    """get_object serving parquet bytes from a {key: DataFrame} map."""

    def __init__(self, store):
        self._store = store

    def get_object(self, Bucket, Key):
        if Key not in self._store:
            raise RuntimeError(f"NoSuchKey {Key}")
        buf = _io.BytesIO()
        self._store[Key].to_parquet(buf)
        buf.seek(0)
        return {"Body": buf}


def test_load_price_cache_arcticdb_primary_no_parquet_needed(monkeypatch):
    """Wave-4 terminal: ArcticDB returns all tickers -> no parquet read,
    no slim, no parity emit."""
    monkeypatch.setattr(
        "analysis.exit_timing.load_universe_ohlcv",
        lambda bucket, symbols: {"AAPL": _pf(), "SPY": _pf(start=500)},
    )
    monkeypatch.setattr("boto3.client", lambda svc: _FakeS3({}))
    cache = _load_price_cache(["AAPL", "SPY"])
    assert set(cache) == {"AAPL", "SPY"}


def test_load_price_cache_falls_back_to_price_cache_when_arctic_empty(monkeypatch):
    """Sole fallback is predictor/price_cache (10y) — the slim leg is gone."""
    monkeypatch.setattr(
        "analysis.exit_timing.load_universe_ohlcv",
        lambda bucket, symbols: {},
    )
    store = {
        # slim is deleted: a slim-only ticker is now unrecoverable.
        "predictor/price_cache_slim/AAPL.parquet": _pf(),
        "predictor/price_cache/MSFT.parquet": _pf(start=300),
    }
    monkeypatch.setattr("boto3.client", lambda svc: _FakeS3(store))
    cache = _load_price_cache(["AAPL", "MSFT", "GONE"])
    # Only the price_cache(10y) leg is consulted now -> MSFT only.
    assert set(cache) == {"MSFT"}


def test_load_price_cache_arctic_failure_is_caught(monkeypatch):
    def _boom(bucket, symbols):
        raise RuntimeError("ArcticDB down")

    monkeypatch.setattr("analysis.exit_timing.load_universe_ohlcv", _boom)
    store = {"predictor/price_cache/AAPL.parquet": _pf()}
    monkeypatch.setattr("boto3.client", lambda svc: _FakeS3(store))
    cache = _load_price_cache(["AAPL"])
    assert set(cache) == {"AAPL"}  # graceful fallback, no raise
