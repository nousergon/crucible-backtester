"""alpha-engine-config-I9708: zero rejected stocks is not an `ok` leakage
measurement. Leakage is "% of rejected stocks that beat SPY", undefined over
zero rejections; analyze() used to return `ok` with a fabricated 0.0."""

import sqlite3

from optimizer import scanner_optimizer


def _db(tmp_path, n_weeks: int = 10) -> str:
    path = tmp_path / "research.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE scanner_evaluations (ticker TEXT, eval_date TEXT, quant_filter_pass INT)")
    conn.execute(
        "CREATE TABLE universe_returns (ticker TEXT, eval_date TEXT, return_5d REAL, beat_spy_5d INT)"
    )
    for w in range(n_weeks):
        d = f"2026-08-{w + 1:02d}"
        conn.execute("INSERT INTO scanner_evaluations VALUES ('AAPL', ?, 1)", (d,))
        conn.execute("INSERT INTO universe_returns VALUES ('AAPL', ?, 0.01, 1)", (d,))
    conn.commit()
    conn.close()
    return str(path)


def test_no_rejected_stocks_is_insufficient_data_not_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(
        scanner_optimizer.retired_table_guard, "source_freshness",
        lambda *a, **k: {"stale": False},
    )
    result = scanner_optimizer.analyze(_db(tmp_path))
    assert result["status"] == "insufficient_data"
    assert "leakage_rate" not in result
    # recommend() passes the non-ok status through, so nothing is actuated.
    assert scanner_optimizer.recommend(result, {})["status"] == "insufficient_data"
