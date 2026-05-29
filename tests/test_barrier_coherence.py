"""Tests for analysis/barrier_coherence.py — the predictor↔executor
triple-barrier coherence diagnostic (Task A).

Pure logic + synthetic temp trades.db, mirroring test_analysis_coverage.py.
"""

import sqlite3
import tempfile

import pytest

from analysis.barrier_coherence import (
    compute_barrier_coherence,
    to_barrier_class,
)
from reporter import _section_barrier_coherence


# ---------------------------------------------------------------------------
# to_barrier_class
# ---------------------------------------------------------------------------


class TestToBarrierClass:
    def test_exact_matches(self):
        assert to_barrier_class("profit_take") == "upper_barrier"
        assert to_barrier_class("atr_trailing_stop") == "lower_barrier"
        assert to_barrier_class("fallback_stop") == "lower_barrier"
        assert to_barrier_class("momentum_exit") == "lower_barrier"
        assert to_barrier_class("time_decay_exit") == "vertical_barrier"
        assert to_barrier_class("time_decay_reduce") == "vertical_barrier"
        assert to_barrier_class("catalyst_hard_exit") == "non_barrier"

    def test_case_insensitive_and_whitespace(self):
        assert to_barrier_class("  ATR_Trailing_Stop ") == "lower_barrier"

    def test_substring_variants(self):
        # prefixed / suffixed variants still resolve
        assert to_barrier_class("intraday_profit_take_v2") == "upper_barrier"
        assert to_barrier_class("daemon_atr_trailing_stop") == "lower_barrier"

    def test_generic_keyword_fallback(self):
        assert to_barrier_class("hard_stop") == "lower_barrier"
        assert to_barrier_class("target_hit") == "upper_barrier"
        assert to_barrier_class("max_time_in_trade") == "vertical_barrier"

    def test_empty_is_unknown(self):
        assert to_barrier_class(None) == "unknown"
        assert to_barrier_class("") == "unknown"
        assert to_barrier_class("   ") == "unknown"

    def test_unrecognized_is_other(self):
        assert to_barrier_class("manual_override") == "other"


# ---------------------------------------------------------------------------
# compute_barrier_coherence — definition divergence (no trades needed)
# ---------------------------------------------------------------------------


class TestDefinitionDivergence:
    def test_runs_without_trades_db(self):
        result = compute_barrier_coherence("/nonexistent/trades.db")
        assert result["status"] == "ok"
        assert result["trades_status"] == "error"
        # the static leg is always present
        div = result["definition_divergence"]
        assert div["vertical"]["label_horizon_trading_days"] == 21
        assert div["horizontal"]["coherent"] is False
        # trade-based legs report their error, not silently absent
        assert result["horizon_coherence"]["status"] == "error"
        assert result["barrier_touch_mix"]["status"] == "error"

    def test_injected_exec_params_reflected(self):
        result = compute_barrier_coherence(
            "/nonexistent/trades.db",
            exec_params={"time_decay_exit_days": 21, "atr_multiplier": 3.0},
            exec_params_source="live S3 config/executor_params.json (sweep-tuned)",
        )
        vert = result["definition_divergence"]["vertical"]
        # 21d exec time barrier vs 21d label horizon → horizon gap closes
        assert vert["exec_time_barrier_trading_days"] == 21
        assert vert["horizon_gap_days"] == 0
        assert vert["coherent"] is True
        assert result["exec_params_source"].startswith("live S3")

    def test_default_horizon_gap(self):
        result = compute_barrier_coherence("/nonexistent/trades.db")
        vert = result["definition_divergence"]["vertical"]
        # 21d label vs 14d exec default → 7d gap, incoherent
        assert vert["horizon_gap_days"] == 7
        assert vert["coherent"] is False


# ---------------------------------------------------------------------------
# compute_barrier_coherence — trade-based legs
# ---------------------------------------------------------------------------


def _make_trades_db(rows: list[tuple]) -> str:
    """rows: (action, entry_trade_id, exit_reason, realized_return_pct,
    realized_alpha_pct, days_held)."""
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    conn = sqlite3.connect(f.name)
    conn.execute(
        """
        CREATE TABLE trades (
            ticker TEXT, date TEXT, action TEXT, entry_trade_id TEXT,
            exit_reason TEXT, realized_return_pct REAL,
            realized_alpha_pct REAL, days_held INTEGER
        )
        """
    )
    for i, (action, etid, reason, ret, alpha, dh) in enumerate(rows):
        conn.execute(
            "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?)",
            (f"T{i}", "2026-04-01", action, etid, reason, ret, alpha, dh),
        )
    conn.commit()
    conn.close()
    return f.name


class TestTradeBasedLegs:
    def _populated_db(self):
        rows = [
            # exits (entry_trade_id set) — mix of barrier classes + hold times
            ("EXIT", "e1", "profit_take", 8.0, 4.0, 18),
            ("EXIT", "e2", "atr_trailing_stop", -3.0, -2.0, 6),
            ("EXIT", "e3", "atr_trailing_stop", -2.0, -1.0, 5),
            ("EXIT", "e4", "time_decay_exit", 1.0, 0.5, 14),
            ("REDUCE", "e5", "profit_take", 5.0, 3.0, 22),
            ("EXIT", "e6", "momentum_exit", -4.0, -3.0, 3),
            # an ENTER row (no entry_trade_id) must be excluded
            ("ENTER", None, None, None, None, None),
        ]
        return _make_trades_db(rows)

    def test_horizon_coherence_ok(self):
        db = self._populated_db()
        result = compute_barrier_coherence(db, min_trades=3)
        assert result["trades_status"] == "ok"
        hc = result["horizon_coherence"]
        assert hc["status"] == "ok"
        assert hc["n"] == 6  # ENTER row excluded
        assert hc["label_horizon_days"] == 21
        # 5 of 6 exits held < 21d (only the 22d REDUCE is >= 21)
        assert hc["n_exit_before_label_horizon"] == 5
        assert hc["pct_exit_before_label_horizon"] == pytest.approx(5 / 6, abs=1e-3)

    def test_barrier_touch_mix_ok(self):
        db = self._populated_db()
        result = compute_barrier_coherence(db, min_trades=3)
        mix = result["barrier_touch_mix"]
        assert mix["status"] == "ok"
        assert mix["n"] == 6
        classes = {r["barrier_class"]: r for r in mix["by_class"]}
        # 2 profit_take → upper; 3 stop/momentum → lower; 1 time_decay → vertical
        assert classes["upper_barrier"]["n"] == 2
        assert classes["lower_barrier"]["n"] == 3
        assert classes["vertical_barrier"]["n"] == 1
        assert mix["pct_lower"] == pytest.approx(3 / 6, abs=1e-3)

    def test_insufficient_data(self):
        db = _make_trades_db([("EXIT", "e1", "profit_take", 8.0, 4.0, 18)])
        result = compute_barrier_coherence(db, min_trades=3)
        assert result["status"] == "ok"  # definition leg still ok
        assert result["horizon_coherence"]["status"] == "insufficient_data"
        assert result["barrier_touch_mix"]["status"] == "insufficient_data"

    def test_only_enter_rows(self):
        db = _make_trades_db([("ENTER", None, None, None, None, None)])
        result = compute_barrier_coherence(db, min_trades=3)
        # no roundtrip exits → trade legs insufficient, but no crash
        assert result["trades_status"] == "ok"
        assert result["horizon_coherence"]["status"] == "insufficient_data"


# ---------------------------------------------------------------------------
# reporter section smoke
# ---------------------------------------------------------------------------


class TestReporterSection:
    def test_renders_with_full_result(self):
        db = _make_trades_db([
            ("EXIT", "e1", "profit_take", 8.0, 4.0, 18),
            ("EXIT", "e2", "atr_trailing_stop", -3.0, -2.0, 6),
            ("EXIT", "e3", "time_decay_exit", 1.0, 0.5, 14),
        ])
        result = compute_barrier_coherence(db, min_trades=1)
        lines = _section_barrier_coherence(result)
        text = "\n".join(lines)
        assert "## Barrier coherence" in text
        assert "Definition divergence" in text
        assert "Realized holding period" in text
        assert "Realized barrier-touch mix" in text

    def test_renders_with_no_trades(self):
        result = compute_barrier_coherence("/nonexistent/trades.db")
        lines = _section_barrier_coherence(result)
        text = "\n".join(lines)
        # definition leg renders; trade legs show their unavailable state
        assert "Definition divergence" in text
        assert "Unavailable" in text
