"""Backtester run-date keys by the NYSE trading day, not the calendar date
(DATE_CONVENTIONS / L4466).

The Saturday SF threads $.run_date = date(Execution.StartTime) (calendar — Sat
2026-05-30) but Research + signals.json + the standalone scanner key by trading
day (Fri 2026-05-29). Keying backtest/{date}/ (incl. pit_parity.json +
parity_metrics) by the calendar date misaligned the backtester with signals and
surfaced as research↔backtester pit-parity drift. These pin the trading-day
normalization at all three chokepoints (pipeline_common helper + both Python
entry points + the spot_backtest.sh bash chokepoint).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from pipeline_common import resolve_trading_day

_REPO = Path(__file__).resolve().parent.parent


class TestResolveTradingDay:
    @pytest.mark.parametrize(
        "given,expected",
        [
            ("2026-05-30", "2026-05-29"),  # Sat → Fri
            ("2026-05-31", "2026-05-29"),  # Sun → Fri
            ("2026-05-29", "2026-05-29"),  # Fri → Fri (idempotent / trading day passes through)
            ("2026-05-25", "2026-05-22"),  # Memorial Day (Mon holiday) → prior Fri
            ("2026-05-30T09:00:00Z", "2026-05-29"),  # tolerates ISO datetime prefix
        ],
    )
    def test_normalizes_to_trading_day(self, given, expected):
        assert resolve_trading_day(given) == expected

    def test_idempotent(self):
        once = resolve_trading_day("2026-05-30")
        assert resolve_trading_day(once) == once

    def test_default_is_a_trading_day_on_or_before_today(self):
        out = resolve_trading_day()
        assert out <= dt.date.today().isoformat()


class TestEntryPointWiring:
    """Both Python entry points must normalize args.date so backtest/{date}/
    artifacts land on the trading-day key (guards against a revert)."""

    @pytest.mark.parametrize("entry", ["backtest.py", "evaluate.py"])
    def test_entry_point_normalizes_args_date(self, entry):
        src = (_REPO / entry).read_text()
        assert "resolve_trading_day(args.date)" in src, (
            f"{entry} must normalize args.date via resolve_trading_day "
            "(trading-day artifact keying, L4466)."
        )

    def test_spot_script_normalizes_run_date(self):
        src = (_REPO / "infrastructure" / "spot_backtest.sh").read_text()
        # Normalizes RUN_DATE via the lib trading-calendar at the bash chokepoint,
        # before threading into --date + the s3 uploads.
        assert "trading_calendar" in src and "_RUN_DATE_TD" in src, (
            "spot_backtest.sh must normalize RUN_DATE to the trading day at the "
            "single bash chokepoint (so python --date and bash s3 uploads never split)."
        )
