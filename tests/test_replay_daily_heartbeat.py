"""tests/test_replay_daily_heartbeat.py — Phase 1 sim-on-every-weekday tests.

Covers the bootstrap-mode replay path that iterates every NYSE trading day
in [bootstrap.as_of, max(dates)] and routes the right signals dict to each
day. See ROADMAP P0 "2026-04-26 (Sun) — Finalize parity + downstream"
for context.

Two layers:

* Unit tests for ``_build_replay_signals_by_date`` — pure function with
  signal_loader patched out.
* Smoke test for ``replay_for_dates`` bootstrap branch — patches
  ``_setup_simulation`` + ``_load_initial_state_from_eod_pnl`` so we can
  observe which sim_dates the loop produces and which signals are routed
  to each one. No S3, no executor.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

import backtest as bt


# ── _build_replay_signals_by_date unit tests ───────────────────────────────

class TestBuildReplaySignalsByDate:
    def _make_loader(self, store: dict[str, dict]):
        """Return a mock signal_loader.load that reads from ``store``.

        Keys absent from ``store`` raise FileNotFoundError — same shape as
        the real signal_loader's behavior on a NoSuchKey response.
        """
        def _load(bucket: str, d: str) -> dict:
            if d not in store:
                raise FileNotFoundError(d)
            return store[d]
        return _load

    def test_signal_day_returns_full_signals(self):
        store = {
            "2026-04-13": {"date": "2026-04-13", "buy_candidates": [{"ticker": "AAPL"}],
                           "universe": [{"ticker": "MSFT"}]},
        }
        with patch("loaders.signal_loader.load", side_effect=self._make_loader(store)):
            out = bt._build_replay_signals_by_date(
                bucket="test", sim_dates=["2026-04-13"],
                signal_dates=["2026-04-13"],
            )
        assert "2026-04-13" in out
        assert out["2026-04-13"]["buy_candidates"] == [{"ticker": "AAPL"}]
        assert out["2026-04-13"]["universe"] == [{"ticker": "MSFT"}]

    def test_non_signal_day_uses_prior_with_stripped_buy_candidates(self):
        store = {
            "2026-04-13": {"date": "2026-04-13",
                           "buy_candidates": [{"ticker": "AAPL"}, {"ticker": "NVDA"}],
                           "universe": [{"ticker": "MSFT"}]},
        }
        with patch("loaders.signal_loader.load", side_effect=self._make_loader(store)):
            out = bt._build_replay_signals_by_date(
                bucket="test",
                sim_dates=["2026-04-13", "2026-04-14", "2026-04-15"],
                signal_dates=["2026-04-13"],
            )
        # 4-13 is the signal day — full signals
        assert out["2026-04-13"]["buy_candidates"] == [
            {"ticker": "AAPL"}, {"ticker": "NVDA"},
        ]
        # 4-14 + 4-15 carry forward 4-13's universe but with empty candidates
        assert out["2026-04-14"]["buy_candidates"] == []
        assert out["2026-04-14"]["universe"] == [{"ticker": "MSFT"}]
        assert out["2026-04-15"]["buy_candidates"] == []
        assert out["2026-04-15"]["universe"] == [{"ticker": "MSFT"}]

    def test_does_not_mutate_cached_signals(self):
        """Stripping buy_candidates on carry-forward must not bleed into
        the next signal-day load. A shared dict reference would mean
        the strip leaks across all carry-forward days simultaneously."""
        store = {
            "2026-04-13": {"date": "2026-04-13",
                           "buy_candidates": [{"ticker": "AAPL"}],
                           "universe": [{"ticker": "MSFT"}]},
            "2026-04-20": {"date": "2026-04-20",
                           "buy_candidates": [{"ticker": "NVDA"}],
                           "universe": [{"ticker": "MSFT"}]},
        }
        with patch("loaders.signal_loader.load", side_effect=self._make_loader(store)):
            out = bt._build_replay_signals_by_date(
                bucket="test",
                sim_dates=["2026-04-13", "2026-04-14", "2026-04-20"],
                signal_dates=["2026-04-13", "2026-04-20"],
            )
        # The 4-14 strip must NOT have mutated the cached 4-13 signals.
        assert out["2026-04-13"]["buy_candidates"] == [{"ticker": "AAPL"}]
        assert out["2026-04-14"]["buy_candidates"] == []
        assert out["2026-04-20"]["buy_candidates"] == [{"ticker": "NVDA"}]

    def test_no_prior_signal_skips_day(self):
        store = {"2026-04-20": {"date": "2026-04-20", "buy_candidates": [],
                                 "universe": []}}
        with patch("loaders.signal_loader.load", side_effect=self._make_loader(store)):
            out = bt._build_replay_signals_by_date(
                bucket="test",
                sim_dates=["2026-04-15", "2026-04-16", "2026-04-20"],
                signal_dates=["2026-04-20"],
            )
        # 4-15 + 4-16 predate the only signal — nothing to carry forward
        assert "2026-04-15" not in out
        assert "2026-04-16" not in out
        assert "2026-04-20" in out

    def test_signal_load_filenotfound_skips_day(self):
        """When a signal-date's file is missing in S3, we must skip that day
        rather than poisoning the cache and sweeping later carry-forward
        days into an unhandled error."""
        store = {
            "2026-04-13": {"date": "2026-04-13", "buy_candidates": [], "universe": []},
            # 2026-04-20 deliberately absent from store
        }
        with patch("loaders.signal_loader.load", side_effect=self._make_loader(store)):
            out = bt._build_replay_signals_by_date(
                bucket="test",
                sim_dates=["2026-04-13", "2026-04-14", "2026-04-20", "2026-04-21"],
                signal_dates=["2026-04-13", "2026-04-20"],
            )
        # 4-13 OK; 4-14 carries forward 4-13.
        assert "2026-04-13" in out
        assert "2026-04-14" in out and out["2026-04-14"]["buy_candidates"] == []
        # 4-20 itself fails to load — skip
        assert "2026-04-20" not in out
        # 4-21 would carry forward 4-20 (the most-recent signal date).
        # Since 4-20 fails to load, 4-21 is also skipped — we never silently
        # fall back further to 4-13 for a day whose nearest signal is missing.
        assert "2026-04-21" not in out

    def test_cache_loads_each_signal_only_once(self):
        store = {
            "2026-04-13": {"date": "2026-04-13", "buy_candidates": [], "universe": []},
        }
        loader = MagicMock(side_effect=self._make_loader(store))
        with patch("loaders.signal_loader.load", loader):
            bt._build_replay_signals_by_date(
                bucket="test",
                sim_dates=["2026-04-13", "2026-04-14", "2026-04-15", "2026-04-16"],
                signal_dates=["2026-04-13"],
            )
        # 4-13 loaded once for itself; 4-14/15/16 reuse the cache, 0 reloads.
        assert loader.call_count == 1


# ── replay_for_dates bootstrap-branch smoke ────────────────────────────────

class TestReplayForDatesBootstrapDailyHeartbeat:
    """Smoke test that the bootstrap branch iterates every trading day in
    [as_of, max(dates)] and routes signals correctly per day. Mocks out
    the heavy paths (executor, ArcticDB, S3) but exercises the real
    iteration logic in replay_for_dates."""

    def _trading_day_index(self, dates_iso: list[str]) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(pd.to_datetime(dates_iso))

    def _setup_simulation_stub(self, signal_dates_iso: list[str],
                               trading_days_iso: list[str]):
        """Return a ``_setup_simulation`` substitute that yields a deterministic
        all_signal_dates / price_matrix.index without touching S3."""
        executor_run = MagicMock(return_value=[])
        sim_client_class = MagicMock()
        sim_client_class.return_value = MagicMock(
            _cash=0.0, _positions={}, _peak_nav=0.0,
            _prices={}, _simulation_date=None,
        )
        price_matrix = pd.DataFrame(
            index=self._trading_day_index(trading_days_iso),
            data={"AAPL": [100.0] * len(trading_days_iso)},
        )
        return (
            executor_run, sim_client_class, list(signal_dates_iso),
            price_matrix, 1_000_000.0, {},
        )

    def test_bootstrap_iterates_every_trading_day_in_window(self):
        # Window: 2026-04-13 (Mon) → 2026-04-17 (Fri). Signals only on Monday.
        # Bootstrap as_of = 2026-04-13. Requested dates = [2026-04-17] (Fri).
        # Expected sim_dates: every day in price_matrix.index intersecting
        # [2026-04-13, 2026-04-17] = all 5 weekdays.
        signal_dates = ["2026-04-13"]
        trading_days = ["2026-04-13", "2026-04-14", "2026-04-15",
                        "2026-04-16", "2026-04-17"]
        setup = self._setup_simulation_stub(signal_dates, trading_days)

        signals_store = {
            "2026-04-13": {
                "date": "2026-04-13",
                "buy_candidates": [{"ticker": "AAPL", "signal": "ENTER"}],
                "universe": [{"ticker": "MSFT", "signal": "HOLD"}],
            },
        }

        def _load(bucket: str, d: str) -> dict:
            if d not in signals_store:
                raise FileNotFoundError(d)
            return signals_store[d]

        bootstrap = {
            "positions": {}, "cash": 1_000_000.0, "peak_nav": 1_000_000.0,
            "as_of": "2026-04-13",
        }

        # Capture the signals_override seen by _simulate_single_date per call.
        seen: list[tuple[str, dict | None]] = []

        def _simulate_single_date_stub(**kwargs):
            seen.append((kwargs["signal_date"], kwargs["signals_override"]))
            return [], None

        with (
            patch.object(bt, "_setup_simulation", return_value=setup),
            patch.object(bt, "_load_initial_state_from_eod_pnl", return_value=bootstrap),
            patch("loaders.signal_loader.load", side_effect=_load),
            patch.object(bt, "_simulate_single_date", side_effect=_simulate_single_date_stub),
            patch("nousergon_lib.arcticdb.get_universe_symbols", return_value={"AAPL", "MSFT"}),
        ):
            bt.replay_for_dates(["2026-04-17"], {
                "trades_db_path": "/tmp/fake.db",
                "signals_bucket": "test",
                "init_cash": 1_000_000.0,
                "executor_paths": ["/tmp/nonexistent"],  # never used: _setup is patched
            })

        sim_dates_seen = [d for d, _ in seen]
        # Every trading day in the window — the daily heartbeat
        assert sim_dates_seen == [
            "2026-04-13", "2026-04-14", "2026-04-15",
            "2026-04-16", "2026-04-17",
        ]
        # Monday gets full signals (signal day)
        assert seen[0][1]["buy_candidates"] == [{"ticker": "AAPL", "signal": "ENTER"}]
        # Tue–Fri get carry-forward with stripped buy_candidates
        for _, sig in seen[1:]:
            assert sig is not None
            assert sig["buy_candidates"] == []
            assert sig["universe"] == [{"ticker": "MSFT", "signal": "HOLD"}]

    def test_bootstrap_skips_days_with_no_prior_signals(self):
        # Bootstrap as_of = 2026-04-13. Earliest signal date = 2026-04-15.
        # Trading days 2026-04-13/14 must be skipped (no prior signal yet).
        signal_dates = ["2026-04-15"]
        trading_days = ["2026-04-13", "2026-04-14", "2026-04-15", "2026-04-16"]
        setup = self._setup_simulation_stub(signal_dates, trading_days)

        signals_store = {
            "2026-04-15": {
                "date": "2026-04-15",
                "buy_candidates": [], "universe": [],
            },
        }

        def _load(bucket: str, d: str) -> dict:
            if d not in signals_store:
                raise FileNotFoundError(d)
            return signals_store[d]

        bootstrap = {
            "positions": {}, "cash": 1_000_000.0, "peak_nav": 1_000_000.0,
            "as_of": "2026-04-13",
        }

        seen: list[str] = []

        def _simulate_single_date_stub(**kwargs):
            seen.append(kwargs["signal_date"])
            return [], None

        with (
            patch.object(bt, "_setup_simulation", return_value=setup),
            patch.object(bt, "_load_initial_state_from_eod_pnl", return_value=bootstrap),
            patch("loaders.signal_loader.load", side_effect=_load),
            patch.object(bt, "_simulate_single_date", side_effect=_simulate_single_date_stub),
            patch("nousergon_lib.arcticdb.get_universe_symbols", return_value=set()),
        ):
            bt.replay_for_dates(["2026-04-16"], {
                "trades_db_path": "/tmp/fake.db",
                "signals_bucket": "test",
                "init_cash": 1_000_000.0,
                "executor_paths": ["/tmp/nonexistent"],
            })

        # 4-13 + 4-14 predate the only signal — skipped before invoking executor.
        # 4-15 = signal day; 4-16 carries forward.
        assert seen == ["2026-04-15", "2026-04-16"]

    def test_captured_orders_filtered_to_requested(self):
        """Orders fired on non-requested intermediate trading days must NOT
        appear in the captured set — only orders on dates listed in the
        original ``dates`` argument are returned."""
        signal_dates = ["2026-04-13"]
        trading_days = ["2026-04-13", "2026-04-14", "2026-04-15"]
        setup = self._setup_simulation_stub(signal_dates, trading_days)

        signals_store = {
            "2026-04-13": {"date": "2026-04-13",
                           "buy_candidates": [], "universe": []},
        }

        def _load(bucket: str, d: str) -> dict:
            return signals_store[d]

        bootstrap = {
            "positions": {}, "cash": 1_000_000.0, "peak_nav": 1_000_000.0,
            "as_of": "2026-04-13",
        }

        # Every day produces an order; we only want the requested-date one.
        def _simulate_single_date_stub(**kwargs):
            d = kwargs["signal_date"]
            return [{"ticker": "AAPL", "action": "EXIT", "date": d}], None

        with (
            patch.object(bt, "_setup_simulation", return_value=setup),
            patch.object(bt, "_load_initial_state_from_eod_pnl", return_value=bootstrap),
            patch("loaders.signal_loader.load", side_effect=_load),
            patch.object(bt, "_simulate_single_date", side_effect=_simulate_single_date_stub),
            patch("nousergon_lib.arcticdb.get_universe_symbols", return_value=set()),
        ):
            captured = bt.replay_for_dates(["2026-04-15"], {
                "trades_db_path": "/tmp/fake.db",
                "signals_bucket": "test",
                "init_cash": 1_000_000.0,
                "executor_paths": ["/tmp/nonexistent"],
            })

        # Only the 2026-04-15 order is captured; 4-13 + 4-14 evolve sim
        # state but their orders are not part of the parity output.
        assert len(captured) == 1
        assert captured[0]["date"] == "2026-04-15"


# ── per_date_bootstrap branch ──────────────────────────────────────────────

class TestReplayForDatesPerDateBootstrap:
    """Covers the per-parity-date bootstrap mode (ROADMAP P1
    "Per-parity-date bootstrap (alternative parity test mode)").

    Each parity date gets a fresh ``sim_client`` anchored to live's
    eod_pnl row strictly before that date — no cumulative state across
    dates. Mocks the heavy paths but exercises the real per-date
    iteration + bootstrap-call ordering in ``replay_for_dates``."""

    def _setup_simulation_stub(self, trading_days_iso: list[str]):
        """Same setup helper as the bootstrap-daily-heartbeat class — fixed
        price matrix, a SimulatedIBKRClient class that records cash/positions
        assignments through ordinary attribute writes."""
        executor_run = MagicMock(return_value=[])

        def _sim_client_factory(*, prices=None, nav=1_000_000.0):
            return MagicMock(
                _cash=0.0, _positions={}, _peak_nav=0.0,
                _prices=prices or {}, _simulation_date=None,
            )
        sim_client_class = MagicMock(side_effect=_sim_client_factory)
        price_matrix = pd.DataFrame(
            index=pd.DatetimeIndex(pd.to_datetime(trading_days_iso)),
            data={"AAPL": [100.0] * len(trading_days_iso)},
        )
        return (
            executor_run, sim_client_class, [],
            price_matrix, 1_000_000.0, {},
        )

    def test_per_date_bootstrap_calls_load_state_per_date(self):
        # Three parity dates → three separate bootstrap calls, each anchored
        # to its own date. No cumulative state carried between them.
        parity_dates = ["2026-04-13", "2026-04-15", "2026-04-17"]
        setup = self._setup_simulation_stub(parity_dates)

        bootstrap_args_seen: list[str] = []

        def _bootstrap(trades_db_path, parity_date):
            bootstrap_args_seen.append(parity_date)
            return {
                "positions": {f"TKR_{parity_date}": {"shares": 10,
                                                      "avg_cost": 50.0,
                                                      "entry_date": parity_date,
                                                      "sector": "Tech"}},
                "cash": 100_000.0,
                "peak_nav": 1_000_000.0,
                "as_of": parity_date,
            }

        sim_clients_seen: list[MagicMock] = []

        def _simulate(**kwargs):
            sim_clients_seen.append(kwargs["sim_client"])
            return [{"ticker": "AAPL", "action": "EXIT",
                     "date": kwargs["signal_date"]}], None

        with (
            patch.object(bt, "_setup_simulation", return_value=setup),
            patch.object(bt, "_load_initial_state_from_eod_pnl",
                         side_effect=_bootstrap),
            patch.object(bt, "_simulate_single_date", side_effect=_simulate),
            patch("nousergon_lib.arcticdb.get_universe_symbols",
                  return_value={"AAPL"}),
        ):
            captured = bt.replay_for_dates(parity_dates, {
                "trades_db_path": "/tmp/fake.db",
                "signals_bucket": "test",
                "init_cash": 1_000_000.0,
                "executor_paths": ["/tmp/nonexistent"],
            }, per_date_bootstrap=True)

        # Bootstrap called once per parity date, in sorted order.
        assert bootstrap_args_seen == sorted(parity_dates)
        # Three distinct sim_client instances (no reuse across dates).
        assert len(sim_clients_seen) == 3
        assert len({id(c) for c in sim_clients_seen}) == 3
        # Each sim_client carries its date-specific bootstrap state.
        for sim_client, parity_date in zip(sim_clients_seen, sorted(parity_dates)):
            assert sim_client._cash == 100_000.0
            assert sim_client._peak_nav == 1_000_000.0
            assert f"TKR_{parity_date}" in sim_client._positions
        # All three parity dates' orders captured.
        assert sorted(o["date"] for o in captured) == sorted(parity_dates)

    def test_per_date_bootstrap_requires_trades_db_path(self):
        setup = self._setup_simulation_stub(["2026-04-15"])
        with (
            patch.object(bt, "_setup_simulation", return_value=setup),
        ):
            with pytest.raises(RuntimeError, match="trades_db_path"):
                bt.replay_for_dates(["2026-04-15"], {
                    "signals_bucket": "test",
                    "init_cash": 1_000_000.0,
                    "executor_paths": ["/tmp/nonexistent"],
                }, per_date_bootstrap=True)

    def test_per_date_bootstrap_skips_dates_with_no_eod_pnl_row(self):
        # The 2026-03-09 date predates any eod_pnl row in this hypothetical
        # trades.db; the helper returns None and the date is silently dropped
        # from the output with a WARNING log.
        parity_dates = ["2026-03-09", "2026-04-15"]
        setup = self._setup_simulation_stub(parity_dates)

        def _bootstrap(trades_db_path, parity_date):
            if parity_date == "2026-03-09":
                return None
            return {
                "positions": {}, "cash": 100_000.0,
                "peak_nav": 1_000_000.0, "as_of": "2026-04-14",
            }

        def _simulate(**kwargs):
            return [{"ticker": "AAPL", "action": "EXIT",
                     "date": kwargs["signal_date"]}], None

        with (
            patch.object(bt, "_setup_simulation", return_value=setup),
            patch.object(bt, "_load_initial_state_from_eod_pnl",
                         side_effect=_bootstrap),
            patch.object(bt, "_simulate_single_date", side_effect=_simulate),
            patch("nousergon_lib.arcticdb.get_universe_symbols",
                  return_value=set()),
        ):
            captured = bt.replay_for_dates(parity_dates, {
                "trades_db_path": "/tmp/fake.db",
                "signals_bucket": "test",
                "init_cash": 1_000_000.0,
                "executor_paths": ["/tmp/nonexistent"],
            }, per_date_bootstrap=True)

        # 03-09 dropped; only 04-15 in output.
        assert len(captured) == 1
        assert captured[0]["date"] == "2026-04-15"

    def test_per_date_bootstrap_does_not_carry_orders_across_dates(self):
        # Even if _simulate_single_date returns orders on every date, the
        # captured list aggregates them as separate per-date entries — no
        # carry-forward of state from prior dates implied by the API.
        parity_dates = ["2026-04-13", "2026-04-14"]
        setup = self._setup_simulation_stub(parity_dates)

        def _bootstrap(trades_db_path, parity_date):
            return {
                "positions": {}, "cash": 100_000.0,
                "peak_nav": 1_000_000.0, "as_of": parity_date,
            }

        n_calls = 0

        def _simulate(**kwargs):
            nonlocal n_calls
            n_calls += 1
            # Every call sees a fresh sim_client (None _simulation_date
            # confirms the constructor was invoked anew per-date).
            assert kwargs["sim_client"]._simulation_date in (None,
                                                              kwargs["signal_date"])
            return [{"ticker": "X", "action": "EXIT",
                     "date": kwargs["signal_date"], "n_call": n_calls}], None

        with (
            patch.object(bt, "_setup_simulation", return_value=setup),
            patch.object(bt, "_load_initial_state_from_eod_pnl",
                         side_effect=_bootstrap),
            patch.object(bt, "_simulate_single_date", side_effect=_simulate),
            patch("nousergon_lib.arcticdb.get_universe_symbols",
                  return_value=set()),
        ):
            captured = bt.replay_for_dates(parity_dates, {
                "trades_db_path": "/tmp/fake.db",
                "signals_bucket": "test",
                "init_cash": 1_000_000.0,
                "executor_paths": ["/tmp/nonexistent"],
            }, per_date_bootstrap=True)

        # Each call's order is captured; no cumulative aggregation.
        assert len(captured) == 2
        assert {o["date"] for o in captured} == set(parity_dates)
        assert {o["n_call"] for o in captured} == {1, 2}
