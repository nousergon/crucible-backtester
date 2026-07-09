"""Point-in-time (survivorship-free) universe threading (#1942 Leg 2).

Pins the per-date PIT set flowing through the primary threading points:

  1. ``_build_pit_universe_resolver`` is OFF by default (non-breaking):
     returns ``None`` unless ``config['survivorship_free_universe']`` is on.
  2. When on, it resolves per-date membership via
     ``nousergon_lib.arcticdb.get_universe_symbols(bucket, as_of=<date>)``,
     memoizes per date, and hard-fails on a PIT-map read error.
  3. ``_precompute_signal_lookups(universe_resolver=...)`` applies the
     resolver's per-date set (not one date-agnostic set) to each date —
     so a name dropped from the index on an earlier date is filtered out
     of that date's signals but kept on a later date where it was a member.
  4. The synthetic path's ``build_signals_by_date(pit_universe_resolver=...)``
     restricts each date's candidates to as-of membership before scoring.
"""
from __future__ import annotations

import datetime

import pytest

import backtest
from backtest import (
    _build_pit_universe_resolver,
    _precompute_signal_lookups,
)


def _signals(date_str: str, tickers: list[str]) -> dict:
    return {
        "date": date_str,
        "market_regime": "neutral",
        "sector_ratings": {},
        "enter": [],
        "exit": [],
        "reduce": [],
        "hold": [],
        "universe": [{"ticker": t, "sector": "Technology"} for t in tickers],
        "buy_candidates": [
            {"ticker": t, "sector": "Technology", "score": 80} for t in tickers
        ],
    }


# ── resolver: off by default ────────────────────────────────────────────────


def test_resolver_none_by_default():
    """Non-breaking default: no config flag -> no resolver -> legacy
    date-agnostic behavior is preserved exactly."""
    assert _build_pit_universe_resolver("b", {}, {"AAA"}) is None
    assert (
        _build_pit_universe_resolver(
            "b", {"survivorship_free_universe": False}, {"AAA"}
        )
        is None
    )


def test_resolver_built_when_flag_on(monkeypatch):
    calls: list = []

    def _fake_get_universe_symbols(bucket, *, as_of=None, **kw):
        calls.append((bucket, as_of))
        return {"AAA", "BBB"}

    monkeypatch.setattr(
        "nousergon_lib.arcticdb.get_universe_symbols", _fake_get_universe_symbols
    )
    resolver = _build_pit_universe_resolver(
        "bkt", {"survivorship_free_universe": True}, {"AAA"}
    )
    assert resolver is not None
    got = resolver("2021-06-15")
    assert got == {"AAA", "BBB"}
    assert calls == [("bkt", datetime.date(2021, 6, 15))]


def test_resolver_memoizes_per_date(monkeypatch):
    n = {"calls": 0}

    def _fake(bucket, *, as_of=None, **kw):
        n["calls"] += 1
        return {"AAA"}

    monkeypatch.setattr("nousergon_lib.arcticdb.get_universe_symbols", _fake)
    resolver = _build_pit_universe_resolver(
        "b", {"survivorship_free_universe": True}, {"AAA"}
    )
    resolver("2021-01-01")
    resolver("2021-01-01")  # cached — no second lib read
    assert n["calls"] == 1


def test_resolver_hard_fails_on_pit_map_error(monkeypatch):
    def _boom(bucket, *, as_of=None, **kw):
        raise RuntimeError("PIT constituent map missing")

    monkeypatch.setattr("nousergon_lib.arcticdb.get_universe_symbols", _boom)
    resolver = _build_pit_universe_resolver(
        "b", {"survivorship_free_universe": True}, {"AAA"}
    )
    with pytest.raises(RuntimeError, match="survivorship_free_universe is enabled"):
        resolver("2021-01-01")


# ── primary threading: per-date PIT set flows to _precompute_signal_lookups ──


def test_precompute_applies_per_date_resolver():
    """Each date gets its OWN universe: a name that was an index member on
    date A but not on date B is kept on A and dropped on B — proving the PIT
    set (not one static set) reaches _filter_signals_to_universe per date."""
    signals_by_date = {
        "2021-01-01": _signals("2021-01-01", ["AAA", "OLDCO"]),
        "2023-01-01": _signals("2023-01-01", ["AAA", "OLDCO"]),
    }

    # OLDCO was in the index in 2021 but removed by 2023.
    per_date = {
        "2021-01-01": {"AAA", "OLDCO"},
        "2023-01-01": {"AAA"},
    }

    def resolver(date_str):
        return per_date[date_str]

    lookups = _precompute_signal_lookups(
        signals_by_date, universe_symbols={"AAA"}, universe_resolver=resolver
    )

    # 2021: OLDCO kept (was a member).
    assert "OLDCO" in lookups["2021-01-01"].signals_by_ticker
    # 2023: OLDCO dropped (not a member on that date).
    assert "OLDCO" not in lookups["2023-01-01"].signals_by_ticker
    assert "AAA" in lookups["2023-01-01"].signals_by_ticker


def test_precompute_falls_back_to_static_when_resolver_returns_none():
    """A ``None`` from the resolver (date on/after latest index change) means
    'current roster' -> fall back to the date-agnostic ``universe_symbols``."""
    signals_by_date = {"2026-01-01": _signals("2026-01-01", ["AAA", "BBB"])}

    def resolver(date_str):
        return None  # e.g. recent date, no pre-change snapshot

    lookups = _precompute_signal_lookups(
        signals_by_date, universe_symbols={"AAA"}, universe_resolver=resolver
    )
    # BBB not in static universe -> dropped via fallback.
    assert "AAA" in lookups["2026-01-01"].signals_by_ticker
    assert "BBB" not in lookups["2026-01-01"].signals_by_ticker


def test_precompute_no_resolver_is_legacy_behavior():
    """Without a resolver, the single ``universe_symbols`` applies to every
    date — identical to pre-#1942."""
    signals_by_date = {"2021-01-01": _signals("2021-01-01", ["AAA", "BBB"])}
    lookups = _precompute_signal_lookups(
        signals_by_date, universe_symbols={"AAA"}
    )
    assert "AAA" in lookups["2021-01-01"].signals_by_ticker
    assert "BBB" not in lookups["2021-01-01"].signals_by_ticker


# ── synthetic path: build_signals_by_date per-date PIT filter ────────────────


def test_synthetic_build_signals_by_date_pit_filter(monkeypatch):
    from synthetic import predictor_backtest as pb

    # Predictions offer OLDCO on both dates; PIT membership excludes OLDCO on
    # the later date. build_signals_by_date must strip OLDCO before scoring so
    # it can never become an ENTER candidate on the later date.
    predictions_by_date = {
        "2021-01-01": {"AAA": 0.9, "OLDCO": 0.8},
        "2023-01-01": {"AAA": 0.9, "OLDCO": 0.8},
    }
    seen: dict[str, set] = {}

    def _fake_predictions_to_signals(*, predictions, date, **kw):
        seen[date] = set(predictions.keys())
        return {"date": date, "buy_candidates": [], "universe": []}

    monkeypatch.setattr(
        pb, "predictions_to_signals", _fake_predictions_to_signals
    )
    # Neutralize the indicator precompute (irrelevant to the universe filter).
    monkeypatch.setattr(
        "synthetic.signal_generator.precompute_indicator_series",
        lambda ohlcv: {},
    )
    monkeypatch.setattr(
        "synthetic.signal_generator.indicators_from_precomputed",
        lambda pre, tickers, date: {},
    )

    def resolver(date_str):
        return {"AAA", "OLDCO"} if date_str == "2021-01-01" else {"AAA"}

    pb.build_signals_by_date(
        predictions_by_date,
        sector_map={},
        ohlcv_by_ticker={},
        pit_universe_resolver=resolver,
    )

    assert seen["2021-01-01"] == {"AAA", "OLDCO"}  # OLDCO kept (member)
    assert seen["2023-01-01"] == {"AAA"}  # OLDCO dropped (not a member)


def test_synthetic_resolver_off_by_default():
    from synthetic.predictor_backtest import build_pit_universe_resolver

    assert build_pit_universe_resolver("b", {}) is None
