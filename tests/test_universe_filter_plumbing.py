"""
tests/test_universe_filter_plumbing.py — smoke fixture's tickers_allowlist
threads through loaders and ArcticDB reader.

Covers the contract:
- `price_loader.build_matrix(tickers_allowlist=...)` intersects signal-
  resolved tickers with the allowlist, and passes it to
  `load_universe_from_arctic`.
- `store.feature_maps.load_precomputed_feature_maps(tickers_allowlist=...)`
  intersects ArcticDB `list_symbols()` with the allowlist before the
  bulk read loop.
- `load_universe_from_arctic(tickers_allowlist=...)` intersects
  `universe.list_symbols()` with the allowlist before the read loop.
- None passed through → existing production behavior (no filter).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# arcticdb is a heavy optional dep not installed in the local venv.
# store/arctic_reader.py does `import arcticdb as adb` at module top,
# so we stub sys.modules BEFORE importing the module under test. The
# production path is unaffected; this only matters for offline tests.
if "arcticdb" not in sys.modules:
    sys.modules["arcticdb"] = MagicMock()


# ── load_universe_from_arctic filter ────────────────────────────────────────


def _make_fake_arctic(symbols: list[str]):
    """Build a MagicMock hierarchy matching the ArcticDB API surface.

    Since config#804, ``load_universe_from_arctic`` opens the ``universe``
    library via the shared ``alpha_engine_lib.arcticdb.open_universe_lib``
    helper (see test patching below), while the ``macro`` open and the #826
    empty-universe diagnostic still go through the local ``_get_arctic``
    handle. So the returned ``arctic`` mock owns ``macro`` (+ ``list_libraries``
    for the diagnostic) and the returned ``universe`` mock must be wired in
    via an ``open_universe_lib`` patch.
    """
    def _fake_read(symbol):
        df = pd.DataFrame(
            {"Close": [100.0, 101.0], "Open": [99.0, 100.0],
             "High": [102.0, 102.5], "Low": [98.0, 99.5]},
            index=pd.to_datetime(["2026-04-20", "2026-04-21"]),
        )
        return MagicMock(data=df)

    universe = MagicMock()
    universe.list_symbols.return_value = list(symbols)
    universe.read.side_effect = _fake_read

    macro = MagicMock()
    macro.list_symbols.return_value = []
    macro.read.side_effect = _fake_read

    arctic = MagicMock()
    # ``universe`` is now opened via open_universe_lib, not get_library; the
    # local handle only serves ``macro`` (the get_library branch is kept
    # tolerant in case a future caller reintroduces a local universe open).
    arctic.get_library.side_effect = lambda name: (
        universe if name == "universe" else macro
    )
    return arctic, universe


def test_load_universe_respects_tickers_allowlist():
    """allowlist=[AAPL,MSFT] against a 4-ticker ArcticDB catalog reads
    only 2 tickers (plus macro, which isn't filtered)."""
    from store import arctic_reader

    arctic, universe = _make_fake_arctic(["AAPL", "MSFT", "NVDA", "TSLA"])
    with patch.object(arctic_reader, "_get_arctic", return_value=arctic), \
         patch("nousergon_lib.arcticdb.open_universe_lib", return_value=universe):
        price_data, features = arctic_reader.load_universe_from_arctic(
            bucket="test-bucket",
            tickers_allowlist={"AAPL", "MSFT"},
        )

    # Only the allowlist-intersected symbols got read from universe
    read_calls = [c.args[0] for c in universe.read.call_args_list]
    assert set(read_calls) == {"AAPL", "MSFT"}


def test_load_universe_none_allowlist_reads_all():
    """Production default: no filter → reads every symbol."""
    from store import arctic_reader

    arctic, universe = _make_fake_arctic(["AAPL", "MSFT", "NVDA"])
    with patch.object(arctic_reader, "_get_arctic", return_value=arctic), \
         patch("nousergon_lib.arcticdb.open_universe_lib", return_value=universe):
        arctic_reader.load_universe_from_arctic(bucket="test-bucket")

    read_calls = [c.args[0] for c in universe.read.call_args_list]
    assert set(read_calls) == {"AAPL", "MSFT", "NVDA"}


def test_load_universe_allowlist_misses_fall_through():
    """A typo or missing ticker in the allowlist shouldn't crash — we
    just read fewer tickers. Matches the 'fail soft, log loud' contract."""
    from store import arctic_reader

    arctic, universe = _make_fake_arctic(["AAPL", "MSFT"])
    with patch.object(arctic_reader, "_get_arctic", return_value=arctic), \
         patch("nousergon_lib.arcticdb.open_universe_lib", return_value=universe):
        price_data, _ = arctic_reader.load_universe_from_arctic(
            bucket="test-bucket",
            tickers_allowlist={"AAPL", "NONEXISTENT"},  # NONEXISTENT not in catalog
        )

    # Only AAPL got read; NONEXISTENT silently dropped
    read_calls = [c.args[0] for c in universe.read.call_args_list]
    assert set(read_calls) == {"AAPL"}


# ── price_loader.build_matrix filter ────────────────────────────────────────


def test_build_matrix_intersects_signals_with_allowlist():
    """build_matrix filters its resolved ticker set via allowlist before
    calling load_universe_from_arctic."""
    from loaders import price_loader

    # Stub the signals resolver to return a known ticker set per date
    with patch.object(
        price_loader, "_tickers_from_signals",
        return_value=["AAPL", "MSFT", "NVDA", "TSLA"],
    ):
        # Stub load_universe_from_arctic to observe what it's called with
        fake_load = MagicMock(return_value=({
            "AAPL": pd.DataFrame({"Close": [100.0]}, index=pd.to_datetime(["2026-04-20"])),
            "MSFT": pd.DataFrame({"Close": [200.0]}, index=pd.to_datetime(["2026-04-20"])),
        }, {}))
        with patch.object(price_loader, "load_universe_from_arctic", fake_load):
            price_loader.build_matrix(
                dates=["2026-04-20"],
                bucket="test-bucket",
                tickers_allowlist={"AAPL", "MSFT"},
            )

    # Verify the allowlist got threaded through to the reader
    fake_load.assert_called_once()
    call_kwargs = fake_load.call_args.kwargs
    assert call_kwargs.get("tickers_allowlist") == {"AAPL", "MSFT"}


def test_build_matrix_without_allowlist_passes_none():
    """Production default: no filter propagates through."""
    from loaders import price_loader

    with patch.object(
        price_loader, "_tickers_from_signals",
        return_value=["AAPL"],
    ):
        fake_load = MagicMock(return_value=({
            "AAPL": pd.DataFrame({"Close": [100.0]}, index=pd.to_datetime(["2026-04-20"])),
        }, {}))
        with patch.object(price_loader, "load_universe_from_arctic", fake_load):
            price_loader.build_matrix(dates=["2026-04-20"], bucket="test-bucket")

    call_kwargs = fake_load.call_args.kwargs
    assert call_kwargs.get("tickers_allowlist") is None


# ── load_precomputed_feature_maps filter ────────────────────────────────────


def test_feature_maps_respects_tickers_allowlist():
    """feature_maps.load_precomputed_feature_maps intersects ArcticDB
    list_symbols with the allowlist before the bulk-read loop."""
    from store import feature_maps

    # Build a stub ArcticDB library with 4 symbols
    fake_lib = MagicMock()
    fake_lib.list_symbols.return_value = ["AAPL", "MSFT", "NVDA", "TSLA"]
    # read() returns an empty frame so we short-circuit the feature-math
    # branches. Only the read-call count matters for this test.
    fake_lib.read.return_value = MagicMock(data=pd.DataFrame())

    fake_arctic = MagicMock()
    fake_arctic.get_library.return_value = fake_lib

    # Stub the adb.Arctic constructor to return our fake
    fake_adb = MagicMock()
    fake_adb.Arctic.return_value = fake_arctic

    with patch.dict("sys.modules", {"arcticdb": fake_adb}):
        feature_maps.load_precomputed_feature_maps(
            bucket="test-bucket",
            tickers_allowlist={"AAPL", "MSFT"},
        )

    # Verify only 2 tickers got read, not all 4
    read_symbols = [c.args[0] for c in fake_lib.read.call_args_list]
    assert set(read_symbols) == {"AAPL", "MSFT"}
