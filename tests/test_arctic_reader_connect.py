"""Regression: store.arctic_reader._get_arctic first-call diagnostic log.

The L2771 ArcticDB chokepoint migration replaced the local
`uri = ...; adb.Arctic(uri)` with `open_arctic(bucket)` but left the
first-call INFO log referencing the removed `uri`/`region` locals, raising
`NameError: name 'uri' is not defined` on every fresh-process connect. The
main backtest reads prices from S3/yfinance so it never hit this; pit_parity's
replay subprocess connects to ArcticDB and failed with backtester_replay_error
(NameError) every cycle. This pins the first-call log path so it can't recur.
"""

from __future__ import annotations

import logging

import pandas as pd
from unittest.mock import MagicMock


def test_get_arctic_first_call_does_not_nameerror(monkeypatch, caplog):
    import store.arctic_reader as ar

    class _FakeArctic:
        def list_libraries(self):
            return ["universe", "macro"]

    # open_arctic is imported inside _get_arctic at call time → patch the lib attr.
    monkeypatch.setattr(
        "alpha_engine_lib.arcticdb.open_arctic", lambda bucket: _FakeArctic()
    )
    # Force the first-call (INFO-logging) branch.
    monkeypatch.setattr(ar, "_ARCTIC_LOGGED", False)

    with caplog.at_level(logging.INFO):
        result = ar._get_arctic("alpha-engine-research")  # must NOT raise NameError

    assert isinstance(result, _FakeArctic)
    # Diagnostic log fired with the library list — the load-bearing divergence
    # signal (the 2026-04-24 incident was 910 vs 0 universe symbols).
    assert "ArcticDB connected" in caplog.text
    assert "universe" in caplog.text
    # Guard against re-introducing the removed locals in the log template.
    assert "uri=" not in caplog.text


def test_get_arctic_list_libraries_failure_is_non_fatal(monkeypatch, caplog):
    """A list_libraries() failure must degrade to a logged marker, not crash
    the connect (the except branch builds the libs string)."""
    import store.arctic_reader as ar

    class _FakeArctic:
        def list_libraries(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "alpha_engine_lib.arcticdb.open_arctic", lambda bucket: _FakeArctic()
    )
    monkeypatch.setattr(ar, "_ARCTIC_LOGGED", False)

    with caplog.at_level(logging.INFO):
        result = ar._get_arctic("alpha-engine-research")
    assert isinstance(result, _FakeArctic)
    assert "list_libraries failed" in caplog.text


def test_load_universe_opens_universe_via_helper_macro_via_local_arctic(monkeypatch):
    """config#804: ``load_universe_from_arctic`` must open the ``universe``
    library via the shared ``open_universe_lib`` helper. The ``macro``
    library open and the #826 empty-universe diagnostic still go through the
    local ``_get_arctic`` handle — assert that split so a future refactor
    can't silently reroute (or break) the diagnostic path.
    """
    import store.arctic_reader as ar

    dates = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    universe_df = pd.DataFrame(
        {"Open": [1.0, 2.0], "High": [1.0, 2.0], "Low": [1.0, 2.0],
         "Close": [1.0, 2.0], "Volume": [10, 20], "rsi_14": [50.0, 55.0]},
        index=dates,
    )

    universe_lib = MagicMock()
    universe_lib.list_symbols.return_value = ["AAPL"]
    universe_read = MagicMock()
    universe_read.data = universe_df
    universe_lib.read.return_value = universe_read

    macro_lib = MagicMock()
    macro_lib.read.side_effect = KeyError("no macro symbol")

    # Local arctic handle still owns ``macro`` (+ list_libraries for the
    # #826 diagnostic) — proving that block is untouched by the migration.
    fake_arctic = MagicMock()
    fake_arctic.get_library.return_value = macro_lib

    monkeypatch.setattr(ar, "_get_arctic", lambda bucket: fake_arctic)

    helper = MagicMock(return_value=universe_lib)
    monkeypatch.setattr("alpha_engine_lib.arcticdb.open_universe_lib", helper)

    price_data, features = ar.load_universe_from_arctic("alpha-engine-research")

    # Universe opened via the shared helper, with the bucket.
    helper.assert_called_once_with("alpha-engine-research")
    # Macro still opened off the local arctic handle (diagnostic block intact).
    fake_arctic.get_library.assert_called_once_with("macro")
    assert "AAPL" in price_data
    assert "AAPL" in features
