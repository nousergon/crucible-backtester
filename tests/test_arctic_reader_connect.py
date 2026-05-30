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
