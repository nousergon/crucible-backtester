"""Unit tests for loaders.price_loader — post-Phase-0 ArcticDB-only path.

Covers build_matrix() behavior when the underlying ArcticDB read is mocked:
  * attrs populated (price_gap_warnings, unfilled_gaps, staleness_warning,
    stale_circuit_break, no_data_dates)
  * ffill limit of 5 days — tickers with larger gaps dropped
  * Empty dates list → empty DataFrame with attrs intact
  * Signal-ticker resolution filters ArcticDB output to the requested universe
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# arcticdb is a heavy C-extension dep only available on the spot instance;
# stub it in sys.modules so the module-level `import arcticdb as adb` in
# store.arctic_reader resolves during local test runs. Real calls are
# patched per test via mock.patch.
sys.modules.setdefault("arcticdb", MagicMock())

# Force submodule attribute registration so @patch("store.arctic_reader...") +
# @patch("loaders.price_loader...") dotted-path resolution succeeds (store/ and
# loaders/ are namespace packages — submodules aren't auto-exposed as attrs).
import loaders.price_loader  # noqa: E402, F401
import store.arctic_reader  # noqa: E402, F401


def _make_arctic_mock(
    ticker_series_map: dict[str, pd.Series],
    field: str = "Close",
    include_source: bool = False,
) -> tuple:
    """Build a (price_data, features_by_ticker) tuple matching ArcticDB's shape.

    Each series becomes a DataFrame with Open/High/Low/Close/Volume columns,
    plus the row-provenance ``source`` column when ``include_source=True``
    (mirrors the 2026-05-09 daily_append rollout). All non-NaN rows are
    stamped ``"polygon"``; tests that want a NaN-source row should patch
    the returned dict in place.
    """
    price_data: dict[str, pd.DataFrame] = {}
    for ticker, series in ticker_series_map.items():
        cols = {
            "Open": series,
            "High": series,
            "Low": series,
            field: series,
            "Volume": pd.Series(1_000_000, index=series.index, dtype="int64"),
        }
        if include_source:
            cols["source"] = pd.Series(
                ["polygon"] * len(series), index=series.index, dtype=object,
            ).where(series.notna(), other=None)
        price_data[ticker] = pd.DataFrame(cols)
    return price_data, {}


class TestBuildMatrixArcticDB:

    @patch("loaders.price_loader._tickers_from_signals")
    @patch("loaders.price_loader.load_universe_from_arctic")
    def test_returns_df_with_required_attrs(self, mock_arctic, mock_signals):
        from loaders.price_loader import build_matrix

        dates = [f"2026-03-{d:02d}" for d in range(2, 12)]
        mock_signals.return_value = ["AAPL", "MSFT"]

        idx = pd.to_datetime(dates)
        mock_arctic.return_value = _make_arctic_mock({
            "AAPL": pd.Series(np.arange(100, 110, dtype=float), index=idx),
            "MSFT": pd.Series(np.arange(200, 210, dtype=float), index=idx),
        })

        df = build_matrix(dates, bucket="test")

        for key in ("price_gap_warnings", "unfilled_gaps", "staleness_warning",
                    "stale_circuit_break", "no_data_dates"):
            assert key in df.attrs, f"missing attr: {key}"

    @patch("loaders.price_loader._tickers_from_signals")
    @patch("loaders.price_loader.load_universe_from_arctic")
    def test_ffill_limit_drops_tickers_with_wide_gaps(self, mock_arctic, mock_signals):
        """Tickers with gaps > 5 days get dropped from the matrix."""
        from loaders.price_loader import build_matrix

        dates = [f"2026-03-{d:02d}" for d in range(2, 12)]
        mock_signals.return_value = ["AAPL", "MSFT"]

        full_idx = pd.to_datetime(dates)
        sparse_idx = pd.to_datetime(["2026-03-02"])

        price_data, _ = _make_arctic_mock({
            "AAPL": pd.Series([100.0], index=sparse_idx),     # 1 date of 10
            "MSFT": pd.Series(np.arange(200, 210, dtype=float), index=full_idx),
        })
        mock_arctic.return_value = (price_data, {})

        df = build_matrix(dates, bucket="test")
        assert "AAPL" not in df.columns, "AAPL has wide gap — must be dropped"
        assert "AAPL" in df.attrs["unfilled_gaps"]
        assert "MSFT" in df.columns

    @patch("loaders.price_loader._tickers_from_signals")
    @patch("loaders.price_loader.load_universe_from_arctic")
    def test_empty_dates_returns_empty_df(self, mock_arctic, mock_signals):
        from loaders.price_loader import build_matrix

        df = build_matrix([], bucket="test")
        assert df.empty
        assert df.attrs["no_data_dates"] == []
        # ArcticDB read should not have happened for zero signal tickers
        mock_arctic.assert_not_called()

    @patch("loaders.price_loader._tickers_from_signals", return_value=[])
    @patch("loaders.price_loader.load_universe_from_arctic")
    def test_no_signals_returns_empty_df(self, mock_arctic, mock_signals):
        """When signals.json resolves zero tickers, skip the ArcticDB read."""
        from loaders.price_loader import build_matrix

        df = build_matrix(["2026-03-10"], bucket="test")
        assert df.empty
        assert df.attrs["no_data_dates"] == ["2026-03-10"]
        mock_arctic.assert_not_called()


class TestGapWarningSourceAware:
    """price_gap_warnings — source-aware + pre-IPO-aware semantics.

    Pins the L1066 ROADMAP entry's expected behavior: the gap-warning
    metric only fires on cells where neither yfinance nor polygon
    persisted a row (i.e. ``source`` column is NaN) within the
    [first_listed_date, simulation_end] window per ticker, excluding
    calendar-mismatch macros (VIX/TNX/IRX).
    """

    @patch("loaders.price_loader._tickers_from_signals")
    @patch("loaders.price_loader.load_universe_from_arctic")
    def test_pre_ipo_dates_excluded_from_gap_count(self, mock_arctic, mock_signals):
        """A late-IPO ticker shouldn't warn for its pre-IPO date range.

        IPO-2026-03-07 ticker over a 10-day window: 7 cells of pre-IPO
        history (no source row, by definition) + 3 post-IPO cells
        (source="polygon"). The pre-IPO 7 must NOT contribute to the gap
        count — only post-IPO source-NaN cells count.
        """
        from loaders.price_loader import build_matrix

        dates = [f"2026-03-{d:02d}" for d in range(2, 12)]  # 2026-03-02 .. 2026-03-11
        mock_signals.return_value = ["IPOCO", "MSFT"]

        post_ipo_idx = pd.to_datetime([f"2026-03-{d:02d}" for d in (7, 8, 9, 10, 11)])
        full_idx = pd.to_datetime(dates)

        price_data, _ = _make_arctic_mock(
            {
                "IPOCO": pd.Series(np.arange(100, 105, dtype=float), index=post_ipo_idx),
                "MSFT": pd.Series(np.arange(200, 210, dtype=float), index=full_idx),
            },
            include_source=True,
        )
        mock_arctic.return_value = (price_data, {})

        df = build_matrix(dates, bucket="test")
        assert "IPOCO" not in df.attrs["price_gap_warnings"], (
            "Pre-IPO dates must not count as gaps — IPOCO's history starts "
            "2026-03-07, prior dates aren't ingestion failures."
        )
        assert "MSFT" not in df.attrs["price_gap_warnings"]

    @patch("loaders.price_loader._tickers_from_signals")
    @patch("loaders.price_loader.load_universe_from_arctic")
    def test_calendar_mismatch_macros_excluded(self, mock_arctic, mock_signals):
        """VIX/TNX/IRX trade on different calendars — exclude from gap warning.

        The simulation universe includes equities with full NYSE coverage
        and a macro with a missing-row pattern that would trip the
        threshold. The macro must be skipped.
        """
        from loaders.price_loader import build_matrix

        dates = [f"2026-03-{d:02d}" for d in range(2, 16)]  # 14 days
        mock_signals.return_value = ["MSFT", "VIX"]

        full_idx = pd.to_datetime(dates)
        sparse_idx = pd.to_datetime(["2026-03-02", "2026-03-15"])  # 2-of-14
        price_data, _ = _make_arctic_mock(
            {
                "MSFT": pd.Series(np.arange(200, 200 + len(full_idx), dtype=float), index=full_idx),
                "VIX": pd.Series([20.0, 22.0], index=sparse_idx),
            },
            include_source=True,
        )
        mock_arctic.return_value = (price_data, {})

        df = build_matrix(dates, bucket="test")
        assert "VIX" not in df.attrs["price_gap_warnings"], (
            "VIX trades on the CBOE volatility calendar; missing NYSE-equity "
            "dates are calendar mismatches, not data gaps."
        )

    @patch("loaders.price_loader._tickers_from_signals")
    @patch("loaders.price_loader.load_universe_from_arctic")
    def test_source_nan_within_window_does_warn(self, mock_arctic, mock_signals):
        """A long-history ticker with a mid-window source-NaN streak warns.

        Genuine ingestion gap class — the only population the metric is
        meant to surface. Mirrors the ~11 IPO/spinoff structural cases
        in the L1066 diagnostic.
        """
        from loaders.price_loader import build_matrix

        dates = [f"2026-03-{d:02d}" for d in range(2, 16)]  # 14 days
        mock_signals.return_value = ["GAPPY", "MSFT"]

        full_idx = pd.to_datetime(dates)
        gappy_series = pd.Series(np.arange(100, 100 + len(full_idx), dtype=float), index=full_idx)
        msft_series = pd.Series(np.arange(200, 200 + len(full_idx), dtype=float), index=full_idx)

        price_data, _ = _make_arctic_mock(
            {"GAPPY": gappy_series, "MSFT": msft_series},
            include_source=True,
        )
        # Drop source rows across 9 calendar days (~7 business days) in
        # the middle of the window — keep Close populated so this
        # exercises the source-aware path, not Close-NaN fallback.
        # Mirrors "polygon outage but yfinance backfill still produced a
        # Close at zero authority" hypothetical that the new metric
        # should still flag.
        gap_dates = pd.to_datetime([f"2026-03-{d:02d}" for d in range(5, 14)])
        price_data["GAPPY"].loc[gap_dates, "source"] = None
        mock_arctic.return_value = (price_data, {})

        df = build_matrix(dates, bucket="test")
        assert "GAPPY" in df.attrs["price_gap_warnings"], (
            "7-day source-NaN streak inside [first_listed, sim_end] window "
            "must trip the gap-warning threshold."
        )
        assert df.attrs["price_gap_warnings"]["GAPPY"] >= 6
        assert "MSFT" not in df.attrs["price_gap_warnings"]

    @patch("loaders.price_loader._tickers_from_signals")
    @patch("loaders.price_loader.load_universe_from_arctic")
    def test_legacy_series_without_source_falls_back_to_close(self, mock_arctic, mock_signals):
        """Pre-2026-05-09 ArcticDB rows lack the ``source`` column.

        For those tickers the metric falls back to a Close-NaN check on
        the same windowed range so legacy series still produce sane
        warnings during the migration tail.
        """
        from loaders.price_loader import build_matrix

        dates = [f"2026-03-{d:02d}" for d in range(2, 16)]  # 14 days
        mock_signals.return_value = ["LEGACY", "MSFT"]

        full_idx = pd.to_datetime(dates)
        sparse_idx = pd.to_datetime(["2026-03-02"])  # 1-of-14

        price_data, _ = _make_arctic_mock(
            {
                "LEGACY": pd.Series([100.0], index=sparse_idx),
                "MSFT": pd.Series(np.arange(200, 200 + len(full_idx), dtype=float), index=full_idx),
            },
            include_source=False,  # legacy: no source column
        )
        mock_arctic.return_value = (price_data, {})

        df = build_matrix(dates, bucket="test")
        # LEGACY's first_seen is 2026-03-02 (only date with data), so the
        # window covers all 14 days; 13 Close-NaN cells > threshold.
        assert "LEGACY" in df.attrs["price_gap_warnings"]
        assert "MSFT" not in df.attrs["price_gap_warnings"]


class TestArcticFreshnessGate:
    """_verify_arctic_fresh guards against stale/missing SPY in ArcticDB."""

    def _macro_lib(self, last_date=None, raise_exc=None):
        from unittest.mock import MagicMock
        lib = MagicMock()
        if raise_exc is not None:
            lib.read.side_effect = raise_exc
            return lib
        if last_date is None:
            lib.read.return_value.data = pd.DataFrame(columns=["Close"])
        else:
            idx = pd.DatetimeIndex([pd.Timestamp(last_date)])
            lib.read.return_value.data = pd.DataFrame({"Close": [500.0]}, index=idx)
        return lib

    @patch("nousergon_lib.arcticdb.open_macro_lib")
    @patch("store.arctic_reader._get_arctic")
    def test_missing_spy_raises(self, mock_get_arctic, mock_open_macro):
        import pytest

        from store.arctic_reader import _verify_arctic_fresh

        # config#804: macro opens via the shared open_macro_lib helper;
        # _get_arctic is still called (patched harmless) for the diagnostic log.
        mock_open_macro.return_value = self._macro_lib(raise_exc=Exception("SymbolNotFound"))

        with pytest.raises(RuntimeError, match="unreadable"):
            _verify_arctic_fresh(bucket="test", min_date="2026-04-16")

    @patch("nousergon_lib.arcticdb.open_macro_lib")
    @patch("store.arctic_reader._get_arctic")
    def test_fresh_spy_passes(self, mock_get_arctic, mock_open_macro):
        from store.arctic_reader import _verify_arctic_fresh

        mock_open_macro.return_value = self._macro_lib(last_date="2026-04-16")

        _verify_arctic_fresh(bucket="test", min_date="2026-04-16")  # should not raise

    @patch("nousergon_lib.arcticdb.open_macro_lib")
    @patch("store.arctic_reader._get_arctic")
    def test_stale_spy_raises(self, mock_get_arctic, mock_open_macro):
        import pytest

        from store.arctic_reader import _verify_arctic_fresh

        mock_open_macro.return_value = self._macro_lib(last_date="2026-04-15")

        with pytest.raises(RuntimeError, match="stale"):
            _verify_arctic_fresh(bucket="test", min_date="2026-04-16")
