"""Tests for store/feature_maps.py — the precomputed ATR + VWAP maps
that let the backtester skip per-call ArcticDB reads in the executor
(alpha-engine PR #91's ``atr_map`` / ``vwap_map`` kwargs).

The 2026-04-22 Saturday SF dry-run timed out at the 2h SSM ceiling
still mid-param-sweep because each ``_simulate_single_date`` call
triggered 20+ ``universe.read(ticker)`` round-trips — once for ATR,
once for VWAP, per ticker. py-spy stack pinned the hot path. This
module's bulk-read-once + in-memory-resolve-per-date shape is the
fix; tests lock byte-equivalence against the executor's per-call
semantics.

Direct unit tests (not source inspection) because ``load_precomputed_
feature_maps`` + ``resolve_vwap_map_for_date`` are small pure
functions; easy to exercise the walk-back edge cases (no valid VWAP
in window, single-point series, tz-aware input).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from store import feature_maps


# ── load_precomputed_feature_maps ────────────────────────────────────────────


def _mock_arctic_library(ticker_rows: dict[str, pd.DataFrame]) -> MagicMock:
    """Return a mock ArcticDB universe library whose .read(ticker) returns
    the given DataFrame, and .list_symbols returns the dict keys.
    """
    lib = MagicMock()
    lib.list_symbols.return_value = list(ticker_rows.keys())

    def _read(ticker):
        if ticker not in ticker_rows:
            raise KeyError(f"no such symbol: {ticker}")
        result = MagicMock()
        result.data = ticker_rows[ticker]
        return result

    lib.read.side_effect = _read
    return lib


def _patched_arctic(lib: MagicMock):
    """Context manager that patches ``arcticdb.Arctic`` to return a mock
    with the given library for get_library("universe").
    """
    arctic_instance = MagicMock()
    arctic_instance.get_library.return_value = lib
    arctic_cls = MagicMock(return_value=arctic_instance)
    return patch.object(feature_maps, "load_precomputed_feature_maps", wraps=feature_maps.load_precomputed_feature_maps), \
           patch.dict("sys.modules", {"arcticdb": MagicMock(Arctic=arctic_cls)})


def _mock_bulk_read(rows: dict[str, pd.DataFrame]):
    """Helper: sets up arcticdb mock so load_precomputed_feature_maps
    returns the given per-ticker DataFrames."""
    lib = _mock_arctic_library(rows)
    arctic_instance = MagicMock()
    arctic_instance.get_library.return_value = lib
    fake_adb = MagicMock()
    fake_adb.Arctic.return_value = arctic_instance
    return patch.dict("sys.modules", {"arcticdb": fake_adb})


class TestLoadPrecomputedFeatureMaps:
    def test_extracts_atr_last_row_per_ticker(self):
        dates = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
        rows = {
            "AAPL": pd.DataFrame(
                {"atr_14_pct": [0.01, 0.02, 0.0238], "VWAP": [150.0, 151.0, 152.0]},
                index=dates,
            ),
            "MSFT": pd.DataFrame(
                {"atr_14_pct": [0.005, 0.006, 0.007], "VWAP": [400.0, 401.0, 402.0]},
                index=dates,
            ),
        }
        with _mock_bulk_read(rows):
            atr, vwap, cov = feature_maps.load_precomputed_feature_maps("test-bucket")

        assert atr == {"AAPL": 0.0238, "MSFT": 0.007}, (
            "ATR must be the last-row value per ticker — mirrors executor's "
            "load_atr_14_pct semantics."
        )
        assert set(vwap.keys()) == {"AAPL", "MSFT"}
        assert list(vwap["AAPL"].values) == [150.0, 151.0, 152.0]
        # Coverage: 1 feature column (atr_14_pct, last-row non-NaN) / 1 = 1.0.
        # VWAP is excluded from the coverage denominator per _COVERAGE_OHLCV_COLS.
        assert cov == {"AAPL": 1.0, "MSFT": 1.0}

    def test_skips_tickers_missing_atr_column(self):
        rows = {
            "FULL": pd.DataFrame(
                {"atr_14_pct": [0.01], "VWAP": [100.0]},
                index=pd.DatetimeIndex(["2024-01-02"]),
            ),
            "NO_ATR": pd.DataFrame(
                {"VWAP": [50.0]}, index=pd.DatetimeIndex(["2024-01-02"]),
            ),
        }
        with _mock_bulk_read(rows):
            atr, vwap, _ = feature_maps.load_precomputed_feature_maps("test-bucket")

        # Ticker without atr_14_pct omitted from atr map (matches
        # load_atr_14_pct's dict semantics — .get returns None for missing).
        assert "FULL" in atr
        assert "NO_ATR" not in atr
        # But VWAP is still captured when column exists
        assert "NO_ATR" in vwap

    def test_skips_non_positive_atr_values(self):
        """Zero or NaN ATR values are omitted — matches load_atr_14_pct's
        ``if pd.notna(val) and val > 0:`` validation."""
        dates = pd.DatetimeIndex(["2024-01-02"])
        rows = {
            "ZERO": pd.DataFrame({"atr_14_pct": [0.0], "VWAP": [100.0]}, index=dates),
            "NAN": pd.DataFrame({"atr_14_pct": [float("nan")], "VWAP": [100.0]}, index=dates),
            "NEG": pd.DataFrame({"atr_14_pct": [-0.01], "VWAP": [100.0]}, index=dates),
            "GOOD": pd.DataFrame({"atr_14_pct": [0.02], "VWAP": [100.0]}, index=dates),
        }
        with _mock_bulk_read(rows):
            atr, _, _ = feature_maps.load_precomputed_feature_maps("test-bucket")

        assert set(atr.keys()) == {"GOOD"}, (
            f"Expected only GOOD; got {atr}"
        )

    def test_dropped_atr_tickers_named_in_warning(self, caplog):
        """No-silent-fails (L3147): dropped tickers + reasons must be named
        in a WARN, not just counted. A cohort ticker in this set is the
        opaque ``atr_map missing {ticker}`` failure deep in decide_entries."""
        import logging
        dates = pd.DatetimeIndex(["2024-01-02"])
        rows = {
            "NANATR": pd.DataFrame({"atr_14_pct": [float("nan")], "VWAP": [100.0]}, index=dates),
            "NOATRCOL": pd.DataFrame({"VWAP": [50.0]}, index=dates),
            "GOOD": pd.DataFrame({"atr_14_pct": [0.02], "VWAP": [100.0]}, index=dates),
        }
        with _mock_bulk_read(rows):
            with caplog.at_level(logging.WARNING, logger="store.feature_maps"):
                feature_maps.load_precomputed_feature_maps("test-bucket")

        warns = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
        assert "NANATR" in warns and "nan_last_row" in warns
        assert "NOATRCOL" in warns and "missing_atr_14_pct_column" in warns
        assert "GOOD" not in warns

    def test_library_open_failure_hard_fails(self):
        fake_adb = MagicMock()
        fake_adb.Arctic.side_effect = RuntimeError("arctic unreachable")

        with patch.dict("sys.modules", {"arcticdb": fake_adb}):
            with pytest.raises(RuntimeError, match="ArcticDB universe library open failed"):
                feature_maps.load_precomputed_feature_maps("test-bucket")

    def test_empty_universe_returns_empty_maps(self):
        with _mock_bulk_read({}):
            atr, vwap, cov = feature_maps.load_precomputed_feature_maps("test-bucket")

        assert atr == {}
        assert vwap == {}
        assert cov == {}


class TestFeatureCoverageBulkExtract:
    """Coverage = non_nan_feature_cols / total_feature_cols on the last
    row. Feature cols = every column outside _COVERAGE_OHLCV_COLS.
    Must match executor.price_cache.load_feature_coverage byte-for-byte."""

    def test_partial_coverage_multiple_features(self):
        """Short-history ticker: some long-window features NaN, others
        populated. Coverage = populated / total."""
        dates = pd.DatetimeIndex(["2024-01-02"])
        rows = {
            "SHORT": pd.DataFrame(
                {
                    "Open": [100.0], "High": [101.0], "Low": [99.0],
                    "Close": [100.5], "Volume": [1_000_000], "VWAP": [100.2],
                    "atr_14_pct": [0.02],  # populated
                    "return_252d": [float("nan")],  # unpopulated (short history)
                    "dist_from_52w_high": [float("nan")],
                    "rsi_14": [55.0],  # populated
                },
                index=dates,
            ),
        }
        with _mock_bulk_read(rows):
            _, _, cov = feature_maps.load_precomputed_feature_maps("test-bucket")

        # 4 feature cols (atr_14_pct, return_252d, dist_from_52w_high, rsi_14),
        # 2 non-NaN → coverage = 0.5
        assert cov == {"SHORT": 0.5}

    def test_zero_coverage_on_no_feature_cols(self):
        """A frame with ONLY OHLCV columns (no features) returns 0.0
        coverage — matches load_feature_coverage's no_features branch."""
        dates = pd.DatetimeIndex(["2024-01-02"])
        rows = {
            "RAW_ONLY": pd.DataFrame(
                {"Open": [100.0], "High": [101.0], "Low": [99.0],
                 "Close": [100.5], "Volume": [1_000_000], "VWAP": [100.2]},
                index=dates,
            ),
        }
        with _mock_bulk_read(rows):
            _, _, cov = feature_maps.load_precomputed_feature_maps("test-bucket")

        assert cov == {"RAW_ONLY": 0.0}

    def test_full_coverage_all_features_populated(self):
        dates = pd.DatetimeIndex(["2024-01-02"])
        rows = {
            "FULL": pd.DataFrame(
                {"Close": [100.0], "atr_14_pct": [0.02],
                 "rsi_14": [50.0], "return_252d": [0.15]},
                index=dates,
            ),
        }
        with _mock_bulk_read(rows):
            _, _, cov = feature_maps.load_precomputed_feature_maps("test-bucket")

        # 3 feature cols, all populated → 1.0
        assert cov == {"FULL": 1.0}

    def test_coverage_ohlcv_cols_match_executor_side(self):
        """The _COVERAGE_OHLCV_COLS frozenset in store/feature_maps.py
        must byte-equal executor/price_cache.py's definition. Drift
        would silently change backtest coverage values vs live
        executor values."""
        expected = frozenset({
            "Open", "High", "Low", "Close", "Adj_Close", "Volume", "VWAP",
        })
        assert feature_maps._COVERAGE_OHLCV_COLS == expected, (
            "_COVERAGE_OHLCV_COLS drifted from executor's definition — "
            "backtest and live will compute different coverage values."
        )

    def test_adj_close_excluded_from_feature_count(self):
        """Adj_Close is a raw-market-data column, should NOT count as
        a feature — same as Open/High/Low/Close/Volume/VWAP."""
        dates = pd.DatetimeIndex(["2024-01-02"])
        rows = {
            "WITH_ADJ": pd.DataFrame(
                {"Close": [100.0], "Adj_Close": [99.5],
                 "atr_14_pct": [0.02]},
                index=dates,
            ),
        }
        with _mock_bulk_read(rows):
            _, _, cov = feature_maps.load_precomputed_feature_maps("test-bucket")

        # Only atr_14_pct counts as a feature → 1/1 = 1.0
        # If Adj_Close leaked into the denominator it'd be 1/2 = 0.5
        assert cov == {"WITH_ADJ": 1.0}


# ── resolve_vwap_map_for_date ────────────────────────────────────────────────


class TestResolveVwapMapForDate:
    """Walk-back semantics must mirror executor.price_cache.load_daily_vwap.
    For each ticker, walk back up to ``max_lookback`` trading days from
    ``simulate_date`` and return the first positive, non-NaN value."""

    def test_exact_date_match(self):
        series = pd.Series(
            [100.0, 101.0, 102.0],
            index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"]),
        )
        result = feature_maps.resolve_vwap_map_for_date(
            {"AAPL": series}, ["AAPL"], "2024-01-04",
        )
        assert result == {"AAPL": 102.0}

    def test_walks_back_over_weekend(self):
        """simulate_date on a Saturday → last trading day's value."""
        series = pd.Series(
            [100.0, 101.0, 102.0],
            index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"]),
        )
        result = feature_maps.resolve_vwap_map_for_date(
            {"AAPL": series}, ["AAPL"], "2024-01-06",  # Saturday
        )
        assert result == {"AAPL": 102.0}

    def test_skips_nan_values(self):
        series = pd.Series(
            [100.0, float("nan"), float("nan"), 103.0],
            index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        )
        # Ask for 2024-01-04 with lookback 5 → walks back over NaN to 100.0
        result = feature_maps.resolve_vwap_map_for_date(
            {"AAPL": series}, ["AAPL"], "2024-01-04", max_lookback=5,
        )
        assert result == {"AAPL": 100.0}

    def test_skips_non_positive_values(self):
        """Zero or negative VWAP omitted — matches ``if v > 0`` in
        load_daily_vwap."""
        series = pd.Series(
            [0.0, -5.0, 100.0],
            index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"]),
        )
        # Ask for 2024-01-03 — only the 0.0 and -5.0 are available → no resolution
        result = feature_maps.resolve_vwap_map_for_date(
            {"AAPL": series}, ["AAPL"], "2024-01-03",
        )
        assert result == {}, f"Expected empty; got {result}"

    def test_lookback_exhausted_no_valid_vwap(self):
        """All-NaN series → ticker absent from result (mirrors load_daily_vwap
        behavior of omitting ticker when no_valid_vwap_in_window)."""
        series = pd.Series(
            [float("nan"), float("nan"), float("nan")],
            index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"]),
        )
        result = feature_maps.resolve_vwap_map_for_date(
            {"AAPL": series}, ["AAPL"], "2024-01-04",
        )
        assert result == {}

    def test_ticker_not_in_precomputed_map_is_omitted(self):
        result = feature_maps.resolve_vwap_map_for_date(
            {"AAPL": pd.Series([100.0], index=pd.DatetimeIndex(["2024-01-02"]))},
            ["AAPL", "UNKNOWN"],
            "2024-01-02",
        )
        assert result == {"AAPL": 100.0}

    def test_date_before_all_history(self):
        series = pd.Series(
            [100.0, 101.0],
            index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"]),
        )
        result = feature_maps.resolve_vwap_map_for_date(
            {"AAPL": series}, ["AAPL"], "2023-12-31",
        )
        assert result == {}

    def test_empty_tickers_returns_empty(self):
        result = feature_maps.resolve_vwap_map_for_date({}, [], "2024-01-02")
        assert result == {}

    def test_multi_ticker_resolution(self):
        dates = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
        vwap_map = {
            "AAPL": pd.Series([150.0, 151.0, 152.0], index=dates),
            "MSFT": pd.Series([400.0, 401.0, 402.0], index=dates),
            "NVDA": pd.Series([float("nan"), 500.0, float("nan")], index=dates),
        }
        result = feature_maps.resolve_vwap_map_for_date(
            vwap_map, ["AAPL", "MSFT", "NVDA"], "2024-01-04",
        )
        assert result == {
            "AAPL": 152.0,
            "MSFT": 402.0,
            "NVDA": 500.0,  # Walks back from 2024-01-04 (NaN) to 2024-01-03 (500.0)
        }

    def test_tz_aware_series_normalized_for_lookup(self):
        """VWAP series loaded from ArcticDB may have tz-aware index —
        ``resolve_vwap_map_for_date`` must handle that without exploding."""
        # Pre-normalize in the loader so this test case exercises the
        # tz-naive path (which is what production sees post-loader).
        series = pd.Series(
            [100.0, 101.0],
            index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"]),
        )
        result = feature_maps.resolve_vwap_map_for_date(
            {"AAPL": series}, ["AAPL"], "2024-01-03",
        )
        assert result == {"AAPL": 101.0}
