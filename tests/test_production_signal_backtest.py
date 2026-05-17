"""Tests for synthetic.production_signal_backtest (ROADMAP L124 PR 2).

Pins the production-signal input producer contract:
- rebalance dates = signals/ ∩ predictor/predictions/
- cohort per date = predictions ∪ signals["universe"] tickers, minus SPY/CASH
- α̂ = predicted_alpha or canonical_predicted_alpha or 0.0 (finite float)
- output shape matches synthetic.predictor_backtest.run()
  ({status, predictions_by_date, price_matrix, spy_prices, sector_map})
- degenerate states (no overlap, no cohort, SPY-missing) → non-ok status
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import synthetic.production_signal_backtest as psb


# ── pure helpers ─────────────────────────────────────────────────────────


class TestExtractUniverseTickers:
    def test_dict_shape(self):
        u = [{"ticker": "AAPL", "signal": "ENTER"}, {"ticker": "MSFT"}]
        assert psb._extract_universe_tickers(u) == ["AAPL", "MSFT"]

    def test_flat_string_shape(self):
        assert psb._extract_universe_tickers(["AAPL", "MSFT"]) == ["AAPL", "MSFT"]

    def test_mixed_and_junk_skipped(self):
        u = ["AAPL", {"ticker": "MSFT"}, {"no_ticker": 1}, 42, {"ticker": ""}]
        assert psb._extract_universe_tickers(u) == ["AAPL", "MSFT"]

    def test_non_list_returns_empty(self):
        assert psb._extract_universe_tickers(None) == []
        assert psb._extract_universe_tickers("AAPL") == []


class TestAlphaOf:
    def test_predicted_alpha_preferred(self):
        assert psb._alpha_of(
            {"predicted_alpha": 0.07, "canonical_predicted_alpha": 0.99}
        ) == pytest.approx(0.07)

    def test_canonical_fallback(self):
        assert psb._alpha_of(
            {"predicted_alpha": 0.0, "canonical_predicted_alpha": 0.05}
        ) == pytest.approx(0.05)

    def test_missing_is_zero(self):
        assert psb._alpha_of({}) == 0.0

    def test_garbage_is_zero(self):
        assert psb._alpha_of({"predicted_alpha": "not-a-number"}) == 0.0

    def test_non_finite_is_zero(self):
        assert psb._alpha_of({"predicted_alpha": float("inf")}) == 0.0


class TestIsIsoDate:
    @pytest.mark.parametrize("tok", ["2026-05-15", "2026-01-01"])
    def test_valid(self, tok):
        assert psb._is_iso_date(tok)

    @pytest.mark.parametrize(
        "tok", ["latest", "2026-5-1", "2026/05/15", "2026-13-99", ""]
    )
    def test_invalid(self, tok):
        assert not psb._is_iso_date(tok)


# ── build_production_signal_inputs ───────────────────────────────────────


def _fake_s3(signal_dates, pred_dates, signals_by_date, preds_by_date):
    """A MagicMock S3 client with a paginator that yields signals/ folders +
    predictor/predictions/ keys, and get_object serving the JSON bodies."""
    s3 = MagicMock()

    def paginate(Bucket, Prefix, Delimiter=None):
        if Prefix == "signals/":
            return [
                {"CommonPrefixes": [{"Prefix": f"signals/{d}/"} for d in signal_dates]}
            ]
        if Prefix == "predictor/predictions/":
            return [
                {"Contents": [
                    {"Key": f"predictor/predictions/{d}.json"} for d in pred_dates
                ]}
            ]
        return [{}]

    paginator = MagicMock()
    paginator.paginate.side_effect = paginate
    s3.get_paginator.return_value = paginator

    def get_object(Bucket, Key):
        if Key.startswith("signals/"):
            d = Key.split("/")[1]
            body = signals_by_date[d]
        else:  # predictor/predictions/{d}.json
            d = Key.split("/")[-1].removesuffix(".json")
            body = preds_by_date[d]
        return {"Body": MagicMock(read=lambda b=json.dumps(body).encode(): b)}

    s3.get_object.side_effect = get_object
    return s3


@pytest.fixture
def patched_price_layer():
    """Stub the ArcticDB load + price/sector helpers so the producer test is
    pure (no ArcticDB / no predictor repo on disk)."""
    idx = pd.to_datetime(["2026-05-13", "2026-05-14", "2026-05-15"])
    price_data = {
        "AAPL": pd.DataFrame({"Close": [1.0, 1.1, 1.2]}, index=idx),
        "MSFT": pd.DataFrame({"Close": [2.0, 2.1, 2.2]}, index=idx),
        "SPY": pd.DataFrame({"Close": [4.0, 4.1, 4.2]}, index=idx),
    }
    pm = pd.DataFrame(
        {"AAPL": [1.0, 1.1, 1.2], "MSFT": [2.0, 2.1, 2.2]}, index=idx
    )
    spy = pd.Series([4.0, 4.1, 4.2], index=idx)
    with patch.object(psb, "_resolve_predictor_path", return_value="/fake/pred"), \
         patch.object(
             psb, "build_price_matrix", return_value=pm
         ) as _bpm, \
         patch.object(psb, "_extract_close", return_value=spy), \
         patch.object(
             psb, "load_sector_map", return_value={"AAPL": "XLK", "MSFT": "XLK"}
         ), \
         patch(
             "store.arctic_reader.load_universe_from_arctic",
             return_value=(price_data, {}),
         ):
        yield


def test_happy_path_shape_and_cohort(patched_price_layer):
    signals = {
        "2026-05-14": {"universe": [{"ticker": "AAPL"}, {"ticker": "MSFT"}]},
        "2026-05-15": {"universe": [{"ticker": "AAPL"}]},
    }
    preds = {
        "2026-05-14": {"predictions": [
            {"ticker": "AAPL", "predicted_alpha": 0.07},
            {"ticker": "MSFT", "predicted_alpha": -0.02},
        ]},
        "2026-05-15": {"predictions": [
            {"ticker": "AAPL", "canonical_predicted_alpha": 0.03},
            {"ticker": "NVDA", "predicted_alpha": 0.10},  # in preds, not universe
        ]},
    }
    s3 = _fake_s3(
        ["2026-05-14", "2026-05-15"], ["2026-05-14", "2026-05-15"],
        signals, preds,
    )
    out = psb.build_production_signal_inputs({"signals_bucket": "b"}, s3_client=s3)

    assert out["status"] == "ok"
    assert out["signal_source"] == "production"
    assert set(out.keys()) >= {
        "predictions_by_date", "price_matrix", "spy_prices", "sector_map",
    }
    # cohort = predictions ∪ universe; NVDA (preds-only) included, SPY/CASH excl.
    assert out["predictions_by_date"]["2026-05-14"] == {
        "AAPL": pytest.approx(0.07), "MSFT": pytest.approx(-0.02),
    }
    assert out["predictions_by_date"]["2026-05-15"] == {
        "AAPL": pytest.approx(0.03), "NVDA": pytest.approx(0.10),
    }
    assert out["production_window"] == ["2026-05-14", "2026-05-15"]


def test_only_intersection_of_archives_used(patched_price_layer):
    # signals has an extra date with no matching predictions → dropped
    signals = {
        "2026-05-13": {"universe": [{"ticker": "AAPL"}]},
        "2026-05-15": {"universe": [{"ticker": "AAPL"}]},
    }
    preds = {"2026-05-15": {"predictions": [{"ticker": "AAPL", "predicted_alpha": 0.05}]}}
    s3 = _fake_s3(
        ["2026-05-13", "2026-05-15"], ["2026-05-15"], signals, preds,
    )
    out = psb.build_production_signal_inputs({"signals_bucket": "b"}, s3_client=s3)
    assert out["status"] == "ok"
    assert list(out["predictions_by_date"]) == ["2026-05-15"]


def test_no_overlap_returns_no_production_data(patched_price_layer):
    s3 = _fake_s3(
        ["2026-05-13"], ["2026-05-15"],
        {"2026-05-13": {"universe": []}},
        {"2026-05-15": {"predictions": []}},
    )
    out = psb.build_production_signal_inputs({"signals_bucket": "b"}, s3_client=s3)
    assert out["status"] == "no_production_data"


def test_max_dates_keeps_most_recent(patched_price_layer):
    dates = ["2026-05-13", "2026-05-14", "2026-05-15"]
    signals = {d: {"universe": [{"ticker": "AAPL"}]} for d in dates}
    preds = {
        d: {"predictions": [{"ticker": "AAPL", "predicted_alpha": 0.05}]}
        for d in dates
    }
    s3 = _fake_s3(dates, dates, signals, preds)
    out = psb.build_production_signal_inputs(
        {"signals_bucket": "b"}, s3_client=s3, max_dates=2,
    )
    assert list(out["predictions_by_date"]) == ["2026-05-14", "2026-05-15"]


def test_spy_missing_returns_error():
    signals = {"2026-05-15": {"universe": [{"ticker": "AAPL"}]}}
    preds = {"2026-05-15": {"predictions": [{"ticker": "AAPL", "predicted_alpha": 0.05}]}}
    s3 = _fake_s3(["2026-05-15"], ["2026-05-15"], signals, preds)
    idx = pd.to_datetime(["2026-05-15"])
    with patch.object(psb, "_resolve_predictor_path", return_value="/fake"), \
         patch.object(psb, "build_price_matrix",
                      return_value=pd.DataFrame({"AAPL": [1.0]}, index=idx)), \
         patch.object(psb, "_extract_close", return_value=None), \
         patch.object(psb, "load_sector_map", return_value={}), \
         patch("store.arctic_reader.load_universe_from_arctic",
               return_value=({"AAPL": pd.DataFrame({"Close": [1.0]}, index=idx)}, {})):
        out = psb.build_production_signal_inputs({"signals_bucket": "b"}, s3_client=s3)
    assert out["status"] == "error"
    assert "SPY" in out["error"]


def test_empty_arctic_returns_error(patched_price_layer):
    signals = {"2026-05-15": {"universe": [{"ticker": "AAPL"}]}}
    preds = {"2026-05-15": {"predictions": [{"ticker": "AAPL", "predicted_alpha": 0.05}]}}
    s3 = _fake_s3(["2026-05-15"], ["2026-05-15"], signals, preds)
    with patch("store.arctic_reader.load_universe_from_arctic",
               return_value=({}, {})):
        out = psb.build_production_signal_inputs({"signals_bucket": "b"}, s3_client=s3)
    assert out["status"] == "error"
