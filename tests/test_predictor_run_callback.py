"""tests/test_predictor_run_callback.py — Stage 4 in-run features
persistence callback contract.

Stage 4 of the c5.large optimization arc added
``persist_features_callback`` to ``synthetic.predictor_backtest.run()``.
The callback fires right after GBM inference and before
``build_signals_by_date``, letting backtest.py persist features to S3
durably while run() drops the in-memory dict to free ~1.1 GB at the
post_build_signals checkpoint.

These tests validate the contract WITHOUT running the full pipeline
(no GBM, no ArcticDB). The pipeline orchestration is exercised by the
spot Saturday SF run; here we lock the small invariants:

1. Callback receives the actual features_by_ticker dict (not a copy)
2. Mutually-exclusive guard: keep_features=True + callback raises
3. Callback failure surfaces (not swallowed) — the spot must HardFail,
   not silently lose features that downstream Phase 4a/4c need

The full integration (callback + S3 persistence + Phase 4 lazy-load)
is exercised in test_predictor_data_prep_auto_skip.py's roundtrip.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from synthetic.predictor_backtest import run as predictor_run


class TestPersistFeaturesCallbackContract:
    def test_keep_features_and_callback_mutually_exclusive(self):
        """Both set is a contract bug — callback drops features, but
        keep_features=True implies they must survive into the result."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            predictor_run(
                config={},
                keep_features=True,
                persist_features_callback=lambda f: None,
            )

    def test_callback_invoked_with_features_then_dropped_from_result(self):
        """Callback receives the populated features dict before run()
        drops it. The result dict does NOT contain features_by_ticker
        when the callback path is taken."""
        # Stub everything that would do real work — we only care about
        # the callback wiring, not the GBM/ArcticDB pipeline.
        captured: dict[str, object] = {}

        def _capture(features):
            captured["received"] = type(features).__name__
            captured["n_tickers"] = len(features)

        # Build a fake features_by_ticker payload. The contract is that
        # the callback sees it before run() drops it; we don't need a
        # real one for the contract test.
        fake_features = {"AAPL": object(), "MSFT": object()}
        fake_predictions = {"2026-01-01": {"AAPL": 0.01}}

        # Patch the heavy steps so the callback wiring is the only thing
        # actually executed. We stub at the lowest level we can without
        # over-mocking.
        with (
            patch("synthetic.predictor_backtest.os.path.isdir", return_value=True),
            patch("synthetic.predictor_backtest.load_sector_map", return_value={}),
            patch(
                "store.arctic_reader.load_universe_from_arctic",
                return_value=({"AAPL": None}, fake_features),
            ),
            patch("synthetic.predictor_backtest._resolve_trading_dates",
                  return_value=["2026-01-01"]),
            patch("synthetic.predictor_backtest.build_price_matrix",
                  return_value=__import__("pandas").DataFrame()),
            patch("synthetic.predictor_backtest._extract_close",
                  return_value=None),
            patch("synthetic.predictor_backtest.build_ohlcv_df_by_ticker",
                  return_value={}),
            patch("synthetic.predictor_backtest.download_gbm_model",
                  return_value="/tmp/_test_gbm.txt"),
            patch("synthetic.predictor_backtest.run_inference",
                  return_value=fake_predictions),
            patch("synthetic.predictor_backtest.build_signals_by_date",
                  return_value={"2026-01-01": {"buy_candidates": []}}),
            patch("os.unlink"),
        ):
            result = predictor_run(
                config={"predictor_paths": ["/tmp"]},
                persist_features_callback=_capture,
            )

        # Callback fired before drop — saw 2 tickers
        assert captured["received"] == "dict"
        assert captured["n_tickers"] == 2

        # Result has the pipeline outputs but NOT features
        assert result["status"] == "ok"
        assert "features_by_ticker" not in result

    def test_keep_predictions_includes_adv_dollar_no_unbound_error(self):
        """Regression (L4485-c / #270): the W3.4 adv_dollar computation must
        not reference ``price_data`` after it's ``del``'d.

        The original code called ``_compute_adv_dollar(price_data)`` at
        result-assembly time — AFTER ``del price_data`` — so every
        ``keep_predictions=True`` run (pit_parity's run_predictor_backtest,
        --mode=portfolio-optimizer-backtest) raised
        ``UnboundLocalError: ... 'price_data' ...``. The pit_parity stage
        swallowed it as observational, so it surfaced only as a silently
        missing ``horizon_net_alpha.json``. This drives run() with
        keep_predictions=True and asserts the result carries
        ``adv_dollar_by_ticker`` with no UnboundLocalError.
        """
        fake_features = {"AAPL": object()}
        fake_predictions = {"2026-01-01": {"AAPL": 0.01}}

        with (
            patch("synthetic.predictor_backtest.os.path.isdir", return_value=True),
            patch("synthetic.predictor_backtest.load_sector_map", return_value={}),
            patch("synthetic.predictor_backtest._assert_ram_headroom"),
            patch(
                "store.arctic_reader.load_universe_from_arctic",
                return_value=({"AAPL": None}, fake_features),
            ),
            patch("synthetic.predictor_backtest._resolve_trading_dates",
                  return_value=["2026-01-01"]),
            patch("synthetic.predictor_backtest.build_price_matrix",
                  return_value=__import__("pandas").DataFrame()),
            patch("synthetic.predictor_backtest._extract_close",
                  return_value=None),
            patch("synthetic.predictor_backtest.build_ohlcv_df_by_ticker",
                  return_value={}),
            patch("synthetic.predictor_backtest.download_gbm_model",
                  return_value="/tmp/_test_gbm.txt"),
            patch("synthetic.predictor_backtest.run_inference",
                  return_value=fake_predictions),
            patch("synthetic.predictor_backtest.build_signals_by_date",
                  return_value={"2026-01-01": {"buy_candidates": []}}),
            # _compute_adv_dollar is exercised for arg-evaluation (the bug
            # was the unbound ``price_data`` arg); stub its body so we don't
            # need real OHLCV volume frames.
            patch("synthetic.predictor_backtest._compute_adv_dollar",
                  return_value={"AAPL": 1.0e6}),
            patch("os.unlink"),
        ):
            result = predictor_run(
                config={"predictor_paths": ["/tmp"]},
                keep_predictions=True,
            )

        assert result["status"] == "ok"
        assert result.get("predictions_by_date") == fake_predictions
        assert result.get("adv_dollar_by_ticker") == {"AAPL": 1.0e6}

    def test_callback_exception_propagates_not_swallowed(self):
        """If the persist callback raises (e.g. S3 IAM denied), the
        spot must HardFail — features can't reach Phase 4 lazy-load
        without persistence, and silently continuing would let
        Phase 4 evaluators run without the features they need."""
        def _failing(_features):
            raise RuntimeError("S3 IAM denied")

        with (
            patch("synthetic.predictor_backtest.os.path.isdir", return_value=True),
            patch("synthetic.predictor_backtest.load_sector_map", return_value={}),
            patch(
                "store.arctic_reader.load_universe_from_arctic",
                return_value=({"AAPL": None}, {"AAPL": object()}),
            ),
            patch("synthetic.predictor_backtest._resolve_trading_dates",
                  return_value=["2026-01-01"]),
            patch("synthetic.predictor_backtest.build_price_matrix",
                  return_value=__import__("pandas").DataFrame()),
            patch("synthetic.predictor_backtest._extract_close",
                  return_value=None),
            patch("synthetic.predictor_backtest.build_ohlcv_df_by_ticker",
                  return_value={}),
            patch("synthetic.predictor_backtest.download_gbm_model",
                  return_value="/tmp/_test_gbm.txt"),
            patch("synthetic.predictor_backtest.run_inference",
                  return_value={}),
            patch("os.unlink"),
        ):
            with pytest.raises(RuntimeError, match="S3 IAM denied"):
                predictor_run(
                    config={"predictor_paths": ["/tmp"]},
                    persist_features_callback=_failing,
                )
