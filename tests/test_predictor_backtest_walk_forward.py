"""Unit tests for PR 1 slice C — the point-in-time walk-forward wiring
(`synthetic/predictor_backtest.py::run_walk_forward_inference` + the
`--walk-forward` dispatch). ROADMAP L2371 / Backtester Phase 2; plan
``alpha-engine-docs/private/pit-discipline-260515.md``.

The pure fold-splitter building block is locked by test_pit_folds.py. These
tests lock the *wiring*: that every purged + embargoed fold is scored with the
deterministic momentum baseline (config#1518 — the momentum GBM retired
2026-05-09, so there is no per-fold archived-weight resolution and no
cold-start-from-missing-weights exclusion), that the inference tensor is built
exactly once, that the scorer receives the canonical momentum feature set, and
that the flag is OFF by default so the legacy single-pass path is byte-unchanged
until the manual flip.

The predictor's ``model.momentum_scorer.predict_array`` is stubbed via
sys.modules so no predictor checkout / real booster is needed.
"""

from __future__ import annotations

import datetime as dt
import inspect
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from botocore.exceptions import ClientError

from synthetic import predictor_backtest


# ── stubs ──────────────────────────────────────────────────────────────────

# The canonical momentum feature set the adapter feeds the deterministic scorer.
_MOM_FEATURES = predictor_backtest._MOMENTUM_FEATURE_NAMES


@pytest.fixture
def _stub_momentum_scorer(monkeypatch):
    """Register a fake ``model.momentum_scorer`` so the runtime import inside
    run_walk_forward_inference resolves without a predictor checkout.

    The stub ``predict_array`` returns a simple deterministic function of the
    raw features (row-sum) and records the feature_names it is handed, so tests
    can assert both the values flow through and the canonical feature set is
    passed.
    """
    calls: dict = {"feature_names": [], "n": 0}

    def _predict_array(X, feature_names):
        calls["feature_names"].append(list(feature_names))
        calls["n"] += 1
        return X.sum(axis=1).astype(float)

    mod = types.ModuleType("model.momentum_scorer")
    mod.predict_array = _predict_array
    pkg = types.ModuleType("model")
    monkeypatch.setitem(sys.modules, "model", pkg)
    monkeypatch.setitem(sys.modules, "model.momentum_scorer", mod)
    return calls


def _trading_dates(n: int) -> list[str]:
    base = dt.date(2026, 1, 1)
    return [(base + dt.timedelta(days=i)).isoformat() for i in range(n)]


def _features(dates: list[str], vals: dict | None = None) -> dict[str, pd.DataFrame]:
    """Per-ticker feature frames over the canonical momentum columns.

    ``vals`` maps ticker -> the constant value for every feature cell of that
    ticker, so the stub row-sum is ``len(_MOM_FEATURES) * value`` and therefore
    predictable per ticker. Default: AAA=1.0, BBB=2.0.
    """
    vals = vals or {"AAA": 1.0, "BBB": 2.0}
    idx = pd.to_datetime(dates)
    return {
        t: pd.DataFrame(
            np.full((len(dates), len(_MOM_FEATURES)), v, dtype=float),
            index=idx, columns=_MOM_FEATURES,
        )
        for t, v in vals.items()
    }


_WF = {"test_window": 2, "min_train": 4, "purge": 1, "embargo": 0,
       "train_mode": "expanding"}


# ── behavioural: deterministic per-fold scoring ────────────────────────────

def test_walk_forward_scores_every_fold_deterministically(_stub_momentum_scorer):
    """12 dates → folds test idx [4,5][6,7][8,9][10,11]; test-start dates
    01-05 / 01-07 / 01-09 / 01-11. With a deterministic momentum baseline
    there is no archived-weight resolution, so EVERY fold scores (no cold-start
    exclusion) and all eight test-window dates 01-05..01-12 get predictions —
    the 01-05 fold that the old archive-gated design excluded is now scored.
    """
    dates = _trading_dates(12)

    preds, stats = predictor_backtest.run_walk_forward_inference(
        _features(dates), dates, "/nonexistent/predictor",
        bucket="b", wf_params=_WF,
    )

    assert stats["n_folds"] == 4
    assert stats["n_folds_scored"] == 4
    assert stats["n_cold_start_excluded"] == 0
    assert stats["cold_start_test_starts"] == []
    assert stats["momentum_source"] == "deterministic_baseline"
    assert stats["enabled"] is True
    assert set(preds) == {
        "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08",
        "2026-01-09", "2026-01-10", "2026-01-11", "2026-01-12",
    }
    assert stats["n_test_dates_scored"] == 8
    # The adapter fed the deterministic scorer the canonical momentum features.
    assert _stub_momentum_scorer["feature_names"]
    assert all(fn == _MOM_FEATURES for fn in _stub_momentum_scorer["feature_names"])


def test_predictions_flow_from_deterministic_scorer(_stub_momentum_scorer):
    """The scored alpha for each (date, ticker) is exactly the scorer output
    for that ticker's raw features — locks that the tensor rows reach
    predict_array and its result is what lands in predictions_by_date. Stub
    returns the row-sum, so AAA(=1.0 across 4 feats)→4.0, BBB(=2.0)→8.0.
    """
    dates = _trading_dates(12)

    preds, _ = predictor_backtest.run_walk_forward_inference(
        _features(dates), dates, "/x", bucket="b", wf_params=_WF,
    )

    n_feat = len(_MOM_FEATURES)
    for row in preds.values():
        assert row["AAA"] == pytest.approx(1.0 * n_feat)
        assert row["BBB"] == pytest.approx(2.0 * n_feat)


def test_tensor_built_exactly_once(monkeypatch, _stub_momentum_scorer):
    """The momentum feature set is fixed, so the inference tensor is built once
    (the perf invariant that keeps WF ≈ single-pass cost), with the canonical
    feature names — not once per fold."""
    dates = _trading_dates(12)

    build_calls: list[list[str]] = []
    real_build = predictor_backtest.build_inference_tensor
    monkeypatch.setattr(
        predictor_backtest, "build_inference_tensor",
        lambda fbt, names: build_calls.append(list(names)) or real_build(fbt, names),
    )

    predictor_backtest.run_walk_forward_inference(
        _features(dates), dates, "/x", bucket="b", wf_params=_WF,
    )

    assert len(build_calls) == 1
    assert build_calls[0] == _MOM_FEATURES


def test_all_masked_rows_yield_empty_predictions_and_loud_error(
    _stub_momentum_scorer, caplog
):
    """Folds exist but every (date, ticker) feature row is NaN → all rows
    NaN-masked in _predict_from_tensor → zero signals + a loud ERROR
    (feedback_no_silent_fails), attributing the emptiness to the feature store,
    not the momentum leg."""
    dates = _trading_dates(12)
    feats = _features(dates)
    for df in feats.values():
        df.iloc[:, :] = np.nan

    with caplog.at_level("ERROR"):
        preds, stats = predictor_backtest.run_walk_forward_inference(
            feats, dates, "/x", bucket="b", wf_params=_WF,
        )

    assert preds == {}
    assert stats["n_folds"] == 4
    assert stats["n_test_dates_scored"] == 0
    assert any("ZERO test dates scored" in r.message for r in caplog.records)


def test_no_folds_yields_empty_predictions(_stub_momentum_scorer):
    """Too few dates for even one fold → empty predictions, no crash, no
    spurious error (folds==0 is a benign short-history case, not a failure)."""
    dates = _trading_dates(3)  # < min_train + test_window

    preds, stats = predictor_backtest.run_walk_forward_inference(
        _features(dates), dates, "/x", bucket="b", wf_params=_WF,
    )

    assert preds == {}
    assert stats["n_folds"] == 0
    assert stats["n_folds_scored"] == 0
    assert stats["n_test_dates_scored"] == 0


# ── wiring / default-OFF guards ────────────────────────────────────────────

def test_walk_forward_flag_defaults_off_and_is_dispatched():
    """run() must branch on config['walk_forward'] and only call the PIT
    path when set — the legacy single-pass path stays the default."""
    src = inspect.getsource(predictor_backtest.run)
    assert 'config.get("walk_forward"' in src
    assert "run_walk_forward_inference(" in src
    # download_gbm_model (single-pass legacy baseline) must still be reachable
    # on the else branch — the default path is unchanged (config#1518 scoped it
    # out deliberately).
    assert "download_gbm_model(bucket=bucket)" in src


def test_metadata_carries_walk_forward_block():
    """The wf_stats block must be threaded into metadata so PR 3's
    --pit-parity report can consume it."""
    src = inspect.getsource(predictor_backtest.run)
    assert '"walk_forward": wf_stats' in src


def test_wf_defaults_match_plan():
    """Plan pit-discipline-260515.md §D1 locked defaults."""
    d = predictor_backtest._WF_DEFAULTS
    assert d == {
        "test_window": 21, "min_train": 504, "purge": 21,
        "embargo": 2, "train_mode": "expanding",
    }


def test_download_helper_messages_survive_pit_keys():
    """The _download_gbm_to_temp helper (still used by the single-pass
    download_gbm_model path) keeps the operator-facing error substrings the
    model_source guard asserts, for any key."""
    s3 = MagicMock()
    # botocore raises ClientError (NoSuchKey) for a missing object — the real
    # type _download_gbm_to_temp's narrowed except now catches (#806).
    s3.download_file.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist."}},
        "GetObject",
    )
    akey = "predictor/weights/meta/archive/2026-05-10/momentum_model.txt"
    with pytest.raises(RuntimeError) as exc:
        predictor_backtest._download_gbm_to_temp(s3, "b", akey, akey + ".meta.json")
    msg = str(exc.value)
    assert "PredictorTraining" in msg and akey in msg
