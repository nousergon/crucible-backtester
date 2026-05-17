"""Unit tests for PR 1 slice C — the point-in-time walk-forward wiring
(`synthetic/predictor_backtest.py::run_walk_forward_inference` + the
`--walk-forward` dispatch). ROADMAP L2371 / Backtester Phase 2; plan
``alpha-engine-docs/private/pit-discipline-260515.md``.

The pure building blocks (pit_weights resolver, pit_folds splitter) are
locked by test_pit_weights.py / test_pit_folds.py. These tests lock the
*wiring*: that each fold is scored with weights resolved at knowledge-time
≤ its decision date, that cold-start folds are excluded + counted (never
future-substituted), that distinct archives are downloaded/built once, and
that the flag is OFF by default so the legacy single-pass path is
byte-unchanged until the manual flip.

S3 is mocked with unittest.mock per the repo convention; GBMScorer is
stubbed via sys.modules so no predictor repo / real booster is needed.
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

from synthetic import predictor_backtest
from synthetic.pit_weights import _ARCHIVE_PREFIX


# ── stubs ──────────────────────────────────────────────────────────────────

class _StubScorer:
    """Minimal GBMScorer stand-in: fixed feature_names, zero predictions."""

    feature_names = ["f0", "f1"]

    @classmethod
    def load(cls, _path):
        return cls()

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


@pytest.fixture
def _stub_gbm_scorer(monkeypatch):
    """Register a fake ``model.gbm_scorer`` so the runtime import inside
    run_walk_forward_inference resolves without a predictor checkout."""
    mod = types.ModuleType("model.gbm_scorer")
    mod.GBMScorer = _StubScorer
    pkg = types.ModuleType("model")
    monkeypatch.setitem(sys.modules, "model", pkg)
    monkeypatch.setitem(sys.modules, "model.gbm_scorer", mod)
    # _download_gbm_to_temp would hit boto/disk; the resolver already
    # guarantees the archive dir exists, so a no-op temp path is enough.
    monkeypatch.setattr(
        predictor_backtest, "_download_gbm_to_temp",
        lambda s3, bucket, model_key, meta_key: "/tmp/_wf_stub_model.txt",
    )
    return None


def _trading_dates(n: int) -> list[str]:
    base = dt.date(2026, 1, 1)
    return [(base + dt.timedelta(days=i)).isoformat() for i in range(n)]


def _features(dates: list[str]) -> dict[str, pd.DataFrame]:
    idx = pd.to_datetime(dates)
    cols = _StubScorer.feature_names
    return {
        t: pd.DataFrame(
            np.arange(len(dates) * len(cols), dtype=float).reshape(len(dates), -1),
            index=idx, columns=cols,
        )
        for t in ("AAA", "BBB")
    }


def _s3_listing(*archive_dates: str):
    s3 = MagicMock()
    s3.list_objects_v2.return_value = {
        "CommonPrefixes": [
            {"Prefix": f"{_ARCHIVE_PREFIX}{d}/"} for d in archive_dates
        ],
        "IsTruncated": False,
    }
    return s3


_WF = {"test_window": 2, "min_train": 4, "purge": 1, "embargo": 0,
       "train_mode": "expanding"}


# ── behavioural: PIT resolution + cold-start accounting ────────────────────

def test_walk_forward_resolves_per_fold_and_excludes_cold_start(_stub_gbm_scorer):
    """12 dates → folds test idx [4,5][6,7][8,9][10,11]; test-start dates
    01-05 / 01-07 / 01-09 / 01-11. Archives {01-06, 01-08}:
      - 01-05 fold → no archive ≤ it → cold-start excluded (NOT 01-06)
      - 01-07 fold → archive 01-06
      - 01-09 fold → archive 01-08
      - 01-11 fold → archive 01-08 (reused, cached)
    """
    dates = _trading_dates(12)
    s3 = _s3_listing("2026-01-06", "2026-01-08")

    preds, stats = predictor_backtest.run_walk_forward_inference(
        _features(dates), dates, "/nonexistent/predictor",
        bucket="b", wf_params=_WF, s3_client=s3,
    )

    assert stats["n_folds"] == 4
    assert stats["n_cold_start_excluded"] == 1
    assert stats["n_folds_scored"] == 3
    # No-future-fallback: the 01-05 fold is excluded, NOT pulled forward to
    # the 01-06 archive.
    assert stats["cold_start_test_starts"] == ["2026-01-05"]
    assert stats["archive_dates_used"] == ["2026-01-06", "2026-01-08"]
    assert stats["n_distinct_archives"] == 2
    # Scored windows: idx[6,7],[8,9],[10,11] → dates 01-07..01-12 (6 dates).
    assert set(preds) == {
        "2026-01-07", "2026-01-08", "2026-01-09",
        "2026-01-10", "2026-01-11", "2026-01-12",
    }
    assert "2026-01-05" not in preds and "2026-01-06" not in preds
    assert stats["n_test_dates_scored"] == 6
    assert stats["enabled"] is True


def test_distinct_archive_downloaded_once(monkeypatch, _stub_gbm_scorer):
    """The 01-08 archive serves two folds — booster download + tensor build
    must happen once per distinct archive / feature signature, not per fold
    (the perf invariant that keeps WF ≈ single-pass cost)."""
    dates = _trading_dates(12)
    s3 = _s3_listing("2026-01-06", "2026-01-08")

    dl_calls: list[str] = []
    monkeypatch.setattr(
        predictor_backtest, "_download_gbm_to_temp",
        lambda s3, bucket, mk, meta: dl_calls.append(mk) or "/tmp/m.txt",
    )
    build_calls: list[int] = []
    real_build = predictor_backtest.build_inference_tensor
    monkeypatch.setattr(
        predictor_backtest, "build_inference_tensor",
        lambda fbt, names: build_calls.append(1) or real_build(fbt, names),
    )

    predictor_backtest.run_walk_forward_inference(
        _features(dates), dates, "/x", bucket="b", wf_params=_WF, s3_client=s3,
    )

    # 2 distinct archives → 2 downloads (not 3, despite 3 scored folds).
    assert len(dl_calls) == 2
    # Single stable feature signature → tensor built exactly once.
    assert len(build_calls) == 1


def test_all_cold_start_yields_empty_predictions(_stub_gbm_scorer, caplog):
    """Every archive later than every decision date → all folds excluded,
    zero signals, and a loud ERROR (feedback_no_silent_fails)."""
    dates = _trading_dates(12)
    s3 = _s3_listing("2026-06-01")  # far after every fold decision date

    with caplog.at_level("ERROR"):
        preds, stats = predictor_backtest.run_walk_forward_inference(
            _features(dates), dates, "/x", bucket="b",
            wf_params=_WF, s3_client=s3,
        )

    assert preds == {}
    assert stats["n_folds_scored"] == 0
    assert stats["n_cold_start_excluded"] == stats["n_folds"] == 4
    assert any("ALL" in r.message and "cold-start" in r.message
               for r in caplog.records)


# ── wiring / default-OFF guards ────────────────────────────────────────────

def test_walk_forward_flag_defaults_off_and_is_dispatched():
    """run() must branch on config['walk_forward'] and only call the PIT
    path when set — the legacy single-pass path stays the default."""
    src = inspect.getsource(predictor_backtest.run)
    assert 'config.get("walk_forward"' in src
    assert "run_walk_forward_inference(" in src
    # download_gbm_model (live weights) must still be reachable on the
    # else branch — the default path is unchanged.
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
    """The refactored _download_gbm_to_temp keeps the operator-facing
    error substrings the model_source guard asserts, for any key — so a
    missing archived booster is just as legible as a missing live one."""
    s3 = MagicMock()
    s3.download_file.side_effect = Exception("NoSuchKey")
    akey = "predictor/weights/meta/archive/2026-05-10/momentum_model.txt"
    with pytest.raises(RuntimeError) as exc:
        predictor_backtest._download_gbm_to_temp(s3, "b", akey, akey + ".meta.json")
    msg = str(exc.value)
    assert "PredictorTraining" in msg and akey in msg
