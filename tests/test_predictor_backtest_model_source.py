"""Guard: download_gbm_model must load the v3 Layer-1A momentum GBM,
not the legacy v2 artifact.

Context (2026-04-21): the backtester's predictor-backtest mode was
loading ``predictor/weights/gbm_latest.txt`` — a v2 artifact last
updated 2026-03-28 and ripped from production on 2026-04-13. Every
Saturday since has been measuring a dead model, and the last few
Saturday SF dry-runs have failed because the v2 meta.json stored its
feature list under the legacy ``feature_list`` key (current save writes
``feature_names``).

Fix: point download_gbm_model at ``predictor/weights/meta/momentum_model.txt``
— the v3 Layer-1A quant GBM, freshly trained every Saturday with current
metadata format. This measures the quant component in isolation, which
is the right standalone baseline for the stacked ensemble per
mnemon feedback_component_baseline_validation.

These tests guard against regressions back to the v2 path.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError


def _no_such_key_error(operation: str = "GetObject") -> ClientError:
    """Build the ClientError boto3 actually raises for a missing S3 key.

    The download-failure tests originally raised a bare ``Exception`` to
    simulate a missing object; that never happens in production — botocore
    surfaces a NoSuchKey as a ``ClientError``. Using the real type keeps the
    tests honest against ``_download_gbm_to_temp``'s narrowed
    ``except (ClientError, BotoCoreError, OSError)`` (#806).
    """
    return ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist."}},
        operation,
    )


def test_download_gbm_model_pulls_momentum_model(monkeypatch):
    """download_gbm_model must hit predictor/weights/meta/momentum_model.txt
    (and its meta.json), NOT predictor/weights/gbm_latest.txt."""
    calls: list[tuple] = []

    def fake_download_file(bucket, key, local_path):
        calls.append((bucket, key, local_path))

    mock_s3 = MagicMock()
    mock_s3.download_file.side_effect = fake_download_file
    monkeypatch.setattr("boto3.client", lambda *a, **kw: mock_s3)

    from synthetic import predictor_backtest
    predictor_backtest.download_gbm_model(bucket="alpha-engine-research")

    # Exactly the two expected S3 keys, in order.
    keys_downloaded = [key for _bucket, key, _path in calls]
    assert keys_downloaded == [
        "predictor/weights/meta/momentum_model.txt",
        "predictor/weights/meta/momentum_model.txt.meta.json",
    ], f"unexpected S3 keys: {keys_downloaded}"

    # Explicit regression guard — the legacy v2 paths must NEVER appear.
    for _bucket, key, _path in calls:
        assert "gbm_latest.txt" not in key, (
            f"legacy v2 artifact path used: {key}"
        )
        assert key.startswith("predictor/weights/meta/"), (
            f"key {key!r} is outside predictor/weights/meta/ — v3 path violated"
        )


def test_download_gbm_model_hard_fails_when_model_missing(monkeypatch):
    """Missing momentum_model.txt is a PredictorTraining-pipeline
    problem, not something to silently fall back on."""
    mock_s3 = MagicMock()
    mock_s3.download_file.side_effect = _no_such_key_error()
    monkeypatch.setattr("boto3.client", lambda *a, **kw: mock_s3)

    from synthetic import predictor_backtest
    with pytest.raises(RuntimeError) as exc:
        predictor_backtest.download_gbm_model(bucket="alpha-engine-research")
    msg = str(exc.value)
    assert "momentum_model.txt" in msg
    assert "PredictorTraining" in msg, (
        "error message should name the upstream owner so operators "
        "know where to investigate"
    )


def test_download_gbm_model_hard_fails_when_meta_missing(monkeypatch):
    """Booster downloads but meta.json is missing — must hard-fail, not
    continue with an unusable model (without meta.json the downstream
    feature_names check crashes in a less useful place)."""
    call_count = {"n": 0}

    def fake_download_file(bucket, key, local_path):
        call_count["n"] += 1
        # First call (booster) succeeds. Second call (meta.json) fails.
        if call_count["n"] == 2:
            raise _no_such_key_error()

    mock_s3 = MagicMock()
    mock_s3.download_file.side_effect = fake_download_file
    monkeypatch.setattr("boto3.client", lambda *a, **kw: mock_s3)

    from synthetic import predictor_backtest
    with pytest.raises(RuntimeError) as exc:
        predictor_backtest.download_gbm_model(bucket="alpha-engine-research")
    assert "metadata not found" in str(exc.value)
    assert "feature_names alignment" in str(exc.value)
