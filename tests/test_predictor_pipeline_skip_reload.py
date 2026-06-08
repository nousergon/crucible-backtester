"""
tests/test_predictor_pipeline_skip_reload.py — L4527 skip/resume: skipping the
~121-min predictor pipeline reloads its artifacts from S3 instead of re-running.

Covers `_load_predictor_artifacts` (the S3 reloader) against an in-memory fake
S3. The end-to-end `predictor_pipeline` block wiring (ctx.skipped → reload) is
exercised by the existing pipeline tests; here we pin the reloader contract.
"""

from __future__ import annotations

import io
import json

import pandas as pd
import pytest
from botocore.exceptions import ClientError

import backtest


class _FakeS3:
    def __init__(self):
        self.store: dict[tuple[str, str], bytes] = {}
        self.error_code: str | None = None  # set to force get_object errors

    def put(self, bucket, key, body: bytes):
        self.store[(bucket, key)] = body

    def get_object(self, *, Bucket, Key):
        if self.error_code is not None:
            raise ClientError(
                {"Error": {"Code": self.error_code, "Message": "x"}}, "GetObject"
            )
        if (Bucket, Key) not in self.store:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject"
            )
        body = self.store[(Bucket, Key)]

        class _B:
            def __init__(self, b): self._b = b
            def read(self): return self._b

        return {"Body": _B(body)}


@pytest.fixture
def fake_s3(monkeypatch):
    fake = _FakeS3()
    monkeypatch.setattr(backtest.boto3, "client", lambda svc: fake)
    return fake


_CFG = {"signals_bucket": "alpha-engine-research"}
_DATE = "2026-06-08"
_PREFIX = "backtest/2026-06-08"


def _seed_stats(fake, stats):
    fake.put("alpha-engine-research", f"{_PREFIX}/predictor_stats.json",
             json.dumps(stats).encode())


def _seed_sweep(fake, df):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    fake.put("alpha-engine-research", f"{_PREFIX}/predictor_sweep_df.parquet",
             buf.getvalue())


def test_reload_both_artifacts_present(fake_s3):
    _seed_stats(fake_s3, {"status": "ok", "ic": 0.12})
    _seed_sweep(fake_s3, pd.DataFrame({"combo": [1, 2], "sharpe": [0.5, 0.8]}))
    stats, sweep = backtest._load_predictor_artifacts(_CFG, _DATE)
    assert stats == {"status": "ok", "ic": 0.12}
    assert list(sweep["combo"]) == [1, 2]


def test_reload_missing_artifacts_returns_none_non_fatal(fake_s3, caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="backtest"):
        stats, sweep = backtest._load_predictor_artifacts(_CFG, _DATE)
    assert stats is None
    assert sweep is None
    assert any("absent" in r.getMessage() for r in caplog.records)


def test_reload_partial_stats_only(fake_s3):
    _seed_stats(fake_s3, {"status": "ok"})
    stats, sweep = backtest._load_predictor_artifacts(_CFG, _DATE)
    assert stats == {"status": "ok"}
    assert sweep is None


def test_reload_non_404_error_fails_loud(fake_s3):
    fake_s3.error_code = "AccessDenied"
    with pytest.raises(ClientError):
        backtest._load_predictor_artifacts(_CFG, _DATE)


def test_reload_uses_output_bucket_when_set(monkeypatch):
    fake = _FakeS3()
    monkeypatch.setattr(backtest.boto3, "client", lambda svc: fake)
    fake.put("other-bucket", f"{_PREFIX}/predictor_stats.json",
             json.dumps({"status": "ok"}).encode())
    stats, _ = backtest._load_predictor_artifacts(
        {"output_bucket": "other-bucket", "signals_bucket": "alpha-engine-research"},
        _DATE,
    )
    assert stats == {"status": "ok"}
