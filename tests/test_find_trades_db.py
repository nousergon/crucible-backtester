"""Tests for pipeline_common.find_trades_db — local + S3-fallback resolution (Phase B1b)."""

from pathlib import Path

from botocore.exceptions import ClientError

import pipeline_common
from pipeline_common import find_trades_db


class _FakeS3:
    def __init__(self, *, succeed: bool):
        self.succeed = succeed
        self.calls = []

    def download_file(self, bucket, key, dest):
        self.calls.append((bucket, key, dest))
        if not self.succeed:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "GetObject")
        Path(dest).write_text("fake-sqlite")


def test_local_path_preferred(tmp_path, monkeypatch):
    (tmp_path / "trades.db").write_text("local")
    # boto3 must not be touched when a local DB exists.
    monkeypatch.setattr(pipeline_common.boto3, "client", lambda *a, **k: 1 / 0)
    assert find_trades_db({"executor_paths": [str(tmp_path)]}) == str(tmp_path / "trades.db")


def test_s3_fallback_pulls_latest(tmp_path, monkeypatch):
    fake = _FakeS3(succeed=True)
    monkeypatch.setattr(pipeline_common.boto3, "client", lambda *a, **k: fake)
    out = find_trades_db({"executor_paths": [str(tmp_path)], "signals_bucket": "b"})
    assert out is not None and Path(out).exists()
    assert fake.calls == [("b", "trades/trades_latest.db", out)]


def test_s3_missing_returns_none_with_warning(tmp_path, monkeypatch, caplog):
    fake = _FakeS3(succeed=False)
    monkeypatch.setattr(pipeline_common.boto3, "client", lambda *a, **k: fake)
    with caplog.at_level("WARNING"):
        out = find_trades_db({"executor_paths": [str(tmp_path)], "signals_bucket": "b"})
    assert out is None
    assert any("N/A-MISSING-INPUT" in r.message for r in caplog.records)


def test_no_bucket_no_local_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_common.boto3, "client", lambda *a, **k: 1 / 0)
    assert find_trades_db({"executor_paths": [str(tmp_path)]}) is None
