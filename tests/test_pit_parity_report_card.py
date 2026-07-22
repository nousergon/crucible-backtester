"""Unit tests for config#3120's Report Card surface: ``evaluate.py``'s
best-effort read of ``backtest/{date}/pit_parity.json`` into
``pipeline_health`` for the weekly Report Card (rendered by
``reporter._section_pipeline_health``, tested separately in
tests/test_reporter.py).

Acceptance: a fixture failure record (status=failed) makes it through
``_load_pit_parity_report`` into the shape ``_section_pipeline_health``
renders as a visible FAILED row.
"""

from __future__ import annotations

import json

import pytest
from botocore.exceptions import ClientError

import evaluate


class _FakeBody:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload


class _FakeS3:
    """Minimal boto3-S3-client stand-in: get_object returns a canned body,
    raises a NoSuchKey ClientError, or raises an arbitrary ClientError,
    per the key's entry in ``responses``."""

    def __init__(self, responses: dict):
        self._responses = responses

    def get_object(self, Bucket, Key):  # noqa: N803 - mirrors boto3 signature
        outcome = self._responses[Key]
        if isinstance(outcome, Exception):
            raise outcome
        return {"Body": _FakeBody(json.dumps(outcome).encode())}


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "boom"}}, "GetObject")


def test_load_pit_parity_report_returns_failed_record():
    report = {
        "schema": "pit_parity-1.0.0",
        "run_date": "2026-07-17",
        "status": "failed",
        "error_class": "RuntimeError",
        "error_msg": "boom",
    }
    s3 = _FakeS3({"backtest/2026-07-17/pit_parity.json": report})
    out = evaluate._load_pit_parity_report(s3, "bucket", "backtest/2026-07-17")
    assert out == report


def test_load_pit_parity_report_returns_none_on_missing_key():
    """pit_parity disabled / not yet run for this date — expected, non-fatal."""
    s3 = _FakeS3({"backtest/2026-07-17/pit_parity.json": _client_error("NoSuchKey")})
    out = evaluate._load_pit_parity_report(s3, "bucket", "backtest/2026-07-17")
    assert out is None


def test_load_pit_parity_report_returns_none_on_other_client_error():
    """A real S3 problem (not NoSuchKey) must not raise — this is a
    secondary observability surface, never allowed to fail the evaluator."""
    s3 = _FakeS3({"backtest/2026-07-17/pit_parity.json": _client_error("AccessDenied")})
    out = evaluate._load_pit_parity_report(s3, "bucket", "backtest/2026-07-17")
    assert out is None


def test_load_pit_parity_report_returns_none_on_parse_error():
    class _BadBody:
        def read(self):
            return b"not json"

    class _BadS3:
        def get_object(self, Bucket, Key):  # noqa: N803
            return {"Body": _BadBody()}

    out = evaluate._load_pit_parity_report(_BadS3(), "bucket", "backtest/2026-07-17")
    assert out is None


def test_load_pit_parity_report_ok_record_roundtrips():
    report = {"schema": "pit_parity-1.0.0", "run_date": "2026-07-17", "pbo": None}
    s3 = _FakeS3({"backtest/2026-07-17/pit_parity.json": report})
    out = evaluate._load_pit_parity_report(s3, "bucket", "backtest/2026-07-17")
    assert out == report
    assert "status" not in out  # a completed run carries no status key
