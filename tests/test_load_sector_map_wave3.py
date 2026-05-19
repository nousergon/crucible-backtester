"""Wave-3 reader migration regression guard for
``pipeline_common.load_sector_map`` (ROADMAP L1401).

Pins read-order contract during the producer write-both soak:

    1. Try ``reference/price_cache/sector_map.json`` (new).
    2. Fall back to ``predictor/price_cache/sector_map.json`` (legacy).
    3. Both missing → None + WARNING.

After Wave-3 PR4 retires the legacy prefix, the fallback branch
becomes dead and can be deleted; this test will need its legacy
fixture removed at that time.
"""

from __future__ import annotations

import json

import pytest

import pipeline_common


class _StubS3:
    """Minimal S3 stub recording reads in order."""

    class _NoSuchKey(Exception):
        pass

    def __init__(self, store: dict[str, dict]) -> None:
        self._store = store
        self.reads: list[str] = []

    def get_object(self, *, Bucket, Key):
        self.reads.append(Key)
        if Key not in self._store:
            raise self._NoSuchKey(f"NoSuchKey: {Bucket}/{Key}")
        from io import BytesIO
        return {"Body": BytesIO(json.dumps(self._store[Key]).encode())}


@pytest.fixture
def _patch_boto3(monkeypatch):
    """Yield a setter that swaps boto3.client for a given stub."""
    stubs: dict[str, _StubS3] = {}

    def _set(stub: _StubS3) -> None:
        stubs["s3"] = stub
        monkeypatch.setattr(
            "boto3.client", lambda svc, *a, **kw: stub,
        )

    yield _set


def test_reads_reference_prefix_first(_patch_boto3):
    """When both prefixes have sector_map.json (the soak state), the
    reader MUST pick the new ``reference/`` key."""
    stub = _StubS3({
        "reference/price_cache/sector_map.json": {"AAPL": "tech_NEW"},
        "predictor/price_cache/sector_map.json": {"AAPL": "tech_OLD"},
    })
    _patch_boto3(stub)
    out = pipeline_common.load_sector_map({"predictor_paths": []})
    assert out == {"AAPL": "tech_NEW"}
    # Order matters: new prefix consulted first; legacy never read.
    assert stub.reads == ["reference/price_cache/sector_map.json"]


def test_falls_back_to_legacy_when_new_missing(_patch_boto3):
    """During the early soak window the new prefix might lag the legacy
    one. The reader must fall back without crashing."""
    stub = _StubS3({
        "predictor/price_cache/sector_map.json": {"AAPL": "tech_OLD"},
    })
    _patch_boto3(stub)
    out = pipeline_common.load_sector_map({"predictor_paths": []})
    assert out == {"AAPL": "tech_OLD"}
    assert stub.reads == [
        "reference/price_cache/sector_map.json",
        "predictor/price_cache/sector_map.json",
    ]


def test_returns_none_when_both_prefixes_missing(_patch_boto3, caplog):
    """If neither prefix has the artifact the reader returns None with
    a single WARNING. Mirrors the pre-migration behavior."""
    stub = _StubS3({})  # empty bucket
    _patch_boto3(stub)
    out = pipeline_common.load_sector_map({"predictor_paths": []})
    assert out is None
    # Both prefixes attempted.
    assert stub.reads == [
        "reference/price_cache/sector_map.json",
        "predictor/price_cache/sector_map.json",
    ]
