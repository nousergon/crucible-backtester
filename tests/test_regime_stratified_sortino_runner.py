"""Tests for analysis/regime_stratified_sortino_runner.py — T2 pipeline wiring.

Exercises the end-to-end runner (load score_performance → stratify →
spread → assemble → publish canonical eval-artifact) against
synthetic SQLite DBs + in-memory S3 stubs. No real boto3.

The compute layer (regime_stratified_sortino.py) is covered by
test_regime_stratified_sortino.py — these tests focus on the
runner-specific pieces: artifact write shape, S3 prefix correctness,
graceful handling of empty / pre-migration DBs, and the dry-run path.
"""
from __future__ import annotations

import json
import sqlite3
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest


from analysis.regime_stratified_sortino_runner import (
    REGIME_STRATIFIED_SORTINO_PREFIX,
    run_regime_stratified_sortino,
)


class _FakeS3:
    """In-memory boto3 S3 stub."""

    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str | None = None) -> dict:
        self._objects[(Bucket, Key)] = Body if isinstance(Body, bytes) else Body.encode("utf-8")
        return {}

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        if (Bucket, Key) not in self._objects:
            raise KeyError(Key)
        return {"Body": BytesIO(self._objects[(Bucket, Key)])}


def _create_score_perf_db(path: Path, rows: list[dict]) -> None:
    """Create a synthetic score_performance table populated with rows."""
    conn = sqlite3.connect(path)
    try:
        conn.execute("""
            CREATE TABLE score_performance (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                score_date TEXT,
                eval_date_10d TEXT,
                eval_date_30d TEXT,
                market_regime TEXT,
                return_10d REAL,
                return_30d REAL,
                spy_10d_return REAL,
                spy_30d_return REAL,
                beat_spy_10d INTEGER,
                beat_spy_30d INTEGER
            )
        """)
        cols = [
            "ticker", "score_date", "eval_date_10d", "eval_date_30d",
            "market_regime", "return_10d", "return_30d",
            "spy_10d_return", "spy_30d_return",
            "beat_spy_10d", "beat_spy_30d",
        ]
        placeholders = ",".join(["?"] * len(cols))
        for r in rows:
            conn.execute(
                f"INSERT INTO score_performance ({','.join(cols)}) VALUES ({placeholders})",
                tuple(r.get(c) for c in cols),
            )
        conn.commit()
    finally:
        conn.close()


def _seed_well_populated_rows() -> list[dict]:
    """Enough rows to clear DEFAULT_MIN_PICKS_PER_STRATUM in bull + bear."""
    rows: list[dict] = []
    base_dates = pd.date_range("2025-01-06", periods=80, freq="W-MON")  # 80 weeks
    # 40 bull + 40 bear; bull-picks beat SPY, bear-picks lag SPY
    for i, d in enumerate(base_dates[:40]):
        rows.append({
            "ticker": f"BU{i}",
            "score_date": d.date().isoformat(),
            "eval_date_10d": (d + pd.Timedelta(days=10)).date().isoformat(),
            "eval_date_30d": (d + pd.Timedelta(days=30)).date().isoformat(),
            "market_regime": "bull",
            "return_10d": 0.04 + (i % 7) * 0.002,
            "return_30d": 0.10 + (i % 7) * 0.004,
            "spy_10d_return": 0.015,
            "spy_30d_return": 0.04,
            "beat_spy_10d": 1,
            "beat_spy_30d": 1,
        })
    for i, d in enumerate(base_dates[40:]):
        rows.append({
            "ticker": f"BE{i}",
            "score_date": d.date().isoformat(),
            "eval_date_10d": (d + pd.Timedelta(days=10)).date().isoformat(),
            "eval_date_30d": (d + pd.Timedelta(days=30)).date().isoformat(),
            "market_regime": "bear",
            "return_10d": -0.03 + (i % 5) * 0.002,
            "return_30d": -0.08 + (i % 5) * 0.004,
            "spy_10d_return": -0.01,
            "spy_30d_return": -0.03,
            "beat_spy_10d": 0,
            "beat_spy_30d": 0,
        })
    return rows


# pandas imported here to avoid clutter at top
import pandas as pd  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# run_regime_stratified_sortino — end-to-end
# ─────────────────────────────────────────────────────────────────────


class TestRunRegimeStratifiedSortino:
    def test_well_populated_db_produces_artifact(self, tmp_path):
        db = tmp_path / "research.db"
        _create_score_perf_db(db, _seed_well_populated_rows())
        s3 = _FakeS3()
        with patch("analysis.regime_stratified_sortino_runner.boto3") as mock_boto3:
            mock_boto3.client.return_value = s3
            result = run_regime_stratified_sortino(
                db_path=str(db), s3_bucket="test-bucket",
            )

        assert result["status"] == "ok"
        assert result["wrote"] is True
        assert result["artifact_key"].startswith("regime/stratified_sortino/")
        assert result["latest_key"] == "regime/stratified_sortino/latest.json"
        # Both bull + bear strata × 2 horizons = at least 4 (depending on
        # how many of the supported horizons land)
        assert result["n_strata"] >= 4

    def test_artifact_carries_t2_schema(self, tmp_path):
        db = tmp_path / "research.db"
        _create_score_perf_db(db, _seed_well_populated_rows())
        s3 = _FakeS3()
        with patch("analysis.regime_stratified_sortino_runner.boto3") as mock_boto3:
            mock_boto3.client.return_value = s3
            result = run_regime_stratified_sortino(
                db_path=str(db), s3_bucket="test-bucket",
            )
        payload = result["payload"]
        assert payload["schema_version"] == 1
        assert payload["eval_tier"] == "T2_downstream_stratified_sortino"
        assert "spread_10d" in payload
        assert "spread_30d" in payload
        assert "strata" in payload
        assert "method_metadata" in payload

    def test_artifact_body_matches_latest_sidecar(self, tmp_path):
        """T2's sidecar mirrors the artifact body — small enough that
        duplication is fine + simpler consumer. Pin so a refactor doesn't
        silently divergent the two."""
        db = tmp_path / "research.db"
        _create_score_perf_db(db, _seed_well_populated_rows())
        s3 = _FakeS3()
        with patch("analysis.regime_stratified_sortino_runner.boto3") as mock_boto3:
            mock_boto3.client.return_value = s3
            run_regime_stratified_sortino(
                db_path=str(db), s3_bucket="test-bucket",
            )

        artifact_keys = [
            k for (b, k) in s3._objects.keys()
            if b == "test-bucket" and k.startswith("regime/stratified_sortino/")
        ]
        # Two writes — forensic + latest sidecar
        assert "regime/stratified_sortino/latest.json" in artifact_keys
        artifact_key = next(
            k for k in artifact_keys
            if k != "regime/stratified_sortino/latest.json"
        )
        artifact_body = s3._objects[("test-bucket", artifact_key)]
        latest_body = s3._objects[("test-bucket", "regime/stratified_sortino/latest.json")]
        assert artifact_body == latest_body

    def test_no_write_when_bucket_missing(self, tmp_path):
        """Without s3_bucket, runner skips the write but still returns
        the payload — useful for ad-hoc CLI replays via the spot or local."""
        db = tmp_path / "research.db"
        _create_score_perf_db(db, _seed_well_populated_rows())
        result = run_regime_stratified_sortino(
            db_path=str(db), s3_bucket=None,
        )
        assert result["status"] == "ok"
        assert result["wrote"] is False
        assert "payload" in result
        assert "artifact_key" not in result

    def test_dry_run_no_write(self, tmp_path):
        """``write=False`` returns the payload without touching S3 even
        if a bucket is supplied — useful for unit/integration tests."""
        db = tmp_path / "research.db"
        _create_score_perf_db(db, _seed_well_populated_rows())
        s3 = _FakeS3()
        with patch("analysis.regime_stratified_sortino_runner.boto3") as mock_boto3:
            mock_boto3.client.return_value = s3
            result = run_regime_stratified_sortino(
                db_path=str(db), s3_bucket="test-bucket", write=False,
            )
        assert result["wrote"] is False
        assert s3._objects == {}

    def test_empty_db_returns_placeholder_payload(self, tmp_path):
        """Pre-data state: score_performance exists but is empty. Runner
        emits a payload with empty strata / null spread — must NOT crash
        so the evaluator's tracker can mark it 'no_data' gracefully."""
        db = tmp_path / "research.db"
        _create_score_perf_db(db, rows=[])
        s3 = _FakeS3()
        with patch("analysis.regime_stratified_sortino_runner.boto3") as mock_boto3:
            mock_boto3.client.return_value = s3
            result = run_regime_stratified_sortino(
                db_path=str(db), s3_bucket="test-bucket",
            )
        assert result["status"] == "ok"
        assert result["wrote"] is True
        assert result["n_strata"] == 0
        # Spread is "insufficient_sample" with no strata
        assert result["spread_10d_interpretation"] == "insufficient_sample"
        assert result["spread_30d_interpretation"] == "insufficient_sample"

    def test_prefix_constant_is_canonical(self):
        """The prefix anchors the dashboard reader + judge auditor; pin
        the value so a refactor can't silently move the artifact."""
        assert REGIME_STRATIFIED_SORTINO_PREFIX == "regime/stratified_sortino"


# ─────────────────────────────────────────────────────────────────────
# evaluate.py wiring — pin the module is registered
# ─────────────────────────────────────────────────────────────────────


def test_evaluate_imports_regime_stratified_sortino_runner():
    """Catch a refactor that removes the import or moves the runner —
    silent drift in evaluate.py would leave the eval running but the
    T2 module silently missing from the Saturday eval results."""
    import evaluate
    assert hasattr(evaluate, "regime_stratified_sortino_runner")


def test_evaluate_diagnostic_includes_t2_module():
    """The T2 module hook must appear inside _run_diagnostics — pinned
    by searching the source for the registry key. Catches accidental
    removal during evaluate.py refactors."""
    src = Path(__file__).resolve().parents[1] / "evaluate.py"
    body = src.read_text()
    assert '"regime_stratified_sortino"' in body, (
        "evaluate.py must register the regime_stratified_sortino module "
        "via tracker.run_module so it runs each Saturday alongside the "
        "other diagnostics. Wire it into _run_diagnostics."
    )
    assert "regime_stratified_sortino_runner.run_regime_stratified_sortino" in body
