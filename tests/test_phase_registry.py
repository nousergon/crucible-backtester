"""
tests/test_phase_registry.py — PhaseRegistry auto-skip + skip/force flags.

Covers the decision matrix (should_run + reasons) and the marker
read/write contract. S3 is stubbed via a minimal in-memory fake client
so the tests run offline.

Motivated by ROADMAP Backtester P0 "Phase-selective backtest execution —
skip already-successful phases on retry".
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from pipeline_common import PhaseRegistry, _marker_key


class _FakeS3:
    """Minimal in-memory stand-in for a boto3 S3 client.

    Supports get_object / put_object keyed by (Bucket, Key). get_object
    raises a NoSuchKey ClientError on miss (matching the real contract
    that PhaseRegistry swallows).
    """

    def __init__(self):
        self.store: dict[tuple[str, str], bytes] = {}
        self.put_calls: list[dict] = []
        self.get_calls: list[dict] = []
        self.head_calls: list[dict] = []

    def get_object(self, *, Bucket, Key):
        self.get_calls.append({"Bucket": Bucket, "Key": Key})
        if (Bucket, Key) not in self.store:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "not found"}},
                "GetObject",
            )
        body = self.store[(Bucket, Key)]

        class _Body:
            def __init__(self, b): self._b = b
            def read(self): return self._b

        return {"Body": _Body(body)}

    def head_object(self, *, Bucket, Key):
        self.head_calls.append({"Bucket": Bucket, "Key": Key})
        if (Bucket, Key) not in self.store:
            # Real S3 head_object returns a 404 (Code "404"/"NotFound"),
            # not the NoSuchKey body get_object returns.
            raise ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}},
                "HeadObject",
            )
        return {"ContentLength": len(self.store[(Bucket, Key)])}

    def put_object(self, *, Bucket, Key, Body, ContentType=None):
        self.put_calls.append({"Bucket": Bucket, "Key": Key, "Body": Body})
        if isinstance(Body, str):
            Body = Body.encode()
        self.store[(Bucket, Key)] = Body

    def seed(self, bucket: str, date: str, phase: str, marker: dict):
        self.store[(bucket, _marker_key(date, phase))] = json.dumps(marker).encode()


@pytest.fixture
def s3():
    return _FakeS3()


def _make_registry(s3, **kwargs):
    defaults = dict(date="2026-04-23", bucket="test-bucket", s3_client=s3)
    defaults.update(kwargs)
    return PhaseRegistry(**defaults)


# ── should_run decision matrix ──────────────────────────────────────────────


def test_default_runs(s3):
    r = _make_registry(s3)
    run, reason = r.should_run("simulate", supports_auto_skip=True)
    assert run is True
    assert reason == "default_run"


def test_explicit_skip(s3):
    r = _make_registry(s3, skip_phases=["simulate"])
    run, reason = r.should_run("simulate", supports_auto_skip=True)
    assert run is False
    assert reason == "explicit_skip"


def test_only_phases_filter(s3):
    r = _make_registry(s3, only_phases=["param_sweep"])
    run, reason = r.should_run("simulate", supports_auto_skip=True)
    assert run is False
    assert reason == "only_phases_filter"
    # but param_sweep itself runs
    run, reason = r.should_run("param_sweep", supports_auto_skip=True)
    assert run is True


def test_auto_skip_honored_when_marker_present(s3):
    s3.seed("test-bucket", "2026-04-23", "simulate", {
        "phase": "simulate", "date": "2026-04-23", "status": "ok",
    })
    r = _make_registry(s3)
    run, reason = r.should_run("simulate", supports_auto_skip=True)
    assert run is False
    assert reason == "auto_skip_marker_ok"


def test_auto_skip_ignored_without_opt_in(s3):
    """Phases that don't yet know how to persist + reload their outputs
    must pass supports_auto_skip=False. Even with a marker present,
    should_run returns True so the phase re-executes."""
    s3.seed("test-bucket", "2026-04-23", "simulate", {
        "phase": "simulate", "date": "2026-04-23", "status": "ok",
    })
    r = _make_registry(s3)
    run, reason = r.should_run("simulate", supports_auto_skip=False)
    assert run is True
    assert reason == "not_auto_skippable"


def test_force_overrides_marker(s3):
    s3.seed("test-bucket", "2026-04-23", "simulate", {
        "phase": "simulate", "date": "2026-04-23", "status": "ok",
    })
    r = _make_registry(s3, force=True)
    run, reason = r.should_run("simulate", supports_auto_skip=True)
    assert run is True
    assert reason == "force_rerun"


def test_force_phases_targets_specific(s3):
    s3.seed("test-bucket", "2026-04-23", "simulate", {
        "phase": "simulate", "date": "2026-04-23", "status": "ok",
    })
    s3.seed("test-bucket", "2026-04-23", "param_sweep", {
        "phase": "param_sweep", "date": "2026-04-23", "status": "ok",
    })
    r = _make_registry(s3, force_phases=["simulate"])
    run_s, reason_s = r.should_run("simulate", supports_auto_skip=True)
    run_p, reason_p = r.should_run("param_sweep", supports_auto_skip=True)
    assert run_s is True
    assert reason_s == "force_phase_rerun"
    # param_sweep's marker is still honored
    assert run_p is False
    assert reason_p == "auto_skip_marker_ok"


def test_error_marker_does_not_auto_skip(s3):
    """A phase that failed last time should re-run by default."""
    s3.seed("test-bucket", "2026-04-23", "simulate", {
        "phase": "simulate", "date": "2026-04-23", "status": "error",
    })
    r = _make_registry(s3)
    run, reason = r.should_run("simulate", supports_auto_skip=True)
    assert run is True
    assert reason == "default_run"


# ── L4524: artifact-validated checkpoints (markers can't lie) ───────────────


def _seed_artifact(s3, key: str, body: bytes = b"x"):
    s3.store[("test-bucket", key)] = body


def test_auto_skip_honored_when_declared_artifact_present(s3):
    """A status=ok marker whose declared artifact exists auto-skips."""
    art = "backtest/2026-04-23/portfolio_stats.json"
    _seed_artifact(s3, art)
    s3.seed("test-bucket", "2026-04-23", "simulate", {
        "phase": "simulate", "date": "2026-04-23", "status": "ok",
        "artifact_keys": [art],
    })
    r = _make_registry(s3)
    run, reason = r.should_run("simulate", supports_auto_skip=True)
    assert run is False
    assert reason == "auto_skip_marker_ok"
    # The declared artifact was actually probed.
    assert any(c["Key"] == art for c in s3.head_calls)


def test_marker_invalid_when_declared_artifact_absent(s3, caplog):
    """A status=ok marker whose declared artifact is GONE is a lie — the
    phase must re-run (the L4518/L4521 trust-and-yield-empty failure)."""
    import logging
    s3.seed("test-bucket", "2026-04-23", "param_sweep", {
        "phase": "param_sweep", "date": "2026-04-23", "status": "ok",
        # Declares sweep_df.parquet but it was never written / was pruned.
        "artifact_keys": ["backtest/2026-04-23/sweep_df.parquet"],
    })
    r = _make_registry(s3)
    with caplog.at_level(logging.WARNING, logger="pipeline_common"):
        run, reason = r.should_run("param_sweep", supports_auto_skip=True)
    assert run is True
    assert reason == "marker_artifact_missing"
    assert any("INVALID" in rec.getMessage() for rec in caplog.records)


def test_marker_invalid_when_any_one_declared_artifact_absent(s3):
    """All declared artifacts must exist — one missing invalidates the marker."""
    present = "backtest/2026-04-23/portfolio_stats.json"
    _seed_artifact(s3, present)
    s3.seed("test-bucket", "2026-04-23", "simulate", {
        "phase": "simulate", "date": "2026-04-23", "status": "ok",
        "artifact_keys": [present, "backtest/2026-04-23/sweep_df.parquet"],
    })
    r = _make_registry(s3)
    run, reason = r.should_run("simulate", supports_auto_skip=True)
    assert run is True
    assert reason == "marker_artifact_missing"


def test_marker_with_no_declared_artifacts_still_auto_skips(s3):
    """Back-compat: a marker that declares no artifacts has nothing to
    validate and is honored (existing markers / artifact-free phases)."""
    s3.seed("test-bucket", "2026-04-23", "simulate", {
        "phase": "simulate", "date": "2026-04-23", "status": "ok",
    })
    r = _make_registry(s3)
    run, reason = r.should_run("simulate", supports_auto_skip=True)
    assert run is False
    assert reason == "auto_skip_marker_ok"
    # Nothing declared → no head_object probes.
    assert s3.head_calls == []


def test_artifact_validation_cached_across_repeat_should_run(s3):
    """The phase() context manager re-asks should_run; the head_object
    probes must not be re-issued (mirrors the marker-read cache)."""
    art = "backtest/2026-04-23/portfolio_stats.json"
    _seed_artifact(s3, art)
    s3.seed("test-bucket", "2026-04-23", "simulate", {
        "phase": "simulate", "date": "2026-04-23", "status": "ok",
        "artifact_keys": [art],
    })
    r = _make_registry(s3)
    r.should_run("simulate", supports_auto_skip=True)
    r.should_run("simulate", supports_auto_skip=True)
    r.should_run("simulate", supports_auto_skip=True)
    assert len(s3.head_calls) == 1


def test_artifact_validation_raises_on_non_404_error(s3):
    """A transient/permission S3 error during validation must fail loud,
    not silently flip the skip decision (fail-loud rule)."""
    art = "backtest/2026-04-23/portfolio_stats.json"
    s3.seed("test-bucket", "2026-04-23", "simulate", {
        "phase": "simulate", "date": "2026-04-23", "status": "ok",
        "artifact_keys": [art],
    })

    def _boom(*, Bucket, Key):
        raise ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "nope"}}, "HeadObject"
        )

    s3.head_object = _boom
    r = _make_registry(s3)
    with pytest.raises(ClientError):
        r.should_run("simulate", supports_auto_skip=True)


def test_phase_context_reruns_when_declared_artifact_absent(s3):
    """End-to-end through the context manager: a marker whose artifact is
    gone yields a non-skipped phase (ctx.skipped is False)."""
    s3.seed("test-bucket", "2026-04-23", "param_sweep", {
        "phase": "param_sweep", "date": "2026-04-23", "status": "ok",
        "artifact_keys": ["backtest/2026-04-23/sweep_df.parquet"],
    })
    r = _make_registry(s3)
    with r.phase("param_sweep", supports_auto_skip=True) as ctx:
        assert ctx.skipped is False
        assert ctx.skip_reason == "marker_artifact_missing"


def test_explicit_skip_trumps_force(s3):
    """Operator passed BOTH --skip-phases and --force: explicit skip wins
    (you asked for a skip explicitly, force only relaxes auto-skip)."""
    r = _make_registry(s3, skip_phases=["simulate"], force=True)
    run, reason = r.should_run("simulate", supports_auto_skip=True)
    assert run is False
    assert reason == "explicit_skip"


def test_only_phases_trumps_force(s3):
    r = _make_registry(s3, only_phases=["param_sweep"], force=True)
    run, reason = r.should_run("simulate", supports_auto_skip=True)
    assert run is False
    assert reason == "only_phases_filter"


# ── Marker read/write contract ──────────────────────────────────────────────


def test_marker_cache_reads_s3_once_per_phase(s3):
    """Repeated should_run calls for the same phase don't re-hit S3."""
    s3.seed("test-bucket", "2026-04-23", "simulate", {
        "phase": "simulate", "date": "2026-04-23", "status": "ok",
    })
    r = _make_registry(s3)
    r.should_run("simulate", supports_auto_skip=True)
    r.should_run("simulate", supports_auto_skip=True)
    r.should_run("simulate", supports_auto_skip=True)
    assert len(s3.get_calls) == 1


def test_corrupt_marker_treated_as_absent(s3, caplog):
    """Unparseable JSON shouldn't wedge the pipeline."""
    import logging
    s3.store[("test-bucket", _marker_key("2026-04-23", "simulate"))] = b"not json"
    r = _make_registry(s3)
    with caplog.at_level(logging.WARNING, logger="pipeline_common"):
        run, reason = r.should_run("simulate", supports_auto_skip=True)
    assert run is True
    assert reason == "default_run"
    assert any("malformed" in rec.getMessage() for rec in caplog.records)


def test_phase_context_writes_ok_marker(s3):
    r = _make_registry(s3)
    with r.phase("simulate", supports_auto_skip=True) as ctx:
        assert ctx.skipped is False
        ctx.record_artifact("backtest/2026-04-23/.phases/simulate.parquet")

    # Exactly one put_object call, keyed to the expected path
    assert len(s3.put_calls) == 1
    call = s3.put_calls[0]
    assert call["Key"] == "backtest/2026-04-23/.phases/simulate.json"
    marker = json.loads(call["Body"])
    assert marker["phase"] == "simulate"
    assert marker["date"] == "2026-04-23"
    assert marker["status"] == "ok"
    assert marker["schema_version"] == 1
    assert marker["artifact_keys"] == [
        "backtest/2026-04-23/.phases/simulate.parquet"
    ]
    assert marker["duration_s"] >= 0
    assert marker["started_at"].endswith("Z")
    assert marker["completed_at"].endswith("Z")
    assert marker["error"] is None


def test_phase_context_writes_error_marker_on_exception(s3):
    r = _make_registry(s3)
    with pytest.raises(RuntimeError, match="boom"):
        with r.phase("simulate", supports_auto_skip=True):
            raise RuntimeError("boom")

    assert len(s3.put_calls) == 1
    marker = json.loads(s3.put_calls[0]["Body"])
    assert marker["status"] == "error"
    assert "RuntimeError" in marker["error"]


def test_phase_context_yields_skipped_when_marker_ok(s3):
    s3.seed("test-bucket", "2026-04-23", "simulate", {
        "phase": "simulate", "date": "2026-04-23", "status": "ok",
    })
    r = _make_registry(s3)
    n_puts_before = len(s3.put_calls)
    with r.phase("simulate", supports_auto_skip=True) as ctx:
        assert ctx.skipped is True
        assert ctx.skip_reason == "auto_skip_marker_ok"

    # Skipped phases don't overwrite their marker.
    assert len(s3.put_calls) == n_puts_before


def test_phase_context_swallows_marker_write_failure(s3, caplog):
    """A put_object failure must NOT fail the phase — the compute
    already succeeded. Loud warning only."""
    import logging
    r = _make_registry(s3)
    s3.put_object = MagicMock(side_effect=RuntimeError("s3 down"))

    with caplog.at_level(logging.WARNING, logger="pipeline_common"):
        with r.phase("simulate", supports_auto_skip=True):
            pass  # compute succeeds

    assert any("failed to write marker" in rec.getMessage() for rec in caplog.records)


def test_record_artifact_rejects_empty_string(s3):
    r = _make_registry(s3)
    with pytest.raises(ValueError):
        with r.phase("simulate") as ctx:
            ctx.record_artifact("")


def test_transient_s3_error_raises(s3):
    """NoSuchKey → absent (recompute). Other errors bubble so a transient
    network blip doesn't silently cause the pipeline to redo ~2h of work."""
    s3.get_object = MagicMock(side_effect=ClientError(
        {"Error": {"Code": "InternalError", "Message": "server down"}},
        "GetObject",
    ))
    r = _make_registry(s3)
    with pytest.raises(ClientError):
        r.should_run("simulate", supports_auto_skip=True)
