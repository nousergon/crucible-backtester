"""L4471 L2 — within-run sim checkpoint/resume correctness.

Focuses on the delicate parts: the input-fingerprint invalidation guard (the
load-bearing correctness anchor — a stale checkpoint must NOT resume) and the
pickle round-trip preserving exact types (determinism). Plus a source-grep
contract that the simulate loop wires resume/checkpoint/clear so it can't be
silently reverted.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from store import sim_checkpoint as sc

_REPO = Path(__file__).resolve().parent.parent


class _FakeS3:
    """In-memory S3 stand-in (bytes store) for put/get/delete."""

    def __init__(self):
        self.store: dict = {}

    def put_object(self, Bucket, Key, Body):
        self.store[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.store:
            raise KeyError("NoSuchKey")  # mimics botocore ClientError path (caught)
        return {"Body": _Body(self.store[(Bucket, Key)])}

    def delete_object(self, Bucket, Key):
        self.store.pop((Bucket, Key), None)


class _Body:
    def __init__(self, b):
        self._b = b

    def read(self):
        return self._b


# ── Fingerprint (invalidation key) ──────────────────────────────────────────

def test_fingerprint_deterministic():
    cfg = {"min_score": 70, "atr_multiplier": 2.0, "init_cash": 1e6}
    dates = ["2026-05-27", "2026-05-28", "2026-05-29"]
    assert sc.compute_fingerprint(cfg, dates) == sc.compute_fingerprint(dict(cfg), list(dates))


def test_fingerprint_changes_on_config():
    dates = ["2026-05-29"]
    a = sc.compute_fingerprint({"min_score": 70}, dates)
    b = sc.compute_fingerprint({"min_score": 57}, dates)  # the weekly optimizer rewrite
    assert a != b, "a change in a sim-relevant executor param MUST invalidate the checkpoint"


def test_fingerprint_changes_on_dates():
    cfg = {"min_score": 70}
    assert sc.compute_fingerprint(cfg, ["2026-05-29"]) != sc.compute_fingerprint(cfg, ["2026-05-29", "2026-05-30"])


def test_fingerprint_changes_on_code_version(monkeypatch):
    cfg, dates = {"min_score": 70}, ["2026-05-29"]
    before = sc.compute_fingerprint(cfg, dates)
    monkeypatch.setattr(sc, "SIM_CODE_VERSION", "999")
    assert sc.compute_fingerprint(cfg, dates) != before, "a sim-code bump MUST invalidate checkpoints"


# ── Save / load round-trip + invalidation ───────────────────────────────────

def _save(s3, fp, idx=2, last="2026-05-29"):
    sc.save_checkpoint(
        bucket="b", run_date="2026-05-29", fingerprint=fp, idx=idx, last_date=last,
        sim_state={"cash": 950000.0, "positions": {"AAPL": {"shares": 100, "entry_date": dt.date(2026, 5, 20)}}, "peak_nav": 1.01e6},
        all_orders=[{"ticker": "AAPL", "action": "ENTER", "date": dt.date(2026, 5, 20)}],
        dates_simulated=3, skip_reasons={"no_signals": 1}, rejected_ticker_counter={"TSM": 4},
        s3_client=s3,
    )


def test_roundtrip_preserves_types():
    s3 = _FakeS3()
    _save(s3, "fp1")
    out = sc.load_checkpoint(bucket="b", run_date="2026-05-29", fingerprint="fp1", s3_client=s3)
    assert out is not None
    assert out["idx"] == 2 and out["dates_simulated"] == 3
    # pickle (not JSON) must preserve the date type for determinism on resume.
    assert out["sim_state"]["positions"]["AAPL"]["entry_date"] == dt.date(2026, 5, 20)
    assert out["all_orders"][0]["date"] == dt.date(2026, 5, 20)
    assert out["rejected_ticker_counter"] == {"TSM": 4}


def test_fingerprint_mismatch_returns_none():
    s3 = _FakeS3()
    _save(s3, "fp1")
    assert sc.load_checkpoint(bucket="b", run_date="2026-05-29", fingerprint="DIFFERENT", s3_client=s3) is None


def test_absent_returns_none():
    assert sc.load_checkpoint(bucket="b", run_date="2026-05-29", fingerprint="fp1", s3_client=_FakeS3()) is None


def test_schema_mismatch_returns_none(monkeypatch):
    s3 = _FakeS3()
    _save(s3, "fp1")
    monkeypatch.setattr(sc, "_SCHEMA_VERSION", 999)
    assert sc.load_checkpoint(bucket="b", run_date="2026-05-29", fingerprint="fp1", s3_client=s3) is None


def test_clear_then_absent():
    s3 = _FakeS3()
    _save(s3, "fp1")
    sc.clear_checkpoint(bucket="b", run_date="2026-05-29", s3_client=s3)
    assert sc.load_checkpoint(bucket="b", run_date="2026-05-29", fingerprint="fp1", s3_client=s3) is None


def test_save_never_raises_on_s3_failure():
    class _BrokenS3:
        def put_object(self, **k):
            raise RuntimeError("s3 down")
    # best-effort: a checkpoint write failure must NOT abort the sim
    _save(_BrokenS3(), "fp1")  # no raise


# ── Loop-wiring contract (guards against silent revert) ─────────────────────

def test_simulate_loop_wires_resume_checkpoint_clear():
    src = (_REPO / "backtest.py").read_text()
    assert "resilience_ctx" in src
    assert "load_checkpoint(" in src and "save_checkpoint(" in src and "clear_checkpoint(" in src
    assert "RESUMING from checkpoint" in src
    assert "SLOW DATE" in src and "PROJECTED OVERRUN" in src  # L1 instrumentation
    # checkpoint/resume must be gated (not run in param sweeps)
    assert 'resilience_ctx.get("enabled")' in src
