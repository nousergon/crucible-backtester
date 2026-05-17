"""Unit tests for PR 2 — bitemporal config archive (point-in-time
discipline, ROADMAP L2371 / Backtester Phase 3; plan
``alpha-engine-docs/private/pit-discipline-260515.md`` §D3).

Locks the PIT invariants for optimizer-config resolution:
  - knowledge-time ≤ decision-time; tie-break later-run-id-wins
  - **no-future-fallback** — no eligible snapshot ⇒ genesis FACTORY_DEFAULTS,
    NEVER a later snapshot (the central trap)
  - the changelog index write is best-effort (apply() never fails on it)
  - flag OFF ⇒ read_current_params path byte-unchanged

S3 is faked with an in-memory key→bytes store (the get/put round-trip is
the whole point of the changelog RMW, so MagicMock return-values aren't
enough).
"""

from __future__ import annotations

import datetime as dt
import json

import pytest
from botocore.exceptions import ClientError

from optimizer import config_archive as ca
from optimizer import executor_optimizer, research_optimizer, scanner_optimizer


class _FakeS3:
    """Minimal in-memory S3: get_object / put_object over a dict."""

    def __init__(self):
        self.store: dict[str, bytes] = {}

    def put_object(self, *, Bucket, Key, Body, ContentType=None):
        self.store[Key] = Body.encode() if isinstance(Body, str) else Body

    def get_object(self, *, Bucket, Key):
        if Key not in self.store:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey"}}, "GetObject"
            )
        return {"Body": _Body(self.store[Key])}


class _Body:
    def __init__(self, b):
        self._b = b

    def read(self):
        return self._b


# ── as_of_date_from_config ────────────────────────────────────────────────

def test_as_of_none_when_walk_forward_off():
    assert ca.as_of_date_from_config({}) is None
    assert ca.as_of_date_from_config({"walk_forward": False}) is None


def test_as_of_uses_run_date_when_on():
    got = ca.as_of_date_from_config(
        {"walk_forward": True, "_run_date": "2026-05-10"}
    )
    assert got == dt.date(2026, 5, 10)


def test_as_of_tolerates_smoke_label_and_defaults_today():
    # Uninterpretable label → today (logged), never a crash mid-backtest.
    got = ca.as_of_date_from_config(
        {"walk_forward": True, "_run_date": ".smoke/nonsense"}
    )
    assert got == dt.date.today()
    # Missing _run_date → today.
    assert ca.as_of_date_from_config({"walk_forward": True}) == dt.date.today()


# ── record_apply: bitemporal index RMW ────────────────────────────────────

def test_record_apply_appends_entry(monkeypatch):
    s3 = _FakeS3()
    ok = ca.record_apply(
        "b", "executor_params",
        history_key="config/executor_params_history/2605101437_eval.json",
        knowledge_date="2026-05-10", run_id="2605101437", s3_client=s3,
    )
    assert ok is True
    entries = json.loads(s3.store[ca.CHANGELOG_KEY])
    assert len(entries) == 1
    e = entries[0]
    assert e["config_type"] == "executor_params"
    assert e["knowledge_date"] == "2026-05-10"
    assert e["effective_date"] == "2026-05-10"  # defaults to knowledge
    assert e["run_id"] == "2605101437"
    # A second apply appends (RMW), does not overwrite.
    ca.record_apply(
        "b", "research_params",
        history_key="config/research_params_history/2605171000_eval.json",
        knowledge_date="2026-05-17", run_id="2605171000", s3_client=s3,
    )
    assert len(json.loads(s3.store[ca.CHANGELOG_KEY])) == 2


def test_record_apply_best_effort_swallows_errors(caplog):
    class _Broken:
        def get_object(self, **kw):
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "g")

        def put_object(self, **kw):
            raise RuntimeError("S3 down")

    with caplog.at_level("WARNING"):
        ok = ca.record_apply(
            "b", "executor_params", history_key="k",
            knowledge_date="2026-05-10", run_id="r", s3_client=_Broken(),
        )
    assert ok is False  # never raises — apply() already durable
    assert any("best-effort" in r.message for r in caplog.records)


def test_record_apply_rejects_unknown_type():
    s3 = _FakeS3()
    assert ca.record_apply(
        "b", "predictor_params", history_key="k",
        knowledge_date="2026-05-10", run_id="r", s3_client=s3,
    ) is False


# ── resolve_as_of: the cardinal PIT rule + no-future-fallback ─────────────

def _seed(s3, *entries):
    s3.put_object(Bucket="b", Key=ca.CHANGELOG_KEY, Body=json.dumps(list(entries)))


def _entry(ct, kd, run_id, payload, s3):
    hk = f"config/{ct}_history/{run_id}.json"
    s3.put_object(Bucket="b", Key=hk, Body=json.dumps(payload))
    return {"config_type": ct, "knowledge_date": kd, "effective_date": kd,
            "history_key": hk, "run_id": run_id}


def test_resolve_picks_latest_knowledge_le_as_of():
    s3 = _FakeS3()
    e1 = _entry("executor_params", "2026-05-01", "2605010000", {"v": 1}, s3)
    e2 = _entry("executor_params", "2026-05-08", "2605080000", {"v": 2}, s3)
    e3 = _entry("executor_params", "2026-05-15", "2605150000", {"v": 3}, s3)
    _seed(s3, e1, e2, e3)

    # as_of between e2 and e3 → e2 (latest knowledge ≤ as_of), NOT e3.
    got = ca.resolve_as_of("b", "executor_params", dt.date(2026, 5, 10), s3_client=s3)
    assert got == {"v": 2}
    # same-day allowed (≤, not <)
    got = ca.resolve_as_of("b", "executor_params", dt.date(2026, 5, 15), s3_client=s3)
    assert got == {"v": 3}


def test_resolve_no_future_fallback_returns_none():
    """The central trap: every snapshot is AFTER the decision date → None
    (caller uses genesis), never the nearest/earliest-future snapshot."""
    s3 = _FakeS3()
    e = _entry("executor_params", "2026-06-01", "2606010000", {"v": 9}, s3)
    _seed(s3, e)
    assert ca.resolve_as_of(
        "b", "executor_params", dt.date(2026, 5, 10), s3_client=s3
    ) is None


def test_resolve_unreadable_snapshot_is_not_future_fallback(caplog):
    """Index points at a snapshot we cannot fetch → None (genesis), NOT a
    fall-forward to a newer readable snapshot."""
    s3 = _FakeS3()
    s3.put_object(Bucket="b", Key=ca.CHANGELOG_KEY, Body=json.dumps([
        {"config_type": "executor_params", "knowledge_date": "2026-05-01",
         "effective_date": "2026-05-01",
         "history_key": "config/missing.json", "run_id": "2605010000"},
    ]))
    with caplog.at_level("WARNING"):
        got = ca.resolve_as_of(
            "b", "executor_params", dt.date(2026, 5, 10), s3_client=s3
        )
    assert got is None
    assert any("NOT future-fallback" in r.message for r in caplog.records)


def test_resolve_filters_by_config_type():
    s3 = _FakeS3()
    e_ex = _entry("executor_params", "2026-05-01", "2605010000", {"who": "ex"}, s3)
    e_rs = _entry("research_params", "2026-05-01", "2605010001", {"who": "rs"}, s3)
    _seed(s3, e_ex, e_rs)
    assert ca.resolve_as_of(
        "b", "research_params", dt.date(2026, 5, 10), s3_client=s3
    ) == {"who": "rs"}


# ── optimizer read_params_as_of: genesis fallback + contract parity ───────

def test_executor_read_params_as_of_genesis_on_miss(monkeypatch):
    # read_params_as_of does `from optimizer.config_archive import
    # resolve_as_of` at call time, so patching it on the module resolves.
    monkeypatch.setattr(
        "optimizer.config_archive.resolve_as_of", lambda *a, **k: None
    )
    out = executor_optimizer.read_params_as_of("b", dt.date(2026, 5, 10))
    assert out == executor_optimizer.FACTORY_DEFAULTS.copy()


@pytest.mark.parametrize("mod", [research_optimizer, scanner_optimizer])
def test_research_scanner_as_of_genesis_on_miss(monkeypatch, mod):
    monkeypatch.setattr(
        "optimizer.config_archive.resolve_as_of", lambda *a, **k: None
    )
    out = mod.read_params_as_of("b", dt.date(2026, 5, 10))
    assert out == mod.FACTORY_DEFAULTS.copy()


def test_research_as_of_merges_snapshot_over_defaults(monkeypatch):
    key = next(iter(research_optimizer.FACTORY_DEFAULTS))
    snap = {key: research_optimizer.FACTORY_DEFAULTS[key], "updated_at": "2026-05-08"}
    monkeypatch.setattr(
        "optimizer.config_archive.resolve_as_of", lambda *a, **k: snap
    )
    out = research_optimizer.read_params_as_of("b", dt.date(2026, 5, 10))
    # Same shape as read_current_params: full defaults overlaid with snapshot.
    assert set(out) >= set(research_optimizer.FACTORY_DEFAULTS)


# ── dispatcher: flag OFF == legacy path ───────────────────────────────────

def test_pit_or_current_off_calls_read_current(monkeypatch):
    calls = {"current": 0, "as_of": 0}

    class _Mod:
        @staticmethod
        def read_current_params(bucket):
            calls["current"] += 1
            return {"x": 1}

        @staticmethod
        def read_params_as_of(bucket, ao):
            calls["as_of"] += 1
            return {"x": 2}

    # OFF (default) → read_current_params, never as_of.
    assert ca.read_params_pit_or_current(_Mod, "b", {}) == {"x": 1}
    assert calls == {"current": 1, "as_of": 0}

    # ON → read_params_as_of only.
    assert ca.read_params_pit_or_current(
        _Mod, "b", {"walk_forward": True, "_run_date": "2026-05-10"}
    ) == {"x": 2}
    assert calls == {"current": 1, "as_of": 1}
