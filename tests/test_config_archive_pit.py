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
    """Minimal in-memory S3: get/put/list over a dict.

    ``last_modified`` is optional per key: real S3 always reports it, but the
    pre-I11503 tests below pin the changelog semantics alone, so a key with
    no recorded write time is listed without one (key-date only).
    """

    def __init__(self, page_size: int = 1000):
        self.store: dict[str, bytes] = {}
        self.last_modified: dict[str, dt.datetime] = {}
        self.page_size = page_size
        self.list_calls = 0

    def put_object(self, *, Bucket, Key, Body, ContentType=None, LastModified=None):
        self.store[Key] = Body.encode() if isinstance(Body, str) else Body
        if LastModified is not None:
            self.last_modified[Key] = LastModified

    def list_objects_v2(self, *, Bucket, Prefix="", ContinuationToken=None):
        self.list_calls += 1
        keys = sorted(k for k in self.store if k.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = keys[start:start + self.page_size]
        contents = []
        for k in page:
            obj = {"Key": k, "Size": len(self.store[k])}
            if k in self.last_modified:
                obj["LastModified"] = self.last_modified[k]
            contents.append(obj)
        resp = {"Contents": contents, "IsTruncated": start + self.page_size < len(keys)}
        if resp["IsTruncated"]:
            resp["NextContinuationToken"] = str(start + self.page_size)
        return resp

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


# ── alpha-engine-config-I11503: resolve from the snapshots actually on S3 ──
#
# The assembler (sole live writer since cutover) writes the live key and
# config/{type}_history/{YYMMDDHHMM}.json and never indexes the changelog,
# so the 2026-09-23 rehearsal found no executor_params snapshot ≤ run_date
# and simulated genesis defaults (min_score 70 / max_position_pct 5%) while
# live ran 75 / 10%. Layout below is the live bucket's, read 2026-09-24.

def _utc(y, m, d, hh=12, mm=0):
    return dt.datetime(y, m, d, hh, mm, tzinfo=dt.timezone.utc)


def _seed_assembler_layout(s3):
    """Mirror of s3://alpha-engine-research/config/executor_params* as read
    on 2026-09-24 (trimmed): legacy dated keys, suffixed operator snapshot,
    YYMMDDHHMM run ids, latest.json sidecar, live key, _previous, shadow
    history and the assembler's own audit prefix."""
    def put(key, payload, when):
        s3.put_object(Bucket="b", Key=key, Body=json.dumps(payload), LastModified=when)

    put("config/executor_params_history/2026-05-09.json",
        {"min_score": 75, "max_position_pct": 0.1, "updated_at": "2026-05-09"},
        _utc(2026, 5, 9, 16, 23))
    put("config/executor_params_history/2026-05-20T2349Z-pre-min-score-reset.json",
        {"min_score": 75, "max_position_pct": 0.1, "updated_at": "2026-05-07"},
        _utc(2026, 5, 20, 18, 36))
    put("config/executor_params_history/2609121449.json",
        {"min_score": 75, "max_position_pct": 0.1, "atr_multiplier": 2.0,
         "updated_at": "2026-09-11", "marker": "0912"},
        _utc(2026, 9, 12, 14, 49))
    put("config/executor_params_history/2609191405.json",
        {"min_score": 75, "max_position_pct": 0.1, "atr_multiplier": 2.0,
         "updated_at": "2026-09-18", "marker": "0919"},
        _utc(2026, 9, 19, 14, 5))
    put("config/executor_params_history/2609240336.json",
        {"min_score": 80, "max_position_pct": 0.1, "updated_at": "2026-09-23",
         "marker": "0924"},
        _utc(2026, 9, 24, 3, 36))
    put("config/executor_params_history/latest.json",
        {"min_score": 80, "marker": "latest"}, _utc(2026, 9, 24, 3, 36))
    put("config/executor_params.json",
        {"min_score": 80, "marker": "live"}, _utc(2026, 9, 24, 3, 36))
    put("config/executor_params_previous.json",
        {"min_score": 1, "marker": "previous"}, _utc(2026, 9, 24, 3, 36))
    put("config/executor_params_shadow_history/2605180030.json",
        {"min_score": 2, "marker": "shadow"}, _utc(2026, 5, 18, 0, 30))
    put("config/executor_params/assembled/2609191405.json",
        {"marker": "audit"}, _utc(2026, 9, 19, 14, 5))


def test_resolves_assembler_history_with_no_changelog():
    """The rehearsal case: run_date 2026-09-23 must load the 09-19 snapshot
    (live params), not genesis, and not the 09-24 write (look-ahead)."""
    s3 = _FakeS3()
    _seed_assembler_layout(s3)
    got = ca.resolve_as_of("b", "executor_params", dt.date(2026, 9, 23), s3_client=s3)
    assert got is not None
    assert got["marker"] == "0919"
    assert got["min_score"] == 75 and got["max_position_pct"] == 0.1


def test_live_key_resolves_once_written_on_or_before_as_of():
    s3 = _FakeS3()
    _seed_assembler_layout(s3)
    got = ca.resolve_as_of("b", "executor_params", dt.date(2026, 9, 24), s3_client=s3)
    # 09-24 has two writes (history + live, same body); either is correct,
    # neither may be latest.json/_previous/shadow/audit.
    assert got["marker"] in ("0924", "live")


def test_sidecars_and_foreign_prefixes_never_resolve():
    s3 = _FakeS3()
    _seed_assembler_layout(s3)
    for as_of in (dt.date(2026, 5, 18), dt.date(2026, 9, 19), dt.date(2026, 12, 31)):
        got = ca.resolve_as_of("b", "executor_params", as_of, s3_client=s3)
        assert got is not None
        assert got.get("marker") not in ("latest", "previous", "shadow", "audit")


def test_last_modified_is_a_knowledge_floor():
    """A key NAMED for 2026-05-01 but written 2026-06-01 was not knowable on
    2026-05-10 — the name cannot backdate it (no look-ahead)."""
    s3 = _FakeS3()
    s3.put_object(Bucket="b", Key="config/executor_params_history/2026-05-01.json",
                  Body=json.dumps({"v": "backdated"}), LastModified=_utc(2026, 6, 1))
    assert ca.resolve_as_of(
        "b", "executor_params", dt.date(2026, 5, 10), s3_client=s3,
    ) is None
    assert ca.resolve_as_of(
        "b", "executor_params", dt.date(2026, 6, 1), s3_client=s3,
    ) == {"v": "backdated"}


def test_live_key_only_resolves_research_params():
    """research_params has a live key (2026-05-02) and no history prefix at
    all — the live object is itself a snapshot knowable since its write."""
    s3 = _FakeS3()
    s3.put_object(Bucket="b", Key="config/research_params.json",
                  Body=json.dumps({"who": "live-research"}), LastModified=_utc(2026, 5, 2))
    assert ca.resolve_as_of(
        "b", "research_params", dt.date(2026, 9, 23), s3_client=s3,
    ) == {"who": "live-research"}
    assert ca.resolve_as_of(
        "b", "research_params", dt.date(2026, 5, 1), s3_client=s3,
    ) is None


def test_listing_is_paginated():
    s3 = _FakeS3(page_size=2)
    _seed_assembler_layout(s3)
    got = ca.resolve_as_of("b", "executor_params", dt.date(2026, 9, 23), s3_client=s3)
    assert got["marker"] == "0919"
    assert s3.list_calls > 1


def test_changelog_and_history_dedupe_to_later_knowledge():
    s3 = _FakeS3()
    key = "config/executor_params_history/2605010000.json"
    s3.put_object(Bucket="b", Key=key, Body=json.dumps({"v": 1}),
                  LastModified=_utc(2026, 5, 3))
    _seed(s3, {"config_type": "executor_params", "knowledge_date": "2026-05-01",
               "effective_date": "2026-05-01", "history_key": key, "run_id": "2605010000"})
    # The index says 05-01, the object was written 05-03: 05-02 must not see it.
    assert ca.resolve_as_of("b", "executor_params", dt.date(2026, 5, 2), s3_client=s3) is None
    assert ca.resolve_as_of("b", "executor_params", dt.date(2026, 5, 3), s3_client=s3) == {"v": 1}


def test_listing_failure_falls_back_to_changelog_and_names_it(caplog):
    class _NoList(_FakeS3):
        def list_objects_v2(self, **kw):
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "ListObjectsV2")

    s3 = _NoList()
    e = _entry("executor_params", "2026-05-01", "2605010000", {"v": 1}, s3)
    _seed(s3, e)
    assert ca.resolve_as_of("b", "executor_params", dt.date(2026, 5, 10), s3_client=s3) == {"v": 1}

    empty = _NoList()
    with caplog.at_level("WARNING"):
        assert ca.resolve_as_of(
            "b", "executor_params", dt.date(2026, 5, 10), s3_client=empty,
        ) is None
    assert any("cannot list" in r.message for r in caplog.records)


@pytest.mark.parametrize("key,expected", [
    ("config/x_history/2609191405.json", dt.date(2026, 9, 19)),
    ("config/x_history/2026-05-09.json", dt.date(2026, 5, 9)),
    ("config/x_history/2026-05-20T2349Z-pre-min-score-reset.json", dt.date(2026, 5, 20)),
    ("config/x_history/latest.json", None),
    ("config/x_history/2613991405.json", None),  # month 13 → not a date
])
def test_key_date(key, expected):
    assert ca._key_date(key) == expected


# ── a genesis fallback is a DEGRADED stage, not an INFO line ──────────────

def test_genesis_fallback_marks_config_degraded(monkeypatch, caplog):
    s3 = _FakeS3()  # nothing on S3 at all
    monkeypatch.setattr(ca, "_client", lambda _c: s3)
    config = {"walk_forward": True, "_run_date": "2026-09-23"}
    with caplog.at_level("WARNING"):
        out = ca.read_params_pit_or_current(executor_optimizer, "b", config)
    assert out == executor_optimizer.FACTORY_DEFAULTS.copy()
    warnings = ca.pit_degraded_warnings(config)
    assert len(warnings) == 1
    assert "executor_params" in warnings[0] and "GENESIS" in warnings[0]
    assert any("DEGRADED" in r.message for r in caplog.records)

    # The stage's health status is derived from these warnings.
    from nousergon_lib.health import Deliverable, derive_status
    assert derive_status(
        [Deliverable(name="backtest_run", required=True, produced=True)],
        warnings=warnings,
    ) == "degraded"

    # Idempotent across repeated reads in one run (two call sites).
    ca.read_params_pit_or_current(executor_optimizer, "b", config)
    assert len(ca.pit_degraded_warnings(config)) == 1


def test_resolved_snapshot_leaves_stage_clean(monkeypatch):
    s3 = _FakeS3()
    _seed_assembler_layout(s3)
    monkeypatch.setattr(ca, "_client", lambda _c: s3)
    config = {"walk_forward": True, "_run_date": "2026-09-23"}
    out = ca.read_params_pit_or_current(executor_optimizer, "b", config)
    assert out["min_score"] == 75 and out["max_position_pct"] == 0.1
    assert ca.pit_degraded_warnings(config) == []


def test_unreadable_snapshot_is_degraded_too(monkeypatch):
    s3 = _FakeS3()
    s3.put_object(Bucket="b", Key=ca.CHANGELOG_KEY, Body=json.dumps([
        {"config_type": "executor_params", "knowledge_date": "2026-05-01",
         "effective_date": "2026-05-01",
         "history_key": "config/missing.json", "run_id": "2605010000"},
    ]))
    monkeypatch.setattr(ca, "_client", lambda _c: s3)
    config = {"walk_forward": True, "_run_date": "2026-05-10"}
    ca.read_params_pit_or_current(executor_optimizer, "b", config)
    assert "unreadable" in ca.pit_degraded_warnings(config)[0]


def test_walk_forward_off_is_never_degraded():
    config = {}

    class _Mod:
        @staticmethod
        def read_current_params(bucket):
            return {"x": 1}

    ca.read_params_pit_or_current(_Mod, "b", config)
    assert ca.pit_degraded_warnings(config) == []
