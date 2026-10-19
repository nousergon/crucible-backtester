"""Rehearsal isolation through a real botocore client, with no moto and no
network (alpha-engine-config-I12351).

A recorder registered on ``before-parameter-build.s3`` AFTER the isolation
hook sees the parameters the call is finally serialized with, so these tests
see exactly the key a rehearsal would write or read; botocore's Stubber only
supplies the responses (no network). ``tests/test_rehearsal_isolation.py``
covers the same behaviour end to end on moto where it is installed.
"""

from __future__ import annotations

import pytest

import rehearsal_isolation as ri

boto3 = pytest.importorskip("boto3")
stub = pytest.importorskip("botocore.stub")

BUCKET = "alpha-engine-research"
EXEC = "rehearsal-2026-10-16-2"
NS = f"rehearsal/{EXEC}/"


@pytest.fixture
def ctx():
    c = ri.RehearsalContext(EXEC, entrypoint="test")
    ri._ACTIVE = c  # noqa: SLF001
    yield c
    ri._ACTIVE = None  # noqa: SLF001


class _Client:
    """A stubbed S3 client with the isolation hook, recording sent params."""

    def __init__(self, ctx):
        self.client = ctx.attach(
            boto3.session.Session().client(
                "s3",
                region_name="us-east-1",
                aws_access_key_id="testing",
                aws_secret_access_key="testing",  # noqa: S106 — stubbed, never sent
            )
        )
        self.sent: list[tuple[str, str, str]] = []
        self.client.meta.events.register(
            "before-parameter-build.s3", self._record, unique_id="test-recorder"
        )
        self.stubber = stub.Stubber(self.client)
        self.stubber.activate()

    def _record(self, params, model, **_):
        self.sent.append((model.name, params.get("Bucket", ""), params.get("Key", "")))

    def call(self, op, **kwargs):
        self.stubber.add_response(op, {})
        getattr(self.client, op)(**kwargs)
        return self.sent[-1]


def _client(ctx):
    return _Client(ctx)


@pytest.mark.parametrize(
    "key",
    [
        "predictor/optimizer_gate/latest.json",
        "backtest/latest/attestation.json",
        "backtest/2026-10-15/.phases/simulate/sweep.parquet",
        "predictor/metrics/latest.json",
        "config/metrics_history/latest.json",
        "regime/stratified_sortino/latest.json",
        "evaluator/diagnostics/2026-10-15/report.json",
        "decision_artifacts/_attribution_adequacy/history.jsonl",
        "backtest/cumulative_trial_count.json",
        "health/backtester.json",
    ],
)
def test_a_live_write_is_rekeyed(ctx, key):
    c = _client(ctx)
    assert c.call("put_object", Bucket=BUCKET, Key=key, Body=b"{}") == ("PutObject", BUCKET, NS + key)
    assert ctx.is_overlaid(BUCKET, key)


@pytest.mark.parametrize(
    "key",
    [
        f"backtest/2026-10-15/.phases/.rehearsal/{EXEC}/simulate.json",
        f"health/weekly_preflight_on_spot/2026-10-15/{EXEC}.json",
        "_flow_doctor/heartbeat/backtester-rehearsal/x.json",
        "decision_artifacts/_cost_raw/2026-10-15/backtester.jsonl",
        f"{NS}already.json",
    ],
)
def test_isolated_and_telemetry_keys_pass_through(ctx, key):
    c = _client(ctx)
    assert c.call("put_object", Bucket=BUCKET, Key=key, Body=b"{}") == ("PutObject", BUCKET, key)


def test_reads_follow_the_overlay(ctx):
    c = _client(ctx)
    k = "predictor/metrics/latest.json"
    assert c.call("get_object", Bucket=BUCKET, Key=k)[2] == k  # not yet written: live
    c.call("put_object", Bucket=BUCKET, Key=k, Body=b"{}")
    assert c.call("get_object", Bucket=BUCKET, Key=k)[2] == NS + k
    assert c.call("head_object", Bucket=BUCKET, Key=k)[2] == NS + k
    live = "signals/2026-10-15/signals.json"
    assert c.call("get_object", Bucket=BUCKET, Key=live)[2] == live


def test_other_buckets_pass_through(ctx):
    c = _client(ctx)
    sent = c.call("put_object", Bucket="alpha-engine-cost-telemetry", Key="x.json", Body=b"{}")
    assert sent == ("PutObject", "alpha-engine-cost-telemetry", "x.json")


def test_an_inactive_context_changes_nothing(ctx):
    c = _client(ctx)
    ri._ACTIVE = None  # noqa: SLF001 — a cached client outside the rehearsal
    k = "predictor/metrics/latest.json"
    assert c.call("put_object", Bucket=BUCKET, Key=k, Body=b"{}") == ("PutObject", BUCKET, k)


def test_delete_never_reaches_a_live_key(ctx):
    c = _client(ctx)
    k = "backtest/2026-10-15/metrics.json"
    assert c.call("delete_object", Bucket=BUCKET, Key=k)[2] == NS + k


def test_production_activate_touches_nothing():
    assert ri.activate(entrypoint="backtest.py", env={"RUN_TOKEN": "weekly-2026-10-17"}) is None
    assert ri.activate(entrypoint="backtest.py", env={}) is None
    assert ri.active() is None
