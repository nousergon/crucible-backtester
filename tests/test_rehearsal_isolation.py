"""S3-layer rehearsal isolation (alpha-engine-config-I12351).

rehearsal-2026-10-16-2's backtester and evaluator spots wrote ~60 live keys
(predictor/optimizer_gate/latest.json, backtest/latest/attestation.json, the
dated backtest reports and phase outputs, predictor/metrics/*,
config/metrics_history/*, regime/stratified_sortino/latest.json,
evaluator/diagnostics/*, decision_artifacts/_attribution_adequacy/history.jsonl).
These tests pin that a rehearsal process writes none of them at a live key,
reads its own writes back, still reads live inputs, and that a production
process is untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

import rehearsal_isolation as ri

BUCKET = "alpha-engine-research"
EXEC = "rehearsal-2026-10-16-2"
REHEARSAL_ENV = {"RUN_TOKEN": EXEC}
NS = f"rehearsal/{EXEC}/"

#: One representative key per live-write family measured on 2026-10-16.
LEAKED_KEYS = [
    "predictor/optimizer_gate/latest.json",
    "predictor/optimizer_gate/2026-10-15.json",
    "backtest/latest/attestation.json",
    "backtest/2026-10-15/metrics.json",
    "backtest/2026-10-15/.phases/simulate/sweep.parquet",
    "predictor/metrics/latest.json",
    "predictor/metrics/production_health.json",
    "config/metrics_history/latest.json",
    "regime/stratified_sortino/latest.json",
    "evaluator/diagnostics/2026-10-15/report.json",
    "decision_artifacts/_attribution_adequacy/history.jsonl",
]


@pytest.fixture
def s3(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    with mock_aws():
        boto3.setup_default_session(region_name="us-east-1")
        client = boto3.client("s3")
        client.create_bucket(Bucket=BUCKET)
        client.create_bucket(Bucket="alpha-engine-cost-telemetry")
        yield client
        ri.deactivate()
        boto3.DEFAULT_SESSION = None


def _keys(client, prefix=""):
    resp = client.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    return sorted(o["Key"] for o in resp.get("Contents") or [])


def test_marker():
    assert ri.rehearsal_name({"RUN_TOKEN": EXEC}) == EXEC
    assert ri.rehearsal_name({"RUN_TOKEN": f"arn:aws:states:us-east-1:1:execution:sf:{EXEC}"}) == EXEC
    assert ri.rehearsal_name({"AE_PIPELINE_ROLE": "rehearsal"}) == "rehearsal-manual-run"
    assert ri.rehearsal_name({"RUN_TOKEN": "weekly-2026-10-17"}) is None
    assert ri.rehearsal_name({}) is None


def test_production_is_untouched(s3):
    assert ri.activate(entrypoint="backtest.py", env={"RUN_TOKEN": "weekly-2026-10-17"}) is None
    assert ri.active() is None
    c = boto3.client("s3")
    for key in LEAKED_KEYS:
        c.put_object(Bucket=BUCKET, Key=key, Body=b"{}")
    assert _keys(s3) == sorted(LEAKED_KEYS)


def test_a_rehearsal_writes_no_live_key(s3):
    ctx = ri.activate(entrypoint="backtest.py", env=REHEARSAL_ENV)
    assert ctx is not None and ctx.prefix == NS
    c = boto3.client("s3")
    for key in LEAKED_KEYS:
        c.put_object(Bucket=BUCKET, Key=key, Body=b"{}")
    assert _keys(s3) == sorted(NS + k for k in LEAKED_KEYS)
    assert ctx.summary()["writes_redirected"] == len(LEAKED_KEYS)


def test_resource_api_upload_file_and_copy_are_isolated(s3, tmp_path):
    ri.activate(entrypoint="evaluate.py", env=REHEARSAL_ENV)
    src = tmp_path / "x.json"
    src.write_text("{}")
    boto3.client("s3").upload_file(str(src), BUCKET, "backtest/2026-10-15/up.json")
    boto3.resource("s3").Object(BUCKET, "backtest/2026-10-15/res.json").put(Body=b"{}")
    boto3.client("s3").copy_object(
        Bucket=BUCKET,
        Key="backtest/latest/attestation.json",
        CopySource={"Bucket": BUCKET, "Key": "backtest/2026-10-15/up.json"},
    )
    assert _keys(s3) == sorted(
        NS + k
        for k in (
            "backtest/2026-10-15/up.json",
            "backtest/2026-10-15/res.json",
            "backtest/latest/attestation.json",
        )
    )


def test_reads_see_the_rehearsals_own_writes_and_live_inputs(s3):
    s3.put_object(Bucket=BUCKET, Key="signals/2026-10-15/signals.json", Body=b'{"live": true}')
    s3.put_object(Bucket=BUCKET, Key="predictor/metrics/latest.json", Body=b'{"v": "live"}')
    ri.activate(entrypoint="backtest.py", env=REHEARSAL_ENV)
    c = boto3.client("s3")
    c.put_object(Bucket=BUCKET, Key="predictor/metrics/latest.json", Body=b'{"v": "rehearsal"}')
    got = json.loads(c.get_object(Bucket=BUCKET, Key="predictor/metrics/latest.json")["Body"].read())
    assert got == {"v": "rehearsal"}
    live = json.loads(c.get_object(Bucket=BUCKET, Key="signals/2026-10-15/signals.json")["Body"].read())
    assert live == {"live": True}
    # The live object is untouched.
    raw = json.loads(s3.get_object(Bucket=BUCKET, Key="predictor/metrics/latest.json")["Body"].read())
    assert raw == {"v": "live"}


def test_a_later_process_reads_what_an_earlier_stage_wrote(s3):
    # The Backtester stage's process wrote this; the evaluator spot reads it.
    s3.put_object(Bucket=BUCKET, Key=f"{NS}backtest/2026-10-15/metrics.json", Body=b'{"stage": 1}')
    s3.put_object(Bucket=BUCKET, Key="backtest/2026-10-15/metrics.json", Body=b'{"stage": "live"}')
    ri.activate(entrypoint="evaluate.py", env=REHEARSAL_ENV)
    body = boto3.client("s3").get_object(Bucket=BUCKET, Key="backtest/2026-10-15/metrics.json")["Body"]
    assert json.loads(body.read()) == {"stage": 1}


@pytest.mark.parametrize(
    "key",
    [
        f"backtest/2026-10-15/.phases/.rehearsal/{EXEC}/simulate.json",
        f"health/.rehearsal/{EXEC}/backtester.json",
        f"health/weekly_preflight_on_spot/2026-10-15/{EXEC}.json",
        f"health/exec_code_pin/{EXEC}/backtester.json",
        "_flow_doctor/heartbeat/backtester-rehearsal/x.json",
        "decision_artifacts/_cost_raw/2026-10-15/backtester.jsonl",
        f"{NS}already/namespaced.json",
    ],
)
def test_already_isolated_and_telemetry_keys_pass_through(s3, key):
    ri.activate(entrypoint="backtest.py", env=REHEARSAL_ENV)
    boto3.client("s3").put_object(Bucket=BUCKET, Key=key, Body=b"{}")
    assert _keys(s3) == [key]


def test_other_buckets_pass_through(s3):
    ri.activate(entrypoint="backtest.py", env=REHEARSAL_ENV)
    c = boto3.client("s3")
    c.put_object(Bucket="alpha-engine-cost-telemetry", Key="x/y.json", Body=b"{}")
    resp = c.list_objects_v2(Bucket="alpha-engine-cost-telemetry")
    assert [o["Key"] for o in resp["Contents"]] == ["x/y.json"]


def test_delete_objects_is_isolated(s3):
    s3.put_object(Bucket=BUCKET, Key="backtest/2026-10-15/live.json", Body=b"{}")
    ri.activate(entrypoint="backtest.py", env=REHEARSAL_ENV)
    c = boto3.client("s3")
    c.delete_objects(Bucket=BUCKET, Delete={"Objects": [{"Key": "backtest/2026-10-15/live.json"}]})
    c.delete_object(Bucket=BUCKET, Key="backtest/2026-10-15/live.json")
    assert "backtest/2026-10-15/live.json" in _keys(s3)


def test_an_unknown_write_shaped_operation_is_refused():
    ctx = ri.RehearsalContext(EXEC, entrypoint="t")
    ri._ACTIVE = ctx  # noqa: SLF001
    try:
        model = type("M", (), {"name": "PutBucketPolicy"})()
        with pytest.raises(ri.RehearsalLiveWriteRefused):
            ctx._hook({"Bucket": BUCKET, "Policy": "{}"}, model)  # noqa: SLF001
        # A bucket-level read passes.
        ctx._hook({"Bucket": BUCKET}, type("M", (), {"name": "GetBucketVersioning"})())  # noqa: SLF001
    finally:
        ri._ACTIVE = None  # noqa: SLF001


def test_activate_is_idempotent(s3):
    a = ri.activate(entrypoint="backtest.py", env=REHEARSAL_ENV)
    assert ri.activate(entrypoint="backtest.py", env=REHEARSAL_ENV) is a


def test_both_spot_entrypoints_activate_right_after_freeze():
    root = Path(__file__).resolve().parents[1]
    for name in ("backtest.py", "evaluate.py"):
        body = (root / name).read_text()
        freeze = body.index(f'freeze_if_rehearsal(args, entrypoint="{name}")')
        iso = body.index(f'_isolate_rehearsal(entrypoint="{name}")')
        assert 0 < iso - freeze < 400, name
