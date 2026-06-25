"""Backfill-date threading for the 5 produce_artifact / apply chains (config#1017).

`today_iso()` trading-day-normalizes the LIVE path (config#886), but an explicit
`--date` backfill still stamped today's trading day rather than the backfill
date — the recommendation artifact partition would not match the backfilled
`assemble` read. config#1017 threads an explicit `run_date` through each
optimizer's `produce_artifact()` / `apply()`. These pins prove:

  1. an explicit run_date keys the artifact (S3 Key partition + body.run_date)
     to the BACKFILL date, not now();
  2. run_date=None preserves the live default (today_iso()).

All S3 calls mocked. Covers all 5 optimizers that produce a RecommendationArtifact:
trigger / predictor_sizing / barrier_sizing / stance_sizing / executor.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from optimizer.assembler import set_cutover_enabled

BACKFILL_DATE = "2026-05-29"  # a Friday trading day


@pytest.fixture(autouse=True)
def _reset_cutover_flag():
    set_cutover_enabled(False)
    yield
    set_cutover_enabled(False)


# Each entry: (module path, ok-status result for that optimizer).
# The result only needs the fields each produce_artifact reads for an "ok" path.
_OPTIMIZERS = {
    "trigger_optimizer": {
        "status": "ok",
        "disabled_triggers": ["pullback"],
        "total_evaluated": 4,
        "min_trades_threshold": 50,
    },
    "predictor_sizing_optimizer": {
        "status": "ok",
        "recommendation": "enable",
        "use_p_up_sizing": True,
    },
    "barrier_sizing_optimizer": {
        "status": "ok",
        "recommendation": "enable",
    },
    "stance_sizing_optimizer": {
        "status": "ok",
        "recommendation": "enable",
    },
    "executor_optimizer": {
        "status": "ok",
        "fit_target": "sharpe_legacy",
        "recommended_params": {"max_position_pct": 0.05},
    },
}


def _produce_artifact(module_name: str):
    mod = __import__(f"optimizer.{module_name}", fromlist=["produce_artifact"])
    return mod.produce_artifact


@pytest.mark.parametrize("module_name", sorted(_OPTIMIZERS))
def test_explicit_run_date_keys_artifact_to_backfill_date(module_name):
    result = _OPTIMIZERS[module_name]
    with patch("optimizer.recommendation_artifact.boto3") as mock_boto3:
        s3 = MagicMock()
        mock_boto3.client.return_value = s3
        outcome = _produce_artifact(module_name)(
            result, bucket="test-bucket", run_date=BACKFILL_DATE,
        )
        assert outcome["written"] is True, f"{module_name}: {outcome}"
        # S3 Key partition carries the backfill date, not now().
        key = s3.put_object.call_args.kwargs["Key"]
        assert f"/recommendations/{BACKFILL_DATE}/" in key, (
            f"{module_name}: artifact keyed to {key}, expected backfill partition "
            f"/recommendations/{BACKFILL_DATE}/"
        )
        # And the artifact body's run_date matches.
        body = json.loads(s3.put_object.call_args.kwargs["Body"])
        assert body["run_date"] == BACKFILL_DATE


@pytest.mark.parametrize("module_name", sorted(_OPTIMIZERS))
def test_none_run_date_falls_back_to_today_iso(module_name):
    result = _OPTIMIZERS[module_name]
    sentinel = "2099-01-01"  # not a real trading day, but proves the fallback path
    # today_iso is imported lazily inside produce_artifact from the
    # recommendation_artifact module, so patch it at its source.
    with patch("optimizer.recommendation_artifact.boto3") as mock_boto3, \
            patch("optimizer.recommendation_artifact.today_iso", return_value=sentinel) as ti:
        s3 = MagicMock()
        mock_boto3.client.return_value = s3
        outcome = _produce_artifact(module_name)(result, bucket="test-bucket")
        assert outcome["written"] is True, f"{module_name}: {outcome}"
        ti.assert_called()  # the live default path was taken
        body = json.loads(s3.put_object.call_args.kwargs["Body"])
        assert body["run_date"] == sentinel


@pytest.mark.parametrize("module_name", sorted(_OPTIMIZERS))
def test_apply_threads_run_date_to_produce_artifact(module_name):
    """apply(result, bucket, run_date) must forward run_date to produce_artifact."""
    mod = __import__(f"optimizer.{module_name}", fromlist=["apply"])
    result = _OPTIMIZERS[module_name]
    with patch(f"optimizer.{module_name}.produce_artifact") as mock_produce, \
            patch("optimizer.recommendation_artifact.boto3") as mock_boto3:
        mock_boto3.client.return_value = MagicMock()
        mock_produce.return_value = {"written": True, "key": "k", "run_id": "r"}
        # apply may do additional S3 work after produce_artifact; tolerate it.
        try:
            mod.apply(result, "test-bucket", BACKFILL_DATE)
        except Exception:
            # The legacy live-write path may fail under the bare MagicMock; the
            # produce_artifact forwarding (asserted below) happens first.
            pass
        assert mock_produce.called, f"{module_name}: apply did not call produce_artifact"
        # run_date forwarded either positionally or by keyword.
        call = mock_produce.call_args
        forwarded = call.kwargs.get("run_date")
        if forwarded is None and len(call.args) >= 3:
            forwarded = call.args[2]
        assert forwarded == BACKFILL_DATE, (
            f"{module_name}: apply forwarded run_date={forwarded!r}, expected {BACKFILL_DATE!r}"
        )
