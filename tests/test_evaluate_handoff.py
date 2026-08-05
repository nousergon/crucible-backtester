"""Unit tests for evaluate_handoff.py — the S3-mediated diagnostics→optimizer
snapshot handoff (config-I3112 deliverable 2).

The weekly SF 'Evaluator' state is decomposed into EvaluatorDiagnostics
(--mode diagnostics) and EvaluatorOptimize (--mode optimize); the optimize
half reads everything the diagnostics half produced (diagnostics dict +
signal-quality outputs + df_base) from S3 instead of starting empty. These
tests pin the snapshot's write/read contract with a dict-backed S3 stub.

Facts pinned:
  * write_snapshot writes exactly the 3 canonical keys under
    evaluator/diagnostics/{date}/ and FAIL-LOUDs on S3 errors.
  * numpy/pandas scalars round-trip as Python natives (default=str would
    silently stringify them and corrupt the optimizer inputs).
  * df_base round-trips as parquet; absent parquet → df_base None.
  * load_snapshot returns None on a MISSING snapshot (manual --mode optimize
    without a prior diagnostics run keeps the empty-dict fallback) but
    re-raises transport errors (the SF must see them, not degrade silently).
"""

from __future__ import annotations

import json
import pickle  # noqa: F401  (parquet round-trip via fastparquet/pyarrow)

import numpy as np
import pandas as pd
import pytest
from botocore.exceptions import ClientError

from evaluate_handoff import (
    load_snapshot,
    snapshot_keys,
    write_snapshot,
)


class DictS3:
    """Minimal dict-backed S3 stub — put/get with real error semantics."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def put_object(self, **kwargs):
        key = kwargs["Key"]
        self.objects[key] = kwargs.get("Body") or b""

    def get_object(self, **kwargs):
        key = kwargs["Key"]
        if key not in self.objects:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                "GetObject",
            )
        return {"Body": _BytesReader(self.objects[key])}


class _BytesReader:
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data


_BUCKET = "alpha-engine-research"
_DATE = "2026-07-31"


def _sample_snapshot_inputs():
    diagnostics = {
        "trigger_scorecard": {"status": "ok", "triggers": [{"n_trades": np.int64(3)}]},
        "e2e_lift": {"status": "ok", "scanner_lift": np.float64(1.25)},
        "factor_blend_sensitivity": {"has_data": np.bool_(True), "n": np.int32(42)},
        "alpha_dist": {"samples": np.array([1.0, 2.5, 3.0])},
        "monte_carlo": {"as_of": pd.Timestamp("2026-07-30")},
        "plain": {"value": 1.5, "flag": True, "text": "x"},
    }
    sq_result = {"status": "ok", "grade": "B+", "score": np.float64(72.3)}
    regime_rows = [{"regime": "risk_on", "sortino": np.float64(0.9)}]
    score_rows = [{"threshold": 70, "accuracy": np.float64(0.55)}]
    attr_result = {"status": "ok", "attribution": 0.4}
    df_base = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "eval_date": ["2026-07-28", "2026-07-29", "2026-07-30"],
            "score": np.array([61.5, 72.0, 83.5], dtype=np.float64),
            "outcome": np.array([0.01, 0.02, -0.01], dtype=np.float64),
        }
    ).set_index("ticker")
    return {
        "diagnostics": diagnostics,
        "sq_result": sq_result,
        "regime_rows": regime_rows,
        "score_rows": score_rows,
        "attr_result": attr_result,
        "df_base": df_base,
    }


class TestSnapshotKeys:
    def test_canonical_layout(self):
        keys = snapshot_keys("2026-07-31")
        assert keys == {
            "diagnostics": "evaluator/diagnostics/2026-07-31/diagnostics.json",
            "signal_quality": "evaluator/diagnostics/2026-07-31/signal_quality.json",
            "df_base": "evaluator/diagnostics/2026-07-31/df_base.parquet",
        }


class TestWriteSnapshot:
    def test_writes_all_three_artifacts(self):
        s3 = DictS3()
        inputs = _sample_snapshot_inputs()
        keys = write_snapshot(_BUCKET, _DATE, s3_client=s3, **inputs)
        for key in keys.values():
            assert key in s3.objects, f"missing artifact: {key}"

    def test_artifact_keys_match_canonical_layout(self):
        s3 = DictS3()
        inputs = _sample_snapshot_inputs()
        keys = write_snapshot(_BUCKET, _DATE, s3_client=s3, **inputs)
        assert set(keys.values()) == set(snapshot_keys(_DATE).values())

    def test_no_df_base_artifact_when_none(self):
        s3 = DictS3()
        inputs = _sample_snapshot_inputs()
        inputs["df_base"] = None
        keys = write_snapshot(_BUCKET, _DATE, s3_client=s3, **inputs)
        assert keys["df_base"] not in s3.objects
        assert "diagnostics.json" in keys["diagnostics"]
        assert "signal_quality.json" in keys["signal_quality"]

    def test_fail_loud_on_s3_error(self):
        class ExplodingS3:
            def put_object(self, **kwargs):
                raise ClientError(
                    {"Error": {"Code": "InternalError", "Message": "boom"}},
                    "PutObject",
                )

        inputs = _sample_snapshot_inputs()
        with pytest.raises(ClientError):
            write_snapshot(_BUCKET, _DATE, s3_client=ExplodingS3(), **inputs)


class TestLoadSnapshot:
    def test_round_trip_restores_python_native_types(self):
        s3 = DictS3()
        inputs = _sample_snapshot_inputs()
        write_snapshot(_BUCKET, _DATE, s3_client=s3, **inputs)
        loaded = load_snapshot(_BUCKET, _DATE, s3_client=s3)
        assert loaded is not None
        assert loaded["diagnostics"]["trigger_scorecard"]["triggers"][0]["n_trades"] == 3
        assert isinstance(
            loaded["diagnostics"]["trigger_scorecard"]["triggers"][0]["n_trades"], int
        ), "np.int64 must round-trip as Python int, not str"
        assert loaded["diagnostics"]["e2e_lift"]["scanner_lift"] == 1.25
        assert isinstance(
            loaded["diagnostics"]["e2e_lift"]["scanner_lift"], float
        ), "np.float64 must round-trip as Python float, not str"
        assert loaded["diagnostics"]["factor_blend_sensitivity"]["has_data"] is True
        assert isinstance(
            loaded["diagnostics"]["factor_blend_sensitivity"]["has_data"], bool
        )
        assert loaded["diagnostics"]["alpha_dist"]["samples"] == [1.0, 2.5, 3.0]
        assert loaded["diagnostics"]["monte_carlo"]["as_of"] == "2026-07-30T00:00:00"
        assert loaded["sq_result"]["grade"] == "B+"
        assert loaded["sq_result"]["score"] == 72.3
        assert loaded["regime_rows"] == [{"regime": "risk_on", "sortino": 0.9}]
        assert loaded["attr_result"] == {"status": "ok", "attribution": 0.4}

    def test_round_trip_restores_df_base_frame(self):
        s3 = DictS3()
        inputs = _sample_snapshot_inputs()
        write_snapshot(_BUCKET, _DATE, s3_client=s3, **inputs)
        loaded = load_snapshot(_BUCKET, _DATE, s3_client=s3)
        df = loaded["df_base"]
        assert df is not None
        assert list(df.columns) == ["eval_date", "score", "outcome"]
        assert df["score"].tolist() == [61.5, 72.0, 83.5]
        assert list(df.index) == ["AAA", "BBB", "CCC"], "index must survive the round-trip"

    def test_missing_snapshot_returns_none(self):
        s3 = DictS3()
        assert load_snapshot(_BUCKET, _DATE, s3_client=s3) is None

    def test_missing_df_base_still_returns_json_artifacts(self):
        s3 = DictS3()
        inputs = _sample_snapshot_inputs()
        inputs["df_base"] = None
        write_snapshot(_BUCKET, _DATE, s3_client=s3, **inputs)
        loaded = load_snapshot(_BUCKET, _DATE, s3_client=s3)
        assert loaded is not None
        assert loaded["df_base"] is None
        assert loaded["diagnostics"]["e2e_lift"]["status"] == "ok"

    def test_transport_error_propagates(self):
        s3 = DictS3()
        inputs = _sample_snapshot_inputs()
        write_snapshot(_BUCKET, _DATE, s3_client=s3, **inputs)

        class FailingGet:
            def __init__(self, inner):
                self._inner = inner

            def get_object(self, **kwargs):
                if kwargs.get("Key", "").endswith("diagnostics.json"):
                    raise ClientError(
                        {"Error": {"Code": "InternalError", "Message": "boom"}},
                        "GetObject",
                    )
                return self._inner.get_object(**kwargs)

        with pytest.raises(ClientError):
            load_snapshot(_BUCKET, _DATE, s3_client=FailingGet(s3))

    def test_corrupt_diagnostics_json_returns_none(self):
        s3 = DictS3()
        s3.objects["evaluator/diagnostics/2026-07-31/diagnostics.json"] = b"not json {"
        s3.objects["evaluator/diagnostics/2026-07-31/signal_quality.json"] = b"{}"
        assert load_snapshot(_BUCKET, _DATE, s3_client=s3) is None

    def test_non_dict_diagnostics_returns_none(self):
        s3 = DictS3()
        s3.objects["evaluator/diagnostics/2026-07-31/diagnostics.json"] = json.dumps(
            [1, 2, 3]
        ).encode()
        s3.objects["evaluator/diagnostics/2026-07-31/signal_quality.json"] = b"{}"
        assert load_snapshot(_BUCKET, _DATE, s3_client=s3) is None
