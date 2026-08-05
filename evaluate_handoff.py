"""
evaluate_handoff.py — S3-mediated diagnostics→optimizer snapshot handoff
(config-I3112 deliverable 2).

The weekly SF 'Evaluator' state used to bundle evaluate.py's whole internal
pipeline — signal quality → ~20 diagnostics → ~9 optimizers → assembler →
champion promotion → report/email — into ONE opaque SSM task with ONE
executionTimeout. The 2026-07-20 incident (watch-rerun-2026-07-18-10/-11, a
3600s SIGKILL) showed why that is a poor design: no per-task budget, zero
visibility into which internal phase was running, and a retry can only replay
the whole bundle. config-I3112 (Brian design ruling 2026-07-20) decomposes
the state into two sequential SF states — EvaluatorDiagnostics
(--mode diagnostics) and EvaluatorOptimize (--mode optimize) — each with its
own executionTimeout and Catch path, reusing the same spot box.

For the optimize half to be FAITHFUL to the bundled run — not a degraded
shell — everything the optimizer/report stages consume that the diagnostics
half produces must cross the state boundary. That is more than the
diagnostics dict (trigger_scorecard / e2e_lift / factor_blend_sensitivity,
the three keys the issue named): the weight/veto/research/pillar optimizers
and the action-entropy/team-metrics report bundles all compute on
``df_base`` (the finalized-signal frame), the report card reads
``sq_result``/``regime_rows``/``score_rows``/``attr_result``, and the digest
email reads ``sq_result``. Pre-split, running ``--mode optimize`` standalone
passed an empty diagnostics dict AND None df_base — silently skipping ~7
optimizer modules and degrading every report tile that reads them. That is
the exact silent-degradation class this decomposition exists to kill, so the
handoff carries the whole diagnostics-stage snapshot, not just the dict.

Single producer, single reader:
    writer: evaluate.py --mode diagnostics --upload
    reader: evaluate.py --mode optimize --upload

Layout under the signals bucket (``evaluator/diagnostics/{date}/`` — the
prefix the issue specifies, date = the normalized trading-day label):

    diagnostics.json       the diagnostics results dict (~20 modules)
    signal_quality.json    {sq_result, regime_rows, score_rows, attr_result}
    df_base.parquet        the finalized-signal frame (absent when signal
                           quality did not produce one)

Write is FAIL-LOUD (the artifact is load-bearing: the SF's optimize state
would otherwise silently degrade); read is fail-soft on a MISSING snapshot
(NoSuchKey → None, and the caller warns + falls back to today's empty-dict
behavior — a manual ``--mode optimize`` against a date with no prior
diagnostics run must keep working) but re-raises transport errors so a real
S3 failure surfaces in the SF instead of silently degrading.

The dicts are sanitized through a numpy/pandas→Python-native pass before
JSON encoding so numeric types survive the round-trip (``default=str`` in
``phase_artifacts.save_json`` would stringify numpy scalars, silently
changing the types the optimizers and report card read back).
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError
import pandas as pd

logger = logging.getLogger(__name__)

SNAPSHOT_PREFIX = "evaluator/diagnostics/{date}"

# Artifact names within the prefix.
_DIAGNOSTICS_ARTIFACT = "diagnostics.json"
_SIGNAL_QUALITY_ARTIFACT = "signal_quality.json"
_DF_BASE_ARTIFACT = "df_base.parquet"

# Entries of the signal_quality.json payload — mirrors the tuple returned by
# evaluate.py's _run_signal_quality minus df_base (which travels as parquet).
_SQ_KEYS = ("sq_result", "regime_rows", "score_rows", "attr_result")


def snapshot_keys(date: str) -> dict[str, str]:
    """Canonical S3 keys for a snapshot, keyed by logical artifact name."""
    prefix = SNAPSHOT_PREFIX.format(date=date)
    return {
        "diagnostics": f"{prefix}/{_DIAGNOSTICS_ARTIFACT}",
        "signal_quality": f"{prefix}/{_SIGNAL_QUALITY_ARTIFACT}",
        "df_base": f"{prefix}/{_DF_BASE_ARTIFACT}",
    }


def _client(s3_client=None):
    return s3_client if s3_client is not None else boto3.client("s3")


# ── numpy/pandas → Python-native sanitizer ──────────────────────────────────


def _to_native(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars and arrays to Python natives.

    The diagnostics/signal-quality dicts are built from pandas computations
    and routinely contain np.float64/np.int64/np.bool_/pd.Timestamp values.
    json.dumps alone would choke on them, and ``default=str`` would silently
    stringify them — corrupting the numeric types the optimizer modules and
    report card read back from the handoff. Unknown object types fall back to
    str() so an unexpected value can never break the write (the write is
    load-bearing, but its FAILURE MODE is the S3 error, not a type surprise).
    """
    import numpy as np

    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.ndarray, np.generic)):
        if obj.ndim == 0:
            return _to_native(obj.item())
        return [_to_native(v) for v in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, pd.Series):
        return [_to_native(v) for v in obj.tolist()]
    if isinstance(obj, (pd.DataFrame,)):
        return {"__dataframe__": [_to_native(v) for v in obj.to_dict("records")]}
    return str(obj)


def _json_body(obj: Any) -> bytes:
    return json.dumps(_to_native(obj), indent=2).encode()


# ── Write (single producer: --mode diagnostics --upload) ────────────────────


def write_snapshot(
    bucket: str,
    date: str,
    *,
    diagnostics: dict,
    sq_result: dict,
    regime_rows: list,
    score_rows: list,
    attr_result: dict,
    df_base: Optional[pd.DataFrame],
    s3_client=None,
) -> dict[str, str]:
    """Write the diagnostics-stage snapshot and return {artifact: key}.

    FAIL-LOUD: any S3 error propagates — the SF's EvaluatorDiagnostics state
    must fail rather than let the downstream optimize state silently degrade
    (its consumers would otherwise read an absent snapshot and skip the
    diagnostics-dependent optimizers, the exact regression this split exists
    to prevent).
    """
    client = _client(s3_client)
    keys = snapshot_keys(date)

    client.put_object(
        Bucket=bucket,
        Key=keys["diagnostics"],
        Body=_json_body(diagnostics),
        ContentType="application/json",
    )
    client.put_object(
        Bucket=bucket,
        Key=keys["signal_quality"],
        Body=_json_body(
            {key: value for key, value in (
                ("sq_result", sq_result),
                ("regime_rows", regime_rows),
                ("score_rows", score_rows),
                ("attr_result", attr_result),
            )}
        ),
        ContentType="application/json",
    )
    if df_base is not None:
        buf = io.BytesIO()
        df_base.to_parquet(buf)
        client.put_object(
            Bucket=bucket,
            Key=keys["df_base"],
            Body=buf.getvalue(),
            ContentType="application/vnd.apache.parquet",
        )
    return keys


# ── Read (single reader: --mode optimize --upload) ──────────────────────────


def load_snapshot(
    bucket: str,
    date: str,
    s3_client=None,
) -> Optional[dict]:
    """Load the snapshot written by the diagnostics half.

    Returns a dict with the same logical shape as write_snapshot's inputs:

        {
            "diagnostics": dict,
            "sq_result": dict,
            "regime_rows": list,
            "score_rows": list,
            "attr_result": dict,
            "df_base": Optional[pd.DataFrame],
        }

    Fail-soft on a MISSING snapshot (NoSuchKey → None — a manual
    ``--mode optimize`` without a prior diagnostics run keeps today's
    empty-dict behavior); any OTHER S3 error propagates so a real transport
    failure surfaces in the SF instead of silently degrading the run.
    """
    client = _client(s3_client)
    keys = snapshot_keys(date)

    def _get_json(key: str):
        """Read + parse one JSON artifact. Returns None on missing OR corrupt
        (both are 'the diagnostics half did not produce a usable artifact' —
        degrade the same way, with a warning); re-raises transport errors."""
        try:
            obj = client.get_object(Bucket=bucket, Key=key)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise
        try:
            return json.loads(obj["Body"].read())
        except (ValueError, TypeError) as e:
            logger.warning(
                "Diagnostics snapshot artifact %s (s3://%s/%s) is corrupt "
                "(%s) — treating as missing",
                key, bucket, key, e,
            )
            return None

    diagnostics = _get_json(keys["diagnostics"])
    if diagnostics is None:
        logger.warning(
            "Diagnostics snapshot missing for %s at s3://%s/%s — "
            "optimize runs against an empty diagnostics dict "
            "(diagnostics-dependent optimizers will be skipped)",
            date, bucket, keys["diagnostics"],
        )
        return None

    signal_quality = _get_json(keys["signal_quality"]) or {}
    if not isinstance(diagnostics, dict):
        logger.warning(
            "Diagnostics snapshot for %s is not a dict (got %s) — "
            "treating as missing", date, type(diagnostics).__name__,
        )
        return None

    df_base = None
    df_key = keys["df_base"]
    try:
        obj = client.get_object(Bucket=bucket, Key=df_key)
        df_base = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            logger.warning(
                "df_base.parquet missing from the %s snapshot (s3://%s/%s) — "
                "df_base-dependent optimizers will be skipped",
                date, bucket, df_key,
            )
        else:
            raise

    return {
        "diagnostics": diagnostics,
        "sq_result": signal_quality.get("sq_result", {"status": "skipped"}),
        "regime_rows": signal_quality.get("regime_rows", []),
        "score_rows": signal_quality.get("score_rows", []),
        "attr_result": signal_quality.get("attr_result", {"status": "skipped"}),
        "df_base": df_base,
    }
