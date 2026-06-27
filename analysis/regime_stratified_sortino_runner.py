"""
analysis/regime_stratified_sortino_runner.py — Pipeline-wiring for T2.

Closes the pipeline-wiring side of Stage C.2 T2 per
regime-v3-260514.md §5.3.3. The substrate (regime_stratified_sortino.py)
ships the pure compute; this module glues it into evaluate.py's
``tracker.run_module`` flow and writes the canonical eval-artifact to
S3.

Why a runner module instead of a Lambda?
----------------------------------------
The backtester runs as a c5.large spot EC2 once per Saturday (see
infrastructure/spot_backtest.sh) — not as a Lambda. score_performance
is a SQLite DB on local EC2 disk that the backtester already pulls.
Wiring T2 through the spot's existing ``python evaluate.py --mode all``
invocation gives us:

  * Zero new infra (no Lambda, no SF state, no IAM grant).
  * Atomic with the rest of the eval modules — same data freshness,
    same tracker completeness reporting, same per-phase artifact write.
  * Lambda would have to pull the DB from S3 anyway; the spot already
    has it on disk.

The artifact lands at
``s3://alpha-engine-research/regime/stratified_sortino/{run_id}.json``
+ ``latest.json`` sidecar, mirroring the canonical eval_artifacts shape
used by T1 (alpha-engine-predictor) and the substrate Lambda.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import boto3

from nousergon_lib.dates import now_dual
from nousergon_lib.eval_artifacts import (
    eval_artifact_key,
    eval_latest_key,
    new_eval_run_id,
)

from analysis.regime_stratified_sortino import (
    DEFAULT_MIN_PICKS_PER_STRATUM,
    SUPPORTED_HORIZONS,
    assemble_t2_eval_payload,
    compute_regime_spread,
    load_with_subscores_and_regime,
    stratified_sortino_by_regime,
)


logger = logging.getLogger(__name__)

REGIME_STRATIFIED_SORTINO_PREFIX = "regime/stratified_sortino"


def run_regime_stratified_sortino(
    *,
    db_path: str,
    s3_bucket: str | None,
    min_picks_per_stratum: int = DEFAULT_MIN_PICKS_PER_STRATUM,
    write: bool = True,
) -> dict[str, Any]:
    """End-to-end T2 eval — load score_performance → stratify → spread →
    assemble payload → publish canonical eval-artifact.

    Returns the assembled payload + S3 keys (when written). On any
    failure mode (empty DB, missing market_regime column, S3 write
    error) returns a partial payload with a ``status`` field so the
    evaluator's tracker.run_module can report it as a partial-success
    rather than crash the whole Saturday eval pipeline.
    """
    df = load_with_subscores_and_regime(db_path)
    if df.empty:
        logger.info(
            "[T2] score_performance is empty — emitting placeholder payload "
            "with n_strata=0"
        )
        strata: list = []
    else:
        strata = stratified_sortino_by_regime(
            df, min_picks_per_stratum=min_picks_per_stratum,
            horizons=SUPPORTED_HORIZONS,
        )

    spread_10d = compute_regime_spread(strata, horizon_days=10)
    spread_30d = compute_regime_spread(strata, horizon_days=30)

    dual = now_dual()
    run_id = new_eval_run_id()
    payload = assemble_t2_eval_payload(
        strata=strata,
        spread_10d=spread_10d,
        spread_30d=spread_30d,
        run_id=run_id,
        calendar_date=str(dual.calendar_date),
        trading_day=str(dual.trading_day),
        min_picks_per_stratum=min_picks_per_stratum,
    )

    if not write or not s3_bucket:
        return {
            "status": "ok",
            "wrote": False,
            "payload": payload,
            "n_strata": len(strata),
            "spread_10d_interpretation": spread_10d.get("interpretation"),
            "spread_30d_interpretation": spread_30d.get("interpretation"),
        }

    keys = _write_t2_eval_artifact(payload, bucket=s3_bucket)
    return {
        "status": "ok",
        "wrote": True,
        "payload": payload,
        "n_strata": len(strata),
        "spread_10d_interpretation": spread_10d.get("interpretation"),
        "spread_30d_interpretation": spread_30d.get("interpretation"),
        **keys,
    }


def _write_t2_eval_artifact(
    payload: dict[str, Any],
    *,
    bucket: str,
    prefix: str = REGIME_STRATIFIED_SORTINO_PREFIX,
) -> dict[str, str]:
    """Publish a T2 eval payload to S3 in canonical eval_artifacts shape.

    Forensic artifact at ``{prefix}/{run_id}.json`` always; ``{prefix}/latest.json``
    sidecar carries the headline interpretation + spread for the
    dashboard reader.

    Sidecar payload mirrors the artifact body for T2 — the headline
    summary fields are already at the top level of the assembled payload
    (spread_10d / spread_30d are first-class blocks), so writing the
    same body to both keys keeps consumers simple. T1 splits a slimmer
    sidecar from the full artifact because its body is heavier
    (per-week pairings). T2's body is small (~K strata × 2 horizons +
    2 spread blocks), so the duplication is negligible.
    """
    s3 = boto3.client("s3")
    run_id = payload["run_id"]
    artifact_key = eval_artifact_key(prefix, run_id)
    latest_key = eval_latest_key(prefix)
    body = json.dumps(payload, default=str, indent=2).encode("utf-8")

    s3.put_object(
        Bucket=bucket, Key=artifact_key, Body=body,
        ContentType="application/json",
    )
    s3.put_object(
        Bucket=bucket, Key=latest_key, Body=body,
        ContentType="application/json",
    )
    logger.info(
        "[T2] wrote run_id=%s → s3://%s/%s (latest=%s) | "
        "interpretation_10d=%s | interpretation_30d=%s",
        run_id, bucket, artifact_key, latest_key,
        payload["spread_10d"].get("interpretation"),
        payload["spread_30d"].get("interpretation"),
    )
    return {"artifact_key": artifact_key, "latest_key": latest_key}
