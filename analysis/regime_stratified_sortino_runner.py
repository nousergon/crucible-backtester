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

from nousergon_lib.quant.horizons import DEFAULT_POLICY

from analysis.regime_stratified_sortino import (
    DEFAULT_MIN_PICKS_PER_STRATUM,
    STATUS_UNMEASURABLE,
    SUPPORTED_HORIZONS,
    InputWindow,
    ReturnUnits,
    assemble_t2_eval_payload,
    assess_input_freshness,
    compute_regime_spread,
    input_window,
    load_with_subscores_and_regime,
    stratified_sortino_by_regime,
)


logger = logging.getLogger(__name__)

REGIME_STRATIFIED_SORTINO_PREFIX = "regime/stratified_sortino"

# ``attach_outcomes`` reproduces the legacy wide-column convention exactly —
# ``round(decimal * 100, 2)`` — so the arithmetic return columns this runner
# hands to the metric core are PERCENT POINTS, not fractions. Declaring it is
# mandatory (alpha-engine-config-I7661): the metric core has no default and
# raises rather than guessing. The primary horizon does not use this at all —
# it resolves the canonical decimal ``log_alpha_21d`` the store publishes.
OUTCOME_RETURN_UNITS = ReturnUnits.PERCENT

# score_performance is FROZEN at this score_date by ruling, not broken: its
# seed path inserts only rating == "BUY" and the quant-only signals envelope
# has rated every name HOLD since 2026-07-18 (alpha-engine-config-I11609;
# nousergon-data collectors/signal_returns.py
# ::_SCORE_PERFORMANCE_V1_FROZEN_THROUGH, which this must equal). T2 is
# therefore unmeasurable every week until v2 supplies an outcome source, and
# an ERROR that fires every week for a recorded decision pages for nothing.
SCORE_PERFORMANCE_V1_FROZEN_THROUGH = "2026-07-10"


def _unmeasurable_because_frozen(window: InputWindow) -> bool:
    """The newest input is exactly the declared freeze point. Any OTHER stale
    newest date is a producer that stopped for an undeclared reason, and that
    still logs at ERROR."""
    return window.max_score_date == SCORE_PERFORMANCE_V1_FROZEN_THROUGH


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
    primary_horizon = DEFAULT_POLICY.primary_horizon
    diagnostic_horizon = DEFAULT_POLICY.diagnostic_horizons[0]

    df = load_with_subscores_and_regime(db_path)
    if df.empty:
        logger.info(
            "[T2] score_performance is empty — emitting placeholder payload "
            "with n_strata=0"
        )
        strata: list = []
        window = InputWindow(None, None, 0)
    else:
        strata = stratified_sortino_by_regime(
            df,
            units=OUTCOME_RETURN_UNITS,
            min_picks_per_stratum=min_picks_per_stratum,
            horizons=SUPPORTED_HORIZONS,
        )
        window = input_window(df)

    spread_primary = compute_regime_spread(strata, horizon_days=primary_horizon)
    spread_diagnostic = compute_regime_spread(strata, horizon_days=diagnostic_horizon)

    dual = now_dual()
    run_id = new_eval_run_id()

    # Freshness is a property of the INPUTS, not of the write time
    # (alpha-engine-config-I7661). Four consecutive weekly artifacts computed
    # off rows frozen in March, and every write-time check the fleet has read
    # them as healthy.
    status, status_reason = assess_input_freshness(
        window, trading_day=str(dual.trading_day), horizon_days=primary_horizon,
    )
    if status == STATUS_UNMEASURABLE and _unmeasurable_because_frozen(window):
        status_reason = (
            f"{status_reason} Expected: score_performance is frozen through "
            f"{SCORE_PERFORMANCE_V1_FROZEN_THROUGH} by ruling "
            f"(alpha-engine-config-I11609, alpha-engine-config-I11481)."
        )
        logger.warning(
            "[T2] regime_stratified_sortino is UNMEASURABLE (declared freeze): %s",
            status_reason,
        )
    elif status == STATUS_UNMEASURABLE:
        logger.error(
            "[T2] regime_stratified_sortino is UNMEASURABLE: %s", status_reason,
        )

    payload = assemble_t2_eval_payload(
        strata=strata,
        spread_primary=spread_primary,
        spread_diagnostic=spread_diagnostic,
        run_id=run_id,
        calendar_date=str(dual.calendar_date),
        trading_day=str(dual.trading_day),
        window=window,
        status=status,
        status_reason=status_reason,
        units=OUTCOME_RETURN_UNITS,
        min_picks_per_stratum=min_picks_per_stratum,
        policy=DEFAULT_POLICY,
    )

    summary = {
        # The artifact's own verdict, not a blanket "ok". A run whose inputs
        # could not support a measurement says so on the surface the tracker
        # and the dashboard read.
        "status": status,
        "status_reason": status_reason,
        "payload": payload,
        "n_strata": len(strata),
        "input_window": window.as_dict(),
        f"spread_{primary_horizon}d_interpretation": spread_primary.get("interpretation"),
        f"spread_{diagnostic_horizon}d_interpretation": spread_diagnostic.get("interpretation"),
    }
    if not write or not s3_bucket:
        return {**summary, "wrote": False}

    keys = _write_t2_eval_artifact(payload, bucket=s3_bucket)
    return {**summary, "wrote": True, **keys}


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
    (spread_21d / spread_5d are first-class blocks — policy-derived names
    since alpha-engine-config-I7661), so writing the
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
    primary = DEFAULT_POLICY.primary_horizon
    diagnostic = DEFAULT_POLICY.diagnostic_horizons[0]
    logger.info(
        "[T2] wrote run_id=%s → s3://%s/%s (latest=%s) | status=%s | "
        "inputs=%s..%s (n=%s) | interpretation_%sd=%s | interpretation_%sd=%s",
        run_id, bucket, artifact_key, latest_key,
        payload.get("status"),
        (payload.get("input_window") or {}).get("min_score_date"),
        (payload.get("input_window") or {}).get("max_score_date"),
        (payload.get("input_window") or {}).get("n_rows"),
        primary, (payload.get(f"spread_{primary}d") or {}).get("interpretation"),
        diagnostic, (payload.get(f"spread_{diagnostic}d") or {}).get("interpretation"),
    )
    return {"artifact_key": artifact_key, "latest_key": latest_key}
