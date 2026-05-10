"""
predictor_sizing_optimizer.py — recommend using p_up for position sizing.

Computes rank IC of predictor p_up vs realized canonical alpha from
predictor_outcomes. If IC is consistently positive over sufficient samples,
recommends enabling p_up-weighted position sizing in the executor.

Reads canonical alpha via pipeline_common.ALPHA_COALESCE_SQL — decimal-
scale `actual_log_alpha` for new rows post 2026-05-09 21d migration,
or `actual_5d_return / 100` for legacy rows. Min-data gate: requires
>= 30 resolved predictions.
"""

import json
import logging
import sqlite3
from datetime import date

import boto3
import pandas as pd

from pipeline_common import (
    ALPHA_COALESCE_SQL,
    HORIZON_COALESCE_SQL,
    OUTCOMES_RESOLVED_SQL,
)

logger = logging.getLogger(__name__)

S3_PARAMS_KEY = "config/executor_params.json"

_MIN_SAMPLES = 30
_MIN_IC_TO_ENABLE = 0.05  # rank IC must exceed 0.05 to recommend p_up sizing
_MIN_POSITIVE_WEEKS = 6   # at least 6 out of 8 rolling weeks must have positive IC
_ROLLING_WEEKS = 8

_cfg: dict = {}


def init_config(config: dict) -> None:
    global _cfg
    _cfg = config.get("predictor_sizing_optimizer", {})


def analyze(research_db_path: str) -> dict:
    """
    Compute rank IC of p_up vs realized 5d returns.

    Returns dict with IC metrics and recommendation.
    """
    min_samples = _cfg.get("min_samples", _MIN_SAMPLES)
    min_ic = _cfg.get("min_ic_to_enable", _MIN_IC_TO_ENABLE)

    try:
        conn = sqlite3.connect(research_db_path)
        df = pd.read_sql_query(
            "SELECT prediction_date, symbol, p_up, "
            f"{ALPHA_COALESCE_SQL} AS canonical_actual, "
            f"{HORIZON_COALESCE_SQL} AS horizon_days "
            "FROM predictor_outcomes "
            f"WHERE p_up IS NOT NULL AND {OUTCOMES_RESOLVED_SQL} "
            "ORDER BY prediction_date",
            conn,
        )
        conn.close()
    except Exception as e:
        return {"status": "error", "error": str(e)}

    if len(df) < min_samples:
        return {
            "status": "insufficient_data",
            "n_samples": len(df),
            "min_required": min_samples,
        }

    # Overall rank IC
    overall_ic = float(df["p_up"].corr(df["canonical_actual"], method="spearman"))

    # Weekly IC for rolling consistency check
    df["week"] = pd.to_datetime(df["prediction_date"]).dt.isocalendar().week.astype(int)
    df["year_week"] = (
        pd.to_datetime(df["prediction_date"]).dt.year.astype(str) + "-W"
        + df["week"].astype(str).str.zfill(2)
    )
    weekly_ic = []
    for yw, group in df.groupby("year_week"):
        if len(group) >= 5:
            ic = float(group["p_up"].corr(group["canonical_actual"], method="spearman"))
            weekly_ic.append({"week": yw, "ic": round(ic, 4), "n": len(group)})

    positive_weeks = sum(1 for w in weekly_ic if w["ic"] > 0)
    total_weeks = len(weekly_ic)
    min_pos_weeks = _cfg.get("min_positive_weeks", _MIN_POSITIVE_WEEKS)
    rolling = _cfg.get("rolling_weeks", _ROLLING_WEEKS)

    # Use the most recent N weeks for the rolling check
    recent_weekly = weekly_ic[-rolling:] if len(weekly_ic) >= rolling else weekly_ic
    recent_positive = sum(1 for w in recent_weekly if w["ic"] > 0)
    recent_mean_ic = (
        sum(w["ic"] for w in recent_weekly) / len(recent_weekly)
        if recent_weekly else 0
    )

    should_enable = (
        overall_ic >= min_ic
        and recent_positive >= min(min_pos_weeks, len(recent_weekly))
        and recent_mean_ic >= min_ic
    )

    # Compute value-add: p_up-weighted return vs equal-weight return
    df["rank_pct"] = df.groupby("prediction_date")["p_up"].rank(pct=True)
    weighted_return = (df["rank_pct"] * df["canonical_actual"]).mean()
    equal_weight_return = df["canonical_actual"].mean()
    sizing_lift = weighted_return - equal_weight_return

    return {
        "status": "ok",
        "n_samples": len(df),
        "overall_rank_ic": round(overall_ic, 4),
        "recent_mean_ic": round(recent_mean_ic, 4),
        "recent_positive_weeks": recent_positive,
        "recent_total_weeks": len(recent_weekly),
        "total_positive_weeks": positive_weeks,
        "total_weeks": total_weeks,
        "sizing_lift": round(sizing_lift, 6),
        "equal_weight_return": round(equal_weight_return, 6),
        "weighted_return": round(weighted_return, 6),
        "recommendation": "enable" if should_enable else "keep_disabled",
        "weekly_ic": weekly_ic[-12:],  # last 12 weeks for report
    }


def _build_overlay_params(result: dict) -> tuple[dict, list[str]]:
    """Compute the field_overlay payload this optimizer wants applied.

    Returns (params_dict, overlay_keys). Used by both ``apply()`` (legacy
    read-modify-write path) and ``produce_artifact()`` (single source of
    truth for what this optimizer recommends).
    """
    params = {
        "use_p_up_sizing": True,
        "p_up_sizing_blend": _cfg.get("blend_factor", 0.3),
        "p_up_sizing_updated_at": str(date.today()),
        "p_up_sizing_ic": result.get("overall_rank_ic"),
    }
    return params, list(params.keys())


def produce_artifact(result: dict, bucket: str, run_id: str | None = None) -> dict:
    """
    Convert a predictor_sizing_optimizer ``recommend()`` result into a
    typed ``RecommendationArtifact`` and write it to S3 at
    ``config/executor_params/recommendations/{date}/from_predictor_sizing_optimizer.json``.

    Part of the optimizer-artifact-assembler arc — see
    ``alpha-engine-docs/private/optimizer-artifact-assembler-260509.md``.
    Always writes regardless of ``result.status`` / ``recommendation`` so
    the audit trail captures every invocation. Uses ``field_overlay`` kind
    because this optimizer touches a narrow set of named fields without
    replacing the broader executor_params block.

    Returns ``{"written": True, "key": str}`` on success or
    ``{"written": False, "reason": str}`` on non-fatal failure.
    """
    from optimizer.recommendation_artifact import (
        RecommendationArtifact, derive_promotion_intent, today_iso, write_artifact,
    )

    try:
        # Always compute the overlay payload — even when status != ok or
        # recommendation == "keep_disabled" — so the artifact records what
        # the optimizer would HAVE recommended. promotion_intent reflects
        # the actual gate decision.
        if result.get("status") == "ok" and result.get("recommendation") == "enable":
            params, overlay_keys = _build_overlay_params(result)
            intent = derive_promotion_intent(result)
        else:
            params = {}
            overlay_keys = None
            intent = "skip"

        diagnostic = {
            k: result.get(k)
            for k in (
                "status", "recommendation", "n_samples", "overall_rank_ic",
                "recent_mean_ic", "recent_positive_weeks", "recent_total_weeks",
                "total_positive_weeks", "total_weeks", "sizing_lift",
                "equal_weight_return", "weighted_return",
            )
            if result.get(k) is not None
        }
        artifact = RecommendationArtifact(
            fit_target="sizing_ic",
            optimizer_name="predictor_sizing_optimizer",
            run_date=today_iso(),
            recommendation_kind="field_overlay",
            recommended_params=params,
            overlay_keys=overlay_keys,
            promotion_intent=intent,
            diagnostic=diagnostic,
            notes=result.get("note", "") or "",
        )
        if run_id is not None:
            artifact.run_id = run_id
        key = write_artifact(artifact, bucket, config_type="executor_params")
        return {"written": True, "key": key, "run_id": artifact.run_id}
    except Exception as e:
        logger.warning(
            "Failed to write predictor_sizing_optimizer recommendation "
            "artifact: %s (non-fatal — legacy live write still proceeds)", e,
        )
        return {"written": False, "reason": str(e)}


def apply(result: dict, bucket: str) -> dict:
    """Write use_p_up_sizing flag to executor_params.json on S3.

    Additionally — and unconditionally — produces a per-optimizer
    recommendation artifact at
    ``config/executor_params/recommendations/{date}/from_predictor_sizing_optimizer.json``
    via ``produce_artifact()``. Part of the optimizer-artifact-assembler
    arc; the artifact is additive (no behavior change to the legacy live
    write) and is consumed by the future assembler module.
    """
    # Always produce the artifact — captures every invocation for audit.
    produce_artifact(result, bucket)

    # Cutover gate: when assembler.cutover_enabled is true, the assembler
    # is the sole writer of the live key.
    from optimizer.assembler import is_cutover_enabled
    if is_cutover_enabled():
        return {
            "applied": False,
            "reason": "cutover_mode — assembler is sole live writer",
        }

    if result.get("status") != "ok":
        return {"applied": False, "reason": f"status={result.get('status')}"}

    if result.get("recommendation") != "enable":
        return {
            "applied": False,
            "reason": f"IC insufficient (overall={result.get('overall_rank_ic')}, "
                      f"recent_mean={result.get('recent_mean_ic')})",
        }

    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=S3_PARAMS_KEY)
        current = json.loads(obj["Body"].read())
    except Exception:
        current = {}

    if current.get("use_p_up_sizing") is True:
        return {"applied": False, "reason": "already enabled"}

    overlay_params, _ = _build_overlay_params(result)
    current.update(overlay_params)

    body = json.dumps(current, indent=2)
    s3.put_object(Bucket=bucket, Key=S3_PARAMS_KEY, Body=body, ContentType="application/json")
    logger.info("p_up sizing enabled in S3 (IC=%.3f)", result.get("overall_rank_ic", 0))

    return {"applied": True, "ic": result.get("overall_rank_ic")}
