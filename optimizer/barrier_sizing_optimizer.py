"""
barrier_sizing_optimizer.py — recommend using barrier_win_prob for sizing.

Task B3 of the predictor↔executor triple-barrier coherence arc. The sibling
of ``predictor_sizing_optimizer`` (the p_up precedent): computes the rank IC of
the predictor's calibrated ``barrier_win_prob`` = P(profit/upper barrier touched
before stop/lower barrier) vs realized canonical alpha from ``predictor_outcomes``.
If IC is consistently positive over sufficient samples, recommends flipping
``barrier_win_prob_sizing_enabled`` on in the executor (Task B2 ships the
consumer dormant).

This is the institutional validation for a brand-new per-ticker sizing field —
offline criterion-IC over accumulated production weeks, NOT a grid sweep (the
backtester simulates with ``predictions_by_ticker={}`` so a sweep multiplier is
inert). Mirrors the p_up promotion gate exactly.

ACTIVATION PREREQUISITE: ``predictor_outcomes`` must carry a ``barrier_win_prob``
column (recorded by alpha-engine-data when it joins predictions.json to realized
outcomes, mirroring how ``p_up`` got there). Until that producer-side change
lands AND ≥``_MIN_SAMPLES`` rows resolve, ``analyze`` returns
``barrier_win_prob_column_absent`` / ``insufficient_data`` — informational, not
an error. See the ROADMAP B3 activation-prereq follow-up.

Reads canonical alpha via pipeline_common.ALPHA_COALESCE_SQL (decimal-scale
``actual_log_alpha`` for post-2026-05-09 rows, ``actual_5d_return / 100`` legacy).
"""

import json
import logging
import sqlite3
from datetime import date

import boto3
import pandas as pd

from pipeline_common import (
    ALPHA_COALESCE_SQL,
    CURRENT_HORIZON_FILTER_SQL,
    HORIZON_COALESCE_SQL,
    OUTCOMES_RESOLVED_SQL,
)

logger = logging.getLogger(__name__)

S3_PARAMS_KEY = "config/executor_params.json"

_MIN_SAMPLES = 30
_MIN_IC_TO_ENABLE = 0.05   # rank IC must exceed 0.05 to recommend the flip
_MIN_POSITIVE_WEEKS = 6    # at least 6 of the recent rolling weeks positive
_ROLLING_WEEKS = 8

_cfg: dict = {}


def init_config(config: dict) -> None:
    global _cfg
    _cfg = config.get("barrier_sizing_optimizer", {})


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


def analyze(research_db_path: str) -> dict:
    """Compute rank IC of barrier_win_prob vs realized canonical alpha.

    Returns a dict with IC metrics + an enable/keep_disabled recommendation,
    or an informational status when the field hasn't been recorded yet.
    """
    min_samples = _cfg.get("min_samples", _MIN_SAMPLES)
    min_ic = _cfg.get("min_ic_to_enable", _MIN_IC_TO_ENABLE)

    try:
        conn = sqlite3.connect(research_db_path)
        # The field is brand new — until alpha-engine-data records it into
        # predictor_outcomes the column is absent. Detect explicitly so this
        # reads as "not yet available", not a query error.
        if not _column_exists(conn, "predictor_outcomes", "barrier_win_prob"):
            conn.close()
            return {
                "status": "barrier_win_prob_column_absent",
                "note": (
                    "predictor_outcomes has no barrier_win_prob column yet — "
                    "awaiting the alpha-engine-data producer change that records "
                    "it from predictions.json (B3 activation prerequisite)."
                ),
            }
        df = pd.read_sql_query(
            "SELECT prediction_date, symbol, barrier_win_prob, "
            f"{ALPHA_COALESCE_SQL} AS canonical_actual, "
            f"{HORIZON_COALESCE_SQL} AS horizon_days "
            "FROM predictor_outcomes "
            "WHERE barrier_win_prob IS NOT NULL "
            f"  AND {OUTCOMES_RESOLVED_SQL} "
            f"  AND {CURRENT_HORIZON_FILTER_SQL} "
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
    overall_ic = float(df["barrier_win_prob"].corr(df["canonical_actual"], method="spearman"))

    # Weekly IC for rolling consistency check
    df["week"] = pd.to_datetime(df["prediction_date"]).dt.isocalendar().week.astype(int)
    df["year_week"] = (
        pd.to_datetime(df["prediction_date"]).dt.year.astype(str) + "-W"
        + df["week"].astype(str).str.zfill(2)
    )
    weekly_ic = []
    for yw, group in df.groupby("year_week"):
        if len(group) >= 5:
            ic = float(group["barrier_win_prob"].corr(group["canonical_actual"], method="spearman"))
            weekly_ic.append({"week": yw, "ic": round(ic, 4), "n": len(group)})

    positive_weeks = sum(1 for w in weekly_ic if w["ic"] > 0)
    total_weeks = len(weekly_ic)
    min_pos_weeks = _cfg.get("min_positive_weeks", _MIN_POSITIVE_WEEKS)
    rolling = _cfg.get("rolling_weeks", _ROLLING_WEEKS)

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

    # Value-add: barrier_win_prob-rank-weighted return vs equal-weight.
    df["rank_pct"] = df.groupby("prediction_date")["barrier_win_prob"].rank(pct=True)
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
        "weekly_ic": weekly_ic[-12:],
    }


def _build_overlay_params(result: dict) -> tuple[dict, list[str]]:
    """The field_overlay payload this optimizer recommends applying."""
    params = {
        "barrier_win_prob_sizing_enabled": True,
        "barrier_win_prob_sizing_min": _cfg.get("sizing_min", 0.70),
        "barrier_win_prob_sizing_range": _cfg.get("sizing_range", 0.60),
        "barrier_win_prob_sizing_updated_at": str(date.today()),
        "barrier_win_prob_sizing_ic": result.get("overall_rank_ic"),
    }
    return params, list(params.keys())


def produce_artifact(result: dict, bucket: str, run_id: str | None = None) -> dict:
    """Write a typed RecommendationArtifact to S3 (always — full audit trail).

    Mirrors predictor_sizing_optimizer.produce_artifact; field_overlay kind
    (touches a narrow named set without replacing the executor_params block).
    """
    from optimizer.recommendation_artifact import (
        RecommendationArtifact, derive_promotion_intent, today_iso, write_artifact,
    )

    try:
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
            fit_target="barrier_sizing_ic",
            optimizer_name="barrier_sizing_optimizer",
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
            "Failed to write barrier_sizing_optimizer recommendation artifact: "
            "%s (non-fatal)", e,
        )
        return {"written": False, "reason": str(e)}


def apply(result: dict, bucket: str) -> dict:
    """Write barrier_win_prob_sizing_enabled flag to executor_params.json on S3.

    Always produces the recommendation artifact first (audit trail). Honors the
    assembler cutover gate, mirroring predictor_sizing_optimizer.apply.
    """
    produce_artifact(result, bucket)

    from optimizer.assembler import is_cutover_enabled
    if is_cutover_enabled():
        return {"applied": False, "reason": "cutover_mode — assembler is sole live writer"}

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

    if current.get("barrier_win_prob_sizing_enabled") is True:
        return {"applied": False, "reason": "already enabled"}

    overlay_params, _ = _build_overlay_params(result)
    current.update(overlay_params)

    body = json.dumps(current, indent=2)
    s3.put_object(Bucket=bucket, Key=S3_PARAMS_KEY, Body=body, ContentType="application/json")
    logger.info("barrier_win_prob sizing enabled in S3 (IC=%.3f)", result.get("overall_rank_ic", 0))

    return {"applied": True, "ic": result.get("overall_rank_ic")}
