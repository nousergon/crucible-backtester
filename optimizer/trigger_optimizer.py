"""
trigger_optimizer.py — auto-disable underperforming entry triggers.

Reads trigger scorecard results from analysis/trigger_scorecard.py.
If a trigger type has consistently negative entry timing alpha over
sufficient samples, recommends disabling it by writing a disabled_triggers
list to config/executor_params.json on S3.

Min-data gate: requires >= 50 trades per trigger type before recommending
any disabling. Until then, all triggers remain active.
"""

import json
import logging
from datetime import date

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

S3_PARAMS_KEY = "config/executor_params.json"

_MIN_TRADES_PER_TRIGGER = 50
_MAX_NEGATIVE_ALPHA_PCT = -0.005  # -0.5% avg alpha → candidate for disabling
_MAX_UNFAVORABLE_SLIPPAGE_BPS = 20  # 20bps mean unfavorable slippage

_cfg: dict = {}


def init_config(config: dict) -> None:
    global _cfg
    _cfg = config.get("trigger_optimizer", {})


def analyze(trigger_scorecard: dict) -> dict:
    """
    Analyze trigger scorecard and recommend which triggers to disable.

    Args:
        trigger_scorecard: dict from trigger_scorecard.compute_trigger_scorecard()

    Returns:
        dict with status, recommendations, and disable list.
    """
    if not trigger_scorecard or trigger_scorecard.get("status") != "ok":
        return {"status": "insufficient_data", "note": "No trigger scorecard available"}

    triggers = trigger_scorecard.get("triggers", [])
    if not triggers:
        return {"status": "insufficient_data", "note": "No trigger data in scorecard"}

    min_trades = _cfg.get("min_trades_per_trigger", _MIN_TRADES_PER_TRIGGER)
    alpha_threshold = _cfg.get("max_negative_alpha_pct", _MAX_NEGATIVE_ALPHA_PCT)
    slippage_threshold = _cfg.get("max_unfavorable_slippage_bps", _MAX_UNFAVORABLE_SLIPPAGE_BPS)

    recommendations = []
    disable_list = []
    total_evaluated = 0

    for t in triggers:
        name = t.get("trigger", "unknown")
        n = t.get("n_trades", 0)
        avg_alpha = t.get("avg_realized_alpha", 0)
        avg_slippage_vs_open = t.get("avg_slippage_vs_open_pct", 0)

        if n < min_trades:
            recommendations.append({
                "trigger": name,
                "action": "keep",
                "reason": f"insufficient data ({n} < {min_trades} trades)",
                "n_trades": n,
            })
            continue

        total_evaluated += 1
        should_disable = False
        reasons = []

        if avg_alpha is not None and avg_alpha < alpha_threshold:
            should_disable = True
            reasons.append(f"negative avg alpha ({avg_alpha:.3%})")

        if avg_slippage_vs_open is not None and avg_slippage_vs_open * 10000 > slippage_threshold:
            reasons.append(f"high slippage ({avg_slippage_vs_open * 10000:.0f}bps)")

        win_rate = t.get("win_rate_vs_spy")
        if win_rate is not None and win_rate < 0.40 and n >= min_trades:
            should_disable = True
            reasons.append(f"low win rate ({win_rate:.0%})")

        action = "disable" if should_disable else "keep"
        if should_disable:
            disable_list.append(name)

        recommendations.append({
            "trigger": name,
            "action": action,
            "reasons": reasons,
            "n_trades": n,
            "avg_alpha": round(avg_alpha, 4) if avg_alpha is not None else None,
            "win_rate": round(win_rate, 3) if win_rate is not None else None,
        })

    # Never disable ALL triggers — always keep at least time_expiry as fallback
    if len(disable_list) >= len([t for t in triggers if t.get("n_trades", 0) >= min_trades]):
        disable_list = [d for d in disable_list if d != "time_expiry"]
        for r in recommendations:
            if r["trigger"] == "time_expiry" and r["action"] == "disable":
                r["action"] = "keep"
                r["reasons"] = ["preserved as fallback — cannot disable all triggers"]

    return {
        "status": "ok" if total_evaluated > 0 else "insufficient_data",
        "total_evaluated": total_evaluated,
        "recommendations": recommendations,
        "disabled_triggers": disable_list,
        "min_trades_threshold": min_trades,
    }


def _build_overlay_params(result: dict) -> tuple[dict, list[str]]:
    """Compute the field_overlay payload this optimizer wants applied.

    Returns (params_dict, overlay_keys). Used by both ``apply()`` (legacy
    read-modify-write path) and ``produce_artifact()`` (single source of
    truth for what this optimizer recommends).
    """
    params = {
        "disabled_triggers": result.get("disabled_triggers", []),
        "disabled_triggers_updated_at": str(date.today()),
    }
    return params, list(params.keys())


def produce_artifact(result: dict, bucket: str, run_id: str | None = None) -> dict:
    """
    Convert a trigger_optimizer ``recommend()`` result into a typed
    ``RecommendationArtifact`` and write it to S3 at
    ``config/executor_params/recommendations/{date}/from_trigger_optimizer.json``.

    Part of the optimizer-artifact-assembler arc — see
    ``alpha-engine-docs/private/optimizer-artifact-assembler-260509.md``.
    Always writes regardless of ``result.status`` so the audit trail
    captures every invocation. Uses ``field_overlay`` kind because this
    optimizer touches the ``disabled_triggers`` field set without
    replacing the broader executor_params block.

    Returns ``{"written": True, "key": str}`` on success or
    ``{"written": False, "reason": str}`` on non-fatal failure.
    """
    from optimizer.recommendation_artifact import (
        RecommendationArtifact, derive_promotion_intent, today_iso, write_artifact,
    )

    try:
        if result.get("status") == "ok":
            params, overlay_keys = _build_overlay_params(result)
            intent = derive_promotion_intent(result)
        else:
            params = {}
            overlay_keys = None
            intent = "skip"

        diagnostic = {
            k: result.get(k)
            for k in (
                "status", "total_evaluated", "min_trades_threshold",
                "recommendations",
            )
            if result.get(k) is not None
        }
        artifact = RecommendationArtifact(
            fit_target="entry_timing_alpha",
            optimizer_name="trigger_optimizer",
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
            "Failed to write trigger_optimizer recommendation artifact: %s "
            "(non-fatal — legacy live write still proceeds)", e,
        )
        return {"written": False, "reason": str(e)}


def apply(result: dict, bucket: str) -> dict:
    """Write disabled_triggers list to executor_params.json on S3.

    Additionally — and unconditionally — produces a per-optimizer
    recommendation artifact at
    ``config/executor_params/recommendations/{date}/from_trigger_optimizer.json``
    via ``produce_artifact()``. Part of the optimizer-artifact-assembler
    arc; the artifact is additive (no behavior change to the legacy live
    write) and is consumed by the future assembler module.
    """
    # Always produce the artifact — captures every invocation for audit.
    produce_artifact(result, bucket)

    if result.get("status") != "ok":
        return {"applied": False, "reason": f"status={result.get('status')}"}

    disable_list = result.get("disabled_triggers", [])

    # Read current executor params
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=S3_PARAMS_KEY)
        current = json.loads(obj["Body"].read())
    except Exception:
        current = {}

    current_disabled = current.get("disabled_triggers", [])
    if set(disable_list) == set(current_disabled):
        return {"applied": False, "reason": "no change from current disabled list"}

    overlay_params, _ = _build_overlay_params(result)
    current.update(overlay_params)

    body = json.dumps(current, indent=2)
    s3.put_object(Bucket=bucket, Key=S3_PARAMS_KEY, Body=body, ContentType="application/json")
    logger.info("Disabled triggers updated in S3: %s", disable_list)

    return {"applied": True, "disabled_triggers": disable_list}
