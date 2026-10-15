"""
trigger_optimizer.py — auto-disable underperforming entry triggers.

Reads trigger scorecard results from analysis/trigger_scorecard.py.
If a trigger type has consistently negative entry timing alpha over
sufficient samples, recommends disabling it by writing a disabled_triggers
list to config/executor_params.json on S3.

Min-data gate: requires >= 50 CLOSED entries (entries with a realized
outcome, the scorecard's ``n_closed``) per trigger type before recommending
any disabling. Until then, all triggers remain active.

Units (alpha-engine-config-I12231): the scorecard reports alpha and slippage
in PERCENT POINTS. The thresholds below are fractions (alpha) and basis
points (slippage), so the values are converted before every comparison.

OBSERVE-ONLY (alpha-engine-config-I12231, Brian's ruling #19 option (a),
2026-10-14): with a truthful join the scorecard now carries outcomes, and
this optimizer's recommendation would reach the LIVE
``config/executor_params.json`` either directly (``apply()``) or through the
assembler (a ``promote`` artifact). Until a later, explicit flip it does
neither: ``apply()`` never touches the live key, and the recommendation
artifact is written with ``promotion_intent="shadow"``, which the assembler
records but never merges. The flip is ``trigger_optimizer.live_apply: true``
in the backtester config, and only the literal boolean ``True`` arms it.
"""

import json
import logging
from datetime import date

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

S3_PARAMS_KEY = "config/executor_params.json"

_MIN_TRADES_PER_TRIGGER = 50  # compared against n_closed, not n_trades
# A FRACTION (-0.005 = -0.5%). The scorecard's avg_realized_alpha is in
# percent points and is divided by 100 before it is compared with this.
_MAX_NEGATIVE_ALPHA_PCT = -0.005  # -0.5% avg alpha → candidate for disabling
_MAX_UNFAVORABLE_SLIPPAGE_BPS = 20  # 20bps mean unfavorable slippage
_MIN_WIN_RATE = 0.40

# Scorecard category -> the key the executor's EntryTriggerEngine checks in
# ``disabled_triggers`` (crucible-executor executor/entry_triggers.py). A
# category with no entry here cannot be disabled: ``time_expiry`` is the
# executor's unconditional near-close fallback, and ``other`` / ``unknown``
# name no single trigger. Emitting a scorecard category the executor does not
# check (e.g. "vwap" for "vwap_discount") would be a silent no-op.
_EXECUTOR_TRIGGER_KEYS = {
    "pullback": "pullback",
    "vwap": "vwap_discount",
    "support": "support_bounce",
    "graduated_entry": "graduated_entry",
}

APPLY_MODE_OBSERVE = "observe"
APPLY_MODE_LIVE = "live"

_cfg: dict = {}


def init_config(config: dict) -> None:
    global _cfg
    _cfg = config.get("trigger_optimizer", {})


def apply_mode() -> str:
    """``observe`` unless the config explicitly arms the live write.

    Only the literal boolean ``True`` arms it, so a stray string such as
    ``"false"`` can never flip the trading path live.
    """
    return APPLY_MODE_LIVE if _cfg.get("live_apply") is True else APPLY_MODE_OBSERVE


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
        n_entries = t.get("n_trades", 0)
        # Every gate below is over CLOSED entries: the win rate and alpha are
        # computed over them, so the sample-size gate must count them too. A
        # scorecard without ``n_closed`` predates the outcome join and has no
        # outcomes at all.
        n_closed = t.get("n_closed") or 0
        executor_key = _EXECUTOR_TRIGGER_KEYS.get(name)

        if executor_key is None:
            recommendations.append({
                "trigger": name,
                "action": "keep",
                "reason": "not a disable-able executor trigger",
                "n_trades": n_entries,
                "n_closed": n_closed,
            })
            continue

        if n_closed < min_trades:
            recommendations.append({
                "trigger": name,
                "executor_key": executor_key,
                "action": "keep",
                "reason": f"insufficient data ({n_closed} < {min_trades} closed entries)",
                "n_trades": n_entries,
                "n_closed": n_closed,
            })
            continue

        total_evaluated += 1
        should_disable = False
        reasons = []

        avg_alpha_pct = t.get("avg_realized_alpha")  # percent points
        avg_alpha = avg_alpha_pct / 100.0 if avg_alpha_pct is not None else None
        if avg_alpha is not None and avg_alpha < alpha_threshold:
            should_disable = True
            reasons.append(f"negative avg alpha ({avg_alpha:.3%})")

        slip_open_pct = t.get("avg_slippage_vs_open")  # percent points
        slip_open_bps = slip_open_pct * 100.0 if slip_open_pct is not None else None
        if slip_open_bps is not None and slip_open_bps > slippage_threshold:
            reasons.append(f"high slippage ({slip_open_bps:.0f}bps)")

        win_rate = t.get("win_rate_vs_spy")
        if win_rate is not None and win_rate < _MIN_WIN_RATE:
            should_disable = True
            reasons.append(f"low win rate ({win_rate:.0%})")

        action = "disable" if should_disable else "keep"
        if should_disable:
            disable_list.append(executor_key)

        recommendations.append({
            "trigger": name,
            "executor_key": executor_key,
            "action": action,
            "reasons": reasons,
            "n_trades": n_entries,
            "n_closed": n_closed,
            "avg_alpha": round(avg_alpha, 6) if avg_alpha is not None else None,
            "avg_alpha_pct": round(avg_alpha_pct, 4) if avg_alpha_pct is not None else None,
            "win_rate": round(win_rate, 3) if win_rate is not None else None,
        })

    # No "never disable all" carve-out is needed: time_expiry is not in
    # _EXECUTOR_TRIGGER_KEYS, and the executor fires it unconditionally near
    # the close, so every approved entry keeps a fallback.

    return {
        "status": "ok" if total_evaluated > 0 else "insufficient_data",
        "total_evaluated": total_evaluated,
        "recommendations": recommendations,
        "disabled_triggers": disable_list,
        "min_trades_threshold": min_trades,
        "apply_mode": apply_mode(),
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


def produce_artifact(
    result: dict, bucket: str, run_id: str | None = None, run_date: str | None = None,
) -> dict:
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
            if apply_mode() != APPLY_MODE_LIVE:
                # Observe-only: the assembler merges ONLY ``promote``
                # artifacts, so ``shadow`` records the would-be overlay for
                # audit without it ever reaching the live key.
                intent = "shadow"
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
        diagnostic["apply_mode"] = apply_mode()
        artifact = RecommendationArtifact(
            fit_target="entry_timing_alpha",
            optimizer_name="trigger_optimizer",
            # config#1017: explicit backfill run_date over ambient today_iso().
            # A live run passes None → today_iso() (current trading day); an
            # explicit --date backfill threads config["_run_date"] so the
            # artifact partition matches the backfilled assemble read.
            run_date=run_date or today_iso(),
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


def apply(result: dict, bucket: str, run_date: str | None = None) -> dict:
    """Write disabled_triggers list to executor_params.json on S3.

    Additionally — and unconditionally — produces a per-optimizer
    recommendation artifact at
    ``config/executor_params/recommendations/{date}/from_trigger_optimizer.json``
    via ``produce_artifact()``. Part of the optimizer-artifact-assembler
    arc; the artifact is additive (no behavior change to the legacy live
    write) and is consumed by the future assembler module.
    """
    # Always produce the artifact — captures every invocation for audit.
    # Thread the backfill run_date (config#1017) through to the artifact.
    produce_artifact(result, bucket, run_date=run_date)

    # Observe-only gate (alpha-engine-config-I12231): checked BEFORE the
    # cutover branch and before any S3 client exists, so the live key is
    # neither read nor written whichever writer is active.
    if apply_mode() != APPLY_MODE_LIVE:
        return {
            "applied": False,
            "reason": (
                "observe_only — trigger_optimizer.live_apply is not true "
                "(alpha-engine-config-I12231); recommendation recorded as a "
                "shadow artifact only"
            ),
            "would_disable": result.get("disabled_triggers", []),
        }

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
