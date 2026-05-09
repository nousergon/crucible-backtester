"""
pipeline_optimizer.py — team slot allocation + CIO fallback optimization.

4b: If a sector team consistently underperforms its sector over 8+ weeks,
    reduce its slot count. Writes to config/team_slots.json on S3.

4c: If CIO underperforms a simple score-ranking baseline over 8+ weeks,
    recommend switching to deterministic CIO mode. Writes cio_mode flag
    to config/research_params.json on S3.

Both items depend on end_to_end.py lift metrics being populated.
Min-data gate: 8 weeks of team/CIO lift data.
"""

import json
import logging
import math
from datetime import date

import boto3

logger = logging.getLogger(__name__)

S3_TEAM_SLOTS_KEY = "config/team_slots.json"
S3_RESEARCH_PARAMS_KEY = "config/research_params.json"

_MIN_WEEKS = 8
_NEGATIVE_LIFT_THRESHOLD = -0.005  # -0.5% avg lift → underperforming

# Symmetric with the absolute-lift threshold above. The original 0.0
# was much tighter than -0.005 and tripped on -0.06% noise (n=31, 8 wks)
# as observed in the 2026-05-09 evaluator email. Mirroring lets the
# vs-ranking gate fire only when the gap is at least the same scale.
_CIO_MIN_LIFT_TO_KEEP = -0.005  # CIO vs ranking must clear -0.5%

# Sample-size floor on n_advance. With 8 weeks of typical run cadence
# the prior gate accumulated ~30 ADVANCE picks — too few for the t-stat
# below to clear noise. 100 is a defensible "decision-grade" threshold
# (≈ 25 picks/wk × 4 wk equivalent of cumulative advance volume).
_MIN_ADVANCE_SAMPLES = 100

# Welch-style |t| threshold. 1.96 ≈ 95% two-sided for large df. We use
# this as a one-sided gate: only flip to deterministic when the lift is
# meaningfully NEGATIVE (t ≤ -1.96), not when it's noisy near zero.
_MIN_T_FOR_DETERMINISTIC = 1.96

DEFAULT_TEAM_SLOTS = {
    "technology": 3,
    "healthcare": 3,
    "financial": 3,
    "consumer": 3,
    "industrial": 3,
    "defensive": 3,
}

_cfg: dict = {}


def init_config(config: dict) -> None:
    global _cfg
    _cfg = config.get("pipeline_optimizer", {})


def analyze_team_performance(e2e_lift: dict) -> dict:
    """
    Analyze sector team lift and recommend slot adjustments.

    Args:
        e2e_lift: dict from end_to_end.compute_lift_metrics()

    Returns:
        dict with per-team analysis and slot recommendations.
    """
    if not e2e_lift or e2e_lift.get("status") != "ok":
        return {"status": "insufficient_data", "note": "No lift metrics available"}

    team_lift = e2e_lift.get("team_lift")
    if not team_lift:
        return {"status": "insufficient_data", "note": "No team lift data"}

    # `team_lift` is always a list[dict] after the producer-side
    # normalization in end_to_end._team_lift (#13). The old dict-handling
    # branch wrapped a status dict as `[status_dict]`, which then tried
    # to grade a ghost "team" with None lift — semantically wrong.
    teams = team_lift

    n_dates = e2e_lift.get("n_dates", 0)
    min_weeks = _cfg.get("min_weeks", _MIN_WEEKS)
    if n_dates < min_weeks:
        return {
            "status": "insufficient_data",
            "n_weeks": n_dates,
            "min_required": min_weeks,
        }

    threshold = _cfg.get("negative_lift_threshold", _NEGATIVE_LIFT_THRESHOLD)
    team_analysis = []

    for t in teams:
        team_id = t.get("team_id", "unknown")
        lift = t.get("lift")
        lift_vs_quant = t.get("lift_vs_quant")
        n_picks = t.get("n_picks", 0)

        if lift is None:
            assessment = "no_data"
            slot_change = 0
        elif lift < threshold:
            assessment = "underperforming"
            slot_change = -1
        elif lift > abs(threshold):
            assessment = "outperforming"
            slot_change = 1
        else:
            assessment = "neutral"
            slot_change = 0

        team_analysis.append({
            "team_id": team_id,
            "lift_vs_sector": lift,
            "lift_vs_quant": lift_vs_quant,
            "n_picks": n_picks,
            "assessment": assessment,
            "recommended_slot_change": slot_change,
        })

    return {
        "status": "ok",
        "n_weeks": n_dates,
        "team_analysis": team_analysis,
    }


def recommend_team_slots(analysis: dict, current_slots: dict | None = None) -> dict:
    """Generate recommended team slot allocation."""
    if analysis.get("status") != "ok":
        return {"status": analysis.get("status", "error")}

    if current_slots is None:
        current_slots = DEFAULT_TEAM_SLOTS.copy()

    recommended = dict(current_slots)
    changes = {}

    for ta in analysis.get("team_analysis", []):
        team_id = ta.get("team_id")
        change = ta.get("recommended_slot_change", 0)
        if team_id in recommended and change != 0:
            old = recommended[team_id]
            new_val = max(1, min(old + change, 5))  # clamp to [1, 5]
            if new_val != old:
                recommended[team_id] = new_val
                changes[team_id] = new_val - old

    if not changes:
        return {"status": "no_change", "current_slots": current_slots}

    return {
        "status": "ok",
        "current_slots": current_slots,
        "recommended_slots": recommended,
        "changes": changes,
    }


def apply_team_slots(result: dict, bucket: str) -> dict:
    """Write team slot allocation to S3."""
    if result.get("status") != "ok":
        return {"applied": False, "reason": f"status={result.get('status')}"}

    payload = {
        **result.get("recommended_slots", {}),
        "updated_at": str(date.today()),
    }

    s3 = boto3.client("s3")
    body = json.dumps(payload, indent=2)
    s3.put_object(Bucket=bucket, Key=S3_TEAM_SLOTS_KEY, Body=body, ContentType="application/json")
    logger.info("Team slots updated in S3: %s", result.get("changes"))

    return {"applied": True, "slots": result.get("recommended_slots"), "changes": result.get("changes")}


def _welch_t_stat(
    advance_avg: float | None,
    all_recs_avg: float | None,
    advance_std: float | None,
    all_recs_std: float | None,
    n_advance: int | None,
    n_recs: int | None,
) -> float | None:
    """Welch's two-sample t-stat for ``advance vs all_recs``.

    Returns None if any input is missing or undefined (n<2, zero
    variance with mismatched means, etc.). The consumer treats a None
    as "confidence check unavailable" and refuses to flip — matching
    the executor optimizer's PSR-unavailable behavior.
    """
    if (
        advance_avg is None or all_recs_avg is None
        or advance_std is None or all_recs_std is None
        or n_advance is None or n_recs is None
        or n_advance < 2 or n_recs < 2
    ):
        return None
    var_a = advance_std ** 2
    var_r = all_recs_std ** 2
    se = math.sqrt(var_a / n_advance + var_r / n_recs)
    if se == 0:
        return None
    return (advance_avg - all_recs_avg) / se


def analyze_cio_performance(e2e_lift: dict) -> dict:
    """
    Analyze CIO lift vs score-ranking baseline.

    Recommends switching to deterministic CIO mode only when:
      - n_dates ≥ MIN_WEEKS                           (calendar coverage)
      - n_advance ≥ MIN_ADVANCE_SAMPLES               (sample-size floor)
      - lift < NEGATIVE_LIFT_THRESHOLD                (effect-size floor)
      - vs-ranking lift < CIO_MIN_LIFT_TO_KEEP        (symmetric gate)
      - Welch t-stat ≤ -MIN_T_FOR_DETERMINISTIC       (significance gate)

    The previous version flipped on ``cio_vs_ranking < 0.0`` with no
    sample-size or significance gate — tripping on -0.06% noise from
    n_advance=31 in the 2026-05-09 evaluator email. The four gates
    here mirror the executor optimizer's PSR-confidence pattern
    (PR #168) on the agent-grading axis instead of the param-sweep axis.
    """
    if not e2e_lift or e2e_lift.get("status") != "ok":
        return {"status": "insufficient_data", "note": "No lift metrics available"}

    cio_lift = e2e_lift.get("cio_lift")
    cio_vs_ranking = e2e_lift.get("cio_vs_ranking")

    n_dates = e2e_lift.get("n_dates", 0)
    min_weeks = _cfg.get("min_weeks", _MIN_WEEKS)
    if n_dates < min_weeks:
        return {
            "status": "insufficient_data",
            "n_weeks": n_dates,
            "min_required": min_weeks,
        }

    # CIO lift: ADVANCE vs all recommendations
    cio_lift_val: float | None = None
    n_advance: int | None = None
    n_recs: int | None = None
    advance_avg: float | None = None
    all_recs_avg: float | None = None
    advance_std: float | None = None
    all_recs_std: float | None = None
    if isinstance(cio_lift, dict):
        cio_lift_val = cio_lift.get("lift")
        n_advance = cio_lift.get("n_advance")
        n_recs = cio_lift.get("n_recs")
        advance_avg = cio_lift.get("advance_avg")
        all_recs_avg = cio_lift.get("all_recs_avg")
        advance_std = cio_lift.get("advance_std_5d")
        all_recs_std = cio_lift.get("all_recs_std_5d")
    elif isinstance(cio_lift, (int, float)):
        cio_lift_val = float(cio_lift)

    # CIO vs ranking baseline
    ranking_lift_val: float | None = None
    if isinstance(cio_vs_ranking, dict):
        ranking_lift_val = cio_vs_ranking.get("lift")
    elif isinstance(cio_vs_ranking, (int, float)):
        ranking_lift_val = float(cio_vs_ranking)

    min_lift = _cfg.get("cio_min_lift", _CIO_MIN_LIFT_TO_KEEP)
    min_advance = _cfg.get("min_advance_samples", _MIN_ADVANCE_SAMPLES)
    min_t = _cfg.get("min_t_for_deterministic", _MIN_T_FOR_DETERMINISTIC)
    neg_lift_threshold = _cfg.get(
        "negative_lift_threshold", _NEGATIVE_LIFT_THRESHOLD,
    )

    # Sample-size gate: don't flip on small samples regardless of lift sign.
    if n_advance is not None and n_advance < min_advance:
        return {
            "status": "insufficient_advance_samples",
            "n_weeks": n_dates,
            "n_advance": n_advance,
            "min_required": min_advance,
            "cio_lift": cio_lift_val,
            "cio_vs_ranking_lift": ranking_lift_val,
            "recommendation": "keep_llm",
            "reasoning": (
                f"n_advance={n_advance} below min={min_advance} — "
                f"insufficient sample to flip CIO mode"
            ),
        }

    # Significance gate: lift must be statistically distinguishable from
    # zero at the negative tail. Returns None if stdev/n inputs missing
    # (legacy artifacts pre-dating the std emission); we then treat the
    # confidence check as unavailable and decline to flip.
    t_stat = _welch_t_stat(
        advance_avg=advance_avg,
        all_recs_avg=all_recs_avg,
        advance_std=advance_std,
        all_recs_std=all_recs_std,
        n_advance=n_advance,
        n_recs=n_recs,
    )

    # Effect-size gates: lift must be meaningfully negative on BOTH the
    # absolute-lift axis and the vs-ranking axis. Symmetric thresholds
    # (both at -0.005 by default) avoid the prior asymmetry where
    # vs-ranking ≤ 0.0 was a much tighter bar than absolute ≤ -0.005.
    lift_negative = (
        cio_lift_val is not None and cio_lift_val < neg_lift_threshold
    )
    ranking_negative = (
        ranking_lift_val is not None and ranking_lift_val < min_lift
    )

    # Confidence gate: require |t| ≥ threshold AND t < 0 (negative tail).
    confidence_ok = (
        t_stat is not None and t_stat <= -min_t
    )

    should_fallback = lift_negative and ranking_negative and confidence_ok

    if not should_fallback and not confidence_ok and t_stat is not None:
        # Effect could be in either tail or near zero — surface for ops.
        confidence_note = (
            f"t={t_stat:.2f} above significance gate (|t|≥{min_t})"
        )
    elif t_stat is None:
        confidence_note = "t-stat unavailable (missing stdev or n inputs)"
    else:
        confidence_note = f"t={t_stat:.2f} (significance gate met)"

    return {
        "status": "ok",
        "n_weeks": n_dates,
        "n_advance": n_advance,
        "cio_lift": cio_lift_val,
        "cio_vs_ranking_lift": ranking_lift_val,
        "t_stat": round(t_stat, 3) if t_stat is not None else None,
        "recommendation": "deterministic" if should_fallback else "keep_llm",
        "reasoning": (
            f"n_advance={n_advance}, lift={cio_lift_val}, "
            f"vs ranking={ranking_lift_val}, {confidence_note} — "
            + ("underperforming with significance, recommend deterministic"
               if should_fallback
               else "below confidence threshold, keeping LLM mode")
        ),
    }


def apply_cio_mode(result: dict, bucket: str) -> dict:
    """Write cio_mode to research_params.json on S3."""
    if result.get("status") != "ok":
        return {"applied": False, "reason": f"status={result.get('status')}"}

    if result.get("recommendation") != "deterministic":
        return {"applied": False, "reason": "CIO performing adequately — keeping LLM mode"}

    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=S3_RESEARCH_PARAMS_KEY)
        current = json.loads(obj["Body"].read())
    except Exception:
        current = {}

    if current.get("cio_mode") == "deterministic":
        return {"applied": False, "reason": "already in deterministic mode"}

    current["cio_mode"] = "deterministic"
    current["cio_mode_updated_at"] = str(date.today())
    current["cio_mode_reason"] = result.get("reasoning")

    body = json.dumps(current, indent=2)
    s3.put_object(Bucket=bucket, Key=S3_RESEARCH_PARAMS_KEY, Body=body, ContentType="application/json")
    logger.info("CIO mode set to deterministic in S3")

    return {"applied": True, "mode": "deterministic"}
