"""
stance_sizing_optimizer.py — offline-IC gate for stance-conditional sizing (L300).

The sibling of ``predictor_sizing_optimizer`` (p_up) and ``barrier_sizing_optimizer``
(barrier_win_prob). Those tune CONTINUOUS per-ticker sizing signals; this tunes
the CATEGORICAL ``stance`` sizing multipliers (``stance_size_{momentum,value,
quality,catalyst}``).

WHY OFFLINE (L300, 2026-06-01): the backtester sim runs with
``predictions_by_ticker={}`` → per-ticker ``stance`` is always None → the
executor's ``stance_adj`` resolves to 1.0 → grid-sweeping these multipliers in
``param_sweep`` was a SILENT NO-OP. The institutional fix is to tune them OFFLINE
against realized per-stance alpha (the ``by_stance`` cohort surfaced by
``analysis.signal_quality``), exactly as p_up/barrier sizing are tuned against
realized outcomes — NOT a predictionless grid sweep.

METHOD: read ``score_performance`` (research.db); per stance, compute realized
mean alpha (canonical ``log_alpha_21d``) + rolling-week consistency. Anchor
on the executor's thesis-ordered FACTORY defaults (momentum 1.0 ≥ quality 0.8 ≥
value 0.7 ≥ catalyst 0.6 — higher-uncertainty theses get smaller stakes) and
NUDGE each qualifying stance toward/away by its realized alpha relative to the
book, bounded to ``[size_floor, size_cap]``. Stances that don't clear the
sample/consistency gate keep their FACTORY default. Emits a ``field_overlay``
RecommendationArtifact (cutover-gated via the assembler), mirroring barrier.

ACTIVATION PREREQUISITE: ``score_performance`` must carry a ``stance`` column
(the research.db migration that joins predictions.stance into realized
outcomes). Until then ``analyze`` returns ``stance_column_absent`` — informational,
not an error (mirrors barrier's ``barrier_win_prob_column_absent``). See the
L300 follow-up + analysis/signal_quality.py's graceful-degrade ``by_stance``.
"""

import json
import logging
import sqlite3
from datetime import date

import boto3
import pandas as pd

logger = logging.getLogger(__name__)

S3_PARAMS_KEY = "config/executor_params.json"

_STANCES = ("momentum", "value", "quality", "catalyst")

_MIN_SAMPLES_PER_STANCE = 30   # resolved rows per stance before it can move
_MIN_STANCE_SPREAD = 0.005     # max−min per-stance alpha must exceed this to act
_MIN_POSITIVE_WEEKS = 6
_ROLLING_WEEKS = 8
_SENSITIVITY = 2.0             # nudge gain on relative alpha
_SIZE_FLOOR = 0.4
_SIZE_CAP = 1.1

_cfg: dict = {}


def init_config(config: dict) -> None:
    global _cfg
    _cfg = config.get("stance_sizing_optimizer", {})


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


def _factory_defaults() -> dict:
    """The executor's thesis-ordered stance defaults (single source: the
    backtester FACTORY_DEFAULTS, which the test suite pins == executor)."""
    from optimizer.executor_optimizer import FACTORY_DEFAULTS
    return {s: float(FACTORY_DEFAULTS[f"stance_size_{s}"]) for s in _STANCES}


def analyze(research_db_path: str) -> dict:
    """Compute realized per-stance alpha + a bounded multiplier recommendation.

    Returns a dict with per-stance metrics + an enable/keep_disabled
    recommendation, or an informational status when stance isn't recorded yet.
    """
    min_samples = _cfg.get("min_samples_per_stance", _MIN_SAMPLES_PER_STANCE)
    min_spread = _cfg.get("min_stance_spread", _MIN_STANCE_SPREAD)

    try:
        conn = sqlite3.connect(research_db_path)
        if not _column_exists(conn, "score_performance", "stance"):
            conn.close()
            return {
                "status": "stance_column_absent",
                "note": (
                    "score_performance has no stance column yet — awaiting the "
                    "research.db migration that joins predictions.stance into "
                    "realized outcomes (L300 activation prerequisite)."
                ),
            }
        # config#1452 + config#1451: the prior query selected `prediction_date`
        # (which score_performance does NOT have — it has `score_date`) AND the
        # retired 10d outcome (`return_10d`/`spy_10d_return`, dark since April).
        # Use the canonical `log_alpha_21d` (21d log-domain market-relative alpha)
        # keyed by `score_date`.
        df = pd.read_sql_query(
            "SELECT score_date, symbol, stance, log_alpha_21d "
            "FROM score_performance "
            "WHERE stance IS NOT NULL "
            "  AND log_alpha_21d IS NOT NULL "
            "ORDER BY score_date",
            conn,
        )
        conn.close()
    except Exception as e:
        return {"status": "error", "error": str(e)}

    if df.empty:
        return {"status": "insufficient_stance_history", "n_samples": 0}

    df["alpha_21d"] = df["log_alpha_21d"].astype(float)
    overall_alpha = float(df["alpha_21d"].mean())
    df["year_week"] = (
        pd.to_datetime(df["score_date"]).dt.year.astype(str) + "-W"
        + pd.to_datetime(df["score_date"]).dt.isocalendar().week.astype(int).astype(str).str.zfill(2)
    )

    defaults = _factory_defaults()
    per_stance: dict[str, dict] = {}
    stance_alpha_samples: dict[str, list] = {}
    for stance in _STANCES:
        sub = df[df["stance"] == stance]
        n = len(sub)
        if n == 0:
            per_stance[stance] = {"n": 0, "qualifies": False, "mean_alpha": None,
                                  "recent_positive_weeks": 0}
            continue
        mean_alpha = float(sub["alpha_21d"].mean())
        stance_alpha_samples[stance] = sub["alpha_21d"].to_numpy()
        weekly = [
            float(g["alpha_21d"].mean())
            for _, g in sub.groupby("year_week") if len(g) >= 3
        ]
        rolling = _cfg.get("rolling_weeks", _ROLLING_WEEKS)
        recent = weekly[-rolling:] if len(weekly) >= rolling else weekly
        recent_pos = sum(1 for a in recent if a > 0)
        recent_neg = sum(1 for a in recent if a < 0)
        min_consistent = min(_cfg.get("min_positive_weeks", _MIN_POSITIVE_WEEKS), len(recent))
        # Qualify on CONSISTENCY OF SIGN, not positivity — a reliably-NEGATIVE
        # stance is equally informative (it should be sized DOWN). The
        # multiplier nudge direction below follows the sign of the alpha.
        qualifies = (
            n >= min_samples
            and len(recent) > 0
            and max(recent_pos, recent_neg) >= min_consistent
        )
        per_stance[stance] = {
            "n": n, "mean_alpha": round(mean_alpha, 6),
            "recent_positive_weeks": recent_pos, "recent_negative_weeks": recent_neg,
            "recent_total_weeks": len(recent), "qualifies": bool(qualifies),
        }

    qualifying = {s: v for s, v in per_stance.items() if v.get("qualifies")}
    alphas = [v["mean_alpha"] for v in qualifying.values() if v["mean_alpha"] is not None]
    spread = (max(alphas) - min(alphas)) if len(alphas) >= 2 else 0.0
    should_enable = len(qualifying) >= 2 and spread >= min_spread

    # Bounded multiplier: anchor on the thesis default, nudge by realized alpha
    # relative to the book. Non-qualifying stances keep their default.
    sensitivity = _cfg.get("sensitivity", _SENSITIVITY)
    floor = _cfg.get("size_floor", _SIZE_FLOOR)
    cap = _cfg.get("size_cap", _SIZE_CAP)
    scale = abs(overall_alpha) if abs(overall_alpha) > 1e-6 else (max(abs(a) for a in alphas) if alphas else 1.0)
    recommended: dict[str, float] = {}
    for stance in _STANCES:
        base = defaults[stance]
        v = per_stance[stance]
        if should_enable and v.get("qualifies") and v.get("mean_alpha") is not None:
            rel = (v["mean_alpha"] - overall_alpha) / scale if scale else 0.0
            mult = base * (1.0 + sensitivity * rel)
            recommended[stance] = round(min(max(mult, floor), cap), 4)
        else:
            recommended[stance] = round(base, 4)

    # Observe-first significance verdict (config#1426 Phase 3). NON-ENFORCING:
    # the gate enables stance multipliers on a per-stance alpha SPREAD >= 0.005
    # with no significance test; this asks whether the best vs worst QUALIFYING
    # stance's mean alpha is statistically distinguishable (two-sample bootstrap
    # CI of the mean difference excludes zero). NEVER changes the recommendation.
    # Swallow rationale ("fail loud" carve-out): (a) failure = observe
    # instrumentation error on a SECONDARY path; the recommendation is
    # unaffected; (c) recorded surface = WARN log.
    significance_observe = None
    if bool(_cfg.get("significance_observe_enabled", True)):
        try:
            from optimizer.significance_observe import observe_stance_spread
            qualifying_samples = {
                s: stance_alpha_samples[s]
                for s in qualifying if s in stance_alpha_samples
            }
            significance_observe = observe_stance_spread(qualifying_samples, cfg=_cfg)
        except Exception as e:  # observe-only: must not break the optimizer
            logger.warning(
                "stance_sizing significance_observe failed (non-fatal, observe-only): %s", e,
            )

    return {
        "status": "ok",
        "n_samples": len(df),
        "overall_alpha_21d": round(overall_alpha, 6),
        "per_stance": per_stance,
        "stance_alpha_spread": round(spread, 6),
        "recommended_multipliers": recommended,
        "recommendation": "enable" if should_enable else "keep_disabled",
        "significance_observe": significance_observe,
    }


def _build_overlay_params(result: dict) -> tuple[dict, list[str]]:
    """The field_overlay payload this optimizer recommends applying."""
    rec = result.get("recommended_multipliers", {})
    params = {f"stance_size_{s}": rec[s] for s in _STANCES if s in rec}
    params["stance_sizing_updated_at"] = str(date.today())
    params["stance_sizing_alpha_spread"] = result.get("stance_alpha_spread")
    return params, list(params.keys())


def produce_artifact(
    result: dict, bucket: str, run_id: str | None = None, run_date: str | None = None,
) -> dict:
    """Write a typed field_overlay RecommendationArtifact to S3 (full audit
    trail). Mirrors barrier_sizing_optimizer.produce_artifact."""
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
            for k in ("status", "recommendation", "n_samples", "overall_alpha_21d",
                      "stance_alpha_spread", "per_stance", "recommended_multipliers")
            if result.get(k) is not None
        }
        artifact = RecommendationArtifact(
            fit_target="stance_sizing_alpha",
            optimizer_name="stance_sizing_optimizer",
            # config#1017: explicit backfill run_date over ambient today_iso()
            # (None on a live run → current trading day).
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
            "Failed to write stance_sizing_optimizer recommendation artifact: "
            "%s (non-fatal)", e,
        )
        return {"written": False, "reason": str(e)}


def apply(result: dict, bucket: str, run_date: str | None = None) -> dict:
    """Write stance_size_* multipliers to executor_params.json on S3 (field
    overlay). Always produces the artifact first; honors the assembler cutover
    gate. Mirrors barrier_sizing_optimizer.apply."""
    produce_artifact(result, bucket, run_date=run_date)

    from optimizer.assembler import is_cutover_enabled
    if is_cutover_enabled():
        return {"applied": False, "reason": "cutover_mode — assembler is sole live writer"}

    if result.get("status") != "ok":
        return {"applied": False, "reason": f"status={result.get('status')}"}
    if result.get("recommendation") != "enable":
        return {"applied": False, "reason": f"spread/consistency insufficient "
                                            f"(spread={result.get('stance_alpha_spread')})"}

    # Significance ENFORCE gate (config#1426 Phase 4) — default OFF. Blocks the
    # live promotion when the best-vs-worst stance mean-alpha difference is not
    # statistically significant (two-sample bootstrap CI includes zero →
    # would_block), or the verdict couldn't be computed (conservative block).
    # Enforce can only BLOCK a promotion the live gate already allowed.
    if bool(_cfg.get("enforce_significance", False)):
        from optimizer.significance_observe import significance_would_block
        verdict = result.get("significance_observe")
        if significance_would_block(verdict):
            logger.info(
                "stance_sizing: significance enforce BLOCKED promotion (config#1426)"
            )
            return {
                "applied": False,
                "reason": "stance_sizing: blocked by significance enforce "
                          "(config#1426) — undefended evidence",
                "observe_verdict": verdict,
            }

    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=S3_PARAMS_KEY)
        current = json.loads(obj["Body"].read())
    except Exception:
        current = {}

    overlay_params, _ = _build_overlay_params(result)
    current.update(overlay_params)
    body = json.dumps(current, indent=2)
    s3.put_object(Bucket=bucket, Key=S3_PARAMS_KEY, Body=body, ContentType="application/json")
    logger.info("stance_size multipliers updated in S3: %s", result.get("recommended_multipliers"))
    return {"applied": True, "multipliers": result.get("recommended_multipliers")}
