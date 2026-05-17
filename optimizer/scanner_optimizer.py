"""
scanner_optimizer.py — auto-relax scanner filter thresholds if leakage is high.

Reads scanner_evaluations + universe_returns from research.db to compute
filter leakage (% of rejected stocks that would have beaten SPY).
If leakage is consistently high over 8+ weeks, recommends relaxed thresholds.

Writes to config/scanner_params.json on S3. Research module reads this at
cold-start to override hardcoded filter thresholds.

Min-data gate: requires >= 8 weeks of scanner_evaluations data.
"""

import json
import logging
import sqlite3
from datetime import date

from alpha_engine_lib.eval_artifacts import (
    eval_artifact_key,
    eval_latest_key,
    new_eval_run_id,
)

import boto3
import pandas as pd

logger = logging.getLogger(__name__)

S3_PARAMS_KEY = "config/scanner_params.json"

_MIN_WEEKS = 8
_LEAKAGE_THRESHOLD = 0.15  # if >15% of rejected stocks beat SPY, filter is too tight
_MAX_RELAXATION_PCT = 0.20  # max 20% relaxation per iteration
_BLEND_FACTOR = 0.30        # conservative: 30% data-driven, 70% current

FACTORY_DEFAULTS = {
    "tech_score_min": 60,
    "max_atr_pct": 8.0,
    "min_avg_volume": 500_000,
    "min_price": 10.0,
    "momentum_top_n": 60,
    "deep_value_max_atr_pct": 12.0,
    "max_debt_to_equity": 3.0,
    "min_current_ratio": 0.5,
}

_cfg: dict = {}


def init_config(config: dict) -> None:
    global _cfg
    _cfg = config.get("scanner_optimizer", {})


def read_current_params(bucket: str) -> dict:
    """Read current scanner params from S3, falling back to factory defaults."""
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=S3_PARAMS_KEY)
        data = json.loads(obj["Body"].read())
        params = {k: data[k] for k in FACTORY_DEFAULTS if k in data}
        if params:
            logger.info("Scanner params from S3 (updated %s): %s",
                        data.get("updated_at", "unknown"), params)
            return {**FACTORY_DEFAULTS, **params}
    except Exception as e:
        logger.info("No scanner params in S3 (%s), using factory defaults", e)
    return FACTORY_DEFAULTS.copy()


def read_params_as_of(bucket: str, as_of_date) -> dict:
    """Point-in-time sibling of :func:`read_current_params` (PIT walk-forward,
    ROADMAP L2371 / Backtester Phase 3).

    Resolves the scanner-params snapshot whose knowledge time ≤
    ``as_of_date``. No eligible snapshot → genesis ``FACTORY_DEFAULTS``,
    **never** a later snapshot (no-future-fallback, plan §3 / D3). Return
    shape mirrors :func:`read_current_params` exactly.
    """
    from optimizer.config_archive import resolve_as_of

    data = resolve_as_of(bucket, "scanner_params", as_of_date)
    if not data:
        return FACTORY_DEFAULTS.copy()
    params = {k: data[k] for k in FACTORY_DEFAULTS if k in data}
    return {**FACTORY_DEFAULTS, **params}


def analyze(research_db_path: str) -> dict:
    """
    Compute scanner filter leakage from scanner_evaluations + universe_returns.

    Returns dict with leakage metrics, per-gate analysis, and recommendations.
    """
    min_weeks = _cfg.get("min_weeks", _MIN_WEEKS)
    leakage_threshold = _cfg.get("leakage_threshold", _LEAKAGE_THRESHOLD)

    try:
        conn = sqlite3.connect(research_db_path)

        # Check if tables exist
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]

        if "scanner_evaluations" not in tables or "universe_returns" not in tables:
            conn.close()
            return {
                "status": "insufficient_data",
                "note": "scanner_evaluations or universe_returns table not found",
            }

        se = pd.read_sql_query("SELECT * FROM scanner_evaluations", conn)
        ur = pd.read_sql_query(
            "SELECT ticker, eval_date, return_5d, beat_spy_5d FROM universe_returns",
            conn,
        )
        conn.close()
    except Exception as e:
        return {"status": "error", "error": str(e)}

    if se.empty or ur.empty:
        return {"status": "insufficient_data", "note": "Empty tables"}

    n_weeks = se["eval_date"].nunique() if "eval_date" in se.columns else 0
    if n_weeks < min_weeks:
        return {
            "status": "insufficient_data",
            "n_weeks": n_weeks,
            "min_required": min_weeks,
        }

    # Join scanner evaluations with universe returns
    merged = se.merge(ur, on=["ticker", "eval_date"], how="left")

    # Separate passed vs rejected
    if "quant_filter_pass" not in merged.columns:
        return {"status": "insufficient_data", "note": "quant_filter_pass column missing"}

    passed = merged[merged["quant_filter_pass"] == 1]
    rejected = merged[merged["quant_filter_pass"] == 0]

    if rejected.empty:
        return {"status": "ok", "leakage_rate": 0.0, "note": "No rejected stocks",
                "recommendations": [], "n_weeks": n_weeks}

    # Overall leakage: % of rejected stocks that beat SPY
    rejected_with_returns = rejected[rejected["beat_spy_5d"].notna()]
    if rejected_with_returns.empty:
        return {"status": "insufficient_data", "note": "No resolved returns for rejected stocks"}

    leakage_rate = float(rejected_with_returns["beat_spy_5d"].mean())

    # Per-gate leakage (which filter gates reject the most winners?)
    gate_cols = [c for c in ["liquidity_pass", "volatility_gate_pass", "tech_score"]
                 if c in merged.columns]
    gate_analysis = []
    for gate in gate_cols:
        if gate == "tech_score":
            gate_rejected = merged[merged[gate] < 60] if gate in merged.columns else pd.DataFrame()
        else:
            gate_rejected = merged[merged[gate] == 0]

        if not gate_rejected.empty and "beat_spy_5d" in gate_rejected.columns:
            resolved = gate_rejected[gate_rejected["beat_spy_5d"].notna()]
            if not resolved.empty:
                gate_leakage = float(resolved["beat_spy_5d"].mean())
                gate_analysis.append({
                    "gate": gate,
                    "n_rejected": len(resolved),
                    "leakage_rate": round(gate_leakage, 4),
                    "avg_return_5d": round(float(resolved["return_5d"].mean()), 4)
                    if "return_5d" in resolved.columns else None,
                })

    # Filter lift: passing stocks vs all 900
    pass_avg = float(passed["return_5d"].mean()) if not passed.empty and "return_5d" in passed.columns else None
    all_avg = float(merged["return_5d"].mean()) if "return_5d" in merged.columns else None
    filter_lift = (pass_avg - all_avg) if pass_avg is not None and all_avg is not None else None

    recommendations = []
    if leakage_rate > leakage_threshold:
        # Recommend relaxing the leakiest gates
        for ga in sorted(gate_analysis, key=lambda x: x["leakage_rate"], reverse=True):
            if ga["leakage_rate"] > leakage_threshold:
                recommendations.append({
                    "gate": ga["gate"],
                    "current_leakage": ga["leakage_rate"],
                    "action": "relax",
                })

    return {
        "status": "ok",
        "n_weeks": n_weeks,
        "n_total_stocks": len(merged),
        "n_passed": len(passed),
        "n_rejected": len(rejected),
        "leakage_rate": round(leakage_rate, 4),
        "leakage_threshold": leakage_threshold,
        "filter_lift": round(filter_lift, 4) if filter_lift is not None else None,
        "gate_analysis": gate_analysis,
        "recommendations": recommendations,
        "high_leakage": leakage_rate > leakage_threshold,
    }


def recommend(analysis_result: dict, current_params: dict) -> dict:
    """Generate relaxed parameter recommendations based on leakage analysis."""
    if analysis_result.get("status") != "ok":
        return {"status": analysis_result.get("status", "error"), **analysis_result}

    if not analysis_result.get("high_leakage"):
        return {
            "status": "no_change",
            "note": f"Leakage rate {analysis_result.get('leakage_rate', 0):.1%} "
                    f"below threshold {analysis_result.get('leakage_threshold', 0):.1%}",
            "current_params": current_params,
        }

    blend = _cfg.get("blend_factor", _BLEND_FACTOR)
    max_relax = _cfg.get("max_relaxation_pct", _MAX_RELAXATION_PCT)
    recommended = dict(current_params)

    gate_recommendations = analysis_result.get("recommendations", [])
    changes = {}

    for rec in gate_recommendations:
        gate = rec.get("gate")
        if gate == "tech_score" and "tech_score_min" in recommended:
            old = recommended["tech_score_min"]
            target = old * (1 - max_relax)  # lower threshold
            new_val = round(old * (1 - blend) + target * blend)
            new_val = max(new_val, 40)  # floor
            recommended["tech_score_min"] = new_val
            changes["tech_score_min"] = new_val - old

        elif gate == "volatility_gate_pass" and "max_atr_pct" in recommended:
            old = recommended["max_atr_pct"]
            target = old * (1 + max_relax)  # raise threshold
            new_val = round(old * (1 - blend) + target * blend, 1)
            new_val = min(new_val, 15.0)  # ceiling
            recommended["max_atr_pct"] = new_val
            changes["max_atr_pct"] = round(new_val - old, 1)

        elif gate == "liquidity_pass" and "min_avg_volume" in recommended:
            old = recommended["min_avg_volume"]
            target = old * (1 - max_relax)  # lower threshold
            new_val = int(old * (1 - blend) + target * blend)
            new_val = max(new_val, 100_000)  # floor
            recommended["min_avg_volume"] = new_val
            changes["min_avg_volume"] = new_val - old

    if not any(abs(v) > 0 for v in changes.values()):
        return {"status": "no_change", "note": "No meaningful changes", "current_params": current_params}

    return {
        "status": "ok",
        "current_params": current_params,
        "recommended_params": recommended,
        "changes": changes,
        "leakage_rate": analysis_result.get("leakage_rate"),
        "n_weeks": analysis_result.get("n_weeks"),
    }


def apply(result: dict, bucket: str) -> dict:
    """Write recommended scanner params to S3."""
    if result.get("status") != "ok":
        return {"applied": False, "reason": f"status={result.get('status')}"}

    recommended = result.get("recommended_params", {})
    if not recommended:
        return {"applied": False, "reason": "no recommended params"}

    payload = {
        **recommended,
        "updated_at": str(date.today()),
        "leakage_rate": result.get("leakage_rate"),
        "n_weeks": result.get("n_weeks"),
    }

    s3 = boto3.client("s3")
    body = json.dumps(payload, indent=2)
    s3.put_object(Bucket=bucket, Key=S3_PARAMS_KEY, Body=body, ContentType="application/json")
    logger.info("Scanner params updated in S3: %s", result.get("changes"))

    # Canonical eval-style archive layout per lib v0.8.0
    run_id = new_eval_run_id()
    history_prefix = "config/scanner_params_history"
    history_key = eval_artifact_key(history_prefix, run_id)
    history_latest_key = eval_latest_key(history_prefix)
    s3.put_object(Bucket=bucket, Key=history_key, Body=body, ContentType="application/json")
    s3.put_object(
        Bucket=bucket, Key=history_latest_key, Body=body,
        ContentType="application/json",
    )

    # Bitemporal knowledge-time index for PIT walk-forward resolution
    # (best-effort, never fatal — live + history already durable). plan §D3.
    from optimizer.config_archive import record_apply
    record_apply(
        bucket, "scanner_params",
        history_key=history_key,
        knowledge_date=payload["updated_at"],
        run_id=run_id,
        s3_client=s3,
    )

    return {"applied": True, "params": recommended, "changes": result.get("changes")}
