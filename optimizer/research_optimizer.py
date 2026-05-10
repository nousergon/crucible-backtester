"""
research_optimizer.py — auto-tune research signal boost parameters.

Correlates signal boost values (short interest, institutional accumulation,
consistency thresholds) with beat_spy outcomes from score_performance.
Suggests revised params and applies them to S3 if guardrails pass.

Reads s3://{bucket}/config/research_params.json (current active params)
and writes optimized values back. The Research Lambda reads this file
at cold-start via config.get_research_params().

Param sweep space (all tunable via config.yaml research_optimizer section):
  - short_interest_buy_threshold_pct: [10, 15, 20, 25, 30]
  - short_interest_high_threshold_pct: [30, 35, 40, 45, 50]
  - short_interest_buy_boost: [1.0, 1.5, 2.0, 2.5, 3.0]
  - short_interest_high_boost: [2.0, 3.0, 4.0, 5.0, 6.0]
  - institutional_boost: [1.0, 2.0, 3.0, 4.0, 5.0]
  - institutional_min_funds: [2, 3, 4, 5]
  - consistency_bullish_dominance: [0.6, 0.65, 0.7, 0.75, 0.8]
  - consistency_bearish_dominance: [0.2, 0.25, 0.3, 0.35, 0.4]
  - consistency_low_score: [35, 40, 45]
  - consistency_high_score: [65, 70, 75]
"""

import json
import logging
from datetime import date

from alpha_engine_lib.eval_artifacts import (
    eval_artifact_key,
    eval_latest_key,
    new_eval_run_id,
)

import boto3
import pandas as pd
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

S3_PARAMS_KEY = "config/research_params.json"

# Factory defaults — match universe.yaml research_params section.
FACTORY_DEFAULTS = {
    "atr_period": 20,
    "short_interest_buy_threshold_pct": 20,
    "short_interest_high_threshold_pct": 40,
    "short_interest_buy_boost": 2.0,
    "short_interest_high_boost": 4.0,
    "institutional_min_funds": 3,
    "institutional_boost": 3.0,
    "consistency_bullish_dominance": 0.7,
    "consistency_bearish_dominance": 0.3,
    "consistency_low_score": 40,
    "consistency_high_score": 70,
}

# Params safe to auto-tune (all signal boost params).
# atr_period is excluded — it's a standard technical definition, not a scoring param.
SAFE_PARAMS = [
    "short_interest_buy_threshold_pct",
    "short_interest_high_threshold_pct",
    "short_interest_buy_boost",
    "short_interest_high_boost",
    "institutional_min_funds",
    "institutional_boost",
    "consistency_bullish_dominance",
    "consistency_bearish_dominance",
    "consistency_low_score",
    "consistency_high_score",
]

# Default sweep grid — override via config.yaml research_optimizer.sweep_grid
DEFAULT_SWEEP_GRID = {
    "short_interest_buy_threshold_pct": [10, 15, 20, 25, 30],
    "short_interest_high_threshold_pct": [30, 35, 40, 45, 50],
    "short_interest_buy_boost": [1.0, 1.5, 2.0, 2.5, 3.0],
    "short_interest_high_boost": [2.0, 3.0, 4.0, 5.0, 6.0],
    "institutional_boost": [1.0, 2.0, 3.0, 4.0, 5.0],
    "institutional_min_funds": [2, 3, 4, 5],
    "consistency_bullish_dominance": [0.6, 0.65, 0.7, 0.75, 0.8],
    "consistency_bearish_dominance": [0.2, 0.25, 0.3, 0.35, 0.4],
    "consistency_low_score": [35, 40, 45],
    "consistency_high_score": [65, 70, 75],
}

# ── Fallback defaults (override via research_optimizer section in config.yaml) ──
_MIN_SAMPLES = 200  # deferred until 6+ months of live data for reliable correlations
_MIN_IMPROVEMENT = 0.05  # 5% improvement in hit rate to recommend
_MAX_SINGLE_CHANGE_PCT = 0.50  # max 50% change in any single param value
_BLEND_FACTOR = 0.30  # conservative: 30% data-driven, 70% current

# Module-level config ref — set by init_config() from backtest.py
_cfg: dict = {}


def init_config(config: dict) -> None:
    """Load research_optimizer section from backtester config."""
    global _cfg
    _cfg = config.get("research_optimizer", {})


def read_current_params(bucket: str) -> dict:
    """
    Read current research params from S3, falling back to factory defaults.

    Returns dict of all research params (safe + non-safe).
    """
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=S3_PARAMS_KEY)
        data = json.loads(obj["Body"].read())
        params = {k: data[k] for k in FACTORY_DEFAULTS if k in data}
        if params:
            logger.info(
                "Current research params from S3 (updated %s): %s",
                data.get("updated_at", "unknown"), params,
            )
            return {**FACTORY_DEFAULTS, **params}
    except Exception as e:
        logger.info("No research params in S3 (%s), using factory defaults", e)

    return FACTORY_DEFAULTS.copy()


def compute_boost_correlations(
    df: pd.DataFrame,
    bucket: str,
    signals_prefix: str = "signals",
) -> dict:
    """
    Correlate individual signal boosts with beat_spy outcomes.

    Loads boost values (short_interest_adj, institutional_boost) from signals.json
    and computes Pearson correlation with beat_spy_10d / beat_spy_30d.

    Returns dict with per-boost correlation data and sample sizes.
    """
    if df is None or df.empty:
        return {"status": "insufficient_data", "note": "No score_performance data"}

    populated = df[df["beat_spy_10d"].notna()].copy()
    min_samples = _cfg.get("min_samples", _MIN_SAMPLES)
    if len(populated) < min_samples:
        return {
            "status": "insufficient_data",
            "n_samples": len(populated),
            "min_required": min_samples,
        }

    # Load boost values from signals.json
    dates = populated["score_date"].unique().tolist()
    s3 = boto3.client("s3")

    boost_cols = ["short_interest_adj", "institutional_boost"]
    rows = []

    for d in dates:
        key = f"{signals_prefix}/{d}/signals.json"
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj["Body"].read())
        except Exception:
            continue

        for stock in data.get("universe", []) + data.get("buy_candidates", []):
            ticker = stock.get("ticker") or stock.get("symbol")
            if ticker:
                row = {"symbol": ticker, "score_date": d}
                for col in boost_cols:
                    row[col] = stock.get(col, 0.0)
                rows.append(row)

    if not rows:
        return {"status": "no_boost_data", "note": "No boost values found in signals.json"}

    boost_df = pd.DataFrame(rows)
    merged = populated.merge(boost_df, on=["symbol", "score_date"], how="left")

    # Compute correlations
    correlations = {}
    for col in boost_cols:
        if col not in merged.columns:
            continue
        valid = merged[[col, "beat_spy_10d"]].dropna()
        nonzero = valid[valid[col] != 0]

        corr_10d = float(valid[col].corr(valid["beat_spy_10d"])) if len(valid) >= 20 else None

        valid_30d = merged[[col, "beat_spy_30d"]].dropna() if "beat_spy_30d" in merged.columns else pd.DataFrame()
        corr_30d = float(valid_30d[col].corr(valid_30d["beat_spy_30d"])) if len(valid_30d) >= 20 else None

        correlations[col] = {
            "corr_beat_spy_10d": round(corr_10d, 4) if corr_10d is not None else None,
            "corr_beat_spy_30d": round(corr_30d, 4) if corr_30d is not None else None,
            "n_total": len(valid),
            "n_nonzero": len(nonzero),
            "mean_when_nonzero": round(float(nonzero[col].mean()), 3) if len(nonzero) > 0 else None,
        }

    return {
        "status": "ok",
        "n_samples": len(populated),
        "correlations": correlations,
    }


def recommend(
    correlation_result: dict,
    current_params: dict,
) -> dict:
    """
    Recommend research param updates based on boost correlation analysis.

    Simple heuristic approach:
    - If a boost has positive correlation with beat_spy → keep or increase
    - If a boost has negative correlation → reduce
    - Apply conservative blend (30% data, 70% current)

    Args:
        correlation_result: dict from compute_boost_correlations()
        current_params: current research params from S3

    Returns:
        {
            "status": "ok" | "insufficient_data" | "no_improvement",
            "current_params": {...},
            "recommended_params": {...},
            "changes": {...},
        }
    """
    if correlation_result.get("status") != "ok":
        return {"status": correlation_result.get("status", "error"), **correlation_result}

    correlations = correlation_result.get("correlations", {})
    if not correlations:
        return {"status": "no_boost_data", "note": "No boost correlations available"}

    blend = _cfg.get("blend_factor", _BLEND_FACTOR)
    recommended = dict(current_params)

    # Short interest: if correlation is positive, boosts are working → keep/increase
    si_corr = correlations.get("short_interest_adj", {})
    si_c10 = si_corr.get("corr_beat_spy_10d")
    if si_c10 is not None and si_corr.get("n_nonzero", 0) >= 10:
        if si_c10 > 0.05:
            # Positive correlation — increase boosts slightly
            for key in ("short_interest_buy_boost", "short_interest_high_boost"):
                current_val = current_params.get(key, FACTORY_DEFAULTS[key])
                target = current_val * 1.15
                recommended[key] = round(current_val * (1 - blend) + target * blend, 2)
        elif si_c10 < -0.05:
            # Negative correlation — reduce boosts
            for key in ("short_interest_buy_boost", "short_interest_high_boost"):
                current_val = current_params.get(key, FACTORY_DEFAULTS[key])
                target = current_val * 0.85
                recommended[key] = round(current_val * (1 - blend) + target * blend, 2)

    # Institutional boost: same logic
    inst_corr = correlations.get("institutional_boost", {})
    inst_c10 = inst_corr.get("corr_beat_spy_10d")
    if inst_c10 is not None and inst_corr.get("n_nonzero", 0) >= 10:
        if inst_c10 > 0.05:
            current_val = current_params.get("institutional_boost", FACTORY_DEFAULTS["institutional_boost"])
            target = current_val * 1.15
            recommended["institutional_boost"] = round(current_val * (1 - blend) + target * blend, 2)
        elif inst_c10 < -0.05:
            current_val = current_params.get("institutional_boost", FACTORY_DEFAULTS["institutional_boost"])
            target = current_val * 0.85
            recommended["institutional_boost"] = round(current_val * (1 - blend) + target * blend, 2)

    # Compute changes
    changes = {}
    any_meaningful = False
    max_change_pct = _cfg.get("max_single_change_pct", _MAX_SINGLE_CHANGE_PCT)

    for key in SAFE_PARAMS:
        old_val = current_params.get(key, FACTORY_DEFAULTS.get(key, 0))
        new_val = recommended.get(key, old_val)
        if old_val != 0:
            change_pct = abs(new_val - old_val) / abs(old_val)
            if change_pct > max_change_pct:
                # Clamp to max change
                direction = 1 if new_val > old_val else -1
                recommended[key] = round(old_val * (1 + direction * max_change_pct), 4)
                new_val = recommended[key]
        delta = round(new_val - old_val, 4)
        changes[key] = delta
        if abs(delta) > 0.001:
            any_meaningful = True

    if not any_meaningful:
        return {
            "status": "no_improvement",
            "current_params": current_params,
            "recommended_params": recommended,
            "changes": changes,
            "correlations": correlations,
            "note": "No meaningful changes recommended — current params are near-optimal",
        }

    return {
        "status": "ok",
        "current_params": current_params,
        "recommended_params": {k: v for k, v in recommended.items() if k in SAFE_PARAMS or k in FACTORY_DEFAULTS},
        "changes": changes,
        "correlations": correlations,
        "n_samples": correlation_result.get("n_samples"),
        "note": f"Recommended changes based on {correlation_result.get('n_samples', 0)} signals",
    }


def apply(result: dict, bucket: str) -> dict:
    """
    Write recommended research params to S3 if recommendation is valid.

    Writes to s3://{bucket}/config/research_params.json and archives
    to config/research_params_history/{date}.json.
    """
    if result.get("status") != "ok":
        return {"applied": False, "reason": f"status={result.get('status')}"}

    recommended = result.get("recommended_params", {})
    if not recommended:
        return {"applied": False, "reason": "no recommended params"}

    payload = {
        **recommended,
        "updated_at": str(date.today()),
        "n_samples": result.get("n_samples"),
        "correlations": result.get("correlations"),
    }

    from optimizer.rollback import save_previous
    save_previous(bucket, "research_params")

    s3 = boto3.client("s3")
    body = json.dumps(payload, indent=2)

    s3.put_object(Bucket=bucket, Key=S3_PARAMS_KEY, Body=body, ContentType="application/json")
    logger.info("Research params updated in S3: %s", {k: v for k, v in recommended.items() if k in SAFE_PARAMS})

    # Canonical eval-style archive layout per lib v0.8.0
    run_id = new_eval_run_id()
    history_prefix = "config/research_params_history"
    history_key = eval_artifact_key(history_prefix, run_id)
    history_latest_key = eval_latest_key(history_prefix)
    s3.put_object(Bucket=bucket, Key=history_key, Body=body, ContentType="application/json")
    s3.put_object(
        Bucket=bucket, Key=history_latest_key, Body=body,
        ContentType="application/json",
    )
    logger.info(
        "Research params archived to s3://%s/%s (+ latest.json sidecar)",
        bucket, history_key,
    )

    return {
        "applied": True,
        "params": {k: v for k, v in recommended.items() if k in SAFE_PARAMS},
        "n_samples": result.get("n_samples"),
    }
