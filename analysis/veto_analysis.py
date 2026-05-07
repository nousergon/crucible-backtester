"""
veto_analysis.py — analyze Predictor veto gate effectiveness.

Sweeps confidence thresholds against historical outcomes to find the
optimal veto_confidence setting. For each threshold, measures:
- How many BUY signals would have been vetoed (predicted DOWN + high confidence)
- Whether vetoed signals actually underperformed (precision of veto)
- How much alpha was missed from false vetoes (cost of being too aggressive)

Writes recommended threshold to S3 for the Predictor Lambda to read.
"""

import json
import logging
from datetime import date

import boto3
import pandas as pd
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

S3_PARAMS_KEY = "config/predictor_params.json"

# ── Fallback defaults (override via veto_analysis section in config.yaml) ──
_CONFIDENCE_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
_CURRENT_DEFAULT_THRESHOLD = 0.60
_MIN_PREDICTIONS = 30
_MIN_VETO_DECISIONS = 10
_MIN_THRESHOLD_CHANGE = 0.10
_COST_PENALTY_WEIGHT = 0.30

# Module-level config ref — set by init_config() from backtest.py
_cfg: dict = {}


def init_config(config: dict) -> None:
    """Load veto_analysis section from backtester config."""
    global _cfg
    _cfg = config.get("veto_analysis", {})


def _load_predictions_for_dates(dates: list[str], bucket: str) -> dict:
    """
    Load predictor predictions from S3 for each date.

    Returns: {date_str: {ticker: {predicted_direction, prediction_confidence, p_up, p_down}}}
    """
    s3 = boto3.client("s3")
    predictions_by_date = {}

    for d in dates:
        key = f"predictor/predictions/{d}.json"
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj["Body"].read())
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                continue
            logger.warning("S3 error loading predictions for %s: %s", d, e)
            continue

        by_ticker = {}
        for pred in data.get("predictions", []):
            ticker = pred.get("ticker")
            if ticker:
                by_ticker[ticker] = {
                    "predicted_direction": pred.get("predicted_direction"),
                    "prediction_confidence": pred.get("prediction_confidence", 0),
                    "p_up": pred.get("p_up", 0),
                    "p_down": pred.get("p_down", 0),
                }
        if by_ticker:
            predictions_by_date[d] = by_ticker

    logger.info("Loaded predictions for %d/%d dates", len(predictions_by_date), len(dates))
    return predictions_by_date


def _load_all_predictions(bucket: str) -> dict:
    """
    Load ALL predictor predictions from S3 (all dates).

    Returns: {date_str: {ticker: {predicted_direction, prediction_confidence, p_up, p_down}}}
    """
    s3 = boto3.client("s3")

    # List all prediction files
    try:
        resp = s3.list_objects_v2(
            Bucket=bucket, Prefix="predictor/predictions/", Delimiter="/",
        )
        keys = [
            obj["Key"] for obj in resp.get("Contents", [])
            if obj["Key"].endswith(".json") and "latest" not in obj["Key"]
        ]
    except ClientError as e:
        logger.warning("Failed to list prediction files: %s", e)
        return {}

    dates = [k.split("/")[-1].replace(".json", "") for k in keys]
    return _load_predictions_for_dates(dates, bucket)


def analyze_veto_effectiveness(df: pd.DataFrame, bucket: str) -> dict:
    """
    Analyze veto gate effectiveness across confidence thresholds.

    Args:
        df: score_performance DataFrame with beat_spy_10d, return_10d columns.
        bucket: S3 bucket containing predictor/predictions/{date}.json.

    Returns:
        {
            "status": "ok" | "insufficient_data",
            "current_threshold": 0.65,
            "n_predictions_loaded": int,
            "thresholds": [{confidence, n_vetoes, true_negatives, false_negatives,
                            precision, missed_alpha}, ...],
            "recommended_threshold": float,
            "recommendation_reason": str,
        }
    """
    if df is None or df.empty:
        return {"status": "insufficient_data", "note": "No score_performance data"}

    # Only look at rows with beat_spy_10d outcome resolved
    min_preds = _cfg.get("min_predictions", _MIN_PREDICTIONS)
    populated = df[df["beat_spy_10d"].notna()].copy()
    if len(populated) < min_preds:
        return {
            "status": "insufficient_data",
            "n_rows": len(populated),
            "min_required": min_preds,
            "note": f"Only {len(populated)} rows with outcomes (need {min_preds})",
        }

    # Load ALL available prediction dates from S3 (not just score_dates,
    # since prediction dates rarely match research run dates exactly).
    predictions_by_date = _load_all_predictions(bucket)

    if not predictions_by_date:
        return {
            "status": "no_predictions",
            "note": "No predictor predictions found in S3",
        }

    # For each score_performance row, find the nearest prediction on or before
    # the score_date. This is the prediction that would have been active when
    # the executor made its entry decision.
    pred_dates_sorted = sorted(predictions_by_date.keys())

    def _nearest_prediction(score_date_str: str, ticker: str) -> dict | None:
        """Find the prediction closest to (but not after) the score date."""
        for pd_str in reversed(pred_dates_sorted):
            if pd_str <= score_date_str:
                preds = predictions_by_date.get(pd_str, {}).get(ticker)
                if preds:
                    return preds
        return None

    # Join predictions with outcomes
    has_sector = "sector" in populated.columns
    rows = []
    for _, row in populated.iterrows():
        d = str(row["score_date"].date()) if hasattr(row["score_date"], "date") else str(row["score_date"])
        ticker = row["symbol"]
        preds = _nearest_prediction(d, ticker)
        if preds and preds.get("predicted_direction") == "DOWN":
            entry = {
                "symbol": ticker,
                "score_date": d,
                "prediction_confidence": float(preds["prediction_confidence"]),
                "beat_spy_10d": float(row["beat_spy_10d"]),
                "return_10d": float(row.get("return_10d", 0)),
            }
            if has_sector and pd.notna(row.get("sector")):
                entry["sector"] = row["sector"]
            rows.append(entry)

    # Diagnostic: track direction distribution for reporting
    all_directions = []
    for _, row in populated.iterrows():
        d = str(row["score_date"].date()) if hasattr(row["score_date"], "date") else str(row["score_date"])
        preds = _nearest_prediction(d, row["symbol"])
        if preds:
            all_directions.append(preds.get("predicted_direction", "UNKNOWN"))
    direction_counts = {}
    for d in all_directions:
        direction_counts[d] = direction_counts.get(d, 0) + 1

    if not rows:
        # Disambiguate against the Net veto value section's "221 DOWN
        # predictions evaluated" — this analysis is the sweep of recently
        # SCORED signals with realized 10d outcomes that happen to have
        # overlapping predictions; the Net veto value section is the full
        # corpus of resolved DOWN-direction predictions over an 8-week
        # window. Different filters → different counts; reading them
        # side-by-side without this context made today's email look
        # contradictory.
        n_window_signals = len(populated)
        return {
            "status": "no_down_predictions",
            "n_predictions_loaded": sum(len(v) for v in predictions_by_date.values()),
            "direction_distribution": direction_counts,
            "note": (
                f"No DOWN predictions overlap with the {n_window_signals} recently-scored "
                "signals that have realized 10d outcomes — veto gate sweep has nothing "
                f"to fit on. Direction distribution in this overlap window: {direction_counts}. "
                "(Note: the Net veto value section below evaluates a wider 8-week corpus of "
                "resolved DOWN predictions and reports its own count separately.)"
            ),
        }

    down_df = pd.DataFrame(rows)
    n_down = len(down_df)
    logger.info("Found %d DOWN predictions with outcomes for veto analysis", n_down)

    # Base rate: % of all BUY signals (in populated df) that beat SPY
    base_rate = float(populated["beat_spy_10d"].mean())
    logger.info("Veto base rate: %.1f%% of BUY signals beat SPY at 10d", base_rate * 100)

    # Sweep thresholds and select best
    thresholds = _cfg.get("confidence_thresholds", _CONFIDENCE_THRESHOLDS)
    current_default = _cfg.get("current_default_threshold", _CURRENT_DEFAULT_THRESHOLD)
    min_veto_dec = _cfg.get("min_veto_decisions", _MIN_VETO_DECISIONS)
    cost_weight = _cfg.get("cost_penalty_weight", _COST_PENALTY_WEIGHT)

    threshold_results = _sweep_thresholds(down_df, base_rate, thresholds)
    n_preds_loaded = sum(len(v) for v in predictions_by_date.values())

    result = _select_best_threshold(
        threshold_results, base_rate, cost_weight, current_default,
        min_veto_dec, n_down, n_preds_loaded,
    )

    # Per-sector veto precision at the recommended threshold
    if "sector" in down_df.columns and down_df["sector"].notna().any():
        rec_thresh = result.get("recommended_threshold", current_default)
        by_sector = []
        for sector in sorted(down_df["sector"].dropna().unique()):
            s_df = down_df[down_df["sector"] == sector]
            vetoed = s_df[s_df["prediction_confidence"] >= rec_thresh]
            n_vetoes = len(vetoed)
            if n_vetoes == 0:
                by_sector.append({"sector": sector, "n_down": len(s_df), "n_vetoes": 0, "precision": None, "recall": None})
                continue
            tn = int((vetoed["beat_spy_10d"] == 0).sum())
            total_under = int((s_df["beat_spy_10d"] == 0).sum())
            precision = tn / n_vetoes if n_vetoes > 0 else None
            recall = tn / total_under if total_under > 0 else None
            by_sector.append({
                "sector": sector,
                "n_down": len(s_df),
                "n_vetoes": n_vetoes,
                "precision": round(precision, 4) if precision is not None else None,
                "recall": round(recall, 4) if recall is not None else None,
            })
        result["by_sector"] = by_sector

    return result


def _sweep_thresholds(
    down_df: pd.DataFrame, base_rate: float, thresholds: list[float],
) -> list[dict]:
    """Evaluate veto precision, recall, F1, and missed alpha at each threshold."""
    from analysis.signal_quality import _wilson_ci

    # Total actual underperformers across ALL DOWN predictions (for recall denominator)
    total_actual_underperformers = int((down_df["beat_spy_10d"] == 0).sum())

    results = []
    for threshold in thresholds:
        vetoed = down_df[down_df["prediction_confidence"] >= threshold]
        n_vetoes = len(vetoed)

        if n_vetoes == 0:
            results.append({
                "confidence": threshold,
                "n_vetoes": 0,
                "true_negatives": 0,
                "false_negatives": 0,
                "precision": None,
                "recall": None,
                "f1": None,
                "precision_ci_95": None,
                "low_confidence": True,
                "missed_alpha": 0.0,
                "missed_alpha_per_winner": 0.0,
                "lift": None,
            })
            continue

        true_neg = int((vetoed["beat_spy_10d"] == 0).sum())
        false_neg = int((vetoed["beat_spy_10d"] == 1).sum())
        precision = true_neg / n_vetoes

        # Recall: of all actual underperformers, how many did we veto?
        recall = true_neg / total_actual_underperformers if total_actual_underperformers > 0 else None
        # F1: harmonic mean of precision and recall
        if recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = None

        precision_ci = _wilson_ci(true_neg, n_vetoes)
        low_confidence = n_vetoes < 30

        vetoed_winners = vetoed[vetoed["beat_spy_10d"] == 1]
        missed_total = float(vetoed_winners["return_10d"].sum())
        missed_per_winner = float(vetoed_winners["return_10d"].mean()) if len(vetoed_winners) > 0 else 0.0

        lift = precision - base_rate

        results.append({
            "confidence": threshold,
            "n_vetoes": n_vetoes,
            "true_negatives": true_neg,
            "false_negatives": false_neg,
            "precision": round(precision, 4),
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "precision_ci_95": precision_ci,
            "low_confidence": low_confidence,
            "missed_alpha": round(missed_total, 4),
            "missed_alpha_per_winner": round(missed_per_winner, 4),
            "lift": round(lift, 4),
        })

    return results


def _select_best_threshold(
    threshold_results: list[dict],
    base_rate: float,
    cost_weight: float,
    current_default: float,
    min_veto_dec: int,
    n_down: int,
    n_preds_loaded: int,
) -> dict:
    """Score thresholds and select recommendation, applying lift gate and cost sensitivity."""
    scoreable = [t for t in threshold_results if t["n_vetoes"] >= min_veto_dec]

    if not scoreable:
        return {
            "status": "insufficient_vetoes",
            "current_threshold": current_default,
            "base_rate": round(base_rate, 4),
            "n_down_predictions": n_down,
            "thresholds": threshold_results,
            "note": (
                f"No threshold has {min_veto_dec}+ veto decisions. "
                "Need more prediction history for reliable analysis."
            ),
        }

    max_missed = max(abs(t["missed_alpha"]) for t in scoreable) or 1.0
    for t in scoreable:
        cost_penalty = cost_weight * (abs(t["missed_alpha"]) / max_missed)
        t["_score"] = t["precision"] - cost_penalty

    best = max(scoreable, key=lambda t: t["_score"])
    recommended = best["confidence"]

    # Insufficient lift gate: if best lift < 5pp, don't recommend
    best_lift = best.get("lift", 0.0)
    if best_lift is not None and best_lift < 0.05:
        return {
            "status": "insufficient_lift",
            "current_threshold": current_default,
            "base_rate": round(base_rate, 4),
            "n_down_predictions": n_down,
            "n_predictions_loaded": n_preds_loaded,
            "thresholds": threshold_results,
            "recommended_threshold": recommended,
            "recommendation_reason": (
                f"Best threshold {recommended:.2f} has precision {best['precision']:.1%} "
                f"but lift over base rate is only {best_lift:.1%} (need 5%+). "
                f"Base rate: {base_rate:.1%}."
            ),
        }

    # Cost sensitivity analysis: sweep cost_weight values
    cost_sensitivity_weights = [0.15, 0.30, 0.50, 0.70]
    cost_sensitivity_results = {}
    for cw in cost_sensitivity_weights:
        for t in scoreable:
            cp = cw * (abs(t["missed_alpha"]) / max_missed)
            t[f"_score_{cw}"] = t["precision"] - cp
        cw_best = max(scoreable, key=lambda t: t[f"_score_{cw}"])
        cost_sensitivity_results[str(cw)] = cw_best["confidence"]

    unique_thresholds = set(cost_sensitivity_results.values())
    cost_sensitivity = "high" if len(unique_thresholds) > 2 else "low" if len(unique_thresholds) == 1 else "moderate"

    return {
        "status": "ok",
        "current_threshold": current_default,
        "base_rate": round(base_rate, 4),
        "n_down_predictions": n_down,
        "n_predictions_loaded": n_preds_loaded,
        "thresholds": threshold_results,
        "recommended_threshold": recommended,
        "recommendation_reason": (
            f"Confidence {recommended:.2f}: precision {best['precision']:.1%} "
            f"(lift +{best_lift:.1%} over {base_rate:.1%} base rate) "
            f"with {best['missed_alpha']:.4f} missed alpha "
            f"({best['n_vetoes']} vetoes, {best['true_negatives']} correct)"
        ),
        "cost_sensitivity": cost_sensitivity,
        "cost_sensitivity_details": cost_sensitivity_results,
    }


def _read_current_veto_threshold(bucket: str) -> float | None:
    """Read the current veto threshold from S3 (last backtester-optimized value)."""
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=S3_PARAMS_KEY)
        data = json.loads(obj["Body"].read())
        if "veto_confidence" in data:
            logger.info(
                "Current veto threshold from S3: %.2f (updated %s)",
                data["veto_confidence"], data.get("updated_at", "unknown"),
            )
            return float(data["veto_confidence"])
    except Exception as e:
        logger.info("No predictor params in S3 (%s), using config default", e)
    return None


def apply(result: dict, bucket: str) -> dict:
    """
    Write recommended veto threshold to S3 if guardrails pass.

    Writes to s3://{bucket}/config/predictor_params.json and archives
    to config/predictor_params_history/{date}.json.
    """
    if result.get("status") != "ok":
        return {"applied": False, "reason": f"status={result.get('status')}"}

    config_default = _cfg.get("current_default_threshold", _CURRENT_DEFAULT_THRESHOLD)
    min_change = _cfg.get("min_threshold_change", _MIN_THRESHOLD_CHANGE)

    recommended = result.get("recommended_threshold")
    # Use S3 value (last known optimal) as the current baseline, not hardcoded default
    s3_current = _read_current_veto_threshold(bucket)
    current = s3_current if s3_current is not None else result.get("current_threshold", config_default)

    if recommended is None:
        return {"applied": False, "reason": "no recommended threshold"}

    if abs(recommended - current) < min_change:
        return {
            "applied": False,
            "reason": (
                f"Recommended ({recommended:.2f}) too close to current "
                f"({current:.2f}) — need {min_change}+ difference"
            ),
        }

    payload = {
        "veto_confidence": recommended,
        "precision": next(
            (t["precision"] for t in result.get("thresholds", [])
             if t["confidence"] == recommended), None
        ),
        "n_vetoes": next(
            (t["n_vetoes"] for t in result.get("thresholds", [])
             if t["confidence"] == recommended), None
        ),
        "updated_at": str(date.today()),
        "recommendation_reason": result.get("recommendation_reason"),
    }

    from optimizer.rollback import save_previous
    save_previous(bucket, "predictor_params")

    s3 = boto3.client("s3")
    body = json.dumps(payload, indent=2)

    try:
        s3.put_object(Bucket=bucket, Key=S3_PARAMS_KEY, Body=body, ContentType="application/json")
        logger.info("Predictor veto threshold updated in S3: %s", recommended)
    except Exception as e:
        logger.error("CRITICAL: Failed to write predictor params to S3: %s", e)
        return {"applied": False, "reason": f"S3 write failed: {e}"}

    history_key = f"config/predictor_params_history/{date.today().isoformat()}.json"
    try:
        s3.put_object(Bucket=bucket, Key=history_key, Body=body, ContentType="application/json")
        logger.info("Predictor params archived to s3://%s/%s", bucket, history_key)
    except Exception as e:
        logger.warning("Failed to archive predictor params history (non-fatal): %s", e)

    return {
        "applied": True,
        "veto_confidence": recommended,
        "previous": current,
    }
