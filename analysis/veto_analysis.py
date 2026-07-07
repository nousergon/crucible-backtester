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
from nousergon_lib.eval_artifacts import (
    eval_artifact_key,
    eval_latest_key,
    new_eval_run_id,
)
from nousergon_lib.quant.horizons import DEFAULT_POLICY
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

S3_PARAMS_KEY = "config/predictor_params.json"
S3_SHADOW_PREFIX = "config/predictor_params_shadow_history"

# ── Canonical outcome columns (config#1483/#1528) ────────────────────────────
# Resolved from the fleet HorizonPolicy chokepoint, never hardcoded literals.
# The outcome DATA is long-format (score_performance_outcomes), attached to the
# df upstream by analysis.outcome_store.attach_outcomes under these names.
_PRIMARY_OC = DEFAULT_POLICY.outcome_columns(DEFAULT_POLICY.primary_horizon)
_BEAT = _PRIMARY_OC.beat_spy          # canonical beat-SPY flag column
_RET = _PRIMARY_OC.stock_return       # canonical stock-return column (2dp %)

# ── Fallback defaults (override via veto_analysis section in config.yaml) ──
_CONFIDENCE_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
_CURRENT_DEFAULT_THRESHOLD = 0.60
_MIN_PREDICTIONS = 30
_MIN_VETO_DECISIONS = 10
_MIN_THRESHOLD_CHANGE = 0.10
_COST_PENALTY_WEIGHT = 0.30
_MIN_LIFT_OVER_BASE_RATE = 0.05  # 5pp lift gate (legacy + skill modes)

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
        df: score_performance DataFrame with the canonical primary-horizon
            outcome columns attached (analysis.outcome_store.attach_outcomes).
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

    # Only look at rows with the canonical 21d outcome resolved (config#1451:
    # the legacy 10d horizon was retired in the canonical-alpha cutover).
    min_preds = _cfg.get("min_predictions", _MIN_PREDICTIONS)
    if _BEAT not in df.columns:
        logger.warning(
            "veto_analysis STARVED: %r absent from the outcome frame "
            "— schema drift / horizon retirement?", _BEAT,
        )
        return {"status": "insufficient_data", "n_rows": 0, "note": f"{_BEAT} column absent"}
    populated = df[df[_BEAT].notna()].copy()
    if len(populated) < min_preds:
        # Fail-loud (config#1451): a silent insufficient_data is how the 10d
        # retirement starved this gate for months.
        logger.warning(
            "veto_analysis STARVED: only %d rows with resolved %s "
            "(of %d, need %d) — check the outcome backfill / horizon.",
            len(populated), _BEAT, len(df), min_preds,
        )
        return {
            "status": "insufficient_data",
            "n_rows": len(populated),
            "min_required": min_preds,
            "note": f"Only {len(populated)} rows with resolved {_BEAT} (need {min_preds})",
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
                _BEAT: float(row[_BEAT]),
                _RET: float(row.get(_RET, 0)),
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
    base_rate = float(populated[_BEAT].mean())
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
            tn = int((vetoed[_BEAT] == 0).sum())
            total_under = int((s_df[_BEAT] == 0).sum())
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

        # Per-sector veto thresholds (config#921). The by_sector block above
        # only *measures* precision at the GLOBAL recommended threshold. This
        # block answers the issue: where veto precision varies by sector, fit a
        # SEPARATE threshold per sector by running the same sweep + gate machinery
        # on that sector's DOWN-prediction sub-corpus.
        result["per_sector_thresholds"] = compute_per_sector_thresholds(
            down_df,
            base_rate,
            thresholds,
            cost_weight,
            current_default,
            min_veto_dec,
            global_recommended=result.get("recommended_threshold", current_default),
        )

    # Observe-first significance verdict (config#1426 Phase 3). NON-ENFORCING:
    # the legacy gate promotes a veto threshold on a 5pp POINT lift over base
    # rate; this asks whether the recommended threshold's precision lift is
    # statistically significant (Wilson lower bound > base rate — the same shape
    # already enforced in skill-composite mode, computed uniformly here). It
    # NEVER changes the recommendation. Swallow rationale ("fail loud"
    # carve-out): (a) failure = observe instrumentation error on a SECONDARY
    # path; the recommendation is unaffected; (c) recorded surface = WARN log.
    if result.get("status") == "ok" and bool(_cfg.get("significance_observe_enabled", True)):
        try:
            from optimizer.significance_observe import observe_veto
            result["significance_observe"] = observe_veto(
                result.get("thresholds", []),
                result.get("recommended_threshold"),
                result.get("base_rate", 0.0),
                cfg=_cfg,
            )
        except Exception as e:  # observe-only: must not break the optimizer
            logger.warning(
                "veto_analysis significance_observe failed (non-fatal, observe-only): %s", e,
            )
            result["significance_observe"] = None

    return result


def compute_per_sector_thresholds(
    down_df: pd.DataFrame,
    base_rate: float,
    thresholds: list[float],
    cost_weight: float,
    current_default: float,
    min_veto_dec: int,
    *,
    global_recommended: float,
) -> dict:
    """Fit a per-sector veto confidence threshold (config#921).

    For each sector with a DOWN-prediction sub-corpus, run the SAME
    ``_sweep_thresholds`` + ``_select_best_threshold`` pipeline the global
    analysis uses (so per-sector recommendations inherit the lift gate, the
    confidence-bounded DSR gate, and the cost-sensitivity logic). A sector only
    earns an OVERRIDE when its recommendation:

      * has ``status == "ok"`` (cleared all gates on its own data), AND
      * differs from the global recommendation by at least the configured
        ``min_threshold_change`` (otherwise the global threshold is fine).

    Returns::

        {
          "status": "ok" | "no_sector_column" | "insufficient_sector_data",
          "global_recommended": float,
          "min_threshold_change": float,
          "min_veto_decisions": int,
          "by_sector": {sector: {recommended_threshold, status, n_down,
                                 precision, lift, is_override, delta_vs_global,
                                 recommendation_reason}},
          "overrides": {sector: threshold},   # only sectors that earned one
        }

    The ``overrides`` map is the actionable output — the per-sector confidence
    thresholds the predictor's veto gate would consume. It is deliberately
    sparse: sectors without enough data or without a materially different
    optimum fall back to the global threshold.
    """
    out: dict = {
        "status": "ok",
        "global_recommended": global_recommended,
        "min_threshold_change": _cfg.get("min_threshold_change", _MIN_THRESHOLD_CHANGE),
        "min_veto_decisions": min_veto_dec,
        "by_sector": {},
        "overrides": {},
    }
    if "sector" not in down_df.columns or not down_df["sector"].notna().any():
        out["status"] = "no_sector_column"
        return out

    min_change = out["min_threshold_change"]
    any_sector_fit = False

    for sector in sorted(down_df["sector"].dropna().unique()):
        s_df = down_df[down_df["sector"] == sector]
        n_down = len(s_df)
        # Base rate is global on purpose: the lift gate measures a sector's veto
        # precision against the SAME baseline the global gate uses, so a sector
        # override must beat the portfolio-wide hit rate, not just its own.
        sweep = _sweep_thresholds(s_df, base_rate, thresholds)
        sel = _select_best_threshold(
            sweep, base_rate, cost_weight, current_default,
            min_veto_dec, n_down, n_preds_loaded=n_down,
        )
        rec = sel.get("recommended_threshold")
        status = sel.get("status", "ok")
        if status == "ok":
            any_sector_fit = True
        best = next(
            (t for t in sweep if t.get("confidence") == rec), {}
        ) if rec is not None else {}
        delta = abs(rec - global_recommended) if rec is not None else None
        is_override = bool(
            status == "ok"
            and rec is not None
            and delta is not None
            and delta >= min_change
        )
        out["by_sector"][sector] = {
            "recommended_threshold": rec,
            "status": status,
            "n_down": n_down,
            "precision": best.get("precision"),
            "lift": best.get("lift"),
            "delta_vs_global": round(delta, 4) if delta is not None else None,
            "is_override": is_override,
            "recommendation_reason": sel.get("recommendation_reason"),
        }
        if is_override:
            out["overrides"][sector] = rec

    if not any_sector_fit:
        out["status"] = "insufficient_sector_data"
    return out


def _sweep_thresholds(
    down_df: pd.DataFrame, base_rate: float, thresholds: list[float],
) -> list[dict]:
    """Evaluate veto precision, recall, F1, and missed alpha at each threshold."""
    from analysis.signal_quality import _wilson_ci

    # Total actual underperformers across ALL DOWN predictions (for recall denominator)
    total_actual_underperformers = int((down_df[_BEAT] == 0).sum())

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

        true_neg = int((vetoed[_BEAT] == 0).sum())
        false_neg = int((vetoed[_BEAT] == 1).sum())
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

        vetoed_winners = vetoed[vetoed[_BEAT] == 1]
        missed_total = float(vetoed_winners[_RET].sum())
        missed_per_winner = float(vetoed_winners[_RET].mean()) if len(vetoed_winners) > 0 else 0.0

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

    # Ranking: legacy (precision − cost_penalty × |missed_alpha|) vs
    # skill-composite (F1 with confidence-bounded lift gate). Mirrors the
    # weight_optimizer + executor_optimizer fit-target switch shipped 2026-05-09.
    # Per Brian: alpha vs SPY is presentation framing, not an optimizer fit
    # target — skill-composite ranking drops the `missed_alpha` cost term
    # and ranks on F1 (precision × recall harmonic mean), which rewards
    # catching real underperformers without the alpha-axis pollution.
    use_skill_composite = bool(_cfg.get("use_skill_composite_target", False))

    if use_skill_composite:
        # F1 may be None when recall is undefined (no actual underperformers
        # in the corpus yet) — fall back to precision in that case.
        for t in scoreable:
            t["_score"] = t.get("f1") if t.get("f1") is not None else t["precision"]
        best = max(scoreable, key=lambda t: t["_score"])
        fit_target = "skill_composite"
    else:
        max_missed = max(abs(t["missed_alpha"]) for t in scoreable) or 1.0
        for t in scoreable:
            cost_penalty = cost_weight * (abs(t["missed_alpha"]) / max_missed)
            t["_score"] = t["precision"] - cost_penalty
        best = max(scoreable, key=lambda t: t["_score"])
        fit_target = "precision_minus_alpha_cost_legacy"

    recommended = best["confidence"]

    # Insufficient lift gate: if best lift < 5pp, don't recommend
    best_lift = best.get("lift", 0.0)
    min_lift = _cfg.get("min_lift_over_base_rate", _MIN_LIFT_OVER_BASE_RATE)
    if best_lift is not None and best_lift < min_lift:
        return {
            "status": "insufficient_lift",
            "blocked_by": ["min_lift_over_base_rate"],
            "fit_target": fit_target,
            "current_threshold": current_default,
            "base_rate": round(base_rate, 4),
            "n_down_predictions": n_down,
            "n_predictions_loaded": n_preds_loaded,
            "thresholds": threshold_results,
            "recommended_threshold": recommended,
            "recommendation_reason": (
                f"Best threshold {recommended:.2f} has precision {best['precision']:.1%} "
                f"but lift over base rate is only {best_lift:.1%} "
                f"(need {min_lift:.0%}+). Base rate: {base_rate:.1%}."
            ),
        }

    # DSR-style confidence-bounded lift gate (skill-composite mode only).
    # Workstream D bullet 3 of evaluator-revamp-260506.md: don't promote a
    # threshold whose precision lift is statistically indistinguishable from
    # zero. Implemented as: precision_ci_95 lower bound must beat base_rate.
    # This gate fires AFTER the point-estimate lift gate above.
    if use_skill_composite:
        ci_lower = None
        ci = best.get("precision_ci_95")
        if isinstance(ci, (list, tuple)) and len(ci) >= 2:
            ci_lower = ci[0]
        if ci_lower is not None and ci_lower <= base_rate:
            return {
                "status": "insufficient_confidence",
                "blocked_by": ["precision_ci_below_base_rate"],
                "fit_target": fit_target,
                "current_threshold": current_default,
                "base_rate": round(base_rate, 4),
                "n_down_predictions": n_down,
                "n_predictions_loaded": n_preds_loaded,
                "thresholds": threshold_results,
                "recommended_threshold": recommended,
                "recommendation_reason": (
                    f"Best threshold {recommended:.2f} has precision {best['precision']:.1%} "
                    f"with 95% CI lower bound {ci_lower:.1%} which does not exceed "
                    f"base rate {base_rate:.1%} — lift is statistically indistinguishable "
                    f"from zero. Need more vetoes for confidence-bounded promotion."
                ),
            }

    # Cost sensitivity analysis: sweep cost_weight values. Legacy-only —
    # under skill_composite mode the cost penalty isn't part of the
    # ranking, so the sensitivity table is meaningless.
    cost_sensitivity_results: dict = {}
    cost_sensitivity = "n/a (skill_composite mode)"
    if not use_skill_composite:
        cost_sensitivity_weights = [0.15, 0.30, 0.50, 0.70]
        for cw in cost_sensitivity_weights:
            for t in scoreable:
                cp = cw * (abs(t["missed_alpha"]) / max_missed)
                t[f"_score_{cw}"] = t["precision"] - cp
            cw_best = max(scoreable, key=lambda t: t[f"_score_{cw}"])
            cost_sensitivity_results[str(cw)] = cw_best["confidence"]
        unique_thresholds = set(cost_sensitivity_results.values())
        cost_sensitivity = (
            "high" if len(unique_thresholds) > 2
            else "low" if len(unique_thresholds) == 1
            else "moderate"
        )

    return {
        "status": "ok",
        "fit_target": fit_target,
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
            f"({best['n_vetoes']} vetoes, {best['true_negatives']} correct) "
            f"[fit_target={fit_target}]"
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

    Two write paths, mirroring the executor_optimizer cutover pattern:

    - **Production** (default and any time ``fit_target`` is the legacy
      ranking, OR ``enforce_skill_composite`` is true under skill mode):
      writes to ``config/predictor_params.json`` (live) +
      ``config/predictor_params_history/{date}.json`` (audit).
    - **Shadow** (skill ranking computed but ``enforce_skill_composite``
      is false): writes to
      ``config/predictor_params_shadow_history/{date}.json`` only —
      live config is unchanged. Lets the skill_composite ranking
      validate against a few Sat SF cycles before becoming authoritative.

    Returns ``{"applied": True, ...}`` on production write,
    ``{"applied": False, "reason": ..., "shadow_key": ...}`` on shadow
    write or guardrail rejection.
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

    # Bootstrap: if no S3 artifact exists (s3_current is None), allow the first write
    # even if it equals the default. The artifact's existence establishes the baseline
    # for future min_threshold_change gates. Only write if significance verdict permits.
    is_bootstrap = s3_current is None

    if not is_bootstrap and abs(recommended - current) < min_change:
        return {
            "applied": False,
            "blocked_by": ["min_threshold_change"],
            "reason": (
                f"Recommended ({recommended:.2f}) too close to current "
                f"({current:.2f}) — need {min_change}+ difference"
            ),
        }

    # For bootstrap writes: check significance verdict first, block if failing
    if is_bootstrap:
        if bool(_cfg.get("enforce_significance", False)):
            from optimizer.significance_observe import significance_would_block
            verdict = result.get("significance_observe")
            if significance_would_block(verdict):
                logger.info(
                    "veto_analysis: bootstrap seed BLOCKED by significance floor (config#1426)"
                )
                return {
                    "applied": False,
                    "blocked_by": ["significance_floor"],
                    "reason": "veto_analysis: bootstrap seed blocked by significance enforce "
                              "(config#1426) — undefended evidence",
                    "observe_verdict": verdict,
                }
        logger.info(
            "veto_analysis: bootstrap seed write (no prior S3 artifact) with "
            "recommendation %.2f", recommended
        )

    fit_target = result.get("fit_target", "precision_minus_alpha_cost_legacy")
    enforce_skill_composite = bool(_cfg.get("enforce_skill_composite", False))
    shadow_only = fit_target == "skill_composite" and not enforce_skill_composite

    payload = {
        "veto_confidence": recommended,
        "fit_target": fit_target,
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

    s3 = boto3.client("s3")
    body = json.dumps(payload, indent=2)

    if shadow_only:
        # Canonical eval-style archive layout per lib v0.8.0 — flat
        # {prefix}/{run_id}.json + latest.json sidecar (YYMMDDHHMM run_id)
        run_id = new_eval_run_id()
        shadow_key = eval_artifact_key(S3_SHADOW_PREFIX, run_id)
        shadow_latest_key = eval_latest_key(S3_SHADOW_PREFIX)
        try:
            s3.put_object(
                Bucket=bucket, Key=shadow_key, Body=body, ContentType="application/json",
            )
            s3.put_object(
                Bucket=bucket, Key=shadow_latest_key, Body=body,
                ContentType="application/json",
            )
            logger.info(
                "Veto threshold written to shadow archive "
                "(enforce_skill_composite=False): s3://%s/%s (+ latest.json sidecar)",
                bucket, shadow_key,
            )
        except Exception as e:
            logger.error("Failed to write veto threshold shadow archive: %s", e)
            return {"applied": False, "reason": f"shadow S3 write failed: {e}"}
        return {
            "applied": False,
            "reason": (
                "shadow mode — fit_target=skill_composite, "
                "enforce_skill_composite=False"
            ),
            "shadow_key": shadow_key,
            "fit_target": fit_target,
            "veto_confidence": recommended,
        }

    # Significance ENFORCE gate (config#1426 Phase 4) — default OFF. The existing
    # 5pp point-lift gate stays as-is; enforce ADDS a block when the veto
    # precision lift is not statistically significant (Wilson lower bound not
    # above base rate → would_block). Missing verdict → conservative block.
    # Enforce can only BLOCK a promotion the live gate already allowed.
    # NOTE: bootstrap writes already checked this above; skip for non-bootstrap.
    if not is_bootstrap and bool(_cfg.get("enforce_significance", False)):
        from optimizer.significance_observe import significance_would_block
        verdict = result.get("significance_observe")
        if significance_would_block(verdict):
            logger.info(
                "veto_analysis: significance enforce BLOCKED promotion (config#1426)"
            )
            return {
                "applied": False,
                "blocked_by": ["significance_floor"],
                "reason": "veto_analysis: blocked by significance enforce "
                          "(config#1426) — undefended evidence",
                "observe_verdict": verdict,
            }

    from optimizer.rollback import save_previous
    save_previous(bucket, "predictor_params")

    try:
        s3.put_object(Bucket=bucket, Key=S3_PARAMS_KEY, Body=body, ContentType="application/json")
        logger.info(
            "Predictor veto threshold updated in S3: %s (fit_target=%s)",
            recommended, fit_target,
        )
    except Exception as e:
        logger.error("CRITICAL: Failed to write predictor params to S3: %s", e)
        return {"applied": False, "reason": f"S3 write failed: {e}"}

    # Canonical eval-style archive layout per lib v0.8.0 — see shadow path above
    history_run_id = new_eval_run_id()
    history_prefix = "config/predictor_params_history"
    history_key = eval_artifact_key(history_prefix, history_run_id)
    history_latest_key = eval_latest_key(history_prefix)
    try:
        s3.put_object(Bucket=bucket, Key=history_key, Body=body, ContentType="application/json")
        s3.put_object(
            Bucket=bucket, Key=history_latest_key, Body=body,
            ContentType="application/json",
        )
        logger.info(
            "Predictor params archived to s3://%s/%s (+ latest.json sidecar)",
            bucket, history_key,
        )
    except Exception as e:
        logger.warning("Failed to archive predictor params history (non-fatal): %s", e)

    return {
        "applied": True,
        "fit_target": fit_target,
        "veto_confidence": recommended,
        "previous": current,
    }
