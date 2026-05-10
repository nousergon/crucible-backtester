"""
weight_optimizer.py — scoring weight recommendation based on sub-score attribution.

Joins score_performance outcomes (research.db) with sub-scores from signals.json (S3)
to compute which sub-scores (quant / qual) best predict outperformance.
Suggests revised weights and applies them to S3 if guardrails pass.

Horizon separation: Research uses quant + qual only (6–12 month fundamental
attractiveness). Technical analysis is handled by Predictor (GBM) and Executor.

Current default weights: quant=0.50, qual=0.50
"""

import json
import logging
from datetime import date

import boto3
import pandas as pd
from alpha_engine_lib.eval_artifacts import (
    eval_artifact_key,
    eval_latest_key,
    new_eval_run_id,
)
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

SUB_SCORES = ["quant", "qual"]
S3_WEIGHTS_KEY = "config/scoring_weights.json"
S3_SHADOW_WEIGHTS_PREFIX = "config/scoring_weights_shadow_history"

# ── Fallback defaults (override via weight_optimizer section in config.yaml) ──
_DEFAULT_WEIGHTS = {"quant": 0.50, "qual": 0.50}
_MAX_SINGLE_CHANGE = 0.10
_MIN_MEANINGFUL_CHANGE = 0.03
_BLEND_FACTOR = 0.20
_CONFIDENCE_LOW = 100
_CONFIDENCE_MEDIUM = 300
_HORIZON_BLEND = {"beat_spy_10d": 0.50, "beat_spy_30d": 0.50}

# Module-level config ref — set by init_config() from backtest.py
_cfg: dict = {}


def init_config(config: dict) -> None:
    """Load weight_optimizer section from backtester config."""
    global _cfg
    _cfg = config.get("weight_optimizer", {})


def load_with_subscores(
    df: pd.DataFrame,
    bucket: str,
    signals_prefix: str = "signals",
) -> pd.DataFrame:
    """
    Enrich a score_performance DataFrame with sub-scores from signals.json in S3.

    For each unique score_date in df, loads the corresponding signals.json and
    extracts quant/qual sub-scores per symbol. Merges back by
    (symbol, score_date).

    Reads sub_scores from the signals dict in each signals.json file.

    Post-migration #12 (research.db, 2026-05-08), score_performance carries
    quant_score / qual_score as canonical columns; this path becomes a
    backfill-only round-trip for legacy rows with NULL sub-scores.

    Args:
        df:              score_performance DataFrame (from signal_quality.load_score_performance).
        bucket:          S3 bucket containing signals/{date}/signals.json.
        signals_prefix:  S3 prefix for signals files (default "signals").

    Returns:
        DataFrame with quant_score, qual_score columns populated. Canonical
        values from score_performance take precedence; S3 fills only NULLs.
        Rows where neither source resolves are kept with NaN sub-scores.
    """
    if df.empty:
        return df

    has_canonical = "quant_score" in df.columns and "qual_score" in df.columns
    if has_canonical:
        nulls = df["quant_score"].isna() | df["qual_score"].isna()
        if not nulls.any():
            logger.info(
                "Sub-scores fully populated from score_performance (%d rows); "
                "skipping S3 backfill.",
                len(df),
            )
            return df
        logger.info(
            "Sub-scores present on score_performance for %d/%d rows; "
            "backfilling %d NULL rows from S3.",
            (~nulls).sum(),
            len(df),
            nulls.sum(),
        )

    dates = df["score_date"].unique().tolist()
    logger.info("Loading sub-scores for %d signal dates from S3...", len(dates))

    # Build lookup: {score_date: {symbol: {quant: N, qual: N}}}
    subscores_by_date: dict[str, dict] = {}
    s3 = boto3.client("s3")

    for d in dates:
        # score_dates may be Timestamps — normalize to YYYY-MM-DD string
        d_str = str(d)[:10]
        key = f"{signals_prefix}/{d_str}/signals.json"
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj["Body"].read())
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.debug("No signals.json for %s — sub-scores unavailable for this date", d)
            else:
                logger.warning("S3 error loading signals for %s: %s", d, e)
            continue

        by_symbol: dict[str, dict] = {}

        def _extract_subscores(sig: dict) -> dict | None:
            """Extract quant/qual sub-scores from either nested or flat format."""
            # Nested: sub_scores: {quant: N, qual: N}
            sub = sig.get("sub_scores", {})
            if sub and any(sub.get(k) is not None for k in SUB_SCORES):
                return {k: sub.get(k) for k in SUB_SCORES}
            # Flat: quant_score, qual_score
            flat = {k: sig.get(f"{k}_score") for k in SUB_SCORES}
            if any(v is not None for v in flat.values()):
                return flat
            # Legacy format (pre-2026-03-29): technical ≈ quant, avg(news,research) ≈ qual
            if sub:
                tech = sub.get("technical")
                news = sub.get("news")
                research = sub.get("research")
                if tech is not None:
                    qual_parts = [v for v in (news, research) if v is not None]
                    qual = sum(qual_parts) / len(qual_parts) if qual_parts else None
                    return {"quant": tech, "qual": qual}
            return None

        # Check signals dict (v1 format)
        for ticker, sig in data.get("signals", {}).items():
            scores = _extract_subscores(sig)
            if ticker and scores:
                by_symbol[ticker] = scores

        # Also check universe list (v2 format) for any tickers not yet found
        for sig in data.get("universe", []):
            ticker = sig.get("ticker")
            if ticker and ticker not in by_symbol:
                scores = _extract_subscores(sig)
                if scores:
                    by_symbol[ticker] = scores

        subscores_by_date[d] = by_symbol
        logger.debug("Loaded sub-scores for %d symbols on %s", len(by_symbol), d)

    loaded_dates = len(subscores_by_date)
    logger.info(
        "Sub-scores loaded for %d/%d dates", loaded_dates, len(dates)
    )

    if not subscores_by_date:
        logger.warning(
            "No sub-scores found in S3. signals.json must include sub_scores per stock. "
            "Attribution and weight optimization will be skipped."
        )
        return df

    # Build a flat sub-score DataFrame and merge
    rows = []
    for d, by_symbol in subscores_by_date.items():
        for symbol, sub in by_symbol.items():
            rows.append({"symbol": symbol, "score_date": d, **{f"{k}_score": v for k, v in sub.items()}})

    sub_df = pd.DataFrame(rows)
    if has_canonical:
        # score_performance owns the fact; S3 only fills NULLs. Suffix the S3
        # columns, COALESCE canonical → S3, drop the suffix.
        merged = df.merge(
            sub_df, on=["symbol", "score_date"], how="left", suffixes=("", "_s3")
        )
        for col in ("quant_score", "qual_score"):
            s3_col = f"{col}_s3"
            if s3_col in merged.columns:
                merged[col] = merged[col].fillna(merged[s3_col])
                merged = merged.drop(columns=[s3_col])
    else:
        merged = df.merge(sub_df, on=["symbol", "score_date"], how="left")
    filled = merged[["quant_score", "qual_score"]].notna().any(axis=1).sum()
    logger.info("Sub-scores matched for %d/%d score_performance rows", filled, len(merged))
    return merged


def _validate_and_split(
    df: pd.DataFrame,
    current_weights: dict,
    min_samples: int,
) -> dict | tuple[pd.DataFrame, pd.DataFrame, dict[str, str], int]:
    """Validate inputs, detect sub-score columns, and split into train/test.

    Returns early-exit dict on validation failure, or
    (train_set, test_set, sub_cols, n) on success.
    """
    populated = df[df["beat_spy_10d"].notna()].copy()
    n = len(populated)

    if n < min_samples:
        return {
            "status": "insufficient_data",
            "n_samples": n,
            "min_required": min_samples,
            "current_weights": current_weights,
            "note": (
                f"Only {n} rows with beat_spy_10d populated "
                f"(need {min_samples}). Weight recommendation deferred."
            ),
        }

    sub_cols = {s: f"{s}_score" for s in SUB_SCORES if f"{s}_score" in populated.columns}
    if not sub_cols:
        return {
            "status": "no_subscores",
            "n_samples": n,
            "current_weights": current_weights,
            "note": (
                "Sub-score columns not found. signals.json may not include sub_scores. "
                "Run load_with_subscores() before compute_weights()."
            ),
        }

    populated = populated.sort_values("score_date")
    split_idx = int(len(populated) * 0.7)
    train_set = populated.iloc[:split_idx]
    test_set = populated.iloc[split_idx:]

    if len(train_set) < min_samples:
        return {
            "status": "insufficient_data",
            "n_samples": n,
            "min_required": min_samples,
            "current_weights": current_weights,
            "note": (
                f"Only {len(train_set)} train rows after 70/30 split "
                f"(need {min_samples}). Weight recommendation deferred."
            ),
        }

    if len(test_set) < 10:
        return {
            "status": "insufficient_data",
            "n_samples": n,
            "min_required": min_samples,
            "current_weights": current_weights,
            "note": (
                f"Only {len(test_set)} test rows after 70/30 split "
                f"(need 10). Weight recommendation deferred."
            ),
        }

    return train_set, test_set, sub_cols, n


def _compute_correlations(
    data: pd.DataFrame, sub_cols: dict[str, str],
) -> dict[str, dict[str, float | None]]:
    """Compute correlation between each sub-score and beat_spy targets.

    Guards against zero-variance columns (which produce NaN correlations).
    """
    correlations: dict[str, dict] = {}
    for label, col in sub_cols.items():
        corr: dict[str, float | None] = {}
        for target in ("beat_spy_10d", "beat_spy_30d"):
            valid = data[[col, target]].dropna()
            if len(valid) >= 10 and valid[col].std() > 1e-10 and valid[target].std() > 1e-10:
                r = float(valid[col].corr(valid[target]))
                corr[target] = r if not pd.isna(r) else None
            else:
                corr[target] = None
        correlations[label] = corr
    return correlations


def _compute_ic_correlations(
    data: pd.DataFrame, sub_cols: dict[str, str],
) -> dict[str, dict[str, float | None]]:
    """Compute Spearman IC of each sub-score against continuous returns.

    Skill-composite fit target (per evaluator-revamp-260506.md). Replaces
    the legacy Pearson-on-binary-beat-SPY signal — IC ranks picks by
    return magnitude, so a +10% NVDA pick contributes more to the weight
    signal than a +1% KO pick that happens to also beat SPY by a hair.
    Same shape as ``_compute_correlations`` for drop-in substitution
    upstream.

    Targets ``return_10d`` and ``return_30d`` are continuous; the result
    dict still keys by ``beat_spy_10d`` / ``beat_spy_30d`` so the rest
    of the pipeline (`_validate_oos`, blend, write) treats it uniformly.
    """
    correlations: dict[str, dict] = {}
    target_map = {"beat_spy_10d": "return_10d", "beat_spy_30d": "return_30d"}
    for label, col in sub_cols.items():
        corr: dict[str, float | None] = {}
        for legacy_key, return_col in target_map.items():
            if return_col not in data.columns:
                corr[legacy_key] = None
                continue
            valid = data[[col, return_col]].dropna()
            if (
                len(valid) >= 10
                and valid[col].std() > 1e-10
                and valid[return_col].std() > 1e-10
            ):
                # Spearman = Pearson on ranks. Avoids requiring scipy here.
                ranks_col = valid[col].rank()
                ranks_ret = valid[return_col].rank()
                r = float(ranks_col.corr(ranks_ret))
                corr[legacy_key] = r if not pd.isna(r) else None
            else:
                corr[legacy_key] = None
        correlations[label] = corr
    return correlations


def _validate_oos(
    train_correlations: dict,
    test_correlations: dict,
    sub_cols: dict[str, str],
    w10: float,
    w30: float,
) -> tuple[bool, float]:
    """Compare train vs test correlations and return (oos_passed, degradation)."""
    train_total = 0.0
    test_total = 0.0
    for label in sub_cols:
        for target, weight in [("beat_spy_10d", w10), ("beat_spy_30d", w30)]:
            train_val = train_correlations.get(label, {}).get(target) or 0.0
            test_val = test_correlations.get(label, {}).get(target) or 0.0
            train_total += weight * abs(train_val)
            test_total += weight * abs(test_val)

    degradation = 1.0 - (test_total / train_total) if train_total > 0 else 0.0
    return degradation < 0.20, degradation


def compute_weights(
    df: pd.DataFrame,
    current_weights: dict | None = None,
    min_samples: int = 30,
    bucket: str | None = None,
) -> dict:
    """
    Compute suggested scoring weights from sub-score vs. beat_spy correlations.

    Args:
        df:               score_performance DataFrame with quant_score,
                          qual_score columns (from load_with_subscores).
        current_weights:  Current weights dict. Defaults to DEFAULT_WEIGHTS.
        min_samples:      Minimum rows with beat_spy_10d populated to proceed.

    Returns:
        {
            "status": "ok" | "insufficient_data" | "no_subscores",
            "n_samples": int,
            "confidence": "low" | "medium" | "high",
            "current_weights": {"quant": 0.50, "qual": 0.50},
            "correlations": {
                "quant":   {"beat_spy_10d": 0.11, "beat_spy_30d": 0.14},
                "qual":    {"beat_spy_10d": 0.18, "beat_spy_30d": 0.22},
            },
            "suggested_weights": {"quant": 0.48, "qual": 0.52},
            "changes": {"quant": -0.02, "qual": +0.02},
            "note": "..."
        }
    """
    if current_weights is None:
        current_weights = _cfg.get("default_weights", _DEFAULT_WEIGHTS).copy()

    # Phase 1: Validate inputs and split
    result = _validate_and_split(df, current_weights, min_samples)
    if isinstance(result, dict):
        return result  # early exit
    train_set, test_set, sub_cols, n = result

    # Phase 2: Compute correlations on train set.
    # The skill-composite fit target (evaluator-revamp-260506.md) uses
    # Spearman IC against continuous returns instead of Pearson against
    # binary beat_spy. Default off — flip via config.weight_optimizer
    # `use_skill_composite_target: true` once shadow data validates the
    # weight-shift direction.
    use_skill_composite = bool(_cfg.get("use_skill_composite_target", False))
    if use_skill_composite:
        correlations = _compute_ic_correlations(train_set, sub_cols)
    else:
        correlations = _compute_correlations(train_set, sub_cols)

    # Derive suggested weights from horizon-blended correlations
    horizon = _cfg.get("horizon_blend", _HORIZON_BLEND)
    w10 = horizon.get("beat_spy_10d", 0.50)
    w30 = horizon.get("beat_spy_30d", 0.50)
    weighted_corrs: dict[str, float] = {}
    for label, corr in correlations.items():
        c10 = corr.get("beat_spy_10d") or 0.0
        c30 = corr.get("beat_spy_30d") or 0.0
        weighted_corrs[label] = max(0.0, w10 * c10 + w30 * c30)

    total_corr = sum(weighted_corrs.values())
    if total_corr == 0:
        pure_suggested = current_weights.copy()
    else:
        pure_suggested = {k: v / total_corr for k, v in weighted_corrs.items()}

    # Blend toward data-driven weights — scale blend factor with sample size
    min_blend = _cfg.get("blend_factor_min", _cfg.get("blend_factor", _BLEND_FACTOR))
    max_blend = _cfg.get("blend_factor_max", 0.50)
    blend_ramp_samples = _cfg.get("blend_ramp_samples", 500)
    blend = min(max_blend, min_blend + (max_blend - min_blend) * (len(train_set) / blend_ramp_samples))
    logger.info("Blend factor: %.3f (n_train=%d, ramp=%d)", blend, len(train_set), blend_ramp_samples)
    suggested = {
        k: round(current_weights.get(k, 0.0) * (1 - blend) + pure_suggested.get(k, 0.0) * blend, 3)
        for k in SUB_SCORES
    }

    # Re-normalize to ensure sum == 1.0
    total = sum(suggested.values())
    suggested = {k: round(v / total, 3) for k, v in suggested.items()}

    changes = {k: round(suggested[k] - current_weights.get(k, 0.0), 3) for k in SUB_SCORES}

    conf_med = _cfg.get("confidence_medium", _CONFIDENCE_MEDIUM)
    conf_low = _cfg.get("confidence_low", _CONFIDENCE_LOW)
    confidence = (
        "high" if n >= conf_med
        else "medium" if n >= conf_low
        else "low"
    )

    # Phase 3: Out-of-sample validation. Use the matching correlation
    # estimator on the test set so train↔test parity is preserved.
    if use_skill_composite:
        test_correlations = _compute_ic_correlations(test_set, sub_cols)
    else:
        test_correlations = _compute_correlations(test_set, sub_cols)
    oos_passed, oos_degradation = _validate_oos(
        correlations, test_correlations, sub_cols, w10, w30,
    )

    stability = _check_stability(suggested, bucket=bucket)

    return {
        "status": "ok",
        "n_samples": n,
        "n_train": len(train_set),
        "n_test": len(test_set),
        "confidence": confidence,
        "current_weights": current_weights,
        "correlations": correlations,
        "test_correlations": test_correlations,
        "oos_passed": oos_passed,
        "oos_degradation": round(oos_degradation, 4),
        "suggested_weights": suggested,
        "changes": changes,
        "blend_factor": round(blend, 3),
        "stability": stability,
        "fit_target": "skill_composite_ic" if use_skill_composite else "beat_spy_pearson",
        "note": (
            f"Based on {n} signals (train={len(train_set)}, test={len(test_set)}). "
            f"Confidence: {confidence}. Blend factor: {blend:.2f}. "
            f"OOS degradation: {oos_degradation:.1%} "
            f"({'PASS' if oos_passed else 'FAIL — weights not applied'}). "
            f"Fit target: {'IC vs return_10d/30d (skill composite)' if use_skill_composite else 'Pearson vs beat_spy (legacy)'}."
        ),
    }


def _check_stability(suggested: dict, bucket: str | None = None) -> dict:
    """
    Load prior 3 weeks' weight recommendations from S3 history and check
    for direction reversals (e.g., quant weight increased last week
    but decreased this week).

    Returns:
        {"weeks_loaded": N, "reversals": [...], "stable": True/False}
    """
    if bucket is None:
        # Bucket not available at compute time — will be populated by caller
        return {"weeks_loaded": 0, "reversals": [], "stable": True, "note": "no bucket provided"}

    from datetime import date as _date, timedelta
    history = []
    s3 = boto3.client("s3")

    # Load prior 3 weeks' history files
    for weeks_ago in range(1, 4):
        d = _date.today() - timedelta(weeks=weeks_ago)
        key = f"config/scoring_weights_history/{d.isoformat()}.json"
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj["Body"].read())
            history.append({"date": d.isoformat(), "weights": {k: data.get(k) for k in SUB_SCORES}})
        except Exception:
            continue

    if not history:
        return {"weeks_loaded": 0, "reversals": [], "stable": True}

    # Add current suggestion to form the full window
    all_weeks = history[::-1]  # oldest first
    all_weeks.append({"date": _date.today().isoformat(), "weights": suggested})

    reversals = []
    for k in SUB_SCORES:
        deltas = []
        for i in range(1, len(all_weeks)):
            prev = all_weeks[i - 1]["weights"].get(k)
            curr = all_weeks[i]["weights"].get(k)
            if prev is not None and curr is not None:
                deltas.append(curr - prev)
        # Check for direction reversal (sign change in consecutive deltas)
        for i in range(1, len(deltas)):
            if deltas[i - 1] != 0 and deltas[i] != 0:
                if (deltas[i - 1] > 0) != (deltas[i] > 0):
                    direction_prev = "↑" if deltas[i - 1] > 0 else "↓"
                    direction_curr = "↑" if deltas[i] > 0 else "↓"
                    reversals.append(
                        f"{k}: {direction_prev} → {direction_curr} "
                        f"(weeks {i} → {i + 1})"
                    )

    return {
        "weeks_loaded": len(history),
        "reversals": reversals,
        "stable": len(reversals) == 0,
    }


def apply_weights(result: dict, bucket: str) -> dict:
    """
    Apply suggested weights to S3 if guardrails pass.

    Writes s3://{bucket}/config/scoring_weights.json. The research Lambda
    reads this file at cold-start and uses it in place of universe.yaml defaults.

    Guardrails (all must pass):
      - confidence must be "medium" or "high" (>= 50 samples)
      - no single weight changes by more than MAX_SINGLE_CHANGE (15%)
      - at least one weight changes by more than MIN_MEANINGFUL_CHANGE (2%)

    Args:
        result: dict returned by compute_weights().
        bucket: S3 bucket (same as signals_bucket).

    Returns:
        {"applied": True, "weights": {...}, "n_samples": int, "confidence": str}
        or {"applied": False, "reason": str}
    """
    if result.get("status") != "ok":
        return {"applied": False, "reason": f"status={result.get('status')}"}

    if result.get("oos_passed") is False:
        return {"applied": False, "reason": f"OOS validation failed (degradation={result.get('oos_degradation', 0):.1%})"}

    confidence = result.get("confidence", "low")
    if confidence == "low":
        return {"applied": False, "reason": "confidence too low — need medium or high (50+ samples)"}

    max_single = _cfg.get("max_single_change", _MAX_SINGLE_CHANGE)
    min_meaningful = _cfg.get("min_meaningful_change", _MIN_MEANINGFUL_CHANGE)

    changes = result.get("changes", {})
    max_change = max(abs(v) for v in changes.values()) if changes else 0
    meaningful = any(abs(v) >= min_meaningful for v in changes.values())

    if max_change > max_single:
        return {
            "applied": False,
            "reason": f"largest change {max_change:.1%} exceeds {max_single:.0%} limit — skipping to avoid instability",
        }

    if not meaningful:
        return {
            "applied": False,
            "reason": f"all changes < {min_meaningful:.0%} — not worth updating",
        }

    suggested = result.get("suggested_weights", {})
    payload = {
        **suggested,
        "updated_at": str(date.today()),
        "n_samples": result.get("n_samples"),
        "confidence": confidence,
        "fit_target": result.get("fit_target", "beat_spy_pearson"),
    }

    # Shadow mode (evaluator-revamp PR 6): when the skill-composite fit
    # target is enabled but its `enforce` sub-flag is False, write the
    # candidate weights to a shadow archive instead of the live config.
    # Operator inspects shadow vs production weight deltas for ~4 weeks
    # before flipping enforce → on (per evaluator-revamp-260506.md).
    fit_target = result.get("fit_target", "beat_spy_pearson")
    skill_composite_enabled = fit_target == "skill_composite_ic"
    enforce_skill_composite = bool(_cfg.get("enforce_skill_composite", False))
    shadow_only = skill_composite_enabled and not enforce_skill_composite

    if shadow_only:
        s3 = boto3.client("s3")
        # Canonical eval-style archive layout per lib v0.8.0 — flat
        # {prefix}/{run_id}.json + latest.json sidecar (YYMMDDHHMM run_id)
        run_id = new_eval_run_id()
        shadow_key = eval_artifact_key(S3_SHADOW_WEIGHTS_PREFIX, run_id)
        shadow_latest_key = eval_latest_key(S3_SHADOW_WEIGHTS_PREFIX)
        body_bytes = json.dumps(payload, indent=2)
        try:
            s3.put_object(
                Bucket=bucket, Key=shadow_key, Body=body_bytes,
                ContentType="application/json",
            )
            s3.put_object(
                Bucket=bucket, Key=shadow_latest_key, Body=body_bytes,
                ContentType="application/json",
            )
            logger.info(
                "Skill-composite weights logged to shadow path "
                "(enforce_skill_composite=False): s3://%s/%s (+ latest.json sidecar)",
                bucket, shadow_key,
            )
        except Exception as e:
            logger.warning("Shadow weights write failed (non-fatal): %s", e)
        return {
            "applied": False,
            "reason": "shadow mode — skill_composite enabled, enforce_skill_composite=False",
            "shadow_weights": suggested,
            "shadow_key": shadow_key,
            "fit_target": fit_target,
        }

    from optimizer.rollback import save_previous
    save_previous(bucket, "scoring_weights")

    s3 = boto3.client("s3")
    body = json.dumps(payload, indent=2)
    try:
        s3.put_object(
            Bucket=bucket,
            Key=S3_WEIGHTS_KEY,
            Body=body,
            ContentType="application/json",
        )
        logger.info(
            "Scoring weights updated in S3: %s (n=%s, confidence=%s, fit_target=%s)",
            suggested, payload["n_samples"], confidence, fit_target,
        )
    except Exception as e:
        logger.error("CRITICAL: Failed to write scoring weights to S3: %s", e)
        return {"applied": False, "reason": f"S3 write failed: {e}"}

    # Canonical eval-style archive layout per lib v0.8.0 — see shadow path above
    history_run_id = new_eval_run_id()
    history_prefix = "config/scoring_weights_history"
    history_key = eval_artifact_key(history_prefix, history_run_id)
    history_latest_key = eval_latest_key(history_prefix)
    try:
        s3.put_object(
            Bucket=bucket, Key=history_key, Body=body,
            ContentType="application/json",
        )
        s3.put_object(
            Bucket=bucket, Key=history_latest_key, Body=body,
            ContentType="application/json",
        )
        logger.info(
            "Scoring weights archived to s3://%s/%s (+ latest.json sidecar)",
            bucket, history_key,
        )
    except Exception as e:
        logger.warning("Failed to archive scoring weights history (non-fatal): %s", e)

    return {
        "applied": True,
        "weights": suggested,
        "n_samples": result.get("n_samples"),
        "confidence": confidence,
    }
