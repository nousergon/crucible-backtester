"""
executor_optimizer.py — recommend optimal executor parameters from param sweep results.

Reads the param sweep DataFrame (grid search over risk + strategy params),
extracts the best-performing combination, and writes recommendations to S3
for the Executor Lambda to read on cold-start.

Two ranking paths, gated by ``executor_optimizer.use_skill_composite_target``
in config (default: legacy):

- **Legacy (Sharpe-with-drawdown):** ``_combined_score = sharpe_ratio
  - drawdown_penalty_weight × |max_drawdown|``. Stamps
  ``fit_target="sharpe_legacy"``. Pre-evaluator-revamp behavior.
- **Skill-composite:** ranks by ``sortino_ratio`` (primary, skilled
  downside-aware return) with ``total_alpha`` as tiebreaker. Stamps
  ``fit_target="skill_composite"``. Aligns with the evaluator-revamp
  2026-05-06 metric stack — Sortino + CVaR + risk-matched-alpha are the
  skilled-risk-taking signals; raw alpha vs SPY is presentation framing
  ("did we beat the market") that doesn't reward taking the right risk
  per unit of downside variance. Mirrors the activation pattern shipped
  for ``weight_optimizer`` (PR #145 / PR 6 of evaluator revamp).

Promotion to live S3 (``config/executor_params.json``) is gated by
``executor_optimizer.enforce_skill_composite``: when the skill-composite
ranking is computed but enforcement is off, ``apply()`` writes to a shadow
archive at ``config/executor_params_shadow_history/{date}.json`` instead.
This mirrors the two-stage activation pattern from ``weight_optimizer``.

Only safe-to-tune params are recommended; drawdown circuit breaker and
sector/equity limits are excluded from auto-tuning.
"""

import json
import logging
from datetime import date

import boto3
import pandas as pd
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def _safe_float(v) -> float | None:
    """Convert to float, returning None for NaN/None."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return round(float(v), 4)


def _to_native(v):
    """Convert numpy/pandas scalars to native Python types for JSON serialization."""
    import numpy as np
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v

S3_PARAMS_KEY = "config/executor_params.json"
S3_SHADOW_PREFIX = "config/executor_params_shadow_history"

# Params safe to auto-tune via sweep results.
# Excluded: drawdown_circuit_breaker, max_sector_pct, max_equity_pct (too dangerous)
SAFE_PARAMS = [
    "atr_multiplier",
    "time_decay_reduce_days",
    "time_decay_exit_days",
    "min_score",
    "max_position_pct",
    "reduce_fraction",
    "atr_sizing_target_risk",
    "confidence_sizing_min",
    "confidence_sizing_range",
    "staleness_decay_per_day",
    "earnings_sizing_reduction",
    "earnings_proximity_days",
    "momentum_gate_threshold",
    "correlation_block_threshold",
    "profit_take_pct",
    "momentum_exit_threshold",
]

# Factory defaults — the values the executor uses when no S3 config exists.
# These match executor/strategies/config.py + risk.yaml.example shipped defaults.
# Used for drift monitoring in weekly reports.
FACTORY_DEFAULTS = {
    "atr_multiplier": 2.5,
    "time_decay_reduce_days": 7,
    "time_decay_exit_days": 14,
    "min_score": 70,
    "max_position_pct": 0.05,
    "reduce_fraction": 0.50,
    "atr_sizing_target_risk": 0.02,
    "confidence_sizing_min": 0.70,
    "confidence_sizing_range": 0.60,
    "staleness_decay_per_day": 0.03,
    "earnings_sizing_reduction": 0.50,
    "earnings_proximity_days": 5,
    "momentum_gate_threshold": -5.0,
    "correlation_block_threshold": 0.80,
    "profit_take_pct": 0.25,
    "momentum_exit_threshold": -15.0,
}

# ── Fallback defaults (override via executor_optimizer section in config.yaml) ──
_MIN_VALID_COMBOS = 5
_MIN_SHARPE_IMPROVEMENT = 0.10
_MIN_SORTINO_IMPROVEMENT = 0.05
_MIN_TRADES_TO_PROMOTE = 50

# Module-level config ref — set by init_config() from backtest.py
_cfg: dict = {}


def init_config(config: dict) -> None:
    """Load executor_optimizer section from backtester config."""
    global _cfg
    _cfg = config.get("executor_optimizer", {})


def read_current_params(bucket: str) -> dict:
    """
    Read current executor params from S3, falling back to factory defaults.

    Returns a dict of safe-to-tune params only (keys in SAFE_PARAMS).
    """
    from botocore.exceptions import ClientError

    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=S3_PARAMS_KEY)
        data = json.loads(obj["Body"].read())
        params = {k: data[k] for k in SAFE_PARAMS if k in data}
        if params:
            logger.info(
                "Current executor params from S3 (updated %s): %s",
                data.get("updated_at", "unknown"), params,
            )
            return params
    except ClientError as e:
        # NoSuchKey is expected on the first run (no params have been
        # auto-applied yet) — fall back to factory defaults cleanly.
        # Any other ClientError (permissions, network) means the optimizer
        # would compare sweep results against the wrong baseline and may
        # oscillate params week-over-week. Raise so the run fails loud
        # instead of silently making decisions on FACTORY_DEFAULTS.
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            logger.info(
                "No executor params in s3://%s/%s yet (first run?) — "
                "using factory defaults",
                bucket, S3_PARAMS_KEY,
            )
        else:
            logger.error(
                "Failed to read current executor params from s3://%s/%s: %s. "
                "Optimizer cannot safely run without a valid baseline.",
                bucket, S3_PARAMS_KEY, e, exc_info=True,
            )
            raise
    except Exception as e:
        # JSON parse error, type error, etc. The params file exists but is
        # corrupted — optimizer should not proceed.
        logger.error(
            "Failed to parse current executor params from s3://%s/%s: %s. "
            "Optimizer cannot safely run against a corrupt baseline.",
            bucket, S3_PARAMS_KEY, e, exc_info=True,
        )
        raise

    return FACTORY_DEFAULTS.copy()


def recommend(sweep_df: pd.DataFrame, base_config: dict, current_params: dict | None = None) -> dict:
    """
    Extract best executor params from param sweep results.

    Args:
        sweep_df: DataFrame from param_sweep.sweep().
        base_config: Base config dict (used for current/baseline values).
        current_params: Current executor params from S3 (the values the system is
            actually using). When provided, the sweep baseline is the combo closest
            to these params — so the optimizer iterates on last week's best rather
            than comparing against the worst combo.

    Returns:
        {
            "status": "ok" | "insufficient_data" | "no_improvement" | ...,
            "fit_target": "sharpe_legacy" | "skill_composite",
            "baseline_params": {...},
            "recommended_params": {...},
            "factory_defaults": {...},
            "baseline_sharpe": float,
            "best_sharpe": float,
            "best_alpha": float | None,
            "baseline_alpha": float | None,
            "improvement_pct": float,
            ...
        }
    """
    if sweep_df is None or sweep_df.empty:
        return {"status": "insufficient_data", "note": "No sweep results available"}

    if getattr(sweep_df, "attrs", {}).get("sweep_low_completion"):
        pct = sweep_df.attrs.get("sweep_completion_pct", 0)
        return {
            "status": "insufficient_data",
            "note": f"Sweep completion rate too low ({pct:.0%} < 50%) — results unreliable",
        }

    min_combos = _cfg.get("min_valid_combos", _MIN_VALID_COMBOS)
    if "sharpe_ratio" not in sweep_df.columns:
        return {
            "status": "insufficient_data",
            "note": "No simulations produced sharpe_ratio — all combos failed or had insufficient coverage",
        }
    valid = sweep_df[sweep_df["sharpe_ratio"].notna()].copy()
    if len(valid) < min_combos:
        return {
            "status": "insufficient_data",
            "n_valid": len(valid),
            "min_required": min_combos,
            "note": f"Only {len(valid)} valid combos (need {min_combos})",
        }

    # Identify param columns (everything that's not a stat column)
    stat_cols = {
        "total_return", "total_alpha", "spy_return", "sharpe_ratio",
        "max_drawdown", "calmar_ratio", "total_trades", "win_rate",
        "error", "status", "dates_simulated", "total_orders", "note",
    }
    param_cols = [c for c in valid.columns if c in SAFE_PARAMS]

    if not param_cols:
        return {"status": "no_params", "note": "No safe params found in sweep results"}

    # Gate: refuse to promote when the best combo has too few trades.
    # With small sample sizes the Sharpe is dominated by noise — any
    # "optimal" params are just overfitting to a handful of outcomes.
    min_trades = _cfg.get("min_trades_to_promote", _MIN_TRADES_TO_PROMOTE)
    if "total_trades" in valid.columns:
        best_trades = valid["total_trades"].max()
        if best_trades < min_trades:
            return {
                "status": "insufficient_trades",
                "best_trades": int(best_trades),
                "min_required": min_trades,
                "note": (
                    f"Best combo has only {int(best_trades)} trades (need {min_trades}+). "
                    f"Refusing to promote — sample size too small to trust."
                ),
            }

    # Ranking: legacy (Sharpe-with-drawdown) vs skill-composite (alpha-first)
    # — gated by `use_skill_composite_target` config. Mirrors the
    # weight_optimizer fit-target switch shipped 2026-05-06 (PR #145).
    use_skill_composite = bool(_cfg.get("use_skill_composite_target", False))

    if use_skill_composite:
        # Skill-composite ranking: sortino_ratio (primary, skilled
        # downside-aware return) + total_alpha (tiebreaker, presentation).
        # Sortino aligns with the evaluator-revamp metric stack: it rewards
        # configs that extract return per unit of *downside* variance — the
        # exact "intelligent risk-taking" signal. Alpha vs SPY is kept as a
        # tiebreaker because among Sortino-equivalent configs, beating SPY
        # is weakly preferable for end-user-headline framing.
        if "sortino_ratio" not in valid.columns:
            return {
                "status": "insufficient_data",
                "note": (
                    "use_skill_composite_target is on but sweep produced no "
                    "sortino_ratio column — cannot rank by skill-composite."
                ),
                "fit_target": "skill_composite",
            }
        valid_sortino = valid["sortino_ratio"].notna().sum()
        if valid_sortino == 0:
            return {
                "status": "insufficient_data",
                "note": "All sortino_ratio values are NaN — cannot rank by skill-composite.",
                "fit_target": "skill_composite",
            }
        sort_cols = ["sortino_ratio"]
        if "total_alpha" in valid.columns:
            sort_cols.append("total_alpha")
        valid = valid.sort_values(sort_cols, ascending=False)
        fit_target = "skill_composite"
    else:
        # Legacy multi-metric ranking: Sharpe primary, penalize drawdown.
        # Combined score = sharpe_ratio - drawdown_penalty_weight * |max_drawdown|
        # This prevents promoting fragile param sets where one big win masks many losses.
        dd_penalty_weight = _cfg.get("drawdown_penalty_weight", 0.5)
        if "max_drawdown" in valid.columns:
            valid["_combined_score"] = (
                valid["sharpe_ratio"]
                - dd_penalty_weight * valid["max_drawdown"].fillna(0).abs()
            )
        else:
            valid["_combined_score"] = valid["sharpe_ratio"]
        valid = valid.sort_values("_combined_score", ascending=False)
        fit_target = "sharpe_legacy"

    # Baseline: find the combo closest to current S3 params (iterative learning).
    # Falls back to worst combo by ranking metric if no current_params provided.
    if current_params:
        baseline_row = _find_closest_combo(valid, param_cols, current_params)
    else:
        baseline_row = valid.iloc[-1]

    baseline_sharpe = baseline_row["sharpe_ratio"]
    best_row = valid.iloc[0]
    best_sharpe = best_row["sharpe_ratio"]

    # Alpha + Sortino — informational under legacy, gating under skill-composite.
    best_alpha = _safe_float(best_row.get("total_alpha"))
    baseline_alpha = _safe_float(baseline_row.get("total_alpha"))
    best_sortino = _safe_float(best_row.get("sortino_ratio"))
    baseline_sortino = _safe_float(baseline_row.get("sortino_ratio"))

    if use_skill_composite:
        # Improvement gate is sortino-based when ranking is sortino-first.
        if baseline_sortino is None or best_sortino is None:
            improvement_pct = 0.0
        elif baseline_sortino == 0:
            improvement_pct = float("inf") if best_sortino > 0 else 0.0
        else:
            improvement_pct = (best_sortino - baseline_sortino) / abs(baseline_sortino)
    else:
        if baseline_sharpe == 0:
            improvement_pct = float("inf") if best_sharpe > 0 else 0.0
        else:
            improvement_pct = (best_sharpe - baseline_sharpe) / abs(baseline_sharpe)

    recommended = {col: best_row[col] for col in param_cols if pd.notna(best_row[col])}
    # Convert numpy types to native Python
    recommended = {k: _to_native(v) for k, v in recommended.items()}

    baseline = {col: baseline_row[col] for col in param_cols if pd.notna(baseline_row[col])}
    baseline = {k: float(v) if isinstance(v, (int, float)) else v for k, v in baseline.items()}

    # Baseline diagnostics: rank of baseline combo and distance info
    baseline_idx = valid.index.get_loc(baseline_row.name) if baseline_row.name in valid.index else len(valid) - 1
    baseline_combo_rank = int(baseline_idx) + 1  # 1-based rank by combined_score

    # Count combos closer in param space to current S3 params
    n_closer_combos = 0
    if current_params:
        baseline_dist = _l2_distance(baseline_row, param_cols, current_params, valid)
        for idx, row in valid.iterrows():
            if idx == baseline_row.name:
                continue
            row_dist = _l2_distance(row, param_cols, current_params, valid)
            if row_dist < baseline_dist:
                n_closer_combos += 1
    else:
        baseline_dist = 0.0

    common_fields = {
        "fit_target": fit_target,
        "baseline_params": baseline,
        "recommended_params": recommended,
        "factory_defaults": FACTORY_DEFAULTS.copy(),
        "baseline_sharpe": round(float(baseline_sharpe), 4),
        "best_sharpe": round(float(best_sharpe), 4),
        "best_alpha": best_alpha,
        "baseline_alpha": baseline_alpha,
        "best_sortino": best_sortino,
        "baseline_sortino": baseline_sortino,
        "improvement_pct": round(improvement_pct, 4),
        "baseline_combo_rank": baseline_combo_rank,
        "baseline_distance": round(float(baseline_dist), 4),
        "n_closer_combos": n_closer_combos,
    }

    if use_skill_composite:
        # Guard: never auto-apply params from a negative-Sortino optimization.
        # Negative Sortino = the strategy's downside-aware return is loss-making.
        # Mirrors the negative-Sharpe guard but on the skilled-risk-taking axis.
        if best_sortino is None:
            return {
                "status": "insufficient_data",
                **common_fields,
                "note": "Best combo has no sortino_ratio — cannot evaluate skill-composite gate.",
            }
        if best_sortino < 0:
            return {
                "status": "negative_sortino",
                **common_fields,
                "note": (
                    f"Best Sortino ({best_sortino:.4f}) is negative — every combo's "
                    f"downside-aware return is loss-making. Refusing to auto-apply. "
                    f"Review signal quality and backtest data before tuning."
                ),
            }

        min_improvement = _cfg.get("min_sortino_improvement", _MIN_SORTINO_IMPROVEMENT)
        if improvement_pct < min_improvement:
            return {
                "status": "no_improvement",
                **common_fields,
                "note": (
                    f"Best Sortino ({best_sortino:.4f}) only {improvement_pct:.1%} better than "
                    f"baseline ({baseline_sortino:.4f}). Need {min_improvement:.0%}+ to recommend."
                ),
            }

        return {
            "status": "ok",
            **common_fields,
            "n_combos_tested": len(valid),
            "note": (
                f"Best combo improves Sortino by {improvement_pct:.1%} "
                f"({baseline_sortino:.4f} → {best_sortino:.4f}) across {len(valid)} combos "
                f"(skill_composite ranking: sortino primary, alpha tiebreaker; "
                f"best_alpha={best_alpha} shown for presentation framing only)."
            ),
        }

    # ── Legacy Sharpe-with-drawdown path ──
    # Guard: never auto-apply params from a negative-Sharpe optimization.
    # A negative best Sharpe means every combo lost money — the "best" is just
    # the least bad, and auto-applying it can silently break live trading.
    if best_sharpe < 0:
        return {
            "status": "negative_sharpe",
            **common_fields,
            "note": (
                f"Best Sharpe ({best_sharpe:.4f}) is negative — all combos lost money. "
                f"Refusing to auto-apply. Review signal quality and backtest data before tuning."
            ),
        }

    min_improvement = _cfg.get("min_sharpe_improvement", _MIN_SHARPE_IMPROVEMENT)
    if improvement_pct < min_improvement:
        return {
            "status": "no_improvement",
            **common_fields,
            "note": (
                f"Best Sharpe ({best_sharpe:.4f}) only {improvement_pct:.1%} better than "
                f"baseline ({baseline_sharpe:.4f}). Need {min_improvement:.0%}+ to recommend."
            ),
        }

    return {
        "status": "ok",
        **common_fields,
        "n_combos_tested": len(valid),
        "note": (
            f"Best combo improves Sharpe by {improvement_pct:.1%} "
            f"({baseline_sharpe:.4f} → {best_sharpe:.4f}) across {len(valid)} combos."
        ),
    }


def _find_closest_combo(
    valid: pd.DataFrame, param_cols: list[str], target: dict
) -> pd.Series:
    """
    Find the sweep row whose params are closest to `target` (L2 distance,
    normalized per-param by range). Returns the closest row.
    """
    # Compute normalized distance for each row
    distances = pd.Series(0.0, index=valid.index)
    for col in param_cols:
        if col not in target:
            continue
        col_vals = pd.to_numeric(valid[col], errors="coerce")
        col_range = col_vals.max() - col_vals.min()
        if col_range == 0:
            continue
        distances += ((col_vals - float(target[col])) / col_range) ** 2

    return valid.loc[distances.idxmin()]


def _l2_distance(row: pd.Series, param_cols: list[str], target: dict, valid: pd.DataFrame) -> float:
    """Compute normalized L2 distance from a single row to target params."""
    dist = 0.0
    for col in param_cols:
        if col not in target:
            continue
        col_vals = pd.to_numeric(valid[col], errors="coerce")
        col_range = col_vals.max() - col_vals.min()
        if col_range == 0:
            continue
        val = float(row[col]) if pd.notna(row[col]) else 0.0
        dist += ((val - float(target[col])) / col_range) ** 2
    return dist ** 0.5


def validate_holdout(
    result: dict,
    sim_fn,
    dates: list[str],
    config: dict,
) -> dict:
    """
    Re-run best params on the last 30% of signal dates as a holdout check.

    Compares holdout Sharpe to train Sharpe; requires holdout >= 50% of train.
    Updates result in-place with holdout metrics.

    Args:
        result: dict from recommend() with status="ok".
        sim_fn: callable(combo_config) -> stats dict (same as param sweep sim_fn).
        dates: full list of signal dates (chronological).
        config: base config dict.

    Returns:
        result dict with added holdout_sharpe, holdout_passed fields.
    """
    if result.get("status") != "ok":
        return result

    recommended = result.get("recommended_params", {})
    if not recommended:
        return result

    # Split dates 70/30
    split_idx = int(len(dates) * 0.7)
    holdout_dates = dates[split_idx:]

    if len(holdout_dates) < 3:
        result["holdout_passed"] = True
        result["holdout_note"] = f"Only {len(holdout_dates)} holdout dates — skipped validation"
        return result

    # Build holdout config with recommended params
    holdout_config = {**config, **recommended}

    try:
        holdout_stats = sim_fn(holdout_config)
    except Exception as e:
        logger.warning("Holdout validation failed: %s", e)
        result["holdout_passed"] = True
        result["holdout_note"] = f"Holdout simulation error: {e}"
        return result

    holdout_sharpe = holdout_stats.get("sharpe_ratio")
    holdout_sortino = holdout_stats.get("sortino_ratio")
    fit_target = result.get("fit_target", "sharpe_legacy")

    # Pick the metric matching the recommend()-side ranking axis. Skill-composite
    # ranks by sortino-first, so the holdout-stability check follows sortino;
    # legacy ranks by Sharpe, so the holdout check follows Sharpe.
    if fit_target == "skill_composite":
        metric_name = "Sortino"
        holdout_value = holdout_sortino
        train_value = result.get("best_sortino", 0)
    else:
        metric_name = "Sharpe"
        holdout_value = holdout_sharpe
        train_value = result.get("best_sharpe", 0)

    if holdout_value is None:
        result["holdout_passed"] = True
        result["holdout_note"] = f"Holdout produced no {metric_name} — skipped validation"
        return result

    result[f"holdout_{metric_name.lower()}"] = round(float(holdout_value), 4)

    # Gate: both train and holdout metric must be positive
    if train_value is None or train_value < 0:
        result["holdout_ratio"] = 0.0
        result["holdout_passed"] = False
        result["status"] = "holdout_failed"
        result["holdout_note"] = (
            f"Train {metric_name} is non-positive ({train_value}) — "
            f"cannot validate holdout ratio"
        )
        logger.warning("Executor optimizer holdout failed: %s", result["holdout_note"])
        return result

    ratio = holdout_value / train_value if train_value != 0 else 0.0
    result["holdout_ratio"] = round(ratio, 4)
    result["holdout_passed"] = ratio >= 0.50 and holdout_value > 0

    if not result["holdout_passed"]:
        result["status"] = "holdout_failed"
        if holdout_value <= 0:
            result["holdout_note"] = (
                f"Holdout {metric_name} is non-positive ({holdout_value:.4f}) — "
                f"strategy is loss-making on holdout set"
            )
        else:
            result["holdout_note"] = (
                f"Holdout {metric_name} ({holdout_value:.4f}) is {ratio:.0%} of train "
                f"({train_value:.4f}) — need >= 50%"
            )
        logger.warning("Executor optimizer holdout failed: %s", result["holdout_note"])
    else:
        result["holdout_note"] = (
            f"Holdout {metric_name} ({holdout_value:.4f}) is {ratio:.0%} of train — PASS"
        )
        logger.info("Executor optimizer holdout passed: %s", result["holdout_note"])

    return result


def apply(result: dict, bucket: str) -> dict:
    """
    Write recommended executor params to S3 if recommendation is valid.

    Two write paths:
    - **Production:** ``s3://{bucket}/config/executor_params.json`` (live)
      + ``config/executor_params_history/{date}.json`` (audit). Used when
      ``fit_target == "sharpe_legacy"`` OR when ``enforce_skill_composite``
      is true.
    - **Shadow:** ``s3://{bucket}/config/executor_params_shadow_history/{date}.json``
      only. Used when ``fit_target == "skill_composite"`` AND
      ``enforce_skill_composite`` is false. Live config is unchanged.

    Args:
        result: dict from recommend().
        bucket: S3 bucket name.

    Returns:
        {"applied": True, ...} or {"applied": False, "reason": ...}
    """
    if result.get("status") != "ok":
        return {"applied": False, "reason": f"status={result.get('status')}"}

    recommended = result.get("recommended_params", {})
    if not recommended:
        return {"applied": False, "reason": "no recommended params"}

    fit_target = result.get("fit_target", "sharpe_legacy")
    enforce_skill_composite = bool(_cfg.get("enforce_skill_composite", False))
    shadow_only = fit_target == "skill_composite" and not enforce_skill_composite

    payload = {
        **recommended,
        "updated_at": str(date.today()),
        "fit_target": fit_target,
        "best_sharpe": result.get("best_sharpe"),
        "best_alpha": result.get("best_alpha"),
        "best_sortino": result.get("best_sortino"),
        "improvement_pct": result.get("improvement_pct"),
        "n_combos_tested": result.get("n_combos_tested"),
    }

    s3 = boto3.client("s3")
    body = json.dumps(payload, indent=2)

    if shadow_only:
        shadow_key = f"{S3_SHADOW_PREFIX}/{date.today().isoformat()}.json"
        try:
            s3.put_object(
                Bucket=bucket, Key=shadow_key, Body=body, ContentType="application/json"
            )
            logger.info(
                "Executor params written to shadow archive (enforce_skill_composite=False): "
                "s3://%s/%s",
                bucket, shadow_key,
            )
        except Exception as e:
            logger.error("Failed to write executor params shadow archive: %s", e)
            return {"applied": False, "reason": f"shadow S3 write failed: {e}"}
        return {
            "applied": False,
            "reason": "shadow mode — fit_target=skill_composite, enforce_skill_composite=False",
            "shadow_key": shadow_key,
            "fit_target": fit_target,
            "params": recommended,
            "best_sharpe": result.get("best_sharpe"),
            "best_alpha": result.get("best_alpha"),
            "best_sortino": result.get("best_sortino"),
            "improvement_pct": result.get("improvement_pct"),
        }

    from optimizer.rollback import save_previous
    save_previous(bucket, "executor_params")

    try:
        s3.put_object(Bucket=bucket, Key=S3_PARAMS_KEY, Body=body, ContentType="application/json")
        logger.info(
            "Executor params updated in S3: %s (fit_target=%s)",
            recommended, fit_target,
        )
    except Exception as e:
        logger.error("CRITICAL: Failed to write executor params to S3: %s", e)
        return {"applied": False, "reason": f"S3 write failed: {e}"}

    history_key = f"config/executor_params_history/{date.today().isoformat()}.json"
    try:
        s3.put_object(Bucket=bucket, Key=history_key, Body=body, ContentType="application/json")
        logger.info("Executor params archived to s3://%s/%s", bucket, history_key)
    except Exception as e:
        logger.warning("Failed to archive executor params history (non-fatal): %s", e)

    return {
        "applied": True,
        "fit_target": fit_target,
        "params": recommended,
        "best_sharpe": result.get("best_sharpe"),
        "best_alpha": result.get("best_alpha"),
        "best_sortino": result.get("best_sortino"),
        "improvement_pct": result.get("improvement_pct"),
    }
