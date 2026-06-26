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
- **Skill-composite:** ranks by ``sortino_ratio`` — skilled downside-aware
  return — with no tiebreaker. Stamps ``fit_target="skill_composite"``.
  Aligns with the evaluator-revamp 2026-05-06 metric stack: Sortino, CVaR,
  risk-matched-alpha are the skilled-risk-taking signals; raw alpha vs SPY
  is presentation framing ("did we beat the market") that doesn't reward
  taking the right risk per unit of downside variance. Alpha still appears
  in the result dict + S3 payload for operator display — it just doesn't
  drive the ranking. Exact-Sortino ties are effectively measure-zero on a
  continuous-param sweep; pandas stable-sort resolves them deterministically.
  Mirrors the activation pattern shipped for ``weight_optimizer``
  (PR #145 / PR 6 of evaluator revamp).

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
from alpha_engine_lib.eval_artifacts import (
    eval_artifact_key,
    eval_latest_key,
    new_eval_run_id,
)
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
    # L300 (2026-06-01): confidence_sizing_min/range removed — the param sweep
    # over them was a silent no-op (predictionless sim → prediction_confidence
    # None → confidence_adj 1.0). Confidence sizing is tuned offline via p_up
    # (predictor_sizing_optimizer). FACTORY_DEFAULTS below keep them for drift
    # monitoring + executor fallback.
    "staleness_decay_per_day",
    "earnings_sizing_reduction",
    "earnings_proximity_days",
    "momentum_gate_threshold",
    "correlation_block_threshold",
    "profit_take_pct",
    "momentum_exit_threshold",
    # L300-a (2026-06-01): value_stance_drawdown_min / quality_stance_momentum_
    # threshold removed — audit confirmed they're entry GATE thresholds gated on
    # ``stance == "value"`` / ``"quality"`` in deciders.py, and stance is sourced
    # only from predictions (None in the predictionless sim) → the gate branches
    # never fire → sweeping them was a silent no-op. FACTORY_DEFAULTS below keep
    # them for executor fallback + drift monitoring. A (stance × momentum)
    # offline gate is the deferred follow-up.
    # L300 (2026-06-01): stance_size_{momentum,value,quality,catalyst} removed —
    # the sweep over them was a silent no-op (predictionless sim → stance None →
    # stance_adj 1.0). They are tuned offline against realized per-stance alpha
    # by optimizer/stance_sizing_optimizer.py (rank-IC gate, field_overlay).
    # FACTORY_DEFAULTS below keep them for drift monitoring + executor fallback.
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
    # Stance gates (default values match executor's _plan_entries defaults)
    "value_stance_drawdown_min": -0.05,
    "quality_stance_momentum_threshold": -15.0,
    # Stance-conditional sizing multipliers (default values match
    # executor's position_sizer compute_position_size defaults)
    "stance_size_momentum": 1.0,
    "stance_size_value": 0.7,
    "stance_size_quality": 0.8,
    "stance_size_catalyst": 0.6,
}

# ── Fallback defaults (override via executor_optimizer section in config.yaml) ──
_MIN_VALID_COMBOS = 5
_MIN_SHARPE_IMPROVEMENT = 0.10
_MIN_SORTINO_IMPROVEMENT = 0.05
_MIN_TRADES_TO_PROMOTE = 50
# Probabilistic Sharpe Ratio threshold — confidence that true SR > 0.
# 0.95 = 95% confidence. Used by skill-composite mode as a DSR-style
# gate before live promotion. Mirrors the precision_ci_95 gate from the
# veto skill-composite cutover (alpha-engine-backtester #166).
_MIN_PSR = 0.95

# Floor for the |baseline| denominator in improvement_pct, preventing
# blow-ups when the baseline rank metric (Sortino / Sharpe / risk-matched
# alpha) is near zero. Without this, baseline≈0 produced inf or 9828×
# readings (observed in executor_params_shadow_history/latest.json
# 2026-05-18). Operators should read `improvement_delta` (signed
# absolute) for the meaningful number when |baseline| < this floor.
_IMPROVEMENT_DENOM_FLOOR = 1e-6

# Minimum |baseline_rank| magnitude at which the percent-improvement
# framing is statistically meaningful, per rank metric. Below this
# floor, the clamp-floor above still prevents division blow-ups but
# the resulting `improvement_pct` is structurally misleading (e.g.
# baseline=1e-4 + best=0.01 → 99× "improvement" reading from absolute
# deltas of 0.01). When the baseline magnitude falls under the floor,
# the optimizer refuses to promote — the strategy's baseline is
# statistical noise and no ratio off noise can justify a config flip.
# Matches the institutional "constrained optimization, not post-hoc
# check" framing from the 2026-05-20 alpha-floor arc (CLAUDE.md SOTA
# rule).
#
# Floors are metric-specific because the rank columns roll in
# structurally different units:
#  * sortino_ratio / sharpe_ratio: dimensionless risk-adjusted return
#    ratios; typical 0–2 range, 0.05 is the floor at which the ratio
#    becomes distinguishable from zero on typical 10–12 month sweep
#    windows.
#  * alpha_vs_ew_high_vol: raw return (portfolio - vol-matched basket);
#    typical 0.01–0.10 range, 50bps (0.005) is the noise floor at
#    which the risk-matched-alpha basket comparison becomes
#    statistically meaningful.
# Operator overrides:
#  * ``executor_optimizer.min_baseline_magnitude`` (single float) is
#    honoured for backward compatibility — applied uniformly when set.
#  * ``executor_optimizer.min_baseline_magnitude_by_rank`` (dict) is
#    the per-rank-metric override path (e.g.
#    ``{sortino_ratio: 0.1, alpha_vs_ew_high_vol: 0.01}``).
_MIN_BASELINE_MAGNITUDE_BY_RANK: dict[str, float] = {
    "sortino_ratio": 0.05,
    "sharpe_ratio": 0.05,
    "alpha_vs_ew_high_vol": 0.005,
}


def _resolve_min_baseline_magnitude(rank_metric: str) -> float:
    """Pick the significance floor for ``rank_metric``, honouring
    operator overrides at config-write time."""
    by_rank_override = _cfg.get("min_baseline_magnitude_by_rank") or {}
    if rank_metric in by_rank_override:
        return float(by_rank_override[rank_metric])
    single_override = _cfg.get("min_baseline_magnitude")
    if single_override is not None:
        return float(single_override)
    return _MIN_BASELINE_MAGNITUDE_BY_RANK.get(rank_metric, 0.05)

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


def read_params_as_of(bucket: str, as_of_date) -> dict:
    """Point-in-time sibling of :func:`read_current_params` (PIT walk-forward,
    ROADMAP L2371 / Backtester Phase 3).

    Resolves the executor-params snapshot whose knowledge time ≤
    ``as_of_date`` (``optimizer.config_archive.resolve_as_of``). No eligible
    snapshot → genesis ``FACTORY_DEFAULTS`` (the documented shipped
    defaults), **never** a later snapshot — the no-future-fallback invariant
    (plan §3 / D3). Return shape mirrors :func:`read_current_params` exactly
    (``SAFE_PARAMS`` subset, or full defaults when empty) so call sites are
    contract-identical whichever path they take.
    """
    from optimizer.config_archive import resolve_as_of

    data = resolve_as_of(bucket, "executor_params", as_of_date)
    if not data:
        return FACTORY_DEFAULTS.copy()
    params = {k: data[k] for k in SAFE_PARAMS if k in data}
    return params if params else FACTORY_DEFAULTS.copy()


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
            "improvement_delta": float | None,  # signed absolute delta, read this
                                                 # when |baseline_rank| < ε
            "improvement_significant": bool,    # False ⇒ baseline magnitude is
                                                 # noise, refusing to promote
            "min_baseline_magnitude": float,    # threshold for above
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

    # Hard alpha-floor constraint — canonical-alpha framework gate.
    #
    # Filter `valid` down to combos whose backtest total_alpha >= alpha_floor
    # BEFORE ranking. Without this, single-objective ranking on Sharpe or
    # Sortino will happily promote alpha-negative "do nothing" configurations
    # (the 2026-05-20 incident: live executor_params.json carried
    # min_score=75 with best_alpha=-2.5427, both Sharpe-ranked live AND
    # Sortino-ranked shadow agreed because both ratios reward variance-
    # reduction). Per the system Objective in ~/Development/CLAUDE.md
    # ("Maximize long-term alpha") and the canonical-alpha framework spec
    # ([[anchor-gates-on-skilled-risk-not-sharpe]]), alpha-positive is
    # a hard CONSTRAINT, not a side-output — combos that fail it should
    # never reach the ranker.
    #
    # ``alpha_floor=None`` (default in this module; activated via
    # executor_optimizer.alpha_floor in backtester config.yaml) leaves the
    # gate inactive — preserves prior behavior for legacy callers / tests.
    # ``alpha_floor=0.0`` is the SOTA default; positive values (e.g. require
    # a 200bps alpha cushion) compose cleanly.
    alpha_floor = _cfg.get("alpha_floor")
    if alpha_floor is not None and "total_alpha" in valid.columns:
        n_before = len(valid)
        alpha_pos = valid[valid["total_alpha"] >= alpha_floor].copy()
        n_dropped = n_before - len(alpha_pos)
        best_alpha_in_sweep = _safe_float(valid["total_alpha"].max())
        if len(alpha_pos) == 0:
            return {
                "status": "alpha_below_floor",
                "alpha_floor": float(alpha_floor),
                "n_combos_below_floor": int(n_dropped),
                "best_alpha_in_sweep": best_alpha_in_sweep,
                "note": (
                    f"All {n_before} valid combos backtested with "
                    f"total_alpha < {alpha_floor} (best in sweep: "
                    f"{best_alpha_in_sweep}). Refusing to promote — per the "
                    f"canonical-alpha framework, alpha-positive is a hard "
                    f"constraint, not a side-output. Either signal quality "
                    f"or the param sweep grid needs review."
                ),
            }
        logger.info(
            "executor_optimizer: alpha_floor=%s dropped %d/%d combos; "
            "%d alpha-positive combos remain for ranking (best alpha in "
            "sweep: %s).",
            alpha_floor, n_dropped, n_before, len(alpha_pos),
            best_alpha_in_sweep,
        )
        valid = alpha_pos

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
    # L2170 PR 3: within the skill-composite path, optionally swap the
    # rank column from `sortino_ratio` to `alpha_vs_ew_high_vol` — the
    # Workstream D institutional risk-matched skill metric. Default off
    # until ≥2 Saturday SF cycles of shadow runs confirm the basket is
    # producing sensible numbers (high-vol portfolios show smaller
    # underperformance vs EW-high-vol than vs SPY by construction). The
    # flag composes with use_skill_composite_target — when both true,
    # rank by alpha_vs_ew_high_vol; when only the parent flag is true,
    # rank by sortino_ratio (the existing path).
    prefer_risk_matched_alpha = bool(_cfg.get("prefer_risk_matched_alpha", False))

    if use_skill_composite:
        rank_col = (
            "alpha_vs_ew_high_vol"
            if prefer_risk_matched_alpha
            else "sortino_ratio"
        )
        # Skill-composite ranking. Sortino_ratio is the default — skilled
        # downside-aware return; rewards configs that extract return per
        # unit of *downside* variance (the "intelligent risk-taking" signal
        # from evaluator-revamp-260506.md). When prefer_risk_matched_alpha
        # is on, swap to alpha_vs_ew_high_vol — Workstream D's risk-matched
        # skill metric (portfolio return minus the EW-high-vol basket's
        # return over the same dates). Alpha vs SPY surfaces in the result
        # dict for operator display but does not drive ranking; raw alpha
        # is end-user-headline framing, not the optimizer's fit target.
        # Exact-rank-col ties are effectively measure-zero on a continuous-
        # param sweep; pandas stable-sort handles them deterministically.
        if rank_col not in valid.columns:
            return {
                "status": "insufficient_data",
                "note": (
                    f"use_skill_composite_target is on (rank_col={rank_col}) "
                    f"but sweep produced no {rank_col} column — cannot rank "
                    f"by skill-composite."
                ),
                "fit_target": "skill_composite",
            }
        valid_rank = valid[rank_col].notna().sum()
        if valid_rank == 0:
            return {
                "status": "insufficient_data",
                "note": (
                    f"All {rank_col} values are NaN — cannot rank by "
                    f"skill-composite."
                ),
                "fit_target": "skill_composite",
            }
        valid = valid.sort_values(rank_col, ascending=False)
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

    # Alpha + Sortino + risk-matched-alpha — informational under legacy,
    # gating under skill-composite (which one gates depends on
    # prefer_risk_matched_alpha within the skill-composite path).
    best_alpha = _safe_float(best_row.get("total_alpha"))
    baseline_alpha = _safe_float(baseline_row.get("total_alpha"))
    best_sortino = _safe_float(best_row.get("sortino_ratio"))
    baseline_sortino = _safe_float(baseline_row.get("sortino_ratio"))
    best_alpha_vs_ew_high_vol = _safe_float(best_row.get("alpha_vs_ew_high_vol"))
    baseline_alpha_vs_ew_high_vol = _safe_float(baseline_row.get("alpha_vs_ew_high_vol"))

    # Baseline-significance floor: when |baseline_rank| is below this
    # magnitude, the % framing of improvement_pct is structurally
    # noise — operator can read `improvement_delta` but the optimizer
    # refuses to promote off such a baseline (see the gate branches
    # below). Floor is per rank metric (different units across the
    # three paths); _resolve_min_baseline_magnitude honours operator
    # overrides.
    if use_skill_composite:
        _rank_metric_name = (
            "alpha_vs_ew_high_vol" if prefer_risk_matched_alpha
            else "sortino_ratio"
        )
    else:
        _rank_metric_name = "sharpe_ratio"
    min_baseline_magnitude = _resolve_min_baseline_magnitude(_rank_metric_name)

    if use_skill_composite:
        # Improvement gate is rank-col-based: sortino_ratio (default) or
        # alpha_vs_ew_high_vol when prefer_risk_matched_alpha is on.
        if prefer_risk_matched_alpha:
            best_rank = best_alpha_vs_ew_high_vol
            baseline_rank = baseline_alpha_vs_ew_high_vol
        else:
            best_rank = best_sortino
            baseline_rank = baseline_sortino
        if baseline_rank is None or best_rank is None:
            improvement_pct = 0.0
            improvement_delta = None
            improvement_significant = False
        else:
            improvement_delta = best_rank - baseline_rank
            improvement_pct = improvement_delta / max(
                abs(baseline_rank), _IMPROVEMENT_DENOM_FLOOR
            )
            improvement_significant = (
                abs(baseline_rank) >= min_baseline_magnitude
            )
    else:
        improvement_delta = best_sharpe - baseline_sharpe
        improvement_pct = improvement_delta / max(
            abs(baseline_sharpe), _IMPROVEMENT_DENOM_FLOOR
        )
        improvement_significant = (
            abs(baseline_sharpe) >= min_baseline_magnitude
        )

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
        "rank_metric": (
            "alpha_vs_ew_high_vol" if (use_skill_composite and prefer_risk_matched_alpha)
            else ("sortino_ratio" if use_skill_composite else "sharpe_drawdown")
        ),
        "baseline_params": baseline,
        "recommended_params": recommended,
        "factory_defaults": FACTORY_DEFAULTS.copy(),
        "baseline_sharpe": round(float(baseline_sharpe), 4),
        "best_sharpe": round(float(best_sharpe), 4),
        "best_alpha": best_alpha,
        "baseline_alpha": baseline_alpha,
        "best_sortino": best_sortino,
        "baseline_sortino": baseline_sortino,
        "best_alpha_vs_ew_high_vol": best_alpha_vs_ew_high_vol,
        "baseline_alpha_vs_ew_high_vol": baseline_alpha_vs_ew_high_vol,
        "improvement_pct": round(improvement_pct, 4),
        "improvement_delta": (
            round(float(improvement_delta), 6) if improvement_delta is not None else None
        ),
        "improvement_significant": bool(improvement_significant),
        "min_baseline_magnitude": min_baseline_magnitude,
        "baseline_combo_rank": baseline_combo_rank,
        "baseline_distance": round(float(baseline_dist), 4),
        "n_closer_combos": n_closer_combos,
    }

    if use_skill_composite:
        # Guard: never auto-apply params from a negative-Sortino optimization.
        # Negative Sortino = the strategy's downside-aware return is loss-making —
        # strategy-quality sanity check that fires regardless of the rank column.
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

        # Additional guard for the risk-matched-alpha rank path: don't
        # auto-apply when even the best combo can't beat the vol-matched
        # baseline. The "fishing in volatile waters" framing — if we're not
        # outperforming the dumb-vol-quartile basket, the strategy isn't
        # adding skill on the metric we're optimizing for.
        if prefer_risk_matched_alpha:
            if best_alpha_vs_ew_high_vol is None:
                return {
                    "status": "insufficient_data",
                    **common_fields,
                    "note": (
                        "prefer_risk_matched_alpha is on but best combo has no "
                        "alpha_vs_ew_high_vol — basket likely empty for this "
                        "window (insufficient history). Re-run after corpus "
                        "depth crosses the 60-day vol-lookback threshold."
                    ),
                }
            if best_alpha_vs_ew_high_vol < 0:
                return {
                    "status": "negative_alpha_vs_ew_high_vol",
                    **common_fields,
                    "note": (
                        f"Best alpha_vs_ew_high_vol "
                        f"({best_alpha_vs_ew_high_vol:.4f}) is negative — even "
                        f"the best combo underperforms the vol-matched baseline. "
                        f"Refusing to auto-apply: strategy fails the Workstream "
                        f"D risk-matched skill check."
                    ),
                }

        # Baseline-significance gate — refuse promotion off a near-zero
        # baseline ratio (improvement_pct is structurally noise when
        # |baseline_rank| < min_baseline_magnitude; see the constant's
        # docstring above). Ordered AFTER negative-Sortino /
        # negative-alpha_vs_ew_high_vol so those clearer-failure modes
        # surface their own statuses first, and BEFORE the
        # improvement-pct gate so the latter never fires off a
        # noise-baseline ratio. Constrained-optimization framing per
        # the 2026-05-20 alpha-floor arc (CLAUDE.md SOTA rule).
        if not improvement_significant:
            if prefer_risk_matched_alpha:
                rank_baseline = baseline_alpha_vs_ew_high_vol
                rank_label = "alpha_vs_ew_high_vol"
            else:
                rank_baseline = baseline_sortino
                rank_label = "Sortino"
            _baseline_display = (
                f"{rank_baseline:.4f}" if rank_baseline is not None else "None"
            )
            _delta_display = (
                f"{improvement_delta:.4f}"
                if improvement_delta is not None else "None"
            )
            return {
                "status": "baseline_insignificant",
                **common_fields,
                "note": (
                    f"Baseline {rank_label} ({_baseline_display}) magnitude is "
                    f"below the {min_baseline_magnitude:.3f} significance floor — "
                    f"any improvement ratio off this baseline is structural "
                    f"noise. Refusing to auto-apply (improvement_delta="
                    f"{_delta_display}). Need more sweep history / higher-"
                    f"quality baseline before promotion."
                ),
            }

        min_improvement = _cfg.get("min_sortino_improvement", _MIN_SORTINO_IMPROVEMENT)
        if improvement_pct < min_improvement:
            # Failure note references whichever metric is the rank column so
            # the operator sees the right comparison.
            if prefer_risk_matched_alpha:
                rank_best = best_alpha_vs_ew_high_vol
                rank_baseline = baseline_alpha_vs_ew_high_vol
                rank_label = "alpha_vs_ew_high_vol"
            else:
                rank_best = best_sortino
                rank_baseline = baseline_sortino
                rank_label = "Sortino"
            return {
                "status": "no_improvement",
                **common_fields,
                "note": (
                    f"Best {rank_label} ({rank_best:.4f}) only {improvement_pct:.1%} "
                    f"better than baseline ({rank_baseline:.4f}). Need "
                    f"{min_improvement:.0%}+ to recommend."
                ),
            }

        # Probabilistic Sharpe Ratio gate — confidence-bounded promotion.
        # Workstream D bullet 3 of evaluator-revamp-260506.md: don't promote
        # a combo whose Sharpe is statistically indistinguishable from zero
        # given its sample size + skewness/kurtosis. PSR is computed inline
        # in vectorbt_bridge.portfolio_stats() and flows through sweep_df
        # as a scalar column. None means PSR couldn't be computed (e.g.
        # < 30 daily-return observations) — skip the gate in that case
        # (insufficient data is the baseline-data signal, not a gate).
        # Mirrors the precision_ci_95 gate in the veto skill-composite
        # cutover (alpha-engine-backtester #166).
        best_psr = best_row.get("psr") if "psr" in valid.columns else None
        if best_psr is not None and not pd.isna(best_psr):
            min_psr = _cfg.get("min_psr", _MIN_PSR)
            if float(best_psr) < min_psr:
                return {
                    "status": "insufficient_psr_confidence",
                    **common_fields,
                    "best_psr": _safe_float(best_psr),
                    "note": (
                        f"Best Sortino combo has PSR={float(best_psr):.3f} "
                        f"(P(true Sharpe > 0)) — below {min_psr:.2f} threshold. "
                        f"Refusing to auto-apply: improvement is statistically "
                        f"indistinguishable from zero given the sample size. "
                        f"Need more sweep history before confidence-bounded "
                        f"promotion."
                    ),
                }

        if prefer_risk_matched_alpha:
            rank_best = best_alpha_vs_ew_high_vol
            rank_baseline = baseline_alpha_vs_ew_high_vol
            rank_label = "alpha_vs_ew_high_vol"
            rank_subtext = (
                "skill_composite ranking: alpha_vs_ew_high_vol "
                "(Workstream D risk-matched skill)"
            )
        else:
            rank_best = best_sortino
            rank_baseline = baseline_sortino
            rank_label = "Sortino"
            rank_subtext = "skill_composite ranking: sortino only"
        return {
            "status": "ok",
            **common_fields,
            "n_combos_tested": len(valid),
            "best_psr": _safe_float(best_psr) if best_psr is not None else None,
            "note": (
                f"Best combo improves {rank_label} by {improvement_pct:.1%} "
                f"({rank_baseline:.4f} → {rank_best:.4f}) across {len(valid)} combos "
                f"({rank_subtext}; "
                f"best_alpha={best_alpha} shown for operator display, not gating)."
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

    # Baseline-significance gate — mirror of the skill-composite path.
    # Refuse promotion when baseline_sharpe magnitude is below the
    # significance floor (improvement_pct would be a noise ratio).
    if not improvement_significant:
        return {
            "status": "baseline_insignificant",
            **common_fields,
            "note": (
                f"Baseline Sharpe ({baseline_sharpe:.4f}) magnitude is below "
                f"the {min_baseline_magnitude:.3f} significance floor — any "
                f"improvement ratio off this baseline is structural noise. "
                f"Refusing to auto-apply (improvement_delta={improvement_delta:.4f}). "
                f"Need more sweep history / higher-quality baseline before promotion."
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


def _call_sim_fn(sim_fn, combo_config: dict, dates: list[str] | None):
    """Invoke a sweep ``sim_fn`` over an optional date SUBSET.

    Historically ``sim_fn`` was ``callable(combo_config) -> stats`` and closed
    over the *full* date range, so the holdout path computed ``holdout_dates``
    but never actually restricted the simulation to them — the "holdout" check
    ran over every date (latent bug; the walk-forward work, config#950, depends
    on real date-windowing). This shim lets a date-aware ``sim_fn`` —
    ``callable(combo_config, dates=...)`` — receive the window while remaining
    backward-compatible with the old single-arg form.
    """
    import inspect
    if dates is None:
        return sim_fn(combo_config)
    try:
        params = inspect.signature(sim_fn).parameters
        accepts_dates = "dates" in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except (TypeError, ValueError):
        accepts_dates = False
    if accepts_dates:
        return sim_fn(combo_config, dates=dates)
    # Old-style sim_fn cannot window — fall back to the full-range run. The
    # caller records this so a non-windowed validation is never mistaken for a
    # true out-of-sample check.
    return sim_fn(combo_config)


def _grade_fold(
    holdout_stats: dict, result: dict, fit_target: str,
) -> dict:
    """Grade one out-of-sample window against the train metric.

    Returns a dict with ``metric_name``, ``holdout_value``, ``train_value``,
    ``ratio``, ``passed`` and a human ``note`` — the same pass rule the legacy
    single-split holdout used (metric matches the ranking axis; holdout must be
    positive AND >= 50% of train). Pure: no result mutation, so it composes for
    multi-fold walk-forward."""
    holdout_sharpe = holdout_stats.get("sharpe_ratio")
    holdout_sortino = holdout_stats.get("sortino_ratio")

    if fit_target == "skill_composite":
        metric_name = "Sortino"
        holdout_value = holdout_sortino
        train_value = result.get("best_sortino", 0)
    else:
        metric_name = "Sharpe"
        holdout_value = holdout_sharpe
        train_value = result.get("best_sharpe", 0)

    if holdout_value is None:
        return {
            "metric_name": metric_name, "holdout_value": None,
            "train_value": train_value, "ratio": None, "passed": True,
            "note": f"Holdout produced no {metric_name} — skipped",
        }
    if train_value is None or train_value < 0:
        return {
            "metric_name": metric_name, "holdout_value": round(float(holdout_value), 4),
            "train_value": train_value, "ratio": 0.0, "passed": False,
            "note": (f"Train {metric_name} is non-positive ({train_value}) — "
                     f"cannot validate holdout ratio"),
        }
    ratio = holdout_value / train_value if train_value != 0 else 0.0
    passed = ratio >= 0.50 and holdout_value > 0
    if passed:
        note = f"Holdout {metric_name} ({holdout_value:.4f}) is {ratio:.0%} of train — PASS"
    elif holdout_value <= 0:
        note = (f"Holdout {metric_name} is non-positive ({holdout_value:.4f}) — "
                f"loss-making out-of-sample")
    else:
        note = (f"Holdout {metric_name} ({holdout_value:.4f}) is {ratio:.0%} of train "
                f"({train_value:.4f}) — need >= 50%")
    return {
        "metric_name": metric_name, "holdout_value": round(float(holdout_value), 4),
        "train_value": train_value, "ratio": round(ratio, 4),
        "passed": passed, "note": note,
    }


def _rolling_windows(
    dates: list[str], n_folds: int, test_frac: float,
) -> list[tuple[list[str], list[str]]]:
    """Build ``n_folds`` expanding-train / rolling-test splits (chronological).

    Each fold trains on everything up to a cut point and tests on the next
    ``test_frac`` slice — anchored walk-forward, the standard cross-validation
    for time series (no look-ahead; test windows are disjoint and advance
    forward). Returns ``(train_dates, test_dates)`` per fold; folds whose test
    window is too small to grade are dropped by the caller."""
    n = len(dates)
    test_len = max(1, int(n * test_frac))
    # Place the LAST test window flush against the end, earlier ones stepping
    # back by test_len, so the most-recent data is always validated.
    windows: list[tuple[list[str], list[str]]] = []
    for k in range(n_folds):
        test_end = n - k * test_len
        test_start = test_end - test_len
        if test_start <= 0:
            break
        windows.append((dates[:test_start], dates[test_start:test_end]))
    windows.reverse()  # chronological order (oldest fold first)
    return windows


def validate_walk_forward(
    result: dict,
    sim_fn,
    dates: list[str],
    config: dict,
    *,
    n_folds: int = 3,
    test_frac: float = 0.30,
    min_pass_fraction: float = 1.0,
) -> dict:
    """Rolling walk-forward cross-validation of the recommended params.

    Replaces the single 70/30 holdout (config#950) with ``n_folds`` rolling
    out-of-sample windows, each graded by :func:`_grade_fold`. The recommendation
    PASSES only if at least ``min_pass_fraction`` of gradeable folds pass — default
    1.0 (ALL folds), the conservative choice for a gate that auto-applies params
    to LIVE trading.

    Requires a date-aware ``sim_fn`` (``callable(combo_config, dates=...)``) to
    actually run out-of-sample; with an old single-arg ``sim_fn`` it degrades to
    the legacy single-window behavior and says so (``walk_forward_degraded``).

    Updates ``result`` in place with ``walk_forward`` (per-fold detail),
    ``holdout_passed``, ``holdout_ratio`` (worst-fold ratio), and on failure
    ``status="holdout_failed"``.
    """
    if result.get("status") != "ok":
        return result
    recommended = result.get("recommended_params", {})
    if not recommended:
        return result

    fit_target = result.get("fit_target", "sharpe_legacy")
    holdout_config = {**config, **recommended}

    import inspect
    try:
        date_aware = "dates" in inspect.signature(sim_fn).parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in inspect.signature(sim_fn).parameters.values()
        )
    except (TypeError, ValueError):
        date_aware = False

    windows = _rolling_windows(dates, n_folds, test_frac) if date_aware else []
    # Drop folds whose test window is too short to grade (mirrors the legacy
    # <3-date skip).
    windows = [(tr, te) for tr, te in windows if len(te) >= 3]

    if not windows:
        # Either too little data for multiple folds, or a non-date-aware sim_fn:
        # fall back to the single-split holdout so behavior never regresses.
        result["walk_forward_degraded"] = (
            "no date-aware sim_fn or insufficient dates for rolling folds — "
            "fell back to single-window holdout"
        )
        return validate_holdout(result, sim_fn, dates, config)

    fold_results = []
    for i, (_train_dates, test_dates) in enumerate(windows):
        try:
            stats = _call_sim_fn(sim_fn, holdout_config, test_dates)
        except Exception as e:
            logger.warning("Walk-forward fold %d simulation failed: %s", i, e)
            fold_results.append({
                "fold": i, "passed": True, "ratio": None,
                "note": f"fold {i} simulation error: {e}", "skipped": True,
            })
            continue
        graded = _grade_fold(stats, result, fit_target)
        graded["fold"] = i
        graded["n_test_dates"] = len(test_dates)
        fold_results.append(graded)

    gradeable = [f for f in fold_results if f.get("ratio") is not None and not f.get("skipped")]
    n_pass = sum(1 for f in gradeable if f["passed"])
    n_grade = len(gradeable)

    result["walk_forward"] = {
        "n_folds": len(fold_results),
        "n_gradeable": n_grade,
        "n_passed": n_pass,
        "test_frac": test_frac,
        "min_pass_fraction": min_pass_fraction,
        "folds": fold_results,
    }

    if n_grade == 0:
        result["holdout_passed"] = True
        result["holdout_note"] = "Walk-forward: no gradeable folds — skipped validation"
        return result

    pass_fraction = n_pass / n_grade
    worst_ratio = min((f["ratio"] for f in gradeable if f["ratio"] is not None), default=0.0)
    result["holdout_ratio"] = round(worst_ratio, 4)
    metric_name = gradeable[0]["metric_name"]
    result[f"holdout_{metric_name.lower()}"] = min(
        (f["holdout_value"] for f in gradeable if f["holdout_value"] is not None),
        default=None,
    )

    if pass_fraction + 1e-9 >= min_pass_fraction:
        result["holdout_passed"] = True
        result["holdout_note"] = (
            f"Walk-forward {n_pass}/{n_grade} folds PASS "
            f"(worst-fold ratio {worst_ratio:.0%}) — PASS"
        )
        logger.info("Executor optimizer walk-forward passed: %s", result["holdout_note"])
    else:
        result["holdout_passed"] = False
        result["status"] = "holdout_failed"
        failing = [f"fold {f['fold']}: {f['note']}" for f in gradeable if not f["passed"]]
        result["holdout_note"] = (
            f"Walk-forward {n_pass}/{n_grade} folds pass "
            f"(need {min_pass_fraction:.0%}) — "
            + "; ".join(failing[:3])
        )
        logger.warning("Executor optimizer walk-forward failed: %s", result["holdout_note"])
    return result


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
        sim_fn: callable(combo_config[, dates=...]) -> stats dict. A date-aware
            sim_fn restricts the simulation to the holdout window (true
            out-of-sample); a legacy single-arg sim_fn runs the full range and
            the result is marked ``holdout_degraded``.
        dates: full list of signal dates (chronological).
        config: base config dict.

    Returns:
        result dict with added holdout_sharpe, holdout_passed fields.

    See also :func:`validate_walk_forward` (config#950) for the rolling
    multi-fold cross-validation that supersedes this single split.
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

    import inspect
    try:
        _date_aware = "dates" in inspect.signature(sim_fn).parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in inspect.signature(sim_fn).parameters.values()
        )
    except (TypeError, ValueError):
        _date_aware = False
    if not _date_aware:
        # Legacy sim_fn cannot window — the "holdout" run is the full range.
        # Surface that rather than silently passing it off as out-of-sample.
        result["holdout_degraded"] = (
            "sim_fn is not date-aware — holdout ran over the full date range, "
            "not the held-out window (not a true out-of-sample check)"
        )

    try:
        holdout_stats = _call_sim_fn(
            sim_fn, holdout_config, holdout_dates if _date_aware else None,
        )
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


def produce_artifact(
    result: dict, bucket: str, run_id: str | None = None, run_date: str | None = None,
) -> dict:
    """
    Convert an executor_optimizer ``recommend()`` result into a typed
    ``RecommendationArtifact`` and write it to S3 at
    ``config/executor_params/recommendations/{date}/from_executor_optimizer.json``.

    Part of the optimizer-artifact-assembler arc — see
    ``alpha-engine-docs/private/optimizer-artifact-assembler-260509.md``.
    Always writes regardless of ``result.status`` so the audit trail
    captures every optimizer invocation, including no-improvement /
    insufficient-data / negative-sortino paths. The assembler ignores
    artifacts with ``promotion_intent="skip"`` for merge purposes.

    Args:
        result: dict from ``recommend()``. May or may not have
            ``apply_result`` populated yet (artifact records intent).
        bucket: S3 bucket name.
        run_id: Optional UUID for this artifact. Generated if absent.

    Returns:
        ``{"written": True, "key": str}`` on success;
        ``{"written": False, "reason": str}`` on non-fatal failure (logged
        warn; caller continues with legacy live write).
    """
    from optimizer.recommendation_artifact import (
        RecommendationArtifact, derive_promotion_intent, today_iso, write_artifact,
    )

    try:
        diagnostic = {
            k: result.get(k)
            for k in (
                "status", "best_sharpe", "best_alpha", "best_sortino",
                "baseline_sharpe", "baseline_alpha", "baseline_sortino",
                "improvement_pct", "n_combos_tested", "baseline_combo_rank",
                "baseline_distance", "n_closer_combos",
                "holdout_sharpe", "holdout_sortino", "holdout_ratio",
                "holdout_passed",
            )
            if result.get(k) is not None
        }
        artifact = RecommendationArtifact(
            fit_target=result.get("fit_target", "sharpe_legacy"),
            optimizer_name="executor_optimizer",
            # config#1017: explicit backfill run_date over ambient today_iso()
            # (None on a live run → current trading day).
            run_date=run_date or today_iso(),
            recommendation_kind="full_replace",
            recommended_params=result.get("recommended_params", {}),
            promotion_intent=derive_promotion_intent(result),
            diagnostic=diagnostic,
            notes=result.get("note", "") or "",
        )
        if run_id is not None:
            artifact.run_id = run_id
        key = write_artifact(artifact, bucket, config_type="executor_params")
        return {"written": True, "key": key, "run_id": artifact.run_id}
    except Exception as e:
        # Non-fatal during dual-write window: the legacy live write still
        # happens via apply()'s existing logic. The artifact is additive.
        logger.warning(
            "Failed to write executor_optimizer recommendation artifact: %s "
            "(non-fatal — legacy live write still proceeds)", e,
        )
        return {"written": False, "reason": str(e)}


def apply(result: dict, bucket: str, run_date: str | None = None) -> dict:
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

    Additionally — and unconditionally — produces a per-optimizer
    recommendation artifact at
    ``config/executor_params/recommendations/{date}/from_executor_optimizer.json``
    via ``produce_artifact()``. Part of the optimizer-artifact-assembler
    arc; the artifact is additive (no behavior change to the legacy live
    write) and is consumed by the future assembler module.

    Args:
        result: dict from recommend().
        bucket: S3 bucket name.

    Returns:
        {"applied": True, ...} or {"applied": False, "reason": ...}
    """
    # Produce the per-optimizer recommendation artifact regardless of
    # outcome — captures every invocation for audit. Non-fatal on failure.
    # config#1017: thread the backfill run_date through to the artifact.
    produce_artifact(result, bucket, run_date=run_date)

    # Cutover gate: when assembler.cutover_enabled is true, the assembler
    # is the sole writer of the live key. Skip the legacy live + history +
    # shadow writes. The artifact is already produced above; the assembler
    # reads it during its merge.
    from optimizer.assembler import is_cutover_enabled
    if is_cutover_enabled():
        return {
            "applied": False,
            "reason": "cutover_mode — assembler is sole live writer",
            "fit_target": result.get("fit_target", "sharpe_legacy"),
            "params": result.get("recommended_params", {}),
        }

    if result.get("status") != "ok":
        return {"applied": False, "reason": f"status={result.get('status')}"}

    recommended = result.get("recommended_params", {})
    if not recommended:
        return {"applied": False, "reason": "no recommended params"}

    fit_target = result.get("fit_target", "sharpe_legacy")
    enforce_skill_composite = bool(_cfg.get("enforce_skill_composite", False))

    # config#1053 Phase C: the legacy 1/n-path executor-param sweep no longer
    # auto-writes LIVE config. The daily MVO portfolio optimizer (cutover
    # 2026-05-13) owns sizing + entry, so sweep params tuned on the BYPASSED 1/n
    # path (min_score, max_position_pct, ...) must not silently overwrite the
    # live optimizer-era config. A non-skill_composite (legacy) recommendation is
    # routed to the SHADOW archive by default — the sweep still runs, reports,
    # and archives for observability, but live config is untouched. Re-enabling
    # the legacy live write is an explicit opt-in (`legacy_executor_params_live_apply`,
    # OFF by default). The real fix — sweeping the OPTIMIZER's own params (γ,
    # turnover/ambiguity penalty, weight caps) against the production-faithful
    # backtest — is the Phase-2 follow-up (config#1053 → its child issue).
    legacy_live_apply = bool(_cfg.get("legacy_executor_params_live_apply", False))
    if fit_target == "skill_composite":
        shadow_only = not enforce_skill_composite
        shadow_reason = (
            "shadow mode — fit_target=skill_composite, enforce_skill_composite=False"
        )
    else:
        shadow_only = not legacy_live_apply
        shadow_reason = (
            "shadow mode — legacy executor-param sweep retired from live auto-apply "
            "(config#1053 Phase C; the MVO optimizer owns sizing). Set "
            "legacy_executor_params_live_apply=true to re-enable the live write."
        )

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
        # Canonical eval-style archive layout per alpha_engine_lib.eval_artifacts
        # (v0.8.0). Flat {prefix}/{run_id}.json + latest.json sidecar with
        # YYMMDDHHMM run_id. Replaces the prior {prefix}/{date}.json shape so
        # same-day re-runs preserve forensic capture instead of overwriting.
        run_id = new_eval_run_id()
        shadow_key = eval_artifact_key(S3_SHADOW_PREFIX, run_id)
        shadow_latest_key = eval_latest_key(S3_SHADOW_PREFIX)
        try:
            s3.put_object(
                Bucket=bucket, Key=shadow_key, Body=body, ContentType="application/json"
            )
            s3.put_object(
                Bucket=bucket, Key=shadow_latest_key, Body=body,
                ContentType="application/json",
            )
            logger.info(
                "Executor params written to shadow archive (%s): "
                "s3://%s/%s (+ latest.json sidecar)",
                shadow_reason, bucket, shadow_key,
            )
        except Exception as e:
            logger.error("Failed to write executor params shadow archive: %s", e)
            return {"applied": False, "reason": f"shadow S3 write failed: {e}"}
        return {
            "applied": False,
            "reason": shadow_reason,
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

    # Canonical eval-style archive layout per lib v0.8.0 — see shadow path above
    history_run_id = new_eval_run_id()
    history_prefix = "config/executor_params_history"
    history_key = eval_artifact_key(history_prefix, history_run_id)
    history_latest_key = eval_latest_key(history_prefix)
    try:
        s3.put_object(Bucket=bucket, Key=history_key, Body=body, ContentType="application/json")
        s3.put_object(
            Bucket=bucket, Key=history_latest_key, Body=body,
            ContentType="application/json",
        )
        logger.info(
            "Executor params archived to s3://%s/%s (+ latest.json sidecar)",
            bucket, history_key,
        )
    except Exception as e:
        logger.warning("Failed to archive executor params history (non-fatal): %s", e)

    # Index this apply in the bitemporal knowledge-time changelog so the
    # PIT walk-forward backtest can resolve it (best-effort, never fatal —
    # live + history are already durable). plan §D3.
    from optimizer.config_archive import record_apply
    record_apply(
        bucket, "executor_params",
        history_key=history_key,
        knowledge_date=payload["updated_at"],
        run_id=history_run_id,
        s3_client=s3,
    )

    return {
        "applied": True,
        "fit_target": fit_target,
        "params": recommended,
        "best_sharpe": result.get("best_sharpe"),
        "best_alpha": result.get("best_alpha"),
        "best_sortino": result.get("best_sortino"),
        "improvement_pct": result.get("improvement_pct"),
    }
