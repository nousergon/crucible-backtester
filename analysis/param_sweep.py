"""
param_sweep.py — parameter sweep over risk.yaml parameters using Mode 2 simulation.

Supports two modes:
  - grid:   exhaustive search over all parameter combinations (cartesian product)
  - random: randomly sample from the parameter space (Bergstra & Bengio 2012)

Random search with n trials has probability 1 - (1-p)^n of finding a combo in the
top-p fraction. With n=60 trials, there is a 95% chance of finding a top-5% combo
regardless of grid size — making it statistically on par with full grid search for
practical purposes while scaling to arbitrarily large parameter spaces.

Runs executor.main.run(simulate=True) for each parameter combination across all
historical signal dates and compares portfolio outcomes.
"""

from __future__ import annotations

import itertools
import logging
import math
import random
from copy import deepcopy
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)


def _deepcopy_safe_config(base: dict) -> dict:
    """Deepcopy a config dict while excluding keys whose values are not
    deepcopy-safe (boto3 clients, PhaseRegistry, other runtime objects
    with cyclic refs). Underscore-prefixed keys are treated as runtime
    refs by convention and re-attached shallow to the copy.

    Without this, the 2026-04-23 post-filter smoke-param-sweep hit
    `maximum recursion depth exceeded` because `config["_phase_registry"]`
    holds a boto3 S3 client whose internal cyclic refs broke deepcopy.
    """
    serializable: dict[str, Any] = {
        k: v for k, v in base.items() if not k.startswith("_")
    }
    copied = deepcopy(serializable)
    runtime = {k: v for k, v in base.items() if k.startswith("_")}
    copied.update(runtime)
    return copied

# Core 6 parameters — high-frequency, regime-invariant risk/exit rules that
# affect every trade.  60 random trials gives 95% confidence of finding a
# top-5% combination (Bergstra & Bengio).  Grid size: 4×3×4×3×3×4 = 1,728.
#
# Deferred parameters (revisit at 6+ months of live data):
#   reduce_fraction, atr_sizing_target_risk, confidence_sizing_*,
#   staleness_decay_per_day, earnings_*, momentum_gate/exit_threshold,
#   correlation_block_threshold, drawdown_circuit_breaker (safety param,
#   never auto-applied).
DEFAULT_GRID = {
    "min_score": [45, 50, 55, 60, 65, 70, 75, 80],
    "max_position_pct": [0.05, 0.10, 0.15],
    "atr_multiplier": [2.0, 2.5, 3.0, 4.0],
    "time_decay_reduce_days": [5, 7, 10],
    "time_decay_exit_days": [10, 15, 20],
    "profit_take_pct": [0.15, 0.20, 0.25, 0.30],
}

# Extended grid for future use — includes low-frequency params.
# Activate by setting param_sweep in config.yaml to this grid.
EXTENDED_GRID = {
    "min_score": [45, 50, 55, 60, 65, 70, 75, 80],
    "max_position_pct": [0.05, 0.10, 0.15],
    "atr_multiplier": [2.0, 2.5, 3.0, 4.0],
    "time_decay_reduce_days": [5, 7, 10],
    "time_decay_exit_days": [10, 15, 20],
    "profit_take_pct": [0.15, 0.20, 0.25, 0.30],
    "reduce_fraction": [0.25, 0.33, 0.50],
    "atr_sizing_target_risk": [0.01, 0.02, 0.03],
    "confidence_sizing_min": [0.6, 0.7, 0.8],
    "confidence_sizing_range": [0.4, 0.6, 0.8],
    "staleness_decay_per_day": [0.02, 0.03, 0.05],
    "earnings_sizing_reduction": [0.30, 0.50, 0.70],
    "earnings_proximity_days": [3, 5, 7],
    "momentum_gate_threshold": [-10.0, -5.0, -2.0],
    "correlation_block_threshold": [0.70, 0.75, 0.80, 0.85],
    "momentum_exit_threshold": [-20.0, -15.0, -10.0],
    # Stance taxonomy arc PR 4 (2026-05-11) — backtester-tunable gates.
    # Activation gated on ≥4 weeks of stance-tagged history (predictor
    # started emitting stance on 2026-05-11). Until then, sweep results
    # for these params will be insufficient_data per ``MIN_SAMPLES``.
    # Ranges are bracketed around the cold-start defaults (-0.05 and
    # -15.0) with one tighter and one looser candidate so the optimizer
    # can move them in either direction once attribution data exists.
    "value_stance_drawdown_min": [-0.10, -0.05, -0.03],
    "quality_stance_momentum_threshold": [-20.0, -15.0, -10.0],
    # Stance-conditional sizing multipliers (2026-05-11). Brackets
    # around the cold-start defaults so the optimizer can move each
    # in either direction. Momentum kept tight around 1.0 because
    # that's the baseline; the other three span wider so the value/
    # quality/catalyst conviction-discount magnitude is the
    # learn-from-data parameter.
    "stance_size_momentum": [0.9, 1.0, 1.1],
    "stance_size_value": [0.5, 0.7, 0.9],
    "stance_size_quality": [0.6, 0.8, 1.0],
    "stance_size_catalyst": [0.4, 0.6, 0.8],
}

# ── Defaults for sweep mode (override via param_sweep_settings in config.yaml) ──
_DEFAULT_SWEEP_MODE = "random"
_DEFAULT_TOP_FRACTION = 0.05      # target top 5% of parameter space
_DEFAULT_CONFIDENCE = 0.95        # 95% probability of hitting target

# Auto-scaling: sample trial_pct of the grid, clamped to [min_trials, max_trials].
# Floor guarantees statistical coverage; ceiling caps runtime for large grids.
_DEFAULT_TRIAL_PCT = 0.25         # sample 25% of the grid
_DEFAULT_MIN_TRIALS = 50          # floor: statistical minimum
_DEFAULT_MAX_TRIALS = 400         # ceiling: cap runtime


def compute_n_trials(
    top_fraction: float = 0.05,
    confidence: float = 0.95,
) -> int:
    """
    Compute the number of random trials needed to find a combo in the top-p
    fraction with the given confidence level.

    Formula: n = ceil(ln(1 - confidence) / ln(1 - top_fraction))

    Examples:
        top 5%  at 95% confidence → 59 trials
        top 5%  at 99% confidence → 90 trials
        top 1%  at 95% confidence → 299 trials
    """
    if top_fraction <= 0 or top_fraction >= 1:
        raise ValueError(f"top_fraction must be in (0, 1), got {top_fraction}")
    if confidence <= 0 or confidence >= 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    return math.ceil(math.log(1 - confidence) / math.log(1 - top_fraction))


def auto_n_trials(
    total_grid: int,
    trial_pct: float | None = None,
    min_trials: int | None = None,
    max_trials: int | None = None,
) -> int:
    """
    Compute the number of random trials scaled to grid size.

    Uses trial_pct of total_grid, clamped to [min_trials, max_trials].
    Guarantees the statistical floor (60 = 95% top-5%) for small grids,
    and caps runtime for large grids.

    Examples:
        grid=216,  30% → 65  (floor wins: 60 → 65 after rounding)
        grid=972,  30% → 292
        grid=5000, 30% → 500 (ceiling wins)
    """
    pct = trial_pct if trial_pct is not None else _DEFAULT_TRIAL_PCT
    floor = min_trials if min_trials is not None else _DEFAULT_MIN_TRIALS
    ceiling = max_trials if max_trials is not None else _DEFAULT_MAX_TRIALS

    scaled = math.ceil(total_grid * pct)
    n = max(floor, min(scaled, ceiling))

    # Never exceed the grid itself
    return min(n, total_grid)


def _generate_random_combos(
    grid: dict,
    n_trials: int,
    seed: int | None = None,
) -> list[dict]:
    """
    Sample n_trials unique parameter combinations from the grid.

    If n_trials >= total grid size, falls back to exhaustive grid (no benefit
    to random sampling when you can cover everything).
    """
    keys = list(grid.keys())
    values = list(grid.values())
    total_combos = 1
    for v in values:
        total_combos *= len(v)

    if n_trials >= total_combos:
        logger.info(
            "max_trials (%d) >= grid size (%d) — using exhaustive grid search",
            n_trials, total_combos,
        )
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    rng = random.Random(seed)
    seen: set[tuple] = set()
    combos: list[dict] = []

    while len(combos) < n_trials:
        sample = tuple(rng.choice(v) for v in values)
        if sample not in seen:
            seen.add(sample)
            combos.append(dict(zip(keys, sample)))

    return combos


def _run_combos(
    combinations: list[dict],
    run_simulation_fn: Callable[[dict], dict],
    base_config: dict,
) -> pd.DataFrame:
    """Run simulation for each parameter combination and return results DataFrame."""
    import time as _time
    rows = []
    n = len(combinations)
    t_sweep_start = _time.monotonic()
    for i, params in enumerate(combinations, 1):
        config = _deepcopy_safe_config(base_config)
        config.update(params)

        # Per-combo progress at INFO so the sweep never goes silent. Each
        # combo is a full simulation (~30-90s); without this, 60 combos
        # run in complete silence at default INFO level and look like a
        # 60-min hang to any log reader. See ROADMAP Backtester P0
        # "Diagnose the silent-phase bottleneck" (2026-04-22).
        t_combo = _time.monotonic()
        logger.info("Sweep combo %d/%d: %s", i, n, params)
        try:
            stats = run_simulation_fn(config)
            rows.append({**params, **stats})
        except Exception as e:
            logger.warning("Simulation failed for params %s: %s", params, e)
            rows.append({**params, "error": str(e)})
        logger.info(
            "Sweep combo %d/%d done in %.1fs (sweep elapsed %.1fs)",
            i, n, _time.monotonic() - t_combo, _time.monotonic() - t_sweep_start,
        )

    df = pd.DataFrame(rows)
    # Sort by total_alpha (primary) then sharpe_ratio (tiebreaker)
    if "total_alpha" in df.columns:
        df.sort_values("total_alpha", ascending=False, inplace=True)
    elif "sharpe_ratio" in df.columns:
        df.sort_values("sharpe_ratio", ascending=False, inplace=True)

    return df


def sweep(
    grid: dict,
    run_simulation_fn: Callable[[dict], dict],
    base_config: dict,
    sweep_settings: dict | None = None,
) -> pd.DataFrame:
    """
    Parameter sweep over combinations from the grid.

    Args:
        grid: Dict mapping param name → list of values to try.
        run_simulation_fn: Callable that accepts a config dict and returns a
              stats dict (total_return, sharpe_ratio, max_drawdown, ...).
        base_config: Base config dict; each combination overrides relevant keys.
        sweep_settings: Dict from config.yaml param_sweep_settings section.
            Keys: mode ("grid"|"random"), max_trials (int, optional),
            trial_pct (float), min_trials (int), max_trials_cap (int),
            seed (int, optional).

    Returns:
        DataFrame with one row per parameter combination, sorted by
        ``total_alpha`` (primary) with ``sharpe_ratio`` as tiebreaker
        (per the sort applied by ``_run_combos``). Sweep metadata stored
        in df.attrs for reporting.
    """
    settings = sweep_settings or {}
    mode = settings.get("mode", _DEFAULT_SWEEP_MODE)
    seed = settings.get("seed")

    keys = list(grid.keys())
    values = list(grid.values())
    total_grid = 1
    for v in values:
        total_grid *= len(v)

    if mode == "random":
        # If max_trials is explicitly set, use it; otherwise auto-scale
        explicit_max = settings.get("max_trials")
        if explicit_max is not None:
            n = int(explicit_max)
        else:
            n = auto_n_trials(
                total_grid,
                trial_pct=settings.get("trial_pct"),
                min_trials=settings.get("min_trials"),
                max_trials=settings.get("max_trials_cap"),
            )
        combinations = _generate_random_combos(grid, n, seed=seed)
        actual_mode = "random" if len(combinations) < total_grid else "grid (auto-fallback)"
        coverage = len(combinations) / total_grid
        top_frac = _DEFAULT_TOP_FRACTION
        prob = 1 - (1 - top_frac) ** len(combinations)

        logger.info(
            "Random sweep: %d/%d combos (%.0f%% coverage). "
            "%.1f%% probability of finding top-%.0f%% combo across %s",
            len(combinations), total_grid, coverage * 100,
            prob * 100, top_frac * 100, keys,
        )
    else:
        combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        actual_mode = "grid"
        coverage = 1.0
        logger.info(
            "Grid sweep: %d combinations across %s",
            len(combinations), keys,
        )

    df = _run_combos(combinations, run_simulation_fn, base_config)

    # Gate: require at least 50% of combos to succeed
    n_total = len(combinations)
    if not df.empty and "error" in df.columns:
        n_failed = df["error"].notna().sum()
        n_valid = n_total - n_failed
        completion_pct = n_valid / n_total if n_total > 0 else 0
        if completion_pct < 0.50:
            logger.warning(
                "Param sweep: only %d/%d combos succeeded (%.0f%%) — "
                "below 50%% threshold, results may be unreliable",
                n_valid, n_total, completion_pct * 100,
            )
            df.attrs["sweep_low_completion"] = True
            df.attrs["sweep_completion_pct"] = round(completion_pct, 2)

    # Add metadata for reporting
    if not df.empty:
        df.attrs["sweep_mode"] = actual_mode
        df.attrs["sweep_total_grid"] = total_grid
        df.attrs["sweep_trials"] = len(combinations)
        df.attrs["sweep_coverage"] = coverage

    return df


def best_params(sweep_df: pd.DataFrame, metric: str = "sharpe_ratio") -> dict:
    """
    Return the parameter combination with the best value of `metric`.
    """
    if metric not in sweep_df.columns:
        raise ValueError(f"Metric '{metric}' not found in sweep results")

    best_row = sweep_df.dropna(subset=[metric]).iloc[0]
    stat_cols = {
        "total_return", "sharpe_ratio", "max_drawdown", "calmar_ratio",
        "total_trades", "win_rate", "error", "status", "dates_simulated",
        "total_orders", "note",
    }
    param_cols = [c for c in sweep_df.columns if c not in stat_cols]
    return {col: best_row[col] for col in param_cols}
