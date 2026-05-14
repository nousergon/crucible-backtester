"""factor_blend_sensitivity.py — does configured factor_blend match realized?

Observability layer for the factor blend regime weights configured in
``alpha-engine-config/research/scoring.yaml`` ``aggregator.factor_blend``.
Cross-checks the per-regime stance ordering against realized outcomes
in score_performance.

The question this answers:
  "In BULL regimes, the config says momentum stance gets +0.40 weight
   (highest). Do BULL-regime momentum-stance picks ACTUALLY realize the
   highest risk-adjusted return? If not, the blend is miscalibrated."

This is observability — it does NOT auto-apply changes to S3 / scoring.yaml.
A future PR can promote misalignment flags to a recommendation engine once
we have enough history to trust the signal (Week 8+ per attribution.py's
calibration note).

Composes with:
  - alpha-engine-research migrations v12 (quant/qual/conviction/
    sector_modifier/market_regime) + v16 (stance) on score_performance
  - alpha-engine-research factor blend (Phase 3) emits stance via factor
    composites (Phase 1c); picks are scored under the configured blend
  - The scanner-placement arc's focus_list reads the same blend formula

Plan doc: alpha-engine-docs/private/scanner-260514.md PR 6.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Stance labels emitted by alpha-engine-research's factor_scoring.py +
# scoring/focus_list.py (the 4 within-sector composites + "unknown" for
# tickers with no factor profile coverage).
KNOWN_STANCES: tuple[str, ...] = (
    "momentum", "quality", "value", "low_vol",
)


# Default regime weights mirror the canonical values in
# alpha-engine-config/research/scoring.yaml aggregator.factor_blend.
# Used by the weekly wire-in (evaluate.py) when the backtester config
# doesn't carry an override. Updating scoring.yaml weights should be
# accompanied by a mirror update here — drift would silently break the
# mismatch detector since the analyzer would compare realized outcomes
# against stale config weights. Pinned by test against the values in
# alpha-engine-config (see test_factor_blend_sensitivity_defaults_align).
DEFAULT_REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "bull": {
        "momentum_score": 0.40,
        "quality_score": 0.30,
        "value_score": 0.20,
        "low_vol_score": -0.10,
    },
    "bear": {
        "low_vol_score": 0.40,
        "quality_score": 0.30,
        "momentum_score": -0.20,
        "value_score": 0.10,
    },
    "neutral": {
        "momentum_score": 0.25,
        "quality_score": 0.25,
        "value_score": 0.25,
        "low_vol_score": 0.25,
    },
}

# Minimum samples per (regime, stance) cell before we trust the realized
# stats. Below this the cell is reported but ``trustworthy=False`` — the
# downstream consumer should not act on the mismatch flag for tiny cells.
# Pinned to 20 per attribution.py's calibration window heuristic (noisy
# < 200 total rows, meaningful at Week 8+).
MIN_TRUSTWORTHY_SAMPLES: int = 20


def _sortino(returns: pd.Series, target: float = 0.0) -> float | None:
    """Sortino ratio — mean excess return / downside deviation.

    Returns ``None`` on insufficient data (< 2 obs) or zero downside
    deviation (all returns ≥ target — degenerate, no risk to scale by).
    Not annualized; the caller decides annualization scale.
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return None
    excess = returns - target
    downside = excess[excess < 0]
    if len(downside) == 0:
        return None  # no losses — Sortino undefined
    downside_dev = math.sqrt((downside ** 2).mean())
    if downside_dev == 0:
        return None
    return float(excess.mean() / downside_dev)


def _hit_rate(beats: pd.Series) -> float | None:
    """Fraction of rows with beat_spy=1. None on empty."""
    beats = beats.dropna()
    if len(beats) == 0:
        return None
    return float((beats == 1).mean())


def compute_stance_outcomes(
    df: pd.DataFrame,
    horizon: str = "10d",
) -> pd.DataFrame:
    """Aggregate realized return outcomes by (market_regime, stance).

    Args:
        df: score_performance rows with at least columns ``market_regime``,
            ``stance``, ``return_{horizon}``, ``spy_{horizon}_return``,
            ``beat_spy_{horizon}``. Other columns ignored.
        horizon: "10d" or "30d". Picks the return + beat columns to use.

    Returns:
        DataFrame with one row per (regime, stance) and columns:
            n_picks, mean_alpha, sortino, hit_rate_beat_spy, trustworthy.
        ``alpha`` is ``return_{horizon} - spy_{horizon}_return``.
        Empty input → empty result.
    """
    return_col = f"return_{horizon}"
    spy_col = f"spy_{horizon}_return"
    beat_col = f"beat_spy_{horizon}"
    needed = {"market_regime", "stance", return_col, spy_col, beat_col}
    if df.empty or not needed.issubset(df.columns):
        missing = needed - set(df.columns)
        if missing:
            logger.info(
                "[factor_blend_sensitivity] skipping — missing columns: %s",
                sorted(missing),
            )
        return pd.DataFrame()

    # Filter rows with both stance + regime present + a realized return
    keep = (
        df["market_regime"].notna()
        & df["stance"].notna()
        & df[return_col].notna()
        & df[spy_col].notna()
    )
    work = df.loc[keep].copy()
    if work.empty:
        return pd.DataFrame()

    work["alpha"] = work[return_col] - work[spy_col]

    rows: list[dict] = []
    for (regime, stance), sub in work.groupby(["market_regime", "stance"]):
        rows.append({
            "market_regime": regime,
            "stance": stance,
            "n_picks": len(sub),
            "mean_alpha": float(sub["alpha"].mean()),
            "sortino": _sortino(sub["alpha"]),
            "hit_rate_beat_spy": _hit_rate(sub[beat_col]),
            "trustworthy": len(sub) >= MIN_TRUSTWORTHY_SAMPLES,
        })
    return pd.DataFrame(rows).sort_values(
        ["market_regime", "sortino"], ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)


def detect_mismatches(
    outcomes: pd.DataFrame,
    regime_weights: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Detect regimes where configured stance ranking ≠ realized ranking.

    For each regime, we compute:
      - ``config_order``      — stances ranked by configured weight desc.
                                Signed weights are used as-is (negative
                                weights penalize the stance, so they
                                naturally fall to the bottom).
      - ``realized_order``    — stances ranked by realized Sortino desc.
                                Cells with ``trustworthy=False`` or
                                Sortino=None drop out of the ranking.
      - ``mismatch``          — True iff config's #1 stance ≠ realized
                                #1 stance among trustworthy cells.

    Returns one row per regime present in BOTH the outcomes table and
    the regime_weights config. ``mismatch=NaN`` when realized_order is
    empty (no trustworthy cells yet).
    """
    _empty_columns = [
        "market_regime", "config_top_stance", "realized_top_stance",
        "config_order", "realized_order", "n_trustworthy_cells", "mismatch",
    ]
    if outcomes.empty:
        return pd.DataFrame(columns=_empty_columns)

    rows: list[dict] = []
    for regime, weights in regime_weights.items():
        regime_lower = regime.strip().lower()
        sub = outcomes[outcomes["market_regime"] == regime_lower]
        if sub.empty:
            continue

        # Config order: signed weight desc. Filter to KNOWN_STANCES so
        # spurious keys don't pollute the ranking.
        config_stance_weights = {
            k.replace("_score", ""): float(v)
            for k, v in weights.items()
            if k.replace("_score", "") in KNOWN_STANCES
        }
        config_order = sorted(
            config_stance_weights.keys(),
            key=lambda s: config_stance_weights[s],
            reverse=True,
        )

        # Realized order: Sortino desc among trustworthy cells with non-None
        trustworthy = sub[(sub["trustworthy"]) & (sub["sortino"].notna())]
        realized_order = (
            trustworthy.sort_values("sortino", ascending=False)["stance"]
            .tolist()
        )

        mismatch: Optional[bool]
        if not realized_order or not config_order:
            mismatch = None
        else:
            mismatch = config_order[0] != realized_order[0]

        rows.append({
            "market_regime": regime_lower,
            "config_top_stance": config_order[0] if config_order else None,
            "realized_top_stance": (
                realized_order[0] if realized_order else None
            ),
            "config_order": config_order,
            "realized_order": realized_order,
            "n_trustworthy_cells": len(trustworthy),
            "mismatch": mismatch,
        })
    if not rows:
        return pd.DataFrame(columns=_empty_columns)
    return pd.DataFrame(rows)


def build_sensitivity_report(
    score_performance: pd.DataFrame,
    regime_weights: dict[str, dict[str, float]],
    horizon: str = "10d",
) -> dict:
    """Assemble the full sensitivity report.

    Returns a dict with:
      - ``outcomes``    — per-(regime, stance) realized stats DataFrame
      - ``mismatches``  — per-regime config-vs-realized comparison DataFrame
      - ``horizon``     — return horizon used
      - ``n_total``     — total rows that contributed (had regime + stance
                          + return + spy_return)
      - ``has_data``    — True iff at least one row contributed
    """
    outcomes = compute_stance_outcomes(score_performance, horizon=horizon)
    mismatches = detect_mismatches(outcomes, regime_weights)
    return {
        "outcomes": outcomes,
        "mismatches": mismatches,
        "horizon": horizon,
        "n_total": int(outcomes["n_picks"].sum()) if not outcomes.empty else 0,
        "has_data": not outcomes.empty,
    }
