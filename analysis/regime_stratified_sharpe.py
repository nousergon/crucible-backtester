"""
analysis/regime_stratified_sharpe.py — Stage C.2 T2 downstream-stratified
performance.

Closes Stage C.2 T2 per regime-v3-260514.md §5.3.3. Distinct from
``analysis/regime_analysis.py`` (which stratifies signal *accuracy*
by regime); this module stratifies signal *Sharpe* + pick alpha,
answering the question: did the macro agent's regime call enable
better risk-adjusted returns?

What this measures
------------------
A regime classification with high label accuracy is worthless if
downstream consumers don't act on it usefully. Conversely a regime
classification with moderate label accuracy that enables strong
portfolio performance IS valuable. T2 validates the institutional
*purpose* of regime classification, not its label correctness.

The headline metric is the **regime-stratified Sharpe spread** —
``bull_sharpe - bear_sharpe`` over the alphas of picks the macro
agent made when each regime was active. Positive spread = the
regime call is doing useful work (bull-called picks outperform on
risk-adjusted basis vs bear-called picks). Spread near zero or
negative = regime call is providing no actionable signal.

Why pick-alpha Sharpe, not portfolio Sharpe
-------------------------------------------
``score_performance`` has per-pick alphas (return_N - spy_N_return).
Sharpe over those alphas treats each pick as an independent
observation — cross-sectional Sharpe, not time-series portfolio
Sharpe. Both are valid stratifications. Cross-sectional has cleaner
attribution (each pick is its own datapoint, independent of
position sizing or portfolio construction); portfolio Sharpe would
mix the regime-call quality with the position-sizing + sector-mix
decisions downstream.

Three-tier framework status
---------------------------
T1 (retrospective HMM smoothing) — correctness vs retrospective truth
  Shipped via alpha-engine-predictor #158
T2 (THIS MODULE) — downstream outcome stratified by regime call
T3 (regime_decision_process rubric) — contemporaneous LLM-judge
  scoring of decision-process quality
  Shipped via alpha-engine-config #179

T1 + T2 are correctness measures; T3 is the only contemporaneous
signal. Each alone is gameable; together they triangulate.
"""
from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# Trading days per year — used for Sharpe annualization. Mirrors
# analysis/dsr.py:_TRADING_DAYS_PER_YEAR.
_TRADING_DAYS_PER_YEAR: int = 252


# Default minimum picks per regime stratum before computing Sharpe.
# Below this, the stratum's Sharpe is reported as None — too few
# observations to be statistically meaningful, and the rolling
# headline metric would be noise.
DEFAULT_MIN_PICKS_PER_STRATUM: int = 20


# Horizons we report on. 10d is the primary signal horizon; 30d
# adds a longer-window cross-check. Picks must have NON-NULL
# return + spy_return on that horizon to count.
SUPPORTED_HORIZONS: tuple[int, ...] = (10, 30)


@dataclass(frozen=True)
class StratumMetrics:
    """Per-regime Sharpe + alpha statistics over a horizon."""

    market_regime: str
    horizon_days: int
    n_picks: int
    mean_alpha: float | None
    std_alpha: float | None
    annualized_sharpe: float | None
    hit_rate: float | None  # Fraction of picks where alpha > 0


def load_with_subscores_and_regime(db_path: str) -> pd.DataFrame:
    """Load score_performance with the canonical market_regime column
    plus the return + alpha columns needed for stratified Sharpe.

    Returns a DataFrame with at minimum:
      - market_regime (str | NaN for pre-migration rows)
      - return_10d, spy_10d_return (alpha = return_10d - spy_10d_return)
      - return_30d, spy_30d_return (alpha = return_30d - spy_30d_return)
      - beat_spy_10d, beat_spy_30d (booleans for hit rate)

    Pre-migration rows with NULL market_regime are kept in the
    DataFrame but filtered out at the per-regime grouping stage —
    callers see the row count for observability without needing
    to know about the migration.
    """
    path = Path(db_path).expanduser()
    conn = sqlite3.connect(path)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM score_performance ORDER BY score_date",
            conn,
            parse_dates=["score_date", "eval_date_10d", "eval_date_30d"],
        )
    finally:
        conn.close()

    if "market_regime" not in df.columns:
        # Pre-migration #12 schema. Inject as all-NULL so downstream
        # grouping treats every row as ungrouped (skipped) rather
        # than raising KeyError.
        df["market_regime"] = pd.NA

    n_with_regime = int(df["market_regime"].notna().sum())
    logger.info(
        "Loaded %d score_performance rows for stratified Sharpe (%d with market_regime populated)",
        len(df), n_with_regime,
    )
    return df


def _annualized_sharpe_from_alphas(
    alphas: np.ndarray,
    horizon_days: int,
) -> float | None:
    """Annualize per-pick alpha Sharpe from a sample of N picks each
    observed over ``horizon_days`` of forward returns.

    Per-pick alphas are sampled across stocks at signal time; the
    horizon converts a per-pick standard deviation into an
    annualization scale via the per-pick observation window length.

    ``risk_free=0`` per dsr.py convention. Returns ``None`` on
    insufficient sample or zero variance — signal callers see the
    None and skip the stratum.
    """
    if alphas.size < 2:
        return None
    mean = float(alphas.mean())
    std = float(alphas.std(ddof=1))
    # Near-zero tolerance — IEEE 754 makes ``[0.05, 0.05, 0.05]`` std
    # = ~1e-18 rather than exact 0. Treat both as undefined-Sharpe.
    if not np.isfinite(std) or std < 1e-12:
        return None
    # Annualization: alpha is measured over a horizon_days window;
    # there are _TRADING_DAYS_PER_YEAR / horizon_days such windows
    # per year. Sharpe scales by sqrt(periods_per_year).
    periods_per_year = _TRADING_DAYS_PER_YEAR / horizon_days
    return mean / std * math.sqrt(periods_per_year)


def _stratum_metrics(
    slice_df: pd.DataFrame,
    market_regime: str,
    horizon_days: int,
    min_picks: int,
) -> StratumMetrics:
    """Compute per-stratum metrics. Returns NaN-padded StratumMetrics
    when the stratum is below ``min_picks`` — the caller can filter."""
    return_col = f"return_{horizon_days}d"
    spy_col = f"spy_{horizon_days}d_return"
    beat_col = f"beat_spy_{horizon_days}d"

    if return_col not in slice_df.columns or spy_col not in slice_df.columns:
        return StratumMetrics(
            market_regime=market_regime,
            horizon_days=horizon_days,
            n_picks=0,
            mean_alpha=None,
            std_alpha=None,
            annualized_sharpe=None,
            hit_rate=None,
        )

    populated = slice_df[slice_df[return_col].notna() & slice_df[spy_col].notna()]
    n_picks = len(populated)
    if n_picks < min_picks:
        return StratumMetrics(
            market_regime=market_regime,
            horizon_days=horizon_days,
            n_picks=n_picks,
            mean_alpha=None,
            std_alpha=None,
            annualized_sharpe=None,
            hit_rate=None,
        )

    alphas = (populated[return_col] - populated[spy_col]).to_numpy()
    sharpe = _annualized_sharpe_from_alphas(alphas, horizon_days=horizon_days)
    hit_rate: float | None = None
    if beat_col in populated.columns:
        beat_populated = populated[populated[beat_col].notna()]
        if len(beat_populated) > 0:
            hit_rate = float(beat_populated[beat_col].astype(bool).mean())

    return StratumMetrics(
        market_regime=market_regime,
        horizon_days=horizon_days,
        n_picks=n_picks,
        mean_alpha=float(alphas.mean()),
        std_alpha=float(alphas.std(ddof=1)),
        annualized_sharpe=sharpe,
        hit_rate=hit_rate,
    )


def stratified_sharpe_by_regime(
    df: pd.DataFrame,
    *,
    min_picks_per_stratum: int = DEFAULT_MIN_PICKS_PER_STRATUM,
    horizons: Sequence[int] = SUPPORTED_HORIZONS,
) -> list[StratumMetrics]:
    """Group score_performance by market_regime, compute Sharpe + alpha
    + hit-rate per (regime, horizon) stratum.

    Returns one StratumMetrics per (regime, horizon) combination
    discovered in the data. Strata with fewer than
    ``min_picks_per_stratum`` populated picks have None metrics
    (n_picks still reflects how many were found) — callers see the
    None and either skip or surface "insufficient sample" in their
    UI/report.
    """
    if "market_regime" not in df.columns:
        return []

    df_with_regime = df[df["market_regime"].notna()]
    regimes = sorted(df_with_regime["market_regime"].unique())

    out: list[StratumMetrics] = []
    for regime in regimes:
        regime_slice = df_with_regime[df_with_regime["market_regime"] == regime]
        for horizon in horizons:
            out.append(
                _stratum_metrics(
                    slice_df=regime_slice,
                    market_regime=str(regime),
                    horizon_days=horizon,
                    min_picks=min_picks_per_stratum,
                )
            )
    return out


def compute_regime_spread(
    strata: Sequence[StratumMetrics],
    horizon_days: int = 10,
) -> dict[str, Any]:
    """Compute the headline Sharpe-spread metric for T2.

    Bull-Sharpe minus bear-Sharpe at the given horizon. Positive
    spread means the regime call enabled better risk-adjusted picks
    when bull-regime was declared vs when bear-regime was declared.
    Negative or near-zero spread means the regime call is providing
    no actionable signal at this horizon.

    Also returns the constituent values + an interpretation flag for
    dashboards/reports. None values propagate cleanly when a stratum
    is insufficient — the spread is None too.
    """
    by_regime: dict[str, StratumMetrics] = {
        s.market_regime: s for s in strata if s.horizon_days == horizon_days
    }
    bull = by_regime.get("bull")
    bear = by_regime.get("bear")
    bull_sharpe = bull.annualized_sharpe if bull else None
    bear_sharpe = bear.annualized_sharpe if bear else None

    spread: float | None
    interpretation: str
    if bull_sharpe is None or bear_sharpe is None:
        spread = None
        interpretation = "insufficient_sample"
    else:
        spread = bull_sharpe - bear_sharpe
        if spread > 0.2:
            interpretation = "regime_signal_useful"
        elif spread > -0.2:
            interpretation = "regime_signal_neutral"
        else:
            interpretation = "regime_signal_inverted"

    return {
        "horizon_days": horizon_days,
        "bull_sharpe": bull_sharpe,
        "bear_sharpe": bear_sharpe,
        "neutral_sharpe": (
            by_regime["neutral"].annualized_sharpe
            if by_regime.get("neutral") and by_regime["neutral"].annualized_sharpe is not None
            else None
        ),
        "caution_sharpe": (
            by_regime["caution"].annualized_sharpe
            if by_regime.get("caution") and by_regime["caution"].annualized_sharpe is not None
            else None
        ),
        "spread_bull_minus_bear": spread,
        "interpretation": interpretation,
        "bull_n_picks": bull.n_picks if bull else 0,
        "bear_n_picks": bear.n_picks if bear else 0,
    }


def assemble_t2_eval_payload(
    *,
    strata: Sequence[StratumMetrics],
    spread_10d: Mapping[str, Any],
    spread_30d: Mapping[str, Any],
    run_id: str,
    calendar_date: str,
    trading_day: str,
    min_picks_per_stratum: int = DEFAULT_MIN_PICKS_PER_STRATUM,
) -> dict[str, Any]:
    """Assemble the canonical eval-artifact JSON payload for T2.
    Headed for s3://alpha-engine-research/regime/stratified_sharpe/
    via the lib's alpha_engine_lib.eval_artifacts writers in a
    follow-up pipeline-wiring PR (this module is pure logic + tests).
    """
    strata_serialized = [
        {
            "market_regime": s.market_regime,
            "horizon_days": s.horizon_days,
            "n_picks": s.n_picks,
            "mean_alpha": s.mean_alpha,
            "std_alpha": s.std_alpha,
            "annualized_sharpe": s.annualized_sharpe,
            "hit_rate": s.hit_rate,
        }
        for s in strata
    ]
    return {
        "calendar_date": calendar_date,
        "trading_day": trading_day,
        "run_id": run_id,
        "schema_version": 1,
        "eval_tier": "T2_downstream_stratified_sharpe",
        "min_picks_per_stratum": min_picks_per_stratum,
        "spread_10d": dict(spread_10d),
        "spread_30d": dict(spread_30d),
        "strata": strata_serialized,
        "method_metadata": {
            "annualization_basis": f"{_TRADING_DAYS_PER_YEAR}_trading_days_per_year",
            "alpha_definition": "return_Nd - spy_Nd_return (per-pick cross-sectional alpha)",
            "sharpe_convention": "annualized_sample_std_ddof1_risk_free_zero",
            "interpretation_thresholds": {
                "useful_above": 0.2,
                "neutral_band": "(-0.2, 0.2)",
                "inverted_below": -0.2,
            },
        },
    }
