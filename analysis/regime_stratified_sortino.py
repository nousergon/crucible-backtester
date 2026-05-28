"""
analysis/regime_stratified_sortino.py — Stage C.2 T2 downstream-stratified
performance.

Closes Stage C.2 T2 per regime-v3-260514.md §5.3.3. Distinct from
``analysis/regime_analysis.py`` (which stratifies signal *accuracy*
by regime); this module stratifies signal *Sortino* + pick alpha,
answering the question: did the macro agent's regime call enable
better risk-adjusted returns?

Canonical-alpha conventions (per [[feedback_anchor_gates_on_skilled_risk_not_sharpe]])
- Alpha = ``log(1 + return) − log(1 + spy_return)`` (log domain, NOT
  arithmetic). ``score_performance`` exposes arithmetic returns only;
  the log conversion happens in-module at the pick level.
- Headline metric: **Sortino** (downside-only deviation in the
  denominator), NOT raw Sharpe. Sortino is anchored on skilled-risk
  variance — only realizations below the threshold (here zero alpha)
  enter the denominator. Reflects the institutional preference that
  asymmetric downside is what we're insuring against, not symmetric
  volatility.
- Sharpe surfaced as a SECONDARY diagnostic in every stratum so the
  Sortino→Sharpe ratio is auditable during the v3 transition + the
  legacy Sharpe number is still queryable.

What this measures
------------------
A regime classification with high label accuracy is worthless if
downstream consumers don't act on it usefully. Conversely a regime
classification with moderate label accuracy that enables strong
portfolio performance IS valuable. T2 validates the institutional
*purpose* of regime classification, not its label correctness.

The headline metric is the **regime-stratified Sortino spread** —
``bull_sortino - bear_sortino`` over the log-domain pick alphas
when each regime was active. Positive spread = the regime call is
doing useful work (bull-called picks outperform on downside-risk-
adjusted basis vs bear-called picks). Spread near zero = no
actionable signal. Negative = regime model is inverted.

Pick-level (cross-sectional) — not portfolio-level (time-series)
----------------------------------------------------------------
``score_performance`` has per-pick log-alphas. Sortino over those
treats each pick as an independent observation — cross-sectional,
not time-series portfolio Sortino. Both are valid stratifications.
Cross-sectional has cleaner attribution (each pick is its own
datapoint, independent of position sizing or portfolio construction);
portfolio Sortino would mix regime-call quality with the position-
sizing + sector-mix decisions downstream.

PSR + max DD are NOT computed here — they require time-series
path-dependent data. A separate portfolio-level T2 (reading
``eod_pnl.csv`` daily portfolio returns stratified by the regime
active each day) is the natural home for those metrics.

Three-tier framework status
---------------------------
T1 (retrospective HMM smoothing) — correctness vs retrospective truth.
  Shipped via alpha-engine-predictor #158.
T2 (THIS MODULE) — downstream outcome stratified by regime call.
T3 (regime_decision_process rubric) — contemporaneous LLM-judge
  scoring of decision-process quality. Shipped via
  alpha-engine-config #179.

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


# Trading days per year — used for annualization. Mirrors
# analysis/dsr.py:_TRADING_DAYS_PER_YEAR.
_TRADING_DAYS_PER_YEAR: int = 252


# Minimum picks per regime stratum before computing risk-adjusted
# metrics. Below this, the stratum reports n_picks but None metrics —
# too few observations to be statistically meaningful, and the
# rolling headline metric would be noise.
DEFAULT_MIN_PICKS_PER_STRATUM: int = 20


# Horizons reported. 10d is the primary signal horizon; 30d adds a
# longer-window cross-check. Picks must have NON-NULL return +
# spy_return on a horizon to count for that horizon.
SUPPORTED_HORIZONS: tuple[int, ...] = (10, 30)


# Sortino spread interpretation thresholds. Different scale than
# Sharpe — Sortino's denominator is downside-only deviation, so
# |spread| values are typically larger for the same data distribution.
# Calibrated against early synthetic + empirical observation; revisit
# after the first 13-week rolling sample.
_SORTINO_USEFUL_THRESHOLD: float = 0.3
_SORTINO_INVERTED_THRESHOLD: float = -0.3


@dataclass(frozen=True)
class StratumMetrics:
    """Per-regime risk-adjusted statistics over a horizon.

    All metrics computed over **log-domain pick alphas** (canonical
    framework). Sortino is the headline; Sharpe is a secondary
    diagnostic surfaced for cross-reference + legacy continuity.
    """

    market_regime: str
    horizon_days: int
    n_picks: int
    # Log-alpha statistics (per-pick cross-sectional)
    mean_log_alpha: float | None
    std_log_alpha: float | None
    downside_std_log_alpha: float | None
    # Risk-adjusted metrics — annualized
    annualized_sortino: float | None       # HEADLINE
    annualized_sharpe: float | None        # secondary diagnostic
    hit_rate: float | None                 # Fraction of picks where log-alpha > 0


def _arithmetic_to_log_alpha(
    arithmetic_return: pd.Series,
    arithmetic_spy_return: pd.Series,
) -> pd.Series:
    """Convert arithmetic per-pick returns to log-domain pick alpha.

    log_alpha = log(1 + return) − log(1 + spy_return)

    Identity holds exactly when the underlying is a true return
    (price ratio − 1). Approximate at small returns (≈ arithmetic
    alpha minus a second-order correction). Canonical-alpha framework
    requires log domain for variance-bearing computations because
    log returns are additive in time + symmetric in sign around zero
    (a +50% then −33% round-trip is exactly 0 in log domain;
    arithmetic gives 0 too here but the equivalence breaks for
    compounded windows).

    NaN propagation: if either input is NaN, output is NaN. Caller
    filters those out before metric computation.
    """
    # Guard against log(0) — a return of -1.0 means the position went
    # to zero; log domain is undefined. Clip with a tiny epsilon so
    # such picks emit a large negative log return rather than crashing.
    one_plus_ret = np.maximum(1.0 + arithmetic_return, 1e-9)
    one_plus_spy = np.maximum(1.0 + arithmetic_spy_return, 1e-9)
    return np.log(one_plus_ret) - np.log(one_plus_spy)


def load_with_subscores_and_regime(db_path: str) -> pd.DataFrame:
    """Load score_performance with the canonical market_regime column
    plus the arithmetic return columns needed for stratified Sortino.
    Log conversion happens in-module at metric-compute time.

    Returns a DataFrame with at minimum:
      - market_regime (str | NaN for pre-migration rows)
      - return_10d, spy_10d_return (arithmetic; converted to log alpha downstream)
      - return_30d, spy_30d_return (arithmetic; same)
      - beat_spy_10d, beat_spy_30d (booleans for hit rate)

    Pre-migration rows with NULL market_regime are kept in the
    DataFrame but filtered out at the per-regime grouping stage —
    callers see the row count for observability without needing to
    know about the migration.
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
        "Loaded %d score_performance rows for stratified Sortino (%d with market_regime populated)",
        len(df), n_with_regime,
    )
    return df


def _annualization_factor(horizon_days: int) -> float:
    """sqrt(periods_per_year) for cross-sectional pick-alpha annualization.

    Each per-pick alpha is observed over ``horizon_days`` of forward
    return; there are _TRADING_DAYS_PER_YEAR / horizon_days such
    windows per year. Sharpe/Sortino scale by sqrt(that ratio).
    """
    return math.sqrt(_TRADING_DAYS_PER_YEAR / horizon_days)


def _annualized_sortino_from_log_alphas(
    log_alphas: np.ndarray,
    horizon_days: int,
) -> float | None:
    """Annualized Sortino on per-pick log alphas.

    Sortino = mean(log_alpha) / downside_std(log_alpha) × sqrt(periods/year)

    Downside std uses ONLY observations below zero (threshold). The
    canonical-alpha framework anchors on downside-only variance
    because that's the variance we're insuring against — symmetric
    Sharpe penalizes upside volatility equally, which is wrong-headed
    for an alpha strategy.

    Returns ``None`` on insufficient sample, near-zero downside std
    (IEEE-754 tolerance), or no downside observations at all.
    """
    if log_alphas.size < 2:
        return None
    mean = float(log_alphas.mean())
    # Downside-only deviation — RMS of the negative-side observations.
    # Picks with log_alpha > 0 are excluded from the denominator
    # (they're "upside volatility" — not risk).
    downside = log_alphas[log_alphas < 0.0]
    if downside.size == 0:
        # Pure upside sample — Sortino undefined but the regime is
        # clearly favorable. Caller treats None as "insufficient
        # downside sample, skip from headline" rather than infinity.
        return None
    downside_std = float(np.sqrt(np.mean(downside ** 2)))
    if not np.isfinite(downside_std) or downside_std < 1e-12:
        return None
    return mean / downside_std * _annualization_factor(horizon_days)


def _annualized_sharpe_from_log_alphas(
    log_alphas: np.ndarray,
    horizon_days: int,
) -> float | None:
    """Annualized Sharpe on per-pick log alphas — secondary diagnostic.

    Standard Sharpe formula (mean / sample-std × sqrt(periods/year)).
    Surfaced alongside Sortino so cross-reference is possible during
    the v3 transition and the legacy number is still queryable.
    """
    if log_alphas.size < 2:
        return None
    mean = float(log_alphas.mean())
    std = float(log_alphas.std(ddof=1))
    if not np.isfinite(std) or std < 1e-12:
        return None
    return mean / std * _annualization_factor(horizon_days)


def _downside_std(log_alphas: np.ndarray) -> float | None:
    """Downside-only RMS deviation. Surfaced as a separate stratum
    field so the dashboard can render the Sortino denominator
    independently of the ratio."""
    downside = log_alphas[log_alphas < 0.0]
    if downside.size == 0:
        return None
    return float(np.sqrt(np.mean(downside ** 2)))


def _stratum_metrics(
    slice_df: pd.DataFrame,
    market_regime: str,
    horizon_days: int,
    min_picks: int,
) -> StratumMetrics:
    """Compute per-stratum metrics over log-domain pick alphas.

    Returns NaN-padded StratumMetrics when the stratum is below
    ``min_picks`` — the caller filters those out of the headline
    spread metric.
    """
    return_col = f"return_{horizon_days}d"
    spy_col = f"spy_{horizon_days}d_return"
    beat_col = f"beat_spy_{horizon_days}d"

    if return_col not in slice_df.columns or spy_col not in slice_df.columns:
        return StratumMetrics(
            market_regime=market_regime,
            horizon_days=horizon_days,
            n_picks=0,
            mean_log_alpha=None,
            std_log_alpha=None,
            downside_std_log_alpha=None,
            annualized_sortino=None,
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
            mean_log_alpha=None,
            std_log_alpha=None,
            downside_std_log_alpha=None,
            annualized_sortino=None,
            annualized_sharpe=None,
            hit_rate=None,
        )

    # Convert arithmetic → log domain (canonical framework)
    log_alphas = _arithmetic_to_log_alpha(
        populated[return_col], populated[spy_col],
    ).to_numpy()

    sortino = _annualized_sortino_from_log_alphas(log_alphas, horizon_days=horizon_days)
    sharpe = _annualized_sharpe_from_log_alphas(log_alphas, horizon_days=horizon_days)
    hit_rate: float | None = None
    if beat_col in populated.columns:
        beat_populated = populated[populated[beat_col].notna()]
        if len(beat_populated) > 0:
            hit_rate = float(beat_populated[beat_col].astype(bool).mean())

    return StratumMetrics(
        market_regime=market_regime,
        horizon_days=horizon_days,
        n_picks=n_picks,
        mean_log_alpha=float(log_alphas.mean()),
        std_log_alpha=float(np.std(log_alphas, ddof=1)),
        downside_std_log_alpha=_downside_std(log_alphas),
        annualized_sortino=sortino,
        annualized_sharpe=sharpe,
        hit_rate=hit_rate,
    )


def stratified_sortino_by_regime(
    df: pd.DataFrame,
    *,
    min_picks_per_stratum: int = DEFAULT_MIN_PICKS_PER_STRATUM,
    horizons: Sequence[int] = SUPPORTED_HORIZONS,
) -> list[StratumMetrics]:
    """Group score_performance by market_regime, compute Sortino +
    Sharpe + log-alpha + hit-rate per (regime, horizon) stratum.

    Returns one StratumMetrics per (regime, horizon) combination
    discovered in the data. Strata with fewer than
    ``min_picks_per_stratum`` populated picks have None risk-adjusted
    metrics; n_picks still reflects how many were found.
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
    """Compute the headline Sortino-spread metric for T2.

    Bull-Sortino minus bear-Sortino at the given horizon. Positive
    spread = regime call enabled better downside-risk-adjusted picks
    when bull-regime was declared vs when bear-regime was declared.
    Negative or near-zero spread = regime call is providing no
    actionable signal at this horizon.

    Sharpe spread surfaced alongside Sortino as a secondary
    diagnostic. Interpretation flag anchors on Sortino spread.
    """
    by_regime: dict[str, StratumMetrics] = {
        s.market_regime: s for s in strata if s.horizon_days == horizon_days
    }
    bull = by_regime.get("bull")
    bear = by_regime.get("bear")
    bull_sortino = bull.annualized_sortino if bull else None
    bear_sortino = bear.annualized_sortino if bear else None
    bull_sharpe_diag = bull.annualized_sharpe if bull else None
    bear_sharpe_diag = bear.annualized_sharpe if bear else None

    spread: float | None
    sharpe_spread_diagnostic: float | None
    interpretation: str
    if bull_sortino is None or bear_sortino is None:
        spread = None
        interpretation = "insufficient_sample"
    else:
        spread = bull_sortino - bear_sortino
        if spread > _SORTINO_USEFUL_THRESHOLD:
            interpretation = "regime_signal_useful"
        elif spread > _SORTINO_INVERTED_THRESHOLD:
            interpretation = "regime_signal_neutral"
        else:
            interpretation = "regime_signal_inverted"

    if bull_sharpe_diag is None or bear_sharpe_diag is None:
        sharpe_spread_diagnostic = None
    else:
        sharpe_spread_diagnostic = bull_sharpe_diag - bear_sharpe_diag

    return {
        "horizon_days": horizon_days,
        # Headline (Sortino) — per canonical-alpha framework
        "bull_sortino": bull_sortino,
        "bear_sortino": bear_sortino,
        "neutral_sortino": (
            by_regime["neutral"].annualized_sortino
            if by_regime.get("neutral") and by_regime["neutral"].annualized_sortino is not None
            else None
        ),
        # caution_sortino — 3-class Ang-Bekaert taxonomy retired the
        # macro caution tier in v0.42.0 (caution-regime-retirement-260528.md).
        # Field preserved for grandfather attribution on pre-v0.42.0
        # rows; new emissions never populate by_regime["caution"]
        # (returns None — backward-compatible with the existing
        # field's nullable contract).
        "caution_sortino": (
            by_regime["caution"].annualized_sortino
            if by_regime.get("caution") and by_regime["caution"].annualized_sortino is not None
            else None
        ),
        "spread_bull_minus_bear_sortino": spread,
        "interpretation": interpretation,
        "bull_n_picks": bull.n_picks if bull else 0,
        "bear_n_picks": bear.n_picks if bear else 0,
        # Sharpe — secondary diagnostic
        "diagnostic_sharpe_spread_bull_minus_bear": sharpe_spread_diagnostic,
        "diagnostic_bull_sharpe": bull_sharpe_diag,
        "diagnostic_bear_sharpe": bear_sharpe_diag,
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
    Headed for s3://alpha-engine-research/regime/stratified_sortino/
    via the lib's alpha_engine_lib.eval_artifacts writers in a
    follow-up pipeline-wiring PR (this module is pure logic + tests).
    """
    strata_serialized = [
        {
            "market_regime": s.market_regime,
            "horizon_days": s.horizon_days,
            "n_picks": s.n_picks,
            "mean_log_alpha": s.mean_log_alpha,
            "std_log_alpha": s.std_log_alpha,
            "downside_std_log_alpha": s.downside_std_log_alpha,
            "annualized_sortino": s.annualized_sortino,
            "annualized_sharpe_diagnostic": s.annualized_sharpe,
            "hit_rate": s.hit_rate,
        }
        for s in strata
    ]
    return {
        "calendar_date": calendar_date,
        "trading_day": trading_day,
        "run_id": run_id,
        "schema_version": 1,
        "eval_tier": "T2_downstream_stratified_sortino",
        "min_picks_per_stratum": min_picks_per_stratum,
        "spread_10d": dict(spread_10d),
        "spread_30d": dict(spread_30d),
        "strata": strata_serialized,
        "method_metadata": {
            "annualization_basis": f"{_TRADING_DAYS_PER_YEAR}_trading_days_per_year",
            "alpha_definition": (
                "log(1+return_Nd) - log(1+spy_Nd_return) per pick cross-sectional"
            ),
            "headline_metric": "annualized_sortino (downside-only std denominator)",
            "secondary_diagnostic": "annualized_sharpe (full-sample std denominator)",
            "downside_threshold": "0.0 (log-alpha; below this is risk-bearing)",
            "interpretation_thresholds": {
                "useful_above": _SORTINO_USEFUL_THRESHOLD,
                "neutral_band": (
                    f"({_SORTINO_INVERTED_THRESHOLD}, {_SORTINO_USEFUL_THRESHOLD})"
                ),
                "inverted_below": _SORTINO_INVERTED_THRESHOLD,
            },
            "psr_max_dd_note": (
                "PSR + max DD not computed at pick-cross-sectional level "
                "(require time-series path). Portfolio-level T2 reading "
                "eod_pnl.csv is the separate analyzer where they apply."
            ),
        },
    }
