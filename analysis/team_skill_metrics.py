"""
team_skill_metrics.py — orchestrator for the per-team + portfolio
skilled-risk-taking metric stack.

Activates the graders added in PR 3 (`_grade_sector_team` skill
composite, `_grade_calibration_diagnostics`, `_grade_action_entropy`,
`_grade_excursion`) by computing the input dicts they expect from data
already available in evaluate.py.

Graceful degrade: every per-team sub-metric is computed independently
and any sub-metric that's missing data returns ``status="insufficient_data"``.
The grading layer then drops that contribution from the weighted average
without breaking the team's overall grade.

Inputs:
  - score_performance_df: canonical per-pick outcome table from research.db
    (loaded by analysis.signal_quality.load_score_performance)
  - team_lift: list[dict] from analysis.end_to_end._team_lift, where each
    team has a ``picks`` field of (ticker, eval_date, short-horizon return,
    conviction)
  - prices (optional): pd.DataFrame for risk-matched-benchmark + excursion
  - spy_daily_returns (optional): pd.Series for beta-matched-SPY benchmark
  - ohlc (optional): dict[ticker, ohlc_df] for MFE/MAE excursion

Outputs:
  - team_metrics: dict keyed by team_id with ic / expectancy / excursion
    / alpha_vs_ew_high_vol / alpha_vs_beta_spy sub-results
  - calibration_diagnostics: result of compute_calibration on the
    portfolio-wide (score → primary beat-SPY) corpus
  - action_entropy / excursion_summary at portfolio level: optional,
    computed when the requisite inputs are provided

Pure-compute. No I/O. The evaluate.py caller is responsible for loading
score_performance + (optionally) prices/SPY/OHLC and passing them in.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from nousergon_lib.quant.horizons import DEFAULT_POLICY

from analysis.calibration_diagnostics import compute_calibration
from analysis.expectancy import compute_expectancy
from analysis.excursion import (
    ExcursionRecord,
    compute_per_pick_excursion,
    summarize_excursions,
)
from analysis.information_coefficient import compute_ic
from analysis.risk_matched_benchmark import (
    compute_alpha_vs_benchmark,
    construct_beta_matched_spy_benchmark,
    construct_ew_high_vol_benchmark,
)
from analysis.team_daily_returns import compute_team_daily_returns

logger = logging.getLogger(__name__)

# Outcome column / pick-field names resolve from the fleet HorizonPolicy rather
# than hardcoded `_Nd` literals (config#1483/#1529). Names are unchanged; the
# score_performance_df is re-sourced from score_performance_outcomes upstream
# (signal_quality.load_score_performance). The pick records carry the same
# short-horizon stock-return field name by construction.
_POLICY = DEFAULT_POLICY
_RET_PRIMARY = _POLICY.outcome_columns(_POLICY.primary_horizon).stock_return  # return_21d
_RET_SHORT = _POLICY.outcome_columns(_POLICY.diagnostic_horizons[0]).stock_return  # return_5d
_BEAT_PRIMARY = _POLICY.outcome_columns(_POLICY.primary_horizon).beat_spy  # beat_spy_21d


def compute_team_metrics(
    team_lift: list[dict],
    score_performance_df: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    spy_daily_returns: pd.Series | None = None,
    ohlc: dict[str, pd.DataFrame] | None = None,
    horizon_days: int = 10,
) -> dict[str, dict]:
    """Per-team skilled-risk-taking metric bundle.

    Returns a dict ``{team_id: {ic, expectancy, excursion,
    alpha_vs_ew_high_vol, alpha_vs_beta_spy}}`` that
    ``analysis.grading._grade_sector_team`` consumes via the
    ``team_metrics`` kwarg in ``compute_scorecard``.

    Each sub-metric is computed independently. Sub-metrics requiring data
    that's not provided fall back to ``{"status": "insufficient_data",
    "reason": "..."}`` so the team's overall grade still composes from
    the available pieces.
    """
    if not team_lift:
        return {}

    out: dict[str, dict] = {}
    for team in team_lift:
        team_id = team.get("team_id")
        if not team_id:
            continue
        picks = team.get("picks") or []
        team_bundle: dict[str, Any] = {}

        # ── IC + expectancy: need the per-pick (score → return_5d) join.
        # picks records carry return_5d directly (added in PR 1).
        if picks and score_performance_df is not None and not score_performance_df.empty:
            team_bundle["ic"] = _compute_team_ic(team_id, picks, score_performance_df)
            team_bundle["expectancy"] = _compute_team_expectancy(picks)
        else:
            team_bundle["ic"] = {"status": "insufficient_data",
                                 "reason": "no picks or no score_performance"}
            team_bundle["expectancy"] = {"status": "insufficient_data",
                                          "reason": "no picks"}

        # ── Excursion: needs OHLC per ticker over the holding window.
        if picks and ohlc:
            picks_df = pd.DataFrame(picks)
            try:
                records: list[ExcursionRecord] = compute_per_pick_excursion(
                    picks_df, ohlc, horizon_days=horizon_days,
                )
                team_bundle["excursion"] = summarize_excursions(records)
            except Exception as e:
                logger.warning("team %s excursion failed: %s", team_id, e)
                team_bundle["excursion"] = {"status": "insufficient_data",
                                            "reason": str(e)}
        else:
            team_bundle["excursion"] = {"status": "insufficient_data",
                                         "reason": "no ohlc data"}

        # ── Risk-matched alpha vs both benchmarks: needs prices + SPY.
        if picks and prices is not None and not prices.empty:
            try:
                picks_df = pd.DataFrame(picks)
                team_returns = compute_team_daily_returns(
                    picks_df, prices, horizon_days=horizon_days,
                )
                series = team_returns.get(team_id)
                if series is None or series.empty:
                    raise ValueError("no daily returns for team")

                ew_bench = construct_ew_high_vol_benchmark(
                    prices, vol_quantile=0.75, vol_lookback_days=60,
                )
                team_bundle["alpha_vs_ew_high_vol"] = compute_alpha_vs_benchmark(
                    series, ew_bench, label="ew_high_vol",
                )

                if spy_daily_returns is not None and not spy_daily_returns.empty:
                    bm_bench = construct_beta_matched_spy_benchmark(
                        series, spy_daily_returns, beta_lookback_days=60,
                    )
                    team_bundle["alpha_vs_beta_spy"] = compute_alpha_vs_benchmark(
                        series, bm_bench, label="beta_matched_spy",
                    )
                else:
                    team_bundle["alpha_vs_beta_spy"] = {
                        "status": "insufficient_data",
                        "reason": "no spy daily returns",
                    }
            except Exception as e:
                logger.warning("team %s risk-matched alpha failed: %s", team_id, e)
                team_bundle["alpha_vs_ew_high_vol"] = {"status": "insufficient_data",
                                                       "reason": str(e)}
                team_bundle["alpha_vs_beta_spy"] = {"status": "insufficient_data",
                                                    "reason": str(e)}
        else:
            team_bundle["alpha_vs_ew_high_vol"] = {"status": "insufficient_data",
                                                   "reason": "no prices"}
            team_bundle["alpha_vs_beta_spy"] = {"status": "insufficient_data",
                                                "reason": "no prices"}

        out[team_id] = team_bundle

    return out


def _compute_team_ic(
    team_id: str, picks: list[dict], score_perf: pd.DataFrame,
) -> dict:
    """IC over the team's picks: score → forward return."""
    pick_keys = {(p["ticker"], str(p["eval_date"])) for p in picks}
    df = score_perf[
        score_perf.apply(
            lambda r: (r["symbol"], str(r["score_date"])) in pick_keys
            if "symbol" in r and "score_date" in r else False,
            axis=1,
        )
    ]
    if df.empty:
        return {"status": "insufficient_data", "reason": "no score_performance match"}
    # config#1456: prefer canonical primary-horizon return, fall back to short.
    fwd_col = (
        _RET_PRIMARY
        if _RET_PRIMARY in df.columns and df[_RET_PRIMARY].notna().any()
        else _RET_SHORT
    )
    return compute_ic(
        conviction=df["score"].to_numpy(),
        forward_return=df[fwd_col].to_numpy(),
        min_samples=10,  # team-level samples are smaller than universe
    )


def _compute_team_expectancy(picks: list[dict]) -> dict:
    """Expectancy over the team's picks' short-horizon return (or alpha)."""
    returns = [p[_RET_SHORT] for p in picks if p.get(_RET_SHORT) is not None]
    if len(returns) < 5:
        return {"status": "insufficient_data", "reason": f"only {len(returns)} picks"}
    return compute_expectancy(np.array(returns), threshold=0.0, min_samples=5)


def compute_portfolio_calibration(
    score_performance_df: pd.DataFrame | None,
    score_col: str = "score",
    outcome_col: str = _BEAT_PRIMARY,
) -> dict:
    """Reliability diagram on portfolio-wide (score → outcome) corpus.

    Normalizes ``score`` (0-100) to [0, 1] before computing, since
    ``compute_calibration`` expects probabilities. The score column is a
    composite-score proxy for "agent's probability this pick beats SPY";
    its calibration as a probability is the question we're asking.
    """
    if score_performance_df is None or score_performance_df.empty:
        return {"status": "insufficient_data", "reason": "no score_performance"}
    if score_col not in score_performance_df.columns:
        return {"status": "insufficient_data", "reason": f"no {score_col} column"}
    if outcome_col not in score_performance_df.columns:
        return {"status": "insufficient_data", "reason": f"no {outcome_col} column"}

    df = score_performance_df.dropna(subset=[score_col, outcome_col])
    if len(df) < 30:
        return {"status": "insufficient_data", "n": len(df)}

    return compute_calibration(
        predicted_probability=(df[score_col] / 100.0).to_numpy(),
        realized_outcome=df[outcome_col].astype(float).to_numpy(),
        min_total_samples=30,
    )


def compute_portfolio_excursion_summary(
    score_performance_df: pd.DataFrame | None,
    ohlc: dict[str, pd.DataFrame] | None,
    horizon_days: int = 10,
    score_threshold: int = 60,
) -> dict:
    """Aggregate MFE/MAE across all BUY-grade picks portfolio-wide.

    Only picks with ``score >= score_threshold`` (i.e. ENTER candidates)
    are included — this is what the executor would have considered, so
    the excursion stats reflect actionable positions.
    """
    if score_performance_df is None or score_performance_df.empty:
        return {"status": "insufficient_data", "reason": "no score_performance"}
    if ohlc is None or not ohlc:
        return {"status": "insufficient_data", "reason": "no ohlc data"}

    df = score_performance_df[score_performance_df["score"] >= score_threshold]
    if df.empty:
        return {"status": "insufficient_data",
                "reason": f"no picks with score >= {score_threshold}"}

    picks_df = df[["symbol", "score_date"]].rename(
        columns={"symbol": "ticker", "score_date": "eval_date"},
    )
    try:
        records = compute_per_pick_excursion(picks_df, ohlc, horizon_days=horizon_days)
        return summarize_excursions(records)
    except Exception as e:
        logger.warning("portfolio excursion failed: %s", e)
        return {"status": "insufficient_data", "reason": str(e)}
