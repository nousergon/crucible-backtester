"""
quant_rank_quality.py — Per-sector skill of the technical scorer's ranking.

The most direct measure of whether the quant filter is even ordering
correctly: ``corr(quant_rank, return_5d)`` per sector over a rolling
window. A skilled ranker produces a strongly NEGATIVE correlation
(rank #1 = best → highest realized return). A correlation near zero
means the rank is noise; a positive correlation means the rank is
*anti-skilled* — top picks systematically underperform.

Surfaced from the 2026-05-09 evaluator-email post-mortem: per-sector
correlations of +0.331 (healthcare), +0.350 (industrials), +0.360
(technology) showed top quant ranks were systematically picking
losers. Without this diagnostic running weekly, the inversion was
caught only in retrospect via per-stage skill decomposition.

Reads ``team_candidates`` joined to ``universe_returns`` from
research.db. Both tables already exist; no schema change needed.
Computes:

  * Rank correlation: corr(quant_rank, return_5d) per (team_id, eval_date).
  * Score correlation: corr(tech_score, return_5d) — same correlation
    expressed against the continuous score rather than the discrete rank.
    Useful when picks per sector vary week-to-week and rank cardinality
    differs.
  * Hit rate: % of top-3-by-rank picks that beat SPY at 5d horizon.

Aggregates over a rolling N-week window and reports per-team summary
plus an overall metric. Emits CloudWatch metrics dimensioned by team_id
so the same drift-detection alarms used elsewhere in the substrate can
fire on this signal.

Returns the standard backtester-evaluator status dict so the existing
``CompletenessTracker.run_module`` pattern handles it without bespoke
wiring (mirrors ``analysis/macro_eval.py`` shape).
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ── Canonical sectors (mirrors decision_capture_coverage) ───────────────────

CANONICAL_SECTORS = (
    "consumer", "defensives", "financials",
    "healthcare", "industrials", "technology",
)

DEFAULT_LOOKBACK_WEEKS = 8
DEFAULT_CW_NAMESPACE = "AlphaEngine/QuantRankQuality"

# Anti-skill threshold. A rolling correlation above this is flagged in
# the email body and (when the CW alarm wires up) in the substrate's
# health surface. +0.10 is loose — empirical evidence showed +0.33+
# in the broken sectors, with -0.09 in the working sector (financials).
ANTI_SKILL_THRESHOLD = 0.10


# ── Correlation helper ──────────────────────────────────────────────────────


def _safe_pearson(x: list[float], y: list[float]) -> float | None:
    """Pearson correlation. Returns None on n<3 or zero-variance inputs.

    The evaluator surface treats None as "insufficient data" and skips
    the cell rather than substituting 0 or NaN, which would dilute the
    rolling-mean view and conflate "no signal" with "no data."
    """
    if len(x) < 3 or len(y) < 3:
        return None
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den_x = sum((xi - mx) ** 2 for xi in x)
    den_y = sum((yi - my) ** 2 for yi in y)
    if den_x == 0 or den_y == 0:
        return None
    return num / ((den_x * den_y) ** 0.5)


# ── Per-team aggregation ────────────────────────────────────────────────────


def _team_rank_quality(
    conn: sqlite3.Connection,
    *,
    team_id: str,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    """Compute rank-quality metrics for one team over the window.

    Returns dict with rank_corr, score_corr, hit_rate_top3, n_obs,
    n_dates. None on any field whose denominator is below the cutoff.
    """
    rows = conn.execute(
        """
        SELECT tc.quant_rank, tc.quant_score, tc.eval_date,
               ur.return_5d, ur.beat_spy_5d
        FROM team_candidates tc
        INNER JOIN universe_returns ur
          ON tc.ticker = ur.ticker AND tc.eval_date = ur.eval_date
        WHERE tc.team_id = ?
          AND tc.eval_date BETWEEN ? AND ?
          AND ur.return_5d IS NOT NULL
        """,
        (team_id, start_date, end_date),
    ).fetchall()

    if not rows:
        return {
            "team_id": team_id,
            "n_obs": 0, "n_dates": 0,
            "rank_corr": None, "score_corr": None,
            "hit_rate_top3": None, "avg_5d_top3": None,
        }

    ranks = [r[0] for r in rows if r[0] is not None]
    rets_for_rank = [r[3] for r in rows if r[0] is not None]
    scores = [r[1] for r in rows if r[1] is not None]
    rets_for_score = [r[3] for r in rows if r[1] is not None]
    dates = {r[2] for r in rows}

    rank_corr = _safe_pearson(ranks, rets_for_rank)
    score_corr = _safe_pearson(scores, rets_for_score)

    # Top-3 hit rate: of all rows with quant_rank ≤ 3, what fraction
    # beat SPY at 5d? Mirrors how the team uses the ranking in practice
    # (the LLM team picks 2-3 from the top of the ranking).
    top3 = [r for r in rows if r[0] is not None and r[0] <= 3]
    top3_beats = [r[4] for r in top3 if r[4] is not None]
    top3_rets = [r[3] for r in top3 if r[3] is not None]
    hit_rate_top3 = (
        round(sum(top3_beats) / len(top3_beats) * 100, 2)
        if top3_beats else None
    )
    avg_5d_top3 = (
        round(sum(top3_rets) / len(top3_rets) * 100, 4)
        if top3_rets else None
    )

    return {
        "team_id": team_id,
        "n_obs": len(rows),
        "n_dates": len(dates),
        "rank_corr": round(rank_corr, 4) if rank_corr is not None else None,
        "score_corr": round(score_corr, 4) if score_corr is not None else None,
        "hit_rate_top3": hit_rate_top3,
        "avg_5d_top3": avg_5d_top3,
        "n_top3": len(top3),
    }


# ── CloudWatch emission ─────────────────────────────────────────────────────


def _emit_cw_metrics(
    cw: Any,
    *,
    namespace: str,
    run_date: datetime,
    per_team: list[dict[str, Any]],
) -> None:
    """Emit per-team rank-correlation metrics to CloudWatch.

    Dimensioned by ``team_id`` so the substrate's alarm machinery can
    fire on sustained corr > ANTI_SKILL_THRESHOLD (anti-skill drift)
    or corr below a "definitely-working" threshold (skill-loss drift).
    """
    metric_data: list[dict[str, Any]] = []
    for entry in per_team:
        if entry["rank_corr"] is None:
            continue
        dims = [{"Name": "team_id", "Value": entry["team_id"]}]
        metric_data.append({
            "MetricName": "rank_corr_5d",
            "Dimensions": dims,
            "Value": float(entry["rank_corr"]),
            "Unit": "None",
            "Timestamp": run_date,
        })
        if entry["score_corr"] is not None:
            metric_data.append({
                "MetricName": "score_corr_5d",
                "Dimensions": dims,
                "Value": float(entry["score_corr"]),
                "Unit": "None",
                "Timestamp": run_date,
            })
        if entry["hit_rate_top3"] is not None:
            metric_data.append({
                "MetricName": "hit_rate_top3",
                "Dimensions": dims,
                "Value": float(entry["hit_rate_top3"]),
                "Unit": "Percent",
                "Timestamp": run_date,
            })

    for i in range(0, len(metric_data), 20):
        chunk = metric_data[i: i + 20]
        cw.put_metric_data(Namespace=namespace, MetricData=chunk)


# ── Public entry point ──────────────────────────────────────────────────────


def compute_quant_rank_quality(
    db_path: str | None = None,
    db_conn: sqlite3.Connection | None = None,
    run_date: str | None = None,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    cloudwatch_client: Any = None,
    cw_namespace: str = DEFAULT_CW_NAMESPACE,
    emit_metrics: bool = True,
) -> dict[str, Any]:
    """Compute per-sector quant-rank quality over a rolling window.

    Args:
        db_path: path to research.db on disk. Either this or db_conn
            must be provided.
        db_conn: already-open SQLite connection (tests + reusing the
            evaluator's already-pulled DB).
        run_date: ISO date. Defaults to today (UTC). Window end.
        lookback_weeks: trailing N weeks to aggregate over. 8 mirrors
            other rolling-window optimizers in the repo.
        cloudwatch_client: injected boto3 cloudwatch client (tests).
        cw_namespace: CW namespace. Default ``AlphaEngine/QuantRankQuality``.
        emit_metrics: if False, skip CW emission (used in --freeze runs).

    Returns dict:
        status: "ok" | "no_data" | "error"
        run_date, window_start, window_end
        per_team: list[dict] with rank_corr, score_corr, hit_rate_top3,
                  n_obs, n_dates per canonical sector
        overall: pooled corr across all sectors
        anti_skill_teams: list of team_ids with rank_corr above the
                          ANTI_SKILL_THRESHOLD (the canary that should
                          fire CW alarms in production)
    """
    if db_conn is None and db_path is None:
        return {"status": "error", "error": "must provide db_path or db_conn"}

    run_date = run_date or datetime.utcnow().strftime("%Y-%m-%d")
    try:
        end_dt = datetime.strptime(run_date, "%Y-%m-%d")
    except ValueError as e:
        return {"status": "error", "error": f"invalid run_date: {e}"}
    start_dt = end_dt - timedelta(weeks=lookback_weeks)
    start_iso = start_dt.strftime("%Y-%m-%d")
    end_iso = end_dt.strftime("%Y-%m-%d")

    own_conn = False
    conn = db_conn
    if conn is None:
        conn = sqlite3.connect(db_path)
        own_conn = True

    try:
        # Fast-fail when team_candidates table is missing entirely (fresh
        # DB / smoke environment). Status=no_data lets the evaluator
        # surface the gap rather than crash.
        try:
            conn.execute("SELECT 1 FROM team_candidates LIMIT 1")
        except sqlite3.OperationalError:
            return {
                "status": "no_data",
                "run_date": run_date,
                "reason": "team_candidates table missing",
            }

        per_team = [
            _team_rank_quality(
                conn, team_id=t,
                start_date=start_iso, end_date=end_iso,
            )
            for t in CANONICAL_SECTORS
        ]

        # Pooled (cross-sector) correlation — caller can use this as a
        # single headline number. Each sector contributes its own rows.
        all_ranks: list[float] = []
        all_rets: list[float] = []
        all_scores: list[float] = []
        for entry in per_team:
            rows = conn.execute(
                """
                SELECT tc.quant_rank, tc.quant_score, ur.return_5d
                FROM team_candidates tc
                INNER JOIN universe_returns ur
                  ON tc.ticker = ur.ticker AND tc.eval_date = ur.eval_date
                WHERE tc.team_id = ?
                  AND tc.eval_date BETWEEN ? AND ?
                  AND ur.return_5d IS NOT NULL
                """,
                (entry["team_id"], start_iso, end_iso),
            ).fetchall()
            for r in rows:
                if r[0] is not None:
                    all_ranks.append(r[0])
                    all_rets.append(r[2])
                    if r[1] is not None:
                        all_scores.append(r[1])

        # Distinct return list for score_corr (excludes rows where
        # quant_score IS NULL, which is rare but possible during the
        # liquidity-fail edge case in scanner.py).
        score_rets = [
            r[2] for entry in per_team for r in conn.execute(
                """
                SELECT tc.quant_rank, tc.quant_score, ur.return_5d
                FROM team_candidates tc
                INNER JOIN universe_returns ur
                  ON tc.ticker = ur.ticker AND tc.eval_date = ur.eval_date
                WHERE tc.team_id = ?
                  AND tc.eval_date BETWEEN ? AND ?
                  AND ur.return_5d IS NOT NULL
                  AND tc.quant_score IS NOT NULL
                """,
                (entry["team_id"], start_iso, end_iso),
            ).fetchall()
        ]
        overall_rank_corr = _safe_pearson(all_ranks, all_rets)
        overall_score_corr = _safe_pearson(all_scores, score_rets)

        anti_skill_teams = sorted(
            entry["team_id"] for entry in per_team
            if entry["rank_corr"] is not None
            and entry["rank_corr"] > ANTI_SKILL_THRESHOLD
        )

        n_total_obs = sum(e["n_obs"] for e in per_team)
        if n_total_obs == 0:
            return {
                "status": "no_data",
                "run_date": run_date,
                "window_start": start_iso,
                "window_end": end_iso,
                "reason": (
                    f"no team_candidates rows joined to universe_returns "
                    f"in window {start_iso}..{end_iso}"
                ),
            }

        # CW emission
        if emit_metrics:
            try:
                import boto3
                cw = cloudwatch_client or boto3.client("cloudwatch")
                _emit_cw_metrics(
                    cw, namespace=cw_namespace, run_date=end_dt,
                    per_team=per_team,
                )
                logger.info(
                    "quant_rank_quality: emitted metrics for %d teams to "
                    "CloudWatch namespace %s on %s",
                    len(per_team), cw_namespace, run_date,
                )
            except Exception as e:
                logger.warning(
                    "quant_rank_quality: CloudWatch emission failed: %s — "
                    "continuing with JSON artifact only", e,
                )

        return {
            "status": "ok",
            "run_date": run_date,
            "window_start": start_iso,
            "window_end": end_iso,
            "lookback_weeks": lookback_weeks,
            "per_team": per_team,
            "overall_rank_corr": round(overall_rank_corr, 4)
                if overall_rank_corr is not None else None,
            "overall_score_corr": round(overall_score_corr, 4)
                if overall_score_corr is not None else None,
            "anti_skill_teams": anti_skill_teams,
            "anti_skill_threshold": ANTI_SKILL_THRESHOLD,
            "n_total_obs": n_total_obs,
        }
    finally:
        if own_conn:
            conn.close()
