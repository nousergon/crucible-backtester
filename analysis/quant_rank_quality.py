"""
quant_rank_quality.py — Per-sector skill of the technical scorer's ranking.

The most direct measure of whether the quant filter is even ordering
correctly: ``corr(quant_rank, realized return)`` per sector over a rolling
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

Horizons (config#1529): both fleet-policy horizons are measured, resolved
from ``nousergon_lib.quant.horizons.HorizonPolicy`` — the diagnostic short
horizon (5d, the module's original measurement; keys keep their legacy
unsuffixed names for consumer compatibility) AND the canonical PRIMARY
horizon (21d — the horizon the system actually trades/grades on), emitted
under explicitly suffixed keys (``rank_corr_21d`` etc.). Computes per
horizon:

  * Rank correlation: corr(quant_rank, realized return) per (team_id, eval_date).
  * Score correlation: corr(tech_score, realized return) — same correlation
    expressed against the continuous score rather than the discrete rank.
    Useful when picks per sector vary week-to-week and rank cardinality
    differs.
  * Hit rate: % of top-3-by-rank picks that beat SPY at that horizon.

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

from nousergon_lib.quant.horizons import DEFAULT_POLICY

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

# ── Horizons (config#1529) ───────────────────────────────────────────────────
# Both policy horizons are measured against universe_returns' per-horizon
# outcome columns. Column names resolve from the HorizonPolicy chokepoint
# (universe_returns shares the `return_{N}d` / `beat_spy_{N}d` naming with the
# wide score_performance columns for these two fields) — never hardcoded
# literals, per the config#1483 bug-class fix.
_POLICY = DEFAULT_POLICY
_DIAG_H = _POLICY.diagnostic_horizons[0]     # 5 — legacy unsuffixed keys
_PRIMARY_H = _POLICY.primary_horizon         # 21 — explicitly suffixed keys
_RET_DIAG = _POLICY.outcome_columns(_DIAG_H).stock_return
_BEAT_DIAG = _POLICY.outcome_columns(_DIAG_H).beat_spy
_RET_PRIMARY = _POLICY.outcome_columns(_PRIMARY_H).stock_return
_BEAT_PRIMARY = _POLICY.outcome_columns(_PRIMARY_H).beat_spy

# Primary-horizon artifact key suffix — e.g. "rank_corr_21d". Mirrors the CW
# metric naming (rank_corr_5d / score_corr_5d) so alarms and artifact keys
# stay aligned.
_P_SUF = f"{_PRIMARY_H}d"


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


def _universe_returns_has_primary(conn: sqlite3.Connection) -> bool:
    """True iff universe_returns carries the primary-horizon outcome columns.

    Live schema has carried them since the canonical-alpha cutover; an old
    local DB may not. Absence degrades the primary-horizon fields to None
    with a loud WARN — the diagnostic (5d) channel keeps working — rather
    than crashing the whole quant_rank_quality artifact.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(universe_returns)")}
    return _RET_PRIMARY in cols and _BEAT_PRIMARY in cols


# ── Per-horizon aggregation ─────────────────────────────────────────────────


def _horizon_metrics(rows: list[tuple]) -> dict[str, Any]:
    """Metrics for one team × one horizon from (rank, score, date, ret, beat)
    rows whose return is already non-NULL.

    Returns rank_corr, score_corr, hit_rate_top3, avg_top3, n_obs, n_dates,
    n_top3 — None on any field whose denominator is below the cutoff.
    """
    ranks = [r[0] for r in rows if r[0] is not None]
    rets_for_rank = [r[3] for r in rows if r[0] is not None]
    scores = [r[1] for r in rows if r[1] is not None]
    rets_for_score = [r[3] for r in rows if r[1] is not None]
    dates = {r[2] for r in rows}

    # Top-3 hit rate: of all rows with quant_rank ≤ 3, what fraction
    # beat SPY at this horizon? Mirrors how the team uses the ranking in
    # practice (the LLM team picks 2-3 from the top of the ranking).
    top3 = [r for r in rows if r[0] is not None and r[0] <= 3]
    top3_beats = [r[4] for r in top3 if r[4] is not None]
    top3_rets = [r[3] for r in top3 if r[3] is not None]

    rank_corr = _safe_pearson(ranks, rets_for_rank)
    score_corr = _safe_pearson(scores, rets_for_score)
    return {
        "rank_corr": round(rank_corr, 4) if rank_corr is not None else None,
        "score_corr": round(score_corr, 4) if score_corr is not None else None,
        "hit_rate_top3": (
            round(sum(top3_beats) / len(top3_beats) * 100, 2)
            if top3_beats else None
        ),
        "avg_top3": (
            round(sum(top3_rets) / len(top3_rets) * 100, 4)
            if top3_rets else None
        ),
        "n_obs": len(rows),
        "n_dates": len(dates),
        "n_top3": len(top3),
    }


def _team_rank_quality(
    conn: sqlite3.Connection,
    *,
    team_id: str,
    start_date: str,
    end_date: str,
    has_primary: bool = True,
) -> dict[str, Any]:
    """Compute rank-quality metrics for one team over the window, at both
    policy horizons.

    Diagnostic-horizon (5d) metrics keep their legacy unsuffixed key names
    (rank_corr, score_corr, hit_rate_top3, avg_5d_top3, n_obs, n_dates,
    n_top3 — consumer compatibility); primary-horizon (21d) metrics are
    emitted under explicitly suffixed keys (rank_corr_21d, …). None on any
    field whose denominator is below the cutoff.
    """
    primary_select = (
        f", ur.{_RET_PRIMARY}, ur.{_BEAT_PRIMARY}" if has_primary else ""
    )
    primary_where = (
        f" OR ur.{_RET_PRIMARY} IS NOT NULL" if has_primary else ""
    )
    rows = conn.execute(
        f"""
        SELECT tc.quant_rank, tc.quant_score, tc.eval_date,
               ur.{_RET_DIAG}, ur.{_BEAT_DIAG}{primary_select}
        FROM team_candidates tc
        INNER JOIN universe_returns ur
          ON tc.ticker = ur.ticker AND tc.eval_date = ur.eval_date
        WHERE tc.team_id = ?
          AND tc.eval_date BETWEEN ? AND ?
          AND (ur.{_RET_DIAG} IS NOT NULL{primary_where})
        """,
        (team_id, start_date, end_date),
    ).fetchall()

    diag_rows = [r[:5] for r in rows if r[3] is not None]
    primary_rows = (
        [(r[0], r[1], r[2], r[5], r[6]) for r in rows if r[5] is not None]
        if has_primary else []
    )

    diag = _horizon_metrics(diag_rows)
    primary = _horizon_metrics(primary_rows)

    return {
        "team_id": team_id,
        # Diagnostic horizon (5d) — legacy unsuffixed keys.
        "n_obs": diag["n_obs"],
        "n_dates": diag["n_dates"],
        "rank_corr": diag["rank_corr"],
        "score_corr": diag["score_corr"],
        "hit_rate_top3": diag["hit_rate_top3"],
        f"avg_{_DIAG_H}d_top3": diag["avg_top3"],
        "n_top3": diag["n_top3"],
        # Primary (canonical) horizon — explicitly suffixed keys.
        f"n_obs_{_P_SUF}": primary["n_obs"],
        f"rank_corr_{_P_SUF}": primary["rank_corr"],
        f"score_corr_{_P_SUF}": primary["score_corr"],
        f"hit_rate_top3_{_P_SUF}": primary["hit_rate_top3"],
        f"avg_{_P_SUF}_top3": primary["avg_top3"],
        f"n_top3_{_P_SUF}": primary["n_top3"],
    }


def _pooled_corrs(
    conn: sqlite3.Connection,
    *,
    ret_col: str,
    start_date: str,
    end_date: str,
) -> tuple[float | None, float | None]:
    """Pooled (cross-sector) rank + score correlations for one horizon.

    Each canonical sector contributes its own rows. Rank correlation pairs
    quant_rank with the realized return; score correlation pairs quant_score
    with it — each over the rows where its own regressor is non-NULL (the
    pre-config#1529 code paired the score list against a differently-filtered
    return list, silently misaligning the two when quant_rank was NULL on a
    scored row; pairing per-regressor is the correct fix).
    """
    placeholders = ",".join("?" for _ in CANONICAL_SECTORS)
    rows = conn.execute(
        f"""
        SELECT tc.quant_rank, tc.quant_score, ur.{ret_col}
        FROM team_candidates tc
        INNER JOIN universe_returns ur
          ON tc.ticker = ur.ticker AND tc.eval_date = ur.eval_date
        WHERE tc.team_id IN ({placeholders})
          AND tc.eval_date BETWEEN ? AND ?
          AND ur.{ret_col} IS NOT NULL
        """,
        (*CANONICAL_SECTORS, start_date, end_date),
    ).fetchall()

    ranks = [r[0] for r in rows if r[0] is not None]
    rank_rets = [r[2] for r in rows if r[0] is not None]
    scores = [r[1] for r in rows if r[1] is not None]
    score_rets = [r[2] for r in rows if r[1] is not None]
    return _safe_pearson(ranks, rank_rets), _safe_pearson(scores, score_rets)


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
    Metric names carry the horizon suffix (rank_corr_5d / rank_corr_21d);
    the legacy diagnostic hit-rate name (hit_rate_top3, unsuffixed) is
    kept for alarm continuity, with the primary-horizon one suffixed.
    """
    # (per_team key, MetricName, Unit) per horizon.
    metric_map = [
        ("rank_corr", f"rank_corr_{_DIAG_H}d", "None"),
        ("score_corr", f"score_corr_{_DIAG_H}d", "None"),
        ("hit_rate_top3", "hit_rate_top3", "Percent"),
        (f"rank_corr_{_P_SUF}", f"rank_corr_{_P_SUF}", "None"),
        (f"score_corr_{_P_SUF}", f"score_corr_{_P_SUF}", "None"),
        (f"hit_rate_top3_{_P_SUF}", f"hit_rate_top3_{_P_SUF}", "Percent"),
    ]
    metric_data: list[dict[str, Any]] = []
    for entry in per_team:
        if entry["rank_corr"] is None and entry.get(f"rank_corr_{_P_SUF}") is None:
            continue
        dims = [{"Name": "team_id", "Value": entry["team_id"]}]
        for key, name, unit in metric_map:
            value = entry.get(key)
            if value is None:
                continue
            metric_data.append({
                "MetricName": name,
                "Dimensions": dims,
                "Value": float(value),
                "Unit": unit,
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
    """Compute per-sector quant-rank quality over a rolling window, at both
    fleet-policy horizons (diagnostic 5d + canonical primary 21d).

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
        primary_horizon_days / diagnostic_horizon_days: the policy horizons
        per_team: list[dict] with rank_corr, score_corr, hit_rate_top3,
                  n_obs, n_dates per canonical sector (diagnostic 5d,
                  legacy key names) plus the suffixed primary-horizon
                  variants (rank_corr_21d, score_corr_21d, ...)
        overall: pooled corr across all sectors, per horizon
        anti_skill_teams / anti_skill_teams_21d: team_ids with rank_corr
                  above the ANTI_SKILL_THRESHOLD at each horizon (the
                  canary that should fire CW alarms in production)
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

        has_primary = _universe_returns_has_primary(conn)
        if not has_primary:
            logger.warning(
                "quant_rank_quality: universe_returns lacks the primary-horizon "
                "(%dd) outcome columns %s/%s — primary-horizon metrics degrade "
                "to None; diagnostic (%dd) channel unaffected. Pre-canonical-"
                "alpha DB?", _PRIMARY_H, _RET_PRIMARY, _BEAT_PRIMARY, _DIAG_H,
            )

        per_team = [
            _team_rank_quality(
                conn, team_id=t,
                start_date=start_iso, end_date=end_iso,
                has_primary=has_primary,
            )
            for t in CANONICAL_SECTORS
        ]

        # Pooled (cross-sector) correlations per horizon — caller can use
        # these as single headline numbers.
        overall_rank_corr, overall_score_corr = _pooled_corrs(
            conn, ret_col=_RET_DIAG, start_date=start_iso, end_date=end_iso,
        )
        if has_primary:
            overall_rank_corr_p, overall_score_corr_p = _pooled_corrs(
                conn, ret_col=_RET_PRIMARY,
                start_date=start_iso, end_date=end_iso,
            )
        else:
            overall_rank_corr_p = overall_score_corr_p = None

        anti_skill_teams = sorted(
            entry["team_id"] for entry in per_team
            if entry["rank_corr"] is not None
            and entry["rank_corr"] > ANTI_SKILL_THRESHOLD
        )
        anti_skill_teams_primary = sorted(
            entry["team_id"] for entry in per_team
            if entry.get(f"rank_corr_{_P_SUF}") is not None
            and entry[f"rank_corr_{_P_SUF}"] > ANTI_SKILL_THRESHOLD
        )

        n_total_obs = sum(e["n_obs"] for e in per_team)
        n_total_obs_primary = sum(e[f"n_obs_{_P_SUF}"] for e in per_team)
        if n_total_obs == 0 and n_total_obs_primary == 0:
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
            "diagnostic_horizon_days": _DIAG_H,
            "primary_horizon_days": _PRIMARY_H,
            "per_team": per_team,
            "overall_rank_corr": round(overall_rank_corr, 4)
                if overall_rank_corr is not None else None,
            "overall_score_corr": round(overall_score_corr, 4)
                if overall_score_corr is not None else None,
            f"overall_rank_corr_{_P_SUF}": round(overall_rank_corr_p, 4)
                if overall_rank_corr_p is not None else None,
            f"overall_score_corr_{_P_SUF}": round(overall_score_corr_p, 4)
                if overall_score_corr_p is not None else None,
            "anti_skill_teams": anti_skill_teams,
            f"anti_skill_teams_{_P_SUF}": anti_skill_teams_primary,
            "anti_skill_threshold": ANTI_SKILL_THRESHOLD,
            "n_total_obs": n_total_obs,
            f"n_total_obs_{_P_SUF}": n_total_obs_primary,
        }
    finally:
        if own_conn:
            conn.close()
