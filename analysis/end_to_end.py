"""
end_to_end.py — Full-pipeline attribution across all six decision boundaries.

Joins universe_returns with scanner_evaluations, team_candidates, cio_evaluations,
predictor_outcomes, executor_shadow_book, and trades to compute lift at each stage
of the pipeline. Answers: "Did this step improve on what it was given?"

Decision boundaries (upstream → downstream):
  1. Scanner filter:  900 → 50-70
  2. Sector teams:    50-70 → 12-18
  3. CIO promotion:   12-18 → ~5-8
  4. Predictor veto:  ~25 → ~20
  5. Executor trading: ~20 → ~15

All tables join on (ticker, eval_date). Every downstream table is a strict subset
of the one above it.

Writer: backtester (weekly, after universe_returns is populated).
Output: lift summary dict + optional e2e_attribution.csv on S3.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from analysis.classification_metrics import compute_binary_metrics

logger = logging.getLogger(__name__)


def compute_lift_metrics(
    research_db_path: str,
    trades_db_path: str | None = None,
    eval_date: str | None = None,
) -> dict:
    """
    Compute lift at each decision boundary for the given eval_date(s).

    Args:
        research_db_path: path to research.db (universe_returns, scanner_evaluations,
                          team_candidates, cio_evaluations, predictor_outcomes)
        trades_db_path: path to trades.db (trades, executor_shadow_book).
                        Optional — executor lift is skipped if not available.
        eval_date: optional filter. If None, computes across all available dates.

    Returns dict with:
        status: "ok" | "insufficient_data" | "error"
        n_dates: number of eval dates analyzed
        scanner_lift: {passing_avg, universe_avg, lift, n_passing, n_universe}
        team_lift: [{team_id, pick_avg, sector_avg, quant_avg, lift, lift_vs_quant, ...}]
        cio_lift: {advance_avg, all_recs_avg, lift, n_advance, n_recs}
        cio_vs_ranking: {cio_avg, ranking_avg, lift, cio_beats_ranking, ...}
        predictor_lift: {up_avg, all_avg, down_avg, lift, n_up, n_down, n_all}
        executor_lift: {traded_avg, approved_avg, lift, n_traded, n_approved}
        pipeline_lift: {traded_avg, universe_avg, lift}
    """
    if not Path(research_db_path).exists():
        return {"status": "error", "error": f"research.db not found at {research_db_path}"}

    conn = sqlite3.connect(research_db_path)

    # Check if universe_returns has data
    try:
        ur_count = conn.execute("SELECT COUNT(*) FROM universe_returns").fetchone()[0]
    except sqlite3.OperationalError:
        conn.close()
        return {"status": "error", "error": "universe_returns table does not exist"}

    if ur_count == 0:
        conn.close()
        return {"status": "insufficient_data", "error": "universe_returns is empty"}

    date_filter = ""
    params: list = []
    if eval_date:
        date_filter = " WHERE eval_date = ?"
        params = [eval_date]

    try:
        # Load universe_returns as base
        ur = pd.read_sql_query(
            f"SELECT * FROM universe_returns{date_filter} ORDER BY eval_date, ticker",
            conn, params=params,
        )

        n_dates = ur["eval_date"].nunique()
        if n_dates == 0:
            conn.close()
            return {"status": "insufficient_data", "error": "no data for specified date"}

        # Only use rows with 5d returns populated
        ur = ur[ur["return_5d"].notna()]
        if ur.empty:
            conn.close()
            return {"status": "insufficient_data", "error": "no rows with return_5d populated"}

        result: dict = {
            "status": "ok",
            "n_dates": n_dates,
            "n_universe_rows": len(ur),
        }

        # 1. Scanner lift
        result["scanner_lift"] = _scanner_lift(conn, ur, date_filter, params)

        # 2. Team lift
        result["team_lift"] = _team_lift(conn, ur, date_filter, params)

        # 3. CIO lift
        result["cio_lift"] = _cio_lift(conn, ur, date_filter, params)

        # 3b. CIO vs score-ranking baseline (2e)
        result["cio_vs_ranking"] = _cio_vs_ranking_lift(conn, ur, date_filter, params)

        # 4. Predictor lift
        result["predictor_lift"] = _predictor_lift(conn, ur, date_filter, params)

        # 5. Executor lift (requires trades.db)
        if trades_db_path and Path(trades_db_path).exists():
            result["executor_lift"] = _executor_lift(trades_db_path, ur)
        else:
            result["executor_lift"] = {"status": "skipped", "reason": "trades.db not available"}

        # 6. Full pipeline lift
        result["pipeline_lift"] = _pipeline_lift(ur, result)

        conn.close()
        return result

    except Exception as e:
        conn.close()
        logger.error("end_to_end.compute_lift_metrics failed: %s", e)
        return {"status": "error", "error": str(e)}


def build_attribution_table(
    research_db_path: str,
    trades_db_path: str | None = None,
    eval_date: str | None = None,
) -> pd.DataFrame:
    """
    Build the full-pipeline attribution table — one row per ticker per eval_date.

    Joins all evaluation tables to produce a wide DataFrame for analysis and CSV export.
    """
    if not Path(research_db_path).exists():
        return pd.DataFrame()

    conn = sqlite3.connect(research_db_path)

    date_filter = ""
    params: list = []
    if eval_date:
        date_filter = " WHERE ur.eval_date = ?"
        params = [eval_date]

    try:
        query = f"""
        SELECT
            ur.ticker,
            ur.eval_date,
            ur.sector,
            ur.close_price,
            ur.return_5d,
            ur.return_10d,
            ur.spy_return_5d,
            ur.spy_return_10d,
            ur.beat_spy_5d,
            ur.beat_spy_10d,
            ur.sector_etf,
            ur.sector_etf_return_5d,
            ur.beat_sector_5d,
            se.tech_score,
            se.scan_path,
            se.quant_filter_pass,
            se.liquidity_pass,
            se.volatility_pass,
            se.balance_sheet_pass,
            se.filter_fail_reason,
            tc.team_id,
            tc.quant_rank,
            tc.quant_score AS team_quant_score,
            tc.qual_score AS team_qual_score,
            tc.team_recommended,
            ce.combined_score AS cio_combined_score,
            ce.macro_shift AS cio_macro_shift,
            ce.final_score AS cio_final_score,
            ce.cio_decision,
            ce.cio_conviction,
            ce.cio_rank,
            po.predicted_direction,
            po.prediction_confidence,
            po.p_up,
            po.p_down,
            -- Canonical alpha (decimal scale) at the row's horizon-of-record.
            -- Post predictor 21d migration (2026-05-09): `actual_log_alpha`
            -- is decimal log-units at horizon_days; legacy `actual_5d_return`
            -- is pct points at 5d (divide by 100 to align scale).
            COALESCE(po.actual_log_alpha, po.actual_5d_return / 100.0) AS predictor_actual_alpha,
            COALESCE(po.horizon_days, 5) AS predictor_horizon_days
        FROM universe_returns ur
        LEFT JOIN scanner_evaluations se
            ON ur.ticker = se.ticker AND ur.eval_date = se.eval_date
        LEFT JOIN team_candidates tc
            ON ur.ticker = tc.ticker AND ur.eval_date = tc.eval_date
        LEFT JOIN cio_evaluations ce
            ON ur.ticker = ce.ticker AND ur.eval_date = ce.eval_date
        LEFT JOIN predictor_outcomes po
            ON ur.ticker = po.symbol AND ur.eval_date = po.prediction_date
        {date_filter}
        ORDER BY ur.eval_date, ur.ticker
        """
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        # Join trades if available
        if trades_db_path and Path(trades_db_path).exists():
            trades_conn = sqlite3.connect(trades_db_path)
            try:
                trades = pd.read_sql_query(
                    "SELECT ticker, date AS eval_date, action, fill_price, "
                    "realized_return_pct, trigger_type, exit_type "
                    "FROM trades WHERE action = 'ENTER'",
                    trades_conn,
                )
                if not trades.empty:
                    df = df.merge(trades, on=["ticker", "eval_date"], how="left")

                # Shadow book
                try:
                    shadow = pd.read_sql_query(
                        "SELECT ticker, date AS eval_date, block_reason, "
                        "intended_position_pct "
                        "FROM executor_shadow_book",
                        trades_conn,
                    )
                    if not shadow.empty:
                        df = df.merge(
                            shadow, on=["ticker", "eval_date"], how="left",
                            suffixes=("", "_shadow"),
                        )
                except sqlite3.OperationalError:
                    pass  # shadow book table may not exist yet
            finally:
                trades_conn.close()

        return df

    except Exception as e:
        logger.error("build_attribution_table failed: %s", e)
        conn.close()
        return pd.DataFrame()


def format_lift_report(metrics: dict) -> list[str]:
    """Format lift metrics as markdown lines for inclusion in the weekly report."""
    lines = ["## Pipeline evaluation — Decision boundary lift"]

    if metrics.get("status") != "ok":
        lines.append(f"\n> {metrics.get('status', 'unknown')}: {metrics.get('error', '')}")
        return lines

    lines.append(f"\n*{metrics['n_dates']} evaluation dates, {metrics['n_universe_rows']} universe rows*\n")

    lines.append("| Decision | Population | Avg 5d return | Baseline 5d return | Lift | n |")
    lines.append("|----------|------------|---------------|-------------------|------|---|")

    sl = metrics.get("scanner_lift", {})
    if sl and sl.get("status") != "skipped":
        lines.append(
            f"| Scanner filter | {sl.get('n_passing', '?')} / {sl.get('n_universe', '?')} | "
            f"{_pct(sl.get('passing_avg'))} | {_pct(sl.get('universe_avg'))} | "
            f"{_pct(sl.get('lift'))} | {sl.get('n_passing', '')} |"
        )

    cl = metrics.get("cio_lift", {})
    if cl and cl.get("status") != "skipped":
        lines.append(
            f"| CIO promotion | {cl.get('n_advance', '?')} / {cl.get('n_recs', '?')} | "
            f"{_pct(cl.get('advance_avg'))} | {_pct(cl.get('all_recs_avg'))} | "
            f"{_pct(cl.get('lift'))} | {cl.get('n_advance', '')} |"
        )

    pl = metrics.get("predictor_lift", {})
    if pl and pl.get("status") != "skipped":
        lines.append(
            f"| Predictor (UP) | {pl.get('n_up', '?')} / {pl.get('n_all', '?')} | "
            f"{_pct(pl.get('up_avg'))} | {_pct(pl.get('all_avg'))} | "
            f"{_pct(pl.get('lift'))} | {pl.get('n_up', '')} |"
        )

    el = metrics.get("executor_lift", {})
    if el and el.get("status") != "skipped":
        lines.append(
            f"| Executor trading | {el.get('n_traded', '?')} / {el.get('n_approved', '?')} | "
            f"{_pct(el.get('traded_avg'))} | {_pct(el.get('approved_avg'))} | "
            f"{_pct(el.get('lift'))} | {el.get('n_traded', '')} |"
        )

    pipl = metrics.get("pipeline_lift", {})
    if pipl and pipl.get("status") != "skipped":
        lines.append(
            f"| **Full pipeline** | — | "
            f"{_pct(pipl.get('traded_avg'))} | {_pct(pipl.get('universe_avg'))} | "
            f"**{_pct(pipl.get('lift'))}** | — |"
        )

    # CIO vs score-ranking baseline (2e)
    cvr = metrics.get("cio_vs_ranking", {})
    if cvr and cvr.get("status") != "skipped":
        verdict = "CIO outperforms" if cvr.get("cio_beats_ranking") else "Score ranking outperforms"
        lines.append(
            f"| CIO vs ranking | {cvr.get('n_picks', '?')} picks | "
            f"{_pct(cvr.get('cio_avg'))} | {_pct(cvr.get('ranking_avg'))} | "
            f"{_pct(cvr.get('lift'))} | {verdict} |"
        )

    # Team lift breakdown (2b: vs sector, 2c: vs quant).
    # `team_lift` is always a list[dict] after the producer-side
    # normalization in _team_lift (#13) — the old `isinstance(tl, dict)`
    # guard is dead code and dropped here.
    tl = metrics.get("team_lift", [])
    if tl:
        lines.append("\n### Sector team lift (2b: picks vs sector, 2c: picks vs quant candidates)\n")
        lines.append("| Team | Pick avg | Sector avg | Lift vs sector | Quant avg | Lift vs quant | Picks / Candidates |")
        lines.append("|------|----------|------------|----------------|-----------|---------------|-------------------|")
        for t in tl:
            lines.append(
                f"| {t.get('team_id', '?')} | {_pct(t.get('pick_avg'))} | "
                f"{_pct(t.get('sector_avg'))} | {_pct(t.get('lift'))} | "
                f"{_pct(t.get('quant_avg'))} | {_pct(t.get('lift_vs_quant'))} | "
                f"{t.get('n_picks', 0)} / {t.get('n_candidates', 0)} |"
            )

        # Summary insight
        if len(tl) > 1:
            lifts = [t["lift"] for t in tl if t.get("lift") is not None]
            quant_lifts = [t["lift_vs_quant"] for t in tl if t.get("lift_vs_quant") is not None]
            if lifts:
                avg_lift = sum(lifts) / len(lifts)
                lines.append(f"\n> Avg team lift vs sector: {_pct(avg_lift)}")
            if quant_lifts:
                avg_ql = sum(quant_lifts) / len(quant_lifts)
                if avg_ql > 0:
                    lines.append(f"> LLM qual/peer review adds {_pct(avg_ql)} over quant alone")
                else:
                    lines.append(f"> LLM qual/peer review subtracts {_pct(avg_ql)} vs quant alone — consider simplifying")

    # Classification metrics summary table (precision/recall/F1 per decision boundary)
    clf_rows = []
    for label, key in [
        ("Scanner", "scanner_lift"),
        ("CIO", "cio_lift"),
        ("Predictor (UP)", "predictor_lift"),
        ("Executor", "executor_lift"),
    ]:
        sub = metrics.get(key, {})
        clf = sub.get("classification") if isinstance(sub, dict) else None
        if clf:
            clf_rows.append((label, clf))

    # Per-team classification
    if tl and not isinstance(tl, dict):
        for t in tl:
            clf = t.get("classification")
            if clf:
                clf_rows.append((f"  {t.get('team_id', '?')}", clf))

    if clf_rows:
        lines.append("\n### Classification metrics (precision / recall / F1)\n")
        lines.append("| Decision | Precision | Recall | F1 | TP | FP | FN | TN | n |")
        lines.append("|----------|-----------|--------|----|----|----|----|----|---|")
        for label, c in clf_rows:
            p = _pct(c.get("precision")) if c.get("precision") is not None else "—"
            r = _pct(c.get("recall")) if c.get("recall") is not None else "—"
            f = f"{c['f1']:.3f}" if c.get("f1") is not None else "—"
            lines.append(
                f"| {label} | {p} | {r} | {f} | "
                f"{c.get('tp', '—')} | {c.get('fp', '—')} | {c.get('fn', '—')} | {c.get('tn', '—')} | {c.get('n', '—')} |"
            )

    return lines


# ── Internal lift computations ───────────────────────────────────────────────

def _scanner_lift(conn, ur: pd.DataFrame, date_filter: str, params: list) -> dict:
    """Scanner filter lift: passing stocks vs. full universe."""
    try:
        se_filter = date_filter.replace("eval_date", "se.eval_date") if date_filter else ""
        se = pd.read_sql_query(
            f"SELECT ticker, eval_date, quant_filter_pass FROM scanner_evaluations se{se_filter}",
            conn, params=params,
        )
        if se.empty:
            return {"status": "skipped", "reason": "scanner_evaluations empty"}

        merged = ur.merge(se, on=["ticker", "eval_date"], how="inner")
        passing = merged[merged["quant_filter_pass"] == 1]

        universe_avg = float(merged["return_5d"].mean())
        passing_avg = float(passing["return_5d"].mean()) if not passing.empty else None
        lift = (passing_avg - universe_avg) if passing_avg is not None else None

        # Classification metrics: selected=passed filter, positive=beat SPY
        clf = None
        if "beat_spy_5d" in merged.columns:
            has_outcome = merged["beat_spy_5d"].notna()
            if has_outcome.any():
                m = merged[has_outcome]
                selected = (m["quant_filter_pass"] == 1).tolist()
                positive = (m["beat_spy_5d"] == 1).tolist()
                clf = compute_binary_metrics(
                    tp=sum(s and p for s, p in zip(selected, positive)),
                    fp=sum(s and not p for s, p in zip(selected, positive)),
                    fn=sum(not s and p for s, p in zip(selected, positive)),
                    tn=sum(not s and not p for s, p in zip(selected, positive)),
                )

        return {
            "universe_avg": round(universe_avg, 4),
            "passing_avg": round(passing_avg, 4) if passing_avg is not None else None,
            "lift": round(lift, 4) if lift is not None else None,
            "n_universe": len(merged),
            "n_passing": len(passing),
            "classification": clf,
        }
    except sqlite3.OperationalError:
        return {"status": "skipped", "reason": "scanner_evaluations table not found"}


def _team_lift(conn, ur: pd.DataFrame, date_filter: str, params: list) -> list[dict]:
    """Sector team lift (2b): team picks vs. own sector average from full 900.

    The baseline is the average return of ALL stocks in the same sector from
    universe_returns — not just the quant candidates. This measures whether
    each team's picks outperform their sector's random baseline.

    Returns a list of team-lift dicts. Returns an empty list on skip/error
    conditions (missing table, empty data) — downstream consumers iterate
    team_lift and the empty-list case is handled naturally. A prior version
    returned a status dict in these cases, which violated the list contract
    and crashed grading._grade_sector_team on 2026-04-11.
    """
    try:
        tc_filter = date_filter.replace("eval_date", "tc.eval_date") if date_filter else ""
        tc = pd.read_sql_query(
            f"SELECT ticker, eval_date, team_id, team_recommended FROM team_candidates tc{tc_filter}",
            conn, params=params,
        )
        if tc.empty:
            logger.info("team_lift: team_candidates table empty — returning []")
            return []

        merged = ur.merge(tc, on=["ticker", "eval_date"], how="inner")
        results = []

        for team_id in sorted(merged["team_id"].unique()):
            team_data = merged[merged["team_id"] == team_id]
            picks = team_data[team_data["team_recommended"] == 1]

            # Get the sector(s) this team covers from universe_returns
            pick_sectors = picks["sector"].dropna().unique() if "sector" in picks.columns else []

            # Full sector average from the 900-stock universe (not just quant candidates)
            if len(pick_sectors) > 0:
                sector_universe = ur[ur["sector"].isin(pick_sectors)]
                full_sector_avg = float(sector_universe["return_5d"].mean()) if not sector_universe.empty else None
                n_sector_universe = len(sector_universe)
            else:
                full_sector_avg = None
                n_sector_universe = 0

            # Quant candidates average (for 2c comparison)
            quant_avg = float(team_data["return_5d"].mean())
            pick_avg = float(picks["return_5d"].mean()) if not picks.empty else None

            # Primary lift: picks vs full sector (2b)
            lift_vs_sector = (pick_avg - full_sector_avg) if pick_avg is not None and full_sector_avg is not None else None
            # Secondary lift: picks vs quant candidates (2c)
            lift_vs_quant = (pick_avg - quant_avg) if pick_avg is not None else None

            # Classification: selected=team picked, positive=beat sector ETF
            clf = None
            beat_col = "beat_sector_5d" if "beat_sector_5d" in team_data.columns else "beat_spy_5d"
            if beat_col in team_data.columns:
                has_outcome = team_data[beat_col].notna()
                if has_outcome.any():
                    m = team_data[has_outcome]
                    selected = (m["team_recommended"] == 1).tolist()
                    positive = (m[beat_col] == 1).tolist()
                    clf = compute_binary_metrics(
                        tp=sum(s and p for s, p in zip(selected, positive)),
                        fp=sum(s and not p for s, p in zip(selected, positive)),
                        fn=sum(not s and p for s, p in zip(selected, positive)),
                        tn=sum(not s and not p for s, p in zip(selected, positive)),
                    )

            # Emit the per-pick records so downstream metric modules
            # (information_coefficient, expectancy, excursion, the
            # team-daily-returns sleeve simulator) can consume the
            # specific (ticker, eval_date) tuples without re-querying
            # research.db. Conviction is forwarded when present in
            # universe_returns; absent → None (older rows). Schema:
            #   [{ticker, eval_date, return_5d, conviction|None}, ...]
            picks_records: list[dict] = []
            if not picks.empty:
                pick_cols = ["ticker", "eval_date", "return_5d"]
                if "conviction" in picks.columns:
                    pick_cols.append("conviction")
                for _, p in picks[pick_cols].iterrows():
                    rec = {
                        "ticker": p["ticker"],
                        "eval_date": str(p["eval_date"]),
                        "return_5d": float(p["return_5d"]) if pd.notna(p["return_5d"]) else None,
                    }
                    if "conviction" in pick_cols:
                        rec["conviction"] = p["conviction"] if pd.notna(p["conviction"]) else None
                    picks_records.append(rec)

            results.append({
                "team_id": team_id,
                "pick_avg": round(pick_avg, 4) if pick_avg is not None else None,
                "sector_avg": round(full_sector_avg, 4) if full_sector_avg is not None else None,
                "quant_avg": round(quant_avg, 4),
                "lift": round(lift_vs_sector, 4) if lift_vs_sector is not None else None,
                "lift_vs_quant": round(lift_vs_quant, 4) if lift_vs_quant is not None else None,
                "n_picks": len(picks),
                "n_candidates": len(team_data),
                "n_sector_universe": n_sector_universe,
                "classification": clf,
                "picks": picks_records,
            })

        return results
    except sqlite3.OperationalError:
        logger.info("team_lift: team_candidates table not found — returning []")
        return []


def _cio_lift(conn, ur: pd.DataFrame, date_filter: str, params: list) -> dict:
    """CIO lift (2d): ADVANCE stocks vs. all sector recommendations.

    Emits per-bucket stdev so downstream optimizers can compute a
    Welch-style confidence bound on whether ``advance_avg < all_recs_avg``
    is statistically distinguishable from sampling noise. Without this,
    the CIO-fallback recommendation triggers on small-sample noise.
    """
    try:
        ce_filter = date_filter.replace("eval_date", "ce.eval_date") if date_filter else ""
        ce = pd.read_sql_query(
            f"SELECT ticker, eval_date, cio_decision, final_score FROM cio_evaluations ce{ce_filter}",
            conn, params=params,
        )
        if ce.empty:
            return {"status": "skipped", "reason": "cio_evaluations empty"}

        merged = ur.merge(ce, on=["ticker", "eval_date"], how="inner")
        advance = merged[merged["cio_decision"] == "ADVANCE"]

        all_recs_avg = float(merged["return_5d"].mean())
        advance_avg = float(advance["return_5d"].mean()) if not advance.empty else None
        lift = (advance_avg - all_recs_avg) if advance_avg is not None else None

        reject = merged[merged["cio_decision"] == "REJECT"]
        reject_avg = float(reject["return_5d"].mean()) if not reject.empty else None

        # Welch-style inputs: stdev (ddof=1) of return_5d per bucket. Only
        # meaningful at n ≥ 2; emit None otherwise so the consumer skips
        # the confidence check rather than dividing by zero.
        def _std(s):
            v = s.dropna()
            return float(v.std(ddof=1)) if len(v) >= 2 else None
        all_recs_std = _std(merged["return_5d"])
        advance_std = _std(advance["return_5d"]) if not advance.empty else None
        reject_std = _std(reject["return_5d"]) if not reject.empty else None

        # Classification: selected=CIO advanced, positive=beat SPY
        clf = None
        if "beat_spy_5d" in merged.columns:
            has_outcome = merged["beat_spy_5d"].notna()
            if has_outcome.any():
                m = merged[has_outcome]
                selected = (m["cio_decision"] == "ADVANCE").tolist()
                positive = (m["beat_spy_5d"] == 1).tolist()
                clf = compute_binary_metrics(
                    tp=sum(s and p for s, p in zip(selected, positive)),
                    fp=sum(s and not p for s, p in zip(selected, positive)),
                    fn=sum(not s and p for s, p in zip(selected, positive)),
                    tn=sum(not s and not p for s, p in zip(selected, positive)),
                )

        return {
            "all_recs_avg": round(all_recs_avg, 4),
            "advance_avg": round(advance_avg, 4) if advance_avg is not None else None,
            "reject_avg": round(reject_avg, 4) if reject_avg is not None else None,
            "lift": round(lift, 4) if lift is not None else None,
            "n_recs": len(merged),
            "n_advance": len(advance),
            "n_reject": len(reject),
            "all_recs_std_5d": round(all_recs_std, 4) if all_recs_std is not None else None,
            "advance_std_5d": round(advance_std, 4) if advance_std is not None else None,
            "reject_std_5d": round(reject_std, 4) if reject_std is not None else None,
            "classification": clf,
        }
    except sqlite3.OperationalError:
        return {"status": "skipped", "reason": "cio_evaluations table not found"}


def _cio_vs_ranking_lift(conn, ur: pd.DataFrame, date_filter: str, params: list) -> dict:
    """CIO vs score-ranking baseline (2e): does LLM judgment beat mechanical ranking?

    Compares CIO's actual ADVANCE picks vs. a counterfactual where we simply
    take the top N candidates by final_score (no LLM judgment). If the score-
    ranking baseline performs equally well, the CIO step can be simplified.
    """
    try:
        ce_filter = date_filter.replace("eval_date", "ce.eval_date") if date_filter else ""
        ce = pd.read_sql_query(
            f"SELECT ticker, eval_date, cio_decision, final_score FROM cio_evaluations ce{ce_filter}",
            conn, params=params,
        )
        if ce.empty:
            return {"status": "skipped", "reason": "cio_evaluations empty"}

        merged = ur.merge(ce, on=["ticker", "eval_date"], how="inner")
        if merged.empty or "final_score" not in merged.columns:
            return {"status": "skipped", "reason": "no matched CIO rows with final_score"}

        advance = merged[merged["cio_decision"] == "ADVANCE"]
        if advance.empty:
            return {"status": "skipped", "reason": "no ADVANCE decisions"}

        results_by_date = []
        for eval_date in merged["eval_date"].unique():
            date_pool = merged[merged["eval_date"] == eval_date]
            date_advance = date_pool[date_pool["cio_decision"] == "ADVANCE"]
            n_advance = len(date_advance)
            if n_advance == 0 or date_pool["final_score"].isna().all():
                continue

            # Score-ranking baseline: top N by final_score
            top_n = date_pool.nlargest(n_advance, "final_score")

            results_by_date.append({
                "cio_avg": float(date_advance["return_5d"].mean()),
                "ranking_avg": float(top_n["return_5d"].mean()),
                "n": n_advance,
                # Overlap: how many of CIO's picks are also in the top-N?
                "overlap": len(set(date_advance["ticker"]) & set(top_n["ticker"])),
            })

        if not results_by_date:
            return {"status": "skipped", "reason": "no dates with valid score ranking data"}

        rdf = pd.DataFrame(results_by_date)

        cio_avg = round(float(rdf["cio_avg"].mean()), 4)
        ranking_avg = round(float(rdf["ranking_avg"].mean()), 4)
        lift = round(cio_avg - ranking_avg, 4)
        avg_overlap = round(float(rdf["overlap"].mean()), 1)
        total_n = int(rdf["n"].sum())

        return {
            "cio_avg": cio_avg,
            "ranking_avg": ranking_avg,
            "lift": lift,
            "avg_overlap": avg_overlap,
            "n_dates": len(rdf),
            "n_picks": total_n,
            "cio_beats_ranking": lift > 0,
        }
    except sqlite3.OperationalError:
        return {"status": "skipped", "reason": "cio_evaluations table not found"}


def _predictor_lift(conn, ur: pd.DataFrame, date_filter: str, params: list) -> dict:
    """Predictor lift: UP-predicted vs. all portfolio stocks."""
    try:
        po_filter = date_filter.replace("eval_date = ?", "prediction_date = ?") if date_filter else ""
        po = pd.read_sql_query(
            f"SELECT symbol AS ticker, prediction_date AS eval_date, "
            f"predicted_direction, prediction_confidence "
            f"FROM predictor_outcomes{po_filter}",
            conn, params=params,
        )
        if po.empty:
            return {"status": "skipped", "reason": "predictor_outcomes empty"}

        merged = ur.merge(po, on=["ticker", "eval_date"], how="inner")
        if merged.empty:
            return {"status": "skipped", "reason": "no matching predictor_outcomes in universe_returns"}

        up = merged[merged["predicted_direction"] == "UP"]
        down = merged[merged["predicted_direction"] == "DOWN"]

        all_avg = float(merged["return_5d"].mean())
        up_avg = float(up["return_5d"].mean()) if not up.empty else None
        down_avg = float(down["return_5d"].mean()) if not down.empty else None
        lift = (up_avg - all_avg) if up_avg is not None else None

        # Classification: selected=predicted UP, positive=beat SPY
        clf = None
        if "beat_spy_5d" in merged.columns:
            has_outcome = merged["beat_spy_5d"].notna()
            if has_outcome.any():
                m = merged[has_outcome]
                selected = (m["predicted_direction"] == "UP").tolist()
                positive = (m["beat_spy_5d"] == 1).tolist()
                clf = compute_binary_metrics(
                    tp=sum(s and p for s, p in zip(selected, positive)),
                    fp=sum(s and not p for s, p in zip(selected, positive)),
                    fn=sum(not s and p for s, p in zip(selected, positive)),
                    tn=sum(not s and not p for s, p in zip(selected, positive)),
                )

        return {
            "all_avg": round(all_avg, 4),
            "up_avg": round(up_avg, 4) if up_avg is not None else None,
            "down_avg": round(down_avg, 4) if down_avg is not None else None,
            "lift": round(lift, 4) if lift is not None else None,
            "n_all": len(merged),
            "n_up": len(up),
            "n_down": len(down),
            "classification": clf,
        }
    except sqlite3.OperationalError:
        return {"status": "skipped", "reason": "predictor_outcomes table not found"}


def _executor_lift(trades_db_path: str, ur: pd.DataFrame) -> dict:
    """Executor lift: traded returns vs. approved (non-blocked) entries."""
    try:
        trades_conn = sqlite3.connect(trades_db_path)
        trades = pd.read_sql_query(
            "SELECT ticker, date AS eval_date, realized_return_pct "
            "FROM trades WHERE action = 'ENTER'",
            trades_conn,
        )

        # Shadow book (blocked entries)
        try:
            shadow = pd.read_sql_query(
                "SELECT ticker, date AS eval_date, block_reason "
                "FROM executor_shadow_book",
                trades_conn,
            )
        except sqlite3.OperationalError:
            shadow = pd.DataFrame()

        trades_conn.close()

        if trades.empty:
            return {"status": "skipped", "reason": "no ENTER trades"}

        # Merge trades with universe_returns for forward returns
        traded = ur.merge(trades, on=["ticker", "eval_date"], how="inner")

        # Approved = traded + not blocked
        # "Approved" baseline is all portfolio stocks that weren't blocked
        approved = ur.merge(
            trades[["ticker", "eval_date"]], on=["ticker", "eval_date"], how="inner"
        )
        if not shadow.empty:
            blocked = ur.merge(shadow[["ticker", "eval_date"]], on=["ticker", "eval_date"], how="inner")
            approved = pd.concat([approved, blocked])

        traded_avg = float(traded["return_5d"].mean()) if not traded.empty else None
        approved_avg = float(approved["return_5d"].mean()) if not approved.empty else None
        lift = (traded_avg - approved_avg) if traded_avg is not None and approved_avg is not None else None

        # Classification: selected=traded (not blocked), positive=beat SPY
        clf = None
        if not shadow.empty and "beat_spy_5d" in approved.columns:
            has_outcome = approved["beat_spy_5d"].notna()
            if has_outcome.any():
                m = approved[has_outcome]
                traded_tickers = set(zip(traded["ticker"], traded["eval_date"]))
                selected = [
                    (r["ticker"], r["eval_date"]) in traded_tickers
                    for _, r in m.iterrows()
                ]
                positive = (m["beat_spy_5d"] == 1).tolist()
                clf = compute_binary_metrics(
                    tp=sum(s and p for s, p in zip(selected, positive)),
                    fp=sum(s and not p for s, p in zip(selected, positive)),
                    fn=sum(not s and p for s, p in zip(selected, positive)),
                    tn=sum(not s and not p for s, p in zip(selected, positive)),
                )

        return {
            "traded_avg": round(traded_avg, 4) if traded_avg is not None else None,
            "approved_avg": round(approved_avg, 4) if approved_avg is not None else None,
            "lift": round(lift, 4) if lift is not None else None,
            "n_traded": len(traded),
            "n_approved": len(approved),
            "classification": clf,
        }
    except Exception as e:
        return {"status": "skipped", "reason": str(e)}


def _pipeline_lift(ur: pd.DataFrame, result: dict) -> dict:
    """Full pipeline lift: best available traded return vs. universe average."""
    universe_avg = float(ur["return_5d"].mean())

    # Use executor traded_avg if available, otherwise CIO advance_avg
    el = result.get("executor_lift", {})
    cl = result.get("cio_lift", {})

    traded_avg = el.get("traded_avg") if el.get("status") != "skipped" else None
    if traded_avg is None:
        traded_avg = cl.get("advance_avg") if cl.get("status") != "skipped" else None

    if traded_avg is None:
        return {"status": "skipped", "reason": "no downstream returns available"}

    return {
        "universe_avg": round(universe_avg, 4),
        "traded_avg": round(traded_avg, 4),
        "lift": round(traded_avg - universe_avg, 4),
    }


def _pct(val) -> str:
    """Format a decimal return as percentage string."""
    if val is None:
        return "—"
    return f"{val * 100:+.2f}%"
