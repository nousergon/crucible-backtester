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

import json
import logging
import sqlite3
from pathlib import Path

import pandas as pd

from analysis.classification_metrics import compute_binary_metrics

logger = logging.getLogger(__name__)

# config#2318: since the 2026-06-29 attractiveness champion-feed cutover, the
# tech_score scanner (``scanner_evaluations.quant_filter_pass``) runs only as a
# recorded BASELINE arm — it no longer feeds live candidate generation. Scanner-
# edge metrics below still read that table, so they are explicitly labeled with
# this arm so downstream report-card/Director surfaces cannot present a retired
# gate's record as "the scanner" unlabeled. Additive-only field (S3 contract
# discipline) — does not replace/rename any existing key.
SCANNER_METRIC_ARM = "tech_score_baseline (retired from live feed 2026-06-29)"

# config#1580 / config-I2993: the six-team + macro-economist + CIO research
# orchestration was RETIRED. The live ``ne-weekly-freshness-pipeline`` has no
# state that invokes that graph; ``research.db`` ``team_candidates`` /
# ``cio_evaluations`` stop at 2026-07-10 (last producing cycle Sat 2026-07-11)
# and will never gain new rows. ``_team_lift`` / ``_cio_lift`` were re-summing
# that ENTIRE dead history at full weight every weekly cycle (live 2026-07-17
# artifact: n_dates=153 back to 2025-12), polluting the Report Card + Director.
# The live path now emits explicit retired markers instead of live-weight
# aggregates (the functions are RETAINED, uncalled-in-live, for direct/historical
# readouts + estimator-math tests).
RESEARCH_GRAPH_RETIRED_DATE = "2026-07-12"

# config-I2994 live-arm labels. The champion feed's ranking score is the scanner
# universe-board ``attractiveness_score`` (copied verbatim into signals.json
# ``score`` by crucible-research ``scoring/signals_envelope.py``); its
# date-clustered IC vs realized 21d alpha is produced canonically by
# ``attractiveness_eval.py`` (``composite_ic``) — NOT duplicated here. The Think
# Tank challenger's shadow score is the only research-authored composite left.
SCANNER_ATTRACTIVENESS_ARM = (
    "scanner_attractiveness (live champion feed — universe-board "
    "attractiveness_score; canonical IC in attractiveness_eval.json:composite_ic)"
)
THINKTANK_SHADOW_ARM = (
    "thinktank_coverage (observe-only challenger shadow scores; "
    "signals_shadow/thinktank_coverage/)"
)


# ── Canonical-horizon research-edge helpers (ROADMAP L4551) ─────────────────
#
# The system targets 21-day log-domain market-relative alpha (canonical-alpha
# cutover 2026-05-09). The legacy lift metrics below measure selection skill on
# `return_5d` / `beat_spy_5d` only — a 5-trading-day window mismatched to the
# 21-day thesis horizon the research picks are constructed for, which collapses
# every selector's precision toward the base rate regardless of real 21d edge.
# These helpers emit the canonical 21d classification + log-domain alpha lift
# ADDITIVELY (5d retained as a short-horizon diagnostic); the report-card tiles
# grade the 21d block, falling back to 5d for pre-2026-06-07 artifacts.


def _classification_for(merged: pd.DataFrame, selected_mask, beat_col: str) -> dict | None:
    """Binary selection metrics for a given realized-outcome column.

    ``selected_mask`` is a boolean Series aligned to ``merged`` (True = the
    selector chose this row). Returns ``None`` when ``beat_col`` is absent or has
    no closed outcomes yet (e.g. the 21d forward window hasn't closed) so the
    caller emits ``None`` and the grader reads N/A rather than a phantom zero.
    """
    if beat_col not in merged.columns:
        return None
    has = merged[beat_col].notna()
    if not has.any():
        return None
    m = merged[has]
    sel = selected_mask[has].tolist()
    pos = (m[beat_col] == 1).tolist()
    return compute_binary_metrics(
        tp=sum(s and p for s, p in zip(sel, pos)),
        fp=sum(s and not p for s, p in zip(sel, pos)),
        fn=sum(not s and p for s, p in zip(sel, pos)),
        tn=sum(not s and not p for s, p in zip(sel, pos)),
    )


def _alpha_21d_log_lift(merged: pd.DataFrame, selected_mask) -> dict | None:
    """Selected-vs-baseline 21d log-domain market-relative alpha lift.

    Uses ``log_return_21d - log_spy_return_21d`` — the exact unit the system
    trades toward. Returns ``None`` when the canonical columns are absent or no
    21d outcomes have closed.

    Emits THREE baselines so a funnel stage's edge can be read on a clean
    yardstick (config#967 funnel-measurement, 2026-06-22):

    * ``lift`` — selected vs the FULL input pool (legacy; selected ⊆ baseline,
      so it DILUTES the contrast — kept for back-compat / existing graders).
    * ``lift_vs_rejected`` — selected vs the REJECTED complement (the un-diluted
      selection-skill measure: did this stage keep the better names *out of*
      what it was handed?).
    * ``sn_lift_vs_rejected`` — selected vs rejected on the SECTOR(+cycle)-
      NEUTRALIZED residual (alpha minus its per-``(eval_date, sector)`` group
      mean), so a stage can't look skillful merely by tilting into a strong
      sector or a strong week. Present only when ``sector`` is in the frame.

    All new fields are ADDITIVE (S3 schema-contract safe). Beta-matched
    baselines are a follow-up (need per-name beta from ArcticDB; the alpha is
    already SPY-relative so beta is a second-order confound).
    """
    if "log_return_21d" not in merged.columns or "log_spy_return_21d" not in merged.columns:
        return None
    alpha = merged["log_return_21d"] - merged["log_spy_return_21d"]
    base = alpha.dropna()
    if base.empty:
        return None
    sel_alpha = alpha[selected_mask].dropna()
    rej_alpha = alpha[~selected_mask].dropna()
    sel_avg = float(sel_alpha.mean()) if not sel_alpha.empty else None
    rej_avg = float(rej_alpha.mean()) if not rej_alpha.empty else None
    base_avg = float(base.mean())
    lift = (sel_avg - base_avg) if sel_avg is not None else None
    lift_vs_rej = (
        (sel_avg - rej_avg) if (sel_avg is not None and rej_avg is not None) else None
    )

    out = {
        "selected_avg": round(sel_avg, 5) if sel_avg is not None else None,
        "baseline_avg": round(base_avg, 5),
        "lift": round(lift, 5) if lift is not None else None,
        "n_selected": int(sel_alpha.shape[0]),
        "n_baseline": int(base.shape[0]),
        # Complement (selected vs rejected) — the un-diluted selection contrast.
        "rejected_avg": round(rej_avg, 5) if rej_avg is not None else None,
        "n_rejected": int(rej_alpha.shape[0]),
        "lift_vs_rejected": round(lift_vs_rej, 5) if lift_vs_rej is not None else None,
    }

    # Sector(+cycle)-neutral selected-vs-rejected: residualize alpha against its
    # per-(eval_date, sector) group mean so the lift reflects WITHIN-peer-group
    # selection skill, not a sector/time tilt.
    if "sector" in merged.columns:
        grp_cols = (["eval_date"] if "eval_date" in merged.columns else []) + ["sector"]
        valid = alpha.notna() & merged["sector"].notna()
        if "eval_date" in grp_cols:
            valid = valid & merged["eval_date"].notna()
        if valid.any():
            keys = [merged.loc[valid, c] for c in grp_cols]
            gmean = alpha[valid].groupby(keys).transform("mean")
            resid = pd.Series(float("nan"), index=merged.index, dtype="float64")
            resid.loc[gmean.index] = alpha[valid] - gmean
            sn_sel = resid[selected_mask].dropna()
            sn_rej = resid[~selected_mask].dropna()
            sn_sel_avg = float(sn_sel.mean()) if not sn_sel.empty else None
            sn_rej_avg = float(sn_rej.mean()) if not sn_rej.empty else None
            sn_lift = (
                (sn_sel_avg - sn_rej_avg)
                if (sn_sel_avg is not None and sn_rej_avg is not None)
                else None
            )
            out["sn_basis"] = "+".join(grp_cols)
            out["sn_selected_avg"] = round(sn_sel_avg, 5) if sn_sel_avg is not None else None
            out["sn_rejected_avg"] = round(sn_rej_avg, 5) if sn_rej_avg is not None else None
            out["sn_lift_vs_rejected"] = round(sn_lift, 5) if sn_lift is not None else None
            out["sn_n_selected"] = int(sn_sel.shape[0])
            out["sn_n_rejected"] = int(sn_rej.shape[0])

    return out


def _cio_selection_skill(merged: pd.DataFrame) -> dict | None:
    """CIO entrant-gate SELECTION skill at the canonical 21d horizon (L4561).

    The CIO gate's job is to ADVANCE the team-recommended names that will realize
    HIGHER forward alpha than the ones it REJECTs. This measures whether it does:

      * ``selection_gap_21d`` = mean(ADVANCE 21d log-alpha) − mean(REJECT 21d
        log-alpha). POSITIVE = the gate adds selection value; NEGATIVE = it is
        anti-selecting (advancing the worse names). With a Mann-Whitney p so a
        small-sample gap is not mistaken for a real one.
      * ``conviction_ic_21d`` = Spearman rank-IC of ``cio_conviction`` vs realized
        21d alpha across all evaluated names — does conviction order outcomes?

    Returns ``None`` when the canonical 21d columns or any closed-21d outcomes
    are absent (no phantom zeros). This is a measurement instrument: it makes the
    gate's skill visible every cycle rather than silently inferred — it does NOT
    itself change any selection. Graded under the L4562 reliability contract: a
    statistically-insignificant gap reads WATCH + reliability=low, not a
    confident RED.
    """
    if "log_return_21d" not in merged.columns or "log_spy_return_21d" not in merged.columns:
        return None
    df = merged.copy()
    df["alpha21"] = df["log_return_21d"] - df["log_spy_return_21d"]
    df = df.dropna(subset=["alpha21"])
    if df.empty:
        return None
    adv = df[df["cio_decision"].isin(("ADVANCE", "ADVANCE_FORCED"))]["alpha21"]
    rej = df[df["cio_decision"] == "REJECT"]["alpha21"]
    out: dict = {
        "n_advance": int(adv.shape[0]),
        "n_reject": int(rej.shape[0]),
        "advance_alpha_21d": round(float(adv.mean()), 5) if not adv.empty else None,
        "reject_alpha_21d": round(float(rej.mean()), 5) if not rej.empty else None,
        "selection_gap_21d": None,
        "selection_gap_p": None,
        "conviction_ic_21d": None,
        "conviction_ic_p": None,
    }
    if not adv.empty and not rej.empty:
        out["selection_gap_21d"] = round(float(adv.mean() - rej.mean()), 5)
        if adv.shape[0] >= 3 and rej.shape[0] >= 3:
            from scipy.stats import mannwhitneyu
            try:
                _, p = mannwhitneyu(adv, rej, alternative="two-sided")
                out["selection_gap_p"] = round(float(p), 4)
            except ValueError:
                pass
    if "cio_conviction" in df.columns:
        cc = df.dropna(subset=["cio_conviction"])
        if cc.shape[0] >= 5 and cc["cio_conviction"].nunique() >= 2 and cc["alpha21"].nunique() >= 2:
            from scipy.stats import spearmanr
            rho, p = spearmanr(cc["cio_conviction"], cc["alpha21"])
            if rho == rho:
                out["conviction_ic_21d"] = round(float(rho), 4)
            if p == p:
                out["conviction_ic_p"] = round(float(p), 4)
    return out


def _trailing_sector_neutral(
    df: pd.DataFrame,
    *,
    score_col: str = "combined_score",
    date_col: str = "eval_date",
    sector_col: str = "sector",
    k_min: int = 6,
) -> tuple[pd.Series, float]:
    """Leak-free sector-neutral transform of ``score_col`` (L4564).

    The six sector teams share one rubric, so raw scores carry a persistent
    per-sector level/scale bias (a 70 from Tech ≠ a 70 from Defensives). This
    strips that bias deterministically, so the residual is comparable across
    sectors — the precondition for the CIO ranking on stock quality and
    applying the sector tilt SEPARATELY (rather than letting the rubric's
    sector bias double-count with the explicit sector ratings).

    For each row at eval_date ``d`` in sector ``s``::

        q = (score − μ_s) / σ_s

    where μ_s, σ_s are the mean/std of that sector's scores over STRICTLY
    PRIOR eval_dates (``< d``) — no look-ahead, exactly what is reconstructable
    live from ``research.db`` at decision time (the CIO candidate pool is only
    ~2–3 names/sector/cycle, far too thin for a within-cycle within-sector
    z-score, so the baseline MUST come from history). Sectors with < ``k_min``
    prior samples (cold start / thin) fall back to the within-cycle pool-wide
    percentile rank, so every row gets a comparable value rather than a NaN.

    Returns ``(series_aligned_to_df_index, frac_neutralized)`` — the second
    element is the share of valued rows that used the true trailing-sector
    transform vs the pool-wide fallback (transparency for the L4564 gate).
    """
    import numpy as np

    out = pd.Series(np.nan, index=df.index, dtype="float64")
    if not {score_col, date_col, sector_col}.issubset(df.columns):
        return out, 0.0
    # Pool-wide within-cycle percentile rank — the cold-start / thin fallback.
    pool_rank = df.groupby(date_col)[score_col].rank(pct=True)
    n_neutral = 0
    # eval_date is a YYYY-MM-DD string — lexical order is chronological order.
    for d in sorted(df[date_col].dropna().unique()):
        cur = df[df[date_col] == d]
        prior = df[df[date_col] < d]
        for s, idx in cur.groupby(sector_col).groups.items():
            prior_s = prior[prior[sector_col] == s][score_col].dropna()
            sd = prior_s.std(ddof=1) if len(prior_s) >= k_min else None
            if sd is not None and sd == sd and sd > 1e-9:
                out.loc[idx] = (df.loc[idx, score_col] - prior_s.mean()) / sd
                n_neutral += int(df.loc[idx, score_col].notna().sum())
            else:
                out.loc[idx] = pool_rank.loc[idx]
    valid = int(out.notna().sum())
    return out, (n_neutral / valid if valid else 0.0)


def _cio_layer_attribution(merged: pd.DataFrame) -> dict | None:
    """Attribute realized 21d alpha to each layer the CIO orchestrates (L4561/L4562).

    The CIO synthesizes several inputs — the sector-team stock score
    (``combined_score``, pre-macro), the macro/sector tilt (``macro_shift``), the
    blended ``final_score`` it currently ranks on, and its own ``cio_conviction``.
    This emits the Spearman rank-IC (+p) of EACH vs realized 21d alpha across all
    CIO-evaluated names, so the harness can see which layer carries forward signal
    and which is noise/anti-signal — the precondition for de-blending the
    orchestration. Diagnostic; ``None`` when canonical 21d cols / closed outcomes
    are absent.
    """
    if "log_return_21d" not in merged.columns or "log_spy_return_21d" not in merged.columns:
        return None
    df = merged.copy()
    df["alpha21"] = df["log_return_21d"] - df["log_spy_return_21d"]
    df = df.dropna(subset=["alpha21"])
    if df.shape[0] < 5:
        return None
    from scipy.stats import spearmanr

    out: dict = {"n": int(df.shape[0])}
    for layer in ("combined_score", "macro_shift", "final_score", "cio_conviction"):
        if layer not in df.columns:
            out[f"{layer}_ic"] = None
            out[f"{layer}_ic_p"] = None
            continue
        sub = df.dropna(subset=[layer])
        if sub.shape[0] >= 5 and sub[layer].nunique() >= 2 and sub["alpha21"].nunique() >= 2:
            rho, p = spearmanr(sub[layer], sub["alpha21"])
            out[f"{layer}_ic"] = round(float(rho), 4) if rho == rho else None
            out[f"{layer}_ic_p"] = round(float(p), 4) if p == p else None
        else:
            out[f"{layer}_ic"] = None
            out[f"{layer}_ic_p"] = None

    # Date-clustered IC significance (de-pseudo-replication of the pooled ICs above).
    # The pooled ``{layer}_ic`` Spearman counts every CIO-evaluated name as an
    # independent observation, but a research cycle's signal lives at the eval_date
    # level — and ``macro_shift`` in particular is a per-(sector,date) tilt with only
    # a handful of distinct values per cycle, so pooling ~K weeks of names as N≈K·25
    # draws manufactures significance the panel does not have (on live data the
    # pooled ``final_score_ic`` reads p≈0.02 while the date-clustered estimator reads
    # ≈0.18 on the SAME rows — a textbook pseudo-replication artifact that was driving
    # a false "significant negative composite IC" report-card flag). The institutional
    # estimator is the Grinold-Kahn IC t-stat: compute each eval_date's cross-sectional
    # Spearman IC, then test the mean across dates with each date as ONE observation
    # (``n_eval_dates`` = the honest effective N). Emitted additively so the grader can
    # read significance off ``{layer}_date_ic_p`` instead of the inflated pooled p.
    # A layer with < 3 usable dates is under-powered → ``None`` (let it accumulate).
    if "eval_date" in df.columns:
        import numpy as np
        from scipy.stats import ttest_1samp

        out["n_eval_dates"] = int(df["eval_date"].nunique())
        for layer in ("combined_score", "macro_shift", "final_score", "cio_conviction"):
            if layer not in df.columns:
                continue
            per_date_ics: list[float] = []
            for _, sub in df.dropna(subset=[layer]).groupby("eval_date"):
                if sub[layer].nunique() >= 3 and sub["alpha21"].nunique() >= 3:
                    rho, _p = spearmanr(sub[layer], sub["alpha21"])
                    if rho == rho:  # not NaN
                        per_date_ics.append(float(rho))
            if len(per_date_ics) >= 3:
                t_stat, p_val = ttest_1samp(per_date_ics, 0.0)
                out[f"{layer}_date_ic"] = round(float(np.mean(per_date_ics)), 4)
                out[f"{layer}_date_ic_p"] = round(float(p_val), 4) if p_val == p_val else None
                out[f"{layer}_date_ic_n"] = len(per_date_ics)
            else:
                out[f"{layer}_date_ic"] = None
                out[f"{layer}_date_ic_p"] = None
                out[f"{layer}_date_ic_n"] = len(per_date_ics)

    # De-blending substrate (L4563): does CROSS-SECTIONALLY rank-normalizing the
    # stock-quality score within each cycle recover forward signal the raw score
    # lacks? (The apples-to-apples hypothesis — a 70 from Tech ≠ a 70 from
    # Defensives.) Rank combined_score within each eval_date (pct), then IC vs
    # 21d alpha — tracked every cycle so the Phase-3 de-blended cutover has a
    # measured before/after, not a hunch.
    if "combined_score" in df.columns and "eval_date" in df.columns:
        cs = df.dropna(subset=["combined_score"]).copy()
        if cs.shape[0] >= 5:
            cs["cs_xs_rank"] = cs.groupby("eval_date")["combined_score"].rank(pct=True)
            if cs["cs_xs_rank"].nunique() >= 2 and cs["alpha21"].nunique() >= 2:
                rho, p = spearmanr(cs["cs_xs_rank"], cs["alpha21"])
                out["combined_score_xs_rank_ic"] = round(float(rho), 4) if rho == rho else None
                out["combined_score_xs_rank_ic_p"] = round(float(p), 4) if p == p else None

    # Sector-neutral stock quality (L4564): strip the rubric's persistent
    # per-sector bias via a LEAK-FREE trailing-sector z-score (μ,σ from
    # strictly prior eval_dates), with a pool-wide-rank fallback for cold-start
    # sectors. This is the de-blended quality signal the CIO will rank on once
    # the Phase-B flag flips — measured here FIRST so the cutover has a real
    # before/after, not a hunch. The 3-way A/B is: raw ``combined_score_ic``
    # (no normalization) vs ``combined_score_xs_rank_ic`` (pool-wide rank —
    # removes scale only, NOT the sector-mean bias) vs this (sector-neutral).
    if {"combined_score", "eval_date", "sector"}.issubset(df.columns):
        sn = df.dropna(subset=["combined_score"]).copy()
        if sn.shape[0] >= 5:
            q_neutral, frac = _trailing_sector_neutral(sn)
            sn = sn.assign(q_neutral=q_neutral).dropna(subset=["q_neutral"])
            if (sn.shape[0] >= 5 and sn["q_neutral"].nunique() >= 2
                    and sn["alpha21"].nunique() >= 2):
                rho, p = spearmanr(sn["q_neutral"], sn["alpha21"])
                out["combined_score_sector_neutral_ic"] = round(float(rho), 4) if rho == rho else None
                out["combined_score_sector_neutral_ic_p"] = round(float(p), 4) if p == p else None
                out["combined_score_sector_neutral_n"] = int(sn.shape[0])
                out["combined_score_sector_neutral_frac"] = round(float(frac), 3)
    return out


def compute_lift_metrics(
    research_db_path: str,
    trades_db_path: str | None = None,
    eval_date: str | None = None,
    factor_loadings: dict | None = None,
    pillar_profiles: dict | None = None,
    trajectory_scores: dict | None = None,
    bucket: str = "alpha-engine-research",
) -> dict:
    """
    Compute lift at each decision boundary for the given eval_date(s).

    Args:
        research_db_path: path to research.db (universe_returns, scanner_evaluations,
                          team_candidates, cio_evaluations, predictor_outcomes)
        trades_db_path: path to trades.db (trades, executor_shadow_book).
                        Optional — executor lift is skipped if not available.
        eval_date: optional filter. If None, computes across all available dates.
        bucket: signals bucket — source of the research-free backfill parquet
                hydrated for the 3d scanner→predictor counterfactual.

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

        # 1a2. Scanner multi-factor counterfactual (config#967) — would a
        # multi-factor (or single-sleeve) candidate generation beat the
        # momentum-only scanner? Fail-soft; needs injected ArcticDB loadings.
        try:
            result["scanner_factor_counterfactual"] = _scanner_factor_counterfactual(
                conn, factor_loadings or None, pillar_profiles or None
            )
        except Exception as _sfc:  # pragma: no cover - defensive
            logger.warning("scanner_factor_counterfactual failed (non-fatal): %s", _sfc)
            result["scanner_factor_counterfactual"] = {"status": "error", "reason": str(_sfc)}

        # 1b. Breadth-conditioned momentum IC (config#1140) — fail-soft so a
        # producer error can never break the existing e2e_lift contract.
        try:
            result["momentum_regime_ic"] = _momentum_regime_ic(conn, ur, date_filter, params)
        except Exception as _mre:  # pragma: no cover - defensive
            logger.warning("momentum_regime_ic failed (non-fatal): %s", _mre)
            result["momentum_regime_ic"] = {"status": "error", "reason": str(_mre)}

        # 1c. Historical Barra-neutralized composite counterfactual (config#1142)
        # — the raw->neutralized before/after on history, so the gated LIVE
        # cutover can be decided on robust-metric selection now instead of
        # waiting for 4 forward OBSERVE cohorts. Fail-soft like 1b.
        try:
            result["neutralized_composite_ic"] = _neutralized_composite_ic(
                conn, factor_loadings or {}
            )
        except Exception as _nci:  # pragma: no cover - defensive
            logger.warning("neutralized_composite_ic failed (non-fatal): %s", _nci)
            result["neutralized_composite_ic"] = {"status": "error", "reason": str(_nci)}

        # 1d. GRADED LIVE forward efficacy from the PERSISTED neutralized score
        # (config#1187). Unlike 1c (which re-derives neutralization from history
        # via _xs_neutralize), this reads the dual field the live cutover now
        # persists to cio_evaluations and joins THE ACTUAL live ranking score to
        # realized 21d alpha — the true forward measurement. Honest skip until
        # the research.db migration + a live cutover cohort land. Fail-soft.
        try:
            result["neutralization_live_forward_ic"] = _neutralized_live_forward_ic(conn)
        except Exception as _nlf:  # pragma: no cover - defensive
            logger.warning("neutralization_live_forward_ic failed (non-fatal): %s", _nlf)
            result["neutralization_live_forward_ic"] = {"status": "error", "reason": str(_nlf)}

        # 1e. OBSERVE-mode rolling forward-IC of the attractiveness-trajectory
        # signal (crucible-research #337 / config#1392). Joins the persisted
        # weekly trajectory artifact scores (pre_repricing_score, attr_slope_z)
        # to realized 21d alpha and reports the rolling weekly Spearman rank-IC +
        # n_cohorts, so the observe->cutover gate (the console's
        # ``provisional_ic: accruing``) is decidable. Pure measurement: does NOT
        # auto-promote. Fail-soft like 1b-1d.
        try:
            result["trajectory_forward_ic"] = _trajectory_forward_ic(
                conn, trajectory_scores or None
            )
        except Exception as _tfi:  # pragma: no cover - defensive
            logger.warning("trajectory_forward_ic failed (non-fatal): %s", _tfi)
            result["trajectory_forward_ic"] = {"status": "error", "reason": str(_tfi)}

        # 2 + 3. Team lift / CIO lift — RETIRED (config#1580 / config-I2993).
        # The six-team+CIO orchestration no longer produces; re-aggregating its
        # frozen history at full weight every cycle was a live-metric defect
        # (see RESEARCH_GRAPH_RETIRED_DATE). Emit explicit retired markers so the
        # evaluator's retired/N-A path fires instead of scoring a dead graph.
        # ``team_lift`` stays a LIST ([]), never a status dict — downstream
        # consumers (evaluate.py sleeve simulator, analysis/grading, team_skill_
        # metrics) iterate it and a dict crashes them (regression 2026-04-11).
        # The retired_date + component list ride on the additive
        # ``research_graph_retired`` marker below (S3 contract: additive only).
        result["research_graph_retired"] = {
            "retired_date": RESEARCH_GRAPH_RETIRED_DATE,
            "reason": (
                "six-team + macro-economist + CIO research orchestration retired "
                "(config#1580); last producing cycle 2026-07-11, research.db "
                "team_candidates/cio_evaluations rows end 2026-07-10"
            ),
            "components": [
                "team_lift", "cio_lift", "selection_skill_21d", "layer_attribution_21d",
            ],
            "superseded_by": (
                "live_arm_score_ic.thinktank_coverage (challenger) + "
                "attractiveness_eval.json:composite_ic (scanner_attractiveness, "
                "config-I2994)"
            ),
        }
        result["team_lift"] = []  # retired — list contract preserved
        result["cio_lift"] = {
            "status": "retired",
            "retired_date": RESEARCH_GRAPH_RETIRED_DATE,
            "note": (
                "six-team+CIO graph retired (config#1580 / config-I2993); "
                "selection_skill_21d + layer_attribution_21d retired with it"
            ),
        }

        # 3a. Live-arm score-IC (config-I2994) — the score-IC of the scores the
        # LIVE architecture actually emits, replacing the retired CIO composite-
        # IC. Two labeled arms:
        #   * scanner_attractiveness — the champion feed's ranking score. Its
        #     universe-board attractiveness_score→21d-alpha date-clustered IC is
        #     ALREADY produced canonically by attractiveness_eval.py
        #     (``composite_ic``, config#1389) and graded by the evaluator's
        #     ``attractiveness_ic`` component. NOT recomputed here — a parallel
        #     copy is the exact duplicate/divergence defect this issue fixes and
        #     config-I2994's closes-when forbids ("no duplicate attractiveness-IC
        #     metric exists"). Emit a self-documenting delegated pointer only.
        #   * thinktank_coverage — the observe-only challenger shadow scores (the
        #     only research-authored composite score left in the system). Computed
        #     here with the SAME date-clustered estimator attractiveness_eval uses
        #     (_clustered_ic_block), so the two live arms are methodologically
        #     comparable. Fail-soft so a shadow-read error can never break the
        #     e2e_lift contract (observability producer; recorded as status=error).
        try:
            _tt_arm = _thinktank_shadow_ic(conn, bucket)
        except Exception as _tt:  # pragma: no cover - defensive
            logger.warning("thinktank_shadow_ic failed (non-fatal): %s", _tt)
            _tt_arm = {"status": "error", "reason": str(_tt), "arm": THINKTANK_SHADOW_ARM}
        result["live_arm_score_ic"] = {
            "scanner_attractiveness": {
                "status": "delegated",
                "arm": SCANNER_ATTRACTIVENESS_ARM,
                "source_artifact": "backtest/{date}/attractiveness_eval.json",
                "source_block": "composite_ic",
                "note": (
                    "Canonical scanner universe-board attractiveness_score→21d-alpha "
                    "date-clustered IC is produced by attractiveness_eval.py "
                    "(config#1389) and graded by the evaluator's attractiveness_ic "
                    "component. Not recomputed here to avoid a duplicate/divergent "
                    "metric (config-I2994 dedupe / config-I2993 defect class)."
                ),
            },
            "thinktank_coverage": _tt_arm,
        }

        # 3b. CIO vs score-ranking baseline (2e)
        result["cio_vs_ranking"] = _cio_vs_ranking_lift(conn, ur, date_filter, params)

        # 3c. CIO consolidation counterfactual (config#967/#968): does a
        # deterministic top-N selection beat the LLM CIO ADVANCE gate? Fail-soft.
        try:
            result["cio_consolidation_counterfactual"] = _cio_consolidation_counterfactual(
                conn, ur, factor_loadings or None
            )
        except Exception as _ccf:  # pragma: no cover - defensive
            logger.warning("cio_consolidation_counterfactual failed (non-fatal): %s", _ccf)
            result["cio_consolidation_counterfactual"] = {"status": "error", "reason": str(_ccf)}

        # 3d. Scanner -> research-free predictor direct (arm 4, config#1405): does
        # ranking the scanner-passing pool by a research-free predicted_alpha beat
        # the live agentic CIO path? The backfill runs on the PredictorBacktest
        # box against ITS OWN throwaway research.db pull — its rows only reach
        # this box through the canonical S3 parquet, hydrated here before the
        # read (materialize_from_s3 docstring has the full seam explanation).
        # Fail-soft so it can never break the existing e2e_lift contract.
        try:
            from analysis.scanner_predictor_research_free_backfill import (
                materialize_from_s3 as _rf_materialize,
            )

            _rf_materialize(conn, bucket=bucket)
            result["scanner_then_predictor_counterfactual"] = _scanner_then_predictor_topN(conn)
        except Exception as _stp:  # pragma: no cover - defensive
            logger.warning("scanner_then_predictor_topN failed (non-fatal): %s", _stp)
            result["scanner_then_predictor_counterfactual"] = {"status": "error", "reason": str(_stp)}

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
            ur.return_21d,
            ur.spy_return_5d,
            ur.spy_return_21d,
            ur.beat_spy_5d,
            ur.beat_spy_21d,
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
    if cl and cl.get("status") not in ("skipped", "retired"):
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

        # Canonical 21d horizon (L4551): classification on beat_spy_21d + the
        # log-domain alpha lift, additive alongside the 5d diagnostic above.
        sel_mask = merged["quant_filter_pass"] == 1
        clf_21d = _classification_for(merged, sel_mask, "beat_spy_21d")
        lift_21d = _alpha_21d_log_lift(merged, sel_mask)

        return {
            "universe_avg": round(universe_avg, 4),
            "passing_avg": round(passing_avg, 4) if passing_avg is not None else None,
            "lift": round(lift, 4) if lift is not None else None,
            "n_universe": len(merged),
            "n_passing": len(passing),
            "classification": clf,
            "classification_21d": clf_21d,
            "lift_21d_log": lift_21d,
            "arm": SCANNER_METRIC_ARM,
        }
    except sqlite3.OperationalError:
        return {"status": "skipped", "reason": "scanner_evaluations table not found"}


def _momentum_regime_ic(conn, ur: pd.DataFrame, date_filter: str, params: list) -> dict:
    """Breadth-conditioned momentum IC (config#1140).

    Tracks the decisive regime-dependence behind the negative research edge
    (config#1060): short-horizon momentum (scanner ``tech_score``) skill flips
    sign with universe breadth (confirmed 2026-06-18 — tech_score IC -0.115 in
    low-breadth weeks vs +0.030 in high-breadth; corr(breadth, IC)=+0.58). We
    compute the per-week cross-sectional Spearman rank-IC of tech_score vs
    realized 21d log market-relative alpha, then stratify the weekly ICs by
    weekly breadth (fraction of the cohort beating SPY over 21d) into low/high
    halves and report the breadth<->IC correlation. Per-week-then-average (not
    pooled) so cross-week temporal structure cannot masquerade as
    cross-sectional skill. This is the validation target for the Phase-2
    momentum-neutralization fix (config#1142).
    """
    try:
        # Column-existence check up front so a legacy scanner_evaluations schema
        # (no tech_score) SKIPs cleanly rather than surfacing a DB error.
        se_cols = [r[1] for r in conn.execute("PRAGMA table_info(scanner_evaluations)")]
        if not se_cols:
            return {"status": "skipped", "reason": "scanner_evaluations table not found"}
        if "tech_score" not in se_cols:
            return {"status": "skipped", "reason": "scanner_evaluations has no tech_score column"}
        se_filter = date_filter.replace("eval_date", "se.eval_date") if date_filter else ""
        se = pd.read_sql_query(
            f"SELECT ticker, eval_date, tech_score FROM scanner_evaluations se{se_filter}",
            conn, params=params,
        )
        if se.empty:
            return {"status": "skipped", "reason": "scanner_evaluations empty"}
        if "log_return_21d" not in ur.columns or "log_spy_return_21d" not in ur.columns:
            return {"status": "skipped", "reason": "universe_returns lacks log_return_21d"}
        m = ur.merge(se, on=["ticker", "eval_date"], how="inner")
        m = m[
            m["log_return_21d"].notna()
            & m["log_spy_return_21d"].notna()
            & m["tech_score"].notna()
        ].copy()
        if m.empty:
            return {"status": "insufficient_data", "reason": "no realized-21d rows with tech_score"}
        m["log_alpha_21d"] = m["log_return_21d"] - m["log_spy_return_21d"]
        weeks = []
        for d, g in m.groupby("eval_date"):
            if len(g) < 10:  # need enough names for a stable weekly cross-sectional IC
                continue
            ic = g["tech_score"].corr(g["log_alpha_21d"], method="spearman")
            if ic != ic:  # NaN (e.g. constant tech_score within the week)
                continue
            breadth = float((g["log_return_21d"] > g["log_spy_return_21d"]).mean())
            weeks.append({"breadth": breadth, "ic": float(ic), "n": int(len(g))})
        n_weeks = len(weeks)
        if n_weeks < 4:
            return {
                "status": "insufficient_data",
                "reason": f"only {n_weeks} weekly cohorts with realized 21d outcomes",
                "n_weeks": n_weeks,
            }
        wdf = pd.DataFrame(weeks)
        med = float(wdf["breadth"].median())
        low = wdf[wdf["breadth"] <= med]["ic"]
        high = wdf[wdf["breadth"] > med]["ic"]
        bic = wdf["breadth"].corr(wdf["ic"])
        return {
            "status": "ok",
            "horizon": "21d",
            "n_weeks": n_weeks,
            "mean_weekly_ic": round(float(wdf["ic"].mean()), 4),
            "low_breadth_ic": round(float(low.mean()), 4) if len(low) else None,
            "high_breadth_ic": round(float(high.mean()), 4) if len(high) else None,
            "breadth_ic_corr": round(float(bic), 4) if bic == bic else None,
            "median_breadth": round(med, 4),
            "n_low_weeks": int(len(low)),
            "n_high_weeks": int(len(high)),
        }
    except sqlite3.OperationalError:
        return {"status": "skipped", "reason": "scanner_evaluations table not found"}
    except Exception as e:  # pragma: no cover - defensive; never break e2e_lift
        return {"status": "error", "reason": str(e)}


# The Barra factor set the composite is residualized against (config#1142) —
# the same momentum/beta/size factors the live OBSERVE shadow uses
# (research/scoring/neutralization_shadow.py). The config#1060 diagnosis pinned
# the funnel as an unintended short-momentum bet that inverts in narrow-breadth
# tape.
#
# We residualize against the RAW ArcticDB feature columns, NOT the cross-
# sectional ``*_zscore`` loadings: the z-scores are computed at feature-store
# snapshot time and are NOT persisted per-name in the ArcticDB universe library
# (the only historical source); the dated S3 ``factor_loading.parquet`` snapshots
# only exist from 2026-06-05 on — too recent to overlap the realized-21d
# cohorts — whereas the raw inputs go back to 2016. This is EXACT, not an
# approximation: OLS residualization is invariant to affine transforms of the
# regressors, and ``_xs_neutralize`` standardizes each factor internally, so
# residualizing on raw ``momentum_20d`` is identical to residualizing on
# ``momentum_20d_zscore``. size is the only non-affine case (Barra SIZE =
# z(log(mktcap))), so the loader emits ``size_log = log(market_cap_raw)`` to
# match the production definition.
DEFAULT_NEUTRALIZE_FACTORS: tuple[str, ...] = (
    "momentum_20d",
    "return_60d",
    "beta_60d",
    "size_log",
)

# The date the score-neutralization went LIVE (config#1142 — research
# scoring.yaml ``aggregator.neutralization.live_enabled`` flipped true,
# private-first). Weeks on/after this date are the LIVE-era cohorts: because the
# backtester reconstructs the neutralized score with the SAME cross-sectional
# residualizer the live path uses (``_xs_neutralize`` mirrors research
# scoring/neutralize.py), the post-cutover segment of ``_neutralized_composite_ic``
# IS the live neutralization's realized forward efficacy — the measurement the
# report card otherwise lacks (config#1187; the live cutover only rewrites
# signals.json score, which no IC producer reads back).
NEUTRALIZATION_LIVE_CUTOVER_DATE: str = "2026-06-22"

# Observe-mode forward-IC gate for the attractiveness-trajectory signal
# (crucible-research #337 / config#1392). The signal is written weekly to the
# trajectory artifact but its forward predictive power has never been measured.
# ``_trajectory_forward_ic`` joins the persisted per-name ``pre_repricing_score``
# and ``attr_slope_z`` to realized 21d log market-relative alpha and reports the
# rolling weekly Spearman rank-IC. Fewer than this many mature weekly cohorts ->
# status ``accruing`` (honest None IC, NOT a crash): the observe->cutover gate is
# not yet decidable. This is the SAME maturity floor the live console header
# (``provisional_ic: accruing``) is waiting on. Crossing it does NOT auto-promote
# the signal — the observe->cutover decision is the operator's; this producer only
# computes + surfaces the IC and the n_cohorts so the gate is decidable.
TRAJECTORY_FORWARD_IC_MIN_COHORTS: int = 4
# Minimum names per weekly cross-section for a stable per-week IC (matches the
# >=10 floor _neutralized_live_forward_ic uses for its weekly Spearman).
TRAJECTORY_FORWARD_IC_MIN_NAMES_PER_WEEK: int = 10

# ArcticDB raw column -> (exposure key, transform). The loader reads these raw
# columns from the universe feature history and emits the exposure keys above.
_RAW_FACTOR_SOURCE: dict = {
    "momentum_20d": ("momentum_20d", None),
    "return_60d": ("return_60d", None),
    "beta_60d": ("beta_60d", None),
    "market_cap_raw": ("size_log", "log"),
    # Additional raw factors for the scanner multi-factor counterfactual
    # (config#967) — value / quality / low-vol sleeves. Passed through raw; the
    # scanner producer z-scores them cross-sectionally per cycle.
    "pe_ratio": ("pe_ratio", None),
    "pb_ratio": ("pb_ratio", None),
    "roe": ("roe", None),
    "fcf_yield": ("fcf_yield", None),
    "realized_vol_63d": ("realized_vol_63d", None),
    "idio_vol_60d": ("idio_vol_60d", None),
}

# Every exposure key the loader can emit — used by callers that want the full
# superset (the neutralization producer needs 4 of these; the scanner
# counterfactual needs 8; a single load serves both).
ALL_LOADING_FACTORS: tuple = tuple(key for key, _ in _RAW_FACTOR_SOURCE.values())


def _xs_neutralize(
    scores: dict,
    exposures: dict,
    factors: list,
    *,
    min_names: int = 20,
    rescale: bool = True,
) -> dict:
    """Cross-sectional factor-residualizer — the backtester's counterfactual
    mirror of research/scoring/neutralize.py::neutralize_scores (config#1142).

    Regress the score cross-section on the standardized factor exposures and
    return the residual (the idiosyncratic component after removing the
    intended-neutral momentum/beta/size tilt). Identity passthrough on every
    degenerate case (no factors / too-few names / constant or missing exposures
    / rank-deficient design / NaN) — fail-soft, a name is never dropped.

    Kept INLINE rather than imported from the research repo to avoid
    contract-bypassing cross-repo coupling (the M0 slot-contract discipline) —
    mirrors the existing inline ``_sector_neutral`` transform. Now that there
    are two adopters (research live shadow + this backtester counterfactual),
    lifting the generic residualizer into nousergon-lib is a P3 follow-up.
    """
    import numpy as np

    out = {t: s for t, s in scores.items()}  # identity baseline (never lose a name)
    if not factors:
        return out
    try:
        fit_t: list = []
        rows: list = []
        y: list = []
        for t, s in scores.items():
            if s is None or not np.isfinite(s):
                continue
            ex = exposures.get(t) or {}
            vals = [ex.get(f) for f in factors]
            if any(v is None or not np.isfinite(v) for v in vals):
                continue
            fit_t.append(t)
            rows.append([float(v) for v in vals])
            y.append(float(s))
        if len(fit_t) < min_names:
            return out
        X = np.asarray(rows, dtype=float)
        yv = np.asarray(y, dtype=float)
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        keep = sd > 1e-12
        if not keep.any():
            return out
        Xz = (X[:, keep] - mu[keep]) / sd[keep]
        A = np.column_stack([np.ones(len(yv)), Xz])
        coef, _res, rank, _sv = np.linalg.lstsq(A, yv, rcond=None)
        if rank < A.shape[1]:
            return out
        resid = yv - A @ coef
        if rescale:
            r_sd = resid.std()
            if r_sd > 1e-12:
                resid = (resid - resid.mean()) / r_sd * yv.std() + yv.mean()
            else:
                resid = resid - resid.mean() + yv.mean()
        for t, r in zip(fit_t, resid):
            out[t] = float(r)
        return out
    except Exception:  # fail-soft — the counterfactual must never break e2e_lift
        return {t: s for t, s in scores.items()}


def _neutralized_composite_ic(
    conn,
    loadings: dict,
    factors: tuple = DEFAULT_NEUTRALIZE_FACTORS,
) -> dict:
    """Historical Barra-neutralized composite counterfactual (config#1142/#1060).

    Answers the cutover-gate question directly on HISTORY instead of waiting for
    4 forward OBSERVE cohorts: does residualizing the research **composite**
    (``team_candidates.quant_score`` — the wide team-stage quant composite)
    against the momentum/beta/size factor exposures RECOVER forward 21d-alpha
    skill? Computes the per-week cross-sectional Spearman rank-IC of BOTH the raw
    and the neutralized composite vs realized 21d log market-relative alpha,
    stratified by weekly breadth (same construction as ``_momentum_regime_ic``),
    so the harness sees a measured raw->neutralized before/after.

    ``team_candidates`` is the right cross-section: ~48 names/week — above the
    20-name floor ``_xs_neutralize`` needs for a stable cross-sectional
    regression. The thinner ~17-name ``score_performance`` / ``cio_evaluations``
    pools fall below it most weeks, so neutralization passes through identity and
    never engages there (the 2026-06-22 finding behind this repoint). Realized
    21d alpha comes from the ``universe_returns`` join.

    This only MEASURES the counterfactual; the neutralized LIVE cutover stays
    gated/OFF (config#1142). ``loadings``: {(eval_date, ticker): {factor:
    exposure}} — injected by the caller from ArcticDB feature history (see
    :func:`load_historical_factor_loadings`); empty -> status ``skipped``.
    """
    try:
        if not loadings:
            return {"status": "skipped", "reason": "no factor loadings provided"}
        tabs = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "team_candidates" not in tabs or "universe_returns" not in tabs:
            return {"status": "skipped", "reason": "team_candidates / universe_returns table not found"}
        tc_cols = {r[1] for r in conn.execute("PRAGMA table_info(team_candidates)")}
        if "quant_score" not in tc_cols:
            return {"status": "skipped", "reason": "team_candidates has no quant_score column"}
        ur_cols = {r[1] for r in conn.execute("PRAGMA table_info(universe_returns)")}
        if not {"log_return_21d", "log_spy_return_21d"}.issubset(ur_cols):
            return {"status": "skipped", "reason": "universe_returns lacks log_return_21d / log_spy_return_21d"}
        # team_candidates.quant_score is the WIDE composite cross-section
        # (~48 names/week — clears the 20-name floor _xs_neutralize needs to
        # engage; the ~17-name score_performance / CIO pools do not). Join
        # universe_returns for the realized 21d log market-relative alpha.
        m = pd.read_sql_query(
            "SELECT t.ticker AS ticker, t.eval_date AS eval_date, "
            "t.quant_score AS score, "
            "u.log_return_21d AS r21, u.log_spy_return_21d AS s21, "
            "(u.log_return_21d - u.log_spy_return_21d) AS log_alpha_21d "
            "FROM team_candidates t JOIN universe_returns u "
            "ON u.ticker = t.ticker AND u.eval_date = t.eval_date "
            "WHERE u.log_return_21d IS NOT NULL AND u.log_spy_return_21d IS NOT NULL "
            "AND t.quant_score IS NOT NULL",
            conn,
        )
        if m.empty:
            return {"status": "insufficient_data", "reason": "no team_candidates rows with realized 21d outcomes"}
        factors_l = list(factors)
        raw_weeks: list = []
        neu_weeks: list = []
        n_neutralized_weeks = 0
        cov_names = 0
        tot_names = 0
        for d, g in m.groupby("eval_date"):
            if len(g) < 10:  # need enough names for a stable weekly cross-sectional IC
                continue
            scores = dict(zip(g["ticker"], g["score"].astype(float)))
            exposures = {t: (loadings.get((d, t)) or {}) for t in scores}
            tot_names += len(scores)
            cov_names += sum(
                1 for t in scores if all(f in exposures[t] for f in factors_l)
            )
            neu = _xs_neutralize(scores, exposures, factors_l)
            if any(abs(neu[t] - scores[t]) > 1e-9 for t in scores):
                n_neutralized_weeks += 1
            alpha = dict(zip(g["ticker"], g["log_alpha_21d"].astype(float)))
            tks = list(scores)
            raw_ic = pd.Series([scores[t] for t in tks]).corr(
                pd.Series([alpha[t] for t in tks]), method="spearman"
            )
            neu_ic = pd.Series([neu[t] for t in tks]).corr(
                pd.Series([alpha[t] for t in tks]), method="spearman"
            )
            if raw_ic != raw_ic or neu_ic != neu_ic:  # NaN (constant within week)
                continue
            breadth = float((g["r21"] > g["s21"]).mean())
            raw_weeks.append({"eval_date": str(d), "breadth": breadth, "ic": float(raw_ic)})
            neu_weeks.append({"eval_date": str(d), "breadth": breadth, "ic": float(neu_ic)})
        n_weeks = len(raw_weeks)
        if n_weeks < 4:
            return {
                "status": "insufficient_data",
                "reason": f"only {n_weeks} weekly cohorts with realized 21d outcomes",
                "n_weeks": n_weeks,
            }
        rdf = pd.DataFrame(raw_weeks)
        ndf = pd.DataFrame(neu_weeks)
        med = float(rdf["breadth"].median())

        def _split(df: pd.DataFrame):
            low = df[df["breadth"] <= med]["ic"]
            high = df[df["breadth"] > med]["ic"]
            return (
                round(float(df["ic"].mean()), 4),
                round(float(low.mean()), 4) if len(low) else None,
                round(float(high.mean()), 4) if len(high) else None,
            )

        raw_mean, raw_low, raw_high = _split(rdf)
        neu_mean, neu_low, neu_high = _split(ndf)

        def _delta(a, b):
            return round(a - b, 4) if (a is not None and b is not None) else None

        # The neutralization "recovers edge" only if it lifts the overall weekly
        # IC to non-negative AND does not make the low-breadth bucket worse —
        # the bucket where the un-neutralized momentum bet does its damage.
        recovers = bool(
            neu_mean is not None
            and neu_mean >= 0
            and (raw_mean is None or neu_mean > raw_mean)
            and (neu_low is None or raw_low is None or neu_low >= raw_low)
        )

        # ── LIVE forward efficacy (config#1187) ──────────────────────────────
        # The block above is the HISTORICAL counterfactual (all weeks). This
        # segments the SAME per-week raw/neutralized IC series at the live
        # cutover date so the report card can see whether the neutralization is
        # actually recovering edge on LIVE post-cutover cohorts — the question
        # neither research_composite_ic (reads raw research.db scores) nor the
        # OBSERVE shadow (no outcome join) answers. Because _xs_neutralize mirrors
        # the live residualizer, the post-cutover neutralized IC equals what the
        # live system realized. Each week is ONE observation: the per-week paired
        # delta (neu_ic − raw_ic) gets a Grinold-Kahn one-sample t-test
        # (config#1164), n_weeks = effective N. Under-powered (<3 wks) → p=None,
        # WATCH/accumulating downstream.
        from scipy.stats import ttest_1samp as _ttest_1samp

        rdf2 = rdf.assign(neu_ic=ndf["ic"].values, delta=ndf["ic"].values - rdf["ic"].values)

        def _segment(seg: pd.DataFrame) -> dict:
            n = int(len(seg))
            out: dict = {
                "n_weeks": n,
                "raw_mean_weekly_ic": round(float(seg["ic"].mean()), 4) if n else None,
                "neutralized_mean_weekly_ic": round(float(seg["neu_ic"].mean()), 4) if n else None,
                "mean_weekly_delta": round(float(seg["delta"].mean()), 4) if n else None,
                "delta_t_p": None,
                "recovers_edge_live": False,
                "significant": False,
            }
            if n >= 3 and seg["delta"].nunique() >= 2:
                _t, _p = _ttest_1samp(seg["delta"], 0.0)
                out["delta_t_p"] = round(float(_p), 4) if _p == _p else None
            out["recovers_edge_live"] = bool(out["mean_weekly_delta"] is not None and out["mean_weekly_delta"] > 0)
            out["significant"] = bool(
                n >= 4 and out["delta_t_p"] is not None and out["delta_t_p"] < 0.05
            )
            return out

        cutover = NEUTRALIZATION_LIVE_CUTOVER_DATE
        live_forward = _segment(rdf2[rdf2["eval_date"] >= cutover])
        pre_cutover = _segment(rdf2[rdf2["eval_date"] < cutover])

        return {
            "status": "ok",
            "horizon": "21d",
            "n_weeks": n_weeks,
            "factors": factors_l,
            "factor_coverage_frac": round(cov_names / tot_names, 3) if tot_names else 0.0,
            "n_neutralized_weeks": n_neutralized_weeks,
            "median_breadth": round(med, 4),
            "raw_mean_weekly_ic": raw_mean,
            "neutralized_mean_weekly_ic": neu_mean,
            "raw_low_breadth_ic": raw_low,
            "neutralized_low_breadth_ic": neu_low,
            "raw_high_breadth_ic": raw_high,
            "neutralized_high_breadth_ic": neu_high,
            "ic_improvement": _delta(neu_mean, raw_mean),
            "low_breadth_ic_improvement": _delta(neu_low, raw_low),
            "neutralization_recovers_edge": recovers,
            "cutover_date": cutover,
            "live_forward": live_forward,
            "pre_cutover": pre_cutover,
        }
    except sqlite3.OperationalError:
        return {"status": "skipped", "reason": "team_candidates / universe_returns table not found"}
    except Exception as e:  # pragma: no cover - defensive; never break e2e_lift
        return {"status": "error", "reason": str(e)}


def _neutralized_live_forward_ic(conn) -> dict:
    """Graded LIVE forward efficacy of the #1142 neutralization (config#1187).

    The companion ``_neutralized_composite_ic.live_forward`` block measures a
    RE-DERIVED counterfactual: it residualizes ``team_candidates.quant_score``
    with the backtester's ``_xs_neutralize`` (which only MIRRORS the live
    residualizer) and segments at the cutover. It never reads the score the live
    system ACTUALLY ranked on — because, before config#1187, that score was
    written only to ``signals.json`` and never persisted to ``research.db``.

    config#1187 fixes that at the source: ``cio_evaluations`` now persists BOTH
    the raw composite (``final_score``) AND the live neutralized ranking score
    (``neutralized_final_score``) as a DUAL field, populated at the exact point
    the live cutover rewrites the signal. This producer is the consumer: it
    joins the PERSISTED neutralized score to realized 21d log market-relative
    alpha (``universe_returns``) and computes the per-week cross-sectional
    Spearman rank-IC for BOTH the raw and the LIVE neutralized score, then the
    per-week paired delta (neutralized − raw) with a date-clustered
    Grinold-Kahn one-sample t-test (config#1164). Each week is ONE observation;
    ``n_weeks`` = effective N.

    Unlike the re-derived counterfactual, this is the ACTUAL live ranking score's
    realized forward efficacy — the measurement the issue's 'closes when'
    requires. Rows where ``neutralized_final_score`` is NULL (the live gate was
    OFF, or the name had no neutralized score) are identity: neutralized == raw,
    so a pre-cutover / gate-off week shows a zero delta rather than spurious lift.

    Fail-soft: any precondition miss / error -> status skipped/insufficient_data/
    error, never breaking the e2e_lift contract. Returns ``live_forward`` (rows
    with a NON-NULL persisted neutralized score, i.e. the live cutover cohorts)
    and ``all_weeks`` (every week with realized outcomes, for context).
    """
    try:
        tabs = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "cio_evaluations" not in tabs or "universe_returns" not in tabs:
            return {"status": "skipped", "reason": "cio_evaluations / universe_returns absent"}
        ce_cols = {r[1] for r in conn.execute("PRAGMA table_info(cio_evaluations)")}
        if "neutralized_final_score" not in ce_cols:
            # research.db predates the config#1187 migration — the persisted
            # dual field does not exist yet. Honest skip (NOT an error): the
            # measurement only becomes possible once the migrated writer has run.
            return {
                "status": "skipped",
                "reason": "cio_evaluations.neutralized_final_score not present "
                          "(research.db pre-config#1187 migration)",
            }
        if "final_score" not in ce_cols:
            return {"status": "skipped", "reason": "cio_evaluations missing final_score"}
        ur_cols = {r[1] for r in conn.execute("PRAGMA table_info(universe_returns)")}
        if not {"log_return_21d", "log_spy_return_21d"}.issubset(ur_cols):
            return {"status": "skipped", "reason": "universe_returns lacks log_return_21d / log_spy_return_21d"}

        m = pd.read_sql_query(
            "SELECT ce.ticker AS ticker, ce.eval_date AS eval_date, "
            "ce.final_score AS raw_score, "
            "ce.neutralized_final_score AS neu_score, "
            "(u.log_return_21d - u.log_spy_return_21d) AS log_alpha_21d "
            "FROM cio_evaluations ce "
            "JOIN universe_returns u ON u.ticker = ce.ticker AND u.eval_date = ce.eval_date "
            "WHERE u.log_return_21d IS NOT NULL AND u.log_spy_return_21d IS NOT NULL "
            "AND ce.final_score IS NOT NULL",
            conn,
        )
        if m.empty:
            return {"status": "insufficient_data", "reason": "no cio_evaluations rows with realized 21d outcomes"}

        # Where the live gate did NOT rewrite this name (NULL persisted
        # neutralized score), the live ranking == the raw composite. Treating it
        # as identity is exactly right: the neutralization had no live effect on
        # that name, so its raw and neutralized forward IC must coincide.
        m["neu_eff"] = m["neu_score"].where(m["neu_score"].notna(), m["raw_score"])
        # A row is a LIVE neutralization cohort member iff a neutralized score
        # was actually persisted for it (gate ON for that run).
        m["is_live"] = m["neu_score"].notna()

        def _weekly_ic(df: pd.DataFrame) -> list:
            weeks = []
            for d, g in df.groupby("eval_date"):
                if len(g) < 10:  # need enough names for a stable weekly IC
                    continue
                raw_ic = g["raw_score"].corr(g["log_alpha_21d"], method="spearman")
                neu_ic = g["neu_eff"].corr(g["log_alpha_21d"], method="spearman")
                if raw_ic != raw_ic or neu_ic != neu_ic:  # NaN (constant within week)
                    continue
                weeks.append({
                    "eval_date": str(d),
                    "raw_ic": float(raw_ic),
                    "neu_ic": float(neu_ic),
                    "delta": float(neu_ic - raw_ic),
                })
            return weeks

        from scipy.stats import ttest_1samp as _ttest_1samp

        def _segment(df: pd.DataFrame) -> dict:
            weeks = _weekly_ic(df)
            n = len(weeks)
            out: dict = {
                "n_weeks": n,
                "raw_mean_weekly_ic": round(
                    float(sum(w["raw_ic"] for w in weeks) / n), 4) if n else None,
                "neutralized_mean_weekly_ic": round(
                    float(sum(w["neu_ic"] for w in weeks) / n), 4) if n else None,
                "mean_weekly_delta": round(
                    float(sum(w["delta"] for w in weeks) / n), 4) if n else None,
                "delta_t_p": None,
                "recovers_edge_live": False,
                "significant": False,
            }
            deltas = [w["delta"] for w in weeks]
            if n >= 3 and len({round(x, 9) for x in deltas}) >= 2:
                _t, _p = _ttest_1samp(deltas, 0.0)
                out["delta_t_p"] = round(float(_p), 4) if _p == _p else None
            out["recovers_edge_live"] = bool(
                out["mean_weekly_delta"] is not None and out["mean_weekly_delta"] > 0
            )
            out["significant"] = bool(
                n >= 4 and out["delta_t_p"] is not None and out["delta_t_p"] < 0.05
            )
            return out

        live_df = m[m["is_live"]]
        live_forward = _segment(live_df)
        all_weeks = _segment(m)

        return {
            "status": "ok",
            "horizon": "21d",
            "source": "persisted cio_evaluations.neutralized_final_score (config#1187)",
            "cutover_date": NEUTRALIZATION_LIVE_CUTOVER_DATE,
            "n_live_rows": int(m["is_live"].sum()),
            "n_total_rows": int(len(m)),
            "live_forward": live_forward,
            "all_weeks": all_weeks,
        }
    except sqlite3.OperationalError:
        return {"status": "skipped", "reason": "cio_evaluations / universe_returns table not found"}
    except Exception as e:  # pragma: no cover - defensive; never break e2e_lift
        return {"status": "error", "reason": str(e)}


def load_historical_factor_loadings(
    bucket: str,
    eval_dates,
    factors: tuple = DEFAULT_NEUTRALIZE_FACTORS,
) -> dict:
    """Build {(eval_date, ticker): {exposure_key: value}} from the RAW ArcticDB
    universe feature history, for the neutralized-composite counterfactual
    (config#1142).

    ArcticDB is the authoritative per-date feature store (the same source the
    backtester uses for prices) and carries the RAW factor inputs back to 2016 —
    ``momentum_20d`` / ``return_60d`` / ``beta_60d`` / ``market_cap_raw`` (see
    ``_RAW_FACTOR_SOURCE``). We deliberately use raw inputs rather than the
    cross-sectional ``*_zscore`` loadings: the z-scores are NOT persisted in the
    universe library, and residualizing on raw vs z-scored exposures is
    mathematically identical (affine invariance + ``_xs_neutralize`` standardizes
    internally). ``market_cap_raw`` gets the ``log`` transform so size matches
    Barra SIZE = z(log(mktcap)).

    A research-cycle ``eval_date`` may not be a trading day, so each is matched
    AS-OF (the last trading row at or before it). Fail-soft: any error (ArcticDB
    unreachable, a column absent, a malformed index) returns ``{}`` so the
    counterfactual reports status ``skipped`` and the e2e_lift contract is never
    broken.
    """
    try:
        import numpy as np

        from store.arctic_reader import load_universe_from_arctic

        # exposure_key -> (raw_column, transform), restricted to the requested
        # factor set so callers can subset.
        wanted_keys = set(factors)
        src = {
            raw: (key, tf)
            for raw, (key, tf) in _RAW_FACTOR_SOURCE.items()
            if key in wanted_keys
        }
        if not src:
            return {}
        raw_cols = list(src)

        _prices, features_by_ticker = load_universe_from_arctic(bucket)
        dates_sorted = sorted(set(eval_dates))
        dt_index = pd.to_datetime(dates_sorted)
        out: dict = {}
        for ticker, fdf in (features_by_ticker or {}).items():
            cols = [c for c in raw_cols if c in getattr(fdf, "columns", [])]
            if not cols:
                continue
            sub = fdf[cols].copy()
            sub.index = pd.to_datetime(sub.index).normalize()
            sub = sub[~sub.index.duplicated(keep="last")].sort_index()
            # AS-OF align to each eval_date: last trading row <= eval_date.
            aligned = sub.reindex(dt_index, method="ffill")
            for d_ts, d_str in zip(dt_index, dates_sorted):
                row = aligned.loc[d_ts]
                rec: dict = {}
                for raw in cols:
                    key, tf = src[raw]
                    v = row.get(raw)
                    if v is None or not pd.notna(v):
                        continue
                    if tf == "log":
                        if v <= 0:
                            continue
                        v = float(np.log(v))
                    rec[key] = float(v)
                if rec:
                    out[(d_str, ticker)] = rec
        return out
    except Exception as e:  # fail-soft — never break the diagnostics run
        logger.warning("load_historical_factor_loadings failed (non-fatal): %s", e)
        return {}


def _cio_consolidation_counterfactual(conn, ur: pd.DataFrame, loadings: dict | None = None) -> dict:
    """Does a DETERMINISTIC top-N consolidation beat the LLM CIO's ADVANCE gate?
    (config#967/#968 — the "better way to consolidate sector-team picks" test.)

    The CIO's job is to select the final entrants from the candidate pool it
    scores (``cio_evaluations``). The funnel measurement (config#967) showed the
    CIO is value-subtractive — its ADVANCE set realizes LOWER 21d alpha than its
    REJECT set, and it doesn't beat a naive ranking. This counterfactual makes
    the alternative explicit: per cycle, hold the entrant COUNT fixed at the
    CIO's own ADVANCE count, and compare the realized 21d log-alpha the CIO's
    ADVANCE picks achieved vs what a deterministic top-N by each candidate
    ranking signal would have achieved from the SAME pool:

      * ``cio_advance``           — the live LLM ADVANCE/ADVANCE_FORCED set (baseline)
      * ``combined_score_topN``   — the CIO's OWN combined_score, top-N (isolates
                                    the LLM advance DECISION from its RANKING)
      * ``quant_score_topN``      — raw team quant score, top-N
      * ``sector_neutral_quant_topN`` — quant score demeaned per (cycle, sector)
      * ``predictor_p_up_topN``   — the ML predictor's p_up, top-N (thin coverage)

    Reports each method's pooled realized 21d alpha (the avg alpha of the names
    it would have held) + sector-neutral alpha + lift vs the CIO. Pure read;
    changes nothing. Fail-soft -> status skipped/error.
    """
    try:
        tabs = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "cio_evaluations" not in tabs or "universe_returns" not in tabs:
            return {"status": "skipped", "reason": "cio_evaluations / universe_returns absent"}
        ce_cols = {r[1] for r in conn.execute("PRAGMA table_info(cio_evaluations)")}
        if not {"cio_decision", "quant_score", "combined_score"}.issubset(ce_cols):
            return {"status": "skipped", "reason": "cio_evaluations missing decision/score columns"}
        has_pred = "predictor_outcomes" in tabs
        pred_join = (
            "LEFT JOIN predictor_outcomes po ON po.symbol=ce.ticker "
            "AND po.prediction_date=ce.eval_date"
            if has_pred else ""
        )
        pred_sel = ", po.p_up AS p_up" if has_pred else ""
        m = pd.read_sql_query(
            f"SELECT ce.ticker, ce.eval_date, ce.cio_decision, ce.quant_score, "
            f"ce.combined_score, u.sector, "
            f"(u.log_return_21d - u.log_spy_return_21d) AS alpha21{pred_sel} "
            f"FROM cio_evaluations ce "
            f"JOIN universe_returns u ON u.ticker=ce.ticker AND u.eval_date=ce.eval_date "
            f"{pred_join} "
            f"WHERE u.log_return_21d IS NOT NULL AND u.log_spy_return_21d IS NOT NULL",
            conn,
        )
        if m.empty:
            return {"status": "insufficient_data", "reason": "no cio_evaluations rows with realized 21d"}

        ADVANCE = {"ADVANCE", "ADVANCE_FORCED"}
        # accumulate per-method realized alpha (pooled across cycles) + sector-neutral
        methods = ("cio_advance", "combined_score_topN", "quant_score_topN",
                   "sector_neutral_quant_topN", "factor_neutral_quant_topN",
                   "predictor_p_up_topN")
        picked: dict = {k: [] for k in methods}       # realized alpha of picked names
        picked_sn: dict = {k: [] for k in methods}    # sector-neutral alpha
        pred_cov_num = pred_cov_den = 0
        n_cycles = 0
        n_adv_total = 0

        for d, g in m.groupby("eval_date"):
            g = g[g["alpha21"].notna()].copy()
            if g.empty:
                continue
            adv_mask = g["cio_decision"].isin(ADVANCE)
            n_adv = int(adv_mask.sum())
            if n_adv == 0:
                continue
            n_cycles += 1
            n_adv_total += n_adv
            # sector-neutral alpha within this cycle (demean by sector)
            if g["sector"].notna().any():
                g["alpha_sn"] = g["alpha21"] - g.groupby("sector")["alpha21"].transform("mean")
            else:
                g["alpha_sn"] = g["alpha21"]

            def _take(order_col, mask=None, ascending=False):
                sub = g if mask is None else g[mask]
                sub = sub[sub[order_col].notna()]
                if sub.empty:
                    return None
                top = sub.sort_values(order_col, ascending=ascending).head(n_adv)
                return top

            # baseline: the actual CIO ADVANCE set
            cio = g[adv_mask]
            sel = {
                "cio_advance": cio,
                "combined_score_topN": _take("combined_score"),
                "quant_score_topN": _take("quant_score"),
            }
            # sector-neutral quant rank
            g["quant_sn"] = g["quant_score"] - g.groupby("sector")["quant_score"].transform("mean")
            sel["sector_neutral_quant_topN"] = _take("quant_sn")
            # factor-neutral (momentum/beta/size) quant rank — the #1142 lever,
            # the only ranking signal with positive forward evidence. Needs the
            # injected Barra loadings; absent -> method skipped (None).
            if loadings:
                qscores = {
                    t: float(s) for t, s in zip(g["ticker"], g["quant_score"])
                    if s is not None and pd.notna(s)
                }
                exposures = {t: (loadings.get((d, t)) or {}) for t in qscores}
                neu = _xs_neutralize(qscores, exposures, list(DEFAULT_NEUTRALIZE_FACTORS))
                g["quant_fn"] = g["ticker"].map(neu)
                sel["factor_neutral_quant_topN"] = (
                    _take("quant_fn") if g["quant_fn"].notna().any() else None
                )
            else:
                sel["factor_neutral_quant_topN"] = None
            # predictor p_up (only where present)
            if has_pred and "p_up" in g.columns and g["p_up"].notna().any():
                sel["predictor_p_up_topN"] = _take("p_up")
                pred_cov_num += int(g["p_up"].notna().sum())
                pred_cov_den += len(g)
            else:
                sel["predictor_p_up_topN"] = None

            for k, sub in sel.items():
                if sub is not None and not sub.empty:
                    picked[k].extend(sub["alpha21"].tolist())
                    picked_sn[k].extend(sub["alpha_sn"].tolist())

        if n_cycles < 3:
            return {"status": "insufficient_data", "reason": f"only {n_cycles} cycles", "n_cycles": n_cycles}

        def _mean(xs):
            return round(float(sum(xs) / len(xs)), 5) if xs else None

        cio_mean = _mean(picked["cio_advance"])
        cio_sn = _mean(picked_sn["cio_advance"])
        out_methods: dict = {}
        best = None
        best_alpha = cio_mean if cio_mean is not None else -1e9
        for k in methods:
            ma = _mean(picked[k])
            sn = _mean(picked_sn[k])
            entry = {
                "mean_alpha_21d": ma,
                "sector_neutral_mean_alpha_21d": sn,
                "n_picks": len(picked[k]),
            }
            if k != "cio_advance" and ma is not None and cio_mean is not None:
                entry["lift_vs_cio"] = round(ma - cio_mean, 5)
                entry["sn_lift_vs_cio"] = (
                    round(sn - cio_sn, 5) if (sn is not None and cio_sn is not None) else None
                )
                if ma > best_alpha:
                    best_alpha, best = ma, k
            out_methods[k] = entry

        return {
            "status": "ok",
            "horizon": "21d",
            "n_cycles": n_cycles,
            "n_advanced_total": n_adv_total,
            "selection_count_basis": "per-cycle top-N matched to the CIO ADVANCE count",
            "predictor_coverage_frac": (
                round(pred_cov_num / pred_cov_den, 3) if pred_cov_den else 0.0
            ),
            "methods": out_methods,
            "best_method": best or "cio_advance",
            "any_deterministic_beats_cio": best is not None,
        }
    except sqlite3.OperationalError as e:
        return {"status": "skipped", "reason": f"sqlite: {e}"}
    except Exception as e:  # pragma: no cover - defensive; never break e2e_lift
        return {"status": "error", "reason": str(e)}


# Multi-factor candidate-generation composite for the scanner counterfactual
# (config#1186). Each sleeve is the mean of its cross-sectionally z-scored,
# sign-oriented (higher = better) raw factors; the composite is the equal-weight
# mean of the sleeves. This is the institutional alternative to the current
# scanner's momentum-only ``tech_score``. Raw factor column names match the
# ArcticDB universe feature set (the loader emits them).
_SCANNER_FACTOR_SLEEVES: dict = {
    "momentum": [("momentum_20d", 1.0), ("return_60d", 1.0)],
    "value": [("pe_ratio", -1.0), ("pb_ratio", -1.0)],          # cheap = good
    "quality": [("roe", 1.0), ("fcf_yield", 1.0)],
    "low_vol": [("realized_vol_63d", -1.0), ("idio_vol_60d", -1.0)],  # calm = good
}
SCANNER_RAW_FACTORS: tuple = tuple(
    f for sl in _SCANNER_FACTOR_SLEEVES.values() for f, _ in sl
)

# The LIVE universe-board attractiveness composite's pillar set (config#1398 /
# ARCHITECTURE §43). These are the 6 sector-neutral pillar percentiles persisted
# by crucible-research to ``factors/profiles/{date}/by_ticker.json`` — the exact
# inputs to ``scoring/universe_board.py::compute_cross_sectional_attractiveness``
# (each already 0-100, higher = better). The counterfactual ranks the scanned
# universe by the SAME blend (cross-sectional winsorized-z per pillar, clip ±3,
# coverage-renormalized equal-weight mean) the live board uses; the board's
# terminal cross-sectional PERCENTILE step is a monotone transform, so the
# top-N SELECTION this test makes is identical to ranking by the board's
# ``attractiveness_score``. This measures the EXACT live attractiveness feed (vs
# the deep-history ``multifactor_topN`` raw-factor proxy), so it matures to robust
# N only as profile history accrues (~7 weekly snapshots at birth, 2026-06).
_ATTRACTIVENESS_PILLARS: tuple = (
    "quality_score", "value_score", "momentum_score",
    "growth_score", "stewardship_score", "low_vol_score",
)


def load_historical_pillar_profiles(bucket: str, eval_dates) -> dict:
    """Build ``{(eval_date, ticker): {pillar_score_key: value}}`` from the
    persisted research factor-profile artifacts for the attractiveness
    counterfactual (config#1398).

    Reads ``s3://{bucket}/factors/profiles/{eval_date}/by_ticker.json`` (the
    exact 6-pillar sector-neutral percentile scores the live universe board
    consumes). A research-cycle ``eval_date`` is matched EXACTLY — unlike the raw
    ArcticDB loader these are research-keyed artifacts, not a trading-day series,
    so an as-of ffill would silently attribute a stale cohort's pillars to a date
    that never produced them. A date with no profile artifact is simply absent
    (the method reports reduced cohort coverage). Fail-soft: any error returns
    ``{}`` so the e2e_lift contract is never broken.
    """
    try:
        import json as _json

        import boto3

        s3 = boto3.client("s3")
        out: dict = {}
        for d_str in sorted(set(eval_dates)):
            key = f"factors/profiles/{d_str}/by_ticker.json"
            try:
                body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            except Exception:
                continue  # no profile for this cohort -> reduced coverage
            try:
                prof = _json.loads(body)
            except Exception:
                continue
            if not isinstance(prof, dict):
                continue
            for ticker, rec in prof.items():
                if not isinstance(rec, dict):
                    continue
                pillars = {
                    p: float(rec[p])
                    for p in _ATTRACTIVENESS_PILLARS
                    if rec.get(p) is not None and isinstance(rec.get(p), (int, float))
                }
                if pillars:
                    out[(d_str, ticker)] = pillars
        return out
    except Exception as e:  # fail-soft — never break the diagnostics run
        logger.warning("load_historical_pillar_profiles failed (non-fatal): %s", e)
        return {}


def load_historical_trajectory_scores(bucket: str, eval_dates) -> dict:
    """Build ``{(eval_date, ticker): {"pre_repricing_score": .., "attr_slope_z": ..}}``
    from the persisted attractiveness-trajectory artifacts, for the observe-mode
    forward-IC gate (crucible-research #337 / config#1392).

    Reads ``s3://{bucket}/scanner/universe/trajectory/{eval_date}/trajectory.json``
    — the exact weekly artifact ``crucible-research`` writes (see
    ``scoring/attractiveness_trajectory.compute_and_write_trajectory``). Each name
    in the artifact's ``stocks`` list carries ``pre_repricing_score`` (the OLS
    residual of ``attr_slope_z`` on sector-relative price momentum — rising
    attractiveness the market has NOT yet repriced) and ``attr_slope_z`` (the
    cross-sectional z of the Theil-Sen attractiveness slope).

    A research-cycle ``eval_date`` is matched EXACTLY (like
    :func:`load_historical_pillar_profiles`, NOT as-of ffill): the trajectory
    artifacts are research-keyed weekly snapshots, not a trading-day series, so an
    as-of match would attribute a stale cohort's scores to a date that never
    produced them. A date with no artifact is simply absent (reduced cohort
    coverage). Fail-soft: any error returns ``{}`` so the e2e_lift contract is
    never broken.
    """
    try:
        import json as _json

        import boto3

        s3 = boto3.client("s3")
        out: dict = {}
        for d_str in sorted(set(eval_dates)):
            key = f"scanner/universe/trajectory/{d_str}/trajectory.json"
            try:
                body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            except Exception:
                continue  # no trajectory artifact for this cohort -> reduced coverage
            try:
                art = _json.loads(body)
            except Exception:
                continue
            if not isinstance(art, dict):
                continue
            for rec in art.get("stocks", []) or []:
                if not isinstance(rec, dict):
                    continue
                ticker = rec.get("ticker")
                if not ticker:
                    continue
                scores = {
                    k: float(rec[k])
                    for k in ("pre_repricing_score", "attr_slope_z")
                    if rec.get(k) is not None and isinstance(rec.get(k), (int, float))
                }
                if scores:
                    out[(d_str, str(ticker))] = scores
        return out
    except Exception as e:  # fail-soft — never break the diagnostics run
        logger.warning("load_historical_trajectory_scores failed (non-fatal): %s", e)
        return {}


def _trajectory_forward_ic(conn, trajectory_scores: dict | None = None) -> dict:
    """OBSERVE-mode rolling forward-IC of the attractiveness-trajectory signal
    (crucible-research #337 / config#1392).

    The trajectory signal (``scoring/attractiveness_trajectory.py``) is written
    weekly but has been pure OBSERVE-mode: the artifact schema hard-codes
    ``provisional_ic: None`` and the console shows ``provisional_ic: accruing``
    because nothing ever measured its forward predictive power. This producer is
    the missing measurement. It joins the PERSISTED per-name trajectory scores
    (``pre_repricing_score`` and ``attr_slope_z``, injected via
    :func:`load_historical_trajectory_scores`) to realized 21d log
    market-relative alpha (``universe_returns``: ``log_return_21d -
    log_spy_return_21d``) on ``(eval_date, ticker)`` — the SAME realized-return
    source and join key ``_neutralized_live_forward_ic`` uses — and computes the
    per-week cross-sectional Spearman rank-IC of EACH score vs forward alpha via
    the shared quant engine (``analysis.information_coefficient.compute_ic``,
    re-export of ``nousergon_lib.quant.stats.information_coefficient`` — one shared
    IC, not per-repo Spearman). Each mature week is ONE cohort observation; the
    reported IC is the mean of the per-week ICs and ``n_cohorts`` = effective N.

    Maturity gate (mirrors how ``_neutralized_live_forward_ic`` treats immature
    cohorts as an HONEST None rather than a crash): a week counts only when it has
    >= ``TRAJECTORY_FORWARD_IC_MIN_NAMES_PER_WEEK`` names with both a trajectory
    score and a realized 21d outcome. With fewer than
    ``TRAJECTORY_FORWARD_IC_MIN_COHORTS`` mature weeks the status is ``accruing``
    and the mean ICs are None — the value the console header surfaces in place of
    ``provisional_ic: accruing``, and the value that makes the observe->cutover
    gate decidable WITHOUT this producer ever auto-promoting the signal (that
    decision is the operator's).

    Fail-soft: any precondition miss / error -> status skipped / insufficient_data
    / accruing / error, never breaking the e2e_lift contract.
    """
    try:
        from analysis.information_coefficient import compute_ic

        if not trajectory_scores:
            # No injected artifacts (loader returned {} — no trajectory artifacts
            # in the bucket yet, or evaluate had no bucket). Honest accruing, NOT
            # an error: the measurement only becomes possible once weekly
            # trajectory artifacts accrue and are read.
            return {
                "status": "accruing",
                "reason": "no persisted trajectory artifacts available",
                "n_cohorts": 0,
                "min_cohorts": TRAJECTORY_FORWARD_IC_MIN_COHORTS,
            }
        tabs = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "universe_returns" not in tabs:
            return {"status": "skipped", "reason": "universe_returns absent"}
        ur_cols = {r[1] for r in conn.execute("PRAGMA table_info(universe_returns)")}
        if not {"log_return_21d", "log_spy_return_21d"}.issubset(ur_cols):
            return {"status": "skipped", "reason": "universe_returns lacks log_return_21d / log_spy_return_21d"}

        ur = pd.read_sql_query(
            "SELECT ticker, eval_date, "
            "(log_return_21d - log_spy_return_21d) AS log_alpha_21d "
            "FROM universe_returns "
            "WHERE log_return_21d IS NOT NULL AND log_spy_return_21d IS NOT NULL",
            conn,
        )
        if ur.empty:
            return {"status": "insufficient_data", "reason": "no universe_returns rows with realized 21d outcomes"}

        # Join persisted trajectory scores to realized forward alpha on
        # (eval_date, ticker) — exact match, the artifacts are research-keyed.
        recs = []
        for (d_str, ticker), scores in trajectory_scores.items():
            recs.append({
                "eval_date": d_str,
                "ticker": str(ticker),
                "pre_repricing_score": scores.get("pre_repricing_score"),
                "attr_slope_z": scores.get("attr_slope_z"),
            })
        if not recs:
            return {
                "status": "accruing",
                "reason": "trajectory artifacts carried no scored names",
                "n_cohorts": 0,
                "min_cohorts": TRAJECTORY_FORWARD_IC_MIN_COHORTS,
            }
        traj = pd.DataFrame.from_records(recs)
        m = traj.merge(ur, on=["eval_date", "ticker"], how="inner")
        if m.empty:
            return {
                "status": "accruing",
                "reason": "no trajectory names have a realized 21d outcome yet",
                "n_cohorts": 0,
                "min_cohorts": TRAJECTORY_FORWARD_IC_MIN_COHORTS,
            }

        signals = ("pre_repricing_score", "attr_slope_z")
        # per-signal lists of mature weekly ICs
        weekly: dict = {s: [] for s in signals}
        n_mature_weeks = 0
        n_join_rows = int(len(m))
        for _d, g in m.groupby("eval_date"):
            g = g[g["log_alpha_21d"].notna()]
            if len(g) < TRAJECTORY_FORWARD_IC_MIN_NAMES_PER_WEEK:
                continue
            week_counted = False
            for s in signals:
                sub = g[g[s].notna()]
                if len(sub) < TRAJECTORY_FORWARD_IC_MIN_NAMES_PER_WEEK:
                    continue
                # Shared quant engine: Spearman rank-IC. min_samples is the
                # per-week names floor (compute_ic's default 20 is tuned for the
                # whole-sample p-value, not a single weekly cross-section).
                res = compute_ic(
                    sub[s], sub["log_alpha_21d"],
                    min_samples=TRAJECTORY_FORWARD_IC_MIN_NAMES_PER_WEEK,
                )
                if res.get("status") == "ok":
                    weekly[s].append(float(res["ic"]))
                    week_counted = True
            if week_counted:
                n_mature_weeks += 1

        def _summary(ics: list) -> dict:
            n = len(ics)
            mature = n >= TRAJECTORY_FORWARD_IC_MIN_COHORTS
            return {
                "n_cohorts": n,
                "mean_weekly_ic": round(float(sum(ics) / n), 4) if (n and mature) else None,
                "positive": bool(mature and (sum(ics) / n) > 0),
                "status": "ok" if mature else "accruing",
            }

        per_signal = {s: _summary(weekly[s]) for s in signals}
        # Overall maturity is gated on the primary signal (pre_repricing_score,
        # the orthogonalized headline metric); n_cohorts is its mature-week count.
        primary = per_signal["pre_repricing_score"]
        overall_status = "ok" if primary["n_cohorts"] >= TRAJECTORY_FORWARD_IC_MIN_COHORTS else "accruing"

        return {
            "status": overall_status,
            "horizon": "21d",
            "source": "persisted scanner/universe/trajectory/{date}/trajectory.json (config#1392)",
            "signal": "attractiveness_trajectory (crucible-research #337, observe-mode)",
            "min_cohorts": TRAJECTORY_FORWARD_IC_MIN_COHORTS,
            "min_names_per_week": TRAJECTORY_FORWARD_IC_MIN_NAMES_PER_WEEK,
            "n_join_rows": n_join_rows,
            "n_mature_weeks": n_mature_weeks,
            # headline for the console "Attractiveness Trends" header (replaces
            # the hard-coded ``provisional_ic: accruing``).
            "n_cohorts": primary["n_cohorts"],
            "provisional_ic": primary["mean_weekly_ic"],
            "pre_repricing_score": per_signal["pre_repricing_score"],
            "attr_slope_z": per_signal["attr_slope_z"],
        }
    except sqlite3.OperationalError:
        return {"status": "skipped", "reason": "universe_returns table not found"}
    except Exception as e:  # pragma: no cover - defensive; never break e2e_lift
        return {"status": "error", "reason": str(e)}


def _scanner_factor_counterfactual(
    conn, loadings: dict | None = None, pillar_profiles: dict | None = None
) -> dict:
    """Would a MULTI-FACTOR candidate-generation beat the momentum-only scanner?
    (config#1186 — the candidate-generation / scanner-edge test + reconciliation
    with the live momentum-neutralization #1142.)

    The funnel analysis showed candidate-pool alpha is absolutely negative, so
    the bottleneck is candidate generation. Today's scanner ranks ~900 -> ~60 on
    a momentum/technical ``tech_score`` only. This counterfactual asks, on the
    full scanned universe (``scanner_evaluations`` x realized 21d), per cycle and
    count-matched to the actual pass count: does ranking by an institutional
    multi-factor composite — or by any single factor SLEEVE (momentum / value /
    quality / low-vol) — produce a candidate pool with better realized 21d
    log-alpha than the names the live scanner actually passed?

    The composite is built per cycle (cross-sectional z-score within the scanned
    universe; sleeves equal-weighted). ``loadings``: {(eval_date, ticker):
    {raw_factor: value}} from ArcticDB — empty -> status skipped. Pure read.
    """
    try:
        if not loadings and not pillar_profiles:
            return {"status": "skipped", "reason": "no factor loadings or pillar profiles provided"}
        tabs = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "scanner_evaluations" not in tabs or "universe_returns" not in tabs:
            return {"status": "skipped", "reason": "scanner_evaluations / universe_returns absent"}
        se_cols = {r[1] for r in conn.execute("PRAGMA table_info(scanner_evaluations)")}
        if "quant_filter_pass" not in se_cols:
            return {"status": "skipped", "reason": "scanner_evaluations has no quant_filter_pass"}
        # tech_score is the live scanner's CONTINUOUS score — pulled (when present)
        # so the live scanner can be ranked on the cross-sectional rank-IC axis
        # alongside the factor sleeves, which is the axis #1142 neutralizes.
        has_tech = "tech_score" in se_cols
        tech_sel = "se.tech_score, " if has_tech else ""
        m = pd.read_sql_query(
            f"SELECT se.ticker, se.eval_date, se.quant_filter_pass, {tech_sel}u.sector, "
            "(u.log_return_21d - u.log_spy_return_21d) AS alpha21 "
            "FROM scanner_evaluations se "
            "JOIN universe_returns u ON u.ticker=se.ticker AND u.eval_date=se.eval_date "
            "WHERE u.log_return_21d IS NOT NULL AND u.log_spy_return_21d IS NOT NULL",
            conn,
        )
        if m.empty:
            return {"status": "insufficient_data", "reason": "no scanned rows with realized 21d"}

        sleeve_names = list(_SCANNER_FACTOR_SLEEVES)
        methods = ["actual_scanner_pass", "multifactor_topN"] + [s + "_sleeve_topN" for s in sleeve_names]
        if pillar_profiles:
            methods = methods + ["attractiveness_topN"]
        picked: dict = {k: [] for k in methods}
        picked_sn: dict = {k: [] for k in methods}
        universe_alpha: list = []
        n_pass_total = 0
        n_cycles = 0
        n_attr_cycles = 0  # cycles where the live attractiveness composite was computable
        # Per-cycle records for the objective-asymmetry reconciliation (config#1186
        # vs #1142). Each cycle contributes ONE observation per variant on two
        # axes: (a) ``topn`` — mean realized 21d alpha of the count-matched top-N
        # selection (the scanner's OWN objective: pick names to BUY); (b) ``ic`` —
        # the full cross-sectional Spearman rank-IC of the variant's score vs
        # realized alpha (the axis #1142 neutralizes). Date-clustered post-loop so
        # significance uses weeks-as-N, not pooled names (the pseudo-replication
        # trap, config#1164).
        cycle_recs: list[dict] = []
        from scipy.stats import spearmanr

        def _xs_ic(score: pd.Series, alpha: pd.Series):
            """Full cross-sectional Spearman rank-IC within one cycle (or None if
            under-powered: <5 names or a degenerate score/alpha distribution)."""
            valid = score.notna() & alpha.notna()
            if valid.sum() < 5 or score[valid].nunique() < 3 or alpha[valid].nunique() < 3:
                return None
            rho, _p = spearmanr(score[valid], alpha[valid])
            return float(rho) if rho == rho else None

        for d, g in m.groupby("eval_date"):
            g = g[g["alpha21"].notna()].copy()
            if g.empty:
                continue
            n_pass = int((g["quant_filter_pass"] == 1).sum())
            if n_pass == 0:
                continue
            n_cycles += 1
            n_pass_total += n_pass
            universe_alpha.extend(g["alpha21"].tolist())
            if g["sector"].notna().any():
                g["alpha_sn"] = g["alpha21"] - g.groupby("sector")["alpha21"].transform("mean")
            else:
                g["alpha_sn"] = g["alpha21"]

            sleeve_score: dict = {}
            composite = None
            if loadings:
                exrows = {t: (loadings.get((d, t)) or {}) for t in g["ticker"]}
                fac = pd.DataFrame.from_dict(exrows, orient="index")
                z = pd.DataFrame(index=fac.index)
                for col in SCANNER_RAW_FACTORS:
                    if col in fac.columns:
                        s = pd.to_numeric(fac[col], errors="coerce")
                        sd = s.std(ddof=0)
                        z[col] = (s - s.mean()) / sd if sd and sd > 1e-12 else 0.0
                for sl, facs in _SCANNER_FACTOR_SLEEVES.items():
                    cols = [(f, sign) for f, sign in facs if f in z.columns]
                    if cols:
                        sleeve_score[sl] = pd.concat([sign * z[f] for f, sign in cols], axis=1).mean(axis=1)
                if sleeve_score:
                    composite = pd.concat(sleeve_score.values(), axis=1).mean(axis=1)

            # Exact live attractiveness composite (config#1398): cross-sectional
            # winsorized-z per pillar (clip ±3, matching universe_board) then a
            # coverage-renormalized equal-weight blend == row-mean of the present
            # pillar z's. Top-N selection is invariant to the board's terminal
            # cross-sectional percentile, so this IS the live attractiveness feed.
            attractiveness = None
            if pillar_profiles:
                prows = {t: (pillar_profiles.get((d, t)) or {}) for t in g["ticker"]}
                pdf = pd.DataFrame.from_dict(prows, orient="index")
                if not pdf.empty:
                    pz = pd.DataFrame(index=pdf.index)
                    for p in _ATTRACTIVENESS_PILLARS:
                        if p in pdf.columns:
                            s = pd.to_numeric(pdf[p], errors="coerce")
                            sd = s.std(ddof=0)
                            zz = (s - s.mean()) / sd if sd and sd > 1e-12 else s * 0.0
                            pz[p] = zz.clip(-3.0, 3.0)
                    if pz.shape[1] > 0 and bool(pz.notna().any().any()):
                        attractiveness = pz.mean(axis=1)

            if composite is None and attractiveness is None:
                continue
            g = g.set_index("ticker")
            if composite is not None:
                g["_composite"] = composite
            for sl, sc in sleeve_score.items():
                g["_sleeve_" + sl] = sc
            if attractiveness is not None:
                g["_attractiveness"] = attractiveness.reindex(g.index)
                n_attr_cycles += 1

            def _topN_alpha(score_col):
                sub = g[g[score_col].notna()]
                if sub.empty:
                    return None, None
                top = sub.sort_values(score_col, ascending=False).head(n_pass)
                return top["alpha21"].tolist(), top["alpha_sn"].tolist()

            actual = g[g["quant_filter_pass"] == 1]
            picked["actual_scanner_pass"].extend(actual["alpha21"].tolist())
            picked_sn["actual_scanner_pass"].extend(actual["alpha_sn"].tolist())
            mf_a = mf_sn = None
            if "_composite" in g.columns:
                mf_a, mf_sn = _topN_alpha("_composite")
                if mf_a:
                    picked["multifactor_topN"].extend(mf_a)
                    picked_sn["multifactor_topN"].extend(mf_sn)
            for sl in sleeve_names:
                col = "_sleeve_" + sl
                if col in g.columns:
                    a, sn = _topN_alpha(col)
                    if a:
                        picked[sl + "_sleeve_topN"].extend(a)
                        picked_sn[sl + "_sleeve_topN"].extend(sn)
            at_a = None
            if "_attractiveness" in g.columns:
                at_a, at_sn = _topN_alpha("_attractiveness")
                if at_a:
                    picked["attractiveness_topN"].extend(at_a)
                    picked_sn["attractiveness_topN"].extend(at_sn)

            # --- per-cycle records for the two-axis reconciliation ---
            rec_topn: dict = {}
            rec_ic: dict = {}
            if len(actual):
                rec_topn["actual_scanner_pass"] = float(actual["alpha21"].mean())
            if has_tech and "tech_score" in g.columns:
                ic_ts = _xs_ic(g["tech_score"], g["alpha21"])
                if ic_ts is not None:
                    rec_ic["actual_scanner_pass"] = ic_ts
            if mf_a:
                rec_topn["multifactor_topN"] = float(sum(mf_a) / len(mf_a))
            if "_composite" in g.columns:
                ic_mf = _xs_ic(g["_composite"], g["alpha21"])
                if ic_mf is not None:
                    rec_ic["multifactor_topN"] = ic_mf
            if at_a:
                rec_topn["attractiveness_topN"] = float(sum(at_a) / len(at_a))
            if "_attractiveness" in g.columns:
                ic_at = _xs_ic(g["_attractiveness"], g["alpha21"])
                if ic_at is not None:
                    rec_ic["attractiveness_topN"] = ic_at
            for sl in sleeve_names:
                col = "_sleeve_" + sl
                if col not in g.columns:
                    continue
                a2, _sn2 = _topN_alpha(col)
                if a2:
                    rec_topn[sl + "_sleeve_topN"] = float(sum(a2) / len(a2))
                ic_sl = _xs_ic(g[col], g["alpha21"])
                if ic_sl is not None:
                    rec_ic[sl + "_sleeve_topN"] = ic_sl
            cycle_recs.append(
                {"breadth": float((g["alpha21"] > 0).mean()), "topn": rec_topn, "ic": rec_ic}
            )

        if n_cycles < 3:
            return {"status": "insufficient_data", "reason": f"only {n_cycles} cycles", "n_cycles": n_cycles}

        def _mean(xs):
            return round(float(sum(xs) / len(xs)), 5) if xs else None

        actual_mean = _mean(picked["actual_scanner_pass"])
        actual_sn = _mean(picked_sn["actual_scanner_pass"])
        out_methods: dict = {}
        best = None
        best_alpha = actual_mean if actual_mean is not None else -1e9
        for k in methods:
            ma = _mean(picked[k])
            sn = _mean(picked_sn[k])
            e = {"mean_alpha_21d": ma, "sector_neutral_mean_alpha_21d": sn, "n_picks": len(picked[k])}
            if k == "actual_scanner_pass":
                # config#2318: this method replays the retired tech_score gate
                # (``scanner_evaluations.quant_filter_pass``), not the live
                # champion-feed selection — label it explicitly (additive-only).
                e["arm"] = SCANNER_METRIC_ARM
            if k != "actual_scanner_pass" and ma is not None and actual_mean is not None:
                e["lift_vs_actual_scanner"] = round(ma - actual_mean, 5)
                e["sn_lift_vs_actual_scanner"] = (
                    round(sn - actual_sn, 5) if (sn is not None and actual_sn is not None) else None
                )
                if ma > best_alpha:
                    best_alpha, best = ma, k
            out_methods[k] = e

        best_sleeve = None
        sleeve_best = -1e9
        for sl in sleeve_names:
            ma = out_methods[sl + "_sleeve_topN"]["mean_alpha_21d"]
            if ma is not None and ma > sleeve_best:
                sleeve_best, best_sleeve = ma, sl

        # ---- Objective-asymmetry reconciliation (config#1186 vs #1142) --------
        # The pooled means above answer "which candidate-gen picks the best top-N
        # pool" but pool draws across weeks as if independent (pseudo-replication,
        # config#1164). Here we date-cluster EACH variant on TWO axes so the
        # +0.050 momentum-sleeve read is tested honestly and reconciled with the
        # live momentum-NEUTRALIZATION (#1142): the scanner's own objective is
        # long-only top-N selection (momentum should help), while #1142 targets
        # the cross-sectional rank-IC (where momentum is toxic in low-breadth).
        import numpy as np
        from scipy.stats import ttest_1samp

        def _date_cluster(values: list) -> dict:
            vals = [v for v in values if v is not None and v == v]
            if len(vals) < 3:
                return {"mean": (round(float(np.mean(vals)), 5) if vals else None), "p": None, "n": len(vals)}
            _t, p = ttest_1samp(vals, 0.0)
            return {"mean": round(float(np.mean(vals)), 5), "p": round(float(p), 4) if p == p else None, "n": len(vals)}

        axes: dict = {}
        for k in methods:
            axes[k] = {
                "longonly_topn_alpha": _date_cluster([r["topn"].get(k) for r in cycle_recs if k in r["topn"]]),
                "xs_rank_ic": _date_cluster([r["ic"].get(k) for r in cycle_recs if k in r["ic"]]),
            }

        def _paired_lift(variant_key: str, axis: str) -> dict:
            # Per-cycle diff vs the live scanner on ``axis`` (paired where both present).
            diffs = [
                r[axis][variant_key] - r[axis]["actual_scanner_pass"]
                for r in cycle_recs
                if variant_key in r[axis] and "actual_scanner_pass" in r[axis]
            ]
            return _date_cluster(diffs)

        mom_key = "momentum_sleeve_topN"
        mom_longonly_lift = _paired_lift(mom_key, "topn")
        mom_xs_ic = axes.get(mom_key, {}).get("xs_rank_ic", {"mean": None, "p": None, "n": 0})

        breadths = [r["breadth"] for r in cycle_recs]
        breadth_strat = None
        if len(breadths) >= 4:
            med_b = float(np.median(breadths))

            def _regime(lo: bool, key: str, axis: str) -> dict:
                return _date_cluster([
                    r[axis].get(key) for r in cycle_recs
                    if key in r[axis] and ((r["breadth"] <= med_b) == lo)
                ])

            breadth_strat = {"median_breadth": round(med_b, 4)}
            for key in (mom_key, "actual_scanner_pass"):
                breadth_strat[key] = {
                    "low_breadth": {"longonly_topn_alpha": _regime(True, key, "topn"), "xs_rank_ic": _regime(True, key, "ic")},
                    "high_breadth": {"longonly_topn_alpha": _regime(False, key, "topn"), "xs_rank_ic": _regime(False, key, "ic")},
                }

        ll_mean, ll_p = mom_longonly_lift.get("mean"), mom_longonly_lift.get("p")
        ic_mean, ic_p = mom_xs_ic.get("mean"), mom_xs_ic.get("p")
        sleeve_beats_live_longonly_significant = bool(
            ll_mean is not None and ll_mean > 0 and ll_p is not None and ll_p < 0.05
        )
        # Reconciliation holds when momentum's value is selection-tail (long-only),
        # NOT a significant-positive cross-sectional rank skill — i.e. exactly what
        # #1142 neutralizes out.
        consistent_with_1142 = bool(ic_mean is None or ic_mean <= 0 or ic_p is None or ic_p >= 0.05)
        if sleeve_beats_live_longonly_significant and consistent_with_1142:
            verdict = (
                "momentum sleeve beats the live scanner on the long-only top-N objective with "
                "date-clustered significance while its cross-sectional rank-IC is flat/negative — "
                "asymmetry confirmed: the scanner (long-only selection) should KEEP momentum even "
                "though the composite (cross-sectional rank) neutralizes it out (#1142). PROCEED to "
                "an observe-mode shadow scanner (A3)."
            )
        elif ll_mean is not None and ll_mean > 0:
            verdict = (
                "momentum sleeve leads the live scanner on the long-only objective but NOT with "
                "date-clustered significance — the original +0.050 pooled read was pseudo-replicated; "
                "ACCUMULATE more cohorts before standing up a shadow scanner."
            )
        else:
            verdict = (
                "momentum sleeve does NOT beat the live scanner on the long-only objective once "
                "date-clustered — do NOT promote a momentum-sleeve scanner."
            )
        reconciliation = {
            "axis_definitions": {
                "longonly_topn_alpha": "the scanner's OWN objective — mean realized 21d alpha of the count-matched top-N BUY selection",
                "xs_rank_ic": "the axis #1142 neutralizes — full cross-sectional Spearman rank-IC of the score vs realized alpha",
            },
            "momentum_sleeve_longonly_lift_vs_live": mom_longonly_lift,
            "momentum_sleeve_xs_rank_ic": mom_xs_ic,
            "sleeve_beats_live_longonly_significant": sleeve_beats_live_longonly_significant,
            "consistent_with_1142_neutralization": consistent_with_1142,
            "verdict": verdict,
        }

        return {
            "status": "ok",
            "horizon": "21d",
            "n_cycles": n_cycles,
            "n_pass_total": n_pass_total,
            "composite": "equal-weight z(momentum,value,quality,low_vol)",
            "attractiveness_composite": (
                "equal-weight winsorized-z of 6 live pillars "
                "(quality,value,momentum,growth,stewardship,low_vol) — the live "
                "universe-board attractiveness feed (config#1398)"
            ),
            "attractiveness_profile_cohorts": n_attr_cycles,
            "selection_count_basis": "per-cycle top-N matched to the live scanner pass count",
            "universe_mean_alpha_21d": _mean(universe_alpha),
            "methods": out_methods,
            "best_method": best or "actual_scanner_pass",
            "best_sleeve": best_sleeve,
            "any_factor_beats_actual_scanner": best is not None,
            "objective_axes": axes,
            "breadth_stratified": breadth_strat,
            "reconciliation": reconciliation,
        }
    except sqlite3.OperationalError as e:
        return {"status": "skipped", "reason": f"sqlite: {e}"}
    except Exception as e:  # pragma: no cover - defensive; never break e2e_lift
        return {"status": "error", "reason": str(e)}


def _scanner_then_predictor_topN(conn) -> dict:
    """Arm-4 counterfactual (config#1405): scanner -> research-free predictor direct.

    Bypasses the research/agentic layer entirely. Among the scanner-passing
    universe (``scanner_evaluations.quant_filter_pass=1``), rank by the
    research-free predictor's ``predicted_alpha`` (table
    ``predictor_outcomes_research_free`` — the meta-ensemble's
    ``canonical_predicted_alpha`` run with the 4 research meta-features omitted ->
    0.0, per the issue's research-free definition) and take the count-matched
    top-N, then compare realized 21d log-alpha against:

      * ``actual_scanner_pass``  — the full live scanner pass pool (the input the
        arm re-ranks), and
      * ``agentic_cio_advance``  — the live CIO ADVANCE selection (the agentic
        path this arm would REPLACE).

    Answers config#1405's question: *does the research layer add anything over
    the ML predictor on scanner candidates?* Count basis: per cycle, N = the live
    CIO ADVANCE count (``ADVANCE`` + ``ADVANCE_FORCED``) so the predictor selects
    the same NUMBER of names the agentic stack ultimately holds — an
    apples-to-apples selection-size match (mirrors the count-matched top-N of
    ``_scanner_factor_counterfactual``). Cycles with no CIO advance are skipped
    (no agentic baseline to match). Sector(+cycle)-neutral residual = alpha minus
    its per-``sector`` group mean WITHIN the cycle. Pure read; fail-soft to
    ``status="skipped"`` so it can never break the e2e_lift contract — and it
    stays ``skipped`` until the Saturday spot-box backfill populates
    ``predictor_outcomes_research_free``.
    """
    try:
        tabs = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        need = {
            "scanner_evaluations",
            "universe_returns",
            "predictor_outcomes_research_free",
            "cio_evaluations",
        }
        missing = need - tabs
        if missing:
            return {"status": "skipped", "reason": f"missing tables: {sorted(missing)}"}
        se_cols = {r[1] for r in conn.execute("PRAGMA table_info(scanner_evaluations)")}
        if "quant_filter_pass" not in se_cols:
            return {"status": "skipped", "reason": "scanner_evaluations has no quant_filter_pass"}
        prf_cols = {r[1] for r in conn.execute("PRAGMA table_info(predictor_outcomes_research_free)")}
        if "predicted_alpha" not in prf_cols:
            return {"status": "skipped", "reason": "predictor_outcomes_research_free has no predicted_alpha"}
        has_nmiss = "n_research_features_missing" in prf_cols
        nmiss_sel = "prf.n_research_features_missing, " if has_nmiss else ""

        # Scanner-passing universe x realized 21d x research-free predicted_alpha.
        # LEFT JOIN so the actual-scanner-pass pool is the FULL passing set even
        # where the backfill lacks a prediction (predictor ranks only the scored
        # subset, but the scanner baseline must reflect every name it passed).
        passing = pd.read_sql_query(
            "SELECT se.ticker, se.eval_date, u.sector, "
            "(u.log_return_21d - u.log_spy_return_21d) AS alpha21, "
            f"prf.predicted_alpha, {nmiss_sel}"
            "se.quant_filter_pass "
            "FROM scanner_evaluations se "
            "JOIN universe_returns u ON u.ticker=se.ticker AND u.eval_date=se.eval_date "
            "LEFT JOIN predictor_outcomes_research_free prf "
            "  ON prf.ticker=se.ticker AND prf.prediction_date=se.eval_date "
            "WHERE se.quant_filter_pass=1 "
            "  AND u.log_return_21d IS NOT NULL AND u.log_spy_return_21d IS NOT NULL",
            conn,
        )
        if passing.empty:
            return {"status": "insufficient_data", "reason": "no scanner-passing rows with realized 21d"}

        # Live agentic CIO ADVANCE selection x realized 21d (the path replaced).
        cio = pd.read_sql_query(
            "SELECT ce.ticker, ce.eval_date, ce.cio_decision, u.sector, "
            "(u.log_return_21d - u.log_spy_return_21d) AS alpha21 "
            "FROM cio_evaluations ce "
            "JOIN universe_returns u ON u.ticker=ce.ticker AND u.eval_date=ce.eval_date "
            "WHERE u.log_return_21d IS NOT NULL AND u.log_spy_return_21d IS NOT NULL",
            conn,
        )
        cio_adv = cio[cio["cio_decision"].isin(("ADVANCE", "ADVANCE_FORCED"))]

        methods = ["actual_scanner_pass", "scanner_then_predictor_topN", "agentic_cio_advance"]
        picked: dict = {k: [] for k in methods}
        picked_sn: dict = {k: [] for k in methods}
        n_cycles = 0
        n_scored_total = 0
        nmiss_vals: list = []

        def _sector_neutral(frame: pd.DataFrame) -> pd.Series:
            if frame["sector"].notna().any():
                return frame["alpha21"] - frame.groupby("sector")["alpha21"].transform("mean")
            return frame["alpha21"]

        for d, g in passing.groupby("eval_date"):
            g = g[g["alpha21"].notna()].copy()
            if g.empty:
                continue
            adv = cio_adv[cio_adv["eval_date"] == d].copy()
            adv = adv[adv["alpha21"].notna()]
            n_adv = len(adv)
            if n_adv == 0:
                continue  # no agentic baseline to count-match against this cycle
            n_cycles += 1
            g["alpha_sn"] = _sector_neutral(g)

            # actual scanner pass pool (full passing set)
            picked["actual_scanner_pass"].extend(g["alpha21"].tolist())
            picked_sn["actual_scanner_pass"].extend(g["alpha_sn"].tolist())

            # research-free predictor: rank the SCORED passing names, count-matched
            scored = g[g["predicted_alpha"].notna()]
            n_scored_total += len(scored)
            if has_nmiss:
                nmiss_vals.extend(scored["n_research_features_missing"].dropna().tolist())
            if not scored.empty:
                top = scored.sort_values("predicted_alpha", ascending=False).head(min(n_adv, len(scored)))
                picked["scanner_then_predictor_topN"].extend(top["alpha21"].tolist())
                picked_sn["scanner_then_predictor_topN"].extend(top["alpha_sn"].tolist())

            # agentic CIO advance picks this cycle
            adv["alpha_sn"] = _sector_neutral(adv)
            picked["agentic_cio_advance"].extend(adv["alpha21"].tolist())
            picked_sn["agentic_cio_advance"].extend(adv["alpha_sn"].tolist())

        if n_cycles < 1:
            return {"status": "insufficient_data", "reason": "no cycles with both scanner pass + CIO advance"}
        if not picked["scanner_then_predictor_topN"]:
            return {"status": "skipped", "reason": "no research-free predictions matched the scanner-passing universe"}

        def _mean(xs):
            return round(float(sum(xs) / len(xs)), 5) if xs else None

        scanner_mean = _mean(picked["actual_scanner_pass"])
        scanner_sn = _mean(picked_sn["actual_scanner_pass"])
        cio_mean = _mean(picked["agentic_cio_advance"])
        cio_sn = _mean(picked_sn["agentic_cio_advance"])

        out_methods: dict = {}
        for k in methods:
            out_methods[k] = {
                "mean_alpha_21d": _mean(picked[k]),
                "sector_neutral_mean_alpha_21d": _mean(picked_sn[k]),
                "n_picks": len(picked[k]),
            }
        pred = out_methods["scanner_then_predictor_topN"]
        pm, psn = pred["mean_alpha_21d"], pred["sector_neutral_mean_alpha_21d"]
        pred["lift_vs_actual_scanner"] = (
            round(pm - scanner_mean, 5) if (pm is not None and scanner_mean is not None) else None
        )
        pred["sn_lift_vs_actual_scanner"] = (
            round(psn - scanner_sn, 5) if (psn is not None and scanner_sn is not None) else None
        )
        pred["lift_vs_agentic_cio"] = (
            round(pm - cio_mean, 5) if (pm is not None and cio_mean is not None) else None
        )
        pred["sn_lift_vs_agentic_cio"] = (
            round(psn - cio_sn, 5) if (psn is not None and cio_sn is not None) else None
        )

        predictor_beats_agentic = bool(
            pred["lift_vs_agentic_cio"] is not None and pred["lift_vs_agentic_cio"] > 0
        )
        predictor_beats_scanner = bool(
            pred["lift_vs_actual_scanner"] is not None and pred["lift_vs_actual_scanner"] > 0
        )
        nmiss_mode = None
        if nmiss_vals:
            from collections import Counter

            nmiss_mode = Counter(int(x) for x in nmiss_vals).most_common(1)[0][0]

        return {
            "status": "ok",
            "horizon": "21d",
            "n_cycles": n_cycles,
            "selection_count_basis": "per-cycle top-N matched to the live CIO ADVANCE count",
            "n_predictor_scored": n_scored_total,
            # Expect 4 (the 4 research meta-features omitted) — a guard that the
            # backfill ran truly research-free, not with research features present.
            "research_features_missing_mode": nmiss_mode,
            "methods": out_methods,
            "predictor_beats_agentic_cio": predictor_beats_agentic,
            "predictor_beats_actual_scanner": predictor_beats_scanner,
            "interpretation": (
                "scanner_then_predictor_topN > agentic_cio_advance => the research/agentic layer "
                "subtracts value vs a research-free predictor on scanner candidates; <= => the "
                "agentic layer adds selection skill the predictor alone lacks. Directional until "
                "enough 21d cohorts mature (config#1405)."
            ),
        }
    except sqlite3.OperationalError as e:
        return {"status": "skipped", "reason": f"sqlite: {e}"}
    except Exception as e:  # pragma: no cover - defensive; never break e2e_lift
        return {"status": "error", "reason": str(e)}


def _team_lift(conn, ur: pd.DataFrame, date_filter: str, params: list) -> list[dict]:
    """Sector team lift (2b): team picks vs. own sector average from full 900.

    RETIRED from the live ``compute_lift_metrics`` path (config#1580 /
    config-I2993 — the six-team graph no longer produces). RETAINED, uncalled in
    the live path, for direct/historical/windowed readouts and estimator-math
    tests; the live path emits a ``research_graph_retired`` marker instead.


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

            # Canonical 21d horizon (L4551): the system's objective is alpha
            # vs SPY, so the 21d classification uses beat_spy_21d (market-
            # relative) uniformly across selectors — not a sector-relative
            # baseline. Additive alongside the 5d/sector classification above.
            rec_mask = team_data["team_recommended"] == 1
            clf_21d = _classification_for(team_data, rec_mask, "beat_spy_21d")
            lift_21d = _alpha_21d_log_lift(team_data, rec_mask)

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
                "classification_21d": clf_21d,
                "lift_21d_log": lift_21d,
                "picks": picks_records,
            })

        return results
    except sqlite3.OperationalError:
        logger.info("team_lift: team_candidates table not found — returning []")
        return []


def _cio_lift(conn, ur: pd.DataFrame, date_filter: str, params: list) -> dict:
    """CIO lift (2d): ADVANCE stocks vs. all sector recommendations.

    RETIRED from the live ``compute_lift_metrics`` path (config#1580 /
    config-I2993 — the CIO gate no longer produces). RETAINED, uncalled in the
    live path, for direct/historical readouts and estimator-math tests; the live
    path emits a retired ``cio_lift`` marker instead. Also produces the
    ``selection_skill_21d`` / ``layer_attribution_21d`` sub-blocks, retired with it.


    Emits per-bucket stdev so downstream optimizers can compute a
    Welch-style confidence bound on whether ``advance_avg < all_recs_avg``
    is statistically distinguishable from sampling noise. Without this,
    the CIO-fallback recommendation triggers on small-sample noise.
    """
    try:
        ce_filter = date_filter.replace("eval_date", "ce.eval_date") if date_filter else ""
        ce = pd.read_sql_query(
            f"SELECT ticker, eval_date, cio_decision, final_score, cio_conviction, "
            f"combined_score, macro_shift "
            f"FROM cio_evaluations ce{ce_filter}",
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

        # Canonical 21d horizon (L4551): ADVANCE-gate classification on
        # beat_spy_21d + log-domain alpha lift, additive alongside 5d.
        adv_mask = merged["cio_decision"] == "ADVANCE"
        clf_21d = _classification_for(merged, adv_mask, "beat_spy_21d")
        lift_21d = _alpha_21d_log_lift(merged, adv_mask)
        # CIO entrant-gate selection skill (L4561): does ADVANCE beat REJECT at 21d?
        selection_skill = _cio_selection_skill(merged)
        # Layer attribution (L4561): which orchestrated input carries 21d signal?
        layer_attribution = _cio_layer_attribution(merged)

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
            "classification_21d": clf_21d,
            "lift_21d_log": lift_21d,
            "selection_skill_21d": selection_skill,
            "layer_attribution_21d": layer_attribution,
        }
    except sqlite3.OperationalError:
        return {"status": "skipped", "reason": "cio_evaluations table not found"}


def _thinktank_shadow_ic(conn, bucket: str, *, s3_client=None, horizon_days: int | None = None) -> dict:
    """Observe-only Think Tank challenger-arm score IC (config-I2994).

    The Think Tank challenger writes per-cycle shadow signals to
    ``s3://{bucket}/signals_shadow/thinktank_coverage/{eval_date}/signals.json``
    (shape: ``{"date": eval_date, "signals": {ticker: {"score": float, ...}}}``).
    This is the ONLY research-authored composite score left in the live
    architecture — the champion path's signals.json ``score`` is the scanner
    ``attractiveness_score`` verbatim, measured separately by
    ``attractiveness_eval.composite_ic``.

    Computes the SAME date-clustered Spearman rank-IC estimator the scanner arm
    uses (``attractiveness_eval._clustered_ic_block``), joining the shadow score
    to realized 21d market-relative log-alpha from ``universe_returns``
    (``_load_forward_alpha``) — so the two live arms are methodologically
    comparable. The 21d realization lag is respected NATURALLY:
    ``_load_forward_alpha`` returns only rows with a realized ``log_return_21d``,
    so unevaluable recent cycles are excluded.

    Observe-only: labeled, never gates. Honest small-N — ``status
    "insufficient_data"`` with explicit counts (``n_shadow_dates`` /
    ``series_start`` / ``n_eval_dates``) until a shadow cycle has realized 21d
    alpha and enough weekly cohorts accrue for the clustered t-stat.
    """
    # Reuse the scanner arm's estimator + realized-alpha loader (no third copy
    # of the date-clustered IC math; same package, no import cycle).
    from analysis.attractiveness_eval import _clustered_ic_block, _load_forward_alpha
    from nousergon_lib.quant.horizons import DEFAULT_POLICY

    hz = int(horizon_days if horizon_days is not None else DEFAULT_POLICY.primary_horizon)
    arm = THINKTANK_SHADOW_ARM
    prefix = "signals_shadow/thinktank_coverage/"
    try:
        import boto3

        s3 = s3_client or boto3.client("s3")
        rows: list[dict] = []
        shadow_dates: set[str] = set()
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith("/signals.json"):
                    continue
                doc = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
                eval_date = doc.get("date")
                signals = doc.get("signals")
                if not eval_date or not isinstance(signals, dict):
                    continue
                shadow_dates.add(eval_date)
                for ticker, entry in signals.items():
                    if not isinstance(entry, dict):
                        continue
                    score = entry.get("score")
                    if score is None:
                        continue
                    rows.append({"eval_date": eval_date, "ticker": ticker, "score": float(score)})
    except Exception as e:  # expected-absence warm-up guard, WARN-logged (no silent swallow)
        logger.warning("thinktank_shadow_ic: shadow read failed (non-fatal): %s", e)
        return {"status": "error", "arm": arm, "reason": str(e), "horizon_days": hz}

    if not rows:
        return {
            "status": "insufficient_data", "arm": arm, "horizon_days": hz,
            "reason": "no thinktank_coverage shadow signals found",
            "n_shadow_dates": 0, "n_eval_dates": 0,
        }

    scores = pd.DataFrame(rows)
    series_start = min(shadow_dates)
    fwd = _load_forward_alpha(conn, hz)
    if fwd is None or fwd.empty:
        return {
            "status": "insufficient_data", "arm": arm, "horizon_days": hz,
            "reason": "no realized forward alpha in universe_returns",
            "n_shadow_dates": len(shadow_dates), "series_start": series_start,
            "n_eval_dates": 0,
        }
    merged = scores.merge(
        fwd[["eval_date", "ticker", "alpha"]], on=["eval_date", "ticker"], how="inner"
    )
    if merged.empty:
        return {
            "status": "insufficient_data", "arm": arm, "horizon_days": hz,
            "reason": "no shadow cycle has realized 21d alpha yet (21d realization lag)",
            "n_shadow_dates": len(shadow_dates), "series_start": series_start,
            "n_eval_dates": 0,
        }
    block = _clustered_ic_block(merged, "score")
    block.update({
        "status": "ok", "arm": arm, "horizon_days": hz,
        "series_start": series_start, "n_shadow_dates": len(shadow_dates),
    })
    return block


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
