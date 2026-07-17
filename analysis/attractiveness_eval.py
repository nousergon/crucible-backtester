"""attractiveness_eval — universe-board attractiveness composite vs realized
forward alpha (config#1389 / config#1392 / config#1398 measurement halves).

Read-only evaluation producer. Measures the weekly universe-board
attractiveness composite (``scanner/universe/history/attractiveness_history.parquet``,
written append-only by crucible-research ``scoring/attractiveness_history.py``)
against realized forward market-relative log-alpha from ``universe_returns``
— the SAME realized-return source and ``(eval_date, ticker)`` exact join the
sibling producers use (``end_to_end._trajectory_forward_ic`` /
``_neutralized_live_forward_ic``). Emits ONE artifact,
``backtest/{date}/attractiveness_eval.json`` (frozen cross-repo schema v2,
``nousergon_lib.contracts`` ``attractiveness_eval``), alongside ``e2e_lift.json``.

What it computes
----------------
1. **Composite IC** — date-clustered Spearman rank-IC of
   ``attractiveness_score`` vs realized forward alpha. Each weekly
   cross-section is ONE observation (weeks-as-N, Grinold-Kahn t-stat) —
   mirrors the de-pseudo-replicated estimator in
   ``end_to_end._cio_layer_attribution`` (``{layer}_date_ic``); the pooled IC
   (+ p) is reported as a secondary, explicitly-inflated reference.
2. **Per-pillar ICs** (config#1389) — the same date-clustered estimator per
   pillar column of the history parquet.
3. **Suggested pillar weights** (config#1389) — IC-proportional weights
   SHRUNK toward the 1/N prior (DeMiguel/Garlappi/Uppal 2009: raw
   sample-moment weights lose to 1/N out-of-sample at small N). Shrinkage
   ``lambda`` (the weight ON the 1/N prior) scales inversely with the number
   of eval dates: ``lambda = min(1, SHRINKAGE_FULL_DATES / n_eval_dates)`` —
   at ``n_eval_dates <= SHRINKAGE_FULL_DATES`` (8) the output IS 1/N.
   These are SUGGESTIONS inside this eval artifact ONLY: this module never
   writes ``config/factor_attractiveness_weights.json`` (the live-edge
   artifact research auto-consumes) or any other live config.
4. **Trajectory forward-IC** (config#1392) — date-clustered IC of the
   persisted trajectory signals (``pre_repricing_score``, ``attr_slope_z``,
   injected via ``end_to_end.load_historical_trajectory_scores``) vs
   realized forward alpha.
5. **Counterfactual** (config#1398) — top-N-by-attractiveness selections
   (N = 60/120/200, sector-balanced + unbalanced) vs the live ``tech_score``
   survivor set (``scanner_evaluations.quant_filter_pass == 1``, the same
   live-gate identification ``end_to_end._scanner_factor_counterfactual``
   uses). Per selection: ex-post winner **capture rate** (per cycle, the
   fraction of the cycle's top-K realized-alpha names — K = selection size,
   winners drawn from the full scanner-evaluated cycle universe — that the
   selection captured, averaged equally across cycles) and **mean realized
   forward alpha** (per-cycle selection means, averaged equally across
   cycles, i.e. date-clustered).

Units + horizon
---------------
All alpha values are DECIMAL log-alpha (0.043 = 4.3%), never percent. The
horizon comes from ``nousergon_lib.quant.horizons.DEFAULT_POLICY``
(primary_horizon = 21) — never hardcoded; the realized-return columns are
derived from it (``log_return_{h}d`` / ``log_spy_return_{h}d``).

Failure posture
---------------
Runs under ``evaluate.py``'s ``tracker.run_module`` error isolation. Expected
data-absence states (history parquet not yet written, no resolved forward
outcomes, trajectory artifacts still accruing) return HONEST
``status="insufficient_data"`` bodies with explicit n-fields — never
fabricated zeros. Unexpected errors RAISE and surface as the tracker's
``error`` module status (the recording surface the evaluator grades
N/A-with-reason). No bare excepts; the only guarded call is the S3 history
read, whose absence is an expected warm-up state (mirrors
``scoring/attractiveness_history.read_history``).
"""

from __future__ import annotations

import io
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.information_coefficient import compute_ic
from nousergon_lib.quant.horizons import DEFAULT_POLICY

logger = logging.getLogger(__name__)

# Frozen cross-repo schema version — bump ONLY with a coordinated consumer
# change (crucible-evaluator is built against exactly this shape). The schema
# now lives in ``nousergon_lib.contracts`` (config#1861 second-adoption lift);
# v2 renamed the counterfactual ``mean_alpha_21d`` field to ``mean_alpha``
# (horizon-is-a-parameter, ARCHITECTURE §48 / config#1483 — the horizon is
# carried by the top-level ``horizon_days``, never hardcoded in the field name).
SCHEMA_VERSION = 2

# Producer key of the attractiveness history parquet (crucible-research
# scoring/attractiveness_history.HISTORY_KEY — schema authority).
HISTORY_KEY = "scanner/universe/history/attractiveness_history.parquet"

# Non-pillar columns of the history parquet. Pillar columns are discovered
# dynamically as "everything else" so a research-side pillar add/rename flows
# through without a lockstep edit here (the artifact keys are the pillar
# names themselves).
_HISTORY_META_COLS = frozenset(
    {"as_of", "ticker", "attractiveness_raw", "attractiveness_score",
     "sector", "industry"}
)

# Per-week cross-section floor for a date to count as one IC observation.
# Deliberately above the degenerate-Spearman floor (5) used for tiny CIO
# cohorts — the scanned universe is ~900 names, so a week with <10 joined
# names is a data problem, not a cohort.
MIN_NAMES_PER_DATE = 10

# Minimum weekly observations for a date-clustered t-test — mirrors
# end_to_end._cio_layer_attribution's >= 3 floor.
MIN_EVAL_DATES_T = 3

# 1/N shrinkage: at or below this many eval dates the suggested weights ARE
# the 1/N prior (lambda = 1). Above it, lambda = SHRINKAGE_FULL_DATES / n.
SHRINKAGE_FULL_DATES = 8

# Counterfactual top-N variants (config#1398).
COUNTERFACTUAL_TOP_NS = (60, 120, 200)

# Trajectory signals measured (schema authority: crucible-research
# scoring/attractiveness_trajectory.build_trajectory stocks[] records).
_TRAJECTORY_SIGNALS = ("pre_repricing_score", "attr_slope_z")


# ── Loaders ──────────────────────────────────────────────────────────────────


def load_attractiveness_history(
    bucket: str, *, s3_client=None
) -> pd.DataFrame | None:
    """Read the append-only attractiveness history parquet → DataFrame, or
    ``None`` when absent/unreadable (an expected warm-up state — the producer
    only started writing 2026-05-21). Mirrors crucible-research
    ``scoring/attractiveness_history.read_history``."""
    try:
        import boto3

        s3 = s3_client or boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=HISTORY_KEY)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()), engine="pyarrow")
    except Exception as e:  # expected-absence guard, WARN-logged (no silent swallow)
        logger.warning(
            "attractiveness history parquet unavailable at s3://%s/%s "
            "(expected during warm-up): %s", bucket, HISTORY_KEY, e,
        )
        return None


def _load_forward_alpha(conn, horizon_days: int) -> pd.DataFrame | None:
    """``universe_returns`` → per-(eval_date, ticker) realized forward
    market-relative log-alpha at the policy horizon. Returns None when the
    table/columns are absent (thin research.db)."""
    ret_col = f"log_return_{horizon_days}d"
    spy_col = f"log_spy_return_{horizon_days}d"
    tabs = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    if "universe_returns" not in tabs:
        return None
    ur_cols = {r[1] for r in conn.execute("PRAGMA table_info(universe_returns)")}
    if not {ret_col, spy_col}.issubset(ur_cols):
        return None
    return pd.read_sql_query(
        f"SELECT ticker, eval_date, sector, ({ret_col} - {spy_col}) AS alpha "
        f"FROM universe_returns "
        f"WHERE {ret_col} IS NOT NULL AND {spy_col} IS NOT NULL",
        conn,
    )


# ── Date-clustered IC estimator ──────────────────────────────────────────────


def _per_date_ics(df: pd.DataFrame, score_col: str, alpha_col: str = "alpha") -> list[float]:
    """One cross-sectional Spearman rank-IC per eval_date (weeks-as-N), via
    the shared lib IC engine — a date counts only with >= MIN_NAMES_PER_DATE
    valid pairs (compute_ic also rejects degenerate/constant cross-sections)."""
    ics: list[float] = []
    for _d, g in df.dropna(subset=[score_col, alpha_col]).groupby("eval_date"):
        if len(g) < MIN_NAMES_PER_DATE:
            continue
        res = compute_ic(g[score_col], g[alpha_col], min_samples=MIN_NAMES_PER_DATE)
        if res.get("status") == "ok":
            ics.append(float(res["ic"]))
    return ics


def _clustered_ic_block(df: pd.DataFrame, score_col: str) -> dict:
    """Full IC block: date-clustered mean/t/p (each weekly cross-section is
    ONE observation, Grinold-Kahn style — mirrors
    end_to_end._cio_layer_attribution's ``{layer}_date_ic`` math) + pooled
    IC/p as the secondary reference. Honest small-N: with fewer than
    MIN_EVAL_DATES_T weekly observations the t/p are None, never fabricated."""
    from scipy.stats import ttest_1samp

    ics = _per_date_ics(df, score_col)
    n_dates = len(ics)
    block: dict = {
        "date_ic_mean": round(float(np.mean(ics)), 4) if ics else None,
        "date_ic_t": None,
        "date_ic_p": None,
        "n_eval_dates": n_dates,
    }
    if n_dates >= MIN_EVAL_DATES_T:
        t_stat, p_val = ttest_1samp(ics, 0.0)
        # np.isfinite (not just NaN-check): a zero-variance IC series (e.g.
        # perfectly rank-aligned synthetic data) yields t=inf, which is not
        # valid strict JSON — report None rather than corrupt the artifact.
        block["date_ic_t"] = round(float(t_stat), 4) if np.isfinite(t_stat) else None
        block["date_ic_p"] = round(float(p_val), 4) if np.isfinite(p_val) else None

    pooled = df.dropna(subset=[score_col, "alpha"])
    pooled_res = compute_ic(pooled[score_col], pooled["alpha"],
                            min_samples=MIN_NAMES_PER_DATE)
    if pooled_res.get("status") == "ok":
        block["pooled_ic"] = round(float(pooled_res["ic"]), 4)
        p = pooled_res.get("p_value")
        block["pooled_ic_p"] = round(float(p), 4) if p is not None and p == p else None
        block["n"] = int(pooled_res["n"])
    else:
        block["pooled_ic"] = None
        block["pooled_ic_p"] = None
        block["n"] = int(len(pooled))
    return block


def _empty_ic_block(n: int = 0) -> dict:
    return {"date_ic_mean": None, "date_ic_t": None, "date_ic_p": None,
            "n_eval_dates": 0, "pooled_ic": None, "pooled_ic_p": None, "n": n}


# ── Suggested pillar weights (config#1389) ───────────────────────────────────


def suggest_pillar_weights(pillar_ic: dict) -> tuple[dict, dict]:
    """IC-proportional pillar weights shrunk toward the 1/N prior
    (DeMiguel/Garlappi/Uppal 2009 — raw IC-proportional weights overfit OOS
    at small N).

    ``w = lambda * (1/N) + (1 - lambda) * w_ic`` where ``w_ic`` is
    proportional to ``max(date_ic_mean, 0)`` (a negative-IC pillar earns no
    IC-proportional mass — its floor is the shrunk prior, mirroring how the
    weight optimizer's guardrails floor live weights) and
    ``lambda = min(1, SHRINKAGE_FULL_DATES / n_eval_dates)`` — at
    ``n_eval_dates <= SHRINKAGE_FULL_DATES`` the output IS exactly 1/N.
    ``n_eval_dates`` is the minimum observation count across pillars with a
    computed IC (the binding sample size). All-nonpositive / all-missing ICs
    degrade to pure 1/N (lambda reported as 1.0).

    Returns ``(weights, shrinkage_block)``; weights sum to 1.0.
    """
    pillars = list(pillar_ic)
    n = len(pillars)
    if n == 0:
        return {}, {"method": "demiguel_1overN", "lambda": 1.0, "n_eval_dates": 0}

    ic_means = {p: pillar_ic[p].get("date_ic_mean") for p in pillars}
    observed = [pillar_ic[p]["n_eval_dates"] for p in pillars
                if ic_means[p] is not None]
    n_eval_dates = min(observed) if observed else 0

    lam = 1.0 if n_eval_dates <= SHRINKAGE_FULL_DATES else SHRINKAGE_FULL_DATES / n_eval_dates
    clipped = {p: max(float(ic_means[p]), 0.0) if ic_means[p] is not None else 0.0
               for p in pillars}
    total = sum(clipped.values())
    if total <= 0.0:
        lam = 1.0  # nothing to tilt toward — pure prior, reported honestly
        w_ic = {p: 1.0 / n for p in pillars}
    else:
        w_ic = {p: v / total for p, v in clipped.items()}

    weights = {p: lam * (1.0 / n) + (1.0 - lam) * w_ic[p] for p in pillars}
    norm = sum(weights.values())  # exact 1.0 up to float error; renormalize anyway
    weights = {p: round(w / norm, 4) for p, w in weights.items()}
    shrinkage = {
        "method": "demiguel_1overN",
        "lambda": round(lam, 4),
        "n_eval_dates": int(n_eval_dates),
    }
    return weights, shrinkage


# ── Counterfactual (config#1398) ─────────────────────────────────────────────


def _sector_balanced_top_n(g: pd.DataFrame, score_col: str, n: int) -> pd.DataFrame:
    """Top-``n`` by ``score_col`` with sector-proportional allocation
    (largest-remainder rounding over the cycle universe's sector counts;
    quota deficits — sectors with fewer scored names than their quota —
    backfilled globally by score). ``g`` must carry ``sector`` (NaN →
    "Unknown", which degrades gracefully to the unbalanced selection when
    sector data is absent)."""
    g = g.copy()
    g["sector"] = g["sector"].fillna("Unknown")
    counts = g["sector"].value_counts()
    total = int(counts.sum())
    if total == 0:
        return g.head(0)
    exact = counts * n / total
    quota = exact.astype(int)
    remainder = (exact - quota).sort_values(ascending=False)
    for s in remainder.index[: n - int(quota.sum())]:
        quota[s] += 1

    picks = []
    for s, q in quota.items():
        if q <= 0:
            continue
        sub = g[(g["sector"] == s) & g[score_col].notna()]
        picks.append(sub.sort_values(score_col, ascending=False).head(int(q)))
    sel = pd.concat(picks) if picks else g.head(0)
    if len(sel) < n:  # backfill under-filled quotas globally by score
        rest = g[g[score_col].notna() & ~g.index.isin(sel.index)]
        sel = pd.concat([sel, rest.sort_values(score_col, ascending=False)
                        .head(n - len(sel))])
    return sel


def _capture_rate(cycle: pd.DataFrame, selection: pd.DataFrame) -> float | None:
    """Ex-post winner capture for ONE cycle: fraction of the cycle's top-K
    realized-alpha names (K = selection size, winners drawn from the full
    cycle universe) present in the selection."""
    k = len(selection)
    if k == 0:
        return None
    winners = set(cycle.sort_values("alpha", ascending=False).head(k)["ticker"])
    return len(winners & set(selection["ticker"])) / float(k)


def _counterfactual(conn, merged: pd.DataFrame, ur: pd.DataFrame) -> dict:
    """Top-N-by-attractiveness vs the live tech_score survivor gate
    (config#1398). The live survivor set is ``scanner_evaluations`` rows with
    ``quant_filter_pass == 1`` — the same identification
    ``end_to_end._scanner_factor_counterfactual`` uses. A cycle counts when it
    has a live pass set AND attractiveness coverage, both with realized
    forward alpha. Per-cycle metrics are averaged equally across cycles
    (date-clustered means, never pooled names)."""
    tabs = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    empty = {"top_n": [], "live_gate": {"capture_rate": None,
                                        "mean_alpha": None,
                                        "n_survivors": 0},
             "n_cycles": 0}
    if "scanner_evaluations" not in tabs:
        empty["reason"] = "scanner_evaluations table absent"
        return empty
    se_cols = {r[1] for r in conn.execute("PRAGMA table_info(scanner_evaluations)")}
    if "quant_filter_pass" not in se_cols:
        empty["reason"] = "scanner_evaluations has no quant_filter_pass"
        return empty

    se = pd.read_sql_query(
        "SELECT ticker, eval_date, quant_filter_pass FROM scanner_evaluations",
        conn,
    )
    # Cycle universe = ALL scanner-evaluated names with realized alpha (the
    # same base _scanner_factor_counterfactual uses — winners and the live
    # gate must not be restricted to history-covered names); the
    # attractiveness score left-joins on (eval_date, ticker) where the
    # history parquet covers the cycle.
    attr = merged[["eval_date", "ticker", "attractiveness_score"]]
    ua = ur[["eval_date", "ticker", "alpha", "sector"]].drop_duplicates(
        subset=["eval_date", "ticker"])
    base = se.merge(ua, on=["eval_date", "ticker"], how="inner")
    base = base.merge(attr, on=["eval_date", "ticker"], how="left")

    variants = [(n, sb) for n in COUNTERFACTUAL_TOP_NS for sb in (False, True)]
    per_variant: dict = {v: {"capture": [], "alpha": []} for v in variants}
    live = {"capture": [], "alpha": [], "n_survivors": 0}
    n_cycles = 0

    for _d, g in base.groupby("eval_date"):
        g = g[g["alpha"].notna()]
        survivors = g[g["quant_filter_pass"] == 1]
        scored = g[g["attractiveness_score"].notna()]
        if survivors.empty or len(scored) < MIN_NAMES_PER_DATE:
            continue
        n_cycles += 1

        live["capture"].append(_capture_rate(g, survivors))
        live["alpha"].append(float(survivors["alpha"].mean()))
        live["n_survivors"] += int(len(survivors))

        for n, sector_balanced in variants:
            if len(scored) < n:
                continue  # a truncated "top-N" wouldn't be the named variant
            if sector_balanced:
                sel = _sector_balanced_top_n(scored, "attractiveness_score", n)
            else:
                sel = scored.sort_values(
                    "attractiveness_score", ascending=False).head(n)
            per_variant[(n, sector_balanced)]["capture"].append(_capture_rate(g, sel))
            per_variant[(n, sector_balanced)]["alpha"].append(float(sel["alpha"].mean()))

    def _mean(xs: list) -> float | None:
        vals = [x for x in xs if x is not None]
        return round(float(np.mean(vals)), 5) if vals else None

    top_n = []
    for (n, sector_balanced) in variants:
        v = per_variant[(n, sector_balanced)]
        top_n.append({
            "n": n,
            "sector_balanced": sector_balanced,
            "capture_rate": _mean(v["capture"]),
            "mean_alpha": _mean(v["alpha"]),
            "n_cycles": len(v["alpha"]),
        })
    return {
        "top_n": top_n,
        "live_gate": {
            "capture_rate": _mean(live["capture"]),
            "mean_alpha": _mean(live["alpha"]),
            "n_survivors": int(live["n_survivors"]),
        },
        "n_cycles": n_cycles,
    }


# ── Trajectory forward-IC (config#1392) ──────────────────────────────────────


def _trajectory_ic(ur: pd.DataFrame, trajectory_scores: dict | None) -> dict:
    """Date-clustered IC of the persisted trajectory signals vs realized
    forward alpha. ``trajectory_scores``: ``{(eval_date, ticker): {signal:
    value}}`` from ``end_to_end.load_historical_trajectory_scores`` (the
    exact weekly ``scanner/universe/trajectory/{date}/trajectory.json``
    artifacts). Absent/immature history → honest ``accruing`` blocks."""
    if not trajectory_scores:
        return {
            s: {**_empty_ic_block(), "status": "accruing",
                "reason": "no persisted trajectory artifacts available"}
            for s in _TRAJECTORY_SIGNALS
        }
    recs = [
        {"eval_date": d, "ticker": str(t),
         **{s: scores.get(s) for s in _TRAJECTORY_SIGNALS}}
        for (d, t), scores in trajectory_scores.items()
    ]
    traj = pd.DataFrame.from_records(recs).merge(
        ur[["eval_date", "ticker", "alpha"]], on=["eval_date", "ticker"],
        how="inner",
    )
    out: dict = {}
    for s in _TRAJECTORY_SIGNALS:
        if traj.empty or traj[s].notna().sum() == 0:
            out[s] = {**_empty_ic_block(), "status": "accruing",
                      "reason": "no trajectory names with a realized forward outcome yet"}
            continue
        block = _clustered_ic_block(traj, s)
        block["status"] = ("ok" if block["n_eval_dates"] >= MIN_EVAL_DATES_T
                           else "accruing")
        out[s] = block
    return out


# ── Entry point ──────────────────────────────────────────────────────────────


def compute_attractiveness_eval(
    research_db_path: str | None,
    *,
    as_of: str,
    bucket: str | None = None,
    history_df: pd.DataFrame | None = None,
    trajectory_scores: dict | None = None,
    s3_client=None,
) -> dict:
    """Build the ``attractiveness_eval.json`` artifact (frozen schema v2 —
    ``nousergon_lib.contracts`` ``attractiveness_eval``).

    Args:
        research_db_path: research.db (``universe_returns`` +
            ``scanner_evaluations``).
        as_of: run date (``YYYY-MM-DD``) stamped on the artifact.
        bucket: S3 bucket for the history parquet read (ignored when
            ``history_df`` is injected).
        history_df: injectable attractiveness history frame (tests).
        trajectory_scores: ``{(eval_date, ticker): {signal: value}}`` — the
            dict evaluate.py already loads via
            ``end_to_end.load_historical_trajectory_scores`` (reused, no
            second S3 read).
        s3_client: injectable boto3 client (tests).
    """
    horizon_days = int(DEFAULT_POLICY.primary_horizon)

    def _base(status: str, reason: str | None = None) -> dict:
        art = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "as_of": as_of,
            "horizon_days": horizon_days,
            "composite_ic": _empty_ic_block(),
            "pillar_ic": {},
            "suggested_pillar_weights": {},
            "shrinkage": {"method": "demiguel_1overN", "lambda": 1.0,
                          "n_eval_dates": 0},
            "trajectory_ic": {
                s: {**_empty_ic_block(), "status": "accruing"}
                for s in _TRAJECTORY_SIGNALS
            },
            "counterfactual": {
                "top_n": [],
                "live_gate": {"capture_rate": None, "mean_alpha": None,
                              "n_survivors": 0},
                "n_cycles": 0,
            },
        }
        if reason:
            art["reason"] = reason
        return art

    if not research_db_path or not Path(research_db_path).exists():
        return _base("insufficient_data", "research.db not available")

    if history_df is None:
        if not bucket:
            return _base("insufficient_data",
                         "no attractiveness history (no bucket configured)")
        history_df = load_attractiveness_history(bucket, s3_client=s3_client)
    if history_df is None or history_df.empty:
        return _base("insufficient_data",
                     "attractiveness history parquet absent/empty (warm-up)")

    required = {"as_of", "ticker", "attractiveness_score"}
    if not required.issubset(history_df.columns):
        # Producer-side schema drift — a real contract break, not warm-up.
        raise ValueError(
            "attractiveness history parquet missing required columns "
            f"{sorted(required - set(history_df.columns))} — schema authority "
            "is crucible-research scoring/attractiveness_history.py"
        )

    conn = sqlite3.connect(research_db_path)
    try:
        ur = _load_forward_alpha(conn, horizon_days)
        if ur is None:
            return _base(
                "insufficient_data",
                f"universe_returns absent or lacks log_return_{horizon_days}d "
                f"/ log_spy_return_{horizon_days}d",
            )
        if ur.empty:
            return _base("insufficient_data",
                         "no universe_returns rows with realized forward outcomes")

        hist = history_df.rename(columns={"as_of": "eval_date"})
        hist["ticker"] = hist["ticker"].astype(str)
        merged = hist.merge(
            ur[["eval_date", "ticker", "alpha", "sector"]],
            on=["eval_date", "ticker"], how="inner", suffixes=("_hist", ""),
        )
        # Prefer universe_returns sector (the sibling counterfactual's
        # source); fall back to the history parquet's own sector column.
        if "sector_hist" in merged.columns:
            merged["sector"] = merged["sector"].fillna(merged["sector_hist"])

        composite_ic = (_clustered_ic_block(merged, "attractiveness_score")
                        if not merged.empty else _empty_ic_block())

        pillar_cols = [c for c in history_df.columns if c not in _HISTORY_META_COLS]
        pillar_ic: dict = {}
        for p in pillar_cols:
            if merged.empty or merged[p].notna().sum() == 0:
                pillar_ic[p] = {"date_ic_mean": None, "date_ic_p": None,
                                "n_eval_dates": 0}
                continue
            ics = _per_date_ics(merged, p)
            entry: dict = {
                "date_ic_mean": round(float(np.mean(ics)), 4) if ics else None,
                "date_ic_p": None,
                "n_eval_dates": len(ics),
            }
            if len(ics) >= MIN_EVAL_DATES_T:
                from scipy.stats import ttest_1samp

                _t, p_val = ttest_1samp(ics, 0.0)
                entry["date_ic_p"] = round(float(p_val), 4) if np.isfinite(p_val) else None
            pillar_ic[p] = entry

        weights, shrinkage = suggest_pillar_weights(pillar_ic)
        trajectory_ic = _trajectory_ic(ur, trajectory_scores)
        counterfactual = _counterfactual(conn, merged, ur)

        status = ("ok" if composite_ic["n_eval_dates"] >= MIN_EVAL_DATES_T
                  else "insufficient_data")
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "as_of": as_of,
            "horizon_days": horizon_days,
            "composite_ic": composite_ic,
            "pillar_ic": pillar_ic,
            "suggested_pillar_weights": weights,
            "shrinkage": shrinkage,
            "trajectory_ic": trajectory_ic,
            "counterfactual": counterfactual,
        }
        if status != "ok":
            artifact["reason"] = (
                f"only {composite_ic['n_eval_dates']} weekly cross-sections "
                f"with resolved {horizon_days}d outcomes joined to the "
                f"attractiveness history (need >= {MIN_EVAL_DATES_T})"
            )
        return artifact
    finally:
        conn.close()
