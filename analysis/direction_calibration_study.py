"""
direction_calibration_study.py — conviction-threshold calibration study for the
predictor's 21d direction call (Refs nousergon/alpha-engine-config#12388).

OFFLINE / SHADOW ONLY. Reads ``predictor_outcomes`` (research.db) and the
per-day ``predictor/predictions/{date}.json`` files and emits one JSON. It
writes nothing to S3 or config, and nothing it computes is consumed by a live
rule: the output feeds a human decision (see the decision table in the issue).

Pre-registered design (issue #12388, scoping comment 2026-10-18):

  Population  predictor_outcomes, horizon_days = 21, resolved, direction in
              {UP, DOWN}, model_version = ``meta-v3.0-8models``, prediction
              date in the window. Outcome ``y = 1[actual_log_alpha > 0]``
              (beat SPY over 21d; binary, no FLAT band).
  Conviction  signed ``c = p_up`` for UP calls, ``1 - p_up`` for DOWN calls.
              An unsigned ``prediction_confidence`` threshold measures
              conviction the wrong way on the ~10.8% of calls whose direction
              (sign of alpha) disagrees with their own p_up.
  M1          precision lift at ``t*`` over the covered rows' base rate, UP and
              DOWN separately, 90% moving-block bootstrap CI over whole dates
              (21-date blocks), coverage reported alongside.
  M2          precision by signed-conviction quintile, Spearman(bucket, prec).
  M3          reliability (8 fixed bins), ECE10, Brier, BSS vs trailing
              climatology (expanding base rate, lagged 30 calendar days), ECE
              judged against a perfect-calibration null with a date-level common
              shock rather than against 0.
  M4          per-date spread P(beat|UP) - P(beat|DOWN) and per-date rank IC of
              predicted_alpha vs realised alpha (context, base-rate free).

Interpretation choices the issue left open (stated so the freeze comment can
ratify or override them; each is a parameter or a documented rule here):

  * "Base rate of the rows t* covers" is the DATE-MATCHED base rate: for each
    covered row, the share of ALL served rows on its prediction date with the
    same outcome the call predicts (y = 1 for UP, y = 0 for DOWN). It removes
    day-level market drift from the lift without conditioning on the call.
  * Coverage is the share of that direction's calls kept (>= 20% required).
  * ``t*`` search runs over a fixed grid and returns the smallest qualifying
    value, exactly as written. When the smallest qualifier is the grid floor
    (covers every call) the result carries ``threshold_non_binding = True``:
    the pre-registered test would then be "does the unconditional call clear
    +3pp", not a conviction-threshold test. That is surfaced, not silently
    accepted.

Confirmation guard: a window that extends past ``DISCOVERY_END`` cannot be read
unless ``t_star`` (frozen on the discovery window) is supplied, so the
confirmation read is a single test of a fixed hypothesis.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from nousergon_lib.quant.stats.calibration import expected_calibration_error
from scipy.special import ndtri

logger = logging.getLogger(__name__)

STUDY_HORIZON_DAYS = 21
STUDY_MODEL_VERSION = "meta-v3.0-8models"
BINARY_ERA_START = "2026-07-06"  # config#1815: direction = sign(alpha), no FLAT class
POST_CUTOVER_START = "2026-05-09"  # canonical cutover (sensitivity window)
DISCOVERY_END = "2026-09-17"
CONFIRMATION_START = "2026-09-18"
CONFIRMATION_END = "2026-11-30"

MIN_LIFT = 0.03
MIN_COVERAGE = 0.20
CI_ALPHA = 0.10  # 90% two-sided
BLOCK_DATES = 21
N_BOOT = 2000
N_RELIABILITY_BINS = 8
CLIMATOLOGY_LAG_DAYS = 30
SHOCK_WINDOW_DAYS = 21
RHO_SENSITIVITY = (0.1, 0.3, 0.5)
ALT_CUT_COVERAGES = (0.2, 0.4, 0.6, 0.8)
DEFAULT_GRID = tuple(round(x, 2) for x in np.arange(0.0, 1.0, 0.01))
DIRECTIONS = ("UP", "DOWN")

PredictionsLoader = Callable[[str], Optional[dict]]


# ── Loading ──────────────────────────────────────────────────────────────────


def load_outcomes(
    db_path: str,
    start: str,
    end: str,
    model_version: str = STUDY_MODEL_VERSION,
) -> pd.DataFrame:
    """Resolved 21d UP/DOWN outcome rows in ``[start, end]`` with signed conviction."""
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT symbol, prediction_date, predicted_direction, "
            "prediction_confidence, p_up, actual_log_alpha "
            "FROM predictor_outcomes "
            "WHERE horizon_days = ? AND actual_log_alpha IS NOT NULL "
            "AND predicted_direction IN ('UP','DOWN') AND model_version = ? "
            "AND prediction_date >= ? AND prediction_date <= ?",
            conn,
            params=(STUDY_HORIZON_DAYS, model_version, start, end),
        )
    finally:
        conn.close()
    return prepare_frame(df)


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add y, signed conviction and the date-matched base rate; drop null p_up."""
    df = df.copy()
    n_in = len(df)
    df = df[df["p_up"].notna()].copy()
    df.attrs["n_dropped_null_p_up"] = n_in - len(df)
    df["y"] = (df["actual_log_alpha"] > 0).astype(int)
    df["c"] = np.where(df["predicted_direction"] == "UP", df["p_up"], 1.0 - df["p_up"])
    df["hit"] = np.where(df["predicted_direction"] == "UP", df["y"], 1 - df["y"])
    date_beat = df.groupby("prediction_date")["y"].transform("mean")
    df["base"] = np.where(df["predicted_direction"] == "UP", date_beat, 1.0 - date_beat)
    return df.sort_values(["prediction_date", "symbol"]).reset_index(drop=True)


def attach_predictions(df: pd.DataFrame, loader: PredictionsLoader) -> pd.DataFrame:
    """Left-join ``predicted_alpha`` / ``p_up_platt`` / ``calibration_basis`` by (symbol, date)."""
    rows = []
    for d in sorted(df["prediction_date"].unique()):
        payload = loader(d)
        if not payload:
            continue
        for p in payload.get("predictions", []):
            rows.append(
                {
                    "symbol": p.get("ticker"),
                    "prediction_date": d,
                    "predicted_alpha": p.get("predicted_alpha"),
                    "p_up_platt": p.get("p_up_platt"),
                    "calibration_basis": p.get("calibration_basis"),
                }
            )
    out = df.copy()
    if not rows:
        for col in ("predicted_alpha", "p_up_platt", "calibration_basis"):
            out[col] = np.nan
        out.attrs["join_match_rate"] = 0.0
        return out
    pj = pd.DataFrame(rows).drop_duplicates(["symbol", "prediction_date"])
    out = out.merge(pj, on=["symbol", "prediction_date"], how="left")
    out.attrs["join_match_rate"] = float(out["predicted_alpha"].notna().mean())
    out.attrs["n_dropped_null_p_up"] = df.attrs.get("n_dropped_null_p_up", 0)
    return out


def directory_loader(path: str) -> PredictionsLoader:
    import os

    def _load(date: str) -> Optional[dict]:
        fp = os.path.join(path, f"{date}.json")
        if not os.path.exists(fp):
            return None
        with open(fp) as fh:
            return json.load(fh)

    return _load


def s3_loader(bucket: str = "alpha-engine-research", prefix: str = "predictor/predictions/") -> PredictionsLoader:
    import boto3
    from botocore.exceptions import ClientError

    client = boto3.client("s3")

    def _load(date: str) -> Optional[dict]:
        try:
            obj = client.get_object(Bucket=bucket, Key=f"{prefix}{date}.json")
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise
        return json.loads(obj["Body"].read())

    return _load


# ── Block bootstrap over whole dates ─────────────────────────────────────────


def block_bootstrap(
    date_stats: np.ndarray,
    stat_fn: Callable[[np.ndarray], np.ndarray],
    block_len: int = BLOCK_DATES,
    n_boot: int = N_BOOT,
    seed: int = 0,
    alpha: float = CI_ALPHA,
) -> dict:
    """Moving-block bootstrap over date-level sufficient statistics.

    ``date_stats`` is (D, k): one row per prediction date, in date order, of
    additive sufficient statistics. ``stat_fn`` maps summed stats ``(B, k)`` to
    ``(B,)`` (NaN where undefined). Whole dates are resampled in blocks of
    consecutive dates so within-date and neighbouring-date dependence (shared
    market move, overlapping 21d windows) is kept inside a block.
    """
    d = date_stats.shape[0]
    point = float(stat_fn(date_stats.sum(axis=0, keepdims=True))[0]) if d else float("nan")
    if d < 2:
        return {"point": point, "lo": None, "hi": None, "n_dates": d, "block_len": 0, "n_valid": 0}
    b = max(1, min(block_len, d))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(d / b))
    starts = rng.integers(0, d - b + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(n_boot, -1)[:, :d]
    sums = date_stats[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = stat_fn(sums)
    vals = vals[np.isfinite(vals)]
    if len(vals) < max(10, n_boot // 10):
        return {"point": point, "lo": None, "hi": None, "n_dates": d, "block_len": b, "n_valid": int(len(vals))}
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "n_dates": d,
        "block_len": b,
        "n_valid": int(len(vals)),
    }


def _per_date(df: pd.DataFrame, cols: dict[str, Callable[[pd.DataFrame], pd.Series]], all_dates) -> np.ndarray:
    """(D, k) per-date sums of the given column builders, zero-filled to ``all_dates``."""
    parts = {name: fn(df) for name, fn in cols.items()}
    g = pd.DataFrame(parts).groupby(df["prediction_date"]).sum()
    return g.reindex(all_dates).fillna(0.0).to_numpy(dtype=float)


# ── M1: precision lift at a threshold ────────────────────────────────────────


def lift_at(df: pd.DataFrame, direction: str, t: float) -> dict:
    """Point precision / date-matched base / lift / coverage for one direction at ``t``."""
    sub = df[df["predicted_direction"] == direction]
    cov = sub[sub["c"] >= t]
    n = len(cov)
    if n == 0:
        return {"t": t, "n": 0, "n_direction_calls": len(sub), "coverage": 0.0, "precision": None, "base": None, "lift": None}
    prec = float(cov["hit"].mean())
    base = float(cov["base"].mean())
    return {
        "t": t,
        "n": n,
        "n_direction_calls": len(sub),
        "coverage": n / len(sub),
        "precision": prec,
        "base": base,
        "lift": prec - base,
    }


def lift_curve(df: pd.DataFrame, direction: str, grid=DEFAULT_GRID) -> list[dict]:
    return [lift_at(df, direction, t) for t in grid]


def select_t_star(
    df: pd.DataFrame,
    direction: str,
    grid=DEFAULT_GRID,
    min_lift: float = MIN_LIFT,
    min_coverage: float = MIN_COVERAGE,
) -> dict:
    """Pre-registered rule: smallest grid threshold with lift >= ``min_lift`` and
    coverage >= ``min_coverage``; else the coverage-``min_coverage`` quantile of c.
    """
    sub = df[df["predicted_direction"] == direction]
    curve = lift_curve(df, direction, grid)
    for row in curve:
        if row["lift"] is not None and row["lift"] >= min_lift and row["coverage"] >= min_coverage:
            return {
                "direction": direction,
                "t_star": row["t"],
                "rule": "smallest_qualifying",
                "threshold_non_binding": bool(row["coverage"] >= 1.0 - 1e-12),
                "discovery_point": row,
                "curve": curve,
            }
    if sub.empty:
        return {"direction": direction, "t_star": None, "rule": "no_calls", "threshold_non_binding": False, "discovery_point": None, "curve": curve}
    t = float(np.quantile(sub["c"], 1.0 - min_coverage))
    return {
        "direction": direction,
        "t_star": t,
        "rule": "coverage_quantile_fallback",
        "threshold_non_binding": False,
        "discovery_point": lift_at(df, direction, t),
        "curve": curve,
    }


def m1_lift_ci(df: pd.DataFrame, direction: str, t: float, all_dates, seed: int = 0) -> dict:
    """Block-bootstrap CI for precision lift at ``t`` (UP or DOWN)."""
    sub = df[(df["predicted_direction"] == direction) & (df["c"] >= t)]
    ds = _per_date(
        sub,
        {"n": lambda x: pd.Series(1.0, index=x.index), "hit": lambda x: x["hit"].astype(float), "base": lambda x: x["base"]},
        all_dates,
    )

    def lift_fn(s):
        return s[:, 1] / s[:, 0] - s[:, 2] / s[:, 0]

    out = block_bootstrap(ds, lift_fn, seed=seed)
    point = lift_at(df, direction, t)
    out.update({"direction": direction, "t": t, "n": point["n"], "coverage": point["coverage"], "precision": point["precision"], "base": point["base"]})
    out["meets_rule"] = bool(
        out["lo"] is not None and point["lift"] is not None and point["lift"] >= MIN_LIFT and out["lo"] > 0 and point["coverage"] >= MIN_COVERAGE
    )
    return out


def m1_alternative_cuts(df: pd.DataFrame, direction: str, all_dates, seed: int = 0) -> list[dict]:
    """DESCRIPTIVE, not the pre-registered test: lift + CI at fixed coverage
    quantiles of signed conviction. Shows what a binding threshold would look
    like when the smallest-qualifying rule is non-binding."""
    sub = df[df["predicted_direction"] == direction]
    if sub.empty:
        return []
    rows = []
    for cov in ALT_CUT_COVERAGES:
        t = float(np.quantile(sub["c"], 1.0 - cov))
        row = m1_lift_ci(df, direction, t, all_dates, seed=seed)
        row["target_coverage"] = cov
        rows.append(row)
    return rows


# ── M2: precision by signed-conviction quintile ──────────────────────────────


def _spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 3:
        return None
    rx, ry = pd.Series(x).rank(), pd.Series(y).rank()
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(rx.corr(ry))


def m2_buckets(df: pd.DataFrame, direction: str, edges: Optional[list[float]] = None) -> dict:
    """Precision per signed-conviction quintile. ``edges`` are the 4 interior cut
    points; when omitted they are the window's own 20/40/60/80% quantiles."""
    sub = df[df["predicted_direction"] == direction]
    if sub.empty:
        return {"direction": direction, "edges": edges, "buckets": [], "spearman": None}
    if edges is None:
        edges = [float(q) for q in np.quantile(sub["c"], [0.2, 0.4, 0.6, 0.8])]
    bucket = np.digitize(sub["c"].to_numpy(), edges)  # 0..4
    buckets = []
    for b in range(len(edges) + 1):
        m = bucket == b
        n = int(m.sum())
        buckets.append(
            {
                "bucket": b + 1,
                "n": n,
                "c_mean": float(sub["c"].to_numpy()[m].mean()) if n else None,
                "precision": float(sub["hit"].to_numpy()[m].mean()) if n else None,
                "base": float(sub["base"].to_numpy()[m].mean()) if n else None,
            }
        )
    filled = [b for b in buckets if b["n"] > 0]
    rho = _spearman(np.array([b["bucket"] for b in filled], dtype=float), np.array([b["precision"] for b in filled], dtype=float))
    return {"direction": direction, "edges": edges, "buckets": buckets, "spearman": rho}


# ── M3: calibration of p_up ──────────────────────────────────────────────────


def trailing_climatology(df: pd.DataFrame, lag_days: int = CLIMATOLOGY_LAG_DAYS) -> pd.Series:
    """Expanding base rate of rows whose prediction date is >= ``lag_days`` before
    each row's date (approximates 'outcome already known'); NaN before any exist."""
    dates = pd.to_datetime(df["prediction_date"])
    daily = df.assign(_d=dates).groupby("_d")["y"].agg(["sum", "count"]).sort_index()
    cs_sum, cs_cnt = daily["sum"].cumsum(), daily["count"].cumsum()
    out = {}
    for d in daily.index:
        cutoff = d - pd.Timedelta(days=lag_days)
        pos = cs_cnt.index.searchsorted(cutoff, side="right") - 1
        out[d] = (cs_sum.iloc[pos] / cs_cnt.iloc[pos]) if pos >= 0 else np.nan
    return dates.map(out).astype(float).set_axis(df.index)


def _date_icc(resid: np.ndarray, codes: np.ndarray) -> float:
    """One-way ANOVA intraclass correlation of ``resid`` grouped by ``codes``."""
    k = codes.max() + 1
    n_g = np.bincount(codes, minlength=k).astype(float)
    keep = n_g > 0
    sum_g = np.bincount(codes, weights=resid, minlength=k)
    n, groups = n_g.sum(), keep.sum()
    if groups < 2 or n <= groups:
        return 0.0
    grand = resid.sum() / n
    ssb = (n_g[keep] * (sum_g[keep] / n_g[keep] - grand) ** 2).sum()
    ssw = (resid**2).sum() - (sum_g[keep] ** 2 / n_g[keep]).sum()
    msb, msw = ssb / (groups - 1), ssw / (n - groups)
    n0 = (n - (n_g[keep] ** 2).sum() / n) / (groups - 1)
    return float(max(0.0, (msb - msw) / (msb + (n0 - 1) * msw))) if (msb + (n0 - 1) * msw) > 0 else 0.0


def _simulate_outcomes(p: np.ndarray, tpos: np.ndarray, rho: float, rng: np.random.Generator, window: int) -> np.ndarray:
    """y ~ Bernoulli(p) marginally, with a common shock per trading day that is a
    ``window``-day moving sum of daily innovations (overlapping 21d outcome windows)."""
    lo, hi = tpos.min(), tpos.max()
    innov = rng.standard_normal(hi - lo + window)
    cs = np.concatenate([[0.0], np.cumsum(innov)])
    shock = (cs[window:] - cs[:-window])[: hi - lo + 1] / np.sqrt(window)
    z = shock[tpos - lo]
    latent = np.sqrt(rho) * z + np.sqrt(1.0 - rho) * rng.standard_normal(len(p))
    return (latent < ndtri(np.clip(p, 1e-6, 1 - 1e-6))).astype(int)


def ece_null(
    p: np.ndarray,
    y_obs: np.ndarray,
    dates: pd.Series,
    n_sims: int = 500,
    seed: int = 0,
    window: int = SHOCK_WINDOW_DAYS,
) -> dict:
    """ECE10 distribution under PERFECT calibration with a date-level common shock.

    The latent-shock share ``rho`` is the grid value whose simulated date-level
    intraclass correlation of (y - p) best matches the observed one, so the null
    carries the correlation structure the real rows have rather than assuming
    independence (which would make ECE look too significant).
    """
    d = pd.to_datetime(dates).dt.strftime("%Y-%m-%d").to_numpy()
    codes = pd.factorize(d)[0]
    tpos = np.busday_count(np.datetime64("2000-01-03"), d.astype("datetime64[D]"))
    rng = np.random.default_rng(seed)
    icc_obs = _date_icc(y_obs - p, codes)
    best_rho, best_gap = 0.0, float("inf")
    for rho in np.arange(0.0, 0.951, 0.05):
        sims = [_date_icc(_simulate_outcomes(p, tpos, rho, rng, window) - p, codes) for _ in range(40)]
        gap = abs(float(np.mean(sims)) - icc_obs)
        if gap < best_gap:
            best_rho, best_gap = float(rho), gap
    def _p(rho_: float) -> np.ndarray:
        e = np.empty(n_sims)
        for i in range(n_sims):
            e[i] = expected_calibration_error(p, _simulate_outcomes(p, tpos, rho_, rng, window), n_bins=10)["ece"]
        return e

    eces = _p(best_rho)
    return {
        "rho": round(best_rho, 2),
        "icc_observed": icc_obs,
        "ece_null_p50": float(np.quantile(eces, 0.5)),
        "ece_null_p95": float(np.quantile(eces, 0.95)),
        # The estimated rho is identified only from same-date residual clustering;
        # these show how far the p95 moves if the common shock is stronger.
        "ece_null_p95_at_fixed_rho": {str(r): float(np.quantile(_p(r), 0.95)) for r in RHO_SENSITIVITY},
        "n_sims": n_sims,
    }


def m3_calibration(df: pd.DataFrame, n_sims: int = 500, seed: int = 0) -> dict:
    p, y = df["p_up"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float)
    rel = expected_calibration_error(p, y, n_bins=N_RELIABILITY_BINS)
    ece10 = expected_calibration_error(p, y, n_bins=10)
    brier = float(np.mean((p - y) ** 2))
    clim = trailing_climatology(df)
    ok = clim.notna().to_numpy()
    if ok.sum():
        b_model = float(np.mean((p[ok] - y[ok]) ** 2))
        b_clim = float(np.mean((clim.to_numpy()[ok] - y[ok]) ** 2))
        bss = 1.0 - b_model / b_clim if b_clim > 0 else None
    else:
        b_model = b_clim = bss = None
    null = ece_null(p, y, df["prediction_date"], n_sims=n_sims, seed=seed) if len(df) >= 30 else None
    ece_val = ece10.get("ece")
    # Level vs shape: a market-wide base-rate shift moves every bin at once, and on
    # a few independent 21d windows it cannot be told apart from miscalibration.
    # ECE after removing the global offset isolates the shape (tail) component.
    offset = float(y.mean() - p.mean())
    ece_level_adj = expected_calibration_error(np.clip(p + offset, 0.0, 1.0), y, n_bins=10).get("ece")
    return {
        "n": len(df),
        "mean_p_up": float(p.mean()),
        "mean_p_up_minus_base_rate": float(p.mean() - y.mean()),
        "ece10_after_removing_level_offset": ece_level_adj,
        "reliability_8bin": rel.get("bins"),
        "ece10": ece_val,
        "ece10_null": null,
        "ece_above_null_p95": bool(null is not None and ece_val is not None and ece_val > null["ece_null_p95"]),
        "brier": brier,
        "bss_vs_trailing_climatology": bss,
        "bss_n": int(ok.sum()),
        "brier_model_on_bss_rows": b_model,
        "brier_climatology_on_bss_rows": b_clim,
        "recalibration_trigger": bool(
            (null is not None and ece_val is not None and ece_val > null["ece_null_p95"]) or (bss is not None and bss <= 0)
        ),
    }


# ── M4: base-rate-free context ───────────────────────────────────────────────


def m4_spread_and_ic(df: pd.DataFrame, all_dates, seed: int = 0) -> dict:
    up = (df["predicted_direction"] == "UP").astype(float)
    dn = 1.0 - up
    tmp = pd.DataFrame({"nu": up, "yu": up * df["y"], "nd": dn, "yd": dn * df["y"]})
    ds = tmp.groupby(df["prediction_date"]).sum().reindex(all_dates).fillna(0.0).to_numpy()

    def spread_fn(s):
        return s[:, 1] / s[:, 0] - s[:, 3] / s[:, 2]

    spread = block_bootstrap(ds, spread_fn, seed=seed)

    ic = {"mean": None, "share_dates_positive": None, "n_dates": 0}
    if "predicted_alpha" in df and df["predicted_alpha"].notna().any():
        vals = []
        for _, g in df.dropna(subset=["predicted_alpha"]).groupby("prediction_date"):
            if len(g) >= 5 and g["predicted_alpha"].nunique() > 1 and g["actual_log_alpha"].nunique() > 1:
                vals.append(g["predicted_alpha"].rank().corr(g["actual_log_alpha"].rank()))
        if vals:
            ic = {"mean": float(np.mean(vals)), "share_dates_positive": float(np.mean(np.array(vals) > 0)), "n_dates": len(vals)}
    return {"spread_up_minus_down": spread, "rank_ic": ic}


def continuity_accuracy(df: pd.DataFrame) -> dict:
    """Binary hit rate vs always-majority, for continuity with the Director's headline only."""
    base = float(df["y"].mean())
    majority = max(base, 1.0 - base)
    acc = float(df["hit"].mean())
    return {"accuracy": acc, "majority_baseline": majority, "lift": acc - majority, "base_rate_beat_spy": base}


# ── Orchestration ────────────────────────────────────────────────────────────


@dataclass
class StudyParams:
    start: str = BINARY_ERA_START
    end: str = DISCOVERY_END
    t_star: Optional[dict] = None  # {"UP": float, "DOWN": float}, frozen on discovery
    bucket_edges: Optional[dict] = None  # {"UP": [4 floats], "DOWN": [4 floats]}
    n_sims: int = 500
    seed: int = 0


def run_study(df: pd.DataFrame, params: StudyParams) -> dict:
    """Compute M1-M4 on a prepared frame (see ``prepare_frame`` / ``attach_predictions``)."""
    if params.end > DISCOVERY_END and not params.t_star:
        raise ValueError(
            f"window ends {params.end} > discovery end {DISCOVERY_END}: the confirmation read needs "
            "t_star frozen on the discovery window (single test of a fixed hypothesis)"
        )
    all_dates = sorted(df["prediction_date"].unique())
    out: dict = {
        "issue": "nousergon/alpha-engine-config#12388",
        "mode": "offline-shadow-only",
        "window": {"start": params.start, "end": params.end},
        "population": {
            "n": len(df),
            "n_dates": len(all_dates),
            "n_symbols": int(df["symbol"].nunique()),
            "n_up": int((df["predicted_direction"] == "UP").sum()),
            "n_down": int((df["predicted_direction"] == "DOWN").sum()),
            "base_rate_beat_spy": float(df["y"].mean()) if len(df) else None,
            "n_dropped_null_p_up": df.attrs.get("n_dropped_null_p_up"),
            "predictions_join_match_rate": df.attrs.get("join_match_rate"),
            "share_direction_disagrees_with_p_up": float(
                (((df["predicted_direction"] == "UP") & (df["p_up"] < 0.5)) | ((df["predicted_direction"] == "DOWN") & (df["p_up"] > 0.5))).mean()
            )
            if len(df)
            else None,
        },
        "continuity_accuracy_vs_majority": continuity_accuracy(df) if len(df) else None,
    }
    selection, m1, m2, alt = {}, {}, {}, {}
    for i, d in enumerate(DIRECTIONS):
        if params.t_star and params.t_star.get(d) is not None:
            t = float(params.t_star[d])
            selection[d] = {"t_star": t, "rule": "frozen_input"}
        else:
            selection[d] = select_t_star(df, d)
            t = selection[d]["t_star"]
        m1[d] = m1_lift_ci(df, d, t, all_dates, seed=params.seed + i) if t is not None else None
        edges = (params.bucket_edges or {}).get(d)
        m2[d] = m2_buckets(df, d, edges)
        alt[d] = m1_alternative_cuts(df, d, all_dates, seed=params.seed + i)
    out["t_star_selection"] = selection
    out["M1_precision_lift"] = m1
    out["M1_alternative_cuts_descriptive"] = alt
    out["M2_conviction_buckets"] = m2
    out["M3_calibration"] = m3_calibration(df, n_sims=params.n_sims, seed=params.seed)
    out["M4_context"] = m4_spread_and_ic(df, all_dates, seed=params.seed)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", required=True, help="path to a local research.db snapshot")
    ap.add_argument("--predictions-dir", help="local dir of {date}.json; default reads S3 predictor/predictions/")
    ap.add_argument("--start", default=BINARY_ERA_START)
    ap.add_argument("--end", default=DISCOVERY_END)
    ap.add_argument("--model-version", default=STUDY_MODEL_VERSION)
    ap.add_argument("--t-star-up", type=float, help="frozen UP threshold (required for any window past discovery end)")
    ap.add_argument("--t-star-down", type=float, help="frozen DOWN threshold")
    ap.add_argument("--bucket-edges-json", help='frozen quintile edges, e.g. \'{"UP":[a,b,c,d],"DOWN":[...]}\'')
    ap.add_argument("--n-sims", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write JSON here instead of stdout")
    a = ap.parse_args(argv)

    t_star = None
    if a.t_star_up is not None or a.t_star_down is not None:
        t_star = {"UP": a.t_star_up, "DOWN": a.t_star_down}
    params = StudyParams(
        start=a.start,
        end=a.end,
        t_star=t_star,
        bucket_edges=json.loads(a.bucket_edges_json) if a.bucket_edges_json else None,
        n_sims=a.n_sims,
        seed=a.seed,
    )
    if params.end > DISCOVERY_END and not params.t_star:
        ap.error("a window past the discovery end needs --t-star-up/--t-star-down frozen on discovery")
    df = load_outcomes(a.db, a.start, a.end, a.model_version)
    loader = directory_loader(a.predictions_dir) if a.predictions_dir else s3_loader()
    df = attach_predictions(df, loader)
    result = run_study(df, params)
    text = json.dumps(result, indent=2, default=float)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
