#!/usr/bin/env python3
"""Era-correct, read-only read of what the live ``gbm_veto`` actually removed.

Refs nousergon/alpha-engine-config#12390.

Why this exists
---------------
``analysis/contribution_lift/groups/predictor_gates.py`` ablates the veto on the
predictor's WHOLE scored pool, ranked by ``predicted_alpha``. Under that ranking
the veto can never bind (every vetoed name has ``predicted_alpha < 0`` and sits
below the order width), so the replay reports N/A (backtester#824). That says
nothing about what the gate did in the era when research's ENTER calls set the
candidate pool (2026-04-07 .. 2026-05-13, 227 ``GBM veto:`` rows in
``executor_shadow_book``).  This script grades THAT record directly, from the
rows the executor actually wrote.

Read-only: reads a trades.db snapshot, predictions JSON files, and ArcticDB
closes.  Writes only the ``--out`` JSON.  Touches no live rule, threshold,
conviction gate or sizing.

Metrics (fixed before computing; see ``PREREG`` in the output):
  M1  veto precision  = share of first-veto events whose 21d log-alpha < 0.
  M2  lift            = mean(vetoed) - mean(entered), date-cluster bootstrap CI.
  M3  era binding     = on dates with >=1 ENTER, does ablating the veto change
                        the top-N (N = #ENTER) of ENTER U blocked-candidates
                        ranked by predicted_alpha?
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

HORIZON = 21
BOOT = 10_000
SEED = 0

PREREG = {
    "V": "executor_shadow_book rows block_reason LIKE 'GBM veto%'; unit=(ticker, first veto date)",
    "E": "trades action='ENTER'; unit=(ticker, first ENTER date) inside the V date window",
    "outcome": "ln(P[t+21td]/P[t]) - ln(SPY[t+21td]/SPY[t]), t = decision date close",
    "M1": "share of V events with outcome<0, Wilson 95% CI",
    "M2": "mean(V)-mean(E); 95% CI by date-cluster bootstrap (10000, seed 0); negative = veto removed worse names",
    "M3": "per date with >=1 ENTER: pool=ENTER U non-'already in portfolio' shadow rows; "
          "N=#ENTER; rank by predictions/{date}.json predicted_alpha; count dates top-N differs with vs without veto",
    "M3b": "AMENDED before any outcome/binding was computed: the backtester replay's exact definition applied to the era: "
           "pool = every name in predictions/{date}.json, N = #distinct live ENTER that date (>=1), "
           "baseline = top-N by predicted_alpha among gbm_veto-falsy, ablated = top-N among all; count dates the sets differ",
    "scope": "descriptive; no threshold, gate or sizing conclusion",
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def cluster_boot(by_date_v: dict, by_date_e: dict, rng) -> tuple[float, float, float]:
    """Difference of pooled means, resampling DATES (clusters) within each arm."""
    dates_v, dates_e = sorted(by_date_v), sorted(by_date_e)

    def mean_of(dates, src):
        xs = [x for d in dates for x in src[d]]
        return sum(xs) / len(xs)

    point = mean_of(dates_v, by_date_v) - mean_of(dates_e, by_date_e)
    draws = []
    for _ in range(BOOT):
        dv = [dates_v[rng.randrange(len(dates_v))] for _ in dates_v]
        de = [dates_e[rng.randrange(len(dates_e))] for _ in dates_e]
        draws.append(mean_of(dv, by_date_v) - mean_of(de, by_date_e))
    draws.sort()
    return point, draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT)]


def load_events(db: str):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    veto_rows = c.execute(
        "SELECT date, ticker, block_reason FROM executor_shadow_book "
        "WHERE block_reason LIKE 'GBM veto%'"
    ).fetchall()
    first_v: dict[str, str] = {}
    for d, t, _ in sorted(veto_rows):
        first_v.setdefault(t, d)
    lo = min(d for d, _, _ in veto_rows)
    hi = max(d for d, _, _ in veto_rows)
    enters = c.execute(
        "SELECT date, ticker FROM trades WHERE action='ENTER' AND date BETWEEN ? AND ?",
        (lo, hi),
    ).fetchall()
    first_e: dict[str, str] = {}
    for d, t in sorted(enters):
        first_e.setdefault(t, d)
    blocked = c.execute(
        "SELECT date, ticker, block_reason FROM executor_shadow_book "
        "WHERE block_reason NOT LIKE 'already in portfolio%' AND date BETWEEN ? AND ?",
        (lo, hi),
    ).fetchall()
    return veto_rows, first_v, enters, first_e, blocked, (lo, hi)


def realized(universe, macro, ticker: str, date: str):
    import pandas as pd

    try:
        px = universe.read(ticker, columns=["Close"]).data["Close"].dropna()
    except Exception:  # noqa: BLE001 - symbol absent => excluded, counted
        return None
    spy = macro.read("SPY", columns=["Close"]).data["Close"].dropna()
    ts = pd.Timestamp(date)
    if ts not in px.index or ts not in spy.index:
        return None
    i, j = px.index.get_loc(ts), spy.index.get_loc(ts)
    if i + HORIZON >= len(px) or j + HORIZON >= len(spy):
        return None
    return math.log(px.iloc[i + HORIZON] / px.iloc[i]) - math.log(spy.iloc[j + HORIZON] / spy.iloc[j])


def era_binding(enters, blocked, veto_rows, pred_dir: Path):
    vetoed = defaultdict(set)
    for d, t, _ in veto_rows:
        vetoed[d].add(t)
    enter_by = defaultdict(set)
    for d, t in enters:
        enter_by[d].add(t)
    pool_by = defaultdict(set)
    for d, t, _ in blocked:
        pool_by[d].add(t)
    out = {"dates_with_enter": 0, "dates_with_veto_in_pool": 0, "dates_topN_differs": 0,
           "dates_no_predictions": 0, "detail": []}
    for d in sorted(enter_by):
        out["dates_with_enter"] += 1
        pool = enter_by[d] | pool_by[d]
        if not (vetoed[d] & pool):
            continue
        out["dates_with_veto_in_pool"] += 1
        f = pred_dir / f"{d}.json"
        if not f.exists():
            out["dates_no_predictions"] += 1
            continue
        raw = json.loads(f.read_text())
        rows = raw.get("predictions", raw) if isinstance(raw, dict) else {}
        if isinstance(rows, list):
            rows = {r.get("ticker"): r for r in rows if isinstance(r, dict)}
        alpha = {t: rows[t].get("predicted_alpha") for t in pool
                 if isinstance(rows.get(t), dict) and isinstance(rows[t].get("predicted_alpha"), (int, float))}
        n = len(enter_by[d])
        rank = lambda names: sorted(sorted(names), key=lambda t: -alpha[t])[:n]  # noqa: E731
        base = set(rank([t for t in alpha if t not in vetoed[d]]))
        abl = set(rank(list(alpha)))
        differs = base != abl
        out["dates_topN_differs"] += int(differs)
        out["detail"].append({"date": d, "N": n, "pool": len(alpha), "vetoed_in_pool": len(vetoed[d] & set(alpha)),
                              "topN_differs": differs})
    return out


def replay_definition_on_era(enters, pred_dir: Path):
    """M3b: predictor_gates.build_arms' selection rule, applied to era dates."""
    enter_by = defaultdict(set)
    for d, t in enters:
        enter_by[d].add(t)
    out = {"dates_with_enter": len(enter_by), "dates_scored": 0, "dates_topN_differs": 0,
           "dates_no_predictions": 0, "detail": []}
    for d in sorted(enter_by):
        f = pred_dir / f"{d}.json"
        if not f.exists():
            out["dates_no_predictions"] += 1
            continue
        rows = json.loads(f.read_text()).get("predictions", [])
        sc = {r["ticker"]: r for r in rows if isinstance(r.get("predicted_alpha"), (int, float))}
        n = min(len(enter_by[d]), len([t for t in sc if not sc[t].get("gbm_veto")]))
        top = lambda names: set(sorted(sorted(names), key=lambda t: -sc[t]["predicted_alpha"])[:n])  # noqa: E731
        base, abl = top([t for t in sc if not sc[t].get("gbm_veto")]), top(list(sc))
        out["dates_scored"] += 1
        out["dates_topN_differs"] += int(base != abl)
        out["detail"].append({"date": d, "N": n, "pool": len(sc), "vetoed": sum(1 for t in sc if sc[t].get("gbm_veto")),
                              "topN_differs": base != abl, "ablated_only": sorted(abl - base),
                              "ablated_only_alpha": [sc[t]["predicted_alpha"] for t in sorted(abl - base)]})
    return out


def main() -> None:
    import random

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trades-db", required=True)
    ap.add_argument("--predictions-dir", required=True, help="dir of {date}.json copied from predictor/predictions/")
    ap.add_argument("--bucket", default="alpha-engine-research")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from nousergon_lib.arcticdb import open_arctic

    arctic = open_arctic(a.bucket)
    universe, macro = arctic.get_library("universe"), arctic.get_library("macro")

    veto_rows, first_v, enters, first_e, blocked, window = load_events(a.trades_db)
    res = {"prereg": PREREG, "window": window, "inputs": {"trades_db": a.trades_db}}
    res["counts"] = {"veto_rows": len(veto_rows), "veto_events": len(first_v), "enter_events": len(first_e)}

    def grade(events):
        by_date, excluded, per = defaultdict(list), [], []
        for t, d in events.items():
            r = realized(universe, macro, t, d)
            if r is None:
                excluded.append(f"{t}@{d}")
                continue
            by_date[d].append(r)
            per.append({"ticker": t, "date": d, "log_alpha_21d": r})
        return by_date, excluded, per

    v_by, v_ex, v_per = grade(first_v)
    e_by, e_ex, e_per = grade(first_e)
    vx = [p["log_alpha_21d"] for p in v_per]
    ex = [p["log_alpha_21d"] for p in e_per]
    k = sum(1 for x in vx if x < 0)
    lo, hi = wilson(k, len(vx))
    res["M1"] = {"n": len(vx), "underperformed": k, "precision": k / len(vx), "wilson95": [lo, hi],
                 "mean_log_alpha_21d": statistics.fmean(vx), "median": statistics.median(vx),
                 "excluded": v_ex}
    res["E"] = {"n": len(ex), "mean_log_alpha_21d": statistics.fmean(ex), "median": statistics.median(ex),
                "share_negative": sum(1 for x in ex if x < 0) / len(ex), "excluded": e_ex}
    point, blo, bhi = cluster_boot(v_by, e_by, random.Random(SEED))
    res["M2"] = {"mean_V_minus_mean_E": point, "ci95_date_cluster_boot": [blo, bhi],
                 "dates_V": len(v_by), "dates_E": len(e_by)}
    res["M3"] = era_binding(enters, blocked, veto_rows, Path(a.predictions_dir))
    res["M3b"] = replay_definition_on_era(enters, Path(a.predictions_dir))
    res["events_V"], res["events_E"] = v_per, e_per
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k_: res[k_] for k_ in ("window", "counts", "M1", "E", "M2")}, indent=1, default=str))
    for m in ("M3", "M3b"):
        print(m, json.dumps({k_: v for k_, v in res[m].items() if k_ != "detail"}))


if __name__ == "__main__":
    main()
