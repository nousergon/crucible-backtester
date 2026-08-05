"""analysis/wide_feed_counterfactual_review.py — read-only review CLI for the
config#1398 wide-feed-by-attractiveness counterfactual (alpha-engine-config
issue #1398).

Purpose
-------
The counterfactual this issue asks for — top-N-by-attractiveness (N =
60/120/200, sector-balanced + unbalanced) vs the live ``tech_score`` survivor
gate, ex-post winner capture rate + realized 21d alpha — already ships as
production code: ``analysis.attractiveness_eval._counterfactual`` (config#1398,
PR #416/#467), wired into ``evaluate.py`` and emitted weekly as
``backtest/{date}/attractiveness_eval.json``. This module does NOT duplicate
that computation. It:

1. Calls ``compute_attractiveness_eval`` to reproduce the artifact against a
   given research.db + the live S3 attractiveness history (so the reviewer
   can point it at a fresh pull and see current numbers, or a captured
   research.db for a repeatable review).
2. Adds ONE thing the production artifact omits: a no-skill capture-rate
   REFERENCE per N, so ``capture_rate`` is interpretable. ``capture_rate`` in
   the artifact is the fraction of a cycle's top-K realized-alpha names (K =
   selection size) a selection contains — but that number alone doesn't say
   whether the composite shows skill, because a bigger N mechanically nets a
   bigger share of any fixed universe even with ZERO skill. The reference:
   for a selection of size N drawn UNIFORMLY AT RANDOM (no skill) from a
   cohort of ``n_universe`` scanner-evaluated names, the expected overlap
   with the cycle's own top-N winner set is a hypergeometric draw —
   ``E[overlap] = N*N / n_universe``, so ``E[capture_rate] = N / n_universe``.
   ``capture_rate_vs_null`` (actual minus null) is the number that actually
   answers "does ranking by attractiveness beat picking at random."

Read-only guarantee: this script's only network access is the SAME S3 GET the
production producer already performs to read the attractiveness history
parquet (via ``attractiveness_eval.load_attractiveness_history``) plus local
SELECT-only queries against the given research.db path. It writes no S3 key,
no config, no research.db row — output goes to stdout and, optionally, a
local JSON file the caller names explicitly.

Usage
-----
    python -m analysis.wide_feed_counterfactual_review \\
        --research-db /path/to/research.db --bucket alpha-engine-research

    # Also dump the full annotated JSON for downstream use:
    python -m analysis.wide_feed_counterfactual_review \\
        --research-db /path/to/research.db --json-out /tmp/i1398_review.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sqlite3
from pathlib import Path

import pandas as pd

from analysis.attractiveness_eval import (
    compute_attractiveness_eval,
    load_attractiveness_history,
)

logger = logging.getLogger(__name__)

# Below this many scored (attractiveness-covered) names, a cycle is too thin
# to count toward the cohort-size average — mirrors attractiveness_eval's own
# MIN_NAMES_PER_DATE floor for a cross-section to count as one observation.
_MIN_SCORED_FOR_COHORT = 10


def cohort_sizes(conn: sqlite3.Connection, history_df: pd.DataFrame | None) -> dict[str, dict]:
    """Per-eval_date cohort sizes behind the config#1398 counterfactual:
    ``n_universe`` (scanner-evaluated names with resolved 21d alpha — the pool
    ``_capture_rate`` draws each cycle's ex-post winners from) and
    ``n_scored`` (the subset with an attractiveness_score — the pool top-N
    selections are drawn from).

    Needed to interpret ``counterfactual.top_n[].capture_rate`` against a
    no-skill reference (see :func:`null_capture_rate`) — a number the
    production ``attractiveness_eval.json`` artifact does not itself carry.
    Pure/read-only: two SELECT queries + a groupby; no upstream writes. Empty
    dict when the required tables/columns are absent (mirrors the producer's
    own fail-soft posture)."""
    tabs = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "scanner_evaluations" not in tabs or "universe_returns" not in tabs:
        return {}
    se_cols = {r[1] for r in conn.execute("PRAGMA table_info(scanner_evaluations)")}
    if "quant_filter_pass" not in se_cols:
        return {}

    from nousergon_lib.quant.horizons import DEFAULT_POLICY

    horizon = int(DEFAULT_POLICY.primary_horizon)
    ret_col, spy_col = f"log_return_{horizon}d", f"log_spy_return_{horizon}d"
    ur_cols = {r[1] for r in conn.execute("PRAGMA table_info(universe_returns)")}
    if not {ret_col, spy_col}.issubset(ur_cols):
        return {}

    se = pd.read_sql_query(
        "SELECT ticker, eval_date, quant_filter_pass FROM scanner_evaluations", conn,
    )
    ur = pd.read_sql_query(
        f"SELECT ticker, eval_date FROM universe_returns "
        f"WHERE {ret_col} IS NOT NULL AND {spy_col} IS NOT NULL", conn,
    )
    base = se.merge(ur, on=["ticker", "eval_date"], how="inner")

    scored_keys: set[tuple[str, str]] = set()
    if history_df is not None and not history_df.empty:
        h = history_df.rename(columns={"as_of": "eval_date"})
        scored_keys = set(zip(h["eval_date"], h["ticker"].astype(str)))

    out: dict[str, dict] = {}
    for d, g in base.groupby("eval_date"):
        n_survivors = int((g["quant_filter_pass"] == 1).sum())
        n_scored = (
            sum(1 for t in g["ticker"] if (d, t) in scored_keys) if scored_keys else 0
        )
        out[str(d)] = {
            "n_universe": int(len(g)),
            "n_survivors": n_survivors,
            "n_scored": int(n_scored),
        }
    return out


def null_capture_rate(n: int, n_universe_avg: float | None) -> float | None:
    """Expected ex-post winner capture rate for a size-``n`` selection drawn
    with NO skill (uniformly at random) from a cohort of ``n_universe_avg``
    names, against that cycle's own top-``n`` winner set.

    Two size-``n`` subsets of an ``n_universe``-name pool overlap, in
    expectation under a uniform-random draw, by a hypergeometric mean:
    ``E[overlap] = n * n / n_universe``, so ``E[capture_rate] =
    E[overlap] / n = n / n_universe``. Returns ``None`` when the cohort size
    is unknown/non-positive or smaller than ``n`` (a null rate above 1 is not
    meaningful — the same truncation the producer itself applies to top-N
    selection: "a truncated top-N wouldn't be the named variant")."""
    if not n_universe_avg or n_universe_avg <= 0 or n > n_universe_avg:
        return None
    return round(n / n_universe_avg, 5)


def build_review(research_db_path: str, bucket: str, as_of: str) -> dict:
    """Recompute the config#1398 artifact against the given research.db +
    live S3 attractiveness history, and annotate its ``counterfactual.top_n``
    rows with the no-skill capture-rate reference. Read-only: performs ONE S3
    GET for the history parquet (shared with ``compute_attractiveness_eval``
    via the ``history_df`` injection point — not a second independent read),
    no S3/DB writes."""
    history_df = load_attractiveness_history(bucket)
    artifact = compute_attractiveness_eval(
        research_db_path, as_of=as_of, bucket=bucket, history_df=history_df,
    )

    conn = sqlite3.connect(research_db_path)
    try:
        sizes = cohort_sizes(conn, history_df)
    finally:
        conn.close()

    covered = [
        v for v in sizes.values()
        if v["n_scored"] >= _MIN_SCORED_FOR_COHORT and v["n_survivors"] > 0
    ]
    n_universe_avg = (
        sum(v["n_universe"] for v in covered) / len(covered) if covered else None
    )
    n_scored_avg = (
        sum(v["n_scored"] for v in covered) / len(covered) if covered else None
    )

    top_n = artifact.get("counterfactual", {}).get("top_n", [])
    for row in top_n:
        null_rate = null_capture_rate(row["n"], n_universe_avg)
        row["null_capture_rate"] = null_rate
        if null_rate is not None and row.get("capture_rate") is not None:
            row["capture_rate_vs_null"] = round(row["capture_rate"] - null_rate, 5)
        else:
            row["capture_rate_vs_null"] = None

    return {
        "artifact": artifact,
        "cohort": {
            "n_cycles_covered": len(covered),
            "n_universe_avg": round(n_universe_avg, 1) if n_universe_avg else None,
            "n_scored_avg": round(n_scored_avg, 1) if n_scored_avg else None,
            "eval_dates": sorted(sizes.keys()),
        },
    }


def format_report(review: dict) -> str:
    """Render the annotated review as a markdown fragment (stdout /
    EXPERIMENTS.md-pastable)."""
    art = review["artifact"]
    cf = art.get("counterfactual", {})
    lg = cf.get("live_gate", {})
    cic = art.get("composite_ic", {})
    cohort = review["cohort"]
    lines = [
        "# Wide-feed counterfactual review (config#1398)",
        "",
        f"as_of={art.get('as_of')} status={art.get('status')} "
        f"horizon_days={art.get('horizon_days')} n_cycles={cf.get('n_cycles')} "
        f"cohort_n_universe_avg={cohort.get('n_universe_avg')} "
        f"cohort_n_scored_avg={cohort.get('n_scored_avg')}",
        "",
        f"composite_ic: date_ic_mean={cic.get('date_ic_mean')} "
        f"p={cic.get('date_ic_p')} n_eval_dates={cic.get('n_eval_dates')}",
        "",
        f"live_gate (tech_score survivors, same cycles): "
        f"capture_rate={lg.get('capture_rate')} mean_alpha={lg.get('mean_alpha')} "
        f"n_survivors={lg.get('n_survivors')}",
        "",
        "| N | sector_balanced | capture_rate | null_capture_rate | "
        "capture_rate_vs_null | mean_alpha | n_cycles |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in cf.get("top_n", []):
        lines.append(
            f"| {row['n']} | {row['sector_balanced']} | {row.get('capture_rate')} | "
            f"{row.get('null_capture_rate')} | {row.get('capture_rate_vs_null')} | "
            f"{row.get('mean_alpha')} | {row.get('n_cycles')} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only review of the config#1398 wide-feed-by-attractiveness "
            "counterfactual: reproduces analysis.attractiveness_eval's "
            "top-N-vs-tech_score-gate comparison and adds a no-skill "
            "capture-rate reference. Writes nothing to S3 or research.db."
        ),
    )
    parser.add_argument("--research-db", required=True, help="path to a local research.db")
    parser.add_argument("--bucket", default="alpha-engine-research")
    parser.add_argument(
        "--as-of", default=None, help="artifact stamp date (default: today, UTC)"
    )
    parser.add_argument(
        "--json-out", default=None,
        help="optional local path to write the full annotated JSON review",
    )
    args = parser.parse_args(argv)

    as_of = args.as_of or datetime.date.today().isoformat()
    review = build_review(args.research_db, args.bucket, as_of)
    print(format_report(review))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(review, indent=2, default=str))
        print(f"\nWrote full JSON to {args.json_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
