"""
signal_quality.py — Mode 1: aggregate score_performance from research.db.

Reads the score_performance table (populated by the research pipeline) and
computes accuracy metrics: % of BUY signals that beat SPY at the diagnostic
short and primary horizons. Outcome columns are re-sourced from the
long-format score_performance_outcomes store via analysis.outcome_store
(config#1483/#1529); the horizons + column names resolve from
nousergon_lib.quant.horizons.HorizonPolicy (primary 21d + diagnostic 5d)
rather than hardcoded horizon-suffixed literals.

The legacy 10d/30d outcome horizons were RETIRED in the canonical-alpha
cutover (config#1456); the store carries policy horizons only. Gating +
reporting is on the primary horizon (the system's prediction target).

Data availability: meaningful results require ~200 populated rows.
This module returns insufficient_data until ~200 rows carry the primary
beat-SPY outcome.
"""

import logging
import sqlite3
from pathlib import Path

import pandas as pd
from nousergon_lib.quant.horizons import DEFAULT_POLICY, HorizonPolicy

from analysis.outcome_store import attach_outcomes
from analysis.stats_utils import benjamini_hochberg

logger = logging.getLogger(__name__)

# Minimum rows needed before reporting results — avoid misleading metrics on tiny samples
MIN_SAMPLES = 30

# Outcome column names resolve from the fleet-wide HorizonPolicy (primary 21d +
# diagnostic 5d) rather than hardcoded `_Nd` literals — the config#1483/#1529
# cutover. The physical column NAMES are unchanged (attach_outcomes re-sources
# them under the same names during the soak), but the SOURCE is now the
# long-format score_performance_outcomes store, not the wide score_performance
# columns. Deriving the names here (a) kills the scattered-literal bug class and
# (b) makes the burn-down guard pass without touching the value semantics.
_POLICY = DEFAULT_POLICY
_SHORT_H = _POLICY.diagnostic_horizons[0]  # 5
_LONG_H = _POLICY.primary_horizon  # 21
_SHORT_COLS = _POLICY.outcome_columns(_SHORT_H)
_LONG_COLS = _POLICY.outcome_columns(_LONG_H)
_BEAT_5D = _SHORT_COLS.beat_spy
_BEAT_21D = _LONG_COLS.beat_spy
_RET_5D = _SHORT_COLS.stock_return
_RET_21D = _LONG_COLS.stock_return
_SPY_5D = _SHORT_COLS.spy_return
_SPY_21D = _LONG_COLS.spy_return


def load_score_performance(
    db_path: str,
    policy: HorizonPolicy = _POLICY,
) -> pd.DataFrame:
    """
    Load score_performance from research.db, with the outcome columns
    re-sourced from the long-format ``score_performance_outcomes`` store.

    The wide horizon-suffixed outcome columns (beat-SPY flag, stock / SPY
    returns, and the primary-horizon log-alpha) are DROPPED from the raw
    ``score_performance`` read and replaced with values from
    ``score_performance_outcomes`` filtered by the HorizonPolicy horizons
    (config#1483/#1528/#1529). Values are byte-identical to the legacy wide
    read (the store is canonical DECIMAL; :func:`attach_outcomes` reproduces the
    legacy 2dp-percent convention on returns at that single boundary). If the
    store table is absent (pre-cutover DB), the raw wide columns pass through
    unchanged — a graceful no-op.

    Args:
        db_path: path to research.db.
        policy:  HorizonPolicy resolving which outcome columns to attach.
                 Defaults to the fleet-wide DEFAULT_POLICY (primary 21d +
                 diagnostic 5d). Callers building the alpha-decay-curve
                 ladder pass a policy spanning (1, 3, 5, 10, 15, 21)d.

    Returns a DataFrame carrying the score_performance non-outcome columns
    (symbol, score_date, score, price_on_date, …) plus the per-horizon outcome
    columns named by ``HorizonPolicy.outcome_columns`` (beat-SPY flag, stock /
    SPY returns, and — primary horizon only — the canonical log-alpha),
    re-sourced from score_performance_outcomes.
    """
    path = Path(db_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"research.db not found at {path}")

    conn = sqlite3.connect(path)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM score_performance ORDER BY score_date",
            conn,
            parse_dates=["score_date"],
        )

        # Enrich with sector from universe_returns (if table exists)
        if "sector" not in df.columns:
            try:
                ur_sectors = pd.read_sql_query(
                    "SELECT DISTINCT ticker, sector FROM universe_returns WHERE sector IS NOT NULL AND sector != ''",
                    conn,
                )
                if not ur_sectors.empty:
                    df = df.merge(ur_sectors, left_on="symbol", right_on="ticker", how="left")
                    df.drop(columns=["ticker"], inplace=True, errors="ignore")
                    logger.info("Enriched score_performance with sector from universe_returns (%d mapped)", df["sector"].notna().sum())
            except Exception:
                pass  # universe_returns may not exist yet
    finally:
        conn.close()

    # Re-source outcome columns from the long-format store (config#1529). The
    # wide columns are dropped and rebuilt from score_performance_outcomes.
    df = attach_outcomes(df, str(path), policy=policy)

    logger.info("Loaded %d rows from score_performance", len(df))
    return df


def compute_accuracy(df: pd.DataFrame, min_samples: int = MIN_SAMPLES) -> dict:
    """
    Given score_performance rows, compute accuracy metrics.

    Returns a dict with:
        - overall: {accuracy_5d, accuracy_21d, avg_alpha_5d/21d, n_5d/21d}
        - by_score_bucket: accuracy split into [60-70, 70-80, 80-90, 90+]
        - by_conviction: accuracy split by conviction (rising/stable/declining)
        - status: "insufficient_data" if not enough rows are populated yet
    """
    populated_5d = df[df[_BEAT_5D].notna()] if _BEAT_5D in df.columns else pd.DataFrame()
    # config#1456: gate on the canonical 21d horizon. The 10d/30d horizons were
    # retired in the canonical-alpha cutover (dark since April) — gating on the
    # retired horizon left the whole signal-quality tile insufficient/dark.
    populated_21d = df[df[_BEAT_21D].notna()] if _BEAT_21D in df.columns else pd.DataFrame()

    if len(populated_21d) < min_samples:
        logger.warning(
            "Only %d rows with canonical %s populated (need %d).",
            len(populated_21d), _BEAT_21D, min_samples,
        )
        return {
            "status": "insufficient_data",
            "rows_5d_populated": len(populated_5d),
            "rows_21d_populated": len(populated_21d),
            "rows_needed": min_samples,
        }

    result = {
        "status": "ok",
        "rows_5d_populated": len(populated_5d),
        "rows_21d_populated": len(populated_21d),
        "overall": _compute_slice_metrics(populated_5d, populated_21d),
        "by_score_bucket": _accuracy_by_score_bucket(populated_5d, populated_21d),
    }

    if "conviction" in df.columns:
        result["by_conviction"] = _accuracy_by_field(populated_5d, populated_21d, "conviction")

    if "sector" in df.columns and df["sector"].notna().any():
        result["by_sector"] = _accuracy_by_field(populated_5d, populated_21d, "sector")

    # Per-stance attribution (stance taxonomy arc PR 4, 2026-05-11).
    # Stance was added to predictions.json on 2026-05-11; if upstream
    # has joined predictions.stance into score_performance via a future
    # data-layer migration, this lights up automatically. Until then,
    # the field is absent and we skip — graceful degrade. The 4-stance
    # cohort split is the load-bearing observability for the executor's
    # stance-conditional gates: each stance's accuracy + alpha capture
    # answers "did this stance actually outperform / underperform vs
    # the others?" — directly informing whether to keep / drop / merge
    # stances in the taxonomy after the 4-12 week observation window.
    if "stance" in df.columns and df["stance"].notna().any():
        result["by_stance"] = _accuracy_by_field(
            populated_5d, populated_21d, "stance",
        )

    return result


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (round(max(0.0, centre - spread), 4), round(min(1.0, centre + spread), 4))


def _compute_slice_metrics(df_5d: pd.DataFrame, df_21d: pd.DataFrame) -> dict:
    n_5d = len(df_5d)
    n_21d = len(df_21d)

    acc_5d = float(df_5d[_BEAT_5D].mean()) if n_5d > 0 else None
    # config#1456: canonical 21d horizon (the system's prediction target).
    acc_21d = float(df_21d[_BEAT_21D].mean()) if n_21d > 0 else None

    # Wilson score 95% confidence intervals
    ci_5d = _wilson_ci(int(df_5d[_BEAT_5D].sum()), n_5d) if n_5d > 0 else None
    ci_21d = _wilson_ci(int(df_21d[_BEAT_21D].sum()), n_21d) if n_21d > 0 else None

    # Classification metrics at the canonical 21d horizon.
    # For BUY signals: precision = % that beat SPY (same as accuracy).
    # Recall is not computable here (requires universe-level data from end_to_end.py).
    tp_21d = int(df_21d[_BEAT_21D].sum()) if n_21d > 0 else 0
    fp_21d = n_21d - tp_21d

    return {
        "accuracy_5d": acc_5d,
        "accuracy_21d": acc_21d,
        "ci_95_5d": ci_5d,
        "ci_95_21d": ci_21d,
        "avg_alpha_5d": float((df_5d[_RET_5D] - df_5d[_SPY_5D]).mean()) if n_5d > 0 else None,
        "avg_alpha_21d": float((df_21d[_RET_21D] - df_21d[_SPY_21D]).mean()) if n_21d > 0 else None,
        "n_5d": n_5d,
        "n_21d": n_21d,
        "precision_21d": round(tp_21d / n_21d, 4) if n_21d > 0 else None,
        "tp_21d": tp_21d,
        "fp_21d": fp_21d,
    }


def _accuracy_by_score_bucket(df_5d: pd.DataFrame, df_21d: pd.DataFrame) -> list[dict]:
    buckets = [(60, 70), (70, 80), (80, 90), (90, 101)]
    rows = []
    for lo, hi in buckets:
        label = f"{lo}-{min(hi, 100)}" if hi <= 100 else f"{lo}+"
        slice_5d = df_5d[(df_5d["score"] >= lo) & (df_5d["score"] < hi)] if not df_5d.empty else pd.DataFrame()
        slice_21d = df_21d[(df_21d["score"] >= lo) & (df_21d["score"] < hi)]
        if len(slice_21d) == 0:
            continue
        exploratory_threshold = 20
        rows.append({
            "bucket": label,
            "exploratory": len(slice_21d) < exploratory_threshold,
            **_compute_slice_metrics(slice_5d, slice_21d),
        })

    # Apply BH FDR correction across bucket accuracy p-values
    # Derive implied p-values from Wilson CIs: a bucket is "significant" if
    # the CI excludes 0.50 (coin flip). Use a two-sided z-test approximation.
    import math
    p_values = []
    for row in rows:
        n = row.get("n_21d", 0)
        acc = row.get("accuracy_21d")
        if n > 0 and acc is not None:
            # Two-sided z-test for proportion vs 0.50
            z = abs(acc - 0.50) / max(math.sqrt(0.25 / n), 1e-10)
            # Approximate two-sided p-value using normal CDF
            p = 2.0 * (1.0 - _norm_cdf(z))
            p_values.append(p)
        else:
            p_values.append(1.0)

    fdr_results = benjamini_hochberg(p_values, alpha=0.05)
    for i, row in enumerate(rows):
        row["fdr_exploratory"] = not fdr_results[i]

    return rows


def _norm_cdf(z: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    import math
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _accuracy_by_field(df_5d: pd.DataFrame, df_21d: pd.DataFrame, field: str) -> list[dict]:
    values = df_21d[field].dropna().unique()
    rows = []
    for val in sorted(values):
        slice_5d = df_5d[df_5d[field] == val] if not df_5d.empty and field in df_5d.columns else pd.DataFrame()
        slice_21d = df_21d[df_21d[field] == val]
        rows.append({
            field: val,
            **_compute_slice_metrics(slice_5d, slice_21d),
        })
    return rows
