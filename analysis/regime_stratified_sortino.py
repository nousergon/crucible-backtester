"""
analysis/regime_stratified_sortino.py — regime-stratified pick-alpha Sortino.

**Hybrid module (LV2-AE.b, 2026-06-03).** The pure metric core was lifted to the
shared ``alpha_engine_lib.quant.stats.regime_sortino`` so the backtester and
robodashboard consume one engine. This module:
  - keeps ``load_with_subscores_and_regime`` — the SQLite I/O that reads
    ``score_performance`` is storage-specific and stays here;
  - re-exports the pure core (metrics, spread, payload assembly + helpers) from
    the lib, preserving the ``analysis.regime_stratified_sortino`` import surface
    for ``regime_stratified_sortino_runner``, ``evaluate.py``, and the tests.

See the lib module for the full canonical-alpha / Sortino-primary math + tests.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

# Pure metric core — implementation + unit tests live in the lib.
from alpha_engine_lib.quant.stats.regime_sortino import (
    DEFAULT_MIN_PICKS_PER_STRATUM,
    SUPPORTED_HORIZONS,
    StratumMetrics,
    _annualization_factor,
    _annualized_sharpe_from_log_alphas,
    _annualized_sortino_from_log_alphas,
    _arithmetic_to_log_alpha,
    _downside_std,
    _SORTINO_INVERTED_THRESHOLD,
    _SORTINO_USEFUL_THRESHOLD,
    _stratum_metrics,
    _TRADING_DAYS_PER_YEAR,
    assemble_t2_eval_payload,
    compute_regime_spread,
    stratified_sortino_by_regime,
)

logger = logging.getLogger(__name__)


def load_with_subscores_and_regime(db_path: str) -> pd.DataFrame:
    """Load score_performance with the canonical market_regime column
    plus the arithmetic return columns needed for stratified Sortino.
    Log conversion happens at metric-compute time (in the lib core).

    Returns a DataFrame with at minimum:
      - market_regime (str | NaN for pre-migration rows)
      - return_10d, spy_10d_return (arithmetic; converted to log alpha downstream)
      - return_30d, spy_30d_return (arithmetic; same)
      - beat_spy_10d, beat_spy_30d (booleans for hit rate)

    Pre-migration rows with NULL market_regime are kept in the
    DataFrame but filtered out at the per-regime grouping stage —
    callers see the row count for observability without needing to
    know about the migration.
    """
    path = Path(db_path).expanduser()
    conn = sqlite3.connect(path)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM score_performance ORDER BY score_date",
            conn,
            parse_dates=["score_date", "eval_date_10d", "eval_date_30d"],
        )
    finally:
        conn.close()

    if "market_regime" not in df.columns:
        # Pre-migration #12 schema. Inject as all-NULL so downstream
        # grouping treats every row as ungrouped (skipped) rather
        # than raising KeyError.
        df["market_regime"] = pd.NA

    n_with_regime = int(df["market_regime"].notna().sum())
    logger.info(
        "Loaded %d score_performance rows for stratified Sortino (%d with market_regime populated)",
        len(df), n_with_regime,
    )
    return df


__all__ = [
    "load_with_subscores_and_regime",
    # Re-exported pure core (from alpha_engine_lib.quant.stats.regime_sortino)
    "DEFAULT_MIN_PICKS_PER_STRATUM",
    "SUPPORTED_HORIZONS",
    "StratumMetrics",
    "stratified_sortino_by_regime",
    "compute_regime_spread",
    "assemble_t2_eval_payload",
    "_arithmetic_to_log_alpha",
    "_annualization_factor",
    "_annualized_sortino_from_log_alphas",
    "_annualized_sharpe_from_log_alphas",
    "_downside_std",
    "_stratum_metrics",
    "_SORTINO_USEFUL_THRESHOLD",
    "_SORTINO_INVERTED_THRESHOLD",
    "_TRADING_DAYS_PER_YEAR",
]
