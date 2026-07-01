"""
regime_analysis.py — split signal accuracy metrics by market_regime.

Reads market_regime from the canonical score_performance column (populated
by alpha-engine-data signal_returns collector post research migration #12,
2026-05-08). Pre-migration rows with NULL market_regime are silently
excluded from regime-split metrics.

Data availability: requires score_performance to be populated (Week 4+).
"""

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from analysis.signal_quality import _compute_slice_metrics, MIN_SAMPLES

logger = logging.getLogger(__name__)


def load_with_regime(db_path: str) -> pd.DataFrame:
    """
    Load score_performance with the canonical market_regime column.

    Pre-2026-05-10 this function did a LEFT JOIN against macro_snapshots
    to source market_regime. That join broke on Saturday 2026-05-09:
    research migration #12 had added market_regime as a column on
    score_performance, so ``SELECT sp.*, ms.market_regime`` returned
    two columns named market_regime; pandas surfaced df["market_regime"]
    as a DataFrame (not a Series) and the downstream logger / accuracy
    split crashed with ``TypeError: %d format: a real number is
    required, not Series``. macro_snapshots was also 7+ weeks stale
    (latest 2026-03-16), so the join had been contributing nothing for
    recent rows anyway.

    Single canonical source removes both failure modes.
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
        # Pre-migration #12 schema. Inject the column as all-NULL so
        # downstream accuracy_by_regime treats it as "insufficient data"
        # rather than KeyError.
        df["market_regime"] = pd.NA

    logger.info(
        "Loaded %d score_performance rows (%d with market_regime populated)",
        len(df),
        int(df["market_regime"].notna().sum()),
    )
    return df


def accuracy_by_regime(df: pd.DataFrame, min_samples: int = MIN_SAMPLES) -> list[dict]:
    """
    Compute accuracy metrics grouped by market_regime.

    Returns list of dicts, one per regime, each with the same structure as
    signal_quality._compute_slice_metrics().
    """
    if "market_regime" not in df.columns:
        return []

    populated_5d = df[df["beat_spy_5d"].notna()] if "beat_spy_5d" in df.columns else pd.DataFrame()
    # config#1456: canonical 21d horizon (10d/30d outcomes retired).
    populated_21d = df[df["beat_spy_21d"].notna()] if "beat_spy_21d" in df.columns else pd.DataFrame()

    if len(populated_21d) < min_samples:
        logger.warning(
            "Only %d rows with beat_spy_21d populated — regime analysis deferred until Week 4.",
            len(populated_21d),
        )
        return []

    regimes = populated_21d["market_regime"].dropna().unique()
    results = []

    for regime in sorted(regimes):
        slice_5d = populated_5d[populated_5d["market_regime"] == regime] if not populated_5d.empty else pd.DataFrame()
        slice_21d = populated_21d[populated_21d["market_regime"] == regime]
        metrics = _compute_slice_metrics(slice_5d, slice_21d)
        results.append({"market_regime": regime, **metrics})

    return results
