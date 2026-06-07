"""Pins that the param sweep strips nested per-combo time-series so the
resulting sweep_df round-trips through parquet.

L4529 (2026-06-07): `vectorbt_bridge.run_vectorbt_simulation` returns
`daily_returns` / `daily_log_returns` as full pandas Series (a ~2500-row
return path per combo). `_run_combos` merged the whole stats dict into each
row, so those columns became dtype `object` holding Series. `to_parquet` then
died with:

    pyarrow.lib.ArrowInvalid: Could not convert <Series …> with type Series:
    did not recognize Python value type when inferring an Arrow data type
    (Conversion failed for column daily_returns with type object)

at backtest.py:_run_simulation_pipeline → phase_artifacts.save_dataframe. The
L4518 fail-loud export guard escalated that to a backtest-stage kill, so
sweep_df.parquet + portfolio_stats.json were never written and the Evaluator's
critical-artifact gate failed — taking the whole Saturday SF down.

Fix: strip the nested Series in `_run_combos` (mirrors the existing
`stats.pop(...)` pattern in analysis/portfolio_optimizer_backtest.py). The
sweep only needs scalar metrics per combo; nothing reads those Series back
from the parquet.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from analysis.param_sweep import _run_combos


def _stub_sim_with_nested_series(config: dict) -> dict:
    """Mimic vectorbt_bridge.run_vectorbt_simulation: scalar metrics PLUS the
    two nested Series that broke parquet serialization."""
    path = pd.Series(
        np.linspace(-0.01, 0.01, 2527),
        index=pd.date_range("2016-05-31", periods=2527, freq="D"),
        name="daily_return",
    )
    return {
        "sharpe_ratio": 1.23,
        "sortino_ratio": 1.45,
        "total_alpha": 0.05,
        "max_drawdown": -0.12,
        "daily_returns": path,
        "daily_log_returns": np.log1p(path.clip(lower=-0.999999)),
    }


def test_run_combos_drops_nested_series_columns():
    combos = [{"min_score": 55}, {"min_score": 60}]
    df = _run_combos(combos, _stub_sim_with_nested_series, base_config={})

    assert "daily_returns" not in df.columns, (
        "daily_returns (a nested Series) must be stripped before the row enters "
        "sweep_df — it breaks to_parquet (L4529)"
    )
    assert "daily_log_returns" not in df.columns, (
        "daily_log_returns must be stripped too (same ArrowInvalid failure mode)"
    )
    # scalar sweep metrics survive
    assert "sharpe_ratio" in df.columns and "sortino_ratio" in df.columns
    assert "min_score" in df.columns


def test_run_combos_output_is_parquet_serializable():
    """The end-to-end invariant: whatever _run_combos returns must survive the
    exact save_dataframe → to_parquet call the SF Backtester runs. Before the
    fix this raised ArrowInvalid and killed the backtest stage."""
    combos = [{"min_score": 55}, {"min_score": 60}, {"min_score": 65}]
    df = _run_combos(combos, _stub_sim_with_nested_series, base_config={})

    buf = io.BytesIO()
    df.to_parquet(buf, index=False)  # must not raise ArrowInvalid
    buf.seek(0)
    round_tripped = pd.read_parquet(buf)
    assert len(round_tripped) == len(combos)
    assert "sharpe_ratio" in round_tripped.columns
