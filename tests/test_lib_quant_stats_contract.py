"""Consumer contract: backtester's analysis stat shims ⇄ alpha_engine_lib.quant.stats.

The statistical-evaluation library (PSR/DSR, IC, expectancy, BH-FDR, risk-matched
benchmarks) was lifted to the shared lib (LV2-AE leverage arc, 2026-06-03).
``analysis/{dsr,information_coefficient,expectancy,stats_utils,risk_matched_benchmark}.py``
are now re-export shims over ``alpha_engine_lib.quant.stats.*``. This pins the
contract so a lib version that drops/renames a consumed symbol fails here, not at
Saturday backtest time. Exhaustive math tests live in the lib; the real consumers
(vectorbt_bridge, team_skill_metrics, signal_quality, attribution, backtest) plus
test_analysis_coverage.py (BH-FDR) exercise the shims end-to-end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_shims_re_export_consumed_symbols():
    from analysis.dsr import compute_dsr, compute_psr
    from analysis.expectancy import compute_expectancy
    from analysis.information_coefficient import compute_ic
    from analysis.risk_matched_benchmark import (
        compute_alpha_vs_benchmark,
        construct_beta_matched_spy_benchmark,
        construct_ew_high_vol_benchmark,
    )
    from analysis.stats_utils import benjamini_hochberg

    for sym in (
        compute_psr, compute_dsr, compute_ic, compute_expectancy, benjamini_hochberg,
        compute_alpha_vs_benchmark, construct_beta_matched_spy_benchmark,
        construct_ew_high_vol_benchmark,
    ):
        assert callable(sym)


def test_shim_is_identity_with_lib():
    from nousergon_lib.quant.stats import dsr as lib_dsr

    from analysis.dsr import compute_psr

    assert compute_psr is lib_dsr.compute_psr


def test_psr_smoke():
    from analysis.dsr import compute_psr

    rng = np.random.RandomState(0)
    r = rng.normal(0.0008, 0.01, 252)
    out = compute_psr(r, sharpe_benchmark=0.0)
    assert out["status"] == "ok"
    assert 0.0 <= out["psr"] <= 1.0


def test_ic_smoke():
    from analysis.information_coefficient import compute_ic

    rng = np.random.RandomState(1)
    conv = rng.normal(0, 1, 50)
    fwd = conv * 0.3 + rng.normal(0, 1, 50)
    out = compute_ic(conv, fwd)
    assert out["status"] == "ok"
    assert out["ic"] > 0


def test_benchmark_smoke():
    from analysis.risk_matched_benchmark import compute_alpha_vs_benchmark

    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    port = pd.Series(np.full(60, 0.001), index=idx)
    bench = pd.Series(np.full(60, 0.0005), index=idx)
    out = compute_alpha_vs_benchmark(port, bench, label="test")
    assert out["status"] == "ok"
    assert out["excess_return"] > 0
