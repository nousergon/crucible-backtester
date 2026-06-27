"""risk_matched_benchmark — re-export shim over
``nousergon_lib.quant.stats.risk_matched_benchmark``.

Lifted to the shared alpha-engine-lib (LV2-AE leverage arc, 2026-06-03). This
shim preserves the ``analysis.risk_matched_benchmark`` import surface; the
implementation + its unit tests now live in the lib.
"""

from __future__ import annotations

from nousergon_lib.quant.stats.risk_matched_benchmark import (
    BenchmarkResult,
    compute_alpha_vs_benchmark,
    construct_beta_matched_spy_benchmark,
    construct_ew_high_vol_benchmark,
)

__all__ = [
    "BenchmarkResult",
    "compute_alpha_vs_benchmark",
    "construct_beta_matched_spy_benchmark",
    "construct_ew_high_vol_benchmark",
]
