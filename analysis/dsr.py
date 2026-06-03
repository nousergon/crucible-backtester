"""dsr — re-export shim over ``alpha_engine_lib.quant.stats.dsr``.

The PSR/DSR math was lifted to the shared alpha-engine-lib (LV2-AE leverage arc,
2026-06-03) so the backtester and robodashboard consume one engine. This shim
preserves the ``analysis.dsr`` import surface; the implementation + its unit
tests now live in the lib.
"""

from __future__ import annotations

from alpha_engine_lib.quant.stats.dsr import (
    DSRResult,
    PSRResult,
    compute_dsr,
    compute_psr,
)

__all__ = ["PSRResult", "DSRResult", "compute_psr", "compute_dsr"]
