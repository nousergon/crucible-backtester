"""expectancy — re-export shim over ``nousergon_lib.quant.stats.expectancy``.

Lifted to the shared alpha-engine-lib (LV2-AE leverage arc, 2026-06-03). This
shim preserves the ``analysis.expectancy`` import surface; the implementation +
its unit tests now live in the lib.
"""

from __future__ import annotations

from nousergon_lib.quant.stats.expectancy import (
    ExpectancyResult,
    compute_expectancy,
    compute_expectancy_by_group,
)

__all__ = ["ExpectancyResult", "compute_expectancy", "compute_expectancy_by_group"]
