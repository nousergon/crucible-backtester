"""information_coefficient — re-export shim over
``nousergon_lib.quant.stats.information_coefficient``.

Lifted to the shared alpha-engine-lib (LV2-AE leverage arc, 2026-06-03). This
shim preserves the ``analysis.information_coefficient`` import surface; the
implementation + its unit tests now live in the lib.
"""

from __future__ import annotations

from nousergon_lib.quant.stats.information_coefficient import (
    ICResult,
    compute_ic,
    compute_ic_by_bucket,
)

__all__ = ["ICResult", "compute_ic", "compute_ic_by_bucket"]
