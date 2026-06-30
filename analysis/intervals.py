"""intervals — re-export shim over ``nousergon_lib.quant.stats.intervals``.

The bootstrap-CI / Newey-West / Wilson inference primitives were lifted to the
shared lib (LV2-AE leverage arc, 2026-06-03). This shim preserves an
``analysis.intervals`` import surface for backtester consumers; the
implementation + its unit tests now live in the lib.
"""

from __future__ import annotations

from nousergon_lib.quant.stats.intervals import (
    BootstrapCIResult,
    NeweyWestResult,
    WilsonScoreResult,
    bootstrap_ci,
    newey_west_se,
    wilson_score_interval,
)

__all__ = [
    "BootstrapCIResult",
    "NeweyWestResult",
    "WilsonScoreResult",
    "bootstrap_ci",
    "newey_west_se",
    "wilson_score_interval",
]
