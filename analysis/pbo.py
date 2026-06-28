"""pbo — re-export shim over ``nousergon_lib.quant.stats.pbo``.

The CSCV Probability of Backtest Overfitting math (Bailey, Borwein, López de
Prado & Zhu 2014) was lifted to the shared nousergon-lib (config#1318) so the
backtester and the predictor consume one engine — the second-adopter
consolidation this module's own provenance note called for. This shim preserves
the ``analysis.pbo`` import surface; the implementation + its unit tests now
live in the lib.
"""

from __future__ import annotations

from nousergon_lib.quant.stats.pbo import cscv_pbo

__all__ = ["cscv_pbo"]
