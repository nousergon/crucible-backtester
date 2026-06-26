"""stats_utils — re-export shim over ``nousergon_lib.quant.stats.multiple_testing``.

The Benjamini-Hochberg FDR helper was lifted to the shared alpha-engine-lib
(LV2-AE leverage arc, 2026-06-03) as ``quant.stats.multiple_testing``. This shim
preserves the ``analysis.stats_utils`` import surface; the implementation now
lives in the lib.
"""

from __future__ import annotations

from nousergon_lib.quant.stats.multiple_testing import benjamini_hochberg

__all__ = ["benjamini_hochberg"]
