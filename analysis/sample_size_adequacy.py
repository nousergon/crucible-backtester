"""
sample_size_adequacy.py — report-card producer (config#1151 Batch C).

The System Report Card grades a *critical* ``sample_size_adequacy`` component:
are the per-cycle finalized-signal counts that feed the accuracy / regime /
attribution analyses actually ABOVE the floor each needs for a stable estimate,
or are those grades being computed on too-few samples to mean anything? Without
this, a GREEN/RED accuracy tile on N=8 reads identically to one on N=200 — the
report card can't tell the Director "this grade is well-powered" from "this grade
is noise."

Pure-compute over the already-computed analysis results (no new data read): it
reads the finalized-signal counts the analyses themselves used, compares each to
its documented floor, and headlines the WEAKEST-LINK adequacy ratio (the smallest
n/floor across analyses) — because the report card is only as well-powered as its
least-sampled input. Always-emit (even insufficient_data) so the evaluator can
distinguish "producer didn't run" from "ran, genuinely too few samples".
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Documented per-analysis sample floors (finalized signals with a realized
# outcome). signal_quality: ~2-3 weeks of 21d-realized cross-sectional outcomes
# for a stable beat-rate; attribution: 100 (raised 2026, config#946 — sub-score
# correlation needs the larger N). A cycle below a floor means that analysis's
# grade is under-powered, not necessarily wrong.
SIGNAL_QUALITY_SAMPLE_FLOOR = 60
ATTRIBUTION_SAMPLE_FLOOR = 100


def _ratio(n: int | None, floor: int) -> float | None:
    if n is None or floor <= 0:
        return None
    return round(n / floor, 4)


def compute_sample_size_adequacy(
    signal_quality: dict | None,
    attribution: dict | None = None,
) -> dict:
    """Per-analysis finalized-signal count vs documented floor (config#1151).

    Args:
        signal_quality: the ``compute_accuracy`` result ({status, overall:{n_10d,
            n_30d, ...}}). The realized-outcome count the accuracy grade used.
        attribution: optional ``attribution`` result; its sample count (``n`` /
            ``n_samples``) is compared to ``ATTRIBUTION_SAMPLE_FLOOR`` when present.

    Returns a dict with the per-analysis breakdown and the WEAKEST-LINK headline
    ``adequacy_ratio`` (min n/floor across analyses) + ``adequate`` bool. Status
    ``insufficient_data`` when no analysis reported a usable count.
    """
    per_analysis: dict[str, dict] = {}

    sq = signal_quality or {}
    if sq.get("status") == "ok":
        overall = sq.get("overall") or {}
        # Prefer the longer-horizon realized slice (more decision-relevant at the
        # 21d canonical horizon); fall back to the 10d count.
        n_sq = overall.get("n_30d") or overall.get("n_10d")
        if n_sq is not None:
            per_analysis["signal_quality"] = {
                "n": int(n_sq), "floor": SIGNAL_QUALITY_SAMPLE_FLOOR,
                "adequacy_ratio": _ratio(int(n_sq), SIGNAL_QUALITY_SAMPLE_FLOOR),
            }

    attr = attribution or {}
    if attr.get("status") == "ok":
        n_attr = attr.get("n", attr.get("n_samples"))
        if n_attr is not None:
            per_analysis["attribution"] = {
                "n": int(n_attr), "floor": ATTRIBUTION_SAMPLE_FLOOR,
                "adequacy_ratio": _ratio(int(n_attr), ATTRIBUTION_SAMPLE_FLOOR),
            }

    ratios = [v["adequacy_ratio"] for v in per_analysis.values() if v["adequacy_ratio"] is not None]
    if not ratios:
        return {
            "status": "insufficient_data",
            "reason": "no analysis reported a usable finalized-signal count this cycle",
            "per_analysis": per_analysis,
        }

    # Weakest link: the report card is only as well-powered as its least-sampled
    # input, so the headline is the minimum adequacy ratio across analyses.
    weakest_ratio = min(ratios)
    weakest = min(per_analysis.items(), key=lambda kv: (kv[1]["adequacy_ratio"] is None, kv[1]["adequacy_ratio"]))
    return {
        "status": "ok",
        "adequacy_ratio": weakest_ratio,
        "adequate": bool(weakest_ratio >= 1.0),
        "weakest_analysis": weakest[0],
        "per_analysis": per_analysis,
    }
