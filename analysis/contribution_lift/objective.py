"""objective.py — the ONE objective, and the statistics computed over it.

Report Card v3 §1:

    objective(cycle) = log_return_21d(portfolio, net_of_cost) − log_return_21d(SPY)

Everything here operates on the simulator's own ``daily_log_returns`` series
(``vectorbt_bridge.portfolio_stats``) and the SPY close series. Nothing here
reads ``sharpe_ratio`` or Sortino — those fields are corrupted by the open
I7236 / I7237 / I7271 defects, and the spec's blocking note forbids deriving
any status from them. The lift is a sum of log returns, so it is not blocked.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Collection

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CountMatch:
    """The count-match verdict plus the full date-support census behind it.

    ``matched`` is True only when every date in the UNION of both arms'
    declared supports is either width-matched or named ``unpriceable``, and at
    least one date is width-matched. The per-class date tuples are the census
    (alpha-engine-config-I12243): a date one arm trades and the other does not
    is never silently dropped from the assurance.
    """

    matched: bool
    reason: str
    matched_dates: tuple[str, ...] = ()
    mismatched_dates: tuple[str, ...] = ()
    baseline_only_dates: tuple[str, ...] = ()
    ablated_only_dates: tuple[str, ...] = ()
    unpriceable_dates: tuple[str, ...] = ()

    def census(self) -> dict:
        """Counts per support class plus every non-matched date, named."""
        return {
            "n_union_dates": (
                len(self.matched_dates) + len(self.mismatched_dates)
                + len(self.baseline_only_dates) + len(self.ablated_only_dates)
                + len(self.unpriceable_dates)
            ),
            "n_matched": len(self.matched_dates),
            "n_mismatched": len(self.mismatched_dates),
            "n_baseline_only": len(self.baseline_only_dates),
            "n_ablated_only": len(self.ablated_only_dates),
            "n_unpriceable": len(self.unpriceable_dates),
            "mismatched_dates": list(self.mismatched_dates),
            "baseline_only_dates": list(self.baseline_only_dates),
            "ablated_only_dates": list(self.ablated_only_dates),
            "unpriceable_dates": list(self.unpriceable_dates),
        }


def per_cycle_log_alpha(
    daily_log_returns: pd.Series,
    spy_prices: pd.Series,
    trading_axis: pd.DatetimeIndex,
    cycle_dates: list[str],
    *,
    horizon_days: int,
) -> dict[str, float]:
    """``{cycle_date: log_alpha}`` over the ``horizon_days`` rows after each cycle.

    For cycle date ``d``:

    * ``w = trading_axis[pos : pos + horizon_days]`` where ``pos`` is the first
      row STRICTLY after ``d`` — the objective measures what happened *after*
      the cycle's decision, never the decision day's own move.
    * ``arm_leg  = Σ daily_log_returns over w``. The simulator's daily log
      return dated ``w[0]`` is the move from ``close(w[0]-1) == close(d)`` into
      ``close(w[0])``, so the sum telescopes to exactly
      ``log(NAV(w[-1]) / NAV(d))``.
    * ``spy_leg  = log(spy[w[-1]] / spy[axis[pos-1]])`` — the SAME endpoints, so
      the two legs are measured over an identical window rather than one being
      offset by a day.
    * ``log_alpha = arm_leg − spy_leg``.

    A cycle with fewer than ``horizon_days`` rows remaining is OMITTED, not
    truncated: a short window would compare a 12-day arm leg to a 12-day SPY
    leg and then be pooled with 21-day draws, which quietly reweights the
    sample toward the end of the window. The loader's window selection
    (``inputs.load_replay_inputs``) already guarantees a full horizon for
    every cycle it returns; this is the belt-and-braces check.

    Reindexing note: ``daily_log_returns`` covers the portfolio's ACTIVE
    window only. Rows in ``w`` outside it are filled with 0.0 — outside its
    active window the arm holds cash and earns nothing, which is the correct
    contribution, not a swallowed gap. It is recorded in the emitted
    ``per_cycle_log_alpha_21d`` map, so a component whose cycles are mostly
    zeros is visible on the artifact rather than inferred.
    """
    if not isinstance(daily_log_returns, pd.Series):
        raise TypeError(
            "per_cycle_log_alpha requires the simulator's daily_log_returns "
            f"Series; got {type(daily_log_returns).__name__}"
        )
    axis = pd.DatetimeIndex(trading_axis).sort_values()
    if len(axis) == 0:
        return {}

    arm = daily_log_returns.copy()
    arm.index = pd.DatetimeIndex(arm.index)
    arm = arm.reindex(axis).fillna(0.0)

    spy = spy_prices.copy()
    spy.index = pd.DatetimeIndex(spy.index)
    spy = spy.reindex(axis).ffill()

    out: dict[str, float] = {}
    for raw in cycle_dates:
        d = pd.Timestamp(raw)
        pos = int(axis.searchsorted(d, side="right"))
        if pos == 0:
            # No prior row to anchor SPY's leg against — the cycle predates
            # the price matrix entirely.
            continue
        if pos + horizon_days > len(axis):
            continue
        window = axis[pos: pos + horizon_days]
        anchor = axis[pos - 1]

        arm_leg = float(arm.loc[window].sum())

        spy_start = spy.loc[anchor]
        spy_end = spy.loc[window[-1]]
        if (
            pd.isna(spy_start)
            or pd.isna(spy_end)
            or float(spy_start) <= 0.0
            or float(spy_end) <= 0.0
        ):
            # A non-positive or missing SPY close makes the benchmark leg
            # undefined; the cycle is dropped from the sample (and its absence
            # is visible as a smaller n_samples on the artifact) rather than
            # being scored against a fabricated benchmark.
            logger.warning(
                "contribution_lift: SPY close undefined for cycle %s "
                "(anchor=%s end=%s) — cycle dropped",
                raw, anchor.date(), window[-1].date(),
            )
            continue
        spy_leg = math.log(float(spy_end) / float(spy_start))

        value = arm_leg - spy_leg
        if not math.isfinite(value):
            logger.warning(
                "contribution_lift: non-finite log-alpha for cycle %s — dropped", raw
            )
            continue
        out[pd.Timestamp(raw).strftime("%Y-%m-%d")] = value
    return out


def paired_diffs(
    baseline: dict[str, float], ablated: dict[str, float]
) -> list[float]:
    """``baseline − ablated`` over cycles where BOTH arms are defined.

    Paired, not two independent means: the arms share cohort dates and market
    regime, so pairing removes the market's own variance from the estimate and
    is what makes a 60-cycle sample informative at all.
    """
    shared = sorted(set(baseline) & set(ablated))
    return [float(baseline[d] - ablated[d]) for d in shared]


def paired_bootstrap_ci(diffs: list[float]) -> dict:
    """Percentile bootstrap over the paired per-cycle differences.

    Fixed ``seed=0`` — the artifact must be byte-reproducible for the same
    inputs (contract §Rules: no RNG except the bootstrap seed).
    """
    from nousergon_lib.quant.stats.intervals import bootstrap_ci

    result = bootstrap_ci(
        np.asarray(diffs, dtype=float),
        statistic=np.mean,
        ci_level=0.95,
        n_resamples=1000,
        seed=0,
    )
    if result.get("status") != "ok":
        raise ValueError(
            "bootstrap_ci over paired contribution-lift diffs returned "
            f"status={result.get('status')!r} for n={len(diffs)}"
        )
    return {
        "estimate": float(result["estimate"]),
        "ci_low": float(result["ci_low"]),
        "ci_high": float(result["ci_high"]),
        "ci_method": "bootstrap",
    }


def check_count_match(
    baseline_width: dict[str, int],
    ablated_width: dict[str, int],
    *,
    unpriceable: Collection[str] = (),
) -> CountMatch:
    """Both arms must declare the SAME dates, at identical picks-per-cycle.

    Spec §3: a comparison of arms of different width silently favors whichever
    arm trades more and cannot isolate the component's own effect. Such a
    replay reports a GAP naming the mismatch, never a lift.

    The check runs over the UNION of both arms' declared date supports, not
    their intersection (alpha-engine-config-I12243). Comparing only shared
    dates let a date that one arm traded and the other did not disappear from
    the assurance: ``baseline={d1: 2, d2: 2}`` vs ``ablated={d1: 2}`` returned
    ``matched=True``. Every union date is classed as exactly one of:

    * ``matched`` — both arms declare it at the same width;
    * ``mismatched`` — both declare it, at different widths;
    * ``baseline_only`` / ``ablated_only`` — one arm declares it, the other
      does not. This is unmatched support and is a gap;
    * ``unpriceable`` — named by the caller in ``unpriceable``: no arm that
      declares it can price a single one of its entries there, so neither
      simulated book trades on it and it cannot bias the comparison. It is
      excluded from matching and reported by name, never silently dropped.

    A zero width is no support (an arm that lists a date with no names trades
    exactly as one that omits it), so zero-width entries are normalized away
    before classing.
    """
    base = {str(d): int(w) for d, w in baseline_width.items() if int(w) > 0}
    abl = {str(d): int(w) for d, w in ablated_width.items() if int(w) > 0}
    skip = {str(d) for d in unpriceable}
    union = sorted(set(base) | set(abl))

    matched: list[str] = []
    mismatched: list[tuple[str, int, int]] = []
    baseline_only: list[str] = []
    ablated_only: list[str] = []
    unpriced: list[str] = []
    for d in union:
        if d in skip:
            unpriced.append(d)
        elif d in base and d in abl:
            if base[d] == abl[d]:
                matched.append(d)
            else:
                mismatched.append((d, base[d], abl[d]))
        elif d in base:
            baseline_only.append(d)
        else:
            ablated_only.append(d)

    def _verdict(ok: bool, reason: str) -> CountMatch:
        return CountMatch(
            matched=ok,
            reason=reason,
            matched_dates=tuple(matched),
            mismatched_dates=tuple(d for d, _b, _a in mismatched),
            baseline_only_dates=tuple(baseline_only),
            ablated_only_dates=tuple(ablated_only),
            unpriceable_dates=tuple(unpriced),
        )

    if not matched and not mismatched:
        return _verdict(
            False,
            "no cycle is present in both arms "
            f"(baseline={len(base)} cycles, ablated={len(abl)} cycles, "
            f"unpriceable={len(unpriced)}) — nothing to count-match",
        )

    problems: list[str] = []
    shared = len(matched) + len(mismatched)
    if mismatched:
        sample = "; ".join(
            f"{d}: baseline={b} vs ablated={a}" for d, b, a in mismatched[:3]
        )
        problems.append(
            f"width mismatch on {len(mismatched)}/{shared} shared cycles ({sample})"
        )
    if baseline_only or ablated_only:
        parts = []
        if baseline_only:
            parts.append(
                f"{len(baseline_only)} baseline-only date(s) "
                f"({', '.join(baseline_only[:3])})"
            )
        if ablated_only:
            parts.append(
                f"{len(ablated_only)} ablated-only date(s) "
                f"({', '.join(ablated_only[:3])})"
            )
        problems.append(
            "one-sided support: " + " and ".join(parts)
            + f" beside {shared} shared cycles"
        )
    if not problems:
        return _verdict(True, "")
    return _verdict(
        False,
        "; ".join(problems)
        + " — an unmatched comparison favors whichever arm trades more (or "
        "trades at all), so this is reported as a gap, not a measured lift",
    )


def compute_pbo(per_arm_alpha: dict[str, dict[str, float]]) -> dict | None:
    """CSCV-PBO over a grid's per-cycle log-alpha matrix.

    ``per_arm_alpha`` is ``{arm_label: {cycle_date: log_alpha}}``. Rows are the
    cycles present in EVERY arm (a ragged matrix would let an arm be judged on
    a friendlier subsample); columns are arms.

    Returns ``None`` when the grid is too small for CSCV to split — reported as
    an explicit null rather than a fabricated 0.0.
    """
    from nousergon_lib.quant.stats.pbo import cscv_pbo

    labels = sorted(per_arm_alpha)
    if len(labels) < 2:
        return None
    shared = sorted(set.intersection(*(set(per_arm_alpha[a]) for a in labels)))
    if len(shared) < 8:
        return None
    matrix = pd.DataFrame(
        {a: [per_arm_alpha[a][d] for d in shared] for a in labels},
        index=shared,
    )
    result = cscv_pbo(matrix)
    if result.get("status") != "ok":
        logger.warning(
            "contribution_lift: cscv_pbo returned status=%s over %d arms x %d "
            "cycles — pbo emitted as null",
            result.get("status"), len(labels), len(shared),
        )
        return None
    return {"pbo": float(result["pbo"]), "n_trials": len(labels)}


def window_overlap(
    sampled_dates: list[str],
    trading_axis: pd.DatetimeIndex,
    *,
    horizon_days: int,
) -> dict:
    """How much the sampled ``horizon_days`` windows overlap each other.

    Each paired sample is the whole-book log-alpha over the ``horizon_days``
    rows STRICTLY after a cycle date (:func:`per_cycle_log_alpha`). Two cycle
    dates fewer than ``horizon_days`` rows apart therefore share rows of the
    same NAV path, and their paired differences are not independent draws.
    :func:`paired_bootstrap_ci` resamples them iid, so this measures — it does
    not correct — how far the sample departs from that assumption
    (alpha-engine-config-I12243). The estimator is unchanged here; changing
    it is reserved to the critical-metric/estimator authority.

    Returns counts only — no inferred effective-N factor is fabricated:

    * ``n_windows`` — sampled windows (the CI's ``n_samples``);
    * ``n_adjacent_pairs`` / ``n_overlapping_adjacent_pairs`` — consecutive
      sampled windows, and how many share at least one row;
    * ``min/median/max_gap_trading_rows`` — row distance between consecutive
      window starts;
    * ``mean_overlap_fraction`` — mean shared-row fraction over adjacent pairs;
    * ``n_non_overlapping_windows`` — the largest set of sampled windows no two
      of which share a row (greedy from the earliest, which is optimal for
      equal-length intervals).
    """
    axis = pd.DatetimeIndex(trading_axis).sort_values()
    starts = sorted(
        int(axis.searchsorted(pd.Timestamp(d), side="right")) for d in sampled_dates
    )
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    h = int(horizon_days)
    overlapping = [g for g in gaps if g < h]
    non_overlapping = 0
    next_free = None
    for s in starts:
        if next_free is None or s >= next_free:
            non_overlapping += 1
            next_free = s + h
    return {
        "horizon_days": h,
        "n_windows": len(starts),
        "n_adjacent_pairs": len(gaps),
        "n_overlapping_adjacent_pairs": len(overlapping),
        "min_gap_trading_rows": min(gaps) if gaps else None,
        "median_gap_trading_rows": float(np.median(gaps)) if gaps else None,
        "max_gap_trading_rows": max(gaps) if gaps else None,
        "mean_overlap_fraction": (
            float(np.mean([max(0, h - g) / h for g in gaps])) if gaps else None
        ),
        "n_non_overlapping_windows": non_overlapping,
    }
