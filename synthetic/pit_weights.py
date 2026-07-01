"""Point-in-time resolution of archived predictor weights for walk-forward backtesting.

.. note::
   **Currently unused by the momentum leg (config#1518, 2026-07-01).** The Layer-1A
   momentum component retired its trained GBM on 2026-05-09 and is now a fixed
   deterministic formula (``crucible-predictor/model/momentum_scorer.py``), which
   carries zero look-ahead risk and needs no per-fold archived booster —
   ``run_walk_forward_inference`` no longer calls :func:`resolve_momentum_weights`.
   This module is retained (not deleted) as the tested reference implementation of
   the no-future-fallback PIT invariant, so any *future* dated-weight artifact that
   genuinely needs point-in-time resolution can reuse it. Do not repurpose silently.

Enforces the cardinal PIT invariant: at simulated decision date ``D``, only weights
whose *knowledge time* <= ``D`` may be used. Plan:
``alpha-engine-docs/private/pit-discipline-260515.md`` (ROADMAP L2349 / Backtester
Phase 2, P1).

The predictor archives promoted weights every Saturday under
``predictor/weights/meta/archive/{YYYY-MM-DD}/`` (``meta_trainer.py:2237`` writes
``{prefix}archive/{date_str}/{filename}``; ``date_str`` is
``train_handler.py:886`` -> ``datetime.now().strftime("%Y-%m-%d")``). This module
resolves the *latest archive dir with date <= decision_date* and **refuses any
future fallback**: if none exists the fold is a cold-start exclusion (the
walk-forward caller skips it), NEVER the nearest or earliest-future snapshot —
substituting a future snapshot is exactly how look-ahead silently re-enters a
backtest (the central trap, plan D2 / invariant 3).

This module is intentionally pure (one injected S3 client, no global state, no
wiring into the sweep) so the highest-risk PIT design decision is unit-tested in
isolation before anything consumes it.
"""

from __future__ import annotations

import datetime as _dt
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_ARCHIVE_PREFIX = "predictor/weights/meta/archive/"
_MOMENTUM_FILENAME = "momentum_model.txt"
# Archive dirs are written via strftime("%Y-%m-%d"); match exactly so a stray
# non-date CommonPrefix (e.g. a future schema change) can never be misread as a date.
_DATE_DIR_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})$")


class ColdStartExclusion(Exception):
    """No archived weights exist with knowledge time <= the decision date.

    Raised (never silently substituted) so the walk-forward caller excludes the
    fold as cold-start and counts it as a run-quality metric. Returning the
    nearest / earliest-future snapshot instead would reintroduce look-ahead — the
    no-future-fallback invariant (plan invariant 3 / D2).
    """

    def __init__(self, decision_date: _dt.date, n_archives: int):
        self.decision_date = decision_date
        self.n_archives = n_archives
        super().__init__(
            f"No predictor weight archive with date <= {decision_date.isoformat()} "
            f"({n_archives} archive snapshot(s) exist, all later) — "
            f"fold excluded as cold-start."
        )


@dataclass(frozen=True)
class ResolvedWeights:
    """Result of a point-in-time weight resolution.

    ``archive_date`` is the knowledge date of the chosen snapshot and is always
    ``<= decision_date``; callers should log it so a parity report can show which
    snapshot each fold actually replayed.
    """

    archive_date: _dt.date
    model_key: str
    meta_key: str


def _list_archive_dates(
    s3, bucket: str, prefix: str = _ARCHIVE_PREFIX
) -> list[_dt.date]:
    """Return all parseable ``YYYY-MM-DD`` archive snapshot dates, ascending.

    Uses ``Delimiter='/'`` so only the date-directory ``CommonPrefixes`` are
    listed (no per-object fan-out across every weight file). Unparseable
    directory names are skipped defensively rather than raising — a malformed or
    future-schema dir must not crash a PIT resolution.
    """
    dates: list[_dt.date] = []
    token: str | None = None
    while True:
        kwargs: dict = {"Bucket": bucket, "Prefix": prefix, "Delimiter": "/"}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for cp in resp.get("CommonPrefixes", []) or []:
            tail = cp.get("Prefix", "")[len(prefix):].rstrip("/")
            m = _DATE_DIR_RE.match(tail)
            if not m:
                logger.debug("pit_weights: skipping non-date archive dir %r", tail)
                continue
            try:
                dates.append(_dt.date.fromisoformat(m.group(1)))
            except ValueError:
                # e.g. "2026-13-99" matches the shape but is not a real date.
                logger.debug("pit_weights: skipping invalid archive date %r", tail)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return sorted(dates)


def resolve_momentum_weights(
    s3,
    bucket: str,
    decision_date: _dt.date,
    *,
    prefix: str = _ARCHIVE_PREFIX,
) -> ResolvedWeights:
    """Resolve the point-in-time momentum-GBM weight keys for ``decision_date``.

    Returns the snapshot with the greatest archive date ``D_a`` such that
    ``D_a <= decision_date`` (knowledge time <= decision time). Same-day is
    permitted: the predictor's Saturday training completes pre-market (Saturday
    SF cron 09:00 UTC) before any weekday decision in the simulation consumes
    weights, so ``D_a == decision_date`` carries no look-ahead.

    Raises :class:`ColdStartExclusion` if no eligible snapshot exists. Never
    falls back to a later snapshot.
    """
    dates = _list_archive_dates(s3, bucket, prefix)
    eligible = [d for d in dates if d <= decision_date]
    if not eligible:
        raise ColdStartExclusion(decision_date, len(dates))
    chosen = eligible[-1]  # ascending -> last eligible is the latest <= decision_date
    base = f"{prefix}{chosen.isoformat()}/"
    return ResolvedWeights(
        archive_date=chosen,
        model_key=f"{base}{_MOMENTUM_FILENAME}",
        meta_key=f"{base}{_MOMENTUM_FILENAME}.meta.json",
    )
