"""Purged & embargoed walk-forward fold splitter for point-in-time backtesting.

Second pure building block of PIT discipline (ROADMAP L2349 / Backtester
Phase 2, P1). Plan: ``alpha-engine-docs/private/pit-discipline-260515.md``.

Institutional basis (López de Prado, *Advances in Financial ML* Ch. 7):
walk-forward folds with a **purge** gap between train-end and test-start (so the
test window's label-formation period cannot leak into training) and an
**embargo** after each test window (so serially-correlated features just after a
test fold cannot leak into the next train set).

Consistency-with-production note (grounded against current code 2026-05-15):
the predictor's own walk-forward (``meta_trainer.py`` ~1058-1080) is
**expanding-train + purge** (``train_mask = d <= train_end_date`` — no lower
bound), advancing the fold start by one test window, with
``WF_TEST_WINDOW_DAYS`` / ``WF_MIN_TRAIN_DAYS`` / ``WF_PURGE_DAYS``. This module
mirrors that index logic exactly and *adds* the embargo (the predictor has purge
but no embargo). ``train_mode`` defaults to ``"expanding"`` to genuinely match
the predictor (the plan doc's "rolling matches predictor" wording is a
documentation error — predictor is expanding; the plan's *intent* of
predictor-consistency is honored here, and the doc should be corrected). A
``"rolling"`` mode is provided for the sweepable-variant the plan anticipated.

Pure: operates on an ordered list of unique trading dates, no I/O, no S3, no
sweep wiring — unit-tested in isolation before anything consumes it.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold over an ordered unique-date axis.

    Index fields are positions into the ``unique_dates`` list passed to
    :func:`build_walk_forward_folds`. ``train_end_date`` precedes
    ``test_start_date`` by at least ``purge`` trading days; the next fold's train
    set excludes the ``embargo`` days immediately after ``test_end_date``.
    """

    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int
    train_start_date: _dt.date
    train_end_date: _dt.date
    test_start_date: _dt.date
    test_end_date: _dt.date


def build_walk_forward_folds(
    unique_dates: list[_dt.date],
    *,
    test_window: int,
    min_train: int,
    purge: int,
    embargo: int,
    train_mode: str = "expanding",
) -> list[Fold]:
    """Build purged + embargoed walk-forward folds.

    Parameters mirror the predictor's WF config so the two stay consistent:
      - ``test_window``  -> WF_TEST_WINDOW_DAYS (e.g. ~21 trading days = 1mo)
      - ``min_train``    -> WF_MIN_TRAIN_DAYS (first fold's minimum train length)
      - ``purge``        -> WF_PURGE_DAYS; default per plan = canonical label
        horizon = 21 trading days (the test label window cannot touch train)
      - ``embargo``      -> trading days after each test window excluded from the
        *next* fold's train set (plan default = 2; LdP lower bound)
      - ``train_mode``   -> "expanding" (default, matches predictor:
        train = all dates <= train_end) or "rolling" (train =
        ``test_window``-bounded lookback ending at train_end)

    Returns folds in chronological order. A fold is emitted only if a valid
    train block of length >= ``min_train`` // 2 exists after purging — matching
    the predictor's guard so degenerate early folds are skipped rather than
    silently producing a tiny train set.

    Raises ValueError on non-positive sizing params or an unknown train_mode so
    a misconfigured sweep fails loud rather than producing leaky folds.
    """
    if test_window <= 0 or min_train <= 0:
        raise ValueError("test_window and min_train must be positive")
    if purge < 0 or embargo < 0:
        raise ValueError("purge and embargo must be non-negative")
    if train_mode not in ("expanding", "rolling"):
        raise ValueError(f"unknown train_mode {train_mode!r}")

    n = len(unique_dates)
    folds: list[Fold] = []
    fold_start_idx = min_train
    while fold_start_idx < n:
        remaining = n - fold_start_idx
        # Predictor guard: stop once less than half a test window remains.
        if remaining < test_window // 2:
            break

        test_start_idx = fold_start_idx
        test_end_idx = min(fold_start_idx + test_window - 1, n - 1)

        # Purge: train ends `purge` trading days before the test window opens.
        train_end_idx = fold_start_idx - purge
        if train_end_idx < min_train // 2:
            fold_start_idx += test_window
            continue

        if train_mode == "expanding":
            train_start_idx = 0
        else:  # rolling: bounded lookback of one test_window ending at train_end
            train_start_idx = max(0, train_end_idx - test_window + 1)

        if train_end_idx < train_start_idx:
            fold_start_idx += test_window
            continue

        folds.append(
            Fold(
                train_start_idx=train_start_idx,
                train_end_idx=train_end_idx,
                test_start_idx=test_start_idx,
                test_end_idx=test_end_idx,
                train_start_date=unique_dates[train_start_idx],
                train_end_date=unique_dates[train_end_idx],
                test_start_date=unique_dates[test_start_idx],
                test_end_date=unique_dates[test_end_idx],
            )
        )

        # Embargo: the next fold's train set must not include the `embargo`
        # trading days immediately after this test window. We advance the fold
        # cursor by one test window (predictor cadence) and the embargo is
        # enforced structurally because the next fold's train_end_idx =
        # next_fold_start - purge, and purge >= embargo in the canonical config
        # (purge=21, embargo=2) so the post-test embargo region is already
        # outside the next train block. When embargo > purge we additionally
        # push the cursor so the gap is at least `embargo`.
        advance = test_window
        if embargo > purge:
            advance = max(test_window, test_window + (embargo - purge))
        fold_start_idx += advance

    return folds
