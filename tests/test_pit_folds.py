"""Unit tests for the purged/embargoed walk-forward fold splitter.

Locks the institutional invariants before wiring:
  - purge gap: train_end strictly precedes test_start by >= purge positions
  - embargo: next fold's train block does not include the embargo region after
    the prior test window
  - expanding (default, matches predictor meta_trainer.py) vs rolling modes
  - predictor-parity guards (min_train, half-window stop) reproduced
  - misconfiguration fails loud
"""

import datetime as dt

import pytest

from synthetic.pit_folds import Fold, build_walk_forward_folds


def _days(n, start=dt.date(2024, 1, 1)):
    # Contiguous calendar days are fine — the splitter operates on an ordered
    # unique-date *axis*, not on calendar arithmetic (positions are what matter).
    return [start + dt.timedelta(days=i) for i in range(n)]


def test_purge_gap_enforced_between_train_end_and_test_start():
    dates = _days(400)
    folds = build_walk_forward_folds(
        dates, test_window=21, min_train=120, purge=21, embargo=2
    )
    assert folds, "expected at least one fold"
    for f in folds:
        # test_start_idx - train_end_idx must be >= purge (21)
        assert f.test_start_idx - f.train_end_idx >= 21
        assert f.train_end_date < f.test_start_date


def test_expanding_is_default_and_matches_predictor_shape():
    dates = _days(400)
    folds = build_walk_forward_folds(
        dates, test_window=21, min_train=120, purge=21, embargo=2
    )
    # expanding => every fold trains from index 0
    assert all(f.train_start_idx == 0 for f in folds)
    assert all(f.train_start_date == dates[0] for f in folds)


def test_rolling_mode_bounds_the_train_window():
    dates = _days(400)
    folds = build_walk_forward_folds(
        dates, test_window=21, min_train=120, purge=21, embargo=2,
        train_mode="rolling",
    )
    assert folds
    for f in folds:
        # rolling train block is at most one test_window long
        assert f.train_end_idx - f.train_start_idx + 1 <= 21
        assert f.train_start_idx > 0  # not anchored at 0 like expanding


def test_test_windows_are_chronological_and_sized():
    dates = _days(300)
    folds = build_walk_forward_folds(
        dates, test_window=21, min_train=120, purge=21, embargo=2
    )
    prev_end = -1
    for f in folds:
        assert f.test_end_idx >= f.test_start_idx
        assert f.test_end_idx - f.test_start_idx + 1 <= 21
        assert f.test_start_idx > prev_end  # non-overlapping, advancing
        prev_end = f.test_end_idx


def test_embargo_larger_than_purge_increases_fold_advance():
    dates = _days(400)
    base = build_walk_forward_folds(
        dates, test_window=21, min_train=120, purge=21, embargo=2
    )
    big_embargo = build_walk_forward_folds(
        dates, test_window=21, min_train=120, purge=21, embargo=40
    )
    # embargo (40) > purge (21) must space folds out further => fewer folds
    assert len(big_embargo) < len(base)


def test_min_train_guard_skips_degenerate_early_folds():
    dates = _days(200)
    folds = build_walk_forward_folds(
        dates, test_window=21, min_train=120, purge=21, embargo=2
    )
    # first fold's train_end must clear the predictor min_train//2 guard
    assert folds
    assert folds[0].train_end_idx >= 120 // 2


def test_short_history_yields_no_folds_rather_than_leaky_ones():
    dates = _days(60)  # < min_train
    folds = build_walk_forward_folds(
        dates, test_window=21, min_train=120, purge=21, embargo=2
    )
    assert folds == []


@pytest.mark.parametrize(
    "kw",
    [
        {"test_window": 0, "min_train": 120, "purge": 21, "embargo": 2},
        {"test_window": 21, "min_train": 0, "purge": 21, "embargo": 2},
        {"test_window": 21, "min_train": 120, "purge": -1, "embargo": 2},
        {"test_window": 21, "min_train": 120, "purge": 21, "embargo": -1},
    ],
)
def test_invalid_sizing_params_fail_loud(kw):
    with pytest.raises(ValueError):
        build_walk_forward_folds(_days(400), **kw)


def test_unknown_train_mode_fails_loud():
    with pytest.raises(ValueError):
        build_walk_forward_folds(
            _days(400), test_window=21, min_train=120, purge=21, embargo=2,
            train_mode="sliding",
        )


def test_returns_fold_dataclass_instances():
    folds = build_walk_forward_folds(
        _days(300), test_window=21, min_train=120, purge=21, embargo=2
    )
    assert all(isinstance(f, Fold) for f in folds)
