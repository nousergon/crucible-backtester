"""Unit tests for the point-in-time weight-archive resolver (synthetic/pit_weights.py).

Locks the institutional PIT invariants before the resolver is wired into the
walk-forward sweep:
  - knowledge time <= decision time (strict <=, same-day allowed)
  - NO future fallback — missing snapshot raises ColdStartExclusion, never the
    nearest/earliest-future snapshot
  - defensive: malformed archive dirs skipped, pagination handled

S3 is mocked with unittest.mock per the repo convention
(tests/test_recommendation_artifact.py:195 uses s3.list_objects_v2.return_value).
"""

import datetime as dt
from unittest.mock import MagicMock

import pytest

from synthetic.pit_weights import (
    ColdStartExclusion,
    ResolvedWeights,
    resolve_momentum_weights,
    _ARCHIVE_PREFIX,
)


def _cp(*date_strs):
    """Build a single-page list_objects_v2 response with the given date dirs."""
    return {
        "CommonPrefixes": [
            {"Prefix": f"{_ARCHIVE_PREFIX}{d}/"} for d in date_strs
        ],
        "IsTruncated": False,
    }


def _s3_with(*date_strs):
    s3 = MagicMock()
    s3.list_objects_v2.return_value = _cp(*date_strs)
    return s3


def test_picks_latest_on_or_before_decision_date():
    s3 = _s3_with("2026-04-04", "2026-04-11", "2026-04-18", "2026-04-25")
    r = resolve_momentum_weights(s3, "bkt", dt.date(2026, 4, 20))
    assert isinstance(r, ResolvedWeights)
    assert r.archive_date == dt.date(2026, 4, 18)
    assert r.model_key == f"{_ARCHIVE_PREFIX}2026-04-18/momentum_model.txt"
    assert r.meta_key == f"{_ARCHIVE_PREFIX}2026-04-18/momentum_model.txt.meta.json"


def test_exact_match_date_is_eligible_same_day_allowed():
    s3 = _s3_with("2026-04-11", "2026-04-18")
    r = resolve_momentum_weights(s3, "bkt", dt.date(2026, 4, 18))
    assert r.archive_date == dt.date(2026, 4, 18)


def test_no_future_fallback_never_returns_a_later_snapshot():
    # Only future snapshots exist relative to the decision date.
    s3 = _s3_with("2026-05-02", "2026-05-09")
    with pytest.raises(ColdStartExclusion) as ei:
        resolve_momentum_weights(s3, "bkt", dt.date(2026, 4, 25))
    assert ei.value.decision_date == dt.date(2026, 4, 25)
    assert ei.value.n_archives == 2


def test_future_snapshots_ignored_when_an_eligible_one_exists():
    # The eligible 04-18 must win even though later snapshots are present.
    s3 = _s3_with("2026-04-18", "2026-05-02", "2026-05-09")
    r = resolve_momentum_weights(s3, "bkt", dt.date(2026, 4, 25))
    assert r.archive_date == dt.date(2026, 4, 18)


def test_empty_archive_prefix_is_cold_start():
    s3 = MagicMock()
    s3.list_objects_v2.return_value = {"IsTruncated": False}  # no CommonPrefixes
    with pytest.raises(ColdStartExclusion) as ei:
        resolve_momentum_weights(s3, "bkt", dt.date(2026, 4, 25))
    assert ei.value.n_archives == 0


def test_unparseable_and_invalid_dirs_are_skipped():
    s3 = _s3_with("latest", "2026-04-11", "2026-13-99", "_tmp", "2026-04-18")
    r = resolve_momentum_weights(s3, "bkt", dt.date(2026, 4, 30))
    # "latest"/"_tmp" non-date, "2026-13-99" shape-matches but invalid -> only
    # 04-11 and 04-18 are real; latest <= 04-30 is 04-18.
    assert r.archive_date == dt.date(2026, 4, 18)


def test_pagination_is_followed():
    s3 = MagicMock()
    page1 = {
        "CommonPrefixes": [{"Prefix": f"{_ARCHIVE_PREFIX}2026-04-04/"}],
        "IsTruncated": True,
        "NextContinuationToken": "tok1",
    }
    page2 = {
        "CommonPrefixes": [{"Prefix": f"{_ARCHIVE_PREFIX}2026-04-25/"}],
        "IsTruncated": False,
    }
    s3.list_objects_v2.side_effect = [page1, page2]
    r = resolve_momentum_weights(s3, "bkt", dt.date(2026, 4, 30))
    assert r.archive_date == dt.date(2026, 4, 25)
    # second call must have passed the continuation token through
    _, kwargs = s3.list_objects_v2.call_args_list[1]
    assert kwargs["ContinuationToken"] == "tok1"


def test_listing_uses_delimiter_to_avoid_object_fanout():
    s3 = _s3_with("2026-04-18")
    resolve_momentum_weights(s3, "bkt", dt.date(2026, 4, 20))
    _, kwargs = s3.list_objects_v2.call_args_list[0]
    assert kwargs["Delimiter"] == "/"
    assert kwargs["Prefix"] == _ARCHIVE_PREFIX
