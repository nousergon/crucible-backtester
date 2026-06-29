"""Tests for analysis/attribution_persistence.py (config#946 part 2)."""

from __future__ import annotations

from unittest.mock import patch

from analysis.attribution_persistence import (
    ATTRIBUTION_INSUFFICIENT_PERSISTENCE_ALERT,
    count_trailing_insufficient,
    evaluate_attribution_persistence,
    record_and_evaluate,
    render_attribution_persistence_section,
)

INSUF = "insufficient_data"


# ── count_trailing_insufficient ──────────────────────────────────────────────

def test_count_trailing_all_insufficient():
    assert count_trailing_insufficient([INSUF, INSUF, INSUF]) == 3


def test_count_trailing_resets_on_ok():
    # an `ok` in the middle resets the clock — only the trailing run counts.
    assert count_trailing_insufficient([INSUF, INSUF, "ok", INSUF]) == 1


def test_count_trailing_zero_when_latest_ok():
    assert count_trailing_insufficient([INSUF, INSUF, "ok"]) == 0


def test_count_trailing_empty():
    assert count_trailing_insufficient([]) == 0


def test_count_trailing_ignores_other_statuses_as_breaks():
    # a non-insufficient terminal status (skipped) breaks the streak.
    assert count_trailing_insufficient([INSUF, "skipped"]) == 0


# ── evaluate_attribution_persistence ─────────────────────────────────────────

def test_persistent_at_threshold():
    statuses = [INSUF] * ATTRIBUTION_INSUFFICIENT_PERSISTENCE_ALERT
    r = evaluate_attribution_persistence(statuses)
    assert r["persistent"] is True
    assert r["consecutive_insufficient"] == ATTRIBUTION_INSUFFICIENT_PERSISTENCE_ALERT
    assert r["latest_status"] == INSUF


def test_not_persistent_below_threshold():
    statuses = [INSUF] * (ATTRIBUTION_INSUFFICIENT_PERSISTENCE_ALERT - 1)
    r = evaluate_attribution_persistence(statuses)
    assert r["persistent"] is False
    assert r["consecutive_insufficient"] == ATTRIBUTION_INSUFFICIENT_PERSISTENCE_ALERT - 1


def test_custom_threshold():
    r = evaluate_attribution_persistence([INSUF, INSUF], threshold=2)
    assert r["persistent"] is True
    assert r["threshold"] == 2


def test_ok_latest_not_persistent():
    r = evaluate_attribution_persistence([INSUF, INSUF, INSUF, "ok"], threshold=2)
    assert r["persistent"] is False
    assert r["consecutive_insufficient"] == 0
    assert r["latest_status"] == "ok"


# ── record_and_evaluate (S3 history mocked) ──────────────────────────────────

def _attr(status, n):
    if status == "ok":
        return {"status": "ok", "rows_analyzed": n}
    return {"status": INSUF, "rows_populated": n}


@patch("analysis.attribution_persistence._persist_history")
@patch("analysis.attribution_persistence._load_history")
def test_record_includes_current_cycle(mock_load, mock_persist):
    # 3 prior insufficient cycles in history; this cycle is the 4th → fires.
    mock_load.return_value = [
        {"date": "2026-06-06", "status": INSUF, "n": 40},
        {"date": "2026-06-13", "status": INSUF, "n": 55},
        {"date": "2026-06-20", "status": INSUF, "n": 70},
    ]
    r = record_and_evaluate(
        _attr(INSUF, 80), run_date="2026-06-27", bucket="b", upload=True,
        threshold=4,
    )
    assert r["consecutive_insufficient"] == 4
    assert r["persistent"] is True
    assert r["latest_n"] == 80
    mock_persist.assert_called_once()
    # history persisted with the new cycle appended (4 rows).
    persisted_rows = mock_persist.call_args[0][1]
    assert len(persisted_rows) == 4
    assert persisted_rows[-1]["date"] == "2026-06-27"


@patch("analysis.attribution_persistence._persist_history")
@patch("analysis.attribution_persistence._load_history")
def test_record_idempotent_same_date(mock_load, mock_persist):
    # a same-date retry replaces the prior row rather than inflating the streak.
    mock_load.return_value = [
        {"date": "2026-06-20", "status": INSUF, "n": 70},
        {"date": "2026-06-27", "status": INSUF, "n": 75},
    ]
    r = record_and_evaluate(
        _attr(INSUF, 80), run_date="2026-06-27", bucket="b", upload=True,
        threshold=4,
    )
    persisted_rows = mock_persist.call_args[0][1]
    assert len(persisted_rows) == 2  # not 3 — same-date row replaced
    assert persisted_rows[-1]["n"] == 80
    assert r["consecutive_insufficient"] == 2


@patch("analysis.attribution_persistence._persist_history")
@patch("analysis.attribution_persistence._load_history")
def test_record_no_write_when_not_uploading(mock_load, mock_persist):
    mock_load.return_value = []
    r = record_and_evaluate(
        _attr(INSUF, 80), run_date="2026-06-27", bucket="b", upload=False,
    )
    mock_persist.assert_not_called()
    # current cycle still counted for the evaluation even without a write.
    assert r["consecutive_insufficient"] == 1


@patch("analysis.attribution_persistence._persist_history")
@patch("analysis.attribution_persistence._load_history")
def test_record_ok_cycle_resets(mock_load, mock_persist):
    mock_load.return_value = [
        {"date": "2026-06-13", "status": INSUF, "n": 55},
        {"date": "2026-06-20", "status": INSUF, "n": 70},
    ]
    r = record_and_evaluate(
        _attr("ok", 130), run_date="2026-06-27", bucket="b", upload=True,
        threshold=2,
    )
    assert r["consecutive_insufficient"] == 0
    assert r["persistent"] is False
    assert r["latest_n"] == 130


@patch("analysis.attribution_persistence._persist_history")
@patch("analysis.attribution_persistence._load_history")
def test_record_handles_skipped_attribution(mock_load, mock_persist):
    mock_load.return_value = []
    r = record_and_evaluate(
        {"status": "skipped"}, run_date="2026-06-27", bucket="b", upload=False,
    )
    assert r["latest_status"] == "skipped"
    assert r["consecutive_insufficient"] == 0


# ── render section ───────────────────────────────────────────────────────────

def test_render_warns_when_persistent():
    md = render_attribution_persistence_section({
        "consecutive_insufficient": 4, "threshold": 4, "persistent": True,
        "latest_status": INSUF, "latest_n": 80,
    })
    assert "## Attribution sample adequacy (persistence)" in md
    assert "⚠️" in md
    assert "4 consecutive" in md


def test_render_monitoring_when_below_threshold():
    md = render_attribution_persistence_section({
        "consecutive_insufficient": 2, "threshold": 4, "persistent": False,
        "latest_status": INSUF, "latest_n": 80,
    })
    assert "⚠️" not in md
    assert "2/4 consecutive" in md


def test_render_reset_on_ok():
    md = render_attribution_persistence_section({
        "consecutive_insufficient": 0, "threshold": 4, "persistent": False,
        "latest_status": "ok", "latest_n": 130,
    })
    assert "well-powered" in md
    assert "reset" in md.lower()


def test_render_never_raises_on_missing_keys():
    # always-emit contract: a degraded result dict still renders a section.
    md = render_attribution_persistence_section({})
    assert "## Attribution sample adequacy (persistence)" in md
