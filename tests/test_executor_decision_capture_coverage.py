"""Tests for analysis.executor_decision_capture_coverage (L2308 PR 5).

Mirrors the test surface of tests/test_decision_capture_coverage.py for
the research-side sibling. Covers:
- Most-recent-weekday resolution (walks back ≤7 days, skips Sat/Sun)
- Insufficient_data when env-flag wasn't enabled / no artifacts in window
- Per-component coverage counting against the 4 canonical components
- Uncategorized executor:* components flagged for visibility
- Meta-prefix filtering (_eval/_replay/_cost etc. excluded)
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from analysis.executor_decision_capture_coverage import (
    EXECUTOR_COMPONENTS,
    N_CANONICAL,
    compute_executor_decision_capture_coverage,
)


def _make_s3_paginate_response(keys: list[str]) -> list[dict]:
    """Simulate a single-page paginator response with the given keys."""
    return [{"Contents": [{"Key": k} for k in keys]}]


def _make_s3_stub(keys_by_prefix: dict[str, list[str]]) -> MagicMock:
    """Stub S3 client where paginator.paginate(Bucket, Prefix) returns
    the keys associated with that prefix in the dict."""
    s3 = MagicMock()
    paginator = MagicMock()
    s3.get_paginator.return_value = paginator

    def _paginate(*, Bucket: str, Prefix: str):
        keys = keys_by_prefix.get(Prefix, [])
        return _make_s3_paginate_response(keys)

    paginator.paginate.side_effect = _paginate
    return s3


def _prefix_for(date_str: str) -> str:
    """``decision_artifacts/{Y}/{M}/{D}/`` prefix."""
    y, m, d = date_str.split("-")
    return f"decision_artifacts/{y}/{m}/{d}/"


# ── Sanity ───────────────────────────────────────────────────────────────


class TestCanonicalComponents:
    def test_canonical_components_locked(self):
        # If a new producer is added to the L2308 arc (e.g. PR 4b's
        # planner-side exit_rules), this test must be updated alongside
        # the producer PR — flag if the canonical set drifts.
        assert EXECUTOR_COMPONENTS == (
            "executor:entry_triggers",
            "executor:position_sizer",
            "executor:risk_guard",
            "executor:exit_rules",
        )
        assert N_CANONICAL == 4


# ── Insufficient data ────────────────────────────────────────────────────


class TestInsufficientData:
    def test_empty_window_returns_insufficient_data(self):
        # No keys at any date partition in the lookback → insufficient.
        s3 = _make_s3_stub({})
        result = compute_executor_decision_capture_coverage(
            run_date="2026-05-15",
            s3_client=s3,
            max_lookback_days=7,
        )
        assert result["status"] == "insufficient_data"
        assert "ALPHA_ENGINE_DECISION_CAPTURE_ENABLED" in result["reason"]
        assert result["lookback_days"] == 7
        assert result["end_date"] == "2026-05-15"

    def test_only_research_artifacts_returns_insufficient_data(self):
        # Research-side artifacts present (sector_quant etc.) but zero
        # executor:* — module is executor-scoped and ignores research
        # captures, so this still resolves to insufficient_data.
        keys = [
            f"{_prefix_for('2026-05-15')}sector_quant:technology/run-1.json",
            f"{_prefix_for('2026-05-15')}macro_economist/run-1.json",
        ]
        # 2026-05-15 is Friday (weekday 4); module looks at this date.
        s3 = _make_s3_stub({_prefix_for("2026-05-15"): keys})
        result = compute_executor_decision_capture_coverage(
            run_date="2026-05-15",
            s3_client=s3,
        )
        assert result["status"] == "insufficient_data"

    def test_skips_weekends_in_lookback(self):
        # 2026-05-16 = Sat, 2026-05-17 = Sun. Module should skip both
        # when walking back. Artifacts only on Friday 5/15.
        keys = [
            f"{_prefix_for('2026-05-15')}executor:entry_triggers/run-1.json",
        ]
        s3 = _make_s3_stub({_prefix_for("2026-05-15"): keys})
        # Asking from Sunday 5/17 should find Friday 5/15 (skip Sat 5/16,
        # also skip Sun 5/17 since it's a weekend).
        result = compute_executor_decision_capture_coverage(
            run_date="2026-05-17",
            s3_client=s3,
            max_lookback_days=7,
        )
        assert result["status"] == "ok"
        assert result["date"] == "2026-05-15"


# ── Coverage computation ─────────────────────────────────────────────────


class TestCoverageComputation:
    def test_full_coverage_4_of_4(self):
        # All 4 canonical executor components present on 5/15.
        prefix = _prefix_for("2026-05-15")
        keys = [
            f"{prefix}executor:entry_triggers/run-1.json",
            f"{prefix}executor:position_sizer/run-1.json",
            f"{prefix}executor:risk_guard/run-1.json",
            f"{prefix}executor:exit_rules/run-1.json",
        ]
        s3 = _make_s3_stub({prefix: keys})
        result = compute_executor_decision_capture_coverage(
            run_date="2026-05-15", s3_client=s3,
        )
        assert result["status"] == "ok"
        assert result["coverage_pct"] == 100.0
        assert result["n_canonical_present"] == 4
        assert result["n_canonical_expected"] == 4
        assert result["total_artifacts"] == 4
        for component in EXECUTOR_COMPONENTS:
            assert result["per_component"][component]["present"] is True
            assert result["per_component"][component]["n_artifacts"] == 1

    def test_partial_coverage_2_of_4(self):
        prefix = _prefix_for("2026-05-15")
        keys = [
            f"{prefix}executor:entry_triggers/run-1.json",
            f"{prefix}executor:entry_triggers/run-2.json",  # multi
            f"{prefix}executor:exit_rules/run-1.json",
        ]
        s3 = _make_s3_stub({prefix: keys})
        result = compute_executor_decision_capture_coverage(
            run_date="2026-05-15", s3_client=s3,
        )
        assert result["status"] == "ok"
        assert result["coverage_pct"] == 50.0
        assert result["n_canonical_present"] == 2
        assert result["total_artifacts"] == 3
        assert result["per_component"]["executor:entry_triggers"]["n_artifacts"] == 2
        assert result["per_component"]["executor:exit_rules"]["n_artifacts"] == 1
        assert result["per_component"]["executor:position_sizer"]["present"] is False
        assert result["per_component"]["executor:risk_guard"]["present"] is False

    def test_uncategorized_executor_components_flagged(self):
        """A new executor:* producer shipped without updating canonical
        set should surface in uncategorized_executor_components."""
        prefix = _prefix_for("2026-05-15")
        keys = [
            f"{prefix}executor:entry_triggers/run-1.json",
            f"{prefix}executor:future_producer/run-1.json",
        ]
        s3 = _make_s3_stub({prefix: keys})
        result = compute_executor_decision_capture_coverage(
            run_date="2026-05-15", s3_client=s3,
        )
        assert result["status"] == "ok"
        assert result["uncategorized_executor_components"] == [
            "executor:future_producer",
        ]

    def test_meta_prefixes_filtered(self):
        """`_eval/`, `_replay/`, etc. are not real captures — must be
        excluded from the coverage listing."""
        prefix = _prefix_for("2026-05-15")
        keys = [
            f"{prefix}executor:entry_triggers/run-1.json",
            # Meta-prefix entries that should be ignored:
            f"{prefix}_eval/executor:entry_triggers/run-1.json",
            f"{prefix}_replay/executor:entry_triggers/run-1.json",
            f"{prefix}_cost/raw/run-1.json",
        ]
        s3 = _make_s3_stub({prefix: keys})
        result = compute_executor_decision_capture_coverage(
            run_date="2026-05-15", s3_client=s3,
        )
        assert result["status"] == "ok"
        assert result["total_artifacts"] == 1
        assert result["per_component"]["executor:entry_triggers"]["n_artifacts"] == 1

    def test_research_artifacts_excluded_from_executor_listing(self):
        """Research-side agents (macro_economist, sector_*, ic_cio) live
        in the same date partition — module must exclude them by
        checking the executor:* prefix on agent_id."""
        prefix = _prefix_for("2026-05-15")
        keys = [
            f"{prefix}executor:entry_triggers/run-1.json",
            f"{prefix}macro_economist/run-1.json",
            f"{prefix}ic_cio/run-1.json",
            f"{prefix}sector_quant:technology/run-1.json",
            f"{prefix}thesis_update:tech:AAPL/run-1.json",
        ]
        s3 = _make_s3_stub({prefix: keys})
        result = compute_executor_decision_capture_coverage(
            run_date="2026-05-15", s3_client=s3,
        )
        assert result["total_artifacts"] == 1
        assert result["uncategorized_executor_components"] == []


# ── Lookback resolution ──────────────────────────────────────────────────


class TestLookbackResolution:
    def test_returns_most_recent_weekday(self):
        # Captures present on 5/13 (Wed) and 5/14 (Thu); query from
        # 5/15 (Fri) with no captures on 5/15 → returns 5/14 (most recent).
        s3 = _make_s3_stub({
            _prefix_for("2026-05-14"): [
                f"{_prefix_for('2026-05-14')}executor:entry_triggers/run-1.json",
            ],
            _prefix_for("2026-05-13"): [
                f"{_prefix_for('2026-05-13')}executor:entry_triggers/run-1.json",
            ],
        })
        result = compute_executor_decision_capture_coverage(
            run_date="2026-05-15", s3_client=s3, max_lookback_days=7,
        )
        assert result["status"] == "ok"
        assert result["date"] == "2026-05-14"

    def test_lookback_bound_respected(self):
        # Captures only on 5/01 — outside default 7-day lookback from 5/15.
        s3 = _make_s3_stub({
            _prefix_for("2026-05-01"): [
                f"{_prefix_for('2026-05-01')}executor:entry_triggers/run-1.json",
            ],
        })
        result = compute_executor_decision_capture_coverage(
            run_date="2026-05-15", s3_client=s3, max_lookback_days=7,
        )
        assert result["status"] == "insufficient_data"


# ── Invalid run_date ─────────────────────────────────────────────────────


class TestInvalidRunDate:
    def test_invalid_run_date_falls_back_to_today(self):
        """Defensive: malformed run_date should log + fall back to today
        rather than crash the evaluator."""
        s3 = _make_s3_stub({})
        result = compute_executor_decision_capture_coverage(
            run_date="not-a-date",
            s3_client=s3,
            max_lookback_days=7,
        )
        # Falls back to today (UTC) — empty bucket → insufficient_data.
        # The fact that we got a clean dict back (not a crash) is the
        # load-bearing assertion.
        assert result["status"] == "insufficient_data"
