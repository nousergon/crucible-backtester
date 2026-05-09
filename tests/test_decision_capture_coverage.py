"""Unit tests for analysis.decision_capture_coverage.

Phase 2 transparency-inventory — closes the *agent decisions* row in
the gate checklist. The metric this module emits is the canonical
8-agent capture-coverage % per Saturday SF run, plus a rolling N-week
average. Tests here lock the contract:

- Full canonical set captured → 100% coverage
- Missing canonical agent → coverage drops by 12.5% per missing agent
- thesis_update reported as count, NOT in coverage denominator
- Uncategorized agent_ids surfaced for visibility
- Walk-back to the most-recent Saturday when run_date isn't itself a
  Saturday with captures
- ``status="no_recent_sf_run"`` when the trailing 7-day window is empty
- Rolling window aggregates skip empty Saturdays (the SF didn't run)
  rather than counting them as 0% (which would tank the rolling mean
  with non-runs).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from analysis.decision_capture_coverage import (
    CANONICAL_AGENTS,
    CANONICAL_SECTORS,
    N_CANONICAL,
    SECTOR_SUB_STAGES,
    compute_decision_capture_coverage,
    _list_agent_artifact_counts,
    _saturday_coverage_for,
)


# ── S3 stub matching tests/test_replay_batch.py shape ────────────────────────


def _build_s3_stub(keys: list[str]) -> MagicMock:
    """Stub S3 client with paginate() returning the keys whose path
    starts with the requested Prefix."""
    s3 = MagicMock()
    paginator = MagicMock()

    def paginate(*, Bucket, Prefix):
        return [{
            "Contents": [{"Key": k} for k in keys if k.startswith(Prefix)],
        }]

    paginator.paginate.side_effect = paginate
    s3.get_paginator.return_value = paginator
    return s3


# Helper: build the full per-stage key set for a given Saturday partition.
# 1 macro + 1 ic_cio + 6 sectors × 3 sub-stages = 20 artifact keys, which
# virtualize to 8 canonical agents (sector_team:{sector} present iff all
# 3 sub-stages emitted).
def _canonical_keys(date: str) -> list[str]:
    """date format: 'YYYY/MM/DD'."""
    keys = [
        f"decision_artifacts/{date}/macro_economist/run.json",
        f"decision_artifacts/{date}/ic_cio/run.json",
    ]
    for sector in CANONICAL_SECTORS:
        for stage in SECTOR_SUB_STAGES:
            keys.append(f"decision_artifacts/{date}/{stage}:{sector}/run.json")
    return keys


def _sub_stage_keys(date: str, sector: str) -> list[str]:
    """All 3 sub-stage keys for one sector on one Saturday."""
    return [
        f"decision_artifacts/{date}/{stage}:{sector}/run.json"
        for stage in SECTOR_SUB_STAGES
    ]


# ── Constants sanity ─────────────────────────────────────────────────────────


def test_canonical_set_is_eight_agents():
    """The canonical set is the gate denominator. Locked at 8 (1 macro
    + 1 ic_cio + 6 sector_team). Changing this is a deliberate scope
    decision, not an accident — so the test is a reminder."""
    assert N_CANONICAL == 8
    assert "macro_economist" in CANONICAL_AGENTS
    assert "ic_cio" in CANONICAL_AGENTS
    assert sum(1 for a in CANONICAL_AGENTS if a.startswith("sector_team:")) == 6


# ── _list_agent_artifact_counts ──────────────────────────────────────────────


class TestListAgentArtifactCounts:
    def test_groups_by_agent_id(self):
        from datetime import datetime
        keys = _canonical_keys("2026/05/02") + [
            "decision_artifacts/2026/05/02/sector_quant:technology/run2.json",
        ]
        s3 = _build_s3_stub(keys)
        counts = _list_agent_artifact_counts(
            s3, bucket="b", capture_prefix="decision_artifacts",
            date=datetime(2026, 5, 2),
        )
        # 1 macro + 1 ic_cio + 18 sub-stage entries, technology quant has 2 runs.
        assert counts["macro_economist"] == 1
        assert counts["sector_quant:technology"] == 2
        assert counts["sector_qual:technology"] == 1
        assert counts["sector_peer_review:technology"] == 1

    def test_excludes_meta_prefixes(self):
        from datetime import datetime
        keys = _canonical_keys("2026/05/02") + [
            "decision_artifacts/2026/05/02/_eval/x.json",
            "decision_artifacts/2026/05/02/_eval_judge_only/y.json",
            "decision_artifacts/2026/05/02/_replay/z.json",
            "decision_artifacts/2026/05/02/_replay_summary/w.json",
            "decision_artifacts/2026/05/02/_cost/c.json",
            "decision_artifacts/2026/05/02/_cost_raw/r.json",
            "decision_artifacts/2026/05/02/_analysis/a.json",
            "decision_artifacts/2026/05/02/_diagnostics/d.json",
        ]
        s3 = _build_s3_stub(keys)
        counts = _list_agent_artifact_counts(
            s3, bucket="b", capture_prefix="decision_artifacts",
            date=datetime(2026, 5, 2),
        )
        # No "_eval", "_replay", etc. agent_ids leaked through.
        assert "_eval" not in counts
        assert "_replay" not in counts
        assert "_cost" not in counts
        assert "_analysis" not in counts
        # 1 macro + 1 ic_cio + 6 sectors × 3 sub-stages = 20 canonical artifacts.
        assert sum(counts.values()) == 20

    def test_empty_partition_returns_empty_dict(self):
        from datetime import datetime
        s3 = _build_s3_stub([])
        counts = _list_agent_artifact_counts(
            s3, bucket="b", capture_prefix="decision_artifacts",
            date=datetime(2026, 5, 2),
        )
        assert counts == {}


# ── _saturday_coverage_for ───────────────────────────────────────────────────


class TestSaturdayCoverageFor:
    def test_full_canonical_set_is_100_pct(self):
        from datetime import datetime
        s3 = _build_s3_stub(_canonical_keys("2026/05/02"))
        result = _saturday_coverage_for(
            s3, bucket="b", capture_prefix="decision_artifacts",
            saturday=datetime(2026, 5, 2),
        )
        assert result["coverage_pct"] == 100.0
        assert result["n_canonical_present"] == 8
        assert result["n_canonical_expected"] == 8
        assert all(v["present"] for v in result["per_agent"].values())

    def test_missing_one_sector_team_is_875_pct(self):
        """7/8 canonical agents present → 87.5%. Dropping all 3
        sub-stages of one sector should fail that sector's virtual
        ``present`` check."""
        from datetime import datetime
        keys = _canonical_keys("2026/05/02")
        # Drop all 3 technology sub-stages.
        keys = [
            k for k in keys
            if not any(
                f"/{stage}:technology/" in k for stage in SECTOR_SUB_STAGES
            )
        ]
        s3 = _build_s3_stub(keys)
        result = _saturday_coverage_for(
            s3, bucket="b", capture_prefix="decision_artifacts",
            saturday=datetime(2026, 5, 2),
        )
        assert result["coverage_pct"] == 87.5
        assert result["per_agent"]["sector_team:technology"]["present"] is False
        assert result["per_agent"]["macro_economist"]["present"] is True

    def test_partial_sub_stages_keep_sector_absent(self):
        """If only 2 of 3 sub-stages emit for a sector, the virtual
        ``sector_team:{sector}`` is NOT present — partial captures
        signal an interrupted pipeline, not a successful decision."""
        from datetime import datetime
        keys = _canonical_keys("2026/05/02")
        # Drop only the peer_review sub-stage for technology, leaving
        # quant + qual artifacts in place.
        keys = [
            k for k in keys
            if "/sector_peer_review:technology/" not in k
        ]
        s3 = _build_s3_stub(keys)
        result = _saturday_coverage_for(
            s3, bucket="b", capture_prefix="decision_artifacts",
            saturday=datetime(2026, 5, 2),
        )
        # Coverage drops by exactly one sector — 7/8.
        assert result["coverage_pct"] == 87.5
        tech_entry = result["per_agent"]["sector_team:technology"]
        assert tech_entry["present"] is False
        assert tech_entry["sub_stages"] == {
            "sector_quant": 1, "sector_qual": 1, "sector_peer_review": 0,
        }
        # n_artifacts reflects what DID land — useful for the email body.
        assert tech_entry["n_artifacts"] == 2

    def test_sub_stage_ids_not_uncategorized(self):
        """Sub-stage IDs (sector_quant:*, sector_qual:*, sector_peer_review:*)
        roll up into the virtual sector_team:{sector} entry — they must
        not appear in the uncategorized list."""
        from datetime import datetime
        s3 = _build_s3_stub(_canonical_keys("2026/05/02"))
        result = _saturday_coverage_for(
            s3, bucket="b", capture_prefix="decision_artifacts",
            saturday=datetime(2026, 5, 2),
        )
        assert result["uncategorized_agents"] == []
        assert result["coverage_pct"] == 100.0

    def test_thesis_update_counted_separately(self):
        """thesis_update:* artifacts are reported as a count, never in
        the coverage denominator."""
        from datetime import datetime
        keys = _canonical_keys("2026/05/02") + [
            "decision_artifacts/2026/05/02/thesis_update:NVDA/run.json",
            "decision_artifacts/2026/05/02/thesis_update:AAPL/run.json",
            "decision_artifacts/2026/05/02/thesis_update:TSLA/run.json",
        ]
        s3 = _build_s3_stub(keys)
        result = _saturday_coverage_for(
            s3, bucket="b", capture_prefix="decision_artifacts",
            saturday=datetime(2026, 5, 2),
        )
        # Coverage is still based on the 8 canonical only.
        assert result["coverage_pct"] == 100.0
        assert result["thesis_update_count"] == 3
        # thesis_update:* doesn't show up as uncategorized.
        assert result["uncategorized_agents"] == []

    def test_uncategorized_agents_surfaced(self):
        """Unknown agent_ids should bubble up — they signal either a new
        agent rolling out (intentional) or a typo (mistake)."""
        from datetime import datetime
        keys = _canonical_keys("2026/05/02") + [
            "decision_artifacts/2026/05/02/new_experimental_agent/run.json",
            "decision_artifacts/2026/05/02/typo_macro/run.json",
        ]
        s3 = _build_s3_stub(keys)
        result = _saturday_coverage_for(
            s3, bucket="b", capture_prefix="decision_artifacts",
            saturday=datetime(2026, 5, 2),
        )
        assert "new_experimental_agent" in result["uncategorized_agents"]
        assert "typo_macro" in result["uncategorized_agents"]

    def test_empty_partition_is_zero_pct(self):
        from datetime import datetime
        s3 = _build_s3_stub([])
        result = _saturday_coverage_for(
            s3, bucket="b", capture_prefix="decision_artifacts",
            saturday=datetime(2026, 5, 2),
        )
        assert result["coverage_pct"] == 0.0
        assert result["n_canonical_present"] == 0


# ── compute_decision_capture_coverage (integration) ─────────────────────────


class TestComputeDecisionCaptureCoverage:
    def test_run_date_is_saturday_with_full_set(self):
        s3 = _build_s3_stub(_canonical_keys("2026/05/02"))
        result = compute_decision_capture_coverage(
            bucket="b", run_date="2026-05-02",
            lookback_weeks=1, s3_client=s3,
        )
        assert result["status"] == "ok"
        assert result["coverage_pct"] == 100.0
        assert result["most_recent_sf_date"] == "2026-05-02"

    def test_run_date_midweek_walks_back_to_saturday(self):
        """Tue 2026-05-06 should resolve to Sat 2026-05-02 — the
        most-recent Saturday with captures."""
        s3 = _build_s3_stub(_canonical_keys("2026/05/02"))
        result = compute_decision_capture_coverage(
            bucket="b", run_date="2026-05-06",
            lookback_weeks=1, s3_client=s3,
        )
        assert result["status"] == "ok"
        assert result["most_recent_sf_date"] == "2026-05-02"
        assert result["coverage_pct"] == 100.0

    def test_no_captures_in_window_returns_status_skip(self):
        s3 = _build_s3_stub([])
        result = compute_decision_capture_coverage(
            bucket="b", run_date="2026-05-06",
            lookback_weeks=1, s3_client=s3,
        )
        assert result["status"] == "no_recent_sf_run"
        assert "no Saturday with captures" in result["reason"]

    def test_invalid_run_date_returns_error(self):
        s3 = _build_s3_stub([])
        result = compute_decision_capture_coverage(
            bucket="b", run_date="not-a-date", s3_client=s3,
        )
        assert result["status"] == "error"

    def test_partial_canonical_drops_below_99(self):
        """7/8 → 87.5% < 99%, so this trips the inventory gate."""
        keys = _canonical_keys("2026/05/02")
        # Drop all 3 industrials sub-stages.
        keys = [
            k for k in keys
            if not any(
                f"/{stage}:industrials/" in k for stage in SECTOR_SUB_STAGES
            )
        ]
        s3 = _build_s3_stub(keys)
        result = compute_decision_capture_coverage(
            bucket="b", run_date="2026-05-02",
            lookback_weeks=1, s3_client=s3,
        )
        assert result["status"] == "ok"
        assert result["coverage_pct"] == 87.5
        assert result["per_agent"]["sector_team:industrials"]["present"] is False

    def test_rolling_window_averages_two_saturdays(self):
        """Two Saturdays of data: 5/2 (100%) + 4/25 (87.5%) → mean 93.75%."""
        keys_502 = _canonical_keys("2026/05/02")
        keys_425 = _canonical_keys("2026/04/25")
        # Drop all 3 financials sub-stages from 4/25.
        keys_425 = [
            k for k in keys_425
            if not any(
                f"/{stage}:financials/" in k for stage in SECTOR_SUB_STAGES
            )
        ]
        s3 = _build_s3_stub(keys_502 + keys_425)
        result = compute_decision_capture_coverage(
            bucket="b", run_date="2026-05-02",
            lookback_weeks=2, s3_client=s3,
        )
        assert result["status"] == "ok"
        rolling = result["rolling"]
        assert rolling["n_saturdays_with_data"] == 2
        assert rolling["coverage_pct_mean"] == 93.75
        assert rolling["coverage_pct_min"] == 87.5
        assert rolling["coverage_pct_max"] == 100.0

    def test_rolling_window_skips_empty_saturdays(self):
        """A Saturday where the SF didn't run should be EXCLUDED from
        the rolling mean — counting it as 0% coverage would conflate
        SF-not-run with SF-ran-but-failed-capture, both of which trigger
        different alarms."""
        # Only one Saturday has data; lookback_weeks=4 → 3 empty Saturdays
        keys = _canonical_keys("2026/05/02")
        s3 = _build_s3_stub(keys)
        result = compute_decision_capture_coverage(
            bucket="b", run_date="2026-05-02",
            lookback_weeks=4, s3_client=s3,
        )
        rolling = result["rolling"]
        assert rolling["n_saturdays_with_data"] == 1
        assert rolling["coverage_pct_mean"] == 100.0  # not pulled down by empty Saturdays

    def test_thesis_update_count_in_top_level_result(self):
        keys = _canonical_keys("2026/05/02") + [
            "decision_artifacts/2026/05/02/thesis_update:NVDA/run.json",
            "decision_artifacts/2026/05/02/thesis_update:KO/run.json",
        ]
        s3 = _build_s3_stub(keys)
        result = compute_decision_capture_coverage(
            bucket="b", run_date="2026-05-02",
            lookback_weeks=1, s3_client=s3,
        )
        assert result["thesis_update_count"] == 2

    def test_rolling_uses_saturday_with_only_thesis_update(self):
        """A Saturday where ONLY thesis_update fired (no canonical
        agents) should still show up in rolling. Coverage == 0%, but
        there's signal — partial-failure case where research crashed
        but thesis_update batch succeeded."""
        keys = (
            _canonical_keys("2026/05/02")
            + ["decision_artifacts/2026/04/25/thesis_update:KO/run.json"]
        )
        s3 = _build_s3_stub(keys)
        result = compute_decision_capture_coverage(
            bucket="b", run_date="2026-05-02",
            lookback_weeks=2, s3_client=s3,
        )
        rolling = result["rolling"]
        # Both Saturdays are in the window — 5/2 (100%) + 4/25 (0% canonical).
        assert rolling["n_saturdays_with_data"] == 2
        assert rolling["coverage_pct_min"] == 0.0
