"""Tests for the LLM look-ahead-bias disclosure (L4581 · #655, G7)."""

from __future__ import annotations

from analysis.lookahead_disclosure import (
    STATUS_CLEAN,
    STATUS_OVERLAP,
    STATUS_UNKNOWN,
    build_disclosure,
    render_section,
)

_CUTOFFS = {
    "claude-haiku-4-5-20251001": "2025-01-31",
    "claude-sonnet-4-6": "2025-03-31",
}


def test_window_before_cutoff_flags_overlap():
    d = build_disclosure(
        backtest_start="2024-01-01",
        backtest_end="2024-12-31",
        model_ids=["claude-haiku-4-5-20251001"],
        training_cutoffs=_CUTOFFS,
    )
    assert d.has_overlap
    assert not d.is_clean
    assert d.models[0].status == STATUS_OVERLAP


def test_window_after_cutoff_is_clean():
    d = build_disclosure(
        backtest_start="2025-06-01",
        backtest_end="2025-12-31",
        model_ids=["claude-haiku-4-5-20251001", "claude-sonnet-4-6"],
        training_cutoffs=_CUTOFFS,
    )
    assert d.is_clean
    assert not d.has_overlap and not d.has_unknown
    assert {m.status for m in d.models} == {STATUS_CLEAN}


def test_window_straddling_cutoff_flags_overlap():
    # start (2025-01-01) <= cutoff (2025-01-31) => overlap, even though end
    # is well past the cutoff.
    d = build_disclosure(
        backtest_start="2025-01-01",
        backtest_end="2025-12-31",
        model_ids=["claude-haiku-4-5-20251001"],
        training_cutoffs=_CUTOFFS,
    )
    assert d.models[0].status == STATUS_OVERLAP


def test_unknown_cutoff_is_flagged_not_assumed_clean():
    d = build_disclosure(
        backtest_start="2025-06-01",
        backtest_end="2025-12-31",
        model_ids=["some-unlisted-model"],
        training_cutoffs=_CUTOFFS,
    )
    assert d.has_unknown
    assert not d.is_clean
    assert d.models[0].status == STATUS_UNKNOWN


def test_missing_cutoffs_mapping_treats_all_unknown():
    d = build_disclosure(
        backtest_start="2025-06-01",
        backtest_end="2025-12-31",
        model_ids=["claude-sonnet-4-6"],
        training_cutoffs=None,
    )
    assert d.has_unknown


def test_unknown_start_is_conservative_overlap():
    d = build_disclosure(
        backtest_start=None,
        backtest_end="2025-12-31",
        model_ids=["claude-sonnet-4-6"],
        training_cutoffs=_CUTOFFS,
    )
    assert d.models[0].status == STATUS_OVERLAP


def test_models_deduped_preserving_order():
    d = build_disclosure(
        backtest_start="2025-06-01",
        backtest_end="2025-12-31",
        model_ids=["claude-sonnet-4-6", "claude-sonnet-4-6"],
        training_cutoffs=_CUTOFFS,
    )
    assert len(d.models) == 1


def test_render_overlap_section_leads_with_warning():
    d = build_disclosure(
        backtest_start="2024-01-01",
        backtest_end="2024-12-31",
        model_ids=["claude-haiku-4-5-20251001"],
        training_cutoffs=_CUTOFFS,
    )
    md = "\n".join(render_section(d))
    assert "Look-Ahead-Bias Disclosure (G7)" in md
    assert "LOOK-AHEAD OVERLAP" in md
    assert "claude-haiku-4-5-20251001" in md


def test_render_empty_models_flags_missing_disclosure():
    d = build_disclosure(
        backtest_start="2024-01-01",
        backtest_end="2024-12-31",
        model_ids=[],
        training_cutoffs=_CUTOFFS,
    )
    md = "\n".join(render_section(d))
    assert "MISSING" in md


def test_render_clean_section():
    d = build_disclosure(
        backtest_start="2025-06-01",
        backtest_end="2025-12-31",
        model_ids=["claude-sonnet-4-6"],
        training_cutoffs=_CUTOFFS,
    )
    md = "\n".join(render_section(d))
    assert "CLEAN" in md
