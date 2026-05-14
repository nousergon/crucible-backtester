"""Tests for the factor_blend_sensitivity wire-in to reporter + evaluator.

PR 6 follow-up of the scanner-placement arc (alpha-engine-docs/private/
scanner-260514.md). Validates:
  - reporter._section_factor_blend_sensitivity renders both empty + populated
  - reporter.build_report accepts the new kwarg + threads through
  - DEFAULT_REGIME_WEIGHTS mirrors the canonical scoring.yaml values
"""

import numpy as np
import pandas as pd
import pytest

from analysis.factor_blend_sensitivity import (
    DEFAULT_REGIME_WEIGHTS,
    MIN_TRUSTWORTHY_SAMPLES,
    build_sensitivity_report,
)
from reporter import _section_factor_blend_sensitivity, build_report as build_md


# ── DEFAULT_REGIME_WEIGHTS mirrors scoring.yaml ─────────────────────────────


class TestDefaultRegimeWeights:
    def test_three_regimes_present(self):
        assert set(DEFAULT_REGIME_WEIGHTS.keys()) == {"bull", "bear", "neutral"}

    def test_bull_favors_momentum_with_negative_low_vol(self):
        """BULL signed weights should rank momentum first and put low_vol
        last (negative). Pin so config drift triggers a test failure."""
        bull = DEFAULT_REGIME_WEIGHTS["bull"]
        assert bull["momentum_score"] == 0.40
        assert bull["quality_score"] == 0.30
        assert bull["value_score"] == 0.20
        assert bull["low_vol_score"] == -0.10

    def test_bear_favors_low_vol_with_negative_momentum(self):
        """BEAR inverts — low_vol top, momentum negative."""
        bear = DEFAULT_REGIME_WEIGHTS["bear"]
        assert bear["low_vol_score"] == 0.40
        assert bear["quality_score"] == 0.30
        assert bear["momentum_score"] == -0.20
        assert bear["value_score"] == 0.10

    def test_neutral_balanced(self):
        """NEUTRAL all four equal."""
        neutral = DEFAULT_REGIME_WEIGHTS["neutral"]
        assert neutral["momentum_score"] == 0.25
        assert neutral["quality_score"] == 0.25
        assert neutral["value_score"] == 0.25
        assert neutral["low_vol_score"] == 0.25


# ── _section_factor_blend_sensitivity ────────────────────────────────────────


class TestSectionRendering:
    def test_none_report_renders_deferred(self):
        lines = _section_factor_blend_sensitivity(None)
        assert "Factor blend sensitivity" in lines[0]
        assert any("Deferred" in line for line in lines)

    def test_empty_report_renders_deferred(self):
        empty = build_sensitivity_report(pd.DataFrame(), DEFAULT_REGIME_WEIGHTS)
        lines = _section_factor_blend_sensitivity(empty)
        assert any("Deferred" in line for line in lines)

    def test_populated_report_renders_tables(self):
        """When the analyzer produced outcomes, both the mismatch table and
        the per-cell outcomes table appear in the markdown."""
        rng = np.random.default_rng(seed=10)
        rows = []
        for r in rng.normal(0.04, 0.02, MIN_TRUSTWORTHY_SAMPLES + 5):
            rows.append({
                "market_regime": "bull", "stance": "momentum",
                "return_10d": r, "spy_10d_return": 0.01,
                "beat_spy_10d": int(r > 0.01),
                "return_30d": r * 2, "spy_30d_return": 0.02,
                "beat_spy_30d": int(r * 2 > 0.02),
            })
        for r in rng.normal(0.02, 0.02, MIN_TRUSTWORTHY_SAMPLES + 5):
            rows.append({
                "market_regime": "bull", "stance": "quality",
                "return_10d": r, "spy_10d_return": 0.01,
                "beat_spy_10d": int(r > 0.01),
                "return_30d": r * 2, "spy_30d_return": 0.02,
                "beat_spy_30d": int(r * 2 > 0.02),
            })
        report = build_sensitivity_report(pd.DataFrame(rows), DEFAULT_REGIME_WEIGHTS)
        lines = _section_factor_blend_sensitivity(report)
        joined = "\n".join(lines)
        # Headline mismatch table heading
        assert "Config vs realized stance ordering" in joined
        # Per-cell outcomes table heading
        assert "Realized per-stance outcomes" in joined
        # Both regimes-and-stances render — at least one row per
        assert "momentum" in joined
        assert "quality" in joined

    def test_mismatch_flag_rendered(self):
        """Mismatch flag bold-renders as **YES** when realized != config."""
        rng = np.random.default_rng(seed=11)
        rows = []
        # momentum (config #1) loses on alpha
        for r in rng.normal(-0.01, 0.03, MIN_TRUSTWORTHY_SAMPLES + 5):
            rows.append({
                "market_regime": "bull", "stance": "momentum",
                "return_10d": r, "spy_10d_return": 0.01,
                "beat_spy_10d": int(r > 0.01),
                "return_30d": r * 2, "spy_30d_return": 0.02,
                "beat_spy_30d": int(r * 2 > 0.02),
            })
        # quality wins
        for r in rng.normal(0.05, 0.025, MIN_TRUSTWORTHY_SAMPLES + 5):
            rows.append({
                "market_regime": "bull", "stance": "quality",
                "return_10d": r, "spy_10d_return": 0.01,
                "beat_spy_10d": int(r > 0.01),
                "return_30d": r * 2, "spy_30d_return": 0.02,
                "beat_spy_30d": int(r * 2 > 0.02),
            })
        report = build_sensitivity_report(pd.DataFrame(rows), DEFAULT_REGIME_WEIGHTS)
        lines = _section_factor_blend_sensitivity(report)
        joined = "\n".join(lines)
        assert "**YES**" in joined  # mismatch flagged


# ── build_report integration ────────────────────────────────────────────────


class TestBuildReportIntegration:
    def test_kwarg_accepted_and_section_appears(self):
        """build_report(factor_blend_sensitivity=...) renders the section
        when populated."""
        empty_report = build_sensitivity_report(pd.DataFrame(), DEFAULT_REGIME_WEIGHTS)
        md = build_md(
            run_date="2026-05-17",
            signal_quality={"status": "skipped"},
            regime_analysis=[],
            score_analysis=[],
            attribution={"status": "skipped"},
            factor_blend_sensitivity=empty_report,
        )
        assert "Factor blend sensitivity" in md

    def test_kwarg_default_none_no_section(self):
        """When factor_blend_sensitivity is omitted (None), the section
        does NOT render — keeps the report unchanged when the analyzer
        hasn't run."""
        md = build_md(
            run_date="2026-05-17",
            signal_quality={"status": "skipped"},
            regime_analysis=[],
            score_analysis=[],
            attribution={"status": "skipped"},
        )
        assert "Factor blend sensitivity" not in md
