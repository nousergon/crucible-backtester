"""Unit tests for reporter.py — pipeline health section and report structure."""
import pytest

from reporter import _section_pipeline_health, build_report


# ── _section_pipeline_health ─────────────────────────────────────────────────


class TestSectionPipelineHealth:

    def test_healthy_pipeline(self):
        """All systems OK should produce a clean section."""
        health = {
            "db_pull_status": "ok",
            "coverage": 1.0,
            "dates_simulated": 12,
            "dates_expected": 12,
        }
        lines = _section_pipeline_health(health)
        text = "\n".join(lines)
        assert "## Pipeline Health" in text
        assert "Research DB: Loaded" in text
        assert "12/12 dates (100%)" in text

    def test_missing_db(self):
        """Failed DB pull should show MISSING warning."""
        health = {"db_pull_status": "failed"}
        lines = _section_pipeline_health(health)
        text = "\n".join(lines)
        assert "MISSING" in text
        assert "signal quality analysis skipped" in text

    def test_staleness_warning_shown(self):
        """Staleness warning should appear as blockquote."""
        health = {
            "db_pull_status": "ok",
            "staleness_warning": "STALE price data: last date 2026-03-20",
        }
        lines = _section_pipeline_health(health)
        text = "\n".join(lines)
        assert "> STALE price data" in text

    def test_coverage_with_skip_reasons(self):
        """Low coverage with skip reasons should show breakdown."""
        health = {
            "db_pull_status": "ok",
            "coverage": 0.5,
            "dates_simulated": 6,
            "dates_expected": 12,
            "skip_reasons": {"no_price_index": 4, "no_signals": 2},
        }
        lines = _section_pipeline_health(health)
        text = "\n".join(lines)
        assert "6/12 dates (50%)" in text
        assert "no_price_index" in text

    def test_price_gaps_shown(self):
        """Price gap and unfilled gap counts should appear."""
        health = {
            "db_pull_status": "ok",
            "price_gap_warnings": {"AAPL": 8, "TSLA": 12},
            "unfilled_gaps": {"TSLA": 7},
        }
        lines = _section_pipeline_health(health)
        text = "\n".join(lines)
        assert "Price gaps (>5 days): 2 tickers" in text
        assert "Unfilled gaps after ffill: 1 tickers" in text

    def test_predictor_feature_skips(self):
        """Predictor feature skip reasons should appear."""
        health = {
            "db_pull_status": "ok",
            "feature_skip_reasons": {"too_short": 50, "computation_error": 3},
        }
        lines = _section_pipeline_health(health)
        text = "\n".join(lines)
        assert "too_short" in text

    def test_empty_health_dict(self):
        """Empty health dict should still produce a valid section."""
        lines = _section_pipeline_health({})
        text = "\n".join(lines)
        assert "## Pipeline Health" in text
        assert "unknown" in text  # db_pull_status defaults to "unknown"


# ── build_report() ───────────────────────────────────────────────────────────


class TestBuildReport:

    def test_returns_string_with_header(self):
        """Report should be a string starting with the title."""
        md = build_report(
            run_date="2026-03-29",
            signal_quality={"status": "ok", "overall": {}},
            regime_analysis=[],
            score_analysis=[],
            attribution={"status": "skipped"},
        )
        assert isinstance(md, str)
        assert "# Alpha Engine Backtest Report" in md
        assert "2026-03-29" in md

    def test_pipeline_health_included_when_provided(self):
        """Pipeline health section should appear when health dict is passed."""
        md = build_report(
            run_date="2026-03-29",
            signal_quality={"status": "ok", "overall": {}},
            regime_analysis=[],
            score_analysis=[],
            attribution={"status": "skipped"},
            pipeline_health={"db_pull_status": "ok", "coverage": 0.9,
                             "dates_simulated": 9, "dates_expected": 10},
        )
        assert "## Pipeline Health" in md
        assert "9/10 dates" in md

    def test_pipeline_health_absent_when_none(self):
        """No pipeline health section when not provided."""
        md = build_report(
            run_date="2026-03-29",
            signal_quality={"status": "skipped"},
            regime_analysis=[],
            score_analysis=[],
            attribution={"status": "skipped"},
        )
        assert "## Pipeline Health" not in md

    def test_skipped_mode_produces_valid_report(self):
        """All-skipped inputs should still produce a valid markdown report."""
        md = build_report(
            run_date="2026-03-29",
            signal_quality={"status": "skipped"},
            regime_analysis=[],
            score_analysis=[],
            attribution={"status": "skipped"},
        )
        assert isinstance(md, str)
        assert len(md) > 50


class TestOptimizerStatusFiltering:
    """The simulation email (`backtest.py`) does not run the weight or
    veto optimizers — those run in `evaluate.py`'s evaluator email. The
    Optimizer Status section should filter out None results so we don't
    misleadingly render "NOT RUN — mode not included in this run" for
    optimizers that belong to a different code path entirely."""

    def test_simulation_email_only_renders_executor_row(self):
        """Backtester simulation email passes only `executor_rec`. The
        Optimizer Status table should show ONE row (Executor params),
        not three with two stub "NOT RUN" entries.
        """
        md = build_report(
            run_date="2026-05-09",
            signal_quality={"status": "skipped"},
            regime_analysis=[],
            score_analysis=[],
            attribution={"status": "skipped"},
            executor_rec={
                "status": "ok",
                "fit_target": "sharpe_legacy",
                "apply_result": {"applied": True},
            },
            # weight_result + veto_result deliberately omitted (None default)
        )
        assert "### Optimizer Status" in md
        assert "Executor params" in md
        assert "PROMOTED" in md
        # Phantom rows for optimizers that weren't computed on this path
        # must not render.
        assert "Scoring weights" not in md
        assert "Veto threshold" not in md
        assert "NOT RUN" not in md
        assert "mode not included in this run" not in md

    def test_evaluator_email_renders_all_three_rows(self):
        """Evaluator email passes all three results. All three rows render."""
        md = build_report(
            run_date="2026-05-09",
            signal_quality={"status": "skipped"},
            regime_analysis=[],
            score_analysis=[],
            attribution={"status": "skipped"},
            weight_result={"status": "ok", "apply_result": {"applied": True}},
            veto_result={"status": "ok", "apply_result": {"applied": False, "reason": "no improvement"}},
            executor_rec={"status": "ok", "apply_result": {"applied": True}},
        )
        assert "### Optimizer Status" in md
        assert "Scoring weights" in md
        assert "Executor params" in md
        assert "Veto threshold" in md

    def test_skipped_optimizer_renders_as_skipped_not_filtered(self):
        """When an optimizer ran on this path but skipped due to missing
        inputs (e.g. research_db unavailable → tracker.run_module returns
        a `{"status": "skipped"}` dict), the row SHOULD render as SKIPPED
        — that's a different code-path event from the simulation-email
        case where the optimizer wasn't on this path at all (None).
        """
        md = build_report(
            run_date="2026-05-09",
            signal_quality={"status": "skipped"},
            regime_analysis=[],
            score_analysis=[],
            attribution={"status": "skipped"},
            weight_result={"status": "skipped", "note": "research_db unavailable"},
            executor_rec={"status": "ok", "apply_result": {"applied": True}},
        )
        # The skipped optimizer renders as SKIPPED with reason — it ran on
        # this path, just produced no output.
        assert "Scoring weights" in md
        assert "SKIPPED" in md
        assert "research_db unavailable" in md

    def test_optimizer_status_section_omitted_when_no_results(self):
        """When NO optimizer results are passed (e.g. failure path before
        any optimizer ran), the Optimizer Status section header should not
        render at all — no empty table."""
        md = build_report(
            run_date="2026-05-09",
            signal_quality={"status": "skipped"},
            regime_analysis=[],
            score_analysis=[],
            attribution={"status": "skipped"},
            # No weight, veto, OR executor passed.
        )
        assert "### Optimizer Status" not in md


class TestSectionSkillVsBeta:
    """The Skill vs. Beta panel (PR 4) — renders only when the
    evaluator-revamp metrics are wired through."""

    def test_renders_when_team_metrics_present(self):
        from reporter import _section_skill_vs_beta
        grading = {
            "research": {
                "components": {
                    "sector_teams": [
                        {
                            "team_id": "tech",
                            "grade": 75.0, "letter": "B+",
                            "detail": {
                                "ic": 0.06,
                                "hit_rate": "58.0%",
                                "win_loss_ratio": 1.6,
                                "mfe_mae_ratio": 1.5,
                                "alpha_vs_ew_high_vol": "+1.20%",
                                "alpha_vs_beta_spy": "+0.80%",
                                "n_picks": 12,
                            },
                        },
                    ],
                    "calibration_diagnostics": {
                        "grade": 90.0, "letter": "A",
                        "detail": {"ece": 0.04, "n": 200, "quality": "good"},
                    },
                },
            },
        }
        lines = _section_skill_vs_beta(grading)
        assert any("Skill vs. Beta" in l for l in lines)
        assert any("Per-Team Skill Composite" in l for l in lines)
        assert any("Calibration" in l for l in lines)
        assert any("tech" in l.lower() for l in lines)
        assert any("0.04" in l for l in lines)  # ECE

    def test_returns_empty_when_legacy_team_path(self):
        from reporter import _section_skill_vs_beta
        grading = {
            "research": {
                "components": {
                    "sector_teams": [
                        {
                            "team_id": "tech",
                            "grade": 75.0, "letter": "B+",
                            "detail": {
                                "lift_vs_sector": "+2.50%",
                                "lift_vs_quant": "+1.10%",
                                "n_picks": 12,
                            },
                        },
                    ],
                },
            },
        }
        # No team has IC + alpha_vs_*; calibration not provided.
        # Should return [] so no header appears.
        lines = _section_skill_vs_beta(grading)
        assert lines == []

    def test_renders_calibration_only_when_skill_teams_absent(self):
        from reporter import _section_skill_vs_beta
        grading = {
            "research": {
                "components": {
                    "sector_teams": [],
                    "calibration_diagnostics": {
                        "grade": 65.0, "letter": "B",
                        "detail": {"ece": 0.08, "n": 150, "quality": "acceptable"},
                    },
                },
            },
        }
        lines = _section_skill_vs_beta(grading)
        assert any("Calibration" in l for l in lines)
        assert not any("Per-Team Skill Composite" in l for l in lines)

    def test_returns_empty_grading_with_no_research(self):
        from reporter import _section_skill_vs_beta
        assert _section_skill_vs_beta({}) == []
