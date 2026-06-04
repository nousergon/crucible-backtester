"""Unit tests for reporter.py — pipeline health section and report structure."""
import pandas as pd
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
        """Empty health dict (db_pull_status unset / never attempted) must
        render the explicit MISSING message — never a bare value/None
        passthrough that reads as a rendering bug. Updated for the
        2026-05-16 not-loaded-always-MISSING contract."""
        lines = _section_pipeline_health({})
        text = "\n".join(lines)
        assert "## Pipeline Health" in text
        assert "Research DB: **MISSING** — signal quality analysis skipped" in text
        # Regression guard: the old behavior leaked the literal default
        # ("unknown") or a bare "None" into the rendered line.
        assert "Research DB: unknown" not in text
        assert "Research DB: None" not in text

    def test_unset_db_pull_status_renders_missing_not_none(self):
        """A not-loaded state from a path that never attempted the pull
        (db_pull_status is None — the exact 2026-05-16 SF-spot symptom)
        must hit the clear MISSING message, not `- Research DB: None`."""
        lines = _section_pipeline_health({"db_pull_status": None})
        text = "\n".join(lines)
        assert "Research DB: **MISSING** — signal quality analysis skipped" in text
        assert "Research DB: None" not in text

    def test_unexpected_db_pull_status_renders_missing(self):
        """Any unexpected db_pull_status value is a not-loaded state and
        must render MISSING, never pass the raw value through."""
        lines = _section_pipeline_health({"db_pull_status": "weird-value"})
        text = "\n".join(lines)
        assert "Research DB: **MISSING** — signal quality analysis skipped" in text
        assert "weird-value" not in text


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


class TestReporterCleanupBundle:
    """Closes the 4 reporter-side items from the 2026-05-09 P2 ROADMAP entry
    + the related L1907 / L1913 entries:

    - L1907: Mode 1 / Score threshold / Regime / Sub-score attribution
      sections suppressed entirely when signal_quality.status == "skipped"
      (the simulation email path; research_db isn't loaded there).
    - L1913: predictor-only backtest header renamed from
      \"(2y historical)\" → \"Layer-1A Momentum-Only Synthetic Backtest
      (10y component sanity check)\" + disclaimer block.
    - Item B: executor-recommendations table drops rows for params not in
      the sweep grid; footer names them explicitly.
    - Item C: predictor param sweep header is honest about sort axis;
      sortino_ratio + cvar_95 + calmar_ratio rendered as stat columns,
      not param columns.
    """

    def test_skipped_signal_quality_suppresses_4_sections(self):
        md = build_report(
            run_date="2026-05-09",
            signal_quality={"status": "skipped"},
            regime_analysis=[],
            score_analysis=[],
            attribution={"status": "skipped"},
        )
        # All 4 derived sections suppressed — no headers anywhere.
        assert "## Mode 1" not in md
        assert "## Score threshold analysis" not in md
        assert "## Regime analysis" not in md
        assert "## Sub-score attribution" not in md
        # Headline still present (other sections still render).
        assert "# Alpha Engine Backtest Report" in md

    def test_ok_signal_quality_keeps_sections(self):
        md = build_report(
            run_date="2026-05-09",
            signal_quality={
                "status": "ok",
                "overall": {"accuracy_10d": 0.62, "n_10d": 80},
            },
            regime_analysis=[],
            score_analysis=[],
            attribution={"status": "ok"},
        )
        assert "## Mode 1" in md
        # Score / Regime / Sub-score render (with their own deferred-fallback
        # bodies for empty rows lists — that's existing behavior).
        assert "## Score threshold analysis" in md
        assert "## Regime analysis" in md
        assert "## Sub-score attribution" in md

    def test_predictor_backtest_header_renamed_with_disclaimer(self):
        md = build_report(
            run_date="2026-05-09",
            signal_quality={"status": "skipped"},
            regime_analysis=[],
            score_analysis=[],
            attribution={"status": "skipped"},
            predictor_stats={
                "status": "ok",
                "total_alpha": -3.085,
                "total_return": 0.053,
                "spy_return": 3.137,
                "sharpe_ratio": 0.191,
                "predictor_metadata": {
                    "n_tickers": 904,
                    "n_dates": 2500,
                    "date_range_start": "2016-05-31",
                    "date_range_end": "2026-05-08",
                    "top_n_per_day": 20,
                    "min_score": 70,
                },
            },
        )
        # Header renamed.
        assert "Layer-1A Momentum-Only Synthetic Backtest" in md
        assert "10y component sanity check" in md
        assert "(2y historical)" not in md
        # Disclaimer block present.
        assert "Component-level sanity check" in md
        assert "not the production v3 ensemble" in md

    def test_executor_recommendations_drop_unswept_rows(self):
        from reporter import _section_executor_recommendations
        # Sweep covered 5 of 16 params; baseline + recommended only have
        # those 5 keys; factory has all 16.
        result = {
            "status": "ok",
            "n_combos_tested": 60,
            "improvement_pct": 0.063,
            "baseline_sharpe": 0.6439,
            "best_sharpe": 0.6842,
            "best_alpha": -2.5881,
            "baseline_combo_rank": 2,
            "factory_defaults": {
                "atr_multiplier": 2.5, "min_score": 70, "max_position_pct": 0.05,
                "time_decay_reduce_days": 7, "time_decay_exit_days": 14,
                # The "—" rows from today's email — present in factory, NOT in sweep:
                "atr_sizing_target_risk": 0.02,
                "confidence_sizing_min": 0.70,
                "confidence_sizing_range": 0.60,
                "correlation_block_threshold": 0.80,
                "earnings_proximity_days": 5,
                "earnings_sizing_reduction": 0.50,
                "momentum_exit_threshold": -15.0,
                "momentum_gate_threshold": -5.0,
                "profit_take_pct": 0.25,
                "reduce_fraction": 0.50,
                "staleness_decay_per_day": 0.03,
            },
            "baseline_params": {
                "atr_multiplier": 2.0, "min_score": 75, "max_position_pct": 0.10,
                "time_decay_reduce_days": 7, "time_decay_exit_days": 10,
            },
            "recommended_params": {
                "atr_multiplier": 3.0, "min_score": 75, "max_position_pct": 0.10,
                "time_decay_reduce_days": 7, "time_decay_exit_days": 15,
            },
            "apply_result": {"applied": True},
        }
        md = "\n".join(_section_executor_recommendations(result))

        # Swept params render as rows.
        assert "atr_multiplier" in md
        assert "time_decay_exit_days" in md
        # Unswept params are NOT rendered as rows (no `—`/`—`/`—` clutter).
        # They WILL appear in the footer's "Not in sweep grid" list — assert
        # the row count is restricted, not full key presence.
        row_lines = [
            line for line in md.split("\n")
            if line.startswith("| atr_") or line.startswith("| confidence_")
            or line.startswith("| earnings_") or line.startswith("| momentum_")
            or line.startswith("| profit_") or line.startswith("| reduce_")
            or line.startswith("| staleness_") or line.startswith("| correlation_")
        ]
        # Only atr_multiplier swept among the atr_*; the others appear in footer not rows.
        swept_row_lines = [r for r in row_lines if "| atr_multiplier |" in r]
        assert len(swept_row_lines) == 1
        unswept_row_lines = [
            r for r in row_lines
            if "| atr_sizing_target_risk |" in r
            or "| confidence_sizing_min |" in r
            or "| earnings_proximity_days |" in r
        ]
        assert len(unswept_row_lines) == 0

        # Header reflects coverage: "5 of 16".
        assert "5 of 16" in md
        # Footer names the unswept set.
        assert "Not in sweep grid" in md
        assert "atr_sizing_target_risk" in md
        assert "confidence_sizing_min" in md

    def test_executor_recommendations_caption_skill_mode_leads_with_sortino_and_psr(self):
        """When fit_target=skill_composite, the caption surfaces Sortino +
        PSR as the gating axes; alpha vs SPY is labeled presentation-only.
        Mirrors the post-2026-05-09 framing that alpha is not the
        optimizer's fit target."""
        from reporter import _section_executor_recommendations
        result = {
            "status": "ok",
            "fit_target": "skill_composite",
            "n_combos_tested": 60,
            "improvement_pct": 0.47,  # Sortino improvement
            "baseline_sortino": 0.65,
            "best_sortino": 0.95,
            "best_psr": 0.97,
            "best_alpha": -2.59,  # presentation only
            "best_sharpe": 0.55,
            "baseline_sharpe": 0.50,
            "baseline_combo_rank": 2,
            "factory_defaults": {"atr_multiplier": 2.5, "min_score": 70},
            "baseline_params": {"atr_multiplier": 2.0, "min_score": 75},
            "recommended_params": {"atr_multiplier": 3.0, "min_score": 75},
            "apply_result": {"applied": True},
        }
        md = "\n".join(_section_executor_recommendations(result))
        # Skill-mode caption surfaces Sortino + PSR explicitly.
        assert "Sortino improvement: 47" in md
        assert "0.6500 → 0.9500" in md
        assert "PSR (P(true SR>0)): 0.970" in md
        # Alpha vs SPY clearly labeled as presentation-only.
        assert "Alpha vs SPY:" in md
        assert "presentation only" in md
        # fit_target stamped in caption for operator visibility.
        assert "skill_composite" in md
        # Legacy "Sharpe improvement" phrasing must NOT appear under skill mode.
        assert "Sharpe improvement:" not in md

    def test_executor_recommendations_caption_legacy_mode_unchanged(self):
        """Legacy fit_target preserves the pre-cutover caption shape exactly
        (Sharpe improvement + Best alpha as the headline numbers)."""
        from reporter import _section_executor_recommendations
        result = {
            "status": "ok",
            "fit_target": "sharpe_legacy",
            "n_combos_tested": 60,
            "improvement_pct": 0.063,  # Sharpe improvement
            "baseline_sharpe": 0.6439,
            "best_sharpe": 0.6842,
            "best_alpha": -2.5881,
            "baseline_combo_rank": 2,
            "factory_defaults": {"atr_multiplier": 2.5},
            "baseline_params": {"atr_multiplier": 2.0},
            "recommended_params": {"atr_multiplier": 3.0},
            "apply_result": {"applied": True},
        }
        md = "\n".join(_section_executor_recommendations(result))
        # Legacy caption — Sharpe improvement leads, alpha is unlabeled
        # (no "presentation only" tag). Identical to pre-cutover output.
        assert "Sharpe improvement: 6.3%" in md
        assert "0.6439 → 0.6842" in md
        assert "Best alpha:" in md
        # Skill-mode-specific text must NOT appear under legacy.
        assert "PSR" not in md
        assert "presentation only" not in md
        assert "skill_composite" not in md

    def test_predictor_param_sweep_renders_sortino_cvar_as_stats_not_params(self):
        from reporter import _section_param_sweep_predictor
        df = pd.DataFrame([
            {
                "min_score": 75, "max_position_pct": 0.10,
                "atr_multiplier": 3.0, "time_decay_exit_days": 15,
                "total_alpha": -2.58,    # presentation column
                "sortino_ratio": 0.97,   # skill-aligned (PR #141 evaluator-revamp)
                "cvar_95": -0.0105,
                "sharpe_ratio": 0.68,
            },
            {
                "min_score": 75, "max_position_pct": 0.10,
                "atr_multiplier": 2.0, "time_decay_exit_days": 10,
                "total_alpha": -2.59,
                "sortino_ratio": 0.92,
                "cvar_95": -0.0104,
                "sharpe_ratio": 0.69,
            },
        ])
        md = "\n".join(_section_param_sweep_predictor(df))
        # Header names the real sort axis: Sortino primary, total_alpha
        # tiebreaker (param_sweep.py::_sort_sweep_df_skilled_risk). Updated
        # for the 2026-05-16 caption-vs-sort reconciliation.
        assert "sorted by Sortino, total_alpha tiebreaker" in md
        assert "sorted by total_alpha)" not in md  # old misleading caption gone
        # Header column row contains stat columns in the preferred order.
        # sortino_ratio + cvar_95 must NOT be in param_cols position.
        header_row = next(
            line for line in md.split("\n") if line.startswith("| min_score")
        )
        # Param cols come first; stat cols after. total_alpha position is
        # past time_decay_exit_days but before sortino_ratio.
        params_end = header_row.index("time_decay_exit_days")
        alpha_pos = header_row.index("total_alpha")
        sortino_pos = header_row.index("sortino_ratio")
        cvar_pos = header_row.index("cvar_95")
        # Stats appear after the last param column, in the documented order.
        assert params_end < alpha_pos < sortino_pos < cvar_pos


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


class TestSortinoHeadlineMetric:
    """Per the Sharpe→Sortino skilled-risk evaluator revamp, Sortino is
    the primary/headline risk-adjusted metric in the Mode 2 and Layer-1A
    tables. Sharpe is kept as a secondary line (not deleted)."""

    def _stats(self, **over):
        s = {
            "status": "ok",
            "total_return": 0.42,
            "total_alpha": 0.11,
            "sortino_ratio": 1.930,
            "sharpe_ratio": 1.210,
            "max_drawdown": -0.18,
            "calmar_ratio": 0.9,
            "total_trades": 120,
            "win_rate": 0.55,
        }
        s.update(over)
        return s

    def test_mode2_table_headlines_sortino_keeps_sharpe_secondary(self):
        from reporter import _section_portfolio
        md = "\n".join(_section_portfolio(self._stats()))
        # Sortino is bolded as the headline risk-adjusted metric.
        assert "| **Sortino ratio** | **1.930** |" in md
        # Sharpe is retained as a secondary (non-bold) line — NOT deleted.
        assert "| Sharpe ratio | 1.210 |" in md
        # Sortino row appears before the Sharpe row.
        assert md.index("Sortino ratio") < md.index("Sharpe ratio")

    def test_layer1a_table_headlines_sortino_keeps_sharpe_secondary(self):
        from reporter import _section_predictor_backtest
        stats = self._stats(
            alpha_vs_ew_high_vol=0.03,
            spy_return=0.31,
            ew_high_vol_return=0.34,
            dates_simulated=2400,
            total_orders=900,
            predictor_metadata={
                "n_tickers": 60, "n_dates": 2400,
                "date_range_start": "2016-01-04",
                "date_range_end": "2026-05-15",
                "top_n_per_day": 5, "min_score": 70,
            },
        )
        md = "\n".join(_section_predictor_backtest(stats))
        assert "| **Sortino ratio** | **1.930** |" in md
        assert "| Sharpe ratio | 1.210 |" in md
        assert md.index("Sortino ratio") < md.index("Sharpe ratio")


# ── always-emit contract for freshness-monitored artifacts ──────────────────


class TestAlwaysEmitDecisionCapture:
    """``save`` must write decision_capture_coverage.json (and the executor
    sibling) on EVERY non-None producer return — including no-data / error
    statuses — so that absence of the S3 object means "diagnostic never ran",
    never "ran but found no upstream captures". Regression guard for the
    substrate-health agent_decisions false-absence bug (2026-05-29)."""

    def _save(self, tmp_path, **kwargs):
        from reporter import save
        return save(
            report_md="# r",
            signal_quality={"status": "ok", "overall": {}},
            score_analysis=[],
            run_date="2026-05-29",
            results_dir=str(tmp_path),
            **kwargs,
        )

    def test_no_recent_sf_run_is_still_written(self, tmp_path):
        out = self._save(
            tmp_path,
            decision_capture_coverage={
                "status": "no_recent_sf_run", "coverage_pct": 0.0,
                "reason": "no Saturday with captures",
            },
        )
        import json
        f = out / "decision_capture_coverage.json"
        assert f.exists(), "no_recent_sf_run must still emit the artifact"
        assert json.loads(f.read_text())["status"] == "no_recent_sf_run"

    def test_error_status_is_still_written(self, tmp_path):
        out = self._save(
            tmp_path,
            executor_decision_capture_coverage={
                "status": "insufficient_data", "coverage_pct": 0.0,
            },
        )
        assert (out / "executor_decision_capture_coverage.json").exists()

    def test_ok_status_is_written(self, tmp_path):
        out = self._save(
            tmp_path,
            decision_capture_coverage={"status": "ok", "coverage_pct": 100.0},
        )
        assert (out / "decision_capture_coverage.json").exists()

    def test_none_is_not_written(self, tmp_path):
        """None means the module was never invoked → absence is correct."""
        out = self._save(tmp_path, decision_capture_coverage=None)
        assert not (out / "decision_capture_coverage.json").exists()

    def test_ok_only_artifact_skips_error_status(self, tmp_path):
        """A non-always-emit artifact (barrier_coherence) keeps ok-only."""
        out = self._save(
            tmp_path,
            barrier_coherence={"status": "error", "error": "boom"},
        )
        assert not (out / "barrier_coherence.json").exists()


# ── save() — Phase B1a artifact persistence ──────────────────────────────────


class TestSavePersistence:
    """The 6 computed-but-previously-unpersisted diagnostic dicts now land in
    results/{date}/ for the evaluator (Report Card v2, Option B) to read over S3."""

    def _save(self, tmp_path, **kw):
        from reporter import save
        return save(
            report_md="# r",
            signal_quality={},
            score_analysis=[],
            run_date="2026-06-04",
            results_dir=str(tmp_path),
            **kw,
        )

    def test_ok_artifacts_persisted(self, tmp_path):
        import json
        out = self._save(
            tmp_path,
            score_calibration={"status": "ok", "ece": 0.04},
            macro_eval={"status": "ok", "accuracy": 0.6},
            team_metrics={"tech": {"grade": 80}},
            calibration_diagnostics={"status": "ok", "x": 1},
            excursion_summary={"status": "ok", "mfe_mae": 1.2},
        )
        for fn in (
            "score_calibration.json",
            "macro_eval.json",
            "team_metrics.json",
            "portfolio_calibration.json",
            "portfolio_excursion.json",
        ):
            assert (out / fn).exists(), f"{fn} not written"
        assert json.loads((out / "score_calibration.json").read_text())["ece"] == 0.04
        assert json.loads((out / "team_metrics.json").read_text())["tech"]["grade"] == 80

    def test_na_status_not_persisted(self, tmp_path):
        out = self._save(
            tmp_path,
            score_calibration={"status": "insufficient_data"},
            calibration_diagnostics={"status": "insufficient_data"},
        )
        assert not (out / "score_calibration.json").exists()
        assert not (out / "portfolio_calibration.json").exists()

    def test_empty_team_metrics_not_persisted(self, tmp_path):
        # evaluate.py passes `team_metrics or None`, so empty {} arrives as None.
        out = self._save(tmp_path, team_metrics=None)
        assert not (out / "team_metrics.json").exists()
