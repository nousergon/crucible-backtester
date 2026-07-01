"""Tests for analysis/grading.py — unified component scorecard."""

import pytest

from analysis.grading import (
    _band_to_grade,
    _clamp,
    _cvar_to_grade,
    _grade_action_entropy,
    _grade_calibration,
    _grade_calibration_diagnostics,
    _grade_excursion,
    _ic_to_grade,
    _letter,
    _lift_to_grade,
    _pct_to_grade,
    _ratio_to_grade,
    _weighted_avg,
    compute_scorecard,
)


class TestGradeCalibration:
    """Robust score→alpha calibration grade (Spearman-first, legacy fallback)."""

    def test_positive_rho_high_grade(self):
        g, d = _grade_calibration({"spearman_rho": 0.4, "spearman_p": 0.001,
                                   "calibration_assessment": "positive"})
        assert g == pytest.approx(70.0)  # 50 + 50*0.4
        assert d["calibration"] == "positive"

    def test_negative_rho_low_grade(self):
        g, _ = _grade_calibration({"spearman_rho": -0.3, "spearman_p": 0.01,
                                   "calibration_assessment": "negative"})
        assert g == pytest.approx(35.0)  # 50 + 50*-0.3

    def test_flat_is_neutral_not_punished(self):
        # The fix: an insignificant calibration is neutral (60), not ~RED.
        g, _ = _grade_calibration({"spearman_rho": -0.05, "spearman_p": 0.7,
                                   "calibration_assessment": "flat"})
        assert g == pytest.approx(60.0)

    def test_legacy_monotonic_fallback(self):
        assert _grade_calibration({"monotonic": True})[0] == pytest.approx(90.0)
        assert _grade_calibration({"monotonic": False})[0] == pytest.approx(40.0)

    def test_none_when_no_signal(self):
        assert _grade_calibration(None) == (None, {})
        assert _grade_calibration({})[0] is None


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestLetter:
    def test_a(self):
        assert _letter(92) == "A"

    def test_b_plus(self):
        assert _letter(75) == "B+"

    def test_f(self):
        assert _letter(10) == "F"

    def test_none(self):
        assert _letter(None) == "N/A"

    def test_clamped_above_100(self):
        assert _letter(105) == "A"

    def test_zero(self):
        assert _letter(0) == "F"


class TestPctToGrade:
    def test_baseline_maps_to_30(self):
        g = _pct_to_grade(0.50, baseline=0.50, ceiling=0.80)
        assert g == pytest.approx(30.0)

    def test_ceiling_maps_to_95(self):
        g = _pct_to_grade(0.80, baseline=0.50, ceiling=0.80)
        assert g == pytest.approx(95.0)

    def test_none_returns_none(self):
        assert _pct_to_grade(None) is None

    def test_below_baseline_clamps(self):
        g = _pct_to_grade(0.20, baseline=0.50, ceiling=0.80)
        assert g >= 0.0


class TestLiftToGrade:
    def test_zero_lift_maps_to_40(self):
        g = _lift_to_grade(0.0)
        assert g == pytest.approx(40.0)

    def test_positive_lift(self):
        g = _lift_to_grade(1.5, floor=-2.0, ceiling=3.0)
        assert 40.0 < g < 100.0

    def test_negative_lift(self):
        g = _lift_to_grade(-1.0, floor=-2.0, ceiling=3.0)
        assert 0.0 < g < 40.0

    def test_none_returns_none(self):
        assert _lift_to_grade(None) is None


class TestIcToGrade:
    def test_zero_ic(self):
        g = _ic_to_grade(0.0)
        assert g == pytest.approx(20.0)

    def test_good_ic(self):
        g = _ic_to_grade(0.05)
        assert g == pytest.approx(55.0)

    def test_great_ic(self):
        g = _ic_to_grade(0.10)
        assert g == pytest.approx(90.0)

    def test_none(self):
        assert _ic_to_grade(None) is None


class TestWeightedAvg:
    def test_simple(self):
        result = _weighted_avg([(1.0, 80.0), (1.0, 60.0)])
        assert result == pytest.approx(70.0)

    def test_skips_none(self):
        result = _weighted_avg([(1.0, 80.0), (1.0, None), (1.0, 60.0)])
        assert result == pytest.approx(70.0)

    def test_all_none(self):
        assert _weighted_avg([(1.0, None), (1.0, None)]) is None

    def test_weighted(self):
        result = _weighted_avg([(3.0, 90.0), (1.0, 50.0)])
        assert result == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# Scorecard integration tests
# ---------------------------------------------------------------------------


class TestComputeScorecard:
    def test_empty_returns_insufficient(self):
        result = compute_scorecard()
        assert result["status"] == "insufficient_data"
        assert result["overall"]["grade"] is None
        assert result["research"]["grade"] is None
        assert result["predictor"]["grade"] is None
        assert result["executor"]["grade"] is None

    def test_partial_data(self):
        result = compute_scorecard(
            signal_quality={
                "status": "ok",
                "overall": {"accuracy_21d": 0.58, "avg_alpha_21d": 1.5, "n_21d": 50},
                "by_score_bucket": [{"bucket": "90+", "accuracy_21d": 0.71}],
            },
        )
        # Only portfolio sub-component has data → executor partial, research/predictor empty
        assert result["status"] in ("partial", "insufficient_data")

    def test_full_data_produces_grades(self):
        result = compute_scorecard(
            signal_quality={
                "status": "ok",
                "overall": {"accuracy_21d": 0.58, "avg_alpha_21d": 1.5, "n_21d": 50},
                "by_score_bucket": [{"bucket": "90+", "accuracy_21d": 0.71}],
            },
            e2e_lift={
                "status": "ok",
                "scanner_lift": {"lift": 1.2, "n_passing": 55, "n_universe": 900},
                "team_lift": [
                    {"team_id": "technology", "lift": 2.5, "lift_vs_quant": 1.1, "n_picks": 12},
                ],
                "cio_lift": {
                    "lift": 1.5, "advance_avg": 2.1, "reject_avg": -0.3,
                    "n_advance": 15, "n_reject": 20,
                },
                "cio_vs_ranking": {
                    "lift": 0.8, "cio_beats_ranking": True,
                    "n_dates": 8, "n_picks": 15, "avg_overlap": 0.6,
                    "cio_avg": 2.1, "ranking_avg": 1.3,
                },
            },
            predictor_sizing={
                "status": "ok", "overall_rank_ic": 0.06,
                "recent_positive_weeks": 6, "recent_total_weeks": 8,
                "sizing_lift": 0.3, "n_samples": 100,
            },
            veto_result={
                "status": "ok", "current_threshold": 0.65, "base_rate": 0.55,
                "thresholds": [{
                    "confidence": 0.65, "precision": 0.68, "lift": 13.0,
                    "n_vetoes": 25, "true_negatives": 17, "false_negatives": 8,
                    "missed_alpha": 2.1,
                }],
                "recommended_threshold": 0.65,
            },
            veto_value={"net_value": 420.0},
            trigger_scorecard={
                "status": "ok",
                "triggers": [{"trigger": "pullback", "n_trades": 20, "avg_slippage_vs_signal": -0.3, "win_rate_vs_spy": 0.55}],
                "summary": {"total_entries": 35, "avg_slippage_vs_signal": -0.4, "win_rate_vs_spy": 0.57, "avg_realized_alpha": 1.2},
            },
            shadow_book={
                "status": "ok", "n_blocked": 12, "n_traded": 35,
                "guard_lift": 1.5, "blocked_beat_spy_pct": 0.33, "assessment": "appropriate",
            },
            exit_timing={
                "status": "ok", "n_roundtrips": 28,
                "summary": {"avg_capture_ratio": 0.62, "avg_realized_return": 1.8},
                "diagnosis": "exits_could_improve",
            },
        )

        assert result["status"] == "ok"
        assert result["overall"]["grade"] is not None
        assert 0 <= result["overall"]["grade"] <= 100
        assert result["overall"]["letter"] != "N/A"

        # All modules should have grades
        assert result["research"]["grade"] is not None
        assert result["predictor"]["grade"] is not None
        assert result["executor"]["grade"] is not None

    def test_team_grades_ordered(self):
        """A team with higher lift should get a higher grade."""
        result = compute_scorecard(
            e2e_lift={
                "status": "ok",
                "scanner_lift": {"lift": 1.0, "n_passing": 50, "n_universe": 900},
                "team_lift": [
                    {"team_id": "good_team", "lift": 3.0, "lift_vs_quant": 2.0, "n_picks": 15},
                    {"team_id": "bad_team", "lift": -1.5, "lift_vs_quant": -1.0, "n_picks": 10},
                ],
                "cio_lift": {"lift": 1.0, "advance_avg": 1.5, "reject_avg": -0.5, "n_advance": 10, "n_reject": 8},
                "cio_vs_ranking": {"lift": 0.5, "cio_beats_ranking": True, "n_dates": 8, "n_picks": 10, "avg_overlap": 0.5, "cio_avg": 1.5, "ranking_avg": 1.0},
            },
        )
        teams = result["research"]["components"]["sector_teams"]
        good = next(t for t in teams if t["team_id"] == "good_team")
        bad = next(t for t in teams if t["team_id"] == "bad_team")
        assert good["grade"] > bad["grade"]

    def test_insufficient_team_picks(self):
        """Teams with fewer than 3 picks get N/A."""
        result = compute_scorecard(
            e2e_lift={
                "status": "ok",
                "scanner_lift": {"lift": 1.0, "n_passing": 50, "n_universe": 900},
                "team_lift": [
                    {"team_id": "tiny_team", "lift": 5.0, "lift_vs_quant": 3.0, "n_picks": 2},
                ],
                "cio_lift": {"lift": 1.0, "advance_avg": 1.5, "reject_avg": -0.5, "n_advance": 10, "n_reject": 8},
                "cio_vs_ranking": {"lift": 0.5, "cio_beats_ranking": True, "n_dates": 8, "n_picks": 10, "avg_overlap": 0.5, "cio_avg": 1.5, "ranking_avg": 1.0},
            },
        )
        teams = result["research"]["components"]["sector_teams"]
        tiny = next(t for t in teams if t["team_id"] == "tiny_team")
        assert tiny["grade"] is None
        assert tiny["letter"] == "N/A"

    def test_scorecard_structure(self):
        """Verify the scorecard has the expected structure."""
        result = compute_scorecard()
        assert "status" in result
        assert "overall" in result
        assert "research" in result
        assert "predictor" in result
        assert "executor" in result
        assert "grade" in result["overall"]
        assert "letter" in result["overall"]
        assert "components" in result["research"]
        assert "components" in result["predictor"]
        assert "components" in result["executor"]

    def test_classification_metrics_in_grading(self):
        """When e2e_lift includes classification dicts, grading uses them."""
        clf = {"precision": 0.65, "recall": 0.30, "f1": 0.41, "tp": 20, "fp": 11, "fn": 47, "tn": 22, "n": 100}
        result = compute_scorecard(
            e2e_lift={
                "status": "ok",
                "scanner_lift": {"lift": 1.0, "n_passing": 50, "n_universe": 900, "classification": clf},
                "team_lift": [
                    {"team_id": "tech", "lift": 2.0, "lift_vs_quant": 1.0, "n_picks": 10,
                     "classification": {"precision": 0.70, "recall": 0.35, "f1": 0.47, "tp": 7, "fp": 3, "fn": 13, "tn": 17, "n": 40}},
                ],
                "cio_lift": {
                    "lift": 1.0, "advance_avg": 2.0, "reject_avg": -0.5,
                    "n_advance": 10, "n_reject": 8,
                    "classification": {"precision": 0.60, "recall": 0.50, "f1": 0.55, "tp": 6, "fp": 4, "fn": 6, "tn": 4, "n": 20},
                },
                "cio_vs_ranking": {"lift": 0.5, "cio_beats_ranking": True, "n_dates": 4, "n_picks": 10, "avg_overlap": 0.5, "cio_avg": 2.0, "ranking_avg": 1.5},
            },
            shadow_book={
                "status": "ok", "n_blocked": 10, "n_traded": 30,
                "guard_lift": 1.0, "assessment": "appropriate",
                "classification": {"precision": 0.70, "recall": 0.20, "f1": 0.31, "tp": 7, "fp": 3, "fn": 28, "tn": 12, "n": 50},
            },
            veto_result={
                "status": "ok", "current_threshold": 0.65, "base_rate": 0.55,
                "thresholds": [{
                    "confidence": 0.65, "precision": 0.68, "recall": 0.40, "f1": 0.50,
                    "lift": 13.0, "n_vetoes": 25, "true_negatives": 17, "false_negatives": 8, "missed_alpha": 2.0,
                }],
                "recommended_threshold": 0.65,
            },
        )

        # Scanner should show P/R in detail
        scanner = result["research"]["components"]["scanner"]
        assert "precision" in scanner["detail"]
        assert "recall" in scanner["detail"]
        assert "f1" in scanner["detail"]

        # Team should show P/R in detail
        teams = result["research"]["components"]["sector_teams"]
        assert "precision" in teams[0]["detail"]
        assert "recall" in teams[0]["detail"]

        # CIO should show P/R in detail
        cio = result["research"]["components"]["cio"]
        assert "precision" in cio["detail"]
        assert "recall" in cio["detail"]

        # Risk guard should show P/R in detail
        guard = result["executor"]["components"]["risk_guard"]
        assert "precision" in guard["detail"]
        assert "recall" in guard["detail"]

        # Veto gate should show recall
        veto = result["predictor"]["components"]["veto_gate"]
        assert "recall" in veto["detail"]
        assert "f1" in veto["detail"]


# ---------------------------------------------------------------------------
# Evaluator-revamp helpers (PR 3): _band_to_grade + _cvar_to_grade
# ---------------------------------------------------------------------------


class TestBandToGrade:
    def test_three_anchor_mapping(self):
        # floor=0, mid=1, ceiling=2 → 0/50/100 at the anchors.
        assert _band_to_grade(0.0, 0.0, 1.0, 2.0) == pytest.approx(0.0)
        assert _band_to_grade(1.0, 0.0, 1.0, 2.0) == pytest.approx(50.0)
        assert _band_to_grade(2.0, 0.0, 1.0, 2.0) == pytest.approx(100.0)
        # Linear in each half.
        assert _band_to_grade(0.5, 0.0, 1.0, 2.0) == pytest.approx(25.0)
        assert _band_to_grade(1.5, 0.0, 1.0, 2.0) == pytest.approx(75.0)

    def test_clamps_outside_range(self):
        assert _band_to_grade(-5.0, 0.0, 1.0, 2.0) == 0.0
        assert _band_to_grade(99.0, 0.0, 1.0, 2.0) == 100.0

    def test_invalid_anchors_raise(self):
        with pytest.raises(ValueError):
            _band_to_grade(0.5, 1.0, 1.0, 2.0)  # floor == mid
        with pytest.raises(ValueError):
            _band_to_grade(0.5, 2.0, 1.0, 0.5)  # decreasing

    def test_none_returns_none(self):
        assert _band_to_grade(None, 0.0, 1.0, 2.0) is None


class TestCvarToGrade:
    def test_zero_or_positive_full_grade(self):
        assert _cvar_to_grade(0.0) == 100.0
        assert _cvar_to_grade(0.01) == 100.0

    def test_baseline_anchor(self):
        # Default baseline -4% → 30 (D+).
        assert _cvar_to_grade(-0.04) == pytest.approx(30.0)

    def test_ceiling_anchor(self):
        # Default ceiling -1% → 95 (A).
        assert _cvar_to_grade(-0.01) == pytest.approx(95.0)

    def test_linear_in_band(self):
        # Mid-point between -1% and -4% is -2.5% → halfway between 30 and 95 = 62.5.
        assert _cvar_to_grade(-0.025) == pytest.approx(62.5)


# ---------------------------------------------------------------------------
# Evaluator-revamp graders (PR 3): calibration / action entropy / excursion
# ---------------------------------------------------------------------------


class TestGradeCalibrationDiagnostics:
    def test_good_calibration_high_grade(self):
        result = _grade_calibration_diagnostics({
            "status": "ok", "ece": 0.03, "n": 200, "quality": "good",
        })
        assert result["grade"] == 90.0
        assert result["letter"] == "A"

    def test_poor_calibration_low_grade(self):
        result = _grade_calibration_diagnostics({
            "status": "ok", "ece": 0.25, "n": 200, "quality": "poor",
        })
        assert result["grade"] == 10.0
        assert result["letter"] == "F"

    def test_missing_ece(self):
        result = _grade_calibration_diagnostics({"status": "ok"})
        assert result["grade"] is None

    def test_status_not_ok(self):
        result = _grade_calibration_diagnostics({"status": "insufficient_data"})
        assert result["grade"] is None

    def test_none_input(self):
        result = _grade_calibration_diagnostics(None)
        assert result["grade"] is None


class TestGradeActionEntropy:
    def test_uniform_distribution_high_grade(self):
        result = _grade_action_entropy({
            "status": "ok", "entropy_normalized": 1.0, "n": 50,
            "most_common": "BUY", "most_common_fraction": 0.34, "alarm": False,
        })
        assert result["grade"] == 100.0

    def test_collapsed_distribution_low_grade(self):
        result = _grade_action_entropy({
            "status": "ok", "entropy_normalized": 0.0, "n": 50,
            "most_common": "HOLD", "most_common_fraction": 1.0, "alarm": True,
        })
        assert result["grade"] == 0.0

    def test_below_alarm_threshold_capped_at_40(self):
        # entropy_norm = 0.2 → between 0 and 0.3 mid → grade ~33.3
        # Should still be < 40 (alarm cap).
        result = _grade_action_entropy({
            "status": "ok", "entropy_normalized": 0.2, "n": 50,
            "most_common": "HOLD", "most_common_fraction": 0.85, "alarm": True,
        })
        assert result["grade"] is not None
        assert result["grade"] < 40.0
        assert result["detail"]["alarm"] is True

    def test_status_not_ok(self):
        result = _grade_action_entropy({"status": "insufficient_data", "n": 5})
        assert result["grade"] is None


class TestGradeExcursion:
    def test_high_quality_team_high_grade(self):
        # mean_mfe_mae_ratio = 1.8 (between mid 1.5 and ceiling 2.0)
        # pct_high_quality = 0.6 (at ceiling) → 95
        result = _grade_excursion({
            "status": "ok", "n": 30,
            "mean_mfe_mae_ratio": 1.8,
            "median_mfe_mae_ratio": 1.7,
            "pct_high_quality": 0.6,
            "pct_mfe_gt_mae": 0.7,
        })
        assert result["grade"] is not None
        assert result["grade"] > 75.0

    def test_low_quality_team_low_grade(self):
        # ratio ≈ 1.0 (below mid 1.5), pct_high = 0.1 (below baseline 0.3)
        result = _grade_excursion({
            "status": "ok", "n": 30,
            "mean_mfe_mae_ratio": 1.0,
            "median_mfe_mae_ratio": 1.0,
            "pct_high_quality": 0.1,
            "pct_mfe_gt_mae": 0.3,
        })
        assert result["grade"] is not None
        assert result["grade"] < 40.0

    def test_status_not_ok(self):
        result = _grade_excursion({"status": "insufficient_data", "n": 0})
        assert result["grade"] is None


# ---------------------------------------------------------------------------
# compute_scorecard wiring for new graders + portfolio rewire
# ---------------------------------------------------------------------------


class TestComputeScorecardEvaluatorRevamp:
    def test_calibration_appears_in_research_when_provided(self):
        result = compute_scorecard(
            calibration_diagnostics={
                "status": "ok", "ece": 0.04, "n": 200, "quality": "good",
            },
        )
        assert "calibration_diagnostics" in result["research"]["components"]
        assert result["research"]["components"]["calibration_diagnostics"]["grade"] == 90.0

    def test_calibration_absent_does_not_break_research(self):
        # Pre-existing test pinned 'sector_teams' etc. — calibration must
        # not appear when the kwarg is None.
        result = compute_scorecard()
        assert "calibration_diagnostics" not in result["research"]["components"]

    def test_excursion_and_entropy_appear_in_executor_when_provided(self):
        result = compute_scorecard(
            excursion_summary={
                "status": "ok", "n": 30,
                "mean_mfe_mae_ratio": 1.6,
                "median_mfe_mae_ratio": 1.5,
                "pct_high_quality": 0.45,
                "pct_mfe_gt_mae": 0.6,
            },
            action_entropy={
                "status": "ok", "entropy_normalized": 0.85, "n": 50,
                "most_common": "BUY", "most_common_fraction": 0.4, "alarm": False,
            },
        )
        assert "excursion" in result["executor"]["components"]
        assert "action_entropy" in result["executor"]["components"]
        assert result["executor"]["components"]["excursion"]["grade"] is not None
        assert result["executor"]["components"]["action_entropy"]["grade"] is not None

    def test_portfolio_uses_sortino_path_when_new_fields_present(self):
        result = compute_scorecard(
            signal_quality={
                "status": "ok",
                "overall": {"accuracy_21d": 0.55, "avg_alpha_21d": 1.0, "n_21d": 50},
                "by_score_bucket": [],
            },
            portfolio_stats={
                "sharpe_ratio": 0.8, "sortino_ratio": 1.2, "calmar_ratio": 0.5,
                "max_drawdown": -0.10, "cvar_95": -0.025, "total_return": 0.05,
            },
        )
        portfolio_detail = result["executor"]["components"]["portfolio"]["detail"]
        assert "sortino" in portfolio_detail
        assert "cvar_95" in portfolio_detail
        assert "calmar" in portfolio_detail
        # Sharpe still emitted as side-channel diagnostic.
        assert "sharpe" in portfolio_detail

    def test_portfolio_falls_back_to_legacy_when_new_fields_absent(self):
        result = compute_scorecard(
            signal_quality={
                "status": "ok",
                "overall": {"accuracy_21d": 0.55, "avg_alpha_21d": 1.0, "n_21d": 50},
                "by_score_bucket": [],
            },
            portfolio_stats={
                "sharpe_ratio": 1.2, "max_drawdown": -0.10, "total_return": 0.10,
            },
        )
        portfolio_detail = result["executor"]["components"]["portfolio"]["detail"]
        # Legacy fields present
        assert "sharpe" in portfolio_detail
        assert "max_drawdown" in portfolio_detail
        # New fields absent (not in input → not in detail)
        assert "sortino" not in portfolio_detail
        assert "cvar_95" not in portfolio_detail


class TestSectorTeamSkillComposite:
    def test_team_metrics_path_used_when_provided(self):
        result = compute_scorecard(
            e2e_lift={
                "status": "ok",
                "scanner_lift": {"lift": 1.0, "n_passing": 50, "n_universe": 900},
                "team_lift": [
                    {"team_id": "tech", "lift": 2.0, "lift_vs_quant": 1.0, "n_picks": 12},
                ],
                "cio_lift": {"lift": 1.0, "advance_avg": 1.5, "reject_avg": -0.5,
                             "n_advance": 10, "n_reject": 8},
                "cio_vs_ranking": {"lift": 0.5, "cio_beats_ranking": True, "n_dates": 8,
                                   "n_picks": 10, "avg_overlap": 0.5,
                                   "cio_avg": 1.5, "ranking_avg": 1.0},
            },
            team_metrics={
                "tech": {
                    "ic": {"status": "ok", "ic": 0.08, "n": 60, "n_buckets": 10},
                    "expectancy": {
                        "status": "ok", "n": 12, "hit_rate": 0.58,
                        "avg_win": 0.05, "avg_loss": 0.03,
                        "win_loss_ratio": 1.67, "expectancy": 0.018,
                        "expectancy_per_unit_loss": 0.6,
                    },
                    "excursion": {
                        "status": "ok", "n": 12, "mean_mfe_mae_ratio": 1.7,
                        "median_mfe_mae_ratio": 1.6,
                        "pct_mfe_gt_mae": 0.65, "pct_high_quality": 0.55,
                    },
                    "alpha_vs_ew_high_vol": {
                        "status": "ok", "excess_return": 0.02,
                        "information_ratio": 1.2,
                    },
                    "alpha_vs_beta_spy": {
                        "status": "ok", "excess_return": 0.015,
                        "information_ratio": 0.8,
                    },
                },
            },
        )
        teams = result["research"]["components"]["sector_teams"]
        tech = next(t for t in teams if t["team_id"] == "tech")
        # Skill-composite detail keys (not legacy precision/recall/lift_vs_sector).
        assert "ic" in tech["detail"]
        assert "expectancy" in tech["detail"]
        assert "mfe_mae_ratio" in tech["detail"]
        assert "alpha_vs_ew_high_vol" in tech["detail"]
        assert "alpha_vs_beta_spy" in tech["detail"]
        assert tech["grade"] is not None

    def test_team_metrics_absent_falls_back_to_legacy(self):
        # No team_metrics kwarg → legacy path. Existing test contract.
        result = compute_scorecard(
            e2e_lift={
                "status": "ok",
                "scanner_lift": {"lift": 1.0, "n_passing": 50, "n_universe": 900},
                "team_lift": [
                    {"team_id": "tech", "lift": 2.0, "lift_vs_quant": 1.0, "n_picks": 12},
                ],
                "cio_lift": {"lift": 1.0, "advance_avg": 1.5, "reject_avg": -0.5,
                             "n_advance": 10, "n_reject": 8},
                "cio_vs_ranking": {"lift": 0.5, "cio_beats_ranking": True, "n_dates": 8,
                                   "n_picks": 10, "avg_overlap": 0.5,
                                   "cio_avg": 1.5, "ranking_avg": 1.0},
            },
        )
        teams = result["research"]["components"]["sector_teams"]
        tech = next(t for t in teams if t["team_id"] == "tech")
        # Legacy detail keys.
        assert "lift_vs_sector" in tech["detail"]
        assert "lift_vs_quant" in tech["detail"]
