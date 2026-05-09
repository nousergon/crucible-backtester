"""Unit tests for analysis.veto_analysis skill-composite cutover.

Mirrors the executor_optimizer skill-composite cutover from PR #158:
flag-gated ranking switch + DSR-style confidence-bounded lift gate +
shadow-vs-production write paths in apply().

Per Brian (2026-05-09): alpha vs SPY is presentation framing, not the
optimizer's fit target. The veto skill-composite drops the
``cost_penalty_weight × |missed_alpha|`` term and ranks on F1 (precision +
recall harmonic mean), with a confidence-bounded lift gate that blocks
promotion when the precision CI lower bound doesn't beat base rate.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from analysis.veto_analysis import (
    S3_PARAMS_KEY,
    S3_SHADOW_PREFIX,
    _select_best_threshold,
    apply,
    init_config,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _init_config(extra: dict | None = None):
    cfg = {
        "veto_analysis": {
            "confidence_thresholds": [0.50, 0.60, 0.70, 0.80],
            "current_default_threshold": 0.60,
            "min_predictions": 10,
            "min_veto_decisions": 3,
            "min_threshold_change": 0.05,
            "cost_penalty_weight": 0.30,
            "min_lift_over_base_rate": 0.05,
        }
    }
    if extra:
        cfg["veto_analysis"].update(extra)
    init_config(cfg)


def _build_threshold_results(rows: list[dict]) -> list[dict]:
    """Construct threshold_results matching what _sweep_thresholds would emit.

    Each row needs: confidence, n_vetoes, true_negatives, false_negatives,
    precision, recall, f1, missed_alpha, lift, precision_ci_95.
    """
    return [
        {
            "confidence": r["confidence"],
            "n_vetoes": r.get("n_vetoes", 10),
            "true_negatives": r.get("true_negatives", 7),
            "false_negatives": r.get("false_negatives", 3),
            "precision": r["precision"],
            "recall": r.get("recall", 0.50),
            "f1": r.get("f1", 0.55),
            "missed_alpha": r.get("missed_alpha", -0.01),
            "lift": r.get("lift", 0.10),
            "precision_ci_95": r.get("precision_ci_95", (0.55, 0.85)),
            "low_confidence": False,
        }
        for r in rows
    ]


# ── _select_best_threshold dispatch on flag ─────────────────────────────────


class TestSelectBestThresholdRanking:

    def test_legacy_picks_max_precision_minus_alpha_cost(self):
        """Legacy ranking: combos with same precision but different missed_alpha
        rank by alpha-cost-penalty. Higher missed_alpha = lower score."""
        _init_config()  # legacy mode (use_skill_composite_target=False default)
        # Combo A: precision 0.70, low missed_alpha → favored under legacy
        # Combo B: precision 0.70, high missed_alpha → penalized
        threshold_results = _build_threshold_results([
            {"confidence": 0.60, "precision": 0.70, "missed_alpha": -0.01,
             "f1": 0.50, "lift": 0.20},
            {"confidence": 0.70, "precision": 0.70, "missed_alpha": -0.05,
             "f1": 0.65, "lift": 0.20},
        ])
        result = _select_best_threshold(
            threshold_results, base_rate=0.50, cost_weight=0.30,
            current_default=0.55, min_veto_dec=3, n_down=20, n_preds_loaded=100,
        )
        assert result["status"] == "ok"
        assert result["fit_target"] == "precision_minus_alpha_cost_legacy"
        # Combo A wins under legacy (lower alpha cost) despite identical precision.
        assert result["recommended_threshold"] == 0.60

    def test_skill_picks_max_f1(self):
        """Skill-composite ranking: drops alpha cost, ranks on F1.
        Same fixture as legacy test → different winner because F1 differs."""
        _init_config({"use_skill_composite_target": True})
        threshold_results = _build_threshold_results([
            {"confidence": 0.60, "precision": 0.70, "missed_alpha": -0.01,
             "f1": 0.50, "lift": 0.20},
            {"confidence": 0.70, "precision": 0.70, "missed_alpha": -0.05,
             "f1": 0.65, "lift": 0.20},
        ])
        result = _select_best_threshold(
            threshold_results, base_rate=0.50, cost_weight=0.30,
            current_default=0.55, min_veto_dec=3, n_down=20, n_preds_loaded=100,
        )
        assert result["status"] == "ok"
        assert result["fit_target"] == "skill_composite"
        # Combo B wins under skill (higher F1) — alpha cost is presentation,
        # not gating.
        assert result["recommended_threshold"] == 0.70

    def test_skill_falls_back_to_precision_when_f1_none(self):
        """When F1 is None (recall undefined — no actual underperformers),
        skill ranking falls back to precision."""
        _init_config({"use_skill_composite_target": True})
        threshold_results = _build_threshold_results([
            {"confidence": 0.60, "precision": 0.65, "f1": None, "lift": 0.15},
            {"confidence": 0.70, "precision": 0.75, "f1": None, "lift": 0.25},
        ])
        result = _select_best_threshold(
            threshold_results, base_rate=0.50, cost_weight=0.30,
            current_default=0.55, min_veto_dec=3, n_down=20, n_preds_loaded=100,
        )
        # Higher precision wins.
        assert result["recommended_threshold"] == 0.70
        assert result["fit_target"] == "skill_composite"


class TestSkillCompositeConfidenceGate:
    """Workstream D bullet 3 of evaluator-revamp: DSR-style gate blocks
    promotion when precision CI lower bound doesn't beat base rate."""

    def test_skill_blocks_promotion_when_ci_does_not_clear_base_rate(self):
        _init_config({"use_skill_composite_target": True})
        # Best combo: precision 0.55, base_rate 0.50, but CI lower 0.48 < base_rate
        threshold_results = _build_threshold_results([
            {"confidence": 0.70, "precision": 0.55, "f1": 0.40,
             "lift": 0.10, "precision_ci_95": (0.48, 0.62)},
        ])
        result = _select_best_threshold(
            threshold_results, base_rate=0.50, cost_weight=0.30,
            current_default=0.55, min_veto_dec=3, n_down=20, n_preds_loaded=100,
        )
        assert result["status"] == "insufficient_confidence"
        assert "statistically indistinguishable" in result["recommendation_reason"]

    def test_skill_promotes_when_ci_clears_base_rate(self):
        _init_config({"use_skill_composite_target": True})
        # CI lower 0.55 > base_rate 0.50 → confidence gate passes
        threshold_results = _build_threshold_results([
            {"confidence": 0.70, "precision": 0.65, "f1": 0.55,
             "lift": 0.15, "precision_ci_95": (0.55, 0.75)},
        ])
        result = _select_best_threshold(
            threshold_results, base_rate=0.50, cost_weight=0.30,
            current_default=0.55, min_veto_dec=3, n_down=20, n_preds_loaded=100,
        )
        assert result["status"] == "ok"

    def test_legacy_skips_confidence_gate(self):
        """Confidence gate is skill-mode only — legacy path passes through."""
        _init_config()  # legacy default
        threshold_results = _build_threshold_results([
            {"confidence": 0.70, "precision": 0.55, "f1": 0.40,
             "lift": 0.10, "precision_ci_95": (0.48, 0.62)},
        ])
        result = _select_best_threshold(
            threshold_results, base_rate=0.50, cost_weight=0.30,
            current_default=0.55, min_veto_dec=3, n_down=20, n_preds_loaded=100,
        )
        # Legacy doesn't gate on CI — just checks the lift threshold.
        assert result["status"] == "ok"


class TestApplyShadowVsProduction:
    """The two-stage activation: skill_composite computed → shadow archive
    until enforce_skill_composite is also flipped. Mirrors PR #158's
    executor_optimizer cutover pattern exactly."""

    def _ok_result(self, fit_target: str, threshold: float = 0.70) -> dict:
        return {
            "status": "ok",
            "fit_target": fit_target,
            "current_threshold": 0.55,
            "base_rate": 0.50,
            "n_down_predictions": 30,
            "thresholds": [
                {"confidence": threshold, "precision": 0.65, "n_vetoes": 12,
                 "f1": 0.55, "lift": 0.15},
            ],
            "recommended_threshold": threshold,
            "recommendation_reason": "test",
        }

    @patch("analysis.veto_analysis.boto3")
    def test_legacy_writes_to_production_key(self, mock_boto3):
        _init_config()
        s3 = MagicMock()
        # Force read_current_veto_threshold to return None (NoSuchKey path)
        s3.get_object.side_effect = Exception("NoSuchKey")
        mock_boto3.client.return_value = s3
        with patch("optimizer.rollback.save_previous"):
            outcome = apply(
                self._ok_result("precision_minus_alpha_cost_legacy"),
                bucket="test-bucket",
            )
        assert outcome["applied"] is True
        keys_written = [c.kwargs["Key"] for c in s3.put_object.call_args_list]
        assert S3_PARAMS_KEY in keys_written
        # No shadow write under legacy.
        assert not any(S3_SHADOW_PREFIX in k for k in keys_written)

    @patch("analysis.veto_analysis.boto3")
    def test_skill_without_enforce_writes_to_shadow_only(self, mock_boto3):
        _init_config({
            "use_skill_composite_target": True,
            "enforce_skill_composite": False,
        })
        s3 = MagicMock()
        s3.get_object.side_effect = Exception("NoSuchKey")
        mock_boto3.client.return_value = s3
        outcome = apply(
            self._ok_result("skill_composite"), bucket="test-bucket",
        )
        assert outcome["applied"] is False
        assert "shadow mode" in outcome["reason"].lower()
        assert outcome["fit_target"] == "skill_composite"
        keys_written = [c.kwargs["Key"] for c in s3.put_object.call_args_list]
        assert any(S3_SHADOW_PREFIX in k for k in keys_written)
        # Live key NOT written.
        assert S3_PARAMS_KEY not in keys_written

    @patch("analysis.veto_analysis.boto3")
    def test_skill_with_enforce_writes_to_production(self, mock_boto3):
        _init_config({
            "use_skill_composite_target": True,
            "enforce_skill_composite": True,
        })
        s3 = MagicMock()
        s3.get_object.side_effect = Exception("NoSuchKey")
        mock_boto3.client.return_value = s3
        with patch("optimizer.rollback.save_previous"):
            outcome = apply(
                self._ok_result("skill_composite"), bucket="test-bucket",
            )
        assert outcome["applied"] is True
        assert outcome["fit_target"] == "skill_composite"
        keys_written = [c.kwargs["Key"] for c in s3.put_object.call_args_list]
        assert S3_PARAMS_KEY in keys_written
        assert not any(S3_SHADOW_PREFIX in k for k in keys_written)

    @patch("analysis.veto_analysis.boto3")
    def test_apply_payload_includes_fit_target(self, mock_boto3):
        _init_config({
            "use_skill_composite_target": True,
            "enforce_skill_composite": True,
        })
        s3 = MagicMock()
        s3.get_object.side_effect = Exception("NoSuchKey")
        mock_boto3.client.return_value = s3
        with patch("optimizer.rollback.save_previous"):
            apply(self._ok_result("skill_composite"), bucket="test-bucket")
        live_call = next(
            c for c in s3.put_object.call_args_list
            if c.kwargs["Key"] == S3_PARAMS_KEY
        )
        body = json.loads(live_call.kwargs["Body"])
        assert body["fit_target"] == "skill_composite"
        assert body["veto_confidence"] == 0.70
