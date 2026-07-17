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

from optimizer.assembler import set_cutover_enabled
from analysis.veto_analysis import (
    S3_PARAMS_KEY,
    S3_SHADOW_PREFIX,
    _select_best_threshold,
    apply,
    init_config,
    produce_artifact,
)


@pytest.fixture(autouse=True)
def _reset_cutover_flag():
    """Always reset the assembler cutover flag around each test so
    set_cutover_enabled(True) in one test never leaks into the next."""
    set_cutover_enabled(False)
    yield
    set_cutover_enabled(False)


@pytest.fixture(autouse=True)
def _mock_recommendation_artifact_s3():
    """apply() now unconditionally calls produce_artifact() (config#2054),
    which writes to S3 via optimizer.recommendation_artifact — a module none
    of the pre-existing tests in this file mock. Autouse so every test is
    protected from making a real AWS call."""
    with patch("optimizer.recommendation_artifact.boto3") as mock_boto3:
        mock_boto3.client.return_value = MagicMock()
        yield mock_boto3


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

    def _ok_result_with_threshold(self, threshold: float) -> dict:
        """Helper for bootstrap tests — no significance verdict."""
        result = self._ok_result("precision_minus_alpha_cost_legacy", threshold)
        return result

    def _ok_result_with_verdict(self, verdict: dict) -> dict:
        """Helper for significance tests."""
        result = self._ok_result("precision_minus_alpha_cost_legacy")
        result["significance_observe"] = verdict
        return result

    def _ok_result_with_threshold_and_verdict(self, threshold: float, verdict: dict) -> dict:
        """Helper for bootstrap + significance tests."""
        result = self._ok_result("precision_minus_alpha_cost_legacy", threshold)
        result["significance_observe"] = verdict
        return result

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

    # ── config#1426 Phase 4 — significance ENFORCE (default OFF) ─────────────

    @patch("analysis.veto_analysis.boto3")
    def test_enforce_default_off_applies_even_when_would_block(self, mock_boto3):
        """CRITICAL non-enforcement guarantee: default (no enforce flag) leaves
        the live promotion untouched even when the verdict is would_block."""
        _init_config()  # no enforce_significance → defaults False
        s3 = MagicMock()
        s3.get_object.side_effect = Exception("NoSuchKey")
        mock_boto3.client.return_value = s3
        with patch("optimizer.rollback.save_previous"):
            outcome = apply(
                self._ok_result_with_verdict({"would_block": True, "significant": False}),
                bucket="test-bucket",
            )
        assert outcome["applied"] is True

    @patch("analysis.veto_analysis.boto3")
    def test_enforce_blocks_insignificant(self, mock_boto3):
        _init_config({"enforce_significance": True})
        s3 = MagicMock()
        s3.get_object.side_effect = Exception("NoSuchKey")
        mock_boto3.client.return_value = s3
        with patch("optimizer.rollback.save_previous"):
            outcome = apply(
                self._ok_result_with_verdict({"would_block": True, "significant": False}),
                bucket="test-bucket",
            )
        assert outcome["applied"] is False
        assert "significance enforce" in outcome["reason"]
        assert S3_PARAMS_KEY not in [c.kwargs["Key"] for c in s3.put_object.call_args_list]

    @patch("analysis.veto_analysis.boto3")
    def test_enforce_allows_significant(self, mock_boto3):
        _init_config({"enforce_significance": True})
        s3 = MagicMock()
        s3.get_object.side_effect = Exception("NoSuchKey")
        mock_boto3.client.return_value = s3
        with patch("optimizer.rollback.save_previous"):
            outcome = apply(
                self._ok_result_with_verdict({"would_block": False, "significant": True}),
                bucket="test-bucket",
            )
        assert outcome["applied"] is True

    @patch("analysis.veto_analysis.boto3")
    def test_enforce_missing_verdict_blocks_conservatively(self, mock_boto3):
        _init_config({"enforce_significance": True})
        s3 = MagicMock()
        s3.get_object.side_effect = Exception("NoSuchKey")
        mock_boto3.client.return_value = s3
        with patch("optimizer.rollback.save_previous"):
            outcome = apply(
                self._ok_result_with_verdict(None), bucket="test-bucket",
            )
        assert outcome["applied"] is False

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

    @patch("analysis.veto_analysis.boto3")
    def test_bootstrap_seed_write_when_no_prior_artifact(self, mock_boto3):
        """Bootstrap: first write occurs even when recommendation = default (fixed-point trap fix)."""
        _init_config({"enforce_significance": False})
        s3 = MagicMock()
        s3.get_object.side_effect = Exception("NoSuchKey")  # No prior S3 artifact
        mock_boto3.client.return_value = s3
        with patch("optimizer.rollback.save_previous"):
            # Recommendation = 0.65 (the default) — normally blocked by min_threshold_change
            # But bootstrap allows it when artifact doesn't exist
            outcome = apply(
                self._ok_result_with_threshold(0.65),
                bucket="test-bucket",
            )
        assert outcome["applied"] is True
        live_call = next(
            c for c in s3.put_object.call_args_list
            if c.kwargs["Key"] == S3_PARAMS_KEY
        )
        body = json.loads(live_call.kwargs["Body"])
        assert body["veto_confidence"] == 0.65

    @patch("analysis.veto_analysis.boto3")
    def test_bootstrap_seed_blocked_by_significance_floor(self, mock_boto3):
        """Bootstrap seed respects significance floor when enforce_significance=true."""
        _init_config({"enforce_significance": True})
        s3 = MagicMock()
        s3.get_object.side_effect = Exception("NoSuchKey")  # No prior artifact
        mock_boto3.client.return_value = s3
        outcome = apply(
            self._ok_result_with_threshold_and_verdict(0.65, {"would_block": True}),
            bucket="test-bucket",
        )
        assert outcome["applied"] is False
        assert "bootstrap seed blocked by significance enforce" in outcome["reason"]
        assert S3_PARAMS_KEY not in [c.kwargs["Key"] for c in s3.put_object.call_args_list]

    @patch("analysis.veto_analysis.boto3")
    def test_bootstrap_seed_allowed_when_significance_passes(self, mock_boto3):
        """Bootstrap seed writes when significance verdict is clean."""
        _init_config({"enforce_significance": True})
        s3 = MagicMock()
        s3.get_object.side_effect = Exception("NoSuchKey")  # No prior artifact
        mock_boto3.client.return_value = s3
        with patch("optimizer.rollback.save_previous"):
            outcome = apply(
                self._ok_result_with_threshold_and_verdict(0.65, {"would_block": False}),
                bucket="test-bucket",
            )
        assert outcome["applied"] is True


class TestApplyPerSectorShadowSoak:
    """config#921 shadow soak (Brian's ruling 2026-07-07): fitted per-sector
    overrides get attached to the SAME predictor_params.json / shadow-history
    artifact veto_confidence already writes to — gated behind
    ``veto_sector_shadow_enabled`` (default False), never a new S3 prefix.
    """

    def _ok_result_with_overrides(self, overrides: dict | None) -> dict:
        result = {
            "status": "ok",
            "fit_target": "precision_minus_alpha_cost_legacy",
            "current_threshold": 0.55,
            "base_rate": 0.50,
            "n_down_predictions": 30,
            "thresholds": [
                {"confidence": 0.65, "precision": 0.65, "n_vetoes": 12},
            ],
            "recommended_threshold": 0.65,
            "recommendation_reason": "test",
        }
        if overrides is not None:
            result["per_sector_thresholds"] = {
                "status": "ok",
                "overrides": overrides,
            }
        return result

    @patch("analysis.veto_analysis.boto3")
    def test_flag_off_omits_per_sector_overrides(self, mock_boto3):
        """Default (flag unset → False): payload never gets a
        per_sector_overrides key, even when overrides were computed."""
        _init_config()  # veto_sector_shadow_enabled defaults False
        s3 = MagicMock()
        s3.get_object.side_effect = Exception("NoSuchKey")
        mock_boto3.client.return_value = s3
        with patch("optimizer.rollback.save_previous"):
            apply(
                self._ok_result_with_overrides({"Energy": 0.70}),
                bucket="test-bucket",
            )
        live_call = next(
            c for c in s3.put_object.call_args_list
            if c.kwargs["Key"] == S3_PARAMS_KEY
        )
        body = json.loads(live_call.kwargs["Body"])
        assert "per_sector_overrides" not in body
        # Live scalar threshold is unaffected either way.
        assert body["veto_confidence"] == 0.65

    @patch("analysis.veto_analysis.boto3")
    def test_flag_on_no_overrides_computed_omits_key(self, mock_boto3):
        """Flag on but compute_per_sector_thresholds produced an empty
        overrides map — nothing to attach, key stays absent."""
        _init_config({"veto_sector_shadow_enabled": True})
        s3 = MagicMock()
        s3.get_object.side_effect = Exception("NoSuchKey")
        mock_boto3.client.return_value = s3
        with patch("optimizer.rollback.save_previous"):
            apply(
                self._ok_result_with_overrides({}),
                bucket="test-bucket",
            )
        live_call = next(
            c for c in s3.put_object.call_args_list
            if c.kwargs["Key"] == S3_PARAMS_KEY
        )
        body = json.loads(live_call.kwargs["Body"])
        assert "per_sector_overrides" not in body

    @patch("analysis.veto_analysis.boto3")
    def test_flag_on_missing_per_sector_thresholds_key_is_noop(self, mock_boto3):
        """Graceful degrade: result dict without a per_sector_thresholds block
        at all (e.g. no sector column upstream) must not raise."""
        _init_config({"veto_sector_shadow_enabled": True})
        s3 = MagicMock()
        s3.get_object.side_effect = Exception("NoSuchKey")
        mock_boto3.client.return_value = s3
        with patch("optimizer.rollback.save_previous"):
            outcome = apply(
                self._ok_result_with_overrides(None),
                bucket="test-bucket",
            )
        assert outcome["applied"] is True
        live_call = next(
            c for c in s3.put_object.call_args_list
            if c.kwargs["Key"] == S3_PARAMS_KEY
        )
        body = json.loads(live_call.kwargs["Body"])
        assert "per_sector_overrides" not in body

    @patch("analysis.veto_analysis.boto3")
    def test_flag_on_with_overrides_attaches_to_production_payload(self, mock_boto3):
        """Flag on + non-empty overrides: attached verbatim to the live
        predictor_params.json write, alongside the unchanged scalar
        veto_confidence."""
        _init_config({"veto_sector_shadow_enabled": True})
        s3 = MagicMock()
        s3.get_object.side_effect = Exception("NoSuchKey")
        mock_boto3.client.return_value = s3
        overrides = {"Financial Services": 0.60, "Industrials": 0.55}
        with patch("optimizer.rollback.save_previous"):
            outcome = apply(
                self._ok_result_with_overrides(overrides),
                bucket="test-bucket",
            )
        assert outcome["applied"] is True
        live_call = next(
            c for c in s3.put_object.call_args_list
            if c.kwargs["Key"] == S3_PARAMS_KEY
        )
        body = json.loads(live_call.kwargs["Body"])
        assert body["per_sector_overrides"] == overrides
        # Live scalar decision path is untouched by the shadow attachment.
        assert body["veto_confidence"] == 0.65

    @patch("analysis.veto_analysis.boto3")
    def test_flag_on_with_overrides_attaches_to_shadow_payload(self, mock_boto3):
        """Same attachment behavior on the shadow-archive write path
        (skill_composite computed, enforce_skill_composite still False)."""
        _init_config({
            "veto_sector_shadow_enabled": True,
            "use_skill_composite_target": True,
            "enforce_skill_composite": False,
        })
        s3 = MagicMock()
        s3.get_object.side_effect = Exception("NoSuchKey")
        mock_boto3.client.return_value = s3
        overrides = {"Energy": 0.65}
        result = self._ok_result_with_overrides(overrides)
        result["fit_target"] = "skill_composite"
        outcome = apply(result, bucket="test-bucket")
        assert outcome["applied"] is False
        assert "shadow mode" in outcome["reason"].lower()
        shadow_call = next(
            c for c in s3.put_object.call_args_list
            if S3_SHADOW_PREFIX in c.kwargs["Key"]
        )
        body = json.loads(shadow_call.kwargs["Body"])
        assert body["per_sector_overrides"] == overrides


# ═══════════════════════════════════════════════════════════════════════════════
# config#2054: optimizer-artifact-assembler arc extended to predictor_params
# ═══════════════════════════════════════════════════════════════════════════════


def _ok_veto_result(fit_target: str = "precision_minus_alpha_cost_legacy", threshold: float = 0.70) -> dict:
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


class TestProduceArtifact:

    def test_writes_to_canonical_key(self, _mock_recommendation_artifact_s3):
        outcome = produce_artifact(
            _ok_veto_result(), bucket="test-bucket", promotion_intent="promote",
            recommended_params={"veto_confidence": 0.70},
        )
        assert outcome["written"] is True
        assert outcome["key"].startswith("config/predictor_params/recommendations/")
        assert outcome["key"].endswith("/from_veto_analysis.json")
        s3 = _mock_recommendation_artifact_s3.client.return_value
        body = json.loads(s3.put_object.call_args.kwargs["Body"])
        assert body["promotion_intent"] == "promote"
        assert body["recommendation_kind"] == "full_replace"


class TestApplyCutoverGate:
    """When ``optimizer.assembler.is_cutover_enabled()`` returns True, the
    legacy live-key write path is skipped — the assembler is the sole writer
    of ``config/predictor_params.json``."""

    @patch("analysis.veto_analysis.boto3")
    def test_cutover_enabled_skips_legacy_live_write(self, mock_boto3, _mock_recommendation_artifact_s3):
        _init_config()
        set_cutover_enabled(True)
        legacy_s3 = MagicMock()
        legacy_s3.get_object.side_effect = Exception("NoSuchKey")
        mock_boto3.client.return_value = legacy_s3

        outcome = apply(_ok_veto_result(), bucket="test-bucket")

        assert outcome["applied"] is False
        assert "cutover_mode" in outcome["reason"]
        assert legacy_s3.put_object.call_args_list == []
        artifact_s3 = _mock_recommendation_artifact_s3.client.return_value
        artifact_keys = [c.kwargs["Key"] for c in artifact_s3.put_object.call_args_list]
        assert any(k.endswith("/from_veto_analysis.json") for k in artifact_keys)

    @patch("analysis.veto_analysis.boto3")
    def test_cutover_disabled_keeps_legacy_write(self, mock_boto3, _mock_recommendation_artifact_s3):
        _init_config()
        legacy_s3 = MagicMock()
        legacy_s3.get_object.side_effect = Exception("NoSuchKey")
        mock_boto3.client.return_value = legacy_s3

        with patch("optimizer.rollback.save_previous"):
            outcome = apply(_ok_veto_result(), bucket="test-bucket")

        assert outcome["applied"] is True
        legacy_keys = [c.kwargs["Key"] for c in legacy_s3.put_object.call_args_list]
        assert S3_PARAMS_KEY in legacy_keys

    @patch("analysis.veto_analysis.boto3")
    def test_min_threshold_change_block_still_produces_skip_artifact(self, mock_boto3, _mock_recommendation_artifact_s3):
        """Audit completeness: a blocked recommendation still writes a
        recommendation artifact (promotion_intent=skip)."""
        _init_config()
        legacy_s3 = MagicMock()
        legacy_s3.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=json.dumps({"veto_confidence": 0.70}).encode())),
        }
        mock_boto3.client.return_value = legacy_s3

        outcome = apply(_ok_veto_result(threshold=0.70), bucket="test-bucket")

        assert outcome["applied"] is False
        assert outcome["blocked_by"] == ["min_threshold_change"]
        artifact_s3 = _mock_recommendation_artifact_s3.client.return_value
        body = json.loads(artifact_s3.put_object.call_args.kwargs["Body"])
        assert body["promotion_intent"] == "skip"
