"""Unit tests for optimizer.weight_optimizer — guardrail logic, no S3 calls."""
import pytest
from unittest.mock import patch, MagicMock

import pandas as pd

from optimizer.weight_optimizer import (
    apply_weights,
    compute_weights,
    init_config,
    load_with_subscores,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_df(n: int = 100, quant_corr: float = 0.1, qual_corr: float = 0.2):
    """Create a synthetic score_performance DataFrame with sub-scores.

    Generates data where quant_score and qual_score have approximate
    correlations with beat_spy_21d and beat_spy_5d outcomes.
    """
    import random
    random.seed(42)

    rows = []
    base_date = "2026-01-"
    for i in range(n):
        day = (i % 28) + 1
        score_date = f"{base_date}{day:02d}"
        quant = random.uniform(30, 90)
        qual = random.uniform(30, 90)
        # Higher sub-scores → higher probability of beating SPY
        p_beat = 0.3 + quant_corr * (quant / 100) + qual_corr * (qual / 100)
        beat_10d = 1 if random.random() < min(p_beat, 1.0) else 0
        beat_30d = 1 if random.random() < min(p_beat, 1.0) else 0
        rows.append({
            "symbol": f"STOCK{i % 20}",
            "score_date": score_date,
            "quant_score": quant,
            "qual_score": qual,
            "beat_spy_21d": beat_10d,
            "beat_spy_5d": beat_30d,
        })
    return pd.DataFrame(rows)


def _init_default_config():
    """Initialize the module config with default values."""
    init_config({
        "weight_optimizer": {
            "default_weights": {"quant": 0.50, "qual": 0.50},
            "max_single_change": 0.10,
            "min_meaningful_change": 0.03,
            "blend_factor": 0.20,
            "confidence_low": 100,
            "confidence_medium": 300,
            "horizon_blend": {"beat_spy_21d": 0.50, "beat_spy_5d": 0.50},
            "blend_factor_min": 0.20,
            "blend_factor_max": 0.50,
            "blend_ramp_samples": 500,
        }
    })


# ═══════════════════════════════════════════════════════════════════════════════
# compute_weights — guardrail and normalization logic
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeWeights:

    def setup_method(self):
        _init_default_config()

    def test_weights_normalize_to_one(self):
        """Suggested weights should always sum to 1.0."""
        df = _make_df(n=200)
        result = compute_weights(df, current_weights={"quant": 0.50, "qual": 0.50})
        assert result["status"] == "ok"
        total = sum(result["suggested_weights"].values())
        assert abs(total - 1.0) < 0.01

    def test_insufficient_data_returns_early(self):
        """Fewer than min_samples rows → status=insufficient_data."""
        df = _make_df(n=10)
        result = compute_weights(df, min_samples=30)
        assert result["status"] == "insufficient_data"

    def test_no_subscores_returns_status(self):
        """DataFrame without sub-score columns → status=no_subscores."""
        df = _make_df(n=100)
        df = df.drop(columns=["quant_score", "qual_score"])
        result = compute_weights(df)
        assert result["status"] == "no_subscores"

    def test_oos_degradation_computed(self):
        """Result should include oos_passed and oos_degradation fields."""
        df = _make_df(n=200)
        result = compute_weights(df)
        assert result["status"] == "ok"
        assert "oos_passed" in result
        assert "oos_degradation" in result
        assert isinstance(result["oos_degradation"], float)

    def test_changes_dict_present(self):
        """Result should include changes dict showing delta from current."""
        df = _make_df(n=200)
        result = compute_weights(df, current_weights={"quant": 0.50, "qual": 0.50})
        assert "changes" in result
        assert "quant" in result["changes"]
        assert "qual" in result["changes"]


# ═══════════════════════════════════════════════════════════════════════════════
# apply_weights — guardrail enforcement
# ═══════════════════════════════════════════════════════════════════════════════


class TestApplyWeights:

    def setup_method(self):
        _init_default_config()

    def test_max_single_weight_change_enforced(self):
        """If any single weight changes > 10%, apply should reject."""
        result = {
            "status": "ok",
            "confidence": "high",
            "oos_passed": True,
            "suggested_weights": {"quant": 0.35, "qual": 0.65},
            "changes": {"quant": -0.15, "qual": 0.15},
            "n_samples": 500,
        }
        outcome = apply_weights(result, bucket="test-bucket")
        assert outcome["applied"] is False
        assert "exceeds" in outcome["reason"].lower() or "limit" in outcome["reason"].lower()

    def test_min_meaningful_change_gate(self):
        """If all changes < 3%, apply should skip as not worth updating."""
        result = {
            "status": "ok",
            "confidence": "high",
            "oos_passed": True,
            "suggested_weights": {"quant": 0.51, "qual": 0.49},
            "changes": {"quant": 0.01, "qual": -0.01},
            "n_samples": 500,
        }
        outcome = apply_weights(result, bucket="test-bucket")
        assert outcome["applied"] is False
        assert "not worth" in outcome["reason"].lower()

    def test_low_confidence_blocks_apply(self):
        """Low confidence (< 50 samples) → should not apply."""
        result = {
            "status": "ok",
            "confidence": "low",
            "oos_passed": True,
            "suggested_weights": {"quant": 0.45, "qual": 0.55},
            "changes": {"quant": -0.05, "qual": 0.05},
            "n_samples": 40,
        }
        outcome = apply_weights(result, bucket="test-bucket")
        assert outcome["applied"] is False
        assert "confidence" in outcome["reason"].lower()

    def test_oos_degradation_blocks_apply(self):
        """OOS validation failure → should not apply."""
        result = {
            "status": "ok",
            "confidence": "high",
            "oos_passed": False,
            "oos_degradation": 0.35,
            "suggested_weights": {"quant": 0.45, "qual": 0.55},
            "changes": {"quant": -0.05, "qual": 0.05},
            "n_samples": 500,
        }
        outcome = apply_weights(result, bucket="test-bucket")
        assert outcome["applied"] is False
        assert "OOS" in outcome["reason"] or "oos" in outcome["reason"].lower()

    def test_non_ok_status_blocks_apply(self):
        """Non-ok status → should not apply."""
        result = {"status": "insufficient_data"}
        outcome = apply_weights(result, bucket="test-bucket")
        assert outcome["applied"] is False

    @patch("optimizer.weight_optimizer.boto3")
    @patch("optimizer.weight_optimizer.save_previous", create=True)
    def test_valid_changes_applied_to_s3(self, mock_save, mock_boto3):
        """Changes within all guardrails → applied = True."""
        # Patch the import inside apply_weights
        import optimizer.weight_optimizer as wo
        with patch.dict("sys.modules", {"optimizer.rollback": MagicMock()}):
            mock_s3 = MagicMock()
            mock_boto3.client.return_value = mock_s3

            result = {
                "status": "ok",
                "confidence": "medium",
                "oos_passed": True,
                "suggested_weights": {"quant": 0.45, "qual": 0.55},
                "changes": {"quant": -0.05, "qual": 0.05},
                "n_samples": 200,
            }
            outcome = apply_weights(result, bucket="test-bucket")
            assert outcome["applied"] is True
            assert outcome["weights"] == {"quant": 0.45, "qual": 0.55}


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluator-revamp PR 6: skill-composite fit target + shadow mode
# ═══════════════════════════════════════════════════════════════════════════════


def _make_df_with_continuous_returns(n: int = 200, signal_strength: float = 0.5):
    """Synthetic score_performance with both binary beat_spy + continuous return cols.

    Higher quant_score → higher log_alpha_21d (signal). qual_score is noise.
    Used to verify the IC-on-continuous-returns fit target picks up
    quant > qual when only the continuous channel carries the signal.
    """
    import random
    random.seed(7)
    rows = []
    for i in range(n):
        day = (i % 28) + 1
        score_date = f"2026-01-{day:02d}"
        quant = random.uniform(40, 90)
        qual = random.uniform(40, 90)
        # Continuous return: scales with quant. qual is pure noise.
        ret_10d = (quant - 65) / 100.0 * signal_strength + random.gauss(0, 0.02)
        ret_30d = ret_10d * 1.5
        rows.append({
            "symbol": f"S{i % 20}",
            "score_date": score_date,
            "quant_score": quant,
            "qual_score": qual,
            "log_alpha_21d": ret_10d,
            "return_5d": ret_30d,
            "beat_spy_21d": int(ret_10d > 0),
            "beat_spy_5d": int(ret_30d > 0),
        })
    return pd.DataFrame(rows)


class TestSkillCompositeFitTarget:

    def test_default_off_uses_legacy_path(self):
        _init_default_config()
        df = _make_df_with_continuous_returns(n=300)
        result = compute_weights(df, current_weights={"quant": 0.50, "qual": 0.50})
        assert result["status"] == "ok"
        # Default fit_target stamps as legacy.
        assert result["fit_target"] == "beat_spy_pearson"

    def test_flag_on_switches_to_ic_path(self):
        init_config({
            "weight_optimizer": {
                "default_weights": {"quant": 0.50, "qual": 0.50},
                "max_single_change": 0.10,
                "min_meaningful_change": 0.03,
                "blend_factor": 0.20,
                "confidence_low": 100,
                "confidence_medium": 300,
                "horizon_blend": {"beat_spy_21d": 0.50, "beat_spy_5d": 0.50},
                "blend_factor_min": 0.20,
                "blend_factor_max": 0.50,
                "blend_ramp_samples": 500,
                "use_skill_composite_target": True,
            }
        })
        df = _make_df_with_continuous_returns(n=300, signal_strength=0.8)
        result = compute_weights(df, current_weights={"quant": 0.50, "qual": 0.50})
        assert result["status"] == "ok"
        assert result["fit_target"] == "skill_composite_ic"
        # IC path should pick up that quant carries the signal → suggested
        # quant weight > qual weight (or at least correlations[quant] >
        # correlations[qual] on a return horizon).
        c_quant = result["correlations"]["quant"]["beat_spy_21d"] or 0.0
        c_qual = result["correlations"]["qual"]["beat_spy_21d"] or 0.0
        assert c_quant > c_qual

    def test_skill_composite_continuous_returns_when_binary_signal_drowns_in_noise(self):
        """Continuous-return IC sees signal strength that binary
        beat_spy obscures via the threshold transformation."""
        # Tight return distribution around zero → most "beats" are noise.
        df = _make_df_with_continuous_returns(n=400, signal_strength=0.3)

        init_config({
            "weight_optimizer": {
                "default_weights": {"quant": 0.50, "qual": 0.50},
                "max_single_change": 0.10,
                "min_meaningful_change": 0.03,
                "blend_factor": 0.20,
                "confidence_low": 100,
                "confidence_medium": 300,
                "horizon_blend": {"beat_spy_21d": 0.50, "beat_spy_5d": 0.50},
                "blend_factor_min": 0.20,
                "blend_factor_max": 0.50,
                "blend_ramp_samples": 500,
                "use_skill_composite_target": True,
            }
        })
        ic_result = compute_weights(df, current_weights={"quant": 0.50, "qual": 0.50})

        _init_default_config()  # reset to legacy
        legacy_result = compute_weights(df, current_weights={"quant": 0.50, "qual": 0.50})

        ic_quant = ic_result["correlations"]["quant"]["beat_spy_21d"] or 0.0
        legacy_quant = legacy_result["correlations"]["quant"]["beat_spy_21d"] or 0.0
        # IC should detect the quant signal more strongly than Pearson-on-binary.
        assert abs(ic_quant) >= abs(legacy_quant)


class TestShadowMode:

    def _enable_skill_composite(self, enforce: bool):
        init_config({
            "weight_optimizer": {
                "default_weights": {"quant": 0.50, "qual": 0.50},
                "max_single_change": 0.10,
                "min_meaningful_change": 0.03,
                "blend_factor": 0.20,
                "confidence_low": 100,
                "confidence_medium": 300,
                "horizon_blend": {"beat_spy_21d": 0.50, "beat_spy_5d": 0.50},
                "blend_factor_min": 0.20,
                "blend_factor_max": 0.50,
                "blend_ramp_samples": 500,
                "use_skill_composite_target": True,
                "enforce_skill_composite": enforce,
            }
        })

    @patch("optimizer.weight_optimizer.boto3")
    def test_skill_composite_without_enforce_writes_to_shadow_only(self, mock_boto3):
        """When skill_composite is on but enforce is off → shadow path,
        production scoring_weights.json is not overwritten."""
        self._enable_skill_composite(enforce=False)
        with patch.dict("sys.modules", {"optimizer.rollback": MagicMock()}):
            mock_s3 = MagicMock()
            mock_boto3.client.return_value = mock_s3

            result = {
                "status": "ok",
                "confidence": "medium",
                "oos_passed": True,
                "suggested_weights": {"quant": 0.45, "qual": 0.55},
                "changes": {"quant": -0.05, "qual": 0.05},
                "n_samples": 200,
                "fit_target": "skill_composite_ic",
            }
            outcome = apply_weights(result, bucket="test-bucket")
            assert outcome["applied"] is False
            assert "shadow mode" in outcome["reason"].lower()
            # Live config key should NOT be written.
            keys_written = [
                call.kwargs.get("Key") for call in mock_s3.put_object.call_args_list
            ]
            assert "config/scoring_weights.json" not in keys_written
            # Shadow archive WAS written.
            assert any(
                k and k.startswith("config/scoring_weights_shadow_history/")
                for k in keys_written
            )

    @patch("optimizer.weight_optimizer.boto3")
    def test_skill_composite_with_enforce_applies_to_production(self, mock_boto3):
        """When skill_composite is on AND enforce is on → live path."""
        self._enable_skill_composite(enforce=True)
        with patch.dict("sys.modules", {"optimizer.rollback": MagicMock()}):
            mock_s3 = MagicMock()
            mock_boto3.client.return_value = mock_s3

            result = {
                "status": "ok",
                "confidence": "medium",
                "oos_passed": True,
                "suggested_weights": {"quant": 0.45, "qual": 0.55},
                "changes": {"quant": -0.05, "qual": 0.05},
                "n_samples": 200,
                "fit_target": "skill_composite_ic",
            }
            outcome = apply_weights(result, bucket="test-bucket")
            assert outcome["applied"] is True
            keys_written = [
                call.kwargs.get("Key") for call in mock_s3.put_object.call_args_list
            ]
            assert "config/scoring_weights.json" in keys_written

    @patch("optimizer.weight_optimizer.boto3")
    def test_legacy_path_unaffected_by_enforce_flag(self, mock_boto3):
        """When skill_composite is OFF, the enforce flag is irrelevant —
        legacy callers see no behavior change."""
        _init_default_config()  # use_skill_composite_target=False
        with patch.dict("sys.modules", {"optimizer.rollback": MagicMock()}):
            mock_s3 = MagicMock()
            mock_boto3.client.return_value = mock_s3

            result = {
                "status": "ok",
                "confidence": "medium",
                "oos_passed": True,
                "suggested_weights": {"quant": 0.45, "qual": 0.55},
                "changes": {"quant": -0.05, "qual": 0.05},
                "n_samples": 200,
                "fit_target": "beat_spy_pearson",
            }
            outcome = apply_weights(result, bucket="test-bucket")
            assert outcome["applied"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# config#1426 Phase 4 — significance ENFORCE wiring (default OFF)
# ═══════════════════════════════════════════════════════════════════════════════


def _init_enforce_config(enforce: bool):
    """Init weight_optimizer config with the enforce_significance flag set."""
    init_config({
        "weight_optimizer": {
            "default_weights": {"quant": 0.50, "qual": 0.50},
            "max_single_change": 0.10,
            "min_meaningful_change": 0.03,
            "enforce_significance": enforce,
        }
    })


def _enforce_result(verdict: dict) -> dict:
    """A result that clears every LIVE guardrail (would apply without enforce)."""
    return {
        "status": "ok",
        "confidence": "medium",
        "oos_passed": True,
        "suggested_weights": {"quant": 0.45, "qual": 0.55},
        "changes": {"quant": -0.05, "qual": 0.05},
        "n_samples": 200,
        "fit_target": "beat_spy_pearson",
        "significance_observe": verdict,
    }


def _wverdict(would_block: bool, canonical: dict, five_d: dict | None = None) -> dict:
    """Build a weight observe verdict with a single 'quant' sub-score carrying
    per-horizon return_5d + canonical log_alpha_21d IC verdicts."""
    horizons = {"log_alpha_21d": canonical}
    if five_d is not None:
        horizons["return_5d"] = five_d
    return {
        "gate": "weight_optimizer", "would_block": would_block,
        "significant": not would_block, "enforced": True,
        "detail": {"per_subscore": {"quant": {"significant": not would_block,
                                              "horizons": horizons}},
                   "n_test": 100},
    }


# Nothing significant on any horizon ⇒ would_block ⇒ BLOCK.
_VERDICT_WOULD_BLOCK = _wverdict(
    True,
    canonical={"significant": False, "ic": 0.0},
    five_d={"significant": False, "ic": 0.01},
)
# Significant + POSITIVE canonical IC ≥ 0.03 ⇒ DEFENDED ⇒ proceeds.
_VERDICT_SIG_POSITIVE_CANONICAL = _wverdict(
    False,
    canonical={"significant": True, "ic": 0.08},
    five_d={"significant": False, "ic": 0.02},
)
# The re-replay bug: significant driven by a large NEGATIVE 5d IC; canonical null.
# An absolute |IC| floor would wrongly defend it — signed+canonical floor BLOCKS.
_VERDICT_SIG_NEGATIVE_5D = _wverdict(
    False,
    canonical={"significant": False, "ic": 0.0},
    five_d={"significant": True, "ic": -0.254},
)
# Significant ONLY on return_5d (positive), canonical null ⇒ BLOCK (wrong horizon).
_VERDICT_SIG_ONLY_5D = _wverdict(
    False,
    canonical={"significant": False, "ic": 0.0},
    five_d={"significant": True, "ic": 0.12},
)
# Significant on canonical but a TRIVIAL positive IC (< 0.03) ⇒ BLOCK (below floor).
_VERDICT_SIG_TRIVIAL_CANONICAL = _wverdict(
    False,
    canonical={"significant": True, "ic": 0.01},
    five_d={"significant": False, "ic": 0.0},
)


class TestApplyWeightsSignificanceEnforce:

    @patch("optimizer.weight_optimizer.boto3")
    @patch("optimizer.weight_optimizer.save_previous", create=True)
    def test_default_off_applies_even_when_would_block(self, _mock_save, mock_boto3):
        """CRITICAL non-enforcement guarantee: with enforce_significance unset
        (default False), a promotion the live gate allows STILL applies even
        when the significance verdict is would_block. This PR ships the
        capability only — the default path is byte-for-byte unchanged."""
        _init_default_config()  # no enforce_significance key → defaults False
        with patch.dict("sys.modules", {"optimizer.rollback": MagicMock()}):
            mock_boto3.client.return_value = MagicMock()
            outcome = apply_weights(_enforce_result(_VERDICT_WOULD_BLOCK), bucket="b")
        assert outcome["applied"] is True

    @patch("optimizer.weight_optimizer.boto3")
    @patch("optimizer.weight_optimizer.save_previous", create=True)
    def test_enforce_blocks_insignificant(self, _mock_save, mock_boto3):
        _init_enforce_config(True)
        with patch.dict("sys.modules", {"optimizer.rollback": MagicMock()}):
            mock_s3 = MagicMock()
            mock_boto3.client.return_value = mock_s3
            outcome = apply_weights(_enforce_result(_VERDICT_WOULD_BLOCK), bucket="b")
        assert outcome["applied"] is False
        assert "significance enforce" in outcome["reason"]
        assert outcome["observe_verdict"] is _VERDICT_WOULD_BLOCK
        # No live write happened.
        assert mock_s3.put_object.call_args_list == []

    @patch("optimizer.weight_optimizer.boto3")
    @patch("optimizer.weight_optimizer.save_previous", create=True)
    def test_enforce_allows_significant_positive_canonical(self, _mock_save, mock_boto3):
        """Significant + POSITIVE canonical log_alpha_21d IC ≥ 0.03 ⇒ proceeds."""
        _init_enforce_config(True)
        with patch.dict("sys.modules", {"optimizer.rollback": MagicMock()}):
            mock_boto3.client.return_value = MagicMock()
            outcome = apply_weights(
                _enforce_result(_VERDICT_SIG_POSITIVE_CANONICAL), bucket="b")
        assert outcome["applied"] is True

    @patch("optimizer.weight_optimizer.boto3")
    @patch("optimizer.weight_optimizer.save_previous", create=True)
    def test_enforce_blocks_significant_negative_5d_ic(self, _mock_save, mock_boto3):
        """The re-replay bug: significance driven by a large NEGATIVE 5d IC with a
        null canonical horizon MUST block — a signed canonical floor refuses it
        where an absolute |IC| floor would have wrongly 'defended' it."""
        _init_enforce_config(True)
        with patch.dict("sys.modules", {"optimizer.rollback": MagicMock()}):
            mock_s3 = MagicMock()
            mock_boto3.client.return_value = mock_s3
            outcome = apply_weights(
                _enforce_result(_VERDICT_SIG_NEGATIVE_5D), bucket="b")
        assert outcome["applied"] is False
        assert "significance enforce" in outcome["reason"]
        assert mock_s3.put_object.call_args_list == []

    @patch("optimizer.weight_optimizer.boto3")
    @patch("optimizer.weight_optimizer.save_previous", create=True)
    def test_enforce_blocks_significant_only_on_5d_horizon(self, _mock_save, mock_boto3):
        """Significant positive IC but only on the legacy return_5d horizon (null
        canonical) ⇒ block — the floor is canonical-horizon-only."""
        _init_enforce_config(True)
        with patch.dict("sys.modules", {"optimizer.rollback": MagicMock()}):
            mock_boto3.client.return_value = MagicMock()
            outcome = apply_weights(_enforce_result(_VERDICT_SIG_ONLY_5D), bucket="b")
        assert outcome["applied"] is False

    @patch("optimizer.weight_optimizer.boto3")
    @patch("optimizer.weight_optimizer.save_previous", create=True)
    def test_enforce_blocks_trivial_positive_canonical(self, _mock_save, mock_boto3):
        """Significant on canonical but a trivial positive IC (< 0.03) ⇒ block."""
        _init_enforce_config(True)
        with patch.dict("sys.modules", {"optimizer.rollback": MagicMock()}):
            mock_boto3.client.return_value = MagicMock()
            outcome = apply_weights(
                _enforce_result(_VERDICT_SIG_TRIVIAL_CANONICAL), bucket="b")
        assert outcome["applied"] is False

    @patch("optimizer.weight_optimizer.boto3")
    @patch("optimizer.weight_optimizer.save_previous", create=True)
    def test_enforce_missing_verdict_blocks_conservatively(self, _mock_save, mock_boto3):
        _init_enforce_config(True)
        with patch.dict("sys.modules", {"optimizer.rollback": MagicMock()}):
            mock_boto3.client.return_value = MagicMock()
            outcome = apply_weights(_enforce_result(None), bucket="b")
        assert outcome["applied"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# load_with_subscores — canonical (score_performance) vs S3 backfill paths.
# Regression for the 2026-05-09 P0: research.db migration #12 added
# quant_score/qual_score to score_performance, so df arrives carrying those
# columns. The pre-fix merge collided with the S3 sub_df and produced
# quant_score_x / quant_score_y, then crashed on merged[["quant_score","qual_score"]].
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoadWithSubscoresCanonicalSource:

    @patch("optimizer.weight_optimizer.boto3")
    def test_canonical_fully_populated_skips_s3(self, mock_boto3):
        """Post-migration df with no NULL sub-scores: no S3 round-trip,
        returned columns are quant_score/qual_score (not _x/_y)."""
        df = pd.DataFrame([
            {"symbol": "AAPL", "score_date": "2026-05-01",
             "quant_score": 72.0, "qual_score": 65.0, "beat_spy_21d": 1},
            {"symbol": "MSFT", "score_date": "2026-05-01",
             "quant_score": 80.0, "qual_score": 70.0, "beat_spy_21d": 0},
        ])
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        out = load_with_subscores(df, bucket="test-bucket")

        assert "quant_score" in out.columns
        assert "qual_score" in out.columns
        assert "quant_score_x" not in out.columns
        assert "quant_score_y" not in out.columns
        assert out.loc[out["symbol"] == "AAPL", "quant_score"].iloc[0] == 72.0
        mock_s3.get_object.assert_not_called()

    @patch("optimizer.weight_optimizer.boto3")
    def test_canonical_partial_null_backfilled_from_s3(self, mock_boto3):
        """Mix of canonical-populated + canonical-NULL rows: canonical values
        win; only NULL rows pick up S3 values. No _x/_y suffix leakage."""
        import json

        df = pd.DataFrame([
            {"symbol": "AAPL", "score_date": "2026-05-01",
             "quant_score": 72.0, "qual_score": 65.0, "beat_spy_21d": 1},
            {"symbol": "MSFT", "score_date": "2026-05-01",
             "quant_score": None, "qual_score": None, "beat_spy_21d": 0},
        ])
        signals_json = {
            "signals": {
                "AAPL": {"sub_scores": {"quant": 11.0, "qual": 12.0}},
                "MSFT": {"sub_scores": {"quant": 88.0, "qual": 77.0}},
            }
        }
        body = MagicMock()
        body.read.return_value = json.dumps(signals_json).encode()
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": body}
        mock_boto3.client.return_value = mock_s3

        out = load_with_subscores(df, bucket="test-bucket")

        assert "quant_score_x" not in out.columns
        assert "quant_score_s3" not in out.columns
        aapl = out[out["symbol"] == "AAPL"].iloc[0]
        msft = out[out["symbol"] == "MSFT"].iloc[0]
        # Canonical row preserved.
        assert aapl["quant_score"] == 72.0
        assert aapl["qual_score"] == 65.0
        # NULL row backfilled.
        assert msft["quant_score"] == 88.0
        assert msft["qual_score"] == 77.0

    @patch("optimizer.weight_optimizer.boto3")
    def test_legacy_df_without_canonical_columns_still_merges(self, mock_boto3):
        """Pre-migration DataFrames (no quant_score/qual_score columns) take
        the original merge path: S3 sub_df is joined directly."""
        import json

        df = pd.DataFrame([
            {"symbol": "AAPL", "score_date": "2026-05-01", "beat_spy_21d": 1},
        ])
        signals_json = {
            "signals": {
                "AAPL": {"sub_scores": {"quant": 55.0, "qual": 50.0}},
            }
        }
        body = MagicMock()
        body.read.return_value = json.dumps(signals_json).encode()
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": body}
        mock_boto3.client.return_value = mock_s3

        out = load_with_subscores(df, bucket="test-bucket")

        assert out["quant_score"].iloc[0] == 55.0
        assert out["qual_score"].iloc[0] == 50.0
        assert "quant_score_x" not in out.columns
        assert "quant_score_y" not in out.columns


# ═══════════════════════════════════════════════════════════════════════════════
# Round-trip: apply_weights() write → evaluate._read_current_weights() read
# (config#1679 regression)
# ═══════════════════════════════════════════════════════════════════════════════
#
# apply_weights() persists s3://{bucket}/config/scoring_weights.json under
# SUB_SCORES ("quant"/"qual") — has done so since the 2026-03-29 rename
# (commit 92c5067, "Rename sub_scores from news/research to quant/qual").
# evaluate._read_current_weights() had never been updated to match and was
# still reading the pre-rename "news"/"research" keys, so it could never find
# the persisted weights and silently fell back to default_weights. That wrong
# baseline then fed compute_weights()'s delta/churn computation every cycle.
# This test drives the real write path (apply_weights) and the real read path
# (evaluate._read_current_weights) against the same fake S3 object store and
# asserts the read recovers exactly what was applied — not the fallback.


class TestApplyReadRoundTrip:

    def setup_method(self):
        _init_default_config()

    @patch("optimizer.weight_optimizer.boto3")
    @patch("optimizer.weight_optimizer.save_previous", create=True)
    def test_read_current_weights_matches_last_applied(self, mock_save, mock_wo_boto3):
        """apply_weights() writes quant/qual; _read_current_weights() must
        read back those exact values, not the default_weights fallback."""
        import json
        from evaluate import _read_current_weights

        # Fake S3 object store shared across the write (apply_weights) and
        # read (_read_current_weights) sides of the round trip.
        store: dict[str, bytes] = {}

        def _put_object(Bucket, Key, Body, **kwargs):
            store[Key] = Body.encode() if isinstance(Body, str) else Body

        mock_s3 = MagicMock()
        mock_s3.put_object.side_effect = _put_object
        mock_wo_boto3.client.return_value = mock_s3

        result = {
            "status": "ok",
            "confidence": "high",
            "oos_passed": True,
            "suggested_weights": {"quant": 0.58, "qual": 0.42},
            "changes": {"quant": 0.08, "qual": -0.08},
            "n_samples": 500,
        }
        with patch.dict("sys.modules", {"optimizer.rollback": MagicMock()}):
            outcome = apply_weights(result, bucket="test-bucket")
        assert outcome["applied"] is True

        written_body = store["config/scoring_weights.json"]
        payload = json.loads(written_body)
        # Pins the producer side of the contract too: quant/qual, not
        # news/research.
        assert payload["quant"] == 0.58
        assert payload["qual"] == 0.42
        assert "news" not in payload and "research" not in payload

        # Read it back through evaluate._read_current_weights against the
        # same fake object store.
        def _get_object(Bucket, Key):
            body = MagicMock()
            body.read.return_value = store[Key]
            return {"Body": body}

        read_s3 = MagicMock()
        read_s3.get_object.side_effect = _get_object
        with patch("evaluate.boto3") as mock_eval_boto3:
            mock_eval_boto3.client.return_value = read_s3
            current = _read_current_weights({"signals_bucket": "test-bucket"})

        assert current == {"quant": 0.58, "qual": 0.42}
        # Must not have silently fallen through to the 0.50/0.50 default.
        import optimizer.weight_optimizer as wo
        assert current != wo._DEFAULT_WEIGHTS

    def test_read_current_weights_migrates_legacy_news_research_keys(self):
        """A scoring_weights.json object persisted before the 2026-03-29
        news/research → quant/qual rename (and never since overwritten)
        must still be read back correctly, not silently dropped to the
        default_weights fallback."""
        import json
        from evaluate import _read_current_weights

        legacy_payload = json.dumps({
            "news": 0.55, "research": 0.45, "updated_at": "2026-02-01",
        }).encode()

        def _get_object(Bucket, Key):
            body = MagicMock()
            body.read.return_value = legacy_payload
            return {"Body": body}

        read_s3 = MagicMock()
        read_s3.get_object.side_effect = _get_object
        with patch("evaluate.boto3") as mock_eval_boto3:
            mock_eval_boto3.client.return_value = read_s3
            current = _read_current_weights({"signals_bucket": "test-bucket"})

        assert current == {"quant": 0.55, "qual": 0.45}
