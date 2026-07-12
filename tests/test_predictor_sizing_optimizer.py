"""Unit tests for optimizer.predictor_sizing_optimizer — apply() + produce_artifact()
+ dual-write guarantee. All S3 calls mocked.

Part of the optimizer-artifact-assembler arc (PR 2).
"""
import json
import sqlite3
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from optimizer.assembler import set_cutover_enabled
from optimizer.predictor_sizing_optimizer import (
    S3_PARAMS_KEY,
    _build_overlay_params,
    analyze,
    apply,
    produce_artifact,
)
from pipeline_common import ACTIVE_HORIZON_DAYS


@pytest.fixture(autouse=True)
def _reset_cutover_flag():
    """Reset assembler cutover flag around each test."""
    set_cutover_enabled(False)
    yield
    set_cutover_enabled(False)


def _set_module_cfg(extra: dict | None = None):
    """Set the module-level _cfg used by _build_overlay_params for blend_factor."""
    from optimizer import predictor_sizing_optimizer as mod
    mod._cfg = {"blend_factor": 0.3}
    if extra:
        mod._cfg.update(extra)


def _make_db(rows: list[tuple]) -> str:
    """rows: (prediction_date, symbol, p_up, actual_log_alpha)."""
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    conn = sqlite3.connect(f.name)
    conn.execute(
        "CREATE TABLE predictor_outcomes (prediction_date TEXT, symbol TEXT, "
        "p_up REAL, actual_log_alpha REAL, actual_5d_return REAL, horizon_days INTEGER)"
    )
    for pred_date, sym, p_up, alpha in rows:
        conn.execute(
            "INSERT INTO predictor_outcomes "
            "(prediction_date, symbol, p_up, actual_log_alpha, actual_5d_return, horizon_days) "
            "VALUES (?,?,?,?,?,?)",
            (pred_date, sym, p_up, alpha, None, ACTIVE_HORIZON_DAYS),
        )
    conn.commit()
    conn.close()
    return f.name


def _random_panel_rows(n: int, seed: int) -> list[tuple]:
    """A single-date random panel: n symbols, uncorrelated p_up / alpha.

    One prediction_date so pandas' groupby-by-week collapses to a single
    week's IC (matches ``overall_ic`` exactly — no cross-week mixing) —
    keeps the cross-check a direct apples-to-apples read of the same
    (p_up, canonical_actual) vectors analyze() feeds into pandas' Spearman.
    """
    rng = np.random.default_rng(seed)
    p_up = rng.uniform(0.0, 1.0, size=n)
    alpha = rng.normal(0.0, 0.02, size=n)
    return [
        ("2026-06-01", f"T{i}", float(p_up[i]), float(alpha[i]))
        for i in range(n)
    ]


# ── spearmanr cross-check (config#1958 deliverable 5e) ──────────────────────


class TestSpearmanCrossCheck:
    """analyze()'s ``overall_rank_ic`` is pandas' ``Series.corr(method="spearman")``
    — a different code path from ``scipy.stats.spearmanr`` (config#1958's
    anchor for the predictor's rank-correlation cross-check). This pins that
    the two independent implementations agree on random panels, so a future
    swap of either implementation can't silently drift the promotion-gating
    IC without a test noticing.
    """

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_overall_rank_ic_matches_scipy_spearmanr_on_random_panels(self, seed):
        from scipy.stats import spearmanr

        rows = _random_panel_rows(n=60, seed=seed)
        db = _make_db(rows)
        result = analyze(db)
        assert result["status"] == "ok"

        p_up = np.array([r[2] for r in rows])
        alpha = np.array([r[3] for r in rows])
        expected_rho, _p = spearmanr(p_up, alpha)

        assert result["overall_rank_ic"] == pytest.approx(float(expected_rho), abs=1e-4)

    def test_matches_scipy_with_tied_ranks(self):
        # Ties in p_up (repeated values) exercise pandas'/scipy's tie-handling
        # (average-rank) — both must still agree.
        from scipy.stats import spearmanr

        rng = np.random.default_rng(42)
        n = 50
        p_up = rng.choice([0.1, 0.2, 0.3, 0.4, 0.5], size=n)  # heavy ties
        alpha = rng.normal(0.0, 0.02, size=n)
        rows = [("2026-06-01", f"T{i}", float(p_up[i]), float(alpha[i])) for i in range(n)]
        db = _make_db(rows)
        result = analyze(db)
        assert result["status"] == "ok"

        expected_rho, _p = spearmanr(p_up, alpha)
        assert result["overall_rank_ic"] == pytest.approx(float(expected_rho), abs=1e-4)

    def test_matches_scipy_on_strongly_correlated_panel(self):
        # A non-trivial (non-near-zero) rho, so the cross-check also covers
        # the "clearly enable" regime, not just noise-around-zero panels.
        from scipy.stats import spearmanr

        rng = np.random.default_rng(7)
        n = 80
        p_up = rng.uniform(0.0, 1.0, size=n)
        alpha = 0.05 * p_up + rng.normal(0.0, 0.005, size=n)  # strong positive rank relation
        rows = [("2026-06-01", f"T{i}", float(p_up[i]), float(alpha[i])) for i in range(n)]
        db = _make_db(rows)
        result = analyze(db)
        assert result["status"] == "ok"
        assert result["overall_rank_ic"] > 0.5  # sanity: genuinely correlated

        expected_rho, _p = spearmanr(p_up, alpha)
        assert result["overall_rank_ic"] == pytest.approx(float(expected_rho), abs=1e-4)


# ── _build_overlay_params ────────────────────────────────────────────────────


class TestBuildOverlayParams:

    def test_emits_4_overlay_fields(self):
        _set_module_cfg()
        result = {
            "status": "ok",
            "recommendation": "enable",
            "overall_rank_ic": 0.10,
        }
        params, keys = _build_overlay_params(result)
        assert set(keys) == {
            "use_p_up_sizing",
            "p_up_sizing_blend",
            "p_up_sizing_updated_at",
            "p_up_sizing_ic",
        }
        assert params["use_p_up_sizing"] is True
        assert params["p_up_sizing_blend"] == 0.3
        assert params["p_up_sizing_ic"] == 0.10

    def test_blend_factor_from_config(self):
        _set_module_cfg({"blend_factor": 0.5})
        result = {"status": "ok", "recommendation": "enable", "overall_rank_ic": 0.08}
        params, _ = _build_overlay_params(result)
        assert params["p_up_sizing_blend"] == 0.5


# ── produce_artifact ─────────────────────────────────────────────────────────


class TestProduceArtifact:

    @patch("optimizer.recommendation_artifact.boto3")
    def test_status_ok_recommendation_enable_promotes(self, mock_boto3):
        _set_module_cfg()
        s3 = MagicMock()
        mock_boto3.client.return_value = s3
        result = {
            "status": "ok",
            "recommendation": "enable",
            "overall_rank_ic": 0.10,
            "recent_mean_ic": 0.08,
            "n_samples": 60,
        }
        outcome = produce_artifact(result, bucket="test-bucket")
        assert outcome["written"] is True
        assert outcome["key"].endswith("/from_predictor_sizing_optimizer.json")
        body = json.loads(s3.put_object.call_args.kwargs["Body"])
        assert body["promotion_intent"] == "promote"
        assert body["recommendation_kind"] == "field_overlay"
        assert body["fit_target"] == "sizing_ic"
        assert body["recommended_params"]["use_p_up_sizing"] is True
        assert "use_p_up_sizing" in body["overlay_keys"]
        # diagnostic fields persisted
        assert body["diagnostic"]["overall_rank_ic"] == 0.10
        assert body["diagnostic"]["n_samples"] == 60

    @patch("optimizer.recommendation_artifact.boto3")
    def test_recommendation_keep_disabled_skips(self, mock_boto3):
        _set_module_cfg()
        s3 = MagicMock()
        mock_boto3.client.return_value = s3
        result = {
            "status": "ok",
            "recommendation": "keep_disabled",
            "overall_rank_ic": 0.02,  # below threshold
        }
        outcome = produce_artifact(result, bucket="test-bucket")
        assert outcome["written"] is True
        body = json.loads(s3.put_object.call_args.kwargs["Body"])
        # Intent is skip — gate didn't pass — but artifact still written for audit.
        assert body["promotion_intent"] == "skip"
        assert body["recommended_params"] == {}
        assert body["overlay_keys"] is None
        # Diagnostic still records what the optimizer found.
        assert body["diagnostic"]["overall_rank_ic"] == 0.02
        assert body["diagnostic"]["recommendation"] == "keep_disabled"

    @patch("optimizer.recommendation_artifact.boto3")
    def test_status_insufficient_data_skips(self, mock_boto3):
        _set_module_cfg()
        s3 = MagicMock()
        mock_boto3.client.return_value = s3
        result = {"status": "insufficient_data", "n_samples": 12}
        outcome = produce_artifact(result, bucket="test-bucket")
        assert outcome["written"] is True
        body = json.loads(s3.put_object.call_args.kwargs["Body"])
        assert body["promotion_intent"] == "skip"

    @patch("optimizer.recommendation_artifact.boto3")
    def test_swallows_s3_errors_non_fatal(self, mock_boto3):
        _set_module_cfg()
        s3 = MagicMock()
        s3.put_object.side_effect = Exception("S3 disconnected")
        mock_boto3.client.return_value = s3
        outcome = produce_artifact({
            "status": "ok",
            "recommendation": "enable",
            "overall_rank_ic": 0.10,
        }, bucket="test-bucket")
        assert outcome["written"] is False
        assert "S3 disconnected" in outcome["reason"]


# ── apply() dual-write contract ──────────────────────────────────────────────


class TestApplyDualWrite:

    @patch("optimizer.predictor_sizing_optimizer.boto3")
    @patch("optimizer.recommendation_artifact.boto3")
    def test_legacy_apply_path_also_writes_artifact(
        self, mock_artifact_boto3, mock_apply_boto3,
    ):
        _set_module_cfg()
        legacy_s3 = MagicMock()
        artifact_s3 = MagicMock()
        # Simulate empty current executor_params (NoSuchKey path).
        legacy_s3.get_object.side_effect = Exception("NoSuchKey")
        mock_apply_boto3.client.return_value = legacy_s3
        mock_artifact_boto3.client.return_value = artifact_s3

        result = {
            "status": "ok",
            "recommendation": "enable",
            "overall_rank_ic": 0.10,
        }
        outcome = apply(result, bucket="test-bucket")

        # Legacy live write happened.
        assert outcome["applied"] is True
        legacy_keys = [c.kwargs["Key"] for c in legacy_s3.put_object.call_args_list]
        assert S3_PARAMS_KEY in legacy_keys

        # Artifact write happened too.
        artifact_keys = [c.kwargs["Key"] for c in artifact_s3.put_object.call_args_list]
        assert any(
            k.endswith("/from_predictor_sizing_optimizer.json")
            for k in artifact_keys
        )

    @patch("optimizer.predictor_sizing_optimizer.boto3")
    @patch("optimizer.recommendation_artifact.boto3")
    def test_keep_disabled_still_writes_artifact(
        self, mock_artifact_boto3, mock_apply_boto3,
    ):
        # Even though apply() refuses to promote (recommendation=keep_disabled),
        # produce_artifact() still fires for audit. promotion_intent=skip.
        _set_module_cfg()
        legacy_s3 = MagicMock()
        artifact_s3 = MagicMock()
        mock_apply_boto3.client.return_value = legacy_s3
        mock_artifact_boto3.client.return_value = artifact_s3

        result = {
            "status": "ok",
            "recommendation": "keep_disabled",
            "overall_rank_ic": 0.02,
            "recent_mean_ic": 0.01,
        }
        outcome = apply(result, bucket="test-bucket")
        assert outcome["applied"] is False

        # Legacy NEVER wrote (apply returned early).
        legacy_writes = [c for c in legacy_s3.put_object.call_args_list]
        assert len(legacy_writes) == 0

        # Artifact STILL wrote.
        artifact_call = next(
            c for c in artifact_s3.put_object.call_args_list
            if c.kwargs["Key"].endswith("/from_predictor_sizing_optimizer.json")
        )
        body = json.loads(artifact_call.kwargs["Body"])
        assert body["promotion_intent"] == "skip"

    @patch("optimizer.predictor_sizing_optimizer.boto3")
    @patch("optimizer.recommendation_artifact.boto3")
    def test_already_enabled_idempotent_skip_still_writes_artifact(
        self, mock_artifact_boto3, mock_apply_boto3,
    ):
        _set_module_cfg()
        legacy_s3 = MagicMock()
        artifact_s3 = MagicMock()
        # Existing config already has use_p_up_sizing=True.
        legacy_s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps({
                "use_p_up_sizing": True,
                "atr_multiplier": 2.0,
            }).encode()),
        }
        mock_apply_boto3.client.return_value = legacy_s3
        mock_artifact_boto3.client.return_value = artifact_s3

        result = {
            "status": "ok",
            "recommendation": "enable",
            "overall_rank_ic": 0.10,
        }
        outcome = apply(result, bucket="test-bucket")
        assert outcome["applied"] is False
        assert "already enabled" in outcome["reason"]

        # Even on the idempotent-skip branch, artifact was still written
        # for this run (with promote intent — the optimizer's gate passed
        # even though apply ended up being a no-op).
        artifact_writes = [
            c for c in artifact_s3.put_object.call_args_list
            if c.kwargs["Key"].endswith("/from_predictor_sizing_optimizer.json")
        ]
        assert len(artifact_writes) == 1


class TestApplySignificanceEnforce:
    """config#1426 Phase 4 — significance ENFORCE wiring (default OFF)."""

    def _enable_result(self, verdict):
        return {
            "status": "ok",
            "recommendation": "enable",
            "overall_rank_ic": 0.10,
            "significance_observe": verdict,
        }

    @patch("optimizer.predictor_sizing_optimizer.boto3")
    @patch("optimizer.recommendation_artifact.boto3")
    def test_default_off_applies_even_when_would_block(self, mock_art, mock_apply):
        """CRITICAL non-enforcement guarantee: default path unchanged."""
        _set_module_cfg()  # no enforce_significance → defaults False
        legacy_s3 = MagicMock()
        legacy_s3.get_object.side_effect = Exception("NoSuchKey")
        mock_apply.client.return_value = legacy_s3
        mock_art.client.return_value = MagicMock()
        outcome = apply(self._enable_result({"would_block": True}), bucket="b")
        assert outcome["applied"] is True

    @patch("optimizer.predictor_sizing_optimizer.boto3")
    @patch("optimizer.recommendation_artifact.boto3")
    def test_enforce_blocks_insignificant(self, mock_art, mock_apply):
        _set_module_cfg(extra={"enforce_significance": True})
        legacy_s3 = MagicMock()
        mock_apply.client.return_value = legacy_s3
        mock_art.client.return_value = MagicMock()
        outcome = apply(self._enable_result({"would_block": True}), bucket="b")
        assert outcome["applied"] is False
        assert "significance enforce" in outcome["reason"]
        assert legacy_s3.put_object.call_args_list == []  # no live write

    @patch("optimizer.predictor_sizing_optimizer.boto3")
    @patch("optimizer.recommendation_artifact.boto3")
    def test_enforce_allows_significant(self, mock_art, mock_apply):
        _set_module_cfg(extra={"enforce_significance": True})
        legacy_s3 = MagicMock()
        legacy_s3.get_object.side_effect = Exception("NoSuchKey")
        mock_apply.client.return_value = legacy_s3
        mock_art.client.return_value = MagicMock()
        outcome = apply(self._enable_result({"would_block": False}), bucket="b")
        assert outcome["applied"] is True

    @patch("optimizer.predictor_sizing_optimizer.boto3")
    @patch("optimizer.recommendation_artifact.boto3")
    def test_enforce_missing_verdict_blocks_conservatively(self, mock_art, mock_apply):
        _set_module_cfg(extra={"enforce_significance": True})
        mock_apply.client.return_value = MagicMock()
        mock_art.client.return_value = MagicMock()
        outcome = apply(self._enable_result(None), bucket="b")
        assert outcome["applied"] is False


class TestApplyCutoverSkip:
    """When assembler cutover is enabled, predictor_sizing_optimizer's
    legacy read-modify-write of executor_params.json is skipped — the
    assembler is the sole writer."""

    @patch("optimizer.predictor_sizing_optimizer.boto3")
    @patch("optimizer.recommendation_artifact.boto3")
    def test_cutover_enabled_skips_legacy_live_write(
        self, mock_artifact_boto3, mock_apply_boto3,
    ):
        _set_module_cfg()
        set_cutover_enabled(True)
        legacy_s3 = MagicMock()
        artifact_s3 = MagicMock()
        mock_apply_boto3.client.return_value = legacy_s3
        mock_artifact_boto3.client.return_value = artifact_s3

        result = {
            "status": "ok",
            "recommendation": "enable",
            "overall_rank_ic": 0.10,
        }
        outcome = apply(result, bucket="test-bucket")
        assert outcome["applied"] is False
        assert "cutover_mode" in outcome["reason"]

        # Legacy NEVER read or wrote the live key.
        assert legacy_s3.put_object.call_args_list == []
        assert legacy_s3.get_object.call_args_list == []

        # Artifact still written.
        artifact_writes = [
            c for c in artifact_s3.put_object.call_args_list
            if c.kwargs["Key"].endswith("/from_predictor_sizing_optimizer.json")
        ]
        assert len(artifact_writes) == 1
