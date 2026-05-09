"""Unit tests for optimizer.executor_optimizer — guardrail logic, no S3 calls.

Two ranking paths are tested:
- Legacy (Sharpe-with-drawdown) — default behavior, unchanged from pre-revamp.
- Skill-composite (sortino only, no tiebreaker) — gated by
  ``executor_optimizer.use_skill_composite_target`` config flag. Sortino
  is the skilled-risk-taking signal (downside-aware return); total_alpha
  vs SPY surfaces in the result dict for operator display but doesn't
  drive ranking. Exact-Sortino ties are deferred to pandas stable-sort.

Production-vs-shadow promotion paths are also tested via mocked S3.
"""
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from optimizer.assembler import set_cutover_enabled
from optimizer.executor_optimizer import (
    S3_PARAMS_KEY,
    S3_SHADOW_PREFIX,
    apply,
    init_config,
    produce_artifact,
    recommend,
)


@pytest.fixture(autouse=True)
def _reset_cutover_flag():
    """Always reset the assembler cutover flag around each test so
    set_cutover_enabled(True) in one test never leaks into the next."""
    set_cutover_enabled(False)
    yield
    set_cutover_enabled(False)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _init_default_config(extra: dict | None = None):
    cfg = {
        "executor_optimizer": {
            "min_valid_combos": 2,
            "min_sharpe_improvement": 0.05,
            "min_sortino_improvement": 0.05,
            "min_trades_to_promote": 0,  # disable trade-count gate for these tests
            "drawdown_penalty_weight": 0.5,
        }
    }
    if extra:
        cfg["executor_optimizer"].update(extra)
    init_config(cfg)


def _make_sweep_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# Three combos with deliberately divergent (sharpe, sortino, alpha) profiles
# so each ranking path picks a different combo. This pins the design intent:
#   - Legacy → highest Sharpe-minus-drawdown (combo A).
#   - Skill-composite → highest Sortino (combo B); alpha is tiebreaker only.
COMBOS = [
    # combo A: highest Sharpe (legacy picks this), modest Sortino, bad alpha
    {"atr_multiplier": 3.0, "min_score": 75, "max_position_pct": 0.10,
     "sharpe_ratio": 0.70, "total_alpha": -2.5, "sortino_ratio": 0.85,
     "max_drawdown": -0.05, "total_trades": 100},
    # combo B: highest Sortino (skill picks this), modest Sharpe, positive alpha
    {"atr_multiplier": 2.0, "min_score": 70, "max_position_pct": 0.05,
     "sharpe_ratio": 0.55, "total_alpha": 0.30, "sortino_ratio": 0.95,
     "max_drawdown": -0.10, "total_trades": 80},
    # combo C: middle on every axis
    {"atr_multiplier": 2.5, "min_score": 70, "max_position_pct": 0.05,
     "sharpe_ratio": 0.60, "total_alpha": -0.50, "sortino_ratio": 0.70,
     "max_drawdown": -0.08, "total_trades": 90},
]


# ── Legacy path (default — flag off) ─────────────────────────────────────────


class TestLegacyRanking:
    """When use_skill_composite_target is off (default), behavior unchanged."""

    def test_default_off_uses_legacy_path(self):
        _init_default_config()
        df = _make_sweep_df(COMBOS)
        result = recommend(df, base_config={})
        assert result["status"] == "ok"
        assert result["fit_target"] == "sharpe_legacy"

    def test_legacy_picks_highest_sharpe_minus_drawdown(self):
        _init_default_config()
        df = _make_sweep_df(COMBOS)
        result = recommend(df, base_config={})
        # Combo A has the highest Sharpe (0.70) — even with drawdown penalty
        # (0.05 × 0.5 = 0.025 → score 0.675), it beats B (0.55 - 0.05 = 0.50)
        # and C (0.60 - 0.04 = 0.56). Legacy ranking picks combo A.
        assert result["recommended_params"]["atr_multiplier"] == 3.0
        assert result["best_sharpe"] == 0.7
        # Alpha is negative for combo A — but legacy doesn't gate on it.
        assert result["best_alpha"] == -2.5

    def test_legacy_negative_sharpe_blocks(self):
        _init_default_config()
        rows = [
            {**r, "sharpe_ratio": -0.1} for r in COMBOS
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={})
        assert result["status"] == "negative_sharpe"

    def test_legacy_no_improvement_blocks(self):
        _init_default_config()
        # All combos within 1% of each other on Sharpe → fails 5% gate.
        rows = [
            {**r, "sharpe_ratio": 0.5 + (i * 0.001)}
            for i, r in enumerate(COMBOS)
        ]
        df = _make_sweep_df(rows)
        # Pass current_params close to combo C → baseline ~ best.
        result = recommend(df, base_config={}, current_params={
            "atr_multiplier": 2.5, "min_score": 70, "max_position_pct": 0.05,
        })
        assert result["status"] == "no_improvement"


# ── Skill-composite path (flag on) ───────────────────────────────────────────


class TestSkillCompositeRanking:
    """When use_skill_composite_target is on, ranks by sortino + alpha."""

    def test_flag_on_switches_to_skill_path(self):
        _init_default_config({"use_skill_composite_target": True})
        df = _make_sweep_df(COMBOS)
        result = recommend(df, base_config={})
        assert result["fit_target"] == "skill_composite"

    def test_skill_picks_highest_sortino_not_highest_sharpe(self):
        """The exact divergence the 2026-05-09 email surfaced. Combo A has
        higher Sharpe (0.70 vs 0.55) but combo B has higher Sortino (0.95
        vs 0.85) — skill_composite picks B because Sortino rewards skilled
        risk-taking, Sharpe doesn't."""
        _init_default_config({"use_skill_composite_target": True})
        df = _make_sweep_df(COMBOS)
        result = recommend(df, base_config={})
        assert result["status"] == "ok"
        assert result["recommended_params"]["atr_multiplier"] == 2.0
        assert result["best_sortino"] == 0.95
        # Alpha is reported (presentation tiebreaker) but didn't drive selection.
        assert result["best_alpha"] == 0.30

    def test_skill_exact_sortino_ties_resolve_via_stable_sort(self):
        """Sortino-only ranking: exact ties (rare on continuous sweeps) defer
        to pandas stable-sort — original DataFrame order preserved among
        equal Sortinos. Higher alpha does NOT win; alpha is presentation,
        not a ranking axis.
        """
        _init_default_config({"use_skill_composite_target": True})
        # Both rows have identical Sortino (0.9). Higher-alpha row is SECOND
        # in the input. Stable sort preserves original order → first row wins
        # despite second having higher alpha.
        rows = [
            {"atr_multiplier": 2.0, "sharpe_ratio": 0.5, "total_alpha": 0.10,
             "sortino_ratio": 0.9, "total_trades": 50},
            {"atr_multiplier": 3.0, "sharpe_ratio": 0.5, "total_alpha": 0.30,
             "sortino_ratio": 0.9, "total_trades": 50},
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={})
        # Stable sort: first-found-among-equals wins; alpha is irrelevant to ranking.
        assert result["recommended_params"]["atr_multiplier"] == 2.0
        # Alpha still surfaces in the result for operator display.
        assert result["best_alpha"] == 0.10

    def test_skill_negative_sortino_blocks(self):
        """Mirrors the negative-Sharpe guard but on the skilled-risk-taking
        axis. A config whose downside-aware return is loss-making shouldn't
        be auto-promoted regardless of Sharpe or alpha."""
        _init_default_config({"use_skill_composite_target": True})
        rows = [
            {**r, "sortino_ratio": -0.5 - (i * 0.1)}
            for i, r in enumerate(COMBOS)
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={})
        assert result["status"] == "negative_sortino"
        # Best sortino across all combos is the highest (least-negative).
        assert result["best_sortino"] == -0.5

    def test_skill_no_improvement_blocks(self):
        _init_default_config({
            "use_skill_composite_target": True,
            "min_sortino_improvement": 0.20,  # need 20% sortino lift
        })
        rows = [
            {"atr_multiplier": 2.0, "sharpe_ratio": 0.5, "total_alpha": 0.50,
             "sortino_ratio": 0.50, "total_trades": 50},
            {"atr_multiplier": 3.0, "sharpe_ratio": 0.5, "total_alpha": 0.51,
             "sortino_ratio": 0.51, "total_trades": 50},
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={}, current_params={
            "atr_multiplier": 2.0,
        })
        # Best Sortino (0.51) only 2% better than baseline (0.50) — below 20% gate.
        assert result["status"] == "no_improvement"

    def test_skill_negative_alpha_does_not_block(self):
        """Sortino > 0 with negative SPY-alpha is ALLOWED to promote — alpha
        vs SPY is presentation framing, not a gate. A high-vol portfolio
        with positive downside-aware return but a flat/down-market period
        relative to SPY can still be the right risk-taking config."""
        _init_default_config({"use_skill_composite_target": True})
        rows = [
            {"atr_multiplier": 2.0, "sharpe_ratio": 0.5, "total_alpha": -0.10,
             "sortino_ratio": 0.80, "total_trades": 50},
            {"atr_multiplier": 3.0, "sharpe_ratio": 0.5, "total_alpha": -0.50,
             "sortino_ratio": 0.50, "total_trades": 50},
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={}, current_params={"atr_multiplier": 3.0})
        # Negative alpha across the board, but Sortino is positive and improving.
        assert result["status"] == "ok"
        assert result["recommended_params"]["atr_multiplier"] == 2.0
        assert result["best_alpha"] == -0.10  # presentation only

    def test_skill_missing_sortino_column_returns_insufficient(self):
        _init_default_config({"use_skill_composite_target": True})
        rows = [
            {"atr_multiplier": 2.0, "sharpe_ratio": 0.5, "total_alpha": 0.1, "total_trades": 50},
            {"atr_multiplier": 3.0, "sharpe_ratio": 0.6, "total_alpha": 0.2, "total_trades": 50},
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={})
        assert result["status"] == "insufficient_data"
        assert "sortino_ratio" in result["note"]


# ── Apply: shadow vs production write paths ──────────────────────────────────


class TestApplyShadowVsProduction:
    """The two-stage activation: skill_composite computed → shadow archive
    until enforce_skill_composite is also flipped."""

    def _ok_result(self, fit_target: str) -> dict:
        return {
            "status": "ok",
            "fit_target": fit_target,
            "recommended_params": {"atr_multiplier": 2.0, "min_score": 70},
            "best_sharpe": 0.55,
            "best_alpha": 0.30,
            "best_sortino": 0.95,
            "improvement_pct": 0.10,
            "n_combos_tested": 60,
        }

    @patch("optimizer.executor_optimizer.boto3")
    def test_legacy_writes_to_production_key(self, mock_boto3):
        _init_default_config()  # enforce_skill_composite default false
        s3 = MagicMock()
        mock_boto3.client.return_value = s3
        with patch("optimizer.rollback.save_previous"):
            outcome = apply(self._ok_result("sharpe_legacy"), bucket="test-bucket")
        assert outcome["applied"] is True
        # First put_object: live key. Second: history archive.
        keys_written = [c.kwargs["Key"] for c in s3.put_object.call_args_list]
        assert S3_PARAMS_KEY in keys_written
        assert any("executor_params_history" in k for k in keys_written)
        # Did NOT write to shadow archive.
        assert not any(S3_SHADOW_PREFIX in k for k in keys_written)

    @patch("optimizer.executor_optimizer.boto3")
    def test_skill_without_enforce_writes_to_shadow_only(self, mock_boto3):
        _init_default_config({
            "use_skill_composite_target": True,
            "enforce_skill_composite": False,
        })
        s3 = MagicMock()
        mock_boto3.client.return_value = s3
        outcome = apply(self._ok_result("skill_composite"), bucket="test-bucket")
        assert outcome["applied"] is False
        assert "shadow mode" in outcome["reason"].lower()
        assert outcome["fit_target"] == "skill_composite"
        # Shadow key was written; live key was NOT.
        keys_written = [c.kwargs["Key"] for c in s3.put_object.call_args_list]
        assert any(S3_SHADOW_PREFIX in k for k in keys_written)
        assert S3_PARAMS_KEY not in keys_written

    @patch("optimizer.executor_optimizer.boto3")
    def test_skill_with_enforce_writes_to_production(self, mock_boto3):
        _init_default_config({
            "use_skill_composite_target": True,
            "enforce_skill_composite": True,
        })
        s3 = MagicMock()
        mock_boto3.client.return_value = s3
        with patch("optimizer.rollback.save_previous"):
            outcome = apply(self._ok_result("skill_composite"), bucket="test-bucket")
        assert outcome["applied"] is True
        assert outcome["fit_target"] == "skill_composite"
        keys_written = [c.kwargs["Key"] for c in s3.put_object.call_args_list]
        assert S3_PARAMS_KEY in keys_written
        # No shadow write when enforced.
        assert not any(S3_SHADOW_PREFIX in k for k in keys_written)

    @patch("optimizer.executor_optimizer.boto3")
    def test_apply_payload_includes_fit_target_and_alpha(self, mock_boto3):
        _init_default_config({
            "use_skill_composite_target": True,
            "enforce_skill_composite": True,
        })
        s3 = MagicMock()
        mock_boto3.client.return_value = s3
        with patch("optimizer.rollback.save_previous"):
            apply(self._ok_result("skill_composite"), bucket="test-bucket")
        # Inspect the body written to the production key.
        live_calls = [c for c in s3.put_object.call_args_list
                      if c.kwargs["Key"] == S3_PARAMS_KEY]
        assert len(live_calls) == 1
        import json
        body = json.loads(live_calls[0].kwargs["Body"])
        assert body["fit_target"] == "skill_composite"
        assert body["best_alpha"] == 0.30
        assert body["best_sortino"] == 0.95


class TestProduceArtifact:
    """``produce_artifact`` writes the per-optimizer recommendation artifact
    at ``config/executor_params/recommendations/{date}/from_executor_optimizer.json``.
    Foundation for the optimizer-artifact-assembler arc — see
    ``optimizer-artifact-assembler-260509.md``.

    During the dual-write window (PRs 1-3 of the arc), ``apply()`` calls
    ``produce_artifact`` AND continues to write the legacy live key. After
    the cutover (PR 4), ``apply()`` will be reduced to ``produce_artifact``
    only and the assembler becomes the sole writer of the live key.
    """

    @patch("optimizer.recommendation_artifact.boto3")
    def test_produce_artifact_writes_to_canonical_key(self, mock_boto3):
        s3 = MagicMock()
        mock_boto3.client.return_value = s3
        result = {
            "status": "ok",
            "fit_target": "skill_composite",
            "recommended_params": {"atr_multiplier": 3.0, "min_score": 75},
            "best_sortino": 0.95,
            "best_alpha": 0.30,
            "best_sharpe": 0.55,
            "improvement_pct": 0.10,
            "n_combos_tested": 60,
            "note": "test recommendation",
        }
        outcome = produce_artifact(result, bucket="test-bucket")
        assert outcome["written"] is True
        # Key is config-type prefixed; date is dynamic so we match by suffix.
        assert outcome["key"].startswith("config/executor_params/recommendations/")
        assert outcome["key"].endswith("/from_executor_optimizer.json")
        s3.put_object.assert_called_once()

    @patch("optimizer.recommendation_artifact.boto3")
    def test_produce_artifact_stamps_promotion_intent(self, mock_boto3):
        s3 = MagicMock()
        mock_boto3.client.return_value = s3

        # Status ok with apply_result.applied=True → promote
        produce_artifact({
            "status": "ok",
            "apply_result": {"applied": True},
            "fit_target": "skill_composite",
            "recommended_params": {},
        }, bucket="test-bucket")
        body = json.loads(s3.put_object.call_args.kwargs["Body"])
        assert body["promotion_intent"] == "promote"

        # Status ok with apply_result.applied=False → shadow
        produce_artifact({
            "status": "ok",
            "apply_result": {"applied": False, "reason": "shadow mode"},
            "fit_target": "skill_composite",
            "recommended_params": {},
        }, bucket="test-bucket")
        body = json.loads(s3.put_object.call_args.kwargs["Body"])
        assert body["promotion_intent"] == "shadow"

        # Non-ok status → skip (artifact still written for audit)
        produce_artifact({
            "status": "negative_sortino",
            "fit_target": "skill_composite",
            "recommended_params": {},
        }, bucket="test-bucket")
        body = json.loads(s3.put_object.call_args.kwargs["Body"])
        assert body["promotion_intent"] == "skip"

    @patch("optimizer.recommendation_artifact.boto3")
    def test_produce_artifact_carries_diagnostics(self, mock_boto3):
        s3 = MagicMock()
        mock_boto3.client.return_value = s3
        result = {
            "status": "ok",
            "fit_target": "skill_composite",
            "recommended_params": {"atr_multiplier": 3.0},
            "best_sortino": 0.95,
            "best_alpha": 0.30,
            "best_sharpe": 0.55,
            "improvement_pct": 0.10,
            "n_combos_tested": 60,
        }
        produce_artifact(result, bucket="test-bucket")
        body = json.loads(s3.put_object.call_args.kwargs["Body"])
        assert body["diagnostic"]["best_sortino"] == 0.95
        assert body["diagnostic"]["best_alpha"] == 0.30
        assert body["diagnostic"]["improvement_pct"] == 0.10
        assert body["diagnostic"]["n_combos_tested"] == 60
        assert body["recommendation_kind"] == "full_replace"

    @patch("optimizer.recommendation_artifact.boto3")
    def test_produce_artifact_swallows_s3_errors_non_fatal(self, mock_boto3):
        # During dual-write window, artifact write failure must NOT break
        # the legacy live write path. produce_artifact returns a non-fatal
        # error dict and logs a warning.
        s3 = MagicMock()
        s3.put_object.side_effect = Exception("S3 disconnected")
        mock_boto3.client.return_value = s3
        outcome = produce_artifact({
            "status": "ok",
            "fit_target": "skill_composite",
            "recommended_params": {},
        }, bucket="test-bucket")
        assert outcome["written"] is False
        assert "S3 disconnected" in outcome["reason"]


class TestApplyWritesArtifactInDualWriteMode:
    """During PR 1's dual-write window, every ``apply()`` invocation must
    write BOTH the legacy live key (or shadow archive) AND the new
    recommendation artifact. Pin both writes happen for every status path.
    """

    def _ok_result(self) -> dict:
        return {
            "status": "ok",
            "fit_target": "sharpe_legacy",
            "recommended_params": {"atr_multiplier": 2.0, "min_score": 70},
            "best_sharpe": 0.55,
            "best_alpha": 0.30,
            "best_sortino": 0.95,
            "improvement_pct": 0.10,
            "n_combos_tested": 60,
        }

    @patch("optimizer.executor_optimizer.boto3")
    @patch("optimizer.recommendation_artifact.boto3")
    def test_legacy_apply_path_also_writes_artifact(self, mock_artifact_boto3, mock_apply_boto3):
        _init_default_config()
        legacy_s3 = MagicMock()
        artifact_s3 = MagicMock()
        mock_apply_boto3.client.return_value = legacy_s3
        mock_artifact_boto3.client.return_value = artifact_s3
        with patch("optimizer.rollback.save_previous"):
            outcome = apply(self._ok_result(), bucket="test-bucket")

        # Legacy live + history writes happened (existing behavior).
        assert outcome["applied"] is True
        legacy_keys = [c.kwargs["Key"] for c in legacy_s3.put_object.call_args_list]
        assert S3_PARAMS_KEY in legacy_keys

        # Artifact write happened (new behavior).
        artifact_keys = [c.kwargs["Key"] for c in artifact_s3.put_object.call_args_list]
        assert any(
            k.startswith("config/executor_params/recommendations/")
            and k.endswith("/from_executor_optimizer.json")
            for k in artifact_keys
        )

    @patch("optimizer.executor_optimizer.boto3")
    @patch("optimizer.recommendation_artifact.boto3")
    def test_non_ok_status_still_writes_artifact(self, mock_artifact_boto3, mock_apply_boto3):
        # negative_sortino path: apply() returns early, but artifact
        # should still be written for audit — promotion_intent="skip".
        _init_default_config({"use_skill_composite_target": True})
        legacy_s3 = MagicMock()
        artifact_s3 = MagicMock()
        mock_apply_boto3.client.return_value = legacy_s3
        mock_artifact_boto3.client.return_value = artifact_s3

        bad_result = {
            "status": "negative_sortino",
            "fit_target": "skill_composite",
            "recommended_params": {"atr_multiplier": 3.0},
            "best_sortino": -0.5,
            "best_alpha": -1.0,
        }
        outcome = apply(bad_result, bucket="test-bucket")
        assert outcome["applied"] is False

        # Artifact written even though apply refused to promote.
        artifact_keys = [c.kwargs["Key"] for c in artifact_s3.put_object.call_args_list]
        assert any(
            k.startswith("config/executor_params/recommendations/")
            and k.endswith("/from_executor_optimizer.json")
            for k in artifact_keys
        )
        # Intent stamped as skip.
        artifact_call = next(
            c for c in artifact_s3.put_object.call_args_list
            if c.kwargs["Key"].endswith("/from_executor_optimizer.json")
        )
        body = json.loads(artifact_call.kwargs["Body"])
        assert body["promotion_intent"] == "skip"


class TestApplyCutoverSkip:
    """When ``optimizer.assembler.is_cutover_enabled()`` returns True,
    the legacy live-key write path is skipped — the assembler is the sole
    writer of ``config/executor_params.json``. The artifact write is
    unaffected (still required for the assembler to read)."""

    def _ok_result(self) -> dict:
        return {
            "status": "ok",
            "fit_target": "sharpe_legacy",
            "recommended_params": {"atr_multiplier": 2.0, "min_score": 70},
            "best_sharpe": 0.55,
            "best_alpha": 0.30,
            "best_sortino": 0.95,
            "improvement_pct": 0.10,
            "n_combos_tested": 60,
        }

    @patch("optimizer.executor_optimizer.boto3")
    @patch("optimizer.recommendation_artifact.boto3")
    def test_cutover_enabled_skips_legacy_live_write(
        self, mock_artifact_boto3, mock_apply_boto3,
    ):
        _init_default_config()
        set_cutover_enabled(True)
        legacy_s3 = MagicMock()
        artifact_s3 = MagicMock()
        mock_apply_boto3.client.return_value = legacy_s3
        mock_artifact_boto3.client.return_value = artifact_s3

        outcome = apply(self._ok_result(), bucket="test-bucket")

        assert outcome["applied"] is False
        assert "cutover_mode" in outcome["reason"]
        # Legacy NEVER touched live or history keys.
        assert legacy_s3.put_object.call_args_list == []
        # Artifact STILL written (assembler will read it).
        artifact_keys = [c.kwargs["Key"] for c in artifact_s3.put_object.call_args_list]
        assert any(
            k.endswith("/from_executor_optimizer.json") for k in artifact_keys
        )

    @patch("optimizer.executor_optimizer.boto3")
    @patch("optimizer.recommendation_artifact.boto3")
    def test_cutover_disabled_default_keeps_legacy_write(
        self, mock_artifact_boto3, mock_apply_boto3,
    ):
        # Belt-and-suspenders: confirms the autouse fixture's reset works
        # AND that with the flag off, legacy still writes (PR 1 behavior
        # preserved when assembler hasn't taken over yet).
        _init_default_config()
        # Flag is False by default per fixture; do not call set_cutover_enabled.
        legacy_s3 = MagicMock()
        artifact_s3 = MagicMock()
        mock_apply_boto3.client.return_value = legacy_s3
        mock_artifact_boto3.client.return_value = artifact_s3

        with patch("optimizer.rollback.save_previous"):
            outcome = apply(self._ok_result(), bucket="test-bucket")

        assert outcome["applied"] is True
        legacy_keys = [c.kwargs["Key"] for c in legacy_s3.put_object.call_args_list]
        assert S3_PARAMS_KEY in legacy_keys
