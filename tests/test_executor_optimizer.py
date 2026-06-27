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
    validate_holdout,
    validate_walk_forward,
    _rolling_windows,
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

    def test_skill_composite_emits_rank_metric_sortino_by_default(self):
        """rank_metric field in result identifies which axis drove the
        ranking. Default skill-composite path uses sortino_ratio."""
        _init_default_config({"use_skill_composite_target": True})
        df = _make_sweep_df(COMBOS)
        result = recommend(df, base_config={})
        assert result["rank_metric"] == "sortino_ratio"

    def test_skill_blocks_when_psr_below_threshold(self):
        """PSR confidence gate (Workstream D bullet 3): refuse to promote
        when best combo's Probabilistic Sharpe Ratio < min_psr threshold —
        Sharpe is statistically indistinguishable from zero given the
        sample size + skewness/kurtosis."""
        _init_default_config({
            "use_skill_composite_target": True,
            "min_psr": 0.95,
        })
        rows = [
            {"atr_multiplier": 2.0, "sharpe_ratio": 0.5, "total_alpha": 0.10,
             "sortino_ratio": 0.6, "psr": 0.60,  # 60% confidence — far below 0.95
             "total_trades": 50},
            {"atr_multiplier": 3.0, "sharpe_ratio": 0.5, "total_alpha": 0.30,
             "sortino_ratio": 0.9, "psr": 0.65,
             "total_trades": 50},
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={}, current_params={"atr_multiplier": 2.0})
        assert result["status"] == "insufficient_psr_confidence"
        assert "PSR=0.650" in result["note"]
        assert result["best_psr"] == 0.65

    def test_skill_promotes_when_psr_clears_threshold(self):
        _init_default_config({
            "use_skill_composite_target": True,
            "min_psr": 0.95,
        })
        rows = [
            {"atr_multiplier": 2.0, "sharpe_ratio": 0.5, "total_alpha": 0.10,
             "sortino_ratio": 0.6, "psr": 0.92,
             "total_trades": 50},
            {"atr_multiplier": 3.0, "sharpe_ratio": 0.5, "total_alpha": 0.30,
             "sortino_ratio": 0.9, "psr": 0.97,  # clears 0.95
             "total_trades": 50},
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={}, current_params={"atr_multiplier": 2.0})
        assert result["status"] == "ok"
        assert result["best_psr"] == 0.97

    def test_skill_psr_absent_skips_gate(self):
        """When PSR isn't in sweep_df (older sweep runs / vectorbt_bridge
        couldn't compute due to <30 obs), the PSR gate is skipped. Other
        gates (negative-sortino, no_improvement) still apply."""
        _init_default_config({"use_skill_composite_target": True})
        rows = [
            {"atr_multiplier": 2.0, "sharpe_ratio": 0.5, "total_alpha": 0.10,
             "sortino_ratio": 0.6, "total_trades": 50},
            {"atr_multiplier": 3.0, "sharpe_ratio": 0.5, "total_alpha": 0.30,
             "sortino_ratio": 0.9, "total_trades": 50},
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={}, current_params={"atr_multiplier": 2.0})
        # No psr column at all → gate skipped → ok status.
        assert result["status"] == "ok"
        assert result["best_psr"] is None

    def test_legacy_path_does_not_apply_psr_gate(self):
        """PSR gate is skill-composite-only. Legacy path proceeds even with
        low PSR — preserves pre-cutover behavior."""
        _init_default_config()  # legacy default
        rows = [
            {"atr_multiplier": 2.0, "sharpe_ratio": 0.5, "total_alpha": 0.10,
             "sortino_ratio": 0.6, "max_drawdown": -0.05,
             "psr": 0.30,  # low — skill mode would block; legacy ignores
             "total_trades": 100},
            {"atr_multiplier": 3.0, "sharpe_ratio": 0.6, "total_alpha": 0.30,
             "sortino_ratio": 0.7, "max_drawdown": -0.05,
             "psr": 0.30,
             "total_trades": 100},
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={}, current_params={"atr_multiplier": 2.0})
        # Legacy ignores PSR — promotes on Sharpe-with-drawdown ranking.
        assert result["status"] == "ok"
        assert result["fit_target"] == "sharpe_legacy"

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
    def test_legacy_writes_to_production_key_when_opted_in(self, mock_boto3):
        # config#1053 Phase C: the legacy live write is now opt-in only.
        _init_default_config({"legacy_executor_params_live_apply": True})
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
    def test_legacy_default_writes_to_shadow_only(self, mock_boto3):
        """config#1053 Phase C: by DEFAULT the legacy 1/n-path sweep no longer
        overwrites live config — the MVO optimizer owns sizing. It routes to the
        shadow archive instead, leaving config/executor_params.json untouched."""
        _init_default_config()  # legacy_executor_params_live_apply default False
        s3 = MagicMock()
        mock_boto3.client.return_value = s3
        outcome = apply(self._ok_result("sharpe_legacy"), bucket="test-bucket")
        assert outcome["applied"] is False
        assert "shadow mode" in outcome["reason"].lower()
        assert "live auto-apply" in outcome["reason"].lower()
        keys_written = [c.kwargs["Key"] for c in s3.put_object.call_args_list]
        # Shadow archive written; live key NOT touched.
        assert any(S3_SHADOW_PREFIX in k for k in keys_written)
        assert S3_PARAMS_KEY not in keys_written

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
        # config#1053 Phase C: legacy live write is opt-in; flag on to exercise it.
        _init_default_config({"legacy_executor_params_live_apply": True})
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
        # Belt-and-suspenders: confirms the autouse fixture's reset works AND
        # that with the CUTOVER flag off, the legacy write path runs (assembler
        # hasn't taken over). config#1053 Phase C: the legacy live write is now
        # opt-in, so enable it here to isolate the cutover-gate behavior under test.
        _init_default_config({"legacy_executor_params_live_apply": True})
        # Cutover flag is False by default per fixture; do not call set_cutover_enabled.
        legacy_s3 = MagicMock()
        artifact_s3 = MagicMock()
        mock_apply_boto3.client.return_value = legacy_s3
        mock_artifact_boto3.client.return_value = artifact_s3

        with patch("optimizer.rollback.save_previous"):
            outcome = apply(self._ok_result(), bucket="test-bucket")

        assert outcome["applied"] is True
        legacy_keys = [c.kwargs["Key"] for c in legacy_s3.put_object.call_args_list]
        assert S3_PARAMS_KEY in legacy_keys


# ── prefer_risk_matched_alpha flag (PR 3 — Workstream D rank-metric swap) ────


class TestPreferRiskMatchedAlpha:
    """When `prefer_risk_matched_alpha` is on AND `use_skill_composite_target`
    is on, ranking swaps from `sortino_ratio` to `alpha_vs_ew_high_vol`
    (Workstream D risk-matched skill metric — portfolio return minus the
    EW-high-vol vol-matched basket's return). The negative-sortino sanity
    guard still fires regardless of rank column; an additional
    negative-alpha-vs-EW guard catches the "best combo can't even beat the
    dumb-vol basket" failure mode. Default off — flips on after 2+ Saturday
    SF shadow cycles confirm the basket is sensibly populated.
    """

    def _combos_with_ew_alpha(self):
        # Three combos. Sortino ranking would pick combo B (highest sortino).
        # alpha_vs_ew_high_vol ranking should pick combo C (highest risk-
        # matched skill). This makes the rank-column swap observable.
        return [
            # combo A: high sharpe, modest sortino, very modest risk-matched skill
            {"atr_multiplier": 3.0, "min_score": 75, "max_position_pct": 0.10,
             "sharpe_ratio": 0.70, "total_alpha": 0.20, "sortino_ratio": 0.85,
             "alpha_vs_ew_high_vol": 0.02,
             "max_drawdown": -0.05, "total_trades": 100},
            # combo B: highest Sortino — would win under default skill-composite
            {"atr_multiplier": 2.0, "min_score": 70, "max_position_pct": 0.05,
             "sharpe_ratio": 0.55, "total_alpha": 0.30, "sortino_ratio": 0.95,
             "alpha_vs_ew_high_vol": 0.03,
             "max_drawdown": -0.10, "total_trades": 80},
            # combo C: highest alpha_vs_ew_high_vol — wins under
            # prefer_risk_matched_alpha
            {"atr_multiplier": 2.5, "min_score": 70, "max_position_pct": 0.05,
             "sharpe_ratio": 0.60, "total_alpha": 0.25, "sortino_ratio": 0.75,
             "alpha_vs_ew_high_vol": 0.08,
             "max_drawdown": -0.08, "total_trades": 90},
        ]

    def test_flag_off_default_keeps_sortino_rank_path(self):
        """Default (`prefer_risk_matched_alpha=False`) is unchanged —
        skill-composite still ranks by sortino. Combo B wins."""
        _init_default_config({"use_skill_composite_target": True})
        df = _make_sweep_df(self._combos_with_ew_alpha())
        result = recommend(df, base_config={})
        assert result["status"] == "ok"
        assert result["rank_metric"] == "sortino_ratio"
        assert result["recommended_params"]["atr_multiplier"] == 2.0  # combo B

    def test_flag_on_swaps_rank_to_alpha_vs_ew_high_vol(self):
        """With `prefer_risk_matched_alpha=True`, combo C (highest
        risk-matched alpha) wins instead of combo B (highest sortino).
        This is the institutional Workstream D rank metric."""
        _init_default_config({
            "use_skill_composite_target": True,
            "prefer_risk_matched_alpha": True,
        })
        df = _make_sweep_df(self._combos_with_ew_alpha())
        result = recommend(df, base_config={})
        assert result["status"] == "ok"
        assert result["rank_metric"] == "alpha_vs_ew_high_vol"
        assert result["recommended_params"]["atr_multiplier"] == 2.5  # combo C
        assert result["best_alpha_vs_ew_high_vol"] == 0.08

    def test_flag_on_without_parent_flag_is_noop(self):
        """`prefer_risk_matched_alpha=True` with `use_skill_composite_target`
        off → legacy Sharpe-with-drawdown path still runs (the inner flag
        only takes effect inside the skill-composite branch)."""
        _init_default_config({"prefer_risk_matched_alpha": True})
        df = _make_sweep_df(self._combos_with_ew_alpha())
        result = recommend(df, base_config={})
        assert result["fit_target"] == "sharpe_legacy"
        assert result["rank_metric"] == "sharpe_drawdown"

    def test_flag_on_blocks_when_best_risk_matched_alpha_negative(self):
        """The Workstream D analogue of the negative-Sortino guard: refuse
        to auto-apply when even the best combo underperforms the
        vol-matched baseline."""
        _init_default_config({
            "use_skill_composite_target": True,
            "prefer_risk_matched_alpha": True,
        })
        # All three combos UNDER-perform the basket but sortino is positive.
        rows = [
            {"atr_multiplier": 2.0, "sharpe_ratio": 0.5, "total_alpha": 0.10,
             "sortino_ratio": 0.6, "alpha_vs_ew_high_vol": -0.05,
             "total_trades": 50},
            {"atr_multiplier": 3.0, "sharpe_ratio": 0.5, "total_alpha": 0.30,
             "sortino_ratio": 0.7, "alpha_vs_ew_high_vol": -0.02,
             "total_trades": 60},
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={})
        assert result["status"] == "negative_alpha_vs_ew_high_vol"

    def test_flag_on_blocks_when_basket_unpopulated(self):
        """If `alpha_vs_ew_high_vol` is missing from sweep_df (basket couldn't
        be constructed — insufficient corpus depth, short window), return
        insufficient_data instead of falling back silently to a different
        rank column. The operator should know the basket failed."""
        _init_default_config({
            "use_skill_composite_target": True,
            "prefer_risk_matched_alpha": True,
        })
        # NO alpha_vs_ew_high_vol column → the rank_col=alpha_vs_ew_high_vol
        # branch can't find it.
        rows = [
            {"atr_multiplier": 2.0, "sharpe_ratio": 0.5, "total_alpha": 0.10,
             "sortino_ratio": 0.6, "total_trades": 50},
            {"atr_multiplier": 3.0, "sharpe_ratio": 0.5, "total_alpha": 0.30,
             "sortino_ratio": 0.7, "total_trades": 60},
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={})
        assert result["status"] == "insufficient_data"
        assert "alpha_vs_ew_high_vol" in result["note"]

    def test_flag_on_negative_sortino_still_blocks_strategy_quality(self):
        """Sortino sanity guard fires regardless of rank column. Even when
        ranking on alpha_vs_ew_high_vol, a strategy whose downside-aware
        return is loss-making won't be promoted — separate strategy-quality
        check from the rank-metric check."""
        _init_default_config({
            "use_skill_composite_target": True,
            "prefer_risk_matched_alpha": True,
        })
        rows = [
            {"atr_multiplier": 2.0, "sharpe_ratio": 0.5, "total_alpha": 0.10,
             "sortino_ratio": -0.5, "alpha_vs_ew_high_vol": 0.05,  # +alpha vs basket but losing money
             "total_trades": 50},
            {"atr_multiplier": 3.0, "sharpe_ratio": 0.5, "total_alpha": 0.30,
             "sortino_ratio": -0.3, "alpha_vs_ew_high_vol": 0.02,
             "total_trades": 60},
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={})
        # Sortino guard fires before the alpha-vs-EW check.
        assert result["status"] == "negative_sortino"

    def test_flag_on_no_improvement_uses_alpha_threshold(self):
        """The min-improvement gate is rank-metric-based. Reuses
        min_sortino_improvement as the threshold (same scale)."""
        _init_default_config({
            "use_skill_composite_target": True,
            "prefer_risk_matched_alpha": True,
            "min_sortino_improvement": 0.20,
        })
        rows = [
            {"atr_multiplier": 2.0, "sharpe_ratio": 0.5, "total_alpha": 0.50,
             "sortino_ratio": 0.50, "alpha_vs_ew_high_vol": 0.05,
             "total_trades": 50},
            {"atr_multiplier": 3.0, "sharpe_ratio": 0.5, "total_alpha": 0.51,
             "sortino_ratio": 0.51, "alpha_vs_ew_high_vol": 0.051,
             "total_trades": 50},
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={}, current_params={"atr_multiplier": 2.0})
        # 0.051 vs 0.05 = 2% lift, below 20% gate.
        assert result["status"] == "no_improvement"
        assert "alpha_vs_ew_high_vol" in result["note"]


# ── Alpha-floor constraint (canonical-alpha framework SOTA gate) ─────────────
#
# Origin: 2026-05-20 audit. Live executor_params.json carried min_score=75
# with best_alpha=-2.5427 (Sharpe-ranked) while the Sortino-ranked shadow
# history converged on the same params (best_alpha=-2.5641) — both single-
# objective rankings reward variance-reduction, so absent an alpha-floor
# constraint both paths converge on alpha-negative "do nothing" configs.
# Per the system Objective ("Maximize long-term alpha") and the canonical-
# alpha framework ([[anchor-gates-on-skilled-risk-not-sharpe]]), alpha-
# positive is a hard constraint, not a presentation field.


class TestAlphaFloorConstraint:
    """alpha_floor config knob — when set, filter sweep_df to alpha >= floor
    BEFORE ranking. Default off (None) preserves prior behavior."""

    def test_alpha_floor_unset_preserves_prior_behavior(self):
        """alpha_floor=None (default): negative-alpha combos can still rank +
        promote under the skill-composite path (matches
        test_skill_negative_alpha_does_not_block — alpha is presentation
        framing unless the gate is activated)."""
        _init_default_config({"use_skill_composite_target": True})
        rows = [
            {"atr_multiplier": 2.0, "sharpe_ratio": 0.5, "total_alpha": -2.5,
             "sortino_ratio": 0.95, "total_trades": 50},
            {"atr_multiplier": 3.0, "sharpe_ratio": 0.5, "total_alpha": -1.0,
             "sortino_ratio": 0.50, "total_trades": 50},
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={}, current_params={"atr_multiplier": 3.0})
        assert result["status"] == "ok"
        assert result["best_alpha"] == -2.5  # promoted despite negative alpha

    def test_alpha_floor_zero_blocks_when_all_combos_alpha_negative(self):
        """alpha_floor=0.0 with all combos alpha-negative — refuse to
        promote. Reproduces the exact 2026-05-20 incident config (both
        combos backtest alpha-negative, best Sortino=0.95 looks great
        without the alpha gate, but is a "trade nothing" config)."""
        _init_default_config({
            "use_skill_composite_target": True,
            "alpha_floor": 0.0,
        })
        rows = [
            {"atr_multiplier": 2.0, "sharpe_ratio": 0.5, "total_alpha": -2.5,
             "sortino_ratio": 0.95, "total_trades": 50},
            {"atr_multiplier": 3.0, "sharpe_ratio": 0.5, "total_alpha": -1.0,
             "sortino_ratio": 0.50, "total_trades": 50},
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={})
        assert result["status"] == "alpha_below_floor"
        assert result["alpha_floor"] == 0.0
        assert result["n_combos_below_floor"] == 2
        assert result["best_alpha_in_sweep"] == -1.0
        assert "canonical-alpha framework" in result["note"]

    def test_alpha_floor_zero_filters_then_ranks_remaining(self):
        """alpha_floor=0.0 with mixed combos — drop alpha-negative ones,
        then rank by Sortino among the survivors. The combo with the
        highest Sortino overall (combo B, sortino=0.95) is alpha-negative
        and gets dropped; the alpha-positive combo with the lower Sortino
        (combo A, sortino=0.7) wins the filtered ranking."""
        _init_default_config({
            "use_skill_composite_target": True,
            "alpha_floor": 0.0,
            "min_sortino_improvement": -1.0,  # disable improvement gate
        })
        rows = [
            # combo A: alpha-positive but lower Sortino — wins after filter
            {"atr_multiplier": 2.0, "sharpe_ratio": 0.5, "total_alpha": 0.15,
             "sortino_ratio": 0.70, "total_trades": 50},
            # combo B: highest Sortino but alpha-negative — filtered out
            {"atr_multiplier": 3.0, "sharpe_ratio": 0.5, "total_alpha": -0.5,
             "sortino_ratio": 0.95, "total_trades": 50},
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={}, current_params={"atr_multiplier": 2.0})
        assert result["status"] == "ok"
        assert result["recommended_params"]["atr_multiplier"] == 2.0
        assert result["best_sortino"] == 0.70
        assert result["best_alpha"] == 0.15

    def test_alpha_floor_positive_value_requires_alpha_cushion(self):
        """alpha_floor>0 (e.g. 0.05 = 500bps alpha cushion) raises the bar.
        Combo at 0.03 alpha gets dropped despite being alpha-positive;
        only the 0.10-alpha combo survives."""
        _init_default_config({
            "use_skill_composite_target": True,
            "alpha_floor": 0.05,
            "min_sortino_improvement": -1.0,
        })
        rows = [
            {"atr_multiplier": 2.0, "sharpe_ratio": 0.5, "total_alpha": 0.10,
             "sortino_ratio": 0.80, "total_trades": 50},
            {"atr_multiplier": 3.0, "sharpe_ratio": 0.5, "total_alpha": 0.03,
             "sortino_ratio": 0.95, "total_trades": 50},  # below 0.05 floor
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={}, current_params={"atr_multiplier": 2.0})
        assert result["status"] == "ok"
        assert result["recommended_params"]["atr_multiplier"] == 2.0
        assert result["best_alpha"] == 0.10

    def test_alpha_floor_with_no_total_alpha_column_skips_gate(self):
        """Older sweep runs without total_alpha column — gate is a no-op
        (the column-absence path means the filter never runs). Other gates
        still apply; we don't fail the run for missing diagnostics."""
        _init_default_config({
            "use_skill_composite_target": True,
            "alpha_floor": 0.0,
        })
        rows = [
            {"atr_multiplier": 2.0, "sharpe_ratio": 0.5, "sortino_ratio": 0.6,
             "total_trades": 50},
            {"atr_multiplier": 3.0, "sharpe_ratio": 0.5, "sortino_ratio": 0.9,
             "total_trades": 50},
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={}, current_params={"atr_multiplier": 2.0})
        assert result["status"] == "ok"
        assert result["best_sortino"] == 0.9

    def test_alpha_floor_at_zero_allows_exact_zero_alpha(self):
        """alpha_floor uses >= (non-negative), not > (strict positive). A
        flat-alpha config (SPY clone) is at worst neutral, not destructive,
        and shouldn't be filtered."""
        _init_default_config({
            "use_skill_composite_target": True,
            "alpha_floor": 0.0,
            "min_sortino_improvement": -1.0,
        })
        rows = [
            {"atr_multiplier": 2.0, "sharpe_ratio": 0.5, "total_alpha": 0.0,
             "sortino_ratio": 0.70, "total_trades": 50},
            {"atr_multiplier": 3.0, "sharpe_ratio": 0.5, "total_alpha": -0.01,
             "sortino_ratio": 0.95, "total_trades": 50},  # just below floor
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={}, current_params={"atr_multiplier": 2.0})
        assert result["status"] == "ok"
        assert result["recommended_params"]["atr_multiplier"] == 2.0
        assert result["best_alpha"] == 0.0

    def test_alpha_floor_legacy_path_also_filters(self):
        """alpha_floor applies in BOTH paths (legacy Sharpe + skill-composite).
        Per canonical-alpha framework, alpha-positive is a system-wide
        constraint, not a path-specific one. Legacy callers that explicitly
        set alpha_floor get the same protection."""
        _init_default_config({"alpha_floor": 0.0})  # default legacy path
        rows = [
            {"atr_multiplier": 2.0, "sharpe_ratio": 0.7, "total_alpha": -2.5,
             "sortino_ratio": 0.85, "max_drawdown": -0.05, "total_trades": 50},
            {"atr_multiplier": 3.0, "sharpe_ratio": 0.6, "total_alpha": -1.0,
             "sortino_ratio": 0.70, "max_drawdown": -0.08, "total_trades": 50},
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={})
        assert result["status"] == "alpha_below_floor"


class TestImprovementPctNearZeroBaseline:
    """Regression for ROADMAP L120 — improvement_pct used to blow up
    (inf, or 9828× misleading readings) when the baseline rank metric
    was at or near zero. The fix clamps the denominator to
    _IMPROVEMENT_DENOM_FLOOR (1e-6) and exposes the signed
    `improvement_delta` as the operator-meaningful absolute number.

    Observed in production: executor_params_shadow_history/latest.json
    (2026-05-18) carried `improvement_pct: 9828.0`."""

    def test_skill_baseline_sortino_exactly_zero_does_not_return_inf(self):
        # Sortino = 0 for the closest-to-current combo, > 0 for best.
        # Pre-fix: improvement_pct = float("inf"); post-fix: bounded.
        _init_default_config({"use_skill_composite_target": True})
        rows = [
            {"atr_multiplier": 2.0, "min_score": 70, "max_position_pct": 0.05,
             "sharpe_ratio": 0.55, "total_alpha": 0.30, "sortino_ratio": 0.5,
             "max_drawdown": -0.10, "total_trades": 80},
            {"atr_multiplier": 3.0, "min_score": 75, "max_position_pct": 0.10,
             "sharpe_ratio": 0.50, "total_alpha": 0.10, "sortino_ratio": 0.0,
             "max_drawdown": -0.05, "total_trades": 100},
        ]
        df = _make_sweep_df(rows)
        # Pin baseline to combo 2 (sortino=0) via current_params proximity.
        result = recommend(df, base_config={}, current_params={
            "atr_multiplier": 3.0, "min_score": 75, "max_position_pct": 0.10,
        })
        assert result["baseline_sortino"] == 0.0
        # No inf. improvement_pct stays finite (a large but bounded number,
        # roughly best_sortino / 1e-6).
        import math
        assert math.isfinite(result["improvement_pct"])
        # improvement_delta surfaces the operator-meaningful absolute value:
        # 0.5 - 0.0 = 0.5.
        assert result["improvement_delta"] == 0.5

    def test_skill_baseline_sortino_near_zero_no_overflow(self):
        # Sortino baseline = 1e-4 (the 2026-05-18 9828× case had a
        # divisor in this ballpark). Pre-fix: 9828×-style reading.
        # Post-fix: improvement_pct = improvement_delta / max(|baseline|, 1e-6).
        # When |baseline| > 1e-6, the floor is inert (improvement_pct stays
        # equal to the pre-clamp ratio); `improvement_delta` surfaces the
        # signed absolute number so operators don't have to interpret a
        # near-zero-baseline ratio.
        _init_default_config({"use_skill_composite_target": True})
        rows = [
            {"atr_multiplier": 2.0, "min_score": 70, "max_position_pct": 0.05,
             "sharpe_ratio": 0.55, "total_alpha": 0.30, "sortino_ratio": 0.01,
             "max_drawdown": -0.10, "total_trades": 80},
            {"atr_multiplier": 3.0, "min_score": 75, "max_position_pct": 0.10,
             "sharpe_ratio": 0.50, "total_alpha": 0.10,
             "sortino_ratio": 0.0001, "max_drawdown": -0.05,
             "total_trades": 100},
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={}, current_params={
            "atr_multiplier": 3.0, "min_score": 75, "max_position_pct": 0.10,
        })
        assert result["baseline_sortino"] == 0.0001
        # improvement_delta is the meaningful number: 0.01 - 0.0001 ≈ 0.0099.
        assert abs(result["improvement_delta"] - 0.0099) < 1e-6
        # improvement_pct = improvement_delta / 0.0001 = 99.0
        # (NOT 9828.0; the small but non-zero baseline divides cleanly).
        # Pin the math so a future refactor of the floor doesn't silently
        # change the contract.
        assert abs(result["improvement_pct"] - 99.0) < 1e-3

    def test_legacy_baseline_sharpe_exactly_zero_does_not_return_inf(self):
        # Same regression on the legacy Sharpe path. Pre-fix:
        # improvement_pct = float("inf"); post-fix: bounded.
        _init_default_config()
        rows = [
            {"atr_multiplier": 2.0, "min_score": 70, "max_position_pct": 0.05,
             "sharpe_ratio": 0.5, "total_alpha": 0.0, "sortino_ratio": 0.5,
             "max_drawdown": -0.05, "total_trades": 50},
            {"atr_multiplier": 3.0, "min_score": 75, "max_position_pct": 0.10,
             "sharpe_ratio": 0.0, "total_alpha": 0.0, "sortino_ratio": 0.0,
             "max_drawdown": -0.05, "total_trades": 50},
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={}, current_params={
            "atr_multiplier": 3.0, "min_score": 75, "max_position_pct": 0.10,
        })
        assert result["baseline_sharpe"] == 0.0
        import math
        assert math.isfinite(result["improvement_pct"])
        assert result["improvement_delta"] == 0.5

    def test_improvement_delta_surfaces_for_skill_path(self):
        # Sanity: improvement_delta is always present (not None) when
        # both best_rank and baseline_rank are computable.
        _init_default_config({"use_skill_composite_target": True})
        df = _make_sweep_df(COMBOS)
        result = recommend(df, base_config={})
        assert "improvement_delta" in result
        assert result["improvement_delta"] is not None
        # COMBOS: best Sortino 0.95 (combo B), worst-by-rank 0.70 (combo C)
        # → improvement_delta = 0.95 - 0.70 = 0.25.
        assert abs(result["improvement_delta"] - 0.25) < 1e-6


class TestBaselineSignificanceGate:
    """Per CLAUDE.md SOTA rule + ROADMAP L120 follow-up: a near-zero
    baseline rank metric is structural noise; promoting based on a
    ratio off such a baseline is misleading regardless of the
    improvement_pct value. The gate refuses promotion when the
    baseline magnitude falls under the per-metric significance floor
    (constrained optimization, not post-hoc check)."""

    def test_skill_baseline_below_floor_refuses_promotion(self):
        # baseline_sortino = 0.0 → magnitude 0 < default floor 0.05 →
        # status: baseline_insignificant. Pre-gate this would have
        # returned status: ok (the clamp made improvement_pct finite +
        # large + >> the 10% min_sortino_improvement threshold).
        _init_default_config({"use_skill_composite_target": True})
        rows = [
            {"atr_multiplier": 2.0, "min_score": 70, "max_position_pct": 0.05,
             "sharpe_ratio": 0.55, "total_alpha": 0.30, "sortino_ratio": 0.5,
             "max_drawdown": -0.10, "total_trades": 80},
            {"atr_multiplier": 3.0, "min_score": 75, "max_position_pct": 0.10,
             "sharpe_ratio": 0.50, "total_alpha": 0.10, "sortino_ratio": 0.0,
             "max_drawdown": -0.05, "total_trades": 100},
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={}, current_params={
            "atr_multiplier": 3.0, "min_score": 75, "max_position_pct": 0.10,
        })
        assert result["status"] == "baseline_insignificant"
        assert result["improvement_significant"] is False
        assert result["baseline_sortino"] == 0.0
        # The deviation/operator info still surfaces — refusing to
        # promote does NOT suppress the diagnostic numbers.
        assert result["improvement_delta"] == 0.5
        assert result["min_baseline_magnitude"] == 0.05

    def test_skill_baseline_at_floor_promotes(self):
        # baseline_sortino = 0.06 ≥ 0.05 floor → significance ok →
        # normal min_improvement gate applies.
        _init_default_config({"use_skill_composite_target": True})
        rows = [
            {"atr_multiplier": 2.0, "min_score": 70, "max_position_pct": 0.05,
             "sharpe_ratio": 0.55, "total_alpha": 0.30, "sortino_ratio": 0.5,
             "max_drawdown": -0.10, "total_trades": 80},
            {"atr_multiplier": 3.0, "min_score": 75, "max_position_pct": 0.10,
             "sharpe_ratio": 0.50, "total_alpha": 0.10, "sortino_ratio": 0.06,
             "max_drawdown": -0.05, "total_trades": 100},
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={}, current_params={
            "atr_multiplier": 3.0, "min_score": 75, "max_position_pct": 0.10,
        })
        assert result["status"] == "ok"
        assert result["improvement_significant"] is True
        assert result["baseline_sortino"] == 0.06

    def test_legacy_sharpe_baseline_below_floor_refuses_promotion(self):
        # Symmetric guard on the legacy Sharpe path. baseline_sharpe =
        # 0.0 (< 0.05 floor) → baseline_insignificant.
        _init_default_config()
        rows = [
            {"atr_multiplier": 2.0, "min_score": 70, "max_position_pct": 0.05,
             "sharpe_ratio": 0.5, "total_alpha": 0.0, "sortino_ratio": 0.5,
             "max_drawdown": -0.05, "total_trades": 50},
            {"atr_multiplier": 3.0, "min_score": 75, "max_position_pct": 0.10,
             "sharpe_ratio": 0.0, "total_alpha": 0.0, "sortino_ratio": 0.0,
             "max_drawdown": -0.05, "total_trades": 50},
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={}, current_params={
            "atr_multiplier": 3.0, "min_score": 75, "max_position_pct": 0.10,
        })
        assert result["status"] == "baseline_insignificant"
        assert result["improvement_significant"] is False
        assert result["min_baseline_magnitude"] == 0.05

    def test_risk_matched_alpha_floor_lower_than_sortino_floor(self):
        # alpha_vs_ew_high_vol rolls in raw-return units (typical
        # 0.01–0.10), not ratio units; its default floor is 0.005
        # (50bps), not 0.05. baseline_alpha_vs_ew_high_vol = 0.01
        # passes that floor; pre-gate this raised the bug where the
        # original 0.05 single-floor would have wrongly refused.
        _init_default_config({
            "use_skill_composite_target": True,
            "prefer_risk_matched_alpha": True,
        })
        rows = [
            # combo A: best risk-matched alpha (0.08)
            {"atr_multiplier": 2.0, "min_score": 70, "max_position_pct": 0.05,
             "sharpe_ratio": 0.55, "total_alpha": 0.30, "sortino_ratio": 0.75,
             "alpha_vs_ew_high_vol": 0.08, "max_drawdown": -0.10, "total_trades": 80},
            # combo B: baseline (0.01 — just above the 0.005 floor)
            {"atr_multiplier": 3.0, "min_score": 75, "max_position_pct": 0.10,
             "sharpe_ratio": 0.50, "total_alpha": 0.10, "sortino_ratio": 0.5,
             "alpha_vs_ew_high_vol": 0.01, "max_drawdown": -0.05, "total_trades": 100},
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={}, current_params={
            "atr_multiplier": 3.0, "min_score": 75, "max_position_pct": 0.10,
        })
        assert result["status"] == "ok"
        assert result["improvement_significant"] is True
        # The resolved floor matches the per-metric default for the
        # risk-matched-alpha rank column.
        assert result["min_baseline_magnitude"] == 0.005

    def test_operator_can_override_floor_uniformly(self):
        # Single-float override (back-compat ergonomics) applies
        # uniformly across the rank column.
        _init_default_config({
            "use_skill_composite_target": True,
            "min_baseline_magnitude": 0.10,
        })
        rows = [
            {"atr_multiplier": 2.0, "min_score": 70, "max_position_pct": 0.05,
             "sharpe_ratio": 0.55, "total_alpha": 0.30, "sortino_ratio": 0.5,
             "max_drawdown": -0.10, "total_trades": 80},
            # baseline_sortino = 0.08 — clears default 0.05 floor but
            # NOT the operator's tighter 0.10 override.
            {"atr_multiplier": 3.0, "min_score": 75, "max_position_pct": 0.10,
             "sharpe_ratio": 0.50, "total_alpha": 0.10, "sortino_ratio": 0.08,
             "max_drawdown": -0.05, "total_trades": 100},
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={}, current_params={
            "atr_multiplier": 3.0, "min_score": 75, "max_position_pct": 0.10,
        })
        assert result["status"] == "baseline_insignificant"
        assert result["min_baseline_magnitude"] == 0.10

    def test_operator_can_override_floor_per_metric(self):
        # Per-metric override path (the institutional one — different
        # rank columns have different units).
        _init_default_config({
            "use_skill_composite_target": True,
            "min_baseline_magnitude_by_rank": {"sortino_ratio": 0.01},
        })
        rows = [
            {"atr_multiplier": 2.0, "min_score": 70, "max_position_pct": 0.05,
             "sharpe_ratio": 0.55, "total_alpha": 0.30, "sortino_ratio": 0.5,
             "max_drawdown": -0.10, "total_trades": 80},
            # baseline_sortino = 0.02 — below the default 0.05 floor,
            # but above the per-metric override 0.01.
            {"atr_multiplier": 3.0, "min_score": 75, "max_position_pct": 0.10,
             "sharpe_ratio": 0.50, "total_alpha": 0.10, "sortino_ratio": 0.02,
             "max_drawdown": -0.05, "total_trades": 100},
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={}, current_params={
            "atr_multiplier": 3.0, "min_score": 75, "max_position_pct": 0.10,
        })
        assert result["status"] == "ok"
        assert result["min_baseline_magnitude"] == 0.01

    def test_significance_gate_fires_before_min_improvement_gate(self):
        # Ordering invariant: baseline_insignificant takes precedence
        # over no_improvement so the operator sees the right diagnosis
        # (baseline is noise, not just "improvement is too small").
        # baseline_sortino = 0.0001 (well below floor) + best_sortino
        # = 0.0002 (improvement_pct = ~99%, way above 10% gate). Pre-
        # gate this would have reported no_improvement-or-ok; post-
        # gate it reports baseline_insignificant.
        _init_default_config({"use_skill_composite_target": True})
        rows = [
            {"atr_multiplier": 2.0, "min_score": 70, "max_position_pct": 0.05,
             "sharpe_ratio": 0.55, "total_alpha": 0.30, "sortino_ratio": 0.0002,
             "max_drawdown": -0.10, "total_trades": 80},
            {"atr_multiplier": 3.0, "min_score": 75, "max_position_pct": 0.10,
             "sharpe_ratio": 0.50, "total_alpha": 0.10, "sortino_ratio": 0.0001,
             "max_drawdown": -0.05, "total_trades": 100},
        ]
        df = _make_sweep_df(rows)
        result = recommend(df, base_config={}, current_params={
            "atr_multiplier": 3.0, "min_score": 75, "max_position_pct": 0.10,
        })
        assert result["status"] == "baseline_insignificant"
        assert result["improvement_significant"] is False
        # The misleading 99× reading is still in the diagnostic
        # (clamped + visible), but the operator sees the
        # baseline_insignificant status and the improvement_delta tiny
        # number (0.0001), which together correctly tell the story.
        assert abs(result["improvement_delta"] - 0.0001) < 1e-9


# ── Walk-forward cross-validation (config#950) ───────────────────────────────


class TestRollingWindows:
    def test_rolling_windows_disjoint_test_sets_chronological(self):
        dates = [f"d{i:02d}" for i in range(30)]
        windows = _rolling_windows(dates, n_folds=3, test_frac=0.30)
        assert len(windows) == 3
        # Chronological (oldest fold first); test windows disjoint + advancing.
        test_sets = [te for _tr, te in windows]
        flat = [d for te in test_sets for d in te]
        assert flat == sorted(flat)  # ascending, no overlap
        # Anchored walk-forward: each train set is a prefix ending where its
        # test set begins (no look-ahead).
        for tr, te in windows:
            assert tr == dates[: dates.index(te[0])]

    def test_rolling_windows_last_fold_flush_to_end(self):
        dates = [f"d{i:02d}" for i in range(30)]
        windows = _rolling_windows(dates, n_folds=3, test_frac=0.30)
        assert windows[-1][1][-1] == dates[-1]  # most-recent data validated


def _date_aware_sim(stats_by_window):
    """Return a date-aware sim_fn mapping a window-key → stats. The window key
    is the test window's first+last date so each fold gets distinct stats."""
    def sim_fn(combo_config, dates=None):
        key = (dates[0], dates[-1]) if dates else "full"
        return stats_by_window.get(key, {"sharpe_ratio": 1.0, "sortino_ratio": 1.0})
    return sim_fn


class TestWalkForward:
    def _ok_result(self):
        return {
            "status": "ok",
            "recommended_params": {"atr_multiplier": 3.0},
            "fit_target": "sharpe_legacy",
            "best_sharpe": 1.0,
        }

    def test_all_folds_pass_records_advisory_consistency(self):
        dates = [f"d{i:02d}" for i in range(30)]
        sim = _date_aware_sim({})  # every fold returns sharpe 1.0 == train → ratio 100%
        result = validate_walk_forward(self._ok_result(), sim, dates, {})
        # No daily_returns / pbo_top_combos in this synthetic sim → PSR/DSR/PBO
        # all `insufficient` (non-blocking) → gate passes.
        assert result["holdout_passed"] is True
        assert result["status"] == "ok"
        wf = result["walk_forward"]
        # min_pass_fraction is now an ADVISORY secondary diagnostic.
        assert wf["n_passed"] == wf["n_gradeable"]
        assert wf["consistency_ok"] is True
        assert "promotion_gate" in result

    def test_failing_fold_is_advisory_not_blocking(self):
        """min_pass_fraction is DEMOTED (config#950): a failing fold lowers the
        advisory consistency flag but does NOT block promotion — the PSR/DSR/PBO
        gate is the decision, and here it is non-blocking (insufficient data)."""
        dates = [f"d{i:02d}" for i in range(30)]
        windows = _rolling_windows(dates, n_folds=3, test_frac=0.30)
        mid_key = (windows[1][1][0], windows[1][1][-1])
        sim = _date_aware_sim({mid_key: {"sharpe_ratio": 0.2, "sortino_ratio": 0.2}})
        result = validate_walk_forward(self._ok_result(), sim, dates, {})
        # Advisory consistency reflects the fold failure...
        assert result["walk_forward"]["consistency_ok"] is False
        # ...but promotion is NOT blocked by it (gate sub-stats insufficient).
        assert result["holdout_passed"] is True
        assert result["status"] == "ok"

    def test_promotion_gate_blocks_when_a_sub_gate_fails(self):
        """A computable sub-gate failure (here DSR) blocks promotion regardless
        of fold consistency."""
        import numpy as np
        import optimizer.executor_optimizer as eo
        dates = pd.date_range("2025-01-01", periods=240, freq="B").strftime("%Y-%m-%d").tolist()

        def sim(combo_config, dates=None):  # noqa: D401
            rng = np.random.RandomState(7)
            r = pd.Series(rng.normal(0.0003, 0.012, len(dates)), index=pd.to_datetime(dates))
            sh = r.mean() / r.std() * np.sqrt(252)
            return {"status": "ok", "daily_returns": r,
                    "sharpe_ratio": float(sh), "sortino_ratio": float(sh)}

        res = dict(self._ok_result())
        res["n_combos_swept"] = 400  # heavy deflation → DSR well below 0.90
        res["pbo_top_combos"] = [{"atr_multiplier": x} for x in (3.0, 2.5, 3.5, 2.0)]
        out = validate_walk_forward(res, sim, dates, {}, n_folds=3, test_frac=0.30)
        assert out["holdout_passed"] is False
        assert out["status"] == "holdout_failed"
        assert out["promotion_gate"]["sub_gates"]["dsr"]["status"] == "ok"

    def test_non_date_aware_sim_falls_back_to_single_holdout(self):
        dates = [f"d{i:02d}" for i in range(30)]

        def legacy_sim(combo_config):  # no dates kwarg
            return {"sharpe_ratio": 1.0, "sortino_ratio": 1.0}

        result = validate_walk_forward(self._ok_result(), legacy_sim, dates, {})
        assert "walk_forward_degraded" in result
        # Fell back to single-window holdout, which still grades.
        assert "holdout_passed" in result


class TestHoldoutDateWindowing:
    def test_holdout_runs_on_held_out_window_when_date_aware(self):
        """Regression: the held-out 30% must actually be simulated, not the
        full range (the pre-fix latent bug)."""
        dates = [f"d{i:02d}" for i in range(30)]
        seen = {}

        def sim_fn(combo_config, dates=None):
            seen["dates"] = dates
            return {"sharpe_ratio": 1.0, "sortino_ratio": 1.0}

        result = {
            "status": "ok", "recommended_params": {"atr_multiplier": 3.0},
            "fit_target": "sharpe_legacy", "best_sharpe": 1.0,
        }
        validate_holdout(result, sim_fn, dates, {})
        # Only the last 30% (≈9 dates) should be simulated.
        assert seen["dates"] == dates[int(len(dates) * 0.7):]
        assert "holdout_degraded" not in result

    def test_legacy_sim_marks_degraded(self):
        dates = [f"d{i:02d}" for i in range(30)]

        def legacy_sim(combo_config):
            return {"sharpe_ratio": 1.0, "sortino_ratio": 1.0}

        result = {
            "status": "ok", "recommended_params": {"atr_multiplier": 3.0},
            "fit_target": "sharpe_legacy", "best_sharpe": 1.0,
        }
        validate_holdout(result, legacy_sim, dates, {})
        assert "holdout_degraded" in result
