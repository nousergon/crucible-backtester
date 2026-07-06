"""Fail-loud keyed-config-block guard (config#1842).

The bug class: a config.yaml block keyed by canonical names silently drifts
when the canonical names are renamed — every ``configured.get(canonical_key)``
resolves to 0.0 / an inert fallback and the feature runs as a zero-filled
no-op for months. Live occurrence: ``weight_optimizer.default_weights``
carried pre-2026-03-29 ``news``/``research`` keys (canonical:
``quant``/``qual``), so every weight proposal was measured against a phantom
0.0 baseline and permanently tripped ``max_single_change`` —
``config/scoring_weights.json`` was never written.

These tests pin the guard: a stale-keys block RAISES ``ConfigKeyDriftError``
at the read chokepoint instead of zero-filling.
"""
import pytest

from optimizer.config_guards import ConfigKeyDriftError, validate_keyed_block
from optimizer import weight_optimizer
from optimizer.weight_optimizer import (
    SUB_SCORES,
    _configured_horizon_blend,
    compute_weights,
    configured_default_weights,
    init_config,
)


# ── validate_keyed_block primitive ───────────────────────────────────────────


class TestValidateKeyedBlock:

    def test_absent_block_is_valid(self):
        validate_keyed_block(None, ["a", "b"], config_path="x.y")

    def test_exact_match_passes(self):
        validate_keyed_block({"a": 1, "b": 2}, ["a", "b"], config_path="x.y")

    def test_unknown_key_raises(self):
        with pytest.raises(ConfigKeyDriftError, match="unknown key"):
            validate_keyed_block({"a": 1, "z": 2}, ["a", "b"], config_path="x.y")

    def test_missing_key_raises_in_exact_mode(self):
        with pytest.raises(ConfigKeyDriftError, match="missing"):
            validate_keyed_block({"a": 1}, ["a", "b"], config_path="x.y")

    def test_subset_mode_allows_missing_keys(self):
        validate_keyed_block(
            {"a": 1}, ["a", "b"], config_path="x.y", allow_subset=True,
        )

    def test_subset_mode_still_rejects_unknown_keys(self):
        with pytest.raises(ConfigKeyDriftError, match="unknown key"):
            validate_keyed_block(
                {"z": 1}, ["a", "b"], config_path="x.y", allow_subset=True,
            )

    def test_error_message_names_config_path_and_canonical(self):
        with pytest.raises(ConfigKeyDriftError) as exc:
            validate_keyed_block(
                {"news": 0.5, "research": 0.5}, SUB_SCORES,
                config_path="weight_optimizer.default_weights",
            )
        msg = str(exc.value)
        assert "weight_optimizer.default_weights" in msg
        assert "news" in msg and "research" in msg
        assert "quant" in msg and "qual" in msg


# ── weight_optimizer.default_weights guard (the config#1842 case) ────────────


_STALE_KEYS_CFG = {
    "weight_optimizer": {
        # The exact stale block the live config carried until config PR #1848.
        "default_weights": {"news": 0.50, "research": 0.50},
    }
}


class TestDefaultWeightsGuard:

    def teardown_method(self):
        init_config({})

    def test_stale_keys_config_raises(self):
        """The config#1842 closes-when: a stale-keys default_weights block
        raises instead of silently zero-filling quant/qual to 0.0."""
        init_config(_STALE_KEYS_CFG)
        with pytest.raises(ConfigKeyDriftError, match="default_weights"):
            configured_default_weights()

    def test_compute_weights_raises_on_stale_config_fallback(self):
        """compute_weights(current_weights=None) resolves through the guarded
        chokepoint — a stale block fails the run loudly (isolated per-module
        by tracker.run_module) rather than proposing against a 0.0 baseline."""
        import pandas as pd
        init_config(_STALE_KEYS_CFG)
        with pytest.raises(ConfigKeyDriftError):
            compute_weights(pd.DataFrame(), current_weights=None)

    def test_compute_weights_rejects_stale_caller_provided_weights(self):
        """Defense-in-depth: drifted keys from ANY caller path (S3 legacy
        object, universe.yaml) are rejected at the compute chokepoint."""
        import pandas as pd
        init_config({})
        with pytest.raises(ConfigKeyDriftError):
            compute_weights(
                pd.DataFrame(), current_weights={"news": 0.5, "research": 0.5},
            )

    def test_correct_keys_pass_and_are_copied(self):
        init_config({"weight_optimizer": {"default_weights": {"quant": 0.6, "qual": 0.4}}})
        weights = configured_default_weights()
        assert weights == {"quant": 0.6, "qual": 0.4}
        weights["quant"] = 0.0  # mutating the copy must not touch config
        assert configured_default_weights() == {"quant": 0.6, "qual": 0.4}

    def test_absent_block_falls_back_to_code_default(self):
        init_config({})
        assert configured_default_weights() == {"quant": 0.50, "qual": 0.50}


# ── weight_optimizer.horizon_blend guard (same class, bit 2026-07-01) ────────


class TestHorizonBlendGuard:

    def teardown_method(self):
        init_config({})

    def test_retired_horizon_keys_raise(self):
        """The pre-canonical beat_spy_10d/beat_spy_30d keys — which silently
        made the tuned 60/40 blend inert until 2026-07-01 — now raise."""
        init_config({
            "weight_optimizer": {
                "horizon_blend": {"beat_spy_10d": 0.60, "beat_spy_30d": 0.40},
            }
        })
        with pytest.raises(ConfigKeyDriftError, match="horizon_blend"):
            _configured_horizon_blend()

    def test_canonical_keys_pass(self):
        init_config({
            "weight_optimizer": {
                "horizon_blend": {
                    weight_optimizer._SHORT_OUTCOME: 0.60,
                    weight_optimizer._LONG_OUTCOME: 0.40,
                }
            }
        })
        blend = _configured_horizon_blend()
        assert blend[weight_optimizer._SHORT_OUTCOME] == 0.60

    def test_absent_block_falls_back_to_code_default(self):
        init_config({})
        assert _configured_horizon_blend() == weight_optimizer._HORIZON_BLEND


# ── executor_optimizer.min_baseline_magnitude_by_rank guard (subset mode) ────


class TestExecutorByRankOverrideGuard:

    def teardown_method(self):
        from optimizer import executor_optimizer
        executor_optimizer.init_config({})

    def test_unknown_rank_metric_raises(self):
        from optimizer import executor_optimizer
        executor_optimizer.init_config({
            "executor_optimizer": {
                # e.g. a renamed/typo'd rank metric — would be silently ignored
                "min_baseline_magnitude_by_rank": {"sortino": 0.1},
            }
        })
        with pytest.raises(ConfigKeyDriftError, match="min_baseline_magnitude_by_rank"):
            executor_optimizer._resolve_min_baseline_magnitude("sortino_ratio")

    def test_known_subset_override_honoured(self):
        from optimizer import executor_optimizer
        executor_optimizer.init_config({
            "executor_optimizer": {
                "min_baseline_magnitude_by_rank": {"sortino_ratio": 0.123},
            }
        })
        assert executor_optimizer._resolve_min_baseline_magnitude("sortino_ratio") == 0.123
        # Metrics not named in the subset fall through to the code default.
        assert executor_optimizer._resolve_min_baseline_magnitude("sharpe_ratio") == 0.05


# ── evaluate._read_current_weights fallback path ─────────────────────────────


class TestReadCurrentWeightsGuard:

    def teardown_method(self):
        init_config({})

    def test_stale_universe_yaml_scoring_weights_raise(self, tmp_path, monkeypatch):
        """A drifted scoring_weights block in research's universe.yaml is
        rejected at the evaluate read site instead of flowing downstream."""
        import evaluate as evaluate_mod

        research_dir = tmp_path / "research"
        (research_dir / "config").mkdir(parents=True)
        (research_dir / "config" / "universe.yaml").write_text(
            "scoring_weights:\n  news: 0.5\n  research: 0.5\n"
        )
        init_config({})
        # S3 path must fail so the yaml fallback is reached.
        fake_s3 = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
        fake_s3.get_object.side_effect = Exception("no creds in tests")
        monkeypatch.setattr(
            evaluate_mod.boto3, "client", lambda *a, **k: fake_s3,
        )
        with pytest.raises(ConfigKeyDriftError, match="universe.yaml"):
            evaluate_mod._read_current_weights({"research_paths": [str(research_dir)]})

    def test_fallback_uses_validated_chokepoint(self, monkeypatch):
        """With no S3 and no universe.yaml, the final fallback resolves via
        configured_default_weights() — stale keys raise."""
        import evaluate as evaluate_mod

        init_config(_STALE_KEYS_CFG)
        fake_s3 = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
        fake_s3.get_object.side_effect = Exception("no creds in tests")
        monkeypatch.setattr(evaluate_mod.boto3, "client", lambda *a, **k: fake_s3)
        with pytest.raises(ConfigKeyDriftError, match="default_weights"):
            evaluate_mod._read_current_weights({"research_paths": []})
