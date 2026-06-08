"""
Tests for BacktesterPreflight mode composition.

The BasePreflight primitives (check_env_vars / check_s3_bucket) are
tested in alpha-engine-lib. These tests only verify that each mode
calls the expected primitives in the expected order.

Data-freshness checks (universe + macro/SPY) moved upstream to
alpha-engine-data's preflight 2026-05-05; the data step in the Saturday
SF hard-fails on staleness before the backtester runs, so re-checking
here is redundant.
"""

from __future__ import annotations

import sys
import os
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preflight import BacktesterPreflight


class TestBacktesterPreflight:
    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="unknown mode"):
            BacktesterPreflight(bucket="b", mode="bogus")

    def test_backtest_mode_runs_env_and_artifact_checks(self):
        """backtest mode: env + S3 + lib version + imports + vectorized
        signal extraction + predictor weights + executor config. No
        data-freshness primitives — those moved upstream."""
        pf = BacktesterPreflight(bucket="b", mode="backtest")
        with patch.object(pf, "check_env_vars") as env, \
             patch.object(pf, "check_s3_bucket") as s3, \
             patch.object(pf, "check_arcticdb_fresh") as adb, \
             patch.object(pf, "_check_artifact_contract") as contract, \
             patch.object(pf, "_check_executor_config") as exec_cfg, \
             patch.object(pf, "_check_lib_version") as lib_v, \
             patch.object(pf, "_check_imports") as imports, \
             patch.object(pf, "_check_vectorized_signal_extraction") as vec, \
             patch.object(pf, "_check_predictor_weights") as pred_w:
            pf.run()
        env.assert_called_once_with("AWS_REGION")
        s3.assert_called_once()
        adb.assert_not_called()
        contract.assert_called_once()
        exec_cfg.assert_called_once()
        lib_v.assert_called_once()
        imports.assert_called_once()
        vec.assert_called_once()
        pred_w.assert_called_once()

    def test_artifact_contract_check_passes_on_real_manifest(self):
        """The shipped manifest's contract holds for the SF mode → no raise."""
        BacktesterPreflight(bucket="b", mode="backtest")._check_artifact_contract()

    def test_artifact_contract_check_raises_on_violation(self, monkeypatch):
        """A manifest contract violation must fail loud at preflight (L4513
        guard), not 2h into the spot run."""
        import pipeline_manifest
        monkeypatch.setattr(
            pipeline_manifest, "contract_violations",
            lambda mode: [f"--mode={mode}: orphaned sweep_df.parquet (L4513)"],
        )
        pf = BacktesterPreflight(bucket="b", mode="backtest")
        with pytest.raises(RuntimeError, match="artifact-contract violation"):
            pf._check_artifact_contract()

    def test_evaluate_mode_runs_env_and_s3_only(self):
        pf = BacktesterPreflight(bucket="b", mode="evaluate")
        with patch.object(pf, "check_env_vars") as env, \
             patch.object(pf, "check_s3_bucket") as s3, \
             patch.object(pf, "check_arcticdb_fresh") as adb, \
             patch.object(pf, "_check_executor_config") as exec_cfg, \
             patch.object(pf, "_check_lib_version") as lib_v, \
             patch.object(pf, "_check_imports") as imports, \
             patch.object(pf, "_check_predictor_weights") as pred_w:
            pf.run()
        env.assert_called_once_with("AWS_REGION")
        s3.assert_called_once()
        adb.assert_not_called()
        exec_cfg.assert_not_called()
        lib_v.assert_not_called()
        imports.assert_not_called()
        pred_w.assert_not_called()

    def test_lambda_health_mode_runs_env_and_s3_only(self):
        pf = BacktesterPreflight(bucket="b", mode="lambda_health")
        with patch.object(pf, "check_env_vars") as env, \
             patch.object(pf, "check_s3_bucket") as s3, \
             patch.object(pf, "check_arcticdb_fresh") as adb, \
             patch.object(pf, "_check_executor_config") as exec_cfg, \
             patch.object(pf, "_check_lib_version") as lib_v, \
             patch.object(pf, "_check_imports") as imports, \
             patch.object(pf, "_check_predictor_weights") as pred_w:
            pf.run()
        env.assert_called_once_with("AWS_REGION")
        s3.assert_called_once()
        adb.assert_not_called()
        exec_cfg.assert_not_called()
        lib_v.assert_not_called()
        imports.assert_not_called()
        pred_w.assert_not_called()

    def test_no_mode_calls_data_freshness_primitives(self):
        """Regression: no mode may call macro freshness or universe
        freshness scan. Both moved upstream to alpha-engine-data."""
        for mode in ("backtest", "evaluate", "lambda_health"):
            pf = BacktesterPreflight(bucket="b", mode=mode)
            with patch.object(pf, "check_env_vars"), \
                 patch.object(pf, "check_s3_bucket"), \
                 patch.object(pf, "check_arcticdb_fresh") as adb, \
                 patch.object(pf, "_check_executor_config"), \
                 patch.object(pf, "_check_lib_version"), \
                 patch.object(pf, "_check_imports"), \
                 patch.object(pf, "_check_vectorized_signal_extraction"), \
                 patch.object(pf, "_check_predictor_weights"):
                pf.run()
            adb.assert_not_called()


class TestExecutorConfigCheck:
    """
    _check_executor_config loads the executor's risk.yaml via the same
    canonical search order executor/config_loader.py uses. Covers the
    2026-04-20 silent-fallback-to-placeholder incident.

    All tests patch HOME to an empty tmp dir so the ~-anchored search
    path doesn't resolve to the developer's real alpha-engine-config.
    """

    @pytest.fixture
    def isolated_home(self, tmp_path, monkeypatch):
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        return fake_home

    def test_no_executor_paths_skips_check(self, isolated_home):
        # Defense-in-depth with executor-side hard-fail — if caller
        # doesn't pass executor_paths, skip (no exception, no IO).
        pf = BacktesterPreflight(bucket="alpha-engine-research", mode="backtest")
        pf._check_executor_config()  # no exception

    def test_executor_root_missing_skips_check(self, tmp_path, isolated_home):
        pf = BacktesterPreflight(
            bucket="alpha-engine-research",
            mode="backtest",
            executor_paths=[str(tmp_path / "does-not-exist")],
        )
        pf._check_executor_config()  # no exception

    def test_risk_yaml_missing_raises(self, tmp_path, isolated_home):
        executor_root = tmp_path / "alpha-engine"
        executor_root.mkdir()
        (executor_root / "config").mkdir()
        # No risk.yaml in any of the three canonical locations.
        pf = BacktesterPreflight(
            bucket="alpha-engine-research",
            mode="backtest",
            executor_paths=[str(executor_root)],
        )
        with pytest.raises(RuntimeError, match="executor risk.yaml not found"):
            pf._check_executor_config()

    def test_placeholder_bucket_raises(self, tmp_path, isolated_home):
        executor_root = tmp_path / "alpha-engine"
        (executor_root / "config").mkdir(parents=True)
        (executor_root / "config" / "risk.yaml").write_text(
            "signals_bucket: your-research-bucket-name\n"
            "trades_bucket: your-executor-bucket-name\n"
        )
        pf = BacktesterPreflight(
            bucket="alpha-engine-research",
            mode="backtest",
            executor_paths=[str(executor_root)],
        )
        with pytest.raises(RuntimeError, match="placeholder"):
            pf._check_executor_config()

    def test_bucket_mismatch_raises(self, tmp_path, isolated_home):
        executor_root = tmp_path / "alpha-engine"
        (executor_root / "config").mkdir(parents=True)
        (executor_root / "config" / "risk.yaml").write_text(
            "signals_bucket: different-bucket\ntrades_bucket: different-bucket\n"
        )
        pf = BacktesterPreflight(
            bucket="alpha-engine-research",
            mode="backtest",
            executor_paths=[str(executor_root)],
        )
        with pytest.raises(RuntimeError, match="does not match"):
            pf._check_executor_config()

    def test_matching_real_config_passes(self, tmp_path, isolated_home):
        executor_root = tmp_path / "alpha-engine"
        (executor_root / "config").mkdir(parents=True)
        (executor_root / "config" / "risk.yaml").write_text(
            "signals_bucket: alpha-engine-research\n"
            "trades_bucket: alpha-engine-research\n"
        )
        pf = BacktesterPreflight(
            bucket="alpha-engine-research",
            mode="backtest",
            executor_paths=[str(executor_root)],
        )
        pf._check_executor_config()  # no exception

    def test_empty_value_raises(self, tmp_path, isolated_home):
        executor_root = tmp_path / "alpha-engine"
        (executor_root / "config").mkdir(parents=True)
        (executor_root / "config" / "risk.yaml").write_text(
            "signals_bucket: ''\ntrades_bucket: alpha-engine-research\n"
        )
        pf = BacktesterPreflight(
            bucket="alpha-engine-research",
            mode="backtest",
            executor_paths=[str(executor_root)],
        )
        with pytest.raises(RuntimeError, match="empty value"):
            pf._check_executor_config()
