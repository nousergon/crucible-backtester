"""End-to-end integration test for the optimizer-artifact-assembler arc.

PR 7 of the arc — closes the test surface by simulating today's
2026-05-09 chain end-to-end across all 5 modules touched by the arc:

    backtester.executor_optimizer.apply()  →  artifact + legacy live write
    evaluator.executor_optimizer.apply()   →  no-op (current=best post-backtester)
    assembler.assemble()                   →  reads artifacts, writes assembled audit
    regression_monitor.check_regression()  →  detects regression vs baseline
    rollback.rollback_all()                →  reverts live to _previous
    regression_monitor.write_rollback_audit() → captures rejected recommendations

The test uses an ``InMemoryS3`` simulator so each module exercises its
real code path against a coherent fake S3. The killer property is the
final assertion: the rollback_audit artifact's
``rejected_recommendations`` section contains the executor_optimizer's
exact recommended_params — the same payload the rollback discarded. This
is the forensic question that motivated the arc: "which optimizer set
``atr_multiplier`` to 3.0 today?" becomes a single S3 read.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from optimizer.assembler import assemble, set_cutover_enabled
from optimizer.executor_optimizer import apply as executor_apply
from optimizer.executor_optimizer import init_config as executor_init_config
from optimizer.regression_monitor import check_regression


# ── In-memory S3 simulator ───────────────────────────────────────────────────


class InMemoryS3:
    """Dict-backed S3 client supporting put_object / get_object /
    list_objects_v2 / copy_object — the four operations used across the
    arc's modules. NoSuchKey raises a botocore ClientError that callers
    already handle."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        # Per-method call counters for assertions.
        self.put_count = 0
        self.copy_count = 0

    def put_object(self, Bucket: str, Key: str, Body: Any, **kwargs):
        if isinstance(Body, str):
            Body = Body.encode("utf-8")
        elif isinstance(Body, bytes):
            pass
        else:
            Body = json.dumps(Body).encode("utf-8")
        self.objects[(Bucket, Key)] = Body
        self.put_count += 1
        return {}

    def get_object(self, Bucket: str, Key: str):
        if (Bucket, Key) not in self.objects:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey"}}, "GetObject",
            )
        body = self.objects[(Bucket, Key)]
        return {"Body": MagicMock(read=lambda b=body: b)}

    def list_objects_v2(self, Bucket: str, Prefix: str = "", **kwargs):
        contents = [
            {"Key": k}
            for (b, k) in self.objects
            if b == Bucket and k.startswith(Prefix)
        ]
        return {"Contents": contents} if contents else {}

    def copy_object(self, Bucket: str, CopySource: dict, Key: str, **kwargs):
        src = (CopySource["Bucket"], CopySource["Key"])
        if src not in self.objects:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey"}}, "CopyObject",
            )
        self.objects[(Bucket, Key)] = self.objects[src]
        self.copy_count += 1
        return {}

    def read_json(self, Bucket: str, Key: str) -> dict | None:
        """Test convenience: read + parse a JSON artifact directly."""
        try:
            obj = self.get_object(Bucket=Bucket, Key=Key)
            return json.loads(obj["Body"].read())
        except ClientError:
            return None


@pytest.fixture
def s3() -> InMemoryS3:
    """Fresh in-memory S3 per test."""
    return InMemoryS3()


@pytest.fixture(autouse=True)
def _reset_cutover_flag():
    set_cutover_enabled(False)
    yield
    set_cutover_enabled(False)


# ── End-to-end test — today's 2026-05-09 chain reproducible ───────────────


class TestAssemblerArcEndToEnd:
    """Exercises the full chain across all 5 modules touched by the arc.

    The chain mirrors today's 2026-05-09 Sat SF observed timeline:
        09:23:47 — Backtester apply() writes legacy live + history + artifact
        09:29:32 — Evaluator apply() no-ops (current=best); assembler runs
                   (shadow); regression detected → rollback fires; audit
                   artifact captures rejected recommendations
    """

    BUCKET = "alpha-engine-research"
    RUN_DATE = "2026-05-09"

    def _seed_prior_live_config(self, s3: InMemoryS3) -> None:
        """Simulate the pre-2026-05-09 state: live = 5/7 atr 2.0 data."""
        prior_live = {
            "atr_multiplier": 2.0,
            "min_score": 75,
            "max_position_pct": 0.10,
            "time_decay_exit_days": 10,
            "updated_at": "2026-05-07",
        }
        s3.put_object(
            Bucket=self.BUCKET, Key="config/executor_params.json",
            Body=json.dumps(prior_live),
        )

    def _make_recommendation_result(self) -> dict:
        """Today's executor_optimizer recommendation — atr 3.0 promotion
        with the alpha-misaligned profile that made the system flag
        regression."""
        return {
            "status": "ok",
            "fit_target": "sharpe_legacy",
            "recommended_params": {
                "atr_multiplier": 3.0,
                "min_score": 75,
                "max_position_pct": 0.10,
                "time_decay_exit_days": 15,
            },
            "best_sharpe": 0.6842,
            "best_alpha": -2.5881,
            "best_sortino": 0.7,
            "improvement_pct": 0.063,
            "n_combos_tested": 60,
            "apply_result": {"applied": True},
        }

    def _saved_baseline(self) -> dict:
        """Promotion baseline reflecting prior weeks' performance — high
        enough that today's portfolio_stats trip the 20% Sortino-drop gate.
        Fresh (saved_at within the 21-day max age vs RUN_DATE 2026-05-09) so
        the stale-baseline guard does not pre-empt the regression check."""
        return {
            "sortino_ratio": 1.0,       # baseline (primary risk-adjusted gate)
            "sharpe_ratio": 1.0,
            "accuracy_21d": 0.62,
            "saved_at": "2026-05-05",
        }

    def _todays_metrics(self) -> dict:
        """Today's actual portfolio metrics — Sortino collapsed enough to
        trip the 20% drop threshold, with adequate sample sizes so the
        min-sample guard does not suppress the rollback."""
        return {
            "sortino_ratio": 0.241,      # 75% drop from baseline
            "sharpe_ratio": 0.241,
            "accuracy_21d": 0.60,
            "total_trades": 80,
            "n_signals": 80,
        }

    def test_full_chain_emits_audit_with_rejected_recommendations(self, s3):
        """The arc-closing property: after backtester writes, evaluator
        no-ops, assembler assembles, regression detected, rollback fires
        → audit artifact captures the executor_optimizer recommendation
        that the rollback discarded.

        Single S3 read at ``config/rollback_audit/{date}.json`` answers
        "which optimizer set atr_multiplier to 3.0?" — replacing the
        4-key + 3-module forensic the live system required pre-arc.
        """
        self._seed_prior_live_config(s3)

        # ── Step 1: Backtester runs executor_optimizer.apply() ───────────
        # Writes: artifact + legacy live key (atr 3.0) + history.
        executor_init_config({
            "executor_optimizer": {
                # config#1053 Phase C: the legacy live write is now opt-in; enable
                # it so this assembler-chain e2e exercises the live-write path.
                "legacy_executor_params_live_apply": True,
                "min_valid_combos": 0, "min_sharpe_improvement": 0.05,
                "min_trades_to_promote": 0, "drawdown_penalty_weight": 0.5,
            },
        })

        # Pin today_iso to RUN_DATE so the artifact key the test asserts on
        # matches regardless of UTC clock at run time. recommendation_artifact
        # exposes today_iso() specifically as a patch seam — without this,
        # CI runs that cross the UTC midnight boundary land the artifact at
        # `{tomorrow}/...` while the assertion looks for `{RUN_DATE}/...`.
        with patch("optimizer.executor_optimizer.boto3") as exec_boto3, \
             patch("optimizer.recommendation_artifact.boto3") as art_boto3, \
             patch("optimizer.rollback.boto3") as rb_boto3, \
             patch("optimizer.recommendation_artifact.today_iso",
                   return_value=self.RUN_DATE):
            exec_boto3.client.return_value = s3
            art_boto3.client.return_value = s3
            rb_boto3.client.return_value = s3
            executor_apply(self._make_recommendation_result(), bucket=self.BUCKET)

        # Verify backtester's writes landed.
        live_after_backtester = s3.read_json(
            self.BUCKET, "config/executor_params.json",
        )
        assert live_after_backtester is not None
        assert live_after_backtester["atr_multiplier"] == 3.0

        artifact_key = (
            f"config/executor_params/recommendations/{self.RUN_DATE}/"
            f"from_executor_optimizer.json"
        )
        artifact = s3.read_json(self.BUCKET, artifact_key)
        assert artifact is not None
        assert artifact["recommended_params"]["atr_multiplier"] == 3.0
        assert artifact["promotion_intent"] == "promote"
        assert artifact["fit_target"] == "sharpe_legacy"

        previous_after_backtester = s3.read_json(
            self.BUCKET, "config/executor_params_previous.json",
        )
        assert previous_after_backtester is not None
        assert previous_after_backtester["atr_multiplier"] == 2.0  # 5/7 snapshot

        # ── Step 2: Evaluator runs assembler.assemble() (shadow) ─────────
        with patch("optimizer.assembler.boto3") as asm_boto3, \
             patch("optimizer.recommendation_artifact.boto3") as art_boto3:
            asm_boto3.client.return_value = s3
            art_boto3.client.return_value = s3
            assemble_result = assemble(
                bucket=self.BUCKET, config_type="executor_params",
                run_date=self.RUN_DATE, write_assembled=True,
                cutover_enabled=False,  # shadow mode for this test
            )

        assert assemble_result.status == "ok"
        # Assembled audit captures executor_optimizer's full_replace
        # contribution. Canonical lib v0.8.0 layout — pull from
        # latest.json sidecar (the runner mirrors body to that key).
        assembled = s3.read_json(
            self.BUCKET, "config/executor_params/assembled/latest.json",
        )
        assert assembled is not None
        assert assembled["assembled_params"]["atr_multiplier"] == 3.0
        assert (
            assembled["merge_summary"]["atr_multiplier"]["writer"]
            == "executor_optimizer"
        )
        # Live key is unchanged from backtester's write (shadow mode).
        live_after_shadow = s3.read_json(
            self.BUCKET, "config/executor_params.json",
        )
        assert live_after_shadow["atr_multiplier"] == 3.0

        # ── Step 3: regression_monitor detects regression → rollback ─────
        # Saved baseline is from prior weeks; today's portfolio_stats trip
        # the 20% Sortino drop gate → rollback_all reverts live to _previous.
        baseline = self._saved_baseline()
        with patch("optimizer.regression_monitor.boto3") as reg_boto3, \
             patch("optimizer.rollback.boto3") as rb_boto3, \
             patch("optimizer.assembler.boto3") as asm_boto3, \
             patch("optimizer.recommendation_artifact.boto3") as art_boto3, \
             patch("optimizer.regression_monitor._load_baseline",
                   return_value=baseline):
            reg_boto3.client.return_value = s3
            rb_boto3.client.return_value = s3
            asm_boto3.client.return_value = s3
            art_boto3.client.return_value = s3
            regression_result = check_regression(
                bucket=self.BUCKET,
                current_metrics=self._todays_metrics(),
                run_date=self.RUN_DATE,
            )

        # ── Step 4: Verify rollback fired + reverted live to 5/7 data ────
        assert regression_result["regression_detected"] is True
        assert regression_result["rollback_triggered"] is True
        assert regression_result["details"]["sortino_drop_pct"] > 0.20

        live_post_rollback = s3.read_json(
            self.BUCKET, "config/executor_params.json",
        )
        # Reverted to 5/7 atr 2.0 data via rollback_all → copy _previous → live.
        assert live_post_rollback["atr_multiplier"] == 2.0
        assert live_post_rollback["updated_at"] == "2026-05-07"

        # ── Step 5: KILLER PROPERTY — audit captures rejected recommendation
        audit_key = regression_result["rollback_audit_key"]
        assert audit_key == f"config/rollback_audit/{self.RUN_DATE}.json"
        audit = s3.read_json(self.BUCKET, audit_key)
        assert audit is not None

        # Audit captures the trigger.
        assert audit["trigger"]["regression_detected"] is True
        assert audit["trigger"]["details"]["sortino_drop_pct"] > 0.20

        # Audit captures the rolled-back configs.
        rolled_back_executor = next(
            r for r in audit["rollback_results"]
            if r.get("config_type") == "executor_params"
        )
        assert rolled_back_executor["rolled_back"] is True

        # Audit captures the rejected per-optimizer recommendations —
        # this is the forensic-killer feature. Pre-arc: required reading
        # 4 S3 keys + tracing 3 modules. Post-arc: single dict lookup.
        rejected = audit["rejected_recommendations"]
        assert "executor_params" in rejected
        rejected_executor = rejected["executor_params"]["from_optimizers"][
            "executor_optimizer"
        ]
        assert rejected_executor["recommended_params"]["atr_multiplier"] == 3.0
        assert rejected_executor["promotion_intent"] == "promote"
        assert rejected_executor["fit_target"] == "sharpe_legacy"
        # Diagnostic from the rejected recommendation is preserved —
        # answers "what numbers led to the alpha-misaligned promotion?"
        assert rejected_executor["diagnostic"]["best_alpha"] == -2.5881

        # Audit captures the assembled output that the rollback discarded.
        rejected_assembled = rejected["executor_params"]["assembled"]
        assert rejected_assembled["assembled_params"]["atr_multiplier"] == 3.0
        assert (
            rejected_assembled["merge_summary"]["atr_multiplier"]["writer"]
            == "executor_optimizer"
        )

    def test_no_regression_no_rollback_no_audit(self, s3):
        """Negative-control: when current metrics match baseline (no
        regression), neither rollback nor audit emission fires. Live key
        keeps the backtester's promotion."""
        self._seed_prior_live_config(s3)

        executor_init_config({
            "executor_optimizer": {
                # config#1053 Phase C: the legacy live write is now opt-in; enable
                # it so this assembler-chain e2e exercises the live-write path.
                "legacy_executor_params_live_apply": True,
                "min_valid_combos": 0, "min_sharpe_improvement": 0.05,
                "min_trades_to_promote": 0, "drawdown_penalty_weight": 0.5,
            },
        })

        # Pin today_iso to RUN_DATE so the artifact key the test asserts on
        # matches regardless of UTC clock at run time. recommendation_artifact
        # exposes today_iso() specifically as a patch seam — without this,
        # CI runs that cross the UTC midnight boundary land the artifact at
        # `{tomorrow}/...` while the assertion looks for `{RUN_DATE}/...`.
        with patch("optimizer.executor_optimizer.boto3") as exec_boto3, \
             patch("optimizer.recommendation_artifact.boto3") as art_boto3, \
             patch("optimizer.rollback.boto3") as rb_boto3, \
             patch("optimizer.recommendation_artifact.today_iso",
                   return_value=self.RUN_DATE):
            exec_boto3.client.return_value = s3
            art_boto3.client.return_value = s3
            rb_boto3.client.return_value = s3
            executor_apply(self._make_recommendation_result(), bucket=self.BUCKET)

        # Today's metrics ~= baseline → no regression
        baseline = {"sharpe_ratio": 0.7, "accuracy_21d": 0.61}
        stable_metrics = {"sharpe_ratio": 0.69, "accuracy_21d": 0.60}
        with patch("optimizer.regression_monitor.boto3") as reg_boto3, \
             patch("optimizer.rollback.boto3") as rb_boto3, \
             patch("optimizer.regression_monitor._load_baseline",
                   return_value=baseline):
            reg_boto3.client.return_value = s3
            rb_boto3.client.return_value = s3
            regression_result = check_regression(
                bucket=self.BUCKET, current_metrics=stable_metrics,
                run_date=self.RUN_DATE,
            )

        assert regression_result["regression_detected"] is False
        assert regression_result.get("rollback_triggered") is False
        assert "rollback_audit_key" not in regression_result

        # Live key still has backtester's atr 3.0 promotion (not rolled back).
        live = s3.read_json(self.BUCKET, "config/executor_params.json")
        assert live["atr_multiplier"] == 3.0

        # No audit artifact for this date.
        audit = s3.read_json(
            self.BUCKET, f"config/rollback_audit/{self.RUN_DATE}.json",
        )
        assert audit is None
