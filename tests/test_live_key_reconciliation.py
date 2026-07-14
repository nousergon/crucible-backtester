"""live_key_reconciliation — post-optimizer live-key freshness check
(config#2332).

Pins: (1) a loop whose apply_audit outcome is "promoted" this run but whose
live S3 key predates run_start pages via ops_alerts at severity="critical";
(2) a missing live key on a promoted loop also pages; (3) loops that were
never promoted (or not promoted THIS run) are never reconciled — "absence
is correct" for scoring_weights/predictor_params before their first apply;
(4) a fresh live key (LastModified >= run_start) is silent; (5) a
head_object error OTHER than 404 raises (fail-loud — a check that could not
be performed must not read as "checked, clean"); (6) end-to-end wiring:
build_audit's real output, fed through run_reconciliation, pages within the
same call when a promoted loop's key is stale.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from optimizer.apply_audit import build_audit
from optimizer.live_key_reconciliation import (
    LIVE_KEYS,
    ReconciliationFinding,
    reconcile_promoted_live_keys,
    run_reconciliation,
)

RUN_START = datetime(2026, 7, 12, 14, 0, 0, tzinfo=timezone.utc)


def _audit(loops: dict) -> dict:
    return {"schema_version": 1, "as_of": "2026-07-12", "loops": loops}


def _head_object_side_effect(fresh_keys=(), stale_keys=(), missing_keys=(),
                              stale_by=timedelta(days=7)):
    def _side_effect(Bucket, Key):
        if Key in missing_keys:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        if Key in stale_keys:
            return {"LastModified": RUN_START - stale_by}
        if Key in fresh_keys:
            return {"LastModified": RUN_START + timedelta(minutes=5)}
        raise AssertionError(f"unexpected head_object call for key={Key}")
    return _side_effect


class TestReconcilePromotedLiveKeys:

    def test_fresh_key_is_silent(self):
        s3 = MagicMock()
        s3.head_object.side_effect = _head_object_side_effect(
            fresh_keys=[LIVE_KEYS["executor_params"]],
        )
        audit = _audit({"executor_params": {"outcome": "promoted"}})
        findings = reconcile_promoted_live_keys("bucket", audit, RUN_START, s3_client=s3)
        assert findings == []

    def test_stale_key_on_promoted_loop_is_a_finding(self):
        s3 = MagicMock()
        s3.head_object.side_effect = _head_object_side_effect(
            stale_keys=[LIVE_KEYS["executor_params"]],
        )
        audit = _audit({"executor_params": {"outcome": "promoted"}})
        findings = reconcile_promoted_live_keys("bucket", audit, RUN_START, s3_client=s3)
        assert len(findings) == 1
        assert findings[0].loop == "executor_params"
        assert findings[0].reason == "stale"
        assert findings[0].live_key == "config/executor_params.json"

    def test_missing_key_on_promoted_loop_is_a_finding(self):
        # The sharpest form of #2054's orphaned-write class: audit claims
        # promoted, but the live key was never written at all.
        s3 = MagicMock()
        s3.head_object.side_effect = _head_object_side_effect(
            missing_keys=[LIVE_KEYS["scoring_weights"]],
        )
        audit = _audit({"scoring_weights": {"outcome": "promoted"}})
        findings = reconcile_promoted_live_keys("bucket", audit, RUN_START, s3_client=s3)
        assert len(findings) == 1
        assert findings[0].loop == "scoring_weights"
        assert findings[0].reason == "missing"
        assert findings[0].last_modified is None

    def test_never_promoted_loop_is_never_checked(self):
        """Gotcha from the issue: scoring_weights/predictor_params have
        never promoted — their live key legitimately doesn't exist.
        Non-promoted outcomes must not trigger a head_object call at all,
        let alone page."""
        s3 = MagicMock()
        audit = _audit({
            "scoring_weights": {"outcome": "blocked"},
            "predictor_params": {"outcome": "insufficient_data"},
            "research_params": {"outcome": "disabled"},
            "executor_params": {"outcome": "error"},
        })
        findings = reconcile_promoted_live_keys("bucket", audit, RUN_START, s3_client=s3)
        assert findings == []
        assert s3.head_object.call_args_list == []

    def test_mixed_promoted_and_not_only_checks_promoted(self):
        s3 = MagicMock()
        s3.head_object.side_effect = _head_object_side_effect(
            fresh_keys=[LIVE_KEYS["executor_params"]],
        )
        audit = _audit({
            "executor_params": {"outcome": "promoted"},
            "scoring_weights": {"outcome": "blocked"},
        })
        findings = reconcile_promoted_live_keys("bucket", audit, RUN_START, s3_client=s3)
        assert findings == []
        # Only executor_params' key was probed.
        called_keys = {c.kwargs["Key"] for c in s3.head_object.call_args_list}
        assert called_keys == {LIVE_KEYS["executor_params"]}

    def test_non_404_head_object_error_raises(self):
        """A permission/network error means the check could NOT be
        performed — must fail loud, never silently read as clean."""
        s3 = MagicMock()
        s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied"}}, "HeadObject",
        )
        audit = _audit({"executor_params": {"outcome": "promoted"}})
        with pytest.raises(ClientError):
            reconcile_promoted_live_keys("bucket", audit, RUN_START, s3_client=s3)

    def test_empty_audit_is_a_noop(self):
        s3 = MagicMock()
        findings = reconcile_promoted_live_keys("bucket", {}, RUN_START, s3_client=s3)
        assert findings == []
        assert s3.head_object.call_args_list == []

    def test_naive_run_start_treated_as_utc(self):
        s3 = MagicMock()
        s3.head_object.side_effect = _head_object_side_effect(
            fresh_keys=[LIVE_KEYS["executor_params"]],
        )
        naive_start = datetime(2026, 7, 12, 14, 0, 0)  # no tzinfo
        audit = _audit({"executor_params": {"outcome": "promoted"}})
        findings = reconcile_promoted_live_keys("bucket", audit, naive_start, s3_client=s3)
        assert findings == []


class TestRunReconciliationAlerting:
    """Integration: findings actually page via the injected alert function,
    at severity=critical, within the same call — no deferred/async step."""

    def test_clean_run_does_not_page(self):
        s3 = MagicMock()
        s3.head_object.side_effect = _head_object_side_effect(
            fresh_keys=[LIVE_KEYS["executor_params"]],
        )
        publish = MagicMock()
        audit = _audit({"executor_params": {"outcome": "promoted"}})
        findings = run_reconciliation(
            "bucket", audit, RUN_START, "2026-07-12",
            s3_client=s3, publish_alert=publish,
        )
        assert findings == []
        publish.assert_not_called()

    def test_stale_promoted_loop_pages_critical_same_call(self):
        s3 = MagicMock()
        s3.head_object.side_effect = _head_object_side_effect(
            stale_keys=[LIVE_KEYS["executor_params"]],
        )
        publish = MagicMock()
        audit = _audit({"executor_params": {"outcome": "promoted"}})
        findings = run_reconciliation(
            "bucket", audit, RUN_START, "2026-07-12",
            s3_client=s3, publish_alert=publish,
        )
        assert len(findings) == 1
        publish.assert_called_once()
        _, kwargs = publish.call_args
        assert kwargs["severity"] == "critical"
        assert "executor_params" in kwargs.get("dedup_key", "")

    def test_missing_promoted_loop_pages(self):
        s3 = MagicMock()
        s3.head_object.side_effect = _head_object_side_effect(
            missing_keys=[LIVE_KEYS["research_params"]],
        )
        publish = MagicMock()
        audit = _audit({"research_params": {"outcome": "promoted"}})
        run_reconciliation(
            "bucket", audit, RUN_START, "2026-07-12",
            s3_client=s3, publish_alert=publish,
        )
        publish.assert_called_once()

    def test_alert_publish_failure_does_not_raise(self):
        """The page attempt failing must not crash the pipeline — the
        finding is already durably logged at ERROR."""
        s3 = MagicMock()
        s3.head_object.side_effect = _head_object_side_effect(
            stale_keys=[LIVE_KEYS["executor_params"]],
        )
        publish = MagicMock(side_effect=RuntimeError("SNS down"))
        audit = _audit({"executor_params": {"outcome": "promoted"}})
        # Must not raise.
        findings = run_reconciliation(
            "bucket", audit, RUN_START, "2026-07-12",
            s3_client=s3, publish_alert=publish,
        )
        assert len(findings) == 1

    def test_check_failure_still_raises_through_run_reconciliation(self):
        s3 = MagicMock()
        s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied"}}, "HeadObject",
        )
        publish = MagicMock()
        audit = _audit({"executor_params": {"outcome": "promoted"}})
        with pytest.raises(ClientError):
            run_reconciliation(
                "bucket", audit, RUN_START, "2026-07-12",
                s3_client=s3, publish_alert=publish,
            )
        publish.assert_not_called()


class TestEndToEndWithRealAuditBuilder:
    """The full acceptance shape: a REAL build_audit() output (not a hand-
    rolled fixture) whose executor_params classifies "promoted" via the
    assembler-cutover path, reconciled against a live key that predates
    run_start — pages within the same pipeline execution."""

    def test_promoted_via_real_build_audit_pages_when_stale(self):
        opt_results = {
            "weight_result": {"status": "insufficient_data"},
            "executor_rec": {
                "status": "ok",
                "recommended_params": {"min_score": 60},
                "baseline_params": {"min_score": 57},
                "apply_result": {
                    "applied": False,
                    "reason": "cutover_mode — assembler is sole live writer",
                },
            },
            "veto_result": {"status": "insufficient_data"},
            "research_params": {"status": "no_boost_data", "note": "n/a"},
        }
        assembler_summary = {
            "status": "ok",
            "cutover_status": "applied",
            "writers": ["executor_optimizer"],
            "notes": "cutover writes ok",
        }
        audit = build_audit(
            "2026-07-12", opt_results, assembler_summaries={"executor_params": assembler_summary},
        )
        assert audit["loops"]["executor_params"]["outcome"] == "promoted"

        s3 = MagicMock()
        s3.head_object.side_effect = _head_object_side_effect(
            stale_keys=[LIVE_KEYS["executor_params"]],
        )
        publish = MagicMock()
        findings = run_reconciliation(
            "bucket", audit, RUN_START, "2026-07-12",
            s3_client=s3, publish_alert=publish,
        )
        assert [f.loop for f in findings] == ["executor_params"]
        publish.assert_called_once()
        assert publish.call_args.kwargs["severity"] == "critical"

    def test_promoted_via_real_build_audit_silent_when_fresh(self):
        opt_results = {
            "weight_result": {"status": "insufficient_data"},
            "executor_rec": {
                "status": "ok",
                "recommended_params": {"min_score": 60},
                "baseline_params": {"min_score": 57},
                "apply_result": {
                    "applied": False,
                    "reason": "cutover_mode — assembler is sole live writer",
                },
            },
            "veto_result": {"status": "insufficient_data"},
            "research_params": {"status": "no_boost_data", "note": "n/a"},
        }
        assembler_summary = {
            "status": "ok",
            "cutover_status": "applied",
            "writers": ["executor_optimizer"],
            "notes": "cutover writes ok",
        }
        audit = build_audit(
            "2026-07-12", opt_results, assembler_summaries={"executor_params": assembler_summary},
        )
        s3 = MagicMock()
        s3.head_object.side_effect = _head_object_side_effect(
            fresh_keys=[LIVE_KEYS["executor_params"]],
        )
        publish = MagicMock()
        findings = run_reconciliation(
            "bucket", audit, RUN_START, "2026-07-12",
            s3_client=s3, publish_alert=publish,
        )
        assert findings == []
        publish.assert_not_called()
