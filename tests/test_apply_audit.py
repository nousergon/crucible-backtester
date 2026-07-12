"""apply_audit — per-run outcome record for the four auto-apply loops
(config#1841).

Pins: (1) classification of every loop result shape into the frozen outcome
vocabulary (a BLOCKED apply emits an outcome record — the issue's
closes-when); (2) the consecutive_blocked_weeks carry-forward semantics;
(3) frozen-schema conformance (contracts/apply_audit.schema.json) on every
outcome path; (4) the upload gate + fail-loud write posture; (5) the
except-log-emit-reraise wiring in evaluate.py (source-level).
"""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from optimizer.apply_audit import (
    BLOCKED_BY_SLUGS,
    LOOPS,
    build_audit,
    classify_loop,
    emit_apply_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "contracts" / "apply_audit.schema.json"


# ── Fixture result shapes (mirroring the real producers) ─────────────────────


def _weight_blocked():
    return {
        "status": "ok",
        "confidence": "high",
        "oos_passed": True,
        "n_samples": 376,
        "current_weights": {"quant": 0.5, "qual": 0.5},
        "suggested_weights": {"quant": 0.297, "qual": 0.703},
        "changes": {"quant": -0.203, "qual": 0.203},
        "apply_result": {
            "applied": False,
            "blocked_by": ["max_single_change"],
            "reason": "largest change 20.3% exceeds 15% limit — skipping to avoid instability",
        },
    }


def _weight_promoted():
    return {
        "status": "ok",
        "current_weights": {"quant": 0.5, "qual": 0.5},
        "suggested_weights": {"quant": 0.55, "qual": 0.45},
        "apply_result": {"applied": True, "weights": {"quant": 0.55, "qual": 0.45}},
    }


def _veto_insufficient_lift():
    return {
        "status": "insufficient_lift",
        "blocked_by": ["min_lift_over_base_rate"],
        "current_threshold": 0.65,
        "recommended_threshold": 0.65,
        "recommendation_reason": "Best threshold 0.65 has lift 1.2% (need 5%+).",
    }


def _research_no_boost_data():
    return {"status": "no_boost_data", "note": "No boost values found in signals.json"}


def _executor_cutover():
    return {
        "status": "ok",
        "recommended_params": {"min_score": 60},
        "baseline_params": {"min_score": 57},
        "apply_result": {
            "applied": False,
            "reason": "cutover_mode — assembler is sole live writer",
        },
    }


def _all_results(**overrides):
    results = {
        "weight_result": _weight_blocked(),
        "executor_rec": _executor_cutover(),
        "veto_result": _veto_insufficient_lift(),
        "research_params": _research_no_boost_data(),
    }
    results.update(overrides)
    return results


_ASSEMBLER_OK = {
    "status": "ok",
    "cutover_status": "applied",
    "writers": ["executor_optimizer"],
    "notes": "cutover writes ok",
}

# config#2331: merge succeeded (status=ok) but the live-key put_object
# raised — cutover_status is the field classify_loop must key on. A record
# built from this fixture must NEVER classify as "promoted".
_ASSEMBLER_CUTOVER_FAILED = {
    "status": "ok",
    "cutover_status": "failed",
    "writers": ["executor_optimizer"],
    "notes": "cutover_failed: live write to s3://bucket/config/executor_params.json — S3 disconnected",
}


# ── Classification ───────────────────────────────────────────────────────────


class TestClassifyLoop:

    def test_blocked_apply_emits_outcome_record(self):
        """config#1841 closes-when: a blocked apply emits the outcome record
        with the guardrail slug."""
        rec = classify_loop("scoring_weights", _weight_blocked())
        assert rec["outcome"] == "blocked"
        assert rec["blocked_by"] == ["max_single_change"]
        assert rec["proposed"] == {"quant": 0.297, "qual": 0.703}
        assert rec["current"] == {"quant": 0.5, "qual": 0.5}
        assert "exceeds" in rec["detail"]

    def test_promoted(self):
        rec = classify_loop("scoring_weights", _weight_promoted())
        assert rec["outcome"] == "promoted"

    def test_significance_floor_slug(self):
        result = _weight_promoted()
        result["apply_result"] = {
            "applied": False,
            "blocked_by": ["significance_floor"],
            "reason": "weight_optimizer: blocked by significance enforce (config#1426) — undefended evidence",
        }
        rec = classify_loop("scoring_weights", result)
        assert rec["outcome"] == "blocked"
        assert rec["blocked_by"] == ["significance_floor"]

    def test_veto_pre_apply_lift_gate_is_blocked(self):
        rec = classify_loop("predictor_params", _veto_insufficient_lift())
        assert rec["outcome"] == "blocked"
        assert rec["blocked_by"] == ["min_lift_over_base_rate"]
        assert rec["proposed"] == {"veto_confidence": 0.65}

    def test_veto_min_threshold_change(self):
        result = {
            "status": "ok",
            "current_threshold": 0.65,
            "recommended_threshold": 0.65,
            "apply_result": {
                "applied": False,
                "blocked_by": ["min_threshold_change"],
                "reason": "Recommended (0.65) too close to current (0.65) — need 0.05+ difference",
            },
        }
        rec = classify_loop("predictor_params", result)
        assert rec["outcome"] == "blocked"
        assert rec["blocked_by"] == ["min_threshold_change"]

    def test_veto_starvation_statuses_are_insufficient_data(self):
        for status in ("insufficient_data", "no_predictions", "no_down_predictions", "insufficient_vetoes"):
            rec = classify_loop("predictor_params", {"status": status})
            assert rec["outcome"] == "insufficient_data", status

    def test_research_no_boost_data_is_insufficient_data(self):
        rec = classify_loop("research_params", _research_no_boost_data())
        assert rec["outcome"] == "insufficient_data"
        assert "signals.json" in rec["detail"]

    def test_research_no_improvement_maps_to_min_meaningful_change(self):
        rec = classify_loop("research_params", {"status": "no_improvement", "note": "near-optimal"})
        assert rec["outcome"] == "blocked"
        assert rec["blocked_by"] == ["min_meaningful_change"]

    def test_executor_status_guardrails(self):
        for status, slug in [
            ("alpha_below_floor", "alpha_floor"),
            ("insufficient_trades", "min_trades_to_promote"),
            ("negative_sortino", "negative_rank_metric"),
            ("baseline_insignificant", "baseline_magnitude_floor"),
            ("no_improvement", "min_improvement"),
            ("insufficient_psr_confidence", "min_psr"),
        ]:
            rec = classify_loop("executor_params", {"status": status})
            assert rec["outcome"] == "blocked", status
            assert rec["blocked_by"] == [slug], status

    def test_executor_cutover_promoted_via_assembler(self):
        rec = classify_loop(
            "executor_params", _executor_cutover(), assembler_summary=_ASSEMBLER_OK,
        )
        assert rec["outcome"] == "promoted"
        assert "assembler" in rec["detail"]

    def test_executor_cutover_assembler_skip(self):
        rec = classify_loop(
            "executor_params", _executor_cutover(),
            assembler_summary={"status": "all_skip", "writers": [], "notes": "every artifact shadow/skip"},
        )
        assert rec["outcome"] == "blocked"
        assert rec["blocked_by"] == ["assembler_skip"]

    def test_executor_cutover_assembler_missing_is_error(self):
        rec = classify_loop("executor_params", _executor_cutover(), assembler_summary=None)
        assert rec["outcome"] == "error"

    def test_executor_cutover_live_write_failure_is_error_not_promoted(self):
        """config#2331: a failed live-key put_object (assembler merge status
        stays "ok", only cutover_status flips to "failed") must classify as
        outcome "error" — NEVER "promoted". This is the exact defect: before
        the fix, classify_loop keyed on assembler_summary["status"] alone,
        so a swallowed live-write failure (status still "ok") graded as
        promoted and reset consecutive_blocked_weeks."""
        rec = classify_loop(
            "executor_params", _executor_cutover(),
            assembler_summary=_ASSEMBLER_CUTOVER_FAILED,
        )
        assert rec["outcome"] == "error"
        assert rec["outcome"] != "promoted"
        assert "FAILED" in rec["detail"] or "failed" in rec["detail"]

    def test_executor_cutover_status_missing_key_does_not_promote(self):
        """Defense in depth: an assembler_summary dict that predates the
        cutover_status field (e.g. a stale caller) must not default to
        "promoted" — absence of cutover_status must never be silently
        treated as success."""
        rec = classify_loop(
            "executor_params", _executor_cutover(),
            assembler_summary={"status": "ok", "writers": [], "notes": "no cutover_status key"},
        )
        assert rec["outcome"] != "promoted"

    def test_shadow_mode_is_disabled(self):
        result = _weight_promoted()
        result["apply_result"] = {
            "applied": False,
            "reason": "shadow mode — skill_composite enabled, enforce_skill_composite=False",
        }
        rec = classify_loop("scoring_weights", result)
        assert rec["outcome"] == "disabled"

    def test_freeze_is_disabled(self):
        result = _weight_promoted()
        result["apply_result"] = {"applied": False, "reason": "frozen (--freeze flag)"}
        rec = classify_loop("scoring_weights", result)
        assert rec["outcome"] == "disabled"

    def test_loop_error_status(self):
        rec = classify_loop("scoring_weights", {"status": "error", "error": "boom"})
        assert rec["outcome"] == "error"
        assert "boom" in rec["detail"]

    def test_skipped_is_insufficient_data(self):
        rec = classify_loop("scoring_weights", {"status": "skipped"})
        assert rec["outcome"] == "insufficient_data"

    def test_missing_result_with_run_error(self):
        rec = classify_loop("scoring_weights", None, run_error="stage aborted")
        assert rec["outcome"] == "error"
        assert "stage aborted" in rec["detail"]

    def test_s3_write_failure_is_error(self):
        result = _weight_promoted()
        result["apply_result"] = {"applied": False, "reason": "S3 write failed: denied"}
        rec = classify_loop("scoring_weights", result)
        assert rec["outcome"] == "error"

    def test_unknown_apply_rejection_is_loud_not_misbinned(self):
        result = _weight_promoted()
        result["apply_result"] = {"applied": False, "reason": "some future gate"}
        rec = classify_loop("scoring_weights", result)
        assert rec["outcome"] == "blocked"
        assert rec["blocked_by"] == ["unclassified_guardrail"]

    def test_unknown_status_is_error(self):
        rec = classify_loop("scoring_weights", {"status": "brand_new_status"})
        assert rec["outcome"] == "error"


# ── Carry-forward counter ────────────────────────────────────────────────────


class TestCarryForward:

    def _prior(self, **weeks):
        loops = {
            loop: {"outcome": "blocked", "consecutive_blocked_weeks": n}
            for loop, n in weeks.items()
        }
        return {"schema_version": 1, "as_of": "2026-06-28", "loops": loops}

    def test_absent_prior_blocked_starts_at_one(self):
        audit = build_audit("2026-07-05", _all_results(), prior=None)
        assert audit["loops"]["scoring_weights"]["outcome"] == "blocked"
        assert audit["loops"]["scoring_weights"]["consecutive_blocked_weeks"] == 1

    def test_blocked_increments_prior(self):
        audit = build_audit(
            "2026-07-05", _all_results(), prior=self._prior(scoring_weights=7),
        )
        assert audit["loops"]["scoring_weights"]["consecutive_blocked_weeks"] == 8

    def test_promoted_resets(self):
        audit = build_audit(
            "2026-07-05", _all_results(weight_result=_weight_promoted()),
            prior=self._prior(scoring_weights=7),
        )
        assert audit["loops"]["scoring_weights"]["outcome"] == "promoted"
        assert audit["loops"]["scoring_weights"]["consecutive_blocked_weeks"] == 0

    def test_insufficient_data_resets(self):
        audit = build_audit(
            "2026-07-05", _all_results(), prior=self._prior(research_params=3),
        )
        assert audit["loops"]["research_params"]["outcome"] == "insufficient_data"
        assert audit["loops"]["research_params"]["consecutive_blocked_weeks"] == 0

    def test_error_carries_prior_unchanged(self):
        audit = build_audit(
            "2026-07-05",
            _all_results(weight_result={"status": "error", "error": "boom"}),
            prior=self._prior(scoring_weights=7),
        )
        assert audit["loops"]["scoring_weights"]["outcome"] == "error"
        assert audit["loops"]["scoring_weights"]["consecutive_blocked_weeks"] == 7

    def test_disabled_carries_prior_unchanged(self):
        frozen = _weight_promoted()
        frozen["apply_result"] = {"applied": False, "reason": "frozen (--freeze flag)"}
        audit = build_audit(
            "2026-07-05", _all_results(weight_result=frozen),
            prior=self._prior(scoring_weights=4),
        )
        assert audit["loops"]["scoring_weights"]["outcome"] == "disabled"
        assert audit["loops"]["scoring_weights"]["consecutive_blocked_weeks"] == 4

    def test_cutover_live_write_failure_preserves_blocked_counter(self):
        """config#2331 acceptance: a mocked live-key put failure must
        classify as outcome "error" AND leave consecutive_blocked_weeks
        untouched (error carries forward unchanged — it is neither evidence
        of blocking nor of unblocking). Before the fix, this scenario
        classified as "promoted", which RESET the counter — exactly the
        silent-recovery-from-failure bug the issue describes."""
        audit = build_audit(
            "2026-07-05", _all_results(),
            assembler_summary=_ASSEMBLER_CUTOVER_FAILED,
            prior=self._prior(executor_params=5),
        )
        assert audit["loops"]["executor_params"]["outcome"] == "error"
        assert audit["loops"]["executor_params"]["consecutive_blocked_weeks"] == 5


# ── Frozen-schema conformance ────────────────────────────────────────────────

jsonschema = pytest.importorskip(
    "jsonschema",
    reason="needs nousergon-lib[contracts] (jsonschema) for schema validation",
)


def _validate(audit: dict) -> None:
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(instance=audit, schema=schema)


class TestSchemaConformance:

    def test_mixed_outcomes_conform(self):
        audit = build_audit("2026-07-05", _all_results(), assembler_summary=_ASSEMBLER_OK)
        _validate(audit)
        assert set(audit["loops"]) == set(LOOPS)

    def test_all_error_path_conforms(self):
        audit = build_audit("2026-07-05", {}, run_error="stage aborted")
        _validate(audit)
        for rec in audit["loops"].values():
            assert rec["outcome"] == "error"

    def test_promoted_and_disabled_conform(self):
        frozen_veto = {
            "status": "ok",
            "recommended_threshold": 0.55,
            "current_threshold": 0.65,
            "apply_result": {"applied": False, "reason": "frozen (--freeze flag)"},
        }
        audit = build_audit(
            "2026-07-05",
            _all_results(weight_result=_weight_promoted(), veto_result=frozen_veto),
            assembler_summary=_ASSEMBLER_OK,
        )
        _validate(audit)

    def test_strict_json_serializable(self):
        audit = build_audit("2026-07-05", _all_results())
        json.dumps(audit, allow_nan=False)

    def test_module_slugs_exactly_match_schema_enum(self):
        """The slug vocabulary is frozen in BOTH the module and the schema —
        drift between them breaks the evaluator consumer."""
        schema = json.loads(SCHEMA_PATH.read_text())
        enum = schema["$defs"]["loop_record"]["properties"]["blocked_by"]["oneOf"][1]["items"]["enum"]
        assert set(enum) == set(BLOCKED_BY_SLUGS)

    def test_stale_slug_fails_validation(self):
        """Non-vacuousness: an out-of-vocabulary slug is rejected."""
        audit = build_audit("2026-07-05", _all_results())
        audit["loops"]["scoring_weights"]["blocked_by"] = ["made_up_gate"]
        with pytest.raises(jsonschema.ValidationError):
            _validate(audit)


# ── Emission: upload gate + fail-loud write ──────────────────────────────────


class TestEmit:

    def test_no_upload_skips_s3_entirely(self):
        s3 = MagicMock()
        audit = emit_apply_audit(
            bucket="b", run_date="2026-07-05", opt_results=_all_results(),
            upload=False, s3_client=s3,
        )
        assert s3.put_object.call_args_list == []
        assert s3.get_object.call_args_list == []  # no prior read either
        assert audit["loops"]["scoring_weights"]["outcome"] == "blocked"

    def test_upload_writes_dated_and_latest(self):
        s3 = MagicMock()
        s3.get_object.side_effect = _no_such_key()
        emit_apply_audit(
            bucket="b", run_date="2026-07-05", opt_results=_all_results(),
            assembler_result=None, upload=True, s3_client=s3,
        )
        keys = [c.kwargs.get("Key") for c in s3.put_object.call_args_list]
        assert keys == [
            "config/apply_audit/2026-07-05.json",
            "config/apply_audit/latest.json",
        ]
        body = json.loads(s3.put_object.call_args_list[0].kwargs["Body"])
        _validate(body)

    def test_prior_read_feeds_carry_forward(self):
        s3 = MagicMock()
        prior = {
            "schema_version": 1,
            "as_of": "2026-06-28",
            "loops": {"scoring_weights": {"outcome": "blocked", "consecutive_blocked_weeks": 4}},
        }
        s3.get_object.return_value = {"Body": _body(prior)}
        audit = emit_apply_audit(
            bucket="b", run_date="2026-07-05", opt_results=_all_results(),
            upload=True, s3_client=s3,
        )
        assert audit["loops"]["scoring_weights"]["consecutive_blocked_weeks"] == 5

    def test_write_failure_raises_without_pending_error(self):
        """The audit write is load-bearing: a swallowed failure would recreate
        the silence defect. No pending stage error → the write failure raises."""
        s3 = MagicMock()
        s3.get_object.side_effect = _no_such_key()
        s3.put_object.side_effect = RuntimeError("s3 down")
        with pytest.raises(RuntimeError, match="s3 down"):
            emit_apply_audit(
                bucket="b", run_date="2026-07-05", opt_results=_all_results(),
                upload=True, s3_client=s3,
            )

    def test_write_failure_does_not_mask_pending_stage_error(self):
        s3 = MagicMock()
        s3.get_object.side_effect = _no_such_key()
        s3.put_object.side_effect = RuntimeError("s3 down")
        audit = emit_apply_audit(
            bucket="b", run_date="2026-07-05", opt_results={},
            upload=True, run_error=RuntimeError("original stage failure"),
            s3_client=s3,
        )
        # No raise from emit — evaluate re-raises the ORIGINAL error; the
        # audit body still classified every loop as error.
        for rec in audit["loops"].values():
            assert rec["outcome"] == "error"

    def test_run_error_classifies_missing_loops_as_error(self):
        audit = emit_apply_audit(
            bucket="b", run_date="2026-07-05",
            opt_results={"weight_result": _weight_blocked()},
            upload=False, run_error=RuntimeError("manifest write failed"),
        )
        assert audit["loops"]["scoring_weights"]["outcome"] == "blocked"
        assert audit["loops"]["executor_params"]["outcome"] == "error"
        assert "manifest write failed" in audit["loops"]["executor_params"]["detail"]


def _no_such_key():
    from botocore.exceptions import ClientError
    return ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")


def _body(payload: dict):
    class _B:
        def __init__(self, data):
            self._data = json.dumps(data).encode()

        def read(self):
            return self._data

    return _B(payload)


# ── evaluate.py wiring (source-level) ────────────────────────────────────────


class TestEvaluateWiring:

    def test_emission_is_wired_with_reraise(self):
        """except-log-emit-reraise: evaluate wraps the optimizer stage,
        emits the audit, and re-raises the original stage error."""
        src = (REPO_ROOT / "evaluate.py").read_text()
        assert "emit_apply_audit(" in src
        assert "raise opt_stage_error" in src
        # The upload gate mirrors sibling artifacts (args.upload + not
        # freeze). config#2332 hoisted the gate into a named
        # `apply_audit_upload` variable (reused by the post-optimizer live-
        # key reconciliation step) — pin the gate's definition and its use
        # at the emit_apply_audit call site rather than requiring the
        # expression to be inlined there.
        assert (
            'apply_audit_upload = bool(getattr(args, "upload", False)) '
            "and not args.freeze"
        ) in src
        emit_call = src.split("emit_apply_audit(")[-1].split("run_error=")[0]
        assert "upload=apply_audit_upload" in emit_call
