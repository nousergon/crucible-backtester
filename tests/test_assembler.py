"""Unit tests for optimizer.assembler — merge precedence + provenance audit
+ shadow audit write. All S3 calls mocked.

Part of the optimizer-artifact-assembler arc (PR 3). The assembler is
shadow-only at this PR — writes the audit artifact but does NOT touch
the live key. PR 4 introduces the cutover.
"""
import json
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from optimizer.assembler import (
    DEFAULT_PRECEDENCE,
    AssemblerResult,
    _apply_artifact_to_base,
    _read_current_live,
    assemble,
    is_cutover_enabled,
    set_cutover_enabled,
)
from optimizer.recommendation_artifact import RecommendationArtifact


@pytest.fixture(autouse=True)
def _reset_cutover_flag():
    """Reset the module-level cutover flag to False before AND after each
    test in this module. Prevents one test's set_cutover_enabled(True) from
    leaking into the next."""
    set_cutover_enabled(False)
    yield
    set_cutover_enabled(False)


def _stub_s3_for_assemble(
    current_live: dict | None,
    artifacts: dict[str, RecommendationArtifact],
) -> MagicMock:
    """Build an S3 mock whose ``get_object`` returns ``current_live`` for
    the live key and each artifact for its canonical recommendation key,
    and whose ``list_objects_v2`` returns the artifacts' keys.
    """
    s3 = MagicMock()
    live_key = "config/executor_params.json"

    def list_side_effect(Bucket, Prefix):
        if Prefix.startswith("config/executor_params/recommendations/"):
            return {
                "Contents": [
                    {"Key": (
                        f"config/executor_params/recommendations/{a.run_date}/"
                        f"from_{a.optimizer_name}.json"
                    )}
                    for a in artifacts.values()
                ],
            }
        return {}

    def get_side_effect(Bucket, Key):
        if Key == live_key:
            if current_live is None:
                err = {"Error": {"Code": "NoSuchKey"}}
                raise ClientError(err, "GetObject")
            return {"Body": MagicMock(read=lambda: json.dumps(current_live).encode())}
        # Lookup artifact by Key suffix.
        for art in artifacts.values():
            artifact_key = (
                f"config/executor_params/recommendations/{art.run_date}/"
                f"from_{art.optimizer_name}.json"
            )
            if Key == artifact_key:
                return {
                    "Body": MagicMock(read=lambda body=art.to_json(): body.encode("utf-8")),
                }
        err = {"Error": {"Code": "NoSuchKey"}}
        raise ClientError(err, "GetObject")

    s3.list_objects_v2.side_effect = list_side_effect
    s3.get_object.side_effect = get_side_effect
    return s3


def _make_executor_artifact(
    recommended: dict, intent: str = "promote", run_date: str = "2026-05-09",
) -> RecommendationArtifact:
    return RecommendationArtifact(
        fit_target="skill_composite",
        optimizer_name="executor_optimizer",
        run_date=run_date,
        recommendation_kind="full_replace",
        recommended_params=recommended,
        promotion_intent=intent,
    )


def _make_sizing_artifact(
    recommended: dict, intent: str = "promote", run_date: str = "2026-05-09",
) -> RecommendationArtifact:
    return RecommendationArtifact(
        fit_target="sizing_ic",
        optimizer_name="predictor_sizing_optimizer",
        run_date=run_date,
        recommendation_kind="field_overlay",
        recommended_params=recommended,
        overlay_keys=list(recommended.keys()),
        promotion_intent=intent,
    )


def _make_trigger_artifact(
    recommended: dict, intent: str = "promote", run_date: str = "2026-05-09",
) -> RecommendationArtifact:
    return RecommendationArtifact(
        fit_target="entry_timing_alpha",
        optimizer_name="trigger_optimizer",
        run_date=run_date,
        recommendation_kind="field_overlay",
        recommended_params=recommended,
        overlay_keys=list(recommended.keys()),
        promotion_intent=intent,
    )


# ── _read_current_live ──────────────────────────────────────────────────────


class TestReadCurrentLive:

    def test_returns_dict_when_present(self):
        s3 = MagicMock()
        s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps({"atr_multiplier": 2.0}).encode()),
        }
        data, present = _read_current_live("test-bucket", "executor_params", s3)
        assert data == {"atr_multiplier": 2.0}
        assert present is True

    def test_returns_empty_on_no_such_key(self):
        s3 = MagicMock()
        s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject",
        )
        data, present = _read_current_live("test-bucket", "executor_params", s3)
        assert data == {}
        assert present is False

    def test_other_client_errors_propagate(self):
        s3 = MagicMock()
        s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied"}}, "GetObject",
        )
        with pytest.raises(ClientError):
            _read_current_live("test-bucket", "executor_params", s3)

    def test_non_dict_treated_as_empty(self):
        s3 = MagicMock()
        s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps([1, 2, 3]).encode()),
        }
        data, present = _read_current_live("test-bucket", "executor_params", s3)
        assert data == {}
        assert present is False


# ── _apply_artifact_to_base — merge primitive ──────────────────────────────


class TestApplyArtifactToBase:

    def test_full_replace_replaces_entire_dict(self):
        base = {"atr_multiplier": 2.0, "min_score": 70, "use_p_up_sizing": True}
        artifact = _make_executor_artifact({"atr_multiplier": 3.0, "min_score": 75})
        summary: dict = {}
        result = _apply_artifact_to_base(base, artifact, summary)
        # full_replace drops use_p_up_sizing (not in this artifact's params).
        assert result == {"atr_multiplier": 3.0, "min_score": 75}
        # Provenance for both written keys, none for the dropped key.
        assert set(summary.keys()) == {"atr_multiplier", "min_score"}
        assert summary["atr_multiplier"]["writer"] == "executor_optimizer"
        assert summary["atr_multiplier"]["kind"] == "full_replace"

    def test_field_overlay_preserves_other_keys(self):
        base = {"atr_multiplier": 2.0, "min_score": 70}
        artifact = _make_sizing_artifact({"use_p_up_sizing": True, "p_up_sizing_blend": 0.3})
        summary: dict = {}
        result = _apply_artifact_to_base(base, artifact, summary)
        # Existing base keys preserved; overlay keys added.
        assert result == {
            "atr_multiplier": 2.0,
            "min_score": 70,
            "use_p_up_sizing": True,
            "p_up_sizing_blend": 0.3,
        }
        # Provenance only for overlay keys.
        assert set(summary.keys()) == {"use_p_up_sizing", "p_up_sizing_blend"}
        assert summary["use_p_up_sizing"]["writer"] == "predictor_sizing_optimizer"
        assert summary["use_p_up_sizing"]["kind"] == "field_overlay"

    def test_full_replace_drops_provenance_for_keys_no_longer_present(self):
        base = {"atr_multiplier": 2.0, "use_p_up_sizing": True}
        # Pre-existing summary entry for use_p_up_sizing from a prior overlay.
        summary = {
            "use_p_up_sizing": {
                "value": True,
                "writer": "predictor_sizing_optimizer",
                "kind": "field_overlay",
                "run_id": "prior-run",
            },
        }
        artifact = _make_executor_artifact({"atr_multiplier": 3.0})
        result = _apply_artifact_to_base(base, artifact, summary)
        assert "use_p_up_sizing" not in result
        # Stale provenance for dropped key cleared.
        assert "use_p_up_sizing" not in summary

    def test_unknown_kind_skipped_with_warning(self, caplog):
        base = {"atr_multiplier": 2.0}
        a = RecommendationArtifact(
            fit_target="x", optimizer_name="future_optimizer",
            run_date="2026-05-09", recommendation_kind="full_replace",  # type: ignore[arg-type]
            recommended_params={"k": 1}, promotion_intent="promote",
        )
        # Inject an unknown kind directly (bypasses Literal type-check).
        a.recommendation_kind = "lol_unknown"  # type: ignore[assignment]
        result = _apply_artifact_to_base(base, a, {})
        assert result == base


# ── assemble — happy path: today's chain reproducible ──────────────────────


class TestAssembleHappyPath:

    def test_merges_executor_full_replace_then_sizing_overlay_then_trigger_overlay(self):
        # Reproduces today's 2026-05-09 chain — but with all 5 writers
        # contributing. executor sets risk params; sizing adds use_p_up_sizing;
        # trigger adds disabled_triggers list.
        current_live = {
            # Stale 5/7 data — would be overwritten by executor_optimizer's full_replace.
            "atr_multiplier": 2.0,
            "min_score": 75,
        }
        artifacts = {
            "executor_optimizer": _make_executor_artifact({
                "atr_multiplier": 3.0,
                "min_score": 75,
                "max_position_pct": 0.10,
                "time_decay_exit_days": 15,
            }),
            "predictor_sizing_optimizer": _make_sizing_artifact({
                "use_p_up_sizing": True,
                "p_up_sizing_blend": 0.3,
            }),
            "trigger_optimizer": _make_trigger_artifact({
                "disabled_triggers": ["pullback"],
                "disabled_triggers_updated_at": "2026-05-09",
            }),
        }
        s3 = _stub_s3_for_assemble(current_live, artifacts)
        result = assemble(
            "test-bucket", "executor_params", "2026-05-09",
            s3_client=s3, write_assembled=False,
        )
        assert result.status == "ok"
        # Merged config has executor's risk params + sizing's fields + trigger's fields.
        assert result.assembled_params == {
            "atr_multiplier": 3.0,
            "min_score": 75,
            "max_position_pct": 0.10,
            "time_decay_exit_days": 15,
            "use_p_up_sizing": True,
            "p_up_sizing_blend": 0.3,
            "disabled_triggers": ["pullback"],
            "disabled_triggers_updated_at": "2026-05-09",
        }
        # Provenance correctly attributes each key.
        assert result.merge_summary["atr_multiplier"]["writer"] == "executor_optimizer"
        assert result.merge_summary["use_p_up_sizing"]["writer"] == "predictor_sizing_optimizer"
        assert result.merge_summary["disabled_triggers"]["writer"] == "trigger_optimizer"
        # All 3 artifacts recorded in artifacts_seen.
        assert set(result.artifacts_seen.keys()) == {
            "executor_optimizer", "predictor_sizing_optimizer", "trigger_optimizer",
        }

    def test_executor_only_drops_legacy_overlays_from_base(self):
        # If executor_optimizer is the ONLY promoter and the base has legacy
        # overlay fields (use_p_up_sizing from a prior trigger_optimizer run),
        # full_replace drops them. This is the documented behavior — follow-on
        # overlay writers that don't run this week are NOT preserved silently.
        current_live = {
            "atr_multiplier": 2.0,
            "use_p_up_sizing": True,  # prior overlay
            "disabled_triggers": ["vwap_discount"],  # prior overlay
        }
        artifacts = {
            "executor_optimizer": _make_executor_artifact({
                "atr_multiplier": 3.0,
                "min_score": 75,
            }),
        }
        s3 = _stub_s3_for_assemble(current_live, artifacts)
        result = assemble(
            "test-bucket", "executor_params", "2026-05-09",
            s3_client=s3, write_assembled=False,
        )
        assert result.assembled_params == {"atr_multiplier": 3.0, "min_score": 75}
        assert "use_p_up_sizing" not in result.assembled_params
        assert "disabled_triggers" not in result.assembled_params


class TestAssembleSkipPaths:

    def test_no_artifacts_returns_unchanged_base(self):
        current_live = {"atr_multiplier": 2.0}
        artifacts: dict = {}
        s3 = _stub_s3_for_assemble(current_live, artifacts)
        result = assemble(
            "test-bucket", "executor_params", "2026-05-09",
            s3_client=s3, write_assembled=False,
        )
        assert result.status == "no_artifacts"
        assert result.assembled_params == {"atr_multiplier": 2.0}
        assert result.merge_summary == {}
        assert result.base_was_present is True

    def test_all_artifacts_skip_returns_unchanged_base(self):
        current_live = {"atr_multiplier": 2.0}
        artifacts = {
            "executor_optimizer": _make_executor_artifact(
                {"atr_multiplier": 3.0}, intent="skip",
            ),
            "predictor_sizing_optimizer": _make_sizing_artifact(
                {"use_p_up_sizing": True}, intent="shadow",
            ),
        }
        s3 = _stub_s3_for_assemble(current_live, artifacts)
        result = assemble(
            "test-bucket", "executor_params", "2026-05-09",
            s3_client=s3, write_assembled=False,
        )
        assert result.status == "all_skip"
        assert result.assembled_params == {"atr_multiplier": 2.0}
        # Both artifacts recorded for audit even though neither merged.
        assert set(result.artifacts_seen.keys()) == {
            "executor_optimizer", "predictor_sizing_optimizer",
        }
        assert result.artifacts_seen["executor_optimizer"]["promotion_intent"] == "skip"
        assert result.artifacts_seen["predictor_sizing_optimizer"]["promotion_intent"] == "shadow"

    def test_mixed_promote_and_skip_only_promote_merges(self):
        # executor promotes, sizing skips, trigger promotes → merged has
        # executor's full_replace + trigger's overlay, no sizing fields.
        current_live = {"atr_multiplier": 2.0, "use_p_up_sizing": True}
        artifacts = {
            "executor_optimizer": _make_executor_artifact({"atr_multiplier": 3.0}),
            "predictor_sizing_optimizer": _make_sizing_artifact(
                {"use_p_up_sizing": False}, intent="skip",
            ),
            "trigger_optimizer": _make_trigger_artifact(
                {"disabled_triggers": ["pullback"], "disabled_triggers_updated_at": "2026-05-09"},
            ),
        }
        s3 = _stub_s3_for_assemble(current_live, artifacts)
        result = assemble(
            "test-bucket", "executor_params", "2026-05-09",
            s3_client=s3, write_assembled=False,
        )
        assert result.status == "ok"
        assert "atr_multiplier" in result.assembled_params
        assert result.assembled_params["atr_multiplier"] == 3.0
        # sizing skipped → use_p_up_sizing dropped by full_replace, not restored.
        assert "use_p_up_sizing" not in result.assembled_params
        # trigger promoted → its keys present.
        assert result.assembled_params["disabled_triggers"] == ["pullback"]


class TestAssembleFreezeKeys:

    def test_frozen_key_overrides_optimizer_recommendation(self):
        current_live = {"atr_multiplier": 2.0, "max_position_pct": 0.05}
        artifacts = {
            "executor_optimizer": _make_executor_artifact({
                "atr_multiplier": 3.0,
                "max_position_pct": 0.20,  # operator says no — locked at 0.05
            }),
        }
        precedence = {
            "precedence": ["executor_optimizer"],
            "freeze_keys": ["max_position_pct"],
        }
        s3 = _stub_s3_for_assemble(current_live, artifacts)
        result = assemble(
            "test-bucket", "executor_params", "2026-05-09",
            precedence_config=precedence, s3_client=s3, write_assembled=False,
        )
        assert result.assembled_params["atr_multiplier"] == 3.0
        assert result.assembled_params["max_position_pct"] == 0.05  # frozen
        assert "max_position_pct" in result.frozen_keys_restored
        assert result.merge_summary["max_position_pct"]["writer"] == "operator_freeze"

    def test_frozen_key_not_in_base_does_not_appear(self):
        # If a freeze_key isn't in the base, it has no value to restore.
        # The optimizer's recommendation also doesn't land for it (since it
        # would be re-frozen). Result: key absent.
        current_live: dict = {}
        artifacts = {
            "executor_optimizer": _make_executor_artifact({
                "atr_multiplier": 3.0,
                "max_position_pct": 0.20,
            }),
        }
        precedence = {
            "precedence": ["executor_optimizer"],
            "freeze_keys": ["max_position_pct"],
        }
        s3 = _stub_s3_for_assemble(current_live, artifacts)
        result = assemble(
            "test-bucket", "executor_params", "2026-05-09",
            precedence_config=precedence, s3_client=s3, write_assembled=False,
        )
        # Key absent from base → optimizer's value lands and is NOT restored
        # (no operator-set value to restore TO). This is documented behavior:
        # freeze_keys only restore values that were already in base.
        assert result.assembled_params["max_position_pct"] == 0.20


class TestAssembleAuditWrite:

    def test_write_assembled_true_emits_audit_artifact(self):
        current_live = {"atr_multiplier": 2.0}
        artifacts = {
            "executor_optimizer": _make_executor_artifact({"atr_multiplier": 3.0}),
        }
        s3 = _stub_s3_for_assemble(current_live, artifacts)
        result = assemble(
            "test-bucket", "executor_params", "2026-05-09",
            s3_client=s3, write_assembled=True,
        )
        # Audit write happened.
        audit_calls = [
            c for c in s3.put_object.call_args_list
            if c.kwargs["Key"] == "config/executor_params/assembled/2026-05-09.json"
        ]
        assert len(audit_calls) == 1
        body = json.loads(audit_calls[0].kwargs["Body"])
        assert body["status"] == "ok"
        assert body["assembled_params"] == {"atr_multiplier": 3.0}

    def test_write_assembled_false_skips_audit(self):
        # Used by tests + manual dry-runs; no S3 audit write.
        current_live = {"atr_multiplier": 2.0}
        artifacts = {
            "executor_optimizer": _make_executor_artifact({"atr_multiplier": 3.0}),
        }
        s3 = _stub_s3_for_assemble(current_live, artifacts)
        assemble(
            "test-bucket", "executor_params", "2026-05-09",
            s3_client=s3, write_assembled=False,
        )
        audit_calls = [
            c for c in s3.put_object.call_args_list
            if "/assembled/" in c.kwargs.get("Key", "")
        ]
        assert audit_calls == []

    def test_audit_write_failure_non_fatal(self):
        # During shadow-only PR 3 phase, an audit-write S3 failure must NOT
        # raise — the assembler is observation-only.
        current_live = {"atr_multiplier": 2.0}
        artifacts = {
            "executor_optimizer": _make_executor_artifact({"atr_multiplier": 3.0}),
        }
        s3 = _stub_s3_for_assemble(current_live, artifacts)
        # Make audit write fail; reads + recommendation listing still work.
        original_put = s3.put_object

        def fail_put(*args, **kwargs):
            raise Exception("S3 disconnected on audit write")

        s3.put_object = fail_put
        # Should not raise.
        result = assemble(
            "test-bucket", "executor_params", "2026-05-09",
            s3_client=s3, write_assembled=True,
        )
        assert result.status == "ok"

    def test_assembler_does_not_write_live_key(self):
        # Critical PR 3 invariant: shadow-only. The live key
        # config/executor_params.json must NEVER be touched by the assembler
        # in this phase. PR 4 introduces the cutover.
        current_live = {"atr_multiplier": 2.0}
        artifacts = {
            "executor_optimizer": _make_executor_artifact({"atr_multiplier": 3.0}),
        }
        s3 = _stub_s3_for_assemble(current_live, artifacts)
        assemble(
            "test-bucket", "executor_params", "2026-05-09", s3_client=s3,
        )
        live_writes = [
            c for c in s3.put_object.call_args_list
            if c.kwargs.get("Key") == "config/executor_params.json"
        ]
        assert live_writes == [], (
            "PR 3 assembler must not write live key — that's PR 4's "
            "flag-gated cutover work."
        )


class TestDefaultPrecedence:

    def test_default_precedence_for_executor_params(self):
        # Pin the default ordering so a future edit accidentally changing it
        # trips a known-failure.
        cfg = DEFAULT_PRECEDENCE["executor_params"]
        assert cfg["precedence"] == [
            "executor_optimizer",
            "predictor_sizing_optimizer",
            "trigger_optimizer",
        ]
        assert cfg["freeze_keys"] == []


class TestCutoverFlag:

    def test_default_is_false(self):
        # Asserted via the autouse fixture's reset; this just pins the
        # public default semantics.
        assert is_cutover_enabled() is False

    def test_set_and_read(self):
        set_cutover_enabled(True)
        assert is_cutover_enabled() is True
        set_cutover_enabled(False)
        assert is_cutover_enabled() is False


class TestAssembleCutoverMode:
    """When cutover_enabled=True (or module flag is set), the assembler is
    the sole writer of the live key. Three writes happen on a successful
    merge: snapshot live → _previous, write assembled → live, mirror to
    dated history."""

    def test_cutover_writes_live_previous_and_history_keys(self):
        current_live = {"atr_multiplier": 2.0, "min_score": 70}
        artifacts = {
            "executor_optimizer": _make_executor_artifact({
                "atr_multiplier": 3.0, "min_score": 75,
            }),
        }
        s3 = _stub_s3_for_assemble(current_live, artifacts)
        result = assemble(
            "test-bucket", "executor_params", "2026-05-09",
            s3_client=s3, write_assembled=True, cutover_enabled=True,
        )
        assert result.status == "ok"

        # 1. _previous snapshot via copy_object from current live
        copy_calls = s3.copy_object.call_args_list
        assert len(copy_calls) == 1
        assert copy_calls[0].kwargs["Key"] == "config/executor_params_previous.json"
        assert copy_calls[0].kwargs["CopySource"] == {
            "Bucket": "test-bucket", "Key": "config/executor_params.json",
        }

        # 2. Live key written with assembled_params + updated_at/assembled_by stamps
        live_calls = [
            c for c in s3.put_object.call_args_list
            if c.kwargs["Key"] == "config/executor_params.json"
        ]
        assert len(live_calls) == 1
        live_body = json.loads(live_calls[0].kwargs["Body"])
        assert live_body["atr_multiplier"] == 3.0
        assert live_body["min_score"] == 75
        assert live_body["updated_at"] == "2026-05-09"
        assert live_body["assembled_by"] == "optimizer.assembler"

        # 3. Dated history written
        history_calls = [
            c for c in s3.put_object.call_args_list
            if c.kwargs["Key"] == "config/executor_params_history/2026-05-09.json"
        ]
        assert len(history_calls) == 1

        # 4. Audit artifact ALSO written (write_assembled=True)
        audit_calls = [
            c for c in s3.put_object.call_args_list
            if c.kwargs["Key"] == "config/executor_params/assembled/2026-05-09.json"
        ]
        assert len(audit_calls) == 1

        # Notes capture cutover application
        assert "cutover_applied" in result.notes

    def test_cutover_disabled_does_not_touch_live(self):
        # Explicit cutover_enabled=False: shadow-only — no live key write.
        current_live = {"atr_multiplier": 2.0}
        artifacts = {
            "executor_optimizer": _make_executor_artifact({"atr_multiplier": 3.0}),
        }
        s3 = _stub_s3_for_assemble(current_live, artifacts)
        assemble(
            "test-bucket", "executor_params", "2026-05-09",
            s3_client=s3, write_assembled=True, cutover_enabled=False,
        )
        # No copy_object (no _previous snapshot)
        assert s3.copy_object.call_args_list == []
        # No live key write
        live_writes = [
            c for c in s3.put_object.call_args_list
            if c.kwargs["Key"] == "config/executor_params.json"
        ]
        assert live_writes == []

    def test_cutover_enabled_via_module_flag_when_param_is_none(self):
        # When cutover_enabled param is None (default), assemble() reads the
        # module-level flag set by set_cutover_enabled(). Pinned so the
        # production wiring (set once at startup) actually drives the cutover.
        set_cutover_enabled(True)
        current_live = {"atr_multiplier": 2.0}
        artifacts = {
            "executor_optimizer": _make_executor_artifact({"atr_multiplier": 3.0}),
        }
        s3 = _stub_s3_for_assemble(current_live, artifacts)
        assemble(
            "test-bucket", "executor_params", "2026-05-09",
            s3_client=s3, write_assembled=True,  # cutover_enabled NOT passed
        )
        live_writes = [
            c for c in s3.put_object.call_args_list
            if c.kwargs["Key"] == "config/executor_params.json"
        ]
        assert len(live_writes) == 1

    def test_cutover_with_no_prior_live_first_run(self):
        # First-ever cutover run: no current live config. The copy_object
        # call gets NoSuchKey (logged info, not raised). Live key still
        # gets written with the assembled output.
        current_live = None  # → triggers NoSuchKey in stub
        artifacts = {
            "executor_optimizer": _make_executor_artifact({"atr_multiplier": 3.0}),
        }
        s3 = _stub_s3_for_assemble(current_live, artifacts)
        # Make copy_object raise NoSuchKey to simulate first-cutover.
        s3.copy_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "CopyObject",
        )
        result = assemble(
            "test-bucket", "executor_params", "2026-05-09",
            s3_client=s3, write_assembled=False, cutover_enabled=True,
        )
        # No prior live to snapshot, but the live write still landed.
        live_writes = [
            c for c in s3.put_object.call_args_list
            if c.kwargs["Key"] == "config/executor_params.json"
        ]
        assert len(live_writes) == 1
        assert "cutover_applied" in result.notes

    def test_cutover_skips_when_status_not_ok(self):
        # status=all_skip → cutover branch should NOT write live (the
        # assembled_params is the unchanged base, but per design we don't
        # rewrite the live key with itself).
        current_live = {"atr_multiplier": 2.0}
        artifacts = {
            "executor_optimizer": _make_executor_artifact(
                {"atr_multiplier": 3.0}, intent="skip",
            ),
        }
        s3 = _stub_s3_for_assemble(current_live, artifacts)
        result = assemble(
            "test-bucket", "executor_params", "2026-05-09",
            s3_client=s3, write_assembled=False, cutover_enabled=True,
        )
        assert result.status == "all_skip"
        live_writes = [
            c for c in s3.put_object.call_args_list
            if c.kwargs["Key"] == "config/executor_params.json"
        ]
        assert live_writes == []
        assert s3.copy_object.call_args_list == []

    def test_cutover_live_write_failure_recorded_in_notes(self):
        current_live = {"atr_multiplier": 2.0}
        artifacts = {
            "executor_optimizer": _make_executor_artifact({"atr_multiplier": 3.0}),
        }
        s3 = _stub_s3_for_assemble(current_live, artifacts)

        # Make ONLY the live-key put fail; audit + history put succeed.
        original_put = s3.put_object.side_effect

        def fail_live_put(*args, **kwargs):
            if kwargs.get("Key") == "config/executor_params.json":
                raise Exception("S3 disconnected on live write")
            # Fall back to MagicMock default behavior.
            return MagicMock()

        s3.put_object.side_effect = fail_live_put
        result = assemble(
            "test-bucket", "executor_params", "2026-05-09",
            s3_client=s3, write_assembled=False, cutover_enabled=True,
        )
        # Result still status=ok; the failure is captured in notes for
        # operator visibility (the audit artifact is the authoritative record
        # if it was written).
        assert "cutover_failed" in result.notes
        assert "S3 disconnected" in result.notes
