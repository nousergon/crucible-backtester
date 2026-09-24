"""A rehearsal run never writes live config (alpha-engine-config-I11505).

``rehearsal-2026-09-23-2`` ran EvaluatorOptimize with cutover ON and wrote
``config/executor_params.json``. These tests pin every layer of the guard:
the role detection, the entrypoint freeze, both live-write backstops, the
launcher forwarding the SF execution name onto the spot, and the decided
ordering between the assembler cutover and ``apply_audit``.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from optimizer import run_role
from optimizer.run_role import (
    RehearsalLiveWriteRefused,
    freeze_if_rehearsal,
    refuse_live_config_write_in_rehearsal,
    rehearsal_reason,
)

REPO = Path(__file__).resolve().parent.parent


# ── role detection ────────────────────────────────────────────────────────


@pytest.mark.parametrize("env,expected", [
    ({"RUN_TOKEN": "rehearsal-2026-09-23-2"}, True),
    ({"AE_PIPELINE_ROLE": "rehearsal"}, True),
    ({"AE_PIPELINE_ROLE": "Rehearsal "}, True),
    # Real weekly runs, their recovery reruns and shell runs are NOT
    # rehearsals — execution names measured live 2026-09-24.
    ({"RUN_TOKEN": "watch-rerun-2026-09-18-4"}, False),
    ({"RUN_TOKEN": "2b6ab316-8060-4011-8140-cccf0f2194bd"}, False),
    ({"RUN_TOKEN": ""}, False),
    ({}, False),
    ({"AE_PIPELINE_ROLE": "weekly"}, False),
])
def test_rehearsal_reason(env, expected):
    assert (rehearsal_reason(env) is not None) is expected


def test_freeze_if_rehearsal_forces_freeze():
    args = argparse.Namespace(freeze=False)
    reason = freeze_if_rehearsal(
        args, entrypoint="evaluate.py", env={"RUN_TOKEN": "rehearsal-x-1"},
    )
    assert args.freeze is True
    assert "rehearsal-x-1" in reason


def test_freeze_if_rehearsal_leaves_real_runs_alone():
    args = argparse.Namespace(freeze=False)
    assert freeze_if_rehearsal(
        args, entrypoint="evaluate.py", env={"RUN_TOKEN": "watch-rerun-2026-09-18-4"},
    ) is None
    assert args.freeze is False


def test_backstop_raises_only_in_rehearsal():
    refuse_live_config_write_in_rehearsal("config/x.json", env={})
    with pytest.raises(RehearsalLiveWriteRefused, match="config/x.json"):
        refuse_live_config_write_in_rehearsal(
            "config/x.json", env={"RUN_TOKEN": "rehearsal-2026-09-23-2"},
        )


# ── entrypoints: rehearsal ⇒ --freeze before anything else ───────────────


@pytest.mark.parametrize("entry", ["evaluate.py", "backtest.py"])
def test_entrypoint_freezes_rehearsal_right_after_parse(entry):
    src = (REPO / entry).read_text()
    m = re.search(
        r"args = _parse_args\(\)\n(?:\s*#.*\n)*\s*from optimizer\.run_role import "
        r"freeze_if_rehearsal\n\s*freeze_if_rehearsal\(args, ",
        src,
    )
    assert m, f"{entry} must call freeze_if_rehearsal immediately after _parse_args()"


def test_evaluate_freeze_skips_assembler_and_all_config_writes():
    # The assembler (the only cutover writer) is gated on `not args.freeze`.
    src = (REPO / "evaluate.py").read_text()
    assert "if run_optimizers and opt_stage_error is None and not args.freeze:" in src


def test_backtest_predictor_apply_is_freeze_gated():
    src = (REPO / "backtest.py").read_text()
    assert 'config["_freeze"] = bool(args.freeze)' in src
    assert 'if config.get("_freeze"):' in src


# ── backstop 1: the assembler's cutover never writes in a rehearsal ──────


def _trigger_artifact(run_date="2026-09-23"):
    from optimizer.recommendation_artifact import RecommendationArtifact

    return RecommendationArtifact(
        fit_target="entry_timing_alpha",
        optimizer_name="trigger_optimizer",
        run_date=run_date,
        recommendation_kind="field_overlay",
        recommended_params={"disabled_triggers": [],
                            "disabled_triggers_updated_at": "2026-09-24"},
        overlay_keys=["disabled_triggers", "disabled_triggers_updated_at"],
        promotion_intent="promote",
    )


def _executor_artifact(intent, run_date="2026-09-23"):
    from optimizer.recommendation_artifact import RecommendationArtifact

    return RecommendationArtifact(
        fit_target="skill_composite",
        optimizer_name="executor_optimizer",
        run_date=run_date,
        recommendation_kind="full_replace",
        recommended_params={"min_score": 60, "max_position_pct": 0.2},
        promotion_intent=intent,
    )


def _stub_s3(current_live, artifacts):
    from botocore.exceptions import ClientError

    s3 = MagicMock()
    keys = {
        f"config/executor_params/recommendations/{a.run_date}/from_{a.optimizer_name}.json": a
        for a in artifacts
    }

    def list_side_effect(Bucket, Prefix):
        return {"Contents": [{"Key": k} for k in keys if k.startswith(Prefix)]}

    def get_side_effect(Bucket, Key):
        if Key == "config/executor_params.json":
            return {"Body": MagicMock(read=lambda: json.dumps(current_live).encode())}
        if Key in keys:
            body = keys[Key].to_json()
            return {"Body": MagicMock(read=lambda b=body: b.encode())}
        raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

    s3.list_objects_v2.side_effect = list_side_effect
    s3.get_object.side_effect = get_side_effect
    return s3


LIVE = {"min_score": 75, "max_position_pct": 0.1, "atr_multiplier": 2.0,
        "disabled_triggers": [], "updated_at": "2026-09-18"}


def _live_writes(s3):
    return [
        c for c in s3.put_object.call_args_list
        if c.kwargs["Key"] == "config/executor_params.json"
    ] + list(s3.copy_object.call_args_list)


def test_rehearsal_cutover_writes_nothing_live(monkeypatch):
    from optimizer.assembler import assemble

    monkeypatch.setenv("RUN_TOKEN", "rehearsal-2026-09-23-2")
    s3 = _stub_s3(LIVE, [_trigger_artifact()])
    with pytest.raises(RehearsalLiveWriteRefused):
        assemble("b", "executor_params", "2026-09-23", s3_client=s3,
                 write_assembled=False, cutover_enabled=True)
    assert _live_writes(s3) == [], "no live key and no _previous snapshot"


def test_real_run_cutover_still_writes(monkeypatch):
    from optimizer.assembler import assemble

    monkeypatch.setenv("RUN_TOKEN", "watch-rerun-2026-09-18-4")
    s3 = _stub_s3(LIVE, [_trigger_artifact()])
    result = assemble("b", "executor_params", "2026-09-23", s3_client=s3,
                      write_assembled=False, cutover_enabled=True)
    assert result.cutover_status == "applied"
    assert _live_writes(s3)


# ── backstop 2: predictor_params ─────────────────────────────────────────


def test_rehearsal_predictor_apply_refused_before_any_s3_call(monkeypatch):
    from optimizer import predictor_optimizer

    monkeypatch.setenv("RUN_TOKEN", "rehearsal-2026-09-23-2")
    fake = MagicMock()
    monkeypatch.setattr(predictor_optimizer.boto3, "client", lambda *a, **k: fake)
    with pytest.raises(RehearsalLiveWriteRefused, match="predictor_params"):
        predictor_optimizer.apply_recommendations(
            {"recommended_mode": "stacked"}, None, "b",
        )
    fake.put_object.assert_not_called()


# ── the launcher carries the execution name onto the spot ────────────────


@pytest.mark.parametrize("script", ["infrastructure/_spot_common.sh",
                                    "infrastructure/spot_backtest.sh"])
def test_launchers_forward_run_token(script):
    text = (REPO / script).read_text()
    assert "export RUN_TOKEN='" in text, f"{script} must forward RUN_TOKEN to the spot"
    assert "//[^A-Za-z0-9._-]/}" in text, "the forwarded value must be sanitised"


def test_run_role_env_names_match_krepis():
    from krepis import ssm_log_capture

    assert run_role.RUN_TOKEN_ENV == ssm_log_capture.CORRELATION_ID_ENV_VAR


# ── decided ordering: apply_audit is a record; overlays merge on their own ──


def test_blocked_sweep_params_are_not_written_but_the_overlay_is():
    """The rehearsal log: trigger overlay promoted + cut over, then
    apply_audit[executor_params]=blocked(alpha_floor). By design: the audit
    judges only executor_optimizer's sweep params, and those were NOT
    written — the live write carries the base plus the overlay's keys."""
    from optimizer.apply_audit import classify_loop, summarize_assembler
    from optimizer.assembler import assemble

    s3 = _stub_s3(LIVE, [_trigger_artifact(), _executor_artifact("skip")])
    result = assemble("b", "executor_params", "2026-09-23", s3_client=s3,
                      write_assembled=False, cutover_enabled=True)
    assert result.cutover_status == "applied"
    live_body = json.loads(next(
        c.kwargs["Body"] for c in s3.put_object.call_args_list
        if c.kwargs["Key"] == "config/executor_params.json"
    ))
    assert live_body["min_score"] == 75 and live_body["max_position_pct"] == 0.1
    assert live_body["disabled_triggers"] == []

    record = classify_loop(
        "executor_params",
        {"status": "alpha_below_floor", "blocked_by": ["alpha_floor"],
         "recommendation_reason": "best alpha below floor"},
        assembler_summary=summarize_assembler(result),
    )
    assert record["outcome"] == "blocked"
    assert record["blocked_by"] == ["alpha_floor"]


def test_apply_audit_runs_after_the_assembler_in_evaluate():
    src = (REPO / "evaluate.py").read_text()
    assert src.index('registry.phase("evaluator_assembler")') < src.index(
        'registry.phase("evaluator_apply_audit")'
    )
