"""Unit tests for the batch replay pipeline (PR C).

Covers:

- Listing: per-day pagination + agent_filter + meta-prefix exclusion
  (_eval, _analysis, _cost, _replay, _replay_summary).
- Aggregation: per-(agent_id_base, target_model) mean/min/max/stdev,
  thin-sample skip threshold.
- CloudWatch emission shape (dimensions + metric names).
- Per-target-model summary persisted to the canonical S3 path.
- Dry-run path: lists candidate artifacts without LLM calls.
- Multi-target replay: correct grouping when 2 target models are
  passed.
- Cost cap: max_artifacts truncates when corpus exceeds the cap.
- Failure isolation: a single replay raise keeps the batch going.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


# ── Fixtures ─────────────────────────────────────────────────────────────


def _make_artifact(agent_id: str, run_id: str = "r1") -> dict:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "timestamp": "2026-05-03T12:00:00Z",
        "agent_id": agent_id,
        "model_metadata": {"model_name": "claude-sonnet-4-6"},
        "full_prompt_context": {
            "system_prompt": "s", "user_prompt": "u", "tool_definitions": [],
        },
        "input_data_snapshot": {},
        "agent_output": {"ranked_picks": [{"ticker": "X", "quant_score": 80, "rationale": "ok"}]},
    }


def _build_s3_stub_with_artifacts(artifacts_by_key: dict[str, dict]) -> MagicMock:
    """Build a stub S3 client backed by per-day prefix listing."""
    s3 = MagicMock()

    by_prefix: dict[str, list[str]] = {}
    for key in artifacts_by_key:
        parts = key.split("/")
        prefix = "/".join(parts[:4]) + "/"
        by_prefix.setdefault(prefix, []).append(key)

    paginator = MagicMock()

    def paginate(*, Bucket, Prefix):
        return [{"Contents": [{"Key": k} for k in by_prefix.get(Prefix, [])]}]

    paginator.paginate.side_effect = paginate
    s3.get_paginator.return_value = paginator

    def get_object(*, Bucket, Key):
        body = MagicMock()
        body.read.return_value = json.dumps(artifacts_by_key[Key]).encode("utf-8")
        return {"Body": body}

    s3.get_object.side_effect = get_object
    s3.put_object = MagicMock()
    return s3


def _stub_replay(
    *, agreement_score: float, agent_id_base: str = "sector_quant",
    served_provider: str | None = None,
):
    """Build a stand-in for replay_artifact's ReplayOutput."""
    from replay.runner import ReplayOutput

    replay_cost = {"input_tokens": 100, "output_tokens": 50}
    if served_provider is not None:
        replay_cost["served_provider"] = served_provider
    return ReplayOutput(
        original_run_id="r1",
        original_agent_id=f"{agent_id_base}:tech",
        original_model="claude-sonnet-4-6",
        replay_model="claude-haiku-4-5",
        replay_output={"ranked_picks": []},
        replay_output_kind="structured",
        replay_cost=replay_cost,
        replay_latency_ms=200,
        comparison={
            "agreement_score": agreement_score,
            "scorer": agent_id_base,
            "agent_id_base": agent_id_base,
            "diff_summary": f"agreement={agreement_score:.2f}",
        },
    )


# ── Listing ──────────────────────────────────────────────────────────────


class TestListArtifactKeys:
    def test_excludes_meta_prefixes(self):
        from replay.batch import _list_artifact_keys_in_window

        end = datetime(2026, 5, 5, tzinfo=timezone.utc)
        artifacts = {
            "decision_artifacts/2026/05/05/sector_quant:tech/r1.json": {},
            "decision_artifacts/2026/05/05/_eval/x.json": {},
            "decision_artifacts/2026/05/05/_replay/y.json": {},
            "decision_artifacts/2026/05/05/_replay_summary/z.json": {},
            "decision_artifacts/2026/05/05/_cost/c.json": {},
            "decision_artifacts/2026/05/05/_analysis/a.json": {},
        }
        s3 = _build_s3_stub_with_artifacts(artifacts)
        # Adjust paginator to return ALL keys for the day-prefix only.
        s3.get_paginator.return_value.paginate.side_effect = (
            lambda *, Bucket, Prefix: [{
                "Contents": [{"Key": k} for k in artifacts
                             if k.startswith(Prefix)]
            }] if Prefix == "decision_artifacts/2026/05/05/" else [{"Contents": []}]
        )

        keys = _list_artifact_keys_in_window(
            s3, bucket="b", capture_prefix="decision_artifacts",
            end_date=end, window_days=1,
        )
        # Only the production capture should remain.
        assert keys == [
            "decision_artifacts/2026/05/05/sector_quant:tech/r1.json"
        ]

    def test_agent_filter_excludes_unmatched(self):
        from replay.batch import _list_artifact_keys_in_window

        end = datetime(2026, 5, 5, tzinfo=timezone.utc)
        artifacts = {
            "decision_artifacts/2026/05/05/sector_quant:tech/r1.json": {},
            "decision_artifacts/2026/05/05/macro_economist/r2.json": {},
            "decision_artifacts/2026/05/05/ic_cio/r3.json": {},
        }
        s3 = _build_s3_stub_with_artifacts(artifacts)

        keys = _list_artifact_keys_in_window(
            s3, bucket="b", capture_prefix="decision_artifacts",
            end_date=end, window_days=1,
            agent_filter=["sector_quant", "ic_cio"],
        )
        # macro_economist is filtered out.
        assert sorted(keys) == [
            "decision_artifacts/2026/05/05/ic_cio/r3.json",
            "decision_artifacts/2026/05/05/sector_quant:tech/r1.json",
        ]


class TestAgentIdBaseFromKey:
    def test_strips_namespace(self):
        from replay.batch import _agent_id_base_from_key

        assert _agent_id_base_from_key(
            "decision_artifacts/2026/05/05/sector_quant:technology/r1.json"
        ) == "sector_quant"

    def test_plain_agent_id(self):
        from replay.batch import _agent_id_base_from_key

        assert _agent_id_base_from_key(
            "decision_artifacts/2026/05/05/macro_economist/r1.json"
        ) == "macro_economist"


# ── Aggregation ──────────────────────────────────────────────────────────


class TestAggregateGroup:
    def test_typical_observations(self):
        from replay.batch import _aggregate_group

        agg = _aggregate_group([0.9, 0.8, 0.95, 0.85])
        assert agg["n"] == 4
        assert 0.87 < agg["mean"] < 0.89
        assert agg["min"] == 0.8
        assert agg["max"] == 0.95
        assert agg["stdev"] > 0

    def test_single_observation(self):
        from replay.batch import _aggregate_group

        agg = _aggregate_group([0.5])
        assert agg["n"] == 1
        assert agg["mean"] == 0.5
        assert agg["stdev"] == 0.0

    def test_empty_returns_none_mean(self):
        from replay.batch import _aggregate_group

        agg = _aggregate_group([])
        assert agg["n"] == 0
        assert agg["mean"] is None

    def test_filters_non_numeric_defensively(self):
        from replay.batch import _aggregate_group

        agg = _aggregate_group([0.9, None, 0.8, "bad"])  # type: ignore[list-item]
        assert agg["n"] == 2  # Only 0.9 and 0.8 survive.
        assert agg["mean"] == pytest.approx(0.85)


# ── End-to-end pipeline ──────────────────────────────────────────────────


class TestComputeAndEmitConcordance:
    def test_dry_run_lists_without_replay(self):
        from replay.batch import compute_and_emit_concordance

        end = datetime(2026, 5, 9, tzinfo=timezone.utc)
        artifacts = {
            f"decision_artifacts/2026/05/09/sector_quant:tech/r{i}.json":
                _make_artifact("sector_quant:tech", f"r{i}")
            for i in range(5)
        }
        s3 = _build_s3_stub_with_artifacts(artifacts)

        with patch("replay.batch.replay_artifact") as mock_replay:
            summary = compute_and_emit_concordance(
                target_models=["claude-haiku-4-5"],
                end_time=end, window_days=1,
                s3_client=s3,
                dry_run=True,
            )

        assert summary["dry_run"] is True
        assert summary["would_replay"] == 5
        # No replay calls fired.
        mock_replay.assert_not_called()
        # No put_object — dry-run persists nothing.
        s3.put_object.assert_not_called()

    def test_thin_sample_skipped(self):
        """Below MIN_OBSERVATIONS_FOR_CONCORDANCE the per-group mean
        is statistically meaningless — group should be reported as
        skipped rather than emitted."""
        from replay.batch import compute_and_emit_concordance

        end = datetime(2026, 5, 9, tzinfo=timezone.utc)
        # Only 2 artifacts → below the floor of 3.
        artifacts = {
            "decision_artifacts/2026/05/09/sector_quant:tech/r1.json":
                _make_artifact("sector_quant:tech", "r1"),
            "decision_artifacts/2026/05/09/sector_quant:tech/r2.json":
                _make_artifact("sector_quant:tech", "r2"),
        }
        s3 = _build_s3_stub_with_artifacts(artifacts)
        cw = MagicMock()

        with patch("replay.batch.replay_artifact",
                   return_value=_stub_replay(agreement_score=0.9)):
            summary = compute_and_emit_concordance(
                target_models=["claude-haiku-4-5"],
                end_time=end, window_days=1,
                s3_client=s3, cloudwatch_client=cw,
            )

        target = summary["per_target_model"][0]
        assert target["agents_analyzed"] == 0
        assert target["agents_skipped_thin_sample"][0]["agent_id_base"] == "sector_quant"
        # No CW emission for skipped group.
        cw.put_metric_data.assert_not_called()

    def test_served_providers_seen_deduped_and_sorted(self):
        # config#3006 — batch-level jurisdiction observation: which
        # upstream backends actually served this run, deduped across
        # every replayed artifact.
        from replay.batch import compute_and_emit_concordance

        end = datetime(2026, 5, 9, tzinfo=timezone.utc)
        artifacts = {
            f"decision_artifacts/2026/05/09/sector_quant:tech/r{i}.json":
                _make_artifact("sector_quant:tech", f"r{i}")
            for i in range(3)
        }
        s3 = _build_s3_stub_with_artifacts(artifacts)
        providers = ["DeepInfra", "AtlasCloud", "DeepInfra"]

        def fake_replay(*, artifact_key, target_model, **kwargs):
            return _stub_replay(
                agreement_score=0.9, served_provider=providers.pop(0),
            )

        with patch("replay.batch.replay_artifact", side_effect=fake_replay):
            summary = compute_and_emit_concordance(
                target_models=["claude-haiku-4-5"],
                end_time=end, window_days=1,
                s3_client=s3,
            )

        target = summary["per_target_model"][0]
        assert target["served_providers_seen"] == ["AtlasCloud", "DeepInfra"]

    def test_served_providers_seen_empty_when_none_reported(self):
        # Pre-v0.18.0 krepis pin, or a provider that doesn't report the
        # field — informational absence, not a failure.
        from replay.batch import compute_and_emit_concordance

        end = datetime(2026, 5, 9, tzinfo=timezone.utc)
        artifacts = {
            f"decision_artifacts/2026/05/09/sector_quant:tech/r{i}.json":
                _make_artifact("sector_quant:tech", f"r{i}")
            for i in range(3)
        }
        s3 = _build_s3_stub_with_artifacts(artifacts)

        with patch("replay.batch.replay_artifact",
                   return_value=_stub_replay(agreement_score=0.9)):
            summary = compute_and_emit_concordance(
                target_models=["claude-haiku-4-5"],
                end_time=end, window_days=1,
                s3_client=s3,
            )

        target = summary["per_target_model"][0]
        assert target["served_providers_seen"] == []

    def test_aggregates_per_agent_and_emits(self):
        from replay.batch import compute_and_emit_concordance

        end = datetime(2026, 5, 9, tzinfo=timezone.utc)
        # 4 sector_quant + 4 ic_cio captures → both above the floor of 3.
        artifacts = {}
        for i in range(4):
            artifacts[
                f"decision_artifacts/2026/05/09/sector_quant:tech/r{i}.json"
            ] = _make_artifact("sector_quant:tech", f"sq-{i}")
            artifacts[
                f"decision_artifacts/2026/05/09/ic_cio/r{i}.json"
            ] = _make_artifact("ic_cio", f"cio-{i}")
        s3 = _build_s3_stub_with_artifacts(artifacts)
        cw = MagicMock()

        # Stub replay_artifact to return varied agreement scores grouped
        # by agent so the aggregator has real data to summarize.
        def fake_replay(*, artifact_key, target_model, **kwargs):
            base = "sector_quant" if "sector_quant" in artifact_key else "ic_cio"
            score = 0.95 if base == "sector_quant" else 0.6
            return _stub_replay(agreement_score=score, agent_id_base=base)

        with patch("replay.batch.replay_artifact", side_effect=fake_replay):
            summary = compute_and_emit_concordance(
                target_models=["claude-haiku-4-5"],
                end_time=end, window_days=1,
                s3_client=s3, cloudwatch_client=cw,
            )

        target = summary["per_target_model"][0]
        assert target["agents_analyzed"] == 2
        per_agent = {a["agent_id_base"]: a for a in target["per_agent"]}
        assert per_agent["sector_quant"]["mean"] == 0.95
        assert per_agent["ic_cio"]["mean"] == 0.6

        # Two CW put calls — one per agent group. Each call carries
        # both the primary metric + the _n_observations counter.
        assert cw.put_metric_data.call_count == 2
        names_emitted = []
        for call in cw.put_metric_data.call_args_list:
            for d in call.kwargs["MetricData"]:
                names_emitted.append(d["MetricName"])
        assert "agent_cheap_model_concordance" in names_emitted
        assert "agent_cheap_model_concordance_n_observations" in names_emitted

        # Summary JSON persisted under the canonical eval_artifacts layout:
        # a flat dated key {run_id}_{target}.json + a latest.json sidecar.
        put_keys = [c.kwargs["Key"] for c in s3.put_object.call_args_list]
        assert all(
            c.kwargs["Bucket"] == "alpha-engine-research"
            for c in s3.put_object.call_args_list
        )
        # end_time=2026-05-09 00:00 UTC → run_id "2605090000".
        dated_keys = [
            k for k in put_keys
            if not k.endswith("/latest.json")
        ]
        latest_keys = [k for k in put_keys if k.endswith("/latest.json")]
        assert dated_keys == [
            "decision_artifacts/_replay_summary/2605090000_claude-haiku-4-5.json"
        ]
        assert latest_keys == ["decision_artifacts/_replay_summary/latest.json"]
        # No legacy {YYYY-MM-DD}/ date partition in the new layout.
        assert all("/2026-05-09/" not in k for k in put_keys)

    def test_multi_target_model_grouping(self):
        from replay.batch import compute_and_emit_concordance

        end = datetime(2026, 5, 9, tzinfo=timezone.utc)
        artifacts = {
            f"decision_artifacts/2026/05/09/sector_quant:tech/r{i}.json":
                _make_artifact("sector_quant:tech", f"r{i}")
            for i in range(3)
        }
        s3 = _build_s3_stub_with_artifacts(artifacts)
        cw = MagicMock()

        def fake_replay(*, target_model, **kwargs):
            score = 0.9 if target_model == "claude-haiku-4-5" else 0.99
            return _stub_replay(agreement_score=score)

        with patch("replay.batch.replay_artifact", side_effect=fake_replay):
            summary = compute_and_emit_concordance(
                target_models=["claude-haiku-4-5", "claude-sonnet-4-6"],
                end_time=end, window_days=1,
                s3_client=s3, cloudwatch_client=cw,
            )

        # Both target models analyzed, each gets its own summary.
        assert len(summary["per_target_model"]) == 2
        means_by_target = {
            t["target_model"]: t["per_agent"][0]["mean"]
            for t in summary["per_target_model"]
            if t["per_agent"]
        }
        assert means_by_target["claude-haiku-4-5"] == pytest.approx(0.9)
        assert means_by_target["claude-sonnet-4-6"] == pytest.approx(0.99)
        # 2 dated summary JSONs (one per target_model) + 2 latest.json
        # sidecar mirrors = 4 put_object calls. Each target writes its own
        # dated key but all mirror into the single shared latest sidecar.
        put_keys = [c.kwargs["Key"] for c in s3.put_object.call_args_list]
        dated_keys = sorted(k for k in put_keys if not k.endswith("/latest.json"))
        assert dated_keys == [
            "decision_artifacts/_replay_summary/2605090000_claude-haiku-4-5.json",
            "decision_artifacts/_replay_summary/2605090000_claude-sonnet-4-6.json",
        ]
        assert s3.put_object.call_count == 4

    def test_max_artifacts_caps_corpus(self):
        from replay.batch import compute_and_emit_concordance

        end = datetime(2026, 5, 9, tzinfo=timezone.utc)
        # 10 artifacts, max_artifacts=3.
        artifacts = {
            f"decision_artifacts/2026/05/09/sector_quant:tech/r{i}.json":
                _make_artifact("sector_quant:tech", f"r{i}")
            for i in range(10)
        }
        s3 = _build_s3_stub_with_artifacts(artifacts)
        cw = MagicMock()

        with patch("replay.batch.replay_artifact",
                   return_value=_stub_replay(agreement_score=0.9)) as m:
            summary = compute_and_emit_concordance(
                target_models=["claude-haiku-4-5"],
                end_time=end, window_days=1, max_artifacts=3,
                s3_client=s3, cloudwatch_client=cw,
            )

        assert summary["artifacts_discovered"] == 3  # cap applied.
        # replay_artifact called only 3× even though corpus had 10.
        assert m.call_count == 3

    def test_replay_failure_does_not_halt_batch(self):
        from replay.batch import compute_and_emit_concordance

        end = datetime(2026, 5, 9, tzinfo=timezone.utc)
        artifacts = {
            f"decision_artifacts/2026/05/09/sector_quant:tech/r{i}.json":
                _make_artifact("sector_quant:tech", f"r{i}")
            for i in range(4)
        }
        s3 = _build_s3_stub_with_artifacts(artifacts)
        cw = MagicMock()

        # First call raises; remaining 3 succeed.
        side_effects = [
            RuntimeError("transient API failure"),
            _stub_replay(agreement_score=0.9),
            _stub_replay(agreement_score=0.85),
            _stub_replay(agreement_score=0.92),
        ]
        with patch("replay.batch.replay_artifact", side_effect=side_effects):
            summary = compute_and_emit_concordance(
                target_models=["claude-haiku-4-5"],
                end_time=end, window_days=1,
                s3_client=s3, cloudwatch_client=cw,
            )

        target = summary["per_target_model"][0]
        # 3 successful observations → above thin-sample floor.
        assert target["agents_analyzed"] == 1
        assert target["per_agent"][0]["n"] == 3
        # 1 failure recorded but didn't halt.
        assert any(
            f["stage"] == "replay_artifact_call"
            for f in target["replay_failures"]
        )


# ── Skips counted separately from failures (config#1035) ────────────────


class TestReplaySkipsCountedSeparately:
    def test_skipped_replay_lands_in_replay_skips_not_failures(self):
        from datetime import datetime, timezone
        from unittest.mock import MagicMock, patch

        from replay.batch import compute_and_emit_concordance
        from replay.runner import ReplayOutput

        end = datetime(2026, 5, 9, tzinfo=timezone.utc)
        artifacts = {
            f"decision_artifacts/2026/05/09/sector_quant:tech/r{i}.json":
                _make_artifact("sector_quant:tech", f"r{i}")
            for i in range(4)
        }
        s3 = _build_s3_stub_with_artifacts(artifacts)
        cw = MagicMock()

        skipped = ReplayOutput(
            original_run_id="r0",
            original_agent_id="thesis_update:consumer:GOOG",
            original_model="claude-haiku-4-5",
            replay_model="claude-haiku-4-5",
            replay_output={},
            replay_output_kind="skipped",
            replay_cost={},
            replay_latency_ms=0,
            replay_error=(
                "placeholder prompt context (capture wiring gap) — "
                "nothing meaningful to replay"
            ),
            comparison={
                "agreement_score": 0.0,
                "diff_summary": "skipped — placeholder prompt context",
                "scorer": "skipped",
                "agent_id_base": "thesis_update",
            },
        )
        side_effects = [
            skipped,
            _stub_replay(agreement_score=0.9),
            _stub_replay(agreement_score=0.85),
            _stub_replay(agreement_score=0.92),
        ]
        with patch("replay.batch.replay_artifact", side_effect=side_effects):
            summary = compute_and_emit_concordance(
                target_models=["claude-haiku-4-5"],
                end_time=end, window_days=1,
                s3_client=s3, cloudwatch_client=cw,
            )

        target = summary["per_target_model"][0]
        # The skip is its own counted category — NOT a replay failure.
        assert target["replay_failures"] == []
        assert len(target["replay_skips"]) == 1
        assert target["replay_skips"][0]["stage"] == "skipped"
        assert "placeholder prompt context" in target["replay_skips"][0]["reason"]
        # The skipped artifact contributes no concordance observation.
        assert target["agents_analyzed"] == 1
        assert target["per_agent"][0]["n"] == 3
