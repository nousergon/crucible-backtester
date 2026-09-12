"""Zero-candidate-corpus observability (alpha-engine-config-I7183).

Covers the fix for the 2026-09-12 weekly-SF `FAILED / DegradedRun`:
``ReplayConcordance`` ran OK and legitimately made zero LLM calls (its
default ``agent_filter`` matched no artifact newer than 2026-07-11), so it
emitted no cost record, and ``AggregateCosts``' fan-in coverage check —
which reads ``replay-concordance`` as a ``required_producers`` entry —
read the correctly-silent stage identically to one that swallowed its
records.

Three things this file pins:

1. ``_emit_zero_call_cost_record`` hands the configured cost sink exactly
   one priced-at-zero, explicitly-marked record per target model, and
   never raises regardless of what the sink does.
2. ``compute_and_emit_concordance`` calls it (and only it — no real
   ``replay_artifact`` call) exactly when the window's candidate corpus is
   empty, once per target model, and does NOT call it on a non-empty
   corpus or on a dry run.
3. The empty-corpus finding is logged at ERROR and published as an ops
   alert exactly once per invocation (not once per target model), naming
   the configured ``agent_filter``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest


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
    s3.put_object = MagicMock()
    return s3


def _empty_s3_stub() -> MagicMock:
    """An S3 client whose every prefix listing returns nothing."""
    s3 = MagicMock()
    paginator = MagicMock()
    paginator.paginate.side_effect = lambda *, Bucket, Prefix: [{"Contents": []}]
    s3.get_paginator.return_value = paginator
    s3.put_object = MagicMock()
    return s3


def _resolve_target_spec_stub(model_id, *, max_tokens=8192):
    spec = MagicMock()
    spec.provider = "openrouter"
    spec.model = f"resolved/{model_id}"
    return spec, {"route": "litellm_proxy", "deployment_id": model_id,
                  "exec_context": "lambda"}


# ── _emit_zero_call_cost_record ─────────────────────────────────────────


class TestEmitZeroCallCostRecord:
    def test_hands_sink_a_zeroed_marked_record(self):
        from replay.batch import _emit_zero_call_cost_record

        sink = MagicMock()
        spec = MagicMock(provider="openrouter", model="deepseek/deepseek-v4-flash")

        with patch("krepis.cost_sink.default_sink_from_env", return_value=sink):
            _emit_zero_call_cost_record("deepseek-v4-flash", spec)

        sink.assert_called_once()
        record = sink.call_args[0][0]
        assert record["callsite_id"] == "replay-concordance"
        assert record["input_tokens"] == 0
        assert record["output_tokens"] == 0
        assert record["cost_usd"] == 0.0
        assert record["cost_source"] == "producer_ran_no_calls"
        assert record["event"] == "producer_ran_no_calls"
        assert record["n_calls"] == 0
        assert record["provider"] == "openrouter"
        assert record["model"] == "deepseek/deepseek-v4-flash"
        assert "schema_version" in record
        assert "ts" in record

    def test_noop_when_no_sink_configured(self):
        from replay.batch import _emit_zero_call_cost_record

        spec = MagicMock(provider="openrouter", model="m")
        with patch("krepis.cost_sink.default_sink_from_env", return_value=None):
            _emit_zero_call_cost_record("deepseek-v4-flash", spec)
        # No exception — nothing to assert on beyond "did not raise".

    def test_never_raises_when_sink_call_fails(self):
        from replay.batch import _emit_zero_call_cost_record

        sink = MagicMock(side_effect=RuntimeError("S3 unreachable"))
        spec = MagicMock(provider="openrouter", model="m")
        with patch("krepis.cost_sink.default_sink_from_env", return_value=sink):
            _emit_zero_call_cost_record("deepseek-v4-flash", spec)  # must not raise

    def test_falls_back_to_target_model_when_spec_unset(self):
        """A spec missing provider/model (defensive — resolve_target_spec is
        stubbed in some tests) still produces a usable record."""
        from replay.batch import _emit_zero_call_cost_record

        sink = MagicMock()
        spec = MagicMock(spec=[])  # no provider/model attrs at all
        with patch("krepis.cost_sink.default_sink_from_env", return_value=sink):
            _emit_zero_call_cost_record("deepseek-v4-flash", spec)

        record = sink.call_args[0][0]
        assert record["model"] == "deepseek-v4-flash"
        assert record["provider"] == "unknown"


# ── compute_and_emit_concordance wiring ─────────────────────────────────


class TestEmptyCorpusWiring:
    def test_emits_one_zero_call_record_per_target_on_empty_corpus(self):
        from replay import batch as batch_mod

        end = datetime(2026, 9, 12, tzinfo=timezone.utc)
        s3 = _empty_s3_stub()
        sink = MagicMock()

        with patch.object(batch_mod, "resolve_target_spec",
                           side_effect=_resolve_target_spec_stub), \
             patch.object(batch_mod, "replay_artifact") as mock_replay, \
             patch("krepis.cost_sink.default_sink_from_env", return_value=sink), \
             patch.object(batch_mod, "_publish_empty_corpus_alert") as mock_alert:
            summary = batch_mod.compute_and_emit_concordance(
                target_models=["deepseek-v4-flash", "claude-haiku-4-5"],
                end_time=end, window_days=56,
                s3_client=s3, emit_metrics=False,
            )

        mock_replay.assert_not_called()
        assert sink.call_count == 2, "one zero-call record per target model"
        emitted_models = {c.args[0].get("model") for c in sink.call_args_list}
        assert "resolved/deepseek-v4-flash" in emitted_models
        assert "resolved/claude-haiku-4-5" in emitted_models
        for target_summary in summary["per_target_model"]:
            assert target_summary["n_artifacts_candidate"] == 0
            assert target_summary["n_artifacts_replayed"] == 0
        # The alert is a per-invocation finding, not per-target.
        mock_alert.assert_called_once()

    def test_does_not_emit_zero_call_record_on_nonempty_corpus(self):
        from replay import batch as batch_mod

        end = datetime(2026, 5, 9, tzinfo=timezone.utc)
        artifacts = {
            f"decision_artifacts/2026/05/09/sector_quant:tech/r{i}.json":
                _make_artifact("sector_quant:tech", f"r{i}")
            for i in range(3)
        }
        s3 = _build_s3_stub_with_artifacts(artifacts)

        from replay.runner import ReplayOutput
        stub_replay = ReplayOutput(
            original_run_id="r1", original_agent_id="sector_quant:tech",
            original_model="claude-sonnet-4-6", replay_model="claude-haiku-4-5",
            replay_output={"ranked_picks": []}, replay_output_kind="structured",
            replay_cost={"input_tokens": 10, "output_tokens": 5},
            replay_latency_ms=200,
            comparison={"agreement_score": 0.9, "scorer": "sector_quant",
                        "agent_id_base": "sector_quant",
                        "diff_summary": "agreement=0.90"},
        )

        with patch.object(batch_mod, "resolve_target_spec",
                           side_effect=_resolve_target_spec_stub), \
             patch.object(batch_mod, "replay_artifact", return_value=stub_replay), \
             patch.object(batch_mod, "_emit_zero_call_cost_record") as mock_zero, \
             patch.object(batch_mod, "_publish_empty_corpus_alert") as mock_alert:
            batch_mod.compute_and_emit_concordance(
                target_models=["deepseek-v4-flash"],
                end_time=end, window_days=1,
                s3_client=s3, emit_metrics=False,
            )

        mock_zero.assert_not_called()
        mock_alert.assert_not_called()

    def test_dry_run_never_touches_the_cost_sink_or_alert(self):
        from replay import batch as batch_mod

        end = datetime(2026, 9, 12, tzinfo=timezone.utc)
        s3 = _empty_s3_stub()

        with patch.object(batch_mod, "resolve_target_spec",
                           side_effect=_resolve_target_spec_stub), \
             patch.object(batch_mod, "_emit_zero_call_cost_record") as mock_zero, \
             patch.object(batch_mod, "_publish_empty_corpus_alert") as mock_alert:
            summary = batch_mod.compute_and_emit_concordance(
                target_models=["deepseek-v4-flash"],
                end_time=end, window_days=56,
                s3_client=s3, dry_run=True,
            )

        assert summary["dry_run"] is True
        mock_zero.assert_not_called()
        mock_alert.assert_not_called()


# ── empty-corpus log + alert ─────────────────────────────────────────────


class TestEmptyCorpusAlert:
    def test_logs_error_and_publishes_warning_alert(self, caplog):
        from replay import batch as batch_mod

        end = datetime(2026, 9, 12, tzinfo=timezone.utc)
        s3 = _empty_s3_stub()

        with patch.object(batch_mod, "resolve_target_spec",
                           side_effect=_resolve_target_spec_stub), \
             patch("krepis.cost_sink.default_sink_from_env", return_value=None), \
             patch("ops_alerts.publish_ops_alert") as mock_publish, \
             caplog.at_level("ERROR"):
            batch_mod.compute_and_emit_concordance(
                target_models=["deepseek-v4-flash"],
                end_time=end, window_days=56,
                s3_client=s3, emit_metrics=False,
            )

        assert any("zero-candidate corpus" in r.message for r in caplog.records)
        mock_publish.assert_called_once()
        _, kwargs = mock_publish.call_args
        assert kwargs["severity"] == "warning"
        assert "dedup_key" in kwargs

    def test_alert_publish_failure_never_raises(self):
        from replay import batch as batch_mod

        end = datetime(2026, 9, 12, tzinfo=timezone.utc)
        s3 = _empty_s3_stub()

        with patch.object(batch_mod, "resolve_target_spec",
                           side_effect=_resolve_target_spec_stub), \
             patch("krepis.cost_sink.default_sink_from_env", return_value=None), \
             patch("ops_alerts.publish_ops_alert",
                   side_effect=RuntimeError("SNS down")):
            batch_mod.compute_and_emit_concordance(
                target_models=["deepseek-v4-flash"],
                end_time=end, window_days=56,
                s3_client=s3, emit_metrics=False,
            )  # must not raise


# ── _find_last_matching_dated_prefix ────────────────────────────────────


class TestFindLastMatchingDatedPrefix:
    def test_finds_the_most_recent_matching_day(self):
        from replay.batch import _find_last_matching_dated_prefix

        before = datetime(2026, 9, 12, tzinfo=timezone.utc)
        match_day = before - timedelta(days=5)
        key = (
            f"decision_artifacts/{match_day.strftime('%Y')}/"
            f"{match_day.strftime('%m')}/{match_day.strftime('%d')}/"
            "sector_quant:tech/r1.json"
        )
        s3 = MagicMock()
        paginator = MagicMock()

        def paginate(*, Bucket, Prefix):
            if Prefix == key.rsplit("/", 2)[0] + "/":
                return [{"Contents": [{"Key": key}]}]
            return [{"Contents": []}]

        paginator.paginate.side_effect = paginate
        s3.get_paginator.return_value = paginator

        found = _find_last_matching_dated_prefix(
            s3, bucket="b", capture_prefix="decision_artifacts",
            before_date=before, agent_filter=["sector_quant"],
            max_lookback_days=30,
        )
        assert found == key.rsplit("/", 2)[0] + "/"

    def test_returns_none_when_nothing_matches_within_lookback(self):
        from replay.batch import _find_last_matching_dated_prefix

        s3 = _empty_s3_stub()
        found = _find_last_matching_dated_prefix(
            s3, bucket="b", capture_prefix="decision_artifacts",
            before_date=datetime(2026, 9, 12, tzinfo=timezone.utc),
            agent_filter=["sector_quant"],
            max_lookback_days=5,
        )
        assert found is None

    def test_never_raises_on_listing_failure(self):
        from replay.batch import _find_last_matching_dated_prefix

        s3 = MagicMock()
        s3.get_paginator.side_effect = RuntimeError("access denied")
        found = _find_last_matching_dated_prefix(
            s3, bucket="b", capture_prefix="decision_artifacts",
            before_date=datetime(2026, 9, 12, tzinfo=timezone.utc),
            agent_filter=None, max_lookback_days=5,
        )
        assert found is None
