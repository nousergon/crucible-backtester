"""Tests for analysis/provenance_grounding.py — fourth leg of the
agent-justification stack (tool-call + input-trace metrics on captured
decision artifacts)."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from analysis.provenance_grounding import (
    CANONICAL_AGENTS,
    TOOL_EQUIPPED_AGENTS,
    _agent_metrics,
    _artifact_metrics,
    _walk_tool_calls,
    compute_provenance_grounding,
)


# ── Walker ──────────────────────────────────────────────────────────────────


class TestWalkToolCalls:
    def test_top_level_tool_calls(self):
        agent_output = {
            "tool_calls": [
                {"tool": "fetch_news", "ticker": "AAPL"},
                {"tool": "rag_query", "ticker": None},
            ],
        }
        assert len(_walk_tool_calls(agent_output)) == 2

    def test_nested_under_quant_output(self):
        """sector_team agents stash tool_calls under quant_output / qual_output
        because the team is a sub-graph with sub-agents."""
        agent_output = {
            "team_id": "technology",
            "quant_output": {
                "tool_calls": [
                    {"tool": "screen_by_volume"},
                    {"tool": "screen_by_volume"},
                ]
            },
            "qual_output": {
                "tool_calls": [{"tool": "fetch_news"}],
            },
            "tool_calls": [],  # top-level, sector_team leaves empty
        }
        out = _walk_tool_calls(agent_output)
        assert len(out) == 3
        tools = sorted(tc["tool"] for tc in out)
        assert tools == ["fetch_news", "screen_by_volume", "screen_by_volume"]

    def test_no_tool_calls(self):
        assert _walk_tool_calls({"team_id": "x"}) == []
        assert _walk_tool_calls({}) == []

    def test_handles_non_list_tool_calls_gracefully(self):
        # Defensively: malformed shape (string instead of list) shouldn't crash
        agent_output = {"tool_calls": "broken"}
        assert _walk_tool_calls(agent_output) == []


# ── Per-artifact metrics ────────────────────────────────────────────────────


class TestArtifactMetrics:
    def _make(self, **overrides):
        artifact = {
            "agent_output": {
                "tool_calls": [
                    {"tool": "fetch_news", "ticker": "AAPL"},
                    {"tool": "fetch_news", "ticker": "MSFT"},
                    {"tool": "rag_query", "ticker": "AAPL"},
                ],
                "ranked_picks": [{"ticker": "AAPL", "rationale": "AAPL momentum"}],
            },
            "input_data_snapshot": {
                "team_id": "technology",
                "market_regime": "bull",
                "ranked_picks": [],  # key mentioned in output blob
            },
        }
        artifact.update(overrides)
        return artifact

    def test_basic_metrics(self):
        m = _artifact_metrics(self._make())
        assert m["n_tool_calls"] == 3
        assert m["n_distinct_tools"] == 2
        assert m["distinct_tools"] == ["fetch_news", "rag_query"]
        assert m["tool_distribution"] == {"fetch_news": 2, "rag_query": 1}

    def test_zero_tool_calls(self):
        artifact = self._make(agent_output={"team_id": "ic_cio"})
        m = _artifact_metrics(artifact)
        assert m["n_tool_calls"] == 0
        assert m["n_distinct_tools"] == 0

    def test_input_consumption_ratio(self):
        # input_data_snapshot has 3 keys; "ranked_picks" appears in output
        # blob. At minimum 1/3 of keys match; allow rounding-tolerance.
        m = _artifact_metrics(self._make())
        assert m["input_consumption_ratio"] >= 0.333

    def test_truncation_flag(self):
        artifact = self._make(input_data_truncated_at=2_000_000)
        assert _artifact_metrics(artifact)["is_truncated"] is True
        artifact = self._make(input_data_truncated_at=None)
        assert _artifact_metrics(artifact)["is_truncated"] is False


# ── Per-agent aggregation ───────────────────────────────────────────────────


class TestAgentMetrics:
    def test_empty(self):
        m = _agent_metrics([])
        assert m == {
            "n_artifacts": 0,
            "n_zero_call_artifacts": 0,
            "pct_zero_call_outputs": 0.0,
            "mean_n_tool_calls": 0.0,
            "mean_n_distinct_tools": 0.0,
            "tool_distribution": {},
            "mean_input_consumption_ratio": 0.0,
            "n_truncated": 0,
        }

    def test_aggregation(self):
        per_artifact = [
            {
                "n_tool_calls": 5,
                "n_distinct_tools": 2,
                "distinct_tools": ["a", "b"],
                "tool_distribution": {"a": 3, "b": 2},
                "input_consumption_ratio": 0.6,
                "input_snapshot_n_keys": 5,
                "is_truncated": False,
            },
            {
                "n_tool_calls": 0,
                "n_distinct_tools": 0,
                "distinct_tools": [],
                "tool_distribution": {},
                "input_consumption_ratio": 0.2,
                "input_snapshot_n_keys": 5,
                "is_truncated": True,
            },
        ]
        m = _agent_metrics(per_artifact)
        assert m["n_artifacts"] == 2
        assert m["n_zero_call_artifacts"] == 1
        assert m["pct_zero_call_outputs"] == 50.0
        assert m["mean_n_tool_calls"] == 2.5
        assert m["mean_n_distinct_tools"] == 1.0
        assert m["tool_distribution"] == {"a": 3, "b": 2}
        assert m["mean_input_consumption_ratio"] == 0.4
        assert m["n_truncated"] == 1


# ── Tool-equipped agent set ─────────────────────────────────────────────────


class TestToolEquippedAgents:
    def test_includes_macro_and_all_sector_quant_qual(self):
        """sector_quant + sector_qual are the tool-bearing sub-stages.
        sector_peer_review is a synthesizer (no fetch tools)."""
        assert "macro_economist" in TOOL_EQUIPPED_AGENTS
        for sector in ("consumer", "defensives", "financials",
                       "healthcare", "industrials", "technology"):
            assert f"sector_quant:{sector}" in TOOL_EQUIPPED_AGENTS
            assert f"sector_qual:{sector}" in TOOL_EQUIPPED_AGENTS
            # peer_review is a synthesizer — excluded from the alarm.
            assert f"sector_peer_review:{sector}" not in TOOL_EQUIPPED_AGENTS

    def test_excludes_synthesizers(self):
        # CIO + peer_review are synthesizers — legitimate zero-call agents
        assert "ic_cio" not in TOOL_EQUIPPED_AGENTS
        for sector in ("consumer", "defensives", "financials",
                       "healthcare", "industrials", "technology"):
            assert f"sector_peer_review:{sector}" not in TOOL_EQUIPPED_AGENTS

    def test_tool_equipped_size_is_13(self):
        # 1 macro + 6 sector_quant + 6 sector_qual = 13
        assert len(TOOL_EQUIPPED_AGENTS) == 13

    def test_canonical_set_is_20(self):
        # 1 macro + 1 ic_cio + 6 sectors × 3 sub-stages = 20
        assert len(CANONICAL_AGENTS) == 20


# ── compute_provenance_grounding (top-level) ────────────────────────────────


def _mock_s3_with_artifacts(artifacts_by_key: dict[str, dict]):
    """Build a MagicMock S3 client that lists + returns the given
    artifact bodies."""
    s3 = MagicMock()

    def get_paginator(op):
        paginator = MagicMock()
        def paginate(Bucket, Prefix):
            matching = [
                {"Key": k} for k in artifacts_by_key
                if k.startswith(Prefix)
            ]
            yield {"Contents": matching}
        paginator.paginate.side_effect = paginate
        return paginator

    s3.get_paginator.side_effect = get_paginator

    def get_object(Bucket, Key):
        body = json.dumps(artifacts_by_key[Key]).encode()
        return {"Body": MagicMock(read=lambda: body)}

    s3.get_object.side_effect = get_object
    return s3


class TestComputeProvenanceGrounding:
    def test_no_recent_sf_run(self):
        s3 = _mock_s3_with_artifacts({})
        result = compute_provenance_grounding(
            bucket="test-bucket", run_date="2026-05-09", s3_client=s3,
        )
        assert result["status"] == "no_recent_sf_run"

    def test_invalid_run_date(self):
        result = compute_provenance_grounding(
            bucket="test-bucket", run_date="not-a-date", s3_client=MagicMock(),
        )
        assert result["status"] == "error"

    def test_basic_compute_one_saturday(self):
        # Sat 2026-05-09 with 1 ic_cio + 1 sector_quant:technology artifact
        # (the per-stage write site)
        artifacts = {
            "decision_artifacts/2026/05/09/ic_cio/run1.json": {
                "agent_output": {"team_id": "cio"},
                "input_data_snapshot": {"candidates": []},
            },
            "decision_artifacts/2026/05/09/sector_quant:technology/run1.json": {
                "agent_output": {
                    "team_id": "technology",
                    "tool_calls": [
                        {"tool": "screen_by_volume"},
                        {"tool": "fetch_news"},
                    ],
                },
                "input_data_snapshot": {"team_id": "technology", "market_regime": "bull"},
            },
        }
        s3 = _mock_s3_with_artifacts(artifacts)

        result = compute_provenance_grounding(
            bucket="test-bucket", run_date="2026-05-10",
            lookback_weeks=1, s3_client=s3,
        )

        assert result["status"] == "ok"
        assert result["most_recent_sf_date"] == "2026-05-09"
        assert result["n_total_artifacts_read"] == 2
        # ic_cio = synthesizer, no tools, NOT in tool_equipped → not alarmed
        assert "ic_cio" not in result["tool_equipped_alarms"]
        # sector_quant:technology made 2 tool calls → not alarmed either
        assert "sector_quant:technology" not in result["tool_equipped_alarms"]

        tech = result["per_agent"]["sector_quant:technology"]
        assert tech["n_artifacts"] == 1
        assert tech["mean_n_tool_calls"] == 2.0
        assert tech["mean_n_distinct_tools"] == 2.0
        assert tech["pct_zero_call_outputs"] == 0.0

    def test_zero_call_alarm_fires_on_sector_quant(self):
        """A sector_quant:{sector} artifact with zero tool calls is a
        hallucination signal — it must trigger the tool-equipped alarm."""
        artifacts = {
            "decision_artifacts/2026/05/09/sector_quant:financials/run.json": {
                "agent_output": {"team_id": "financials"},  # no tool_calls
                "input_data_snapshot": {"team_id": "financials"},
            },
        }
        s3 = _mock_s3_with_artifacts(artifacts)
        result = compute_provenance_grounding(
            bucket="test-bucket", run_date="2026-05-10",
            lookback_weeks=1, s3_client=s3,
        )
        assert result["status"] == "ok"
        assert "sector_quant:financials" in result["tool_equipped_alarms"]

    def test_peer_review_zero_calls_does_not_alarm(self):
        """sector_peer_review:* is a synthesizer — zero tool calls is
        the expected steady state, NOT an alarm."""
        artifacts = {
            "decision_artifacts/2026/05/09/sector_peer_review:technology/run.json": {
                "agent_output": {"team_id": "technology"},  # no tool_calls
                "input_data_snapshot": {"team_id": "technology"},
            },
        }
        s3 = _mock_s3_with_artifacts(artifacts)
        result = compute_provenance_grounding(
            bucket="test-bucket", run_date="2026-05-10",
            lookback_weeks=1, s3_client=s3,
        )
        assert result["status"] == "ok"
        assert "sector_peer_review:technology" not in result["tool_equipped_alarms"]

    def test_tool_equipped_alarm_fires_on_zero_calls(self):
        # macro_economist with zero tool calls → alarm
        artifacts = {
            "decision_artifacts/2026/05/09/macro_economist/run1.json": {
                "agent_output": {"market_regime": "bull"},  # no tool_calls
                "input_data_snapshot": {"macro_data": {}},
            },
        }
        s3 = _mock_s3_with_artifacts(artifacts)
        result = compute_provenance_grounding(
            bucket="test-bucket", run_date="2026-05-10",
            lookback_weeks=1, s3_client=s3,
        )
        assert result["status"] == "ok"
        assert "macro_economist" in result["tool_equipped_alarms"]
        macro = result["per_agent"]["macro_economist"]
        assert macro["pct_zero_call_outputs"] == 100.0

    def test_meta_prefixes_excluded(self):
        # _eval/, _replay/, etc. should not be counted as agent artifacts
        artifacts = {
            "decision_artifacts/2026/05/09/sector_quant:technology/run1.json": {
                "agent_output": {"tool_calls": [{"tool": "x"}]},
                "input_data_snapshot": {},
            },
            "decision_artifacts/2026/05/09/_eval/sector_quant:technology/run1.json": {
                "agent_output": {},
                "input_data_snapshot": {},
            },
            "decision_artifacts/2026/05/09/_replay/sector_quant:technology/run1.json": {
                "agent_output": {},
                "input_data_snapshot": {},
            },
        }
        s3 = _mock_s3_with_artifacts(artifacts)
        result = compute_provenance_grounding(
            bucket="test-bucket", run_date="2026-05-10",
            lookback_weeks=1, s3_client=s3,
        )
        assert result["n_total_artifacts_read"] == 1
        assert list(result["per_agent"].keys()) == ["sector_quant:technology"]

    def test_thesis_update_excluded_from_canonical(self):
        # thesis_update:* is variable-cardinality, excluded from per_agent
        artifacts = {
            "decision_artifacts/2026/05/09/thesis_update:AAPL/run1.json": {
                "agent_output": {"tool_calls": [{"tool": "rag"}]},
                "input_data_snapshot": {},
            },
            "decision_artifacts/2026/05/09/ic_cio/run1.json": {
                "agent_output": {},
                "input_data_snapshot": {},
            },
        }
        s3 = _mock_s3_with_artifacts(artifacts)
        result = compute_provenance_grounding(
            bucket="test-bucket", run_date="2026-05-10",
            lookback_weeks=1, s3_client=s3,
        )
        assert "thesis_update:AAPL" not in result["per_agent"]
        assert "ic_cio" in result["per_agent"]

    def test_cw_metrics_emitted_for_most_recent_saturday(self):
        artifacts = {
            "decision_artifacts/2026/05/09/sector_quant:technology/run.json": {
                "agent_output": {
                    "tool_calls": [
                        {"tool": "screen_by_volume"},
                        {"tool": "fetch_news"},
                    ],
                },
                "input_data_snapshot": {"team_id": "technology"},
            },
        }
        s3 = _mock_s3_with_artifacts(artifacts)
        cw = MagicMock()

        result = compute_provenance_grounding(
            bucket="test-bucket", run_date="2026-05-10",
            lookback_weeks=1, s3_client=s3, cloudwatch_client=cw,
        )

        assert result["status"] == "ok"
        assert cw.put_metric_data.called

        # Verify the dim + metric names emitted
        all_metrics: list[dict] = []
        for call in cw.put_metric_data.call_args_list:
            kwargs = call.kwargs
            assert kwargs["Namespace"] == "AlphaEngine/Provenance"
            all_metrics.extend(kwargs["MetricData"])

        metric_names = {m["MetricName"] for m in all_metrics}
        assert metric_names == {
            "pct_zero_call_outputs",
            "mean_n_tool_calls",
            "mean_n_distinct_tools",
            "mean_input_consumption_ratio",
            "n_artifacts",
        }

        # Every datapoint dim'd by judged_agent_id = sector_quant:technology
        assert all(
            m["Dimensions"][0]["Name"] == "judged_agent_id"
            and m["Dimensions"][0]["Value"] == "sector_quant:technology"
            for m in all_metrics
        )

    def test_emit_metrics_false_skips_cw_call(self):
        artifacts = {
            "decision_artifacts/2026/05/09/macro_economist/run.json": {
                "agent_output": {"tool_calls": [{"tool": "fetch_macro"}]},
                "input_data_snapshot": {},
            },
        }
        s3 = _mock_s3_with_artifacts(artifacts)
        cw = MagicMock()

        compute_provenance_grounding(
            bucket="test-bucket", run_date="2026-05-10",
            lookback_weeks=1, s3_client=s3, cloudwatch_client=cw,
            emit_metrics=False,
        )

        cw.put_metric_data.assert_not_called()

    def test_default_cw_client_creation_failure_does_not_break_compute(self, monkeypatch):
        """When cloudwatch_client=None and boto3 can't construct a client
        (e.g. CI without AWS_DEFAULT_REGION), the failure must be caught
        so the JSON path still completes."""
        from botocore.exceptions import NoRegionError

        artifacts = {
            "decision_artifacts/2026/05/09/macro_economist/run.json": {
                "agent_output": {"tool_calls": [{"tool": "fetch_macro"}]},
                "input_data_snapshot": {},
            },
        }
        s3 = _mock_s3_with_artifacts(artifacts)

        import analysis.provenance_grounding as mod
        def _raise_no_region(service):
            raise NoRegionError()
        monkeypatch.setattr(mod.boto3, "client", _raise_no_region)

        result = compute_provenance_grounding(
            bucket="test-bucket", run_date="2026-05-10",
            lookback_weeks=1, s3_client=s3,
            # cloudwatch_client=None → forces boto3.client path
        )
        assert result["status"] == "ok"
        assert "macro_economist" in result["per_agent"]

    def test_cw_emission_failure_does_not_break_compute(self):
        artifacts = {
            "decision_artifacts/2026/05/09/macro_economist/run.json": {
                "agent_output": {"tool_calls": [{"tool": "fetch_macro"}]},
                "input_data_snapshot": {},
            },
        }
        s3 = _mock_s3_with_artifacts(artifacts)
        cw = MagicMock()
        cw.put_metric_data.side_effect = RuntimeError("CW unreachable")

        result = compute_provenance_grounding(
            bucket="test-bucket", run_date="2026-05-10",
            lookback_weeks=1, s3_client=s3, cloudwatch_client=cw,
        )

        # JSON artifact compute still succeeds despite CW failure
        assert result["status"] == "ok"
        assert "macro_economist" in result["per_agent"]

    def test_rolling_aggregate_per_agent(self):
        # Multiple Saturdays with sector_quant:tech captures
        artifacts = {}
        for date_str in ("2026/04/25", "2026/05/02", "2026/05/09"):
            artifacts[
                f"decision_artifacts/{date_str}/sector_quant:technology/run.json"
            ] = {
                "agent_output": {
                    "tool_calls": [{"tool": "x"}, {"tool": "y"}],
                },
                "input_data_snapshot": {"team_id": "technology"},
            }
        s3 = _mock_s3_with_artifacts(artifacts)
        result = compute_provenance_grounding(
            bucket="test-bucket", run_date="2026-05-10",
            lookback_weeks=4, s3_client=s3,
        )
        rolling_tech = result["rolling"]["per_agent"]["sector_quant:technology"]
        assert rolling_tech["n_saturdays"] == 3
        assert rolling_tech["n_artifacts_total"] == 3
        assert rolling_tech["n_distinct_tools"] == 2
