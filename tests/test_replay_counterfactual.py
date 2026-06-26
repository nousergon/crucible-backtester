"""Unit tests for replay/counterfactual.py.

Coverage strategy: per-agent feature extraction (ic_cio + macro_economist
v1, others UNSUPPORTED), tree-fit numerics (perfect fit on separable
data, low fit on random data, single-class skip, thin-sample skip),
end-to-end pipeline with stubbed S3 + CloudWatch (happy path,
unsupported-agent bucket, multi-agent grouping, persistence shape).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest


# ── Per-agent extraction ─────────────────────────────────────────────────


class TestExtractIcCio:
    def test_per_candidate_rows(self):
        from replay.counterfactual import extract_features_and_decision

        snapshot = {
            "candidates": [
                {"ticker": "NVDA", "composite_score": 85, "conviction": 75,
                 "sector_modifier": 1.15},
                {"ticker": "WMT", "composite_score": 45, "conviction": 30,
                 "sector_modifier": 1.0},
            ],
        }
        output = {
            "ic_decisions": [
                {"ticker": "NVDA", "decision": "ADVANCE"},
                {"ticker": "WMT", "decision": "REJECT"},
            ],
        }
        rows = extract_features_and_decision("ic_cio", snapshot, output)
        assert isinstance(rows, list)
        assert len(rows) == 2
        # First row: NVDA ADVANCE with composite=85
        f, d = rows[0]
        assert d == "ADVANCE"
        assert f["composite_score"] == 85.0
        assert f["sector_modifier"] == 1.15

    def test_skips_no_advance_deadlock(self):
        from replay.counterfactual import extract_features_and_decision

        snapshot = {"candidates": [{"ticker": "NVDA", "composite_score": 50}]}
        output = {
            "ic_decisions": [
                {"ticker": "NVDA", "decision": "NO_ADVANCE_DEADLOCK"},
            ],
        }
        rows = extract_features_and_decision("ic_cio", snapshot, output)
        # Deadlock decisions are filtered — they're a separate concern
        # from the binary ADVANCE/REJECT call.
        assert rows == []

    def test_missing_candidate_features_default_to_zero(self):
        from replay.counterfactual import extract_features_and_decision

        # Decision present but candidate missing from snapshot — defaults
        # all features to 0.0 + sector_modifier defaults to 1.0.
        output = {
            "ic_decisions": [{"ticker": "NVDA", "decision": "ADVANCE"}],
        }
        rows = extract_features_and_decision("ic_cio", {}, output)
        assert len(rows) == 1
        f, d = rows[0]
        assert d == "ADVANCE"
        assert f["composite_score"] == 0.0
        assert f["sector_modifier"] == 1.0


class TestExtractMacroEconomist:
    def test_per_run_row_with_indicators(self):
        from replay.counterfactual import extract_features_and_decision

        snapshot = {
            "macro_indicators": {
                "spy_20d_return": 0.04,
                "vix_level": 18.5,
                "yield_curve_slope": 0.6,
                "market_breadth": 0.62,
            },
        }
        output = {"market_regime": "bull"}
        rows = extract_features_and_decision("macro_economist", snapshot, output)
        assert len(rows) == 1
        f, d = rows[0]
        assert d == "bull"
        assert f["spy_20d_return"] == 0.04
        assert f["vix_level"] == 18.5

    def test_invalid_regime_skipped(self):
        from replay.counterfactual import extract_features_and_decision

        rows = extract_features_and_decision(
            "macro_economist",
            {"macro_indicators": {}},
            {"market_regime": "exuberant"},  # not in literal set
        )
        assert rows == []


class TestUnsupportedAgents:
    @pytest.mark.parametrize("agent_id", [
        "sector_quant:tech",
        "sector_qual:healthcare",
        "sector_peer_review:financials",
        "thesis_update:AAPL",
    ])
    def test_v1_unsupported_agents_marked(self, agent_id):
        from replay.counterfactual import (
            UNSUPPORTED_AGENT, extract_features_and_decision,
        )

        result = extract_features_and_decision(agent_id, {}, {})
        assert result == UNSUPPORTED_AGENT

    def test_unknown_agent_unsupported(self):
        from replay.counterfactual import (
            UNSUPPORTED_AGENT, extract_features_and_decision,
        )

        result = extract_features_and_decision("brand_new_agent", {}, {})
        assert result == UNSUPPORTED_AGENT

    def test_empty_agent_output_returns_empty_list_for_supported(self):
        # Supported agent with empty output → empty rows (not unsupported
        # marker — the agent IS supported, just no rows extractable).
        from replay.counterfactual import extract_features_and_decision

        assert extract_features_and_decision("ic_cio", {}, {}) == []
        assert extract_features_and_decision("macro_economist", {}, {}) == []


# ── Tree fit ──────────────────────────────────────────────────────────────


class TestFitCounterfactualTree:
    def test_perfectly_separable_advance_reject(self):
        """Build 20 rows where decision strictly follows composite > 60.
        A 3-deep tree should achieve 1.0 match rate."""
        from replay.counterfactual import fit_counterfactual_tree

        rows = []
        for i in range(20):
            score = float(i * 5)  # 0, 5, 10, ..., 95
            decision = "ADVANCE" if score > 60 else "REJECT"
            rows.append(({"composite_score": score}, decision))

        fit = fit_counterfactual_tree(rows)
        assert fit["match_rate"] == pytest.approx(1.0)
        assert fit["n_classes"] == 2
        assert "composite_score" in fit["feature_names"]
        # Feature importance should concentrate on composite_score.
        assert fit["feature_importances"]["composite_score"] > 0.9

    def test_thin_sample_skipped(self):
        from replay.counterfactual import fit_counterfactual_tree

        rows = [({"x": float(i)}, "A" if i < 3 else "B") for i in range(5)]
        fit = fit_counterfactual_tree(rows)
        assert fit["match_rate"] is None
        assert "thin_sample" in fit.get("skip_reason", "")

    def test_single_class_skipped(self):
        from replay.counterfactual import fit_counterfactual_tree

        rows = [({"x": float(i)}, "ADVANCE") for i in range(20)]
        fit = fit_counterfactual_tree(rows)
        assert fit["match_rate"] is None
        assert "single_class" in fit.get("skip_reason", "")

    def test_random_decisions_lower_match_rate(self):
        """Random decisions on 20 rows shouldn't fit perfectly with a
        3-deep tree. The bound here is loose (we just want < 1.0) since
        even random labels can be partially fit."""
        import random
        from replay.counterfactual import fit_counterfactual_tree

        random.seed(42)
        rows = [
            ({"a": float(i), "b": float(20 - i)}, random.choice(["A", "B"]))
            for i in range(20)
        ]
        fit = fit_counterfactual_tree(rows)
        # 20 random labels in 2 classes — depth-3 tree (max 8 leaves)
        # can fit some but not all. Should be in [0.5, 0.95] range.
        assert 0.4 < fit["match_rate"] < 1.0


# ── End-to-end pipeline ──────────────────────────────────────────────────


def _make_ic_cio_artifact(advance_count: int, reject_count: int) -> dict:
    """Build an artifact where ic_decisions has N ADVANCE + M REJECT
    rows separable by composite_score > 60."""
    candidates = []
    decisions = []
    for i in range(advance_count):
        candidates.append({
            "ticker": f"A{i}",
            "composite_score": 75 + i,  # all > 60
            "sector_modifier": 1.0,
            "conviction": 70,
        })
        decisions.append({"ticker": f"A{i}", "decision": "ADVANCE"})
    for i in range(reject_count):
        candidates.append({
            "ticker": f"R{i}",
            "composite_score": 30 + i,  # all < 60
            "sector_modifier": 1.0,
            "conviction": 30,
        })
        decisions.append({"ticker": f"R{i}", "decision": "REJECT"})

    return {
        "schema_version": 1,
        "run_id": "r1",
        "timestamp": "2026-05-03T12:00:00Z",
        "agent_id": "ic_cio",
        "model_metadata": {"model_name": "claude-sonnet-4-6"},
        "full_prompt_context": {
            "system_prompt": "s", "user_prompt": "u", "tool_definitions": [],
        },
        "input_data_snapshot": {"candidates": candidates},
        "agent_output": {"ic_decisions": decisions},
    }


def _build_s3_stub(artifacts_by_key: dict[str, dict]) -> MagicMock:
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


class TestComputeAndEmit:
    def test_perfect_fit_emits_match_rate(self):
        """Single artifact carrying 20 separable ic_cio decisions →
        match_rate 1.0 emitted, analysis persisted."""
        from replay.counterfactual import compute_and_emit

        end = datetime(2026, 5, 9, tzinfo=timezone.utc)
        artifact = _make_ic_cio_artifact(advance_count=10, reject_count=10)
        artifacts = {
            "decision_artifacts/2026/05/09/ic_cio/r1.json": artifact,
        }
        s3 = _build_s3_stub(artifacts)
        cw = MagicMock()

        summary = compute_and_emit(
            end_time=end, window_days=1,
            s3_client=s3, cloudwatch_client=cw,
        )

        assert summary["agents_analyzed"] == 1
        per_agent = summary["per_agent"][0]
        assert per_agent["agent_id_base"] == "ic_cio"
        assert per_agent["n_samples"] == 20
        assert per_agent["match_rate"] == pytest.approx(1.0)

        # CloudWatch emission: 2 datapoints (match_rate + n_samples).
        cw.put_metric_data.assert_called_once()
        names = [
            d["MetricName"]
            for d in cw.put_metric_data.call_args.kwargs["MetricData"]
        ]
        assert "agent_counterfactual_rule_fit" in names
        assert "agent_counterfactual_rule_fit_n_samples" in names

        # Per-agent analysis persisted under the canonical eval_artifacts
        # layout: keep the {agent_id_base}/ partition, swap the weekly
        # {YYYY-Www} file to {run_id}.json (YYMMDDHHMM), add a per-agent
        # latest.json sidecar. end_time 2026-05-09 00:00 UTC → "2605090000".
        put_keys = [c.kwargs["Key"] for c in s3.put_object.call_args_list]
        dated = sorted(k for k in put_keys if not k.endswith("/latest.json"))
        latest = sorted(k for k in put_keys if k.endswith("/latest.json"))
        assert dated == [
            "decision_artifacts/_counterfactual/ic_cio/2605090000.json"
        ]
        assert latest == [
            "decision_artifacts/_counterfactual/ic_cio/latest.json"
        ]
        # No legacy ISO-week filename in the new layout.
        assert all("2026-W" not in k for k in put_keys)

    def test_unsupported_agent_excluded_from_per_agent(self):
        from replay.counterfactual import compute_and_emit

        end = datetime(2026, 5, 9, tzinfo=timezone.utc)
        # One supported agent (ic_cio with 20 rows) + one unsupported.
        artifacts = {
            "decision_artifacts/2026/05/09/ic_cio/r1.json":
                _make_ic_cio_artifact(advance_count=10, reject_count=10),
            "decision_artifacts/2026/05/09/sector_quant:tech/r2.json": {
                "schema_version": 1,
                "run_id": "r2", "timestamp": "2026-05-09T00:00:00Z",
                "agent_id": "sector_quant:tech",
                "model_metadata": {"model_name": "claude-haiku-4-5"},
                "full_prompt_context": {
                    "system_prompt": "s", "user_prompt": "u",
                    "tool_definitions": [],
                },
                "input_data_snapshot": {},
                "agent_output": {"ranked_picks": []},
            },
        }
        s3 = _build_s3_stub(artifacts)
        cw = MagicMock()

        summary = compute_and_emit(
            end_time=end, window_days=1,
            s3_client=s3, cloudwatch_client=cw,
        )

        # ic_cio analyzed; sector_quant tracked under unsupported.
        assert summary["agents_analyzed"] == 1
        assert summary["agents_unsupported"] == ["sector_quant"]

    def test_thin_sample_skipped_at_pipeline(self):
        from replay.counterfactual import compute_and_emit

        end = datetime(2026, 5, 9, tzinfo=timezone.utc)
        # 1 ic_cio artifact with 5 rows — below MIN_SAMPLES_FOR_FIT.
        artifacts = {
            "decision_artifacts/2026/05/09/ic_cio/r1.json":
                _make_ic_cio_artifact(advance_count=3, reject_count=2),
        }
        s3 = _build_s3_stub(artifacts)
        cw = MagicMock()

        summary = compute_and_emit(
            end_time=end, window_days=1,
            s3_client=s3, cloudwatch_client=cw,
        )
        assert summary["agents_analyzed"] == 0
        assert summary["agents_skipped_thin_sample"][0]["agent_id_base"] == "ic_cio"
        # No metric emission for thin-sample skip.
        cw.put_metric_data.assert_not_called()


# ── ROADMAP L293 (2026-05-19) — per-agent artifact cap regression suite ──


class TestPerAgentArtifactCap:
    """Bounds the artifact-scan ceiling so the Saturday-SF Counterfactual
    Lambda stays under its 600s timeout regardless of how heavy any single
    agent's decision-artifact backlog grows. The cap applies at the
    list-keys stage (pre ``_load_artifact``) so it directly reduces the
    expensive S3 get_object loop."""

    def test_cap_drops_oldest_when_one_agent_dominates(self):
        """Single agent with more artifacts than the cap → most-recent-first
        truncation keeps only ``max_artifacts_per_agent`` keys for that
        agent. The list iterates day-by-day backward from end_date so the
        first ``cap`` keys are the freshest."""
        from replay.counterfactual import _list_artifact_keys_in_window

        end = datetime(2026, 5, 9, tzinfo=timezone.utc)
        # Build artifacts spread across 5 days for a single agent, 3/day
        artifacts: dict[str, dict] = {}
        for day_offset in range(5):
            day = end - timedelta(days=day_offset)
            day_key = day.strftime("%Y/%m/%d")
            for i in range(3):
                k = f"decision_artifacts/{day_key}/ic_cio/r{day_offset}_{i}.json"
                artifacts[k] = {"agent_id": "ic_cio"}
        s3 = _build_s3_stub(artifacts)

        # Cap at 7 — should keep day 0 (3 keys) + day 1 (3 keys) + 1 from day 2.
        keys = _list_artifact_keys_in_window(
            s3,
            bucket="b",
            capture_prefix="decision_artifacts",
            end_date=end,
            window_days=5,
            max_artifacts_per_agent=7,
        )
        assert len(keys) == 7
        # All retained keys are from the 3 most-recent days (5/9, 5/8, 5/7).
        retained_days = {k.split("/")[3] for k in keys}
        assert retained_days <= {"09", "08", "07"}

    def test_cap_none_returns_all_keys(self):
        """``max_artifacts_per_agent=None`` disables the cap — full corpus."""
        from replay.counterfactual import _list_artifact_keys_in_window

        end = datetime(2026, 5, 9, tzinfo=timezone.utc)
        artifacts = {
            f"decision_artifacts/2026/05/09/ic_cio/r{i}.json": {"agent_id": "ic_cio"}
            for i in range(15)
        }
        s3 = _build_s3_stub(artifacts)

        keys = _list_artifact_keys_in_window(
            s3,
            bucket="b",
            capture_prefix="decision_artifacts",
            end_date=end,
            window_days=1,
            max_artifacts_per_agent=None,
        )
        assert len(keys) == 15

    def test_cap_zero_returns_all_keys(self):
        """``max_artifacts_per_agent=0`` is treated as unbounded (defensive
        — operator may set 0 expecting "no cap" or "no work"; the
        less-surprising behavior is to disable the cap rather than emit
        zero data)."""
        from replay.counterfactual import _list_artifact_keys_in_window

        end = datetime(2026, 5, 9, tzinfo=timezone.utc)
        artifacts = {
            f"decision_artifacts/2026/05/09/ic_cio/r{i}.json": {"agent_id": "ic_cio"}
            for i in range(10)
        }
        s3 = _build_s3_stub(artifacts)

        keys = _list_artifact_keys_in_window(
            s3,
            bucket="b",
            capture_prefix="decision_artifacts",
            end_date=end,
            window_days=1,
            max_artifacts_per_agent=0,
        )
        assert len(keys) == 10

    def test_cap_applied_per_agent_id_base(self):
        """Two agents each exceeding the cap → each independently truncated
        to ``max_artifacts_per_agent``. Cap is a per-agent ceiling, not a
        global one."""
        from replay.counterfactual import _list_artifact_keys_in_window

        end = datetime(2026, 5, 9, tzinfo=timezone.utc)
        artifacts: dict[str, dict] = {}
        for i in range(10):
            artifacts[f"decision_artifacts/2026/05/09/ic_cio/r{i}.json"] = {}
            artifacts[f"decision_artifacts/2026/05/09/macro_economist/r{i}.json"] = {}
        s3 = _build_s3_stub(artifacts)

        keys = _list_artifact_keys_in_window(
            s3,
            bucket="b",
            capture_prefix="decision_artifacts",
            end_date=end,
            window_days=1,
            max_artifacts_per_agent=3,
        )
        # 3 per agent × 2 agents = 6 total.
        assert len(keys) == 6
        ic_cio_keys = [k for k in keys if "/ic_cio/" in k]
        macro_keys = [k for k in keys if "/macro_economist/" in k]
        assert len(ic_cio_keys) == 3
        assert len(macro_keys) == 3
