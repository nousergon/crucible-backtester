"""
tests/test_agent_justification.py — unit tests for the agent-justification
S3 summarizer used by the evaluator email post-2026-05-07 SF reorder.

Mocks boto3 list/get calls so tests don't hit S3. Focuses on the four
shape contracts the email renderer depends on: judge / clustering /
concordance / counterfactual. Each summarizer must return a status dict
with ``status`` ∈ {"ok", "no_data", "no_recent_sf_run"} and the
top-level fields the renderer reads.
"""

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis import agent_justification


def _mock_s3_client(list_dirs: dict, list_objects: dict, get_objects: dict) -> MagicMock:
    """Build a moto-free stub that returns scripted responses by prefix.

    list_dirs: dict[prefix] -> list of subdir basenames (CommonPrefixes)
    list_objects: dict[prefix] -> list of object keys (Contents)
    get_objects: dict[key] -> dict body (will be JSON-serialized)
    """
    s3 = MagicMock()

    def list_v2(Bucket, Prefix, Delimiter=None):
        if Delimiter == "/":
            cp = list_dirs.get(Prefix, [])
            return {
                "CommonPrefixes": [{"Prefix": f"{Prefix}{name}/"} for name in cp],
            }
        objs = list_objects.get(Prefix, [])
        return {"Contents": [{"Key": k} for k in objs]}

    def get(Bucket, Key):
        body = get_objects.get(Key)
        if body is None:
            from botocore.exceptions import ClientError
            raise ClientError(
                {"Error": {"Code": "NoSuchKey"}}, "GetObject",
            )
        return {"Body": io.BytesIO(json.dumps(body).encode("utf-8"))}

    s3.list_objects_v2.side_effect = list_v2
    s3.get_object.side_effect = get
    return s3


# ── Judge summarizer ──────────────────────────────────────────────────────


class TestSummarizeJudge:
    def test_no_recent_sf_run_returns_status(self):
        s3 = _mock_s3_client({}, {}, {})
        out = agent_justification.summarize_judge(
            "test-bucket", "2026-05-07", s3_client=s3,
        )
        assert out["status"] == "no_recent_sf_run"

    def test_aggregates_overall_score_across_agents(self):
        s3 = _mock_s3_client(
            list_dirs={
                "decision_artifacts/_eval/": ["2026-05-02"],
                "decision_artifacts/_eval/2026-05-02/": ["agent_a", "agent_b"],
            },
            list_objects={
                "decision_artifacts/_eval/2026-05-02/agent_a/": [
                    "decision_artifacts/_eval/2026-05-02/agent_a/2026-05-02.haiku.json",
                ],
                "decision_artifacts/_eval/2026-05-02/agent_b/": [
                    "decision_artifacts/_eval/2026-05-02/agent_b/2026-05-02.haiku.json",
                ],
            },
            get_objects={
                "decision_artifacts/_eval/2026-05-02/agent_a/2026-05-02.haiku.json": {
                    "overall_score": 4.2,
                },
                "decision_artifacts/_eval/2026-05-02/agent_b/2026-05-02.haiku.json": {
                    "overall_score": 3.8,
                },
            },
        )
        out = agent_justification.summarize_judge(
            "test-bucket", "2026-05-07", s3_client=s3,
        )
        assert out["status"] == "ok"
        assert out["n_agents"] == 2
        assert out["n_scored"] == 2
        assert out["mean_score"] == 4.0
        assert out["min_score"] == 3.8
        assert out["max_score"] == 4.2


# ── Counterfactual + Clustering ───────────────────────────────────────────


class TestSummarizeCounterfactual:
    def test_aggregates_match_rate(self):
        s3 = _mock_s3_client(
            list_dirs={
                "decision_artifacts/_counterfactual/": ["ic_cio", "macro_economist"],
            },
            list_objects={
                "decision_artifacts/_counterfactual/ic_cio/": [
                    "decision_artifacts/_counterfactual/ic_cio/2026-W19.json",
                ],
                "decision_artifacts/_counterfactual/macro_economist/": [
                    "decision_artifacts/_counterfactual/macro_economist/2026-W19.json",
                ],
            },
            get_objects={
                "decision_artifacts/_counterfactual/ic_cio/2026-W19.json": {
                    "match_rate": 0.95,
                },
                "decision_artifacts/_counterfactual/macro_economist/2026-W19.json": {
                    "match_rate": 0.85,
                },
            },
        )
        out = agent_justification.summarize_counterfactual(
            "test-bucket", "2026-05-07", s3_client=s3,
        )
        assert out["status"] == "ok"
        assert out["n_agents"] == 2
        assert out["mean_match_rate"] == 0.9
        assert out["agents"] == ["ic_cio", "macro_economist"]

    def test_no_data_when_prefix_empty(self):
        s3 = _mock_s3_client({}, {}, {})
        out = agent_justification.summarize_counterfactual(
            "test-bucket", "2026-05-07", s3_client=s3,
        )
        assert out["status"] == "no_data"


class TestSummarizeClustering:
    def test_aggregates_top3_concentration(self):
        s3 = _mock_s3_client(
            list_dirs={
                "decision_artifacts/_analysis/": ["ic_cio"],
            },
            list_objects={
                "decision_artifacts/_analysis/ic_cio/": [
                    "decision_artifacts/_analysis/ic_cio/2026-W19.json",
                ],
            },
            get_objects={
                "decision_artifacts/_analysis/ic_cio/2026-W19.json": {
                    "top3_concentration": 0.42,
                },
            },
        )
        out = agent_justification.summarize_clustering(
            "test-bucket", "2026-05-07", s3_client=s3,
        )
        assert out["status"] == "ok"
        assert out["n_agents"] == 1
        assert out["mean_top3_concentration"] == 0.42


class TestSummarizeConcordance:
    def test_no_recent_when_prefix_empty(self):
        # Concordance Lambda may not have written summaries yet —
        # summarizer returns no_recent_sf_run rather than raising so
        # the email renderer can show a gap message.
        s3 = _mock_s3_client({}, {}, {})
        out = agent_justification.summarize_concordance(
            "test-bucket", "2026-05-07", s3_client=s3,
        )
        assert out["status"] == "no_recent_sf_run"


# ── Composite ─────────────────────────────────────────────────────────────


class TestSummarizeAll:
    def test_returns_all_four_keys(self):
        s3 = _mock_s3_client({}, {}, {})
        out = agent_justification.summarize_all(
            "test-bucket", "2026-05-07", s3_client=s3,
        )
        assert set(out.keys()) == {"judge", "clustering", "concordance", "counterfactual"}
        # All four return status dicts even when nothing exists at S3.
        for v in out.values():
            assert "status" in v
