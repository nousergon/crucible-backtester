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
            # Real S3 with a delimiter returns BOTH the CommonPrefixes
            # (sub-"directories") AND any objects sitting directly under
            # the prefix with no further "/" (Contents). Mirror that so
            # the flat canonical-layout reader path is exercised.
            cp = list_dirs.get(Prefix, [])
            flat = [
                k for k in list_objects.get(Prefix, [])
                if "/" not in k[len(Prefix):]
            ]
            return {
                "CommonPrefixes": [{"Prefix": f"{Prefix}{name}/"} for name in cp],
                "Contents": [{"Key": k} for k in flat],
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

    def test_aggregates_dimension_scores_across_agents(self):
        """Each rubric file carries `dimension_scores` (list of per-dim
        1-5 entries). Per-agent overall is the mean across dimensions;
        global overall is the mean across agents.

        Regression target: 2026-05-07 v2 quadfecta-email run rendered
        "Judge — no rubric data within 14d of run_date" while six
        agents had captured rubrics at decision_artifacts/_eval/
        2026-05-07/. Looked like a judge-side producer gap — was
        actually a consumer-side schema mismatch (the loader read
        top-level `overall_score`/`score`, neither of which exists in
        the actual rubric schema).
        """
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
                # agent_a: six dims, mean = 3.5
                "decision_artifacts/_eval/2026-05-02/agent_a/2026-05-02.haiku.json": {
                    "schema_version": 1,
                    "judged_agent_id": "agent_a",
                    "judge_skip_reason": None,
                    "dimension_scores": [
                        {"dimension": "d1", "score": 4},
                        {"dimension": "d2", "score": 3},
                        {"dimension": "d3", "score": 4},
                        {"dimension": "d4", "score": 3},
                        {"dimension": "d5", "score": 3},
                        {"dimension": "d6", "score": 4},
                    ],
                },
                # agent_b: six dims, mean = 3.0
                "decision_artifacts/_eval/2026-05-02/agent_b/2026-05-02.haiku.json": {
                    "schema_version": 1,
                    "judged_agent_id": "agent_b",
                    "judge_skip_reason": None,
                    "dimension_scores": [
                        {"dimension": "d1", "score": 3},
                        {"dimension": "d2", "score": 3},
                        {"dimension": "d3", "score": 3},
                        {"dimension": "d4", "score": 3},
                        {"dimension": "d5", "score": 3},
                        {"dimension": "d6", "score": 3},
                    ],
                },
            },
        )
        out = agent_justification.summarize_judge(
            "test-bucket", "2026-05-07", s3_client=s3,
        )
        assert out["status"] == "ok"
        assert out["n_agents"] == 2
        assert out["n_scored"] == 2
        assert out["mean_score"] == 3.25  # mean of agent-overalls [3.5, 3.0]
        assert out["min_score"] == 3.0
        assert out["max_score"] == 3.5

    def test_skips_rubrics_with_judge_skip_reason(self):
        """Rubrics where the judge bailed early (e.g. tool-equipped
        alarm fired) carry `judge_skip_reason` and typically no
        dimension_scores. These count toward n_agents (presence) but
        not n_scored (data) so the email surfaces both rates."""
        s3 = _mock_s3_client(
            list_dirs={
                "decision_artifacts/_eval/": ["2026-05-02"],
                "decision_artifacts/_eval/2026-05-02/": ["agent_a", "agent_skipped"],
            },
            list_objects={
                "decision_artifacts/_eval/2026-05-02/agent_a/": [
                    "decision_artifacts/_eval/2026-05-02/agent_a/2026-05-02.haiku.json",
                ],
                "decision_artifacts/_eval/2026-05-02/agent_skipped/": [
                    "decision_artifacts/_eval/2026-05-02/agent_skipped/2026-05-02.haiku.json",
                ],
            },
            get_objects={
                "decision_artifacts/_eval/2026-05-02/agent_a/2026-05-02.haiku.json": {
                    "judge_skip_reason": None,
                    "dimension_scores": [{"dimension": "d1", "score": 4}],
                },
                "decision_artifacts/_eval/2026-05-02/agent_skipped/2026-05-02.haiku.json": {
                    "judge_skip_reason": "tool_equipped_alarm",
                    "dimension_scores": [],
                },
            },
        )
        out = agent_justification.summarize_judge(
            "test-bucket", "2026-05-07", s3_client=s3,
        )
        assert out["n_agents"] == 2
        assert out["n_scored"] == 1
        assert out["mean_score"] == 4.0


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

    def test_reads_canonical_run_id_layout(self):
        """config#792 cutover: per-agent files are now {run_id}.json
        (YYMMDDHHMM) + a latest.json sidecar instead of {YYYY-Www}.json.
        The reader must pick the most-recent dated run_id and IGNORE the
        latest.json sidecar (otherwise it double-counts / mis-labels)."""
        s3 = _mock_s3_client(
            list_dirs={
                "decision_artifacts/_counterfactual/": ["ic_cio"],
            },
            list_objects={
                "decision_artifacts/_counterfactual/ic_cio/": [
                    # Two dated runs in the same week + the sidecar mirror.
                    "decision_artifacts/_counterfactual/ic_cio/2605030900.json",
                    "decision_artifacts/_counterfactual/ic_cio/2605100900.json",
                    "decision_artifacts/_counterfactual/ic_cio/latest.json",
                ],
            },
            get_objects={
                "decision_artifacts/_counterfactual/ic_cio/2605030900.json": {
                    "match_rate": 0.70,
                },
                "decision_artifacts/_counterfactual/ic_cio/2605100900.json": {
                    "match_rate": 0.95,
                },
                # latest.json mirrors the newest run; if the reader wrongly
                # selected it the match_rate would still be 0.95, so give it
                # a sentinel value that would corrupt the result if read.
                "decision_artifacts/_counterfactual/ic_cio/latest.json": {
                    "match_rate": -1.0,
                },
            },
        )
        out = agent_justification.summarize_counterfactual(
            "test-bucket", "2026-05-07", s3_client=s3,
        )
        assert out["status"] == "ok"
        assert out["n_agents"] == 1
        # Most-recent dated run (2605100900), NOT the sidecar sentinel.
        assert out["mean_match_rate"] == 0.95

    def test_tolerates_mixed_legacy_and_canonical_agents(self):
        """During cutover, some agents may have legacy {YYYY-Www} files
        and others canonical {run_id} files. Both must read cleanly."""
        s3 = _mock_s3_client(
            list_dirs={
                "decision_artifacts/_counterfactual/": ["ic_cio", "macro_economist"],
            },
            list_objects={
                # Legacy ISO-week file.
                "decision_artifacts/_counterfactual/ic_cio/": [
                    "decision_artifacts/_counterfactual/ic_cio/2026-W19.json",
                ],
                # Canonical run_id file + sidecar.
                "decision_artifacts/_counterfactual/macro_economist/": [
                    "decision_artifacts/_counterfactual/macro_economist/2605100900.json",
                    "decision_artifacts/_counterfactual/macro_economist/latest.json",
                ],
            },
            get_objects={
                "decision_artifacts/_counterfactual/ic_cio/2026-W19.json": {
                    "match_rate": 0.80,
                },
                "decision_artifacts/_counterfactual/macro_economist/2605100900.json": {
                    "match_rate": 0.90,
                },
                "decision_artifacts/_counterfactual/macro_economist/latest.json": {
                    "match_rate": -1.0,
                },
            },
        )
        out = agent_justification.summarize_counterfactual(
            "test-bucket", "2026-05-07", s3_client=s3,
        )
        assert out["status"] == "ok"
        assert out["n_agents"] == 2
        assert out["mean_match_rate"] == 0.85  # (0.80 + 0.90) / 2


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

    def test_reads_canonical_flat_layout(self):
        """config#792 cutover: summaries are now flat
        {run_id}_{target_model}.json directly under _replay_summary/ +
        a latest.json sidecar, instead of the {date}/{target_model}.json
        partition. Reader groups by target_model, keeps the most-recent
        dated key per model, and skips the latest.json sidecar."""
        prefix = "decision_artifacts/_replay_summary/"
        s3 = _mock_s3_client(
            list_dirs={},  # flat layout has no date sub-partitions
            list_objects={
                prefix: [
                    # Two runs for haiku (older + newer) + one for sonnet
                    # + the shared latest.json sidecar.
                    f"{prefix}2605030900_claude-haiku-4-5.json",
                    f"{prefix}2605100900_claude-haiku-4-5.json",
                    f"{prefix}2605100900_claude-sonnet-4-6.json",
                    f"{prefix}latest.json",
                ],
            },
            get_objects={
                f"{prefix}2605030900_claude-haiku-4-5.json": {"n_artifacts_replayed": 1},
                f"{prefix}2605100900_claude-haiku-4-5.json": {"n_artifacts_replayed": 9},
                f"{prefix}2605100900_claude-sonnet-4-6.json": {"n_artifacts_replayed": 7},
                f"{prefix}latest.json": {"n_artifacts_replayed": 999},
            },
        )
        out = agent_justification.summarize_concordance(
            "test-bucket", "2026-05-07", s3_client=s3,
        )
        assert out["status"] == "ok"
        assert out["layout"] == "canonical"
        # Two distinct target models, sidecar excluded.
        assert out["n_target_models"] == 2
        assert set(out["per_target"].keys()) == {
            "claude-haiku-4-5", "claude-sonnet-4-6",
        }
        # Most-recent dated key won for haiku (9, not the older 1).
        assert out["per_target"]["claude-haiku-4-5"]["n_artifacts_replayed"] == 9

    def test_falls_back_to_legacy_date_partition(self):
        """When no canonical flat keys are present (pre-cutover bucket),
        the reader falls back to the legacy {date}/{target_model}.json
        partition so existing data is never stranded."""
        prefix = "decision_artifacts/_replay_summary/"
        s3 = _mock_s3_client(
            list_dirs={
                # Legacy date partitions appear as CommonPrefixes.
                prefix: ["2026-05-09"],
            },
            list_objects={
                # No flat Contents directly under the prefix...
                prefix: [],
                # ...only the legacy date-partitioned files.
                f"{prefix}2026-05-09/": [
                    f"{prefix}2026-05-09/claude-haiku-4-5.json",
                ],
            },
            get_objects={
                f"{prefix}2026-05-09/claude-haiku-4-5.json": {
                    "n_artifacts_replayed": 5,
                },
            },
        )
        out = agent_justification.summarize_concordance(
            "test-bucket", "2026-05-09", s3_client=s3,
        )
        assert out["status"] == "ok"
        assert out["layout"] == "legacy"
        assert out["n_target_models"] == 1
        assert "claude-haiku-4-5" in out["per_target"]


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
