"""Unit tests for replay/comparison.py — per-agent agreement scorers.

Coverage strategy: each agent scorer gets a "perfect agreement → 1.0",
"complete disagreement → 0.0 (or close)", and a partial-overlap case.
Plus generic-fallback dispatch + edge cases (empty inputs, missing
fields, unknown agent_id).
"""

from __future__ import annotations

import pytest


# ── Generic helpers ──────────────────────────────────────────────────────


class TestJaccard:
    def test_perfect_agreement(self):
        from replay.comparison import _jaccard

        assert _jaccard([1, 2, 3], [1, 2, 3]) == 1.0

    def test_zero_overlap(self):
        from replay.comparison import _jaccard

        assert _jaccard([1, 2], [3, 4]) == 0.0

    def test_partial_overlap(self):
        from replay.comparison import _jaccard

        # |{1,2,3} ∩ {2,3,4}| = 2; union = 4 → 0.5
        assert _jaccard([1, 2, 3], [2, 3, 4]) == 0.5

    def test_both_empty_returns_zero(self):
        # Mathematician's convention is 1 (vacuously true), but for
        # "did the models agree?" a missing-data case must NOT be
        # reported as perfect agreement.
        from replay.comparison import _jaccard

        assert _jaccard([], []) == 0.0


class TestPearson:
    def test_perfect_correlation(self):
        from replay.comparison import _pearson

        assert _pearson([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)

    def test_perfect_anticorrelation(self):
        from replay.comparison import _pearson

        assert _pearson([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_too_few_points_returns_none(self):
        from replay.comparison import _pearson

        assert _pearson([1], [2]) is None

    def test_zero_variance_returns_none(self):
        from replay.comparison import _pearson

        assert _pearson([5, 5, 5], [1, 2, 3]) is None


class TestTextSimilarity:
    def test_identical_strings(self):
        from replay.comparison import _text_similarity

        assert _text_similarity("hello world", "hello world") == 1.0

    def test_disjoint_strings(self):
        from replay.comparison import _text_similarity

        assert _text_similarity("apple banana", "ferret giraffe") == 0.0

    def test_punctuation_ignored(self):
        from replay.comparison import _text_similarity

        # Word-set-based; punctuation doesn't perturb token boundaries.
        assert _text_similarity("the cat, sat", "the cat sat!") == 1.0


# ── sector_quant ─────────────────────────────────────────────────────────


class TestSectorQuantScorer:
    def test_perfect_agreement(self):
        from replay.comparison import _score_sector_quant

        picks = [
            {"ticker": "NVDA", "quant_score": 88},
            {"ticker": "AAPL", "quant_score": 75},
            {"ticker": "MSFT", "quant_score": 70},
        ]
        out = _score_sector_quant({"ranked_picks": picks}, {"ranked_picks": picks})
        assert out["agreement_score"] == 1.0
        assert out["top5_jaccard"] == 1.0
        assert out["ticker_overlap_count"] == 3
        assert out["score_correlation"] == pytest.approx(1.0)

    def test_zero_overlap(self):
        from replay.comparison import _score_sector_quant

        orig = {"ranked_picks": [{"ticker": "NVDA", "quant_score": 88}]}
        repl = {"ranked_picks": [{"ticker": "WMT", "quant_score": 70}]}
        out = _score_sector_quant(orig, repl)
        assert out["agreement_score"] == 0.0

    def test_partial_overlap_top5(self):
        from replay.comparison import _score_sector_quant

        # Top-5: 3 shared, 2 each unique → Jaccard = 3/7 ≈ 0.43
        orig_picks = [{"ticker": t, "quant_score": 80} for t in
                      ["A", "B", "C", "D", "E"]]
        repl_picks = [{"ticker": t, "quant_score": 75} for t in
                      ["A", "B", "C", "F", "G"]]
        out = _score_sector_quant(
            {"ranked_picks": orig_picks}, {"ranked_picks": repl_picks}
        )
        assert out["top5_overlap_count"] == 3
        assert out["top5_jaccard"] == pytest.approx(3 / 7)

    def test_empty_inputs_zero_agreement(self):
        from replay.comparison import _score_sector_quant

        out = _score_sector_quant({}, {})
        assert out["agreement_score"] == 0.0


# ── sector_qual ──────────────────────────────────────────────────────────


class TestSectorQualScorer:
    def test_perfect_ticker_overlap(self):
        from replay.comparison import _score_sector_qual

        a = [
            {"ticker": "PFE", "conviction": 70, "bull_case": "pipeline"},
            {"ticker": "MRK", "conviction": 65, "bull_case": "oncology"},
        ]
        out = _score_sector_qual({"assessments": a}, {"assessments": a})
        assert out["agreement_score"] == 1.0
        assert out["conviction_correlation"] == pytest.approx(1.0)


# ── sector_peer_review ───────────────────────────────────────────────────


class TestSectorPeerReviewScorer:
    def test_recommendations_plus_additional(self):
        from replay.comparison import _score_sector_peer_review

        recs = [{"ticker": "JPM"}, {"ticker": "BAC"}]
        out = _score_sector_peer_review(
            {"recommendations": recs, "additional_accepted": ["WFC"]},
            {"recommendations": recs, "additional_accepted": ["WFC"]},
        )
        assert out["agreement_score"] == 1.0
        assert out["additional_accepted_match"] is True

    def test_additional_accepted_mismatch_does_not_zero(self):
        from replay.comparison import _score_sector_peer_review

        # Tickers fully overlap; additional pick differs.
        recs = [{"ticker": "JPM"}, {"ticker": "BAC"}]
        out = _score_sector_peer_review(
            {"recommendations": recs, "additional_accepted": ["WFC"]},
            {"recommendations": recs, "additional_accepted": ["GS"]},
        )
        # 4 elements total (JPM + BAC + WFC + GS); 2 shared → 0.5.
        assert out["agreement_score"] == 0.5
        assert out["additional_accepted_match"] is False


# ── macro_economist ──────────────────────────────────────────────────────


class TestMacroEconomistScorer:
    def test_regime_match_drives_agreement(self):
        from replay.comparison import _score_macro_economist

        out = _score_macro_economist(
            {"market_regime": "BULL", "sector_modifiers": {"tech": 1.2, "fin": 1.0}},
            {"market_regime": "BULL", "sector_modifiers": {"tech": 1.2, "fin": 1.0}},
        )
        # Perfect: 0.6 × 1 (regime) + 0.4 × 1 (mod corr) = 1.0
        assert out["agreement_score"] == pytest.approx(1.0)
        assert out["regime_match"] is True

    def test_regime_mismatch_caps_lower(self):
        from replay.comparison import _score_macro_economist

        out = _score_macro_economist(
            {"market_regime": "BULL"}, {"market_regime": "BEAR"},
        )
        # 0.6 × 0 + 0.4 × 0 = 0.0 (no sector_modifiers either)
        assert out["agreement_score"] == 0.0
        assert out["regime_match"] is False
        assert out["original_regime"] == "BULL"
        assert out["replay_regime"] == "BEAR"


# ── ic_cio ───────────────────────────────────────────────────────────────


class TestIcCioScorer:
    def test_perfect_advanced_overlap(self):
        from replay.comparison import _score_ic_cio

        adv = ["NVDA", "AAPL", "MSFT"]
        decs = [
            {"ticker": "NVDA", "decision": "ADVANCE"},
            {"ticker": "AAPL", "decision": "ADVANCE"},
            {"ticker": "WMT", "decision": "REJECT"},
        ]
        out = _score_ic_cio(
            {"advanced_tickers": adv, "ic_decisions": decs},
            {"advanced_tickers": adv, "ic_decisions": decs},
        )
        # Perfect: 0.7 × 1.0 (adv jaccard) + 0.3 × 1.0 (decisions) = 1.0
        assert out["agreement_score"] == pytest.approx(1.0)
        assert out["advanced_jaccard"] == 1.0
        assert out["decision_agreement"] == 1.0

    def test_partial_advanced_partial_decisions(self):
        from replay.comparison import _score_ic_cio

        # 2 of 3 advanced overlap → adv_jaccard = 2/4 = 0.5
        # Both decisions agree on the overlapping ticker → 1.0
        # Score: 0.7 × 0.5 + 0.3 × 1.0 = 0.65
        out = _score_ic_cio(
            {
                "advanced_tickers": ["A", "B", "C"],
                "ic_decisions": [
                    {"ticker": "A", "decision": "ADVANCE"},
                    {"ticker": "B", "decision": "ADVANCE"},
                ],
            },
            {
                "advanced_tickers": ["A", "B", "D"],
                "ic_decisions": [
                    {"ticker": "A", "decision": "ADVANCE"},
                    {"ticker": "B", "decision": "ADVANCE"},
                ],
            },
        )
        assert out["agreement_score"] == pytest.approx(0.65)


# ── thesis_update ────────────────────────────────────────────────────────


class TestThesisUpdateScorer:
    def test_numeric_only(self):
        from replay.comparison import _score_thesis_update

        out = _score_thesis_update(
            {"final_score": 70, "conviction": 65},
            {"final_score": 70, "conviction": 65},
        )
        # Pure numeric path (no text fields) → agreement = 1.0
        assert out["agreement_score"] == pytest.approx(1.0)
        assert out["final_score_agreement"] == 1.0
        assert out["conviction_agreement"] == 1.0

    def test_score_drift_lowers_agreement(self):
        from replay.comparison import _score_thesis_update

        # final_score drift of 20 → 1 - 20/100 = 0.8
        out = _score_thesis_update(
            {"final_score": 70}, {"final_score": 50},
        )
        assert out["final_score_agreement"] == pytest.approx(0.8)


# ── Generic fallback ─────────────────────────────────────────────────────


class TestGenericFallback:
    def test_unknown_agent_uses_generic(self):
        from replay.comparison import compute_comparison

        out = compute_comparison(
            agent_id="brand_new_agent",
            original_output={"a": 1, "b": 2},
            replay_output={"a": 1, "c": 3},
        )
        assert out["scorer"] == "generic"
        # Keys jaccard: {a, b} ∪ {a, c} = 3, ∩ = 1 → 1/3
        assert out["key_jaccard"] == pytest.approx(1 / 3)
        assert out["agreement_score"] == pytest.approx(1 / 3)


# ── Dispatch ─────────────────────────────────────────────────────────────


class TestDispatch:
    @pytest.mark.parametrize("agent_id,expected_scorer", [
        ("sector_quant:tech", "sector_quant"),
        ("sector_qual:healthcare", "sector_qual"),
        ("sector_peer_review:financials", "sector_peer_review"),
        ("macro_economist", "macro_economist"),
        ("ic_cio", "ic_cio"),
        ("thesis_update:AAPL", "thesis_update"),
        ("brand_new", "generic"),
    ])
    def test_scorer_dispatch_by_agent_id(self, agent_id, expected_scorer):
        from replay.comparison import compute_comparison

        out = compute_comparison(
            agent_id=agent_id,
            original_output={},
            replay_output={},
        )
        assert out["scorer"] == expected_scorer
        assert out["agent_id_base"] == agent_id.split(":", 1)[0]

    def test_compute_comparison_always_returns_required_fields(self):
        from replay.comparison import compute_comparison

        out = compute_comparison(
            agent_id="sector_quant:x",
            original_output={"ranked_picks": [{"ticker": "A"}]},
            replay_output={"ranked_picks": [{"ticker": "A"}]},
        )
        assert "agreement_score" in out
        assert "diff_summary" in out
        assert "scorer" in out
        assert "agent_id_base" in out
        assert isinstance(out["agreement_score"], float)
        assert 0.0 <= out["agreement_score"] <= 1.0


# ── Runner integration ──────────────────────────────────────────────────


class TestRunnerIntegration:
    def test_replay_artifact_populates_comparison_block(self):
        """The runner should call compute_comparison and stamp the
        ReplayOutput.comparison field on every structured replay.

        alpha-engine-config-I2997 (2026-07-19): fake transport rebuilt on
        the krepis.llm.LLMClient OpenRouter client_factory seam (mirrors
        tests/test_replay_runner.py) — was a fake ChatAnthropic factory.
        """
        from unittest.mock import MagicMock
        import json
        from replay.runner import replay_artifact
        from tests.test_replay_runner import _make_krepis_factory

        artifact = {
            "schema_version": 1,
            "run_id": "r1",
            "timestamp": "2026-05-03T00:00:00Z",
            "agent_id": "sector_quant:tech",
            "model_metadata": {"model_name": "claude-sonnet-4-6"},
            "full_prompt_context": {
                "system_prompt": "s", "user_prompt": "u",
                "tool_definitions": [{"name": "emit", "description": "x", "input_schema": {}}],
            },
            "input_data_snapshot": {},
            "agent_output": {
                "ranked_picks": [
                    {"ticker": "NVDA", "quant_score": 88},
                    {"ticker": "AAPL", "quant_score": 75},
                ],
            },
        }

        s3 = MagicMock()
        body = MagicMock()
        body.read.return_value = json.dumps(artifact).encode("utf-8")
        s3.get_object.return_value = {"Body": body}
        s3.put_object = MagicMock()

        from nousergon_lib.agent_schemas import QuantAnalystOutput
        parsed = QuantAnalystOutput(ranked_picks=[
            {"ticker": "NVDA", "quant_score": 85, "rationale": "x"},
            {"ticker": "AAPL", "quant_score": 73, "rationale": "y"},
        ])
        factory, _ = _make_krepis_factory(
            content=json.dumps(parsed.model_dump()),
            prompt_tokens=100, completion_tokens=50,
        )

        replay = replay_artifact(
            artifact_key="k.json", target_model="deepseek/deepseek-v4-flash",
            s3_client=s3, client_factory=factory, api_key="sk-or-test",
            persist=False,
        )

        # Comparison block populated with sector_quant scorer.
        assert replay.comparison["scorer"] == "sector_quant"
        assert replay.comparison["agreement_score"] == 1.0  # tickers identical
        assert replay.comparison["top5_jaccard"] == 1.0

    def test_error_replay_skips_comparison_with_marker(self):
        from unittest.mock import MagicMock
        import json
        from replay.runner import replay_artifact
        from tests.test_replay_runner import _make_krepis_factory

        artifact = {
            "schema_version": 1,
            "run_id": "r1", "timestamp": "2026-05-03T00:00:00Z",
            "agent_id": "sector_quant:tech",
            "model_metadata": {"model_name": "claude-sonnet-4-6"},
            "full_prompt_context": {
                "system_prompt": "s", "user_prompt": "u", "tool_definitions": [],
            },
            "input_data_snapshot": {},
            "agent_output": {"ranked_picks": []},
        }
        s3 = MagicMock()
        body = MagicMock()
        body.read.return_value = json.dumps(artifact).encode("utf-8")
        s3.get_object.return_value = {"Body": body}
        s3.put_object = MagicMock()

        # Transport client raises on create — replay must capture the
        # error rather than propagate.
        factory, _ = _make_krepis_factory(raise_on_create=RuntimeError("API down"))

        replay = replay_artifact(
            artifact_key="k.json", target_model="deepseek/deepseek-v4-flash",
            s3_client=s3, client_factory=factory, api_key="sk-or-test",
            persist=False,
        )

        assert replay.replay_output_kind == "error"
        # Skip-marker comparison rather than running generic scorer
        # against empty output (which would emit a misleading 0.0
        # signal that PR C's CW metric would aggregate as
        # "concordance dropped").
        assert replay.comparison["scorer"] == "skipped"
        assert replay.comparison["agreement_score"] == 0.0
