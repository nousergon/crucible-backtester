"""champion_promotion — weekly winner-take-all champion/challenger gate
(alpha-engine-config-I2518 / epic I2515, 2026-07-14 ruling; supersedes the
config#2364/#2367 HAC-significance/hysteresis/cooldown engine).

Pins: (1) the seat swap — VALID_CHAMPIONS is now
(scanner_predictor_direct, thinktank_coverage), "agentic" is read-tolerated
on pointer/audit READ paths (WARN + normalize) but write-forbidden; (2)
weekly winner-take-all — whichever arm's realized score is higher this
week wins, no significance/hysteresis/cooldown; (3) validity guards —
either arm's score being unavailable (missing leaderboard, thinktank_coverage
not yet in the leaderboard, no resolved outcomes) is a NO-CONTEST, never a
default win; (4) --freeze suppresses the pointer write but the audit record
is still written every week (the liveness proxy, config#2054); (5)
frozen-schema (v2) conformance against contracts/producer_champion.schema.json
and contracts/producer_champion_audit.schema.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from optimizer.champion_promotion import (
    OUTCOMES,
    VALID_CHAMPIONS,
    build_champion_audit,
    build_leaderboard_artifact,
    build_weekly_arm_scores,
    evaluate_gates,
    hac_significance,
    leaderboard_entry_from_e2e_lift,
    leaderboard_gate_inputs,
    read_champion_pointer,
    read_prior_leaderboard_history,
    read_research_producer_leaderboard,
    run_weekly_evaluation,
    write_champion_pointer,
    write_leaderboard,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
POINTER_SCHEMA_PATH = REPO_ROOT / "contracts" / "producer_champion.schema.json"
AUDIT_SCHEMA_PATH = REPO_ROOT / "contracts" / "producer_champion_audit.schema.json"


# ── HAC/Newey-West significance (retained utility, not gate-connected) ─────


class TestHacSignificance:
    """hac_significance is retained as an independently-tested utility (see
    module docstring) even though the winner-take-all gate no longer calls
    it — these pins guard against accidental regression of code that other
    future work may still depend on."""

    def test_insufficient_data_below_two_points(self):
        assert hac_significance([])["status"] == "insufficient_data"
        assert hac_significance([0.01])["status"] == "insufficient_data"

    def test_significant_positive_series_flagged(self):
        series = [0.02, 0.021, 0.019, 0.022, 0.02, 0.018, 0.021]
        result = hac_significance(series)
        assert result["status"] == "ok"
        assert result["significant"] is True
        assert result["mean"] > 0

    def test_noisy_near_zero_series_not_significant(self):
        series = [0.01, -0.015, 0.008, -0.006, 0.002, -0.009, 0.004]
        result = hac_significance(series)
        assert result["status"] == "ok"
        assert result["significant"] is False

    def test_lag_selection_is_horizon_over_cadence(self):
        series = [0.01] * 10
        result = hac_significance(series)
        assert result["lags"] == 3


# ── Weekly arm-score sourcing ────────────────────────────────────────────────


def _e2e_lift_ok(sn_lift=0.02, n_cycles=6):
    return {
        "scanner_then_predictor_counterfactual": {
            "status": "ok",
            "n_cycles": n_cycles,
            "methods": {
                "scanner_then_predictor_topN": {
                    "mean_alpha_21d": 0.03,
                    "sector_neutral_mean_alpha_21d": 0.025,
                    "n_picks": 40,
                    "sn_lift_vs_agentic_cio": sn_lift,
                },
                "agentic_cio_advance": {
                    "mean_alpha_21d": 0.01,
                    "sector_neutral_mean_alpha_21d": 0.005,
                    "n_picks": 40,
                },
            },
        },
    }


def _tt_leaderboard_ok(run_date="2026-07-18", mean=0.015, n_dates_scored=5):
    """Mimics the REAL crucible-research schema
    (scoring/leaderboard_producers.py::build_producer_leaderboard /
    scoring/leaderboard_scoring.py::score_leaderboard) verified against the
    crucible-research checkout 2026-07-14."""
    return {
        "champion": "agentic_sector_teams",
        "horizon_days": 21,
        "top_n": 50,
        "n_dates": 12,
        "date": run_date,
        "leaderboard_id": "producer",
        "specs": [
            {
                "name": "agentic_sector_teams", "kind": "champion",
                "realized_rank_ic": {"mean": 0.05, "se": 0.02, "t_stat": 2.5, "n_dates": 12},
                "topn_alpha_vs_champion": None, "n_dates_scored": 12,
            },
            {
                "name": "no_agent_quant", "kind": "challenger",
                "realized_rank_ic": {"mean": 0.03, "se": 0.02, "t_stat": 1.5, "n_dates": 12},
                "topn_alpha_vs_champion": {"mean": 0.008, "se": 0.01, "t_stat": 0.8, "n_dates": 12},
                "n_dates_scored": 12,
            },
            {
                "name": "thinktank_coverage", "kind": "challenger",
                "realized_rank_ic": {"mean": 0.04, "se": 0.02, "t_stat": 2.0, "n_dates": n_dates_scored},
                "topn_alpha_vs_champion": {"mean": mean, "se": 0.01, "t_stat": 1.5, "n_dates": n_dates_scored},
                "n_dates_scored": n_dates_scored,
            },
        ],
    }


class TestBuildWeeklyArmScores:
    def test_both_sides_valid(self):
        result = build_weekly_arm_scores(
            _e2e_lift_ok(sn_lift=0.02), _tt_leaderboard_ok(mean=0.015), run_date="2026-07-18",
        )
        assert result["scores"]["scanner_predictor_direct"] == 0.02
        assert result["scores"]["thinktank_coverage"] == 0.015
        assert result["unavailable_reasons"] == {}

    def test_missing_e2e_lift(self):
        result = build_weekly_arm_scores(None, _tt_leaderboard_ok(), run_date="2026-07-18")
        assert result["scores"]["scanner_predictor_direct"] is None
        assert result["unavailable_reasons"]["scanner_predictor_direct"] == (
            "scanner_predictor_direct_counterfactual_unavailable"
        )

    def test_missing_leaderboard(self):
        result = build_weekly_arm_scores(_e2e_lift_ok(), None, run_date="2026-07-18")
        assert result["scores"]["thinktank_coverage"] is None
        assert result["unavailable_reasons"]["thinktank_coverage"] == "leaderboard_unavailable"

    def test_stale_leaderboard_date_mismatch(self):
        stale = _tt_leaderboard_ok(run_date="2026-07-11")
        result = build_weekly_arm_scores(_e2e_lift_ok(), stale, run_date="2026-07-18")
        assert result["scores"]["thinktank_coverage"] is None
        assert result["unavailable_reasons"]["thinktank_coverage"] == "leaderboard_stale"

    def test_thinktank_coverage_not_registered_in_leaderboard(self):
        """The KNOWN, TRACKED GAP (module docstring / alpha-engine-config
        -I2519): thinktank_coverage isn't yet registered in crucible
        -research's producers/registry.py, so its row is simply absent from
        specs -- must be an honest no-contest reason, not a crash."""
        lb = _tt_leaderboard_ok()
        lb["specs"] = [s for s in lb["specs"] if s["name"] != "thinktank_coverage"]
        result = build_weekly_arm_scores(_e2e_lift_ok(), lb, run_date="2026-07-18")
        assert result["scores"]["thinktank_coverage"] is None
        assert result["unavailable_reasons"]["thinktank_coverage"] == (
            "thinktank_coverage_not_in_leaderboard"
        )

    def test_thinktank_coverage_zero_dates_scored(self):
        lb = _tt_leaderboard_ok(n_dates_scored=0)
        for s in lb["specs"]:
            if s["name"] == "thinktank_coverage":
                s["topn_alpha_vs_champion"] = None
        result = build_weekly_arm_scores(_e2e_lift_ok(), lb, run_date="2026-07-18")
        assert result["scores"]["thinktank_coverage"] is None
        assert result["unavailable_reasons"]["thinktank_coverage"] == (
            "thinktank_coverage_no_resolved_outcomes"
        )

    def test_malformed_specs_is_leaderboard_unavailable(self):
        lb = {"date": "2026-07-18", "specs": "not-a-list"}
        result = build_weekly_arm_scores(_e2e_lift_ok(), lb, run_date="2026-07-18")
        assert result["scores"]["thinktank_coverage"] is None
        assert result["unavailable_reasons"]["thinktank_coverage"] == "leaderboard_unavailable"


# ── Gate engine (pure function, weekly winner-take-all) ────────────────────


class TestEvaluateGates:
    def test_seat_swap_valid_champions(self):
        assert VALID_CHAMPIONS == ("scanner_predictor_direct", "thinktank_coverage")
        assert "agentic" not in VALID_CHAMPIONS

    def test_challenger_wins_promotes(self):
        arm_scores = {
            "scores": {"scanner_predictor_direct": 0.01, "thinktank_coverage": 0.03},
            "unavailable_reasons": {},
        }
        result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=False,
        )
        assert result["outcome"] == "promoted"
        assert result["champion_after"] == "thinktank_coverage"
        assert result["champion_score"] == 0.01
        assert result["challenger_score"] == 0.03
        assert result["blocked_by"] is None

    def test_champion_still_winning_stays_champion(self):
        arm_scores = {
            "scores": {"scanner_predictor_direct": 0.03, "thinktank_coverage": 0.01},
            "unavailable_reasons": {},
        }
        result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=False,
        )
        assert result["outcome"] == "unchanged_winner_already_champion"
        assert result["champion_after"] == "scanner_predictor_direct"
        assert result["blocked_by"] is None

    def test_exact_tie_favors_incumbent(self):
        arm_scores = {
            "scores": {"scanner_predictor_direct": 0.02, "thinktank_coverage": 0.02},
            "unavailable_reasons": {},
        }
        result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=False,
        )
        assert result["outcome"] == "unchanged_winner_already_champion"
        assert result["champion_after"] == "scanner_predictor_direct"

    def test_bidirectional_thinktank_coverage_as_champion(self):
        """Same rule set, reversed seats: thinktank_coverage is champion,
        scanner_predictor_direct challenges and wins -> promotes."""
        arm_scores = {
            "scores": {"scanner_predictor_direct": 0.04, "thinktank_coverage": 0.01},
            "unavailable_reasons": {},
        }
        result = evaluate_gates(
            champion_before="thinktank_coverage", arm_scores=arm_scores, freeze=False,
        )
        assert result["outcome"] == "promoted"
        assert result["challenger"] == "scanner_predictor_direct"
        assert result["champion_after"] == "scanner_predictor_direct"

    def test_no_contest_champion_score_missing(self):
        arm_scores = {
            "scores": {"scanner_predictor_direct": None, "thinktank_coverage": 0.02},
            "unavailable_reasons": {"scanner_predictor_direct": "scanner_predictor_direct_counterfactual_unavailable"},
        }
        result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=False,
        )
        assert result["outcome"] == "no_contest"
        assert result["blocked_by"] == ["scanner_predictor_direct_counterfactual_unavailable"]
        assert result["champion_after"] == "scanner_predictor_direct"

    def test_no_contest_challenger_score_missing(self):
        arm_scores = {
            "scores": {"scanner_predictor_direct": 0.02, "thinktank_coverage": None},
            "unavailable_reasons": {"thinktank_coverage": "thinktank_coverage_not_in_leaderboard"},
        }
        result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=False,
        )
        assert result["outcome"] == "no_contest"
        assert result["blocked_by"] == ["thinktank_coverage_not_in_leaderboard"]

    def test_no_contest_both_sides_missing(self):
        arm_scores = {"scores": {"scanner_predictor_direct": None, "thinktank_coverage": None},
                       "unavailable_reasons": {}}
        result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=False,
        )
        assert result["outcome"] == "no_contest"
        assert result["blocked_by"] == ["arm_score_unavailable", "arm_score_unavailable"]

    def test_freeze_suppresses_pointer_move_but_reports_would_be_outcome(self):
        arm_scores = {
            "scores": {"scanner_predictor_direct": 0.01, "thinktank_coverage": 0.03},
            "unavailable_reasons": {},
        }
        result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=True,
        )
        assert result["outcome"] == "promoted"
        assert result["blocked_by"] == ["frozen"]
        assert result["champion_after"] == "scanner_predictor_direct"  # NOT advanced under freeze


# ── Pointer writer (single writer, dual promotion_source caller) ───────────


class TestWriteChampionPointer:
    def test_writes_expected_schema(self):
        s3 = MagicMock()
        pointer = write_champion_pointer(
            "bucket", "scanner_predictor_direct",
            promotion_source="gate_engine", upload=True, s3_client=s3,
        )
        assert pointer["schema_version"] == 1
        assert pointer["champion"] == "scanner_predictor_direct"
        assert pointer["promotion_source"] == "gate_engine"
        assert "promoted_at" in pointer
        s3.put_object.assert_called_once()
        call = s3.put_object.call_args
        assert call.kwargs["Key"] == "config/producer_champion.json"
        body = json.loads(call.kwargs["Body"])
        assert body == pointer

    def test_writes_thinktank_coverage(self):
        s3 = MagicMock()
        pointer = write_champion_pointer(
            "bucket", "thinktank_coverage",
            promotion_source="gate_engine", upload=True, s3_client=s3,
        )
        assert pointer["champion"] == "thinktank_coverage"

    def test_upload_false_skips_s3(self):
        s3 = MagicMock()
        write_champion_pointer(
            "bucket", "scanner_predictor_direct", promotion_source="gate_engine",
            upload=False, s3_client=s3,
        )
        s3.put_object.assert_not_called()

    def test_rejects_unknown_champion(self):
        with pytest.raises(ValueError):
            write_champion_pointer(
                "bucket", "not_a_real_arm", promotion_source="gate_engine",
                upload=False,
            )

    def test_rejects_retired_agentic_seat(self):
        """Write-forbidden half of the read-tolerated/write-forbidden
        posture: 'agentic' is no longer in VALID_CHAMPIONS, so any attempt
        to write it must raise -- belt-and-braces against ever re-writing
        the retired seat."""
        with pytest.raises(ValueError):
            write_champion_pointer(
                "bucket", "agentic", promotion_source="gate_engine", upload=False,
            )


class TestReadChampionPointer:
    def test_missing_key_returns_none(self):
        from botocore.exceptions import ClientError
        s3 = MagicMock()
        s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject",
        )
        assert read_champion_pointer("bucket", s3_client=s3) is None

    def test_reads_legacy_agentic_pointer_without_crashing(self):
        """A historical pointer object (or a defensive/manual read) carrying
        'agentic' must be readable without error -- normalization to the
        base-case arm happens in run_weekly_evaluation, not here (this
        function is a raw, faithful reader)."""
        s3 = MagicMock()
        body = json.dumps({"schema_version": 1, "champion": "agentic",
                            "promoted_at": "2026-07-01T00:00:00Z",
                            "promotion_source": "operator_bootstrap"}).encode()
        s3.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=body))}
        pointer = read_champion_pointer("bucket", s3_client=s3)
        assert pointer["champion"] == "agentic"


# ── Leaderboard artifact (research/producer_leaderboard_champion_gate/{date}.json) ───────
# Retained for observability / config#2452 continuity — see module docstring.


def test_leaderboard_key_distinct_from_research_producer_leaderboard():
    """config#2452 regression guard: this module's OWN observability key
    must never collide with crucible-research's
    scoring/leaderboard_producers.py key (research/producer_leaderboard/
    {date}.json) -- a prior version of this module shared that exact key
    with an incompatible schema."""
    from optimizer.champion_promotion import LEADERBOARD_KEY_TMPL
    assert LEADERBOARD_KEY_TMPL != "research/producer_leaderboard/{date}.json"
    assert LEADERBOARD_KEY_TMPL.format(date="2026-07-13") != "research/producer_leaderboard/2026-07-13.json"


class _FakeS3:
    """Minimal in-memory S3 stand-in supporting exactly the get/put calls
    champion_promotion.py issues, keyed by (Bucket, Key)."""

    def __init__(self):
        self.store: dict[str, bytes] = {}

    def get_object(self, Bucket, Key):
        from botocore.exceptions import ClientError
        full = f"{Bucket}/{Key}"
        if full not in self.store:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": MagicMock(read=MagicMock(return_value=self.store[full]))}

    def put_object(self, Bucket, Key, Body, ContentType=None):
        full = f"{Bucket}/{Key}"
        self.store[full] = Body if isinstance(Body, bytes) else Body.encode()


class TestLeaderboardObservability:
    def test_entry_extraction_from_e2e_lift(self):
        entry = leaderboard_entry_from_e2e_lift(_e2e_lift_ok(sn_lift=0.017))
        assert entry["sn_lift_vs_agentic_cio"] == 0.017
        assert entry["n_cycles"] == 6

    def test_entry_extraction_handles_missing_or_skipped(self):
        assert leaderboard_entry_from_e2e_lift(None) is None
        assert leaderboard_entry_from_e2e_lift({}) is None
        skipped = {"scanner_then_predictor_counterfactual": {"status": "skipped", "reason": "x"}}
        assert leaderboard_entry_from_e2e_lift(skipped) is None

    def test_build_leaderboard_appends_and_dedupes_by_date(self):
        history = [{"date": "2026-07-04", "sn_lift_vs_agentic_cio": 0.01}]
        entry = leaderboard_entry_from_e2e_lift(_e2e_lift_ok(sn_lift=0.02))
        artifact = build_leaderboard_artifact("2026-07-11", history, entry)
        dates = [p["date"] for p in artifact["weekly_points"]]
        assert dates == ["2026-07-04", "2026-07-11"]

    def test_history_scan_anchors_on_run_date_not_wall_clock(self):
        s3 = _FakeS3()
        prior_artifact = {
            "schema_version": 1, "as_of": "2020-01-03",
            "weekly_points": [{"date": "2020-01-03", "sn_lift_vs_agentic_cio": 0.011}],
        }
        write_leaderboard("bucket", "2020-01-03", prior_artifact, s3_client=s3)
        history = read_prior_leaderboard_history("bucket", "2020-01-10", s3_client=s3)
        assert history == [{"date": "2020-01-03", "sn_lift_vs_agentic_cio": 0.011}]

    def test_history_scan_cold_start_returns_empty(self):
        s3 = _FakeS3()
        history = read_prior_leaderboard_history("bucket", "2026-07-11", s3_client=s3)
        assert history == []

    def test_gate_inputs_reduction(self):
        artifact = {
            "weekly_points": [
                {"date": "2026-06-27", "sn_lift_vs_agentic_cio": 0.01},
                {"date": "2026-07-04", "sn_lift_vs_agentic_cio": 0.02},
                {"date": "2026-07-11", "sn_lift_vs_agentic_cio": None},
            ],
        }
        gi = leaderboard_gate_inputs(artifact)
        assert gi["challenger_matured_cohorts"] == 2
        assert gi["challenger_weekly_sn_lift"] == [0.01, 0.02]


class TestReadResearchProducerLeaderboard:
    """The NEW (I2518) read of crucible-research's real champion/challenger
    producer leaderboard — thinktank_coverage's evidence source."""

    def test_missing_key_returns_none(self):
        s3 = _FakeS3()
        assert read_research_producer_leaderboard("bucket", "2026-07-18", s3_client=s3) is None

    def test_reads_existing_artifact(self):
        s3 = _FakeS3()
        lb = _tt_leaderboard_ok(run_date="2026-07-18")
        s3.put_object(Bucket="bucket", Key="research/producer_leaderboard/2026-07-18.json",
                       Body=json.dumps(lb).encode())
        result = read_research_producer_leaderboard("bucket", "2026-07-18", s3_client=s3)
        assert result["date"] == "2026-07-18"
        names = [s["name"] for s in result["specs"]]
        assert "thinktank_coverage" in names

    def test_key_matches_crucible_research_producer(self):
        from optimizer.champion_promotion import RESEARCH_PRODUCER_LEADERBOARD_KEY_TMPL
        assert RESEARCH_PRODUCER_LEADERBOARD_KEY_TMPL == "research/producer_leaderboard/{date}.json"


# ── Weekly audit record (config/apply_audit/producer_champion/{date}.json) ─


class TestBuildChampionAudit:
    def test_error_path_conforms_and_reports_unavailable_leaderboard(self):
        audit = build_champion_audit("2026-07-18", None, freeze=False, error="leaderboard missing")
        assert audit["outcome"] == "error"
        assert audit["blocked_by"] == ["leaderboard_unavailable"]
        assert audit["champion_before"] is None
        assert audit["schema_version"] == 2

    def test_no_contest_path_records_zero_pointer_movement(self):
        arm_scores = {
            "scores": {"scanner_predictor_direct": 0.01, "thinktank_coverage": None},
            "unavailable_reasons": {"thinktank_coverage": "thinktank_coverage_not_in_leaderboard"},
        }
        gate_result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=False,
        )
        audit = build_champion_audit("2026-07-18", gate_result, freeze=False)
        assert audit["outcome"] == "no_contest"
        assert audit["champion_before"] == audit["champion_after"] == "scanner_predictor_direct"

    @pytest.mark.parametrize("outcome", OUTCOMES)
    def test_all_outcomes_are_in_frozen_vocabulary(self, outcome):
        assert outcome in (
            "promoted", "no_contest", "unchanged_winner_already_champion", "error",
        )


# ── Synthetic end-to-end transitions via run_weekly_evaluation ──────────────


class TestRunWeeklyEvaluation:
    BUCKET = "test-bucket"

    def test_challenger_wins_promotes_and_writes_pointer(self):
        s3 = _FakeS3()
        result = run_weekly_evaluation(
            bucket=self.BUCKET, run_date="2026-07-18",
            e2e_lift=_e2e_lift_ok(sn_lift=0.01),
            tt_leaderboard=_tt_leaderboard_ok(run_date="2026-07-18", mean=0.03),
            freeze=False, upload=True, s3_client=s3,
        )
        assert result["outcome"] == "promoted"
        assert result["champion_after"] == "thinktank_coverage"
        pointer = json.loads(s3.store[f"{self.BUCKET}/config/producer_champion.json"])
        assert pointer["champion"] == "thinktank_coverage"
        assert pointer["promotion_source"] == "gate_engine"
        assert f"{self.BUCKET}/config/apply_audit/producer_champion/2026-07-18.json" in s3.store
        assert f"{self.BUCKET}/config/apply_audit/producer_champion/latest.json" in s3.store

    def test_champion_defends_no_pointer_write(self):
        s3 = _FakeS3()
        result = run_weekly_evaluation(
            bucket=self.BUCKET, run_date="2026-07-18",
            e2e_lift=_e2e_lift_ok(sn_lift=0.03),
            tt_leaderboard=_tt_leaderboard_ok(run_date="2026-07-18", mean=0.01),
            freeze=False, upload=True, s3_client=s3,
        )
        assert result["outcome"] == "unchanged_winner_already_champion"
        assert f"{self.BUCKET}/config/producer_champion.json" not in s3.store

    def test_no_contest_missing_tt_week_no_pointer_write(self):
        s3 = _FakeS3()
        result = run_weekly_evaluation(
            bucket=self.BUCKET, run_date="2026-07-18",
            e2e_lift=_e2e_lift_ok(sn_lift=0.03),
            tt_leaderboard=None,
            freeze=False, upload=True, s3_client=s3,
        )
        assert result["outcome"] == "no_contest"
        assert result["blocked_by"] == ["leaderboard_unavailable"]
        assert f"{self.BUCKET}/config/producer_champion.json" not in s3.store
        assert f"{self.BUCKET}/config/apply_audit/producer_champion/2026-07-18.json" in s3.store

    def test_no_contest_thinktank_not_yet_registered(self):
        """Mirrors the CURRENT real-world state (2026-07-14): thinktank
        _coverage's row is absent from the real leaderboard until
        crucible-research registers it -- must be an honest no-contest."""
        s3 = _FakeS3()
        lb = _tt_leaderboard_ok(run_date="2026-07-18")
        lb["specs"] = [s for s in lb["specs"] if s["name"] != "thinktank_coverage"]
        result = run_weekly_evaluation(
            bucket=self.BUCKET, run_date="2026-07-18",
            e2e_lift=_e2e_lift_ok(sn_lift=0.03),
            tt_leaderboard=lb,
            freeze=False, upload=True, s3_client=s3,
        )
        assert result["outcome"] == "no_contest"
        assert result["blocked_by"] == ["thinktank_coverage_not_in_leaderboard"]

    def test_freeze_suppresses_pointer_write_but_audit_always_written(self):
        s3 = _FakeS3()
        result = run_weekly_evaluation(
            bucket=self.BUCKET, run_date="2026-07-18",
            e2e_lift=_e2e_lift_ok(sn_lift=0.01),
            tt_leaderboard=_tt_leaderboard_ok(run_date="2026-07-18", mean=0.03),
            freeze=True, upload=True, s3_client=s3,
        )
        assert result["outcome"] == "promoted"
        assert result["blocked_by"] == ["frozen"]
        assert f"{self.BUCKET}/config/producer_champion.json" not in s3.store
        assert f"{self.BUCKET}/config/apply_audit/producer_champion/2026-07-18.json" in s3.store

    def test_legacy_agentic_pointer_normalizes_without_crashing(self):
        """Belt-and-braces: a pointer object carrying the retired 'agentic'
        value must be readable/normalized without raising -- treated as
        scanner_predictor_direct for gate purposes. Not expected in
        practice (the live pointer has been scanner_predictor_direct since
        2026-07-13), but must be safe if it ever occurs."""
        s3 = _FakeS3()
        s3.put_object(
            Bucket=self.BUCKET, Key="config/producer_champion.json",
            Body=json.dumps({
                "schema_version": 1, "champion": "agentic",
                "promoted_at": "2026-07-01T00:00:00Z",
                "promotion_source": "operator_bootstrap",
            }).encode(),
        )
        result = run_weekly_evaluation(
            bucket=self.BUCKET, run_date="2026-07-18",
            e2e_lift=_e2e_lift_ok(sn_lift=0.03),
            tt_leaderboard=_tt_leaderboard_ok(run_date="2026-07-18", mean=0.01),
            freeze=False, upload=True, s3_client=s3,
        )
        # champion_before normalized to scanner_predictor_direct, which wins
        # this synthetic week (0.03 > 0.01) -> unchanged (already champion).
        assert result["champion_before"] == "scanner_predictor_direct"
        assert result["outcome"] == "unchanged_winner_already_champion"

    def test_scoring_exception_is_error_outcome_but_still_audited(self):
        """A malformed thinktank_coverage row (topn_alpha_vs_champion.mean
        is non-numeric) raises inside _score_thinktank_coverage's float()
        call -- run_weekly_evaluation's own try/except must catch it,
        record outcome='error', and STILL write the audit record (the
        liveness proxy, config#2054) rather than propagating the crash."""
        s3 = _FakeS3()
        lb = _tt_leaderboard_ok(run_date="2026-07-18")
        for spec in lb["specs"]:
            if spec["name"] == "thinktank_coverage":
                spec["topn_alpha_vs_champion"] = {"mean": "not-a-number"}
        result = run_weekly_evaluation(
            bucket=self.BUCKET, run_date="2026-07-18",
            e2e_lift=_e2e_lift_ok(sn_lift=0.03),
            tt_leaderboard=lb,
            freeze=False, upload=True, s3_client=s3,
        )
        assert result["outcome"] == "error"
        assert result["champion_before"] is None
        assert result["champion_after"] is None
        assert f"{self.BUCKET}/config/apply_audit/producer_champion/2026-07-18.json" in s3.store
        assert f"{self.BUCKET}/config/producer_champion.json" not in s3.store


# ── Frozen-schema conformance ────────────────────────────────────────────────


class TestSchemaConformance:
    def _validate(self, schema_path, instance):
        jsonschema = pytest.importorskip("jsonschema", reason="jsonschema not installed")
        schema = json.loads(schema_path.read_text())
        jsonschema.validate(instance=instance, schema=schema)

    def test_pointer_conforms(self):
        pointer = write_champion_pointer(
            "bucket", "scanner_predictor_direct",
            promotion_source="gate_engine", upload=False,
        )
        self._validate(POINTER_SCHEMA_PATH, pointer)

    def test_pointer_conforms_thinktank_coverage(self):
        pointer = write_champion_pointer(
            "bucket", "thinktank_coverage", promotion_source="gate_engine", upload=False,
        )
        self._validate(POINTER_SCHEMA_PATH, pointer)

    def test_legacy_agentic_pointer_shape_is_schema_valid(self):
        """Read-tolerance: a historical pointer-shaped object with
        champion='agentic' must still validate against the schema (the
        schema enum keeps 'agentic' for exactly this reason) even though
        write_champion_pointer itself refuses to produce one."""
        legacy = {
            "schema_version": 1, "champion": "agentic",
            "promoted_at": "2026-07-01T00:00:00Z",
            "promotion_source": "operator_bootstrap",
        }
        self._validate(POINTER_SCHEMA_PATH, legacy)

    def test_promoted_audit_conforms(self):
        arm_scores = {
            "scores": {"scanner_predictor_direct": 0.01, "thinktank_coverage": 0.03},
            "unavailable_reasons": {},
        }
        gate_result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=False,
        )
        audit = build_champion_audit("2026-07-18", gate_result, freeze=False)
        self._validate(AUDIT_SCHEMA_PATH, audit)

    def test_no_contest_audit_conforms(self):
        arm_scores = {
            "scores": {"scanner_predictor_direct": 0.01, "thinktank_coverage": None},
            "unavailable_reasons": {"thinktank_coverage": "thinktank_coverage_not_in_leaderboard"},
        }
        gate_result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=False,
        )
        audit = build_champion_audit("2026-07-18", gate_result, freeze=False)
        self._validate(AUDIT_SCHEMA_PATH, audit)

    def test_unchanged_winner_audit_conforms(self):
        arm_scores = {
            "scores": {"scanner_predictor_direct": 0.03, "thinktank_coverage": 0.01},
            "unavailable_reasons": {},
        }
        gate_result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=False,
        )
        audit = build_champion_audit("2026-07-18", gate_result, freeze=False)
        self._validate(AUDIT_SCHEMA_PATH, audit)

    def test_error_audit_conforms(self):
        audit = build_champion_audit("2026-07-18", None, freeze=False, error="boom")
        self._validate(AUDIT_SCHEMA_PATH, audit)

    def test_legacy_v1_audit_shape_still_schema_valid_by_git_history(self):
        """v1 historical records are NOT expected to validate against the
        v2 schema (schema_version const changed, fields renamed) -- this is
        by design (module/schema docstrings): v1 documents remain valid
        under the FROZEN v1 shape recoverable via git history, and this
        repo's tests only ever validate newly-built (v2) records. This test
        simply pins that expectation so a future reader doesn't mistake the
        absence of v1 conformance testing for an oversight."""
        legacy_v1 = {
            "schema_version": 1, "date": "2026-07-13", "outcome": "promoted",
            "champion_before": "agentic", "champion_after": "scanner_predictor_direct",
            "challenger_matured_cohorts": 0, "sn_lift_vs_champion": None,
            "consecutive_wins": 0, "cooldown_until": "2026-07-27", "blocked_by": None,
        }
        jsonschema = pytest.importorskip("jsonschema", reason="jsonschema not installed")
        schema = json.loads(AUDIT_SCHEMA_PATH.read_text())
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=legacy_v1, schema=schema)

    def test_valid_champions_subset_of_schema_enum(self):
        """The schema enum is a SUPERSET of VALID_CHAMPIONS (it additionally
        read-tolerates the retired 'agentic' seat) -- not an exact-set
        match like the pre-I2518 engine had, by design."""
        schema = json.loads(POINTER_SCHEMA_PATH.read_text())
        enum = set(schema["properties"]["champion"]["enum"])
        assert set(VALID_CHAMPIONS).issubset(enum)
        assert "agentic" in enum
