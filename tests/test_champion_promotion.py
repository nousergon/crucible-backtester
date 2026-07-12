"""champion_promotion — gated weekly champion promotion/demotion engine
(config#2364 / config#2367).

Pins: (1) the HAC/Newey-West-adjusted significance gate inflates the
standard error vs a naive i.i.d. assumption for an overlapping series and
reduces to the naive case as overlap -> 0; (2) all five gates (matured
cohorts, overlap-aware significance, 2-week hysteresis, bidirectional
symmetry, cooldown) individually block a pointer move; (3) a synthetic
leaderboard fixture drives promote -> cooldown-hold -> demote transitions;
(4) --freeze suppresses the pointer write but the audit record is still
written every week (the liveness proxy, config#2054); (5) frozen-schema
conformance against contracts/producer_champion.schema.json and
contracts/producer_champion_audit.schema.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from optimizer.champion_promotion import (
    OUTCOMES,
    VALID_CHAMPIONS,
    build_champion_audit,
    build_leaderboard_artifact,
    evaluate_gates,
    hac_significance,
    leaderboard_entry_from_e2e_lift,
    leaderboard_gate_inputs,
    read_champion_pointer,
    read_prior_leaderboard_history,
    run_weekly_evaluation,
    write_champion_pointer,
    write_leaderboard,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
POINTER_SCHEMA_PATH = REPO_ROOT / "contracts" / "producer_champion.schema.json"
AUDIT_SCHEMA_PATH = REPO_ROOT / "contracts" / "producer_champion_audit.schema.json"


# ── HAC/Newey-West significance gate ────────────────────────────────────────


class TestHacSignificance:
    def test_insufficient_data_below_two_points(self):
        assert hac_significance([])["status"] == "insufficient_data"
        assert hac_significance([0.01])["status"] == "insufficient_data"

    def test_hac_se_inflates_vs_naive_for_overlapping_series(self):
        """A 3-period moving-average series (mirrors the 21d/7d ~= 3x
        overlap the issue describes) must get a LARGER standard error under
        the HAC/Newey-West adjustment than the naive i.i.d. s/sqrt(n) SE —
        this is the whole point of the gate (naive pooling manufactures
        false significance on overlapping windows)."""
        import random
        rng = random.Random(42)
        noise = [rng.gauss(0.005, 0.02) for _ in range(60)]
        overlapping = [
            sum(noise[max(0, i - 2):i + 1]) / len(noise[max(0, i - 2):i + 1])
            for i in range(len(noise))
        ]
        n = len(overlapping)
        mean = sum(overlapping) / n
        naive_se = math.sqrt(sum((x - mean) ** 2 for x in overlapping) / n / n)

        result = hac_significance(overlapping)
        assert result["status"] == "ok"
        assert result["se"] > naive_se, (
            f"HAC SE ({result['se']}) must exceed naive SE ({naive_se}) for "
            "an autocorrelated/overlapping series"
        )
        # A meaningful inflation, not a rounding artifact.
        assert result["se"] / naive_se > 1.1

    def test_hac_reduces_to_naive_as_overlap_approaches_zero(self):
        """With lag=0 forced (the overlap -> 0 limit), the HAC SE must equal
        the naive i.i.d. SE exactly (Bartlett kernel with zero lags is just
        the sample variance of the mean)."""
        from nousergon_lib.quant.stats.intervals import newey_west_se
        import random
        rng = random.Random(7)
        series = [rng.gauss(0.0, 0.01) for _ in range(40)]
        n = len(series)
        mean = sum(series) / n
        naive_se = math.sqrt(sum((x - mean) ** 2 for x in series) / n / n)
        hac0 = newey_west_se(series, max_lags=0)
        assert hac0["se"] == pytest.approx(naive_se, rel=1e-9)

    def test_significant_positive_series_flagged(self):
        # Strongly, consistently positive with tight dispersion -> significant.
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
        # Default horizon=21, cadence=7 -> lag=3, verified via the public
        # hac_significance return (lags echoed from newey_west_se).
        series = [0.01] * 10
        result = hac_significance(series)
        assert result["lags"] == 3


# ── Gate engine (pure function, synthetic leaderboard fixtures) ────────────


class TestEvaluateGates:
    def _base_kwargs(self, **overrides):
        kwargs = dict(
            champion_before="agentic",
            challenger_matured_cohorts=6,
            challenger_weekly_sn_lift=[0.02, 0.021, 0.019, 0.022, 0.02, 0.018],
            prior_consecutive_wins=0,
            cooldown_until=None,
            as_of="2026-07-18",
            freeze=False,
        )
        kwargs.update(overrides)
        return kwargs

    def test_held_insufficient_data_below_cohort_floor(self):
        result = evaluate_gates(**self._base_kwargs(
            challenger_matured_cohorts=2,
            challenger_weekly_sn_lift=[0.02, 0.021],
        ))
        assert result["outcome"] == "held_insufficient_data"
        assert result["blocked_by"] == ["insufficient_matured_cohorts"]
        assert result["champion_after"] == "agentic"
        assert result["consecutive_wins"] == 0

    def test_held_not_significant(self):
        result = evaluate_gates(**self._base_kwargs(
            challenger_weekly_sn_lift=[0.01, -0.015, 0.008, -0.006, 0.002, -0.009],
        ))
        assert result["outcome"] == "held_not_significant"
        assert result["blocked_by"] == ["not_significant_hac_adjusted"]
        assert result["champion_after"] == "agentic"
        assert result["consecutive_wins"] == 0

    def test_first_winning_week_holds_for_hysteresis(self):
        """Gate clears significance on week 1 but hysteresis requires 2
        consecutive weeks — must hold, not promote."""
        result = evaluate_gates(**self._base_kwargs(prior_consecutive_wins=0))
        assert result["outcome"] == "held_not_significant"
        assert result["blocked_by"] == ["hysteresis_not_satisfied"]
        assert result["consecutive_wins"] == 1
        assert result["champion_after"] == "agentic"

    def test_second_consecutive_winning_week_promotes(self):
        result = evaluate_gates(**self._base_kwargs(prior_consecutive_wins=1))
        assert result["outcome"] == "promoted"
        assert result["champion_after"] == "scanner_predictor_direct"
        assert result["consecutive_wins"] == 2
        assert result["blocked_by"] is None
        assert result["cooldown_until"] == "2026-08-01"  # +2 weeks from as_of

    def test_losing_week_resets_hysteresis_streak(self):
        result = evaluate_gates(**self._base_kwargs(
            prior_consecutive_wins=1,
            challenger_weekly_sn_lift=[0.01, -0.015, 0.008, -0.006, 0.002, -0.009],
        ))
        assert result["consecutive_wins"] == 0
        assert result["outcome"] == "held_not_significant"

    def test_held_cooldown_blocks_move_even_when_gates_clear(self):
        result = evaluate_gates(**self._base_kwargs(
            prior_consecutive_wins=1,
            cooldown_until="2026-07-25",  # in the future relative to as_of
        ))
        assert result["outcome"] == "held_cooldown"
        assert result["blocked_by"] == ["cooldown_active"]
        assert result["champion_after"] == "agentic"
        # cooldown_until is carried forward unchanged, not advanced.
        assert result["cooldown_until"] == "2026-07-25"

    def test_cooldown_expired_allows_move(self):
        result = evaluate_gates(**self._base_kwargs(
            prior_consecutive_wins=1,
            cooldown_until="2026-07-18",  # exactly as_of -> not < as_of -> expired
        ))
        assert result["outcome"] == "promoted"

    def test_bidirectional_demote_symmetric_to_promote(self):
        """Same rule set demotes: champion is currently
        scanner_predictor_direct, challenger (agentic) beats it 2 weeks
        running -> demote back to agentic."""
        result = evaluate_gates(**self._base_kwargs(
            champion_before="scanner_predictor_direct",
            prior_consecutive_wins=1,
        ))
        assert result["outcome"] == "demoted"
        assert result["challenger"] == "agentic"
        assert result["champion_after"] == "agentic"

    def test_freeze_suppresses_pointer_move_but_reports_would_be_outcome(self):
        result = evaluate_gates(**self._base_kwargs(prior_consecutive_wins=1, freeze=True))
        assert result["outcome"] == "promoted"
        assert result["blocked_by"] == ["frozen"]
        # champion_after must NOT advance under freeze — idempotent/safe.
        assert result["champion_after"] == "agentic"
        assert result["cooldown_until"] is None


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

    def test_operator_bootstrap_source_uses_same_writer(self):
        """The bootstrap script's contract: same writer function, different
        promotion_source — never a parallel writer implementation."""
        s3 = MagicMock()
        pointer = write_champion_pointer(
            "bucket", "agentic",
            promotion_source="operator_bootstrap", upload=True, s3_client=s3,
        )
        assert pointer["promotion_source"] == "operator_bootstrap"
        s3.put_object.assert_called_once()

    def test_upload_false_skips_s3(self):
        s3 = MagicMock()
        write_champion_pointer(
            "bucket", "agentic", promotion_source="gate_engine",
            upload=False, s3_client=s3,
        )
        s3.put_object.assert_not_called()

    def test_rejects_unknown_champion(self):
        with pytest.raises(ValueError):
            write_champion_pointer(
                "bucket", "not_a_real_arm", promotion_source="gate_engine",
                upload=False,
            )


class TestReadChampionPointer:
    def test_missing_key_returns_none(self):
        from botocore.exceptions import ClientError
        s3 = MagicMock()
        s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject",
        )
        assert read_champion_pointer("bucket", s3_client=s3) is None

    def test_reads_existing_pointer(self):
        s3 = MagicMock()
        body = json.dumps({"schema_version": 1, "champion": "agentic",
                            "promoted_at": "2026-07-01T00:00:00Z",
                            "promotion_source": "operator_bootstrap"}).encode()
        s3.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=body))}
        pointer = read_champion_pointer("bucket", s3_client=s3)
        assert pointer["champion"] == "agentic"


# ── Leaderboard artifact (research/producer_leaderboard/{date}.json) ───────


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


class TestLeaderboard:
    def _e2e_lift_ok(self, sn_lift=0.02, n_cycles=6):
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

    def test_entry_extraction_from_e2e_lift(self):
        entry = leaderboard_entry_from_e2e_lift(self._e2e_lift_ok(sn_lift=0.017))
        assert entry["sn_lift_vs_agentic_cio"] == 0.017
        assert entry["n_cycles"] == 6

    def test_entry_extraction_handles_missing_or_skipped(self):
        assert leaderboard_entry_from_e2e_lift(None) is None
        assert leaderboard_entry_from_e2e_lift({}) is None
        skipped = {"scanner_then_predictor_counterfactual": {"status": "skipped", "reason": "x"}}
        assert leaderboard_entry_from_e2e_lift(skipped) is None

    def test_build_leaderboard_appends_and_dedupes_by_date(self):
        history = [{"date": "2026-07-04", "sn_lift_vs_agentic_cio": 0.01}]
        entry = leaderboard_entry_from_e2e_lift(self._e2e_lift_ok(sn_lift=0.02))
        artifact = build_leaderboard_artifact("2026-07-11", history, entry)
        dates = [p["date"] for p in artifact["weekly_points"]]
        assert dates == ["2026-07-04", "2026-07-11"]

        # Re-running the SAME date replaces rather than duplicates.
        entry2 = leaderboard_entry_from_e2e_lift(self._e2e_lift_ok(sn_lift=0.03))
        artifact2 = build_leaderboard_artifact("2026-07-11", artifact["weekly_points"], entry2)
        dates2 = [p["date"] for p in artifact2["weekly_points"]]
        assert dates2 == ["2026-07-04", "2026-07-11"]
        assert artifact2["weekly_points"][-1]["sn_lift_vs_agentic_cio"] == 0.03

    def test_history_scan_anchors_on_run_date_not_wall_clock(self):
        """A --date backfill run (run_date far from wall-clock today) must
        seed history relative to the BACKFILLED date, not the day the
        backfill script happens to execute — otherwise a backfill can never
        find its own prior week's leaderboard artifact."""
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
                {"date": "2026-07-11", "sn_lift_vs_agentic_cio": None},  # immature/missing
            ],
        }
        gi = leaderboard_gate_inputs(artifact)
        assert gi["challenger_matured_cohorts"] == 2
        assert gi["challenger_weekly_sn_lift"] == [0.01, 0.02]


# ── Weekly audit record (config/apply_audit/producer_champion/{date}.json) ─


class TestBuildChampionAudit:
    def test_error_path_conforms_and_reports_unavailable_leaderboard(self):
        audit = build_champion_audit("2026-07-18", None, freeze=False, error="leaderboard missing")
        assert audit["outcome"] == "error"
        assert audit["blocked_by"] == ["leaderboard_unavailable"]
        assert audit["champion_before"] is None

    def test_held_path_records_zero_pointer_movement(self):
        gate_result = evaluate_gates(
            champion_before="agentic",
            challenger_matured_cohorts=2,
            challenger_weekly_sn_lift=[0.01, 0.02],
            prior_consecutive_wins=0,
            cooldown_until=None,
            as_of="2026-07-18",
            freeze=False,
        )
        audit = build_champion_audit("2026-07-18", gate_result, freeze=False)
        assert audit["outcome"] == "held_insufficient_data"
        assert audit["champion_before"] == audit["champion_after"] == "agentic"

    @pytest.mark.parametrize("outcome", OUTCOMES)
    def test_all_outcomes_are_in_frozen_vocabulary(self, outcome):
        assert outcome in (
            "promoted", "demoted", "held_insufficient_data", "held_cooldown",
            "held_not_significant", "error",
        )


# ── Synthetic-leaderboard end-to-end transitions ────────────────────────────
# Drives the closes-when scenario directly: a synthetic leaderboard history
# takes the engine through promote -> cooldown-hold -> demote across
# successive weekly runs, using run_weekly_evaluation's public surface with
# an in-memory MagicMock S3 standing in for state carry-forward.


class TestSyntheticLeaderboardTransitions:
    BUCKET = "test-bucket"

    def _run(self, s3, run_date, weekly_sn_lift, matured_cohorts, freeze=False):
        leaderboard = {
            "challenger_matured_cohorts": matured_cohorts,
            "challenger_weekly_sn_lift": weekly_sn_lift,
        }
        return run_weekly_evaluation(
            bucket=self.BUCKET, run_date=run_date, leaderboard=leaderboard,
            freeze=freeze, upload=True, s3_client=s3,
        )

    def test_promote_then_cooldown_hold_then_demote(self):
        s3 = _FakeS3()
        winning = [0.02, 0.021, 0.019, 0.022, 0.02, 0.018]

        # Week 1: significant win, but hysteresis requires 2 consecutive weeks.
        r1 = self._run(s3, "2026-07-11", winning, 6)
        assert r1["outcome"] == "held_not_significant"
        assert r1["blocked_by"] == ["hysteresis_not_satisfied"]
        assert "_pointer_write" not in r1

        # Week 2: second consecutive win -> promotes, sets cooldown +2 weeks.
        r2 = self._run(s3, "2026-07-18", winning, 6)
        assert r2["outcome"] == "promoted"
        assert r2["champion_after"] == "scanner_predictor_direct"
        assert r2["cooldown_until"] == "2026-08-01"
        assert s3.store[f"{self.BUCKET}/config/producer_champion.json"]
        pointer = json.loads(s3.store[f"{self.BUCKET}/config/producer_champion.json"])
        assert pointer["champion"] == "scanner_predictor_direct"
        assert pointer["promotion_source"] == "gate_engine"

        # Week 3: champion is now scanner_predictor_direct; challenger
        # (agentic) even if it were winning cannot move the pointer yet —
        # cooldown active until 2026-08-01.
        r3 = self._run(s3, "2026-07-25", [0.02, 0.02, 0.02, 0.02, 0.02, 0.02], 6)
        assert r3["outcome"] == "held_cooldown"
        assert r3["blocked_by"] == ["cooldown_active"]
        pointer_after_r3 = json.loads(s3.store[f"{self.BUCKET}/config/producer_champion.json"])
        assert pointer_after_r3["champion"] == "scanner_predictor_direct"  # untouched

        # Week 4: cooldown has expired (as_of >= cooldown_until) and the
        # challenger (agentic) now wins 2 consecutive weeks -> demote.
        # consecutive_wins carries from r3 (which held on cooldown, not
        # significance, so the streak was still extended to 1 there)... but
        # r3 already logged a winning week; this 4th call is the 2nd
        # consecutive win post-cooldown-check ordering, so it promotes/demotes.
        r4 = self._run(s3, "2026-08-01", [0.02, 0.02, 0.02, 0.02, 0.02, 0.02], 6)
        assert r4["outcome"] == "demoted"
        assert r4["champion_after"] == "agentic"
        pointer_after_r4 = json.loads(s3.store[f"{self.BUCKET}/config/producer_champion.json"])
        assert pointer_after_r4["champion"] == "agentic"

    def test_held_not_significant_transition(self):
        s3 = _FakeS3()
        noisy = [0.01, -0.015, 0.008, -0.006, 0.002, -0.009]
        r = self._run(s3, "2026-07-11", noisy, 6)
        assert r["outcome"] == "held_not_significant"
        assert r["blocked_by"] == ["not_significant_hac_adjusted"]
        assert f"{self.BUCKET}/config/producer_champion.json" not in s3.store

    def test_held_insufficient_data_transition(self):
        s3 = _FakeS3()
        r = self._run(s3, "2026-07-11", [0.02, 0.021], 2)
        assert r["outcome"] == "held_insufficient_data"
        assert r["blocked_by"] == ["insufficient_matured_cohorts"]
        assert f"{self.BUCKET}/config/producer_champion.json" not in s3.store

    def test_freeze_suppresses_pointer_write_but_audit_always_written(self):
        s3 = _FakeS3()
        winning = [0.02, 0.021, 0.019, 0.022, 0.02, 0.018]
        self._run(s3, "2026-07-11", winning, 6)  # week 1: builds hysteresis
        r2 = self._run(s3, "2026-07-18", winning, 6, freeze=True)
        assert r2["outcome"] == "promoted"
        assert r2["blocked_by"] == ["frozen"]
        # Pointer must NOT be written under freeze.
        assert f"{self.BUCKET}/config/producer_champion.json" not in s3.store
        # But the weekly audit record IS written every week (liveness proxy).
        assert f"{self.BUCKET}/config/apply_audit/producer_champion/2026-07-11.json" in s3.store
        assert f"{self.BUCKET}/config/apply_audit/producer_champion/2026-07-18.json" in s3.store
        assert f"{self.BUCKET}/config/apply_audit/producer_champion/latest.json" in s3.store

    def test_leaderboard_unavailable_is_error_but_still_audited(self):
        s3 = _FakeS3()
        result = run_weekly_evaluation(
            bucket=self.BUCKET, run_date="2026-07-11", leaderboard=None,
            freeze=False, upload=True, s3_client=s3,
        )
        assert result["outcome"] == "error"
        assert f"{self.BUCKET}/config/apply_audit/producer_champion/2026-07-11.json" in s3.store
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

    def test_bootstrap_pointer_conforms(self):
        pointer = write_champion_pointer(
            "bucket", "agentic", promotion_source="operator_bootstrap", upload=False,
        )
        self._validate(POINTER_SCHEMA_PATH, pointer)

    def test_promoted_audit_conforms(self):
        gate_result = evaluate_gates(
            champion_before="agentic",
            challenger_matured_cohorts=6,
            challenger_weekly_sn_lift=[0.02, 0.021, 0.019, 0.022, 0.02, 0.018],
            prior_consecutive_wins=1,
            cooldown_until=None,
            as_of="2026-07-18",
            freeze=False,
        )
        audit = build_champion_audit("2026-07-18", gate_result, freeze=False)
        self._validate(AUDIT_SCHEMA_PATH, audit)

    def test_held_audit_conforms(self):
        gate_result = evaluate_gates(
            champion_before="agentic",
            challenger_matured_cohorts=2,
            challenger_weekly_sn_lift=[0.02, 0.021],
            prior_consecutive_wins=0,
            cooldown_until=None,
            as_of="2026-07-18",
            freeze=False,
        )
        audit = build_champion_audit("2026-07-18", gate_result, freeze=False)
        self._validate(AUDIT_SCHEMA_PATH, audit)

    def test_error_audit_conforms(self):
        audit = build_champion_audit("2026-07-18", None, freeze=False, error="boom")
        self._validate(AUDIT_SCHEMA_PATH, audit)

    def test_valid_champions_match_schema_enum(self):
        schema = json.loads(POINTER_SCHEMA_PATH.read_text())
        assert set(schema["properties"]["champion"]["enum"]) == set(VALID_CHAMPIONS)
