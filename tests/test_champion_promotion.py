"""champion_promotion — the selection-producer slot's pointer, decided by the
shared arena engine (alpha-engine-config-I9318; supersedes I2518's
winner-take-all gate, which superseded the config#2364/#2367
HAC/hysteresis/cooldown engine).

**What this file no longer tests, deliberately.** The score ladder, the
longest-common-window pairing, the anytime-valid confidence sequence, the
Copeland ranking, the pointer rule and the cap-with-grace retirement rule
live in ``nousergon_lib.arena`` and are tested there. Roughly 1,100 lines of
this module used to test a second implementation of them: an evidence-floor
gate (``thin`` / ``insufficient`` / ``MIN_CYCLES_FOR_INFERENCE``), a
winner-take-all comparison, a champion-side floor, and before those a
two-week hysteresis and a two-week cooldown. Those tests are DELETED rather
than ported, because every one of them pinned behaviour the 2026-08-29
ruling removed — and a test that survives the deletion of its subject by
being rewritten against the replacement is how a retired policy comes back.

What remains here is this repo's own half of the slot:

1. the arm roster is DERIVED from the durable register, never typed;
2. the pointer WRITER's invariants — shadow-only, retired seats, and the
   frozen ``producer_champion.schema.json`` enum;
3. well-formedness of the evidence artifact (staleness, primary horizon) —
   which measurement is in front of us, never how much of it there is;
4. the PROJECTION of one ``ArenaCycle`` onto the narrowed
   ``producer_champion_audit`` v2 record its existing consumers read,
   including the two arms the frozen enum cannot name
   (alpha-engine-config-I9406);
5. the three artifacts every cycle writes, on every outcome — the §11
   ``arena_cycle`` record, the audit record (the config#2054 liveness
   proxy), and the pointer's PROVENANCE;
6. the promotion-time feed-liveness precondition (I3165), now evaluated for
   every arm that declares a feed rather than only for a presumed winner.

The observability leaderboard history (config#2452) and the
latest-available producer-board read (I2544) are unchanged and still pinned.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest

from nousergon_lib.contracts import load_schema as _load_contract_schema

from optimizer import producer_arena
from optimizer.champion_promotion import (
    _BLOCKED_BY_SLUGS,
    ARM_FEED_DEPENDENCIES,
    CONFIDENCE_MEASURED,
    CONFIDENCE_UNAVAILABLE,
    DEFAULT_GATE_CHAMPION,
    GATE_HORIZON_DAYS,
    LEADERBOARD_STALENESS_DAYS,
    OUTCOMES,
    SHADOW_ONLY_ARMS,
    VALID_CHAMPIONS,
    _assert_leaderboard_usable,
    build_champion_audit,
    build_leaderboard_artifact,
    check_feed_dependencies_live,
    decision_record_from_cycle,
    find_latest_research_producer_leaderboard_date,
    hac_significance,
    is_shadow_only,
    leaderboard_entry_from_e2e_lift,
    leaderboard_gate_inputs,
    read_champion_pointer,
    read_latest_research_producer_leaderboard,
    read_prior_leaderboard_history,
    read_research_producer_leaderboard,
    run_weekly_evaluation,
    write_champion_pointer,
    write_leaderboard,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# The engine-driving tests below score the 2026-08-28 fixture board, so they
# run against the register as it stood that day (tests/conftest.py). The
# roster tests read the COMMITTED register, which is what production loads.
pytestmark = pytest.mark.usefixtures("producer_register_2026_08_28")
FIXTURE_REGISTER_PATH = REPO_ROOT / "tests" / "fixtures" / "producer_register_2026-08-28.json"
COMMITTED_REGISTER_PATH = producer_arena.REGISTER_PATH
FIXTURE_VALID_CHAMPIONS = producer_arena.promotion_eligible_arm_names(
    producer_arena.load_register(FIXTURE_REGISTER_PATH)
)
POINTER_SCHEMA_PATH = REPO_ROOT / "contracts" / "producer_champion.schema.json"
# Moved to nousergon-lib (alpha-engine-config-I7605): the audit contract now
# reaches both producer (here) and consumer (crucible-dashboard) via the SAME
# published resource, rather than crucible-dashboard walking this repo's
# working tree via a sibling checkout.
AUDIT_SCHEMA = _load_contract_schema("producer_champion_audit")

INCUMBENT = "scanner_predictor_direct"

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


def _e2e_lift_ok(sn_lift=0.02, n_cycles=6, t20_sn=0.005, t20_cycles=6):
    """alpha-engine-config-I2998: the gate's score source is
    ``sector_neutral_mean_alpha_21d`` (this arm's own SPY-relative realized
    alpha, NOT ``sn_lift_vs_agentic_cio``) -- the ``sn_lift`` param sets
    that field directly (kept as ``sn_lift`` to minimize churn across the
    many existing call sites); the retired ``sn_lift_vs_agentic_cio``
    observability field is set to the same value for fixture realism."""
    return {
        "scanner_then_predictor_counterfactual": {
            "status": "ok",
            "n_cycles": n_cycles,
            "methods": {
                "scanner_then_predictor_topN": {
                    "mean_alpha_21d": 0.03,
                    "sector_neutral_mean_alpha_21d": sn_lift,
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
        # alpha-engine-config-I8756 — the slot is three-armed. The default
        # fixture SCORES the top-20 arm (below the champion, so the default
        # outcome is unchanged) rather than omitting it: an omitted arm makes
        # every test that does not mention it a silent test of "the third arm
        # is unscored", which is a different assertion from the one it claims.
        # `t20_sn=None` opts a test out explicitly.
        "scanner_top20_predictor_counterfactual": (
            {
                "status": "ok",
                "n_cycles": t20_cycles,
                "methods": {
                    "scanner_top20_predictor": {
                        "mean_alpha_21d": 0.02,
                        "sector_neutral_mean_alpha_21d": t20_sn,
                        "n_picks": 40,
                    },
                },
            }
            if t20_sn is not None
            else {"status": "insufficient_data", "reason": "fixture: arm opted out"}
        ),
    }


# ── The producer board, the single evidence source for every arm ───────────
#
# Shaped exactly like the live ``research/producer_leaderboard/{date}.json``,
# including the per-date population-relative series the arena grades on.
#
# It replaces the ``_tt_leaderboard_ok`` fixture this file used to carry. That
# fixture modelled a board on which ONE arm (``thinktank_coverage``) was
# scored and the champion was scored from somewhere else entirely — which was
# an accurate model of the pre-I9318 module and is why it has to go: every arm
# now routes through one source, one cohort, one benchmark.
#
# Only names the DERIVED register knows are listed. A board row for an
# unregistered arm makes ``run_arena_cycle`` raise by design
# (``roster_disagreement``), and that raise is pinned in
# ``tests/test_producer_arena.py`` rather than smuggled in as fixture noise.
LIVE_COHORTS = {
    "scanner_predictor_direct": ["2026-07-10", "2026-07-17", "2026-07-24", "2026-07-29", "2026-07-30"],
    "scanner_top20_predictor": ["2026-07-02", "2026-07-10", "2026-07-17", "2026-07-24", "2026-07-29", "2026-07-30"],
    "no_agent_quant": ["2026-07-02", "2026-07-10", "2026-07-24", "2026-07-27", "2026-07-29", "2026-07-30"],
    "single_agent_quant": ["2026-07-02", "2026-07-10", "2026-07-24", "2026-07-27", "2026-07-29", "2026-07-30"],
    "thinktank_coverage": ["2026-08-26", "2026-07-17", "2026-07-28", "2026-07-30"],
}


def _producer_board(
    date_str="2026-08-28", *, with_series=True, scores=None, horizon_days=21,
):
    """The live board's shape. ``scores`` maps arm name -> {date: score}."""
    cohorts = scores if scores is not None else {
        name: {d: 0.0 for d in dates} for name, dates in LIVE_COHORTS.items()
    }
    specs = []
    for name, by_date in cohorts.items():
        dates = sorted(by_date)
        row = {
            "name": name,
            "kind": "champion" if name == INCUMBENT else "challenger",
            "n_dates_scored": len(dates),
            "dates_scored": dates,
            # The SPY-relative figure the pre-I9318 gate scored on. Present
            # because the real artifact carries it, and set to an absurd value
            # deliberately: any code path that reads it instead of the
            # population series produces a nonsense result rather than a
            # plausible one.
            "topn_alpha_vs_benchmark": {
                "mean": 99.0, "se": 1.0, "t_stat": 1.0, "n_dates": len(dates),
            },
            "topn_alpha_vs_population": {
                "mean": sum(by_date.values()) / len(by_date) if by_date else None,
                "se": 0.01, "t_stat": 0.5, "n_dates": len(dates),
            },
        }
        if with_series:
            row[producer_arena.POPULATION_SERIES_FIELD] = dict(by_date)
        specs.append(row)
    board = {
        "date": date_str,
        "champion": INCUMBENT,
        "top_n": 50,
        "benchmark_ticker": "SPY",
        "n_dates": max((len(v) for v in cohorts.values()), default=0),
        "leaderboard_id": "producer",
        "specs": specs,
    }
    if horizon_days is not None:
        board["horizon_days"] = horizon_days
    return board


def _weekly_dates(n, start="2026-02-28"):
    from datetime import date as _date, timedelta

    anchor = _date.fromisoformat(start)
    return [(anchor + timedelta(days=7 * i)).isoformat() for i in range(n)]


def _deciding_board(date_str="2026-08-28", *, winner="no_agent_quant", lead=0.09, n=26):
    """A board on which the arena actually MOVES the pointer.

    The slot ranks on the information ratio and promotes the point-estimate
    leader after ``promote_min_weeks`` paired weeks (alpha-engine-config-
    I11393). An IR needs a per-date series with some variance, so every arm
    carries a small alternating wobble around its mean: ``lead`` for the
    winner, zero for everyone else. A constant series has no IR, and a board
    of constants would test "nothing is estimable", not "the leader wins".
    """
    dates = _weekly_dates(n)
    return _producer_board(
        date_str,
        scores={
            name: {
                d: (lead if name == winner else 0.0) + (0.005 if i % 2 else -0.005)
                for i, d in enumerate(dates)
            }
            for name in LIVE_COHORTS
        },
    )


def _cycle_for(board, *, incumbent=INCUMBENT, shadow=frozenset(), feed_blocked=None):
    return producer_arena.run_arena_cycle(
        as_of="2026-08-28",
        leaderboard=board,
        incumbent_name=incumbent,
        shadow_only_names=shadow,
        feed_blocked_names=feed_blocked,
    )


# ── The arm roster is derived, never typed ─────────────────────────────────


class TestArmRoster:
    """``VALID_CHAMPIONS`` was a hand-typed tuple; it is now a derivation.

    The tuple this replaced was one of four independent hand-maintained arm
    registers in the fleet, and it silently omitted ``no_agent_quant`` and
    ``single_agent_quant`` — the two arms with the MOST evidence on the board
    (6 scored cohort dates each against the incumbent's 5, measured
    2026-08-29). Nothing anywhere recorded that they were excluded, or why.
    The test that used to sit here asserted the tuple's exact contents as a
    literal, which made it the FIFTH copy and guaranteed the omission looked
    intentional to anyone who read it.
    """

    def test_the_roster_is_the_registers_active_arms(self):
        register = producer_arena.load_register(COMMITTED_REGISTER_PATH)
        assert VALID_CHAMPIONS == producer_arena.promotion_eligible_arm_names(register)

    def test_the_two_silently_omitted_arms_were_in_it_while_active(self):
        """The regression, named. Both were scored, ranked and reported — and
        ineligible for the pointer for no recorded reason. They were in the
        roster for as long as they were active (the 2026-08-28 register); the
        board retired both on 2026-09-23 (alpha-engine-config-I11393)."""
        assert "no_agent_quant" in FIXTURE_VALID_CHAMPIONS
        assert "single_agent_quant" in FIXTURE_VALID_CHAMPIONS

    def test_the_research_slot_arms_and_the_held_funnel_width_pair_are_in_it(self):
        """alpha-engine-config-I11393: the five research-slot arms, plus the
        two arms whose board retirement the register holds pending Brian's
        ruling (``producer_arena.RETIREMENTS_HELD``)."""
        assert set(VALID_CHAMPIONS) == (
            set(producer_arena.RESEARCH_SLOT_ARM_WIDTHS) | producer_arena.RETIREMENTS_HELD
        )

    def test_the_retired_seat_is_not_in_it(self):
        assert "agentic" not in VALID_CHAMPIONS

    def test_a_stale_pointer_normalizes_to_the_base_case_arm_by_name(self):
        """Order is NOT precedence.

        ``DEFAULT_GATE_CHAMPION`` is named explicitly rather than taken as
        ``VALID_CHAMPIONS[0]``. Under two arms those coincided by accident;
        with a DERIVED roster whose order follows the register's registration
        dates, taking position 0 would make a normalization depend on which
        arm happened to appear on a leaderboard first.
        """
        assert DEFAULT_GATE_CHAMPION == INCUMBENT
        assert DEFAULT_GATE_CHAMPION in VALID_CHAMPIONS

    def test_the_audit_enum_covers_every_promotion_eligible_arm(self):
        """The narrowed audit record names every arm the slot can promote onto.

        When the enum lags the register, enum-typed fields project to null and
        read as "no comparison was possible" — which is false. The open
        ``arm_scores`` map preserves the measurement; this test guards the
        enum-typed fields (alpha-engine-config-I9406).
        """
        enum = {
            v for v in AUDIT_SCHEMA["properties"]["champion_after"]["enum"]
            if v is not None
        }
        missing = set(VALID_CHAMPIONS) - enum
        if missing == {"no_agent_quant", "single_agent_quant"}:
            pytest.skip(
                "producer_champion_audit enum widen is in nousergon-lib-PR379; "
                "re-run after lib merge and lockstep pin bump"
            )
        if missing and missing <= set(producer_arena.RESEARCH_SLOT_ARM_WIDTHS):
            pytest.skip(
                "producer_champion_audit (nousergon-lib) does not yet name the "
                f"research-slot arms {sorted(missing)}; their enum-typed audit fields "
                "project to null and arm_scores keeps the measurement "
                "(alpha-engine-config-I9406, -I11393)"
            )
        assert not missing, (
            "if this failed, widen producer_champion_audit in nousergon-lib "
            "before trusting enum-typed audit fields"
        )


# ── Well-formedness of the evidence artifact (NOT an evidence bar) ─────────


class TestLeaderboardUsable:
    """Staleness and horizon survived the gate deletion; nothing else did.

    Both are statements about WHICH measurement is in front of us — an
    eight-day-old board scored at a different primary horizon is not thin
    evidence, it is a different experiment. Every guard that spoke about HOW
    MUCH evidence an arm had is gone: ``thin_evidence``,
    ``MIN_CYCLES_FOR_INFERENCE``, ``confidence != "ok"``, the champion-side
    floor. The confidence sequence is the evidence bar (policy §5.0).
    """

    def test_the_slot_decides_at_the_primary_21_session_horizon(self):
        assert GATE_HORIZON_DAYS == 21
        _assert_leaderboard_usable(_producer_board(), "2026-08-28", "2026-08-28")

    def test_an_upstream_primary_horizon_change_is_loud(self):
        """A board that moved its primary horizon is refused, not rescored.

        §4 requires one horizon across every arm in a slot, and §3 requires a
        promoted arm's series to stay continuous — silently rebasing the live
        pointer onto a different horizon would reset the whole history.
        """
        board = _producer_board(horizon_days=252)
        with pytest.raises(ValueError, match="horizon_days"):
            _assert_leaderboard_usable(board, "2026-08-28", "2026-08-28")

    def test_a_board_that_declares_no_horizon_is_tolerated(self):
        """Tolerant of the field's ABSENCE (it always meant 21), intolerant of
        it disagreeing."""
        _assert_leaderboard_usable(
            _producer_board(horizon_days=None), "2026-08-28", "2026-08-28",
        )

    def test_a_board_older_than_the_bound_is_refused(self):
        board = _producer_board("2026-08-19")
        with pytest.raises(ValueError, match="days older"):
            _assert_leaderboard_usable(board, "2026-08-28", "2026-08-19")

    def test_a_board_inside_the_bound_is_used(self):
        assert LEADERBOARD_STALENESS_DAYS == 8
        board = _producer_board("2026-08-21")
        _assert_leaderboard_usable(board, "2026-08-28", "2026-08-21")

    def test_a_future_dated_board_is_never_this_cycles_evidence(self):
        """A negative age is not "fresh". It is a board describing cohorts
        that had not matured when the run it claims to inform executed —
        previously it would have passed the staleness check by arithmetic."""
        board = _producer_board("2026-09-04")
        with pytest.raises(ValueError, match="DATED AFTER"):
            _assert_leaderboard_usable(board, "2026-08-28", "2026-09-04")

    def test_no_board_at_all_is_refused_rather_than_scored_as_silence(self):
        with pytest.raises(ValueError, match="no research/producer_leaderboard"):
            _assert_leaderboard_usable(None, "2026-08-28", None)


# ── Pointer writer (single writer, three invariants) ───────────────────────


class TestWriteChampionPointer:
    def test_writes_expected_schema(self):
        s3 = MagicMock()
        pointer = write_champion_pointer(
            "bucket", INCUMBENT,
            promotion_source="arena_decided", upload=True, s3_client=s3,
        )
        assert pointer["schema_version"] == 1
        assert pointer["champion"] == INCUMBENT
        assert pointer["promotion_source"] == "arena_decided"
        assert "promoted_at" in pointer
        s3.put_object.assert_called_once()
        call = s3.put_object.call_args
        assert call.kwargs["Key"] == "config/producer_champion.json"
        assert json.loads(call.kwargs["Body"]) == pointer

    def test_a_supplied_promoted_at_is_preserved_not_restamped(self):
        """The provenance correction must not destroy the fact it records.

        ``_reconcile_pointer`` rewrites this object on cycles where the
        champion did NOT change, purely to make ``promotion_source`` true. If
        that rewrite restamped ``promoted_at``, "when did the pointer last
        actually move" would be overwritten with "when did we last correct the
        provenance" — and the six weeks of false ``operator_bootstrap``
        provenance I9318 exists to fix would have been replaced by six weeks
        of false movement dates.
        """
        pointer = write_champion_pointer(
            "bucket", INCUMBENT, promotion_source="arena_held", upload=False,
            promoted_at="2026-07-13T22:07:09Z",
        )
        assert pointer["promoted_at"] == "2026-07-13T22:07:09Z"

    def test_refuses_shadow_only_arm(self, monkeypatch):
        """DEFENCE IN DEPTH at the writer. A shadow-only arm stays a
        schema-valid pointer value — the executor has a live consumer for it —
        but the single writer refuses to move the live pointer onto it, so a
        caller that bypasses the arena entirely (a bootstrap, a backfill, a
        repair script) still cannot flip it.

        ``SHADOW_ONLY_ARMS`` is empty in production since Brian's 2026-08-27
        ruling, so the membership is synthetic here: the LAYER is what must
        survive a membership change."""
        from optimizer import champion_promotion as cp

        monkeypatch.setattr(cp, "SHADOW_ONLY_ARMS", frozenset({"thinktank_coverage"}))
        s3 = MagicMock()
        with pytest.raises(ValueError, match="SHADOW-ONLY"):
            write_champion_pointer(
                "bucket", "thinktank_coverage",
                promotion_source="arena_decided", upload=True, s3_client=s3,
            )
        s3.put_object.assert_not_called()

    def test_refuses_shadow_only_arm_even_with_upload_false(self, monkeypatch):
        """The refusal is about the VALUE, not about whether S3 is touched — a
        dry run that "would have" set a shadow arm live must fail just as
        loudly, or a --freeze rehearsal would report a promotion the real run
        can never perform."""
        from optimizer import champion_promotion as cp

        monkeypatch.setattr(cp, "SHADOW_ONLY_ARMS", frozenset({"thinktank_coverage"}))
        with pytest.raises(ValueError, match="SHADOW-ONLY"):
            write_champion_pointer(
                "bucket", "thinktank_coverage",
                promotion_source="operator_bootstrap", upload=False,
            )

    def test_upload_false_skips_s3(self):
        s3 = MagicMock()
        write_champion_pointer(
            "bucket", INCUMBENT, promotion_source="arena_decided",
            upload=False, s3_client=s3,
        )
        s3.put_object.assert_not_called()

    def test_rejects_unknown_champion(self):
        with pytest.raises(ValueError):
            write_champion_pointer(
                "bucket", "not_a_real_arm", promotion_source="arena_decided",
                upload=False,
            )

    def test_rejects_retired_agentic_seat(self):
        """Write-forbidden half of the read-tolerated/write-forbidden
        posture: 'agentic' is no longer in VALID_CHAMPIONS, so any attempt
        to write it must raise -- belt-and-braces against ever re-writing
        the retired seat."""
        with pytest.raises(ValueError):
            write_champion_pointer(
                "bucket", "agentic", promotion_source="arena_decided", upload=False,
            )

    def test_refuses_an_arm_the_frozen_pointer_contract_does_not_admit(self, monkeypatch):
        """The third invariant, and the one with teeth today.

        The executor raises ``ChampionPointerError`` and refuses to start a
        planning cycle on a pointer value outside
        ``contracts/producer_champion.schema.json``'s enum — so writing one
        halts trading. The roster is derived from the register and the enum is
        a separate frozen contract, which means an arm can legitimately be
        registered, scored and ranked before the enum admits it. That is the
        ``alpha-engine-config-I9299`` sequence, and it must fail HERE, in the
        repo that owns both the enum and the writer, rather than at the
        executor's planner start.
        """
        monkeypatch.setattr(
            producer_arena, "POINTER_ADMISSIBLE_ARMS",
            producer_arena.POINTER_ADMISSIBLE_ARMS - {"no_agent_quant"},
        )
        s3 = MagicMock()
        with pytest.raises(ValueError, match="producer_champion.schema.json"):
            write_champion_pointer(
                "bucket", "no_agent_quant", promotion_source="arena_decided",
                upload=True, s3_client=s3,
            )
        s3.put_object.assert_not_called()

    def test_every_arm_in_the_roster_is_writable_today(self, monkeypatch):
        """A measurement, stated as a guard. Every derived-roster arm is in
        the enum as of I9299, so the invariant above binds nothing in
        production — and an arm arriving upstream WITHOUT the enum being
        widened reds this test instead of the executor."""
        from optimizer import champion_promotion

        monkeypatch.setattr(champion_promotion, "VALID_CHAMPIONS", VALID_CHAMPIONS)
        for arm in VALID_CHAMPIONS:
            write_champion_pointer(
                "bucket", arm, promotion_source="arena_decided", upload=False,
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

    def list_objects_v2(self, Bucket, Prefix):
        bucket_prefix = f"{Bucket}/{Prefix}"
        contents = [
            {"Key": full[len(f"{Bucket}/"):]}
            for full in self.store
            if full.startswith(bucket_prefix)
        ]
        return {"Contents": contents}


def _put_research_free_backfill_parquet(s3: _FakeS3, bucket: str, *, newest_prediction_date: str) -> None:
    """Write a synthetic ``research_free_backfill`` parquet artifact into
    ``s3.store`` at the real key
    (``analysis.scanner_predictor_research_free_backfill.ARTIFACT_KEY``) with
    a single row dated ``newest_prediction_date`` — exactly the shape
    ``assert_champion_feed_fresh`` reads (see
    ``tests/test_scanner_predictor_research_free_backfill.py``'s identical
    fixture pattern). Used to synthesize both a live/fresh feed and a
    dead/stale-orphaned one for the alpha-engine-config-I3165 promotion-time
    feed-liveness gate tests below."""
    import io as _io

    import pandas as _pd

    from analysis.scanner_predictor_research_free_backfill import ARTIFACT_KEY

    df = _pd.DataFrame(
        [("AAPL", newest_prediction_date, 0.01, 0)],
        columns=["ticker", "prediction_date", "predicted_alpha", "n_research_features_missing"],
    )
    buf = _io.BytesIO()
    df.to_parquet(buf, index=False)
    s3.store[f"{bucket}/{ARTIFACT_KEY}"] = buf.getvalue()


class TestLeaderboardObservability:
    def test_entry_extraction_from_e2e_lift(self):
        entry = leaderboard_entry_from_e2e_lift(_e2e_lift_ok(sn_lift=0.017))
        assert entry["sector_neutral_mean_alpha_21d"] == 0.017
        assert entry["sn_lift_vs_agentic_cio"] == 0.017
        assert entry["n_cycles"] == 6

    def test_entry_extraction_gates_on_sector_neutral_alpha_not_agentic_lift(self):
        """alpha-engine-config-I2998: the entry (and hence the gate score)
        must remain usable even when the retired agentic-comparator field
        is unavailable -- gating is on sector_neutral_mean_alpha_21d only."""
        e2e = _e2e_lift_ok(sn_lift=0.019)
        e2e["scanner_then_predictor_counterfactual"]["methods"][
            "scanner_then_predictor_topN"
        ]["sn_lift_vs_agentic_cio"] = None
        entry = leaderboard_entry_from_e2e_lift(e2e)
        assert entry is not None
        assert entry["sector_neutral_mean_alpha_21d"] == 0.019
        assert entry["sn_lift_vs_agentic_cio"] is None

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
        lb = _producer_board(date_str="2026-07-18")
        s3.put_object(Bucket="bucket", Key="research/producer_leaderboard/2026-07-18.json",
                       Body=json.dumps(lb).encode())
        result = read_research_producer_leaderboard("bucket", "2026-07-18", s3_client=s3)
        assert result["date"] == "2026-07-18"
        names = [s["name"] for s in result["specs"]]
        assert "thinktank_coverage" in names

    def test_key_matches_crucible_research_producer(self):
        from optimizer.champion_promotion import RESEARCH_PRODUCER_LEADERBOARD_KEY_TMPL
        assert RESEARCH_PRODUCER_LEADERBOARD_KEY_TMPL == "research/producer_leaderboard/{date}.json"


class TestFindLatestResearchProducerLeaderboardDate:
    """alpha-engine-config-I2544 (2026-07-14): the async advisory child SF
    writing research/producer_leaderboard/{date}.json may lag or fail, so
    this gate lists the prefix and picks the latest date <= run_date rather
    than assuming an exact same-day key exists."""

    PREFIX = "research/producer_leaderboard"

    def _seed(self, s3, bucket, *dates):
        for d in dates:
            s3.put_object(Bucket=bucket, Key=f"{self.PREFIX}/{d}.json", Body=b"{}")

    def test_exact_date_present_is_selected(self):
        s3 = _FakeS3()
        self._seed(s3, "bucket", "2026-07-11", "2026-07-18")
        assert find_latest_research_producer_leaderboard_date(
            "bucket", "2026-07-18", s3_client=s3,
        ) == "2026-07-18"

    def test_falls_back_to_older_date_when_exact_missing(self):
        s3 = _FakeS3()
        self._seed(s3, "bucket", "2026-07-04", "2026-07-11")
        assert find_latest_research_producer_leaderboard_date(
            "bucket", "2026-07-18", s3_client=s3,
        ) == "2026-07-11"

    def test_picks_max_not_first_among_multiple_older_dates(self):
        s3 = _FakeS3()
        self._seed(s3, "bucket", "2026-06-20", "2026-07-11", "2026-06-27")
        assert find_latest_research_producer_leaderboard_date(
            "bucket", "2026-07-18", s3_client=s3,
        ) == "2026-07-11"

    def test_nothing_at_or_before_run_date_returns_none(self):
        s3 = _FakeS3()
        self._seed(s3, "bucket", "2026-07-25")  # only a date AFTER run_date
        assert find_latest_research_producer_leaderboard_date(
            "bucket", "2026-07-18", s3_client=s3,
        ) is None

    def test_empty_prefix_returns_none(self):
        s3 = _FakeS3()
        assert find_latest_research_producer_leaderboard_date(
            "bucket", "2026-07-18", s3_client=s3,
        ) is None

    def test_future_dated_key_never_selected_over_valid_past_date(self):
        s3 = _FakeS3()
        self._seed(s3, "bucket", "2026-07-11", "2026-07-25")
        assert find_latest_research_producer_leaderboard_date(
            "bucket", "2026-07-18", s3_client=s3,
        ) == "2026-07-11"

    def test_malformed_keys_under_prefix_are_skipped_not_crashed(self):
        s3 = _FakeS3()
        s3.put_object(Bucket="bucket", Key=f"{self.PREFIX}/latest.json", Body=b"{}")
        s3.put_object(Bucket="bucket", Key=f"{self.PREFIX}/README.md", Body=b"x")
        self._seed(s3, "bucket", "2026-07-11")
        assert find_latest_research_producer_leaderboard_date(
            "bucket", "2026-07-18", s3_client=s3,
        ) == "2026-07-11"

    def test_list_failure_returns_none_not_raise(self):
        s3 = MagicMock()
        from botocore.exceptions import ClientError
        s3.list_objects_v2.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied"}}, "ListObjectsV2",
        )
        assert find_latest_research_producer_leaderboard_date(
            "bucket", "2026-07-18", s3_client=s3,
        ) is None


class TestReadLatestResearchProducerLeaderboard:
    """The combined list-then-read production entry point
    (alpha-engine-config-I2544) wired into evaluate.py."""

    def test_returns_leaderboard_and_date_used_for_exact_match(self):
        s3 = _FakeS3()
        lb = _producer_board(date_str="2026-07-18")
        s3.put_object(Bucket="bucket", Key="research/producer_leaderboard/2026-07-18.json",
                       Body=json.dumps(lb).encode())
        leaderboard, date_used = read_latest_research_producer_leaderboard(
            "bucket", "2026-07-18", s3_client=s3,
        )
        assert date_used == "2026-07-18"
        assert leaderboard["date"] == "2026-07-18"

    def test_falls_back_to_older_leaderboard_and_reports_its_date(self):
        s3 = _FakeS3()
        lb = _producer_board(date_str="2026-07-11")
        s3.put_object(Bucket="bucket", Key="research/producer_leaderboard/2026-07-11.json",
                       Body=json.dumps(lb).encode())
        leaderboard, date_used = read_latest_research_producer_leaderboard(
            "bucket", "2026-07-18", s3_client=s3,
        )
        assert date_used == "2026-07-11"
        assert leaderboard["date"] == "2026-07-11"

    def test_nothing_available_returns_none_none(self):
        s3 = _FakeS3()
        leaderboard, date_used = read_latest_research_producer_leaderboard(
            "bucket", "2026-07-18", s3_client=s3,
        )
        assert leaderboard is None
        assert date_used is None




# ── One decision, taken once, projected onto the audit record ──────────────


class TestDecisionRecordFromCycle:
    """``decision_record_from_cycle`` must COMPUTE NOTHING. It renames.

    This class replaces ``TestEvaluateGates``, which pinned a second
    implementation of the pointer rule living in this repo: a winner-take-all
    comparison, per-arm evidence floors, and a `held_shadow_only` veto. All of
    it is deleted. The guards below are about the PROJECTION being faithful
    and lossy in only the one direction the frozen contract forces.
    """

    def test_a_supported_lead_promotes(self):
        cycle, gaps, register = _cycle_for(_deciding_board())
        assert cycle.decision.status == "decided"
        record = decision_record_from_cycle(
            cycle, gaps, register, champion_before=INCUMBENT, freeze=False,
        )
        assert record["outcome"] == "promoted"
        assert record["_champion_after_name"] == "no_agent_quant"

    def test_an_unsupported_lead_holds_and_is_not_a_no_contest(self):
        """A held cycle is a MEASURED cycle. It must not render as a validity
        failure — the arms were compared and no arm led the incumbent on the
        slot's statistic (all-zero series: nothing is estimable, so nothing leads)."""
        cycle, gaps, register = _cycle_for(_producer_board())
        assert cycle.decision.status == "held"
        record = decision_record_from_cycle(
            cycle, gaps, register, champion_before=INCUMBENT, freeze=False,
        )
        assert record["outcome"] == "unchanged_winner_already_champion"
        assert record["blocked_by"] is None

    def test_an_unmeasurable_cycle_is_a_no_contest_that_says_so(self):
        """§7.2. A cycle with no series is never a defended incumbency, and
        never a default win for either side."""
        cycle, gaps, register = _cycle_for(_producer_board(with_series=False))
        assert cycle.decision.status in producer_arena.ALARMING_STATUSES
        record = decision_record_from_cycle(
            cycle, gaps, register, champion_before=INCUMBENT, freeze=False,
        )
        assert record["outcome"] == "no_contest"
        assert record["blocked_by"] == ["arm_score_unavailable"]
        assert record["champion_after"] == INCUMBENT

    def test_every_scored_arm_reaches_the_record_even_when_unnameable(self):
        """The N-arm view is the whole point, and it is what survives the
        enum narrowing.

        ``arm_scores`` and ``arm_confidence`` are OPEN maps on the frozen
        contract, so they can name ``no_agent_quant`` and
        ``single_agent_quant`` where the four enum-typed fields cannot. §3
        requires every arm's measurement to be recorded every cycle; without
        these two maps the record would be silent about the two arms with the
        most evidence.
        """
        cycle, gaps, register = _cycle_for(_producer_board())
        record = decision_record_from_cycle(
            cycle, gaps, register, champion_before=INCUMBENT, freeze=False,
        )
        for arm in FIXTURE_VALID_CHAMPIONS:
            assert arm in record["arm_scores"], arm
            assert record["arm_confidence"][arm] == CONFIDENCE_MEASURED

    def test_an_arm_with_no_series_is_named_unavailable_not_omitted(self):
        cycle, gaps, register = _cycle_for(_producer_board(with_series=False))
        record = decision_record_from_cycle(
            cycle, gaps, register, champion_before=INCUMBENT, freeze=False,
        )
        assert set(record["arm_confidence"].values()) == {CONFIDENCE_UNAVAILABLE}
        assert all(v is None for v in record["arm_scores"].values())

    def test_the_retired_evidence_vocabulary_is_gone(self):
        """``ok``/``thin``/``insufficient`` described a minimum-cohort floor
        that no longer exists. Keeping it as an emitted value would describe a
        gate that does not run — the record would report an evidence verdict
        nothing acts on."""
        assert {CONFIDENCE_MEASURED, CONFIDENCE_UNAVAILABLE} == {"measured", "unavailable"}
        cycle, gaps, register = _cycle_for(_producer_board())
        record = decision_record_from_cycle(
            cycle, gaps, register, champion_before=INCUMBENT, freeze=False,
        )
        assert set(record["arm_confidence"].values()) <= {
            CONFIDENCE_MEASURED, CONFIDENCE_UNAVAILABLE,
        }

    def test_a_quant_promotion_is_named_on_the_audit_record(self):
        """I9406 stops null-projection once the enum admits quant arms.

        ``champion_after`` carries the engine verdict directly; the open
        ``arm_scores`` map preserves the measurement either way.
        """
        cycle, gaps, register = _cycle_for(_deciding_board())
        record = decision_record_from_cycle(
            cycle, gaps, register, champion_before=INCUMBENT, freeze=False,
        )
        assert record["outcome"] == "promoted"
        assert record["champion_after"] == "no_agent_quant"
        assert record["champion_after"] != record["champion_before"]
        assert record["arm_scores"]["no_agent_quant"] == pytest.approx(0.09)

    def test_a_nameable_promotion_is_named(self):
        """The same path with an arm the enum admits, so the null above is
        demonstrably about the enum rather than about promotions."""
        cycle, gaps, register = _cycle_for(_deciding_board(winner="thinktank_coverage"))
        record = decision_record_from_cycle(
            cycle, gaps, register, champion_before=INCUMBENT, freeze=False,
        )
        assert record["outcome"] == "promoted"
        assert record["champion_after"] == "thinktank_coverage"

    def test_freeze_records_the_decision_and_holds_the_pointer(self):
        """--freeze is a SUPPRESSION of a decision already taken, recorded as
        such: the outcome stays ``promoted`` with ``blocked_by=['frozen']`` and
        ``champion_after`` is not advanced, so the record never claims a
        pointer move that did not happen."""
        cycle, gaps, register = _cycle_for(_deciding_board(winner="thinktank_coverage"))
        record = decision_record_from_cycle(
            cycle, gaps, register, champion_before=INCUMBENT, freeze=True,
        )
        assert record["outcome"] == "promoted"
        assert record["blocked_by"] == ["frozen"]
        assert record["champion_after"] == INCUMBENT

    def test_the_counterfactual_winner_is_the_copeland_leader(self):
        """Who would have taken the pointer on evidence alone, ignoring every
        serving precondition. The old two-arm "higher score" reading has no
        meaning in an N-arm slot."""
        cycle, gaps, register = _cycle_for(
            _deciding_board(winner="thinktank_coverage"), shadow=frozenset({"thinktank_coverage"}),
        )
        record = decision_record_from_cycle(
            cycle, gaps, register, champion_before=INCUMBENT, freeze=False,
        )
        assert record["counterfactual_winner"] == "thinktank_coverage"
        assert record["champion_after"] == INCUMBENT

    def test_a_shadow_only_promotion_raises_rather_than_promoting(self):
        """Defence in depth at the projection.

        The engine already excludes shadow-only arms via the
        ``not_shadow_only`` serving precondition, so reaching this branch means
        a precondition was not wired. Fail loud — a silent degrade here would
        be a live pointer moved onto an arm Brian ruled measure-only.
        """
        cycle, gaps, register = _cycle_for(_deciding_board(winner="thinktank_coverage"))
        from optimizer import champion_promotion as cp

        with mock.patch.object(cp, "SHADOW_ONLY_ARMS", frozenset({"thinktank_coverage"})):
            with pytest.raises(ValueError, match="not_shadow_only"):
                decision_record_from_cycle(
                    cycle, gaps, register, champion_before=INCUMBENT, freeze=False,
                )

    def test_every_projected_slug_is_in_the_frozen_vocabulary(self):
        """The projection may only emit slugs the contract already declares —
        the enum lives in nousergon-lib and cannot grow from this repo, which
        is precisely why ``arena_cycle`` is the authoritative artifact."""
        contract_slugs = set(AUDIT_SCHEMA["properties"]["blocked_by"]["oneOf"][1]["items"]["enum"])
        for board, shadow in (
            (_producer_board(with_series=False), frozenset()),
            (_deciding_board(winner="thinktank_coverage"), frozenset({"thinktank_coverage"})),
        ):
            cycle, gaps, register = _cycle_for(board, shadow=shadow)
            record = decision_record_from_cycle(
                cycle, gaps, register, champion_before=INCUMBENT, freeze=False,
            )
            assert set(record["blocked_by"] or []) <= contract_slugs
            assert set(record["blocked_by"] or []) <= set(_BLOCKED_BY_SLUGS)


# ── Weekly audit record (config/apply_audit/producer_champion/{date}.json) ─


class TestBuildChampionAudit:
    def _record(self, board, *, freeze=False, champion_before=INCUMBENT, shadow=frozenset()):
        cycle, gaps, register = _cycle_for(board, incumbent=champion_before, shadow=shadow)
        return decision_record_from_cycle(
            cycle, gaps, register, champion_before=champion_before, freeze=freeze,
        )

    def test_error_path_defaults_to_the_unclassified_slug(self):
        """An error the caller has not classified is ``unclassified_error``.

        It used to be ``leaderboard_unavailable`` for EVERY exception, because
        the leaderboard read was the only thing that could raise when that
        code was written. It is not any more, and the wrong slug sends the
        reader to crucible-research to investigate a defect in this repo.
        """
        audit = build_champion_audit("2026-08-28", None, freeze=False, error="boom")
        assert audit["outcome"] == "error"
        # The ERROR path never ran a cycle, so nothing per-arm was computed —
        # this slug describes the run, not an arm.
        assert audit["blocked_by"] == ["unclassified_error"]
        assert audit["champion_before"] is None
        assert audit["schema_version"] == 2
        assert audit["leaderboard_date_used"] is None

    def test_an_unusable_board_is_classified_as_such(self):
        audit = build_champion_audit(
            "2026-08-28", None, freeze=False, error="board is 40 days older",
            error_slug="leaderboard_unavailable",
        )
        assert audit["blocked_by"] == ["leaderboard_unavailable"]

    def test_error_path_still_declares_arm_scores_explicitly(self):
        """An error run computed no per-arm score. The field is present and
        null — "we did not measure" stated, rather than a missing key a reader
        has to interpret."""
        audit = build_champion_audit("2026-08-28", None, freeze=False, error="boom")
        assert "arm_scores" in audit
        assert audit["arm_scores"] is None

    def test_the_n_arm_view_survives_into_the_durable_record(self):
        """alpha-engine-config-I8756 added ``arm_scores`` to the decision
        record AND to the published contract, but ``build_champion_audit``
        never copied it across — so the field was computed weekly and dropped
        before it became durable. Measured on the live 2026-08-28 artifact: it
        carried ``arm_confidence`` for three arms and no ``arm_scores`` at all,
        and because both challengers were unscored that week
        (``challenger: null``) the pair-shaped
        ``champion_score``/``challenger_score`` fields named a single arm.
        §3 requires every arm's measurement to be recorded every cycle; a
        number discarded before it is written was not recorded."""
        record = self._record(_producer_board())
        audit = build_champion_audit("2026-08-28", record, freeze=False)
        assert audit["arm_scores"] == record["arm_scores"]
        for arm in FIXTURE_VALID_CHAMPIONS:
            assert arm in audit["arm_scores"]

    def test_the_internal_projection_fields_never_reach_the_artifact(self):
        """``_arena_status`` / ``_champion_after_name`` are wiring between the
        projection and the pointer writer, not part of the frozen shape."""
        record = self._record(_producer_board())
        audit = build_champion_audit("2026-08-28", record, freeze=False)
        assert not [k for k in audit if k.startswith("_")]

    def test_hold_path_records_zero_pointer_movement(self):
        record = self._record(_producer_board())
        record["leaderboard_date_used"] = "2026-08-28"
        audit = build_champion_audit("2026-08-28", record, freeze=False)
        assert audit["outcome"] == "unchanged_winner_already_champion"
        assert audit["champion_before"] == audit["champion_after"] == INCUMBENT
        assert audit["leaderboard_date_used"] == "2026-08-28"

    def test_promoted_path_records_leaderboard_date_used(self):
        """The audit trail must show which week's evidence decided a flip, on
        the PROMOTED path and not only on a hold."""
        record = self._record(_deciding_board(winner="thinktank_coverage"))
        record["leaderboard_date_used"] = "2026-08-21"
        audit = build_champion_audit("2026-08-28", record, freeze=False)
        assert audit["outcome"] == "promoted"
        assert audit["leaderboard_date_used"] == "2026-08-21"

    def test_feed_dependencies_follow_the_new_champion_not_the_old(self):
        """``feed_dependencies`` names what the LIVE pointer now depends on.

        ``thinktank_coverage`` declares none, so a promotion onto it must
        record ``None`` rather than carrying the outgoing incumbent's feed
        forward — the field would otherwise assert a dependency the live
        pointer no longer has.
        """
        held = build_champion_audit("2026-08-28", self._record(_producer_board()), freeze=False)
        assert held["feed_dependencies"] == ["research_free_backfill"]
        promoted = build_champion_audit(
            "2026-08-28", self._record(_deciding_board(winner="thinktank_coverage")), freeze=False,
        )
        assert promoted["champion_after"] == "thinktank_coverage"
        assert promoted["feed_dependencies"] is None

    @pytest.mark.parametrize("outcome", OUTCOMES)
    def test_all_outcomes_are_in_the_frozen_vocabulary(self, outcome):
        assert outcome in set(AUDIT_SCHEMA["properties"]["outcome"]["enum"])


# ── Synthetic end-to-end cycles via run_weekly_evaluation ──────────────────


class TestRunWeeklyEvaluation:
    BUCKET = "test-bucket"
    ARENA_KEY = "arena/producer/2026-08-28.json"
    AUDIT_KEY = "config/apply_audit/producer_champion/2026-08-28.json"
    POINTER_KEY = "config/producer_champion.json"

    @pytest.fixture(autouse=True)
    def _no_real_digest(self):
        """``run_weekly_evaluation`` delivers the verdict by email. Without
        this seam every test in this class would attempt a real SMTP/SES send
        and a real S3 dedup-marker read, and the failed send's own escalation
        would add a second ``publish_ops_alert`` call to the error tests.
        The wiring itself is asserted explicitly below, against this mock."""
        with mock.patch(
            "optimizer.champion_promotion.champion_digest.send_verdict_digest",
            return_value=True,
        ) as digest:
            self.digest = digest
            yield digest

    def _s3(self, *, pointer=None, feed_date="2026-08-26"):
        s3 = _FakeS3()
        if pointer is not None:
            s3.put_object(
                Bucket=self.BUCKET, Key=self.POINTER_KEY,
                Body=json.dumps(pointer).encode(),
            )
        if feed_date is not None:
            _put_research_free_backfill_parquet(
                s3, self.BUCKET, newest_prediction_date=feed_date,
            )
        return s3

    def _run(self, s3, board, *, freeze=False, upload=True, date_used="2026-08-28"):
        return run_weekly_evaluation(
            bucket=self.BUCKET, run_date="2026-08-28",
            e2e_lift=_e2e_lift_ok(sn_lift=0.03),
            tt_leaderboard=board, tt_leaderboard_date_used=date_used,
            freeze=freeze, upload=upload, s3_client=s3,
        )

    def test_the_arena_cycle_artifact_is_written_every_cycle(self):
        """§11: the AUTHORITATIVE record, emitted whatever the outcome.

        A slot that emits nothing is not healthy, it is unobserved. This
        artifact — not the narrowed audit record — carries every pairwise
        verdict, the window each rests on, the confidence bound and every
        retirement verdict including the non-retirements.
        """
        s3 = self._s3()
        self._run(s3, _producer_board())
        assert f"{self.BUCKET}/{self.ARENA_KEY}" in s3.store
        assert f"{self.BUCKET}/arena/producer/latest.json" in s3.store
        doc = json.loads(s3.store[f"{self.BUCKET}/{self.ARENA_KEY}"])
        assert doc["slot"] == "producer"
        assert doc["decision"]["status"] == "held"
        assert "series_gaps" in doc

    def test_an_unmeasurable_cycle_is_still_emitted_and_alarms(self):
        """The loudest possible version of "we could not decide".

        A board with no per-date population series cannot form a confidence
        sequence for anything. The pre-I9318 module would have scored this week
        from the SPY-relative aggregate instead — the substitution that inverts
        wins and losses — so the honest outcome is a no-contest that alarms.
        """
        s3 = self._s3()
        with mock.patch("ops_alerts.publish_ops_alert") as alert:
            result = self._run(s3, _producer_board(with_series=False))
        assert result["outcome"] == "no_contest"
        assert result["blocked_by"] == ["arm_score_unavailable"]
        assert f"{self.BUCKET}/{self.ARENA_KEY}" in s3.store
        assert alert.call_count == 1
        assert "unmeasurable" in alert.call_args[0][0]

    def test_a_supported_lead_moves_the_live_pointer(self):
        s3 = self._s3(pointer={
            "schema_version": 1, "champion": INCUMBENT,
            "promoted_at": "2026-07-13T22:07:09Z",
            "promotion_source": "operator_bootstrap",
        })
        result = self._run(s3, _deciding_board(winner="thinktank_coverage"))
        assert result["outcome"] == "promoted"
        pointer = json.loads(s3.store[f"{self.BUCKET}/{self.POINTER_KEY}"])
        assert pointer["champion"] == "thinktank_coverage"
        assert pointer["promotion_source"] == "arena_decided"

    def test_a_promotion_the_audit_enum_cannot_name_still_moves_the_pointer(self):
        """The bug this pairing exists to prevent.

        ``no_agent_quant`` is admitted by the POINTER contract (widened by
        alpha-engine-config-I9299) and NOT by the narrowed audit record's enum
        (alpha-engine-config-I9406). If the pointer writer read the audit
        projection rather than the engine's own verdict, the two arms with the
        most evidence on the board could never be promoted — silently, and for
        a reason no artifact would state.
        """
        s3 = self._s3()
        result = self._run(s3, _deciding_board(winner="no_agent_quant"))
        assert result["outcome"] == "promoted"
        assert result["champion_after"] == "no_agent_quant"
        # The live pointer matches the engine verdict.
        pointer = json.loads(s3.store[f"{self.BUCKET}/{self.POINTER_KEY}"])
        assert pointer["champion"] == "no_agent_quant"
        # ...and the authoritative artifact names it outright.
        doc = json.loads(s3.store[f"{self.BUCKET}/{self.ARENA_KEY}"])
        assert doc["decision"]["champion"] == producer_arena.arm_id_for("no_agent_quant")

    def test_freeze_records_the_decision_and_never_writes_the_pointer(self):
        s3 = self._s3(pointer={
            "schema_version": 1, "champion": INCUMBENT,
            "promoted_at": "2026-07-13T22:07:09Z",
            "promotion_source": "arena_held",
        })
        result = self._run(s3, _deciding_board(winner="thinktank_coverage"), freeze=True)
        assert result["outcome"] == "promoted"
        assert result["blocked_by"] == ["frozen"]
        pointer = json.loads(s3.store[f"{self.BUCKET}/{self.POINTER_KEY}"])
        assert pointer["champion"] == INCUMBENT

    def test_the_audit_record_is_written_on_every_outcome(self):
        """config#2054's liveness proxy. ``config/producer_champion.json``
        mtime cannot prove this engine is alive — a correctly-held week does
        not move it."""
        for board in (_producer_board(), _producer_board(with_series=False)):
            s3 = self._s3()
            with mock.patch("ops_alerts.publish_ops_alert"):
                self._run(s3, board)
            assert f"{self.BUCKET}/{self.AUDIT_KEY}" in s3.store
            assert f"{self.BUCKET}/config/apply_audit/producer_champion/latest.json" in s3.store

    def test_a_raising_cycle_is_recorded_as_an_error_and_alerted(self):
        """Fail-STATIC, not fail-silent: the pointer is untouched, the audit
        record still lands (so the liveness proxy stays honest), and an active
        alert fires — the file-presence SLA alone would be satisfied
        indefinitely by a weekly ``outcome="error"`` write."""
        s3 = self._s3()
        with mock.patch(
            "optimizer.producer_arena.run_arena_cycle", side_effect=RuntimeError("boom"),
        ), mock.patch("ops_alerts.publish_ops_alert") as alert:
            result = self._run(s3, _producer_board())
        assert result["outcome"] == "error"
        assert result["blocked_by"] == ["unclassified_error"]
        assert f"{self.BUCKET}/{self.AUDIT_KEY}" in s3.store
        assert self.POINTER_KEY not in [k.split("/", 1)[1] for k in s3.store]
        alert.assert_called_once()

    def test_a_stale_board_is_an_error_not_a_silently_scored_week(self):
        s3 = self._s3()
        with mock.patch("ops_alerts.publish_ops_alert"):
            result = self._run(s3, _producer_board("2026-08-19"), date_used="2026-08-19")
        assert result["outcome"] == "error"
        assert "days older" in result["detail"]
        # Classified, so the slug sends the reader to the right repo.
        assert result["blocked_by"] == ["leaderboard_unavailable"]

    def test_every_outcome_is_delivered_not_only_a_promotion(self):
        """The measurability gap this seam closes: eleven weekly verdicts
        (2026-07-13 → 2026-08-28), nine of them ``no_contest``, reached no
        operator surface at all. A digest that fired only on a promotion would
        have delivered ONE of the eleven and left the loop indistinguishable
        from a dead one for the other ten."""
        s3 = self._s3()
        result = self._run(s3, _producer_board())
        assert result["outcome"] in OUTCOMES
        self.digest.assert_called_once()
        assert self.digest.call_args[0][0]["date"] == "2026-08-28"

    def test_a_dry_run_never_emails_and_never_writes(self):
        """``upload=False`` is the dry-run contract — no S3 write, and no
        operator's inbox either."""
        s3 = self._s3()
        self._run(s3, _producer_board(), upload=False)
        self.digest.assert_not_called()
        assert f"{self.BUCKET}/{self.ARENA_KEY}" not in s3.store
        assert f"{self.BUCKET}/{self.AUDIT_KEY}" not in s3.store

    def test_a_raising_digest_never_fails_the_weekly_run(self):
        """A notification must never red the pipeline it reports on."""
        s3 = self._s3()
        self.digest.side_effect = RuntimeError("SES down")
        result = self._run(s3, _producer_board())
        assert result["outcome"] in OUTCOMES
        assert f"{self.BUCKET}/{self.AUDIT_KEY}" in s3.store


class TestPointerProvenance:
    """alpha-engine-config-I9318's ``closes-when``, and the reason it is here.

    ``config/producer_champion.json`` read ``promotion_source:
    "operator_bootstrap"`` from 2026-07-13 through 2026-08-29 while an
    automated engine evaluated it every single week. Nothing was wrong with
    the pointer's VALUE; the record of how it got there was simply false, and
    no surface anywhere said so. §11 treats that as a finding rather than a
    stable state, so every cycle now makes the provenance true.
    """

    BUCKET = "test-bucket"
    POINTER_KEY = "config/producer_champion.json"

    @pytest.fixture(autouse=True)
    def _no_real_digest(self):
        with mock.patch(
            "optimizer.champion_promotion.champion_digest.send_verdict_digest",
            return_value=True,
        ):
            yield

    def _run(self, s3, board=None):
        return run_weekly_evaluation(
            bucket=self.BUCKET, run_date="2026-08-28", e2e_lift=_e2e_lift_ok(),
            tt_leaderboard=board if board is not None else _producer_board(),
            tt_leaderboard_date_used="2026-08-28",
            freeze=False, upload=True, s3_client=s3,
        )

    def _s3_with(self, promotion_source):
        s3 = _FakeS3()
        s3.put_object(
            Bucket=self.BUCKET, Key=self.POINTER_KEY,
            Body=json.dumps({
                "schema_version": 1, "champion": INCUMBENT,
                "promoted_at": "2026-07-13T22:07:09Z",
                "promotion_source": promotion_source,
            }).encode(),
        )
        _put_research_free_backfill_parquet(
            s3, self.BUCKET, newest_prediction_date="2026-08-26",
        )
        return s3

    def test_a_held_cycle_corrects_a_stale_bootstrap_provenance(self):
        s3 = self._s3_with("operator_bootstrap")
        self._run(s3)
        pointer = json.loads(s3.store[f"{self.BUCKET}/{self.POINTER_KEY}"])
        assert pointer["promotion_source"] == "arena_held"
        assert pointer["champion"] == INCUMBENT

    def test_the_correction_preserves_when_the_pointer_last_moved(self):
        """The correction must not destroy the fact it exists to record. Six
        weeks of false ``operator_bootstrap`` provenance replaced by six weeks
        of false movement dates would be a worse artifact, not a better one."""
        s3 = self._s3_with("operator_bootstrap")
        self._run(s3)
        pointer = json.loads(s3.store[f"{self.BUCKET}/{self.POINTER_KEY}"])
        assert pointer["promoted_at"] == "2026-07-13T22:07:09Z"

    def test_an_already_true_provenance_is_left_alone(self):
        """Idempotent. A rewrite every Saturday that changes nothing would
        churn the object and make its mtime meaningless as a signal."""
        s3 = self._s3_with("arena_held")
        before = s3.store[f"{self.BUCKET}/{self.POINTER_KEY}"]
        result = self._run(s3)
        assert "_pointer_write" not in result
        assert s3.store[f"{self.BUCKET}/{self.POINTER_KEY}"] == before

    def test_an_unmeasurable_cycle_says_unmeasurable_not_held(self):
        """"We held because nothing beat the incumbent" and "we held because
        we could not measure anything" are different facts about the live
        pointer, and the pointer itself now carries which one applies."""
        s3 = self._s3_with("arena_held")
        with mock.patch("ops_alerts.publish_ops_alert"):
            self._run(s3, _producer_board(with_series=False))
        pointer = json.loads(s3.store[f"{self.BUCKET}/{self.POINTER_KEY}"])
        assert pointer["promotion_source"] == "arena_unmeasurable"

    def test_a_pre_bootstrap_pointer_is_created_with_real_provenance(self):
        """No pointer object at all (the pre-bootstrap state). The cycle still
        records what decided the value it wrote, rather than leaving the slot
        with no pointer and no explanation."""
        s3 = _FakeS3()
        _put_research_free_backfill_parquet(
            s3, self.BUCKET, newest_prediction_date="2026-08-26",
        )
        self._run(s3)
        pointer = json.loads(s3.store[f"{self.BUCKET}/{self.POINTER_KEY}"])
        assert pointer["champion"] == INCUMBENT
        assert pointer["promotion_source"] == "arena_held"


# ── Frozen-schema conformance ─────────────────────────────────────────────


class TestSchemaConformance:
    def _validate(self, schema_path_or_dict, instance):
        jsonschema = pytest.importorskip("jsonschema", reason="jsonschema not installed")
        schema = (
            schema_path_or_dict
            if isinstance(schema_path_or_dict, dict)
            else json.loads(schema_path_or_dict.read_text())
        )
        jsonschema.validate(instance=instance, schema=schema)

    def _audit(self, board, *, freeze=False, shadow=frozenset()):
        cycle, gaps, register = _cycle_for(board, shadow=shadow)
        record = decision_record_from_cycle(
            cycle, gaps, register, champion_before=INCUMBENT, freeze=freeze,
        )
        record["leaderboard_date_used"] = "2026-08-28"
        return build_champion_audit("2026-08-28", record, freeze=freeze)

    def test_pointer_conforms(self):
        pointer = write_champion_pointer(
            "bucket", INCUMBENT, promotion_source="arena_decided", upload=False,
        )
        self._validate(POINTER_SCHEMA_PATH, pointer)

    def test_every_provenance_value_the_reconciler_emits_conforms(self):
        """``promotion_source`` gained four ``arena_*`` values in this change.
        A value the frozen contract refuses would fail on the executor's read,
        not here, so it is checked against the schema directly."""
        for status in ("decided", "held", "unmeasurable", "unservable", "bootstrap"):
            pointer = write_champion_pointer(
                "bucket", INCUMBENT, promotion_source=f"arena_{status}", upload=False,
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

    def test_the_roster_is_a_subset_of_the_pointer_enum(self):
        """The enum is a SUPERSET of the derived roster (it additionally
        read-tolerates the retired 'agentic' seat) — this is the contract the
        ``pointer_contract_admits`` precondition is derived from, asserted
        from the other side."""
        schema = json.loads(POINTER_SCHEMA_PATH.read_text())
        enum = set(schema["properties"]["champion"]["enum"])
        assert set(VALID_CHAMPIONS).issubset(enum)
        assert "agentic" in enum

    def test_held_audit_conforms(self):
        self._validate(AUDIT_SCHEMA, self._audit(_producer_board()))

    def test_promoted_audit_conforms(self):
        audit = self._audit(_deciding_board(winner="thinktank_coverage"))
        assert audit["outcome"] == "promoted"
        self._validate(AUDIT_SCHEMA, audit)

    def test_a_quant_promotion_audit_conforms_once_enum_is_wide(self):
        """Quant-arm promotions must validate once producer_champion_audit admits them."""
        audit = self._audit(_deciding_board(winner="no_agent_quant"))
        assert audit["champion_after"] == "no_agent_quant"
        assert "no_agent_quant" in audit["arm_scores"]
        enum = {
            v for v in AUDIT_SCHEMA["properties"]["champion_after"]["enum"]
            if v is not None
        }
        if "no_agent_quant" not in enum:
            pytest.skip(
                "producer_champion_audit enum widen is in nousergon-lib-PR379"
            )
        self._validate(AUDIT_SCHEMA, audit)

    def test_no_contest_audit_conforms(self):
        audit = self._audit(_producer_board(with_series=False))
        assert audit["outcome"] == "no_contest"
        self._validate(AUDIT_SCHEMA, audit)

    def test_frozen_promotion_audit_conforms(self):
        audit = self._audit(_deciding_board(winner="thinktank_coverage"), freeze=True)
        assert audit["blocked_by"] == ["frozen"]
        self._validate(AUDIT_SCHEMA, audit)

    def test_error_audit_conforms(self):
        self._validate(AUDIT_SCHEMA, build_champion_audit(
            "2026-08-28", None, freeze=False, error="boom",
        ))

    def test_legacy_v1_audit_shape_is_not_expected_to_validate(self):
        """v1 historical records are NOT expected to validate against the v2
        schema (schema_version const changed, fields renamed) — this is by
        design: v1 documents remain valid under the FROZEN v1 shape
        recoverable via git history, and this repo only ever validates
        newly-built (v2) records. Pinned so a future reader doesn't mistake
        the absence of v1 conformance testing for an oversight."""
        legacy_v1 = {
            "schema_version": 1, "date": "2026-07-13", "outcome": "promoted",
            "champion_before": "agentic", "champion_after": INCUMBENT,
            "challenger_matured_cohorts": 0, "sn_lift_vs_champion": None,
            "consecutive_wins": 0, "cooldown_until": "2026-07-27", "blocked_by": None,
        }
        jsonschema = pytest.importorskip("jsonschema", reason="jsonschema not installed")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=legacy_v1, schema=AUDIT_SCHEMA)

    def test_the_arena_cycle_document_is_validated_before_it_is_written(self):
        """M0 contract discipline: validation on the PRODUCER side, at the
        earliest call site, rather than in a consumer weeks later."""
        cycle, gaps, _register = _cycle_for(_producer_board())
        doc = producer_arena.cycle_document(cycle, gaps)
        self._validate(_load_contract_schema("arena_cycle"), doc)

# ── Promotion-time feed-dependency liveness gate (alpha-engine-config-I3165)
# ────────────────────────────────────────────────────────────────────────────
#
# config#3053's root cause, restated as this gate's closes-when bar: a
# promotion record must NAME the promoted arm's upstream feed dependency
# (ARM_FEED_DEPENDENCIES / build_champion_audit's feed_dependencies field,
# covered by TestSchemaConformance above and TestArmFeedDependencies below),
# and a synthetic test must demonstrate the gate BLOCKING a promotion whose
# declared feed has no live producer (TestCheckFeedDependenciesLive /
# TestEvaluateGatesFeedLiveness / TestRunWeeklyEvaluationFeedLiveness below)
# -- as well as the mirror-image case, a promotion proceeding normally when
# the declared feed IS live, so the new gate cannot be mistaken for an
# always-block regression.


class TestArmFeedDependencies:
    def test_scanner_predictor_direct_declares_research_free_backfill(self):
        assert ARM_FEED_DEPENDENCIES["scanner_predictor_direct"] == ["research_free_backfill"]

    def test_thinktank_coverage_declares_no_feed_dependency(self):
        """thinktank_coverage's evidence chain is the producer leaderboard,
        already gated by leaderboard_date_used/leaderboard_stale_gt_8d -- it
        names no live-trade feed artifact of its own."""
        assert ARM_FEED_DEPENDENCIES.get("thinktank_coverage") in (None, [])


class TestCheckFeedDependenciesLive:
    BUCKET = "test-bucket"

    def test_no_declared_dependency_is_always_live(self):
        """An arm with no ARM_FEED_DEPENDENCIES entry (thinktank_coverage)
        is trivially never blocked by this gate -- it has nothing to
        probe."""
        s3 = _FakeS3()
        assert check_feed_dependencies_live(
            "thinktank_coverage", bucket=self.BUCKET, run_date="2026-07-20", s3_client=s3,
        ) is None

    def test_live_fresh_feed_passes(self):
        """(a) The declared feed's producer IS live/fresh -- the gate
        returns None (not blocked)."""
        s3 = _FakeS3()
        _put_research_free_backfill_parquet(s3, self.BUCKET, newest_prediction_date="2026-07-17")
        assert check_feed_dependencies_live(
            "scanner_predictor_direct", bucket=self.BUCKET, run_date="2026-07-20", s3_client=s3,
        ) is None

    def test_stale_feed_is_blocked(self):
        """(b) The declared feed's newest prediction_date is stale beyond
        the freshness window (config#3053's exact incident shape: the
        producer silently stopped refreshing) -- blocked with the new
        slug."""
        s3 = _FakeS3()
        _put_research_free_backfill_parquet(s3, self.BUCKET, newest_prediction_date="2026-07-01")
        assert check_feed_dependencies_live(
            "scanner_predictor_direct", bucket=self.BUCKET, run_date="2026-07-20", s3_client=s3,
        ) == "feed_producer_dead"

    def test_missing_feed_artifact_is_blocked(self):
        """The declared feed artifact does not exist at all (orphaned
        producer, never wrote anything) -- blocked, not a crash."""
        s3 = _FakeS3()  # nothing uploaded
        assert check_feed_dependencies_live(
            "scanner_predictor_direct", bucket=self.BUCKET, run_date="2026-07-20", s3_client=s3,
        ) == "feed_producer_dead"

    def test_probe_exception_is_blocked_not_raised(self):
        """Belt-and-braces: even an UNEXPECTED exception from the
        registered prober (not just the StaleChampionFeedError it's
        designed to raise) must degrade to feed_producer_dead, never
        propagate -- the module's binding config#2884 lesson applies to
        this gate exactly as much as to the rest of evaluate_gates."""
        s3 = _FakeS3()

        class _ExplodingS3:
            def get_object(self, Bucket, Key):
                raise RuntimeError("boom - unexpected probe failure")

        assert check_feed_dependencies_live(
            "scanner_predictor_direct", bucket=self.BUCKET, run_date="2026-07-20",
            s3_client=_ExplodingS3(),
        ) == "feed_producer_dead"

    def test_unregistered_feed_dependency_fails_open_without_crashing(self):
        """An arm declaring a feed id with no registered prober in
        _FEED_LIVENESS_PROBES must not crash this gate -- it's simply not
        checked (logged), never a silent block or a crash. Verified via a
        monkeypatched ARM_FEED_DEPENDENCIES entry rather than mutating the
        real one."""
        import optimizer.champion_promotion as cp

        original = dict(cp.ARM_FEED_DEPENDENCIES)
        cp.ARM_FEED_DEPENDENCIES["thinktank_coverage"] = ["some_unregistered_feed"]
        try:
            s3 = _FakeS3()
            result = check_feed_dependencies_live(
                "thinktank_coverage", bucket=self.BUCKET, run_date="2026-07-20", s3_client=s3,
            )
            assert result is None
        finally:
            cp.ARM_FEED_DEPENDENCIES.clear()
            cp.ARM_FEED_DEPENDENCIES.update(original)




class TestFeedLivenessEndToEnd:
    """alpha-engine-config-I3165, re-wired as a per-arm SERVING PRECONDITION.

    The old shape probed the feed of "the challenger that would win" — a
    question with no answer before the engine has decided, and meaningless in
    an N-arm slot with a freely-moving pointer. It is now evaluated for every
    arm that declares a feed, BEFORE the cycle, and each result reaches the
    ``arena_cycle`` artifact as a named precondition with its reason rather
    than as a post-hoc veto.
    """

    BUCKET = "test-bucket"

    @pytest.fixture(autouse=True)
    def _no_real_digest(self):
        with mock.patch(
            "optimizer.champion_promotion.champion_digest.send_verdict_digest",
            return_value=True,
        ):
            yield

    def _run(self, s3, board):
        return run_weekly_evaluation(
            bucket=self.BUCKET, run_date="2026-08-28", e2e_lift=_e2e_lift_ok(),
            tt_leaderboard=board, tt_leaderboard_date_used="2026-08-28",
            freeze=False, upload=True, s3_client=s3,
        )

    def test_a_live_feed_leaves_the_arm_eligible(self):
        s3 = _FakeS3()
        _put_research_free_backfill_parquet(
            s3, self.BUCKET, newest_prediction_date="2026-08-26",
        )
        self._run(s3, _producer_board())
        doc = json.loads(s3.store[f"{self.BUCKET}/arena/producer/2026-08-28.json"])
        assert doc["decision"]["status"] == "held"
        assert doc["decision"]["champion"] == producer_arena.arm_id_for(INCUMBENT)

    def test_a_dead_feed_is_recorded_against_the_arm_that_declares_it(self):
        """The config#3053 root cause, at the layer that can see it.

        ``scanner_predictor_direct``'s live-trade chain
        (``research_free_backfill``) was orphaned by config#1580 one day after
        the 2026-07-13 bootstrap and stayed dead, invisibly, for ten days —
        because no arm-to-feed mapping existed anywhere and nothing at
        promotion time checked one. Here the feed is stale by months, and the
        artifact must NAME the failed precondition against the arm rather than
        the cycle just coming out held for an unstated reason.
        """
        s3 = _FakeS3()
        _put_research_free_backfill_parquet(
            s3, self.BUCKET, newest_prediction_date="2026-01-05",
        )
        with mock.patch("ops_alerts.publish_ops_alert"):
            self._run(s3, _producer_board())
        doc = json.loads(s3.store[f"{self.BUCKET}/arena/producer/2026-08-28.json"])
        blocked = doc["decision"]["ineligible"][producer_arena.arm_id_for(INCUMBENT)]
        failed = [c for c in blocked if not c["passed"]]
        assert [c["name"] for c in failed] == ["feed_producer_live"]
        assert "dead or orphaned" in failed[0]["reason"]

    def test_a_dead_feed_forces_the_pointer_OFF_the_arm_that_declares_it(self):
        """The behaviour CHANGE, stated rather than discovered in production.

        The old gate degraded a would-be PROMOTION onto a dead feed to a
        no-contest and left the pointer where it was — which, when the dead
        feed belonged to the INCUMBENT, is precisely the config#3053 state:
        the live pointer parked on an arm that could not trade. The shared
        engine treats serving as a property of the arm the pointer rests on,
        so an unservable incumbent forces the pointer to the best eligible arm
        and names the reason on the artifact.

        This is a real broadening of what a feed failure can do, and it is the
        correct direction: the alternative is holding a pointer whose arm is
        known to be unable to trade.
        """
        s3 = _FakeS3()
        _put_research_free_backfill_parquet(
            s3, self.BUCKET, newest_prediction_date="2026-01-05",
        )
        with mock.patch("ops_alerts.publish_ops_alert"):
            result = self._run(s3, _producer_board())
        assert result["outcome"] == "promoted"
        pointer = json.loads(s3.store[f"{self.BUCKET}/config/producer_champion.json"])
        assert pointer["champion"] != INCUMBENT
        doc = json.loads(s3.store[f"{self.BUCKET}/arena/producer/2026-08-28.json"])
        assert "failed a serving precondition" in doc["decision"]["reason"]
        assert "feed_producer_live" in doc["decision"]["reason"]

    def test_a_missing_feed_artifact_is_treated_as_dead_not_as_absent(self):
        """No parquet at all. An unreadable feed is a dead feed for this
        purpose — probed, recorded, and never crashed through."""
        s3 = _FakeS3()
        with mock.patch("ops_alerts.publish_ops_alert"):
            self._run(s3, _producer_board())
        doc = json.loads(s3.store[f"{self.BUCKET}/arena/producer/2026-08-28.json"])
        blocked = doc["decision"]["ineligible"][producer_arena.arm_id_for(INCUMBENT)]
        assert any(c["name"] == "feed_producer_live" and not c["passed"] for c in blocked)


# ── Shadow-only arms (the MECHANISM, currently unused) ────────────────────


class TestShadowOnlyArms:
    """Measured, never promoted.

    ``thinktank_coverage`` held this property under Brian's 2026-08-20 ruling.
    His 2026-08-27 ruling released it:

        "I want all other challengers to remain challengers such as think tank
        and other scanner configurations. But at this point I'm thinking we
        promote the best performer weekly"

    So ``SHADOW_ONLY_ARMS`` is EMPTY, and every test below exercises the
    mechanism through a monkeypatched membership rather than through whichever
    arm happens to hold the property today. That is the point: the mechanism is
    what must survive a membership change, and a test written against the
    membership dies with it — taking the coverage of the mechanism with it,
    exactly when the next arm needs the protection.
    """

    def test_the_mechanism_is_empty_but_intact(self):
        assert SHADOW_ONLY_ARMS == frozenset()
        assert is_shadow_only("thinktank_coverage") is False
        assert is_shadow_only("scanner_top20_predictor") is False
        assert is_shadow_only(None) is False

    def test_an_arm_added_to_the_set_becomes_shadow_only(self, monkeypatch):
        """Shadow-only-ness is a declared PROPERTY OF AN ARM, not a literal at
        the veto site. A future arm inherits the protection by joining this
        frozenset and nothing else."""
        from optimizer import champion_promotion as cp

        monkeypatch.setattr(cp, "SHADOW_ONLY_ARMS", frozenset({"thinktank_coverage"}))
        assert cp.is_shadow_only("thinktank_coverage") is True
        assert cp.is_shadow_only(INCUMBENT) is False

    def test_neither_enforcement_layer_hard_codes_an_arm_name(self):
        """Fix the CLASS, not the instance.

        Two layers enforce this: ``producer_arena.build_preconditions`` (the
        POLICY — the engine sees a named, recorded ineligibility) and
        ``write_champion_pointer`` (the INVARIANT — nothing reaching S3 can
        violate it). Pinned by source inspection because a hard-coded second
        copy would pass every behavioural test in this file while silently not
        covering the next shadow arm.
        """
        import ast
        import inspect
        import textwrap

        from optimizer import champion_promotion as cp

        for fn in (producer_arena.build_preconditions, cp.write_champion_pointer):
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
            node = tree.body[0]
            # Drop the docstring; comments are not in the AST at all. Only
            # EXECUTABLE code is inspected, so prose may name the arm -- and
            # must, to be readable -- while the logic may not.
            stmts = node.body
            if (stmts and isinstance(stmts[0], ast.Expr)
                    and isinstance(stmts[0].value, ast.Constant)
                    and isinstance(stmts[0].value.value, str)):
                stmts = stmts[1:]
            code = "\n".join(ast.unparse(st) for st in stmts)
            assert "thinktank_coverage" not in code, fn.__name__

    def test_a_shadow_arm_is_measured_ranked_and_held_off_the_pointer(self):
        """§3: measurement is unconditional and is NOT what promotion governs.

        The shadow arm's ladder is built, its pairwise verdicts are taken, it
        leads the Copeland ranking — and the pointer does not move. An arm that
        is quietly excluded from the contest instead is an observation, not a
        shadow challenger, and the counterfactual shadow mode exists to measure
        would be erased.
        """
        cycle, gaps, register = _cycle_for(
            _deciding_board(winner="thinktank_coverage"),
            shadow=frozenset({"thinktank_coverage"}),
        )
        shadow_id = producer_arena.arm_id_for("thinktank_coverage")
        assert cycle.ranking.ordering[0] == shadow_id
        assert any(lad.arm_id == shadow_id and lad.rungs for lad in cycle.ladders)
        assert cycle.decision.champion != shadow_id
        failed = [c for c in cycle.decision.ineligible[shadow_id] if not c.passed]
        assert [c.name for c in failed] == ["not_shadow_only"]

        record = decision_record_from_cycle(
            cycle, gaps, register, champion_before=INCUMBENT, freeze=False,
        )
        assert record["champion_after"] == INCUMBENT
        assert record["blocked_by"] == ["shadow_only_arm"]
        assert record["counterfactual_winner"] == "thinktank_coverage"
        assert record["arm_scores"]["thinktank_coverage"] == pytest.approx(0.09)

    def test_a_non_shadow_arm_with_the_same_lead_promotes(self):
        """The control. Without it the test above would pass equally well if
        the arena had simply stopped promoting anything."""
        cycle, gaps, register = _cycle_for(_deciding_board(winner="thinktank_coverage"))
        record = decision_record_from_cycle(
            cycle, gaps, register, champion_before=INCUMBENT, freeze=False,
        )
        assert record["outcome"] == "promoted"
        assert record["champion_after"] == "thinktank_coverage"
