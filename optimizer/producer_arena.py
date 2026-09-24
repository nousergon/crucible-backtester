"""producer_arena.py — the selection-producer slot's arena wiring (alpha-engine-config-I9318).

**This module owns the pointer DECISION for the selection-producer slot**
(``signals/{date}/signals.json``). It does not implement any of it: every
rule in ``champion-challenger-policy.md`` §§3–6 — the score ladder, the
longest-common-window pairing, the anytime-valid confidence sequence, the
Copeland ranking, the pointer rule and the cap-with-grace retirement rule —
comes from ``nousergon_lib.arena``, the fleet's single implementation
(`shared-code-policy.md`; a slot re-implementing §§3–6 is a defect, §10).
What lives here is the four things the engine deliberately does NOT do:

1. **The register** — which arms exist, when each was first observed, and
   which are retired. Derived from crucible-research's producer registry as
   projected onto ``research/producer_leaderboard/{date}.json``; persisted as
   a durable append-only artifact (:data:`REGISTER_PATH`), never recomputed
   from scratch per cycle.
2. **The benchmark and the series** — the per-date, POPULATION-relative
   score each arm is graded on.
3. **The serving preconditions** — shadow-only arms, feed liveness, and
   whether the executor can actually serve an arm — passed in as results.
4. **The artifact** — ``arena/producer/{date}.json``, schema-valid
   against ``nousergon_lib.contracts``'s ``arena_cycle``, emitted EVERY
   cycle whatever the outcome (§11).

**Which contract is authoritative.** From this change the ``arena_cycle``
artifact is the AUTHORITATIVE record of the slot's decision: the pointer,
every pairwise verdict with the window it rests on, the confidence-sequence
bound, and every retirement verdict including the non-retirements.
``config/apply_audit/producer_champion/{date}.json``
(``producer_champion_audit`` v2) is retained as a NARROWED, derived view for
its existing consumers — crucible-dashboard ``views/46_Experiments.py``,
crucible-evaluator ``grading/attestation.py`` /
``grading/tiles/backtester.py`` / ``director/report_card_digest.py`` — and
its ``outcome``/``champion_after``/``blocked_by`` fields are now PROJECTIONS
of the arena decision rather than an independent computation. It is
deliberately not dual-WRITTEN in the sense §10 forbids: there is exactly one
decision, taken once, in :func:`run_arena_cycle`, and two renderings of it.
``producer_champion_audit`` RETIRES when every consumer above reads
``arena_cycle`` instead, which is option (B) of
``alpha-engine-config-I9406`` — so "when" is a tracked issue rather than a
mood. The same issue records the narrowed view's live cost: its four
enum-typed arm fields cannot NAME ``no_agent_quant`` or
``single_agent_quant``, so the projection nulls them and logs, and the open
``arm_scores`` map is what keeps their measurement from being lost.

**Why the benchmark may not be SPY.** ``ArenaConfig`` REFUSES a
selection-stage slot graded against anything but the population it selected
from, and the refusal is load-bearing rather than stylistic: on 2026-08-17
SPY trailed the drawn-from population by 140bp at 21d, which inverts wins
and losses outright. The producer leaderboard also publishes the
SPY-relative ``topn_alpha_vs_benchmark``; this module reads the per-date
``topn_alpha_vs_population`` series and NEVER the SPY figure, and it refuses
to score rather than substitute one for the other. The engine ranks the arms
on the information ratio of that series (``ARENA_CONFIG.promote_statistic``,
alpha-engine-config-I11393).

**Fail loud, never fill in.** An arm the register knows about that the board
cannot supply a series for is recorded as an explicit, named
:class:`SeriesGap` and reaches the artifact as an unmeasurable comparison —
never as a zero, never as an omission, never as a silently-narrowed roster
(§7.2: the fleet's dominant bug class is a well-formed artifact containing
nothing). A cycle whose decision status is ``unmeasurable`` or
``unservable`` publishes an ops alert (§11).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3

from nousergon_lib.arena import (
    ArenaConfig,
    ArenaCycle,
    ArmRegister,
    ArmSeries,
    ServingPrecondition,
    run_cycle,
)
from nousergon_lib.contracts import validate as validate_contract

logger = logging.getLogger(__name__)

__all__ = [
    "ARENA_CONFIG",
    "ARENA_CYCLE_PREFIX",
    "BOARD_SNAPSHOT_PATH",
    "CREATED_DATE_EARLIEST_COHORT",
    "CREATED_DATE_FIRST_BOARD",
    "NEW_ARM_CREATED_DATE_RULE",
    "PINNED_RESEARCH_PREFILTER",
    "RESEARCH_SLOT_ARM_WIDTHS",
    "RETIREMENTS_HELD",
    "POINTER_CONTRACT_PATH",
    "POPULATION_SERIES_FIELD",
    "REGISTER_PATH",
    "SLOT",
    "UNBOARDED_ARMS",
    "SLOT_KIND",
    "WRITE_FORBIDDEN_ARMS",
    "SeriesGap",
    "append_only_violations",
    "arm_id_for",
    "arm_statistics",
    "board_snapshot",
    "build_series",
    "load_register",
    "pointer_admissible_arms",
    "promotion_eligible_arm_names",
    "register_events_from_boards",
    "roster_disagreement",
    "run_arena_cycle",
    "write_arena_cycle",
]

SLOT = "producer"

#: ``ArenaConfig`` REFUSES ``benchmark != "population"`` for this value. That
#: refusal is the mechanism, not a convention — see the module docstring.
SLOT_KIND = "selection_producer"

#: The durable, append-only arm register for this slot. A COMMITTED artifact,
#: not something recomputed per cycle: ``created_date`` drives the four-week
#: grace period, so a value that could silently change between cycles would
#: make retirement non-reproducible. Backfilled once by
#: ``scripts/backfill_producer_arena_register.py`` from the real
#: ``research/producer_leaderboard/`` history; extended (never rewritten) when
#: a new arm first appears on the board.
REGISTER_PATH = Path(__file__).resolve().parent / "arena" / "producer_register.json"

#: The committed projection of every ``research/producer_leaderboard/`` board
#: the register was last folded from: per board, each arm's name, ``kind`` and
#: earliest scored cohort — exactly the three facts
#: :func:`register_events_from_boards` reads, and nothing else.
#:
#: It exists so CI can prove the register is not behind its source WITHOUT
#: S3 credentials: ``scripts/backfill_producer_arena_register.py --check``
#: re-folds this snapshot onto the committed register and fails if that would
#: append anything (alpha-engine-config-I11490). The backfill writes both
#: files in one run, so they cannot be refreshed separately.
BOARD_SNAPSHOT_PATH = Path(__file__).resolve().parent / "arena" / "producer_board_snapshot.json"

# ── Which date a NEWLY registered arm is created on (OPEN — Brian, I11393) ──
#
# ``created_date`` starts an arm's four-week grace period (§6), so it decides
# whether a new arm can be retired on its first cycle. Two readings exist and
# the ruling between them is Brian's, not this module's:
#
#   CREATED_DATE_FIRST_BOARD       the date of the first producer leaderboard
#                                  that lists the arm. Grace starts when the
#                                  arm enters the contest. For the five
#                                  research-slot arms that is 2026-09-23.
#   CREATED_DATE_EARLIEST_COHORT   the earliest cohort the board has scored
#                                  for it, backfilled cohorts included (the
#                                  derivation's rule before this change). For
#                                  attractiveness_60 that is 2026-05-29, so
#                                  its grace period has already elapsed on the
#                                  day it first appears.
#
# Either rule applies ONLY to an arm the committed register does not already
# hold. An arm already registered keeps the date it was registered with,
# whichever rule is in force — the register is append-only.
CREATED_DATE_FIRST_BOARD = "first_board_appearance"
CREATED_DATE_EARLIEST_COHORT = "earliest_backfilled_cohort"
CREATED_DATE_RULES = (CREATED_DATE_FIRST_BOARD, CREATED_DATE_EARLIEST_COHORT)

#: DECISION LINE (alpha-engine-config-I11393, open decision 3). Changing the
#: rule is this one line plus a re-run of the backfill script.
NEW_ARM_CREATED_DATE_RULE = CREATED_DATE_FIRST_BOARD

#: Arms whose ``kind == "retired"`` on the board is NOT recorded as a
#: retirement in this register (alpha-engine-config-I11393, open decision 1).
#:
#: ``scanner_predictor_direct`` (the live pointer's arm) and
#: ``scanner_top20_predictor`` are the 60-wide / 20-wide funnel-width
#: experiment that Amendment 1 of I11393 keeps, and Brian said on 2026-09-24
#: "don't retire R arms". The research board has marked both retired since
#: 2026-09-23; this set stops the register copying that.
#:
#: DECISION LINE: removing a name records its retirement on the next backfill.
RETIREMENTS_HELD: frozenset[str] = frozenset({"scanner_predictor_direct", "scanner_top20_predictor"})

# ── The research slot's arms (alpha-engine-config-I11393, Amendment 2) ────
#
# The five arms all draw from ONE pinned pre-filter and each emits its own
# DECLARED width. The width is part of the arm's immutable recipe (§3.1), so
# it is hashed into the arm id: an arm that changes its width becomes a new
# arm, it does not keep the old arm's track record.
#
# Mirrors crucible-research ``producers/registry.py`` (``PINNED_RESEARCH_
# PREFILTER`` and each ``ProducerSpec.width``). The board publishes the width
# each arm actually emitted (``widths``), and :func:`build_series` refuses to
# score an arm whose emitted width differs from the one declared here.
PINNED_RESEARCH_PREFILTER = "attractiveness_top_60"
RESEARCH_SLOT_ARM_WIDTHS: dict[str, int] = {
    "attractiveness_60": 60,
    "attractiveness_20": 20,
    "tech_score_20": 20,
    "predictor_from_60": 20,
    "thinktank_20": 20,
}

#: Lineage recorded on the register (``ArmRecord.supersedes``), from
#: crucible-research ``ProducerSpec.supersedes``.
RESEARCH_SLOT_SUPERSEDES: dict[str, str] = {"thinktank_20": "thinktank_coverage"}

#: ``arena/producer/{date}.json`` + ``latest.json`` — the §11 artifact.
#:
#: MIRRORS the S-slot's layout (``optimizer/strategy_arena.py``'s
#: ``arena/strategy/{date}.json``, alpha-engine-config-I9320) rather than
#: inventing a second prefix family under ``research/``. One ``arena/``
#: namespace means a console adapter, a freshness row or a backfill can
#: enumerate every slot's cycles with one prefix — the alternative is a
#: per-slot convention that has to be discovered slot by slot.
ARENA_CYCLE_PREFIX = "arena/producer"

#: The per-date, POPULATION-relative score series this slot grades on, read
#: off each ``specs[]`` row of ``research/producer_leaderboard/{date}.json``.
#:
#: crucible-research COMPUTES this series today — ``scoring/
#: leaderboard_scoring.py::_topn_alpha_vs_population_metric`` builds the
#: per-date list and hands it to ``date_clustered_stats`` — and then publishes
#: only the aggregate (``topn_alpha_vs_population``: mean/se/t_stat/n_dates).
#: The per-date values are discarded before the artifact is written.
#:
#: A confidence sequence is a statement about a SEQUENCE of paired per-date
#: differences; it cannot be formed from two cumulative means, and feeding it
#: one would be a false statement about what the interval covers. So this
#: module reads the per-date field and, when the board does not carry it,
#: records a named :class:`SeriesGap` and lets the cycle come out
#: ``unmeasurable`` — LOUDLY, on the artifact and on an ops alert — rather
#: than substituting a number that would decide the live pointer on a
#: statistic nobody can defend.
#:
#: Emitting it is a one-field additive change in the repo that OWNS the
#: measurement (crucible-research), which is the correct architectural layer;
#: tracked as alpha-engine-config-I9405.
POPULATION_SERIES_FIELD = "topn_alpha_vs_population_by_date"

#: The frozen cross-repo pointer contract THIS repo owns and the executor
#: reads. Its ``champion`` enum is the authoritative statement of which values
#: ``config/producer_champion.json`` may carry.
POINTER_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1] / "contracts" / "producer_champion.schema.json"
)

#: Enum values that are READ-TOLERATED but WRITE-FORBIDDEN. ``agentic`` names
#: the retired multi-agent pipeline; the enum keeps it so a historical pointer
#: object still validates, and the pointer writer has refused to emit it since
#: the 2026-07-14 seat swap.
WRITE_FORBIDDEN_ARMS: frozenset[str] = frozenset({"agentic"})


def pointer_admissible_arms(path: Path | None = None) -> frozenset[str]:
    """Arms the live pointer may be moved ONTO, read off the frozen contract.

    DERIVED, not typed — and deliberately derived from the POINTER CONTRACT
    rather than from a mirror of ``crucible-executor/executor/champion.py``.
    A mirror of another repo's literal is what this whole change exists to
    delete: the previous version of this constant was written on 2026-08-29
    as a copy of that tuple and was stale within hours, because
    ``alpha-engine-config-I9299`` landed the same day and made
    ``no_agent_quant`` and ``single_agent_quant`` servable. A copy cannot
    detect that it has gone stale; a contract read from disk cannot go stale
    without the file changing.

    It is the right boundary as well as the safer one: the pointer's
    ``champion`` enum is the promise this repo makes to the executor, the
    executor fail-louds on a value outside it (``ChampionPointerError``), and
    the enum is additive-only — so widening it is the deliberate act that
    admits a new arm, in the same repo as the writer.
    """
    schema = json.loads((path or POINTER_CONTRACT_PATH).read_text())
    return frozenset(schema["properties"]["champion"]["enum"]) - WRITE_FORBIDDEN_ARMS


#: Evaluated once at import: a contract that cannot be read is a reason to
#: refuse to load, not to guess a permissive default.
POINTER_ADMISSIBLE_ARMS: frozenset[str] = pointer_admissible_arms()

#: Arms this repo knows are real but that NO producer leaderboard has ever
#: listed, with the earliest date each is documented to have started producing
#: and where that date comes from.
#:
#: ``scanner_top20_predictor`` is the live example and the reason this exists:
#: Brian's 2026-08-27 ruling names it, ``config/producer_champion.json``'s
#: schema admits it as a pointer value, and crucible-executor can serve it —
#: yet it appears on no board, because until alpha-engine-config-I9307 it was
#: scored ONLY as a crucible-backtester end-to-end counterfactual, on a
#: different source and a different cohort from every other arm. Leaving it
#: out of the register because the board is silent about it would reproduce
#: exactly the silent-omission defect this change closes; registering it
#: makes its silence visible as a named :class:`SeriesGap` every cycle
#: instead.
#:
#: A seed is used only when the arm is not yet registered. If a board also
#: lists the arm at that point, the earlier of the seed and the board date
#: wins. Once the arm is registered, its date is never rewritten.
UNBOARDED_ARMS: dict[str, tuple[str, str]] = {
    "scanner_top20_predictor": (
        "2026-07-30",
        "the top-20 cut began 2026-07-30 — recorded in this repo at "
        "optimizer/champion_promotion.py::_score_scanner_top20_predictor and in "
        "crucible-research producers/registry.py. No producer leaderboard has "
        "ever carried a row for this arm (verified across every artifact under "
        "research/producer_leaderboard/, 2026-08-03 .. 2026-08-28).",
    ),
}


# ── The slot's ArenaConfig (champion-challenger-policy.md §10) ─────────────
#
# §10 requires every slot to name its metric, benchmark and every ArenaConfig
# parameter in the registry that owns it, deliberately NOT in the policy —
# they are per-slot facts CI can check against code. This IS that registry row
# for the selection-producer slot.
#
#   metric                  topn_alpha_vs_population, per cohort date, 21
#                           trading sessions forward, top-N equal weight
#   benchmark               population (the scanner candidate set the arm
#                           narrowed) — SPY is REFUSED by the engine here
#   width                   DECLARED PER ARM (RESEARCH_SLOT_ARM_WIDTHS), not
#                           count-matched. The board sets per_arm_width=true
#                           (alpha-engine-config-I11393, Amendment 2)
#   primary statistic       information ratio of each arm over the pair's
#                           common window (promote_statistic below). A raw
#                           mean with free widths rewards the narrowest arm
#                           for stopping early, because mean alpha per name
#                           falls with depth whenever a ranking has skill.
#                           IR prices that concentration (IR ~= IC x
#                           sqrt(breadth))
#   reported, not ranked    realized_rank_ic per arm, which is ranking skill
#                           independent of width. It is written to the cycle
#                           artifact beside the IR (arm_statistics) so a win
#                           can be attributed to ranking or to breadth
#
# ``diff_clip`` — declared bound on a per-date score DIFFERENCE between two
# arms, in the score's own units (21-day excess return over the drawn-from
# population, as a fraction). Justified from the observed range rather than
# picked: across every producer leaderboard that carries the population
# metric, the per-arm cumulative means run from -0.070646 (thinktank_coverage,
# 2026-08-28) to +0.018975 (no_agent_quant, 2026-08-21) — a widest observed
# CROSS-ARM gap of 0.0861 (no_agent_quant vs thinktank_coverage, 2026-08-21)
# and 0.0752 on 2026-08-28. 0.10 sits just above the widest observed gap, so
# the clip bounds the sub-Gaussian scale without truncating a difference the
# slot has actually produced. Clipping tighter would bias every comparison
# toward the incumbent by shrinking real leads; the count of clipped
# observations is reported on the artifact (`n_clipped`) so the choice stays
# reviewable rather than assumed.
#
# Consequence, stated rather than discovered later: with
# ``variance_mode="declared"`` the sub-Gaussian scale IS 0.10, so the interval
# is wide and a promotion needs a sustained lead rather than one good cohort.
# That is the intended trade — the alternative is a bar this slot's history
# (2026-07-13 pointer, never moved on evidence) shows it cannot honestly
# clear. If the slot proves unable to promote within ``opt_n`` cycles on a
# real lead, the declared next step is ``variance_mode="empirical"``, which is
# tighter and is what most practitioners use, at the cost of an interval that
# is no longer checkable from configuration alone.
ARENA_CONFIG = ArenaConfig(
    slot=SLOT,
    slot_kind=SLOT_KIND,
    benchmark="population",
    alpha=0.05,
    diff_clip=0.10,
    variance_mode="declared",
    opt_n=26,
    # Well-formedness only: one paired date is the least from which any
    # statistic can be formed. NOT an evidence bar — the confidence sequence
    # is the evidence bar, and every `thin_evidence` / minimum-cohort /
    # minimum-week floor on this slot's decision path is deleted by this
    # change (issue deliverable 6).
    min_paired_dates=1,
    # alpha-engine-config-I11393 methodology: rank on the information ratio,
    # promote the point-estimate leader once the pair has 2 paired weeks.
    # That is Brian's universe_cut ruling (2026-09-12, I10546) applied to
    # this slot, and it needs its own §5.2(B) delta record in
    # champion-challenger-policy.md. The engine also REQUIRES point evidence
    # with the IR statistic: the confidence sequence bounds a mean of per-date
    # differences and says nothing about a difference of two ratios. It is
    # still computed on mean_diff and emitted for every comparison.
    promote_statistic="information_ratio",
    promote_evidence="point",
    promote_min_weeks=2,
    cap=5,
    grace_weeks=4,
    min_active_arms=3,
    # Matches crucible-research's RETIRED_TRAILING_WINDOW_CYCLES = 8, so the
    # two repos score a retired arm over the same window.
    retired_trailing_cycles=8,
    retire_evidence="point",
    max_ladder_weeks=26,
)


@dataclass(frozen=True)
class SeriesGap:
    """An arm the register knows about that this cycle could not score.

    Carried onto the cycle's own record and into the ops alert. Deliberately
    a first-class value rather than a log line: an arm that silently drops
    out of the roster is the ``thinktank_coverage`` defect, and the whole
    point of §7.2 is that absence must be as legible as presence.
    """

    arm_name: str
    arm_id: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"arm_name": self.arm_name, "arm_id": self.arm_id, "reason": self.reason}


# ── The register ──────────────────────────────────────────────────────────


def _spec_for(name: str) -> dict[str, Any]:
    """The immutable identity a producer arm's ``arm_id`` hashes.

    Deliberately minimal and STABLE. The arm's real recipe — features,
    prompt, refit cadence — lives in crucible-research's
    ``producers/registry.py`` and is not published on the leaderboard
    artifact, so hashing anything the board DOES carry (``kind``,
    ``promotion_eligible``) would mint a brand-new arm, and destroy the
    track record, the first time an arm was retired or its eligibility
    changed. Identity is the arm's NAME in the producer registry; the
    register's ``notes`` record where the recipe itself lives.

    A research-slot arm (:data:`RESEARCH_SLOT_ARM_WIDTHS`) also hashes its
    pinned pre-filter and its declared width. Both are part of its recipe
    under alpha-engine-config-I11393, so an arm that changed either would get
    a new id instead of inheriting this one's record. Arms registered before
    that slot keep their original two-field spec, so their ids do not move.
    """
    spec: dict[str, Any] = {
        "producer_name": name,
        "registry": "crucible-research/producers/registry.py",
    }
    if name in RESEARCH_SLOT_ARM_WIDTHS:
        spec["prefilter_cut"] = PINNED_RESEARCH_PREFILTER
        spec["width"] = RESEARCH_SLOT_ARM_WIDTHS[name]
    return spec


def arm_id_for(name: str) -> str:
    from nousergon_lib.arena import derive_arm_id

    return derive_arm_id(SLOT, name, _spec_for(name))


def load_register(path: Path | None = None) -> ArmRegister:
    """Load the durable append-only register. RAISES if it is missing.

    No fallback to an empty register: an empty one would silently reset every
    arm's ``created_date`` to today and disable the four-week grace period
    across the whole pool, which is exactly the shape of failure this slot
    already suffered once.
    """
    target = path or REGISTER_PATH
    payload = json.loads(target.read_text())
    return ArmRegister.from_dicts(payload["events"])


def _board_rows(board: dict) -> list[dict]:
    return [r for r in (board.get("arms") or board.get("specs") or []) if isinstance(r, dict)]


def _earliest_cohort(row: dict) -> str | None:
    """The earliest cohort a board row reports, from a full board or a snapshot."""
    dates = [d for d in (row.get("dates_scored") or []) if isinstance(d, str)]
    if isinstance(row.get("earliest_cohort"), str):
        dates.append(row["earliest_cohort"])
    return min(dates) if dates else None


def board_snapshot(boards: list[dict]) -> list[dict]:
    """Project boards onto exactly what :func:`register_events_from_boards` reads.

    Folding the snapshot gives the same events as folding the full boards,
    which ``tests/test_producer_arena.py`` asserts. That is what lets the
    committed snapshot stand in for S3 in CI.
    """
    out: list[dict] = []
    for board in sorted(boards, key=lambda b: b.get("date") or ""):
        arms = []
        for row in _board_rows(board):
            if not row.get("name"):
                continue
            arms.append(
                {
                    "name": row["name"],
                    "kind": row.get("kind"),
                    "earliest_cohort": _earliest_cohort(row),
                }
            )
        out.append({"date": board["date"], "arms": arms})
    return out


def _registered_record(name: str, created_date: str, notes: str) -> dict:
    arm_id = arm_id_for(name)
    supersedes = RESEARCH_SLOT_SUPERSEDES.get(name)
    return {
        "kind": "registered",
        "arm_id": arm_id,
        "date": created_date,
        "reason": "",
        "record": {
            "arm_id": arm_id,
            "slot": SLOT,
            "name": name,
            "spec_hash": arm_id.rsplit(":", 1)[-1],
            "created_date": created_date,
            "supersedes": arm_id_for(supersedes) if supersedes else None,
            "bootstrap": False,
            "notes": notes,
        },
    }


def _new_arm_notes(name: str, rule: str, first_board: str | None, earliest: str | None) -> str:
    if rule == CREATED_DATE_FIRST_BOARD:
        basis = (
            f"created_date is the arm's FIRST APPEARANCE on research/producer_leaderboard/ "
            f"({first_board}); earlier backfilled cohorts ({earliest or 'none'}) do not "
            "start its grace period"
        )
    else:
        basis = (
            "created_date is the EARLIEST COHORT research/producer_leaderboard/ has "
            f"scored for the arm ({earliest or first_board}), backfilled cohorts "
            f"included; it first appeared on the board on {first_board}"
        )
    parts = [f"{basis} (NEW_ARM_CREATED_DATE_RULE={rule!r})."]
    if name in RESEARCH_SLOT_ARM_WIDTHS:
        parts.append(
            f"Research slot (alpha-engine-config-I11393): pinned pre-filter "
            f"{PINNED_RESEARCH_PREFILTER}, declared width {RESEARCH_SLOT_ARM_WIDTHS[name]}."
        )
    parts.append("Recipe: crucible-research/producers/registry.py::RESEARCH_PRODUCERS.")
    return " ".join(parts)


def register_events_from_boards(
    boards: list[dict],
    *,
    existing_events: list[dict] | tuple[dict, ...] = (),
    created_date_rule: str | None = None,
    held_retirements: frozenset[str] | None = None,
) -> list[dict]:
    """Fold producer leaderboards onto the register, APPEND-ONLY.

    This is the backfill as a pure function, so the committed artifact is
    reproducible and testable rather than a hand-typed fixture.

    ``existing_events`` (the committed register) is returned first, verbatim
    and in order. The fold never rewrites or reorders an existing event. So an
    arm already registered keeps its ``created_date`` even when S3 later
    gains boards or cohorts dated earlier than it. The previous version took
    the minimum over every board on each run, which moved four registered
    arms' dates between runs (alpha-engine-config-I11490). ``created_date``
    starts the §6 grace period, so a date that moves makes retirement
    non-reproducible.

    Appended after the existing events:

    * a ``registered`` event for each arm on a board that the register does
      not hold, dated by ``created_date_rule`` (default
      :data:`NEW_ARM_CREATED_DATE_RULE`);
    * a ``registered`` event for each :data:`UNBOARDED_ARMS` seed not yet held;
    * a ``retired`` event for each registered arm a board marks
      ``kind == "retired"`` and the register has not retired, dated to the
      first such board, unless the arm is in ``held_retirements`` (default
      :data:`RETIREMENTS_HELD`).
    """
    rule = created_date_rule or NEW_ARM_CREATED_DATE_RULE
    if rule not in CREATED_DATE_RULES:
        raise ValueError(f"created_date_rule must be one of {CREATED_DATE_RULES}; got {rule!r}")
    held = RETIREMENTS_HELD if held_retirements is None else held_retirements

    events: list[dict] = [dict(e) for e in existing_events]
    registered: set[str] = set()
    retired_ids: set[str] = set()
    for event in events:
        if event.get("kind") == "registered":
            registered.add(event["record"]["name"])
        elif event.get("kind") == "retired":
            retired_ids.add(event["arm_id"])

    first_board: dict[str, str] = {}
    earliest: dict[str, str] = {}
    first_retired: dict[str, str] = {}
    for board in sorted(boards, key=lambda b: b.get("date") or ""):
        board_date = board["date"]
        for row in _board_rows(board):
            name = row.get("name")
            if not name:
                continue
            first_board.setdefault(name, board_date)
            cohort = _earliest_cohort(row)
            if cohort is not None and (name not in earliest or cohort < earliest[name]):
                earliest[name] = cohort
            if row.get("kind") == "retired":
                first_retired.setdefault(name, board_date)

    new_arms: list[dict] = []
    for name, board_date in first_board.items():
        if name in registered:
            continue
        if rule == CREATED_DATE_EARLIEST_COHORT:
            created = min(board_date, earliest.get(name, board_date))
        else:
            created = board_date
        notes = _new_arm_notes(name, rule, board_date, earliest.get(name))
        seed = UNBOARDED_ARMS.get(name)
        if seed is not None and seed[0] < created:
            created, notes = seed
        new_arms.append(_registered_record(name, created, notes))
    for name, (seed_date, provenance) in UNBOARDED_ARMS.items():
        if name not in registered and name not in first_board:
            new_arms.append(_registered_record(name, seed_date, provenance))
    new_arms.sort(key=lambda e: (e["date"], e["record"]["name"]))
    events.extend(new_arms)

    new_retirements: list[dict] = []
    for name, retired_date in first_retired.items():
        arm_id = arm_id_for(name)
        if name in held or arm_id in retired_ids:
            continue
        new_retirements.append(
            {
                "kind": "retired",
                "arm_id": arm_id,
                "date": retired_date,
                "reason": (
                    "kind=='retired' on research/producer_leaderboard/ from this date; "
                    "scored for the champion-challenger-policy.md §3 trailing window"
                ),
            }
        )
    new_retirements.sort(key=lambda e: (e["date"], e["arm_id"]))
    events.extend(new_retirements)
    return events


def append_only_violations(base_events: list[dict], events: list[dict]) -> list[str]:
    """How ``events`` fails to extend ``base_events``, or ``[]`` if it does.

    The register is append-only: every event in the base must still be there,
    unchanged and in the same position. An empty return is the only pass.
    """
    problems: list[str] = []
    if len(events) < len(base_events):
        problems.append(
            f"the register shrank from {len(base_events)} to {len(events)} events"
        )
    for i, (old, new) in enumerate(zip(base_events, events)):
        if old != new:
            problems.append(
                f"event {i} ({old.get('kind')} {old.get('arm_id')}) was rewritten: "
                f"{json.dumps(old, sort_keys=True)} -> {json.dumps(new, sort_keys=True)}"
            )
    return problems


def promotion_eligible_arm_names(register: ArmRegister | None = None) -> tuple[str, ...]:
    """Every ACTIVE arm's producer name, in register order.

    THE resolution of the arm roster for this repo. ``champion_promotion.
    VALID_CHAMPIONS`` is this, and is no longer a hand-typed tuple: the tuple
    was a second, independent register that silently omitted ``no_agent_quant``
    and ``single_agent_quant`` — the two arms with the most evidence — with
    nothing anywhere recording the omission or its reason.
    """
    reg = register or load_register()
    return tuple(reg.state(a).record.name for a in reg.active_arms())


def roster_disagreement(register: ArmRegister, leaderboard: dict | None) -> list[str]:
    """Arms the board scores that the register does not know about.

    The class defect this closes is four hand-maintained rosters drifting
    apart. One is now derived; this is the guard that the DERIVED one has not
    fallen behind its source. Returns names, never raises — the caller decides
    whether the disagreement is fatal for the cycle it is running.
    """
    if not isinstance(leaderboard, dict):
        return []
    known = {register.state(a).record.name for a in register.all_arms()}
    rows = list(leaderboard.get("arms") or leaderboard.get("specs") or [])
    return sorted({r["name"] for r in rows if isinstance(r, dict) and r.get("name")} - known)


# ── The series ────────────────────────────────────────────────────────────


def _emitted_width(name: str, row: dict, leaderboard: dict | None) -> int | None:
    """The width the board says this arm emitted: its row's ``top_n``, else ``widths``."""
    value = row.get("top_n")
    if value is None and isinstance(leaderboard, dict):
        value = (leaderboard.get("widths") or {}).get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _width_mismatch(name: str, row: dict, leaderboard: dict | None) -> str:
    """Why a research-slot arm's series may not be scored as that arm, or ``""``.

    alpha-engine-config-I11393 Amendment 2 (and -I11385 as recast): each arm
    declares its width, and the emitted width must match it. A series built at
    another width is a different recipe's track record, so it is refused
    rather than scored under this arm's id. An arm the board reports no width
    for is refused too, because the check could not be made.
    """
    declared = RESEARCH_SLOT_ARM_WIDTHS.get(name)
    if declared is None:
        return ""
    emitted = _emitted_width(name, row, leaderboard)
    if emitted == declared:
        return ""
    return (
        f"declared width {declared} (RESEARCH_SLOT_ARM_WIDTHS) but "
        f"research/producer_leaderboard/ reports it emitted "
        f"{'no width' if emitted is None else emitted}. An arm's width is part of "
        "its immutable recipe (alpha-engine-config-I11393 Amendment 2), so a "
        "series at another width is not this arm's record and is not scored"
    )


def arm_statistics(
    register: ArmRegister, leaderboard: dict | None, as_of: str,
) -> dict[str, dict[str, Any]]:
    """Per-arm width, information ratio and rank IC, as the board reports them.

    REPORTED, never decided on. The pointer is decided by the engine on the IR
    it computes itself from the paired per-date series (``promote_statistic``).
    These are the board's whole-history figures, written beside the decision
    so a reader can see whether an arm's IR came from ranking skill
    (``realized_rank_ic``, which does not depend on width) or from its breadth
    (alpha-engine-config-I11393 Amendment 2: "decide the pointer on IR, report
    IC beside it"). A value the board does not carry is ``None``, never 0.
    """
    rows_by_name: dict[str, dict] = {}
    if isinstance(leaderboard, dict):
        for row in leaderboard.get("specs") or []:
            if isinstance(row, dict) and row.get("name"):
                rows_by_name[row["name"]] = row
    out: dict[str, dict[str, Any]] = {}
    for arm_id in register.scored_arms(as_of, ARENA_CONFIG.retired_trailing_cycles):
        name = register.state(arm_id).record.name
        row = rows_by_name.get(name) or {}
        out[name] = {
            "arm_id": arm_id,
            "declared_width": RESEARCH_SLOT_ARM_WIDTHS.get(name),
            "emitted_width": _emitted_width(name, row, leaderboard) if row else None,
            "information_ratio": row.get("information_ratio"),
            "realized_rank_ic": row.get("realized_rank_ic"),
        }
    return out



def build_series(
    register: ArmRegister, leaderboard: dict | None, as_of: str,
) -> tuple[dict[str, ArmSeries], list[SeriesGap]]:
    """Per-date POPULATION-relative scores for every arm the cycle must score.

    Covers exactly ``register.scored_arms(as_of, retired_trailing_cycles)`` —
    active arms plus retired arms inside their §3 trailing window — because
    ``run_cycle`` RAISES on a missing series and RAISES on a series for an
    unregistered arm. Both raises are the point: the first stops an arm
    quietly dropping out of the contest, the second is the
    ``thinktank_coverage`` defect (output written with no register row, data
    rotting unnoticed).

    Every arm routes through THIS one path — the champion included. There is
    no per-arm scorer, no arm scored from a different source on a different
    cohort, and therefore no way for one arm's silence to render as another
    arm's thinness. That asymmetry is what hid the champion's two-date cohort
    behind two challengers' six.
    """
    rows_by_name: dict[str, dict] = {}
    if isinstance(leaderboard, dict):
        for row in leaderboard.get("specs") or []:
            if isinstance(row, dict) and row.get("name"):
                rows_by_name[row["name"]] = row

    series: dict[str, ArmSeries] = {}
    gaps: list[SeriesGap] = []
    for arm_id in register.scored_arms(as_of, ARENA_CONFIG.retired_trailing_cycles):
        name = register.state(arm_id).record.name
        row = rows_by_name.get(name)
        if row is None:
            gaps.append(
                SeriesGap(
                    name,
                    arm_id,
                    "not present in research/producer_leaderboard/ specs — the arm is "
                    "registered but the board built no history for it (no "
                    "signals_shadow/ writer, or the board predates its registration)",
                )
            )
            series[arm_id] = ArmSeries(arm_id=arm_id, scores={}, misses=frozenset())
            continue
        dates_scored = frozenset(
            d for d in (row.get("dates_scored") or []) if isinstance(d, str)
        )
        width_problem = _width_mismatch(name, row, leaderboard)
        if width_problem:
            gaps.append(SeriesGap(name, arm_id, width_problem))
            series[arm_id] = ArmSeries(arm_id=arm_id, scores={}, misses=dates_scored)
            continue
        by_date = row.get(POPULATION_SERIES_FIELD)
        if not isinstance(by_date, dict) or not by_date:
            gaps.append(
                SeriesGap(
                    name,
                    arm_id,
                    f"research/producer_leaderboard/ carries no {POPULATION_SERIES_FIELD!r} "
                    "for this arm; it publishes only the aggregate "
                    "topn_alpha_vs_population. A confidence sequence cannot be formed "
                    "from cumulative means, and the SPY-relative "
                    "topn_alpha_vs_benchmark is REFUSED for a selection-stage slot "
                    "(SPY trailed the drawn-from population by 140bp at 21d on "
                    "2026-08-17, which inverts wins and losses). Upstream one-field "
                    "emission tracked as alpha-engine-config-I9405",
                )
            )
            series[arm_id] = ArmSeries(arm_id=arm_id, scores={}, misses=dates_scored)
            continue
        scores = {d: float(v) for d, v in by_date.items() if v is not None}
        series[arm_id] = ArmSeries(
            arm_id=arm_id,
            scores=scores,
            misses=frozenset(dates_scored - set(scores)),
        )
    return series, gaps


# ── Serving preconditions ─────────────────────────────────────────────────


def build_preconditions(
    register: ArmRegister,
    series_by_arm: dict[str, ArmSeries],
    *,
    shadow_only_names: frozenset[str],
    feed_blocked_names: dict[str, str] | None = None,
) -> dict[str, tuple[ServingPrecondition, ...]]:
    """The slot's hard gates on SERVING, evaluated here and passed in.

    Three, each a per-arm FACT recorded with its reason on the artifact
    rather than a hidden veto: the arm is declared shadow-only (measured,
    never served); the arm's live-trade feed producer looks dead; the frozen
    pointer contract does not admit the arm as a ``champion`` value
    (:data:`POINTER_ADMISSIBLE_ARMS`).

    The third one passes for every arm registered today, and that is a
    measurement rather than a design flaw — every arm on the board is in the
    enum as of ``alpha-engine-config-I9299``. It is reachable and it fires:
    an arm that appears on the producer board before this repo widens the
    enum is held off the pointer and named, which is exactly the sequence
    I9299 was filed for after the executor was found to have no handler for
    two arms Brian had already ruled eligible.

    These are the ONLY things that stop an arm taking the pointer. There is
    no hysteresis margin, no cooldown, and no evidence floor here — the
    confidence sequence is the evidence bar and the pointer moves freely in
    both directions (Brian ruling 2026-08-29, policy §5.2).
    """
    blocked = feed_blocked_names or {}
    out: dict[str, tuple[ServingPrecondition, ...]] = {}
    for arm_id in series_by_arm:
        name = register.state(arm_id).record.name
        out[arm_id] = (
            ServingPrecondition(
                name="not_shadow_only",
                passed=name not in shadow_only_names,
                reason=(
                    "" if name not in shadow_only_names
                    else f"{name} is declared shadow-only: measured every cycle, never served"
                ),
            ),
            ServingPrecondition(
                name="feed_producer_live",
                passed=name not in blocked,
                reason=blocked.get(name, ""),
            ),
            ServingPrecondition(
                name="pointer_contract_admits",
                passed=name in POINTER_ADMISSIBLE_ARMS,
                reason=(
                    "" if name in POINTER_ADMISSIBLE_ARMS
                    else (
                        f"contracts/producer_champion.schema.json does not admit {name!r} as a "
                        f"`champion` value (admitted: {sorted(POINTER_ADMISSIBLE_ARMS)}). The "
                        "executor fail-louds on an unrecognized pointer and refuses to start "
                        "a planning cycle, so moving the pointer here would halt trading. "
                        "Widening the enum is the deliberate act that admits the arm"
                    )
                ),
            ),
        )
    return out


# ── The cycle ─────────────────────────────────────────────────────────────


def run_arena_cycle(
    *,
    as_of: str,
    leaderboard: dict | None,
    incumbent_name: str | None,
    shadow_only_names: frozenset[str],
    feed_blocked_names: dict[str, str] | None = None,
    register: ArmRegister | None = None,
) -> tuple[ArenaCycle, list[SeriesGap], ArmRegister]:
    """One evaluation cycle of the selection-producer slot.

    ``training=None``: this slot's arms are selection RECIPES, not fitted
    models — there is no per-arm fit for the slot to vouch for, and asserting
    a training status it cannot observe would be provenance that is not true
    by construction (§7.5).
    """
    reg = register or load_register()
    unknown = roster_disagreement(reg, leaderboard)
    if unknown:
        # Fail LOUD on a producer. A board scoring an arm this repo has never
        # heard of means the derived roster has fallen behind its source, and
        # every comparison this cycle would be taken over an incomplete pool.
        raise ValueError(
            f"research/producer_leaderboard/ scores arm(s) {unknown} that "
            f"{REGISTER_PATH.name} does not register. The register is DERIVED from that "
            "board and has fallen behind it; re-run "
            "scripts/backfill_producer_arena_register.py and commit the result. "
            "Deciding the live pointer over an incomplete pool is the defect "
            "champion-challenger-policy.md §3 exists to prevent."
        )

    series_by_arm, gaps = build_series(reg, leaderboard, as_of)
    incumbent_id = arm_id_for(incumbent_name) if incumbent_name else None
    if incumbent_id is not None and incumbent_id not in series_by_arm:
        # A pointer sitting on an arm no longer in the scored set (retired
        # past its trailing window, or removed upstream). Not silently
        # bootstrapped away: `decide_pointer` handles `incumbent not in
        # series` as the §9.1 bootstrap path and says so on the artifact.
        logger.warning(
            "[producer_arena] incumbent %r is not in this cycle's scored set — the "
            "engine will take the §9.1 bootstrap path and the artifact will say so",
            incumbent_name,
        )
    cycle = run_cycle(
        config=ARENA_CONFIG,
        as_of=as_of,
        register=reg,
        series_by_arm=series_by_arm,
        incumbent=incumbent_id,
        preconditions=build_preconditions(
            reg,
            series_by_arm,
            shadow_only_names=shadow_only_names,
            feed_blocked_names=feed_blocked_names,
        ),
        training=None,
    )
    return cycle, gaps, reg


def cycle_document(
    cycle: ArenaCycle,
    gaps: list[SeriesGap],
    statistics: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The durable artifact body, validated against the ``arena_cycle`` contract.

    Validation happens HERE — on the producer side, before the write — so a
    contract break surfaces at the earliest call site rather than in a
    consumer weeks later (M0 contract discipline).
    """
    doc = cycle.to_dict()
    validate_contract("arena_cycle", doc)
    # Additive, after validation so it can never be the reason a valid cycle
    # is refused: the arms this cycle could not score, by name and reason.
    # `scored_arms` alone cannot express "present in the roster, no series".
    doc["series_gaps"] = [g.to_dict() for g in gaps]
    # Additive for the same reason: each arm's declared and emitted width, IR
    # and rank IC, reported beside the decision (see arm_statistics).
    if statistics is not None:
        doc["arm_statistics"] = statistics
    return doc


def write_arena_cycle(
    bucket: str, as_of: str, doc: dict[str, Any], *, upload: bool, s3_client=None,
) -> str | None:
    """Write ``arena/producer/{date}.json`` + ``latest.json``.

    RAISES on failure. §11: a slot that emits nothing is not healthy, it is
    unobserved — a swallowed write here would make an unrun cycle and a
    silent one indistinguishable, which is the exact class this artifact
    exists to retire.
    """
    dated_key = f"{ARENA_CYCLE_PREFIX}/{as_of}.json"
    if not upload:
        logger.info("[producer_arena] arena_cycle write skipped (upload=False): %s", dated_key)
        return None
    s3 = s3_client or boto3.client("s3")
    body = json.dumps(doc, indent=2, allow_nan=False).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=dated_key, Body=body, ContentType="application/json")
    s3.put_object(
        Bucket=bucket, Key=f"{ARENA_CYCLE_PREFIX}/latest.json", Body=body,
        ContentType="application/json",
    )
    logger.info("[producer_arena] arena_cycle written: s3://%s/%s (+ latest.json)", bucket, dated_key)
    return dated_key


#: Decision statuses that mean the slot could not decide on evidence. §11:
#: both are first-class, both carry a reason, and both ALARM.
ALARMING_STATUSES = ("unmeasurable", "unservable")


def publish_cycle_alert(cycle: ArenaCycle, gaps: list[SeriesGap]) -> None:
    """Alarm on an ``unmeasurable`` / ``unservable`` cycle (§11).

    Best-effort: an alerting failure must not red the weekly pipeline it
    reports on. It cannot become a second silence — the status and every gap
    are already durable on the ``arena_cycle`` artifact before this runs, and
    the failure is logged with a traceback.
    """
    if cycle.decision.status not in ALARMING_STATUSES:
        return
    detail = "; ".join(f"{g.arm_name}: {g.reason}" for g in gaps) or "no per-arm gap recorded"
    message = (
        f"producer arena cycle {cycle.as_of} is {cycle.decision.status}: "
        f"{cycle.decision.reason}. The live config/producer_champion.json pointer is "
        f"HELD at {cycle.decision.champion!r}. Per-arm gaps: {detail}. "
        f"See s3 {ARENA_CYCLE_PREFIX}/{cycle.as_of}.json."
    )
    try:
        from ops_alerts import publish_ops_alert

        publish_ops_alert(
            message,
            severity="error",
            # `alpha-engine-backtester/`, not `crucible-backtester/`: the
            # canonical alert-source prefix the overseer's `alert_classes`
            # registry keys on (alpha-engine-config-I3302 completed that
            # rename). A second prefix for the same repo would land the class
            # in the registry twice under two names.
            source="alpha-engine-backtester/optimizer/producer_arena.py::run_arena_cycle",
            dedup_key=f"producer_arena_{cycle.decision.status}_{cycle.as_of}",
            dedup_window_min=720,
        )
    except Exception:  # noqa: BLE001 — alerting must never crash the weekly run
        logger.exception(
            "[producer_arena] %s alert publish failed (best-effort); the status and "
            "every gap remain durable on the arena_cycle artifact",
            cycle.decision.status,
        )
