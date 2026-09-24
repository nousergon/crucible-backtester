"""champion_promotion.py — the selection-producer slot's pointer.

**The decision is not taken here.** ``nousergon_lib.arena.engine.run_cycle``
takes it, once, from one ``ArenaCycle``; this module owns the three artifacts
around it and nothing about the rule itself
(``optimizer/producer_arena.py`` is the slot's wiring, and
``champion-challenger-policy.md`` §§3–6 is the rule). What lives here:

  1. ``config/producer_champion.json`` — THE single writer for the live
     pointer the executor's ``executor/champion.py::load_champion_pointer``
     reads at planner start. Three invariants, all fail-loud: the arm is in
     the derived roster, it is not declared shadow-only, and the frozen
     ``contracts/producer_champion.schema.json`` enum admits it.
  2. ``config/apply_audit/producer_champion/{date}.json`` (+ ``latest.json``)
     — the ``producer_champion_audit`` v2 record, written UNCONDITIONALLY
     every week including on ``outcome="error"``. This is the liveness proxy
     (config#2054 lesson): pointer mtime cannot prove the engine is alive
     because a correctly-held week does not touch it.
  3. ``research/producer_leaderboard_champion_gate/{date}.json`` — this
     module's own observability history (config#2367, key deliberately
     distinct from crucible-research's under config#2452). No longer feeds
     any decision.

**alpha-engine-config-I9318 (2026-08-29) deleted this module's decision
engine.** Everything below is gone, not refactored:

  - the winner-take-all comparison (I2518) and, before it, the
    HAC-significance / two-week-hysteresis / two-week-cooldown gate
    (config#2367);
  - the per-arm evidence floors — ``thin_evidence``,
    ``MIN_CYCLES_FOR_INFERENCE``, ``confidence != "ok"``,
    ``leaderboard_row_confidence`` and the champion-side half of the same
    (I7542/I7549). The anytime-valid confidence sequence IS the evidence bar
    (policy §5.0), and a floor on top of it is a second, weaker bar;
  - the three per-arm scorers reading three different sources on three
    different cohorts. THAT asymmetry is the defect: the incumbent was
    scored from this repo's own ``sector_neutral_mean_alpha_21d``
    counterfactual while challengers were scored from crucible-research's
    board, so the champion's own thin weeks could never render as thinness —
    they rendered as another arm's. Every arm now routes through
    ``producer_arena.build_series``: one board, one cohort, one benchmark.

Two guards survived, because they are about WHICH measurement is in front of
us rather than how much of it there is: ``LEADERBOARD_STALENESS_DAYS`` (8)
and ``GATE_HORIZON_DAYS`` (21, the board's primary horizon — §4 requires one
horizon across a slot and §3 requires a promoted arm's series to stay
continuous). Both raise ``LeaderboardUnusable`` and produce a classified
``outcome="error"`` rather than a silently rescored pointer.

**The roster is DERIVED.** ``VALID_CHAMPIONS`` was a hand-typed tuple — one
of four independent arm registers in the fleet — and it silently omitted
``no_agent_quant`` and ``single_agent_quant``, the two arms with the most
evidence on the board, with nothing anywhere recording the omission. It is
now ``producer_arena.promotion_eligible_arm_names()``, off the slot's durable
register, and ``producer_arena.roster_disagreement`` raises when the
derivation falls behind its source. ``agentic`` remains READ-TOLERATED (a
historical pointer or audit value must never crash this engine —
``_normalize_champion_before`` WARNs and treats it as the base-case arm) and
WRITE-FORBIDDEN.

**``arena_cycle`` is authoritative; this audit record is a PROJECTION of it.**
``decision_record_from_cycle`` computes nothing — it renames. The projection
is lossy in exactly one direction and only because the frozen contract forces
it: the audit enum admits four arm names and the roster has five, so a
promotion onto ``no_agent_quant`` or ``single_agent_quant`` is recorded with
``champion_after: null`` (never ``champion_before``, which would assert the
pointer did not move) plus a WARN, while the open ``arm_scores`` map carries
every arm's number and ``arena_cycle`` names the arm outright. Retiring this
record in favour of ``arena_cycle`` is option (B) of
alpha-engine-config-I9406. ``_reconcile_pointer`` reads the ENGINE's verdict,
never this projection, so the live pointer is never narrowed by a rendering.

**Pointer PROVENANCE is now maintained every cycle (I9318's ``closes-when``).**
``config/producer_champion.json`` read ``promotion_source:
"operator_bootstrap"`` from 2026-07-13 through 2026-08-29 while an automated
engine evaluated it weekly. The value was right and the record of how it got
there was false, and no surface said so. Every cycle now sets
``promotion_source`` to ``arena_decided``/``arena_held``/
``arena_unmeasurable``/``arena_unservable``/``arena_bootstrap`` and PRESERVES
``promoted_at`` whenever the champion did not change, so "when did the
pointer last move" survives the correction.

**Shadow-only arms** (Brian 2026-08-20, released 2026-08-27) are measured
every cycle and never served. ``SHADOW_ONLY_ARMS`` is EMPTY today; the
mechanism stays, enforced at two layers — ``producer_arena.build_preconditions``
(the policy: a named, recorded ineligibility on the artifact) and
``write_champion_pointer`` (the invariant: nothing reaching S3 can violate
it) — so a future arm inherits the protection by joining that frozenset and
nothing else.

**Promotion-time feed liveness (alpha-engine-config-I3165)** is now a per-arm
SERVING PRECONDITION, probed for every arm in ``ARM_FEED_DEPENDENCIES``
before the cycle rather than for "the challenger that would win" — a question
with no answer in an N-arm slot before the engine has decided. Note the
consequence, which is a real broadening: an unservable INCUMBENT no longer
parks the pointer on an arm that cannot trade (the config#3053 state); the
engine forces the pointer to the best eligible arm and says so.

``hac_significance`` (Newey-West/HAC overlap-aware significance) is RETAINED
below, unchanged and independently unit-tested. It is not wired into any
decision and is kept as an available diagnostic.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import date, datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError

from optimizer import champion_digest, producer_arena

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1          # config/producer_champion.json pointer — unchanged shape
AUDIT_SCHEMA_VERSION = 2    # config/apply_audit/producer_champion/{date}.json — v2 shape (I2518)
POINTER_KEY = "config/producer_champion.json"
AUDIT_PREFIX = "config/apply_audit/producer_champion"

# ── The arm roster — DERIVED, never typed (alpha-engine-config-I9318) ─────
#
# alpha-engine-config-I8756 — Brian's ruling 2026-08-27:
#
#     "I want the champion/challenger for research to include the scanner top 20
#      (not top 60) passed directly to the predictor. I want all other
#      challengers to remain challengers such as think tank and other scanner
#      configurations. But at this point I'm thinking we promote the best
#      performer weekly"
#
# This WAS a hand-typed tuple of three arm names. It was a SECOND, independent
# arm register — the fleet carried four of them (this tuple, crucible-executor's
# `executor/champion.py::VALID_CHAMPIONS`, crucible-research's
# `producers/registry.py::RESEARCH_PRODUCERS`, and a test asserting this tuple
# as a literal) — and it silently omitted `no_agent_quant` and
# `single_agent_quant`, the two arms with the MOST evidence on the board
# (6 scored cohort dates each against the incumbent's 5, measured 2026-08-29).
# Nothing anywhere recorded that they were excluded, or why: they were simply
# not typed here. "Scored but ineligible, for no recorded reason" is the same
# silent-omission class alpha-engine-config-I9277 closed one hop upstream,
# recurring here.
#
# It is now DERIVED from the slot's durable arm register
# (`optimizer/arena/producer_register.json`), itself derived from
# crucible-research's producer leaderboard. `producer_arena.roster_disagreement`
# RAISES when the derived roster falls behind its source, so drift becomes a
# loud failure rather than an arm quietly dropping out of the contest.
#
# ORDER IS NOT PRECEDENCE — the arena engine ranks by Copeland score and picks
# the largest supported lead, never by position in this tuple.
VALID_CHAMPIONS: tuple[str, ...] = producer_arena.promotion_eligible_arm_names()

# The arm an unrecognized or legacy pointer normalizes to FOR GATE PURPOSES.
#
# Named explicitly rather than taken as `DEFAULT_GATE_CHAMPION`, which is what it
# used to be. Under two arms those were the same value by accident; adding a
# third made TUPLE POSITION silently decide which arm a legacy `agentic`
# pointer is treated as — so reordering the tuple, a cosmetic edit, would have
# changed which arm a stale pointer normalizes to. Caught by
# `TestShadowOnlyEndToEnd` before this shipped, not by inspection.
#
# It is the BASE-CASE arm: the one live since 2026-07-13, not the one the
# 2026-08-27 ruling adds. A normalization is a guess about what a stale pointer
# meant, and the only defensible guess is the incumbent.
DEFAULT_GATE_CHAMPION = "scanner_predictor_direct"

# Retired seat(s) — READ-TOLERATED (a historical pointer/audit artifact using
# this value must never crash the engine) but WRITE-FORBIDDEN (excluded from
# VALID_CHAMPIONS, so write_champion_pointer raises on it).
_LEGACY_CHAMPIONS = ("agentic",)

# ── Shadow-only arms — the single source of truth for "measured, never
# promoted" (Brian's ruling, 2026-08-20, recorded on
# alpha-engine-config-I2515) ──────────────────────────────────────────────
#
#     "research should now be think tank in shadow mode only with the main
#     research process skipped by passing a scanner top 20 to predictor"
#                                                     — Brian, 2026-08-20
#
# An arm listed here is scored every week exactly like any other arm
# (champion-challenger-policy.md §3: measurement is unconditional and is
# NOT what promotion governs), keeps its leaderboard row, and this gate
# keeps recording that it WOULD have won — but it may never take the live
# `config/producer_champion.json` pointer by winning on score. Promoting a
# shadow-only arm requires its own separate ruling from Brian; executing
# that ruling is a ONE-LINE data change (remove the arm from this
# frozenset) and nothing else.
#
# Shadow-only-ness is deliberately a PROPERTY OF AN ARM declared once here,
# not a string literal at the veto site: a future arm introduced in shadow
# mode inherits the protection by being added to this set, at both
# enforcement layers (evaluate_gates, the policy; write_champion_pointer,
# the invariant) simultaneously. Membership here is INDEPENDENT of
# VALID_CHAMPIONS — a shadow-only arm is a legal pointer VALUE to read
# (the executor has a live consumer for it, executor/champion.py::
# _apply_thinktank_coverage) and a legal arm to score; it is simply not a
# legal arm for this engine to promote.
# **Brian's ruling 2026-08-27 SUPERSEDES the 2026-08-20 shadow-only ruling**
# recorded above, on the single question of whether `thinktank_coverage` may
# take the pointer:
#
#     "I want all other challengers to remain challengers such as think tank and
#      other scanner configurations. But at this point I'm thinking we promote
#      the best performer weekly"
#                                                     — Brian, 2026-08-27
#
# An arm that can never win is an observation, not a challenger. The set is now
# EMPTY, which is the honest way to say "every registered arm can take the seat
# this week."
#
# The SET AND BOTH ENFORCEMENT LAYERS STAY. They are the mechanism, not the
# membership: a future arm introduced in shadow mode inherits the protection at
# `evaluate_gates` (the policy) and `write_champion_pointer` (the invariant)
# simultaneously, by being added here and nowhere else. Deleting the mechanism
# because it is currently unused is how it would have to be rebuilt — and
# rebuilt at one layer instead of two — the next time an arm needs it.
SHADOW_ONLY_ARMS: frozenset[str] = frozenset()


def is_shadow_only(arm: str | None) -> bool:
    """True when ``arm`` is declared shadow-only (measured, never promoted).
    The ONLY way any code in this module asks that question — see
    ``SHADOW_ONLY_ARMS`` for the ruling and the one-line unblock."""
    return arm in SHADOW_ONLY_ARMS


OUTCOMES = (
    "promoted",
    "no_contest",
    "unchanged_winner_already_champion",
    # alpha-engine-config-I2515 (2026-08-20 shadow-only ruling): the winner
    # on score alone was a shadow-only arm, so the pointer was deliberately
    # held. Deliberately NOT reused from the existing vocabulary:
    # "no_contest" asserts the week produced no comparable evidence, which
    # would be false — there WAS a contest and the shadow arm won it; and
    # "unchanged_winner_already_champion" asserts the incumbent scored
    # highest, which would also be false. Both would erase the very
    # counterfactual shadow mode exists to measure. The record carries
    # `counterfactual_winner` naming who actually won.
    "held_shadow_only",
    "error",
)

# blocked_by slugs — union of the current winner-take-all vocabulary and two
# retired vocabularies kept for read-tolerance of historical audit records:
# the pre-I2518 HAC/hysteresis/cooldown engine, and the pre-I2544
# exact-date-only leaderboard read (superseded same-day by the
# latest-available read below; no code path in this module writes either
# retired group again). Slug vocabulary is unchanged by the I2998 direct
# -lift rescoring — only the underlying score SOURCE field changed per arm,
# not the failure-mode taxonomy.
_BLOCKED_BY_SLUGS = (
    # current (winner-take-all + latest-available leaderboard read + direct
    # per-arm lift scoring, I2518/I2544/I2998)
    "no_valid_scanner_predictor_direct_selections",
    "no_valid_thinktank_coverage_selections",
    "scanner_predictor_direct_counterfactual_unavailable",
    "thinktank_coverage_not_in_leaderboard",
    "thinktank_coverage_no_resolved_outcomes",
    "thinktank_coverage_thin_evidence",
    "thinktank_coverage_confidence_unknown",
    # The same two verdicts on the champion side of the comparison
    # (alpha-engine-config-I7549 champion-side half). Separate slugs per arm
    # because the operator response differs by arm: a thin leaderboard row
    # waits on crucible-research's cohort, a thin counterfactual waits on this
    # repo's own research.db cycles.
    "scanner_predictor_direct_thin_evidence",
    "scanner_predictor_direct_confidence_unknown",
    # alpha-engine-config-I8756 — the third arm's own slugs. Separate per arm
    # because the operator response differs: a thin top-20 arm waits on the
    # 21-day maturation of a cut that only began on 2026-07-30, which is time,
    # not a defect to chase.
    "scanner_top20_predictor_counterfactual_unavailable",
    "scanner_top20_predictor_thin_evidence",
    "scanner_top20_predictor_confidence_unknown",
    "leaderboard_unavailable",
    "leaderboard_stale_gt_8d",
    "leaderboard_horizon_mismatch",
    "arm_score_unavailable",
    "feed_producer_dead",
    # alpha-engine-config-I2515 (2026-08-20 shadow-only ruling): the arm
    # that won on score is declared in SHADOW_ONLY_ARMS — measured, never
    # promoted. Paired with outcome="held_shadow_only" and the record's
    # `counterfactual_winner` field.
    "shadow_only_arm",
    "frozen",
    "unclassified_error",
    # retired (pre-I2518 HAC/hysteresis/cooldown engine) — historical read-only
    "insufficient_matured_cohorts",
    "cooldown_active",
    "not_significant_hac_adjusted",
    "hysteresis_not_satisfied",
    # retired (pre-I2544 exact-date-only leaderboard read) — historical
    # read-only: this slug fired when the artifact's self-reported "date"
    # field disagreed with an exact-match run_date key read; the
    # latest-available read no longer requires an exact match, so this
    # condition can no longer occur (superseded by leaderboard_stale_gt_8d
    # for the age-bound case).
    "leaderboard_stale",
)

# Honest staleness bound (alpha-engine-config-I2544, 2026-07-14): a selected
# research/producer_leaderboard/{date}.json artifact older than this many
# calendar days relative to run_date is treated as unavailable rather than
# scored — see find_latest_research_producer_leaderboard_date /
# _score_thinktank_coverage.
LEADERBOARD_STALENESS_DAYS = 8

# ── The per-arm evidence verdict carried on the audit record ──────────────
#
# The `ok`/`thin`/`insufficient`/`unknown`/`unrecognised` vocabulary is
# RETIRED as an emitted value (alpha-engine-config-I9318). It existed to say
# how far an arm was from a minimum-cohort floor, and there is no floor any
# more: the anytime-valid confidence sequence IS the evidence bar, and it
# expresses "not enough evidence yet" as a wide interval rather than as a
# refusal to compare (champion-challenger-policy.md §5.0). Keeping a
# thinness verdict on the record would describe a gate that no longer runs.
#
# Two values remain, and they answer the only question the record can still
# honestly answer: did this arm have a series to be scored on at all?
# The old vocabulary stays read-tolerated for historical audit records —
# the upstream contract's `arm_confidence` is an open string map.
CONFIDENCE_MEASURED = "measured"
CONFIDENCE_UNAVAILABLE = "unavailable"

# The horizon this gate decides on, in trading sessions. See the module
# docstring's HORIZON section: 21 is the leaderboard's PRIMARY horizon (whose
# block is spread across the artifact's top level) and the only horizon at
# which BOTH arms are scored. Named here so the choice is an assertion rather
# than an accident of which block happens to sit at the top level.
GATE_HORIZON_DAYS = 21

RESEARCH_PRODUCER_LEADERBOARD_PREFIX = "research/producer_leaderboard/"
_RESEARCH_PRODUCER_LEADERBOARD_KEY_RE = re.compile(
    r"^research/producer_leaderboard/(\d{4}-\d{2}-\d{2})\.json$"
)

# ── Promotion-time feed-dependency liveness gate (alpha-engine-config-I3165,
# 2026-07-23) ─────────────────────────────────────────────────────────────
#
# config#3053 root cause: scanner_predictor_direct was promoted 2026-07-13
# with its live-trade feed chain (research_free_backfill, itself sourced
# from scanner_evaluations) undeclared anywhere in the promotion record —
# config#1580 orphaned that chain's ultimate upstream producer the very
# next day, and nothing at promotion time (or afterward, until the
# freshness monitor's config-I3086 critical_while_champion_arm mechanism)
# checked it. That mechanism is ONGOING monitoring, wired to the ARTIFACT
# _REGISTRY row's static severity being coerced dynamic while a listed arm
# is champion; THIS gate is the complementary PROMOTION-TIME check, run
# once per weekly evaluation, at the moment a challenger would newly become
# champion.
#
# ARM_FEED_DEPENDENCIES is the small, static, source-of-truth mapping this
# repo previously lacked entirely (no arm->feed mapping existed anywhere —
# scanner_predictor_direct's chain was documented only in this module's own
# sibling analysis/scanner_predictor_research_free_backfill.py docstring).
# Each value is a list of ARTIFACT_REGISTRY.yaml artifact_ids (alpha-engine
# -config/private-docs/ARTIFACT_REGISTRY.yaml) — deliberately the DIRECT
# feed the arm's live trading reads, not its deeper transitive upstreams
# (research_free_backfill's own upstream, scanner_evaluations, has no
# ARTIFACT_REGISTRY row at all as of this writing — registering it is a
# separate, out-of-scope gap). thinktank_coverage carries no entry (and
# none is required): its evidence chain is the producer leaderboard, already
# gated above by leaderboard_date_used/leaderboard_stale_gt_8d — it names no
# live-trade feed artifact of its own.
ARM_FEED_DEPENDENCIES: dict[str, list[str]] = {
    "scanner_predictor_direct": ["research_free_backfill"],
}

# artifact_id -> liveness prober. Each prober takes (bucket, run_date,
# s3_client) and RAISES on a dead/stale/missing/unreadable producer, returns
# None (no exception) when live — the same shape
# assert_champion_feed_fresh already uses. Deliberately reuses that
# existing, already-tested producer-side check (config#3053) rather than
# hand-rolling a second freshness reader for the same artifact: it already
# encodes the correct content-derived staleness rule for
# research_free_backfill (newest prediction_date vs run_date, not S3
# LastModified, which a no-op rewrite would falsely refresh). New feed
# dependencies added to ARM_FEED_DEPENDENCIES in the future need a prober
# registered here (or, if a cheap presence/HEAD check suffices, a lighter
# adapter) -- an arm whose feed has no registered prober here is simply not
# checked (fails open on THIS gate; the config-I3086 ongoing monitor still
# covers it once promoted) rather than crashing the run.
def _check_research_free_backfill_live(bucket: str, run_date: str, s3_client) -> None:
    from analysis.scanner_predictor_research_free_backfill import assert_champion_feed_fresh

    assert_champion_feed_fresh(bucket, run_date=run_date, s3_client=s3_client)


_FEED_LIVENESS_PROBES = {
    "research_free_backfill": _check_research_free_backfill_live,
}


def check_feed_dependencies_live(
    arm: str, *, bucket: str, run_date: str, s3_client=None,
) -> str | None:
    """Probe every feed artifact ``arm`` declares in
    ``ARM_FEED_DEPENDENCIES`` for producer liveness. Returns
    ``"feed_producer_dead"`` (the ``blocked_by`` slug) the first time a
    declared dependency's registered prober raises anything at all;
    returns ``None`` when ``arm`` declares no dependencies, every declared
    dependency has no registered prober, or every registered prober passed.

    Never raises — a probe failure (dead feed, unreadable artifact, an
    unexpected exception in the prober itself) must degrade the gate to a
    no-contest, exactly like every other validity guard in this module
    (module docstring's binding config#2884 lesson: an error here must
    never silently default to a promotion, and must never crash the weekly
    evaluation either).
    """
    for feed_id in ARM_FEED_DEPENDENCIES.get(arm, []):
        probe = _FEED_LIVENESS_PROBES.get(feed_id)
        if probe is None:
            logger.warning(
                "champion_promotion: %r declares feed dependency %r with no "
                "registered liveness prober — skipping (not checked by this "
                "gate; add a _FEED_LIVENESS_PROBES entry to cover it)",
                arm, feed_id,
            )
            continue
        try:
            probe(bucket, run_date, s3_client)
        except Exception as e:  # noqa: BLE001 — a dead/unreadable feed (or any
            # unexpected prober failure) degrades this promotion to
            # no_contest; it must never crash the weekly evaluation and
            # must never be silently swallowed into a promotion either.
            logger.warning(
                "champion_promotion: feed dependency %r for arm %r looks "
                "dead/orphaned at promotion time (%s) — blocking this "
                "promotion (feed_producer_dead)", feed_id, arm, e,
            )
            return "feed_producer_dead"
    return None


# HAC lag helper constants — still consulted by hac_significance() below,
# which is retained as an independent, tested utility (see module docstring)
# even though it no longer gates the promotion decision.
_HORIZON_DAYS = 21              # 21d forward-alpha horizon (config#1405 basis)
_CADENCE_DAYS = 7               # weekly evaluation cadence

_cfg: dict = {}


def init_config(config: dict) -> None:
    """Called unconditionally by evaluate.py at optimizer-stage start. The
    winner-take-all engine currently defines no configurable thresholds
    (no significance level, no hysteresis/cooldown weeks) — this is kept as
    a stable entry point for evaluate.py's wiring and for hac_significance's
    still-configurable horizon/cadence (lag = round(horizon/cadence))."""
    global _cfg
    _cfg = config.get("champion_promotion", {})


def _hac_lag() -> int:
    """Bartlett-kernel lag = round(horizon_days / cadence_days). Feeds
    ``hac_significance`` only — not load-bearing for the winner-take-all
    decision itself."""
    horizon = int(_cfg.get("horizon_days", _HORIZON_DAYS))
    cadence = int(_cfg.get("cadence_days", _CADENCE_DAYS))
    return max(0, round(horizon / cadence))


# ── Retained utility: HAC/Newey-West-adjusted overlap-aware significance ───
# Not wired into evaluate_gates() under the winner-take-all policy (see
# module docstring) — kept as an independently unit-tested, available
# diagnostic.


def hac_significance(
    weekly_sn_lift: list[float], *, alpha: float = 0.05,
) -> dict:
    """Overlap-aware two-sided significance test of whether a weekly
    sector-neutral lift series is significantly different from zero, using
    the Newey-West (1994) HAC standard error of the mean (Bartlett kernel;
    lag = ``_hac_lag()``) in place of the naive i.i.d. ``s/sqrt(n)`` standard
    error. See the module history (git log) for the full derivation this
    docstring previously carried in-line; unchanged behavior, retained as an
    available utility (not currently gate-connected — see module docstring).

    Uses the vendored, independently-unit-tested
    ``nousergon_lib.quant.stats.intervals.newey_west_se``.

    Returns a dict:
      ``{"status": "ok", "n": int, "mean": float, "se": float, "lags": int,
         "t_stat": float, "p_value": float, "significant": bool}``
      or ``{"status": "insufficient_data", "n": int}`` when fewer than 2
      finite observations are available.
    """
    from nousergon_lib.quant.stats.intervals import newey_west_se

    clean = [float(x) for x in weekly_sn_lift if x is not None and not math.isnan(x)]
    n = len(clean)
    if n < 2:
        return {"status": "insufficient_data", "n": n}

    nw = newey_west_se(clean, max_lags=_hac_lag())
    if nw.get("status") != "ok":
        return {"status": "insufficient_data", "n": n}

    mean = nw["estimate"]
    se = nw["se"]
    if se <= 0:
        return {
            "status": "ok", "n": n, "mean": mean, "se": 0.0, "lags": nw["lags"],
            "t_stat": math.inf if mean != 0 else 0.0,
            "p_value": 0.0 if mean != 0 else 1.0,
            "significant": bool(mean != 0),
        }

    from scipy.stats import t as _student_t

    t_stat = mean / se
    dof = max(n - 1, 1)
    p_value = float(2.0 * _student_t.sf(abs(t_stat), dof))
    return {
        "status": "ok",
        "n": n,
        "mean": mean,
        "se": se,
        "lags": nw["lags"],
        "t_stat": float(t_stat),
        "p_value": p_value,
        "significant": bool(p_value < alpha),
    }


# ── Gate engine (weekly winner-take-all) ────────────────────────────────────


def _normalize_champion_before(champion: str) -> str:
    """Normalize a pointer/default champion value for GATE purposes only —
    never mutates the pointer itself (a held/no-contest week must never
    write). ``agentic`` (the retired seat, config-I2518 seat swap) WARNs and
    is treated as ``scanner_predictor_direct`` — belt-and-braces: the live
    pointer has been ``scanner_predictor_direct`` since 2026-07-13, so this
    path is not expected to fire in practice, only to guarantee a stale or
    hand-inspected historical pointer can never crash this engine. Any
    other unrecognized value is treated the same way (WARN + default to the
    base-case arm)."""
    if champion in VALID_CHAMPIONS:
        return champion
    if champion in _LEGACY_CHAMPIONS:
        logger.warning(
            "Champion pointer had legacy champion=%r (retired seat, "
            "alpha-engine-config-I2518 seat swap) — normalizing to %r for "
            "gate purposes only; the pointer itself is left untouched unless "
            "this week's gates clear a move.",
            champion, DEFAULT_GATE_CHAMPION,
        )
        return DEFAULT_GATE_CHAMPION
    logger.warning(
        "Champion pointer had unrecognized champion=%r — treating as %r for "
        "gate purposes only (the pointer itself is left untouched unless "
        "gates clear a move)", champion, DEFAULT_GATE_CHAMPION,
    )
    return DEFAULT_GATE_CHAMPION


# ── config/producer_champion.json writer (single writer, dual caller) ──────


def write_champion_pointer(
    bucket: str,
    champion: str,
    *,
    promotion_source: str,
    upload: bool,
    s3_client=None,
    promoted_at: str | None = None,
) -> dict:
    """THE single writer for ``config/producer_champion.json``. Both the
    gate engine (``promotion_source="gate_engine"``) and the one-shot
    2026-07-13 operator bootstrap (``promotion_source="operator_bootstrap"``)
    call this function — never write the pointer directly.

    ``champion`` MUST be in ``VALID_CHAMPIONS`` — this is the write-forbidden
    half of the read-tolerated/write-forbidden posture for retired seats
    (e.g. ``agentic``): raises ValueError for anything else, including every
    ``_LEGACY_CHAMPIONS`` value.

    ``champion`` MUST ALSO NOT be in ``SHADOW_ONLY_ARMS`` (Brian's ruling,
    2026-08-20, alpha-engine-config-I2515) — raises ValueError. This is
    DEFENCE IN DEPTH, deliberately duplicating the ``evaluate_gates`` veto
    one layer down: the gate is the POLICY (it decides, and records why the
    pointer was held), this writer is the INVARIANT (nothing that reaches
    S3 can violate it). A future caller that bypasses ``evaluate_gates``
    entirely — a new operator bootstrap, a backfill, a repair script —
    still cannot flip the live pointer onto a shadow-only arm. Note the
    asymmetry with reads: ``read_champion_pointer`` and
    ``_normalize_champion_before`` remain fully tolerant of a shadow-only
    value, since the executor has a live consumer for one
    (``executor/champion.py::_apply_thinktank_coverage``) and a historical
    or hand-inspected pointer must never crash this engine.

    Idempotent / bidirectional-safe: callers only invoke this when a gate
    decision has already determined the pointer SHOULD move (a no-contest or
    unchanged week must never call this).

    Raises on S3 write failure when ``upload=True`` — a swallowed failure
    here would silently leave the live executor trading the wrong arm.
    """
    if champion not in VALID_CHAMPIONS:
        raise ValueError(
            f"write_champion_pointer: champion={champion!r} not in {VALID_CHAMPIONS}"
        )
    if is_shadow_only(champion):
        # Fail LOUD (module posture: no silent swallows on a writer). A
        # caller reaching here has already bypassed the evaluate_gates
        # veto, so degrading quietly would reproduce exactly the defect
        # this guard exists to prevent — a live pointer moved onto an arm
        # Brian ruled measure-only.
        raise ValueError(
            f"write_champion_pointer: champion={champion!r} is declared "
            f"SHADOW-ONLY ({sorted(SHADOW_ONLY_ARMS)}) — measured, never "
            "promoted (Brian's ruling 2026-08-20, alpha-engine-config"
            "-I2515). The live config/producer_champion.json pointer may "
            "not be moved onto it. Promoting it requires its own ruling "
            "plus removing it from SHADOW_ONLY_ARMS."
        )
    if champion not in producer_arena.POINTER_ADMISSIBLE_ARMS:
        # Fail LOUD, same posture as the shadow-only invariant above and for
        # the same reason: this is the WRITER, and a pointer value the frozen
        # contract does not admit halts the executor's planner rather than
        # degrading. The arm is still registered, still scored, still ranked
        # and still on the authoritative arena_cycle artifact — the pointer
        # simply may not carry it until the enum is widened.
        raise ValueError(
            f"write_champion_pointer: champion={champion!r} is not admitted by "
            "contracts/producer_champion.schema.json's `champion` enum "
            f"({sorted(producer_arena.POINTER_ADMISSIBLE_ARMS)}). The executor raises "
            "ChampionPointerError on an unrecognized value and refuses to start a "
            "planning cycle, so writing it would halt trading."
        )
    pointer = {
        "schema_version": SCHEMA_VERSION,
        "champion": champion,
        # PRESERVED when the caller supplies it — a provenance correction on an
        # unmoved pointer must not overwrite when the pointer last actually
        # MOVED, or the correction destroys the fact it exists to record.
        "promoted_at": promoted_at or datetime.now(timezone.utc).isoformat(),
        "promotion_source": promotion_source,
    }
    if upload:
        s3 = s3_client or boto3.client("s3")
        body = json.dumps(pointer, indent=2, allow_nan=False).encode("utf-8")
        s3.put_object(
            Bucket=bucket, Key=POINTER_KEY, Body=body, ContentType="application/json",
        )
        logger.info(
            "Champion pointer written: s3://%s/%s champion=%s source=%s",
            bucket, POINTER_KEY, champion, promotion_source,
        )
    else:
        logger.info(
            "Champion pointer write skipped (upload=False) — would have set "
            "champion=%s source=%s", champion, promotion_source,
        )
    return pointer


def read_champion_pointer(bucket: str, s3_client=None) -> dict | None:
    """Read the current pointer. Returns None on 404/NoSuchKey (pre-bootstrap
    — no promotion has ever been written; callers should treat champion as
    the base-case arm, ``DEFAULT_GATE_CHAMPION`` ('scanner_predictor_direct'),
    mirroring executor/champion.py's own default post-I2515). Any other
    error is NOT swallowed here (this is the producer side, not the
    fail-loud executor consumer) but is logged and returns None so a
    transient read hiccup degrades to the base-case default rather than
    crashing the whole weekly evaluate run — the outcome is recorded as
    ``error`` by the caller either way, never a silent pointer write."""
    s3 = s3_client or boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=POINTER_KEY)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            logger.info("No champion pointer at s3://%s/%s (pre-bootstrap)", bucket, POINTER_KEY)
        else:
            logger.warning(
                "Champion pointer read failed (%s) — treating as pre-bootstrap "
                "base-case default", e,
            )
        return None
    except Exception as e:  # noqa: BLE001 — degraded-read carve-out, see docstring.
        logger.warning(
            "Champion pointer read failed (%s) — treating as pre-bootstrap "
            "base-case default", e,
        )
        return None


# ── config/apply_audit/producer_champion/{date}.json writer ────────────────


def load_prior_audit(bucket: str, s3_client=None) -> dict | None:
    """Read the prior weekly audit record (latest.json). Retained for API
    parity with the pre-I2518 engine (some callers/tests may still probe
    prior-run state for observability) — no longer consulted by
    evaluate_gates itself (winner-take-all carries no state forward:
    no hysteresis counter, no cooldown date). Absent artifact (first-ever
    run) -> None; any other read failure logs WARN and returns None."""
    s3 = s3_client or boto3.client("s3")
    key = f"{AUDIT_PREFIX}/latest.json"
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            logger.info("No prior producer_champion audit at s3://%s/%s (first run)", bucket, key)
        else:
            logger.warning("Prior producer_champion audit read failed (%s) — state restarts", e)
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("Prior producer_champion audit read failed (%s) — state restarts", e)
        return None


class LeaderboardUnusable(ValueError):
    """The evidence artifact does not describe this cycle.

    A distinct type so the audit record can say WHICH failure occurred. Before
    this existed the error path emitted ``leaderboard_unavailable`` for every
    exception, so a bug anywhere in the cycle was recorded as "the producer
    board was missing" — a slug that sends the reader to crucible-research to
    investigate a defect in this repo.
    """


def build_champion_audit(
    as_of: str,
    gate_result: dict | None,
    *,
    freeze: bool,
    error: str | None = None,
    error_slug: str = "unclassified_error",
) -> dict:
    """Build the weekly audit record (schema v2, published in
    ``nousergon_lib.contracts`` as ``producer_champion_audit`` — moved out of
    this repo's local ``contracts/`` dir under alpha-engine-config-I7605 so
    the dashboard consumer reads the same resource this producer does,
    instead of walking this repo's working tree. Bumped from v1 under
    alpha-engine-config-I2518: the HAC/hysteresis/cooldown fields
    (``challenger_matured_cohorts``, ``sn_lift_vs_champion``,
    ``consecutive_wins``, ``cooldown_until``) are retired in favor of
    ``champion_score``/``challenger_score`` — no live consumer outside this
    repo reads the audit record's fields (verified 2026-07-14), so no
    cross-repo coordination was required for the bump; v1 historical
    records remain valid documents under the frozen v1 shape and are not
    revalidated against v2). Written every week regardless of outcome —
    this IS the liveness proxy (config#2054).

    ``leaderboard_date_used`` (additive, alpha-engine-config-I2544,
    2026-07-14) is the date of the ``research/producer_leaderboard/
    {date}.json`` artifact actually consulted this run (the latest
    available <= ``as_of``, or None when no leaderboard was available at
    all / evaluation aborted before scoring) — always present (nullable),
    on every outcome including ``error``, so the audit trail is never
    silent about which week's evidence decided a flip.

    ``counterfactual_winner`` (additive, alpha-engine-config-I2515,
    2026-08-20) is the arm with the strictly higher weekly score — who
    WOULD have taken the pointer on score alone — or None when no
    comparison was possible (no_contest / error). It equals
    ``champion_after`` on an ordinary promotion and ``champion_before`` on
    a defended incumbency; the case it exists for is
    ``outcome="held_shadow_only"``, where it names the shadow-only arm that
    won and was deliberately not promoted. This is what keeps shadow-mode
    measurement legible in the durable weekly record.

    ``feed_dependencies`` (additive, alpha-engine-config-I3165, 2026-07-23)
    is ``ARM_FEED_DEPENDENCIES.get(champion_after)`` — the live-trade feed
    artifact_id(s) the record's ``champion_after`` arm declares, or ``None``
    when ``champion_after`` declares none (``thinktank_coverage``) or is
    itself ``None`` (``outcome="error"``, evaluation aborted before a
    champion could be read). Always derived from ``champion_after``, never
    ``champion_before`` — this field names what the LIVE pointer now
    depends on, which is unchanged from before this run on every
    non-promoted outcome and newly the challenger's feed on a promotion.

    ``arm_confidence`` (additive, alpha-engine-config-I7549, 2026-08-17) is
    the per-arm evidence verdict that decided — or declined to decide — this
    week: ``{"scanner_predictor_direct": "ok"|"thin"|"insufficient"|"unknown",
    "thinktank_coverage": "ok"|"thin"|"insufficient"|"unknown"|
    "unrecognised"}`` — one vocabulary, both arms (alpha-engine-config-I7549
    champion-side half; ``not_leaderboard_scored`` was emitted for the champion
    arm between #688 and that change and remains read-tolerated). Null only on ``outcome="error"`` (evaluation aborted
    before scoring). Without it, a week held at ``blocked_by=
    ["thinktank_coverage_thin_evidence"]`` records THAT the gate declined but
    not on what — and the whole point of the I7549 change is that a thin row
    is a different fact from an absent one (champion-challenger-policy.md
    §7.2)."""
    if error is not None or gate_result is None:
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "date": as_of,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "outcome": "error",
            "champion_before": None,
            "champion_after": None,
            "champion_score": None,
            "challenger_score": None,
            "blocked_by": [error_slug],
            "freeze": freeze,
            "detail": error or "gate evaluation did not run",
            "leaderboard_date_used": None,
            "feed_dependencies": None,
            "counterfactual_winner": None,
            "arm_confidence": None,
            "arm_scores": None,
        }
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "date": as_of,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "outcome": gate_result["outcome"],
        "champion_before": gate_result["champion_before"],
        "champion_after": gate_result["champion_after"],
        "champion_score": gate_result["champion_score"],
        "challenger_score": gate_result["challenger_score"],
        "blocked_by": gate_result["blocked_by"],
        "challenger": gate_result["challenger"],
        "freeze": freeze,
        "leaderboard_date_used": gate_result.get("leaderboard_date_used"),
        "feed_dependencies": ARM_FEED_DEPENDENCIES.get(gate_result["champion_after"]) or None,
        # alpha-engine-config-I2515 (2026-08-20 shadow-only ruling): the arm
        # that won on score this week, which on outcome="held_shadow_only"
        # is NOT champion_after. Without it the audit record could not
        # express "the shadow arm would have won", and a shadow arm that
        # silently stops being visible in the record defeats the whole
        # point of shadow mode (champion-challenger-policy.md §3).
        "counterfactual_winner": gate_result.get("counterfactual_winner"),
        "arm_confidence": gate_result.get("arm_confidence"),
        # alpha-engine-config-I8756 built this N-arm view in ``evaluate_gates``
        # and then dropped it HERE, so it never reached the durable artifact:
        # every audit record from 2026-08-21 onward carries ``arm_confidence``
        # but no ``arm_scores``, and on 2026-08-28 (``challenger: null``, both
        # challengers unscored) the record therefore named NO arm's score at
        # all. ``champion_score``/``challenger_score`` collapse the week into
        # the pair that happened to matter; with three arms that is no longer
        # the verdict. champion-challenger-policy.md §3 requires every arm's
        # measurement to be recorded every cycle, and a number computed and
        # then discarded before it becomes durable was never recorded.
        "arm_scores": gate_result.get("arm_scores"),
    }


def write_champion_audit(bucket: str, run_date: str, audit: dict, s3_client=None) -> str:
    """Write dated + latest audit artifacts. RAISES on failure — mirrors
    ``optimizer/apply_audit.write_audit``'s fail-loud posture: this record
    is the load-bearing liveness proxy, a swallowed write failure would
    recreate the exact invisible-silence defect config#2054 exists to
    retire."""
    s3 = s3_client or boto3.client("s3")
    body = json.dumps(audit, indent=2, allow_nan=False).encode("utf-8")
    dated_key = f"{AUDIT_PREFIX}/{run_date}.json"
    latest_key = f"{AUDIT_PREFIX}/latest.json"
    s3.put_object(Bucket=bucket, Key=dated_key, Body=body, ContentType="application/json")
    s3.put_object(Bucket=bucket, Key=latest_key, Body=body, ContentType="application/json")
    logger.info("Champion audit written: s3://%s/%s (+ latest.json)", bucket, dated_key)
    return dated_key


# ── Weekly arm-score sourcing ────────────────────────────────────────────────


def _publish_gate_error_alert(run_date: str, error: str) -> None:
    """Best-effort active alert when ``run_weekly_evaluation`` catches an
    exception during gate evaluation (config#2884). The weekly winner-take
    -all gate is correctly fail-STATIC on an internal error -- the pointer
    is never moved on a bad read -- but until now the ONLY liveness signal
    was the ARTIFACT_REGISTRY's file-PRESENCE SLA on the audit JSON, which
    a weekly ``outcome="error"`` write satisfies indefinitely. Without an
    active alert, a persistent bug could freeze ``config/producer_champion
    .json`` on a stale/losing arm for an unbounded number of weeks,
    discoverable only by manually reading each week's audit record.
    Mirrors the ``_publish_executor_opt_rejection_alert`` pattern in
    ``evaluate.py``. Never raises -- alerting must not crash the run this
    gate error already threatened to interrupt.
    """
    try:
        from ops_alerts import publish_ops_alert
    except ImportError as e:
        logger.warning(
            "[champion_promotion] gate-error alert skipped — ops_alerts "
            "unavailable: %s", e,
        )
        return
    message = (
        f"champion_promotion gate evaluation raised on {run_date}: {error}. "
        f"config/producer_champion.json was NOT re-evaluated this week "
        f"(fail-static — pointer unchanged). See "
        f"config/apply_audit/producer_champion/{run_date}.json (outcome=error)."
    )
    try:
        publish_ops_alert(
            message,
            severity="error",
            source="alpha-engine-backtester/optimizer/champion_promotion.py::run_weekly_evaluation",
            dedup_key=f"champion_promotion_gate_error_{run_date}",
            dedup_window_min=720,  # one alert per Saturday cycle, mirrors pit_parity.py
        )
    except Exception:  # noqa: BLE001 — alerting must never crash the run
        logger.exception(
            "[champion_promotion] gate-error alert publish failed (best-effort, swallowed)",
        )


# ── The decision: taken by nousergon_lib.arena, projected onto the audit ──


def _ladder_mean(cycle, arm_id: str) -> float | None:
    """The arm's LONGEST-window mean score this cycle, or None if it has none.

    The audit record's `arm_scores` is a per-arm scalar by contract, so the
    N-arm view it carries is the longest rung of each arm's ladder. It is an
    observability projection ONLY: no decision anywhere reads it. The decision
    reads paired per-date windows, which is the whole point — an incumbent
    scored over 2 dates successfully defended against a challenger rejected
    for having 4, on windows that barely overlapped (measured 2026-08-29).
    """
    for ladder in cycle.ladders:
        if ladder.arm_id == arm_id:
            longest = ladder.longest
            return longest.mean_score if longest else None
    return None


def decision_record_from_cycle(
    cycle, gaps, register, *, champion_before: str, freeze: bool,
) -> dict:
    """Project one :class:`ArenaCycle` onto the frozen audit-record shape.

    **There is exactly one decision, taken once, by
    ``nousergon_lib.arena.engine.run_cycle``.** This function computes
    nothing — it renames. Every gate that used to live here is gone: the
    winner-take-all comparison, the `thin_evidence` / `MIN_CYCLES_FOR_INFERENCE`
    / `confidence != ok` evidence floors, the champion-side floor, the
    two-week hysteresis and the two-week cooldown before them. The evidence
    bar is the anytime-valid confidence sequence and nothing else
    (champion-challenger-policy.md §5.0/§5.2, Brian ruling 2026-08-29).

    ``blocked_by`` is projected onto the CLOSED slug enum the frozen
    `producer_champion_audit` contract declares. The arena's own richer
    reasons — the per-pair window, the confidence bound, the named serving
    precondition that failed — are on `arena_cycle`, which is the
    authoritative record.
    """
    name_of = {a: register.state(a).record.name for a in register.all_arms()}
    decision = cycle.decision
    champion_after_name = name_of.get(decision.champion) if decision.champion else None

    # Who WOULD have taken the pointer on evidence alone, ignoring every
    # serving precondition. The Copeland leader is the honest answer for an
    # N-arm slot; the old two-arm "higher score" reading has no meaning here.
    counterfactual = None
    if cycle.ranking is not None and cycle.ranking.ordering:
        counterfactual = name_of.get(cycle.ranking.ordering[0])

    # The best-placed arm that is NOT the incumbent — the audit's `challenger`.
    challenger = None
    if cycle.ranking is not None:
        for arm in cycle.ranking.ordering:
            if name_of.get(arm) != champion_before:
                challenger = name_of.get(arm)
                break

    arm_scores = {name_of[a]: _ladder_mean(cycle, a) for a in cycle.scored_arms if a in name_of}
    gap_names = {g.arm_name for g in gaps}
    arm_confidence = {
        name: (CONFIDENCE_UNAVAILABLE if name in gap_names else CONFIDENCE_MEASURED)
        for name in arm_scores
    }

    record: dict[str, Any] = {
        "champion_before": champion_before,
        "champion_after": champion_before,
        "challenger": challenger,
        "champion_score": arm_scores.get(champion_before),
        "challenger_score": arm_scores.get(challenger) if challenger else None,
        "arm_scores": arm_scores,
        "arm_confidence": arm_confidence,
        "counterfactual_winner": counterfactual,
        "blocked_by": None,
        "leaderboard_date_used": None,  # filled by the caller
        # Not part of the frozen audit shape — carried on the returned record
        # so `run_weekly_evaluation` can wire the pointer write without
        # re-deriving the engine's verdict. Stripped by `build_champion_audit`.
        "_arena_status": decision.status,
        "_arena_reason": decision.reason,
        "_champion_after_name": champion_after_name,
    }

    if decision.status in producer_arena.ALARMING_STATUSES:
        # §7.2: an unmeasurable/unservable cycle is a definitional hold that
        # SAYS SO. It is never a default win for either side, and it never
        # renders as a defended incumbency.
        record["outcome"] = "no_contest"
        record["blocked_by"] = _blocked_slugs(decision, gaps)
        return record

    if champion_after_name is None or champion_after_name == champion_before:
        record["outcome"] = "unchanged_winner_already_champion"
        # WHY it was held, when the reason is an arm the engine ruled
        # ineligible rather than an unbeaten incumbent. Without this the
        # record is indistinguishable from a defended incumbency, which is a
        # different and false claim — and on a shadow-only hold it would erase
        # the counterfactual shadow mode exists to measure (§7.2).
        leader = cycle.ranking.ordering[0] if (
            cycle.ranking is not None and cycle.ranking.ordering
        ) else None
        failed = [
            c for c in decision.ineligible.get(leader, ()) if not c.passed
        ] if leader is not None else []
        if failed:
            record["blocked_by"] = _blocked_slugs_for(failed)
            if all(c.name == "not_shadow_only" for c in failed):
                record["outcome"] = "held_shadow_only"
        return record

    # The pointer moved. One thing can still hold it: --freeze, a suppression
    # of a decision the engine has already taken (recorded as such, with
    # champion_after NOT advanced, so the carry-forward state is real).
    if is_shadow_only(champion_after_name):
        # Defence in depth. The engine already excluded shadow-only arms via
        # the `not_shadow_only` serving precondition, so reaching here means a
        # precondition was not wired — fail loud rather than promote.
        raise ValueError(
            f"arena promoted shadow-only arm {champion_after_name!r}; the "
            "`not_shadow_only` serving precondition was not applied. A shadow-only "
            "arm is measured, never served."
        )
    if freeze:
        record["outcome"] = "promoted"
        record["blocked_by"] = ["frozen"]
        return record

    record["outcome"] = "promoted"
    record["champion_after"] = champion_after_name
    return record


def _blocked_slugs_for(failed_checks) -> list[str]:
    """Project failed serving preconditions onto the contract's CLOSED enum.

    The enum lives in `nousergon_lib.contracts` and cannot grow from this
    repo, so the projection is lossy BY CONSTRUCTION — which is exactly why
    `arena_cycle` is the authoritative artifact and this record is a narrowed
    view of it. Every slug emitted here is one the contract already declares;
    `pointer_contract_admits` has no slug of its own and lands on the generic
    one rather than inventing a value the consumer would reject.

    Stable and de-duplicated: a reader diffing two weeks' records should see a
    change only when the reason changed.
    """
    by_name = {
        "not_shadow_only": "shadow_only_arm",
        "feed_producer_live": "feed_producer_dead",
    }
    return sorted({
        by_name.get(c.name, "arm_score_unavailable") for c in failed_checks
    })


def _blocked_slugs(decision, gaps) -> list[str]:
    """The same projection across EVERY ineligible arm, plus series gaps."""
    failed = [c for checks in decision.ineligible.values() for c in checks if not c.passed]
    slugs = set(_blocked_slugs_for(failed))
    if gaps or not slugs:
        slugs.add("arm_score_unavailable")
    return sorted(slugs)


def run_weekly_evaluation(
    *,
    bucket: str,
    run_date: str,
    e2e_lift: dict | None,
    tt_leaderboard: dict | None,
    tt_leaderboard_date_used: str | None = None,
    freeze: bool,
    upload: bool,
    s3_client=None,
) -> dict:
    """Top-level entry point wired into evaluate.py. Runs ONE arena cycle and
    writes three artifacts:

      1. ``arena/producer/{date}.json`` (+ ``latest.json``) — the
         AUTHORITATIVE `arena_cycle` record, schema-validated before the
         write, emitted every cycle whatever the outcome
         (champion-challenger-policy.md §11).
      2. ``config/apply_audit/producer_champion/{date}.json`` (+
         ``latest.json``) — the narrowed `producer_champion_audit` v2 view its
         existing consumers still read. ALWAYS written, any outcome.
      3. ``config/producer_champion.json`` — the live pointer. Rewritten when
         the arena moved it, and ALSO rewritten in place (same champion, same
         `promoted_at`) when the recorded `promotion_source` no longer
         describes how the pointer got there. That second case is the one
         alpha-engine-config-I9318 exists for: this pointer read
         ``promotion_source: "operator_bootstrap"`` from 2026-07-13 for six
         weeks of automated evaluations, and no surface said so (§11).

    ``e2e_lift`` is retained for the observability leaderboard history only —
    it no longer feeds any decision. ``tt_leaderboard`` is the parsed
    ``research/producer_leaderboard/{date}.json`` (the LATEST available <=
    ``run_date``), which is now the single evidence source for EVERY arm
    including the incumbent: one board, one cohort, one benchmark. The
    per-arm scorers reading three different sources on three different
    cohorts are deleted — that asymmetry is what let the champion's silence
    render as another arm's thinness.
    """
    pointer = read_champion_pointer(bucket, s3_client=s3_client)
    champion_before = _normalize_champion_before(
        (pointer or {}).get("champion", DEFAULT_GATE_CHAMPION)
    )

    record = None
    cycle = None
    gaps: list = []
    error = None
    error_slug = "unclassified_error"
    try:
        _assert_leaderboard_usable(tt_leaderboard, run_date, tt_leaderboard_date_used)
        # Feed liveness stays a per-arm SERVING PRECONDITION, probed for every
        # arm that declares a dependency rather than only for "the challenger
        # that would win" — with N arms and a free-moving pointer there is no
        # single such arm before the engine has decided.
        feed_blocked = {
            arm: "declared feed producer looks dead or orphaned at promotion time"
            for arm in ARM_FEED_DEPENDENCIES
            if check_feed_dependencies_live(
                arm, bucket=bucket, run_date=run_date, s3_client=s3_client,
            ) is not None
        }
        cycle, gaps, register = producer_arena.run_arena_cycle(
            as_of=run_date,
            leaderboard=tt_leaderboard,
            incumbent_name=champion_before,
            shadow_only_names=SHADOW_ONLY_ARMS,
            feed_blocked_names=feed_blocked,
        )
        doc = producer_arena.cycle_document(
            cycle, gaps, producer_arena.arm_statistics(register, tt_leaderboard, run_date),
        )
        producer_arena.write_arena_cycle(
            bucket, run_date, doc, upload=upload, s3_client=s3_client,
        )
        producer_arena.publish_cycle_alert(cycle, gaps)
        record = decision_record_from_cycle(
            cycle, gaps, register, champion_before=champion_before, freeze=freeze,
        )
        record["leaderboard_date_used"] = tt_leaderboard_date_used
    except Exception as e:  # noqa: BLE001 — the cycle must never crash the
        # weekly evaluate run; it is recorded as an error outcome (still
        # written, per the liveness posture) and alerted.
        logger.exception("Producer arena cycle raised")
        error = str(e)
        if isinstance(e, LeaderboardUnusable):
            error_slug = "leaderboard_unavailable"
        _publish_gate_error_alert(run_date, error)

    audit = build_champion_audit(
        run_date, record, freeze=freeze, error=error, error_slug=error_slug,
    )

    pointer_written = None
    if record is not None:
        pointer_written = _reconcile_pointer(
            bucket,
            pointer=pointer,
            record=record,
            freeze=freeze,
            upload=upload,
            s3_client=s3_client,
        )

    logger.info(
        "producer_champion evaluation: outcome=%s champion_before=%s champion_after=%s "
        "arena_status=%s blocked_by=%s leaderboard_date_used=%s",
        audit["outcome"], audit["champion_before"], audit["champion_after"],
        (record or {}).get("_arena_status"), audit["blocked_by"],
        audit.get("leaderboard_date_used"),
    )

    if upload:
        try:
            write_champion_audit(bucket, run_date, audit, s3_client=s3_client)
        except Exception:
            logger.exception(
                "producer_champion audit S3 write failed — this is the "
                "liveness proxy, surfacing loudly",
            )
            raise
    else:
        logger.info("producer_champion audit S3 write skipped (upload=%s) — logged only", upload)

    # ── Deliver the verdict (alpha-engine-config-I2364 measurability gap) ──
    # A verdict nobody is told is indistinguishable from a loop that stopped
    # running (principles.md §2.7). Gated on ``upload`` for the same reason
    # the S3 write is, and never allowed to fail the weekly evaluate run.
    if upload:
        try:
            champion_digest.send_verdict_digest(audit)
        except Exception:  # noqa: BLE001 — notification must never crash the run
            logger.exception(
                "producer_champion verdict digest raised — the verdict is still "
                "durable in the audit record, but no operator was told",
            )

    result = dict(audit)
    if pointer_written is not None:
        result["_pointer_write"] = pointer_written
    return result


def _assert_leaderboard_usable(
    leaderboard: dict | None, run_date: str, date_used: str | None,
) -> None:
    """Well-formedness of the evidence artifact. NOT an evidence bar.

    Staleness and a horizon mismatch are statements about whether the
    artifact describes THIS cycle at all — an eight-day-old board scored at a
    different primary horizon is not thin evidence, it is a different
    measurement. Every gate that spoke about HOW MUCH evidence an arm had is
    deleted; these two survive because they speak about WHICH measurement is
    in front of us.
    """
    if not isinstance(leaderboard, dict) or date_used is None:
        raise LeaderboardUnusable(
            "no research/producer_leaderboard/{date}.json available at or before "
            f"{run_date} — the slot has no evidence artifact to score any arm from"
        )
    age_days = (date.fromisoformat(run_date) - date.fromisoformat(date_used)).days
    if age_days < 0:
        raise LeaderboardUnusable(
            f"producer leaderboard {date_used} is DATED AFTER the run date {run_date}; "
            "a future artifact is never this cycle's evidence"
        )
    if age_days > LEADERBOARD_STALENESS_DAYS:
        raise LeaderboardUnusable(
            f"producer leaderboard {date_used} is {age_days} days older than {run_date} "
            f"(bound {LEADERBOARD_STALENESS_DAYS}) — refusing to decide the live pointer "
            "on evidence this stale"
        )
    declared_horizon = leaderboard.get("horizon_days")
    if declared_horizon is not None and declared_horizon != GATE_HORIZON_DAYS:
        raise LeaderboardUnusable(
            f"producer leaderboard {date_used} declares horizon_days={declared_horizon!r} but "
            f"this slot decides at {GATE_HORIZON_DAYS} sessions; an upstream primary-horizon "
            "change must surface loudly, never silently rescore the live pointer"
        )


def _reconcile_pointer(
    bucket: str,
    *,
    pointer: dict | None,
    record: dict,
    freeze: bool,
    upload: bool,
    s3_client=None,
) -> dict | None:
    """Move the pointer when the arena moved it; otherwise make its PROVENANCE true.

    The second half is alpha-engine-config-I9318's ``closes-when``:
    ``config/producer_champion.json`` read ``promotion_source:
    "operator_bootstrap"`` from 2026-07-13 through 2026-08-29 while an
    automated engine evaluated it every week. That claim was false about the
    live system, and §11 calls it a finding rather than a stable state.

    So on every cycle the recorded provenance is set to what actually decided
    this pointer's current value — ``arena_decided``/``arena_held``/
    ``arena_unmeasurable``/``arena_bootstrap`` — and ``promoted_at`` is
    PRESERVED whenever the champion did not change, so "when did the pointer
    last move" survives the provenance correction. ``operator_bootstrap``
    can now only appear when a human bootstrap genuinely wrote it and no
    cycle has run since.
    """
    status = record.get("_arena_status") or "unknown"
    source = f"arena_{status}"
    # The arm the ENGINE decided, NOT the audit record's projection of it. The
    # projection nulls any arm the frozen `producer_champion_audit` enum cannot
    # name (`no_agent_quant`, `single_agent_quant` today), and the live pointer
    # must never be narrowed by a rendering: the POINTER contract admits both
    # arms, so reading the audit field here would silently refuse to promote
    # exactly the two arms with the most evidence.
    champion_after = record["_champion_after_name"]
    moved = record["outcome"] == "promoted" and not freeze
    if moved and champion_after is None:
        raise ValueError(
            "arena reported a promotion with no champion arm; refusing to write the "
            "live pointer from an incomplete decision"
        )

    if not moved:
        current_source = (pointer or {}).get("promotion_source")
        if pointer is not None and current_source == source:
            return None  # already true; nothing to say
        return write_champion_pointer(
            bucket, record["champion_before"],
            promotion_source=source, upload=upload, s3_client=s3_client,
            promoted_at=(pointer or {}).get("promoted_at"),
        )
    return write_champion_pointer(
        bucket, champion_after,
        promotion_source=source, upload=upload, s3_client=s3_client,
    )


# ── research/producer_leaderboard_champion_gate/{date}.json ─────────────────
#
# This module's OWN observability artifact (config#2367) — a per-run
# snapshot of scanner_predictor_direct's realized lift vs the agentic
# baseline, appended to a running history. Under the pre-I2518 HAC engine
# this fed the significance/hysteresis gate directly; under winner-take-all
# it is NO LONGER consumed by the gate (which only needs THIS week's point,
# taken straight from ``e2e_lift`` via ``_score_scanner_predictor_direct``
# above) but is STILL MAINTAINED here for observability/history continuity
# and because config#2452 (the key-collision fix that gave this artifact its
# own distinct key, distinct from crucible-research's
# research/producer_leaderboard/{date}.json) has an open live-verification
# tail expecting this artifact to keep being written every Saturday.
#
# config#2452 (found 2026-07-13, same day as first live run post-merge): this
# key was originally `research/producer_leaderboard/{date}.json` — the SAME
# key crucible-research's `scoring/leaderboard_producers.py` already writes,
# with an incompatible schema. Renamed before any collision occurred.


LEADERBOARD_KEY_TMPL = "research/producer_leaderboard_champion_gate/{date}.json"
_LEADERBOARD_HISTORY_KEEP_WEEKS = 26  # ~6 months of weekly points

RESEARCH_PRODUCER_LEADERBOARD_KEY_TMPL = "research/producer_leaderboard/{date}.json"


def leaderboard_entry_from_e2e_lift(e2e_lift: dict | None) -> dict | None:
    """Extract this week's sector-neutral alpha point (scanner_then_predictor's
    OWN realized 21d alpha, already SPY-relative) from the e2e_lift
    diagnostic already computed earlier in the same evaluate run. Returns
    None when the counterfactual is unavailable this week
    (skipped/insufficient_data/error/missing) — an honest "no new point this
    week" rather than fabricating one.

    ``sector_neutral_mean_alpha_21d`` (alpha-engine-config-I2998) is the
    current gate's scanner_predictor_direct score source (see
    ``_score_scanner_predictor_direct``) — the arm's direct lift vs the SPY
    zero-line, gated on THIS field's presence rather than the retired
    ``sn_lift_vs_agentic_cio`` (still carried for observability, may be
    None if the agentic-CIO comparator itself is unavailable that week —
    that no longer blocks this entry from being usable).
    """
    if not isinstance(e2e_lift, dict):
        return None
    cf = e2e_lift.get("scanner_then_predictor_counterfactual")
    if not isinstance(cf, dict) or cf.get("status") != "ok":
        return None
    methods = cf.get("methods", {})
    pred = methods.get("scanner_then_predictor_topN")
    if not isinstance(pred, dict):
        return None
    sn_alpha = pred.get("sector_neutral_mean_alpha_21d")
    if sn_alpha is None:
        return None
    sn_lift = pred.get("sn_lift_vs_agentic_cio")
    return {
        "sector_neutral_mean_alpha_21d": float(sn_alpha),
        "sn_lift_vs_agentic_cio": float(sn_lift) if sn_lift is not None else None,
        "n_picks": pred.get("n_picks"),
        "n_cycles": cf.get("n_cycles"),
    }


def build_leaderboard_artifact(run_date: str, history: list[dict], new_entry: dict | None) -> dict:
    """Append ``new_entry`` (if any) to ``history`` (oldest-first list of
    ``{"date": ..., "sector_neutral_mean_alpha_21d": ..., "sn_lift_vs_agentic_cio": ...,
    "n_picks": ..., "n_cycles": ...}``), trim to the retention window, and
    return the full artifact to write to
    ``research/producer_leaderboard_champion_gate/{run_date}.json``.
    """
    points = list(history)
    if new_entry is not None:
        points = [p for p in points if p.get("date") != run_date]
        points.append({"date": run_date, **new_entry})
    points = points[-_LEADERBOARD_HISTORY_KEEP_WEEKS:]
    return {
        "schema_version": 1,
        "as_of": run_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "weekly_points": points,
    }


def leaderboard_gate_inputs(artifact: dict) -> dict:
    """Reduce a leaderboard artifact to matured cohort count + weekly SN-lift
    series. Retained for API parity / observability (e.g. a future
    diagnostic reusing hac_significance) — no longer consumed by
    evaluate_gates under winner-take-all."""
    points = artifact.get("weekly_points", []) if isinstance(artifact, dict) else []
    lifts = [p["sn_lift_vs_agentic_cio"] for p in points if p.get("sn_lift_vs_agentic_cio") is not None]
    return {
        "challenger_matured_cohorts": len(lifts),
        "challenger_weekly_sn_lift": lifts,
    }


def read_leaderboard(bucket: str, run_date: str, s3_client=None) -> dict | None:
    s3 = s3_client or boto3.client("s3")
    key = LEADERBOARD_KEY_TMPL.format(date=run_date)
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise
    except Exception:
        return None


def read_prior_leaderboard_history(bucket: str, run_date: str, s3_client=None) -> list[dict]:
    """Read the most recent leaderboard artifact available (by scanning
    backward from ``run_date`` — NOT wall-clock today, so a ``--date``
    backfill run seeds history relative to the backfilled trading day, not
    the day the backfill happens to execute) to seed ``weekly_points``
    history."""
    s3 = s3_client or boto3.client("s3")
    from datetime import date as _date, timedelta

    anchor = _date.fromisoformat(run_date)
    for back in range(1, 15):
        probe_date = (anchor - timedelta(days=back)).isoformat()
        key = LEADERBOARD_KEY_TMPL.format(date=probe_date)
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj["Body"].read())
            return list(data.get("weekly_points", []))
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                continue
            logger.warning("Leaderboard history probe failed at %s: %s", key, e)
            return []
        except Exception as e:  # noqa: BLE001
            logger.warning("Leaderboard history probe failed at %s: %s", key, e)
            return []
    return []


def write_leaderboard(bucket: str, run_date: str, artifact: dict, s3_client=None) -> str:
    s3 = s3_client or boto3.client("s3")
    body = json.dumps(artifact, indent=2, allow_nan=False).encode("utf-8")
    key = LEADERBOARD_KEY_TMPL.format(date=run_date)
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    logger.info("producer_leaderboard written: s3://%s/%s (%d weekly points)", bucket, key, len(artifact.get("weekly_points", [])))
    return key


def update_leaderboard_and_get_gate_inputs(
    bucket: str, run_date: str, e2e_lift: dict | None, *, upload: bool, s3_client=None,
) -> dict:
    """Full leaderboard maintenance step: read prior history, append this
    week's point (if the counterfactual matured this run), write the
    updated artifact, and return the gate-ready reduction. Called once per
    evaluate run, BEFORE ``run_weekly_evaluation`` — maintained for
    observability / config#2452 continuity (see module docstring); its
    return value is no longer fed into the gate decision.
    """
    history = read_prior_leaderboard_history(bucket, run_date, s3_client=s3_client) if upload else []
    new_entry = leaderboard_entry_from_e2e_lift(e2e_lift)
    artifact = build_leaderboard_artifact(run_date, history, new_entry)
    if upload:
        try:
            write_leaderboard(bucket, run_date, artifact, s3_client=s3_client)
        except Exception:
            logger.exception(
                "producer_leaderboard write failed — this observability "
                "artifact will be missing this week (non-fatal to the gate, "
                "which no longer depends on it)",
            )
    return leaderboard_gate_inputs(artifact)


def read_research_producer_leaderboard(bucket: str, run_date: str, s3_client=None) -> dict | None:
    """Read crucible-research's REAL champion/challenger producer leaderboard
    (``scoring/leaderboard_producers.py::build_producer_leaderboard``,
    config#1221/#1223) for ``run_date`` — the evidence source for
    thinktank_coverage's weekly score (see module docstring). Distinct key
    from this module's OWN ``research/producer_leaderboard_champion_gate/
    {date}.json`` (config#2452 collision fix) — this function only READS
    the crucible-research-owned artifact, never writes it.

    Returns None on 404/NoSuchKey (not yet written this week — e.g. before
    the Saturday eval_rolling_mean Lambda step runs, or any week
    crucible-research's build fails) or any other read/parse failure
    (logged) — a missing/malformed leaderboard degrades to a no-contest week
    for thinktank_coverage (``_score_thinktank_coverage``), never a crash
    and never a fabricated score."""
    s3 = s3_client or boto3.client("s3")
    key = RESEARCH_PRODUCER_LEADERBOARD_KEY_TMPL.format(date=run_date)
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            logger.info(
                "No crucible-research producer_leaderboard at s3://%s/%s "
                "(not yet written this week)", bucket, key,
            )
        else:
            logger.warning("crucible-research producer_leaderboard read failed (%s)", e)
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("crucible-research producer_leaderboard read failed (%s)", e)
        return None


def find_latest_research_producer_leaderboard_date(
    bucket: str, run_date: str, s3_client=None,
) -> str | None:
    """List the ``research/producer_leaderboard/`` prefix (crucible-research
    -owned, config#1221/#1223) and return the latest well-formed date <=
    ``run_date`` found among its dated keys, or None if none exist at or
    before ``run_date``.

    alpha-engine-config-I2544 (2026-07-14 ruling): the ASYNC advisory child
    SF that now writes this artifact may not have finished — or may have
    failed outright — by the time this Evaluator-stage gate runs in the
    MAIN weekly SF, so an exact same-day key read is no longer a safe
    assumption. Reading the LATEST AVAILABLE leaderboard <= ``run_date`` is
    the semantically CORRECT read, not a compromise: the gate scores
    realized (matured) outcomes of PRIOR weeks' ``thinktank_coverage``
    selections — a same-day leaderboard could not contain resolved
    outcomes for same-day picks even if it existed on time.

    A single ``list_objects_v2`` call (no pagination) is sufficient: this
    prefix is written at most weekly, so even several years of history
    stays far under the 1000-key single-page ceiling — mirrors
    ``factor_blend_optimizer._read_recent_shadow_archives``'s identical
    single-call reasoning for its own weekly-cadence prefix. Any key under
    the prefix that doesn't match the ``{date}.json`` shape (e.g. a future
    ``latest.json`` sidecar some other consumer adds) is silently skipped,
    never crashes the scan. A list failure (ClientError or otherwise) is
    logged and treated as "nothing available" — degrades to a no-contest
    week downstream, never a crash.
    """
    s3 = s3_client or boto3.client("s3")
    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=RESEARCH_PRODUCER_LEADERBOARD_PREFIX)
    except ClientError as e:
        logger.warning(
            "research/producer_leaderboard/ list failed (%s) — treating as "
            "no leaderboard available", e,
        )
        return None
    except Exception as e:  # noqa: BLE001 — list must never crash the gate
        logger.warning(
            "research/producer_leaderboard/ list failed (%s) — treating as "
            "no leaderboard available", e,
        )
        return None

    anchor = date.fromisoformat(run_date)
    best: date | None = None
    for obj in resp.get("Contents") or []:
        m = _RESEARCH_PRODUCER_LEADERBOARD_KEY_RE.match(obj.get("Key", ""))
        if not m:
            continue
        candidate = date.fromisoformat(m.group(1))
        if candidate <= anchor and (best is None or candidate > best):
            best = candidate
    if best is None:
        logger.info(
            "No research/producer_leaderboard/ artifact <= %s found under "
            "s3://%s/%s", run_date, bucket, RESEARCH_PRODUCER_LEADERBOARD_PREFIX,
        )
        return None
    return best.isoformat()


def read_latest_research_producer_leaderboard(
    bucket: str, run_date: str, s3_client=None,
) -> tuple[dict | None, str | None]:
    """Combined list-then-read: find the latest
    ``research/producer_leaderboard/{date}.json`` <= ``run_date``
    (``find_latest_research_producer_leaderboard_date``) and read it
    (``read_research_producer_leaderboard``, reused unchanged — it is
    still the correct exact-date-read primitive once the date to read has
    been selected).

    THE production entry point for thinktank_coverage's evidence as of
    alpha-engine-config-I2544 — supersedes calling
    ``read_research_producer_leaderboard`` directly with ``run_date`` (an
    exact-match read that assumed same-day availability the async advisory
    child SF can no longer guarantee).

    Returns ``(leaderboard_dict, leaderboard_date_used)``: ``(None, None)``
    when no artifact <= ``run_date`` exists yet (or the list itself
    failed); ``(None, None)`` also when a date was found but the S3 read
    then failed (``read_research_producer_leaderboard`` already logs the
    specifics) — never a partial/inconsistent pairing of a leaderboard
    with the wrong date or a date with no leaderboard.
    """
    latest_date = find_latest_research_producer_leaderboard_date(
        bucket, run_date, s3_client=s3_client,
    )
    if latest_date is None:
        return None, None
    leaderboard = read_research_producer_leaderboard(bucket, latest_date, s3_client=s3_client)
    if leaderboard is None:
        return None, None
    return leaderboard, latest_date
