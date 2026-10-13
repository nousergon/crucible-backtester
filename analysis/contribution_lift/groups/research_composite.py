"""research.research_composite — the research-tile substitution replays (config-I7478).

Five graded `crucible-evaluator/grading/tiles/research.py` components, all
answering the same question in the RC v3 objective (net-of-cost realized 21d
log-alpha vs SPY, spec §1): **how much of the measured alpha survives when this
component's ranking information is taken away?**

All five use the *substitution* pattern (spec §3): the arms differ only in WHICH
names were selected, never in how they were sized or costed, so the harness's
shared equal-weight sizing is exactly what isolates selection.

Coverage
--------

``research_composite_ic`` — **measured**. The live universe-board attractiveness
composite, rebuilt per cycle from the six sector-neutral pillar percentiles
persisted to ``factors/profiles/{date}/by_ticker.json``
(``quality_score``, ``value_score``, ``momentum_score``, ``growth_score``,
``stewardship_score``, ``low_vol_score``) by the exact blend
``analysis/end_to_end.py::_scanner_factor_counterfactual`` uses: cross-sectional
winsorized-z per pillar, clipped +/-3, then a coverage-renormalized equal-weight
row mean. The board's terminal cross-sectional percentile is a monotone
transform, so ranking on this blend IS ranking on the live
``attractiveness_score``.

* baseline — top-N by that composite, N = the live selection's width for the
  same cycle (``harness.live_picks_by_date``: the executed ENTERs in trades.db,
  champion-aware per config-I7501 — count-matched to the book the system
  actually opened).
* ablated  — **every pillar replaced by its own cross-sectional mean**. That is
  the leave-one-out for a *ranking* component: substituting each input by its
  cross-sectional mean collapses every winsorized-z to zero, so the composite
  becomes a constant and carries no ordering at all, while the universe, the
  cycle set, the width and the sizing are untouched. It is a substitution of the
  component's own inputs, not a different selector bolted on beside it.

  A constant score leaves the selection entirely to the tie-break, and the
  obvious tie-break — plain ascending ticker — is rejected here: it returns the
  SAME alphabetically-first names on every cycle, which is not an
  uninformative-selection null but one persistent concentrated portfolio whose
  per-cycle diffs are strongly correlated across cycles (understating the paired
  bootstrap's variance). The tie-break used instead is a **deterministic
  rotation** of the ticker-sorted universe, offset by the cycle's position in
  the window: still pure, still reproducible byte-for-byte, still ties-by-ticker
  within a cycle, but it walks the whole universe over the window so the null
  averages over the cross-section instead of betting on one corner of the
  alphabet. See :func:`_rotated_slice`.

``neutralization_live_efficacy`` — **measured**. The component is the
cross-sectional Barra residualizer (momentum/beta/size, config#1142); the
leave-one-out is to bypass it.

* baseline — top-N by the composite AFTER ``end_to_end._xs_neutralize`` against
  ``DEFAULT_NEUTRALIZE_FACTORS``, exposures loaded read-only from the ArcticDB
  universe feature history via ``end_to_end.load_historical_factor_loadings``.
* ablated  — top-N by the RAW composite (neutralization bypassed).

  Cycles where the residualizer passes through as the identity (its <20-name
  floor, or missing exposures) are dropped from both arms: on those cycles the
  two arms are the same portfolio and a 0.0 diff would be recorded as
  "neutralization contributes nothing" when the truth is that it never engaged.
  If NO cycle engages, the component reports ``N/A-MISSING-INPUT`` naming the
  exposure coverage rather than a zero.

``scanner_feed_counterfactual`` — **measured**. The component is the scanner's
ranking of the ~900-name PIT universe down to the passing pool.

* baseline — the live selection: the names actually entered that cycle
  (``harness.live_picks_by_date``, config-I7501).
* ablated  — the same count drawn from the **pre-scanner** PIT population
  (every name the scanner evaluated for the as-of research cycle, pass or
  fail, i.e. the names the scanner had in front of it) with its ranking
  removed, by the same deterministic rotation.

  The population comes from the **live** record,
  ``candidates/{run_date}/candidates.json::scanner_eval_log`` — the
  standalone Scanner's per-ticker gate verdict over its whole universe,
  published every cycle since 2026-07-10 (config#3053), read through
  ``end_to_end._load_live_scanner_cuts`` so this component and
  ``scanner_lift_live`` share one cohort rule. ``scanner_evaluations`` is
  **NOT live** (alpha-engine-config-I8757): its writer was retired with the
  rest of the research graph on 2026-07-12 and its newest row is 2026-07-17.
  It is read only for research cycles BEFORE the first published live cohort,
  so the window's spring cycles keep a population; from the first live cohort
  on, the live record is the only source. Until alpha-engine-config-I12231 the
  table was the ONLY source, so the component sat at ``N/A-MISSING-INPUT`` on
  every card while the producer was emitting the population it needed.

  The staleness refusal stays: the component emits ``N/A-MISSING-INPUT`` once
  the newest population cycle (from either source) is further behind the
  replayed cycles than ``retired_table_guard.DEFAULT_MAX_AGE_DAYS`` — a frozen
  population makes the ablated arm a comparison against a system that no
  longer exists, which is the same reason ``sector_teams_avg`` below is N/A.

``sector_teams_avg`` and ``cio_selection_skill`` — **N/A-RETIRED**. Their only
sources are the six-team / CIO research-graph tables, retired 2026-07-12
(``analysis/end_to_end.RESEARCH_GRAPH_RETIRED_DATE``); the producer stopped
writing them and nothing has replaced their per-team / per-CIO selection
record. A replay reading them would measure a system that no longer exists, and
the package's own guard test forbids naming those tables in executable code.
There is no substitute selection to count-match against, so these emit an
explicit ``N/A-RETIRED`` with the retirement date rather than a fabricated
value.

Determinism
-----------
Every arm is a pure function of :class:`ReplayInputs` plus read-only archive
reads (ArcticDB feature history for the exposures; the published
``candidates.json`` artifacts and, for pre-2026-07-10 cycles, the research
SQLite for the pre-scanner population). No RNG anywhere: the only non-score ordering
in this module is the ticker sort and the cycle-index rotation.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from analysis import retired_table_guard
from analysis.contribution_lift.harness import (
    ArmSet,
    NotAvailable,
    ReplayInputs,
    ReplaySpec,
    live_picks_by_date,
    live_selection_label,
    no_live_selection,
    picks_arm,
)

logger = logging.getLogger(__name__)

ISSUE = "alpha-engine-config-I7478"

#: Winsorization bound of the live universe board's per-pillar z-score.
_PILLAR_CLIP = 3.0

#: Cross-sectional standard deviation below which a pillar carries no ordering.
_DEGENERATE_SD = 1e-12

#: Two neutralized scores closer than this are the identity passthrough.
_IDENTITY_TOL = 1e-9

_PROFILES_KEY = "factors/profiles/{date}/by_ticker.json"

#: The live pre-scanner population (config#3053; alpha-engine-config-I12231).
_LIVE_POPULATION_SOURCE = (
    "s3://{bucket}/candidates/{{run_date}}/candidates.json::scanner_eval_log"
)


# --------------------------------------------------------------------------
# Live selection (the count-matching base)
# --------------------------------------------------------------------------


# The live selection — and the width every arm is count-matched to — comes
# from ``harness.live_picks_by_date`` (config-I7501), which reads the executed
# ENTERs in trades.db rather than the signals feed the champion left empty.

# --------------------------------------------------------------------------
# The live attractiveness composite
# --------------------------------------------------------------------------


def _pillar_composite(
    profiles: Mapping[str, Mapping[str, Any]], universe: Sequence[str]
) -> pd.Series | None:
    """The live universe-board composite over ``universe`` for one cycle.

    Mirrors ``end_to_end._scanner_factor_counterfactual``'s attractiveness
    block exactly: cross-sectional z per pillar, clipped to +/-``_PILLAR_CLIP``,
    then a coverage-renormalized equal-weight blend (== the row mean of the
    pillars present for that name).

    Returns ``None`` when no pillar in the cycle carries any ordering.
    """
    from analysis.end_to_end import _ATTRACTIVENESS_PILLARS

    rows = {t: dict(profiles.get(t) or {}) for t in universe if profiles.get(t)}
    if not rows:
        return None
    frame = pd.DataFrame.from_dict(rows, orient="index")
    zed = pd.DataFrame(index=frame.index)
    for pillar in _ATTRACTIVENESS_PILLARS:
        if pillar not in frame.columns:
            continue
        series = pd.to_numeric(frame[pillar], errors="coerce")
        sd = series.std(ddof=0)
        scaled = (
            (series - series.mean()) / sd
            if sd and sd > _DEGENERATE_SD
            else series * 0.0
        )
        zed[pillar] = scaled.clip(-_PILLAR_CLIP, _PILLAR_CLIP)
    if zed.shape[1] == 0 or not bool(zed.notna().any().any()):
        return None
    composite = zed.mean(axis=1).dropna()
    composite.index = composite.index.map(str)
    return composite if not composite.empty else None


# --------------------------------------------------------------------------
# Deterministic selectors
# --------------------------------------------------------------------------


def _top_n(score: pd.Series, n: int) -> tuple[str, ...]:
    """The ``n`` highest-scoring names, ties broken by ticker ascending."""
    ordered = sorted(
        ((str(t), float(v)) for t, v in score.items()),
        key=lambda kv: (-kv[1], kv[0]),
    )
    return tuple(sorted(t for t, _v in ordered[:n]))


def _rotated_slice(universe: Sequence[str], n: int, cycle_index: int) -> tuple[str, ...]:
    """``n`` names from a ticker-sorted universe, rotated by the cycle index.

    The equal-score / no-ranking selector. Pure and reproducible: the same
    inputs always produce the same names. Rotating by ``cycle_index * n`` (mod
    the universe size) walks the window across the whole cross-section instead
    of re-buying the alphabetically-first names on every cycle — see this
    module's docstring for why that matters to the paired bootstrap.
    """
    ordered = sorted(str(t) for t in universe)
    size = len(ordered)
    if size == 0 or n <= 0:
        return ()
    start = (cycle_index * n) % size
    return tuple(sorted({ordered[(start + k) % size] for k in range(min(n, size))}))


# --------------------------------------------------------------------------
# research_composite_ic
# --------------------------------------------------------------------------


def _no_enter(inputs: ReplayInputs) -> NotAvailable:
    return no_live_selection(
        inputs, ISSUE, needs="there is no live width to count-match against"
    )


def build_research_composite_ic_arms(inputs: ReplayInputs) -> ArmSet | NotAvailable:
    """Top-N by the live attractiveness composite vs the same width, unranked."""
    live_width = live_picks_by_date(inputs)
    if not live_width:
        return _no_enter(inputs)
    if not inputs.pillar_profiles_by_date:
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                f"no pillar profiles loaded for any of the {len(inputs.dates)} "
                f"cycles (s3://{inputs.bucket}/{_PROFILES_KEY}) — the live "
                f"attractiveness composite cannot be rebuilt ({ISSUE})"
            ),
        )

    columns = set(map(str, inputs.price_matrix.columns))
    baseline_cycles: list[dict] = []
    ablated_cycles: list[dict] = []
    for index, date in enumerate(sorted(live_width)):
        n = len(live_width[date])
        profiles = inputs.pillar_profiles_by_date.get(date) or {}
        universe = sorted(str(t) for t in profiles if str(t) in columns)
        if len(universe) < n:
            continue
        composite = _pillar_composite(profiles, universe)
        if composite is None or len(composite) < n or composite.nunique() < 2:
            continue
        baseline_cycles.append({"date": date, "picks": _top_n(composite, n)})
        ablated_cycles.append({
            "date": date,
            "picks": _rotated_slice(list(composite.index), n, index),
        })

    if not baseline_cycles:
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                f"none of the {len(live_width)} ENTER cycles has a priceable "
                f"pillar cross-section wide enough, and varying enough, to rank "
                f"(s3://{inputs.bucket}/{_PROFILES_KEY}) ({ISSUE})"
            ),
        )
    return ArmSet(
        baseline=picks_arm("live attractiveness composite, top-N", baseline_cycles),
        ablated=picks_arm(
            "every pillar replaced by its cross-sectional mean "
            "(constant composite; cycle-rotated ticker tie-break)",
            ablated_cycles,
        ),
    )


# --------------------------------------------------------------------------
# neutralization_live_efficacy
# --------------------------------------------------------------------------


def _load_loadings(inputs: ReplayInputs) -> dict:
    """``{(date, ticker): {exposure: value}}`` for the neutralization arms.

    Read-only ArcticDB feature history via the single existing reader. Memoized
    per (bucket, window) so the two neutralization arms and any future caller
    share one universe load. Indirected through this function so tests can
    substitute a fixture without reaching AWS.
    """
    from analysis.end_to_end import (
        DEFAULT_NEUTRALIZE_FACTORS,
        load_historical_factor_loadings,
    )

    key = (inputs.bucket, tuple(inputs.dates))
    cached = _LOADINGS_CACHE.get(key)
    if cached is None:
        cached = load_historical_factor_loadings(
            inputs.bucket, inputs.dates, DEFAULT_NEUTRALIZE_FACTORS
        )
        _LOADINGS_CACHE[key] = cached
    return cached


_LOADINGS_CACHE: dict[tuple, dict] = {}


def build_neutralization_live_efficacy_arms(
    inputs: ReplayInputs,
) -> ArmSet | NotAvailable:
    """Neutralized composite vs raw composite — the residualizer's own lift."""
    from analysis.end_to_end import DEFAULT_NEUTRALIZE_FACTORS, _xs_neutralize

    live_width = live_picks_by_date(inputs)
    if not live_width:
        return _no_enter(inputs)
    if not inputs.pillar_profiles_by_date:
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                f"no pillar profiles loaded for any of the {len(inputs.dates)} "
                f"cycles (s3://{inputs.bucket}/{_PROFILES_KEY}) — there is no "
                f"composite to residualize ({ISSUE})"
            ),
        )

    loadings = _load_loadings(inputs)
    if not loadings:
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                "no factor exposures returned by "
                "end_to_end.load_historical_factor_loadings over the ArcticDB "
                f"universe feature history for bucket {inputs.bucket} "
                f"(needs {', '.join(DEFAULT_NEUTRALIZE_FACTORS)}) — the "
                f"neutralized arm cannot be built ({ISSUE})"
            ),
        )

    factors = list(DEFAULT_NEUTRALIZE_FACTORS)
    columns = set(map(str, inputs.price_matrix.columns))
    neutralized_cycles: list[dict] = []
    raw_cycles: list[dict] = []
    n_cycles_seen = 0
    for date in sorted(live_width):
        n = len(live_width[date])
        profiles = inputs.pillar_profiles_by_date.get(date) or {}
        universe = sorted(str(t) for t in profiles if str(t) in columns)
        if len(universe) < n:
            continue
        composite = _pillar_composite(profiles, universe)
        if composite is None or len(composite) < n or composite.nunique() < 2:
            continue
        n_cycles_seen += 1
        scores = {str(t): float(v) for t, v in composite.items()}
        exposures = {t: (loadings.get((date, t)) or {}) for t in scores}
        neutralized = _xs_neutralize(scores, exposures, factors)
        engaged = any(
            abs(neutralized[t] - scores[t]) > _IDENTITY_TOL for t in scores
        )
        if not engaged:
            # Identity passthrough: the residualizer never ran on this cycle
            # (its own name floor, or exposures missing for too many names).
            # Both arms would be the same portfolio and the recorded 0.0 would
            # read as "neutralization contributes nothing" — which is not what
            # happened. Dropped from BOTH arms so the cycle sets stay identical.
            continue
        neutralized_cycles.append({
            "date": date,
            "picks": _top_n(pd.Series(neutralized), n),
        })
        raw_cycles.append({"date": date, "picks": _top_n(composite, n)})

    if not neutralized_cycles:
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                f"the cross-sectional residualizer passed through as the "
                f"identity on all {n_cycles_seen} rankable cycles — too few "
                f"names carry the full {len(factors)}-factor exposure set in "
                f"the ArcticDB universe feature history, so there is no "
                f"neutralized arm distinct from the raw one ({ISSUE})"
            ),
        )
    return ArmSet(
        baseline=picks_arm(
            "Barra-neutralized composite, top-N "
            f"({'/'.join(factors)})",
            neutralized_cycles,
        ),
        ablated=picks_arm("raw composite, neutralization bypassed", raw_cycles),
    )


# --------------------------------------------------------------------------
# scanner_feed_counterfactual
# --------------------------------------------------------------------------


def _pre_scanner_population(research_db_path: str) -> dict[str, tuple[str, ...]]:
    """``{eval_date: (tickers evaluated,)}`` — the population BEFORE the rank.

    Every ``scanner_evaluations`` row for a research cycle is a name the
    scanner had in front of it, pass or fail; that whole set is the
    pre-scanner PIT population. Opened read-only. Fails LOUD: a missing table
    or an unreadable database is a real absence and must surface as one, never
    as a silently narrower population.
    """
    conn = sqlite3.connect(f"file:{research_db_path}?mode=ro", uri=True)
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "scanner_evaluations" not in tables:
            return {}
        frame = pd.read_sql_query(
            "SELECT DISTINCT ticker, eval_date FROM scanner_evaluations "
            "WHERE ticker IS NOT NULL AND eval_date IS NOT NULL",
            conn,
        )
    finally:
        conn.close()
    if frame.empty:
        return {}
    return {
        str(date): tuple(sorted({str(t) for t in group["ticker"]}))
        for date, group in frame.groupby("eval_date")
    }


def _live_scanner_population(inputs: ReplayInputs) -> dict[str, tuple[str, ...]]:
    """``{run_date: (tickers evaluated,)}`` from the live Scanner's published log.

    Reads ``candidates/{run_date}/candidates.json::scanner_eval_log`` through
    ``end_to_end._load_live_scanner_cuts`` — the reader ``scanner_lift_live``
    already uses — so both components see the same cohorts (one per ISO week,
    the latest artifact that week). Every evaluated name is kept, pass or fail:
    the rejected names are the population the ranking was applied to.

    A directly-constructed :class:`ReplayInputs` with no ``s3_client`` reads
    nothing here, the same posture ``predictor_ensemble._load_manifest``
    takes; ``load_replay_inputs`` always supplies one.
    """
    if inputs.s3_client is None:
        return {}
    from analysis.end_to_end import _load_live_scanner_cuts

    frame, _provenance = _load_live_scanner_cuts(
        inputs.bucket, s3_client=inputs.s3_client
    )
    if frame.empty:
        return {}
    return {
        str(date): tuple(sorted({str(t) for t in group["ticker"]}))
        for date, group in frame.groupby("eval_date")
    }


def _merged_population(
    live: Mapping[str, tuple[str, ...]],
    retired: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    """The live record, with the retired table filling only the dates before it.

    From the first published live cohort on, the live record is the only
    source: a ``scanner_evaluations`` row dated after it would be the retired
    writer's population interleaved with the live one, and ``_as_of`` would
    pick whichever happened to be newer.
    """
    if not live:
        return dict(retired)
    first_live = min(live)
    merged = {d: names for d, names in retired.items() if d < first_live}
    merged.update(live)
    return merged


def _as_of(eval_dates: Sequence[str], cycle_date: str) -> str | None:
    """The last research cycle at or before ``cycle_date`` (PIT-correct)."""
    prior = [d for d in eval_dates if d <= cycle_date]
    return prior[-1] if prior else None


def build_scanner_feed_counterfactual_arms(
    inputs: ReplayInputs,
) -> ArmSet | NotAvailable:
    """The live selection vs the same width drawn unranked from its own feed."""
    live_width = live_picks_by_date(inputs)
    if not live_width:
        return _no_enter(inputs)

    live_source = _LIVE_POPULATION_SOURCE.format(bucket=inputs.bucket)
    path = inputs.research_db_path
    db_present = bool(path) and Path(path).exists()

    # alpha-engine-config-I12231: the live Scanner has published the whole
    # pre-scanner population every cycle since 2026-07-10. This component read
    # only the retired table, so it went N/A on every card while the producer
    # was emitting exactly the input it said was missing.
    live = _live_scanner_population(inputs)
    retired = _pre_scanner_population(path) if db_present else {}
    population = _merged_population(live, retired)
    if not population:
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                "the pre-scanner PIT population is empty in both sources: no "
                f"published cohort in {live_source}, and "
                + (
                    f"scanner_evaluations in {path} holds no usable "
                    "(ticker, eval_date) rows"
                    if db_present
                    else f"research.db is not available (research_db_path={path!r}) "
                    "for the pre-2026-07-10 scanner_evaluations record"
                )
                + f" — there is nothing to draw the unranked arm from ({ISSUE})"
            ),
        )

    eval_dates = sorted(population)

    # alpha-engine-config-I8757. `_as_of` takes the last research cycle at or
    # before each replayed cycle. If the newest population cycle is far behind
    # the replay, every later cycle draws its unranked arm from that one old
    # population — a live selection measured against a universe that has not
    # existed for weeks, reported as a lift number. That was permanent while
    # the retired table was the only source; with the live record it fires
    # only if the Scanner stops publishing. A stale population is a MISSING
    # current population, so it takes the existing N/A status rather than
    # inventing a vocabulary the report card has never seen.
    newest_replayed = sorted(live_width)[-1]
    population_age = retired_table_guard.age_days(eval_dates[-1], newest_replayed)
    if (population_age is not None
            and population_age > retired_table_guard.DEFAULT_MAX_AGE_DAYS):
        newest_source = (
            live_source if eval_dates[-1] in live
            else f"scanner_evaluations in {path}"
        )
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                f"the pre-scanner PIT population is frozen: its newest cycle "
                f"is {eval_dates[-1]} ({newest_source}), "
                f"{population_age}d before the newest replayed ENTER cycle "
                f"{newest_replayed} (bound "
                f"{retired_table_guard.DEFAULT_MAX_AGE_DAYS}d). "
                + (
                    "No live cohort is published in "
                    f"{live_source}, and the scanner_evaluations writer was "
                    "retired 2026-07-12 with the rest of the research graph "
                    "(alpha-engine-config-I8757), "
                    if not live
                    else "The live Scanner has stopped publishing its eval log, "
                )
                + "so the unranked arm would be drawn from a universe that no "
                f"longer exists and the lift would not be a measurement of "
                f"this window ({ISSUE})"
            ),
        )
    columns = set(map(str, inputs.price_matrix.columns))
    baseline_cycles: list[dict] = []
    ablated_cycles: list[dict] = []
    drawn_from_live = 0
    for index, date in enumerate(sorted(live_width)):
        picks = live_width[date]
        n = len(picks)
        as_of = _as_of(eval_dates, date)
        if as_of is None:
            continue
        pool = sorted(t for t in population[as_of] if t in columns)
        if len(pool) < n:
            continue
        drawn_from_live += as_of in live
        baseline_cycles.append({"date": date, "picks": picks})
        ablated_cycles.append({
            "date": date,
            "picks": _rotated_slice(pool, n, index),
        })

    if not baseline_cycles:
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                f"none of the {len(live_width)} ENTER cycles has a research "
                f"cycle at or before it whose priceable pre-scanner "
                f"population is at least as wide as the live selection "
                f"({len(live)} cycles from {live_source}, "
                f"{len(eval_dates) - len(live)} earlier from "
                f"scanner_evaluations in {path}) ({ISSUE})"
            ),
        )
    return ArmSet(
        baseline=picks_arm(
            f"live scanner-fed selection — {live_selection_label(inputs)}",
            baseline_cycles,
        ),
        ablated=picks_arm(
            "same width drawn unranked from the pre-scanner PIT population "
            "(cycle-rotated ticker order; "
            f"{drawn_from_live}/{len(ablated_cycles)} cycles from the live "
            "candidates.json scanner_eval_log, the rest from the retired "
            "scanner_evaluations table)",
            ablated_cycles,
        ),
    )


# --------------------------------------------------------------------------
# Retired components
# --------------------------------------------------------------------------


def _retired(component: str, producer: str, replacement: str):
    def build_arms(inputs: ReplayInputs) -> NotAvailable:
        from analysis.end_to_end import RESEARCH_GRAPH_RETIRED_DATE

        return NotAvailable(
            status="N/A-RETIRED",
            reason=(
                f"{component}'s only source is {producer}, part of the six-team "
                f"/ CIO research graph retired on {RESEARCH_GRAPH_RETIRED_DATE} "
                "(analysis/end_to_end.RESEARCH_GRAPH_RETIRED_DATE). The "
                "producer no longer writes it and "
                f"{replacement}, so there is no arm to count-match and no "
                f"substitute selection to replay ({ISSUE})"
            ),
        )

    return build_arms


# --------------------------------------------------------------------------
# Specs
# --------------------------------------------------------------------------

SPEC_RESEARCH_COMPOSITE_IC = ReplaySpec(
    name="research_composite_ic",
    module="research",
    criticality="critical",
    pattern="substitution",
    issue=ISSUE,
    build_arms=build_research_composite_ic_arms,
)

SPEC_SECTOR_TEAMS_AVG = ReplaySpec(
    name="sector_teams_avg",
    module="research",
    criticality="critical",
    pattern="substitution",
    issue=ISSUE,
    build_arms=_retired(
        "sector_teams_avg",
        "the retired per-team candidate table",
        "nothing has replaced the per-team selection record",
    ),
)

SPEC_CIO_SELECTION_SKILL = ReplaySpec(
    name="cio_selection_skill",
    module="research",
    criticality="critical",
    pattern="substitution",
    issue=ISSUE,
    build_arms=_retired(
        "cio_selection_skill",
        "the retired CIO evaluation table",
        "nothing has replaced the CIO accept/reject record",
    ),
)

SPEC_NEUTRALIZATION_LIVE_EFFICACY = ReplaySpec(
    name="neutralization_live_efficacy",
    module="research",
    criticality="critical",
    pattern="substitution",
    issue=ISSUE,
    build_arms=build_neutralization_live_efficacy_arms,
)

SPEC_SCANNER_FEED_COUNTERFACTUAL = ReplaySpec(
    name="scanner_feed_counterfactual",
    module="research",
    criticality="supporting",
    pattern="substitution",
    issue=ISSUE,
    build_arms=build_scanner_feed_counterfactual_arms,
)

SPECS: list[ReplaySpec] = [
    SPEC_RESEARCH_COMPOSITE_IC,
    SPEC_SECTOR_TEAMS_AVG,
    SPEC_CIO_SELECTION_SKILL,
    SPEC_NEUTRALIZATION_LIVE_EFFICACY,
    SPEC_SCANNER_FEED_COUNTERFACTUAL,
]
