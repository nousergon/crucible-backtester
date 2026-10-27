"""harness.py — the shared contribution-lift replay harness (RC v3 T5).

Report Card v3 §1 fixes ONE objective for the whole system:

    objective(cycle) = log_return_21d(portfolio, net_of_cost) − log_return_21d(SPY)

and §3 replaces every declared component weight with a **measured marginal
contribution** to that objective: replay the pipeline twice — once as
configured (the *baseline* arm), once with the component ablated (the
*ablated* arm) — and report the paired per-cycle difference.

This module is the machinery every T5 group shares. A group contributes ONE
:class:`ReplaySpec` to ``registry.SPECS``; the spec's ``build_arms`` callable
turns the live :class:`ReplayInputs` into an :class:`ArmSet` (or a
:class:`NotAvailable` when its input does not exist), and everything after
that — sizing, simulation, the objective, the paired bootstrap CI, count
matching, DSR/PBO, the emitted component dict — happens here exactly once.

Two arm shapes are supported, because "removing" a component means two
different things (spec §3):

* **picks** — ``[{"date": "YYYY-MM-DD", "picks": ["AAPL", ...]}, ...]``.
  The harness applies its own deterministic equal-weight sizing and a fixed
  hold of ``horizon_days`` trading rows. This is the *substitution* pattern:
  research-side arms differ only in WHICH names were selected, so a shared,
  boring sizing rule is exactly what isolates selection from sizing.
* **orders** — an explicit ``[{"date", "ticker", "action", "shares",
  "price_at_order"}, ...]`` list. This is the *null_arm* pattern: an
  executor-side arm has to reproduce real fills and real sizes, because the
  thing being ablated (a guard, an exit rule, a sizer) acts ON those.

Both shapes go through the SAME cost-bearing simulator
(``vectorbt_bridge.orders_to_portfolio`` + ``portfolio_stats``) that
``backtest._run_simulation_loop`` uses, at the config's own ``fees`` /
``slippage_bps`` — a replay that skips the cost model is measuring a proxy,
not the objective (spec §1).

Determinism: no RNG anywhere except the bootstrap's fixed ``seed=0``; ties in
selection are broken by ticker ascending.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from nousergon_lib.quant.horizons import DEFAULT_POLICY

from analysis.contribution_lift import objective as _objective

logger = logging.getLogger(__name__)

#: The fleet's canonical primary horizon (21 trading days). Never hardcoded —
#: the objective is defined against whatever DEFAULT_POLICY says it is.
HORIZON_DAYS: int = int(DEFAULT_POLICY.primary_horizon)

#: Cycles below this cannot support a lift claim at all (spec §1: 0.5 × n_floor).
MIN_SAMPLES: int = 30

#: The objective's own floor, inherited from the portfolio_outcome tile.
N_FLOOR: int = 60

#: The unit every contribution_lift record carries (krepis#158 `unit` field).
UNIT: str = "log_alpha_21d"

#: trial_accumulator producer name for this harness.
TRIAL_PRODUCER: str = "contribution_lift"

#: Version of the additive per-component ``support`` diagnostics block
#: (alpha-engine-config-I12243). Bumped on any change to its keys or meaning.
SUPPORT_SCHEMA_VERSION: int = 1

#: The estimand every measured record carries, declared rather than implied.
#: Each paired sample is the WHOLE-BOOK log-alpha difference over the
#: ``horizon_days`` rows after one window date in ``ReplayInputs.dates`` — not
#: a per-entry cohort return, and not one independent selection event per
#: sample. ``n_samples`` counts sampled windows.
ESTIMAND: dict = {
    "name": "whole_book_window_log_alpha_difference",
    "unit_of_sample": "sampled whole-book horizon window per ReplayInputs.dates entry",
    "per_entry_cohort": False,
    "samples_are_independent_selection_events": False,
}

#: Provenance the support block does NOT yet carry, named so its absence is
#: explicit on the artifact instead of read as "checked and fine".
SUPPORT_NOT_COVERED: tuple[str, ...] = (
    "point_in_time_coefficient_provenance",
)

_VALID_CRITICALITY = ("critical", "supporting", "diagnostic")
_VALID_PATTERN = ("substitution", "null_arm")
_VALID_NA_STATUS = (
    "N/A-MISSING-INPUT",
    "N/A-LOW-N",
    "N/A-RETIRED",
    "N/A-NOT-LIFT-SHAPED",
)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


@dataclass
class ReplayInputs:
    """Everything a ``build_arms`` callable is allowed to read.

    Assembled ONCE per run by :func:`analysis.contribution_lift.inputs.
    load_replay_inputs` and handed to every spec, so eight sibling groups
    share one S3/ArcticDB load rather than each doing their own.

    ``status``/``reason`` carry the loader's own verdict: ``"ok"`` with a
    populated window, or ``"skipped"`` with a reason when there were zero
    usable signal dates. An observational artifact is emitted either way —
    absence of ``contribution_lift.json`` must always mean "the producer
    never ran" (reporter.py's always-emit contract).

    ``live_selection`` is the champion-aware ground truth for "which names did
    this system actually open on this cycle" — see :func:`build_live_selection`.
    It is assembled once by the loader; a directly-constructed ``ReplayInputs``
    (tests, ad-hoc replays) leaves it ``None`` and :func:`live_picks_by_date`
    derives it lazily from the same two feeds.
    """

    run_date: str
    dates: list[str]
    signals_by_date: dict[str, dict]
    predictions_by_date: dict[str, dict]
    pillar_profiles_by_date: dict[str, dict]
    price_matrix: pd.DataFrame
    spy_prices: pd.Series
    bucket: str
    fees: float
    slippage_bps: float
    init_cash: float
    trades_db_path: str | None = None
    research_db_path: str | None = None
    s3_client: Any = None
    status: str = "ok"
    reason: str | None = None
    source_paths: list[str] = field(default_factory=list)
    live_selection: "LiveSelection | None" = None

    @property
    def horizon_days(self) -> int:
        return HORIZON_DAYS


# --------------------------------------------------------------------------
# Live selection — the champion-aware baseline (config-I7501)
# --------------------------------------------------------------------------


#: ``trades.db`` action value for an opened position. The same table and
#: discriminator ``analysis/post_trade.py`` and ``analysis/shadow_book.py``
#: already read — reused, never re-derived.
_ENTER = "ENTER"

#: Feed identifiers carried on :class:`LiveSelection.source` and surfaced on
#: every baseline arm's label, so an artifact reader can tell which record the
#: comparison was anchored to without re-deriving it.
SOURCE_TRADES = "trades.db"
SOURCE_SIGNALS = "signals.json"


@dataclass(frozen=True)
class LiveSelection:
    """Which names the system actually opened, per cycle, and from which feed.

    **Why this is not simply ``signals.json``.** Entry selection moved OUT of
    the signals producer on 2026-07-13, when ``scanner_predictor_direct``
    became the champion arm (config-I7216): the executor now draws its
    candidates directly from the predictor and ``signals/{date}/signals.json``
    has carried **zero ENTER rows on every cycle since 2026-07-18** — correctly,
    not as a fault. A replay whose baseline arm reads that feed therefore
    measures an empty book on every recent cycle and grades green while
    measuring nothing (config-I7501).

    The executed record in ``trades.db`` is the ground truth in BOTH eras: it
    is what the pre-champion signals path produced *and* what the champion arm
    produces, so it needs no era switch and no champion-name lookup. The
    ``signals.json`` reader survives only as the fallback for a run with no
    ``trades_db_path`` at all, and the arm label names which one was used.
    """

    source: str
    by_date: Mapping[str, tuple[str, ...]]

    @property
    def label(self) -> str:
        if self.source == SOURCE_TRADES:
            return "live executed selection (trades.db ENTER)"
        return "live signalled selection (signals.json ENTER)"


def read_trades_enters(trades_db_path: str) -> dict[str, tuple[str, ...]]:
    """``{cycle_date: (tickers entered,)}`` from ``trades.db``.

    Same SQL as ``analysis/post_trade.py``. "no such table" is the only
    recoverable schema state (a fresh trades.db with no history yet); anything
    else propagates (contract §Rules: fail loud, no silent swallow).
    """
    if not trades_db_path or not Path(trades_db_path).exists():
        return {}
    conn = sqlite3.connect(f"file:{trades_db_path}?mode=ro", uri=True)
    try:
        frame = pd.read_sql_query(
            f"SELECT date, ticker FROM trades WHERE action = '{_ENTER}'", conn
        )
    except pd.errors.DatabaseError as exc:
        msg = str(exc).lower()
        if "no such table" in msg or "no such column" in msg:
            return {}
        raise
    finally:
        conn.close()
    if frame.empty:
        return {}
    out: dict[str, set[str]] = {}
    for row in frame.itertuples(index=False):
        ticker = str(row.ticker) if row.ticker is not None else ""
        # trades.db stores a date string; a timestamped value is truncated to
        # its calendar day so it joins the cycle axis the price matrix uses.
        date = str(row.date)[:10] if row.date is not None else ""
        if ticker and date:
            out.setdefault(date, set()).add(ticker)
    return {d: tuple(sorted(t)) for d, t in sorted(out.items())}


def read_signals_enters(signals_by_date: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """``{cycle_date: (ENTER-signalled tickers,)}`` from the signals artifact.

    The pre-champion feed, kept only as the no-trades.db fallback. Reads it
    exactly as ``loaders/signal_loader`` documents it: ``signals`` is a
    ``{ticker: row}`` mapping (legacy payloads emit a list) and the
    discriminator is ``row["signal"]``.
    """
    out: dict[str, tuple[str, ...]] = {}
    for date, body in (signals_by_date or {}).items():
        raw = (body or {}).get("signals") or {}
        rows = list(raw.values()) if isinstance(raw, dict) else list(raw)
        picks = sorted({
            row["ticker"]
            for row in rows
            if isinstance(row, dict)
            and isinstance(row.get("ticker"), str)
            and str(row.get("signal", "")).upper() == _ENTER
        })
        if picks:
            out[str(date)] = tuple(picks)
    return out


def build_live_selection(
    *,
    trades_db_path: str | None = None,
    signals_by_date: Mapping[str, Any],
    trades_enters: Mapping[str, tuple[str, ...]] | None = None,
) -> LiveSelection:
    """The champion-aware live selection, preferring the executed record.

    ``trades.db`` wins whenever it is present and carries ENTER rows. The
    signals fallback exists for a replay run without a trades database at all
    (an isolated unit test, an ad-hoc historical replay) — never as a silent
    substitute when the executed record is merely empty for the window, which
    is a real absence and is reported as one by the groups.

    ``trades_enters`` lets a caller that has already read the database (the
    loader, which needs the ENTER dates to choose its window) hand the result
    in rather than pay a second read of the same file.
    """
    if trades_db_path or trades_enters is not None:
        # The run HAS an executed record. Whatever it says is the answer,
        # including "nothing" — an empty trades.db over the window is a real
        # absence, and quietly substituting the signals feed there would put
        # the pre-champion feed back in the baseline by the back door.
        by_date = (
            dict(trades_enters)
            if trades_enters is not None
            else read_trades_enters(trades_db_path)
        )
        return LiveSelection(source=SOURCE_TRADES, by_date=by_date)
    return LiveSelection(
        source=SOURCE_SIGNALS, by_date=read_signals_enters(signals_by_date)
    )


def live_selection_of(inputs: ReplayInputs) -> LiveSelection:
    """The run's :class:`LiveSelection`, derived lazily when the loader did not."""
    if inputs.live_selection is not None:
        return inputs.live_selection
    return build_live_selection(
        trades_db_path=inputs.trades_db_path,
        signals_by_date=inputs.signals_by_date,
    )


def live_picks_by_date(inputs: ReplayInputs) -> dict[str, tuple[str, ...]]:
    """``{cycle_date: (tickers,)}`` for the cycles in this run's window.

    Restricted to the run's ``dates`` and to names the price matrix can
    actually price — an unpriceable name is silently dropped by
    :func:`picks_to_orders`, so counting it here would make the declared width
    disagree with the traded width and turn a correct comparison into a
    spurious count-match gap.
    """
    selection = live_selection_of(inputs)
    columns = {str(c) for c in inputs.price_matrix.columns}
    window = set(inputs.dates)
    out: dict[str, tuple[str, ...]] = {}
    for date, tickers in selection.by_date.items():
        if window and date not in window:
            continue
        priceable = tuple(t for t in tickers if not columns or t in columns)
        if priceable:
            out[date] = priceable
    return out


def live_picks_by_cycle(inputs: ReplayInputs) -> list[dict]:
    """``[{"date", "picks"}, ...]`` — the picks-arm shape of the live selection."""
    return [
        {"date": date, "picks": list(picks)}
        for date, picks in sorted(live_picks_by_date(inputs).items())
    ]


def live_widths(inputs: ReplayInputs) -> dict[str, int]:
    """``{cycle_date: n_names opened}`` — the width every arm is matched to."""
    return {d: len(t) for d, t in live_picks_by_date(inputs).items()}


def live_selection_label(inputs: ReplayInputs) -> str:
    """The baseline arm's feed name, for the artifact's ``arms.baseline.label``."""
    return live_selection_of(inputs).label


def no_live_selection(inputs: ReplayInputs, issue: str, *, needs: str) -> NotAvailable:
    """The shared ``N/A-MISSING-INPUT`` for a window with no live selection.

    Names the feed that was actually read, so the artifact distinguishes "the
    executed record is empty over this window" from "this run had no trades
    database and the retired signals feed was consulted instead".
    """
    selection = live_selection_of(inputs)
    return NotAvailable(
        status="N/A-MISSING-INPUT",
        reason=(
            f"no cycle in the {len(inputs.dates)}-date window carries a "
            f"priceable live selection from {selection.source} "
            f"({selection.label}) — {needs} ({issue})"
        ),
    )


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One replayed configuration of the pipeline.

    Exactly one of ``picks`` / ``orders`` must be set.

    ``fees`` / ``slippage_bps`` default to the run's configured cost model
    (``None`` ⇒ inherit). They exist so a group can ablate the COST MODEL
    itself (the zero-cost arm behind ``cost_adjusted_quality``); every other
    group leaves them ``None`` so both arms are priced identically and the
    measured difference is attributable to the component alone.
    """

    label: str
    picks: tuple[tuple[str, tuple[str, ...]], ...] | None = None
    orders: tuple[Mapping[str, Any], ...] | None = None
    fees: float | None = None
    slippage_bps: float | None = None

    def __post_init__(self) -> None:
        if (self.picks is None) == (self.orders is None):
            raise ValueError(
                f"Arm {self.label!r}: exactly one of picks/orders must be set "
                f"(got picks={self.picks is not None}, orders={self.orders is not None})"
            )


def picks_arm(
    label: str,
    picks_by_cycle: Sequence[Mapping[str, Any]],
    *,
    fees: float | None = None,
    slippage_bps: float | None = None,
) -> Arm:
    """Build a picks-shaped :class:`Arm` from ``[{"date", "picks"}, ...]``.

    Normalizes to an immutable, ticker-sorted structure so two runs over the
    same inputs produce byte-identical arms (determinism, contract §Rules).
    """
    normalized: list[tuple[str, tuple[str, ...]]] = []
    for cycle in picks_by_cycle:
        d = str(cycle["date"])
        tickers = tuple(sorted({str(t) for t in (cycle.get("picks") or [])}))
        normalized.append((d, tickers))
    normalized.sort(key=lambda row: row[0])
    return Arm(
        label=label,
        picks=tuple(normalized),
        fees=fees,
        slippage_bps=slippage_bps,
    )


def orders_arm(
    label: str,
    orders: Sequence[Mapping[str, Any]],
    *,
    fees: float | None = None,
    slippage_bps: float | None = None,
) -> Arm:
    """Build an orders-shaped :class:`Arm` from an explicit order list."""
    ordered = sorted(
        (dict(o) for o in orders),
        key=lambda o: (str(o.get("date")), str(o.get("ticker")), str(o.get("action"))),
    )
    return Arm(
        label=label,
        orders=tuple(ordered),
        fees=fees,
        slippage_bps=slippage_bps,
    )


@dataclass(frozen=True)
class ArmSet:
    """The arms one spec wants replayed.

    ``sweep`` carries ADDITIONAL arms beyond the baseline/ablated pair. It is
    what distinguishes a grid from a single leave-one-out comparison: PBO is
    reported only when ``sweep`` is non-empty, because CSCV-PBO answers
    "is this the cherry-picked winner of a search?" and a two-arm
    leave-one-out ran no search (spec §1).
    """

    baseline: Arm
    ablated: Arm
    sweep: tuple[Arm, ...] = ()

    def all_arms(self) -> tuple[Arm, ...]:
        return (self.baseline, self.ablated, *self.sweep)


@dataclass(frozen=True)
class NotAvailable:
    """A spec declining to measure, with the reason an operator needs.

    Never a fabricated value: a component whose input does not exist (retired
    producer, per-model predictions never persisted, a metric that is not
    lift-shaped by construction) emits an ``N/A-*`` component naming the
    missing artifact and its tracking issue.
    """

    status: str
    reason: str

    def __post_init__(self) -> None:
        if self.status not in _VALID_NA_STATUS:
            raise ValueError(
                f"NotAvailable.status must be one of {_VALID_NA_STATUS}, got {self.status!r}"
            )


# --------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplaySpec:
    """One graded component's replay.

    Attributes:
        name: the evaluator tile component name — the join key between this
            artifact and ``crucible-evaluator``'s tile. Must match exactly.
        module: owning tile (``research`` / ``predictor`` / ``executor`` /
            ``agent`` / ``behavioral``).
        criticality: inherited from the component's existing tile record
            (spec §3) — never re-derived here.
        pattern: ``substitution`` (re-score/re-select) or ``null_arm``
            (disable a gate), per spec §3's two named patterns.
        issue: ``alpha-engine-config-I<N>`` this replay closes.
        build_arms: pure function of :class:`ReplayInputs`. Returns an
            :class:`ArmSet` to be measured, or :class:`NotAvailable`.
    """

    name: str
    module: str
    criticality: str
    pattern: str
    issue: str
    build_arms: Callable[[ReplayInputs], "ArmSet | NotAvailable"]

    def __post_init__(self) -> None:
        if self.criticality not in _VALID_CRITICALITY:
            raise ValueError(
                f"{self.name}: criticality must be one of {_VALID_CRITICALITY}, "
                f"got {self.criticality!r}"
            )
        if self.pattern not in _VALID_PATTERN:
            raise ValueError(
                f"{self.name}: pattern must be one of {_VALID_PATTERN}, "
                f"got {self.pattern!r}"
            )


# --------------------------------------------------------------------------
# Sizing / order construction
# --------------------------------------------------------------------------


def picks_to_orders(
    picks: Sequence[tuple[str, Sequence[str]]],
    price_matrix: pd.DataFrame,
    *,
    init_cash: float,
    per_cycle_slots: int,
    hold_days: int = HORIZON_DAYS,
) -> list[dict]:
    """Deterministic equal-weight order list for a picks-shaped arm.

    Every name gets ``init_cash / per_cycle_slots`` of notional and is held
    for ``hold_days`` trading rows. ``per_cycle_slots`` is computed ONCE
    across every arm of the set (see :func:`_slot_count`) so two arms compared
    against each other are sized identically — a per-arm budget would let the
    narrower arm silently lever up and the "lift" would be a sizing artifact.

    Mirrors ``analysis.factor_blend_counterfactual_replay.build_orders_for_
    weights``: same order shape, same floor-to-whole-shares rule, same
    snap-forward when a cycle date is not itself a price row.
    """
    dates = price_matrix.index
    orders: list[dict] = []
    if len(dates) == 0:
        return orders
    per_name_budget = float(init_cash) / max(1, int(per_cycle_slots))

    for raw_date, tickers in picks:
        cdate = pd.Timestamp(raw_date)
        if cdate not in dates:
            future = dates[dates >= cdate]
            if len(future) == 0:
                continue
            cdate = future[0]
        entry_pos = dates.get_loc(cdate)
        exit_pos = min(entry_pos + hold_days, len(dates) - 1)
        exit_date = dates[exit_pos]

        for ticker in sorted(tickers):
            if ticker not in price_matrix.columns:
                continue
            entry_price = price_matrix.at[cdate, ticker]
            if pd.isna(entry_price) or float(entry_price) <= 0.0:
                continue
            shares = float(np.floor(per_name_budget / float(entry_price)))
            if shares <= 0.0:
                continue
            orders.append({
                "date": cdate.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "action": "ENTER",
                "shares": shares,
                "price_at_order": float(entry_price),
            })
            if exit_pos > entry_pos:
                orders.append({
                    "date": exit_date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "action": "EXIT",
                })
    return orders


def _slot_count(arm_set: ArmSet) -> int:
    """Widest per-cycle pick count across every arm — the shared sizing base."""
    widest = 0
    for arm in arm_set.all_arms():
        if arm.picks is None:
            continue
        for _d, tickers in arm.picks:
            widest = max(widest, len(tickers))
    return max(1, widest)


def arm_orders(arm: Arm, arm_set: ArmSet, inputs: ReplayInputs) -> list[dict]:
    """Resolve an arm to a concrete order list."""
    if arm.orders is not None:
        return [dict(o) for o in arm.orders]
    assert arm.picks is not None  # guaranteed by Arm.__post_init__
    return picks_to_orders(
        arm.picks,
        inputs.price_matrix,
        init_cash=inputs.init_cash,
        per_cycle_slots=_slot_count(arm_set),
        hold_days=inputs.horizon_days,
    )


def picks_per_cycle(arm: Arm) -> dict[str, int]:
    """``{cycle_date: n_names}`` for an arm, whichever shape it carries.

    For an orders-shaped arm the width of a cycle is the number of DISTINCT
    tickers entered on that date — which is what count-matching constrains.
    """
    if arm.picks is not None:
        return {d: len(t) for d, t in arm.picks}
    assert arm.orders is not None
    by_date: dict[str, set[str]] = {}
    for order in arm.orders:
        if str(order.get("action")) != "ENTER":
            continue
        by_date.setdefault(str(order.get("date")), set()).add(str(order.get("ticker")))
    return {d: len(t) for d, t in by_date.items()}


def _priceable(price_matrix: pd.DataFrame, when: pd.Timestamp, ticker: str) -> bool:
    """True when the simulator can fill ``ticker`` on ``when`` at all.

    ``vectorbt_bridge.orders_to_portfolio`` drops an order whose date is not a
    price row or whose ticker is not a column, and vectorbt cannot fill at a
    missing or non-positive close.
    """
    if ticker not in price_matrix.columns or when not in price_matrix.index:
        return False
    value = price_matrix.at[when, ticker]
    return not pd.isna(value) and float(value) > 0.0


def entry_census(arm: Arm, arm_set: ArmSet, inputs: ReplayInputs) -> dict:
    """Per declared date: names requested, ENTER orders generated, priceable.

    The three can differ, and only the first is what count-matching compares
    (alpha-engine-config-I12243): a picks arm's ``picks_to_orders`` drops a
    name with no positive close or a whole-share budget of zero, and the
    simulator drops an order-shaped ENTER it cannot price. Returned as::

        {"by_declared_date": {date: {"requested", "generated", "priceable"}},
         "priceable_by_order_date": {date: n}}

    ``priceable_by_order_date`` is keyed by the date the ENTER is actually
    placed on (a picks arm snaps a non-trading cycle date forward), which is
    the key observed fills are reported under.
    """
    pm = inputs.price_matrix
    by_declared: dict[str, dict[str, int]] = {}
    by_order_date: dict[str, int] = {}
    if arm.picks is not None:
        slots = _slot_count(arm_set)
        for d, tickers in arm.picks:
            enters = [
                o for o in picks_to_orders(
                    [(d, tickers)], pm,
                    init_cash=inputs.init_cash,
                    per_cycle_slots=slots,
                    hold_days=inputs.horizon_days,
                )
                if o["action"] == "ENTER"
            ]
            row = by_declared.setdefault(
                d, {"requested": 0, "generated": 0, "priceable": 0}
            )
            row["requested"] += len(tickers)
            row["generated"] += len(enters)
            # picks_to_orders only emits an ENTER it could price.
            row["priceable"] += len(enters)
            for o in enters:
                by_order_date[o["date"]] = by_order_date.get(o["date"], 0) + 1
    else:
        assert arm.orders is not None
        names: dict[str, set[str]] = {}
        for order in arm.orders:
            if str(order.get("action")) != "ENTER":
                continue
            d = str(order.get("date"))
            ticker = str(order.get("ticker"))
            names.setdefault(d, set()).add(ticker)
            row = by_declared.setdefault(
                d, {"requested": 0, "generated": 0, "priceable": 0}
            )
            row["generated"] += 1
            shares = order.get("shares")
            if (
                _priceable(pm, pd.Timestamp(d), ticker)
                and shares is not None
                and float(shares) > 0.0
            ):
                row["priceable"] += 1
                by_order_date[d] = by_order_date.get(d, 0) + 1
        for d, tickers in names.items():
            by_declared[d]["requested"] = len(tickers)
    return {
        "by_declared_date": dict(sorted(by_declared.items())),
        "priceable_by_order_date": dict(sorted(by_order_date.items())),
    }


def unpriceable_dates(censuses: Sequence[Mapping[str, Any]]) -> set[str]:
    """Declared dates on which NO arm declaring them can price one entry.

    Neither simulated book trades on such a date, so one-sided declared
    support there cannot bias the comparison; it is named ``unpriceable`` in
    the census instead of being reported as a support gap.
    """
    declared: dict[str, list[int]] = {}
    for census in censuses:
        for d, row in census["by_declared_date"].items():
            if row["requested"] > 0:
                declared.setdefault(d, []).append(row["priceable"])
    return {d for d, counts in declared.items() if not any(counts)}


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------


def _buy_fills_by_date(pf: Any, index: pd.Index) -> dict[str, int]:
    """``{date: buy fills}`` from the simulated portfolio's own order records.

    The executed-fill witness: an ENTER the simulator could not or did not
    fill (unpriceable, already held, no cash) is absent here even though it
    was generated.
    """
    records = pf.orders.values
    out: dict[str, int] = {}
    for idx, side in zip(records["idx"], records["side"]):
        if int(side) != 0:  # vectorbt OrderSide.Buy == 0
            continue
        d = pd.Timestamp(index[int(idx)]).strftime("%Y-%m-%d")
        out[d] = out.get(d, 0) + 1
    return dict(sorted(out.items()))


def has_partial_exits(orders: Sequence[Mapping[str, Any]]) -> bool:
    """True when any EXIT/REDUCE carries ``shares`` — a partial sell leg.

    Only then does the simulator run with ``partial_exits=True``; every other
    arm (picks arms, terminal-exit-only order arms) takes the default path
    unchanged (alpha-engine-config-I12683).
    """
    return any(
        str(o.get("action")) in ("EXIT", "REDUCE") and o.get("shares") is not None
        for o in orders
    )


def simulate_arm(arm: Arm, arm_set: ArmSet, inputs: ReplayInputs) -> dict:
    """Run one arm through the production simulator and return its stats.

    Returns the ``vectorbt_bridge.portfolio_stats`` dict verbatim plus
    ``n_orders`` — including its ``daily_log_returns`` Series, which is the
    ONLY series the objective is computed from. Sharpe/Sortino are carried
    through as diagnostics and are never used for status derivation while
    I7236 / I7237 / I7271 are open (spec §1 blocking note).
    """
    # Imported lazily: vectorbt is a heavy import and the unit tests that
    # exercise the objective/CI math never touch the simulator.
    from vectorbt_bridge import orders_to_portfolio, portfolio_stats

    orders = arm_orders(arm, arm_set, inputs)
    fees = inputs.fees if arm.fees is None else float(arm.fees)
    slippage_bps = (
        inputs.slippage_bps if arm.slippage_bps is None else float(arm.slippage_bps)
    )
    pf = orders_to_portfolio(
        orders,
        inputs.price_matrix,
        init_cash=inputs.init_cash,
        fees=fees,
        slippage_bps=slippage_bps,
        partial_exits=has_partial_exits(orders),
    )
    stats = portfolio_stats(pf, spy_prices=inputs.spy_prices)
    stats["n_orders"] = len(orders)
    stats["buy_fills_by_date"] = _buy_fills_by_date(pf, inputs.price_matrix.index)
    return stats


# --------------------------------------------------------------------------
# Running one spec
# --------------------------------------------------------------------------


def _na_component(
    spec: ReplaySpec, na: NotAvailable, *, run_date: str, bucket: str
) -> dict:
    return _component_skeleton(spec, run_date=run_date, bucket=bucket) | {
        "status": na.status,
        "status_reason": na.reason,
        "value": None,
        "ci_low": None,
        "ci_high": None,
        "ci_method": "bootstrap",
        "n_samples": 0,
        "count_matched": None,
        "arms": {},
        "dsr": None,
        "pbo": None,
        # Nothing was replayed, so there is no support to census.
        "support": None,
    }


def _component_skeleton(spec: ReplaySpec, *, run_date: str, bucket: str) -> dict:
    return {
        "name": spec.name,
        "module": spec.module,
        "criticality": spec.criticality,
        "pattern": spec.pattern,
        "issue": spec.issue,
        "unit": UNIT,
        "n_floor": N_FLOOR,
        "source_path": (
            f"s3://{bucket}/backtest/{run_date}/contribution_lift.json"
            f"#components/{spec.name}"
        ),
    }


def _arm_payload(
    label: str,
    stats: Mapping[str, Any],
    per_cycle: Mapping[str, float],
    width: Mapping[str, int],
    n_trials: int,
) -> dict:
    from nousergon_lib.quant.stats.dsr import compute_dsr

    widths = sorted(set(width.values()))
    daily_returns = stats.get("daily_returns")
    dsr_value: float | None = None
    if isinstance(daily_returns, pd.Series) and len(daily_returns.dropna()) > 2:
        # compute_dsr requires n_trials >= 1; a counter that has not been
        # written yet reads 0, which is "no trial history", not "zero trials".
        dsr_result = compute_dsr(daily_returns.dropna(), max(1, int(n_trials)))
        if dsr_result.get("status") == "ok":
            dsr_value = _finite(dsr_result.get("dsr"))
    return {
        "label": label,
        "n_orders": int(stats.get("n_orders", 0)),
        # A single integer when every cycle has the same width (the normal,
        # count-matched case); the observed set otherwise, so the gap's own
        # status_reason has something concrete to name.
        "picks_per_cycle": widths[0] if len(widths) == 1 else widths,
        "total_return": _finite(stats.get("total_return")),
        "total_alpha": _finite(stats.get("total_alpha")),
        "sortino_ratio": _finite(stats.get("sortino_ratio")),
        "sharpe_ratio": _finite(stats.get("sharpe_ratio")),
        "max_drawdown": _finite(stats.get("max_drawdown")),
        "psr": _finite(stats.get("psr")),
        "dsr": dsr_value,
        "per_cycle_log_alpha_21d": {d: float(v) for d, v in sorted(per_cycle.items())},
    }


def _finite(value: Any) -> float | None:
    """Coerce to a JSON-safe finite float, or None.

    ``+inf``/``NaN`` reaching an artifact is the config-I7237 failure mode
    (metrics.json emitting ``+inf`` for flat-market Sharpe and thereby not
    being strict JSON). Recorded as an explicit null instead — the consumer
    reads null as "not defined this run", which is true, and never parses a
    bare ``Infinity`` token.
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _arm_support(census: Mapping[str, Any], stats: Mapping[str, Any]) -> dict:
    by_declared = census["by_declared_date"]
    priceable_by_order = census["priceable_by_order_date"]
    fills = stats.get("buy_fills_by_date")
    out = {
        "n_declared_dates": sum(1 for r in by_declared.values() if r["requested"] > 0),
        "requested_picks": sum(r["requested"] for r in by_declared.values()),
        "generated_enters": sum(r["generated"] for r in by_declared.values()),
        "priceable_enters": sum(priceable_by_order.values()),
        "unpriced_declared_dates": sorted(
            d for d, r in by_declared.items()
            if r["requested"] > 0 and r["priceable"] < r["requested"]
        ),
    }
    if not isinstance(fills, Mapping):
        # The simulator did not report fills (never a real vectorbt run). An
        # explicit null, not a zero: nothing was observed.
        out |= {
            "fills_observed": False,
            "observed_buy_fills": None,
            "unfilled_order_dates": None,
        }
        return out
    out |= {
        "fills_observed": True,
        "observed_buy_fills": int(sum(fills.values())),
        "unfilled_order_dates": sorted(
            d for d, n in priceable_by_order.items() if int(fills.get(d, 0)) < n
        ),
    }
    return out


def _support_block(
    *,
    inputs: ReplayInputs,
    match: "_objective.CountMatch",
    censuses: Mapping[str, Mapping[str, Any]],
    stats: Mapping[str, Mapping[str, Any]],
    sampled_dates: list[str],
) -> dict:
    """The additive, versioned support/fill/sampling/overlap diagnostics.

    alpha-engine-config-I12243. Diagnostics only: nothing here feeds
    ``status``/``value``/CI, so a reader can see exactly what the count-match
    verdict and the paired CI rested on without any grade having moved.
    """
    declared_union = set()
    for census in censuses.values():
        declared_union |= {
            d for d, r in census["by_declared_date"].items() if r["requested"] > 0
        }
    base_rows = censuses["baseline"]["by_declared_date"]
    abl_rows = censuses["ablated"]["by_declared_date"]
    effective_mismatch = sorted(
        d for d in match.matched_dates
        if base_rows.get(d, {}).get("priceable", 0)
        != abl_rows.get(d, {}).get("priceable", 0)
    )
    without_entry = sorted(d for d in sampled_dates if d not in declared_union)
    overlap = _objective.window_overlap(
        sampled_dates, inputs.price_matrix.index, horizon_days=inputs.horizon_days
    )
    overlapping = (overlap["n_overlapping_adjacent_pairs"] or 0) > 0
    return {
        "schema_version": SUPPORT_SCHEMA_VERSION,
        "estimand": dict(ESTIMAND),
        "n_eligible_dates": len(inputs.dates),
        "n_sampled_windows": len(sampled_dates),
        "census": match.census(),
        "effective_width_mismatch_dates": effective_mismatch,
        "n_sampled_windows_without_entry": len(without_entry),
        "sampled_windows_without_entry": without_entry,
        "arms": {
            label: _arm_support(censuses[label], stats[label])
            for label in ("baseline", "ablated")
        },
        "window_overlap": overlap,
        "inference_assurance": {
            "ci_method": "bootstrap",
            "resampling": "iid_over_paired_windows",
            "overlap_aware": False,
            "status": "UNASSURED" if overlapping else "NO_WINDOW_OVERLAP",
            "reason": (
                f"{overlap['n_overlapping_adjacent_pairs']}/"
                f"{overlap['n_adjacent_pairs']} adjacent sampled windows share "
                f"rows of the same {overlap['horizon_days']}-row NAV path, so "
                "the iid bootstrap CI is not assured; the estimator is "
                "unchanged pending the estimator authority"
                if overlapping else
                "no two sampled windows share a row"
            ),
        },
        "not_covered": list(SUPPORT_NOT_COVERED),
    }


def run_spec(
    spec: ReplaySpec,
    inputs: ReplayInputs,
    *,
    n_trials: int,
    built: "ArmSet | NotAvailable | None" = None,
) -> dict:
    """Replay one spec and return its contract-shaped component dict.

    ``n_trials`` is the fleet-cumulative trial count AFTER this run's
    increment — DSR corrects for selection across every trial the fleet has
    run, not just this replay's own arms (spec §1).

    ``built`` lets the caller supply an already-constructed arm set so the
    trial count and the simulated arms describe the same object (see
    ``report._build_all``); omitted, the spec is built here.
    """
    if built is None:
        built = spec.build_arms(inputs)
    if isinstance(built, NotAvailable):
        return _na_component(
            spec, built, run_date=inputs.run_date, bucket=inputs.bucket
        )
    if not isinstance(built, ArmSet):
        raise TypeError(
            f"{spec.name}.build_arms returned {type(built).__name__}; "
            "expected ArmSet or NotAvailable"
        )

    component = _component_skeleton(
        spec, run_date=inputs.run_date, bucket=inputs.bucket
    )

    baseline_stats = simulate_arm(built.baseline, built, inputs)
    ablated_stats = simulate_arm(built.ablated, built, inputs)

    baseline_alpha = _objective.per_cycle_log_alpha(
        baseline_stats["daily_log_returns"],
        inputs.spy_prices,
        inputs.price_matrix.index,
        inputs.dates,
        horizon_days=inputs.horizon_days,
    )
    ablated_alpha = _objective.per_cycle_log_alpha(
        ablated_stats["daily_log_returns"],
        inputs.spy_prices,
        inputs.price_matrix.index,
        inputs.dates,
        horizon_days=inputs.horizon_days,
    )

    baseline_width = picks_per_cycle(built.baseline)
    ablated_width = picks_per_cycle(built.ablated)
    censuses = {
        "baseline": entry_census(built.baseline, built, inputs),
        "ablated": entry_census(built.ablated, built, inputs),
    }
    match = _objective.check_count_match(
        baseline_width,
        ablated_width,
        unpriceable=unpriceable_dates(list(censuses.values())),
    )

    paired = _objective.paired_diffs(baseline_alpha, ablated_alpha)
    n_samples = len(paired)
    component["support"] = _support_block(
        inputs=inputs,
        match=match,
        censuses=censuses,
        stats={"baseline": baseline_stats, "ablated": ablated_stats},
        sampled_dates=sorted(set(baseline_alpha) & set(ablated_alpha)),
    )

    component["arms"] = {
        "baseline": _arm_payload(
            built.baseline.label, baseline_stats, baseline_alpha,
            baseline_width, n_trials,
        ),
        "ablated": _arm_payload(
            built.ablated.label, ablated_stats, ablated_alpha,
            ablated_width, n_trials,
        ),
    }
    component["dsr"] = {
        "baseline": component["arms"]["baseline"]["dsr"],
        "ablated": component["arms"]["ablated"]["dsr"],
        "n_trials": n_trials,
    }
    component["n_samples"] = n_samples
    component["ci_method"] = "bootstrap"

    # PBO is reported ONLY for a genuine grid. A baseline-vs-ablated pair is a
    # single leave-one-out comparison — nothing was selected, so there is no
    # overfit-of-selection to estimate (spec §1).
    if built.sweep:
        sweep_alpha = {built.baseline.label: baseline_alpha,
                       built.ablated.label: ablated_alpha}
        for arm in built.sweep:
            stats = simulate_arm(arm, built, inputs)
            sweep_alpha[arm.label] = _objective.per_cycle_log_alpha(
                stats["daily_log_returns"],
                inputs.spy_prices,
                inputs.price_matrix.index,
                inputs.dates,
                horizon_days=inputs.horizon_days,
            )
        component["pbo"] = _objective.compute_pbo(sweep_alpha)
    else:
        component["pbo"] = None

    if not match.matched:
        # An unmatched-width comparison silently favors whichever arm trades
        # more; spec §3 binds every T5 replay to report that as a GAP rather
        # than a measured lift. No value is emitted.
        component["status"] = "gap"
        component["status_reason"] = match.reason
        component["value"] = None
        component["ci_low"] = None
        component["ci_high"] = None
        component["count_matched"] = False
        return component

    component["count_matched"] = True

    if n_samples < MIN_SAMPLES:
        component["status"] = "N/A-LOW-N"
        component["status_reason"] = (
            f"{n_samples} paired cycles with both arms defined; "
            f"below the {MIN_SAMPLES}-cycle minimum (0.5 x n_floor={N_FLOOR})"
        )
        component["value"] = None
        component["ci_low"] = None
        component["ci_high"] = None
        return component

    ci = _objective.paired_bootstrap_ci(paired)
    component["status"] = "ok"
    component["status_reason"] = (
        f"{n_samples} paired cycles, count-matched; mean paired "
        f"{UNIT} difference over the {inputs.horizon_days}d horizon"
    )
    component["value"] = ci["estimate"]
    component["ci_low"] = ci["ci_low"]
    component["ci_high"] = ci["ci_high"]
    return component
