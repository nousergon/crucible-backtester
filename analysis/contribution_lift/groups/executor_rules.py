"""executor_rules — the trades.db-derived executor null-arm replays.

Two T5 issues share this module because they share the same adapter: turning
``trades`` (and, for ``risk_guard``, ``executor_shadow_book``) rows in
``trades.db`` into the harness's ``orders``-shaped :class:`Arm` (spec §3's
*null_arm* pattern — "re-run the simulator with the rule disabled ... both
through vectorbt_bridge.orders_to_portfolio + portfolio_stats").

* **alpha-engine-config-I7481** — ``exit_rules``, ``position_sizing``,
  ``entry_triggers`` (crucible-evaluator/grading/tiles/executor.py). Each
  isolates ONE rule by holding the other two constant at their as-executed
  value and varying only the rule under test:

    - ``exit_rules``: same ENTER (date, ticker, shares) as baseline; EXIT
      replaced with a fixed hold-to-30-trading-day close instead of the
      trade's actual exit date/reason.
    - ``position_sizing``: same ENTER/EXIT dates and tickers as baseline;
      shares replaced with equal-weight ``init_cash / picks_that_day``.
    - ``entry_triggers``: same ticker, shares and EXIT date as baseline;
      ENTER date replaced with ``signal_date`` (the pre-trigger date the
      order was sourced from, ``trade_logger.py``'s
      ``ALTER TABLE trades ADD COLUMN signal_date`` — "signals/{date}/
      signals.json filename date the order was sourced from") instead of
      the trigger-selected fill ``date``. **Limitation**: the daily
      ``price_matrix`` this harness simulates against is close-only, and
      ``vectorbt_bridge.orders_to_portfolio`` prices every ENTER at the
      order's own date's close regardless of ``price_at_order``/
      ``fill_price`` — so this ablation can only express a trigger's
      effect on WHICH TRADING DAY the fill happens, not its intraday
      price-selection effect (pullback/VWAP/support entries that fill the
      SAME day as ``signal_date`` are indistinguishable from "no trigger"
      here). Cycles where the trigger caused a same-day fill therefore
      count-match trivially; cycles where it delayed the fill to a later
      day change that day's roster and are reported as a ``gap`` by the
      harness's own count-match rule (spec §3) rather than a fabricated
      lift — this is disclosed, not silently absorbed.

* **alpha-engine-config-I7482** — ``risk_guard``. Baseline is the same
  as-executed order stream; ablated adds every ``executor_shadow_book``
  row (a risk_guard block — free-text ``block_reason``, sourced from
  ``executor/deciders.py``'s guard checks, e.g. "already in portfolio",
  "no price available", "shares round to 0 (...)", stance/GBM-veto
  reasons; schema ``executor/trade_logger.py::CREATE_SHADOW_BOOK_TABLE``,
  crucible-executor origin/main) as an executed ENTER, sized from
  ``intended_shares``/``intended_dollars`` at ``current_price``, and
  exits every added name at the baseline's median ``days_held`` (a fixed
  hold, matching the issue's own framing of a single guard-off
  counterfactual). This is issue #7482's own "all flagged trades allowed
  to execute" — by construction WIDER than the baseline on any cycle with
  a block, so it does not count-match; the harness marks it ``gap`` with
  the width named, per spec §3's binding rule, rather than being
  down-selected to a narrower baseline that would understate the guard's
  true scope.

All four specs share the same ``trades`` table columns established by
``executor/trade_logger.py::CREATE_TRADES_TABLE`` (crucible-executor
origin/main, measured 2026-08-17): ``trade_id, date, ticker, action,
shares, price_at_order, fill_price, fill_time, entry_trade_id,
signal_price, signal_date, trigger_type, exit_reason, days_held,
realized_return_pct, realized_alpha_pct, spy_return_during_hold,
slippage_vs_signal``. ``trades_db_path`` absent (the live weekly config
never injects it — ``inputs.load_replay_inputs`` reads
``config.get("_trades_db")``/``config.get("trades_db")``, both unset by
default) is the expected steady state today; every spec below emits
``N/A-MISSING-INPUT`` naming that in that case.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from analysis.contribution_lift import objective as _objective
from analysis.contribution_lift.harness import (
    HORIZON_DAYS,
    MIN_SAMPLES,
    UNIT,
    ArmSet,
    NotAvailable,
    ReplayInputs,
    ReplaySpec,
    orders_arm,
    run_spec,
)

ISSUE_7481 = "alpha-engine-config-I7481"
ISSUE_7482 = "alpha-engine-config-I7482"

#: Fixed hold-to-N-trading-day exit for the exit_rules null arm (issue
#: #7481's own wording: "exits revert to a fixed hold-to-30d").
EXIT_RULES_FIXED_HOLD_DAYS = 30

_ENTRY_COLS = (
    "trade_id, date, ticker, shares, fill_price, signal_date"
)


# --------------------------------------------------------------------------
# trades.db adapter — shared by every spec in this module
# --------------------------------------------------------------------------


def _missing_db(component: str, issue: str) -> NotAvailable:
    return NotAvailable(
        status="N/A-MISSING-INPUT",
        reason=(
            f"{component}: ReplayInputs.trades_db_path is None — the live "
            "weekly config never injects trades.db (config['_trades_db'] / "
            "config['trades_db'] unset; inputs.load_replay_inputs reads "
            "either key verbatim). A trades.db snapshot would need to be "
            f"persisted and wired into the run config for {component} to be "
            f"measurable ({issue})."
        ),
    )


def _load_roundtrips(trades_db_path: str) -> pd.DataFrame:
    """One row per completed roundtrip: ``trade_id, date, ticker, shares,
    fill_price, signal_date, exit_date``.

    Reads ``trades`` exactly as ``analysis/post_trade.py`` and
    ``analysis/shadow_book.py`` already do (same table, same connection
    pattern). An ENTER row without a matching terminal ``action='EXIT'`` row
    (still open, or only ever partially ``REDUCE``d) has no defined exit and
    is dropped — a documented simplification: partial ``REDUCE`` legs are
    not modeled as separate sub-positions, the roundtrip closes at the
    terminal ``EXIT`` row. A ``trade_id`` closed more than once (not
    expected under the schema) keeps its LATEST exit row, a deterministic
    tie-break.
    """
    if not Path(trades_db_path).exists():
        return pd.DataFrame(
            columns=["trade_id", "date", "ticker", "shares", "fill_price",
                     "signal_date", "exit_date"]
        )
    conn = sqlite3.connect(trades_db_path)
    try:
        entries = pd.read_sql_query(
            f"SELECT {_ENTRY_COLS} FROM trades WHERE action = 'ENTER'", conn
        )
        exits = pd.read_sql_query(
            "SELECT entry_trade_id, date AS exit_date, days_held FROM trades "
            "WHERE action = 'EXIT' AND entry_trade_id IS NOT NULL",
            conn,
        )
    finally:
        conn.close()
    if entries.empty or exits.empty:
        return pd.DataFrame(
            columns=["trade_id", "date", "ticker", "shares", "fill_price",
                     "signal_date", "exit_date", "days_held"]
        )
    exits = exits.sort_values("exit_date").drop_duplicates(
        "entry_trade_id", keep="last"
    )
    merged = entries.merge(
        exits, left_on="trade_id", right_on="entry_trade_id", how="inner"
    )
    return merged


def _snap_forward(axis: pd.DatetimeIndex, raw_date: str) -> pd.Timestamp | None:
    """The first trading day in ``axis`` on/after ``raw_date``, or ``None``."""
    d = pd.Timestamp(raw_date)
    future = axis[axis >= d]
    return future[0] if len(future) else None


def _hold_to_n_days_exit(
    axis: pd.DatetimeIndex, entry_date: pd.Timestamp, n_days: int
) -> pd.Timestamp:
    pos = axis.get_loc(entry_date)
    exit_pos = min(pos + int(n_days), len(axis) - 1)
    return axis[exit_pos]


def _baseline_orders(roundtrips: pd.DataFrame) -> list[dict]:
    orders: list[dict] = []
    for row in roundtrips.itertuples(index=False):
        orders.append({
            "date": str(row.date),
            "ticker": str(row.ticker),
            "action": "ENTER",
            "shares": float(row.shares),
            "price_at_order": float(row.fill_price) if pd.notna(row.fill_price) else None,
        })
        orders.append({
            "date": str(row.exit_date),
            "ticker": str(row.ticker),
            "action": "EXIT",
        })
    return orders


def _picks_per_day(roundtrips: pd.DataFrame) -> dict[str, int]:
    return {
        str(d): int(g["ticker"].nunique())
        for d, g in roundtrips.groupby("date")
    }


# --------------------------------------------------------------------------
# exit_rules (I7481) — fixed hold-to-30d vs the trade's actual exit
# --------------------------------------------------------------------------


def _exit_rules_build_arms(inputs: ReplayInputs) -> ArmSet | NotAvailable:
    if not inputs.trades_db_path:
        return _missing_db("exit_rules", ISSUE_7481)
    roundtrips = _load_roundtrips(inputs.trades_db_path)
    if roundtrips.empty:
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                "exit_rules: trades.db has no ENTER row with a matching "
                f"terminal EXIT row ({ISSUE_7481})"
            ),
        )
    axis = pd.DatetimeIndex(inputs.price_matrix.index).sort_values()
    baseline = orders_arm("as-executed order stream (trades.db)", _baseline_orders(roundtrips))

    ablated_orders: list[dict] = []
    for row in roundtrips.itertuples(index=False):
        entry_snap = _snap_forward(axis, str(row.date))
        if entry_snap is None or entry_snap not in axis:
            continue
        ablated_orders.append({
            "date": str(row.date),
            "ticker": str(row.ticker),
            "action": "ENTER",
            "shares": float(row.shares),
            "price_at_order": float(row.fill_price) if pd.notna(row.fill_price) else None,
        })
        fixed_exit = _hold_to_n_days_exit(axis, entry_snap, EXIT_RULES_FIXED_HOLD_DAYS)
        ablated_orders.append({
            "date": fixed_exit.strftime("%Y-%m-%d"),
            "ticker": str(row.ticker),
            "action": "EXIT",
        })
    ablated = orders_arm(
        f"hold-to-{EXIT_RULES_FIXED_HOLD_DAYS}-trading-day fixed exit "
        "(exit rule disabled)",
        ablated_orders,
    )
    return ArmSet(baseline=baseline, ablated=ablated)


SPEC_EXIT_RULES = ReplaySpec(
    name="exit_rules",
    module="executor",
    criticality="critical",
    pattern="null_arm",
    issue=ISSUE_7481,
    build_arms=_exit_rules_build_arms,
)


# --------------------------------------------------------------------------
# exit_rules per-family decomposition (alpha-engine-config-I12624) — SHADOW
# --------------------------------------------------------------------------
#
# ``exit_rules`` above ablates EVERY terminal exit at once, so its lift
# cannot say which rule destroys the alpha. These arms ablate ONE family at a
# time: only the roundtrips whose terminal ``EXIT`` row carries a reason in
# family F are moved to a fixed hold; every other exit, every entry and every
# share count stay as executed, so each cycle count-matches by construction.
#
# They are deliberately NOT in ``registry.SPECS``: the evaluator grades every
# registered component, and these are a diagnostic of the existing
# ``exit_rules`` row, not new graded components. ``exit_family_decomposition``
# runs them and the report writes the result to its own shadow artifact,
# beside ``contribution_lift.json``, without touching it. Nothing here reads
# or writes executor config; no live exit, size or limit moves.
#
# Families come from the executed ``exit_reason`` taxonomy on terminal
# ``EXIT`` rows (crucible-executor ``executor/trade_logger.py``). Partial
# ``REDUCE`` legs (``optimizer_scale_down``, ``time_decay_reduce``,
# ``intraday_profit_take``) are not modelled by ``_load_roundtrips`` in either
# arm, so they are not ablatable here — the same simplification the
# whole-group row already makes.

ISSUE_12624 = "alpha-engine-config-I12624"

#: Terminal ``EXIT`` ``exit_reason`` -> ablation family. A reason not listed
#: falls into ``other`` rather than being dropped, so a new executor exit
#: reason shows up as its own row instead of silently joining a family.
EXIT_REASON_FAMILY: dict[str, str] = {
    "optimizer_target_zero": "optimizer",
    "time_decay_exit": "time",
    "intraday_trailing_stop": "stop",
    "atr_trailing_stop": "stop",
    "position_loss_floor": "stop",
    "intraday_collapse": "stop",
    "research_signal": "signal",
    "momentum_exit": "momentum",
}

#: Emission order of the family rows.
EXIT_FAMILIES: tuple[str, ...] = (
    "optimizer", "time", "stop", "signal", "momentum", "other",
)

#: Fixed holds each family is ablated at: 30 matches the whole-group
#: ``exit_rules`` row (so the family rows decompose THAT number), and the
#: 21-day primary horizon is the sensitivity row — a 30-day hold scored on a
#: 21-day objective mixes two horizons.
EXIT_FAMILY_HOLD_DAYS: tuple[int, ...] = (EXIT_RULES_FIXED_HOLD_DAYS, HORIZON_DAYS)


def exit_family_of(reason: object) -> str:
    """The ablation family of one terminal ``exit_reason`` (``other`` if unknown)."""
    if not isinstance(reason, str) or not reason:
        return "other"
    return EXIT_REASON_FAMILY.get(reason, "other")


def _load_terminal_exit_reasons(trades_db_path: str) -> dict[str, str]:
    """``{entry_trade_id: exit_reason}`` of each roundtrip's terminal EXIT.

    Same latest-row tie-break as :func:`_load_roundtrips`, so the reason is
    the one on the exit that roundtrip actually closes at. A database without
    an ``exit_reason`` column (an old schema, or a test fixture) returns
    ``{}`` and every roundtrip lands in ``other``.
    """
    if not Path(trades_db_path).exists():
        return {}
    conn = sqlite3.connect(trades_db_path)
    try:
        rows = pd.read_sql_query(
            "SELECT entry_trade_id, date AS exit_date, exit_reason FROM trades "
            "WHERE action = 'EXIT' AND entry_trade_id IS NOT NULL",
            conn,
        )
    except pd.errors.DatabaseError as exc:
        if "no such column" in str(exc).lower():
            return {}
        raise
    finally:
        conn.close()
    if rows.empty:
        return {}
    rows = rows.sort_values("exit_date").drop_duplicates(
        "entry_trade_id", keep="last"
    )
    return {
        str(r.entry_trade_id): (r.exit_reason if isinstance(r.exit_reason, str) else "")
        for r in rows.itertuples(index=False)
    }


def build_exit_family_arms(
    inputs: ReplayInputs,
    family: str,
    *,
    hold_days: int = EXIT_RULES_FIXED_HOLD_DAYS,
) -> ArmSet | NotAvailable:
    """Baseline vs. "family ``family`` disabled": only its exits become a hold.

    Same baseline and the same entry/share stream as ``exit_rules``; a
    roundtrip whose terminal exit is NOT in ``family`` keeps its executed exit
    date, one that is exits ``hold_days`` trading rows after entry.
    """
    if family not in EXIT_FAMILIES:
        raise ValueError(f"unknown exit family {family!r}; one of {EXIT_FAMILIES}")
    if not inputs.trades_db_path:
        return _missing_db(f"exit_rules.{family}", ISSUE_12624)
    roundtrips = _load_roundtrips(inputs.trades_db_path)
    if roundtrips.empty:
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                f"exit_rules.{family}: trades.db has no ENTER row with a "
                f"matching terminal EXIT row ({ISSUE_12624})"
            ),
        )
    reasons = _load_terminal_exit_reasons(inputs.trades_db_path)
    in_family = [
        exit_family_of(reasons.get(str(row.trade_id))) == family
        for row in roundtrips.itertuples(index=False)
    ]
    if not any(in_family):
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                f"exit_rules.{family}: no roundtrip in trades.db closes on a "
                f"{family} exit, so there is nothing to ablate ({ISSUE_12624})"
            ),
        )
    axis = pd.DatetimeIndex(inputs.price_matrix.index).sort_values()
    baseline = orders_arm(
        "as-executed order stream (trades.db)", _baseline_orders(roundtrips)
    )

    ablated_orders: list[dict] = []
    for row, ablate in zip(roundtrips.itertuples(index=False), in_family):
        entry = {
            "date": str(row.date),
            "ticker": str(row.ticker),
            "action": "ENTER",
            "shares": float(row.shares),
            "price_at_order": float(row.fill_price) if pd.notna(row.fill_price) else None,
        }
        if not ablate:
            ablated_orders.append(entry)
            ablated_orders.append(
                {"date": str(row.exit_date), "ticker": str(row.ticker), "action": "EXIT"}
            )
            continue
        entry_snap = _snap_forward(axis, str(row.date))
        if entry_snap is None or entry_snap not in axis:
            continue
        ablated_orders.append(entry)
        fixed_exit = _hold_to_n_days_exit(axis, entry_snap, hold_days)
        ablated_orders.append({
            "date": fixed_exit.strftime("%Y-%m-%d"),
            "ticker": str(row.ticker),
            "action": "EXIT",
        })
    ablated = orders_arm(
        f"{family} exits replaced by a hold-to-{hold_days}-trading-day exit "
        "(that family disabled; every other exit as executed)",
        ablated_orders,
    )
    return ArmSet(baseline=baseline, ablated=ablated)


def exit_family_spec(family: str, hold_days: int) -> ReplaySpec:
    """The shadow :class:`ReplaySpec` for one family at one fixed hold."""
    return ReplaySpec(
        name=f"exit_rules.{family}.hold{hold_days}",
        module="executor",
        criticality="supporting",
        pattern="null_arm",
        issue=ISSUE_12624,
        build_arms=lambda inputs: build_exit_family_arms(
            inputs, family, hold_days=hold_days
        ),
    )


def exit_family_census(trades_db_path: str | None) -> dict[str, int]:
    """``{family: n_roundtrips}`` closing on each family (terminal exits only)."""
    if not trades_db_path:
        return {}
    roundtrips = _load_roundtrips(trades_db_path)
    reasons = _load_terminal_exit_reasons(trades_db_path)
    out = {f: 0 for f in EXIT_FAMILIES}
    for row in roundtrips.itertuples(index=False):
        out[exit_family_of(reasons.get(str(row.trade_id)))] += 1
    return out


#: The family pre-declared as the shadow arm for alpha-engine-config-I12624:
#: the one the 2026-10-23 decomposition measured as the worst (MVO
#: ``optimizer_target_zero``). Fixed here, not re-picked each run, so the
#: shadow series is one pre-registered arm; ``worst_family`` is reported
#: beside it so a change in which family is worst is visible, not silent.
SHADOW_ARM_FAMILY = "optimizer"

#: Window cuts reported per family. ``mvo_cutover``: the 2026-05-13 MVO
#: cutover, before which no optimizer exit exists. ``destance_965``: the
#: crucible-executor #965 de-stance-ification (closed 2026-10-24) — cycles on
#: or after it measure the current rule set.
EXIT_FAMILY_WINDOW_CUTS: dict[str, str] = {
    "mvo_cutover": "2026-05-13",
    "destance_965": "2026-10-24",
}


#: The per-cycle series key ``run_spec`` writes on each arm payload.
_PER_CYCLE_KEY = f"per_cycle_{UNIT}"


def _window_summary(diffs_by_date: dict[str, float], lo: str | None, hi: str | None) -> dict:
    sel = [
        v for d, v in sorted(diffs_by_date.items())
        if (lo is None or d >= lo) and (hi is None or d < hi)
    ]
    out: dict = {"n_samples": len(sel), "from": lo, "before": hi}
    if len(sel) < MIN_SAMPLES:
        out.update(status="N/A-LOW-N", value=None, ci_low=None, ci_high=None)
        return out
    ci = _objective.paired_bootstrap_ci(sel)
    out.update(status="ok", value=ci["estimate"], ci_low=ci["ci_low"], ci_high=ci["ci_high"])
    return out


def exit_family_decomposition(inputs: ReplayInputs, *, n_trials: int) -> dict:
    """SHADOW: per-family ``exit_rules`` ablation, never a graded component.

    Returns the ``exit_rules_decomposition`` block the report writes under its
    top-level ``shadow`` key. Each row is ``run_spec`` on one
    :func:`exit_family_spec`; only its summary is kept (no arm series, no DSR —
    these arms are not candidates for promotion and are not added to the
    fleet trial accumulator).
    """
    rows: list[dict] = []
    for hold in EXIT_FAMILY_HOLD_DAYS:
        for family in EXIT_FAMILIES:
            spec = exit_family_spec(family, hold)
            comp = run_spec(spec, inputs, n_trials=n_trials)
            row = {
                "name": spec.name,
                "family": family,
                "hold_days": hold,
                "status": comp.get("status"),
                "status_reason": comp.get("status_reason"),
                "value": comp.get("value"),
                "ci_low": comp.get("ci_low"),
                "ci_high": comp.get("ci_high"),
                "n_samples": comp.get("n_samples"),
                "count_matched": comp.get("count_matched"),
            }
            arms = comp.get("arms") or {}
            if comp.get("status") == "ok" and arms:
                base = arms["baseline"][_PER_CYCLE_KEY]
                abl = arms["ablated"][_PER_CYCLE_KEY]
                diffs = {d: float(base[d] - abl[d]) for d in sorted(set(base) & set(abl))}
                cut_mvo = EXIT_FAMILY_WINDOW_CUTS["mvo_cutover"]
                cut_965 = EXIT_FAMILY_WINDOW_CUTS["destance_965"]
                row["windows"] = {
                    "pre_mvo_cutover": _window_summary(diffs, None, cut_mvo),
                    "post_mvo_pre_965": _window_summary(diffs, cut_mvo, cut_965),
                    "post_965": _window_summary(diffs, cut_965, None),
                }
                row["ablated_total_alpha"] = arms["ablated"].get("total_alpha")
                row["baseline_total_alpha"] = arms["baseline"].get("total_alpha")
            rows.append(row)

    primary = [
        r for r in rows
        if r["hold_days"] == EXIT_RULES_FIXED_HOLD_DAYS and r["status"] == "ok"
        and r["ci_high"] is not None and r["ci_high"] < 0.0
    ]
    worst = min(primary, key=lambda r: r["value"]) if primary else None
    shadow_name = f"exit_rules.{SHADOW_ARM_FAMILY}.hold{EXIT_RULES_FIXED_HOLD_DAYS}"
    return {
        "schema_version": 1,
        "issue": ISSUE_12624,
        "mode": "shadow",
        "graded": False,
        "note": (
            "Diagnostic of the exit_rules component: one family of terminal "
            "exits replaced by a fixed hold at a time, every other order as "
            "executed. Not a graded component, not in registry.SPECS, not "
            "counted in n_trials_cumulative; moves no live exit, size or limit."
        ),
        "family_of_exit_reason": dict(EXIT_REASON_FAMILY),
        "census_roundtrips": exit_family_census(inputs.trades_db_path),
        "shadow_arm": {
            "family": SHADOW_ARM_FAMILY,
            "name": shadow_name,
            "row": next((r for r in rows if r["name"] == shadow_name), None),
        },
        "worst_family": None if worst is None else {
            "family": worst["family"], "name": worst["name"], "value": worst["value"],
        },
        "families": rows,
    }



# --------------------------------------------------------------------------
# position_sizing (I7481) — equal-weight vs the trade's actual shares
# --------------------------------------------------------------------------


def _position_sizing_build_arms(inputs: ReplayInputs) -> ArmSet | NotAvailable:
    if not inputs.trades_db_path:
        return _missing_db("position_sizing", ISSUE_7481)
    roundtrips = _load_roundtrips(inputs.trades_db_path)
    if roundtrips.empty:
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                "position_sizing: trades.db has no ENTER row with a "
                f"matching terminal EXIT row ({ISSUE_7481})"
            ),
        )
    baseline = orders_arm("as-executed order stream (trades.db)", _baseline_orders(roundtrips))

    picks_per_day = _picks_per_day(roundtrips)
    price_matrix = inputs.price_matrix
    axis = pd.DatetimeIndex(price_matrix.index).sort_values()

    ablated_orders: list[dict] = []
    for row in roundtrips.itertuples(index=False):
        entry_snap = _snap_forward(axis, str(row.date))
        if entry_snap is None or row.ticker not in price_matrix.columns:
            continue
        entry_price = price_matrix.at[entry_snap, row.ticker]
        if pd.isna(entry_price) or float(entry_price) <= 0.0:
            continue
        n_slots = max(1, picks_per_day.get(str(row.date), 1))
        shares = float(int(inputs.init_cash / n_slots // float(entry_price)))
        if shares <= 0.0:
            continue
        ablated_orders.append({
            "date": str(row.date), "ticker": str(row.ticker), "action": "ENTER",
            "shares": shares, "price_at_order": float(entry_price),
        })
        ablated_orders.append({
            "date": str(row.exit_date), "ticker": str(row.ticker), "action": "EXIT",
        })
    ablated = orders_arm(
        "equal-weight sizing (init_cash / picks_that_day), same entries/exits "
        "(sizing rule disabled)",
        ablated_orders,
    )
    return ArmSet(baseline=baseline, ablated=ablated)


SPEC_POSITION_SIZING = ReplaySpec(
    name="position_sizing",
    module="executor",
    criticality="supporting",
    pattern="null_arm",
    issue=ISSUE_7481,
    build_arms=_position_sizing_build_arms,
)


# --------------------------------------------------------------------------
# entry_triggers (I7481) — signal_date fill vs the trigger-selected fill date
# --------------------------------------------------------------------------


def _entry_triggers_build_arms(inputs: ReplayInputs) -> ArmSet | NotAvailable:
    if not inputs.trades_db_path:
        return _missing_db("entry_triggers", ISSUE_7481)
    roundtrips = _load_roundtrips(inputs.trades_db_path)
    if roundtrips.empty:
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                "entry_triggers: trades.db has no ENTER row with a matching "
                f"terminal EXIT row ({ISSUE_7481})"
            ),
        )
    if "signal_date" not in roundtrips.columns or roundtrips["signal_date"].isna().all():
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                "entry_triggers: trades.signal_date is NULL on every "
                "roundtrip in this window — the pre-trigger source date is "
                f"not persisted for these rows ({ISSUE_7481})"
            ),
        )
    baseline = orders_arm("as-executed order stream (trades.db)", _baseline_orders(roundtrips))

    ablated_orders: list[dict] = []
    for row in roundtrips.itertuples(index=False):
        pre_trigger_date = row.signal_date if pd.notna(row.signal_date) else row.date
        ablated_orders.append({
            "date": str(pre_trigger_date), "ticker": str(row.ticker), "action": "ENTER",
            "shares": float(row.shares),
            "price_at_order": float(row.fill_price) if pd.notna(row.fill_price) else None,
        })
        ablated_orders.append({
            "date": str(row.exit_date), "ticker": str(row.ticker), "action": "EXIT",
        })
    ablated = orders_arm(
        "entry at signal_date (pre-trigger source date), no trigger delay; "
        "same shares and exit date as baseline",
        ablated_orders,
    )
    return ArmSet(baseline=baseline, ablated=ablated)


SPEC_ENTRY_TRIGGERS = ReplaySpec(
    name="entry_triggers",
    module="executor",
    criticality="critical",
    pattern="null_arm",
    issue=ISSUE_7481,
    build_arms=_entry_triggers_build_arms,
)


# --------------------------------------------------------------------------
# risk_guard (I7482) — as-executed vs as-executed + shadow_book-blocked
# --------------------------------------------------------------------------


def _load_shadow_book(trades_db_path: str) -> pd.DataFrame:
    """``executor_shadow_book`` rows — the risk_guard's own block log.

    Same table/columns ``analysis/shadow_book.py::compute_shadow_book_
    analysis`` reads. Schema (``executor/trade_logger.py::CREATE_SHADOW_
    BOOK_TABLE``, crucible-executor origin/main): ``block_reason`` is
    free-text, populated by every guard check in ``executor/deciders.py``
    (portfolio-membership, price-availability, share-rounding, stance/GBM
    veto reasons among them).
    """
    if not Path(trades_db_path).exists():
        return pd.DataFrame()
    conn = sqlite3.connect(trades_db_path)
    try:
        shadow = pd.read_sql_query(
            "SELECT date, ticker, block_reason, intended_shares, "
            "intended_dollars, current_price FROM executor_shadow_book",
            conn,
        )
    except pd.errors.DatabaseError as exc:
        msg = str(exc).lower()
        if "no such table" in msg or "no such column" in msg:
            return pd.DataFrame()
        raise
    finally:
        conn.close()
    return shadow


def _risk_guard_build_arms(inputs: ReplayInputs) -> ArmSet | NotAvailable:
    if not inputs.trades_db_path:
        return _missing_db("risk_guard", ISSUE_7482)
    roundtrips = _load_roundtrips(inputs.trades_db_path)
    if roundtrips.empty:
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                "risk_guard: trades.db has no ENTER row with a matching "
                f"terminal EXIT row ({ISSUE_7482})"
            ),
        )
    shadow = _load_shadow_book(inputs.trades_db_path)
    if shadow.empty:
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                "risk_guard: executor_shadow_book has no blocked-entry rows "
                f"in trades.db — nothing to allow through ({ISSUE_7482})"
            ),
        )

    baseline_orders = _baseline_orders(roundtrips)
    baseline = orders_arm("as-executed order stream (trades.db)", baseline_orders)

    median_hold = roundtrips["days_held"].dropna()
    fixed_hold = (
        int(round(float(median_hold.median())))
        if not median_hold.empty
        else HORIZON_DAYS
    )
    axis = pd.DatetimeIndex(inputs.price_matrix.index).sort_values()

    ablated_orders = list(baseline_orders)
    for row in shadow.itertuples(index=False):
        entry_snap = _snap_forward(axis, str(row.date))
        if entry_snap is None or entry_snap not in axis:
            continue
        if row.intended_shares is not None and pd.notna(row.intended_shares) and float(row.intended_shares) > 0:
            shares = float(row.intended_shares)
        elif (
            row.intended_dollars is not None and pd.notna(row.intended_dollars)
            and row.current_price is not None and pd.notna(row.current_price)
            and float(row.current_price) > 0.0
        ):
            shares = float(int(float(row.intended_dollars) / float(row.current_price)))
        else:
            continue
        if shares <= 0.0:
            continue
        ablated_orders.append({
            "date": str(row.date), "ticker": str(row.ticker), "action": "ENTER",
            "shares": shares,
            "price_at_order": float(row.current_price) if pd.notna(row.current_price) else None,
        })
        fixed_exit = _hold_to_n_days_exit(axis, entry_snap, fixed_hold)
        ablated_orders.append({
            "date": fixed_exit.strftime("%Y-%m-%d"), "ticker": str(row.ticker),
            "action": "EXIT",
        })
    ablated = orders_arm(
        "as-executed + shadow_book-blocked names allowed to execute "
        f"(risk_guard disabled), blocked names exited at baseline median "
        f"hold ({fixed_hold}d) — WIDER by construction on any blocked "
        "cycle; expect a width-mismatch gap where a block occurred",
        ablated_orders,
    )
    return ArmSet(baseline=baseline, ablated=ablated)


SPEC_RISK_GUARD = ReplaySpec(
    name="risk_guard",
    module="executor",
    criticality="critical",
    pattern="null_arm",
    issue=ISSUE_7482,
    build_arms=_risk_guard_build_arms,
)
