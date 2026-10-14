"""inputs.py — the LIVE loader for the contribution-lift harness.

The two existing replay precedents in this repo — ``analysis/factor_blend_
counterfactual_replay.py`` and ``analysis/sector_constraint_replay.py`` — are
both default-OFF and take their cycles/price matrix as INJECTED config. Nothing
in the live weekly path ever injects them, so ``evaluate.py::_run_factor_blend_
counterfactual_replay`` is a guaranteed ``skipped`` on every Saturday run and
``sector_constraint_replay`` has zero live callers. A replay that never runs
measures nothing; the report card cannot be built on one.

This loader is therefore the deliberate departure: it assembles its OWN inputs
from the live archive, so ``contribution_lift`` runs on every weekly cycle
without an operator injecting anything.

Sources (all live as of 2026-08-17):

* ``trades.db`` (``trades WHERE action='ENTER'``) — the selection actually made
* ``s3://{bucket}/signals/{date}/signals.json``     — the per-cycle scored universe
* ``s3://{bucket}/predictor/predictions/{date}.json`` — the per-name alpha hats
* ``s3://{bucket}/factors/profiles/{date}/by_ticker.json`` — sub-factor pillars
* ArcticDB (via ``store.arctic_reader.load_universe_from_arctic``) — prices + SPY

The cycle axis is the **live selection's** dates, not the signals partitions'
(config-I7501). Entry selection left the signals producer on 2026-07-13 when
``scanner_predictor_direct`` became the champion, so ``signals.json`` carries a
scored universe but no ENTER rows; the executed record in ``trades.db`` is the
only feed that spans both eras, and it reaches back to 2026-04 where ``signals/``
holds ~58 partitions. Anchoring the window to it is what makes the harness's
60-cycle ``n_floor`` reachable at all.

Explicitly NOT an input: the retired six-team + CIO research-graph tables
(``team_candidates`` / ``cio_evaluations``, retired 2026-07-12 per
``analysis/end_to_end.RESEARCH_GRAPH_RETIRED_DATE``). A replay reading a
retired producer measures a system that no longer exists.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from analysis.contribution_lift.harness import (
    HORIZON_DAYS,
    LiveSelection,
    ReplayInputs,
    build_live_selection,
    read_trades_enters,
)

logger = logging.getLogger(__name__)

#: Default cycle window. 90 signal dates comfortably clears the 60-cycle
#: n_floor even after the horizon-tail cycles are dropped.
DEFAULT_LOOKBACK_CYCLES: int = 90

_SPY = "SPY"
_CASH = "CASH"


def _cost_model(config: dict) -> tuple[float, float, float]:
    """Read the SAME cost keys ``backtest._run_simulation_loop`` reads.

    If the objective is to be net-of-cost under the production cost model,
    it has to be the production cost model — not a second set of defaults
    that drifts away from it silently.
    """
    fees = float(config.get("simulation_fees", 0.001))
    slippage_bps = float((config.get("simulation") or {}).get("slippage_bps", 0))
    init_cash = float(config.get("init_cash", 1_000_000.0))
    return fees, slippage_bps, init_cash


def _empty_matrix() -> pd.DataFrame:
    return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))


def _skipped(
    run_date: str,
    bucket: str,
    config: dict,
    reason: str,
    s3_client: Any,
    live_selection: LiveSelection | None = None,
) -> ReplayInputs:
    fees, slippage_bps, init_cash = _cost_model(config)
    return ReplayInputs(
        live_selection=live_selection,
        run_date=run_date,
        dates=[],
        signals_by_date={},
        predictions_by_date={},
        pillar_profiles_by_date={},
        price_matrix=_empty_matrix(),
        spy_prices=pd.Series(dtype=float),
        bucket=bucket,
        fees=fees,
        slippage_bps=slippage_bps,
        init_cash=init_cash,
        s3_client=s3_client,
        status="skipped",
        reason=reason,
    )


def load_replay_inputs(
    config: dict,
    *,
    run_date: str | None = None,
    lookback_cycles: int | None = None,
    s3_client: Any = None,
) -> ReplayInputs:
    """Assemble the live replay inputs shared by every T5 spec.

    Window selection: the most recent ``lookback_cycles`` **cycle dates that
    have a live selection** (config-I7501) and that still have at least
    ``HORIZON_DAYS`` price rows AFTER them, so every cycle the harness returns
    has both a book to price and a fully-defined objective. Cycles inside the
    horizon tail are excluded here rather than truncated later — a short window
    is a different measurement, not a noisier one.

    Fails LOUD when ArcticDB is unreachable or returns nothing for a non-empty
    cohort: an empty price matrix would make every arm's return 0.0 and every
    lift exactly 0.0, which is indistinguishable on the artifact from "this
    component contributes nothing". Degrades to ``status="skipped"`` ONLY when
    there is genuinely nothing to replay — no cycle carries a live selection,
    or none of them clears the horizon — and the report is still emitted
    (always-emit contract).
    """
    import boto3

    from synthetic.predictor_backtest import _extract_close, build_price_matrix
    from synthetic.production_signal_backtest import _list_date_partitions, _load_json

    bucket = config.get("signals_bucket", "alpha-engine-research")
    s3 = s3_client or boto3.client("s3")
    lookback = int(
        lookback_cycles
        if lookback_cycles is not None
        else (config.get("contribution_lift") or {}).get(
            "lookback_cycles", DEFAULT_LOOKBACK_CYCLES
        )
    )
    run_date = run_date or pd.Timestamp.today().strftime("%Y-%m-%d")

    trades_db_path = config.get("_trades_db") or config.get("trades_db")
    signal_dates = _list_date_partitions(s3, bucket, "signals/", "/")

    # The cycle axis comes from the EXECUTED record, which spans both the
    # pre-champion and the champion era; signals partitions contribute only the
    # per-cycle scored universe the substitution arms re-rank (config-I7501).
    trades_enters = read_trades_enters(trades_db_path) if trades_db_path else {}
    if not trades_enters and not signal_dates:
        return _skipped(
            run_date, bucket, config,
            (
                f"no live selection to replay: trades.db ({trades_db_path!r}) "
                f"carries no ENTER rows and there are no "
                f"signals/{{date}}/signals.json partitions under "
                f"s3://{bucket}/signals/"
            ),
            s3,
        )
    candidate_dates = sorted(set(signal_dates) | set(trades_enters))[
        -max(lookback * 2, lookback):
    ]

    pred_dates = set(
        _list_date_partitions(s3, bucket, "predictor/predictions/", ".json")
    )

    signals_by_date: dict[str, dict] = {}
    predictions_by_date: dict[str, dict] = {}
    # Every executed name is in the cohort whether or not that cycle has a
    # signals partition — under the champion the book is opened from the
    # predictor feed, and a name absent from the price matrix is unpriceable.
    cohort: set[str] = {t for names in trades_enters.values() for t in names}
    signal_date_set = set(signal_dates)
    for d in candidate_dates:
        sig = _load_json(s3, bucket, f"signals/{d}/signals.json") if d in signal_date_set else None
        if isinstance(sig, dict):
            signals_by_date[d] = sig
            signals = sig.get("signals") or {}
            rows = list(signals.values()) if isinstance(signals, dict) else list(signals)
            for row in rows:
                if isinstance(row, dict) and isinstance(row.get("ticker"), str):
                    cohort.add(row["ticker"])
        if d in pred_dates:
            prd = _load_json(s3, bucket, f"predictor/predictions/{d}.json")
            if isinstance(prd, dict):
                by_ticker = {
                    p["ticker"]: p
                    for p in (prd.get("predictions") or [])
                    if isinstance(p, dict) and isinstance(p.get("ticker"), str)
                }
                predictions_by_date[d] = by_ticker
                cohort.update(by_ticker)
    cohort -= {_SPY, _CASH}

    selection = build_live_selection(
        trades_db_path=trades_db_path,
        signals_by_date=signals_by_date,
        # None (not {}) when this run has no trades database at all, so the
        # signals fallback engages there and ONLY there.
        trades_enters=trades_enters if trades_db_path else None,
    )
    if not selection.by_date:
        return _skipped(
            run_date, bucket, config,
            (
                f"{len(candidate_dates)} candidate cycles inspected but none "
                f"carries a live selection: trades.db ({trades_db_path!r}) has "
                f"no ENTER rows and no signals.json body in the window carries "
                f"an ENTER-signalled name"
            ),
            s3,
            selection,
        )

    from store.arctic_reader import (  # type: ignore[import-not-found]
        load_universe_from_arctic,
    )

    # Fail loud: no try/except. An ArcticDB outage must surface as an errored
    # module in the completeness tracker, never as a silently empty replay.
    price_data, _features = load_universe_from_arctic(
        bucket=bucket, tickers_allowlist=cohort | {_SPY},
    )
    if not price_data:
        raise RuntimeError(
            f"ArcticDB returned no price data for {len(cohort)} contribution-lift "
            "cohort tickers — every arm would simulate to a flat 0.0 return and "
            "every measured lift would be exactly 0.0, which is indistinguishable "
            "from 'this component contributes nothing'."
        )

    all_dates: set[pd.Timestamp] = set()
    for df in price_data.values():
        all_dates.update(pd.to_datetime(df.index))
    trading_dates = [d.strftime("%Y-%m-%d") for d in sorted(all_dates)]
    price_matrix = build_price_matrix(price_data, trading_dates)

    spy_prices = _extract_close(price_data, _SPY)
    if spy_prices is None or spy_prices.dropna().empty:
        raise RuntimeError(
            "SPY close series absent from ArcticDB price_data — the objective "
            "is benchmark-relative by definition and cannot be computed without it"
        )

    axis = pd.DatetimeIndex(price_matrix.index).sort_values()
    dates = _select_window(sorted(selection.by_date), axis, lookback)
    if not dates:
        return _skipped(
            run_date, bucket, config,
            f"{len(selection.by_date)} cycles with a live selection "
            f"({selection.source}) but none has {HORIZON_DAYS} price rows "
            "after it — every cycle's objective would be undefined",
            s3,
            selection,
        )

    pillar_profiles_by_date = _load_pillar_profiles(bucket, dates)

    source_paths = [
        f"{trades_db_path or 'trades.db (absent)'} (trades WHERE action='ENTER')",
        f"s3://{bucket}/signals/{{date}}/signals.json",
        f"s3://{bucket}/predictor/predictions/{{date}}.json",
        f"s3://{bucket}/factors/profiles/{{date}}/by_ticker.json",
        # Read lazily by scanner_feed_counterfactual (alpha-engine-config-I12231).
        f"s3://{bucket}/candidates/{{run_date}}/candidates.json::scanner_eval_log",
        f"arcticdb://{bucket} (store.arctic_reader.load_universe_from_arctic)",
    ]

    fees, slippage_bps, init_cash = _cost_model(config)
    logger.info(
        "contribution_lift inputs: %d cycles (%s -> %s) from %s, cohort=%d, "
        "price_matrix=%dx%d, fees=%.4f slippage_bps=%.1f",
        len(dates), dates[0], dates[-1], selection.source, len(cohort),
        len(price_matrix), len(price_matrix.columns), fees, slippage_bps,
    )
    return ReplayInputs(
        live_selection=selection,
        run_date=run_date,
        dates=dates,
        signals_by_date={d: signals_by_date[d] for d in dates if d in signals_by_date},
        predictions_by_date={
            d: predictions_by_date[d] for d in dates if d in predictions_by_date
        },
        pillar_profiles_by_date=pillar_profiles_by_date,
        price_matrix=price_matrix,
        spy_prices=spy_prices,
        bucket=bucket,
        fees=fees,
        slippage_bps=slippage_bps,
        init_cash=init_cash,
        trades_db_path=trades_db_path,
        research_db_path=config.get("research_db"),
        s3_client=s3,
        status="ok",
        source_paths=source_paths,
    )


def _select_window(
    cycle_dates: list[str], axis: pd.DatetimeIndex, lookback: int
) -> list[str]:
    """Most recent ``lookback`` live-selection dates with a FULL horizon after."""
    usable: list[str] = []
    for d in cycle_dates:
        pos = int(axis.searchsorted(pd.Timestamp(d), side="right"))
        if pos == 0 or pos + HORIZON_DAYS > len(axis):
            continue
        usable.append(d)
    return usable[-lookback:]


def _load_pillar_profiles(bucket: str, dates: list[str]) -> dict[str, dict]:
    """``{date: {ticker: {pillar: score}}}`` from ``factors/profiles/``.

    Delegates to ``analysis.end_to_end.load_historical_pillar_profiles`` (the
    single live reader of that artifact) and re-keys its ``(date, ticker)``
    output by date. That function is fail-soft by design; a date with no
    profile is simply absent, and a spec that needs pillars for a cycle it
    does not have emits ``N/A-MISSING-INPUT`` naming the artifact.
    """
    from analysis.end_to_end import load_historical_pillar_profiles

    flat = load_historical_pillar_profiles(bucket, dates)
    by_date: dict[str, dict] = {}
    for key, pillars in (flat or {}).items():
        try:
            date_str, ticker = key
        except (TypeError, ValueError):
            continue
        by_date.setdefault(str(date_str), {})[str(ticker)] = pillars
    return by_date
