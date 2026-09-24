"""attestation.py — the backtester's RUNTIME numeric correctness verdict.

WHY THIS EXISTS
---------------
The L4593 correctness battery already proves the simulation engine computes the
right numbers: leg (a) null calibration, leg (b) the independent NumPy oracle
(``synthetic/reference_sim.py``), leg (c) golden benchmarks against published
external truth. **All three run in CI.**

The weekly backtest does not run in CI. It runs on a spot instance that
``pip install -r requirements.txt``s from scratch at boot. ``vectorbt~=0.28.5`` is
a compatible-release specifier, and ``numpy`` / ``pandas`` / ``numba`` resolve
transitively alongside it — so the engine that produces every live number can be a
different build from the one CI proved correct, and nothing in the pipeline
notices. A fill-price, fee-side or NAV-marking change in a patch release would
bias **every** backtest identically: the optimizer would still emit a
plausible-looking ``executor_params.json``, the Report Card would still render a
grade, and no artifact anywhere would say the arithmetic had moved.

`sf-pipeline-policy.md` §2.3a names this exact shape: a **data artifact** missing
makes a consumer fail visibly; a **correctness verdict** missing makes every
consumer succeed *as though the check had passed*. This module is that verdict for
the backtester — computed **where the numbers are computed**, in the deployed
interpreter, on the deployed wheels, on every cycle.

WHAT IT ASSERTS
---------------
A small battery of hand-computable scenarios driven through the **production**
path (``vectorbt_bridge.orders_to_portfolio`` + ``portfolio_stats``), compared
against literals derived **by hand** (not by running this code) plus one
cross-check against the independent oracle. Each covers a distinct accounting axis
a systematic bug rides on:

===========================  ==============================================
``pnl_no_fees``              fill price + share accounting + NAV marking
``fee_charged_both_sides``   fee side/sign (entry AND exit, on notional)
``drawdown_peak_to_trough``  running-peak drawdown definition
``alpha_over_active_window`` benchmark window alignment (the 2026-05-24 bug)
``oracle_nav_agreement``     whole NAV path vs a from-scratch accountant
``classification_counts_*``  the tp/fp/fn/tn the Report Card's precision is
                             built from (config-I6975; see below)
===========================  ==============================================

WHAT THE COUNT CHECKS ADD (config-I6975)
----------------------------------------
Everything above attests **arithmetic**. The ``classification_counts_*`` family
attests **what is being counted**, one layer up — the same distinction as the
``avg_volume_20d`` incident, where the arithmetic was right and the quantity was
not.

``crucible-evaluator grading/tiles/research.py::_precision_metric`` derives
precision, edge-over-base-rate and a Wilson CI from the ``tp``/``fp``/``fn``/``tn``
this repo aggregated into ``backtest/{date}/e2e_lift.json``. The statistical layer
on top is now attested on the evaluator side; the **counts themselves** were not.
A mislabelled selection, an off-by-one in the outcome join, or an unresolved
21-day outcome silently counted as a negative all produce internally consistent,
plausible, entirely wrong precision on Tile 1 — and every other check in this
module would still say PASS, because none of them touch the counting.

So the battery drives the PRODUCTION classification path
(``analysis/end_to_end.py``'s universe-returns admission rule + ``_scanner_lift``
→ ``_classification_for`` → ``compute_binary_metrics``) over a frozen
signal/outcome pair whose confusion matrix is derived **on paper**, and compares.
The frozen pair deliberately contains four rows that must be EXCLUDED — an
unresolved 21-day outcome, a universe row with no scanner evaluation, a scanner
evaluation with no universe row, and a row whose eval_date matches on one side
only — because "counted something it should not have" is the failure this exists
to catch, and no all-inclusive fixture can catch it.

The literals are reproduced from ``tests/test_closed_form_scenarios.py``'s
hand-computed layer, which documents the arithmetic for each. They are duplicated
here **deliberately**: the CI test proves the engine on the CI runner, this module
proves it on the box that produced the week's numbers. Neither substitutes for the
other, and the shared literals are what make a divergence between the two
diagnostic rather than confusing.

CONTRACT
--------
``run_attestation()`` **never raises**. Any failure — an import error, an engine
that crashes, an unexpected exception — resolves to ``verdict == "UNKNOWN"`` with
the error class and message recorded. This is the one legitimate broad catch in
this module: the failure mode swallowed is "the attestation itself could not run";
the primary deliverable (the weekly backtest's own artifacts) survives untouched;
and the recording surface is the emitted ``attestation.json`` body plus an ERROR
log line. A verdict-producing stage that dies must not kill the stages that do not
depend on it (§2.3a), and must equally not let the ones that do proceed unmarked —
hence ``UNKNOWN``, never a silent pass.

Consumers: ``reporter.save`` always-emits ``backtest/{run_date}/attestation.json``
and, when uploads are enabled, the standing ``backtest/latest/attestation.json``
pointer (alpha-engine-config-I8769 — see ``LATEST_ATTESTATION_KEY`` below);
``crucible-evaluator``'s Backtester tile grades the dated key as a **critical**
component and the Report Card carries the verdict at top level. See
`sf-pipeline-policy.md` §2.3a rules 1–3.
"""

from __future__ import annotations

import logging
import platform
import time
from typing import Callable, NamedTuple

logger = logging.getLogger(__name__)

SCHEMA = "backtest_attestation-1.0.0"

PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"

#: Comparison tolerances. These are *closed-form* identities, so agreement is
#: expected to near machine precision — the bands are deliberately far tighter
#: than any plausible accounting bug. A relative band of 1e-9 catches a 1bp
#: systematic bias five orders of magnitude before it would move a grade.
_RTOL = 1e-9
_ATOL = 1e-12

_INIT_CASH = 1_000_000.0


class Scenario(NamedTuple):
    """One known-answer check.

    ``expected`` is a literal derived by hand (see the module docstring); ``compute``
    drives the production path and returns the comparable observed number. They are
    kept separate — rather than ``compute`` returning a bool — so the emitted
    artifact carries both numbers and a later divergence is diagnosable from the
    artifact alone (`principles.md` §2.1).
    """

    name: str
    description: str
    expected: float
    compute: Callable[[], float]
    rtol: float = _RTOL
    atol: float = _ATOL


def verdict_is_pass(verdict: str | None) -> bool:
    """True only for an explicit PASS.

    §2.3a rule 2: ``UNKNOWN`` — and anything else, including ``None`` and a
    truthy-looking ``"ok"`` — withholds the guarantee. Consumers call this rather
    than testing truthiness so the "missing reads as pass" bug cannot be written.
    """
    return verdict == PASS


# ════════════════════════════════════════════════════════════════════════════
# Scenarios
# ════════════════════════════════════════════════════════════════════════════

def _bdays(n: int):
    import pandas as pd

    return pd.bdate_range("2024-01-01", periods=n)


def _enter(date: str, ticker: str, shares: float, price: float) -> dict:
    return {"date": date, "ticker": ticker, "action": "ENTER",
            "shares": shares, "price_at_order": price}


def _exit(date: str, ticker: str, shares: float, price: float) -> dict:
    return {"date": date, "ticker": ticker, "action": "EXIT",
            "shares": shares, "price_at_order": price}


def _pnl_no_fees() -> float:
    """Buy 10 @ 100 (day 0), sell 10 @ 130 (day 3), zero fees.

    P&L = 10 * (130 - 100) = +300 on 1,000,000 → total_return = 3e-4 exactly.
    """
    import pandas as pd

    from vectorbt_bridge import orders_to_portfolio, portfolio_stats

    idx = _bdays(5)
    prices = pd.DataFrame({"AAA": [100., 110., 120., 130., 140.]}, index=idx)
    orders = [_enter("2024-01-01", "AAA", 10, 100.),
              _exit("2024-01-04", "AAA", 10, 130.)]
    pf = orders_to_portfolio(orders, prices, init_cash=_INIT_CASH, fees=0.0)
    return float(portfolio_stats(pf, spy_expected=False)["total_return"])


def _fee_charged_both_sides() -> float:
    """Same trade at 10 bps. Entry fee = 1000 * 0.001 = 1.0; exit fee =
    1300 * 0.001 = 1.3. Net P&L = 300 - 1.0 - 1.3 = 297.7.

    A fee applied on one side only, or on shares rather than notional, moves this.
    """
    import pandas as pd

    from vectorbt_bridge import orders_to_portfolio, portfolio_stats

    idx = _bdays(5)
    prices = pd.DataFrame({"AAA": [100., 110., 120., 130., 140.]}, index=idx)
    orders = [_enter("2024-01-01", "AAA", 10, 100.),
              _exit("2024-01-04", "AAA", 10, 130.)]
    pf = orders_to_portfolio(orders, prices, init_cash=_INIT_CASH, fees=0.001)
    return float(portfolio_stats(pf, spy_expected=False)["total_return"])


def _drawdown_peak_to_trough() -> float:
    """Hold 1000 sh through 100→120→130→110→90→140.

    Peak NAV at price 130 (1,030,000); trough at price 90 (990,000).
    max_drawdown = -40,000 / 1,030,000 = -0.03883495145631068.
    """
    import pandas as pd

    from vectorbt_bridge import orders_to_portfolio, portfolio_stats

    idx = _bdays(6)
    prices = pd.DataFrame({"AAA": [100., 120., 130., 110., 90., 140.]}, index=idx)
    orders = [_enter("2024-01-01", "AAA", 1000, 100.),
              _exit("2024-01-08", "AAA", 1000, 140.)]
    pf = orders_to_portfolio(orders, prices, init_cash=_INIT_CASH, fees=0.0)
    return float(portfolio_stats(pf, spy_expected=False)["max_drawdown"])


def _alpha_over_active_window() -> float:
    """total_alpha = total_return - spy_return over the ACTIVE window.

    The portfolio is flat until day 1; SPY runs 400 → 440 (+10%) with all of the
    move outside the flat prefix. Anchoring the benchmark on the full wrapper
    instead of the active window is the exact class that produced
    ``alpha_vs_ew_high_vol: -954%`` on 2026-05-24. Expected = 3e-4 - 0.10.
    """
    import pandas as pd

    from vectorbt_bridge import orders_to_portfolio, portfolio_stats

    idx = _bdays(5)
    prices = pd.DataFrame({"AAA": [100., 110., 120., 130., 140.]}, index=idx)
    spy = pd.Series([400., 400., 400., 400., 440.], index=idx)
    orders = [_enter("2024-01-01", "AAA", 10, 100.),
              _exit("2024-01-04", "AAA", 10, 130.)]
    pf = orders_to_portfolio(orders, prices, init_cash=_INIT_CASH, fees=0.0)
    return float(portfolio_stats(pf, spy_prices=spy)["total_alpha"])


def _oracle_nav_agreement() -> float:
    """Max |NAV_production - NAV_oracle| / init_cash over a two-asset, fee-and-
    slippage-bearing scenario. Expected 0 — the oracle never imports vectorbt.

    This is the only check whose expectation is not a hand-derived literal; its
    independence comes from the second implementation rather than from arithmetic.
    """
    import numpy as np
    import pandas as pd

    from synthetic.reference_sim import simulate_reference
    from vectorbt_bridge import orders_to_portfolio

    idx = _bdays(8)
    prices = pd.DataFrame({
        "AAA": [100., 105., 110., 108., 112., 120., 118., 125.],
        "BBB": [50., 48., 52., 55., 53., 51., 60., 58.],
    }, index=idx)
    orders = [
        _enter("2024-01-02", "AAA", 500, 105.),
        _enter("2024-01-03", "BBB", 1000, 52.),
        _exit("2024-01-05", "AAA", 500, 112.),
        _exit("2024-01-09", "BBB", 1000, 60.),
    ]
    fees, slippage_bps = 0.0015, 5.0
    pf = orders_to_portfolio(
        orders, prices, init_cash=_INIT_CASH, fees=fees, slippage_bps=slippage_bps,
    )
    ref = simulate_reference(
        orders, prices, init_cash=_INIT_CASH, fees=fees, slippage_bps=slippage_bps,
    )
    prod_nav = np.asarray(pf.value().to_numpy(), dtype=np.float64)
    ref_nav = np.asarray(ref.nav, dtype=np.float64)
    if prod_nav.shape != ref_nav.shape:
        # A shape mismatch is a real divergence, not a comparison error — report
        # it as an out-of-band deviation so the check FAILs rather than raising.
        return float("inf")
    return float(np.max(np.abs(prod_nav - ref_nav)) / _INIT_CASH)


# ════════════════════════════════════════════════════════════════════════════
# Classification-count layer (config-I6975)
# ════════════════════════════════════════════════════════════════════════════

#: The frozen eval-date grid. Two RESOLVED cohort dates and one date deliberately
#: inside the canonical horizon window as of ``_FROZEN_ASOF`` — see
#: ``_assert_frozen_grid_resolves``.
_FROZEN_D0 = "2024-01-02"
_FROZEN_D1 = "2024-01-03"
_FROZEN_D_UNRESOLVED = "2024-02-20"
_FROZEN_ASOF = "2024-03-01"

#: ``(ticker, eval_date, return_5d, beat_spy_21d, beat_spy_5d)``.
#: ``beat_spy_5d`` is the exact INVERSE of ``beat_spy_21d`` on every resolved row,
#: so a wiring bug that graded the 21d block off the 5d column lands on
#: (2, 2, 3, 2) rather than the expected (2, 2, 2, 3) and FAILs. A fixture where
#: the two horizons agree could not tell them apart.
_FROZEN_UNIVERSE: tuple[tuple, ...] = (
    # ── counted ──────────────────────────────────────────────────────────────
    ("AAA", _FROZEN_D0, 0.01, 1, 0),    # selected, beat   → TP
    ("BBB", _FROZEN_D0, 0.02, 1, 0),    # selected, beat   → TP
    ("CCC", _FROZEN_D0, -0.01, 0, 1),   # selected, missed → FP
    ("DDD", _FROZEN_D0, 0.03, 1, 0),    # rejected, beat   → FN
    ("EEE", _FROZEN_D0, -0.02, 0, 1),   # rejected, missed → TN
    ("FFF", _FROZEN_D0, 0.00, 0, 1),    # rejected, missed → TN
    ("GGG", _FROZEN_D0, 0.01, 0, 1),    # rejected, missed → TN
    ("HHH", _FROZEN_D1, -0.03, 0, 1),   # selected, missed → FP
    ("III", _FROZEN_D1, 0.04, 1, 0),    # rejected, beat   → FN
    # ── must be EXCLUDED; each one is a distinct miscounting bug ─────────────
    # 21d outcome has NOT resolved yet. Counting it as "did not beat" would
    # inflate fp and depress precision on every recent cohort — the exact shape
    # the config-I6975 gotcha names.
    ("JJJ", _FROZEN_D_UNRESOLVED, 0.02, None, None),
    # in universe_returns, never scanner-evaluated: the join is INNER, so this
    # is not a missed winner. Counting it would inflate fn and the base rate.
    ("KKK", _FROZEN_D0, 0.01, 1, 0),
    # return_5d absent → rejected by the producer's own admission rule before
    # any classification happens. Counting it would inflate tp.
    ("MMM", _FROZEN_D0, None, 1, 0),
    # scanner-evaluated on a DIFFERENT date. A join on ticker alone (or an
    # off-by-one on eval_date) pairs this row with the wrong cohort → tp 3.
    ("NNN", _FROZEN_D0, 0.05, 1, 0),
)

#: ``(ticker, eval_date, quant_filter_pass)``.
_FROZEN_SCANNER: tuple[tuple, ...] = (
    ("AAA", _FROZEN_D0, 1), ("BBB", _FROZEN_D0, 1), ("CCC", _FROZEN_D0, 1),
    ("DDD", _FROZEN_D0, 0), ("EEE", _FROZEN_D0, 0), ("FFF", _FROZEN_D0, 0),
    ("GGG", _FROZEN_D0, 0),
    ("HHH", _FROZEN_D1, 1), ("III", _FROZEN_D1, 0),
    ("JJJ", _FROZEN_D_UNRESOLVED, 1),
    ("MMM", _FROZEN_D0, 1),
    ("NNN", _FROZEN_D1, 1),   # date mismatch vs NNN's universe row at D0
    ("LLL", _FROZEN_D0, 1),   # scanner-only ticker: no universe row to join to
)

#: Hand-derived from the table above, by reading it — NOT by running the
#: producer. An expectation produced by running the same code would agree with
#: any counting behaviour the code ever adopts, which is the defect this closes.
#:
#:   admitted    = 13 universe rows − MMM (no return_5d)                  = 12
#:   inner join  = 12 − KKK (no scanner row) − NNN (date mismatch)        = 10
#:   resolved    = 10 − JJJ (beat_spy_21d NULL)                           =  9
#:   selected    = AAA BBB CCC HHH                    (JJJ excluded above)
#:   positive    = AAA BBB DDD III
_EXPECTED_COUNTS: dict[str, int] = {"tp": 2, "fp": 2, "fn": 2, "tn": 3}


def _assert_frozen_grid_resolves() -> str:
    """Bind the frozen grid to ``DEFAULT_POLICY`` and RAISE if it stops holding.

    Two things must be true for the fixture to certify anything, and both are
    properties of the fleet horizon policy rather than of this file:

    1. the outcome column the producer classifies on is the one the fixture
       populates (a policy rename must not leave us attesting a dead column);
    2. the resolved eval dates really are ≥ ``primary_horizon`` trading days
       before the frozen as-of date, and the unresolved one really is inside the
       window. If a policy change to the primary horizon broke that, every row
       would fall into one bucket and the check would certify nothing while
       still reporting PASS.

    Raising here surfaces as an ERRORED check → verdict ``UNKNOWN``. Absence of
    evidence, never a pass.
    """
    import pandas as pd

    from nousergon_lib.quant.horizons import DEFAULT_POLICY

    horizon = int(DEFAULT_POLICY.primary_horizon)
    beat_col = DEFAULT_POLICY.outcome_columns(horizon).beat_spy
    if beat_col != "beat_spy_21d":
        raise ValueError(
            f"horizon policy primary outcome column is {beat_col!r}, but the frozen "
            "classification fixture populates 'beat_spy_21d' — re-derive the fixture "
            "against the new policy rather than attesting a column nobody reads."
        )

    def _elapsed(start: str) -> int:
        # Trading-day distance, exclusive of the start date.
        return max(0, len(pd.bdate_range(start, _FROZEN_ASOF)) - 1)

    for resolved in (_FROZEN_D0, _FROZEN_D1):
        if _elapsed(resolved) < horizon:
            raise ValueError(
                f"frozen grid: {resolved} is only {_elapsed(resolved)} trading days "
                f"before {_FROZEN_ASOF}, inside the {horizon}d horizon — the rows it "
                "carries could not have resolved, so the fixture certifies nothing."
            )
    if _elapsed(_FROZEN_D_UNRESOLVED) >= horizon:
        raise ValueError(
            f"frozen grid: {_FROZEN_D_UNRESOLVED} is {_elapsed(_FROZEN_D_UNRESOLVED)} "
            f"trading days before {_FROZEN_ASOF}, OUTSIDE the {horizon}d horizon — the "
            "unresolved-outcome case is no longer exercised."
        )
    return beat_col


def _frozen_research_db():
    """An in-memory research.db carrying only the two tables the path reads.

    In-memory on purpose: the attestation runs inside the live backtest process
    and must touch neither the network nor the filesystem.
    """
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE universe_returns ("
        "ticker TEXT, eval_date TEXT, return_5d REAL, "
        "beat_spy_21d INTEGER, beat_spy_5d INTEGER)"
    )
    conn.executemany(
        "INSERT INTO universe_returns VALUES (?, ?, ?, ?, ?)", _FROZEN_UNIVERSE,
    )
    conn.execute(
        "CREATE TABLE scanner_evaluations ("
        "ticker TEXT, eval_date TEXT, quant_filter_pass INTEGER)"
    )
    conn.executemany(
        "INSERT INTO scanner_evaluations VALUES (?, ?, ?)", _FROZEN_SCANNER,
    )
    conn.commit()
    return conn


_OBSERVED_COUNTS_CACHE: dict | None = None


def _observed_counts() -> dict:
    """Drive the production classification path over the frozen pair.

    Cached for the life of the process so the four count scenarios pay for one
    run, not four. The input is frozen and the path is pure, so the cache cannot
    mask a change: a redeploy is a new process.
    """
    global _OBSERVED_COUNTS_CACHE
    if _OBSERVED_COUNTS_CACHE is not None:
        return _OBSERVED_COUNTS_CACHE

    from analysis.end_to_end import _scanner_lift, load_universe_returns_frame

    _assert_frozen_grid_resolves()
    conn = _frozen_research_db()
    try:
        ur = load_universe_returns_frame(conn)
        block = _scanner_lift(conn, ur, "", [])
    finally:
        conn.close()

    clf = block.get("classification_21d")
    if not clf:
        raise ValueError(
            "the production path emitted no classification_21d block for the frozen "
            f"pair (scanner_lift status={block.get('status')!r}) — the counts the "
            "Report Card's precision is built from were not produced at all."
        )
    _OBSERVED_COUNTS_CACHE = clf
    return clf


def _count_scenarios() -> list[Scenario]:
    """One scenario per confusion-matrix cell.

    Four scenarios rather than one composite check, deliberately: the emitted
    artifact's ``expected``/``observed`` fields are scalars (see
    ``contracts/backtest_attestation.schema.json``), and splitting keeps each
    divergence diagnosable from the artifact alone — "fn 2 → 3" names the bug,
    "counts disagreed" does not. They share the ``classification_counts_``
    prefix so they read as one family on the tile and in the Director digest,
    and they flow through the existing verdict/propagation with no new wiring.
    """
    return [
        Scenario(
            name=f"classification_counts_{cell}",
            description=(
                f"scanner_lift.classification_21d.{cell} over the frozen "
                f"signal/outcome pair (hand-derived {expected})"
            ),
            expected=float(expected),
            # Bind `cell` per iteration — a late-bound closure would make all
            # four scenarios read the same key and three of them vacuous.
            compute=(lambda c=cell: float(_observed_counts()[c])),
            rtol=0.0,
            atol=0.0,   # integer counts: exact agreement or nothing
        )
        for cell, expected in _EXPECTED_COUNTS.items()
    ]


def _SCENARIOS() -> list[Scenario]:
    """The battery. A callable (not a module constant) so a test can substitute
    it, and so no engine import happens at import time of this module."""
    return [
        Scenario(
            name="pnl_no_fees",
            description="10 sh @100 → @130, no fees: total_return = 300/1e6",
            expected=300.0 / _INIT_CASH,
            compute=_pnl_no_fees,
        ),
        Scenario(
            name="fee_charged_both_sides",
            description="same trade @10bps: 300 - 1.0 - 1.3 = 297.7",
            expected=297.7 / _INIT_CASH,
            compute=_fee_charged_both_sides,
        ),
        Scenario(
            name="drawdown_peak_to_trough",
            description="peak 1,030,000 → trough 990,000: -40000/1030000",
            expected=-40_000.0 / 1_030_000.0,
            compute=_drawdown_peak_to_trough,
        ),
        Scenario(
            name="alpha_over_active_window",
            description="total_return 3e-4 minus SPY +10% on the active window",
            expected=300.0 / _INIT_CASH - 0.10,
            compute=_alpha_over_active_window,
        ),
        Scenario(
            name="oracle_nav_agreement",
            description="max NAV deviation vs synthetic.reference_sim (no vectorbt)",
            expected=0.0,
            compute=_oracle_nav_agreement,
            atol=1e-9,
        ),
        # config-I6975 — the counting layer, one level above the arithmetic.
        *_count_scenarios(),
    ]


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

def _engine_versions() -> dict:
    """Name the build that was attested. A verdict that does not identify the
    engine cannot explain a later divergence."""
    versions = {"python": platform.python_version()}
    for mod in ("vectorbt", "numpy", "pandas", "numba"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception as exc:  # noqa: BLE001 — a version probe never blocks
            versions[mod] = f"<unavailable: {type(exc).__name__}>"
    return versions


def run_attestation(
    run_date: str | None = None,
    *,
    scenario_provider: "Callable[[], list[Scenario]] | None" = None,
    component: str = "backtester",
    schema: str = SCHEMA,
) -> dict:
    """Run a known-answer battery on the deployed engine and return the verdict.

    Never raises — see the module docstring's CONTRACT section. Returns a dict
    conforming to ``schema``.

    ``scenario_provider`` / ``component`` / ``schema`` exist so a second stage in
    this repo can emit its own §2.3a verdict through **this** machinery rather
    than growing a parallel one: the outcome taxonomy (disagreed ⇒ FAIL, could
    not run ⇒ UNKNOWN) is the part that must not be reimplemented per stage —
    that is exactly where "could not measure" gets collapsed into "found a
    defect". Defaults reproduce the backtester battery byte-for-byte.
    """
    started = time.monotonic()
    checks: list[dict] = []
    provider = scenario_provider or _SCENARIOS
    try:
        scenarios = provider()
    except Exception as exc:  # noqa: BLE001 — see CONTRACT: this becomes UNKNOWN
        logger.error(
            "attestation: battery could not be constructed (%s: %s) — verdict UNKNOWN. "
            "Every consumer must withhold the correctness guarantee this cycle.",
            type(exc).__name__, exc, exc_info=True,
        )
        return {
            "schema": schema,
            "component": component,
            "run_date": run_date,
            "status": "error",
            "verdict": UNKNOWN,
            "checks": [],
            "n_checks": 0,
            "n_failed": 0,
            "engine": _engine_versions(),
            "error_class": type(exc).__name__,
            "error_msg": str(exc)[:500],
            "wall_clock_seconds": round(time.monotonic() - started, 3),
        }

    for sc in scenarios:
        record = {
            "name": sc.name,
            "description": sc.description,
            "expected": sc.expected,
            "observed": None,
            "abs_error": None,
            "rtol": sc.rtol,
            "atol": sc.atol,
            "passed": False,
        }
        try:
            observed = float(sc.compute())
            band = max(sc.atol, sc.rtol * abs(sc.expected))
            err = abs(observed - sc.expected)
            record["observed"] = observed
            record["abs_error"] = err
            record["passed"] = bool(err <= band)
            if not record["passed"]:
                logger.error(
                    "attestation check FAILED: %s expected=%r observed=%r abs_error=%r band=%r",
                    sc.name, sc.expected, observed, err, band,
                )
        except Exception as exc:  # noqa: BLE001 — a check that could not run is UNKNOWN
            record["errored"] = True
            record["error_class"] = type(exc).__name__
            record["error_msg"] = str(exc)[:500]
            logger.error(
                "attestation check ERRORED: %s (%s: %s)", sc.name, type(exc).__name__, exc,
                exc_info=True,
            )
        checks.append(record)

    # Outcome taxonomy: a check that DISAGREED is evidence the numbers are wrong
    # (FAIL); a check that could not RUN is absence of evidence (UNKNOWN). Both
    # withhold the guarantee, but only the first accuses the engine — collapsing
    # them would make an environment problem read as a correctness regression.
    n_failed = sum(1 for c in checks if not c["passed"] and not c.get("errored"))
    n_errored = sum(1 for c in checks if c.get("errored"))
    if n_failed:
        verdict = FAIL
    elif n_errored or not checks:
        verdict = UNKNOWN
    else:
        verdict = PASS
    result = {
        "schema": schema,
        "component": component,
        "run_date": run_date,
        "status": "ok",
        "verdict": verdict,
        "checks": checks,
        "n_checks": len(checks),
        "n_failed": n_failed,
        "n_errored": n_errored,
        "engine": _engine_versions(),
        "wall_clock_seconds": round(time.monotonic() - started, 3),
    }
    if verdict == UNKNOWN:
        logger.error(
            "attestation UNKNOWN — %d/%d known-answer checks could not run. The "
            "correctness guarantee is WITHHELD this cycle (never granted by default).",
            n_errored, len(checks),
        )
    elif verdict == PASS:
        logger.info(
            "attestation PASS — %d/%d known-answer checks agreed on %s (vectorbt %s, numpy %s)",
            len(checks), len(checks), result["engine"]["python"],
            result["engine"]["vectorbt"], result["engine"]["numpy"],
        )
    else:
        logger.error(
            "attestation FAIL — %d/%d known-answer checks DISAGREE with their "
            "hand-derived expectation. THIS RUN'S NUMBERS ARE NOT TRUSTWORTHY; "
            "every consumer must withhold the guarantee.",
            n_failed, len(checks),
        )
    return result


# ════════════════════════════════════════════════════════════════════════════
# Fail-closed consumer
# ════════════════════════════════════════════════════════════════════════════

def attestation_key(run_date: str) -> str:
    """S3 key of the backtest engine's verdict for ``run_date``."""
    return f"backtest/{run_date}/attestation.json"


# The standing, continuously-maintained pointer (alpha-engine-config-I8769),
# mirroring `crucible-evaluator/director/handler.py::LATEST_ACTION_PLAN_KEY` /
# `latest_action_plan_key()`'s `director/latest/action_plan.json` pattern
# (itself mirroring `grading/aggregate.py::LATEST_REPORT_CARD_KEY`'s
# `evaluator/latest/report_card.json`): `reporter.save` overwrites this key
# with the SAME body it writes to the dated `attestation_key(run_date)`, plus
# a `trading_day` field, so a fixed-key console `document-fields` binding has
# something to read every week without templating on `run_date` — a binding
# on the date-templated key renders ABSENT every day the template isn't the
# key that exists, which is a detector that cries wolf rather than one that
# measures (`nous-ergon-ops-PR614`, restated `alpha-engine-config-I7157`).
# The DATED key stays the record any date-scoped reader (this module's own
# `read_attestation`, the evaluator's Backtester tile) consumes; this
# `latest` pointer is deliberately never read back by this module, same as
# the report-card / action-plan `latest` pointers are never read by their own
# producers.
LATEST_ATTESTATION_KEY = "backtest/latest/attestation.json"


def latest_attestation_key() -> str:
    return LATEST_ATTESTATION_KEY


def read_attestation(
    bucket: str,
    run_date: str,
    key: str | None = None,
    s3_client=None,
) -> dict:
    """Read a §2.3a verdict artifact, resolving EVERY failure path to UNKNOWN.

    Returns ``{"verdict": PASS|FAIL|UNKNOWN, "reason": str, "as_of": str|None,
    "key": str, "body": dict|None}``.

    The whole point of this function is that there is no code path through it on
    which an absent, unreadable, mis-stamped or unrecognised verdict comes back
    as a pass. In particular:

    * **absent object** → UNKNOWN. "Older artifact, proceed" is the precise
      blindness §2.3a exists to remove, so absence is a first-class not-verified
      state, never a skip.
    * **unparseable body** → UNKNOWN.
    * **body stamped with a different ``run_date``** → UNKNOWN. A verdict from
      another cycle is never inherited (§2.3a rule 1) — last week's PASS says
      nothing about this week's numbers, and S3 will happily serve it.
    * **any verdict string other than the literal ``"PASS"``** → passed through
      if it is a recognised state, UNKNOWN otherwise. A producer that starts
      writing ``"ok"`` degrades loudly instead of silently granting the
      guarantee.

    Never raises: a consumer that crashes reading the verdict would take down the
    stage the verdict was meant to merely *qualify*. The swallowed failure mode is
    "the verdict could not be read"; the primary deliverable (the evaluation) is
    untouched; the recording surface is the returned ``reason`` — which is
    written into this stage's own emitted artifact — plus an ERROR log line.
    """
    resolved_key = key or attestation_key(run_date)
    out = {"verdict": UNKNOWN, "reason": "", "as_of": None,
           "key": resolved_key, "body": None}
    try:
        import json

        import boto3

        client = s3_client or boto3.client("s3")
        obj = client.get_object(Bucket=bucket, Key=resolved_key)
        raw = obj["Body"].read()
        last_modified = obj.get("LastModified")
        out["as_of"] = (
            last_modified.strftime("%Y-%m-%dT%H:%M:%SZ")
            if last_modified is not None else None
        )
        body = json.loads(raw)
    except Exception as exc:  # noqa: BLE001 — see docstring: every path is UNKNOWN
        out["reason"] = (
            f"could not read s3://{bucket}/{resolved_key} "
            f"({type(exc).__name__}: {str(exc)[:200]})"
        )
        logger.error(
            "attestation consumer: %s — verdict UNKNOWN, correctness guarantee WITHHELD.",
            out["reason"],
        )
        return out

    if not isinstance(body, dict):
        out["reason"] = f"{resolved_key} is not a JSON object"
        logger.error("attestation consumer: %s — verdict UNKNOWN.", out["reason"])
        return out

    out["body"] = body
    stamped = body.get("run_date")
    if stamped != run_date:
        out["reason"] = (
            f"{resolved_key} is stamped run_date={stamped!r}, expected {run_date!r} — "
            "a verdict from another cycle is never inherited"
        )
        logger.error("attestation consumer: %s — verdict UNKNOWN.", out["reason"])
        return out

    raw_verdict = body.get("verdict")
    if raw_verdict not in (PASS, FAIL, UNKNOWN):
        out["reason"] = (
            f"{resolved_key} carries unrecognised verdict {raw_verdict!r} — "
            f"only the literal {PASS!r} is a pass"
        )
        logger.error("attestation consumer: %s — verdict UNKNOWN.", out["reason"])
        return out

    out["verdict"] = raw_verdict
    out["reason"] = (
        "verdict read" if raw_verdict == PASS
        else f"producer reported {raw_verdict}"
    )
    return out


def worst(*verdicts: str | None) -> str:
    """Combine verdicts: any FAIL ⇒ FAIL, else any non-PASS ⇒ UNKNOWN, else PASS.

    ``None`` and unrecognised values are non-PASS, so a combine can never be
    talked into a pass by an absent input.
    """
    values = list(verdicts)
    if not values:
        return UNKNOWN
    if any(v == FAIL for v in values):
        return FAIL
    if all(v == PASS for v in values):
        return PASS
    return UNKNOWN
