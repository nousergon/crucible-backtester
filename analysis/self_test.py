"""self_test.py — the backtester's published known-answer SELF-TEST.

WHY THIS EXISTS
---------------
Brian, 2026-08-13: *"is there a way we can input certain params into backtester
and evaluator to confirm they are even computing correctly and include outputs of
this test?"*

CI proves the code is correct **on a runner**. It does not prove the **deployed
instrument** is: the weekly backtest runs on a spot instance that
``pip install -r requirements.txt``s from scratch at boot, so ``vectorbt``,
``numpy``, ``pandas``, ``numba`` and ``nousergon_lib`` resolve to whatever the
index serves that morning. A changed ``ddof``, annualization factor or
downside-deviation denominator moves **every** risk metric at once, coherently
and plausibly, and CI stays green throughout.

So this battery runs **in the pipeline, in the process that produces the week's
numbers**, drives the **production** path (``vectorbt_bridge.orders_to_portfolio``
+ ``portfolio_stats``), and publishes its inputs, expectations, observations and
per-case verdicts as an artifact a human can read: ``backtest/{run_date}/self_test.json``.
**The library versions in the header are the point** — they are what make this an
instrument check rather than a code check.

RELATIONSHIP TO ``analysis/attestation.py``
-------------------------------------------
``attestation.py`` is the machine-facing §2.3a *verdict*: a terse PASS/FAIL/UNKNOWN
the evaluator's Backtester tile and the Director consume to decide whether to act
on the week's numbers. This module is the human-facing *evidence* behind that
question, and it deliberately covers three axes ``attestation.py`` does not:

1. **Metamorphic relations** (fee additivity, currency-scale invariance, size
   linearity). These catch bugs where the correct ANSWER is unknown but the
   correct RELATIONSHIP is known — a class that is structurally invisible to a
   closed-form test, because a closed-form test can only check inputs whose
   answer someone already wrote down.
2. **Undefined-vs-measured-zero.** A flat market has no Sharpe ratio. A metric
   that reports ``0.0`` there is indistinguishable from a measured zero, which is
   `principles.md` §2.7 at the metric layer.
3. **Published provenance.** Every case carries its ``inputs`` and the header
   carries the ``importlib.metadata`` version of every quant library actually
   loaded at runtime plus the code SHA — so a divergence six weeks from now is
   diagnosable from the two artifacts alone, with nobody's memory involved.

Neither module substitutes for the other, and both share the fleet verdict
vocabulary (``PASS``/``FAIL``/``UNKNOWN``) so a reader never has to translate.

CONVENTIONS THIS BATTERY PINS (read before changing an expectation)
-------------------------------------------------------------------
``closed_form_sharpe`` pins the backtester's Sharpe annualization convention.

**2026-08-13, Brian ruling on config-I7236 ("use sota"): FIXED to √252.**
Until this PR, ``vbt.Portfolio.sharpe_ratio()`` (called via ``freq="D"``, which
vectorbt reads as calendar days) annualized the headline Sharpe by **√365**,
while ``_compute_sortino_ratio`` in the same function, and
``nousergon_lib.quant.riskstats.sharpe_ratio`` on the evaluator side, both
annualized by **√252** — a ~20.3% systematic gap between two Sharpe numbers on
the same Report Card. ``portfolio_stats`` in ``vectorbt_bridge.py`` now calls
``nousergon_lib.quant.riskstats.sharpe_ratio`` directly on the same
``daily_returns`` series ``_compute_sortino_ratio`` uses, instead of
``pf.sharpe_ratio()`` — the shared library function, not a copied constant, so
the two conventions cannot silently diverge again (agreement-tested in
``tests/test_vectorbt_bridge.py`` to ~1e-9). It still computes over the FULL
return series including the leading ``0.0`` vectorbt emits for day 0 (no prior
NAV, matching ``daily_returns``' own convention elsewhere in ``portfolio_stats``)
— that dilution question was explicitly deferred by the issue and is unchanged
by this fix.

**Every historical backtester Sharpe is now ~17% lower than previously
reported** (a multiplicative ×1/√(365/252) ≈ ×0.8305 rescale) — this is the
expected, intended effect of the fix, not a regression. Numbers from before
2026-08-13 will not match numbers after it for that reason alone.

CONTRACT
--------
``run_self_test()`` **never raises**, and the caller writes its output
unconditionally. A case that DISAGREED with its expectation is ``FAIL`` (evidence
the numbers are wrong); a case that could not RUN is ``UNKNOWN`` (absence of
evidence). Collapsing the two would make a broken venv read as a correctness
regression. Per Brian's ruling 2026-08-13, **a case that exceeds its time budget
is FAIL, never UNKNOWN** — a self-test that cannot finish in seconds is itself a
defect, and "the check timed out" must not buy the same benefit of the doubt as
"the check could not be constructed".

This module never sets a hard-fail path in the pipeline. It writes its artifact
and, on a non-PASS verdict, the caller raises the run's existing degraded flag.
Withholding a guarantee beats failing the run (`sf-pipeline-policy.md` §2.3a).

LIFTED RUNNER (alpha-engine-config-I7238)
------------------------------------------
The outcome taxonomy, ``Case`` shape, SIGALRM budget and provenance helpers
below are imported from ``nousergon_lib.quant.selftest`` — the second/third/
fourth adoption of this scaffolding (evaluator, predictor, research) triggered
the lift per ``shared-code-policy``. This module keeps only what is genuinely
domain-specific: the scenario fixtures, ``build_cases()``, and the artifact key.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable

from nousergon_lib.quant.selftest import (
    CASE_TIMEOUT_SECONDS,
    Case,
    code_sha as _lib_code_sha,
    resolved_library_versions as _lib_resolved_library_versions,
    run_self_test as _lib_run_self_test,
)
from nousergon_lib.quant.selftest import FAIL as FAIL  # noqa: PLC0414 — re-exported (st.FAIL)
from nousergon_lib.quant.selftest import PASS as PASS  # noqa: PLC0414 — re-exported (st.PASS)
from nousergon_lib.quant.selftest import UNKNOWN as UNKNOWN  # noqa: PLC0414 — re-exported (st.UNKNOWN)
from nousergon_lib.quant.selftest import (
    verdict_is_pass as verdict_is_pass,  # noqa: PLC0414 — re-exported (st.verdict_is_pass)
)

logger = logging.getLogger(__name__)

SCHEMA = "backtest_self_test-1.0.0"
COMPONENT = "backtester"

#: S3 prefix-relative artifact name. The full key is ``backtest/{run_date}/self_test.json``.
ARTIFACT_FILENAME = "self_test.json"

#: Quant libraries whose resolved version decides what every number in this
#: repo means. Recorded via ``importlib.metadata`` (the DISTRIBUTION version
#: actually installed) rather than ``module.__version__`` (an attribute a
#: package may lie about, lack, or inherit from a vendored copy).
_TRACKED_DISTRIBUTIONS = (
    "vectorbt",
    "numpy",
    "pandas",
    "numba",
    "scipy",
    "nousergon-lib",
    "krepis",
)

_INIT_CASH = 1_000_000.0


# ════════════════════════════════════════════════════════════════════════════
# Scenario helpers — all deterministic, in-memory, no network, no market data
# ════════════════════════════════════════════════════════════════════════════

def _bdays(n: int):
    import pandas as pd

    return pd.bdate_range("2024-01-01", periods=n)


def _order(action: str, date: str, ticker: str, shares: float, price: float) -> dict:
    return {"date": date, "ticker": ticker, "action": action,
            "shares": shares, "price_at_order": price}


def _run(prices, orders, *, init_cash: float = _INIT_CASH, fees: float = 0.0,
         spy_prices=None) -> dict:
    """Drive the PRODUCTION path and return its stats dict."""
    from vectorbt_bridge import orders_to_portfolio, portfolio_stats

    pf = orders_to_portfolio(orders, prices, init_cash=init_cash, fees=fees)
    # A case with no SPY is a no-benchmark case by construction.
    return portfolio_stats(pf, spy_prices=spy_prices,
                           spy_expected=spy_prices is not None)


# ── case 1: single round trip ───────────────────────────────────────────────
#
# Buy 100 @ $10.00, sell @ $11.00, $1.00 of total fees.
#   gross P&L = 100 * (11.00 - 10.00) = $100.00
#   fees      = fee_rate * (100*10.00 + 100*11.00) = fee_rate * 2100
#   fee_rate  = 1/2100 makes total fees exactly $1.00
#   net P&L   = 100.00 - 1.00 = $99.00
#
# ``fees`` is a rate on notional in the production signature, so a flat $1.00 is
# expressed as the rate that produces it. Stating it that way keeps the number
# Brian can check ($99.00) while driving the real parameter the engine takes.
_RT_SHARES = 100.0
_RT_BUY = 10.0
_RT_SELL = 11.0
_RT_NOTIONAL = _RT_SHARES * (_RT_BUY + _RT_SELL)   # 2100.0
_RT_FEE_DOLLARS = 1.00
_RT_FEE_RATE = _RT_FEE_DOLLARS / _RT_NOTIONAL


def _single_round_trip_pnl() -> float:
    import pandas as pd

    idx = _bdays(4)
    prices = pd.DataFrame({"AAA": [10.0, 10.5, 11.0, 11.0]}, index=idx)
    orders = [_order("ENTER", "2024-01-01", "AAA", _RT_SHARES, _RT_BUY),
              _order("EXIT", "2024-01-03", "AAA", _RT_SHARES, _RT_SELL)]
    stats = _run(prices, orders, fees=_RT_FEE_RATE)
    # Report in DOLLARS: the number a human checks is $99.00, not 9.9e-05.
    return float(stats["total_return"]) * _INIT_CASH


# ── case 2: flat market ─────────────────────────────────────────────────────
#
# Every price constant, one full round trip at the same price, zero fees.
#   total_return  = 0.0 exactly (a MEASURED zero — the correct answer)
#   max_drawdown  = 0.0 exactly (NAV never falls below its running peak)
#   sharpe_ratio  = UNDEFINED (zero mean over zero volatility) — and must not be
#                   reported as a finite number, or it is indistinguishable from
#                   a measured zero.
_FLAT_PRICE = 50.0
_FLAT_SHARES = 100.0
_FLAT_DAYS = 6


def _flat_market_stats() -> dict:
    import pandas as pd

    idx = _bdays(_FLAT_DAYS)
    prices = pd.DataFrame({"AAA": [_FLAT_PRICE] * _FLAT_DAYS}, index=idx)
    orders = [_order("ENTER", "2024-01-01", "AAA", _FLAT_SHARES, _FLAT_PRICE),
              _order("EXIT", "2024-01-04", "AAA", _FLAT_SHARES, _FLAT_PRICE)]
    return _run(prices, orders, fees=0.0)


def _flat_market_total_return() -> float:
    return float(_flat_market_stats()["total_return"])


def _flat_market_max_drawdown() -> float:
    return float(_flat_market_stats()["max_drawdown"])


def _flat_market_sharpe_is_undefined() -> float:
    """1.0 iff a flat market's Sharpe is NOT reported as a finite number.

    ``None``, ``NaN`` and ``+/-inf`` all satisfy the contract Brian named: an
    undefined value must be distinguishable from a measured zero. Which of those
    three the engine emits is a separate (open) question — this case pins only
    that it is not a finite number, so it certifies the property without
    certifying the representation.
    """
    sharpe = _flat_market_stats()["sharpe_ratio"]
    if sharpe is None:
        return 1.0
    return 0.0 if math.isfinite(float(sharpe)) else 1.0


# ── case 3: fee monotonicity (metamorphic) ──────────────────────────────────
#
# The same scenario at fee 0 and at 10 bps. Return must fall by EXACTLY the fee
# charged on both legs' notional and by nothing else:
#   notional = 10*100 + 10*130 = 2300; fee = 2300 * 0.001 = $2.30
#   expected residual (delta_return - 2.30/init_cash) = 0.0
#
# Catches a fee applied on one side only, on shares rather than notional, or at
# the wrong sign — none of which any single-scenario closed form can see.
_FEE_SHARES = 10.0
_FEE_BUY = 100.0
_FEE_SELL = 130.0
_FEE_RATE = 0.001
_FEE_NOTIONAL = _FEE_SHARES * (_FEE_BUY + _FEE_SELL)   # 2300.0


def _fee_scenario(*, fees: float, shares: float = _FEE_SHARES,
                  scale: float = 1.0, init_cash: float = _INIT_CASH) -> float:
    import pandas as pd

    idx = _bdays(5)
    path = [100.0, 110.0, 120.0, 130.0, 140.0]
    prices = pd.DataFrame({"AAA": [p * scale for p in path]}, index=idx)
    orders = [_order("ENTER", "2024-01-01", "AAA", shares, _FEE_BUY * scale),
              _order("EXIT", "2024-01-04", "AAA", shares, _FEE_SELL * scale)]
    return float(_run(prices, orders, init_cash=init_cash, fees=fees)["total_return"])


def _fee_charged_once_per_side() -> float:
    delta = _fee_scenario(fees=0.0) - _fee_scenario(fees=_FEE_RATE)
    return delta - (_FEE_NOTIONAL * _FEE_RATE / _INIT_CASH)


# ── case 4: currency-scale invariance (metamorphic) ─────────────────────────
#
# The simulation is homogeneous of degree ZERO in currency units: multiply every
# price AND the starting cash by 3 and the RETURN is unchanged. (Scaling prices
# alone is not the invariance — it triples the dollar exposure of a fixed share
# count, so it triples the return. The scale-free statement is the one that
# holds, and it is the one an accounting bug breaks.)
_SCALE_FACTOR = 3.0


def _currency_scale_invariance() -> float:
    base = _fee_scenario(fees=_FEE_RATE)
    scaled = _fee_scenario(fees=_FEE_RATE, scale=_SCALE_FACTOR,
                           init_cash=_INIT_CASH * _SCALE_FACTOR)
    return scaled - base


# ── case 5: size linearity (metamorphic) ────────────────────────────────────
#
# Doubling the share count on both legs doubles the P&L exactly — fees included,
# since a rate on notional is itself linear in size. Catches a fee or fill-price
# term that is constant in size, or a rounding rule applied per-order.
def _size_linearity() -> float:
    single = _fee_scenario(fees=_FEE_RATE, shares=_FEE_SHARES)
    double = _fee_scenario(fees=_FEE_RATE, shares=_FEE_SHARES * 2.0)
    return double - 2.0 * single


# ── case 6: closed-form Sharpe ──────────────────────────────────────────────
#
# A fully-invested portfolio on a deterministic price path whose daily returns
# alternate +2% / -1% for 20 days. The portfolio holds init_cash/P0 shares from
# day 0 with zero fees, so NAV_t = shares * P_t and the portfolio's return series
# IS the price return series.
#
# The observation vector `daily_returns` computes over is [0.0] + the 20
# returns — day 0 has no prior NAV, and vectorbt emits 0.0 for it rather than
# NaN, so it survives the `dropna()` in portfolio_stats() and is included.
# Over those n = 21 observations:
#   mean = (10*0.02 + 10*(-0.01)) / 21 = 0.1/21
#   var  = sum((r - mean)^2) / (n - 1)          [ddof = 1]
#   sharpe = mean / sqrt(var) * sqrt(252)       [config-I7236: trading days]
#
# The leading-zero inclusion is a measured property of the deployed engine,
# not an endorsement (see this module's docstring) — √252 is the fixed,
# institutional-standard annualization, matching Sortino beside it and
# nousergon_lib.quant.riskstats.
_SHARPE_UP = 0.02
_SHARPE_DOWN = -0.01
_SHARPE_CYCLES = 10
_SHARPE_ANNUALIZATION_DAYS = 252


def _sharpe_return_path() -> list[float]:
    return [_SHARPE_UP, _SHARPE_DOWN] * _SHARPE_CYCLES


def _expected_closed_form_sharpe() -> float:
    """The analytic Sharpe, computed here from the definition above.

    Deliberately written out in plain arithmetic rather than obtained from any
    library: an expectation produced by the code under test agrees with whatever
    that code ever does, which is the defect this whole module exists to close.
    """
    observations = [0.0] + _sharpe_return_path()
    n = len(observations)
    mean = sum(observations) / n
    variance = sum((r - mean) ** 2 for r in observations) / (n - 1)
    return mean / math.sqrt(variance) * math.sqrt(_SHARPE_ANNUALIZATION_DAYS)


def _closed_form_sharpe() -> float:
    import pandas as pd

    returns = _sharpe_return_path()
    path = [100.0]
    for r in returns:
        path.append(path[-1] * (1.0 + r))
    idx = _bdays(len(path))
    prices = pd.DataFrame({"AAA": path}, index=idx)
    orders = [_order("ENTER", str(idx[0].date()), "AAA", _INIT_CASH / 100.0, 100.0)]
    return float(_run(prices, orders, fees=0.0)["sharpe_ratio"])


# ── case 7: null control ────────────────────────────────────────────────────
#
# A seeded zero-drift random walk for the holding and an INDEPENDENT one for the
# benchmark. There is no true alpha, so a correctly-built confidence interval on
# the daily active return must COVER ZERO. A CI that excludes zero on a null
# input is manufacturing significance — the failure mode no closed-form check
# looks for, because it is a property of the estimator rather than of a number.
#
# Deterministic twice over: the price paths come from a seeded Generator, and
# ``bootstrap_ci`` itself takes a fixed seed.
_NULL_SEED = 20260813
_NULL_DAYS = 120
_NULL_SIGMA = 0.01


def _null_control_ci_covers_zero() -> float:
    import numpy as np
    import pandas as pd
    from nousergon_lib.quant.stats.intervals import bootstrap_ci

    rng = np.random.default_rng(_NULL_SEED)
    asset_r = rng.normal(0.0, _NULL_SIGMA, _NULL_DAYS)
    bench_r = rng.normal(0.0, _NULL_SIGMA, _NULL_DAYS)

    def _to_path(returns) -> list[float]:
        path = [100.0]
        for r in returns:
            path.append(path[-1] * (1.0 + float(r)))
        return path

    asset_path = _to_path(asset_r)
    bench_path = _to_path(bench_r)
    idx = _bdays(len(asset_path))
    prices = pd.DataFrame({"AAA": asset_path}, index=idx)
    spy = pd.Series(bench_path, index=idx)
    orders = [_order("ENTER", str(idx[0].date()), "AAA", _INIT_CASH / 100.0, 100.0)]
    stats = _run(prices, orders, fees=0.0, spy_prices=spy)

    portfolio_daily = np.asarray(stats["daily_returns"].to_numpy(), dtype=float)
    bench_daily = np.asarray(spy.pct_change().to_numpy(), dtype=float)
    n = min(portfolio_daily.size, bench_daily.size)
    active = portfolio_daily[-n:] - bench_daily[-n:]
    active = active[np.isfinite(active)]

    result = bootstrap_ci(active)
    if result.get("status") != "ok":
        raise AssertionError(f"bootstrap_ci did not run on the null sample: {result!r}")
    covers = float(result["ci_low"]) <= 0.0 <= float(result["ci_high"])
    if not covers:
        logger.error(
            "null control: the 95%% CI on a ZERO-alpha sample is [%r, %r] and does "
            "not cover zero — the interval estimator is manufacturing significance.",
            result["ci_low"], result["ci_high"],
        )
    return 1.0 if covers else 0.0


# ════════════════════════════════════════════════════════════════════════════
# The battery
# ════════════════════════════════════════════════════════════════════════════

def build_cases() -> list[Case]:
    """The battery. A callable (not a module constant) so no engine import
    happens at import time of this module, and so a test can substitute it."""
    return [
        Case(
            name="single_round_trip",
            description=(
                f"Buy {_RT_SHARES:.0f} @ ${_RT_BUY:.2f}, sell @ ${_RT_SELL:.2f}, "
                f"${_RT_FEE_DOLLARS:.2f} of total fees: "
                f"P&L = {_RT_SHARES:.0f}*({_RT_SELL:.2f}-{_RT_BUY:.2f}) - "
                f"{_RT_FEE_DOLLARS:.2f} = $99.00"
            ),
            inputs={"shares": _RT_SHARES, "buy_price": _RT_BUY, "sell_price": _RT_SELL,
                    "fee_rate": _RT_FEE_RATE, "total_fees_usd": _RT_FEE_DOLLARS,
                    "init_cash": _INIT_CASH, "units": "USD"},
            expected=_RT_SHARES * (_RT_SELL - _RT_BUY) - _RT_FEE_DOLLARS,
            compute=_single_round_trip_pnl,
            tolerance=0.005,
        ),
        Case(
            name="flat_market_total_return",
            description="Every price constant, one round trip, zero fees: total_return = 0.0",
            inputs={"price": _FLAT_PRICE, "days": _FLAT_DAYS, "shares": _FLAT_SHARES,
                    "fee_rate": 0.0, "units": "fraction"},
            expected=0.0,
            compute=_flat_market_total_return,
            tolerance=1e-12,
        ),
        Case(
            name="flat_market_max_drawdown",
            description="Flat NAV never falls below its running peak: max_drawdown = 0.0",
            inputs={"price": _FLAT_PRICE, "days": _FLAT_DAYS, "shares": _FLAT_SHARES,
                    "fee_rate": 0.0, "units": "fraction"},
            expected=0.0,
            compute=_flat_market_max_drawdown,
            tolerance=1e-12,
        ),
        Case(
            name="flat_market_sharpe_is_undefined",
            description=(
                "A flat market has NO Sharpe ratio (zero mean over zero volatility). "
                "1.0 iff the engine reports a non-finite value (None/NaN/inf) rather "
                "than a finite number — a measured zero and an undefined value must "
                "be distinguishable"
            ),
            inputs={"price": _FLAT_PRICE, "days": _FLAT_DAYS, "shares": _FLAT_SHARES,
                    "fee_rate": 0.0, "units": "boolean-encoded contract"},
            expected=1.0,
            compute=_flat_market_sharpe_is_undefined,
            tolerance=0.0,
        ),
        Case(
            name="fee_charged_once_per_side",
            description=(
                f"METAMORPHIC. Same scenario at fee 0 vs {_FEE_RATE*1e4:.0f}bps: the "
                f"return falls by exactly fee_rate * notional "
                f"(${_FEE_NOTIONAL:.2f} * {_FEE_RATE} = "
                f"${_FEE_NOTIONAL*_FEE_RATE:.2f}) and by nothing else. "
                "Residual expected 0.0"
            ),
            inputs={"shares": _FEE_SHARES, "buy_price": _FEE_BUY, "sell_price": _FEE_SELL,
                    "fee_rate_a": 0.0, "fee_rate_b": _FEE_RATE,
                    "notional": _FEE_NOTIONAL, "init_cash": _INIT_CASH,
                    "units": "fraction (residual)"},
            expected=0.0,
            compute=_fee_charged_once_per_side,
            tolerance=1e-12,
        ),
        Case(
            name="currency_scale_invariance",
            description=(
                f"METAMORPHIC. Multiply every price AND the starting cash by "
                f"{_SCALE_FACTOR:.0f}: the simulation is homogeneous of degree zero in "
                "currency units, so total_return is unchanged. Difference expected 0.0"
            ),
            inputs={"scale_factor": _SCALE_FACTOR, "shares": _FEE_SHARES,
                    "fee_rate": _FEE_RATE, "init_cash_base": _INIT_CASH,
                    "init_cash_scaled": _INIT_CASH * _SCALE_FACTOR,
                    "units": "fraction (difference)"},
            expected=0.0,
            compute=_currency_scale_invariance,
            tolerance=1e-12,
        ),
        Case(
            name="size_linearity",
            description=(
                "METAMORPHIC. Doubling the share count on both legs doubles the P&L "
                "exactly, fees included (a rate on notional is linear in size). "
                "Residual expected 0.0"
            ),
            inputs={"shares_a": _FEE_SHARES, "shares_b": _FEE_SHARES * 2.0,
                    "fee_rate": _FEE_RATE, "init_cash": _INIT_CASH,
                    "units": "fraction (residual)"},
            expected=0.0,
            compute=_size_linearity,
            tolerance=1e-12,
        ),
        Case(
            name="closed_form_sharpe",
            description=(
                f"Fully-invested portfolio on a deterministic path of "
                f"{_SHARPE_CYCLES} x (+{_SHARPE_UP:.0%}, {_SHARPE_DOWN:.0%}) daily "
                f"returns. Analytic Sharpe = mean/stdev(ddof=1) * sqrt("
                f"{_SHARPE_ANNUALIZATION_DAYS}) over the 21-observation series "
                "portfolio_stats computes on (a leading 0.0 for day 0, which has "
                "no prior NAV). PINS the FIXED sqrt(252) convention (config-I7236, "
                "2026-08-13): sharpe_ratio, sortino_ratio beside it, and the "
                "evaluator's nousergon_lib sharpe all now annualize by sqrt(252)"
            ),
            inputs={"daily_return_up": _SHARPE_UP, "daily_return_down": _SHARPE_DOWN,
                    "cycles": _SHARPE_CYCLES, "n_observations": 2 * _SHARPE_CYCLES + 1,
                    "leading_zero_observation": True, "ddof": 1,
                    "annualization_days": _SHARPE_ANNUALIZATION_DAYS,
                    "init_cash": _INIT_CASH, "fee_rate": 0.0, "units": "ratio"},
            expected=_expected_closed_form_sharpe(),
            compute=_closed_form_sharpe,
            tolerance=1e-9,
        ),
        Case(
            name="null_control_alpha_ci_covers_zero",
            description=(
                f"NULL CONTROL. {_NULL_DAYS} days of seeded zero-drift random walk for "
                "the holding and an independent one for the benchmark: there is no "
                "true alpha, so the 95% bootstrap CI on the daily active return must "
                "COVER ZERO. 1.0 iff it does"
            ),
            inputs={"seed": _NULL_SEED, "days": _NULL_DAYS, "sigma": _NULL_SIGMA,
                    "drift": 0.0, "ci_level": 0.95, "bootstrap_seed": 0,
                    "units": "boolean-encoded contract"},
            expected=1.0,
            compute=_null_control_ci_covers_zero,
            tolerance=0.0,
        ),
    ]


# ════════════════════════════════════════════════════════════════════════════
# Provenance + runner — thin wrappers over nousergon_lib.quant.selftest
# ════════════════════════════════════════════════════════════════════════════

def resolved_library_versions(
    distributions: tuple[str, ...] = _TRACKED_DISTRIBUTIONS,
) -> dict[str, str]:
    """The installed version of every quant distribution loaded at runtime.

    Thin wrapper binding this repo's tracked distribution tuple to the lifted
    lib implementation (alpha-engine-config-I7238).
    """
    return _lib_resolved_library_versions(distributions)


def code_sha() -> str:
    """The SHA of the code that ran, without shelling out.

    Thin wrapper binding this repo's checkout root to the lifted lib
    implementation (alpha-engine-config-I7238).
    """
    return _lib_code_sha(Path(__file__).resolve().parents[1])


def run_self_test(
    run_date: str | None = None,
    *,
    case_provider: "Callable[[], list[Case]] | None" = None,
    component: str = COMPONENT,
    schema: str = SCHEMA,
    case_timeout_seconds: float = CASE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run the known-answer battery on the DEPLOYED instrument and return the artifact body.

    Never raises — see the module docstring's CONTRACT section. Delegates the
    outcome taxonomy, provenance header and timeout handling to
    ``nousergon_lib.quant.selftest.run_self_test`` (alpha-engine-config-I7238);
    this wrapper supplies only this repo's identity (``component``/``schema``),
    default battery (``build_cases``), and resolved provenance.
    """
    return _lib_run_self_test(
        run_date,
        case_provider=case_provider or build_cases,
        component=component,
        schema=schema,
        resolved_libraries=resolved_library_versions(),
        code_sha_value=code_sha(),
        case_timeout_seconds=case_timeout_seconds,
    )


def self_test_key(run_date: str) -> str:
    """S3 key of the backtester's published self-test for ``run_date``."""
    return f"backtest/{run_date}/{ARTIFACT_FILENAME}"
