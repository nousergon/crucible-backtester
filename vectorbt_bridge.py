"""
vectorbt_bridge.py — convert executor order list to a vectorbt Portfolio.

This is the only custom glue needed between the executor's output and vectorbt's
analytics engine. Everything else (Sharpe, drawdown, alpha, benchmark comparison)
is handled by vbt.Portfolio methods directly.
"""

import logging
import math

import numpy as np
import pandas as pd
import vectorbt as vbt

from analysis.risk_matched_benchmark import compute_alpha_vs_benchmark

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252


def _compute_active_window(pf: vbt.Portfolio) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Compute the portfolio's active date range — first non-flat NAV through last wrapper date.

    For a portfolio that doesn't trade for some prefix of its wrapper.index (typical when
    a 10y price-matrix is supplied but the simulator only enters positions in the final
    weeks), returns the window starting at the first NAV change. Anchoring benchmark
    comparisons (SPY, EW-high-vol) on this window prevents apples-to-oranges errors
    where the flat-prefix portfolio's `total_return` reflects only its active period
    but the benchmark return compounds over the full wrapper — the symptom that
    surfaced on the 2026-05-24 backtest as `ew_high_vol_return: 960%`,
    `alpha_vs_ew_high_vol: -954%` (10y basket compound vs 27-day portfolio return).

    Returns ``None`` if the portfolio never trades (NAV constant across the whole
    wrapper) or if the wrapper has fewer than 2 dates.
    """
    wrapper_index = pf.wrapper.index
    if len(wrapper_index) < 2:
        return None

    value = pf.value()
    initial_value = float(value.iloc[0])
    nav_changes = value[~np.isclose(value.to_numpy(dtype=np.float64), initial_value, atol=1e-9)]
    if len(nav_changes) == 0:
        return None

    return nav_changes.index[0], wrapper_index[-1]


def _compute_benchmark_leg(
    *,
    total_return: float,
    active_window: tuple[pd.Timestamp, pd.Timestamp] | None,
    benchmark_daily_returns: pd.Series | None,
    label: str,
) -> tuple[float | None, float | None]:
    """Shared benchmark-leg primitive: (benchmark_total_return, alpha) or (None, None).

    Generalizes the per-leg logic that used to be hand-duplicated across the
    SPY and EW-high-vol blocks in ``portfolio_stats`` (each ~40 lines,
    including the active-window-anchoring fix below). Every benchmark leg —
    SPY, EW-high-vol, full-universe EW, per-sector-ETF — now funnels through
    this one helper so the anchoring fix applies uniformly instead of only to
    whichever leg happened to get it first.

    Parameters
    ----------
    total_return : float
        ``pf.total_return()`` — the portfolio's total return (same scalar
        used for every leg; NAV is flat before the active window so this is
        already equivalent to the active-window return).
    active_window : (Timestamp, Timestamp) | None
        Portfolio's active date range from ``_compute_active_window``.
        ``None`` means the portfolio never traded — leg degrades to
        ``(None, None)``.
    benchmark_daily_returns : pd.Series | None
        Daily simple returns for the benchmark (SPY, EW-high-vol basket,
        full-universe EW basket, or a single sector ETF), indexed by trading
        day. ``None`` means the caller didn't provide this leg's data.
    label : str
        Stamped onto the ``compute_alpha_vs_benchmark`` call and used only in
        log messages here — callers own the stats-dict key names.

    Returns
    -------
    (benchmark_total_return, alpha) both ``None`` if the leg can't be
    computed (missing data, no active window, or <2 aligned points within
    the active window — mirrors the 2026-05-24 anchoring fix so every leg,
    not just SPY, avoids comparing an active-window ``total_return`` against
    a benchmark compounded over the full (possibly multi-year) wrapper).

    Note: this deliberately does NOT accept the portfolio's own daily-returns
    series. ``compute_alpha_vs_benchmark`` inner-joins its two Series
    arguments before computing ``benchmark_total_return`` — passing the real
    portfolio returns here would let a portfolio-side data gap inside the
    active window silently truncate the benchmark's compounded total (caught
    in review: an 8-day synthetic portfolio gap changed a benchmark leg's
    total return by several points even though the benchmark series itself
    had no gap). A benchmark leg's total return must depend ONLY on the
    benchmark's own aligned dates. ``compute_alpha_vs_benchmark`` is still
    called below (self-joined against its own aligned series) so the
    compounding arithmetic runs through the shared library primitive per
    config#834's "don't hand-roll a parallel implementation" constraint —
    only its ``benchmark_total_return`` field is used; ``alpha`` is computed
    locally from ``total_return`` for parity with every leg's pre-existing
    semantics.
    """
    if benchmark_daily_returns is None:
        return None, None
    if active_window is None:
        logger.warning(
            "vectorbt_bridge._compute_benchmark_leg[%s]: portfolio NAV never "
            "changed across wrapper.index (no active window) — leg emits as null",
            label,
        )
        return None, None

    active_start, active_end = active_window
    bench_aligned = benchmark_daily_returns.loc[active_start:active_end].dropna()
    if len(bench_aligned) < 2:
        logger.warning(
            "vectorbt_bridge._compute_benchmark_leg[%s]: benchmark series has "
            "<2 aligned values within active window [%s, %s] — leg emits as null",
            label, active_start, active_end,
        )
        return None, None

    # IMPORTANT: compute_alpha_vs_benchmark inner-joins portfolio_daily_returns
    # against benchmark_daily_returns before computing `benchmark_total_return`
    # — so if the caller's portfolio series has ANY gap inside the active
    # window relative to the benchmark's dates (not the case for
    # `daily_returns` from `pf.returns()`, which is dense over the whole
    # active window in the normal vectorbt case, but plausible for an
    # externally-fetched sector-ETF/ew_universe series with its own data
    # gaps), the join would silently truncate `benchmark_total_return` to
    # fewer days than `bench_aligned` actually has — a benchmark leg's total
    # return must depend ONLY on the benchmark's own aligned dates, not on
    # portfolio-return availability (this was caught in review: an 8-day
    # synthetic portfolio gap changed a benchmark leg's total return by
    # several points even though the benchmark series itself was complete).
    # Passing `bench_aligned` as BOTH arguments makes the internal join a
    # self-join (no-op truncation) while still routing the total-return
    # arithmetic through the shared library primitive per config#834's
    # "don't hand-roll a parallel implementation" constraint — the discarded
    # `portfolio_total_return`/`excess_return`/`information_ratio` fields
    # from this call are meaningless (bench vs itself) and are never read;
    # only `benchmark_total_return` (independent of the second argument's
    # values, only its date index) is used below.
    result = compute_alpha_vs_benchmark(
        portfolio_daily_returns=bench_aligned,
        benchmark_daily_returns=bench_aligned,
        label=label,
    )
    if result.get("status") != "ok":
        logger.warning(
            "vectorbt_bridge._compute_benchmark_leg[%s]: compute_alpha_vs_benchmark "
            "returned status=%s within active window [%s, %s] — leg emits as null",
            label, result.get("status"), active_start, active_end,
        )
        return None, None

    benchmark_total_return = float(result["benchmark_total_return"])
    alpha = total_return - benchmark_total_return
    return benchmark_total_return, alpha


def _compute_sortino_ratio(daily_returns: pd.Series, target: float = 0.0) -> float:
    """Annualized Sortino ratio (target = 0 by default).

    Definition: (mean(r) - target) / downside_deviation * sqrt(252), where
    downside_deviation = sqrt(mean(min(r - target, 0)**2)) — i.e. RMS of
    only the below-target excursions. Dropped NaN before compute.

    Returns 0.0 when there are no below-target days (no downside) or when
    the input series has fewer than 2 valid observations. Sample-std-style
    ddof=0 in the downside RMS to match the standard definition (Sortino
    1991); this differs from the sample std used in Sharpe.
    """
    r = daily_returns.dropna().to_numpy(dtype=np.float64)
    if r.size < 2:
        return 0.0
    excess = r - target
    downside = np.minimum(excess, 0.0)
    downside_var = float(np.mean(downside * downside))
    if downside_var <= 0.0:
        return 0.0
    downside_dev = math.sqrt(downside_var)
    return float(excess.mean() / downside_dev * math.sqrt(_TRADING_DAYS_PER_YEAR))


def _compute_cvar(daily_returns: pd.Series, q: float = 0.05) -> float:
    """Conditional Value at Risk at the q-quantile (default 5%).

    CVaR_q = mean of the worst q-fraction of daily returns. Reported as the
    raw return value — a negative number means "in the worst 5% of days,
    mean return is X%". Convention: lower (more negative) = worse tail.

    Returns 0.0 when fewer than ceil(1/q) observations are available
    (insufficient resolution to define the tail).
    """
    if not (0.0 < q < 1.0):
        raise ValueError(f"q must be in (0, 1), got {q}")
    r = daily_returns.dropna().to_numpy(dtype=np.float64)
    min_n = math.ceil(1.0 / q)
    if r.size < min_n:
        return 0.0
    sorted_r = np.sort(r)
    n_tail = max(1, int(math.floor(r.size * q)))
    return float(sorted_r[:n_tail].mean())


def orders_to_portfolio(
    orders: list[dict],
    prices: pd.DataFrame,
    init_cash: float = 1_000_000.0,
    fees: float = 0.001,
    slippage_bps: float = 0.0,
    assume_next_day_fill: bool = False,
) -> vbt.Portfolio:
    """
    Convert executor order list to a vectorbt Portfolio.

    Args:
        orders: List of order dicts from executor.main.run(simulate=True):
            [{"date": "2026-03-06", "ticker": "PLTR", "action": "ENTER",
              "shares": 100, "price_at_order": 84.12}, ...]
        prices: DataFrame indexed by date (datetime), columns by ticker.
                Build with price_loader.build_matrix().
        init_cash: Starting portfolio NAV.
        fees: Base transaction fees (fraction, e.g. 0.001 = 10bps round-trip).
        slippage_bps: Additional slippage per side in basis points (e.g. 10 = 10bps).
        assume_next_day_fill: If True, shift ENTER orders forward by 1 trading day
            to simulate next-day-close fills. EXIT/REDUCE stay same-day (conservative).

    Returns:
        vbt.Portfolio with full analytics available via .sharpe_ratio(),
        .max_drawdown(), .total_return(), .plot(), etc.
    """
    tickers = prices.columns.tolist()
    dates = prices.index

    entries = pd.DataFrame(False, index=dates, columns=tickers)
    exits   = pd.DataFrame(False, index=dates, columns=tickers)
    sizes   = pd.DataFrame(0.0,   index=dates, columns=tickers)

    for order in orders:
        d = pd.Timestamp(order["date"])
        t = order["ticker"]
        if t not in tickers or d not in entries.index:
            continue
        if order["action"] == "ENTER":
            fill_date = d
            if assume_next_day_fill:
                # Shift to next trading day in the price matrix
                idx_pos = dates.get_loc(d)
                if idx_pos + 1 < len(dates):
                    fill_date = dates[idx_pos + 1]
                else:
                    continue  # no next trading day available — skip order
            entries.loc[fill_date, t] = True
            sizes.loc[fill_date, t]   = float(order.get("shares", 0))
        elif order["action"] in ("EXIT", "REDUCE"):
            exits.loc[d, t] = True

    # Combine base fees with slippage: total_fees = base_fees + slippage_bps/10000
    total_fees = fees + slippage_bps / 10_000

    return vbt.Portfolio.from_signals(
        close=prices,
        entries=entries,
        exits=exits,
        size=sizes,
        size_type="Amount",
        init_cash=init_cash,
        cash_sharing=True,
        group_by=True,
        fees=total_fees,
        freq="D",
    )


def portfolio_stats(
    pf: vbt.Portfolio,
    spy_prices: pd.Series | None = None,
    ew_high_vol_basket_returns: pd.Series | None = None,
    ew_universe_basket_returns: pd.Series | None = None,
    sector_etf_basket_returns: dict[str, pd.Series] | None = None,
) -> dict:
    """
    Extract key metrics from a vectorbt Portfolio into a plain dict.

    Suitable for writing to metrics.json or printing as a summary.

    If ``spy_prices`` (Close series for SPY, same DatetimeIndex as portfolio)
    is provided, ``total_alpha = portfolio return - SPY return`` is computed
    (presentation framing — cap-weighted SPY is the headline benchmark in
    morning emails + dashboards).

    If ``ew_high_vol_basket_returns`` (daily simple returns Series from
    ``analysis.risk_matched_benchmark.construct_ew_high_vol_benchmark``,
    indexed by trading day) is provided, ``alpha_vs_ew_high_vol`` is
    computed against the same active date range. This is the institutional
    skill-isolation framing per evaluator-revamp-260506.md Workstream D:
    "given how much risk you took, did you outperform the dumb version of
    taking that risk?" The basket holds the top vol-quartile of the agent's
    decision universe, equal-weighted + rebalanced weekly. Both columns
    are emitted alongside; callers decide which to anchor on.

    If ``ew_universe_basket_returns`` (daily simple returns Series, e.g. from
    ``construct_ew_high_vol_benchmark(prices, vol_quantile=<full universe>)``
    or an equivalent full-universe equal-weight constructor) is provided,
    ``alpha_vs_ew_universe`` is computed the same way. This isolates
    stock-selection alpha from cap-weighted-tilt alpha: unlike
    ``ew_high_vol_basket_returns`` (top vol-quartile only), this basket holds
    the FULL decision universe, equal-weighted — "did you beat picking
    everything, unweighted?" as opposed to "did you beat picking risky
    stuff?" (config#834).

    If ``sector_etf_basket_returns`` (``{ticker: daily simple returns
    Series}``, e.g. built from the executor's ``sector_etf_histories``/
    ``SECTOR_ETF_MAP`` price plumbing, one entry per sector ETF present in
    the run's universe) is provided, one ``alpha_vs_sector_<TICKER>`` +
    ``sector_<TICKER>_return`` pair is emitted per ticker — sector-relative
    alpha ("did the system's healthcare picks beat XLV?", config#834).

    Every benchmark leg above is computed by the same ``_compute_benchmark_leg``
    primitive (SPY's daily returns are derived from ``spy_prices`` first) —
    see that function's docstring for the active-window-anchoring contract
    shared by all of them.

    Includes a daily return series + log return series + downside-aware
    metrics (Sortino, CVaR(95)) needed by the evaluator-revamp metric
    stack (see evaluator-revamp-260506.md). The series are pd.Series
    indexed by trading day so downstream consumers (risk-matched
    benchmark, IR, PSR/DSR) can align with SPY/sector returns.
    """
    total_return = float(pf.total_return())

    # Daily return series — aligned with the portfolio's date index.
    # vectorbt's `pf.returns()` returns a Series; coerce to float64 + drop
    # the first-day NaN (no prior NAV to diff against) so downstream
    # consumers can use it without re-cleaning.
    daily_returns = pd.Series(pf.returns(), name="daily_return").astype(np.float64)
    daily_returns = daily_returns.dropna()
    # log(1 + r); guard against r <= -1 (full wipeout) to avoid -inf.
    daily_log_returns = np.log1p(daily_returns.clip(lower=-0.999999))
    daily_log_returns.name = "daily_log_return"

    sortino = _compute_sortino_ratio(daily_returns)
    cvar_95 = _compute_cvar(daily_returns, q=0.05)

    # Probabilistic Sharpe Ratio — confidence that the true Sharpe > 0
    # given the observed sample (Bailey & López de Prado 2012). Computed
    # inline so it survives the parquet round-trip into sweep_df as a
    # scalar (the daily_returns Series doesn't). Used by
    # executor_optimizer.recommend()'s skill-composite mode as a
    # confidence gate before live promotion.
    psr_scalar: float | None = None
    try:
        from analysis.dsr import compute_psr
        psr_result = compute_psr(daily_returns, sharpe_benchmark=0.0)
        if psr_result.get("status") == "ok":
            psr_scalar = float(psr_result["psr"])
    except Exception:
        # PSR is best-effort — failure to compute (e.g. degenerate
        # returns) must not break portfolio_stats. None signals "not
        # available" to downstream gates.
        psr_scalar = None

    # Resolve the portfolio's active date range up front — SPY + basket legs
    # both need it to avoid the 10y-wrapper-vs-27-day-portfolio mismatch that
    # produced the 2026-05-24 `ew_high_vol_return: 960%` / `alpha_vs_ew_high_vol:
    # -954%` regression. ``None`` means "portfolio never traded" — both
    # benchmark legs degrade to ``None`` for that case (and emit a WARN).
    active_window = _compute_active_window(pf)
    null_legs: list[str] = []

    stats = {
        "total_return": total_return,
        "sharpe_ratio": float(pf.sharpe_ratio()),
        "sortino_ratio": sortino,
        "max_drawdown": float(pf.max_drawdown()),
        "calmar_ratio": float(pf.calmar_ratio()),
        "cvar_95": cvar_95,
        "psr": psr_scalar,
        "total_trades": int(pf.trades.count()),
        "win_rate": float(pf.trades.win_rate()),
        "daily_returns": daily_returns,
        "daily_log_returns": daily_log_returns,
    }

    # SPY's leg input is a price series (not returns) everywhere else in this
    # function — convert to daily simple returns so it can go through the
    # same `_compute_benchmark_leg` primitive as every other leg. CAUTION:
    # must slice to the active window BEFORE differencing, not after — the
    # prior hand-rolled SPY block computed `P[active_end]/P[active_start] - 1`
    # (a ratio of the two window ENDPOINTS). Differencing the full price
    # series into daily returns first and THEN slicing `[active_start,
    # active_end]` would instead compound to `P[active_end]/P[active_start-1]
    # - 1` — the return "into" active_start (dated active_start, computed
    # against the prior close) falls inside that date-range slice even
    # though it reflects a day mostly BEFORE active_start. Slicing the
    # price level first and differencing within the slice reproduces the
    # exact endpoint ratio (telescoping product over only the in-window
    # diffs), matching `TestOracleAgreement`'s independent NumPy reference.
    if spy_prices is not None and active_window is not None:
        spy_active_start, spy_active_end = active_window
        spy_daily_returns = (
            spy_prices.loc[spy_active_start:spy_active_end].pct_change().dropna()
        )
    elif spy_prices is not None:
        # No active window (portfolio never traded) — _compute_benchmark_leg
        # degrades to null regardless of this series' shape; pct_change over
        # the full series is a harmless placeholder for that branch.
        spy_daily_returns = spy_prices.pct_change().dropna()
    else:
        spy_daily_returns = None

    def _emit_leg(
        *, benchmark_daily_returns, label, return_key, alpha_key,
        canonical: bool = False, missing_warning: str | None = None,
    ):
        """Compute one leg and write its two stats keys + null_legs bookkeeping.

        ``canonical=True`` (SPY only, matching the pre-existing convention):
        an omitted kwarg (``benchmark_daily_returns is None``) still counts
        as "could not compute" for ``null_legs`` purposes, since SPY is the
        always-expected headline benchmark. Every other leg is opt-in — an
        omitted kwarg means the caller never asked for that leg, so it's
        silently ``None`` and NOT flagged in ``null_legs`` (only "requested
        but failed to compute" cases are, e.g. no active window or <2
        aligned points).
        """
        bench_return, alpha = _compute_benchmark_leg(
            total_return=total_return,
            active_window=active_window,
            benchmark_daily_returns=benchmark_daily_returns,
            label=label,
        )
        stats[return_key] = bench_return
        stats[alpha_key] = alpha
        if bench_return is None and (canonical or benchmark_daily_returns is not None):
            if benchmark_daily_returns is None and missing_warning:
                logger.warning(missing_warning)
            null_legs.append(return_key)
            null_legs.append(alpha_key)

    # Compute alpha vs SPY over the portfolio's ACTIVE date range — anchoring
    # on `pf.wrapper.index` directly (the prior implementation) compares
    # `total_return` (effectively a 27-day P&L expressed as fraction of initial
    # capital) against `spy_return` compounded over the full 10y wrapper.
    _emit_leg(
        benchmark_daily_returns=spy_daily_returns,
        label="spy",
        return_key="spy_return",
        alpha_key="total_alpha",
        canonical=True,
        missing_warning=(
            "vectorbt_bridge.portfolio_stats: spy_prices not provided — "
            "spy_return/total_alpha emit as null"
        ),
    )

    # Compute alpha vs EW-high-vol basket — institutional skill-isolation
    # framing per evaluator-revamp-260506.md Workstream D. Basket holds the
    # top vol-quartile of the agent's decision universe, equal-weighted +
    # rebalanced weekly; ``construct_ew_high_vol_benchmark`` produces the
    # daily-returns series this kwarg expects. Compounded over the
    # portfolio's ACTIVE date range to a single total-return scalar before
    # differencing — matches ``total_alpha``'s shape so consumers can swap
    # which one they rank on without other plumbing changes.
    _emit_leg(
        benchmark_daily_returns=ew_high_vol_basket_returns,
        label="ew_high_vol",
        return_key="ew_high_vol_return",
        alpha_key="alpha_vs_ew_high_vol",
    )

    # Compute alpha vs the full-universe equal-weight basket — the
    # stock-selection-vs-cap-weighted-tilt isolation leg (config#834).
    # Distinct from ew_high_vol above (top vol-quartile only): this basket
    # is the agent's ENTIRE decision universe, equal-weighted + rebalanced.
    # "Given everything you could have picked, unweighted, did you add value
    # by picking specific names?" — same active-window anchoring as every
    # other leg.
    _emit_leg(
        benchmark_daily_returns=ew_universe_basket_returns,
        label="ew_universe",
        return_key="ew_universe_return",
        alpha_key="alpha_vs_ew_universe",
    )

    # Compute alpha vs each sector ETF present in the run's universe —
    # sector-relative alpha (config#834): "does the system's healthcare
    # picks beat XLV?" One pair of keys per ticker in
    # ``sector_etf_basket_returns`` (caller is responsible for scoping the
    # dict to the sectors actually represented in the run's universe, e.g.
    # via the executor's SECTOR_ETF_MAP + sector_etf_histories plumbing).
    for ticker, sector_daily_returns in sorted((sector_etf_basket_returns or {}).items()):
        _emit_leg(
            benchmark_daily_returns=sector_daily_returns,
            label=f"sector_{ticker}",
            return_key=f"sector_{ticker}_return",
            alpha_key=f"alpha_vs_sector_{ticker}",
        )

    if null_legs:
        stats["null_legs"] = null_legs

    return stats

_BRIDGE_REV = "vb-7f3a91c4e2"
