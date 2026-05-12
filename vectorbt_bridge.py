"""
vectorbt_bridge.py — convert executor order list to a vectorbt Portfolio.

This is the only custom glue needed between the executor's output and vectorbt's
analytics engine. Everything else (Sharpe, drawdown, alpha, benchmark comparison)
is handled by vbt.Portfolio methods directly.
"""

import math

import numpy as np
import pandas as pd
import vectorbt as vbt

_TRADING_DAYS_PER_YEAR = 252


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

    # Compute alpha vs SPY over the portfolio's active date range
    if spy_prices is not None:
        pf_dates = pf.wrapper.index
        spy_aligned = spy_prices.reindex(pf_dates).dropna()
        if len(spy_aligned) >= 2:
            spy_return = float((spy_aligned.iloc[-1] / spy_aligned.iloc[0]) - 1.0)
            stats["spy_return"] = spy_return
            stats["total_alpha"] = total_return - spy_return
        else:
            stats["spy_return"] = None
            stats["total_alpha"] = None
    else:
        stats["spy_return"] = None
        stats["total_alpha"] = None

    # Compute alpha vs EW-high-vol basket — institutional skill-isolation
    # framing per evaluator-revamp-260506.md Workstream D. Basket holds the
    # top vol-quartile of the agent's decision universe, equal-weighted +
    # rebalanced weekly; ``construct_ew_high_vol_benchmark`` produces the
    # daily-returns series this kwarg expects. Compounded over the
    # portfolio's active date range to a single total-return scalar before
    # differencing — matches ``total_alpha``'s shape so consumers can swap
    # which one they rank on without other plumbing changes.
    if ew_high_vol_basket_returns is not None:
        pf_dates = pf.wrapper.index
        basket_aligned = ew_high_vol_basket_returns.reindex(pf_dates).dropna()
        if len(basket_aligned) >= 2:
            basket_total_return = float((1.0 + basket_aligned).prod() - 1.0)
            stats["ew_high_vol_return"] = basket_total_return
            stats["alpha_vs_ew_high_vol"] = total_return - basket_total_return
        else:
            stats["ew_high_vol_return"] = None
            stats["alpha_vs_ew_high_vol"] = None
    else:
        stats["ew_high_vol_return"] = None
        stats["alpha_vs_ew_high_vol"] = None

    return stats
