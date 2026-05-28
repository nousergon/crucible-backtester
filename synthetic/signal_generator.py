"""
synthetic/signal_generator.py — convert GBM predictions + technical indicators
to executor-compatible signals.

Previous version used a broken alpha-to-score mapping (50 + alpha * 1000) that
clustered all scores at 45-55, producing zero ENTER signals.  This version
computes real technical scores from OHLCV price history and enriches them with
GBM alpha predictions (±10 pts max).

Score composition:
    technical_score = weighted RSI(14) + MACD + MA50 + MA200 + momentum
    trading_score   = technical_score + clip(gbm_alpha * max_enrichment, -10, +10)

Signal assignment:
    trading_score >= min_score AND top_n → ENTER
    trading_score < 30                   → EXIT
    else                                 → HOLD

Conviction (for position sizing):
    alpha >= 0.02   → "rising"
    alpha <= -0.01  → "declining"
    else            → "stable"
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Sector ETF → human-readable sector name ─────────────────────────────────
_ETF_TO_SECTOR = {
    "XLK": "Technology",
    "XLF": "Financial Services",
    "XLV": "Healthcare",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLY": "Consumer Cyclical",
    "XLP": "Consumer Defensive",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
    "XLB": "Basic Materials",
}


# ── Technical scoring (same formulas as executor/technical_scorer.py) ────────
# Inlined here to avoid cross-repo import from alpha-engine.

def _score_rsi(
    rsi: float,
    market_regime: str = "neutral",
    drawdown_tier: str | None = None,
) -> float:
    # 3-class Ang-Bekaert macro regime (v0.42.0 / 2026-05-28 —
    # caution-regime-retirement-260528.md). Legacy 4-class "caution"
    # at market_regime is grandfathered for replay over historical
    # signals.json artifacts; new synthetic signal generation should
    # pass drawdown_tier on the orthogonal drawdown axis for the
    # protective-RSI window (the institutional 3-state Bridgewater
    # hysteresis pattern preserved in the drawdown leg).
    bear_window = (
        market_regime == "bear"
        or market_regime == "caution"  # legacy grandfather
        or (drawdown_tier is not None and drawdown_tier in ("caution", "risk_off"))
    )
    if market_regime == "bull" and not bear_window:
        overbought, oversold, max_os = 80, 30, 100.0
    elif bear_window:
        overbought, oversold, max_os = 70, 40, 65.0
    else:
        overbought, oversold, max_os = 70, 30, 100.0

    if rsi >= overbought:
        return 0.0
    if rsi <= oversold:
        return max_os
    return max_os * (overbought - rsi) / (overbought - oversold)


def _score_macd(macd_cross: float, macd_above_zero: bool) -> float:
    if macd_cross == 1.0:
        return 100.0 if macd_above_zero else 70.0
    if macd_cross == -1.0:
        return 30.0 if macd_above_zero else 0.0
    return 60.0 if macd_above_zero else 40.0


def _score_price_vs_ma(pct_diff: Optional[float]) -> float:
    if pct_diff is None:
        return 50.0
    if pct_diff >= 5:
        return min(100.0, 80.0 + (pct_diff - 5) * (20.0 / 15.0))
    if pct_diff >= 0:
        return 50.0 + pct_diff * 6.0
    if pct_diff > -5:
        return 50.0 + pct_diff * 4.0
    return max(0.0, 30.0 - (abs(pct_diff) - 5) * 1.5)


def _score_momentum(
    momentum_20d: Optional[float],
    percentile_rank: Optional[float] = None,
) -> float:
    if percentile_rank is not None:
        return float(percentile_rank)
    if momentum_20d is None:
        return 50.0
    return max(0.0, min(100.0, 50.0 + momentum_20d * 3.0))


def _compute_technical_score(
    indicators: dict,
    market_regime: str = "neutral",
    momentum_percentile: Optional[float] = None,
) -> float:
    rsi = _score_rsi(indicators.get("rsi_14", 50.0), market_regime)
    macd = _score_macd(indicators.get("macd_cross", 0.0), indicators.get("macd_above_zero", False))
    ma50 = _score_price_vs_ma(indicators.get("price_vs_ma50"))
    ma200 = _score_price_vs_ma(indicators.get("price_vs_ma200"))
    mom = _score_momentum(indicators.get("momentum_20d"), momentum_percentile)
    return round(max(0.0, min(100.0, rsi * 0.25 + macd * 0.20 + ma50 * 0.20 + ma200 * 0.20 + mom * 0.15)), 2)


def _compute_momentum_percentiles(
    momentum_data: dict[str, Optional[float]],
) -> dict[str, float]:
    valid = [(t, m) for t, m in momentum_data.items() if m is not None]
    if not valid:
        return {t: 50.0 for t in momentum_data}
    tickers, values = zip(*valid)
    arr = np.array(values, dtype=float)
    ranks = (arr.argsort().argsort() / max(len(arr) - 1, 1)) * 100
    result = {t: round(float(r), 1) for t, r in zip(tickers, ranks)}
    for t in momentum_data:
        result.setdefault(t, 50.0)
    return result


# ── Indicator computation from OHLCV ─────────────────────────────────────────

def precompute_indicator_series(
    ohlcv_by_ticker: "dict[str, pd.DataFrame]",
) -> dict[str, pd.DataFrame]:
    """
    Vectorized pre-computation of 6 technical indicators as full
    date-indexed Series per ticker. Used by ``build_signals_by_date``
    to replace a O(dates × tickers × bars) per-date Python rescan with
    a single O(tickers × bars) pandas vectorized pass +
    O(dates × tickers) hashtable lookups.

    Motivation: 2026-04-21 dry-run profiling showed ``build_signals_by_date``
    took ~75 minutes on 2277 dates × ~900 tickers — the per-date loop was
    filtering each ticker's full 10y bar list by ``<=date`` and rebuilding a
    pandas Series from scratch, 2277 times per ticker. That's ~5B string
    comparisons in pure Python interpreter. Every cost is pure algorithmic
    laziness — the underlying data already has a date axis, pandas can roll
    everything in one pass.

    Expected speedup: 50-100x for this phase.

    Input contract: ``dict[str, pd.DataFrame]`` per
    ``build_ohlcv_df_by_ticker`` (DatetimeIndex + lowercase ``close``
    column). The pre-2026-04-23 list-of-dicts producer was deleted in
    Option A step 9 cleanup; legacy artifacts now hard-fail at the
    ``load_ohlcv_by_ticker`` layer.

    Returns
    -------
    {ticker: DataFrame} where each DataFrame has the ticker's bar dates as
    its index (string dates, YYYY-MM-DD) and these columns:
        rsi_14, macd_cross, macd_above_zero, price_vs_ma50,
        price_vs_ma200, momentum_20d
    Tickers with zero bars are omitted.
    """
    out: dict[str, pd.DataFrame] = {}
    for ticker, df in ohlcv_by_ticker.items():
        if df is None or df.empty:
            continue
        # DatetimeIndex + lowercase "close" column. Copy the close series
        # so the .index reassignment below doesn't mutate the caller's
        # DataFrame. Convert index to YYYY-MM-DD strings so the output
        # contract (string-date-indexed frames consumed by
        # indicators_from_precomputed via df.index.get_loc) is preserved.
        close = df["close"].astype(float).copy()
        close.index = close.index.strftime("%Y-%m-%d")

        # RSI(14) via Wilder's smoothing — vectorized: one EWMA pass
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + rs))

        # MACD(12, 26, 9) — vectorized
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_above_zero = (macd_line > 0)

        # macd_cross semantics (per scalar impl above): at date T, output is
        # the most recent cross direction within the 3-bar window {T-2, T-1, T},
        # or 0 if no cross. Vectorized: mark cross bars, forward-fill up to
        # 2 NaN gaps (= 3-bar window inclusive), fill remaining NaN with 0.
        diff = macd_line - signal_line
        prev = diff.shift(1)
        up_cross = (diff >= 0) & (prev < 0)
        down_cross = (diff < 0) & (prev >= 0)
        raw_cross = pd.Series(0.0, index=close.index)
        raw_cross[up_cross] = 1.0
        raw_cross[down_cross] = -1.0
        macd_cross = raw_cross.replace(0.0, float("nan")).ffill(limit=2).fillna(0.0)

        # Price vs MA50 / MA200 — vectorized
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()
        price_vs_ma50 = ((close - ma50) / ma50.replace(0, float("nan"))) * 100
        price_vs_ma200 = ((close - ma200) / ma200.replace(0, float("nan"))) * 100

        # 20-day momentum — vectorized
        momentum_20d = (close / close.shift(20) - 1) * 100

        out[ticker] = pd.DataFrame({
            "rsi_14": rsi,
            "macd_cross": macd_cross,
            "macd_above_zero": macd_above_zero,
            "price_vs_ma50": price_vs_ma50,
            "price_vs_ma200": price_vs_ma200,
            "momentum_20d": momentum_20d,
        })

    return out


def indicators_from_precomputed(
    precomputed: dict[str, pd.DataFrame],
    tickers: list[str] | set[str] | dict,
    date_str: str,
    min_bars: int = 210,
) -> dict[str, dict]:
    """Look up the indicator row for ``date_str`` from each ticker's
    precomputed DataFrame. Returns ``{ticker: indicator_dict}`` with
    keys ``rsi_14, macd_cross, macd_above_zero, price_vs_ma50,
    price_vs_ma200, momentum_20d``. Tickers are excluded when:
      - the ticker is not in ``precomputed``,
      - ``date_str`` isn't in that ticker's index,
      - fewer than ``min_bars`` bars of history exist up to and including
        ``date_str`` (matches the scalar path's ``min_bars=210`` gate —
        short-history tickers don't produce reliable indicators even
        though EWMA technically yields a value from bar 1).
    """
    result: dict[str, dict] = {}
    for ticker in tickers:
        df = precomputed.get(ticker)
        if df is None:
            continue
        try:
            # get_loc + slice is the fast way to both check membership and
            # measure position (= "bars up to this date") in one lookup.
            pos = df.index.get_loc(date_str)
        except KeyError:
            continue
        if (pos + 1) < min_bars:
            continue
        row = df.iloc[pos]
        # rsi_14 is the first-populated indicator; if it's NaN at this
        # position the whole row is unreliable.
        if pd.isna(row["rsi_14"]):
            continue
        result[ticker] = {
            "rsi_14": float(row["rsi_14"]),
            "macd_cross": float(row["macd_cross"]),
            "macd_above_zero": bool(row["macd_above_zero"]),
            "price_vs_ma50": (
                None if pd.isna(row["price_vs_ma50"])
                else float(row["price_vs_ma50"])
            ),
            "price_vs_ma200": (
                None if pd.isna(row["price_vs_ma200"])
                else float(row["price_vs_ma200"])
            ),
            "momentum_20d": (
                None if pd.isna(row["momentum_20d"])
                else float(row["momentum_20d"])
            ),
        }
    return result


# ── Signal generation ─────────────────────────────────────────────────────────

def predictions_to_signals(
    predictions: dict[str, float],
    date: str,
    sector_map: dict[str, str],
    precomputed_indicators: dict[str, dict],
    market_regime: str = "neutral",
    top_n: int = 20,
    min_score: float = 60,
    gbm_enrichment_max: float = 10.0,
) -> dict:
    """
    Convert GBM alpha predictions + precomputed indicators to executor signals.

    For each ticker:
    1. Look up technical indicators from ``precomputed_indicators``
    2. Score via _compute_technical_score() → 0-100
    3. Enrich with GBM alpha: ±gbm_enrichment_max pts
    4. Assign signal (ENTER/EXIT/HOLD) based on trading_score

    Parameters
    ----------
    predictions : {ticker: alpha_score} from GBM inference.
    date : date string (YYYY-MM-DD).
    sector_map : {ticker: sector_etf_symbol}.
    precomputed_indicators : {ticker: indicator_dict} — the result of
        ``indicators_from_precomputed`` for this date. Required input
        (Option A step 9 cleanup deleted the scalar fallback path that
        accepted raw OHLCV here).
    market_regime : 'bull' | 'neutral' | 'bear' (3-class Ang-Bekaert; the
        legacy 4th value 'caution' is grandfathered on read for historical
        signals.json artifacts post v0.42.0 —
        caution-regime-retirement-260528.md).
    top_n : max ENTER signals per date.
    min_score : minimum trading_score for ENTER.
    gbm_enrichment_max : max ±pts GBM can adjust technical score.
    """
    indicators_by_ticker = precomputed_indicators

    # Step 2: Compute momentum percentiles across all scored tickers
    momentum_data = {
        t: ind.get("momentum_20d") for t, ind in indicators_by_ticker.items()
    }
    percentiles = _compute_momentum_percentiles(momentum_data)

    # Step 3: Score each ticker
    scored = []
    for ticker, alpha in predictions.items():
        indicators = indicators_by_ticker.get(ticker)
        if indicators is None:
            # No price data — skip this ticker
            continue

        tech_score = _compute_technical_score(
            indicators,
            market_regime=market_regime,
            momentum_percentile=percentiles.get(ticker),
        )

        # GBM enrichment
        gbm_adj = max(-gbm_enrichment_max, min(gbm_enrichment_max, alpha * 500.0))
        trading_score = round(max(0.0, min(100.0, tech_score + gbm_adj)), 2)

        conviction = _assign_conviction(alpha)
        sector_etf = sector_map.get(ticker, "")
        sector = _ETF_TO_SECTOR.get(sector_etf, "Technology")

        signal = "HOLD"
        if trading_score >= min_score:
            signal = "ENTER"
        elif trading_score < 30:
            signal = "EXIT"

        # Memory: this dict materializes ~2M times across a 10y backtest
        # (911 tickers × 2316 dates). Each spare field carries ~80-120 B
        # of dict-slot + key-string overhead, so dropping 3 unused
        # diagnostic fields reclaims ~300 MB at peak. The full executor
        # surface consumes only ticker / signal / score / conviction /
        # sector / rating; technical_score / gbm_adjustment /
        # alpha_predicted were preserved historically as human-readable
        # diagnostics. Both upstream inputs (predictions, ohlcv,
        # scoring formula) are deterministic and persisted, so a
        # diagnostic field can be reconstructed offline from
        # predictor/predictions/{date}.json + ArcticDB universe if
        # needed. Stage 2 of the optimization arc.
        scored.append({
            "ticker": ticker,
            "score": trading_score,
            "signal": signal,
            "conviction": conviction,
            "sector": sector,
            "rating": "BUY" if signal == "ENTER" else ("SELL" if signal == "EXIT" else "HOLD"),
        })

    # Sort by score descending for top-N filtering
    scored.sort(key=lambda s: s["score"], reverse=True)

    # Cap ENTER signals at top_n
    enter_count = 0
    for s in scored:
        if s["signal"] == "ENTER":
            if enter_count >= top_n:
                s["signal"] = "HOLD"
                s["rating"] = "HOLD"
            else:
                enter_count += 1

    buy_candidates = [s for s in scored if s["signal"] == "ENTER"]
    universe = [s for s in scored if s["signal"] != "ENTER"]

    sector_ratings = {
        name: {"rating": "market_weight", "modifier": 1.0, "rationale": "synthetic"}
        for name in _ETF_TO_SECTOR.values()
    }

    return {
        "date": date,
        "market_regime": market_regime,
        "sector_ratings": sector_ratings,
        "buy_candidates": buy_candidates,
        "universe": universe,
    }


def _assign_conviction(alpha: float) -> str:
    if alpha >= 0.02:
        return "rising"
    elif alpha <= -0.01:
        return "declining"
    else:
        return "stable"
