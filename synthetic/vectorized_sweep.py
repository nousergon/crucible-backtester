"""Vectorized predictor-param-sweep orchestrator (Tier 4 PR 4, 2026-04-27).

Replaces the per-combo ``_run_simulation_loop`` × 60 with a single
matrix-axis simulation. All combos evaluate per-date as numpy
broadcasts; only the time loop is sequential (path dependency).

Composes Tier 4 PR 1-3 building blocks:

  * VectorizedSimulator (PR 1) — state matrices + apply_buy / apply_sell
  * compute_vectorized_exits (PR 2) — strategy-exit cascade
  * compute_vectorized_entries (PR 3) — entry pipeline + sizing

Plus this module's job:

  * Materialize feature matrices ``[n_dates, n_tickers]`` from
    ``FeatureLookup`` ONCE at sweep entry (eliminates per-date asof
    lookups).
  * Translate per-combo strategy_config dicts to per-combo numpy arrays
    of VectorizedExitConfig + VectorizedEntryConfig.
  * Translate per-date signals (signal_lookup) to per-signal arrays
    expected by compute_vectorized_entries.
  * Translate per-date research actions to ``[n_tickers]`` int codes
    consumed by compute_vectorized_exits.
  * Per-date correlation matrix (single np.corrcoef call).
  * Accumulate per-combo orders lists for downstream
    ``vectorbt_bridge.orders_to_portfolio`` portfolio simulation.

Output contract: a list of per-combo order lists, one list per combo,
each list shape-compatible with the existing scalar
``_run_simulation_loop`` order accumulator. The caller pipes those
through ``orders_to_portfolio`` + ``compute_portfolio_stats`` to build
the same sweep_df DataFrame the scalar path produces.

Wiring posture
--------------
This module ships behind a config flag (``use_vectorized_sweep``). The
scalar path remains the default for v13/v14 dispatch until end-to-end
parity is validated on real spot data. PR 5 flips the default and
retightens the predictor_pipeline cap.
"""

from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from synthetic.vectorized_entries import (
    BLOCK_NONE,
    CONV_DECLINING,
    CONV_RISING,
    CONV_STABLE,
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_CAUTION,
    REGIME_NEUTRAL,
    SIZING_ARM_CONVICTION,
    SIZING_ARM_FRACTIONAL_KELLY,
    SIZING_ARM_RISK_PARITY,
    SR_MARKET_WEIGHT,
    SR_OVERWEIGHT,
    SR_UNDERWEIGHT,
    EntryDecisions,
    VectorizedEntryConfig,
    apply_vectorized_entries,
    compute_correlation_matrix,
    compute_realized_vol_20d,
    compute_vectorized_entries,
)
from synthetic.vectorized_exits import (
    ACTION_EXIT,
    ACTION_REDUCE,
    RA_ENTER,
    RA_EXIT,
    RA_HOLD,
    RA_REDUCE,
    REASON_ATR,
    REASON_FALLBACK,
    REASON_MOMENTUM,
    REASON_PROFIT,
    REASON_TIME_EXIT,
    REASON_TIME_REDUCE,
    VectorizedExitConfig,
    apply_vectorized_exits,
    compute_vectorized_exits,
)
from synthetic.vectorized_sim import VectorizedSimulator

logger = logging.getLogger(__name__)


# ── Translation tables ──────────────────────────────────────────────


_SECTOR_RATING_CODE = {
    "overweight": SR_OVERWEIGHT,
    "market_weight": SR_MARKET_WEIGHT,
    "underweight": SR_UNDERWEIGHT,
}

_CONVICTION_CODE = {
    "rising": CONV_RISING,
    "stable": CONV_STABLE,
    "declining": CONV_DECLINING,
}

# Market regime codes. 3-class Ang-Bekaert macro vocabulary post v0.42.0
# (caution-regime-retirement-260528.md). The legacy "caution" → REGIME_CAUTION
# mapping is grandfathered for replay over historical signals.json
# artifacts (their market_regime field carried "caution" prior to the
# retirement); new emissions from the macro-agent are 3-class only,
# and the drawdown leg's protective hysteresis (risk_on/caution/risk_off)
# is a SEPARATE axis read via the drawdown_protective_severity ordinal
# in the predictor regime substrate JSON.
_REGIME_CODE = {
    "bull": REGIME_BULL,
    "neutral": REGIME_NEUTRAL,
    "bear": REGIME_BEAR,
    "caution": REGIME_CAUTION,  # legacy grandfather for historical replay
}

# Default sector ETF map (mirrors executor.strategies.exit_manager.SECTOR_ETF_MAP)
DEFAULT_SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financial": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
    "Industrials": "XLI",
    "Communication Services": "XLC",
}


# ── Feature matrix builder ─────────────────────────────────────────


def build_feature_matrices(
    feature_lookup,
    ticker_to_idx: dict,
    dates: pd.DatetimeIndex,
) -> dict:
    """Reindex per-ticker FeatureLookup Series to [n_dates, n_tickers] matrices.

    Uses ``reindex(method="ffill")`` so a query at date d retrieves the
    most recent computed value at-or-before d (matches scalar
    ``Series.asof`` semantics used by FeatureLookup).
    """
    n_tickers = len(ticker_to_idx)
    n_dates = len(dates)
    out = {}
    for name, store in (
        ("atr_dollar", feature_lookup.atr_dollar),
        ("rsi", feature_lookup.rsi),
        ("momentum_20d_pct", feature_lookup.momentum_20d_pct),
        ("returns", feature_lookup.returns),
    ):
        mat = np.full((n_dates, n_tickers), np.nan, dtype=np.float64)
        for ticker, t_idx in ticker_to_idx.items():
            s = store.get(ticker)
            if s is None:
                continue
            aligned = s.reindex(dates, method="ffill")
            mat[:, t_idx] = aligned.to_numpy(dtype=np.float64)
        out[name] = mat
    return out


def build_lookback_return_matrix(
    price_matrix: pd.DataFrame, lookback: int = 20,
) -> np.ndarray:
    """Per-(date, ticker) lookback return = price[d] / price[d-lookback] - 1.

    Used by:
      * Sector-relative veto (compute_vectorized_exits)
    Returns ``[n_dates, n_tickers]`` with NaN for the first ``lookback``
    rows.
    """
    arr = price_matrix.to_numpy(dtype=np.float64)
    n_dates, n_tickers = arr.shape
    out = np.full((n_dates, n_tickers), np.nan, dtype=np.float64)
    if n_dates > lookback:
        prior = arr[:-lookback, :]
        out[lookback:, :] = arr[lookback:, :] / np.where(prior > 0, prior, np.nan) - 1.0
    return out


def build_sector_arrays(
    ticker_to_idx: dict,
    sector_map: dict,
    sector_etf_map: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build sector arrays for vectorized exit/entry gates.

    Returns
    -------
    sector_idx_per_ticker : int32[n_tickers]
        Sector-axis index per ticker. ``-1`` for tickers without a
        known sector.
    sector_etf_ticker_idx : int32[n_sectors]
        For each sector index, the ticker_idx of that sector's ETF in
        the simulator's ticker space. ``-1`` if the ETF isn't in the
        ticker_index (in which case the sector-relative veto cannot
        fire for that sector).
    sector_label_to_idx : dict[str, int]
        Sector-name → index map. Only includes sectors that actually
        appear in ``sector_map`` plus any extra sectors implied by
        ``sector_etf_map``.
    """
    if sector_etf_map is None:
        sector_etf_map = DEFAULT_SECTOR_ETF_MAP

    n_tickers = len(ticker_to_idx)
    sector_labels = sorted({s for s in sector_map.values() if s} | set(sector_etf_map.keys()))
    sector_label_to_idx = {label: i for i, label in enumerate(sector_labels)}

    sector_idx_per_ticker = np.full(n_tickers, -1, dtype=np.int32)
    for ticker, t_idx in ticker_to_idx.items():
        label = sector_map.get(ticker)
        if label and label in sector_label_to_idx:
            sector_idx_per_ticker[t_idx] = sector_label_to_idx[label]

    n_sectors = len(sector_labels)
    sector_etf_ticker_idx = np.full(n_sectors, -1, dtype=np.int32)
    for label, etf in sector_etf_map.items():
        if label not in sector_label_to_idx:
            continue
        sec_idx = sector_label_to_idx[label]
        if etf in ticker_to_idx:
            sector_etf_ticker_idx[sec_idx] = ticker_to_idx[etf]

    return sector_idx_per_ticker, sector_etf_ticker_idx, sector_label_to_idx


# ── Combo config builder ────────────────────────────────────────────


def _get_strategy_param(combo_cfg: dict, key: str, default):
    """Read a strategy param from a combo config dict.

    Combo configs from the sweep may carry strategy params either at
    top-level (flat) or nested under ``strategy.exit_manager``. The
    sweep's ``_PARAM_MAP`` translation pushes flat keys into nested
    paths but doesn't always remove the flat key, so we read either.
    """
    nested = (
        combo_cfg.get("strategy", {})
        .get("exit_manager", {})
    )
    if key in nested:
        return nested[key]
    if key in combo_cfg:
        return combo_cfg[key]
    return default


def build_combo_configs(
    combo_configs: list,
) -> tuple[VectorizedExitConfig, VectorizedEntryConfig]:
    """Flatten a list of combo dicts to vectorized config arrays.

    Each combo's strategy_config + risk params translate to
    ``[n_combos]`` arrays for the exit + entry pipelines. Defaults
    match the scalar fallbacks in
    ``executor.strategies.config.load_strategy_config`` and
    ``executor.position_sizer.compute_position_size``.
    """
    n = len(combo_configs)

    # ── Exit config ────────────────────────────────────────────────
    exit_cfg = VectorizedExitConfig(
        atr_trailing_enabled=np.array([
            _get_strategy_param(c, "atr_trailing_enabled", True)
            for c in combo_configs
        ], dtype=bool),
        fallback_stop_enabled=np.array([
            _get_strategy_param(c, "fallback_stop_enabled", True)
            for c in combo_configs
        ], dtype=bool),
        profit_take_enabled=np.array([
            _get_strategy_param(c, "profit_take_enabled", True)
            for c in combo_configs
        ], dtype=bool),
        momentum_exit_enabled=np.array([
            _get_strategy_param(c, "momentum_exit_enabled", True)
            for c in combo_configs
        ], dtype=bool),
        time_decay_enabled=np.array([
            _get_strategy_param(c, "time_decay_enabled", True)
            for c in combo_configs
        ], dtype=bool),
        sector_relative_veto_enabled=np.array([
            _get_strategy_param(c, "sector_relative_veto_enabled", True)
            for c in combo_configs
        ], dtype=bool),
        atr_multiplier=np.array([
            _get_strategy_param(c, "atr_multiplier", 2.5) for c in combo_configs
        ], dtype=np.float64),
        fallback_stop_pct=np.array([
            _get_strategy_param(c, "fallback_stop_pct", 0.10) for c in combo_configs
        ], dtype=np.float64),
        profit_take_pct=np.array([
            _get_strategy_param(c, "profit_take_pct", 0.25) for c in combo_configs
        ], dtype=np.float64),
        momentum_exit_threshold=np.array([
            _get_strategy_param(c, "momentum_exit_threshold", -15.0)
            for c in combo_configs
        ], dtype=np.float64),
        momentum_exit_rsi=np.array([
            _get_strategy_param(c, "momentum_exit_rsi", 30) for c in combo_configs
        ], dtype=np.float64),
        time_decay_reduce_days=np.array([
            int(_get_strategy_param(c, "time_decay_reduce_days", 7))
            for c in combo_configs
        ], dtype=np.int32),
        time_decay_exit_days=np.array([
            int(_get_strategy_param(c, "time_decay_exit_days", 14))
            for c in combo_configs
        ], dtype=np.int32),
        sector_relative_outperform_threshold=np.array([
            _get_strategy_param(c, "sector_relative_outperform_threshold", 0.05)
            for c in combo_configs
        ], dtype=np.float64),
        # MAE hard floor (L4549a #238) — default enabled at -0.15, mirroring
        # the scalar ``check_position_loss_floor`` inline defaults so a combo
        # that doesn't set it still carries the falling-knife backstop.
        position_loss_floor_enabled=np.array([
            _get_strategy_param(c, "position_loss_floor_enabled", True)
            for c in combo_configs
        ], dtype=bool),
        position_loss_floor_pct=np.array([
            _get_strategy_param(c, "position_loss_floor_pct", -0.15)
            for c in combo_configs
        ], dtype=np.float64),
        reduce_fraction=np.array([
            c.get("reduce_fraction", 0.50) for c in combo_configs
        ], dtype=np.float64),
    )

    # ── Entry config ───────────────────────────────────────────────
    entry_cfg = VectorizedEntryConfig(
        min_score_to_enter=np.array([
            c.get("min_score_to_enter", c.get("min_score", 70.0))
            for c in combo_configs
        ], dtype=np.float64),
        momentum_gate_enabled=np.array([
            c.get("momentum_gate_enabled", True) for c in combo_configs
        ], dtype=bool),
        momentum_gate_threshold=np.array([
            c.get("momentum_gate_threshold", -5.0) for c in combo_configs
        ], dtype=np.float64),
        max_position_pct=np.array([
            c.get("max_position_pct", 0.05) for c in combo_configs
        ], dtype=np.float64),
        bear_max_position_pct=np.array([
            c.get("bear_max_position_pct", 0.025) for c in combo_configs
        ], dtype=np.float64),
        max_sector_pct=np.array([
            c.get("max_sector_pct", 0.25) for c in combo_configs
        ], dtype=np.float64),
        max_equity_pct=np.array([
            c.get("max_equity_pct", 0.90) for c in combo_configs
        ], dtype=np.float64),
        bear_block_underweight=np.array([
            c.get("bear_block_underweight", True) for c in combo_configs
        ], dtype=bool),
        sector_adj_overweight=np.array([
            c.get("sector_adj", {}).get("overweight", 1.05) for c in combo_configs
        ], dtype=np.float64),
        sector_adj_market_weight=np.array([
            c.get("sector_adj", {}).get("market_weight", 1.00) for c in combo_configs
        ], dtype=np.float64),
        sector_adj_underweight=np.array([
            c.get("sector_adj", {}).get("underweight", 0.85) for c in combo_configs
        ], dtype=np.float64),
        conviction_decline_adj=np.array([
            c.get("conviction_decline_adj", 0.70) for c in combo_configs
        ], dtype=np.float64),
        upside_fail_adj=np.array([
            c.get("upside_fail_adj", 0.70) for c in combo_configs
        ], dtype=np.float64),
        min_price_target_upside=np.array([
            c.get("min_price_target_upside", 0.05) for c in combo_configs
        ], dtype=np.float64),
        atr_sizing_enabled=np.array([
            c.get("atr_sizing_enabled", True) for c in combo_configs
        ], dtype=bool),
        atr_sizing_target_risk=np.array([
            c.get("atr_sizing_target_risk", 0.02) for c in combo_configs
        ], dtype=np.float64),
        atr_sizing_floor=np.full(n, 0.5, dtype=np.float64),
        atr_sizing_ceiling=np.full(n, 1.5, dtype=np.float64),
        confidence_sizing_enabled=np.array([
            c.get("confidence_sizing_enabled", True) for c in combo_configs
        ], dtype=bool),
        confidence_sizing_min=np.array([
            c.get("confidence_sizing_min", 0.7) for c in combo_configs
        ], dtype=np.float64),
        confidence_sizing_range=np.array([
            c.get("confidence_sizing_range", 0.6) for c in combo_configs
        ], dtype=np.float64),
        use_p_up_sizing=np.array([
            c.get("use_p_up_sizing", False) for c in combo_configs
        ], dtype=bool),
        p_up_sizing_blend=np.array([
            c.get("p_up_sizing_blend", 0.3) for c in combo_configs
        ], dtype=np.float64),
        staleness_discount_enabled=np.array([
            c.get("staleness_discount_enabled", True) for c in combo_configs
        ], dtype=bool),
        signal_cadence_days=np.array([
            int(c.get("signal_cadence_days", 7)) for c in combo_configs
        ], dtype=np.int32),
        staleness_decay_per_day=np.array([
            c.get("staleness_decay_per_day", 0.03) for c in combo_configs
        ], dtype=np.float64),
        staleness_floor=np.array([
            c.get("staleness_floor", 0.70) for c in combo_configs
        ], dtype=np.float64),
        earnings_sizing_enabled=np.array([
            c.get("earnings_sizing_enabled", True) for c in combo_configs
        ], dtype=bool),
        earnings_proximity_days=np.array([
            int(c.get("earnings_proximity_days", 5)) for c in combo_configs
        ], dtype=np.int32),
        earnings_sizing_reduction=np.array([
            c.get("earnings_sizing_reduction", 0.50) for c in combo_configs
        ], dtype=np.float64),
        coverage_sizing_enabled=np.array([
            c.get("coverage_sizing_enabled", True) for c in combo_configs
        ], dtype=bool),
        coverage_derate_floor=np.array([
            c.get("coverage_derate_floor", 0.25) for c in combo_configs
        ], dtype=np.float64),
        min_position_dollar=np.array([
            c.get("min_position_dollar", 500.0) for c in combo_configs
        ], dtype=np.float64),
        correlation_block_enabled=np.array([
            c.get("correlation_block_enabled", True) for c in combo_configs
        ], dtype=bool),
        correlation_block_threshold=np.array([
            c.get("correlation_block_threshold", 0.80) for c in combo_configs
        ], dtype=np.float64),
        correlation_lookback_days=np.array([
            int(c.get("correlation_lookback_days", 60)) for c in combo_configs
        ], dtype=np.int32),
        # Fractional-Kelly sizing arm (config#3081). Ignored unless the
        # sweep runs with sizing_arm="fractional_kelly"; default 0.25
        # matches VectorizedEntryConfig.from_uniform's default.
        kelly_fraction=np.array([
            c.get("kelly_fraction", 0.25) for c in combo_configs
        ], dtype=np.float64),
    )
    return exit_cfg, entry_cfg


# ── Per-date input extraction ──────────────────────────────────────


def _signal_to_codes(
    signal: dict, default_sector: str = "",
) -> tuple[int, int, float]:
    """Return (sector_rating_code, conviction_code, upside_or_nan)."""
    rating = (signal.get("sector_rating") or "market_weight").lower()
    rating_code = _SECTOR_RATING_CODE.get(rating, SR_MARKET_WEIGHT)
    conviction = (signal.get("conviction") or "stable").lower()
    conviction_code = _CONVICTION_CODE.get(conviction, CONV_STABLE)
    upside_raw = signal.get("price_target_upside")
    upside = float(upside_raw) if upside_raw is not None else float("nan")
    return rating_code, conviction_code, upside


def extract_signal_arrays(
    signal_lookup,
    predictions: dict,
    ticker_to_idx: dict,
    sector_label_to_idx: dict,
    atr_pct_by_ticker: dict,
    coverage_by_ticker: dict,
    earnings_by_ticker: dict,
    momentum_at_date_per_ticker: np.ndarray,
) -> dict:
    """Extract per-signal arrays for one date from a SignalLookup.

    Returns a dict with the kwargs ``compute_vectorized_entries``
    expects. Skips signals whose ticker isn't in ``ticker_to_idx``.

    Reads from ``signal_lookup.actionable`` — the post-
    ``get_actionable_signals`` transformation populated once per date by
    ``_build_signal_lookup``. The raw envelope from
    ``synthetic.signal_generator.predictions_to_signals`` carries
    ``buy_candidates`` + ``universe`` keys but NOT ``enter`` —
    historically reading ``signals_raw_filtered.get("enter")`` here
    silently returned ``[]`` and the vectorized sweep produced zero
    orders on the full 10y fixture (caught by Tier 4 Layer 3 v14
    parity vs scalar predictor_single_run on 2026-04-28).
    """
    enter_signals = signal_lookup.actionable.get("enter", [])
    valid: list[dict] = []
    for s in enter_signals:
        if not isinstance(s, dict):
            continue
        t = s.get("ticker")
        if not t or t not in ticker_to_idx:
            continue
        valid.append(s)

    n_signals = len(valid)
    if n_signals == 0:
        return {
            "signal_ticker_idx": np.zeros(0, dtype=np.int32),
            "signal_score": np.zeros(0, dtype=np.float64),
            "signal_sector_idx": np.zeros(0, dtype=np.int32),
            "signal_sector_rating": np.zeros(0, dtype=np.int8),
            "signal_conviction": np.zeros(0, dtype=np.int8),
            "signal_upside": np.zeros(0, dtype=np.float64),
            "signal_atr_pct": np.zeros(0, dtype=np.float64),
            "signal_pred_confidence": np.zeros(0, dtype=np.float64),
            "signal_p_up": np.zeros(0, dtype=np.float64),
            "signal_days_to_earnings": np.zeros(0, dtype=np.int32),
            "signal_feature_coverage": np.zeros(0, dtype=np.float64),
            "signal_gbm_veto": np.zeros(0, dtype=bool),
            "signal_momentum_at_date": np.zeros(0, dtype=np.float64),
            "tickers": [],
        }

    signal_ticker_idx = np.zeros(n_signals, dtype=np.int32)
    signal_score = np.zeros(n_signals, dtype=np.float64)
    signal_sector_idx = np.full(n_signals, -1, dtype=np.int32)
    signal_sector_rating = np.full(n_signals, SR_MARKET_WEIGHT, dtype=np.int8)
    signal_conviction = np.full(n_signals, CONV_STABLE, dtype=np.int8)
    signal_upside = np.full(n_signals, np.nan, dtype=np.float64)
    signal_atr_pct = np.full(n_signals, np.nan, dtype=np.float64)
    signal_pred_confidence = np.full(n_signals, np.nan, dtype=np.float64)
    signal_p_up = np.full(n_signals, np.nan, dtype=np.float64)
    signal_days_to_earnings = np.full(n_signals, -1, dtype=np.int32)
    signal_feature_coverage = np.full(n_signals, np.nan, dtype=np.float64)
    signal_gbm_veto = np.zeros(n_signals, dtype=bool)
    signal_momentum_at_date = np.full(n_signals, np.nan, dtype=np.float64)
    tickers: list[str] = []

    for i, s in enumerate(valid):
        ticker = s["ticker"]
        tickers.append(ticker)
        t_idx = ticker_to_idx[ticker]
        signal_ticker_idx[i] = t_idx
        signal_score[i] = float(s.get("score") or 0.0)
        sector_label = s.get("sector") or ""
        if sector_label in sector_label_to_idx:
            signal_sector_idx[i] = sector_label_to_idx[sector_label]
        rating_code, conv_code, upside = _signal_to_codes(s)
        signal_sector_rating[i] = rating_code
        signal_conviction[i] = conv_code
        signal_upside[i] = upside

        # Per-ticker auxiliary maps
        atr_pct = atr_pct_by_ticker.get(ticker)
        if atr_pct is not None:
            signal_atr_pct[i] = float(atr_pct)
        cov = coverage_by_ticker.get(ticker)
        if cov is not None:
            signal_feature_coverage[i] = float(cov)
        de = earnings_by_ticker.get(ticker)
        if de is not None:
            signal_days_to_earnings[i] = int(de)

        # Predictions
        pred = predictions.get(ticker, {})
        conf = pred.get("prediction_confidence")
        if conf is not None:
            signal_pred_confidence[i] = float(conf)
        p_up = pred.get("p_up")
        if p_up is not None:
            signal_p_up[i] = float(p_up)
        signal_gbm_veto[i] = bool(pred.get("gbm_veto", False))

        # Momentum from precomputed matrix
        signal_momentum_at_date[i] = momentum_at_date_per_ticker[t_idx]

    return {
        "signal_ticker_idx": signal_ticker_idx,
        "signal_score": signal_score,
        "signal_sector_idx": signal_sector_idx,
        "signal_sector_rating": signal_sector_rating,
        "signal_conviction": signal_conviction,
        "signal_upside": signal_upside,
        "signal_atr_pct": signal_atr_pct,
        "signal_pred_confidence": signal_pred_confidence,
        "signal_p_up": signal_p_up,
        "signal_days_to_earnings": signal_days_to_earnings,
        "signal_feature_coverage": signal_feature_coverage,
        "signal_gbm_veto": signal_gbm_veto,
        "signal_momentum_at_date": signal_momentum_at_date,
        "tickers": tickers,
    }


def extract_research_actions(
    signal_lookup, ticker_to_idx: dict, n_tickers: int,
) -> np.ndarray:
    """Build a [n_tickers] int8 array of research-action codes for one date.

    HOLD is the default; entries from ``enter`` / ``exit`` / ``reduce``
    lists override at their ticker positions.

    Reads from ``signal_lookup.actionable`` for the same reason as
    ``extract_signal_arrays``: the synthetic envelope shape doesn't
    pre-segment by signal. See the docstring there for the full bug
    history.
    """
    actions = np.full(n_tickers, RA_HOLD, dtype=np.int8)
    raw = signal_lookup.actionable

    def _set(field: str, code: int) -> None:
        for s in raw.get(field, []):
            if not isinstance(s, dict):
                continue
            t = s.get("ticker")
            if t and t in ticker_to_idx:
                actions[ticker_to_idx[t]] = code

    _set("enter", RA_ENTER)
    _set("exit", RA_EXIT)
    _set("reduce", RA_REDUCE)
    return actions


# ── Drawdown multiplier (per-combo from config) ────────────────────


def compute_dd_multiplier(
    sim: VectorizedSimulator,
    circuit_breaker_per_combo: np.ndarray,
    tiers: list,
) -> np.ndarray:
    """Wrapper for ``sim.drawdown_multiplier`` matching scalar tier semantics."""
    return sim.drawdown_multiplier(circuit_breaker_per_combo, tiers)


# ── Order recording ────────────────────────────────────────────────


def _exit_reason_str(code: int) -> str:
    return {
        REASON_ATR: "atr_trailing_stop",
        REASON_FALLBACK: "fallback_stop",
        REASON_PROFIT: "profit_take",
        REASON_MOMENTUM: "momentum_exit",
        REASON_TIME_EXIT: "time_decay_exit",
        REASON_TIME_REDUCE: "time_decay_reduce",
    }.get(code, "")


# ── Main orchestrator ──────────────────────────────────────────────


def run_vectorized_sweep(
    *,
    combo_configs: list,
    price_matrix: pd.DataFrame,
    ohlcv_by_ticker: dict,
    signal_lookups: dict,
    feature_lookup,
    spy_prices: pd.Series | None,
    sector_map: dict,
    sector_etf_map: dict | None = None,
    atr_pct_by_ticker: dict | None = None,
    coverage_by_ticker: dict | None = None,
    earnings_by_ticker: dict | None = None,
    predictions_by_date: dict | None = None,
    init_cash: float = 1_000_000.0,
    drawdown_circuit_breaker: float = 0.08,
    drawdown_tiers: list | None = None,
    market_regime: str = "neutral",
    fee_rate: float = 0.0,
    sizing_arm: str = SIZING_ARM_CONVICTION,
) -> tuple[list, dict]:
    """Run all combos in parallel via VectorizedSimulator.

    ``sizing_arm`` (config#3081 S-slot sizing shootout) selects which
    raw-weight sizing formula ``compute_vectorized_entries`` uses for
    every date in this sweep — see
    ``synthetic.vectorized_entries.compute_vectorized_entries`` module
    docstring for the "conviction" (default, unchanged incumbent) /
    "risk_parity" / "fractional_kelly" arms. When the arm is not
    "conviction", this function computes a ``[n_dates, n_tickers]``
    trailing-20d realized-vol matrix from ``returns_mat`` (already
    materialized below for the ATR/momentum features) via
    ``compute_realized_vol_20d`` and slices it per date/signal; for
    "fractional_kelly" it also derives a per-signal predicted-alpha
    proxy from ``signal_upside`` (price-target upside — the most
    defensible existing per-signal "predicted alpha" already threaded
    through ``extract_signal_arrays``; see
    ``synthetic.sizing_shootout`` / ``run_sizing_shootout`` docstring
    for the full rationale). This does NOT change behavior for the
    default "conviction" arm — the vol matrix and alpha proxy are only
    computed and passed through when a caller opts into another arm.

    Returns
    -------
    orders_per_combo : list[list[dict]]
        Per-combo order lists ready for ``orders_to_portfolio``. Order
        dicts carry ``date / ticker / action / shares / price_at_order /
        portfolio_nav_at_order / position_pct / exit_reason``.
    diagnostics : dict
        Per-sweep telemetry (n_combos, n_dates, walltime, action counts).
    """
    if drawdown_tiers is None:
        drawdown_tiers = [
            (-0.02, 1.00),
            (-0.04, 0.50),
            (-0.06, 0.25),
        ]

    # Strip the description column if present (load_strategy_config returns
    # 3-tuples in some paths; we accept both for resilience).
    drawdown_tiers = [
        (float(t[0]), float(t[1])) for t in drawdown_tiers
    ]

    n_combos = len(combo_configs)
    if n_combos == 0:
        return [], {"n_combos": 0, "n_dates": 0}

    # Build ticker index from price_matrix columns.
    tickers: list[str] = list(price_matrix.columns)
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    n_tickers = len(tickers)

    # Sector mapping
    sector_idx_per_ticker, sector_etf_ticker_idx, sector_label_to_idx = (
        build_sector_arrays(ticker_to_idx, sector_map, sector_etf_map)
    )

    # Build vectorized configs from per-combo dicts
    exit_cfg, entry_cfg = build_combo_configs(combo_configs)

    # Per-combo circuit breaker (overrideable per combo)
    circuit_breaker_per_combo = np.array([
        c.get("drawdown_circuit_breaker", drawdown_circuit_breaker)
        for c in combo_configs
    ], dtype=np.float64)

    # Materialize feature matrices
    dates = price_matrix.index
    n_dates = len(dates)
    t_setup = _time.monotonic()
    feature_mats = build_feature_matrices(feature_lookup, ticker_to_idx, dates)
    atr_mat = feature_mats["atr_dollar"]            # [n_dates, n_tickers]
    rsi_mat = feature_mats["rsi"]
    momentum_mat = feature_mats["momentum_20d_pct"]
    returns_mat = feature_mats["returns"]
    lookback_return_mat = build_lookback_return_matrix(
        price_matrix, lookback=20,
    )
    logger.info(
        "vectorized_sweep: feature matrices built (%.1fs) — %d dates × %d tickers",
        _time.monotonic() - t_setup, n_dates, n_tickers,
    )

    # Trailing-20d realized-vol matrix (config#3081) — only computed when
    # a non-incumbent sizing arm is selected, since the incumbent
    # ("conviction") path never reads it. Built once per sweep via a
    # fully vectorized rolling window (numpy sliding_window_view — no
    # per-date Python loop): row d's vol uses returns[d-19:d+1] per
    # ticker (trailing 20 trading days INCLUDING date d, consistent
    # with atr_mat/momentum_mat being "as of date d" quantities
    # elsewhere in this loop). The first 19 rows have <20 trading days
    # of history and are left NaN (matches momentum_mat's own
    # first-N-rows-NaN convention above) — the sizing-arm helpers in
    # vectorized_entries.py fall back sensibly (equal-weight /
    # zero-Kelly) for NaN vol.
    realized_vol_mat = None
    if sizing_arm != SIZING_ARM_CONVICTION:
        realized_vol_mat = np.full((n_dates, n_tickers), np.nan, dtype=np.float64)
        if n_dates >= 20:
            windows = np.lib.stride_tricks.sliding_window_view(
                returns_mat, window_shape=20, axis=0,
            )  # [n_dates - 19, n_tickers, 20]
            # compute_realized_vol_20d expects [n_rows, >=20] with rows
            # as the "signal" axis — reshape to put (date, ticker) pairs
            # on axis 0, the 20-day window on axis 1.
            n_windows = windows.shape[0]
            flat = windows.reshape(n_windows * n_tickers, 20)
            vol_flat = compute_realized_vol_20d(flat)
            realized_vol_mat[19:, :] = vol_flat.reshape(n_windows, n_tickers)

    # Initialize simulator. fee_rate flows from production config
    # (`simulation_fees`, default 0.001) to mirror scalar single_run's
    # vectorbt fee semantics. Default 0.0 in this signature preserves
    # zero-fee accounting for unit-test fixtures that don't pass a rate.
    sim = VectorizedSimulator(
        n_combos=n_combos, ticker_index=ticker_to_idx,
        init_cash=init_cash, fee_rate=fee_rate,
    )

    # Sweep market regime to int code (single value per sweep — current
    # design assumes regime is constant across the sweep window, matching
    # how the scalar path treats it via signals_raw["market_regime"]).
    regime_code = _REGIME_CODE.get(market_regime.lower(), REGIME_NEUTRAL)

    # Aux per-ticker maps
    atr_pct_by_ticker = atr_pct_by_ticker or {}
    coverage_by_ticker = coverage_by_ticker or {}
    earnings_by_ticker = earnings_by_ticker or {}
    predictions_by_date = predictions_by_date or {}

    # Per-combo order accumulators — columnar storage to bound memory.
    # Replaces the prior `list[list[dict]]` (~300 B/order × 1.49M orders
    # ≈ 450 MB on a typical 60-combo × 2500-date × 907-ticker run, which
    # OOM-killed v15 on c5.large 2026-04-28). Columnar storage drops
    # this to ~90 MB at sweep end + ~7 MB peak materialization per combo.
    from synthetic.vectorized_orders import VectorizedOrderStore
    orders_per_combo = VectorizedOrderStore(n_combos)

    # Correlation lookback (max across combos — caller must precompute
    # returns history >= max). For initial implementation use a uniform
    # 60-bar window.
    correlation_lookback = int(np.max(entry_cfg.correlation_lookback_days))

    n_exits_total = 0
    n_entries_total = 0
    t_loop = _time.monotonic()

    prices_arr = price_matrix.to_numpy(dtype=np.float64)

    # Per-combo NAV trajectory. Recorded each date AFTER `sim.update_nav`
    # (so it reflects mark-to-market on yesterday's prices, before
    # today's exits/entries fire). Drives the post-loop stats compute
    # in `synthetic.vectorized_stats` — replaces the per-combo
    # `vectorbt.Portfolio.from_orders` + `sharpe_ratio()` path that
    # hung the v16 (2026-04-28) Layer 3 dispatch (60 combos × 26k orders
    # each × 6 vectorbt stat calls = >90 min, watchdog tripped).
    # Memory: 60 × 2500 × 8 bytes = 1.2 MB. Trivial.
    nav_history = np.zeros((n_combos, n_dates), dtype=np.float64)

    for date_idx in range(n_dates):
        date_ts = dates[date_idx]
        date_str = date_ts.strftime("%Y-%m-%d")
        signal_lookup = signal_lookups.get(date_str)

        prices = prices_arr[date_idx]

        # Build highs vector for this date from ohlcv_by_ticker.
        # Falls back to close prices if highs aren't available — caller
        # convention: the simulator uses highs for highest_high tracking
        # which feeds ATR trailing stops.
        highs = np.copy(prices)
        if ohlcv_by_ticker:
            for t, t_idx in ticker_to_idx.items():
                df = ohlcv_by_ticker.get(t)
                if df is None or "high" not in df.columns:
                    continue
                if date_ts in df.index:
                    h = df.loc[date_ts, "high"]
                    if pd.notna(h):
                        highs[t_idx] = float(h)

        sim.update_nav(prices)
        sim.update_highest_high(highs)

        # Snapshot post-MTM NAV for every combo at this date. .copy()
        # is critical — sim.nav is mutated in place by subsequent
        # update_nav calls, so a view assignment would leave every
        # date_idx pointing at the final-date NAV.
        nav_history[:, date_idx] = sim.nav

        if signal_lookup is None:
            continue

        # Drawdown multiplier per combo
        dd_mult = compute_dd_multiplier(
            sim, circuit_breaker_per_combo, drawdown_tiers,
        )

        # Per-ticker research actions
        research_actions = extract_research_actions(
            signal_lookup, ticker_to_idx, n_tickers,
        )

        # ── Exits ──────────────────────────────────────────────────
        atr_at = atr_mat[date_idx]
        rsi_at = rsi_mat[date_idx]
        mom_at = momentum_mat[date_idx]
        sec_lookback_ret = lookback_return_mat[date_idx]

        nav_before = sim.nav.copy()

        exit_decisions = compute_vectorized_exits(
            sim,
            prices=prices,
            atr_dollar_at_date=atr_at,
            rsi_at_date=rsi_at,
            momentum_at_date=mom_at,
            sector_lookback_return=sec_lookback_ret,
            research_action_per_ticker=research_actions,
            sector_idx_per_ticker=sector_idx_per_ticker,
            sector_etf_ticker_idx=sector_etf_ticker_idx,
            date_idx=date_idx,
            config=exit_cfg,
        )

        # Record exit orders BEFORE applying (we need positions / avg_costs
        # for sizing — apply_sell would mutate them). Columnar
        # accumulator: integer date_idx + ticker_idx + reason_code, no
        # dict allocation per order. Final string materialization happens
        # at consumer-time per combo via VectorizedOrderStore.__getitem__.
        if np.any(exit_decisions.exit_action != 0):
            ec, et = np.nonzero(exit_decisions.exit_action != 0)
            for c, t in zip(ec, et):
                action_code = int(exit_decisions.exit_action[c, t])
                shares = float(exit_decisions.exit_shares[c, t])
                if shares <= 0:
                    continue
                t_idx = int(t)
                c_idx = int(c)
                orders_per_combo.add_exit(
                    combo_idx=c_idx,
                    date_idx=date_idx,
                    ticker_idx=t_idx,
                    action_code=action_code,
                    shares=int(shares),
                    price=float(prices[t_idx]),
                    nav=float(nav_before[c_idx]),
                    reason_code=int(exit_decisions.exit_reason[c, t]),
                )
            n_exits_total += int(ec.size)

        apply_vectorized_exits(sim, exit_decisions, prices)

        # ── Entries ────────────────────────────────────────────────
        # Precompute correlation matrix from returns_mat slice.
        # Lookback window of `correlation_lookback` bars ending at date_idx-1
        # (don't include today's return since we're deciding on today's open).
        corr_matrix = None
        if (
            correlation_lookback > 0
            and date_idx >= correlation_lookback
            and np.any(entry_cfg.correlation_block_enabled)
        ):
            window_start = date_idx - correlation_lookback
            window = returns_mat[window_start:date_idx, :]  # [lookback, n_tickers]
            corr_matrix = compute_correlation_matrix(window.T)  # transpose → [n_tickers, lookback]

        predictions = predictions_by_date.get(date_str, {})

        sig_arrays = extract_signal_arrays(
            signal_lookup,
            predictions=predictions,
            ticker_to_idx=ticker_to_idx,
            sector_label_to_idx=sector_label_to_idx,
            atr_pct_by_ticker=atr_pct_by_ticker,
            coverage_by_ticker=coverage_by_ticker,
            earnings_by_ticker=earnings_by_ticker,
            momentum_at_date_per_ticker=mom_at,
        )
        signal_tickers_list = sig_arrays.pop("tickers")

        n_signals = sig_arrays["signal_ticker_idx"].shape[0]
        if n_signals == 0:
            continue

        # Per-signal realized-vol / alpha-proxy slices for the
        # risk_parity / fractional_kelly sizing arms (config#3081).
        # None for the incumbent arm (unused by compute_vectorized_entries
        # in that branch). alpha proxy = signal_upside (price-target
        # upside) — the most defensible per-signal "predicted alpha"
        # already threaded through extract_signal_arrays; see
        # run_sizing_shootout's docstring for the full rationale.
        sig_realized_vol_20d = None
        sig_alpha = None
        if sizing_arm != SIZING_ARM_CONVICTION:
            sig_realized_vol_20d = realized_vol_mat[date_idx, sig_arrays["signal_ticker_idx"]]
            if sizing_arm == SIZING_ARM_FRACTIONAL_KELLY:
                sig_alpha = sig_arrays["signal_upside"]

        # Signal age (calendar days between signals_raw['date'] and run_date)
        signal_date_str = signal_lookup.signals_raw_filtered.get("date", date_str)
        try:
            signal_age_days = (
                pd.Timestamp(date_str) - pd.Timestamp(signal_date_str)
            ).days
        except (ValueError, TypeError):
            signal_age_days = 0

        nav_before_entries = sim.nav.copy()

        entry_decisions = compute_vectorized_entries(
            sim,
            **sig_arrays,
            prices=prices,
            nav_per_combo=sim.nav,
            dd_multiplier_per_combo=dd_mult,
            market_regime=regime_code,
            signal_age_days=int(signal_age_days),
            config=entry_cfg,
            correlation_matrix=corr_matrix,
            sector_idx_per_ticker=sector_idx_per_ticker,
            sizing_arm=sizing_arm,
            signal_realized_vol_20d=sig_realized_vol_20d,
            signal_alpha=sig_alpha,
        )

        # Record entry orders before applying — columnar accumulator,
        # see exits-side comment above.
        if np.any(entry_decisions.entry_passed):
            ec, es = np.nonzero(entry_decisions.entry_passed)
            for c, s_idx in zip(ec, es):
                shares = float(entry_decisions.entry_shares[c, s_idx])
                if shares <= 0:
                    continue
                s_int = int(s_idx)
                ticker = signal_tickers_list[s_int]
                t_idx = ticker_to_idx[ticker]
                c_idx = int(c)
                nav_at_order = float(nav_before_entries[c_idx])
                position_pct = (
                    float(entry_decisions.entry_dollar[c, s_idx]) / nav_at_order
                    if nav_at_order > 0 else 0.0
                )
                orders_per_combo.add_entry(
                    combo_idx=c_idx,
                    date_idx=date_idx,
                    ticker_idx=t_idx,
                    shares=int(shares),
                    price=float(prices[t_idx]),
                    nav=nav_at_order,
                    position_pct=position_pct,
                )
            n_entries_total += int(ec.size)

        apply_vectorized_entries(
            sim, entry_decisions,
            signal_ticker_idx=sig_arrays["signal_ticker_idx"],
            prices=prices, date_idx=date_idx,
        )

    walltime = _time.monotonic() - t_loop
    logger.info(
        "vectorized_sweep: %d combos × %d dates in %.1fs "
        "(entries=%d, exits=%d)",
        n_combos, n_dates, walltime, n_entries_total, n_exits_total,
    )

    # Provide the lookups consumers need to materialize per-combo
    # dict-lists. Producer-side: the tickers list and price_matrix.index
    # are stable across the sweep, so handing them to the store now
    # gives __getitem__ everything it needs.
    orders_per_combo.finalize(price_matrix.index, tickers)

    diagnostics = {
        "n_combos": n_combos,
        "n_dates": n_dates,
        "walltime_sec": walltime,
        "entries_applied": n_entries_total,
        "exits_applied": n_exits_total,
        "nav_history": nav_history,  # [n_combos, n_dates], for stats
    }
    return orders_per_combo, diagnostics


# ── S-slot sizing shootout (config#3081) ────────────────────────────


# Kelly fractions swept by default — at least 2 distinct values so
# "swept as a parameter" is genuinely honored, not one hardcoded
# fraction (config#3081 requirement). 0.25 / 0.5 are the conventional
# "quarter-Kelly" / "half-Kelly" realism anchors.
DEFAULT_KELLY_FRACTIONS: tuple[float, ...] = (0.25, 0.5)


def _arm_label(arm: str, kelly_fraction: float | None = None) -> str:
    """Canonical per-arm result key.

    "conviction" / "risk_parity" as-is; fractional_kelly expands to
    ``fractional_kelly_{fraction}`` (e.g. ``fractional_kelly_0.25``)
    per combo-slot so each swept fraction is independently addressable
    in the shootout's results dict. Naming convention documented here
    (config#3081) — the only place arm-name strings are constructed.
    """
    if arm == SIZING_ARM_FRACTIONAL_KELLY:
        if kelly_fraction is None:
            raise ValueError("fractional_kelly arm requires a kelly_fraction")
        return f"{SIZING_ARM_FRACTIONAL_KELLY}_{kelly_fraction:g}"
    return arm


def run_sizing_shootout(
    *,
    combo_configs: list,
    arms: tuple[str, ...] = (
        SIZING_ARM_CONVICTION, SIZING_ARM_RISK_PARITY, SIZING_ARM_FRACTIONAL_KELLY,
    ),
    kelly_fractions: tuple[float, ...] = DEFAULT_KELLY_FRACTIONS,
    **sweep_kwargs,
) -> dict[str, tuple]:
    """Run the SAME signal stream through 2-4 sizing arms for comparison.

    config#3081 "S-slot sizing shootout": reuses ``run_vectorized_sweep``
    unchanged (no new orchestration framework) — this is a thin fan-out
    wrapper that calls it once per arm with the SAME base combo config
    (price_matrix / signal_lookups / feature_lookup / sector_map /
    fee_rate / cash policy / caps — everything except the sizing-arm
    selector and, for Kelly, the fraction) so the universe/exposure
    constraints (sector caps, position caps, cash policy) are IDENTICAL
    across arms and only the raw-weight sizing formula differs — the
    apples-to-apples comparison the issue asks for.

    Each ``combo_configs`` entry is used as the BASE config for every
    arm; this function does not fan out over additional param-grid
    axes itself (a caller wanting a param-grid comparison per arm can
    still pass a multi-combo ``combo_configs`` list — each combo is
    carried through unmodified to every arm, so a 3-combo grid produces
    3-combo results per arm, all directly comparable index-for-index).

    ``fee_rate`` (and every other ``run_vectorized_sweep`` kwarg) is
    forwarded verbatim via ``**sweep_kwargs`` to EVERY arm's call — so
    all arms are scored fee-aware (or fee-free) identically; a caller
    cannot accidentally run one arm with costs and another without,
    which the issue calls out as an invalid comparison (a cost-free
    Kelly "win" is not evidence of anything).

    Returns
    -------
    dict[str, tuple[VectorizedOrderStore, dict]]
        Keyed by arm label (``_arm_label``): "conviction", "risk_parity",
        "fractional_kelly_0.25", "fractional_kelly_0.5", ... — each
        value is the same ``(orders_per_combo, diagnostics)`` tuple
        ``run_vectorized_sweep`` returns for that arm.
    """
    results: dict[str, tuple] = {}
    for arm in arms:
        if arm == SIZING_ARM_FRACTIONAL_KELLY:
            for frac in kelly_fractions:
                frac_combos = [
                    {**c, "kelly_fraction": frac} for c in combo_configs
                ]
                label = _arm_label(arm, frac)
                results[label] = run_vectorized_sweep(
                    combo_configs=frac_combos,
                    sizing_arm=SIZING_ARM_FRACTIONAL_KELLY,
                    **sweep_kwargs,
                )
        else:
            label = _arm_label(arm)
            results[label] = run_vectorized_sweep(
                combo_configs=combo_configs,
                sizing_arm=arm,
                **sweep_kwargs,
            )
    return results
