"""Tests for analysis/regime_stratified_sharpe.py — Stage C.2 T2.

Covers:
- DB loader resilience (missing market_regime column → pre-migration fallback)
- Annualized Sharpe formula on per-pick alphas (mean/std × sqrt(periods/year))
- Per-(regime, horizon) stratum metrics: n_picks, mean alpha, Sharpe, hit rate
- Min-sample gate (n_picks below threshold → None metrics)
- Headline spread metric (bull - bear at 10d horizon) + interpretation flags
- Eval-artifact payload assembly
"""
from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


from analysis.regime_stratified_sharpe import (
    DEFAULT_MIN_PICKS_PER_STRATUM,
    SUPPORTED_HORIZONS,
    StratumMetrics,
    _annualized_sharpe_from_alphas,
    assemble_t2_eval_payload,
    compute_regime_spread,
    load_with_subscores_and_regime,
    stratified_sharpe_by_regime,
)


# ---------------------------------------------------------------------------
# DB loader resilience
# ---------------------------------------------------------------------------


def _create_score_perf_db(
    path: Path,
    *,
    rows: list[dict],
    with_regime_column: bool = True,
) -> None:
    """Create a synthetic score_performance SQLite table."""
    conn = sqlite3.connect(path)
    try:
        regime_col = "market_regime TEXT," if with_regime_column else ""
        conn.execute(f"""
            CREATE TABLE score_performance (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                score_date TEXT,
                eval_date_10d TEXT,
                eval_date_30d TEXT,
                {regime_col}
                return_10d REAL,
                return_30d REAL,
                spy_10d_return REAL,
                spy_30d_return REAL,
                beat_spy_10d INTEGER,
                beat_spy_30d INTEGER
            )
        """)
        cols = ["ticker", "score_date", "eval_date_10d", "eval_date_30d"]
        if with_regime_column:
            cols.append("market_regime")
        cols += ["return_10d", "return_30d", "spy_10d_return", "spy_30d_return",
                 "beat_spy_10d", "beat_spy_30d"]
        placeholders = ",".join(["?"] * len(cols))
        for r in rows:
            conn.execute(
                f"INSERT INTO score_performance ({','.join(cols)}) VALUES ({placeholders})",
                tuple(r.get(c) for c in cols),
            )
        conn.commit()
    finally:
        conn.close()


class TestLoadWithSubscoresAndRegime:
    def test_loads_regime_column(self, tmp_path):
        db = tmp_path / "t.db"
        _create_score_perf_db(db, rows=[
            {"ticker": "AAPL", "score_date": "2026-01-05", "market_regime": "bull",
             "return_10d": 0.05, "spy_10d_return": 0.02, "beat_spy_10d": 1},
        ])
        df = load_with_subscores_and_regime(str(db))
        assert "market_regime" in df.columns
        assert df["market_regime"].iloc[0] == "bull"

    def test_premigration_no_regime_column_filled_null(self, tmp_path):
        """Pre-migration #12 schema lacks market_regime column. Loader
        synthesizes it as all-NaN so downstream grouping skips those rows."""
        db = tmp_path / "t.db"
        _create_score_perf_db(db, with_regime_column=False, rows=[
            {"ticker": "AAPL", "score_date": "2026-01-05",
             "return_10d": 0.05, "spy_10d_return": 0.02, "beat_spy_10d": 1},
        ])
        df = load_with_subscores_and_regime(str(db))
        assert "market_regime" in df.columns
        assert df["market_regime"].isna().all()


# ---------------------------------------------------------------------------
# Annualized Sharpe formula
# ---------------------------------------------------------------------------


class TestAnnualizedSharpeFromAlphas:
    def test_positive_mean_zero_variance_returns_none(self):
        """All identical alphas → std=0 → undefined Sharpe → None."""
        alphas = np.array([0.05, 0.05, 0.05])
        assert _annualized_sharpe_from_alphas(alphas, horizon_days=10) is None

    def test_returns_none_when_sample_too_small(self):
        assert _annualized_sharpe_from_alphas(np.array([0.05]), horizon_days=10) is None
        assert _annualized_sharpe_from_alphas(np.array([]), horizon_days=10) is None

    def test_annualization_scales_with_horizon(self):
        """A given mean/std ratio should produce a smaller annualized
        Sharpe when the horizon is longer (fewer windows per year)."""
        alphas = np.array([0.10, -0.05, 0.08, 0.02, -0.03])
        sharpe_10d = _annualized_sharpe_from_alphas(alphas, horizon_days=10)
        sharpe_30d = _annualized_sharpe_from_alphas(alphas, horizon_days=30)
        # 30d horizon → sqrt(252/30) ≈ 2.90 vs 10d → sqrt(252/10) ≈ 5.02
        assert sharpe_10d is not None
        assert sharpe_30d is not None
        assert abs(sharpe_10d) > abs(sharpe_30d)

    def test_matches_manual_formula(self):
        """Pin the Sharpe formula explicitly: mean/std × sqrt(252/horizon)."""
        alphas = np.array([0.04, 0.02, -0.01, 0.03, -0.02])
        mean = alphas.mean()
        std = alphas.std(ddof=1)
        expected = mean / std * math.sqrt(252 / 10)
        actual = _annualized_sharpe_from_alphas(alphas, horizon_days=10)
        assert actual == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Stratified Sharpe by regime
# ---------------------------------------------------------------------------


def _synthetic_stratified_df(
    *,
    n_per_regime: int = 50,
    seed: int = 7,
) -> pd.DataFrame:
    """Synthetic score_performance with three regimes:
    - bull picks have positive mean alpha (+0.02 per 10d)
    - bear picks have negative mean alpha (-0.02 per 10d) — agent
      called bear correctly so picks on average underperformed
    - neutral picks have zero mean alpha

    The expected stratified Sharpe spread (bull - bear) should be
    strongly positive — the regime call enabled differentiating
    high-alpha from low-alpha picks.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for regime, mean_alpha in [("bull", 0.02), ("neutral", 0.0), ("bear", -0.02)]:
        for i in range(n_per_regime):
            alpha_10d = rng.normal(mean_alpha, 0.04)
            alpha_30d = rng.normal(mean_alpha * 1.5, 0.06)
            spy_10d = rng.normal(0.005, 0.02)
            spy_30d = rng.normal(0.015, 0.04)
            rows.append({
                "ticker": f"T{i}_{regime}",
                "score_date": "2026-01-01",
                "market_regime": regime,
                "return_10d": spy_10d + alpha_10d,
                "spy_10d_return": spy_10d,
                "return_30d": spy_30d + alpha_30d,
                "spy_30d_return": spy_30d,
                "beat_spy_10d": int(alpha_10d > 0),
                "beat_spy_30d": int(alpha_30d > 0),
            })
    return pd.DataFrame(rows)


class TestStratifiedSharpeByRegime:
    def test_returns_strata_per_regime_per_horizon(self):
        df = _synthetic_stratified_df()
        strata = stratified_sharpe_by_regime(df)
        # 3 regimes × 2 horizons = 6 strata
        assert len(strata) == 6
        regimes_horizons = {(s.market_regime, s.horizon_days) for s in strata}
        for regime in ("bull", "neutral", "bear"):
            for horizon in (10, 30):
                assert (regime, horizon) in regimes_horizons

    def test_bull_sharpe_higher_than_bear_on_synthetic(self):
        """On synthetic data with +0.02 mean alpha in bull and -0.02 in
        bear, the bull stratum's Sharpe should be materially higher."""
        df = _synthetic_stratified_df()
        strata = stratified_sharpe_by_regime(df)
        by_key = {(s.market_regime, s.horizon_days): s for s in strata}
        bull_10d = by_key[("bull", 10)].annualized_sharpe
        bear_10d = by_key[("bear", 10)].annualized_sharpe
        assert bull_10d > bear_10d
        # And the magnitudes should differ enough to be detectable above sampling noise
        assert bull_10d - bear_10d > 1.0

    def test_min_picks_gate_returns_none_metrics(self):
        """Stratum with fewer than min_picks_per_stratum picks has
        None Sharpe + mean + std (caller can filter from the headline
        metric while still seeing n_picks in the report)."""
        df = _synthetic_stratified_df(n_per_regime=10)
        # min_picks=20 > n_per_regime=10 → all strata insufficient
        strata = stratified_sharpe_by_regime(df, min_picks_per_stratum=20)
        for s in strata:
            assert s.n_picks == 10
            assert s.annualized_sharpe is None
            assert s.mean_alpha is None

    def test_skips_rows_with_null_regime(self):
        """Pre-migration rows with NULL market_regime are dropped from
        stratification — they wouldn't have a meaningful regime to
        attribute the pick alpha to."""
        df = pd.DataFrame([
            {"market_regime": "bull", "return_10d": 0.05, "spy_10d_return": 0.02,
             "return_30d": 0.08, "spy_30d_return": 0.03, "beat_spy_10d": 1, "beat_spy_30d": 1},
            {"market_regime": None, "return_10d": 0.03, "spy_10d_return": 0.02,
             "return_30d": 0.04, "spy_30d_return": 0.03, "beat_spy_10d": 1, "beat_spy_30d": 1},
        ])
        strata = stratified_sharpe_by_regime(df, min_picks_per_stratum=1)
        regimes = {s.market_regime for s in strata}
        assert regimes == {"bull"}  # NULL row excluded

    def test_no_market_regime_column_returns_empty(self):
        df = pd.DataFrame({"return_10d": [0.05], "spy_10d_return": [0.02]})
        strata = stratified_sharpe_by_regime(df)
        assert strata == []

    def test_hit_rate_computed_when_beat_col_populated(self):
        df = _synthetic_stratified_df()
        strata = stratified_sharpe_by_regime(df)
        # Hit rate should be in (0, 1) for populated strata
        for s in strata:
            if s.n_picks >= DEFAULT_MIN_PICKS_PER_STRATUM:
                assert s.hit_rate is not None
                assert 0.0 <= s.hit_rate <= 1.0
        # Bull stratum's hit rate should exceed bear stratum's
        by_key = {(s.market_regime, s.horizon_days): s for s in strata}
        bull_hit = by_key[("bull", 10)].hit_rate
        bear_hit = by_key[("bear", 10)].hit_rate
        assert bull_hit is not None and bear_hit is not None
        assert bull_hit > bear_hit


# ---------------------------------------------------------------------------
# Headline spread metric
# ---------------------------------------------------------------------------


class TestComputeRegimeSpread:
    def test_positive_spread_useful_interpretation(self):
        strata = [
            StratumMetrics("bull", 10, 30, 0.02, 0.04, 1.5, 0.7),
            StratumMetrics("bear", 10, 30, -0.02, 0.04, -0.8, 0.3),
        ]
        spread = compute_regime_spread(strata, horizon_days=10)
        assert spread["spread_bull_minus_bear"] == pytest.approx(2.3)
        assert spread["interpretation"] == "regime_signal_useful"

    def test_neutral_band_interpretation(self):
        """Small absolute spread → regime signal is neutral (within noise band)."""
        strata = [
            StratumMetrics("bull", 10, 30, 0.0, 0.04, 0.1, 0.5),
            StratumMetrics("bear", 10, 30, 0.0, 0.04, 0.0, 0.5),
        ]
        spread = compute_regime_spread(strata, horizon_days=10)
        assert spread["spread_bull_minus_bear"] == pytest.approx(0.1)
        assert spread["interpretation"] == "regime_signal_neutral"

    def test_inverted_interpretation(self):
        """Bear-called picks outperforming bull-called picks → regime
        signal is inverted (the model is wrong-way around)."""
        strata = [
            StratumMetrics("bull", 10, 30, -0.02, 0.04, -0.8, 0.3),
            StratumMetrics("bear", 10, 30, 0.02, 0.04, 1.5, 0.7),
        ]
        spread = compute_regime_spread(strata, horizon_days=10)
        assert spread["spread_bull_minus_bear"] == pytest.approx(-2.3)
        assert spread["interpretation"] == "regime_signal_inverted"

    def test_insufficient_sample_when_either_side_is_none(self):
        strata = [
            StratumMetrics("bull", 10, 30, 0.02, 0.04, 1.5, 0.7),
            StratumMetrics("bear", 10, 5, None, None, None, None),  # n=5, below min
        ]
        spread = compute_regime_spread(strata, horizon_days=10)
        assert spread["spread_bull_minus_bear"] is None
        assert spread["interpretation"] == "insufficient_sample"

    def test_horizon_30d_pulls_correct_stratum(self):
        """30d horizon should pull strata where horizon_days=30, not 10."""
        strata = [
            StratumMetrics("bull", 10, 30, 0.02, 0.04, 5.0, 0.7),
            StratumMetrics("bull", 30, 30, 0.04, 0.06, 2.0, 0.7),  # smaller annualized
            StratumMetrics("bear", 10, 30, -0.02, 0.04, -5.0, 0.3),
            StratumMetrics("bear", 30, 30, -0.04, 0.06, -2.0, 0.3),
        ]
        spread_30d = compute_regime_spread(strata, horizon_days=30)
        # Should compute from the 30d strata: 2.0 - (-2.0) = 4.0
        assert spread_30d["bull_sharpe"] == pytest.approx(2.0)
        assert spread_30d["bear_sharpe"] == pytest.approx(-2.0)
        assert spread_30d["spread_bull_minus_bear"] == pytest.approx(4.0)

    def test_neutral_caution_strata_surfaced(self):
        """Per regime-v3-260514.md §5.1, the agent's 4-class taxonomy
        includes caution. T2 reports Sharpe for all 4 even though the
        headline spread is bull-bear."""
        strata = [
            StratumMetrics("bull", 10, 30, 0.02, 0.04, 1.5, 0.7),
            StratumMetrics("neutral", 10, 30, 0.005, 0.04, 0.3, 0.5),
            StratumMetrics("caution", 10, 30, -0.005, 0.04, -0.4, 0.45),
            StratumMetrics("bear", 10, 30, -0.02, 0.04, -0.8, 0.3),
        ]
        spread = compute_regime_spread(strata, horizon_days=10)
        assert spread["neutral_sharpe"] == pytest.approx(0.3)
        assert spread["caution_sharpe"] == pytest.approx(-0.4)


# ---------------------------------------------------------------------------
# Eval-artifact payload assembly
# ---------------------------------------------------------------------------


class TestAssembleT2EvalPayload:
    def test_payload_shape(self):
        strata = [
            StratumMetrics("bull", 10, 30, 0.02, 0.04, 1.5, 0.7),
            StratumMetrics("bear", 10, 30, -0.02, 0.04, -0.8, 0.3),
        ]
        spread_10d = compute_regime_spread(strata, horizon_days=10)
        spread_30d = {"horizon_days": 30, "spread_bull_minus_bear": None,
                      "interpretation": "insufficient_sample"}
        payload = assemble_t2_eval_payload(
            strata=strata,
            spread_10d=spread_10d,
            spread_30d=spread_30d,
            run_id="2605170230",
            calendar_date="2026-05-17",
            trading_day="2026-05-15",
        )
        assert payload["calendar_date"] == "2026-05-17"
        assert payload["run_id"] == "2605170230"
        assert payload["eval_tier"] == "T2_downstream_stratified_sharpe"
        assert payload["spread_10d"]["interpretation"] == "regime_signal_useful"
        assert payload["spread_30d"]["interpretation"] == "insufficient_sample"
        # Method metadata pins the annualization basis + alpha definition
        md = payload["method_metadata"]
        assert "252_trading_days" in md["annualization_basis"]
        assert "per-pick cross-sectional alpha" in md["alpha_definition"]


# ---------------------------------------------------------------------------
# Default pins
# ---------------------------------------------------------------------------


class TestDefaultsPins:
    def test_min_picks_default(self):
        assert DEFAULT_MIN_PICKS_PER_STRATUM == 20

    def test_supported_horizons(self):
        assert SUPPORTED_HORIZONS == (10, 30)
