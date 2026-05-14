"""Tests for analysis/regime_stratified_sortino.py — Stage C.2 T2.

Covers:
- DB loader resilience (missing market_regime column → pre-migration fallback)
- Arithmetic → log-alpha conversion (canonical-alpha framework)
- Annualized Sortino formula on per-pick log alphas
  (mean/downside_std × sqrt(periods/year))
- Sharpe surfaced as a SECONDARY diagnostic
- Per-(regime, horizon) stratum metrics: n_picks, log-alpha stats,
  Sortino, Sharpe, hit rate
- Min-sample gate (n_picks below threshold → None metrics)
- Headline spread metric (bull − bear Sortino at 10d horizon) +
  interpretation flags
- Eval-artifact payload assembly
"""
from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


from analysis.regime_stratified_sortino import (
    DEFAULT_MIN_PICKS_PER_STRATUM,
    SUPPORTED_HORIZONS,
    StratumMetrics,
    _annualized_sharpe_from_log_alphas,
    _annualized_sortino_from_log_alphas,
    _arithmetic_to_log_alpha,
    assemble_t2_eval_payload,
    compute_regime_spread,
    load_with_subscores_and_regime,
    stratified_sortino_by_regime,
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
        db = tmp_path / "t.db"
        _create_score_perf_db(db, with_regime_column=False, rows=[
            {"ticker": "AAPL", "score_date": "2026-01-05",
             "return_10d": 0.05, "spy_10d_return": 0.02, "beat_spy_10d": 1},
        ])
        df = load_with_subscores_and_regime(str(db))
        assert "market_regime" in df.columns
        assert df["market_regime"].isna().all()


# ---------------------------------------------------------------------------
# Arithmetic → log alpha conversion (canonical framework)
# ---------------------------------------------------------------------------


class TestArithmeticToLogAlpha:
    def test_zero_return_zero_spy_returns_zero(self):
        log_alpha = _arithmetic_to_log_alpha(
            pd.Series([0.0]), pd.Series([0.0]),
        )
        assert log_alpha.iloc[0] == pytest.approx(0.0)

    def test_small_returns_approximate_arithmetic_alpha(self):
        """For small returns, log_alpha ≈ arithmetic alpha (Taylor series
        approximation). Pin that the conversion isn't introducing
        systematic bias at the small-return regime."""
        ret = pd.Series([0.01, 0.005])    # 1%, 0.5%
        spy = pd.Series([0.003, 0.002])   # 0.3%, 0.2%
        log_alpha = _arithmetic_to_log_alpha(ret, spy)
        arithmetic_alpha = ret - spy
        for la, aa in zip(log_alpha, arithmetic_alpha):
            assert abs(la - aa) < 0.001  # very close at small returns

    def test_large_returns_differ_from_arithmetic(self):
        """At large returns log and arithmetic diverge — pin the
        log formula gives a different value (not silently aliasing
        to arithmetic)."""
        log_alpha = _arithmetic_to_log_alpha(
            pd.Series([0.50]), pd.Series([0.10]),
        )
        # log(1.50) - log(1.10) ≈ 0.4055 - 0.0953 = 0.3102
        expected = math.log(1.50) - math.log(1.10)
        assert log_alpha.iloc[0] == pytest.approx(expected, abs=1e-6)
        # Distinct from arithmetic 0.50 - 0.10 = 0.40
        arithmetic = 0.50 - 0.10
        assert log_alpha.iloc[0] != pytest.approx(arithmetic, abs=0.01)

    def test_negative_return_handled(self):
        """A losing pick — return -20% on a stock vs +5% SPY → strongly
        negative log alpha."""
        log_alpha = _arithmetic_to_log_alpha(
            pd.Series([-0.20]), pd.Series([0.05]),
        )
        # log(0.80) - log(1.05) ≈ -0.2231 - 0.0488 = -0.2719
        expected = math.log(0.80) - math.log(1.05)
        assert log_alpha.iloc[0] == pytest.approx(expected, abs=1e-6)
        assert log_alpha.iloc[0] < 0


# ---------------------------------------------------------------------------
# Sortino + Sharpe formulas
# ---------------------------------------------------------------------------


class TestSortinoFormula:
    def test_returns_none_when_sample_too_small(self):
        assert _annualized_sortino_from_log_alphas(np.array([0.05]), horizon_days=10) is None
        assert _annualized_sortino_from_log_alphas(np.array([]), horizon_days=10) is None

    def test_returns_none_when_no_downside_sample(self):
        """A purely positive sample has no negative log-alphas → downside
        std undefined → Sortino reported as None (caller treats as
        'insufficient downside sample, skip from headline')."""
        log_alphas = np.array([0.01, 0.02, 0.03])
        assert _annualized_sortino_from_log_alphas(log_alphas, horizon_days=10) is None

    def test_sortino_penalizes_downside_only(self):
        """Compare two samples with the same mean but different
        downside profiles. Sample A has small downside; sample B has
        large downside. Sortino should rank A > B even if Sharpe
        couldn't distinguish them."""
        # Both samples have mean ~0.01, but distinct downside RMS
        sample_a = np.array([0.05, 0.05, 0.05, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005, -0.005])
        # Same mean (≈0.0085) but a single big drawdown
        sample_b = np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, -0.235])
        # Compute means to confirm parity
        assert sample_a.mean() == pytest.approx(sample_b.mean(), abs=1e-3)
        sortino_a = _annualized_sortino_from_log_alphas(sample_a, horizon_days=10)
        sortino_b = _annualized_sortino_from_log_alphas(sample_b, horizon_days=10)
        assert sortino_a is not None and sortino_b is not None
        # A has small symmetric downside; B has one big drawdown — A's Sortino higher
        assert sortino_a > sortino_b

    def test_near_zero_downside_std_returns_none(self):
        """Identical small-magnitude downside observations would give
        near-zero downside std — IEEE 754 precision floor; treat as
        undefined."""
        log_alphas = np.array([0.05, 0.05, -1e-15, -1e-15])
        sortino = _annualized_sortino_from_log_alphas(log_alphas, horizon_days=10)
        assert sortino is None


class TestSharpeDiagnostic:
    def test_secondary_sharpe_uses_full_sample_std(self):
        """The secondary Sharpe diagnostic uses standard sample std
        (ddof=1) — different denominator from Sortino's downside-only.
        For a symmetric distribution they should be comparable; for
        an asymmetric one they'll differ."""
        log_alphas = np.array([0.04, -0.02, 0.03, -0.01, 0.05, -0.02, 0.03, -0.01])
        sharpe = _annualized_sharpe_from_log_alphas(log_alphas, horizon_days=10)
        sortino = _annualized_sortino_from_log_alphas(log_alphas, horizon_days=10)
        assert sharpe is not None
        assert sortino is not None
        # Both finite, both annualized via the same factor — just different denominators
        # Sortino's downside-only denominator is smaller than full std → Sortino > Sharpe
        assert sortino > sharpe

    def test_annualization_scales_with_horizon(self):
        """10d horizon Sortino should be larger in magnitude than 30d
        Sortino on the same data — sqrt(252/10) > sqrt(252/30)."""
        log_alphas = np.array([0.02, -0.01, 0.03, -0.02, 0.01, -0.005, 0.025])
        sortino_10d = _annualized_sortino_from_log_alphas(log_alphas, horizon_days=10)
        sortino_30d = _annualized_sortino_from_log_alphas(log_alphas, horizon_days=30)
        assert sortino_10d is not None and sortino_30d is not None
        assert abs(sortino_10d) > abs(sortino_30d)


# ---------------------------------------------------------------------------
# Stratified Sortino by regime
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

    The expected stratified Sortino spread (bull − bear) should be
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


class TestStratifiedSortinoByRegime:
    def test_returns_strata_per_regime_per_horizon(self):
        df = _synthetic_stratified_df()
        strata = stratified_sortino_by_regime(df)
        # 3 regimes × 2 horizons = 6 strata
        assert len(strata) == 6
        keys = {(s.market_regime, s.horizon_days) for s in strata}
        for regime in ("bull", "neutral", "bear"):
            for horizon in (10, 30):
                assert (regime, horizon) in keys

    def test_bull_sortino_higher_than_bear_on_synthetic(self):
        """On synthetic data with +0.02 mean alpha in bull and -0.02 in
        bear, the bull stratum's Sortino should be materially higher."""
        df = _synthetic_stratified_df()
        strata = stratified_sortino_by_regime(df)
        by_key = {(s.market_regime, s.horizon_days): s for s in strata}
        bull_10d = by_key[("bull", 10)].annualized_sortino
        bear_10d = by_key[("bear", 10)].annualized_sortino
        assert bull_10d is not None and bear_10d is not None
        assert bull_10d > bear_10d
        # Magnitudes should differ enough to be detectable above sampling noise
        assert bull_10d - bear_10d > 1.0

    def test_both_sortino_and_sharpe_reported(self):
        """Every populated stratum reports both Sortino (headline) and
        Sharpe (secondary diagnostic)."""
        df = _synthetic_stratified_df()
        strata = stratified_sortino_by_regime(df)
        for s in strata:
            if s.n_picks >= DEFAULT_MIN_PICKS_PER_STRATUM:
                assert s.annualized_sortino is not None
                assert s.annualized_sharpe is not None

    def test_log_alpha_stats_computed(self):
        """Headline stratum surfaces mean_log_alpha + std_log_alpha +
        downside_std_log_alpha so the Sortino denominator is
        auditable independently of the ratio."""
        df = _synthetic_stratified_df()
        strata = stratified_sortino_by_regime(df)
        bull_10d = next(s for s in strata if s.market_regime == "bull" and s.horizon_days == 10)
        assert bull_10d.mean_log_alpha is not None
        assert bull_10d.std_log_alpha is not None
        assert bull_10d.downside_std_log_alpha is not None
        # Downside std ≤ full std by construction
        assert bull_10d.downside_std_log_alpha <= bull_10d.std_log_alpha

    def test_min_picks_gate_returns_none_metrics(self):
        df = _synthetic_stratified_df(n_per_regime=10)
        strata = stratified_sortino_by_regime(df, min_picks_per_stratum=20)
        for s in strata:
            assert s.n_picks == 10
            assert s.annualized_sortino is None
            assert s.annualized_sharpe is None
            assert s.mean_log_alpha is None

    def test_skips_rows_with_null_regime(self):
        df = pd.DataFrame([
            {"market_regime": "bull", "return_10d": 0.05, "spy_10d_return": 0.02,
             "return_30d": 0.08, "spy_30d_return": 0.03, "beat_spy_10d": 1, "beat_spy_30d": 1},
            {"market_regime": None, "return_10d": 0.03, "spy_10d_return": 0.02,
             "return_30d": 0.04, "spy_30d_return": 0.03, "beat_spy_10d": 1, "beat_spy_30d": 1},
        ])
        strata = stratified_sortino_by_regime(df, min_picks_per_stratum=1)
        regimes = {s.market_regime for s in strata}
        assert regimes == {"bull"}

    def test_no_market_regime_column_returns_empty(self):
        df = pd.DataFrame({"return_10d": [0.05], "spy_10d_return": [0.02]})
        strata = stratified_sortino_by_regime(df)
        assert strata == []

    def test_hit_rate_computed_when_beat_col_populated(self):
        df = _synthetic_stratified_df()
        strata = stratified_sortino_by_regime(df)
        for s in strata:
            if s.n_picks >= DEFAULT_MIN_PICKS_PER_STRATUM:
                assert s.hit_rate is not None
                assert 0.0 <= s.hit_rate <= 1.0
        by_key = {(s.market_regime, s.horizon_days): s for s in strata}
        bull_hit = by_key[("bull", 10)].hit_rate
        bear_hit = by_key[("bear", 10)].hit_rate
        assert bull_hit > bear_hit


# ---------------------------------------------------------------------------
# Headline spread metric
# ---------------------------------------------------------------------------


def _stratum(regime: str, horizon: int, n: int, sortino: float, sharpe: float = 0.0):
    """Convenience constructor for spread-metric tests."""
    return StratumMetrics(
        market_regime=regime,
        horizon_days=horizon,
        n_picks=n,
        mean_log_alpha=0.02,
        std_log_alpha=0.04,
        downside_std_log_alpha=0.03,
        annualized_sortino=sortino,
        annualized_sharpe=sharpe,
        hit_rate=0.5,
    )


class TestComputeRegimeSpread:
    def test_positive_spread_useful_interpretation(self):
        strata = [
            _stratum("bull", 10, 30, sortino=1.5),
            _stratum("bear", 10, 30, sortino=-0.8),
        ]
        spread = compute_regime_spread(strata, horizon_days=10)
        assert spread["spread_bull_minus_bear_sortino"] == pytest.approx(2.3)
        assert spread["interpretation"] == "regime_signal_useful"

    def test_neutral_band_interpretation(self):
        strata = [
            _stratum("bull", 10, 30, sortino=0.1),
            _stratum("bear", 10, 30, sortino=0.0),
        ]
        spread = compute_regime_spread(strata, horizon_days=10)
        assert spread["spread_bull_minus_bear_sortino"] == pytest.approx(0.1)
        assert spread["interpretation"] == "regime_signal_neutral"

    def test_inverted_interpretation(self):
        strata = [
            _stratum("bull", 10, 30, sortino=-0.8),
            _stratum("bear", 10, 30, sortino=1.5),
        ]
        spread = compute_regime_spread(strata, horizon_days=10)
        assert spread["spread_bull_minus_bear_sortino"] == pytest.approx(-2.3)
        assert spread["interpretation"] == "regime_signal_inverted"

    def test_insufficient_sample_when_either_side_is_none(self):
        strata = [
            _stratum("bull", 10, 30, sortino=1.5),
            StratumMetrics("bear", 10, 5, None, None, None, None, None, None),
        ]
        spread = compute_regime_spread(strata, horizon_days=10)
        assert spread["spread_bull_minus_bear_sortino"] is None
        assert spread["interpretation"] == "insufficient_sample"

    def test_sharpe_spread_surfaced_as_secondary_diagnostic(self):
        """The headline spread is Sortino; Sharpe is reported alongside
        as a secondary diagnostic for cross-reference + legacy continuity."""
        strata = [
            _stratum("bull", 10, 30, sortino=1.5, sharpe=1.2),
            _stratum("bear", 10, 30, sortino=-0.8, sharpe=-0.6),
        ]
        spread = compute_regime_spread(strata, horizon_days=10)
        # Headline (Sortino)
        assert spread["spread_bull_minus_bear_sortino"] == pytest.approx(2.3)
        # Secondary diagnostic (Sharpe)
        assert spread["diagnostic_sharpe_spread_bull_minus_bear"] == pytest.approx(1.8)
        assert spread["diagnostic_bull_sharpe"] == pytest.approx(1.2)
        assert spread["diagnostic_bear_sharpe"] == pytest.approx(-0.6)

    def test_horizon_30d_pulls_correct_stratum(self):
        strata = [
            _stratum("bull", 10, 30, sortino=5.0),
            _stratum("bull", 30, 30, sortino=2.0),
            _stratum("bear", 10, 30, sortino=-5.0),
            _stratum("bear", 30, 30, sortino=-2.0),
        ]
        spread_30d = compute_regime_spread(strata, horizon_days=30)
        assert spread_30d["bull_sortino"] == pytest.approx(2.0)
        assert spread_30d["bear_sortino"] == pytest.approx(-2.0)
        assert spread_30d["spread_bull_minus_bear_sortino"] == pytest.approx(4.0)

    def test_neutral_caution_strata_surfaced(self):
        strata = [
            _stratum("bull", 10, 30, sortino=1.5),
            _stratum("neutral", 10, 30, sortino=0.3),
            _stratum("caution", 10, 30, sortino=-0.4),
            _stratum("bear", 10, 30, sortino=-0.8),
        ]
        spread = compute_regime_spread(strata, horizon_days=10)
        assert spread["neutral_sortino"] == pytest.approx(0.3)
        assert spread["caution_sortino"] == pytest.approx(-0.4)


# ---------------------------------------------------------------------------
# Eval-artifact payload assembly
# ---------------------------------------------------------------------------


class TestAssembleT2EvalPayload:
    def test_payload_shape(self):
        strata = [
            _stratum("bull", 10, 30, sortino=1.5, sharpe=1.0),
            _stratum("bear", 10, 30, sortino=-0.8, sharpe=-0.5),
        ]
        spread_10d = compute_regime_spread(strata, horizon_days=10)
        spread_30d = {"horizon_days": 30, "spread_bull_minus_bear_sortino": None,
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
        assert payload["eval_tier"] == "T2_downstream_stratified_sortino"
        assert payload["spread_10d"]["interpretation"] == "regime_signal_useful"
        assert payload["spread_30d"]["interpretation"] == "insufficient_sample"
        # Method metadata pins the canonical-alpha framework conventions
        md = payload["method_metadata"]
        assert "log(1+return" in md["alpha_definition"]
        assert "sortino" in md["headline_metric"].lower()
        assert "sharpe" in md["secondary_diagnostic"].lower()
        assert "PSR + max DD not computed" in md["psr_max_dd_note"]

    def test_strata_serialize_both_sortino_and_sharpe(self):
        strata = [_stratum("bull", 10, 30, sortino=1.5, sharpe=1.0)]
        payload = assemble_t2_eval_payload(
            strata=strata,
            spread_10d={}, spread_30d={},
            run_id="X", calendar_date="2026-05-17", trading_day="2026-05-15",
        )
        s = payload["strata"][0]
        assert s["annualized_sortino"] == pytest.approx(1.5)
        assert s["annualized_sharpe_diagnostic"] == pytest.approx(1.0)
        assert "mean_log_alpha" in s
        assert "downside_std_log_alpha" in s


# ---------------------------------------------------------------------------
# Default pins
# ---------------------------------------------------------------------------


class TestDefaultsPins:
    def test_min_picks_default(self):
        assert DEFAULT_MIN_PICKS_PER_STRATUM == 20

    def test_supported_horizons(self):
        assert SUPPORTED_HORIZONS == (10, 30)
