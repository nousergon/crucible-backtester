"""Tests for optimizer.significance_observe — observe-mode significance verdicts
(config#1426 Phase 2).

Pins:
  1. A strong monotone conviction↔return relationship → significant=True
     (bootstrap IC CI excludes zero).
  2. Independent (null) conviction/return → significant=False (CI brackets zero) —
     this is the L4593 leg-f case the gate must catch.
  3. Insufficient data / no-variance → status set, significant=False.
  4. Determinism: same seed → identical verdict (the report card must be stable).
  5. observe_weight_optimizer: strong sub-score → would_block=False; null
     sub-scores → would_block=True.
  6. build_observe_record: promotes_on_undefended_evidence semantics; observe is
     NEVER enforced (enforced is always False).
  7. Observe is non-enforcing: an instrumentation failure must not break
     compute_weights, and apply_weights ignores the significance verdict.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from optimizer.significance_observe import (  # noqa: E402
    build_observe_record,
    ic_significance_verdict,
    observe_weight_optimizer,
)


# ── ic_significance_verdict ──────────────────────────────────────────────────

class TestICSignificanceVerdict:
    def test_strong_signal_is_significant(self):
        rng = np.random.default_rng(1)
        conviction = np.arange(120, dtype=float)
        forward = conviction + rng.normal(0, 5, size=120)  # near-perfect rank order
        v = ic_significance_verdict(conviction, forward)
        assert v["status"] == "ok"
        assert v["significant"] is True
        assert v["ci_low"] > 0  # CI excludes zero on the positive side
        assert v["ic"] > 0.8

    def test_null_signal_is_not_significant(self):
        rng = np.random.default_rng(42)
        conviction = rng.normal(size=150)
        forward = rng.normal(size=150)  # independent → no real IC
        v = ic_significance_verdict(conviction, forward)
        assert v["status"] == "ok"
        assert v["significant"] is False
        assert v["ci_low"] <= 0 <= v["ci_high"]  # CI brackets zero

    def test_insufficient_data(self):
        v = ic_significance_verdict([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], min_samples=20)
        assert v["status"] == "insufficient_data"
        assert v["significant"] is False

    def test_no_variance(self):
        conviction = np.ones(50)  # constant → IC undefined
        forward = np.arange(50, dtype=float)
        v = ic_significance_verdict(conviction, forward)
        assert v["status"] == "no_variance"
        assert v["significant"] is False

    def test_deterministic_same_seed(self):
        rng = np.random.default_rng(7)
        c = rng.normal(size=80)
        r = c * 0.3 + rng.normal(0, 1, size=80)
        a = ic_significance_verdict(c, r, seed=0)
        b = ic_significance_verdict(c, r, seed=0)
        assert a == b

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            ic_significance_verdict([1.0, 2.0, 3.0], [1.0, 2.0])


# ── observe_weight_optimizer ─────────────────────────────────────────────────

def _make_test_set(n: int, *, signal: bool, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    quant = rng.normal(size=n)
    if signal:
        ret10 = quant * 0.5 + rng.normal(0, 0.5, size=n)
    else:
        ret10 = rng.normal(size=n)  # independent of quant
    return pd.DataFrame({
        "quant_score": quant,
        "qual_score": rng.normal(size=n),  # always null
        "return_10d": ret10,
        "return_30d": ret10 + rng.normal(0, 1.0, size=n),
    })


class TestObserveWeightOptimizer:
    SUB_COLS = {"quant": "quant_score", "qual": "qual_score"}

    def test_strong_subscore_not_blocked(self):
        ts = _make_test_set(200, signal=True, seed=3)
        rec = observe_weight_optimizer(ts, self.SUB_COLS)
        assert rec["gate"] == "weight_optimizer"
        assert rec["significant"] is True
        assert rec["would_block"] is False
        assert rec["detail"]["per_subscore"]["quant"]["significant"] is True

    def test_null_subscores_blocked(self):
        ts = _make_test_set(200, signal=False, seed=9)
        rec = observe_weight_optimizer(ts, self.SUB_COLS)
        assert rec["significant"] is False
        assert rec["would_block"] is True
        assert rec["enforced"] is False

    def test_missing_return_column_is_handled(self):
        ts = _make_test_set(60, signal=True, seed=4).drop(columns=["return_30d"])
        rec = observe_weight_optimizer(ts, self.SUB_COLS, return_cols=("return_10d", "return_30d"))
        q = rec["detail"]["per_subscore"]["quant"]["horizons"]
        assert q["return_30d"]["status"] == "missing_column"


# ── build_observe_record ─────────────────────────────────────────────────────

class TestBuildObserveRecord:
    def test_promotes_on_undefended_when_promote_and_not_significant(self):
        rec = build_observe_record(gate="g", significant=False, did_promote=True)
        assert rec["would_block"] is True
        assert rec["promotes_on_undefended_evidence"] is True
        assert rec["enforced"] is False

    def test_defended_when_significant(self):
        rec = build_observe_record(gate="g", significant=True, did_promote=True)
        assert rec["would_block"] is False
        assert rec["promotes_on_undefended_evidence"] is False

    def test_unknown_promote_is_none(self):
        rec = build_observe_record(gate="g", significant=False, did_promote=None)
        assert rec["promotes_on_undefended_evidence"] is None


# ── non-enforcement guarantees ───────────────────────────────────────────────

class TestObserveIsNonEnforcing:
    def test_compute_weights_survives_observe_failure(self, monkeypatch):
        """An observe-instrumentation error must NOT break the optimizer."""
        from optimizer import weight_optimizer

        def _boom(*a, **k):
            raise RuntimeError("induced observe failure")

        monkeypatch.setattr(
            "optimizer.significance_observe.observe_weight_optimizer", _boom,
        )
        # Build a minimal valid score_performance frame.
        rng = np.random.default_rng(0)
        n = 80
        df = pd.DataFrame({
            "symbol": [f"T{i}" for i in range(n)],
            "score_date": pd.date_range("2025-01-01", periods=n, freq="D"),
            # canonical post-cutover outcome columns (config#1451)
            "beat_spy_21d": rng.integers(0, 2, size=n).astype(float),
            "beat_spy_5d": rng.integers(0, 2, size=n).astype(float),
            "log_alpha_21d": rng.normal(size=n),
            "return_5d": rng.normal(size=n),
            "quant_score": rng.normal(size=n),
            "qual_score": rng.normal(size=n),
        })
        weight_optimizer.init_config({"weight_optimizer": {}})
        result = weight_optimizer.compute_weights(df, min_samples=20)
        assert result["status"] == "ok"  # optimizer unaffected by observe failure
        assert result["significance_observe"] is None  # swallowed, recorded as None


# ── Phase 3: additional verdicts ─────────────────────────────────────────────

from optimizer.significance_observe import (  # noqa: E402
    mean_diff_significance_verdict,
    observe_ic_gate,
    observe_stance_spread,
    observe_veto,
    proportion_lift_significance_verdict,
)


class TestProportionLiftVerdict:
    def test_clear_lift_is_significant(self):
        # 90/100 precision vs 0.50 base rate — Wilson lower bound well above base.
        v = proportion_lift_significance_verdict(90, 100, 0.50)
        assert v["status"] == "ok"
        assert v["significant"] is True
        assert v["ci_low"] > 0.50

    def test_no_lift_is_not_significant(self):
        # 11/20 ≈ 0.55 vs 0.50 base rate — small N, CI brackets the base rate.
        v = proportion_lift_significance_verdict(11, 20, 0.50)
        assert v["significant"] is False

    def test_invalid_counts(self):
        assert proportion_lift_significance_verdict(0, 0, 0.5)["significant"] is False
        assert proportion_lift_significance_verdict(5, 3, 0.5)["significant"] is False


class TestMeanDiffVerdict:
    def test_clear_difference_is_significant(self):
        rng = np.random.default_rng(2)
        a = rng.normal(1.0, 0.5, size=120)
        b = rng.normal(-1.0, 0.5, size=120)
        v = mean_diff_significance_verdict(a, b)
        assert v["status"] == "ok"
        assert v["significant"] is True
        assert v["ci_low"] > 0

    def test_same_distribution_not_significant(self):
        rng = np.random.default_rng(5)
        a = rng.normal(0.0, 1.0, size=150)
        b = rng.normal(0.0, 1.0, size=150)
        v = mean_diff_significance_verdict(a, b)
        assert v["significant"] is False
        assert v["ci_low"] <= 0 <= v["ci_high"]

    def test_insufficient(self):
        v = mean_diff_significance_verdict([1.0] * 5, [2.0] * 5, min_samples=20)
        assert v["status"] == "insufficient_data"
        assert v["significant"] is False


class TestObserveICGate:
    def test_strong_signal_not_blocked(self):
        rng = np.random.default_rng(1)
        conv = np.arange(120, dtype=float)
        ret = conv + rng.normal(0, 5, size=120)
        rec = observe_ic_gate(conv, ret, gate="predictor_sizing")
        assert rec["gate"] == "predictor_sizing"
        assert rec["would_block"] is False
        assert rec["enforced"] is False

    def test_null_signal_blocked(self):
        rng = np.random.default_rng(42)
        rec = observe_ic_gate(rng.normal(size=150), rng.normal(size=150), gate="barrier_sizing")
        assert rec["would_block"] is True


class TestObserveVeto:
    THRESHOLDS = [
        {"confidence": 0.60, "true_negatives": 11, "n_vetoes": 20},
        {"confidence": 0.70, "true_negatives": 90, "n_vetoes": 100},
    ]

    def test_significant_threshold_not_blocked(self):
        rec = observe_veto(self.THRESHOLDS, 0.70, 0.50)
        assert rec["gate"] == "veto_analysis"
        assert rec["would_block"] is False

    def test_weak_threshold_blocked(self):
        rec = observe_veto(self.THRESHOLDS, 0.60, 0.50)
        assert rec["would_block"] is True

    def test_missing_threshold_row_is_insufficient(self):
        rec = observe_veto(self.THRESHOLDS, 0.99, 0.50)
        assert rec["would_block"] is True
        assert rec["detail"]["status"] == "insufficient_data"


class TestObserveStanceSpread:
    def test_distinct_stances_not_blocked(self):
        rng = np.random.default_rng(3)
        samples = {
            "momentum": rng.normal(0.02, 0.01, size=120),
            "value": rng.normal(-0.02, 0.01, size=120),
        }
        rec = observe_stance_spread(samples)
        assert rec["gate"] == "stance_sizing"
        assert rec["would_block"] is False
        assert rec["detail"]["best_stance"] == "momentum"

    def test_indistinct_stances_blocked(self):
        rng = np.random.default_rng(8)
        samples = {
            "momentum": rng.normal(0.0, 0.02, size=120),
            "value": rng.normal(0.0, 0.02, size=120),
        }
        rec = observe_stance_spread(samples)
        assert rec["would_block"] is True

    def test_single_stance_insufficient(self):
        rec = observe_stance_spread({"momentum": np.zeros(30)})
        assert rec["would_block"] is True
        assert rec["detail"]["status"] == "insufficient_stances"
