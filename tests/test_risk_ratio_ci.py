"""Unit tests for analysis.risk_ratio_ci (config#976 — magnitude-uncertainty monitor).

Verifies the Director L4558 discipline is operationalized:
  * CIs that straddle zero ⇒ magnitude_certain=False (size by direction only);
  * a strong, well-powered signal ⇒ CI one side of zero ⇒ magnitude_certain=True;
  * the producer is DETERMINISTIC (seeded bootstrap — same returns ⇒ same CI);
  * always-emit / graceful degrade (insufficient_data, no_benchmark).
"""

import numpy as np
import pytest

from analysis.risk_ratio_ci import (
    RISK_RATIO_SAMPLE_FLOOR,
    _information_ratio,
    _sharpe,
    _sortino,
    compute_risk_ratio_ci,
)


def _returns(mean, sd, n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(mean, sd, size=n)


class TestPointEstimators:
    def test_sharpe_none_on_too_few(self):
        assert _sharpe(np.array([0.01])) is None

    def test_sharpe_none_on_zero_vol(self):
        assert _sharpe(np.array([0.01, 0.01, 0.01])) is None

    def test_sharpe_sign_matches_mean(self):
        assert _sharpe(_returns(0.002, 0.01, 300)) > 0
        assert _sharpe(_returns(-0.002, 0.01, 300)) < 0

    def test_sortino_none_when_no_downside(self):
        assert _sortino(np.array([0.01, 0.02, 0.03])) is None

    def test_sortino_finite_with_downside(self):
        val = _sortino(_returns(0.001, 0.01, 300))
        assert val is not None and np.isfinite(val)

    def test_information_ratio_none_on_zero_te(self):
        assert _information_ratio(np.array([0.0, 0.0, 0.0])) is None


class TestMagnitudeCertainty:
    def test_small_noisy_sample_is_uncertain(self):
        """N=63 weak signal — the Director's cited regime: CI straddles zero,
        magnitude NOT certain (size by direction only)."""
        pr = _returns(-0.0003, 0.012, 63, seed=1)
        out = compute_risk_ratio_ci(pr)
        assert out["status"] == "ok"
        assert out["n_samples"] == 63
        assert out["n_adequate"] is False  # below the 126-day floor
        # Below floor ⇒ never magnitude_certain regardless of CI.
        for name in ("sharpe_ratio", "sortino_ratio"):
            assert out["ratios"][name]["magnitude_certain"] is False
        assert out["all_magnitude_certain"] is False

    def test_strong_well_powered_signal_is_certain(self):
        """Large N + strong positive drift ⇒ CI one side of zero ⇒ magnitude
        certain (sign resolved, magnitude quotable)."""
        pr = _returns(0.0015, 0.006, 400, seed=2)
        out = compute_risk_ratio_ci(pr)
        assert out["n_adequate"] is True
        sharpe = out["ratios"]["sharpe_ratio"]
        assert sharpe["ci_95"] is not None
        assert sharpe["straddles_zero"] is False
        assert sharpe["ci_95"][0] > 0  # whole CI positive
        assert sharpe["magnitude_certain"] is True

    def test_below_floor_blocks_certainty_even_if_ci_clears_zero(self):
        """Strong signal but N < floor ⇒ still not magnitude_certain (the floor
        is a hard gate — the Director wants the SAMPLE widened first)."""
        pr = _returns(0.002, 0.005, RISK_RATIO_SAMPLE_FLOOR - 1, seed=3)
        out = compute_risk_ratio_ci(pr)
        assert out["n_adequate"] is False
        assert out["ratios"]["sharpe_ratio"]["magnitude_certain"] is False


class TestDeterminism:
    def test_same_returns_yield_identical_ci(self):
        pr = _returns(0.0005, 0.01, 200, seed=7)
        a = compute_risk_ratio_ci(pr)
        b = compute_risk_ratio_ci(pr)
        assert a["ratios"]["sharpe_ratio"]["ci_95"] == b["ratios"]["sharpe_ratio"]["ci_95"]
        assert a["ratios"]["sortino_ratio"]["ci_95"] == b["ratios"]["sortino_ratio"]["ci_95"]


class TestInformationRatio:
    def test_ir_computed_with_benchmark(self):
        pr = _returns(0.0012, 0.008, 300, seed=4)
        spy = _returns(0.0004, 0.007, 300, seed=5)
        out = compute_risk_ratio_ci(pr, spy)
        ir = out["ratios"]["information_ratio"]
        assert "ci_95" in ir and ir["ci_95"] is not None

    def test_ir_na_without_benchmark(self):
        out = compute_risk_ratio_ci(_returns(0.0, 0.01, 300, seed=6))
        assert out["ratios"]["information_ratio"]["status"] == "no_benchmark"

    def test_ir_na_on_length_mismatch(self):
        pr = _returns(0.0, 0.01, 300, seed=6)
        spy = _returns(0.0, 0.01, 100, seed=6)
        out = compute_risk_ratio_ci(pr, spy)
        assert out["ratios"]["information_ratio"]["status"] == "no_benchmark"


class TestGracefulDegrade:
    def test_insufficient_data(self):
        out = compute_risk_ratio_ci([0.01])
        assert out["status"] == "insufficient_data"
        assert out["all_magnitude_certain"] is False

    def test_accepts_list_and_series(self):
        data = list(_returns(0.0005, 0.01, 200, seed=8))
        out = compute_risk_ratio_ci(data)
        assert out["status"] == "ok"
        pd = pytest.importorskip("pandas")
        out2 = compute_risk_ratio_ci(pd.Series(data))
        assert out2["ratios"]["sharpe_ratio"]["ci_95"] == out["ratios"]["sharpe_ratio"]["ci_95"]

    def test_floor_constant_is_six_months(self):
        assert RISK_RATIO_SAMPLE_FLOOR == 126
