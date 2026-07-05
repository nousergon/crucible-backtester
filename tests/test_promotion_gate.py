"""Tests for the SOTA promotion gate — PSR + DSR + CSCV-PBO (config#950).

Covers the López de Prado selection-bias battery wired into
``executor_optimizer.validate_walk_forward``:
  * ``analysis.pbo.cscv_pbo`` — Probability of Backtest Overfitting.
  * ``_chronological_blocks`` — CSCV block partition.
  * ``_compute_pbo`` — top-K × block matrix → PBO verdict.
  * ``_evaluate_promotion_gate`` — PSR/DSR on the OOS stream + PBO, with the
    honest-N/A (insufficient → non-blocking) posture.
"""

from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

from analysis.pbo import cscv_pbo
from optimizer.executor_optimizer import (
    _chronological_blocks,
    _compute_pbo,
    _evaluate_promotion_gate,
    init_config,
)


def _stable_seed(*parts) -> int:
    """Process-independent seed derived from ``parts``.

    Python's builtin ``hash()`` salts str/tuple hashes with a per-process
    random seed (``PYTHONHASHSEED``) unless explicitly pinned, so a sim_fn
    keyed off ``hash((...))`` draws a *different* RNG stream every run. Under
    an unlucky salt the per-combo noise stops separating cleanly across CSCV
    blocks and PBO can land at 1.0 by chance — flaking
    ``test_strong_low_trial_passes`` on totally unrelated CI runs (#1751).
    ``hashlib`` digests are stable across processes/interpreters, so the same
    inputs always yield the same seed and thus the same simulated returns.
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


def _returns_sim(mu: float, sigma: float = 0.01, seed_salt: int = 0):
    """Date-aware sim_fn emitting a return series + Sharpe/Sortino. Mean return
    scales with the combo's ``atr_multiplier`` so combos are separable for PBO,
    with per-combo deterministic noise."""
    def sim_fn(combo_config, dates=None):
        d = list(dates)
        scale = float(combo_config.get("atr_multiplier", 1.0))
        seed = _stable_seed(round(scale, 4), len(d), d[0] if d else "", seed_salt)
        rng = np.random.RandomState(seed)
        r = pd.Series(rng.normal(mu * scale, sigma, len(d)), index=pd.to_datetime(d))
        sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
        return {"status": "ok", "daily_returns": r,
                "sharpe_ratio": float(sh), "sortino_ratio": float(sh)}
    return sim_fn


class TestCscvPbo:
    def test_ok_over_sufficient_matrix(self):
        rng = np.random.RandomState(0)
        m = rng.normal(0, 1, (6, 4))
        res = cscv_pbo(m)
        assert res["status"] == "ok"
        assert 0.0 <= res["pbo"] <= 1.0
        assert res["n_splits"] == 6 and res["n_specs"] == 4

    def test_consistent_winner_has_low_pbo(self):
        # combo 0 is best on every split → IS winner always wins OOS → PBO 0.
        m = np.tile(np.array([3.0, 2.0, 1.0, 0.0]), (6, 1)) + \
            np.random.RandomState(1).normal(0, 0.01, (6, 4))
        res = cscv_pbo(m)
        assert res["status"] == "ok"
        assert res["pbo"] == 0.0

    def test_insufficient_too_few_splits(self):
        assert cscv_pbo(np.zeros((2, 4)))["status"] == "insufficient"

    def test_insufficient_one_combo(self):
        assert cscv_pbo(np.zeros((6, 1)))["status"] == "insufficient"

    def test_nan_rows_dropped(self):
        m = np.random.RandomState(2).normal(0, 1, (8, 3))
        m[0, 0] = np.nan  # one dirty row dropped → 7 clean
        res = cscv_pbo(m)
        assert res["status"] == "ok"
        assert res["n_splits"] == 7


class TestChronologicalBlocks:
    def test_partitions_contiguously_and_covers_all(self):
        dates = [f"d{i:02d}" for i in range(60)]
        blocks = _chronological_blocks(dates, 6)
        assert len(blocks) == 6
        assert [d for b in blocks for d in b] == dates  # full cover, in order

    def test_remainder_absorbed_into_last(self):
        dates = [f"d{i:02d}" for i in range(13)]
        blocks = _chronological_blocks(dates, 6)
        assert sum(len(b) for b in blocks) == 13
        assert blocks[-1][-1] == "d12"

    def test_empty(self):
        assert _chronological_blocks([], 6) == []


class TestComputePbo:
    def _dates(self, n=240):
        return pd.date_range("2025-01-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()

    def test_ok_matrix_built_and_graded(self):
        init_config({"executor_optimizer": {}})
        combos = [{"atr_multiplier": x} for x in (3.0, 2.5, 2.0, 1.5)]
        res = _compute_pbo(combos, _returns_sim(0.001), self._dates(), {}, "sharpe_legacy",
                           n_blocks=6, max_pbo=0.5, top_k=20)
        assert res["status"] == "ok"
        assert res["blocking"] is True
        assert res["n_blocks"] >= 4 and res["n_combos"] == 4
        assert isinstance(res["passed"], bool)

    def test_insufficient_when_under_two_combos(self):
        res = _compute_pbo([{"atr_multiplier": 3.0}], _returns_sim(0.001), self._dates(),
                           {}, "sharpe_legacy", n_blocks=6, max_pbo=0.5, top_k=20)
        assert res["status"] == "insufficient"
        assert res["blocking"] is False

    def test_insufficient_when_too_few_blocks(self):
        # Only ~9 dates → blocks of <3 dropped → <4 usable blocks.
        res = _compute_pbo([{"atr_multiplier": 3.0}, {"atr_multiplier": 2.0}],
                           _returns_sim(0.001), [f"d{i}" for i in range(9)],
                           {}, "sharpe_legacy", n_blocks=6, max_pbo=0.5, top_k=20)
        assert res["status"] == "insufficient"
        assert res["blocking"] is False


class TestEvaluatePromotionGate:
    def _dates(self, n=240):
        return pd.date_range("2025-01-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()

    def _result(self, n_trials):
        return {"n_combos_swept": n_trials,
                "pbo_top_combos": [{"atr_multiplier": x} for x in (3.0, 2.5, 2.0, 1.5)]}

    def test_insufficient_oos_is_non_blocking(self):
        init_config({"executor_optimizer": {}})
        gate = _evaluate_promotion_gate(
            pd.Series(dtype="float64"), 60, self._result(60),
            _returns_sim(0.001), self._dates(), {}, "sharpe_legacy",
        )
        # Empty OOS → PSR/DSR insufficient; PBO still computes. Gate passes iff
        # no computable sub-gate fails.
        assert gate["sub_gates"]["psr"]["status"] == "insufficient"
        assert gate["sub_gates"]["dsr"]["status"] == "insufficient"
        assert gate["sub_gates"]["psr"]["blocking"] is False

    def test_strong_low_trial_passes(self):
        init_config({"executor_optimizer": {"min_dsr": 0.5, "max_pbo": 0.9}})
        rng = np.random.RandomState(3)
        oos = pd.Series(rng.normal(0.0015, 0.008, 220),
                        index=pd.date_range("2025-01-01", periods=220, freq="B"))
        gate = _evaluate_promotion_gate(oos, 5, self._result(5),
                                        _returns_sim(0.0015), self._dates(), {}, "sharpe_legacy")
        assert gate["sub_gates"]["psr"]["status"] == "ok"
        assert gate["sub_gates"]["dsr"]["status"] == "ok"
        assert gate["passed"] is True

    def test_high_trial_count_deflates_dsr_and_blocks(self):
        init_config({"executor_optimizer": {"min_dsr": 0.90}})
        rng = np.random.RandomState(4)
        # Decent but not extraordinary Sharpe → survives PSR, fails DSR once
        # deflated for a 400-combo sweep.
        oos = pd.Series(rng.normal(0.0006, 0.011, 220),
                        index=pd.date_range("2025-01-01", periods=220, freq="B"))
        gate = _evaluate_promotion_gate(oos, 400, self._result(400),
                                        _returns_sim(0.0006), self._dates(), {}, "sharpe_legacy")
        assert gate["sub_gates"]["dsr"]["status"] == "ok"
        assert gate["sub_gates"]["dsr"]["passed"] is False
        assert gate["passed"] is False

    def test_thresholds_are_config_driven(self):
        rng = np.random.RandomState(5)
        oos = pd.Series(rng.normal(0.0006, 0.011, 220),
                        index=pd.date_range("2025-01-01", periods=220, freq="B"))
        # Floor thresholds let the same series through (same series that fails
        # the default 0.90 DSR bar — proving the gate is config-driven, not
        # hard-coded).
        init_config({"executor_optimizer": {"min_psr": 0.0, "min_dsr": 0.0, "max_pbo": 1.0}})
        gate = _evaluate_promotion_gate(oos, 400, self._result(400),
                                        _returns_sim(0.0006), self._dates(), {}, "sharpe_legacy")
        assert gate["passed"] is True
