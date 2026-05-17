"""Unit tests for PR 3 — the point-in-time contamination report
(``analysis/pit_parity.py``; ROADMAP L2371 / plan §D4).

Locks: the basket is Sortino/PSR/CVaR/maxDD + log-domain headline with
**no Sharpe**; log-domain cumulative return is summed not compounded;
deltas are pit−current; the 2-split CSCV PBO is honest (None without a
sweep pair, never a fabricated number); the report is observational and
flip-gated; run_pit_parity runs both passes with the flag flipped and
never raises on an S3 upload failure.
"""

from __future__ import annotations

import datetime as dt
import sys
import types

import numpy as np
import pandas as pd
import pytest

from analysis import pit_parity as pp


def _stats(sortino, psr, cvar, mdd, log_rets, total_alpha):
    return {
        "sortino_ratio": sortino, "psr": psr, "cvar_95": cvar,
        "max_drawdown": mdd, "total_alpha": total_alpha,
        "daily_log_returns": np.array(log_rets, dtype=float),
        "total_return": float(np.expm1(np.sum(log_rets))),
        "status": "ok",
    }


def test_log_cum_return_is_summed_not_compounded():
    s = _stats(1.0, 0.9, -0.02, -0.1, [0.01, 0.02, -0.005], 0.03)
    # Time-additive: sum of daily log returns (plan invariant 5).
    assert pp._log_cum_return(s) == pytest.approx(0.025)


def test_log_cum_return_falls_back_to_log1p_total_return():
    assert pp._log_cum_return({"total_return": 0.10}) == pytest.approx(
        np.log1p(0.10)
    )
    assert pp._log_cum_return({"total_return": None}) is None


def test_basket_has_no_sharpe():
    s = _stats(1.2, 0.95, -0.03, -0.15, [0.01], 0.04)
    s["sharpe_ratio"] = 2.5  # present in stats but must NOT enter the basket
    b = pp._basket(s)
    assert "sharpe_ratio" not in b
    assert set(b) == {"sortino_ratio", "psr", "cvar_95",
                      "max_drawdown", "log_cum_return", "total_alpha"}


def test_delta_is_pit_minus_current_and_none_safe():
    cur = pp._basket(_stats(1.0, 0.9, -0.04, -0.20, [0.0], 0.01))
    pit = pp._basket(_stats(0.7, 0.8, -0.05, -0.25, [0.0], -0.01))
    d = pp._delta(pit, cur)
    assert d["sortino_ratio"] == pytest.approx(-0.3)   # pit − current
    assert d["max_drawdown"] == pytest.approx(-0.05)
    # None on either side → None, never a crash.
    assert pp._delta({"sortino_ratio": None}, {"sortino_ratio": 1.0})[
        "sortino_ratio"] is None


def test_pbo_none_without_sweep_pair():
    assert pp._pbo_two_split(None, None) is None


def test_pbo_two_split_detects_overfit():
    # In-sample ranks configs c0>c1>c2; out-of-sample reverses → the best
    # IS config (c0) lands at OOS percentile 0.0 < 0.5 ⇒ overfit=True.
    cur = pd.DataFrame({"config_id": [0, 1, 2], "sortino_ratio": [3.0, 2.0, 1.0]})
    pit = pd.DataFrame({"config_id": [0, 1, 2], "sortino_ratio": [1.0, 2.0, 3.0]})
    r = pp._pbo_two_split(cur, pit)
    assert r["n_configs"] == 3
    assert r["overfit"] is True
    assert r["best_in_sample_config_oos_percentile"] == pytest.approx(0.0)
    assert r["spearman_rank_corr"] == pytest.approx(-1.0)


def test_pbo_two_split_stable_when_ranks_agree():
    cur = pd.DataFrame({"config_id": [0, 1, 2], "sortino_ratio": [3.0, 2.0, 1.0]})
    pit = pd.DataFrame({"config_id": [0, 1, 2], "sortino_ratio": [3.1, 2.2, 0.9]})
    r = pp._pbo_two_split(cur, pit)
    assert r["overfit"] is False
    assert r["best_in_sample_config_oos_percentile"] == pytest.approx(1.0)


def test_build_report_shape_and_materiality():
    cur = _stats(1.20, 0.96, -0.030, -0.12, [0.012, 0.004], 0.05)
    pit = _stats(0.85, 0.88, -0.041, -0.18, [0.006, 0.001], 0.02)
    rep = pp.build_contamination_report(
        cur, pit, run_date="2026-05-17",
        wf_meta={"n_folds": 40, "n_cold_start_excluded": 6},
    )
    assert rep["schema"] == pp.SCHEMA
    assert "Sharpe deliberately absent" in rep["anchor"]
    # ΔSortino = 0.85 − 1.20 = −0.35 → |Δ| ≥ 0.10 ⇒ material.
    assert rep["delta_pit_minus_current"]["sortino_ratio"] == pytest.approx(-0.35)
    assert rep["materiality"]["material"] is True
    assert rep["pbo"] is None  # no sweep pair in single-pass parity
    assert rep["observational"] is True
    assert "Brian-gated" in rep["flip_gate"]
    assert rep["run_quality"]["walk_forward"]["n_cold_start_excluded"] == 6
    assert rep["headline_log_alpha_delta"] == pytest.approx(
        (0.006 + 0.001) - (0.012 + 0.004)
    )


def test_run_pit_parity_runs_both_passes_and_survives_upload_failure(monkeypatch):
    seen: list[bool] = []

    def fake_run_predictor_backtest(cfg):
        seen.append(cfg["walk_forward"])
        # PIT pass carries wf metadata through predictor_metadata.
        s = _stats(1.0 if not cfg["walk_forward"] else 0.6,
                   0.9, -0.03, -0.15, [0.01, 0.0], 0.03)
        if cfg["walk_forward"]:
            s["predictor_metadata"] = {"walk_forward": {"n_folds": 12,
                                                        "n_cold_start_excluded": 2}}
        return s

    fake_bt = types.ModuleType("backtest")
    fake_bt.run_predictor_backtest = fake_run_predictor_backtest
    monkeypatch.setitem(sys.modules, "backtest", fake_bt)

    # S3 upload must be best-effort: a boto failure cannot raise.
    class _BoomS3:
        def put_object(self, **kw):
            raise RuntimeError("S3 down")

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda *a, **k: _BoomS3()
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    rep = pp.run_pit_parity({"signals_bucket": "b", "_run_date": "2026-05-17"})

    # Both passes ran, in order, with the flag flipped — and the original
    # config was deep-copied (not mutated).
    assert seen == [False, True]
    assert rep["delta_pit_minus_current"]["sortino_ratio"] == pytest.approx(-0.4)
    assert rep["run_quality"]["walk_forward"]["n_cold_start_excluded"] == 2
    assert "_s3_key" not in rep  # upload failed but run_pit_parity returned


def test_run_pit_parity_incomplete_pass_yields_status_report(monkeypatch):
    fake_bt = types.ModuleType("backtest")
    fake_bt.run_predictor_backtest = lambda cfg: {"status": "insufficient_data"}
    monkeypatch.setitem(sys.modules, "backtest", fake_bt)
    rep = pp.run_pit_parity({"signals_bucket": "b", "_run_date": "2026-05-17"})
    assert rep["status"] == "incomplete"
    assert rep["observational"] is True
