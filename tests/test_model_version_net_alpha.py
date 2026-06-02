"""Tests for L4488b — per-model-version net-of-cost alpha under a FIXED policy.

Pins: the better-ranking version wins net alpha under the SAME policy (execution
isolation); the loader groups predictions by model_version across the live +
shadow tables and degrades when the shadow table is absent; the Deflated-Sharpe
best-of-N control behaves (None on <2 trials / zero spread, a probability
otherwise).
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from analysis.model_version_net_alpha import (
    _deflated_sharpe_best_of_n,
    compute_model_version_net_alpha,
    load_version_predictions,
)
from analysis.transaction_cost import TransactionCostModel

_ZERO_COST = TransactionCostModel(half_spread_bps=0.0, impact_coef_bps=0.0,
                                  commission_bps=0.0, min_cost_bps=0.0)


def _panel(n_dates=400, n_good=20, n_bad=30, good_drift=0.0012, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_dates)
    good = [f"G{i:02d}" for i in range(n_good)]
    bad = [f"B{i:02d}" for i in range(n_bad)]
    cols = good + bad + ["SPY"]
    prices = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for t in cols:
        drift = good_drift if t in good else 0.0
        prices[t] = 100.0 * np.cumprod(1.0 + drift + rng.normal(0, 0.002, n_dates))
    dates = [d.strftime("%Y-%m-%d") for d in idx]
    good_preds = {d: {**{t: 1.0 for t in good}, **{t: 0.0 for t in bad}} for d in dates}
    bad_preds = {d: {**{t: 0.0 for t in good}, **{t: 1.0 for t in bad}} for d in dates}
    return good_preds, bad_preds, prices, prices["SPY"], good, bad


def test_better_version_wins_net_alpha_under_fixed_policy():
    good_preds, bad_preds, prices, spy, *_ = _panel()
    out = compute_model_version_net_alpha(
        {"good": good_preds, "bad": bad_preds}, prices, spy,
        cost_model=_ZERO_COST, horizon=21, top_n=20,
    )
    g = out["versions"]["good"]["net_alpha_ann"]
    b = out["versions"]["bad"]["net_alpha_ann"]
    assert g > b  # the version that ranks the rising names higher wins...
    assert out["net_alpha_leader"] == "good"  # ...under the IDENTICAL fixed policy


def test_no_predictions_status():
    _, _, prices, spy, *_ = _panel()
    out = compute_model_version_net_alpha({"empty": {}}, prices, spy, cost_model=_ZERO_COST)
    assert out["versions"]["empty"]["status"] == "no_predictions"
    assert out["net_alpha_leader"] is None


def test_deflated_sharpe_best_of_n():
    assert _deflated_sharpe_best_of_n([1.2], 30) is None          # <2 trials
    assert _deflated_sharpe_best_of_n([1.0, 1.0, 1.0], 30) is None  # zero spread
    assert _deflated_sharpe_best_of_n([1.5, 0.2, 0.4], 2) is None  # too few obs
    dsr = _deflated_sharpe_best_of_n([1.8, 0.3, 0.5, 0.1], 40)
    assert dsr is not None and 0.0 <= dsr <= 1.0


def _make_db(tmp_path, *, with_shadow=True):
    db = tmp_path / "research.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE predictor_outcomes (id INTEGER PRIMARY KEY, symbol TEXT, "
            "prediction_date TEXT, p_up REAL, model_version TEXT)"
        )
        conn.executemany(
            "INSERT INTO predictor_outcomes (symbol, prediction_date, p_up, model_version) VALUES (?,?,?,?)",
            [("AAPL", "2026-01-02", 0.7, "champ"), ("MSFT", "2026-01-02", 0.4, "champ"),
             ("AAPL", "2026-01-03", 0.6, None)],  # legacy NULL → champion-legacy
        )
        if with_shadow:
            conn.execute(
                "CREATE TABLE predictor_outcomes_shadow (id INTEGER PRIMARY KEY, symbol TEXT, "
                "prediction_date TEXT, p_up REAL, model_version TEXT)"
            )
            conn.executemany(
                "INSERT INTO predictor_outcomes_shadow (symbol, prediction_date, p_up, model_version) VALUES (?,?,?,?)",
                [("AAPL", "2026-01-02", 0.9, "V1"), ("MSFT", "2026-01-02", 0.2, "V1")],
            )
        conn.commit()
    return str(db)


def test_loader_groups_by_version_across_tables(tmp_path):
    out = load_version_predictions(_make_db(tmp_path))
    assert out["champ"]["2026-01-02"] == {"AAPL": 0.7, "MSFT": 0.4}
    assert out["champion-legacy"]["2026-01-03"] == {"AAPL": 0.6}
    assert out["V1"]["2026-01-02"] == {"AAPL": 0.9, "MSFT": 0.2}


def test_loader_degrades_when_shadow_table_absent(tmp_path):
    out = load_version_predictions(_make_db(tmp_path, with_shadow=False))
    assert "champ" in out and "V1" not in out  # champion-only, no crash


def test_loader_missing_db_returns_empty():
    assert load_version_predictions("/nonexistent/research.db") == {}
