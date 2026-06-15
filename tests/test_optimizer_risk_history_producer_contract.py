"""Producer contract for config/optimizer_risk_history/{run_id}.json.

Pins the VOCABULARY of the optimizer-risk-history record the backtester posts
each run (consumed by the dashboard Optimizer-Risk page). The record is built by
``optimizer.optimizer_risk_history.build_optimizer_risk_record`` from the
already-computed cov-sweep / γ-sweep / gate verdicts.

If this fails you are either (a) adding a field — declare it in RECORD_KEYS AND
teach the dashboard loader/page to read it, or (b) removing one — drop it from
RECORD_KEYS and the consumer. The dashboard tolerates missing fields (older
records), so additive changes are safe; renames/removals are the hazard.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer.optimizer_risk_history import (
    RECORD_KEYS,
    SCHEMA_VERSION,
    build_optimizer_risk_record,
)

# Mirror of the executor's OPTIMIZER_CONFIG_DEFAULTS static levers (the producer
# reads the real dict at runtime; the test supplies a stand-in so it needs no
# executor on path).
_DEFAULTS = {
    "vol_target_annual": None,
    "risk_aversion": 5.0,
    "tcost_bps": 5.0,
    "cash_sleeve_pct": 0.03,
    "max_sector_pct": 0.25,
    "covariance_shrinkage": "ledoit_wolf",
    "sigma_horizon_days": 1,
    "ewma_lambda_decay": 0.94,
    "max_daily_turnover": 0.20,
    "alpha_uncertainty_penalty": 0.0,
}


def _cell_metrics(**over) -> dict:
    m = {
        "sortino_ratio": 1.2, "psr": 0.97, "cvar_95": -0.03,
        "max_drawdown": -0.18, "calmar_ratio": 0.9, "sharpe_ratio": 1.1,
        "tracking_error_ann": 0.04, "mean_active_share": 0.15,
        "turnover_one_way_ann": 0.5, "total_alpha": 0.06, "win_rate": 0.55,
        "n_solver_failures": 0,
        "cell_cfg": {"covariance_shrinkage": "oas", "sigma_horizon_days": 21,
                     "risk_aversion": 0.238},
    }
    m.update(over)
    return m


def _cov_payload(winner: str | None = "oas_h21") -> dict:
    return {
        "run_date": "2026-06-13", "status": "ok", "signal_source": "synthetic",
        "baseline_name": "ledoit_wolf_h1",
        "winner_name": winner,
        "cells": {
            "ledoit_wolf_h1": _cell_metrics(
                cell_cfg={"covariance_shrinkage": "ledoit_wolf",
                          "sigma_horizon_days": 1, "risk_aversion": 5.0}),
            "oas_h21": _cell_metrics(),
        },
    }


def _gate_payload(passed: bool = False) -> dict:
    # Mirrors predictor/optimizer_gate/{date}.json: metrics live under
    # comparison.optimizer; deployed config (no swept overrides).
    opt = {k: v for k, v in _cell_metrics().items() if k != "cell_cfg"}
    return {
        "run_date": "2026-06-13", "signal_source": "synthetic", "passed": passed,
        "comparison": {"optimizer": opt, "signal_source": "synthetic"},
    }


def _build(**over):
    kw = dict(
        cov_payload=_cov_payload(),
        optimizer_defaults=_DEFAULTS,
        trading_day="2026-06-13",
        updated_at="2026-06-13",
        run_id="2606131200",
    )
    kw.update(over)
    return build_optimizer_risk_record(**kw)


def test_record_keys_equal_declared_vocabulary():
    record = _build()
    assert set(record.keys()) == set(RECORD_KEYS), (
        f"emitted keys != RECORD_KEYS; "
        f"extra={sorted(set(record) - set(RECORD_KEYS))} "
        f"missing={sorted(set(RECORD_KEYS) - set(record))}"
    )


def test_levers_resolve_from_winner_cell_and_defaults():
    record = _build()
    # Swept levers come from the winning cell's cfg ...
    assert record["covariance_shrinkage"] == "oas"
    assert record["sigma_horizon_days"] == 21
    assert record["risk_aversion"] == 0.238
    # ... static levers come from the executor defaults.
    assert record["tcost_bps"] == 5.0
    assert record["cash_sleeve_pct"] == 0.03
    assert record["max_sector_pct"] == 0.25
    assert record["vol_target_annual"] is None
    assert record["schema_version"] == SCHEMA_VERSION


def test_metrics_come_from_selected_cell():
    record = _build()
    assert record["sortino_ratio"] == 1.2
    assert record["psr"] == 0.97
    assert record["max_drawdown"] == -0.18
    assert record["cov_selected_name"] == "oas_h21"
    assert record["cov_selected_is_winner"] is True
    assert record["metrics_source"] == "cov_sweep"


def test_falls_back_to_gate_metrics_when_cov_absent():
    # Production reality today: cov_sweep not producing → record keys off the
    # optimizer gate's deployed-config metrics, with deployed-default levers.
    record = _build(cov_payload=None, gate_payload=_gate_payload(passed=True))
    assert record is not None
    assert record["metrics_source"] == "optimizer_gate"
    assert record["sortino_ratio"] == 1.2          # from comparison.optimizer
    assert record["max_drawdown"] == -0.18
    assert record["cov_selected_name"] is None
    assert record["cov_selected_is_winner"] is False
    # Levers = deployed defaults (no swept overrides).
    assert record["risk_aversion"] == 5.0
    assert record["covariance_shrinkage"] == "ledoit_wolf"
    assert record["sigma_horizon_days"] == 1
    assert record["gate_passed"] is True


def test_falls_back_to_baseline_when_no_winner():
    record = _build(cov_payload=_cov_payload(winner=None))
    assert record["cov_selected_name"] == "ledoit_wolf_h1"
    assert record["cov_selected_is_winner"] is False
    assert record["covariance_shrinkage"] == "ledoit_wolf"
    assert record["sigma_horizon_days"] == 1


def test_gamma_winner_overrides_alpha_uncertainty_penalty():
    gamma_payload = {
        "status": "ok", "baseline_name": "baseline_gamma_0",
        "winner_name": "gamma_100",
        "cells": {
            "baseline_gamma_0": _cell_metrics(
                cell_cfg={"alpha_uncertainty_penalty": 0.0}),
            "gamma_100": _cell_metrics(
                cell_cfg={"alpha_uncertainty_penalty": 100.0}),
        },
    }
    record = _build(gamma_payload=gamma_payload)
    assert record["alpha_uncertainty_penalty"] == 100.0
    assert record["gamma_status"] == "ok"
    assert record["gamma_winner_name"] == "gamma_100"


def test_gamma_absent_defaults_penalty_zero():
    record = _build(gamma_payload=None)
    assert record["alpha_uncertainty_penalty"] == 0.0
    assert record["gamma_status"] == "absent"
    assert record["gamma_winner_name"] is None


def test_gate_passed_lifted_from_gate_payload():
    record = _build(gate_payload={"passed": True})
    assert record["gate_passed"] is True
    record2 = _build(gate_payload=None)
    assert record2["gate_passed"] is None


def test_returns_none_when_no_source_has_metrics():
    # Robustness: when NEITHER the cov-sweep (no cells) NOR the gate (no
    # comparison.optimizer) has metrics, no record is written.
    assert _build(cov_payload={"status": "skipped", "reason": "x"}, gate_payload=None) is None
    assert _build(cov_payload=None, gate_payload=None) is None
    assert _build(cov_payload=None, gate_payload={"passed": False}) is None  # no comparison


def test_old_schema_cell_missing_metrics_coerces_to_none():
    # Backfill case: an older verdict cell missing some metric fields still
    # produces a contract-valid record (missing -> None), never a KeyError.
    sparse = {"sortino_ratio": 0.8, "cell_cfg": {
        "covariance_shrinkage": "ledoit_wolf", "sigma_horizon_days": 1}}
    payload = {"status": "ok", "baseline_name": "ledoit_wolf_h1",
               "winner_name": None, "cells": {"ledoit_wolf_h1": sparse}}
    record = _build(cov_payload=payload)
    assert set(record.keys()) == set(RECORD_KEYS)
    assert record["sortino_ratio"] == 0.8
    assert record["psr"] is None
    assert record["cvar_95"] is None
