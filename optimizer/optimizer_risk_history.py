"""Optimizer risk-history record — observability artifact for the dashboard.

Every backtester run that produces a covariance-estimator sweep verdict posts a
flat, versioned snapshot of the portfolio optimizer's **risk-tolerance levers**
(the swept dimensions + the static optimizer-config defaults) together with the
**risk metrics** for the SELECTED cell (the cov-sweep winner, else the legacy
baseline). The dashboard's Optimizer-Risk page reads the append-per-run history
(``config/optimizer_risk_history/{run_id}.json`` + ``…/latest.json``) to chart
the optimizer's risk posture over time.

Why a dedicated artifact rather than reading the dated ``backtest/{day}/
cov_sweep.json`` files directly: those are keyed by ``trading_day`` and
overwrite on same-day reruns, and their shape is the full per-cell verdict — not
a clean time-series. This record mirrors the blessed
``config/executor_params_history`` precedent (immutable ``run_id``, latest
sidecar) on the optimizer's risk axis.

Failure posture — SECONDARY observability hung off the backtester's primary
sweep deliverables (``cov_sweep.json`` / ``gamma_sweep.json``, already durable).
The write is best-effort: a failure is logged WARN and the artifact's absence is
recorded by the ARTIFACT_REGISTRY freshness monitor (severity: warning). This is
the no-silent-fails "secondary observability hung off a primary path that
records the failure" carve-out — it must never block the Saturday run.
"""
from __future__ import annotations

import json
import logging

from alpha_engine_lib.eval_artifacts import (
    eval_artifact_key,
    eval_latest_key,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
HISTORY_PREFIX = "config/optimizer_risk_history"

# ── Declared record vocabulary (pinned by the producer contract test) ─────────
# Provenance / metadata.
_META_KEYS = (
    "schema_version", "run_id", "trading_day", "updated_at", "signal_source",
    "cov_baseline_name", "cov_winner_name", "cov_selected_name",
    "cov_selected_is_winner", "gamma_status", "gamma_winner_name", "gate_passed",
)
# Risk-tolerance levers (effective config of the selected cell).
_LEVER_KEYS = (
    "risk_aversion", "tcost_bps", "covariance_shrinkage", "sigma_horizon_days",
    "ewma_lambda_decay", "alpha_uncertainty_penalty", "vol_target_annual",
    "max_daily_turnover", "cash_sleeve_pct", "max_sector_pct",
)
# Risk metrics of the selected cell's backtest.
_METRIC_KEYS = (
    "sortino_ratio", "psr", "cvar_95", "max_drawdown", "calmar_ratio",
    "sharpe_ratio", "tracking_error_ann", "mean_active_share",
    "turnover_one_way_ann", "total_alpha", "win_rate", "n_solver_failures",
)
RECORD_KEYS: frozenset[str] = frozenset(_META_KEYS + _LEVER_KEYS + _METRIC_KEYS)

# Levers sourced from the selected cell's effective optimizer cfg.
_LEVER_FROM_CFG = (
    "risk_aversion", "tcost_bps", "covariance_shrinkage", "sigma_horizon_days",
    "ewma_lambda_decay", "vol_target_annual", "max_daily_turnover",
    "cash_sleeve_pct", "max_sector_pct",
)


def _selected_cell(sweep_payload: dict | None) -> tuple[str | None, dict, bool]:
    """Return (selected_name, selected_metrics, is_winner) for a sweep payload.

    Selection = the gate winner if one cleared, else the baseline (the legacy
    config still in force). Returns ("", {}, False) when the payload is absent or
    carries no usable ``cells`` (e.g. status=skipped)."""
    if not isinstance(sweep_payload, dict):
        return None, {}, False
    cells = sweep_payload.get("cells")
    if not isinstance(cells, dict) or not cells:
        return None, {}, False
    winner = sweep_payload.get("winner_name")
    baseline = sweep_payload.get("baseline_name")
    selected = winner if winner else baseline
    metrics = cells.get(selected) if selected else None
    if not isinstance(metrics, dict):
        return selected, {}, bool(winner)
    return selected, metrics, bool(winner)


def build_optimizer_risk_record(
    *,
    cov_payload: dict | None,
    optimizer_defaults: dict,
    trading_day: str,
    updated_at: str,
    run_id: str,
    gamma_payload: dict | None = None,
    gate_payload: dict | None = None,
) -> dict | None:
    """Assemble one flat optimizer-risk-history record from already-computed
    sweep verdicts.

    Args:
        cov_payload: the ``run_cov_estimator_sweep_stage`` return (the primary
            source — must carry ``cells``; returns None if it does not).
        optimizer_defaults: the executor's ``OPTIMIZER_CONFIG_DEFAULTS`` (so the
            static levers are read from the real source, not hard-coded here).
        trading_day / updated_at: artifact date stamps (calendar/trading day).
        run_id: immutable id for the dated key (wall-clock for live runs,
            day-derived for backfill).
        gamma_payload: the ``run_gamma_sweep_stage`` return (optional — γ-sweep
            is data-gated and often skipped; when absent, γ is the baseline 0.0).
        gate_payload: the ``run_portfolio_optimizer_gate`` return (optional —
            supplies ``gate_passed``).

    Returns the record dict, or None when no usable cov-sweep cells exist (the
    caller skips the write cleanly — that is not an error)."""
    cov_name, cov_metrics, cov_is_winner = _selected_cell(cov_payload)
    if not cov_metrics:
        return None

    # Effective config of the selected cov cell = optimizer defaults overlaid
    # with the cell's swept overrides (covariance_shrinkage / sigma_horizon_days
    # / risk_aversion / ewma_lambda_decay).
    eff_cfg = {**(optimizer_defaults or {}), **dict(cov_metrics.get("cell_cfg") or {})}

    # γ overlay: the gamma-sweep winner's penalty when the sweep ran with a
    # winner, else the deployed baseline (0.0). The cov sweep does not vary γ.
    gamma_name, gamma_metrics, _gamma_is_winner = _selected_cell(gamma_payload)
    if isinstance(gamma_payload, dict):
        gamma_status = gamma_payload.get("status", "ok")
    else:
        gamma_status = "absent"
    if gamma_metrics:
        gamma_cfg = dict(gamma_metrics.get("cell_cfg") or {})
        alpha_unc_penalty = gamma_cfg.get("alpha_uncertainty_penalty", 0.0)
    else:
        alpha_unc_penalty = float(eff_cfg.get("alpha_uncertainty_penalty", 0.0) or 0.0)

    record: dict = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "trading_day": trading_day,
        "updated_at": updated_at,
        "signal_source": (cov_payload or {}).get("signal_source", "synthetic"),
        "cov_baseline_name": (cov_payload or {}).get("baseline_name"),
        "cov_winner_name": (cov_payload or {}).get("winner_name"),
        "cov_selected_name": cov_name,
        "cov_selected_is_winner": cov_is_winner,
        "gamma_status": gamma_status,
        "gamma_winner_name": gamma_name,
        "gate_passed": (gate_payload or {}).get("passed") if isinstance(gate_payload, dict) else None,
        "alpha_uncertainty_penalty": alpha_unc_penalty,
    }
    for k in _LEVER_FROM_CFG:
        record[k] = eff_cfg.get(k)
    for k in _METRIC_KEYS:
        record[k] = cov_metrics.get(k)

    # Defensive: never emit a key outside the declared vocabulary (keeps the
    # cross-repo contract honest — see test_optimizer_risk_history_producer_contract).
    extra = set(record) - RECORD_KEYS
    if extra:
        raise ValueError(
            f"optimizer-risk record emits undeclared key(s) {sorted(extra)} — "
            "add them to RECORD_KEYS and the dashboard loader/page."
        )
    return record


def write_optimizer_risk_history(record: dict, *, bucket: str, s3) -> dict:
    """Write the record to ``{HISTORY_PREFIX}/{run_id}.json`` + ``…/latest.json``.

    Best-effort: logs WARN on failure (the artifact's absence is monitored via
    ARTIFACT_REGISTRY freshness) and returns a status dict; never raises."""
    run_id = record["run_id"]
    dated_key = eval_artifact_key(HISTORY_PREFIX, run_id)
    latest_key = eval_latest_key(HISTORY_PREFIX)
    body = json.dumps(record, default=str, indent=2).encode("utf-8")
    try:
        s3.put_object(Bucket=bucket, Key=dated_key, Body=body, ContentType="application/json")
        s3.put_object(Bucket=bucket, Key=latest_key, Body=body, ContentType="application/json")
        logger.info(
            "optimizer_risk_history: persisted s3://%s/%s (+ latest sidecar) "
            "cov_selected=%s sortino=%s",
            bucket, dated_key, record.get("cov_selected_name"),
            record.get("sortino_ratio"),
        )
        return {"written": True, "key": dated_key}
    except Exception as exc:  # noqa: BLE001 — secondary observability, see module docstring
        logger.warning(
            "optimizer_risk_history: S3 persist failed (non-fatal): %s", exc,
        )
        return {"written": False, "reason": str(exc)}
