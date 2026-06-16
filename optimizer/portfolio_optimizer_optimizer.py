"""
portfolio_optimizer_optimizer.py — recommend + apply the MVO optimizer's OWN
params (``risk_aversion`` × ``tcost_bps``) from the weekly optimizer-param sweep,
with hard safeguards so a bad recommendation can't botch the live book
(config#1057 increment 2).

This writes LIVE optimizer config (``config/portfolio_optimizer.json``) that the
executor's ``solve_target_weights`` reads (via the ``optimizer_shadow`` merge,
the sibling PR). Because it touches live trading, the safeguards ARE the feature:

  - **Writable-set allowlist** — ONLY ``risk_aversion`` + ``tcost_bps`` are ever
    written. The safety knobs (turnover governor, drawdown circuit breaker,
    sector/position caps, vol target) are NEVER auto-tuned; they stay
    operator-owned, so an applied config can't loosen a guardrail.
  - **Hard clamp** — every written value is clamped to a sane band
    (``PARAM_BOUNDS``) even if the sweep/logic somehow produced an out-of-band
    number.
  - **Promote gate** — the winner must (a) differ from baseline, (b) beat
    baseline Sortino by ≥ ``promote_margin`` (conservative default 0.15), and
    (c) have cleared the sweep's absolute risk floors (PSR/maxDD/CVaR — the
    sweep's ``gate_passes_per_cell``). Else: no write.
  - **Rollback + regression auto-revert** — ``config/portfolio_optimizer.json``
    is registered in ``rollback.CONFIG_KEYS``, so ``save_previous`` snapshots the
    prior config before overwrite (one-command revert) AND the weekly regression
    monitor auto-reverts it if next week's portfolio Sortino regresses.
  - **Kill-switch** — ``portfolio_optimizer_tuner.auto_apply_enabled`` (default
    True). Off → recommendation is shadow-archived only, live config untouched.
  - **Shadow audit** — every recommendation is shadow-archived regardless of
    whether it applies.
  - **Independent live nets** — even an applied config can't bypass the turnover
    governor (caps the first-day book move), the drawdown circuit breaker, or the
    position/sector caps, so the worst case is a mild, capped, revertible tilt —
    not a blowup.
"""

from __future__ import annotations

import json
import logging
from datetime import date

import boto3
from alpha_engine_lib.eval_artifacts import (
    eval_artifact_key,
    eval_latest_key,
    new_eval_run_id,
)

logger = logging.getLogger(__name__)

# ── Safeguards: what may be written, and within what bounds ──────────────────
# ONLY these two alpha-bearing knobs are ever auto-applied. Everything else in
# OPTIMIZER_CONFIG_DEFAULTS (sector/position caps, turnover governor, drawdown
# circuit breaker, vol target) is operator-owned and never touched here.
WRITABLE_PARAMS: tuple[str, ...] = ("risk_aversion", "tcost_bps")
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "risk_aversion": (3.0, 10.0),   # λ — variance-penalty strength
    "tcost_bps": (1.0, 20.0),       # turnover penalty (bps per $1 traded)
}
# Generic public defaults above are the frozen reference band. The OPERATING
# risk-aversion floor is a private risk-policy variable: override it via the
# gitignored tuner config (`portfolio_optimizer_tuner.risk_aversion_floor`) — see
# _effective_param_bounds — so an aggressive floor (e.g. 1.0) never ships in the
# public repo (divergence policy: alpha-bearing values stay private). MUST stay
# in lockstep with the executor read-side override
# executor/optimizer_shadow.py (portfolio_optimizer.tuner_risk_aversion_floor).

S3_PARAMS_KEY = "config/portfolio_optimizer.json"
S3_SHADOW_PREFIX = "config/portfolio_optimizer_shadow_history"
S3_HISTORY_PREFIX = "config/portfolio_optimizer_history"
ROLLBACK_CONFIG_TYPE = "portfolio_optimizer"

# Conservative default: a challenger must beat the live baseline's Sortino by
# this fraction before it overwrites live optimizer config. High by design —
# on the short predictions-archive window the regression-monitor auto-revert is
# the second net, so the bar to flip live in the first place is deliberately steep.
_DEFAULT_PROMOTE_MARGIN = 0.15
# Floor for the |baseline Sortino| denominator so a near-zero baseline can't
# manufacture an infinite "improvement".
_MARGIN_DENOM_FLOOR = 0.05


def _tuner_cfg(config: dict | None) -> dict:
    return (config or {}).get("portfolio_optimizer_tuner", {}) or {}


def _effective_param_bounds(config: dict | None) -> dict[str, tuple[float, float]]:
    """PARAM_BOUNDS with the risk_aversion floor overridden from the PRIVATE
    tuner config (`portfolio_optimizer_tuner.risk_aversion_floor`, gitignored).
    Public default floor (3.0) ships in the repo; the operating floor (e.g. 1.0,
    a more aggressive posture) lives only in private config."""
    bounds = dict(PARAM_BOUNDS)
    floor = _tuner_cfg(config).get("risk_aversion_floor")
    if floor is not None:
        lo, hi = bounds["risk_aversion"]
        bounds["risk_aversion"] = (float(floor), hi)
    return bounds


def _clamp(params: dict, config: dict | None = None) -> tuple[dict, list[str]]:
    """Clamp each writable param to its (private-config-resolved) bound; return
    (clamped, notes). A clamp is a loud event — it means the sweep recommended an
    out-of-band value."""
    bounds = _effective_param_bounds(config)
    clamped: dict = {}
    notes: list[str] = []
    for k, v in params.items():
        lo, hi = bounds.get(k, (None, None))
        if lo is None:
            clamped[k] = v
            continue
        cv = min(max(float(v), lo), hi)
        clamped[k] = cv
        if cv != float(v):
            notes.append(f"{k}: {v} → clamped to {cv} (band [{lo}, {hi}])")
    return clamped, notes


def recommend(sweep_report: dict, *, config: dict | None = None) -> dict:
    """Decide whether the weekly optimizer-param sweep's winner should overwrite
    live optimizer config, applying the promote gate + writable-set + clamps.

    ``sweep_report`` is the dict from ``run_optimizer_param_sweep`` (or its stage
    payload). Returns ``{"status": "ok", "recommended_params", ...}`` when a
    promotable winner clears the gate, else ``{"status": <reason>, ...}`` with no
    params — the caller's ``apply`` no-ops on a non-"ok" status."""
    if not sweep_report or sweep_report.get("status") not in (None, "ok"):
        return {"status": "blocked", "reason": f"sweep status={sweep_report.get('status')!r}"}

    winner = sweep_report.get("winner_name")
    baseline = sweep_report.get("baseline_name")
    cells = sweep_report.get("cells", {}) or {}
    if not winner:
        return {"status": "no_change", "reason": "no cell cleared the sweep gate"}
    if winner == baseline:
        return {"status": "no_change", "reason": "baseline cell is the winner"}

    win_m = cells.get(winner, {}) or {}
    base_m = cells.get(baseline, {}) or {}
    win_s = win_m.get("sortino_ratio")
    base_s = base_m.get("sortino_ratio")
    if win_s is None or base_s is None:
        return {"status": "blocked", "reason": "winner/baseline Sortino missing"}

    margin = (win_s - base_s) / max(abs(base_s), _MARGIN_DENOM_FLOOR)
    promote_margin = float(_tuner_cfg(config).get("promote_margin", _DEFAULT_PROMOTE_MARGIN))
    if margin < promote_margin:
        return {
            "status": "blocked",
            "reason": (
                f"insufficient margin: winner Sortino {win_s:.3f} vs baseline "
                f"{base_s:.3f} = {margin:+.1%} < promote_margin {promote_margin:.0%}"
            ),
            "winner_name": winner, "baseline_name": baseline, "margin": margin,
        }

    # Writable-set: take ONLY the allowlisted knobs from the winner's cfg.
    win_cfg = win_m.get("cell_cfg", {}) or {}
    raw = {k: win_cfg[k] for k in WRITABLE_PARAMS if k in win_cfg}
    if not raw:
        return {"status": "blocked", "reason": "winner cfg has no writable params"}
    recommended, clamp_notes = _clamp(raw, config)

    return {
        "status": "ok",
        "recommended_params": recommended,
        "winner_name": winner,
        "baseline_name": baseline,
        "winner_sortino": win_s,
        "baseline_sortino": base_s,
        "margin": margin,
        "promote_margin": promote_margin,
        "clamp_notes": clamp_notes,
    }


def apply(result: dict, bucket: str, *, config: dict | None = None) -> dict:
    """Write the recommended optimizer params to live S3 — flag-gated, clamped,
    rollback-snapshotted, shadow-audited. Returns ``{"applied": bool, ...}``.

    Live write happens only when ``result["status"] == "ok"`` AND
    ``portfolio_optimizer_tuner.auto_apply_enabled`` (default True). Otherwise the
    recommendation is shadow-archived only (audit trail) and live config is
    untouched."""
    status = result.get("status")
    recommended = result.get("recommended_params", {}) or {}
    auto_apply = bool(_tuner_cfg(config).get("auto_apply_enabled", True))

    if status != "ok" or not recommended:
        return {"applied": False, "reason": f"no promotable recommendation (status={status})"}

    # Defensive re-clamp at the write boundary (never trust an upstream value).
    recommended, clamp_notes = _clamp(recommended, config)

    payload = {
        **recommended,
        "updated_at": str(date.today()),
        "winner_name": result.get("winner_name"),
        "baseline_name": result.get("baseline_name"),
        "winner_sortino": result.get("winner_sortino"),
        "baseline_sortino": result.get("baseline_sortino"),
        "margin": result.get("margin"),
        "clamp_notes": (result.get("clamp_notes") or []) + clamp_notes,
        "writable_params": list(WRITABLE_PARAMS),
        "param_bounds": {k: list(v) for k, v in PARAM_BOUNDS.items()},
    }
    body = json.dumps(payload, indent=2)
    s3 = boto3.client("s3")

    # Always shadow-archive the recommendation (audit), regardless of apply.
    run_id = new_eval_run_id()
    try:
        s3.put_object(Bucket=bucket, Key=eval_artifact_key(S3_SHADOW_PREFIX, run_id),
                      Body=body, ContentType="application/json")
        s3.put_object(Bucket=bucket, Key=eval_latest_key(S3_SHADOW_PREFIX),
                      Body=body, ContentType="application/json")
    except Exception as e:  # noqa: BLE001 — audit is best-effort
        logger.warning("portfolio-optimizer params shadow archive failed (non-fatal): %s", e)

    if not auto_apply:
        logger.info(
            "portfolio-optimizer params NOT applied (auto_apply_enabled=False) — "
            "shadow-archived only: %s", recommended,
        )
        return {"applied": False, "reason": "auto_apply_enabled=False (shadow only)",
                "params": recommended, "shadow_only": True}

    # Snapshot the prior live config for one-command + regression auto-rollback
    # (config/portfolio_optimizer.json is registered in rollback.CONFIG_KEYS).
    from optimizer.rollback import save_previous
    save_previous(bucket, ROLLBACK_CONFIG_TYPE)

    try:
        s3.put_object(Bucket=bucket, Key=S3_PARAMS_KEY, Body=body, ContentType="application/json")
        logger.info("Portfolio-optimizer params updated in live S3: %s", recommended)
    except Exception as e:
        logger.error("CRITICAL: failed to write portfolio-optimizer params to S3: %s", e)
        return {"applied": False, "reason": f"S3 write failed: {e}"}

    try:
        history_key = eval_artifact_key(S3_HISTORY_PREFIX, run_id)
        s3.put_object(Bucket=bucket, Key=history_key, Body=body, ContentType="application/json")
        s3.put_object(Bucket=bucket, Key=eval_latest_key(S3_HISTORY_PREFIX),
                      Body=body, ContentType="application/json")
    except Exception as e:  # noqa: BLE001
        logger.warning("portfolio-optimizer params history archive failed (non-fatal): %s", e)

    return {
        "applied": True,
        "params": recommended,
        "winner_name": result.get("winner_name"),
        "margin": result.get("margin"),
        "clamp_notes": payload["clamp_notes"],
    }
