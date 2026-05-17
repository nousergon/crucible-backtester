"""
Cutover-gate validator for the portfolio-optimizer arc — PR 4 of
alpha-engine-docs/private/portfolio-optimizer-260511.md.

Consumes the side-by-side metric dict produced by
``analysis.portfolio_optimizer_backtest.compare_to_legacy`` and returns a
per-criterion pass/fail report plus an overall verdict. Gating anchors on
the skilled-risk basket (Sortino + PSR ≥ 0.95 + CVaR + max DD) per
[[anchor_gates_on_skilled_risk_not_sharpe]] / [[evaluator_revamp_skilled_risk]].
Raw Sharpe and α vs SPY remain observability/presentation-only.

Pure function — no I/O, no S3 calls. The caller (backtest.py's
``--mode portfolio-optimizer-backtest``) is responsible for orchestrating
the backtest, calling compare_to_legacy, and persisting the gate report
to S3.

Decision rule:
    pass = all hard gates pass
    All criteria are reported individually so the operator (and PR 5
    cutover decision) can see which gates passed/failed even when the
    overall verdict is FAIL. Optional gates (those whose threshold is None
    because legacy_metrics was absent) report status "skipped_no_legacy"
    and do NOT block the overall verdict — that path is for first-run
    operator inspection, not gated promotion.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)


_PASS = "pass"
_FAIL = "fail"
_SKIPPED = "skipped_no_legacy"


@dataclass(frozen=True)
class GateResult:
    name: str
    status: str
    value: float | None
    threshold: float | list | None
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_gate(comparison: dict) -> dict:
    """
    Evaluate the portfolio-optimizer cutover gate.

    Args:
        comparison: output of
            ``analysis.portfolio_optimizer_backtest.compare_to_legacy``.
            Must contain keys ``optimizer``, ``legacy``, ``gate_thresholds``.

    Returns:
        Gate report dict:
            {
              "verdict": "pass" | "fail",
              "signal_source": "synthetic" | "production" | "unknown",
              "n_pass": int,
              "n_fail": int,
              "n_skipped": int,
              "criteria": [GateResult.as_dict(), ...],
              "summary": str,
            }
    """
    if not isinstance(comparison, dict):
        raise TypeError(f"comparison must be dict, got {type(comparison).__name__}")
    for required in ("optimizer", "gate_thresholds"):
        if required not in comparison:
            raise KeyError(f"comparison missing required key: {required!r}")

    optimizer = comparison["optimizer"] or {}
    thresholds = comparison["gate_thresholds"] or {}
    has_legacy = comparison.get("legacy") is not None
    signal_source = comparison.get("signal_source", "unknown")

    criteria = [
        _check_sortino(optimizer, thresholds, has_legacy),
        _check_psr(optimizer, thresholds),
        _check_max_drawdown(optimizer, thresholds),
        _check_cvar(optimizer, thresholds),
        _check_turnover(optimizer, thresholds, has_legacy),
        _check_tracking_error(optimizer, thresholds),
        _check_active_share(optimizer, thresholds),
    ]

    n_pass = sum(1 for c in criteria if c.status == _PASS)
    n_fail = sum(1 for c in criteria if c.status == _FAIL)
    n_skipped = sum(1 for c in criteria if c.status == _SKIPPED)

    verdict = _PASS if n_fail == 0 and n_pass > 0 else _FAIL
    summary = _build_summary(verdict, n_pass, n_fail, n_skipped, criteria,
                             signal_source)

    return {
        "verdict": verdict,
        "signal_source": signal_source,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_skipped": n_skipped,
        "criteria": [c.as_dict() for c in criteria],
        "summary": summary,
    }


def _check_sortino(optimizer: dict, thresholds: dict, has_legacy: bool) -> GateResult:
    name = "sortino_min"
    threshold = thresholds.get("sortino_min")
    value = optimizer.get("sortino_ratio")
    if threshold is None or not has_legacy:
        return GateResult(name, _SKIPPED, value, threshold,
                          "No legacy baseline available")
    if value is None:
        return GateResult(name, _FAIL, None, threshold,
                          "Optimizer sortino_ratio missing")
    status = _PASS if value >= threshold else _FAIL
    return GateResult(name, status, float(value), float(threshold),
                      f"sortino_opt={value:.4f} {'≥' if status == _PASS else '<'} {threshold:.4f}")


def _check_psr(optimizer: dict, thresholds: dict) -> GateResult:
    name = "psr_min"
    threshold = thresholds.get("psr_min")
    value = optimizer.get("psr")
    if threshold is None:
        return GateResult(name, _SKIPPED, value, threshold,
                          "No PSR threshold configured")
    if value is None:
        return GateResult(name, _SKIPPED, None, threshold,
                          "PSR not computed (insufficient daily returns)")
    status = _PASS if value >= threshold else _FAIL
    return GateResult(name, status, float(value), float(threshold),
                      f"psr_opt={value:.4f} {'≥' if status == _PASS else '<'} {threshold:.4f}")


def _check_max_drawdown(optimizer: dict, thresholds: dict) -> GateResult:
    name = "max_drawdown_floor"
    threshold = thresholds.get("max_drawdown_floor")
    value = optimizer.get("max_drawdown")
    # ROADMAP L124: absolute risk floor — applies with or without a legacy
    # baseline. Only skip if no floor is configured at all.
    if threshold is None:
        return GateResult(name, _SKIPPED, value, threshold,
                          "No absolute max-drawdown floor configured")
    if value is None:
        return GateResult(name, _FAIL, None, threshold,
                          "Optimizer max_drawdown missing")
    status = _PASS if value >= threshold else _FAIL
    return GateResult(name, status, float(value), float(threshold),
                      f"max_dd_opt={value:.4f} {'≥' if status == _PASS else '<'} {threshold:.4f} (less-negative=better)")


def _check_cvar(optimizer: dict, thresholds: dict) -> GateResult:
    name = "cvar_95_floor"
    threshold = thresholds.get("cvar_95_floor")
    value = optimizer.get("cvar_95")
    # ROADMAP L124: absolute tail-risk floor — applies with or without a
    # legacy baseline. Only skip if no floor is configured at all.
    if threshold is None:
        return GateResult(name, _SKIPPED, value, threshold,
                          "No absolute CVaR(95) floor configured")
    if value is None:
        return GateResult(name, _FAIL, None, threshold,
                          "Optimizer cvar_95 missing")
    status = _PASS if value >= threshold else _FAIL
    return GateResult(name, status, float(value), float(threshold),
                      f"cvar95_opt={value:.4f} {'≥' if status == _PASS else '<'} {threshold:.4f} (less-negative=better)")


def _check_turnover(optimizer: dict, thresholds: dict, has_legacy: bool) -> GateResult:
    name = "turnover_max"
    threshold = thresholds.get("turnover_max")
    value = optimizer.get("turnover_one_way_ann")
    if threshold is None or not has_legacy:
        return GateResult(name, _SKIPPED, value, threshold,
                          "No legacy baseline available")
    if value is None:
        return GateResult(name, _FAIL, None, threshold,
                          "Optimizer turnover_one_way_ann missing")
    status = _PASS if value <= threshold else _FAIL
    return GateResult(name, status, float(value), float(threshold),
                      f"turnover_opt={value:.4f} {'≤' if status == _PASS else '>'} {threshold:.4f}")


def _check_tracking_error(optimizer: dict, thresholds: dict) -> GateResult:
    name = "tracking_error_range"
    rng = thresholds.get("tracking_error_range")
    value = optimizer.get("tracking_error_ann")
    if rng is None:
        return GateResult(name, _SKIPPED, value, rng,
                          "No tracking-error range configured")
    if value is None:
        return GateResult(name, _SKIPPED, None, rng,
                          "Tracking error not computed (insufficient SPY-aligned days)")
    low, high = float(rng[0]), float(rng[1])
    status = _PASS if low <= value <= high else _FAIL
    return GateResult(name, status, float(value), [low, high],
                      f"TE={value:.4f} {'∈' if status == _PASS else '∉'} [{low:.2f}, {high:.2f}]")


def _check_active_share(optimizer: dict, thresholds: dict) -> GateResult:
    name = "active_share_range"
    rng = thresholds.get("active_share_range")
    value = optimizer.get("mean_active_share")
    if rng is None:
        return GateResult(name, _SKIPPED, value, rng,
                          "No active-share range configured")
    if value is None:
        return GateResult(name, _SKIPPED, None, rng,
                          "Active share not computed (no rebalances)")
    low, high = float(rng[0]), float(rng[1])
    status = _PASS if low <= value <= high else _FAIL
    return GateResult(name, status, float(value), [low, high],
                      f"AS={value:.4f} {'∈' if status == _PASS else '∉'} [{low:.2f}, {high:.2f}]")


def _build_summary(
    verdict: str, n_pass: int, n_fail: int, n_skipped: int,
    criteria: list[GateResult], signal_source: str = "unknown",
) -> str:
    lines = [
        f"GATE VERDICT: {verdict.upper()}  ({n_pass} pass / {n_fail} fail / {n_skipped} skipped)",
        f"SIGNAL SOURCE: {signal_source}  "
        f"(thresholds interpretable only against the matching input distribution)",
        "",
    ]
    for c in criteria:
        marker = {_PASS: "✓", _FAIL: "✗", _SKIPPED: "—"}[c.status]
        lines.append(f"  {marker} {c.name:24s} {c.status:18s} {c.note}")
    return "\n".join(lines)


def gate_passed(report: dict) -> bool:
    """Convenience predicate: True iff overall verdict is 'pass'."""
    return report.get("verdict") == _PASS


def run_gate_against_predictor_backtest(
    config: dict,
    legacy_metrics: dict | None = None,
    rebalance_freq_days: int = 5,
    universe_cap: int = 30,
    signal_source: str = "synthetic",
) -> dict:
    """
    End-to-end gate runner — orchestrates {predictor_backtest |
    production_signal_backtest} → optimizer backtest → compare_to_legacy →
    evaluate_gate.

    Args:
        config: full backtester config dict (signals_bucket, executor_paths,
            predictor_paths, predictor_backtest section, etc.).
        legacy_metrics: optional dict of legacy backtest metrics for
            side-by-side comparison. When None, the gate skips only the
            legacy-relative criteria (sortino_min, turnover_max) while
            still checking ALL absolute criteria (psr_min,
            max_drawdown_floor, cvar_95_floor, tracking_error_range,
            active_share_range) — ROADMAP L124: risk floors are absolute,
            not gated on a legacy baseline.
        rebalance_freq_days: passthrough to run_optimizer_backtest.
        universe_cap: passthrough to run_optimizer_backtest.
        signal_source: "synthetic" (default — 10y predictor-GBM replay over
            the full universe) or "production" (ROADMAP L124 PR 2 — the
            deployed research cohort from signals/{date}/signals.json +
            predictor/predictions/{date}.json). The optimizer kernel is
            identical for both; only the input distribution differs. The
            enhanced-index TE/active-share bands are calibrated for the
            production distribution — a synthetic-source verdict's
            band criteria should be read as a stress test, not a gate.

    Returns:
        {
            "comparison": <compare_to_legacy output>,
            "gate_report": <evaluate_gate output>,
            "optimizer_diagnostics": list[dict],
            "n_rebalances": int,
            "n_solver_failures": int,
        }

    Caller (Saturday SF integration via alpha-engine-config, or ad-hoc CLI)
    is responsible for persisting the result to S3 and acting on the
    gate verdict.
    """
    import os
    from analysis.portfolio_optimizer_backtest import (
        compare_to_legacy,
        run_optimizer_backtest,
    )

    if signal_source not in ("synthetic", "production"):
        raise ValueError(
            f"signal_source must be 'synthetic' or 'production', got {signal_source!r}"
        )

    executor_paths = config.get("executor_paths", [])
    if isinstance(executor_paths, str):
        executor_paths = [executor_paths]
    executor_path = next((p for p in executor_paths if os.path.isdir(p)), None)
    if not executor_path:
        raise ValueError(
            f"executor_paths not found on disk: {executor_paths}. "
            "Add the alpha-engine repo root to executor_paths in config.yaml."
        )

    if signal_source == "production":
        from synthetic.production_signal_backtest import (
            build_production_signal_inputs,
        )

        logger.info(
            "Gate runner: building inputs from the PRODUCTION archive "
            "(signals/ ∪ predictor/predictions/) — measures deployed behavior"
        )
        pred_result = build_production_signal_inputs(config)
    else:
        from synthetic.predictor_backtest import run as run_predictor_pipeline

        logger.info(
            "Gate runner: invoking synthetic predictor backtest "
            "with keep_predictions=True"
        )
        pred_result = run_predictor_pipeline(config, keep_predictions=True)

    if pred_result.get("status") != "ok":
        return {
            "comparison": None,
            "gate_report": {
                "verdict": _FAIL,
                "summary": (
                    f"{signal_source} input producer failed: "
                    f"status={pred_result.get('status')} "
                    f"({pred_result.get('error', 'no detail')})"
                ),
                "criteria": [],
                "n_pass": 0,
                "n_fail": 1,
                "n_skipped": 0,
            },
            "optimizer_diagnostics": [],
            "n_rebalances": 0,
            "n_solver_failures": 0,
            "signal_source": signal_source,
        }

    logger.info(
        "Gate runner: invoking optimizer backtest "
        f"(rebalance_freq={rebalance_freq_days}d, universe_cap={universe_cap})"
    )
    opt_result = run_optimizer_backtest(
        predictions_by_date=pred_result["predictions_by_date"],
        price_matrix=pred_result["price_matrix"],
        spy_prices=pred_result["spy_prices"],
        sector_map=pred_result["sector_map"],
        executor_path=executor_path,
        rebalance_freq_days=rebalance_freq_days,
        universe_cap=universe_cap,
    )

    # The optimizer kernel (executor.portfolio_optimizer.solve_target_weights)
    # is identical for both sources; only the input distribution differs.
    # signal_source is threaded into the verdict so band criteria are
    # interpretable (L124 PR 1 added the discriminator; PR 2 makes it real).
    comparison = compare_to_legacy(
        opt_result.metrics, legacy_metrics, signal_source=signal_source
    )
    report = evaluate_gate(comparison)

    logger.info(
        f"Gate runner: verdict={report['verdict']} "
        f"({report['n_pass']} pass / {report['n_fail']} fail / {report['n_skipped']} skipped)"
    )

    return {
        "comparison": comparison,
        "gate_report": report,
        "optimizer_diagnostics": opt_result.diagnostics_per_rebalance,
        "n_rebalances": opt_result.n_rebalances,
        "n_solver_failures": opt_result.n_solver_failures,
        "signal_source": signal_source,
        "production_window": pred_result.get("production_window"),
    }
