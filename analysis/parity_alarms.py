"""parity_alarms.py — Leg (g) of the L4593 backtester correctness battery:
tolerance bands + step-change alarms on the parity deltas.

WHY THIS EXISTS
---------------
``pit_parity`` (and ``parity_replay``) emit per-metric deltas between the
point-in-time walk-forward run and the legacy look-ahead run — the contamination
magnitude on the skilled-risk basket (Sortino / PSR / CVaR / max-DD) plus the
log-domain cumulative-return and SPY-relative alpha headlines. Today that delta is
``observational: True`` with a single |ΔSortino| ≥ 0.10 materiality flag and no
run-over-run tracking: a parity regression only surfaces if someone reads the
Saturday report.

This module adds the missing alarm layer:
- **Tolerance bands** — a per-metric absolute band; a delta outside its band is a
  breach (generalizes the lone ΔSortino materiality threshold to every basket
  metric).
- **Step-change detection** — compares this run's delta to the prior run's delta;
  a sudden jump beyond a per-metric step band is a breach even when the absolute
  level is still inside its band (catches a parity gap that is rapidly widening).
- **observe → paging** — defaults to OBSERVE (compute the verdict, log, return it;
  page nobody). When ``paging_enabled=True`` a breach fans out to SNS + Telegram
  via the canonical ``nousergon_lib.alerts.publish`` primitive (mirrors
  ``analysis.stance_distribution._publish_drift_alert``). This is the graduation
  path: soak in observe, flip to paging once the band thresholds are validated
  against live Saturday cadence.

The load-bearing surface in observe mode is the returned verdict dict + the WARN
log — the publish is best-effort. See [[feedback_no_silent_fails]] and
[[feedback_sota_institutional_default_no_shortcuts]].
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_ALERT_DISABLED_ENV_VAR = "ALPHA_ENGINE_PARITY_ALARM_DISABLED"

# Metrics carried on the pit_parity ``delta_pit_minus_current`` basket. Sortino's
# band matches the existing materiality threshold; the others are seeded to
# sensible defaults to be tightened against live Saturday cadence during soak.
DEFAULT_TOLERANCE_BANDS: dict[str, float] = {
    "sortino_ratio": 0.10,    # matches build_contamination_report materiality
    "psr": 0.10,              # probability units
    "cvar_95": 0.05,          # daily-return fraction
    "max_drawdown": 0.05,     # fraction
    "log_cum_return": 0.02,   # log-domain cumulative (≈ 2% contamination)
    "total_alpha": 0.02,      # SPY-relative headline (fraction)
}

# A run-over-run JUMP this large is a breach even if the level is still in band.
# Default = 2× the band (a parity gap doubling its tolerance in one week).
DEFAULT_STEP_BANDS: dict[str, float] = {k: 2.0 * v for k, v in DEFAULT_TOLERANCE_BANDS.items()}


def _truthy(val: str | None) -> bool:
    return (val or "").strip().lower() in ("1", "true", "yes", "on")


def evaluate_tolerance_bands(
    delta: dict,
    bands: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Per-metric absolute-band check on a parity delta.

    ``delta`` maps metric → (pit − current) value or None. Returns
    ``{metric: {"delta", "band", "breach"}}`` for every metric that has both a
    non-None delta and a configured band. A None delta is skipped (can't assess).
    """
    bands = bands or DEFAULT_TOLERANCE_BANDS
    out: dict[str, dict] = {}
    for metric, band in bands.items():
        d = delta.get(metric)
        if d is None:
            continue
        d = float(d)
        out[metric] = {"delta": d, "band": band, "breach": abs(d) > band}
    return out


def evaluate_step_change(
    delta: dict,
    prior_delta: dict | None,
    step_bands: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Per-metric run-over-run step-change check.

    Compares ``|delta - prior_delta|`` to the per-metric step band. Returns
    ``{metric: {"step", "band", "breach"}}`` for metrics present (non-None) on
    BOTH runs. Empty when ``prior_delta`` is None (first run — no baseline).
    """
    if prior_delta is None:
        return {}
    step_bands = step_bands or DEFAULT_STEP_BANDS
    out: dict[str, dict] = {}
    for metric, band in step_bands.items():
        cur, prev = delta.get(metric), prior_delta.get(metric)
        if cur is None or prev is None:
            continue
        step = abs(float(cur) - float(prev))
        out[metric] = {"step": step, "band": band, "breach": step > band}
    return out


def evaluate_parity_alarms(
    delta: dict,
    prior_delta: dict | None = None,
    *,
    bands: dict[str, float] | None = None,
    step_bands: dict[str, float] | None = None,
    paging_enabled: bool = False,
    publish_fn: Optional[Callable[..., object]] = None,
    run_date: str | None = None,
    source: str = "alpha-engine-backtester/analysis/parity_alarms.py",
) -> dict:
    """Evaluate tolerance-band + step-change alarms on a parity delta.

    Args:
        delta:          this run's ``delta_pit_minus_current`` (metric → value/None).
        prior_delta:    the previous run's delta, for step-change (None = first run).
        paging_enabled: OBSERVE when False (default) — compute + log + return, page
                        nobody. When True, a breach fans out an alert (suppressed
                        by env ``ALPHA_ENGINE_PARITY_ALARM_DISABLED``).
        publish_fn:     injectable ``callable(message, severity, source)`` for tests;
                        defaults to ``nousergon_lib.alerts.publish`` (lazy import).

    Returns a verdict dict: ``{mode, status, band_breaches, step_breaches,
    n_breaches, breached_metrics, paged}``. ``status`` is ``"ok"`` or ``"breach"``.
    """
    band_results = evaluate_tolerance_bands(delta, bands)
    step_results = evaluate_step_change(delta, prior_delta, step_bands)

    band_breaches = {m: r for m, r in band_results.items() if r["breach"]}
    step_breaches = {m: r for m, r in step_results.items() if r["breach"]}
    breached = sorted(set(band_breaches) | set(step_breaches))
    n_breaches = len(breached)
    status = "breach" if n_breaches else "ok"
    mode = "paging" if paging_enabled else "observe"

    verdict = {
        "mode": mode,
        "status": status,
        "run_date": run_date,
        "band_breaches": band_breaches,
        "step_breaches": step_breaches,
        "n_breaches": n_breaches,
        "breached_metrics": breached,
        "paged": False,
    }

    if status == "breach":
        logger.warning(
            "[parity_alarms] %s breach on %d metric(s): %s (mode=%s)",
            "PAGING" if paging_enabled else "OBSERVE", n_breaches, breached, mode,
        )
        if paging_enabled:
            verdict["paged"] = _publish_parity_alarm(verdict, source, publish_fn)

    return verdict


def _format_alarm_message(verdict: dict, source: str) -> str:
    lines = []
    for m, r in verdict["band_breaches"].items():
        lines.append(f"{m} Δ={r['delta']:+.4f} (band ±{r['band']})")
    for m, r in verdict["step_breaches"].items():
        lines.append(f"{m} step={r['step']:.4f} (step band {r['band']})")
    return (
        f"pit_parity alarm on {verdict.get('run_date') or '?'}: "
        f"{verdict['n_breaches']} metric(s) breached — {', '.join(verdict['breached_metrics'])}. "
        + " | ".join(lines)
        + ". Parity contamination outside tolerance; review before the flip gate."
    )


def _publish_parity_alarm(
    verdict: dict,
    source: str,
    publish_fn: Optional[Callable[..., object]],
) -> bool:
    """Best-effort fan-out of a parity breach to SNS + Telegram. Returns True iff a
    publish was attempted (env opt-out / import failure → False).

    Mirrors ``analysis.stance_distribution._publish_drift_alert``: the load-bearing
    surface is the WARN log + returned verdict; transport failures swallow at WARN.
    """
    if _truthy(os.environ.get(_ALERT_DISABLED_ENV_VAR)):
        logger.info("[parity_alarms] paging suppressed by %s", _ALERT_DISABLED_ENV_VAR)
        return False

    message = _format_alarm_message(verdict, source)

    if publish_fn is None:
        try:
            from nousergon_lib import alerts  # noqa: PLC0415 — lazy import
        except ImportError as e:
            logger.warning(
                "[parity_alarms] paging skipped — nousergon_lib.alerts "
                "unavailable: %s", e,
            )
            return False
        publish_fn = alerts.publish

    try:
        publish_fn(message, severity="error", source=source)
        return True
    except Exception as e:  # noqa: BLE001 — best-effort; verdict+log is load-bearing
        logger.warning("[parity_alarms] paging publish failed (swallowed): %s", e)
        return False
