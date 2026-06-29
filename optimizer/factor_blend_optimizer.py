"""
factor_blend_optimizer.py — Recommend per-regime factor-blend stance-weight
reorderings from realized (regime, stance) outcomes.

Auto-apply companion to the observability-only
``analysis/factor_blend_sensitivity.py`` (crucible-backtester #206/#207).
Where the sensitivity analyzer DETECTS regimes whose CONFIGURED stance
ranking disagrees with the REALIZED Sortino ranking among trustworthy
(regime, stance) cells, this module turns a *persistent* such mismatch into
a concrete recommended override of the research-side
``aggregator.factor_blend`` regime weights, and (under a default-off,
reproduction-gated apply path) writes that override to S3 for crucible-research
to consume.

Implements nousergon/alpha-engine-config#748 (backtester half). Modeled
deliberately on ``optimizer/tech_weight_ablation.py`` so the evaluator wiring,
shadow/reproduction contract, and status-dict shape are uniform across
optimizers; it does NOT re-derive the heavy holdout/walk-forward machinery of
``executor_optimizer.py`` because the recommendation here is a discrete
weight-REORDERING (swap the realized-top stance into the top weight slot),
not a continuous param sweep needing PBO/holdout cross-validation.

Two-stage activation (mirrors tech_weight_ablation / executor_optimizer):

  1. ``use_factor_blend_target=True`` (default false) — every weekly run that
     produces an ``ok`` recommendation writes a shadow payload to
     ``config/factor_blend_params_shadow_history/{run_id}.json`` (+
     ``latest.json`` sidecar). Live config is untouched. Pure observability:
     compare shadow trajectories against the live ``scoring.yaml`` baselines
     week-over-week.

  2. ``enforce_factor_blend=True`` (default false) — in addition to the
     shadow archive, the apply path writes the live
     ``config/factor_blend_params.json`` key **only if** the reproduction
     gate passes: the same recommended ``regime_weights`` payload must
     reproduce across the last ``_MIN_CONSECUTIVE_WEEKS`` shadow archives.
     This prevents a single noisy Saturday from flipping live regime weights.

Inputs: the already-computed ``factor_blend_sensitivity.build_sensitivity_report``
result (passed in by evaluate.py — no recomputation). Produces
``insufficient_data`` until at least one regime has a trustworthy mismatch
(per-cell trustworthiness is itself gated at
``factor_blend_sensitivity.MIN_TRUSTWORTHY_SAMPLES`` = 20 samples per
(regime, stance) cell, ~Week 8+).

Returns the standard backtester-evaluator status dict so the existing
``CompletenessTracker.run_module`` pattern handles it without bespoke wiring.

UNVALIDATED (draft): the guardrail thresholds below — the realized-Sortino
margin floor ``_MIN_SORTINO_MARGIN`` and the reproduction window
``_MIN_CONSECUTIVE_WEEKS`` — are first-cut values chosen to mirror the
sibling optimizers' posture; they should be confirmed against a real
multi-Saturday sensitivity-report history before ``enforce_factor_blend`` is
ever turned on. See the PR description for the exact validation command.
"""

from __future__ import annotations

import json
import logging
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)


# ── S3 contract (auto-apply) ─────────────────────────────────────────────────

S3_LIVE_KEY = "config/factor_blend_params.json"
S3_SHADOW_PREFIX = "config/factor_blend_params_shadow_history"

# Reproduction gate: live write only fires when the same recommended
# regime_weights payload reproduces across this many consecutive shadow
# archives. Mirrors tech_weight_ablation's reproduction window and the
# issue's "mismatch persists >= 2 consecutive Saturdays" framing — set to 2
# here (the issue's lower bound) rather than 4 because the mismatch flag is
# itself already gated on >=20-sample trustworthy cells; the persistence
# window guards against a single-week ranking flip, not against thin data.
# UNVALIDATED — confirm against real history before enforce is enabled.
_MIN_CONSECUTIVE_WEEKS = 2

# Minimum realized-Sortino margin (realized #1 stance Sortino minus the
# currently-configured #1 stance's realized Sortino, within the same regime)
# before a mismatch is worth acting on. A mismatch where the realized-top
# barely edges the config-top is noise; require a material gap. UNVALIDATED
# tuning call — first-cut value mirrors the spirit of executor_optimizer's
# min-improvement gate, applied on the Sortino axis.
_MIN_SORTINO_MARGIN = 0.10

# Module-level config ref — set by init_config() from evaluate.py.
_cfg: dict = {}


def init_config(config: dict) -> None:
    """Load the factor_blend_optimizer section from backtester config.

    Recognized keys (all default false):
      - ``use_factor_blend_target``: enable shadow-archive writes
      - ``enforce_factor_blend``: enable live writes (also requires the
        reproduction gate to pass)
      - ``auto_apply``: convenience alias kept false in
        alpha-engine-config/backtester/config.yaml; ``enforce_factor_blend``
        is the operative live-write flag.
    """
    global _cfg
    _cfg = config.get("factor_blend_optimizer", {}) or {}


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation
# ─────────────────────────────────────────────────────────────────────────────


def _stance_sortino(outcomes: pd.DataFrame, regime: str, stance: str) -> float | None:
    """Realized Sortino for one (regime, stance) cell, or None if absent."""
    sub = outcomes[
        (outcomes["market_regime"] == regime) & (outcomes["stance"] == stance)
    ]
    if sub.empty:
        return None
    val = sub["sortino"].iloc[0]
    return None if pd.isna(val) else float(val)


def _reweight_regime(
    current_weights: dict[str, float], new_top_stance: str
) -> dict[str, float] | None:
    """Return a copy of ``current_weights`` with ``new_top_stance`` swapped
    into the highest-weight slot.

    We preserve the EXISTING weight VALUES (the magnitudes are a separate
    tuning surface owned by scoring.yaml) and only permute WHICH stance holds
    which weight: the new top stance takes the current max weight, and the
    stance that previously held the max takes the new top stance's old weight.
    This is a minimal, reversible reordering — not a re-derivation of weight
    magnitudes — which is exactly what the sensitivity mismatch signal
    licenses (it ranks stances; it does not estimate optimal magnitudes).

    Keys are the ``*_score`` suffixed names as they appear in scoring.yaml /
    DEFAULT_REGIME_WEIGHTS. ``new_top_stance`` is the bare stance name
    (e.g. ``"quality"``). Returns None if the stance is not present or is
    already the top-weighted stance (no-op).
    """
    key = f"{new_top_stance}_score"
    if key not in current_weights:
        return None
    current_top_key = max(current_weights, key=lambda k: current_weights[k])
    if current_top_key == key:
        return None  # already top — nothing to reorder
    out = dict(current_weights)
    out[key], out[current_top_key] = out[current_top_key], out[key]
    return out


def recommend(sensitivity_report: dict, regime_weights: dict[str, dict[str, float]]) -> dict:
    """Turn a sensitivity report into per-regime weight-reordering recs.

    Args:
        sensitivity_report: output of
            ``factor_blend_sensitivity.build_sensitivity_report`` — must carry
            ``outcomes`` (per-(regime, stance) stats DataFrame) and
            ``mismatches`` (per-regime config-vs-realized comparison DataFrame).
        regime_weights: the CURRENTLY-configured regime weights
            (``{regime: {stance_score: weight}}``) — the same dict that was fed
            into the sensitivity report, so config_top stays consistent.

    Returns the standard status dict:
        status: "ok" | "insufficient_data" | "no_data" | "error"
        recommendations: {regime: {stance_score: weight}}  (only regimes with
                         an actionable, material, trustworthy mismatch)
        per_regime: list[dict] with the decision trace for each regime
    """
    if not isinstance(sensitivity_report, dict):
        return {"status": "error", "error": "sensitivity_report is not a dict"}
    if sensitivity_report.get("status") not in (None, "ok") and not sensitivity_report.get("has_data"):
        # The sensitivity module returns has_data; some callers stamp status.
        return {"status": "no_data", "reason": "sensitivity report has no data"}

    outcomes = sensitivity_report.get("outcomes")
    mismatches = sensitivity_report.get("mismatches")
    if not isinstance(outcomes, pd.DataFrame) or not isinstance(mismatches, pd.DataFrame):
        return {"status": "no_data", "reason": "sensitivity report missing outcomes/mismatches frames"}
    if outcomes.empty or mismatches.empty:
        return {"status": "no_data", "reason": "no trustworthy (regime, stance) cells yet"}

    recommendations: dict[str, dict[str, float]] = {}
    per_regime: list[dict] = []

    for _, row in mismatches.iterrows():
        regime = row["market_regime"]
        trace: dict = {
            "market_regime": regime,
            "config_top_stance": row.get("config_top_stance"),
            "realized_top_stance": row.get("realized_top_stance"),
            "n_trustworthy_cells": int(row.get("n_trustworthy_cells") or 0),
            "mismatch": row.get("mismatch"),
        }

        # No mismatch (or not yet computable) → keep current.
        if not row.get("mismatch"):
            trace["decision"] = "kept_current"
            trace["reason"] = (
                "no trustworthy mismatch"
                if row.get("mismatch") is False
                else "mismatch not yet computable (no trustworthy cells)"
            )
            per_regime.append(trace)
            continue

        config_top = row.get("config_top_stance")
        realized_top = row.get("realized_top_stance")
        if not realized_top or not config_top:
            trace["decision"] = "kept_current"
            trace["reason"] = "missing config/realized top stance"
            per_regime.append(trace)
            continue

        # Materiality gate: realized-top must beat config-top by a margin on
        # the realized Sortino axis within this regime.
        realized_top_sortino = _stance_sortino(outcomes, regime, realized_top)
        config_top_sortino = _stance_sortino(outcomes, regime, config_top)
        trace["realized_top_sortino"] = realized_top_sortino
        trace["config_top_sortino"] = config_top_sortino
        if realized_top_sortino is None or config_top_sortino is None:
            trace["decision"] = "kept_current"
            trace["reason"] = "config-top stance not trustworthy/realized in this regime"
            per_regime.append(trace)
            continue
        margin = realized_top_sortino - config_top_sortino
        trace["sortino_margin"] = margin
        if margin < _MIN_SORTINO_MARGIN:
            trace["decision"] = "kept_current"
            trace["reason"] = (
                f"sortino margin {margin:.3f} < floor {_MIN_SORTINO_MARGIN}"
            )
            per_regime.append(trace)
            continue

        current = regime_weights.get(regime) or regime_weights.get(regime.lower())
        if not current:
            trace["decision"] = "kept_current"
            trace["reason"] = f"no configured weights for regime {regime!r}"
            per_regime.append(trace)
            continue

        reweighted = _reweight_regime(current, realized_top)
        if reweighted is None:
            trace["decision"] = "kept_current"
            trace["reason"] = "realized-top stance absent from config / already top"
            per_regime.append(trace)
            continue

        recommendations[regime] = reweighted
        trace["decision"] = "switch"
        trace["new_weights"] = reweighted
        per_regime.append(trace)

    if not recommendations:
        return {
            "status": "insufficient_data",
            "reason": "no actionable trustworthy+material mismatch this cycle",
            "min_sortino_margin": _MIN_SORTINO_MARGIN,
            "per_regime": per_regime,
        }

    return {
        "status": "ok",
        "run_date": str(date.today()),
        "min_sortino_margin": _MIN_SORTINO_MARGIN,
        "recommendations": recommendations,
        "per_regime": per_regime,
        "n_regimes_with_recommendation": len(recommendations),
        # Recommendation-only; apply() is the cutover gate.
        "applied": False,
        "apply_note": (
            "see apply() — gated on use_factor_blend_target + "
            "enforce_factor_blend flags + reproduction gate"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Auto-apply path
# ─────────────────────────────────────────────────────────────────────────────


def _build_payload(result: dict) -> dict:
    """Translate a recommend() result into the S3 override payload that
    crucible-research's config loader reads as an override on
    ``aggregator.factor_blend``.

    Shape: ``{"regime_weights": {regime: {stance_score: weight}}, ...}``.
    Regimes with no recommendation are omitted — absent regime means
    "no override, use scoring.yaml default".
    """
    return {
        "regime_weights": result.get("recommendations") or {},
        "updated_at": str(date.today()),
        "source": "factor_blend_optimizer",
        "run_date": result.get("run_date"),
        "min_sortino_margin": result.get("min_sortino_margin"),
    }


def _read_recent_shadow_archives(s3, bucket: str, n: int) -> list[dict]:
    """Read up to ``n`` most-recent shadow archives, newest first.

    Mirrors tech_weight_ablation._read_recent_shadow_archives: lists the
    ``{prefix}/...json`` artifacts (skipping the ``latest.json`` sidecar),
    sorts by the YYMMDDHHMM-encoded key (which doubles as time order), and
    parses each. Missing/corrupt artifacts are skipped with a warning.
    """
    from botocore.exceptions import ClientError

    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=f"{S3_SHADOW_PREFIX}/")
    except ClientError as e:
        logger.warning(
            "[factor_blend_optimizer] shadow archive list failed (%s) — "
            "treating as no history available",
            type(e).__name__,
        )
        return []
    keys = sorted(
        (
            obj["Key"]
            for obj in (resp.get("Contents") or [])
            if obj["Key"].endswith(".json") and not obj["Key"].endswith("/latest.json")
        ),
        reverse=True,
    )[:n]

    out: list[dict] = []
    for key in keys:
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            out.append(json.loads(obj["Body"].read()))
        except Exception as e:  # noqa: BLE001 — skip unreadable, keep going
            logger.warning(
                "[factor_blend_optimizer] shadow archive %s unreadable (%s) — skipping",
                key,
                type(e).__name__,
            )
    return out


def _check_reproduction_gate(
    s3, bucket: str, current_payload: dict, *, min_consecutive: int = _MIN_CONSECUTIVE_WEEKS
) -> dict:
    """Pass = the same ``regime_weights`` payload reproduces across the last
    ``min_consecutive`` shadow archives.

    Returns ``{"passed": bool, "reason": str, "n_consecutive": int}``. One
    archive that disagrees breaks the streak — no tolerance for intermittent
    shadow drift, matching the issue's ">= consecutive Saturdays" framing.
    """
    archives = _read_recent_shadow_archives(s3, bucket, min_consecutive)
    if len(archives) < min_consecutive:
        return {
            "passed": False,
            "reason": (
                f"reproduction gate: only {len(archives)} prior shadow "
                f"archive(s) available; need {min_consecutive}"
            ),
            "n_consecutive": len(archives),
        }
    for i, archive in enumerate(archives):
        prior = archive.get("regime_weights") or {}
        if prior != current_payload:
            return {
                "passed": False,
                "reason": (
                    f"reproduction gate broken at archive[-{i + 1}]: "
                    f"regime_weights payload differs from this week"
                ),
                "n_consecutive": i,
            }
    return {
        "passed": True,
        "reason": (
            f"regime_weights payload reproduced across last {min_consecutive} shadow archives"
        ),
        "n_consecutive": min_consecutive,
    }


def apply(result: dict, bucket: str) -> dict:
    """Write the factor-blend recommendation to S3 under the two-stage
    activation contract documented at module top.

    - **Shadow** (``use_factor_blend_target=True``): canonical eval-style
      archive at ``config/factor_blend_params_shadow_history/{run_id}.json``
      + ``latest.json`` sidecar. Live config untouched.
    - **Live** (``use_factor_blend_target`` AND ``enforce_factor_blend`` AND
      reproduction gate passes): writes ``config/factor_blend_params.json``.
      Always also writes the shadow archive.

    Mirrors ``tech_weight_ablation.apply()`` / ``executor_optimizer.apply()``
    so the evaluator wiring is uniform.
    """
    import boto3
    from nousergon_lib.eval_artifacts import (
        eval_artifact_key,
        eval_latest_key,
        new_eval_run_id,
    )

    use_shadow = bool(_cfg.get("use_factor_blend_target", False))
    enforce = bool(_cfg.get("enforce_factor_blend", False))

    if not use_shadow:
        return {"applied": False, "reason": "use_factor_blend_target=False"}
    if result.get("status") != "ok":
        return {"applied": False, "reason": f"compute status={result.get('status')}"}

    payload = _build_payload(result)
    if not payload.get("regime_weights"):
        return {"applied": False, "reason": "no regime recommendation to apply"}
    body = json.dumps(payload, indent=2)

    s3 = boto3.client("s3")

    run_id = new_eval_run_id()
    shadow_key = eval_artifact_key(S3_SHADOW_PREFIX, run_id)
    shadow_latest_key = eval_latest_key(S3_SHADOW_PREFIX)
    try:
        s3.put_object(Bucket=bucket, Key=shadow_key, Body=body, ContentType="application/json")
        s3.put_object(
            Bucket=bucket, Key=shadow_latest_key, Body=body, ContentType="application/json"
        )
        logger.info(
            "[factor_blend_optimizer] shadow archive written: s3://%s/%s", bucket, shadow_key
        )
    except Exception as e:  # noqa: BLE001
        logger.error("[factor_blend_optimizer] shadow archive write failed: %s", e)
        return {"applied": False, "reason": f"shadow S3 write failed: {e}"}

    if not enforce:
        return {
            "applied": False,
            "reason": "shadow mode (enforce_factor_blend=False)",
            "shadow_key": shadow_key,
            "regime_weights": payload["regime_weights"],
        }

    # Live-write gate: reproduction across last N shadow archives (the current
    # week's archive was just written, so the gate reads N starting with this one).
    gate = _check_reproduction_gate(s3, bucket, payload["regime_weights"])
    if not gate["passed"]:
        return {
            "applied": False,
            "reason": gate["reason"],
            "shadow_key": shadow_key,
            "regime_weights": payload["regime_weights"],
            "reproduction_gate": gate,
        }

    try:
        s3.put_object(Bucket=bucket, Key=S3_LIVE_KEY, Body=body, ContentType="application/json")
        logger.info(
            "[factor_blend_optimizer] live config updated: s3://%s/%s — regime_weights=%s",
            bucket,
            S3_LIVE_KEY,
            payload["regime_weights"],
        )
    except Exception as e:  # noqa: BLE001
        logger.error("[factor_blend_optimizer] CRITICAL: live S3 write failed: %s", e)
        return {
            "applied": False,
            "reason": f"live S3 write failed: {e}",
            "shadow_key": shadow_key,
            "regime_weights": payload["regime_weights"],
        }

    return {
        "applied": True,
        "reason": "live config written + shadow archive recorded",
        "live_key": S3_LIVE_KEY,
        "shadow_key": shadow_key,
        "regime_weights": payload["regime_weights"],
        "reproduction_gate": gate,
    }
