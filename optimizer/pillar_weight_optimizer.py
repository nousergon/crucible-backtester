"""
pillar_weight_optimizer.py — SHADOW-ONLY per-pillar scoring-weight recommender.

Phase 6 of the attractiveness-pillars arc (alpha-engine-config#789). This module
sweeps candidate weight vectors over the canonical 6-pillar composite space
(quality / value / momentum / growth / stewardship / defensiveness) plus the
within-pillar ``qual_weight`` and the ``legacy_blend`` mix terms, reconstructs
each candidate's composite score from the persisted
``investment_thesis.composite_breakdown`` inputs, ranks the candidates by
Sortino-on-skilled-risk under a hard alpha-floor constraint, and RECOMMENDS the
best vector.

Design (audit 2026-05-22, **Option A**): this mirrors the sweep-then-rank
``recommend()`` pattern of :mod:`optimizer.executor_optimizer` (Sortino anchor +
``alpha_floor`` hard constraint + rank-and-recommend). It deliberately does NOT
reuse the correlation-blend pattern of :mod:`optimizer.weight_optimizer` — that
legacy module is left untouched as the sub-score (quant/qual) fallback.

Pillar shapes are the canonical nousergon-lib types
(:mod:`nousergon_lib.pillars` — ``PILLARS``, ``CompositeBreakdown``,
``PillarContribution``, ``LegacyComponentBlend``); this module reuses them rather
than re-deriving the composite arithmetic, so it scores against exactly the
breakdown the research module persists.

SHADOW MODE ONLY — HARD INVARIANT
---------------------------------
This optimizer NEVER writes the live scoring-weights config key
(``weight_optimizer.S3_WEIGHTS_KEY`` = ``config/scoring_weights.json``) and has
NO live-cutover code path. :func:`apply` writes ONLY to the shadow archive
prefix ``weight_optimizer.S3_SHADOW_WEIGHTS_PREFIX``
(``config/scoring_weights_shadow_history``) — the previously-dead constant this
phase wires up. It returns ``{"applied": False, ...}`` unconditionally.
"""

import json
import logging
import math
import random
from datetime import date

import boto3
import pandas as pd
from nousergon_lib.eval_artifacts import (
    eval_artifact_key,
    eval_latest_key,
    new_eval_run_id,
)
from nousergon_lib.pillars import PILLARS
from nousergon_lib.quant.horizons import DEFAULT_POLICY

# Reuse (do NOT redefine) the previously-dead shadow-archive prefix constant.
# Phase 6 wires this up as the pillar-weight shadow-history S3 prefix.
from optimizer.weight_optimizer import S3_SHADOW_WEIGHTS_PREFIX

logger = logging.getLogger(__name__)

# The LIVE scoring-weights key — imported ONLY so callers/tests can assert this
# module never writes it. This module has no code path that puts to it.
from optimizer.weight_optimizer import S3_WEIGHTS_KEY  # noqa: E402  (invariant anchor)

# ── Weight space ──────────────────────────────────────────────────────────────
# The six canonical pillars (Σ pillar weights == 1.0), plus two within-composite
# mix terms that are sampled independently in [0, 1] (they are NOT part of the
# Σ==1 pillar simplex — they mix quant↔qual within a pillar and legacy↔pillar at
# the composite level respectively):
#   • qual_weight   — within-pillar quant↔qual blend (PillarContribution.
#                     within_pillar_qual_weight); blended = (1-w)*quant + w*qual.
#   • legacy_blend  — composite-level mix of the legacy quant/qual blend against
#                     the pillar-weighted base (0 = pure pillars, 1 = pure legacy).
PILLAR_WEIGHT_KEYS = list(PILLARS)  # quality, value, momentum, growth, stewardship, defensiveness
WITHIN_PILLAR_KEY = "qual_weight"
LEGACY_BLEND_KEY = "legacy_blend"

# Sum-to-one constraint tolerance for the pillar simplex.
_SUM_TOL = 1e-6

# ── Auto-scaled random search (mirrors analysis/param_sweep.py idioms) ─────────
# We reuse compute_n_trials / auto_n_trials from param_sweep for the trial count
# rather than reinventing the Bergstra-Bengio math. The pillar simplex is
# continuous, so there is no finite "grid size"; we scale off a nominal grid
# proxy (resolution ** n_free_dims) clamped by auto_n_trials' floor/ceiling.
_GRID_RESOLUTION = 6  # nominal per-dim granularity for the n_trials proxy
_DEFAULT_SEED = 0

# ── Ranking / constraint defaults (mirror executor_optimizer) ─────────────────
_MIN_VALID_CANDIDATES = 5
_MIN_SORTINO_IMPROVEMENT = 0.05
_MIN_NAMES_PER_DATE = 3          # need a cross-section to form a top-decile book
_MIN_RESOLVED_DATES = 10         # need a return stream to compute Sortino
# Sortino significance floor — below this |baseline| the improvement_pct framing
# is structural noise (mirrors executor_optimizer._MIN_BASELINE_MAGNITUDE_BY_RANK).
_MIN_BASELINE_SORTINO = 0.05
_IMPROVEMENT_DENOM_FLOOR = 1e-6

# ── Outcome horizon (single chokepoint — never hardcode column literals) ──────
_POLICY = DEFAULT_POLICY
_LONG_HORIZON = _POLICY.primary_horizon
# Continuous skilled-risk target: canonical primary-horizon log-alpha.
_SKILL_TARGET = _POLICY.skill_target_column(_LONG_HORIZON)
_RESOLVED_OUTCOME = _POLICY.resolved_gate_column()

# Module-level config ref — set by init_config() from evaluate.py / backtest.py
_cfg: dict = {}


def init_config(config: dict) -> None:
    """Load the ``pillar_weight_optimizer`` section from backtester config."""
    global _cfg
    _cfg = config.get("pillar_weight_optimizer", {})


def _default_weights() -> dict:
    """Equal-pillar starting point (Σ pillar weights == 1.0) + neutral mix terms."""
    n = len(PILLAR_WEIGHT_KEYS)
    w = {k: round(1.0 / n, 6) for k in PILLAR_WEIGHT_KEYS}
    w[WITHIN_PILLAR_KEY] = 0.5
    w[LEGACY_BLEND_KEY] = 0.0
    return w


# ── Sampling ──────────────────────────────────────────────────────────────────


def sample_weight_vector(rng: random.Random) -> dict:
    """Sample one candidate weight vector honouring the Σ==1 pillar constraint.

    The six pillar weights are drawn from a flat Dirichlet(1,...,1) (uniform over
    the simplex) so ``Σ pillar_weights == 1.0`` holds by construction; the two
    mix terms are drawn uniformly in [0, 1]. Every returned vector satisfies
    :func:`_sum_to_one_ok`.
    """
    # Dirichlet(1..1) via normalized Exponentials — no numpy dependency needed.
    raw = [-math.log(1.0 - rng.random()) for _ in PILLAR_WEIGHT_KEYS]
    total = sum(raw)
    vec = {k: raw[i] / total for i, k in enumerate(PILLAR_WEIGHT_KEYS)}
    # Renormalize to kill floating-point drift so Σ==1 within _SUM_TOL exactly.
    s = sum(vec.values())
    vec = {k: v / s for k, v in vec.items()}
    vec[WITHIN_PILLAR_KEY] = rng.random()
    vec[LEGACY_BLEND_KEY] = rng.random()
    return vec


def _sum_to_one_ok(vec: dict) -> bool:
    """True iff the six pillar weights sum to 1.0 within tolerance."""
    return abs(sum(vec[k] for k in PILLAR_WEIGHT_KEYS) - 1.0) <= _SUM_TOL


def normalize_pillar_weights(vec: dict) -> dict:
    """Project an arbitrary weight dict onto the Σ==1 pillar simplex (mix terms
    clamped to [0, 1]). Used to normalize operator-supplied current weights."""
    out = dict(vec)
    pill = {k: max(0.0, float(vec.get(k, 0.0))) for k in PILLAR_WEIGHT_KEYS}
    total = sum(pill.values())
    if total <= 0:
        pill = {k: 1.0 / len(PILLAR_WEIGHT_KEYS) for k in PILLAR_WEIGHT_KEYS}
    else:
        pill = {k: v / total for k, v in pill.items()}
    out.update(pill)
    out[WITHIN_PILLAR_KEY] = min(1.0, max(0.0, float(vec.get(WITHIN_PILLAR_KEY, 0.5))))
    out[LEGACY_BLEND_KEY] = min(1.0, max(0.0, float(vec.get(LEGACY_BLEND_KEY, 0.0))))
    return out


def _n_trials() -> int:
    """Auto-scaled trial count, reusing param_sweep's floor/ceiling clamps."""
    from analysis.param_sweep import auto_n_trials

    n_free = len(PILLAR_WEIGHT_KEYS) + 2  # pillars + qual_weight + legacy_blend
    grid_proxy = _GRID_RESOLUTION ** min(n_free, 5)  # cap the proxy exponent
    return auto_n_trials(
        grid_proxy,
        trial_pct=_cfg.get("trial_pct"),
        min_trials=_cfg.get("min_trials"),
        max_trials=_cfg.get("max_trials"),
    )


# ── Composite reconstruction (reuses the nousergon-lib pillar arithmetic) ─────


def _score_name(row: pd.Series, vec: dict) -> float | None:
    """Reconstruct one name's composite score under weight vector ``vec``.

    Mirrors ``CompositeBreakdown`` arithmetic: each pillar's blended value is the
    within-pillar quant↔qual mix, pillar-weighted and summed to the weighted
    base; the composite is then a legacy_blend mix of that base against the
    legacy quant/qual blend. Names missing every pillar component are unscored
    (None) and excluded from that date's cross-section.
    """
    qw = vec[WITHIN_PILLAR_KEY]
    weighted_base = 0.0
    any_pillar = False
    for pillar in PILLAR_WEIGHT_KEYS:
        quant = row.get(f"{pillar}_quant")
        qual = row.get(f"{pillar}_qual")
        if quant is None and qual is None:
            continue
        if pd.isna(quant) if quant is not None else True:
            quant = None
        if pd.isna(qual) if qual is not None else True:
            qual = None
        if quant is None and qual is None:
            continue
        if quant is None:
            blended = qual
        elif qual is None:
            blended = quant
        else:
            blended = (1.0 - qw) * float(quant) + qw * float(qual)
        weighted_base += vec[pillar] * float(blended)
        any_pillar = True
    if not any_pillar:
        return None

    legacy = row.get("legacy_blend_score")
    lb = vec[LEGACY_BLEND_KEY]
    if legacy is not None and not pd.isna(legacy):
        return (1.0 - lb) * weighted_base + lb * float(legacy)
    return weighted_base


def _candidate_return_stream(df: pd.DataFrame, vec: dict) -> pd.Series | None:
    """Realized per-date alpha stream for the top-decile book under ``vec``.

    For each resolved score_date, score every name, take the top decile (min
    :data:`_MIN_NAMES_PER_DATE`), and average their realized ``_SKILL_TARGET``
    (canonical log-alpha). The resulting per-date series is the candidate's
    realized return stream, fed to Sortino. ``None`` when too few dates resolve.
    """
    top_frac = float(_cfg.get("top_fraction", 0.10))
    per_date: dict = {}
    for score_date, group in df.groupby("score_date"):
        scored = []
        for _, row in group.iterrows():
            s = _score_name(row, vec)
            alpha = row.get(_SKILL_TARGET)
            if s is None or alpha is None or pd.isna(alpha):
                continue
            scored.append((s, float(alpha)))
        if len(scored) < _MIN_NAMES_PER_DATE:
            continue
        scored.sort(key=lambda t: t[0], reverse=True)
        k = max(_MIN_NAMES_PER_DATE, int(math.ceil(len(scored) * top_frac)))
        book = scored[:k]
        per_date[score_date] = sum(a for _, a in book) / len(book)

    if len(per_date) < _MIN_RESOLVED_DATES:
        return None
    return pd.Series(per_date).sort_index()


def _sortino(returns: pd.Series, target: float = 0.0) -> float | None:
    """Sortino ratio — mean excess / downside deviation (not annualized).

    Same definition as ``analysis.factor_blend_sensitivity._sortino`` /
    ``vectorbt_bridge._compute_sortino_ratio``: RMS of below-target excursions.
    ``None`` on < 2 obs or zero downside deviation.
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return None
    excess = returns - target
    downside = excess[excess < 0]
    if len(downside) == 0:
        return None
    downside_dev = math.sqrt((downside ** 2).mean())
    if downside_dev == 0:
        return None
    return float(excess.mean() / downside_dev)


def _prepare(df: pd.DataFrame) -> pd.DataFrame | dict:
    """Filter to resolved rows; return early-exit dict on starvation."""
    if df is None or df.empty:
        return {"status": "insufficient_data", "note": "No pillar-breakdown rows available"}
    if _RESOLVED_OUTCOME not in df.columns:
        return {
            "status": "insufficient_data",
            "note": (
                f"resolved-outcome column {_RESOLVED_OUTCOME!r} absent — "
                f"schema drift / horizon retirement?"
            ),
        }
    resolved = df[df[_RESOLVED_OUTCOME].notna()].copy()
    n_dates = resolved["score_date"].nunique() if "score_date" in resolved.columns else 0
    if n_dates < _MIN_RESOLVED_DATES:
        return {
            "status": "insufficient_data",
            "n_resolved_dates": int(n_dates),
            "min_required": _MIN_RESOLVED_DATES,
            "note": (
                f"Only {n_dates} resolved score_dates (need {_MIN_RESOLVED_DATES}) — "
                f"pillar-weight recommendation deferred."
            ),
        }
    return resolved


# ── recommend() — the sweep-then-rank entry point ─────────────────────────────


def recommend(
    df: pd.DataFrame,
    current_weights: dict | None = None,
) -> dict:
    """Sweep candidate pillar-weight vectors and recommend the best by Sortino.

    Args:
        df: per-name DataFrame carrying, per pillar, ``{pillar}_quant`` /
            ``{pillar}_qual`` components (from the persisted
            ``investment_thesis.composite_breakdown.pillar_contributions``),
            an optional ``legacy_blend_score``, the realized canonical log-alpha
            (:data:`_SKILL_TARGET`), a ``score_date`` and the resolved-outcome
            gate column.
        current_weights: the weights the system is currently scoring with (the
            sweep baseline). Defaults to equal-pillar weights.

    Returns a result dict with ``status`` ∈ {``ok``, ``insufficient_data``,
    ``alpha_below_floor``, ``negative_sortino``, ``baseline_insignificant``,
    ``no_improvement``} mirroring executor_optimizer's shape, plus
    ``recommended_weights`` / ``baseline_weights`` / ``best_sortino``.

    NOTE: recommend() is pure — it performs NO S3 writes. Shadow archival is
    :func:`apply`.
    """
    prepared = _prepare(df)
    if isinstance(prepared, dict):
        return prepared  # early exit
    resolved = prepared

    if current_weights is None:
        current_weights = _default_weights()
    current_weights = normalize_pillar_weights(current_weights)

    seed = _cfg.get("seed", _DEFAULT_SEED)
    rng = random.Random(seed)
    n_trials = _n_trials()

    # Baseline candidate = current weights; sweep candidates = random simplex draws.
    candidates: list[dict] = [dict(current_weights)]
    seen: set = {tuple(round(current_weights[k], 6) for k in PILLAR_WEIGHT_KEYS)}
    attempts = 0
    while len(candidates) < n_trials + 1 and attempts < n_trials * 20:
        attempts += 1
        vec = sample_weight_vector(rng)
        key = tuple(round(vec[k], 6) for k in PILLAR_WEIGHT_KEYS)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(vec)

    # Score every candidate: realized-return stream → Sortino + mean alpha.
    rows = []
    for i, vec in enumerate(candidates):
        assert _sum_to_one_ok(vec), "sampled vector violates Σ==1 constraint"
        stream = _candidate_return_stream(resolved, vec)
        if stream is None:
            continue
        sortino = _sortino(stream)
        rows.append(
            {
                "_candidate_idx": i,
                "_is_baseline": i == 0,
                **{k: round(vec[k], 6) for k in PILLAR_WEIGHT_KEYS},
                WITHIN_PILLAR_KEY: round(vec[WITHIN_PILLAR_KEY], 6),
                LEGACY_BLEND_KEY: round(vec[LEGACY_BLEND_KEY], 6),
                "sortino_ratio": sortino,
                "total_alpha": float(stream.mean()),
                "n_dates": int(len(stream)),
            }
        )

    if not rows:
        return {
            "status": "insufficient_data",
            "note": "No candidate produced a scorable return stream (too few names per date).",
        }

    valid = pd.DataFrame(rows)
    min_valid = _cfg.get("min_valid_candidates", _MIN_VALID_CANDIDATES)
    if len(valid) < min_valid:
        return {
            "status": "insufficient_data",
            "n_valid": int(len(valid)),
            "min_required": int(min_valid),
            "note": f"Only {len(valid)} candidates scored (need {min_valid}).",
        }

    n_swept = int(len(valid))

    # ── Hard alpha-floor constraint (mirrors executor_optimizer) ──────────────
    # Filter candidates whose realized top-decile alpha < floor BEFORE ranking.
    # Sortino alone will happily promote an alpha-negative "defensive" weighting;
    # alpha-positive is a hard CONSTRAINT per the canonical-alpha framework, not
    # a side-output. alpha_floor=None (default) leaves the gate inactive.
    alpha_floor = _cfg.get("alpha_floor")
    if alpha_floor is not None:
        n_before = len(valid)
        best_alpha_in_sweep = float(valid["total_alpha"].max())
        valid = valid[valid["total_alpha"] >= alpha_floor].copy()
        if valid.empty:
            return {
                "status": "alpha_below_floor",
                "alpha_floor": float(alpha_floor),
                "n_combos_below_floor": int(n_before),
                "best_alpha_in_sweep": round(best_alpha_in_sweep, 6),
                "note": (
                    f"All {n_before} candidates realized top-decile "
                    f"total_alpha < {alpha_floor} (best: {best_alpha_in_sweep:.6f}). "
                    f"Refusing to recommend — alpha-positive is a hard constraint."
                ),
            }
        logger.info(
            "pillar_weight_optimizer: alpha_floor=%s dropped %d/%d candidates; "
            "%d alpha-positive remain.",
            alpha_floor, n_before - len(valid), n_before, len(valid),
        )

    # ── Rank by Sortino-on-skilled-risk (the executor_optimizer anchor) ───────
    valid = valid[valid["sortino_ratio"].notna()].copy()
    if valid.empty:
        return {
            "status": "insufficient_data",
            "note": "No candidate produced a defined Sortino (all streams had zero downside).",
        }
    valid = valid.sort_values("sortino_ratio", ascending=False)

    best_row = valid.iloc[0]
    # Baseline = the current-weights candidate if it survived; else worst-ranked.
    baseline_mask = valid["_is_baseline"]
    if baseline_mask.any():
        baseline_row = valid[baseline_mask].iloc[0]
    else:
        baseline_row = valid.iloc[-1]

    def _vec_of(row) -> dict:
        out = {k: float(row[k]) for k in PILLAR_WEIGHT_KEYS}
        out[WITHIN_PILLAR_KEY] = float(row[WITHIN_PILLAR_KEY])
        out[LEGACY_BLEND_KEY] = float(row[LEGACY_BLEND_KEY])
        return out

    recommended = _vec_of(best_row)
    baseline = _vec_of(baseline_row)
    best_sortino = round(float(best_row["sortino_ratio"]), 4)
    baseline_sortino = round(float(baseline_row["sortino_ratio"]), 4)
    best_alpha = round(float(best_row["total_alpha"]), 6)
    baseline_alpha = round(float(baseline_row["total_alpha"]), 6)

    improvement_delta = best_sortino - baseline_sortino
    improvement_pct = improvement_delta / max(
        abs(baseline_sortino), _IMPROVEMENT_DENOM_FLOOR
    )
    min_baseline = float(_cfg.get("min_baseline_sortino", _MIN_BASELINE_SORTINO))
    improvement_significant = abs(baseline_sortino) >= min_baseline

    common = {
        "fit_target": "pillar_sortino_skill_composite",
        "rank_metric": "sortino_ratio",
        "n_combos_swept": n_swept,
        "n_candidates_scored": int(len(valid)),
        "baseline_weights": {k: round(v, 6) for k, v in baseline.items()},
        "recommended_weights": {k: round(v, 6) for k, v in recommended.items()},
        "best_sortino": best_sortino,
        "baseline_sortino": baseline_sortino,
        "best_alpha": best_alpha,
        "baseline_alpha": baseline_alpha,
        "improvement_pct": round(improvement_pct, 4),
        "improvement_delta": round(improvement_delta, 6),
        "improvement_significant": bool(improvement_significant),
        "min_baseline_magnitude": min_baseline,
        "constraint": "sum_pillar_weights_eq_1",
    }

    # Guard: never recommend off a negative-Sortino optimization (skilled-risk
    # anchor sanity check — every candidate's downside-aware return is loss-making).
    if best_sortino < 0:
        return {
            "status": "negative_sortino",
            **common,
            "note": (
                f"Best Sortino ({best_sortino:.4f}) is negative — every candidate's "
                f"downside-aware realized alpha is loss-making. Refusing to recommend."
            ),
        }

    # Baseline-significance gate — refuse to promote off a noise baseline ratio.
    if not improvement_significant:
        return {
            "status": "baseline_insignificant",
            **common,
            "note": (
                f"Baseline Sortino ({baseline_sortino:.4f}) magnitude is below the "
                f"{min_baseline:.3f} significance floor — improvement ratio is "
                f"structural noise (improvement_delta={improvement_delta:.4f})."
            ),
        }

    min_improvement = _cfg.get("min_sortino_improvement", _MIN_SORTINO_IMPROVEMENT)
    if improvement_pct < min_improvement:
        return {
            "status": "no_improvement",
            **common,
            "note": (
                f"Best Sortino ({best_sortino:.4f}) only {improvement_pct:.1%} better "
                f"than baseline ({baseline_sortino:.4f}). Need {min_improvement:.0%}+."
            ),
        }

    return {
        "status": "ok",
        **common,
        "note": (
            f"Best candidate improves Sortino by {improvement_pct:.1%} "
            f"({baseline_sortino:.4f} → {best_sortino:.4f}) across {n_swept} candidates "
            f"(best realized top-decile alpha={best_alpha} shown for display, not gating)."
        ),
    }


# ── apply() — SHADOW ARCHIVE ONLY (no live write, no cutover path) ────────────


def apply(result: dict, bucket: str) -> dict:
    """Archive the recommended pillar weights to the SHADOW history prefix.

    HARD INVARIANT: this writes ONLY under
    :data:`S3_SHADOW_WEIGHTS_PREFIX` (``config/scoring_weights_shadow_history``)
    — the timestamped eval-artifact layout (``{prefix}/{run_id}.json`` +
    ``latest.json`` sidecar). It NEVER writes the live scoring-weights config key
    (:data:`S3_WEIGHTS_KEY` = ``config/scoring_weights.json``) and has no
    live-cutover branch. Always returns ``{"applied": False, ...}`` — shadow
    archival is not a live apply.
    """
    if result.get("status") != "ok":
        return {"applied": False, "reason": f"status={result.get('status')}"}

    recommended = result.get("recommended_weights", {})
    if not recommended:
        return {"applied": False, "reason": "no recommended weights"}

    payload = {
        **recommended,
        "updated_at": str(date.today()),
        "mode": "shadow",
        "fit_target": result.get("fit_target", "pillar_sortino_skill_composite"),
        "constraint": result.get("constraint", "sum_pillar_weights_eq_1"),
        "best_sortino": result.get("best_sortino"),
        "baseline_sortino": result.get("baseline_sortino"),
        "best_alpha": result.get("best_alpha"),
        "improvement_pct": result.get("improvement_pct"),
        "n_combos_swept": result.get("n_combos_swept"),
    }

    # Canonical eval-style archive layout (lib eval_artifacts): flat
    # {prefix}/{run_id}.json + latest.json sidecar with a YYMMDDHHMM run_id, so
    # same-day re-runs preserve forensic capture instead of overwriting.
    s3 = boto3.client("s3")
    body = json.dumps(payload, indent=2)
    run_id = new_eval_run_id()
    shadow_key = eval_artifact_key(S3_SHADOW_WEIGHTS_PREFIX, run_id)
    shadow_latest_key = eval_latest_key(S3_SHADOW_WEIGHTS_PREFIX)
    try:
        s3.put_object(
            Bucket=bucket, Key=shadow_key, Body=body, ContentType="application/json",
        )
        s3.put_object(
            Bucket=bucket, Key=shadow_latest_key, Body=body,
            ContentType="application/json",
        )
        logger.info(
            "Pillar weights archived to SHADOW history (observe-only, live config "
            "untouched): s3://%s/%s (+ latest.json sidecar)",
            bucket, shadow_key,
        )
    except Exception as e:
        logger.warning("Shadow pillar-weights write failed (non-fatal): %s", e)
        return {"applied": False, "reason": f"shadow S3 write failed: {e}"}

    return {
        "applied": False,  # shadow-only — never a live apply
        "reason": "shadow mode — pillar_weight_optimizer is observe-only (config#789 Phase 6)",
        "shadow_key": shadow_key,
        "shadow_weights": recommended,
        "fit_target": payload["fit_target"],
        "best_sortino": result.get("best_sortino"),
        "best_alpha": result.get("best_alpha"),
        "improvement_pct": result.get("improvement_pct"),
    }
