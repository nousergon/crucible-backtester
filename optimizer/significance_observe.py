"""
significance_observe.py — observe-mode (shadow) statistical-significance
verdicts for the backtester's auto-apply optimizers (config#1426, Phase 2/3).

WHY THIS EXISTS
---------------
Of the optimizers that auto-write live trading config to S3, only the
``executor_optimizer`` gates promotion on a statistical-significance battery
(PSR / DSR / PBO, per the LdP selection-bias framework). The rest — weight,
predictor-veto, and the three sizing optimizers — promote on
statistically-undefended evidence (a point IC ≥ 0.05, a 5pp lift, an
OOS-degradation threshold). The L4593 **leg-f** finding proved the bug class:
the weight gate promotes a live config change ~10% of the time on PURE NULL
sub-scores. An IC of 0.05 on 30 samples is not distinguishable from noise.

WHAT THIS DOES
--------------
This module is the **observe-first** instrumentation for that arc. It computes,
for a given optimizer's promotion evidence, whether that evidence would clear a
significance bar (a seeded bootstrap IC confidence interval that excludes zero,
corroborated by the Spearman p-value), and records a ``would_block`` /
``promotes_on_undefended_evidence`` verdict alongside what the live gate
actually decided.

It is **NON-ENFORCING**. It never changes a promote/reject decision. The verdict
is logged + attached to the optimizer result for operator review. Phase 4 (the
human gate) ratifies *cost-calibrated* thresholds per gate and flips
observe→enforce — that flip does NOT live here.

REUSE
-----
The significance primitives were already lifted to the shared lib (config#1426
Phase 1 — PSR/DSR 2026-06-03, cscv_pbo config#1318 / v0.72.0). This module only
composes them; it implements no new statistics:

  * ``compute_ic``    — Spearman IC + two-sided p-value (via the
                        ``analysis.information_coefficient`` shim).
  * ``bootstrap_ci``  — seeded percentile CI (via the ``analysis.intervals``
                        shim). We bootstrap the IC by resampling *paired
                        indices*, so the (conviction, return) pairing is
                        preserved across resamples.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from analysis.information_coefficient import compute_ic
from analysis.intervals import bootstrap_ci

logger = logging.getLogger(__name__)

# Defaults (override per-optimizer via config). Deliberately conservative —
# this is a *significance* bar (is the evidence distinguishable from noise?),
# not a *cost-calibrated* bar (Phase 4 ratifies those).
_DEFAULT_ALPHA = 0.05
_DEFAULT_CI_LEVEL = 0.95
_DEFAULT_N_RESAMPLES = 1000
_DEFAULT_SEED = 0  # determinism: same cycle → same verdict across re-renders.
_DEFAULT_MIN_SAMPLES = 20


def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman IC = Pearson on ranks. NaN-safe degenerate → 0.0.

    Used *inside* the bootstrap resample loop (1000×), so it avoids a per-resample
    scipy call. Identical value to ``scipy.stats.spearmanr`` for the point IC;
    the point estimate + p-value still come from the lib's ``compute_ic``.
    """
    if x.size < 2:
        return 0.0
    xr = np.argsort(np.argsort(x)).astype(np.float64)
    yr = np.argsort(np.argsort(y)).astype(np.float64)
    if xr.std() == 0.0 or yr.std() == 0.0:
        return 0.0
    return float(np.corrcoef(xr, yr)[0, 1])


def ic_significance_verdict(
    conviction: Sequence[float] | np.ndarray,
    forward_return: Sequence[float] | np.ndarray,
    *,
    alpha: float = _DEFAULT_ALPHA,
    ci_level: float = _DEFAULT_CI_LEVEL,
    n_resamples: int = _DEFAULT_N_RESAMPLES,
    seed: int = _DEFAULT_SEED,
    min_samples: int = _DEFAULT_MIN_SAMPLES,
) -> dict:
    """Is the IC between ``conviction`` and ``forward_return`` significant?

    Significance is decided primarily by a **seeded percentile bootstrap CI of
    the IC that excludes zero** (robust, scipy-independent), corroborated by the
    Spearman two-sided p-value when scipy is available.

    Returns a dict::

        {
          "status": "ok" | "insufficient_data" | "no_variance",
          "n": int,
          "ic": float,                # point IC (lib compute_ic)
          "p_value": float | None,    # Spearman two-sided p (None if scipy absent)
          "ci_low": float, "ci_high": float, "ci_level": float,
          "alpha": float,
          "significant": bool,        # CI excludes 0 (and p < alpha when finite)
          "method": "bootstrap_ic_ci + spearman_p",
        }

    ``significant`` is False (⇒ caller treats the evidence as undefended) whenever
    the data is insufficient, has no variance, or the CI brackets zero.
    """
    c = np.asarray(conviction, dtype=np.float64)
    r = np.asarray(forward_return, dtype=np.float64)
    if c.size != r.size:
        raise ValueError(
            f"conviction (n={c.size}) and forward_return (n={r.size}) must be same length"
        )
    valid = np.isfinite(c) & np.isfinite(r)
    c = c[valid]
    r = r[valid]
    n = int(c.size)

    base = {"alpha": float(alpha), "method": "bootstrap_ic_ci + spearman_p", "n": n}

    # Point IC + p-value from the lib (single scipy call).
    ic_res = compute_ic(c, r, min_samples=min_samples)
    status = ic_res.get("status")
    if status != "ok":
        # insufficient_data / no_variance → evidence cannot be defended.
        return {**base, "status": status, "ic": ic_res.get("ic"),
                "p_value": ic_res.get("p_value"), "significant": False}

    ic_point = float(ic_res["ic"])
    p_value = ic_res.get("p_value")
    p_value = float(p_value) if p_value is not None and np.isfinite(p_value) else None

    # Paired bootstrap: resample index positions, recompute IC on the paired
    # resample. Passing indices (all finite) through bootstrap_ci means its
    # internal NaN-drop never touches the pairing.
    idx_space = np.arange(n, dtype=np.float64)

    def _ic_stat(idx_float: np.ndarray) -> float:
        idx = idx_float.astype(np.int64)
        return _spearman_ic(c[idx], r[idx])

    ci = bootstrap_ci(
        idx_space, statistic=_ic_stat,
        ci_level=ci_level, n_resamples=n_resamples, seed=seed,
    )
    if ci.get("status") != "ok":
        return {**base, "status": "insufficient_data", "ic": ic_point,
                "p_value": p_value, "significant": False}

    ci_low = float(ci["ci_low"])
    ci_high = float(ci["ci_high"])
    ci_excludes_zero = (ci_low > 0.0) or (ci_high < 0.0)
    # Corroborate with the asymptotic p-value when we have it; CI is the
    # authoritative signal (it survives scipy being absent).
    p_ok = True if p_value is None else (p_value < alpha)
    significant = bool(ci_excludes_zero and p_ok)

    return {
        **base,
        "status": "ok",
        "ic": ic_point,
        "p_value": p_value,
        "ci_low": round(ci_low, 6),
        "ci_high": round(ci_high, 6),
        "ci_level": float(ci_level),
        "significant": significant,
    }


def build_observe_record(
    *,
    gate: str,
    significant: bool,
    did_promote: bool | None,
    detail: dict | None = None,
) -> dict:
    """Wrap a significance result into the standard observe record.

    ``did_promote`` is what the LIVE (currently-shipping) gate decided — pass
    ``None`` when that decision isn't known at the call site. The headline field
    is ``promotes_on_undefended_evidence``: the live gate promoted while the
    significance bar would have blocked (the leg-f failure mode).
    """
    would_block = not significant
    promotes_on_undefended = bool(did_promote and would_block) if did_promote is not None else None
    rec = {
        "gate": gate,
        "enabled": True,
        "significant": bool(significant),
        "would_block": would_block,
        "did_promote": did_promote,
        "promotes_on_undefended_evidence": promotes_on_undefended,
        "enforced": False,  # observe-first: this verdict NEVER changes a decision.
    }
    if detail:
        rec["detail"] = detail
    return rec


def observe_weight_optimizer(
    test_set,
    sub_cols: dict[str, str],
    *,
    return_cols: tuple[str, ...] = ("return_10d", "return_30d"),
    cfg: dict | None = None,
) -> dict:
    """Phase-2 observe verdict for ``weight_optimizer`` (config#1426).

    The weight gate shifts live scoring weights based on each sub-score's
    correlation with forward returns. This computes, on the **OOS (test) set**,
    a bootstrap IC CI per (sub_score × horizon) and asks: does ANY sub-score
    driving the weight change have a significant OOS IC? If none do, the weight
    shift rests on noise — ``would_block = True``.

    Returns the standard observe record with a per-sub-score ``detail`` block.
    ``did_promote`` is left None here (the apply-gate decision is made later in
    ``apply_weights``); the caller fills it in when it logs the comparison.
    """
    cfg = cfg or {}
    alpha = float(cfg.get("significance_alpha", _DEFAULT_ALPHA))
    n_resamples = int(cfg.get("significance_n_resamples", _DEFAULT_N_RESAMPLES))
    seed = int(cfg.get("significance_seed", _DEFAULT_SEED))

    per_subscore: dict[str, dict] = {}
    any_significant = False
    for label, col in sub_cols.items():
        horizons: dict[str, dict] = {}
        sub_significant = False
        for ret_col in return_cols:
            if col not in test_set.columns or ret_col not in test_set.columns:
                horizons[ret_col] = {"status": "missing_column", "significant": False}
                continue
            pair = test_set[[col, ret_col]].dropna()
            verdict = ic_significance_verdict(
                pair[col].to_numpy(), pair[ret_col].to_numpy(),
                alpha=alpha, n_resamples=n_resamples, seed=seed,
            )
            horizons[ret_col] = verdict
            sub_significant = sub_significant or verdict.get("significant", False)
        per_subscore[label] = {"significant": sub_significant, "horizons": horizons}
        any_significant = any_significant or sub_significant

    return build_observe_record(
        gate="weight_optimizer",
        significant=any_significant,
        did_promote=None,
        detail={"per_subscore": per_subscore, "n_test": int(len(test_set))},
    )
