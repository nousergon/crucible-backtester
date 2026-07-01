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
from analysis.intervals import bootstrap_ci, wilson_score_interval

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
    enforced: bool = False,
) -> dict:
    """Wrap a significance result into the standard observe record.

    ``did_promote`` is what the LIVE (currently-shipping) gate decided — pass
    ``None`` when that decision isn't known at the call site. The headline field
    is ``promotes_on_undefended_evidence``: the live gate promoted while the
    significance bar would have blocked (the leg-f failure mode).

    ``enforced`` records the MODE the verdict was emitted under (config#1426
    Phase 4). It defaults to ``False`` (observe-only). When an optimizer's
    ``enforce_significance`` flag is on, the emitted record carries
    ``enforced=True`` so the artifact reflects that the verdict was load-bearing
    for that cycle. This field is descriptive metadata only — the actual
    block/allow decision is made in each optimizer's ``apply()``.
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
        "enforced": bool(enforced),
    }
    if detail:
        rec["detail"] = detail
    return rec


def _cfg_enforced(cfg: dict | None) -> bool:
    """Resolve the ``enforce_significance`` mode flag from an optimizer's config
    section (config#1426 Phase 4). Defaults False — observe-only."""
    return bool((cfg or {}).get("enforce_significance", False))


def significance_would_block(verdict: dict | None) -> bool:
    """Conservative enforce-mode block decision (config#1426 Phase 4).

    Returns True (⇒ block promotion) when the observe verdict is missing/None or
    reports ``would_block`` (evidence not statistically significant). A missing
    verdict blocks under enforce: we refuse to promote live config on evidence we
    could not even measure to defend — the SAFE direction. Enforce can only
    BLOCK a promotion the live gate already allowed; it never enables one.
    """
    if not verdict:
        return True
    return bool(verdict.get("would_block", True))


_WEIGHT_CANONICAL_HORIZON = "log_alpha_21d"


def weight_canonical_signed_floor_fails(
    verdict: dict | None,
    min_signed_ic: float,
    *,
    canonical_horizon: str = _WEIGHT_CANONICAL_HORIZON,
) -> bool:
    """Signed, canonical-horizon effect-size floor for the weight gate under
    enforce (config#1426 Phase 4; refined 2026-07-01 after a re-replay).

    Returns True (⇒ BLOCK) UNLESS at least one sub-score's per-horizon verdict on
    the CANONICAL ``log_alpha_21d`` horizon is BOTH significant (bootstrap CI
    excludes zero) AND has a POSITIVE signed IC ≥ ``min_signed_ic``.

    Why signed + canonical-only (not absolute + any-horizon): the re-replay
    showed the weight gate's ``significant=True`` was driven ENTIRELY by a large
    NEGATIVE IC (quant × return_5d = −0.254) on the LEGACY 5d horizon while the
    canonical 21d horizon was null. A significant negative IC means "down-weight"
    — not a promotable reweight — so an absolute |IC| floor on any horizon would
    wrongly "defend" the promotion. The floor is therefore signed (≥ +0.03) and
    read ONLY off the canonical horizon. A missing/None verdict has no qualifying
    horizon ⇒ blocks (conservative — enforce can only block, never enable).
    """
    detail = (verdict or {}).get("detail") or {}
    per_subscore = detail.get("per_subscore") or {}
    for sub in per_subscore.values():
        h = (sub.get("horizons") or {}).get(canonical_horizon) or {}
        if h.get("significant") and float(h.get("ic") or 0.0) >= float(min_signed_ic):
            return False
    return True


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

    # The per-subscore × per-horizon verdicts (each carrying signed ``ic`` +
    # ``significant``) are the substrate the Phase-4 weight enforce floor reads
    # directly — see ``weight_canonical_signed_floor_fails``. No summary field is
    # surfaced: the floor is signed + canonical-horizon-specific, so a scalar
    # (e.g. max |IC|) would lose exactly the sign/horizon it must discriminate.
    return build_observe_record(
        gate="weight_optimizer",
        significant=any_significant,
        did_promote=None,
        detail={"per_subscore": per_subscore, "n_test": int(len(test_set))},
        enforced=_cfg_enforced(cfg),
    )


# ── Phase 3: additional significance primitives ──────────────────────────────


def proportion_lift_significance_verdict(
    successes: int,
    n: int,
    base_rate: float,
    *,
    ci_level: float = _DEFAULT_CI_LEVEL,
) -> dict:
    """Is a hit-rate (e.g. veto precision) significantly ABOVE a base rate?

    Reuses the lib's Wilson score interval (the small-N-robust binomial CI).
    ``significant`` iff the Wilson lower bound exceeds ``base_rate`` — i.e. the
    lift over the baseline is statistically distinguishable from zero. This is
    the same shape veto_analysis already enforces in skill-composite mode; here
    it is computed uniformly for the observe verdict.
    """
    if n <= 0 or successes < 0 or successes > n:
        return {"status": "insufficient_data", "n": int(max(n, 0)), "significant": False}
    ci = wilson_score_interval(successes, n, ci_level=ci_level)
    if ci.get("status") != "ok":
        return {"status": ci.get("status", "insufficient_data"), "n": int(n), "significant": False}
    ci_low = float(ci["ci_low"])
    rate = float(ci["rate"])
    significant = bool(ci_low > base_rate)
    return {
        "status": "ok",
        "n": int(n),
        "rate": round(rate, 6),
        "base_rate": round(float(base_rate), 6),
        "ci_low": round(ci_low, 6),
        "ci_high": round(float(ci["ci_high"]), 6),
        "ci_level": float(ci_level),
        "lift": round(rate - float(base_rate), 6),
        "significant": significant,
        "method": "wilson_lower_bound_vs_base_rate",
    }


def mean_diff_significance_verdict(
    sample_a: Sequence[float] | np.ndarray,
    sample_b: Sequence[float] | np.ndarray,
    *,
    ci_level: float = _DEFAULT_CI_LEVEL,
    n_resamples: int = _DEFAULT_N_RESAMPLES,
    seed: int = _DEFAULT_SEED,
    min_samples: int = _DEFAULT_MIN_SAMPLES,
) -> dict:
    """Is mean(a) − mean(b) significantly non-zero (seeded two-sample bootstrap)?

    Same seeded percentile-bootstrap method as the lib's ``bootstrap_ci``, in the
    two-independent-sample shape it doesn't cover (each group resampled
    independently with replacement). ``significant`` iff the CI of the mean
    difference excludes zero. Used for the stance-sizing gate, whose promotion
    rests on a per-stance alpha *spread* rather than a correlation.
    """
    a = np.asarray(sample_a, dtype=np.float64)
    b = np.asarray(sample_b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    na, nb = int(a.size), int(b.size)
    base = {"n_a": na, "n_b": nb, "method": "two_sample_mean_diff_bootstrap"}
    if na < min_samples or nb < min_samples:
        return {**base, "status": "insufficient_data", "significant": False}

    estimate = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, na, size=(n_resamples, na))
    ib = rng.integers(0, nb, size=(n_resamples, nb))
    boot = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return {**base, "status": "insufficient_data", "significant": False}

    tail = (1.0 - ci_level) / 2.0
    ci_low = float(np.percentile(boot, 100.0 * tail))
    ci_high = float(np.percentile(boot, 100.0 * (1.0 - tail)))
    significant = bool(ci_low > 0.0 or ci_high < 0.0)
    return {
        **base,
        "status": "ok",
        "estimate": round(estimate, 6),
        "ci_low": round(ci_low, 6),
        "ci_high": round(ci_high, 6),
        "ci_level": float(ci_level),
        "significant": significant,
    }


def _opt_cfg(cfg: dict | None) -> tuple[float, int, int]:
    """Resolve (alpha, n_resamples, seed) from an optimizer's config section."""
    cfg = cfg or {}
    return (
        float(cfg.get("significance_alpha", _DEFAULT_ALPHA)),
        int(cfg.get("significance_n_resamples", _DEFAULT_N_RESAMPLES)),
        int(cfg.get("significance_seed", _DEFAULT_SEED)),
    )


def observe_ic_gate(
    conviction,
    forward_return,
    *,
    gate: str,
    cfg: dict | None = None,
) -> dict:
    """Observe verdict for a rank-IC promotion gate (predictor/barrier sizing).

    The optimizer promotes when an IC (signal vs realized alpha) clears 0.05 with
    no significance test. This asks whether that IC's bootstrap CI excludes zero.
    """
    alpha, n_resamples, seed = _opt_cfg(cfg)
    verdict = ic_significance_verdict(
        conviction, forward_return, alpha=alpha, n_resamples=n_resamples, seed=seed,
    )
    return build_observe_record(
        gate=gate, significant=verdict.get("significant", False),
        did_promote=None, detail=verdict, enforced=_cfg_enforced(cfg),
    )


def observe_veto(
    thresholds: list[dict],
    recommended_threshold,
    base_rate: float,
    *,
    cfg: dict | None = None,
) -> dict:
    """Observe verdict for the predictor-veto gate.

    The gate promotes a veto threshold on a 5pp point lift over base rate. This
    asks whether the recommended threshold's precision lift is statistically
    significant (Wilson lower bound > base rate).
    """
    row = next(
        (t for t in (thresholds or []) if t.get("confidence") == recommended_threshold),
        None,
    )
    enforced = _cfg_enforced(cfg)
    if not row or row.get("true_negatives") is None or not row.get("n_vetoes"):
        return build_observe_record(
            gate="veto_analysis", significant=False, did_promote=None,
            detail={"status": "insufficient_data", "recommended_threshold": recommended_threshold},
            enforced=enforced,
        )
    verdict = proportion_lift_significance_verdict(
        int(row["true_negatives"]), int(row["n_vetoes"]), float(base_rate),
    )
    verdict["recommended_threshold"] = recommended_threshold
    return build_observe_record(
        gate="veto_analysis", significant=verdict.get("significant", False),
        did_promote=None, detail=verdict, enforced=enforced,
    )


def observe_stance_spread(
    stance_alpha_samples: dict[str, Sequence[float]],
    *,
    cfg: dict | None = None,
) -> dict:
    """Observe verdict for the stance-sizing gate.

    The gate promotes on a per-stance alpha *spread* ≥ 0.005 with no significance
    test. This asks whether the best vs worst qualifying stance's mean alpha is
    statistically distinguishable (two-sample mean-difference bootstrap excludes
    zero). ``stance_alpha_samples`` maps stance → its per-name alpha samples.
    """
    _, n_resamples, seed = _opt_cfg(cfg)
    enforced = _cfg_enforced(cfg)
    means = {
        s: float(np.nanmean(np.asarray(v, dtype=np.float64)))
        for s, v in (stance_alpha_samples or {}).items()
        if v is not None and len(v) > 0
    }
    if len(means) < 2:
        return build_observe_record(
            gate="stance_sizing", significant=False, did_promote=None,
            detail={"status": "insufficient_stances", "n_stances": len(means)},
            enforced=enforced,
        )
    best = max(means, key=means.get)
    worst = min(means, key=means.get)
    verdict = mean_diff_significance_verdict(
        stance_alpha_samples[best], stance_alpha_samples[worst],
        n_resamples=n_resamples, seed=seed,
    )
    verdict["best_stance"] = best
    verdict["worst_stance"] = worst
    return build_observe_record(
        gate="stance_sizing", significant=verdict.get("significant", False),
        did_promote=None, detail=verdict, enforced=enforced,
    )
