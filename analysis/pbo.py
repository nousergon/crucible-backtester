"""pbo — CSCV Probability of Backtest Overfitting (Bailey, Borwein, López de
Prado & Zhu 2014).

The selection-overfitting lens for any best-of-N param sweep that auto-promotes
the top combo: ``compute_dsr`` (``analysis.dsr``) asks "is THIS combo's Sharpe
real after deflating for N trials?"; ``cscv_pbo`` asks the complementary
question — "when we pick the best of N trials, how often does that pick land in
the bottom half out-of-sample?". A high PBO means the sweep is selecting noise
winners that do not generalize, even when the winner's own DSR looks healthy.

Provenance / consolidation
--------------------------
This is a verbatim port of ``alpha_engine_predictor.training.deflated_sharpe
.cscv_pbo`` (ROADMAP L4582), the fleet's first adopter. The backtester is
pinned pre-rename on ``alpha-engine-lib@v0.59.8`` (``alpha_engine_lib``), where
``quant.stats`` exposes ``dsr`` but not a PBO primitive, so — mirroring the
existing ``analysis/dsr.py`` local-shim reality — the implementation is carried
here rather than dragging a full cross-rename lib repin into a risk-gate change.
Second adoption is the consolidation signal: lift ``cscv_pbo`` into
``nousergon_lib.quant.stats`` alongside ``dsr`` and have both the predictor and
this module consume the lib copy once the backtester is repinned across the
v0.60.0 rename boundary. Tracked: see config#950 follow-up.

Pure-compute. Operates on an aligned (n_splits, n_trials) performance matrix; no
I/O.
"""

from __future__ import annotations

import math

import numpy as np


def cscv_pbo(ic_matrix, *, spec_ids=None, min_splits: int = 4) -> dict:
    """CSCV Probability of Backtest Overfitting over an aligned trial matrix
    (Bailey, Borwein, López de Prado & Zhu 2014).

    ``ic_matrix`` is shape ``(n_splits, n_trials)`` — one row per evaluation
    split (here: chronological CSCV blocks of the signal-date range), one column
    per param combo, every cell the SAME performance metric (Sortino for the
    skill-composite axis, Sharpe for legacy) evaluated on that split. All cells
    must come from the same data vintage so rows align.

    Leave-one-split-out symmetric selection test: for each held-out split ``c``,
    pick the combo with the best mean metric over the OTHER splits (in-sample
    selection), read that pick's relative rank ``w ∈ (0,1)`` among combos at
    split ``c`` (out-of-sample), ``lambda = logit(w)``; ``PBO = frac(lambda <=
    0)`` — the probability the in-sample winner lands in the bottom half
    out-of-sample.

    Returns ``status="insufficient"`` (with the reason inline) rather than a
    fabricated number when there are <2 combos or <``min_splits`` clean rows —
    the honest-N/A posture: an insufficient PBO must not silently pass the gate.
    """
    from scipy.stats import rankdata as _rankdata

    m = np.asarray(ic_matrix, dtype=float)
    if m.ndim != 2 or m.shape[1] < 2:
        return {
            "status": "insufficient",
            "reason": "needs >=2 aligned combos",
            "n_splits": 0,
            "n_specs": int(m.shape[1]) if m.ndim == 2 else 0,
            "pbo": float("nan"),
        }
    clean = m[np.isfinite(m).all(axis=1)]
    n_splits, n_specs = clean.shape
    if n_splits < min_splits:
        return {
            "status": "insufficient",
            "reason": f"{n_splits} clean splits < min_splits={min_splits}",
            "n_splits": int(n_splits),
            "n_specs": int(n_specs),
            "pbo": float("nan"),
        }
    ids = list(spec_ids) if spec_ids is not None else list(range(n_specs))
    logits: list[float] = []
    selected_counts: dict = {}
    total = clean.sum(axis=0)
    for c in range(n_splits):
        is_mean = (total - clean[c]) / (n_splits - 1)   # IS = all splits but c
        s_star = int(np.argmax(is_mean))
        selected_counts[ids[s_star]] = selected_counts.get(ids[s_star], 0) + 1
        # OOS relative rank of the IS pick at the held-out split. rankdata
        # average-ties so a degenerate all-equal row yields w=0.5 (logit 0,
        # counted as underperformance — conservative).
        w = float(_rankdata(clean[c])[s_star]) / (n_specs + 1)
        logits.append(math.log(w / (1.0 - w)))
    lam = np.asarray(logits, dtype=float)

    def _r(v, p=6):
        return round(float(v), p) if np.isfinite(v) else float("nan")

    return {
        "status": "ok",
        "n_splits": int(n_splits),
        "n_specs": int(n_specs),
        "spec_ids": ids,
        "pbo": _r(float((lam <= 0).mean())),
        "mean_logit": _r(float(lam.mean())),
        "selected_counts": selected_counts,
    }


__all__ = ["cscv_pbo"]
