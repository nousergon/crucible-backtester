"""rolling_regime_params.py — 10-year rolling-window regime-dependent
parameter identification (config#952).

WHAT THIS ANSWERS
-----------------
``factor_blend_sensitivity`` asks, over the *full* score_performance span,
"in each market_regime, which stance realizes the best risk-adjusted alpha?"
That is a single snapshot. This module asks the *stability* question Brian
signed off on (Decision Queue 2026-07-07: "i think we want the true 10y"):

  "If we slide a **10-year** window across history and re-identify each
   regime's best stance parameterization in every window, do those
   identified parameters PERSIST, or are they window-specific noise?"

A regime-dependent parameter is only trustworthy if it is *stable* across
rolling windows. A stance whose realized-alpha sign flips window-to-window
is regime noise, not a durable regime parameter — surfacing that is the
whole point (it is the overfitting guardrail, not an afterthought).

DELIBERATE DESIGN DECISIONS (the methodology Brian's ruling authorized)
-----------------------------------------------------------------------
1. **Regime definition is REUSED, not invented.** We key off the canonical
   ``market_regime`` column (bull/bear/neutral) that already drives
   ``regime_analysis`` and ``factor_blend_sensitivity``. Inventing a new
   regime taxonomy here would fork the fleet's regime semantics.
2. **The identified "parameter" is REUSED.** Per (regime, stance) realized
   alpha / Sortino — the exact family ``factor_blend_sensitivity`` and the
   configured ``scoring.yaml aggregator.factor_blend`` weights live in. So a
   window's identified ranking is directly comparable to the configured
   weights, and we are extending existing observability rather than bolting
   on an orthogonal metric.
3. **The window is the TRUE 10 years.** ``WINDOW_DAYS`` is ~10 calendar
   years. When history is shorter than that (the system is younger than
   10y), we do NOT silently shrink the window to whatever fits — that would
   just re-derive the already-shipped short-window signal Brian explicitly
   did NOT want. Instead we emit a single expanding-span estimate and flag
   ``insufficient_history_for_rolling`` loudly, so the "true 10y" rolling
   view activates honestly only once 10y of history exists.
4. **Overfitting guardrails are structural, not cosmetic:**
     (a) a (window, regime, stance) cell contributes to a parameter estimate
         only when it clears ``min_samples`` (``trustworthy``);
     (b) the headline output is cross-window STABILITY (sign-consistency +
         dispersion), which by construction refuses to certify a parameter
         that is only significant in one window;
     (c) analysis-only — see below.

FIREWALL / ANALYSIS-ONLY (do NOT remove)
----------------------------------------
Like ``factor_blend_sensitivity``, this module is OBSERVABILITY. It writes
nothing to S3 / scoring.yaml and returns no "apply" surface. Identified
parameters must NEVER be auto-fed into the configured factor blend: a
10y-rolling fit auto-applied back into scoring would be a curve-fitting loop.
Any future promotion to a recommendation engine is a separate, explicitly
human-gated change — do not wire this dict into an optimizer.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import pandas as pd

from analysis.factor_blend_sensitivity import (
    MIN_TRUSTWORTHY_SAMPLES,
    compute_stance_outcomes,
)

logger = logging.getLogger(__name__)

# ~10 calendar years. The issue's literal, Brian-ratified ask ("the true
# 10y"). Kept in calendar days (score_date is a calendar timestamp) rather
# than ~2520 trading days so the windowing needs no trading-calendar lookup.
WINDOW_DAYS: int = 3653  # 10 * 365 + 3 leap days
# Quarterly step — a rolling identification cadence coarse enough that
# adjacent windows carry meaningfully different tails, fine enough to see a
# regime parameter drift within a couple of years.
STEP_DAYS: int = 91

# A stance parameter is "stable" only if it appears in >= this many windows
# AND its realized-alpha sign is consistent in >= this fraction of them.
_MIN_STABILITY_WINDOWS: int = 2
_SIGN_CONSISTENCY_STABLE: float = 0.75


def load_rolling_regime_frame(db_path: str) -> pd.DataFrame:
    """Load score_performance with canonical outcomes + market_regime.

    Thin reuse of ``regime_analysis.load_with_regime`` (which re-sources
    outcome columns from the long-format store via ``attach_outcomes`` and
    guarantees a ``market_regime`` column). Kept as a named seam so callers
    (evaluate.py wire-in, tests) share one loader.
    """
    from analysis.regime_analysis import load_with_regime

    return load_with_regime(db_path)


def _window_bounds(
    dates: pd.Series, window_days: int, step_days: int
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Rolling ``[start, end]`` inclusive bounds covering ``dates``.

    Anchors the final window's ``end`` at ``max(dates)`` and walks
    backwards by ``step_days`` so the most recent window is always the
    freshest full 10y slice; earlier windows step back from there. Returns
    windows oldest-first. Empty when the span is shorter than one window
    (the caller then falls back to the expanding-span estimate).
    """
    lo = pd.Timestamp(dates.min())
    hi = pd.Timestamp(dates.max())
    span = (hi - lo).days
    if span < window_days:
        return []

    win = pd.Timedelta(days=window_days)
    step = pd.Timedelta(days=step_days)
    bounds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    end = hi
    while end - win >= lo:
        bounds.append((end - win, end))
        end = end - step
    # Guarantee the oldest full window (anchored at lo) is included even if
    # the backward step overshoots it.
    oldest = (lo, lo + win)
    if not bounds or bounds[-1][0] > oldest[0]:
        bounds.append(oldest)
    bounds.reverse()
    return bounds


def _params_from_outcomes(outcomes: pd.DataFrame, min_samples: int) -> dict:
    """Turn a ``compute_stance_outcomes`` frame into a nested
    ``{regime: {stance: {mean_alpha, sortino, n_picks, trustworthy}}}`` dict.

    Every cell is carried (so the report can show thin cells too), but only
    ``trustworthy`` cells (``n_picks >= min_samples``) are later allowed to
    drive a stability verdict.
    """
    params: dict[str, dict[str, dict]] = {}
    if outcomes is None or outcomes.empty:
        return params
    for _, row in outcomes.iterrows():
        regime = str(row["market_regime"])
        stance = str(row["stance"])
        sortino = row.get("sortino")
        params.setdefault(regime, {})[stance] = {
            "mean_alpha": float(row["mean_alpha"]),
            "sortino": None if pd.isna(sortino) else float(sortino),
            "n_picks": int(row["n_picks"]),
            "trustworthy": bool(row["n_picks"] >= min_samples),
        }
    return params


def _top_stance(regime_params: dict) -> Optional[str]:
    """Best stance in a regime by risk-adjusted realized alpha (Sortino),
    among trustworthy cells only. None if no trustworthy cell exists — an
    honest "not yet identified", never a coin-flip.

    A trustworthy cell can have an *undefined* Sortino when it has no
    downside deviation (``factor_blend_sensitivity._sortino`` returns None on
    all-winning cells). That is the ideal case, not a disqualifier: a
    strictly-winning stance ranks at the top, a degenerate one at the bottom.
    Ranking on Sortino-when-defined keeps this consistent with the repo's
    risk-adjusted convention where it matters (cells that actually take
    losses).
    """
    trustworthy = {
        s: p for s, p in regime_params.items() if p["trustworthy"]
    }
    if not trustworthy:
        return None

    def _key(stance: str) -> float:
        p = trustworthy[stance]
        if p["sortino"] is not None:
            return p["sortino"]
        return math.inf if p["mean_alpha"] > 0 else -math.inf

    return max(trustworthy, key=_key)


def summarize_parameter_stability(windows: list[dict], min_samples: int) -> dict:
    """Cross-window stability of each (regime, stance) identified parameter.

    For each (regime, stance) we gather ``mean_alpha`` across every window
    where the cell was ``trustworthy`` and report:
      - ``n_windows_present``   how many windows identified the cell at all
      - ``mean_alpha_mean/std`` central tendency + dispersion of the estimate
      - ``coef_var``            std / |mean| (None when mean ~ 0)
      - ``sign_consistency``    fraction of windows agreeing with the modal sign
      - ``stable``              enough windows AND sign-consistent enough
    """
    # regime -> stance -> list[mean_alpha] over trustworthy windows
    series: dict[str, dict[str, list[float]]] = {}
    for w in windows:
        for regime, stances in w["params"].items():
            for stance, cell in stances.items():
                if not cell["trustworthy"]:
                    continue
                series.setdefault(regime, {}).setdefault(stance, []).append(
                    cell["mean_alpha"]
                )

    stability: dict[str, dict[str, dict]] = {}
    for regime, stances in series.items():
        for stance, alphas in stances.items():
            n = len(alphas)
            mean = sum(alphas) / n
            if n >= 2:
                var = sum((a - mean) ** 2 for a in alphas) / (n - 1)
                std = math.sqrt(var)
            else:
                std = 0.0
            pos = sum(1 for a in alphas if a > 0)
            neg = sum(1 for a in alphas if a < 0)
            modal = max(pos, neg)
            sign_consistency = modal / n if n else 0.0
            coef_var = None if abs(mean) < 1e-12 else std / abs(mean)
            stable = (
                n >= _MIN_STABILITY_WINDOWS
                and sign_consistency >= _SIGN_CONSISTENCY_STABLE
            )
            stability.setdefault(regime, {})[stance] = {
                "n_windows_present": n,
                "mean_alpha_mean": mean,
                "mean_alpha_std": std,
                "coef_var": coef_var,
                "sign_consistency": sign_consistency,
                "stable": stable,
            }
    return stability


def _top_stance_stability(windows: list[dict]) -> dict:
    """Per regime: is the *best* stance the same window-to-window?

    The headline durability signal — a regime whose top-ranked stance keeps
    changing has no stable parameterization even if individual cells look
    stable in isolation.
    """
    tops: dict[str, list[str]] = {}
    for w in windows:
        for regime, regime_params in w["params"].items():
            top = _top_stance(regime_params)
            if top is not None:
                tops.setdefault(regime, []).append(top)

    out: dict[str, dict] = {}
    for regime, seq in tops.items():
        n = len(seq)
        if not n:
            continue
        modal = max(set(seq), key=seq.count)
        out[regime] = {
            "modal_top_stance": modal,
            "top_stance_consistency": seq.count(modal) / n,
            "n_windows_ranked": n,
        }
    return out


def identify_rolling_regime_params(
    df: pd.DataFrame,
    *,
    horizon: str = "21d",
    window_days: int = WINDOW_DAYS,
    step_days: int = STEP_DAYS,
    min_samples: int = MIN_TRUSTWORTHY_SAMPLES,
) -> dict:
    """Identify regime-dependent stance parameters over rolling 10y windows.

    Args:
        df: score_performance rows carrying at least ``score_date``,
            ``market_regime``, ``stance`` and the horizon outcome columns
            (``return_{h}``, ``spy_{h}_return``, ``beat_spy_{h}``).
        horizon: outcome horizon to identify against (default "21d", the
            canonical primary horizon per config#1456).
        window_days / step_days: rolling window + step in calendar days.
        min_samples: (window, regime, stance) cell threshold for trust.

    Returns the artifact dict documented in the module header. Never raises
    on thin/empty data — returns ``status="insufficient"`` with real counts.
    """
    base = {
        "status": "insufficient",
        "horizon": horizon,
        "window_days": window_days,
        "step_days": step_days,
        "min_samples": min_samples,
        "span_days": 0,
        "n_rows": 0,
        "insufficient_history_for_rolling": True,
        "mode": "expanding_single",
        "n_windows": 0,
        "windows": [],
        "stability": {},
        "top_stance_stability": {},
    }

    if df is None or df.empty or "score_date" not in df.columns:
        return base

    work = df.copy()
    work["score_date"] = pd.to_datetime(work["score_date"])
    work = work[work["score_date"].notna()]
    if work.empty:
        return base

    lo = pd.Timestamp(work["score_date"].min())
    hi = pd.Timestamp(work["score_date"].max())
    span_days = int((hi - lo).days)
    base["span_days"] = span_days
    base["n_rows"] = int(len(work))

    bounds = _window_bounds(work["score_date"], window_days, step_days)
    if not bounds:
        # < 10y of history: honest single expanding-span estimate, loudly
        # flagged as NOT the rolling view.
        outcomes = compute_stance_outcomes(work, horizon=horizon)
        params = _params_from_outcomes(outcomes, min_samples)
        windows = [{
            "window_index": 0,
            "start": lo.isoformat(),
            "end": hi.isoformat(),
            "n_rows": int(len(work)),
            "params": params,
        }]
        any_trust = any(
            c["trustworthy"] for st in params.values() for c in st.values()
        )
        base.update({
            "status": "ok" if any_trust else "insufficient",
            "insufficient_history_for_rolling": True,
            "mode": "expanding_single",
            "n_windows": 1,
            "windows": windows,
            # No cross-window stability from a single window — but still
            # report the (regime -> top stance) identification for it.
            "stability": {},
            "top_stance_stability": _top_stance_stability(windows),
        })
        logger.info(
            "rolling_regime_params: span %dd < window %dd — expanding-span "
            "estimate only (rolling 10y view deferred until history matures)",
            span_days, window_days,
        )
        return base

    windows: list[dict] = []
    for i, (start, end) in enumerate(bounds):
        mask = (work["score_date"] >= start) & (work["score_date"] <= end)
        win_df = work.loc[mask]
        outcomes = compute_stance_outcomes(win_df, horizon=horizon)
        windows.append({
            "window_index": i,
            "start": pd.Timestamp(start).isoformat(),
            "end": pd.Timestamp(end).isoformat(),
            "n_rows": int(len(win_df)),
            "params": _params_from_outcomes(outcomes, min_samples),
        })

    stability = summarize_parameter_stability(windows, min_samples)
    any_trust = any(
        c["trustworthy"]
        for w in windows for st in w["params"].values() for c in st.values()
    )
    base.update({
        "status": "ok" if any_trust else "insufficient",
        "insufficient_history_for_rolling": False,
        "mode": "rolling",
        "n_windows": len(windows),
        "windows": windows,
        "stability": stability,
        "top_stance_stability": _top_stance_stability(windows),
    })
    return base


def build_rolling_regime_params_report_section(db_path: Optional[str]) -> str:
    """Self-contained always-emit section for the evaluator email.

    Loads score_performance (+ canonical outcomes + market_regime) from
    ``db_path``, identifies the rolling 10y regime parameters, and renders
    the markdown. Never raises — a missing/unreadable DB yields the
    informational "insufficient data" block, matching the report-section
    contract used by ``build_calibration_section`` / ``build_cost_section``.
    """
    try:
        if not db_path:
            return build_rolling_regime_params_section({"status": "insufficient"})
        df = load_rolling_regime_frame(db_path)
        result = identify_rolling_regime_params(df)
        return build_rolling_regime_params_section(result)
    except Exception as err:  # never sink the evaluator email
        return "\n".join([
            "## 10y rolling regime parameters (stability)",
            "",
            f"- _Section render failed: `{err}`._",
            "  Investigate `analysis/rolling_regime_params.py`.",
            "",
        ])


def _fmt(x: Optional[float], nd: int = 4) -> str:
    return "n/a" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.{nd}f}"


def build_rolling_regime_params_section(result: dict) -> str:
    """Always-emit markdown ``## 10y rolling regime parameters`` section.

    Mirrors the repo's report-section contract: never raises; on thin data
    or a malformed result it returns an informational block, never a crash.
    """
    lines = ["## 10y rolling regime parameters (stability)", ""]
    try:
        if not result or result.get("status") != "ok":
            n_rows = (result or {}).get("n_rows", 0)
            span = (result or {}).get("span_days", 0)
            lines += [
                f"- _Insufficient data to identify regime parameters "
                f"(rows={n_rows}, span={span}d). Reported as they accrue._",
                "",
            ]
            return "\n".join(lines)

        mode = result.get("mode")
        if result.get("insufficient_history_for_rolling"):
            lines += [
                f"- **History < 10y** (span {result.get('span_days', 0)}d < "
                f"window {result.get('window_days')}d): showing a single "
                f"expanding-span identification, NOT the rolling 10y view. "
                f"The true-10y rolling stability activates once history "
                f"reaches the window.",
                "",
            ]
        else:
            lines += [
                f"- Rolling **10y** window, {result.get('n_windows')} windows "
                f"(step {result.get('step_days')}d), horizon "
                f"{result.get('horizon')}, min {result.get('min_samples')} "
                f"picks/cell.",
                "",
            ]

        tss = result.get("top_stance_stability") or {}
        if tss:
            lines += ["**Best stance per regime (durability):**", ""]
            lines += ["| regime | modal top stance | consistency | windows |",
                      "|---|---|---|---|"]
            for regime in sorted(tss):
                r = tss[regime]
                lines.append(
                    f"| {regime} | {r['modal_top_stance']} | "
                    f"{r['top_stance_consistency']:.0%} | "
                    f"{r['n_windows_ranked']} |"
                )
            lines.append("")

        stability = result.get("stability") or {}
        if mode == "rolling" and stability:
            lines += ["**Per-(regime, stance) parameter stability:**", ""]
            lines += [
                "| regime | stance | mean α | σ | sign-consistency | stable |",
                "|---|---|---|---|---|---|",
            ]
            for regime in sorted(stability):
                for stance in sorted(stability[regime]):
                    s = stability[regime][stance]
                    lines.append(
                        f"| {regime} | {stance} | "
                        f"{_fmt(s['mean_alpha_mean'])} | "
                        f"{_fmt(s['mean_alpha_std'])} | "
                        f"{s['sign_consistency']:.0%} | "
                        f"{'yes' if s['stable'] else 'no'} |"
                    )
            lines.append("")
            unstable = [
                f"{regime}/{stance}"
                for regime in stability
                for stance, s in stability[regime].items()
                if not s["stable"] and s["n_windows_present"] >= _MIN_STABILITY_WINDOWS
            ]
            if unstable:
                lines += [
                    "- ⚠️ Sign-unstable across windows (regime noise, do NOT "
                    f"harden into config): {', '.join(sorted(unstable))}",
                    "",
                ]
        lines += [
            "- _Observability only — identified parameters are NOT auto-applied "
            "to the factor blend (curve-fitting firewall)._",
            "",
        ]
        return "\n".join(lines)
    except Exception as err:  # never sink the evaluator email
        return "\n".join([
            "## 10y rolling regime parameters (stability)",
            "",
            f"- _Section render failed: `{err}`._",
            "  Investigate `analysis/rolling_regime_params.py`.",
            "",
        ])
