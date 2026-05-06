"""Per-agent agreement scorers for replay output.

Compares an original ``DecisionArtifact.agent_output`` against the
replayed equivalent and emits a single ``agreement_score`` in ``[0, 1]``
plus per-agent diagnostic detail. PR A's ``ReplayOutput.comparison``
field carries this dict.

Why per-agent dispatch (not a single generic scorer):

  Agents emit different output shapes (sector_quant has ranked picks +
  scores; macro_economist has a regime call + sector multipliers;
  ic_cio has per-candidate ADVANCE/REJECT decisions). What "agreement"
  means is structural per agent — there's no useful single metric across
  all of them. The dispatcher resolves the agent_id family and delegates
  to a function that knows the relevant fields.

What's measured per agent:

  - sector_quant: ranked_picks ticker overlap (top-5 + top-10) +
    rank-correlation of quant_score across the overlap.
  - sector_qual: assessments ticker overlap + per-ticker conviction
    correlation.
  - sector_peer_review: recommendations ticker overlap + per-ticker
    accept/reject decision agreement.
  - macro_economist: market_regime exact match + sector_modifiers
    value-pair correlation.
  - ic_cio: advanced_tickers Jaccard + per-decision ADVANCE/REJECT
    agreement.
  - thesis_update: numeric score field diffs + structural similarity
    of bull_case + conviction_rationale.

Unknown agent families fall through to a generic structural scorer
that just compares top-level dict keys + value types. Better to emit
SOME signal than skip.

Cheap-model concordance reads ``agreement_score`` directly: an agent
with rolling mean concordance > 0.9 against Haiku is a candidate for
demotion (deliverable #7's signal #3).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any


# ── Generic helpers ──────────────────────────────────────────────────────


def _jaccard(a: Iterable, b: Iterable) -> float:
    """|A ∩ B| / |A ∪ B|. Returns 0 when both sets are empty (vs the
    mathematician's convention of 1, which would falsely report
    perfect agreement on missing data)."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _overlap_at_k(a: list, b: list, k: int) -> int:
    """|top-k(A) ∩ top-k(B)|. Order-preserving — used for ranked-pick
    overlap where position matters less than presence in the top-K
    cohort."""
    return len(set(a[:k]) & set(b[:k]))


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation. Returns None when the input is too short
    or has zero variance — caller treats None as "not measurable"
    rather than 0 (which would imply uncorrelated, which is different
    from undefined)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


_WORD_RE = re.compile(r"[a-z]+")


def _text_similarity(a: str, b: str) -> float:
    """Word-set Jaccard on lowercased alpha tokens. Cheap, robust to
    minor wording variation, doesn't need scikit-learn or
    sentence-transformers. For deeper similarity (catching paraphrase
    or semantic equivalence) wrap this with an embedding-based scorer
    later — out of scope for the v1 cheap-model-concordance signal."""
    if not a and not b:
        return 0.0
    wa = set(_WORD_RE.findall(a.lower()))
    wb = set(_WORD_RE.findall(b.lower()))
    return _jaccard(wa, wb)


def _safe_list(d: dict, key: str) -> list:
    v = d.get(key)
    return v if isinstance(v, list) else []


# ── Per-agent scorers ────────────────────────────────────────────────────


def _score_sector_quant(orig: dict, repl: dict) -> dict[str, Any]:
    """Compare ranked_picks across ticker overlap + score correlation.

    agreement_score = 0.6 × top5_overlap_jaccard + 0.4 × top10_overlap_jaccard.
    Top-5 weighted heavier because it's the prod-relevant cohort
    (executor reads top-5 picks for entry candidates).
    """
    o_picks = _safe_list(orig, "ranked_picks")
    r_picks = _safe_list(repl, "ranked_picks")
    o_tickers = [p.get("ticker") for p in o_picks if isinstance(p, dict)]
    r_tickers = [p.get("ticker") for p in r_picks if isinstance(p, dict)]

    top5_jaccard = _jaccard(o_tickers[:5], r_tickers[:5])
    top10_jaccard = _jaccard(o_tickers[:10], r_tickers[:10])

    # Score correlation across overlapping tickers (where comparable).
    overlap = set(o_tickers) & set(r_tickers)
    o_scores = [
        p["quant_score"] for p in o_picks
        if isinstance(p, dict) and p.get("ticker") in overlap
        and isinstance(p.get("quant_score"), (int, float))
    ]
    r_scores = [
        p["quant_score"] for p in r_picks
        if isinstance(p, dict) and p.get("ticker") in overlap
        and isinstance(p.get("quant_score"), (int, float))
    ]
    score_corr = _pearson(o_scores, r_scores) if len(o_scores) == len(r_scores) else None

    agreement = 0.6 * top5_jaccard + 0.4 * top10_jaccard

    return {
        "agreement_score": agreement,
        "diff_summary": (
            f"top5_jaccard={top5_jaccard:.2f} "
            f"top10_jaccard={top10_jaccard:.2f} "
            f"|overlap|={len(overlap)}"
        ),
        "top5_overlap_count": _overlap_at_k(o_tickers, r_tickers, 5),
        "top5_jaccard": top5_jaccard,
        "top10_jaccard": top10_jaccard,
        "ticker_overlap_count": len(overlap),
        "score_correlation": score_corr,
    }


def _score_sector_qual(orig: dict, repl: dict) -> dict[str, Any]:
    """Compare assessments across ticker overlap + conviction
    correlation. Doesn't compare bull_case prose directly — that's
    cross-week clustering's job, not concordance's."""
    o_a = _safe_list(orig, "assessments")
    r_a = _safe_list(repl, "assessments")
    o_tickers = [a.get("ticker") for a in o_a if isinstance(a, dict)]
    r_tickers = [a.get("ticker") for a in r_a if isinstance(a, dict)]

    ticker_jaccard = _jaccard(o_tickers, r_tickers)
    overlap = set(o_tickers) & set(r_tickers)

    # Conviction correlation across overlapping tickers.
    o_by_t = {a["ticker"]: a for a in o_a if isinstance(a, dict) and a.get("ticker")}
    r_by_t = {a["ticker"]: a for a in r_a if isinstance(a, dict) and a.get("ticker")}
    o_conv = [
        o_by_t[t].get("conviction") for t in overlap
        if isinstance(o_by_t[t].get("conviction"), (int, float))
    ]
    r_conv = [
        r_by_t[t].get("conviction") for t in overlap
        if isinstance(r_by_t[t].get("conviction"), (int, float))
    ]
    conv_corr = _pearson(o_conv, r_conv) if len(o_conv) == len(r_conv) else None

    return {
        "agreement_score": ticker_jaccard,
        "diff_summary": (
            f"ticker_jaccard={ticker_jaccard:.2f} "
            f"|overlap|={len(overlap)}"
        ),
        "ticker_jaccard": ticker_jaccard,
        "ticker_overlap_count": len(overlap),
        "conviction_correlation": conv_corr,
    }


def _score_sector_peer_review(orig: dict, repl: dict) -> dict[str, Any]:
    """Compare recommendations across ticker overlap + accept/reject
    decision agreement. ``additional_accepted`` carries the team's
    optional 6th pick — included in the ticker set."""
    o_recs = _safe_list(orig, "recommendations")
    r_recs = _safe_list(repl, "recommendations")
    o_tickers = [r.get("ticker") for r in o_recs if isinstance(r, dict)]
    r_tickers = [r.get("ticker") for r in r_recs if isinstance(r, dict)]

    # Optional 6th pick.
    o_add = _safe_list(orig, "additional_accepted")
    r_add = _safe_list(repl, "additional_accepted")
    o_all = o_tickers + [t for t in o_add if isinstance(t, str)]
    r_all = r_tickers + [t for t in r_add if isinstance(t, str)]

    ticker_jaccard = _jaccard(o_all, r_all)

    return {
        "agreement_score": ticker_jaccard,
        "diff_summary": f"ticker_jaccard={ticker_jaccard:.2f}",
        "ticker_jaccard": ticker_jaccard,
        "additional_accepted_match": (
            sorted(o_add) == sorted(r_add) if (o_add or r_add) else None
        ),
    }


def _score_macro_economist(orig: dict, repl: dict) -> dict[str, Any]:
    """Compare regime call (exact match) + sector_modifiers numeric
    correlation. The regime call is the highest-impact field (drives
    sector multipliers downstream) so it's weighted 0.6 against
    correlation's 0.4."""
    o_regime = orig.get("market_regime")
    r_regime = repl.get("market_regime")
    regime_match = 1.0 if o_regime == r_regime and o_regime is not None else 0.0

    o_mods = orig.get("sector_modifiers") or {}
    r_mods = repl.get("sector_modifiers") or {}
    common = set(o_mods.keys()) & set(r_mods.keys())
    o_vals = [
        o_mods[s] for s in common if isinstance(o_mods[s], (int, float))
    ]
    r_vals = [
        r_mods[s] for s in common if isinstance(r_mods[s], (int, float))
    ]
    mod_corr = _pearson(o_vals, r_vals) if len(o_vals) == len(r_vals) else None

    # Map None → 0 for the agreement weighting only (not for the raw
    # diagnostic field).
    mod_component = mod_corr if mod_corr is not None else 0.0
    agreement = 0.6 * regime_match + 0.4 * max(0.0, mod_component)

    mod_fmt = f"{mod_corr:.2f}" if mod_corr is not None else "N/A"
    return {
        "agreement_score": agreement,
        "diff_summary": f"regime={o_regime}->{r_regime} mod_corr={mod_fmt}",
        "regime_match": bool(regime_match),
        "original_regime": o_regime,
        "replay_regime": r_regime,
        "sector_modifier_correlation": mod_corr,
        "common_sectors_count": len(common),
    }


def _score_ic_cio(orig: dict, repl: dict) -> dict[str, Any]:
    """Compare advanced_tickers Jaccard + per-decision ADVANCE/REJECT
    agreement on overlap.

    advanced_tickers is the load-bearing CIO output (executor reads
    it for entry candidates), so it's the primary agreement signal.
    """
    o_adv = set(_safe_list(orig, "advanced_tickers"))
    r_adv = set(_safe_list(repl, "advanced_tickers"))
    adv_jaccard = _jaccard(o_adv, r_adv)

    # Per-decision agreement on overlapping candidates.
    o_decs = _safe_list(orig, "ic_decisions")
    r_decs = _safe_list(repl, "ic_decisions")
    o_by_t = {
        d["ticker"]: d.get("decision") for d in o_decs
        if isinstance(d, dict) and d.get("ticker")
    }
    r_by_t = {
        d["ticker"]: d.get("decision") for d in r_decs
        if isinstance(d, dict) and d.get("ticker")
    }
    common_tickers = set(o_by_t.keys()) & set(r_by_t.keys())
    if common_tickers:
        agreed = sum(
            1 for t in common_tickers if o_by_t[t] == r_by_t[t]
        )
        decision_agreement = agreed / len(common_tickers)
    else:
        decision_agreement = None

    # Weight 0.7 advanced_tickers (load-bearing) + 0.3 decision
    # agreement on candidates the two models considered in common.
    decision_component = (
        decision_agreement if decision_agreement is not None else 0.0
    )
    agreement = 0.7 * adv_jaccard + 0.3 * decision_component

    da_fmt = (
        f"{decision_agreement:.2f}" if decision_agreement is not None else "N/A"
    )
    return {
        "agreement_score": agreement,
        "diff_summary": (
            f"advanced_jaccard={adv_jaccard:.2f} decision_agreement={da_fmt}"
        ),
        "advanced_jaccard": adv_jaccard,
        "decision_agreement": decision_agreement,
        "common_candidate_count": len(common_tickers),
        "advanced_tickers_original_count": len(o_adv),
        "advanced_tickers_replay_count": len(r_adv),
    }


def _score_thesis_update(orig: dict, repl: dict) -> dict[str, Any]:
    """Compare numeric scores (final_score, conviction) + structural
    similarity of narrative fields. Per-field word-set Jaccard for
    bull_case + conviction_rationale + thesis_summary.

    Numeric agreement is more informative than text similarity here
    (the held-stock thesis_update is a structured update; prose can
    drift while numerics stay stable, or vice versa). Weight numeric
    0.6 / text 0.4.
    """
    def _abs_diff_norm(a: float | None, b: float | None) -> float | None:
        """1 - |a-b|/100 normalized for fields in [0, 100]. None when
        either side is missing or not numeric."""
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return None
        return max(0.0, 1.0 - abs(a - b) / 100.0)

    score_agree = _abs_diff_norm(orig.get("final_score"), repl.get("final_score"))
    conv_agree = _abs_diff_norm(orig.get("conviction"), repl.get("conviction"))

    text_sims = []
    for k in ("bull_case", "conviction_rationale", "thesis_summary"):
        o = orig.get(k)
        r = repl.get(k)
        if isinstance(o, str) and isinstance(r, str):
            text_sims.append(_text_similarity(o, r))

    numeric = [v for v in (score_agree, conv_agree) if v is not None]
    text = sum(text_sims) / len(text_sims) if text_sims else None

    if numeric and text is not None:
        numeric_avg = sum(numeric) / len(numeric)
        agreement = 0.6 * numeric_avg + 0.4 * text
    elif numeric:
        agreement = sum(numeric) / len(numeric)
    elif text is not None:
        agreement = text
    else:
        agreement = 0.0

    score_fmt = f"{score_agree:.2f}" if score_agree is not None else "N/A"
    text_fmt = f"{text:.2f}" if text is not None else "N/A"
    return {
        "agreement_score": agreement,
        "diff_summary": f"score_agree={score_fmt} text_sim={text_fmt}",
        "final_score_agreement": score_agree,
        "conviction_agreement": conv_agree,
        "text_similarity_avg": text,
    }


def _score_generic(orig: dict, repl: dict) -> dict[str, Any]:
    """Fallback for unknown agent families: compare top-level dict
    keys + value types. Better than skipping — at least surfaces
    structural drift even if we can't measure semantic agreement.

    agreement_score = (matched keys / union keys) — Jaccard on the
    key set. Any structural mismatch (a key present in one side and
    missing from the other) drops the score.
    """
    keys_jaccard = _jaccard(orig.keys() if orig else [], repl.keys() if repl else [])
    return {
        "agreement_score": keys_jaccard,
        "diff_summary": (
            f"key_jaccard={keys_jaccard:.2f} "
            f"orig_keys={sorted(orig.keys()) if orig else []} "
            f"repl_keys={sorted(repl.keys()) if repl else []}"
        ),
        "key_jaccard": keys_jaccard,
        "scorer": "generic",
    }


# ── Dispatch ─────────────────────────────────────────────────────────────


_SCORER_BY_BASE = {
    "sector_quant": _score_sector_quant,
    "sector_qual": _score_sector_qual,
    "sector_peer_review": _score_sector_peer_review,
    "macro_economist": _score_macro_economist,
    "ic_cio": _score_ic_cio,
    "thesis_update": _score_thesis_update,
}


def compute_comparison(
    *,
    agent_id: str,
    original_output: dict,
    replay_output: dict,
) -> dict[str, Any]:
    """Dispatch on ``agent_id`` family, score, return the comparison
    dict. Always returns ``{agreement_score: float, diff_summary: str,
    ...}`` — falls through to ``_score_generic`` for unknown families.

    Empty inputs are handled by each scorer (each maps to 0.0
    agreement when neither side has the relevant fields).
    """
    base_id = (agent_id or "").split(":", 1)[0]
    scorer = _SCORER_BY_BASE.get(base_id, _score_generic)
    result = scorer(original_output or {}, replay_output or {})
    # Stamp the scorer used so consumers can tell whether a generic
    # fallback fired (vs an agent-specific scorer).
    result.setdefault("scorer", base_id if base_id in _SCORER_BY_BASE else "generic")
    result["agent_id_base"] = base_id
    return result
