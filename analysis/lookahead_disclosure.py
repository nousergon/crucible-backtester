"""LLM look-ahead-bias disclosure (L4581 · #655, gap G7).

Any backtest that consumes LLM-generated signals over a period that falls
*inside* the agent/judge model's training window is exposed to look-ahead
bias: the model may have "seen the future" of that period during
pre-training, so its signals can encode hindsight that no live trader had.
This is the Look-Ahead-Bench class of concern. It is cheap to disclose,
few harnesses do, and disclosing it is a credibility differentiator for a
published harness — so we surface it as an explicit flag in the experiment
writeup rather than leaving it implicit.

**Design — disclose, never silently pass.** The model training-cutoff
dates are *data*, not hardcoded guesses: they are read from config
(``llm_training_cutoffs``) so an operator owns the authoritative values.
For any model whose cutoff is unknown (absent from config), this module
emits an explicit "cutoff unknown — cannot rule out look-ahead" flag
rather than assuming the backtest is clean. That fail-loud default is the
credibility-correct behaviour: an undisclosed unknown is worse than a
disclosed one.

Pure module: no I/O, no global state. ``build_disclosure`` takes the
backtest window + the model→cutoff mapping and returns a structured
disclosure; ``render_section`` formats it for the markdown report.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any

# Status values for a single model's look-ahead assessment.
STATUS_OVERLAP = "overlap"  # backtest window intersects the training window
STATUS_CLEAN = "clean"  # backtest window is entirely after the cutoff
STATUS_UNKNOWN = "unknown"  # no cutoff on record — cannot rule out look-ahead


def _parse_date(value: Any) -> _dt.date | None:
    """Best-effort parse of an ISO ``YYYY-MM-DD`` (or date/datetime)."""
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    try:
        return _dt.date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class ModelLookahead:
    """Per-model look-ahead assessment for the backtest window."""

    model_id: str
    cutoff: _dt.date | None
    status: str
    note: str


@dataclass(frozen=True)
class LookaheadDisclosure:
    """Structured look-ahead-bias disclosure for one backtest run."""

    backtest_start: _dt.date | None
    backtest_end: _dt.date | None
    models: list[ModelLookahead] = field(default_factory=list)

    @property
    def has_overlap(self) -> bool:
        return any(m.status == STATUS_OVERLAP for m in self.models)

    @property
    def has_unknown(self) -> bool:
        return any(m.status == STATUS_UNKNOWN for m in self.models)

    @property
    def is_clean(self) -> bool:
        """True only if every model is positively known to be clean."""
        return bool(self.models) and all(
            m.status == STATUS_CLEAN for m in self.models
        )


def build_disclosure(
    *,
    backtest_start: Any,
    backtest_end: Any,
    model_ids: list[str],
    training_cutoffs: dict[str, Any] | None,
) -> LookaheadDisclosure:
    """Assess look-ahead exposure for each LLM model over the backtest window.

    Parameters
    ----------
    backtest_start, backtest_end
        The backtest period (ISO date strings / date / datetime / None).
    model_ids
        The LLM model IDs whose signals the backtest consumed (e.g. the
        research ``per_stock_model`` / ``strategic_model``).
    training_cutoffs
        Operator-owned mapping ``model_id -> cutoff date`` (config
        ``llm_training_cutoffs``). Models absent here are flagged UNKNOWN.

    Overlap rule: a model is flagged ``overlap`` when any part of the
    backtest window is on or before its training cutoff (i.e.
    ``backtest_start <= cutoff``). If the start is unknown we conservatively
    treat the window as potentially reaching back into training.
    """
    cutoffs = training_cutoffs or {}
    start = _parse_date(backtest_start)
    end = _parse_date(backtest_end)

    assessments: list[ModelLookahead] = []
    # De-dup while preserving order (per_stock_model and strategic_model can
    # coincide; the same model shouldn't be disclosed twice).
    seen: set[str] = set()
    for model_id in model_ids:
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)

        cutoff = _parse_date(cutoffs.get(model_id))
        if cutoff is None:
            assessments.append(
                ModelLookahead(
                    model_id=model_id,
                    cutoff=None,
                    status=STATUS_UNKNOWN,
                    note=(
                        "no training cutoff on record (config "
                        "`llm_training_cutoffs`); cannot rule out look-ahead"
                    ),
                )
            )
            continue

        # Unknown start => conservatively assume it may reach training.
        overlaps = start is None or start <= cutoff
        if overlaps:
            assessments.append(
                ModelLookahead(
                    model_id=model_id,
                    cutoff=cutoff,
                    status=STATUS_OVERLAP,
                    note=(
                        f"backtest window starts on/before the {cutoff} "
                        f"training cutoff — signals over the overlapping "
                        f"span may encode hindsight"
                    ),
                )
            )
        else:
            assessments.append(
                ModelLookahead(
                    model_id=model_id,
                    cutoff=cutoff,
                    status=STATUS_CLEAN,
                    note=(
                        f"backtest window starts after the {cutoff} training "
                        f"cutoff — no look-ahead from this model"
                    ),
                )
            )

    return LookaheadDisclosure(
        backtest_start=start,
        backtest_end=end,
        models=assessments,
    )


def render_section(disclosure: LookaheadDisclosure) -> list[str]:
    """Render the disclosure as markdown report lines (G7).

    Always emits a section (an absent disclosure is itself a credibility
    gap). Leads with the strongest flag present so a reader can't miss it.
    """
    lines: list[str] = ["## LLM Look-Ahead-Bias Disclosure (G7)", ""]

    if not disclosure.models:
        lines.append(
            "_No LLM models recorded for this run — if this backtest "
            "consumed LLM-generated signals, the look-ahead disclosure is "
            "MISSING and must be wired (config `llm_training_cutoffs` + the "
            "consumed model IDs)._"
        )
        lines.append("")
        return lines

    window = (
        f"{disclosure.backtest_start or '?'} → {disclosure.backtest_end or '?'}"
    )
    if disclosure.has_overlap:
        headline = (
            "⚠️ **LOOK-AHEAD OVERLAP** — the backtest window intersects an "
            "LLM training window; results over the overlap may be optimistic."
        )
    elif disclosure.has_unknown:
        headline = (
            "⚠️ **CUTOFF UNKNOWN** — one or more model cutoffs are not on "
            "record; look-ahead cannot be ruled out."
        )
    else:
        headline = (
            "✅ **CLEAN** — every consumed model's training cutoff predates "
            "the backtest window."
        )

    lines.append(headline)
    lines.append("")
    lines.append(f"Backtest window: `{window}`")
    lines.append("")
    lines.append("| Model | Training cutoff | Status | Note |")
    lines.append("| --- | --- | --- | --- |")
    for m in disclosure.models:
        cutoff_s = m.cutoff.isoformat() if m.cutoff else "unknown"
        lines.append(
            f"| `{m.model_id}` | {cutoff_s} | {m.status} | {m.note} |"
        )
    lines.append("")
    return lines
