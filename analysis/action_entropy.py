"""
action_entropy.py — Shannon entropy of the BUY/HOLD/SELL action stream.

Catches a common LLM-trading degenerate-behavior failure mode: the agent
collapses to "always-hold" or "always-trade" — looks fine on returns
because flat is flat, but signals upstream pathology (prompt drift,
confidence collapse, rubric divergence). Standard Sharpe/Sortino/IR
metrics don't catch it; entropy does.

Reports normalized Shannon entropy (in [0, 1]) so a value < 0.3 always
means "pick distribution is collapsing" regardless of action vocabulary
size.

Pure-compute. Operates on a sequence of action labels.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Iterable, TypedDict

import pandas as pd

logger = logging.getLogger(__name__)


# Decision-stream column on the score_performance frame (``df_base``), in
# preference order. ``stance`` is the within-sector factor composite the
# scorer assigns each finalized signal (alpha-engine-research factor_scoring.py:
# momentum / quality / value / low_vol / unknown) — the system's per-decision
# action label. ``conviction`` (rising / stable / declining) is the fallback
# when the stance column hasn't been joined into score_performance yet (it was
# added 2026-05-11 and back-joins lag). Collapse to one stance across a whole
# cycle is the degenerate-behavior tripwire entropy catches: returns-based
# metrics look fine, but the decision distribution has stopped discriminating.
_ACTION_STREAM_COLUMNS: tuple[str, ...] = ("stance", "conviction")


class EntropyResult(TypedDict, total=False):
    status: str
    n: int
    entropy: float           # raw Shannon entropy in nats (natural log)
    entropy_normalized: float  # entropy / log(K) where K = n distinct actions; in [0, 1]
    distribution: dict[str, float]  # action → fraction
    most_common: str
    most_common_fraction: float
    alarm: bool              # True if entropy_normalized < alarm_threshold


def shannon_entropy(distribution: dict[str, float]) -> float:
    """Shannon entropy in nats. Input is a {label: probability} dict.

    Probabilities don't need to sum to 1 — function normalizes
    internally. Zero-probability labels skipped.
    """
    total = sum(p for p in distribution.values() if p > 0)
    if total <= 0:
        return 0.0
    h = 0.0
    for p in distribution.values():
        if p <= 0:
            continue
        norm = p / total
        h -= norm * math.log(norm)
    return h


def compute_action_entropy(
    actions: Iterable[str],
    alarm_threshold: float = 0.3,
    min_samples: int = 10,
) -> EntropyResult:
    """Compute Shannon entropy of an action stream.

    Parameters
    ----------
    actions : iterable of str
        Sequence of action labels (e.g. ['BUY', 'HOLD', 'BUY', 'SELL', ...]).
        Case-sensitive — caller should normalize upstream.
    alarm_threshold : float
        Normalized-entropy floor. Default 0.3 — below this, the
        distribution is judged "collapsed" (>~85% concentration in one
        action for a 3-action vocabulary). Tunable per use case.
    min_samples : int
        Minimum stream length for a meaningful entropy estimate.
        Default 10. Below this returns insufficient_data.

    Returns
    -------
    EntropyResult dict with:
        status: "ok" | "insufficient_data"
        n: stream length
        entropy: raw Shannon entropy (nats)
        entropy_normalized: entropy / log(K) where K = distinct actions;
                            1.0 = uniform, 0.0 = single-action collapse
        distribution: {action: fraction}
        most_common: most frequent action label
        most_common_fraction: fraction of the most common action
        alarm: True if entropy_normalized < alarm_threshold

    Notes
    -----
    Normalization by log(K_observed): if the agent only ever emits 1
    action, K = 1 and the formula is undefined — we return entropy = 0
    and entropy_normalized = 0 explicitly (max-collapse).
    """
    arr = list(actions)
    n = len(arr)
    if n < min_samples:
        return {"status": "insufficient_data", "n": n}

    counts = Counter(arr)
    k = len(counts)
    distribution = {label: count / n for label, count in counts.items()}

    if k <= 1:
        return {
            "status": "ok",
            "n": n,
            "entropy": 0.0,
            "entropy_normalized": 0.0,
            "distribution": distribution,
            "most_common": next(iter(counts.keys())),
            "most_common_fraction": 1.0,
            "alarm": True,
        }

    h = shannon_entropy(distribution)
    h_max = math.log(k)
    h_norm = h / h_max if h_max > 0 else 0.0

    most_common_label, most_common_count = counts.most_common(1)[0]

    return {
        "status": "ok",
        "n": n,
        "entropy": h,
        "entropy_normalized": float(h_norm),
        "distribution": distribution,
        "most_common": most_common_label,
        "most_common_fraction": most_common_count / n,
        "alarm": bool(h_norm < alarm_threshold),
    }


def compute_rolling_entropy(
    actions: pd.Series,
    window: int = 40,
    alarm_threshold: float = 0.3,
) -> pd.DataFrame:
    """Rolling-window action entropy over a time-indexed action stream.

    Parameters
    ----------
    actions : pd.Series
        Indexed by trading day (or any time-like index), values are
        action labels (str).
    window : int
        Rolling window size in observations. Default 40 (~8 weeks of
        trading days; matches the Saturday-SF rationale-clustering
        rolling window).
    alarm_threshold : float
        Normalized-entropy floor for the alarm flag.

    Returns
    -------
    pd.DataFrame indexed by the same time axis as ``actions`` with
    columns:
        entropy, entropy_normalized, alarm

    Rows where the rolling window is incomplete (start of series) are
    dropped to avoid spurious low-entropy alarms on partial data.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    rows: list[dict] = []
    valid_index: list = []
    values = actions.tolist()
    index = actions.index
    for i in range(window - 1, len(values)):
        chunk = values[i - window + 1 : i + 1]
        result = compute_action_entropy(chunk, alarm_threshold=alarm_threshold,
                                        min_samples=2)
        if result.get("status") != "ok":
            continue
        rows.append({
            "entropy": result["entropy"],
            "entropy_normalized": result["entropy_normalized"],
            "alarm": result["alarm"],
        })
        valid_index.append(index[i])

    if not rows:
        return pd.DataFrame(columns=["entropy", "entropy_normalized", "alarm"])
    return pd.DataFrame(rows, index=pd.Index(valid_index, name=actions.index.name))


def compute_action_entropy_artifact(
    df_base: pd.DataFrame | None,
    *,
    alarm_threshold: float = 0.3,
    min_samples: int = 20,
) -> dict:
    """Report-card producer (config#1151 Batch C) — decision-stream entropy.

    Extracts the per-signal decision label (``stance``, falling back to
    ``conviction``) from the finalized-signal frame the evaluator already has
    (``df_base`` = score_performance rows) and computes the normalized Shannon
    entropy of that stream. This is the EXECUTOR-tile ``action_entropy``
    diagnostic the report card grades off ``entropy_normalized``: a value near 1
    means the system is discriminating across the decision vocabulary; a value
    collapsing toward 0 means it has degenerated to one action (prompt drift /
    confidence collapse / rubric divergence) — pathology the Sharpe/Sortino
    surface can't see.

    Pure-compute over already-loaded analysis inputs (no new data read), mirroring
    the ``sample_size_adequacy`` producer. ALWAYS-EMIT (even
    ``insufficient_data`` / ``no_decision_stream``) so the evaluator distinguishes
    "producer didn't run" from "ran, genuinely too few / no labelled decisions".

    Args:
        df_base: score_performance frame (the finalized-signal stream). ``None``
            or empty → ``insufficient_data`` with ``n=0``.
        alarm_threshold: normalized-entropy floor for the collapse alarm.
        min_samples: minimum stream length for a meaningful estimate (default 20,
            matching the executor tile's ``n_floor``).

    Returns the ``compute_action_entropy`` result dict, annotated with
    ``action_field`` (which column supplied the stream) and ``n_signals`` (rows
    seen). ``no_decision_stream`` status when neither label column is present.
    """
    if df_base is None or len(df_base) == 0:
        return {"status": "insufficient_data", "n": 0, "n_signals": 0,
                "reason": "no finalized-signal frame this cycle"}

    field = next(
        (c for c in _ACTION_STREAM_COLUMNS
         if c in df_base.columns and df_base[c].notna().any()),
        None,
    )
    if field is None:
        return {
            "status": "no_decision_stream",
            "n": 0,
            "n_signals": int(len(df_base)),
            "reason": (f"no labelled decision column in score_performance "
                       f"(looked for {list(_ACTION_STREAM_COLUMNS)})"),
        }

    actions = [str(a) for a in df_base[field].dropna().tolist()]
    result = compute_action_entropy(
        actions, alarm_threshold=alarm_threshold, min_samples=min_samples,
    )
    result["action_field"] = field
    result["n_signals"] = int(len(df_base))
    return dict(result)
