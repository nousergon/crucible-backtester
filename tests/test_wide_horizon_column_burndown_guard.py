"""Burn-down guard v2 — forbid NEW reads of the wide horizon-suffixed
score_performance columns (EPIC config#1483 Phase 3).

The eval horizon was historically encoded in scattered wide-suffixed column
names (`beat_spy_5d`, `beat_spy_21d`, `spy_21d_return`, `return_5d`,
`log_alpha_21d`, …). Changing a horizon meant a fleet-wide rename, and an
incomplete rename silently starved consumers (config#1451/#1452/#1456). The
config#1483 fix makes the horizon a PARAMETER: consumers read the long-format
`score_performance_outcomes` store (one row per signal/date/horizon) filtered by
`nousergon_lib.quant.horizons.HorizonPolicy`, instead of hardcoding `_Nd`
outcome-column literals.

SHARED PRIMITIVE (nousergon-lib#149, v0.78.0; adopted here config#1529)
----------------------------------------------------------------------
The ratchet mechanics — scan production files for wide-column literals (comments
stripped, string literals + docstrings matched), forbid reads outside the
allowlists, and fail on STALE allowlist entries so the sets burn down to {} — now
live in `nousergon_lib.quant.horizon_guard.assert_burndown`. This test consumes
that primitive instead of a local mirror (the second consumer-repo adoption the
prior local guard's docstring anticipated). The `_MIGRATING` / exempt sets stay
repo-side (they are this repo's inventory), and the backtester-specific
`_UNIVERSE_RETURNS_EXEMPT` HONESTY test below stays repo-side too — the shared
primitive intentionally does not encode the universe_returns-vs-score_performance
source distinction, which is a property of THIS repo's tables.

config#1529 completed the analysis/reporting READ-cluster cutover: every file
that read a wide score_performance OUTCOME column now reads the long-format store
via analysis.outcome_store (keyed on HorizonPolicy + OutcomeColumns), so
`_MIGRATING` is empty. The files still MATCHING a wide-column literal are all
genuine `universe_returns` readers (or render universe_returns-sourced artifact
keys whose names incidentally contain a wide-column substring, e.g.
`avg_return_5d`, `log_return_21d`) — permanently exempt.

Relationship to v1 (`test_retired_outcome_columns_guard.py`): v1 guards the
RETIRED 10d/30d SPY-benchmark columns (dead forever, allowlist already {}). v2
is the broader MIGRATION ratchet over the full wide outcome-column set. v1's
retired subset stays permanently forbidden; v2's `_MIGRATING` is now {} —
config#1483 Phase 3 is complete for this repo.
"""

from __future__ import annotations

from pathlib import Path

from nousergon_lib.quant.horizon_guard import assert_burndown, wide_columns_in

_REPO = Path(__file__).resolve().parent.parent

# Directories / files that are not production import paths (mirrors the lib's
# default excludes; `synthetic/gate_calibration.py` is a synthetic-fixture
# generator, not a production consumer of the live store).
_EXCLUDE_PREFIXES = (
    "tests/", ".venv", ".claude/", ".git/", "synthetic/gate_calibration.py",
)

# config#1483 Phase 3 is COMPLETE for the backtester analysis/reporting cluster
# (config#1528 optimizers + config#1529 analysis/reporting). No production file
# still reads a wide score_performance OUTCOME column — the allowlist has burned
# down to {}. A NEW entry here would be a REGRESSION (re-introducing the bug
# class); prefer reading analysis.outcome_store instead.
_MIGRATING: frozenset[str] = frozenset()

# Files that read the ambiguous `return_/beat_spy_{N}d` (or substring-colliding)
# literals FROM universe_returns (the upstream source) / render
# universe_returns-sourced artifact keys, NOT score_performance — verified by
# the honesty test below. Permanently exempt: they are not config#1483 consumers
# and cannot burn down. A file that starts reading score_performance outcome
# columns must be MOVED off this set and onto analysis.outcome_store.
_UNIVERSE_RETURNS_EXEMPT = frozenset({
    "optimizer/scanner_optimizer.py",      # SELECT ... FROM universe_returns
    "optimizer/tech_weight_ablation.py",   # universe_returns loadings only
    "analysis/shadow_book.py",             # SELECT ... FROM universe_returns
    "analysis/cio_rule_tag_precision.py",  # SELECT ur.beat_spy_5d FROM universe_returns
    "analysis/macro_eval.py",              # SELECT ... FROM universe_returns
    # analysis/quant_rank_quality.py left this set 2026-07-06 (config#1529):
    # its universe_returns column names now resolve from HorizonPolicy
    # (OutcomeColumns attribute access), so no wide-column literal remains.
    "analysis/end_to_end.py",              # universe_returns joins + locally-computed alpha
    "reporter.py",                         # renders universe_returns-sourced artifact keys
    "evaluate.py",                         # reads u.log_return_21d FROM universe_returns
})


def test_wide_horizon_column_burndown():
    """Shared-primitive ratchet: no ungrandfathered wide-column reads, no stale
    `_MIGRATING`/exempt entries, no missing allowlist files. Drives the migration
    to completion and makes the config#1456 bug class un-repeatable."""
    assert_burndown(
        _REPO,
        migrating=_MIGRATING,
        exempt=_UNIVERSE_RETURNS_EXEMPT,
        exclude_prefixes=_EXCLUDE_PREFIXES,
    )


def test_migrating_set_is_empty():
    """config#1529: the analysis/reporting cutover is done — `_MIGRATING` must
    stay {}. A non-empty set here means a consumer regressed onto the wide
    columns instead of reading analysis.outcome_store."""
    assert _MIGRATING == frozenset(), (
        "The wide-column burn-down is complete; a new _MIGRATING entry is a "
        f"regression: {sorted(_MIGRATING)}"
    )


def test_universe_returns_exempt_entries_are_honest():
    """REPO-SPECIFIC (not in the shared primitive): every exempt file must
    (a) still read a wide-column literal and (b) read it FROM universe_returns,
    not score_performance — so the permanent exemption can't silently hide a
    real score_performance outcome-column read. This is the honesty invariant
    the shared `assert_burndown` cannot express (it does not know this repo's
    table topology)."""
    problems = {}
    for rel in _UNIVERSE_RETURNS_EXEMPT:
        text = (_REPO / rel).read_text(errors="ignore")
        if not wide_columns_in(_REPO / rel):
            problems[rel] = "no longer reads any wide-column literal — remove from exempt"
        elif "score_performance" in text and "universe_returns" not in text:
            problems[rel] = "reads score_performance, not universe_returns — migrate to outcome_store"
    assert not problems, f"dishonest _UNIVERSE_RETURNS_EXEMPT entries: {problems}"
