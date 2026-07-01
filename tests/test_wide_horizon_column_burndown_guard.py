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

This is the RATCHET that drives that migration to completion + makes the bug
class un-repeatable:
  * any production (non-test) read of a wide horizon-suffixed outcome column
    fails CI, UNLESS the file is on `_MIGRATING` (the seed allowlist of files
    that still read the wide columns as of Phase 3 kickoff);
  * a `_MIGRATING` file that is now CLEAN also fails — forcing the allowlist to
    burn down to {} as each consumer-cutover PR lands.

Relationship to v1 (`test_retired_outcome_columns_guard.py`): v1 guards the
RETIRED 10d/30d SPY-benchmark columns (dead forever, allowlist already {}). v2
is the broader MIGRATION ratchet over the full wide outcome-column set
(including the canonical 5d/21d columns consumers must move OFF of onto the
long-format store). v1's retired subset stays permanently forbidden; v2 shrinks
to {} as Phase 3 completes, at which point the horizon is fully parameterized.

Comments are stripped before matching (migration-explaining comments don't trip
it); string literals + docstrings ARE matched (a SQL SELECT / dict key / f-string
building a wide-column name is a real read).

Backtester is a pure READER of these columns (the producer lives in
alpha-engine-data); this guard therefore targets reads cleanly. The same ratchet
should be mirrored into the other consumer repos (predictor / research /
dashboard / evaluator) as their cutovers begin — or lifted to a shared
nousergon_lib testing primitive on the second adoption.
"""

from __future__ import annotations

import io
import tokenize
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

# The wide horizon-suffixed score_performance OUTCOME columns the long-format
# store replaces. Consumers must read score_performance_outcomes (filtered by
# horizon_days) instead. `return_{N}d` is included because the long-format
# `stock_return` replaces it; matches on universe_returns' identically-named
# columns keep a file grandfathered until its score_performance read migrates,
# which is acceptable ratchet friction (never a false PASS).
_WIDE_COLUMNS = (
    "beat_spy_5d", "beat_spy_10d", "beat_spy_21d", "beat_spy_30d",
    "spy_5d_return", "spy_10d_return", "spy_21d_return", "spy_30d_return",
    "return_5d", "return_10d", "return_21d", "return_30d",
    "log_alpha_21d",
)

# Files with KNOWN wide-column reads as of Phase 3 kickoff (2026-07-01), seeded
# from a scan of production files. REMOVE each entry in the PR that migrates it
# to the long-format store — the ratchet (test below) fails if a listed file is
# already clean, forcing this set to empty as config#1483 Phase 3 completes.
_MIGRATING = frozenset({
    "analysis/alpha_distribution.py",
    "analysis/attribution.py",
    "analysis/cio_rule_tag_precision.py",
    "analysis/end_to_end.py",
    "analysis/macro_eval.py",
    "analysis/quant_rank_quality.py",
    "analysis/regime_analysis.py",
    "analysis/regime_stratified_sortino.py",
    "analysis/score_analysis.py",
    "analysis/shadow_book.py",
    "analysis/signal_quality.py",
    "analysis/team_skill_metrics.py",
    "analysis/veto_analysis.py",
    "evaluate.py",
    "optimizer/research_optimizer.py",
    "optimizer/scanner_optimizer.py",
    "optimizer/significance_observe.py",
    "optimizer/stance_sizing_optimizer.py",
    "optimizer/tech_weight_ablation.py",
    "optimizer/weight_optimizer.py",
    "reporter.py",
})

# Directories / files that are not production import paths.
_EXCLUDE_PREFIXES = ("tests/", ".venv", ".claude/", "synthetic/gate_calibration.py")


def _strip_comments(src: str) -> str:
    try:
        return "\n".join(
            tok.string
            for tok in tokenize.generate_tokens(io.StringIO(src).readline)
            if tok.type != tokenize.COMMENT
        )
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return src


def _production_py_files() -> list[Path]:
    out = []
    for f in _REPO.rglob("*.py"):
        rel = f.relative_to(_REPO).as_posix()
        if any(rel.startswith(p) or f"/{p}" in f"/{rel}" for p in _EXCLUDE_PREFIXES):
            continue
        out.append(f)
    return out


def _wide_columns_in(path: Path) -> list[str]:
    code = _strip_comments(path.read_text(errors="ignore"))
    return sorted({c for c in _WIDE_COLUMNS if c in code})


def test_no_ungrandfathered_wide_column_reads():
    """A NEW file reading a wide horizon column (not on _MIGRATING) fails — the
    bug class cannot be reintroduced once Phase 3 completes."""
    violations = {}
    for f in _production_py_files():
        rel = f.relative_to(_REPO).as_posix()
        if rel in _MIGRATING:
            continue
        hits = _wide_columns_in(f)
        if hits:
            violations[rel] = hits
    assert not violations, (
        "Production reads of wide horizon-suffixed score_performance columns "
        "(config#1483). Read the long-format score_performance_outcomes store "
        "filtered by nousergon_lib.quant.horizons.HorizonPolicy instead, or — "
        "only if genuinely unavoidable — add to _MIGRATING with a tracking "
        f"note:\n{violations}"
    )


def test_migrating_set_has_no_stale_entries():
    """Ratchet: a _MIGRATING file that no longer reads any wide column must be
    REMOVED from the allowlist (in its cutover PR). Forces the allowlist to {}
    as config#1483 Phase 3 completes."""
    stale = [rel for rel in _MIGRATING if not _wide_columns_in(_REPO / rel)]
    assert not stale, (
        "These files are clean — remove them from _MIGRATING (config#1483 "
        f"Phase 3 burn-down): {stale}"
    )


def test_seed_allowlist_matches_current_scan():
    """Sanity: every _MIGRATING entry exists + still reads a wide column, and
    no un-listed production file does. Equivalent to the two tests above jointly,
    but asserted as one snapshot so a drift in either direction is obvious at
    seed time."""
    current = {
        f.relative_to(_REPO).as_posix(): _wide_columns_in(f)
        for f in _production_py_files()
    }
    reading = {rel for rel, hits in current.items() if hits}
    assert reading == set(_MIGRATING), (
        "Seed allowlist drift — files reading wide columns but NOT listed: "
        f"{sorted(reading - set(_MIGRATING))}; listed but NOT reading: "
        f"{sorted(set(_MIGRATING) - reading)}"
    )
