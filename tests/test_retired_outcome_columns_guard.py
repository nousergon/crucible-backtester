"""Guard against reads of RETIRED score_performance outcome columns (config#1456).

The canonical-alpha cutover (2026-05-09) retired the 10d/30d outcome horizons in
`score_performance`; resolution now writes the canonical 5d/21d columns
(`beat_spy_5d`, `beat_spy_21d`, `spy_5d_return`, `spy_21d_return`, `return_5d`,
`return_21d`, `log_alpha_21d`). Consumers that still read the dead columns are
silently starved/stale — the bug class behind config#1451/#1452/#1456.

This guard makes that class mechanically un-repeatable:
  * any production (non-test) read of a retired column fails CI, UNLESS the file
    is on `_GRANDFATHERED` (files still being migrated under config#1456);
  * a grandfathered file that is now CLEAN also fails — forcing the allowlist to
    burn down to {} as each migration PR lands (the ratchet).

`return_10d` is NOT retired (the raw 10d stock return still resolves) — only the
SPY-benchmark/beat columns at 10d/30d are dead.

Comments are stripped (tokenize) before matching, so migration-explaining
comments don't trip the guard; string literals + docstrings ARE matched (a SQL
SELECT / dict key / docstring example referencing a dead column is a real
signal).
"""

from __future__ import annotations

import io
import tokenize
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

_RETIRED_COLUMNS = ("beat_spy_10d", "beat_spy_30d", "spy_10d_return", "spy_30d_return")

# Files with KNOWN pre-existing reads, being migrated under config#1456.
# REMOVE each entry in the PR that migrates it — the guard fails if a listed
# file is already clean, so this set is forced to empty as the arc completes.
# config#1456 COMPLETE: all consumers migrated to canonical 5d/21d; allowlist empty.
_GRANDFATHERED = frozenset()

# Directories that are not production import paths.
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


def _dead_columns_in(path: Path) -> list[str]:
    code = _strip_comments(path.read_text(errors="ignore"))
    return [c for c in _RETIRED_COLUMNS if c in code]


def test_no_ungrandfathered_retired_column_reads():
    violations = {}
    for f in _production_py_files():
        rel = f.relative_to(_REPO).as_posix()
        if rel in _GRANDFATHERED:
            continue
        hits = _dead_columns_in(f)
        if hits:
            violations[rel] = hits
    assert not violations, (
        "Production reads of RETIRED score_performance columns (config#1456). "
        "Migrate to canonical 5d/21d / log_alpha_21d, or — only if truly needed — "
        f"add to _GRANDFATHERED with a tracking note:\n{violations}"
    )


def test_grandfathered_set_has_no_stale_entries():
    """Ratchet: a grandfathered file that no longer reads a retired column must
    be REMOVED from _GRANDFATHERED (in its migration PR). Keeps the allowlist
    honest and forces it to empty as config#1456 completes."""
    stale = [
        rel for rel in _GRANDFATHERED
        if not _dead_columns_in(_REPO / rel)
    ]
    assert not stale, (
        "These files are clean — remove them from _GRANDFATHERED (config#1456 "
        f"burn-down): {stale}"
    )
