"""Class guard: no `except Exception` handler in this repo's core source
directories may swallow the failure into `logger.debug(...)` or a bare
`pass`, with the root logger at INFO on every backtester entrypoint —
that means the record is emitted **nowhere** (alpha-engine-config-I10226).

Detector is shared via `nousergon_lib.testing.debug_swallow_guard`, lifted
out of `crucible-executor/tests/test_no_debug_only_swallows.py`
(alpha-engine-config-I10031, `crucible-executor-PR547`) on second adoption
per `policy-shared-code`. This file is a thin call-site: it names the
source directories to scan and loads/checks the repo-local allowlist. See
that module's docstring for the exact class of swallow this catches and
what is out of scope (a handler that also re-raises, records via
`logger.error`/`logger.warning`, returns an in-band error value, or calls
`fd.report(...)` is not in scope — the invisible-record shape is
specifically a body with nothing else in it).

A new site with no allowlist entry fails the build (`_UNCOVERED` case
below). An allowlist entry whose `expires` has passed fails loudly —
re-justify or remove, never silently re-grandfather (`_EXPIRED` case). An
entry that no longer matches anything ALSO fails, so the allowance cannot
quietly widen after the site it covered is fixed or moves (`_STALE` case)
— mirrors `.provider-linkage-allowlist.yaml` /
`nousergon-lib/scripts/provider_linkage_guard.py`
(alpha-engine-config-I9295).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from nousergon_lib.testing.debug_swallow_guard import (
    check_against_allowlist,
    check_allowlist_entries_self_contained,
    find_debug_only_swallows,
    load_allowlist,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ALLOWLIST_PATH = _REPO_ROOT / ".debug-swallow-allowlist.yaml"

# Non-recursive, matching find_debug_only_swallows's *.py-directly-under-dir
# shape — this repo has no single top-level package (repo root carries
# root-level modules like backtest.py/evaluate.py/reporter.py alongside
# analysis/, optimizer/, replay/, loaders/, store/, synthetic/, and the
# lambda_* handler directories), so each source directory is scanned
# separately and the results merged, same shape as [tool.coverage.run]'s
# `source = ["."]` comment in pyproject.toml.
_SOURCE_DIRS = [
    _REPO_ROOT,
    _REPO_ROOT / "analysis",
    _REPO_ROOT / "analysis" / "contribution_lift",
    _REPO_ROOT / "lambda_concordance",
    _REPO_ROOT / "lambda_counterfactual",
    _REPO_ROOT / "lambda_health",
    _REPO_ROOT / "loaders",
    _REPO_ROOT / "optimizer",
    _REPO_ROOT / "optimizer" / "arena",
    _REPO_ROOT / "replay",
    _REPO_ROOT / "scripts",
    _REPO_ROOT / "store",
    _REPO_ROOT / "synthetic",
]


def _all_swallow_sites() -> dict[str, set[int]]:
    merged: dict[str, set[int]] = {}
    for source_dir in _SOURCE_DIRS:
        if not source_dir.is_dir():
            continue
        for path, lines in find_debug_only_swallows(source_dir, repo_root=_REPO_ROOT).items():
            if lines:
                merged.setdefault(path, set()).update(lines)
    return merged


def test_no_new_debug_only_swallows_outside_allowlist():
    """Every debug-only-or-pass `except Exception` swallow in this repo's
    core source directories is either fixed (raised, or recorded at
    WARNING/ERROR+) or has a non-expired, matching entry in
    `.debug-swallow-allowlist.yaml`."""
    live_sites = _all_swallow_sites()
    allowlist = load_allowlist(_ALLOWLIST_PATH)
    failures = check_against_allowlist(live_sites, allowlist)
    assert not failures, "\n".join(failures)


def test_allowlist_entries_are_self_contained():
    """Every entry names a reason, an expiry, and a tracking issue — a
    swallow with no named recording surface is not a swallow, it is a
    deletion (alpha-engine-config-I10226 deliverable)."""
    allowlist = load_allowlist(_ALLOWLIST_PATH)
    failures = check_allowlist_entries_self_contained(allowlist)
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
