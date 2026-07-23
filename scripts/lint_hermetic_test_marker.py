#!/usr/bin/env python3
"""Lint: ensure tests importing live-service modules carry @pytest.mark.live.

Fleet-wide convention (§121 sub-note, alpha-engine-config#3219):
  Tests that hit real network / AWS / subprocess resources MUST be marked
  ``@pytest.mark.live`` (or ``@pytest.mark.parity``, ``@pytest.mark.skipif``)
  or be named ``test_live_*.py`` / ``*_live.py``.  Unmarked tests are assumed
  hermetic (fixture/mock-backed) and the CI step ``addopts = -m "not live and
  not parity"`` ensures they never run by accident on a box without credentials.

This script checks test files for module-level imports of modules that
typically indicate a live-resource test.  It does NOT flag imports inside
function/class bodies, ``pytest.importorskip()`` calls, or conftest.py
(which sets up infrastructure stubs).

Exit code: 0 = clean, 1 = violations found.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Module roots that strongly indicate a test connects to live AWS/network
# resources when imported at module level (outside a mock/stub).
#
# NOTE: ``botocore`` is deliberately excluded — ``from botocore.exceptions
# import ClientError`` is a common pattern in mocked tests for catching S3
# errors, NOT a live-resource indicator.  ``subprocess`` is also excluded
# because many hermetic tests use it for local CLI parsing / git commands
# that never touch the network; the file-naming convention
# (``test_live_*`` / ``*_live.py``) is the right boundary for those.
LIVE_IMPORTS: frozenset[str] = frozenset({
    "boto3",
    "requests",
    "arcticdb",
    "psycopg2",
    "redis",
})

# conftest.py is infrastructure (stubs/mocks), not a test.
# test_live_* / *_live.py are explicitly-named live test files.
_LIVENAME_PREFIX = "test_live_"
_LIVENAME_SUFFIX = "_live.py"
_EXEMPT_FILENAMES: frozenset[str] = frozenset({"conftest.py", "__init__.py"})


def _is_module_level_import(tree: ast.AST, mod_root: str) -> bool:
    """Return True if *mod_root* is imported in a top-level statement."""
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top == mod_root:
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                top = node.module.split(".")[0]
                if top == mod_root:
                    return True
    return False


def _has_live_marker(content: str) -> bool:
    """Return True if the file carries a marker that exempts it from this lint."""
    markers = ("@pytest.mark.live", "@pytest.mark.skipif", "@pytest.mark.parity")
    return any(m in content for m in markers)


def _has_importorskip(content: str) -> bool:
    """``pytest.importorskip`` is a conditional import — the test can skip gracefully."""
    return "pytest.importorskip" in content


def check_test_file(path: Path) -> str | None:
    """Return an error message if *path* violates the convention, else None."""
    name = path.name

    # Exempt known non-test files and explicitly-named live files.
    if name in _EXEMPT_FILENAMES:
        return None
    if name.startswith(_LIVENAME_PREFIX) or name.endswith(_LIVENAME_SUFFIX):
        return None

    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    # Exempt files that already carry a live/skip/parity marker.
    if _has_live_marker(content):
        return None

    # Exempt files that use pytest.importorskip (conditional import pattern).
    if _has_importorskip(content):
        return None

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None  # can't parse; let pytest catch the syntax error

    offenders: list[str] = []
    for mod in sorted(LIVE_IMPORTS):
        if _is_module_level_import(tree, mod):
            offenders.append(mod)

    if offenders:
        joined = ", ".join(offenders)
        return (
            f"{path}: module-level import(s) [{joined}] without "
            f"@pytest.mark.live marker (or @pytest.mark.skipif / @pytest.mark.parity / "
            f"test_live_* / *_live.py naming)"
        )
    return None


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    test_dir = repo_root / "tests"
    if not test_dir.is_dir():
        print(f"ERROR: tests/ directory not found at {test_dir}")
        return 1

    errors: list[str] = []
    for path in sorted(test_dir.iterdir()):
        if path.suffix != ".py":
            continue
        err = check_test_file(path)
        if err is not None:
            errors.append(err)

    if errors:
        sep = "\n"
        print(f"ERROR: {len(errors)} test file(s) violate the hermetic-vs-live convention:\n{sep.join(errors)}")
        return 1

    print("OK — all test files satisfy the hermetic-vs-live convention.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
