"""Regression: spot_backtest.sh never functionally imports ``alpha_engine_lib``.

config#1172 (fleet-wide alpha_engine_lib -> nousergon_lib conversion): the
package was renamed at nousergon-lib v0.60.0 and ``alpha_engine_lib`` is now a
DEPRECATED alias shim, slated for removal in a future major bump. A live grep
during the config#1172 groom pass (2026-07-14) found the RUN_DATE trading-day
normalization chokepoint still doing
``python -c "... from alpha_engine_lib import trading_calendar as tc ..."`` —
a functional (not prose/comment) dependency on the shim that a fleet-wide grep
for ``-m alpha_engine_lib`` (test_no_runpy_alias_invocation-style guards) does
not catch, because this is a bare ``import``, not a ``-m`` runpy invocation.

This guard closes that gap: no box-executed shell script may functionally
import ``alpha_engine_lib`` (via ``import alpha_engine_lib`` or
``from alpha_engine_lib import ...``) — only ``nousergon_lib``. Comment/prose
lines referencing the deprecated name for historical context are exempt.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Bare/`from` import of the deprecated alias — NOT a `-m` runpy invocation
# (that class is covered separately) and NOT a prose mention.
_IMPORT_ALIAS_RE = re.compile(r"\bimport\s+alpha_engine_lib\b")


def _iter_box_scripts():
    infra = _REPO_ROOT / "infrastructure"
    if not infra.is_dir():
        return
    yield from infra.rglob("*.sh")


def _collect_violations():
    violations: list[tuple[Path, int, str]] = []
    for path in _iter_box_scripts():
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            if _IMPORT_ALIAS_RE.search(line):
                violations.append((path.relative_to(_REPO_ROOT), lineno, line.strip()))
    return violations


def test_no_functional_alpha_engine_lib_import_in_box_scripts():
    violations = _collect_violations()
    assert not violations, (
        "Found functional `import alpha_engine_lib` / `from alpha_engine_lib "
        "import ...` in box script(s) — alpha_engine_lib is a deprecated "
        "alias shim over nousergon_lib (config#1172), slated for removal. "
        "Use `nousergon_lib` instead:\n"
        + "\n".join(f"  {p}:{ln}  {src}" for p, ln, src in violations)
    )
