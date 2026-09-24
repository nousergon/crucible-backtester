"""The spot launchers' ops-alert fan-out runs under $LIB_PYTHON.

alpha-engine-config-I11508. Both cleanup traps (``spot_backtest.sh`` and the
``_spot_common.sh`` one that ``spot_backtester.sh`` and
``spot_portfolio_optimizer_backtest.sh`` source) used to probe
``$(dirname "$0")/../.venv/bin/python`` and fall back to bare ``python3``.
The dispatcher box never builds this repo's ``.venv``, and system python3
has no ``nousergon_lib``, so every fan-out died at ``from ops_alerts import``
— and ``> /dev/null 2>&1`` threw away the traceback that said so. The weekly
rehearsal of 2026-09-23 printed only "(ops alert fan-out failed ...)".
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_INFRA = Path(__file__).resolve().parent.parent / "infrastructure"


def _fanout_block(name: str) -> str:
    text = (_INFRA / name).read_text()
    start = text.index("local _alert_python _alert_msg")
    end = text.index("ops alert fan-out failed", start)
    return text[start : end + 200]


@pytest.mark.parametrize("name", ["spot_backtest.sh", "_spot_common.sh"])
def test_fanout_uses_the_launchers_declared_interpreter(name):
    """The same $LIB_PYTHON the launcher resolves krepis through — the one
    interpreter the host is known to carry nousergon_lib in (see
    test_launchers_resolve_the_declared_krepis_guard.py for which it is)."""
    block = _fanout_block(name)
    assert '_alert_python="$LIB_PYTHON"' in block
    assert ".venv/bin/python\" ]" not in block, "no repo-.venv probe"
    assert "command -v python3" not in block, "no bare-python3 fallback"


@pytest.mark.parametrize("name", ["spot_backtest.sh", "_spot_common.sh"])
def test_fanout_failure_names_its_cause(name):
    block = _fanout_block(name)
    assert re.search(r"2>&1 >/dev/null\)\"", block), (
        "stderr must be captured, not discarded"
    )
    assert "${_fo_err##*" in block, "the failure line must carry the error"
