"""Pins spot_backtest.sh ENV_SOURCE to export AWS_REGION into the spot shell.

The .env-deprecation arc deleted the sourced `.env`, so AWS_REGION/
AWS_DEFAULT_REGION — which boto3 and the lib preflight require — are no
longer set in the spot shell. Same #247 regression as alpha-engine-data's
spot scripts; spot_backtest.sh was in a sibling repo the original arc did
not touch. Surfaced 2026-05-16 (Saturday SF PredictorTraining failure on
the spot_train.sh sibling); audited forward to prevent the identical
Backtester/Parity/Evaluator failure (all three SF states run this script
via --skip-stages).
"""

from __future__ import annotations

import re
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "infrastructure" / "spot_backtest.sh"
)


def test_spot_backtest_exists():
    assert _SCRIPT.is_file()


def test_env_source_exports_aws_region():
    text = _SCRIPT.read_text()
    m = re.search(r"^ENV_SOURCE=.*$", text, re.MULTILINE)
    assert m, "no ENV_SOURCE assignment found — spot_backtest.sh structure changed"
    env_source = m.group(0)
    assert "export AWS_REGION=" in env_source, (
        "ENV_SOURCE must export AWS_REGION — without it the spot shell has no "
        "region (no .env post-deprecation) and boto3 / lib preflight fail. "
        "See 2026-05-16 PredictorTraining sibling failure (#247 class)."
    )
    assert "export AWS_DEFAULT_REGION=" in env_source, (
        "ENV_SOURCE must also export AWS_DEFAULT_REGION."
    )
