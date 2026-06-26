"""Pins spot_backtest.sh's executor risk.yaml resolution to experiment-package
precedence (config#1042, step 4).

The spot_backtest.sh SSM/EC2 dispatch path cannot be exercised in CI, so this
is a static-analysis test (mirrors test_spot_backtest_preflight_only.py /
test_spot_backtest_aws_region.py). It guards the structural invariant that the
launcher resolves
  alpha-engine-config/experiments/$ALPHA_ENGINE_EXPERIMENT_ID/executor/risk.yaml
BEFORE the legacy top-level alpha-engine-config/executor/risk.yaml, matching
pipeline_common.load_config + preflight._check_executor_config. config#1159 made
the package copy byte-identical to legacy, so the repoint is behavior-preserving;
these assertions lock the search ORDER against a future edit that would drop the
package path or sort it after legacy.
"""

from __future__ import annotations

from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "infrastructure" / "spot_backtest.sh"
)


def _text() -> str:
    return _SCRIPT.read_text()


def test_spot_backtest_exists():
    assert _SCRIPT.is_file()


def test_experiment_id_defaults_to_reference():
    text = _text()
    assert 'EXPERIMENT_ID="${ALPHA_ENGINE_EXPERIMENT_ID:-reference}"' in text, (
        "launcher must read ALPHA_ENGINE_EXPERIMENT_ID with a `reference` default, "
        "matching pipeline_common.load_config"
    )


def test_package_path_present():
    text = _text()
    assert (
        '"$HOME/alpha-engine-config/experiments/$EXPERIMENT_ID/executor/risk.yaml"'
        in text
    ), "launcher must include the experiment-package risk.yaml candidate"


def test_package_path_precedes_legacy():
    text = _text()
    pkg = text.index(
        "$HOME/alpha-engine-config/experiments/$EXPERIMENT_ID/executor/risk.yaml"
    )
    legacy = text.index('"$HOME/alpha-engine-config/executor/risk.yaml"')
    assert pkg < legacy, (
        "experiment-package risk.yaml must be searched BEFORE the legacy "
        "top-level path (package-first resolution)"
    )


def test_legacy_path_retained_as_fallback():
    text = _text()
    assert '"$HOME/alpha-engine-config/executor/risk.yaml"' in text, (
        "legacy top-level risk.yaml must remain as a fallback — its deletion is "
        "a separate, operator-gated sequencing step (config#1042 step 4 deferred)"
    )
