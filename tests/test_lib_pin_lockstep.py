"""Pin ``requirements.txt`` and all three Lambda ``Dockerfile``s to the same
alpha-engine-lib version.

The backtester repo ships three Lambdas (lambda_health, lambda_concordance,
lambda_counterfactual), each with its own Dockerfile and hardcoded
``pip install "alpha-engine-lib@vX.Y.Z"`` line. They don't read
``requirements.txt`` for the lib install, so bumping the project-root pin
alone leaves all three Lambda images stuck on whatever tag was hardcoded.

This drift class has bitten production multiple times across the org:

  - 2026-05-06 (research): requirements.txt bumped @v0.4.0 → @v0.5.1
    but Dockerfile kept v0.3.0; Research Lambda canary failed with
    ``ModuleNotFoundError: alpha_engine_lib.agent_schemas``.
  - 2026-05-12 (predictor + data): requirements bumped to @v0.12.0 but
    Lambda-side pins stayed stale; canary failed with
    ``ModuleNotFoundError: alpha_engine_lib.secrets``.

This test re-greps all four deploy artifacts on every CI run.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_REQUIREMENTS_PIN_RE = re.compile(
    r"alpha-engine-lib\[[^\]]*\]\s*@\s*git\+https://github\.com/cipher813/alpha-engine-lib@(v[0-9]+\.[0-9]+\.[0-9]+)"
)
_DOCKERFILE_PIN_RE = re.compile(
    r'"alpha-engine-lib\[[^\]]*\]\s*@\s*git\+https://github\.com/cipher813/alpha-engine-lib@(v[0-9]+\.[0-9]+\.[0-9]+)"'
)


def _read_pin(filename: str, regex: re.Pattern[str]) -> str:
    text = (_REPO_ROOT / filename).read_text()
    match = regex.search(text)
    assert match is not None, (
        f"could not find alpha-engine-lib pin in {filename}"
    )
    return match.group(1)


def test_all_deploy_artifacts_pin_same_lib_version():
    pins = {
        "requirements.txt": _read_pin("requirements.txt", _REQUIREMENTS_PIN_RE),
        "lambda_health/Dockerfile": _read_pin("lambda_health/Dockerfile", _DOCKERFILE_PIN_RE),
        "lambda_concordance/Dockerfile": _read_pin(
            "lambda_concordance/Dockerfile", _DOCKERFILE_PIN_RE
        ),
        "lambda_counterfactual/Dockerfile": _read_pin(
            "lambda_counterfactual/Dockerfile", _DOCKERFILE_PIN_RE
        ),
    }
    unique = set(pins.values())
    assert len(unique) == 1, (
        f"alpha-engine-lib pin drift across deploy artifacts:\n"
        + "\n".join(f"  {name}: {pin}" for name, pin in pins.items())
        + "\n\nAll four must move in lockstep — each Lambda Dockerfile has "
        f"its own hardcoded `pip install \"alpha-engine-lib@vX.Y.Z\"` line "
        f"that is independent of requirements.txt."
    )
