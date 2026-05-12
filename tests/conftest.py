"""Add project root to sys.path so optimizer.* and analysis.* imports work,
plus centralized arcticdb stubbing for unit tests.

Also pins ``ALPHA_ENGINE_SECRETS_SOURCE=env`` for the test process so
``alpha_engine_lib.secrets.get_secret()`` (post 2026-05-12 .env→SSM
migration, PR 5 of the arc) reads from monkeypatched env vars only —
never real SSM. Set at module-import time so the toggle is in place
before any test module imports emailer.py / analysis/retrain_alert.py.
"""
import sys
import os
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Pin secrets source BEFORE any test module imports a get_secret() caller.
os.environ.setdefault("ALPHA_ENGINE_SECRETS_SOURCE", "env")

# Stub arcticdb by default for all unit tests — they must never hit real S3,
# and CI (GitHub Actions) has no AWS credentials, so real arcticdb calls
# would 403 (observed 2026-04-24 CI on PR #76). Integration tests that need
# the real module (parity replay on the spot) set USE_REAL_ARCTICDB=1 before
# invoking pytest; spot_backtest.sh's parity stage passes this env var.
#
# History: this lived as an unconditional sys.modules.setdefault inside
# test_parity_replay.py — which ran at module-import time and silently
# shadowed the real arcticdb for the parity integration test itself,
# producing a false-positive "ArcticDB universe library returned 0 symbols"
# failure (MagicMock.list_libraries() iterates to []). Moving it here
# centralizes the stub and lets integration tests opt in.
if not os.environ.get("USE_REAL_ARCTICDB"):
    sys.modules.setdefault("arcticdb", MagicMock())


@pytest.fixture(autouse=True)
def _isolate_secrets_from_ssm(monkeypatch):
    """Re-pin ``ALPHA_ENGINE_SECRETS_SOURCE=env`` per test + clear the
    per-process secret cache. See
    ``alpha-engine-docs/private/env-to-ssm-260512.md`` § Risks.
    """
    monkeypatch.setenv("ALPHA_ENGINE_SECRETS_SOURCE", "env")
    try:
        from alpha_engine_lib.secrets import clear_cache
    except ImportError:
        yield
        return
    clear_cache()
    yield
    clear_cache()
