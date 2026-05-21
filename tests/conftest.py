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


@pytest.fixture(autouse=True)
def _block_real_alerts_publish(monkeypatch):
    """Default-deny real ``alpha_engine_lib.alerts.publish`` for every test.

    History: 2026-05-21 a buggy monkeypatch in test_cost_report.py let the
    real publish through on a failing test run, firing a real Telegram
    alert + (likely) SNS publish to the operator. The corrected test pins
    the publish via ``monkeypatch.setattr`` — but that's opt-in per test;
    a future test that forgets to stub can again reach production channels.

    This autouse fixture closes the recurrence class: every test starts
    with publish replaced by a no-op that returns a synthetic success.
    Tests that want to assert on publish calls override this with their
    own ``monkeypatch.setattr("alpha_engine_lib.alerts.publish", spy)``
    — which works because monkeypatch reverts in LIFO order at teardown.
    """
    try:
        import alpha_engine_lib.alerts  # noqa: F401
    except ImportError:
        # lib pin <v0.21.0 → no alerts module to block. Pre-v0.21.0
        # callers can't reach the channels anyway.
        yield
        return

    class _Chan:
        ok = True
        detail = "blocked by conftest autouse fixture"

    class _Result:
        sns = _Chan()
        telegram = _Chan()
        any_ok = True
        all_ok = True

    def _noop(*args, **kwargs):
        return _Result()

    monkeypatch.setattr("alpha_engine_lib.alerts.publish", _noop)
    yield
