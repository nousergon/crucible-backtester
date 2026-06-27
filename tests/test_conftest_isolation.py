"""Pin the conftest autouse fixtures' contract.

History — both fixtures exist as recurrence-class fixes:

- ``_isolate_secrets_from_ssm``: 2026-05-12 .env→SSM migration risk that
  tests could silently fetch real secrets from real SSM if AWS creds
  were present in the venv.
- ``_block_real_alerts_publish``: 2026-05-21 a buggy monkeypatch in
  test_cost_report.py let the real ``nousergon_lib.alerts.publish``
  through on a failing test run, firing a real Telegram alert + (likely)
  SNS publish to the operator. Default-deny via autouse fixture closes
  the class — a future test bug can never again reach production
  channels just by failing to stub.

These tests fail loudly if the autouse fixtures are removed or weakened,
making the fixture configuration a tested contract not an unobservable
piece of plumbing.
"""
from __future__ import annotations


def test_alerts_publish_is_blocked_without_explicit_setattr():
    """Real publish must be a no-op by default — any test that doesn't
    explicitly stub publish should still NOT reach the production channels.
    """
    try:
        from nousergon_lib import alerts
    except ImportError:
        # Pre-v0.21.0 lib pin → nothing to block.
        return

    # No monkeypatch here — relies entirely on the conftest autouse fixture
    # to have replaced publish with the no-op.
    result = alerts.publish(
        "this should NEVER reach Telegram or SNS",
        severity="error",
        source="test_conftest_isolation",
    )
    # The conftest no-op returns a synthetic-success result so callers
    # that gate on `any_ok` see success-shaped output. Real publish would
    # have returned a PublishResult dataclass whose channels reflect
    # actual transport outcomes.
    assert result.any_ok is True
    assert result.sns.detail == "blocked by conftest autouse fixture"
    assert result.telegram.detail == "blocked by conftest autouse fixture"


def test_per_test_setattr_overrides_conftest_default(monkeypatch):
    """Tests that want to spy on publish via their own monkeypatch must
    still be able to — monkeypatch reverts in LIFO order at teardown so
    the per-test override wins for the duration of the test.
    """
    try:
        from nousergon_lib import alerts
    except ImportError:
        return

    calls: list = []

    def _spy(message, **kwargs):
        calls.append({"message": message, **kwargs})
        return type("R", (), {
            "sns": type("C", (), {"ok": True, "detail": ""}),
            "telegram": type("C", (), {"ok": True, "detail": ""}),
            "any_ok": True, "all_ok": True,
        })

    monkeypatch.setattr("nousergon_lib.alerts.publish", _spy)
    alerts.publish("intercepted by per-test spy", severity="info")
    assert len(calls) == 1
    assert calls[0]["message"] == "intercepted by per-test spy"
    assert calls[0]["severity"] == "info"
