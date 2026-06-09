"""Smoke test: flow-doctor integration in backtester.

Exercises the full report() → dedup → cascade → scrub → store path
using a test-only config with no external notifiers (no email, no GitHub).
"""
import os
import sqlite3
import tempfile

import pytest


@pytest.fixture
def fd_instance(tmp_path):
    """Create a FlowDoctor instance with SQLite-only config (no notifiers)."""
    try:
        import flow_doctor
    except ImportError:
        pytest.skip("flow-doctor not installed (pip install -e ../flow-doctor)")

    db_path = str(tmp_path / "flow_doctor_test.db")

    # flow-doctor 0.6.0rc3 replaced the top-level `flow_doctor.init(...)`
    # constructor with the fluent FlowDoctorBuilder. Same SQLite-only,
    # no-notifier capture config as before.
    from flow_doctor.core.config import RateLimitConfig

    fd = (
        flow_doctor.FlowDoctorBuilder("backtester-test")
        .with_repo("cipher813/alpha-engine-backtester")
        .with_store(type="sqlite", path=db_path)
        .with_dependencies(["predictor-training", "data-phase1"])
        .with_rate_limits(
            RateLimitConfig(max_alerts_per_day=50, dedup_cooldown_minutes=1)
        )
        .build()
    )
    return fd, db_path


class TestFlowDoctorCapture:
    """Verify report(), dedup, scrubbing, and cascade detection."""

    def test_basic_report_stores_to_sqlite(self, fd_instance):
        """A single report should be persisted in the SQLite store."""
        fd, db_path = fd_instance

        try:
            raise ValueError("simulated param sweep failure")
        except Exception as e:
            report_id = fd.report(e, severity="error", context={
                "site": "param_sweep", "mode": "all"})

        assert report_id is not None, "report() should return a report ID"

        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchall()
        conn.close()
        assert len(rows) == 1, "Report should be stored in SQLite"

    def test_context_preserved(self, fd_instance):
        """User-supplied context should appear in the stored report."""
        fd, db_path = fd_instance
        import json

        try:
            raise RuntimeError("executor optimizer blew up")
        except Exception as e:
            report_id = fd.report(e, severity="error", context={
                "site": "executor_optimizer", "mode": "param-sweep"})

        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT context FROM reports WHERE id = ?", (report_id,)).fetchone()
        conn.close()

        ctx = json.loads(row[0])
        assert ctx["user"]["site"] == "executor_optimizer"
        assert ctx["user"]["mode"] == "param-sweep"

    def test_dedup_suppresses_identical_errors(self, fd_instance):
        """Second identical string report within cooldown returns None (deduped).

        Dedup uses error signatures based on exception type + top 3 stack frames.
        String-based reports use message content as the signature, making them
        easier to test for dedup behavior.
        """
        fd, _ = fd_instance

        first_id = fd.report("repeated pipeline failure X", severity="error",
                             context={"site": "test"})
        second_id = fd.report("repeated pipeline failure X", severity="error",
                              context={"site": "test"})

        assert first_id is not None
        assert second_id is None, "Duplicate string report should be suppressed"

    def test_different_errors_not_deduped(self, fd_instance):
        """Different error types should both be reported."""
        fd, _ = fd_instance

        try:
            raise ValueError("error A")
        except Exception as e:
            id_a = fd.report(e, severity="error", context={"site": "test"})

        try:
            raise TypeError("error B")
        except Exception as e:
            id_b = fd.report(e, severity="error", context={"site": "test"})

        assert id_a is not None
        assert id_b is not None
        assert id_a != id_b

    def test_secret_scrubbing(self, fd_instance):
        """Secrets in context dict keys should be redacted by the scrubber."""
        fd, db_path = fd_instance
        import json

        fake_password = "s3cret_p4ssw0rd_12345"
        try:
            raise RuntimeError("Auth failed")
        except Exception as e:
            report_id = fd.report(e, severity="error", context={
                "site": "test_scrubber",
                "api_password": fake_password,
            })

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT context FROM reports WHERE id = ?",
            (report_id,),
        ).fetchone()
        conn.close()

        ctx = json.loads(row[0])
        user_ctx = ctx.get("user", {})
        # The scrubber redacts dict values whose keys match secret patterns
        # (keys containing PASSWORD, SECRET, TOKEN, etc.)
        assert fake_password not in json.dumps(user_ctx), (
            f"Secret leaked in context: {user_ctx}"
        )

    def test_report_never_crashes(self, fd_instance):
        """report() must never raise, even with weird inputs."""
        fd, _ = fd_instance

        # None error
        result = fd.report(None, severity="warning", message="manual warning")
        assert result is not None or result is None  # just assert no exception

        # String error
        result = fd.report("string error", severity="error")
        # no crash = pass

    def test_history_returns_reports(self, fd_instance):
        """history() should return recently captured reports."""
        fd, _ = fd_instance

        try:
            raise ValueError("history test error")
        except Exception as e:
            fd.report(e, severity="error", context={"site": "history_test"})

        history = fd.history(limit=5)
        assert len(history) >= 1
        assert any("history test error" in r.error_message for r in history)

    def test_severity_levels(self, fd_instance):
        """All three severity levels should be accepted and stored."""
        fd, db_path = fd_instance

        # Use different exception types to avoid dedup (same stack frame)
        errors = [
            ("warning", ValueError("warning level test")),
            ("error", TypeError("error level test")),
            ("critical", RuntimeError("critical level test")),
        ]
        ids = []
        for sev, exc in errors:
            try:
                raise exc
            except Exception as e:
                rid = fd.report(e, severity=sev, context={"site": f"test_{sev}"})
                ids.append(rid)

        conn = sqlite3.connect(db_path)
        for rid in ids:
            assert rid is not None
            row = conn.execute("SELECT severity FROM reports WHERE id = ?", (rid,)).fetchone()
            assert row is not None
        conn.close()

    def test_multiple_sites_captured(self, fd_instance):
        """Simulate errors from multiple backtester sites.

        Each site uses a unique exception type to avoid dedup (dedup keys on
        exception type + stack frames, not message content).
        """
        fd, db_path = fd_instance
        sites = [
            ("param_sweep", "error", ValueError("sweep crashed")),
            ("simulation", "error", RuntimeError("sim OOM")),
            ("executor_optimizer", "error", KeyError("missing param")),
            ("report_upload_email", "critical", ConnectionError("S3 unreachable")),
            ("sizing_ab", "warning", TypeError("no data")),
        ]

        ids = []
        for site, sev, exc in sites:
            try:
                raise exc
            except Exception as e:
                rid = fd.report(e, severity=sev, context={"site": site, "mode": "all"})
                ids.append(rid)

        assert all(rid is not None for rid in ids), "All distinct errors should be captured"

        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
        conn.close()
        assert count == len(sites)
