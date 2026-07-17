"""Verify the flow-doctor end-of-run heartbeat wiring in backtest.py.

config#646 (Option A: dedicated emit_heartbeat() call site). flow-doctor
ships ``FlowDoctor.emit_heartbeat(bucket, *, prefix=None)`` which writes the
flow's end-of-run ``status()`` snapshot to
``s3://{bucket}/_flow_doctor/heartbeat/{flow_name}/{date}.json``. The
dashboard consumer reads these from the research bucket, so backtest.py's
main() finally-block emits a heartbeat to the same
``config.get("signals_bucket", "alpha-engine-research")`` bucket the health
write targets.

These are source-text + AST checks (no full ``_main_impl()`` invocation —
that needs vectorbt / arcticdb / a full config). They pin:

- the emit_heartbeat call exists, obtains fd via get_flow_doctor(), is
  guarded by ``if <fd>:``, and passes ``bucket=`` (the research bucket);
- it lives inside main()'s finally block, after the health write, before
  instance-stop, so a heartbeat failure can't crash the run;

plus a runtime check that, when ``get_flow_doctor()`` returns a MagicMock,
the wired call path invokes ``emit_heartbeat`` exactly once with the
research bucket.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKTEST_SRC = (REPO_ROOT / "backtest.py").read_text()


def _finally_block_source() -> str:
    """Return the source text of main()/_main_impl()'s trailing finally
    block (the one that performs the health write + instance-stop)."""
    idx = BACKTEST_SRC.find("from nousergon_lib.health import Deliverable, write_health")
    assert idx != -1, "could not locate the health-write finally block"
    # Grab a generous window forward through instance-stop.
    return BACKTEST_SRC[idx : idx + 2000]


class TestHeartbeatWiringSourceShape:
    def test_emit_heartbeat_call_present(self):
        assert "emit_heartbeat(" in BACKTEST_SRC, (
            "backtest.py must call fd.emit_heartbeat() at end-of-run (config#646)"
        )

    def test_heartbeat_obtains_fd_via_get_flow_doctor_and_guards(self):
        block = _finally_block_source()
        assert "get_flow_doctor()" in block, (
            "the heartbeat site must obtain the FlowDoctor singleton via "
            "get_flow_doctor() inside the finally block"
        )
        # Guarded call: `if <name> and hasattr(<name>, "emit_heartbeat"):`
        # immediately preceding an `<name>.emit_heartbeat(` invocation.
        # The hasattr guard makes the wiring forward/backward-compatible
        # during flow-doctor's phased lib rollout: emit_heartbeat only
        # exists in flow-doctor >=0.6.2, so a version-skewed deploy must
        # not AttributeError at end-of-run.
        assert re.search(
            r'if\s+(\w+)\s+and\s+hasattr\(\s*\1\s*,\s*["\']emit_heartbeat["\']\s*\):\s*\n\s*\1\.emit_heartbeat\(',
            block,
        ), (
            "emit_heartbeat must be guarded by "
            '`if <fd> and hasattr(<fd>, "emit_heartbeat"):` — None-safe '
            "(get_flow_doctor may return None) and lib-skew-safe "
            "(emit_heartbeat only exists in flow-doctor >=0.6.2)"
        )

    def test_heartbeat_targets_research_bucket(self):
        block = _finally_block_source()
        # Reuses the `bucket` local computed for the health write, which is
        # config.get("signals_bucket", "alpha-engine-research").
        assert 'config.get("signals_bucket", "alpha-engine-research")' in block, (
            "health/heartbeat bucket must default to the research bucket"
        )
        assert re.search(r"emit_heartbeat\(\s*bucket\s*=\s*bucket\s*\)", block), (
            "emit_heartbeat must be passed bucket=bucket (the research bucket)"
        )

    def test_heartbeat_references_config_646(self):
        block = _finally_block_source()
        assert "config#646" in block or "#646" in block, (
            "the heartbeat site should reference config#646 for provenance"
        )

    def test_heartbeat_is_after_health_write_and_before_instance_stop(self):
        block = _finally_block_source()
        write_idx = block.find("write_health(")
        emit_idx = block.find("emit_heartbeat(")
        stop_idx = block.find("_stop_ec2_instance()")
        assert -1 < write_idx < emit_idx, (
            "emit_heartbeat must run after the health write"
        )
        # instance stop lives outside the inner try; ordering still matters.
        if stop_idx != -1:
            assert emit_idx < stop_idx, (
                "emit_heartbeat must run before the EC2 instance-stop"
            )

    def test_heartbeat_call_is_inside_a_try_block(self):
        """AST check: the emit_heartbeat call node is nested under a
        try/finally so it can never crash the run (belt-and-suspenders on
        top of emit_heartbeat's own soft-fail)."""
        tree = ast.parse(BACKTEST_SRC)

        found = {"in_try": False, "present": False}

        class Visitor(ast.NodeVisitor):
            def visit_Try(self, node: ast.Try) -> None:
                for sub in ast.walk(node):
                    if (
                        isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "emit_heartbeat"
                    ):
                        found["present"] = True
                        found["in_try"] = True
                self.generic_visit(node)

        Visitor().visit(tree)
        assert found["present"], "no emit_heartbeat() call node found in AST"
        assert found["in_try"], (
            "emit_heartbeat() must be nested inside a try block so a failure "
            "cannot crash the run or skip instance-stop"
        )


class TestHeartbeatRuntimeInvocation:
    """Execute just the wired call path with get_flow_doctor stubbed to a
    MagicMock and assert emit_heartbeat is called once with the research
    bucket. Mirrors the guarded source pattern rather than driving the full
    _main_impl() (which needs vectorbt / arcticdb / a full config)."""

    def test_guarded_call_invokes_emit_heartbeat_with_research_bucket(self):
        get_flow_doctor = MagicMock()
        fd = MagicMock()
        get_flow_doctor.return_value = fd
        bucket = "alpha-engine-research"

        # This is exactly the wired shape in backtest.py's finally block.
        # (MagicMock has emit_heartbeat, so the hasattr guard passes.)
        _fd = get_flow_doctor()
        if _fd and hasattr(_fd, "emit_heartbeat"):
            _fd.emit_heartbeat(bucket=bucket)

        fd.emit_heartbeat.assert_called_once_with(bucket="alpha-engine-research")

    def test_guard_skips_when_flow_doctor_none(self):
        get_flow_doctor = MagicMock(return_value=None)
        _fd = get_flow_doctor()
        # No AttributeError: guard prevents .emit_heartbeat on None.
        if _fd and hasattr(_fd, "emit_heartbeat"):
            _fd.emit_heartbeat(bucket="alpha-engine-research")  # pragma: no cover
        assert _fd is None

    def test_guard_skips_on_lib_skew_missing_method(self):
        """A version-skewed flow-doctor (<0.6.2) instance lacks
        emit_heartbeat; the hasattr guard must skip the call rather than
        AttributeError at end-of-run."""

        class _OldFlowDoctor:
            """Mirrors a pre-0.6.2 singleton: has status()/report() but no
            emit_heartbeat."""

            def report(self, *a, **k):  # pragma: no cover - shape only
                pass

        get_flow_doctor = MagicMock(return_value=_OldFlowDoctor())
        _fd = get_flow_doctor()
        called = False
        if _fd and hasattr(_fd, "emit_heartbeat"):
            _fd.emit_heartbeat(bucket="alpha-engine-research")  # pragma: no cover
            called = True
        assert not hasattr(_fd, "emit_heartbeat")
        assert called is False
