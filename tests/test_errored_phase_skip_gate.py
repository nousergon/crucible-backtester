"""Tests for the I3281 status:error chokepoint — PhaseRegistry gate against
silently skipping over errored phases via --skip-phases / --force-skip-errored.

The core contract tested here (three-layer chokepoint):
1. Without --force-skip-errored: `should_run` REFUSES to explicitly skip a
   phase whose prior marker has status=error (returns ``force_run_over_errored_marker``).
2. With --force-skip-errored: allows the skip but logs the degradation.
3. ``check_critical_deliverables``: end-of-run hard backstop — any critical
   phase with an error marker fails the stage regardless of force flag.
"""

from __future__ import annotations

from unittest import mock

import pytest

from pipeline_common import PhaseRegistry, _CRITICAL_PHASES


# -- Helpers ----------------------------------------------------------------

_ERROR_MARKER = {"status": "error", "phase": "test_phase"}
_OK_MARKER = {"status": "ok", "phase": "test_phase"}
_CRITICAL_ERROR = {"status": "error", "phase": "simulation_pipeline"}


def _make_registry(
    skip_phases: list[str] | None = None,
    force_skip_errored: bool = False,
    critical_phases: set[str] | None = None,
    markers: dict[str, dict | None] | None = None,
    force: bool = False,
) -> PhaseRegistry:
    """Construct a PhaseRegistry with cached markers (no S3 calls)."""
    reg = PhaseRegistry(
        date="2026-07-23",
        bucket="test-bucket",
        skip_phases=skip_phases,
        force_skip_errored=force_skip_errored,
        critical_phases=critical_phases,
        force=force,
        s3_client=mock.MagicMock(),
    )
    if markers:
        reg._markers = dict(markers)
    return reg


# -- should_run: explicit skip of errored phase (layer 1) --------------------


def test_explicit_skip_ok_marker_allows_skip():
    """An ok-markered phase can be explicitly skipped without force flag — the
    marker says it completed successfully, so skipping it is safe."""
    reg = _make_registry(
        skip_phases=["test_phase"],
        markers={"test_phase": _OK_MARKER},
    )
    run, reason = reg.should_run("test_phase")
    assert not run
    assert reason == "explicit_skip"


def test_explicit_skip_no_marker_allows_skip():
    """A phase with no marker at all can be explicitly skipped — there's no
    evidence it ran before, and the operator is explicitly requesting a skip."""
    reg = _make_registry(
        skip_phases=["test_phase"],
        markers={"test_phase": None},
    )
    run, reason = reg.should_run("test_phase")
    assert not run
    assert reason == "explicit_skip"


def test_explicit_skip_errored_marker_blocked_without_force():
    """Layer 1: without --force-skip-errored, should_run REFUSES to skip a
    phase whose prior marker has status=error, returning force_run_over_errored_marker.
    This forces the operator to either re-run the phase or explicitly acknowledge."""
    reg = _make_registry(
        skip_phases=["test_phase"],
        markers={"test_phase": _ERROR_MARKER},
        force_skip_errored=False,
    )
    run, reason = reg.should_run("test_phase")
    assert run
    assert reason == "force_run_over_errored_marker"


def test_explicit_skip_errored_marker_allowed_with_force():
    """Layer 2: with --force-skip-errored, should_run allows the skip and
    returns explicit_skip_errored_forced. The phase is tracked in
    _errored_skipped_phases for auditing."""
    reg = _make_registry(
        skip_phases=["test_phase"],
        markers={"test_phase": _ERROR_MARKER},
        force_skip_errored=True,
    )
    run, reason = reg.should_run("test_phase")
    assert not run
    assert reason == "explicit_skip_errored_forced"
    assert "test_phase" in reg._errored_skipped_phases


def test_explicit_skip_other_phases_unaffected():
    """Explicit skip of a phase with an ok marker or no marker is not affected
    by the force_skip_errored flag — only errored-marker phases are gated."""
    reg_off = _make_registry(
        skip_phases=["ok_phase", "absent_phase"],
        markers={"ok_phase": _OK_MARKER, "absent_phase": None},
        force_skip_errored=False,
    )
    run_ok, reason_ok = reg_off.should_run("ok_phase")
    assert not run_ok and reason_ok == "explicit_skip"
    run_absent, reason_absent = reg_off.should_run("absent_phase")
    assert not run_absent and reason_absent == "explicit_skip"

    reg_on = _make_registry(
        skip_phases=["ok_phase", "absent_phase"],
        markers={"ok_phase": _OK_MARKER, "absent_phase": None},
        force_skip_errored=True,
    )
    run_ok2, reason_ok2 = reg_on.should_run("ok_phase")
    assert not run_ok2 and reason_ok2 == "explicit_skip"


# -- check_critical_deliverables (layer 3: hard backstop) --------------------


def test_critical_check_passes_with_no_markers():
    """No markers → no critical errors → passes when critical_phases is empty."""
    reg = _make_registry(critical_phases=set())
    # Should not raise.
    reg.check_critical_deliverables()


def test_critical_check_passes_with_ok_markers():
    """All critical phases have status=ok → passes."""
    reg = _make_registry(
        critical_phases={"simulation_pipeline", "predictor_pipeline"},
        markers={
            "simulation_pipeline": {"status": "ok", "phase": "simulation_pipeline"},
            "predictor_pipeline": {"status": "ok", "phase": "predictor_pipeline"},
        },
    )
    reg.check_critical_deliverables()


def test_critical_check_raises_on_errored_critical():
    """Layer 3: a critical phase with status=error marker raises RuntimeError
    in check_critical_deliverables, even without any --skip-phases involvement."""
    reg = _make_registry(
        critical_phases={"simulation_pipeline"},
        markers={"simulation_pipeline": _CRITICAL_ERROR},
    )
    with pytest.raises(RuntimeError, match="simulation_pipeline"):
        reg.check_critical_deliverables()


def test_critical_check_raises_on_errored_critical_with_force():
    """The end-of-run check fires regardless of --force-skip-errored — the
    flag only controls the per-phase skip gate, not the hard backstop."""
    reg = _make_registry(
        critical_phases={"simulation_pipeline"},
        markers={"simulation_pipeline": _CRITICAL_ERROR},
        force_skip_errored=True,
    )
    with pytest.raises(RuntimeError, match="simulation_pipeline"):
        reg.check_critical_deliverables()


def test_critical_check_passes_non_critical_errors_ignored():
    """Non-critical phases with error markers are NOT checked — stages like
    cov_estimator_sweep / gamma_sweep are intentionally non-fatal by design."""
    reg = _make_registry(
        critical_phases={"simulation_pipeline", "predictor_pipeline"},
        markers={
            # Non-critical phases with errors — should not trigger
            "cov_estimator_sweep": {"status": "error", "phase": "cov_estimator_sweep"},
            "gamma_sweep": {"status": "error", "phase": "gamma_sweep"},
            # Critical phases with ok markers
            "simulation_pipeline": {"status": "ok", "phase": "simulation_pipeline"},
            "predictor_pipeline": {"status": "ok", "phase": "predictor_pipeline"},
        },
    )
    # Should not raise
    reg.check_critical_deliverables()


def test_critical_check_reports_all_errored_critical():
    """When multiple critical phases have error markers, all are listed."""
    reg = _make_registry(
        critical_phases={"simulation_pipeline", "predictor_pipeline"},
        markers={
            "simulation_pipeline": _CRITICAL_ERROR,
            "predictor_pipeline": {"status": "error", "phase": "predictor_pipeline"},
        },
    )
    with pytest.raises(RuntimeError) as exc_info:
        reg.check_critical_deliverables()
    msg = str(exc_info.value)
    assert "simulation_pipeline" in msg
    assert "predictor_pipeline" in msg


def test_critical_check_skippable_overrides_respected():
    """When a phase is explicitly in --skip-phases AND has an error marker
    AND force_skip_errored is True, the per-phase check allows it — but the
    end-of-run check STILL catches it for critical phases."""
    reg = _make_registry(
        skip_phases=["simulation_pipeline"],
        critical_phases={"simulation_pipeline"},
        markers={"simulation_pipeline": _CRITICAL_ERROR},
        force_skip_errored=True,
    )
    # Per-phase should_run allows the skip
    run, reason = reg.should_run("simulation_pipeline")
    assert not run
    assert reason == "explicit_skip_errored_forced"
    # But end-of-run still fails
    with pytest.raises(RuntimeError, match="simulation_pipeline"):
        reg.check_critical_deliverables()
