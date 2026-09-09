"""Per-stage output-coverage assertion wiring for the two Lambda handlers
in this repo (alpha-engine-config-I7214) — ReplayConcordance and
Counterfactual, the only crucible-backtester stages that are Lambda
handlers rather than spot launchers (see
test_stage_coverage_assertion_wiring.py for the shell-launcher coverage).

Both handlers call the ONE shared `krepis.stage_coverage` module, which
ships from krepis 0.59.4 onward (this repo's `requirements.txt` and both
Lambda Dockerfiles floor krepis at that version so the primitive is
guaranteed present wherever these handlers actually run — a floor that
does not carry the module makes the assertion a silent no-op in the
deployed artifact, which is the alpha-engine-config-I7334 defect class).

These tests verify BOTH sides of the contract, each against a SIMULATED
condition rather than against whatever krepis happens to be installed:

1. module ABSENT — the handler degrades loudly-but-harmlessly (ImportError
   caught, logged at ERROR, handler outcome unchanged). The absence is
   injected (``sys.modules["krepis.stage_coverage"] = None``), not
   inherited from the environment. It used to be inherited: these tests
   asserted "the installed krepis has no such module" as a fact, and went
   red on 2026-08-14 the moment krepis published it, with nothing in this
   repo having changed.
2. module PRESENT — its verdict lands under the `stage_coverage` key in
   the returned payload (fake module injected into sys.modules).

`TestPrimitiveIsActuallyInstalled` pins the third thing neither of those
can see: that the real module is importable at all under this repo's
declared pins. A coverage assertion that cannot import reports nothing,
and reports it in a shape indistinguishable from having found nothing.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent

_HANDLERS = {
    "lambda_concordance": {
        "path": _REPO_ROOT / "lambda_concordance" / "handler.py",
        "compute_target": "replay.batch.compute_and_emit_concordance",
        "sf_stage": "ReplayConcordance",
    },
    "lambda_counterfactual": {
        "path": _REPO_ROOT / "lambda_counterfactual" / "handler.py",
        "compute_target": "replay.counterfactual.compute_and_emit",
        "sf_stage": "Counterfactual",
    },
}


def _load_handler_module(name: str, path: Path):
    module_name = f"{name}_stage_coverage_test_under_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(params=sorted(_HANDLERS))
def handler_case(request):
    name = request.param
    cfg = _HANDLERS[name]
    mod = _load_handler_module(name, cfg["path"])
    mod._init_done = False
    yield name, mod, cfg


def _ok_summary(handler_name: str) -> dict:
    if handler_name == "lambda_concordance":
        return {
            "artifacts_discovered": 3,
            "per_target_model": [
                {
                    "target_model": "deepseek/deepseek-v4-flash",
                    "n_artifacts_replayed": 3,
                    "replay_failures": [],
                    "budget_stopped": False,
                },
            ],
        }
    return {
        "agents_analyzed": 2,
        "agents_skipped_thin_sample": [],
        "agents_unsupported": [],
        "load_failures": [],
        "fit_failures": [],
    }


# ── Simulated absence: krepis.stage_coverage is NOT importable ──────────────


@pytest.fixture
def absent_stage_coverage(monkeypatch):
    """Make ``from krepis.stage_coverage import ...`` raise ImportError.

    ``None`` in ``sys.modules`` is the documented way to force that. The
    root ``tests/conftest.py`` installs the same default for the whole
    suite (so no handler test reaches real S3 through the primitive);
    requesting it explicitly here states the precondition these tests
    depend on instead of borrowing it.
    """
    monkeypatch.setitem(sys.modules, "krepis.stage_coverage", None)
    yield


class TestModuleAbsentDegradesLoudlyNotSilently:
    def test_import_error_does_not_change_handler_outcome(self, handler_case, absent_stage_coverage):
        name, mod, cfg = handler_case
        with patch.object(mod, "_ensure_init"), \
             patch(cfg["compute_target"], return_value=_ok_summary(name)):
            result = mod.handler({"run_date": "2026-09-04"}, context=None)
        # Handler's own status/summary contract is unaffected by the
        # absent module — this IS the observe-mode degrade contract.
        assert result["status"] == "OK"
        assert "summary" in result

    def test_import_error_is_logged_not_swallowed(self, handler_case, absent_stage_coverage):
        name, mod, cfg = handler_case
        with patch.object(mod, "_ensure_init"), \
             patch(cfg["compute_target"], return_value=_ok_summary(name)), \
             patch.object(mod.logger, "error") as mock_error:
            mod.handler({"run_date": "2026-09-04"}, context=None)
        assert mock_error.called, (
            f"{name}: ImportError for krepis.stage_coverage must be "
            f"logged (logger.error), not silently passed"
        )
        logged_msg = mock_error.call_args[0][0]
        assert "stage-coverage" in logged_msg

    def test_stage_coverage_key_absent_when_module_unavailable(self, handler_case, absent_stage_coverage):
        name, mod, cfg = handler_case
        with patch.object(mod, "_ensure_init"), \
             patch(cfg["compute_target"], return_value=_ok_summary(name)):
            result = mod.handler({"run_date": "2026-09-04"}, context=None)
        assert "stage_coverage" not in result


# ── Simulated post-pin-bump environment: module IS importable ───────────────


@pytest.fixture
def fake_stage_coverage_module():
    """Inject a fake krepis.stage_coverage into sys.modules so the
    handler's `from krepis.stage_coverage import assert_stage_coverage`
    succeeds AND returns a verdict this test controls. A fake rather than
    the real module because the real one builds boto3 S3/CloudWatch
    clients and reads a registry object out of S3. Restores prior state on
    teardown so this fixture cannot leak into other tests."""
    calls = []

    def _assert_stage_coverage(stage, *, run_date, window_start):
        calls.append({"stage": stage, "run_date": run_date, "window_start": window_start})
        return {"stage": stage, "run_date": run_date, "status": "COVERED"}

    fake_mod = types.ModuleType("krepis.stage_coverage")
    fake_mod.assert_stage_coverage = _assert_stage_coverage

    had_parent = "krepis" in sys.modules
    parent = sys.modules.get("krepis")
    had_submodule_attr = had_parent and hasattr(parent, "stage_coverage")

    if not had_parent:
        import krepis as parent  # noqa: PLC0415 — real import, establishes sys.modules entry
    sys.modules["krepis.stage_coverage"] = fake_mod
    setattr(parent, "stage_coverage", fake_mod)

    yield calls

    del sys.modules["krepis.stage_coverage"]
    if had_submodule_attr:
        pass  # was already there before us (unlikely) — leave as-is
    else:
        if hasattr(parent, "stage_coverage"):
            delattr(parent, "stage_coverage")


class TestModulePresentVerdictLandsInPayload:
    def test_verdict_merged_under_stage_coverage_key(self, handler_case, fake_stage_coverage_module):
        name, mod, cfg = handler_case
        with patch.object(mod, "_ensure_init"), \
             patch(cfg["compute_target"], return_value=_ok_summary(name)):
            result = mod.handler({"run_date": "2026-09-04"}, context=None)
        assert "stage_coverage" in result, f"{name}: verdict did not land in the returned payload"
        assert result["stage_coverage"]["stage"] == cfg["sf_stage"]
        assert result["stage_coverage"]["status"] == "COVERED"

    def test_called_with_correct_sf_stage_name(self, handler_case, fake_stage_coverage_module):
        name, mod, cfg = handler_case
        with patch.object(mod, "_ensure_init"), \
             patch(cfg["compute_target"], return_value=_ok_summary(name)):
            mod.handler({"run_date": "2026-09-04"}, context=None)
        assert len(fake_stage_coverage_module) == 1
        assert fake_stage_coverage_module[0]["stage"] == cfg["sf_stage"]

    def test_window_start_is_tz_aware_datetime_captured_at_entry(self, handler_case, fake_stage_coverage_module):
        name, mod, cfg = handler_case
        before = datetime.now(timezone.utc)
        with patch.object(mod, "_ensure_init"), \
             patch(cfg["compute_target"], return_value=_ok_summary(name)):
            mod.handler({"run_date": "2026-09-04"}, context=None)
        after = datetime.now(timezone.utc)
        window_start = fake_stage_coverage_module[0]["window_start"]
        assert window_start.tzinfo is not None
        assert before <= window_start <= after

    def test_the_sf_threaded_run_date_beats_the_execution_start_time(
        self, handler_case, fake_stage_coverage_module,
    ):
        """alpha-engine-config-I10171, the defect in one test.

        `end_time_iso` is `$$.Execution.StartTime` — the CALENDAR date —
        while `$.run_date` is the cycle's TRADING day. On the Saturday
        2026-09-05 cycle for trading day 2026-09-04, keying on the calendar
        date wrote the verdict to `_stage_coverage/2026-09-05/`, which no
        reader consults once the dual-partition fallback expired that same
        day. `run_date` must win whenever the SF threads it.
        """
        name, mod, cfg = handler_case
        with patch.object(mod, "_ensure_init"), \
             patch(cfg["compute_target"], return_value=_ok_summary(name)):
            result = mod.handler(
                {"end_time_iso": "2026-09-05T06:00:00Z", "run_date": "2026-09-04"},
                context=None,
            )
        assert fake_stage_coverage_module[0]["run_date"] == "2026-09-04"
        assert result["stage_coverage"]["run_date_source"] == "event.run_date"
        assert "run_date_fallback_reason" not in result["stage_coverage"]

    def test_end_time_iso_is_a_recorded_fallback_not_a_silent_one(
        self, handler_case, fake_stage_coverage_module, caplog,
    ):
        """The pre-step-(1) live Payload shape: no `run_date` on the event.
        The handler still records a verdict — an off-cycle operator
        invocation is legitimate — but it says WHICH field it used, and
        warns. A fallback nobody can see afterwards is the defect itself."""
        import logging

        name, mod, cfg = handler_case
        with caplog.at_level(logging.WARNING), \
             patch.object(mod, "_ensure_init"), \
             patch(cfg["compute_target"], return_value=_ok_summary(name)):
            result = mod.handler({"end_time_iso": "2026-05-09T00:00:00Z"}, context=None)
        assert fake_stage_coverage_module[0]["run_date"] == "2026-05-09"
        assert result["stage_coverage"]["run_date_source"] == "fallback:event.end_time_iso"
        assert result["stage_coverage"]["run_date_fallback_reason"]
        assert any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_no_fabricated_date_when_no_identity_is_on_the_event(
        self, handler_case, fake_stage_coverage_module,
    ):
        """alpha-engine-config-I8155's forbidden class, live in this repo
        until now: `(end_time or _started)` substituted the Lambda's OWN
        wall-clock for a genuinely-absent execution identity, filing a
        verdict against a date no execution ever named. A missing identity
        must record UNMEASURED, never a substitute."""
        name, mod, cfg = handler_case
        with patch.object(mod, "_ensure_init"), \
             patch(cfg["compute_target"], return_value=_ok_summary(name)):
            result = mod.handler({}, context=None)
        assert result["stage_coverage"]["status"] == "UNMEASURED"
        assert result["stage_coverage"]["reason"]
        assert fake_stage_coverage_module == [], (
            "a verdict was filed against a fabricated date"
        )

    def test_verdict_call_does_not_raise_on_error_status_path(self, handler_case, fake_stage_coverage_module):
        """The assertion sits after the try/except around compute_and_emit*
        — an ERROR-status early return must NOT reach the assertion (there
        is no successful output to assert coverage of)."""
        name, mod, cfg = handler_case
        with patch.object(mod, "_ensure_init"), \
             patch(cfg["compute_target"], side_effect=RuntimeError("boom")):
            result = mod.handler({"run_date": "2026-09-04"}, context=None)
        assert result["status"] == "ERROR"
        assert "stage_coverage" not in result
        assert fake_stage_coverage_module == []


# ── dry_run must never assert (and so never overwrite) coverage ─────────────


class TestDryRunNeverAssertsCoverage:
    """alpha-engine-config-I8206.

    Measured 2026-08-22: the real weekly ``ReplayConcordance`` Task
    invocation (05:21:24 PT) replayed the corpus and persisted
    ``decision_artifacts/_replay_summary/2608220900_deepseek-v4-
    flash.json``, and its own embedded assertion recorded ``status:
    COVERED``. ``deploy_concordance.sh``'s deploy-time canary — a
    ``dry_run=True, window_days=14`` invocation of the SAME handler, run
    against Lambda version 240 eleven seconds after that version
    published (18:01:06 UTC / 18:00:55 UTC) — reached the SAME
    unconditional assertion at the end of ``_run``, with ``window_start``
    captured at the CANARY's own (much later) entry time. It overwrote
    ``_stage_coverage/2026-08-22/ReplayConcordance.json`` with `status:
    STALE, covered: []` — a stage that ran and wrote its declared
    artifact hours earlier reported, after the fact, as having produced
    nothing. Only a functional exercise of the ``dry_run=True`` path
    catches this: the sibling ``TestModulePresentVerdictLandsInPayload``
    class above never passes ``dry_run`` and so never reached the branch
    that was wrong.

    lambda_counterfactual is not exercised here: its canary shape and
    dry-run semantics are a separate contract (Counterfactual's own
    handler has no ``dry_run`` kwarg on ``compute_and_emit`` today) —
    this issue and its evidence are ReplayConcordance-specific.
    """

    def test_dry_run_never_calls_assert_stage_coverage(self, fake_stage_coverage_module):
        mod = _load_handler_module("lambda_concordance", _HANDLERS["lambda_concordance"]["path"])
        mod._init_done = False
        with patch.object(mod, "_ensure_init"), \
             patch(
                 "replay.batch.compute_and_emit_concordance",
                 return_value={
                     "dry_run": True,
                     "would_replay": 12,
                     "target_resolution": [{
                         "target_model": "deepseek-v4-flash",
                         "resolved": True,
                         "deployment_id": "deepseek-v4-flash",
                         "route": "litellm_proxy",
                         "exec_context": "lambda",
                     }],
                 },
             ):
            result = mod.handler({"dry_run": True, "window_days": 14}, context=None)
        assert result["status"] == "OK"
        assert fake_stage_coverage_module == [], (
            "a dry_run=True (deploy canary) invocation must never call "
            "assert_stage_coverage — doing so overwrites the real weekly "
            "run's COVERED verdict with a false STALE one (config-I8206)"
        )

    def test_dry_run_result_carries_an_explicit_skip_marker(self, fake_stage_coverage_module):
        mod = _load_handler_module("lambda_concordance", _HANDLERS["lambda_concordance"]["path"])
        mod._init_done = False
        with patch.object(mod, "_ensure_init"), \
             patch(
                 "replay.batch.compute_and_emit_concordance",
                 return_value={
                     "dry_run": True,
                     "would_replay": 12,
                     "target_resolution": [{
                         "target_model": "deepseek-v4-flash",
                         "resolved": True,
                         "deployment_id": "deepseek-v4-flash",
                         "route": "litellm_proxy",
                         "exec_context": "lambda",
                     }],
                 },
             ):
            result = mod.handler({"dry_run": True}, context=None)
        # Never silently omit the key (no-silent-swallows) — a dry run
        # states plainly why no verdict was asserted.
        assert result["stage_coverage"]["stage"] == "ReplayConcordance"
        assert result["stage_coverage"]["status"] == "SKIPPED"

    def test_non_dry_run_still_asserts_coverage(self, handler_case, fake_stage_coverage_module):
        """Guards the fix from over-correcting into never asserting at all."""
        name, mod, cfg = handler_case
        with patch.object(mod, "_ensure_init"), \
             patch(cfg["compute_target"], return_value=_ok_summary(name)):
            mod.handler({"run_date": "2026-09-04"}, context=None)
        assert len(fake_stage_coverage_module) == 1
        assert fake_stage_coverage_module[0]["stage"] == cfg["sf_stage"]


# ── The primitive must actually be installable, not merely called ───────────


class TestPrimitiveIsActuallyInstalled:
    """alpha-engine-config-I7334 class: a coverage assertion whose import
    fails emits nothing, and "emitted nothing" is byte-identical to "found
    nothing wrong". Both simulated paths above pass whether or not krepis
    really carries the module, so one test has to look at the real one."""

    def test_krepis_stage_coverage_is_importable(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "krepis.stage_coverage", raising=False)
        import importlib  # noqa: PLC0415 — deliberately deferred past the delitem

        mod = importlib.import_module("krepis.stage_coverage")
        assert callable(mod.assert_stage_coverage), (
            "krepis.stage_coverage imported but has no assert_stage_coverage — "
            "the handlers' observe-mode assertion would log an AttributeError "
            "and measure nothing"
        )

    def test_assert_stage_coverage_accepts_the_signature_both_handlers_call(
        self, monkeypatch
    ):
        """Both handlers call it as
        ``assert_stage_coverage(stage, run_date=..., window_start=...)``.
        A signature drift in krepis would surface only in production."""
        monkeypatch.delitem(sys.modules, "krepis.stage_coverage", raising=False)
        import importlib  # noqa: PLC0415 — deliberately deferred past the delitem
        import inspect  # noqa: PLC0415

        mod = importlib.import_module("krepis.stage_coverage")
        sig = inspect.signature(mod.assert_stage_coverage)
        sig.bind("ReplayConcordance", run_date="2026-05-09", window_start=None)
