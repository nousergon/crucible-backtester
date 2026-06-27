"""Tests for `_publish_executor_opt_rejection_alert` — the (c) named-alert
retrofit closing 5/23-SF P0 sweep item (c).

Pins:
  1. Publishes WARN alert when status != "ok".
  2. NO publish when status == "ok".
  3. Dedup_key includes (run_date, status) so recurring classes don't N-spam.
  4. Suppression respects ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS env var.
  5. ImportError on alerts module is best-effort (logged, no raise).
  6. Publish failure is best-effort (logged, no raise).
  7. `_run_executor_opt` calls the alert helper on the `degraded` short-circuit
     path AND on a non-ok `recommend()` result.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _clear_suppress_env(monkeypatch):
    """Ensure tests run without the suppress env override leaking in."""
    monkeypatch.delenv("ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS", raising=False)


def _make_alerts_module_mock(any_ok: bool = True, dedup_skipped: bool = False):
    mod = MagicMock()
    result = MagicMock()
    result.sns.ok = any_ok
    result.telegram.ok = any_ok
    result.any_ok = any_ok
    result.dedup_skipped = dedup_skipped
    mod.publish.return_value = result
    return mod


def test_publishes_warn_on_non_ok_status():
    from evaluate import _publish_executor_opt_rejection_alert
    result = {"status": "alpha_below_floor", "note": "All combos below floor"}
    config = {"run_date": "2026-05-24"}
    alerts_mod = _make_alerts_module_mock()
    with patch.dict("sys.modules", {"nousergon_lib": MagicMock(alerts=alerts_mod),
                                     "nousergon_lib.alerts": alerts_mod}):
        _publish_executor_opt_rejection_alert(result, config)
    alerts_mod.publish.assert_called_once()
    call_kwargs = alerts_mod.publish.call_args.kwargs
    assert call_kwargs["severity"] == "warning"
    assert "alpha_below_floor" in alerts_mod.publish.call_args.args[0]
    assert "2026-05-24" in alerts_mod.publish.call_args.args[0]


def test_no_publish_on_ok_status():
    from evaluate import _publish_executor_opt_rejection_alert
    result = {"status": "ok"}
    config = {"run_date": "2026-05-24"}
    alerts_mod = _make_alerts_module_mock()
    with patch.dict("sys.modules", {"nousergon_lib": MagicMock(alerts=alerts_mod),
                                     "nousergon_lib.alerts": alerts_mod}):
        _publish_executor_opt_rejection_alert(result, config)
    alerts_mod.publish.assert_not_called()


def test_dedup_key_includes_run_date_and_status():
    from evaluate import _publish_executor_opt_rejection_alert
    result = {"status": "insufficient_data", "note": "Only 5 valid combos"}
    config = {"run_date": "2026-05-30"}
    alerts_mod = _make_alerts_module_mock()
    with patch.dict("sys.modules", {"nousergon_lib": MagicMock(alerts=alerts_mod),
                                     "nousergon_lib.alerts": alerts_mod}):
        _publish_executor_opt_rejection_alert(result, config)
    call_kwargs = alerts_mod.publish.call_args.kwargs
    assert call_kwargs["dedup_key"] == "executor_optimizer_rejected_2026-05-30_insufficient_data"
    assert call_kwargs["dedup_window_min"] == 1440


def test_suppress_env_blocks_publish(monkeypatch):
    monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS", "1")
    from evaluate import _publish_executor_opt_rejection_alert
    result = {"status": "alpha_below_floor"}
    config = {"run_date": "2026-05-24"}
    alerts_mod = _make_alerts_module_mock()
    with patch.dict("sys.modules", {"nousergon_lib": MagicMock(alerts=alerts_mod),
                                     "nousergon_lib.alerts": alerts_mod}):
        _publish_executor_opt_rejection_alert(result, config)
    alerts_mod.publish.assert_not_called()


def test_alerts_import_error_is_best_effort(caplog):
    """If `from nousergon_lib import alerts` fails (lib pin too old or
    deps missing), the helper logs WARN and returns without raising.
    Narrow-scope import patch via meta_path finder to avoid clobbering
    unrelated imports (`os`, `logging`)."""
    import logging
    import importlib.abc
    import importlib.machinery
    from evaluate import _publish_executor_opt_rejection_alert
    result = {"status": "alpha_below_floor"}
    config = {"run_date": "2026-05-24"}

    class _BlockAlertsFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "nousergon_lib.alerts" or fullname == "nousergon_lib":
                raise ImportError(f"blocked for test: {fullname}")
            return None

    import sys
    blocker = _BlockAlertsFinder()
    # Drop any cached `nousergon_lib*` so the import attempt actually runs.
    cached = [k for k in list(sys.modules.keys()) if k.startswith("nousergon_lib")]
    saved = {k: sys.modules.pop(k) for k in cached}
    sys.meta_path.insert(0, blocker)
    try:
        with caplog.at_level(logging.WARNING, logger="evaluate"):
            _publish_executor_opt_rejection_alert(result, config)
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)


def test_publish_exception_is_best_effort():
    from evaluate import _publish_executor_opt_rejection_alert
    result = {"status": "alpha_below_floor"}
    config = {"run_date": "2026-05-24"}
    alerts_mod = MagicMock()
    alerts_mod.publish.side_effect = RuntimeError("SNS unreachable")
    with patch.dict("sys.modules", {"nousergon_lib": MagicMock(alerts=alerts_mod),
                                     "nousergon_lib.alerts": alerts_mod}):
        # Should not raise
        _publish_executor_opt_rejection_alert(result, config)
    alerts_mod.publish.assert_called_once()


def test_run_executor_opt_publishes_on_degraded_short_circuit():
    """When `sweep_df` is None, `_run_executor_opt` short-circuits to
    `status=degraded` and must fire the alert from that branch."""
    from evaluate import _run_executor_opt
    config = {"run_date": "2026-05-24"}
    alerts_mod = _make_alerts_module_mock()
    with patch.dict("sys.modules", {"nousergon_lib": MagicMock(alerts=alerts_mod),
                                     "nousergon_lib.alerts": alerts_mod}):
        result = _run_executor_opt(config, sweep_df=None, freeze=False)
    assert result["status"] == "degraded"
    alerts_mod.publish.assert_called_once()


def test_run_executor_opt_publishes_when_recommend_returns_non_ok():
    """When `executor_optimizer.recommend()` returns a non-ok status,
    `_run_executor_opt` must fire the alert."""
    from evaluate import _run_executor_opt
    config = {"signals_bucket": "alpha-engine-research", "run_date": "2026-05-24"}
    alerts_mod = _make_alerts_module_mock()
    sweep_df = pd.DataFrame({"sharpe_ratio": [0.1, 0.2]})
    with patch("evaluate.executor_optimizer.recommend",
               return_value={"status": "alpha_below_floor",
                             "note": "All 60 combos below floor"}), \
         patch("evaluate.read_params_pit_or_current",
               return_value=None), \
         patch.dict("sys.modules", {"nousergon_lib": MagicMock(alerts=alerts_mod),
                                     "nousergon_lib.alerts": alerts_mod}):
        result = _run_executor_opt(config, sweep_df, freeze=False)
    assert result["status"] == "alpha_below_floor"
    alerts_mod.publish.assert_called_once()
