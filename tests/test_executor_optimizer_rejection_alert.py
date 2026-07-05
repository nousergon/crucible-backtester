"""Tests for `_publish_executor_opt_rejection_alert` — the (c) named-alert
retrofit closing 5/23-SF P0 sweep item (c).

Pins:
  1. Publishes WARN alert when status != "ok".
  2. NO publish when status == "ok".
  3. Dedup_key includes (run_date, status) so recurring classes don't N-spam.
  4. Suppression respects ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS env var.
  5. ImportError on ops_alerts is best-effort (logged, no raise).
  6. Publish failure is best-effort (logged, no raise).
  7. `_run_executor_opt` calls the alert helper on the `degraded` short-circuit
     path AND on a non-ok `recommend()` result.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _clear_suppress_env(monkeypatch):
    """Ensure tests run without the suppress env override leaking in."""
    monkeypatch.delenv("ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS", raising=False)


def _fake_publish_result(any_ok: bool = True, dedup_skipped: bool = False):
    result = MagicMock()
    result.sns.ok = any_ok
    result.telegram.ok = False
    result.any_ok = any_ok
    result.dedup_skipped = dedup_skipped
    return result


def test_publishes_warn_on_non_ok_status():
    from evaluate import _publish_executor_opt_rejection_alert
    result = {"status": "alpha_below_floor", "note": "All combos below floor"}
    config = {"run_date": "2026-05-24"}
    with patch("ops_alerts.publish_ops_alert", return_value=_fake_publish_result()) as mock_publish:
        _publish_executor_opt_rejection_alert(result, config)
    mock_publish.assert_called_once()
    call_kwargs = mock_publish.call_args.kwargs
    assert call_kwargs["severity"] == "warning"
    assert "alpha_below_floor" in mock_publish.call_args.args[0]
    assert "2026-05-24" in mock_publish.call_args.args[0]


def test_no_publish_on_ok_status():
    from evaluate import _publish_executor_opt_rejection_alert
    result = {"status": "ok"}
    config = {"run_date": "2026-05-24"}
    with patch("ops_alerts.publish_ops_alert") as mock_publish:
        _publish_executor_opt_rejection_alert(result, config)
    mock_publish.assert_not_called()


def test_dedup_key_includes_run_date_and_status():
    from evaluate import _publish_executor_opt_rejection_alert
    result = {"status": "insufficient_data", "note": "Only 5 valid combos"}
    config = {"run_date": "2026-05-30"}
    with patch("ops_alerts.publish_ops_alert", return_value=_fake_publish_result()) as mock_publish:
        _publish_executor_opt_rejection_alert(result, config)
    call_kwargs = mock_publish.call_args.kwargs
    assert call_kwargs["dedup_key"] == "executor_optimizer_rejected_2026-05-30_insufficient_data"
    assert call_kwargs["dedup_window_min"] == 1440


def test_suppress_env_blocks_publish(monkeypatch):
    monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS", "1")
    from evaluate import _publish_executor_opt_rejection_alert
    result = {"status": "alpha_below_floor"}
    config = {"run_date": "2026-05-24"}
    with patch("ops_alerts.publish_ops_alert") as mock_publish:
        _publish_executor_opt_rejection_alert(result, config)
    mock_publish.assert_not_called()


def test_ops_alerts_import_error_is_best_effort(caplog):
    import importlib.abc
    import logging
    import sys
    from evaluate import _publish_executor_opt_rejection_alert

    class _BlockOpsAlertsFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "ops_alerts":
                raise ImportError("blocked for test: ops_alerts")
            return None

    blocker = _BlockOpsAlertsFinder()
    saved = sys.modules.pop("ops_alerts", None)
    sys.meta_path.insert(0, blocker)
    try:
        with caplog.at_level(logging.WARNING, logger="evaluate"):
            _publish_executor_opt_rejection_alert(
                {"status": "alpha_below_floor"},
                {"run_date": "2026-05-24"},
            )
    finally:
        sys.meta_path.remove(blocker)
        if saved is not None:
            sys.modules["ops_alerts"] = saved


def test_publish_exception_is_best_effort():
    from evaluate import _publish_executor_opt_rejection_alert
    result = {"status": "alpha_below_floor"}
    config = {"run_date": "2026-05-24"}
    with patch(
        "ops_alerts.publish_ops_alert",
        side_effect=RuntimeError("SNS unreachable"),
    ) as mock_publish:
        _publish_executor_opt_rejection_alert(result, config)
    mock_publish.assert_called_once()


def test_run_executor_opt_publishes_on_degraded_short_circuit():
    from evaluate import _run_executor_opt
    config = {"run_date": "2026-05-24"}
    with patch("ops_alerts.publish_ops_alert", return_value=_fake_publish_result()) as mock_publish:
        result = _run_executor_opt(config, sweep_df=None, freeze=False)
    assert result["status"] == "degraded"
    mock_publish.assert_called_once()


def test_run_executor_opt_publishes_when_recommend_returns_non_ok():
    from evaluate import _run_executor_opt
    config = {"signals_bucket": "alpha-engine-research", "run_date": "2026-05-24"}
    sweep_df = pd.DataFrame({"sharpe_ratio": [0.1, 0.2]})
    with patch("evaluate.executor_optimizer.recommend",
               return_value={"status": "alpha_below_floor",
                             "note": "All 60 combos below floor"}), \
         patch("evaluate.read_params_pit_or_current",
               return_value=None), \
         patch("ops_alerts.publish_ops_alert", return_value=_fake_publish_result()) as mock_publish:
        result = _run_executor_opt(config, sweep_df, freeze=False)
    assert result["status"] == "alpha_below_floor"
    mock_publish.assert_called_once()
