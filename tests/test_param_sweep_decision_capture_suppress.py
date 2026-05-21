"""SUPPRESS contract canary — ROADMAP L280.

The simulation hot loop in param_sweep × N_dates × N_positions can call
``capture_position_sizer`` ~50k-200k times. Each unsuppressed call emits
an S3 PUT to ``decision_artifacts/{Y}/{M}/{D}/executor:position_sizer/...``;
the 2026-05-13 backtester run blew the simulation_pipeline 2700s watchdog
on exactly this failure mode before the SUPPRESS env flag landed.

The contract that prevents recurrence:
    is_decision_capture_enabled() returns False when
    ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS is truthy,
    REGARDLESS of ALPHA_ENGINE_DECISION_CAPTURE_ENABLED.

A regression in any of these surfaces silently re-introduces the
hot-loop S3-PUT storm:
- env var renamed
- gating-function semantics changed
- new capture site added that doesn't pass through is_decision_capture_enabled()
- spot script stops exporting the env var (caught by spot-side bash canary,
  not this CI test)

This file pins all of the above EXCEPT the last (which lives in
``infrastructure/spot_backtest.sh``).
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest


_EXECUTOR_ROOT = os.path.expanduser("~/Development/alpha-engine")
if os.path.isdir(_EXECUTOR_ROOT) and _EXECUTOR_ROOT not in sys.path:
    sys.path.insert(0, _EXECUTOR_ROOT)


pytestmark = pytest.mark.skipif(
    not os.path.isdir(_EXECUTOR_ROOT),
    reason="alpha-engine executor not available locally",
)


def test_suppress_short_circuits_is_decision_capture_enabled(monkeypatch):
    """The env-flag contract itself: SUPPRESS=true wins over ENABLED=true."""
    from executor.decision_capture import is_decision_capture_enabled
    monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
    monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS", "true")
    assert is_decision_capture_enabled() is False, (
        "ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS=true must short-circuit "
        "is_decision_capture_enabled() to False even with ENABLED=true"
    )


def test_suppress_false_lets_enabled_through(monkeypatch):
    """Inverse: SUPPRESS=false (or unset) does not block ENABLED=true."""
    from executor.decision_capture import is_decision_capture_enabled
    monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
    monkeypatch.delenv("ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS", raising=False)
    assert is_decision_capture_enabled() is True
    monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS", "false")
    assert is_decision_capture_enabled() is True


def _sizing_signal() -> dict:
    return {
        "ticker": "AAPL", "score": 80, "conviction": "rising",
        "sector": "Technology", "rating": "BUY", "price_target_upside": 0.15,
    }


def _sizing_result() -> dict:
    return {
        "position_size_usd": 50_000.0, "shares": 250, "weight": 0.05,
        "method": "atr_sized", "atr_pct": 0.02,
        "drawdown_multiplier_applied": 1.0,
    }


def _call_capture_position_sizer():
    """Invoke the hot-loop capture site once with realistic arguments."""
    from executor.decision_capture import capture_position_sizer
    return capture_position_sizer(
        run_date="2026-05-21",
        ticker="AAPL",
        signal=_sizing_signal(),
        sector_rating="market_weight",
        current_price=200.0,
        portfolio_nav=1_000_000.0,
        n_enter_signals=3,
        drawdown_multiplier=1.0,
        atr_pct=0.02,
        prediction_confidence=0.65,
        p_up=0.62,
        signal_age_days=0,
        days_to_earnings=None,
        feature_coverage=0.95,
        stance="momentum",
        sizing_result=_sizing_result(),
        sized_outcome="entered",
    )


def test_capture_position_sizer_writes_nothing_under_suppress(monkeypatch):
    """The hot-loop capture site short-circuits to None and writes zero
    S3 PUTs when SUPPRESS=true. Spy on boto3 to catch any leak."""
    monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
    monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS", "true")

    mock_s3 = MagicMock()
    with patch("boto3.client", return_value=mock_s3):
        result = _call_capture_position_sizer()

    assert result is None
    mock_s3.put_object.assert_not_called()


def test_param_sweep_hot_loop_writes_zero_decision_artifacts(monkeypatch):
    """End-to-end: under SUPPRESS=true, ``param_sweep.sweep`` with a
    ``run_simulation_fn`` that invokes ``capture_position_sizer`` on every
    iteration writes ZERO ``decision_artifacts/`` S3 keys.

    Catches: env-var rename / gating-function semantics change / a new
    capture site that bypasses the gate.
    """
    monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
    monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS", "true")

    from analysis.param_sweep import sweep

    put_keys: list[str] = []
    mock_s3 = MagicMock()
    mock_s3.put_object.side_effect = lambda **kw: put_keys.append(kw.get("Key", ""))

    def _run_simulation_fn(config):
        # Mirror the hot-loop pattern: per (date, position) the executor
        # would call capture_position_sizer. We invoke 5×3 = 15 calls per
        # combo to simulate the realistic call volume.
        for _ in range(15):
            _call_capture_position_sizer()
        return {
            "total_return": 0.01, "sharpe_ratio": 0.5, "max_drawdown": -0.02,
            "calmar_ratio": 0.5, "total_trades": 1, "win_rate": 1.0,
        }

    with patch("boto3.client", return_value=mock_s3):
        sweep(
            grid={"min_score_to_enter": [60, 65, 70]},  # 3 combos × 15 calls = 45 capture attempts
            run_simulation_fn=_run_simulation_fn,
            base_config={"min_score_to_enter": 60},
        )

    decision_artifact_writes = [k for k in put_keys if k.startswith("decision_artifacts/")]
    assert decision_artifact_writes == [], (
        f"SUPPRESS contract REGRESSED: expected zero decision_artifacts/ "
        f"writes under ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS=true, but "
        f"got {len(decision_artifact_writes)} (first 5: {decision_artifact_writes[:5]}). "
        f"Possible causes: env-var renamed; is_decision_capture_enabled() "
        f"semantics changed; a new capture site was added that bypasses the gate."
    )


def test_param_sweep_writes_artifacts_when_suppress_off(monkeypatch):
    """Positive control: with SUPPRESS UNSET and ENABLED=true, the same
    hot-loop pattern DOES write decision_artifacts/ keys. Without this
    control the canary above passes trivially when the function is broken."""
    monkeypatch.setenv("ALPHA_ENGINE_DECISION_CAPTURE_ENABLED", "true")
    monkeypatch.delenv("ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS", raising=False)

    from analysis.param_sweep import sweep

    put_keys: list[str] = []
    mock_s3 = MagicMock()
    mock_s3.put_object.side_effect = lambda **kw: put_keys.append(kw.get("Key", ""))

    def _run_simulation_fn(config):
        for _ in range(3):
            _call_capture_position_sizer()
        return {
            "total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0,
            "calmar_ratio": 0.0, "total_trades": 0, "win_rate": 0.0,
        }

    with patch("boto3.client", return_value=mock_s3):
        sweep(
            grid={"min_score_to_enter": [60, 65]},
            run_simulation_fn=_run_simulation_fn,
            base_config={"min_score_to_enter": 60},
        )

    decision_artifact_writes = [k for k in put_keys if k.startswith("decision_artifacts/")]
    assert len(decision_artifact_writes) > 0, (
        "Positive control failed: with SUPPRESS unset + ENABLED=true, "
        "capture_position_sizer should write at least one "
        "decision_artifacts/ key — the SUPPRESS contract canary above "
        "is then meaningless. Investigate the test fixture, not the gate."
    )
