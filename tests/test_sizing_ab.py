"""Tests for analysis.sizing_ab — A/B comparison of current sizing vs equal-weight."""

from copy import deepcopy

import pytest

from analysis.sizing_ab import _MIN_TRADES, run_sizing_ab


def _base_config():
    return {
        "atr_sizing_enabled": True,
        "confidence_sizing_enabled": True,
        "staleness_discount_enabled": True,
        "earnings_sizing_enabled": True,
        "sector_adj": {"overweight": 1.2, "market_weight": 1.0, "underweight": 0.8},
        "conviction_decline_adj": 0.5,
        "upside_fail_adj": 0.6,
    }


def _make_sim_fn(stats_by_config_key, key_fn=None):
    """Build a sim_fn that returns different stats depending on config shape.

    ``key_fn`` extracts a key from the config to look up stats. Default uses
    the atr_sizing_enabled flag — A config has it on, B has it off.
    """
    if key_fn is None:
        key_fn = lambda c: bool(c.get("atr_sizing_enabled"))
    calls = []

    def sim_fn(config):
        calls.append(deepcopy(config))
        return stats_by_config_key[key_fn(config)]

    sim_fn.calls = calls  # type: ignore[attr-defined]
    return sim_fn


def test_sizing_helps_when_current_outperforms():
    sim_fn = _make_sim_fn({
        True: {"total_trades": 100, "sharpe_ratio": 1.50, "total_return": 0.18, "total_alpha": 0.06, "max_drawdown": 0.10},
        False: {"total_trades": 100, "sharpe_ratio": 1.20, "total_return": 0.15, "total_alpha": 0.04, "max_drawdown": 0.12},
    })

    result = run_sizing_ab(sim_fn, _base_config())

    assert result["status"] == "ok"
    assert result["assessment"] == "sizing_helps"
    assert result["sharpe_diff"] == pytest.approx(0.30)
    assert result["return_diff"] == pytest.approx(0.03)
    assert result["alpha_diff"] == pytest.approx(0.02)
    assert result["current_sizing"]["total_trades"] == 100
    assert result["equal_weight"]["total_trades"] == 100
    assert "Current sizing Sharpe" in result["detail"]


def test_equal_weight_better_when_diff_negative():
    sim_fn = _make_sim_fn({
        True: {"total_trades": 80, "sharpe_ratio": 0.80, "total_return": 0.10, "total_alpha": 0.02, "max_drawdown": 0.18},
        False: {"total_trades": 80, "sharpe_ratio": 1.30, "total_return": 0.15, "total_alpha": 0.05, "max_drawdown": 0.10},
    })

    result = run_sizing_ab(sim_fn, _base_config())

    assert result["status"] == "ok"
    assert result["assessment"] == "equal_weight_better"
    assert result["sharpe_diff"] == pytest.approx(-0.50)
    assert "Equal-weight Sharpe" in result["detail"]


def test_no_difference_when_within_tolerance():
    sim_fn = _make_sim_fn({
        True: {"total_trades": 60, "sharpe_ratio": 1.005, "total_return": 0.10, "total_alpha": 0.03, "max_drawdown": 0.11},
        False: {"total_trades": 60, "sharpe_ratio": 1.00, "total_return": 0.10, "total_alpha": 0.03, "max_drawdown": 0.11},
    })

    result = run_sizing_ab(sim_fn, _base_config())

    assert result["status"] == "ok"
    assert result["assessment"] == "no_difference"
    assert abs(result["sharpe_diff"]) < 0.1


def test_insufficient_data_below_min_trades():
    sim_fn = _make_sim_fn({
        True: {"total_trades": 10, "sharpe_ratio": 1.5},
        False: {"total_trades": 12, "sharpe_ratio": 1.4},
    })

    result = run_sizing_ab(sim_fn, _base_config(), min_trades=_MIN_TRADES)

    assert result["status"] == "insufficient_data"
    assert result["trades_a"] == 10
    assert result["trades_b"] == 12
    assert result["min_required"] == _MIN_TRADES


def test_sim_fn_exception_returns_error():
    def failing_sim(config):
        raise RuntimeError("simulator blew up")

    result = run_sizing_ab(failing_sim, _base_config())

    assert result["status"] == "error"
    assert "simulator blew up" in result["error"]


def test_empty_stats_returns_error():
    sim_fn = _make_sim_fn({True: {}, False: {}})
    # Both branches return empty → status="error"
    result = run_sizing_ab(sim_fn, _base_config(), min_trades=1)
    assert result["status"] == "error"


def test_config_b_disables_sizing_knobs():
    """Verify Config B mutation matches the contract: sizing flags off, sector_adj neutral."""
    captured = []

    def sim_fn(config):
        captured.append(deepcopy(config))
        return {"total_trades": 100, "sharpe_ratio": 1.0, "total_return": 0.1, "total_alpha": 0.02, "max_drawdown": 0.1}

    run_sizing_ab(sim_fn, _base_config())

    assert len(captured) == 2
    config_a, config_b = captured

    assert config_a["atr_sizing_enabled"] is True
    assert config_b["atr_sizing_enabled"] is False
    assert config_b["confidence_sizing_enabled"] is False
    assert config_b["staleness_discount_enabled"] is False
    assert config_b["earnings_sizing_enabled"] is False
    assert config_b["sector_adj"] == {"overweight": 1.0, "market_weight": 1.0, "underweight": 1.0}
    assert config_b["conviction_decline_adj"] == 1.0
    assert config_b["upside_fail_adj"] == 1.0


def test_missing_alpha_diff_handled_gracefully():
    sim_fn = _make_sim_fn({
        True: {"total_trades": 100, "sharpe_ratio": 1.5, "total_return": 0.15, "total_alpha": None, "max_drawdown": 0.1},
        False: {"total_trades": 100, "sharpe_ratio": 1.2, "total_return": 0.12, "total_alpha": None, "max_drawdown": 0.12},
    })

    result = run_sizing_ab(sim_fn, _base_config())

    assert result["status"] == "ok"
    assert result["alpha_diff"] is None
    assert result["sharpe_diff"] == pytest.approx(0.30)


def test_zero_sharpe_no_difference_branch():
    """When both sharpe values are 0, sharpe_diff is None and detail falls to 'Unable to compare'."""
    sim_fn = _make_sim_fn({
        True: {"total_trades": 100, "sharpe_ratio": 0, "total_return": 0.05, "total_alpha": 0.01, "max_drawdown": 0.08},
        False: {"total_trades": 100, "sharpe_ratio": 0, "total_return": 0.05, "total_alpha": 0.01, "max_drawdown": 0.08},
    })

    result = run_sizing_ab(sim_fn, _base_config())

    assert result["status"] == "ok"
    assert result["assessment"] == "no_difference"
    assert result["detail"] == "Unable to compare"
