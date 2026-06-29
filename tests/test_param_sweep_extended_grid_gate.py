"""Pins the config#947 data gate on the extended param-sweep grid.

config#947 ("Extended parameter grid — 16 params (from 6) when 6+ months of
data support it") asked the backtester to widen the sweep to the low-frequency
params only once enough history exists to holdout-validate them. The EXTENDED_GRID
capability already shipped; what was missing was the *gate* — previously the grid
only widened via an undocumented manual ``config["param_sweep"]`` override.

``param_sweep.select_grid`` makes the "6+ months of data" condition a real code
gate: auto-select EXTENDED_GRID only when the signal-date window spans
>= EXTENDED_GRID_MIN_DAYS (~6mo), else the conservative DEFAULT_GRID, while still
honoring an explicit config override.
"""

from __future__ import annotations

from datetime import date, timedelta

from analysis import param_sweep


def _window(n_days: int) -> list[str]:
    """A 2-point sorted YYYY-MM-DD window spanning exactly n_days."""
    start = date(2026, 1, 1)
    end = start + timedelta(days=n_days)
    return [start.isoformat(), end.isoformat()]


def test_short_window_uses_default_grid():
    """A ~3-month window (< 6mo) must use the conservative core-6 grid."""
    grid = param_sweep.select_grid(_window(90), config={})
    assert grid == param_sweep.DEFAULT_GRID


def test_six_month_window_uses_extended_grid():
    """A window spanning >= 6 months unlocks the extended grid."""
    grid = param_sweep.select_grid(_window(param_sweep.EXTENDED_GRID_MIN_DAYS), config={})
    assert grid == param_sweep.EXTENDED_GRID
    assert len(grid) > len(param_sweep.DEFAULT_GRID)


def test_gate_boundary_is_inclusive():
    """Exactly EXTENDED_GRID_MIN_DAYS qualifies; one day short does not."""
    assert param_sweep.select_grid(
        _window(param_sweep.EXTENDED_GRID_MIN_DAYS), config={}
    ) == param_sweep.EXTENDED_GRID
    assert param_sweep.select_grid(
        _window(param_sweep.EXTENDED_GRID_MIN_DAYS - 1), config={}
    ) == param_sweep.DEFAULT_GRID


def test_explicit_config_override_always_wins():
    """An explicit config['param_sweep'] grid is honored verbatim, regardless
    of the data window (operators can pin any grid for offline studies)."""
    custom = {"min_score": [50, 60]}
    # Short window would otherwise pick DEFAULT_GRID — override must win.
    assert param_sweep.select_grid(_window(10), config={"param_sweep": custom}) is custom
    # Long window would otherwise pick EXTENDED_GRID — override still wins.
    assert param_sweep.select_grid(_window(400), config={"param_sweep": custom}) is custom


def test_force_extended_grid_respects_gate():
    """param_sweep_settings.force_extended_grid opts into the extended grid
    without hand-copying the dict, but STILL respects the data gate."""
    long_cfg = {"param_sweep_settings": {"force_extended_grid": True}}
    assert param_sweep.select_grid(_window(400), config=long_cfg) == param_sweep.EXTENDED_GRID

    short_cfg = {"param_sweep_settings": {"force_extended_grid": True}}
    # < 6 months: force is overruled by the gate to avoid in-sample over-fit.
    assert param_sweep.select_grid(_window(30), config=short_cfg) == param_sweep.DEFAULT_GRID


def test_insufficient_dates_falls_back_to_default():
    """0 or 1 signal dates → span 0 → conservative DEFAULT_GRID, no crash."""
    assert param_sweep.select_grid(None, config={}) == param_sweep.DEFAULT_GRID
    assert param_sweep.select_grid([], config={}) == param_sweep.DEFAULT_GRID
    assert param_sweep.select_grid(["2026-01-01"], config={}) == param_sweep.DEFAULT_GRID


def test_malformed_dates_are_skipped():
    """Non-ISO date strings are ignored when computing the window span."""
    dates = ["not-a-date", "2026-01-01", "2026-09-01", "garbage"]
    # 2026-01-01 .. 2026-09-01 spans > 6 months → EXTENDED.
    assert param_sweep.select_grid(dates, config={}) == param_sweep.EXTENDED_GRID
