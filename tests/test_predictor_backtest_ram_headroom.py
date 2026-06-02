"""Tests for the pre-pipeline RAM-headroom guard (L4485).

The full 10y × ~900-ticker predictor backtest peaks at ~2.8 GB RSS and
OOM-killed predictor_pipeline on a 4 GB c5.large on 2026-06-01 — surfacing
as an opaque SIGKILL ~60 min in with no stdout / exit code. The guard reads
MemAvailable up front and converts that into a fast, legible startup error.

These tests pin the guard's contract: it raises below the floor, passes
above it, and degrades to a no-op (never raises) when /proc/meminfo is
unreadable (non-Linux / local dev), so it can't false-fail off-spot.
"""

from __future__ import annotations

from io import StringIO
from unittest import mock

import pytest

from synthetic import predictor_backtest as pb


# A realistic /proc/meminfo excerpt — MemAvailable is the load-bearing line.
_MEMINFO_8GB = (
    "MemTotal:        8146384 kB\n"
    "MemFree:         5012345 kB\n"
    "MemAvailable:    7340032 kB\n"  # ~7.0 GB
    "Buffers:          123456 kB\n"
)
_MEMINFO_4GB = (
    "MemTotal:        3996384 kB\n"
    "MemFree:         1012345 kB\n"
    "MemAvailable:    3670016 kB\n"  # ~3.5 GB
)


def test_available_ram_gb_parses_memavailable():
    with mock.patch("builtins.open", return_value=StringIO(_MEMINFO_8GB)):
        gb = pb._available_ram_gb()
    assert gb is not None
    assert 6.9 < gb < 7.1, f"expected ~7.0 GB, got {gb}"


def test_available_ram_gb_returns_none_when_proc_absent():
    with mock.patch("builtins.open", side_effect=FileNotFoundError):
        assert pb._available_ram_gb() is None


def test_assert_ram_headroom_raises_below_floor():
    with mock.patch.object(pb, "_available_ram_gb", return_value=3.5):
        with pytest.raises(RuntimeError, match="RAM headroom check FAILED"):
            pb._assert_ram_headroom(6.0)


def test_assert_ram_headroom_passes_above_floor():
    with mock.patch.object(pb, "_available_ram_gb", return_value=7.0):
        # No raise == pass.
        pb._assert_ram_headroom(6.0)


def test_assert_ram_headroom_skips_when_unreadable():
    """Off-Linux / local dev: MemAvailable unreadable → no-op, never raises.

    The guard must not false-fail where /proc is absent (Darwin tests, CI
    containers) — the production target is the Linux spot instance.
    """
    with mock.patch.object(pb, "_available_ram_gb", return_value=None):
        pb._assert_ram_headroom(6.0)  # must not raise


def test_default_min_ram_floor_is_8gb_class():
    """Floor sits between a 4 GB instance (~3.5 GB avail) and an 8 GB one
    (~7 GB avail) so it rejects the OOM-prone 4 GB c5.large and admits the
    ≥8 GB instances the mode-aware floor in spot_backtest.sh selects."""
    assert 3.5 < pb._DEFAULT_MIN_RAM_GB < 7.0
