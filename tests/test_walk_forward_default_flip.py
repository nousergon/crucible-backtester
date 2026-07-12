"""Pin the 2026-07-08 default-on flip for point-in-time walk-forward
backtesting (config#833 — ROADMAP L2371 / Backtester Phase 2).

Before 2026-07-08: `walk_forward` defaulted to False; explicit opt-in via
`--walk-forward` was required to engage the PIT-honest fold-scored replay.
The default stayed OFF pending Brian's review of a `pit_parity.json` report
(plan pit-discipline-260515.md §5).

After 2026-07-08 (config#833, console Decision Queue "Option A" — approved
2026-07-07 once the feature-engineering PIT audit (nousergon-data#638) and
the momentum-weight dependency unblock (config#1518 / crucible-backtester
#432) were both merged): default is True; explicit opt-out via
`--no-walk-forward` falls back to the legacy single-pass look-ahead path
(momentum_model.txt). Both flags are mutually exclusive (argparse-enforced)
and the legacy path is one CLI flag away for emergency rollback / A-B
comparison.

These tests pin the argparse + config wiring for the flip, mirroring
test_vectorized_default_flip.py's structure for the analogous 2026-04-28
vectorized-sweep flip.
"""
from __future__ import annotations

import argparse
import sys
from unittest.mock import patch

import pytest

import backtest


# ── Argparse surface ────────────────────────────────────────────────────────


def test_default_run_has_neither_flag_set():
    """Bare invocation: neither --walk-forward nor --no-walk-forward is
    passed. args.walk_forward is None; the default-on logic kicks in via
    setdefault in _init_pipeline."""
    with patch.object(sys, "argv", ["backtest.py"]):
        args = backtest._parse_args()
    assert args.walk_forward is None


def test_explicit_walk_forward_flag_accepted():
    with patch.object(sys, "argv", ["backtest.py", "--walk-forward"]):
        args = backtest._parse_args()
    assert args.walk_forward is True


def test_explicit_no_walk_forward_flag_accepted():
    with patch.object(sys, "argv", ["backtest.py", "--no-walk-forward"]):
        args = backtest._parse_args()
    assert args.walk_forward is False


def test_both_flags_mutually_exclusive():
    """Both flags together is a usage error — argparse mutex group enforces."""
    with patch.object(
        sys, "argv",
        ["backtest.py", "--walk-forward", "--no-walk-forward"],
    ):
        with pytest.raises(SystemExit):
            backtest._parse_args()


# ── Config wiring (the actual default-on behavior) ──────────────────────────


def _blank_args(**overrides) -> argparse.Namespace:
    """Args with only the field _init_pipeline's flag-routing block touches."""
    base = dict(walk_forward=None)
    base.update(overrides)
    return argparse.Namespace(**base)


def _apply_flag_logic(args, config: dict) -> None:
    """Mirror the logic in backtest.py's `_init_pipeline` flag-routing
    block (the walk_forward stanza just below the use_vectorized_sweep
    one). If that block moves or changes, this test function moves with it.
    """
    if args.walk_forward is not None:
        config["walk_forward"] = args.walk_forward
    else:
        config.setdefault("walk_forward", True)


class TestDefaultOnFlip:
    def test_neither_flag_yields_default_on(self):
        """Bare run → walk-forward engaged. This is the 2026-07-08 flip
        (config#833)."""
        config: dict = {}
        _apply_flag_logic(_blank_args(), config)
        assert config["walk_forward"] is True

    def test_explicit_walk_forward_redundant_under_default_on(self):
        config: dict = {}
        _apply_flag_logic(_blank_args(walk_forward=True), config)
        assert config["walk_forward"] is True

    def test_explicit_no_walk_forward_opts_out(self):
        """`--no-walk-forward` flips back to the legacy single-pass path.
        Emergency rollback / A-B comparison semantics."""
        config: dict = {}
        _apply_flag_logic(_blank_args(walk_forward=False), config)
        assert config["walk_forward"] is False

    def test_config_yaml_can_opt_out_without_cli_flag(self):
        """Operator can also opt out by setting `walk_forward: false` in
        config.yaml — setdefault preserves the explicit config value."""
        config = {"walk_forward": False}
        _apply_flag_logic(_blank_args(), config)
        assert config["walk_forward"] is False

    def test_config_yaml_explicit_true_unchanged(self):
        """`walk_forward: true` in config preserved (no-op since default is
        also True, but the contract should be explicit)."""
        config = {"walk_forward": True}
        _apply_flag_logic(_blank_args(), config)
        assert config["walk_forward"] is True

    def test_cli_no_walk_forward_overrides_config_yaml_true(self):
        """If config.yaml says walk_forward: true but CLI says
        --no-walk-forward, CLI wins."""
        config = {"walk_forward": True}
        _apply_flag_logic(_blank_args(walk_forward=False), config)
        assert config["walk_forward"] is False

    def test_cli_walk_forward_overrides_config_yaml_false(self):
        """If config.yaml says walk_forward: false but CLI says
        --walk-forward, CLI wins."""
        config = {"walk_forward": False}
        _apply_flag_logic(_blank_args(walk_forward=True), config)
        assert config["walk_forward"] is True
