"""Pin the L4472 phase-split executor-apply semantics for _run_predictor_pipeline.

Background (ROADMAP L4472): the monolithic ``backtest.py --mode=all`` run was
split into independent Step Function states — ``Backtester`` (--mode=param-sweep,
the simulation pipeline) and ``PredictorBacktest`` (--mode=predictor-backtest).
Both the simulation pipeline and the predictor pipeline can call
``executor_optimizer.apply()``, which hardcodes
``optimizer_name="executor_optimizer"`` and therefore writes the SAME S3
recommendation artifact key
(``config/executor_params/recommendations/{date}/from_executor_optimizer.json``).

If the predictor pipeline applied executor params when running in its own SF
state, it would CLOBBER the sim-based artifact written by the param-sweep state
that ran first — predictor-based params would silently win on the live
config the assembler promotes. That is the L4472 fork.

The fix gates the predictor-side executor-apply on ``args.mode == "all"``:
- In ``--mode=all`` (monolithic / in-process), behavior is byte-for-byte
  unchanged: predictor applies executor params only as the in-process fallback
  when the sim sweep didn't produce an "ok" recommendation.
- In ``--mode=predictor-backtest`` (the split SF state), predictor NEVER
  applies executor params — the simulation pipeline (param-sweep state) is the
  sole writer of the executor recommendation. This is semantically equivalent
  to all-mode because the PredictorBacktest SF state only runs when the
  Backtester state SUCCEEDED (sim produced an "ok" rec — exactly the case
  all-mode suppresses predictor's apply).

These tests pin that gate so a future refactor can't silently reintroduce the
cross-state artifact collision.
"""
from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pandas as pd

import backtest


def _args(mode: str) -> argparse.Namespace:
    return argparse.Namespace(mode=mode, freeze=False)


def _nonempty_sweep_df() -> pd.DataFrame:
    return pd.DataFrame({"combo": [1], "sharpe": [1.23]})


def _run(mode: str, executor_rec):
    """Drive _run_predictor_pipeline with the predictor sweep mocked to a
    non-empty frame and the executor optimizer fully mocked. Returns the
    apply mock so callers can assert call counts."""
    cfg = {"signals_bucket": "alpha-engine-research"}
    with patch.object(
        backtest, "run_predictor_param_sweep",
        return_value=({"status": "ok"}, _nonempty_sweep_df()),
    ), patch.object(backtest, "executor_optimizer") as mock_opt:
        mock_opt.recommend.return_value = {"status": "ok"}
        mock_opt.apply.return_value = {"applied": True}
        backtest._run_predictor_pipeline(
            _args(mode), cfg, executor_rec, current_executor_params=None, fd=None,
        )
        return mock_opt


def test_split_predictor_mode_does_not_apply_executor_params():
    """--mode=predictor-backtest (split SF state): predictor must NOT apply
    executor params even when the sim recommendation is absent (executor_rec
    is None) and the predictor sweep is non-empty. Prevents the L4472
    cross-state artifact collision."""
    mock_opt = _run("predictor-backtest", executor_rec=None)
    mock_opt.apply.assert_not_called()
    mock_opt.recommend.assert_not_called()


def test_all_mode_applies_executor_params_as_fallback_when_sim_rec_absent():
    """--mode=all (monolithic): unchanged in-process fallback — predictor
    applies executor params when the sim sweep produced no recommendation."""
    mock_opt = _run("all", executor_rec=None)
    mock_opt.recommend.assert_called_once()
    mock_opt.apply.assert_called_once()


def test_all_mode_suppresses_predictor_apply_when_sim_rec_ok():
    """--mode=all: when the sim sweep already produced an 'ok' recommendation,
    predictor's executor-apply stays suppressed (pre-existing semantics)."""
    mock_opt = _run("all", executor_rec={"status": "ok"})
    mock_opt.apply.assert_not_called()
    mock_opt.recommend.assert_not_called()
