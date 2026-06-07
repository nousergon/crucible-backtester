"""`analysis.param_sweep._run_combos` must degrade a single bad combo to an
error-row — never let it escape and kill the whole sweep.

L4525 (2026-06-06): recovery8's param-sweep raised an exception that escaped
`sweep()` → backtest.py set `sweep_df = None` → the L4523 export guard correctly
treated an ABSENT sweep as fatal. The escape was possible because
`_deepcopy_safe_config(base_config)` ran OUTSIDE the per-combo try, so a deepcopy
/ recursion failure on one combo's config killed the process instead of
degrading to an error-row. The docstring of `_deepcopy_safe_config` records a
prior `maximum recursion depth exceeded` from a boto3 client held in config —
exactly this failure class.

Fix: the deepcopy is now inside the per-combo try. A bad combo is a logged
warning + an error-row; the sweep completes and returns a (possibly all-error)
DataFrame, which the export guard treats as EMPTY/valid — never a fatal ABSENT.
Per the L4523 outcome taxonomy + [[feedback_no_silent_fails]].
"""

from __future__ import annotations

import pandas as pd

from analysis.param_sweep import _run_combos


class _Undeepcopyable:
    """An object that raises when deepcopied — stands in for the boto3-client /
    cyclic-ref configs that historically broke `_deepcopy_safe_config`."""

    def __deepcopy__(self, memo):  # noqa: D401 — test stub
        raise RuntimeError("cannot deepcopy this combo config object")


def test_deepcopy_failure_degrades_to_error_row_not_sweep_kill():
    # bad_obj is a NON-underscore key, so _deepcopy_safe_config will try to
    # deepcopy it and raise — the failure must be caught per-combo.
    base = {"good_key": 1, "bad_obj": _Undeepcopyable()}
    combos = [{"min_score": 50}, {"min_score": 60}]

    def sim_fn(cfg):  # should never be reached — deepcopy fails first
        return {"sortino_ratio": 1.0, "total_alpha": 0.1}

    df = _run_combos(combos, sim_fn, base)  # must NOT raise

    assert len(df) == len(combos), "every combo must yield a row (error-rows)"
    assert "error" in df.columns
    assert df["error"].notna().all(), (
        "a deepcopy failure must degrade to an error-row per combo, not escape "
        "the sweep (recovery8 symptom)"
    )


def test_sim_failure_degrades_to_error_row():
    base = {"good_key": 1}
    combos = [{"min_score": 50}]

    def sim_fn(cfg):
        raise ValueError("sim blew up")

    df = _run_combos(combos, sim_fn, base)

    assert len(df) == 1
    assert df.iloc[0]["error"] == "sim blew up"


def test_mixed_success_and_failure_completes_sweep():
    base = {"good_key": 1}
    combos = [{"min_score": 50}, {"min_score": 60}, {"min_score": 70}]
    calls = {"n": 0}

    def sim_fn(cfg):
        calls["n"] += 1
        if cfg["min_score"] == 60:
            raise ValueError("just this one fails")
        return {"sortino_ratio": 1.0, "total_alpha": 0.1}

    df = _run_combos(combos, sim_fn, base)

    assert len(df) == 3
    assert isinstance(df, pd.DataFrame)
    # two good rows have stats; exactly one carries an error
    assert "error" in df.columns
    assert df["error"].notna().sum() == 1
