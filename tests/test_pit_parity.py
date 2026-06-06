"""Unit tests for PR 3 — the point-in-time contamination report
(``analysis/pit_parity.py``; ROADMAP L2371 / plan §D4).

Locks: the basket is Sortino/PSR/CVaR/maxDD + log-domain headline with
**no Sharpe**; log-domain cumulative return is summed not compounded;
deltas are pit−current; the 2-split CSCV PBO is honest (None without a
sweep pair, never a fabricated number); the report is observational and
flip-gated; run_pit_parity runs both passes with the flag flipped and
never raises on an S3 upload failure.
"""

from __future__ import annotations

import datetime as dt
import sys
import types

import numpy as np
import pandas as pd
import pytest

from analysis import pit_parity as pp


def _stats(sortino, psr, cvar, mdd, log_rets, total_alpha):
    return {
        "sortino_ratio": sortino, "psr": psr, "cvar_95": cvar,
        "max_drawdown": mdd, "total_alpha": total_alpha,
        "daily_log_returns": np.array(log_rets, dtype=float),
        "total_return": float(np.expm1(np.sum(log_rets))),
        "status": "ok",
    }


def test_log_cum_return_is_summed_not_compounded():
    s = _stats(1.0, 0.9, -0.02, -0.1, [0.01, 0.02, -0.005], 0.03)
    # Time-additive: sum of daily log returns (plan invariant 5).
    assert pp._log_cum_return(s) == pytest.approx(0.025)


def test_log_cum_return_falls_back_to_log1p_total_return():
    assert pp._log_cum_return({"total_return": 0.10}) == pytest.approx(
        np.log1p(0.10)
    )
    assert pp._log_cum_return({"total_return": None}) is None


def test_basket_has_no_sharpe():
    s = _stats(1.2, 0.95, -0.03, -0.15, [0.01], 0.04)
    s["sharpe_ratio"] = 2.5  # present in stats but must NOT enter the basket
    b = pp._basket(s)
    assert "sharpe_ratio" not in b
    assert set(b) == {"sortino_ratio", "psr", "cvar_95",
                      "max_drawdown", "log_cum_return", "total_alpha"}


def test_delta_is_pit_minus_current_and_none_safe():
    cur = pp._basket(_stats(1.0, 0.9, -0.04, -0.20, [0.0], 0.01))
    pit = pp._basket(_stats(0.7, 0.8, -0.05, -0.25, [0.0], -0.01))
    d = pp._delta(pit, cur)
    assert d["sortino_ratio"] == pytest.approx(-0.3)   # pit − current
    assert d["max_drawdown"] == pytest.approx(-0.05)
    # None on either side → None, never a crash.
    assert pp._delta({"sortino_ratio": None}, {"sortino_ratio": 1.0})[
        "sortino_ratio"] is None


def test_pbo_none_without_sweep_pair():
    assert pp._pbo_two_split(None, None) is None


def test_pbo_two_split_detects_overfit():
    # In-sample ranks configs c0>c1>c2; out-of-sample reverses → the best
    # IS config (c0) lands at OOS percentile 0.0 < 0.5 ⇒ overfit=True.
    cur = pd.DataFrame({"config_id": [0, 1, 2], "sortino_ratio": [3.0, 2.0, 1.0]})
    pit = pd.DataFrame({"config_id": [0, 1, 2], "sortino_ratio": [1.0, 2.0, 3.0]})
    r = pp._pbo_two_split(cur, pit)
    assert r["n_configs"] == 3
    assert r["overfit"] is True
    assert r["best_in_sample_config_oos_percentile"] == pytest.approx(0.0)
    assert r["spearman_rank_corr"] == pytest.approx(-1.0)


def test_pbo_two_split_stable_when_ranks_agree():
    cur = pd.DataFrame({"config_id": [0, 1, 2], "sortino_ratio": [3.0, 2.0, 1.0]})
    pit = pd.DataFrame({"config_id": [0, 1, 2], "sortino_ratio": [3.1, 2.2, 0.9]})
    r = pp._pbo_two_split(cur, pit)
    assert r["overfit"] is False
    assert r["best_in_sample_config_oos_percentile"] == pytest.approx(1.0)


def test_build_report_shape_and_materiality():
    cur = _stats(1.20, 0.96, -0.030, -0.12, [0.012, 0.004], 0.05)
    pit = _stats(0.85, 0.88, -0.041, -0.18, [0.006, 0.001], 0.02)
    rep = pp.build_contamination_report(
        cur, pit, run_date="2026-05-17",
        wf_meta={"n_folds": 40, "n_cold_start_excluded": 6},
    )
    assert rep["schema"] == pp.SCHEMA
    assert "Sharpe deliberately absent" in rep["anchor"]
    # ΔSortino = 0.85 − 1.20 = −0.35 → |Δ| ≥ 0.10 ⇒ material.
    assert rep["delta_pit_minus_current"]["sortino_ratio"] == pytest.approx(-0.35)
    assert rep["materiality"]["material"] is True
    assert rep["pbo"] is None  # no sweep pair in single-pass parity
    assert rep["observational"] is True
    assert "Brian-gated" in rep["flip_gate"]
    assert rep["run_quality"]["walk_forward"]["n_cold_start_excluded"] == 6
    assert rep["headline_log_alpha_delta"] == pytest.approx(
        (0.006 + 0.001) - (0.012 + 0.004)
    )


def test_run_pit_parity_runs_both_passes_and_survives_upload_failure(monkeypatch):
    seen: list[bool] = []

    # L4487: each pass runs in its own subprocess (backtest.py --pit-parity-pass),
    # so the in-process backtest mock is bypassed by the child. Mock at the
    # isolation seam — the test exercises run_pit_parity's orchestration.
    def fake_pass(safe_config, which, run_date):
        wf = (which == "walkforward")
        seen.append(wf)
        s = _stats(1.0 if not wf else 0.6,
                   0.9, -0.03, -0.15, [0.01, 0.0], 0.03)
        if wf:
            s["predictor_metadata"] = {"walk_forward": {"n_folds": 12,
                                                        "n_cold_start_excluded": 2}}
        return s

    monkeypatch.setattr(pp, "_run_predictor_pass_isolated", fake_pass)

    # S3 upload must be best-effort: a boto failure cannot raise.
    class _BoomS3:
        def put_object(self, **kw):
            raise RuntimeError("S3 down")

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda *a, **k: _BoomS3()
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    rep = pp.run_pit_parity({"signals_bucket": "b", "_run_date": "2026-05-17"})

    # Both passes ran, in order, with the flag flipped — and the original
    # config was deep-copied (not mutated).
    assert seen == [False, True]
    assert rep["delta_pit_minus_current"]["sortino_ratio"] == pytest.approx(-0.4)
    assert rep["run_quality"]["walk_forward"]["n_cold_start_excluded"] == 2
    assert "_s3_key" not in rep  # upload failed but run_pit_parity returned


def test_run_pit_parity_incomplete_pass_yields_status_report(monkeypatch):
    # L4487: mock at the subprocess-isolation seam (see both-passes test).
    monkeypatch.setattr(
        pp, "_run_predictor_pass_isolated",
        lambda safe_config, which, run_date: {"status": "insufficient_data"},
    )

    # Always-emit-artifact contract: the incomplete-status path must also
    # upload, not just return a dict. Prior to 2026-05-27 this path was
    # silent — same bug class as the cyclic-deepcopy incident.
    captured: list[dict] = []

    class _RecordingS3:
        def put_object(self, **kw):
            captured.append(kw)

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda *a, **k: _RecordingS3()
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    rep = pp.run_pit_parity({"signals_bucket": "b", "_run_date": "2026-05-17"})
    assert rep["status"] == "incomplete"
    assert rep["observational"] is True
    assert rep["_s3_key"] == "backtest/2026-05-17/pit_parity.json"
    assert len(captured) == 1
    body = captured[0]["Body"]
    assert b'"status": "incomplete"' in body if isinstance(body, bytes) else \
        '"status": "incomplete"' in body


def test_run_pit_parity_survives_cyclic_runtime_handle(monkeypatch):
    """Regression for 2026-05-17→2026-05-24 silent failure.

    The live ``config`` carries ``_phase_registry`` whose ``.s3_client``
    has botocore service-model backrefs that recurse past the Python
    stack limit under ``copy.deepcopy``. Saturday SF firings since
    #221 merged silently swallowed the RecursionError, leaving Brian's
    manual-flip gate unreachable for 11 days.

    Mirror the existing strip-pattern at ``backtest.py:862`` and pin
    behaviour against the failure shape that bit us.
    """
    # Build a self-referential cyclic object — same shape as
    # PhaseRegistry.s3_client's botocore service_model chain. Plain
    # deepcopy on this raises RecursionError.
    class _Cycle:
        pass
    cyclic = _Cycle()
    cyclic.back = cyclic  # type: ignore[attr-defined]

    seen: list[bool] = []

    def fake_pass(safe_config, which, run_date):
        # run_pit_parity must strip the cyclic runtime handle BEFORE handing
        # safe_config to the isolation seam (it is also what gets JSON-dumped
        # to the child — a cyclic/unstrippable handle would break that too).
        assert "_phase_registry" not in safe_config
        wf = (which == "walkforward")
        seen.append(wf)
        s = _stats(1.0 if not wf else 0.7,
                   0.9, -0.03, -0.15, [0.01, 0.0], 0.03)
        if wf:
            s["predictor_metadata"] = {"walk_forward": {"n_folds": 8}}
        return s

    monkeypatch.setattr(pp, "_run_predictor_pass_isolated", fake_pass)

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda *a, **k: type(
        "_S", (), {"put_object": lambda self, **kw: None}
    )()
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    config = {
        "signals_bucket": "b",
        "_run_date": "2026-05-24",
        "_phase_registry": cyclic,  # the failure-mode key
    }

    # Must not raise — and BOTH passes must run with walk_forward toggled.
    rep = pp.run_pit_parity(config)
    assert seen == [False, True]
    assert rep["delta_pit_minus_current"]["sortino_ratio"] == pytest.approx(-0.3)


def test_config_without_runtime_handles_explicit_allowlist():
    """The strip is an explicit-allowlist, NOT a prefix filter.

    Load-bearing ``_run_date`` (read at line 213 of run_pit_parity) must
    survive the strip; only the named runtime handles are dropped.
    """
    cfg = {
        "signals_bucket": "b",
        "_run_date": "2026-05-24",
        "_phase_registry": object(),
        "walk_forward": False,
    }
    safe = pp._config_without_runtime_handles(cfg)
    assert "_phase_registry" not in safe
    assert safe["_run_date"] == "2026-05-24"  # not stripped
    assert safe["signals_bucket"] == "b"
    assert safe["walk_forward"] is False


def test_write_failure_artifact_uploads_status_failed(monkeypatch):
    """The outer-exception path emits a ``status=failed`` artifact so
    Brian's manual-flip gate always has something to read. Pins the
    artifact shape that the operator's review process depends on.
    """
    captured: list[dict] = []

    class _RecordingS3:
        def put_object(self, **kw):
            captured.append(kw)

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda *a, **k: _RecordingS3()
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    config = {"signals_bucket": "b", "_run_date": "2026-05-24"}
    rep = pp.write_failure_artifact(config, RecursionError("maximum recursion depth exceeded"))

    assert rep["status"] == "failed"
    assert rep["error_class"] == "RecursionError"
    assert "recursion" in rep["error_msg"]
    assert rep["_s3_key"] == "backtest/2026-05-24/pit_parity.json"
    assert len(captured) == 1


def test_write_failure_artifact_swallows_upload_error(monkeypatch):
    """Failure-artifact write is itself observational — an S3 failure
    on the failure-write path must not raise. The Telegram alert in
    ``backtest.py::main`` is the redundant surface."""
    class _Boom:
        def put_object(self, **kw):
            raise RuntimeError("S3 down")
    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda *a, **k: _Boom()
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    rep = pp.write_failure_artifact(
        {"signals_bucket": "b", "_run_date": "2026-05-24"},
        ValueError("synthetic"),
    )
    assert rep["status"] == "failed"
    assert "_s3_key" not in rep  # upload failed but write_failure_artifact returned


def test_passes_run_via_subprocess_run_not_multiprocessing():
    """L4487: both passes go through _run_predictor_pass_isolated, which uses
    subprocess.run (a fresh `backtest.py --pit-parity-pass`, cwd=backtester) —
    NOT multiprocessing (whose spawn __main__ re-import collided on the `analysis`
    package, #285). Source assertion; the real path is validated by a scoped SF run."""
    import inspect
    import analysis.pit_parity as ppmod

    runner = inspect.getsource(ppmod._run_predictor_pass_isolated)
    assert "subprocess.run" in runner, "must use subprocess.run for true process isolation"
    # No multiprocessing USAGE (the docstring may mention it to explain why not).
    assert "import multiprocessing" not in runner and "ProcessPoolExecutor(" not in runner, (
        "must NOT use multiprocessing (spawn __main__ re-import collision, #285)"
    )
    assert "--pit-parity-pass" in runner and "cwd=" in runner, (
        "must invoke backtest.py --pit-parity-pass with cwd=backtester repo"
    )

    orch = inspect.getsource(ppmod.run_pit_parity)
    assert orch.count("_run_predictor_pass_isolated(") == 2, "both passes via the seam"
    assert '"lookahead"' in orch and '"walkforward"' in orch
    assert "run_predictor_backtest(" not in orch, "parent must not run a pass in-process"


def test_backtest_has_pit_parity_pass_child_submode_with_rss_guard():
    """L4487: the child sub-mode + the anti-degradation RSS-budget guard exist."""
    import re
    bt = (_SCRIPT.parent.parent / "backtest.py").read_text() if False else None  # noqa
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "backtest.py").read_text()
    assert 'if args.pit_parity_pass:' in src, "child sub-mode handler missing"
    assert "--pit-parity-pass" in src and "--stats-out" in src and "--config-json" in src
    # anti-degradation guard: per-pass peak RSS checked against a budget + alert
    assert "ru_maxrss" in src and "PIT_PARITY_PASS_RSS_BUDGET_MB" in src, (
        "per-pass RSS-budget guard (the 'these always degrade' fix) missing"
    )


def test_isolated_pass_surfaces_child_stderr_on_failure():
    """L4487b: a non-zero child exit must surface the child's stderr (the actual
    cause), not a bare 'exit 1'. The earlier design used check=True with inherited
    fds, so the SSM-relayed stream dropped the traceback -> opaque failures
    (no-silent-fails violation in our own code)."""
    import inspect
    import analysis.pit_parity as ppmod
    src = inspect.getsource(ppmod._run_predictor_pass_isolated)
    assert "stderr=subprocess.PIPE" in src, "child stderr must be captured"
    assert "returncode" in src and "raise RuntimeError" in src, (
        "non-zero child exit must raise with the captured stderr tail"
    )
    assert "proc.stderr" in src, "the raised error/log must include the child's stderr"
