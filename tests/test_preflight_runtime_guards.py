"""Guard: BacktesterPreflight must catch environment issues that would
otherwise surface deep in a 60-80 minute spot run.

Motivated by the 2026-04-21 Saturday SF dry-run that burned ~80 minutes
of c5.large compute before failing with
``No module named 'alpha_engine_lib.arcticdb'`` inside
``_run_simulation_loop``. These tests verify the three startup-class
preflight checks (lib version, imports, predictor weights) hard-fail
in ~1-2 seconds with a useful error, rather than letting the full
backtest start.
"""

from __future__ import annotations

import sys
import types

import pytest


# ── _check_lib_version ──────────────────────────────────────────────────────


def test_check_lib_version_passes_when_installed_meets_minimum(monkeypatch):
    from preflight import BacktesterPreflight, MIN_LIB_VERSION

    # Force the installed version to be exactly the minimum — must pass.
    import nousergon_lib
    monkeypatch.setattr(nousergon_lib, "__version__", MIN_LIB_VERSION, raising=False)
    preflight = BacktesterPreflight(bucket="test-bucket", mode="backtest")
    preflight._check_lib_version()  # must not raise


def test_check_lib_version_passes_when_installed_exceeds_minimum(monkeypatch):
    from preflight import BacktesterPreflight

    import nousergon_lib
    monkeypatch.setattr(nousergon_lib, "__version__", "99.99.99", raising=False)
    preflight = BacktesterPreflight(bucket="test-bucket", mode="backtest")
    preflight._check_lib_version()  # must not raise


def test_check_lib_version_fails_when_installed_below_minimum(monkeypatch):
    from preflight import BacktesterPreflight

    import nousergon_lib
    monkeypatch.setattr(nousergon_lib, "__version__", "0.0.1", raising=False)
    preflight = BacktesterPreflight(bucket="test-bucket", mode="backtest")

    with pytest.raises(RuntimeError) as exc:
        preflight._check_lib_version()
    msg = str(exc.value)
    assert "0.0.1" in msg
    assert "required" in msg
    assert "80 min" in msg or "80-min" in msg  # incident reference in error


def test_check_lib_version_fails_when_version_missing(monkeypatch):
    from preflight import BacktesterPreflight

    import nousergon_lib
    # Pretend __version__ isn't defined
    monkeypatch.delattr(nousergon_lib, "__version__", raising=False)
    preflight = BacktesterPreflight(bucket="test-bucket", mode="backtest")

    with pytest.raises(RuntimeError, match="no __version__"):
        preflight._check_lib_version()


# ── _check_imports ──────────────────────────────────────────────────────────


def test_check_imports_fails_with_named_module_on_import_error(monkeypatch):
    """When one of the critical imports raises ImportError, the preflight
    failure must name the specific module so the operator sees the fix
    (e.g. pip pin, requirements.txt)."""
    from preflight import BacktesterPreflight, _CRITICAL_IMPORTS_BACKTEST, _LOCAL_PREIMPORTS_BACKTEST
    import importlib

    broken_name = _CRITICAL_IMPORTS_BACKTEST[0]  # e.g. alpha_engine_lib.arcticdb

    def fake_import_module(name):
        # Local preimports must succeed (they're backtester-owned code
        # and always importable in-repo); fail only on the critical list.
        if name in _LOCAL_PREIMPORTS_BACKTEST:
            return types.ModuleType(name)
        raise ImportError(f"simulated: no module named {name!r}")

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    preflight = BacktesterPreflight(bucket="test-bucket", mode="backtest")
    with pytest.raises(RuntimeError) as exc:
        preflight._check_imports()
    msg = str(exc.value)
    assert broken_name in msg
    assert "requirements.txt" in msg
    assert "pip install" in msg


def test_check_imports_preloads_local_modules_before_sys_path_insert(monkeypatch, tmp_path):
    """Sibling-repo collision defense: backtester's own store.arctic_reader
    must be imported and cached in sys.modules BEFORE sibling predictor_path
    lands on sys.path[0]. Otherwise synthetic.predictor_backtest's
    ``from store.arctic_reader import load_universe_from_arctic`` resolves
    to predictor's same-named module (missing that function)."""
    from preflight import BacktesterPreflight, _LOCAL_PREIMPORTS_BACKTEST
    import sys
    import importlib

    ordered_calls: list[str] = []

    def tracking_import_module(name):
        ordered_calls.append(name)
        return types.ModuleType(name)  # stub every import (including the local preimports)

    monkeypatch.setattr(importlib, "import_module", tracking_import_module)

    pred_root = tmp_path / "pred"
    pred_root.mkdir()
    preflight = BacktesterPreflight(
        bucket="test-bucket",
        mode="backtest",
        predictor_paths=[str(pred_root)],
    )
    preflight._check_imports()

    # Every _LOCAL_PREIMPORTS_BACKTEST entry must appear BEFORE the first
    # sibling-repo module import. Look for the position of
    # "synthetic.predictor_backtest" (predictor-sibling) — it must come
    # after all local preimports.
    synthetic_pos = ordered_calls.index("synthetic.predictor_backtest")
    for local in _LOCAL_PREIMPORTS_BACKTEST:
        assert local in ordered_calls, f"{local} not preimported"
        assert ordered_calls.index(local) < synthetic_pos, (
            f"{local} was imported at position {ordered_calls.index(local)}, "
            f"after synthetic.predictor_backtest at {synthetic_pos} — sibling "
            "collision defense broken"
        )


def test_check_imports_inserts_executor_and_predictor_paths(monkeypatch, tmp_path):
    """_check_imports must prepend executor_paths + predictor_paths
    entries to sys.path before attempting the executor/predictor module
    imports — otherwise they fail with ModuleNotFoundError even when
    the repos are cloned locally. Mirrors what backtest._setup_simulation
    does later in the pipeline."""
    from preflight import BacktesterPreflight
    import sys
    import importlib

    # Build two tmp dirs that stand in for alpha-engine + alpha-engine-predictor.
    exec_root = tmp_path / "exec"
    pred_root = tmp_path / "pred"
    exec_root.mkdir()
    pred_root.mkdir()

    recorded: list[str] = []
    real_import_module = importlib.import_module

    def fake_import_module(name):
        recorded.append(name)
        return types.ModuleType(name)  # succeed silently for every name

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    preflight = BacktesterPreflight(
        bucket="test-bucket",
        mode="backtest",
        executor_paths=[str(exec_root)],
        predictor_paths=[str(pred_root)],
    )
    preflight._check_imports()

    # Both tmp roots should now be on sys.path so that executor/predictor
    # modules would resolve if they lived there.
    assert str(exec_root) in sys.path
    assert str(pred_root) in sys.path
    # Every critical module was import-attempted.
    assert "nousergon_lib.arcticdb" in recorded
    assert "executor.main" in recorded
    assert "model.gbm_scorer" in recorded


def test_check_imports_passes_when_all_modules_resolve(monkeypatch):
    """Happy path: when importlib.import_module returns cleanly for every
    listed module (local + critical), _check_imports must not raise.

    The critical-imports list includes executor/predictor modules that
    are only on sys.path in the spot deploy layout, not in the local
    backtester dev venv — so we monkeypatch import_module here rather
    than relying on the local env to actually have them."""
    from preflight import BacktesterPreflight
    import importlib

    monkeypatch.setattr(importlib, "import_module", lambda name: types.ModuleType(name))

    preflight = BacktesterPreflight(bucket="test-bucket", mode="backtest")
    preflight._check_imports()  # must not raise


def test_critical_imports_list_is_nonempty_and_stringy():
    """Cheap assertion that the constant is well-formed. Regression guard
    against an empty tuple sneaking in and silently disabling the check."""
    from preflight import _CRITICAL_IMPORTS_BACKTEST
    assert len(_CRITICAL_IMPORTS_BACKTEST) >= 5
    assert all(isinstance(n, str) and "." in n for n in _CRITICAL_IMPORTS_BACKTEST)
    # These two are the specific modules we added after the 80-min burn
    # — test guards against accidental removal.
    assert "nousergon_lib.arcticdb" in _CRITICAL_IMPORTS_BACKTEST
    assert "synthetic.predictor_backtest" in _CRITICAL_IMPORTS_BACKTEST


# ── _check_predictor_weights ────────────────────────────────────────────────


def test_check_predictor_weights_passes_when_head_succeeds(monkeypatch):
    """S3 HEAD returns 200 → both keys exist → preflight passes."""
    from preflight import BacktesterPreflight

    mock_calls: list[tuple] = []

    class _MockS3:
        def head_object(self, Bucket, Key):
            mock_calls.append((Bucket, Key))
            return {"ContentLength": 1}

    import boto3
    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: _MockS3())

    preflight = BacktesterPreflight(bucket="test-bucket", mode="backtest")
    preflight._check_predictor_weights()

    keys = sorted(k for _, k in mock_calls)
    assert keys == [
        "predictor/weights/meta/momentum_model.txt",
        "predictor/weights/meta/momentum_model.txt.meta.json",
    ]


def test_check_predictor_weights_fails_with_named_upstream_on_missing(monkeypatch):
    """When the HEAD fails, the error message must name the upstream
    owner (PredictorTraining) so the operator knows where to look."""
    from preflight import BacktesterPreflight

    class _MockS3:
        def head_object(self, Bucket, Key):
            raise Exception("NoSuchKey")

    import boto3
    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: _MockS3())

    preflight = BacktesterPreflight(bucket="test-bucket", mode="backtest")
    with pytest.raises(RuntimeError) as exc:
        preflight._check_predictor_weights()
    msg = str(exc.value)
    assert "momentum_model.txt" in msg
    assert "PredictorTraining" in msg


# ── _check_universe_freshness — REMOVED 2026-05-05 ──────────────────────────
# The per-ticker universe-freshness scan moved upstream to
# alpha-engine-data's preflight, which runs as the Saturday SF's
# DataPhase1 step before the backtester runs. SF ordering is the gate;
# re-checking here was redundant. Any future ASGN/MOH-class incidents
# will surface in alpha-engine-data's preflight or sf_preflight.py.
