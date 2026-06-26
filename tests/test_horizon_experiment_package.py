"""Experiment-package precedence for the default search paths of the
secondary config readers repointed in config#1042 (step 4):

  * pipeline_common._load_active_horizon_days — predictor/predictor.yaml
  * preflight.BacktesterPreflight._check_executor_config — executor/risk.yaml

Both mirror pipeline_common.load_config's precedence exactly:
experiments/$ALPHA_ENGINE_EXPERIMENT_ID/<subdir>/<file> first, then the
legacy top-level <subdir>/<file>, then the repo-local fallback. The package
copies were made byte-identical to legacy in config#1159, so these repoints
are behavior-preserving — the assertions below pin the *resolution order*,
not a behavior change.
"""

from pathlib import Path

import yaml

import pipeline_common


def _write(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data))


# ── pipeline_common._load_active_horizon_days ─────────────────────────────────


def test_horizon_experiment_package_wins_over_legacy(tmp_path, monkeypatch):
    home = tmp_path / "home"
    cfg_repo = home / "alpha-engine-config"
    _write(cfg_repo / "predictor" / "predictor.yaml", {"labeling": {"forward_days": 5}})
    _write(
        cfg_repo / "experiments" / "reference" / "predictor" / "predictor.yaml",
        {"labeling": {"forward_days": 21}},
    )
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.delenv("ALPHA_ENGINE_EXPERIMENT_ID", raising=False)

    # Point the repo-local fallback root at an empty dir so it can't shadow.
    assert pipeline_common._load_active_horizon_days() == 21


def test_horizon_experiment_id_selects_slot(tmp_path, monkeypatch):
    home = tmp_path / "home"
    cfg_repo = home / "alpha-engine-config"
    _write(
        cfg_repo / "experiments" / "reference" / "predictor" / "predictor.yaml",
        {"labeling": {"forward_days": 21}},
    )
    _write(
        cfg_repo / "experiments" / "myexp" / "predictor" / "predictor.yaml",
        {"labeling": {"forward_days": 63}},
    )
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.setenv("ALPHA_ENGINE_EXPERIMENT_ID", "myexp")

    assert pipeline_common._load_active_horizon_days() == 63


def test_horizon_falls_back_to_legacy_when_no_package(tmp_path, monkeypatch):
    home = tmp_path / "home"
    cfg_repo = home / "alpha-engine-config"
    _write(cfg_repo / "predictor" / "predictor.yaml", {"labeling": {"forward_days": 7}})
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.delenv("ALPHA_ENGINE_EXPERIMENT_ID", raising=False)

    assert pipeline_common._load_active_horizon_days() == 7


def test_horizon_default_when_nothing_resolves(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.delenv("ALPHA_ENGINE_EXPERIMENT_ID", raising=False)

    assert pipeline_common._load_active_horizon_days(default=99) == 99
