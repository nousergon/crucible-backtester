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
    # Uses search_paths (not a Path.home patch): resolve_experiment_config's
    # candidate list also includes a repo-root-relative sibling-clone root
    # (repo_root.parent/alpha-engine-config, config#1042's "repo-parent
    # clone" case) that a Path.home patch can't reach. On a dev machine that
    # actually clones alpha-engine-config as a sibling of this repo (the
    # standard ~/Development layout), that candidate resolves to the REAL
    # config repo's experiments/reference/predictor/predictor.yaml and leaks
    # its real forward_days into this test BEFORE the tmp-path legacy
    # candidate is ever reached — the two tests above happen to pass only
    # because their expected value already matches the experiment-package
    # candidate that wins first. search_paths is exposed by
    # _load_active_horizon_days precisely so tests don't need to stub
    # pathlib/repo-layout internals at all.
    legacy_path = tmp_path / "alpha-engine-config" / "predictor" / "predictor.yaml"
    _write(legacy_path, {"labeling": {"forward_days": 7}})
    monkeypatch.delenv("ALPHA_ENGINE_EXPERIMENT_ID", raising=False)

    assert pipeline_common._load_active_horizon_days(search_paths=[legacy_path]) == 7


def test_horizon_default_when_nothing_resolves(tmp_path, monkeypatch):
    # See test_horizon_falls_back_to_legacy_when_no_package: search_paths
    # sidesteps the repo-root-relative candidate a Path.home patch can't reach
    # on a machine with a real sibling alpha-engine-config clone.
    monkeypatch.delenv("ALPHA_ENGINE_EXPERIMENT_ID", raising=False)

    assert pipeline_common._load_active_horizon_days(default=99, search_paths=[]) == 99


# ── pipeline_common._load_label_barrier_config (config#723) ───────────────────


def test_label_barrier_config_reads_triple_barrier_block(tmp_path):
    p = tmp_path / "predictor.yaml"
    _write(p, {"triple_barrier": {"forward_window": 21, "vol_multiplier": 2.0}})
    tb = pipeline_common._load_label_barrier_config(search_paths=[p])
    assert tb == {"forward_window": 21, "vol_multiplier": 2.0}


def test_label_barrier_config_first_existing_path_wins(tmp_path):
    pkg = tmp_path / "pkg" / "predictor.yaml"
    legacy = tmp_path / "legacy" / "predictor.yaml"
    _write(pkg, {"triple_barrier": {"forward_window": 10}})
    _write(legacy, {"triple_barrier": {"forward_window": 21}})
    tb = pipeline_common._load_label_barrier_config(search_paths=[pkg, legacy])
    assert tb["forward_window"] == 10


def test_label_barrier_config_none_when_block_absent(tmp_path):
    p = tmp_path / "predictor.yaml"
    _write(p, {"labeling": {"forward_days": 21}})  # no triple_barrier block
    assert pipeline_common._load_label_barrier_config(search_paths=[p]) is None


def test_label_barrier_config_none_when_no_path_exists(tmp_path):
    missing = tmp_path / "nope" / "predictor.yaml"
    assert pipeline_common._load_label_barrier_config(search_paths=[missing]) is None
