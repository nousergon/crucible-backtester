"""Experiment-package precedence for pipeline_common.load_config (config#1042)."""

from pathlib import Path

import yaml

import pipeline_common


def _write(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data))


# _validate_config requires signals_bucket; include it so loads succeed.
def _cfg(source: str) -> dict:
    return {"source": source, "signals_bucket": "alpha-engine-research"}


def test_experiment_package_wins_over_legacy(tmp_path, monkeypatch):
    home = tmp_path / "home"
    cfg_repo = home / "alpha-engine-config"
    _write(cfg_repo / "backtester" / "config.yaml", _cfg("legacy"))
    _write(cfg_repo / "experiments" / "reference" / "backtester" / "config.yaml", _cfg("package"))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.delenv("ALPHA_ENGINE_EXPERIMENT_ID", raising=False)

    assert pipeline_common.load_config("nonexistent.yaml")["source"] == "package"


def test_experiment_id_selects_slot(tmp_path, monkeypatch):
    home = tmp_path / "home"
    cfg_repo = home / "alpha-engine-config"
    _write(cfg_repo / "experiments" / "reference" / "backtester" / "config.yaml", _cfg("reference"))
    _write(cfg_repo / "experiments" / "myexp" / "backtester" / "config.yaml", _cfg("myexp"))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.setenv("ALPHA_ENGINE_EXPERIMENT_ID", "myexp")

    assert pipeline_common.load_config("nonexistent.yaml")["source"] == "myexp"


def test_falls_back_to_legacy_when_no_package(tmp_path, monkeypatch):
    home = tmp_path / "home"
    cfg_repo = home / "alpha-engine-config"
    _write(cfg_repo / "backtester" / "config.yaml", _cfg("legacy"))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.delenv("ALPHA_ENGINE_EXPERIMENT_ID", raising=False)

    assert pipeline_common.load_config("nonexistent.yaml")["source"] == "legacy"
