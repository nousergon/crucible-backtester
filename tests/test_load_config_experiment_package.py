"""Experiment-package precedence for pipeline_common.load_config (config#1042)."""

from pathlib import Path

import yaml

import nousergon_lib.config as nlc
import pipeline_common


def _write(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data))


# _validate_config requires signals_bucket; include it so loads succeed.
def _cfg(source: str) -> dict:
    return {"source": source, "signals_bucket": "alpha-engine-research"}


def _pin_config_root(monkeypatch, home: Path) -> None:
    """Pin load_config's experiment-config search to ``home`` only — hermetically.

    The lib resolver (``nousergon_lib.config._config_roots``) searches BOTH
    ``Path.home()/alpha-engine-config`` AND ``<repo_root>/../alpha-engine-config``,
    and ``load_config`` passes ``repo_root=<this repo>`` — so the second root is the
    repo-parent *sibling* clone. Monkeypatching ``Path.home`` alone does NOT
    neutralize that sibling root, so on any layout where the backtester sits beside
    a real ``alpha-engine-config`` checkout (the developer ``~/Development`` tree,
    and the spot instance, which clones config beside the repo) these tests leaked
    into the real config file and the legacy-fallback case failed with a KeyError —
    invisibly to CI, whose runner has no sibling checkout. Pinning ``_config_roots``
    to the tmp root makes resolution deterministic regardless of on-disk layout.
    """
    monkeypatch.setattr(nlc, "_config_roots", lambda *a, **k: [home / "alpha-engine-config"])


def test_experiment_package_wins_over_legacy(tmp_path, monkeypatch):
    home = tmp_path / "home"
    cfg_repo = home / "alpha-engine-config"
    _write(cfg_repo / "backtester" / "config.yaml", _cfg("legacy"))
    _write(cfg_repo / "experiments" / "reference" / "backtester" / "config.yaml", _cfg("package"))
    _pin_config_root(monkeypatch, home)
    monkeypatch.delenv("ALPHA_ENGINE_EXPERIMENT_ID", raising=False)

    assert pipeline_common.load_config("nonexistent.yaml")["source"] == "package"


def test_experiment_id_selects_slot(tmp_path, monkeypatch):
    home = tmp_path / "home"
    cfg_repo = home / "alpha-engine-config"
    _write(cfg_repo / "experiments" / "reference" / "backtester" / "config.yaml", _cfg("reference"))
    _write(cfg_repo / "experiments" / "myexp" / "backtester" / "config.yaml", _cfg("myexp"))
    _pin_config_root(monkeypatch, home)
    monkeypatch.setenv("ALPHA_ENGINE_EXPERIMENT_ID", "myexp")

    assert pipeline_common.load_config("nonexistent.yaml")["source"] == "myexp"


def test_falls_back_to_legacy_when_no_package(tmp_path, monkeypatch):
    home = tmp_path / "home"
    cfg_repo = home / "alpha-engine-config"
    _write(cfg_repo / "backtester" / "config.yaml", _cfg("legacy"))
    _pin_config_root(monkeypatch, home)
    monkeypatch.delenv("ALPHA_ENGINE_EXPERIMENT_ID", raising=False)

    assert pipeline_common.load_config("nonexistent.yaml")["source"] == "legacy"
