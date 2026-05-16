"""Regression: the standalone backtest stage MUST pull research.db.

Root cause pinned here (2026-05-16): commit c852393 ("Split evaluation
from backtester into independent entry point", 2026-04-09) removed the
research.db pull from backtest.py and relocated it to evaluate.py
(init_research_db, evaluate.py:123 — the ONLY caller after the split).
The later Saturday-SF evaluator-stage split (evaluator-split-260507 /
PR #250-era) runs the `backtest` stage as a standalone SF state with
`--skip-stages=evaluator`, so the dedicated Evaluator state never runs
inside the Backtester state. Before the fix, backtest.py's
``_init_pipeline`` only set ``research_db`` when ``--db`` was passed —
on the SF/spot path (``args.db is None``) the pull was *never attempted*,
``_db_pull_status`` stayed unset, and the reporter rendered a bare
``- Research DB: None``.

These tests pin the contract: under the SF arg set (``args.db=None``,
the run that the Backtester state issues with ``--skip-stages=evaluator``)
``_init_pipeline`` invokes ``init_research_db`` and the research.db pull
IS attempted, and ``--db`` still short-circuits the S3 pull.
"""
import argparse

import pytest

import backtest
import pipeline_common


def _sf_args(db=None):
    """The arg set the Saturday-SF Backtester state effectively passes:
    no --db override (research.db comes from S3), evaluator stage skipped
    at the spot-script level (not a backtest.py arg)."""
    return argparse.Namespace(db=db)


class TestInitPipelinePullsResearchDB:

    def test_init_pipeline_invokes_init_research_db_on_sf_path(self, monkeypatch):
        """SF/spot path (args.db is None): _init_pipeline must call
        init_research_db so the research.db pull is attempted — pins the
        c852393 regression that #250-era SF split exposed."""
        called = {}

        def _fake_init_research_db(db_arg, config):
            called["db_arg"] = db_arg
            called["config_id"] = id(config)
            config["research_db"] = "/tmp/fake.db"
            config["_db_pull_status"] = "ok"

        monkeypatch.setattr(backtest, "init_research_db", _fake_init_research_db)
        monkeypatch.setattr(backtest.executor_optimizer, "init_config", lambda c: None)

        config = {}
        backtest._init_pipeline(_sf_args(db=None), config)

        assert "db_arg" in called, "init_research_db was NOT invoked on the SF path"
        assert called["db_arg"] is None
        assert called["config_id"] == id(config)
        assert config["_db_pull_status"] == "ok"
        assert config["research_db"] == "/tmp/fake.db"

    def test_research_db_pull_attempted_when_no_db_override(self, monkeypatch):
        """End-to-end through real init_research_db: pull_research_db IS
        called when args.db is None — pins the exact 2026-05-16 symptom
        ('the pull was never attempted')."""
        pull_calls = []

        def _fake_pull(bucket, local_path, s3_key="research.db"):
            pull_calls.append((bucket, s3_key))
            return True  # pretend the 188MB research.db landed

        monkeypatch.setattr(pipeline_common, "pull_research_db", _fake_pull)
        monkeypatch.setattr(backtest.executor_optimizer, "init_config", lambda c: None)

        config = {"signals_bucket": "alpha-engine-research"}
        backtest._init_pipeline(_sf_args(db=None), config)

        assert len(pull_calls) == 1, "research.db pull was never attempted"
        assert pull_calls[0] == ("alpha-engine-research", "research.db")
        assert config["_db_pull_status"] == "ok"
        assert config["research_db"] is not None

    def test_failed_pull_degrades_gracefully_not_crash(self, monkeypatch):
        """A pull failure must degrade (research_db=None,
        _db_pull_status='failed') — NOT crash the predictor-only /
        synthetic modes that don't consume research.db. The failure
        surfaces loudly via the reporter's MISSING line, per convention."""
        monkeypatch.setattr(
            pipeline_common, "pull_research_db",
            lambda *a, **k: False,
        )
        monkeypatch.setattr(backtest.executor_optimizer, "init_config", lambda c: None)

        config = {"signals_bucket": "alpha-engine-research"}
        backtest._init_pipeline(_sf_args(db=None), config)  # must not raise

        assert config["research_db"] is None
        assert config["_db_pull_status"] == "failed"

    def test_db_override_short_circuits_s3_pull(self, monkeypatch):
        """--db <path> still bypasses the S3 pull (local-iteration path)."""
        pull_calls = []
        monkeypatch.setattr(
            pipeline_common, "pull_research_db",
            lambda *a, **k: pull_calls.append(a) or True,
        )
        monkeypatch.setattr(backtest.executor_optimizer, "init_config", lambda c: None)

        config = {}
        backtest._init_pipeline(_sf_args(db="/local/research.db"), config)

        assert pull_calls == [], "S3 pull should be skipped when --db is given"
        assert config["research_db"] == "/local/research.db"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
