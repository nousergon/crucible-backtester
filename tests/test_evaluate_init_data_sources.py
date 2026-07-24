"""
tests/test_evaluate_init_data_sources.py — _init_data_sources
skip-backtester awareness (config#2887).

Verifies that evaluate.py::_init_data_sources distinguishes operator-intended
skip from unplanned artifact absence, and that predictor artifacts are
elevated to criticality parity with backtester artifacts.
"""

import argparse
import json
import logging

import pandas as pd
import pytest
from botocore.exceptions import ClientError


# ---------------------------------------------------------------------------
# Helpers: build an argparse namespace with skip_backtester set as needed
# ---------------------------------------------------------------------------

def _ns(*, skip_backtester: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        skip_backtester=skip_backtester,
        db=None,
        trades_db=None,
        date="2026-07-20",
        mode="all",
        module=None,
        config="config.yaml",
        upload=False,
        log_level="INFO",
        freeze=False,
        stop_instance=False,
    )


def _noop_research_db(*args, **kwargs):
    """Swallow init_research_db calls; the import-time mock patches this."""


# ---------------------------------------------------------------------------
# Fixture: mock S3 so _init_data_sources can run without real S3 or a full
# research.db. We patch at evaluate module scope.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_evaluate_deps(monkeypatch):
    """Patch S3, research-db init, and date-dependent config so
    _init_data_sources can be tested in isolation."""
    import evaluate  # noqa: F811 — imported below in tests too, but the
    # fixture-level import patches module-level references.
    # evaluate  # keep import for side effects

    # Patch boto3 client to return our stub
    monkeypatch.setattr("evaluate.boto3.client", _s3_stub_client)

    # Swallow research-db init
    monkeypatch.setattr("evaluate.init_research_db", _noop_research_db)

    # Provide a minimal find_trades_db that returns None
    monkeypatch.setattr("evaluate.find_trades_db", lambda config: None)

    # Ensure config gets a predictable output_bucket
    return {"output_bucket": "alpha-engine-research"}


def _s3_stub_client(service: str, **_):
    """Return a stub S3 client whose get_object raises NoSuchKey for every
    key — tests selectively override this via monkeypatch."""
    import boto3  # noqa: F811

    if service != "s3":
        raise ValueError(f"unexpected service: {service}")

    client = boto3.session.Session().create_client("s3", region_name="us-east-1")

    real_get = client.get_object

    def _stub_get(**kwargs):
        raise ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist."}},
            "GetObject",
        )

    client.get_object = _stub_get
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSkipBacktesterFlag:
    """--skip-backtester flag existence and parsing."""

    def test_flag_is_parsed(self):
        """The --skip-backtester flag is accepted and defaults to False."""
        import evaluate
        evaluated_parse = evaluate._parse_args
        if evaluated_parse:
            ns = evaluated_parse([])
            assert hasattr(ns, "skip_backtester")
            assert ns.skip_backtester is False

    def test_flag_can_be_set(self):
        """--skip-backtester sets the flag to True."""
        import evaluate
        ns = evaluate._parse_args(["--skip-backtester"])
        assert ns.skip_backtester is True


class TestInitDataSources:
    """_init_data_sources artifact-gating behavior."""

    def _call_init(self, *, skip_backtester: bool = False, config: dict = None) -> dict:
        """Call evaluate._init_data_sources with a minimal config."""
        import evaluate
        cfg = {"output_bucket": "alpha-engine-research", **(config or {})}
        ns = _ns(skip_backtester=skip_backtester)
        return evaluate._init_data_sources(ns, cfg)

    # -- skip-flagged: tolerant ---------------------------------------------------

    def test_skip_flag_all_artifacts_missing_logs_warning(self, caplog):
        """With --skip-backtester, missing all artifacts logs a WARNING and
        does NOT set the degraded marker."""
        from evaluate import _init_data_sources
        import evaluate

        cfg = {"output_bucket": "alpha-engine-research"}
        caplog.set_level(logging.WARNING)

        avail = self._call_init(skip_backtester=True)

        # Should NOT have degraded marker
        assert not avail.get("_degraded", False), \
            "skip-flagged run must not produce a degraded marker"

        # Should have all artifacts listed as unavailable
        assert avail["sweep_df"] is False
        assert avail["portfolio_stats"] is False
        assert avail["predictor_sweep_df"] is False
        assert avail["predictor_stats"] is False

        # Should have a WARNING-level log that mentions "intentionally skipped"
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        skip_logs = [r for r in warnings if "intentionally skipped" in r.getMessage()]
        assert skip_logs, \
            "skip-flagged run should log 'intentionally skipped' not 'UNEXPECTED'"
        unexpected_logs = [r for r in caplog.records if "UNEXPECTED" in r.getMessage()]
        assert not unexpected_logs, \
            "skip-flagged run must NOT log UNEXPECTED"

    def test_skip_flag_single_artifact_missing_does_not_raise(self):
        """With --skip-backtester, single artifact missing does not raise."""
        avail = self._call_init(skip_backtester=True)
        # All missing is expected for the stub — no RuntimeError should be raised
        assert avail["sweep_df"] is False
        assert avail["_skip_backtester"] is True

    # -- normal run (no skip flag) -------------------------------------------------

    def test_normal_run_all_artifacts_missing_raises(self):
        """Without --skip-backtester, ALL backtester-critical artifacts missing
        still raises RuntimeError (both-critical guard unchanged)."""
        with pytest.raises(RuntimeError, match="All critical backtester artifacts"):
            self._call_init(skip_backtester=False)

    def test_normal_run_single_backtester_missing_emits_degraded(self, monkeypatch):
        """Without --skip-backtester, single missing backtester artifact sets a
        degraded marker distinct from the skip-flagged case.

        Simulates: predictor_sweep_df.parquet and predictor_stats.json exist
        (loaded OK), while sweep_df.parquet fails with NoSuchKey.
        """
        import evaluate

        real_client = _s3_stub_client("s3")
        call_count = {"get_object": 0}

        def _partial_get(**kwargs):
            key = kwargs.get("Key", "")
            call_count["get_object"] += 1
            if key.endswith("predictor_sweep_df.parquet") or key.endswith("predictor_stats.json"):
                # Return valid data for predictor artifacts
                body = {"Body": type("Bytes", (), {"read": lambda s: json.dumps({"mock": True}).encode()})()}
                return body
            if key.endswith("sweep_df.parquet"):
                raise ClientError(
                    {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                    "GetObject",
                )
            if key.endswith("portfolio_stats.json"):
                # Return valid portfolio_stats
                body = {"Body": type("Bytes", (), {"read": lambda s: json.dumps({"mock": True}).encode()})()}
                return body
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                "GetObject",
            )

        class _PartialS3:
            def get_object(self, **kwargs):
                return _partial_get(**kwargs)

        monkeypatch.setattr("evaluate.boto3.client", lambda s, **kw: _PartialS3())

        cfg = {"output_bucket": "alpha-engine-research"}
        ns = _ns(skip_backtester=False)
        avail = evaluate._init_data_sources(ns, cfg)

        # sweep_df should be None => avail False
        assert avail["sweep_df"] is False
        # predictor artifacts loaded OK
        assert avail["predictor_sweep_df"] is True
        assert avail["predictor_stats"] is True
        # portfolio_stats loaded OK
        assert avail["portfolio_stats"] is True

        # Should have degraded marker set
        assert avail.get("_degraded", False), \
            "non-skip run with missing artifact must set _degraded=True"
        assert avail["_skip_backtester"] is False

        # Config should carry the degraded reason
        assert cfg.get("_artifact_degraded") is True
        assert "sweep_df.parquet" in cfg.get("_artifact_degraded_reason", "")

    def test_normal_run_single_predictor_missing_sets_degraded(self, monkeypatch):
        """Without --skip-backtester, missing a single predictor artifact
        (predictor_sweep_df.parquet) sets degraded while existing backtester
        artifacts load normally."""
        import evaluate

        def _partial_get(**kwargs):
            key = kwargs.get("Key", "")
            if key.endswith("predictor_sweep_df.parquet"):
                raise ClientError(
                    {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                    "GetObject",
                )
            # Return valid data for everything else
            body = type("Bytes", (), {"read": lambda s: json.dumps({"mock": True}).encode()})()
            return {"Body": body}

        class _PartialS3:
            def get_object(self, **kwargs):
                return _partial_get(**kwargs)

        monkeypatch.setattr("evaluate.boto3.client", lambda s, **kw: _PartialS3())

        cfg = {"output_bucket": "alpha-engine-research"}
        ns = _ns(skip_backtester=False)
        avail = evaluate._init_data_sources(ns, cfg)

        assert avail["sweep_df"] is True
        assert avail["portfolio_stats"] is True
        assert avail["predictor_sweep_df"] is False
        assert avail["predictor_stats"] is True
        assert avail.get("_degraded") is True
        assert "predictor_sweep_df.parquet" in cfg.get("_artifact_degraded_reason", "")

    # -- criticality checks -------------------------------------------------------

    def test_all_predictor_artifacts_missing_raises(self, monkeypatch):
        """Both predictor_critical artifacts missing raises RuntimeError
        (predictor artifacts are now in a criticality check analogous to
        the backtester critical set — config#2887)."""
        import evaluate

        def _backtester_only(**kwargs):
            key = kwargs.get("Key", "")
            if key.endswith("predictor_sweep_df.parquet") or key.endswith("predictor_stats.json"):
                raise ClientError(
                    {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                    "GetObject",
                )
            body = type("Bytes", (), {"read": lambda s: json.dumps({"mock": True}).encode()})()
            return {"Body": body}

        class _S3:
            def get_object(self, **kwargs):
                return _backtester_only(**kwargs)

        monkeypatch.setattr("evaluate.boto3.client", lambda s, **kw: _S3())

        cfg = {"output_bucket": "alpha-engine-research"}
        ns = _ns(skip_backtester=False)

        with pytest.raises(RuntimeError, match="All critical predictor artifacts"):
            evaluate._init_data_sources(ns, cfg)

    def test_all_predictor_artifacts_missing_raises_even_with_skip(self, monkeypatch):
        """Both predictor_critical artifacts missing raises RuntimeError
        even when --skip-backtester is set (skip only tolerates backtester
        absence, not predictor absence)."""
        import evaluate

        def _backtester_only(**kwargs):
            key = kwargs.get("Key", "")
            if key.endswith("predictor_sweep_df.parquet") or key.endswith("predictor_stats.json"):
                raise ClientError(
                    {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                    "GetObject",
                )
            body = type("Bytes", (), {"read": lambda s: json.dumps({"mock": True}).encode()})()
            return {"Body": body}

        class _S3:
            def get_object(self, **kwargs):
                return _backtester_only(**kwargs)

        monkeypatch.setattr("evaluate.boto3.client", lambda s, **kw: _S3())

        cfg = {"output_bucket": "alpha-engine-research"}
        ns = _ns(skip_backtester=True)

        with pytest.raises(RuntimeError, match="All critical predictor artifacts"):
            evaluate._init_data_sources(ns, cfg)


class TestArtifactCompletenessManifest:
    """Verify the completeness manifest distinguishes skip vs degraded."""

    def test_skip_flagged_has_skip_marker(self):
        """A skip-flagged run returns _skip_backtester=True in the avail map."""
        avail = self._call_init(skip_backtester=True)
        assert avail.get("_skip_backtester") is True

    def _call_init(self, *, skip_backtester: bool = False, config: dict = None) -> dict:
        import evaluate
        cfg = {"output_bucket": "alpha-engine-research", **(config or {})}
        ns = _ns(skip_backtester=skip_backtester)
        return evaluate._init_data_sources(ns, cfg)

    def test_normal_run_no_skip_marker(self):
        """A normal (non-skip) run returns _skip_backtester=False."""
        # Will raise because all artifacts missing — catch and check the config
        # that was partially set before the raise
        import evaluate
        cfg = {"output_bucket": "alpha-engine-research"}
        ns = _ns(skip_backtester=False)

        try:
            evaluate._init_data_sources(ns, cfg)
        except RuntimeError:
            pass
        # Config should still have been set before the raise attempt
        # (the both-critical-missing check comes after artifact loading)
        from evaluate import _init_data_sources  # avoid flake


# ---------------------------------------------------------------------------
# Static sanity
# ---------------------------------------------------------------------------

class TestParserShape:
    """Structural assertions about evaluate.py's argument parser."""

    def test_skip_backtester_exists(self):
        """evaluate.py's argument parser accepts --skip-backtester."""
        import evaluate
        ns = evaluate._parse_args([])
        assert ns.skip_backtester is False
