"""Tests for analysis/backfill_cumulative_trial_count.py (config#2454).

Uses moto to fake a small dated-archive history across the 3 backfillable
producers (optimizer_param_sweep / gamma_sweep / cov_estimator_sweep) and
pins:
  1. Sums n_trials across multiple dated archives per producer.
  2. Falls back to len(cells) for pre-#2454 archives lacking n_trials.
  3. Skips (doesn't count) archives with status != "ok".
  4. Skips (logs, doesn't crash) a corrupt/unparseable archive.
  5. main() with --dry-run prints the sums but does not write the counter.
  6. main() without --dry-run writes the seeded counter artifact.
  7. main() refuses to reseed a non-zero counter without --overwrite.
"""

from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

from analysis.backfill_cumulative_trial_count import (
    main,
    sum_historical_trials,
)

BUCKET = "alpha-engine-research"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _put(s3, key, doc):
    s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(doc).encode())


class TestSumHistoricalTrials:
    def test_sums_n_trials_across_dates(self, s3):
        _put(s3, "backtest/2026-06-01/gamma_sweep.json", {"status": "ok", "n_trials": 5})
        _put(s3, "backtest/2026-06-08/gamma_sweep.json", {"status": "ok", "n_trials": 5})
        _put(s3, "backtest/2026-06-15/gamma_sweep.json", {"status": "ok", "n_trials": 5})
        total, n_read, n_skipped = sum_historical_trials(
            BUCKET, "gamma_sweep", "gamma_sweep.json", s3_client=s3,
        )
        assert total == 15
        assert n_read == 3
        assert n_skipped == 0

    def test_falls_back_to_len_cells_when_n_trials_absent(self, s3):
        """Pre-config#2454 archives never persisted n_trials — only cells."""
        _put(s3, "backtest/2026-01-01/cov_sweep.json", {
            "status": "ok",
            "cells": {f"cell_{i}": {} for i in range(8)},
        })
        total, n_read, _ = sum_historical_trials(
            BUCKET, "cov_estimator_sweep", "cov_sweep.json", s3_client=s3,
        )
        assert total == 8
        assert n_read == 1

    def test_skips_non_ok_status_archives(self, s3):
        _put(s3, "backtest/2026-06-01/optimizer_param_sweep.json", {
            "status": "ok", "n_trials": 9,
        })
        _put(s3, "backtest/2026-06-08/optimizer_param_sweep.json", {
            "status": "skipped", "reason": "production inputs not ready",
        })
        total, n_read, n_skipped = sum_historical_trials(
            BUCKET, "optimizer_param_sweep", "optimizer_param_sweep.json", s3_client=s3,
        )
        assert total == 9
        assert n_read == 1
        assert n_skipped == 1

    def test_skips_corrupt_archive_without_crashing(self, s3, caplog):
        s3.put_object(
            Bucket=BUCKET, Key="backtest/2026-06-01/gamma_sweep.json",
            Body=b"{not valid json",
        )
        _put(s3, "backtest/2026-06-08/gamma_sweep.json", {"status": "ok", "n_trials": 5})
        total, n_read, n_skipped = sum_historical_trials(
            BUCKET, "gamma_sweep", "gamma_sweep.json", s3_client=s3,
        )
        assert total == 5
        assert n_read == 1
        assert n_skipped == 1
        assert any("failed to read/parse" in r.message for r in caplog.records)

    def test_only_matches_exact_filename_suffix(self, s3):
        """A key that merely contains the filename as a substring (not an
        exact /{filename} suffix) must not be double-counted."""
        _put(s3, "backtest/2026-06-01/gamma_sweep.json", {"status": "ok", "n_trials": 5})
        _put(s3, "backtest/2026-06-01/legacy_gamma_sweep.json", {"status": "ok", "n_trials": 999})
        total, n_read, _ = sum_historical_trials(
            BUCKET, "gamma_sweep", "gamma_sweep.json", s3_client=s3,
        )
        assert total == 5
        assert n_read == 1


class TestMainCLI:
    def _seed_all_producers(self, s3):
        _put(s3, "backtest/2026-06-01/optimizer_param_sweep.json", {"status": "ok", "n_trials": 9})
        _put(s3, "backtest/2026-06-01/gamma_sweep.json", {"status": "ok", "n_trials": 5})
        _put(s3, "backtest/2026-06-01/cov_sweep.json", {"status": "ok", "n_trials": 8})

    def test_dry_run_does_not_write_counter(self, s3, capsys, monkeypatch):
        monkeypatch.setattr("boto3.client", lambda *a, **kw: s3)
        self._seed_all_producers(s3)
        rc = main(["--bucket", BUCKET, "--dry-run"])
        assert rc == 0
        out = capsys.readouterr().out
        assert '"total": 22' in out
        with pytest.raises(Exception):
            s3.get_object(Bucket=BUCKET, Key="backtest/cumulative_trial_count.json")

    def test_writes_counter_when_not_dry_run(self, s3, monkeypatch):
        monkeypatch.setattr("boto3.client", lambda *a, **kw: s3)
        self._seed_all_producers(s3)
        rc = main(["--bucket", BUCKET, "--run-date", "2026-07-14"])
        assert rc == 0
        obj = s3.get_object(Bucket=BUCKET, Key="backtest/cumulative_trial_count.json")
        state = json.loads(obj["Body"].read())
        assert state["total"] == 22
        assert state["by_producer"]["optimizer_param_sweep"] == 9
        assert state["by_producer"]["gamma_sweep"] == 5
        assert state["by_producer"]["cov_estimator_sweep"] == 8
        assert "predictor_param_sweep" not in state["by_producer"]

    def test_refuses_reseed_without_overwrite(self, s3, monkeypatch):
        monkeypatch.setattr("boto3.client", lambda *a, **kw: s3)
        self._seed_all_producers(s3)
        rc1 = main(["--bucket", BUCKET, "--run-date", "2026-07-14"])
        assert rc1 == 0
        rc2 = main(["--bucket", BUCKET, "--run-date", "2026-07-21"])
        assert rc2 == 1
        # Original seed untouched.
        obj = s3.get_object(Bucket=BUCKET, Key="backtest/cumulative_trial_count.json")
        state = json.loads(obj["Body"].read())
        assert state["last_updated"] == "2026-07-14"
