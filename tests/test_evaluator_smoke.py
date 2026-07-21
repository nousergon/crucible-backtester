"""tests/test_evaluator_smoke.py — evaluate.py --smoke (config#3121).

Every _KNOWN_STAGES entry in infrastructure/spot_backtest.sh must declare
an execution smoke; the evaluator previously had only input_quality_gate
(an INPUT check on signal quality, not an execution smoke of evaluate.py's
own imports/config/S3-wiring). This adds --smoke: cheap preflight
(BacktesterPreflight(mode="evaluate") + a read-only S3 reachability
probe) that exits 0 before any diagnostics/optimizer work or config
writes — mirroring backtest.py --mode=smoke's philosophy for the
separate evaluate.py entrypoint.
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

import evaluate


def test_smoke_flag_accepted():
    with patch("sys.argv", ["evaluate.py", "--smoke"]):
        args = evaluate._parse_args()
    assert args.smoke is True


def test_smoke_flag_defaults_false():
    with patch("sys.argv", ["evaluate.py"]):
        args = evaluate._parse_args()
    assert args.smoke is False


def test_smoke_probe_s3_calls_list_objects_v2_on_backtest_prefix():
    fake_s3 = MagicMock()
    with patch("boto3.client", return_value=fake_s3):
        evaluate._smoke_probe_s3({"signals_bucket": "my-bucket"})

    fake_s3.list_objects_v2.assert_called_once_with(
        Bucket="my-bucket", Prefix="backtest/", MaxKeys=1,
    )


def test_smoke_probe_s3_raises_on_failure():
    """The whole point of a smoke step is to fail loud and fast — a
    ClientError (bad credentials, wrong region, etc.) must propagate,
    not be swallowed."""
    fake_s3 = MagicMock()
    fake_s3.list_objects_v2.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "nope"}}, "ListObjectsV2",
    )
    with patch("boto3.client", return_value=fake_s3):
        with pytest.raises(ClientError):
            evaluate._smoke_probe_s3({"signals_bucket": "my-bucket"})


def test_smoke_probe_s3_defaults_bucket_when_unset():
    fake_s3 = MagicMock()
    with patch("boto3.client", return_value=fake_s3):
        evaluate._smoke_probe_s3({})

    fake_s3.list_objects_v2.assert_called_once_with(
        Bucket="alpha-engine-research", Prefix="backtest/", MaxKeys=1,
    )


def test_main_impl_smoke_returns_before_data_source_init():
    """Source pin: --smoke must exit BEFORE _init_data_sources (which
    would raise on a real run with no backtest artifacts for --date —
    not a condition a smoke run should depend on) and before any
    optimizer/diagnostics work. Locates the `if args.smoke:` branch and
    asserts it appears strictly before the _init_data_sources call site
    in _main_impl's source."""
    import inspect
    src = inspect.getsource(evaluate._main_impl)
    smoke_idx = src.index("if args.smoke:")
    init_idx = src.index("_init_data_sources(args, config)")
    assert smoke_idx < init_idx, (
        "--smoke must short-circuit before _init_data_sources runs"
    )
    # And the smoke branch itself must return (not fall through).
    branch = src[smoke_idx:smoke_idx + 400]
    assert "return" in branch
    assert "_smoke_probe_s3(config)" in branch


def test_main_impl_smoke_runs_after_preflight():
    """--smoke must run AFTER BacktesterPreflight (so preflight's own
    checks — bucket exists, imports — still apply) but the smoke check
    supersedes needing simulation artifacts to exist."""
    import inspect
    src = inspect.getsource(evaluate._main_impl)
    preflight_idx = src.index('mode="evaluate",')
    smoke_idx = src.index("if args.smoke:")
    assert preflight_idx < smoke_idx
