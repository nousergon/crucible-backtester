"""Unit tests for analysis/calibration_report.py (ROADMAP L480 — the
backtester evaluator-email leg of the judge-calibration κ report)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from analysis.calibration_report import (
    _LATEST_MD_KEY,
    _SECTION_HEADER,
    build_calibration_section,
)

_BUCKET = "alpha-engine-research"

_SAMPLE_MD = (
    "## Judge calibration (κ)\n\n"
    "_2/3 cell(s) at the ≥30-review threshold._\n\n"
    "| rubric · dimension | n | κ blind | κ final | exact | α |\n"
    "|---|---|---|---|---|---|\n"
    "| `thesis_update` · `completeness` | 30/30 | 0.62 | 0.71 | 60% | 0.58 |\n"
)


def _stub_with_body(body: bytes) -> MagicMock:
    stub = MagicMock()
    stub.get_object.return_value = {"Body": MagicMock(read=lambda: body)}
    return stub


def _stub_with_error(code: str) -> MagicMock:
    stub = MagicMock()
    stub.get_object.side_effect = ClientError(
        {"Error": {"Code": code, "Message": code}}, "GetObject"
    )
    return stub


def test_happy_path_embeds_rendered_markdown():
    stub = _stub_with_body(_SAMPLE_MD.encode("utf-8"))
    md = build_calibration_section(s3_client=stub)
    assert md.startswith(_SECTION_HEADER)
    assert "thesis_update" in md
    stub.get_object.assert_called_once_with(Bucket=_BUCKET, Key=_LATEST_MD_KEY)


def test_missing_report_renders_placeholder():
    stub = _stub_with_error("NoSuchKey")
    md = build_calibration_section(s3_client=stub)
    assert md.startswith(_SECTION_HEADER)
    assert "No calibration κ report available" in md
    assert "Calibrate tab" in md


def test_other_s3_error_degrades_to_placeholder_not_raise():
    # Unlike the cost parquet (which raises on AccessDenied), the κ
    # section is purely informational — an IAM gap must not crash the
    # evaluator email, so it degrades to a placeholder with the code.
    stub = _stub_with_error("AccessDenied")
    md = build_calibration_section(s3_client=stub)
    assert md.startswith(_SECTION_HEADER)
    assert "AccessDenied" in md


def test_empty_report_renders_placeholder():
    stub = _stub_with_body(b"   \n")
    md = build_calibration_section(s3_client=stub)
    assert "report was empty" in md


def test_headerless_markdown_is_wrapped():
    stub = _stub_with_body(b"some raw body without a header")
    md = build_calibration_section(s3_client=stub)
    assert md.startswith(_SECTION_HEADER)
    assert "some raw body" in md


def test_section_always_ends_with_newline():
    stub = _stub_with_body(_SAMPLE_MD.encode("utf-8"))
    md = build_calibration_section(s3_client=stub)
    assert md.endswith("\n")
