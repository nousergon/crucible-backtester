"""calibration_report.py — surface the judge-calibration κ report as a
markdown section for the weekly evaluator email (ROADMAP L480).

The metric is computed upstream in alpha-engine-research
(`evals/calibration_kappa.py`, run weekly by the EvalRollingMean Lambda),
which writes a pre-rendered markdown report to
``decision_artifacts/_calibration/_report/latest/kappa.md`` on the
research bucket. Research owns the rendering; this module only fetches
that markdown and embeds it, so there is exactly one κ-rendering source
across the two repos.

Always returns a markdown string — a missing report (no Saturday SF yet,
flag off, IAM denial) renders a placeholder section so the calibration
surface is always visible to operators. Mirrors the cost_report.py
fetch-or-placeholder contract.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

_DEFAULT_BUCKET = "alpha-engine-research"
_LATEST_MD_KEY = "decision_artifacts/_calibration/_report/latest/kappa.md"

_SECTION_HEADER = "## Judge calibration (κ)"


def _placeholder(reason: str) -> str:
    return "\n".join([
        _SECTION_HEADER,
        "",
        f"- _No calibration κ report available ({reason})._",
        "  Computed weekly by alpha-engine-research "
        "`evals/calibration_kappa.py`; seed the corpus via the Calibrate "
        "tab on dashboard page 8.",
        "",
    ])


def build_calibration_section(
    *,
    bucket: Optional[str] = None,
    s3_client: Any = None,
    key: str = _LATEST_MD_KEY,
) -> str:
    """Fetch the pre-rendered κ markdown from S3 and return it as a
    section. Never raises — a missing/unreadable report degrades to a
    placeholder so the evaluator email always carries the κ header."""
    bkt = bucket or _DEFAULT_BUCKET
    client = s3_client if s3_client is not None else boto3.client("s3")

    try:
        md = client.get_object(Bucket=bkt, Key=key)["Body"].read().decode("utf-8")
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404", "NoSuchBucket"):
            logger.info(
                "[calibration_report] no κ report at s3://%s/%s — "
                "placeholder section",
                bkt, key,
            )
            return _placeholder("report not found at expected key")
        logger.warning(
            "[calibration_report] S3 read failed for s3://%s/%s: %s",
            bkt, key, exc,
        )
        return _placeholder(f"S3 read error: {code or exc}")
    except Exception as exc:  # noqa: BLE001 — section must never crash the email
        logger.warning(
            "[calibration_report] unexpected error reading s3://%s/%s: %s",
            bkt, key, exc,
        )
        return _placeholder(f"unexpected error: {exc}")

    md = md.strip()
    if not md:
        return _placeholder("report was empty")
    if not md.startswith(_SECTION_HEADER):
        # Upstream always renders with this header; if it's missing the
        # report is malformed — wrap it so the section is still valid.
        return f"{_SECTION_HEADER}\n\n{md}\n"
    return md + "\n"
