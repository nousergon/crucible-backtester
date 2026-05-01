"""
cost_report.py — render the per-run LLM cost summary as a markdown section
for the weekly evaluator email.

Reads the daily cost parquet emitted by alpha-engine-research's PR 3
aggregator (``scripts/aggregate_costs.py``) at::

    s3://alpha-engine-research/decision_artifacts/_cost/{YYYY-MM-DD}/cost.parquet

…and emits a markdown section with:

- Total cost + token totals (input / output / cache_read / cache_create)
- Drilldowns by sector_team_id, model_name, run_type, agent_id
- A "no cost data" placeholder when the parquet is absent (capture-flag
  off, Saturday SF didn't run, missing IAM, etc.) so the evaluator email
  doesn't break.

Failure posture:

- **Parquet absent** → graceful no-op + visible placeholder. Capture is
  opt-in (gated on ``ALPHA_ENGINE_DECISION_CAPTURE_ENABLED``); not every
  run produces cost data, so absence isn't a hard fail.
- **Parquet corrupt** → raise. Per ``feedback_no_silent_fails``, a
  malformed cost file should surface immediately (the upstream
  aggregator hard-fails on JSONL corruption; if the parquet is corrupt
  it's a writer bug worth investigating, not silent-skip material).

Workstream design: ``alpha-engine-docs/private/ROADMAP.md`` line ~1708
(per-run LLM cost telemetry, PR 4 of 5).
"""

from __future__ import annotations

import io
import logging
from typing import Any, Optional

import boto3
import pandas as pd
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


_DEFAULT_BUCKET = "alpha-engine-research"
_PARQUET_KEY_TEMPLATE = "decision_artifacts/_cost/{date}/cost.parquet"


# ── Markdown rendering ───────────────────────────────────────────────────


def _render_breakdown(title: str, breakdown: dict[str, float]) -> list[str]:
    """Render one breakdown section as a markdown table.

    Empty breakdowns are skipped (no header emitted). Sorted descending
    by spend so the top contributors land at the top of each section.
    """
    if not breakdown:
        return []
    lines = [f"### {title}", "", "| Key | Cost |", "|---|---|"]
    for k, v in sorted(breakdown.items(), key=lambda x: -x[1]):
        # Empty/None key → render as "(none)" so the row is still readable.
        key_str = k if (k and k != "nan") else "(none)"
        lines.append(f"| {key_str} | ${v:.4f} |")
    lines.append("")
    return lines


def _group_sum(df: pd.DataFrame, col: str) -> dict[str, float]:
    """Sum cost_usd grouped by ``col``. Missing column returns empty dict."""
    if col not in df.columns or "cost_usd" not in df.columns:
        return {}
    grouped = df.groupby(col, dropna=False)["cost_usd"].sum().fillna(0)
    return {str(k): float(v) for k, v in grouped.items()}


def render_cost_report_markdown(df: pd.DataFrame, *, run_date: str) -> str:
    """Render the cost section from a parquet-loaded DataFrame.

    Pure function; takes the data, returns the markdown string. Splitting
    rendering from S3 fetching keeps the renderer trivially testable
    without moto.
    """
    if df.empty:
        return _empty_section(run_date, reason="parquet contained zero rows")

    total_cost = float(df["cost_usd"].fillna(0).sum()) if "cost_usd" in df.columns else 0.0
    total_input = int(df["input_tokens"].fillna(0).sum()) if "input_tokens" in df.columns else 0
    total_output = int(df["output_tokens"].fillna(0).sum()) if "output_tokens" in df.columns else 0
    total_cr = int(df["cache_read_tokens"].fillna(0).sum()) if "cache_read_tokens" in df.columns else 0
    total_cc = int(df["cache_create_tokens"].fillna(0).sum()) if "cache_create_tokens" in df.columns else 0

    out: list[str] = [
        "## LLM cost report",
        "",
        f"- Run date: {run_date}",
        f"- Per-call rows: {len(df):,}",
        f"- **Total cost: ${total_cost:.4f}**",
        f"- Total input tokens: {total_input:,}",
        f"- Total output tokens: {total_output:,}",
        f"- Total cache_read tokens: {total_cr:,}",
        f"- Total cache_create tokens: {total_cc:,}",
        "",
    ]
    out.extend(_render_breakdown("By sector team", _group_sum(df, "sector_team_id")))
    out.extend(_render_breakdown("By model", _group_sum(df, "model_name")))
    out.extend(_render_breakdown("By run_type", _group_sum(df, "run_type")))
    out.extend(_render_breakdown("By agent_id", _group_sum(df, "agent_id")))
    return "\n".join(out)


def _empty_section(run_date: str, *, reason: str) -> str:
    """Render the placeholder section when no cost data is available.

    Always renders a section (rather than emitting nothing) so the
    evaluator email signals to the operator that cost capture was
    expected but absent — silent omission would mask a flag-flip
    regression.
    """
    return "\n".join([
        "## LLM cost report",
        "",
        f"- Run date: {run_date}",
        f"- _No cost data available_ ({reason}).",
        "  Capture is gated on `ALPHA_ENGINE_DECISION_CAPTURE_ENABLED`; "
        "set to `true` on the research Lambda to begin populating "
        "`s3://alpha-engine-research/decision_artifacts/_cost_raw/`.",
        "",
    ])


# ── S3 fetcher ───────────────────────────────────────────────────────────


def _build_parquet_key(run_date: str) -> str:
    return _PARQUET_KEY_TEMPLATE.format(date=run_date)


def fetch_cost_parquet(
    run_date: str,
    *,
    bucket: str = _DEFAULT_BUCKET,
    s3_client: Optional[Any] = None,
) -> Optional[pd.DataFrame]:
    """Fetch the cost parquet for ``run_date`` from S3.

    Returns the parsed DataFrame, or ``None`` when the parquet is absent
    (NoSuchKey / 404). Raises on any other S3 failure or on parquet
    corruption per ``feedback_no_silent_fails``.

    Cross-bucket read: backtester runs in its own context but reads from
    the alpha-engine-research bucket where the cost data is staged. The
    spot instance's IAM role already has read access to that bucket.
    """
    client = s3_client if s3_client is not None else boto3.client("s3")
    key = _build_parquet_key(run_date)
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404", "NotFound"):
            logger.info(
                "[cost_report] no cost parquet at s3://%s/%s — "
                "skipping cost section", bucket, key,
            )
            return None
        # Other ClientError (AccessDenied, etc.) is worth raising — it
        # indicates a wiring or IAM regression rather than absent data.
        logger.error(
            "[cost_report] S3 read failed for s3://%s/%s: %s",
            bucket, key, exc,
        )
        raise

    body = obj["Body"].read()
    return pd.read_parquet(io.BytesIO(body), engine="pyarrow")


# ── Public entrypoint ────────────────────────────────────────────────────


def build_cost_section(
    run_date: str,
    *,
    bucket: str = _DEFAULT_BUCKET,
    s3_client: Optional[Any] = None,
) -> str:
    """Build the LLM cost markdown section for the evaluator email.

    Convenience wrapper around ``fetch_cost_parquet`` + ``render_cost_report_markdown``.
    Always returns a markdown string — empty/missing data renders the
    placeholder section so the evaluator email always carries the
    cost-section header (signals to operators that capture is expected).

    Parameters
    ----------
    run_date
        ISO date string (YYYY-MM-DD) — corresponds to the
        ``decision_artifacts/_cost/{date}/`` partition.
    bucket
        S3 bucket holding the parquet. Defaults to the research bucket
        (cross-bucket read from the backtester's context).
    s3_client
        For testing — pass a moto-mocked client. Defaults to
        ``boto3.client("s3")``.
    """
    df = fetch_cost_parquet(run_date, bucket=bucket, s3_client=s3_client)
    if df is None:
        return _empty_section(run_date, reason="parquet not found at expected key")
    return render_cost_report_markdown(df, run_date=run_date)
