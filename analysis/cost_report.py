"""
cost_report.py — render the per-run LLM cost summary as a markdown section
for the weekly evaluator email.

Reads the daily cost parquet emitted by alpha-engine-research's PR 3
aggregator (``scripts/aggregate_costs.py``) at::

    s3://alpha-engine-research/decision_artifacts/_cost/{YYYY-MM-DD}/cost.parquet

…and emits a markdown section with:

- Total cost + token totals (input / output / cache_read / cache_create)
- Drilldowns by sector_team_id, model_name, run_type, agent_id
- **Anomaly section (PR 5b)** — compares this run's total against a
  rolling 4-week baseline; flags + logs WARN if the ratio exceeds
  ``ALPHA_ENGINE_COST_ANOMALY_RATIO`` (default 2.0×).
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
- **Anomaly detected** → log WARN + render the alert section in the
  email. NOT a hard fail — the alert is a notification, not a gate
  (the run-budget hard ceiling in alpha-engine-research PR 5a covers
  the fail-loud safety surface).

Workstream design: ``alpha-engine-docs/private/ROADMAP.md`` line ~1708
(per-run LLM cost telemetry, PR 4-5 of 5).
"""

from __future__ import annotations

import io
import logging
import os
from datetime import date as date_type, datetime, timedelta
from typing import Any, Optional

import boto3
import pandas as pd
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


_DEFAULT_BUCKET = "alpha-engine-research"
_PARQUET_KEY_TEMPLATE = "decision_artifacts/_cost/{date}/cost.parquet"

_ANOMALY_RATIO_ENV_VAR = "ALPHA_ENGINE_COST_ANOMALY_RATIO"
_ANOMALY_RATIO_DEFAULT = 2.0
_ANOMALY_BASELINE_WEEKS = 4  # rolling window per ROADMAP P5


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


# ── Anomaly detection (PR 5b) ────────────────────────────────────────────


def _resolve_anomaly_ratio() -> float:
    """Read ``ALPHA_ENGINE_COST_ANOMALY_RATIO`` from env (default 2.0×).

    Returns the configured ratio. Zero or negative disables the alert
    (matches the run-budget ceiling convention in research PR 5a). On
    parse failure, log WARN + return 0 (disable) — a malformed env var
    shouldn't take down a Sat SF run, but the WARN is loud enough that
    the operator sees it.
    """
    raw = os.environ.get(_ANOMALY_RATIO_ENV_VAR, "")
    if not raw:
        return _ANOMALY_RATIO_DEFAULT
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "[cost_report] %s=%r is not a number; disabling anomaly "
            "alerting (set to a positive float to enable)",
            _ANOMALY_RATIO_ENV_VAR, raw,
        )
        return 0.0


def _previous_weekly_dates(run_date: str, *, weeks: int) -> list[str]:
    """Return ISO dates for the previous N weekly runs ending one week
    before ``run_date``.

    Saturday SF runs weekly; for a Saturday at run_date X, the previous
    4 are at X-7, X-14, X-21, X-28 (in ISO form). Caller filters out
    dates whose parquets don't exist (a recent capture-flag toggle or a
    missed Sat SF means the baseline is shorter than the requested
    window — which the renderer notes explicitly).
    """
    base = date_type.fromisoformat(run_date)
    return [
        (base - timedelta(weeks=i)).isoformat()
        for i in range(1, weeks + 1)
    ]


def _fetch_total_cost_for_date(
    run_date: str,
    *,
    bucket: str,
    s3_client: Any,
) -> Optional[float]:
    """Sum ``cost_usd`` from the parquet at ``run_date``. Returns None
    if the parquet is absent (NoSuchKey); other S3 errors propagate."""
    df = fetch_cost_parquet(run_date, bucket=bucket, s3_client=s3_client)
    if df is None or df.empty or "cost_usd" not in df.columns:
        return None
    return float(df["cost_usd"].fillna(0).sum())


def detect_anomaly(
    run_date: str,
    current_total_cost_usd: float,
    *,
    bucket: str = _DEFAULT_BUCKET,
    s3_client: Optional[Any] = None,
    weeks: int = _ANOMALY_BASELINE_WEEKS,
) -> dict:
    """Compare ``current_total_cost_usd`` against the rolling baseline of
    the previous ``weeks`` Saturday runs.

    Returns a dict with:

    - ``current_total_usd``: passed-through current spend.
    - ``baseline_dates_found``: list of ISO dates that had parquets.
    - ``baseline_dates_missing``: list of ISO dates with no parquet.
    - ``baseline_mean_usd``: mean spend across found dates (None if zero).
    - ``ratio``: ``current / baseline_mean``, or None if baseline empty.
    - ``threshold_ratio``: configured anomaly threshold.
    - ``is_anomaly``: True iff ``ratio`` is finite and exceeds threshold.
    - ``status``: "ok" | "anomaly" | "no_baseline" | "alerting_disabled".

    No raise on missing baselines — the renderer surfaces this state
    instead so operators see "first run, no baseline yet" rather than
    a silent omission.
    """
    threshold = _resolve_anomaly_ratio()
    if threshold <= 0:
        return {
            "current_total_usd": current_total_cost_usd,
            "baseline_dates_found": [],
            "baseline_dates_missing": [],
            "baseline_mean_usd": None,
            "ratio": None,
            "threshold_ratio": threshold,
            "is_anomaly": False,
            "status": "alerting_disabled",
        }

    client = s3_client if s3_client is not None else boto3.client("s3")
    prior_dates = _previous_weekly_dates(run_date, weeks=weeks)
    found: list[tuple[str, float]] = []
    missing: list[str] = []
    for d in prior_dates:
        total = _fetch_total_cost_for_date(d, bucket=bucket, s3_client=client)
        if total is None:
            missing.append(d)
        else:
            found.append((d, total))

    if not found:
        return {
            "current_total_usd": current_total_cost_usd,
            "baseline_dates_found": [],
            "baseline_dates_missing": prior_dates,
            "baseline_mean_usd": None,
            "ratio": None,
            "threshold_ratio": threshold,
            "is_anomaly": False,
            "status": "no_baseline",
        }

    baseline_mean = sum(c for _, c in found) / len(found)
    ratio = (current_total_cost_usd / baseline_mean) if baseline_mean > 0 else None
    is_anomaly = ratio is not None and ratio > threshold

    if is_anomaly:
        logger.warning(
            "[cost_report] cost anomaly: run_date=%s current=$%.4f "
            "baseline_mean=$%.4f (over %d weeks) ratio=%.2fx > threshold=%.2fx",
            run_date, current_total_cost_usd, baseline_mean,
            len(found), ratio, threshold,
        )

    return {
        "current_total_usd": current_total_cost_usd,
        "baseline_dates_found": [d for d, _ in found],
        "baseline_dates_missing": missing,
        "baseline_mean_usd": baseline_mean,
        "ratio": ratio,
        "threshold_ratio": threshold,
        "is_anomaly": is_anomaly,
        "status": "anomaly" if is_anomaly else "ok",
    }


def render_anomaly_section(anomaly: dict) -> str:
    """Render the anomaly detection block as markdown.

    Always emits a section header so capture-flag-on weeks always carry
    the anomaly surface (silent omission would mask a regression).
    """
    status = anomaly.get("status", "ok")
    lines = ["### Anomaly check", ""]
    if status == "alerting_disabled":
        lines.append(
            f"- _Anomaly alerting disabled_ (`{_ANOMALY_RATIO_ENV_VAR}` "
            "≤ 0). Set to a positive float to enable.",
        )
    elif status == "no_baseline":
        missing = anomaly.get("baseline_dates_missing") or []
        lines.append(
            f"- _No baseline available_ (no parquets found for the "
            f"previous {len(missing)} weekly runs). Anomaly check "
            "deferred until baseline accumulates.",
        )
    else:
        current = anomaly.get("current_total_usd", 0.0)
        baseline = anomaly.get("baseline_mean_usd", 0.0) or 0.0
        ratio = anomaly.get("ratio")
        threshold = anomaly.get("threshold_ratio", 0.0)
        n_found = len(anomaly.get("baseline_dates_found") or [])
        n_missing = len(anomaly.get("baseline_dates_missing") or [])
        ratio_str = f"{ratio:.2f}x" if ratio is not None else "N/A"
        lines.append(f"- Current run total: ${current:.4f}")
        lines.append(
            f"- Baseline (mean of {n_found} prior weekly runs): ${baseline:.4f}",
        )
        lines.append(
            f"- Ratio: **{ratio_str}** (threshold: {threshold:.2f}x)",
        )
        if n_missing:
            lines.append(
                f"- Baseline gaps: {n_missing} of {n_found + n_missing} "
                "prior weekly runs had no captured parquet (capture flag "
                "may have been off; check the corresponding evaluator emails).",
            )
        if anomaly.get("is_anomaly"):
            lines.insert(2, "")
            lines.insert(2, "**:warning: ANOMALY DETECTED — current run exceeds threshold above baseline.**")
    lines.append("")
    return "\n".join(lines)


# ── Public entrypoint ────────────────────────────────────────────────────


def build_cost_section(
    run_date: str,
    *,
    bucket: str = _DEFAULT_BUCKET,
    s3_client: Optional[Any] = None,
) -> str:
    """Build the LLM cost markdown section for the evaluator email.

    Convenience wrapper around ``fetch_cost_parquet`` + ``render_cost_report_markdown`` +
    ``detect_anomaly`` + ``render_anomaly_section``. Always returns a
    markdown string — empty/missing data renders the placeholder section
    so the evaluator email always carries the cost-section header
    (signals to operators that capture is expected).

    PR 5b: when the current parquet is present, runs anomaly detection
    against the rolling 4-week baseline and appends an alert section
    (or a "no baseline yet" / "alerting disabled" placeholder).

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
    client = s3_client if s3_client is not None else boto3.client("s3")
    df = fetch_cost_parquet(run_date, bucket=bucket, s3_client=client)
    if df is None:
        return _empty_section(run_date, reason="parquet not found at expected key")

    main_md = render_cost_report_markdown(df, run_date=run_date)
    current_total = float(df["cost_usd"].fillna(0).sum()) if "cost_usd" in df.columns else 0.0
    anomaly = detect_anomaly(
        run_date, current_total, bucket=bucket, s3_client=client,
    )
    return main_md + "\n" + render_anomaly_section(anomaly)
