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

Workstream design: ``alpha-engine-config/private-docs/ROADMAP.md`` line ~1708
(per-run LLM cost telemetry, PR 4-5 of 5).
"""

from __future__ import annotations

import io
import json
import logging
import os
from datetime import date as date_type, datetime, timedelta, timezone
from hashlib import sha1
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

# Opt-out gate for the operator-facing SNS+Telegram fan-out (ROADMAP L717).
# Default-on — anomaly entries always publish unless explicitly disabled.
# Tests set this to "1" to suppress real-network attempts inside the
# moto-stubbed S3 paths.
_ANOMALY_ALERT_DISABLED_ENV_VAR = "ALPHA_ENGINE_COST_ANOMALY_ALERT_DISABLED"

# Registry row id read from nousergon_lib.transparency_inventory.yaml
# to source the cost-telemetry effective_date (the floor of the
# rolling baseline window). Keep in sync with the row id in lib's
# inventory; missing-row → fallback to no-floor (legacy "all gaps
# treated equal" classification, with a WARN log for visibility).
_COST_TELEMETRY_REGISTRY_ROW_ID = "cost_telemetry"

# Schema-1.0.0 changelog corpus (ROADMAP P0 sub-item 5 auto-population —
# cost-anomaly half, 2026-05-07). When detect_anomaly() returns
# status="anomaly", build_cost_section() writes one structured incident
# entry to the same prefix the SNS-mirror + cloudwatch-mirror Lambdas
# already populate, so the retro-mining filter + downstream aggregator
# see cost spikes as first-class events.
_CHANGELOG_PREFIX = "changelog/entries"
_CHANGELOG_SCHEMA_VERSION = "1.0.0"


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


def _telemetry_first_capture_date(
    *,
    inventory: Optional[dict] = None,
) -> Optional[str]:
    """Return the cost-telemetry effective_date from the substrate
    inventory registry, or ``None`` if the row is missing.

    The substrate inventory (alpha-engine-lib's
    ``transparency_inventory.yaml``) is the single source of truth for
    when each measurement-output substrate started capturing data. The
    cost_telemetry row's ``effective_date`` field is the first Saturday
    SF run that captured a parquet; consumers (this anomaly detector +
    the substrate health checker) read it as a structural floor.

    Used by :func:`detect_anomaly` to distinguish:

    1. **Pre-telemetry** — prior weekly dates that fall *before* the
       row's effective_date. These dates inherently have no parquets
       and the renderer surfaces that explicitly.
    2. **Genuine missing** — prior weekly dates *after* the floor that
       still have no parquet. These warrant the "capture flag may have
       been off" framing.

    Inventory load failures (yaml missing, lib import fails, row
    absent) fall back to ``None`` with a WARN log so the legacy
    "all gaps treated equal" classification still fires — the
    detector remains functionally degraded but doesn't take down the
    Saturday SF email.

    ``inventory`` injection lets tests stub the registry without
    monkey-patching the lib loader.
    """
    inv = inventory
    if inv is None:
        try:
            from nousergon_lib.transparency import load_inventory
        except ImportError as exc:
            logger.warning(
                "[cost_report] nousergon_lib.transparency unavailable "
                "(%s) — falling back to no-pre-telemetry-floor "
                "classification. Bump the lib pin to >=0.7.1 to enable.",
                exc,
            )
            return None
        try:
            inv = load_inventory()
        except (OSError, ValueError) as exc:
            logger.warning(
                "[cost_report] load_inventory failed (%s) — falling back "
                "to no-pre-telemetry-floor classification",
                exc,
            )
            return None
    rows = inv.get("inventory") or []
    row = next(
        (r for r in rows if r.get("id") == _COST_TELEMETRY_REGISTRY_ROW_ID),
        None,
    )
    if row is None:
        logger.warning(
            "[cost_report] substrate inventory has no %r row — falling "
            "back to no-pre-telemetry-floor classification. Bump the lib "
            "pin to a version that ships the row.",
            _COST_TELEMETRY_REGISTRY_ROW_ID,
        )
        return None
    effective = row.get("effective_date")
    if effective is None:
        return None
    # YAML may load the date as a datetime.date; coerce to ISO string.
    return str(effective)


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
    inventory: Optional[dict] = None,
) -> dict:
    """Compare ``current_total_cost_usd`` against the rolling baseline of
    the previous ``weeks`` Saturday runs.

    The pre-telemetry floor (the date before which baseline gaps are
    structural rather than operator-flag-off) is sourced from the
    alpha-engine-lib substrate inventory's ``cost_telemetry`` row.
    Tests can inject an ``inventory`` dict to bypass the lib loader.

    Returns a dict with:

    - ``current_total_usd``: passed-through current spend.
    - ``baseline_dates_found``: list of ISO dates that had parquets.
    - ``baseline_dates_missing``: list of ISO dates with no parquet that
      fall *after* telemetry shipped (genuine post-launch gaps —
      capture-flag-off / partial SF / Lambda timeout class).
    - ``baseline_dates_pre_telemetry``: list of ISO dates that pre-date
      telemetry's first captured run. Structurally absent — not a bug.
    - ``telemetry_first_date``: cost_telemetry row's effective_date in
      the substrate inventory, or ``None`` if the row / lib is absent.
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
            "baseline_dates_pre_telemetry": [],
            "telemetry_first_date": None,
            "baseline_mean_usd": None,
            "ratio": None,
            "threshold_ratio": threshold,
            "is_anomaly": False,
            "status": "alerting_disabled",
        }

    client = s3_client if s3_client is not None else boto3.client("s3")
    prior_dates = _previous_weekly_dates(run_date, weeks=weeks)
    telemetry_first = _telemetry_first_capture_date(inventory=inventory)
    found: list[tuple[str, float]] = []
    missing: list[str] = []
    pre_telemetry: list[str] = []
    for d in prior_dates:
        if telemetry_first is not None and d < telemetry_first:
            pre_telemetry.append(d)
            continue
        total = _fetch_total_cost_for_date(d, bucket=bucket, s3_client=client)
        if total is None:
            missing.append(d)
        else:
            found.append((d, total))

    if not found:
        return {
            "current_total_usd": current_total_cost_usd,
            "baseline_dates_found": [],
            "baseline_dates_missing": missing,
            "baseline_dates_pre_telemetry": pre_telemetry,
            "telemetry_first_date": telemetry_first,
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
        "baseline_dates_pre_telemetry": pre_telemetry,
        "telemetry_first_date": telemetry_first,
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
        pre_telemetry = anomaly.get("baseline_dates_pre_telemetry") or []
        telemetry_first = anomaly.get("telemetry_first_date")
        n_pre = len(pre_telemetry)
        n_missing = len(missing)
        n_total = n_pre + n_missing
        if n_pre and not n_missing and telemetry_first:
            lines.append(
                f"- _No baseline available yet_ — all {n_pre} of the "
                f"previous weekly runs pre-date the cost-telemetry feature "
                f"(first captured: {telemetry_first}). Baseline fills as "
                "post-launch Saturdays accumulate.",
            )
        elif n_pre and n_missing:
            lines.append(
                f"- _No baseline available_ — of the previous {n_total} "
                f"weekly runs, {n_pre} pre-date telemetry (first captured: "
                f"{telemetry_first}) and {n_missing} post-launch run(s) "
                "had no captured parquet (capture-flag-off / partial SF / "
                "Lambda timeout — check the corresponding evaluator emails).",
            )
        else:
            lines.append(
                f"- _No baseline available_ (no parquets found for the "
                f"previous {n_total} weekly runs). Anomaly check "
                "deferred until baseline accumulates.",
            )
    else:
        current = anomaly.get("current_total_usd", 0.0)
        baseline = anomaly.get("baseline_mean_usd", 0.0) or 0.0
        ratio = anomaly.get("ratio")
        threshold = anomaly.get("threshold_ratio", 0.0)
        n_found = len(anomaly.get("baseline_dates_found") or [])
        n_missing = len(anomaly.get("baseline_dates_missing") or [])
        n_pre = len(anomaly.get("baseline_dates_pre_telemetry") or [])
        telemetry_first = anomaly.get("telemetry_first_date")
        ratio_str = f"{ratio:.2f}x" if ratio is not None else "N/A"
        lines.append(f"- Current run total: ${current:.4f}")
        lines.append(
            f"- Baseline (mean of {n_found} prior weekly runs): ${baseline:.4f}",
        )
        lines.append(
            f"- Ratio: **{ratio_str}** (threshold: {threshold:.2f}x)",
        )
        if n_pre:
            lines.append(
                f"- Baseline window: {n_pre} of {n_found + n_pre + n_missing} "
                f"prior weekly runs pre-date telemetry (first captured: "
                f"{telemetry_first}); excluded from baseline.",
            )
        if n_missing:
            lines.append(
                f"- Baseline gaps: {n_missing} of "
                f"{n_found + n_missing} post-launch prior weekly runs "
                "had no captured parquet (capture-flag-off / partial SF / "
                "Lambda timeout — check the corresponding evaluator emails).",
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

    fallback_date: Optional[str] = None
    if df is None:
        # Fall back to the most-recent prior `_cost/{date}/cost.parquet`.
        # Matches the same-shape "most-recent SF" pattern used by
        # decision_capture_coverage and provenance_grounding — eval-only
        # mid-week runs (e.g. 2026-05-07's evaluator triggered without
        # Research re-running) have no aggregated cost parquet for
        # today's date, but the operator still wants to see last
        # Saturday's cost data labeled as such, not "no data available".
        fallback_date = _find_most_recent_cost_date(
            run_date, bucket=bucket, s3_client=client,
        )
        if fallback_date is not None:
            df = fetch_cost_parquet(fallback_date, bucket=bucket, s3_client=client)
        if df is None:
            return _empty_section(run_date, reason="parquet not found at expected key")

    cost_label_date = fallback_date or run_date
    main_md = render_cost_report_markdown(df, run_date=cost_label_date)
    if fallback_date is not None:
        main_md = main_md.replace(
            "## LLM cost report",
            f"## LLM cost report\n\n_Showing most-recent captured run "
            f"(`{fallback_date}`) — today's run_date `{run_date}` had no "
            f"aggregated parquet, typically because Research didn't fire "
            f"this run. Anomaly check below is also against this date._",
        )
    current_total = float(df["cost_usd"].fillna(0).sum()) if "cost_usd" in df.columns else 0.0
    anomaly = detect_anomaly(
        cost_label_date, current_total, bucket=bucket, s3_client=client,
    )
    status = anomaly.get("status")
    if status == "anomaly":
        _emit_changelog_anomaly_entry(
            anomaly, run_date=cost_label_date, bucket=bucket, s3_client=client,
        )
        _publish_anomaly_alert(
            anomaly, run_date=cost_label_date,
            bucket=bucket, s3_client=client,
        )
    elif status == "ok":
        # config#867: the anomaly ledger only ever carries a genuine
        # observation (status="ok" means the baseline comparison actually
        # ran and cleared) — "no_baseline"/"alerting_disabled" are non-
        # observations and must not be treated as a recovery signal.
        _maybe_emit_changelog_recovery_entry(
            run_date=cost_label_date, bucket=bucket, s3_client=client,
        )
    return main_md + "\n" + render_anomaly_section(anomaly)


def _find_most_recent_cost_date(
    run_date: str,
    *,
    bucket: str,
    s3_client: Any,
    max_lookback_days: int = 14,
) -> Optional[str]:
    """List `_cost/` prefix and return the most recent ISO date ≤ run_date
    that has a `cost.parquet` object, within `max_lookback_days`.

    Returns None when no parquet is found in the window — the caller
    falls back to the empty-section placeholder.
    """
    base = date_type.fromisoformat(run_date)
    try:
        resp = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix="decision_artifacts/_cost/",
            Delimiter="/",
        )
    except ClientError as exc:
        logger.warning(
            "[cost_report] _cost/ listing failed; cannot fall back to "
            "most-recent date: %s", exc,
        )
        return None
    candidates: list[str] = []
    for cp in resp.get("CommonPrefixes") or []:
        prefix = cp.get("Prefix", "")
        # Format: "decision_artifacts/_cost/{YYYY-MM-DD}/"
        try:
            iso = prefix.rstrip("/").rsplit("/", 1)[-1]
            d = date_type.fromisoformat(iso)
        except ValueError:
            continue
        if d > base:
            continue
        if (base - d).days > max_lookback_days:
            continue
        candidates.append(iso)
    return max(candidates) if candidates else None


def _emit_changelog_anomaly_entry(
    anomaly: dict,
    *,
    run_date: str,
    bucket: str,
    s3_client: Any,
) -> Optional[str]:
    """Auto-populate the system-wide changelog with a cost-anomaly incident.

    Closes ROADMAP P0 sub-item 5 (cost-anomaly half — Item 2 cost-telemetry
    upstream is closed, so this hook is unblocked). Writes one structured
    schema-1.0.0 entry to ``s3://{bucket}/changelog/entries/{date}/{event_id}.json``
    so the corpus aggregator + retro-candidate filter see cost spikes
    alongside SNS-mirror + cloudwatch-mirror entries.

    Severity is **medium** — cost spikes are operational, not capital-at-risk.
    The retro filter requires severity ∈ {high, critical} so these entries
    don't pollute the candidate stream until an operator escalates via a
    follow-up ``changelog-log`` entry.

    Best-effort: any S3 write failure is logged at WARN and swallowed.
    Cost-section rendering must not be blocked by changelog corruption —
    the alert is also rendered into the email + WARN'd in the log, so
    no signal is lost.

    Returns the structured S3 key on success, or None on failure / no-op.
    """
    try:
        ts = datetime.now(timezone.utc)
        ts_utc = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        actor = "alpha-engine-cost-telemetry"
        ratio = anomaly.get("ratio")
        baseline_mean = anomaly.get("baseline_mean_usd")
        current_total = anomaly.get("current_total_usd")
        threshold = anomaly.get("threshold_ratio")

        # event_id + entry_date are STABLE on (run_date, current_total, ratio)
        # so re-running build_cost_section on the same anomaly overwrites
        # the same S3 key instead of writing N copies under wall-clock-
        # stamped names. Pre-2026-05-22 the event_id embedded ``ts_id``
        # (UTC wall-clock to the second) and entry_date used UTC TODAY,
        # so each evaluate.py re-run wrote a NEW changelog entry despite
        # the docstring claiming idempotent re-runs. The fix anchors both
        # to ``run_date`` (the anomaly's logical date) — see ROADMAP P1
        # entry filed 2026-05-22.
        entry_date = run_date
        digest_input = f"{run_date}|{current_total}|{ratio}".encode()
        event_hash = sha1(digest_input).hexdigest()[:7]
        event_id = f"cost_anomaly_{run_date}_{actor}_{event_hash}"

        ratio_str = f"{ratio:.2f}x" if ratio is not None else "n/a"
        baseline_str = f"${baseline_mean:.4f}" if baseline_mean is not None else "n/a"
        current_str = f"${current_total:.4f}" if current_total is not None else "n/a"
        threshold_str = f"{threshold:.2f}x" if threshold is not None else "n/a"
        baseline_dates = anomaly.get("baseline_dates_found") or []
        summary = (
            f"LLM cost anomaly: {ratio_str} of "
            f"{len(baseline_dates)}-week baseline "
            f"(current={current_str}, baseline_mean={baseline_str}, "
            f"threshold={threshold_str})"
        )[:240]
        description = (
            f"Run date: {run_date}\n"
            f"Current total cost: {current_str}\n"
            f"Baseline mean: {baseline_str} "
            f"(over {len(baseline_dates)} prior weeks: "
            f"{', '.join(baseline_dates) if baseline_dates else 'none'})\n"
            f"Ratio: {ratio_str} (threshold: {threshold_str})\n"
            f"Detected by: alpha-engine-backtester analysis/cost_report.py::detect_anomaly\n"
            f"Notification surface: evaluator email '## Anomaly detection' section + WARN log."
        )

        entry = {
            "schema_version": _CHANGELOG_SCHEMA_VERSION,
            "event_id": event_id,
            "ts_utc": ts_utc,
            "event_type": "incident",
            "severity": "medium",
            "subsystem": "telemetry",
            "root_cause_category": "prompt_regression",  # most plausible default; operator overrides via follow-up
            "resolution_type": None,
            "started_at": None,
            "detected_at": ts_utc,
            "resolved_at": None,
            "verified_at": None,
            "summary": summary,
            "description": description,
            "resolution_notes": None,
            "actor": actor,
            "machine": "backtester:analysis/cost_report.py",
            "source": "cost-anomaly-autoemit",
            "auto_emitted": True,
            "git_refs": [],
            "prompt_version": None,
            "run_id": run_date,
            "eval_run_ref": None,
            "cost_anomaly": {
                "ratio": ratio,
                "threshold_ratio": threshold,
                "current_total_usd": current_total,
                "baseline_mean_usd": baseline_mean,
                "baseline_dates_found": baseline_dates,
                "baseline_dates_missing": anomaly.get("baseline_dates_missing") or [],
            },
        }
        key = f"{_CHANGELOG_PREFIX}/{entry_date}/{event_id}.json"
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(entry).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(
            "[cost_report] changelog auto-emit: s3://%s/%s ratio=%s threshold=%s",
            bucket, key, ratio_str, threshold_str,
        )
        # config#867: remember this incident's event_id so the next
        # status="ok" run can emit a paired `recovery` entry instead of
        # leaving the corpus "newest event wins" forever. Best-effort —
        # a ledger write failure only costs the recovery pairing, never
        # the incident entry itself (already durably written above).
        _write_anomaly_ledger(
            {"open_event_id": event_id, "opened_run_date": run_date},
            bucket=bucket, s3_client=s3_client,
        )
        return key
    except Exception as e:
        logger.warning(
            "[cost_report] changelog auto-emit failed (best-effort, swallowed): %s",
            e,
        )
        return None


# config#867: stateful incident↔recovery pairing (Brian's 2026-07-08 ruling
# — "keep P3, clear gate ... auto-emit Lambdas track fired event_ids in S3,
# emit recovery on clear"). This is the only real auto-emit producer in the
# org today that fires an `incident` outside the SNS/CloudWatch mirrors
# (verified 2026-07-09 groom pass — "eval-regression" has no producer yet;
# the cost-anomaly emitter here is the sole candidate). One key holds at
# most one open incident — matches the corpus's existing "newest event
# wins" semantics; this repo only ever has one anomaly condition (the
# rolling-baseline ratio) so there is never more than one to track.
_ANOMALY_LEDGER_KEY = "changelog/_state/cost_anomaly_ledger.json"


def _read_anomaly_ledger(*, bucket: str, s3_client: Any) -> Optional[dict]:
    """Return the ledger dict, or ``None`` if absent/corrupt/unreadable.

    Best-effort: any failure (no ledger yet, transient S3 error, corrupt
    JSON) is treated the same as "no open incident" — recovery pairing is
    a nicety, never allowed to block the cost-report render.
    """
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=_ANOMALY_LEDGER_KEY)
        body = obj["Body"].read()
        return json.loads(body)
    except ClientError:
        return None
    except Exception as e:
        logger.warning(
            "[cost_report] anomaly ledger read failed (best-effort, swallowed): %s",
            e,
        )
        return None


def _write_anomaly_ledger(payload: dict, *, bucket: str, s3_client: Any) -> None:
    """Overwrite the ledger. Best-effort — swallows + logs on failure."""
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=_ANOMALY_LEDGER_KEY,
            Body=json.dumps(payload).encode("utf-8"),
            ContentType="application/json",
        )
    except Exception as e:
        logger.warning(
            "[cost_report] anomaly ledger write failed (best-effort, swallowed): %s",
            e,
        )


def _maybe_emit_changelog_recovery_entry(
    *, run_date: str, bucket: str, s3_client: Any,
) -> Optional[str]:
    """If the ledger has an open incident, emit a paired `recovery` entry
    and clear it. No-op (returns ``None``) when there's nothing open.
    """
    ledger = _read_anomaly_ledger(bucket=bucket, s3_client=s3_client)
    open_event_id = (ledger or {}).get("open_event_id")
    if not open_event_id:
        return None
    key = _emit_changelog_recovery_entry(
        open_event_id, run_date=run_date, bucket=bucket, s3_client=s3_client,
    )
    # Clear the ledger regardless of the write outcome above — a swallowed
    # emit failure is already logged; re-trying every subsequent "ok" run
    # would just spam duplicate recovery entries once the S3 issue clears,
    # which is worse than the (rare) chance of missing one pairing.
    _write_anomaly_ledger(
        {"open_event_id": None, "closed_run_date": run_date},
        bucket=bucket, s3_client=s3_client,
    )
    return key


def _emit_changelog_recovery_entry(
    open_event_id: str, *, run_date: str, bucket: str, s3_client: Any,
) -> Optional[str]:
    """Write one schema-1.0.0 `recovery` entry back-referencing the
    original incident's ``event_id`` via ``git_refs`` (the field the
    SNS-mirror Lambda's own docstring already earmarks for this: "an
    `investigation` entry whose `git_refs` reference the original
    event_id"). ``severity="informational"`` + no ``root_cause_category``
    mirrors ``classify.py``'s ``non_incident()`` convention for the SNS
    OK-transition recovery path, so both recovery-emit paths in the org
    read identically to consumers.

    Best-effort: any S3 write failure is logged at WARN and swallowed —
    same failure posture as the incident emit this pairs with.
    """
    try:
        ts = datetime.now(timezone.utc)
        ts_utc = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        actor = "alpha-engine-cost-telemetry"
        event_hash = sha1(f"{run_date}|{open_event_id}".encode()).hexdigest()[:7]
        event_id = f"cost_recovery_{run_date}_{actor}_{event_hash}"
        entry = {
            "schema_version": _CHANGELOG_SCHEMA_VERSION,
            "event_id": event_id,
            "ts_utc": ts_utc,
            "event_type": "recovery",
            "severity": "informational",
            "subsystem": "telemetry",
            "root_cause_category": None,
            "resolution_type": None,
            "started_at": None,
            "detected_at": ts_utc,
            "resolved_at": ts_utc,
            "verified_at": None,
            "summary": f"LLM cost back within baseline (recovers {open_event_id})",
            "description": (
                f"Run date: {run_date}\n"
                f"Cost anomaly {open_event_id} cleared — this run's total "
                f"is back within the rolling-baseline threshold.\n"
                f"Detected by: alpha-engine-backtester analysis/cost_report.py::detect_anomaly"
            ),
            "resolution_notes": None,
            "actor": actor,
            "machine": "backtester:analysis/cost_report.py",
            "source": "cost-anomaly-autoemit",
            "auto_emitted": True,
            "git_refs": [open_event_id],
            "prompt_version": None,
            "run_id": run_date,
            "eval_run_ref": None,
        }
        key = f"{_CHANGELOG_PREFIX}/{run_date}/{event_id}.json"
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(entry).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(
            "[cost_report] changelog recovery auto-emit: s3://%s/%s resolves=%s",
            bucket, key, open_event_id,
        )
        return key
    except Exception as e:
        logger.warning(
            "[cost_report] changelog recovery auto-emit failed (best-effort, swallowed): %s",
            e,
        )
        return None


def _anomaly_alert_dedup_key(run_date: str, anomaly: dict) -> str:
    """Deterministic dedup key string for a single cost-anomaly event.

    Hashes ``(run_date, current_total, baseline_mean, ratio)`` — the
    same anomaly observed twice produces the same key. Excludes wall-
    clock so re-invocations across the same Saturday SF / smoke / spot
    re-run / mid-week evaluate.py round-trips all collapse to one
    publish.

    Returns a string consumed by ``nousergon_lib.alerts.publish``'s
    ``dedup_key`` parameter — the lib hashes it internally and writes
    its marker at ``s3://{bucket}/_alerts/_dedup/{sha1(...)[:16]}.json``.
    """
    current_total = anomaly.get("current_total_usd")
    baseline_mean = anomaly.get("baseline_mean_usd")
    ratio = anomaly.get("ratio")
    # Format with fixed precision so float-noise doesn't shift the hash.
    return (
        f"cost_anomaly|{run_date}|"
        f"current={current_total:.6f}|"
        f"baseline={baseline_mean:.6f}|"
        f"ratio={ratio:.6f}"
    )


def _publish_anomaly_alert(
    anomaly: dict,
    *,
    run_date: str,
    bucket: str = _DEFAULT_BUCKET,
    s3_client: Optional[Any] = None,
) -> None:
    """Fan an anomaly out to the operator-facing surveillance channels
    via ``nousergon_lib.alerts.publish`` (SNS → email + Telegram).

    Best-effort: any import failure or network exception is logged at
    WARN and swallowed. The WARN log + the rendered email anomaly
    section + the changelog auto-emit all remain in place, so no
    signal is lost when the SNS+Telegram fan-out misfires.

    **Idempotent across invocations** via lib v0.24.0's ``dedup_key``
    primitive — same anomaly across the Saturday-SF main run, smoke-mode
    re-runs, and mid-week ``evaluate.py`` round-trips all collapse to
    one publish. ``dedup_window_min=None`` matches the pre-lift local
    helpers' "marker once, never re-publish" semantics; on lib upgrade
    paths the marker lives at
    ``s3://{ALPHA_ENGINE_ALERTS_DEDUP_BUCKET or bucket}/_alerts/_dedup/{sha1(dedup_key)[:16]}.json``.

    Opt-out via ``ALPHA_ENGINE_COST_ANOMALY_ALERT_DISABLED=1``
    (tests use this to keep moto-only paths from reaching real boto3).
    """
    if os.environ.get(_ANOMALY_ALERT_DISABLED_ENV_VAR, "").strip().lower() in (
        "1", "true", "yes", "on",
    ):
        return
    try:
        from ops_alerts import publish_ops_alert
    except ImportError as e:
        logger.warning(
            "[cost_report] alerts publish skipped — ops_alerts unavailable: %s", e,
        )
        return

    ratio = anomaly.get("ratio")
    baseline_mean = anomaly.get("baseline_mean_usd")
    current_total = anomaly.get("current_total_usd")
    threshold = anomaly.get("threshold_ratio")
    baseline_dates = anomaly.get("baseline_dates_found") or []
    ratio_str = f"{ratio:.2f}x" if ratio is not None else "n/a"
    baseline_str = f"${baseline_mean:.4f}" if baseline_mean is not None else "n/a"
    current_str = f"${current_total:.4f}" if current_total is not None else "n/a"
    threshold_str = f"{threshold:.2f}x" if threshold is not None else "n/a"
    message = (
        f"LLM cost anomaly on {run_date}: {ratio_str} of "
        f"{len(baseline_dates)}-week baseline "
        f"(current={current_str}, baseline_mean={baseline_str}, "
        f"threshold={threshold_str}). "
        f"See evaluator email '## LLM cost report' section + "
        f"changelog/entries/{run_date}/."
    )
    dedup_key = _anomaly_alert_dedup_key(run_date, anomaly)
    try:
        result = publish_ops_alert(
            message,
            severity="error",
            source="alpha-engine-backtester/analysis/cost_report.py",
            dedup_key=dedup_key,
            dedup_window_min=None,  # forever — matches pre-lift "marker once" semantics
        )
        if result.dedup_skipped:
            logger.info(
                "[cost_report] anomaly alert deduped (run_date=%s, "
                "reason=%s) — skipping duplicate publish",
                run_date, result.dedup_reason,
            )
        else:
            logger.info(
                "[cost_report] anomaly alert publish: sns_ok=%s any_ok=%s",
                result.sns.ok, result.any_ok,
            )
    except Exception as e:
        logger.warning(
            "[cost_report] anomaly alert publish failed (best-effort, swallowed): %s",
            e,
        )
