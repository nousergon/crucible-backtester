"""CLI entry point for the replay harness.

Three subcommands:

  * ``single``         — replay one captured DecisionArtifact under a
    target model. Smoke testing + ad-hoc retrospective analysis.
  * ``batch``          — iterate over a date-range corpus × target
    models, emit per-(agent_id, target_model) concordance to
    CloudWatch, and persist a per-target-model batch summary.
    Production cadence runs this from the Saturday SF.
  * ``counterfactual`` — fit a depth-≤3 decision tree per agent on
    captured (input → decision) pairs; emit per-agent match rate to
    CloudWatch + persist tree structure to S3. Third leg of the
    agent-justification triple alongside cross-week clustering +
    cheap-model concordance.

Single examples:

    python -m replay.cli single \\
        --artifact-key decision_artifacts/2026/05/03/sector_quant:technology/2026-05-01.json \\
        --target-model deepseek/deepseek-v4-flash \\
        [--bucket alpha-engine-research] \\
        [--max-tokens 8192] \\
        [--no-persist]

Batch examples:

    # 8-week trailing window, production concordance target, all canonical agents
    python -m replay.cli batch \\
        --target-models deepseek/deepseek-v4-flash \\
        [--window-days 56] \\
        [--agents sector_quant,ic_cio]

    # Dry-run cost estimation (lists candidate artifacts, no LLM calls)
    python -m replay.cli batch \\
        --target-models deepseek/deepseek-v4-flash \\
        --dry-run

Cost note: every replay invocation costs target-model tokens. Use
``--no-persist`` (single) or ``--dry-run`` (batch) for smoke tests
without spend.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime

from replay.runner import (
    DEFAULT_BUCKET,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REPLAY_PREFIX,
    replay_artifact,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="replay",
        description=(
            "Replay captured DecisionArtifacts under a different model. "
            "Single-artifact mode for smoke tests; batch mode for "
            "concordance metrics."
        ),
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging.",
    )

    sub = parser.add_subparsers(dest="mode", required=True)

    # ── single ────────────────────────────────────────────────────────
    single = sub.add_parser(
        "single",
        help="Replay one captured DecisionArtifact under a target model.",
    )
    single.add_argument(
        "--artifact-key",
        required=True,
        help="S3 key of the captured DecisionArtifact.",
    )
    single.add_argument(
        "--target-model",
        required=True,
        help="OpenRouter model id to replay under (e.g. deepseek/deepseek-v4-flash).",
    )
    single.add_argument(
        "--bucket", default=DEFAULT_BUCKET,
        help=f"S3 bucket (default: {DEFAULT_BUCKET}).",
    )
    single.add_argument(
        "--replay-prefix", default=DEFAULT_REPLAY_PREFIX,
        help=f"S3 prefix for replay output (default: {DEFAULT_REPLAY_PREFIX}).",
    )
    single.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"max_tokens for target invocation (default: {DEFAULT_MAX_TOKENS}).",
    )
    single.add_argument(
        "--no-persist", action="store_true",
        help="Compute replay but skip S3 persist — useful for smoke tests.",
    )

    # ── batch ─────────────────────────────────────────────────────────
    batch = sub.add_parser(
        "batch",
        help=(
            "Iterate over date-range corpus × target models; aggregate "
            "concordance per (agent_id, target_model); emit CloudWatch + "
            "persist per-target-model summary."
        ),
    )
    batch.add_argument(
        "--target-models", required=True,
        help=(
            "Comma-separated list of target model identifiers "
            "(e.g. 'deepseek/deepseek-v4-flash' or "
            "'deepseek/deepseek-v4-flash,deepseek/deepseek-v4-pro')."
        ),
    )
    batch.add_argument(
        "--window-days", type=int, default=56,
        help="Trailing-window length in days (default: 56 = 8 weeks).",
    )
    batch.add_argument(
        "--agents", default=None,
        help=(
            "Comma-separated list of agent_id_base values to include "
            "(default: all 6 canonical families)."
        ),
    )
    batch.add_argument(
        "--end-time-iso", default=None,
        help=(
            "Window end as ISO-8601 (e.g. 2026-05-09T00:00:00Z). "
            "Default: now (UTC)."
        ),
    )
    batch.add_argument(
        "--max-artifacts", type=int, default=500,
        help="Hard cap on artifacts replayed per run (default: 500).",
    )
    batch.add_argument(
        "--bucket", default=DEFAULT_BUCKET,
        help=f"S3 bucket (default: {DEFAULT_BUCKET}).",
    )
    batch.add_argument(
        "--dry-run", action="store_true",
        help=(
            "List candidate artifacts + skip LLM calls + persist nothing. "
            "Use for cost estimation."
        ),
    )
    batch.add_argument(
        "--no-emit-metrics", action="store_true",
        help="Skip CloudWatch metric emission (still persists summary JSON).",
    )

    # ── counterfactual ────────────────────────────────────────────────
    cf = sub.add_parser(
        "counterfactual",
        help=(
            "Fit a depth-≤3 decision tree per supported agent on "
            "captured (input → decision) pairs; emit per-agent match "
            "rate to CloudWatch + persist tree structure to S3. "
            "Third leg of the agent-justification triple."
        ),
    )
    cf.add_argument(
        "--window-days", type=int, default=56,
        help="Trailing-window length in days (default: 56 = 8 weeks).",
    )
    cf.add_argument(
        "--max-depth", type=int, default=3,
        help="Decision-tree max depth (default: 3 — \"3-deep rule\").",
    )
    cf.add_argument(
        "--agents", default=None,
        help=(
            "Comma-separated agent_id_base values to include. "
            "Default: all supported (ic_cio, macro_economist; v1)."
        ),
    )
    cf.add_argument(
        "--end-time-iso", default=None,
        help="Window end as ISO-8601. Default: now (UTC).",
    )
    cf.add_argument(
        "--bucket", default=DEFAULT_BUCKET,
        help=f"S3 bucket (default: {DEFAULT_BUCKET}).",
    )
    cf.add_argument(
        "--no-emit-metrics", action="store_true",
        help="Skip CloudWatch metric emission (still persists analysis JSON).",
    )

    return parser


def _run_single(args: argparse.Namespace) -> int:
    replay = replay_artifact(
        artifact_key=args.artifact_key,
        target_model=args.target_model,
        bucket=args.bucket,
        replay_prefix=args.replay_prefix,
        max_tokens=args.max_tokens,
        persist=not args.no_persist,
    )
    summary = {
        "agent_id": replay.original_agent_id,
        "original_model": replay.original_model,
        "replay_model": replay.replay_model,
        "kind": replay.replay_output_kind,
        "agreement_score": replay.comparison.get("agreement_score"),
        "diff_summary": replay.comparison.get("diff_summary"),
        "latency_ms": replay.replay_latency_ms,
        "error": replay.replay_error,
        "tokens": replay.replay_cost,
    }
    print(json.dumps(summary, indent=2))
    return 1 if replay.replay_error else 0


def _run_batch(args: argparse.Namespace) -> int:
    from replay.batch import compute_and_emit_concordance

    target_models = [m.strip() for m in args.target_models.split(",") if m.strip()]
    agent_filter = (
        [a.strip() for a in args.agents.split(",") if a.strip()]
        if args.agents else None
    )
    end_time = (
        datetime.fromisoformat(args.end_time_iso.replace("Z", "+00:00"))
        if args.end_time_iso else None
    )

    summary = compute_and_emit_concordance(
        target_models=target_models,
        end_time=end_time,
        window_days=args.window_days,
        agent_filter=agent_filter,
        bucket=args.bucket,
        max_artifacts=args.max_artifacts,
        emit_metrics=not args.no_emit_metrics,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.mode == "single":
        return _run_single(args)
    if args.mode == "batch":
        return _run_batch(args)
    if args.mode == "counterfactual":
        return _run_counterfactual(args)
    parser.error(f"unknown mode: {args.mode}")


def _run_counterfactual(args: argparse.Namespace) -> int:
    from replay.counterfactual import compute_and_emit

    agent_filter = (
        [a.strip() for a in args.agents.split(",") if a.strip()]
        if args.agents else None
    )
    end_time = (
        datetime.fromisoformat(args.end_time_iso.replace("Z", "+00:00"))
        if args.end_time_iso else None
    )

    summary = compute_and_emit(
        end_time=end_time,
        window_days=args.window_days,
        max_depth=args.max_depth,
        bucket=args.bucket,
        agent_filter=agent_filter,
        emit_metrics=not args.no_emit_metrics,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI invocation only
    sys.exit(main())
