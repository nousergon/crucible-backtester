"""CLI entry point for the replay harness.

PR A scope: single-artifact replay. Batch + date-range modes land in
PR C.

Usage:

    python -m replay.cli \
        --artifact-key decision_artifacts/2026/05/03/sector_quant:technology/2026-05-01.json \
        --target-model claude-haiku-4-5 \
        [--bucket alpha-engine-research] \
        [--replay-prefix decision_artifacts/_replay] \
        [--max-tokens 8192] \
        [--no-persist]

Single replay against an artifact + a target model. Output prints a
JSON summary to stdout (suitable for piping to ``jq``); the full
replay artifact lands at S3 ``{replay_prefix}/{run_id}/{orig}_vs_{target}.json``
unless ``--no-persist`` is set.

Cost note: every replay invocation costs target-model tokens. Use
``--no-persist`` for smoke tests; persistent replays should be batched
through PR C's date-range mode for cost-aggregation discipline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from replay.runner import (
    DEFAULT_BUCKET,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REPLAY_PREFIX,
    replay_artifact,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="replay",
        description="Replay a captured DecisionArtifact under a different model.",
    )
    parser.add_argument(
        "--artifact-key",
        required=True,
        help="S3 key of the captured DecisionArtifact (e.g. decision_artifacts/2026/05/03/sector_quant:technology/run-1.json)",
    )
    parser.add_argument(
        "--target-model",
        required=True,
        help="Model identifier to replay under (e.g. claude-haiku-4-5)",
    )
    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help=f"S3 bucket (default: {DEFAULT_BUCKET})",
    )
    parser.add_argument(
        "--replay-prefix",
        default=DEFAULT_REPLAY_PREFIX,
        help=f"S3 prefix for replay output (default: {DEFAULT_REPLAY_PREFIX})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"max_tokens for target invocation (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Compute replay but skip S3 persist — useful for smoke tests.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

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
        "latency_ms": replay.replay_latency_ms,
        "error": replay.replay_error,
        "tokens": replay.replay_cost,
    }
    print(json.dumps(summary, indent=2))

    return 1 if replay.replay_error else 0


if __name__ == "__main__":  # pragma: no cover — CLI invocation only
    sys.exit(main())
