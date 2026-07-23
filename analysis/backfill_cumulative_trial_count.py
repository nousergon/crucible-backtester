"""analysis/backfill_cumulative_trial_count.py — one-time seed of the
cumulative multiple-testing trial-count counter (config#2454).

The DSR (Deflated Sharpe Ratio) metric in crucible-evaluator's
``grading/tiles/portfolio_outcome.py`` needs ``n_trials``: the cumulative
count of strategy configurations trialed, since inception, across ALL 4
backtester sweep producers — not just the count since the shared
``nousergon_lib.quant.stats.trial_accumulator`` counter was introduced.
Starting that counter at 0 would understate the true historical trial
count (every cycle those 4 producers ran BEFORE this feature shipped
generated real trials that DSR's multiple-testing correction should
account for).

This script recovers that history: it scans every dated
``backtest/{run_date}/{producer}.json`` archive already in S3 for each of
the 4 producers, sums each producer's ``n_trials`` field (falling back to
``len(cells)`` for archives written before the n_trials field itself was
added — see config#2454's producer-1..3 wiring in ``backtest.py``), and
seeds the shared counter via
``nousergon_lib.quant.stats.trial_accumulator.backfill_cumulative_trial_count``.

Run EXACTLY ONCE, before any of the 4 producers' live increments start
landing (the accumulator refuses a non-zero-total backfill unless
``--overwrite`` is passed — see its docstring for why running this twice
would double-count).

Usage (from a box with real AWS creds — this cannot run in an unattended-
groom sandbox with no S3 access to the real ``alpha-engine-research``
archive):

    python -m analysis.backfill_cumulative_trial_count \\
        --bucket alpha-engine-research

    # Dry run first (prints per-producer sums, does not write):
    python -m analysis.backfill_cumulative_trial_count --dry-run

predictor_param_sweep is intentionally excluded from the historical sum:
its trial counts were never persisted as a durable ``n_trials`` field
before config#2454 (the vectorized sweep's ``len(combinations)`` was
computed in-memory and only the resulting ``sweep_df`` parquet — not a
count — was archived under the per-phase artifact path, not a dated
top-level JSON this scanner can enumerate the same way). Its contribution
to the cumulative total starts accruing live from the first post-#2454
cycle instead of being backfilled; this undercounts (rather than
overcounts) history, which is the conservative direction for DSR's
multiple-testing correction (a lower n_trials makes DSR's threshold LESS
strict, so the gap errs toward not over-crediting selection-bias
correction the historical trials didn't actually document).
"""

from __future__ import annotations

import argparse
import json
import logging

import boto3

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "alpha-engine-research"

# producer name -> dated-artifact filename under backtest/{run_date}/
_PRODUCERS = {
    "optimizer_param_sweep": "optimizer_param_sweep.json",
    "gamma_sweep": "gamma_sweep.json",
    "cov_estimator_sweep": "cov_sweep.json",
}


def _n_trials_from_archive(doc: dict) -> int:
    """Prefer the persisted ``n_trials`` field (config#2454); fall back to
    ``len(cells)`` for archives written before that field existed."""
    if isinstance(doc.get("n_trials"), int):
        return doc["n_trials"]
    cells = doc.get("cells")
    if isinstance(cells, dict):
        return len(cells)
    return 0


def sum_historical_trials(
    bucket: str, producer: str, filename: str, *, s3_client=None,
) -> tuple[int, int, int]:
    """Scan every ``backtest/{run_date}/{filename}`` archive and sum
    n_trials. Returns ``(total_trials, n_archives_read, n_archives_skipped)``.

    Skips (rather than fails on) archives with ``status != "ok"`` (the
    producer's own skip path — e.g. "predictor backtest status=error" or
    "insufficient sigma_alpha coverage" — persists a payload with no
    ``cells``/``n_trials``, correctly contributing 0) and any archive that
    fails to parse as JSON (corrupt/partial write) — logged loudly, not
    silently dropped.
    """
    s3 = s3_client or boto3.client("s3")
    total = 0
    n_read = 0
    n_skipped = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix="backtest/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(f"/{filename}"):
                continue
            try:
                resp = s3.get_object(Bucket=bucket, Key=key)
                doc = json.loads(resp["Body"].read())
            except Exception as exc:  # noqa: BLE001 — one bad archive shouldn't abort the scan
                logger.warning(
                    "backfill_cumulative_trial_count: failed to read/parse "
                    "s3://%s/%s (skipping): %s", bucket, key, exc,
                )
                n_skipped += 1
                continue
            if doc.get("status") != "ok":
                n_skipped += 1
                continue
            total += _n_trials_from_archive(doc)
            n_read += 1
    logger.info(
        "backfill_cumulative_trial_count: producer=%s scanned %d ok "
        "archive(s) (%d skipped) → %d historical trials",
        producer, n_read, n_skipped, total,
    )
    return total, n_read, n_skipped


def main(argv: list[str] | None = None) -> int:
    from nousergon_lib.quant.stats.trial_accumulator import (
        backfill_cumulative_trial_count,
        read_cumulative_trial_count,
    )

    parser = argparse.ArgumentParser(
        description=(
            "One-time backfill of the cumulative multiple-testing "
            "trial-count counter (config#2454) from historical dated "
            "sweep archives."
        ),
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument(
        "--run-date", default=None,
        help="last_updated stamp for the seeded artifact (default: today, UTC).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print per-producer sums; do not write the counter artifact.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help=(
            "Force-reseed even if the counter already has a non-zero "
            "total. DANGEROUS if live increments have already landed — "
            "see the accumulator's backfill docstring."
        ),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    s3 = boto3.client("s3")
    run_date = args.run_date
    if run_date is None:
        from datetime import datetime, timezone
        run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    per_producer_totals: dict[str, int] = {}
    for producer, filename in _PRODUCERS.items():
        total, n_read, n_skipped = sum_historical_trials(
            args.bucket, producer, filename, s3_client=s3,
        )
        per_producer_totals[producer] = total

    grand_total = sum(per_producer_totals.values())
    print(json.dumps(
        {"bucket": args.bucket, "run_date": run_date,
         "by_producer": per_producer_totals, "total": grand_total,
         "note": "predictor_param_sweep excluded from backfill — see module docstring"},
        indent=2,
    ))

    if args.dry_run:
        print("--dry-run: not writing the counter artifact.")
        return 0

    existing = read_cumulative_trial_count(args.bucket, s3_client=s3)
    if existing.get("total") and not args.overwrite:
        print(
            f"Refusing to seed: s3://{args.bucket}/backtest/"
            f"cumulative_trial_count.json already has total="
            f"{existing['total']}. Pass --overwrite to force (see the "
            f"accumulator's backfill docstring for why this is dangerous "
            f"if live increments have already landed).",
        )
        return 1

    result = backfill_cumulative_trial_count(
        per_producer_totals, run_date,
        bucket=args.bucket, s3_client=s3, overwrite=args.overwrite,
    )
    print(f"Seeded s3://{args.bucket}/backtest/cumulative_trial_count.json: {result}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
