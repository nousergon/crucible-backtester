#!/usr/bin/env python3
"""One-shot, idempotent backfill of config/optimizer_risk_history/ from the
existing dated backtest sweep verdicts.

The Optimizer-Risk dashboard page reads config/optimizer_risk_history/. New
records accrue going forward from each backtester run, but the history would
otherwise start empty. This script reconstructs a record per prior trading day
from the already-written backtest/{day}/cov_sweep.json (+ gamma_sweep.json +
predictor/optimizer_gate/{day}.json) using the SAME assembler the live producer
uses, so seeded rows are identical in shape to forward rows.

Idempotency: backfilled records key on a trading-day-derived run_id (YYYYMMDD,
8 digits) — distinct from live wall-clock run_ids (YYMMDDHHMM, 10 digits) — so
re-running overwrites the seed rows rather than duplicating them, and never
clobbers a live run's record for the same day.

Usage:
    python scripts/backfill_optimizer_risk_history.py [--bucket B] \
        [--executor-path /path/to/alpha-engine] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3

from optimizer.optimizer_risk_history import (
    build_optimizer_risk_record,
    write_optimizer_risk_history,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("backfill_optimizer_risk_history")

# Documented mirror of the executor's OPTIMIZER_CONFIG_DEFAULTS static levers,
# used only if the executor package is not importable. These are slow-changing
# constants; a WARN is logged when the fallback is used so the seed is auditable.
_FALLBACK_DEFAULTS = {
    "vol_target_annual": None,
    "risk_aversion": 5.0,
    "tcost_bps": 5.0,
    "cash_sleeve_pct": 0.03,
    "max_sector_pct": 0.25,
    "max_daily_turnover": 0.20,
    "alpha_uncertainty_penalty": 0.0,
}


def _load_defaults(executor_path: str | None) -> dict:
    if executor_path and os.path.isdir(executor_path) and executor_path not in sys.path:
        sys.path.insert(0, executor_path)
    try:
        from executor.portfolio_optimizer import OPTIMIZER_CONFIG_DEFAULTS
        logger.info("using executor OPTIMIZER_CONFIG_DEFAULTS for static levers")
        return dict(OPTIMIZER_CONFIG_DEFAULTS)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "executor OPTIMIZER_CONFIG_DEFAULTS unavailable (%s) — using "
            "documented fallback mirror for static levers (pass --executor-path "
            "to source the real values)", exc,
        )
        return dict(_FALLBACK_DEFAULTS)


def _get_json(s3, bucket: str, key: str) -> dict | None:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except Exception:
        return None
    try:
        return json.loads(obj["Body"].read())
    except Exception as exc:  # noqa: BLE001
        logger.warning("parse failed for s3://%s/%s: %s", bucket, key, exc)
        return None


def _list_cov_sweep_dates(s3, bucket: str) -> list[str]:
    """Return trading-day folders under backtest/ that have a cov_sweep.json."""
    dates: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix="backtest/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/cov_sweep.json"):
                parts = key.split("/")
                if len(parts) == 3:  # backtest/{date}/cov_sweep.json
                    dates.append(parts[1])
    return sorted(set(dates))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="alpha-engine-research")
    ap.add_argument("--executor-path", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    s3 = boto3.client("s3")
    defaults = _load_defaults(args.executor_path)
    dates = _list_cov_sweep_dates(s3, args.bucket)
    logger.info("found %d trading day(s) with cov_sweep.json", len(dates))

    seeded = skipped = 0
    for day in dates:
        cov = _get_json(s3, args.bucket, f"backtest/{day}/cov_sweep.json")
        gamma = _get_json(s3, args.bucket, f"backtest/{day}/gamma_sweep.json")
        gate = (
            _get_json(s3, args.bucket, f"predictor/optimizer_gate/{day}.json")
            or _get_json(s3, args.bucket, f"predictor/optimizer_gate/production/{day}.json")
        )
        run_id = day.replace("-", "")  # YYYYMMDD — deterministic, idempotent
        record = build_optimizer_risk_record(
            cov_payload=cov,
            gamma_payload=gamma,
            gate_payload=gate,
            optimizer_defaults=defaults,
            trading_day=day,
            updated_at=day,
            run_id=run_id,
        )
        if record is None:
            logger.warning("skip %s — no usable cov-sweep cells", day)
            skipped += 1
            continue
        if args.dry_run:
            logger.info(
                "[dry-run] would seed %s run_id=%s cov_selected=%s sortino=%s",
                day, run_id, record.get("cov_selected_name"),
                record.get("sortino_ratio"),
            )
            seeded += 1
            continue
        res = write_optimizer_risk_history(record, bucket=args.bucket, s3=s3)
        if res.get("written"):
            seeded += 1
        else:
            logger.warning("write failed for %s: %s", day, res.get("reason"))
            skipped += 1

    logger.info("backfill complete: seeded %d / skipped %d", seeded, skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
