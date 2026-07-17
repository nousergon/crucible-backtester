#!/usr/bin/env python3
"""One-shot operator bootstrap promotion (config#2364 / config#2367).

Brian's ruling (2026-07-11, ratified in-session): the deployed agentic
producer path measured -1.36% (21d) vs +0.65% for the research-free
scanner->predictor path (config#1405 retrospective, 11 overlapping cycles,
count-matched). Rather than wait for the first gate-eligible forward
promotion (no matured forward cohort exists before ~2026-08-03), Brian
decreed an explicit operator-bootstrap promotion to scanner_predictor_direct
now, recorded as promotion_source="operator_bootstrap" in both the pointer
and its audit record. This script is designed to run EXACTLY ONCE: every
promotion/demotion decision after this one is made by the gate engine
(optimizer.champion_promotion.run_weekly_evaluation, wired into evaluate.py).

Calls the module's single writer functions (write_champion_pointer /
write_champion_audit) directly -- never constructs the S3 objects by hand --
so this bootstrap event is indistinguishable, to every downstream consumer,
from a gate-engine promotion except for the promotion_source field.

Usage:
    python3 bootstrap_champion_promotion.py --run-date 2026-07-13 --upload
    python3 bootstrap_champion_promotion.py --run-date 2026-07-13   # dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta

from optimizer.champion_promotion import (
    VALID_CHAMPIONS,
    read_champion_pointer,
    write_champion_audit,
    write_champion_pointer,
)

_COOLDOWN_WEEKS = 2  # mirrors optimizer.champion_promotion._COOLDOWN_WEEKS (gate e)


def _build_bootstrap_audit(run_date: str, champion_before: str, champion_after: str) -> dict:
    """Shaped to satisfy contracts/producer_champion_audit.schema.json (v1).
    ``promotion_source`` is not part of that frozen schema but the schema's
    additive-only contract requires consumers to tolerate unknown extra
    fields, so it rides along here for audit-trail clarity (mirrors the
    pointer schema, which does define promotion_source natively)."""
    cooldown_until = (date.fromisoformat(run_date) + timedelta(weeks=_COOLDOWN_WEEKS)).isoformat()
    return {
        "schema_version": 1,
        "date": run_date,
        "outcome": "promoted",
        "champion_before": champion_before,
        "champion_after": champion_after,
        "challenger_matured_cohorts": 0,
        "sn_lift_vs_champion": None,
        "consecutive_wins": 0,
        "cooldown_until": cooldown_until,
        "blocked_by": None,
        "challenger": champion_after,
        "freeze": False,
        "promotion_source": "operator_bootstrap",
        "detail": (
            "Operator-decreed bootstrap promotion (Brian ruling 2026-07-11, "
            "config#2364/#2367) -- no forward-cohort gate was evaluated "
            "(none existed pre-bootstrap). config#1405 retrospective basis: "
            "research-free predictor top-N +0.65% vs live agentic CIO "
            "ADVANCE -1.36% (21d, 11 overlapping cycles, count-matched). "
            "The gate engine governs every promotion/demotion from here on."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bucket", default="alpha-engine-research")
    parser.add_argument(
        "--champion", default="scanner_predictor_direct", choices=VALID_CHAMPIONS,
        help="Target champion arm to bootstrap-promote to (default: scanner_predictor_direct, per Brian's ruling)",
    )
    parser.add_argument(
        "--run-date", required=True,
        help="Trading-day date (YYYY-MM-DD) this bootstrap event is filed under -- the audit record's 'date' field.",
    )
    parser.add_argument("--upload", action="store_true", help="Actually write to S3 (default: dry-run, log only)")
    args = parser.parse_args()
    # Note: "agentic" cannot reach this point -- argparse's own `choices=
    # VALID_CHAMPIONS` (config-I2518 seat swap: VALID_CHAMPIONS no longer
    # includes "agentic") rejects it at parse time with SystemExit(2) before
    # main()'s body runs. No separate refusal check is needed here.

    existing = read_champion_pointer(args.bucket)
    champion_before = (existing or {}).get("champion", "scanner_predictor_direct")
    if existing is not None:
        print(
            f"REFUSING: a champion pointer already exists (champion={champion_before!r}, "
            f"promotion_source={existing.get('promotion_source')!r}). This script is a "
            "ONE-SHOT bootstrap for the pre-bootstrap state only -- any promotion from "
            "here on must go through the gate engine (run_weekly_evaluation), never this "
            "script again.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Bootstrap promotion: champion_before={champion_before!r} -> champion_after={args.champion!r}")

    pointer = write_champion_pointer(
        bucket=args.bucket,
        champion=args.champion,
        promotion_source="operator_bootstrap",
        upload=args.upload,
    )
    print(f"Pointer {'WRITTEN' if args.upload else '(dry-run, NOT written)'}: {json.dumps(pointer, indent=2)}")

    audit = _build_bootstrap_audit(args.run_date, champion_before, args.champion)
    if args.upload:
        write_champion_audit(args.bucket, args.run_date, audit)
        print(f"Audit record WRITTEN for {args.run_date}")
    else:
        print(f"Audit record (dry-run, NOT written) for {args.run_date}:\n{json.dumps(audit, indent=2)}")


if __name__ == "__main__":
    main()
