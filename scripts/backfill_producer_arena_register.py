#!/usr/bin/env python3
"""Extend optimizer/arena/producer_register.json from the real leaderboard history.

The selection-producer slot's arm register is a DURABLE, append-only artifact
committed to this repo. ``created_date`` starts the four-week grace period, so
a value that could silently change between runs would make retirement
non-reproducible (champion-challenger-policy.md §6).

It is DERIVED, never typed. This script reads every
``research/producer_leaderboard/{date}.json`` and folds them onto the
COMMITTED register through
``optimizer.producer_arena.register_events_from_boards``. The fold only
appends: an arm already registered keeps its recorded ``created_date`` even
when S3 has since gained earlier boards or backfilled cohorts. New arms get a
date chosen by ``producer_arena.NEW_ARM_CREATED_DATE_RULE``.

The same run writes ``optimizer/arena/producer_board_snapshot.json``: each
board projected onto the three facts the fold reads. That is what ``--check``
folds in CI, which has no S3 access.

    python scripts/backfill_producer_arena_register.py --bucket alpha-engine-research
    python scripts/backfill_producer_arena_register.py --from-dir ./boards
    python scripts/backfill_producer_arena_register.py --from-snapshot --base-ref origin/main
    python scripts/backfill_producer_arena_register.py --check --base-ref origin/main

``--base-ref REF`` folds onto the register as it is at that git ref instead of
the working copy. Use it to regenerate the arms a branch appends, for example
after changing ``NEW_ARM_CREATED_DATE_RULE``: arms already registered at the
ref keep their dates, arms the branch adds are dated under the new rule.

``--check`` writes nothing. It exits 1 if the committed register is not
exactly the fold of the boards (``--from-dir``/``--bucket``, else the
committed snapshot) onto the register at ``--base-ref`` (else onto itself).
That catches three things: a register that has fallen behind its boards; an
appended event that the fold would not produce, including one dated under a
rule other than ``NEW_ARM_CREATED_DATE_RULE``; and, with ``--base-ref``, any
event at the ref that was removed, reordered or rewritten
(alpha-engine-config-I11490).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizer import producer_arena  # noqa: E402

_KEY_RE = re.compile(r"^research/producer_leaderboard/(\d{4}-\d{2}-\d{2})\.json$")
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _boards_from_s3(bucket: str) -> list[dict]:
    import boto3

    s3 = boto3.client("s3")
    boards: list[dict] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix="research/producer_leaderboard/"):
        for obj in page.get("Contents", []):
            if not _KEY_RE.match(obj["Key"]):
                continue
            body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
            boards.append(json.loads(body))
    return boards


def _boards_from_dir(path: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(path.glob("*.json"))]


def _load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _events_at_ref(ref: str, path: Path) -> list[dict] | None:
    """The register's events at a git ref, or None if the file is absent there."""
    rel = path.resolve().relative_to(_REPO_ROOT).as_posix()
    proc = subprocess.run(
        ["git", "show", f"{ref}:{rel}"],
        cwd=_REPO_ROOT, capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        if "does not exist" in proc.stderr or "exists on disk, but not in" in proc.stderr:
            return None
        raise RuntimeError(f"git show {ref}:{rel} failed: {proc.stderr.strip()}")
    return json.loads(proc.stdout)["events"]


def render_register(events: list[dict], n_boards: int) -> str:
    payload = {
        "slot": producer_arena.SLOT,
        "derived_from": "research/producer_leaderboard/{date}.json",
        "derived_by": "scripts/backfill_producer_arena_register.py",
        "append_only": True,
        "n_boards": n_boards,
        "events": events,
    }
    return json.dumps(payload, indent=2, sort_keys=False) + "\n"


def render_snapshot(snapshot: list[dict]) -> str:
    payload = {
        "derived_from": "research/producer_leaderboard/{date}.json",
        "derived_by": "scripts/backfill_producer_arena_register.py",
        "fields": ["name", "kind", "earliest_cohort"],
        "boards": snapshot,
    }
    return json.dumps(payload, indent=2, sort_keys=False) + "\n"


def check(
    boards: list[dict] | None,
    *,
    register_path: Path,
    snapshot_path: Path,
    base_ref: str | None,
) -> list[str]:
    """Every reason the committed register fails the check; ``[]`` is a pass."""
    problems: list[str] = []
    committed = _load_json(register_path)
    if committed is None:
        return [f"{register_path} does not exist"]
    events = committed["events"]

    if boards is None:
        snap = _load_json(snapshot_path)
        if snap is None:
            return [f"{snapshot_path} does not exist; nothing to check the register against"]
        boards = snap["boards"]

    base_events = _events_at_ref(base_ref, register_path) if base_ref else None
    if base_events is not None:
        problems.extend(producer_arena.append_only_violations(base_events, events))
    fold_base = events if base_events is None else base_events

    expected = producer_arena.register_events_from_boards(boards, existing_events=fold_base)
    if expected != events:
        missing = [e for e in expected if e not in events]
        unexpected = [e for e in events[len(fold_base):] if e not in expected]

        def _label(e: dict) -> str:
            name = (e.get("record") or {}).get("name") or e["arm_id"]
            return f"{e['kind']} {name} {e['date']}"

        problems.append(
            f"{register_path.name} is not the fold of its boards onto "
            f"{'the register at ' + base_ref if base_events is not None else 'itself'} "
            f"(NEW_ARM_CREATED_DATE_RULE={producer_arena.NEW_ARM_CREATED_DATE_RULE!r}). "
            f"Missing: {[_label(e) for e in missing]}. Not produced by the fold: "
            f"{[_label(e) for e in unexpected]}. Regenerate with this script "
            "(--bucket, or --from-snapshot --base-ref <base>) and commit both files."
        )

    try:
        from nousergon_lib.arena import ArmRegister

        ArmRegister.from_dicts(events)
    except Exception as e:  # noqa: BLE001 — reported, not raised: this is a check
        problems.append(f"{register_path.name} does not load as an ArmRegister: {e}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default=None)
    ap.add_argument("--from-dir", default=None)
    ap.add_argument(
        "--from-snapshot", action="store_true",
        help="fold the committed board snapshot instead of reading boards",
    )
    ap.add_argument("--check", action="store_true")
    ap.add_argument(
        "--base-ref", default=None,
        help="fold onto the register at this git ref; with --check, the ref the "
        "committed register must append-only extend",
    )
    ap.add_argument("--register", default=str(producer_arena.REGISTER_PATH))
    ap.add_argument("--snapshot", default=str(producer_arena.BOARD_SNAPSHOT_PATH))
    args = ap.parse_args(argv)

    register_path = Path(args.register)
    snapshot_path = Path(args.snapshot)

    boards: list[dict] | None = None
    if args.from_dir:
        boards = _boards_from_dir(Path(args.from_dir))
    elif args.bucket:
        boards = _boards_from_s3(args.bucket)
    elif args.from_snapshot:
        snap = _load_json(snapshot_path)
        if snap is None:
            print(f"{snapshot_path} does not exist", file=sys.stderr)
            return 2
        boards = snap["boards"]
    elif not args.check:
        ap.error("one of --bucket, --from-dir or --from-snapshot is required unless --check")

    if boards is not None and not boards:
        # Fail loud. Folding nothing appends nothing, which would hide a broken
        # read behind a green "register is current".
        print("no producer leaderboards found — refusing to proceed", file=sys.stderr)
        return 2

    if args.check:
        problems = check(
            boards, register_path=register_path, snapshot_path=snapshot_path,
            base_ref=args.base_ref,
        )
        for problem in problems:
            print(problem, file=sys.stderr)
        if problems:
            return 1
        print(f"{register_path} is current and append-only")
        return 0

    assert boards is not None
    if args.base_ref:
        existing = _events_at_ref(args.base_ref, register_path) or []
    else:
        committed = _load_json(register_path)
        existing = committed["events"] if committed else []
    events = producer_arena.register_events_from_boards(boards, existing_events=existing)
    violations = producer_arena.append_only_violations(existing, events)
    if violations:  # the fold guarantees this; checked so a regression cannot write
        for v in violations:
            print(v, file=sys.stderr)
        return 1

    register_path.parent.mkdir(parents=True, exist_ok=True)
    register_path.write_text(render_register(events, len(boards)))
    snapshot_path.write_text(render_snapshot(producer_arena.board_snapshot(boards)))
    print(
        f"wrote {register_path} ({len(events) - len(existing)} event(s) appended, "
        f"{len(events)} total, from {len(boards)} boards) and {snapshot_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
