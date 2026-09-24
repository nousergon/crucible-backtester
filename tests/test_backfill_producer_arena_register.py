"""The producer register backfill is append-only and CI-checkable (alpha-engine-config-I11490).

``scripts/backfill_producer_arena_register.py --check`` runs in CI with no S3
access. It folds the committed board snapshot onto the register and fails if
the result is not the committed register. These tests drive the script's own
``main``/``check`` against temporary copies, so nothing under optimizer/arena/
is written.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from optimizer import producer_arena

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_REGISTER = REPO_ROOT / "tests" / "fixtures" / "producer_register_2026-08-28.json"

_spec = importlib.util.spec_from_file_location(
    "backfill_producer_arena_register",
    REPO_ROOT / "scripts" / "backfill_producer_arena_register.py",
)
backfill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backfill)


@pytest.fixture
def work(tmp_path):
    """Temporary copies of the committed register and snapshot."""
    reg = tmp_path / "producer_register.json"
    snap = tmp_path / "producer_board_snapshot.json"
    shutil.copy(producer_arena.REGISTER_PATH, reg)
    shutil.copy(producer_arena.BOARD_SNAPSHOT_PATH, snap)
    return reg, snap


def _args(reg, snap, *extra):
    return ["--register", str(reg), "--snapshot", str(snap), *extra]


def test_the_committed_register_passes_the_ci_check():
    assert backfill.main(["--check"]) == 0


def test_the_committed_register_extends_the_one_on_main(monkeypatch):
    """What CI runs with ``--base-ref``: the 2026-08-28 register is main's."""
    base = json.loads(FIXTURE_REGISTER.read_text())["events"]
    monkeypatch.setattr(backfill, "_events_at_ref", lambda ref, path: base)
    assert backfill.check(
        None,
        register_path=producer_arena.REGISTER_PATH,
        snapshot_path=producer_arena.BOARD_SNAPSHOT_PATH,
        base_ref="origin/main",
    ) == []


def test_a_register_behind_its_boards_fails(work, capsys):
    reg, snap = work
    payload = json.loads(reg.read_text())
    payload["events"] = [
        e for e in payload["events"]
        if (e.get("record") or {}).get("name") != "predictor_from_60"
    ]
    reg.write_text(json.dumps(payload))
    assert backfill.main(_args(reg, snap, "--check")) == 1
    assert "predictor_from_60" in capsys.readouterr().err


def test_a_rewritten_date_fails_against_the_base(work, monkeypatch, capsys):
    """The I11490 defect: an already-registered arm's date moved earlier."""
    reg, snap = work
    base = json.loads(reg.read_text())["events"]
    payload = json.loads(reg.read_text())
    for e in payload["events"]:
        if e["kind"] == "registered" and e["record"]["name"] == "no_agent_quant":
            e["date"] = e["record"]["created_date"] = "2026-06-30"
    reg.write_text(json.dumps(payload))
    monkeypatch.setattr(backfill, "_events_at_ref", lambda ref, path: base)
    assert backfill.main(_args(reg, snap, "--check", "--base-ref", "origin/main")) == 1
    assert "was rewritten" in capsys.readouterr().err


def test_a_new_arm_dated_under_another_rule_fails(work, monkeypatch, capsys):
    """Changing NEW_ARM_CREATED_DATE_RULE without regenerating is caught."""
    reg, snap = work
    base = json.loads(FIXTURE_REGISTER.read_text())["events"]
    monkeypatch.setattr(backfill, "_events_at_ref", lambda ref, path: base)
    other = next(r for r in producer_arena.CREATED_DATE_RULES
                 if r != producer_arena.NEW_ARM_CREATED_DATE_RULE)
    monkeypatch.setattr(producer_arena, "NEW_ARM_CREATED_DATE_RULE", other)
    assert backfill.main(_args(reg, snap, "--check", "--base-ref", "origin/main")) == 1
    assert "attractiveness_60" in capsys.readouterr().err


def test_regenerating_onto_the_base_applies_a_changed_rule(work, monkeypatch):
    """The documented follow-up to flipping the decision line."""
    reg, snap = work
    base = json.loads(FIXTURE_REGISTER.read_text())["events"]
    monkeypatch.setattr(backfill, "_events_at_ref", lambda ref, path: base)
    monkeypatch.setattr(
        producer_arena, "NEW_ARM_CREATED_DATE_RULE", producer_arena.CREATED_DATE_EARLIEST_COHORT,
    )
    assert backfill.main(_args(reg, snap, "--from-snapshot", "--base-ref", "origin/main")) == 0
    events = json.loads(reg.read_text())["events"]
    assert events[: len(base)] == base
    a60 = next(e for e in events if (e.get("record") or {}).get("name") == "attractiveness_60")
    assert a60["record"]["created_date"] == "2026-05-29"
    assert backfill.main(_args(reg, snap, "--check", "--base-ref", "origin/main")) == 0


def test_write_mode_appends_and_refreshes_the_snapshot(tmp_path):
    reg = tmp_path / "producer_register.json"
    snap = tmp_path / "producer_board_snapshot.json"
    shutil.copy(FIXTURE_REGISTER, reg)
    boards = tmp_path / "boards"
    boards.mkdir()
    (boards / "2026-09-23.json").write_text(json.dumps({
        "date": "2026-09-23",
        "arms": [{"name": "tech_score_20", "kind": "challenger", "dates_scored": []}],
    }))
    before = json.loads(reg.read_text())["events"]
    assert backfill.main(_args(reg, snap, "--from-dir", str(boards))) == 0
    after = json.loads(reg.read_text())["events"]
    assert after[: len(before)] == before
    assert [e["record"]["name"] for e in after[len(before):]] == ["tech_score_20"]
    assert json.loads(snap.read_text())["boards"][0]["arms"][0]["name"] == "tech_score_20"


def test_no_boards_is_refused_rather_than_reported_current(tmp_path, work):
    reg, snap = work
    empty = tmp_path / "empty"
    empty.mkdir()
    assert backfill.main(_args(reg, snap, "--check", "--from-dir", str(empty))) == 2
