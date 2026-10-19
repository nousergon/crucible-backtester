"""Hermetic tests for scripts/veto_era_lift.py (alpha-engine-config#12390).

The script's pure parts only: no ArcticDB, no S3. ``realized`` (the price join)
is exercised live by the operator run recorded on the issue.
"""
from __future__ import annotations

import importlib.util
import json
import random
import sqlite3
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "veto_era_lift", Path(__file__).resolve().parents[1] / "scripts" / "veto_era_lift.py"
)
vel = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(vel)


def _db(path: Path) -> str:
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE trades (date TEXT, ticker TEXT, action TEXT)")
    c.execute("CREATE TABLE executor_shadow_book (date TEXT, ticker TEXT, block_reason TEXT)")
    c.executemany("INSERT INTO trades VALUES (?,?,?)", [
        ("2026-04-09", "AAA", "ENTER"), ("2026-04-09", "BBB", "ENTER"),
        ("2026-04-10", "AAA", "ENTER"), ("2026-06-01", "ZZZ", "ENTER"),
    ])
    c.executemany("INSERT INTO executor_shadow_book VALUES (?,?,?)", [
        ("2026-04-07", "VVV", "GBM veto: a=-1%, rank=3"),
        ("2026-04-08", "VVV", "GBM veto: a=-1%, rank=3"),
        ("2026-04-09", "WWW", "GBM veto: a=-2%, rank=4"),
        ("2026-04-09", "XXX", "already in portfolio"),
    ])
    c.commit()
    c.close()
    return str(path)


def _preds(d: Path, date: str, rows: list[tuple[str, float, bool]]) -> None:
    (d / f"{date}.json").write_text(json.dumps({"predictions": [
        {"ticker": t, "predicted_alpha": a, "gbm_veto": v} for t, a, v in rows]}))


def test_wilson_bounds_and_empty():
    lo, hi = vel.wilson(15, 26)
    assert 0.38 < lo < 0.40 and 0.74 < hi < 0.75
    assert vel.wilson(0, 0)[0] != vel.wilson(0, 0)[0]  # NaN


def test_load_events_first_event_dedup_and_window(tmp_path):
    rows, first_v, enters, first_e, blocked, window = vel.load_events(_db(tmp_path / "t.db"))
    assert len(rows) == 3
    assert first_v == {"VVV": "2026-04-07", "WWW": "2026-04-09"}   # re-blocks collapse to the first day
    assert window == ("2026-04-07", "2026-04-09")
    assert first_e == {"AAA": "2026-04-09", "BBB": "2026-04-09"}   # out-of-window ENTER excluded
    assert all(b[2] != "already in portfolio" for b in blocked)


def test_replay_definition_is_noop_when_vetoed_rank_below_width(tmp_path):
    pdir = tmp_path / "p"
    pdir.mkdir()
    # vetoed names have the lowest alpha -> top-2 identical with and without the veto
    _preds(pdir, "2026-04-09", [("AAA", .03, False), ("BBB", .02, False), ("WWW", -.01, True)])
    out = vel.replay_definition_on_era([("2026-04-09", "AAA"), ("2026-04-09", "BBB")], pdir)
    assert out["dates_scored"] == 1 and out["dates_topN_differs"] == 0


def test_replay_definition_detects_binding_veto(tmp_path):
    pdir = tmp_path / "p"
    pdir.mkdir()
    # a vetoed name ranked inside the width: the veto binds
    _preds(pdir, "2026-04-09", [("AAA", .03, False), ("WWW", .02, True), ("BBB", .01, False)])
    out = vel.replay_definition_on_era([("2026-04-09", "AAA"), ("2026-04-09", "BBB")], pdir)
    assert out["dates_topN_differs"] == 1
    assert out["detail"][0]["ablated_only"] == ["WWW"]


def test_replay_definition_counts_missing_predictions(tmp_path):
    out = vel.replay_definition_on_era([("2026-04-09", "AAA")], tmp_path)
    assert out["dates_no_predictions"] == 1 and out["dates_scored"] == 0


def test_cluster_boot_is_deterministic_and_brackets_point():
    v = {"d1": [-.02, -.01], "d2": [.01], "d3": [-.03]}
    e = {"d1": [.0], "d2": [-.01, -.02], "d3": [.02]}
    a = vel.cluster_boot(v, e, random.Random(vel.SEED))
    b = vel.cluster_boot(v, e, random.Random(vel.SEED))
    assert a == b
    point, lo, hi = a
    assert lo <= point <= hi
