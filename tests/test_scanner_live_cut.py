"""``scanner_lift_live`` grades the cut the live scanner published.

alpha-engine-config-I11985 / I11155. ``scanner_lift`` reads
``scanner_evaluations``, whose writer retired 2026-07-12; on the 2026-10-02
artifact it was a 2026-04-12..2026-07-17 read of the retired tech_score gate,
78 days old, that the weekly Director read as this week's scanner. The live
scanner publishes its cut to ``candidates/{run_date}/candidates.json::
scanner_eval_log`` every cycle. These tests pin that ``scanner_lift_live``
measures THAT cut, with the same estimator, and says which cohorts it used.
"""

from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest

from analysis.end_to_end import (
    SCANNER_LIVE_COHORT_RULE,
    SCANNER_LIVE_CUT_ARM,
    SCANNER_METRIC_ARM,
    _scanner_lift,
    _scanner_lift_live,
    compute_lift_metrics,
    load_universe_returns_frame,
)

BUCKET = "test-bucket"
FRI_1 = "2026-08-07"
WED_2 = "2026-08-12"   # mid-week rerun, superseded by FRI_2 in the same ISO week
FRI_2 = "2026-08-14"
FRI_3 = "2026-09-25"   # published, 21d window not closed yet
N = 20                 # scanned names per cohort


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data


class _StubS3:
    """list_objects_v2 (CommonPrefixes) + get_object over candidates.json docs."""

    def __init__(self, docs: dict[str, dict]):
        self._docs = docs  # run_date -> candidates.json body

    def list_objects_v2(self, Bucket, Prefix, Delimiter=None, ContinuationToken=None):
        return {
            "CommonPrefixes": [{"Prefix": f"{Prefix}{d}/"} for d in sorted(self._docs)],
            "IsTruncated": False,
        }

    def get_object(self, Bucket, Key):
        d = Key.split("/")[1]
        return {"Body": _Body(json.dumps(self._docs[d]).encode())}


def _winner(i: int) -> bool:
    return i % 2 == 0  # T00, T02, ... beat SPY over 21d


def _artifact(run_date: str, picks: set[int]) -> dict:
    return {
        "run_date": run_date,
        "scanner_eval_log": [
            {"ticker": f"T{i:02d}", "quant_filter_pass": 1 if i in picks else 0}
            for i in range(N)
        ],
    }


# The live scanner picks the five lowest-index WINNERS every week.
LIVE_PICKS = {0, 2, 4, 6, 8}
# The rerun that was superseded picked only losers — if it were counted it
# would drag precision down, which is how the rule is proven to apply.
RERUN_PICKS = {1, 3, 5, 7, 9}
# The retired tech_score gate (scanner_evaluations) picked losers.
RETIRED_PICKS = {1, 3, 5, 7, 9}


def _docs() -> dict[str, dict]:
    return {
        "2026-06-05": {"run_date": "2026-06-05", "scanner_eval_log": []},  # pre-I1458
        FRI_1: _artifact(FRI_1, LIVE_PICKS),
        WED_2: _artifact(WED_2, RERUN_PICKS),
        FRI_2: _artifact(FRI_2, LIVE_PICKS),
        FRI_3: _artifact(FRI_3, LIVE_PICKS),
    }


def _research_db(tmp_path) -> str:
    db = tmp_path / "research.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE universe_returns ("
        "ticker TEXT, eval_date TEXT, sector TEXT, "
        "return_5d REAL, spy_return_5d REAL, beat_spy_5d INTEGER, "
        "return_21d REAL, spy_return_21d REAL, beat_spy_21d INTEGER, "
        "log_return_21d REAL, log_spy_return_21d REAL)"
    )
    conn.execute(
        "CREATE TABLE scanner_evaluations "
        "(ticker TEXT, eval_date TEXT, quant_filter_pass INTEGER)"
    )
    for d in (FRI_1, WED_2, FRI_2, FRI_3):
        matured = d != FRI_3
        for i in range(N):
            w = _winner(i)
            conn.execute(
                "INSERT INTO universe_returns VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (f"T{i:02d}", d, "Tech" if i < 10 else "Health",
                 0.01, 0.0, 1 if w else 0,
                 (0.04 if w else -0.03) if matured else None,
                 0.0 if matured else None,
                 (1 if w else 0) if matured else None,
                 (0.04 if w else -0.03) if matured else None,
                 0.0 if matured else None),
            )
    # compute_lift_metrics' predictor read must find the table (empty is fine).
    conn.execute("CREATE TABLE predictor_outcomes (symbol TEXT, prediction_date TEXT, "
                 "predicted_direction TEXT, prediction_confidence REAL)")
    # The retired table's last rows, frozen in July (writer retired 2026-07-12).
    for i in range(N):
        conn.execute(
            "INSERT INTO scanner_evaluations VALUES (?,?,?)",
            (f"T{i:02d}", FRI_1, 1 if i in RETIRED_PICKS else 0),
        )
    conn.commit()
    conn.close()
    return str(db)


@pytest.fixture
def ur(tmp_path):
    conn = sqlite3.connect(_research_db(tmp_path))
    try:
        yield load_universe_returns_frame(conn), conn
    finally:
        conn.close()


def test_grades_the_published_cut_not_the_retired_table(ur):
    """The misattributed case: the live block must not echo scanner_evaluations."""
    frame, conn = ur
    live = _scanner_lift_live(frame, BUCKET, s3_client=_StubS3(_docs()))
    retired = _scanner_lift(conn, frame, "", [])

    assert live["status"] == "ok"
    assert live["arm"] == SCANNER_LIVE_CUT_ARM
    assert retired["arm"] == SCANNER_METRIC_ARM
    # Live cut picked winners on both matured cohorts: precision 1.0.
    assert live["classification_21d"]["precision"] == pytest.approx(1.0)
    assert live["classification_21d"]["tp"] == 10
    # The retired gate picked losers: the two blocks measure different selections.
    assert retired["classification_21d"]["precision"] == pytest.approx(0.0)
    assert live["lift_21d_log"]["lift"] > 0 > retired["lift_21d_log"]["lift"]


def test_one_cohort_per_iso_week_and_superseded_reruns_are_named(ur):
    frame, _ = ur
    live = _scanner_lift_live(frame, BUCKET, s3_client=_StubS3(_docs()))
    assert live["cohort_rule"] == SCANNER_LIVE_COHORT_RULE
    assert live["cohort_dates"] == [FRI_1, FRI_2, FRI_3]
    assert live["superseded_dates"] == [WED_2]
    # WED_2's loser picks would have made fp > 0; they are not counted.
    assert live["classification_21d"]["fp"] == 0


def test_empty_eval_logs_are_listed_not_counted(ur):
    frame, _ = ur
    live = _scanner_lift_live(frame, BUCKET, s3_client=_StubS3(_docs()))
    assert live["artifacts_without_eval_log"] == ["2026-06-05"]


def test_unmatured_cohort_is_published_but_not_in_the_21d_read(ur):
    frame, _ = ur
    live = _scanner_lift_live(frame, BUCKET, s3_client=_StubS3(_docs()))
    assert live["newest_published_cohort"] == FRI_3
    assert live["n_cohorts_matured_21d"] == 2
    assert live["first_matured_eval_date_21d"] == FRI_1
    assert live["last_matured_eval_date_21d"] == FRI_2
    assert live["classification_21d"]["n"] == 2 * N
    assert live["last_eval_date"] == FRI_3


def test_no_artifacts_is_insufficient_data_not_a_number(ur):
    frame, _ = ur
    live = _scanner_lift_live(frame, BUCKET, s3_client=_StubS3({}))
    assert live["status"] == "insufficient_data"
    assert "classification_21d" not in live
    assert live["arm"] == SCANNER_LIVE_CUT_ARM


def test_cohorts_without_universe_rows_are_insufficient_data():
    empty = pd.DataFrame(columns=["ticker", "eval_date", "return_5d", "beat_spy_21d"])
    live = _scanner_lift_live(empty, BUCKET, s3_client=_StubS3(_docs()))
    assert live["status"] == "insufficient_data"
    assert live["cohort_dates"] == [FRI_1, FRI_2, FRI_3]


def test_compute_lift_metrics_emits_both_blocks_side_by_side(tmp_path, monkeypatch):
    """Additive: scanner_lift stays as the retired arm's record."""
    from analysis import end_to_end as E

    real = E._scanner_lift_live
    monkeypatch.setattr(
        E, "_scanner_lift_live",
        lambda ur, bucket: real(ur, bucket, s3_client=_StubS3(_docs())),
    )
    out = compute_lift_metrics(_research_db(tmp_path), bucket=BUCKET)
    assert out["scanner_lift"]["arm"] == SCANNER_METRIC_ARM
    assert out["scanner_lift_live"]["arm"] == SCANNER_LIVE_CUT_ARM
    assert out["scanner_lift_live"]["status"] == "ok"
