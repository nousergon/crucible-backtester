"""Champion-scoping fix for the rolling IC / ic_ratio / degradation_flag path
in ``analysis.production_health.compute_production_health``.

Measured 2026-09-09 against live S3 (`production_health.json` evaluated
2026-09-09T13:40:41Z): rolling_30d_ic was computed over ALL 498 resolved
outcomes in the lookback window with NO scoping to which champion served
them, then divided by `_load_training_ic`'s reference — which resolves the
CURRENTLY-deployed champion's training-run OOS IC. The live champion
(v3.0-meta-2026-09-04-cc3271ea) had served only 2026-09-08 and 2026-09-09 at
that point; at a ~21d horizon essentially none of the 498 resolved outcomes
came from it. The ratio therefore compared one (retired) model's live
behaviour against a DIFFERENT model's training reference — the same defect
class as alpha-engine-config-I10290 (a veto compared against itself).

Fix: scope the numerator to the rows served by the SAME champion identified
via each date's ``predictor/predictions/{date}.json::champion_version_id``
("current" = whichever champion served the most recent resolvable date).
Three properties pinned here:

  * a mixed-champion window computes the ratio over the CURRENT champion's
    rows only, not silently over the mixed set;
  * a current champion with too few resolved outcomes yields the honest
    "insufficient_champion_samples" posture — ic_ratio/degradation_flag never
    silently compute from a near-empty or wrong-champion numerator;
  * a single-champion window is unaffected — the ratio matches the (pre-fix)
    whole-window computation exactly.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from pipeline_common import ACTIVE_HORIZON_DAYS

_COLS = (
    "symbol, prediction_date, predicted_direction, prediction_confidence, "
    "p_up, p_flat, p_down, score_modifier_applied, actual_5d_return, "
    "correct_5d, actual_log_alpha, horizon_days, correct"
)

_OLD_CHAMPION = "v3.0-meta-2026-08-14-119e069b"
_NEW_CHAMPION = "v3.0-meta-2026-09-04-cc3271ea"


def _make_db(tmp_path, rows: list[tuple]) -> str:
    db = tmp_path / "research.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE predictor_outcomes ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, prediction_date TEXT, "
        "predicted_direction TEXT, prediction_confidence REAL, p_up REAL, "
        "p_flat REAL, p_down REAL, score_modifier_applied REAL, "
        "actual_5d_return REAL, correct_5d INTEGER, actual_log_alpha REAL, "
        "horizon_days REAL, correct INTEGER)"
    )
    conn.executemany(
        f"INSERT INTO predictor_outcomes ({_COLS}) VALUES ({','.join('?' * 13)})",
        rows,
    )
    conn.commit()
    conn.close()
    return str(db)


def _row(i: int, pred_date: str, p_up: float, p_down: float, alpha: float) -> tuple:
    """A fully graded row at the active horizon with a distinct
    (net_signal, actual) pair so pearsonr over different subsets diverges."""
    return (
        f"T{i}", pred_date, "UP", 0.6, p_up, 0.0, p_down, 0.0,
        None, 1, alpha, ACTIVE_HORIZON_DAYS, 1,
    )


def _rows_for_champion(dates_and_n: list[tuple[str, int]], start_symbol: int = 0) -> list[tuple]:
    """Rows whose (net_signal, actual) are strictly increasing across the
    whole set — so pearsonr(net_signal, actual) == 1.0 exactly when computed
    over that whole set, and measurably < 1.0 if a different, unrelated
    champion's rows (see `_rows_noise`) are mixed in."""
    rows = []
    i = start_symbol
    for date, n in dates_and_n:
        for k in range(n):
            p_up = 0.5 + 0.01 * i
            p_down = 0.5 - 0.01 * i
            alpha = 0.001 * i
            rows.append(_row(i, date, p_up, p_down, alpha))
            i += 1
    return rows


def _rows_noise(date: str, n: int, start_symbol: int) -> list[tuple]:
    """Rows with net_signal/actual UNCORRELATED (alternating sign) with the
    champion fixture above — mixing these in must move a naive whole-window
    IC away from 1.0."""
    rows = []
    for k in range(n):
        i = start_symbol + k
        p_up = 0.5 + (0.02 if k % 2 == 0 else -0.02)
        p_down = 1.0 - p_up
        alpha = -0.001 if k % 2 == 0 else 0.001
        rows.append(_row(i, date, p_up, p_down, alpha))
    return rows


class _Body:
    def __init__(self, raw: bytes):
        self._raw = raw

    def read(self) -> bytes:
        return self._raw


class _StubS3:
    """Keyed by exact S3 Key. `predictions_by_date` maps date -> champion_version_id;
    every predictions/{date}.json for a known date returns a minimal envelope
    carrying just that field (L1/L2 fields absent -> decomposition degrades to
    None, which is fine, out of scope here). `training_summary` is returned
    verbatim for the training_summary_latest.json key. Anything else raises,
    like a real NoSuchKey."""

    def __init__(self, predictions_by_date: dict[str, str], training_summary: dict):
        self._predictions_by_date = predictions_by_date
        self._training_summary = training_summary

    def get_object(self, *, Bucket, Key):  # noqa: N803
        if Key == "predictor/metrics/training_summary_latest.json":
            return {"Body": _Body(json.dumps(self._training_summary).encode())}
        for d, champion in self._predictions_by_date.items():
            if Key == f"predictor/predictions/{d}.json":
                return {"Body": _Body(json.dumps({"champion_version_id": champion, "predictions": []}).encode())}
        raise RuntimeError(f"NoSuchKey: {Bucket}/{Key}")


@pytest.fixture
def patch_s3(monkeypatch):
    def _apply(predictions_by_date: dict[str, str], training_summary: dict):
        from analysis import production_health as ph

        stub = _StubS3(predictions_by_date, training_summary)
        monkeypatch.setattr(ph.boto3, "client", lambda svc, *a, **k: stub)

    return _apply


# ── Property 1: mixed-champion window scopes to the CURRENT champion ────────


def test_mixed_champion_window_scopes_numerator_to_current_champion(tmp_path, patch_s3):
    from analysis.production_health import compute_production_health

    old_date, new_date = "2026-07-01", "2026-07-20"
    rows = (
        _rows_for_champion([(new_date, 15)], start_symbol=0)  # current champion: perfect IC
        + _rows_noise(old_date, 15, start_symbol=100)  # retired champion: uncorrelated noise
    )
    db = _make_db(tmp_path, rows)
    patch_s3(
        predictions_by_date={old_date: _OLD_CHAMPION, new_date: _NEW_CHAMPION},
        training_summary={"meta_model_oos_ic": 0.9},  # deliberately generous — proves scoping, not threshold luck
    )

    result = compute_production_health(db, bucket="b", run_date="2026-08-01", lookback_days=90)

    assert result["champion_scoping_status"] == "ok"
    assert result["champion_version_id"] == _NEW_CHAMPION
    assert result["n_champion_scoped"] == 15
    assert result["n_resolved"] == 30  # whole-window count is unaffected
    # Scoped IC must be the CURRENT champion's own (perfect, ==1.0) — not the
    # mixed value a naive whole-window pearsonr over all 30 rows would give.
    assert result["rolling_30d_ic"] == 1.0


def test_naive_whole_window_ic_would_have_differed(tmp_path):
    """Sanity check for the fixture itself: without scoping, pearsonr over
    the FULL mixed set is measurably below 1.0 — proves the fixture actually
    exercises the mixed-champion defect, not a degenerate no-op case."""
    from scipy.stats import pearsonr

    old_date, new_date = "2026-07-01", "2026-07-20"
    rows = (
        _rows_for_champion([(new_date, 15)], start_symbol=0)
        + _rows_noise(old_date, 15, start_symbol=100)
    )
    net_signal = [r[4] - r[6] for r in rows]  # p_up - p_down
    actual = [r[10] for r in rows]  # actual_log_alpha
    ic, _ = pearsonr(net_signal, actual)
    assert ic < 0.95, "fixture must produce a mixed IC below the scoped 1.0"


# ── Property 2: current champion below _MIN_SAMPLES -> uncomputable ─────────


def test_current_champion_below_min_samples_is_uncomputable(tmp_path, patch_s3):
    """Today's live case: the current champion has served too few dates for
    the horizon to have resolved outcomes yet. Must NOT fall back to a mixed
    or wrong-champion ratio, and must NOT fire degradation_flag."""
    from analysis.production_health import compute_production_health

    old_date, new_date = "2026-07-01", "2026-07-29"
    rows = (
        _rows_for_champion([(new_date, 5)], start_symbol=0)  # current: only 5 rows, < _MIN_SAMPLES
        + _rows_for_champion([(old_date, 15)], start_symbol=100)  # retired: plenty, but not current
    )
    db = _make_db(tmp_path, rows)
    patch_s3(
        predictions_by_date={old_date: _OLD_CHAMPION, new_date: _NEW_CHAMPION},
        # Deliberately tiny reference so an unscoped/mixed ratio would trip
        # the 0.50 degradation threshold if the fix didn't apply.
        training_summary={"meta_model_oos_ic": 0.001},
    )

    result = compute_production_health(db, bucket="b", run_date="2026-08-01", lookback_days=90)

    assert result["champion_scoping_status"] == "insufficient_champion_samples"
    assert result["champion_version_id"] == _NEW_CHAMPION
    assert result["n_champion_scoped"] == 5
    assert result["rolling_30d_ic"] is None
    assert result["ic_ratio"] is None
    assert result["degradation_flag"] is False


def test_uncomputable_posture_does_not_reach_retrain_alert(tmp_path, patch_s3):
    """End-to-end: the insufficient-champion-samples posture must not
    surface as an ic_degradation trigger (evaluate_retrain_triggers reads
    degradation_flag directly)."""
    from analysis.production_health import compute_production_health
    from analysis.retrain_alert import evaluate_retrain_triggers

    old_date, new_date = "2026-07-01", "2026-07-29"
    rows = (
        _rows_for_champion([(new_date, 3)], start_symbol=0)
        + _rows_for_champion([(old_date, 15)], start_symbol=100)
    )
    db = _make_db(tmp_path, rows)
    patch_s3(
        predictions_by_date={old_date: _OLD_CHAMPION, new_date: _NEW_CHAMPION},
        training_summary={"meta_model_oos_ic": 0.001},
    )

    ph = compute_production_health(db, bucket="b", run_date="2026-08-01", lookback_days=90)
    alert = evaluate_retrain_triggers(ph, feature_drift=None, calibration=None)

    assert ph["degradation_flag"] is False
    assert not any(r["trigger"] == "ic_degradation" for r in alert["reasons"])


# ── Property 3: single-champion window is unaffected ────────────────────────


def test_single_champion_window_ratio_unchanged(tmp_path, patch_s3):
    """No champion rotation in the window: scoped == whole window, ic_ratio
    behaves exactly as the pre-fix computation."""
    from analysis.production_health import compute_production_health

    date = "2026-07-20"
    rows = _rows_for_champion([(date, 15)], start_symbol=0)
    db = _make_db(tmp_path, rows)
    patch_s3(
        predictions_by_date={date: _NEW_CHAMPION},
        training_summary={"meta_model_oos_ic": 0.9},
    )

    result = compute_production_health(db, bucket="b", run_date="2026-08-01", lookback_days=90)

    assert result["champion_scoping_status"] == "ok"
    assert result["n_champion_scoped"] == result["n_resolved"] == 15
    assert result["rolling_30d_ic"] == 1.0
    assert result["ic_ratio"] == pytest.approx(1.0 / 0.9, abs=0.01)
    assert result["degradation_flag"] is False


# ── Property 4: champion feed itself unreadable -> fails open, records it ───


def test_champion_feed_unresolved_falls_back_to_whole_window(tmp_path, monkeypatch):
    """If predictor/predictions/{date}.json is unreadable for every date in
    the window (e.g. transient S3 issue on a secondary feed), the primary
    detector must not blackout — it falls back to the pre-fix whole-window
    IC, but records `champion_scoping_status: "unresolved"` so this is
    auditable rather than silently indistinguishable from "ok"."""
    from analysis import production_health as ph

    class _AllFail:
        def get_object(self, *, Bucket, Key):  # noqa: N803
            raise RuntimeError(f"NoSuchKey: {Bucket}/{Key}")

    monkeypatch.setattr(ph.boto3, "client", lambda svc, *a, **k: _AllFail())

    date = "2026-07-20"
    rows = _rows_for_champion([(date, 15)], start_symbol=0)
    db = _make_db(tmp_path, rows)

    result = ph.compute_production_health(db, bucket="b", run_date="2026-08-01", lookback_days=90)

    assert result["champion_scoping_status"] == "unresolved"
    assert result["champion_version_id"] is None
    assert result["n_champion_scoped"] is None
    # training_ic also fails to load against _AllFail -> ic_ratio is None,
    # but rolling_30d_ic itself must still compute over the whole window.
    assert result["rolling_30d_ic"] == 1.0
