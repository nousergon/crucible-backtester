"""Pin the calibration-detector scale fix + adequacy gate.

Root cause of the recurring false ``calibration_breakdown`` retrain alert
(e.g. 2026-06-17, ECE 0.339): ``compute_calibration_validation`` binned
``prediction_confidence`` (= ``|p_up-0.5|*2``, a MARGIN) against the direction
hit-rate, manufacturing a structural ECE on a perfectly calibrated model. The
fix measures the calibrated probability ``p_up`` against the realized UP outcome
(``1[actual_log_alpha>0]``) via ``alpha_engine_lib.quant.stats.calibration``,
and gates on EFFECTIVE-INDEPENDENT WINDOWS (overlapping 21d daily rows are
correlated) before letting ECE drive an alert.
"""
from __future__ import annotations

import sqlite3

import pytest

# Independent 21d cohorts (~31 calendar days apart → 3 windows).
_D1, _D2, _D3 = "2026-05-12", "2026-06-12", "2026-07-13"
_RUN = "2026-07-20"
_LOOKBACK = 120  # cutoff = run - 120d → all three dates included


def _make_db(tmp_path, rows: list[tuple]) -> str:
    db = tmp_path / "research.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE predictor_outcomes ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, prediction_date TEXT, "
        "predicted_direction TEXT, prediction_confidence REAL, p_up REAL, "
        "p_flat REAL, p_down REAL, actual_5d_return REAL, correct_5d INTEGER, "
        "actual_log_alpha REAL, horizon_days REAL, correct INTEGER)"
    )
    conn.executemany(
        "INSERT INTO predictor_outcomes (symbol, prediction_date, predicted_direction, "
        "prediction_confidence, p_up, p_flat, p_down, actual_5d_return, correct_5d, "
        "actual_log_alpha, horizon_days, correct) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return str(db)


def _rows(p_up: float, frac_up: float, dates: list[str], n_per_date: int) -> list[tuple]:
    """Rows at a fixed p_up where ``frac_up`` of them realize a positive alpha."""
    out: list[tuple] = []
    i = 0
    for d in dates:
        n_pos = round(frac_up * n_per_date)
        for k in range(n_per_date):
            alpha = 0.02 if k < n_pos else -0.02
            # prediction_confidence is deliberately a MISLEADING margin — the
            # fixed detector must ignore it and read p_up instead.
            conf = abs(p_up - 0.5) * 2.0
            out.append(
                (f"T{i}", d, "UP", conf, p_up, 0.0, 1 - p_up, None, None, alpha, 21, 1)
            )
            i += 1
    return out


def test_calibrated_p_up_is_not_flagged(tmp_path):
    """p_up=0.75 with 75% realized-up over 3 windows → ECE≈0, quality good.

    The OLD code would bin the margin (conf=0.5) vs hit-rate 0.75 → spurious
    ECE ~0.25 and a 'poor'/alert verdict. The fix reads p_up → no false alarm.
    """
    from analysis.production_health import compute_calibration_validation

    db = _make_db(tmp_path, _rows(0.75, 0.75, [_D1, _D2, _D3], 20))
    res = compute_calibration_validation(db, bucket="b", run_date=_RUN, lookback_days=_LOOKBACK)

    assert res.get("status") != "skipped"
    assert res["measured_on"] == "p_up_vs_realized_up"
    assert res["n_independent_windows"] == 3
    assert res["overall_ece"] < 0.05
    assert res["calibration_quality"] == "good"


def test_genuinely_miscalibrated_p_up_is_flagged(tmp_path):
    """p_up=0.9 but only 50% realize up over 3 windows → high ECE, 'poor'.

    Confirms the detector still catches REAL miscalibration after the fix.
    """
    from analysis.production_health import compute_calibration_validation

    db = _make_db(tmp_path, _rows(0.9, 0.5, [_D1, _D2, _D3], 20))
    res = compute_calibration_validation(db, bucket="b", run_date=_RUN, lookback_days=_LOOKBACK)

    assert res.get("status") != "skipped"
    assert res["overall_ece"] == pytest.approx(0.4, abs=0.05)
    assert res["calibration_quality"] == "poor"


def test_overlapping_window_adequacy_gate_skips(tmp_path):
    """Plenty of rows but all within one 21d window → skip, do not fire.

    This is the 2026-06-17 shape: 59 outcomes over ~3 weeks ≈ 1 independent
    window. ECE over that is noise; the gate must skip rather than alert.
    """
    from analysis.production_health import compute_calibration_validation

    # 60 rows across three dates all within ~2 weeks → 1 independent window.
    db = _make_db(tmp_path, _rows(0.9, 0.5, ["2026-05-12", "2026-05-18", "2026-05-25"], 20))
    res = compute_calibration_validation(db, bucket="b", run_date="2026-05-26", lookback_days=60)

    assert res["status"] == "skipped"
    assert res["reason"] == "insufficient_independent_windows"
    assert res["n_independent_windows"] < 3
    assert "overall_ece" not in res


def test_skip_does_not_trigger_retrain_alert(tmp_path):
    """End-to-end: an inadequate-window calibration result fires no trigger."""
    from analysis.production_health import compute_calibration_validation
    from analysis.retrain_alert import evaluate_retrain_triggers

    db = _make_db(tmp_path, _rows(0.9, 0.5, ["2026-05-12", "2026-05-18"], 20))
    cal = compute_calibration_validation(db, bucket="b", run_date="2026-05-26", lookback_days=60)
    alert = evaluate_retrain_triggers(production_health=None, feature_drift=None, calibration=cal)
    assert alert["triggered"] is False


def test_effective_independent_windows_helper():
    from analysis.production_health import _effective_independent_windows

    assert _effective_independent_windows([]) == 0
    assert _effective_independent_windows(["2026-05-12"]) == 1
    # within one horizon (~30d) → still 1
    assert _effective_independent_windows(["2026-05-12", "2026-05-20"]) == 1
    # three ~31d-spaced dates → 3
    assert _effective_independent_windows(["2026-05-12", "2026-06-12", "2026-07-13"]) == 3
