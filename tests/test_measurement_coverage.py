"""Unit tests for analysis.measurement_coverage (config#909).

The measurement-coverage funnel traces the actionable signal set through four
production stages — signals → predictions → fills → P&L attribution — and emits
the ``coverage.json`` body the dashboard's measurement-coverage panel reads.

Contract locked here:
- Full coverage: every ENTER signal predicted, filled, attributed → all
  ratios 1.0, no gaps, status="ok".
- A gap at each stage drops the corresponding ratio and records the lost
  tickers in gaps_by_stage (one test per stage transition).
- Missing-input tolerance: an absent signals.json / predictions.json /
  trades.db degrades that stage (and everything downstream) to None,
  status="partial", and NEVER raises.
- Strict-subset funnel: a prediction/fill for a ticker that never fired an
  ENTER signal is out of scope (doesn't inflate counts).
- no_signals: signals.json present but no ENTER signals → status="no_signals",
  counts 0/None with no fabricated coverage.
- run_date required; unexpected faults return status="error" (never raise).
"""

from __future__ import annotations

import json
import sqlite3
from io import BytesIO
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from analysis.measurement_coverage import (
    compute_measurement_coverage,
    _ratio,
    _gap,
)

DATE = "2026-06-20"
BUCKET = "test-bucket"


# ── S3 stub ──────────────────────────────────────────────────────────────────


def _body(payload) -> dict:
    return {"Body": BytesIO(json.dumps(payload).encode())}


def _build_s3(objects: dict[str, object]) -> MagicMock:
    """Stub S3 whose get_object returns the JSON payload registered for a key,
    else raises NoSuchKey (the absence path the producer must tolerate)."""
    s3 = MagicMock()

    def get_object(*, Bucket, Key):
        if Key in objects:
            return _body(objects[Key])
        raise ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject"
        )

    s3.get_object.side_effect = get_object
    return s3


def _signals_payload(enter_tickers, *, other=None) -> dict:
    signals = {t: {"ticker": t, "signal": "ENTER"} for t in enter_tickers}
    for t in other or []:
        signals[t] = {"ticker": t, "signal": "HOLD"}
    return {"date": DATE, "signals": signals}


def _signals_key() -> str:
    return f"signals/{DATE}/signals.json"


def _predictions_key() -> str:
    return f"predictor/predictions/{DATE}.json"


def _make_trades_db(tmp_path, rows) -> str:
    """rows: list of (ticker, action, date, realized_return_pct, realized_alpha_pct)."""
    p = tmp_path / "trades.db"
    conn = sqlite3.connect(p)
    conn.execute(
        "CREATE TABLE trades (ticker TEXT, action TEXT, date TEXT, "
        "realized_return_pct REAL, realized_alpha_pct REAL)"
    )
    conn.executemany("INSERT INTO trades VALUES (?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return str(p)


# ── Full coverage ────────────────────────────────────────────────────────────


def test_full_coverage(tmp_path):
    tickers = ["AAA", "BBB", "CCC"]
    s3 = _build_s3({
        _signals_key(): _signals_payload(tickers),
        _predictions_key(): {t: 0.05 for t in tickers},
    })
    db = _make_trades_db(
        tmp_path, [(t, "ENTER", DATE, 1.2, 0.4) for t in tickers]
    )

    out = compute_measurement_coverage(
        bucket=BUCKET, run_date=DATE, trades_db_path=db, s3_client=s3
    )

    assert out["status"] == "ok"
    assert out["signal_count"] == 3
    assert out["predicted_count"] == 3
    assert out["executed_count"] == 3
    assert out["attributed_count"] == 3
    r = out["coverage_ratios"]
    assert r["predicted_of_signal"] == 1.0
    assert r["executed_of_predicted"] == 1.0
    assert r["attributed_of_executed"] == 1.0
    assert r["attributed_of_signal"] == 1.0
    for g in out["gaps_by_stage"].values():
        assert g["lost"] == 0
    assert all(out["stage_availability"].values())


# ── A gap at each stage ──────────────────────────────────────────────────────


def test_gap_at_prediction_stage(tmp_path):
    tickers = ["AAA", "BBB", "CCC"]
    s3 = _build_s3({
        _signals_key(): _signals_payload(tickers),
        # CCC never got a prediction.
        _predictions_key(): {"AAA": 0.05, "BBB": 0.03},
    })
    db = _make_trades_db(
        tmp_path, [(t, "ENTER", DATE, 1.0, 0.2) for t in ("AAA", "BBB")]
    )

    out = compute_measurement_coverage(
        bucket=BUCKET, run_date=DATE, trades_db_path=db, s3_client=s3
    )

    assert out["status"] == "ok"  # all four sources read
    assert out["signal_count"] == 3
    assert out["predicted_count"] == 2
    assert out["executed_count"] == 2
    assert out["attributed_count"] == 2
    assert out["coverage_ratios"]["predicted_of_signal"] == round(2 / 3, 4)
    assert out["coverage_ratios"]["executed_of_predicted"] == 1.0
    gap = out["gaps_by_stage"]["signal_to_prediction"]
    assert gap["lost"] == 1
    assert gap["tickers"] == ["CCC"]


def test_gap_at_execution_stage(tmp_path):
    tickers = ["AAA", "BBB", "CCC"]
    s3 = _build_s3({
        _signals_key(): _signals_payload(tickers),
        _predictions_key(): {t: 0.05 for t in tickers},
    })
    # CCC predicted but never filled.
    db = _make_trades_db(
        tmp_path, [(t, "ENTER", DATE, 1.0, 0.2) for t in ("AAA", "BBB")]
    )

    out = compute_measurement_coverage(
        bucket=BUCKET, run_date=DATE, trades_db_path=db, s3_client=s3
    )

    assert out["predicted_count"] == 3
    assert out["executed_count"] == 2
    assert out["attributed_count"] == 2
    assert out["coverage_ratios"]["executed_of_predicted"] == round(2 / 3, 4)
    gap = out["gaps_by_stage"]["prediction_to_execution"]
    assert gap["lost"] == 1
    assert gap["tickers"] == ["CCC"]


def test_gap_at_attribution_stage(tmp_path):
    tickers = ["AAA", "BBB", "CCC"]
    s3 = _build_s3({
        _signals_key(): _signals_payload(tickers),
        _predictions_key(): {t: 0.05 for t in tickers},
    })
    # CCC filled but P&L not attributed yet (both realized cols NULL).
    db = _make_trades_db(tmp_path, [
        ("AAA", "ENTER", DATE, 1.0, 0.2),
        ("BBB", "ENTER", DATE, 0.5, None),     # alpha null but return present → attributed
        ("CCC", "ENTER", DATE, None, None),    # nothing attributed
    ])

    out = compute_measurement_coverage(
        bucket=BUCKET, run_date=DATE, trades_db_path=db, s3_client=s3
    )

    assert out["executed_count"] == 3
    assert out["attributed_count"] == 2
    assert out["coverage_ratios"]["attributed_of_executed"] == round(2 / 3, 4)
    assert out["coverage_ratios"]["attributed_of_signal"] == round(2 / 3, 4)
    gap = out["gaps_by_stage"]["execution_to_attribution"]
    assert gap["lost"] == 1
    assert gap["tickers"] == ["CCC"]


# ── Strict-subset funnel ─────────────────────────────────────────────────────


def test_out_of_scope_downstream_rows_dont_inflate(tmp_path):
    """A prediction/fill for a ticker with no ENTER signal is out of funnel
    scope — counts stay bounded by the signal set."""
    s3 = _build_s3({
        _signals_key(): _signals_payload(["AAA"], other=["ZZZ"]),
        # ZZZ predicted + filled but never an ENTER signal.
        _predictions_key(): {"AAA": 0.05, "ZZZ": 0.09},
    })
    db = _make_trades_db(tmp_path, [
        ("AAA", "ENTER", DATE, 1.0, 0.2),
        ("ZZZ", "ENTER", DATE, 2.0, 0.5),
    ])

    out = compute_measurement_coverage(
        bucket=BUCKET, run_date=DATE, trades_db_path=db, s3_client=s3
    )

    assert out["signal_count"] == 1
    assert out["predicted_count"] == 1
    assert out["executed_count"] == 1
    assert out["attributed_count"] == 1
    assert out["coverage_ratios"]["attributed_of_signal"] == 1.0


# ── Missing-input tolerance ──────────────────────────────────────────────────


def test_missing_signals_file(tmp_path):
    s3 = _build_s3({_predictions_key(): {"AAA": 0.05}})  # no signals.json
    db = _make_trades_db(tmp_path, [("AAA", "ENTER", DATE, 1.0, 0.2)])

    out = compute_measurement_coverage(
        bucket=BUCKET, run_date=DATE, trades_db_path=db, s3_client=s3
    )

    assert out["status"] == "partial"
    assert out["signal_count"] is None
    assert out["predicted_count"] is None  # no denominator → unmeasured
    assert out["coverage_ratios"]["predicted_of_signal"] is None
    assert out["gaps_by_stage"]["signal_to_prediction"]["lost"] is None
    assert out["stage_availability"]["signals"] is False


def test_missing_predictions_file(tmp_path):
    s3 = _build_s3({_signals_key(): _signals_payload(["AAA", "BBB"])})
    db = _make_trades_db(tmp_path, [("AAA", "ENTER", DATE, 1.0, 0.2)])

    out = compute_measurement_coverage(
        bucket=BUCKET, run_date=DATE, trades_db_path=db, s3_client=s3
    )

    assert out["status"] == "partial"
    assert out["signal_count"] == 2
    assert out["predicted_count"] is None  # predictions stage unmeasured
    assert out["executed_count"] is None   # downstream also unmeasured
    assert out["attributed_count"] is None
    assert out["coverage_ratios"]["predicted_of_signal"] is None
    assert out["stage_availability"]["predictions"] is False
    assert out["stage_availability"]["signals"] is True


def test_missing_trades_db():
    s3 = _build_s3({
        _signals_key(): _signals_payload(["AAA", "BBB"]),
        _predictions_key(): {"AAA": 0.05, "BBB": 0.03},
    })

    out = compute_measurement_coverage(
        bucket=BUCKET, run_date=DATE, trades_db_path=None, s3_client=s3
    )

    assert out["status"] == "partial"
    assert out["signal_count"] == 2
    assert out["predicted_count"] == 2
    assert out["executed_count"] is None   # fills unmeasured
    assert out["attributed_count"] is None
    assert out["coverage_ratios"]["predicted_of_signal"] == 1.0
    assert out["coverage_ratios"]["executed_of_predicted"] is None
    assert out["stage_availability"]["fills"] is False
    assert out["stage_availability"]["attribution"] is False


def test_trades_db_path_nonexistent(tmp_path):
    s3 = _build_s3({
        _signals_key(): _signals_payload(["AAA"]),
        _predictions_key(): {"AAA": 0.05},
    })
    out = compute_measurement_coverage(
        bucket=BUCKET, run_date=DATE,
        trades_db_path=str(tmp_path / "nope.db"), s3_client=s3,
    )
    assert out["status"] == "partial"
    assert out["executed_count"] is None


# ── no_signals / edge cases ──────────────────────────────────────────────────


def test_no_enter_signals(tmp_path):
    s3 = _build_s3({
        _signals_key(): _signals_payload([], other=["HOLDME"]),
        _predictions_key(): {},
    })
    db = _make_trades_db(tmp_path, [])

    out = compute_measurement_coverage(
        bucket=BUCKET, run_date=DATE, trades_db_path=db, s3_client=s3
    )

    assert out["status"] == "no_signals"
    assert out["signal_count"] == 0
    # Empty denominator → ratios None, not fabricated 1.0 or 0.0.
    assert out["coverage_ratios"]["predicted_of_signal"] is None
    assert out["coverage_ratios"]["attributed_of_signal"] is None


def test_run_date_required():
    out = compute_measurement_coverage(bucket=BUCKET, run_date=None)
    assert out["status"] == "error"
    assert "run_date" in out["error"]


def test_unexpected_fault_returns_error_not_raises():
    s3 = MagicMock()
    s3.get_object.side_effect = RuntimeError("boom")
    out = compute_measurement_coverage(
        bucket=BUCKET, run_date=DATE, s3_client=s3
    )
    # _read_signal_tickers only catches ClientError/ValueError/KeyError; a
    # RuntimeError bubbles to the top-level guard → status="error", no raise.
    assert out["status"] == "error"
    assert out["date"] == DATE


def test_predictions_wrapped_envelope(tmp_path):
    """Tolerate the {"predictions": {...}} envelope as well as the flat map."""
    s3 = _build_s3({
        _signals_key(): _signals_payload(["AAA", "BBB"]),
        _predictions_key(): {"predictions": {"AAA": 0.05, "BBB": 0.03}},
    })
    db = _make_trades_db(tmp_path, [("AAA", "ENTER", DATE, 1.0, 0.2)])
    out = compute_measurement_coverage(
        bucket=BUCKET, run_date=DATE, trades_db_path=db, s3_client=s3
    )
    assert out["predicted_count"] == 2


# ── Pure-helper unit checks ──────────────────────────────────────────────────


def test_ratio_helper():
    assert _ratio(1, 2) == 0.5
    assert _ratio(0, 0) is None       # empty denominator
    assert _ratio(None, 5) is None    # unmeasured numerator
    assert _ratio(3, None) is None    # unmeasured denominator


def test_gap_helper():
    assert _gap({"A", "B"}, {"A"}) == {"lost": 1, "tickers": ["B"]}
    assert _gap(None, {"A"}) == {"lost": None, "tickers": []}
    assert _gap({"A"}, None) == {"lost": None, "tickers": []}
