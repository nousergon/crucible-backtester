"""Tests for analysis.stance_distribution — Phase 5 acceptance check.

ROADMAP L1614. Mechanizes the "stance distribution within ±2σ of prior
4-week baseline" gate. Covers:

- Happy path (all stances within band → status ok, no alert).
- One stance breaches 2σ (status fail, alert fires once).
- σ_floor prevents alerts on tiny natural variation when σ=0.
- Insufficient baseline weeks (status insufficient_data).
- Missing current-date prediction (status insufficient_data).
- Malformed current_date (status error).
- _select_baseline_dates picks the most recent in each prior ISO week.
- _load_stance_counts skips missing/corrupted files with a WARN, not raises.
- Alert publish is opt-out via env var.
"""

from __future__ import annotations

import json
import os
from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from analysis import stance_distribution as sd


def _make_pred_response(stance_counts: dict[str, int]) -> dict:
    """Build a fake predictions/{date}.json body matching the prod shape."""
    predictions = []
    for stance, n in stance_counts.items():
        for i in range(n):
            predictions.append({"ticker": f"{stance.upper()}_{i}", "stance": stance})
    return {"predictions": predictions}


def _make_s3_client(file_map: dict[str, dict]) -> MagicMock:
    """Stub boto3 S3 client whose list_objects_v2 + get_object replay file_map.

    `file_map` keys are ISO date strings ("YYYY-MM-DD") → predictions
    dict (or `None` to simulate the key being absent from S3 entirely).
    """
    s3 = MagicMock()

    listed_keys = [
        f"predictor/predictions/{d}.json"
        for d, body in file_map.items() if body is not None
    ]

    def _list(**kwargs):
        return {
            "Contents": [{"Key": k} for k in listed_keys],
            "IsTruncated": False,
        }

    def _get(Bucket, Key):
        date_str = Key.split("/")[-1].replace(".json", "")
        body = file_map.get(date_str)
        if body is None:
            err_response = {"Error": {"Code": "NoSuchKey", "Message": "missing"}}
            raise ClientError(err_response, "GetObject")
        if body == "__corrupt__":
            mock_body = MagicMock()
            mock_body.read.return_value = b"{not json"
            return {"Body": mock_body}
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps(body).encode()
        return {"Body": mock_body}

    s3.list_objects_v2.side_effect = _list
    s3.get_object.side_effect = _get
    return s3


_FRIDAYS = [
    "2026-04-17", "2026-04-24", "2026-05-01", "2026-05-08", "2026-05-15",
]


def _healthy_distribution() -> dict[str, int]:
    return {"momentum": 10, "value": 8, "quality": 12, "catalyst": 2}


@pytest.fixture(autouse=True)
def disable_alert_publish(monkeypatch):
    """Default-disable real alert publishing to keep tests offline.

    Tests that explicitly want to verify the publish-call path opt back in.
    """
    monkeypatch.setenv("ALPHA_ENGINE_STANCE_DRIFT_ALERT_DISABLED", "1")


def test_happy_path_ok():
    """Steady 4-week baseline, this week matches → status ok, no failures."""
    file_map = {
        d: _make_pred_response(_healthy_distribution())
        for d in _FRIDAYS
    }
    s3 = _make_s3_client(file_map)
    report = sd.compute_stance_distribution_drift(
        bucket="test-bucket", current_date="2026-05-15", s3_client=s3,
    )
    assert report["status"] == "ok"
    assert report["failures"] == []
    assert report["n_baseline_weeks"] == 4
    assert report["current_distribution"] == _healthy_distribution()
    assert len(report["baseline_dates"]) == 4
    assert "2026-05-15" not in report["baseline_dates"]


def test_one_stance_breach_fires_fail():
    """Quality count collapses from 12 → 0 in current week → 2σ breach."""
    baseline = _healthy_distribution()
    collapsed = {**baseline, "quality": 0, "value": baseline["value"] + 12}
    file_map = {d: _make_pred_response(baseline) for d in _FRIDAYS[:-1]}
    file_map["2026-05-15"] = _make_pred_response(collapsed)
    s3 = _make_s3_client(file_map)
    report = sd.compute_stance_distribution_drift(
        bucket="test-bucket", current_date="2026-05-15", s3_client=s3,
    )
    assert report["status"] == "fail"
    assert "quality" in report["failures"]
    quality_info = report["per_stance"]["quality"]
    assert quality_info["current"] == 0
    assert quality_info["baseline_mean"] == 12.0
    # σ_floor=1.0 → effective_std≥1.0; 0 vs 12 is 12σ away
    assert abs(quality_info["deviation"]) >= 2.0


def test_sigma_floor_prevents_tiny_drift_alarm():
    """Baseline constant at catalyst=2; current=3 (Δ=1) must not fire under σ_floor=1.0."""
    steady = {"momentum": 10, "value": 8, "quality": 12, "catalyst": 2}
    current = {**steady, "catalyst": 3}
    file_map = {d: _make_pred_response(steady) for d in _FRIDAYS[:-1]}
    file_map["2026-05-15"] = _make_pred_response(current)
    s3 = _make_s3_client(file_map)
    report = sd.compute_stance_distribution_drift(
        bucket="test-bucket", current_date="2026-05-15", s3_client=s3,
    )
    assert report["status"] == "ok"
    catalyst_info = report["per_stance"]["catalyst"]
    assert catalyst_info["baseline_std"] == 0.0
    assert catalyst_info["effective_std"] == 1.0
    assert catalyst_info["deviation"] == 1.0


def test_insufficient_baseline_weeks():
    """Only 2 prior weeks of data → status insufficient_data (need ≥4)."""
    short = ["2026-05-01", "2026-05-08", "2026-05-15"]
    file_map = {d: _make_pred_response(_healthy_distribution()) for d in short}
    s3 = _make_s3_client(file_map)
    report = sd.compute_stance_distribution_drift(
        bucket="test-bucket", current_date="2026-05-15", s3_client=s3,
    )
    assert report["status"] == "insufficient_data"
    assert "baseline" in report["note"]


def test_missing_current_date_prediction():
    """current_date isn't in S3 → status insufficient_data."""
    file_map = {d: _make_pred_response(_healthy_distribution()) for d in _FRIDAYS[:-1]}
    s3 = _make_s3_client(file_map)
    report = sd.compute_stance_distribution_drift(
        bucket="test-bucket", current_date="2026-05-15", s3_client=s3,
    )
    assert report["status"] == "insufficient_data"
    assert "absent" in report["note"]


def test_malformed_current_date_returns_error():
    """Non-ISO current_date → status error, no S3 call."""
    s3 = MagicMock()
    report = sd.compute_stance_distribution_drift(
        bucket="test-bucket", current_date="not-a-date", s3_client=s3,
    )
    assert report["status"] == "error"
    s3.list_objects_v2.assert_not_called()


def test_select_baseline_picks_latest_per_iso_week():
    """Daily predictions Mon–Fri: picks the latest weekday per ISO week."""
    # Build 5 weekdays × 2 ISO weeks ending Fri 2026-04-17 + Fri 2026-04-24
    weekday_dates = [
        date(2026, 4, 13), date(2026, 4, 14), date(2026, 4, 15),
        date(2026, 4, 16), date(2026, 4, 17),  # ISO week 16
        date(2026, 4, 20), date(2026, 4, 21), date(2026, 4, 22),
        date(2026, 4, 23), date(2026, 4, 24),  # ISO week 17
    ]
    picked = sd._select_baseline_dates(
        weekday_dates, current=date(2026, 5, 1), n_weeks=2,
    )
    # Should pick the Friday of each prior ISO week
    assert picked == [date(2026, 4, 17), date(2026, 4, 24)]


def test_select_baseline_skips_current_and_later():
    """current and any future date in input must not appear in picks."""
    all_dates = [
        date(2026, 4, 17), date(2026, 4, 24), date(2026, 5, 1),
        date(2026, 5, 8), date(2026, 5, 15), date(2026, 5, 22),
    ]
    picked = sd._select_baseline_dates(
        all_dates, current=date(2026, 5, 15), n_weeks=4,
    )
    assert date(2026, 5, 15) not in picked
    assert date(2026, 5, 22) not in picked
    assert len(picked) == 4


def test_load_stance_counts_skips_missing_and_corrupt():
    """A missing or corrupt file should WARN but not raise; remaining dates load."""
    file_map = {
        "2026-04-17": _make_pred_response({"momentum": 5, "value": 5,
                                            "quality": 5, "catalyst": 5}),
        "2026-04-24": None,  # treated as NoSuchKey by stub
        "2026-05-01": "__corrupt__",
        "2026-05-08": _make_pred_response({"momentum": 6, "value": 6,
                                            "quality": 6, "catalyst": 6}),
    }
    s3 = _make_s3_client(file_map)
    counts = sd._load_stance_counts(
        bucket="test-bucket",
        dates=[date(2026, 4, 17), date(2026, 4, 24),
               date(2026, 5, 1), date(2026, 5, 8)],
        s3_client=s3,
    )
    assert date(2026, 4, 17) in counts
    assert date(2026, 5, 8) in counts
    assert date(2026, 4, 24) not in counts
    assert date(2026, 5, 1) not in counts


def test_alert_publish_called_on_fail(monkeypatch):
    """When publish is enabled, a FAIL status triggers a single alerts.publish."""
    monkeypatch.setenv("ALPHA_ENGINE_STANCE_DRIFT_ALERT_DISABLED", "0")
    baseline = _healthy_distribution()
    collapsed = {**baseline, "quality": 0, "value": baseline["value"] + 12}
    file_map = {d: _make_pred_response(baseline) for d in _FRIDAYS[:-1]}
    file_map["2026-05-15"] = _make_pred_response(collapsed)
    s3 = _make_s3_client(file_map)

    fake_result = MagicMock()
    fake_result.sns.ok = True
    fake_result.telegram.ok = True
    fake_result.any_ok = True

    with patch("nousergon_lib.alerts.publish", return_value=fake_result) as mock_publish:
        report = sd.compute_stance_distribution_drift(
            bucket="test-bucket", current_date="2026-05-15", s3_client=s3,
        )

    assert report["status"] == "fail"
    mock_publish.assert_called_once()
    call_msg = mock_publish.call_args.args[0]
    assert "Stance-distribution drift on 2026-05-15" in call_msg
    assert "quality" in call_msg


def test_alert_publish_skipped_on_ok():
    """Status ok → no alert publish."""
    file_map = {d: _make_pred_response(_healthy_distribution()) for d in _FRIDAYS}
    s3 = _make_s3_client(file_map)
    with patch("nousergon_lib.alerts.publish") as mock_publish:
        report = sd.compute_stance_distribution_drift(
            bucket="test-bucket", current_date="2026-05-15", s3_client=s3,
        )
    assert report["status"] == "ok"
    mock_publish.assert_not_called()


def test_alert_publish_swallows_import_error():
    """Lib pin <v0.21.0 → ImportError swallowed at WARN; report still returned."""
    baseline = _healthy_distribution()
    collapsed = {**baseline, "quality": 0, "value": baseline["value"] + 12}
    file_map = {d: _make_pred_response(baseline) for d in _FRIDAYS[:-1]}
    file_map["2026-05-15"] = _make_pred_response(collapsed)
    s3 = _make_s3_client(file_map)

    # Force ImportError on `from nousergon_lib import alerts`
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "nousergon_lib":
            raise ImportError("simulated old lib pin")
        return real_import(name, *args, **kwargs)

    os.environ.pop("ALPHA_ENGINE_STANCE_DRIFT_ALERT_DISABLED", None)
    try:
        with patch("builtins.__import__", side_effect=fake_import):
            report = sd.compute_stance_distribution_drift(
                bucket="test-bucket", current_date="2026-05-15", s3_client=s3,
            )
    finally:
        os.environ["ALPHA_ENGINE_STANCE_DRIFT_ALERT_DISABLED"] = "1"
    assert report["status"] == "fail"
    # Did not raise — best-effort swallow worked


def test_unknown_stance_in_predictions_is_ignored():
    """Predictor emitting an unknown stance label is ignored (not counted)."""
    file_map = {d: _make_pred_response(_healthy_distribution()) for d in _FRIDAYS[:-1]}
    # Current week has a junk stance label among the predictions
    current_body = _make_pred_response(_healthy_distribution())
    current_body["predictions"].append({"ticker": "JUNK", "stance": "neutral"})
    file_map["2026-05-15"] = current_body
    s3 = _make_s3_client(file_map)
    report = sd.compute_stance_distribution_drift(
        bucket="test-bucket", current_date="2026-05-15", s3_client=s3,
    )
    assert report["status"] == "ok"
    # Junk stance should NOT show up in current_distribution
    assert set(report["current_distribution"].keys()) == set(sd.KNOWN_STANCES)
