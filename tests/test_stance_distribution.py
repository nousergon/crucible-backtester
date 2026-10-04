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
- stance × stance_source contingency: the report names the pillar-vs-
  heuristic source mix (config#6068), and the alert carries it.
- n + share observability: an n-decline with flat shares stays
  interpretable as volume variation, not a mix shift (config#6068).
"""

from __future__ import annotations

import json
import os
from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from analysis import stance_distribution as sd


def _make_pred_response(stance_counts: dict[str, int], source: str | None = None) -> dict:
    """Build a fake predictions/{date}.json body matching the prod shape.

    ``source`` attaches ``stance_source`` to every entry (prod predictor
    #183 emits it on each prediction); ``None`` leaves the field absent
    (older-file / defensive-path case, bucketed under "unknown").
    """
    predictions = []
    for stance, n in stance_counts.items():
        for i in range(n):
            entry = {"ticker": f"{stance.upper()}_{i}", "stance": stance}
            if source is not None:
                entry["stance_source"] = source
            predictions.append(entry)
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
                                            "quality": 5, "catalyst": 5},
                                           source="heuristic"),
        "2026-04-24": None,  # treated as NoSuchKey by stub
        "2026-05-01": "__corrupt__",
        "2026-05-08": _make_pred_response({"momentum": 6, "value": 6,
                                            "quality": 6, "catalyst": 6},
                                           source="pillar"),
    }
    s3 = _make_s3_client(file_map)
    stats = sd._load_stance_counts(
        bucket="test-bucket",
        dates=[date(2026, 4, 17), date(2026, 4, 24),
               date(2026, 5, 1), date(2026, 5, 8)],
        s3_client=s3,
    )
    assert date(2026, 4, 17) in stats
    assert date(2026, 5, 8) in stats
    assert date(2026, 4, 24) not in stats
    assert date(2026, 5, 1) not in stats
    # Enriched shape (config#6068): counts unchanged, plus n + source split
    first = stats[date(2026, 4, 17)]
    assert first["counts"] == {"momentum": 5, "value": 5, "quality": 5, "catalyst": 5}
    assert first["n"] == 20
    assert first["source_totals"] == {"heuristic": 20}
    assert first["by_source"]["heuristic"]["momentum"] == 5
    assert stats[date(2026, 5, 8)]["source_totals"] == {"pillar": 24}


def test_source_contingency_split():
    """A pillar-vs-heuristic split is named in the report (config#6068)."""
    baseline = _healthy_distribution()
    file_map = {d: _make_pred_response(baseline, source="heuristic")
                for d in _FRIDAYS[:-1]}
    # Current week: quality runs hot through the pillar path (16 pillar
    # picks vs baseline mean 12 → z=+4 breach), the rest heuristic — a
    # breach whose source mix must be answerable from the report.
    pillar_part = _make_pred_response({"quality": 16}, source="pillar")
    heuristic_part = _make_pred_response(
        {"momentum": 6, "value": 8, "catalyst": 2}, source="heuristic")
    file_map["2026-05-15"] = {
        "predictions": pillar_part["predictions"] + heuristic_part["predictions"],
    }
    s3 = _make_s3_client(file_map)
    report = sd.compute_stance_distribution_drift(
        bucket="test-bucket", current_date="2026-05-15", s3_client=s3,
    )
    assert report["status"] == "fail"
    assert "quality" in report["failures"]
    contingency = report["current_source_contingency"]
    assert contingency["pillar"] == {"momentum": 0, "value": 0,
                                     "quality": 16, "catalyst": 0}
    assert contingency["heuristic"]["quality"] == 0
    assert report["source_totals"] == {"pillar": 16, "heuristic": 16}
    assert report["current_n"] == 32
    # Invariant: sum(source_totals) == sum(counts) <= n
    assert sum(report["source_totals"].values()) == sum(
        report["current_distribution"].values()) == 32


def test_n_decline_flat_shares_interpretable():
    """n-decline with flat shares reads as volume, not a mix shift (#6048)."""
    # 4-week baseline at 28 picks each, quality 21/28 = 75% share.
    baseline = {"momentum": 3, "value": 2, "quality": 21, "catalyst": 2}
    # Current week: 24 picks, quality 18/24 = 75% — count breaches the
    # band (21 vs mean, σ_floor=1.0 → z=-3) but share is identical.
    current = {"momentum": 2, "value": 2, "quality": 18, "catalyst": 2}
    file_map = {d: _make_pred_response(baseline, source="heuristic")
                for d in _FRIDAYS[:-1]}
    file_map["2026-05-15"] = _make_pred_response(current, source="heuristic")
    s3 = _make_s3_client(file_map)
    report = sd.compute_stance_distribution_drift(
        bucket="test-bucket", current_date="2026-05-15", s3_client=s3,
    )
    assert report["status"] == "fail"
    assert "quality" in report["failures"]
    # The count band says breach; n + share say the mix is unchanged.
    assert report["current_n"] == 24
    assert report["current_shares"]["quality"] == 0.75
    baseline_share = report["file_volume"]["2026-05-08"]["shares"]["quality"]
    assert baseline_share == 0.75
    assert report["file_volume"]["2026-05-15"]["n"] == 24
    assert report["file_volume"]["2026-04-17"]["n"] == 28


def test_missing_stance_source_buckets_unknown():
    """Entries without stance_source bucket under 'unknown' (defensive)."""
    file_map = {d: _make_pred_response(_healthy_distribution())
                for d in _FRIDAYS}
    s3 = _make_s3_client(file_map)
    report = sd.compute_stance_distribution_drift(
        bucket="test-bucket", current_date="2026-05-15", s3_client=s3,
    )
    assert report["status"] == "ok"
    assert report["source_totals"] == {"unknown": 32}
    assert report["current_source_contingency"]["unknown"]["quality"] == 12
    # Empty file → n=0, all-zero shares, no division-by-zero.
    empty_file = {d: _make_pred_response(_healthy_distribution())
                  for d in _FRIDAYS[:-1]}
    empty_file["2026-05-15"] = {"predictions": []}
    s3 = _make_s3_client(empty_file)
    report = sd.compute_stance_distribution_drift(
        bucket="test-bucket", current_date="2026-05-15", s3_client=s3,
    )
    assert report["current_n"] == 0
    assert report["current_shares"] == {s: 0.0 for s in sd.KNOWN_STANCES}


def test_alert_carries_source_mix_and_n(monkeypatch):
    """The FAIL alert names n, per-stance share, and the source mix."""
    monkeypatch.setenv("ALPHA_ENGINE_STANCE_DRIFT_ALERT_DISABLED", "0")
    baseline = _healthy_distribution()
    collapsed = {**baseline, "quality": 0, "value": baseline["value"] + 12}
    file_map = {d: _make_pred_response(baseline, source="heuristic")
                for d in _FRIDAYS[:-1]}
    file_map["2026-05-15"] = _make_pred_response(collapsed, source="heuristic")
    s3 = _make_s3_client(file_map)

    fake_result = MagicMock()
    fake_result.sns.ok = True
    fake_result.telegram.ok = True
    fake_result.any_ok = True

    with patch("ops_alerts.publish_ops_alert", return_value=fake_result) as mock_publish:
        report = sd.compute_stance_distribution_drift(
            bucket="test-bucket", current_date="2026-05-15", s3_client=s3,
        )

    assert report["status"] == "fail"
    call_msg = mock_publish.call_args.args[0]
    # Share + n are in the breach line: 0 of 32 picks, 0.0%.
    assert "(0.0% of 32 picks)" in call_msg
    assert "Source mix: heuristic 32 (100.0%) (32/32 classified)" in call_msg


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

    with patch("ops_alerts.publish_ops_alert", return_value=fake_result) as mock_publish:
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


# ── config-I7405 ────────────────────────────────────────────────────────────
# The Phase-5 acceptance criterion is a statement about the PILLAR-derived
# distribution. Measured 2026-08-15: across 2026-07-24, 07-31, 08-07, 08-13
# and 08-14, `stance_source == "pillar"` appears ZERO times — the live
# signals_envelope producer emits `sub_scores.qual = null` for all 903
# tickers, so classify_stance correctly falls back to the heuristic for every
# one. The alert nonetheless told every reader to investigate the pillar path.

class TestTheVerdictNamesWhatItMeasured:
    def test_a_zero_pillar_run_does_not_send_the_reader_to_the_pillar_path(self):
        s = sd._verdict_sentence({"heuristic": 23})
        assert "UNMEASURED" in s
        # The directive form must be absent; the explicit negation is what
        # replaces it, so a bare substring check would pass on the wrong text.
        assert "investigate classify_stance pillar-vs-heuristic path" not in s
        assert "do not investigate classify_stance's pillar branch" in s

    def test_it_names_the_observed_sources(self):
        s = sd._verdict_sentence({"heuristic": 23})
        assert "heuristic" in s

    def test_it_still_says_the_breach_is_real(self):
        """Relabelled, never silenced — momentum 1.5 -> 6 is a genuine move."""
        s = sd._verdict_sentence({"heuristic": 23})
        assert "real" in s

    def test_a_pillar_bearing_run_keeps_the_phase_5_sentence(self):
        s = sd._verdict_sentence({"pillar": 20, "heuristic": 3})
        assert "Phase 5 acceptance check" in s
        assert "investigate classify_stance pillar-vs-heuristic path" in s
        assert "UNMEASURED" not in s

    def test_one_single_pillar_entry_is_enough_to_make_it_measurable(self):
        assert sd._phase5_measurable({"pillar": 1, "heuristic": 22})
        assert not sd._phase5_measurable({"heuristic": 23})

    def test_an_empty_source_mix_is_unmeasured_not_measured(self):
        """No classified entries at all is the strongest form of 'unknown'."""
        assert not sd._phase5_measurable({})
        assert "UNMEASURED" in sd._verdict_sentence({})

    def test_the_pillar_literal_matches_the_predictor_contract(self):
        """A spelling drift here silently makes every run read as unmeasured."""
        assert sd.PILLAR_SOURCE == "pillar"


# ── alpha-engine-config-I10530: the current ISO week is never baseline ──────

def test_select_baseline_excludes_the_current_iso_week():
    """Daily Mon–Fri files: a Friday run's baseline is the 4 PRIOR weeks.

    Before I10530 the filter was only ``d < current``, so Thursday of the
    run's own ISO week was picked as "the most recent prior week" and the
    oldest real week fell off the window.
    """
    monday = date(2026, 9, 7)
    weekdays = [
        date.fromordinal(monday.toordinal() + 7 * week + day)
        for week in range(4) for day in range(5)
    ]  # Mon 09-07 .. Fri 10-02, ISO weeks 37-40
    picked = sd._select_baseline_dates(
        weekdays, current=date(2026, 10, 2), n_weeks=4,
    )
    # Three prior weeks exist (37, 38, 39); week 40 is the run's own.
    assert picked == [date(2026, 9, 11), date(2026, 9, 18), date(2026, 9, 25)]


def test_select_baseline_never_returns_a_date_in_the_current_week():
    """Including Thursday 2026-10-01 for a Friday 2026-10-02 run is the bug."""
    all_dates = [
        date(2026, 9, 4), date(2026, 9, 10), date(2026, 9, 11),
        date(2026, 9, 17), date(2026, 9, 18), date(2026, 9, 24),
        date(2026, 9, 25), date(2026, 9, 28), date(2026, 9, 29),
        date(2026, 9, 30), date(2026, 10, 1), date(2026, 10, 2),
    ]
    current = date(2026, 10, 2)
    picked = sd._select_baseline_dates(all_dates, current=current, n_weeks=4)
    assert picked == [date(2026, 9, 4), date(2026, 9, 11),
                      date(2026, 9, 18), date(2026, 9, 25)]
    current_week = current.isocalendar()[:2]
    assert all(d.isocalendar()[:2] != current_week for d in picked)


def test_select_baseline_excludes_current_week_across_iso_year_boundary():
    """ISO week keys compare as (year, week): week 1 of 2027 follows week 53."""
    all_dates = [
        date(2026, 12, 11), date(2026, 12, 18), date(2026, 12, 24),
        date(2026, 12, 31), date(2027, 1, 4), date(2027, 1, 7),
    ]
    picked = sd._select_baseline_dates(
        all_dates, current=date(2027, 1, 8), n_weeks=4,
    )
    assert picked == [date(2026, 12, 11), date(2026, 12, 18),
                      date(2026, 12, 24), date(2026, 12, 31)]


def _live_counts(momentum: int, value: int, quality: int) -> dict[str, int]:
    return {"momentum": momentum, "value": value, "quality": quality, "catalyst": 0}


def test_2026_10_02_live_shape_is_within_band_against_the_four_prior_weeks():
    """Replays the live stance counts of the 2026-10-02 run (I10530).

    Counts read from s3://alpha-engine-research/predictor/predictions/ on
    2026-10-04. Against the old window (09-11, 09-18, 09-25 and Thursday
    10-01) quality=27 scored z=2.60 and paged; against the four prior
    Fridays it is z=1.79, inside the unchanged ±2σ band.
    """
    file_map = {
        "2026-09-04": _make_pred_response(_live_counts(10, 5, 15), source="heuristic"),
        "2026-09-11": _make_pred_response(_live_counts(5, 5, 20), source="heuristic"),
        "2026-09-18": _make_pred_response(_live_counts(6, 6, 23), source="heuristic"),
        "2026-09-25": _make_pred_response(_live_counts(6, 6, 23), source="heuristic"),
        "2026-10-01": _make_pred_response(_live_counts(4, 7, 24), source="heuristic"),
        "2026-10-02": _make_pred_response(_live_counts(4, 4, 27), source="heuristic"),
    }
    report = sd.compute_stance_distribution_drift(
        bucket="test-bucket", current_date="2026-10-02",
        s3_client=_make_s3_client(file_map),
    )
    assert report["baseline_dates"] == [
        "2026-09-04", "2026-09-11", "2026-09-18", "2026-09-25",
    ]
    assert report["status"] == "ok", report["per_stance"]
    assert report["per_stance"]["quality"]["deviation"] == pytest.approx(1.788, abs=1e-3)


def test_verdict_sentence_no_longer_points_at_a_nonexistent_qual_half():
    """The heuristic-only verdict must not send readers to the qual half.

    The champion envelope is quant-only (signals.json 2026-10-02:
    sub_scores.qual null for 904/904 tickers), so "check the upstream
    signals producer's qual half" named work with no object (I10530).
    """
    sentence = sd._verdict_sentence({"heuristic": 35})
    assert "UNMEASURED" in sentence
    assert "Check the upstream signals producer" not in sentence
    assert "quant-only" in sentence
