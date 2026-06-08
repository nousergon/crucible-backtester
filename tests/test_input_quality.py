"""
tests/test_input_quality.py — pre-spend signal input-quality gate (L4525).

Covers the pure assessor (assess_signal_quality) decision matrix and the
loader-backed gate (gate_signal_inputs) observe-vs-enforce behavior. No S3 —
the signal loader is a tiny in-memory fake.
"""

from __future__ import annotations

import logging

import pytest

from analysis.input_quality import (
    InputQualityError,
    InputQualityVerdict,
    assess_signal_quality,
    gate_signal_inputs,
)


def _signals(scores, *, universe=None):
    """Build a signals.json-shaped payload from a list of scores.

    A score of None means the entry omits the score field (schema-break case).
    """
    sig = {}
    for i, s in enumerate(scores):
        entry = {"ticker": f"T{i}", "rating": "BUY"}
        if s is not None:
            entry["score"] = s
        sig[f"T{i}"] = entry
    payload = {"signals": sig}
    if universe is not None:
        payload["universe"] = universe
    return payload


# ── assess_signal_quality: healthy ──────────────────────────────────────────


def test_healthy_spread_of_scores():
    per_date = {
        "2026-06-01": _signals([82, 71, 65, 58, 49]),
        "2026-06-08": _signals([90, 77, 60, 51]),
    }
    v = assess_signal_quality(per_date)
    assert v.healthy is True
    assert v.observations == []
    assert v.metrics["zero_score_entries"] == 0
    assert v.metrics["distinct_nonzero_scores"] >= 5


def test_quiet_but_healthy_week_is_not_garbage():
    """A quiet market still has healthy scores (just few ENTERs) — it must NOT
    trip the garbage gate. This is the false-positive case the enforce
    thresholds are designed to avoid."""
    per_date = {"2026-06-08": _signals([62, 58, 55, 51, 47])}
    v = assess_signal_quality(per_date)
    assert v.healthy is True


# ── assess_signal_quality: garbage (enforce-level) ──────────────────────────


def test_garbage_no_signals_anywhere():
    per_date = {
        "2026-06-01": {"signals": {}},
        "2026-06-08": {"signals": {}},
    }
    v = assess_signal_quality(per_date)
    assert v.healthy is False
    assert "no signals present" in v.reason


def test_garbage_wall_of_zero_scores():
    per_date = {
        "2026-06-01": _signals([0, 0, 0, 0, 0]),
        "2026-06-08": _signals([0, 0, 0, 0]),
    }
    v = assess_signal_quality(per_date)
    assert v.healthy is False
    assert "wall of Score 0.0" in v.reason
    assert v.metrics["zero_score_fraction"] == 1.0


def test_garbage_signals_present_but_no_numeric_score():
    """Schema break: entries exist but none carry a numeric score."""
    per_date = {"2026-06-08": _signals([None, None, None])}
    v = assess_signal_quality(per_date)
    assert v.healthy is False
    assert "NONE carry a numeric score" in v.reason


def test_garbage_no_dates_sampled():
    v = assess_signal_quality({})
    assert v.healthy is False
    assert "no signal dates" in v.reason


def test_non_numeric_score_not_counted_as_zero():
    """A non-numeric score is 'no score', NOT a zero — so a mix of healthy
    numeric scores plus some string scores stays healthy (the string ones
    simply don't count toward scored_entries)."""
    per_date = {"2026-06-08": _signals([80, 70, "n/a", 60])}
    v = assess_signal_quality(per_date)
    assert v.healthy is True
    assert v.metrics["scored_entries"] == 3
    assert v.metrics["zero_score_entries"] == 0


# ── assess_signal_quality: soft observations (healthy, logged not raised) ────


def test_elevated_but_not_total_zero_fraction_is_observation():
    # 4 of 10 scores are zero → 40% — above the 10% observe threshold but below
    # the 99% garbage threshold → healthy with an observation.
    per_date = {"2026-06-08": _signals([0, 0, 0, 0, 60, 70, 55, 65, 58, 62])}
    v = assess_signal_quality(per_date)
    assert v.healthy is True
    assert any("elevated zero-score fraction" in o for o in v.observations)


def test_low_coverage_is_observation():
    # 2 scored entries against a universe of 100 → 2% coverage < 25% threshold.
    per_date = {"2026-06-08": _signals([70, 60], universe=[f"U{i}" for i in range(100)])}
    v = assess_signal_quality(per_date)
    assert v.healthy is True
    assert any("low mean universe coverage" in o for o in v.observations)


def test_verdict_to_dict_is_serializable():
    v = assess_signal_quality({"2026-06-08": _signals([70, 60])})
    d = v.to_dict()
    assert set(d) == {"healthy", "reason", "metrics", "observations"}
    import json
    json.dumps(d)  # must not raise


# ── gate_signal_inputs: loader-backed observe vs enforce ────────────────────


class _FakeLoader:
    def __init__(self, by_date, missing=()):
        self._by_date = by_date
        self._missing = set(missing)
        self.load_calls = []

    def load(self, bucket, date):
        self.load_calls.append(date)
        if date in self._missing:
            raise FileNotFoundError(f"No signals at {date}")
        return self._by_date[date]


def test_gate_passes_on_healthy_inputs():
    loader = _FakeLoader({"2026-06-08": _signals([80, 70, 60])})
    v = gate_signal_inputs(
        "b", ["2026-06-08"], signal_loader=loader, enforce=True,
    )
    assert v.healthy is True


def test_gate_observe_mode_does_not_raise_on_garbage(caplog):
    loader = _FakeLoader({"2026-06-08": _signals([0, 0, 0])})
    with caplog.at_level(logging.WARNING, logger="analysis.input_quality"):
        v = gate_signal_inputs(
            "b", ["2026-06-08"], signal_loader=loader, enforce=False,
        )
    assert v.healthy is False
    assert any("DEGENERATE" in r.getMessage() for r in caplog.records)


def test_gate_enforce_mode_raises_on_garbage():
    loader = _FakeLoader({"2026-06-08": _signals([0, 0, 0])})
    with pytest.raises(InputQualityError, match="wall of Score 0.0"):
        gate_signal_inputs(
            "b", ["2026-06-08"], signal_loader=loader, enforce=True,
        )


def test_gate_samples_only_most_recent():
    by_date = {f"2026-05-{d:02d}": _signals([70, 60]) for d in range(1, 26)}
    loader = _FakeLoader(by_date)
    gate_signal_inputs(
        "b", sorted(by_date), signal_loader=loader, sample_recent=5,
    )
    assert len(loader.load_calls) == 5
    assert loader.load_calls == sorted(by_date)[-5:]


def test_gate_all_loads_missing_is_garbage():
    loader = _FakeLoader({}, missing=["2026-06-07", "2026-06-08"])
    with pytest.raises(InputQualityError, match="failed to load"):
        gate_signal_inputs(
            "b", ["2026-06-07", "2026-06-08"], signal_loader=loader, enforce=True,
        )


def test_gate_alert_publisher_called_on_garbage_best_effort():
    loader = _FakeLoader({"2026-06-08": _signals([0, 0, 0])})
    calls = []
    gate_signal_inputs(
        "b", ["2026-06-08"], signal_loader=loader, enforce=False,
        alert_publisher=lambda **kw: calls.append(kw),
    )
    assert len(calls) == 1
    assert calls[0]["severity"] == "warning"


def test_gate_alert_publisher_failure_does_not_break_gate():
    loader = _FakeLoader({"2026-06-08": _signals([0, 0, 0])})

    def _boom(**kw):
        raise RuntimeError("sns down")

    # enforce=False so the only way this raises is a leaked publisher error.
    v = gate_signal_inputs(
        "b", ["2026-06-08"], signal_loader=loader, enforce=False,
        alert_publisher=_boom,
    )
    assert v.healthy is False
