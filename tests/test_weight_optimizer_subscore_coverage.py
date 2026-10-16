"""alpha-engine-config-I7678 — a sub-score with no producer gets no weight.

Measured 2026-08-18 on ``s3://alpha-engine-research/signals/latest.json``
(903 rows, run_date 2026-08-14): ``qual_score`` non-null on 0/903 rows since
the six-team + CIO research graph was retired 2026-07-12 (config#1580). The
2026-08-14 apply audit nevertheless reported
``scoring_weights.current = {quant: 0.5, qual: 0.5}`` and a ninth consecutive
``blocked`` week — a loop optimising a blend that cannot exist, reported to
the Director as if the 50/50 vector were live.

Two properties are pinned here:

  1. a sub-score column that EXISTS but is null on every row is refused
     admission, and the loop reports ``subscore_absent`` → audit ``disabled``;
  2. the ``current_weights`` the audit renders is the EFFECTIVE vector
     (renormalised over populated sub-scores), never the configured one.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from optimizer import apply_audit
from optimizer.weight_optimizer import (
    _effective_weights,
    compute_weights,
    init_config,
    load_with_subscores,
)


def _config(default_weights: dict | None = None) -> None:
    init_config({
        "weight_optimizer": {
            "default_weights": default_weights or {"quant": 0.50, "qual": 0.50},
            "max_single_change": 0.10,
            "min_meaningful_change": 0.03,
            "blend_factor": 0.20,
            "confidence_low": 100,
            "confidence_medium": 300,
            "horizon_blend": {"beat_spy_21d": 0.50, "beat_spy_5d": 0.50},
            "blend_factor_min": 0.20,
            "blend_factor_max": 0.50,
            "blend_ramp_samples": 500,
        }
    })


def _df(n: int = 120, *, qual_null: bool = True) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "symbol": f"S{i % 20}",
            "score_date": f"2026-01-{(i % 28) + 1:02d}",
            "quant_score": 30.0 + (i % 60),
            "qual_score": None if qual_null else 40.0 + (i % 50),
            "beat_spy_21d": i % 2,
            "beat_spy_5d": (i + 1) % 2,
        })
    return pd.DataFrame(rows)


# ── 1. The live shape: qual null on every row ───────────────────────────────


class TestAbsentSubscoreIsRefused:

    def setup_method(self):
        _config()

    def test_all_null_qual_yields_subscore_absent_not_a_blend(self):
        result = compute_weights(_df(), min_samples=30)
        assert result["status"] == "subscore_absent"
        assert result["absent_subscores"] == ["qual"]
        assert result["subscore_coverage"]["qual"] == 0
        assert result["subscore_coverage"]["quant"] > 0
        # No proposal at all — the loop must not suggest a weight it cannot fit.
        assert "suggested_weights" not in result

    def test_reported_current_weights_are_effective_not_configured(self):
        """The audit's `current` field is the only rendering of the live
        weights the Director sees. It must read the composite that runs."""
        result = compute_weights(_df(), min_samples=30)
        assert result["current_weights"] == {"quant": 1.0, "qual": 0.0}
        assert result["configured_weights"] == {"quant": 0.50, "qual": 0.50}

    def test_note_names_the_absent_subscore(self):
        result = compute_weights(_df(), min_samples=30)
        assert "qual" in result["note"]
        assert "I7678" in result["note"]

    def test_populated_qual_still_fits_a_blend(self):
        """Guard-fails-without-the-fix control: with a real qual producer the
        loop behaves exactly as before, so this gate cannot mask a live blend."""
        result = compute_weights(_df(qual_null=False), min_samples=30)
        assert result["status"] == "ok"
        assert set(result["suggested_weights"]) == {"quant", "qual"}

    def test_missing_columns_entirely_still_reports_no_subscores(self):
        df = _df().drop(columns=["quant_score", "qual_score"])
        result = compute_weights(df, min_samples=30)
        assert result["status"] == "no_subscores"


# ── 2. The audit classification ─────────────────────────────────────────────


class TestAuditClassification:

    def test_subscore_absent_maps_to_disabled_with_no_guardrail_slug(self):
        record = apply_audit.classify_loop(
            "scoring_weights",
            {
                "status": "subscore_absent",
                "n_samples": 120,
                "current_weights": {"quant": 1.0, "qual": 0.0},
                "configured_weights": {"quant": 0.5, "qual": 0.5},
                "absent_subscores": ["qual"],
                "note": "qual null on all 120 rows",
            },

        )
        assert record["outcome"] == "disabled"
        assert record["blocked_by"] is None
        assert record["current"] == {"quant": 1.0, "qual": 0.0}
        assert record["proposed"] is None
        # The reason must reach the Director's surface, not only the log.
        assert record["detail"] == "qual null on all 120 rows"

    def test_disabled_neither_increments_nor_resets_the_blocked_streak(self):
        """Mirrors research_params' "retired" handling: a by-design-off loop
        must not accrue a `blocked` streak the report card escalates on."""
        n, _week = apply_audit._carry_forward(
            "disabled", {"consecutive_blocked_weeks": 9}, "2026-07-05",
        )
        assert n == 9


# ── 3. The renormaliser ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "configured,coverage,expected",
    [
        ({"quant": 0.5, "qual": 0.5}, {"quant": 903, "qual": 0}, {"quant": 1.0, "qual": 0.0}),
        ({"quant": 0.5, "qual": 0.5}, {"quant": 0, "qual": 903}, {"quant": 0.0, "qual": 1.0}),
        ({"quant": 0.5, "qual": 0.5}, {"quant": 10, "qual": 10}, {"quant": 0.5, "qual": 0.5}),
        ({"quant": 0.8, "qual": 0.2}, {"quant": 10, "qual": 0}, {"quant": 1.0, "qual": 0.0}),
        # Nothing populated: no vector can be claimed — all zeros, never a
        # plausible-looking split.
        ({"quant": 0.5, "qual": 0.5}, {"quant": 0, "qual": 0}, {"quant": 0.0, "qual": 0.0}),
    ],
)
def test_effective_weights_renormalise_over_populated_subscores(
    configured, coverage, expected,
):
    assert _effective_weights(configured, coverage) == expected


# ── 4. The detection blindness that hid it ──────────────────────────────────


@patch("optimizer.weight_optimizer.boto3")
def test_load_with_subscores_warns_when_a_subscore_is_null_on_every_row(
    mock_boto3, caplog,
):
    """`.any(axis=1)` counted a row as matched when EITHER half was present,
    so a sub-score with no producer at all logged as full coverage. The live
    signals.json shape (quant populated, qual null) must now WARN by name."""
    df = pd.DataFrame([
        {"symbol": "AAA", "score_date": "2026-08-14",
         "quant_score": None, "qual_score": None, "beat_spy_21d": 1},
        {"symbol": "BBB", "score_date": "2026-08-14",
         "quant_score": None, "qual_score": None, "beat_spy_21d": 0},
    ])
    signals = {
        "signals": {
            "AAA": {"sub_scores": {"quant": 91.5, "qual": None}},
            "BBB": {"sub_scores": {"quant": 40.0, "qual": None}},
        }
    }
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {
        "Body": MagicMock(read=lambda: json.dumps(signals).encode())
    }
    mock_boto3.client.return_value = mock_s3

    with caplog.at_level(logging.WARNING, logger="optimizer.weight_optimizer"):
        out = load_with_subscores(df, bucket="test-bucket")

    assert out["quant_score"].notna().sum() == 2
    assert out["qual_score"].notna().sum() == 0
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "qual_score" in messages
    assert "null on ALL" in messages
    assert "quant_score" not in messages
