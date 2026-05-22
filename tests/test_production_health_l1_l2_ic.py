"""Tests for the per-L1 + L2 IC decomposition added to
``analysis.production_health.compute_production_health`` per ROADMAP
L135.

The decomposition reads per-L1 component values from
``predictor/predictions/{date}.json`` artifacts and joins them with
``predictor_outcomes`` rows in-memory — no schema migration to
``predictor_outcomes`` (the table stays at the L2-only contract per
PR #100's narrowed scope claim).

Three load-bearing properties pinned:

* per-L1 IC computed only when the L1 column is present AND has ≥10
  samples AND non-zero rank variance (the latter catches the
  research-calibrator's flat-region failure mode).
* ``l2_lift_vs_l1_mean`` is the diagnostic that decides whether the
  Ridge meta-learner is contributing alpha above ensemble averaging.
* Missing predictions artifacts degrade gracefully — every L1/L2 key
  is ``None`` and the rest of the production_health report still
  computes (we don't blackout the aggregate IC when the decomposition
  can't join).
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest


# ── Helpers ────────────────────────────────────────────────────────────────


def _outcomes_df(n: int = 12, base_date: str = "2026-05-12") -> pd.DataFrame:
    """Build an ``outcomes_df`` shape matching what
    ``compute_production_health`` constructs after its SQL load + the
    ``net_signal``/``actual``/``correct`` columns are added.
    """
    rows = []
    for i in range(n):
        rows.append(
            {
                "symbol": f"T{i}",
                "prediction_date": base_date,
                "actual": 0.01 * (i - n / 2),  # symmetric around zero
                "net_signal": 0.05 * (i - n / 2),
                "correct": 1 if (i >= n / 2) else 0,
            }
        )
    return pd.DataFrame(rows)


def _predictions_payload(n: int = 12) -> dict:
    """Realistic predictions/{date}.json shape — only the L1/L2 fields
    that ``_load_l1_predictions_for_dates`` reads matter here.
    """
    return {
        "predictions": [
            {
                "ticker": f"T{i}",
                "momentum_confirmation": 0.04 * (i - n / 2),
                "expected_move": 0.015 + 0.001 * i,
                "research_calibrator_prob": 0.5 + 0.02 * (i - n / 2),
                "predicted_alpha": 0.06 * (i - n / 2),
            }
            for i in range(n)
        ]
    }


def _stub_s3(payload_by_date: dict[str, dict]):
    """Return a fake boto3 client whose ``get_object`` returns the
    payload for the requested predictions/{date}.json key, raising
    ``KeyError`` on misses (mimics S3 NoSuchKey for absent dates).
    """
    class _Body:
        def __init__(self, raw: bytes):
            self._raw = raw

        def read(self) -> bytes:
            return self._raw

    class _Client:
        def get_object(self, *, Bucket: str, Key: str) -> dict:  # noqa: N803
            for d, payload in payload_by_date.items():
                if Key.endswith(f"/{d}.json"):
                    return {"Body": _Body(json.dumps(payload).encode())}
            raise KeyError(Key)

    return _Client()


# ── _load_l1_predictions_for_dates ──────────────────────────────────────────


def test_load_l1_predictions_returns_empty_on_no_dates():
    from analysis.production_health import _load_l1_predictions_for_dates

    df = _load_l1_predictions_for_dates("bucket", [])
    assert df.empty


def test_load_l1_predictions_yields_per_ticker_l1_l2_rows():
    from analysis.production_health import _load_l1_predictions_for_dates

    s3 = _stub_s3({"2026-05-12": _predictions_payload(n=5)})
    df = _load_l1_predictions_for_dates("bucket", ["2026-05-12"], s3_client=s3)

    assert len(df) == 5
    assert set(df.columns) >= {
        "symbol", "prediction_date",
        "momentum_confirmation", "expected_move",
        "research_calibrator_prob", "predicted_alpha",
    }
    # Symbols pulled from the payload
    assert set(df["symbol"]) == {f"T{i}" for i in range(5)}


def test_load_l1_predictions_skips_missing_artifacts():
    """Mixed-availability: one date has the artifact, another doesn't.
    Loader yields rows from the one that loaded; doesn't raise.
    """
    from analysis.production_health import _load_l1_predictions_for_dates

    s3 = _stub_s3({"2026-05-12": _predictions_payload(n=3)})
    df = _load_l1_predictions_for_dates(
        "bucket", ["2026-05-12", "2026-05-15"], s3_client=s3
    )
    # Only 2026-05-12 had an artifact
    assert len(df) == 3
    assert set(df["prediction_date"]) == {"2026-05-12"}


# ── _compute_l1_l2_ic_decomposition ─────────────────────────────────────────


def test_decomposition_empty_df_returns_all_none():
    from analysis.production_health import _compute_l1_l2_ic_decomposition

    result = _compute_l1_l2_ic_decomposition(pd.DataFrame(), bucket="bucket")
    assert result["per_l1"] == {
        "momentum": None, "volatility": None, "research_calibrator": None,
    }
    assert result["l2_alpha"] is None
    assert result["l2_lift_vs_l1_mean"] is None


def test_decomposition_no_predictions_artifact_returns_all_none():
    """outcomes_df has rows but no predictions/{date}.json artifact
    exists → join yields no L1 columns → every IC is None, aggregate
    IC path is unaffected.
    """
    from analysis.production_health import _compute_l1_l2_ic_decomposition

    df = _outcomes_df(n=15)
    fake_s3 = _stub_s3({})  # no payloads for any date
    with patch("analysis.production_health.boto3.client", return_value=fake_s3):
        result = _compute_l1_l2_ic_decomposition(df, bucket="bucket")

    assert result["per_l1"]["momentum"] is None
    assert result["l2_alpha"] is None
    assert result["l2_lift_vs_l1_mean"] is None
    assert result["n_joined"] == 0


def test_decomposition_happy_path_yields_per_l1_and_l2_ic():
    from analysis.production_health import _compute_l1_l2_ic_decomposition

    df = _outcomes_df(n=15)
    s3 = _stub_s3({"2026-05-12": _predictions_payload(n=15)})
    with patch("analysis.production_health.boto3.client", return_value=s3):
        result = _compute_l1_l2_ic_decomposition(df, bucket="bucket")

    # Momentum / research_calibrator / predicted_alpha all monotone in i and
    # so is `actual` in the outcomes fixture — Spearman == 1.0 for each.
    assert result["per_l1"]["momentum"] == 1.0
    assert result["per_l1"]["research_calibrator"] == 1.0
    assert result["l2_alpha"] == 1.0
    # `expected_move` is also monotone in i (0.015 + 0.001*i) so its
    # Spearman is 1.0 too — fixture's L1 mean is 1.0, L2 lift over mean is 0.
    assert result["per_l1"]["volatility"] == 1.0
    assert result["l2_lift_vs_l1_mean"] == 0.0
    assert result["n_joined"] == 15


def test_l2_lift_is_positive_when_stacker_beats_l1_average():
    """Construct a fixture where the L1 components are noisy but L2 is
    perfect — l2_lift_vs_l1_mean must be > 0.
    """
    from analysis.production_health import _compute_l1_l2_ic_decomposition

    n = 20
    df = pd.DataFrame(
        [
            {
                "symbol": f"T{i}",
                "prediction_date": "2026-05-12",
                "actual": float(i),
                "net_signal": 0.0,
                "correct": 1,
            }
            for i in range(n)
        ]
    )
    # L1 components: noisy (Spearman ~0.5 each); L2: perfect.
    payload = {
        "predictions": [
            {
                "ticker": f"T{i}",
                # Reversal on every other row → moderate Spearman
                "momentum_confirmation": float(i if i % 2 == 0 else (n - i)),
                "expected_move": float(i if i % 3 != 0 else (n - i)),
                "research_calibrator_prob": float(
                    i if i % 5 != 0 else (n - i)
                ),
                "predicted_alpha": float(i),  # perfectly aligned with actual
            }
            for i in range(n)
        ]
    }
    s3 = _stub_s3({"2026-05-12": payload})
    with patch("analysis.production_health.boto3.client", return_value=s3):
        result = _compute_l1_l2_ic_decomposition(df, bucket="bucket")

    assert result["l2_alpha"] == 1.0
    assert result["l2_lift_vs_l1_mean"] is not None
    assert result["l2_lift_vs_l1_mean"] > 0.0, (
        "L2 stacker is rank-perfect vs actual, L1 components are noisy; lift "
        "must be positive — diagnostic for 'meta-learner is contributing'."
    )


def test_decomposition_skips_l1_component_with_below_min_samples():
    """If one L1 column has < _MIN_SAMPLES non-null values, that
    component reports None — but the others still compute.
    """
    from analysis.production_health import _compute_l1_l2_ic_decomposition

    df = _outcomes_df(n=15)
    payload = _predictions_payload(n=15)
    # Null-out the volatility column on most rows
    for i, p in enumerate(payload["predictions"]):
        if i >= 5:
            p["expected_move"] = None
    s3 = _stub_s3({"2026-05-12": payload})
    with patch("analysis.production_health.boto3.client", return_value=s3):
        result = _compute_l1_l2_ic_decomposition(df, bucket="bucket")

    assert result["per_l1"]["volatility"] is None  # only 5 non-null < min 10
    assert result["per_l1"]["momentum"] == 1.0     # other L1s unaffected
    assert result["l2_alpha"] == 1.0


def test_decomposition_zero_variance_l1_returns_none():
    """A constant L1 (e.g. calibrator stuck at 0.5) has zero rank
    variance — Spearman is undefined; we report None rather than NaN.
    """
    from analysis.production_health import _compute_l1_l2_ic_decomposition

    df = _outcomes_df(n=15)
    payload = _predictions_payload(n=15)
    for p in payload["predictions"]:
        p["research_calibrator_prob"] = 0.5  # all flat
    s3 = _stub_s3({"2026-05-12": payload})
    with patch("analysis.production_health.boto3.client", return_value=s3):
        result = _compute_l1_l2_ic_decomposition(df, bucket="bucket")

    assert result["per_l1"]["research_calibrator"] is None
    # The other L1s + L2 are unaffected
    assert result["per_l1"]["momentum"] == 1.0
    assert result["l2_alpha"] == 1.0
