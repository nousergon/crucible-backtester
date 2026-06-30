"""Regression guards for the canonical-horizon migration (config#1451 / #1452).

The canonical-alpha cutover retired the 10d/30d outcome horizons; the weight /
veto / stance optimizers were left querying the now-dead `beat_spy_10d` (weight,
veto) / `prediction_date` (stance) and starved silently for ~3 months. These
guards pin the fix: the optimizers consume the canonical 5d/21d outcome and
**fail loud** (WARN) on starvation/schema-drift instead of silently no-op'ing.
"""

from __future__ import annotations

import sqlite3
import tempfile

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from analysis import veto_analysis  # noqa: E402
from optimizer import stance_sizing_optimizer, weight_optimizer  # noqa: E402


def test_weight_optimizer_targets_canonical_horizon():
    # The retired horizons must NOT be the resolved-outcome gate.
    assert weight_optimizer._RESOLVED_OUTCOME == "beat_spy_21d"
    assert weight_optimizer._SKILL_TARGET[weight_optimizer._LONG_OUTCOME] == "log_alpha_21d"
    assert "10d" not in weight_optimizer._RESOLVED_OUTCOME


def test_weight_starves_loudly_on_missing_canonical_column(caplog):
    weight_optimizer.init_config({"weight_optimizer": {}})
    # A frame with the RETIRED column but not the canonical one.
    df = pd.DataFrame({
        "symbol": ["A"] * 40, "score_date": pd.date_range("2026-01-01", periods=40),
        "beat_spy_10d": [1.0] * 40, "quant_score": range(40), "qual_score": range(40),
    })
    with caplog.at_level("WARNING"):
        res = weight_optimizer.compute_weights(df, min_samples=20)
    assert res["status"] == "insufficient_data"
    assert any("STARVED" in r.message for r in caplog.records)


def test_veto_starves_loudly_on_missing_canonical_column(caplog):
    veto_analysis.init_config({})
    df = pd.DataFrame({"symbol": ["A"] * 40, "beat_spy_10d": [1.0] * 40})  # no beat_spy_21d
    with caplog.at_level("WARNING"):
        res = veto_analysis.analyze_veto_effectiveness(df, "bucket-unused")
    assert res["status"] == "insufficient_data"
    assert any("STARVED" in r.message for r in caplog.records)


def test_stance_query_uses_score_date_not_prediction_date():
    """The original bug: stance queried `prediction_date` (absent) → SQL error.
    A canonical-schema db must yield a clean status (never 'error')."""
    stance_sizing_optimizer.init_config({})
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    conn = sqlite3.connect(f.name)
    conn.execute("CREATE TABLE score_performance (score_date TEXT, symbol TEXT, "
                 "log_alpha_21d REAL, stance TEXT)")
    conn.executemany(
        "INSERT INTO score_performance VALUES (?,?,?,?)",
        [(f"2026-05-{d:02d}", f"S{d}", 0.01, "momentum") for d in range(1, 20)])
    conn.commit()
    conn.close()
    res = stance_sizing_optimizer.analyze(f.name)
    assert res["status"] != "error"  # the bug was status=error every run
