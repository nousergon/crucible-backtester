"""config#1456: signal_quality gates the report-card tile on the canonical 21d
horizon (not the retired 10d), and emits canonical accuracy_21d/avg_alpha_21d."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from analysis.signal_quality import compute_accuracy


def _frame(n: int, *, with_21d: bool) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    d = {
        "symbol": [f"T{i}" for i in range(n)],
        "score_date": pd.date_range("2026-05-01", periods=n, freq="D").astype(str),
        "score": rng.uniform(40, 90, n),
        # retired 10d outcome present but dead-valued in production
        "beat_spy_10d": rng.integers(0, 2, n).astype(float),
        "return_10d": rng.normal(size=n), "spy_10d_return": rng.normal(size=n),
        "beat_spy_5d": rng.integers(0, 2, n).astype(float),
        "return_5d": rng.normal(size=n), "spy_5d_return": rng.normal(size=n),
        # retired 30d columns present-but-dead (NaN), mirroring the live schema
        "beat_spy_30d": np.nan, "return_30d": np.nan, "spy_30d_return": np.nan,
    }
    if with_21d:
        d.update({
            "beat_spy_21d": rng.integers(0, 2, n).astype(float),
            "return_21d": rng.normal(size=n), "spy_21d_return": rng.normal(size=n),
        })
    return pd.DataFrame(d)


def test_gate_uses_canonical_21d_not_retired_10d():
    # 10d fully populated but NO 21d → tile must be insufficient (gate on 21d).
    res = compute_accuracy(_frame(50, with_21d=False), min_samples=30)
    assert res["status"] == "insufficient_data"


def test_emits_canonical_21d_metrics_when_resolved():
    res = compute_accuracy(_frame(50, with_21d=True), min_samples=30)
    assert res["status"] == "ok"
    overall = res["overall"]
    assert overall["accuracy_21d"] is not None
    assert overall["n_21d"] == 50
    assert "avg_alpha_21d" in overall and "precision_21d" in overall
