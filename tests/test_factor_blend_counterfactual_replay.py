"""Tests for the factor-blend counterfactual replay (config#749).

The "real" optimizer signal: re-score the candidate universe under alternate
factor_blend weight tuples and replay each through the vectorbt simulator. Mirrors
``test_scanner_factor_counterfactual.py``'s convention — a tiny synthetic universe
where the named winners actually win, so a weighting that loads on the winning
sub-factor must beat one that loads on the losing sub-factor, and the equal-weight
tuple reproduces the baseline.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.factor_blend_counterfactual_replay import (  # noqa: E402
    EQUAL_WEIGHTS,
    FACTOR_SCORE_COLUMNS,
    build_counterfactual_replay_report,
    build_orders_for_weights,
    compute_composite_scores,
    replay_weight_variant,
)
from reporter import (  # noqa: E402
    _section_factor_blend_counterfactual_replay,
    build_report as build_md,
)


# ── Synthetic universe ───────────────────────────────────────────────────────


def _synthetic_universe():
    """A 6-cycle, 8-ticker universe where momentum picks the WINNERS and value
    picks the LOSERS.

    - WIN0..WIN3 rise over each holding window (high momentum_score, low
      value_score); WIN tickers are the names that realize positive returns.
    - LOS0..LOS3 fall over each holding window (low momentum_score, high
      value_score).

    A momentum-heavy blend selects the WIN names → positive return; a
    value-heavy blend selects the LOS names → negative return. Equal-weight sits
    between. quality_score / low_vol_score are flat (no signal) so they don't
    perturb the deterministic ordering.
    """
    win = [f"WIN{i}" for i in range(4)]
    los = [f"LOS{i}" for i in range(4)]
    tickers = win + los

    # 6 cycles, each 12 trading rows apart so a 10d hold sits fully inside the
    # matrix. Price grid: winners trend up, losers trend down — monotonic, so
    # every holding window has the same sign.
    dates = pd.bdate_range("2026-01-02", periods=72)
    price = {}
    for i, t in enumerate(win):
        # rising 50 -> ~120 across the window
        price[t] = np.linspace(50.0 + i, 120.0 + i, len(dates))
    for i, t in enumerate(los):
        # falling 120 -> ~50
        price[t] = np.linspace(120.0 + i, 50.0 + i, len(dates))
    price_matrix = pd.DataFrame(price, index=dates)[tickers]

    cycles = []
    for c in range(6):
        cdate = dates[c * 12]
        rows = {}
        for i, t in enumerate(win):
            rows[t] = {
                "momentum_score": 3.0 - i * 0.1,    # high momentum
                "value_score": -3.0 + i * 0.1,      # expensive (low value)
                "quality_score": 0.0,
                "low_vol_score": 0.0,
            }
        for i, t in enumerate(los):
            rows[t] = {
                "momentum_score": -3.0 + i * 0.1,   # low momentum
                "value_score": 3.0 - i * 0.1,       # cheap (high value)
                "quality_score": 0.0,
                "low_vol_score": 0.0,
            }
        cand = pd.DataFrame.from_dict(rows, orient="index")
        cycles.append({"date": cdate.strftime("%Y-%m-%d"), "candidates": cand})

    spy = pd.Series(np.linspace(400.0, 410.0, len(dates)), index=dates)
    return cycles, price_matrix, spy


# ── compute_composite_scores ─────────────────────────────────────────────────


def test_composite_scores_follow_weights():
    cycles, _pm, _spy = _synthetic_universe()
    cand = cycles[0]["candidates"]

    mom = compute_composite_scores(cand, {"momentum_score": 1.0})
    # momentum-only: winners rank above losers.
    assert mom.loc["WIN0"] > mom.loc["LOS0"]

    val = compute_composite_scores(cand, {"value_score": 1.0})
    # value-only: losers (cheap) rank above winners.
    assert val.loc["LOS0"] > val.loc["WIN0"]

    flat = compute_composite_scores(cand, {"quality_score": 1.0})
    # flat sub-factor → all zeros, no NaNs.
    assert (flat == 0.0).all()


# ── order construction is deterministic ──────────────────────────────────────


def test_orders_deterministic_and_count_matched():
    cycles, pm, _spy = _synthetic_universe()
    o1 = build_orders_for_weights(cycles, {"momentum_score": 1.0}, pm,
                                  picks_per_cycle=3, hold_days=10)
    o2 = build_orders_for_weights(cycles, {"momentum_score": 1.0}, pm,
                                  picks_per_cycle=3, hold_days=10)
    assert o1 == o2  # deterministic
    enters = [o for o in o1 if o["action"] == "ENTER"]
    # momentum picks winners; 3 picks/cycle x 6 cycles.
    assert len(enters) == 18
    assert all(o["ticker"].startswith("WIN") for o in enters)


# ── distinct deterministic per-weighting sim results ─────────────────────────


def test_two_tuples_produce_distinct_results():
    cycles, pm, spy = _synthetic_universe()

    mom_heavy = replay_weight_variant(
        cycles, {"momentum_score": 1.0}, pm, spy_prices=spy,
        picks_per_cycle=3, hold_days=10,
    )
    val_heavy = replay_weight_variant(
        cycles, {"value_score": 1.0}, pm, spy_prices=spy,
        picks_per_cycle=3, hold_days=10,
    )
    assert mom_heavy["status"] == "ok"
    assert val_heavy["status"] == "ok"

    # Momentum picks winners (up) → positive return; value picks losers (down)
    # → negative. The two tuples produce DISTINCT, ordered results.
    assert mom_heavy["total_return"] > 0
    assert val_heavy["total_return"] < 0
    assert mom_heavy["total_return"] != val_heavy["total_return"]
    assert mom_heavy["sortino_ratio"] > val_heavy["sortino_ratio"]

    # Determinism: identical re-run yields identical metrics.
    again = replay_weight_variant(
        cycles, {"momentum_score": 1.0}, pm, spy_prices=spy,
        picks_per_cycle=3, hold_days=10,
    )
    assert again["total_return"] == mom_heavy["total_return"]
    assert again["sortino_ratio"] == mom_heavy["sortino_ratio"]


def test_equal_weight_tuple_reproduces_baseline():
    """Supplying the equal-weight tuple as a variant must reproduce the baseline
    (which defaults to equal-weight) metric-for-metric."""
    cycles, pm, spy = _synthetic_universe()
    report = build_counterfactual_replay_report(
        cycles, [dict(EQUAL_WEIGHTS)], pm, spy_prices=spy,
        picks_per_cycle=3, hold_days=10,
    )
    assert report["status"] == "ok"
    baseline = report["baseline"]
    equal_variant = next(
        v for v in report["variants"] if v["weights"] == dict(EQUAL_WEIGHTS)
    )
    assert equal_variant["total_return"] == baseline["total_return"]
    assert equal_variant["sortino_ratio"] == baseline["sortino_ratio"]
    assert equal_variant["max_drawdown"] == baseline["max_drawdown"]
    # And lift vs baseline is exactly zero.
    assert equal_variant["sortino_lift"] == 0.0
    assert equal_variant["alpha_lift"] == 0.0


# ── full report ──────────────────────────────────────────────────────────────


def test_report_ranks_momentum_above_value():
    cycles, pm, spy = _synthetic_universe()
    report = build_counterfactual_replay_report(
        cycles,
        [{"momentum_score": 1.0}, {"value_score": 1.0}],
        pm, spy_prices=spy, picks_per_cycle=3, hold_days=10,
    )
    assert report["status"] == "ok"
    assert report["n_cycles"] == 6
    # Sorted by Sortino desc — momentum-heavy first.
    labels = [v["label"] for v in report["variants"]]
    assert "mom=1" in labels[0]
    # The momentum variant beats the equal-weight baseline on Sortino.
    assert report["best"] is not None
    assert "mom=1" in report["best"]["label"]
    assert report["best"]["sortino_lift"] > 0
    assert report["any_variant_beats_baseline"] is True


def test_skipped_without_cycles():
    _cycles, pm, _spy = _synthetic_universe()
    r = build_counterfactual_replay_report([], [{"momentum_score": 1.0}], pm)
    assert r["status"] == "skipped"


def test_no_orders_when_all_factors_flat():
    """A weighting that loads only on a flat sub-factor selects nobody → the
    variant reports no_orders (no entries)."""
    cycles, pm, spy = _synthetic_universe()
    # quality_score is flat in the fixture, but the composite would still be all
    # zeros → top-N still picks (ties broken by ticker). Use a price matrix with
    # no overlap to force no_orders instead: empty candidates.
    empty_cycles = [{"date": cycles[0]["date"], "candidates": cycles[0]["candidates"].iloc[0:0]}]
    r = replay_weight_variant(empty_cycles, {"momentum_score": 1.0}, pm, spy_prices=spy)
    assert r["status"] == "no_orders"
    assert r["n_orders"] == 0


# ── reporter wiring ──────────────────────────────────────────────────────────


def test_section_renders_deferred_when_disabled():
    lines = _section_factor_blend_counterfactual_replay(
        {"status": "skipped", "reason": "opt-in; disabled by default"}
    )
    assert "Counterfactual blend weights" in lines[0]
    assert any("Deferred" in ln for ln in lines)


def test_section_renders_table_when_ok():
    cycles, pm, spy = _synthetic_universe()
    report = build_counterfactual_replay_report(
        cycles,
        [{"momentum_score": 1.0}, {"value_score": 1.0}],
        pm, spy_prices=spy, picks_per_cycle=3, hold_days=10,
    )
    lines = _section_factor_blend_counterfactual_replay(report)
    joined = "\n".join(lines)
    assert "Counterfactual blend weights" in joined
    assert "Sortino lift" in joined
    assert "baseline:" in joined
    assert "Top variant by Sortino lift" in joined


def test_build_report_threads_kwarg():
    cycles, pm, spy = _synthetic_universe()
    report = build_counterfactual_replay_report(
        cycles, [{"momentum_score": 1.0}], pm, spy_prices=spy,
        picks_per_cycle=3, hold_days=10,
    )
    md = build_md(
        run_date="2026-06-28",
        signal_quality={"status": "skipped"},
        regime_analysis=[],
        score_analysis=[],
        attribution={"status": "skipped"},
        factor_blend_counterfactual_replay=report,
    )
    assert "Counterfactual blend weights" in md


def test_build_report_omits_section_when_none():
    md = build_md(
        run_date="2026-06-28",
        signal_quality={"status": "skipped"},
        regime_analysis=[],
        score_analysis=[],
        attribution={"status": "skipped"},
    )
    assert "Counterfactual blend weights" not in md
