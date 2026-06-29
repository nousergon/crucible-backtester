"""Tests for sector-constraint Sharpe replay (config#923).

Pure selection logic is tested directly; the full WITH-vs-WITHOUT comparison is
smoke-tested through the real vectorbt simulator on a synthetic universe where
the sector cap demonstrably changes the selection (and thus the metrics).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.sector_constraint_replay import (  # noqa: E402
    _verdict,
    build_orders,
    build_sector_balance_report,
    select_unconstrained,
    select_with_sector_cap,
)


def _picks(*specs):
    """specs: (ticker, sector, rank)."""
    return [{"ticker": t, "sector": s, "rank": r} for t, s, r in specs]


class TestSelection:
    def test_unconstrained_takes_top_n_by_rank(self):
        picks = _picks(("C", "Tech", 3), ("A", "Tech", 1), ("B", "Tech", 2))
        sel = select_unconstrained(picks, 2)
        assert [p["ticker"] for p in sel] == ["A", "B"]

    def test_cap_drops_overweight_sector(self):
        # 4 Tech + 1 Energy, cap 0.25 of 4 → max 1 per sector.
        picks = _picks(
            ("T1", "Tech", 1), ("T2", "Tech", 2), ("T3", "Tech", 3),
            ("E1", "Energy", 4), ("H1", "Health", 5),
        )
        sel = select_with_sector_cap(picks, picks_per_cycle=4, max_sector_pct=0.25)
        # max_per_sector = floor(0.25*4) = 1 → one Tech (best), then Energy, Health
        secs = [p["sector"] for p in sel]
        assert secs.count("Tech") == 1
        assert "Energy" in secs and "Health" in secs
        # best Tech kept is T1 (rank 1)
        assert any(p["ticker"] == "T1" for p in sel)
        assert not any(p["ticker"] in ("T2", "T3") for p in sel)

    def test_cap_higher_pct_allows_more(self):
        picks = _picks(
            ("T1", "Tech", 1), ("T2", "Tech", 2), ("T3", "Tech", 3),
            ("T4", "Tech", 4),
        )
        # cap 0.5 of 4 → max 2 Tech
        sel = select_with_sector_cap(picks, picks_per_cycle=4, max_sector_pct=0.5)
        assert len(sel) == 2

    def test_unknown_sector_bucket(self):
        picks = [{"ticker": "X", "rank": 1}, {"ticker": "Y", "rank": 2}]
        sel = select_with_sector_cap(picks, picks_per_cycle=4, max_sector_pct=0.25)
        # both fall in 'Unknown'; cap floor(0.25*4)=1 → only the best admitted
        assert len(sel) == 1 and sel[0]["ticker"] == "X"

    def test_zero_picks_per_cycle(self):
        picks = _picks(("A", "Tech", 1))
        assert select_unconstrained(picks, 0) == []
        assert select_with_sector_cap(picks, 0, 0.25) == []


class TestVerdict:
    def test_helps_hurts_neutral_inconclusive(self):
        assert _verdict(0.20) == "cap_helps"
        assert _verdict(-0.20) == "cap_hurts"
        assert _verdict(0.0) == "neutral"
        assert _verdict(None) == "inconclusive"


# ── Synthetic universe for the full replay ────────────────────────────────────


def _synthetic():
    """8 tickers, 2 sectors. Tech names FALL; Energy/Health names RISE.

    Unconstrained top-N (ranked Tech-first) loads up on the falling Tech sector →
    poor Sharpe. The sector cap forces in the rising Energy/Health names →
    different (and here, better) Sharpe. The point is that the two arms DIFFER;
    the harness then reports which way.
    """
    tech = [f"TECH{i}" for i in range(4)]
    other = ["ENER0", "ENER1", "HLTH0", "HLTH1"]
    tickers = tech + other
    dates = pd.bdate_range("2026-01-02", periods=72)
    price = {}
    for i, t in enumerate(tech):
        price[t] = np.linspace(120.0 + i, 60.0 + i, len(dates))   # falling
    for i, t in enumerate(other):
        price[t] = np.linspace(60.0 + i, 120.0 + i, len(dates))   # rising
    price_matrix = pd.DataFrame(price, index=dates)[tickers]

    cycles = []
    for c in range(5):
        d = dates[c * 12]
        # Tech ranked best (1-4) so unconstrained grabs them; rising names 5-8.
        picks = [
            {"ticker": tech[0], "sector": "Tech", "rank": 1},
            {"ticker": tech[1], "sector": "Tech", "rank": 2},
            {"ticker": tech[2], "sector": "Tech", "rank": 3},
            {"ticker": tech[3], "sector": "Tech", "rank": 4},
            {"ticker": "ENER0", "sector": "Energy", "rank": 5},
            {"ticker": "ENER1", "sector": "Energy", "rank": 6},
            {"ticker": "HLTH0", "sector": "Health", "rank": 7},
            {"ticker": "HLTH1", "sector": "Health", "rank": 8},
        ]
        cycles.append({"date": d.strftime("%Y-%m-%d"), "picks": picks})
    spy = pd.Series(np.linspace(100.0, 110.0, len(dates)), index=dates)
    return cycles, price_matrix, spy


class TestBuildOrders:
    def test_constrained_vs_unconstrained_differ(self):
        cycles, pm, _ = _synthetic()
        unc = build_orders(cycles, pm, constrained=False, picks_per_cycle=4)
        con = build_orders(cycles, pm, constrained=True, picks_per_cycle=4,
                           max_sector_pct=0.25)
        unc_tickers = {o["ticker"] for o in unc if o["action"] == "ENTER"}
        con_tickers = {o["ticker"] for o in con if o["action"] == "ENTER"}
        # unconstrained is all Tech; constrained pulls in rising names
        assert unc_tickers != con_tickers
        assert any(t.startswith("TECH") for t in unc_tickers)
        assert any(t in ("ENER0", "ENER1", "HLTH0", "HLTH1") for t in con_tickers)

    def test_empty_price_matrix(self):
        cycles, _, _ = _synthetic()
        assert build_orders(cycles, pd.DataFrame(), constrained=True) == []


class TestReport:
    def test_skipped_on_empty(self):
        out = build_sector_balance_report([], pd.DataFrame())
        assert out["status"] == "skipped"

    def test_full_replay_reports_sharpe_both_arms(self):
        cycles, pm, spy = _synthetic()
        out = build_sector_balance_report(
            cycles, pm, spy_prices=spy, picks_per_cycle=4, max_sector_pct=0.25,
            hold_days=10,
        )
        assert out["status"] == "ok"
        assert out["unconstrained"]["status"] == "ok"
        assert out["constrained"]["status"] == "ok"
        assert out["unconstrained"]["sharpe_ratio"] is not None
        assert out["constrained"]["sharpe_ratio"] is not None
        assert out["sharpe_delta"] is not None
        assert out["verdict"] in ("cap_helps", "cap_hurts", "neutral")
        # On this universe the cap pulls out falling Tech → constrained Sharpe
        # should beat unconstrained.
        assert out["constrained"]["sharpe_ratio"] > out["unconstrained"]["sharpe_ratio"]
