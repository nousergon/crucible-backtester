"""Tests for the funnel-stage lift measurement upgrade (config#967, 2026-06-22).

`_alpha_21d_log_lift` now emits three baselines so a funnel stage's edge reads on
a clean yardstick:
  * `lift`            — selected vs the full input pool (diluted; selected ⊆ base)
  * `lift_vs_rejected`— selected vs the rejected complement (un-diluted skill)
  * `sn_lift_vs_rejected` — selected vs rejected on the per-(eval_date,sector)
    NEUTRALIZED residual (removes sector/cycle tilt — pure within-peer skill)

The decisive adversarial test: a selection that is PURE sector-tilt (picks an
entire strong sector, no within-sector skill) must show positive raw `lift` but
`sn_lift_vs_rejected ≈ 0`.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.end_to_end import _alpha_21d_log_lift  # noqa: E402


def _frame(rows):
    return pd.DataFrame(rows)


def test_complement_is_sharper_than_pool_lift():
    # Within each (date, sector), the first 2 of 5 names are the winners (+0.02);
    # the rest lose (-0.01). Selected = the within-sector winners.
    rows = []
    for d in ("2026-01-02", "2026-01-09"):
        for sec, base in (("Tech", 0.05), ("Fin", -0.03)):  # a strong + a weak sector
            for i in range(5):
                a = base + (0.02 if i < 2 else -0.01)
                rows.append({
                    "eval_date": d, "sector": sec,
                    "log_return_21d": a, "log_spy_return_21d": 0.0,
                    "sel": i < 2,
                })
    df = _frame(rows)
    r = _alpha_21d_log_lift(df, df["sel"])
    # complement lift is un-diluted -> strictly larger than the pool lift
    assert r["lift_vs_rejected"] > r["lift"], r
    # sector-neutral isolates the +0.02 vs -0.01 = 0.03 within-group gap
    assert abs(r["sn_lift_vs_rejected"] - 0.03) < 1e-6, r
    assert r["n_rejected"] == 12 and r["sn_n_selected"] == 8, r


def test_pure_sector_tilt_collapses_under_sector_neutral():
    # Selection = the ENTIRE strong sector (Tech), none of the weak (Fin).
    # No within-sector skill — picks are just a sector bet. Raw lift is positive,
    # but the sector-neutral selected-vs-rejected must be ~0.
    rows = []
    for d in ("2026-01-02", "2026-01-09"):
        for sec, base in (("Tech", 0.05), ("Fin", -0.03)):
            for i in range(5):
                # within-sector noise symmetric around the sector base, no skill
                a = base + (0.01 if i % 2 == 0 else -0.01)
                rows.append({
                    "eval_date": d, "sector": sec,
                    "log_return_21d": a, "log_spy_return_21d": 0.0,
                    "sel": sec == "Tech",
                })
    df = _frame(rows)
    r = _alpha_21d_log_lift(df, df["sel"])
    # Raw complement lift is large (Tech beat Fin by ~0.08) — looks "skillful"...
    assert r["lift_vs_rejected"] > 0.05, r
    # ...but it's pure sector tilt: within-(date,sector) there is no edge.
    assert abs(r["sn_lift_vs_rejected"]) < 1e-6, r


def test_legacy_fields_unchanged_and_no_sector_column():
    # Back-compat: the original fields still present; sector-neutral block absent
    # when there is no sector column.
    rows = [
        {"log_return_21d": 0.02, "log_spy_return_21d": 0.0, "sel": True},
        {"log_return_21d": 0.04, "log_spy_return_21d": 0.0, "sel": True},
        {"log_return_21d": -0.01, "log_spy_return_21d": 0.0, "sel": False},
        {"log_return_21d": -0.03, "log_spy_return_21d": 0.0, "sel": False},
    ]
    df = _frame(rows)
    r = _alpha_21d_log_lift(df, df["sel"])
    assert set(("selected_avg", "baseline_avg", "lift", "n_selected", "n_baseline")) <= set(r)
    assert r["selected_avg"] == 0.03 and r["n_baseline"] == 4
    assert r["lift_vs_rejected"] == 0.05  # 0.03 - (-0.02)
    assert "sn_lift_vs_rejected" not in r  # no sector column -> no SN block


def test_returns_none_without_canonical_columns():
    df = _frame([{"return_5d": 0.01, "sel": True}])
    assert _alpha_21d_log_lift(df, df["sel"]) is None
