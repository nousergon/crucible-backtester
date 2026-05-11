"""Pins for `analysis.param_sweep._sort_sweep_df_skilled_risk`.

Skilled-risk sort order:
  1. sortino_ratio (primary, per evaluator-revamp-260506.md)
  2. total_alpha (tiebreaker / presentation; NEVER primary per
     [[alpha_vs_spy_is_presentation_not_gating]])
  3. (no Sharpe fallback — observability only)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.param_sweep import _sort_sweep_df_skilled_risk


def test_sortino_is_primary_sort_key():
    df = pd.DataFrame({
        "combo": ["A", "B", "C"],
        "sortino_ratio": [1.0, 2.0, 1.5],
        "total_alpha":   [0.10, 0.05, 0.20],
        "sharpe_ratio":  [3.0, 1.0, 2.0],
    })
    _sort_sweep_df_skilled_risk(df)
    assert df["combo"].tolist() == ["B", "C", "A"], (
        "Primary sort must be sortino_ratio descending (B=2.0 > C=1.5 > A=1.0); "
        "Sharpe must NOT win this contest"
    )


def test_alpha_fallback_when_sortino_absent():
    df = pd.DataFrame({
        "combo": ["A", "B", "C"],
        "total_alpha":   [0.10, 0.30, 0.20],
        "sharpe_ratio":  [3.0, 1.0, 2.0],
    })
    _sort_sweep_df_skilled_risk(df)
    assert df["combo"].tolist() == ["B", "C", "A"], (
        "When sortino absent, total_alpha is the next fallback. "
        "Sharpe still must NOT take over."
    )


def test_alpha_fallback_when_sortino_all_nan():
    df = pd.DataFrame({
        "combo": ["A", "B", "C"],
        "sortino_ratio": [np.nan, np.nan, np.nan],
        "total_alpha":   [0.10, 0.30, 0.20],
        "sharpe_ratio":  [3.0, 1.0, 2.0],
    })
    _sort_sweep_df_skilled_risk(df)
    assert df["combo"].tolist() == ["B", "C", "A"], (
        "All-NaN sortino → fall through to total_alpha, NOT sharpe"
    )


def test_no_sort_when_neither_sortino_nor_alpha_available():
    """Falls through silently — explicitly does NOT re-anchor on sharpe_ratio.
    Sub-optimal display order is preferable to misleading-by-Sharpe order."""
    df = pd.DataFrame({
        "combo": ["A", "B", "C"],
        "sharpe_ratio": [3.0, 1.0, 2.0],
    })
    _sort_sweep_df_skilled_risk(df)
    # Order should be unchanged (natural enumeration)
    assert df["combo"].tolist() == ["A", "B", "C"]


def test_empty_dataframe_is_safe():
    df = pd.DataFrame()
    _sort_sweep_df_skilled_risk(df)
    assert df.empty


def test_sortino_present_but_partial_nan_still_sorts_by_sortino():
    df = pd.DataFrame({
        "combo": ["A", "B", "C"],
        "sortino_ratio": [np.nan, 2.0, 1.0],
        "total_alpha":   [0.30, 0.05, 0.10],
    })
    _sort_sweep_df_skilled_risk(df)
    # B and C have real sortino; A has NaN. pandas sorts NaN last by default.
    assert df["combo"].tolist()[0] == "B", \
        "B has highest non-NaN sortino — should top the list"
    assert df["combo"].tolist()[-1] == "A", \
        "A has NaN sortino — should fall to the bottom"
