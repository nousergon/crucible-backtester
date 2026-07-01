"""Parity + contract tests for ``analysis.outcome_store`` — the config#1528
consumer cutover of the auto-apply optimizers onto the long-format
``score_performance_outcomes`` store (EPIC config#1483 Phase 3).

The acceptance bar (config#1528): the migrated read must produce output
IDENTICAL to the pre-migration wide-column read. The fixture builds BOTH
representations of the same ground truth exactly as the producers do —

  * wide ``score_performance`` columns: 2dp-rounded PERCENT returns
    (alpha-engine-data ``signal_returns`` Step 2: ``round(x * 100, 2)``);
  * long ``score_performance_outcomes`` rows: DECIMAL returns, canonical
    ``log_alpha`` on the primary horizon only (Step 2c);

— then asserts ``attach_outcomes`` reproduces the wide columns byte-identically
(same values, same NaN placement) and that the migrated optimizers' pure
computations are unchanged. The full-history replay against a copy of the live
research.db (the method proven on config#1483) is run at PR time; this fixture
pins the same invariants in CI permanently.
"""

from __future__ import annotations

import logging
import sqlite3

import numpy as np
import pandas as pd
import pytest
from nousergon_lib.quant.horizons import (
    DEFAULT_POLICY,
    PrimaryHorizonMissing,
)

from analysis.outcome_store import attach_outcomes, load_outcomes, store_exists

_PRIMARY = DEFAULT_POLICY.primary_horizon
_DIAG = DEFAULT_POLICY.diagnostic_horizons[0]

# Ground truth: (symbol, score_date, stance, ret5, spy5, ret21, spy21) decimals.
# Includes a negative-alpha name, a beat/no-beat mix, and one UNRESOLVED row.
_TRUTH = [
    ("AAPL", "2026-05-01", "momentum", 0.0213, 0.0100, 0.0432, 0.0201),
    ("MSFT", "2026-05-01", "quality", -0.0150, 0.0100, -0.0311, 0.0201),
    ("NVDA", "2026-05-08", "momentum", 0.0555, 0.0120, 0.1023, 0.0250),
    ("KO", "2026-05-08", "value", 0.0021, 0.0120, 0.0260, 0.0250),
    ("JPM", "2026-05-15", "catalyst", -0.0033, -0.0010, 0.0140, 0.0080),
    ("XOM", "2026-05-22", None, 0.0101, 0.0050, 0.0330, 0.0160),
    # unresolved: no outcomes in either representation
    ("TSLA", "2026-06-20", "momentum", None, None, None, None),
]


def _log_alpha(ret21: float, spy21: float) -> float:
    # The producer stores log-domain alpha: log1p(stock) - log1p(spy), 6dp.
    return round(float(np.log1p(ret21) - np.log1p(spy21)), 6)


@pytest.fixture()
def research_db(tmp_path):
    """A research.db carrying the SAME ground truth in both representations."""
    db = tmp_path / "research.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE score_performance (
            symbol TEXT, score_date TEXT, score REAL,
            quant_score REAL, qual_score REAL, stance TEXT,
            price_on_date REAL,
            return_5d REAL, spy_5d_return REAL, beat_spy_5d INTEGER,
            return_21d REAL, spy_21d_return REAL, beat_spy_21d INTEGER,
            log_alpha_21d REAL
        )"""
    )
    conn.execute(
        """CREATE TABLE score_performance_outcomes (
            id INTEGER PRIMARY KEY, signal_id TEXT NOT NULL,
            symbol TEXT NOT NULL, score_date TEXT NOT NULL,
            horizon_days INTEGER NOT NULL, beat_spy INTEGER,
            stock_return REAL, spy_return REAL, log_alpha REAL,
            is_primary INTEGER NOT NULL, resolved_at TEXT NOT NULL,
            schema_version INTEGER NOT NULL DEFAULT 1,
            UNIQUE(signal_id, horizon_days)
        )"""
    )
    rng = np.random.default_rng(7)
    for i, (sym, d, stance, r5, s5, r21, s21) in enumerate(_TRUTH):
        resolved = r5 is not None
        conn.execute(
            "INSERT INTO score_performance VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                sym, d, 60 + i, float(rng.uniform(40, 90)),
                float(rng.uniform(40, 90)), stance, 100.0,
                round(r5 * 100, 2) if resolved else None,
                round(s5 * 100, 2) if resolved else None,
                (1 if r5 > s5 else 0) if resolved else None,
                round(r21 * 100, 2) if resolved else None,
                round(s21 * 100, 2) if resolved else None,
                (1 if r21 > s21 else 0) if resolved else None,
                _log_alpha(r21, s21) if resolved else None,
            ),
        )
        if not resolved:
            continue
        for h, ret, spy in ((_DIAG, r5, s5), (_PRIMARY, r21, s21)):
            is_primary = h == _PRIMARY
            conn.execute(
                "INSERT INTO score_performance_outcomes "
                "(signal_id, symbol, score_date, horizon_days, beat_spy, "
                " stock_return, spy_return, log_alpha, is_primary, resolved_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    f"{sym}:{d}", sym, d, h, 1 if ret > spy else 0,
                    ret, spy,
                    _log_alpha(r21, s21) if is_primary else None,
                    1 if is_primary else 0, "2026-06-27T00:00:00+00:00",
                ),
            )
    conn.commit()
    conn.close()
    return str(db)


def _wide_df(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM score_performance ORDER BY score_date", conn,
            parse_dates=["score_date"],
        )
    finally:
        conn.close()
    return df


_OUTCOME_COLS = [
    c
    for h in DEFAULT_POLICY.all_horizons
    for oc in (DEFAULT_POLICY.outcome_columns(h),)
    for c in ([oc.beat_spy, oc.stock_return, oc.spy_return]
              + ([oc.log_alpha] if h == _PRIMARY else []))
]


# ── column-level parity: the config#1528 acceptance invariant ────────────────


def test_attach_outcomes_reproduces_wide_columns_exactly(research_db):
    """attach_outcomes must replace the wide-sourced outcome columns with
    long-store-sourced values that are BYTE-IDENTICAL (values + NaN placement)
    — the same invariant the full-history replay verifies against the live DB."""
    wide = _wide_df(research_db)
    attached = attach_outcomes(wide.copy(), research_db)

    assert list(attached["symbol"]) == list(wide["symbol"])
    for col in _OUTCOME_COLS:
        w = wide[col].astype("float64")
        a = attached[col].astype("float64")
        pd.testing.assert_series_equal(a, w, check_names=False, check_exact=True)

    # Non-outcome columns pass through untouched.
    for col in ("score", "quant_score", "qual_score", "stance", "price_on_date"):
        pd.testing.assert_series_equal(
            attached[col], wide[col], check_names=False
        )


def test_attach_outcomes_unresolved_rows_stay_nan(research_db):
    attached = attach_outcomes(_wide_df(research_db), research_db)
    tsla = attached[attached["symbol"] == "TSLA"]
    assert len(tsla) == 1
    for col in _OUTCOME_COLS:
        assert tsla[col].isna().all(), f"{col} should be NaN for unresolved row"


# ── consumer-level parity: migrated optimizers unchanged on both sources ─────


def test_weight_optimizer_correlations_identical(research_db):
    from optimizer import weight_optimizer

    wide = _wide_df(research_db)
    attached = attach_outcomes(wide.copy(), research_db)
    sub_cols = {"quant": "quant_score", "qual": "qual_score"}
    # Loosen the internal n>=10 floor by duplicating the fixture rows — the
    # comparison is wide-vs-long, so any deterministic expansion is fine.
    wide_big = pd.concat([wide] * 3, ignore_index=True)
    attached_big = pd.concat([attached] * 3, ignore_index=True)

    assert weight_optimizer._compute_correlations(
        wide_big, sub_cols
    ) == weight_optimizer._compute_correlations(attached_big, sub_cols)
    assert weight_optimizer._compute_ic_correlations(
        wide_big, sub_cols
    ) == weight_optimizer._compute_ic_correlations(attached_big, sub_cols)


def test_weight_optimizer_policy_constants_match_live_schema():
    """The policy-derived constants must reproduce the historical live map
    EXACTLY — this is the pin that a policy/schema drift breaks loudly."""
    from optimizer import weight_optimizer as w

    assert w._SHORT_OUTCOME == "beat_spy_5d"
    assert w._LONG_OUTCOME == "beat_spy_21d"
    assert w._RESOLVED_OUTCOME == "beat_spy_21d"
    assert w._SKILL_TARGET == {
        "beat_spy_5d": "return_5d",
        "beat_spy_21d": "log_alpha_21d",
    }


def test_veto_sweep_identical_on_both_sources(research_db):
    from analysis import veto_analysis as va

    wide = _wide_df(research_db)
    attached = attach_outcomes(wide.copy(), research_db)

    def down_frame(df):
        resolved = df[df[va._BEAT].notna()]
        return pd.DataFrame(
            {
                "prediction_confidence": np.linspace(0.5, 0.8, len(resolved)),
                va._BEAT: resolved[va._BEAT].astype(float).to_numpy(),
                va._RET: resolved[va._RET].astype(float).to_numpy(),
            }
        )

    base_rate = float(wide[va._BEAT].dropna().mean())
    sweep_wide = va._sweep_thresholds(down_frame(wide), base_rate, [0.5, 0.6, 0.7])
    sweep_long = va._sweep_thresholds(down_frame(attached), base_rate, [0.5, 0.6, 0.7])
    assert sweep_wide == sweep_long


def test_stance_sizing_reads_long_store(research_db):
    """stance_sizing now reads the long store directly; its per-stance mean
    alpha must equal the wide log_alpha_21d means (they are the same 6dp
    values by producer construction)."""
    from optimizer import stance_sizing_optimizer as sso

    sso.init_config({})
    result = sso.analyze(research_db)
    assert result["status"] == "ok"

    wide = _wide_df(research_db)
    resolved = wide[wide["log_alpha_21d"].notna() & wide["stance"].notna()]
    for stance, grp in resolved.groupby("stance"):
        got = result["per_stance"][stance]
        assert got["n"] == len(grp)
        assert got["mean_alpha"] == pytest.approx(
            round(float(grp["log_alpha_21d"].mean()), 6), abs=1e-9
        )


def test_significance_observe_canonical_horizon_is_policy_derived():
    from optimizer import significance_observe as so

    assert so._WEIGHT_CANONICAL_HORIZON == "log_alpha_21d"


def test_research_optimizer_corr_key_is_policy_derived():
    from optimizer import research_optimizer as ro

    assert ro._BEAT == "beat_spy_21d"
    assert ro._CORR_KEY == f"corr_{ro._BEAT}"  # the historical artifact key


# ── store accessor contract ──────────────────────────────────────────────────


def test_load_outcomes_units_are_decimal(research_db):
    long_df = load_outcomes(research_db)
    aapl21 = long_df[
        (long_df["symbol"] == "AAPL") & (long_df["horizon_days"] == _PRIMARY)
    ].iloc[0]
    assert aapl21["stock_return"] == pytest.approx(0.0432)
    assert aapl21["spy_return"] == pytest.approx(0.0201)
    assert aapl21["is_primary"] == 1
    diag = long_df[long_df["horizon_days"] == _DIAG]
    assert diag["log_alpha"].isna().all()


def test_load_outcomes_rejects_non_policy_horizon(research_db):
    with pytest.raises(ValueError, match="not in the active HorizonPolicy"):
        load_outcomes(research_db, horizons=(7,))


def test_load_outcomes_missing_table_is_graceful_and_loud(tmp_path, caplog):
    db = tmp_path / "empty.db"
    sqlite3.connect(db).close()
    with caplog.at_level(logging.WARNING):
        df = load_outcomes(str(db))
    assert df.empty
    assert any("long-format store not yet" in r.message for r in caplog.records)
    conn = sqlite3.connect(db)
    assert not store_exists(conn)
    conn.close()


def test_load_outcomes_fails_loud_on_missing_primary(research_db):
    conn = sqlite3.connect(research_db)
    conn.execute(
        "DELETE FROM score_performance_outcomes WHERE horizon_days = ?",
        (_PRIMARY,),
    )
    conn.commit()
    conn.close()
    with pytest.raises(PrimaryHorizonMissing):
        load_outcomes(research_db)


def test_attach_outcomes_warns_on_wide_long_divergence(research_db, caplog):
    """A row resolved in the wide columns but absent from the long store is a
    producer bug — the coverage guard must surface it loudly."""
    conn = sqlite3.connect(research_db)
    conn.execute(
        "DELETE FROM score_performance_outcomes WHERE symbol = 'NVDA'"
    )
    conn.commit()
    conn.close()
    with caplog.at_level(logging.WARNING):
        attached = attach_outcomes(_wide_df(research_db), research_db)
    assert any("outcome_store divergence" in r.message for r in caplog.records)
    nvda = attached[attached["symbol"] == "NVDA"]
    assert nvda["beat_spy_21d"].isna().all()  # missing → NaN, never fabricated
