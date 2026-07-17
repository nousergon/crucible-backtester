"""Tests for analysis/attractiveness_eval.py — the universe-board
attractiveness composite eval producer (config#1389 / #1392 / #1398).

Covers:
- date-clustered IC math (weeks-as-N, honest small-N t/p),
- per-pillar ICs + DeMiguel 1/N-shrunk suggested weights,
- trajectory forward-IC blocks (accruing vs ok),
- config#1398 counterfactual (top-N variants, sector-balanced allocation,
  live tech_score survivor gate, ex-post winner capture),
- producer-side conformance of the emitted artifact against the FROZEN
  cross-repo JSON Schema (nousergon_lib.contracts "attractiveness_eval") on BOTH
  the ok and insufficient_data paths — the crucible-evaluator consumer is
  built against exactly this shape,
- the read-only guarantee (no S3 writes, no live-config writes) as a static
  source guard,
- reporter always-emit wiring for attractiveness_eval.json.

No test touches real S3: the history frame and trajectory scores are
injected; research.db is a tmp sqlite fixture.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.attractiveness_eval import (
    COUNTERFACTUAL_TOP_NS,
    MIN_EVAL_DATES_T,
    SCHEMA_VERSION,
    SHRINKAGE_FULL_DATES,
    _sector_balanced_top_n,
    compute_attractiveness_eval,
    suggest_pillar_weights,
)
from nousergon_lib import contracts
from nousergon_lib.quant.horizons import DEFAULT_POLICY

REPO_ROOT = Path(__file__).resolve().parents[1]
# The schema now lives in nousergon_lib.contracts (config#1861 second-adoption
# lift); the producer validates the emitted artifact against that single
# authority, not a repo-local copy.

HORIZON = int(DEFAULT_POLICY.primary_horizon)
RET_COL = f"log_return_{HORIZON}d"
SPY_COL = f"log_spy_return_{HORIZON}d"

PILLARS = ("quality", "value", "momentum", "growth", "stewardship", "defensiveness")


# ── Fixture builders ─────────────────────────────────────────────────────────


def _tickers(n: int) -> list[str]:
    return [f"T{i:03d}" for i in range(n)]


def _make_research_db(
    tmp_path,
    dates: list[str],
    n_names: int = 100,
    pass_tickers: dict[str, list[str]] | None = None,
) -> str:
    """research.db with universe_returns (deterministic descending alpha:
    T000 has the highest realized forward alpha in every cycle) and, when
    ``pass_tickers`` is given, scanner_evaluations with quant_filter_pass."""
    db = tmp_path / "research.db"
    conn = sqlite3.connect(db)
    conn.execute(
        f"CREATE TABLE universe_returns (ticker TEXT, eval_date TEXT, "
        f"sector TEXT, {RET_COL} REAL, {SPY_COL} REAL)"
    )
    sectors = ["Tech", "Health", "Fin", "Energy"]
    for d in dates:
        for i, t in enumerate(_tickers(n_names)):
            alpha = 0.30 - i * 0.005  # strictly descending in i
            conn.execute(
                "INSERT INTO universe_returns VALUES (?,?,?,?,?)",
                (t, d, sectors[i % len(sectors)], alpha, 0.0),
            )
    if pass_tickers is not None:
        conn.execute(
            "CREATE TABLE scanner_evaluations (ticker TEXT, eval_date TEXT, "
            "quant_filter_pass INTEGER, tech_score REAL)"
        )
        for d in dates:
            passing = set(pass_tickers.get(d, []))
            for i, t in enumerate(_tickers(n_names)):
                conn.execute(
                    "INSERT INTO scanner_evaluations VALUES (?,?,?,?)",
                    (t, d, 1 if t in passing else 0, float(100 - i)),
                )
    conn.commit()
    conn.close()
    return str(db)


def _make_history(
    dates: list[str], n_names: int = 100, aligned: bool = True, seed: int = 7
) -> pd.DataFrame:
    """Attractiveness history frame mirroring the producer parquet columns
    (crucible-research scoring/attractiveness_history._HISTORY_COLS).
    ``aligned=True`` ranks attractiveness identically to realized alpha
    (T000 best) with mild noise so per-date Spearman is high-positive but the
    weekly IC series is non-degenerate (finite t-stat)."""
    rng = np.random.default_rng(seed)
    rows = []
    for di, d in enumerate(dates):
        for i, t in enumerate(_tickers(n_names)):
            base = (n_names - i) if aligned else float(i)
            noise = rng.normal(0, 3.0)
            score = base + noise + di * 0.1
            rows.append({
                "as_of": d,
                "ticker": t,
                "attractiveness_raw": score / n_names,
                "attractiveness_score": score,
                # quality tracks alpha (positive IC); value anti-tracks
                # (negative IC); the rest are noise.
                "quality": base + rng.normal(0, 5.0),
                "value": -base + rng.normal(0, 5.0),
                "momentum": rng.normal(50, 10.0),
                "growth": rng.normal(50, 10.0),
                "stewardship": rng.normal(50, 10.0),
                "defensiveness": rng.normal(50, 10.0),
                "sector": None,
                "industry": None,
            })
    return pd.DataFrame(rows)


def _trajectory_scores(dates: list[str], n_names: int = 100, seed: int = 11) -> dict:
    """{(eval_date, ticker): {pre_repricing_score, attr_slope_z}} — the shape
    end_to_end.load_historical_trajectory_scores injects. pre_repricing_score
    tracks realized alpha rank (positive IC); attr_slope_z is noise."""
    rng = np.random.default_rng(seed)
    out = {}
    for d in dates:
        for i, t in enumerate(_tickers(n_names)):
            out[(d, t)] = {
                "pre_repricing_score": (n_names - i) + rng.normal(0, 3.0),
                "attr_slope_z": rng.normal(0, 1.0),
            }
    return out


WEEKLY_DATES = ["2026-05-22", "2026-05-29", "2026-06-05", "2026-06-12"]


@pytest.fixture()
def ok_artifact(tmp_path):
    """A fully-populated ok-path artifact: 4 weekly cycles, aligned
    attractiveness, live survivor gate, trajectory scores."""
    passing = {d: _tickers(100)[40:50] for d in WEEKLY_DATES}  # mid-pack picks
    db = _make_research_db(tmp_path, WEEKLY_DATES, pass_tickers=passing)
    return compute_attractiveness_eval(
        db,
        as_of="2026-07-06",
        history_df=_make_history(WEEKLY_DATES),
        trajectory_scores=_trajectory_scores(WEEKLY_DATES),
    )


# ── Composite + pillar ICs ───────────────────────────────────────────────────


class TestCompositeIC:
    def test_ok_status_and_envelope(self, ok_artifact):
        art = ok_artifact
        assert art["status"] == "ok"
        assert art["schema_version"] == SCHEMA_VERSION
        assert art["as_of"] == "2026-07-06"
        assert art["horizon_days"] == HORIZON

    def test_date_clustered_ic_positive_with_weeks_as_n(self, ok_artifact):
        ic = ok_artifact["composite_ic"]
        assert ic["n_eval_dates"] == len(WEEKLY_DATES)
        assert ic["date_ic_mean"] is not None and ic["date_ic_mean"] > 0.8
        assert ic["date_ic_t"] is not None and ic["date_ic_t"] > 0
        assert ic["date_ic_p"] is not None and 0 <= ic["date_ic_p"] <= 1
        # pooled secondary present with the pooled row count
        assert ic["pooled_ic"] is not None and ic["pooled_ic"] > 0.8
        assert ic["n"] == len(WEEKLY_DATES) * 100

    def test_anti_aligned_score_reads_negative(self, tmp_path):
        db = _make_research_db(tmp_path, WEEKLY_DATES)
        art = compute_attractiveness_eval(
            db, as_of="2026-07-06",
            history_df=_make_history(WEEKLY_DATES, aligned=False),
        )
        assert art["composite_ic"]["date_ic_mean"] < -0.8

    def test_small_n_is_honest_never_fabricated(self, tmp_path):
        """Below MIN_EVAL_DATES_T weekly cross-sections: mean reported,
        t/p None, status insufficient_data with a reason."""
        dates = WEEKLY_DATES[: MIN_EVAL_DATES_T - 1]
        db = _make_research_db(tmp_path, dates)
        art = compute_attractiveness_eval(
            db, as_of="2026-07-06", history_df=_make_history(dates),
        )
        assert art["status"] == "insufficient_data"
        assert "reason" in art
        ic = art["composite_ic"]
        assert ic["n_eval_dates"] == len(dates)
        assert ic["date_ic_t"] is None and ic["date_ic_p"] is None
        assert ic["date_ic_mean"] is not None  # honest point estimate

    def test_pillar_ics_directionally_correct(self, ok_artifact):
        pillar_ic = ok_artifact["pillar_ic"]
        assert set(pillar_ic) == set(PILLARS)
        assert pillar_ic["quality"]["date_ic_mean"] > 0.5
        assert pillar_ic["value"]["date_ic_mean"] < -0.5
        for p in PILLARS:
            assert pillar_ic[p]["n_eval_dates"] == len(WEEKLY_DATES)


# ── Suggested weights + shrinkage ────────────────────────────────────────────


class TestSuggestedWeights:
    def test_small_n_collapses_to_1_over_n(self, ok_artifact):
        """4 eval dates < SHRINKAGE_FULL_DATES (8) → lambda 1.0 and the
        suggested weights ARE the 1/N prior, regardless of measured ICs."""
        w = ok_artifact["suggested_pillar_weights"]
        sh = ok_artifact["shrinkage"]
        assert sh == {"method": "demiguel_1overN", "lambda": 1.0,
                      "n_eval_dates": len(WEEKLY_DATES)}
        assert set(w) == set(PILLARS)
        for p in PILLARS:
            assert w[p] == pytest.approx(1.0 / len(PILLARS), abs=1e-3)

    def test_large_n_tilts_toward_positive_ic(self):
        pillar_ic = {
            "a": {"date_ic_mean": 0.30, "date_ic_p": 0.01, "n_eval_dates": 16},
            "b": {"date_ic_mean": 0.10, "date_ic_p": 0.20, "n_eval_dates": 16},
            "c": {"date_ic_mean": -0.20, "date_ic_p": 0.10, "n_eval_dates": 16},
        }
        w, sh = suggest_pillar_weights(pillar_ic)
        assert sh["lambda"] == pytest.approx(SHRINKAGE_FULL_DATES / 16)
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-3)
        assert w["a"] > w["b"] > w["c"]
        # negative-IC pillar floors at the shrunk prior, never below 0
        assert w["c"] == pytest.approx(sh["lambda"] / 3, abs=1e-3)

    def test_all_nonpositive_ics_degrade_to_prior(self):
        pillar_ic = {
            "a": {"date_ic_mean": -0.1, "date_ic_p": None, "n_eval_dates": 20},
            "b": {"date_ic_mean": None, "date_ic_p": None, "n_eval_dates": 0},
        }
        w, sh = suggest_pillar_weights(pillar_ic)
        assert sh["lambda"] == 1.0
        assert w == {"a": 0.5, "b": 0.5}

    def test_never_writes_live_config(self):
        """Static read-only guard: the producer must contain NO S3 write site
        and no optimizer-style apply path. The suggested weights live inside
        the eval artifact ONLY — the live-edge artifact
        (config/factor_attractiveness_weights.json) is never written; it is
        only NAMED in the module docstring, which this test tolerates by
        asserting on write CALLS, not mentions."""
        src = (REPO_ROOT / "analysis" / "attractiveness_eval.py").read_text()
        assert "put_object" not in src
        assert "upload_file" not in src
        assert "apply_result" not in src  # no optimizer-style apply path


# ── Trajectory forward-IC ────────────────────────────────────────────────────


class TestTrajectoryIC:
    def test_signals_measured_when_history_present(self, ok_artifact):
        traj = ok_artifact["trajectory_ic"]
        pre = traj["pre_repricing_score"]
        assert pre["status"] == "ok"
        assert pre["n_eval_dates"] == len(WEEKLY_DATES)
        assert pre["date_ic_mean"] > 0.8
        slope = traj["attr_slope_z"]
        assert slope["status"] == "ok"
        assert abs(slope["date_ic_mean"]) < 0.3  # noise signal ≈ 0

    def test_accruing_when_no_artifacts(self, tmp_path):
        db = _make_research_db(tmp_path, WEEKLY_DATES)
        art = compute_attractiveness_eval(
            db, as_of="2026-07-06", history_df=_make_history(WEEKLY_DATES),
            trajectory_scores=None,
        )
        for s in ("pre_repricing_score", "attr_slope_z"):
            block = art["trajectory_ic"][s]
            assert block["status"] == "accruing"
            assert block["n_eval_dates"] == 0
            assert block["date_ic_mean"] is None


# ── Counterfactual (config#1398) ─────────────────────────────────────────────


class TestCounterfactual:
    def test_live_gate_and_top_n(self, ok_artifact):
        cf = ok_artifact["counterfactual"]
        assert cf["n_cycles"] == len(WEEKLY_DATES)
        lg = cf["live_gate"]
        # survivors are ranks 40-49 of 100 → ex-post top-10 winners are ranks
        # 0-9 → zero overlap → capture 0; mean alpha = mean of ranks 40..49.
        assert lg["n_survivors"] == 10 * len(WEEKLY_DATES)
        assert lg["capture_rate"] == pytest.approx(0.0)
        expected_alpha = float(np.mean([0.30 - i * 0.005 for i in range(40, 50)]))
        assert lg["mean_alpha"] == pytest.approx(expected_alpha, abs=1e-6)

        variants = {(e["n"], e["sector_balanced"]): e for e in cf["top_n"]}
        assert set(variants) == {(n, b) for n in COUNTERFACTUAL_TOP_NS
                                 for b in (False, True)}
        top60 = variants[(60, False)]
        # aligned attractiveness (mild noise) → top-60 selection captures
        # nearly all ex-post top-60 winners and beats the live mid-pack gate
        assert top60["n_cycles"] == len(WEEKLY_DATES)
        assert top60["capture_rate"] > 0.85
        assert top60["mean_alpha"] > lg["mean_alpha"]
        # universe of 100 scored names can't fill a top-120/200 variant —
        # skipped honestly (no truncated pseudo-variant), nulls in the entry
        for n in (120, 200):
            for b in (False, True):
                assert variants[(n, b)]["n_cycles"] == 0
                assert variants[(n, b)]["capture_rate"] is None
                assert variants[(n, b)]["mean_alpha"] is None

    def test_sector_balanced_allocation_is_proportional(self):
        g = pd.DataFrame({
            "ticker": _tickers(80),
            "sector": ["Tech"] * 40 + ["Health"] * 20 + ["Fin"] * 20,
            "attractiveness_score": np.arange(80, dtype=float),
            "alpha": np.arange(80, dtype=float) / 100.0,
        })
        sel = _sector_balanced_top_n(g, "attractiveness_score", 20)
        assert len(sel) == 20
        counts = sel["sector"].value_counts()
        assert counts["Tech"] == 10
        assert counts["Health"] == 5
        assert counts["Fin"] == 5
        # within each sector the picks are that sector's best scores
        tech_best = g[g["sector"] == "Tech"].nlargest(10, "attractiveness_score")
        assert set(sel[sel["sector"] == "Tech"]["ticker"]) == set(tech_best["ticker"])

    def test_absent_scanner_table_yields_empty_block(self, tmp_path):
        db = _make_research_db(tmp_path, WEEKLY_DATES, pass_tickers=None)
        art = compute_attractiveness_eval(
            db, as_of="2026-07-06", history_df=_make_history(WEEKLY_DATES),
        )
        cf = art["counterfactual"]
        assert cf["n_cycles"] == 0
        assert cf["top_n"] == []
        assert cf["live_gate"]["n_survivors"] == 0
        assert cf["live_gate"]["capture_rate"] is None
        # the composite IC is still measured — counterfactual absence must
        # not blank the rest of the artifact
        assert art["composite_ic"]["date_ic_mean"] is not None


# ── Degraded-input paths ─────────────────────────────────────────────────────


class TestDegradedInputs:
    def test_missing_db(self, tmp_path):
        art = compute_attractiveness_eval(
            str(tmp_path / "nope.db"), as_of="2026-07-06",
            history_df=_make_history(WEEKLY_DATES),
        )
        assert art["status"] == "insufficient_data"
        assert "research.db" in art["reason"]

    def test_empty_history(self, tmp_path):
        db = _make_research_db(tmp_path, WEEKLY_DATES)
        art = compute_attractiveness_eval(
            db, as_of="2026-07-06", history_df=pd.DataFrame(),
        )
        assert art["status"] == "insufficient_data"
        assert "warm-up" in art["reason"]

    def test_history_schema_drift_raises(self, tmp_path):
        """A history parquet missing its contract columns is a producer-side
        schema break — RAISE (surfaces as tracker error), never guess."""
        db = _make_research_db(tmp_path, WEEKLY_DATES)
        bad = _make_history(WEEKLY_DATES).drop(columns=["attractiveness_score"])
        with pytest.raises(ValueError, match="schema authority"):
            compute_attractiveness_eval(db, as_of="2026-07-06", history_df=bad)

    def test_no_resolved_outcomes(self, tmp_path):
        """universe_returns rows exist but no forward outcome resolved yet."""
        db = tmp_path / "research.db"
        conn = sqlite3.connect(db)
        conn.execute(
            f"CREATE TABLE universe_returns (ticker TEXT, eval_date TEXT, "
            f"sector TEXT, {RET_COL} REAL, {SPY_COL} REAL)"
        )
        conn.execute("INSERT INTO universe_returns VALUES ('T000','2026-07-02','Tech',NULL,NULL)")
        conn.commit()
        conn.close()
        art = compute_attractiveness_eval(
            str(db), as_of="2026-07-06", history_df=_make_history(WEEKLY_DATES),
        )
        assert art["status"] == "insufficient_data"
        assert "realized forward outcomes" in art["reason"]


# ── FROZEN cross-repo schema conformance (producer side) ─────────────────────


jsonschema = pytest.importorskip(
    "jsonschema",
    reason="needs nousergon-lib[contracts] (jsonschema) for schema validation",
)


class TestSchemaConformance:
    def _validate(self, artifact: dict) -> None:
        schema = contracts.load_schema("attractiveness_eval")
        jsonschema.validate(instance=artifact, schema=schema)

    def test_schema_is_v2(self):
        schema = contracts.load_schema("attractiveness_eval")
        assert schema["properties"]["schema_version"]["const"] == 2
        assert contracts.SCHEMA_VERSIONS["attractiveness_eval"] == 2

    def test_ok_artifact_conforms(self, ok_artifact):
        assert ok_artifact["status"] == "ok"
        self._validate(ok_artifact)

    def test_insufficient_data_artifact_conforms(self, tmp_path):
        art = compute_attractiveness_eval(
            str(tmp_path / "nope.db"), as_of="2026-07-06",
        )
        assert art["status"] == "insufficient_data"
        self._validate(art)

    def test_accruing_trajectory_conforms(self, tmp_path):
        db = _make_research_db(tmp_path, WEEKLY_DATES)
        art = compute_attractiveness_eval(
            db, as_of="2026-07-06", history_df=_make_history(WEEKLY_DATES),
            trajectory_scores=None,
        )
        self._validate(art)

    def test_artifact_is_strict_json(self, ok_artifact):
        """No NaN/Infinity leaks — the artifact must serialize under strict
        JSON (allow_nan=False), matching what a non-Python consumer parses."""
        json.dumps(ok_artifact, allow_nan=False)


# ── Reporter wiring (always-emit) ────────────────────────────────────────────


class TestReporterWiring:
    def _save(self, tmp_path, payload):
        from reporter import save

        return save(
            report_md="# r", signal_quality={"status": "ok"}, score_analysis=[],
            run_date="2026-07-06", results_dir=str(tmp_path),
            attractiveness_eval=payload,
        )

    def test_always_emits_nonok_body(self, tmp_path):
        out = self._save(tmp_path, {"status": "insufficient_data",
                                    "schema_version": 2})
        f = out / "attractiveness_eval.json"
        assert f.exists(), "attractiveness_eval.json must always-emit a non-ok body"
        assert json.loads(f.read_text())["status"] == "insufficient_data"

    def test_none_is_not_written(self, tmp_path):
        out = self._save(tmp_path, None)
        assert not (out / "attractiveness_eval.json").exists()
