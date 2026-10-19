"""Tests for analysis/direction_calibration_study.py (Refs nousergon/alpha-engine-config#12388).

Pins the pre-registered mechanics: population filter, signed conviction, the
date-matched base rate, the smallest-qualifying t* rule (and its non-binding
flag), the no-look-ahead climatology, the confirmation guard, and the CLI.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from analysis import direction_calibration_study as dcs


def _synthetic(n_dates=60, n_sym=20, seed=0, skill=0.0, start="2026-07-06") -> pd.DataFrame:
    """Rows with y ~ Bernoulli(p_up + skill*(p_up-0.5)); direction = side of p_up."""
    rng = np.random.default_rng(seed)
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=n_dates)]
    rows = []
    for d in dates:
        for s in range(n_sym):
            p = float(rng.uniform(0.3, 0.7))
            q = float(np.clip(p + skill * (p - 0.5), 0.01, 0.99))
            y = rng.random() < q
            rows.append(
                {
                    "symbol": f"S{s}",
                    "prediction_date": d,
                    "predicted_direction": "UP" if p > 0.5 else "DOWN",
                    "prediction_confidence": abs(p - 0.5) * 2,
                    "p_up": p,
                    "actual_log_alpha": 0.01 if y else -0.01,
                }
            )
    return pd.DataFrame(rows)


def _make_db(path, rows):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE predictor_outcomes (id INTEGER PRIMARY KEY, symbol TEXT, prediction_date TEXT, "
        "predicted_direction TEXT, prediction_confidence REAL, p_up REAL, actual_log_alpha REAL, "
        "horizon_days INTEGER, model_version TEXT)"
    )
    conn.executemany(
        "INSERT INTO predictor_outcomes (symbol, prediction_date, predicted_direction, prediction_confidence, "
        "p_up, actual_log_alpha, horizon_days, model_version) VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


class TestPopulationAndSignedConviction:
    def test_filters(self, tmp_path):
        mv = dcs.STUDY_MODEL_VERSION
        rows = [
            ("KEEP_UP", "2026-07-07", "UP", 0.4, 0.7, 0.02, 21, mv),
            ("KEEP_DN", "2026-07-07", "DOWN", 0.4, 0.3, -0.02, 21, mv),
            ("FLAT", "2026-07-07", "FLAT", 0.1, 0.5, 0.0, 21, mv),
            ("H5", "2026-07-07", "UP", 0.4, 0.7, 0.02, 5, mv),
            ("UNRES", "2026-07-07", "UP", 0.4, 0.7, None, 21, mv),
            ("OLDMODEL", "2026-07-07", "UP", 0.4, 0.7, 0.02, 21, "other"),
            ("NULLMODEL", "2026-07-07", "UP", 0.4, 0.7, 0.02, 21, None),
            ("EARLY", "2026-07-02", "UP", 0.4, 0.7, 0.02, 21, mv),
            ("LATE", "2026-09-18", "UP", 0.4, 0.7, 0.02, 21, mv),
        ]
        db = str(tmp_path / "r.db")
        _make_db(db, rows)
        df = dcs.load_outcomes(db, "2026-07-06", "2026-09-17")
        assert sorted(df["symbol"]) == ["KEEP_DN", "KEEP_UP"]

    def test_null_p_up_dropped_and_counted(self):
        df = _synthetic(n_dates=3, n_sym=4)
        df.loc[0, "p_up"] = np.nan
        out = dcs.prepare_frame(df)
        assert len(out) == len(df) - 1
        assert out.attrs["n_dropped_null_p_up"] == 1

    def test_signed_conviction_and_hit(self):
        df = pd.DataFrame(
            {
                "symbol": ["A", "B", "C", "D"],
                "prediction_date": ["2026-07-07"] * 4,
                "predicted_direction": ["UP", "DOWN", "UP", "DOWN"],
                "prediction_confidence": [0.4, 0.4, 0.1, 0.1],
                "p_up": [0.7, 0.3, 0.4, 0.6],  # C and D point against their own p_up
                "actual_log_alpha": [0.01, -0.01, -0.01, 0.01],
            }
        )
        out = dcs.prepare_frame(df).set_index("symbol")
        assert out.loc["A", "c"] == pytest.approx(0.7)
        assert out.loc["B", "c"] == pytest.approx(0.7)
        assert out.loc["C", "c"] == pytest.approx(0.4)  # < 0.5: unsigned |p-0.5| would read 0.1 "confident"
        assert out.loc["D", "c"] == pytest.approx(0.4)
        assert out.loc["A", "hit"] == 1 and out.loc["B", "hit"] == 1
        assert out.loc["C", "hit"] == 0 and out.loc["D", "hit"] == 0

    def test_date_matched_base_rate(self):
        df = pd.DataFrame(
            {
                "symbol": list("ABCD"),
                "prediction_date": ["2026-07-07"] * 4,
                "predicted_direction": ["UP", "UP", "DOWN", "DOWN"],
                "prediction_confidence": [0.2] * 4,
                "p_up": [0.6, 0.6, 0.4, 0.4],
                "actual_log_alpha": [0.01, 0.01, 0.01, -0.01],  # 3 of 4 beat -> base 0.75
            }
        )
        out = dcs.prepare_frame(df)
        assert out.loc[out["predicted_direction"] == "UP", "base"].tolist() == [0.75, 0.75]
        assert out.loc[out["predicted_direction"] == "DOWN", "base"].tolist() == [0.25, 0.25]


class TestBlockBootstrap:
    def test_deterministic_and_brackets_point(self):
        rng = np.random.default_rng(1)
        ds = np.column_stack([np.full(40, 10.0), rng.binomial(10, 0.6, 40).astype(float)])
        f = lambda s: s[:, 1] / s[:, 0]  # noqa: E731
        a = dcs.block_bootstrap(ds, f, block_len=5, n_boot=300, seed=7)
        b = dcs.block_bootstrap(ds, f, block_len=5, n_boot=300, seed=7)
        assert a == b
        assert a["lo"] <= a["point"] <= a["hi"]

    def test_block_longer_than_series_is_clamped(self):
        ds = np.column_stack([np.full(6, 5.0), np.full(6, 3.0)])
        out = dcs.block_bootstrap(ds, lambda s: s[:, 1] / s[:, 0], block_len=21, n_boot=100)
        assert out["block_len"] == 6
        assert out["lo"] == pytest.approx(0.6) and out["hi"] == pytest.approx(0.6)

    def test_single_date_has_no_interval(self):
        out = dcs.block_bootstrap(np.array([[5.0, 3.0]]), lambda s: s[:, 1] / s[:, 0])
        assert out["lo"] is None and out["point"] == pytest.approx(0.6)


class TestThresholdSelection:
    def _rising(self):
        """UP precision rises with c (skill), so a high threshold qualifies before t=0 does."""
        rng = np.random.default_rng(3)
        n = 4000
        dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-07-06", periods=100)]
        c = rng.uniform(0.5, 0.9, n)
        y = rng.random(n) < (0.30 + 0.9 * (c - 0.5))  # P(beat) 0.30 .. 0.66
        raw = pd.DataFrame(
            {
                "symbol": [f"S{i}" for i in range(n)],
                "prediction_date": np.repeat(dates, n // 100),
                "predicted_direction": "UP",
                "prediction_confidence": (c - 0.5) * 2,
                "p_up": c,
                "actual_log_alpha": np.where(y, 0.01, -0.01),
            }
        )
        return dcs.prepare_frame(raw)

    def test_smallest_qualifying_threshold_is_binding_when_skill_is_in_the_tail(self):
        df = self._rising()
        sel = dcs.select_t_star(df, "UP")
        assert sel["rule"] == "smallest_qualifying"
        assert sel["t_star"] > 0.5
        assert sel["threshold_non_binding"] is False
        assert sel["discovery_point"]["lift"] >= dcs.MIN_LIFT
        assert sel["discovery_point"]["coverage"] >= dcs.MIN_COVERAGE
        # every smaller grid threshold must have failed the rule
        for row in sel["curve"]:
            if row["t"] < sel["t_star"]:
                assert row["lift"] is None or row["lift"] < dcs.MIN_LIFT or row["coverage"] < dcs.MIN_COVERAGE

    def test_non_binding_flag_when_every_call_already_qualifies(self):
        df = dcs.prepare_frame(_synthetic(n_dates=30, n_sym=40, seed=5, skill=0.0))
        df = df[df["predicted_direction"] == "UP"].copy()
        df["actual_log_alpha"] = 0.01  # all beat -> lift positive at t=0 against any base < 1
        df.loc[df.index[::2], "actual_log_alpha"] = -0.01
        df = dcs.prepare_frame(df.drop(columns=["y", "c", "hit", "base"]))
        df["base"] = 0.3  # force a clear lift at full coverage
        sel = dcs.select_t_star(df, "UP", grid=(0.0, 0.5, 0.6))
        assert sel["t_star"] == 0.0
        assert sel["threshold_non_binding"] is True

    def test_fallback_is_coverage_quantile_when_nothing_qualifies(self):
        df = dcs.prepare_frame(_synthetic(n_dates=40, n_sym=20, seed=2))
        df["base"] = 1.0  # impossible lift
        sel = dcs.select_t_star(df, "DOWN")
        assert sel["rule"] == "coverage_quantile_fallback"
        covered = (df.loc[df["predicted_direction"] == "DOWN", "c"] >= sel["t_star"]).mean()
        assert covered == pytest.approx(dcs.MIN_COVERAGE, abs=0.02)

    def test_no_calls_direction(self):
        df = dcs.prepare_frame(_synthetic(n_dates=5, n_sym=5))
        df = df[df["predicted_direction"] == "UP"]
        sel = dcs.select_t_star(df, "DOWN")
        assert sel["t_star"] is None and sel["rule"] == "no_calls"


class TestBuckets:
    def test_monotone_precision_gives_positive_spearman(self):
        n = 500
        c = np.linspace(0.5, 0.9, n)
        hit = (np.arange(n) % 100) < (np.linspace(30, 80, n)).astype(int)
        df = pd.DataFrame(
            {
                "prediction_date": "2026-07-07",
                "predicted_direction": "UP",
                "c": c,
                "hit": hit.astype(int),
                "base": 0.4,
            }
        )
        out = dcs.m2_buckets(df, "UP")
        assert len(out["buckets"]) == 5 and sum(b["n"] for b in out["buckets"]) == n
        assert out["spearman"] is not None and out["spearman"] > 0

    def test_frozen_edges_are_used_verbatim(self):
        df = dcs.prepare_frame(_synthetic(n_dates=10, n_sym=20))
        out = dcs.m2_buckets(df, "UP", edges=[0.5, 0.55, 0.6, 0.65])
        assert out["edges"] == [0.5, 0.55, 0.6, 0.65]


class TestClimatologyHasNoLookahead:
    def test_only_rows_at_least_lag_days_old(self):
        df = pd.DataFrame(
            {
                "prediction_date": ["2026-07-01", "2026-07-01", "2026-07-20", "2026-08-05", "2026-08-31"],
                "y": [1, 1, 0, 0, 0],
            }
        )
        clim = dcs.trailing_climatology(df, lag_days=30)
        assert np.isnan(clim.iloc[0]) and np.isnan(clim.iloc[1]) and np.isnan(clim.iloc[2])
        assert clim.iloc[3] == pytest.approx(1.0)  # 08-05 sees 07-01 (35d old) only
        assert clim.iloc[4] == pytest.approx(2 / 3)  # 08-31 cutoff is 08-01: sees 07-01 x2 and 07-20, not 08-05

    def test_cutoff_is_inclusive_of_exactly_lag_days(self):
        df = pd.DataFrame({"prediction_date": ["2026-07-01", "2026-07-31"], "y": [1, 0]})
        clim = dcs.trailing_climatology(df, lag_days=30)
        assert clim.iloc[1] == pytest.approx(1.0)


class TestCalibrationNull:
    def test_calibrated_data_is_not_flagged_and_miscalibrated_is(self):
        good = dcs.prepare_frame(_synthetic(n_dates=60, n_sym=25, seed=11))
        res = dcs.m3_calibration(good, n_sims=80, seed=1)
        assert res["ece10_null"]["ece_null_p95"] > 0
        assert res["ece_above_null_p95"] is False

        bad = good.copy()
        bad["p_up"] = np.clip((bad["p_up"] - 0.5) * 4 + 0.5, 0.02, 0.98)  # overconfident tails, same outcomes
        res_bad = dcs.m3_calibration(bad, n_sims=80, seed=1)
        assert res_bad["ece10"] > res["ece10"]
        assert res_bad["ece_above_null_p95"] is True
        assert res_bad["recalibration_trigger"] is True

    def test_stronger_shock_widens_the_null(self):
        df = dcs.prepare_frame(_synthetic(n_dates=60, n_sym=25, seed=4))
        res = dcs.ece_null(df["p_up"].to_numpy(), df["y"].to_numpy(), df["prediction_date"], n_sims=60)
        fixed = res["ece_null_p95_at_fixed_rho"]
        assert fixed["0.5"] > fixed["0.1"] > 0

    def test_level_offset_is_reported(self):
        raw = _synthetic(n_dates=30, n_sym=20, seed=8)
        raw["actual_log_alpha"] = -0.01  # nobody beats SPY: pure level miss
        df = dcs.prepare_frame(raw)
        res = dcs.m3_calibration(df, n_sims=20)
        assert res["mean_p_up_minus_base_rate"] > 0.3
        assert res["ece10_after_removing_level_offset"] < res["ece10"]


class TestRunStudy:
    def test_guard_blocks_confirmation_read_without_frozen_threshold(self):
        df = dcs.prepare_frame(_synthetic(n_dates=10, n_sym=10))
        with pytest.raises(ValueError, match="frozen"):
            dcs.run_study(df, dcs.StudyParams(start=dcs.CONFIRMATION_START, end=dcs.CONFIRMATION_END))
        out = dcs.run_study(
            df,
            dcs.StudyParams(start=dcs.CONFIRMATION_START, end=dcs.CONFIRMATION_END, t_star={"UP": 0.55, "DOWN": 0.55}, n_sims=20),
        )
        assert out["t_star_selection"]["UP"]["rule"] == "frozen_input"
        assert out["M1_precision_lift"]["UP"]["t"] == 0.55

    def test_discovery_run_is_json_serialisable_and_complete(self):
        df = dcs.prepare_frame(_synthetic(n_dates=50, n_sym=20, seed=9, skill=1.0))
        out = dcs.run_study(df, dcs.StudyParams(n_sims=20))
        text = json.dumps(out, default=float)
        for key in (
            "population",
            "t_star_selection",
            "M1_precision_lift",
            "M1_alternative_cuts_descriptive",
            "M2_conviction_buckets",
            "M3_calibration",
            "M4_context",
        ):
            assert key in json.loads(text)
        assert out["mode"] == "offline-shadow-only"
        assert out["population"]["n_dates"] == 50

    def test_skill_shows_positive_spread_and_ic(self):
        df = dcs.prepare_frame(_synthetic(n_dates=60, n_sym=30, seed=10, skill=1.5))
        df["predicted_alpha"] = df["p_up"] - 0.5
        out = dcs.run_study(df, dcs.StudyParams(n_sims=20))
        assert out["M4_context"]["spread_up_minus_down"]["point"] > 0
        assert out["M4_context"]["rank_ic"]["mean"] > 0


class TestPredictionsJoin:
    def test_join_reports_match_rate(self):
        df = dcs.prepare_frame(_synthetic(n_dates=2, n_sym=4))
        first = df["prediction_date"].iloc[0]
        payload = {"predictions": [{"ticker": s, "predicted_alpha": 0.01, "p_up_platt": 0.5, "calibration_basis": "calibrator"} for s in df["symbol"].unique()]}
        out = dcs.attach_predictions(df, lambda d: payload if d == first else None)
        assert out.attrs["join_match_rate"] == pytest.approx(0.5)
        assert out["predicted_alpha"].notna().sum() == 4

    def test_no_files_gives_nan_columns(self):
        df = dcs.prepare_frame(_synthetic(n_dates=2, n_sym=3))
        out = dcs.attach_predictions(df, lambda d: None)
        assert out["predicted_alpha"].isna().all() and out.attrs["join_match_rate"] == 0.0


class TestCli:
    def _fixture(self, tmp_path):
        syn = _synthetic(n_dates=40, n_sym=15, seed=12)
        rows = [
            (r.symbol, r.prediction_date, r.predicted_direction, r.prediction_confidence, r.p_up, r.actual_log_alpha, 21, dcs.STUDY_MODEL_VERSION)
            for r in syn.itertuples()
        ]
        db = str(tmp_path / "r.db")
        _make_db(db, rows)
        preds = tmp_path / "preds"
        preds.mkdir()
        for d, g in syn.groupby("prediction_date"):
            (preds / f"{d}.json").write_text(
                json.dumps({"predictions": [{"ticker": t, "predicted_alpha": p - 0.5, "p_up_platt": p, "calibration_basis": "calibrator"} for t, p in zip(g["symbol"], g["p_up"])]})
            )
        return db, str(preds)

    def test_cli_writes_json(self, tmp_path):
        db, preds = self._fixture(tmp_path)
        out = tmp_path / "o.json"
        rc = dcs.main(["--db", db, "--predictions-dir", preds, "--n-sims", "20", "--out", str(out)])
        assert rc == 0
        res = json.loads(out.read_text())
        assert res["population"]["predictions_join_match_rate"] == pytest.approx(1.0)
        assert res["window"] == {"start": dcs.BINARY_ERA_START, "end": dcs.DISCOVERY_END}

    def test_cli_refuses_confirmation_window_without_frozen_thresholds(self, tmp_path):
        db, preds = self._fixture(tmp_path)
        with pytest.raises(SystemExit) as exc:
            dcs.main(["--db", db, "--predictions-dir", preds, "--end", "2026-11-30"])
        assert exc.value.code == 2
