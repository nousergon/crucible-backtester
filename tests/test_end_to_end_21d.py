"""Tests for the canonical 21d-horizon research-edge lift (ROADMAP L4551).

The selectors (scanner / sector teams / CIO) pick 21-day theses but were graded
on a 5-day window, collapsing precision toward the base rate. These tests build
a research.db where the 5d outcome is uncorrelated with selection while the 21d
outcome cleanly rewards the selected names, and assert the additive
``classification_21d`` + ``lift_21d_log`` blocks reflect the 21d edge.
"""

import sqlite3

import pytest

from analysis.end_to_end import compute_lift_metrics

DATE = "2026-05-01"


def _build_research_db(tmp_path):
    db = tmp_path / "research.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE universe_returns ("
        "ticker TEXT, eval_date TEXT, sector TEXT, "
        "return_5d REAL, spy_return_5d REAL, beat_spy_5d INTEGER, "
        "return_21d REAL, spy_return_21d REAL, beat_spy_21d INTEGER, "
        "log_return_21d REAL, log_spy_return_21d REAL)"
    )
    conn.execute("CREATE TABLE scanner_evaluations (ticker TEXT, eval_date TEXT, quant_filter_pass INTEGER)")
    conn.execute("CREATE TABLE team_candidates (ticker TEXT, eval_date TEXT, team_id TEXT, team_recommended INTEGER)")
    conn.execute("CREATE TABLE cio_evaluations (ticker TEXT, eval_date TEXT, cio_decision TEXT, final_score REAL, cio_conviction REAL, combined_score REAL, macro_shift REAL)")
    # Empty — _predictor_lift queries it; the read must not error on a missing table.
    conn.execute("CREATE TABLE predictor_outcomes (symbol TEXT, prediction_date TEXT, "
                 "predicted_direction TEXT, prediction_confidence REAL)")

    # 20 names. Selected = first 10. 5d beat alternates (uncorrelated w/ select);
    # 21d beat = 1 iff selected (clean 21d edge). 21d log alpha +0.05 selected,
    # -0.02 unselected.
    for i in range(20):
        t = f"T{i:02d}"
        selected = i < 10
        beat_5d = i % 2  # 0/1 alternating, independent of `selected`
        beat_21d = 1 if selected else 0
        log_ret = (0.05 if selected else -0.02)
        conn.execute(
            "INSERT INTO universe_returns VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (t, DATE, "Tech", 1.0, 0.5, beat_5d, 2.0, 1.0, beat_21d, log_ret, 0.0),
        )
        conn.execute("INSERT INTO scanner_evaluations VALUES (?,?,?)", (t, DATE, 1 if selected else 0))
        conn.execute("INSERT INTO team_candidates VALUES (?,?,?,?)", (t, DATE, "tech", 1 if selected else 0))
        conn.execute(
            "INSERT INTO cio_evaluations VALUES (?,?,?,?,?,?,?)",
            (t, DATE, "ADVANCE" if selected else "REJECT", 70.0 if selected else 40.0,
             75.0 if selected else 45.0, 68.0 if selected else 42.0, 2.0 if selected else -2.0),
        )
    conn.commit()
    conn.close()
    return str(db)


def test_scanner_21d_block_present_and_perfect(tmp_path):
    out = compute_lift_metrics(_build_research_db(tmp_path))
    sl = out["scanner_lift"]
    # Legacy 5d precision is at the base rate (selection ⊥ 5d outcome) ~0.5...
    assert sl["classification"]["precision"] == 0.5
    # ...but the 21d classification reflects the real edge: every selected name
    # beat at 21d → precision 1.0.
    assert sl["classification_21d"]["precision"] == 1.0
    assert sl["lift_21d_log"]["lift"] > 0  # selected 21d alpha > universe


def test_scanner_lift_labels_retired_baseline_arm(tmp_path):
    # config#2318: scanner_lift replays scanner_evaluations.quant_filter_pass,
    # which is the retired tech_score gate post the 2026-06-29 champion-feed
    # cutover — it must carry an explicit `arm` label so report-card/Director
    # consumers cannot present it as the live scanner unlabeled.
    out = compute_lift_metrics(_build_research_db(tmp_path))
    sl = out["scanner_lift"]
    assert sl["arm"] == "tech_score_baseline (retired from live feed 2026-06-29)"


def test_cio_21d_block_present(tmp_path):
    out = compute_lift_metrics(_build_research_db(tmp_path))
    cl = out["cio_lift"]
    assert cl["classification"]["precision"] == 0.5
    assert cl["classification_21d"]["precision"] == 1.0


def test_cio_selection_skill_block(tmp_path):
    # Fixture: ADVANCE names have +0.05 21d log-alpha, REJECT -0.02 → positive
    # selection gap; conviction (75 vs 45) tracks alpha → positive IC. (L4561)
    out = compute_lift_metrics(_build_research_db(tmp_path))
    sel = out["cio_lift"]["selection_skill_21d"]
    assert sel is not None
    assert sel["advance_alpha_21d"] == pytest.approx(0.05)
    assert sel["reject_alpha_21d"] == pytest.approx(-0.02)
    assert sel["selection_gap_21d"] == pytest.approx(0.07)
    assert sel["n_advance"] == 10 and sel["n_reject"] == 10
    assert sel["conviction_ic_21d"] is not None and sel["conviction_ic_21d"] > 0


def test_cio_layer_attribution_block(tmp_path):
    # Each orchestrated layer (combined_score, macro_shift, final_score,
    # cio_conviction) gets a rank-IC vs realized 21d alpha. In the fixture all
    # track the selected/+alpha split, so each IC is present (and positive). (L4561)
    out = compute_lift_metrics(_build_research_db(tmp_path))
    attr = out["cio_lift"]["layer_attribution_21d"]
    assert attr is not None and attr["n"] == 20
    for layer in ("combined_score", "macro_shift", "final_score", "cio_conviction"):
        assert attr[f"{layer}_ic"] is not None
    # L4563 de-blending substrate: cross-sectional rank-normalized stock-score IC.
    assert attr["combined_score_xs_rank_ic"] is not None
    # L4564 de-blending substrate: sector-neutral (trailing-baseline) stock-score
    # IC. Single-date fixture has no PRIOR cycle, so every row cold-starts to the
    # pool-wide fallback → key present, frac neutralized == 0.0.
    assert attr["combined_score_sector_neutral_ic"] is not None
    assert attr["combined_score_sector_neutral_frac"] == 0.0
    assert attr["combined_score_sector_neutral_n"] == 20
    # Single-date fixture: the date-clustered (Grinold-Kahn) block needs >= 3 dates,
    # so each layer's date-IC is None but the keys are present and n_eval_dates == 1.
    assert attr["n_eval_dates"] == 1
    for layer in ("combined_score", "macro_shift", "final_score", "cio_conviction"):
        assert attr[f"{layer}_date_ic"] is None
        assert attr[f"{layer}_date_ic_n"] == 0


def test_cio_layer_attribution_date_clustered_block(tmp_path):
    # De-pseudo-replication guard: across 4 eval_dates where combined_score tracks
    # realized 21d alpha cross-sectionally EVERY date, the Grinold-Kahn estimator
    # (mean of per-date ICs, t-test across the 4 dates) yields a positive,
    # significant date-IC — and n_eval_dates is the honest effective N (4), NOT the
    # pooled row count. This is the metric that replaces the pseudo-replicated pooled
    # p in the report-card composite-IC grade (config#1164).
    db = tmp_path / "research.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE universe_returns ("
        "ticker TEXT, eval_date TEXT, sector TEXT, "
        "return_5d REAL, spy_return_5d REAL, beat_spy_5d INTEGER, "
        "return_21d REAL, spy_return_21d REAL, beat_spy_21d INTEGER, "
        "log_return_21d REAL, log_spy_return_21d REAL)"
    )
    conn.execute("CREATE TABLE scanner_evaluations (ticker TEXT, eval_date TEXT, quant_filter_pass INTEGER)")
    conn.execute("CREATE TABLE team_candidates (ticker TEXT, eval_date TEXT, team_id TEXT, team_recommended INTEGER)")
    conn.execute("CREATE TABLE cio_evaluations (ticker TEXT, eval_date TEXT, cio_decision TEXT, final_score REAL, cio_conviction REAL, combined_score REAL, macro_shift REAL)")
    conn.execute("CREATE TABLE predictor_outcomes (symbol TEXT, prediction_date TEXT, "
                 "predicted_direction TEXT, prediction_confidence REAL)")
    dates = ["2026-05-01", "2026-05-08", "2026-05-15", "2026-05-22"]
    for di, d in enumerate(dates):
        for i in range(8):  # 8 names/date, combined_score ~monotone in realized alpha
            t = f"T{i:02d}"
            # Higher score → higher alpha every date; perturb the last date's top two
            # names so per-date ICs are NOT all identically 1.0 (a degenerate
            # zero-variance t-test) — strongly positive but with real dispersion.
            rank_i = (6 if i == 7 else 7 if i == 6 else i) if di == len(dates) - 1 else i
            log_ret = 0.01 * rank_i
            conn.execute(
                "INSERT INTO universe_returns VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (t, d, "Tech", 1.0, 0.5, i % 2, 2.0, 1.0, 1 if i >= 4 else 0, log_ret, 0.0),
            )
            conn.execute(
                "INSERT INTO cio_evaluations VALUES (?,?,?,?,?,?,?)",
                (t, d, "ADVANCE" if i >= 4 else "REJECT", float(50 + i), float(50 + i),
                 float(50 + i), float(i)),
            )
    conn.commit()
    conn.close()
    attr = compute_lift_metrics(str(db))["cio_lift"]["layer_attribution_21d"]
    assert attr["n_eval_dates"] == 4
    assert attr["combined_score_date_ic_n"] == 4
    # 3 dates IC=1.0 + 1 date slightly below → strongly positive mean, real variance.
    assert attr["combined_score_date_ic"] > 0.9
    assert attr["combined_score_date_ic_p"] is not None and attr["combined_score_date_ic_p"] < 0.05


def test_trailing_sector_neutral_leakfree_and_fallback():
    # L4564: leak-free trailing-sector z-score with pool-wide cold-start fallback.
    import math

    import pandas as pd

    from analysis.end_to_end import _trailing_sector_neutral

    # Sector A across 3 dates (chronological string order), sector B once.
    df = pd.DataFrame(
        [
            ("d1", "A", 10.0),
            ("d1", "A", 20.0),   # d1: no prior → both fall back to pool rank
            ("d2", "A", 30.0),
            ("d2", "A", 40.0),   # d2: prior A=[10,20] (n=2≥k_min) → z-scored
            ("d2", "B", 5.0),    # d2: no prior B → fall back
            ("d3", "A", 100.0),  # d3: prior A=[10,20,30,40] → z-scored, leak-free
        ],
        columns=["eval_date", "sector", "combined_score"],
    )
    q, frac = _trailing_sector_neutral(df, k_min=2)

    # d2 sector A is z-scored on the STRICTLY-PRIOR baseline mean=15, sd=√50.
    sd2 = math.sqrt(50.0)
    assert q.iloc[2] == pytest.approx((30.0 - 15.0) / sd2)  # not a 0–1 pool rank
    assert q.iloc[3] == pytest.approx((40.0 - 15.0) / sd2)
    # d3 sector A uses prior [10,20,30,40] (mean=25, sd ddof=1) — the current 100
    # does NOT enter its own baseline (leak-free).
    sd3 = pd.Series([10.0, 20.0, 30.0, 40.0]).std(ddof=1)
    assert q.iloc[5] == pytest.approx((100.0 - 25.0) / sd3)
    # Cold-start rows fell back to within-date pool-wide percentile rank ∈ (0, 1].
    assert 0.0 < q.iloc[0] <= 1.0 and 0.0 < q.iloc[1] <= 1.0
    assert 0.0 < q.iloc[4] <= 1.0
    # 3 of 6 valued rows used the true trailing transform.
    assert frac == pytest.approx(0.5)


def test_team_21d_block_present(tmp_path):
    out = compute_lift_metrics(_build_research_db(tmp_path))
    team = out["team_lift"][0]
    assert team["classification_21d"]["precision"] == 1.0
    assert "lift_21d_log" in team


def test_21d_absent_when_columns_missing(tmp_path):
    # Older DB without 21d columns → 21d blocks are None, 5d still computes.
    db = tmp_path / "research.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE universe_returns (ticker TEXT, eval_date TEXT, sector TEXT, "
        "return_5d REAL, spy_return_5d REAL, beat_spy_5d INTEGER)"
    )
    conn.execute("CREATE TABLE scanner_evaluations (ticker TEXT, eval_date TEXT, quant_filter_pass INTEGER)")
    conn.execute("CREATE TABLE team_candidates (ticker TEXT, eval_date TEXT, team_id TEXT, team_recommended INTEGER)")
    conn.execute("CREATE TABLE cio_evaluations (ticker TEXT, eval_date TEXT, cio_decision TEXT, final_score REAL, cio_conviction REAL, combined_score REAL, macro_shift REAL)")
    conn.execute("CREATE TABLE predictor_outcomes (symbol TEXT, prediction_date TEXT, "
                 "predicted_direction TEXT, prediction_confidence REAL)")
    for i in range(20):
        conn.execute("INSERT INTO universe_returns VALUES (?,?,?,?,?,?)",
                     (f"T{i:02d}", DATE, "Tech", 1.0, 0.5, i % 2))
        conn.execute("INSERT INTO scanner_evaluations VALUES (?,?,?)", (f"T{i:02d}", DATE, 1 if i < 10 else 0))
    conn.commit()
    conn.close()
    out = compute_lift_metrics(str(db))
    sl = out["scanner_lift"]
    assert sl["classification"] is not None
    assert sl["classification_21d"] is None
    assert sl["lift_21d_log"] is None
