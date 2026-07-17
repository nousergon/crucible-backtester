"""Unit tests for the config#816 pit_parity CSCV sweep plumbing in
``backtest.py`` — the opt-in (decision A) config sweep that feeds the full
M-block CSCV PBO engine on the PIT pass ONLY (decision B).

Covers the block-matrix builder + the per-block evaluation seam with a stubbed
``sim_fn`` (no live ArcticDB / executor pipeline needed). The heavy end-to-end
path (run_predictor_backtest → real sim_fn) requires live price history and is
validated by a scoped SF run, not here.
"""

from __future__ import annotations

import pandas as pd

import backtest


def _sweep_df():
    # Already Sortino-desc-sorted (as run_predictor_param_sweep emits). Two grid
    # params so the reconstruction of per-combo configs is exercised.
    return pd.DataFrame(
        {
            "atr_mult": [1.0, 1.5, 2.0, 2.5],
            "top_n": [10, 20, 30, 40],
            "sortino_ratio": [3.0, 2.0, 1.0, 0.5],
        }
    )


def test_eval_combo_on_block_threads_block_dates():
    seen = {}

    def sim_fn(cfg):
        seen.update(cfg)
        return {"sortino_ratio": 1.23}

    out = backtest._eval_combo_on_block(
        sim_fn, {"atr_mult": 2.0}, ["2026-01-01", "2026-01-02"]
    )
    assert out["sortino_ratio"] == 1.23
    # The block dates are threaded through so _run_simulation_loop restricts
    # iteration to the block (decision B — per-block CSCV re-evaluation).
    assert seen["_cscv_block_dates"] == ["2026-01-01", "2026-01-02"]
    assert seen["atr_mult"] == 2.0


def test_build_cscv_matrix_shape_and_metadata():
    dates = [f"2026-01-{d:02d}" for d in range(1, 41)]  # 40 dates → 8 blocks of 5

    calls: list[dict] = []

    def sim_fn(cfg):
        calls.append(cfg)
        # Deterministic per-(combo, block) Sortino so the matrix is checkable.
        return {"sortino_ratio": float(cfg["atr_mult"]) + len(cfg["_cscv_block_dates"])}

    config = {
        "param_sweep": {"atr_mult": [1.0, 1.5, 2.0, 2.5], "top_n": [10, 20, 30, 40]},
        "pit_parity_cscv_n_blocks": 8,
        "pit_parity_cscv_top_k": 4,
    }
    out = backtest._build_pit_parity_cscv_matrix(_sweep_df(), sim_fn, dates, config)

    mat = out["_cscv_block_matrix"]
    # 8 chronological blocks × top-4 combos.
    assert len(mat) == 8
    assert all(len(row) == 4 for row in mat)
    assert out["_cscv_spec_ids"] == [0, 1, 2, 3]
    assert out["_cscv_n_trials"] == 4  # len(sweep_df)
    # Every per-block sim got the block dates threaded + the combo's grid params.
    assert all("_cscv_block_dates" in c and "atr_mult" in c for c in calls)


def test_build_cscv_matrix_needs_two_combos():
    df = pd.DataFrame({"atr_mult": [1.0], "sortino_ratio": [3.0]})
    out = backtest._build_pit_parity_cscv_matrix(
        df, lambda c: {"sortino_ratio": 1.0},
        [f"2026-01-{d:02d}" for d in range(1, 41)],
        {"param_sweep": {"atr_mult": [1.0]}, "pit_parity_cscv_top_k": 4},
    )
    assert out == {}  # <2 combos → no matrix (honest-N/A upstream → pbo null)


def test_build_cscv_matrix_end_to_end_into_report():
    # The block matrix builder + analysis.pit_parity._cscv_pbo compose into a
    # real PBO distribution — the full gap-1+2 path minus the live pipeline.
    from analysis import pit_parity as pp

    dates = [f"2026-{m:02d}-{d:02d}" for m in range(1, 5) for d in range(1, 11)]

    def sim_fn(cfg):
        # Combo 0 overfits: strong early, weak late (drives PBO up).
        blk = cfg["_cscv_block_dates"]
        early = blk[0] < "2026-03-01"
        base = float(cfg["atr_mult"])
        if abs(base - 1.0) < 1e-9:
            base += 5.0 if early else -5.0
        return {"sortino_ratio": base}

    config = {
        "param_sweep": {"atr_mult": [1.0, 1.5, 2.0, 2.5], "top_n": [10, 20, 30, 40]},
        "pit_parity_cscv_n_blocks": 8,
        "pit_parity_cscv_top_k": 4,
    }
    built = backtest._build_pit_parity_cscv_matrix(_sweep_df(), sim_fn, dates, config)
    r = pp._cscv_pbo(built["_cscv_block_matrix"], spec_ids=built["_cscv_spec_ids"])
    assert r["status"] == "ok"
    assert 0.0 <= r["pbo"] <= 1.0
