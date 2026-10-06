"""The alpha floor's unblock predicate — alpha-engine-config-I10064.

The weekly Director read ``executor_params blocked 15w [alpha_floor]`` as a
loop that had stopped working and asked for a root cause. The live record
(``config/apply_audit/2026-10-02.json``) showed a guard refusing: 60/60
MEASURED combos had ``total_alpha < 0.0``, best ``-0.0914``. These tests pin
the predicate that tells those two apart, and they prove that routing
``recommend`` through it changed no accept or refuse decision.

Controlled cases, each with a known answer:

* the live-shaped refusal: every combo measured and negative → ``blocked``;
* a single combo exactly AT the floor → ``unblocked`` (``>=``, not ``>``);
* all-null alpha → ``unmeasured``, never ``blocked`` (config-I7672);
* partial nulls → the measured denominator is reported;
* floor unset → ``inactive``; no alpha column → ``bypassed_no_alpha``;
* a randomized equivalence check: the predicate's verdict and the set
  ``recommend`` ranks are the same thing.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from optimizer import alpha_floor_predicate as afp
from optimizer.apply_audit import build_audit
from optimizer.executor_optimizer import init_config, recommend

_FLOOR = 0.0  # the configured value; these tests read it, they do not choose it
_CFG = {"executor_optimizer": {"alpha_floor": _FLOOR, "min_valid_combos": 3,
                               "min_trades_to_promote": 1}}


@pytest.fixture(autouse=True)
def _init():
    init_config(_CFG)
    yield
    init_config({})


def _sweep(total_alpha, *, with_alpha_col=True):
    n = len(total_alpha)
    df = pd.DataFrame({
        "min_score": [55 + (i % 4) * 5 for i in range(n)],
        "max_position_pct": [0.05 + (i % 3) * 0.025 for i in range(n)],
        "atr_multiplier": [2.0 + (i % 5) * 0.5 for i in range(n)],
        "total_return": [0.03 + i * 0.0005 for i in range(n)],
        "spy_return": [0.15] * n,
        "sharpe_ratio": [0.2 + i * 0.004 for i in range(n)],
        "sortino_ratio": [0.4 + i * 0.008 for i in range(n)],
        "alpha_vs_ew_high_vol": [-0.15] * n,
        "psr": [0.9] * n,
        "total_trades": [80] * n,
        "status": ["ok"] * n,
        "dates_simulated": [57] * n,
        "dates_expected": [57] * n,
        "coverage": [1.0] * n,
    })
    if with_alpha_col:
        df["total_alpha"] = total_alpha
    return df


def _live_shaped():
    """60 measured combos, all negative, best -0.0914 — the 2026-10-02 shape."""
    alphas = list(np.linspace(-0.2114, -0.0914, 60))
    return _sweep(alphas)


# ── The live refusal ────────────────────────────────────────────────────────


def test_live_shape_is_a_measured_safety_refusal():
    v = afp.evaluate_sweep(_live_shaped(), _FLOOR, min_valid_combos=3)
    assert v.state == afp.BLOCKED
    assert v.met is False and v.evaluated and v.refuses_promotion
    assert (v.n_valid_combos, v.n_measured, v.n_at_or_above_floor) == (60, 60, 0)
    assert v.best_alpha == pytest.approx(-0.0914, abs=1e-6)
    assert v.margin_to_floor == pytest.approx(-0.0914, abs=1e-6)
    assert v.window == {"dates_simulated_min": 57, "dates_simulated_max": 57,
                        "dates_expected_max": 57, "coverage_min": 1.0}


def test_recommend_refuses_live_shape_and_carries_the_same_verdict():
    res = recommend(_live_shaped(), {})
    assert res["status"] == "alpha_below_floor"
    rec = res["alpha_floor_predicate"]
    assert rec == afp.evaluate_sweep(_live_shaped(), _FLOOR, min_valid_combos=3).to_record()
    assert rec["state"] == "blocked" and rec["met"] is False
    # The pre-existing fields and note are unchanged for existing readers.
    assert res["n_measured"] == 60
    assert res["best_alpha_in_sweep"] == pytest.approx(-0.0914, abs=1e-4)
    assert "All 60 MEASURED combos (of 60 valid)" in res["note"]


def test_record_states_provenance_a_report_can_quote():
    rec = afp.evaluate_sweep(_live_shaped(), _FLOOR, min_valid_combos=3).to_record()
    assert rec["predicate_id"] == "executor_params.alpha_floor.unblock"
    assert rec["predicate_version"] == 1
    assert (rec["metric"], rec["comparator"], rec["unit"]) == ("total_alpha", ">=", "fraction")
    assert rec["alpha_floor"] == 0.0
    assert rec["floor_source"] == "backtester config.yaml executor_optimizer.alpha_floor"
    assert rec["sample"] == {"n_valid_combos": 60, "n_measured": 60,
                             "n_unmeasured": 0, "min_valid_combos": 3}
    assert rec["downstream_guards"] == list(afp.DOWNSTREAM_GUARDS)
    assert "NOT MET: 0/60 measured combos" in rec["summary"]
    assert ">= 0.0" in rec["unblock_condition"]
    # The best combo is named, so a challenger experiment can start from it.
    assert rec["best_combo"]["min_score"] in (55, 60, 65, 70)
    assert "total_alpha" not in rec["best_combo"]
    json.dumps(rec, allow_nan=False)


# ── Acceptance: the comparator and the boundary ─────────────────────────────


def test_one_combo_exactly_at_the_floor_unblocks():
    alphas = [-0.05] * 9 + [0.0]
    v = afp.evaluate_sweep(_sweep(alphas), _FLOOR, min_valid_combos=3)
    assert v.state == afp.UNBLOCKED and v.met is True and not v.refuses_promotion
    assert v.n_at_or_above_floor == 1 and v.margin_to_floor == 0.0
    res = recommend(_sweep(alphas), {})
    assert res["status"] not in ("alpha_below_floor", "alpha_unmeasured")
    assert res["alpha_floor_predicate"]["state"] == "unblocked"


def test_just_below_the_floor_still_refuses():
    alphas = [-0.05] * 9 + [-1e-9]
    assert afp.evaluate_sweep(_sweep(alphas), _FLOOR, min_valid_combos=3).state == afp.BLOCKED
    assert recommend(_sweep(alphas), {})["status"] == "alpha_below_floor"


def test_unblocked_is_not_promoted_downstream_guards_still_run():
    """Clearing the floor hands the sweep to the next guard; it never promotes
    on its own. Here only one combo passes and it has too few trades."""
    init_config({"executor_optimizer": {"alpha_floor": _FLOOR, "min_valid_combos": 3,
                                        "min_trades_to_promote": 500}})
    res = recommend(_sweep([-0.05] * 9 + [0.01]), {})
    assert res["status"] == "insufficient_trades"
    assert res["alpha_floor_predicate"]["met"] is True


# ── Not-evaluated states are never reported as a measured result ────────────


def test_all_null_alpha_is_unmeasured_not_blocked():
    v = afp.evaluate_sweep(_sweep([np.nan] * 10), _FLOOR, min_valid_combos=3)
    assert v.state == afp.UNMEASURED and not v.evaluated
    assert v.met is False and v.refuses_promotion  # refuses, but says why
    assert v.best_alpha is None and v.margin_to_floor is None
    assert "unmeasured, not alpha-negative" in v.summary()
    res = recommend(_sweep([np.nan] * 10), {})
    assert res["status"] == "alpha_unmeasured"
    assert res["alpha_floor_predicate"]["state"] == "unmeasured"


def test_partial_nulls_report_the_measured_denominator():
    alphas = [np.nan] * 7 + [-0.03, -0.02, 0.01]
    v = afp.evaluate_sweep(_sweep(alphas), _FLOOR, min_valid_combos=3)
    assert v.state == afp.UNBLOCKED
    assert (v.n_valid_combos, v.n_measured, v.n_unmeasured) == (10, 3, 7)
    assert "1/3 measured combos (of 10 valid)" in v.summary()


def test_floor_unset_is_inactive_and_recommend_unchanged():
    init_config({"executor_optimizer": {"min_valid_combos": 3, "min_trades_to_promote": 1}})
    v = afp.evaluate_sweep(_live_shaped(), None, min_valid_combos=3)
    assert v.state == afp.INACTIVE and v.met is None and not v.refuses_promotion
    res = recommend(_live_shaped(), {})
    assert res["status"] != "alpha_below_floor"
    assert res["alpha_floor_predicate"]["state"] == "inactive"


def test_no_alpha_column_is_reported_as_bypassed():
    """Shipped behaviour, made visible: with no total_alpha column the gate does
    not run. The predicate says so; it does not refuse in recommend's place."""
    df = _sweep([0.0] * 10, with_alpha_col=False)
    v = afp.evaluate_sweep(df, _FLOOR, min_valid_combos=3)
    assert v.state == afp.BYPASSED_NO_ALPHA and v.met is None
    res = recommend(df, {})
    assert res["status"] not in ("alpha_below_floor", "alpha_unmeasured")
    assert res["alpha_floor_predicate"]["state"] == "bypassed_no_alpha"


@pytest.mark.parametrize("df", [
    None,
    pd.DataFrame(),
    pd.DataFrame({"total_alpha": [0.1, 0.2, 0.3]}),  # no sharpe_ratio
])
def test_raw_sweep_starved_upstream_is_insufficient_sample(df):
    v = afp.evaluate_sweep(df, _FLOOR, min_valid_combos=3)
    assert v.state == afp.INSUFFICIENT_SAMPLE and v.met is None


def test_low_completion_sweep_is_insufficient_sample():
    df = _live_shaped()
    df.attrs["sweep_low_completion"] = True
    assert afp.evaluate_sweep(df, _FLOOR, min_valid_combos=3).state == afp.INSUFFICIENT_SAMPLE


def test_too_few_valid_combos_is_insufficient_sample():
    df = _sweep([0.1, 0.2])
    assert afp.evaluate_sweep(df, _FLOOR, min_valid_combos=3).state == afp.INSUFFICIENT_SAMPLE


# ── The predicate and the gate cannot drift ─────────────────────────────────


@pytest.mark.parametrize("seed", range(25))
def test_predicate_verdict_equals_recommend_decision(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 30))
    alphas = rng.normal(-0.05, 0.05, n)
    alphas[rng.random(n) < 0.3] = np.nan
    df = _sweep(list(alphas))
    v = afp.evaluate_sweep(df, _FLOOR, min_valid_combos=3)
    res = recommend(df, {})
    refused = res["status"] in ("alpha_below_floor", "alpha_unmeasured")
    assert refused == v.refuses_promotion
    assert res["alpha_floor_predicate"] == v.to_record()
    if v.state == afp.UNBLOCKED and "n_combos_swept" in res:
        assert res["n_combos_swept"] == v.n_at_or_above_floor


# ── The audit record carries it ─────────────────────────────────────────────


def test_apply_audit_carries_the_predicate_on_executor_params_only():
    rec = recommend(_live_shaped(), {})
    audit = build_audit("2026-10-02", {"executor_rec": rec})
    ep = audit["loops"]["executor_params"]
    assert ep["outcome"] == "blocked" and ep["blocked_by"] == ["alpha_floor"]
    assert ep["unblock_predicate"]["state"] == "blocked"
    assert ep["unblock_predicate"]["met"] is False
    for loop in ("scoring_weights", "predictor_params", "research_params"):
        assert "unblock_predicate" not in audit["loops"][loop]
    json.dumps(audit, allow_nan=False)


def test_apply_audit_with_predicate_conforms_to_frozen_v1_schema():
    jsonschema = pytest.importorskip("jsonschema")
    from nousergon_lib import contracts

    audit = build_audit("2026-10-02", {"executor_rec": recommend(_live_shaped(), {})})
    jsonschema.validate(instance=audit, schema=contracts.load_schema("apply_audit"))


def test_apply_audit_without_predicate_is_unchanged():
    audit = build_audit("2026-10-02", {"executor_rec": {"status": "insufficient_data"}})
    assert "unblock_predicate" not in audit["loops"]["executor_params"]


# ── CLI over a saved param_sweep.csv ────────────────────────────────────────


def test_cli_exit_code_states_the_verdict(tmp_path, capsys):
    p = tmp_path / "param_sweep.csv"
    _live_shaped().to_csv(p, index=False)
    assert afp.main(["--sweep-csv", str(p), "--alpha-floor", "0.0",
                     "--min-valid-combos", "3"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["state"] == "blocked"
    _sweep([-0.05] * 9 + [0.02]).to_csv(p, index=False)
    assert afp.main(["--sweep-csv", str(p), "--alpha-floor", "0.0",
                     "--min-valid-combos", "3"]) == 0
