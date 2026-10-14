"""tests/test_contribution_lift_research_composite.py — the research-tile T5 group.

Five substitution replays (config-I7478) over the shared contribution-lift
harness. Everything here is synthetic: no AWS, no ArcticDB, no vectorbt. The
ArcticDB exposure read is indirected through
``research_composite._load_loadings`` and substituted; the research SQLite is a
real temporary database built by the test, so the pre-scanner population query
is exercised for real rather than mocked.

Anchors:
  * contract: contribution_lift.json v1 (crucible-evaluator consumer)
  * spec: alpha-engine-docs/private/report-card-v3-objective-and-attribution-260816.md §1, §3
  * epic alpha-engine-config-I7473; harness I7475; this group I7478
"""

from __future__ import annotations

import dataclasses
import io
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.contribution_lift import harness, objective  # noqa: E402
from analysis.contribution_lift.groups import research_composite as rc  # noqa: E402
from analysis.contribution_lift.harness import (  # noqa: E402
    HORIZON_DAYS,
    ArmSet,
    NotAvailable,
    ReplayInputs,
)
from analysis.contribution_lift.registry import SPECS  # noqa: E402

H = HORIZON_DAYS

TICKERS = [f"T{i:02d}" for i in range(12)]


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _axis(n: int = 200) -> pd.DatetimeIndex:
    return pd.bdate_range("2026-01-01", periods=n)


def _signals(dates: list[str], enter: list[str]) -> dict:
    """ENTER on ``enter``, HOLD on everything else."""
    return {
        d: {
            "date": d,
            "signals": {
                t: {"ticker": t, "signal": "ENTER" if t in enter else "HOLD"}
                for t in TICKERS
            },
        }
        for d in dates
    }


def _profiles(dates: list[str], *, tickers: list[str] | None = None) -> dict:
    """``{date: {ticker: {pillar: score}}}`` with a deterministic ordering.

    Pillar values are a rank ramp rotated by the cycle index, so the composite
    has a real cross-sectional spread and its top-N is not the same set every
    cycle (which would make the rotation test vacuous).
    """
    names = tickers if tickers is not None else TICKERS
    pillars = (
        "quality_score", "value_score", "momentum_score",
        "growth_score", "stewardship_score", "low_vol_score",
    )
    out: dict = {}
    for i, d in enumerate(dates):
        rows = {}
        for j, t in enumerate(names):
            base = float((j + i) % len(names)) * 10.0
            rows[t] = {p: base + k for k, p in enumerate(pillars)}
        out[d] = rows
    return out


def _inputs(
    *,
    dates: list[str] | None = None,
    signals_by_date: dict | None = None,
    pillar_profiles_by_date: dict | None = None,
    research_db_path: str | None = None,
    tickers: list[str] | None = None,
) -> ReplayInputs:
    axis = _axis()
    cols = tickers if tickers is not None else TICKERS
    if dates is None:
        dates = [d.strftime("%Y-%m-%d") for d in axis[:60]]
    return ReplayInputs(
        run_date="2026-08-17",
        dates=dates,
        signals_by_date=signals_by_date or {},
        predictions_by_date={},
        pillar_profiles_by_date=pillar_profiles_by_date or {},
        price_matrix=pd.DataFrame(100.0, index=axis, columns=cols),
        spy_prices=pd.Series(100.0, index=axis),
        bucket="test-bucket",
        fees=0.001,
        slippage_bps=10.0,
        init_cash=1_000_000.0,
        research_db_path=research_db_path,
        source_paths=[],
    )


def _cycle_inputs(n_cycles: int = 40, **kwargs) -> ReplayInputs:
    axis = _axis()
    dates = [d.strftime("%Y-%m-%d") for d in axis[:n_cycles]]
    return _inputs(
        dates=dates,
        signals_by_date=_signals(dates, TICKERS[:4]),
        pillar_profiles_by_date=_profiles(dates),
        **kwargs,
    )


def _widths(arm) -> dict[str, int]:
    return harness.picks_per_cycle(arm)


def _matched(arms: ArmSet) -> bool:
    return objective.check_count_match(
        _widths(arms.baseline), _widths(arms.ablated)
    ).matched


@pytest.fixture(autouse=True)
def _clear_loadings_cache():
    rc._LOADINGS_CACHE.clear()
    yield
    rc._LOADINGS_CACHE.clear()


# --------------------------------------------------------------------------
# Selectors
# --------------------------------------------------------------------------


def test_top_n_breaks_ties_by_ticker():
    score = pd.Series({"BBB": 1.0, "AAA": 1.0, "CCC": 2.0})
    assert rc._top_n(score, 2) == ("AAA", "CCC")


def test_rotated_slice_walks_the_universe_and_wraps():
    universe = ["A", "B", "C", "D", "E"]
    assert rc._rotated_slice(universe, 2, 0) == ("A", "B")
    assert rc._rotated_slice(universe, 2, 1) == ("C", "D")
    # cycle 2 starts at index 4 and wraps back to the front
    assert rc._rotated_slice(universe, 2, 2) == ("A", "E")


def test_rotated_slice_is_deterministic_and_count_exact():
    universe = sorted(TICKERS)
    for index in range(20):
        first = rc._rotated_slice(universe, 5, index)
        assert first == rc._rotated_slice(universe, 5, index)
        assert len(first) == 5


def test_rotated_slice_never_exceeds_the_universe():
    assert rc._rotated_slice(["A", "B"], 5, 3) == ("A", "B")
    assert rc._rotated_slice([], 5, 0) == ()


def test_pillar_composite_orders_by_the_live_blend():
    dates = ["2026-01-05"]
    profiles = _profiles(dates)[dates[0]]
    composite = rc._pillar_composite(profiles, sorted(profiles))
    assert composite is not None
    # the ramp is monotone in the ticker index at cycle 0
    assert list(composite.sort_values(ascending=False).index[:2]) == ["T11", "T10"]


def test_pillar_composite_returns_none_on_a_degenerate_cross_section():
    profiles = {t: {"quality_score": 50.0} for t in TICKERS}
    composite = rc._pillar_composite(profiles, sorted(profiles))
    # every z collapses to 0.0 -> a constant, which carries no ordering
    assert composite is not None
    assert composite.nunique() == 1


# --------------------------------------------------------------------------
# research_composite_ic
# --------------------------------------------------------------------------


def test_research_composite_ic_baseline_is_top_n_by_the_composite():
    inputs = _cycle_inputs()
    arms = rc.build_research_composite_ic_arms(inputs)

    assert isinstance(arms, ArmSet)
    date, picks = arms.baseline.picks[0]
    profiles = inputs.pillar_profiles_by_date[date]
    composite = rc._pillar_composite(profiles, sorted(profiles))
    assert picks == rc._top_n(composite, 4)


def test_research_composite_ic_is_count_matched_to_the_live_enter_width():
    inputs = _cycle_inputs()
    arms = rc.build_research_composite_ic_arms(inputs)

    assert _matched(arms) is True
    assert set(_widths(arms.baseline).values()) == {4}


def test_research_composite_ic_ablated_carries_no_ranking_information():
    """The null arm must not reproduce the composite's own top-N."""
    inputs = _cycle_inputs()
    arms = rc.build_research_composite_ic_arms(inputs)

    baseline = dict(arms.baseline.picks)
    ablated = dict(arms.ablated.picks)
    assert set(baseline) == set(ablated)
    assert any(baseline[d] != ablated[d] for d in baseline)


def test_research_composite_ic_ablated_rotates_across_cycles():
    """Not one fixed alphabetical portfolio — the null walks the universe."""
    inputs = _cycle_inputs()
    arms = rc.build_research_composite_ic_arms(inputs)

    distinct = {picks for _d, picks in arms.ablated.picks}
    assert len(distinct) > 1


def test_research_composite_ic_is_deterministic():
    inputs = _cycle_inputs()
    first = rc.build_research_composite_ic_arms(inputs)
    second = rc.build_research_composite_ic_arms(inputs)
    assert first.baseline.picks == second.baseline.picks
    assert first.ablated.picks == second.ablated.picks


def test_research_composite_ic_na_without_pillar_profiles():
    dates = [d.strftime("%Y-%m-%d") for d in _axis()[:40]]
    inputs = _inputs(dates=dates, signals_by_date=_signals(dates, TICKERS[:4]))

    na = rc.build_research_composite_ic_arms(inputs)

    assert isinstance(na, NotAvailable)
    assert na.status == "N/A-MISSING-INPUT"
    assert "factors/profiles" in na.reason


def test_research_composite_ic_na_without_an_enter_signal():
    dates = [d.strftime("%Y-%m-%d") for d in _axis()[:40]]
    inputs = _inputs(
        dates=dates,
        signals_by_date=_signals(dates, []),
        pillar_profiles_by_date=_profiles(dates),
    )

    na = rc.build_research_composite_ic_arms(inputs)

    assert isinstance(na, NotAvailable)
    assert na.status == "N/A-MISSING-INPUT"
    assert "ENTER" in na.reason


def test_research_composite_ic_skips_a_cycle_narrower_than_the_live_width():
    """A universe smaller than the live ENTER count cannot be count-matched."""
    dates = [d.strftime("%Y-%m-%d") for d in _axis()[:40]]
    profiles = _profiles(dates)
    profiles[dates[0]] = {t: profiles[dates[0]][t] for t in TICKERS[:2]}
    inputs = _inputs(
        dates=dates,
        signals_by_date=_signals(dates, TICKERS[:4]),
        pillar_profiles_by_date=profiles,
    )

    arms = rc.build_research_composite_ic_arms(inputs)

    assert dates[0] not in dict(arms.baseline.picks)
    assert _matched(arms) is True


def test_research_composite_ic_only_selects_priceable_names():
    inputs = _cycle_inputs(tickers=TICKERS[:6])
    arms = rc.build_research_composite_ic_arms(inputs)

    priceable = set(map(str, inputs.price_matrix.columns))
    for _d, picks in arms.baseline.picks + arms.ablated.picks:
        assert set(picks) <= priceable


# --------------------------------------------------------------------------
# neutralization_live_efficacy
# --------------------------------------------------------------------------


def _loadings(dates: list[str], *, spread: bool = True) -> dict:
    """Exposures that make ``_xs_neutralize`` actually engage.

    It needs >= 20 names carrying every factor, so the fixture widens the
    universe on the exposure side; names absent from the price matrix simply
    never appear in a cycle's cross-section.
    """
    factors = ("momentum_20d", "return_60d", "beta_60d", "size_log")
    out: dict = {}
    for i, d in enumerate(dates):
        for j, t in enumerate(TICKERS):
            value = float((j * 7 + i) % 11) if spread else 1.0
            out[(d, t)] = {f: value + k for k, f in enumerate(factors)}
    return out


def test_neutralization_na_without_factor_loadings(monkeypatch):
    inputs = _cycle_inputs()
    monkeypatch.setattr(rc, "_load_loadings", lambda _inputs: {})

    na = rc.build_neutralization_live_efficacy_arms(inputs)

    assert isinstance(na, NotAvailable)
    assert na.status == "N/A-MISSING-INPUT"
    assert "load_historical_factor_loadings" in na.reason


def test_neutralization_na_when_the_residualizer_never_engages(monkeypatch):
    """12 names is below _xs_neutralize's 20-name floor -> identity passthrough."""
    inputs = _cycle_inputs()
    monkeypatch.setattr(
        rc, "_load_loadings", lambda _i: _loadings(inputs.dates)
    )

    na = rc.build_neutralization_live_efficacy_arms(inputs)

    assert isinstance(na, NotAvailable)
    assert na.status == "N/A-MISSING-INPUT"
    assert "identity" in na.reason


def _wide_inputs(n_cycles: int = 40) -> tuple[ReplayInputs, list[str]]:
    """A 30-name universe — wide enough for the residualizer to engage."""
    wide = [f"W{i:02d}" for i in range(30)]
    axis = _axis()
    dates = [d.strftime("%Y-%m-%d") for d in axis[:n_cycles]]
    signals = {
        d: {
            "date": d,
            "signals": {
                t: {"ticker": t, "signal": "ENTER" if t in wide[:4] else "HOLD"}
                for t in wide
            },
        }
        for d in dates
    }
    inputs = _inputs(
        dates=dates,
        signals_by_date=signals,
        pillar_profiles_by_date=_profiles(dates, tickers=wide),
        tickers=wide,
    )
    return inputs, wide


def _wide_loadings(dates: list[str], tickers: list[str]) -> dict:
    factors = ("momentum_20d", "return_60d", "beta_60d", "size_log")
    return {
        (d, t): {f: float((j * 5 + i + k) % 13) for k, f in enumerate(factors)}
        for i, d in enumerate(dates)
        for j, t in enumerate(tickers)
    }


def test_neutralization_arms_differ_and_are_count_matched(monkeypatch):
    inputs, wide = _wide_inputs()
    monkeypatch.setattr(
        rc, "_load_loadings", lambda _i: _wide_loadings(inputs.dates, wide)
    )

    arms = rc.build_neutralization_live_efficacy_arms(inputs)

    assert isinstance(arms, ArmSet)
    assert _matched(arms) is True
    baseline = dict(arms.baseline.picks)
    ablated = dict(arms.ablated.picks)
    assert set(baseline) == set(ablated)
    assert any(baseline[d] != ablated[d] for d in baseline), (
        "the neutralized arm reproduced the raw arm on every cycle — the "
        "residualizer would be measuring nothing"
    )


def test_neutralization_ablated_arm_is_the_raw_composite(monkeypatch):
    inputs, wide = _wide_inputs()
    monkeypatch.setattr(
        rc, "_load_loadings", lambda _i: _wide_loadings(inputs.dates, wide)
    )

    arms = rc.build_neutralization_live_efficacy_arms(inputs)

    date, picks = arms.ablated.picks[0]
    profiles = inputs.pillar_profiles_by_date[date]
    composite = rc._pillar_composite(profiles, sorted(profiles))
    assert picks == rc._top_n(composite, 4)


def test_neutralization_baseline_label_names_the_factors(monkeypatch):
    inputs, wide = _wide_inputs()
    monkeypatch.setattr(
        rc, "_load_loadings", lambda _i: _wide_loadings(inputs.dates, wide)
    )

    arms = rc.build_neutralization_live_efficacy_arms(inputs)

    assert "beta_60d" in arms.baseline.label
    assert "bypassed" in arms.ablated.label


def test_neutralization_is_deterministic(monkeypatch):
    inputs, wide = _wide_inputs()
    monkeypatch.setattr(
        rc, "_load_loadings", lambda _i: _wide_loadings(inputs.dates, wide)
    )

    first = rc.build_neutralization_live_efficacy_arms(inputs)
    second = rc.build_neutralization_live_efficacy_arms(inputs)

    assert first.baseline.picks == second.baseline.picks
    assert first.ablated.picks == second.ablated.picks


def test_loadings_are_loaded_once_per_window(monkeypatch):
    inputs = _cycle_inputs()
    calls: list[int] = []

    def _fake(bucket, dates, factors):
        calls.append(1)
        return {("x", "y"): {}}

    monkeypatch.setattr(
        "analysis.end_to_end.load_historical_factor_loadings", _fake
    )
    rc._load_loadings(inputs)
    rc._load_loadings(inputs)

    assert len(calls) == 1


# --------------------------------------------------------------------------
# scanner_feed_counterfactual
# --------------------------------------------------------------------------


def _research_db(tmp_path: Path, rows: list[tuple[str, str]]) -> str:
    path = tmp_path / "research.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE scanner_evaluations ("
        "id INTEGER PRIMARY KEY, ticker TEXT, eval_date TEXT, "
        "quant_filter_pass INTEGER NOT NULL DEFAULT 0)"
    )
    conn.executemany(
        "INSERT INTO scanner_evaluations (ticker, eval_date, quant_filter_pass) "
        "VALUES (?, ?, 0)",
        rows,
    )
    conn.commit()
    conn.close()
    return str(path)


def test_scanner_na_without_a_research_db():
    inputs = _cycle_inputs()

    na = rc.build_scanner_feed_counterfactual_arms(inputs)

    assert isinstance(na, NotAvailable)
    assert na.status == "N/A-MISSING-INPUT"
    assert "scanner_evaluations" in na.reason


def test_scanner_na_when_the_population_table_is_empty(tmp_path):
    db = _research_db(tmp_path, [])
    inputs = _cycle_inputs(research_db_path=db)

    na = rc.build_scanner_feed_counterfactual_arms(inputs)

    assert isinstance(na, NotAvailable)
    assert na.status == "N/A-MISSING-INPUT"


def test_scanner_baseline_is_the_live_selection(tmp_path):
    axis = _axis()
    dates = [d.strftime("%Y-%m-%d") for d in axis[:40]]
    # alpha-engine-config-I8757: the population must reach the END of the
    # replayed window. A single cycle at dates[0] leaves every later cycle
    # drawing its unranked arm from a population ~2 months old, which the
    # component now refuses — that is the production defect, not a fixture
    # detail. Seeding both ends keeps this test about the BASELINE.
    db = _research_db(
        tmp_path,
        [(t, dates[0]) for t in TICKERS] + [(t, dates[-1]) for t in TICKERS],
    )
    inputs = _cycle_inputs(research_db_path=db)

    arms = rc.build_scanner_feed_counterfactual_arms(inputs)

    assert isinstance(arms, ArmSet)
    assert dict(arms.baseline.picks)[dates[0]] == tuple(sorted(TICKERS[:4]))


def test_scanner_ablated_draws_from_the_pre_scanner_population(tmp_path):
    axis = _axis()
    dates = [d.strftime("%Y-%m-%d") for d in axis[:40]]
    db = _research_db(  # both ends seeded — see I8757 note above
        tmp_path,
        [(t, dates[0]) for t in TICKERS] + [(t, dates[-1]) for t in TICKERS],
    )
    inputs = _cycle_inputs(research_db_path=db)

    arms = rc.build_scanner_feed_counterfactual_arms(inputs)

    population = set(TICKERS)
    for _d, picks in arms.ablated.picks:
        assert set(picks) <= population
    assert _matched(arms) is True
    distinct = {picks for _d, picks in arms.ablated.picks}
    assert len(distinct) > 1


def test_scanner_population_is_point_in_time_as_of(tmp_path):
    """A cycle uses the last research cycle at or BEFORE it, never a later one."""
    axis = _axis()
    dates = [d.strftime("%Y-%m-%d") for d in axis[:40]]
    early, late = dates[0], dates[10]
    rows = (
        [(t, early) for t in TICKERS[:8]]
        + [(t, late) for t in TICKERS]
        # I8757: a cycle at the end of the window, so the population is not
        # frozen relative to the replay. The as-of assertion below is on
        # dates[5], which still resolves to `early`.
        + [(t, dates[-1]) for t in TICKERS]
    )
    db = _research_db(tmp_path, rows)
    inputs = _cycle_inputs(research_db_path=db)

    arms = rc.build_scanner_feed_counterfactual_arms(inputs)

    ablated = dict(arms.ablated.picks)
    assert set(ablated[dates[5]]) <= set(TICKERS[:8])


def test_scanner_skips_cycles_before_the_first_research_cycle(tmp_path):
    axis = _axis()
    dates = [d.strftime("%Y-%m-%d") for d in axis[:40]]
    db = _research_db(  # both ends seeded — see I8757 note above
        tmp_path,
        [(t, dates[5]) for t in TICKERS] + [(t, dates[-1]) for t in TICKERS],
    )
    inputs = _cycle_inputs(research_db_path=db)

    arms = rc.build_scanner_feed_counterfactual_arms(inputs)

    assert dates[0] not in dict(arms.baseline.picks)
    assert dates[5] in dict(arms.baseline.picks)


def test_scanner_population_query_is_read_only(tmp_path):
    db = _research_db(tmp_path, [("AAA", "2026-01-01")])
    population = rc._pre_scanner_population(db)
    assert population == {"2026-01-01": ("AAA",)}


def test_scanner_population_missing_table_is_empty_not_an_exception(tmp_path):
    path = tmp_path / "bare.db"
    sqlite3.connect(path).close()
    assert rc._pre_scanner_population(str(path)) == {}


# alpha-engine-config-I12231: the live Scanner publishes the whole pre-scanner
# population as candidates/{run_date}/candidates.json::scanner_eval_log. The
# component used to read only the retired scanner_evaluations table, so it was
# N/A-MISSING-INPUT on every card while the producer emitted its input.


class _CandidatesS3:
    """A read-only S3 stub holding ``candidates/{date}/candidates.json``."""

    def __init__(self, cohorts: dict[str, list[str]], *, passing: int = 4):
        self._objects = {
            f"candidates/{d}/candidates.json": json.dumps({
                "run_date": d,
                "scanner_eval_log": [
                    {"ticker": t, "quant_filter_pass": 1 if i < passing else 0}
                    for i, t in enumerate(names)
                ],
            }).encode()
            for d, names in cohorts.items()
        }

    def list_objects_v2(self, **kwargs):
        assert kwargs["Prefix"] == "candidates/"
        prefixes = sorted({k.rsplit("/", 1)[0] + "/" for k in self._objects})
        return {"CommonPrefixes": [{"Prefix": p} for p in prefixes]}

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self._objects[Key])}


def _weekly(dates: list[str]) -> list[str]:
    """One date per ISO week (the live reader keeps one cohort per week)."""
    seen: dict[tuple, str] = {}
    for d in dates:
        iso = pd.Timestamp(d).isocalendar()
        seen.setdefault((iso[0], iso[1]), d)
    return sorted(seen.values())


def _with_s3(inputs: ReplayInputs, s3) -> ReplayInputs:
    return dataclasses.replace(inputs, s3_client=s3)


def test_scanner_reads_the_live_population_when_the_table_is_frozen(tmp_path):
    """The production state on 2026-10-09: scanner_evaluations stops early in
    the window, the live record covers the rest. Before I12231 this returned
    N/A-MISSING-INPUT ("population is frozen")."""
    axis = _axis()
    dates = [d.strftime("%Y-%m-%d") for d in axis[:40]]
    db = _research_db(tmp_path, [(t, dates[0]) for t in TICKERS])
    live_dates = _weekly(dates[10:])
    s3 = _CandidatesS3({d: list(TICKERS) for d in live_dates})
    inputs = _with_s3(_cycle_inputs(research_db_path=db), s3)

    arms = rc.build_scanner_feed_counterfactual_arms(inputs)

    assert isinstance(arms, ArmSet), getattr(arms, "reason", arms)
    assert _matched(arms) is True
    ablated = dict(arms.ablated.picks)
    assert dates[-1] in ablated
    # Spring cycles still draw from the retired table, the rest from live.
    assert dates[5] in ablated
    assert f"{len(dates) - 10}/{len(dates)} cycles from the live" in arms.ablated.label


def test_scanner_reads_the_live_population_without_a_research_db():
    axis = _axis()
    dates = [d.strftime("%Y-%m-%d") for d in axis[:40]]
    s3 = _CandidatesS3({d: list(TICKERS) for d in _weekly(dates)})
    inputs = _with_s3(_cycle_inputs(), s3)

    arms = rc.build_scanner_feed_counterfactual_arms(inputs)

    assert isinstance(arms, ArmSet), getattr(arms, "reason", arms)
    assert {d for d, _ in arms.baseline.picks} == set(dates)


def test_scanner_live_population_keeps_names_the_gate_rejected():
    """Rejected names are the population the ranking was applied to."""
    axis = _axis()
    dates = [d.strftime("%Y-%m-%d") for d in axis[:40]]
    s3 = _CandidatesS3({d: list(TICKERS) for d in _weekly(dates)}, passing=1)
    population = rc._live_scanner_population(_with_s3(_cycle_inputs(), s3))

    assert population and all(set(v) == set(TICKERS) for v in population.values())


def test_scanner_live_record_wins_from_its_first_cohort_on(tmp_path):
    """A table row dated after the first live cohort is never used."""
    axis = _axis()
    dates = [d.strftime("%Y-%m-%d") for d in axis[:40]]
    live_dates = _weekly(dates[10:])
    db = _research_db(
        tmp_path,
        [(t, dates[0]) for t in TICKERS]
        + [(t, d) for d in dates[10:] for t in TICKERS[:6]],
    )
    s3 = _CandidatesS3({d: list(TICKERS[6:]) for d in live_dates})
    inputs = _with_s3(_cycle_inputs(research_db_path=db), s3)

    arms = rc.build_scanner_feed_counterfactual_arms(inputs)

    ablated = dict(arms.ablated.picks)
    for d in dates[10:]:
        assert set(ablated[d]) <= set(TICKERS[6:])
    assert dates[5] in ablated


def test_scanner_still_refuses_when_the_live_record_goes_stale():
    """The I8757 guard survives: a Scanner that stops publishing is N/A."""
    axis = _axis()
    dates = [d.strftime("%Y-%m-%d") for d in axis[:40]]
    s3 = _CandidatesS3({dates[0]: list(TICKERS)})
    inputs = _with_s3(_cycle_inputs(), s3)

    na = rc.build_scanner_feed_counterfactual_arms(inputs)

    assert isinstance(na, NotAvailable)
    assert na.status == "N/A-MISSING-INPUT"
    assert "frozen" in na.reason
    assert "candidates.json::scanner_eval_log" in na.reason


def test_scanner_merge_keeps_the_table_only_before_the_live_record():
    merged = rc._merged_population(
        {"2026-07-10": ("B",), "2026-07-17": ("C",)},
        {"2026-07-03": ("A",), "2026-07-10": ("X",), "2026-07-17": ("Y",)},
    )
    assert merged == {
        "2026-07-03": ("A",), "2026-07-10": ("B",), "2026-07-17": ("C",)
    }


# --------------------------------------------------------------------------
# Retired components
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec_name", ["sector_teams_avg", "cio_selection_skill"]
)
def test_retired_components_emit_na_retired_with_the_retirement_date(spec_name):
    from analysis.end_to_end import RESEARCH_GRAPH_RETIRED_DATE

    spec = next(s for s in SPECS if s.name == spec_name)
    na = spec.build_arms(_cycle_inputs())

    assert isinstance(na, NotAvailable)
    assert na.status == "N/A-RETIRED"
    assert RESEARCH_GRAPH_RETIRED_DATE in na.reason
    assert rc.ISSUE in na.reason


def test_retirement_date_is_the_documented_one():
    from analysis.end_to_end import RESEARCH_GRAPH_RETIRED_DATE

    assert RESEARCH_GRAPH_RETIRED_DATE == "2026-07-12"


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


GROUP_COMPONENTS = {
    "research_composite_ic": "critical",
    "sector_teams_avg": "critical",
    "cio_selection_skill": "critical",
    "neutralization_live_efficacy": "critical",
    "scanner_feed_counterfactual": "supporting",
}


def test_all_five_components_are_registered_once():
    names = [s.name for s in SPECS]
    for name in GROUP_COMPONENTS:
        assert names.count(name) == 1, f"{name} registered {names.count(name)}x"


def test_registered_specs_carry_the_tile_criticality_and_pattern():
    by_name = {s.name: s for s in SPECS}
    for name, criticality in GROUP_COMPONENTS.items():
        spec = by_name[name]
        assert spec.module == "research"
        assert spec.criticality == criticality
        assert spec.pattern == "substitution"
        assert spec.issue == rc.ISSUE


# --------------------------------------------------------------------------
# End to end through run_spec (fake simulator)
# --------------------------------------------------------------------------


class _LabelledSim:
    """Maps an arm label to a canned daily log-return series."""

    def __init__(self, by_label: dict[str, pd.Series]):
        self.by_label = by_label

    def __call__(self, arm, arm_set, inputs):
        log_returns = self.by_label[arm.label]
        return {
            "total_return": float(np.expm1(log_returns.sum())),
            "total_alpha": 0.0,
            "sortino_ratio": 1.0,
            "sharpe_ratio": 0.9,
            "max_drawdown": -0.05,
            "psr": 0.6,
            "daily_returns": np.expm1(log_returns),
            "daily_log_returns": log_returns,
            "n_orders": len(harness.arm_orders(arm, arm_set, inputs)),
        }


def test_research_composite_ic_runs_through_the_harness(monkeypatch):
    inputs = _cycle_inputs(n_cycles=40)
    arms = rc.build_research_composite_ic_arms(inputs)
    axis = inputs.price_matrix.index
    monkeypatch.setattr(
        harness,
        "simulate_arm",
        _LabelledSim({
            arms.baseline.label: pd.Series(0.001, index=axis),
            arms.ablated.label: pd.Series(0.0, index=axis),
        }),
    )

    component = harness.run_spec(
        rc.SPEC_RESEARCH_COMPOSITE_IC, inputs, n_trials=100, built=arms
    )

    assert component["name"] == "research_composite_ic"
    assert component["module"] == "research"
    assert component["pattern"] == "substitution"
    assert component["count_matched"] is True
    assert component["unit"] == harness.UNIT
    # baseline earns 0.001/day over the horizon, ablated earns nothing
    assert component["value"] == pytest.approx(0.001 * H, rel=1e-6)


def test_retired_component_flows_through_run_spec_as_na():
    inputs = _cycle_inputs()
    spec = next(s for s in SPECS if s.name == "cio_selection_skill")

    component = harness.run_spec(spec, inputs, n_trials=100)

    assert component["status"] == "N/A-RETIRED"
    assert component["value"] is None
    assert component["arms"] == {}
