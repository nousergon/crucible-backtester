"""alpha_floor_predicate.py — the explicit unblock predicate for the
``executor_params`` alpha floor (alpha-engine-config-I10064).

**The question this answers:** "is the alpha-floor block on the executor
optimizer cleared, and if not, why not?" Before this module the answer was
spread across one ``if`` in ``executor_optimizer.recommend``, a config key in
the private backtester ``config.yaml`` and a carry-forward counter in
``apply_audit``. The weekly Director read the counter (15 consecutive
``blocked``) and asked for a root cause of a malfunction. The live record
(``config/apply_audit/2026-10-02.json``) says something else: 60 of 60 MEASURED
combos had ``total_alpha < 0.0``, best ``-0.0914``. That is the guard refusing,
as designed, not the loop failing.

**What this module does:** it states the predicate once, as data, and
``recommend`` makes its floor decision through it. Every result past the floor
stage carries the record under ``alpha_floor_predicate``, and ``apply_audit``
copies it onto the ``executor_params`` loop record as ``unblock_predicate``.
So the report and the gate read the same fact. Nobody re-derives it from a
note string.

**The predicate (version 1).** The floor is MET when at least one combo of the
weekly executor param sweep has a non-null ``total_alpha`` and that value is
``>= alpha_floor``. That holds only after the sweep cleared the upstream sample
gates in ``recommend``:

* the sweep is non-empty and not flagged ``sweep_low_completion`` (< 50 %);
* at least ``min_valid_combos`` combos have a non-null ``sharpe_ratio``.

* **metric**: ``total_alpha``, written per combo by
  ``vectorbt_bridge.portfolio_stats``.
* **unit**: a decimal fraction. It is the portfolio's simple total return
  minus SPY's simple return over the same window. ``-0.0914`` is -9.14
  percentage points.
* **window**: each combo's own ACTIVE trading window inside the sweep's
  simulated dates, from its first to its last traded date. ``dates_simulated``
  is reported, so a reader can see the window the verdict covers.
* **comparator**: ``>=``. A combo exactly at the floor passes. With the
  configured floor of ``0.0``, an SPY-neutral combo is allowed.

**States.** Exactly one applies:

* ``inactive`` (met None): the floor is not configured, so the gate does not
  run.
* ``insufficient_sample`` (met None): the upstream sample gate was not
  cleared, so the floor was not evaluated.
* ``bypassed_no_alpha`` (met None): a floor is configured, but the sweep has
  no ``total_alpha`` column, so the gate does NOT run. This is reported, not
  refused, because it is how the gate has behaved since it was built.
* ``unmeasured`` (met False): ``total_alpha`` is null on every combo. This is
  a refusal (config-I7672), and it is NOT a strategy result.
* ``blocked`` (met False): alpha was measured and nothing reaches the floor.
  This is a safety refusal.
* ``unblocked`` (met True): at least one measured combo reaches the floor.

``unblocked`` does not mean "promoted". It means only that this guard no
longer refuses. Every guard after it still runs, in order:
``min_trades_to_promote``, ``negative_rank_metric``,
``baseline_magnitude_floor``, ``min_improvement``, ``min_psr``. Walk-forward or
holdout validation and the assembler cutover follow those. Economic recovery,
meaning the live portfolio's alpha, is a separate measure. Clearing this
predicate is not evidence of it.

**This module changes no threshold.** The floor value comes from config, and
Brian owns it. ``recommend``'s accept/refuse decisions are unchanged, and
``tests/test_alpha_floor_predicate.py`` holds them equal.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

PREDICATE_ID = "executor_params.alpha_floor.unblock"
PREDICATE_VERSION = 1

METRIC = "total_alpha"
COMPARATOR = ">="
UNIT = "fraction"
UNIT_DEFINITION = (
    "portfolio simple total return minus SPY simple return over the same "
    "window, as a decimal fraction (-0.0914 = -9.14 percentage points)"
)
WINDOW_DEFINITION = (
    "per-combo active trading window (first to last traded date) inside the "
    "weekly executor param sweep's simulated dates"
)
FLOOR_SOURCE = "backtester config.yaml executor_optimizer.alpha_floor"
CODE_ANCHOR = "optimizer/executor_optimizer.py::recommend"

#: Guards ``recommend`` applies AFTER this one, in order, as ``apply_audit``
#: slugs. Clearing the floor hands the sweep to these. It does not promote.
DOWNSTREAM_GUARDS: tuple[str, ...] = (
    "min_trades_to_promote",
    "negative_rank_metric",
    "baseline_magnitude_floor",
    "min_improvement",
    "min_psr",
)

INACTIVE = "inactive"
INSUFFICIENT_SAMPLE = "insufficient_sample"
BYPASSED_NO_ALPHA = "bypassed_no_alpha"
UNMEASURED = "unmeasured"
BLOCKED = "blocked"
UNBLOCKED = "unblocked"

STATES: tuple[str, ...] = (
    INACTIVE, INSUFFICIENT_SAMPLE, BYPASSED_NO_ALPHA, UNMEASURED, BLOCKED,
    UNBLOCKED,
)

#: States in which this guard itself refuses promotion.
REFUSING_STATES = frozenset({UNMEASURED, BLOCKED})

#: Columns that identify a combo rather than describe its outcome. The record
#: names the best combo by these, so a challenger experiment can start from it.
_NON_PARAM_COLUMNS = frozenset({
    "total_return", "total_alpha", "spy_return", "sharpe_ratio",
    "sortino_ratio", "max_drawdown", "calmar_ratio", "cvar_95", "psr",
    "total_trades", "win_rate", "error", "status", "dates_simulated",
    "dates_expected", "coverage", "total_orders", "note", "null_legs",
    "ew_high_vol_return", "alpha_vs_ew_high_vol", "ew_universe_return",
    "alpha_vs_ew_universe", "daily_returns", "daily_log_returns",
    "_combined_score",
})


def _round(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return round(f, 6)


def _native(v: Any) -> Any:
    if hasattr(v, "item"):
        try:
            return v.item()
        except (ValueError, AttributeError):
            return v
    return v


@dataclass(frozen=True)
class AlphaFloorVerdict:
    """The predicate's verdict on one sweep. ``to_record()`` is the JSON form
    that results and ``apply_audit`` carry."""

    state: str
    alpha_floor: float | None
    n_valid_combos: int
    n_measured: int
    n_at_or_above_floor: int
    best_alpha: float | None
    min_valid_combos: int
    window: dict = field(default_factory=dict)
    best_combo: dict | None = None

    @property
    def evaluated(self) -> bool:
        """Did the floor comparison actually run on at least one measurement?"""
        return self.state in (BLOCKED, UNBLOCKED)

    @property
    def met(self) -> bool | None:
        """True: the floor no longer refuses. False: it refuses. None: the
        floor was not applied, either inactive, upstream-starved or bypassed."""
        if self.state == UNBLOCKED:
            return True
        if self.state in REFUSING_STATES:
            return False
        return None

    @property
    def refuses_promotion(self) -> bool:
        return self.state in REFUSING_STATES

    @property
    def margin_to_floor(self) -> float | None:
        """``best_alpha - alpha_floor``. Negative means short of the floor."""
        if self.best_alpha is None or self.alpha_floor is None:
            return None
        return round(self.best_alpha - self.alpha_floor, 6)

    @property
    def n_unmeasured(self) -> int:
        return self.n_valid_combos - self.n_measured

    def unblock_condition(self) -> str:
        floor = "<unset>" if self.alpha_floor is None else f"{self.alpha_floor}"
        return (
            f"at least 1 combo with non-null {METRIC} {COMPARATOR} {floor} "
            f"in a weekly sweep with completion >= 50% and at least "
            f"{self.min_valid_combos} combos with non-null sharpe_ratio"
        )

    def summary(self) -> str:
        """One line a report can quote verbatim."""
        floor = self.alpha_floor
        if self.state == INACTIVE:
            return "alpha floor inactive (executor_optimizer.alpha_floor unset) — no floor to unblock."
        if self.state == INSUFFICIENT_SAMPLE:
            return (
                f"alpha floor NOT EVALUATED — sweep has {self.n_valid_combos} valid "
                f"combos, below the {self.min_valid_combos}-combo sample minimum."
            )
        if self.state == BYPASSED_NO_ALPHA:
            return (
                f"alpha floor {floor} NOT APPLIED — the sweep carries no {METRIC} "
                f"column, so the gate did not run on {self.n_valid_combos} combos."
            )
        if self.state == UNMEASURED:
            return (
                f"alpha floor {floor} NOT EVALUATED — {METRIC} null on all "
                f"{self.n_valid_combos} valid combos (unmeasured, not alpha-negative)."
            )
        verb = "MET" if self.state == UNBLOCKED else "NOT MET"
        return (
            f"alpha floor {verb}: {self.n_at_or_above_floor}/{self.n_measured} "
            f"measured combos (of {self.n_valid_combos} valid) have {METRIC} "
            f"{COMPARATOR} {floor}; best {self.best_alpha} "
            f"(margin {self.margin_to_floor})."
        )

    def to_record(self) -> dict:
        base = asdict(self)
        return {
            "predicate_id": PREDICATE_ID,
            "predicate_version": PREDICATE_VERSION,
            "state": self.state,
            "met": self.met,
            "evaluated": self.evaluated,
            "refuses_promotion": self.refuses_promotion,
            "metric": METRIC,
            "comparator": COMPARATOR,
            "unit": UNIT,
            "unit_definition": UNIT_DEFINITION,
            "window_definition": WINDOW_DEFINITION,
            "alpha_floor": base["alpha_floor"],
            "floor_source": FLOOR_SOURCE,
            "code_anchor": CODE_ANCHOR,
            "sample": {
                "n_valid_combos": self.n_valid_combos,
                "n_measured": self.n_measured,
                "n_unmeasured": self.n_unmeasured,
                "min_valid_combos": self.min_valid_combos,
            },
            "window": dict(self.window),
            "n_at_or_above_floor": self.n_at_or_above_floor,
            "best_alpha": self.best_alpha,
            "margin_to_floor": self.margin_to_floor,
            "best_combo": dict(self.best_combo) if self.best_combo else None,
            "unblock_condition": self.unblock_condition(),
            "downstream_guards": list(DOWNSTREAM_GUARDS),
            "summary": self.summary(),
        }


def passing_mask(valid: pd.DataFrame, alpha_floor: float) -> pd.Series:
    """The ONE comparison the floor makes. ``recommend`` filters with this mask
    and the predicate counts with it, so the two cannot drift. ``NaN >= x`` is
    False, so an unmeasured combo never passes."""
    return valid[METRIC] >= alpha_floor


def _window(valid: pd.DataFrame) -> dict:
    out: dict[str, Any] = {}
    for col, agg in (
        ("dates_simulated", "min"), ("dates_simulated", "max"),
        ("dates_expected", "max"), ("coverage", "min"),
    ):
        if col in valid.columns:
            s = pd.to_numeric(valid[col], errors="coerce").dropna()
            if not s.empty:
                v = getattr(s, agg)()
                out[f"{col}_{agg}"] = _native(int(v) if col != "coverage" else round(float(v), 6))
    return out


def _best_combo(valid: pd.DataFrame) -> dict | None:
    measured = valid[valid[METRIC].notna()]
    if measured.empty:
        return None
    row = measured.loc[measured[METRIC].idxmax()]
    combo = {
        c: _native(row[c]) for c in valid.columns
        if c not in _NON_PARAM_COLUMNS and not str(c).startswith("_") and pd.notna(row[c])
    }
    for k in ("total_return", "spy_return", "sharpe_ratio", "sortino_ratio", "total_trades"):
        if k in valid.columns and pd.notna(row.get(k)):
            combo[k] = _native(row[k]) if k == "total_trades" else _round(row[k])
    return combo


def evaluate_alpha_floor(
    valid: pd.DataFrame,
    alpha_floor: float | None,
    *,
    min_valid_combos: int,
) -> AlphaFloorVerdict:
    """Evaluate the predicate over the sweep's VALID combos.

    ``valid`` is what ``recommend`` holds at the floor stage: the sweep's rows
    with a non-null ``sharpe_ratio``. A caller holding a raw sweep (the CLI
    below, or a report reading ``param_sweep.csv``) should use
    :func:`evaluate_sweep`, which applies the same upstream filter first.
    """
    n_valid = int(len(valid))
    common = dict(alpha_floor=None if alpha_floor is None else float(alpha_floor),
                  n_valid_combos=n_valid, min_valid_combos=int(min_valid_combos),
                  window=_window(valid))
    if alpha_floor is None:
        return AlphaFloorVerdict(state=INACTIVE, n_measured=0, n_at_or_above_floor=0,
                                 best_alpha=None, **common)
    if n_valid < min_valid_combos:
        return AlphaFloorVerdict(state=INSUFFICIENT_SAMPLE, n_measured=0,
                                 n_at_or_above_floor=0, best_alpha=None, **common)
    if METRIC not in valid.columns:
        return AlphaFloorVerdict(state=BYPASSED_NO_ALPHA, n_measured=0,
                                 n_at_or_above_floor=0, best_alpha=None, **common)
    n_measured = int(valid[METRIC].notna().sum())
    if n_measured == 0:
        return AlphaFloorVerdict(state=UNMEASURED, n_measured=0, n_at_or_above_floor=0,
                                 best_alpha=None, **common)
    n_pass = int(passing_mask(valid, float(alpha_floor)).sum())
    return AlphaFloorVerdict(
        state=UNBLOCKED if n_pass > 0 else BLOCKED,
        n_measured=n_measured,
        n_at_or_above_floor=n_pass,
        best_alpha=_round(valid[METRIC].max()),
        best_combo=_best_combo(valid),
        **common,
    )


def evaluate_sweep(
    sweep_df: pd.DataFrame | None,
    alpha_floor: float | None,
    *,
    min_valid_combos: int,
) -> AlphaFloorVerdict:
    """The predicate over a RAW sweep, with ``recommend``'s upstream sample
    gates applied first: empty, low completion and no ``sharpe_ratio`` each
    yield ``insufficient_sample`` when a floor is configured."""
    empty = pd.DataFrame()
    starved = (
        sweep_df is None or sweep_df.empty
        or bool(getattr(sweep_df, "attrs", {}).get("sweep_low_completion"))
        or "sharpe_ratio" not in sweep_df.columns
    )
    if starved:
        if alpha_floor is None:
            return evaluate_alpha_floor(empty, None, min_valid_combos=min_valid_combos)
        return AlphaFloorVerdict(
            state=INSUFFICIENT_SAMPLE, alpha_floor=float(alpha_floor),
            n_valid_combos=0, n_measured=0, n_at_or_above_floor=0,
            best_alpha=None, min_valid_combos=int(min_valid_combos),
        )
    valid = sweep_df[sweep_df["sharpe_ratio"].notna()]
    return evaluate_alpha_floor(valid, alpha_floor, min_valid_combos=min_valid_combos)


def main(argv: list[str] | None = None) -> int:
    """Evaluate a saved ``param_sweep.csv`` (for example, one copied from
    ``s3://<bucket>/backtest/<date>/param_sweep.csv``). The CLI only reads the
    file. The exit code says whether the floor is met: 0 met, 1 refusing,
    2 not applied."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sweep-csv", required=True)
    p.add_argument("--alpha-floor", type=float, required=True,
                   help="the configured executor_optimizer.alpha_floor (read it; do not choose it)")
    p.add_argument("--min-valid-combos", type=int, required=True,
                   help="the configured executor_optimizer.min_valid_combos")
    args = p.parse_args(argv)
    verdict = evaluate_sweep(pd.read_csv(args.sweep_csv), args.alpha_floor,
                             min_valid_combos=args.min_valid_combos)
    json.dump(verdict.to_record(), sys.stdout, indent=2)
    sys.stdout.write("\n")
    if verdict.met is True:
        return 0
    if verdict.met is False:
        return 1
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
