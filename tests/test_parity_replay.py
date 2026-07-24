"""
tests/test_parity_replay.py — replay parity test (Phase 1.1).

Diffs backtester output against the live trades.db over a historical window.
See docs/trade_mapping.md for the field mapping + tolerance contract.

Opt-in via the `parity` pytest marker so CI on feature branches doesn't
try to reach S3. Spot instance runs it explicitly via spot_backtest.sh
after the weekly backtest completes.

PARITY IS OBSERVABILITY, NOT A GATE
-----------------------------------
The integration test does NOT pass/fail on divergence. Its job is to
GENERATE two artifacts every Saturday SF run:

* ``parity_report.json`` — per-run drill-down (per-date count + ticker-set
  + field divergence breakdowns). Operator-readable; surfaced via the
  spot_backtest.sh upload to ``s3://{bucket}/backtest/{date}/``.
* ``parity_metrics.csv`` — append one row per run to
  ``s3://{bucket}/backtest/parity_metrics.csv`` with five trend-friendly
  numbers. Time series: anomaly is visible as a step-change vs trailing
  trend, not as a tolerance breach.

The reason: 0% historical parity is structurally unreachable for a
system with auto-tuned configs and weekly-evolving executor code. Code
drift, config drift, score-snapshot drift, market-data drift, and
daemon-stage logic gaps all contribute to bounded-but-nonzero variance.
Treating that variance as a binary FAIL produces noise alarms each
Saturday and chases tolerance instead of *understanding* divergence.

Tracking variance over time is what's load-bearing: stable variance =
healthy, sudden swing = root-cause investigation. Specific dates can
still be inspected via the per-run report; the metric trend tells us
when it's worth opening that drill-down.

Usage:
    # Unit tests (diff logic + metric computation) — always run
    pytest tests/test_parity_replay.py

    # Full parity replay — opt-in, requires trades.db + S3 ArcticDB access
    pytest tests/test_parity_replay.py -m parity -v

Environment (for parity run):
    TRADES_DB_PATH                override path to trades.db (else download from S3)
    TRADES_DB_S3_URI              e.g. s3://alpha-engine-research/trades/trades_latest.db
    SIGNALS_BUCKET                default "alpha-engine-research"
    PARITY_WINDOW_DAYS            default 10
    PARITY_RUN_DATE               override the run_date column in the time-series CSV
    PARITY_SKIP_METRICS_WRITE=1   skip the time-series append (e.g. dev runs)
"""

from __future__ import annotations

import io
import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

# arcticdb stubbing lives in tests/conftest.py — unit tests get a MagicMock
# by default, integration tests (this file's @pytest.mark.parity case) opt
# in to the real module by setting USE_REAL_ARCTICDB=1 before pytest starts
# (spot_backtest.sh's parity stage does this).


# ── Tolerance contract (see docs/trade_mapping.md for rationale) ────────────

@dataclass(frozen=True)
class Tolerance:
    rel: float          # relative tolerance (fraction, e.g. 0.001 = 0.1%)
    abs_: float = 0.0   # absolute tolerance (same units as the field)


FIELD_TOLERANCES: dict[str, Tolerance] = {
    "fill_price":             Tolerance(rel=0.001, abs_=0.01),
    "price_at_order":         Tolerance(rel=0.001),
    "trigger_price":          Tolerance(rel=0.002),
    "signal_price":           Tolerance(rel=0.001),
    "research_score":         Tolerance(rel=0.0, abs_=0.5),
    "prediction_confidence":  Tolerance(rel=0.0, abs_=0.02),
    "position_pct":           Tolerance(rel=0.0, abs_=0.005),
}

# Shares have rounding at fill time; allow ±1 share
SHARES_ABS_TOLERANCE = 1

# Exact-match fields (any deviation = divergence)
EXACT_FIELDS = {"ticker", "action", "trigger_type", "predicted_direction"}

# Lifecycle fields populated post-trade by the live executor (forward-looking
# from the trade's perspective) that the backtester sim cannot reproduce —
# its replay is a single-shot decision per signal_date with no time advance.
# These are skipped in diff_fields rather than treated as divergence. See
# DATE_CONVENTIONS.md migration notes + ROADMAP "Backtester ↔ executor parity
# divergence" P1.
LIFECYCLE_SKIP_FIELDS = frozenset({
    "days_held",
    "realized_return_pct",
    "realized_pnl",
    "realized_alpha_pct",
    "spy_return_during_hold",
    "fill_time",
    "created_at",
    "ib_order_id",
    "slippage_vs_signal",
    "execution_latency_ms",
    # The legacy `date` column is the calendar fill date (live) vs signal_date
    # (backtester) — they semantically differ until the broader date-convention
    # migration is fully rolled out. Cohort matching uses signal_trading_day,
    # so the per-row `date` field check would always trip on the legacy column.
    "date",
    # Daemon-stage fields. Live populates these when the intraday daemon's
    # trigger (VWAP / pullback / support / time-expiry / graduated_entry)
    # fires and IB returns a fill: trigger_type names the rule that fired,
    # trigger_price is the gate, signal_price is the daemon's snapshot at
    # decision time, fill_price is what IB filled at. Backtester sim runs
    # the morning planner only — none of these exist at planner stage,
    # so they're null in replay output. Including them as required-match
    # fields produced spurious divergence on every cohort-matched ENTER
    # (observed 2026-04-26 ROST 4/12 trade). Phase 2 (entry_triggers.py
    # daily-bar port) will eventually populate them in sim, but until
    # then they're a pure noise floor. ROADMAP P1 "Backtester ↔ executor
    # parity divergence" root cause #3.
    "trigger_type",
    "trigger_price",
    "signal_price",
    "fill_price",
    # Live writes ``prediction_confidence=NaN`` when GBM coverage is
    # missing for the ticker; backtester serializes the same absent
    # value as ``null``. Float comparison treats NaN != null even though
    # they encode the same "no prediction" semantics. Skip until the
    # serialization gap is closed at the writer side.
    "prediction_confidence",
})

# Per-day divergence thresholds. Used by ``diff_trade_count`` /
# ``diff_ticker_sets`` to surface DATES-OF-INTEREST in the report's
# divergence breakdown. NOT used as a pass/fail gate — see "parity is
# observability, not a binary" in the module docstring above.
TRADE_COUNT_PCT_THRESHOLD = 0.05     # 5%
TICKER_SET_PER_DAY_MAX = 1           # >1 ticker differ per day flagged
TICKER_SET_CUMULATIVE_PCT = 0.05     # OR >5% cumulative across window


# ── Pure diff helpers (unit-tested below) ───────────────────────────────────

def within_tolerance(live: float | None, replay: float | None, tol: Tolerance) -> bool:
    """Return True if `live` and `replay` agree within the tolerance."""
    if live is None and replay is None:
        return True
    if live is None or replay is None:
        return False
    diff = abs(float(live) - float(replay))
    if tol.abs_ > 0 and diff <= tol.abs_:
        return True
    if tol.rel > 0 and abs(float(live)) > 1e-9:
        if diff / abs(float(live)) <= tol.rel:
            return True
    # Both thresholds failed (or rel undefined and abs failed)
    return tol.rel == 0 and tol.abs_ == 0  # both-zero means require exact match (handled by exact-match branch)


def diff_trade_count(live_by_date: dict[str, int], replay_by_date: dict[str, int]) -> dict[str, dict]:
    """Per-day trade count divergence above TRADE_COUNT_PCT_THRESHOLD."""
    out: dict[str, dict] = {}
    all_dates = set(live_by_date) | set(replay_by_date)
    for d in sorted(all_dates):
        n_live = live_by_date.get(d, 0)
        n_replay = replay_by_date.get(d, 0)
        denom = max(n_live, 1)
        pct = abs(n_replay - n_live) / denom
        if pct > TRADE_COUNT_PCT_THRESHOLD:
            out[d] = {"live": n_live, "backtester": n_replay,
                      "diff": n_replay - n_live, "pct": round(pct, 4)}
    return out


def diff_ticker_sets(live_by_date: dict[str, set[str]],
                     replay_by_date: dict[str, set[str]]) -> dict[str, dict]:
    """Per-day ticker set symmetric difference above TICKER_SET_PER_DAY_MAX."""
    out: dict[str, dict] = {}
    all_dates = set(live_by_date) | set(replay_by_date)
    for d in sorted(all_dates):
        live_t = live_by_date.get(d, set())
        replay_t = replay_by_date.get(d, set())
        only_live = sorted(live_t - replay_t)
        only_replay = sorted(replay_t - live_t)
        if len(only_live) + len(only_replay) > TICKER_SET_PER_DAY_MAX:
            out[d] = {"only_live": only_live, "only_backtester": only_replay}
    return out


def diff_fields(live_trade: dict, replay_trade: dict) -> dict[str, dict]:
    """Per-field comparison for a single matched trade. Returns
    ``{field: {live, replay, ...}}`` for violations.

    Fields in ``LIFECYCLE_SKIP_FIELDS`` are excluded from the comparison —
    they're populated post-trade by the live executor (e.g. ``days_held``,
    ``realized_return_pct``) and the backtester sim cannot reproduce them.
    Including them would generate noise on every matched ENTER trade.
    """
    violations: dict[str, dict] = {}

    for field in EXACT_FIELDS:
        if field in LIFECYCLE_SKIP_FIELDS:
            continue
        lv, rv = live_trade.get(field), replay_trade.get(field)
        if lv != rv:
            violations[field] = {"live": lv, "backtester": rv, "match_rule": "exact"}

    # Shares special case (integer, ±1 tolerance)
    lv_shares, rv_shares = live_trade.get("shares"), replay_trade.get("shares")
    if lv_shares is not None and rv_shares is not None:
        if abs(int(lv_shares) - int(rv_shares)) > SHARES_ABS_TOLERANCE:
            violations["shares"] = {"live": lv_shares, "backtester": rv_shares,
                                    "threshold_abs": SHARES_ABS_TOLERANCE}

    for field, tol in FIELD_TOLERANCES.items():
        if field in LIFECYCLE_SKIP_FIELDS:
            continue
        lv, rv = live_trade.get(field), replay_trade.get(field)
        if not within_tolerance(lv, rv, tol):
            violations[field] = {"live": lv, "backtester": rv,
                                 "threshold_rel": tol.rel, "threshold_abs": tol.abs_}

    return violations


# ── Variance metrics (parity-as-observability) ──────────────────────────────

def compute_parity_metrics(
    live_by_date: dict[str, int],
    replay_by_date_count: dict[str, int],
    live_tickers_by_date: dict[str, set[str]],
    replay_tickers_by_date: dict[str, set[str]],
    n_field_violations: int,
    n_cohort_matched: int,
    n_lifecycle_skipped: int,
) -> dict[str, float | int]:
    """Reduce the parity diff into 5 trend-friendly numbers.

    Tracked over time in
    ``s3://alpha-engine-research/backtest/parity_metrics.csv`` so anomaly
    is visible as a step-change vs trailing trend, not as a tolerance
    breach. None of these are gated — they describe variance, not pass/
    fail. The historical drift floor is empirically learned from the
    time series itself.

    * ``capture_rate`` = backtester ENTERs / live ENTERs in the window.
      "How many of live's picks did sim see?"
    * ``ticker_jaccard_avg`` = mean per-date ``|L∩B| / |L∪B|`` across
      cohort dates. "How much do the per-date picks overlap?"
    * ``count_divergence_rms`` = RMS of per-date count gap.
      "How wide are the per-date count gaps on average?"
    * ``field_diff_rate`` = field-violation count / cohort-matched ENTER
      count. "Of the trades that match by key, how many disagree on
      sized fields?"
    * ``n_lifecycle_skipped`` = static count of fields excluded from
      field comparison. Tracked so a future skip-list change is visible
      in the metric history.
    """
    n_live_total = sum(live_by_date.values())
    n_back_total = sum(replay_by_date_count.values())
    capture_rate = (n_back_total / n_live_total) if n_live_total else 0.0

    all_dates = sorted(set(live_by_date) | set(replay_by_date_count)
                       | set(live_tickers_by_date) | set(replay_tickers_by_date))

    jaccards: list[float] = []
    for d in all_dates:
        L = live_tickers_by_date.get(d, set())
        B = replay_tickers_by_date.get(d, set())
        union = L | B
        if not union:
            continue
        jaccards.append(len(L & B) / len(union))
    jaccard_avg = sum(jaccards) / len(jaccards) if jaccards else 0.0

    count_diffs_sq: list[float] = []
    for d in all_dates:
        diff = replay_by_date_count.get(d, 0) - live_by_date.get(d, 0)
        count_diffs_sq.append(float(diff * diff))
    count_rms = (sum(count_diffs_sq) / len(count_diffs_sq)) ** 0.5 if count_diffs_sq else 0.0

    field_diff_rate = (n_field_violations / n_cohort_matched) if n_cohort_matched else 0.0

    return {
        "capture_rate": round(capture_rate, 4),
        "ticker_jaccard_avg": round(jaccard_avg, 4),
        "count_divergence_rms": round(count_rms, 4),
        "field_diff_rate": round(field_diff_rate, 4),
        "n_lifecycle_skipped": int(n_lifecycle_skipped),
        "n_live_enters": int(n_live_total),
        "n_backtester_enters": int(n_back_total),
        "n_cohort_matched_enters": int(n_cohort_matched),
        "n_field_violations": int(n_field_violations),
        "n_dates_in_window": int(len(all_dates)),
    }


def append_parity_metrics_row(
    metrics: dict,
    run_date: str,
    bucket: str,
    s3_key: str = "backtest/parity_metrics.csv",
) -> None:
    """Append (or overwrite) one row in the time-series CSV. Idempotent on
    ``run_date``: re-running the same date overwrites prior values, so
    iteration on a single Saturday cohort doesn't pollute the history.

    Schema (additive-only per S3 contract): ``run_date``, then every key
    in ``metrics`` plus the literal ``window_start`` / ``window_end``
    cohort bounds. New metric keys can be added; existing ones never
    rename or remove.

    Best-effort: if S3 access fails the write is skipped with a WARNING
    rather than failing the run. Parity is observability — the metric
    history is nice-to-have but the per-run report.json is the
    authoritative artifact.
    """
    import io
    import logging
    import boto3
    from botocore.exceptions import ClientError

    log = logging.getLogger("parity_metrics")
    s3 = boto3.client("s3")

    columns = ["run_date"] + sorted(metrics.keys())

    existing_df: pd.DataFrame
    try:
        obj = s3.get_object(Bucket=bucket, Key=s3_key)
        existing_df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            existing_df = pd.DataFrame(columns=columns)
        else:
            log.warning("parity_metrics CSV read failed: %s — skipping append", e)
            return
    except Exception as e:
        log.warning("parity_metrics CSV unexpected error: %s — skipping append", e)
        return

    # Drop any prior row for this run_date (idempotent re-runs)
    if "run_date" in existing_df.columns:
        existing_df = existing_df[existing_df["run_date"] != run_date]

    new_row = {"run_date": run_date, **metrics}
    out_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
    # Preserve any historical-only columns (additive contract)
    for col in columns:
        if col not in out_df.columns:
            out_df[col] = None
    out_df = out_df.sort_values("run_date").reset_index(drop=True)

    try:
        s3.put_object(Bucket=bucket, Key=s3_key, Body=out_df.to_csv(index=False).encode())
        log.info("Appended parity metrics for %s to s3://%s/%s (%d rows total)",
                 run_date, bucket, s3_key, len(out_df))
    except Exception as e:
        log.warning("parity_metrics CSV write failed: %s — skipping append", e)


def _emit_degraded_parity_result(
    data_state: str,
    n_live_trades_total: int,
    n_excluded: int,
    bucket: str,
    note: str,
    n_live_enters_matchable: int = 0,
) -> None:
    """Always-emit fallback when the integration test can't run a full
    cohort comparison. Writes a parity_report.json with zero metrics + an
    explicit ``data_state`` field, AND appends a metrics row to the
    time-series CSV so the trend never has gaps.

    Trend behavior: a switch from healthy data_state="ok" to a
    degraded data_state value will show up as zeros in the metrics
    (capture_rate/jaccard/RMS all 0). The data_state field on the
    report drill-down explains why; the operator sees an anomaly in
    the dashboard and reads the report to find out.

    See module docstring "PARITY IS OBSERVABILITY, NOT A GATE" — every
    Saturday SF run must produce one CSV row regardless of data shape.
    Skipping breaks the always-emit contract.
    """
    import json
    metrics = {
        "capture_rate": 0.0,
        "ticker_jaccard_avg": 0.0,
        "count_divergence_rms": 0.0,
        "field_diff_rate": 0.0,
        "n_lifecycle_skipped": int(len(LIFECYCLE_SKIP_FIELDS)),
        "n_live_enters": 0,
        "n_backtester_enters": 0,
        "n_cohort_matched_enters": 0,
        "n_field_violations": 0,
        "n_dates_in_window": 0,
        "data_state": data_state,
    }
    report = {
        "match_key": "(signal_trading_day, ticker, action)",
        "window_signal_trading_days": [None, None],
        "n_live_trades_total": int(n_live_trades_total),
        "n_live_enters_matchable": int(n_live_enters_matchable),
        "n_live_excluded_no_signal_day": int(n_excluded),
        "n_backtester_orders_total": 0,
        "n_backtester_enters": 0,
        "metrics": metrics,
        "lifecycle_fields_skipped": sorted(LIFECYCLE_SKIP_FIELDS),
        "trade_count_divergence": {},
        "ticker_set_divergence": {},
        "field_divergence": [],
        "data_state": data_state,
        "data_state_note": note,
    }
    report_dir = Path(os.environ.get("PARITY_REPORT_DIR", tempfile.gettempdir()))
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "parity_report.json").write_text(json.dumps(report, indent=2, default=str))

    # L4466/config#886: production sets PARITY_RUN_DATE (trading-day-normalized
    # by spot_backtest.sh); the local/test fallback must resolve the trading day
    # too or a weekend run appends a calendar-dated row to parity_metrics.csv.
    run_date = os.environ.get("PARITY_RUN_DATE")
    if not run_date:
        from pipeline_common import resolve_trading_day
        run_date = resolve_trading_day(pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
    if os.environ.get("PARITY_SKIP_METRICS_WRITE") != "1":
        append_parity_metrics_row(metrics, run_date=run_date, bucket=bucket)

    print(f"\nParity DEGRADED state={data_state}: {note}")


# ── Backtester invocation (Phase 1.1b) ──────────────────────────────────────

def _run_backtester_for_dates(dates: list[str], bucket: str,
                              config_path: str | None = None,
                              trades_db_path: str | None = None) -> list[dict]:
    """Replay the backtester for each date, return the aggregated order list.

    Thin wrapper over ``backtest.replay_for_dates`` — loads config, overrides
    the signals_bucket for the caller's convenience, delegates to the helper
    that factors the per-date orchestration out of ``_run_simulation_loop``.

    Requires ``executor_paths`` in config.yaml to point to a live
    ``alpha-engine`` checkout — the backtester imports the executor directly
    rather than reimplementing it, so ``simulate=True`` actually exercises
    live executor code.

    ``trades_db_path``: when provided, sim bootstraps initial positions/cash
    from ``eod_pnl``'s most recent snapshot strictly before the parity
    window. Replaces the cold-start warmup (Option A long-term parity
    strategy — see ``_load_initial_state_from_eod_pnl`` docstring in
    backtest.py).

    Set ``PARITY_PER_DATE_BOOTSTRAP=1`` in the environment to opt into the
    per-parity-date bootstrap mode (fresh sim_client per parity date,
    anchored to eod_pnl's preceding-day snapshot). Param-sweep paths
    continue using the continuous-state default.
    """
    from pipeline_common import load_config
    import backtest as _bt

    cfg_path = config_path or os.environ.get("BACKTESTER_CONFIG", "config.yaml")
    config = load_config(cfg_path)
    if bucket:
        config["signals_bucket"] = bucket
    if trades_db_path:
        config["trades_db_path"] = trades_db_path

    per_date = os.environ.get("PARITY_PER_DATE_BOOTSTRAP", "").strip() == "1"
    return _bt.replay_for_dates(sorted(dates), config, per_date_bootstrap=per_date)


# ── trades.db access ────────────────────────────────────────────────────────

def _load_trades_from_db(db_path: str, since_date: str | None = None) -> pd.DataFrame:
    """Read the `trades` table. Returns a DataFrame (empty if table missing)."""
    conn = sqlite3.connect(db_path)
    try:
        q = "SELECT * FROM trades"
        params: tuple = ()
        if since_date:
            q += " WHERE date >= ?"
            params = (since_date,)
        q += " ORDER BY date, ticker"
        return pd.read_sql_query(q, conn, params=params)
    except pd.errors.DatabaseError as exc:
        # Graceful: empty DataFrame if table missing (first boot)
        if "no such table" in str(exc).lower():
            return pd.DataFrame()
        raise
    finally:
        conn.close()


def _last_n_trading_dates(trades_df: pd.DataFrame, n: int) -> list[str]:
    """Return the last n unique signal_trading_day values from the trades
    DataFrame — the parity cohort key per DATE_CONVENTIONS.md.

    Falls back to the legacy ``date`` column when ``signal_trading_day`` is
    missing or empty (pre-migration DBs). Operators running the parity test
    against a pre-PR-2 trades.db will see zero matchable rows downstream and
    the integration test will skip with a clear message.
    """
    if trades_df.empty:
        return []
    if "signal_trading_day" in trades_df.columns:
        col_values = trades_df["signal_trading_day"].dropna()
        if not col_values.empty:
            return sorted(col_values.unique())[-n:]
    # Fallback for pre-migration DBs — legacy `date` column. Test will
    # filter to ENTERs with non-null signal_trading_day downstream and skip
    # if nothing matches.
    return sorted(trades_df["date"].dropna().unique())[-n:]


# ── Unit tests (pure diff logic) — always run ───────────────────────────────

class TestWithinTolerance:
    def test_both_none_matches(self):
        assert within_tolerance(None, None, Tolerance(rel=0.001, abs_=0.01))

    def test_one_none_fails(self):
        assert not within_tolerance(None, 100.0, Tolerance(rel=0.001))
        assert not within_tolerance(100.0, None, Tolerance(rel=0.001))

    def test_within_rel_passes(self):
        # 0.08% delta under 0.1% threshold
        assert within_tolerance(100.0, 100.08, Tolerance(rel=0.001))

    def test_within_abs_passes(self):
        # $0.005 delta under $0.01 threshold
        assert within_tolerance(100.000, 100.005, Tolerance(rel=0.0, abs_=0.01))

    def test_outside_rel_and_abs_fails(self):
        # 0.5% delta, exceeds both
        assert not within_tolerance(100.0, 100.5, Tolerance(rel=0.001, abs_=0.01))

    def test_rel_with_abs_fallback_wins(self):
        # Very small absolute OK even though relative exceeds
        assert within_tolerance(100.0, 100.005, Tolerance(rel=0.00001, abs_=0.01))


class TestDiffTradeCount:
    def test_below_threshold_returns_empty(self):
        # 4% diff, under 5% threshold
        live = {"2026-04-10": 100}
        replay = {"2026-04-10": 96}
        assert diff_trade_count(live, replay) == {}

    def test_above_threshold_reported(self):
        # 10% diff
        live = {"2026-04-10": 100}
        replay = {"2026-04-10": 90}
        out = diff_trade_count(live, replay)
        assert "2026-04-10" in out
        assert out["2026-04-10"]["diff"] == -10

    def test_date_only_on_one_side(self):
        # 100% divergence — always exceeds threshold
        live = {"2026-04-10": 5}
        replay = {}
        out = diff_trade_count(live, replay)
        assert "2026-04-10" in out
        assert out["2026-04-10"]["backtester"] == 0


class TestDiffTickerSets:
    def test_zero_diff_returns_empty(self):
        live = {"2026-04-10": {"AAPL", "MSFT"}}
        replay = {"2026-04-10": {"AAPL", "MSFT"}}
        assert diff_ticker_sets(live, replay) == {}

    def test_single_diff_under_threshold(self):
        # 1 ticker differ — at threshold, not exceeding
        live = {"2026-04-10": {"AAPL", "MSFT"}}
        replay = {"2026-04-10": {"AAPL", "MSFT", "NVDA"}}
        assert diff_ticker_sets(live, replay) == {}

    def test_two_diffs_reported(self):
        live = {"2026-04-10": {"AAPL", "MSFT"}}
        replay = {"2026-04-10": {"AAPL", "NVDA", "PLTR"}}
        out = diff_ticker_sets(live, replay)
        assert "2026-04-10" in out
        assert out["2026-04-10"]["only_live"] == ["MSFT"]
        assert out["2026-04-10"]["only_backtester"] == ["NVDA", "PLTR"]


class TestDiffFields:
    def _base_trade(self, **kwargs):
        base = {
            "date": "2026-04-10", "ticker": "AAPL", "action": "ENTER",
            "shares": 100, "fill_price": 172.34, "price_at_order": 172.30,
            "trigger_type": "pullback", "trigger_price": 172.00,
            "signal_price": 172.50, "research_score": 78.0,
            "predicted_direction": "UP", "prediction_confidence": 0.65,
            "position_pct": 0.05, "realized_return_pct": None, "days_held": None,
        }
        base.update(kwargs)
        return base

    def test_identical_trades_no_violations(self):
        a = self._base_trade()
        b = self._base_trade()
        assert diff_fields(a, b) == {}

    def test_price_at_order_within_rel(self):
        # price_at_order is the planner-stage price both sides should have;
        # fill_price moved to LIFECYCLE_SKIP_FIELDS so it's no longer the
        # canonical "rel-tolerance compared field" in the test surface.
        a = self._base_trade(price_at_order=172.30)
        b = self._base_trade(price_at_order=172.39)  # 0.052% — under 0.1%
        assert diff_fields(a, b) == {}

    def test_price_at_order_outside_rel(self):
        a = self._base_trade(price_at_order=172.30)
        b = self._base_trade(price_at_order=173.00)  # 0.41% — exceeds 0.1%
        v = diff_fields(a, b)
        assert "price_at_order" in v

    def test_action_exact_mismatch(self):
        a = self._base_trade(action="ENTER")
        b = self._base_trade(action="EXIT")
        v = diff_fields(a, b)
        assert "action" in v and v["action"]["match_rule"] == "exact"

    def test_shares_within_one(self):
        a = self._base_trade(shares=100)
        b = self._base_trade(shares=101)
        assert diff_fields(a, b) == {}

    def test_shares_beyond_one(self):
        a = self._base_trade(shares=100)
        b = self._base_trade(shares=102)
        v = diff_fields(a, b)
        assert "shares" in v

    def test_predicted_direction_exact_mismatch(self):
        # Replaces the prior test_trigger_type_null_vs_value — trigger_type
        # is now in LIFECYCLE_SKIP_FIELDS. predicted_direction stays in
        # EXACT_FIELDS as the remaining backtester-vs-live exact-match
        # field that's not skipped, so it stands in for the same code path.
        a = self._base_trade(predicted_direction="UP")
        b = self._base_trade(predicted_direction="DOWN")
        v = diff_fields(a, b)
        assert "predicted_direction" in v and v["predicted_direction"]["match_rule"] == "exact"


class TestLoadTradesFromDB:
    def _write_trades(self, path, rows):
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE trades (date TEXT, ticker TEXT, action TEXT, shares INTEGER, fill_price REAL)")
        conn.executemany("INSERT INTO trades VALUES (?,?,?,?,?)", rows)
        conn.commit()
        conn.close()

    def test_empty_table(self, tmp_path):
        db = tmp_path / "trades.db"
        self._write_trades(str(db), [])
        df = _load_trades_from_db(str(db))
        assert df.empty

    def test_missing_table_returns_empty(self, tmp_path):
        db = tmp_path / "empty.db"
        sqlite3.connect(str(db)).close()
        df = _load_trades_from_db(str(db))
        assert df.empty

    def test_since_date_filter(self, tmp_path):
        db = tmp_path / "trades.db"
        self._write_trades(str(db), [
            ("2026-04-01", "AAPL", "ENTER", 100, 170.0),
            ("2026-04-10", "MSFT", "ENTER", 50, 400.0),
        ])
        df = _load_trades_from_db(str(db), since_date="2026-04-05")
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "MSFT"


class TestLastNTradingDates:
    def test_empty(self):
        assert _last_n_trading_dates(pd.DataFrame(), 5) == []

    def test_takes_latest_n(self):
        df = pd.DataFrame({"date": ["2026-04-01", "2026-04-02", "2026-04-03",
                                     "2026-04-04", "2026-04-05"]})
        assert _last_n_trading_dates(df, 3) == ["2026-04-03", "2026-04-04", "2026-04-05"]

    def test_prefers_signal_trading_day_when_present(self):
        """Post-PR-2 DBs have signal_trading_day; cohort matching uses it."""
        df = pd.DataFrame({
            "date": ["2026-04-13", "2026-04-14", "2026-04-15", "2026-04-16", "2026-04-17"],
            "signal_trading_day": ["2026-04-10", "2026-04-10", "2026-04-10", "2026-04-17", "2026-04-17"],
        })
        # Should return unique signal_trading_days, not unique fill dates.
        result = _last_n_trading_dates(df, 5)
        assert result == ["2026-04-10", "2026-04-17"]

    def test_falls_back_to_date_when_signal_trading_day_all_null(self):
        """Pre-backfill DB might have the column but with all NULLs.
        Falls back to the legacy `date` column so the test can still
        function (and the integration path skips later if no matchable
        rows remain after the ENTER + non-null filter)."""
        df = pd.DataFrame({
            "date": ["2026-04-13", "2026-04-14", "2026-04-15"],
            "signal_trading_day": [None, None, None],
        })
        assert _last_n_trading_dates(df, 5) == ["2026-04-13", "2026-04-14", "2026-04-15"]


class TestLifecycleSkipFields:
    """Lifecycle fields populated post-trade by the live executor are
    excluded from diff comparison — they can never match backtester sim."""

    def _trade(self, **kwargs):
        base = {"date": "2026-04-13", "ticker": "AAPL", "action": "ENTER", "shares": 100}
        base.update(kwargs)
        return base

    def test_days_held_difference_not_flagged(self):
        live = self._trade(days_held=5)
        replay = self._trade(days_held=None)
        assert "days_held" not in diff_fields(live, replay)

    def test_realized_return_pct_difference_not_flagged(self):
        live = self._trade(realized_return_pct=2.5)
        replay = self._trade(realized_return_pct=None)
        assert "realized_return_pct" not in diff_fields(live, replay)

    def test_fill_time_difference_not_flagged(self):
        live = self._trade(fill_time="2026-04-13T14:30:00+00:00")
        replay = self._trade(fill_time=None)
        assert "fill_time" not in diff_fields(live, replay)

    def test_legacy_date_difference_not_flagged(self):
        # Live's `date` is the calendar fill day; backtester's is signal_date.
        # Cohort matching uses signal_trading_day, so the per-row `date`
        # column is irrelevant to the diff and is in LIFECYCLE_SKIP_FIELDS.
        live = self._trade(date="2026-04-20")
        replay = self._trade(date="2026-04-17")
        assert "date" not in diff_fields(live, replay)

    def test_non_lifecycle_field_still_compared(self):
        # ticker is exact-match; this confirms the skip-set doesn't
        # accidentally swallow real comparisons.
        live = self._trade(ticker="AAPL")
        replay = self._trade(ticker="MSFT")
        v = diff_fields(live, replay)
        assert "ticker" in v

    def test_trigger_type_difference_not_flagged(self):
        # Daemon-stage field — backtester sim cannot produce a trigger_type
        # at planner stage. ROST 4/12 incident 2026-04-26.
        live = self._trade(trigger_type="graduated_entry (+0.0% vs morning)")
        replay = self._trade(trigger_type=None)
        assert "trigger_type" not in diff_fields(live, replay)

    def test_trigger_price_difference_not_flagged(self):
        live = self._trade(trigger_price=220.59)
        replay = self._trade(trigger_price=None)
        assert "trigger_price" not in diff_fields(live, replay)

    def test_signal_price_difference_not_flagged(self):
        live = self._trade(signal_price=220.56)
        replay = self._trade(signal_price=None)
        assert "signal_price" not in diff_fields(live, replay)

    def test_fill_price_difference_not_flagged(self):
        # Live fill_price comes from IB at fill time; backtester sim has
        # no fill stage. Skip — same posture as the other daemon-stage
        # fields; replaces the prior FIELD_TOLERANCES entry.
        live = self._trade(fill_price=220.53)
        replay = self._trade(fill_price=None)
        assert "fill_price" not in diff_fields(live, replay)

    def test_prediction_confidence_nan_vs_null_not_flagged(self):
        # Live writes NaN when GBM coverage missing; backtester serializes
        # the same absent value as None. Float-compare treats them as
        # different even though the semantics match.
        live = self._trade(prediction_confidence=float("nan"))
        replay = self._trade(prediction_confidence=None)
        assert "prediction_confidence" not in diff_fields(live, replay)


class TestComputeParityMetrics:
    """Pure function — no S3, no I/O. The arithmetic that turns the diff
    state into trend-friendly numbers."""

    def test_perfect_match_yields_unit_capture_zero_diff(self):
        live = {"2026-04-13": 5, "2026-04-20": 7}
        rep = {"2026-04-13": 5, "2026-04-20": 7}
        live_t = {"2026-04-13": {"AAPL", "MSFT"}, "2026-04-20": {"NVDA"}}
        rep_t = {"2026-04-13": {"AAPL", "MSFT"}, "2026-04-20": {"NVDA"}}
        m = compute_parity_metrics(live, rep, live_t, rep_t,
                                   n_field_violations=0, n_cohort_matched=10,
                                   n_lifecycle_skipped=16)
        assert m["capture_rate"] == 1.0
        assert m["ticker_jaccard_avg"] == 1.0
        assert m["count_divergence_rms"] == 0.0
        assert m["field_diff_rate"] == 0.0
        assert m["n_dates_in_window"] == 2

    def test_capture_rate_under_1_when_backtester_undershoots(self):
        live = {"2026-04-13": 10}
        rep = {"2026-04-13": 6}
        m = compute_parity_metrics(live, rep, {}, {}, 0, 0, 0)
        assert m["capture_rate"] == 0.6

    def test_jaccard_partial_overlap(self):
        live_t = {"2026-04-13": {"AAPL", "MSFT", "NVDA"}}
        rep_t = {"2026-04-13": {"AAPL", "MSFT", "TSLA", "GOOG"}}
        # |L ∩ B| = 2 (AAPL, MSFT); |L ∪ B| = 5 → 0.4
        m = compute_parity_metrics({}, {}, live_t, rep_t, 0, 0, 0)
        assert m["ticker_jaccard_avg"] == 0.4

    def test_count_rms_two_dates(self):
        live = {"2026-04-13": 10, "2026-04-20": 5}
        rep = {"2026-04-13": 6, "2026-04-20": 3}
        # diffs: -4, -2 → RMS = sqrt((16+4)/2) = sqrt(10) ≈ 3.1623
        m = compute_parity_metrics(live, rep, {}, {}, 0, 0, 0)
        assert abs(m["count_divergence_rms"] - 3.1623) < 0.001

    def test_field_diff_rate_zero_matched_means_zero(self):
        m = compute_parity_metrics({}, {}, {}, {}, n_field_violations=5,
                                   n_cohort_matched=0, n_lifecycle_skipped=0)
        assert m["field_diff_rate"] == 0.0

    def test_field_diff_rate_partial(self):
        m = compute_parity_metrics({}, {}, {}, {}, n_field_violations=3,
                                   n_cohort_matched=10, n_lifecycle_skipped=0)
        assert m["field_diff_rate"] == 0.3

    def test_dates_only_on_one_side_still_counted(self):
        # backtester sees a date live didn't (or vice-versa) — should still
        # appear in the union and contribute to averages.
        live = {"2026-04-13": 5}
        rep = {"2026-04-13": 5, "2026-04-20": 0}  # extra date with 0 picks
        live_t = {"2026-04-13": {"AAPL"}}
        rep_t = {"2026-04-13": {"AAPL"}, "2026-04-20": set()}
        m = compute_parity_metrics(live, rep, live_t, rep_t, 0, 0, 0)
        # 2026-04-20 has empty union, skipped from jaccard average
        assert m["ticker_jaccard_avg"] == 1.0
        # n_dates_in_window includes both
        assert m["n_dates_in_window"] == 2

    def test_zero_live_doesnt_divide_by_zero(self):
        m = compute_parity_metrics({}, {"2026-04-13": 3}, {}, {}, 0, 0, 0)
        # No live picks — capture_rate is 0 by convention, not nan
        assert m["capture_rate"] == 0.0


class TestAppendParityMetricsRow:
    """Cover the time-series writer's idempotency + schema discipline."""

    def _patch_s3(self, monkeypatch, store: dict):
        """Patch boto3.client to return a fake S3 backed by a dict.
        ``store`` is keyed by S3 key; each value is the bytes of the object."""
        from unittest.mock import MagicMock
        from botocore.exceptions import ClientError

        client = MagicMock()

        def get_object(Bucket, Key):
            if Key not in store:
                err = ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
                raise err
            from io import BytesIO
            return {"Body": BytesIO(store[Key])}

        def put_object(Bucket, Key, Body):
            store[Key] = Body
            return {}

        client.get_object.side_effect = get_object
        client.put_object.side_effect = put_object

        import boto3
        monkeypatch.setattr(boto3, "client", lambda *a, **kw: client)
        return client

    def test_first_run_creates_csv(self, monkeypatch):
        store: dict = {}
        self._patch_s3(monkeypatch, store)
        metrics = {"capture_rate": 0.6, "ticker_jaccard_avg": 0.5,
                   "count_divergence_rms": 2.0, "field_diff_rate": 0.1,
                   "n_lifecycle_skipped": 16}
        append_parity_metrics_row(metrics, run_date="2026-04-26", bucket="b")
        assert "backtest/parity_metrics.csv" in store
        df = pd.read_csv(io.BytesIO(store["backtest/parity_metrics.csv"]))
        assert len(df) == 1
        assert df.iloc[0]["run_date"] == "2026-04-26"
        assert df.iloc[0]["capture_rate"] == 0.6

    def test_subsequent_run_appends_row(self, monkeypatch):
        # Pre-populate with one row
        seed = pd.DataFrame([{"run_date": "2026-04-20", "capture_rate": 0.4,
                              "ticker_jaccard_avg": 0.3, "count_divergence_rms": 5.0,
                              "field_diff_rate": 0.2, "n_lifecycle_skipped": 16}])
        store = {"backtest/parity_metrics.csv": seed.to_csv(index=False).encode()}
        self._patch_s3(monkeypatch, store)
        metrics = {"capture_rate": 0.6, "ticker_jaccard_avg": 0.5,
                   "count_divergence_rms": 2.0, "field_diff_rate": 0.1,
                   "n_lifecycle_skipped": 16}
        append_parity_metrics_row(metrics, run_date="2026-04-26", bucket="b")
        df = pd.read_csv(io.BytesIO(store["backtest/parity_metrics.csv"]))
        assert len(df) == 2
        # Sorted by run_date
        assert list(df["run_date"]) == ["2026-04-20", "2026-04-26"]

    def test_re_run_same_date_is_idempotent(self, monkeypatch):
        # Re-running today's parity overwrites today's row (no duplicates).
        seed = pd.DataFrame([{"run_date": "2026-04-26", "capture_rate": 0.4,
                              "ticker_jaccard_avg": 0.3, "count_divergence_rms": 5.0,
                              "field_diff_rate": 0.2, "n_lifecycle_skipped": 16}])
        store = {"backtest/parity_metrics.csv": seed.to_csv(index=False).encode()}
        self._patch_s3(monkeypatch, store)
        metrics = {"capture_rate": 0.6, "ticker_jaccard_avg": 0.5,
                   "count_divergence_rms": 2.0, "field_diff_rate": 0.1,
                   "n_lifecycle_skipped": 16}
        append_parity_metrics_row(metrics, run_date="2026-04-26", bucket="b")
        df = pd.read_csv(io.BytesIO(store["backtest/parity_metrics.csv"]))
        assert len(df) == 1  # overwritten, not duplicated
        assert df.iloc[0]["capture_rate"] == 0.6  # new value

    def test_s3_error_swallowed_not_raised(self, monkeypatch):
        # Best-effort: a write failure must not break the test path.
        from unittest.mock import MagicMock
        client = MagicMock()
        client.get_object.side_effect = RuntimeError("boom")
        import boto3
        monkeypatch.setattr(boto3, "client", lambda *a, **kw: client)

        # Should NOT raise
        append_parity_metrics_row({"capture_rate": 0.5}, run_date="2026-04-26", bucket="b")


class TestEmitDegradedParityResult:
    """The data-shape conditions formerly handled by ``pytest.skip`` now
    flow through ``_emit_degraded_parity_result``. The invariant: every
    invocation produces a parity_report.json AND a metrics-CSV row, even
    on degraded data states. Skipping breaks the always-emit contract."""

    def _patch_s3(self, monkeypatch, store: dict):
        from unittest.mock import MagicMock
        from botocore.exceptions import ClientError
        client = MagicMock()

        def get_object(Bucket, Key):
            if Key not in store:
                raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
            return {"Body": io.BytesIO(store[Key])}

        def put_object(Bucket, Key, Body):
            store[Key] = Body
            return {}

        client.get_object.side_effect = get_object
        client.put_object.side_effect = put_object
        import boto3
        monkeypatch.setattr(boto3, "client", lambda *a, **kw: client)

    def test_writes_report_with_data_state(self, monkeypatch, tmp_path):
        store: dict = {}
        self._patch_s3(monkeypatch, store)
        monkeypatch.setenv("PARITY_REPORT_DIR", str(tmp_path))
        monkeypatch.setenv("PARITY_RUN_DATE", "2026-04-26")
        _emit_degraded_parity_result(
            data_state="empty_trades_db",
            n_live_trades_total=0,
            n_excluded=0,
            bucket="test-bucket",
            note="trades.db has 0 rows",
        )
        report_path = tmp_path / "parity_report.json"
        assert report_path.exists()
        import json
        report = json.loads(report_path.read_text())
        assert report["data_state"] == "empty_trades_db"
        assert report["metrics"]["data_state"] == "empty_trades_db"
        assert report["metrics"]["capture_rate"] == 0.0
        assert report["n_live_trades_total"] == 0

    def test_appends_metrics_row(self, monkeypatch, tmp_path):
        store: dict = {}
        self._patch_s3(monkeypatch, store)
        monkeypatch.setenv("PARITY_REPORT_DIR", str(tmp_path))
        monkeypatch.setenv("PARITY_RUN_DATE", "2026-04-26")
        _emit_degraded_parity_result(
            data_state="insufficient_cohort_dates",
            n_live_trades_total=10,
            n_excluded=8,
            n_live_enters_matchable=2,
            bucket="test-bucket",
            note="Only 2 cohort dates",
        )
        # Metrics row appended
        assert "backtest/parity_metrics.csv" in store
        df = pd.read_csv(io.BytesIO(store["backtest/parity_metrics.csv"]))
        assert len(df) == 1
        assert df.iloc[0]["data_state"] == "insufficient_cohort_dates"
        assert df.iloc[0]["capture_rate"] == 0.0
        assert df.iloc[0]["run_date"] == "2026-04-26"

    def test_backtester_replay_error_emits_report_and_row(self, monkeypatch, tmp_path):
        # L3147: a raising replay (e.g. cohort ticker with NaN ATR hard-fails
        # decide_entries) must still produce parity_report.json + a metrics
        # row so the trend shows a step-change, not a silent gap.
        store: dict = {}
        self._patch_s3(monkeypatch, store)
        monkeypatch.setenv("PARITY_REPORT_DIR", str(tmp_path))
        monkeypatch.setenv("PARITY_RUN_DATE", "2026-05-30")
        _emit_degraded_parity_result(
            data_state="backtester_replay_error",
            n_live_trades_total=42,
            n_excluded=5,
            n_live_enters_matchable=37,
            bucket="test-bucket",
            note="Backtester replay raised RuntimeError: atr_map missing PRU",
        )
        import json
        report = json.loads((tmp_path / "parity_report.json").read_text())
        assert report["data_state"] == "backtester_replay_error"
        assert report["metrics"]["capture_rate"] == 0.0
        assert report["n_live_trades_total"] == 42
        df = pd.read_csv(io.BytesIO(store["backtest/parity_metrics.csv"]))
        assert df.iloc[0]["data_state"] == "backtester_replay_error"
        assert df.iloc[0]["run_date"] == "2026-05-30"

    def test_skip_metrics_write_env_honored(self, monkeypatch, tmp_path):
        # Local dev runs may want to skip the S3 append; the env flag
        # must still produce a per-run report.
        store: dict = {}
        self._patch_s3(monkeypatch, store)
        monkeypatch.setenv("PARITY_REPORT_DIR", str(tmp_path))
        monkeypatch.setenv("PARITY_SKIP_METRICS_WRITE", "1")
        _emit_degraded_parity_result(
            data_state="empty_trades_db",
            n_live_trades_total=0, n_excluded=0,
            bucket="test-bucket", note="dev run",
        )
        # Report still written
        assert (tmp_path / "parity_report.json").exists()
        # CSV NOT written
        assert "backtest/parity_metrics.csv" not in store


# ── Integration test (opt-in) ───────────────────────────────────────────────

@pytest.mark.parity
@pytest.mark.live
def test_parity_replay_end_to_end():
    """Full parity test — replays backtester over last N live trade dates.

    Opt-in via `pytest -m parity`. Requires:
      * trades.db reachable (TRADES_DB_PATH or S3 download)
      * ArcticDB live (SIGNALS_BUCKET in AWS)
      * Backtester-invocation helper wired (Phase 1.1b)
    """
    bucket = os.environ.get("SIGNALS_BUCKET", "alpha-engine-research")
    window_days = int(os.environ.get("PARITY_WINDOW_DAYS", "10"))

    # Resolve trades.db
    db_path = os.environ.get("TRADES_DB_PATH")
    if not db_path:
        pytest.skip("TRADES_DB_PATH not set — S3 download not yet wired (Phase 1.1b)")

    if not Path(db_path).exists():
        pytest.skip(f"trades.db not found at {db_path}")

    # ── Pre-flight invariant: schema must support cohort matching. ────────
    # signal_trading_day was added by alpha-engine PR #98 (DATE_CONVENTIONS
    # migration, deployed 2026-04-24) and backfilled the same day. If a
    # current trades.db lacks that column, something has regressed (rolled
    # back deploy, restored stale snapshot, wrong trades.db file). Raise
    # rather than skip so the SF alarm fires — this is real infra breakage,
    # NOT bounded variance.
    trades_df = _load_trades_from_db(db_path)
    if "signal_trading_day" not in trades_df.columns:
        raise RuntimeError(
            f"trades.db at {db_path} is missing the signal_trading_day column. "
            f"Schema regression — alpha-engine PR #98 (DATE_CONVENTIONS) added "
            f"this column 2026-04-24. Verify trades.db is current; do NOT "
            f"silently skip. Spot run should fail."
        )

    # ── Data-shape conditions: emit a degraded but always-present result.
    # Per "parity is observability": every Saturday SF run MUST append a
    # row to parity_metrics.csv so a missing data-shape (empty cohort,
    # too few dates) shows up as a step-change in the trend, not as a
    # silently-skipped run that leaves a gap in the time series.
    # Schema for the degraded path is identical to the full path; missing
    # values are zeros + an explicit ``data_state`` field on the report.
    matchable = trades_df[
        (trades_df["action"] == "ENTER")
        & trades_df["signal_trading_day"].notna()
    ]
    n_excluded = len(trades_df) - len(matchable)

    if trades_df.empty or matchable.empty:
        data_state = (
            "empty_trades_db" if trades_df.empty
            else "no_matchable_enter_rows"
        )
        _emit_degraded_parity_result(
            data_state=data_state,
            n_live_trades_total=len(trades_df),
            n_excluded=n_excluded,
            bucket=bucket,
            note=(
                f"trades.db has {len(trades_df)} rows total, "
                f"{len(matchable)} matchable ENTERs (action='ENTER' AND "
                f"signal_trading_day NOT NULL)."
            ),
        )
        return

    dates = _last_n_trading_dates(matchable, window_days)
    if len(dates) < 3:
        _emit_degraded_parity_result(
            data_state="insufficient_cohort_dates",
            n_live_trades_total=len(trades_df),
            n_excluded=n_excluded,
            n_live_enters_matchable=len(matchable),
            bucket=bucket,
            note=(
                f"Only {len(dates)} signal_trading_day(s) in matchable rows; "
                f"need >=3 for a meaningful cohort. Trend will show this as "
                f"low data points until live history accumulates."
            ),
        )
        return

    # Run backtester replay over the same signal_trading_day cohorts.
    # backtest.replay_for_dates iterates the input dates as signal_dates,
    # producing orders tagged with `o["date"] = signal_date` — which equals
    # signal_trading_day on the live side post-backfill. The cohort key
    # matches across both sides without further translation.
    #
    # Always-emit contract (sibling to the pit_parity 5/17→5/27 silence):
    # the replay can raise — e.g. a cohort ticker whose latest atr_14_pct is
    # NaN/non-positive hard-fails decide_entries with `atr_map missing
    # {ticker}` (L3147). Before this guard the exception propagated out of
    # the test, skipping the parity_report.json + metrics-CSV write entirely,
    # so the time series went silent for weeks. Catch, emit a degraded result
    # naming the error (data_state="backtester_replay_error"), and return so
    # the trend shows a step-change instead of a gap. feature_maps WARN-logs
    # the offending ticker + reason at load time (full context lives there).
    try:
        replay_orders = _run_backtester_for_dates(dates, bucket, trades_db_path=db_path)
    except Exception as exc:
        _emit_degraded_parity_result(
            data_state="backtester_replay_error",
            n_live_trades_total=len(trades_df),
            n_excluded=n_excluded,
            n_live_enters_matchable=len(matchable),
            bucket=bucket,
            note=(
                f"Backtester replay raised {exc.__class__.__name__}: {exc}. "
                f"Window={dates[0]}..{dates[-1]} ({len(dates)} cohort dates). "
                f"If 'atr_map missing <TICKER>', see feature_maps WARN for the "
                f"drop reason (NaN/non-positive latest atr_14_pct) — L3147."
            ),
        )
        return

    # Filter both sides to ENTERs for cohort matching. Exits don't have
    # signal_trading_day on the live side (NULL by design), so cohort
    # comparison is ENTER-only. Backtester also produces exits but those
    # are tagged with the same signal_date and could be cohort-matched
    # in a future expansion.
    matchable_enters = matchable
    replay_enters = [o for o in replay_orders if o.get("action") == "ENTER"]

    window = matchable_enters[matchable_enters["signal_trading_day"].isin(dates)]
    live_by_date: dict[str, int] = window.groupby("signal_trading_day").size().to_dict()
    replay_by_date_count: dict[str, int] = {}
    replay_tickers_by_date: dict[str, set[str]] = {}
    for o in replay_enters:
        d = o["date"]
        replay_by_date_count[d] = replay_by_date_count.get(d, 0) + 1
        replay_tickers_by_date.setdefault(d, set()).add(o["ticker"])

    live_tickers_by_date = {
        d: set(window[window["signal_trading_day"] == d]["ticker"].tolist())
        for d in dates
    }

    count_violations = diff_trade_count(live_by_date, replay_by_date_count)
    ticker_violations = diff_ticker_sets(live_tickers_by_date, replay_tickers_by_date)

    # Field-level diffs on matched trades — keyed on
    # (signal_trading_day, ticker, action). Lifecycle fields (days_held,
    # realized_return_pct, etc.) are excluded by LIFECYCLE_SKIP_FIELDS in
    # diff_fields — they're populated post-trade by the live executor and
    # the backtester sim cannot reproduce them.
    field_violations: list[dict] = []
    n_cohort_matched = 0
    for _, row in window.iterrows():
        key = (row["signal_trading_day"], row["ticker"], row["action"])
        match = next(
            (
                o for o in replay_enters
                if (o.get("date"), o.get("ticker"), o.get("action")) == key
            ),
            None,
        )
        if match is None:
            continue
        n_cohort_matched += 1
        vs = diff_fields(row.to_dict(), match)
        if vs:
            field_violations.append({
                "signal_trading_day": row["signal_trading_day"],
                "ticker": row["ticker"],
                "action": row["action"],
                "fields": vs,
            })

    metrics = compute_parity_metrics(
        live_by_date=live_by_date,
        replay_by_date_count=replay_by_date_count,
        live_tickers_by_date=live_tickers_by_date,
        replay_tickers_by_date=replay_tickers_by_date,
        n_field_violations=len(field_violations),
        n_cohort_matched=n_cohort_matched,
        n_lifecycle_skipped=len(LIFECYCLE_SKIP_FIELDS),
    )
    # Schema parity with _emit_degraded_parity_result: always include
    # data_state so the time-series CSV column is consistent across paths.
    metrics["data_state"] = "ok"

    # Parity is observability, not a gate. The report still surfaces
    # divergence categories (count / ticker-set / field) at the prior
    # thresholds — those let the operator drill into specific dates —
    # but the test does NOT fail on them. Trends in ``metrics`` over
    # time (s3://.../backtest/parity_metrics.csv) are the load-bearing
    # signal: stable variance = healthy, step-change = investigate.
    report = {
        "match_key": "(signal_trading_day, ticker, action)",
        "window_signal_trading_days": [dates[0], dates[-1]],
        "n_live_trades_total": int(len(trades_df)),
        "n_live_enters_matchable": int(len(matchable)),
        "n_live_excluded_no_signal_day": int(n_excluded),
        "n_backtester_orders_total": len(replay_orders),
        "n_backtester_enters": len(replay_enters),
        "metrics": metrics,
        "lifecycle_fields_skipped": sorted(LIFECYCLE_SKIP_FIELDS),
        "trade_count_divergence": count_violations,
        "ticker_set_divergence": ticker_violations,
        "field_divergence": field_violations,
    }

    # Write report for the spot-instance post-run hook
    report_dir = Path(os.environ.get("PARITY_REPORT_DIR", tempfile.gettempdir()))
    report_dir.mkdir(parents=True, exist_ok=True)
    import json
    (report_dir / "parity_report.json").write_text(json.dumps(report, indent=2, default=str))

    # Append the time-series row for trend tracking. Best-effort: S3 errors
    # don't fail the test (the per-run report is the authoritative artifact).
    # L4466/config#886: production sets PARITY_RUN_DATE (trading-day-normalized
    # by spot_backtest.sh); the local/test fallback must resolve the trading day
    # too or a weekend run appends a calendar-dated row to parity_metrics.csv.
    run_date = os.environ.get("PARITY_RUN_DATE")
    if not run_date:
        from pipeline_common import resolve_trading_day
        run_date = resolve_trading_day(pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
    if os.environ.get("PARITY_SKIP_METRICS_WRITE") != "1":
        append_parity_metrics_row(metrics, run_date=run_date, bucket=bucket)

    # Always-pass — the test's job is to GENERATE the report + metrics row.
    # Nothing here gates the spot run; SF stops alarming on parity drift.
    print(
        f"\nParity metrics: capture_rate={metrics['capture_rate']:.2%} "
        f"jaccard={metrics['ticker_jaccard_avg']:.2%} "
        f"count_rms={metrics['count_divergence_rms']:.2f} "
        f"field_diff_rate={metrics['field_diff_rate']:.2%} "
        f"({metrics['n_cohort_matched_enters']} matched ENTERs)"
    )
