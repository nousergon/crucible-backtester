"""
measurement_coverage.py — End-to-end measurement-coverage funnel.

Produces the *measurement coverage %* metric the presentation-revamp plan
(`private/alpha-engine-presentation-revamp-260503.md` §2.1, §2.5.6) calls for
on the home page + `/metrics` validation page:

> "% of signals → predictions → fills → P&L attribution complete"

config#909. Distinct from ``decision_capture_coverage`` — that metric measures
whether each *agent* produced a decision artifact (process observability). THIS
metric measures whether each actionable *signal* is traceable through the four
production stages of the trade lifecycle (outcome observability): a signal that
fires should yield a prediction, a filled trade, and an attributed P&L. A drop
at any stage is a measurement gap — the system acted (or chose not to) without
the chain being fully reconstructable.

Decision 11 forbids ad-hoc analytics in dashboard loaders; the panel is a view,
not a measurement layer. So the metric is produced here, in the backtester's
existing evaluation pass (which already loads all four sources), and emitted as
``backtest/{date}/coverage.json`` for the dashboard loader to read + render.

This module is strictly READ-ONLY and off the hot path. It reads existing
artifacts only, never writes to any source, and never participates in the live
signal/trade path. Every input is optional: a missing source degrades that
stage (and everything downstream of it) to ``None`` rather than raising — the
evaluation pipeline must never crash because a coverage diagnostic couldn't
find an input.

Stages and join key
────────────────────
The funnel is a strict-subset chain keyed on ``ticker`` within a single run
date (mirrors ``analysis/end_to_end.py``: "Every downstream table is a strict
subset of the one above it"). Each stage's population is the set of tickers
present at that stage; each ratio is downstream/upstream.

  1. signals       signals/{date}/signals.json  → tickers with signal=="ENTER"
                   (the actionable set — HOLD/EXIT/AVOID don't open a trade and
                    so aren't expected to flow downstream).
  2. predictions   predictor/predictions/{date}.json  → {ticker: alpha}
                   predicted_count = |signal tickers ∩ prediction tickers|.
  3. fills         trades.db `trades` WHERE action='ENTER' AND date=={date}
                   executed_count = |predicted tickers ∩ filled tickers|.
  4. P&L attrib.   the filled rows whose P&L is attributed (realized_return_pct
                   OR realized_alpha_pct is non-null).
                   attributed_count = |executed tickers with attributed P&L|.

Counts are nested (each ⊆ the prior), so the four counts form a monotone
funnel. ``gaps_by_stage`` records, per transition, how many tickers were lost
and (capped) which ones — the queryable "where does measurement break?" answer.

Artifact schema (config#909 acceptance)
────────────────────────────────────────
  {
    "status": "ok" | "partial" | "no_signals" | "error",
    "date": "YYYY-MM-DD",
    "signal_count":     int  | None,
    "predicted_count":  int  | None,
    "executed_count":   int  | None,
    "attributed_count": int  | None,
    "coverage_ratios": {
        "predicted_of_signal":     float|None,   # predicted/signal
        "executed_of_predicted":   float|None,   # executed/predicted
        "attributed_of_executed":  float|None,   # attributed/executed
        "attributed_of_signal":    float|None,   # end-to-end: attributed/signal
    },
    "gaps_by_stage": {
        "signal_to_prediction":    {"lost": int|None, "tickers": [...]},
        "prediction_to_execution": {"lost": int|None, "tickers": [...]},
        "execution_to_attribution":{"lost": int|None, "tickers": [...]},
    },
    "stage_availability": {signals|predictions|fills|attribution: bool},
    "notes": [ ... human-readable degradation reasons ... ],
  }

``None`` is used wherever an input was absent — never a fabricated 0, which the
dashboard would otherwise render as "0% coverage" (a false alarm) rather than
"not measured this run". ``status`` is ``partial`` whenever any stage input was
unavailable, ``ok`` when all four sources were read, ``no_signals`` when the
signals file existed but had no ENTER signals (empty denominator), ``error``
only on an unexpected fault (still returned, never raised).

Returns the standard backtester-evaluator status dict so the existing
``CompletenessTracker.run_module`` pattern handles it (mirrors
``analysis/decision_capture_coverage.py``).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "alpha-engine-research"
SIGNALS_PREFIX = "signals"
PREDICTIONS_PREFIX = "predictor/predictions"

# Cap how many ticker symbols we enumerate per gap — the artifact is a metric,
# not a full ledger. The counts are exact; the lists are a bounded sample for
# eyeballing which names dropped.
_MAX_GAP_TICKERS = 50


# ── Source readers (each returns None on absence, never raises) ───────────────


def _read_signal_tickers(
    s3: Any, *, bucket: str, date: str, notes: list[str]
) -> set[str] | None:
    """Tickers whose signal is actionable (``signal == "ENTER"``) for ``date``.

    Returns ``None`` when the signals.json is absent/unreadable (stage not
    measured). Returns an empty set when the file exists but has no ENTER
    signals (measured, denominator zero)."""
    key = f"{SIGNALS_PREFIX}/{date}/signals.json"
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        payload = json.loads(obj["Body"].read())
    except ClientError as e:
        notes.append(f"signals absent: s3://{bucket}/{key} ({_err(e)})")
        return None
    except (ValueError, KeyError) as e:
        notes.append(f"signals unreadable: s3://{bucket}/{key} ({e})")
        return None

    signals = payload.get("signals")
    if not isinstance(signals, dict):
        notes.append(f"signals malformed (no 'signals' map): s3://{bucket}/{key}")
        return None

    tickers = {
        ticker
        for ticker, s in signals.items()
        if isinstance(s, dict) and s.get("signal") == "ENTER"
    }
    return tickers


def _read_prediction_tickers(
    s3: Any, *, bucket: str, date: str, notes: list[str]
) -> set[str] | None:
    """Tickers that received a predictor output for ``date``.

    ``predictor/predictions/{date}.json`` is a ``{ticker: alpha}`` map. Returns
    ``None`` when absent/unreadable."""
    key = f"{PREDICTIONS_PREFIX}/{date}.json"
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        payload = json.loads(obj["Body"].read())
    except ClientError as e:
        notes.append(f"predictions absent: s3://{bucket}/{key} ({_err(e)})")
        return None
    except (ValueError, KeyError) as e:
        notes.append(f"predictions unreadable: s3://{bucket}/{key} ({e})")
        return None

    # Tolerate both the flat ``{ticker: alpha}`` map and a wrapped
    # ``{"predictions": {ticker: alpha}}`` envelope.
    if isinstance(payload, dict) and isinstance(payload.get("predictions"), dict):
        payload = payload["predictions"]
    if not isinstance(payload, dict):
        notes.append(f"predictions malformed (not a ticker map): s3://{bucket}/{key}")
        return None
    return set(payload.keys())


def _read_fill_and_attribution(
    trades_db_path: str | None, *, date: str, notes: list[str]
) -> tuple[set[str] | None, set[str] | None]:
    """Read filled tickers and the subset with attributed P&L for ``date``.

    Returns ``(filled, attributed)``. Either element is ``None`` when the
    trades.db / ``trades`` table is absent or unreadable. ``attributed`` is a
    subset of ``filled``: a filled ticker is attributed iff its ENTER row has a
    non-null ``realized_return_pct`` or ``realized_alpha_pct``."""
    if not trades_db_path or not Path(trades_db_path).exists():
        notes.append(f"fills absent: trades.db not found ({trades_db_path!r})")
        return None, None

    try:
        conn = sqlite3.connect(trades_db_path)
        try:
            rows = conn.execute(
                "SELECT ticker, realized_return_pct, realized_alpha_pct "
                "FROM trades WHERE action = 'ENTER' AND date = ?",
                (date,),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as e:
        notes.append(f"fills unreadable: trades.db query failed ({e})")
        return None, None

    filled: set[str] = set()
    attributed: set[str] = set()
    for ticker, ret, alpha in rows:
        if ticker is None:
            continue
        filled.add(ticker)
        if ret is not None or alpha is not None:
            attributed.add(ticker)
    return filled, attributed


# ── Aggregation helpers ──────────────────────────────────────────────────────


def _ratio(num: int | None, den: int | None) -> float | None:
    """downstream/upstream, ``None`` when either side is unmeasured or the
    denominator is zero (no upstream population to be covered)."""
    if num is None or den is None or den == 0:
        return None
    return round(num / den, 4)


def _gap(upstream: set[str] | None, downstream: set[str] | None) -> dict[str, Any]:
    """Tickers present upstream but missing downstream. ``lost`` is ``None``
    when either side is unmeasured (we can't attribute the gap)."""
    if upstream is None or downstream is None:
        return {"lost": None, "tickers": []}
    missing = sorted(upstream - downstream)
    return {"lost": len(missing), "tickers": missing[:_MAX_GAP_TICKERS]}


def _count(s: set[str] | None) -> int | None:
    return None if s is None else len(s)


# ── Public entry point ───────────────────────────────────────────────────────


def compute_measurement_coverage(
    bucket: str = DEFAULT_BUCKET,
    run_date: str | None = None,
    trades_db_path: str | None = None,
    s3_client: Any = None,
) -> dict[str, Any]:
    """Compute the signals→predictions→fills→P&L measurement-coverage funnel
    for ``run_date`` and return the ``coverage.json`` body.

    Read-only and crash-safe: any missing/unreadable input degrades the
    affected stage (and everything downstream) to ``None`` and sets
    ``status="partial"``; no input absence raises.

    Args:
        bucket: S3 bucket holding ``signals/`` and ``predictor/predictions/``.
        run_date: ISO date (YYYY-MM-DD) of the run to measure. Required — there
            is no walk-back; coverage is a per-date funnel.
        trades_db_path: local path to trades.db (the executor's fill/P&L store).
            ``None``/missing → fills + attribution stages degrade to ``None``.
        s3_client: injected boto3 client (tests). None → ``boto3.client("s3")``.

    Returns:
        The ``coverage.json`` body dict (see module docstring schema).
    """
    if not run_date:
        return {
            "status": "error",
            "date": None,
            "error": "run_date is required",
        }

    notes: list[str] = []

    try:
        s3 = s3_client or boto3.client("s3")

        signal_tickers = _read_signal_tickers(
            s3, bucket=bucket, date=run_date, notes=notes
        )
        prediction_tickers = _read_prediction_tickers(
            s3, bucket=bucket, date=run_date, notes=notes
        )
        filled_tickers, attributed_tickers = _read_fill_and_attribution(
            trades_db_path, date=run_date, notes=notes
        )

        # Enforce the strict-subset funnel: each stage's population is the
        # intersection with the prior stage's population, restricted to the
        # actionable signal set. A prediction/fill for a ticker that never
        # fired an ENTER signal isn't a measurement gap in this funnel (it's
        # out of scope), so the populations are nested.
        #
        # Strict downstream propagation: a stage whose INPUT is unmeasured
        # (``None``) — or whose UPSTREAM stage is unmeasured — is itself
        # unmeasured. You cannot trace through a missing link, so once the
        # chain breaks every later count is ``None`` (never a fabricated value
        # computed off a skipped intermediate). ``signals`` is the funnel root:
        # if it's absent, nothing is traceable.
        if signal_tickers is None:
            predicted = executed = attributed = None
        else:
            predicted = (
                None if prediction_tickers is None
                else signal_tickers & prediction_tickers
            )
            if predicted is None or filled_tickers is None:
                executed = None
            else:
                executed = predicted & filled_tickers
            if executed is None or attributed_tickers is None:
                attributed = None
            else:
                attributed = executed & attributed_tickers

        signal_count = _count(signal_tickers)
        predicted_count = _count(predicted)
        executed_count = _count(executed)
        attributed_count = _count(attributed)

        coverage_ratios = {
            "predicted_of_signal": _ratio(predicted_count, signal_count),
            "executed_of_predicted": _ratio(executed_count, predicted_count),
            "attributed_of_executed": _ratio(attributed_count, executed_count),
            "attributed_of_signal": _ratio(attributed_count, signal_count),
        }

        gaps_by_stage = {
            "signal_to_prediction": _gap(signal_tickers, predicted),
            "prediction_to_execution": _gap(predicted, executed),
            "execution_to_attribution": _gap(executed, attributed),
        }

        stage_availability = {
            "signals": signal_tickers is not None,
            "predictions": prediction_tickers is not None,
            "fills": filled_tickers is not None,
            "attribution": attributed_tickers is not None,
        }

        if signal_tickers is None:
            status = "partial"
        elif signal_count == 0:
            status = "no_signals"
            notes.append(f"signals.json for {run_date} has no ENTER signals")
        elif all(stage_availability.values()):
            status = "ok"
        else:
            status = "partial"

        return {
            "status": status,
            "date": run_date,
            "signal_count": signal_count,
            "predicted_count": predicted_count,
            "executed_count": executed_count,
            "attributed_count": attributed_count,
            "coverage_ratios": coverage_ratios,
            "gaps_by_stage": gaps_by_stage,
            "stage_availability": stage_availability,
            "notes": notes,
        }

    except Exception as e:  # never crash the pipeline on a diagnostic
        logger.exception("measurement_coverage: unexpected failure")
        return {
            "status": "error",
            "date": run_date,
            "error": str(e),
            "notes": notes,
        }


def _err(e: ClientError) -> str:
    """Compact S3 error code for notes (NoSuchKey / AccessDenied / ...)."""
    try:
        return e.response["Error"]["Code"]
    except Exception:
        return e.__class__.__name__
