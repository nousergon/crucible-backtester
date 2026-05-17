"""Production-signal input producer for the portfolio-optimizer cutover gate.

ROADMAP L124 / L2222 — PR 2 ("production-signal input stream").

PR 1 (alpha-engine-backtester #214) replaced the circular legacy-scaled risk
floors with absolute thresholds and threaded a ``signal_source`` discriminator
through ``compare_to_legacy`` → ``evaluate_gate``. The runner still drove the
**synthetic** 10y predictor-GBM replay and hardcoded
``signal_source="synthetic"``.

The gate's TE / active-share bands ([2%-6%], [8%-25%]) encode an
**enhanced-index** target (decided 2026-05-16 — SPY-anchored core + active
tilt, NOT an active concentrated multi-pick book). The synthetic GBM replay
ranks the *entire* ~900-name universe with no research gating, producing a
~96.6% active-share portfolio — the wrong distribution for an enhanced-index
gate. The deployed system, by contrast, optimizes only over the production
research cohort (the ~25-34 tracked/ENTER names that ``signals/{date}/
signals.json`` selects), which lands ~15.5% active-share — in-band.

This module produces the *same output contract* as
``synthetic.predictor_backtest.run()`` —
``{status, predictions_by_date, price_matrix, spy_prices, sector_map}`` — but
sourced from the **production archive** instead of the synthetic GBM:

* cohort per date  = ``predictions(d).keys() ∪ signals(d)["universe"]`` tickers
  (mirrors ``executor.optimizer_shadow._build_universe``)
* α̂ per ticker     = ``predicted_alpha or canonical_predicted_alpha or 0.0``
  (mirrors ``executor.optimizer_shadow._build_alpha_hat`` /
  ``executor.signal_reader.read_predictions``)

The downstream ``run_optimizer_backtest`` already drives the *production*
solver (``executor.portfolio_optimizer.solve_target_weights``), so swapping
only the input producer means the gate measures the actual deployed behavior
with zero change to the optimizer kernel.

The production window is necessarily short — bounded by the
``predictor/predictions/`` archive depth (~2026-03-13 → present). That is
inherent and correct: this path measures *recent deployed behavior*, not a
10y stress test.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from synthetic.predictor_backtest import (
    _extract_close,
    build_price_matrix,
    load_sector_map,
)

logger = logging.getLogger(__name__)

_SPY = "SPY"
_CASH = "CASH"


def _resolve_predictor_path(config: dict) -> str:
    """Mirror run_gate_against_predictor_backtest's executor_path resolution
    for the predictor repo (sector_map.json lives under predictor_paths)."""
    import os

    paths = config.get("predictor_paths") or config.get("predictor_path") or []
    if isinstance(paths, str):
        paths = [paths]
    found = next((p for p in paths if os.path.isdir(p)), None)
    if not found:
        raise ValueError(
            f"predictor_paths not found on disk: {paths}. Add the "
            "alpha-engine-predictor repo root to predictor_paths in config.yaml."
        )
    return found


def _list_date_partitions(s3, bucket: str, prefix: str, suffix: str) -> list[str]:
    """List ``YYYY-MM-DD`` partitions under ``prefix``.

    ``signals/`` uses ``signals/{date}/signals.json`` (CommonPrefixes);
    ``predictor/predictions/`` uses ``predictor/predictions/{date}.json``
    (object keys). ``suffix`` distinguishes the two shapes.
    """
    paginator = s3.get_paginator("list_objects_v2")
    out: set[str] = set()
    if suffix == "/":
        # date-as-folder: collect CommonPrefixes
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                token = cp["Prefix"][len(prefix):].rstrip("/")
                if _is_iso_date(token):
                    out.add(token)
    else:
        # date-as-filename: collect Keys ending in suffix
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(suffix):
                    continue
                token = key[len(prefix):-len(suffix)]
                if _is_iso_date(token):
                    out.add(token)
    return sorted(out)


def _is_iso_date(token: str) -> bool:
    if len(token) != 10 or token[4] != "-" or token[7] != "-":
        return False
    try:
        pd.Timestamp(token)
        return True
    except ValueError:
        return False


def _load_json(s3, bucket: str, key: str) -> dict | None:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise
    except (ValueError, KeyError) as e:
        logger.warning("Malformed JSON at s3://%s/%s: %s", bucket, key, e)
        return None


def _extract_universe_tickers(universe_list: Any) -> list[str]:
    """Normalize ``signals_raw['universe']`` to ticker strings.

    Mirrors ``executor.optimizer_shadow._extract_universe_tickers`` (kept as a
    local copy rather than a cross-repo private import). Production
    signals.json emits per-ticker dicts; legacy payloads emit flat strings.
    Unknown shapes are skipped silently.
    """
    if not isinstance(universe_list, list):
        return []
    out: list[str] = []
    for el in universe_list:
        if isinstance(el, str):
            out.append(el)
        elif isinstance(el, dict):
            t = el.get("ticker")
            if isinstance(t, str) and t:
                out.append(t)
    return out


def _alpha_of(pred: dict) -> float:
    """Mirror executor.optimizer_shadow._build_alpha_hat's per-ticker rule:
    ``predicted_alpha or canonical_predicted_alpha or 0.0``, coerced to a
    finite float."""
    import math

    raw = pred.get("predicted_alpha") or pred.get("canonical_predicted_alpha") or 0.0
    try:
        a = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return a if math.isfinite(a) else 0.0


def build_production_signal_inputs(
    config: dict,
    s3_client=None,
    max_dates: int | None = None,
) -> dict:
    """Produce the predictor_backtest.run()-shaped output from the production
    archive (``signals/{date}/signals.json`` + ``predictor/predictions/
    {date}.json``).

    Args:
        config: backtester config (``signals_bucket``, ``predictor_paths``).
        s3_client: optional injected boto3 S3 client (tests).
        max_dates: if set, keep only the most-recent N production dates.

    Returns:
        ``{"status": "ok", "predictions_by_date", "price_matrix",
        "spy_prices", "sector_map", "signal_source": "production",
        "n_production_dates", "production_window"}`` on success, or
        ``{"status": "no_production_data" | "error", ...}``.
    """
    bucket = config.get("signals_bucket", "alpha-engine-research")
    s3 = s3_client or boto3.client("s3")

    signal_dates = _list_date_partitions(s3, bucket, "signals/", "/")
    pred_dates = _list_date_partitions(s3, bucket, "predictor/predictions/", ".json")
    dates = sorted(set(signal_dates) & set(pred_dates))
    if max_dates and len(dates) > max_dates:
        dates = dates[-max_dates:]
    if not dates:
        return {
            "status": "no_production_data",
            "error": (
                f"No date with BOTH signals/ and predictor/predictions/ "
                f"(signals={len(signal_dates)}, predictions={len(pred_dates)})"
            ),
        }

    predictions_by_date: dict[str, dict[str, float]] = {}
    cohort: set[str] = set()
    for d in dates:
        sig = _load_json(s3, bucket, f"signals/{d}/signals.json")
        prd = _load_json(s3, bucket, f"predictor/predictions/{d}.json")
        if sig is None or prd is None:
            continue
        preds = {
            p["ticker"]: p
            for p in (prd.get("predictions") or [])
            if isinstance(p, dict) and "ticker" in p
        }
        universe = set(_extract_universe_tickers(sig.get("universe", [])))
        names = (set(preds) | universe) - {_SPY, _CASH}
        row = {t: _alpha_of(preds.get(t, {})) for t in names}
        if row:
            predictions_by_date[d] = row
            cohort.update(row)

    if not predictions_by_date:
        return {
            "status": "no_production_data",
            "error": (
                f"{len(dates)} candidate dates but none yielded a usable "
                "(signals ∪ predictions) cohort"
            ),
        }

    predictor_path = _resolve_predictor_path(config)
    from store.arctic_reader import (  # type: ignore[import-not-found]
        load_universe_from_arctic,
    )

    price_data, _features = load_universe_from_arctic(
        bucket=bucket, tickers_allowlist=cohort,
    )
    if not price_data:
        return {
            "status": "error",
            "error": f"ArcticDB returned no price data for {len(cohort)} cohort tickers",
        }

    # Union of all per-ticker dates → trading-date axis for the price matrix.
    all_dates: set[pd.Timestamp] = set()
    for df in price_data.values():
        all_dates.update(pd.to_datetime(df.index))
    trading_dates = [d.strftime("%Y-%m-%d") for d in sorted(all_dates)]

    price_matrix = build_price_matrix(price_data, trading_dates)
    spy_prices = _extract_close(price_data, _SPY)
    if spy_prices is None or spy_prices.dropna().empty:
        return {
            "status": "error",
            "error": "SPY close series absent from ArcticDB price_data — "
            "cannot compute benchmark-relative gate metrics",
        }

    sector_map = load_sector_map(predictor_path)

    window = (min(predictions_by_date), max(predictions_by_date))
    logger.info(
        "Production-signal inputs: %d dates (%s → %s), cohort=%d tickers, "
        "price_matrix=%d×%d",
        len(predictions_by_date), window[0], window[1], len(cohort),
        len(price_matrix), len(price_matrix.columns),
    )
    return {
        "status": "ok",
        "predictions_by_date": predictions_by_date,
        "price_matrix": price_matrix,
        "spy_prices": spy_prices,
        "sector_map": sector_map,
        "signal_source": "production",
        "n_production_dates": len(predictions_by_date),
        "production_window": list(window),
    }
