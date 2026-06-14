"""
backtest.py — CLI entry point for alpha-engine-backtester (simulation only).

Runs portfolio simulation, parameter sweeps, and predictor backtests.
Evaluation logic (signal quality, diagnostics, optimizers) lives in evaluate.py.

Usage:
    # Portfolio simulation (requires executor_path in config.yaml)
    python backtest.py --mode simulate

    # Parameter sweep over risk + strategy params
    python backtest.py --mode param-sweep

    # Predictor-only backtest (10y synthetic signals, no LLM calls)
    python backtest.py --mode predictor-backtest

    # Portfolio-optimizer cutover-gate runner — replays the constrained MVO
    # optimizer over synthetic predictor history and persists a gate report
    # to s3://{bucket}/predictor/optimizer_gate/{date}.json. Used by the
    # Saturday SF to surface PR 5 cutover readiness; see ROADMAP L2222.
    python backtest.py --mode portfolio-optimizer-backtest

    # Full simulation pipeline (param-sweep + predictor-backtest + optimizer-gate)
    python backtest.py --mode all

    # Upload results to S3
    python backtest.py --mode all --upload

Options:
    --mode          simulate | param-sweep | all | predictor-backtest |
                    portfolio-optimizer-backtest
    --config        path to config.yaml (default: ./config.yaml)
    --db            path to local research.db (skips S3 pull; useful locally)
    --upload        upload results to S3
    --date          run date label for output (default: today)
    --log-level     DEBUG | INFO | WARNING (default: INFO)
"""

import argparse
import json
import logging
import tempfile
import os
import time as _time
import traceback
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# Structured logging + flow-doctor singleton via alpha-engine-lib (shared
# pattern across all 5 entrypoints; see executor/main.py for reference).
# Module-top so import-time errors in vectorbt / boto3 / arcticdb /
# executor modules below are also captured by flow-doctor's ERROR
# handler. backtest.py runs on EC2 spot via spot_backtest.sh; not in
# a Lambda image, so the simple repo-root path resolution works. The
# get_flow_doctor singleton is retrieved inside main() — fd is an
# active consumer (~7 fd.report() call sites for param-sweep /
# simulation / optimizer error escalation).
#
# exclude_patterns starts empty by deliberate convention; add patterns
# only after observing real ERROR-level noise during a backtest run.
from alpha_engine_lib.logging import setup_logging, get_flow_doctor, guard_entrypoint
_FLOW_DOCTOR_EXCLUDE_PATTERNS: list[str] = []
_FLOW_DOCTOR_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow-doctor.yaml")
setup_logging(
    "backtest",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
)

import boto3
import pandas as pd
import yaml

from analysis import param_sweep
from optimizer import executor_optimizer
from optimizer.config_archive import read_params_pit_or_current
from emailer import send_report_email
from reporter import build_report, save, upload_to_s3
from pipeline_common import (
    PhaseOutcome,
    PhaseRegistry,
    PhaseStatus,
    PhaseTimeoutError,
    init_research_db,
    load_config,
    load_phase_hard_caps,
    phase,
    resolve_trading_day,
)

logger = logging.getLogger(__name__)


# ── Simulation setup and execution ──────────────────────────────────────────


def _setup_simulation(config: dict) -> tuple:
    """
    Resolve executor path, import executor modules, load signal dates, build price matrix.

    Returns (executor_run, SimulatedIBKRClient, dates, price_matrix, init_cash, ohlcv_by_ticker).
    price_matrix is None when fewer than min_simulation_dates are available or no prices found.
    ohlcv_by_ticker: {ticker: [{date, open, high, low, close}, ...]} for strategy layer.
    """
    import sys
    import pandas as pd

    executor_paths = config.get("executor_paths", [])
    if isinstance(executor_paths, str):
        executor_paths = [executor_paths]
    executor_path = next((p for p in executor_paths if os.path.isdir(p)), None)
    if not executor_path:
        raise ValueError(
            f"None of the executor_paths exist: {executor_paths}. "
            "Add the alpha-engine repo root to executor_paths in config.yaml."
        )
    if executor_path not in sys.path:
        sys.path.insert(0, executor_path)

    from executor.main import run as executor_run
    from executor.ibkr import SimulatedIBKRClient
    from loaders import signal_loader, price_loader

    bucket = config.get("signals_bucket", "alpha-engine-research")
    min_dates = config.get("min_simulation_dates", 5)
    init_cash = float(config.get("init_cash", 1_000_000.0))

    dates = signal_loader.list_dates(bucket)
    logger.info("Simulation setup: %d signal dates available in S3", len(dates))

    # Smoke harness support: cap signal_dates to the N most recent so a
    # smoke-<phase> run completes in seconds instead of minutes. Config
    # knob only — no effect on normal runs (defaults unchanged). ROADMAP
    # Backtester P0 #3 "Per-phase smoke test harness" (2026-04-22).
    max_signal_dates = config.get("max_signal_dates")
    if max_signal_dates is not None and len(dates) > max_signal_dates:
        logger.info(
            "Simulation setup: capping signal dates to %d most recent (from %d) "
            "per config.max_signal_dates — smoke fixture active",
            max_signal_dates, len(dates),
        )
        dates = list(dates)[-int(max_signal_dates):]

    if len(dates) < min_dates:
        logger.warning(
            "Only %d signal dates available (need %d) — simulation skipped",
            len(dates), min_dates,
        )
        return executor_run, SimulatedIBKRClient, dates, None, init_cash, {}

    ohlcv_by_ticker = {}
    logger.info("Building price matrix for %d dates (ArcticDB)...", len(dates))
    # Smoke fixture universe filter: when config["smoke_tickers"] is set,
    # the ArcticDB bulk read is restricted to that allowlist so we don't
    # pay full-universe cost for a smoke run. Production runs leave the
    # key unset → allowlist stays None → reader behavior unchanged.
    smoke_tickers = config.get("smoke_tickers")
    _allowlist = set(smoke_tickers) if smoke_tickers else None
    price_matrix = price_loader.build_matrix(
        dates, bucket, _ohlcv_out=ohlcv_by_ticker,
        tickers_allowlist=_allowlist,
    )

    if price_matrix.empty:
        return executor_run, SimulatedIBKRClient, dates, None, init_cash, {}

    logger.info("OHLCV captured for %d tickers (strategy layer)", len(ohlcv_by_ticker))
    return executor_run, SimulatedIBKRClient, dates, price_matrix, init_cash, ohlcv_by_ticker


def _save_simulation_setup(
    ctx, bucket: str, date: str, sim_setup: tuple, *, s3_client=None,
) -> None:
    """Persist the reconstructable parts of `_sim_setup` to S3 and record
    their keys on the phase context. Skips persistence when the tuple
    represents the degraded 'insufficient data / empty price matrix' state
    (price_matrix is None) — nothing to reload, the retry should rerun
    setup fresh."""
    from phase_artifacts import (
        save_dataframe, save_json, save_ohlcv_by_ticker,
    )
    _, _, dates, price_matrix, _init_cash, ohlcv_by_ticker = sim_setup
    if price_matrix is None:
        return
    ctx.record_artifact(save_dataframe(
        bucket, date, "simulation_setup", "price_matrix", price_matrix,
        s3_client=s3_client,
    ))
    ctx.record_artifact(save_ohlcv_by_ticker(
        bucket, date, "simulation_setup", "ohlcv_by_ticker", ohlcv_by_ticker,
        s3_client=s3_client,
    ))
    ctx.record_artifact(save_json(
        bucket, date, "simulation_setup", "dates", list(dates),
        s3_client=s3_client,
    ))


def _load_simulation_setup(config: dict, registry) -> tuple:
    """Reconstruct the `_sim_setup` tuple from S3 artifacts.

    Executor callables can't be persisted — re-imported from
    `executor_paths` every run. init_cash re-read from config.
    """
    import sys
    from phase_artifacts import (
        load_dataframe, load_json, load_ohlcv_by_ticker,
    )

    executor_paths = config.get("executor_paths", [])
    if isinstance(executor_paths, str):
        executor_paths = [executor_paths]
    executor_path = next((p for p in executor_paths if os.path.isdir(p)), None)
    if not executor_path:
        raise ValueError(
            f"simulation_setup reload: no executor_paths exist ({executor_paths})"
        )
    if executor_path not in sys.path:
        sys.path.insert(0, executor_path)
    from executor.main import run as executor_run
    from executor.ibkr import SimulatedIBKRClient

    bucket = config.get("signals_bucket", "alpha-engine-research")
    init_cash = float(config.get("init_cash", 1_000_000.0))
    s3 = registry.s3_client

    marker = registry.load_marker("simulation_setup")
    if marker is None:
        raise RuntimeError(
            "simulation_setup auto-skip: marker missing — should not reach "
            "load path without a prior ok marker"
        )
    keys = marker.get("artifact_keys") or []

    def _find(suffix: str) -> str:
        matches = [k for k in keys if k.endswith(suffix)]
        if not matches:
            raise RuntimeError(
                f"simulation_setup reload: artifact ending {suffix!r} missing "
                f"from marker (artifact_keys={keys})"
            )
        return matches[0]

    price_matrix = load_dataframe(bucket, _find("/price_matrix.parquet"), s3_client=s3)
    ohlcv_by_ticker = load_ohlcv_by_ticker(bucket, _find("/ohlcv_by_ticker.parquet"), s3_client=s3)
    dates = load_json(bucket, _find("/dates.json"), s3_client=s3)

    logger.info(
        "simulation_setup auto-skip: reloaded price_matrix (%dx%d), "
        "%d tickers of OHLCV, %d dates from S3 artifacts",
        price_matrix.shape[0], price_matrix.shape[1],
        len(ohlcv_by_ticker), len(dates),
    )
    return (executor_run, SimulatedIBKRClient, dates, price_matrix, init_cash, ohlcv_by_ticker)


def _save_predictor_data_prep(
    ctx, bucket: str, date: str, result: dict, *, s3_client=None,
) -> None:
    """Persist every output of ``synthetic.predictor_backtest.run``.

    Skips persistence when status != 'ok' — a failed prep should re-run,
    not replay a degraded snapshot.

    Memory-shape decisions on disk + on the in-memory result dict:

    * ``features_by_ticker`` is persisted by the run() callback (Stage 4)
      before this function is called. The result dict's
      ``features_by_ticker`` slot is None at this point; we don't try
      to re-save it here.
    * ``signals_by_date`` is persisted as flat parquets via
      ``save_signals_by_date_flat`` (Stage 4b) instead of a single
      ~1 GB JSON. The 2026-04-26 v6 spot validated that the JSON path
      took 27 minutes and tripped the predictor_data_prep watchdog;
      the flat-parquet path completes in ~30-60s.
    * After the flat-parquet save completes, ``result["signals_by_date"]``
      is REPLACED IN PLACE with a ``LazySignalsByDate`` handle pointing
      at the artifact. This drops the ~1 GB dict from memory and
      mirrors the shape produced by the auto-skip reload path —
      downstream consumers (single_run sim, param_sweep) see the same
      Mapping interface regardless of which branch produced ``result``.
    """
    from phase_artifacts import (
        save_dataframe, save_json,
        save_ohlcv_by_ticker, save_series,
        save_signals_by_date_flat,
        load_signals_by_date_lazy,
    )
    if result.get("status") != "ok":
        return
    phase_name = "predictor_data_prep"

    # signals_by_date — flat parquet (records + metadata)
    records_key, metadata_key = save_signals_by_date_flat(
        bucket, date, phase_name, "signals_by_date",
        result["signals_by_date"], s3_client=s3_client,
    )
    ctx.record_artifact(records_key)
    ctx.record_artifact(metadata_key)

    # Swap the in-memory dict for a lazy handle — releases ~1 GB and
    # mirrors the auto-skip reload shape. Caller does NOT need its
    # own gc.collect() afterward; we run one here so the dict is
    # actually reclaimed before the caller touches result again.
    result["signals_by_date"] = load_signals_by_date_lazy(
        bucket, records_key, metadata_key, s3_client=s3_client,
    )
    import gc
    gc.collect()

    ctx.record_artifact(save_dataframe(
        bucket, date, phase_name, "price_matrix", result["price_matrix"],
        s3_client=s3_client,
    ))
    ctx.record_artifact(save_ohlcv_by_ticker(
        bucket, date, phase_name, "ohlcv_by_ticker", result["ohlcv_by_ticker"],
        s3_client=s3_client,
    ))
    spy = result.get("spy_prices")
    if spy is not None and len(spy) > 0:
        ctx.record_artifact(save_series(
            bucket, date, phase_name, "spy_prices", spy, s3_client=s3_client,
        ))
    ctx.record_artifact(save_json(
        bucket, date, phase_name, "metadata", result["metadata"],
        s3_client=s3_client,
    ))
    ctx.record_artifact(save_json(
        bucket, date, phase_name, "sector_map", result.get("sector_map", {}),
        s3_client=s3_client,
    ))
    ctx.record_artifact(save_json(
        bucket, date, phase_name, "trading_dates", list(result.get("trading_dates", [])),
        s3_client=s3_client,
    ))
    ctx.record_artifact(save_json(
        bucket, date, phase_name, "predictions_by_date",
        result.get("predictions_by_date", {}),
        s3_client=s3_client,
    ))
    # features_by_ticker is normally persisted by the Stage 4 callback
    # inside run() — by the time we get here ``result["features_by_
    # ticker"]`` is None and this branch is a no-op. The branch is
    # retained for callers (typically unit tests) that bypass run() and
    # populate features_by_ticker directly into the result dict. In that
    # path we save to S3 here using the same artifact key the callback
    # would use, so the lazy-load helper finds it either way.
    from phase_artifacts import save_dict_of_dataframes
    features = result.get("features_by_ticker") or {}
    if features:
        ctx.record_artifact(save_dict_of_dataframes(
            bucket, date, phase_name, "features_by_ticker", features,
            s3_client=s3_client,
        ))


def _load_features_by_ticker_only(bucket: str, registry) -> dict:
    """Lazy-load just the ``features_by_ticker`` parquet from the
    predictor_data_prep marker. Used by Phase 4a (ensemble_modes) and
    Phase 4c (feature_pruning) — the only consumers of features.

    Avoids holding the ~1.1 GB features dict in memory across the entire
    predictor pipeline (signal_gen, single_run sim, Phase 4b threshold
    eval, param sweep don't read features). Each Phase 4 caller wraps
    the load + run + ``del`` cycle so peak RSS only spikes during the
    evaluator's actual work, not the whole pipeline.

    Raises ``RuntimeError`` if the predictor_data_prep marker is missing
    or its features artifact wasn't written — Phase 4 evaluators can't
    run without features and the spot must surface that as a real
    failure, not silently skip.
    """
    from phase_artifacts import load_dict_of_dataframes
    s3 = registry.s3_client
    marker = registry.load_marker("predictor_data_prep")
    if marker is None:
        raise RuntimeError(
            "lazy features load: predictor_data_prep marker missing — "
            "Phase 4 evaluator cannot run. Re-run predictor_data_prep "
            "with --force-phases=predictor_data_prep."
        )
    keys = marker.get("artifact_keys") or []
    features_key = next(
        (k for k in keys if k.endswith("/features_by_ticker.parquet")),
        None,
    )
    if features_key is None:
        raise RuntimeError(
            "lazy features load: predictor_data_prep marker has no "
            "features_by_ticker.parquet artifact key. Phase 4 evaluator "
            "cannot run. Re-run predictor_data_prep."
        )
    return load_dict_of_dataframes(bucket, features_key, s3_client=s3)


def _load_predictor_data_prep(bucket: str, registry) -> dict:
    """Inverse of ``_save_predictor_data_prep`` — returns the result
    dict shape ``synthetic.predictor_backtest.run`` produces, EXCEPT
    ``features_by_ticker`` is NOT loaded here.

    Stage 3 of the c5.large optimization arc moved the ~1.1 GB features
    dict to lazy-load via ``_load_features_by_ticker_only``: held in
    memory only during Phase 4a + Phase 4c evaluator runs, freed after
    each. The auto-skip reload path mirrors the fresh-run path —
    neither holds features across the wider pipeline.

    Raises loud if marker missing or a required artifact absent from the
    marker's artifact_keys list."""
    from phase_artifacts import (
        load_dataframe, load_json,
        load_ohlcv_by_ticker, load_series,
        load_signals_by_date_lazy,
    )
    s3 = registry.s3_client
    marker = registry.load_marker("predictor_data_prep")
    if marker is None:
        raise RuntimeError(
            "predictor_data_prep auto-skip: marker missing — should not "
            "reach load path without a prior ok marker"
        )
    keys = marker.get("artifact_keys") or []

    def _find(suffix: str, required: bool = True) -> str | None:
        matches = [k for k in keys if k.endswith(suffix)]
        if not matches:
            if required:
                raise RuntimeError(
                    f"predictor_data_prep reload: artifact {suffix!r} missing "
                    f"from marker (artifact_keys={keys})"
                )
            return None
        return matches[0]

    # signals_by_date — Stage 4b flat-parquet shape: two artifacts
    # (records + metadata) lazy-loaded via LazySignalsByDate. Replaces
    # the pre-2026-04-26 single signals_by_date.json that took 27 min
    # to serialize and tripped the predictor_data_prep watchdog.
    signals_records_key = _find("/signals_by_date_records.parquet")
    signals_metadata_key = _find("/signals_by_date_metadata.parquet")

    result = {
        "status": "ok",
        "signals_by_date": load_signals_by_date_lazy(
            bucket, signals_records_key, signals_metadata_key, s3_client=s3,
        ),
        "price_matrix": load_dataframe(bucket, _find("/price_matrix.parquet"), s3_client=s3),
        "ohlcv_by_ticker": load_ohlcv_by_ticker(
            bucket, _find("/ohlcv_by_ticker.parquet"), s3_client=s3,
        ),
        "metadata": load_json(bucket, _find("/metadata.json"), s3_client=s3),
        "sector_map": load_json(bucket, _find("/sector_map.json"), s3_client=s3),
        "trading_dates": load_json(bucket, _find("/trading_dates.json"), s3_client=s3),
        "predictions_by_date": load_json(
            bucket, _find("/predictions_by_date.json"), s3_client=s3,
        ),
    }
    spy_key = _find("/spy_prices.parquet", required=False)
    result["spy_prices"] = load_series(bucket, spy_key, s3_client=s3) if spy_key else None
    # features_by_ticker is intentionally NOT loaded here — Phase 4a / 4c
    # evaluators lazy-load via _load_features_by_ticker_only so we don't
    # hold ~1.1 GB across the wider pipeline. Sentinel ``None`` flags
    # "feature artifact persisted; load on demand" to downstream gating.
    result["features_by_ticker"] = None

    logger.info(
        "predictor_data_prep auto-skip: reloaded signals_by_date (lazy), "
        "price_matrix %s, %d tickers OHLCV, features deferred to lazy-load",
        result["price_matrix"].shape,
        len(result["ohlcv_by_ticker"]),
    )
    return result


def _save_predictor_feature_maps(
    ctx, bucket: str, date: str,
    atr_by_ticker: dict, vwap_series_by_ticker: dict, coverage_by_ticker: dict,
    *, s3_client=None,
) -> None:
    """Persist the three feature maps produced by the bulk ArcticDB load."""
    from phase_artifacts import save_dict_of_series, save_json
    phase_name = "predictor_feature_maps_bulk_load"
    ctx.record_artifact(save_json(
        bucket, date, phase_name, "atr_by_ticker", atr_by_ticker,
        s3_client=s3_client,
    ))
    ctx.record_artifact(save_json(
        bucket, date, phase_name, "coverage_by_ticker", coverage_by_ticker,
        s3_client=s3_client,
    ))
    ctx.record_artifact(save_dict_of_series(
        bucket, date, phase_name, "vwap_series_by_ticker", vwap_series_by_ticker,
        s3_client=s3_client,
    ))


def _load_predictor_feature_maps(bucket: str, registry) -> tuple[dict, dict, dict]:
    """Inverse — returns (atr_by_ticker, vwap_series_by_ticker, coverage_by_ticker)."""
    from phase_artifacts import load_dict_of_series, load_json
    s3 = registry.s3_client
    marker = registry.load_marker("predictor_feature_maps_bulk_load")
    if marker is None:
        raise RuntimeError(
            "predictor_feature_maps_bulk_load auto-skip: marker missing"
        )
    keys = marker.get("artifact_keys") or []

    def _find(suffix: str) -> str:
        matches = [k for k in keys if k.endswith(suffix)]
        if not matches:
            raise RuntimeError(
                f"predictor_feature_maps_bulk_load reload: artifact {suffix!r} "
                f"missing (artifact_keys={keys})"
            )
        return matches[0]

    atr = load_json(bucket, _find("/atr_by_ticker.json"), s3_client=s3)
    coverage = load_json(bucket, _find("/coverage_by_ticker.json"), s3_client=s3)
    vwap = load_dict_of_series(bucket, _find("/vwap_series_by_ticker.parquet"), s3_client=s3)
    logger.info(
        "predictor_feature_maps_bulk_load auto-skip: reloaded %d tickers "
        "(atr) / %d (vwap) / %d (coverage)",
        len(atr), len(vwap), len(coverage),
    )
    return atr, vwap, coverage


_SIGNAL_LIST_FIELDS = ("universe", "buy_candidates", "enter", "exit", "reduce", "hold")


@dataclass(frozen=True)
class SignalLookup:
    """Per-date precomputed signal-derived lookups (Tier 3 Part A,
    2026-04-27).

    Built once per ``(signal_date, signals_raw, universe_symbols)``
    tuple at simulation-loop bootstrap. Shared across all 60 combos in
    a ``predictor_param_sweep`` — these dicts are cross-sectional and
    don't depend on per-combo state (sim_client, NAV, held positions).

    Attributes
    ----------
    signals_raw_filtered : dict
        ``signals_raw`` post ``_filter_signals_to_universe``. Carries
        ``universe`` and ``buy_candidates`` lists (always present) and
        optionally ``enter`` / ``exit`` / ``reduce`` / ``hold`` lists
        (live signals.json may pre-segment these; the synthetic
        envelope from ``predictions_to_signals`` does not).
    signals_by_ticker : dict[str, dict]
        ``{ticker: signal_entry}`` lookup built from the merged
        ``universe`` + ``buy_candidates`` lists. Used by
        ``evaluate_strategy_exits`` and ``decide_drawdown_forced_exits``.
    universe_sectors : dict[str, str]
        ``{ticker: sector}`` map. Passed to
        ``executor.deciders.enrich_positions(universe_sectors=...)``
        which (post-PR-#111) skips its internal rebuild when this is
        provided.
    actionable : dict
        Output of ``executor.signal_reader.get_actionable_signals``
        applied to ``signals_raw_filtered``: keys ``enter`` / ``exit`` /
        ``reduce`` / ``hold`` carry stocks segmented by their ``signal``
        field; ``market_regime`` and ``sector_ratings`` carry the
        envelope-level fields. Required for the Tier 4 vectorized
        sweep, which previously read ``signals_raw_filtered.get("enter")``
        directly and produced 0 entries on the synthetic envelope shape
        (no ``enter`` key). Computed once per date here so the 60-combo
        sweep doesn't re-run ``get_actionable_signals`` 60×.
    """
    signals_raw_filtered: dict
    signals_by_ticker: dict
    universe_sectors: dict
    actionable: dict


def _build_actionable_signals_local(signals_raw: dict) -> dict:
    """Local copy of `executor.signal_reader.get_actionable_signals`.

    Vendored here so `_build_signal_lookup` does not require the
    alpha-engine repo to be importable. The vectorized sweep depends
    on this transformation running at precompute time (Tier 4 Layer 3
    v14 incident, 2026-04-28); the unit-test surface that exercises
    `_build_signal_lookup` runs in CI where the executor repo is not
    checked out (CI installs alpha-engine-lib from PyPI but not the
    full executor repo).

    The canonical implementation lives at
    `alpha-engine/executor/signal_reader.py::get_actionable_signals`.
    Both copies must stay byte-equivalent — the parity test
    `tests/test_actionable_signals_parity.py::test_local_matches_executor`
    asserts this when the executor repo is available on sys.path
    (skip-on-import-error otherwise so CI stays green).

    Schema contract:
        Input  : dict with `universe`, `buy_candidates` lists (each
                 entry carries a `signal` field) plus envelope-level
                 `market_regime` and `sector_ratings`.
        Output : dict with `enter`/`exit`/`reduce`/`hold` lists
                 segmented by `signal`, plus `market_regime` and
                 `sector_ratings` propagated through.

    Implementation notes:
        - Defensively skips non-dict entries (the executor's version
          does not — kept here to match `_filter_signals_to_universe`'s
          tolerance for garbage input).
        - Candidates take precedence over universe in the dedup walk
          (matches `executor.signal_reader.get_actionable_signals`).
    """
    universe = signals_raw.get("universe", []) or []
    candidates = signals_raw.get("buy_candidates", []) or []
    seen: set[str] = set()
    all_stocks: list[dict] = []
    for s in list(candidates) + list(universe):
        if not isinstance(s, dict):
            continue
        ticker = s.get("ticker")
        if ticker and ticker not in seen:
            seen.add(ticker)
            all_stocks.append(s)
    return {
        "enter":  [s for s in all_stocks if s.get("signal") == "ENTER"],
        "exit":   [s for s in all_stocks if s.get("signal") == "EXIT"],
        "reduce": [s for s in all_stocks if s.get("signal") == "REDUCE"],
        "hold":   [s for s in all_stocks if s.get("signal") == "HOLD"],
        "market_regime": signals_raw.get("market_regime", "neutral"),
        "sector_ratings": signals_raw.get("sector_ratings", {}),
    }


def _build_signal_lookup(
    signals_raw: dict,
    universe_symbols: set[str] | None = None,
    rejected_counter: dict[str, int] | None = None,
) -> "SignalLookup":
    """Build a ``SignalLookup`` for one signal date.

    Single-pass construction over the merged ``universe`` +
    ``buy_candidates`` lists — the prior per-call rebuild path
    iterated those lists THREE times (filter, signals_by_ticker rebuild,
    universe_sectors rebuild). Fusing them halves walk cost per build.

    The filter call still walks separately because it touches all six
    ``_SIGNAL_LIST_FIELDS`` (enter / exit / reduce / hold / universe /
    buy_candidates), not just the two that feed the lookups.
    """
    if universe_symbols is not None:
        signals_raw = _filter_signals_to_universe(
            signals_raw, universe_symbols, rejected_counter,
        )

    signals_by_ticker: dict[str, dict] = {}
    universe_sectors: dict[str, str] = {}
    for s in (signals_raw.get("universe", []) + signals_raw.get("buy_candidates", [])):
        if not isinstance(s, dict):
            continue
        t = s.get("ticker")
        if not t:
            continue
        # signals_by_ticker uses first-write-wins (matches the prior
        # ``if t not in signals_by_ticker`` semantic)
        if t not in signals_by_ticker:
            signals_by_ticker[t] = s
        # universe_sectors uses last-write-wins (matches the prior
        # dict-comprehension semantic — duplicates resolve to the last
        # entry's sector)
        universe_sectors[t] = s.get("sector", "")

    # Run the actionable-signal transformation ONCE here so the
    # vectorized sweep can read pre-segmented enter / exit / reduce /
    # hold lists. The scalar path `_simulate_single_date` calls
    # `executor.signal_reader.get_actionable_signals(signals_raw)`
    # per-call (line 921); the vectorized engine previously bypassed
    # this and read raw envelope keys directly, which produced 0 entries
    # on the synthetic envelope shape (no `enter` key — only
    # `buy_candidates` + `universe`). Both paths now consume the same
    # actionable dict so they can never drift apart on segmentation.
    actionable = _build_actionable_signals_local(signals_raw)

    return SignalLookup(
        signals_raw_filtered=signals_raw,
        signals_by_ticker=signals_by_ticker,
        universe_sectors=universe_sectors,
        actionable=actionable,
    )


def _precompute_signal_lookups(
    signals_by_date: dict | None,
    universe_symbols: set[str] | None = None,
    rejected_counter: dict[str, int] | None = None,
) -> dict | None:
    """Precompute ``SignalLookup`` for each date in ``signals_by_date``.

    Returns ``None`` when ``signals_by_date`` is ``None`` (live
    signal-replay path that loads per-date from S3 inside
    ``_simulate_single_date``).

    Cross-combo amortization win: ``run_predictor_param_sweep`` calls
    this ONCE before defining the per-combo ``sim_fn`` closure. All 60
    combos then share the precomputed lookups instead of rebuilding
    2316 dicts × 60 combos = 139k rebuilds.
    """
    if signals_by_date is None:
        return None
    return {
        date_str: _build_signal_lookup(signals_raw, universe_symbols, rejected_counter)
        for date_str, signals_raw in signals_by_date.items()
    }


def _filter_signals_to_universe(
    signals: dict,
    universe_symbols: set[str],
    rejected_counter: dict[str, int] | None,
) -> dict:
    """Return a shallow-copied signals dict where every ticker-carrying list
    is filtered to entries whose ``ticker`` is in ``universe_symbols``.

    Rationale: simulate mode replays historical signals.json files from S3.
    Past constituent turnover (e.g. TSM/ASML dropped 2026-04-20) leaves
    historical signals referencing tickers no longer in ArcticDB. Executor
    hard-fail guards (load_daily_vwap, load_atr_14_pct) then abort the
    simulation. This filter drops those tickers at the simulate boundary —
    NOT at the executor layer, because live executor must preserve EXIT/
    REDUCE/HOLD for real held positions even if the ticker somehow went
    missing from ArcticDB (different concern: alarm, don't silently skip).
    In simulate mode there are no real held positions so the drop is safe.

    ``rejected_counter`` (if provided) accumulates per-ticker reject counts
    across the simulation loop for a single aggregate WARN log at end of run.
    Consistent with feedback_no_silent_fails: rejects are counted and
    reported, not silently dropped.
    """
    filtered = dict(signals)
    for field in _SIGNAL_LIST_FIELDS:
        entries = signals.get(field)
        if not isinstance(entries, list):
            continue
        kept = []
        for e in entries:
            ticker = (e.get("ticker") if isinstance(e, dict) else None) or ""
            ticker = ticker.upper()
            if ticker and ticker in universe_symbols:
                kept.append(e)
            elif ticker and rejected_counter is not None:
                rejected_counter[ticker] = rejected_counter.get(ticker, 0) + 1
        filtered[field] = kept
    return filtered


# Sector ETFs + macro tickers the executor's strategy layer always
# queries via ``price_histories[ticker]`` (executor/strategies/
# exit_manager.py:25 SECTOR_ETF_MAP + main.py:1265 SPY). Always
# included in the per-date slice set regardless of signals/holdings.
_SECTOR_ETF_TICKERS: frozenset[str] = frozenset({
    "SPY", "XLK", "XLV", "XLF", "XLY", "XLP", "XLE", "XLU",
    "XLRE", "XLB", "XLI", "XLC",
})


def _build_filtered_price_histories(
    *,
    ohlcv_by_ticker: dict,
    signal_date: str,
    signals_raw: dict,
    held_tickers: set[str],
) -> dict[str, "pd.DataFrame"]:
    """Slice ``price_histories`` only for tickers the executor will
    query at this signal date.

    The executor accesses ``price_histories`` via ``.get(ticker)``
    exclusively — verified by grep across
    ``alpha-engine/executor/main.py``, ``risk_guard.py``, and
    ``strategies/exit_manager.py``. There is no path that iterates
    keys/values/items, so building entries for tickers the executor
    won't touch is pure waste.

    Queried set per call:
      * ``held_tickers`` — current sim positions (exit_manager,
        risk_guard.held_history)
      * ``signals_raw['buy_candidates']`` tickers — ENTER candidates
        (risk_guard.candidate_history, momentum_gate)
      * ``signals_raw['universe']`` tickers — covers EXIT/REDUCE/HOLD
        for any held entries (defense-in-depth; held_tickers usually
        already covers these)
      * ``_SECTOR_ETF_TICKERS`` — always; sector_relative_veto and
        SPY-relative metrics reference them every call

    Output shape (2026-04-27): ``dict[str, pd.DataFrame]`` to match
    the executor's vectorized per-bar access (alpha-engine PR #108).
    The slice itself is ``df.loc[:signal_date]`` — pandas binary search
    on the DatetimeIndex, no per-row materialization. Each slice is a
    cheap view into the underlying ``ohlcv_by_ticker`` DataFrame; no
    copy until a downstream consumer demands one.
    """
    queried: set[str] = set(_SECTOR_ETF_TICKERS)
    queried.update(held_tickers)
    for field in ("buy_candidates", "universe"):
        for entry in (signals_raw.get(field) or []):
            if isinstance(entry, dict):
                ticker = entry.get("ticker")
                if ticker:
                    queried.add(ticker)

    ts = pd.Timestamp(signal_date)
    out: dict[str, "pd.DataFrame"] = {}
    for ticker in queried:
        df = ohlcv_by_ticker.get(ticker)
        if df is None or df.empty:
            continue
        sliced = df.loc[:ts]
        if sliced.empty:
            continue
        out[ticker] = sliced
    return out


def _build_merged_simulate_config(config: dict) -> tuple[dict, dict]:
    """Build the merged config + flat strategy_config for one simulate
    pass (one param-sweep combo).

    Replaces the per-call config-merge that ``executor.run()`` did under
    ``config_override=`` for every simulate date. Backtester runs this
    ONCE per ``_run_simulation_loop`` call so the 100k+ per-date deciders
    invocations don't repay deepcopy + ``_PARAM_MAP`` traversal each time.

    The merge logic mirrors ``executor.main.run()``'s ``config_override``
    branch byte-for-byte:
      * ``key == "strategy"`` and dict-of-dicts: nested ``.update`` into
        ``config["strategy"][sub_key]``
      * ``key in _PARAM_MAP``: traverse the canonical nested path and
        write the leaf
      * else: top-level assignment

    Imports from ``executor`` are deferred so this function works in
    isolated unit tests (which mock ``_setup_simulation`` and never
    install the executor on sys.path). When the import fails the
    fallback uses an empty ``_PARAM_MAP`` + a no-op
    ``load_strategy_config`` — this is sufficient for test stubs that
    mock ``_simulate_single_date``. Production paths (``run_simulate``,
    ``run_param_sweep``, ``replay_for_dates``) call ``_setup_simulation``
    upfront which puts the executor on sys.path before this runs, so
    the real merge runs there.

    Runtime-object protection: the ``config`` dict carries non-data
    runtime handles under sentinel keys (``_phase_registry`` holds a
    PhaseRegistry whose ``.s3_client`` is a botocore S3Client with
    circular service-model references). ``copy.deepcopy`` blows the
    recursion stack on those (caught 2026-04-27 spot smoke v2 — the
    botocore service_model traversal). Strip these keys before deepcopy
    and re-attach the original handle on the merged result so consumers
    still see the runtime objects without the merge attempting to copy them.
    """
    import copy

    try:
        from executor.main import _PARAM_MAP  # type: ignore[import-not-found]
        from executor.strategies.config import (  # type: ignore[import-not-found]
            load_strategy_config,
        )
    except ImportError:
        _PARAM_MAP = {}

        def load_strategy_config(_cfg):
            return {}

    _RUNTIME_HANDLE_KEYS = ("_phase_registry",)
    runtime_handles = {k: config[k] for k in _RUNTIME_HANDLE_KEYS if k in config}
    config_for_copy = {k: v for k, v in config.items() if k not in _RUNTIME_HANDLE_KEYS}

    merged: dict = copy.deepcopy(config_for_copy)
    # Re-attach runtime handles post-deepcopy. Downstream callers
    # (e.g. _run_simulation_pipeline reads merged["_phase_registry"])
    # still see the same object the live shell registered.
    for k, v in runtime_handles.items():
        merged[k] = v

    override = _build_config_override(config)
    if override:
        for key, val in override.items():
            if key == "strategy" and isinstance(val, dict) and "strategy" in merged:
                for sub_key, sub_val in val.items():
                    if isinstance(sub_val, dict) and isinstance(merged["strategy"].get(sub_key), dict):
                        merged["strategy"][sub_key].update(sub_val)
                    else:
                        merged["strategy"][sub_key] = sub_val
            elif key in _PARAM_MAP:
                path = _PARAM_MAP[key]
                target = merged
                for p in path[:-1]:
                    target = target.setdefault(p, {})
                target[path[-1]] = val
            else:
                merged[key] = val

    strategy_config = load_strategy_config(merged)
    return merged, strategy_config


def _simulate_single_date(
    sim_client,
    signal_date: str,
    price_matrix,
    ohlcv_by_ticker: dict[str, pd.DataFrame] | None,
    bucket: str,
    merged_config: dict,
    strategy_config: dict,
    signals_override: dict | None = None,
    signal_lookup: "SignalLookup | None" = None,
    universe_symbols: set[str] | None = None,
    rejected_ticker_counter: dict[str, int] | None = None,
    atr_by_ticker: dict[str, float] | None = None,
    vwap_series_by_ticker: dict[str, pd.Series] | None = None,
    coverage_by_ticker: dict[str, float] | None = None,
    feature_lookup=None,
) -> tuple[list[dict] | None, str | None]:
    """Run one simulate date through the deciders directly (Tier 2).

    Replaces the prior ``executor_run(simulate=True, ...)`` shell call —
    backtester now invokes ``executor.deciders.decide_entries`` and
    ``decide_exits_and_reduces`` directly with already-loaded state,
    skipping the live shell entirely (~150 ms of per-call overhead from
    ``load_config`` + ``_read_signals`` defense-in-depth checks +
    ``signals_by_ticker`` rebuild + ``universe_sectors`` dict-comp +
    ``OrderBook.load`` etc.).

    Returns ``(orders_or_none, skip_reason)``. On successful run returns
    ``(orders_list, None)`` — may be an empty list. On skip returns
    ``(None, reason_key)`` where reason_key ∈ {no_price_index,
    empty_prices, no_signals}.

    Side effects:
      * ``sim_client._prices`` and ``_simulation_date`` set per call so
        ``sim_client.get_current_price`` returns this date's prices.
      * ``sim_client.place_market_order`` invoked per accepted order so
        position state carries forward across dates.
    """
    ts = pd.Timestamp(signal_date)
    if ts not in price_matrix.index:
        later = price_matrix.index[price_matrix.index > ts]
        if len(later) > 0:
            ts = later[0]
            logger.debug(
                "Signal date %s not in price index — using next trading day %s",
                signal_date, ts.date(),
            )
        else:
            return None, "no_price_index"

    date_prices = price_matrix.loc[ts].dropna().to_dict()
    if not date_prices:
        return None, "empty_prices"

    # Tier 3 Part A (2026-04-27): when ``signal_lookup`` is provided,
    # the caller has already filtered + built signals_by_ticker +
    # universe_sectors at simulation-loop bootstrap. Skip the per-call
    # rebuild path entirely. ``signals_override`` falls back to the
    # legacy path (for tests + replay_for_dates that pre-date the
    # lookup amortization).
    if signal_lookup is not None:
        signals_raw = signal_lookup.signals_raw_filtered
    elif signals_override is not None:
        signals_raw = signals_override
    else:
        from loaders import signal_loader
        try:
            signals_raw = signal_loader.load(bucket, signal_date)
        except FileNotFoundError:
            return None, "no_signals"

    # Deferred executor imports — kept after the early-return guards so
    # tests that bypass _setup_simulation (and don't put executor on
    # sys.path) can still hit the no_price_index / empty_prices /
    # no_signals paths without ImportError. Production callers go
    # through _setup_simulation which sets sys.path before this fires.
    from executor.deciders import (
        compute_signal_age_days,
        decide_drawdown_forced_exits,
        decide_drawdown_response,
        decide_entries,
        decide_exits_and_reduces,
        enrich_positions,
    )
    from executor.signal_reader import get_actionable_signals
    from executor.strategies.exit_manager import SECTOR_ETF_MAP, evaluate_exits

    # Pre-filter signals against the simulation-bootstrap universe set
    # (loaded once at simulation-loop startup). Skipped when
    # ``signal_lookup`` was provided — the lookup already carries the
    # filtered signals. Live executor's filter_buy_candidates_to_universe
    # is gated `if not simulate` post-PR #109 specifically so we don't
    # pay the per-call ArcticDB.list_symbols round-trip here.
    if signal_lookup is None and universe_symbols is not None:
        signals_raw = _filter_signals_to_universe(
            signals_raw, universe_symbols, rejected_ticker_counter,
        )

    sim_client._prices = date_prices
    sim_client._simulation_date = signal_date

    price_histories = None
    if ohlcv_by_ticker:
        price_histories = _build_filtered_price_histories(
            ohlcv_by_ticker=ohlcv_by_ticker,
            signal_date=signal_date,
            signals_raw=signals_raw,
            held_tickers=set(getattr(sim_client, "_positions", {}).keys()),
        )

    atr_map: dict = atr_by_ticker if atr_by_ticker is not None else {}
    coverage_map: dict = coverage_by_ticker if coverage_by_ticker is not None else {}
    vwap_map: dict = {}
    if vwap_series_by_ticker is not None:
        from store.feature_maps import resolve_vwap_map_for_date
        enter_tickers = [
            s["ticker"]
            for s in (signals_raw.get("enter") or [])
            if s.get("ticker")
        ]
        vwap_map = resolve_vwap_map_for_date(
            vwap_series_by_ticker, enter_tickers, signal_date,
        )

    # ── Deciders ────────────────────────────────────────────────────
    # All side-effecting layer (place_market_order, ob.add_entry, S3,
    # logger inside live shell) is owned by THIS function for sim mode;
    # the deciders themselves are pure.

    # Resolve actionable signals (enter/exit/reduce/hold) + market regime.
    signals = get_actionable_signals(signals_raw)
    market_regime = signals["market_regime"]
    sector_ratings = signals["sector_ratings"]
    enter_signals = signals["enter"]

    # Tier 3 Part A: prefer the precomputed lookup when caller provided
    # one. Falls back to per-call rebuild if signal_lookup is None
    # (run_simulate single-pass + replay_for_dates paths that haven't
    # been migrated to the precompute pattern yet, plus tests that
    # construct fixtures inline).
    if signal_lookup is not None:
        signals_by_ticker = signal_lookup.signals_by_ticker
        universe_sectors_for_enrich = signal_lookup.universe_sectors
    else:
        signals_by_ticker = {}
        universe_sectors_for_enrich = None  # let enrich_positions rebuild from signals_raw
        for s in (signals_raw.get("universe", []) + signals_raw.get("buy_candidates", [])):
            t = s.get("ticker")
            if t and t not in signals_by_ticker:
                signals_by_ticker[t] = s

    sector_etf_histories: dict | None = None
    if price_histories:
        sector_etf_histories = {
            t: price_histories[t]
            for t in SECTOR_ETF_MAP.values()
            if t in price_histories
        }
        if "SPY" in price_histories:
            sector_etf_histories["SPY"] = price_histories["SPY"]

    portfolio_nav = sim_client.get_portfolio_nav()
    peak_nav = sim_client.get_peak_nav(None)
    raw_positions = sim_client.get_positions()
    # Pull entry_date from sim positions (set by SimulatedIBKRClient.place_market_order).
    entry_dates = {
        t: pos.get("entry_date") for t, pos in raw_positions.items()
    }
    current_positions = enrich_positions(
        raw_positions, signals_raw, entry_dates,
        universe_sectors=universe_sectors_for_enrich,
    )

    dd_multiplier, dd_reason = decide_drawdown_response(
        portfolio_nav, peak_nav, merged_config,
    )
    if dd_multiplier < 1.0:
        logger.info("Drawdown tier active: %s", dd_reason)

    # Strategy-layer exit signals (ATR trailing, profit-take, momentum,
    # time decay, sector-relative veto). evaluate_exits already accepts
    # any ibkr_client with .get_current_price(); SimulatedIBKRClient's
    # get_current_price is a dict lookup against ._prices set above.
    strategy_exits = evaluate_exits(
        current_positions=current_positions,
        signals_by_ticker=signals_by_ticker,
        run_date=signal_date,
        price_histories=price_histories or {},
        ibkr_client=sim_client,
        strategy_config=strategy_config,
        sector_etf_histories=sector_etf_histories,
        feature_lookup=feature_lookup,
    )

    # Forced exits when drawdown is severe (tiers 2/3 of graduated dd).
    strategy_exits.extend(
        decide_drawdown_forced_exits(
            current_positions=current_positions,
            exit_signals=signals.get("exit", []),
            strategy_exits=strategy_exits,
            signals_by_ticker=signals_by_ticker,
            dd_multiplier=dd_multiplier,
            strategy_config=strategy_config,
        )
    )

    signal_age_days = compute_signal_age_days(signals_raw, signal_date)

    # Entry pipeline.
    entry_plan = decide_entries(
        enter_signals=enter_signals,
        signals_raw=signals_raw,
        predictions_by_ticker={},  # backtester intentionally empty (no GBM veto in sim)
        config=merged_config,
        strategy_config=strategy_config,
        market_regime=market_regime,
        sector_ratings=sector_ratings,
        portfolio_nav=portfolio_nav,
        peak_nav=peak_nav,
        current_positions=current_positions,
        prices_now=date_prices,
        price_histories=price_histories,
        atr_map=atr_map,
        vwap_map=vwap_map,
        coverage_map=coverage_map,
        dd_multiplier=dd_multiplier,
        signal_age_days=signal_age_days,
        earnings_by_ticker={},  # backtester intentionally empty
        run_date=signal_date,
    )

    # Apply ENTER orders to sim_client so position state carries forward
    # to the next simulate date (already-held check on next iteration).
    for o in entry_plan.orders:
        if o["action"] == "ENTER":
            sim_client.place_market_order(o["ticker"], "BUY", o["shares"])

    # Exit + reduce pipeline.
    exit_plan = decide_exits_and_reduces(
        signals=signals,
        strategy_exits=strategy_exits,
        current_positions=current_positions,
        prices_now=date_prices,
        predictions_by_ticker={},
        config=merged_config,
        market_regime=market_regime,
        portfolio_nav=portfolio_nav,
        run_date=signal_date,
    )

    # Apply EXIT/REDUCE orders to sim_client.
    for o in exit_plan.orders:
        if o["action"] in ("EXIT", "REDUCE"):
            sim_client.place_market_order(o["ticker"], "SELL", o["shares"])

    orders = list(entry_plan.orders) + list(exit_plan.orders)
    # Tag each order with the simulation date for downstream parity diffing.
    for order in orders:
        order.setdefault("date", signal_date)
    return orders, None


def _try_construct_ew_high_vol_basket(price_matrix) -> pd.Series | None:
    """Build the EW-high-vol basket from ``price_matrix``; return ``None``
    on failure or insufficient history.

    L2170 Workstream D: per-config risk-matched skill measurement. The
    basket holds the top vol-quartile of the agent's decision universe
    (here: all columns of the universe-wide price matrix the simulator
    already loads), equal-weighted, rebalanced weekly. Pure-compute, no
    new data dependency — `construct_ew_high_vol_benchmark` operates on
    the same price matrix the sweep already has in memory.

    Returns ``None`` for short windows (< 60-day vol-lookback), degenerate
    price matrices, or any construction failure — caller's `portfolio_stats`
    falls back to the SPY-only alpha computation. Best-effort by design;
    the EW-high-vol stat is additive observability, not load-bearing.
    """
    try:
        from analysis.risk_matched_benchmark import construct_ew_high_vol_benchmark
        basket = construct_ew_high_vol_benchmark(price_matrix)
        if basket is None or basket.empty:
            return None
        return basket
    except Exception:
        logger.warning(
            "Could not construct EW-high-vol basket from price matrix — "
            "falling back to SPY-only alpha computation.",
            exc_info=True,
        )
        return None


def _run_simulation_loop(
    executor_run,
    SimulatedIBKRClient,
    dates: list[str],
    price_matrix,
    config: dict,
    ohlcv_by_ticker: dict[str, pd.DataFrame] | None = None,
    signals_by_date: dict | None = None,
    spy_prices: pd.Series | None = None,
    ew_high_vol_basket_returns: pd.Series | None = None,
    atr_by_ticker: dict[str, float] | None = None,
    vwap_series_by_ticker: dict[str, pd.Series] | None = None,
    coverage_by_ticker: dict[str, float] | None = None,
    signal_lookups: dict | None = None,
    feature_lookup=None,
    resilience_ctx: dict | None = None,
) -> dict:
    """
    Run one full simulation pass with the given config and pre-built price matrix.

    resilience_ctx (L4471 L1+L2): when non-None (passed ONLY by the standalone
        ``run_simulate`` simulate phase — never by param-sweep per-combo calls),
        enables per-date timing/fast-fail instrumentation (L1) and within-run
        checkpoint/resume (L2). Keys: ``{enabled, bucket, run_date, fingerprint,
        s3_client, checkpoint_every, per_date_warn_s, budget_warn_s,
        warmup_dates}``. None (param sweeps) preserves the legacy 250-date
        heartbeat with no checkpoint I/O.

    A fresh SimulatedIBKRClient is created per call so param-sweep combinations
    start from the same initial state. Prices are swapped per date; positions
    and NAV carry forward across dates within a single run.

    ohlcv_by_ticker: full OHLCV histories for strategy layer (ATR trailing stops).
        Filtered to <= signal_date before each executor call to prevent lookahead.
    signals_by_date: optional pre-built signals for each date (predictor-only mode).
        When provided, uses these instead of loading from S3 via signal_loader.
    signal_lookups: optional ``{date_str: SignalLookup}`` precomputed at a
        higher scope (e.g. ``run_predictor_param_sweep``). Each
        ``SignalLookup`` carries the filtered signals_raw, signals_by_ticker,
        and universe_sectors derived from a single (signals_raw,
        universe_symbols) tuple. Cross-combo amortization: 60 combos in
        a param sweep all reference the same precomputed lookups
        instead of rebuilding 139k dicts. Tier 3 Part A (2026-04-27).
    """
    from vectorbt_bridge import orders_to_portfolio
    from vectorbt_bridge import portfolio_stats as compute_portfolio_stats

    init_cash = float(config.get("init_cash", 1_000_000.0))
    bucket = config.get("signals_bucket", "alpha-engine-research")

    # Staleness circuit breaker: halt if price data is too old for reliable simulation
    if getattr(price_matrix, "attrs", {}).get("stale_circuit_break"):
        return {
            "status": "stale_prices",
            "staleness_warning": price_matrix.attrs.get("staleness_warning"),
            "note": "Price data too stale for reliable simulation",
        }

    # Tier 2 (2026-04-27) — build merged_config + strategy_config ONCE per
    # simulation loop instead of per simulate date. The deciders accept
    # already-merged config; live executor.run() does the same merge per
    # call, but for the backtester's 100k+ calls per param sweep this is
    # the single biggest per-call overhead remaining (each merge is ~1 ms
    # under deepcopy + _PARAM_MAP traversal).
    merged_config, strategy_config = _build_merged_simulate_config(config)

    # Precompute ATR + VWAP maps once per simulate pass. The executor's
    # ``load_atr_14_pct`` and ``load_daily_vwap`` both hit ArcticDB per
    # ticker per call (20+ round-trips per simulate call). The
    # alpha-engine PR #91 kwargs ``atr_map`` + ``vwap_map`` let the
    # backtester inject pre-resolved maps and skip those reads entirely.
    # Callers can also pass in the pre-built maps to avoid rebuilding per
    # combo in param sweep.
    if (atr_by_ticker is None or vwap_series_by_ticker is None or coverage_by_ticker is None):
        from store.feature_maps import load_precomputed_feature_maps
        _smoke_tickers = config.get("smoke_tickers")
        _allowlist = set(_smoke_tickers) if _smoke_tickers else None
        _atr, _vwap, _cov = load_precomputed_feature_maps(bucket, tickers_allowlist=_allowlist)
        if atr_by_ticker is None:
            atr_by_ticker = _atr
        if vwap_series_by_ticker is None:
            vwap_series_by_ticker = _vwap
        if coverage_by_ticker is None:
            coverage_by_ticker = _cov

    sim_client = SimulatedIBKRClient(prices={}, nav=init_cash)
    all_orders: list[dict] = []
    dates_simulated = 0
    skip_reasons = {"no_price_index": 0, "empty_prices": 0, "no_signals": 0}

    # Load today's ArcticDB universe once — used to filter historical signals
    # that reference since-dropped tickers (e.g. TSM/ASML post-2026-04-20).
    # Hard-fail on ArcticDB library-open error: that's a pipeline precondition,
    # not a simulate-mode edge case to paper over.
    universe_symbols: set[str] | None = None
    rejected_ticker_counter: dict[str, int] = {}
    try:
        from alpha_engine_lib.arcticdb import get_universe_symbols
        universe_symbols = get_universe_symbols(bucket)
    except Exception as exc:
        # Fail loud: simulate would otherwise crash later at load_daily_vwap
        # when a historical signal references a dropped ticker, and the
        # failure surface would be a misleading "daemon cannot plan triggers"
        # instead of a clear "ArcticDB library unreachable."
        raise RuntimeError(
            f"Simulate universe-filter bootstrap failed: could not read "
            f"ArcticDB universe symbols from bucket {bucket!r}: {exc}"
        ) from exc

    # Use signals_by_date keys as iteration dates when available
    if signals_by_date is not None:
        sim_dates = sorted(signals_by_date.keys())
    else:
        sim_dates = dates

    # Tier 3 Part A: if caller didn't precompute signal_lookups (i.e.
    # we're called from run_simulate, NOT from run_predictor_param_sweep),
    # build them ONCE here so the per-date loop within this combo gets
    # the same amortization. The win is smaller for run_simulate (single
    # combo) but consistent — and run_predictor_param_sweep amortizes
    # across all 60 combos by precomputing at its scope.
    if signal_lookups is None and signals_by_date is not None:
        signal_lookups = _precompute_signal_lookups(
            signals_by_date, universe_symbols, rejected_ticker_counter,
        )

    # Tier 3 Part C: precompute FeatureLookup ONCE (vectorized ATR /
    # RSI / momentum / returns / support across all tickers × all
    # dates). Same in-loop / cross-combo amortization story as
    # signal_lookups: built here for run_simulate, hoisted to
    # run_predictor_param_sweep scope to share across 60 combos.
    if feature_lookup is None and ohlcv_by_ticker:
        try:
            from executor.feature_lookup import FeatureLookup
            feature_lookup = FeatureLookup.from_ohlcv_by_ticker(ohlcv_by_ticker)
            logger.info(
                "FeatureLookup built: %d tickers (atr_dollar=%d, rsi=%d)",
                len(ohlcv_by_ticker),
                len(feature_lookup.atr_dollar),
                len(feature_lookup.rsi),
            )
        except ImportError:
            # Test environment without executor on sys.path. Fall
            # through to per-call recompute (legacy behavior).
            feature_lookup = None

    # Per-date heartbeat — emit an INFO line every N dates so a long sim
    # can't go fully silent for more than a minute or two at a time. Before
    # this, ~2000 signal dates iterated with zero log output; combined with
    # a DEBUG-only per-combo log in param_sweep, a predictor-param-sweep
    # could run for >100 min without a single INFO line. See ROADMAP
    # P0 "Diagnose the silent-phase bottleneck" (2026-04-22 4th dry-run).
    _HEARTBEAT_EVERY = 250
    n_dates = len(sim_dates)

    # ── L4471 L2: within-run resume ─────────────────────────────────────────
    # Active only under resilience_ctx (standalone simulate phase). If a valid
    # checkpoint exists for this (run_date, fingerprint), restore the
    # carried-forward state and resume from the next date — a failed run
    # doesn't re-pay the dates it already simulated.
    _rc = resilience_ctx if (resilience_ctx and resilience_ctx.get("enabled")) else None
    start_idx = 0
    if _rc is not None:
        from store.sim_checkpoint import load_checkpoint, save_checkpoint
        _ckpt = load_checkpoint(
            bucket=_rc["bucket"], run_date=_rc["run_date"],
            fingerprint=_rc["fingerprint"], s3_client=_rc["s3_client"],
        )
        if _ckpt is not None:
            _st = _ckpt["sim_state"]
            sim_client._cash = _st["cash"]
            sim_client._positions = _st["positions"]
            sim_client._peak_nav = _st["peak_nav"]
            all_orders = list(_ckpt["all_orders"])
            dates_simulated = _ckpt["dates_simulated"]
            skip_reasons.update(_ckpt["skip_reasons"])
            rejected_ticker_counter.update(_ckpt["rejected_ticker_counter"])
            start_idx = _ckpt["idx"] + 1
            logger.info(
                "[sim] RESUMING from checkpoint: %d/%d dates already done (next=%s)",
                start_idx, n_dates,
                sim_dates[start_idx] if start_idx < n_dates else "(complete)",
            )

    t0 = _time.monotonic()
    _budget_warned = False

    for idx in range(start_idx, n_dates):
        signal_date = sim_dates[idx]
        signals_override = signals_by_date[signal_date] if signals_by_date is not None else None
        signal_lookup = signal_lookups.get(signal_date) if signal_lookups is not None else None
        _date_t0 = _time.monotonic()
        orders, skip = _simulate_single_date(
            sim_client=sim_client,
            signal_date=signal_date,
            price_matrix=price_matrix,
            ohlcv_by_ticker=ohlcv_by_ticker,
            bucket=bucket,
            merged_config=merged_config,
            strategy_config=strategy_config,
            signals_override=signals_override,
            signal_lookup=signal_lookup,
            universe_symbols=universe_symbols,
            rejected_ticker_counter=rejected_ticker_counter,
            atr_by_ticker=atr_by_ticker,
            vwap_series_by_ticker=vwap_series_by_ticker,
            feature_lookup=feature_lookup,
            coverage_by_ticker=coverage_by_ticker,
        )
        _date_dt = _time.monotonic() - _date_t0
        if skip is not None:
            skip_reasons[skip] += 1
        else:
            if orders:
                all_orders.extend(orders)
            dates_simulated += 1

        if _rc is not None:
            # ── L1: per-date instrumentation (makes the simulate cost visible;
            # the prior heartbeat only fired every 250 dates so a <250-date sim
            # logged nothing until the end — exactly why the 2026-05-30 overrun
            # was opaque). Plus a slow-date WARN + projected-overrun WARN. ──
            cum = _time.monotonic() - t0
            logger.info(
                "[sim] date %d/%d %s done in %.1fs (cum %.0fs)",
                idx + 1, n_dates, signal_date, _date_dt, cum,
            )
            if _date_dt > _rc.get("per_date_warn_s", 60):
                logger.warning(
                    "[sim] SLOW DATE %s took %.1fs (> %.0fs) — candidate root cause "
                    "of the simulate-phase overrun (L4470/L4471)",
                    signal_date, _date_dt, _rc.get("per_date_warn_s", 60),
                )
            _done = idx - start_idx + 1
            _budget = _rc.get("budget_warn_s")
            if _budget and _done >= _rc.get("warmup_dates", 5) and not _budget_warned:
                _projected = (cum / _done) * (n_dates - start_idx)
                if _projected > _budget:
                    _budget_warned = True
                    logger.warning(
                        "[sim] PROJECTED OVERRUN: ~%.0fs for the %d-date sim "
                        "(rate %.1fs/date) exceeds budget %.0fs — O(t)-slow or "
                        "stalling; L2 checkpoint lets a re-run resume (L4471)",
                        _projected, n_dates - start_idx, cum / _done, _budget,
                    )
            # ── L2: checkpoint every N dates (best-effort; never aborts) ──
            if (idx + 1) % _rc.get("checkpoint_every", 5) == 0:
                save_checkpoint(
                    bucket=_rc["bucket"], run_date=_rc["run_date"],
                    fingerprint=_rc["fingerprint"], idx=idx, last_date=signal_date,
                    sim_state={
                        "cash": sim_client._cash,
                        "positions": sim_client._positions,
                        "peak_nav": sim_client._peak_nav,
                    },
                    all_orders=all_orders, dates_simulated=dates_simulated,
                    skip_reasons=skip_reasons,
                    rejected_ticker_counter=rejected_ticker_counter,
                    s3_client=_rc["s3_client"],
                )
        elif (idx + 1) % _HEARTBEAT_EVERY == 0 or (idx + 1) == n_dates:
            elapsed = _time.monotonic() - t0
            logger.info(
                "Simulation loop: %d/%d dates processed (%.1fs elapsed, last=%s)",
                idx + 1, n_dates, elapsed, signal_date,
            )

    # L2: the sim completed all dates — clear the checkpoint so a fresh run for
    # this date recomputes (within-run resume is for FAILED runs only; cross-run
    # incremental resume is L3/deferred).
    if _rc is not None:
        from store.sim_checkpoint import clear_checkpoint
        clear_checkpoint(bucket=_rc["bucket"], run_date=_rc["run_date"], s3_client=_rc["s3_client"])

    _MIN_SIMULATION_COVERAGE = 0.80

    dates_expected = len(sim_dates)
    coverage = dates_simulated / dates_expected if dates_expected > 0 else 0
    skipped = {k: v for k, v in skip_reasons.items() if v > 0}
    logger.info(
        "Simulation: %d/%d dates (%.0f%% coverage), %d orders%s",
        dates_simulated, dates_expected, coverage * 100, len(all_orders),
        f" — skipped: {skipped}" if skipped else "",
    )

    if rejected_ticker_counter:
        # Aggregate reject log — loud so data drift (tickers dropped from
        # the universe between signal-write time and replay time) is visible.
        top = sorted(rejected_ticker_counter.items(), key=lambda kv: -kv[1])
        total_rejects = sum(rejected_ticker_counter.values())
        logger.warning(
            "Simulate universe-filter dropped %d signal entries across %d "
            "tickers (tickers present in historical signals but absent from "
            "current ArcticDB universe). Top offenders: %s",
            total_rejects, len(rejected_ticker_counter),
            [f"{t}={n}" for t, n in top[:10]],
        )

    if dates_expected > 0 and coverage < _MIN_SIMULATION_COVERAGE:
        return {
            "status": "insufficient_coverage",
            "dates_simulated": dates_simulated,
            "dates_expected": dates_expected,
            "coverage": round(coverage, 3),
            "skip_reasons": skipped,
            "note": (
                f"Only {dates_simulated}/{dates_expected} dates simulated "
                f"({coverage:.0%}) — below {_MIN_SIMULATION_COVERAGE:.0%} threshold"
            ),
        }

    if not all_orders:
        return {
            "status": "no_orders",
            "dates_simulated": dates_simulated,
            "dates_expected": dates_expected,
            "coverage": round(coverage, 3),
            "note": "No ENTER signals passed risk rules during the simulation period",
        }

    fees = config.get("simulation_fees", 0.001)
    sim_cfg = config.get("simulation", {})
    slippage_bps = float(sim_cfg.get("slippage_bps", 0))
    assume_next_day_fill = bool(sim_cfg.get("assume_next_day_fill", False))
    pf = orders_to_portfolio(
        all_orders, price_matrix, init_cash=init_cash, fees=fees,
        slippage_bps=slippage_bps, assume_next_day_fill=assume_next_day_fill,
    )
    stats = compute_portfolio_stats(
        pf,
        spy_prices=spy_prices,
        ew_high_vol_basket_returns=ew_high_vol_basket_returns,
    )
    # Record simulation assumptions for reporting
    if slippage_bps > 0 or assume_next_day_fill:
        fill_type = "next-day close" if assume_next_day_fill else "same-day close"
        stats["simulation_assumptions"] = f"Fills: {fill_type} + {slippage_bps:.0f}bp slippage"
    stats["status"] = "ok"
    stats["dates_simulated"] = dates_simulated
    stats["dates_expected"] = dates_expected
    stats["coverage"] = round(coverage, 3)
    stats["total_orders"] = len(all_orders)
    if skipped:
        stats["skip_reasons"] = skipped
    # Pass through price data quality metadata for reporting
    if hasattr(price_matrix, 'attrs'):
        if price_matrix.attrs.get("price_gap_warnings"):
            stats["price_gap_warnings"] = price_matrix.attrs["price_gap_warnings"]
        if price_matrix.attrs.get("staleness_warning"):
            stats["staleness_warning"] = price_matrix.attrs["staleness_warning"]
        if price_matrix.attrs.get("unfilled_gaps"):
            stats["unfilled_gaps"] = price_matrix.attrs["unfilled_gaps"]
    return stats


# ── Replay helper for parity testing (Phase 1.1b) ──────────────────────────


def _load_initial_state_from_eod_pnl(
    trades_db_path: str | None,
    parity_window_start: str,
) -> dict | None:
    """Bootstrap sim_client state from live's ``eod_pnl`` snapshot for the
    most recent trading day strictly before ``parity_window_start``.

    Returns ``None`` when no trades.db is available or when ``eod_pnl``
    coverage doesn't reach back to ``parity_window_start - 1`` — caller
    falls back to cold-start warmup. This is the bootstrap step in the
    Option A long-term parity strategy: instead of replaying the entire
    historical signal stream from day 0 (which compounds drift between
    sim and live across every fill, dividend, corporate action, and
    operator intervention), we pin sim's starting state to live's
    actual realized state at the parity window edge. From there, both
    systems run forward continuously and any divergence within the
    window is genuine logic divergence — not warmup drift.

    The ``eod_pnl.positions_snapshot`` JSON column is the source of
    truth: live's executor writes it after each EOD reconcile pass with
    full ticker / shares / avg_cost / sector. ``total_cash`` and
    ``portfolio_nav`` complete the bootstrap state. Peak NAV is queried
    as ``MAX(portfolio_nav)`` over all ``eod_pnl`` rows up to the
    bootstrap date so drawdown gates start with the correct watermark.
    """
    import json
    import os
    import sqlite3

    if not trades_db_path:
        logger.info("Bootstrap skipped: no trades_db_path in config — cold-start warmup")
        return None
    if not os.path.exists(trades_db_path):
        logger.warning("Bootstrap skipped: trades_db_path %s does not exist", trades_db_path)
        return None

    # eod_pnl rows must carry meaningful state to be useful for bootstrap.
    # ``total_cash`` and ``positions_snapshot`` were added by alpha-engine
    # PR #59 (2026-04-17 EOD cash-attribution fixes); historical rows
    # before that have NULL/empty values. Falling back to the earliest
    # row regardless yielded cash=$0/positions={} which makes sim
    # produce zero ENTERs in the entire window. Require non-null cash
    # AND non-empty positions_snapshot — anything else isn't a usable
    # baseline.
    _MEANINGFUL_CLAUSE = (
        "total_cash IS NOT NULL "
        "AND positions_snapshot IS NOT NULL "
        "AND length(positions_snapshot) > 2"
    )

    conn = sqlite3.connect(trades_db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT date, portfolio_nav, total_cash, positions_snapshot "
            f"FROM eod_pnl WHERE date < ? AND {_MEANINGFUL_CLAUSE} "
            "ORDER BY date DESC LIMIT 1",
            (parity_window_start,),
        ).fetchone()
        if row is None:
            # No meaningful eod_pnl row strictly before window_start. Fall
            # back to the earliest meaningful row available. Caller clips
            # parity dates to >= row.date so we don't validate against
            # state we can't reconstruct.
            row = conn.execute(
                "SELECT date, portfolio_nav, total_cash, positions_snapshot "
                f"FROM eod_pnl WHERE {_MEANINGFUL_CLAUSE} "
                "ORDER BY date ASC LIMIT 1"
            ).fetchone()
            if row is None:
                logger.warning(
                    "Bootstrap skipped: no eod_pnl row in %s has both "
                    "non-null total_cash AND non-empty positions_snapshot. "
                    "Live executor's EOD reconcile may not yet be writing "
                    "those columns — see alpha-engine PR #59.",
                    trades_db_path,
                )
                return None
            logger.warning(
                "Bootstrap fallback: no meaningful eod_pnl row strictly "
                "before parity_window_start=%s; using earliest meaningful "
                "row (date=%s). Caller must clip parity dates to >= %s.",
                parity_window_start, row["date"], row["date"],
            )
        peak_row = conn.execute(
            "SELECT MAX(portfolio_nav) FROM eod_pnl WHERE date <= ?",
            (row["date"],),
        ).fetchone()
        peak_nav = float(peak_row[0]) if peak_row[0] is not None else float(row["portfolio_nav"])
        positions_raw = json.loads(row["positions_snapshot"]) if row["positions_snapshot"] else {}
        # Project to sim_client._positions shape: {ticker: {shares, avg_cost,
        # entry_date, sector}}. The eod_pnl snapshot carries richer fields
        # (market_value, unrealized_pnl, daily_return_pct, alpha_contribution)
        # that the sim doesn't read; drop them to keep the bootstrap dict
        # shape symmetric with what place_market_order writes during the run.
        positions: dict[str, dict] = {}
        for ticker, p in positions_raw.items():
            shares = int(p.get("shares") or 0)
            if shares == 0:
                continue
            positions[ticker] = {
                "shares": shares,
                "avg_cost": float(p.get("avg_cost") or 0.0),
                "entry_date": p.get("entry_date"),  # usually missing — enriched below
                "sector": p.get("sector", ""),
            }

        # Enrich entry_date from trades.db. The eod_pnl positions_snapshot
        # JSON omits entry_date, but the strategy layer's time-decay-EXIT
        # and ATR-trailing-stop logic both depend on it (executor/strategies/
        # exit_manager.py). Live executor enriches via
        # ``trade_logger.get_entry_dates`` post-position-load. Mirror that
        # lookup here so bootstrapped positions carry an entry_date.
        # Use most-recent ENTER strictly on-or-before bootstrap.as_of
        # (anything after as_of doesn't exist yet at bootstrap time).
        if positions:
            as_of = row["date"]
            placeholders = ",".join("?" for _ in positions)
            entry_rows = conn.execute(
                "SELECT ticker, MAX(date) AS entry_date "
                "FROM trades WHERE action='ENTER' AND date <= ? "
                f"AND ticker IN ({placeholders}) "
                "GROUP BY ticker",
                [as_of, *positions.keys()],
            ).fetchall()
            n_enriched = 0
            for r in entry_rows:
                ticker, entry_date = r["ticker"], r["entry_date"]
                if ticker in positions and entry_date:
                    positions[ticker]["entry_date"] = entry_date
                    n_enriched += 1
            # Fall back to as_of for positions that have no ENTER on or before
            # the bootstrap date (manually-seeded positions, pre-system holds,
            # or trades.db rows missing the legacy `date` column). They get
            # treated as fresh-at-bootstrap by the strategy layer — slightly
            # pessimistic but not silently broken.
            for ticker in positions:
                if not positions[ticker].get("entry_date"):
                    positions[ticker]["entry_date"] = as_of
            logger.info(
                "Enriched entry_date on %d/%d bootstrapped positions from "
                "trades.db; %d fell back to bootstrap as_of=%s",
                n_enriched, len(positions),
                len(positions) - n_enriched, as_of,
            )

        return {
            "positions": positions,
            "cash": float(row["total_cash"] or 0.0),
            "peak_nav": peak_nav,
            "as_of": row["date"],
        }
    finally:
        conn.close()


def _build_replay_signals_by_date(
    bucket: str,
    sim_dates: list[str],
    signal_dates: list[str],
) -> dict[str, dict]:
    """Pre-build ``{sim_date: signals_dict}`` for the bootstrap-mode replay.

    On signal-generation days: load that day's signals.json verbatim.
    On non-signal trading days: use the most-recent prior signal date's
    signals.json with ``buy_candidates`` stripped — no new ENTERs fire
    on those days. Phase 2 will port ``entry_triggers.py`` to a
    daily-bar approximation; until then, ENTERs only fire on signal
    days. This mirrors the live executor's ``read_signals_with_fallback``
    semantics: every weekday's executor run uses the most recent signals
    available, then the daemon's intraday triggers gate when ENTERs fill.

    Sparse map: dates with no loadable signals (S3 NoSuchKey, no prior
    signals before the date) are simply omitted. Caller treats absence
    as "skip this date".

    See ROADMAP P0 "2026-04-26 (Sun) — Finalize parity + downstream"
    sim-on-every-weekday Phase 1 for the alpha-bearing rationale: live's
    daemon runs EXIT/REDUCE/strategy-exit on every weekday between
    signal generation, so sim must do the same to avoid NAV / equity-
    exposure divergence.
    """
    from bisect import bisect_right
    from botocore.exceptions import ClientError
    from loaders import signal_loader

    sorted_signal_dates = sorted(signal_dates)
    signal_dates_set = set(sorted_signal_dates)
    cache: dict[str, dict | None] = {}
    out: dict[str, dict] = {}

    def _load(d: str) -> dict | None:
        if d in cache:
            return cache[d]
        try:
            data = signal_loader.load(bucket, d)
        except (FileNotFoundError, ClientError):
            cache[d] = None
            return None
        cache[d] = data
        return data

    for d in sim_dates:
        if d in signal_dates_set:
            base = _load(d)
            if base is not None:
                out[d] = base
            continue
        # Non-signal day — fall back to most-recent prior signal date.
        idx = bisect_right(sorted_signal_dates, d)
        if idx == 0:
            continue
        prior = sorted_signal_dates[idx - 1]
        base = _load(prior)
        if base is None:
            continue
        # Strip buy_candidates so no new ENTERs fire on non-signal days.
        # Held-position EXIT/REDUCE/HOLD live in ``universe`` and are
        # left untouched. Strategy-layer exits (ATR, time-decay) read
        # sim_client._positions directly and don't touch this dict.
        stripped = dict(base)
        stripped["buy_candidates"] = []
        out[d] = stripped

    return out


def _replay_for_dates_per_date_bootstrap(
    *,
    dates: list[str],
    config: dict,
    SimulatedIBKRClient,
    init_cash: float,
    price_matrix,
    ohlcv_by_ticker,
    bucket: str,
    merged_config: dict,
    strategy_config: dict,
) -> list[dict]:
    """Per-date-bootstrap implementation of ``replay_for_dates``.

    For each parity date, bootstrap a fresh ``sim_client`` from the eod_pnl
    snapshot strictly preceding that date, then run ``_simulate_single_date``
    for that single date. No cumulative state across dates; each parity
    date validates against live's exact preceding state. Quick-fix
    alternative to the continuous-bootstrap path; param sweeps continue to
    use the continuous path (this is parity-only).

    Closes ROADMAP P1 entry "Per-parity-date bootstrap (alternative parity
    test mode)" — see backtest.py ``replay_for_dates`` docstring's
    ``per_date_bootstrap`` parameter for the public contract.

    Universe symbols are fetched once at entry (not per-date) — ArcticDB
    publishes a single universe membership snapshot per Saturday SF;
    per-date refetch would be cost without changing behavior. Rejected
    ticker counter is aggregated across dates for the post-loop summary
    log line, matching the continuous path's behavior.
    """
    if not config.get("trades_db_path"):
        raise RuntimeError(
            "per_date_bootstrap=True requires trades_db_path in config — "
            "fresh-bootstrap-per-date depends on eod_pnl history. Set "
            "config['trades_db_path'] to the path of a live trades.db "
            "with non-empty eod_pnl rows preceding the parity window."
        )

    try:
        from alpha_engine_lib.arcticdb import get_universe_symbols
        universe_symbols = get_universe_symbols(bucket)
    except Exception as exc:
        raise RuntimeError(
            f"Per-date-bootstrap universe-filter bootstrap failed: could not "
            f"read ArcticDB universe symbols from bucket {bucket!r}: {exc}"
        ) from exc

    rejected_ticker_counter: dict[str, int] = {}
    captured: list[dict] = []
    n_skipped_no_bootstrap = 0
    skipped_dates: list[str] = []

    for parity_date in sorted(dates):
        bootstrap_state = _load_initial_state_from_eod_pnl(
            config["trades_db_path"], parity_date,
        )
        if bootstrap_state is None:
            n_skipped_no_bootstrap += 1
            skipped_dates.append(parity_date)
            logger.warning(
                "Per-date bootstrap skipped for %s: no meaningful eod_pnl "
                "row strictly before. Dropping this parity date from output.",
                parity_date,
            )
            continue

        # Fresh sim_client per-date — disjoint from any prior date's state.
        # NAV is overridden from the bootstrap row immediately below; the
        # constructor's init_cash arg is just a placeholder shape.
        sim_client = SimulatedIBKRClient(prices={}, nav=init_cash)
        sim_client._cash = bootstrap_state["cash"]
        sim_client._positions = bootstrap_state["positions"]
        sim_client._peak_nav = bootstrap_state["peak_nav"]
        logger.info(
            "Per-date bootstrap (%s): as_of=%s, %d positions, "
            "cash=$%.0f, peak_nav=$%.0f",
            parity_date, bootstrap_state["as_of"],
            len(bootstrap_state["positions"]),
            bootstrap_state["cash"], bootstrap_state["peak_nav"],
        )

        orders, _skip = _simulate_single_date(
            sim_client=sim_client,
            signal_date=parity_date,
            price_matrix=price_matrix,
            ohlcv_by_ticker=ohlcv_by_ticker,
            bucket=bucket,
            merged_config=merged_config,
            strategy_config=strategy_config,
            signals_override=None,  # loaded per-date inside helper
            universe_symbols=universe_symbols,
            rejected_ticker_counter=rejected_ticker_counter,
        )
        if orders:
            captured.extend(orders)

    if rejected_ticker_counter:
        top = sorted(rejected_ticker_counter.items(), key=lambda kv: -kv[1])
        total_rejects = sum(rejected_ticker_counter.values())
        logger.warning(
            "Per-date-bootstrap universe-filter dropped %d signal entries "
            "across %d tickers. Top offenders: %s",
            total_rejects, len(rejected_ticker_counter),
            [f"{t}={n}" for t, n in top[:10]],
        )

    logger.info(
        "replay_for_dates (per_date_bootstrap=True): %d orders captured "
        "across %d requested dates (%d skipped — no eod_pnl row before "
        "those dates: %s)",
        len(captured), len(dates), n_skipped_no_bootstrap,
        skipped_dates[:5] + (["…"] if len(skipped_dates) > 5 else []),
    )
    return captured


def replay_for_dates(
    dates: list[str],
    config: dict,
    *,
    warmup_from_full_history: bool = True,
    per_date_bootstrap: bool = False,
) -> list[dict]:
    """
    Replay the backtester for a specific list of signal dates; return
    aggregated orders tagged with ``date``.

    Primary consumer: ``tests/test_parity_replay.py`` (Phase 1.1 replay
    parity test). See ``docs/trade_mapping.md`` for the tolerance contract
    used to diff the returned orders against ``trades.db``.

    Parameters
    ----------
    dates : signal dates to replay orders for, ``"YYYY-MM-DD"`` each.
    config : loaded via ``pipeline_common.load_config``.
    warmup_from_full_history : if True (default), replay the FULL historical
        signal stream up through the latest requested date so the sim_client's
        NAV / positions have time to evolve before the test window. Only
        orders on ``dates`` are returned. If False, only the requested dates
        are simulated starting from ``init_cash`` — fast but NAV-divergent.
    per_date_bootstrap : when True, bootstrap a FRESH sim_client per parity
        date from the eod_pnl row strictly preceding each date, run a single
        date through the executor, capture orders. Loses cumulative state
        across the parity window but each parity date validates against
        live's exact preceding state. Quick-fix alternative to the continuous
        bootstrap path; use case is parity-only validation when continuous
        sim diverges before reaching the parity window. Requires
        ``trades_db_path`` in ``config``; raises ``RuntimeError`` otherwise.
        Param sweeps stay on the continuous path (this flag is parity-only).
        Mutually exclusive with bootstrap-mode signal_by_date construction —
        signals are loaded per-date inside ``_simulate_single_date``.

    Returns
    -------
    list of order dicts — each with at minimum ``date``, ``ticker``, ``action``.
    Empty list on any simulation setup failure (stale prices, no price index).

    State-reconstruction note
    -------------------------
    Neither mode perfectly reconstructs the live executor's state at each
    date — live NAV on any given date reflects prior realized P&L that the
    backtester's simulated P&L can drift from. ``position_pct`` tolerance
    in ``docs/trade_mapping.md`` accounts for small drift; large drift is
    a signal of logic divergence worth investigating.

    When ``trades_db_path`` is configured and an ``eod_pnl`` snapshot exists
    strictly before ``min(dates)``, sim bootstraps from that snapshot and
    runs the full daily-heartbeat path (Phase 1, sim-on-every-weekday):
    every trading day in ``[bootstrap.as_of, max(dates)]`` invokes the
    executor — signal days use that day's signals.json; non-signal days
    use the most-recent prior signals.json with ``buy_candidates`` stripped
    so EXIT/REDUCE/strategy-exit fire while ENTERs are gated until Phase 2
    ports ``entry_triggers.py`` to a daily-bar approximation. Without
    bootstrap, the legacy paths apply (warmup_from_full_history or simple
    requested-dates-only).
    """
    executor_run, SimulatedIBKRClient, all_signal_dates, price_matrix, init_cash, ohlcv_by_ticker = \
        _setup_simulation(config)

    # Hard-fail on setup-level problems per feedback_no_silent_fails.
    # Returning [] here would let the parity test interpret "no orders" as a
    # legitimate backtester outcome, surfacing every live trade as a spurious
    # "only_live" divergence — logic failure indistinguishable from data
    # failure. Raising surfaces the actual cause in the test error message.
    if price_matrix is None:
        raise RuntimeError(
            "replay_for_dates: _setup_simulation returned no price matrix — "
            "cannot replay. Likely causes: ArcticDB unreachable, empty signal "
            "history, or fewer than `min_simulation_dates` signal dates in S3."
        )
    if getattr(price_matrix, "attrs", {}).get("stale_circuit_break"):
        raise RuntimeError(
            f"replay_for_dates: price-matrix staleness circuit-breaker tripped "
            f"({price_matrix.attrs.get('staleness_warning')}). Refusing to "
            f"produce parity output against stale prices."
        )

    bucket = config.get("signals_bucket", "alpha-engine-research")
    merged_config, strategy_config = _build_merged_simulate_config(config)

    # Per-date bootstrap path — diverges from the continuous-bootstrap +
    # warmup paths because each parity date gets a fresh sim_client
    # anchored to live's eod_pnl state at the preceding day. See the
    # docstring's `per_date_bootstrap` parameter and ROADMAP P1 entry
    # "Per-parity-date bootstrap (alternative parity test mode)" for the
    # rationale; closes that entry. No requested-date clipping or
    # carry-forward signal handling — each date is fully self-contained.
    if per_date_bootstrap:
        return _replay_for_dates_per_date_bootstrap(
            dates=dates,
            config=config,
            SimulatedIBKRClient=SimulatedIBKRClient,
            init_cash=init_cash,
            price_matrix=price_matrix,
            ohlcv_by_ticker=ohlcv_by_ticker,
            bucket=bucket,
            merged_config=merged_config,
            strategy_config=strategy_config,
        )

    requested = set(dates)
    sim_client = SimulatedIBKRClient(prices={}, nav=init_cash)

    # Bootstrap sim_client state from live's eod_pnl if a trades.db is
    # available and the parity window starts after eod_pnl coverage begins.
    # See ``_load_initial_state_from_eod_pnl`` docstring for the rationale.
    # When bootstrap succeeds, the long-warmup replay below is REPLACED by
    # the bootstrap state — sim runs only the requested dates with live's
    # actual portfolio at the window edge as its starting point.
    bootstrap_state: dict | None = None
    if dates:
        earliest_requested = min(dates)
        bootstrap_state = _load_initial_state_from_eod_pnl(
            config.get("trades_db_path"), earliest_requested,
        )
        if bootstrap_state:
            sim_client._cash = bootstrap_state["cash"]
            sim_client._positions = bootstrap_state["positions"]
            sim_client._peak_nav = bootstrap_state["peak_nav"]
            logger.info(
                "Bootstrapped sim from live eod_pnl as_of=%s: "
                "%d positions, cash=$%.0f, peak_nav=$%.0f",
                bootstrap_state["as_of"],
                len(bootstrap_state["positions"]),
                bootstrap_state["cash"],
                bootstrap_state["peak_nav"],
            )

    # Load today's ArcticDB universe once — used to filter historical signals
    # that reference since-dropped tickers (e.g. TSM/ASML post-2026-04-20).
    # Mirrors the _run_simulation_loop pattern. Without this, parity replay
    # of a date with a dropped ticker hits the executor's load_daily_vwap
    # NoSuchVersionException hard-fail and aborts the entire replay
    # (observed 2026-04-24 parity dry-run on date 2026-03-09 with TSM).
    universe_symbols: set[str] | None = None
    rejected_ticker_counter: dict[str, int] = {}
    try:
        from alpha_engine_lib.arcticdb import get_universe_symbols
        universe_symbols = get_universe_symbols(bucket)
    except Exception as exc:
        raise RuntimeError(
            f"Replay universe-filter bootstrap failed: could not read "
            f"ArcticDB universe symbols from bucket {bucket!r}: {exc}"
        ) from exc

    signals_by_date: dict[str, dict] | None = None
    if bootstrap_state is not None:
        # Bootstrap pinned sim to live state at as_of. From there we run
        # sim continuously through every TRADING DAY up to max(dates),
        # capturing orders only on the requested dates. Running just the
        # signal dates skipped the 2-5 weekdays between signal generation
        # where live's daemon ran EXIT/REDUCE/strategy-exit logic — sim's
        # NAV diverged within hours of bootstrap (see 2026-04-25 handoff:
        # equity exposure stuck at 90%+ → all parity-window ENTERs blocked).
        # Phase 1 (sim-on-every-weekday) closes that gap by iterating the
        # canonical NYSE-trading-day axis from price_matrix.index, with
        # buy_candidates stripped on non-signal days so no new ENTERs fire
        # without intraday-trigger evaluation (Phase 2 territory).
        #
        # Clip captured orders to dates >= bootstrap.as_of: pre-bootstrap
        # parity dates can't be validated (sim's state came from after
        # they happened). Drop them with a named WARNING.
        as_of = bootstrap_state["as_of"]
        post_bootstrap_requested = sorted(d for d in dates if d >= as_of)
        n_dropped = len(dates) - len(post_bootstrap_requested)
        if n_dropped:
            dropped_dates = sorted(d for d in dates if d < as_of)
            logger.warning(
                "Parity window clipped: %d/%d requested dates predate the "
                "bootstrap as_of=%s — dropping %s. Increase eod_pnl coverage "
                "or shorten the parity window to include only post-bootstrap "
                "dates.",
                n_dropped, len(dates), as_of, dropped_dates,
            )
        if post_bootstrap_requested:
            latest_requested = max(post_bootstrap_requested)
            ts_low = pd.Timestamp(as_of)
            ts_high = pd.Timestamp(latest_requested)
            sim_dates = [
                d.strftime("%Y-%m-%d")
                for d in price_matrix.index
                if ts_low <= d <= ts_high
            ]
            signals_by_date = _build_replay_signals_by_date(
                bucket, sim_dates, all_signal_dates,
            )
            n_signal_days = sum(1 for d in sim_dates if d in set(all_signal_dates))
            n_carry_days = len(signals_by_date) - n_signal_days
            logger.info(
                "Phase 1 daily-heartbeat: %d trading days in [%s, %s] — "
                "%d signal-days + %d carry-forward days (buy_candidates "
                "stripped). %d trading days have no prior signals available "
                "and will be skipped.",
                len(sim_dates), as_of, latest_requested,
                n_signal_days, n_carry_days,
                len(sim_dates) - len(signals_by_date),
            )
        else:
            sim_dates = []
        requested = set(post_bootstrap_requested)
    elif warmup_from_full_history and dates:
        latest_requested = max(dates)
        sim_dates = [d for d in all_signal_dates if d <= latest_requested]
    else:
        sim_dates = sorted(dates)

    captured: list[dict] = []
    for signal_date in sim_dates:
        # signals_by_date is set when bootstrap-mode replay extends sim to
        # every trading day in the parity window (Phase 1). Absence at a
        # given date means no prior signals.json exists yet — skip without
        # invoking the executor (avoids "no_signals" log spam on early
        # parity windows). When unset (long-warmup or simple paths),
        # _simulate_single_date loads signals per date from S3 itself.
        if signals_by_date is not None:
            signals_override = signals_by_date.get(signal_date)
            if signals_override is None:
                continue
        else:
            signals_override = None
        orders, _skip = _simulate_single_date(
            sim_client=sim_client,
            signal_date=signal_date,
            price_matrix=price_matrix,
            ohlcv_by_ticker=ohlcv_by_ticker,
            bucket=bucket,
            merged_config=merged_config,
            strategy_config=strategy_config,
            signals_override=signals_override,
            universe_symbols=universe_symbols,
            rejected_ticker_counter=rejected_ticker_counter,
        )
        if orders and signal_date in requested:
            captured.extend(orders)

    if rejected_ticker_counter:
        top = sorted(rejected_ticker_counter.items(), key=lambda kv: -kv[1])
        total_rejects = sum(rejected_ticker_counter.values())
        logger.warning(
            "Replay universe-filter dropped %d signal entries across %d "
            "tickers (present in historical signals but absent from current "
            "ArcticDB universe). Top offenders: %s",
            total_rejects, len(rejected_ticker_counter),
            [f"{t}={n}" for t, n in top[:10]],
        )

    logger.info(
        "replay_for_dates: %d orders captured across %d requested dates "
        "(warmup=%s, replayed=%d)",
        len(captured), len(requested), warmup_from_full_history, len(sim_dates),
    )
    return captured


# ── Param sweep helpers ─────────────────────────────────────────────────────


def _seed_grid_with_current(grid: dict, current_params: dict | None) -> dict:
    """
    Inject current S3 executor param values into the sweep grid so the
    optimizer iterates on last week's best rather than searching from
    scratch. Values already in the grid are not duplicated.
    """
    if not current_params:
        return grid

    grid = {k: list(v) for k, v in grid.items()}  # shallow copy
    for key, val in current_params.items():
        if key in grid and val not in grid[key]:
            grid[key].append(val)
            grid[key].sort()
            logger.info("Seeded grid[%s] with current S3 value: %s", key, val)
    return grid


_DIRECT_RISK_PARAMS = {"min_score", "max_position_pct", "drawdown_circuit_breaker"}
_STRATEGY_EXIT_PARAMS = {
    "atr_multiplier": "atr_multiplier",
    "time_decay_reduce_days": "time_decay_reduce_days",
    "time_decay_exit_days": "time_decay_exit_days",
    "profit_take_pct": "profit_take_pct",
}
_RECOGNIZED_SWEEP_PARAMS = _DIRECT_RISK_PARAMS | set(_STRATEGY_EXIT_PARAMS)


def _build_config_override(config: dict) -> dict | None:
    """
    Map flat sweep params in config to the nested executor config structure.

    Sweep grid uses flat keys (e.g. atr_multiplier) but the executor expects
    them nested under strategy.exit_manager. This function builds the override
    dict that executor.main.run(config_override=) can merge.
    """
    override = {}

    # Direct risk params (top-level in executor's risk.yaml)
    for key in _DIRECT_RISK_PARAMS:
        if key in config:
            override[key] = config[key]

    # Strategy params → nested under strategy.exit_manager
    exit_manager_overrides = {}
    for sweep_key, config_key in _STRATEGY_EXIT_PARAMS.items():
        if sweep_key in config:
            exit_manager_overrides[config_key] = config[sweep_key]

    if exit_manager_overrides:
        override["strategy"] = {"exit_manager": exit_manager_overrides}

    # Warn about sweep params present in config but not mapped to executor
    from optimizer.executor_optimizer import SAFE_PARAMS
    sweep_params_in_config = {k for k in config if k in SAFE_PARAMS}
    unmapped = sweep_params_in_config - _RECOGNIZED_SWEEP_PARAMS
    if unmapped:
        logger.warning(
            "Sweep params not mapped to executor config (will be ignored): %s", unmapped
        )

    return override if override else None


# ── Convenience wrappers ────────────────────────────────────────────────────


def run_simulate(config: dict) -> dict:
    """
    Run Mode 2: replay all historical signal dates through the executor with
    SimulatedIBKRClient, then compute portfolio metrics via vectorbt.

    Returns a stats dict. Returns {"status": "insufficient_data"} if fewer than
    config["min_simulation_dates"] signal dates exist in S3.
    """
    executor_run, SimulatedIBKRClient, dates, price_matrix, init_cash, ohlcv = _setup_simulation(config)
    min_dates = config.get("min_simulation_dates", 5)

    if price_matrix is None:
        return {
            "status": "insufficient_data",
            "dates_available": len(dates),
            "min_required": min_dates,
        }

    # Build the EW-high-vol basket once per simulate call so portfolio_stats
    # can emit `alpha_vs_ew_high_vol` alongside `total_alpha`. Pure-compute,
    # no new data dependency — uses the same `price_matrix` the simulator
    # already loaded. Returns None on insufficient history; basket is best-
    # effort observability, never load-bearing.
    ew_basket = _try_construct_ew_high_vol_basket(price_matrix)

    # L4471 L1+L2: build the resilience context for the standalone simulate
    # phase ONLY (param sweeps call _run_simulation_loop without this, so they
    # keep the legacy heartbeat + pay no checkpoint I/O). Enables per-date
    # instrumentation + within-run checkpoint/resume keyed on an input
    # fingerprint (invalidates on executor-param / date / sim-code change).
    resilience_ctx = None
    try:
        import boto3
        from store.sim_checkpoint import compute_fingerprint
        _run_date = config.get("_run_date")
        if _run_date:
            resilience_ctx = {
                "enabled": True,
                "bucket": config.get("signals_bucket", "alpha-engine-research"),
                "run_date": _run_date,
                "fingerprint": compute_fingerprint(config, list(dates)),
                "s3_client": boto3.client("s3"),
                "checkpoint_every": int(config.get("sim_checkpoint_every", 5)),
                "per_date_warn_s": float(config.get("sim_per_date_warn_s", 60)),
                # Budget for the projected-overrun WARN = the simulation_pipeline
                # phase hard-cap (timing_budget.yaml, 2700s). When the projected
                # runtime exceeds the cap, WARN early (with the per-date rate) so
                # the cause is localized BEFORE the watchdog kills the phase —
                # the fix is a faster sim / L3, never a wider cap.
                "budget_warn_s": float(config.get("sim_projected_budget_s", 2700)),
                "warmup_dates": int(config.get("sim_warmup_dates", 5)),
            }
    except Exception as exc:  # resilience is best-effort — never block the sim
        logger.warning("[sim] resilience_ctx setup failed (%s) — running without L1/L2", exc)
        resilience_ctx = None

    return _run_simulation_loop(
        executor_run, SimulatedIBKRClient, dates, price_matrix, config,
        ohlcv_by_ticker=ohlcv,
        ew_high_vol_basket_returns=ew_basket,
        resilience_ctx=resilience_ctx,
    )


def run_param_sweep(config: dict) -> pd.DataFrame | None:
    """
    Run Mode 2 across a grid of risk + strategy parameters. Price matrix and
    OHLCV histories are built once and reused for all combinations — only the
    simulation loop re-runs per combo.

    Returns a DataFrame sorted by sharpe_ratio, or an empty DataFrame if
    insufficient data is available.
    """
    import pandas as pd

    executor_run, SimulatedIBKRClient, dates, price_matrix, _, ohlcv = _setup_simulation(config)

    if price_matrix is None:
        logger.warning(
            "Param sweep skipped: only %d signal dates available", len(dates)
        )
        return pd.DataFrame()

    # Precompute ATR + VWAP + coverage maps ONCE across the full combo
    # sweep. Without this, _run_simulation_loop derives them lazily per
    # combo and we repay the ~900-ticker ArcticDB bulk read 60 times.
    from store.feature_maps import load_precomputed_feature_maps
    bucket = config.get("signals_bucket", "alpha-engine-research")
    _smoke_tickers = config.get("smoke_tickers")
    _allowlist = set(_smoke_tickers) if _smoke_tickers else None
    atr_by_ticker, vwap_series_by_ticker, coverage_by_ticker = load_precomputed_feature_maps(
        bucket, tickers_allowlist=_allowlist,
    )

    # Build the EW-high-vol basket once for the full sweep — depends only
    # on prices, not on the executor config combo being swept. Shared
    # across every _run_simulation_loop call so each per-combo
    # portfolio_stats() can emit alpha_vs_ew_high_vol without recomputing
    # the basket per combo.
    ew_basket = _try_construct_ew_high_vol_basket(price_matrix)

    def sim_fn(combo_config: dict) -> dict:
        return _run_simulation_loop(
            executor_run, SimulatedIBKRClient, dates, price_matrix, combo_config,
            ohlcv_by_ticker=ohlcv,
            ew_high_vol_basket_returns=ew_basket,
            atr_by_ticker=atr_by_ticker,
            vwap_series_by_ticker=vwap_series_by_ticker,
            coverage_by_ticker=coverage_by_ticker,
        )

    grid = config.get("param_sweep", param_sweep.DEFAULT_GRID)
    sweep_settings = config.get("param_sweep_settings", {})

    logger.info("Running param sweep (%s): %s", sweep_settings.get("mode", "random"), {k: len(v) for k, v in grid.items()})
    return param_sweep.sweep(grid, sim_fn, config, sweep_settings=sweep_settings)


def run_production_strategy_backtest(config: dict, s3_client=None) -> dict:
    """Run the DEPLOYED strategy's backtest for the weekly report headline
    (config#1053): the production research cohort + α̂ → the production MVO solver
    (``executor.portfolio_optimizer.solve_target_weights``), over the
    ``predictor/predictions/`` archive window. This is the system as it actually
    trades since the 2026-05-13 cutover — NOT the legacy-1/n / synthetic-GBM
    component checks that previously headlined the email.

    Reuses the cutover-gate's input + solver machinery
    (``build_production_signal_inputs`` → ``run_optimizer_backtest``) but returns
    the raw optimizer metrics for presentation rather than a pass/fail verdict.

    FAIL LOUD, never crash: returns ``{"status": "ok", "metrics", ...}`` on
    success, else ``{"status": <reason>, "error"}``. A non-"ok" return makes the
    reporter render a prominent banner — so a component number can never silently
    become the de-facto headline (the 2026-06-12 failure mode). Any exception is
    caught and surfaced the same way (the weekly run must not die on the headline
    backtest)."""
    try:
        import os

        from analysis.portfolio_optimizer_backtest import run_optimizer_backtest
        from synthetic.production_signal_backtest import build_production_signal_inputs

        inputs = build_production_signal_inputs(config, s3_client=s3_client)
        if inputs.get("status") != "ok":
            return {
                "status": inputs.get("status", "error"),
                "error": inputs.get("error", "production signal inputs unavailable"),
            }

        executor_paths = config.get("executor_paths", [])
        if isinstance(executor_paths, str):
            executor_paths = [executor_paths]
        executor_path = next((p for p in executor_paths if os.path.isdir(p)), None)
        if not executor_path:
            return {
                "status": "error",
                "error": (
                    f"executor_paths not found on disk: {executor_paths}; add the "
                    "alpha-engine repo root to executor_paths in config.yaml"
                ),
            }

        opt = run_optimizer_backtest(
            predictions_by_date=inputs["predictions_by_date"],
            price_matrix=inputs["price_matrix"],
            spy_prices=inputs["spy_prices"],
            sector_map=inputs["sector_map"],
            executor_path=executor_path,
        )
        return {
            "status": "ok",
            "metrics": opt.metrics,
            "production_window": inputs.get("production_window"),
            "n_production_dates": inputs.get("n_production_dates"),
            "n_rebalances": opt.n_rebalances,
            "n_solver_failures": opt.n_solver_failures,
        }
    except Exception as e:  # noqa: BLE001 — fail loud in the report, don't crash the run
        logger.warning(
            "production-strategy backtest failed (non-fatal; report banners it): %s",
            e, exc_info=True,
        )
        return {"status": "error", "error": str(e)}


def run_portfolio_optimizer_gate(
    config: dict,
    run_date: str,
    legacy_metrics: dict | None = None,
    s3_client=None,
    signal_source: str = "synthetic",
) -> dict:
    """
    Run the portfolio-optimizer cutover gate and persist the report to S3.

    ROADMAP L2222 PR 4.5 / L124 PR 2. Orchestrates
    ``analysis.portfolio_optimizer_gate``'s
    ``run_gate_against_predictor_backtest`` ({synthetic | production} history
    → optimizer backtest → compare → evaluate_gate).

    ``signal_source="synthetic"`` (default) writes to the existing
    ``s3://{bucket}/predictor/optimizer_gate/{run_date}.json`` +
    ``.../latest.json`` keys the Saturday SF reads as the PR 5 readiness
    signal — unchanged, so the consumer contract is untouched.
    ``signal_source="production"`` (L124 PR 2 — the gate run against the
    deployed research cohort, the verdict the operator can trust as the
    SOLE promotion lever) writes to an *additive* sibling namespace
    ``.../optimizer_gate/production/{run_date}.json`` +
    ``.../production/latest.json``, so the two sources never overwrite
    each other and existing readers are unaffected.

    legacy_metrics: optional dict from a same-run simulate stage's
    ``portfolio_stats`` to give the gate side-by-side comparisons against
    the in-production legacy planner. None is valid — the gate will skip
    legacy-relative criteria and still check the absolute criteria
    (psr_min, tracking_error_range, active_share_range).
    """
    import boto3
    from analysis.portfolio_optimizer_gate import (
        gate_passed,
        run_gate_against_predictor_backtest,
    )

    logger.info(
        "portfolio_optimizer_gate: starting (signal_source=%s, legacy_metrics=%s)",
        signal_source, "provided" if legacy_metrics else "skipped",
    )
    gate_result = run_gate_against_predictor_backtest(
        config=config,
        legacy_metrics=legacy_metrics,
        signal_source=signal_source,
    )

    bucket = config.get("signals_bucket", "alpha-engine-research")
    prefix = (
        "predictor/optimizer_gate/production"
        if signal_source == "production"
        else "predictor/optimizer_gate"
    )
    key = f"{prefix}/{run_date}.json"
    latest_key = f"{prefix}/latest.json"
    payload = {
        "run_date": run_date,
        "signal_source": signal_source,
        "passed": gate_passed(gate_result.get("gate_report") or {}),
        **gate_result,
    }
    body = json.dumps(payload, default=str, indent=2).encode("utf-8")

    s3 = s3_client or boto3.client("s3")
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
        s3.put_object(Bucket=bucket, Key=latest_key, Body=body, ContentType="application/json")
        logger.info("portfolio_optimizer_gate: persisted s3://%s/%s", bucket, key)
    except Exception as exc:
        logger.warning(
            "portfolio_optimizer_gate: S3 persist failed (non-fatal): %s", exc,
        )

    verdict = (gate_result.get("gate_report") or {}).get("verdict")
    logger.info(
        "portfolio_optimizer_gate: verdict=%s passed=%s",
        verdict, payload["passed"],
    )
    return payload


def run_cov_estimator_sweep_stage(
    config: dict,
    run_date: str,
    s3_client=None,
) -> dict:
    """Run the covariance-estimator sweep (A.4) over synthetic predictor
    history and persist the verdict to
    ``s3://{bucket}/backtest/{run_date}/cov_sweep.json``.

    ROADMAP A.4b — without this CLI wiring the sweep verdict is operator-
    on-demand. Saturday SF integration: when ``--mode all`` runs (or
    ``portfolio-optimizer-backtest`` for ad-hoc), this stage fires the
    8-cell LW/OAS/EWMA × H × λ sweep and writes the report alongside
    the existing ``portfolio_optimizer_gate`` artifact for the operator's
    covariance-cutover decision.

    The sweep itself is non-fatal — like the optimizer gate, this is
    observability, not a backtester-pipeline blocker. Returns the
    report dict; raises on construction failure (mirrors
    ``run_portfolio_optimizer_gate`` semantics).

    Default cells (from ``analysis.portfolio_optimizer_backtest.default_cov_sweep_cells``):
    LW (Ledoit-Wolf) × OAS × pure EWMA × EWMA+Ledoit-Wolf, each at
    H ∈ {1, 21} sigma_horizon_days and λ_decay ∈ {0.94, 0.97} for the
    EWMA variants. Baseline (first cell) is LW + H=1 (legacy MVO).
    """
    import os
    import boto3
    from analysis.portfolio_optimizer_backtest import (
        run_cov_estimator_sweep,
    )
    from synthetic.predictor_backtest import run as run_predictor_pipeline

    logger.info("cov_estimator_sweep: starting")

    executor_paths = config.get("executor_paths", [])
    if isinstance(executor_paths, str):
        executor_paths = [executor_paths]
    executor_path = next((p for p in executor_paths if os.path.isdir(p)), None)
    if not executor_path:
        raise ValueError(
            f"executor_paths not found on disk: {executor_paths}. "
            "Add the alpha-engine repo root to executor_paths in config.yaml."
        )

    pred_result = run_predictor_pipeline(config, keep_predictions=True)
    if pred_result.get("status") != "ok":
        logger.warning(
            "cov_estimator_sweep: predictor backtest status=%s — sweep skipped",
            pred_result.get("status"),
        )
        return {
            "run_date": run_date,
            "status": "skipped",
            "reason": f"predictor backtest status={pred_result.get('status')!r}",
        }

    sweep_report = run_cov_estimator_sweep(
        predictions_by_date=pred_result["predictions_by_date"],
        price_matrix=pred_result["price_matrix"],
        spy_prices=pred_result["spy_prices"],
        sector_map=pred_result["sector_map"],
        executor_path=executor_path,
    )

    bucket = config.get("signals_bucket", "alpha-engine-research")
    key = f"backtest/{run_date}/cov_sweep.json"
    payload = {
        "run_date": run_date,
        "status": "ok",
        **sweep_report,
    }
    body = json.dumps(payload, default=str, indent=2).encode("utf-8")

    s3 = s3_client or boto3.client("s3")
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
        logger.info("cov_estimator_sweep: persisted s3://%s/%s", bucket, key)
    except Exception as exc:
        logger.warning(
            "cov_estimator_sweep: S3 persist failed (non-fatal): %s", exc,
        )

    logger.info(
        "cov_estimator_sweep: baseline=%s winner=%s ranking_top=%s",
        sweep_report.get("baseline_name"),
        sweep_report.get("winner_name"),
        (sweep_report.get("ranking") or [(None, None)])[0],
    )
    return payload


def _load_alpha_uncertainty_from_predictions_archive(
    bucket: str,
    target_dates: list[str],
    s3_client=None,
) -> dict[str, dict[str, float]]:
    """Load ``predicted_alpha_std`` per ticker from the production
    predictions archive for each date in ``target_dates``.

    Returns ``{date: {ticker: std}}`` filtered to dates where the
    archive carries non-None std values. Dates without the archive
    file are silently skipped (None means BR cutover hadn't promoted
    yet for that date). The caller decides whether the resulting
    coverage is sufficient.
    """
    import boto3

    s3 = s3_client or boto3.client("s3")
    out: dict[str, dict[str, float]] = {}
    for date_str in target_dates:
        key = f"predictor/predictions/{date_str}.json"
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
        except Exception:
            continue
        try:
            doc = json.loads(obj["Body"].read())
        except Exception:
            continue
        per_ticker = {
            ticker: float(entry["predicted_alpha_std"])
            for ticker, entry in (doc.get("predictions") or {}).items()
            if isinstance(entry, dict)
            and entry.get("predicted_alpha_std") is not None
        }
        if per_ticker:
            out[date_str] = per_ticker
    return out


_GAMMA_SWEEP_MIN_DATE_COVERAGE = 10


def run_gamma_sweep_stage(
    config: dict,
    run_date: str,
    s3_client=None,
) -> dict:
    """Run the α̂-uncertainty γ-sweep (B.4) over synthetic predictor history
    augmented with production σ_α̂, and persist the verdict to
    ``s3://{bucket}/backtest/{run_date}/gamma_sweep.json``.

    ROADMAP B.4b — without this CLI wiring the sweep verdict is operator-
    on-demand. Auto-skips when σ_α̂ coverage is insufficient (i.e.
    before predictor B.1 has accumulated enough Saturday cycles emitting
    non-None ``predicted_alpha_std``); activates automatically once
    coverage clears the ``_GAMMA_SWEEP_MIN_DATE_COVERAGE`` threshold.

    The sweep itself is non-fatal — like the optimizer gate and cov-
    sweep, this is observability, not a backtester-pipeline blocker.

    Default cells from ``default_gamma_sweep_cells()``: γ ∈ {0.0, 0.05,
    0.10, 0.20, 0.40}. Baseline (first cell) is γ=0 (legacy MVO behavior).
    """
    import os
    import boto3
    from analysis.portfolio_optimizer_backtest import (
        run_gamma_sweep,
    )
    from synthetic.predictor_backtest import run as run_predictor_pipeline

    logger.info("gamma_sweep: starting")

    executor_paths = config.get("executor_paths", [])
    if isinstance(executor_paths, str):
        executor_paths = [executor_paths]
    executor_path = next((p for p in executor_paths if os.path.isdir(p)), None)
    if not executor_path:
        raise ValueError(
            f"executor_paths not found on disk: {executor_paths}. "
            "Add the alpha-engine repo root to executor_paths in config.yaml."
        )

    pred_result = run_predictor_pipeline(config, keep_predictions=True)
    if pred_result.get("status") != "ok":
        logger.warning(
            "gamma_sweep: predictor backtest status=%s — sweep skipped",
            pred_result.get("status"),
        )
        return {
            "run_date": run_date,
            "status": "skipped",
            "reason": f"predictor backtest status={pred_result.get('status')!r}",
        }

    bucket = config.get("signals_bucket", "alpha-engine-research")
    target_dates = sorted(pred_result["predictions_by_date"].keys())
    alpha_uncertainty_by_date = (
        _load_alpha_uncertainty_from_predictions_archive(
            bucket=bucket,
            target_dates=target_dates,
            s3_client=s3_client,
        )
    )

    if len(alpha_uncertainty_by_date) < _GAMMA_SWEEP_MIN_DATE_COVERAGE:
        # σ_α̂ coverage too sparse — γ-sweep against missing uncertainty
        # collapses every cell to the baseline solve, so the verdict
        # carries no information. Skip until predictor B.1 has accumulated
        # enough Saturday cycles emitting non-None predicted_alpha_std.
        reason = (
            f"insufficient σ_α̂ coverage: "
            f"{len(alpha_uncertainty_by_date)} of {len(target_dates)} dates "
            f"have non-None predicted_alpha_std "
            f"(threshold {_GAMMA_SWEEP_MIN_DATE_COVERAGE}); "
            "γ-sweep cannot meaningfully run yet. Activates automatically "
            "once predictor B.1 (BayesianRidge) has emitted std on enough "
            "Saturday cycles."
        )
        logger.warning("gamma_sweep: skipped — %s", reason)
        return {
            "run_date": run_date,
            "status": "skipped",
            "reason": reason,
            "n_dates_with_uncertainty": len(alpha_uncertainty_by_date),
            "n_target_dates": len(target_dates),
        }

    sweep_report = run_gamma_sweep(
        predictions_by_date=pred_result["predictions_by_date"],
        price_matrix=pred_result["price_matrix"],
        spy_prices=pred_result["spy_prices"],
        sector_map=pred_result["sector_map"],
        executor_path=executor_path,
        alpha_uncertainty_by_date=alpha_uncertainty_by_date,
    )

    key = f"backtest/{run_date}/gamma_sweep.json"
    payload = {
        "run_date": run_date,
        "status": "ok",
        "n_dates_with_uncertainty": len(alpha_uncertainty_by_date),
        "n_target_dates": len(target_dates),
        **sweep_report,
    }
    body = json.dumps(payload, default=str, indent=2).encode("utf-8")

    s3 = s3_client or boto3.client("s3")
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
        logger.info("gamma_sweep: persisted s3://%s/%s", bucket, key)
    except Exception as exc:
        logger.warning(
            "gamma_sweep: S3 persist failed (non-fatal): %s", exc,
        )

    logger.info(
        "gamma_sweep: baseline=%s winner=%s ranking_top=%s coverage=%d/%d",
        sweep_report.get("baseline_name"),
        sweep_report.get("winner_name"),
        (sweep_report.get("ranking") or [(None, None)])[0],
        len(alpha_uncertainty_by_date),
        len(target_dates),
    )
    return payload


def run_optimizer_param_sweep_stage(
    config: dict,
    run_date: str,
    s3_client=None,
) -> dict:
    """Sweep the MVO optimizer's PRIMARY params (risk_aversion × tcost_bps)
    against the PRODUCTION-faithful backtest (production cohort + α̂ → the live
    solver) and persist the verdict to
    ``s3://{bucket}/backtest/{run_date}/optimizer_param_sweep.json`` (config#1057
    increment 1).

    Observe-only + non-fatal, like the cov-/γ-sweeps — this is the recommendation
    surface; the auto-apply to live optimizer config (a new
    ``config/portfolio_optimizer.json`` key + executor-side merge, behind a
    holdout gate) is increment 2. Skips cleanly when the production
    predictions/signals archive isn't available yet."""
    import os

    import boto3

    from analysis.portfolio_optimizer_backtest import run_optimizer_param_sweep
    from synthetic.production_signal_backtest import build_production_signal_inputs

    logger.info("optimizer_param_sweep: starting")

    executor_paths = config.get("executor_paths", [])
    if isinstance(executor_paths, str):
        executor_paths = [executor_paths]
    executor_path = next((p for p in executor_paths if os.path.isdir(p)), None)
    if not executor_path:
        raise ValueError(
            f"executor_paths not found on disk: {executor_paths}. "
            "Add the alpha-engine repo root to executor_paths in config.yaml."
        )

    inputs = build_production_signal_inputs(config, s3_client=s3_client)
    if inputs.get("status") != "ok":
        logger.warning(
            "optimizer_param_sweep: production inputs status=%s — sweep skipped",
            inputs.get("status"),
        )
        return {
            "run_date": run_date,
            "status": "skipped",
            "reason": f"production inputs status={inputs.get('status')!r}",
        }

    sweep_report = run_optimizer_param_sweep(
        predictions_by_date=inputs["predictions_by_date"],
        price_matrix=inputs["price_matrix"],
        spy_prices=inputs["spy_prices"],
        sector_map=inputs["sector_map"],
        executor_path=executor_path,
    )

    bucket = config.get("signals_bucket", "alpha-engine-research")
    key = f"backtest/{run_date}/optimizer_param_sweep.json"
    payload = {
        "run_date": run_date,
        "status": "ok",
        "production_window": inputs.get("production_window"),
        "n_production_dates": inputs.get("n_production_dates"),
        **sweep_report,
    }
    body = json.dumps(payload, default=str, indent=2).encode("utf-8")

    s3 = s3_client or boto3.client("s3")
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
        logger.info("optimizer_param_sweep: persisted s3://%s/%s", bucket, key)
    except Exception as exc:
        logger.warning("optimizer_param_sweep: S3 persist failed (non-fatal): %s", exc)

    logger.info(
        "optimizer_param_sweep: baseline=%s winner=%s ranking_top=%s",
        sweep_report.get("baseline_name"),
        sweep_report.get("winner_name"),
        (sweep_report.get("ranking") or [(None, None)])[0],
    )
    return payload


def run_predictor_backtest(config: dict) -> dict:
    """
    Run predictor-only historical backtest: generate synthetic signals from
    GBM inference on 2y of slim cache data, then simulate through the full
    executor pipeline (risk guard, position sizing, ATR stops, time decay,
    graduated drawdown).

    Returns a stats dict with portfolio metrics + metadata, or a status dict
    if insufficient data.
    """
    import sys
    from synthetic.predictor_backtest import run as run_predictor_pipeline

    # Prepare data: load cache, compute features, run GBM, generate signals.
    # keep_predictions=True retains the (cheap) continuous alpha panel + ADV so
    # the W3.4 horizon net-alpha study below reuses them — no second 10y
    # inference.
    result = run_predictor_pipeline(config, keep_predictions=True)

    if result.get("status") != "ok":
        return result

    signals_by_date = result["signals_by_date"]
    price_matrix = result["price_matrix"]
    ohlcv_by_ticker = result["ohlcv_by_ticker"]
    spy_prices = result.get("spy_prices")
    metadata = result["metadata"]

    # Import executor modules
    executor_paths = config.get("executor_paths", [])
    if isinstance(executor_paths, str):
        executor_paths = [executor_paths]
    executor_path = next((p for p in executor_paths if os.path.isdir(p)), None)
    if not executor_path:
        return {"status": "error", "error": f"executor_paths not found: {executor_paths}"}
    if executor_path not in sys.path:
        sys.path.insert(0, executor_path)

    from executor.main import run as executor_run
    from executor.ibkr import SimulatedIBKRClient

    # Run simulation
    logger.info("Running predictor-only simulation: %d dates", len(signals_by_date))
    stats = _run_simulation_loop(
        executor_run, SimulatedIBKRClient,
        dates=[],  # not used when signals_by_date is provided
        price_matrix=price_matrix,
        config=config,
        ohlcv_by_ticker=ohlcv_by_ticker,
        signals_by_date=signals_by_date,
        spy_prices=spy_prices,
    )

    # Merge metadata into stats for reporting
    stats["predictor_metadata"] = metadata

    # ── W3.4 (L4469, OBSERVE): turnover-adjusted net alpha per horizon ───────
    # Reuses the predictions/prices/ADV already computed above (no second 10y
    # inference). NET-of-cost is the horizon-cutover judge; gross IC (the
    # predictor manifest's leak-free curve) is not. Gates nothing; the canonical
    # 21d target is unchanged. Self-contained S3 write mirrors the cov/gamma
    # sweep stages.
    try:
        from analysis.horizon_net_alpha import compute_horizon_net_alpha
        from analysis.transaction_cost import TransactionCostModel

        hna_cfg = config.get("horizon_net_alpha", {}) or {}
        if hna_cfg.get("enabled", True) and result.get("predictions_by_date"):
            pb_cfg = config.get("predictor_backtest", {}) or {}
            hna = compute_horizon_net_alpha(
                result["predictions_by_date"],
                result["price_matrix"],
                result.get("spy_prices"),
                cost_model=TransactionCostModel.from_config(config),
                horizons=hna_cfg.get("horizons"),
                top_n=int(hna_cfg.get("top_n", pb_cfg.get("top_n_signals_per_day", 20))),
                init_cash=float(config.get("init_cash", 1_000_000.0)),
                adv_dollar_by_ticker=result.get("adv_dollar_by_ticker"),
            )
            stats["horizon_net_alpha"] = hna
            _run_date = config.get("_run_date")
            bucket = config.get("signals_bucket", "alpha-engine-research")
            if _run_date:
                import boto3
                s3 = boto3.client("s3")
                key = f"backtest/{_run_date}/horizon_net_alpha.json"
                body = json.dumps({"run_date": _run_date, **hna}, default=str, indent=2).encode("utf-8")
                try:
                    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
                    logger.info("horizon_net_alpha: persisted s3://%s/%s", bucket, key)
                except Exception as exc:
                    logger.warning("horizon_net_alpha S3 persist failed (non-fatal): %s", exc)
    except Exception as exc:
        logger.warning("horizon_net_alpha stage failed (OBSERVE, non-fatal): %s", exc)

    # ── L4488b (OBSERVE): per-model-version net-of-cost alpha under a FIXED ──
    # policy. Champion/challenger Phase-2 completion: score each registered
    # version's predictions (from research.db predictor_outcomes[_shadow]) through
    # the SAME top-N fixed policy + cost model as horizon_net_alpha, so the
    # leaderboard's decision metric measures the MODEL, not model×execution.
    # Reuses the prices/ADV already loaded above. Gates nothing.
    try:
        from analysis.model_version_net_alpha import (
            compute_model_version_net_alpha,
            load_version_predictions,
        )
        from analysis.transaction_cost import TransactionCostModel

        mvna_cfg = config.get("model_version_net_alpha", {}) or {}
        version_preds = load_version_predictions(config.get("research_db"))
        if mvna_cfg.get("enabled", True) and version_preds and result.get("price_matrix") is not None:
            pb_cfg = config.get("predictor_backtest", {}) or {}
            mvna = compute_model_version_net_alpha(
                version_preds,
                result["price_matrix"],
                result.get("spy_prices"),
                cost_model=TransactionCostModel.from_config(config),
                horizon=int(mvna_cfg.get("horizon", pb_cfg.get("forward_days", 21))),
                top_n=int(mvna_cfg.get("top_n", pb_cfg.get("top_n_signals_per_day", 20))),
                init_cash=float(config.get("init_cash", 1_000_000.0)),
                adv_dollar_by_ticker=result.get("adv_dollar_by_ticker"),
            )
            stats["model_version_net_alpha"] = mvna
            _run_date = config.get("_run_date")
            bucket = config.get("signals_bucket", "alpha-engine-research")
            if _run_date:
                import boto3
                s3 = boto3.client("s3")
                key = f"backtest/{_run_date}/model_version_net_alpha.json"
                body = json.dumps({"run_date": _run_date, **mvna}, default=str, indent=2).encode("utf-8")
                try:
                    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
                    logger.info("model_version_net_alpha: persisted s3://%s/%s", bucket, key)
                except Exception as exc:
                    logger.warning("model_version_net_alpha S3 persist failed (non-fatal): %s", exc)
    except Exception as exc:
        logger.warning("model_version_net_alpha stage failed (OBSERVE, non-fatal): %s", exc)

    return stats


def _run_vectorized_param_sweep(
    *,
    grid: dict,
    base_config: dict,
    sweep_settings: dict | None,
    price_matrix,
    ohlcv_by_ticker: dict | None,
    signal_lookups: dict,
    feature_lookup,
    spy_prices,
    sector_map: dict,
    atr_pct_by_ticker: dict | None = None,
    coverage_by_ticker: dict | None = None,
    predictions_by_date: dict | None = None,
) -> pd.DataFrame:
    """Vectorized batch dispatch for ``run_predictor_param_sweep``.

    Builds the same combinations ``param_sweep.sweep`` would, runs
    them all in parallel via ``synthetic.vectorized_sweep.run_vectorized_sweep``,
    then computes per-combo portfolio stats by piping each combo's
    order list through ``vectorbt_bridge.orders_to_portfolio``.

    Returns a DataFrame with the same shape as the scalar sweep output:
    one row per combo, sorted by ``total_alpha`` (or ``sharpe_ratio``
    fallback), with sweep metadata in ``df.attrs``.

    Tier 4 PR 5 wire-in (2026-04-27). Gated via
    ``config["use_vectorized_sweep"]`` — DEFAULT ON since 2026-04-28
    after v18 validated stats parity (post fee-rate alignment) +
    cap retighten 5400→1800. Pass ``--use-scalar-sweep`` (or set
    ``use_vectorized_sweep: false`` in config.yaml) for explicit
    opt-out / emergency rollback.

    Known parity caveats (documented for v14 validation):
      * ``market_regime`` is read from the first signal_lookup with a
        ``market_regime`` key and applied as a constant across the
        sweep window. The scalar path reads it per-date. Affects bear
        regime gates if the regime changes mid-window — vanishingly
        rare in practice (sweep windows are 10y, regime classification
        is monthly cadence).
      * Time-decay days-held arithmetic uses ``date_idx - entry_dates``
        on the simulator's date axis (per-signal-date trading days).
        Scalar uses ``_approx_trading_days`` from ISO date strings,
        weekday-walking. Equivalent when the sweep date axis is
        trading days only (the standard predictor_param_sweep config).
    """
    import itertools
    from analysis import param_sweep
    from synthetic.vectorized_sweep import run_vectorized_sweep
    from vectorbt_bridge import orders_to_portfolio
    from vectorbt_bridge import portfolio_stats as compute_pf_stats

    settings = sweep_settings or {}
    mode = settings.get("mode", "random")
    seed = settings.get("seed")

    keys = list(grid.keys())
    values = list(grid.values())
    total_grid = 1
    for v in values:
        total_grid *= len(v)

    if mode == "random":
        explicit_max = settings.get("max_trials")
        if explicit_max is not None:
            n = int(explicit_max)
        else:
            n = param_sweep.auto_n_trials(
                total_grid,
                trial_pct=settings.get("trial_pct"),
                min_trials=settings.get("min_trials"),
                max_trials=settings.get("max_trials_cap"),
            )
        combinations = param_sweep._generate_random_combos(grid, n, seed=seed)
        actual_mode = "random" if len(combinations) < total_grid else "grid (auto-fallback)"
        coverage = len(combinations) / total_grid
    else:
        combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        actual_mode = "grid"
        coverage = 1.0

    logger.info(
        "Vectorized param sweep: %d combos × %d dates × %d tickers (mode=%s)",
        len(combinations), len(price_matrix), len(price_matrix.columns), actual_mode,
    )

    # Build per-combo merged configs by repeating the same merge
    # _run_simulation_loop / executor.run() does (deepcopy base, update
    # with combo overrides, push flat sweep keys into nested strategy
    # paths via _PARAM_MAP).
    merged_combo_configs: list = []
    for params in combinations:
        combo_config = param_sweep._deepcopy_safe_config(base_config)
        combo_config.update(params)
        merged, _ = _build_merged_simulate_config(combo_config)
        merged_combo_configs.append(merged)

    # Read market_regime from first available signal_lookup (assumes
    # constant regime across sweep window — see docstring caveat).
    market_regime = "neutral"
    for sl in signal_lookups.values():
        regime_str = (
            sl.signals_raw_filtered.get("market_regime")
            if hasattr(sl, "signals_raw_filtered") else None
        )
        if regime_str:
            market_regime = regime_str
            break

    init_cash = float(base_config.get("init_cash", 1_000_000.0))
    # Per-side fee fraction — same key the scalar path reads at
    # backtest.py:2171 prior to vectorbt dispatch. Default 0.001 (10 bps)
    # matches scalar `vectorbt_bridge.orders_to_portfolio` and closes
    # the v17 absolute-stats gap (vectorized fee-free → vectorbt
    # fee-aware divergence: ~50% on a $1M / 9k-order config). Relative
    # ranking was unaffected pre-fix; absolute alpha numbers are now
    # comparable to scalar.
    fee_rate = float(base_config.get("simulation_fees", 0.001))

    orders_per_combo, diagnostics = run_vectorized_sweep(
        combo_configs=merged_combo_configs,
        price_matrix=price_matrix,
        ohlcv_by_ticker=ohlcv_by_ticker or {},
        signal_lookups=signal_lookups,
        feature_lookup=feature_lookup,
        spy_prices=spy_prices,
        sector_map=sector_map,
        atr_pct_by_ticker=atr_pct_by_ticker,
        coverage_by_ticker=coverage_by_ticker,
        predictions_by_date=predictions_by_date,
        init_cash=init_cash,
        market_regime=market_regime,
        fee_rate=fee_rate,
    )

    logger.info(
        "Vectorized sweep loop: %d combos × %d dates in %.1fs "
        "(entries=%d, exits=%d)",
        diagnostics["n_combos"], diagnostics["n_dates"],
        diagnostics["walltime_sec"],
        diagnostics["entries_applied"], diagnostics["exits_applied"],
    )

    # Per-combo stats — vectorized numpy ops over the [n_combos, n_dates]
    # NAV trajectory the sweep loop tracked. Replaces the prior per-combo
    # `vectorbt.Portfolio.from_orders` + `compute_pf_stats` chain that
    # hung the v16 (2026-04-28) Layer 3 dispatch (60 combos × 26k orders
    # each × 6 vectorbt stat calls = >90 min, watchdog tripped).
    # See `synthetic.vectorized_stats` docstring for the fee-parity
    # caveat (vectorized sim is fee-free; ~0.9% NAV offset vs scalar's
    # vectorbt-with-fees path; relative ranking unaffected).
    from synthetic.vectorized_stats import compute_vectorized_stats
    nav_history = diagnostics["nav_history"]
    t_stats = _time.monotonic()
    df = compute_vectorized_stats(
        nav_history=nav_history,
        init_cash=init_cash,
        spy_prices=spy_prices,
        dates=price_matrix.index,
        orders_per_combo=orders_per_combo,
        combo_params=combinations,
    )
    logger.info(
        "Vectorized sweep stats: %d combos in %.2fs",
        len(df), _time.monotonic() - t_stats,
    )
    # Sort by sortino_ratio (primary, skilled-risk basket) → total_alpha
    # (tiebreaker/presentation only). Raw Sharpe is observability only and is
    # intentionally absent — see [[anchor_gates_on_skilled_risk_not_sharpe]] +
    # analysis/param_sweep._sort_sweep_df_skilled_risk for the shared helper.
    from analysis.param_sweep import _sort_sweep_df_skilled_risk
    _sort_sweep_df_skilled_risk(df)

    if not df.empty:
        df.attrs["sweep_mode"] = f"{actual_mode} (vectorized)"
        df.attrs["sweep_total_grid"] = total_grid
        df.attrs["sweep_trials"] = len(combinations)
        df.attrs["sweep_coverage"] = coverage
        df.attrs["vectorized_walltime_sec"] = diagnostics["walltime_sec"]
        df.attrs["vectorized_entries"] = diagnostics["entries_applied"]
        df.attrs["vectorized_exits"] = diagnostics["exits_applied"]

    return df


def run_predictor_param_sweep(config: dict) -> tuple[dict, pd.DataFrame]:
    """
    Run predictor-only backtest with param sweep.

    Loads data once (features, GBM inference, signal generation), then runs
    the simulation loop for each parameter combination. Also runs Phase 4
    evaluations (ensemble mode, feature pruning) if features are available.

    Returns (single_run_stats, sweep_df).
    """
    import sys
    from synthetic.predictor_backtest import run as run_predictor_pipeline

    registry = config["_phase_registry"]
    bucket = config.get("signals_bucket", "alpha-engine-research")
    s3 = registry.s3_client

    # Prepare data once. The persist_features_callback hook persists
    # features inside run() right after inference and BEFORE
    # build_signals_by_date — so features (~1.1 GB) and signals (~700
    # MB-1 GB) never coexist in memory. This closes the
    # post_build_signals=2768 MB OOM diagnosed 2026-04-26. Phase 4a/4c
    # lazy-reload via _load_features_by_ticker_only when their
    # evaluator runs. See Stage 4 of the c5.large optimization arc.
    with registry.phase(
        "predictor_data_prep", supports_auto_skip=True,
    ) as ctx:
        if ctx.skipped:
            result = _load_predictor_data_prep(bucket, registry)
        else:
            from phase_artifacts import save_dict_of_dataframes

            def _persist_features_to_s3(features: dict) -> None:
                """In-run persistence callback — saves features to S3 +
                registers the artifact key on the predictor_data_prep
                marker. Closure over ``ctx`` / ``bucket`` / ``s3``.
                """
                key = save_dict_of_dataframes(
                    bucket, registry.date, "predictor_data_prep",
                    "features_by_ticker", features, s3_client=s3,
                )
                ctx.record_artifact(key)

            result = run_predictor_pipeline(
                config, persist_features_callback=_persist_features_to_s3,
            )
            # ``features_by_ticker`` is already absent from result —
            # run() persisted + dropped via the callback. _save_predictor_
            # data_prep's features-save branch is a no-op here.
            _save_predictor_data_prep(
                ctx, bucket, registry.date, result, s3_client=s3,
            )

    if result.get("status") != "ok":
        return result, pd.DataFrame()

    signals_by_date = result["signals_by_date"]
    price_matrix = result["price_matrix"]
    ohlcv_by_ticker = result["ohlcv_by_ticker"]
    spy_prices = result.get("spy_prices")
    metadata = result["metadata"]
    sector_map = result.get("sector_map", {})
    trading_dates = result.get("trading_dates", [])

    # Build the EW-high-vol basket once for the predictor sweep — shared
    # by the default-config single_run AND every per-combo sim. Basket
    # depends only on prices so it's combo-invariant. None on insufficient
    # history; portfolio_stats falls back to SPY-only alpha computation.
    ew_basket = _try_construct_ew_high_vol_basket(price_matrix)

    # One-time-share logic for ATR + VWAP precomputed maps. The
    # predictor-param-sweep is the bottleneck the Saturday SF dry-run
    # timed out on: 60 combos × 2000+ dates × per-ticker ArcticDB reads.
    # Loading once up front collapses that to a single bulk scan (~1-2
    # min for ~900 tickers with 20-way concurrency) — every subsequent
    # _simulate_single_date call reuses the in-memory maps.
    with registry.phase(
        "predictor_feature_maps_bulk_load", supports_auto_skip=True,
    ) as ctx:
        if ctx.skipped:
            atr_by_ticker, vwap_series_by_ticker, coverage_by_ticker = (
                _load_predictor_feature_maps(bucket, registry)
            )
        else:
            from store.feature_maps import load_precomputed_feature_maps
            _smoke_tickers = config.get("smoke_tickers")
            _allowlist = set(_smoke_tickers) if _smoke_tickers else None
            atr_by_ticker, vwap_series_by_ticker, coverage_by_ticker = (
                load_precomputed_feature_maps(bucket, tickers_allowlist=_allowlist)
            )
            _save_predictor_feature_maps(
                ctx, bucket, registry.date,
                atr_by_ticker, vwap_series_by_ticker, coverage_by_ticker,
                s3_client=s3,
            )

    # Import executor modules
    executor_paths = config.get("executor_paths", [])
    if isinstance(executor_paths, str):
        executor_paths = [executor_paths]
    executor_path = next((p for p in executor_paths if os.path.isdir(p)), None)
    if not executor_path:
        return {"status": "error", "error": f"executor_paths not found: {executor_paths}"}, pd.DataFrame()
    if executor_path not in sys.path:
        sys.path.insert(0, executor_path)

    from executor.main import run as executor_run
    from executor.ibkr import SimulatedIBKRClient

    # Single run with default config
    from phase_artifacts import save_json as _save_json_p, load_json as _load_json_p
    logger.info("Running predictor-only simulation (default params): %d dates", len(signals_by_date))
    with registry.phase(
        "predictor_single_run", n_dates=len(signals_by_date),
        supports_auto_skip=True,
    ) as ctx:
        if ctx.skipped:
            marker = registry.load_marker("predictor_single_run") or {}
            keys = marker.get("artifact_keys") or []
            if not keys:
                raise RuntimeError(
                    "predictor_single_run auto-skip: marker missing artifact_keys"
                )
            single_stats = _load_json_p(bucket, keys[0], s3_client=s3)
        else:
            single_stats = _run_simulation_loop(
                executor_run, SimulatedIBKRClient,
                dates=[],
                price_matrix=price_matrix,
                config=config,
                ohlcv_by_ticker=ohlcv_by_ticker,
                signals_by_date=signals_by_date,
                spy_prices=spy_prices,
                ew_high_vol_basket_returns=ew_basket,
                atr_by_ticker=atr_by_ticker,
                vwap_series_by_ticker=vwap_series_by_ticker,
                coverage_by_ticker=coverage_by_ticker,
            )
            ctx.record_artifact(_save_json_p(
                bucket, registry.date, "predictor_single_run",
                "single_stats", single_stats, s3_client=s3,
            ))
    single_stats["predictor_metadata"] = metadata

    # ── Phase 4: Predictor hyperparameter feedback ───────────────────────
    # `skip_phase4_evaluations` (config flag, set by --skip-phase4-evaluations
    # CLI or SF input): bypass the three Phase 4 evaluators wholesale. Each
    # runs a full silent simulation internally and can add tens of minutes
    # to the predictor pipeline. For dry-runs where we only care "does the
    # pipeline complete end-to-end", skipping is cheap and safe — the S3
    # config promotions will have nothing to apply, so the next real run
    # picks up the existing configs unchanged.
    predictions_by_date = result.get("predictions_by_date", {})
    if config.get("skip_phase4_evaluations"):
        logger.info(
            "Phase 4 predictor-hyperparameter feedback SKIPPED "
            "(skip_phase4_evaluations=true). Ensemble mode / signal "
            "threshold / feature pruning evaluators will not run."
        )
    elif trading_dates:
        try:
            from optimizer.predictor_optimizer import (
                evaluate_ensemble_modes,
                evaluate_signal_thresholds,
                evaluate_feature_pruning,
                apply_recommendations,
            )
            bucket = config.get("signals_bucket", "alpha-engine-research")
            import gc

            # Phase 4a: Ensemble mode evaluation. Lazy-loads features
            # (~1.1 GB) only inside this phase block, frees on exit.
            ensemble_result = None
            try:
                with registry.phase(
                    "phase4a_ensemble_modes", supports_auto_skip=True,
                ) as p4a_ctx:
                    if p4a_ctx.skipped:
                        marker = registry.load_marker("phase4a_ensemble_modes") or {}
                        keys = marker.get("artifact_keys") or []
                        ensemble_result = _load_json_p(bucket, keys[0], s3_client=s3) if keys else None
                    else:
                        features_by_ticker = _load_features_by_ticker_only(bucket, registry)
                        try:
                            ensemble_result = evaluate_ensemble_modes(
                                features_by_ticker, price_matrix, ohlcv_by_ticker,
                                spy_prices, sector_map, trading_dates,
                                config, single_stats,
                            )
                        finally:
                            del features_by_ticker
                            gc.collect()
                        if ensemble_result is not None:
                            p4a_ctx.record_artifact(_save_json_p(
                                bucket, registry.date, "phase4a_ensemble_modes",
                                "ensemble_result", ensemble_result, s3_client=s3,
                            ))
                if ensemble_result is not None:
                    single_stats["ensemble_eval"] = ensemble_result
            except Exception as exc:
                # Bumped from warning to error so flow-doctor captures it.
                # Previously logged at warning and the spot run stayed
                # green even when the optimizer couldn't evaluate ensemble
                # mode, which meant param recommendations were based on
                # partial sweep data.
                logger.error(
                    "Phase 4a ensemble mode evaluation failed: %s — "
                    "optimizer recommendations may be incomplete",
                    exc, exc_info=True,
                )

            # Phase 4b: Signal threshold sweep — does NOT consume features,
            # so no lazy-load needed here. The evaluator itself iterates
            # several thresholds; ``evaluate_signal_thresholds`` handles
            # per-iteration cleanup via its own gc.collect() in the loop's
            # finally:.
            threshold_result = None
            if predictions_by_date:
                try:
                    with registry.phase(
                        "phase4b_signal_thresholds", supports_auto_skip=True,
                    ) as p4b_ctx:
                        if p4b_ctx.skipped:
                            marker = registry.load_marker("phase4b_signal_thresholds") or {}
                            keys = marker.get("artifact_keys") or []
                            threshold_result = _load_json_p(bucket, keys[0], s3_client=s3) if keys else None
                        else:
                            threshold_result = evaluate_signal_thresholds(
                                predictions_by_date, sector_map, ohlcv_by_ticker,
                                price_matrix, spy_prices, trading_dates,
                                config, single_stats,
                            )
                            if threshold_result is not None:
                                p4b_ctx.record_artifact(_save_json_p(
                                    bucket, registry.date, "phase4b_signal_thresholds",
                                    "threshold_result", threshold_result, s3_client=s3,
                                ))
                    if threshold_result is not None:
                        single_stats["threshold_eval"] = threshold_result
                except Exception as exc:
                    logger.error(
                        "Phase 4b signal threshold evaluation failed: %s",
                        exc, exc_info=True,
                    )
                finally:
                    # Phase 4b's loop did per-iteration cleanup; force
                    # a top-level collect too so any inter-iteration
                    # allocations are reclaimed before Phase 4c starts
                    # (which lazy-loads features again, ~1.1 GB).
                    gc.collect()

            # Phase 4c: Feature pruning evaluation. Lazy-loads features
            # again (Phase 4a already freed them), frees on exit.
            pruning_result = None
            try:
                with registry.phase(
                    "phase4c_feature_pruning", supports_auto_skip=True,
                ) as p4c_ctx:
                    if p4c_ctx.skipped:
                        marker = registry.load_marker("phase4c_feature_pruning") or {}
                        keys = marker.get("artifact_keys") or []
                        pruning_result = _load_json_p(bucket, keys[0], s3_client=s3) if keys else None
                    else:
                        features_by_ticker = _load_features_by_ticker_only(bucket, registry)
                        try:
                            pruning_result = evaluate_feature_pruning(
                                features_by_ticker, price_matrix, ohlcv_by_ticker,
                                spy_prices, sector_map, trading_dates,
                                config, single_stats,
                            )
                        finally:
                            del features_by_ticker
                            gc.collect()
                        if pruning_result is not None:
                            p4c_ctx.record_artifact(_save_json_p(
                                bucket, registry.date, "phase4c_feature_pruning",
                                "pruning_result", pruning_result, s3_client=s3,
                            ))
                if pruning_result is not None:
                    single_stats["pruning_eval"] = pruning_result
            except Exception as exc:
                logger.error(
                    "Phase 4c feature pruning evaluation failed: %s",
                    exc, exc_info=True,
                )

            # Apply recommendations to S3 (if any)
            try:
                apply_result = apply_recommendations(
                    ensemble_result, pruning_result, bucket,
                    threshold_result=threshold_result,
                )
                single_stats["predictor_optimizer_apply"] = apply_result
            except Exception as exc:
                # This one is especially important — if the apply fails,
                # the optimizer's recommendations don't get persisted to
                # S3, so the predictor keeps running on stale params.
                logger.error(
                    "Predictor optimizer apply failed (recommendations "
                    "not persisted to S3): %s", exc, exc_info=True,
                )

        except ImportError as exc:
            logger.error(
                "Phase 4 optimizer not available (import failed): %s",
                exc, exc_info=True,
            )

    # Param sweep — seed grid with current S3 params for iterative learning
    sweep_df = pd.DataFrame()
    grid = config.get("param_sweep")
    if grid:
        bucket = config.get("signals_bucket", "alpha-engine-research")
        grid = _seed_grid_with_current(
            grid,
            read_params_pit_or_current(executor_optimizer, bucket, config),
        )

        # Tier 3 Part A (2026-04-27): precompute SignalLookup ONCE here,
        # before the sim_fn closure is constructed. All 60 combos in
        # the sweep capture the same precomputed lookups via closure.
        # Without this hoist, each combo's _run_simulation_loop call
        # would rebuild signals_by_ticker + universe_sectors + filter
        # ~900 universe entries × 2316 dates from scratch — ~22 min of
        # redundant compute across the sweep. Universe filtering
        # happens here too: the filter walks the same lists, so fold
        # it into the per-date precompute and skip the per-call
        # _filter_signals_to_universe inside _simulate_single_date.
        try:
            from alpha_engine_lib.arcticdb import get_universe_symbols
            _universe_symbols_for_sweep = get_universe_symbols(bucket)
        except Exception as exc:
            raise RuntimeError(
                f"predictor_param_sweep universe-filter bootstrap failed: "
                f"could not read ArcticDB universe symbols from bucket "
                f"{bucket!r}: {exc}"
            ) from exc
        _sweep_rejected_counter: dict[str, int] = {}
        signal_lookups = _precompute_signal_lookups(
            signals_by_date,
            _universe_symbols_for_sweep,
            _sweep_rejected_counter,
        )
        if _sweep_rejected_counter:
            top = sorted(_sweep_rejected_counter.items(), key=lambda kv: -kv[1])
            total = sum(_sweep_rejected_counter.values())
            logger.warning(
                "predictor_param_sweep precompute: filtered %d signal "
                "entries across %d tickers absent from current ArcticDB "
                "universe. Top: %s",
                total, len(_sweep_rejected_counter),
                [f"{t}={n}" for t, n in top[:10]],
            )

        # Tier 3 Part C (2026-04-27): build FeatureLookup ONCE here so
        # all 60 combos share the precomputed ATR / RSI / momentum /
        # returns / support series. Without this hoist, each combo's
        # _run_simulation_loop call would re-vectorize the bulk
        # precompute (~5-30 sec), wasting 60× the cost. With this
        # hoist: ~5-30 sec ONCE, all combos lookup-hit.
        try:
            from executor.feature_lookup import FeatureLookup
            _t0 = _time.monotonic()
            sweep_feature_lookup = FeatureLookup.from_ohlcv_by_ticker(
                ohlcv_by_ticker,
            )
            logger.info(
                "predictor_param_sweep FeatureLookup precompute: "
                "%d tickers, %.1fs (atr_dollar=%d, rsi=%d, "
                "momentum_20d_pct=%d, returns=%d, support_20_low=%d)",
                len(ohlcv_by_ticker),
                _time.monotonic() - _t0,
                len(sweep_feature_lookup.atr_dollar),
                len(sweep_feature_lookup.rsi),
                len(sweep_feature_lookup.momentum_20d_pct),
                len(sweep_feature_lookup.returns),
                len(sweep_feature_lookup.support_20_low),
            )
        except ImportError:
            sweep_feature_lookup = None

        def sim_fn(combo_config: dict) -> dict:
            return _run_simulation_loop(
                executor_run, SimulatedIBKRClient,
                dates=[],
                price_matrix=price_matrix,
                config=combo_config,
                ohlcv_by_ticker=ohlcv_by_ticker,
                signals_by_date=signals_by_date,
                spy_prices=spy_prices,
                ew_high_vol_basket_returns=ew_basket,
                atr_by_ticker=atr_by_ticker,
                vwap_series_by_ticker=vwap_series_by_ticker,
                coverage_by_ticker=coverage_by_ticker,
                signal_lookups=signal_lookups,
                feature_lookup=sweep_feature_lookup,
            )

        sweep_settings = config.get("param_sweep_settings", {})

        logger.info("Running predictor param sweep (%s): %s", sweep_settings.get("mode", "random"), {k: len(v) for k, v in grid.items()})
        from phase_artifacts import save_dataframe as _save_df_p, load_dataframe as _load_df_p
        with registry.phase(
            "predictor_param_sweep",
            combos=sum(len(v) for v in grid.values()),
            supports_auto_skip=True,
        ) as ps_ctx:
            if ps_ctx.skipped:
                marker = registry.load_marker("predictor_param_sweep") or {}
                keys = marker.get("artifact_keys") or []
                if keys:
                    sweep_df = _load_df_p(bucket, keys[0], s3_client=s3)
                else:
                    sweep_df = pd.DataFrame()
            else:
                # Tier 4 PR 5 (2026-04-27): vectorized sweep behind a config
                # flag. When ``use_vectorized_sweep`` is True (DEFAULT
                # since 2026-04-28 v18 deploy), all combos run in parallel
                # as a numpy axis via run_vectorized_sweep. Set to False
                # via `--use-scalar-sweep` CLI or `use_vectorized_sweep:
                # false` in config.yaml for emergency rollback.
                if config.get("use_vectorized_sweep"):
                    sweep_df = _run_vectorized_param_sweep(
                        grid=grid,
                        base_config=config,
                        sweep_settings=sweep_settings,
                        price_matrix=price_matrix,
                        ohlcv_by_ticker=ohlcv_by_ticker,
                        signal_lookups=signal_lookups,
                        feature_lookup=sweep_feature_lookup,
                        spy_prices=spy_prices,
                        sector_map=sector_map,
                        atr_pct_by_ticker=atr_by_ticker,
                        coverage_by_ticker=coverage_by_ticker,
                        predictions_by_date=predictions_by_date,
                    )
                else:
                    sweep_df = param_sweep.sweep(grid, sim_fn, config, sweep_settings=sweep_settings)
                if sweep_df is not None and not sweep_df.empty:
                    ps_ctx.record_artifact(_save_df_p(
                        bucket, registry.date, "predictor_param_sweep",
                        "sweep_df", sweep_df, preserve_index=False, s3_client=s3,
                    ))

    return single_stats, sweep_df


# ── Infrastructure helpers ──────────────────────────────────────────────────


def _stop_ec2_instance() -> None:
    """Stop the current EC2 instance via metadata endpoint. Best-effort."""
    import urllib.request
    try:
        token = urllib.request.urlopen(
            urllib.request.Request(
                "http://169.254.169.254/latest/api/token",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
                method="PUT",
            ),
            timeout=5,
        ).read().decode()
        instance_id = urllib.request.urlopen(
            urllib.request.Request(
                "http://169.254.169.254/latest/meta-data/instance-id",
                headers={"X-aws-ec2-metadata-token": token},
            ),
            timeout=5,
        ).read().decode()
        logger.info("Stopping instance %s", instance_id)
        boto3.client("ec2").stop_instances(InstanceIds=[instance_id])
    except Exception as e:
        logger.error("Failed to stop instance: %s", e)


# ── Runtime smoke test ──────────────────────────────────────────────────────


_SMOKE_SAMPLE_TICKERS = ("AAPL", "MSFT", "NVDA", "JNJ", "PG")


# ── Per-phase smoke harness ──────────────────────────────────────────────────
#
# ROADMAP Backtester P0 #3. Each `smoke-<phase>` mode:
#   1. applies a tiny-fixture config override (few dates, tiny param grid, short
#      predictor-backtest lookback),
#   2. routes to the equivalent full mode (simulate / param-sweep /
#      predictor-backtest / all),
#   3. optionally restricts to a phase subset via --only-phases (smoke-phase4),
#   4. is wrapped in a wall-clock budget check loaded from timing_budget.yaml.
#
# The fixture overrides leverage EXISTING config knobs wherever possible so the
# harness doesn't change production data-flow code. The only new knob added in
# this PR is `max_signal_dates` (slice cap on the list returned by
# signal_loader.list_dates) — used by _setup_simulation.
#
# Not implemented here: universe-size limit. Smoke still runs against the full
# ArcticDB universe; speed comes from capping dates + combos. A future PR could
# add a ticker filter if per-smoke runtime proves too long on the spot instance.

# Default smoke-fixture ticker allowlist — restricts ArcticDB bulk reads
# to a handful of high-liquidity large-caps. Callers can override per-mode
# by passing a different list into the fixture `smoke_tickers` config key.
# These are the same tickers the existing _runtime_smoke uses as sample
# probes — kept in sync so smoke paths consistently exercise the same
# ticker slice end-to-end.
_SMOKE_FIXTURE_TICKERS = list(_SMOKE_SAMPLE_TICKERS)


# Mapping: smoke mode → (full mode it routes to, config overrides, optional
# --only-phases restriction, optional --skip-phases restriction).
#
# Every fixture sets `smoke_tickers` — a ticker allowlist that propagates
# into loaders/signal_loader + loaders/price_loader.build_matrix +
# store.feature_maps.load_precomputed_feature_maps +
# store.arctic_reader.load_universe_from_arctic, restricting the ArcticDB
# bulk read to ~5 tickers instead of the full ~900-ticker universe. This
# is the dominant speedup lever for smoke: the 2026-04-23 smoke-only
# dry-run revealed that max_signal_dates=5 alone saved very little
# because setup still paid ~380s of full-universe bulk-read cost.
_SMOKE_PHASE_MODES: dict[str, dict] = {
    "smoke-simulate": {
        "route_mode": "simulate",
        "overrides": {
            "max_signal_dates": 5,
            "min_simulation_dates": 2,
            "smoke_tickers": _SMOKE_FIXTURE_TICKERS,
        },
        "only_phases": None,
        "skip_phases": None,
    },
    "smoke-param-sweep": {
        "route_mode": "param-sweep",
        "overrides": {
            "max_signal_dates": 5,
            "min_simulation_dates": 2,
            "smoke_tickers": _SMOKE_FIXTURE_TICKERS,
            # Grid override attempts to narrow the sweep to 3 combos.
            # Note: _apply_smoke_fixture uses _deep_update which MERGES
            # nested dicts — so if config.yaml has its own `param_sweep`
            # block with all 7 risk params × multiple values, our
            # 1-key override just replaces max_positions and the other
            # 6 params stay in the grid, ballooning to 864 combos
            # (observed on 2026-04-23 post-bugfix smoke run).
            #
            # Fix: force mode=random with max_trials=3. Regardless of
            # whether the effective grid ends up with 3 or 864 shapes,
            # _generate_random_combos samples exactly 3 combinations,
            # capping smoke-param-sweep runtime to a predictable budget.
            # Validates the sweep plumbing end-to-end (param_sweep.sweep
            # → _run_combos → run_simulation_fn → simulate) without
            # paying full grid cost.
            "param_sweep": {"max_positions": [5, 10, 15]},
            "param_sweep_settings": {"mode": "random", "max_trials": 3, "seed": 0},
        },
        "only_phases": None,
        "skip_phases": None,
    },
    "smoke-predictor-backtest": {
        "route_mode": "predictor-backtest",
        "overrides": {
            "smoke_tickers": _SMOKE_FIXTURE_TICKERS,
            # Small GBM lookback — enough bars for features (>252 rolling
            # windows aren't needed for a smoke; ArcticDB's feature columns
            # are precomputed so min_trading_days is the slice cap on
            # trading_dates used by run_inference).
            "predictor_backtest": {
                "min_trading_days": 30,
                "max_trading_days": 60,
                "top_n_signals_per_day": 5,
            },
            # Skip the full predictor sweep — smoke just validates the
            # data_prep → single_run path completes.
            "param_sweep": None,
        },
        # preflight + runtime_smoke are included so they actually run (the
        # whole point of smoke is env validation). predictor_pipeline is
        # the parent that wraps the inner phases; without it the parent
        # would be SKIP-but-body-still-runs and inner phases would be
        # flagged as "only_phases_filter" at the top level — see
        # 2026-04-23 post-filter dry-run log traces for this pattern.
        "only_phases": [
            "preflight",
            "runtime_smoke",
            "predictor_pipeline",
            "predictor_data_prep",
            "predictor_feature_maps_bulk_load",
            "predictor_single_run",
        ],
        "skip_phases": None,
    },
    "smoke-phase4": {
        "route_mode": "predictor-backtest",
        "overrides": {
            "smoke_tickers": _SMOKE_FIXTURE_TICKERS,
            "predictor_backtest": {
                "min_trading_days": 30,
                "max_trading_days": 60,
                "top_n_signals_per_day": 5,
            },
            "param_sweep": None,
        },
        "only_phases": [
            "preflight",
            "runtime_smoke",
            "predictor_pipeline",
            "predictor_data_prep",
            "predictor_feature_maps_bulk_load",
            "predictor_single_run",
            "phase4a_ensemble_modes",
            "phase4b_signal_thresholds",
            "phase4c_feature_pruning",
        ],
        "skip_phases": None,
    },
    "smoke-predictor-param-sweep": {
        # Exercises the predictor_param_sweep phase end-to-end on a tiny
        # 2-combo grid. The only smoke mode that routes through
        # predictor_param_sweep — required to validate the Tier 4
        # vectorized branch (gated on config["use_vectorized_sweep"]).
        # Without this, --use-vectorized-sweep on the smoke path is a
        # no-op because no smoke mode reaches that phase.
        "route_mode": "predictor-backtest",
        "overrides": {
            "smoke_tickers": _SMOKE_FIXTURE_TICKERS,
            "predictor_backtest": {
                "min_trading_days": 30,
                "max_trading_days": 60,
                "top_n_signals_per_day": 5,
            },
            # Single-value lists per param keep the cartesian space at 2
            # (driven by min_score). max_trials=2 caps regardless via
            # _generate_random_combos. Validates plumbing, not coverage.
            "param_sweep": {
                "min_score": [65, 70],
                "max_position_pct": [0.10],
                "atr_multiplier": [2.5],
                "time_decay_reduce_days": [7],
                "time_decay_exit_days": [15],
                "profit_take_pct": [0.20],
            },
            "param_sweep_settings": {
                "mode": "random", "max_trials": 2, "seed": 0,
            },
        },
        "only_phases": [
            "preflight",
            "runtime_smoke",
            "predictor_pipeline",
            "predictor_data_prep",
            "predictor_feature_maps_bulk_load",
            "predictor_single_run",
            "predictor_param_sweep",
        ],
        "skip_phases": None,
    },
}


def _is_smoke_phase_mode(mode: str) -> bool:
    return mode in _SMOKE_PHASE_MODES


def _apply_smoke_fixture(mode: str, args, config: dict) -> None:
    """Apply the config overrides for a smoke-<phase> mode.

    Mutates `config` in place. Also rewrites `args.mode` to the routed
    full mode and sets `args.only_phases` / `args.skip_phases` if the
    smoke mode restricts phase selection.
    """
    spec = _SMOKE_PHASE_MODES[mode]

    def _deep_update(target: dict, overrides: dict) -> None:
        """Recursive merge so nested dicts (e.g. predictor_backtest)
        don't clobber sibling keys the smoke override didn't set."""
        for k, v in overrides.items():
            if (
                isinstance(v, dict)
                and isinstance(target.get(k), dict)
            ):
                _deep_update(target[k], v)
            else:
                target[k] = v

    _deep_update(config, spec["overrides"])

    # Route to the underlying full mode for downstream branching
    # (_run_simulation_pipeline, _run_predictor_pipeline).
    args.mode = spec["route_mode"]

    if spec["only_phases"]:
        # Append to whatever the operator already passed — a CLI-passed
        # --only-phases narrows further, never widens beyond what the
        # smoke mode allows.
        existing = [p.strip() for p in (args.only_phases or "").split(",") if p.strip()]
        combined = existing or spec["only_phases"]
        args.only_phases = ",".join(combined)

    if spec["skip_phases"]:
        existing = [p.strip() for p in (args.skip_phases or "").split(",") if p.strip()]
        combined = existing + spec["skip_phases"]
        args.skip_phases = ",".join(combined)

    # Smoke should always run fresh — auto-skip from a prior run on the
    # same args.date would defeat the purpose of the harness (we want to
    # know the PHASE COMPUTE works, not that the S3 artifact is readable).
    args.force = True

    # Smoke runs never promote to S3 configs — the fixture is synthetic
    # enough that any recommendations would be garbage.
    args.freeze = True

    # Namespace smoke markers + artifacts under a separate S3 prefix so
    # they don't collide with production-run markers on the same calendar
    # date. Observed 2026-04-23 SF dry-run: smoke had left ok markers at
    # backtest/2026-04-23/.phases/ which the full SF run then auto-
    # skipped, replaying tiny 5-ticker smoke artifacts and breaking the
    # downstream parity test. Prefixing with ".smoke/" hierarchically
    # isolates smoke state (backtest/.smoke/2026-04-23/.phases/...) and
    # — critically — lex-sorts BEFORE "2026-..." so spot_backtest.sh's
    # "latest date" probe via `aws s3 ls backtest/ | sort | tail -1`
    # still resolves to real dates. Report filename + local results dir
    # inherit the prefix too but smoke emails are suppressed and smoke
    # uploads are disabled below so no user-visible artifacts appear.
    args.date = f".smoke/{args.date}"

    # Suppress top-level S3 upload. Without this, the export_artifacts
    # phase writes smoke's 5-ticker portfolio_stats.json etc. to
    # backtest/{date}/ top-level, overwriting production run outputs.
    # Namespaced date above handles most of this, but args.upload also
    # drives a `backtest/{date}/` upload in reporter.upload_to_s3 —
    # simpler to just short-circuit the upload for smoke.
    args.upload = False

    logger.info(
        "Smoke fixture applied for mode=%s → routing to --mode=%s "
        "with only_phases=%r skip_phases=%r force=True freeze=True "
        "date=%s upload=False (namespaced to isolate from production runs)",
        mode, args.mode, args.only_phases, args.skip_phases, args.date,
    )


def _apply_dry_run_isolation(args) -> None:
    """Apply the --dry-run safety bundle.

    Mirrors the smoke fixture's isolation pattern but preserves the
    operator's choice of mode + universe size. Intended use: ad-hoc
    validation spot runs that exercise the production pipeline without
    polluting production S3 state (phase markers, artifacts, reports,
    config promotions).

    Bundle:
      - args.date = ".dry-run/{date}/" — markers + artifacts + reports
        all namespace-isolated from the scheduled SF on the same
        calendar date. Mirrors smoke's ".smoke/{date}/" pattern.
      - args.freeze = True — no optimizer S3 config writes
        (scoring_weights.json / executor_params.json / predictor_params.json
        / research_params.json stay untouched).
      - args.upload = False — no reporter upload. The dry-run produces
        local artifacts on the spot; the point of the run is to validate
        behavior, not to publish outputs.
      - args.force = True — auto-skip disabled. A dry-run validates a
        code change; loading a prior run's S3 artifact would defeat
        the purpose.

    Motivation (2026-04-24): the --dry-run ask surfaced while preparing
    an ad-hoc validation run for the backtester silent-phase diagnosis
    arc (PRs #65 + #66 + #67). Operator was concerned that a manual
    spot on the same calendar date as the scheduled Sat SF could
    contaminate phase markers. Smoke mode already solved the isolation
    problem with ".smoke/"; --dry-run gives full-universe runs the
    same treatment.
    """
    args.date = f".dry-run/{args.date}"
    args.freeze = True
    args.upload = False
    args.force = True
    logger.warning(
        "══ DRY RUN MODE ══ mode=%s date=%s "
        "(force=True, freeze=True, upload=False, S3 namespace .dry-run/). "
        "No production configs will be written; phase markers + artifacts "
        "are isolated from scheduled-SF output.",
        args.mode, args.date,
    )


def _load_timing_budgets() -> dict[str, float]:
    """Read timing_budget.yaml from the repo root. Returns empty dict if
    missing — budget enforcement is best-effort, not a hard dependency."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "timing_budget.yaml")
    if not os.path.exists(path):
        logger.warning(
            "timing_budget.yaml not found at %s — smoke budget enforcement disabled",
            path,
        )
        return {}
    try:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return {str(k): float(v) for k, v in data.get("smoke_budgets_seconds", {}).items()}
    except Exception as exc:
        logger.warning("timing_budget.yaml parse failed: %s — budgets disabled", exc)
        return {}


def _assert_smoke_within_budget(
    mode: str, elapsed_s: float, registry=None,
) -> None:
    """Hard-fail (SystemExit 2) if a smoke mode's wall-clock exceeds its
    declared budget OR any inner phase completed with status=error.

    Budget MISSING for a mode → log+skip (best-effort).
    Budget EXCEEDED → fail loud.
    Any inner phase errored → fail loud regardless of wall-clock.
    Clean pass → INFO line for trend monitoring.

    The inner-error check closes the false-PASS gap surfaced by the
    2026-04-23 post-filter dry-run: smoke-param-sweep's outer
    simulation_pipeline phase completed status=ok (try/except swallowed
    the error) while the INNER param_sweep phase errored with
    "maximum recursion depth exceeded". The previous wall-clock-only
    check saw 96s < 500s and reported PASSED — hiding a real failure.
    """
    # Inner-error check first — wall-clock can look fine even when a
    # nested phase errored and the outer swallowed it.
    if registry is not None and registry.phase_errors:
        raise SystemExit(
            f"Smoke [{mode}] FAILED: inner phase(s) completed with "
            f"status=error: {registry.phase_errors}. Wall-clock was "
            f"{elapsed_s:.1f}s (budget check would have passed alone). "
            f"Check the PHASE_END logs for the first error — outer phases "
            f"may have swallowed the exception (e.g. _run_simulation_pipeline "
            f"try/except sets sweep_df=None and continues)."
        )

    budgets = _load_timing_budgets()
    budget = budgets.get(mode)
    if budget is None:
        logger.warning(
            "Smoke [%s] completed in %.1fs — no budget declared in "
            "timing_budget.yaml, consider adding one to catch regressions",
            mode, elapsed_s,
        )
        return
    if elapsed_s > budget:
        raise SystemExit(
            f"Smoke [{mode}] BUDGET EXCEEDED: {elapsed_s:.1f}s > {budget:.1f}s. "
            f"A phase inside this smoke regressed. Profile the PHASE_END "
            f"markers to find the slow phase and either fix it or bump the "
            f"budget in timing_budget.yaml with justification."
        )
    logger.info(
        "Smoke [%s] PASSED budget check: %.1fs <= %.1fs (%.0f%% of budget)",
        mode, elapsed_s, budget, 100 * elapsed_s / budget,
    )


def _runtime_smoke(config: dict) -> None:
    """End-to-end smoke test with minimal data.

    Exercises the SAME module imports + S3 reads + ArcticDB reads + model
    load paths as the full backtest, but scoped to a handful of tickers
    and a single recent signal date so it completes in ~30-60 seconds.

    Runs after BacktesterPreflight to catch environment issues that the
    cheap preflight can't see from import checks alone:
      - Actual ArcticDB `read()` works (not just `list_symbols`)
      - signal_loader resolves a usable signals.json
      - The Layer-1A GBM booster loads and predicts on a real feature
        tensor with `scorer.feature_names` populated

    Raises RuntimeError with a named ``[stage=X]`` prefix on the first
    failure so the operator sees exactly where in the end-to-end chain
    the real problem is — not "your 80-minute backtest died in stage N."

    Motivated by the 2026-04-21 Saturday SF dry-run where
    ``No module named 'alpha_engine_lib.arcticdb'`` surfaced ~80 minutes
    into a spot run. With preflight + runtime smoke, the same failure
    would surface in ~2 seconds (preflight) or ~30 seconds (smoke).
    """
    import numpy as np
    bucket = config.get("signals_bucket", "alpha-engine-research")

    def _fail(stage: str, exc: Exception) -> RuntimeError:
        return RuntimeError(
            f"Runtime smoke FAILED [stage={stage}]: {exc}. "
            "Full backtest is aborted to avoid 60-80 minutes of wasted "
            "spot compute. Fix the underlying issue and re-run."
        )

    # Stage 1: universe symbols end-to-end (catches lib/arcticdb issues
    # that preflight's import check only surfaces at import time).
    try:
        from alpha_engine_lib.arcticdb import get_universe_symbols
        symbols = get_universe_symbols(bucket)
        if not symbols:
            raise RuntimeError("empty universe — ArcticDB has zero symbols")
        sample = [t for t in _SMOKE_SAMPLE_TICKERS if t in symbols][:3]
        if not sample:
            raise RuntimeError(
                f"none of {_SMOKE_SAMPLE_TICKERS} are in the current universe "
                f"({len(symbols)} symbols) — universe drift or a broken library"
            )
    except Exception as exc:
        raise _fail("universe_symbols", exc) from exc
    logger.info("Smoke [universe_symbols]: %d symbols, sample=%s", len(symbols), sample)

    # Stage 2: per-ticker ArcticDB read (catches per-symbol read failures
    # that list_symbols alone wouldn't surface).
    try:
        from alpha_engine_lib.arcticdb import open_universe_lib
        lib = open_universe_lib(bucket)
        for t in sample:
            df = lib.read(t).data
            if df.empty:
                raise RuntimeError(f"{t}: empty frame")
    except Exception as exc:
        raise _fail("arcticdb_per_ticker_read", exc) from exc
    logger.info("Smoke [arcticdb_per_ticker_read]: %d tickers read OK", len(sample))

    # Stage 3: recent signals.json loads and parses. Simulate mode
    # depends on this working for every replayed date; if the most
    # recent one can't be loaded the full replay would also fail.
    try:
        from loaders import signal_loader
        # signal_loader has `list_dates` or similar — fall back to a
        # direct S3 list if needed. Scope lookback generously.
        recent = _latest_signals_date(bucket)
        if recent is None:
            raise RuntimeError("no signals/{date}/signals.json found in S3 (14d lookback)")
        signals_raw = signal_loader.load(bucket, recent)
        if not isinstance(signals_raw, dict) or not signals_raw.get("date"):
            raise RuntimeError(f"{recent}/signals.json parsed but missing 'date' field")
    except Exception as exc:
        raise _fail("signals_load", exc) from exc
    logger.info("Smoke [signals_load]: loaded %s/signals.json OK", recent)

    # Stage 4: Layer-1A GBM loads and predicts. Covers the
    # download_gbm_model + GBMScorer.load + scorer.predict path that
    # predictor-backtest mode exercises over 10y. A single tensor of
    # zeros is enough to verify feature_names is populated + the
    # booster is callable.
    try:
        from synthetic.predictor_backtest import download_gbm_model
        from model.gbm_scorer import GBMScorer
        model_path = download_gbm_model(bucket=bucket)
        try:
            scorer = GBMScorer.load(model_path)
            if not scorer.feature_names:
                raise RuntimeError("loaded scorer has empty feature_names")
            X = np.zeros((len(sample), len(scorer.feature_names)), dtype=np.float32)
            preds = scorer.predict(X)
            if len(preds) != len(sample):
                raise RuntimeError(
                    f"prediction shape mismatch: expected {len(sample)}, got {len(preds)}"
                )
        finally:
            # Always clean up the temp file — _runtime_smoke runs before
            # the full modes and the temp downloads would otherwise
            # accumulate in /tmp across smoke + full invocations.
            import os
            for p in (model_path, model_path + ".meta.json"):
                if os.path.exists(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
    except Exception as exc:
        raise _fail("gbm_load_predict", exc) from exc
    logger.info(
        "Smoke [gbm_load_predict]: scorer loaded, feature_names populated (%d features), "
        "predict returned %d values",
        len(scorer.feature_names), len(preds),
    )

    logger.info("Runtime smoke PASSED — proceeding to full backtest modes")


def _latest_signals_date(bucket: str, max_lookback: int = 14) -> str | None:
    """Return the most recent date (YYYY-MM-DD) whose signals.json is in S3,
    or None if none found within ``max_lookback`` calendar days.

    Walked day-by-day via HEAD object rather than listing — a single HEAD
    is cheaper than a ListObjectsV2 call and easier to reason about in
    the smoke path.
    """
    import boto3
    from datetime import date, timedelta
    s3 = boto3.client("s3")
    today = date.today()
    for days_back in range(max_lookback + 1):
        candidate = today - timedelta(days=days_back)
        key = f"signals/{candidate.isoformat()}/signals.json"
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return candidate.isoformat()
        except Exception:
            continue
    return None


# ── Pipeline orchestration ──────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Alpha Engine Backtester (simulation)")
    parser.add_argument(
        "--mode",
        choices=[
            "simulate", "param-sweep", "all", "predictor-backtest",
            "portfolio-optimizer-backtest",
            "smoke",
            "smoke-simulate", "smoke-param-sweep",
            "smoke-predictor-backtest", "smoke-phase4",
            "smoke-predictor-param-sweep",
        ],
        default="simulate",
        help=(
            "Pipeline mode. 'smoke' runs preflight + end-to-end runtime "
            "smoke with minimal data (~30-60s) then exits 0. The "
            "'smoke-<phase>' modes exercise a single phase-family with "
            "a tiny fixture (few dates, tiny grid, short predictor "
            "lookback) and assert completion within the budget declared "
            "in timing_budget.yaml — used to catch phase regressions at "
            "smoke time instead of during a 2h Saturday SF run."
        ),
    )
    parser.add_argument(
        "--signal-source",
        dest="signal_source",
        choices=["synthetic", "production"],
        default="synthetic",
        help=(
            "Input stream for the portfolio-optimizer cutover gate. "
            "'synthetic' (default) replays the 10y predictor-GBM over the "
            "full universe (a stress test). 'production' (ROADMAP L124 PR 2) "
            "runs the gate against the deployed research cohort "
            "(signals/{date}/signals.json + predictor/predictions/{date}.json) "
            "— the verdict the operator trusts as the SOLE promotion lever. "
            "Production runs persist to the additive "
            "predictor/optimizer_gate/production/ S3 namespace."
        ),
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--db", help="Override research_db path from config")
    parser.add_argument("--upload", action="store_true", help="Upload results to S3")
    parser.add_argument("--date", default=date.today().isoformat(), help="Run date label")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--stop-instance", action="store_true",
                        help="Stop this EC2 instance after completion (for scheduled runs)")
    parser.add_argument("--rollback", action="store_true",
                        help="Rollback all S3 configs to previous versions and exit")
    parser.add_argument("--freeze", action="store_true",
                        help="Skip all S3 config promotions (guardrails compute + report but never write)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Exercise the full backtest pipeline without touching production S3 "
                             "state. Bundles --freeze + --force + no upload + date namespaced to "
                             "`.dry-run/{date}/` so phase markers, artifacts, and reports never "
                             "collide with scheduled-SF output on the same calendar date. Use for "
                             "ad-hoc validation runs (e.g. verifying a refactor pre-SF) where you "
                             "want full-universe coverage but no prod pollution. Mirrors the "
                             "existing smoke-mode isolation pattern (smoke uses `.smoke/{date}/`).")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Bypass the runtime smoke test that precedes the full modes. Only for "
                             "genuine restart cases where the operator knows the environment is good; "
                             "default behavior is to always run the ~30-60s smoke before committing "
                             "to 60-80 minutes of full work.")
    parser.add_argument("--skip-phase4-evaluations", action="store_true",
                        help="Skip Phase 4 predictor-hyperparameter feedback (ensemble mode, signal "
                             "threshold, feature pruning). Each Phase 4 evaluator runs a full silent "
                             "simulation internally; skipping all three shaves the predictor-pipeline "
                             "runtime dramatically during dry-runs where we only want 'does the "
                             "pipeline complete end-to-end'. Defaults to running. Routable from the "
                             "Saturday Step Function input as `skip_phase4_evaluations: true`.")
    parser.add_argument("--skip-phases", default="",
                        help="Comma-separated list of phase names to force-skip (e.g. "
                             "'simulate,param_sweep'). Overrides any persisted marker. For testing or "
                             "when a phase is known-broken and you want to run downstream. Caller is "
                             "responsible for ensuring downstream phases can tolerate the skipped "
                             "upstream (via --only-phases or cascade handling in the code).")
    parser.add_argument("--only-phases", default="",
                        help="Comma-separated list of phase names that ARE allowed to run; all others "
                             "are skipped. Useful for targeted testing. Cannot be combined with "
                             "--skip-phases (would be contradictory).")
    parser.add_argument("--force", action="store_true",
                        help="Re-run every phase even if a completion marker exists on S3 for today's "
                             "date. The default is auto-skip-per-date: a phase that completed today "
                             "(same args.date, status=ok) is skipped on retry. Use this to force a "
                             "full recompute from scratch.")
    parser.add_argument("--force-phases", default="",
                        help="Comma-separated list of phase names to force-rerun (overrides markers "
                             "for those phases only). More surgical than --force.")
    # Tier 4 vectorized predictor_param_sweep — default ON since
    # 2026-04-28 v18 validated stats parity (post fee-rate alignment)
    # + cap retighten 5400→1800. Two flags retained for explicit control:
    #   --use-vectorized-sweep : redundant under default-on, kept for
    #                            backward-compat with operator scripts.
    #   --use-scalar-sweep     : opt-out / emergency rollback. The scalar
    #                            path remains in-tree and is one-flag away.
    # See `synthetic.vectorized_sweep` + `synthetic.vectorized_stats` for
    # the matrix-first architecture.
    sweep_engine = parser.add_mutually_exclusive_group()
    sweep_engine.add_argument("--use-vectorized-sweep", action="store_true",
                              help="(default behavior since 2026-04-28) Run "
                                   "predictor_param_sweep through the matrix-axis "
                                   "vectorized engine. All N combos evaluate "
                                   "simultaneously per date.")
    sweep_engine.add_argument("--use-scalar-sweep", action="store_true",
                              help="Opt out of the vectorized sweep — fall back to "
                                   "the scalar per-combo loop (vectorbt path). "
                                   "Reserved for emergency rollback if the "
                                   "vectorized stats diverge from scalar in a "
                                   "spot run; default-on flip happened 2026-04-28.")
    parser.add_argument(
        "--pit-parity", action="store_true",
        help="Observational proof-of-impact: run the predictor backtest both "
             "ways (legacy single-pass vs --walk-forward) over the same date "
             "grid and emit the skilled-risk-basket contamination report to "
             "backtest/{date}/pit_parity.json (ROADMAP L2371 / plan §D4). "
             "Dedicated run — does NOT run the optimizer pipeline and never "
             "writes configs. The --walk-forward default flip is gated on a "
             "human reading this report (plan §5).",
    )
    parser.add_argument(
        "--pit-parity-pass", choices=["lookahead", "walkforward"], default=None,
        help="INTERNAL (L4487): run ONE pit_parity predictor pass in an isolated "
             "subprocess and pickle its stats to --stats-out. run_pit_parity "
             "invokes this twice (lookahead, walkforward) so each pass's RSS is "
             "reclaimed by the OS between passes — bounded-footprint O(max single "
             "pass) instead of the O(sum) that needed a 16 GB instance. Not for "
             "manual use.",
    )
    parser.add_argument(
        "--config-json", default=None,
        help="INTERNAL (L4487): path to the JSON-serialized config the "
             "--pit-parity-pass child should run with.",
    )
    parser.add_argument(
        "--stats-out", default=None,
        help="INTERNAL (L4487): path the --pit-parity-pass child pickles its "
             "predictor-backtest stats dict to.",
    )
    parser.add_argument(
        "--walk-forward", action="store_true",
        help="Point-in-time-honest predictor backtest: resolve archived "
             "momentum weights whose knowledge-time ≤ each fold's decision "
             "date instead of replaying history against the current live "
             "model (ROADMAP L2371 / Backtester Phase 2; plan "
             "pit-discipline-260515.md). DEFAULT OFF — the single-pass "
             "look-ahead path stays the default until PR 3's --pit-parity "
             "report is reviewed and the flip is made manually (plan §5).",
    )
    return parser.parse_args()


def _init_pipeline(args: argparse.Namespace, config: dict) -> None:
    """Initialize optimizer modules and pull research DB if needed.

    Mutates config in place (adds research_db, _db_pull_status).
    """
    executor_optimizer.init_config(config)

    # Pull research.db from S3 (or use --db override) so the backtest
    # stage's reporter can report an honest Pipeline-Health status.
    #
    # Why this lives here and not only in evaluate.py: the Saturday SF
    # split (evaluator-split-260507 / PR #250-era) runs the `backtest`
    # stage as a standalone SF state with `--skip-stages=evaluator`, so
    # the dedicated Evaluator state (the only other init_research_db
    # caller, evaluate.py:123) does NOT run inside the Backtester state.
    # Before this call, backtest.py only set research_db when `--db` was
    # passed, leaving `_db_pull_status` unset on the SF path — the
    # reporter then rendered a bare `- Research DB: None`.
    #
    # The original evaluate.py split (commit c852393, 2026-04-09) removed
    # the research.db pull from backtest.py and relocated it to evaluate.py;
    # the SF evaluator-stage split later exposed that latent gap on the
    # standalone backtest stage. init_research_db degrades gracefully
    # (sets research_db=None + _db_pull_status="failed" on a pull failure)
    # so the predictor-only / synthetic modes — which do not consume
    # research.db — are unaffected; the failure surfaces loudly via the
    # WARNING/ERROR log + the reporter's explicit **MISSING** line per the
    # existing convention rather than crashing those paths.
    init_research_db(args.db, config)

    # Set the assembler-cutover flag from config — when true, executor
    # optimizer's apply() skips its legacy live-key writes (the assembler
    # in evaluate.py becomes the sole writer of config/executor_params.json).
    # Default false; flip via alpha-engine-config under the `assembler:` section.
    from optimizer.assembler import set_cutover_enabled as _set_cutover_enabled
    _set_cutover_enabled(
        config.get("assembler", {}).get("cutover_enabled", False),
    )


def _run_simulation_pipeline(
    args: argparse.Namespace,
    config: dict,
    _sim_setup: tuple | None,
    current_executor_params: dict | None,
    fd=None,
) -> tuple[dict | None, object | None, dict | None]:
    """Run simulate mode, param sweep, executor optimizer, and twin simulation.

    Returns (portfolio_stats, sweep_df, executor_rec).
    """
    portfolio_stats = None
    sweep_df = None
    executor_rec = None

    registry = config["_phase_registry"]

    # Precomputed feature maps — built ONCE per _run_simulation_pipeline
    # invocation, shared across simulate, param-sweep, holdout, and twin
    # sub-stages. Without this hoist, every sim_fn closure below lazily
    # derives the maps inside _run_simulation_loop, and the param-sweep
    # path pays 60× the ~900-ticker ArcticDB bulk read (2026-04-22 13:00
    # PT re-run timed out for exactly this reason — py-spy confirmed every
    # combo was re-entering load_precomputed_feature_maps). The guard
    # matches the shape of the simulate/sweep blocks: skip the read
    # entirely when _sim_setup is None or price_matrix is empty.
    atr_by_ticker = None
    vwap_series_by_ticker = None
    coverage_by_ticker = None
    if (
        _sim_setup is not None
        and _sim_setup[3] is not None  # price_matrix
        and args.mode in ("simulate", "param-sweep", "all")
    ):
        try:
            from store.feature_maps import load_precomputed_feature_maps
            bucket = config.get("signals_bucket", "alpha-engine-research")
            _smoke_tickers = config.get("smoke_tickers")
            _allowlist = set(_smoke_tickers) if _smoke_tickers else None
            atr_by_ticker, vwap_series_by_ticker, coverage_by_ticker = load_precomputed_feature_maps(
                bucket, tickers_allowlist=_allowlist,
            )
        except Exception as exc:
            # Fall through to lazy per-call derivation rather than abort.
            # Preserves existing behavior on a bulk-read failure — each
            # _run_simulation_loop call will hit the slower ArcticDB path
            # individually. Logged loud so the perf regression is visible.
            logger.warning(
                "feature_maps: bulk precompute failed (%s) — falling back "
                "to per-call ArcticDB reads inside _run_simulation_loop. "
                "Param sweep will run at the pre-PR-#50 rate.",
                exc,
            )

    # ── Simulate mode ─────────────────────────────────────────────────────
    # Includes "param-sweep": the baseline single-policy simulation produces
    # portfolio_stats, which the Evaluator REQUIRES (alongside sweep_df from the
    # param-sweep block below). The 2026-05-16 SF backtester split (#249/#250)
    # set the main Backtester state to --mode=param-sweep and moved predictor /
    # portfolio-optimizer to their own states — but no SF state runs
    # simulate/all, so portfolio_stats.json silently stopped being produced and
    # the Evaluator hard-failed on missing critical artifacts from ~2026-05-20
    # (L4513). Running the simulate phase in param-sweep mode restores it.
    bucket = config.get("signals_bucket", "alpha-engine-research")
    s3 = registry.s3_client
    if args.mode in ("simulate", "param-sweep", "all"):
        from phase_artifacts import save_json, load_json
        try:
            with registry.phase(
                "simulate", mode=args.mode, supports_auto_skip=True,
            ) as ctx:
                if ctx.skipped:
                    marker = registry.load_marker("simulate") or {}
                    keys = marker.get("artifact_keys") or []
                    if not keys:
                        raise RuntimeError(
                            "simulate auto-skip: marker has no artifact_keys — "
                            "cannot reload portfolio_stats"
                        )
                    portfolio_stats = load_json(bucket, keys[0], s3_client=s3)
                else:
                    if _sim_setup is None:
                        raise RuntimeError("Simulation setup failed — cannot run simulate")
                    executor_run, SimulatedIBKRClient, dates, price_matrix, init_cash, ohlcv = _sim_setup
                    if price_matrix is None:
                        min_dates = config.get("min_simulation_dates", 5)
                        portfolio_stats = {
                            "status": "insufficient_data",
                            "dates_available": len(dates),
                            "min_required": min_dates,
                        }
                    else:
                        portfolio_stats = _run_simulation_loop(
                            executor_run, SimulatedIBKRClient, dates, price_matrix, config,
                            ohlcv_by_ticker=ohlcv,
                            atr_by_ticker=atr_by_ticker,
                            vwap_series_by_ticker=vwap_series_by_ticker,
                            coverage_by_ticker=coverage_by_ticker,
                        )
                    ctx.record_artifact(save_json(
                        bucket, args.date, "simulate", "portfolio_stats", portfolio_stats,
                        s3_client=s3,
                    ))
        except Exception as e:
            # Tier 2 diagnostic (2026-04-27): include full traceback in
            # the error log so the smoke catch doesn't swallow the stack
            # frame that triggered the failure. Without exc_info=True the
            # smoke harness reports "RecursionError" with no source site.
            logger.error("Mode 2 simulation failed: %s", e, exc_info=True)
            if fd:
                fd.report(e, severity="error", context={
                    "site": "simulation", "mode": args.mode})
            portfolio_stats = {"status": "error", "error": str(e)}

    # ── Param sweep ───────────────────────────────────────────────────────
    if args.mode in ("param-sweep", "all"):
        from phase_artifacts import save_dataframe, load_dataframe
        try:
            with registry.phase(
                "param_sweep", mode=args.mode, supports_auto_skip=True,
            ) as ctx:
                if ctx.skipped:
                    marker = registry.load_marker("param_sweep") or {}
                    keys = marker.get("artifact_keys") or []
                    if not keys:
                        # No persisted sweep (e.g. empty sweep on prior run);
                        # treat as empty DataFrame so executor_optimizer skips cleanly.
                        sweep_df = pd.DataFrame()
                    else:
                        sweep_df = load_dataframe(bucket, keys[0], s3_client=s3)
                elif _sim_setup is None:
                    raise RuntimeError("Simulation setup failed — cannot run param sweep")
                else:
                    executor_run, SimulatedIBKRClient, dates, price_matrix, _, ohlcv = _sim_setup
                    if price_matrix is None:
                        logger.warning("Param sweep skipped: only %d signal dates available", len(dates))
                        sweep_df = pd.DataFrame()
                    else:
                        def sim_fn(combo_config: dict) -> dict:
                            return _run_simulation_loop(
                                executor_run, SimulatedIBKRClient, dates, price_matrix, combo_config,
                                ohlcv_by_ticker=ohlcv,
                                atr_by_ticker=atr_by_ticker,
                                vwap_series_by_ticker=vwap_series_by_ticker,
                                coverage_by_ticker=coverage_by_ticker,
                            )
                        grid = config.get("param_sweep", param_sweep.DEFAULT_GRID)
                        grid = _seed_grid_with_current(grid, current_executor_params)
                        sweep_settings = config.get("param_sweep_settings", {})
                        logger.info("Running param sweep (%s): %s", sweep_settings.get("mode", "random"), {k: len(v) for k, v in grid.items()})
                        sweep_df = param_sweep.sweep(grid, sim_fn, config, sweep_settings=sweep_settings)
                    if sweep_df is not None and not sweep_df.empty:
                        ctx.record_artifact(save_dataframe(
                            bucket, args.date, "param_sweep", "sweep_df", sweep_df,
                            preserve_index=False, s3_client=s3,
                        ))
        except Exception as e:
            # Fail-loud diagnostics (L4525): mirror the simulate except's
            # exc_info=True so the stack frame survives, AND persist the full
            # traceback to S3. The spot log is 24KB-capped and the staging dir
            # is cleaned on exit, so a log-only stack is unrecoverable — this is
            # exactly why recovery8's param_sweep raise could not be diagnosed.
            # See [[feedback_no_silent_fails]].
            logger.error("Param sweep failed: %s", e, exc_info=True)
            # Best-effort secondary observability hung off the primary record
            # above (logger exc_info + fd.report below) — a persist failure must
            # not mask the original; it is itself logged loud. The diagnostic is
            # written under a `_diagnostics` phase namespace so it never collides
            # with the param_sweep phase marker / auto-skip artifact_keys.
            try:
                from phase_artifacts import save_json as _save_json
                _save_json(
                    bucket, args.date, "_diagnostics", "param_sweep_traceback",
                    {
                        "error": str(e),
                        "type": type(e).__name__,
                        "traceback": traceback.format_exc(),
                        "mode": args.mode,
                    },
                    s3_client=s3,
                )
            except Exception as persist_err:  # noqa: BLE001 — best-effort diag
                logger.error(
                    "Failed to persist param_sweep traceback to S3: %s",
                    persist_err, exc_info=True,
                )
            if fd:
                fd.report(e, severity="error", context={
                    "site": "param_sweep", "mode": args.mode})
            sweep_df = None

        # Executor parameter optimization from sweep results
        if sweep_df is not None and not sweep_df.empty:
            from phase_artifacts import save_json, load_json
            try:
                with registry.phase(
                    "executor_optimizer", mode=args.mode, supports_auto_skip=True,
                ) as ctx:
                  if ctx.skipped:
                    marker = registry.load_marker("executor_optimizer") or {}
                    keys = marker.get("artifact_keys") or []
                    if not keys:
                        raise RuntimeError(
                            "executor_optimizer auto-skip: marker has no "
                            "artifact_keys — cannot reload executor_rec"
                        )
                    executor_rec = load_json(bucket, keys[0], s3_client=s3)
                  else:
                    executor_rec = executor_optimizer.recommend(
                        sweep_df, config, current_params=current_executor_params,
                    )
                    if executor_rec.get("status") == "ok" and _sim_setup is not None:
                        executor_run_fn, SimClientCls, sim_dates, pm, _, ohlcv_data = _sim_setup
                        if pm is not None:
                            def holdout_sim_fn(combo_config):
                                return _run_simulation_loop(
                                    executor_run_fn, SimClientCls, sim_dates, pm, combo_config,
                                    ohlcv_by_ticker=ohlcv_data,
                                    atr_by_ticker=atr_by_ticker,
                                    vwap_series_by_ticker=vwap_series_by_ticker,
                                    coverage_by_ticker=coverage_by_ticker,
                                )
                            with registry.phase("executor_holdout", mode=args.mode):
                                executor_rec = executor_optimizer.validate_holdout(
                                    executor_rec, holdout_sim_fn, sim_dates, config,
                                )

                    # Twin simulation: current vs proposed on same dates
                    if executor_rec.get("status") == "ok" and _sim_setup is not None:
                        executor_run_fn, SimClientCls, sim_dates, pm, _, ohlcv_data = _sim_setup
                        if pm is not None and current_executor_params:
                            from optimizer.twin_sim import run_twin_simulation
                            from analysis.param_sweep import _deepcopy_safe_config
                            recommended = executor_rec.get("recommended_params", {})
                            # Use _deepcopy_safe_config — base `config` holds
                            # the PhaseRegistry (boto3 client) which is not
                            # deepcopy-safe. Matches the fix in _run_combos.
                            current_cfg = _deepcopy_safe_config(config)
                            current_cfg.update(current_executor_params)
                            proposed_cfg = _deepcopy_safe_config(config)
                            proposed_cfg.update(recommended)
                            changed_keys = [k for k in recommended if recommended.get(k) != current_executor_params.get(k)]

                            def twin_sim_fn(cfg):
                                return _run_simulation_loop(
                                    executor_run_fn, SimClientCls, sim_dates, pm, cfg,
                                    ohlcv_by_ticker=ohlcv_data,
                                    atr_by_ticker=atr_by_ticker,
                                    vwap_series_by_ticker=vwap_series_by_ticker,
                                    coverage_by_ticker=coverage_by_ticker,
                                )
                            with registry.phase("executor_twin_sim", mode=args.mode):
                                executor_rec["twin_sim"] = run_twin_simulation(
                                    twin_sim_fn, current_cfg, proposed_cfg, changed_keys,
                                )

                    if executor_rec.get("status") == "ok":
                        if args.freeze:
                            executor_rec["apply_result"] = {"applied": False, "reason": "frozen (--freeze flag)"}
                        else:
                            executor_rec["apply_result"] = executor_optimizer.apply(executor_rec, bucket)

                    # Persist final state (includes holdout + twin_sim +
                    # apply_result) so an auto-skipped retry restores the
                    # complete optimizer recommendation tree atomically.
                    ctx.record_artifact(save_json(
                        bucket, args.date, "executor_optimizer", "executor_rec", executor_rec,
                        s3_client=s3,
                    ))
            except Exception as e:
                logger.error("Executor optimizer failed: %s", e)
                if fd:
                    fd.report(e, severity="error", context={
                        "site": "executor_optimizer", "mode": args.mode})
                executor_rec = {"status": "error", "error": str(e)}

    return portfolio_stats, sweep_df, executor_rec


def _run_predictor_pipeline(
    args: argparse.Namespace,
    config: dict,
    executor_rec: dict | None,
    current_executor_params: dict | None,
    fd=None,
) -> tuple[dict | None, object | None, dict | None]:
    """Run predictor backtest and auto-apply executor params from predictor sweep.

    Returns (predictor_stats, predictor_sweep_df, executor_rec).
    """
    predictor_stats = None
    predictor_sweep_df = None

    try:
        predictor_stats, predictor_sweep_df = run_predictor_param_sweep(config)
    except Exception as e:
        logger.error("Predictor backtest failed: %s", e)
        if fd:
            fd.report(e, severity="error", context={
                "site": "predictor_backtest", "mode": args.mode})
        predictor_stats = {"status": "error", "error": str(e)}
        predictor_sweep_df = None

    # Auto-apply executor params from the predictor sweep ONLY as the
    # in-process fallback when the signal-based sweep failed to produce an
    # "ok" recommendation in THIS process (mode=all). Gated on mode=="all"
    # for the L4472 phase-split: when the predictor pipeline runs in its own
    # SF state (--mode=predictor-backtest), the simulation pipeline ran in a
    # SEPARATE state (Backtester, --mode=param-sweep) and is the SOLE writer
    # of the `config/executor_params/recommendations/{date}/
    # from_executor_optimizer.json` artifact. executor_optimizer.apply()
    # hardcodes optimizer_name="executor_optimizer", so a predictor-side
    # apply here would clobber that same S3 key with predictor-based params
    # (the fork). The PredictorBacktest state only runs when the Backtester
    # state SUCCEEDED (CheckBacktesterStatus: Success -> PredictorBacktest),
    # i.e. sim produced an "ok" recommendation — exactly the case mode=all
    # already suppresses predictor's executor-apply. Gating on mode=="all"
    # therefore preserves monolithic semantics (sim wins; predictor-apply is
    # the in-process fallback only) while making the split collision-free.
    # See ROADMAP L4472. mode=all behavior is byte-for-byte unchanged.
    if (
        args.mode == "all"
        and (executor_rec is None or executor_rec.get("status") not in ("ok",))
        and predictor_sweep_df is not None
        and not predictor_sweep_df.empty
    ):
        try:
            executor_rec = executor_optimizer.recommend(
                predictor_sweep_df, config, current_params=current_executor_params,
            )
            if executor_rec.get("status") == "ok":
                bucket = config.get("signals_bucket", "alpha-engine-research")
                if args.freeze:
                    executor_rec["apply_result"] = {"applied": False, "reason": "frozen (--freeze flag)"}
                else:
                    executor_rec["apply_result"] = executor_optimizer.apply(executor_rec, bucket)
        except Exception as e:
            logger.error("Executor optimizer (predictor sweep) failed: %s", e)
            if fd:
                fd.report(e, severity="error", context={
                    "site": "executor_optimizer_predictor", "mode": args.mode})
            executor_rec = {"status": "error", "error": str(e)}

    return predictor_stats, predictor_sweep_df, executor_rec


def _export_simulation_artifacts(
    config: dict,
    run_date: str,
    sweep_df=None,
    predictor_sweep_df=None,
    portfolio_stats: dict | None = None,
    predictor_stats: dict | None = None,
) -> None:
    """Write simulation artifacts to S3 for downstream evaluator consumption.

    The evaluator reads these artifacts to run executor optimization and
    include simulation results in its report. If the backtester fails,
    the evaluator runs without them (degraded mode).
    """
    import io
    bucket = config.get("output_bucket", config.get("signals_bucket", "alpha-engine-research"))
    prefix = f"backtest/{run_date}"
    s3 = boto3.client("s3")
    exported = []

    # Write sweep_df.parquet whenever the frame EXISTS (incl. empty) — an empty
    # sweep is a valid no-op (no admissible combo), and the Evaluator must find
    # the artifact PRESENT to proceed + no-op its optimizer (L4523). Only a None
    # frame (phase didn't run) is skipped — the guard catches that as fatal.
    if sweep_df is not None:
        buf = io.BytesIO()
        sweep_df.to_parquet(buf, index=False)
        s3.put_object(Bucket=bucket, Key=f"{prefix}/sweep_df.parquet", Body=buf.getvalue())
        exported.append("sweep_df.parquet" + (" (empty)" if sweep_df.empty else ""))

    if predictor_sweep_df is not None and not predictor_sweep_df.empty:
        buf = io.BytesIO()
        predictor_sweep_df.to_parquet(buf, index=False)
        s3.put_object(Bucket=bucket, Key=f"{prefix}/predictor_sweep_df.parquet", Body=buf.getvalue())
        exported.append("predictor_sweep_df.parquet")

    if portfolio_stats:
        s3.put_object(Bucket=bucket, Key=f"{prefix}/portfolio_stats.json", Body=json.dumps(portfolio_stats, indent=2, default=str).encode())
        exported.append("portfolio_stats.json")

    if predictor_stats:
        s3.put_object(Bucket=bucket, Key=f"{prefix}/predictor_stats.json", Body=json.dumps(predictor_stats, indent=2, default=str).encode())
        exported.append("predictor_stats.json")

    if exported:
        logger.info("Exported simulation artifacts to s3://%s/%s/: %s", bucket, prefix, ", ".join(exported))


def _load_predictor_artifacts(config: dict, run_date: str):
    """Reload the predictor-backtest artifacts from S3 (L4527 skip/resume).

    Mirrors ``_export_simulation_artifacts``'s predictor writes
    (``predictor_stats.json`` + ``predictor_sweep_df.parquet`` under
    ``backtest/{run_date}/``) so a targeted recovery can SKIP the ~121-min
    predictor pipeline — ``--skip-phases=predictor_pipeline`` — and still feed
    its outputs to the downstream export/report stages instead of redundantly
    re-running it (supersedes L4519).

    Returns ``(predictor_stats, predictor_sweep_df)``. A missing artifact
    yields ``None`` for that slot with a loud WARN — non-fatal, because the
    predictor artifacts are NOT in the Evaluator-critical set (see
    ``pipeline_manifest``), so skipping without a prior run degrades gracefully
    rather than starving the Evaluator. A non-404 S3 error fails loud.
    """
    import io
    from botocore.exceptions import ClientError

    bucket = config.get("output_bucket", config.get("signals_bucket", "alpha-engine-research"))
    prefix = f"backtest/{run_date}"
    s3 = boto3.client("s3")
    predictor_stats = None
    predictor_sweep_df = None

    def _get(key: str):
        try:
            return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NotFound"):
                logger.warning(
                    "predictor_pipeline skipped but s3://%s/%s is absent — "
                    "downstream predictor output will be missing. Run the "
                    "predictor pipeline first, or don't skip it.", bucket, key,
                )
                return None
            raise  # transient / permission → fail loud, don't guess

    stats_body = _get(f"{prefix}/predictor_stats.json")
    if stats_body is not None:
        predictor_stats = json.loads(stats_body)
    sweep_body = _get(f"{prefix}/predictor_sweep_df.parquet")
    if sweep_body is not None:
        predictor_sweep_df = pd.read_parquet(io.BytesIO(sweep_body))

    logger.info(
        "Reloaded predictor artifacts from s3://%s/%s/ (stats=%s, sweep_df=%s rows) "
        "— predictor pipeline skipped",
        bucket, prefix, predictor_stats is not None,
        len(predictor_sweep_df) if predictor_sweep_df is not None else 0,
    )
    return predictor_stats, predictor_sweep_df


def classify_simulation_outcome(
    mode: str,
    portfolio_stats: dict | None,
    sweep_df,
) -> PhaseOutcome:
    """Map the simulate/param-sweep critical artifacts to a 3-way PhaseOutcome.

    The single decision point for the simulate-phase outcome, encoding
    ARCHITECTURE.md §22 (the backtester's 0-result policy). Pure + side-effect
    free so it unit-tests directly (callers act on ``.status``). Replaces the
    inline if-ladder #294 shipped — the taxonomy is now a first-class,
    structured record (L4523) the rest of the arc (L4524/L4526) builds on.

    FAILURE (fail loud, infra/contract break):
      - ``portfolio_stats`` absent — the Evaluator hard-requires it; param-sweep
        was made to produce it (#292). Absence = the phase didn't run / errored,
        NOT a degenerate market result. ("did not produce portfolio_stats")
      - ``sweep_df is None`` — the param_sweep phase didn't run / errored (None,
        not an empty frame). ("sweep_df is ABSENT")
    EMPTY (valid no-op, WARN+alert — see §22):
      - ``sweep_df`` is an EMPTY frame — ran, no admissible combo (all entries
        gated by score/risk this window). Configs HELD, executor-optimizer
        skipped; the empty parquet is still written so the Evaluator finds it
        present + no-ops. Surfaced loudly so a suspicious degeneracy (e.g.
        zero-score inputs) is visible; the L4525 input-quality gate later
        classifies legit-empty vs garbage-input-empty.
    SUCCESS:
      - a non-empty sweep + present portfolio_stats.
    """
    phase_name = "export_artifacts"
    if not portfolio_stats:
        return PhaseOutcome(
            status=PhaseStatus.FAILURE,
            phase=phase_name,
            reason=(
                f"Backtester mode={mode} did not produce portfolio_stats "
                f"(absent, not empty) — evaluate.py hard-requires it. Failing "
                f"loud (L4513/L4518); this is an infra/contract break, not a "
                f"degenerate market result."
            ),
        )
    if sweep_df is None:
        return PhaseOutcome(
            status=PhaseStatus.FAILURE,
            phase=phase_name,
            reason=(
                f"Backtester mode={mode}: sweep_df is ABSENT (param_sweep "
                f"phase did not run / errored — None, not an empty frame). "
                f"Failing loud (L4513/L4524) — distinct from an empty sweep."
            ),
        )
    if getattr(sweep_df, "empty", True):
        return PhaseOutcome(
            status=PhaseStatus.EMPTY,
            phase=phase_name,
            n_admissible=0,
            degeneracy_reason="no admissible param combo (all entries gated by score/risk)",
            reason=(
                f"[outcome] sweep_df is EMPTY (mode={mode}) — no admissible param "
                f"combo this run (all entries gated by score/risk). Treating as "
                f"a valid no-op: configs HELD, executor-optimizer skipped. "
                f"L4525 input-quality gate will classify legit-vs-garbage."
            ),
        )
    return PhaseOutcome(
        status=PhaseStatus.SUCCESS,
        phase=phase_name,
        n_admissible=int(len(sweep_df)),
        reason=f"sweep produced {len(sweep_df)} admissible combo(s) (mode={mode}).",
    )


# ── Main entry point ────────────────────────────────────────────────────────


def main() -> None:
    # flow-doctor default-on (lib v0.58.0): the outer guard catches the
    # uncaught raise from anywhere in the body and reports it before
    # re-raising. No-op when flow-doctor is inactive (dev/CI/pytest).
    # The ~7 explicit fd.report() call sites inside _main_impl() remain;
    # dedup absorbs any same-signature overlap with the guard.
    with guard_entrypoint():
        _main_impl()


def _main_impl() -> None:
    args = _parse_args()

    # DATE_CONVENTIONS: normalize the run-date label to the NYSE trading day so
    # every backtest artifact (backtest/{date}/…, incl. pit_parity.json +
    # parity_metrics) keys by trading day — aligned with signals/{trading_day}/
    # and the ARTIFACT_REGISTRY trading-day axis. Idempotent: the spot already
    # normalizes RUN_DATE in spot_backtest.sh, so this re-normalization of
    # --date is a no-op there and also covers manual `python backtest.py` runs
    # (whose --date defaults to calendar date.today()). See L4466 + research #257.
    _orig_date = args.date
    args.date = resolve_trading_day(args.date)
    if args.date != _orig_date:
        logger.info("Normalized run-date %s (calendar) → %s (trading day)", _orig_date, args.date)

    # setup_logging already ran at module-top (see comment near the
    # alpha_engine_lib.logging import). Apply the user-requested level here.
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    _health_start = _time.time()

    # Retrieve the shared flow-doctor instance for explicit fd.report()
    # call sites in this function (param sweep / simulation / optimizer
    # error escalation). Returns None when FLOW_DOCTOR_ENABLED=0.
    fd = get_flow_doctor()

    config = load_config(args.config)

    # Stamp CLI flags into config so deep-pipeline code (run_predictor_param_sweep
    # and below) can read them without threading args all the way down.
    if args.skip_phase4_evaluations:
        config["skip_phase4_evaluations"] = True
    # Default-on the vectorized sweep (2026-04-28). Explicit opt-out via
    # `--use-scalar-sweep`; explicit opt-in via `--use-vectorized-sweep`
    # is redundant but kept for backward-compat with operator scripts.
    # config.yaml may also set `use_vectorized_sweep: false` to opt out
    # from a config file rather than CLI.
    if args.use_scalar_sweep:
        config["use_vectorized_sweep"] = False
    elif args.use_vectorized_sweep:
        config["use_vectorized_sweep"] = True
    else:
        config.setdefault("use_vectorized_sweep", True)

    # Point-in-time walk-forward (ROADMAP L2371 / Backtester Phase 2).
    # Stamped top-level (not under predictor_backtest) so it survives the
    # mode/smoke-fixture dict replacements that rewrite config["predictor_
    # backtest"]; predictor_backtest.run reads config.get("walk_forward").
    # config.yaml may also set `walk_forward: true` to drive it from a file.
    if args.walk_forward:
        config["walk_forward"] = True
    else:
        config.setdefault("walk_forward", False)
    # Run-date label = the PIT as-of for optimizer-baseline reads under
    # walk-forward (config_archive.as_of_date_from_config). evaluate.py:934
    # sets the same key; mirror it here so backtest-mode call sites
    # (param-sweep grid seed + executor baseline) resolve to a backdated
    # config when --date is backdated, and to "current" for a live run.
    config.setdefault("_run_date", args.date)

    # Smoke-phase mode: apply the fixture BEFORE phase-selection parsing
    # so the fixture's only_phases/skip_phases/force flow through the
    # registry. The fixture also rewrites args.mode to the routed full
    # mode (e.g. smoke-simulate → simulate) so downstream branching in
    # _run_simulation_pipeline / _run_predictor_pipeline is unchanged.
    _original_mode = args.mode
    _is_smoke_phase = _is_smoke_phase_mode(args.mode)
    if _is_smoke_phase:
        _apply_smoke_fixture(args.mode, args, config)

    # Apply dry-run isolation if requested. Bundles the same safety
    # switches as smoke mode but preserves the full-universe mode the
    # operator selected. Smoke already handles its own isolation via
    # _apply_smoke_fixture; --dry-run + smoke-X is redundant but safe
    # (the smoke fixture runs first, then dry-run adds its own guards
    # on top — the .smoke/ prefix wins because the date rewrite in
    # _apply_smoke_fixture has already executed).
    if getattr(args, "dry_run", False) and not _is_smoke_phase:
        _apply_dry_run_isolation(args)

    # Parse + validate phase-selection flags.
    def _split(s: str) -> list[str]:
        return [p.strip() for p in s.split(",") if p.strip()]

    skip_phases = _split(args.skip_phases)
    only_phases = _split(args.only_phases)
    force_phases = _split(args.force_phases)
    if skip_phases and only_phases:
        raise SystemExit(
            "--skip-phases and --only-phases are mutually exclusive — pick one"
        )

    # PhaseRegistry drives auto-skip-per-date + honors the CLI flags above.
    # Stored on config so deep-pipeline code can read it without threading
    # the registry through every function signature. Phases pass
    # supports_auto_skip=True only when they know how to persist + reload
    # their outputs (artifact persistence lands in PR 2/3).
    # Load per-phase hard caps from timing_budget.yaml. A phase exceeding
    # its cap trips the watchdog (all-thread stack dump + PhaseTimeoutError).
    # Missing caps leave the phase unwatchdogged — opt-in per phase.
    hard_caps = load_phase_hard_caps()
    if hard_caps:
        logger.info(
            "Phase watchdog active for %d phase(s): %s",
            len(hard_caps),
            ", ".join(f"{k}={v:.0f}s" for k, v in sorted(hard_caps.items())),
        )

    registry = PhaseRegistry(
        date=args.date,
        bucket=config.get("signals_bucket", "alpha-engine-research"),
        skip_phases=skip_phases,
        only_phases=only_phases or None,
        force=args.force,
        force_phases=force_phases,
        hard_caps=hard_caps,
    )
    config["_phase_registry"] = registry

    # Preflight: external-world handshakes must pass before any 90-min
    # spot run starts. Raises RuntimeError (propagates to non-zero exit)
    # on missing env vars, unreachable S3, or stale ArcticDB macro/SPY.
    # Kept out of --rollback path because rollback touches S3 configs
    # only, not ArcticDB.
    if not args.rollback:
        with registry.phase("preflight", mode=args.mode):
            from preflight import BacktesterPreflight
            BacktesterPreflight(
                bucket=config.get("signals_bucket", "alpha-engine-research"),
                mode="backtest",
                executor_paths=config.get("executor_paths") or [],
                predictor_paths=config.get("predictor_paths") or [],
            ).run()

        # Pre-spend input-quality gate (L4525, plan §6 Phase 3). For the modes
        # that simulate over signals.json history, assert the inputs aren't
        # degenerate (no usable signals / a wall of Score 0.0) BEFORE burning
        # ~121 min on a param-sweep that would silently empty-out on garbage
        # — the L4521/L4529 silent-starvation failure mode. Observe-first:
        # the verdict is always computed + logged + alerted; it only RAISES
        # when config.input_quality_gate.enforce is true (default false) so a
        # first live run can't false-fail before the soak confirms the verdict
        # is sane on real data. predictor-backtest uses synthetic signals, not
        # signals.json, so it's exempt.
        if args.mode in ("simulate", "param-sweep", "all"):
            with registry.phase("input_quality_gate", mode=args.mode):
                from analysis.input_quality import gate_signal_inputs
                from loaders import signal_loader as _sig_loader
                _iq_cfg = config.get("input_quality_gate") or {}
                _bucket = config.get("signals_bucket", "alpha-engine-research")
                try:
                    from alpha_engine_lib.alerts import publish as _iq_alert
                except Exception:  # noqa: BLE001 — alerts optional
                    _iq_alert = None
                gate_signal_inputs(
                    _bucket,
                    _sig_loader.list_dates(_bucket),
                    signal_loader=_sig_loader,
                    sample_recent=int(_iq_cfg.get("sample_recent", 20)),
                    enforce=bool(_iq_cfg.get("enforce", False)),
                    zero_score_garbage_fraction=float(
                        _iq_cfg.get("zero_score_garbage_fraction", 0.99)
                    ),
                    elevated_zero_observe_fraction=float(
                        _iq_cfg.get("elevated_zero_observe_fraction", 0.10)
                    ),
                    low_coverage_observe_fraction=float(
                        _iq_cfg.get("low_coverage_observe_fraction", 0.25)
                    ),
                    alert_publisher=_iq_alert,
                )

        # Runtime smoke: end-to-end sanity with minimal data (~30-60s).
        # Runs after preflight (so any preflight failure surfaces first,
        # in seconds) and before any full mode (so an environment bug
        # doesn't burn 60-80 min of spot compute). --skip-smoke is the
        # escape hatch for genuine restart scenarios; --mode=smoke runs
        # the smoke and exits 0 without doing full work.
        if args.mode == "smoke":
            with registry.phase("runtime_smoke", mode=args.mode):
                _runtime_smoke(config)
            logger.info("Smoke-only mode complete — exiting 0 without full run")
            return
        if not args.skip_smoke:
            with registry.phase("runtime_smoke", mode=args.mode):
                _runtime_smoke(config)

    # Handle --rollback before any other mode
    if args.rollback:
        from optimizer.rollback import rollback_all
        bucket = config.get("signals_bucket", "alpha-engine-research")
        results = rollback_all(bucket)
        for r in results:
            if r.get("rolled_back"):
                print(f"  Rolled back: {r['config_type']} → {r['key']}")
            else:
                print(f"  Skipped: {r.get('reason', 'unknown')}")
        return

    # --pit-parity-pass: INTERNAL child sub-mode (L4487). Runs exactly ONE
    # predictor-backtest pass (lookahead | walkforward) in this fresh process
    # and pickles its stats to --stats-out, then exits — so run_pit_parity can
    # invoke it once per pass via subprocess.run and let the OS reclaim each
    # pass's RSS between passes (bounded O(max single pass), not O(sum) which
    # needed a 16 GB box). Runs BEFORE _init_pipeline so it can never write a
    # config. A fresh `python backtest.py` (cwd=backtester) resolves the
    # `analysis` package correctly — sidesteps the multiprocessing-spawn
    # __main__ re-import collision that killed the earlier attempt (#285).
    if args.pit_parity_pass:
        import pickle
        import resource
        if not args.config_json or not args.stats_out:
            raise SystemExit("--pit-parity-pass requires --config-json and --stats-out")
        with open(args.config_json) as _f:
            pass_cfg = json.load(_f)
        pass_cfg["walk_forward"] = (args.pit_parity_pass == "walkforward")
        stats = run_predictor_backtest(pass_cfg)
        with open(args.stats_out, "wb") as _f:
            pickle.dump(stats, _f)
        # Anti-degradation guard (L4487): each pass is its own process, so
        # ru_maxrss IS this pass's peak RSS. Alert LOUD if it exceeds the sized
        # envelope — converts silent OOM / right-size-drift (the "these always
        # degrade" pattern) into an explicit signal. ru_maxrss is KiB on Linux.
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        budget_mb = float(os.environ.get("PIT_PARITY_PASS_RSS_BUDGET_MB", "4500"))
        logger.info("[pit_parity] pass=%s peak RSS=%.0f MB (budget %.0f MB)",
                    args.pit_parity_pass, peak_mb, budget_mb)
        if peak_mb > budget_mb:
            try:
                from alpha_engine_lib.alerts import publish as _alerts_publish
                _alerts_publish(
                    f"pit_parity pass={args.pit_parity_pass} peak RSS {peak_mb:.0f} MB "
                    f"exceeded budget {budget_mb:.0f} MB on {args.date} — the per-pass "
                    f"footprint has grown; re-check the Parity instance sizing (L4487).",
                    severity="warning",
                    source="alpha-engine-backtester/pit_parity",
                    dedup_key=f"pit_parity_rss_budget_{args.date}",
                    dedup_window_min=720,
                )
            except Exception as _alert_err:  # best-effort; log line is the primary surface
                logger.warning("[pit_parity] RSS-budget alert publish failed: %s", _alert_err)
        return

    # --pit-parity: dedicated observational run (plan §D4). Runs BEFORE
    # _init_pipeline / the optimizer so it can never write a config; emits
    # the contamination report and returns. Never raises into the SF — the
    # spot stage that invokes it is best-effort and non-blocking.
    if args.pit_parity:
        from analysis.pit_parity import run_pit_parity, write_failure_artifact
        try:
            report = run_pit_parity(config)
            print(json.dumps(
                {k: report[k] for k in (
                    "schema", "run_date", "delta_pit_minus_current",
                    "headline_log_alpha_delta", "materiality", "_s3_key",
                ) if k in report},
                indent=2, default=str,
            ))
        except Exception as e:
            # Per ``feedback_no_silent_fails`` secondary-observability
            # carve-out: spot run continues (primary deliverable = weights
            # archive + sweep), pit_parity failure is recorded on two
            # surfaces — (1) always-emit S3 artifact at
            # ``backtest/{date}/pit_parity.json`` with status=failed +
            # error_class + error_msg so the operator's manual-flip gate
            # never sees a missing artifact again; (2) Telegram + SNS
            # alert via ``alpha_engine_lib.alerts.publish`` (sev=warning,
            # dedup-keyed on run_date so a swept-cycle retry collapses to
            # one alert). The 2026-05-17→2026-05-24 incident swallowed
            # 4 silent failures with only an spot-stdout log line —
            # the operator's gate was unreachable for 11 days.
            logger.error(
                "[pit_parity] run failed (observational, non-fatal): %s",
                e, exc_info=True,
            )
            try:
                write_failure_artifact(config, e)
            except Exception as artifact_err:
                logger.error(
                    "[pit_parity] failure-artifact write also failed: %s",
                    artifact_err,
                )
            try:
                from alpha_engine_lib.alerts import publish as _alerts_publish
                run_date = config.get("_run_date") or "unknown"
                _alerts_publish(
                    f"pit_parity failed on {run_date}: "
                    f"{type(e).__name__}: {str(e)[:200]} — "
                    f"see s3://{config.get('signals_bucket', 'alpha-engine-research')}/"
                    f"backtest/{run_date}/pit_parity.json",
                    severity="warning",
                    source="alpha-engine-backtester/pit_parity",
                    dedup_key=f"pit_parity_failed_{run_date}",
                    dedup_window_min=720,  # 12h — one alert per Saturday cycle
                )
            except Exception as alert_err:
                logger.error(
                    "[pit_parity] operator alert publish also failed: %s",
                    alert_err,
                )
        return

    _init_pipeline(args, config)

    # ── Default results (overwritten by each pipeline stage) ──────────────
    portfolio_stats = None
    sweep_df = None
    executor_rec = None
    predictor_stats = None
    predictor_sweep_df = None

    # ── Simulation setup (shared by simulate + param-sweep) ───────────────
    _sim_setup = None
    if args.mode in ("simulate", "param-sweep", "all"):
        bucket = config.get("signals_bucket", "alpha-engine-research")
        try:
            with registry.phase(
                "simulation_setup", mode=args.mode, supports_auto_skip=True,
            ) as ctx:
                if ctx.skipped:
                    _sim_setup = _load_simulation_setup(config, registry)
                else:
                    _sim_setup = _setup_simulation(config)
                    _save_simulation_setup(
                        ctx, bucket, args.date, _sim_setup,
                        s3_client=registry.s3_client,
                    )
        except Exception as e:
            logger.error("Simulation setup failed: %s", e)
            if fd:
                fd.report(e, severity="error", context={
                    "site": "simulation_setup", "mode": args.mode})

    current_executor_params = None
    if args.mode in ("param-sweep", "all", "predictor-backtest"):
        bucket = config.get("signals_bucket", "alpha-engine-research")
        current_executor_params = read_params_pit_or_current(
            executor_optimizer, bucket, config,
        )

    # ── Simulate + param sweep + executor optimizer ───────────────────────
    if args.mode in ("simulate", "param-sweep", "all"):
        with registry.phase("simulation_pipeline", mode=args.mode):
            portfolio_stats, sweep_df, executor_rec = _run_simulation_pipeline(
                args, config, _sim_setup, current_executor_params, fd,
            )

    # ── Predictor backtest ────────────────────────────────────────────────
    # The ~121-min long pole. Honors an explicit skip (L4527 skip/resume): on
    # `--skip-phases=predictor_pipeline` a targeted recovery RELOADS the
    # predictor artifacts from S3 instead of redundantly re-running the whole
    # pipeline to reach a downstream stage (supersedes L4519). A normal run
    # (ctx not skipped) is unchanged. executor_rec is left intact on skip — the
    # skipped pipeline's recommendation isn't recomputed; the existing value
    # (from the simulate path / prior) carries forward.
    if args.mode in ("predictor-backtest", "all"):
        with registry.phase("predictor_pipeline", mode=args.mode) as pp_ctx:
            if pp_ctx.skipped:
                predictor_stats, predictor_sweep_df = _load_predictor_artifacts(
                    config, args.date,
                )
            else:
                predictor_stats, predictor_sweep_df, executor_rec = _run_predictor_pipeline(
                    args, config, executor_rec, current_executor_params, fd,
                )

    # ── Portfolio-optimizer cutover gate ──────────────────────────────────
    # ROADMAP L2222 PR 4.5. Runs the constrained MVO optimizer over synthetic
    # predictor history + persists a per-run gate report. Saturday SF reads
    # the JSON for PR 5 cutover-readiness signal. Non-fatal — gate failures
    # are observability, not a backtester-pipeline blocker.
    if args.mode in ("portfolio-optimizer-backtest", "all"):
        try:
            with registry.phase("portfolio_optimizer_gate", mode=args.mode):
                # Pass simulate-phase portfolio_stats as legacy_metrics when
                # available so the gate can evaluate legacy-relative criteria
                # (sortino_min, max_drawdown_floor, cvar_95_floor, turnover_max).
                # On mode=portfolio-optimizer-backtest (standalone), portfolio_stats
                # is None and the gate reports skipped verdicts for those.
                run_portfolio_optimizer_gate(
                    config=config,
                    run_date=args.date,
                    legacy_metrics=portfolio_stats,
                    signal_source=getattr(args, "signal_source", "synthetic"),
                )
        except Exception as exc:
            logger.warning(
                "portfolio_optimizer_gate phase failed (non-fatal): %s",
                exc, exc_info=True,
            )
            if fd:
                fd.report(exc, severity="warning", context={
                    "site": "portfolio_optimizer_gate", "mode": args.mode})

    # ── Covariance-estimator sweep (A.4) ──────────────────────────────────
    # ROADMAP A.4b — runs the 8-cell LW/OAS/EWMA × H × λ sweep over the
    # SAME predictor backtest history the optimizer gate above consumes
    # (a separate run_predictor_pipeline call inside the stage; ~+4 min
    # bounded). Persists the verdict to s3://{bucket}/backtest/{date}/
    # cov_sweep.json so the operator + Saturday SF Backtester report
    # can read the winner without an on-demand dispatch.
    #
    # Non-fatal — sweep failures are observability, not a backtester-
    # pipeline blocker. The flip from baseline cov estimator to the
    # sweep winner is operator-gated (see ROADMAP A.4 cutover verdict
    # gate: PSR ≥ 0.95, max_dd ≥ -0.35, CVaR ≥ -0.05, Sortino ≥
    # baseline × 0.9). This wiring emits the report; the flip itself is
    # a separate config edit in alpha-engine-config/executor/risk.yaml.
    if args.mode in ("portfolio-optimizer-backtest", "all"):
        try:
            with registry.phase("cov_estimator_sweep", mode=args.mode):
                run_cov_estimator_sweep_stage(
                    config=config,
                    run_date=args.date,
                )
        except Exception as exc:
            logger.warning(
                "cov_estimator_sweep phase failed (non-fatal): %s",
                exc, exc_info=True,
            )
            if fd:
                fd.report(exc, severity="warning", context={
                    "site": "cov_estimator_sweep", "mode": args.mode})

    # ── α̂-uncertainty γ-sweep (B.4) ──────────────────────────────────────
    # ROADMAP B.4b — runs the γ-sweep over the same predictor backtest
    # history the optimizer gate above consumes (a separate
    # run_predictor_pipeline call inside the stage; ~+4 min bounded),
    # augmented with σ_α̂ loaded from the production predictions
    # archive. Persists verdict to s3://{bucket}/backtest/{date}/
    # gamma_sweep.json.
    #
    # Auto-skips when σ_α̂ coverage is insufficient (i.e. before
    # predictor B.1 has accumulated enough Saturday cycles emitting
    # non-None predicted_alpha_std). Activates automatically once
    # coverage clears the _GAMMA_SWEEP_MIN_DATE_COVERAGE threshold —
    # no operator flip needed.
    #
    # Non-fatal — sweep failures are observability, not a backtester-
    # pipeline blocker. The flip from γ=0 (baseline MVO) to the
    # sweep winner is operator-gated (see ROADMAP B.5 cutover gate:
    # PSR ≥ 0.95, max_dd ≥ -0.35, CVaR ≥ -0.05, Sortino ≥ baseline ×
    # 0.9, AND ≥2-3 trading days of shadow-log ablation deltas AND
    # operator explicit go-ahead).
    if args.mode in ("portfolio-optimizer-backtest", "all"):
        try:
            with registry.phase("gamma_sweep", mode=args.mode):
                run_gamma_sweep_stage(
                    config=config,
                    run_date=args.date,
                )
        except Exception as exc:
            logger.warning(
                "gamma_sweep phase failed (non-fatal): %s",
                exc, exc_info=True,
            )
            if fd:
                fd.report(exc, severity="warning", context={
                    "site": "gamma_sweep", "mode": args.mode})

    # ── Export simulation artifacts for evaluator ────────────────────────
    if args.mode in ("simulate", "param-sweep", "all", "predictor-backtest"):
        try:
            with registry.phase("export_artifacts", mode=args.mode):
                _export_simulation_artifacts(config, args.date, sweep_df=sweep_df, predictor_sweep_df=predictor_sweep_df, portfolio_stats=portfolio_stats, predictor_stats=predictor_stats)
        except Exception as e:
            logger.warning("Simulation artifact export failed (non-fatal): %s", e)

        # Outcome taxonomy guard (L4523 — encodes ARCHITECTURE.md §22). The
        # Evaluator treats portfolio_stats.json + sweep_df.parquet as CRITICAL.
        # When a mode is EXPECTED to produce them (simulate / param-sweep / all),
        # classify_simulation_outcome() resolves the 3-way outcome and we act on
        # it: FAILURE (ABSENT — phase didn't run / errored) → raise (the L4513
        # silent-starvation failure mode); EMPTY (ran, no admissible combo) →
        # valid no-op WARN+alert, do NOT crash (a risk/score gate gating out all
        # entries must NOT kill the process — the 2026-06-06 symptom); SUCCESS →
        # proceed. predictor-backtest legitimately produces neither → exempt.
        if args.mode in ("simulate", "param-sweep", "all"):
            outcome = classify_simulation_outcome(args.mode, portfolio_stats, sweep_df)
            logger.info("[outcome] simulate-phase outcome: %s", outcome.to_dict())
            if outcome.is_failure:
                raise RuntimeError(outcome.reason)
            if outcome.is_empty:
                # EMPTY sweep = ran, no admissible combo. VALID no-op: the empty
                # parquet is written by _export_simulation_artifacts so the
                # Evaluator finds it present + no-ops the optimizer. Surface
                # LOUDLY (never silent) so a suspicious degeneracy (e.g.
                # zero-score inputs) is visible; the L4525 input-quality HARD
                # gate is the follow-up that classifies legit-vs-garbage.
                logger.warning("%s", outcome.reason)
                try:
                    import alpha_engine_lib.alerts as _alerts
                    _alerts.publish(
                        message=(
                            f"Backtester empty param-sweep (mode={args.mode}, "
                            f"date={args.date}): no admissible combo; configs held, "
                            f"optimizer skipped. Verify not a zero-score-input issue "
                            f"(L4525)."
                        ),
                        severity="warning",
                        source="alpha-engine-backtester/backtest.py",
                    )
                except Exception:  # noqa: BLE001 — alert is best-effort observability
                    pass

    # ── Report, upload, email, and instance stop ──────────────────────────
    # Wrapped in try/finally so --stop-instance ALWAYS runs.
    try:
        pipeline_health = {
            "db_pull_status": config.get("_db_pull_status"),
            "staleness_warning": portfolio_stats.get("staleness_warning") if portfolio_stats else None,
            "coverage": portfolio_stats.get("coverage") if portfolio_stats else None,
            "dates_simulated": portfolio_stats.get("dates_simulated") if portfolio_stats else None,
            "dates_expected": portfolio_stats.get("dates_expected") if portfolio_stats else None,
            "skip_reasons": portfolio_stats.get("skip_reasons") if portfolio_stats else None,
            "price_gap_warnings": portfolio_stats.get("price_gap_warnings") if portfolio_stats else None,
            "unfilled_gaps": portfolio_stats.get("unfilled_gaps") if portfolio_stats else None,
            "feature_skip_reasons": predictor_stats.get("skip_reasons") if predictor_stats else None,
        }

        # Deployed-strategy headline (config#1053): production research signals
        # through the daily MVO optimizer — the system as it actually trades.
        # FAIL LOUD by construction: a non-"ok" return makes the reporter render
        # a prominent banner instead of letting a component number headline.
        production_stats = run_production_strategy_backtest(config)
        logger.info(
            "production-strategy backtest for report: status=%s",
            production_stats.get("status"),
        )

        # Optimizer-param sweep recommendation (config#1057) — observe-only,
        # best-effort: a failure must not break the report. Surfaced under the
        # deployed headline so the operator sees the recommended risk_aversion ×
        # tcost_bps cell. Auto-apply is increment 2.
        try:
            optimizer_param_sweep = run_optimizer_param_sweep_stage(config, args.date)
        except Exception as e:  # noqa: BLE001 — advisory; report must still ship
            logger.warning("optimizer_param_sweep stage failed (non-fatal): %s", e)
            optimizer_param_sweep = {"status": "error", "reason": str(e)}

        # Eval-only kwargs (weight_result, veto_result, grading, etc.) default
        # to None in build_report — they are populated by evaluate.py, not here.
        report_md = build_report(
            run_date=args.date,
            signal_quality={"status": "skipped"},
            regime_analysis=[],
            score_analysis=[],
            attribution={"status": "skipped"},
            portfolio_stats=portfolio_stats,
            production_stats=production_stats,
            optimizer_param_sweep=optimizer_param_sweep,
            sweep_df=sweep_df,
            config=config,
            predictor_stats=predictor_stats,
            predictor_sweep_df=predictor_sweep_df,
            executor_rec=executor_rec,
            pipeline_health=pipeline_health,
        )

        save_sweep_df = sweep_df
        if predictor_sweep_df is not None and not predictor_sweep_df.empty:
            save_sweep_df = predictor_sweep_df

        out_dir = save(
            report_md=report_md,
            signal_quality={"status": "skipped"},
            score_analysis=[],
            sweep_df=save_sweep_df,
            run_date=args.date,
            results_dir=config.get("results_dir", "results"),
        )

        print(f"\nReport saved to {out_dir}/")
        print(f"\n{'='*60}")
        print(report_md[:2000])
        if len(report_md) > 2000:
            print(f"\n... (truncated — see {out_dir}/report.md for full report)")

        if args.upload:
            upload_to_s3(
                local_dir=out_dir,
                bucket=config.get("output_bucket", "alpha-engine-research"),
                prefix=config.get("output_prefix", "backtest"),
                run_date=args.date,
            )
            print(f"\nUploaded to s3://{config.get('output_bucket')}/{config.get('output_prefix')}/{args.date}/")

        # Suppress email for smoke-phase runs and any run with --freeze
        # set (freeze signals "don't promote / don't notify"; smoke-phase
        # modes are test invocations with synthetic fixtures and their
        # reports would pollute the operator inbox + risk being confused
        # with real Saturday SF emails). Detection uses _is_smoke_phase
        # (captured at main() entry, before args.mode was rewritten to
        # the routed full mode) and args.freeze.
        suppress_email = _is_smoke_phase or args.freeze
        sender = config.get("email_sender")
        recipients = config.get("email_recipients", [])
        if suppress_email:
            logger.info(
                "Email suppressed (mode=%s, freeze=%s) — skipping report email",
                _original_mode, args.freeze,
            )
        elif sender and recipients:
            send_report_email(
                run_date=args.date,
                report_md=report_md,
                status="simulation",
                sender=sender,
                recipients=recipients,
                s3_bucket=config.get("output_bucket") if args.upload else None,
                s3_prefix=config.get("output_prefix", "backtest"),
            )
        else:
            logger.warning("No email_sender/email_recipients in config — skipping email")
    except Exception as e:
        logger.error("Report/upload/email failed: %s", e)
        if fd:
            fd.report(e, severity="critical", context={
                "site": "report_upload_email",
                "mode": args.mode,
                "run_date": args.date,
                "upload": args.upload,
            })
    finally:
        try:
            from health_status import write_health
            configs_applied = []
            if executor_rec and executor_rec.get("apply_result", {}).get("applied"):
                configs_applied.append("executor_params")
            bucket = config.get("signals_bucket", "alpha-engine-research")
            write_health(
                bucket=bucket,
                module_name="backtester",
                status="ok",
                run_date=args.date,
                duration_seconds=_time.time() - _health_start,
                summary={
                    "mode": args.mode,
                    "configs_applied": configs_applied,
                },
            )
        except Exception as _he:
            logger.warning("Health status write failed: %s", _he)

        if args.stop_instance:
            _stop_ec2_instance()

        # Smoke budget enforcement runs LAST, after health write +
        # instance stop, so the stop side effect still fires even if the
        # smoke blew past its budget. Budget failure is a hard exit 2 —
        # catches regressions at smoke time (seconds) instead of during
        # a 2h Saturday SF run. The registry is passed so the check can
        # also fail on inner-phase errors swallowed by outer try/except
        # (false-PASS guard — see 2026-04-23 post-filter dry-run). Per
        # ROADMAP Backtester P0 #3.
        if _is_smoke_phase:
            elapsed = _time.time() - _health_start
            _assert_smoke_within_budget(_original_mode, elapsed, registry=registry)


if __name__ == "__main__":
    main()
