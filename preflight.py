"""
Backtester preflight: connectivity + freshness checks run at the top of
each entrypoint before any real work starts.

Primitives live in ``alpha_engine_lib.preflight.BasePreflight``; this
module only composes them into a mode-specific sequence. See the
alpha-engine-lib README for the 2026-04-14 data-path failure mode that
motivated the library.

Modes:

- ``"backtest"`` — ``backtest.py`` entrypoint (weekly spot instance).
  Verifies that every module the full backtest call chain will import
  is actually importable now (imports, lib version, predictor weights)
  + S3 bucket reachable + ArcticDB ``macro/SPY`` fresh + the executor
  risk.yaml the simulate path will import resolves to a real config
  (not the placeholder .example template). 8-day threshold covers
  Fri→Mon weekly cadence + buffer.

  2026-04-21 incident motivated the import/version/weights additions:
  a Saturday SF dry-run burned ~80 minutes of c5.large compute before
  failing on ``No module named 'alpha_engine_lib.arcticdb'`` deep in
  ``_run_simulation_loop``. The three new preflight checks below all
  run in <2 seconds and would have caught the same bug at startup.
- ``"evaluate"`` — ``evaluate.py`` entrypoint. Reads simulation
  artifacts from S3 only, no ArcticDB. Keep the check cheap.
- ``"lambda_health"`` — daily predictor health check Lambda. Reads
  ``research.db`` + per-day metrics from S3. No ArcticDB.
"""

from __future__ import annotations

import importlib
import os

import yaml

from alpha_engine_lib.preflight import BasePreflight


# Placeholder prefix convention used by every repo's *.yaml.example
# template. A bucket/path value starting with this is definitionally a
# not-filled-in config and must never reach a live S3/ArcticDB read.
_PLACEHOLDER_PREFIX = "your-"

# Minimum alpha-engine-lib version the backtester depends on at runtime.
# Keep in sync with the ``@vX.Y.Z`` pin in ``requirements.txt``. Bump when
# a new symbol from the lib is imported by any backtest call path.
#
# Current floor: 0.1.4 — introduces ``alpha_engine_lib.arcticdb`` which
# ``backtest._run_simulation_loop`` depends on to filter historical
# signals against the current universe.
MIN_LIB_VERSION = "0.1.4"

# Modules whose imports are load-bearing for the backtest modes. Any
# missing or non-importable entry here would surface deep in the call
# chain; listing them explicitly here makes the failure show up in
# seconds at preflight instead of ~80 minutes into a spot run.
_CRITICAL_IMPORTS_BACKTEST = (
    # alpha-engine-lib submodules we directly call
    "alpha_engine_lib.arcticdb",
    "alpha_engine_lib.logging",
    "alpha_engine_lib.preflight",
    # executor modules — simulate path
    "executor.main",
    "executor.ibkr",
    # synthetic — predictor-backtest mode (10y GBM replay)
    "synthetic.predictor_backtest",
    # predictor model — loaded inside download_gbm_model
    "model.gbm_scorer",
)

# S3 keys the backtester's predictor-backtest mode HEADs before spending
# 10y × ~900 tickers on GBM inference. Missing means PredictorTraining
# has not populated the Layer-1A weights — investigate there.
_REQUIRED_PREDICTOR_WEIGHTS = (
    "predictor/weights/meta/momentum_model.txt",
    "predictor/weights/meta/momentum_model.txt.meta.json",
)

# Backtester-local modules that must be pre-imported before sibling repo
# paths land on sys.path. Rationale: the predictor repo ships its own
# ``store.arctic_reader`` with a different API; once predictor_path is
# at sys.path[0], ``from store.arctic_reader import load_universe_from_arctic``
# (in backtester's ``synthetic/predictor_backtest.py``) resolves to the
# wrong module and ImportErrors. In production these modules load via
# backtester's top-level imports BEFORE predictor_path is inserted;
# preflight runs before those top-level imports so we eagerly load
# them here to match the production sys.modules cache ordering.
_LOCAL_PREIMPORTS_BACKTEST = (
    "store.arctic_reader",
)



class BacktesterPreflight(BasePreflight):
    """Preflight checks for the three backtester entrypoints."""

    def __init__(
        self,
        bucket: str,
        mode: str,
        executor_paths: list[str] | None = None,
        predictor_paths: list[str] | None = None,
    ):
        super().__init__(bucket)
        if mode not in ("backtest", "evaluate", "lambda_health"):
            raise ValueError(f"BacktesterPreflight: unknown mode {mode!r}")
        self.mode = mode
        # backtest.py passes config["executor_paths"] + config["predictor_paths"]
        # here so preflight can (a) validate executor's risk.yaml will load with
        # real values, and (b) add both repo roots to sys.path before
        # _check_imports so ``from executor.main`` / ``from model.gbm_scorer``
        # resolve the same way they do in ``_setup_simulation`` later. Without
        # the sys.path inserts, _check_imports fires ModuleNotFoundError on
        # ``executor``/``model`` even when everything is set up correctly.
        self.executor_paths = executor_paths or []
        self.predictor_paths = predictor_paths or []

    def run(self) -> None:
        self.check_env_vars("AWS_REGION")
        self.check_s3_bucket()

        if self.mode == "backtest":
            # Cross-stage artifact contract FIRST — pure + instant (no I/O):
            # the declarative pipeline_manifest asserts that the mode the SF
            # Backtester state runs produces every Evaluator-critical artifact.
            # Enforced at CI by test_evaluator_artifact_contract, but a
            # topology/mode drift that reaches a live host (e.g. a hand-edited
            # SF mode) must also fail HERE, in microseconds, before any spend —
            # not 2h in when the Evaluator starves (the L4513 class). L4526
            # plan §6 Phase 4 (compose the preflight on the manifest).
            self._check_artifact_contract()
            # Environment checks next — cheapest I/O-free-ish, and catch the
            # class of failure where the spot's pip install didn't pull the
            # pin we expected. ~2 seconds total. All three would have
            # caught the 2026-04-21 80-minute burn.
            self._check_lib_version()
            self._check_imports()
            # Schema bridge between scalar signal-generation contract and
            # the vectorized engine's signal-consumption contract. Cheap
            # (~50 ms): builds a synthetic envelope, runs through
            # `_build_signal_lookup` + `extract_signal_arrays`, asserts
            # non-zero output. Catches the class of bug shipped by
            # 2026-04-27 PR #114 + caught by 2026-04-28 Layer 3 v14
            # (vectorized read `signals_raw_filtered.get("enter")`
            # directly; synthetic envelope has no top-level `enter` key,
            # so 0 orders across 60 combos × 2500 dates).
            self._check_vectorized_signal_extraction()
            self._check_predictor_weights()
            # Data-freshness assertions (universe + macro/SPY) live upstream
            # in alpha-engine-data's preflight, which runs as Saturday SF's
            # DataPhase1 step before the backtester step. If upstream data
            # is stale, the data step hard-fails and the SF never reaches
            # the backtester — re-checking here was redundant.
            # backtest.py's simulate path imports executor.main, which
            # in turn imports executor.config_loader and reads the
            # executor's risk.yaml. If it resolves to the placeholder
            # .example template (or to a file with "your-*" bucket
            # names), every downstream S3/ArcticDB read fails deep in
            # the executor-sim call chain. Caught at preflight so the
            # operator sees the real cause in <1s. Hit 2026-04-20.
            self._check_executor_config()

    # ── Pipeline-contract preflight (L4526, plan §6 Phase 4) ─────────────

    def _check_artifact_contract(self) -> None:
        """Fail if the SF Backtester mode wouldn't produce an Evaluator-critical
        artifact, per the declarative ``pipeline_manifest``.

        Pure + instant (no I/O). The same contract the CI test
        (``test_evaluator_artifact_contract``) locks — re-checked at runtime so
        a drift that reached this host (a hand-edited SF mode, a phase-gate edit
        deployed without the test) fails in microseconds at the gate instead of
        2h in when the Evaluator starves (the L4513 silent-starvation class).
        """
        import pipeline_manifest as manifest

        violations = manifest.contract_violations(manifest.SF_BACKTESTER_MODE)
        if violations:
            raise RuntimeError(
                "Pre-flight: pipeline artifact-contract violation for "
                f"--mode={manifest.SF_BACKTESTER_MODE} — the SF Backtester state "
                "would not produce an Evaluator-critical artifact, starving the "
                "Evaluator (L4513 class). Fix pipeline_manifest.py / the SF mode "
                "/ the producer mode-gate:\n  - " + "\n  - ".join(violations)
            )

    # ── Environment primitives (added 2026-04-21 post-80min-burn) ────────

    def _check_lib_version(self) -> None:
        """Fail if the installed alpha_engine_lib is older than the
        minimum the backtester's call chain needs.

        Triggers when the spot's pip install silently fell back to a
        cached older version (or when requirements.txt was bumped but
        MIN_LIB_VERSION here wasn't — same bug in the other direction).
        """
        import alpha_engine_lib
        from packaging.version import Version

        installed = getattr(alpha_engine_lib, "__version__", None)
        if not installed:
            raise RuntimeError(
                "Pre-flight: alpha_engine_lib has no __version__ "
                "attribute — likely a broken install. Re-run the pip "
                "install step on this host."
            )
        if Version(installed) < Version(MIN_LIB_VERSION):
            raise RuntimeError(
                f"Pre-flight: alpha_engine_lib {installed} < required "
                f"{MIN_LIB_VERSION}. Spot's pip install may have pulled "
                "a stale cached version, or requirements.txt drifted "
                "from MIN_LIB_VERSION in preflight.py. The 2026-04-21 "
                "Saturday SF dry-run burned ~80 min on this exact class "
                "of failure before surfacing a deep-call-chain import "
                "error — this check catches it at startup."
            )

    def _check_imports(self) -> None:
        """Actually import every module the deep call chain relies on.

        A Python ImportError from inside ``_run_simulation_loop`` (or
        any of the other deep-call-stack sites) takes minutes-to-hours
        of spot time to surface because nothing before that point tries
        to import the module. Surfacing it at preflight is worth the
        ~1 second of extra import cost at startup.

        ``executor.main`` / ``executor.ibkr`` / ``model.gbm_scorer`` /
        ``synthetic.predictor_backtest`` are only importable once the
        alpha-engine + alpha-engine-predictor repo roots are on
        ``sys.path`` — normally done inside ``backtest._setup_simulation``.
        Preflight does the same inserts first so the import check
        matches production import resolution.

        **Sibling-repo collision defense:** the backtester and predictor
        both ship a ``store/arctic_reader.py`` with different APIs (the
        backtester's has ``load_universe_from_arctic``; the predictor's
        does not). Once predictor_path lands at sys.path[0], Python
        resolves ``store.arctic_reader`` to the predictor version and
        ``synthetic.predictor_backtest``'s top-level import of
        ``load_universe_from_arctic`` fails. In production this
        doesn't bite because backtester's top-level modules have
        already loaded (and cached in sys.modules) before predictor_path
        is prepended. Preflight runs before any top-level backtester
        imports, so we eagerly pre-load the local ``store`` modules
        here to match that production ordering.

        Mode-specific list: only ``backtest`` mode imports the executor
        and predictor repos. ``evaluate`` / ``lambda_health`` have
        their own narrower call chains (validated by their own preflight
        branches as needed).
        """
        import sys

        # Pre-import backtester-local modules that have same-name
        # siblings in executor/predictor repos. Cache wins regardless
        # of subsequent sys.path insert order.
        for local in _LOCAL_PREIMPORTS_BACKTEST:
            try:
                importlib.import_module(local)
            except ImportError as exc:
                raise RuntimeError(
                    f"Pre-flight: could not import local module "
                    f"{local!r} — backtester's own code is broken. "
                    f"Underlying error: {exc}"
                ) from exc

        for candidates in (self.executor_paths, self.predictor_paths):
            for p in candidates:
                if os.path.isdir(p) and p not in sys.path:
                    sys.path.insert(0, p)
                    break  # first hit wins, matches backtest.py behavior

        for name in _CRITICAL_IMPORTS_BACKTEST:
            try:
                importlib.import_module(name)
            except ImportError as exc:
                raise RuntimeError(
                    f"Pre-flight: could not import {name!r} — would "
                    "have crashed deep in the backtest call chain. "
                    "Check requirements.txt pin + that pip install "
                    "completed successfully on this host. If "
                    "executor/predictor imports fail, check config.yaml "
                    "``executor_paths`` / ``predictor_paths`` resolve "
                    f"to real directories on this host. Underlying "
                    f"error: {exc}"
                ) from exc

    def _check_vectorized_signal_extraction(self) -> None:
        """Pin the schema bridge between scalar signal generation and
        the vectorized sweep's signal consumption.

        Background: the scalar path runs each per-date envelope through
        ``executor.signal_reader.get_actionable_signals``, which segments
        ``buy_candidates`` + ``universe`` (each carrying a per-entry
        ``signal`` field) into top-level ``enter`` / ``exit`` / ``reduce``
        / ``hold`` lists. The vectorized engine reads
        ``signal_lookup.actionable.get("enter")`` — populated once per
        date in ``_build_signal_lookup``. If either side regresses
        (vectorized starts reading a non-actionable key, or
        ``_build_signal_lookup`` stops populating ``actionable``), the
        sweep silently produces 0 orders across every combo.

        2026-04-28 Layer 3 v14 caught this: vectorized read
        ``signals_raw_filtered.get("enter")`` directly, the synthetic
        envelope has no top-level ``enter`` key, sweep emitted 0 orders
        × 60 combos × 2500 dates. 90 minutes of c5.large spot burned to
        find a missing dictionary key. This check runs the entire
        translation in ~50 ms against a known-shape envelope and fails
        loud at startup if the contract drifts again.

        Cost: ~50 ms. Catches: schema-drift between the two paths;
        regressions in ``_build_signal_lookup`` (missing actionable
        field, wrong key); regressions in
        ``synthetic.vectorized_sweep.extract_signal_arrays`` (reads
        wrong attribute, signal-shape changes).
        """
        from backtest import _build_signal_lookup
        from synthetic.vectorized_sweep import (
            extract_signal_arrays,
            extract_research_actions,
        )
        import numpy as np

        # Mirror exactly what `synthetic.signal_generator.predictions_to_signals`
        # emits: top-level keys are date / market_regime / sector_ratings /
        # buy_candidates / universe — NO top-level enter / exit / reduce /
        # hold lists. Per-entry `signal` field carries the segmentation.
        synthetic_envelope = {
            "date": "2026-01-01",
            "market_regime": "neutral",
            "sector_ratings": {"Technology": {"rating": "market_weight"}},
            "buy_candidates": [
                {"ticker": "AAPL", "score": 85, "signal": "ENTER",
                 "conviction": "rising", "sector": "Technology",
                 "rating": "BUY"},
                {"ticker": "MSFT", "score": 78, "signal": "ENTER",
                 "conviction": "stable", "sector": "Technology",
                 "rating": "BUY"},
            ],
            "universe": [
                {"ticker": "JPM", "score": 50, "signal": "HOLD",
                 "conviction": "stable", "sector": "Financial",
                 "rating": "HOLD"},
                {"ticker": "BAC", "score": 25, "signal": "EXIT",
                 "conviction": "declining", "sector": "Financial",
                 "rating": "SELL"},
            ],
        }

        try:
            lookup = _build_signal_lookup(synthetic_envelope)
        except Exception as exc:
            raise RuntimeError(
                f"Pre-flight: _build_signal_lookup raised on a known-"
                f"shape synthetic envelope. The signal-precompute path "
                f"that feeds both scalar and vectorized sweeps is "
                f"broken. Underlying error: {exc}"
            ) from exc

        if not hasattr(lookup, "actionable"):
            raise RuntimeError(
                "Pre-flight: SignalLookup is missing the `actionable` "
                "field. The vectorized sweep depends on it (see "
                "synthetic/vectorized_sweep.py:465 / 574). If "
                "`actionable` was removed, the vectorized path will "
                "silently emit 0 orders — same failure mode as Layer "
                "3 v14 (2026-04-28)."
            )

        enter_list = lookup.actionable.get("enter", [])
        if not enter_list:
            raise RuntimeError(
                "Pre-flight: lookup.actionable['enter'] is empty for a "
                "synthetic envelope with 2 buy_candidates carrying "
                "signal=ENTER. The actionable transformation in "
                "_build_signal_lookup is broken or bypassed. Verify "
                "`get_actionable_signals` is being called and is "
                "iterating buy_candidates + universe correctly."
            )

        # End-to-end gate: drive the vectorized signal extractor with
        # the same lookup the sweep would feed it. Any regression in
        # how `extract_signal_arrays` reads from the lookup surfaces
        # here as size==0.
        ticker_to_idx = {"AAPL": 0, "MSFT": 1, "JPM": 2, "BAC": 3}
        try:
            sig_arrays = extract_signal_arrays(
                lookup,
                predictions={},
                ticker_to_idx=ticker_to_idx,
                sector_label_to_idx={"Technology": 0, "Financial": 1},
                atr_pct_by_ticker={},
                coverage_by_ticker={},
                earnings_by_ticker={},
                momentum_at_date_per_ticker=np.zeros(len(ticker_to_idx)),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Pre-flight: synthetic.vectorized_sweep."
                f"extract_signal_arrays raised on a valid SignalLookup. "
                f"Vectorized sweep would crash on every signal date. "
                f"Underlying error: {exc}"
            ) from exc

        if sig_arrays["signal_ticker_idx"].size == 0:
            raise RuntimeError(
                "Pre-flight: extract_signal_arrays returned 0 enter "
                "signals despite the lookup carrying 2 entries in "
                "actionable['enter']. The vectorized engine is reading "
                "from the wrong attribute (regressed from `actionable` "
                "back to `signals_raw_filtered`?). See the 2026-04-28 "
                "v14 incident — this is the exact failure mode that "
                "produced 0 orders × 60 combos × 2500 dates."
            )

        # Spot-check extract_research_actions too — same schema bridge,
        # different vectorized read site.
        try:
            actions = extract_research_actions(
                lookup, ticker_to_idx, n_tickers=4,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Pre-flight: extract_research_actions raised. "
                f"Vectorized sweep would crash. Error: {exc}"
            ) from exc

        # AAPL + MSFT must register as ENTER, BAC as EXIT — anything
        # else means the action-segmentation regressed.
        from synthetic.vectorized_exits import RA_ENTER, RA_EXIT
        if actions[ticker_to_idx["AAPL"]] != RA_ENTER:
            raise RuntimeError(
                "Pre-flight: extract_research_actions did not flag "
                "AAPL as RA_ENTER on a synthetic envelope where AAPL "
                "carries signal=ENTER. Action segmentation regressed."
            )
        if actions[ticker_to_idx["BAC"]] != RA_EXIT:
            raise RuntimeError(
                "Pre-flight: extract_research_actions did not flag "
                "BAC as RA_EXIT on a synthetic envelope where BAC "
                "carries signal=EXIT. Action segmentation regressed."
            )

    def _check_predictor_weights(self) -> None:
        """S3 HEAD on the Layer-1A momentum GBM weights + metadata.

        ``synthetic/predictor_backtest.py::download_gbm_model`` reads
        these keys near the start of predictor-backtest mode. If they
        don't exist, fail now (seconds) instead of after the
        universe-data load (minutes) and report the named upstream
        owner in the error.
        """
        import boto3
        s3 = boto3.client("s3", region_name=self.region)
        for key in _REQUIRED_PREDICTOR_WEIGHTS:
            try:
                s3.head_object(Bucket=self.bucket, Key=key)
            except Exception as exc:
                raise RuntimeError(
                    f"Pre-flight: required key s3://{self.bucket}/{key} "
                    "is missing or unreadable. The Layer-1A momentum "
                    "GBM backtest requires this file; Saturday SF's "
                    "PredictorTraining step must populate "
                    "predictor/weights/meta/momentum_model.txt every "
                    f"run — investigate there. Underlying error: {exc}"
                ) from exc

    # ── Mode-specific primitives ─────────────────────────────────────────

    def _check_executor_config(self) -> None:
        """Validate the executor risk.yaml the simulate path will load.

        Mirrors executor/config_loader.py's canonical search order,
        minus the removed `.example` fallback (alpha-engine#73). Fails
        if no real risk.yaml is reachable, or if the loaded config
        carries placeholder bucket values, or if the executor's
        signals_bucket disagrees with the backtester's (both must read
        the same bucket or the backtest measures data-source drift
        instead of logic drift).
        """
        # If the caller didn't give us an executor repo root, skip —
        # executor's own import-time config_loader now hard-fails on
        # miss (alpha-engine#73), so this is a defense-in-depth check
        # rather than the sole safeguard.
        executor_root = next(
            (p for p in self.executor_paths if os.path.isdir(p)),
            None,
        )
        if executor_root is None:
            return

        # Experiment-package first (config#1042): executor risk.yaml resolves
        # from experiments/$ALPHA_ENGINE_EXPERIMENT_ID/executor/risk.yaml
        # (default experiment `reference`) ahead of the legacy top-level
        # alpha-engine-config/executor/risk.yaml, then the repo-local fallback.
        # Mirrors pipeline_common.load_config's precedence. Behavior-preserving:
        # config#1159 made the package copy byte-identical to legacy.
        exp = os.environ.get("ALPHA_ENGINE_EXPERIMENT_ID", "reference")
        candidate_paths = [
            os.path.expanduser(f"~/alpha-engine-config/experiments/{exp}/executor/risk.yaml"),
            os.path.realpath(
                os.path.join(
                    executor_root, "..", "alpha-engine-config",
                    "experiments", exp, "executor", "risk.yaml",
                )
            ),
            os.path.expanduser("~/alpha-engine-config/executor/risk.yaml"),
            os.path.realpath(
                os.path.join(executor_root, "..", "alpha-engine-config", "executor", "risk.yaml")
            ),
            os.path.realpath(os.path.join(executor_root, "config", "risk.yaml")),
        ]
        resolved = next((p for p in candidate_paths if os.path.isfile(p)), None)
        if resolved is None:
            raise RuntimeError(
                "Pre-flight: executor risk.yaml not found in any of:\n  "
                + "\n  ".join(candidate_paths)
                + "\nBacktester simulate path will hard-fail on import. Clone "
                  "alpha-engine-config next to the alpha-engine repo, or populate "
                  "alpha-engine/config/risk.yaml from the .example template. The "
                  ".example is intentionally NOT a fallback (see alpha-engine#73)."
            )

        try:
            with open(resolved) as f:
                loaded = yaml.safe_load(f) or {}
        except Exception as exc:
            raise RuntimeError(
                f"Pre-flight: executor risk.yaml at {resolved} failed to parse: {exc}"
            ) from exc

        for key in ("signals_bucket", "trades_bucket"):
            value = loaded.get(key)
            if not isinstance(value, str) or not value:
                raise RuntimeError(
                    f"Pre-flight: executor risk.yaml at {resolved} is missing required "
                    f"key {key!r} or has an empty value."
                )
            if value.startswith(_PLACEHOLDER_PREFIX):
                raise RuntimeError(
                    f"Pre-flight: executor risk.yaml at {resolved} has placeholder "
                    f"{key}={value!r}. This is the .example template (or a copy that "
                    "wasn't filled in). Downstream ArcticDB/S3 reads would hit "
                    "nonexistent buckets — matches the 2026-04-20 KeyNotFoundException "
                    "incident."
                )

        executor_signals_bucket = loaded["signals_bucket"]
        if executor_signals_bucket != self.bucket:
            raise RuntimeError(
                f"Pre-flight: executor signals_bucket={executor_signals_bucket!r} does "
                f"not match backtester signals_bucket={self.bucket!r}. Simulate mode "
                "replays archived signals through the executor — both must read from "
                "the same S3 bucket or the backtest measures data-source drift instead "
                "of logic drift."
            )
