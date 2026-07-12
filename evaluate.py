"""
evaluate.py — CLI entry point for the Alpha Engine evaluator.

Runs all evaluation modules (signal quality analysis, component diagnostics,
self-adjustment optimizers) independently of simulation. Reads research.db
and trades.db directly; optionally reads simulation artifacts from S3
(sweep_df, portfolio_stats) if available.

Each module reports its data completeness — whether it had all inputs or
ran in degraded mode. The completeness manifest is saved alongside the
evaluation report.

Usage:
    python evaluate.py --mode all                     # run everything
    python evaluate.py --mode diagnostics             # analysis modules only (no config promotion)
    python evaluate.py --mode optimize                # optimizer modules only
    python evaluate.py --module signal-quality        # single module
    python evaluate.py --upload --freeze              # upload results, skip S3 config writes
    python evaluate.py --db /path/to/research.db      # local DB override
    python evaluate.py --trades-db /path/to/trades.db # local trades.db override
"""

from __future__ import annotations

# arcticdb MUST import before pyarrow/pandas. Both bundle the AWS C SDK; on
# macOS whichever loads first wins the global symbol resolution, and if
# pyarrow's libarrow wins, arcticdb's S3 client binds incompatible aws_*
# symbols and aborts (allocator != NULL fatal) when it builds the S3 client.
# Importing arcticdb first makes its symbols win. No-op on Linux (where there
# is no clash) and harmless if arcticdb is absent (the diagnostics that need it
# fail-soft). Keep this ABOVE the pandas-pulling imports below.
try:  # noqa: SIM105
    import arcticdb  # noqa: F401
except Exception:  # pragma: no cover - arcticdb optional in some envs
    pass

import argparse
import io
import json
import logging
import os
import time as _time
from datetime import date, datetime, timezone
from pathlib import Path

# Structured logging + flow-doctor singleton via alpha-engine-lib (shared
# pattern across all 5 entrypoints; see executor/main.py for reference).
# Module-top so import-time errors in pandas / boto3 / analysis modules
# below are also captured by flow-doctor's ERROR handler. evaluate.py
# runs on EC2 spot via spot_backtest.sh; not in a Lambda image, so the
# simple repo-root path resolution works.
#
# exclude_patterns starts empty by deliberate convention.
from nousergon_lib.logging import setup_logging, guard_entrypoint
from nousergon_lib.quant.horizons import DEFAULT_POLICY as _HORIZON_POLICY
_FLOW_DOCTOR_EXCLUDE_PATTERNS: list[str] = []
_FLOW_DOCTOR_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow-doctor.yaml")
setup_logging(
    "evaluate",
    flow_doctor_yaml=_FLOW_DOCTOR_YAML,
    exclude_patterns=_FLOW_DOCTOR_EXCLUDE_PATTERNS,
)

import boto3
from botocore.exceptions import ClientError
import pandas as pd
import yaml

from analysis import signal_quality, regime_analysis, score_analysis, attribution
from analysis.sample_size_adequacy import compute_sample_size_adequacy
from analysis.risk_ratio_ci import compute_risk_ratio_ci
from analysis.action_entropy import compute_action_entropy_artifact
from analysis.optimizer_churn import compute_optimizer_churn
from analysis.walk_forward_stability import compute_walk_forward_stability
from analysis import factor_blend_sensitivity
from analysis import factor_blend_counterfactual_replay
from analysis import veto_analysis
from analysis import decision_capture_coverage, executor_decision_capture_coverage, measurement_coverage, provenance_grounding, quant_rank_quality
from analysis import cio_rule_tag_precision
from analysis import agent_justification
from analysis import end_to_end
from analysis import attractiveness_eval as attractiveness_eval_analysis
from analysis import trigger_scorecard, alpha_distribution, veto_value
from analysis import shadow_book as shadow_book_analysis
from analysis import behavioral_anomaly as behavioral_anomaly_analysis
from analysis import exit_timing, macro_eval
from analysis import regime_stratified_sortino_runner
from optimizer import weight_optimizer, executor_optimizer, research_optimizer
from optimizer import (
    trigger_optimizer, predictor_sizing_optimizer, barrier_sizing_optimizer,
    stance_sizing_optimizer,
)
from optimizer import scanner_optimizer, pipeline_optimizer, tech_weight_ablation
from optimizer import factor_blend_optimizer
from optimizer.config_archive import read_params_pit_or_current
from emailer import send_digest_email
from reporter import build_digest, build_report, save, upload_to_s3
from completeness import CompletenessTracker
from pipeline_common import (
    load_config,
    pull_research_db,
    init_research_db,
    find_trades_db,
    push_predictor_rolling_metrics,
    resolve_trading_day,
    _load_label_barrier_config,
)

logger = logging.getLogger(__name__)


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alpha Engine Evaluator")
    parser.add_argument(
        "--mode", choices=["all", "diagnostics", "optimize"],
        default="all",
        help="all: run everything. diagnostics: analysis only. optimize: optimizers only.",
    )
    parser.add_argument(
        "--module",
        help="Run a single named module (e.g., signal-quality, weight-optimizer)",
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--db", help="Override research_db path from config")
    parser.add_argument("--trades-db", help="Override trades.db path")
    parser.add_argument("--upload", action="store_true", help="Upload results to S3")
    parser.add_argument("--date", default=date.today().isoformat(), help="Run date label")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--freeze", action="store_true",
        help="Compute recommendations but skip all S3 config promotions",
    )
    parser.add_argument(
        "--stop-instance", action="store_true",
        help="Stop this EC2 instance after completion (for scheduled runs)",
    )
    return parser.parse_args()


# ── Data source initialization ───────────────────────────────────────────────


def _init_data_sources(args: argparse.Namespace, config: dict) -> dict:
    """Initialize all data sources. Returns availability map."""
    # Research DB
    init_research_db(args.db, config)
    db_path = config.get("research_db")
    has_research_db = db_path is not None and os.path.exists(db_path)

    # Trades DB
    trades_db = getattr(args, "trades_db", None) or find_trades_db(config)
    config["_trades_db"] = trades_db
    has_trades_db = trades_db is not None and os.path.exists(trades_db)

    # Simulation artifacts from S3 (written by backtest.py)
    sweep_df = None
    predictor_sweep_df = None
    portfolio_stats = None
    predictor_stats = None
    bucket = config.get("output_bucket", config.get("signals_bucket", "alpha-engine-research"))
    prefix = f"backtest/{args.date}"
    s3 = boto3.client("s3")

    missing_artifacts: list[str] = []
    for artifact, loader in [
        ("sweep_df.parquet", lambda body: pd.read_parquet(io.BytesIO(body))),
        ("predictor_sweep_df.parquet", lambda body: pd.read_parquet(io.BytesIO(body))),
        ("portfolio_stats.json", lambda body: json.loads(body)),
        ("predictor_stats.json", lambda body: json.loads(body)),
    ]:
        try:
            resp = s3.get_object(Bucket=bucket, Key=f"{prefix}/{artifact}")
            data = loader(resp["Body"].read())
            if artifact == "sweep_df.parquet":
                sweep_df = data
            elif artifact == "predictor_sweep_df.parquet":
                predictor_sweep_df = data
            elif artifact == "portfolio_stats.json":
                portfolio_stats = data
            elif artifact == "predictor_stats.json":
                predictor_stats = data
            logger.info("Loaded simulation artifact: %s", artifact)
        except ClientError as e:
            # NoSuchKey is an expected state when backtest.py hasn't run
            # for this date (e.g., first run after --mode param-sweep was
            # skipped). Other ClientErrors are real S3 problems.
            if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                logger.warning(
                    "Simulation artifact not found in S3: %s/%s — evaluator "
                    "will run without it (backtest.py may not have run)",
                    prefix, artifact,
                )
                missing_artifacts.append(artifact)
            else:
                logger.error(
                    "S3 ClientError loading %s/%s: %s — evaluator cannot "
                    "trust results", prefix, artifact, e,
                )
                raise
        except Exception as e:
            logger.error(
                "Failed to parse simulation artifact %s/%s: %s — evaluator "
                "cannot trust results", prefix, artifact, e, exc_info=True,
            )
            raise

    # If ALL critical optimizer artifacts are missing, downstream optimizers
    # will run against zero data and produce garbage config recommendations.
    # Block the run by raising — operators must see the failure loudly so
    # the Saturday pipeline does not auto-promote bad configs.
    critical = {"sweep_df.parquet", "portfolio_stats.json"}
    if critical.issubset(set(missing_artifacts)):
        raise RuntimeError(
            f"All critical simulation artifacts missing from "
            f"s3://{bucket}/{prefix}/: {sorted(critical)}. "
            f"backtest.py must run before evaluate.py in the Saturday "
            f"pipeline. Check the upstream step status."
        )

    config["_sweep_df"] = sweep_df
    config["_predictor_sweep_df"] = predictor_sweep_df
    config["_portfolio_stats"] = portfolio_stats
    config["_predictor_stats"] = predictor_stats

    return {
        "research_db": has_research_db,
        "trades_db": has_trades_db,
        "sweep_df": sweep_df is not None,
        "predictor_sweep_df": predictor_sweep_df is not None,
        "portfolio_stats": portfolio_stats is not None,
        "predictor_stats": predictor_stats is not None,
    }


# ── Signal quality pipeline ──────────────────────────────────────────────────


def _run_signal_quality(config: dict, tracker: CompletenessTracker, avail: dict) -> tuple:
    """Run signal quality analysis. Data seeding/backfilling is handled by
    alpha-engine-data's signal_returns collector — the evaluator reads
    research.db as-is.

    Returns (sq_result, regime_rows, score_rows, attr_result, df_base).
    """
    db_path = config.get("research_db")

    # Check data completeness — warn if data module hasn't populated returns
    if avail["research_db"]:
        _check_data_freshness(db_path)

    sq_result = tracker.run_module(
        "signal_quality",
        lambda: _compute_signal_quality(config),
        required_inputs={"research_db": avail["research_db"]},
        skip_if_missing=["research_db"],
    )

    if sq_result.get("status") in ("skipped", "error", "db_not_found"):
        return sq_result, [], [], {"status": "skipped"}, None

    # Load df_base for downstream modules. Outcome columns are re-sourced from
    # the long-format score_performance_outcomes store (config#1483/#1528/#1529)
    # inside load_score_performance (the repo's single accessor) — every
    # downstream consumer of df_base reads long-format outcome data under the
    # HorizonPolicy-derived column names.
    try:
        df_base = signal_quality.load_score_performance(db_path)
    except Exception:
        df_base = None

    regime_rows = tracker.run_module(
        "regime_analysis",
        lambda: _compute_regime(config),
        required_inputs={"research_db": avail["research_db"]},
    )
    if not isinstance(regime_rows, list):
        regime_rows = regime_rows.get("rows", []) if isinstance(regime_rows, dict) else []

    score_rows = []
    if df_base is not None:
        thresholds = config.get("score_thresholds", [60, 65, 70, 75, 80, 85, 90])
        min_samples = config.get("min_samples", 5)
        score_rows = score_analysis.accuracy_by_threshold(
            df_base, thresholds=thresholds, min_samples=min_samples,
        )

    attr_result = tracker.run_module(
        "attribution",
        lambda: attribution.compute_attribution(df_base) if df_base is not None else {"status": "skipped"},
        required_inputs={"research_db": avail["research_db"], "df_base": df_base is not None},
        skip_if_missing=["df_base"],
    )

    # Compute and push predictor rolling metrics (evaluation output, not data collection)
    if avail["research_db"] and df_base is not None:
        push_predictor_rolling_metrics(config, db_path or "")

    return sq_result, regime_rows, score_rows, attr_result, df_base


def _check_data_freshness(db_path: str) -> None:
    """Warn if score_performance has stale rows that should have been backfilled by data module.

    Non-fatal diagnostic: failures log at ERROR level so flow-doctor sees
    them (corrupt DB, missing table, etc.) but do not raise — the caller's
    main pipeline should continue. Previously this caught all exceptions
    silently with ``pass``, which hid real DB corruption and SQL errors.

    config#1483/#1529: "resolved" is now determined by the long-format
    score_performance_outcomes store, not the wide return_Nd columns. Staleness
    is gated on the PRIMARY horizon only — that is the canonical resolution the
    Phase-2 producer must deliver. The DIAGNOSTIC horizon (5d) is intentionally
    optional (an unproduced diagnostic horizon yields no store rows by design),
    so its absence is NOT a freshness defect and would raise spurious warnings.
    The retired 10d/30d horizons are DROPPED (config#1456 — dead columns). A row
    is stale if its score_date is old enough that the primary horizon should have
    resolved (its trading-day window + a producer-lag grace), yet no primary-
    horizon store row exists. Gating on one horizon also avoids double-counting a
    row across horizons.
    """
    import sqlite3

    from analysis.outcome_store import store_exists

    policy = _HORIZON_POLICY
    primary_h = int(policy.primary_horizon)
    # Calendar-day staleness threshold: the primary horizon's trading-day window
    # (~1.4 calendar days per trading day) plus a producer-lag grace, preserving
    # the spirit of the legacy 21d→~30d threshold.
    threshold_days = int(round(primary_h * 1.4)) + 2
    try:
        conn = sqlite3.connect(db_path)
        try:
            if not store_exists(conn):
                logger.warning(
                    "Data freshness: score_performance_outcomes store absent — "
                    "long-format producer (signal_returns Step 2c) has not run"
                )
                return
            stale = conn.execute(
                "SELECT COUNT(*) FROM score_performance sp "
                "WHERE sp.score_date <= date('now', ?) "
                "  AND NOT EXISTS ("
                "    SELECT 1 FROM score_performance_outcomes o "
                "    WHERE o.symbol = sp.symbol AND o.score_date = sp.score_date "
                "      AND o.horizon_days = ?)",
                (f"-{threshold_days} days", primary_h),
            ).fetchone()[0]
        finally:
            conn.close()
        if stale > 0:
            logger.warning(
                "Data freshness: %d score_performance rows have no resolved "
                "primary-horizon (%dd) outcome (data module may not have run "
                "signal_returns collector)", stale, primary_h,
            )
    except Exception as exc:
        logger.error(
            "Data freshness check failed against %s: %s (non-fatal, "
            "continuing with main pipeline)", db_path, exc, exc_info=True,
        )


def _compute_signal_quality(config: dict) -> dict:
    db_path = config.get("research_db")
    min_samples = config.get("min_samples", 5)
    if not db_path:
        return {"status": "db_not_found"}
    df_base = signal_quality.load_score_performance(db_path)
    return signal_quality.compute_accuracy(df_base, min_samples=min_samples)


def _compute_regime(config: dict) -> dict:
    db_path = config.get("research_db")
    min_samples = config.get("min_samples", 5)
    if not db_path:
        return {"status": "skipped", "rows": []}
    df_regime = regime_analysis.load_with_regime(db_path)
    rows = regime_analysis.accuracy_by_regime(df_regime, min_samples=min_samples)
    return {"status": "ok", "rows": rows}


# ── Diagnostics ──────────────────────────────────────────────────────────────


def _run_diagnostics(
    config: dict,
    tracker: CompletenessTracker,
    avail: dict,
    df_base,
) -> dict:
    """Run all diagnostic modules. Returns dict of results."""
    db_path = config.get("research_db")
    trades_db = config.get("_trades_db")
    results = {}

    # Historical factor loadings (ArcticDB) for the e2e_lift counterfactuals
    # (config#1142 neutralized composite, config#967 consolidation + scanner
    # multi-factor). Built here because evaluate has the bucket (config) + the
    # eval-date set (research.db). Loads the FULL factor superset
    # (ALL_LOADING_FACTORS) so one read serves every counterfactual, and keys
    # off the SCANNER eval-dates (the widest, earliest funnel stage — covers the
    # downstream team/CIO cycles too). Fail-soft: any error -> {} and each
    # counterfactual reports status 'skipped'.
    factor_loadings: dict = {}
    pillar_profiles: dict = {}
    trajectory_scores: dict = {}
    bucket = config.get("signals_bucket", "alpha-engine-research")
    if avail.get("research_db"):
        try:
            import sqlite3 as _sqlite3

            _conn = _sqlite3.connect(db_path)
            try:
                eval_dates = [
                    r[0] for r in _conn.execute(
                        "SELECT DISTINCT se.eval_date FROM scanner_evaluations se "
                        "JOIN universe_returns u "
                        "ON u.ticker = se.ticker AND u.eval_date = se.eval_date "
                        "WHERE u.log_return_21d IS NOT NULL"
                    )
                ]
            finally:
                _conn.close()
            factor_loadings = end_to_end.load_historical_factor_loadings(
                bucket, eval_dates, factors=end_to_end.ALL_LOADING_FACTORS
            )
            # Live 6-pillar attractiveness profiles for the exact-attractiveness
            # counterfactual (config#1398) — same eval-date set, fail-soft.
            pillar_profiles = end_to_end.load_historical_pillar_profiles(
                bucket, eval_dates
            )
            # Persisted attractiveness-trajectory scores for the observe-mode
            # forward-IC gate (crucible-research #337 / config#1392) — same
            # eval-date set, fail-soft.
            trajectory_scores = end_to_end.load_historical_trajectory_scores(
                bucket, eval_dates
            )
        except Exception as _fl:  # fail-soft — never block diagnostics
            logger.warning("factor-loadings load skipped (non-fatal): %s", _fl)

    # End-to-end lift metrics
    results["e2e_lift"] = tracker.run_module(
        "end_to_end_lift",
        lambda: end_to_end.compute_lift_metrics(
            research_db_path=db_path, trades_db_path=trades_db,
            factor_loadings=factor_loadings, pillar_profiles=pillar_profiles,
            trajectory_scores=trajectory_scores, bucket=bucket,
        ),
        required_inputs={"research_db": avail["research_db"]},
        skip_if_missing=["research_db"],
    )

    # Universe-board attractiveness composite vs realized forward alpha
    # (config#1389 per-pillar IC + suggested 1/N-shrunk weights, config#1392
    # trajectory forward-IC, config#1398 top-N counterfactual vs the live
    # tech_score gate). Read-only measurement — emits SUGGESTED weights inside
    # the artifact only, never writes config/factor_attractiveness_weights.json.
    # Reuses the trajectory_scores dict loaded above for e2e_lift (no second
    # S3 read); frozen cross-repo schema v1 in
    # contracts/attractiveness_eval.schema.json.
    results["attractiveness_eval"] = tracker.run_module(
        "attractiveness_eval",
        lambda: attractiveness_eval_analysis.compute_attractiveness_eval(
            db_path,
            as_of=config.get("_run_date") or date.today().isoformat(),
            bucket=config.get("signals_bucket", "alpha-engine-research"),
            trajectory_scores=trajectory_scores or None,
        ),
        required_inputs={"research_db": avail["research_db"]},
        skip_if_missing=["research_db"],
    )

    # Entry trigger scorecard
    results["trigger_scorecard"] = tracker.run_module(
        "trigger_scorecard",
        lambda: trigger_scorecard.compute_trigger_scorecard(trades_db),
        required_inputs={"trades_db": avail["trades_db"]},
        skip_if_missing=["trades_db"],
    )

    # Alpha magnitude distribution
    results["alpha_dist"] = tracker.run_module(
        "alpha_distribution",
        lambda: alpha_distribution.compute_alpha_distribution(db_path),
        required_inputs={"research_db": avail["research_db"]},
        skip_if_missing=["research_db"],
    )

    # Score calibration curve
    results["score_calibration"] = tracker.run_module(
        "score_calibration",
        lambda: alpha_distribution.compute_score_calibration(db_path),
        required_inputs={"research_db": avail["research_db"]},
        skip_if_missing=["research_db"],
    )

    # Net veto value
    results["veto_value"] = tracker.run_module(
        "veto_value",
        lambda: veto_value.compute_veto_value(
            research_db_path=db_path, trades_db_path=trades_db,
        ),
        required_inputs={"research_db": avail["research_db"]},
        skip_if_missing=["research_db"],
    )

    # Predictor confusion matrix
    results["confusion_matrix"] = tracker.run_module(
        "predictor_confusion",
        lambda: _run_confusion_matrix(db_path),
        required_inputs={"research_db": avail["research_db"]},
        skip_if_missing=["research_db"],
    )

    # Shadow book analysis
    results["shadow_book"] = tracker.run_module(
        "shadow_book",
        lambda: shadow_book_analysis.compute_shadow_book_analysis(
            trades_db_path=trades_db,
            research_db_path=db_path if avail["research_db"] else None,
        ),
        required_inputs={"trades_db": avail["trades_db"]},
        skip_if_missing=["trades_db"],
    )

    # Exit timing analysis
    results["exit_timing"] = tracker.run_module(
        "exit_timing",
        lambda: exit_timing.compute_exit_timing(trades_db),
        required_inputs={"trades_db": avail["trades_db"]},
        skip_if_missing=["trades_db"],
    )

    # Behavioral-anomaly metric suite (L4514/config#698): decision reversal,
    # conviction stability, cost-adjusted quality, portfolio-state drift.
    # research.db is optional — the conviction component degrades to
    # insufficient_data without it (mirrors shadow_book's pattern).
    results["behavioral_anomaly"] = tracker.run_module(
        "behavioral_anomaly",
        lambda: behavioral_anomaly_analysis.compute_behavioral_anomaly(
            trades_db,
            research_db_path=db_path if avail["research_db"] else None,
            config=(config or {}).get("behavioral_anomaly"),
        ),
        required_inputs={"trades_db": avail["trades_db"]},
        skip_if_missing=["trades_db"],
    )

    # Post-trade analysis
    results["post_trade"] = tracker.run_module(
        "post_trade",
        lambda: _run_post_trade(trades_db),
        required_inputs={"trades_db": avail["trades_db"]},
        skip_if_missing=["trades_db"],
    )

    # Barrier coherence (predictor labels ↔ executor exits). The static
    # definition-divergence leg runs even with no trades, so this is NOT gated on
    # trades_db availability — it always emits an artifact (per the
    # always-emit-observational-artifact convention).
    results["barrier_coherence"] = tracker.run_module(
        "barrier_coherence",
        lambda: _run_barrier_coherence(config, trades_db),
        required_inputs={},
    )

    # Factor blend sensitivity — config-vs-realized stance ordering check.
    # PR 6 of scanner-placement arc + follow-up wire-in. Reads existing
    # score_performance df_base; backtester config may override regime
    # weights via factor_blend.regime_weights, else falls back to the
    # canonical defaults mirroring alpha-engine-config/research/scoring.yaml.
    fb_cfg = (config or {}).get("factor_blend") or {}
    fb_regime_weights = fb_cfg.get(
        "regime_weights",
        factor_blend_sensitivity.DEFAULT_REGIME_WEIGHTS,
    )
    fb_horizon = fb_cfg.get("horizon", "10d")
    results["factor_blend_sensitivity"] = tracker.run_module(
        "factor_blend_sensitivity",
        lambda: factor_blend_sensitivity.build_sensitivity_report(
            df_base if df_base is not None else __import__("pandas").DataFrame(),
            fb_regime_weights,
            horizon=fb_horizon,
        ),
        required_inputs={
            "research_db": avail["research_db"],
            "df_base": df_base is not None,
        },
        skip_if_missing=["df_base"],
    )

    # Factor-blend COUNTERFACTUAL REPLAY (config#749) — the "real" optimizer
    # signal companion to factor_blend_sensitivity above. Where the sensitivity
    # analyzer reads realized outcomes, this RE-SCORES the candidate universe
    # under alternate factor_blend weight tuples and replays each through the
    # vectorbt simulator. It is heavy (loads a price matrix + runs the sim) and
    # OPT-IN: it only runs when ``factor_blend_counterfactual.enabled`` is truthy
    # AND the caller has supplied the cycles + price matrix (the live weekly
    # evaluator is DB-only and does not, so this stays a no-op there — same
    # default-OFF convention as the other heavy counterfactual analyses).
    results["factor_blend_counterfactual_replay"] = tracker.run_module(
        "factor_blend_counterfactual_replay",
        lambda: _run_factor_blend_counterfactual_replay(config),
        required_inputs={},
    )

    # Macro multiplier evaluation
    results["macro_eval"] = tracker.run_module(
        "macro_eval",
        lambda: macro_eval.compute_macro_evaluation(db_path),
        required_inputs={"research_db": avail["research_db"]},
        skip_if_missing=["research_db"],
    )

    # Regime-stratified Sortino (Stage C.2 T2). Reads score_performance,
    # groups picks by market_regime, computes Sortino/Sharpe/log-alpha
    # per (regime, horizon) and the bull-bear Sortino spread (headline
    # T2 metric). Writes canonical eval-artifact to
    # s3://{bucket}/regime/stratified_sortino/{run_id}.json + latest.json
    # sidecar — consumed by the dashboard's Regime page.
    results["regime_stratified_sortino"] = tracker.run_module(
        "regime_stratified_sortino",
        lambda: regime_stratified_sortino_runner.run_regime_stratified_sortino(
            db_path=db_path,
            s3_bucket=config.get("s3_bucket") or config.get("signals_bucket"),
        ),
        required_inputs={"research_db": avail["research_db"]},
        skip_if_missing=["research_db"],
    )

    # Monte Carlo significance test
    results["monte_carlo"] = tracker.run_module(
        "monte_carlo",
        lambda: _run_monte_carlo(config),
        required_inputs={"research_db": avail["research_db"]},
        skip_if_missing=["research_db"],
    )

    # Production health monitoring
    results["production_health"] = tracker.run_module(
        "production_health",
        lambda: _run_production_health(config),
        required_inputs={"research_db": avail["research_db"]},
        skip_if_missing=["research_db"],
    )

    # Decision-capture coverage (Phase 2 transparency-inventory).
    # Reads S3 only — no local DB inputs required, runs even when the
    # research DB pull fails. Returns "no_recent_sf_run" when the S3
    # capture tree is empty for the trailing week (e.g. a smoke run on
    # a brand-new bucket).
    results["decision_capture_coverage"] = tracker.run_module(
        "decision_capture_coverage",
        lambda: decision_capture_coverage.compute_decision_capture_coverage(
            bucket=config.get("signals_bucket", "alpha-engine-research"),
            run_date=config.get("_run_date"),
        ),
        required_inputs={},
    )

    # Stance-distribution drift — Phase 5 acceptance check
    # (attractiveness-pillars-260520.md). Compares this week's
    # predictor/predictions/{date}.json stance counts to the prior 4 ISO
    # weeks' mean ± 2σ; fires a Telegram + SNS alert via
    # nousergon_lib.alerts.publish on FAIL. Defense-in-depth against
    # the pillar-aware classify_stance code path collapsing the
    # distribution into a single stance without surfacing through NAV
    # for weeks. ROADMAP L1614.
    from analysis import stance_distribution
    results["stance_distribution_drift"] = tracker.run_module(
        "stance_distribution_drift",
        lambda: stance_distribution.compute_stance_distribution_drift(
            bucket=config.get("signals_bucket", "alpha-engine-research"),
            current_date=config.get("_run_date") or date.today().isoformat(),
        ),
        required_inputs={},
    )

    # Executor-side decision-capture coverage (L2308 PR 5). Sibling of the
    # research-side coverage above; reads executor:* artifacts emitted by
    # L2308 PRs 1-4 producers (entry_triggers / position_sizer /
    # risk_guard / exit_rules). Insufficient_data until
    # ALPHA_ENGINE_DECISION_CAPTURE_ENABLED is enabled on the trading EC2
    # AND ≥1 weekday SF run has captured artifacts.
    results["executor_decision_capture_coverage"] = tracker.run_module(
        "executor_decision_capture_coverage",
        lambda: executor_decision_capture_coverage.compute_executor_decision_capture_coverage(
            bucket=config.get("signals_bucket", "alpha-engine-research"),
            run_date=config.get("_run_date"),
        ),
        required_inputs={},
    )

    # Measurement-coverage funnel (config#909) — the *outcome*-observability
    # sibling of the decision-capture coverage above. Traces the actionable
    # signal set through predictions → fills → P&L attribution and emits
    # backtest/{date}/coverage.json for the dashboard's measurement-coverage
    # panel. Read-only over signals/ + predictor/predictions/ (S3) and the
    # trades.db ENTER rows; tolerant of any missing input (degrades the
    # affected stage to null rather than raising). required_inputs left empty
    # so it runs even when the trades.db pull failed — it self-reports which
    # stages were measured via stage_availability.
    results["measurement_coverage"] = tracker.run_module(
        "measurement_coverage",
        lambda: measurement_coverage.compute_measurement_coverage(
            bucket=config.get("signals_bucket", "alpha-engine-research"),
            run_date=config.get("_run_date"),
            trades_db_path=trades_db,
        ),
        required_inputs={},
    )

    # Provenance grounding — fourth leg of agent-justification stack.
    # Per-agent tool-call + input-trace metrics on captured artifacts.
    # Reads S3 only — no local DB inputs required.
    results["provenance_grounding"] = tracker.run_module(
        "provenance_grounding",
        lambda: provenance_grounding.compute_provenance_grounding(
            bucket=config.get("signals_bucket", "alpha-engine-research"),
            run_date=config.get("_run_date"),
        ),
        required_inputs={},
    )

    # Quant rank quality — per-sector corr(quant_rank, realized return) over a
    # rolling 8-week window, at both policy horizons (diagnostic 5d legacy
    # keys + canonical 21d suffixed keys, config#1529). Surfaces "is the
    # technical scorer's
    # ranking even ordering correctly?" before drift compounds. The
    # 2026-05-09 evaluator-email post-mortem found healthcare/industrials/
    # tech rank-correlations at +0.33-0.36 (anti-skill); without this
    # diagnostic running weekly the inversion was caught only in
    # retrospect via per-stage decomposition.
    results["quant_rank_quality"] = tracker.run_module(
        "quant_rank_quality",
        lambda: quant_rank_quality.compute_quant_rank_quality(
            db_path=config.get("research_db"),
            run_date=config.get("_run_date"),
        ),
        required_inputs={"research_db": config.get("research_db")},
    )

    # CIO rule-tag precision — per-tag precision of the LLM CIO's gates over a
    # rolling 8-week window. Consumes cio_evaluations.rule_tags (research
    # migration 14, persisted from crucible-research#152) joined to
    # universe_returns.beat_spy_5d. Per tag: n_decisions, ADVANCE precision
    # (% of ADVANCE-tagged that beat SPY at 5d) and the REJECT-beat rate
    # (per-tag false-negative rate). Gated to insufficient_data until ≥4 weeks
    # of rule_tags data accumulate; surfaces "which CIO gates are systematically
    # over- or under-rejecting?" so the LLM-CIO drop decision stays defensible.
    results["cio_rule_tag_precision"] = tracker.run_module(
        "cio_rule_tag_precision",
        lambda: cio_rule_tag_precision.compute_cio_rule_tag_precision(
            db_path=config.get("research_db"),
            run_date=config.get("_run_date"),
        ),
        required_inputs={"research_db": config.get("research_db")},
    )

    # Agent-justification stack summaries — judge / clustering / concordance /
    # counterfactual aggregated across agents for the most-recent SF date.
    # Pre-2026-05-07 reorder these Lambdas ran AFTER Evaluator so their
    # outputs were absent from the email; the SF reorder moves them
    # upstream of PredictorTraining so this loader has fresh data each
    # week. S3-only reads; no DB inputs.
    results["agent_justification"] = tracker.run_module(
        "agent_justification",
        lambda: agent_justification.summarize_all(
            bucket=config.get("signals_bucket", "alpha-engine-research"),
            run_date=config.get("_run_date"),
        ),
        required_inputs={},
    )

    # Feature drift detection
    results["feature_drift"] = tracker.run_module(
        "feature_drift",
        lambda: _run_feature_drift(config),
        required_inputs={"research_db": avail["research_db"]},
        skip_if_missing=["research_db"],
    )

    return results


def _run_confusion_matrix(db_path: str) -> dict:
    from analysis.predictor_confusion import compute_confusion_matrix
    return compute_confusion_matrix(db_path)


def _run_post_trade(trades_db: str) -> dict:
    from analysis.post_trade import compute_post_trade_analysis
    return compute_post_trade_analysis(trades_db)


def _run_factor_blend_counterfactual_replay(config: dict | None) -> dict:
    """Opt-in factor-blend counterfactual replay producer (config#749).

    Default-OFF: returns ``{"status": "skipped"}`` unless
    ``config["factor_blend_counterfactual"]["enabled"]`` is truthy AND the caller
    has injected the replay inputs (``cycles`` + ``price_matrix``) into that same
    config block. The live weekly evaluator runs DB-only and never injects these,
    so this is a no-op there — keeping the heavy vectorbt replay out of the live
    path while exposing a single, tested entry point a focused backtest run (or a
    future S3-loader wire-in) can call once the inputs are assembled.

    Expected config shape (all under ``factor_blend_counterfactual``):
        enabled: bool
        cycles: [{"date": str, "candidates": DataFrame}, ...]
        price_matrix: DataFrame
        weight_variants: [{momentum_score: .., ...}, ...]   # optional
        baseline_weights / spy_prices / picks_per_cycle / hold_days /
        init_cash / fees                                    # all optional
    """
    cfg = (config or {}).get("factor_blend_counterfactual") or {}
    if not cfg.get("enabled"):
        return {"status": "skipped", "reason": "opt-in; disabled by default"}

    cycles = cfg.get("cycles")
    price_matrix = cfg.get("price_matrix")
    if not cycles or price_matrix is None:
        return {
            "status": "skipped",
            "reason": "enabled but no cycles/price_matrix supplied",
        }

    weight_variants = cfg.get("weight_variants")
    if not weight_variants:
        # Default grid: the canonical per-regime blends from the sensitivity
        # analyzer (so "what if we always ran the BULL/BEAR/NEUTRAL blend").
        weight_variants = list(
            factor_blend_sensitivity.DEFAULT_REGIME_WEIGHTS.values()
        )

    return factor_blend_counterfactual_replay.build_counterfactual_replay_report(
        cycles,
        weight_variants,
        price_matrix,
        baseline_weights=cfg.get("baseline_weights"),
        spy_prices=cfg.get("spy_prices"),
        picks_per_cycle=cfg.get(
            "picks_per_cycle",
            factor_blend_counterfactual_replay.DEFAULT_PICKS_PER_CYCLE,
        ),
        hold_days=cfg.get(
            "hold_days", factor_blend_counterfactual_replay.DEFAULT_HOLD_DAYS
        ),
        init_cash=float(cfg.get("init_cash", 1_000_000.0)),
        fees=float(cfg.get("fees", 0.001)),
    )


def _run_barrier_coherence(config: dict, trades_db: str) -> dict:
    """Predictor↔executor triple-barrier coherence diagnostic.

    Reads BOTH sides live so the comparison is real, not stale-default vs
    stale-default:
      - LABEL side: the predictor ``triple_barrier`` block from
        ``alpha-engine-config/predictor/predictor.yaml`` (the same file the
        predictor parses to build its labels) via
        ``pipeline_common._load_label_barrier_config`` — config#723's
        single-source-of-truth for label barriers.
      - EXECUTION side: the LIVE, sweep-tuned executor barriers from
        ``config/executor_params.json`` on S3.
    On any read failure we fall back to the diagnostic's documented defaults but
    record the fallback in ``label_config_source`` / ``exec_params_source`` and
    WARN-log it — not a silent swallow (the diagnostic still runs; the source is
    visible in the rendered artifact).
    """
    from analysis.barrier_coherence import compute_barrier_coherence

    label_config = _load_label_barrier_config()
    if label_config:
        label_source = "live predictor.yaml triple_barrier (config repo)"
    else:
        label_source = "defaults (predictor.yaml fallback)"
        logger.warning(
            "barrier_coherence: predictor.yaml triple_barrier block not found "
            "on disk; using documented label-barrier defaults"
        )

    bucket = config.get("signals_bucket", "alpha-engine-research")
    exec_params: dict | None = None
    exec_source = "defaults (executor/strategies/config.py)"
    _BARRIER_KEYS = (
        "atr_multiplier",
        "profit_take_pct",
        "time_decay_reduce_days",
        "time_decay_exit_days",
    )
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key="config/executor_params.json")
        data = json.loads(obj["Body"].read())
        exec_params = {k: data[k] for k in _BARRIER_KEYS if k in data}
        if exec_params:
            exec_source = "live S3 config/executor_params.json (sweep-tuned)"
        else:
            exec_params = None
            logger.warning(
                "barrier_coherence: executor_params.json present but carried no "
                "barrier keys; falling back to documented defaults"
            )
    except Exception as e:  # noqa: BLE001 — best-effort live read; fallback recorded, not swallowed.
        logger.warning(
            "barrier_coherence: could not read live executor_params.json (%s); "
            "using documented defaults", e
        )

    return compute_barrier_coherence(
        trades_db,
        label_config=label_config,
        label_config_source=label_source,
        exec_params=exec_params,
        exec_params_source=exec_source,
    )


def _run_monte_carlo(config: dict) -> dict:
    from analysis.monte_carlo import run_monte_carlo
    db_path = config.get("research_db")
    return run_monte_carlo(
        research_db_path=db_path,
        n_permutations=config.get("monte_carlo_permutations", 200),
        horizon=config.get("monte_carlo_horizon", "5d"),
    )


def _run_production_health(config: dict) -> dict:
    from analysis.production_health import compute_production_health
    db_path = config.get("research_db", "")
    bucket = config.get("signals_bucket", "alpha-engine-research")
    return compute_production_health(db_path, bucket)


def _run_feature_drift(config: dict) -> dict:
    from analysis.feature_drift import compute_feature_drift
    db_path = config.get("research_db", "")
    bucket = config.get("signals_bucket", "alpha-engine-research")
    return compute_feature_drift(db_path, bucket)


# ── Optimizers ───────────────────────────────────────────────────────────────


def _read_current_weights(config: dict) -> dict:
    """Read current scoring weights from S3, local config, or defaults."""
    bucket = config.get("signals_bucket", "alpha-engine-research")
    # Canonical keys are weight_optimizer.SUB_SCORES ("quant", "qual") — the
    # payload apply_weights() actually persists to scoring_weights.json.
    # "news"/"research" were the pre-rename key names (commit 92c5067,
    # 2026-03-29, "Rename sub_scores from news/research to quant/qual") and
    # are accepted here only as a read-side migration fallback, in case an
    # S3 object written before that rename has never since been overwritten
    # (config#1679: this mismatch meant the true previous weights could
    # never be read back, so every cycle's delta/churn was computed against
    # the default_weights fallback instead).
    legacy_to_canonical = dict(zip(("news", "research"), weight_optimizer.SUB_SCORES))
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key="config/scoring_weights.json")
        data = json.loads(obj["Body"].read())
        weights = {k: float(data[k]) for k in weight_optimizer.SUB_SCORES if k in data}
        for legacy_key, canonical_key in legacy_to_canonical.items():
            if canonical_key not in weights and legacy_key in data:
                weights[canonical_key] = float(data[legacy_key])
        if len(weights) == len(weight_optimizer.SUB_SCORES):
            return weights
    except Exception:
        pass

    research_paths = config.get("research_paths", [])
    if isinstance(research_paths, str):
        research_paths = [research_paths]
    research_path = next((p for p in research_paths if os.path.isdir(p)), None)
    if research_path:
        universe_yaml = os.path.join(research_path, "config", "universe.yaml")
        try:
            with open(universe_yaml) as f:
                universe = yaml.safe_load(f)
            weights = universe.get("scoring_weights", {})
        except Exception:
            weights = {}
        if weights:
            # Fail-loud key guard (config#1842): a drifted scoring_weights
            # block in research's universe.yaml would reproduce the phantom
            # 0.0 baseline downstream. Raise (isolated per-module by
            # tracker.run_module) instead of returning drifted keys.
            from optimizer.config_guards import validate_keyed_block
            validate_keyed_block(
                weights, weight_optimizer.SUB_SCORES,
                config_path="crucible-research config/universe.yaml scoring_weights",
            )
            return weights

    # Validated chokepoint (config#1842): raises ConfigKeyDriftError on a
    # stale default_weights block instead of silently zero-filling.
    return weight_optimizer.configured_default_weights()


def _run_optimizers(
    config: dict,
    tracker: CompletenessTracker,
    avail: dict,
    df_base,
    freeze: bool,
    diagnostics: dict,
) -> dict:
    """Run all optimizer modules. Returns dict of results."""
    bucket = config.get("signals_bucket", "alpha-engine-research")
    db_path = config.get("research_db")
    # config#1017: explicit backfill run_date threaded into each optimizer's
    # apply()/produce_artifact() so a --date backfill keys recommendation
    # artifacts to the BACKFILL trading day (matching the assemble read),
    # not today's. None on a live run → optimizers fall back to today_iso().
    run_date = config.get("_run_date")
    results = {}

    # Weight optimizer
    results["weight_result"] = tracker.run_module(
        "weight_optimizer",
        lambda: _run_weight_opt(config, df_base, freeze),
        required_inputs={"research_db": avail["research_db"], "df_base": df_base is not None},
        skip_if_missing=["df_base"],
    )

    # Veto analysis optimizer
    results["veto_result"] = tracker.run_module(
        "veto_optimizer",
        lambda: _run_veto_opt(config, df_base, freeze),
        required_inputs={"research_db": avail["research_db"], "df_base": df_base is not None},
        skip_if_missing=["df_base"],
    )

    # Research params optimizer
    results["research_params"] = tracker.run_module(
        "research_optimizer",
        lambda: _run_research_opt(config, df_base, freeze),
        required_inputs={"research_db": avail["research_db"], "df_base": df_base is not None},
        skip_if_missing=["df_base"],
    )

    # Trigger optimizer
    trigger_result = diagnostics.get("trigger_scorecard")
    results["trigger_opt"] = tracker.run_module(
        "trigger_optimizer",
        lambda: _run_trigger_opt(trigger_result, freeze, bucket, run_date),
        required_inputs={"trigger_scorecard": trigger_result is not None and trigger_result.get("status") == "ok"},
        skip_if_missing=["trigger_scorecard"],
    )

    # Predictor sizing optimizer
    results["predictor_sizing"] = tracker.run_module(
        "predictor_sizing",
        lambda: _run_predictor_sizing(db_path, freeze, bucket, run_date),
        required_inputs={"research_db": avail["research_db"]},
        skip_if_missing=["research_db"],
    )

    # Barrier-win-prob sizing optimizer (Task B3). IC-criterion promotion gate
    # for the executor's dormant barrier_win_prob sizing multiplier; mirrors
    # predictor_sizing. Reports column_absent until alpha-engine-data records
    # barrier_win_prob into predictor_outcomes (B3 activation prerequisite).
    results["barrier_sizing"] = tracker.run_module(
        "barrier_sizing",
        lambda: _run_barrier_sizing(db_path, freeze, bucket, run_date),
        required_inputs={"research_db": avail["research_db"]},
        skip_if_missing=["research_db"],
    )

    # Stance-conditional sizing optimizer (L300). Offline-IC gate replacing the
    # inert predictionless param sweep over stance_size_*; tunes the multipliers
    # against realized per-stance alpha. Reports stance_column_absent until the
    # score_performance stance migration lands (mirrors barrier).
    stance_sizing_optimizer.init_config(config)
    results["stance_sizing"] = tracker.run_module(
        "stance_sizing",
        lambda: _run_stance_sizing(db_path, freeze, bucket, run_date),
        required_inputs={"research_db": avail["research_db"]},
        skip_if_missing=["research_db"],
    )

    # Scanner optimizer
    results["scanner_opt"] = tracker.run_module(
        "scanner_optimizer",
        lambda: _run_scanner_opt(config, db_path, freeze, bucket),
        required_inputs={"research_db": avail["research_db"]},
        skip_if_missing=["research_db"],
    )

    # Pipeline optimizer (team slots + CIO)
    e2e_lift = diagnostics.get("e2e_lift")
    has_e2e = e2e_lift is not None and e2e_lift.get("status") == "ok"

    results["team_opt"] = tracker.run_module(
        "team_slot_optimizer",
        lambda: _run_team_opt(e2e_lift, freeze, bucket),
        required_inputs={"e2e_lift": has_e2e},
        skip_if_missing=["e2e_lift"],
    )

    results["cio_opt"] = tracker.run_module(
        "cio_optimizer",
        lambda: _run_cio_opt(e2e_lift),
        required_inputs={"e2e_lift": has_e2e},
        skip_if_missing=["e2e_lift"],
    )

    # Tech weight ablation — per-sector recommendation by sub-score
    # rank-correlation grid search. Pairs with PR-A's quant_rank_quality
    # diagnostic + PR-B's research v15 sub-score persistence.
    # Status="insufficient_data" until ≥30 rows per team have populated
    # sub-scores. Apply path is gated behind two flags
    # (use_tech_ablation_target + enforce_tech_ablation) and a 4-week
    # reproduction guard — see optimizer/tech_weight_ablation.apply().
    results["tech_weight_ablation"] = tracker.run_module(
        "tech_weight_ablation",
        lambda: _run_tech_weight_ablation(config, freeze),
        required_inputs={"research_db": avail["research_db"]},
        skip_if_missing=["research_db"],
    )

    # Factor-blend optimizer (config#748) — auto-apply companion to the
    # observability-only factor_blend_sensitivity diagnostic. Consumes the
    # already-computed sensitivity report (no recomputation) and recommends
    # per-regime stance-weight reorderings where a trustworthy mismatch
    # persists. Two-flag activation (use_factor_blend_target +
    # enforce_factor_blend) + reproduction gate; default-off in
    # alpha-engine-config/backtester/config.yaml so it shadows safely while the
    # (regime, stance) cells mature past the 20-sample trustworthy threshold.
    fb_sensitivity = diagnostics.get("factor_blend_sensitivity")
    fb_has_data = (
        fb_sensitivity is not None and bool(fb_sensitivity.get("has_data"))
    )
    results["factor_blend_opt"] = tracker.run_module(
        "factor_blend_optimizer",
        lambda: _run_factor_blend_optimizer(config, fb_sensitivity, freeze),
        required_inputs={"factor_blend_sensitivity": fb_has_data},
        skip_if_missing=["factor_blend_sensitivity"],
    )

    # Executor optimizer (needs sweep_df from simulation)
    sweep_df = config.get("_sweep_df")
    predictor_sweep_df = config.get("_predictor_sweep_df")
    effective_sweep = sweep_df if sweep_df is not None else predictor_sweep_df

    results["executor_rec"] = tracker.run_module(
        "executor_optimizer",
        lambda: _run_executor_opt(config, effective_sweep, freeze),
        required_inputs={
            "sweep_df": sweep_df is not None,
            "predictor_sweep_df": predictor_sweep_df is not None,
        },
        skip_if_missing=None,  # runs in degraded mode, doesn't skip
    )

    if not freeze:
        manifest_date = run_date or config.get("_run_date")
        if not manifest_date:
            from nousergon_lib.dates import today_iso
            manifest_date = today_iso()
        write_optimizer_run_manifest(
            bucket,
            manifest_date,
            results,
            freeze=freeze,
        )

    return results


def _summarize_optimizer_module(name: str, result: dict | None) -> dict:
    """Compact per-optimizer row for the optimizer_run freshness proxy."""
    if not result:
        return {"ran": False, "applied": False, "reason": "no_result"}
    status = result.get("status")
    if status == "skipped":
        return {
            "ran": False,
            "applied": False,
            "reason": result.get("degradation_reason") or "skipped",
        }
    if status == "error":
        return {"ran": True, "applied": False, "reason": result.get("error", "error")}

    apply = result.get("apply_result")
    if apply is None:
        apply = result.get("apply")
    if isinstance(apply, dict):
        applied = bool(apply.get("applied"))
        reason = apply.get("reason") or result.get("status", "ok")
        return {"ran": True, "applied": applied, "reason": reason}

    return {"ran": True, "applied": False, "reason": result.get("status", "ok")}


def write_optimizer_run_manifest(
    bucket: str,
    run_date: str,
    results: dict,
    *,
    freeze: bool,
    s3_client=None,
) -> str:
    """Write the Saturday optimizer-stage liveness proxy (config#1726).

    Unconditionally records which optimizers ran/applied/declined. Raises on
    S3 failure — this artifact is load-bearing for event_driven config rows.
    """
    if freeze:
        return ""

    import boto3

    client = s3_client or boto3.client("s3")
    key = f"optimizer_run/{run_date}.json"
    payload = {
        "trading_day": run_date,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "freeze": freeze,
        "optimizers": {
            name: _summarize_optimizer_module(name, result)
            for name, result in results.items()
        },
    }
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    logger.info("Wrote optimizer run manifest s3://%s/%s", bucket, key)
    return key


def _portfolio_daily_returns_from_team_lift(team_lift: list[dict], prices, horizon_days: int = 10):
    """Portfolio-wide aligned daily-return series from the e2e team-lift picks.

    ``risk_ratio_ci`` (config#976) needs a single portfolio-level daily-return
    series (mirroring the ``pf_returns_aligned`` series ``backtest.py``'s
    deployed-strategy headline computes via ``_simulate_and_measure`` — a
    process-separate, JSON-only artifact that does not carry raw series back
    to evaluate.py). Rather than re-simulate the MVO solver here, this
    aggregates every team's picks (already loaded for ``compute_team_metrics``
    a few lines above) into one equal-weight portfolio sleeve via the same
    ``compute_team_daily_returns`` helper each team already uses, stamping a
    single synthetic ``team_id="portfolio"`` across all picks.

    Returns ``None`` when there are no usable picks (caller degrades to
    ``risk_ratio_ci`` reporting ``insufficient_data``, matching the module's
    own always-emit contract).
    """
    if not team_lift or prices is None or getattr(prices, "empty", True):
        return None

    from analysis.team_daily_returns import compute_team_daily_returns

    all_picks: list[dict] = []
    for team in team_lift:
        for pick in team.get("picks") or []:
            all_picks.append({**pick, "team_id": "portfolio"})
    if not all_picks:
        return None

    picks_df = pd.DataFrame(all_picks)
    try:
        portfolio_returns = compute_team_daily_returns(
            picks_df, prices, horizon_days=horizon_days,
        )
    except Exception as e:  # noqa: BLE001 — best-effort; risk_ratio_ci degrades gracefully
        logger.warning("[risk_ratio_ci] portfolio daily-returns aggregation failed: %s", e)
        return None
    series = portfolio_returns.get("portfolio")
    if series is None or series.empty:
        return None
    return series


def _run_weight_opt(config: dict, df_base, freeze: bool) -> dict:
    bucket = config.get("signals_bucket", "alpha-engine-research")
    current_weights = _read_current_weights(config)
    min_samples = config.get("weight_optimizer_min_samples", 30)

    df_with_sub = weight_optimizer.load_with_subscores(df_base, bucket)
    result = weight_optimizer.compute_weights(
        df_with_sub, current_weights=current_weights,
        min_samples=min_samples, bucket=bucket,
    )
    if freeze:
        result["apply_result"] = {"applied": False, "reason": "frozen (--freeze flag)"}
    else:
        result["apply_result"] = weight_optimizer.apply_weights(result, bucket)

    # Observe-first significance comparison (config#1426 Phase 2). Stamp what the
    # LIVE gate actually did into the (non-enforcing) significance record and
    # surface the would-promote-vs-did verdict for operator review.
    _emit_significance_observe(
        result.get("significance_observe"),
        did_promote=bool(result.get("apply_result", {}).get("applied")),
        gate_label="weight_optimizer",
    )
    return result


def _emit_significance_observe(record: dict | None, *, did_promote: bool, gate_label: str) -> None:
    """Stamp the live gate's decision into an observe record and log the verdict.

    NON-ENFORCING (config#1426 observe-first): logs only. ``did_promote`` is what
    the currently-shipping gate decided; the record's ``would_block`` is what the
    significance bar would have decided. The case to watch is
    ``promotes_on_undefended_evidence`` — live promoted while significance would
    have blocked (the leg-f failure mode).
    """
    if not record:
        return
    record["did_promote"] = did_promote
    would_block = bool(record.get("would_block"))
    record["promotes_on_undefended_evidence"] = bool(did_promote and would_block)
    msg = (
        "significance_observe [%s]: did_promote=%s would_block=%s "
        "significant=%s promotes_on_undefended_evidence=%s (observe-only, not enforced)"
    )
    args = (
        gate_label, did_promote, would_block, record.get("significant"),
        record["promotes_on_undefended_evidence"],
    )
    if record["promotes_on_undefended_evidence"]:
        logger.warning(msg, *args)
    else:
        logger.info(msg, *args)


_SIGNIFICANCE_OBSERVE_KEYS = (
    "weight_result", "veto_result", "predictor_sizing",
    "barrier_sizing", "stance_sizing",
)


def _collect_significance_observe(opt_results: dict) -> dict | None:
    """Gather the per-optimizer observe verdicts (config#1426) for durable
    persistence into metrics.json.

    Each `_run_*` seam stamps its optimizer result's `significance_observe`
    record with the live gate's decision; this collects them into one block so
    the observe→enforce soak (Phase 4) has a reviewable per-Saturday history at a
    stable S3 path. Returns None when no verdict is present (nothing to persist).
    """
    out: dict = {}
    for key in _SIGNIFICANCE_OBSERVE_KEYS:
        res = opt_results.get(key)
        if isinstance(res, dict):
            rec = res.get("significance_observe")
            if rec:
                out[key] = rec
    return out or None


def _run_veto_opt(config: dict, df_base, freeze: bool) -> dict:
    bucket = config.get("signals_bucket", "alpha-engine-research")
    result = veto_analysis.analyze_veto_effectiveness(df_base, bucket)
    if result.get("status") == "ok":
        if freeze:
            result["apply_result"] = {"applied": False, "reason": "frozen (--freeze flag)"}
        else:
            result["apply_result"] = veto_analysis.apply(result, bucket)
    _emit_significance_observe(
        result.get("significance_observe"),
        did_promote=bool(result.get("apply_result", {}).get("applied")),
        gate_label="veto_analysis",
    )
    return result


def _run_research_opt(config: dict, df_base, freeze: bool) -> dict:
    bucket = config.get("signals_bucket", "alpha-engine-research")
    current_rp = read_params_pit_or_current(research_optimizer, bucket, config)
    corr_result = research_optimizer.compute_boost_correlations(df_base, bucket)
    if corr_result.get("status") != "ok":
        return corr_result
    rp_result = research_optimizer.recommend(corr_result, current_rp)
    if rp_result.get("status") == "ok":
        if freeze:
            rp_result["apply_result"] = {"applied": False, "reason": "frozen (--freeze flag)"}
        else:
            rp_result["apply_result"] = research_optimizer.apply(rp_result, bucket)
    return rp_result


def _run_trigger_opt(
    trigger_result: dict, freeze: bool, bucket: str, run_date: str | None = None,
) -> dict:
    result = trigger_optimizer.analyze(trigger_result)
    if result.get("status") == "ok":
        if freeze:
            result["apply_result"] = {"applied": False, "reason": "frozen (--freeze flag)"}
        else:
            result["apply_result"] = trigger_optimizer.apply(result, bucket, run_date)
    return result


def _run_predictor_sizing(
    db_path: str, freeze: bool, bucket: str, run_date: str | None = None,
) -> dict:
    result = predictor_sizing_optimizer.analyze(db_path)
    if result.get("status") == "ok":
        if freeze:
            result["apply_result"] = {"applied": False, "reason": "frozen (--freeze flag)"}
        elif result.get("recommendation") == "enable":
            result["apply_result"] = predictor_sizing_optimizer.apply(result, bucket, run_date)
    _emit_significance_observe(
        result.get("significance_observe"),
        did_promote=bool(result.get("apply_result", {}).get("applied")),
        gate_label="predictor_sizing",
    )
    return result


def _run_barrier_sizing(
    db_path: str, freeze: bool, bucket: str, run_date: str | None = None,
) -> dict:
    result = barrier_sizing_optimizer.analyze(db_path)
    if result.get("status") == "ok":
        if freeze:
            result["apply_result"] = {"applied": False, "reason": "frozen (--freeze flag)"}
        elif result.get("recommendation") == "enable":
            result["apply_result"] = barrier_sizing_optimizer.apply(result, bucket, run_date)
    _emit_significance_observe(
        result.get("significance_observe"),
        did_promote=bool(result.get("apply_result", {}).get("applied")),
        gate_label="barrier_sizing",
    )
    return result


def _run_stance_sizing(
    db_path: str, freeze: bool, bucket: str, run_date: str | None = None,
) -> dict:
    """L300: offline-IC stance-sizing optimizer. Reports stance_column_absent
    until the score_performance stance migration lands (mirrors barrier)."""
    result = stance_sizing_optimizer.analyze(db_path)
    if result.get("status") == "ok":
        if freeze:
            result["apply_result"] = {"applied": False, "reason": "frozen (--freeze flag)"}
        elif result.get("recommendation") == "enable":
            result["apply_result"] = stance_sizing_optimizer.apply(result, bucket, run_date)
    _emit_significance_observe(
        result.get("significance_observe"),
        did_promote=bool(result.get("apply_result", {}).get("applied")),
        gate_label="stance_sizing",
    )
    return result


def _run_scanner_opt(config: dict, db_path: str, freeze: bool, bucket: str) -> dict:
    analysis = scanner_optimizer.analyze(db_path)
    if analysis.get("status") != "ok":
        return analysis
    current = read_params_pit_or_current(scanner_optimizer, bucket, config)
    result = scanner_optimizer.recommend(analysis, current)
    if result.get("status") == "ok":
        if freeze:
            result["apply_result"] = {"applied": False, "reason": "frozen (--freeze flag)"}
        else:
            result["apply_result"] = scanner_optimizer.apply(result, bucket)
    result["analysis"] = analysis
    return result


def _run_team_opt(e2e_lift: dict, freeze: bool, bucket: str) -> dict:
    analysis = pipeline_optimizer.analyze_team_performance(e2e_lift)
    if analysis.get("status") != "ok":
        return analysis
    result = pipeline_optimizer.recommend_team_slots(analysis)
    if result.get("status") == "ok":
        if freeze:
            result["apply_result"] = {"applied": False, "reason": "frozen (--freeze flag)"}
        else:
            result["apply_result"] = pipeline_optimizer.apply_team_slots(result, bucket)
    result["analysis"] = analysis
    return result


def _run_cio_opt(e2e_lift: dict) -> dict:
    # Observability-only since 2026-07-04 (config#1719): the CIO-vs-ranking
    # MEASUREMENT is computed and surfaced in the eval report, but there is no
    # longer any S3 actuation. ``apply_cio_mode`` wrote a ``cio_mode`` flag no
    # consumer read (a 63-day dead write) and was retired; a real consumed
    # deterministic-CIO mode is scoped under config#1060. No ``freeze``/
    # ``bucket`` params: this path has no side effects to freeze.
    result = pipeline_optimizer.analyze_cio_performance(e2e_lift)
    if result.get("status") == "ok":
        result["apply_result"] = {
            "applied": False,
            "reason": (
                "observability-only — cio_mode actuation retired (config#1719); "
                "CIO-vs-ranking recommendation surfaced as diagnostic (config#1060)"
            ),
        }
    return result


def _run_tech_weight_ablation(config: dict, freeze: bool) -> dict:
    """Run tech_weight_ablation compute + apply path (ROADMAP L2553).

    Apply contract mirrors executor_optimizer: compute the
    recommendation, then call ``apply()`` to (optionally) write shadow
    + live S3 artifacts. ``freeze=True`` short-circuits the apply()
    call entirely so ``--freeze`` evaluator runs produce zero S3 side
    effects.
    """
    bucket = config.get("signals_bucket", "alpha-engine-research")
    result = tech_weight_ablation.compute_tech_weight_ablation(
        db_path=config.get("research_db"),
        run_date=config.get("_run_date"),
    )
    if freeze:
        result["apply_result"] = {
            "applied": False, "reason": "frozen (--freeze flag)",
        }
    else:
        result["apply_result"] = tech_weight_ablation.apply(result, bucket)
    return result


def _run_factor_blend_optimizer(
    config: dict, sensitivity_report: dict | None, freeze: bool
) -> dict:
    """Run factor_blend_optimizer compute + apply path (config#748).

    Consumes the already-computed factor_blend_sensitivity report (passed in
    from diagnostics — no recomputation). Apply contract mirrors
    tech_weight_ablation / executor_optimizer: compute the recommendation, then
    call ``apply()`` to (optionally) write shadow + live S3 artifacts.
    ``freeze=True`` short-circuits apply() so ``--freeze`` evaluator runs
    produce zero S3 side effects.
    """
    bucket = config.get("signals_bucket", "alpha-engine-research")
    fb_cfg = (config or {}).get("factor_blend") or {}
    regime_weights = fb_cfg.get(
        "regime_weights", factor_blend_sensitivity.DEFAULT_REGIME_WEIGHTS
    )
    result = factor_blend_optimizer.recommend(
        sensitivity_report or {}, regime_weights
    )
    if freeze:
        result["apply_result"] = {
            "applied": False, "reason": "frozen (--freeze flag)",
        }
    else:
        result["apply_result"] = factor_blend_optimizer.apply(result, bucket)
    return result


def _publish_executor_opt_rejection_alert(result: dict, config: dict) -> None:
    """Fire a named alert when ``executor_optimizer.recommend()`` returns
    a non-``ok`` status — closes 5/23-SF P0 sweep item (c).

    Pre-fix: REJECTED status surfaced ONLY in ``report.md`` ("Refusing to
    promote — per the canonical-alpha framework, alpha-positive is a hard
    constraint, not a side-output"). No CW metric, no Telegram, no SNS.
    Operator only saw it if they read the report.

    Post-fix: status ∈ {``alpha_below_floor``, ``insufficient_data``,
    ``no_params``, ``no_improvement``, ``insufficient_trades``,
    ``insufficient_psr_confidence``, ``degraded``, ...} → publish a WARN
    alert via ``nousergon_lib.alerts.publish`` with dedup_key keyed on
    ``(run_date, status)`` so a recurring class doesn't N-spam the operator.
    Mirrors the stance_distribution + cost_report patterns.
    """
    import os
    if os.environ.get("ALPHA_ENGINE_DECISION_CAPTURE_SUPPRESS", "").lower() in (
        "1", "true", "yes", "on",
    ):
        return
    status = result.get("status")
    if status == "ok":
        return
    try:
        from ops_alerts import publish_ops_alert
    except ImportError as e:
        logger.warning(
            "[executor_optimizer] alerts publish skipped — ops_alerts unavailable: %s", e,
        )
        return
    run_date = config.get("run_date") or result.get("run_date") or "unknown"
    note = result.get("note") or result.get("degradation_reason") or "(no note)"
    # `degraded` is the "sweep_df missing entirely" wrapper status set
    # immediately above this function; `alpha_below_floor` is the
    # canonical-alpha hard-constraint rejection. Both are operator-
    # visible failures the canonical-alpha framework would want surfaced
    # before the next Saturday cycle.
    message = (
        f"executor_optimizer REJECTED on {run_date}: status={status}. "
        f"Note: {note}. "
        f"Live `executor_params.json` was NOT updated. "
        f"See backtester `report.md` ({run_date}) Executor Optimizer section "
        f"and ROADMAP item (c) of the 5/23-SF aggregate-cycle P0 sweep."
    )
    try:
        publish_result = publish_ops_alert(
            message,
            severity="warning",
            source="alpha-engine-backtester/evaluate.py::_run_executor_opt",
            dedup_key=f"executor_optimizer_rejected_{run_date}_{status}",
            dedup_window_min=1440,  # one alert per (run_date, status) per day
        )
        logger.info(
            "[executor_optimizer] REJECTED alert publish: sns_ok=%s "
            "any_ok=%s dedup_skipped=%s",
            publish_result.sns.ok,
            publish_result.any_ok,
            getattr(publish_result, "dedup_skipped", False),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[executor_optimizer] REJECTED alert publish failed (best-effort, "
            "swallowed): %s", e,
        )


def _run_executor_opt(config: dict, sweep_df, freeze: bool) -> dict:
    if sweep_df is None or (hasattr(sweep_df, "empty") and sweep_df.empty):
        result = {
            "status": "degraded",
            "degradation_reason": "no sweep_df available (simulation did not run or failed)",
        }
        _publish_executor_opt_rejection_alert(result, config)
        return result

    bucket = config.get("signals_bucket", "alpha-engine-research")
    current_params = read_params_pit_or_current(executor_optimizer, bucket, config)
    result = executor_optimizer.recommend(sweep_df, config, current_params=current_params)
    if result.get("status") == "ok":
        if freeze:
            result["apply_result"] = {"applied": False, "reason": "frozen (--freeze flag)"}
        else:
            # config#1017: thread the backfill run_date from config["_run_date"]
            # so the recommendation artifact keys to the backfilled trading day.
            result["apply_result"] = executor_optimizer.apply(
                result, bucket, config.get("_run_date"),
            )
    else:
        _publish_executor_opt_rejection_alert(result, config)
    return result


# ── Regression detection ─────────────────────────────────────────────────────


def _run_regression(
    config: dict,
    tracker: CompletenessTracker,
    sq_result: dict,
    portfolio_stats: dict | None,
    weight_result: dict | None,
    executor_rec: dict | None,
    veto_result: dict | None,
    freeze: bool,
    run_date: str,
) -> dict | None:
    """Run regression detection and save rolling metrics."""
    def _do_regression() -> dict:
        from optimizer.regression_monitor import (
            extract_metrics, save_rolling_metrics, save_promotion_baseline,
            check_regression,
        )
        bucket = config.get("signals_bucket", "alpha-engine-research")
        current_metrics = extract_metrics(portfolio_stats, sq_result)

        if current_metrics:
            save_rolling_metrics(bucket, run_date, current_metrics)

        promoted = []
        for label, res in [
            ("scoring_weights", weight_result),
            ("executor_params", executor_rec),
            ("predictor_params", veto_result),
        ]:
            if res and res.get("apply_result", {}).get("applied"):
                promoted.append(label)

        if promoted and current_metrics:
            save_promotion_baseline(bucket, current_metrics, promoted)

        if current_metrics and not freeze:
            return check_regression(
                bucket, current_metrics, config, run_date=run_date,
            ) or {"status": "ok"}
        return {"status": "ok", "note": "frozen or no metrics"}

    return tracker.run_module(
        "regression_monitor",
        _do_regression,
        required_inputs={"signal_quality": sq_result.get("status") == "ok"},
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    # flow-doctor default-on (lib v0.58.0): the outer guard catches the
    # uncaught raise from anywhere in the body and reports it before
    # re-raising. No-op when flow-doctor is inactive (dev/CI/pytest).
    with guard_entrypoint():
        _main_impl()


def _main_impl() -> None:
    args = _parse_args()

    # DATE_CONVENTIONS: normalize the run-date label to the NYSE trading day so
    # the evaluator reads/writes backtest/{trading_day}/ — aligned with the
    # Backtester + Parity stages (same spot RUN_DATE) and signals/{trading_day}.
    # Idempotent (no-op on the bash-normalized RUN_DATE). See L4466 + research #257.
    _orig_date = args.date
    args.date = resolve_trading_day(args.date)
    if args.date != _orig_date:
        logger.info("Normalized run-date %s (calendar) → %s (trading day)", _orig_date, args.date)

    # setup_logging already ran at module-top (see comment near the
    # nousergon_lib.logging import). Apply the user-requested level here.
    # Note: get_flow_doctor() retrieval was dropped — every fd.report()
    # call site lived in backtest.py, never in this module's body.
    # ERROR-level escalation still flows through the root-attached
    # FlowDoctorHandler from setup_logging.
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    _health_start = _time.time()

    config = load_config(args.config)
    # Stash run_date on config so diagnostic modules that don't already
    # take args directly (e.g. decision_capture_coverage) can read it.
    config["_run_date"] = args.date

    # Preflight: AWS_REGION + S3 bucket reachable. Evaluation reads
    # simulation artifacts from S3 (no ArcticDB), so keep the check
    # cheap and fail fast before any optimizer runs.
    from preflight import BacktesterPreflight
    BacktesterPreflight(
        bucket=config.get("signals_bucket", "alpha-engine-research"),
        mode="evaluate",
    ).run()

    # Initialize optimizer modules
    weight_optimizer.init_config(config)
    executor_optimizer.init_config(config)
    veto_analysis.init_config(config)
    research_optimizer.init_config(config)
    tech_weight_ablation.init_config(config)
    factor_blend_optimizer.init_config(config)

    # Set the assembler-cutover flag from config — when true, individual
    # optimizers' apply() skip their legacy live-key writes and the
    # assembler becomes the sole writer of config/{config_type}.json.
    # Default false so this PR ships the cutover mechanism dark; flip via
    # alpha-engine-config/backtester/config.yaml under the `assembler:` section.
    from optimizer.assembler import set_cutover_enabled as _set_cutover_enabled
    _set_cutover_enabled(
        config.get("assembler", {}).get("cutover_enabled", False),
    )

    # Initialize data sources and check availability
    avail = _init_data_sources(args, config)
    logger.info("Data availability: %s", {k: v for k, v in avail.items()})

    tracker = CompletenessTracker()

    # ── Default results ──────────────────────────────────────────────────
    sq_result: dict = {"status": "skipped"}
    regime_rows: list = []
    score_rows: list = []
    attr_result: dict = {"status": "skipped"}
    df_base = None
    diagnostics: dict = {}
    opt_results: dict = {}

    run_diagnostics = args.mode in ("all", "diagnostics") or args.module
    run_optimizers = args.mode in ("all", "optimize") or args.module

    # ── Signal quality pipeline ──────────────────────────────────────────
    if run_diagnostics and (not args.module or args.module == "signal-quality"):
        sq_result, regime_rows, score_rows, attr_result, df_base = _run_signal_quality(
            config, tracker, avail,
        )

    # ── Component diagnostics ────────────────────────────────────────────
    if run_diagnostics and not args.module:
        diagnostics = _run_diagnostics(config, tracker, avail, df_base)

    # ── Single module mode ───────────────────────────────────────────────
    if args.module and args.module != "signal-quality":
        # Run just the requested module
        if args.module in ("weight-optimizer", "veto-optimizer", "research-optimizer",
                           "trigger-optimizer", "predictor-sizing", "scanner-optimizer",
                           "team-optimizer", "cio-optimizer", "executor-optimizer"):
            run_optimizers = True
        # For diagnostic modules, they'd need to be run individually
        # This is handled by the diagnostics dict being empty

    # ── Optimizers ───────────────────────────────────────────────────────
    # config#1841: the optimizer stage is wrapped so the apply-audit artifact
    # below is emitted even when the stage itself raises (per-LOOP errors are
    # already isolated by tracker.run_module; this catches stage-level raises
    # such as write_optimizer_run_manifest's load-bearing S3 failure). The
    # original error re-raises after emission — except-log-emit-reraise, no
    # swallow.
    opt_stage_error: BaseException | None = None
    if run_optimizers:
        try:
            opt_results = _run_optimizers(
                config, tracker, avail, df_base,
                freeze=args.freeze,
                diagnostics=diagnostics,
            )
        except Exception as e:
            opt_stage_error = e
            logger.error(
                "Optimizer stage raised: %s — emitting apply-audit before re-raising", e,
            )

    # ── Assembler (optimizer-artifact-assembler arc) ────────────────────
    # Reads the per-optimizer recommendation artifacts written by each
    # optimizer's apply() during this run, applies merge precedence, and
    # writes config/{config_type}/assembled/{date}.json for audit. When
    # `assembler.cutover_enabled` is true in config, the assembler ALSO
    # writes the live key + _previous snapshot + dated history — and the
    # individual optimizers' apply() paths skip their legacy live writes
    # (gated by ``optimizer.assembler.is_cutover_enabled()``).
    # Failure is non-fatal: the assembler must not break the pipeline.
    assemble_result = None
    if run_optimizers and opt_stage_error is None and not args.freeze:
        try:
            from optimizer.assembler import assemble, is_cutover_enabled
            bucket = config.get("signals_bucket", "alpha-engine-research")
            assemble_result = assemble(
                bucket=bucket,
                config_type="executor_params",
                run_date=args.date,
                write_assembled=True,
            )
            logger.info(
                "Assembler run: status=%s, promoting=%d, frozen_restored=%d, "
                "cutover=%s",
                assemble_result.status,
                sum(
                    1 for v in assemble_result.artifacts_seen.values()
                    if v["promotion_intent"] == "promote"
                ),
                len(assemble_result.frozen_keys_restored),
                "ON" if is_cutover_enabled() else "OFF (shadow)",
            )
        except Exception as e:
            # Assembler failure must not break the pipeline.
            logger.warning(
                "Assembler run failed (non-fatal — pipeline continues): %s", e,
            )

    # ── Apply-audit artifact (config#1841) ───────────────────────────────
    # Unconditional per-loop outcome record for the four auto-apply loops
    # (promoted / blocked-by-guardrail / insufficient_data / disabled /
    # error) → config/apply_audit/{date}.json + latest.json. The S3 write
    # rides the same args.upload gate as sibling artifacts (--freeze /
    # local runs build + log the audit without persisting); build/log is
    # unconditional so a blocked apply can never again be silent.
    if run_optimizers:
        from optimizer.apply_audit import emit_apply_audit
        emit_apply_audit(
            bucket=config.get("signals_bucket", "alpha-engine-research"),
            run_date=args.date,
            opt_results=opt_results,
            assembler_result=assemble_result,
            upload=bool(getattr(args, "upload", False)) and not args.freeze,
            run_error=opt_stage_error,
        )
        if opt_stage_error is not None:
            raise opt_stage_error

    # ── Regression detection ─────────────────────────────────────────────
    regression_result = _run_regression(
        config, tracker, sq_result,
        config.get("_portfolio_stats"),
        opt_results.get("weight_result"),
        opt_results.get("executor_rec"),
        opt_results.get("veto_result"),
        args.freeze, args.date,
    )

    # ── Report ───────────────────────────────────────────────────────────
    # Track whether the report/upload/email block actually completed so the
    # health status and process exit code reflect reality. A prior version
    # swallowed exceptions in the try block below and still wrote
    # `evaluator → ok` in the finally block — silently masking crashes like
    # the grading.compute_scorecard AttributeError on 2026-04-11.
    report_ok = False
    try:
        portfolio_stats = config.get("_portfolio_stats")
        predictor_stats = config.get("_predictor_stats")

        pipeline_health = {
            "db_pull_status": config.get("_db_pull_status"),
            "staleness_warning": portfolio_stats.get("staleness_warning") if portfolio_stats else None,
            "coverage": portfolio_stats.get("coverage") if portfolio_stats else None,
        }

        # Compute the evaluator-revamp metric bundles. Each piece
        # graceful-degrades to insufficient_data when its inputs are
        # missing, so calls are unconditional and the grading layer
        # drops absent metrics from the composite.
        from analysis.team_skill_metrics import (
            compute_portfolio_calibration,
            compute_portfolio_excursion_summary,
            compute_team_metrics,
        )

        team_metrics = {}
        portfolio_calibration = {"status": "insufficient_data"}
        portfolio_excursion = {"status": "insufficient_data"}
        risk_ratio_ci_result = {
            "status": "insufficient_data",
            "n_samples": 0,
            "ratios": {},
            "all_magnitude_certain": False,
            "note": "evaluator-revamp metric bundle did not run",
        }
        try:
            team_lift_for_metrics = (
                diagnostics.get("e2e_lift") or {}
            ).get("team_lift") or []
            prices_for_metrics = config.get("_prices")
            ohlc_for_metrics = config.get("_ohlcv_by_ticker")
            spy_returns_for_metrics = None
            spy_prices_for_metrics = config.get("_spy_prices")
            if isinstance(spy_prices_for_metrics, pd.Series) and not spy_prices_for_metrics.empty:
                spy_returns_for_metrics = spy_prices_for_metrics.pct_change().dropna()

            team_metrics = compute_team_metrics(
                team_lift=team_lift_for_metrics,
                score_performance_df=df_base,
                prices=prices_for_metrics,
                spy_daily_returns=spy_returns_for_metrics,
                ohlc=ohlc_for_metrics,
                horizon_days=10,
            )
            portfolio_calibration = compute_portfolio_calibration(df_base)
            portfolio_excursion = compute_portfolio_excursion_summary(
                df_base, ohlc_for_metrics, horizon_days=10,
            )

            # Risk-ratio magnitude-certainty monitor (config#976, L4558): block-
            # bootstrap CIs for Sharpe/Sortino/Information Ratio over the same
            # portfolio-wide daily-return sleeve derived from team_lift picks
            # above (no new data read). ALWAYS-EMIT — compute_risk_ratio_ci's own
            # insufficient_data/no_benchmark degrade paths handle thin data; this
            # try/except only guards the aggregation step itself.
            pf_returns_for_risk_ratio = _portfolio_daily_returns_from_team_lift(
                team_lift_for_metrics, prices_for_metrics, horizon_days=10,
            )
            risk_ratio_ci_result = compute_risk_ratio_ci(
                pf_returns_for_risk_ratio, spy_returns_for_metrics,
            )
        except Exception as e:
            logger.warning("evaluator-revamp metric bundle failed: %s", e)

        # Decision-stream entropy (config#1151 Batch C) — the executor-tile
        # action_entropy diagnostic. Computed over the finalized-signal frame's
        # decision label (stance/conviction); ALWAYS-EMIT so the report card
        # distinguishes "producer didn't run" from "ran, decision distribution
        # collapsed / too few labelled decisions". Feeds both the self-grade
        # scorecard below and the persisted action_entropy.json save() reads.
        action_entropy_result = compute_action_entropy_artifact(df_base)

        # Compute grading scorecard
        from analysis.grading import compute_scorecard
        grading_result = compute_scorecard(
            signal_quality=sq_result,
            e2e_lift=diagnostics.get("e2e_lift"),
            macro_eval=diagnostics.get("macro_eval"),
            score_calibration=diagnostics.get("score_calibration"),
            veto_result=opt_results.get("veto_result"),
            veto_value=diagnostics.get("veto_value"),
            trigger_scorecard=diagnostics.get("trigger_scorecard"),
            shadow_book=diagnostics.get("shadow_book"),
            exit_timing=diagnostics.get("exit_timing"),
            sizing_ab=None,  # simulation-only
            predictor_sizing=opt_results.get("predictor_sizing"),
            portfolio_stats=portfolio_stats,
            scanner_opt=opt_results.get("scanner_opt"),
            cio_opt=opt_results.get("cio_opt"),
            team_metrics=team_metrics or None,
            calibration_diagnostics=portfolio_calibration if portfolio_calibration.get("status") == "ok" else None,
            excursion_summary=portfolio_excursion if portfolio_excursion.get("status") == "ok" else None,
            # Decision-stream entropy off the finalized-signal stance/conviction
            # labels (config#1151 Batch C). Only the "ok" result is gradable; any
            # other status (insufficient_data / no_decision_stream) leaves the
            # self-grade component a transparent N/A.
            action_entropy=action_entropy_result if action_entropy_result.get("status") == "ok" else None,
        )

        # Build report using existing reporter (includes completeness)
        report_md = build_report(
            run_date=args.date,
            signal_quality=sq_result,
            regime_analysis=regime_rows,
            score_analysis=score_rows,
            attribution=attr_result,
            portfolio_stats=portfolio_stats,
            sweep_df=config.get("_sweep_df"),
            weight_result=opt_results.get("weight_result"),
            config=config,
            predictor_stats=predictor_stats,
            predictor_sweep_df=config.get("_predictor_sweep_df"),
            veto_result=opt_results.get("veto_result"),
            executor_rec=opt_results.get("executor_rec"),
            regression_result=regression_result,
            pipeline_health=pipeline_health,
            e2e_lift=diagnostics.get("e2e_lift"),
            trigger_scorecard=diagnostics.get("trigger_scorecard"),
            alpha_dist=diagnostics.get("alpha_dist"),
            score_calibration=diagnostics.get("score_calibration"),
            veto_value=diagnostics.get("veto_value"),
            shadow_book=diagnostics.get("shadow_book"),
            exit_timing=diagnostics.get("exit_timing"),
            macro_eval=diagnostics.get("macro_eval"),
            decision_capture_coverage=diagnostics.get("decision_capture_coverage"),
            executor_decision_capture_coverage=diagnostics.get("executor_decision_capture_coverage"),
            provenance_grounding=diagnostics.get("provenance_grounding"),
            quant_rank_quality=diagnostics.get("quant_rank_quality"),
            cio_rule_tag_precision=diagnostics.get("cio_rule_tag_precision"),
            agent_justification=diagnostics.get("agent_justification"),
            trigger_opt=opt_results.get("trigger_opt"),
            predictor_sizing=opt_results.get("predictor_sizing"),
            scanner_opt=opt_results.get("scanner_opt"),
            team_opt=opt_results.get("team_opt"),
            cio_opt=opt_results.get("cio_opt"),
            tech_weight_ablation=opt_results.get("tech_weight_ablation"),
            grading=grading_result,
            confusion_matrix=diagnostics.get("confusion_matrix"),
            post_trade=diagnostics.get("post_trade"),
            monte_carlo=diagnostics.get("monte_carlo"),
            factor_blend_sensitivity=diagnostics.get("factor_blend_sensitivity"),
            factor_blend_counterfactual_replay=diagnostics.get(
                "factor_blend_counterfactual_replay"
            ),
            barrier_coherence=diagnostics.get("barrier_coherence"),
        )

        # Prepend completeness summary to report
        completeness = tracker.summary()
        completeness_header = [
            "## Evaluator Completeness",
            "",
            f"| Status | Count |",
            f"|--------|-------|",
            f"| OK | {completeness.get('ok', 0)} |",
            f"| Degraded | {completeness.get('degraded', 0)} |",
            f"| Skipped | {completeness.get('skipped', 0)} |",
            f"| Error | {completeness.get('error', 0)} |",
            f"| **Total** | **{completeness.get('total', 0)}** |",
            "",
        ]
        degraded = tracker.degraded_modules()
        if degraded:
            completeness_header.append(f"**Degraded modules:** {', '.join(degraded)}")
            completeness_header.append("")
        failed = tracker.failed_modules()
        if failed:
            completeness_header.append(f"**Failed modules:** {', '.join(failed)}")
            completeness_header.append("")

        report_md = "\n".join(completeness_header) + "\n" + report_md

        # Append LLM cost report (PR 4 of cost-telemetry workstream).
        # Reads decision_artifacts/_cost/{date}/cost.parquet from the
        # research bucket; emits a placeholder section if the parquet is
        # absent (capture flag off, no Saturday SF, etc.) so the cost
        # surface is always visible to operators.
        try:
            from analysis.cost_report import build_cost_section
            cost_section = build_cost_section(args.date)
            report_md = report_md + "\n" + cost_section
        except Exception as cost_err:
            # Renderer hard-fail (corrupt parquet, IAM denial, etc.) —
            # log loud and surface in the email rather than crashing the
            # whole evaluator. Per feedback_no_silent_fails this is the
            # narrow exception: the evaluator's primary purpose (signal
            # quality / param sweep) shouldn't be blocked by a cost-
            # report render error. The error message lands in the email.
            logger.error(
                "[cost_report] section render failed: %s — emitting "
                "error placeholder so operators see the regression",
                cost_err,
            )
            report_md = report_md + "\n" + "\n".join([
                "## LLM cost report",
                "",
                f"- _Cost report render failed: `{cost_err}`._",
                "  Investigate `analysis/cost_report.py` + the parquet at "
                f"`s3://alpha-engine-research/decision_artifacts/_cost/{args.date}/cost.parquet`.",
                "",
            ])

        # Judge-calibration κ section (ROADMAP L480). Embeds the
        # pre-rendered markdown written weekly by alpha-engine-research
        # (evals/calibration_kappa.py) from the operator review corpus.
        # Always renders a placeholder if the report is absent, so the
        # calibration surface is always visible. build_calibration_section
        # never raises, but the call is still guarded to match the cost
        # section's contract (the evaluator's primary deliverables must
        # not be blocked by a section render).
        try:
            from analysis.calibration_report import build_calibration_section
            report_md = report_md + "\n" + build_calibration_section()
        except Exception as cal_err:  # noqa: BLE001 — see cost section above
            logger.error(
                "[calibration_report] section render failed: %s — emitting "
                "error placeholder so operators see the regression",
                cal_err,
            )
            report_md = report_md + "\n" + "\n".join([
                "## Judge calibration (κ)",
                "",
                f"- _Calibration report render failed: `{cal_err}`._",
                "  Investigate `analysis/calibration_report.py` + the report at "
                "`s3://alpha-engine-research/decision_artifacts/_calibration/_report/latest/kappa.md`.",
                "",
            ])

        # 10y rolling-window regime-parameter STABILITY section (config#952).
        # Slides a true 10-year window across score_performance and re-
        # identifies each regime's best stance parameterization per window,
        # then reports cross-window stability (a sign-flipping parameter is
        # regime noise, not a durable regime parameter). Observability only —
        # never auto-applied to the factor blend (curve-fitting firewall).
        # Self-loads from research_db and never raises; guarded to match the
        # calibration/cost section contract.
        try:
            from analysis.rolling_regime_params import (
                build_rolling_regime_params_report_section,
            )
            report_md = report_md + "\n" + build_rolling_regime_params_report_section(
                config.get("research_db")
            )
        except Exception as rrp_err:  # noqa: BLE001 — see cost section above
            logger.error(
                "[rolling_regime_params] section render failed: %s — emitting "
                "error placeholder so operators see the regression",
                rrp_err,
            )
            report_md = report_md + "\n" + "\n".join([
                "## 10y rolling regime parameters (stability)",
                "",
                f"- _Rolling-regime-params render failed: `{rrp_err}`._",
                "  Investigate `analysis/rolling_regime_params.py`.",
                "",
            ])

        # Attribution sample-adequacy PERSISTENCE section (config#946 part 2).
        # sample_size_adequacy grades the per-cycle snapshot; this watches the
        # cross-cycle trailing run of attribution `insufficient_data` and warns
        # once it persists past the threshold (structural under-powering vs a
        # transient early-cohort gap). Appends to S3 history only when uploading
        # (matches write-as-you-compute gating). Guarded + always-emit, same
        # contract as the cost / calibration sections above.
        try:
            from analysis.attribution_persistence import (
                record_and_evaluate,
                render_attribution_persistence_section,
            )
            _attr_persistence = record_and_evaluate(
                attr_result,
                run_date=args.date,
                bucket=config.get("output_bucket", "alpha-engine-research"),
                upload=bool(getattr(args, "upload", False)),
            )
            report_md = report_md + "\n" + render_attribution_persistence_section(_attr_persistence)
        except Exception as ap_err:  # noqa: BLE001 — see cost section above
            logger.error(
                "[attribution_persistence] section render failed: %s — emitting "
                "error placeholder so operators see the regression",
                ap_err,
            )
            report_md = report_md + "\n" + "\n".join([
                "## Attribution sample adequacy (persistence)",
                "",
                f"- _Attribution persistence section render failed: `{ap_err}`._",
                "  Investigate `analysis/attribution_persistence.py` + the history at "
                "`s3://alpha-engine-research/decision_artifacts/_attribution_adequacy/history.jsonl`.",
                "",
            ])

        # Report-card Batch C producers (config#1151) — backtester self-grade
        # inputs. Pure-compute over the weight optimizer's already-computed
        # result (no new data read); ALWAYS-EMIT so the evaluator's backtester
        # tile distinguishes "producer didn't run" from "ran, optimizer had no
        # usable recommendation / too little history". optimizer_churn = largest
        # proposed per-param move vs the guardrail cap (config thrash);
        # walk_forward_stability = cross-week recommendation-reversal rate
        # (weekly drift). Both grade in crucible-evaluator's backtester tile.
        weight_result = opt_results.get("weight_result")
        optimizer_churn_result = compute_optimizer_churn(weight_result)
        walk_forward_stability_result = compute_walk_forward_stability(weight_result)

        # Save
        out_dir = save(
            report_md=report_md,
            signal_quality=sq_result,
            score_analysis=score_rows,
            sweep_df=config.get("_sweep_df"),
            attribution=attr_result if attr_result.get("status") not in ("skipped",) else None,
            run_date=args.date,
            results_dir=config.get("results_dir", "results"),
            grading=grading_result,
            trigger_scorecard=diagnostics.get("trigger_scorecard"),
            shadow_book=diagnostics.get("shadow_book"),
            exit_timing=diagnostics.get("exit_timing"),
            behavioral_anomaly=diagnostics.get("behavioral_anomaly"),
            sample_size=compute_sample_size_adequacy(sq_result, attr_result),
            risk_ratio_ci=risk_ratio_ci_result,
            action_entropy=action_entropy_result,
            optimizer_churn=optimizer_churn_result,
            walk_forward_stability=walk_forward_stability_result,
            significance_observe=_collect_significance_observe(opt_results),
            e2e_lift=diagnostics.get("e2e_lift"),
            veto_result=opt_results.get("veto_result"),
            confusion_matrix=diagnostics.get("confusion_matrix"),
            post_trade=diagnostics.get("post_trade"),
            monte_carlo=diagnostics.get("monte_carlo"),
            decision_capture_coverage=diagnostics.get("decision_capture_coverage"),
            executor_decision_capture_coverage=diagnostics.get("executor_decision_capture_coverage"),
            measurement_coverage=diagnostics.get("measurement_coverage"),
            provenance_grounding=diagnostics.get("provenance_grounding"),
            quant_rank_quality=diagnostics.get("quant_rank_quality"),
            cio_rule_tag_precision=diagnostics.get("cio_rule_tag_precision"),
            agent_justification=diagnostics.get("agent_justification"),
            barrier_coherence=diagnostics.get("barrier_coherence"),
            score_calibration=diagnostics.get("score_calibration"),
            macro_eval=diagnostics.get("macro_eval"),
            attractiveness_eval=diagnostics.get("attractiveness_eval"),
            team_metrics=team_metrics or None,
            calibration_diagnostics=portfolio_calibration,
            excursion_summary=portfolio_excursion,
            # B1d — persist the optimizer/diagnostic inputs the evaluator report
            # card reads over S3 (previously computed-but-unpersisted → graded N/A).
            veto_value=diagnostics.get("veto_value"),
            predictor_sizing=opt_results.get("predictor_sizing"),
            scanner_opt=opt_results.get("scanner_opt"),
            cio_opt=opt_results.get("cio_opt"),
            # Write-as-you-compute (config#1190): when uploading, push each
            # tile artifact to S3 the moment it is persisted so a mid-Saturday
            # interruption doesn't strand every computed tile. Gated on the
            # SAME `args.upload` flag as the terminal sweep below, so
            # local-only / dry-run runs incur no per-tile S3 cost.
            upload_bucket=(
                config.get("output_bucket", "alpha-engine-research")
                if args.upload else None
            ),
            upload_prefix=config.get("output_prefix", "evaluation"),
        )

        # Save completeness manifest
        completeness_path = out_dir / "completeness.json"
        completeness_path.write_text(tracker.to_json())
        logger.info("Wrote %s", completeness_path)

        print(f"\nEvaluation report saved to {out_dir}/")
        print(f"\n{'='*60}")
        print(report_md[:2000])
        if len(report_md) > 2000:
            print(f"\n... (truncated — see {out_dir}/report.md for full report)")

        if args.upload:
            upload_to_s3(
                local_dir=out_dir,
                bucket=config.get("output_bucket", "alpha-engine-research"),
                prefix=config.get("output_prefix", "evaluation"),
                run_date=args.date,
            )
            print(f"\nUploaded to s3://{config.get('output_bucket')}/{config.get('output_prefix', 'evaluation')}/{args.date}/")

            # Grade history
            if grading_result and grading_result.get("status") in ("ok", "partial"):
                try:
                    from analysis.grade_history import append_grades
                    append_grades(grading_result, args.date, config.get("output_bucket", "alpha-engine-research"))
                except Exception as e:
                    logger.warning("Grade history update failed (non-fatal): %s", e)

        # Evaluator email — now a THIN digest (System Report Card + What Changed
        # + completeness) that deep-links to the console Analysis page for the
        # full detail, mirroring the EOD and model-zoo patterns. The full
        # report.md is still built + uploaded for the console + the link. The
        # Backtester is a SEPARATE Saturday-SF task and sends its OWN digest from
        # backtest.py — the two are intentionally NOT bundled.
        sender = config.get("email_sender")
        recipients = config.get("email_recipients", [])
        if sender and recipients:
            digest_md = build_digest(
                run_date=args.date,
                title="Evaluation Digest",
                grading=grading_result,
                weight_result=opt_results.get("weight_result"),
                veto_result=opt_results.get("veto_result"),
                executor_rec=opt_results.get("executor_rec"),
                regression_result=regression_result,
                completeness=tracker.summary(),
                degraded_modules=tracker.degraded_modules(),
                failed_modules=tracker.failed_modules(),
                significance_observe=_collect_significance_observe(opt_results),
            )
            send_digest_email(
                run_date=args.date,
                digest_md=digest_md,
                sender=sender,
                recipients=recipients,
                product_name="Evaluator",
                report_prefix=config.get("output_prefix", "evaluation"),
                status=sq_result.get("status", "ok"),
                s3_bucket=config.get("output_bucket") if args.upload else None,
                # config#2291: same dedup rationale as backtest.py's digest
                # call — a watch-rerun of the Evaluator SF state for the same
                # trading_day must not re-send this email. Keyed on run_date
                # only, own "evaluator-digest" namespace so it never collides
                # with the Backtester's dedup_key for the same date.
                dedup_key=f"evaluator-digest:{args.date}",
            )

        report_ok = True

    except Exception as e:
        logger.error("Report/upload/email failed: %s", e)
        import traceback
        traceback.print_exc()
        # Do NOT swallow — we still need the finally block to write health
        # status (reflecting the failure) and then we raise at the end of
        # main() so the spot-run's exit code is non-zero and SSM reports
        # Failed to the Step Function.
    finally:
        # Health status
        try:
            from nousergon_lib.health import Deliverable, write_health
            summary = tracker.summary()
            configs_applied = []
            for label, key in [
                ("scoring_weights", "weight_result"),
                ("executor_params", "executor_rec"),
                ("predictor_params", "veto_result"),
            ]:
                res = opt_results.get(key)
                if res and res.get("apply_result", {}).get("applied"):
                    configs_applied.append(label)

            stage_warnings = []
            if summary.get("error", 0) > 0:
                stage_warnings.append(
                    f"{summary.get('error', 0)} evaluation stage(s) errored"
                )

            bucket = config.get("signals_bucket", "alpha-engine-research")
            write_health(
                module_name="evaluator",
                deliverables=[
                    Deliverable(
                        name="evaluation_stages",
                        required=True,
                        produced=summary.get("ok", 0) > 0,
                    ),
                    Deliverable(
                        name="evaluation_report",
                        required=True,
                        produced=report_ok,
                    ),
                ],
                run_date=args.date,
                duration_seconds=_time.time() - _health_start,
                summary={
                    "mode": args.mode,
                    "completeness": summary,
                    "configs_applied": configs_applied,
                },
                warnings=stage_warnings or None,
                bucket=bucket,
            )
        except Exception as _he:
            logger.warning("Health status write failed: %s", _he)

        if args.stop_instance:
            import urllib.request
            try:
                token = urllib.request.urlopen(
                    urllib.request.Request(
                        "http://169.254.169.254/latest/api/token",
                        headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
                        method="PUT",
                    ), timeout=5,
                ).read().decode()
                instance_id = urllib.request.urlopen(
                    urllib.request.Request(
                        "http://169.254.169.254/latest/meta-data/instance-id",
                        headers={"X-aws-ec2-metadata-token": token},
                    ), timeout=5,
                ).read().decode()
                logger.info("Stopping instance %s", instance_id)
                boto3.client("ec2").stop_instances(InstanceIds=[instance_id])
            except Exception as e:
                logger.error("Failed to stop instance: %s", e)

    # Hard-fail the process if the report/upload/email block crashed.
    # Happens AFTER the finally block writes the health marker and stops
    # the spot instance (if requested), so monitoring sees the failure and
    # we don't leak compute. Matches the no-silent-fails and
    # hard-fail-until-stable preferences.
    if not report_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
