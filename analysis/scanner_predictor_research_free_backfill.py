"""analysis/scanner_predictor_research_free_backfill.py — research-free meta-
ensemble backfill producer (config#1405, arm 4 of the agentic-ablation ladder,
build items 1+2).

Populates ``predictor_outcomes_research_free(ticker, prediction_date,
predicted_alpha, n_research_features_missing)`` in research.db: for every
(ticker, eval_date) where ``scanner_evaluations.quant_filter_pass=1``, runs the
FULL meta-ensemble (``crucible-predictor/model/meta_model.py::MetaModel.
predict_single``) with the 4 research meta-features (``research_calibrator_prob``,
``research_composite_score``, ``research_conviction``, ``sector_macro_modifier``)
omitted -> 0.0, per the issue's research-free definition. The 9 (or however many
the deployed model's own ``feature_names`` schema carries) deterministic/macro
features are computed for real from ArcticDB + the predictor's own deterministic
Layer-1 scorers, matching ``inference/stages/run_inference.py::_run_meta_inference``
as closely as a backfill (rather than a live daily run) reasonably can.

This is the consumer's ("build item 3", ``analysis/end_to_end.py::
_scanner_then_predictor_topN``, shipped in crucible-backtester#419) missing
producer half. That consumer is a pure, fail-soft READ against this table; this
module is the WRITE side.

Design notes / deliberate scope decisions:

- **The S3 parquet (``ARTIFACT_KEY``), not research.db, is the durable output.**
  Every spot stage pulls research.db from S3 to a throwaway temp copy and never
  pushes it back (``pipeline_common.init_research_db`` — push-back would race
  the research module's own S3 backups of a DB it owns). The producer
  (PredictorBacktest box) and consumer (Evaluator box, ``evaluate.py`` ->
  ``end_to_end._scanner_then_predictor_topN``) are SEPARATE instances with
  separate pulls, so the sqlite table is only ever a local materialization of
  the artifact: ``run_backfill`` seeds from it (idempotency) and re-exports
  after writing; the consumer hydrates via ``materialize_from_s3`` before
  reading. First live run 2026-07-11 shipped without this and the producer's
  writes could never have reached the consumer.
- **Weights come from S3, never a checkout-relative path.** The deployed
  champion artifacts (``predictor/weights/meta/meta_model.pkl`` /
  ``volatility_model.txt``) are retrained every Saturday and live in S3; the
  sibling predictor checkout on the spot box carries code only — no synced
  ``weights/`` dir (the 2026-07-11 first live run failed in 0.01s on exactly
  that assumption). ``predictor_path`` is used solely for ``sys.path`` imports.
- **Model-schema-driven, not META_FEATURES-hardcoded.** The live deployed
  ``meta_model.pkl`` is free to swap Layer-1 components (e.g. the 2026-06-15
  cutover from ``momentum_score``/``expected_move`` to ``residual_momentum_score``
  observed in production — see ``model_zoo/spec-residual-mom``). Rather than
  hardcode the module-level ``META_FEATURES`` list, this producer reads the
  LOADED model's own ``mm._feature_names`` (the same attribute
  ``predict_single`` itself falls back to) and builds exactly that feature
  set via ``_assemble_research_free_features``, computing each recognized
  name (``momentum_score`` / ``residual_momentum_score`` / ``expected_move`` /
  ``macro_*`` / ``regime_intensity_z``) from real per-ticker/market data, else
  zero-filling with a warning for any unrecognized name (mirrors
  ``predict_single``'s own ``features.get(f, 0.0)`` graceful-degrade contract,
  and ``run_inference.py::_sanitize_meta_features``'s neutral-impute policy).
- **Cross-sectional rank-normalization is per calendar date over the FULL
  ArcticDB universe that day** (not just the scanner-passing subset) for the
  volatility GBM's ``expected_move`` input — mirrors
  ``inference/stages/run_inference.py``'s ``cross_sectional_rank_normalize``
  batch (a single-ticker-in-isolation rank is meaningless / would place
  every ticker at the 50th percentile).
- **Idempotent / skip-if-cached**: before computing, reads the
  ``(ticker, prediction_date)`` keys already present in
  ``predictor_outcomes_research_free`` and skips them — a re-run (e.g. after
  a new week's Scanner cycle) only computes the delta.
- **Scanner-passing universe sourced from S3, not sqlite (config#3053).**
  ``_pending_universe`` reads ``candidates/{run_date}/candidates.json``'s
  ``scanner_eval_log`` directly rather than the ``scanner_evaluations``
  sqlite table — that table's only writer (the six-team Research graph) was
  removed from the weekly SF by the 2026-07-14 config#1580 restructure,
  which silently orphaned this producer and froze the live champion feed for
  a week before it surfaced as a weekday trading incident. The standalone
  Scanner (crucible-research PR#425) writes the equivalent per-ticker gate
  verdict weekly straight to S3 — a versioned artifact contract, preferred
  over a removed component's private DB table.
- **ArcticDB is the sole feature source** (no S3-parquet fallback, matching
  the rest of the post-2026-04-16 pipeline) via ``nousergon_lib.arcticdb``.
  Contrary to the issue's macOS-vs-Saturday-spot-box framing, this store is
  S3-backed and reachable from ANY host with AWS credentials + the
  ``arcticdb`` package (confirmed reachable from a Linux CI runner during
  this build — see the PR description for the live smoke-test evidence);
  the "spot box only" framing in the issue was about macOS incompatibility
  and a since-removed local-parquet fallback, not a network/EBS locality
  constraint.
"""

from __future__ import annotations

import io
import json
import logging
import sqlite3
import sys
import tempfile
from datetime import date as _date
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

# The 4 research meta-features this arm omits (constants mirrored from
# crucible-predictor's model/meta_model.py::RESEARCH_META_FEATURES so this
# module has no non-optional cross-repo import at module-load time; the sole
# runtime cross-repo import is the MetaModel class itself, loaded lazily
# inside run_backfill via the same sys.path.insert idiom every other
# predictor-consuming phase in this repo uses).
RESEARCH_META_FEATURES = frozenset(
    {
        "research_calibrator_prob",
        "research_composite_score",
        "research_conviction",
        "sector_macro_modifier",
    }
)

# Baseline volatility-GBM input columns (predictor config.py
# ::_BASELINE_VOLATILITY_FEATURES) — also present verbatim in the ArcticDB row.
# A deployed predictor config could override this list via l1_features.volatility;
# _compute_expected_move reads the loaded scorer's own booster.feature_name()
# instead of this constant when a scorer is available, so this is only the
# graceful-degrade default when no volatility scorer could be loaded.
_BASELINE_VOLATILITY_FEATURES = (
    "atr_14_pct", "realized_vol_20d", "vol_ratio_10_60",
    "iv_rank", "dist_from_52w_high", "dist_from_52w_low",
)

TABLE_NAME = "predictor_outcomes_research_free"

# Canonical S3 persistence for the backfill output. research.db is pulled from
# S3 to a THROWAWAY temp copy on every spot stage (pipeline_common.init_research_db)
# and NEVER pushed back — the PredictorBacktest box (producer, backtest.py
# --mode=predictor-backtest) and the Evaluator box (consumer, evaluate.py ->
# end_to_end._scanner_then_predictor_topN) each pull their OWN copy. A row
# written only to the producer's local sqlite therefore evaporates with the
# box; this parquet is the durable wire contract between the two stages. Both
# sides materialize it into their local research.db copy via
# materialize_from_s3().
ARTIFACT_KEY = "predictor/research_free_backfill/predictor_outcomes_research_free.parquet"

# Deployed champion weight artifacts (retrained every Saturday by
# PredictorTraining). The S3 objects ARE the champion; a sibling-checkout
# weights/ dir is not synced on the backtester spot box (first live run
# 2026-07-11 failed on exactly that assumption in 0.01s), so weights are
# always fetched from S3 here — same posture as
# synthetic/predictor_backtest.py::download_gbm_model.
_WEIGHTS_PREFIX = "predictor/weights/meta/"

# config#3053 root-cause: the scanner-passing universe used to come from the
# ``scanner_evaluations`` sqlite table, whose only writer
# (``crucible-research graph/research_graph.py::write_scanner_evaluations``,
# the six-team Research graph) was removed from the weekly SF by the
# 2026-07-14 config#1580 restructure — orphaning this producer's source and
# silently freezing the live champion feed for a week before it surfaced as a
# weekday StaleChampionFeedError trading incident. The standalone Scanner
# (crucible-research PR#425, ``data/scanner_orchestrator.py``) writes the
# same per-ticker gate verdict weekly (``saturday_sf`` cadence, ARTIFACT_
# REGISTRY ``scanner_candidates_json``) directly to S3 as
# ``candidates/{run_date}/candidates.json::scanner_eval_log`` — a versioned
# S3 artifact contract, preferred over pulling from a removed component's
# private DB table (M0 contract preference). ``_pending_universe`` below
# reads THIS artifact now, not the dead table.
_CANDIDATES_PREFIX = "candidates"


class NoCandidatesArtifactError(RuntimeError):
    """Raised when no ``candidates.json`` exists anywhere in the lookback
    window — structurally distinct from a legitimate empty pending set
    (every known passing row already backfilled). The caller (``run_backfill``)
    treats this the same as the old "scanner_evaluations table missing" case:
    a visible ``status: skipped``, not a silent zero-row "ok"."""


class StaleChampionFeedError(RuntimeError):
    """Raised by :func:`assert_champion_feed_fresh` when the research-free
    parquet's newest ``prediction_date`` is already outside the champion
    freshness window right after a backfill attempt. Independently defined
    (not imported) from ``crucible-executor/executor/champion.py``'s
    same-named exception — same posture as that module's own
    ``CHALLENGER_SELECTION_LATEST_KEY`` literal: no non-optional cross-repo
    import, the S3 key + freshness contract is the shared surface. Exists so
    a producer-side gap is caught same-day (Saturday backfill) instead of
    only showing up days later as a weekday order-pipeline incident
    (config#3053)."""


def _list_recent_candidate_dates(
    bucket: str,
    *,
    region: str | None = None,
    s3_client=None,
    lookback_days: int = 120,
) -> list[str]:
    """List ``YYYY-MM-DD`` date strings under ``candidates/{date}/`` in S3
    that are within ``lookback_days`` of today (UTC). Scanner writes one
    ``candidates.json`` per week (``saturday_sf`` cadence), so even a
    multi-year lookback stays well under a single ``list_objects_v2`` page;
    pagination is still handled defensively.
    """
    s3 = s3_client or boto3.client("s3", **({"region_name": region} if region else {}))
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=lookback_days)
    dates: list[str] = []
    continuation_token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": f"{_CANDIDATES_PREFIX}/", "Delimiter": "/"}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        resp = s3.list_objects_v2(**kwargs)
        for cp in resp.get("CommonPrefixes") or []:
            parts = cp.get("Prefix", "").strip("/").split("/")
            if len(parts) != 2:
                continue
            try:
                d = _date.fromisoformat(parts[1])
            except ValueError:
                continue
            if d >= cutoff:
                dates.append(parts[1])
        if resp.get("IsTruncated"):
            continuation_token = resp.get("NextContinuationToken")
        else:
            break
    return sorted(dates)


def assert_champion_feed_fresh(
    bucket: str,
    *,
    run_date: str,
    max_days: int = 8,
    s3_client=None,
    region: str | None = None,
) -> None:
    """Raise :class:`StaleChampionFeedError` if the research-free parquet's
    newest ``prediction_date`` is more than ``max_days`` calendar days before
    ``run_date`` — the SAME calendar-day-diff check
    ``executor/champion.py::_check_freshness`` applies at trade time, run
    here immediately after the Saturday backfill so a producer-side gap is
    caught same-day instead of surfacing as a weekday order-pipeline
    incident days later (config#3053).

    Only meaningful for callers that already know the LIVE champion depends
    on this feed (``config/producer_champion.json`` champion ==
    ``scanner_predictor_direct``) — call unconditionally and this would
    wrongly fail the OBSERVE-only arm while it isn't the live champion.
    """
    s3 = s3_client or boto3.client("s3", **({"region_name": region} if region else {}))
    try:
        obj = s3.get_object(Bucket=bucket, Key=ARTIFACT_KEY)
        body = obj["Body"].read()
    except (ClientError, BotoCoreError) as exc:
        raise StaleChampionFeedError(
            f"scanner_predictor_direct is the live champion but the research-free "
            f"parquet s3://{bucket}/{ARTIFACT_KEY} is unreadable: {exc}. A stale/"
            "missing champion feed must not trade silently."
        ) from exc

    try:
        df = pd.read_parquet(io.BytesIO(body))
    except Exception as exc:  # noqa: BLE001 — any parse failure must raise
        raise StaleChampionFeedError(
            f"scanner_predictor_direct is the live champion but s3://{bucket}/"
            f"{ARTIFACT_KEY} failed to parse: {exc}"
        ) from exc

    if df.empty or "prediction_date" not in df.columns:
        raise StaleChampionFeedError(
            f"scanner_predictor_direct is the live champion but s3://{bucket}/"
            f"{ARTIFACT_KEY} is empty or missing the prediction_date column"
        )

    latest = pd.Timestamp(df["prediction_date"].max()).date()
    run_d = _date.fromisoformat(run_date)
    age_days = (run_d - latest).days
    if age_days > max_days:
        raise StaleChampionFeedError(
            f"scanner_predictor_direct champion feed is stale after this week's "
            f"backfill attempt: freshest prediction_date={latest} is {age_days} "
            f"calendar day(s) before run_date={run_date} (max allowed {max_days}). "
            "The weekly producer did not refresh the live champion feed — failing "
            "loud here instead of at weekday trade time (config#3053)."
        )


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} ("
        "ticker TEXT NOT NULL, "
        "prediction_date TEXT NOT NULL, "
        "predicted_alpha REAL, "
        "n_research_features_missing INTEGER, "
        "PRIMARY KEY (ticker, prediction_date)"
        ")"
    )
    conn.commit()


def _existing_keys(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    try:
        rows = conn.execute(
            f"SELECT ticker, prediction_date FROM {TABLE_NAME}"
        ).fetchall()
    except sqlite3.OperationalError:
        return set()
    return {(r[0], r[1]) for r in rows}


def _pending_universe(
    conn: sqlite3.Connection,
    *,
    bucket: str,
    region: str | None = None,
    s3_client=None,
    lookback_days: int = 120,
) -> pd.DataFrame:
    """The scanner-passing (ticker, eval_date) universe still needing a backfill row.

    Reads the S3 ``candidates/{run_date}/candidates.json`` artifact's
    ``scanner_eval_log`` directly (config#3053) — NOT the ``scanner_evaluations``
    sqlite table, whose only writer was removed from the weekly SF by the
    2026-07-14 config#1580 restructure (see ``_CANDIDATES_PREFIX`` comment
    above for the full root-cause). ``scanner_eval_log`` entries carry the
    exact same field shape the old table did (``ticker``, ``quant_filter_pass``,
    ...); the artifact's own ``run_date`` becomes each row's ``eval_date``.

    Raises :class:`NoCandidatesArtifactError` if no ``candidates.json`` exists
    anywhere in the lookback window — the caller decides how to surface that
    (this module has no fallback universe to fall back to). A week whose
    artifact exists but carries an EMPTY ``scanner_eval_log`` is logged and
    skipped rather than raised (a Scanner-side contract violation, not this
    producer's job to paper over silently, but also not proof no OTHER week
    in the window has real data).
    """
    s3 = s3_client or boto3.client("s3", **({"region_name": region} if region else {}))
    dates = _list_recent_candidate_dates(
        bucket, region=region, s3_client=s3, lookback_days=lookback_days,
    )
    if not dates:
        raise NoCandidatesArtifactError(
            f"no candidates.json artifacts found under s3://{bucket}/"
            f"{_CANDIDATES_PREFIX}/ within the last {lookback_days}d — cannot "
            "determine the scanner-passing universe (scanner_evaluations table "
            "retired by config#1580; config#3053 root-cause)"
        )

    rows: list[dict] = []
    for d in dates:
        key = f"{_CANDIDATES_PREFIX}/{d}/candidates.json"
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            artifact = json.loads(obj["Body"].read())
        except (ClientError, BotoCoreError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning(
                "failed to read/parse s3://%s/%s (skipping this week's eval "
                "log): %s", bucket, key, exc,
            )
            continue
        run_date = artifact.get("run_date") or d
        eval_log = artifact.get("scanner_eval_log") or []
        if not eval_log:
            logger.warning(
                "s3://%s/%s carries an EMPTY scanner_eval_log for run_date=%s "
                "— a scan week producing zero eval rows is a Scanner-side "
                "contract violation, not silently papered over here "
                "(config#3053)", bucket, key, run_date,
            )
            continue
        for rec in eval_log:
            ticker = rec.get("ticker")
            if not ticker or not rec.get("quant_filter_pass"):
                continue
            rows.append({"ticker": ticker, "eval_date": run_date})

    df = pd.DataFrame(rows, columns=["ticker", "eval_date"]).drop_duplicates()
    existing = _existing_keys(conn)
    if existing and not df.empty:
        mask = ~df.apply(lambda r: (r["ticker"], r["eval_date"]) in existing, axis=1)
        df = df[mask]
    return df.sort_values(["eval_date", "ticker"]).reset_index(drop=True)


def _download_weights_to_temp(
    bucket: str,
    filename: str,
    *,
    region: str | None = None,
    s3_client=None,
    sidecar: bool = True,
) -> Path:
    """Download ``predictor/weights/meta/{filename}`` from S3 to a temp dir and
    return the local path. ``sidecar=True`` also fetches ``{filename}.meta.json``
    beside it, best-effort (``MetaModel.load`` reads the sidecar when present;
    v3 pickles embed feature_names so a missing sidecar is harmless).

    Raises RuntimeError on a failed primary download — the deployed champion
    artifact being unreachable is a PredictorTraining/S3 problem the caller
    must see, never a silent zero-output backfill.
    """
    s3 = s3_client or boto3.client("s3", **({"region_name": region} if region else {}))
    tmp_dir = Path(tempfile.mkdtemp(prefix="research_free_weights_"))
    local = tmp_dir / filename
    key = _WEIGHTS_PREFIX + filename
    try:
        s3.download_file(bucket, key, str(local))
        logger.info("Downloaded deployed weight artifact s3://%s/%s", bucket, key)
    except (ClientError, BotoCoreError, OSError) as exc:
        raise RuntimeError(
            f"failed to download deployed weight artifact s3://{bucket}/{key}: {exc}"
        ) from exc
    if sidecar:
        try:
            s3.download_file(bucket, key + ".meta.json", str(local) + ".meta.json")
        except (ClientError, BotoCoreError, OSError) as exc:
            logger.warning(
                "sidecar %s.meta.json not fetched (non-fatal, embedded schema wins): %s",
                key, exc,
            )
    return local


def _load_meta_model(
    predictor_path: str,
    bucket: str,
    *,
    region: str | None = None,
    s3_client=None,
):
    """Load the deployed MetaModel: code via the standard sibling-checkout
    ``sys.path.insert`` idiom (``synthetic/predictor_backtest.py``,
    ``backtest.py::run_predictor_backtest``), weights via S3 download — the
    checkout carries no synced ``weights/`` dir on the spot box.
    """
    if predictor_path not in sys.path:
        sys.path.insert(0, predictor_path)
    from model.meta_model import MetaModel  # noqa: E402

    pkl_path = _download_weights_to_temp(
        bucket, "meta_model.pkl", region=region, s3_client=s3_client,
    )
    mm = MetaModel.load(str(pkl_path))
    if not mm.is_fitted:
        raise RuntimeError(f"MetaModel loaded from s3 ({pkl_path}) is not fitted")
    return mm


def _load_volatility_scorer(
    predictor_path: str,
    bucket: str,
    *,
    region: str | None = None,
    s3_client=None,
):
    """Best-effort load of the deployed volatility GBM (expected_move input).

    None on any failure (missing S3 artifact, lightgbm not installed) — callers
    treat that as "expected_move unavailable -> 0.0", the same neutral default
    ``inference/stages/run_inference.py`` uses when ``vol_scorer is None``.
    """
    try:
        if predictor_path not in sys.path:
            sys.path.insert(0, predictor_path)
        from model.gbm_scorer import GBMScorer

        path = _download_weights_to_temp(
            bucket, "volatility_model.txt",
            region=region, s3_client=s3_client, sidecar=False,
        )
        return GBMScorer.load(str(path))
    except Exception as exc:  # noqa: BLE001 - graceful degrade, never fatal
        logger.warning("volatility scorer load failed (expected_move -> 0.0): %s", exc)
        return None


def materialize_from_s3(
    conn: sqlite3.Connection,
    bucket: str = "alpha-engine-research",
    *,
    region: str | None = None,
    s3_client=None,
) -> int:
    """Materialize the canonical S3 backfill artifact into ``conn``'s
    ``predictor_outcomes_research_free`` table. Returns the number of rows
    materialized; 0 when the artifact doesn't exist yet (first ever run).

    Called on BOTH sides of the producer/consumer seam (see ARTIFACT_KEY
    comment): ``run_backfill`` seeds its idempotency set from it, and the
    e2e-lift consumer (``analysis/end_to_end.py``, running on the separate
    Evaluator box against its own freshly pulled research.db copy) hydrates
    the table before ``_scanner_then_predictor_topN`` reads it.

    A missing artifact is a clean 0 (nothing produced yet — the consumer's
    honest ``skipped``); any OTHER failure (download, parse, insert) raises —
    a corrupt/unreadable artifact must surface, not silently demote the
    counterfactual back to ``skipped``.
    """
    _ensure_table(conn)
    s3 = s3_client or boto3.client("s3", **({"region_name": region} if region else {}))
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp.close()
    try:
        s3.download_file(bucket, ARTIFACT_KEY, tmp.name)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            logger.info(
                "no backfill artifact at s3://%s/%s yet — 0 rows materialized",
                bucket, ARTIFACT_KEY,
            )
            return 0
        raise RuntimeError(
            f"failed to download backfill artifact s3://{bucket}/{ARTIFACT_KEY}: {exc}"
        ) from exc
    df = pd.read_parquet(tmp.name)
    if df.empty:
        return 0
    conn.executemany(
        f"INSERT OR REPLACE INTO {TABLE_NAME} "
        "(ticker, prediction_date, predicted_alpha, n_research_features_missing) "
        "VALUES (?,?,?,?)",
        list(
            df[
                ["ticker", "prediction_date", "predicted_alpha", "n_research_features_missing"]
            ].itertuples(index=False, name=None)
        ),
    )
    conn.commit()
    logger.info(
        "materialized %d rows from s3://%s/%s into %s", len(df), bucket, ARTIFACT_KEY, TABLE_NAME,
    )
    return int(len(df))


def _export_artifact(
    conn: sqlite3.Connection,
    bucket: str,
    *,
    region: str | None = None,
    s3_client=None,
) -> str:
    """Export the FULL local table to the canonical S3 parquet. Raises on any
    failure: a computed-but-unpersisted backfill is indistinguishable from a
    never-run one on the consumer box — exactly the silent failure mode this
    artifact exists to kill.
    """
    df = pd.read_sql_query(
        f"SELECT ticker, prediction_date, predicted_alpha, n_research_features_missing "
        f"FROM {TABLE_NAME}",
        conn,
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp.close()
    df.to_parquet(tmp.name, index=False)
    s3 = s3_client or boto3.client("s3", **({"region_name": region} if region else {}))
    try:
        s3.upload_file(tmp.name, bucket, ARTIFACT_KEY)
    except (ClientError, BotoCoreError, OSError) as exc:
        raise RuntimeError(
            f"failed to upload backfill artifact s3://{bucket}/{ARTIFACT_KEY}: {exc}"
        ) from exc
    logger.info(
        "exported %d rows to s3://%s/%s", len(df), bucket, ARTIFACT_KEY,
    )
    return ARTIFACT_KEY


def _compute_momentum_scores(rows: dict[str, pd.Series]) -> dict[str, float]:
    """model/momentum_scorer.py::predict_dict per ticker — deterministic, no
    model artifact required."""
    from model.momentum_scorer import predict_dict as _momentum_predict_dict

    return {t: float(_momentum_predict_dict(row.to_dict())) for t, row in rows.items()}


def _compute_residual_momentum_scores(
    tickers: list[str],
    close_history: dict[str, pd.Series],
    spy_close: pd.Series | None,
    as_of: pd.Timestamp,
) -> dict[str, float]:
    """model/residual_momentum_scorer.py::predict_dict per ticker, fed by
    ``data/residual_momentum_features.py::compute_residual_momentum_features``
    run on FULL close-price history — NOT the ArcticDB universe-library row's
    precomputed columns.

    Confirmed by direct comparison during this producer's build (2026-07-10
    smoke test against production data): the ArcticDB universe row carries
    differently-named/differently-defined legacy columns
    (``residual_momentum_ratio``, ``mom_12_1_pct``, ``sector_mom_pct``) that do
    NOT match this scorer's expected input names (``resid_mom_vol_scaled`` /
    ``mom_12_1`` / ``mom_1m`` / ``mom_change`` / ``sector_mom``) — feeding them
    directly makes ``predict_dict`` silently degrade every ticker to a neutral
    0.0 (verified: identical ``predicted_alpha`` across an entire day's
    scanner-passing cohort in the first smoke-test pass, traced to this exact
    mismatch). This function closes that gap by computing the CORRECTLY named
    features from scratch via the predictor's own feature-construction module.

    Benchmark: SPY close (``spy_close``) for every ticker — NOT the
    ticker's sector ETF. A per-ticker sector-ETF benchmark is what
    ``training/meta_trainer.py`` uses live, but resolving a
    ticker->sector-ETF map wasn't cleanly reachable from this backfill's data
    surface (the S3 sector-map object found during this build,
    ``market_data/sectors/latest.json``, is a nested by-sector schema, not the
    flat ``{ticker: sector_etf}`` map ``load_sector_map`` expects; the
    predictor repo's own ``data/cache/sector_map.json`` is a runtime-synced
    artifact not present in a fresh checkout). SPY-fallback is an explicit,
    documented simplification — ``compute_residual_momentum_features``
    natively supports it (``benchmark_close=None`` -> ``spy_close``, the same
    fallback ``label_generator`` uses when a ticker has no sector ETF) — so
    this is a real, correctly-computed residual-momentum signal, just
    market-relative rather than sector-relative. A future iteration wiring
    the true sector map would tighten this, not fix a bug.
    """
    from data.residual_momentum_features import compute_residual_momentum_features
    from model.residual_momentum_scorer import predict_dict as _resid_predict_dict

    out: dict[str, float] = {}
    for t in tickers:
        close = close_history.get(t)
        if close is None or close.empty:
            out[t] = 0.0
            continue
        close = close[close.index <= as_of]
        if close.empty:
            out[t] = 0.0
            continue
        try:
            feats_df = compute_residual_momentum_features(close, None, spy_close)
            latest = feats_df.iloc[-1].to_dict()
            out[t] = float(_resid_predict_dict(latest))
        except Exception as exc:  # noqa: BLE001 - one ticker must not abort the run
            logger.warning("residual_momentum_score computation failed for %s: %s", t, exc)
            out[t] = 0.0
    return out


def _compute_expected_move(
    rows: dict[str, pd.Series], vol_scorer, feature_cols: tuple[str, ...] = _BASELINE_VOLATILITY_FEATURES,
) -> dict[str, float]:
    """Cross-sectional rank-normalized volatility-GBM predict, ONE calendar date.

    Mirrors ``run_inference.py``'s ``cross_sectional_rank_normalize`` +
    ``vol_scorer.predict`` batch. ``rows`` must be the FULL per-ticker feature
    rows for tickers trading on this date (not just the scanner-passing
    subset) — the whole point of a cross-sectional rank is that percentile
    membership is computed against the market that day, matching how the
    scorer was trained. Returns 0.0 for every ticker if ``vol_scorer`` is None.
    """
    if vol_scorer is None or not rows:
        return {t: 0.0 for t in rows}
    from data.dataset import cross_sectional_rank_normalize as _rank_norm

    cols = list(getattr(vol_scorer, "_feature_names", None) or feature_cols)
    tickers = list(rows.keys())
    X_raw = np.stack([
        rows[t].reindex(cols).astype(np.float64).fillna(0.0).to_numpy()
        for t in tickers
    ]).astype(np.float32)
    same_date = ["_single_date_"] * len(tickers)  # one cross-section -> single group
    X_ranked = _rank_norm(X_raw, same_date).astype(np.float32)
    preds = vol_scorer.predict(X_ranked)
    return {t: float(p) for t, p in zip(tickers, preds)}


def _compute_macro_row(spy_s, vix_s, vix3m_s, tnx_s, irx_s, close_prices: dict) -> dict[str, float]:
    """The 6 macro META_FEATURES + ``regime_intensity_z``, via
    ``model/regime_predictor.py::RegimePredictor.build_features`` — the SAME
    utility ``run_inference.py`` uses as a pure feature-engineering helper
    (the Tier-0 classifier itself was retired 2026-04-16; only
    ``build_features`` is used here). Returns an all-zero dict (neutral
    macro row) on any failure — matches ``run_inference.py``'s zero-fill
    fallback posture so a single date's macro-build failure degrades one
    row rather than aborting the whole backfill.
    """
    from model.meta_model import MACRO_FEATURE_META_MAP, REGIME_DERIVED_FEATURE_META_MAP
    from model.regime_predictor import RegimePredictor

    macro_row = {name: 0.0 for name in MACRO_FEATURE_META_MAP.values()}
    for name in REGIME_DERIVED_FEATURE_META_MAP.values():
        macro_row[name] = 0.0
    if spy_s is None or len(spy_s) < 20:
        return macro_row
    try:
        regime_df = RegimePredictor().build_features(
            spy_s, vix_s, vix3m_s, tnx_s, irx_s, close_prices,
        )
        if regime_df.empty:
            return macro_row
        latest = regime_df.iloc[-1]
        for src_name, meta_name in MACRO_FEATURE_META_MAP.items():
            macro_row[meta_name] = float(latest.get(src_name, 0.0))
        for src_name, meta_name in REGIME_DERIVED_FEATURE_META_MAP.items():
            macro_row[meta_name] = float(latest.get(src_name, 0.0))
    except Exception as exc:  # noqa: BLE001 - one date's macro build must not abort the run
        logger.warning("Macro feature build failed for one date (zero-fill fallback): %s", exc)
    return macro_row


def _assemble_research_free_features(
    ticker: str,
    feat_names: list[str],
    *,
    momentum_scores: dict[str, float],
    resid_scores: dict[str, float],
    expected_moves: dict[str, float],
    macro_row: dict[str, float],
    log_context: str = "",
) -> dict[str, float]:
    """Build the research-free feature dict for one ticker, given the loaded
    model's own ``feat_names`` schema and this date's already-computed
    per-ticker/market-wide component scores.

    Pure (no IO) — the per-(ticker, date) heart of the research-free
    contract, factored out of ``run_backfill`` so it is directly
    unit-testable without ArcticDB/S3/a real MetaModel artifact. Any
    feature in ``RESEARCH_META_FEATURES`` is unconditionally zeroed
    regardless of whether a value happens to be available (research-free
    by construction, not by absence); any OTHER feature name the loaded
    model expects but this function doesn't know how to compute degrades
    to 0.0 with a logged warning — the same graceful-degrade contract
    ``MetaModel.predict_single`` itself uses for a missing dict key.
    """
    feats: dict[str, float] = {}
    for f in feat_names:
        if f in RESEARCH_META_FEATURES:
            feats[f] = 0.0  # research-free by construction
        elif f == "momentum_score":
            feats[f] = momentum_scores.get(ticker, 0.0)
        elif f == "residual_momentum_score":
            feats[f] = resid_scores.get(ticker, 0.0)
        elif f == "expected_move":
            feats[f] = expected_moves.get(ticker, 0.0)
        elif f in macro_row:
            feats[f] = macro_row[f]
        else:
            logger.warning(
                "%s%s: no feature computer registered for '%s' -> 0.0",
                (log_context + ": ") if log_context else "", ticker, f,
            )
            feats[f] = 0.0
    return feats


def run_backfill(
    conn: sqlite3.Connection,
    *,
    predictor_path: str,
    bucket: str = "alpha-engine-research",
    region: str | None = None,
    max_dates: int | None = None,
) -> dict:
    """Compute + persist research-free ``predicted_alpha`` for the pending
    scanner-passing (ticker, eval_date) universe. Idempotent — rows already
    in ``predictor_outcomes_research_free`` are excluded from ``_pending_universe``
    up front, so a re-run only computes the delta (already-cached keys never
    reach the per-ticker compute loop below).

    Returns a summary dict (``status``, ``n_written``, ``n_dates``,
    ``n_errors``, ``feature_names``, ``n_research_features_missing``). Never
    raises for a per-ticker/per-date
    failure (those are counted in ``n_errors`` and logged); raises only on a
    genuine precondition failure (missing scanner_evaluations table/column,
    unloadable MetaModel, unreachable ArcticDB) — matching
    ``_load_precomputed_features_from_arcticdb``'s "an infra problem is the
    upstream team's job to fix, not ours to mask" posture, since a silently
    empty/wrong backfill here would poison the config#1405 counterfactual.
    """
    from nousergon_lib.arcticdb import load_universe_ohlcv, load_macro_series

    _ensure_table(conn)
    # Idempotency seed: the local research.db is a fresh throwaway pull (see
    # ARTIFACT_KEY comment), so previously computed rows live ONLY in the S3
    # artifact — hydrate them first or every Saturday recomputes the full
    # history from scratch.
    n_seeded = materialize_from_s3(conn, bucket=bucket, region=region)
    try:
        pending = _pending_universe(conn, bucket=bucket, region=region)
    except NoCandidatesArtifactError as exc:
        return {"status": "skipped", "reason": str(exc)}

    if pending.empty:
        return {
            "status": "ok", "n_written": 0, "n_dates": 0, "n_errors": 0,
            "n_seeded_from_artifact": n_seeded,
        }

    mm = _load_meta_model(predictor_path, bucket, region=region)
    vol_scorer = _load_volatility_scorer(predictor_path, bucket, region=region)
    feat_names = list(mm._feature_names)  # the LOADED model's own schema — source of truth
    research_feats_present = [f for f in feat_names if f in RESEARCH_META_FEATURES]
    n_research_missing = len(research_feats_present)
    if n_research_missing == 0:
        logger.warning(
            "Loaded MetaModel's feature_names contain none of the 4 known "
            "RESEARCH_META_FEATURES (%s) — n_research_features_missing will be "
            "recorded as 0. This is either an unusual model variant or a "
            "feature-name drift this producer doesn't recognize yet.",
            sorted(RESEARCH_META_FEATURES),
        )

    dates = list(pending["eval_date"].unique())
    if max_dates is not None:
        dates = dates[:max_dates]
    pending = pending[pending["eval_date"].isin(dates)]

    # Load the FULL per-symbol history ONCE (parallel batch via the shared
    # nousergon_lib reader), spanning the earliest to latest backfill date
    # with generous lookback for the 12-1-month / 252d rolling windows the
    # deterministic scorers and regime builder need. Per-date rows are then
    # sliced in-memory below — this avoids O(dates x symbols) sequential
    # ArcticDB reads (a single symbol's `.read()` call for the full 10y
    # history costs about the same as reading a 1-day slice; re-issuing it
    # per date is pure waste at backfill scale).
    end_ts = pd.Timestamp(max(dates))
    start_ts = pd.Timestamp(min(dates))
    lookback_days = int((end_ts - start_ts).days) + 400  # +400d for rolling-window warmup
    logger.info(
        "Loading ArcticDB universe history: %d dates (%s..%s), lookback=%dd",
        len(dates), start_ts.date(), end_ts.date(), lookback_days,
    )
    try:
        full_history = load_universe_ohlcv(
            bucket, lookback_days=lookback_days, end=end_ts, region=region,
        )
    except Exception as exc:
        raise RuntimeError(f"ArcticDB universe history load failed: {exc}") from exc
    if not full_history:
        raise RuntimeError("ArcticDB universe library returned zero symbols")

    macro_history: dict[str, pd.DataFrame] = {}
    if any(f.startswith("macro_") or f == "regime_intensity_z" for f in feat_names):
        try:
            # ArcticDB's `macro` library uses plain (non-Yahoo-prefixed) symbol
            # names — confirmed against the live production library (2026-07-10
            # smoke test): {'SPY', 'VIX', 'VIX3M', 'TNX', 'IRX', 'GLD', 'USO',
            # 'XL*', 'features'}. NOT '^VIX'/'^TNX'/'^IRX' (the Yahoo-style keys
            # inference/stages/run_inference.py uses for its OWN in-memory
            # ctx.macro dict, which is populated from a different upstream
            # loader — that convention does not apply to ArcticDB symbol names).
            macro_history = load_macro_series(
                bucket,
                ["SPY", "VIX", "VIX3M", "TNX", "IRX"],
                lookback_days=lookback_days, end=end_ts, region=region,
            )
        except Exception as exc:  # noqa: BLE001 - macro is best-effort (zero-fill fallback)
            logger.warning("macro library load failed (macro features -> 0.0): %s", exc)

    # Precomputed once (not per-date): the full-history Close series per
    # symbol, and the SPY series used as the residual-momentum benchmark
    # (see _compute_residual_momentum_scores docstring for why SPY rather
    # than a per-ticker sector ETF).
    close_history = {t: df["Close"] for t, df in full_history.items() if "Close" in df.columns}
    spy_close_full = close_history.get("SPY")

    n_written = 0
    n_errors = 0
    to_insert: list[tuple] = []

    for d in dates:
        d_ts = pd.Timestamp(d)
        tickers_today = pending.loc[pending["eval_date"] == d, "ticker"].tolist()

        # Full-universe as-of-date rows for this date's cross-sectional
        # rank-norm — the scanner-passing subset alone would rank everyone
        # at the 50th percentile and defeat the point of the normalization.
        rows_today: dict[str, pd.Series] = {}
        for sym, df in full_history.items():
            asof = df[df.index <= d_ts]
            if asof.empty:
                continue
            rows_today[sym] = asof.iloc[-1]

        target_rows = {t: rows_today[t] for t in tickers_today if t in rows_today}
        missing_today = set(tickers_today) - set(target_rows)
        if missing_today:
            logger.warning(
                "%s: %d/%d scanner-passing tickers absent from ArcticDB as-of this "
                "date — skipped (n_errors)", d, len(missing_today), len(tickers_today),
            )
            n_errors += len(missing_today)

        momentum_scores = _compute_momentum_scores(target_rows) if "momentum_score" in feat_names else {}
        resid_scores = (
            _compute_residual_momentum_scores(
                list(target_rows.keys()), close_history, spy_close_full, d_ts,
            )
            if "residual_momentum_score" in feat_names else {}
        )
        expected_moves = (
            _compute_expected_move(rows_today, vol_scorer) if "expected_move" in feat_names else {}
        )

        macro_row = {}
        if any(f.startswith("macro_") or f == "regime_intensity_z" for f in feat_names):
            def _asof_close(sym: str):
                df = macro_history.get(sym)
                if df is None or df.empty:
                    return None
                sub = df[df.index <= d_ts]
                return sub["Close"] if not sub.empty and "Close" in sub.columns else None

            spy_s = _asof_close("SPY")
            vix_s = _asof_close("VIX")
            vix3m_s = _asof_close("VIX3M")
            tnx_s = _asof_close("TNX")
            irx_s = _asof_close("IRX")
            close_prices = {t: s[s.index <= d_ts] for t, s in close_history.items()}
            macro_row = _compute_macro_row(spy_s, vix_s, vix3m_s, tnx_s, irx_s, close_prices)

        for t in tickers_today:
            if t not in target_rows:
                continue
            try:
                feats = _assemble_research_free_features(
                    t, feat_names,
                    momentum_scores=momentum_scores,
                    resid_scores=resid_scores,
                    expected_moves=expected_moves,
                    macro_row=macro_row,
                    log_context=str(d),
                )
                alpha = float(mm.predict_single(feats))
            except Exception as exc:  # noqa: BLE001 - one ticker's failure must not abort the run
                logger.warning("%s/%s: predict_single failed: %s", d, t, exc)
                n_errors += 1
                continue
            to_insert.append((t, d, alpha, n_research_missing))
            n_written += 1

    artifact_key = None
    if to_insert:
        conn.executemany(
            f"INSERT OR REPLACE INTO {TABLE_NAME} "
            "(ticker, prediction_date, predicted_alpha, n_research_features_missing) "
            "VALUES (?,?,?,?)",
            to_insert,
        )
        conn.commit()
        # Persist to the canonical S3 artifact — the local research.db copy is
        # discarded with the box; raises on failure (see _export_artifact).
        artifact_key = _export_artifact(conn, bucket, region=region)

    return {
        "status": "ok",
        "n_written": n_written,
        "n_dates": len(dates),
        "n_errors": n_errors,
        "n_seeded_from_artifact": n_seeded,
        "artifact_key": artifact_key,
        "feature_names": feat_names,
        "n_research_features_missing": n_research_missing,
    }
