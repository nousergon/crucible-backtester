"""
tech_weight_ablation.py — Per-sector recommendation of technical scorer
weight configs by rank-correlation ablation.

PR-C of the 2026-05-09 sector-team diagnostic arc:
  PR-A: analysis/quant_rank_quality.py — recurring detection
  PR-B: alpha-engine-research v15 migration — sub-score persistence
  PR-C (this): given persisted sub-scores, find the weight mix per
              sector that minimizes (most-negative) corr(rank, 5d_ret).

The technical scorer is currently 75% trend / 25% mean-reversion across
all sectors uniformly (rsi=0.25, macd=0.20, ma50=0.15, ma200=0.15,
momentum=0.25). The post-mortem showed this is anti-skill in
healthcare/industrials/tech — top quant ranks systematically pick
losers. This module sweeps a grid of alternate weight configs and
recommends per-sector overrides.

Two-stage activation (ROADMAP L2553 auto-apply cutover):

  1. ``use_tech_ablation_target=True`` (default false) — every weekly
     run that produces an ``ok`` recommendation writes a shadow payload
     to ``config/scoring_weights_per_sector_shadow_history/{run_id}.json``
     (+ ``latest.json`` sidecar). Live config is unchanged. Pure
     observability — the operator can compare shadow trajectories
     against the live ``scoring.yaml`` baselines week-over-week.

  2. ``enforce_tech_ablation=True`` (default false) — in addition to
     the shadow archive, the apply path writes the live
     ``config/scoring_weights_per_sector.json`` key **only if** the
     reproduction gate passes: the same per-sector recommendation
     dict must reproduce across the last ``_MIN_CONSECUTIVE_WEEKS``
     shadow archives. This prevents a single noisy week from flipping
     sector weights live.

Reads ``team_candidates`` joined to ``universe_returns`` for rows where
the 5 sub-score columns are non-NULL (populated only after PR-B's v15
migration is in production). Produces ``insufficient_data`` until at
least ``_MIN_WEEKS`` weeks of sub-score data accumulate.

Returns the standard backtester-evaluator status dict so the existing
``CompletenessTracker.run_module`` pattern handles it without bespoke
wiring.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ── S3 contract (auto-apply, ROADMAP L2553) ─────────────────────────────────

S3_LIVE_KEY = "config/scoring_weights_per_sector.json"
S3_SHADOW_PREFIX = "config/scoring_weights_per_sector_shadow_history"

# Reproduction gate: live write only fires when the same per-sector
# recommendation dict reproduces across this many consecutive shadow
# archives. Mirrors the ROADMAP L2553 "4+ consecutive Saturdays"
# acceptance.
_MIN_CONSECUTIVE_WEEKS = 4

# Module-level config ref — set by init_config() from evaluate.py
_cfg: dict = {}


def init_config(config: dict) -> None:
    """Load tech_weight_ablation section from backtester config.

    Recognized keys (all default false):
      - ``use_tech_ablation_target``: enable shadow-archive writes
      - ``enforce_tech_ablation``: enable live writes (also requires
        reproduction gate to pass)
    """
    global _cfg
    _cfg = config.get("tech_weight_ablation", {}) or {}


# ── Canonical sectors (mirrors quant_rank_quality + decision_capture) ───────

CANONICAL_SECTORS = (
    "consumer", "defensives", "financials",
    "healthcare", "industrials", "technology",
)

# Min calendar coverage. Mirrors other optimizers (pipeline_optimizer,
# weight_optimizer); 8 weeks is the system-wide rolling window.
_MIN_WEEKS = 8

# Min sub-score-populated rows per team. Below this, the per-sector
# ablation is too noisy to trust. Conservative: ~5 picks/wk × 8 weeks.
_MIN_ROWS_PER_TEAM = 30

# How much better the best ablation config must be vs the current
# production config to surface a recommendation. Mirrors the existing
# executor_optimizer 5%-improvement gate, applied here on the rank
# correlation axis: best_corr must be at least this much MORE NEGATIVE
# than current_corr to recommend a switch.
_MIN_IMPROVEMENT = 0.10


# ── Weight grid ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WeightConfig:
    """Named weight tuple. Sums to 1.0 (validated at construction)."""
    name: str
    rsi: float
    macd: float
    ma50: float
    ma200: float
    momentum: float

    def __post_init__(self):
        s = self.rsi + self.macd + self.ma50 + self.ma200 + self.momentum
        if abs(s - 1.0) > 1e-6:
            raise ValueError(
                f"WeightConfig '{self.name}' weights sum to {s:.4f}, not 1.0"
            )

    def synthetic_score(
        self,
        rsi: float, macd: float, ma50: float, ma200: float, momentum: float,
    ) -> float:
        return (
            self.rsi * rsi + self.macd * macd
            + self.ma50 * ma50 + self.ma200 * ma200
            + self.momentum * momentum
        )


# Named ablation grid. The "current_default" config mirrors the live
# scoring.yaml composite_weights as of 2026-05-10 (75/25 trend/MR).
# Other configs span the space along the trend-vs-mean-reversion axis.
DEFAULT_GRID: tuple[WeightConfig, ...] = (
    WeightConfig("current_default", rsi=0.25, macd=0.20, ma50=0.15, ma200=0.15, momentum=0.25),
    WeightConfig("balanced_50_50",  rsi=0.50, macd=0.125, ma50=0.125, ma200=0.125, momentum=0.125),
    WeightConfig("mean_rev_heavy",  rsi=0.60, macd=0.10, ma50=0.10, ma200=0.10, momentum=0.10),
    WeightConfig("rsi_only",        rsi=1.00, macd=0.0, ma50=0.0, ma200=0.0, momentum=0.0),
    WeightConfig("momentum_only",   rsi=0.0, macd=0.0, ma50=0.0, ma200=0.0, momentum=1.00),
    WeightConfig("trend_only",      rsi=0.0, macd=0.25, ma50=0.25, ma200=0.25, momentum=0.25),
    WeightConfig("ma_heavy",        rsi=0.20, macd=0.10, ma50=0.30, ma200=0.30, momentum=0.10),
)


# ── Live config reader ──────────────────────────────────────────────────────
#
# Per-sector composite_weights overrides live in alpha-engine-config/research/
# scoring.yaml under technical.composite_weights_per_sector (added 2026-05-11
# per ROADMAP P1 "Per-sector overrides in composite_weights schema"). The
# ablation module reads them so its per-team output surfaces what is actually
# deployed live for each team — not just the hardcoded global default. Gate
# semantics still compare against the static `current_default` config in
# DEFAULT_GRID; future ROADMAP P1 "Tech weight ablation auto-apply" (L2202)
# will flip the gate to use live baselines.


def _load_live_composite_weights_per_sector(
    search_paths: list[Path] | None = None,
) -> dict[str, dict[str, float]]:
    """Read technical.composite_weights_per_sector from alpha-engine-config.

    Returns an empty dict if the file or block is missing / malformed.
    Mirrors the lookup pattern in pipeline_common._load_active_horizon_days
    so spot runs (config repo cloned beside this repo) and local dev (config
    repo in $HOME) both resolve.
    """
    if search_paths is None:
        search_paths = [
            Path.home() / "alpha-engine-config" / "research" / "scoring.yaml",
            Path(__file__).resolve().parent.parent.parent / "alpha-engine-config" / "research" / "scoring.yaml",
        ]
    for p in search_paths:
        if not p.exists():
            continue
        try:
            with open(p) as f:
                cfg = yaml.safe_load(f) or {}
            block = cfg.get("technical", {}).get("composite_weights_per_sector") or {}
            if isinstance(block, dict):
                return block
        except (OSError, yaml.YAMLError) as exc:
            logger.warning(
                "tech_weight_ablation: could not read scoring.yaml from %s: %s", p, exc,
            )
            continue
    return {}


# ── Pearson + per-team eval ─────────────────────────────────────────────────


def _safe_pearson(x: list[float], y: list[float]) -> float | None:
    """Pearson correlation; None on n<3 or zero variance."""
    if len(x) < 3 or len(y) < 3:
        return None
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den_x = sum((xi - mx) ** 2 for xi in x)
    den_y = sum((yi - my) ** 2 for yi in y)
    if den_x == 0 or den_y == 0:
        return None
    return num / ((den_x * den_y) ** 0.5)


def _evaluate_team_under_config(
    team_rows: list[tuple],
    cfg: WeightConfig,
) -> float | None:
    """Re-rank within each (eval_date) by the synthetic score, then
    compute corr(synthetic_rank, return_5d) across the team's rows.

    Args:
        team_rows: list of (eval_date, rsi_sub, macd_sub, ma50_sub,
                   ma200_sub, momentum_sub, return_5d) tuples for one
                   team, all rows with sub-scores populated and
                   return_5d non-NULL.
        cfg: weight config to evaluate.

    Returns:
        Pearson correlation of the synthetic rank against return_5d,
        or None if any per-date pool has fewer than 2 rows (rank is
        undefined for n=1).
    """
    by_date: dict[str, list[tuple]] = {}
    for r in team_rows:
        by_date.setdefault(r[0], []).append(r)

    all_synth_ranks: list[float] = []
    all_rets: list[float] = []
    for date, date_rows in by_date.items():
        if len(date_rows) < 2:
            continue
        scored = []
        for r in date_rows:
            try:
                s = cfg.synthetic_score(r[1], r[2], r[3], r[4], r[5])
            except Exception:
                continue
            scored.append((s, r[6]))
        if len(scored) < 2:
            continue
        # Highest synthetic score → rank 1
        scored.sort(key=lambda t: t[0], reverse=True)
        for synth_rank, (_, ret) in enumerate(scored, start=1):
            all_synth_ranks.append(synth_rank)
            all_rets.append(ret)

    return _safe_pearson(all_synth_ranks, all_rets)


# ── Per-team optimization ───────────────────────────────────────────────────


def _team_ablation(
    conn: sqlite3.Connection,
    *,
    team_id: str,
    start_date: str,
    end_date: str,
    grid: tuple[WeightConfig, ...],
) -> dict[str, Any]:
    """Run the weight grid for one team and return per-config corr +
    best-config recommendation."""
    rows = conn.execute(
        """
        SELECT tc.eval_date, tc.rsi_sub_score, tc.macd_sub_score,
               tc.ma50_sub_score, tc.ma200_sub_score, tc.momentum_sub_score,
               ur.return_5d
        FROM team_candidates tc
        INNER JOIN universe_returns ur
          ON tc.ticker = ur.ticker AND tc.eval_date = ur.eval_date
        WHERE tc.team_id = ?
          AND tc.eval_date BETWEEN ? AND ?
          AND ur.return_5d IS NOT NULL
          AND tc.rsi_sub_score IS NOT NULL
          AND tc.macd_sub_score IS NOT NULL
          AND tc.ma50_sub_score IS NOT NULL
          AND tc.ma200_sub_score IS NOT NULL
          AND tc.momentum_sub_score IS NOT NULL
        """,
        (team_id, start_date, end_date),
    ).fetchall()

    if len(rows) < _MIN_ROWS_PER_TEAM:
        return {
            "team_id": team_id,
            "status": "insufficient_data",
            "n_rows": len(rows),
            "min_required": _MIN_ROWS_PER_TEAM,
        }

    per_config: list[dict[str, Any]] = []
    for cfg in grid:
        corr = _evaluate_team_under_config(rows, cfg)
        per_config.append({
            "config": cfg.name,
            "weights": {
                "rsi": cfg.rsi, "macd": cfg.macd,
                "ma50": cfg.ma50, "ma200": cfg.ma200,
                "momentum": cfg.momentum,
            },
            "rank_corr": round(corr, 4) if corr is not None else None,
        })

    # Current-default config baseline
    current = next(
        (c for c in per_config if c["config"] == "current_default"), None,
    )
    current_corr = current["rank_corr"] if current else None

    # Best ablation: most-negative rank correlation. None entries skipped.
    candidates = [c for c in per_config if c["rank_corr"] is not None]
    if not candidates:
        return {
            "team_id": team_id,
            "status": "no_valid_corr",
            "n_rows": len(rows),
            "per_config": per_config,
        }
    best = min(candidates, key=lambda c: c["rank_corr"])

    # Recommendation gate: best must be MORE NEGATIVE than current by
    # at least _MIN_IMPROVEMENT. Otherwise keep the current config —
    # not enough evidence the alternative is meaningfully better.
    recommend_switch = (
        current_corr is not None
        and best["config"] != "current_default"
        and current_corr - best["rank_corr"] >= _MIN_IMPROVEMENT
    )

    return {
        "team_id": team_id,
        "status": "ok",
        "n_rows": len(rows),
        "per_config": per_config,
        "current_corr": current_corr,
        "best_config": best["config"],
        "best_corr": best["rank_corr"],
        "improvement_vs_current": (
            round(current_corr - best["rank_corr"], 4)
            if current_corr is not None else None
        ),
        "recommendation": (
            "switch_to_" + best["config"] if recommend_switch else "keep_current"
        ),
    }


# ── Public entry point ──────────────────────────────────────────────────────


def compute_tech_weight_ablation(
    db_path: str | None = None,
    db_conn: sqlite3.Connection | None = None,
    run_date: str | None = None,
    lookback_weeks: int = _MIN_WEEKS,
    grid: tuple[WeightConfig, ...] = DEFAULT_GRID,
) -> dict[str, Any]:
    """Run weight ablation per sector over a rolling window.

    Args:
        db_path: path to research.db on disk. Either this or db_conn
            must be provided.
        db_conn: already-open SQLite connection (tests + reusing the
            evaluator's already-pulled DB).
        run_date: ISO date. Defaults to today (UTC). Window end.
        lookback_weeks: trailing N weeks. Default mirrors _MIN_WEEKS.
        grid: tuple of WeightConfig to evaluate. Default = DEFAULT_GRID.

    Returns:
        status: "ok" | "insufficient_data" | "no_data" | "error"
        run_date, window_start, window_end
        per_team: list[dict] with status + per_config rank_corr +
            best_config + recommendation per canonical sector
        recommendations: dict[team_id -> config_name] of teams that
            cleared the improvement gate
    """
    if db_conn is None and db_path is None:
        return {"status": "error", "error": "must provide db_path or db_conn"}

    run_date = run_date or datetime.utcnow().strftime("%Y-%m-%d")
    try:
        end_dt = datetime.strptime(run_date, "%Y-%m-%d")
    except ValueError as e:
        return {"status": "error", "error": f"invalid run_date: {e}"}
    start_dt = end_dt - timedelta(weeks=lookback_weeks)
    start_iso = start_dt.strftime("%Y-%m-%d")
    end_iso = end_dt.strftime("%Y-%m-%d")

    own_conn = False
    conn = db_conn
    if conn is None:
        conn = sqlite3.connect(db_path)
        own_conn = True

    try:
        # Schema check: team_candidates must have the v15 sub-score columns.
        try:
            cols = {
                r[1] for r in conn.execute("PRAGMA table_info(team_candidates)")
            }
        except sqlite3.OperationalError:
            return {
                "status": "no_data", "run_date": run_date,
                "reason": "team_candidates table missing",
            }
        required = {
            "rsi_sub_score", "macd_sub_score", "ma50_sub_score",
            "ma200_sub_score", "momentum_sub_score",
        }
        if not required.issubset(cols):
            missing = sorted(required - cols)
            return {
                "status": "no_data", "run_date": run_date,
                "reason": (
                    f"team_candidates schema missing sub-score columns "
                    f"(needs v15 migration); missing: {missing}"
                ),
            }

        # Surface what is currently deployed per team so the operator can see
        # the gap between the recommendation and the live config. Empty dict
        # is the default (sector-agnostic baseline) — gate semantics still
        # compare against DEFAULT_GRID's `current_default` until L2202 cutover.
        live_overrides = _load_live_composite_weights_per_sector()

        per_team = []
        for t in CANONICAL_SECTORS:
            result = _team_ablation(
                conn, team_id=t,
                start_date=start_iso, end_date=end_iso, grid=grid,
            )
            result["live_baseline_weights"] = live_overrides.get(t)
            per_team.append(result)

        # Sanity: any team have data? If all are insufficient_data, the
        # producer-side wire-up hasn't accumulated enough yet.
        n_ok = sum(1 for t in per_team if t.get("status") == "ok")
        if n_ok == 0:
            return {
                "status": "insufficient_data",
                "run_date": run_date,
                "window_start": start_iso,
                "window_end": end_iso,
                "lookback_weeks": lookback_weeks,
                "min_rows_per_team": _MIN_ROWS_PER_TEAM,
                "per_team": per_team,
                "reason": (
                    f"no team has ≥{_MIN_ROWS_PER_TEAM} rows with "
                    f"sub-scores populated in window {start_iso}..{end_iso} "
                    f"— PR-B v15 migration may not have accumulated data yet"
                ),
            }

        recommendations = {
            t["team_id"]: t["best_config"]
            for t in per_team
            if t.get("status") == "ok"
            and t.get("recommendation", "").startswith("switch_to_")
        }

        return {
            "status": "ok",
            "run_date": run_date,
            "window_start": start_iso,
            "window_end": end_iso,
            "lookback_weeks": lookback_weeks,
            "min_rows_per_team": _MIN_ROWS_PER_TEAM,
            "min_improvement": _MIN_IMPROVEMENT,
            "grid_size": len(grid),
            "per_team": per_team,
            "recommendations": recommendations,
            "n_teams_ok": n_ok,
            "n_teams_with_recommendation": len(recommendations),
            # The compute step is recommendation-only; apply() is the
            # cutover gate (two-flag activation + reproduction guard).
            "applied": False,
            "apply_note": (
                "see apply() — gated on use_tech_ablation_target + "
                "enforce_tech_ablation flags"
            ),
        }
    finally:
        if own_conn:
            conn.close()


# ── Auto-apply path (ROADMAP L2553) ─────────────────────────────────────────


def _config_name_to_weights(config_name: str) -> dict[str, float] | None:
    """Map a WeightConfig name (e.g. ``"rsi_only"``) → its weight dict.

    Returns None when the name is not in DEFAULT_GRID — guards against
    a future schema drift where the grid is reshuffled.
    """
    for cfg in DEFAULT_GRID:
        if cfg.name == config_name:
            return {
                "rsi": cfg.rsi, "macd": cfg.macd,
                "ma50": cfg.ma50, "ma200": cfg.ma200,
                "momentum": cfg.momentum,
            }
    return None


def _build_per_sector_payload(result: dict) -> dict[str, dict[str, float]]:
    """Translate ``recommendations: {team_id -> config_name}`` into a
    per-sector weights payload the research-side consumer can read as
    an override layer on top of ``scoring.yaml``.

    Returns ``{team_id: {weight_name: weight_value, ...}, ...}``.
    Teams with no recommendation (kept_current or non-`switch_to_*`)
    are omitted — absent key means \"no override, use scoring.yaml
    default\".
    """
    out: dict[str, dict[str, float]] = {}
    for team_id, config_name in (result.get("recommendations") or {}).items():
        weights = _config_name_to_weights(config_name)
        if weights is not None:
            out[team_id] = weights
    return out


def _read_recent_shadow_archives(
    s3, bucket: str, n: int,
) -> list[dict]:
    """Read up to ``n`` most-recent shadow archives, newest first.

    Returns parsed JSON dicts. Missing/corrupt artifacts are skipped
    with a warning — the reproduction gate then treats them as
    \"reproduction not yet reached\".
    """
    from botocore.exceptions import ClientError

    # List `{prefix}/...json` artifacts. The eval_artifact layout uses
    # YYMMDDHHMM-encoded keys + a `latest.json` sidecar; sort
    # lexicographically and the YYMMDDHHMM ordering doubles as time
    # ordering. Skip the sidecar so we only score real archives.
    try:
        resp = s3.list_objects_v2(
            Bucket=bucket, Prefix=f"{S3_SHADOW_PREFIX}/",
        )
    except ClientError as e:
        logger.warning(
            "[tech_weight_ablation] shadow archive list failed (%s) — "
            "treating as no history available",
            type(e).__name__,
        )
        return []
    keys = sorted(
        (obj["Key"] for obj in (resp.get("Contents") or [])
         if obj["Key"].endswith(".json")
         and not obj["Key"].endswith("/latest.json")),
        reverse=True,
    )[:n]

    out: list[dict] = []
    for key in keys:
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            out.append(json.loads(obj["Body"].read()))
        except Exception as e:
            logger.warning(
                "[tech_weight_ablation] shadow archive %s unreadable (%s) — "
                "skipping",
                key, type(e).__name__,
            )
    return out


def _check_reproduction_gate(
    s3, bucket: str, current_payload: dict[str, dict[str, float]],
    *, min_consecutive: int = _MIN_CONSECUTIVE_WEEKS,
) -> dict:
    """Pass = the same per-sector recommendation reproduces across the
    last ``min_consecutive`` shadow archives.

    Returns ``{"passed": bool, "reason": str, "n_consecutive": int}``.

    A shadow archive matches when its ``per_sector`` dict equals
    ``current_payload`` byte-for-byte (key set + weight values). One
    week that disagrees breaks the streak — the gate explicitly does
    NOT tolerate intermittent shadow drift, matching the L2553
    \"4+ consecutive Saturdays\" framing.
    """
    archives = _read_recent_shadow_archives(s3, bucket, min_consecutive)
    if len(archives) < min_consecutive:
        return {
            "passed": False,
            "reason": (
                f"reproduction gate: only {len(archives)} prior shadow "
                f"archive(s) available; need {min_consecutive}"
            ),
            "n_consecutive": len(archives),
        }
    for i, archive in enumerate(archives):
        prior = archive.get("per_sector") or {}
        if prior != current_payload:
            return {
                "passed": False,
                "reason": (
                    f"reproduction gate broken at archive[-{i + 1}]: "
                    f"per_sector payload differs from this week"
                ),
                "n_consecutive": i,
            }
    return {
        "passed": True,
        "reason": (
            f"per_sector payload reproduced across last "
            f"{min_consecutive} shadow archives"
        ),
        "n_consecutive": min_consecutive,
    }


def apply(result: dict, bucket: str) -> dict:
    """Write the per-sector tech-weight recommendation to S3 under the
    two-stage activation contract documented at module top.

    Two write paths:

    - **Shadow** (``use_tech_ablation_target=True``,
      ``enforce_tech_ablation`` ignored): canonical eval-style archive
      at ``config/scoring_weights_per_sector_shadow_history/{run_id}.json``
      + ``latest.json`` sidecar. Live config untouched. The shadow
      archives are also what the reproduction gate reads to decide
      whether to fire the live write.

    - **Live** (``use_tech_ablation_target=True`` AND
      ``enforce_tech_ablation=True`` AND reproduction gate passes):
      writes ``config/scoring_weights_per_sector.json`` after the
      reproduction gate confirms the same recommendation reproduced
      across the last ``_MIN_CONSECUTIVE_WEEKS`` shadow archives.
      Always also writes the shadow archive — every fire goes into
      history.

    Returns the standard apply-result dict shape with ``applied`` +
    ``reason`` + per-path bookkeeping. Mirrors
    ``executor_optimizer.apply()`` so the evaluator wiring is uniform.
    """
    import boto3
    from alpha_engine_lib.eval_artifacts import (
        eval_artifact_key, eval_latest_key, new_eval_run_id,
    )

    use_shadow = bool(_cfg.get("use_tech_ablation_target", False))
    enforce = bool(_cfg.get("enforce_tech_ablation", False))

    if not use_shadow:
        # Flag off — nothing to do. Recommendation stays advisory.
        return {
            "applied": False,
            "reason": "use_tech_ablation_target=False",
        }
    if result.get("status") != "ok":
        return {
            "applied": False,
            "reason": f"compute status={result.get('status')}",
        }

    per_sector = _build_per_sector_payload(result)
    if not per_sector:
        return {
            "applied": False,
            "reason": "no per-sector recommendation to apply",
        }

    payload = {
        "per_sector": per_sector,
        "updated_at": str(date.today()),
        "n_teams_with_recommendation": len(per_sector),
        "source": "tech_weight_ablation",
        "run_date": result.get("run_date"),
        "min_improvement": result.get("min_improvement"),
    }
    body = json.dumps(payload, indent=2)

    s3 = boto3.client("s3")

    # Always write shadow when the flag is on — that's the whole
    # observability point. Reproduction gate reads these.
    run_id = new_eval_run_id()
    shadow_key = eval_artifact_key(S3_SHADOW_PREFIX, run_id)
    shadow_latest_key = eval_latest_key(S3_SHADOW_PREFIX)
    try:
        s3.put_object(
            Bucket=bucket, Key=shadow_key, Body=body,
            ContentType="application/json",
        )
        s3.put_object(
            Bucket=bucket, Key=shadow_latest_key, Body=body,
            ContentType="application/json",
        )
        logger.info(
            "[tech_weight_ablation] shadow archive written: s3://%s/%s",
            bucket, shadow_key,
        )
    except Exception as e:
        logger.error(
            "[tech_weight_ablation] shadow archive write failed: %s", e,
        )
        return {
            "applied": False,
            "reason": f"shadow S3 write failed: {e}",
        }

    if not enforce:
        return {
            "applied": False,
            "reason": "shadow mode (enforce_tech_ablation=False)",
            "shadow_key": shadow_key,
            "per_sector": per_sector,
        }

    # Live-write gate: reproduction across last N shadow archives.
    # NB the current week's archive has just been written, so the
    # gate reads N archives starting with this one.
    gate = _check_reproduction_gate(s3, bucket, per_sector)
    if not gate["passed"]:
        return {
            "applied": False,
            "reason": gate["reason"],
            "shadow_key": shadow_key,
            "per_sector": per_sector,
            "reproduction_gate": gate,
        }

    try:
        s3.put_object(
            Bucket=bucket, Key=S3_LIVE_KEY, Body=body,
            ContentType="application/json",
        )
        logger.info(
            "[tech_weight_ablation] live config updated: s3://%s/%s — "
            "per_sector=%s",
            bucket, S3_LIVE_KEY, per_sector,
        )
    except Exception as e:
        logger.error(
            "[tech_weight_ablation] CRITICAL: live S3 write failed: %s", e,
        )
        return {
            "applied": False,
            "reason": f"live S3 write failed: {e}",
            "shadow_key": shadow_key,
            "per_sector": per_sector,
        }

    return {
        "applied": True,
        "reason": "live config written + shadow archive recorded",
        "live_key": S3_LIVE_KEY,
        "shadow_key": shadow_key,
        "per_sector": per_sector,
        "reproduction_gate": gate,
    }
