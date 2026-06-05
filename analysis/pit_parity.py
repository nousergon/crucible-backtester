"""pit_parity.py — proof-of-impact for point-in-time discipline (PR 3, plan
``alpha-engine-docs/private/pit-discipline-260515.md`` §D4; ROADMAP L2371).

Runs the predictor backtest twice over the **same date grid** — once on the
legacy single-pass path (current live weights + current configs:
look-ahead-contaminated) and once on the walk-forward PIT path (PR 1 slice C
+ PR 2) — and emits a contamination report on the **codified skilled-risk
basket**: ΔSortino (primary), ΔPSR, ΔCVaR, Δmax-DD, plus PBO and a
Δlog-alpha-vs-SPY headline. **No Sharpe column** — Sharpe is deprecated as a
fit target (plan §3 invariant 4 / `param_sweep.py:241` sorts on Sortino).
All return/alpha quantities are log-domain market-relative (plan invariant
5): the basket's `daily_log_returns` are summed, never compounded
arithmetically.

This both proves the leak was material (a non-trivial delta ⇒ the optimizer
was selecting params on future-trained weights) and becomes the regression
tripwire: a later code change that re-introduces look-ahead moves these
numbers. It is **observational** — it never gates the Saturday SF and never
writes optimizer configs. The `--walk-forward` default flip stays a manual,
Brian-gated decision made *after* reading this report (plan §5).
"""

from __future__ import annotations

import copy
import datetime as _dt
import json
import logging

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA = "pit_parity-1.0.0"


def _predictor_pass_worker(cfg: dict) -> dict:
    """Module-level worker run in a spawned child process — must be importable
    by qualified name for ``spawn`` pickling. Returns the predictor-backtest
    stats dict back to the parent."""
    from backtest import run_predictor_backtest

    return run_predictor_backtest(cfg)


def _run_predictor_pass_isolated(cfg: dict) -> dict:
    """Run ONE predictor-backtest pass in a fresh subprocess so the OS reclaims
    100% of its RSS on exit before the next pass starts (L4486).

    pit_parity runs TWO full predictor pipelines back-to-back (look-ahead +
    walk-forward). In a single process CPython retains pass-1's ~3 GB RSS
    (glibc keeps the arena mapped), so pass-2's pre-pipeline RAM-headroom guard
    saw only ~3.9 GB free < 6 GB on the 8 GB Parity spot and aborted (2026-06-05
    scoped validation). The earlier ``malloc_trim`` (#284) was best-effort and
    not guaranteed to release pandas/Arctic/lightgbm pools; process isolation is
    the guaranteed fix — pass-1's child fully exits, the OS reclaims everything,
    pass-2's child starts clean at ~320 MB. No bigger instance.

    Uses ``spawn`` (NOT ``fork``): a forked child would inherit the parent's
    boto3 / ArcticDB sockets and HTTP pools and corrupt them mid-flight. The
    child inherits stdout/stderr so its per-pass MEM logs still reach the
    SSM-captured log. A separate single-worker pool per call guarantees the
    child is torn down before the next pass.
    """
    import concurrent.futures as _cf
    import multiprocessing as _mp

    ctx = _mp.get_context("spawn")
    with _cf.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
        return ex.submit(_predictor_pass_worker, cfg).result()

# Runtime handles that live on ``config`` but cannot be deep-copied. The
# PhaseRegistry's ``.s3_client`` carries botocore service-model references
# that recurse past the Python stack limit (caught 2026-04-27 spot smoke v2
# at ``backtest.py::merge_executor_params``; re-bit pit_parity 2026-05-17
# through 2026-05-24 — 4 Saturday firings silently swallowed RecursionError
# in the outer non-fatal handler). Mirror the explicit-allowlist strip
# pattern at ``backtest.py:862`` rather than a prefix-based filter so
# load-bearing ``_run_date`` (read at line 213) is never accidentally
# dropped.
_RUNTIME_HANDLE_KEYS = ("_phase_registry",)


def _config_without_runtime_handles(config: dict) -> dict:
    """Return a shallow dict-view of ``config`` with non-copyable runtime
    handles removed so the result is safe to ``copy.deepcopy``.

    ``run_predictor_backtest`` reads only data keys (``executor_paths``,
    ``init_cash``, ``simulation_fees``, ``use_vectorized_sweep``,
    ``walk_forward``) — none of the runtime handles — so dropping them is
    behaviour-neutral for both parity passes.
    """
    return {k: v for k, v in config.items() if k not in _RUNTIME_HANDLE_KEYS}


def write_failure_artifact(
    config: dict,
    exc: BaseException,
) -> dict:
    """Construct and upload a ``status=failed`` artifact when
    ``run_pit_parity`` raises an unhandled exception (e.g., the
    RecursionError class). The always-emit-artifact contract guarantees
    the operator's manual-flip gate always has something to read — no
    Saturday produces zero artifacts.

    Returns the failure report (with ``_s3_key`` populated on upload
    success). Never raises — observability path only.
    """
    bucket = config.get("signals_bucket", "alpha-engine-research")
    run_date = config.get("_run_date") or _dt.date.today().isoformat()
    report = {
        "schema": SCHEMA,
        "run_date": run_date,
        "status": "failed",
        "error_class": type(exc).__name__,
        "error_msg": str(exc)[:1000],
        "observational": True,
    }
    key = _write_artifact_to_s3(bucket, run_date, report)
    if key is not None:
        report["_s3_key"] = key
    return report


def _write_artifact_to_s3(bucket: str, run_date: str, report: dict) -> str | None:
    """Best-effort upload of the parity report to the canonical S3 key.

    Returns the S3 key on success, ``None`` on failure. Never raises — pit_parity
    is observational and a failed upload must not fail the spot run. The
    fail-loud surface is ``alpha_engine_lib.alerts.publish`` at the
    ``backtest.py::main`` outer handler (per ``feedback_no_silent_fails``
    secondary-observability carve-out: the primary deliverable is the
    weights archive + the spot run; pit_parity is the secondary record).
    """
    key = f"backtest/{run_date}/pit_parity.json"
    try:
        import boto3
        boto3.client("s3").put_object(
            Bucket=bucket, Key=key,
            Body=json.dumps(report, indent=2, default=str),
            ContentType="application/json",
        )
        logger.info("[pit_parity] report → s3://%s/%s", bucket, key)
        return key
    except Exception as e:
        logger.warning(
            "[pit_parity] S3 upload failed (best-effort, observational): %s", e
        )
        return None


# The skilled-risk basket (plan §3 invariant 4). Sortino is primary; PSR is
# the basket's deflation member; CVaR + max-DD are the tail/drawdown legs.
# Sharpe is deliberately absent.
_BASKET_KEYS = ("sortino_ratio", "psr", "cvar_95", "max_drawdown")


def _log_cum_return(stats: dict) -> float | None:
    """Log-domain cumulative portfolio return (time-additive, plan inv. 5).

    Prefers the pre-computed ``daily_log_returns`` series
    (vectorbt_bridge.py:208); falls back to ``log1p(total_return)``.
    """
    dlr = stats.get("daily_log_returns")
    if dlr is not None:
        arr = np.asarray(dlr, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            return float(arr.sum())
    tr = stats.get("total_return")
    if tr is not None and tr > -1.0:
        return float(np.log1p(tr))
    return None


def _basket(stats: dict) -> dict:
    b = {k: stats.get(k) for k in _BASKET_KEYS}
    b["log_cum_return"] = _log_cum_return(stats)
    # Arithmetic SPY-relative alpha kept ONLY as the end-user headline
    # (plan D4 — "not a fit target").
    b["total_alpha"] = stats.get("total_alpha")
    return b


def _delta(pit: dict, cur: dict) -> dict:
    """pit − current, per basket key. None if either side is missing."""
    out = {}
    for k in (*_BASKET_KEYS, "log_cum_return", "total_alpha"):
        a, b = pit.get(k), cur.get(k)
        out[k] = (float(a) - float(b)) if (a is not None and b is not None) else None
    return out


def _pbo_two_split(cur_sweep_df, pit_sweep_df) -> dict | None:
    """Probability-of-Backtest-Overfitting via the degenerate **2-split
    CSCV** the parity pair naturally forms (Bailey & López de Prado 2014):
    the current-weights sweep is the in-sample (look-ahead-optimistic)
    realization, the walk-forward sweep is the out-of-sample one, of the
    *same config grid*. Metric = Sortino rank (plan invariant 4).

    Reports the Spearman rank correlation of config Sortino-ranks across
    the two realizations and the OOS percentile of the best-in-sample
    config. ``overfit`` flags the canonical CSCV condition: the IS-optimal
    config lands below the OOS median. A full M-block CSCV distribution
    needs per-block sweep re-evaluation (S× sweep cost) and is documented
    as a future enhancement, mirroring the plan's CPCV-as-future-option
    discipline — we do not fabricate a distribution we did not compute.

    ``None`` when no sweep pair is supplied (single-pass parity) — the
    report then carries ``pbo: null`` with an explicit method note rather
    than a fabricated number.
    """
    if cur_sweep_df is None or pit_sweep_df is None:
        return None
    col = "sortino_ratio"
    if col not in cur_sweep_df.columns or col not in pit_sweep_df.columns:
        return None
    join = ("config_id" if "config_id" in cur_sweep_df.columns
            and "config_id" in pit_sweep_df.columns else None)
    if join:
        m = cur_sweep_df[[join, col]].merge(
            pit_sweep_df[[join, col]], on=join, suffixes=("_is", "_oos"),
        )
        is_v = m[f"{col}_is"].to_numpy(float)
        oos_v = m[f"{col}_oos"].to_numpy(float)
    else:
        n = min(len(cur_sweep_df), len(pit_sweep_df))
        if n < 2:
            return None
        is_v = cur_sweep_df[col].to_numpy(float)[:n]
        oos_v = pit_sweep_df[col].to_numpy(float)[:n]
    n = is_v.size
    if n < 2 or not np.isfinite(is_v).any() or not np.isfinite(oos_v).any():
        return None
    is_rank = np.argsort(np.argsort(is_v))
    oos_rank = np.argsort(np.argsort(oos_v))
    spearman = float(np.corrcoef(is_rank, oos_rank)[0, 1]) if n > 1 else None
    best_is = int(np.nanargmax(is_v))
    oos_pct = float(oos_rank[best_is] / (n - 1)) if n > 1 else None
    return {
        "method": "2-split CSCV (current=in-sample, walk-forward=OOS); "
                  "Sortino-rank. Full M-block CSCV distribution = future "
                  "enhancement (not fabricated here).",
        "n_configs": int(n),
        "spearman_rank_corr": spearman,
        "best_in_sample_config_oos_percentile": oos_pct,
        "overfit": (oos_pct is not None and oos_pct < 0.5),
    }


def build_contamination_report(
    cur_stats: dict,
    pit_stats: dict,
    *,
    run_date: str | None = None,
    wf_meta: dict | None = None,
    cur_sweep_df=None,
    pit_sweep_df=None,
) -> dict:
    """Assemble the skilled-risk-basket contamination report (pure)."""
    cur_b, pit_b = _basket(cur_stats), _basket(pit_stats)
    delta = _delta(pit_b, cur_b)
    pbo = _pbo_two_split(cur_sweep_df, pit_sweep_df)

    n_cfg = None
    if cur_sweep_df is not None:
        n_cfg = int(len(cur_sweep_df))

    material = bool(
        delta.get("sortino_ratio") is not None
        and abs(delta["sortino_ratio"]) >= 0.10
    )
    return {
        "schema": SCHEMA,
        "run_date": run_date or _dt.date.today().isoformat(),
        "anchor": (
            "skilled_risk_basket — Sortino (primary), PSR (deflation), "
            "CVaR, max-DD; log-domain market-relative (plan inv. 4/5). "
            "Sharpe deliberately absent (deprecated fit target)."
        ),
        "current_lookahead": cur_b,   # legacy single-pass (contaminated)
        "walk_forward_pit": pit_b,    # PR1+PR2 point-in-time
        "delta_pit_minus_current": delta,
        "headline_log_alpha_delta": delta.get("log_cum_return"),
        "pbo": pbo,
        "run_quality": {
            "walk_forward": wf_meta,   # fold count, cold-start exclusions…
            "n_configs_swept": n_cfg,
            "note": (
                "Δ on the basket is the contamination magnitude. A large "
                "|ΔSortino| means the optimizer was ranking configs on "
                "look-ahead-trained weights. Cold-start-excluded folds are "
                "an archive-coverage finding, not a code bug."
            ),
        },
        "materiality": {
            "sortino_delta_threshold": 0.10,
            "material": material,
            "interpretation": (
                "MATERIAL — the look-ahead leak measurably moved the "
                "anchor metric; review before the flip."
                if material else
                "Below the ΔSortino materiality threshold on this grid — "
                "still review run_quality (cold-start coverage can mask a "
                "true delta by shrinking the PIT sample)."
            ),
        },
        "observational": True,
        "flip_gate": (
            "Manual + Brian-gated (plan §5). This report is the input to "
            "that decision; --walk-forward stays DEFAULT OFF until then."
        ),
    }


def run_pit_parity(config: dict) -> dict:
    """Run the predictor backtest both ways over the same grid + build the
    report. Returns the report dict; the caller persists it.

    ``run_predictor_backtest`` (predictor-only sim) is self-contained and
    returns the skilled-risk-basket stats directly, so the parity pair is
    two of those — no param-sweep / phase-registry plumbing. PBO needs a
    sweep pair; in this single-pass mode it is reported ``null`` with a
    method note (no fabricated distribution).
    """
    # NB: run_predictor_backtest is imported INSIDE the subprocess worker
    # (_predictor_pass_worker), not here — each pass runs in its own process.
    bucket = config.get("signals_bucket", "alpha-engine-research")
    run_date = config.get("_run_date") or _dt.date.today().isoformat()

    # Strip non-copyable runtime handles before deepcopy. See module-level
    # ``_RUNTIME_HANDLE_KEYS`` comment for the failure-mode history.
    safe_config = _config_without_runtime_handles(config)

    # L4486: each pass runs in its OWN subprocess so the OS reclaims pass-1's
    # full RSS before pass-2 starts (guaranteed; supersedes the #284 malloc_trim
    # best-effort). See _run_predictor_pass_isolated for the why.
    cur_cfg = copy.deepcopy(safe_config)
    cur_cfg["walk_forward"] = False
    logger.info("[pit_parity] pass 1/2 — legacy single-pass (look-ahead) [isolated subprocess]")
    cur_stats = _run_predictor_pass_isolated(cur_cfg)

    pit_cfg = copy.deepcopy(safe_config)
    pit_cfg["walk_forward"] = True
    logger.info("[pit_parity] pass 2/2 — walk-forward PIT [isolated subprocess]")
    pit_stats = _run_predictor_pass_isolated(pit_cfg)

    if cur_stats.get("status") not in (None, "ok") or \
            pit_stats.get("status") not in (None, "ok"):
        logger.error(
            "[pit_parity] a parity pass did not complete "
            "(current=%s, pit=%s) — emitting a status-only report",
            cur_stats.get("status"), pit_stats.get("status"),
        )
        report = {
            "schema": SCHEMA, "run_date": run_date, "status": "incomplete",
            "current_status": cur_stats.get("status"),
            "pit_status": pit_stats.get("status"),
            "observational": True,
        }
        # Always-emit-artifact contract: the operator's manual-flip gate
        # (plan §5 / ROADMAP L2371) depends on a weekly artifact existing.
        # A silent skip of the upload on the incomplete path would replay
        # the 2026-05-17→2026-05-24 silent-fail incident.
        key = _write_artifact_to_s3(bucket, run_date, report)
        if key is not None:
            report["_s3_key"] = key
        return report

    wf_meta = (pit_stats.get("predictor_metadata") or {}).get("walk_forward")
    report = build_contamination_report(
        cur_stats, pit_stats, run_date=run_date, wf_meta=wf_meta,
    )

    key = _write_artifact_to_s3(bucket, run_date, report)
    if key is not None:
        report["_s3_key"] = key
    return report
