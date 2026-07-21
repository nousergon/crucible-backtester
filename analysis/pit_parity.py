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

import datetime as _dt
import json
import logging
import os
from pathlib import Path

import numpy as np

from analysis.parity_alarms import evaluate_parity_alarms

logger = logging.getLogger(__name__)

SCHEMA = "pit_parity-1.0.0"


def _run_predictor_pass_isolated(safe_config: dict, which: str, run_date: str) -> dict:
    """Run ONE pit_parity predictor pass in a fresh `backtest.py` subprocess and
    return its stats dict (L4487).

    pit_parity runs TWO full predictor pipelines back-to-back (look-ahead +
    walk-forward). In one process CPython/glibc retain pass-1's ~3 GB RSS, so
    pass-2's pre-pipeline headroom guard saw only ~3.9 GB free on an 8 GB box and
    aborted (2026-06-05). Running each pass as a separate OS process bounds the
    footprint to one pass (O(max), not O(sum)) — so the Parity stage runs on the
    cheap 8 GB floor again.

    ``subprocess.run`` (NOT ``multiprocessing``) is deliberate: spawn re-imports
    the parent's ``__main__`` in the child, and with both the backtester and
    predictor repos shipping an ``analysis`` package on ``sys.path`` the child
    resolved the wrong one → ImportError → BrokenProcessPool (#285). A fresh
    ``python backtest.py`` with ``cwd`` = the backtester repo has no such
    re-import and resolves ``analysis`` correctly.

    Config crosses to the child as JSON (it is a plain deepcopy-safe dict); the
    stats dict comes back as pickle (numpy/pandas-safe). Raises on a non-zero
    child exit — the caller's observational handler records it.
    """
    import json
    import pickle
    import subprocess
    import sys
    import tempfile

    repo_root = Path(__file__).resolve().parents[1]
    pass_flag = "walkforward" if which == "walkforward" else "lookahead"
    with tempfile.TemporaryDirectory(prefix="pit_parity_") as td:
        cfg_path = os.path.join(td, "config.json")
        stats_path = os.path.join(td, "stats.pkl")
        with open(cfg_path, "w") as f:
            json.dump(safe_config, f)
        cmd = [
            sys.executable, str(repo_root / "backtest.py"),
            "--pit-parity-pass", pass_flag,
            "--config-json", cfg_path,
            "--stats-out", stats_path,
            "--date", run_date,
            "--log-level", "INFO",
        ]
        # L4487b: capture the child's stderr so a non-zero exit surfaces the
        # ACTUAL cause, not a bare "exit 1". The child's stdout still inherits
        # (live MEM logs reach the captured spot log); only stderr is piped so
        # the traceback is preserved even when the SSM-relayed stream drops it.
        # A crashed pass must surface loud (caught by run_pit_parity's
        # observational handler) — never silently yield empty stats.
        proc = subprocess.run(
            cmd, cwd=str(repo_root), stderr=subprocess.PIPE, text=True,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip()[-2000:]
            logger.error(
                "[pit_parity] %s pass subprocess failed (rc=%s); child stderr tail:\n%s",
                which, proc.returncode, tail,
            )
            raise RuntimeError(
                f"pit_parity {which} pass failed (rc={proc.returncode}): {tail[-500:]}"
            )
        with open(stats_path, "rb") as f:
            return pickle.load(f)

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


def handle_pit_parity_failure(config: dict, exc: BaseException) -> dict:
    """Single chokepoint for the ``run_pit_parity`` outer-exception path
    (``backtest.py::main``'s ``--pit-parity`` branch): (1) always-emit the
    ``status=failed`` S3 artifact via :func:`write_failure_artifact`, then
    (2) page a WARNING-severity Telegram + SNS alert via
    ``nousergon_lib.alerts.publish``, naming ``run_date`` + ``error_class``
    and deduped on ``run_date`` (720 min — one alert per Saturday cycle even
    across a swept-cycle retry).

    config#3120: a persisted ``status=failed`` record with zero alerting
    left the 2026-07-17 week's pit-parity contamination report silently
    absent for 30+ hours (same silent-warning class as the
    2026-05-17->2026-05-24 incident this module's always-emit-artifact
    contract already closed the S3 half of). This closes the alerting half:
    extracted into its own function (previously inlined in
    ``backtest.py::main``) so the paging behavior is directly unit-testable
    with a mocked alert sender, not just reachable via a full spot run.

    Never raises — both the artifact write and the alert publish are
    best-effort observability steps; pit_parity stays non-blocking
    (observational posture unchanged — this pages, it never halts the
    pipeline). Returns the failure report dict (mirrors
    ``write_failure_artifact``'s return value).
    """
    logger.error(
        "[pit_parity] run failed (observational, non-fatal): %s",
        exc, exc_info=True,
    )
    try:
        report = write_failure_artifact(config, exc)
    except Exception as artifact_err:
        logger.error(
            "[pit_parity] failure-artifact write also failed: %s",
            artifact_err,
        )
        report = {
            "schema": SCHEMA,
            "run_date": config.get("_run_date") or _dt.date.today().isoformat(),
            "status": "failed",
            "error_class": type(exc).__name__,
            "error_msg": str(exc)[:1000],
            "observational": True,
        }

    run_date = config.get("_run_date") or "unknown"
    error_class = type(exc).__name__
    bucket = config.get("signals_bucket", "alpha-engine-research")
    try:
        from nousergon_lib.alerts import publish as _alerts_publish
        _alerts_publish(
            f"pit_parity failed on {run_date}: "
            f"{error_class}: {str(exc)[:200]} — "
            f"see s3://{bucket}/backtest/{run_date}/pit_parity.json",
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
    return report


def _write_artifact_to_s3(bucket: str, run_date: str, report: dict) -> str | None:
    """Best-effort upload of the parity report to the canonical S3 key.

    Returns the S3 key on success, ``None`` on failure. Never raises — pit_parity
    is observational and a failed upload must not fail the spot run. The
    fail-loud surface is ``nousergon_lib.alerts.publish`` at the
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


def read_prior_delta(bucket: str, run_date: str, s3_client=None) -> dict | None:
    """Read the most recent prior pit_parity report's
    ``delta_pit_minus_current`` basket, probing backward from ``run_date``
    (NOT wall-clock today — a ``--date`` backfill run must seed its prior
    relative to the backfilled trading day, mirroring
    ``champion_promotion.read_prior_leaderboard_history``'s same anchor
    choice). Returns ``None`` (first-ever run / cold start — the caller
    passes that straight through to ``evaluate_parity_alarms`` as
    ``prior_delta=None``, which is documented as N/A, not silently-passing)
    if no prior report is found within the probe window or any read fails.

    Best-effort / never raises: pit_parity is observational (see
    ``_write_artifact_to_s3``'s docstring) — a prior-delta lookup failure
    must not fail the current run's own report.
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:  # pragma: no cover - boto3 always available in prod
        return None

    s3 = s3_client or boto3.client("s3")
    anchor = _dt.date.fromisoformat(run_date)
    for back in range(1, 15):
        probe_date = (anchor - _dt.timedelta(days=back)).isoformat()
        key = f"backtest/{probe_date}/pit_parity.json"
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj["Body"].read())
            delta = data.get("delta_pit_minus_current")
            if delta is not None:
                return delta
            # Report exists but has no delta (e.g. an "incomplete" status
            # report) — keep probing further back for a real prior delta.
            continue
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                continue
            logger.warning("[pit_parity] prior-delta probe failed at %s: %s", key, e)
            return None
        except Exception as e:  # noqa: BLE001
            logger.warning("[pit_parity] prior-delta probe failed at %s: %s", key, e)
            return None
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


def _cscv_pbo(pit_block_matrix, spec_ids=None, *, min_splits: int = 4) -> dict | None:
    """Full **M-block combinatorial CSCV** PBO on the PIT (walk-forward) pass
    ONLY (#816 gap 1+2; decision B — canonical López de Prado: CSCV runs on
    the strategy you deploy, not the contaminated look-ahead comparator).

    ``pit_block_matrix`` is shape ``(n_blocks, n_combos)`` — one row per
    chronological CSCV block of the walk-forward sweep, one column per param
    combo, every cell the SAME metric (Sortino, plan invariant 4) evaluated on
    that block. This is exactly the aligned-trial matrix
    :func:`analysis.pbo.cscv_pbo` (Bailey/Borwein/López de Prado/Zhu 2014)
    consumes — the same calling convention proven at
    ``optimizer/executor_optimizer.py::_compute_pbo``. The engine's
    leave-one-split-out symmetric selection test yields a real PBO
    *distribution* (``lambda`` logits), not the prior degenerate 2-split
    rank-correlation.

    Returns the engine's result dict (``pbo``, ``n_splits``, ``n_specs``,
    ``status``) verbatim so the honest-N/A posture is preserved:
    ``status="insufficient"`` when there are <2 combos or <``min_splits``
    clean blocks — never a fabricated pass.

    ``None`` when no PIT block matrix is supplied (single-pass parity, no
    opt-in sweep) — the report then carries ``pbo: null`` with a method note
    rather than a fabricated number (A-grade integrity property).
    """
    if pit_block_matrix is None:
        return None
    mat = np.asarray(pit_block_matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] < 2 or mat.shape[1] < 2:
        return None

    from analysis.pbo import cscv_pbo

    ids = list(spec_ids) if spec_ids is not None else list(range(mat.shape[1]))
    res = cscv_pbo(mat.tolist(), spec_ids=ids, min_splits=min_splits)
    res = dict(res)
    res.setdefault(
        "method",
        "full M-block combinatorial CSCV (Bailey/Borwein/López de Prado/Zhu "
        "2014) on the PIT walk-forward sweep ONLY (decision B); metric = "
        "Sortino (plan inv. 4). PBO = P(in-sample winner lands below the OOS "
        "median).",
    )
    return res


def _block_bootstrap_ci(
    delta_series,
    *,
    ci_level: float = 0.95,
    n_resamples: int = 2000,
    block_size: int | None = None,
    seed: int = 0,
) -> dict | None:
    """Moving-block bootstrap CI on the per-date ΔSortino-contributing return
    delta stream (#816 gap 3; decision C — materiality = a block-bootstrap CI
    that **excludes 0** as the primary trigger, replacing the arbitrary
    ``abs(ΔSortino) >= 0.10`` hard threshold).

    The delta stream is a *time series* (per-date PIT-minus-current portfolio
    log-return deltas) with serial dependence, so an IID percentile bootstrap
    understates the CI. We resample overlapping length-``block_size`` blocks
    (Künsch 1989 moving-block bootstrap; block ≈ ``n**(1/3)`` by default) to
    preserve short-range autocorrelation. The shared
    ``nousergon_lib.quant.stats.intervals.bootstrap_ci`` helper is IID-only (not
    block-aware), so the block resampling is done here and the percentile CI is
    computed on the block-bootstrap distribution directly — same percentile
    contract as ``bootstrap_ci`` (``ci_low``/``ci_high``/``estimate``).

    Returns ``{status, n, estimate, ci_low, ci_high, ci_level, method,
    n_resamples, block_size, excludes_zero}``. ``excludes_zero`` is the
    primary materiality trigger: True iff the whole CI lies on one side of 0.
    ``None`` / ``status="insufficient_data"`` when <2 finite observations —
    never a fabricated interval.
    """
    if delta_series is None:
        return None
    arr = np.asarray(delta_series, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 2:
        return {"status": "insufficient_data", "n": int(n),
                "method": "moving-block bootstrap", "excludes_zero": False}

    if block_size is None:
        block_size = max(1, int(round(n ** (1.0 / 3.0))))
    block_size = min(block_size, n)
    n_blocks = int(np.ceil(n / block_size))

    rng = np.random.default_rng(seed)
    # Overlapping start positions for the moving-block bootstrap.
    max_start = n - block_size + 1
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        starts = rng.integers(0, max_start, size=n_blocks)
        sample = np.concatenate([arr[s:s + block_size] for s in starts])[:n]
        means[i] = sample.mean()

    alpha = (1.0 - ci_level) / 2.0
    ci_low = float(np.quantile(means, alpha))
    ci_high = float(np.quantile(means, 1.0 - alpha))
    estimate = float(arr.mean())
    excludes_zero = bool(ci_low > 0.0 or ci_high < 0.0)
    return {
        "status": "ok",
        "n": int(n),
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": float(ci_level),
        "method": "moving-block bootstrap (Künsch 1989); percentile CI on the "
                  "per-date PIT-minus-current return delta.",
        "n_resamples": int(n_resamples),
        "block_size": int(block_size),
        "excludes_zero": excludes_zero,
    }


def _dsr_materiality(pit_stats: dict, n_trials: int | None) -> dict | None:
    """Deflated Sharpe Ratio on the PIT (walk-forward) winner's return stream
    (#816 gap 3; decision C — DSR on the PSR axis, alongside the block-bootstrap
    Δ-CI). DSR deflates the winner's Sharpe for the selection bias of picking
    the best of ``n_trials`` swept combos — a positive DSR means the deployed
    PIT config's skill survives multiple-testing correction.

    ``None`` when the PIT daily return stream or ``n_trials`` is absent (no
    opt-in sweep) — the honest-N/A posture. Never fabricates.
    """
    if not n_trials or n_trials < 1:
        return None
    dlr = pit_stats.get("daily_log_returns")
    if dlr is None:
        return None
    arr = np.asarray(dlr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return None
    try:
        from analysis.dsr import compute_dsr
    except Exception:  # pragma: no cover - dsr shim always present in prod
        return None
    # DSR is defined on simple returns; per plan inv. 5 the PIT stream is
    # log-domain, so convert back to simple returns for the Sharpe moments.
    res = compute_dsr(np.expm1(arr), int(n_trials))
    return dict(res)


def _per_date_return_delta(pit_stats: dict, cur_stats: dict):
    """Aligned per-date PIT-minus-current portfolio log-return delta series,
    the input to the block-bootstrap materiality CI (decision C).

    Both passes run over the SAME date grid (module docstring), so the two
    ``daily_log_returns`` streams are index-aligned; we truncate to the shorter
    (cold-start-excluded PIT folds can shorten the PIT stream). ``None`` when
    either stream is absent or too short — the CI then reports insufficient.
    """
    p, c = pit_stats.get("daily_log_returns"), cur_stats.get("daily_log_returns")
    if p is None or c is None:
        return None
    p = np.asarray(p, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    n = min(p.size, c.size)
    if n < 2:
        return None
    d = p[:n] - c[:n]
    d = d[np.isfinite(d)]
    return d if d.size >= 2 else None


def build_contamination_report(
    cur_stats: dict,
    pit_stats: dict,
    *,
    run_date: str | None = None,
    wf_meta: dict | None = None,
    pit_block_matrix=None,
    pit_spec_ids=None,
    n_trials: int | None = None,
    prior_delta: dict | None = None,
) -> dict:
    """Assemble the skilled-risk-basket contamination report (pure).

    Statistical rigor (#816):

    - **PBO** — full M-block combinatorial CSCV on the PIT walk-forward sweep
      ONLY (``pit_block_matrix``; decision B). ``pit_block_matrix is None``
      (single-pass parity, no opt-in sweep) ⇒ ``pbo: null`` with a method note,
      never a fabricated distribution.
    - **Materiality** — block-bootstrap CI on the per-date return delta that
      **excludes 0** as the primary trigger, plus DSR on the PSR axis
      (decision C), replacing the arbitrary ``abs(ΔSortino) >= 0.10`` hard
      threshold. Falls back to the legacy threshold only when the CI cannot be
      computed (no per-date streams), so single-pass callers still get a signal.

    ``prior_delta`` is the previous run's ``delta_pit_minus_current`` basket
    (config#2449) — the caller (``run_pit_parity``) reads it back via
    ``read_prior_delta`` and passes it in here; this function stays pure
    (no S3 I/O) and simply forwards it to ``evaluate_parity_alarms`` so the
    step-change alarm leg has a real baseline instead of always evaluating
    against ``None``. ``None`` (first-ever run / lookup failure) still means
    "no baseline yet" and the step-change leg stays inert for this run —
    that is documented N/A behavior, not a regression.
    """
    cur_b, pit_b = _basket(cur_stats), _basket(pit_stats)
    delta = _delta(pit_b, cur_b)
    pbo = _cscv_pbo(pit_block_matrix, spec_ids=pit_spec_ids)

    n_cfg = None
    if pit_block_matrix is not None:
        try:
            n_cfg = int(np.asarray(pit_block_matrix).shape[1])
        except Exception:
            n_cfg = None
    elif n_trials:
        n_cfg = int(n_trials)

    # Decision C — bootstrap CI on the per-date delta (primary trigger:
    # CI excludes 0) + DSR on the deployed PIT winner.
    delta_stream = _per_date_return_delta(pit_stats, cur_stats)
    boot_ci = _block_bootstrap_ci(delta_stream)
    dsr = _dsr_materiality(pit_stats, n_trials)

    if boot_ci is not None and boot_ci.get("status") == "ok":
        material = bool(boot_ci.get("excludes_zero"))
        materiality_basis = "block_bootstrap_ci_excludes_zero"
        materiality_interp = (
            "MATERIAL — the block-bootstrap 95% CI on the per-date "
            "PIT-minus-current return delta EXCLUDES 0: the look-ahead leak "
            "moved the anchor by a statistically-distinguishable amount. "
            "Review before the flip."
            if material else
            "Not material — the block-bootstrap 95% CI on the per-date return "
            "delta INCLUDES 0 (delta indistinguishable from noise on this "
            "grid). Still review run_quality (cold-start coverage can shrink "
            "the PIT sample and widen the CI)."
        )
    else:
        # No per-date streams (e.g. legacy single-pass callers passing only
        # scalar stats) — fall back to the legacy ΔSortino threshold so the
        # report still carries a materiality signal.
        material = bool(
            delta.get("sortino_ratio") is not None
            and abs(delta["sortino_ratio"]) >= 0.10
        )
        materiality_basis = "legacy_sortino_delta_threshold_fallback"
        materiality_interp = (
            "MATERIAL (legacy |ΔSortino|>=0.10 fallback — no per-date return "
            "streams available for the bootstrap CI)."
            if material else
            "Below the legacy ΔSortino threshold (bootstrap CI unavailable — "
            "no per-date return streams)."
        )
    # Leg (g) — tolerance-band + step-change alarms over the full basket delta,
    # OBSERVE mode (computed + recorded in the report; never pages from the
    # pure builder — paging_enabled stays False here regardless of prior_delta;
    # the paging flip itself is a separate, Brian-gated decision, config#2449).
    report_date = run_date or _dt.date.today().isoformat()
    alarms = evaluate_parity_alarms(delta, prior_delta, run_date=report_date)
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
            "material": material,
            "basis": materiality_basis,
            "bootstrap_ci": boot_ci,   # block-bootstrap Δ-CI (primary; decision C)
            "dsr": dsr,                # Deflated Sharpe on the PIT winner (PSR axis)
            "interpretation": materiality_interp,
        },
        "alarms": alarms,
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
    # NB: each pass runs run_predictor_backtest in its own subprocess
    # (_run_predictor_pass_isolated) — not imported/called in this process.
    bucket = config.get("signals_bucket", "alpha-engine-research")
    run_date = config.get("_run_date") or _dt.date.today().isoformat()

    # Strip non-copyable runtime handles before deepcopy. See module-level
    # ``_RUNTIME_HANDLE_KEYS`` comment for the failure-mode history.
    safe_config = _config_without_runtime_handles(config)

    # L4487: each pass runs in its OWN subprocess (a fresh `python backtest.py
    # --pit-parity-pass …`), so the OS reclaims pass-1's full RSS before pass-2
    # starts — bounded O(max single pass), not O(sum) which had needed a 16 GB
    # box. subprocess.run (not multiprocessing) is deliberate: a fresh process
    # with cwd=backtester resolves the `analysis` package correctly, sidestepping
    # the spawn __main__ re-import collision that broke the multiprocessing
    # attempt (#285).
    logger.info("[pit_parity] pass 1/2 — legacy single-pass (look-ahead) [isolated subprocess]")
    cur_stats = _run_predictor_pass_isolated(safe_config, "lookahead", run_date)

    logger.info("[pit_parity] pass 2/2 — walk-forward PIT [isolated subprocess]")
    pit_stats = _run_predictor_pass_isolated(safe_config, "walkforward", run_date)

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
    # config#2449: read back the most recent prior run's delta so the
    # step-change alarm leg has a real baseline instead of always
    # evaluating against None (which made it permanently inert).
    prior_delta = read_prior_delta(bucket, run_date)
    # #816 decision B: CSCV inputs come from the PIT (walk-forward) pass ONLY.
    # The walk-forward pass, when the per-block ``pit_parity_sweep`` flag is set
    # (decision A — opt-in; single-pass callers pass no flag and get pbo=null),
    # runs run_predictor_param_sweep and returns the (n_blocks, n_combos)
    # Sortino block matrix under ``_cscv_block_matrix`` in its stats.
    pit_block_matrix = pit_stats.get("_cscv_block_matrix")
    pit_spec_ids = pit_stats.get("_cscv_spec_ids")
    n_trials = pit_stats.get("_cscv_n_trials")
    report = build_contamination_report(
        cur_stats, pit_stats, run_date=run_date, wf_meta=wf_meta,
        pit_block_matrix=pit_block_matrix, pit_spec_ids=pit_spec_ids,
        n_trials=n_trials, prior_delta=prior_delta,
    )

    key = _write_artifact_to_s3(bucket, run_date, report)
    if key is not None:
        report["_s3_key"] = key
    return report
