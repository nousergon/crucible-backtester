"""
pipeline_common.py — Shared utilities for backtest.py and evaluate.py.

Config loading, research DB management, predictor metrics.
Data seeding/backfilling lives in alpha-engine-data/collectors/signal_returns.py.
"""

from __future__ import annotations

import _thread
import faulthandler
import json
import logging
import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import boto3
from botocore.exceptions import ClientError
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_MIN_IC_SAMPLES = 10
_IC_STD_EPSILON = 1e-8


# ── Predictor outcomes column-canonicalization helpers ───────────────────────
#
# Predictor 21d canonical-alpha migration (2026-05-09; plan at
# alpha-engine-docs/private/predictor-21d-migration-260509.md). New rows after
# alpha-engine-data PR #198 populate horizon-agnostic columns
# (actual_log_alpha, horizon_days, correct) — log-domain decimal alpha at the
# row's horizon-of-record (21d post Track A cutover). Old rows retain
# legacy columns (actual_5d_return in pct points, correct_5d at 5d horizon).
#
# Readers MUST use these COALESCE expressions in SQL so downstream computation
# stays scale-uniform across the transition window. Legacy `actual_5d_return`
# is divided by 100 inline so the result is decimal — same scale as the
# log-domain new column. log(1+r) ≈ r for small r, so the threshold/IC math
# works on either representation without per-row branching.
#
# The legacy fallback retires in PR F (~4 weeks of parallel writes); these
# fragments simplify to the new column at that point.
ALPHA_COALESCE_SQL = "COALESCE(actual_log_alpha, actual_5d_return / 100.0)"
CORRECT_COALESCE_SQL = "COALESCE(correct, correct_5d)"
HORIZON_COALESCE_SQL = "COALESCE(horizon_days, 5)"
OUTCOMES_RESOLVED_SQL = (
    "(actual_log_alpha IS NOT NULL OR actual_5d_return IS NOT NULL)"
)
OUTCOMES_GRADED_SQL = (
    "(correct IS NOT NULL OR correct_5d IS NOT NULL)"
)

# Active production horizon for rolling analytics. Derived from
# `labeling.forward_days` in alpha-engine-config/predictor/predictor.yaml —
# the single source of truth that also drives `predictor_outcomes.horizon_days`
# on the data-collector write side. Rolling IC / hit-rate / value-of-veto
# reads on predictor_outcomes scope to this horizon so the transition window
# doesn't blend pre-cutover (5d arithmetic) and post-cutover (21d log) rows,
# whose distributions differ by both scale (variance ~√(21/5)) and label
# semantics. Backfill / historical-range reads (e.g. weight optimizer
# cross-era sweeps) should NOT use this filter.


def _load_active_horizon_days(
    default: int = 21,
    search_paths: list[Path] | None = None,
) -> int:
    """Read `labeling.forward_days` from alpha-engine-config/predictor/predictor.yaml.

    Falls back to ``default`` when no path on ``search_paths`` exists or
    yields a value. Production runs on the spot instance always have the
    file (spot_backtest.sh clones alpha-engine-config beside this repo);
    ``search_paths`` is exposed for tests so they don't need to stub
    pathlib internals.
    """
    if search_paths is None:
        search_paths = [
            Path.home() / "alpha-engine-config" / "predictor" / "predictor.yaml",
            Path(__file__).parent.parent / "alpha-engine-config" / "predictor" / "predictor.yaml",
        ]
    for p in search_paths:
        if not p.exists():
            continue
        try:
            with open(p) as f:
                cfg = yaml.safe_load(f) or {}
            fd = cfg.get("labeling", {}).get("forward_days")
            if fd is None:
                continue
            return int(fd)
        except (OSError, yaml.YAMLError, TypeError, ValueError) as exc:
            logger.warning(
                "pipeline_common: could not read forward_days from %s: %s — "
                "falling back to default=%d", p, exc, default,
            )
            continue
    return default


ACTIVE_HORIZON_DAYS = _load_active_horizon_days()
# Strict equality (NOT `COALESCE(horizon_days, 5) = N`): legacy pre-cutover
# rows have `horizon_days IS NULL` and must be EXCLUDED, not silently
# defaulted to 5. The COALESCE-to-5 pattern smuggled 5d-arithmetic
# outcomes through the filter during the 2026-05-09 21d-log transition
# window and produced the false-positive ic_degradation retrain alert on
# 2026-05-11 (rolling=-0.1005 vs training=0.4634). The data collector
# populates `horizon_days` on the same write that sets `actual_log_alpha`,
# so any post-cutover resolved row always has a non-NULL value.
CURRENT_HORIZON_FILTER_SQL = f"horizon_days = {ACTIVE_HORIZON_DAYS}"

# Canonical-alpha cutover date (2026-05-09; alpha-engine-predictor PRs A-E).
# `horizon_days = 21` alone does NOT isolate the post-cutover model: the
# grading job stamps `horizon_days` at GRADE time, so a PRE-cutover-model
# prediction whose 21d window closed post-migration also gets
# `horizon_days = 21` with a populated `actual_log_alpha`. Those rows carry
# the OLD model's confidence/score semantics and must not drive retrain
# alerts about the CURRENT model. Production-quality analytics (rolling IC,
# regime IC, calibration ECE) therefore scope to predictions MADE on/after
# the cutover. Surfaced 2026-05-15: post-#180 the IC path was safe only by
# luck (its 30d window held zero such rows) while the 60d calibration
# window pooled 415 pre-cutover-model rows → spurious calibration_breakdown.
# Historical / cross-era reads (weight optimizer sweeps) must NOT apply this.
CANONICAL_CUTOVER_DATE = "2026-05-09"
POST_CUTOVER_FILTER_SQL = f"prediction_date >= '{CANONICAL_CUTOVER_DATE}'"


# ── Phase markers ────────────────────────────────────────────────────────────
#
# Structured begin/end log lines around each pipeline phase so any timeout
# investigation can attribute wall time to a specific phase without having
# to correlate log gaps against source code. Motivated by the 2026-04-22
# 4th Saturday SF dry-run: 110 minutes of SSM-agent silence between the
# last visible log and the 2h timeout, with no way to tell which phase
# consumed the time. See ROADMAP P0 "Diagnose the silent-phase bottleneck".
#
# Format is parseable so future tooling (CloudWatch Insights filter, a
# phase-runtime extractor, whatever) can grep on the `PHASE_START ` /
# `PHASE_END ` prefix and pull name + duration from a single line.


def _phase_logger() -> logging.Logger:
    """Dedicated logger for phase markers so callers don't need to pass one."""
    return logging.getLogger("backtest.phase")


@contextmanager
def phase(name: str, **context):
    """Emit `PHASE_START name=X ...` and `PHASE_END name=X duration_s=Y status=ok|error ...`.

    Duration is measured with monotonic time so NTP adjustments don't lie.
    stdout is flushed after each marker — SSM agent death (see the 4th
    2026-04-22 dry-run) ate ~16 minutes of buffered output; explicit flush
    + PYTHONUNBUFFERED in spot_backtest.sh closes both failure modes.
    """
    plog = _phase_logger()
    kv = " ".join(f"{k}={v}" for k, v in context.items())
    plog.info("PHASE_START name=%s %s", name, kv)
    sys.stdout.flush()
    t0 = time.monotonic()
    status = "ok"
    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        dur = time.monotonic() - t0
        plog.info("PHASE_END name=%s duration_s=%.2f status=%s %s", name, dur, status, kv)
        sys.stdout.flush()


# ── Phase registry + S3 completion markers ──────────────────────────────────
#
# Each phase writes a JSON marker to
#   s3://{bucket}/backtest/{date}/.phases/{phase}.json
# at completion. On subsequent runs with the same `date`, the registry
# reads the marker and auto-skips the phase (unless --force overrides).
# Paired with artifact persistence (PR 2/3) this gives us durable resume:
# a pipeline that crashes mid-param-sweep can be restarted and picks up
# from the failed phase without redoing simulate / data_prep / feature_maps.
#
# Marker schema (v1):
#   {
#     "phase": "simulate",
#     "date": "2026-04-23",
#     "status": "ok" | "error",
#     "started_at": "2026-04-23T16:04:12Z",
#     "completed_at": "2026-04-23T16:13:47Z",
#     "duration_s": 575.4,
#     "artifact_keys": ["backtest/2026-04-23/.phases/simulate.json"],
#     "error": null
#   }
#
# Additive fields only — future versions add fields, never rename or
# remove. Per `S3 Contract Safety` in CLAUDE.md.


_MARKER_SCHEMA_VERSION = 1


def _marker_key(date: str, phase_name: str) -> str:
    return f"backtest/{date}/.phases/{phase_name}.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Phase watchdog (trip preflight) ─────────────────────────────────────────
#
# A per-phase hard-cap timer that dumps all-thread stack traces and raises
# TimeoutError in the main thread if a phase exceeds its cap. Motivated by
# the 2026-04-22 4th Saturday SF dry-run where Phase 4 went silent at 55%
# CPU for 110 minutes before the 2h SSM ceiling fired — no stack trace, no
# idea where execution was stuck, and no abort until the whole pipeline
# burned its budget.
#
# Design:
#   - A threading.Timer fires after cap_s seconds if the phase hasn't
#     cancelled it. This is cheaper than signal.SIGALRM and portable to
#     any thread (signals only work from main thread on POSIX).
#   - On trip: faulthandler.dump_traceback(all_threads=True) to stderr
#     gives us the hung call stack; _thread.interrupt_main() raises
#     KeyboardInterrupt in the main thread, which the phase context
#     manager catches as BaseException → records PHASE_END with
#     status=error → converts to PhaseTimeoutError at the boundary.
#   - Caller only sees PhaseTimeoutError at the outer exception handler.
#
# Caps are opt-in per-phase via PhaseRegistry(hard_caps={...}). No cap
# means no watchdog — behavior identical to pre-watchdog code. Per-phase
# caps live in timing_budget.yaml under `full_run_hard_caps_seconds`.


class PhaseTimeoutError(RuntimeError):
    """Raised when a phase exceeds its hard cap. Stack traces of all
    threads have been written to stderr by faulthandler before the
    exception is raised in the main thread."""


def _default_watchdog_trip(name: str, cap_s: float) -> None:
    """Default trip handler: log PHASE_TIMEOUT, dump all-thread stacks,
    interrupt main. Exposed so tests can swap in a no-op handler."""
    plog = _phase_logger()
    plog.warning(
        "PHASE_TIMEOUT name=%s cap_s=%.1f — dumping all-thread stacks to stderr "
        "and raising PhaseTimeoutError in main thread", name, cap_s,
    )
    # Write a header so the faulthandler block is grep-able in SSM/CloudWatch
    sys.stderr.write(
        f"\n── PHASE_TIMEOUT name={name} cap_s={cap_s:.1f} ────────────────\n"
    )
    sys.stderr.flush()
    try:
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    except Exception as dump_exc:
        sys.stderr.write(f"(faulthandler.dump_traceback failed: {dump_exc})\n")
    sys.stderr.flush()
    # interrupt_main raises KeyboardInterrupt in the main thread on its
    # next bytecode dispatch. The phase context manager's except BaseException
    # catches it, records status=error, re-raises; the outer handler in
    # backtest.py maps to PhaseTimeoutError via the `_watchdog_tripped` flag.
    _thread.interrupt_main()


def _start_watchdog(
    name: str,
    cap_s: float,
    on_trip: Callable[[str, float], None] | None = None,
) -> tuple[threading.Timer, dict]:
    """Start a watchdog Timer; return (timer, state-dict).

    State dict has `tripped: bool` so the phase context manager can
    distinguish "KeyboardInterrupt from watchdog" vs "KeyboardInterrupt
    from operator Ctrl+C" and raise PhaseTimeoutError only in the former.
    """
    state = {"tripped": False, "name": name, "cap_s": cap_s}
    handler = on_trip or _default_watchdog_trip

    def _fire():
        state["tripped"] = True
        try:
            handler(name, cap_s)
        except Exception as handler_exc:
            # A broken handler shouldn't leave the watchdog in a weird
            # state. Log loud and fall back to interrupting main so the
            # phase still aborts.
            logger.error(
                "phase watchdog handler raised: %s — falling back to interrupt_main",
                handler_exc,
            )
            _thread.interrupt_main()

    timer = threading.Timer(cap_s, _fire)
    timer.daemon = True
    timer.start()
    return timer, state


class PhaseRegistry:
    """Drives per-phase skip/force decisions and writes completion markers.

    Lifecycle:
      1. Operator constructs a registry in `main()` from CLI flags.
      2. For each phase, caller uses `with registry.phase(name, ...)` —
         either (a) it's already complete for this date → ctx.skipped=True,
         caller loads the artifact from S3 instead of recomputing; or
         (b) caller runs the compute + registers any artifact keys via
         `ctx.record_artifact(key)` before the block exits.
      3. On `__exit__` the registry writes an END marker to S3 with
         duration_s + status + artifact_keys.

    A phase is "auto-skippable" only when the caller passes
    `supports_auto_skip=True`. Phases that don't yet know how to persist
    + reload their outputs must pass False (the default) so a stale
    marker from a prior run doesn't trick the pipeline into skipping a
    phase whose output isn't actually on S3. Artifact-persistence PRs
    will flip each phase's flag to True as they land.

    The registry is designed to be cheap: marker reads are cached per
    phase name, so a phase whose marker is queried during `should_run`
    doesn't re-read S3 when the context manager enters.
    """

    def __init__(
        self,
        *,
        date: str,
        bucket: str,
        skip_phases: Iterable[str] | None = None,
        only_phases: Iterable[str] | None = None,
        force: bool = False,
        force_phases: Iterable[str] | None = None,
        hard_caps: dict[str, float] | None = None,
        s3_client=None,
    ):
        self.date = date
        self.bucket = bucket
        self._explicit_skip = set(skip_phases or [])
        self._only = set(only_phases) if only_phases else None
        self._force_all = bool(force)
        self._force_phases = set(force_phases or [])
        # Per-phase hard caps (seconds). A phase exceeding its cap trips
        # the watchdog: stack traces dumped, PhaseTimeoutError raised.
        # No cap → no watchdog for that phase (behavior identical to
        # pre-watchdog code).
        self._hard_caps = dict(hard_caps or {})
        self._markers: dict[str, dict | None] = {}
        self._s3 = s3_client  # lazy-init if None
        # Names of phases that wrote a marker with status=error during
        # THIS invocation. Used by the smoke-harness budget check to
        # catch false-PASS where the outer phase swallowed an inner
        # error and the wall-clock still looked healthy. See 2026-04-23
        # post-filter run where smoke-param-sweep "passed" at 96s < 500s
        # but param_sweep itself errored with recursion depth exceeded.
        self.phase_errors: list[str] = []

    # ── S3 helpers ───────────────────────────────────────────────────────

    def _client(self):
        if self._s3 is None:
            self._s3 = boto3.client("s3")
        return self._s3

    @property
    def s3_client(self):
        """Public accessor so artifact save/load helpers can use the same
        client the registry writes markers with. Keeps test fakes and
        production clients aligned without global monkey-patching."""
        return self._client()

    def _read_marker(self, phase_name: str) -> dict | None:
        """Return the marker dict for (date, phase), or None if absent/corrupt.

        Result is cached — repeated calls during the same run don't re-hit S3.
        A corrupt marker (unparseable JSON, missing required fields) is
        treated as absent and logged loud so operators can investigate.
        """
        if phase_name in self._markers:
            return self._markers[phase_name]

        key = _marker_key(self.date, phase_name)
        try:
            obj = self._client().get_object(Bucket=self.bucket, Key=key)
            body = obj["Body"].read()
            try:
                marker = json.loads(body)
            except (json.JSONDecodeError, ValueError):
                marker = None
            if not isinstance(marker, dict) or marker.get("status") not in ("ok", "error"):
                logger.warning(
                    "phase_registry: marker at s3://%s/%s malformed — ignoring "
                    "and recomputing phase %s. Body: %s",
                    self.bucket, key, phase_name, body[:200],
                )
                marker = None
            self._markers[phase_name] = marker
            return marker
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404"):
                self._markers[phase_name] = None
                return None
            # Network / permission errors: fail loud rather than silently
            # "marker absent → recompute." A transient S3 blip shouldn't
            # cause a 2h pipeline to silently redo work it already did.
            raise

    def _write_marker(self, marker: dict) -> None:
        key = _marker_key(self.date, marker["phase"])
        self._client().put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(marker, indent=2).encode(),
            ContentType="application/json",
        )
        # Keep cache consistent
        self._markers[marker["phase"]] = marker
        # Track in-invocation error markers so smoke budget check can
        # fail on swallowed inner errors. Only called from inside our
        # contextmanager, so we see status before callers' try/except.
        if marker.get("status") == "error":
            self.phase_errors.append(marker["phase"])

    # ── Decision logic ───────────────────────────────────────────────────

    def should_run(self, phase_name: str, supports_auto_skip: bool = False) -> tuple[bool, str]:
        """Return (run: bool, reason: str).

        Order of precedence:
          1. --only-phases restricts to the set (all others skipped).
          2. --skip-phases / --force-phases take precedence (explicit wins).
          3. --force overrides any auto-skip.
          4. Auto-skip if phase is auto-skippable AND a prior-run marker
             is present with status=ok.
          5. Default: run.

        Reason strings are structured so downstream INFO logs are grep-able:
          "only_phases_filter" | "explicit_skip" | "auto_skip_marker_ok"
          | "force_rerun" | "force_phase_rerun" | "default_run" | "not_auto_skippable"
        """
        if self._only is not None and phase_name not in self._only:
            return False, "only_phases_filter"
        if phase_name in self._explicit_skip:
            return False, "explicit_skip"
        if self._force_all:
            return True, "force_rerun"
        if phase_name in self._force_phases:
            return True, "force_phase_rerun"
        if not supports_auto_skip:
            return True, "not_auto_skippable"
        marker = self._read_marker(phase_name)
        if marker is not None and marker.get("status") == "ok":
            return False, "auto_skip_marker_ok"
        return True, "default_run"

    def load_marker(self, phase_name: str) -> dict | None:
        """Public accessor for a phase's marker — used by loaders in later PRs."""
        return self._read_marker(phase_name)

    # ── Phase context manager ────────────────────────────────────────────

    @contextmanager
    def phase(self, name: str, *, supports_auto_skip: bool = False, **log_ctx):
        """Phase context manager — writes a START/END marker to S3 around the block.

        Yields a `_PhaseContext` the caller can inspect:
          - `ctx.skipped`: True if the phase should not run (caller loads
            its artifact instead of recomputing).
          - `ctx.record_artifact(s3_key)`: call before exiting to attach
            an artifact key to the END marker.

        If `ctx.skipped`, the body still executes — the caller is
        expected to check `ctx.skipped` at the top of the block and load
        from S3 via `load_marker(name)["artifact_keys"]` rather than
        recomputing. This lets the skip decision live with the compute
        code, so a reader of the call site can see both paths.
        """
        run, reason = self.should_run(name, supports_auto_skip=supports_auto_skip)
        plog = _phase_logger()
        kv = " ".join(f"{k}={v}" for k, v in log_ctx.items())

        ctx = _PhaseContext(name=name, skipped=not run, skip_reason=reason)

        if not run:
            plog.info("PHASE_SKIP name=%s reason=%s %s", name, reason, kv)
            sys.stdout.flush()
            yield ctx
            return

        started_at = _now_iso()
        cap_s = self._hard_caps.get(name)
        if cap_s is not None:
            plog.info("PHASE_START name=%s hard_cap_s=%.1f %s", name, cap_s, kv)
        else:
            plog.info("PHASE_START name=%s %s", name, kv)
        sys.stdout.flush()
        t0 = time.monotonic()
        status = "ok"
        err_msg: str | None = None
        watchdog_timer: threading.Timer | None = None
        watchdog_state: dict | None = None
        if cap_s is not None and cap_s > 0:
            watchdog_timer, watchdog_state = _start_watchdog(name, cap_s)
        try:
            yield ctx
        except BaseException as exc:
            status = "error"
            # If the watchdog tripped, surface a PhaseTimeoutError from
            # the KeyboardInterrupt _thread.interrupt_main raises — the
            # caller's except handler reads the more descriptive type.
            if (
                watchdog_state is not None
                and watchdog_state.get("tripped")
                and isinstance(exc, KeyboardInterrupt)
            ):
                err_msg = (
                    f"PhaseTimeoutError: phase {name!r} exceeded hard cap "
                    f"{cap_s:.1f}s (see PHASE_TIMEOUT + faulthandler dump on stderr)"
                )
                raise PhaseTimeoutError(err_msg) from exc
            err_msg = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if watchdog_timer is not None:
                watchdog_timer.cancel()
            dur = time.monotonic() - t0
            completed_at = _now_iso()
            plog.info(
                "PHASE_END name=%s duration_s=%.2f status=%s %s",
                name, dur, status, kv,
            )
            sys.stdout.flush()
            # Best-effort marker write. A marker write failure should NOT
            # fail the whole pipeline — the phase already did its work.
            # But we log loud so silent marker-write drift doesn't build
            # up across runs.
            try:
                self._write_marker({
                    "schema_version": _MARKER_SCHEMA_VERSION,
                    "phase": name,
                    "date": self.date,
                    "status": status,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "duration_s": round(dur, 2),
                    "artifact_keys": sorted(ctx._artifact_keys),
                    "error": err_msg,
                })
            except Exception as marker_exc:
                logger.warning(
                    "phase_registry: failed to write marker for phase %s: %s. "
                    "Phase compute succeeded; future runs will not see this "
                    "completion and will re-run the phase.",
                    name, marker_exc,
                )


class _PhaseContext:
    """Yielded by PhaseRegistry.phase() so callers can query skip state
    and register artifact keys before the phase ends."""

    def __init__(self, *, name: str, skipped: bool, skip_reason: str):
        self.name = name
        self.skipped = skipped
        self.skip_reason = skip_reason
        self._artifact_keys: set[str] = set()

    def record_artifact(self, s3_key: str) -> None:
        """Attach an S3 key to the phase's END marker (recorded on exit).

        Called by phases that persist artifacts so the marker stores a
        durable pointer to what was produced. Downstream phases / loaders
        read `load_marker(name)["artifact_keys"]` to find the outputs.
        """
        if not isinstance(s3_key, str) or not s3_key:
            raise ValueError(f"record_artifact: expected non-empty str, got {s3_key!r}")
        self._artifact_keys.add(s3_key)


def load_phase_hard_caps(
    path: str | Path = "timing_budget.yaml",
) -> dict[str, float]:
    """Load per-phase hard caps from the `full_run_hard_caps_seconds`
    block of timing_budget.yaml. Returns empty dict if file or block
    absent (watchdog stays off — no behavior change).

    Keyed by phase name (e.g. ``phase4a_ensemble_modes``). Values are
    floats interpreted as seconds. Missing caps leave that phase
    unwatchdogged, which is the right default for phases whose typical
    runtime we haven't measured yet."""
    p = Path(path)
    if not p.is_absolute():
        # Resolve relative to repo root (where timing_budget.yaml lives)
        p = Path(__file__).parent / p
    if not p.exists():
        logger.info("timing_budget.yaml not found at %s — no phase watchdogs", p)
        return {}
    try:
        with open(p) as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning(
            "timing_budget.yaml at %s failed to parse: %s — no phase watchdogs",
            p, exc,
        )
        return {}
    caps = data.get("full_run_hard_caps_seconds") or {}
    if not isinstance(caps, dict):
        logger.warning(
            "timing_budget.yaml: full_run_hard_caps_seconds is not a dict (got %s) — "
            "no phase watchdogs", type(caps).__name__,
        )
        return {}
    # Coerce to float; drop non-numeric entries with a loud log.
    out: dict[str, float] = {}
    for name, cap in caps.items():
        try:
            out[str(name)] = float(cap)
        except (TypeError, ValueError):
            logger.warning(
                "timing_budget.yaml: phase %r has non-numeric cap %r — skipping",
                name, cap,
            )
    return out


# ── Config ────────────────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    search_paths = [
        Path.home() / "alpha-engine-config" / "backtester" / "config.yaml",
        Path(__file__).parent.parent / "alpha-engine-config" / "backtester" / "config.yaml",
        Path(path),
    ]
    resolved = next((p for p in search_paths if p.exists()), None)
    if resolved is None:
        raise FileNotFoundError(f"Config not found. Searched: {[str(p) for p in search_paths]}")
    with open(resolved) as f:
        config = yaml.safe_load(f)
    _validate_config(config, str(resolved))
    return config


def _validate_config(config: dict, path: str) -> None:
    """Validate required config keys exist and warn about common issues."""
    warnings = []
    errors = []

    if not config.get("signals_bucket"):
        errors.append("signals_bucket is required")

    executor_paths = config.get("executor_paths", [])
    if isinstance(executor_paths, str):
        executor_paths = [executor_paths]
    if not executor_paths:
        warnings.append("executor_paths not set — simulate/param-sweep modes will fail")
    elif not any(os.path.isdir(p) for p in executor_paths):
        warnings.append(
            f"No executor_paths found on disk: {executor_paths}. "
            "simulate/param-sweep modes will fail."
        )

    if not config.get("email_sender") or not config.get("email_recipients"):
        warnings.append("email_sender/email_recipients not set — email reports will be skipped")

    for w in warnings:
        logger.warning("Config (%s): %s", path, w)
    if errors:
        msg = f"Config validation failed ({path}): " + "; ".join(errors)
        raise ValueError(msg)


# ── Research DB ───────────────────────────────────────────────────────────────


def pull_research_db(bucket: str, local_path: str, s3_key: str = "research.db") -> bool:
    """Pull research.db from S3 to local_path. Returns True on success."""
    s3 = boto3.client("s3")
    try:
        s3.download_file(bucket, s3_key, local_path)
        size = os.path.getsize(local_path)
        logger.info("Pulled research.db from s3://%s/%s (%s bytes)", bucket, s3_key, f"{size:,}")
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            logger.warning("research.db not found in S3 — signal quality analysis will be skipped")
        else:
            logger.error("Failed to pull research.db: %s", e)
        return False


def init_research_db(db_arg: str | None, config: dict) -> None:
    """Pull or set research_db in config. Mutates config in place."""
    if db_arg:
        config["research_db"] = db_arg
        logger.info("Using local research.db: %s", db_arg)
    else:
        tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp_db.close()
        bucket = config.get("signals_bucket", "alpha-engine-research")
        db_pulled = pull_research_db(bucket, tmp_db.name)
        if db_pulled:
            config["research_db"] = tmp_db.name
        else:
            config["research_db"] = None
        config["_db_pull_status"] = "ok" if db_pulled else "failed"


# ── Trades DB ─────────────────────────────────────────────────────────────────


def find_trades_db(config: dict) -> str | None:
    """Find trades.db from executor_paths config."""
    executor_paths = config.get("executor_paths", [])
    if isinstance(executor_paths, str):
        executor_paths = [executor_paths]
    for p in executor_paths:
        db_path = Path(p) / "trades.db"
        if db_path.exists():
            return str(db_path)
    return None


# ── Predictor metrics (evaluation output) ─────────────────────────────────────


def push_predictor_rolling_metrics(config: dict, db_path: str) -> None:
    """Compute 30-day rolling hit rate and IC, merge into predictor/metrics/latest.json."""
    import sqlite3 as _sqlite3
    from datetime import datetime, timedelta

    bucket = config.get("signals_bucket")
    metrics_key = "predictor/metrics/latest.json"
    if not bucket or not db_path or not os.path.exists(db_path):
        return

    try:
        cutoff = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
        conn = _sqlite3.connect(db_path)
        df = pd.read_sql_query(
            "SELECT *, "
            f"{ALPHA_COALESCE_SQL} AS canonical_actual, "
            f"{CORRECT_COALESCE_SQL} AS canonical_correct, "
            f"{HORIZON_COALESCE_SQL} AS canonical_horizon "
            f"FROM predictor_outcomes WHERE {OUTCOMES_GRADED_SQL} "
            "AND prediction_date >= ?",
            conn,
            params=(cutoff,),
        )
        conn.close()
    except (_sqlite3.Error, FileNotFoundError, KeyError) as e:
        logger.warning("push_predictor_rolling_metrics: DB read failed: %s", e)
        return

    if len(df) < 5:
        logger.info("push_predictor_rolling_metrics: < 5 resolved outcomes, skipping S3 update")
        return

    hit_rate = float(pd.to_numeric(df["canonical_correct"], errors="coerce").mean())

    df["net_signal"] = (
        pd.to_numeric(df["p_up"], errors="coerce").fillna(0)
        - pd.to_numeric(df["p_down"], errors="coerce").fillna(0)
    )
    df["actual"] = pd.to_numeric(df["canonical_actual"], errors="coerce")
    valid = df.dropna(subset=["net_signal", "actual"])
    ic_30d = None
    ic_ir_30d = None
    if len(valid) >= _MIN_IC_SAMPLES:
        from scipy.stats import pearsonr
        import numpy as np
        ic_val, _ = pearsonr(valid["net_signal"], valid["actual"])
        ic_30d = round(float(ic_val), 4)
        n_chunks = max(2, len(valid) // 5)
        chunk_size = len(valid) // n_chunks
        chunk_ics = np.array([
            pearsonr(
                valid["net_signal"].iloc[i * chunk_size:(i + 1) * chunk_size],
                valid["actual"].iloc[i * chunk_size:(i + 1) * chunk_size],
            )[0]
            for i in range(n_chunks)
        ])
        ic_ir_30d = round(float(chunk_ics.mean() / (chunk_ics.std() + _IC_STD_EPSILON)), 3)

    s3 = boto3.client("s3")
    existing: dict = {}
    try:
        resp = s3.get_object(Bucket=bucket, Key=metrics_key)
        existing = json.loads(resp["Body"].read())
    except s3.exceptions.NoSuchKey:
        # Expected on first run — metrics file doesn't exist yet.
        logger.info("%s not found in S3 — initializing new metrics file", metrics_key)
    except Exception as e:
        # Non-NoSuchKey errors (S3 permissions, network, parse errors) mean
        # we might be overwriting valid existing metrics with a partial set,
        # or the entire metrics pipeline is broken. Raise so flow-doctor
        # captures it and downstream rolling-window updates don't silently
        # corrupt the metrics history.
        logger.error(
            "Failed to read existing predictor metrics from s3://%s/%s: %s",
            bucket, metrics_key, e, exc_info=True,
        )
        raise

    from datetime import datetime
    existing["hit_rate_30d_rolling"] = round(hit_rate, 4)
    existing["ic_30d"] = ic_30d
    existing["ic_ir_30d"] = ic_ir_30d
    existing["rolling_metrics_updated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    existing["rolling_n"] = len(df)

    try:
        s3.put_object(
            Bucket=bucket,
            Key=metrics_key,
            Body=json.dumps(existing, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(
            "Predictor rolling metrics updated: hit_rate=%.3f  ic_30d=%s  n=%d",
            hit_rate, ic_30d, len(df),
        )
    except Exception as e:
        # Write failure means the rolling metrics never get persisted — next
        # run reads stale values and the retrain alert evaluator bases its
        # decision on week-old IC / hit-rate. Raise so flow-doctor captures
        # it; previously this was a silent warning that kept the pipeline
        # green even when metrics went stale for weeks.
        logger.error(
            "push_predictor_rolling_metrics: S3 write failed for s3://%s/%s: %s",
            bucket, metrics_key, e, exc_info=True,
        )
        raise


# ── Sector map ────────────────────────────────────────────────────────────────


def load_sector_map(config: dict) -> dict[str, str] | None:
    """Load sector_map.json from predictor repo or S3."""
    predictor_paths = config.get("predictor_paths", [])
    if isinstance(predictor_paths, str):
        predictor_paths = [predictor_paths]
    for p in predictor_paths:
        map_path = Path(p) / "data" / "cache" / "sector_map.json"
        if map_path.exists():
            with open(map_path) as f:
                return json.load(f)

    try:
        s3 = boto3.client("s3")
        bucket = config.get("signals_bucket", "alpha-engine-research")
        resp = s3.get_object(
            Bucket=bucket, Key="predictor/price_cache/sector_map.json"
        )
        return json.load(resp["Body"])
    except Exception as e:
        logger.warning("Could not load sector_map.json: %s", e)
        return None
