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


def resolve_trading_day(date_str: str | None = None) -> str:
    """Normalize a date to the most recent NYSE trading day on or before it.

    DATE_CONVENTIONS: every trade artifact keys by the TRADING DAY, not the
    calendar date. The Saturday SF threads a CALENDAR run_date
    (``date(Execution.StartTime)`` — e.g. Sat 2026-05-30) but Research +
    signals.json + the standalone scanner key by trading day (Fri 2026-05-29).
    Keying backtester artifacts (``backtest/{date}/``, incl. pit_parity.json +
    parity_metrics) by the calendar date misaligns them with signals/{trading_day}
    and the ARTIFACT_REGISTRY trading-day axis (the research↔backtester
    pit-parity drift surfaced this, L4466). Mirrors the scanner fix (research #257).

    Idempotent: a trading-day input returns unchanged (so re-normalizing the
    bash-normalized RUN_DATE is a no-op). Default = today UTC. Defensive: on any
    lib/parse failure, return the input unchanged with a WARNING — a date
    normalization miss must not abort the whole backtester.
    """
    import datetime as _dt

    raw = date_str or _dt.date.today().isoformat()
    try:
        from alpha_engine_lib import trading_calendar as _tc

        d = _dt.date.fromisoformat(raw[:10])
        td = d if _tc.is_trading_day(d) else _tc.previous_trading_day(d)
        return td.isoformat()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "resolve_trading_day(%r) failed (%s) — using input unchanged", raw, exc
        )
        return raw


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


# ── Phase outcome taxonomy (3-way: SUCCESS | EMPTY | FAILURE) ────────────────
#
# The durable encoding of ARCHITECTURE.md §22 ("The backtester delivers config
# ACTIONS and a report card; a 0-result is legitimate only when the inputs were
# validated first"). Every backtester stage resolves to exactly one of three
# outcomes — binary ok/fail is WRONG because it forces the EMPTY case into one
# of two harmful answers (treat-as-success silently drops the config action and
# mis-tunes the live book; treat-as-failure kills the whole Saturday SF when a
# risk/score gate merely did its job — the 2026-06-06 symptom, L4506–L4521).
#
#   SUCCESS — produced its declared admissible result.
#   EMPTY   — ran correctly, produced NO admissible result (e.g. all combos
#             gated out). A first-class analytical FINDING, not an error:
#             downstream no-ops gracefully (configs held), but it is surfaced
#             LOUDLY (WARN + alert) so a *suspicious* degeneracy (e.g. a
#             zero-score-input feed defect) is never silent. The input-quality
#             HARD gate (L4525) is what later promotes garbage-input-EMPTY to
#             FAILURE; until it lands, EMPTY is "valid-but-alarmed".
#   FAILURE — an infra/contract break (absent input, exception, contract
#             violation). Fatal, fail-loud.
#
# This is the analytical-outcome sibling of §12's deliverable-presence
# three-state (present-valid / present-invalid / absent). PhaseOutcome is the
# structured record (P8 observability) that L4524 (artifact-validated
# checkpoints) and L4526 (pipeline manifest) build on.


import enum
from dataclasses import dataclass, field


class PhaseStatus(enum.Enum):
    """3-way phase outcome. See module section header + ARCHITECTURE.md §22."""

    SUCCESS = "success"
    EMPTY = "empty"
    FAILURE = "failure"


@dataclass
class PhaseOutcome:
    """Structured result of running (or classifying) a pipeline stage.

    ``status`` is the load-bearing field; the rest are observability so a
    degenerate week is legible at a glance instead of a forensic dig (plan
    §2 P8). ``reason`` is the human/log message; ``degeneracy_reason`` names
    *why* an EMPTY produced nothing; ``n_inputs``/``n_admissible`` quantify
    the gating. ``artifacts_written`` records what hit S3 (foundation for the
    L4524 artifact-validated marker).
    """

    status: PhaseStatus
    phase: str
    reason: str = ""
    n_inputs: int | None = None
    n_admissible: int | None = None
    degeneracy_reason: str | None = None
    artifacts_written: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status is PhaseStatus.SUCCESS

    @property
    def is_empty(self) -> bool:
        return self.status is PhaseStatus.EMPTY

    @property
    def is_failure(self) -> bool:
        return self.status is PhaseStatus.FAILURE

    def to_dict(self) -> dict:
        """JSON-serializable record for structured logging / markers."""
        return {
            "status": self.status.value,
            "phase": self.phase,
            "reason": self.reason,
            "n_inputs": self.n_inputs,
            "n_admissible": self.n_admissible,
            "degeneracy_reason": self.degeneracy_reason,
            "artifacts_written": list(self.artifacts_written),
            "detail": dict(self.detail),
        }


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


# Per-run substrate-operations aggregate. Reaches the report card's substrate
# tile (crucible-evaluator grading/tiles/substrate.py, config#1151) which grades
# `watchdog_firings` off `firing_count`. Top-level under backtest/{date}/ (NOT
# under .phases/) so it sits beside the other evaluator-read artifacts
# (sample_size.json, etc.). Schema is additive-only — future fields add, never
# rename/remove (S3 Contract Safety, CLAUDE.md).
_SUBSTRATE_OPS_SCHEMA_VERSION = 1


def _substrate_ops_key(date: str) -> str:
    return f"backtest/{date}/substrate_ops.json"


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
        # L4524: per-phase cache of artifact-validation results so a marker
        # whose declared artifacts were head_object'd once isn't re-checked
        # on the second should_run call (the phase() context manager re-asks).
        # Value is the first-missing artifact key, or None if all present.
        self._artifact_checks: dict[str, str | None] = {}
        self._s3 = s3_client  # lazy-init if None
        # Names of phases that wrote a marker with status=error during
        # THIS invocation. Used by the smoke-harness budget check to
        # catch false-PASS where the outer phase swallowed an inner
        # error and the wall-clock still looked healthy. See 2026-04-23
        # post-filter run where smoke-param-sweep "passed" at 96s < 500s
        # but param_sweep itself errored with recursion depth exceeded.
        self.phase_errors: list[str] = []
        # Per-run watchdog telemetry. One record per CAPPED phase that ran this
        # invocation: {phase, watchdog_fired, cap_s, wall_time_s}. Aggregated to
        # backtest/{date}/substrate_ops.json on every capped-phase exit so the
        # report card's substrate tile (config#1151) can read the per-run firing
        # count. A "firing" = the phase hit its hard cap (the silent-burn
        # tripwire), which the watchdog converts to a PhaseTimeoutError abort.
        self._watchdog_records: list[dict] = []

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

    def _write_substrate_ops(self) -> None:
        """Persist the per-run watchdog aggregate to backtest/{date}/substrate_ops.json.

        Re-written (idempotently overwritten) after each capped phase so a
        pipeline that aborts on a watchdog trip STILL leaves the firing count on
        S3 — the abort is the loud failure, the artifact is the receipt the
        report card reads. `firing_count` is the headline the evaluator's
        substrate tile grades (config#1151): 0 = no phase hit its hard cap
        (healthy); >0 = one or more phases burned through their silent-compute
        cap and were force-aborted (degradation).
        """
        firing_count = sum(1 for r in self._watchdog_records if r.get("watchdog_fired"))
        ops = {
            "schema_version": _SUBSTRATE_OPS_SCHEMA_VERSION,
            "date": self.date,
            "updated_at": _now_iso(),
            "watchdog": {
                "firing_count": firing_count,
                "capped_phases_run": len(self._watchdog_records),
                "per_phase": list(self._watchdog_records),
            },
        }
        self._client().put_object(
            Bucket=self.bucket,
            Key=_substrate_ops_key(self.date),
            Body=json.dumps(ops, indent=2).encode(),
            ContentType="application/json",
        )

    def _first_missing_artifact(self, artifact_keys: Iterable[str]) -> str | None:
        """Return the first declared artifact key that is absent on S3, or
        None if every key is present (or none were declared).

        L4524 — artifact-validated checkpoints. A `status=ok` marker only
        earns an auto-skip if the outputs it *claims* to have produced are
        actually on S3. A marker whose declared artifact has gone missing is
        LYING (the L4518/L4521 "trust-and-yield-empty" failure: a phase marked
        ok while its critical artifact was never written / was pruned),
        so the marker must be treated as INVALID → re-run rather than skipped.

        Existence is probed with `head_object` (metadata only — never download
        a multi-MB parquet just to confirm it exists). Error posture mirrors
        `_read_marker`: a 404/NotFound means absent (→ invalidate the marker);
        any other S3 error (network blip, permission) RAISES rather than
        silently flipping a skip/re-run decision on incomplete information,
        per the fail-loud rule.
        """
        client = self._client()
        for key in artifact_keys:
            if not key:
                continue
            try:
                client.head_object(Bucket=self.bucket, Key=key)
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code in ("NoSuchKey", "NoSuchBucket", "NotFound", "404"):
                    return key
                # Transient / permission error: fail loud, don't guess.
                raise
        return None

    def _marker_artifact_missing(self, phase_name: str, marker: dict) -> str | None:
        """Validate a phase's marker against its declared artifacts (cached).

        Returns the first-missing artifact key (marker invalid) or None
        (marker honored). Cached per phase so the phase() context manager's
        repeat should_run call doesn't re-issue the head_object probes.
        """
        if phase_name not in self._artifact_checks:
            self._artifact_checks[phase_name] = self._first_missing_artifact(
                marker.get("artifact_keys") or []
            )
        return self._artifact_checks[phase_name]

    # ── Decision logic ───────────────────────────────────────────────────

    def should_run(self, phase_name: str, supports_auto_skip: bool = False) -> tuple[bool, str]:
        """Return (run: bool, reason: str).

        Order of precedence:
          1. --only-phases restricts to the set (all others skipped).
          2. --skip-phases / --force-phases take precedence (explicit wins).
          3. --force overrides any auto-skip.
          4. Auto-skip if phase is auto-skippable AND a prior-run marker
             is present with status=ok AND every artifact the marker
             declares still exists on S3 (L4524 — a marker whose declared
             artifact has gone missing is invalid → re-run).
          5. Default: run.

        Reason strings are structured so downstream INFO logs are grep-able:
          "only_phases_filter" | "explicit_skip" | "auto_skip_marker_ok"
          | "force_rerun" | "force_phase_rerun" | "default_run"
          | "not_auto_skippable" | "marker_artifact_missing"
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
            # L4524: the marker says ok — but does what it claims to have
            # produced actually exist? A missing declared artifact means the
            # marker is lying; invalidate it and re-run rather than skip into
            # a downstream phase that will starve on the absent output.
            missing = self._marker_artifact_missing(phase_name, marker)
            if missing is not None:
                logger.warning(
                    "phase_registry: phase %s (date %s) has a status=ok marker but "
                    "its declared artifact s3://%s/%s is absent — marker INVALID, "
                    "re-running the phase (L4524 artifact-validated checkpoint).",
                    phase_name, self.date, self.bucket, missing,
                )
                return True, "marker_artifact_missing"
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
            # Watchdog telemetry — record only for CAPPED phases (an uncapped
            # phase has no watchdog, so "fired" is undefined for it). A
            # `watchdog_fired` here means the phase hit its hard cap and is being
            # force-aborted with a PhaseTimeoutError. Persist the aggregate
            # NOW (in the finally, before the PhaseTimeoutError propagates) so
            # the firing count survives the abort: fail-loud = count + persist,
            # THEN re-raise. A persist failure must NOT mask the abort, so it's
            # best-effort + loud-logged, never swallowing the timeout.
            if cap_s is not None and cap_s > 0:
                fired = bool(watchdog_state is not None and watchdog_state.get("tripped"))
                self._watchdog_records.append({
                    "phase": name,
                    "watchdog_fired": fired,
                    "cap_s": round(cap_s, 2),
                    "wall_time_s": round(dur, 2),
                })
                try:
                    self._write_substrate_ops()
                except Exception as ops_exc:
                    logger.warning(
                        "phase_registry: failed to write substrate_ops.json for "
                        "date %s after phase %s (watchdog_fired=%s): %s. The phase "
                        "abort/result is unaffected; the report card's "
                        "watchdog_firings may under-count this run.",
                        self.date, name, fired, ops_exc,
                    )
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
    # Experiment-package first (config#1042): backtester config.yaml resolves from
    # experiments/$ALPHA_ENGINE_EXPERIMENT_ID/backtester/config.yaml (default
    # experiment `reference`) ahead of the legacy top-level
    # alpha-engine-config/backtester/config.yaml, then the repo-local fallback.
    # Mirrors alpha-engine-research/config.py + alpha-engine-data weekly_collector.
    exp = os.environ.get("ALPHA_ENGINE_EXPERIMENT_ID", "reference")
    config_roots = [
        Path.home() / "alpha-engine-config",
        Path(__file__).parent.parent / "alpha-engine-config",
    ]
    search_paths = [r / "experiments" / exp / "backtester" / "config.yaml" for r in config_roots]
    search_paths += [r / "backtester" / "config.yaml" for r in config_roots]
    search_paths.append(Path(path))
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
    """Resolve trades.db: local ``executor_paths`` first, else pull from S3.

    On the trading box the DB is local; on the backtester spot (where grading
    runs) it is not — but ``eod_reconcile`` backs it up to
    ``s3://{signals_bucket}/trades/trades_latest.db`` after every close
    (``executor/trade_logger.py::backup_to_s3``). Pulling that lets the
    executor-tile analyses (trigger scorecard / shadow book / exit timing) grade
    real production trades instead of skipping. (Director plan Phase B1b.)

    Returns None if neither a local DB nor the S3 backup is available — the
    dependent analyses then surface ``N/A-MISSING-INPUT`` (never silently),
    logged at WARN here.
    """
    executor_paths = config.get("executor_paths", [])
    if isinstance(executor_paths, str):
        executor_paths = [executor_paths]
    for p in executor_paths:
        db_path = Path(p) / "trades.db"
        if db_path.exists():
            return str(db_path)

    # S3 fallback — the backtester spot has no local trades.db.
    bucket = config.get("signals_bucket")
    if not bucket:
        return None
    local = Path(tempfile.gettempdir()) / "ae_trades_latest.db"
    try:
        boto3.client("s3").download_file(bucket, "trades/trades_latest.db", str(local))
        logger.info(
            "Pulled trades_latest.db from s3://%s/trades/trades_latest.db", bucket
        )
        return str(local)
    except ClientError as e:
        logger.warning(
            "find_trades_db: no local trades.db and S3 pull of "
            "trades/trades_latest.db failed (%s) — executor tiles will surface "
            "N/A-MISSING-INPUT this cycle",
            e,
        )
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

    # Wave-3 reader migration (ROADMAP L1401): try the new
    # ``reference/price_cache/`` prefix first, fall back to legacy
    # ``predictor/price_cache/`` during the producer write-both soak
    # (PR1 alpha-engine-data#270 shipped 2026-05-19; soak ≥1 week to
    # ~2026-05-26). After Wave-3 PR4 retires legacy, the fallback
    # branch becomes dead and can be dropped.
    s3 = boto3.client("s3")
    bucket = config.get("signals_bucket", "alpha-engine-research")
    for key in (
        "reference/price_cache/sector_map.json",
        "predictor/price_cache/sector_map.json",
    ):
        try:
            resp = s3.get_object(Bucket=bucket, Key=key)
            return json.load(resp["Body"])
        except Exception as e:
            logger.debug(
                "[pipeline_common] sector_map.json miss at s3://%s/%s: %s",
                bucket, key, type(e).__name__,
            )
    logger.warning(
        "Could not load sector_map.json from either prefix "
        "(reference/ or predictor/)",
    )
    return None
