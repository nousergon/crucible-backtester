"""Within-run checkpoint/resume for the long ``simulate`` phase (L4471 L2).

The ``simulate`` phase replays per-date and can run tens of minutes; a failure
anywhere inside it re-pays the whole phase (the existing ``PhaseRegistry``
markers are phase-level, not intra-phase). This module persists the simulate
state after every N dates so a re-run for the SAME run_date resumes from the
last checkpoint instead of restarting.

SOTA correctness anchors (Spark/Flink/ML-training checkpointing canon):
  * **Determinism** — a resumed run must produce byte-identical results to a
    from-scratch run. Guaranteed by checkpointing the FULL carried-forward
    state (sim_client cash/positions/peak_nav, accumulated orders, per-reason
    counters) and pickling (preserves exact types incl. dates) rather than a
    lossy JSON round-trip.
  * **Invalidation on input change** — a checkpoint is valid ONLY if the inputs
    that produced it are unchanged. ``compute_fingerprint`` hashes the
    sim-relevant config (executor params + init_cash) + the date list + a
    ``SIM_CODE_VERSION`` that MUST be bumped whenever the per-date sim logic
    changes. A fingerprint mismatch discards the checkpoint and recomputes
    cold — the load-bearing guard against silently resuming on stale inputs.

SCOPE: within-run only (gated to the standalone ``run_simulate`` simulate
phase — NOT the param-sweep per-combo calls). The checkpoint is CLEARED on
successful completion, so a fresh run for the same date recomputes rather than
resuming a finished run. Cross-run incremental resume (O(t)→O(Δt)) is L3
(deferred, P1) and needs a stricter invalidation cadence vs the weekly
``executor_params`` rewrite.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import pickle
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Bump whenever the per-date simulate logic changes in a way that would make an
# old checkpoint's carried-forward state incompatible. This is part of the
# fingerprint, so a bump invalidates every in-flight checkpoint (forces cold).
SIM_CODE_VERSION = "1"
_SCHEMA_VERSION = 1

# Config keys that materially affect the per-date sim (so a change must
# invalidate a checkpoint). Mirrors the executor params the sim consumes +
# the seed/cash. Intentionally explicit (not "hash the whole config") so
# unrelated config noise doesn't force needless cold rebuilds.
_FINGERPRINT_CONFIG_KEYS = (
    "init_cash",
    "atr_multiplier",
    "time_decay_reduce_days",
    "time_decay_exit_days",
    "min_score",
    "max_position_pct",
    "max_sector_pct",
    "max_equity_pct",
    "profit_take_pct",
)


def compute_fingerprint(config: dict, sim_dates: list[str]) -> str:
    """Stable hash of the inputs that determine the simulate result.

    A resumed checkpoint is only valid if this matches — so it must cover
    everything that changes the per-date output: the sim-relevant config
    params, the exact date list, and the sim code version.
    """
    cfg_subset = {k: config.get(k) for k in _FINGERPRINT_CONFIG_KEYS}
    payload = {
        "code_version": SIM_CODE_VERSION,
        "schema": _SCHEMA_VERSION,
        "config": cfg_subset,
        "dates": list(sim_dates),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _checkpoint_key(run_date: str) -> str:
    return f"backtest/{run_date}/_sim_checkpoint/checkpoint.pkl.gz"


def save_checkpoint(
    *,
    bucket: str,
    run_date: str,
    fingerprint: str,
    idx: int,
    last_date: str,
    sim_state: dict,
    all_orders: list,
    dates_simulated: int,
    skip_reasons: dict,
    rejected_ticker_counter: dict,
    s3_client: Any,
) -> None:
    """Persist the carried-forward simulate state. Best-effort: a checkpoint
    write failure must NOT abort the sim (we log + continue; worst case the
    next failure re-pays from the prior checkpoint or from scratch)."""
    payload = {
        "schema": _SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "idx": idx,
        "last_date": last_date,
        "sim_state": sim_state,  # {cash, positions, peak_nav}
        "all_orders": all_orders,
        "dates_simulated": dates_simulated,
        "skip_reasons": dict(skip_reasons),
        "rejected_ticker_counter": dict(rejected_ticker_counter),
    }
    try:
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            gz.write(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        s3_client.put_object(
            Bucket=bucket, Key=_checkpoint_key(run_date), Body=buf.getvalue()
        )
        logger.info(
            "[sim_checkpoint] saved at date %s (idx %d, %d orders)",
            last_date, idx, len(all_orders),
        )
    except Exception as exc:  # best-effort — never abort the sim on a write fail
        logger.warning("[sim_checkpoint] save failed (non-fatal): %s", exc)


def load_checkpoint(
    *, bucket: str, run_date: str, fingerprint: str, s3_client: Any
) -> Optional[dict]:
    """Return the checkpoint payload IFF one exists AND its fingerprint +
    schema match (valid resume). Returns None otherwise — including on a
    fingerprint mismatch (stale inputs → recompute cold, logged loudly)."""
    key = _checkpoint_key(run_date)
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        raw = obj["Body"].read()
    except Exception:
        return None  # absent (NoSuchKey) or unreadable → cold start
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as gz:
            payload = pickle.loads(gz.read())
    except Exception as exc:
        logger.warning("[sim_checkpoint] unreadable/corrupt (%s) — cold start", exc)
        return None
    if payload.get("schema") != _SCHEMA_VERSION:
        logger.warning(
            "[sim_checkpoint] schema %s != %s — discarding, cold start",
            payload.get("schema"), _SCHEMA_VERSION,
        )
        return None
    if payload.get("fingerprint") != fingerprint:
        # The load-bearing invalidation guard: inputs changed since the
        # checkpoint was written (executor params / dates / sim code).
        logger.warning(
            "[sim_checkpoint] fingerprint mismatch (inputs changed) — "
            "discarding stale checkpoint, recomputing cold",
        )
        return None
    logger.info(
        "[sim_checkpoint] valid checkpoint found at date %s (idx %d) — resuming",
        payload.get("last_date"), payload.get("idx"),
    )
    return payload


def clear_checkpoint(*, bucket: str, run_date: str, s3_client: Any) -> None:
    """Delete the checkpoint on successful completion so a fresh run for the
    same date recomputes (within-run resume is for FAILED runs only; cross-run
    incremental resume is L3)."""
    try:
        s3_client.delete_object(Bucket=bucket, Key=_checkpoint_key(run_date))
        logger.info("[sim_checkpoint] cleared on success")
    except Exception as exc:
        logger.warning("[sim_checkpoint] clear failed (non-fatal): %s", exc)
