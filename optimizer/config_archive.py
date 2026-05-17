"""config_archive.py — point-in-time (bitemporal) resolution of optimizer
configs for walk-forward backtesting.

PR 2 of point-in-time discipline (ROADMAP L2371 / Backtester Phase 3;
plan ``alpha-engine-docs/private/pit-discipline-260515.md`` §D3).

The leak this closes (plan §2 audit): ``*_optimizer.read_current_params``
reads the *current live* ``config/{type}_params.json`` regardless of the
simulated decision date, so a backtest replaying date ``D`` evaluates its
sweep against configs that were not knowable until well after ``D`` —
look-ahead contamination feeding the autonomous optimizer.

**No parallel scheme (plan D3 explicit).** The bitemporal store already
exists: every ``optimizer.apply()`` writes a dated snapshot to
``config/{type}_params_history/{run_id}_*.json`` whose payload carries
``updated_at`` = the apply date = its *knowledge date*. This module adds
only the missing knowledge-time **index** (``config/CHANGELOG.json``) so a
resolver can answer "the latest snapshot whose knowledge time ≤ ``D``"
without scanning the whole history prefix, plus the resolver itself.

Invariants (plan §3):
  - knowledge-time ≤ decision-time (cardinal rule).
  - **No-future-fallback** (the central trap): no snapshot with knowledge
    ≤ ``D`` → the resolver returns ``None`` and the optimizer falls back to
    its *genesis* ``FACTORY_DEFAULTS`` (the documented shipped defaults
    ``read_current_params`` already uses on first run), **never** the
    current live config. Substituting a future config is exactly how
    look-ahead silently re-enters.

The changelog write is **best-effort + loud** (mirrors
``cost_report._emit_changelog_anomaly_entry``): ``apply()`` has already
written the live key + the forensic history snapshot by the time
:func:`record_apply` runs, so an index-write failure must not fail the
optimizer — it degrades PIT-resolution for that one apply, logged at
WARNING, recoverable by a backfill.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Single knowledge-time index over every optimizer apply. One small JSON
# (one entry per weekly apply per config_type — ~hundreds/year), read-
# modify-write under the single-writer Saturday-SF cadence.
CHANGELOG_KEY = "config/CHANGELOG.json"

# The config types with a dated history prefix + a FACTORY_DEFAULTS
# genesis. Matches optimizer.rollback.CONFIG_KEYS minus predictor_params
# (predictor veto threshold is tuned by veto_analysis, which has no
# read_current_params baseline-replay call site — not one of the plan's 5).
VALID_CONFIG_TYPES = ("executor_params", "research_params", "scanner_params")


def _client(s3_client):
    return s3_client if s3_client is not None else boto3.client("s3")


def _norm_date(d) -> _dt.date:
    if isinstance(d, _dt.date):
        return d
    # Tolerate "YYYY-MM-DD", "YYYY-MM-DDTHH:MM:SS", or a ".smoke/…" label.
    return _dt.date.fromisoformat(str(d)[:10])


def as_of_date_from_config(config: dict) -> _dt.date | None:
    """The decision date the PIT optimizer baseline should be resolved at.

    ``None`` when walk-forward is off → callers keep the unchanged
    :func:`read_current_params` path (so live Saturday runs and every
    non-PIT mode are byte-for-byte identical). When on, the as-of is the
    run-date label (``config['_run_date']`` — set by evaluate.py:934 and
    stamped in backtest.py main()), so a *backdated* replay reads configs
    knowable as of that date while a live run dated today still resolves
    to today's snapshot (== current). Defaults to today if unset.
    """
    if not config.get("walk_forward"):
        return None
    raw = config.get("_run_date") or _dt.date.today().isoformat()
    try:
        return _norm_date(raw)
    except ValueError:
        logger.warning(
            "[config_archive] uninterpretable _run_date %r — PIT as-of "
            "falling back to today", raw,
        )
        return _dt.date.today()


def read_params_pit_or_current(opt_module, bucket: str, config: dict) -> dict:
    """The one place the PIT-vs-current branch lives, so every call site is a
    one-liner and the flag semantics cannot drift between them.

    walk-forward OFF (default) → ``opt_module.read_current_params(bucket)``,
    byte-for-byte the legacy behavior. ON → ``read_params_as_of`` at the
    run-date as-of (genesis on no-future-fallback). ``opt_module`` is the
    optimizer module (executor_/research_/scanner_optimizer); both functions
    share an identical return contract per module so the caller is agnostic.
    """
    as_of = as_of_date_from_config(config)
    if as_of is None:
        return opt_module.read_current_params(bucket)
    return opt_module.read_params_as_of(bucket, as_of)


def _load_changelog(s3, bucket: str) -> list[dict]:
    """Return the changelog entry list ([] if absent or corrupt).

    Absent is normal (no apply has run since the index was introduced —
    every fold then resolves to genesis, the correct no-future-fallback
    behavior). Corrupt is logged loud but still degrades to genesis
    rather than crashing a backtest.
    """
    try:
        obj = s3.get_object(Bucket=bucket, Key=CHANGELOG_KEY)
        data = json.loads(obj["Body"].read())
        return data if isinstance(data, list) else []
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return []
        logger.warning(
            "[config_archive] cannot read s3://%s/%s (%s) — PIT resolution "
            "degrades to genesis defaults for this run", bucket, CHANGELOG_KEY, e,
        )
        return []
    except Exception as e:
        logger.warning(
            "[config_archive] corrupt %s (%s) — degrading to genesis",
            CHANGELOG_KEY, e,
        )
        return []


def record_apply(
    bucket: str,
    config_type: str,
    *,
    history_key: str,
    knowledge_date: str,
    run_id: str,
    effective_date: str | None = None,
    s3_client=None,
) -> bool:
    """Append one bitemporal entry to ``config/CHANGELOG.json``.

    Called by each optimizer's ``apply()`` right after it writes the
    forensic ``{type}_params_history`` snapshot. Best-effort: returns
    ``True`` on success, ``False`` (WARNING-logged, never raised) on any
    failure — the live config + history snapshot are already durable, so
    the optimizer must not fail on an index hiccup.

    ``knowledge_date`` is the apply date (``payload['updated_at']`` =
    ``str(date.today())``). ``effective_date`` defaults to it: an
    optimizer-applied param set takes effect the day it is written.
    """
    if config_type not in VALID_CONFIG_TYPES:
        logger.warning(
            "[config_archive] record_apply: unknown config_type %r — skipped",
            config_type,
        )
        return False
    try:
        s3 = _client(s3_client)
        entries = _load_changelog(s3, bucket)
        entries.append({
            "config_type": config_type,
            "knowledge_date": _norm_date(knowledge_date).isoformat(),
            "effective_date": _norm_date(
                effective_date if effective_date else knowledge_date
            ).isoformat(),
            "history_key": history_key,
            "run_id": run_id,
            "recorded_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        })
        s3.put_object(
            Bucket=bucket, Key=CHANGELOG_KEY,
            Body=json.dumps(entries, indent=2),
            ContentType="application/json",
        )
        logger.info(
            "[config_archive] indexed %s apply (knowledge=%s) → s3://%s/%s",
            config_type, knowledge_date, bucket, CHANGELOG_KEY,
        )
        return True
    except Exception as e:
        logger.warning(
            "[config_archive] changelog index append failed for %s "
            "(best-effort, swallowed — live+history already durable): %s",
            config_type, e,
        )
        return False


def resolve_as_of(
    bucket: str,
    config_type: str,
    as_of_date,
    *,
    s3_client=None,
) -> dict | None:
    """Resolve the point-in-time config payload for ``config_type``.

    Returns the parsed payload of the snapshot with the greatest
    ``knowledge_date`` such that ``knowledge_date ≤ as_of_date`` (tie-break
    by ``run_id`` descending — later-in-the-day apply wins). Returns
    ``None`` when **no** eligible snapshot exists: the caller must then use
    its genesis ``FACTORY_DEFAULTS`` — never a later snapshot
    (no-future-fallback, plan invariant 3).
    """
    as_of = _norm_date(as_of_date)
    s3 = _client(s3_client)
    eligible = [
        e for e in _load_changelog(s3, bucket)
        if e.get("config_type") == config_type
        and _safe_kd(e) is not None
        and _safe_kd(e) <= as_of
    ]
    if not eligible:
        logger.info(
            "[config_archive] no %s snapshot with knowledge ≤ %s "
            "(no-future-fallback → caller uses genesis defaults)",
            config_type, as_of.isoformat(),
        )
        return None
    chosen = max(eligible, key=lambda e: (_safe_kd(e), e.get("run_id", "")))
    key = chosen["history_key"]
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        payload = json.loads(obj["Body"].read())
        logger.info(
            "[config_archive] PIT %s @ %s → snapshot knowledge=%s s3://%s/%s",
            config_type, as_of.isoformat(),
            chosen.get("knowledge_date"), bucket, key,
        )
        return payload
    except Exception as e:
        # The index pointed at a snapshot we cannot read. This is NOT a
        # cue to fall forward to a newer snapshot (that re-introduces
        # look-ahead). Fail loud → caller uses genesis, same as "no
        # eligible snapshot".
        logger.warning(
            "[config_archive] %s snapshot s3://%s/%s unreadable (%s) — "
            "treating as no-eligible-snapshot (genesis, NOT future-fallback)",
            config_type, bucket, key, e,
        )
        return None


def _safe_kd(entry: dict) -> _dt.date | None:
    try:
        return _dt.date.fromisoformat(entry["knowledge_date"])
    except (KeyError, ValueError, TypeError):
        return None
