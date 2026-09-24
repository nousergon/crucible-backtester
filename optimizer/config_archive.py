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

**The index is not the only source (alpha-engine-config-I11503).** Since
the assembler became the sole writer of live config
(``optimizer/assembler.py::_cutover_apply``), applies write the live key and
``config/{type}_history/{YYMMDDHHMM}.json`` but never call
:func:`record_apply` — the changelog held zero ``executor_params`` entries
while the history prefix held two dozen, and every walk-forward backtest
silently simulated genesis defaults (min_score 70 / max_position_pct 5%)
against a live book running 75 / 10%. The resolver therefore reads the
snapshots that are actually on S3 as well as the index, and a genesis
fallback is recorded so :func:`read_params_pit_or_current` can make the
stage DEGRADED instead of logging an INFO line.

The changelog write is **best-effort + loud** (mirrors
``cost_report._emit_changelog_anomaly_entry``): ``apply()`` has already
written the live key + the forensic history snapshot by the time
:func:`record_apply` runs, so an index-write failure must not fail the
optimizer — it degrades PIT-resolution for that one apply, logged at
WARNING, recoverable by a backfill.
"""

from __future__ import annotations

import contextvars
import datetime as _dt
import json
import logging
import re

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

# Per-call record of what resolve_as_of found, collected by
# read_params_pit_or_current so a genesis fallback can be surfaced as a
# DEGRADED stage (config-I11503) without changing the return contract of
# the three optimizers' read_params_as_of. None = nobody is collecting.
_RESOLUTIONS: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "config_archive_resolutions", default=None,
)

# Where the resolved snapshots are, besides the changelog index. Both are
# written on every apply by the path that actually runs today — the
# assembler's cutover apply (optimizer/assembler.py::_cutover_apply) writes
# the live key + ``{type}_history/{YYMMDDHHMM}.json`` and never indexes the
# changelog, which is why the changelog had no executor_params entry at all
# and every walk-forward backtest resolved to genesis (config-I11503).
_LIVE_KEY = "config/{config_type}.json"
_HISTORY_PREFIX = "config/{config_type}_history/"
_RUN_ID_RE = re.compile(r"^(\d{2})(\d{2})(\d{2})\d{4}$")          # YYMMDDHHMM
_ISO_PREFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")             # YYYY-MM-DD…


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
    token = _RESOLUTIONS.set([])
    try:
        params = opt_module.read_params_as_of(bucket, as_of)
        outcomes = list(_RESOLUTIONS.get() or [])
    finally:
        _RESOLUTIONS.reset(token)
    for outcome in outcomes:
        if outcome.get("status") == "resolved":
            continue
        # A walk-forward run that fell back to genesis is simulating a
        # strategy the live system does not run. That is a DEGRADED stage,
        # never an INFO line (config-I11503): the fact rides on the config
        # dict to the stage's write_health(warnings=...), which derives
        # status "degraded" from it.
        msg = (
            f"{outcome.get('config_type')} @ {as_of.isoformat()}: no snapshot "
            f"with knowledge <= as-of ({outcome.get('reason')}) — simulated "
            "GENESIS defaults, not the params the live system ran "
            "(alpha-engine-config-I11503)"
        )
        warnings = config.setdefault("_pit_params_degraded", [])
        if msg not in warnings:
            warnings.append(msg)
        logger.warning("[config_archive] DEGRADED: %s", msg)
    return params


def pit_degraded_warnings(config: dict) -> list[str]:
    """Warnings a stage must pass to ``write_health(warnings=...)``: one per
    optimizer config a walk-forward read resolved to genesis defaults.
    Empty when every PIT read found a snapshot, or walk-forward is off."""
    return list(config.get("_pit_params_degraded") or [])


def _note(outcome: dict) -> None:
    collected = _RESOLUTIONS.get()
    if collected is not None:
        collected.append(outcome)


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


def _key_date(key: str) -> _dt.date | None:
    """The date a snapshot's key names: ``YYMMDDHHMM`` run ids (the lib
    v0.8.0 eval layout) or a ``YYYY-MM-DD`` prefix (the legacy dated layout,
    including suffixed operator snapshots). ``None`` if it names neither."""
    stem = key.rsplit("/", 1)[-1].removesuffix(".json")
    m = _RUN_ID_RE.match(stem)
    try:
        if m:
            return _dt.date(2000 + int(m.group(1)), int(m.group(2)), int(m.group(3)))
        m = _ISO_PREFIX_RE.match(stem)
        if m:
            return _dt.date.fromisoformat(m.group(1))
    except ValueError:
        return None
    return None


def _as_utc(ts) -> _dt.datetime | None:
    if not isinstance(ts, _dt.datetime):
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=_dt.timezone.utc)


_EPOCH = _dt.datetime.min.replace(tzinfo=_dt.timezone.utc)


def _history_candidates(s3, bucket: str, config_type: str) -> tuple[list[dict], str | None]:
    """Every snapshot on S3 for ``config_type``: the dated history objects and
    the live key itself, each with the knowledge date it can honestly claim.

    Knowledge date = the LATER of the date the key names and the date the
    object was last written. An object cannot have been known before it
    existed, so ``LastModified`` is a floor that no key name can undercut;
    a backfilled or re-copied object therefore only ever resolves LATER
    (towards genesis → DEGRADED), never earlier (look-ahead). The live key
    carries no date in its name and is eligible exactly when it was last
    written on or before the as-of.

    Returns ``(candidates, error)``; ``error`` is set when the listing
    failed, so the caller can say WHY nothing resolved.
    """
    live_key = _LIVE_KEY.format(config_type=config_type)
    history_prefix = _HISTORY_PREFIX.format(config_type=config_type)
    out: list[dict] = []
    token = None
    try:
        while True:
            kwargs = {"Bucket": bucket, "Prefix": f"config/{config_type}"}
            if token:
                kwargs["ContinuationToken"] = token
            resp = s3.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []) or []:
                key = obj.get("Key", "")
                is_live = key == live_key
                is_history = (
                    key.startswith(history_prefix)
                    and key.endswith(".json")
                    and key.rsplit("/", 1)[-1] != "latest.json"
                )
                if not (is_live or is_history):
                    continue
                written = _as_utc(obj.get("LastModified"))
                dates = [d for d in (
                    None if is_live else _key_date(key),
                    written.date() if written else None,
                ) if d is not None]
                if not dates:
                    continue
                out.append({
                    "history_key": key,
                    "knowledge_date": max(dates),
                    "stamp": written or _EPOCH,
                    "source": "live" if is_live else "history",
                })
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
            if not token:
                break
    except Exception as e:  # noqa: BLE001 — surfaced as the miss reason
        logger.warning(
            "[config_archive] cannot list s3://%s/config/%s* (%s) — PIT "
            "resolution falls back to the changelog index alone",
            bucket, config_type, e,
        )
        return out, f"history listing failed: {e}"
    return out, None


def resolve_as_of(
    bucket: str,
    config_type: str,
    as_of_date,
    *,
    s3_client=None,
) -> dict | None:
    """Resolve the point-in-time config payload for ``config_type``.

    Candidates are the union of the changelog index and the snapshots that
    are actually on S3 (``config/{type}_history/*.json`` + the live key —
    see :func:`_history_candidates`), de-duplicated by key with the LATER
    knowledge date winning. Returns the payload of the candidate with the
    greatest knowledge date ``≤ as_of_date`` (tie-break: latest write, then
    key). Returns ``None`` when **no** eligible snapshot exists: the caller
    must then use its genesis ``FACTORY_DEFAULTS`` — never a later snapshot
    (no-future-fallback, plan invariant 3). Every call records its outcome
    for :func:`read_params_pit_or_current`, which turns a ``None`` into a
    DEGRADED stage (config-I11503).
    """
    as_of = _norm_date(as_of_date)
    s3 = _client(s3_client)

    by_key: dict[str, dict] = {}
    for e in _load_changelog(s3, bucket):
        kd = _safe_kd(e)
        key = e.get("history_key")
        if e.get("config_type") != config_type or kd is None or not key:
            continue
        stamp = _EPOCH
        try:
            stamp = _as_utc(_dt.datetime.fromisoformat(e["recorded_at"])) or _EPOCH
        except (KeyError, TypeError, ValueError):
            pass
        by_key[key] = {"history_key": key, "knowledge_date": kd,
                       "stamp": stamp, "source": "changelog"}
    listed, list_error = _history_candidates(s3, bucket, config_type)
    for c in listed:
        prior = by_key.get(c["history_key"])
        if prior is not None:
            c = {**c, "knowledge_date": max(c["knowledge_date"], prior["knowledge_date"]),
                 "stamp": max(c["stamp"], prior["stamp"])}
        by_key[c["history_key"]] = c

    eligible = [c for c in by_key.values() if c["knowledge_date"] <= as_of]
    if not eligible:
        reason = (
            f"{len(by_key)} snapshot(s) known, none with knowledge <= "
            f"{as_of.isoformat()}"
        )
        if list_error:
            reason += f"; {list_error}"
        logger.warning(
            "[config_archive] no %s snapshot with knowledge ≤ %s "
            "(no-future-fallback → caller uses genesis defaults): %s",
            config_type, as_of.isoformat(), reason,
        )
        _note({"config_type": config_type, "as_of": as_of.isoformat(),
               "status": "missing", "reason": reason})
        return None
    chosen = max(
        eligible, key=lambda c: (c["knowledge_date"], c["stamp"], c["history_key"]),
    )
    key = chosen["history_key"]
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        payload = json.loads(obj["Body"].read())
        logger.info(
            "[config_archive] PIT %s @ %s → snapshot knowledge=%s (%s) s3://%s/%s",
            config_type, as_of.isoformat(),
            chosen["knowledge_date"].isoformat(), chosen["source"], bucket, key,
        )
        _note({"config_type": config_type, "as_of": as_of.isoformat(),
               "status": "resolved", "history_key": key,
               "knowledge_date": chosen["knowledge_date"].isoformat()})
        return payload
    except Exception as e:
        # The chosen snapshot cannot be read. This is NOT a cue to fall
        # forward to a newer snapshot (that re-introduces look-ahead). Fail
        # loud → caller uses genesis, same as "no eligible snapshot".
        logger.warning(
            "[config_archive] %s snapshot s3://%s/%s unreadable (%s) — "
            "treating as no-eligible-snapshot (genesis, NOT future-fallback)",
            config_type, bucket, key, e,
        )
        _note({"config_type": config_type, "as_of": as_of.isoformat(),
               "status": "unreadable",
               "reason": f"chosen snapshot {key} unreadable: {e}"})
        return None


def _safe_kd(entry: dict) -> _dt.date | None:
    try:
        return _dt.date.fromisoformat(entry["knowledge_date"])
    except (KeyError, ValueError, TypeError):
        return None
