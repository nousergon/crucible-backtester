"""
assembler.py — single-writer config-recommendation assembler.

Reads all per-optimizer recommendation artifacts for a (config_type, run_date),
applies merge precedence with explicit semantics per artifact's
``recommendation_kind``, and produces the assembled config that should be the
sole content of the live S3 key. Originated as PR 3 of the
optimizer-artifact-assembler arc
(plan: ``~/Development/alpha-engine-docs/private/optimizer-artifact-assembler-260509.md``).

**Cutover has been LIVE since ~2026-05-27** (PR 4): when
``assembler.cutover_enabled`` is true, this module is the SOLE writer of
``config/{config_type}.json`` — individual optimizers' legacy live-key
writes are skipped (gated by ``is_cutover_enabled()``). Always writes
``config/{config_type}/assembled/{date}.json`` for audit, live or shadow.
Because this is the sole writer of live trading config under cutover, a
live-key write failure raises ``CutoverApplyError`` (see ``_cutover_apply``)
rather than being swallowed — a transient S3 failure must never be graded
"promoted" by ``apply_audit.classify_loop``.

## Merge semantics

Precedence is an ordered list of optimizer names. Earlier entries are merged
first; later entries override on overlap. Each artifact's
``recommendation_kind`` controls how its ``recommended_params`` integrates:

- ``full_replace``: replaces the *entire* assembled dict with this optimizer's
  ``recommended_params``. Any prior keys (from the base or earlier overlays)
  not present in this optimizer's params are dropped. Used by
  ``executor_optimizer`` whose output is the canonical risk/strategy block.
- ``field_overlay``: copies specific keys (named in ``overlay_keys``) from
  ``recommended_params`` into the assembled dict, preserving every other
  key. Used by ``predictor_sizing_optimizer`` (adds ``use_p_up_sizing`` etc.)
  and ``trigger_optimizer`` (replaces ``disabled_triggers`` list).
- ``list_overlay``: identical to ``field_overlay`` for the dict-merge layer;
  reserved for optimizers whose output is semantically a single list.

Only artifacts with ``promotion_intent="promote"`` participate in the merge.
``shadow`` and ``skip`` artifacts are recorded in the result for audit but
contribute nothing to assembled_params.

## Frozen keys

The precedence config can name keys in ``freeze_keys`` that no optimizer
can modify. The assembler reads the current live config before merging and
restores those keys' original values after the merge — so a freeze is
operator-controlled and overrides any optimizer recommendation.

## Provenance audit

The result includes ``merge_summary`` — a dict ``{key: {value, writer, kind,
run_id}}`` capturing which optimizer contributed which value. Answers the
forensic question that motivated this arc: "which optimizer set
``atr_multiplier`` to 3.0 today?" becomes a single dict lookup.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import boto3
from nousergon_lib.eval_artifacts import (
    eval_artifact_key,
    eval_latest_key,
    new_eval_run_id,
)
from botocore.exceptions import ClientError

from optimizer.recommendation_artifact import (
    RecommendationArtifact,
    read_all_artifacts_for_date,
)

logger = logging.getLogger(__name__)


# Default precedence per config_type. Encoded here so PR 3 ships
# self-contained; PR 4 (cutover) will move this to alpha-engine-config's
# backtester/config.yaml under an ``assembler:`` section so adding a new
# optimizer to the precedence chain is a config change, not a code change.
DEFAULT_PRECEDENCE = {
    "executor_params": {
        "precedence": [
            # Order matters — earlier entries merge first; later overlays win
            # on overlapping keys. The executor optimizer's full_replace
            # establishes the risk/strategy baseline; field_overlay writers
            # add their narrow fields on top.
            "executor_optimizer",
            "predictor_sizing_optimizer",
            # L300 (2026-06-01): stance_size_* multipliers, tuned offline against
            # realized per-stance alpha (field_overlay), replacing the inert
            # predictionless sweep.
            "stance_sizing_optimizer",
            "trigger_optimizer",
        ],
        "freeze_keys": [],
    },
    # config#2054: extends the cutover arc beyond executor_params. Each of
    # these three config types has exactly one live producer (single-entry
    # precedence, full_replace) — unlike executor_params' multi-writer overlay
    # chain, there is no merge-order ambiguity to resolve here.
    "scoring_weights": {
        "precedence": ["weight_optimizer"],
        "freeze_keys": [],
    },
    "predictor_params": {
        "precedence": ["veto_analysis"],
        "freeze_keys": [],
    },
    "research_params": {
        "precedence": ["research_optimizer"],
        "freeze_keys": [],
    },
}

# Module-level cutover flag. Toggled via ``set_cutover_enabled()`` at
# pipeline startup (called from evaluate.py / backtest.py after they load
# config). When true, individual optimizers' ``apply()`` skips the legacy
# live-key write and the assembler becomes the sole writer of
# ``config/{config_type}.json`` + the rollback ``_previous`` snapshot.
# Default False — PR 4 ships the cutover mechanism dark; alpha-engine-config
# flips ``assembler.cutover_enabled`` to true in a separate operator step
# after at least one shadow Sat SF cycle has validated the assembled output.
_CUTOVER_ENABLED = False


def set_cutover_enabled(enabled: bool) -> None:
    """Set the global cutover flag. Called once per process at startup."""
    global _CUTOVER_ENABLED
    _CUTOVER_ENABLED = bool(enabled)


def is_cutover_enabled() -> bool:
    """Return the current cutover flag. Read by each optimizer's ``apply()``
    to decide whether to skip its legacy live-key write."""
    return _CUTOVER_ENABLED


AssemblerStatus = Literal["ok", "no_artifacts", "all_skip"]

# Structured cutover outcome — kept independent of ``AssemblerStatus`` so a
# live-key write failure can never be conflated with "merge produced no
# promotable artifact". ``classify_loop`` in apply_audit.py keys off this
# field (not ``status``) to decide whether a cutover-mode loop actually
# promoted.
#   - ``not_attempted``: cutover was off, or ``status`` wasn't "ok" so no
#     live write was attempted.
#   - ``applied``: the live-key put_object succeeded (history mirror
#     failures do NOT downgrade this — the live key is the source of truth).
#   - ``failed``: the live-key put_object raised. The live config is
#     UNCHANGED from before this run; this must never be graded "promoted".
CutoverStatus = Literal["not_attempted", "applied", "failed"]


class CutoverApplyError(RuntimeError):
    """Raised when the assembler's live-key write fails under cutover mode.

    Carries the underlying cause; callers that catch this must not treat the
    run as promoted. ``assemble()`` catches this internally to populate
    ``AssemblerResult.cutover_status="failed"`` (fail-loud via ERROR-level
    log + this exception's chain) rather than letting a transient S3 blip
    crash the whole evaluate run — but the *result* it returns can never be
    mistaken for a successful promotion.
    """


@dataclass
class AssemblerResult:
    """Outcome of an assembly run.

    - ``status`` — overall MERGE outcome (says nothing about the live-key
      write — see ``cutover_status`` for that):
        - ``ok``: at least one artifact contributed; ``assembled_params``
          reflects the merged config.
        - ``no_artifacts``: no per-optimizer artifacts found for the date.
          ``assembled_params`` is the unchanged base (current live config).
        - ``all_skip``: artifacts found but every one had
          ``promotion_intent`` of ``shadow`` or ``skip``.
          ``assembled_params`` is the unchanged base.
    - ``assembled_params`` — the merged config dict (would be live key body
      under PR 4 cutover; under PR 3 only the assembled audit artifact body).
    - ``merge_summary`` — provenance per key: ``{key: {value, writer, kind,
      run_id}}``. Answers "who set this field today?".
    - ``artifacts_seen`` — every artifact considered, with intent + run_id.
    - ``frozen_keys_restored`` — keys whose value was restored from base
      because they appeared in ``freeze_keys`` (operator override).
    - ``base_was_present`` — whether a current live config was found at the
      start (False on first-ever run).
    - ``cutover_status`` — outcome of the live-key WRITE under cutover mode
      (independent of ``status``, the merge outcome). ``"not_attempted"``
      when cutover was off or the merge produced nothing to write.
      ``"applied"`` only when the live-key ``put_object`` actually
      succeeded. ``"failed"`` when it raised — the live config is
      unchanged; this must be graded as an error, never "promoted".
    """

    status: AssemblerStatus
    config_type: str
    run_date: str
    assembled_params: dict
    merge_summary: dict
    artifacts_seen: dict
    frozen_keys_restored: list[str] = field(default_factory=list)
    base_was_present: bool = False
    notes: str = ""
    cutover_status: CutoverStatus = "not_attempted"

    def to_dict(self) -> dict:
        return asdict(self)


def read_assembled(
    bucket: str, config_type: str, run_date: str, s3_client=None,
) -> dict | None:
    """Read the assembled audit artifact for a given config_type + run_date.

    Returns the parsed dict (an ``AssemblerResult.to_dict()`` body) or
    ``None`` if no audit artifact exists for that date. Used by the
    rollback audit to capture what would-have-been the assembled config
    when a rollback fires.

    Path resolution (canonical lib v0.8.0 layout):
    1. Try ``config/{config_type}/assembled/latest.json`` first — this is
       the operator-UX sidecar mirroring the most-recently-written audit.
       Single-fetch, fast, the right answer for the typical "give me the
       latest" use case (which is what the rollback path needs).
    2. Fall back to legacy ``config/{config_type}/assembled/{run_date}.json``
       for historical audits written before the canonical-layout cutover.
       Tolerant-reader behavior so the rollback path doesn't break on
       legacy data during the transition window.
    """
    s3 = s3_client or boto3.client("s3")
    canonical_key = f"config/{config_type}/assembled/latest.json"
    try:
        obj = s3.get_object(Bucket=bucket, Key=canonical_key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey"):
            raise
        # Fall through to legacy-layout fallback

    legacy_key = f"config/{config_type}/assembled/{run_date}.json"
    try:
        obj = s3.get_object(Bucket=bucket, Key=legacy_key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
            return None
        raise


def _read_current_live(
    bucket: str, config_type: str, s3_client,
) -> tuple[dict, bool]:
    """Read the current live config (the assembler's merge base).

    Returns ``(config, was_present)``. NoSuchKey returns ``({}, False)``.
    Other ClientErrors propagate.
    """
    key = f"config/{config_type}.json"
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        data = json.loads(obj["Body"].read())
        if not isinstance(data, dict):
            logger.warning(
                "Current live config at s3://%s/%s is not a dict (%s) — "
                "treating as empty base", bucket, key, type(data).__name__,
            )
            return {}, False
        return data, True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
            return {}, False
        raise


def _apply_artifact_to_base(
    base: dict, artifact: RecommendationArtifact, summary: dict,
) -> dict:
    """Apply a single artifact's contribution to the running merged dict.

    Returns a *new* dict (does not mutate ``base``). Updates ``summary``
    in-place with provenance entries for keys this artifact wrote.
    """
    kind = artifact.recommendation_kind
    if kind == "full_replace":
        # Replace the entire dict with this artifact's recommended_params.
        # Provenance: every key in the new dict is owned by this artifact.
        # Earlier-recorded provenance entries for keys that disappear are
        # dropped (they're no longer in the assembled output).
        new_base = dict(artifact.recommended_params)
        for key in list(summary.keys()):
            if key not in new_base:
                del summary[key]
        for key, value in new_base.items():
            summary[key] = {
                "value": value,
                "writer": artifact.optimizer_name,
                "kind": "full_replace",
                "run_id": artifact.run_id,
            }
        return new_base

    if kind in ("field_overlay", "list_overlay"):
        new_base = dict(base)
        overlay_keys = artifact.overlay_keys or list(artifact.recommended_params.keys())
        for key in overlay_keys:
            if key not in artifact.recommended_params:
                # overlay_keys can list keys not present in recommended_params
                # (e.g. when the optimizer downgrades to skip mid-run); skip
                # those defensively.
                continue
            new_base[key] = artifact.recommended_params[key]
            summary[key] = {
                "value": artifact.recommended_params[key],
                "writer": artifact.optimizer_name,
                "kind": kind,
                "run_id": artifact.run_id,
            }
        return new_base

    logger.warning(
        "Unknown recommendation_kind=%s on artifact from %s — skipping",
        kind, artifact.optimizer_name,
    )
    return dict(base)


def assemble(
    bucket: str,
    config_type: str,
    run_date: str,
    precedence_config: dict | None = None,
    s3_client: Any = None,
    write_assembled: bool = True,
    cutover_enabled: bool | None = None,
) -> AssemblerResult:
    """Read all per-optimizer recommendation artifacts for the date, merge
    them via precedence, and write the assembled audit artifact (and,
    under cutover mode, the live key + rollback snapshot).

    Args:
        bucket: S3 bucket name.
        config_type: One of ``executor_params`` / ``scoring_weights`` /
            ``predictor_params`` / ``research_params`` (config#2054).
        run_date: YYYY-MM-DD. Used to scope which artifacts to read.
        precedence_config: Override the default precedence + freeze_keys
            for this config_type. Shape::
                {"precedence": ["executor_optimizer", ...], "freeze_keys": []}
            If None, uses ``DEFAULT_PRECEDENCE[config_type]``.
        s3_client: Optional boto3 client (test injection).
        write_assembled: If True (default), writes
            ``config/{config_type}/assembled/{run_date}.json`` audit
            artifact regardless of cutover mode.
        cutover_enabled: If True, additionally writes the live key
            ``config/{config_type}.json`` + the rollback snapshot
            ``config/{config_type}_previous.json`` + the dated history
            ``config/{config_type}_history/{run_date}.json``. If None
            (default), reads the module-level ``_CUTOVER_ENABLED`` flag
            set by ``set_cutover_enabled()`` at process startup. The
            individual optimizers' ``apply()`` paths read the same flag
            and skip their legacy live writes when it's true — so when
            cutover is on, the assembler is the sole writer.

    Returns:
        AssemblerResult capturing assembled_params + merge_summary +
        artifacts_seen + status. Cutover S3 writes happen as side effects;
        ``result.notes`` records whether cutover writes occurred.
    """
    s3 = s3_client or boto3.client("s3")
    cfg = precedence_config or DEFAULT_PRECEDENCE.get(config_type, {})
    precedence: list[str] = cfg.get("precedence", [])
    freeze_keys: list[str] = cfg.get("freeze_keys", [])

    base, base_was_present = _read_current_live(bucket, config_type, s3)
    artifacts = read_all_artifacts_for_date(bucket, config_type, run_date, s3_client=s3)

    # Snapshot frozen-key values from the original base BEFORE any merge —
    # these get restored at the end so the operator's freeze is authoritative.
    frozen_snapshot = {k: base[k] for k in freeze_keys if k in base}

    artifacts_seen: dict[str, dict] = {}
    for name, artifact in artifacts.items():
        artifacts_seen[name] = {
            "run_id": artifact.run_id,
            "promotion_intent": artifact.promotion_intent,
            "recommendation_kind": artifact.recommendation_kind,
        }

    if not artifacts:
        return AssemblerResult(
            status="no_artifacts",
            config_type=config_type,
            run_date=run_date,
            assembled_params=dict(base),
            merge_summary={},
            artifacts_seen={},
            base_was_present=base_was_present,
            notes=(
                f"No per-optimizer artifacts found at "
                f"config/{config_type}/recommendations/{run_date}/ — "
                f"assembled_params is the unchanged current live config."
            ),
        )

    # Walk precedence in order. Each iteration applies that optimizer's
    # contribution; only artifacts with promotion_intent="promote" merge in.
    merged = dict(base)
    summary: dict = {}
    promoting_count = 0

    for optimizer_name in precedence:
        artifact = artifacts.get(optimizer_name)
        if artifact is None:
            continue  # No artifact from this optimizer this run
        if artifact.promotion_intent != "promote":
            continue  # Recorded for audit but doesn't merge
        merged = _apply_artifact_to_base(merged, artifact, summary)
        promoting_count += 1

    # Restore frozen keys from the original base — operator override is
    # authoritative regardless of optimizer recommendations.
    frozen_keys_restored: list[str] = []
    for key, value in frozen_snapshot.items():
        if merged.get(key) != value:
            merged[key] = value
            frozen_keys_restored.append(key)
            summary[key] = {
                "value": value,
                "writer": "operator_freeze",
                "kind": "freeze_restore",
                "run_id": None,
            }

    if promoting_count == 0:
        result = AssemblerResult(
            status="all_skip",
            config_type=config_type,
            run_date=run_date,
            assembled_params=dict(base),
            merge_summary={},
            artifacts_seen=artifacts_seen,
            frozen_keys_restored=[],
            base_was_present=base_was_present,
            notes=(
                f"{len(artifacts)} artifact(s) found but none had "
                f"promotion_intent=promote — assembled_params is the "
                f"unchanged current live config."
            ),
        )
    else:
        result = AssemblerResult(
            status="ok",
            config_type=config_type,
            run_date=run_date,
            assembled_params=merged,
            merge_summary=summary,
            artifacts_seen=artifacts_seen,
            frozen_keys_restored=frozen_keys_restored,
            base_was_present=base_was_present,
            notes=(
                f"Merged {promoting_count} promoting artifact(s) of "
                f"{len(artifacts)} found."
            ),
        )

    if write_assembled:
        _write_assembled_audit(result, bucket, s3)

    cutover_active = (
        is_cutover_enabled() if cutover_enabled is None else bool(cutover_enabled)
    )
    if cutover_active and result.status == "ok":
        try:
            cutover_outcome = _cutover_apply(result, bucket, s3)
            result.cutover_status = "applied"
            result.notes = (result.notes + " " + cutover_outcome).strip()
        except CutoverApplyError as e:
            # Fail LOUD: this is the sole writer of live trading config.
            # A swallowed failure here previously left result.status="ok",
            # which apply_audit.classify_loop graded as "promoted" — an
            # executor could silently trade on stale params while the audit
            # trail claimed success. cutover_status="failed" is what
            # classify_loop keys on now; status/assembled_params are left
            # alone since the MERGE itself succeeded — only the live WRITE
            # didn't.
            logger.error(
                "Cutover: CRITICAL — live-key write failed for config_type=%s "
                "run_date=%s; live config is UNCHANGED, executor may be "
                "trading on stale params: %s",
                config_type, run_date, e,
            )
            result.cutover_status = "failed"
            result.notes = (result.notes + " " + str(e)).strip()

    return result


def _cutover_apply(
    result: AssemblerResult, bucket: str, s3_client,
) -> str:
    """Promote the assembled config to the live key under cutover mode.

    Three S3 writes:
    1. Snapshot current live → ``config/{config_type}_previous.json``
       (subsumes ``optimizer/rollback.save_previous`` so regression rollback
       still works under the new single-writer regime).
    2. Write assembled_params (plus ``updated_at`` stamp matching legacy
       payload convention) → live key ``config/{config_type}.json``.
    3. Mirror the live write to ``config/{config_type}_history/{run_date}.json``
       (matches the legacy dated-history convention; preserves audit trail
       expected by operators inspecting weekly history).

    Returns a human-readable note for ``result.notes`` on success.

    Step 1 (rollback snapshot) failing is tolerated — it degrades rollback
    safety but does not leave the live key stale, so it logs WARN and
    continues. Step 2 (the live-key write) is THE load-bearing write of
    this function — the whole point of cutover mode is that the assembler
    is the sole writer of live trading config. A failure there RAISES
    ``CutoverApplyError`` instead of being folded into a note string, so
    the caller (``assemble()``) cannot return ``status="ok"``-shaped
    success while the live key silently didn't change. Step 3 (history
    mirror) failing after a successful live write is genuinely non-fatal —
    the live key is already correct — so it stays log-and-continue.
    """
    config_type = result.config_type
    live_key = f"config/{config_type}.json"
    previous_key = f"config/{config_type}_previous.json"
    # Canonical eval-style archive layout per lib v0.8.0 — flat
    # {prefix}/{run_id}.json + latest.json sidecar (YYMMDDHHMM run_id)
    history_run_id = new_eval_run_id()
    history_prefix = f"config/{config_type}_history"
    history_key = eval_artifact_key(history_prefix, history_run_id)
    history_latest_key = eval_latest_key(history_prefix)

    # 1. Snapshot current live → _previous (rollback safety)
    # Skip snapshot on rerun for the same trading_day — prevent clobbering prior snapshot.
    skip_snapshot = False
    try:
        latest_artifact = s3_client.get_object(Bucket=bucket, Key=history_latest_key)
        latest_data = json.loads(latest_artifact["Body"].read())
        if latest_data.get("as_of") == result.run_date:
            logger.info(
                "Cutover: skipping snapshot (rerun for same run_date=%s) — preserving prior rollback snapshot",
                result.run_date,
            )
            skip_snapshot = True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey"):
            logger.warning("Cutover: could not check history latest — proceeding with snapshot: %s", e)
    except Exception as e:
        # Non-ClientError failures (malformed body, transient network error,
        # etc.) must not block the load-bearing live write below — this is
        # only a best-effort idempotency check for the (non-critical)
        # rollback snapshot. Fall through and attempt the snapshot normally.
        logger.warning("Cutover: could not check history latest — proceeding with snapshot: %s", e)

    if not skip_snapshot:
        try:
            s3_client.copy_object(
                Bucket=bucket,
                CopySource={"Bucket": bucket, "Key": live_key},
                Key=previous_key,
            )
            logger.info(
                "Cutover: snapshotted live → s3://%s/%s", bucket, previous_key,
            )
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                logger.info(
                    "Cutover: no current live config to snapshot (first cutover run)",
                )
            else:
                logger.warning(
                    "Cutover: failed to snapshot live → _previous: %s "
                    "(continuing — live write below)", e,
                )
        except Exception as e:
            # Any other snapshot failure (non-ClientError) is likewise
            # non-critical — it degrades rollback safety but must not block
            # the load-bearing live write below (see docstring).
            logger.warning(
                "Cutover: failed to snapshot live → _previous: %s "
                "(continuing — live write below)", e,
            )

    # 2. Write assembled → live key (with updated_at stamp matching legacy
    #    payload shape so consumer Lambdas see no schema drift).
    payload = dict(result.assembled_params)
    payload["updated_at"] = result.run_date
    payload["assembled_by"] = "optimizer.assembler"
    body = json.dumps(payload, indent=2)
    try:
        s3_client.put_object(
            Bucket=bucket, Key=live_key, Body=body, ContentType="application/json",
        )
        logger.info(
            "Cutover: wrote live config to s3://%s/%s (assembler=sole writer)",
            bucket, live_key,
        )
    except Exception as e:
        logger.error(
            "Cutover: CRITICAL failure writing live key s3://%s/%s: %s",
            bucket, live_key, e,
        )
        raise CutoverApplyError(
            f"cutover_failed: live write to s3://{bucket}/{live_key} — {e}",
        ) from e

    # 3. Mirror to dated history — canonical lib v0.8.0 archive layout
    # (flat + latest.json sidecar; YYMMDDHHMM run_id encodes the time).
    try:
        s3_client.put_object(
            Bucket=bucket, Key=history_key, Body=body, ContentType="application/json",
        )
        s3_client.put_object(
            Bucket=bucket, Key=history_latest_key, Body=body,
            ContentType="application/json",
        )
        logger.info(
            "Cutover: wrote dated history s3://%s/%s (+ latest.json sidecar)",
            bucket, history_key,
        )
    except Exception as e:
        logger.warning(
            "Cutover: failed to write dated history s3://%s/%s: %s "
            "(non-fatal — live write succeeded)", bucket, history_key, e,
        )

    return f"cutover_applied: live={live_key}, previous_snapshot={previous_key}"


def _write_assembled_audit(
    result: AssemblerResult, bucket: str, s3_client,
) -> str:
    """Write the assembler result to S3 as an audit artifact.

    Canonical eval-style archive layout per ``nousergon_lib.eval_artifacts``
    (v0.8.0)::

        config/{config_type}/assembled/{run_id}.json    ← per-run audit
        config/{config_type}/assembled/latest.json      ← single-fetch sidecar

    Run_id is YYMMDDHHMM (UTC) — same-minute re-runs collide by design;
    typical Sat-SF cron cadence makes that effectively impossible.

    Failure here is non-fatal REGARDLESS of cutover mode — not because
    cutover is shadow-only (it has been LIVE since ~2026-05-27; that
    rationale is stale and no longer names a true condition). The real
    reason: this artifact is secondary forensic provenance (answers "which
    optimizer set this field?"), separate from and written independently
    of the live trading key. The live key write is the load-bearing one and
    lives in ``_cutover_apply``, which raises ``CutoverApplyError`` (not a
    swallowed note) on failure so it can never be graded "promoted" by
    apply_audit. Losing this audit artifact loses debuggability, not
    correctness of live config — so it logs warn and returns empty string
    rather than raising and aborting the run.
    """
    run_id = new_eval_run_id()
    prefix = f"config/{result.config_type}/assembled"
    key = eval_artifact_key(prefix, run_id)
    latest_key = eval_latest_key(prefix)
    try:
        body = json.dumps(result.to_dict(), indent=2, sort_keys=True)
        s3_client.put_object(
            Bucket=bucket, Key=key, Body=body, ContentType="application/json",
        )
        s3_client.put_object(
            Bucket=bucket, Key=latest_key, Body=body, ContentType="application/json",
        )
        logger.info(
            "Wrote assembler audit artifact: s3://%s/%s (+ latest.json sidecar; "
            "status=%s, promoting=%d, frozen_keys_restored=%d)",
            bucket, key, result.status,
            sum(1 for v in result.artifacts_seen.values() if v["promotion_intent"] == "promote"),
            len(result.frozen_keys_restored),
        )
        return key
    except Exception as e:
        logger.warning(
            "Failed to write assembler audit artifact s3://%s/%s: %s "
            "(non-fatal — this is the secondary provenance artifact, not "
            "the live trading key; live-key write failures are fail-loud "
            "via CutoverApplyError, not swallowed here)", bucket, key, e,
        )
        return ""
