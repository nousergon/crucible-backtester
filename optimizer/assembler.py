"""
assembler.py — single-writer config-recommendation assembler.

Reads all per-optimizer recommendation artifacts for a (config_type, run_date),
applies merge precedence with explicit semantics per artifact's
``recommendation_kind``, and produces the assembled config that should be the
sole content of the live S3 key. PR 3 of the optimizer-artifact-assembler arc
(plan: ``~/Development/alpha-engine-docs/private/optimizer-artifact-assembler-260509.md``).

This PR is **shadow-only**: writes ``config/{config_type}/assembled/{date}.json``
for audit but does NOT write the live key. Individual optimizers still write
the live key via their existing legacy paths. PR 4 flips the cutover so the
assembler becomes the sole writer of the live key.

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
            "trigger_optimizer",
        ],
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


@dataclass
class AssemblerResult:
    """Outcome of an assembly run.

    - ``status`` — overall outcome:
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
    """
    s3 = s3_client or boto3.client("s3")
    key = f"config/{config_type}/assembled/{run_date}.json"
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
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
        config_type: Currently ``executor_params``. Will extend to
            ``scoring_weights`` / ``predictor_params`` in a follow-up arc.
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
        cutover_outcome = _cutover_apply(result, bucket, s3)
        result.notes = (result.notes + " " + cutover_outcome).strip()

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

    Returns a human-readable note for ``result.notes``. All three writes
    are best-effort — failures log warn and are recorded in the note;
    they don't raise. The shadow audit artifact is the authoritative
    record; live-key write failures are visible via that artifact.
    """
    config_type = result.config_type
    live_key = f"config/{config_type}.json"
    previous_key = f"config/{config_type}_previous.json"
    history_key = f"config/{config_type}_history/{result.run_date}.json"

    # 1. Snapshot current live → _previous (rollback safety)
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
        return f"cutover_failed: live write — {e}"

    # 3. Mirror to dated history (matches legacy convention).
    try:
        s3_client.put_object(
            Bucket=bucket, Key=history_key, Body=body, ContentType="application/json",
        )
        logger.info("Cutover: wrote dated history s3://%s/%s", bucket, history_key)
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

    Path: ``config/{config_type}/assembled/{run_date}.json``.
    Failure is non-fatal during the shadow-only PR 3 phase — logs warn,
    returns empty string.
    """
    key = f"config/{result.config_type}/assembled/{result.run_date}.json"
    try:
        body = json.dumps(result.to_dict(), indent=2, sort_keys=True)
        s3_client.put_object(
            Bucket=bucket, Key=key, Body=body, ContentType="application/json",
        )
        logger.info(
            "Wrote assembler audit artifact: s3://%s/%s (status=%s, "
            "promoting=%d, frozen_keys_restored=%d)",
            bucket, key, result.status,
            sum(1 for v in result.artifacts_seen.values() if v["promotion_intent"] == "promote"),
            len(result.frozen_keys_restored),
        )
        return key
    except Exception as e:
        logger.warning(
            "Failed to write assembler audit artifact s3://%s/%s: %s "
            "(non-fatal — shadow-only mode)", bucket, key, e,
        )
        return ""
