"""
recommendation_artifact.py — typed S3 artifact for per-optimizer config recommendations.

Each Alpha Engine optimizer that wants to influence a live config key
(``config/executor_params.json``, ``config/scoring_weights.json``,
``config/predictor_params.json``) writes a ``RecommendationArtifact`` to
its own per-optimizer S3 path. A downstream assembler reads all
per-optimizer artifacts, applies merge precedence, and writes the single
live key. This module provides the contract.

S3 layout (target end-state — see ``optimizer-artifact-assembler-260509.md``)::

    config/
    ├── executor_params/
    │   ├── recommendations/{date}/
    │   │   ├── from_executor_optimizer.json
    │   │   ├── from_predictor_sizing_optimizer.json
    │   │   └── from_trigger_optimizer.json
    │   └── assembled/{date}.json
    ├── executor_params.json                  ← live key (written by assembler in cutover state)
    ├── executor_params_previous.json
    └── executor_params_history/{date}.json

This module is the foundation (PR 1 of the assembler arc). The dual-write
window: every optimizer's existing ``apply()`` keeps writing the legacy
live key AND additionally calls ``produce_artifact()`` to write its
per-optimizer recommendation. After the cutover (PR 4), individual
optimizers stop writing live and the assembler becomes the sole writer.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date as _date
from typing import Literal

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


# Recommendation kinds — encode merge semantics for the assembler.
#
# - ``full_replace``: replace the entire config block with this artifact's
#   ``recommended_params``. Used by ``executor_optimizer`` whose output is
#   the full set of risk/strategy params.
# - ``field_overlay``: merge specific keys from ``recommended_params`` into
#   the assembled output, preserving any keys this artifact doesn't own.
#   Used by ``predictor_sizing_optimizer`` (adds ``use_p_up_sizing`` etc.)
#   and ``trigger_optimizer`` (replaces ``disabled_triggers`` list).
# - ``list_overlay``: replace a single list-valued key (specified in
#   ``overlay_keys``) wholesale. Reserved for optimizers whose output is
#   semantically a list.
RecommendationKind = Literal["full_replace", "field_overlay", "list_overlay"]

# Promotion intent — captures what the optimizer would HAVE done in a
# single-writer world. The assembler decides what actually lands; the
# artifact records the optimizer's intent for audit.
#
# - ``promote``: optimizer's gates passed; this recommendation should land
#   in the assembled live config.
# - ``shadow``: optimizer's gates passed but a feature flag (e.g.
#   ``enforce_skill_composite=false``) keeps it out of live; assembler
#   records but does not promote.
# - ``skip``: optimizer didn't produce a usable recommendation
#   (insufficient_data / no_improvement / negative_sortino / error / etc.).
#   Assembler ignores for merge purposes but the artifact is still
#   written so the audit trail is complete.
PromotionIntent = Literal["promote", "shadow", "skip"]


@dataclass
class RecommendationArtifact:
    """A typed config-recommendation artifact written to S3 per optimizer per run.

    Required fields are positional-or-keyword; optional fields default to
    safe values. ``to_dict`` / ``to_json`` / ``from_dict`` provide JSON
    round-trip; ``s3_key`` is the canonical S3 path for this artifact's
    config_type + run_date + optimizer_name.
    """

    fit_target: str
    optimizer_name: str
    run_date: str
    recommendation_kind: RecommendationKind
    recommended_params: dict
    promotion_intent: PromotionIntent
    schema_version: int = 1
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    overlay_keys: list[str] | None = None
    diagnostic: dict = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict) -> "RecommendationArtifact":
        return cls(**data)

    def s3_key(self, config_type: str) -> str:
        return artifact_s3_key(config_type, self.run_date, self.optimizer_name)


def artifact_s3_key(config_type: str, run_date: str, optimizer_name: str) -> str:
    """Canonical S3 key for a per-optimizer recommendation artifact.

    Example::

        config/executor_params/recommendations/2026-05-09/from_executor_optimizer.json
    """
    return (
        f"config/{config_type}/recommendations/{run_date}/from_{optimizer_name}.json"
    )


def write_artifact(
    artifact: RecommendationArtifact, bucket: str, config_type: str, s3_client=None,
) -> str:
    """Write a recommendation artifact to S3.

    Args:
        artifact: The artifact to write.
        bucket: S3 bucket name.
        config_type: One of ``executor_params`` / ``scoring_weights`` /
            ``predictor_params``.
        s3_client: Optional boto3 client (test injection).

    Returns:
        The S3 key the artifact was written to.

    Raises:
        ClientError on S3 failure (caller decides whether the failure is
        fatal — typically the recommendation artifact write is non-fatal
        during the dual-write window).
    """
    s3 = s3_client or boto3.client("s3")
    key = artifact.s3_key(config_type)
    body = artifact.to_json()
    s3.put_object(
        Bucket=bucket, Key=key, Body=body, ContentType="application/json",
    )
    logger.info(
        "Wrote recommendation artifact: s3://%s/%s (intent=%s, kind=%s)",
        bucket, key, artifact.promotion_intent, artifact.recommendation_kind,
    )
    return key


def read_artifact(
    bucket: str,
    config_type: str,
    run_date: str,
    optimizer_name: str,
    s3_client=None,
) -> RecommendationArtifact | None:
    """Read a single per-optimizer recommendation artifact from S3.

    Returns ``None`` when the artifact does not exist (NoSuchKey). Other
    ClientErrors propagate.
    """
    s3 = s3_client or boto3.client("s3")
    key = artifact_s3_key(config_type, run_date, optimizer_name)
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = json.loads(obj["Body"].read())
        return RecommendationArtifact.from_dict(data)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
            return None
        raise


def read_all_artifacts_for_date(
    bucket: str, config_type: str, run_date: str, s3_client=None,
) -> dict[str, RecommendationArtifact]:
    """List + read every ``from_*.json`` artifact for a given config_type+run_date.

    Returns a ``{optimizer_name: artifact}`` dict. Used by the assembler.
    Skips files that aren't valid recommendation artifacts (logs warning,
    does not raise) so a malformed artifact doesn't block the whole assembly.
    """
    s3 = s3_client or boto3.client("s3")
    prefix = f"config/{config_type}/recommendations/{run_date}/"
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

    artifacts: dict[str, RecommendationArtifact] = {}
    for obj in response.get("Contents", []):
        key = obj["Key"]
        filename = key.rsplit("/", 1)[-1]
        if not filename.startswith("from_") or not filename.endswith(".json"):
            continue
        optimizer_name = filename[len("from_"):-len(".json")]
        try:
            obj_data = s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj_data["Body"].read())
            artifacts[optimizer_name] = RecommendationArtifact.from_dict(data)
        except Exception as e:
            logger.warning(
                "Failed to read recommendation artifact s3://%s/%s: %s — skipping",
                bucket, key, e,
            )

    return artifacts


def derive_promotion_intent(result: dict) -> PromotionIntent:
    """Map an optimizer ``recommend()`` result dict to a PromotionIntent.

    Convention shared across optimizers:
    - ``status="ok"`` AND ``apply_result.applied=True`` → ``promote``
    - ``status="ok"`` AND ``apply_result.applied=False`` → ``shadow``
    - any other status → ``skip``

    Note that ``apply_result`` may not be present at the time the artifact
    is written (we may produce the artifact BEFORE running apply); in that
    case we default to ``promote`` if status is ``ok`` and the caller
    hasn't decided otherwise.
    """
    status = result.get("status")
    if status != "ok":
        return "skip"
    apply_result = result.get("apply_result")
    if apply_result is None:
        return "promote"
    if apply_result.get("applied"):
        return "promote"
    return "shadow"


def today_iso() -> str:
    """Today's date in YYYY-MM-DD. Wrapped for test injection."""
    return str(_date.today())
