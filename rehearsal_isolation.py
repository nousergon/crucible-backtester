"""Process-wide S3 isolation for a weekly-SF rehearsal (alpha-engine-config-I12351).

Brian's rule: rehearsals must be runnable at ANY time and isolated from live
files. ``rehearsal-2026-10-16-2``'s backtester and evaluator spots still wrote
about sixty live keys, measured from the bucket's S3 access logs:
``predictor/optimizer_gate/latest.json`` (a promotion lever), the backtest
attestation and ~45 dated reports, phase outputs, ``predictor/metrics/*``,
``config/metrics_history/*``, ``regime/stratified_sortino/latest.json``,
``evaluator/diagnostics/*`` and the ``_attribution_adequacy`` append log.
``--freeze`` (``optimizer.run_role``) only stops the live-CONFIG writes; it
does not reach these. Per-site guards would miss the next writer, so this is
one S3-layer isolate, a port of crucible-evaluator ``grading/rehearsal.py``
(itself a port of crucible-research ``data/rehearsal.py``), kept compatible in
its namespace layout so every stage of one rehearsal agrees on where its
objects live.

**The marker** is :func:`optimizer.run_role.rehearsal_reason`'s: ``RUN_TOKEN``
(the SF execution name ``krepis.ssm_log_capture`` exports and
``_spot_common.sh`` forwards onto the spot) starting ``rehearsal-``, or
``AE_PIPELINE_ROLE=rehearsal``. Any other process is production and nothing
here changes: :func:`activate` returns ``None`` without touching boto3.

**What isolation does.** :func:`activate` is called once by ``backtest.py``
and ``evaluate.py`` right after argument parsing. Under the marker it
registers a botocore hook on the boto3 default session, so every S3 client
the process creates afterwards (and every forked worker's) re-keys each write
to ``alpha-engine-research`` into ``rehearsal/<execution name>/<original key>``.
Reads see the rehearsal's own objects first: a keyed read of a key this
rehearsal wrote (found by listing the namespace once at activation, plus
every write since) is served from the namespace; every other read goes to the
live key, so live inputs are still read.

Keys that are passed through unchanged, because something else already
isolates them or reads them where they are:

- keys already in a rehearsal namespace (``rehearsal/...``, or a
  ``.rehearsal/`` segment: lib phase markers, lib heartbeats, krepis
  verdicts and RSS rows);
- keys naming this execution (``health/exec_code_pin/<exec>/...``,
  ``health/weekly_preflight_on_spot/<date>/<exec>.json``);
- ``_flow_doctor/`` (a rehearsal reports through ``flow-doctor.rehearsal.yaml``,
  whose findings the nightly sweep reads there);
- ``decision_artifacts/_cost_raw/`` (cost records stay where AggregateCosts
  reads them, as in the evaluator port);
- every bucket other than ``alpha-engine-research``.

Limits, stated rather than hidden:

- ``ListObjects*`` is not overlaid: a listing returns live keys only.
- A write-shaped S3 operation that carries no ``Key`` (other than
  ``DeleteObjects``) RAISES :class:`RehearsalLiveWriteRefused`.
- A client created BEFORE :func:`activate` is not covered;
  :meth:`RehearsalContext.attach` covers one explicitly. Both entrypoints
  activate before building any client.
- Non-S3 side effects (CloudWatch metrics, SNS) are not re-keyed.
- ``krepis.stage_coverage assert`` runs as a separate process after the
  stage, without this isolate, so a rehearsal's verdict probes the live keys
  (it is recorded in krepis' own rehearsal namespace with no metric).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from typing import Any

from optimizer.run_role import (
    PIPELINE_ROLE_ENV,
    REHEARSAL_PREFIX,
    RUN_TOKEN_ENV,
)

logger = logging.getLogger(__name__)

#: Root of the rehearsal namespace inside the bucket (same as the research
#: and evaluator ports).
NAMESPACE_ROOT = "rehearsal"
#: Bucket whose writes are isolated.
ISOLATED_BUCKETS = frozenset({"alpha-engine-research"})
#: Key prefixes passed through unchanged (see the module docstring).
PASSTHROUGH_PREFIXES = ("_flow_doctor/", "decision_artifacts/_cost_raw/")

_COPY_OPS = frozenset({"CopyObject", "UploadPartCopy"})
#: Keyed write operations whose ``Key`` is re-keyed into the namespace.
_KEYED_WRITE_OPS = frozenset(
    {
        "PutObject",
        "CopyObject",
        "DeleteObject",
        "CreateMultipartUpload",
        "UploadPart",
        "UploadPartCopy",
        "CompleteMultipartUpload",
        "AbortMultipartUpload",
        "ListParts",
        "PutObjectTagging",
        "DeleteObjectTagging",
        "PutObjectAcl",
        "PutObjectRetention",
        "PutObjectLegalHold",
        "RestoreObject",
    }
)
#: Keyed reads, served from the namespace when this rehearsal wrote the key.
_KEYED_READ_OPS = frozenset(
    {
        "GetObject",
        "HeadObject",
        "GetObjectTagging",
        "GetObjectAttributes",
        "GetObjectAcl",
        "SelectObjectContent",
    }
)
#: Operation-name prefixes of reads that carry no object key.
_UNKEYED_READ_PREFIXES = ("List", "Get", "Head")


class RehearsalLiveWriteRefused(RuntimeError):
    """A rehearsal reached an S3 write this module cannot re-key. Raised,
    never swallowed: the write must not happen at a live key."""


def rehearsal_name(env: Mapping[str, str] | None = None) -> str | None:
    """The rehearsal's execution name, or ``None`` for any other run.

    Same marker as :func:`optimizer.run_role.rehearsal_reason`.
    """
    import os

    env = os.environ if env is None else env
    token = (env.get(RUN_TOKEN_ENV) or "").strip()
    if token.startswith("arn:"):
        token = token.rsplit(":", 1)[-1]
    if token.startswith(REHEARSAL_PREFIX):
        return token
    if (env.get(PIPELINE_ROLE_ENV) or "").strip().lower() == "rehearsal":
        return f"{REHEARSAL_PREFIX}manual-{token or 'run'}"
    return None


def namespace_prefix(name: str) -> str:
    """``rehearsal/<execution name>/``: where every isolated write lands."""
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)
    return f"{NAMESPACE_ROOT}/{safe}/"


class RehearsalContext:
    """One rehearsal process: its namespace and read overlay."""

    def __init__(self, name: str, *, entrypoint: str) -> None:
        self.name = name
        self.entrypoint = entrypoint
        self.prefix = namespace_prefix(name)
        self._lock = threading.Lock()
        #: (bucket, live key) pairs whose current value lives in the namespace.
        self._overlay: set[tuple[str, str]] = set()
        self.redirected: list[str] = []
        self._uid = f"rehearsal-isolation-{id(self)}"

    # ── key mapping ────────────────────────────────────────────────────────
    def ns_key(self, key: str) -> str:
        return key if key.startswith(self.prefix) else f"{self.prefix}{key}"

    def passes_through(self, bucket: str, key: str) -> bool:
        """True for a key that stays where it is (see the module docstring)."""
        if bucket not in ISOLATED_BUCKETS:
            return True
        if key.startswith(f"{NAMESPACE_ROOT}/") or key.startswith(".rehearsal/"):
            return True
        if "/.rehearsal/" in key or key.startswith(PASSTHROUGH_PREFIXES):
            return True
        segments = key.split("/")
        return self.name in segments or f"{self.name}.json" in segments

    def is_overlaid(self, bucket: str, key: str) -> bool:
        with self._lock:
            return (bucket, key) in self._overlay

    def prime(self, s3_client: Any, bucket: str) -> int:
        """Load the keys earlier stages of this execution wrote into
        ``bucket``'s namespace, so their reads are served from it."""
        n = 0
        token = None
        for _ in range(1000):
            kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": self.prefix}
            if token:
                kwargs["ContinuationToken"] = token
            resp = s3_client.list_objects_v2(**kwargs)
            with self._lock:
                for obj in resp.get("Contents") or []:
                    k = obj.get("Key", "")
                    if k.startswith(self.prefix):
                        self._overlay.add((bucket, k[len(self.prefix) :]))
                        n += 1
            if resp.get("IsTruncated") is not True:
                break
            token = resp.get("NextContinuationToken")
            if not isinstance(token, str) or not token:
                break
        return n

    # ── the botocore hook ──────────────────────────────────────────────────
    def _hook(self, params: dict, model: Any, **_: Any) -> None:
        if _ACTIVE is not self:
            return
        op = getattr(model, "name", "")
        bucket = params.get("Bucket", "")
        if bucket not in ISOLATED_BUCKETS:
            return
        if op in _KEYED_WRITE_OPS:
            if op in _COPY_OPS:
                self._rewrite_copy_source(params)
            key = params.get("Key")
            if not isinstance(key, str) or not key:
                raise RehearsalLiveWriteRefused(
                    f"rehearsal {self.name}: S3 {op} without a Key cannot be re-keyed"
                )
            if self.passes_through(bucket, key):
                return
            params["Key"] = self.ns_key(key)
            with self._lock:
                if op == "DeleteObject":
                    self._overlay.discard((bucket, key))
                elif op in ("PutObject", "CopyObject", "CreateMultipartUpload", "CompleteMultipartUpload"):
                    self._overlay.add((bucket, key))
                    if op != "CreateMultipartUpload":
                        self.redirected.append(f"s3://{bucket}/{key}")
            return
        if op == "DeleteObjects":
            for obj in (params.get("Delete") or {}).get("Objects") or []:
                key = obj.get("Key", "")
                if key and not self.passes_through(bucket, key):
                    obj["Key"] = self.ns_key(key)
                    with self._lock:
                        self._overlay.discard((bucket, key))
            return
        if op in _KEYED_READ_OPS:
            key = params.get("Key")
            if isinstance(key, str) and self.is_overlaid(bucket, key):
                params["Key"] = self.ns_key(key)
            return
        if op.startswith(_UNKEYED_READ_PREFIXES) and "Key" not in params:
            return
        raise RehearsalLiveWriteRefused(
            f"rehearsal {self.name}: S3 {op or '<unknown>'} is not a known read "
            "or keyed write, so it cannot be isolated; refusing it "
            "(alpha-engine-config-I12351)"
        )

    def _rewrite_copy_source(self, params: dict) -> None:
        src = params.get("CopySource")
        if isinstance(src, dict):
            b, k = src.get("Bucket", ""), src.get("Key", "")
            if k and self.is_overlaid(b, k):
                params["CopySource"] = {**src, "Key": self.ns_key(k)}
        elif isinstance(src, str):
            b, _, k = src.lstrip("/").partition("/")
            if k and self.is_overlaid(b, k):
                params["CopySource"] = f"{b}/{self.ns_key(k)}"

    def attach(self, client: Any) -> Any:
        """Register the hook on one already-created client. Idempotent."""
        client.meta.events.register(
            "before-parameter-build.s3", self._hook, unique_id=self._uid
        )
        return client

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "execution_name": self.name,
                "entrypoint": self.entrypoint,
                "namespace": self.prefix,
                "writes_redirected": len(self.redirected),
                "redirected_sample": self.redirected[:25],
            }


_ACTIVE: RehearsalContext | None = None


def active() -> RehearsalContext | None:
    """The rehearsal context of this process, or None."""
    return _ACTIVE


def activate(
    *,
    entrypoint: str,
    session: Any = None,
    env: Mapping[str, str] | None = None,
) -> RehearsalContext | None:
    """Isolate this process's S3 writes when it is a rehearsal.

    Returns None (and changes nothing) for any other run. Idempotent: a
    second call returns the active context.
    """
    global _ACTIVE
    name = rehearsal_name(env)
    if name is None:
        return None
    if _ACTIVE is not None:
        return _ACTIVE

    import boto3

    sess = session or boto3._get_default_session()  # noqa: SLF001 — the session boto3.client() uses
    ctx = RehearsalContext(name, entrypoint=entrypoint)
    lister = sess.client("s3")
    for bucket in sorted(ISOLATED_BUCKETS):
        try:
            n = ctx.prime(lister, bucket)
        except Exception as exc:  # noqa: BLE001 — a missing overlay only means reads go live
            logger.warning(
                "[rehearsal] %s: could not list %s in %s (%s); reads of keys "
                "earlier stages wrote will go to the live key",
                name, ctx.prefix, bucket, exc,
            )
            n = 0
        logger.info("[rehearsal] %s: %d prior namespace object(s) in %s", name, n, bucket)
    sess._session.register(  # noqa: SLF001 — botocore session behind boto3's
        "before-parameter-build.s3", ctx._hook, unique_id=ctx._uid
    )
    _ACTIVE = ctx
    logger.warning(
        "[rehearsal] %s: %s is a REHEARSAL; every S3 write to %s is re-keyed "
        "under %s, no live key is written (alpha-engine-config-I12351)",
        name, entrypoint, ", ".join(sorted(ISOLATED_BUCKETS)), ctx.prefix,
    )
    return ctx


def deactivate(session: Any = None) -> None:
    """Undo :func:`activate` (tests; a process normally keeps it to exit)."""
    global _ACTIVE
    ctx = _ACTIVE
    _ACTIVE = None
    if ctx is None:
        return
    import boto3

    sess = session or boto3._get_default_session()  # noqa: SLF001
    sess._session.unregister("before-parameter-build.s3", unique_id=ctx._uid)  # noqa: SLF001


__all__ = [
    "ISOLATED_BUCKETS",
    "NAMESPACE_ROOT",
    "PASSTHROUGH_PREFIXES",
    "RehearsalContext",
    "RehearsalLiveWriteRefused",
    "activate",
    "active",
    "deactivate",
    "namespace_prefix",
    "rehearsal_name",
]
