"""Which kind of pipeline run is this process part of (alpha-engine-config-I11505)?

The weekly-SF rehearsal ``rehearsal-2026-09-23-2`` ran EvaluatorOptimize with
cutover ON and wrote production ``config/executor_params.json``. Only two
timestamps changed that time, but the next time a promoting overlay carries a
real value, a rehearsal would change what the paper trader runs. Nothing in
the evaluator knew it was a rehearsal: the SF's ``pipeline_role`` never
reaches the spot instance.

What DOES reach it is the Step Functions execution name. Every weekly-SF spot
stage runs under ``krepis.ssm_log_capture --correlation-id
$$.Execution.Name``, which exports the id to its child as ``RUN_TOKEN``. The
launcher (``infrastructure/_spot_common.sh::spot_common_build_env_source``)
now forwards ``RUN_TOKEN`` onto the spot. Rehearsal executions are named
``rehearsal-<date>-<n>`` by the rehearsal launcher, so the prefix is the
signal. ``AE_PIPELINE_ROLE=rehearsal`` is the explicit override for a hand
run.

A rehearsal must never write live config. The entrypoints turn it into
``--freeze`` (the existing "no optimizer S3 config writes" mode), and the two
live-write sites that ``--freeze`` does not reach call
:func:`refuse_live_config_write_in_rehearsal` as a backstop:
``optimizer.assembler._cutover_apply`` and
``optimizer.predictor_optimizer.apply_recommendations``.

Watch-reruns are deliberately NOT covered here. A watch-rerun is the recovery
path for a failed real weekly run, and it may be the only run that promotes
that week's config. Whether it should also be barred from cutover is an
operator decision recorded on the issue, not something to infer here.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

logger = logging.getLogger(__name__)

#: Env var ``krepis.ssm_log_capture`` sets to the SF execution name.
RUN_TOKEN_ENV = "RUN_TOKEN"
#: Explicit override for a hand-launched rehearsal.
PIPELINE_ROLE_ENV = "AE_PIPELINE_ROLE"
#: Execution-name prefix of weekly-SF rehearsals.
REHEARSAL_PREFIX = "rehearsal-"


class RehearsalLiveWriteRefused(RuntimeError):
    """A rehearsal run reached a live-config write. Raised, never swallowed:
    the write must not happen, and the run should say so loudly."""


def rehearsal_reason(env: Mapping[str, str] | None = None) -> str | None:
    """Why this process is a rehearsal, or ``None`` when it is not."""
    env = os.environ if env is None else env
    role = (env.get(PIPELINE_ROLE_ENV) or "").strip().lower()
    if role == "rehearsal":
        return f"{PIPELINE_ROLE_ENV}=rehearsal"
    token = (env.get(RUN_TOKEN_ENV) or "").strip()
    if token.startswith(REHEARSAL_PREFIX):
        return f"{RUN_TOKEN_ENV}={token}"
    return None


def freeze_if_rehearsal(args, *, entrypoint: str, env: Mapping[str, str] | None = None) -> str | None:
    """Force ``args.freeze = True`` for a rehearsal run. Returns the reason
    (or ``None``). Called once by ``evaluate.py`` / ``backtest.py`` right
    after argument parsing, before anything can write."""
    reason = rehearsal_reason(env)
    if reason and not getattr(args, "freeze", False):
        args.freeze = True
        logger.warning(
            "[%s] REHEARSAL run (%s): forcing --freeze — no live config key "
            "may be written by a rehearsal (alpha-engine-config-I11505)",
            entrypoint, reason,
        )
    elif reason:
        logger.info("[%s] rehearsal run (%s): --freeze already set", entrypoint, reason)
    return reason


def refuse_live_config_write_in_rehearsal(
    key: str, *, env: Mapping[str, str] | None = None,
) -> None:
    """Backstop at a live-config write site. Raises
    :class:`RehearsalLiveWriteRefused` when this is a rehearsal."""
    reason = rehearsal_reason(env)
    if reason:
        raise RehearsalLiveWriteRefused(
            f"refusing to write live config key {key!r} from a rehearsal run "
            f"({reason}) — alpha-engine-config-I11505"
        )
