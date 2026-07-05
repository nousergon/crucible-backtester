"""Dual-channel ops alerts for backtester (SNS + flow-doctor forum topics).

Migration arc: config#1740 T3 / config#1749 — retire raw ``telegram=True``
fan-out to Telegram General.
"""

from __future__ import annotations

import logging
from typing import Any

from nousergon_lib.logging import get_flow_doctor

logger = logging.getLogger(__name__)


def _normalize_flow_doctor_severity(severity: str) -> str:
    normalized = severity.lower()
    if normalized == "warn":
        return "warning"
    return normalized


def publish_ops_alert(
    message: str,
    *,
    severity: str,
    source: str,
    dedup_key: str | None = None,
    dedup_window_min: int | None = None,
) -> Any:
    """SNS via ``nousergon_lib.alerts.publish(telegram=False)`` + Telegram via flow-doctor."""
    from nousergon_lib import alerts as _alerts

    kwargs: dict[str, Any] = {
        "message": message,
        "severity": severity,
        "source": source,
        "sns": True,
        "telegram": False,
    }
    if dedup_key is not None:
        kwargs["dedup_key"] = dedup_key
    if dedup_window_min is not None:
        kwargs["dedup_window_min"] = dedup_window_min

    result = _alerts.publish(**kwargs)
    fd = get_flow_doctor()
    if fd is None:
        return result
    try:
        subject = message.split("\n", 1)[0].replace("*", "").strip()
        if not subject:
            subject = f"Backtester alert [{severity.upper()}]"
        fd.notify_event(
            subject,
            body=message,
            severity=_normalize_flow_doctor_severity(severity),
            dedup_key=dedup_key or subject,
            context={"source": source},
        )
    except Exception as exc:
        logger.warning(
            "flow-doctor notify_event failed for ops alert (%s): %s — SNS already sent",
            source,
            exc,
        )
    return result
