"""
analysis/retrain_alert.py — Phase 5: Predictor retraining alerts.

Evaluates trigger conditions from Phase 2 (production health, calibration)
and Phase 3 (feature drift) outputs. When any condition fires, sends an
alert email explaining why retraining is recommended. Does NOT trigger
retraining automatically — the weekly cadence continues as-is.

Trigger conditions (any one sufficient):
  1. Production IC degradation: rolling 30d IC < 50% of training IC
  2. Feature drift: >20% of features show sign flips
  3. Calibration breakdown: overall ECE > 0.10
  4. Regime shift: market regime changed and regime-specific IC is negative
  5. Mode collapse: >75% of predictions are a single direction

Reads from:
  - predictor/metrics/production_health.json (S3, written by Phase 2a)
  - predictor/metrics/feature_drift.json (S3, written by Phase 3)
  - predictor/metrics/calibration_validation.json (S3, written by Phase 2b)

Writes to:
  - predictor/alerts/retrain_alert_latest.json (S3, for dashboard)
  - predictor/alerts/history.jsonl (S3, append-only audit trail)
"""

from __future__ import annotations

import json
import logging
import smtplib
from datetime import date, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import boto3

from alpha_engine_lib.secrets import get_secret

log = logging.getLogger(__name__)

# Thresholds
_IC_DEGRADATION_RATIO = 0.50
_DRIFT_FRACTION_TRIGGER = 0.20
_ECE_THRESHOLD = 0.10
_MODE_COLLAPSE_THRESHOLD = 0.75
_MIN_DAYS_BETWEEN_ALERTS = 2  # suppress duplicate alerts from reruns/retries
_CALIBRATOR_GRACE_DAYS = 30  # skip calibration_breakdown in the N days after a new calibrator deploys


def _calibrator_within_grace(calibration: dict, run_date: str | None = None) -> bool:
    """True when the calibrator is too recent for ECE to be meaningful.

    The predictor fits isotonic calibration on OOF predictions during training
    and ships it to S3. For the first ``_CALIBRATOR_GRACE_DAYS`` after a new
    calibrator lands, the ``predictor_outcomes`` table contains a mix of
    pre-calibrator and post-calibrator confidence semantics — ECE over that
    window is structurally noisy and does not indicate real miscalibration.

    Absent ``calibrator_deployed_at`` → no grace period (legacy behavior).
    """
    deployed_at = calibration.get("calibrator_deployed_at")
    if not deployed_at:
        return False
    try:
        deployed_dt = datetime.fromisoformat(str(deployed_at).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return False
    now = datetime.fromisoformat(run_date) if run_date else datetime.utcnow()
    if deployed_dt.tzinfo is not None:
        deployed_dt = deployed_dt.replace(tzinfo=None)
    age_days = (now - deployed_dt).days
    return 0 <= age_days < _CALIBRATOR_GRACE_DAYS


def evaluate_retrain_triggers(
    production_health: dict | None,
    feature_drift: dict | None,
    calibration: dict | None,
) -> dict:
    """
    Evaluate all trigger conditions and return alert details.

    Returns dict with:
      - triggered: bool
      - reasons: list of {trigger, detail, severity} dicts
      - summary: human-readable summary string
    """
    reasons = []

    # ── Trigger 1: IC degradation ────────────────────────────────────────
    if production_health and production_health.get("degradation_flag"):
        ic_ratio = production_health.get("ic_ratio", 0)
        reasons.append({
            "trigger": "ic_degradation",
            "detail": (
                f"Production IC has dropped to {ic_ratio:.0%} of training IC "
                f"(rolling={production_health.get('rolling_30d_ic', 'N/A')}, "
                f"training={production_health.get('training_ic', 'N/A')})"
            ),
            "severity": "high",
        })

    # ── Trigger 2: Feature drift ─────────────────────────────────────────
    if feature_drift and feature_drift.get("drift_fraction", 0) > _DRIFT_FRACTION_TRIGGER:
        n_drifted = len(feature_drift.get("drifted_features", []))
        total = feature_drift.get("total_features", 0)
        drifted_names = [f["feature"] for f in feature_drift.get("drifted_features", [])[:5]]
        reasons.append({
            "trigger": "feature_drift",
            "detail": (
                f"{n_drifted}/{total} features have drifted "
                f"({feature_drift.get('drift_fraction', 0):.0%}): "
                f"{', '.join(drifted_names)}"
            ),
            "severity": "high",
        })

    # ── Trigger 3: Calibration breakdown ─────────────────────────────────
    if calibration and calibration.get("overall_ece") is not None:
        ece = calibration["overall_ece"]
        if ece > _ECE_THRESHOLD:
            if _calibrator_within_grace(calibration):
                log.info(
                    "Calibration_breakdown suppressed: calibrator deployed_at=%s within %d-day grace window",
                    calibration.get("calibrator_deployed_at"),
                    _CALIBRATOR_GRACE_DAYS,
                )
            else:
                reasons.append({
                    "trigger": "calibration_breakdown",
                    "detail": (
                        f"Expected Calibration Error is {ece:.3f} "
                        f"(threshold: {_ECE_THRESHOLD}) — model confidence "
                        f"no longer matches actual hit rates"
                    ),
                    "severity": "medium",
                })

    # ── Trigger 4: Regime shift with negative IC ─────────────────────────
    if production_health:
        regime_ic = production_health.get("regime_ic", {})
        # Check if any active regime has negative IC
        for regime, ic in regime_ic.items():
            if ic is not None and ic < 0:
                reasons.append({
                    "trigger": "regime_negative_ic",
                    "detail": (
                        f"Model has negative IC ({ic:.4f}) in "
                        f"'{regime}' market regime"
                    ),
                    "severity": "medium",
                })
                break  # one regime trigger is enough

    # ── Trigger 5: Mode collapse ─────────────────────────────────────────
    if production_health and production_health.get("mode_collapse_flag"):
        dist = production_health.get("prediction_distribution", {})
        dominant = max(dist.items(), key=lambda x: x[1]) if dist else ("?", 0)
        reasons.append({
            "trigger": "mode_collapse",
            "detail": (
                f"Model is predicting {dominant[0]} {dominant[1]:.0%} of "
                f"the time — predictions have lost discriminative power"
            ),
            "severity": "high",
        })

    triggered = len(reasons) > 0

    if triggered:
        high_count = sum(1 for r in reasons if r["severity"] == "high")
        summary = (
            f"RETRAIN RECOMMENDED: {len(reasons)} trigger(s) fired "
            f"({high_count} high severity). "
            + " | ".join(r["trigger"] for r in reasons)
        )
    else:
        summary = "Model health OK — no retraining triggers fired"

    return {
        "date": str(date.today()),
        "triggered": triggered,
        "n_triggers": len(reasons),
        "reasons": reasons,
        "summary": summary,
    }


def send_retrain_alert(
    alert: dict,
    config: dict,
    bucket: str,
) -> dict:
    """
    Send alert email and write alert to S3 if triggers fired.

    Suppresses duplicate alerts within _MIN_DAYS_BETWEEN_ALERTS.
    """
    if not alert.get("triggered"):
        log.info("Retrain alert: no triggers — skipping")
        return {"sent": False, "reason": "no_triggers"}

    # Suppress duplicate alerts from reruns/retries (2-day window)
    if _should_suppress(bucket):
        log.info("Retrain alert: suppressed (alerted within %d days)", _MIN_DAYS_BETWEEN_ALERTS)
        return {"sent": False, "reason": "suppressed"}

    # Write alert to S3
    _write_alert_to_s3(alert, bucket)

    # Send email
    sender = config.get("email_sender")
    recipients = config.get("email_recipients", [])
    if not sender or not recipients:
        log.warning("Retrain alert: no email config — alert written to S3 only")
        return {"sent": False, "reason": "no_email_config", "s3_written": True}

    subject = _build_subject(alert)
    html_body = _build_html_body(alert)
    plain_body = _build_plain_body(alert)

    region = config.get("aws_region", "us-east-1")
    gmail_pw = get_secret("GMAIL_APP_PASSWORD", required=False, default="")

    try:
        if gmail_pw:
            _send_smtp(subject, plain_body, html_body, sender, recipients, gmail_pw)
        else:
            _send_ses(subject, plain_body, html_body, sender, recipients, region)
        log.info("Retrain alert sent: %s", subject)
        return {"sent": True, "subject": subject, "n_triggers": alert["n_triggers"]}
    except Exception as exc:
        log.error("Retrain alert email failed: %s", exc)
        return {"sent": False, "reason": f"email_failed: {exc}", "s3_written": True}


# ── S3 persistence ───────────────────────────────────────────────────────────

def _write_alert_to_s3(alert: dict, bucket: str) -> None:
    """Write alert to latest.json and append to history.jsonl."""
    s3 = boto3.client("s3")
    body = json.dumps(alert, indent=2, default=str)

    try:
        s3.put_object(
            Bucket=bucket,
            Key="predictor/alerts/retrain_alert_latest.json",
            Body=body.encode(),
            ContentType="application/json",
        )
    except Exception as exc:
        log.warning("Failed to write retrain alert to S3: %s", exc)

    # Append to history (JSONL)
    try:
        line = json.dumps(alert, default=str) + "\n"
        # Read existing history, append
        existing = ""
        try:
            obj = s3.get_object(Bucket=bucket, Key="predictor/alerts/history.jsonl")
            existing = obj["Body"].read().decode()
        except Exception:
            pass
        s3.put_object(
            Bucket=bucket,
            Key="predictor/alerts/history.jsonl",
            Body=(existing + line).encode(),
            ContentType="application/jsonlines",
        )
    except Exception as exc:
        log.warning("Failed to append retrain alert history: %s", exc)


def _should_suppress(bucket: str) -> bool:
    """Check if we already sent an alert within the suppression window."""
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key="predictor/alerts/retrain_alert_latest.json")
        last_alert = json.loads(obj["Body"].read())
        last_date = datetime.strptime(last_alert["date"], "%Y-%m-%d").date()
        days_since = (date.today() - last_date).days
        return days_since < _MIN_DAYS_BETWEEN_ALERTS
    except Exception:
        return False


# ── Email formatting ─────────────────────────────────────────────────────────

def _build_subject(alert: dict) -> str:
    high_count = sum(1 for r in alert["reasons"] if r["severity"] == "high")
    severity = "HIGH" if high_count > 0 else "MEDIUM"
    return (
        f"Alpha Engine | Predictor Retrain Alert [{severity}] | "
        f"{alert['n_triggers']} trigger(s) | {alert['date']}"
    )


def _build_html_body(alert: dict) -> str:
    rows = ""
    for r in alert["reasons"]:
        color = "#d32f2f" if r["severity"] == "high" else "#f57c00"
        rows += (
            f'<tr>'
            f'<td style="color:{color};font-weight:bold">{r["severity"].upper()}</td>'
            f'<td><b>{r["trigger"]}</b></td>'
            f'<td>{r["detail"]}</td>'
            f'</tr>'
        )

    return f"""\
<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
  body {{ font-family: 'Courier New', monospace; font-size: 13px;
         color: #222; max-width: 720px; margin: 0 auto; padding: 20px; }}
  h1 {{ font-size: 16px; color: #d32f2f; border-bottom: 2px solid #d32f2f; padding-bottom: 6px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
  th {{ background: #f0f0f0; }}
  .summary {{ background: #fff3e0; padding: 12px; border-left: 4px solid #f57c00; margin: 12px 0; }}
  .foot {{ margin-top: 28px; font-size: 11px; color: #888;
           border-top: 1px solid #ccc; padding-top: 8px; }}
</style></head><body>
<h1>Predictor Retrain Alert</h1>
<div class="summary">{alert['summary']}</div>
<table>
<tr><th>Severity</th><th>Trigger</th><th>Detail</th></tr>
{rows}
</table>
<p>The weekly training cadence will retrain the model on the next Saturday
pipeline run. Review these triggers to determine if an earlier manual
retrain is warranted.</p>
<div class="foot">Alpha Engine Backtester | {alert['date']}</div>
</body></html>"""


def _build_plain_body(alert: dict) -> str:
    lines = [
        "PREDICTOR RETRAIN ALERT",
        "=" * 40,
        "",
        alert["summary"],
        "",
    ]
    for r in alert["reasons"]:
        lines.append(f"[{r['severity'].upper()}] {r['trigger']}")
        lines.append(f"  {r['detail']}")
        lines.append("")
    lines.append("The weekly training cadence will retrain the model on the")
    lines.append("next Saturday pipeline run.")
    return "\n".join(lines)


# ── Email transport (same pattern as emailer.py) ─────────────────────────────

def _send_smtp(subject, plain, html, sender, recipients, gmail_pw):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, gmail_pw)
        server.sendmail(sender, recipients, msg.as_string())


def _send_ses(subject, plain, html, sender, recipients, region):
    ses = boto3.client("ses", region_name=region)
    ses.send_email(
        Source=sender,
        Destination={"ToAddresses": recipients},
        Message={
            "Subject": {"Data": subject, "Charset": "UTF-8"},
            "Body": {
                "Text": {"Data": plain, "Charset": "UTF-8"},
                "Html": {"Data": html, "Charset": "UTF-8"},
            },
        },
    )
