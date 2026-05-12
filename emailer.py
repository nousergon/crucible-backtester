"""
emailer.py — send weekly backtest report via Gmail SMTP (primary) or AWS SES (fallback).

Matches the style and config conventions of executor/eod_emailer.py.
Sender and recipients come from config.yaml (email_sender / email_recipients).

Gmail path: set GMAIL_APP_PASSWORD env var (16-char App Password).
SES fallback: used automatically when GMAIL_APP_PASSWORD is absent.
"""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import boto3
from botocore.exceptions import ClientError

from alpha_engine_lib.secrets import get_secret

logger = logging.getLogger(__name__)

_HTML = """\
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8">
<style>
  body  {{ font-family: 'Courier New', monospace; font-size: 13px; line-height: 1.6;
           color: #222; max-width: 720px; margin: 0 auto; padding: 20px; }}
  h1   {{ font-size: 16px; border-bottom: 2px solid #555; padding-bottom: 6px; }}
  h2   {{ font-size: 14px; border-bottom: 1px solid #ccc; padding-bottom: 4px; margin-top: 20px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 8px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 10px; text-align: left; }}
  th   {{ background: #f0f0f0; }}
  blockquote {{ margin: 4px 0 4px 12px; color: #666; border-left: 3px solid #ccc; padding-left: 8px; }}
  pre  {{ background: #f8f8f8; padding: 10px; font-size: 12px; overflow-x: auto; }}
  .foot {{ margin-top: 28px; font-size: 11px; color: #888;
            border-top: 1px solid #ccc; padding-top: 8px; }}
</style>
</head>
<body>
{body}
<div class="foot">Alpha Engine {product_name} | {date}{s3_link}</div>
</body>
</html>
"""


def send_report_email(
    run_date: str,
    report_md: str,
    status: str,
    sender: str,
    recipients: list[str],
    s3_bucket: str | None = None,
    s3_prefix: str = "backtest",
    region: str = "us-east-1",
    product_name: str = "Backtester",
) -> None:
    """
    Send the weekly backtest/evaluator report via SES.

    Args:
        run_date:    Date string for subject line and footer.
        report_md:   Markdown report string from reporter.build_report().
        status:      "ok" | "insufficient_data" | "error" — shown in subject.
        sender:      SES-verified from address.
        recipients:  List of to addresses.
        s3_bucket:   If set, include S3 link to report in footer.
        s3_prefix:   S3 prefix for report location (default "backtest").
        region:      AWS region for SES.
    """
    subject = _build_subject(run_date, status, product_name)
    html_body, plain_body = _build_body(run_date, report_md, s3_bucket, s3_prefix, product_name)

    gmail_pw = get_secret("GMAIL_APP_PASSWORD", required=False, default="")
    if gmail_pw:
        _send_via_smtp(subject, plain_body, html_body, sender, recipients, gmail_pw)
    else:
        _send_via_ses(subject, plain_body, html_body, sender, recipients, region)


def _send_via_smtp(
    subject: str,
    plain_body: str,
    html_body: str,
    sender: str,
    recipients: list[str],
    gmail_pw: str,
) -> None:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(plain_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, gmail_pw)
            server.sendmail(sender, recipients, msg.as_string())
        logger.info("Backtest report email sent via Gmail SMTP: '%s' → %s", subject, recipients)
    except Exception as exc:
        logger.warning("Gmail SMTP failed (%s) — falling back to SES", exc)
        _send_via_ses(subject, plain_body, html_body, sender, recipients, "us-east-1")


def _send_via_ses(
    subject: str,
    plain_body: str,
    html_body: str,
    sender: str,
    recipients: list[str],
    region: str,
) -> None:
    ses = boto3.client("ses", region_name=region)
    try:
        ses.send_email(
            Source=sender,
            Destination={"ToAddresses": recipients},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {
                    "Text": {"Data": plain_body, "Charset": "UTF-8"},
                    "Html": {"Data": html_body, "Charset": "UTF-8"},
                },
            },
        )
        logger.info("Backtest report email sent via SES: '%s' → %s", subject, recipients)
    except ClientError as e:
        logger.error("SES send failed: %s", e.response["Error"]["Message"])
    except Exception as e:
        logger.error("Email error: %s", e)


def _build_subject(run_date: str, status: str, product_name: str = "Backtester") -> str:
    label = {
        "ok": "results ready",
        "insufficient_data": "insufficient data (accumulating)",
        "db_not_found": "ERROR — research.db not found",
        "error": "ERROR",
    }.get(status, status)
    return f"Alpha Engine {product_name} | {run_date} | {label}"


def _build_body(
    run_date: str,
    report_md: str,
    s3_bucket: str | None,
    s3_prefix: str,
    product_name: str = "Backtester",
) -> tuple[str, str]:
    # Convert minimal markdown to HTML (tables, headers, blockquotes, hr)
    import re
    def _md_inline(text: str) -> str:
        """Convert **bold** and _italic_ to HTML inline."""
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)
        return text

    html_lines = []
    in_table = False
    in_code_block = False
    for line in report_md.splitlines():
        # Code block fences (```)
        if line.startswith("```"):
            if in_code_block:
                html_lines.append("</pre>")
                in_code_block = False
            else:
                in_code_block = True
                html_lines.append("<pre>")
            continue
        if in_code_block:
            html_lines.append(line)
            continue

        is_table_line = line.startswith("|")

        # Close table if we were in one and this line isn't a table row
        if in_table and not is_table_line:
            html_lines.append("</table>")
            in_table = False

        if line.startswith("### "):
            html_lines.append(f"<h3>{_md_inline(line[4:])}</h3>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{_md_inline(line[3:])}</h2>")
        elif line.startswith("# "):
            html_lines.append(f"<h1>{_md_inline(line[2:])}</h1>")
        elif line.startswith(">"):
            # Strip > and optional space
            content = line[1:].lstrip(" ") if len(line) > 1 else ""
            html_lines.append(f"<blockquote>{_md_inline(content)}</blockquote>")
        elif line.startswith("---"):
            html_lines.append("<hr>")
        elif is_table_line:
            if not in_table:
                _table_first_row = True
                html_lines.append('<table border="1" cellpadding="4" cellspacing="0" '
                                  'style="border-collapse:collapse; font-size:12px; '
                                  'border-color:#ddd; margin:8px 0;">')
                in_table = True
            else:
                _table_first_row = False
            row_html = _md_table_row(line, is_header=_table_first_row)
            if row_html:  # skip empty separator rows
                html_lines.append(row_html)
        elif line.startswith("_") and line.endswith("_"):
            html_lines.append(f"<p><em>{line.strip('_')}</em></p>")
        elif line.strip() == "":
            html_lines.append("")
        else:
            html_lines.append(f"<p>{_md_inline(line)}</p>")

    # Close any trailing table or code block
    if in_table:
        html_lines.append("</table>")
    if in_code_block:
        html_lines.append("</pre>")

    s3_link = ""
    if s3_bucket:
        url = f"https://{s3_bucket}.s3.amazonaws.com/{s3_prefix}/{run_date}/report.md"
        s3_link = f' | <a href="{url}">S3 report</a>'

    html_body = _HTML.format(
        body="\n".join(html_lines),
        date=run_date,
        s3_link=s3_link,
        product_name=product_name,
    )
    return html_body, report_md  # plain body is just the markdown


def _md_table_row(line: str, is_header: bool = False) -> str:
    """Convert a markdown table row to an HTML table row."""
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if all(set(c) <= set("-: ") for c in cells):
        return ""  # separator row
    tag = "th" if is_header else "td"
    style = ' style="background:#f5f5f5; font-weight:bold;"' if is_header else ""
    inner = "".join(f"<{tag}{style}>{c}</{tag}>" for c in cells)
    return f"<tr>{inner}</tr>"
