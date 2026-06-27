"""
emailer.py — build + send each task's thin weekly digest email.

``send_digest_email`` sends ONE task's thin executive-summary email that
deep-links to the console Analysis page for the full detail (mirroring the EOD
and model-zoo digest patterns). The Backtester and the Evaluator are SEPARATE
Saturday-SF tasks, so each sends its OWN digest (they are NOT bundled) — the
former full-markdown bodies are replaced by a thin summary + console link.
SMTP/SES dispatch is delegated to ``nousergon_lib.email_sender.send_email``
(L4356 chokepoint); this module owns the subject + HTML/MD body building only.
"""

from __future__ import annotations

import logging

from nousergon_lib.email_sender import send_email

logger = logging.getLogger(__name__)

# Deep-link target for the consolidated digest email → the console Analysis
# page (backtester + evaluation detail). The slug is pinned in
# crucible-dashboard app.py (url_path="analysis") and guarded by
# tests/test_analysis_page.py; the page honors ?date=YYYY-MM-DD keyed by the
# backtest run_date (the last completed trading day), so the link opens the
# exact run the digest describes.
DEFAULT_CONSOLE_BASE_URL = "https://console.nousergon.ai"
ANALYSIS_SLUG = "analysis"


def analysis_report_url(run_date: str, console_base_url: str | None = None) -> str:
    """Deep-link to the console Analysis page for ``run_date``."""
    base = (console_base_url or DEFAULT_CONSOLE_BASE_URL).rstrip("/")
    return f"{base}/{ANALYSIS_SLUG}?date={run_date}"

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


def _md_to_html(report_md: str) -> str:
    """Convert the minimal markdown subset (tables, headers, blockquotes, hr,
    bold/italic) used by the report builders to an HTML fragment. Shared by the
    full-report body and the consolidated digest body."""
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

    return "\n".join(html_lines)


def send_digest_email(
    run_date: str,
    digest_md: str,
    sender: str,
    recipients: list[str],
    *,
    product_name: str,
    report_prefix: str,
    status: str = "ok",
    console_base_url: str | None = None,
    s3_bucket: str | None = None,
    region: str = "us-east-1",
) -> None:
    """Send ONE task's thin digest email — a short executive summary that
    deep-links to the console Analysis page for the full detail (mirroring the
    EOD-email and model-zoo-digest patterns).

    The Backtester and the Evaluator are SEPARATE Saturday-SF tasks, so each
    sends its OWN digest via this function (they are NOT bundled): backtest.py
    calls it with ``product_name="Backtester"``, ``report_prefix="backtest"``;
    evaluate.py with ``product_name="Evaluator"``, ``report_prefix="evaluation"``.
    Both land on the same console Analysis page (which has both a Backtester and
    a Pipeline-Evaluation tab), keyed by run_date.

    Args:
        run_date:     The run_date (last completed trading day) — subject +
                      console deep-link + the S3 report.md link.
        digest_md:    Short executive-summary markdown from reporter.build_digest.
        product_name: "Backtester" | "Evaluator" — subject + footer label.
        report_prefix:S3 prefix of THIS task's report.md ("backtest"|"evaluation").
        status:       "ok" | "insufficient_data" | "error" — shown in subject.
        console_base_url: Override for the console base (tests); defaults to prod.
        s3_bucket:    When set, footer links to this task's full report.md.
    """
    url = analysis_report_url(run_date, console_base_url)
    label = {
        "ok": "results ready",
        "insufficient_data": "insufficient data (accumulating)",
        "error": "ERROR",
    }.get(status, status)
    subject = f"Alpha Engine {product_name} | {run_date} | {label}"

    cta_html = (
        f'<p style="font-size:14px;margin:0 0 16px;">&#9654; '
        f'<a href="{url}"><b>View the full {product_name} report on the '
        f'console</b></a></p>'
    )
    foot_links = f' | <a href="{url}">console</a>'
    plain_links = ""
    if s3_bucket:
        report_url = (
            f"https://{s3_bucket}.s3.amazonaws.com/{report_prefix}/{run_date}/report.md"
        )
        foot_links += f' | <a href="{report_url}">report.md</a>'
        plain_links = f"\nFull report: {report_url}\n"

    html_body = _HTML.format(
        body=cta_html + _md_to_html(digest_md),
        date=run_date,
        s3_link=foot_links,
        product_name=product_name,
    )
    plain_body = (
        f"View the full {product_name} report on the console:\n{url}\n\n"
        f"{digest_md}\n{plain_links}"
    )
    send_email(
        subject, plain_body,
        recipients=recipients, html=html_body,
        sender=sender, region=region,
    )


def _md_table_row(line: str, is_header: bool = False) -> str:
    """Convert a markdown table row to an HTML table row."""
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if all(set(c) <= set("-: ") for c in cells):
        return ""  # separator row
    tag = "th" if is_header else "td"
    style = ' style="background:#f5f5f5; font-weight:bold;"' if is_header else ""
    inner = "".join(f"<{tag}{style}>{c}</{tag}>" for c in cells)
    return f"<tr>{inner}</tr>"
