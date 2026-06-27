"""Consolidated weekly Backtest+Eval digest — one thin email + console deep-link.

The two former emails (the simulation email from backtest.py and the evaluator
email from evaluate.py) are folded into ONE digest sent by evaluate.py, which
deep-links to the console Analysis page for the full detail (mirroring the EOD
and model-zoo digest patterns).

Pins:
  - build_digest renders a SHORT executive summary, reusing the same section
    builders as build_report, and omits absent sections cleanly.
  - send_digest_email is thin: a console deep-link (…/analysis?date=…) in both
    bodies + links to BOTH report.md artifacts, keyed by run_date.
  - the slug equals the dashboard's pinned url_path ("analysis").
"""
# emailer imports nousergon_lib.email_sender. In CI the real lib is installed;
# in a bare local venv it may be absent, so provide a stub BEFORE importing
# emailer (the tests monkeypatch emailer.send_email anyway, so SMTP/SES is never
# exercised). The stub is a no-op when the real lib is present.
try:  # pragma: no cover - import-availability shim
    import nousergon_lib.email_sender  # noqa: F401
except Exception:  # noqa: BLE001
    import sys
    import types

    _pkg = sys.modules.setdefault("nousergon_lib", types.ModuleType("nousergon_lib"))
    # Mark it a package so other modules' `nousergon_lib.<x>` imports fail as a
    # normal ModuleNotFound (not "is not a package"), keeping test isolation.
    if not hasattr(_pkg, "__path__"):
        _pkg.__path__ = []  # type: ignore[attr-defined]
    _mod = types.ModuleType("nousergon_lib.email_sender")
    _mod.send_email = lambda *a, **k: None
    sys.modules["nousergon_lib.email_sender"] = _mod

import emailer  # noqa: E402
from reporter import build_digest  # noqa: E402


def test_analysis_report_url_and_slug():
    assert emailer.ANALYSIS_SLUG == "analysis"
    assert (
        emailer.analysis_report_url("2026-06-26")
        == "https://console.nousergon.ai/analysis?date=2026-06-26"
    )
    # Override is honored (tests / non-prod consoles).
    assert emailer.analysis_report_url(
        "2026-06-26", "https://stage.example.com/"
    ) == "https://stage.example.com/analysis?date=2026-06-26"


def test_build_digest_what_changed_and_completeness():
    md = build_digest(
        "2026-06-26",
        weight_result={"status": "ok", "apply_result": {"applied": True}},
        regression_result=None,
        completeness={"ok": 20, "degraded": 1, "skipped": 2, "error": 0, "total": 23},
        degraded_modules=["macro_eval"],
    )
    assert "Weekly Backtest + Evaluation Digest" in md
    assert "2026-06-26" in md
    assert "PROMOTED" in md                       # what-changed reused builder
    assert "Evaluator Completeness" in md
    assert "macro_eval" in md
    # Thin: the digest is an executive summary, not the full 21-section report.
    assert len(md) < 4000


def test_build_digest_omits_absent_sections():
    md = build_digest("2026-06-26")               # no inputs at all
    assert "Weekly Backtest + Evaluation Digest" in md
    assert "Evaluator Completeness" not in md      # omitted when absent


def test_send_digest_email_is_thin_with_console_and_report_links(monkeypatch):
    captured = {}

    def _fake_send(subject, plain, *, recipients, html, sender, region):
        captured.update(subject=subject, plain=plain, html=html,
                        recipients=recipients, sender=sender)

    monkeypatch.setattr(emailer, "send_email", _fake_send)
    emailer.send_digest_email(
        "2026-06-26", "# Digest\n\nGrade: A", "s@x", ["o@x"],
        status="ok", s3_bucket="alpha-engine-research",
    )
    assert "Backtest+Eval | 2026-06-26 | results ready" in captured["subject"]
    url = "https://console.nousergon.ai/analysis?date=2026-06-26"
    assert url in captured["plain"]
    assert f'href="{url}"' in captured["html"]
    # Both full report.md artifacts are one click away.
    assert "backtest/2026-06-26/report.md" in captured["html"]
    assert "evaluation/2026-06-26/report.md" in captured["html"]
    assert captured["recipients"] == ["o@x"]
