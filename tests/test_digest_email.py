"""Backtest + Eval digest emails — thin, console-linked, one per task per day.

The Backtester and the Evaluator are SEPARATE Saturday-SF tasks and each sends
its OWN thin digest (backtest.py / evaluate.py respectively) that deep-links to
the console Analysis page for the full detail (mirroring the EOD and model-zoo
digest patterns) — they are intentionally NOT bundled into one email.

Pins:
  - build_digest renders a SHORT executive summary, reusing the same section
    builders as build_report, and omits absent sections cleanly.
  - send_digest_email is thin: a console deep-link (…/analysis?date=…) +
    a link to ITS OWN report.md artifact, keyed by run_date.
  - the slug equals the dashboard's pinned url_path ("analysis").
  - send_digest_email forwards dedup_key/dedup_window_min through to
    send_email's S3-marker dedup (config#2291) so multiple SF-state
    invocations for the same trading_day collapse to one actual send.
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


def test_build_digest_evaluator_what_changed_and_completeness():
    # The Evaluator builds its OWN digest — report card + what-changed + counts.
    md = build_digest(
        "2026-06-26",
        title="Evaluation Digest",
        weight_result={"status": "ok", "apply_result": {"applied": True}},
        regression_result=None,
        completeness={"ok": 20, "degraded": 1, "skipped": 2, "error": 0, "total": 23},
        degraded_modules=["macro_eval"],
    )
    assert "Evaluation Digest" in md
    assert "2026-06-26" in md
    assert "PROMOTED" in md                       # what-changed reused builder
    assert "Evaluator Completeness" in md
    assert "macro_eval" in md
    # Thin: the digest is an executive summary, not the full 21-section report.
    assert len(md) < 4000


def test_build_digest_title_is_parameterized_and_omits_absent_sections():
    md = build_digest("2026-06-26", title="Backtest Digest")  # no section inputs
    assert "Backtest Digest" in md
    assert "Evaluator Completeness" not in md      # omitted when absent


def _capture_send(monkeypatch):
    captured = {}

    def _fake_send(subject, plain, *, recipients, html, sender, region,
                    dedup_key=None, dedup_window_min=None):
        captured.update(subject=subject, plain=plain, html=html,
                        recipients=recipients, sender=sender,
                        dedup_key=dedup_key, dedup_window_min=dedup_window_min)

    monkeypatch.setattr(emailer, "send_email", _fake_send)
    return captured


def test_send_digest_email_backtester_is_thin_with_console_and_report_link(monkeypatch):
    captured = _capture_send(monkeypatch)
    emailer.send_digest_email(
        "2026-06-26", "# Digest\n\nDeployed: +1.2%", "s@x", ["o@x"],
        product_name="Backtester", report_prefix="backtest",
        status="ok", s3_bucket="alpha-engine-research",
    )
    assert "Alpha Engine Backtester | 2026-06-26 | results ready" in captured["subject"]
    url = "https://console.nousergon.ai/analysis?date=2026-06-26"
    assert url in captured["plain"]
    assert f'href="{url}"' in captured["html"]
    # Links to ITS OWN report.md only (not the evaluator's) — the two are separate.
    assert "backtest/2026-06-26/report.md" in captured["html"]
    assert "evaluation/2026-06-26/report.md" not in captured["html"]
    assert captured["recipients"] == ["o@x"]


def test_send_digest_email_evaluator_is_separate_with_own_subject_and_report(monkeypatch):
    captured = _capture_send(monkeypatch)
    emailer.send_digest_email(
        "2026-06-26", "# Digest\n\nGrade: A", "s@x", ["o@x"],
        product_name="Evaluator", report_prefix="evaluation",
        status="ok", s3_bucket="alpha-engine-research",
    )
    assert "Alpha Engine Evaluator | 2026-06-26 | results ready" in captured["subject"]
    # Separate task → links to the evaluation report.md, not the backtest one.
    assert "evaluation/2026-06-26/report.md" in captured["html"]
    assert "backtest/2026-06-26/report.md" not in captured["html"]


# ── dedup passthrough (config#2291) ─────────────────────────────────────────
#
# The Saturday SF's PredictorBacktest + PortfolioOptimizerBacktest + main-
# backtest states each call backtest.py --upload independently for the same
# trading_day, and each reaches send_digest_email — the fix is a dedup_key
# stable across those states so send_email's S3-marker dedup collapses them
# to one actual send. These tests pin the passthrough contract only (the
# marker mechanism itself is krepis' — see krepis/tests/test_email_sender.py).


def test_send_digest_email_forwards_dedup_key_when_given(monkeypatch):
    captured = _capture_send(monkeypatch)
    emailer.send_digest_email(
        "2026-06-26", "# Digest", "s@x", ["o@x"],
        product_name="Backtester", report_prefix="backtest", status="ok",
        dedup_key="backtester-digest:2026-06-26",
    )
    assert captured["dedup_key"] == "backtester-digest:2026-06-26"


def test_send_digest_email_dedup_key_defaults_to_none(monkeypatch):
    """Legacy callers (no dedup_key passed) behave unchanged — no dedup."""
    captured = _capture_send(monkeypatch)
    emailer.send_digest_email(
        "2026-06-26", "# Digest", "s@x", ["o@x"],
        product_name="Backtester", report_prefix="backtest", status="ok",
    )
    assert captured["dedup_key"] is None


def test_send_digest_email_omits_dedup_window_min_when_unset(monkeypatch):
    """When the caller doesn't pass dedup_window_min, send_digest_email must
    NOT forward an explicit None — send_email's own default (24h) means
    something different from an explicit None ("forever"), so omitting the
    kwarg lets send_email apply its default rather than silently downgrading
    every un-parameterized caller to "forever"."""
    captured = {}

    def _fake_send(subject, plain, *, recipients, html, sender, region, **kwargs):
        captured["got_dedup_window_min_kwarg"] = "dedup_window_min" in kwargs
        captured["dedup_key"] = kwargs.get("dedup_key")

    monkeypatch.setattr(emailer, "send_email", _fake_send)
    emailer.send_digest_email(
        "2026-06-26", "# Digest", "s@x", ["o@x"],
        product_name="Backtester", report_prefix="backtest", status="ok",
        dedup_key="k",
    )
    assert captured["got_dedup_window_min_kwarg"] is False


def test_send_digest_email_forwards_explicit_dedup_window_min(monkeypatch):
    captured = _capture_send(monkeypatch)
    emailer.send_digest_email(
        "2026-06-26", "# Digest", "s@x", ["o@x"],
        product_name="Backtester", report_prefix="backtest", status="ok",
        dedup_key="k", dedup_window_min=60,
    )
    assert captured["dedup_window_min"] == 60
