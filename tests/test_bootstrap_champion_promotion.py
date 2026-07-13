"""Tests for the one-shot operator bootstrap script (config#2364/#2367)."""
from __future__ import annotations

import sys

import pytest

import bootstrap_champion_promotion as bootstrap


def test_build_bootstrap_audit_shape():
    audit = bootstrap._build_bootstrap_audit("2026-07-13", "agentic", "scanner_predictor_direct")
    # contracts/producer_champion_audit.schema.json (v1) required fields.
    for field in (
        "schema_version", "date", "outcome", "champion_before", "champion_after",
        "challenger_matured_cohorts", "sn_lift_vs_champion", "consecutive_wins",
        "cooldown_until", "blocked_by",
    ):
        assert field in audit
    assert audit["schema_version"] == 1
    assert audit["outcome"] == "promoted"
    assert audit["champion_before"] == "agentic"
    assert audit["champion_after"] == "scanner_predictor_direct"
    assert audit["challenger_matured_cohorts"] == 0
    assert audit["sn_lift_vs_champion"] is None
    assert audit["consecutive_wins"] == 0
    assert audit["blocked_by"] is None
    assert audit["cooldown_until"] == "2026-07-27"  # +2 weeks, gate (e)
    assert audit["promotion_source"] == "operator_bootstrap"


def test_refuses_agentic_target(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "bootstrap_champion_promotion.py", "--champion", "agentic", "--run-date", "2026-07-13",
    ])
    with pytest.raises(SystemExit) as exc:
        bootstrap.main()
    assert exc.value.code == 1
    assert "Refusing" in capsys.readouterr().err


def test_refuses_when_pointer_already_exists(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "bootstrap_champion_promotion.py", "--run-date", "2026-07-13",
    ])
    monkeypatch.setattr(
        bootstrap, "read_champion_pointer",
        lambda bucket: {"champion": "scanner_predictor_direct", "promotion_source": "gate_engine"},
    )
    with pytest.raises(SystemExit) as exc:
        bootstrap.main()
    assert exc.value.code == 1
    assert "already exists" in capsys.readouterr().err


def test_dry_run_never_calls_audit_writer(monkeypatch):
    calls = {"pointer": None, "audit": None}
    monkeypatch.setattr(sys, "argv", [
        "bootstrap_champion_promotion.py", "--run-date", "2026-07-13",
    ])
    monkeypatch.setattr(bootstrap, "read_champion_pointer", lambda bucket: None)
    monkeypatch.setattr(
        bootstrap, "write_champion_pointer",
        lambda **kw: calls.__setitem__("pointer", kw) or {"champion": kw["champion"]},
    )
    monkeypatch.setattr(
        bootstrap, "write_champion_audit",
        lambda *a, **kw: calls.__setitem__("audit", (a, kw)),
    )
    bootstrap.main()
    assert calls["pointer"]["upload"] is False
    assert calls["pointer"]["champion"] == "scanner_predictor_direct"
    assert calls["pointer"]["promotion_source"] == "operator_bootstrap"
    assert calls["audit"] is None  # never called in dry-run


def test_upload_flow_calls_both_writers(monkeypatch):
    calls = {"pointer": None, "audit": None}
    monkeypatch.setattr(sys, "argv", [
        "bootstrap_champion_promotion.py", "--run-date", "2026-07-13", "--upload",
    ])
    monkeypatch.setattr(bootstrap, "read_champion_pointer", lambda bucket: None)
    monkeypatch.setattr(
        bootstrap, "write_champion_pointer",
        lambda **kw: calls.__setitem__("pointer", kw) or {"champion": kw["champion"]},
    )
    monkeypatch.setattr(
        bootstrap, "write_champion_audit",
        lambda *a, **kw: calls.__setitem__("audit", (a, kw)),
    )
    bootstrap.main()
    assert calls["pointer"]["upload"] is True
    assert calls["audit"] is not None
    args, kwargs = calls["audit"]
    bucket, run_date, audit = args
    assert run_date == "2026-07-13"
    assert audit["outcome"] == "promoted"
    assert audit["champion_after"] == "scanner_predictor_direct"
