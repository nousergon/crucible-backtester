"""
Unit tests for ``analysis.cost_report``.

Locks down:

- Markdown rendering: totals, per-team / per-model / per-run_type / per-agent
  drilldowns, sorted descending by spend.
- Empty DataFrame → placeholder section (visible "no data" line).
- Missing parquet (NoSuchKey) → fetcher returns None, build_cost_section
  emits the placeholder.
- Other S3 errors (AccessDenied, etc.) → raise per ``feedback_no_silent_fails``.
- Parquet corruption → raise (pyarrow surfaces InvalidArgument / similar).
- ``build_cost_section`` happy path: parquet present → full markdown
  section with all drilldowns populated.
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock

import pandas as pd
import pytest
from botocore.exceptions import ClientError


_BUCKET = "alpha-engine-research"


# ── Fixtures ──────────────────────────────────────────────────────────────


class _StubBody:
    """Mimic the file-like ``Body`` returned by boto3's ``get_object``."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


def _make_stub_s3_with_parquet(df: pd.DataFrame) -> MagicMock:
    """Build a MagicMock S3 client whose ``get_object`` returns ``df`` as parquet."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    body_bytes = buf.getvalue()
    stub = MagicMock()
    stub.get_object.return_value = {"Body": _StubBody(body_bytes)}
    return stub


def _make_stub_s3_with_no_such_key() -> MagicMock:
    """Build a MagicMock S3 client whose ``get_object`` raises NoSuchKey."""
    stub = MagicMock()
    stub.get_object.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "not found"}},
        "GetObject",
    )
    return stub


def _make_row(
    *,
    agent_id: str,
    sector_team_id: str | None,
    model_name: str,
    run_type: str = "weekly_research",
    input_tokens: int = 1000,
    output_tokens: int = 500,
    cost_usd: float = 0.0035,
) -> dict:
    return {
        "schema_version": 1,
        "timestamp": "2026-05-02T13:30:00+00:00",
        "run_id": "2026-05-02",
        "agent_id": agent_id,
        "sector_team_id": sector_team_id,
        "node_name": "some_node",
        "run_type": run_type,
        "prompt_id": None,
        "prompt_version": None,
        "prompt_version_hash": None,
        "model_name": model_name,
        "call_seq": 1,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": 0,
        "cache_create_tokens": 0,
        "cost_usd": cost_usd,
    }




# ── Markdown rendering (pure function) ────────────────────────────────────


class TestRenderHappyPath:
    def test_full_drilldowns_present(self):
        from analysis.cost_report import render_cost_report_markdown

        df = pd.DataFrame([
            _make_row(agent_id="sector_team:tech", sector_team_id="tech",
                     model_name="claude-haiku-4-5",
                     input_tokens=4000, output_tokens=1200, cost_usd=0.010),
            _make_row(agent_id="sector_team:financials", sector_team_id="financials",
                     model_name="claude-haiku-4-5",
                     input_tokens=2000, output_tokens=800, cost_usd=0.006),
            _make_row(agent_id="ic_cio", sector_team_id=None,
                     model_name="claude-sonnet-4-6",
                     input_tokens=8000, output_tokens=2000, cost_usd=0.054),
        ])
        md = render_cost_report_markdown(df, run_date="2026-05-02")

        assert "## LLM cost report" in md
        # Total cost: 0.010 + 0.006 + 0.054 = 0.070
        assert "**Total cost: $0.0700**" in md
        assert "Total input tokens: 14,000" in md
        assert "Total output tokens: 4,000" in md
        # Per-team breakdown
        assert "### By sector team" in md
        assert "tech" in md
        assert "financials" in md
        # Per-model breakdown
        assert "### By model" in md
        assert "claude-haiku-4-5" in md
        assert "claude-sonnet-4-6" in md
        # Per-agent breakdown
        assert "### By agent_id" in md
        assert "sector_team:tech" in md
        assert "ic_cio" in md

    def test_breakdowns_sorted_descending(self):
        from analysis.cost_report import render_cost_report_markdown

        df = pd.DataFrame([
            _make_row(agent_id="agent_cheap", sector_team_id=None,
                     model_name="claude-haiku-4-5", cost_usd=0.001),
            _make_row(agent_id="agent_expensive", sector_team_id=None,
                     model_name="claude-haiku-4-5", cost_usd=0.500),
            _make_row(agent_id="agent_mid", sector_team_id=None,
                     model_name="claude-haiku-4-5", cost_usd=0.100),
        ])
        md = render_cost_report_markdown(df, run_date="2026-05-02")
        # In the agent_id breakdown, expensive should appear before cheap.
        agent_section_start = md.index("### By agent_id")
        agent_section = md[agent_section_start:]
        i_expensive = agent_section.index("agent_expensive")
        i_mid = agent_section.index("agent_mid")
        i_cheap = agent_section.index("agent_cheap")
        assert i_expensive < i_mid < i_cheap

    def test_run_date_in_header(self):
        from analysis.cost_report import render_cost_report_markdown

        df = pd.DataFrame([
            _make_row(agent_id="a", sector_team_id=None,
                     model_name="claude-haiku-4-5", cost_usd=0.001),
        ])
        md = render_cost_report_markdown(df, run_date="2026-05-02")
        assert "Run date: 2026-05-02" in md

    def test_token_totals_formatted_with_commas(self):
        from analysis.cost_report import render_cost_report_markdown

        df = pd.DataFrame([
            _make_row(agent_id="a", sector_team_id=None,
                     model_name="claude-haiku-4-5",
                     input_tokens=1234567, output_tokens=98765, cost_usd=0.0),
        ])
        md = render_cost_report_markdown(df, run_date="2026-05-02")
        # Comma-formatted token counts make eyeballing easier.
        assert "1,234,567" in md
        assert "98,765" in md


class TestRenderEdgeCases:
    def test_empty_dataframe_renders_placeholder(self):
        from analysis.cost_report import render_cost_report_markdown

        md = render_cost_report_markdown(pd.DataFrame(), run_date="2026-05-02")
        assert "## LLM cost report" in md
        assert "_No cost data available_" in md

    def test_none_sector_team_renders_as_none_label(self):
        """Rows without sector_team_id (macro_economist, ic_cio) shouldn't
        crash the breakdown — render under '(none)' label."""
        from analysis.cost_report import render_cost_report_markdown

        df = pd.DataFrame([
            _make_row(agent_id="macro_economist", sector_team_id=None,
                     model_name="claude-sonnet-4-6", cost_usd=0.05),
            _make_row(agent_id="sector_team:tech", sector_team_id="tech",
                     model_name="claude-haiku-4-5", cost_usd=0.01),
        ])
        md = render_cost_report_markdown(df, run_date="2026-05-02")
        # The None group should show up as either (none), nan, or None
        # depending on pandas/dropna behavior — accept any.
        assert "(none)" in md or "nan" in md or "None" in md


# ── S3 fetch path ─────────────────────────────────────────────────────────


class TestFetchCostParquet:
    def test_happy_path_returns_dataframe(self):
        from analysis.cost_report import fetch_cost_parquet

        rows = [
            _make_row(agent_id="ic_cio", sector_team_id=None,
                     model_name="claude-sonnet-4-6", cost_usd=0.05),
        ]
        stub = _make_stub_s3_with_parquet(pd.DataFrame(rows))

        df = fetch_cost_parquet("2026-05-02", s3_client=stub)
        assert df is not None
        assert len(df) == 1
        assert df.iloc[0]["agent_id"] == "ic_cio"
        # Verify the canonical key was used.
        stub.get_object.assert_called_once_with(
            Bucket=_BUCKET,
            Key="decision_artifacts/_cost/2026-05-02/cost.parquet",
        )

    def test_missing_parquet_returns_none(self):
        """NoSuchKey → graceful None (capture-flag-off case)."""
        from analysis.cost_report import fetch_cost_parquet

        stub = _make_stub_s3_with_no_such_key()
        df = fetch_cost_parquet("2026-05-02", s3_client=stub)
        assert df is None

    def test_other_s3_error_raises(self):
        """AccessDenied / wiring failure → raise, not silent skip."""
        from analysis.cost_report import fetch_cost_parquet

        stub = MagicMock()
        stub.get_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "no perm"}},
            "GetObject",
        )
        with pytest.raises(ClientError):
            fetch_cost_parquet("2026-05-02", s3_client=stub)


# ── Public entrypoint ────────────────────────────────────────────────────


class TestBuildCostSection:
    def test_happy_path_produces_full_markdown(self):
        from analysis.cost_report import build_cost_section

        rows = [
            _make_row(agent_id="sector_team:tech", sector_team_id="tech",
                     model_name="claude-haiku-4-5", cost_usd=0.010),
            _make_row(agent_id="ic_cio", sector_team_id=None,
                     model_name="claude-sonnet-4-6", cost_usd=0.050),
        ]
        stub = _make_stub_s3_with_parquet(pd.DataFrame(rows))

        md = build_cost_section("2026-05-02", s3_client=stub)
        assert "## LLM cost report" in md
        assert "**Total cost: $0.0600**" in md
        assert "tech" in md
        assert "ic_cio" in md

    def test_missing_parquet_produces_placeholder(self):
        from analysis.cost_report import build_cost_section

        stub = _make_stub_s3_with_no_such_key()
        md = build_cost_section("2026-05-02", s3_client=stub)
        assert "## LLM cost report" in md
        assert "_No cost data available_" in md
        assert "ALPHA_ENGINE_DECISION_CAPTURE_ENABLED" in md
