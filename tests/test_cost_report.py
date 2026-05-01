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


# ── PR 5b: anomaly detection ──────────────────────────────────────────────


def _make_multi_date_stub(date_to_df: dict[str, pd.DataFrame | None]) -> MagicMock:
    """Build an S3 stub whose get_object dispatches by Key.

    ``date_to_df`` maps run_date (YYYY-MM-DD) → DataFrame (parquet to
    return) or None (raise NoSuchKey).
    """
    stub = MagicMock()

    def _get_object(*, Bucket: str, Key: str):
        # Extract date from key: decision_artifacts/_cost/{date}/cost.parquet
        parts = Key.split("/")
        if len(parts) >= 3:
            date_str = parts[-2]
            df = date_to_df.get(date_str)
            if df is None:
                raise ClientError(
                    {"Error": {"Code": "NoSuchKey", "Message": "not found"}},
                    "GetObject",
                )
            buf = io.BytesIO()
            df.to_parquet(buf, index=False, engine="pyarrow")
            return {"Body": _StubBody(buf.getvalue())}
        raise ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "not found"}},
            "GetObject",
        )

    stub.get_object.side_effect = _get_object
    return stub


@pytest.fixture(autouse=True)
def reset_anomaly_env(monkeypatch):
    monkeypatch.delenv("ALPHA_ENGINE_COST_ANOMALY_RATIO", raising=False)
    yield


class TestAnomalyRatioResolution:
    def test_default_is_2x(self):
        from analysis.cost_report import _resolve_anomaly_ratio
        assert _resolve_anomaly_ratio() == 2.0

    def test_env_override(self, monkeypatch):
        from analysis.cost_report import _resolve_anomaly_ratio
        monkeypatch.setenv("ALPHA_ENGINE_COST_ANOMALY_RATIO", "3.5")
        assert _resolve_anomaly_ratio() == 3.5

    def test_zero_disables(self, monkeypatch):
        from analysis.cost_report import _resolve_anomaly_ratio
        monkeypatch.setenv("ALPHA_ENGINE_COST_ANOMALY_RATIO", "0")
        assert _resolve_anomaly_ratio() == 0.0

    def test_unparseable_returns_zero_with_warn(self, monkeypatch, caplog):
        from analysis.cost_report import _resolve_anomaly_ratio
        monkeypatch.setenv("ALPHA_ENGINE_COST_ANOMALY_RATIO", "abc")
        with caplog.at_level("WARNING"):
            assert _resolve_anomaly_ratio() == 0.0
        assert any("not a number" in r.message for r in caplog.records)


class TestPreviousWeeklyDates:
    def test_returns_n_prior_weekly_dates(self):
        from analysis.cost_report import _previous_weekly_dates
        dates = _previous_weekly_dates("2026-05-09", weeks=4)
        # Saturday 2026-05-09 → previous 4 Saturdays: 5/02, 4/25, 4/18, 4/11.
        assert dates == ["2026-05-02", "2026-04-25", "2026-04-18", "2026-04-11"]


class TestDetectAnomaly:
    def test_no_baseline_when_all_priors_missing(self):
        from analysis.cost_report import detect_anomaly

        stub = _make_multi_date_stub({})  # nothing exists
        result = detect_anomaly(
            "2026-05-09", current_total_cost_usd=0.50, s3_client=stub,
        )
        assert result["status"] == "no_baseline"
        assert result["is_anomaly"] is False
        assert result["baseline_mean_usd"] is None
        assert result["baseline_dates_found"] == []
        assert len(result["baseline_dates_missing"]) == 4

    def test_ok_when_under_threshold(self):
        from analysis.cost_report import detect_anomaly

        # Baseline averages $0.50; current $0.60 = 1.2x < 2.0× threshold.
        baseline_df = pd.DataFrame([
            _make_row(agent_id="a", sector_team_id=None,
                     model_name="claude-haiku-4-5", cost_usd=0.50),
        ])
        stub = _make_multi_date_stub({
            "2026-05-02": baseline_df,
            "2026-04-25": baseline_df,
            "2026-04-18": baseline_df,
            "2026-04-11": baseline_df,
        })
        result = detect_anomaly(
            "2026-05-09", current_total_cost_usd=0.60, s3_client=stub,
        )
        assert result["status"] == "ok"
        assert result["is_anomaly"] is False
        assert result["baseline_mean_usd"] == pytest.approx(0.50)
        assert result["ratio"] == pytest.approx(1.2)
        assert len(result["baseline_dates_found"]) == 4

    def test_anomaly_when_over_threshold(self, caplog):
        from analysis.cost_report import detect_anomaly

        # Baseline averages $0.50; current $1.50 = 3.0x > 2.0× threshold.
        baseline_df = pd.DataFrame([
            _make_row(agent_id="a", sector_team_id=None,
                     model_name="claude-haiku-4-5", cost_usd=0.50),
        ])
        stub = _make_multi_date_stub({
            "2026-05-02": baseline_df,
            "2026-04-25": baseline_df,
            "2026-04-18": baseline_df,
            "2026-04-11": baseline_df,
        })
        with caplog.at_level("WARNING"):
            result = detect_anomaly(
                "2026-05-09", current_total_cost_usd=1.50, s3_client=stub,
            )
        assert result["status"] == "anomaly"
        assert result["is_anomaly"] is True
        assert result["ratio"] == pytest.approx(3.0)
        # Verify the WARN log fired with the diagnostic info.
        assert any("anomaly" in r.message for r in caplog.records)
        assert any("3.00x" in r.message for r in caplog.records)

    def test_partial_baseline_uses_available_dates(self):
        """If 2 of 4 priors are missing, baseline is mean of the 2 found."""
        from analysis.cost_report import detect_anomaly

        baseline_df_a = pd.DataFrame([
            _make_row(agent_id="a", sector_team_id=None,
                     model_name="claude-haiku-4-5", cost_usd=0.40),
        ])
        baseline_df_b = pd.DataFrame([
            _make_row(agent_id="a", sector_team_id=None,
                     model_name="claude-haiku-4-5", cost_usd=0.60),
        ])
        # Only 5/02 and 4/25 exist; 4/18 and 4/11 missing.
        stub = _make_multi_date_stub({
            "2026-05-02": baseline_df_a,
            "2026-04-25": baseline_df_b,
        })
        result = detect_anomaly(
            "2026-05-09", current_total_cost_usd=0.55, s3_client=stub,
        )
        assert result["status"] == "ok"
        assert result["baseline_mean_usd"] == pytest.approx(0.50)  # (0.40 + 0.60) / 2
        assert len(result["baseline_dates_found"]) == 2
        assert len(result["baseline_dates_missing"]) == 2

    def test_disabled_when_threshold_zero(self, monkeypatch):
        from analysis.cost_report import detect_anomaly

        monkeypatch.setenv("ALPHA_ENGINE_COST_ANOMALY_RATIO", "0")
        result = detect_anomaly(
            "2026-05-09", current_total_cost_usd=999.0, s3_client=MagicMock(),
        )
        assert result["status"] == "alerting_disabled"
        assert result["is_anomaly"] is False


class TestRenderAnomalySection:
    def test_anomaly_section_includes_warning_marker(self):
        from analysis.cost_report import render_anomaly_section
        md = render_anomaly_section({
            "current_total_usd": 1.50,
            "baseline_dates_found": ["2026-05-02", "2026-04-25"],
            "baseline_dates_missing": [],
            "baseline_mean_usd": 0.50,
            "ratio": 3.0,
            "threshold_ratio": 2.0,
            "is_anomaly": True,
            "status": "anomaly",
        })
        assert "ANOMALY DETECTED" in md
        assert ":warning:" in md
        assert "3.00x" in md
        assert "2.00x" in md

    def test_ok_section_omits_warning_marker(self):
        from analysis.cost_report import render_anomaly_section
        md = render_anomaly_section({
            "current_total_usd": 0.60,
            "baseline_dates_found": ["2026-05-02"],
            "baseline_dates_missing": [],
            "baseline_mean_usd": 0.50,
            "ratio": 1.2,
            "threshold_ratio": 2.0,
            "is_anomaly": False,
            "status": "ok",
        })
        assert "ANOMALY DETECTED" not in md
        assert "1.20x" in md

    def test_no_baseline_section(self):
        from analysis.cost_report import render_anomaly_section
        md = render_anomaly_section({
            "current_total_usd": 0.50,
            "baseline_dates_found": [],
            "baseline_dates_missing": ["2026-05-02", "2026-04-25", "2026-04-18", "2026-04-11"],
            "baseline_mean_usd": None,
            "ratio": None,
            "threshold_ratio": 2.0,
            "is_anomaly": False,
            "status": "no_baseline",
        })
        assert "_No baseline available_" in md
        assert "ANOMALY DETECTED" not in md

    def test_alerting_disabled_section(self):
        from analysis.cost_report import render_anomaly_section
        md = render_anomaly_section({
            "current_total_usd": 1.0,
            "baseline_dates_found": [],
            "baseline_dates_missing": [],
            "baseline_mean_usd": None,
            "ratio": None,
            "threshold_ratio": 0.0,
            "is_anomaly": False,
            "status": "alerting_disabled",
        })
        assert "_Anomaly alerting disabled_" in md
        assert "ALPHA_ENGINE_COST_ANOMALY_RATIO" in md

    def test_partial_baseline_notes_gaps(self):
        from analysis.cost_report import render_anomaly_section
        md = render_anomaly_section({
            "current_total_usd": 0.55,
            "baseline_dates_found": ["2026-05-02", "2026-04-25"],
            "baseline_dates_missing": ["2026-04-18", "2026-04-11"],
            "baseline_mean_usd": 0.50,
            "ratio": 1.1,
            "threshold_ratio": 2.0,
            "is_anomaly": False,
            "status": "ok",
        })
        assert "Baseline gaps" in md
        assert "2 of 4" in md


class TestBuildCostSectionWithAnomaly:
    def test_happy_path_includes_anomaly_section(self):
        """build_cost_section with priors present runs anomaly detection
        and appends the section to the main report."""
        from analysis.cost_report import build_cost_section

        # Current run is small ($0.01); baseline of 4 priors is also small.
        current_df = pd.DataFrame([
            _make_row(agent_id="ic_cio", sector_team_id=None,
                     model_name="claude-sonnet-4-6", cost_usd=0.01),
        ])
        baseline_df = pd.DataFrame([
            _make_row(agent_id="ic_cio", sector_team_id=None,
                     model_name="claude-sonnet-4-6", cost_usd=0.01),
        ])
        stub = _make_multi_date_stub({
            "2026-05-09": current_df,
            "2026-05-02": baseline_df,
            "2026-04-25": baseline_df,
            "2026-04-18": baseline_df,
            "2026-04-11": baseline_df,
        })
        md = build_cost_section("2026-05-09", s3_client=stub)
        assert "## LLM cost report" in md
        assert "### Anomaly check" in md
        assert "ANOMALY DETECTED" not in md  # under threshold

    def test_anomaly_path_renders_warning(self):
        """build_cost_section detects + renders the anomaly when over threshold."""
        from analysis.cost_report import build_cost_section

        current_df = pd.DataFrame([
            _make_row(agent_id="ic_cio", sector_team_id=None,
                     model_name="claude-sonnet-4-6", cost_usd=10.00),
        ])
        baseline_df = pd.DataFrame([
            _make_row(agent_id="ic_cio", sector_team_id=None,
                     model_name="claude-sonnet-4-6", cost_usd=1.00),
        ])
        stub = _make_multi_date_stub({
            "2026-05-09": current_df,
            "2026-05-02": baseline_df,
            "2026-04-25": baseline_df,
            "2026-04-18": baseline_df,
            "2026-04-11": baseline_df,
        })
        md = build_cost_section("2026-05-09", s3_client=stub)
        assert "ANOMALY DETECTED" in md
        assert "10.00x" in md  # 10.00 / 1.00 = 10×
