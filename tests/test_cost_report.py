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


def _make_multi_date_stub_with_listing(
    date_to_df: dict[str, pd.DataFrame | None],
    *,
    listing_dates: list[str] | None = None,
) -> MagicMock:
    """Like ``_make_multi_date_stub`` but also stubs ``list_objects_v2``
    so ``_telemetry_first_capture_date`` can resolve a min-date floor.

    ``listing_dates`` controls which ISO dates appear in the listing
    (defaults to ``date_to_df.keys()``). Pass an explicit list to test
    cases where listing and per-date fetches diverge (e.g. simulating
    "telemetry shipped 5/2 but the 5/2 parquet itself is gone").
    """
    base = _make_multi_date_stub(date_to_df)
    if listing_dates is None:
        listing_dates = sorted(date_to_df.keys())
    base.list_objects_v2.return_value = {
        "Contents": [
            {"Key": f"decision_artifacts/_cost/{d}/cost.parquet"}
            for d in listing_dates
        ],
    }
    return base


class TestTelemetryFirstCaptureDate:
    def test_returns_min_iso_date(self):
        from analysis.cost_report import _telemetry_first_capture_date

        stub = _make_multi_date_stub_with_listing(
            {}, listing_dates=["2026-05-09", "2026-05-02", "2026-05-06"],
        )
        assert _telemetry_first_capture_date(
            bucket="b", s3_client=stub,
        ) == "2026-05-02"

    def test_returns_none_when_no_partitions(self):
        from analysis.cost_report import _telemetry_first_capture_date

        stub = MagicMock()
        stub.list_objects_v2.return_value = {"Contents": []}
        assert _telemetry_first_capture_date(
            bucket="b", s3_client=stub,
        ) is None

    def test_returns_none_when_listing_omitted(self):
        from analysis.cost_report import _telemetry_first_capture_date

        stub = MagicMock()
        stub.list_objects_v2.return_value = {}  # no Contents key
        assert _telemetry_first_capture_date(
            bucket="b", s3_client=stub,
        ) is None

    def test_skips_non_iso_partition_names(self):
        from analysis.cost_report import _telemetry_first_capture_date

        stub = MagicMock()
        stub.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "decision_artifacts/_cost/garbage/cost.parquet"},
                {"Key": "decision_artifacts/_cost/2026-05-04/cost.parquet"},
                {"Key": "decision_artifacts/_cost/2026-05-02/cost.parquet"},
            ],
        }
        assert _telemetry_first_capture_date(
            bucket="b", s3_client=stub,
        ) == "2026-05-02"

    def test_skips_non_cost_parquet_keys(self):
        from analysis.cost_report import _telemetry_first_capture_date

        stub = MagicMock()
        stub.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "decision_artifacts/_cost/2026-05-02/cost.parquet"},
                {"Key": "decision_artifacts/_cost/2026-05-02/manifest.json"},
                {"Key": "decision_artifacts/_cost/2026-04-25/_other.parquet"},
            ],
        }
        # Only cost.parquet keys count toward the floor.
        assert _telemetry_first_capture_date(
            bucket="b", s3_client=stub,
        ) == "2026-05-02"

    def test_returns_none_on_client_error(self, caplog):
        from analysis.cost_report import _telemetry_first_capture_date

        stub = MagicMock()
        stub.list_objects_v2.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "nope"}},
            "ListObjectsV2",
        )
        with caplog.at_level("WARNING"):
            assert _telemetry_first_capture_date(
                bucket="b", s3_client=stub,
            ) is None
        assert any(
            "no-pre-telemetry-floor" in r.message for r in caplog.records
        )


class TestPreTelemetryBaselineClassification:
    """Regression coverage for the 2026-05-09 evaluator email bug:
    pre-launch Saturday gaps were rendered as "capture flag may have
    been off" when in fact the telemetry feature didn't exist yet.
    """

    def test_classifies_pre_launch_dates_separately(self):
        from analysis.cost_report import detect_anomaly

        # Telemetry first captured 5/2 (single Saturday post-launch).
        # 5/9 run looks back at 5/2 / 4/25 / 4/18 / 4/11 — three are pre-launch.
        baseline_df = pd.DataFrame([
            _make_row(agent_id="a", sector_team_id=None,
                     model_name="claude-haiku-4-5", cost_usd=1.0),
        ])
        stub = _make_multi_date_stub_with_listing(
            {"2026-05-02": baseline_df},
            listing_dates=["2026-05-02"],
        )
        result = detect_anomaly(
            "2026-05-09", current_total_cost_usd=1.10, s3_client=stub,
        )
        assert result["telemetry_first_date"] == "2026-05-02"
        assert result["baseline_dates_found"] == ["2026-05-02"]
        # 4/11, 4/18, 4/25 are pre-launch.
        assert sorted(result["baseline_dates_pre_telemetry"]) == [
            "2026-04-11", "2026-04-18", "2026-04-25",
        ]
        assert result["baseline_dates_missing"] == []

    def test_no_baseline_when_only_pre_launch_priors(self):
        from analysis.cost_report import detect_anomaly

        # Telemetry first captured 5/9 itself (the run we're evaluating).
        # All 4 priors are pre-launch.
        stub = _make_multi_date_stub_with_listing(
            {}, listing_dates=["2026-05-09"],
        )
        result = detect_anomaly(
            "2026-05-09", current_total_cost_usd=1.0, s3_client=stub,
        )
        assert result["status"] == "no_baseline"
        assert result["baseline_dates_found"] == []
        assert result["baseline_dates_missing"] == []
        assert len(result["baseline_dates_pre_telemetry"]) == 4

    def test_distinguishes_post_launch_gap_from_pre_launch(self):
        """If telemetry shipped 4/18, then 4/11 is pre-launch but 4/25
        with no parquet is a genuine post-launch gap.
        """
        from analysis.cost_report import detect_anomaly

        baseline_df = pd.DataFrame([
            _make_row(agent_id="a", sector_team_id=None,
                     model_name="claude-haiku-4-5", cost_usd=1.0),
        ])
        # Listing reports 4/18 as the earliest partition (telemetry floor).
        # Per-date stub: 4/18 + 5/2 have parquets; 4/25 is genuinely missing.
        stub = _make_multi_date_stub_with_listing(
            {"2026-04-18": baseline_df, "2026-05-02": baseline_df},
            listing_dates=["2026-04-18", "2026-05-02"],
        )
        result = detect_anomaly(
            "2026-05-09", current_total_cost_usd=1.10, s3_client=stub,
        )
        assert result["telemetry_first_date"] == "2026-04-18"
        assert sorted(result["baseline_dates_found"]) == [
            "2026-04-18", "2026-05-02",
        ]
        assert result["baseline_dates_missing"] == ["2026-04-25"]
        assert result["baseline_dates_pre_telemetry"] == ["2026-04-11"]

    def test_legacy_path_when_listing_unavailable(self):
        """Without a telemetry floor (list_objects_v2 fails), behavior
        falls back to the original "all-gaps-equal" classification.
        """
        from analysis.cost_report import detect_anomaly

        baseline_df = pd.DataFrame([
            _make_row(agent_id="a", sector_team_id=None,
                     model_name="claude-haiku-4-5", cost_usd=1.0),
        ])

        def _list(*args, **kwargs):
            raise ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "nope"}},
                "ListObjectsV2",
            )

        stub = _make_multi_date_stub({"2026-05-02": baseline_df})
        stub.list_objects_v2.side_effect = _list
        result = detect_anomaly(
            "2026-05-09", current_total_cost_usd=1.0, s3_client=stub,
        )
        assert result["telemetry_first_date"] is None
        assert result["baseline_dates_pre_telemetry"] == []
        # All 3 missing priors are classified as missing (legacy framing).
        assert sorted(result["baseline_dates_missing"]) == [
            "2026-04-11", "2026-04-18", "2026-04-25",
        ]


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

    def test_pre_telemetry_no_baseline_message(self):
        """First-week-after-launch case: all priors pre-date telemetry."""
        from analysis.cost_report import render_anomaly_section
        md = render_anomaly_section({
            "current_total_usd": 1.0,
            "baseline_dates_found": [],
            "baseline_dates_missing": [],
            "baseline_dates_pre_telemetry": [
                "2026-04-11", "2026-04-18", "2026-04-25", "2026-05-02",
            ],
            "telemetry_first_date": "2026-05-09",
            "baseline_mean_usd": None,
            "ratio": None,
            "threshold_ratio": 2.0,
            "is_anomaly": False,
            "status": "no_baseline",
        })
        assert "pre-date the cost-telemetry feature" in md
        assert "first captured: 2026-05-09" in md
        # Should NOT blame the operator capture flag for pre-launch gaps.
        assert "capture-flag-off" not in md
        assert "capture flag may have been off" not in md

    def test_pre_telemetry_excluded_from_baseline_in_ok_path(self):
        """Anomaly OK path: pre-telemetry priors get a separate sentence
        and are excluded from the baseline window.
        """
        from analysis.cost_report import render_anomaly_section
        md = render_anomaly_section({
            "current_total_usd": 1.10,
            "baseline_dates_found": ["2026-05-02"],
            "baseline_dates_missing": [],
            "baseline_dates_pre_telemetry": [
                "2026-04-11", "2026-04-18", "2026-04-25",
            ],
            "telemetry_first_date": "2026-05-02",
            "baseline_mean_usd": 1.0,
            "ratio": 1.10,
            "threshold_ratio": 2.0,
            "is_anomaly": False,
            "status": "ok",
        })
        assert "Baseline window" in md
        assert "3 of 4" in md
        assert "first captured: 2026-05-02" in md
        # Pre-telemetry framing — must not say capture-flag-off.
        assert "capture flag may have been off" not in md

    def test_mixed_pre_telemetry_and_post_launch_gap(self):
        """Both classifications can co-exist: one pre-launch + one
        post-launch genuine gap. Both sentences should render.
        """
        from analysis.cost_report import render_anomaly_section
        md = render_anomaly_section({
            "current_total_usd": 1.10,
            "baseline_dates_found": ["2026-04-18", "2026-05-02"],
            "baseline_dates_missing": ["2026-04-25"],
            "baseline_dates_pre_telemetry": ["2026-04-11"],
            "telemetry_first_date": "2026-04-18",
            "baseline_mean_usd": 1.0,
            "ratio": 1.10,
            "threshold_ratio": 2.0,
            "is_anomaly": False,
            "status": "ok",
        })
        assert "Baseline window" in md
        assert "Baseline gaps" in md
        # Genuine gap counts only post-launch denominator (2 found + 1 missing = 3).
        assert "1 of 3" in md
        assert "post-launch prior weekly runs" in md

    def test_no_baseline_mixed_message(self):
        """All priors gone — some pre-telemetry, some post-launch
        missing. Message blends both classifications.
        """
        from analysis.cost_report import render_anomaly_section
        md = render_anomaly_section({
            "current_total_usd": 1.0,
            "baseline_dates_found": [],
            "baseline_dates_missing": ["2026-05-02"],
            "baseline_dates_pre_telemetry": [
                "2026-04-11", "2026-04-18", "2026-04-25",
            ],
            "telemetry_first_date": "2026-04-25",
            "baseline_mean_usd": None,
            "ratio": None,
            "threshold_ratio": 2.0,
            "is_anomaly": False,
            "status": "no_baseline",
        })
        assert "pre-date telemetry" in md
        assert "first captured: 2026-04-25" in md
        assert "1 post-launch run(s)" in md


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


# ── Changelog auto-emit (ROADMAP P0 sub-item 5 cost-anomaly half) ─────────


def _make_multi_date_stub_with_put(
    date_to_df: dict[str, pd.DataFrame | None],
) -> MagicMock:
    """Like ``_make_multi_date_stub`` but also captures put_object calls.

    Lets tests assert on whether the changelog auto-emit fired and what
    payload it wrote, without mocking the full boto3 API surface.
    """
    stub = _make_multi_date_stub(date_to_df)
    stub.put_object = MagicMock(return_value={"ETag": '"deadbeef"'})
    return stub


class TestChangelogAutoEmitOnAnomaly:
    """Cost-anomaly auto-emit hook into the system-wide changelog corpus.

    ROADMAP P0 line ~2154 sub-item 5 (cost-anomaly half — Item 2
    cost-telemetry upstream is closed, this hook unblocked 2026-05-07).
    """

    def test_anomaly_status_writes_changelog_entry(self):
        """When detect_anomaly returns status=anomaly, build_cost_section
        writes one schema-1.0.0 incident entry to changelog/entries/."""
        from analysis.cost_report import build_cost_section

        current_df = pd.DataFrame([
            _make_row(agent_id="ic_cio", sector_team_id=None,
                     model_name="claude-sonnet-4-6", cost_usd=10.00),
        ])
        baseline_df = pd.DataFrame([
            _make_row(agent_id="ic_cio", sector_team_id=None,
                     model_name="claude-sonnet-4-6", cost_usd=1.00),
        ])
        stub = _make_multi_date_stub_with_put({
            "2026-05-09": current_df,
            "2026-05-02": baseline_df,
            "2026-04-25": baseline_df,
            "2026-04-18": baseline_df,
            "2026-04-11": baseline_df,
        })
        build_cost_section("2026-05-09", s3_client=stub)
        # Exactly one put_object call — the changelog auto-emit
        assert stub.put_object.call_count == 1
        call = stub.put_object.call_args_list[0]
        assert call.kwargs["Bucket"] == "alpha-engine-research"
        assert call.kwargs["Key"].startswith("changelog/entries/")
        assert call.kwargs["Key"].endswith(".json")
        assert call.kwargs["ContentType"] == "application/json"

    def test_changelog_entry_payload_shape(self):
        """The auto-emitted entry carries every schema-1.0.0 field the
        SNS-mirror + cloudwatch-mirror Lambdas already populate, plus a
        ``cost_anomaly`` block with the diagnostic numbers."""
        import json as _json
        from analysis.cost_report import build_cost_section

        current_df = pd.DataFrame([
            _make_row(agent_id="ic_cio", sector_team_id=None,
                     model_name="claude-sonnet-4-6", cost_usd=10.00),
        ])
        baseline_df = pd.DataFrame([
            _make_row(agent_id="ic_cio", sector_team_id=None,
                     model_name="claude-sonnet-4-6", cost_usd=1.00),
        ])
        stub = _make_multi_date_stub_with_put({
            "2026-05-09": current_df,
            "2026-05-02": baseline_df,
            "2026-04-25": baseline_df,
            "2026-04-18": baseline_df,
            "2026-04-11": baseline_df,
        })
        build_cost_section("2026-05-09", s3_client=stub)

        body = _json.loads(stub.put_object.call_args_list[0].kwargs["Body"].decode())
        assert body["schema_version"] == "1.0.0"
        assert body["event_type"] == "incident"
        assert body["severity"] == "medium"  # cost spike isn't capital-at-risk
        assert body["subsystem"] == "telemetry"
        assert body["root_cause_category"] == "prompt_regression"  # plausible default
        assert body["source"] == "cost-anomaly-autoemit"
        assert body["actor"] == "alpha-engine-cost-telemetry"
        assert body["machine"] == "backtester:analysis/cost_report.py"
        assert body["auto_emitted"] is True
        assert body["run_id"] == "2026-05-09"
        # Diagnostic block carries the ratio + baseline numbers
        ca = body["cost_anomaly"]
        assert ca["ratio"] == 10.0
        assert ca["threshold_ratio"] == 2.0
        assert ca["current_total_usd"] == 10.0
        assert ca["baseline_mean_usd"] == 1.0
        assert ca["baseline_dates_found"] == [
            "2026-05-02", "2026-04-25", "2026-04-18", "2026-04-11",
        ]
        # event_id format mirrors the SNS-mirror + cloudwatch-mirror scheme
        parts = body["event_id"].split("_")
        assert len(parts) == 3  # ts_actor_hash
        assert parts[1] == "alpha-engine-cost-telemetry"
        assert len(parts[2]) == 7  # 7-hex sha1 prefix

    def test_ok_status_does_not_emit_changelog_entry(self):
        """No anomaly → no put_object call. Quiet weeks stay quiet."""
        from analysis.cost_report import build_cost_section

        current_df = pd.DataFrame([
            _make_row(agent_id="ic_cio", sector_team_id=None,
                     model_name="claude-sonnet-4-6", cost_usd=0.01),
        ])
        baseline_df = pd.DataFrame([
            _make_row(agent_id="ic_cio", sector_team_id=None,
                     model_name="claude-sonnet-4-6", cost_usd=0.01),
        ])
        stub = _make_multi_date_stub_with_put({
            "2026-05-09": current_df,
            "2026-05-02": baseline_df,
            "2026-04-25": baseline_df,
            "2026-04-18": baseline_df,
            "2026-04-11": baseline_df,
        })
        build_cost_section("2026-05-09", s3_client=stub)
        assert stub.put_object.call_count == 0

    def test_no_baseline_does_not_emit_changelog_entry(self):
        """First-run state (no priors) → status=no_baseline → no auto-emit."""
        from analysis.cost_report import build_cost_section

        current_df = pd.DataFrame([
            _make_row(agent_id="ic_cio", sector_team_id=None,
                     model_name="claude-sonnet-4-6", cost_usd=10.00),
        ])
        stub = _make_multi_date_stub_with_put({
            "2026-05-09": current_df,
            # All priors absent → no_baseline path
        })
        build_cost_section("2026-05-09", s3_client=stub)
        assert stub.put_object.call_count == 0

    def test_alerting_disabled_does_not_emit_changelog_entry(self, monkeypatch):
        """Threshold ≤ 0 → status=alerting_disabled → no auto-emit."""
        from analysis.cost_report import build_cost_section

        monkeypatch.setenv("ALPHA_ENGINE_COST_ANOMALY_RATIO", "0")
        current_df = pd.DataFrame([
            _make_row(agent_id="ic_cio", sector_team_id=None,
                     model_name="claude-sonnet-4-6", cost_usd=10.00),
        ])
        baseline_df = pd.DataFrame([
            _make_row(agent_id="ic_cio", sector_team_id=None,
                     model_name="claude-sonnet-4-6", cost_usd=1.00),
        ])
        stub = _make_multi_date_stub_with_put({
            "2026-05-09": current_df,
            "2026-05-02": baseline_df,
            "2026-04-25": baseline_df,
            "2026-04-18": baseline_df,
            "2026-04-11": baseline_df,
        })
        build_cost_section("2026-05-09", s3_client=stub)
        assert stub.put_object.call_count == 0

    def test_changelog_write_failure_does_not_break_cost_section(self):
        """S3 put_object exception is logged + swallowed; build_cost_section
        still returns the markdown (cost-section rendering must not be
        blocked by changelog corruption — alert is still in email + log)."""
        from analysis.cost_report import build_cost_section

        current_df = pd.DataFrame([
            _make_row(agent_id="ic_cio", sector_team_id=None,
                     model_name="claude-sonnet-4-6", cost_usd=10.00),
        ])
        baseline_df = pd.DataFrame([
            _make_row(agent_id="ic_cio", sector_team_id=None,
                     model_name="claude-sonnet-4-6", cost_usd=1.00),
        ])
        stub = _make_multi_date_stub_with_put({
            "2026-05-09": current_df,
            "2026-05-02": baseline_df,
            "2026-04-25": baseline_df,
            "2026-04-18": baseline_df,
            "2026-04-11": baseline_df,
        })
        stub.put_object = MagicMock(side_effect=ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "denied"}},
            "PutObject",
        ))
        # No exception raised; markdown still produced
        md = build_cost_section("2026-05-09", s3_client=stub)
        assert "ANOMALY DETECTED" in md

    def test_event_id_idempotent_on_same_inputs(self):
        """Re-running build_cost_section with the same anomaly inputs
        produces the same event_id (and the same S3 key — overwrite, not
        duplicate)."""
        from analysis.cost_report import _emit_changelog_anomaly_entry

        anomaly = {
            "current_total_usd": 10.0,
            "baseline_mean_usd": 1.0,
            "ratio": 10.0,
            "threshold_ratio": 2.0,
            "baseline_dates_found": ["2026-05-02"],
            "baseline_dates_missing": [],
            "is_anomaly": True,
            "status": "anomaly",
        }
        stub_a = MagicMock()
        stub_b = MagicMock()
        # Same run_date + ratio + total → same event_hash → same event_id
        # tail (the timestamp prefix differs by wall-clock; we check the hash)
        _emit_changelog_anomaly_entry(anomaly, run_date="2026-05-09",
                                      bucket="alpha-engine-research", s3_client=stub_a)
        _emit_changelog_anomaly_entry(anomaly, run_date="2026-05-09",
                                      bucket="alpha-engine-research", s3_client=stub_b)
        key_a = stub_a.put_object.call_args.kwargs["Key"]
        key_b = stub_b.put_object.call_args.kwargs["Key"]
        # Hash segment (last 7 hex chars before .json) is identical
        hash_a = key_a.rsplit("_", 1)[-1].split(".")[0]
        hash_b = key_b.rsplit("_", 1)[-1].split(".")[0]
        assert hash_a == hash_b
