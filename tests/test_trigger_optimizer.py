"""Unit tests for optimizer.trigger_optimizer — apply() + produce_artifact()
+ dual-write guarantee. All S3 calls mocked.

Part of the optimizer-artifact-assembler arc (PR 2).
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from optimizer import trigger_optimizer
from optimizer.assembler import set_cutover_enabled
from optimizer.trigger_optimizer import (
    S3_PARAMS_KEY,
    _build_overlay_params,
    apply,
    produce_artifact,
)


@pytest.fixture(autouse=True)
def _reset_cutover_flag():
    """Reset assembler cutover flag around each test."""
    set_cutover_enabled(False)
    yield
    set_cutover_enabled(False)


@pytest.fixture(autouse=True)
def _reset_trigger_config():
    """Every test starts in the default (observe-only) mode."""
    trigger_optimizer.init_config({})
    yield
    trigger_optimizer.init_config({})


@pytest.fixture
def live_mode():
    """Arm the legacy live write — the explicit, later flip
    (alpha-engine-config-I12231). Only tests of the armed path use this."""
    trigger_optimizer.init_config({"trigger_optimizer": {"live_apply": True}})


# ── _build_overlay_params ────────────────────────────────────────────────────


class TestBuildOverlayParams:

    def test_emits_disabled_triggers_field_overlay(self):
        result = {
            "status": "ok",
            "disabled_triggers": ["pullback", "vwap_discount"],
        }
        params, keys = _build_overlay_params(result)
        assert set(keys) == {"disabled_triggers", "disabled_triggers_updated_at"}
        assert params["disabled_triggers"] == ["pullback", "vwap_discount"]

    def test_empty_disabled_list_when_no_triggers_to_disable(self):
        result = {"status": "ok", "disabled_triggers": []}
        params, _ = _build_overlay_params(result)
        assert params["disabled_triggers"] == []


# ── produce_artifact ─────────────────────────────────────────────────────────


class TestProduceArtifact:

    @patch("optimizer.recommendation_artifact.boto3")
    def test_status_ok_promotes_only_when_live(self, mock_boto3, live_mode):
        s3 = MagicMock()
        mock_boto3.client.return_value = s3
        result = {
            "status": "ok",
            "disabled_triggers": ["pullback"],
            "total_evaluated": 4,
            "min_trades_threshold": 50,
        }
        outcome = produce_artifact(result, bucket="test-bucket")
        assert outcome["written"] is True
        assert outcome["key"].endswith("/from_trigger_optimizer.json")
        body = json.loads(s3.put_object.call_args.kwargs["Body"])
        assert body["promotion_intent"] == "promote"
        assert body["recommendation_kind"] == "field_overlay"
        assert body["fit_target"] == "entry_timing_alpha"
        assert body["recommended_params"]["disabled_triggers"] == ["pullback"]
        assert "disabled_triggers" in body["overlay_keys"]
        assert body["diagnostic"]["total_evaluated"] == 4
        assert body["diagnostic"]["apply_mode"] == "live"

    @patch("optimizer.recommendation_artifact.boto3")
    def test_status_insufficient_data_skips(self, mock_boto3):
        s3 = MagicMock()
        mock_boto3.client.return_value = s3
        result = {"status": "insufficient_data"}
        outcome = produce_artifact(result, bucket="test-bucket")
        assert outcome["written"] is True
        body = json.loads(s3.put_object.call_args.kwargs["Body"])
        assert body["promotion_intent"] == "skip"
        assert body["recommended_params"] == {}
        assert body["overlay_keys"] is None

    @patch("optimizer.recommendation_artifact.boto3")
    def test_swallows_s3_errors_non_fatal(self, mock_boto3):
        s3 = MagicMock()
        s3.put_object.side_effect = Exception("S3 disconnected")
        mock_boto3.client.return_value = s3
        outcome = produce_artifact({
            "status": "ok",
            "disabled_triggers": ["pullback"],
        }, bucket="test-bucket")
        assert outcome["written"] is False
        assert "S3 disconnected" in outcome["reason"]


# ── apply() dual-write contract ──────────────────────────────────────────────


class TestApplyDualWrite:

    @patch("optimizer.trigger_optimizer.boto3")
    @patch("optimizer.recommendation_artifact.boto3")
    def test_legacy_apply_path_also_writes_artifact(
        self, mock_artifact_boto3, mock_apply_boto3, live_mode,
    ):
        legacy_s3 = MagicMock()
        artifact_s3 = MagicMock()
        # Existing config has empty disabled_triggers; new recommendation differs.
        legacy_s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps({
                "atr_multiplier": 2.0,
                "disabled_triggers": [],
            }).encode()),
        }
        mock_apply_boto3.client.return_value = legacy_s3
        mock_artifact_boto3.client.return_value = artifact_s3

        result = {"status": "ok", "disabled_triggers": ["pullback"]}
        outcome = apply(result, bucket="test-bucket")

        # Legacy live write happened.
        assert outcome["applied"] is True
        legacy_keys = [c.kwargs["Key"] for c in legacy_s3.put_object.call_args_list]
        assert S3_PARAMS_KEY in legacy_keys

        # Artifact write happened too.
        artifact_keys = [c.kwargs["Key"] for c in artifact_s3.put_object.call_args_list]
        assert any(
            k.endswith("/from_trigger_optimizer.json") for k in artifact_keys
        )

    @patch("optimizer.trigger_optimizer.boto3")
    @patch("optimizer.recommendation_artifact.boto3")
    def test_no_change_idempotent_skip_still_writes_artifact(
        self, mock_artifact_boto3, mock_apply_boto3, live_mode,
    ):
        # Recommendation matches current → apply() returns early with
        # "no change" reason. Artifact is still written for audit.
        legacy_s3 = MagicMock()
        artifact_s3 = MagicMock()
        legacy_s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps({
                "disabled_triggers": ["pullback"],
            }).encode()),
        }
        mock_apply_boto3.client.return_value = legacy_s3
        mock_artifact_boto3.client.return_value = artifact_s3

        result = {"status": "ok", "disabled_triggers": ["pullback"]}
        outcome = apply(result, bucket="test-bucket")
        assert outcome["applied"] is False
        assert "no change" in outcome["reason"]

        # Legacy didn't write to live key.
        legacy_writes = [c for c in legacy_s3.put_object.call_args_list]
        assert len(legacy_writes) == 0

        # Artifact still written — the optimizer's gate passed (status=ok),
        # so promotion_intent=promote even though apply was a no-op.
        artifact_writes = [
            c for c in artifact_s3.put_object.call_args_list
            if c.kwargs["Key"].endswith("/from_trigger_optimizer.json")
        ]
        assert len(artifact_writes) == 1
        body = json.loads(artifact_writes[0].kwargs["Body"])
        assert body["promotion_intent"] == "promote"

    @patch("optimizer.trigger_optimizer.boto3")
    @patch("optimizer.recommendation_artifact.boto3")
    def test_status_not_ok_legacy_returns_early_artifact_still_writes(
        self, mock_artifact_boto3, mock_apply_boto3,
    ):
        legacy_s3 = MagicMock()
        artifact_s3 = MagicMock()
        mock_apply_boto3.client.return_value = legacy_s3
        mock_artifact_boto3.client.return_value = artifact_s3

        result = {"status": "insufficient_data"}
        outcome = apply(result, bucket="test-bucket")
        assert outcome["applied"] is False

        legacy_writes = [c for c in legacy_s3.put_object.call_args_list]
        assert len(legacy_writes) == 0

        # Artifact written with skip intent.
        artifact_call = next(
            c for c in artifact_s3.put_object.call_args_list
            if c.kwargs["Key"].endswith("/from_trigger_optimizer.json")
        )
        body = json.loads(artifact_call.kwargs["Body"])
        assert body["promotion_intent"] == "skip"


class TestApplyCutoverSkip:
    """When assembler cutover is enabled, trigger_optimizer's legacy
    read-modify-write of executor_params.json is skipped — the assembler
    is the sole writer."""

    @patch("optimizer.trigger_optimizer.boto3")
    @patch("optimizer.recommendation_artifact.boto3")
    def test_cutover_enabled_skips_legacy_live_write(
        self, mock_artifact_boto3, mock_apply_boto3, live_mode,
    ):
        set_cutover_enabled(True)
        legacy_s3 = MagicMock()
        artifact_s3 = MagicMock()
        mock_apply_boto3.client.return_value = legacy_s3
        mock_artifact_boto3.client.return_value = artifact_s3

        result = {"status": "ok", "disabled_triggers": ["pullback"]}
        outcome = apply(result, bucket="test-bucket")
        assert outcome["applied"] is False
        assert "cutover_mode" in outcome["reason"]

        # Legacy NEVER read or wrote the live key.
        assert legacy_s3.put_object.call_args_list == []
        assert legacy_s3.get_object.call_args_list == []

        # Artifact still written.
        artifact_writes = [
            c for c in artifact_s3.put_object.call_args_list
            if c.kwargs["Key"].endswith("/from_trigger_optimizer.json")
        ]
        assert len(artifact_writes) == 1


# ── analyze(): units and counts (alpha-engine-config-I12231) ────────────────


def _scorecard(*triggers):
    return {"status": "ok", "triggers": list(triggers), "summary": {}}


def _trig(name, n_trades=120, n_closed=80, alpha_pct=0.5, win_rate=0.55, slip_open_pct=0.0):
    return {
        "trigger": name, "n_trades": n_trades, "n_closed": n_closed,
        "avg_realized_alpha": alpha_pct, "win_rate_vs_spy": win_rate,
        "avg_slippage_vs_open": slip_open_pct,
    }


def _rec(result, name):
    return next(r for r in result["recommendations"] if r["trigger"] == name)


class TestAnalyzeCounts:

    def test_gate_counts_closed_entries_not_all_entries(self):
        # The live 2026-10-15 vwap row: 115 entries, 32 closed, win rate 34%.
        # Counting entries (the old rule) cleared the 50 gate and disabled it.
        result = trigger_optimizer.analyze(_scorecard(
            _trig("vwap", n_trades=115, n_closed=32, alpha_pct=-1.29, win_rate=0.344),
        ))
        assert result["status"] == "insufficient_data"
        assert result["disabled_triggers"] == []
        rec = _rec(result, "vwap")
        assert rec["action"] == "keep"
        assert "32 < 50 closed entries" in rec["reason"]

    def test_closed_count_at_the_gate_is_evaluated(self):
        result = trigger_optimizer.analyze(_scorecard(
            _trig("vwap", n_trades=60, n_closed=50, alpha_pct=0.2, win_rate=0.30),
        ))
        assert result["status"] == "ok"
        assert result["disabled_triggers"] == ["vwap_discount"]
        assert _rec(result, "vwap")["n_closed"] == 50

    def test_scorecard_without_n_closed_has_no_outcomes(self):
        trig = _trig("pullback", n_trades=500, alpha_pct=-5.0, win_rate=0.1)
        del trig["n_closed"]
        result = trigger_optimizer.analyze(_scorecard(trig))
        assert result["disabled_triggers"] == []


class TestAnalyzeUnits:

    def test_small_negative_alpha_in_percent_points_does_not_trip_the_fraction_threshold(self):
        # -0.3 pp = -0.003 > -0.005. The old code compared -0.3 to -0.005.
        result = trigger_optimizer.analyze(_scorecard(
            _trig("pullback", alpha_pct=-0.3, win_rate=0.55),
        ))
        rec = _rec(result, "pullback")
        assert rec["action"] == "keep"
        assert rec["avg_alpha"] == pytest.approx(-0.003)
        assert rec["avg_alpha_pct"] == pytest.approx(-0.3)

    def test_alpha_below_minus_half_percent_disables(self):
        result = trigger_optimizer.analyze(_scorecard(
            _trig("pullback", alpha_pct=-1.5, win_rate=0.55),
        ))
        rec = _rec(result, "pullback")
        assert rec["action"] == "disable"
        assert rec["reasons"] == ["negative avg alpha (-1.500%)"]

    def test_slippage_percent_points_convert_to_bps(self):
        result = trigger_optimizer.analyze(_scorecard(
            _trig("pullback", slip_open_pct=0.30),
            _trig("vwap", slip_open_pct=0.10),
        ))
        assert "high slippage (30bps)" in _rec(result, "pullback")["reasons"]
        assert _rec(result, "vwap")["reasons"] == []


class TestAnalyzeExecutorKeys:

    def test_disable_list_uses_the_keys_the_executor_checks(self):
        bad = {"alpha_pct": -2.0, "win_rate": 0.2}
        result = trigger_optimizer.analyze(_scorecard(
            _trig("pullback", **bad), _trig("vwap", **bad),
            _trig("support", **bad), _trig("graduated_entry", **bad),
        ))
        assert result["disabled_triggers"] == [
            "pullback", "vwap_discount", "support_bounce", "graduated_entry",
        ]

    @pytest.mark.parametrize("name", ["time_expiry", "other", "unknown"])
    def test_categories_the_executor_cannot_disable_are_never_listed(self, name):
        result = trigger_optimizer.analyze(_scorecard(
            _trig(name, alpha_pct=-9.0, win_rate=0.05),
        ))
        assert result["disabled_triggers"] == []
        assert _rec(result, name)["reason"] == "not a disable-able executor trigger"


# ── Observe-only: no live effect until the flip (alpha-engine-config-I12231) ──


_WOULD_DISABLE = {
    "status": "ok",
    "disabled_triggers": ["vwap_discount", "graduated_entry"],
    "total_evaluated": 2,
    "min_trades_threshold": 50,
}


class TestObserveOnly:

    def test_default_mode_is_observe(self):
        assert trigger_optimizer.apply_mode() == "observe"

    @pytest.mark.parametrize("value", ["true", "True", 1, "yes", None, False])
    def test_only_the_literal_true_arms_the_live_write(self, value):
        trigger_optimizer.init_config({"trigger_optimizer": {"live_apply": value}})
        assert trigger_optimizer.apply_mode() == "observe"

    @pytest.mark.parametrize("cutover", [False, True])
    @patch("optimizer.trigger_optimizer.boto3")
    @patch("optimizer.recommendation_artifact.boto3")
    def test_apply_never_reads_or_writes_the_live_key(
        self, mock_artifact_boto3, mock_apply_boto3, cutover,
    ):
        set_cutover_enabled(cutover)
        artifact_s3 = MagicMock()
        mock_artifact_boto3.client.return_value = artifact_s3

        outcome = apply(dict(_WOULD_DISABLE), bucket="test-bucket")

        assert outcome["applied"] is False
        assert outcome["reason"].startswith("observe_only")
        assert outcome["would_disable"] == ["vwap_discount", "graduated_entry"]
        # The legacy writer never even built an S3 client.
        mock_apply_boto3.client.assert_not_called()
        written = [c.kwargs["Key"] for c in artifact_s3.put_object.call_args_list]
        assert S3_PARAMS_KEY not in written

    @patch("optimizer.recommendation_artifact.boto3")
    def test_artifact_is_shadow_and_records_the_would_be_overlay(self, mock_boto3):
        s3 = MagicMock()
        mock_boto3.client.return_value = s3
        produce_artifact(dict(_WOULD_DISABLE), bucket="test-bucket")
        body = json.loads(s3.put_object.call_args.kwargs["Body"])
        assert body["promotion_intent"] == "shadow"
        assert body["recommended_params"]["disabled_triggers"] == [
            "vwap_discount", "graduated_entry",
        ]
        assert body["diagnostic"]["apply_mode"] == "observe"

    def test_assembler_under_cutover_leaves_live_disabled_triggers_unchanged(self):
        """End to end: what the executor reads (``disabled_triggers`` in
        config/executor_params.json) is unchanged, so no entry the executor
        would have taken is blocked and no order or position changes."""
        from botocore.exceptions import ClientError

        from optimizer.assembler import assemble
        from optimizer.recommendation_artifact import RecommendationArtifact

        captured = MagicMock()
        with patch("optimizer.recommendation_artifact.boto3") as mock_boto3:
            mock_boto3.client.return_value = captured
            produce_artifact(dict(_WOULD_DISABLE), bucket="b", run_date="2026-10-16")
        art = RecommendationArtifact.from_dict(
            json.loads(captured.put_object.call_args.kwargs["Body"])
        )
        art_key = "config/executor_params/recommendations/2026-10-16/from_trigger_optimizer.json"
        live = {"atr_multiplier": 2.0, "disabled_triggers": []}

        s3 = MagicMock()
        s3.list_objects_v2.return_value = {"Contents": [{"Key": art_key}]}

        def _get(Bucket, Key):
            if Key == S3_PARAMS_KEY:
                return {"Body": MagicMock(read=lambda: json.dumps(live).encode())}
            if Key == art_key:
                return {"Body": MagicMock(read=lambda: art.to_json().encode())}
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

        s3.get_object.side_effect = _get
        result = assemble(
            "b", "executor_params", "2026-10-16",
            s3_client=s3, write_assembled=False, cutover_enabled=True,
        )
        assert result.artifacts_seen["trigger_optimizer"]["promotion_intent"] == "shadow"
        assert result.status == "all_skip"
        assert result.assembled_params["disabled_triggers"] == []
        live_writes = [
            c for c in s3.put_object.call_args_list if c.kwargs.get("Key") == S3_PARAMS_KEY
        ]
        assert live_writes == []


@patch("optimizer.trigger_optimizer.boto3")
@patch("optimizer.recommendation_artifact.boto3")
def test_trades_db_to_apply_end_to_end_is_observe_only(
    mock_artifact_boto3, mock_apply_boto3, tmp_path,
):
    """trades.db -> joined scorecard -> analyze -> evaluate's apply path.
    60 VWAP entries, 55 closed losers: a truthful disable recommendation that
    must still change nothing live."""
    import sqlite3

    from analysis.trigger_scorecard import compute_trigger_scorecard
    from evaluate import _run_trigger_opt

    db = tmp_path / "trades.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE trades (trade_id TEXT, entry_trade_id TEXT, ticker TEXT, "
        "date TEXT, action TEXT, fill_price REAL, price_at_order REAL, "
        "signal_price REAL, trigger_type TEXT, trigger_price REAL, "
        "realized_return_pct REAL, realized_alpha_pct REAL, "
        "spy_return_during_hold REAL, slippage_vs_signal REAL, days_held REAL)"
    )
    rows = []
    for i in range(60):
        rows.append((f"E{i}", None, f"T{i}", "2026-09-01", "ENTER", 100.0, 100.0,
                     100.0, "VWAP discount 0.6% (VWAP=$100.00)", 100.0,
                     None, None, None, 0.0, None))
        if i < 55:
            rows.append((f"X{i}", f"E{i}", f"T{i}", "2026-09-10", "EXIT", 98.0, 98.0,
                         None, "atr_trail", 98.0, -2.0, -3.0, 1.0, None, 9))
    conn.executemany("INSERT INTO trades VALUES (" + ",".join("?" * 15) + ")", rows)
    conn.commit()
    conn.close()

    artifact_s3 = MagicMock()
    mock_artifact_boto3.client.return_value = artifact_s3

    scorecard = compute_trigger_scorecard(str(db))
    assert scorecard["summary"]["n_closed"] == 55
    result = _run_trigger_opt(scorecard, freeze=False, bucket="b", run_date="2026-10-16")

    assert result["disabled_triggers"] == ["vwap_discount"]
    assert result["apply_result"]["applied"] is False
    assert result["apply_result"]["reason"].startswith("observe_only")
    mock_apply_boto3.client.assert_not_called()
    body = json.loads(artifact_s3.put_object.call_args.kwargs["Body"])
    assert body["promotion_intent"] == "shadow"
