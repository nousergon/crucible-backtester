"""Unit tests for optimizer.barrier_sizing_optimizer (Task B3).

analyze() against synthetic predictor_outcomes (incl. the column-absent path),
_build_overlay_params, and apply() with mocked S3. Mirrors
test_predictor_sizing_optimizer.py.
"""
import json
import sqlite3
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from pipeline_common import ACTIVE_HORIZON_DAYS
from optimizer.assembler import set_cutover_enabled
from optimizer.barrier_sizing_optimizer import (
    S3_PARAMS_KEY,
    _build_overlay_params,
    analyze,
    apply,
)


@pytest.fixture(autouse=True)
def _reset_cutover_flag():
    set_cutover_enabled(False)
    yield
    set_cutover_enabled(False)


def _make_db(*, with_column: bool, rows: list[tuple] | None = None) -> str:
    """rows: (prediction_date, symbol, barrier_win_prob, actual_log_alpha)."""
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    conn = sqlite3.connect(f.name)
    cols = (
        "prediction_date TEXT, symbol TEXT, actual_log_alpha REAL, "
        "actual_5d_return REAL, horizon_days INTEGER"
    )
    if with_column:
        cols += ", barrier_win_prob REAL"
    conn.execute(f"CREATE TABLE predictor_outcomes ({cols})")
    for r in (rows or []):
        pred_date, sym, bwp, alpha = r
        conn.execute(
            "INSERT INTO predictor_outcomes "
            "(prediction_date, symbol, actual_log_alpha, actual_5d_return, "
            "horizon_days, barrier_win_prob) VALUES (?,?,?,?,?,?)",
            (pred_date, sym, alpha, None, ACTIVE_HORIZON_DAYS, bwp),
        )
    conn.commit()
    conn.close()
    return f.name


def _correlated_rows(n_weeks=8, per_week=5, sign=1.0):
    """barrier_win_prob correlated (sign) with actual_log_alpha, across weeks."""
    rows = []
    # distinct ISO weeks: Mondays one week apart
    base_days = [5, 12, 19, 26, 33, 40, 47, 54]  # day-of-year-ish offsets
    for wk in range(n_weeks):
        # 2026-based dates, distinct weeks
        month = 1 + (base_days[wk] // 28)
        day = 1 + (base_days[wk] % 28)
        date_str = f"2026-{month:02d}-{day:02d}"
        for j in range(per_week):
            bwp = 0.1 + 0.1 * j  # 0.1..0.5 spread within the day
            alpha = sign * bwp * 0.1  # monotone in bwp → strong rank IC
            rows.append((date_str, f"T{wk}_{j}", bwp, alpha))
    return rows


class TestAnalyze:
    def test_column_absent(self):
        db = _make_db(with_column=False)
        result = analyze(db)
        assert result["status"] == "barrier_win_prob_column_absent"
        assert "note" in result

    def test_insufficient_data(self):
        db = _make_db(with_column=True, rows=_correlated_rows(n_weeks=1, per_week=3))
        result = analyze(db)
        assert result["status"] == "insufficient_data"
        assert result["n_samples"] == 3

    def test_ok_enable_on_positive_ic(self):
        db = _make_db(with_column=True, rows=_correlated_rows(sign=1.0))
        result = analyze(db)
        assert result["status"] == "ok"
        assert result["overall_rank_ic"] > 0.05
        assert result["recommendation"] == "enable"

    def test_keep_disabled_on_negative_ic(self):
        db = _make_db(with_column=True, rows=_correlated_rows(sign=-1.0))
        result = analyze(db)
        assert result["status"] == "ok"
        assert result["overall_rank_ic"] < 0.05
        assert result["recommendation"] == "keep_disabled"


class TestBuildOverlayParams:
    def test_emits_overlay_fields(self):
        params, keys = _build_overlay_params({"overall_rank_ic": 0.12})
        assert set(keys) == {
            "barrier_win_prob_sizing_enabled",
            "barrier_win_prob_sizing_min",
            "barrier_win_prob_sizing_range",
            "barrier_win_prob_sizing_updated_at",
            "barrier_win_prob_sizing_ic",
        }
        assert params["barrier_win_prob_sizing_enabled"] is True
        assert params["barrier_win_prob_sizing_ic"] == 0.12


class TestApply:
    def _enable_result(self):
        return {"status": "ok", "recommendation": "enable", "overall_rank_ic": 0.11}

    @patch("optimizer.barrier_sizing_optimizer.produce_artifact")
    @patch("optimizer.barrier_sizing_optimizer.boto3")
    def test_applies_on_enable(self, mock_boto3, _mock_artifact):
        s3 = MagicMock()
        s3.get_object.side_effect = Exception("no existing params")
        mock_boto3.client.return_value = s3
        res = apply(self._enable_result(), "bucket")
        assert res["applied"] is True
        # wrote the flag
        body = json.loads(s3.put_object.call_args.kwargs["Body"])
        assert body["barrier_win_prob_sizing_enabled"] is True

    @patch("optimizer.barrier_sizing_optimizer.produce_artifact")
    @patch("optimizer.barrier_sizing_optimizer.boto3")
    def test_skips_when_keep_disabled(self, mock_boto3, _mock_artifact):
        res = apply({"status": "ok", "recommendation": "keep_disabled",
                     "overall_rank_ic": 0.01}, "bucket")
        assert res["applied"] is False
        mock_boto3.client.assert_not_called()

    @patch("optimizer.barrier_sizing_optimizer.produce_artifact")
    @patch("optimizer.barrier_sizing_optimizer.boto3")
    def test_skips_when_not_ok(self, mock_boto3, _mock_artifact):
        res = apply({"status": "barrier_win_prob_column_absent"}, "bucket")
        assert res["applied"] is False

    @patch("optimizer.barrier_sizing_optimizer.produce_artifact")
    @patch("optimizer.barrier_sizing_optimizer.boto3")
    def test_skips_when_already_enabled(self, mock_boto3, _mock_artifact):
        s3 = MagicMock()
        s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(
                {"barrier_win_prob_sizing_enabled": True}).encode())
        }
        mock_boto3.client.return_value = s3
        res = apply(self._enable_result(), "bucket")
        assert res["applied"] is False
        assert "already enabled" in res["reason"]
        s3.put_object.assert_not_called()

    @patch("optimizer.barrier_sizing_optimizer.produce_artifact")
    def test_skips_in_cutover_mode(self, _mock_artifact):
        set_cutover_enabled(True)
        res = apply(self._enable_result(), "bucket")
        assert res["applied"] is False
        assert "cutover" in res["reason"]


class TestApplySignificanceEnforce:
    """config#1426 Phase 4 — significance ENFORCE wiring (default OFF)."""

    def setup_method(self):
        import optimizer.barrier_sizing_optimizer as mod
        self._saved_cfg = mod._cfg

    def teardown_method(self):
        import optimizer.barrier_sizing_optimizer as mod
        mod._cfg = self._saved_cfg

    def _set_cfg(self, enforce: bool):
        import optimizer.barrier_sizing_optimizer as mod
        mod._cfg = {"enforce_significance": enforce}

    def _enable_result(self, verdict):
        return {"status": "ok", "recommendation": "enable",
                "overall_rank_ic": 0.11, "significance_observe": verdict}

    @patch("optimizer.barrier_sizing_optimizer.produce_artifact")
    @patch("optimizer.barrier_sizing_optimizer.boto3")
    def test_default_off_applies_even_when_would_block(self, mock_boto3, _art):
        """CRITICAL non-enforcement guarantee: default path unchanged."""
        self._set_cfg(False)
        s3 = MagicMock()
        s3.get_object.side_effect = Exception("no params")
        mock_boto3.client.return_value = s3
        res = apply(self._enable_result({"would_block": True}), "bucket")
        assert res["applied"] is True

    @patch("optimizer.barrier_sizing_optimizer.produce_artifact")
    @patch("optimizer.barrier_sizing_optimizer.boto3")
    def test_enforce_blocks_insignificant(self, mock_boto3, _art):
        self._set_cfg(True)
        s3 = MagicMock()
        mock_boto3.client.return_value = s3
        res = apply(self._enable_result({"would_block": True}), "bucket")
        assert res["applied"] is False
        assert "significance enforce" in res["reason"]
        s3.put_object.assert_not_called()

    @patch("optimizer.barrier_sizing_optimizer.produce_artifact")
    @patch("optimizer.barrier_sizing_optimizer.boto3")
    def test_enforce_allows_significant(self, mock_boto3, _art):
        self._set_cfg(True)
        s3 = MagicMock()
        s3.get_object.side_effect = Exception("no params")
        mock_boto3.client.return_value = s3
        res = apply(self._enable_result({"would_block": False}), "bucket")
        assert res["applied"] is True

    @patch("optimizer.barrier_sizing_optimizer.produce_artifact")
    @patch("optimizer.barrier_sizing_optimizer.boto3")
    def test_enforce_missing_verdict_blocks_conservatively(self, mock_boto3, _art):
        self._set_cfg(True)
        mock_boto3.client.return_value = MagicMock()
        res = apply(self._enable_result(None), "bucket")
        assert res["applied"] is False
