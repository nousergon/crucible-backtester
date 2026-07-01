"""Unit tests for optimizer.stance_sizing_optimizer (L300).

analyze() against synthetic score_performance (incl. the stance-column-absent
path), _build_overlay_params, and apply() with mocked S3 + the assembler
cutover gate. Mirrors test_barrier_sizing_optimizer.py.
"""
import json
import sqlite3
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from optimizer.assembler import set_cutover_enabled
from optimizer.stance_sizing_optimizer import (
    S3_PARAMS_KEY,
    _build_overlay_params,
    analyze,
    apply,
    init_config,
)


@pytest.fixture(autouse=True)
def _reset():
    set_cutover_enabled(False)
    init_config({})  # reset module config to defaults
    yield
    set_cutover_enabled(False)


def _make_db(*, with_stance: bool, rows: list[tuple] | None = None) -> str:
    """rows: (date, symbol, stance, return, spy_return).

    config#1451/#1452: the optimizer now reads the canonical `log_alpha_21d`
    keyed by `score_date` (the retired 10d horizon is dark). We keep the
    (return, spy_return) tuple interface and store alpha = return − spy_return
    as `log_alpha_21d`, so the test-data builders are unchanged.
    """
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    conn = sqlite3.connect(f.name)
    cols = "score_date TEXT, symbol TEXT, log_alpha_21d REAL"
    if with_stance:
        cols += ", stance TEXT"
    conn.execute(f"CREATE TABLE score_performance ({cols})")
    for r in (rows or []):
        d, sym, stance, r10, spy10 = r
        alpha = (r10 - spy10) if (r10 is not None and spy10 is not None) else None
        if with_stance:
            conn.execute(
                "INSERT INTO score_performance "
                "(score_date, symbol, log_alpha_21d, stance) "
                "VALUES (?,?,?,?)", (d, sym, alpha, stance))
        else:
            conn.execute(
                "INSERT INTO score_performance "
                "(score_date, symbol, log_alpha_21d) "
                "VALUES (?,?,?)", (d, sym, alpha))
    conn.commit()
    conn.close()
    return f.name


def _rows_two_qualifying(n_weeks=8, per_week=5):
    """momentum: consistently +2% alpha; value: consistently +0.3% alpha
    (both reliably positive → qualify; spread ~1.7% > 0.5% gate). catalyst:
    reliably NEGATIVE alpha (qualifies on negative-sign consistency)."""
    rows = []
    for w in range(n_weeks):
        d = f"2026-0{1 + w // 4}-{(w % 4) * 7 + 1:02d}"
        for i in range(per_week):
            rows.append((d, f"M{w}{i}", "momentum", 0.025, 0.005))   # +2.0% alpha
            rows.append((d, f"V{w}{i}", "value", 0.008, 0.005))      # +0.3% alpha
            rows.append((d, f"C{w}{i}", "catalyst", -0.015, 0.005))  # -2.0% alpha
    return rows


class TestAnalyze:
    def test_stance_column_absent(self):
        db = _make_db(with_stance=False, rows=[])
        out = analyze(db)
        assert out["status"] == "stance_column_absent"

    def test_empty_history_insufficient(self):
        db = _make_db(with_stance=True, rows=[])
        out = analyze(db)
        assert out["status"] == "insufficient_stance_history"

    def test_enable_when_two_stances_qualify_with_spread(self):
        db = _make_db(with_stance=True, rows=_rows_two_qualifying())
        out = analyze(db)
        assert out["status"] == "ok"
        assert out["recommendation"] == "enable"
        assert out["stance_alpha_spread"] > 0.005
        rec = out["recommended_multipliers"]
        # All four stances present; bounded to [0.4, 1.1].
        for s in ("momentum", "value", "quality", "catalyst"):
            assert 0.4 <= rec[s] <= 1.1
        # Higher-alpha momentum sized >= lower-alpha catalyst.
        assert rec["momentum"] >= rec["catalyst"]

    def test_non_qualifying_stance_keeps_factory_default(self):
        # quality has no rows → keeps its factory default (0.8).
        db = _make_db(with_stance=True, rows=_rows_two_qualifying())
        out = analyze(db)
        assert out["per_stance"]["quality"]["qualifies"] is False
        assert out["recommended_multipliers"]["quality"] == pytest.approx(0.8)


class TestOverlayParams:
    def test_overlay_payload_has_stance_keys(self):
        db = _make_db(with_stance=True, rows=_rows_two_qualifying())
        out = analyze(db)
        params, keys = _build_overlay_params(out)
        for s in ("momentum", "value", "quality", "catalyst"):
            assert f"stance_size_{s}" in params
        assert "stance_sizing_updated_at" in params


class TestApply:
    def test_cutover_mode_defers_to_assembler(self):
        set_cutover_enabled(True)
        db = _make_db(with_stance=True, rows=_rows_two_qualifying())
        out = analyze(db)
        with patch("optimizer.stance_sizing_optimizer.produce_artifact", return_value={"written": True}):
            res = apply(out, bucket="b")
        assert res["applied"] is False
        assert "cutover" in res["reason"]

    def test_keep_disabled_not_applied(self):
        res = apply({"status": "ok", "recommendation": "keep_disabled",
                     "stance_alpha_spread": 0.001}, bucket="b")
        assert res["applied"] is False

    def test_applies_field_overlay_when_enabled(self):
        db = _make_db(with_stance=True, rows=_rows_two_qualifying())
        out = analyze(db)
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: json.dumps({"min_score": 70}).encode())}
        with patch("boto3.client", return_value=mock_s3), \
             patch("optimizer.stance_sizing_optimizer.produce_artifact", return_value={"written": True}):
            res = apply(out, bucket="b")
        assert res["applied"] is True
        mock_s3.put_object.assert_called_once()
        _, kw = mock_s3.put_object.call_args
        body = json.loads(kw["Body"])
        assert "stance_size_momentum" in body
        assert body["min_score"] == 70  # field_overlay preserves existing keys
        assert kw["Key"] == S3_PARAMS_KEY


class TestApplySignificanceEnforce:
    """config#1426 Phase 4 — significance ENFORCE wiring (default OFF)."""

    def _blocked_result(self, verdict):
        # Minimal enable result; _build_overlay_params is not reached when the
        # enforce block short-circuits, so no multipliers needed here.
        return {"status": "ok", "recommendation": "enable",
                "stance_alpha_spread": 0.02, "significance_observe": verdict}

    def test_default_off_applies_even_when_would_block(self):
        """CRITICAL non-enforcement guarantee: default path unchanged."""
        init_config({})  # no enforce_significance → defaults False
        db = _make_db(with_stance=True, rows=_rows_two_qualifying())
        out = analyze(db)
        out["significance_observe"] = {"would_block": True}
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: json.dumps({}).encode())}
        with patch("boto3.client", return_value=mock_s3), \
             patch("optimizer.stance_sizing_optimizer.produce_artifact", return_value={}):
            res = apply(out, bucket="b")
        assert res["applied"] is True

    def test_enforce_blocks_insignificant(self):
        init_config({"stance_sizing_optimizer": {"enforce_significance": True}})
        mock_s3 = MagicMock()
        with patch("boto3.client", return_value=mock_s3), \
             patch("optimizer.stance_sizing_optimizer.produce_artifact", return_value={}):
            res = apply(self._blocked_result({"would_block": True}), bucket="b")
        assert res["applied"] is False
        assert "significance enforce" in res["reason"]
        mock_s3.put_object.assert_not_called()

    def test_enforce_allows_significant(self):
        init_config({"stance_sizing_optimizer": {"enforce_significance": True}})
        db = _make_db(with_stance=True, rows=_rows_two_qualifying())
        out = analyze(db)
        out["significance_observe"] = {"would_block": False}
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: json.dumps({}).encode())}
        with patch("boto3.client", return_value=mock_s3), \
             patch("optimizer.stance_sizing_optimizer.produce_artifact", return_value={}):
            res = apply(out, bucket="b")
        assert res["applied"] is True

    def test_enforce_missing_verdict_blocks_conservatively(self):
        init_config({"stance_sizing_optimizer": {"enforce_significance": True}})
        mock_s3 = MagicMock()
        with patch("boto3.client", return_value=mock_s3), \
             patch("optimizer.stance_sizing_optimizer.produce_artifact", return_value={}):
            res = apply(self._blocked_result(None), bucket="b")
        assert res["applied"] is False
