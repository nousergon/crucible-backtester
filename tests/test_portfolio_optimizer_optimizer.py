"""Tests for optimizer.portfolio_optimizer_optimizer — recommend + safeguarded
apply of the MVO optimizer's own params (config#1057 increment 2).

Focus: the SAFEGUARDS. Only risk_aversion + tcost_bps are writable; values are
clamped; the promote gate (margin + winner≠baseline) decides; live write is
flag-gated, rollback-snapshotted, and shadow-audited.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from optimizer.portfolio_optimizer_optimizer import (
    PARAM_BOUNDS,
    S3_PARAMS_KEY,
    S3_SHADOW_PREFIX,
    WRITABLE_PARAMS,
    _clamp,
    _effective_param_bounds,
    apply,
    recommend,
)


def test_public_default_floor_is_generic():
    # The repo ships the frozen generic floor; the aggressive operating floor is
    # private (divergence policy) — never hardcode <3.0 here.
    assert PARAM_BOUNDS["risk_aversion"][0] == 3.0


def test_private_floor_override_lowers_band():
    # Without the private config, a λ=2.0 recommendation is clamped up to 3.0.
    clamped, notes = _clamp({"risk_aversion": 2.0}, None)
    assert clamped["risk_aversion"] == 3.0 and notes
    # With the private tuner config floor=1.0, λ=2.0 passes through unclamped.
    cfg = {"portfolio_optimizer_tuner": {"risk_aversion_floor": 1.0}}
    assert _effective_param_bounds(cfg)["risk_aversion"] == (1.0, 10.0)
    clamped2, notes2 = _clamp({"risk_aversion": 2.0}, cfg)
    assert clamped2["risk_aversion"] == 2.0 and not notes2


def _sweep(winner, baseline="baseline_ra5_tc5", win_sortino=1.2, base_sortino=0.8,
           win_cfg=None, status="ok"):
    win_cfg = win_cfg or {"risk_aversion": 3.0, "tcost_bps": 2.0}
    cells = {
        baseline: {"sortino_ratio": base_sortino,
                   "cell_cfg": {"risk_aversion": 5.0, "tcost_bps": 5.0}},
    }
    if winner and winner != baseline:
        cells[winner] = {"sortino_ratio": win_sortino, "cell_cfg": win_cfg}
    return {"status": status, "winner_name": winner, "baseline_name": baseline,
            "cells": cells}


# ── recommend: the promote gate + writable-set + clamp ───────────────────────


class TestRecommend:
    def test_ok_when_winner_beats_baseline_by_margin(self):
        r = recommend(_sweep("ra3_tc2", win_sortino=1.2, base_sortino=0.8))  # +50%
        assert r["status"] == "ok"
        assert r["recommended_params"] == {"risk_aversion": 3.0, "tcost_bps": 2.0}
        assert r["margin"] == pytest.approx((1.2 - 0.8) / 0.8)

    def test_blocked_on_insufficient_margin(self):
        # +5% < default 15% promote_margin
        r = recommend(_sweep("ra3_tc2", win_sortino=0.84, base_sortino=0.80))
        assert r["status"] == "blocked"
        assert "insufficient margin" in r["reason"]

    def test_no_change_when_winner_is_baseline(self):
        r = recommend(_sweep("baseline_ra5_tc5"))
        assert r["status"] == "no_change"

    def test_no_change_when_no_winner(self):
        r = recommend(_sweep(None))
        assert r["status"] == "no_change"

    def test_blocked_when_sweep_not_ok(self):
        r = recommend(_sweep("ra3_tc2", status="skipped"))
        assert r["status"] == "blocked"

    def test_only_writable_params_extracted(self):
        # winner cfg carries a non-writable safety knob → must be dropped
        win_cfg = {"risk_aversion": 4.0, "tcost_bps": 3.0,
                   "max_sector_pct": 0.99, "max_daily_turnover": 0.99}
        r = recommend(_sweep("x", win_cfg=win_cfg))
        assert set(r["recommended_params"]) == set(WRITABLE_PARAMS)
        assert "max_sector_pct" not in r["recommended_params"]
        assert "max_daily_turnover" not in r["recommended_params"]

    def test_out_of_band_value_is_clamped(self):
        lo, hi = PARAM_BOUNDS["risk_aversion"]
        win_cfg = {"risk_aversion": hi + 100, "tcost_bps": 2.0}
        r = recommend(_sweep("x", win_cfg=win_cfg))
        assert r["recommended_params"]["risk_aversion"] == hi
        assert any("risk_aversion" in n for n in r["clamp_notes"])

    def test_custom_promote_margin_from_config(self):
        # +20% clears a 10% margin but not a 30% margin
        sweep = _sweep("ra3_tc2", win_sortino=0.96, base_sortino=0.80)
        assert recommend(sweep, config={"portfolio_optimizer_tuner": {"promote_margin": 0.10}})["status"] == "ok"
        assert recommend(sweep, config={"portfolio_optimizer_tuner": {"promote_margin": 0.30}})["status"] == "blocked"


# ── apply: flag-gated live write + rollback snapshot + shadow audit ──────────


class TestApply:
    def _ok_rec(self):
        return {"status": "ok", "recommended_params": {"risk_aversion": 3.0, "tcost_bps": 2.0},
                "winner_name": "ra3_tc2", "baseline_name": "baseline_ra5_tc5",
                "margin": 0.5, "clamp_notes": []}

    @patch("optimizer.portfolio_optimizer_optimizer.boto3")
    def test_applies_to_live_when_enabled(self, mock_boto3):
        s3 = MagicMock()
        mock_boto3.client.return_value = s3
        with patch("optimizer.rollback.save_previous") as save_prev:
            out = apply(self._ok_rec(), "bkt",
                        config={"portfolio_optimizer_tuner": {"auto_apply_enabled": True}})
        assert out["applied"] is True
        keys = [c.kwargs["Key"] for c in s3.put_object.call_args_list]
        assert S3_PARAMS_KEY in keys                       # live write
        assert any(S3_SHADOW_PREFIX in k for k in keys)    # shadow audit
        save_prev.assert_called_once_with("bkt", "portfolio_optimizer")  # rollback snapshot

    @patch("optimizer.portfolio_optimizer_optimizer.boto3")
    def test_shadow_only_when_disabled(self, mock_boto3):
        s3 = MagicMock()
        mock_boto3.client.return_value = s3
        out = apply(self._ok_rec(), "bkt",
                    config={"portfolio_optimizer_tuner": {"auto_apply_enabled": False}})
        assert out["applied"] is False
        assert out["shadow_only"] is True
        keys = [c.kwargs["Key"] for c in s3.put_object.call_args_list]
        assert any(S3_SHADOW_PREFIX in k for k in keys)    # shadow written
        assert S3_PARAMS_KEY not in keys                   # live NOT touched

    @patch("optimizer.portfolio_optimizer_optimizer.boto3")
    def test_not_applied_when_status_not_ok(self, mock_boto3):
        mock_boto3.client.return_value = MagicMock()
        out = apply({"status": "blocked", "reason": "insufficient margin"}, "bkt")
        assert out["applied"] is False

    @patch("optimizer.portfolio_optimizer_optimizer.boto3")
    def test_write_boundary_reclamps(self, mock_boto3):
        s3 = MagicMock()
        mock_boto3.client.return_value = s3
        rec = self._ok_rec()
        rec["recommended_params"] = {"risk_aversion": 999.0, "tcost_bps": 2.0}  # out of band
        with patch("optimizer.rollback.save_previous"):
            apply(rec, "bkt")
        live = [c for c in s3.put_object.call_args_list if c.kwargs["Key"] == S3_PARAMS_KEY]
        body = json.loads(live[0].kwargs["Body"])
        assert body["risk_aversion"] == PARAM_BOUNDS["risk_aversion"][1]  # clamped to hi


def test_registered_in_rollback_config_keys():
    """The live key must be in rollback.CONFIG_KEYS so save_previous + the weekly
    regression monitor's rollback_all auto-revert cover it (key safeguard)."""
    from optimizer.rollback import CONFIG_KEYS
    assert CONFIG_KEYS.get("portfolio_optimizer") == "config/portfolio_optimizer.json"
