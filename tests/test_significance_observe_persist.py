"""Tests for durable persistence of the observe-first significance verdicts
(config#1426). The verdict must land in the per-Saturday metrics.json so the
observe→enforce soak (Phase 4) has a reviewable history — not just ephemeral
spot-instance logs.

Pins:
  1. reporter.save() writes significance_observe into metrics.json when present.
  2. reporter.save() omits the key entirely when no verdict is supplied.
  3. evaluate._collect_significance_observe gathers per-optimizer records and
     drops absent/None ones (CI-only — importing evaluate needs the full deps).
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("pandas")

from reporter import save  # noqa: E402

_SAMPLE = {
    "weight_result": {
        "gate": "weight_optimizer", "significant": False, "would_block": True,
        "did_promote": True, "promotes_on_undefended_evidence": True, "enforced": False,
    },
    "predictor_sizing": {
        "gate": "predictor_sizing", "significant": True, "would_block": False,
        "did_promote": True, "promotes_on_undefended_evidence": False, "enforced": False,
    },
}


def _read_metrics(tmp_path, run_date="2026-07-04"):
    return json.loads((tmp_path / run_date / "metrics.json").read_text())


class TestSavePersistsSignificanceObserve:
    def test_present_block_is_persisted(self, tmp_path):
        save(
            report_md="# report",
            signal_quality={"status": "ok", "overall": {"n": 1}},
            score_analysis=[],
            run_date="2026-07-04",
            results_dir=str(tmp_path),
            significance_observe=_SAMPLE,
        )
        metrics = _read_metrics(tmp_path)
        assert metrics["significance_observe"] == _SAMPLE
        assert metrics["significance_observe"]["weight_result"]["enforced"] is False

    def test_absent_block_is_omitted(self, tmp_path):
        save(
            report_md="# report",
            signal_quality={"status": "ok", "overall": {"n": 1}},
            score_analysis=[],
            run_date="2026-07-04",
            results_dir=str(tmp_path),
            significance_observe=None,
        )
        assert "significance_observe" not in _read_metrics(tmp_path)


class TestCollectSignificanceObserve:
    """Imports evaluate (needs the full backtester deps) — runs in CI, skips
    in the lean isolated venv."""

    def _collect(self):
        evaluate = pytest.importorskip("evaluate")
        return evaluate._collect_significance_observe

    def test_gathers_present_records_and_drops_absent(self):
        collect = self._collect()
        opt_results = {
            "weight_result": {"status": "ok", "significance_observe": _SAMPLE["weight_result"]},
            "veto_result": {"status": "ok", "significance_observe": None},
            "predictor_sizing": {"status": "ok", "significance_observe": _SAMPLE["predictor_sizing"]},
            "barrier_sizing": {"status": "barrier_win_prob_column_absent"},
            "stance_sizing": "not-a-dict",
        }
        out = collect(opt_results)
        assert set(out) == {"weight_result", "predictor_sizing"}

    def test_none_when_empty(self):
        collect = self._collect()
        assert collect({"weight_result": {"status": "ok"}}) is None
        assert collect({}) is None
