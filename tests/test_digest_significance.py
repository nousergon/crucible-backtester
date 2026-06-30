"""Tests for the digest's promotion-gate significance headline (config#1444
item 1, email half)."""

from __future__ import annotations

from reporter import _section_significance_observe, build_digest

_UNDEFENDED = {
    "weight_result": {"promotes_on_undefended_evidence": True,
                      "detail": {"per_subscore": {}, "n_test": 100}},
    "predictor_sizing": {"promotes_on_undefended_evidence": False,
                         "detail": {"status": "ok", "ic": 0.1}},
    "barrier_sizing": {"promotes_on_undefended_evidence": False,
                       "detail": {"status": "insufficient_data"}},  # no verdict
}
_ALL_CLEAR = {
    "weight_result": {"promotes_on_undefended_evidence": False,
                      "detail": {"per_subscore": {}, "n_test": 100}},
    "predictor_sizing": {"promotes_on_undefended_evidence": False,
                         "detail": {"status": "ok", "ic": 0.1}},
}


class TestSection:
    def test_undefended_headline_counts_only_verdicts(self):
        out = "\n".join(_section_significance_observe(_UNDEFENDED))
        assert "Promotion-Gate Significance (observe)" in out
        assert "1 of 2" in out          # 1 undefended of 2 with a verdict (barrier excluded)
        assert "⚠" in out

    def test_all_clear(self):
        out = "\n".join(_section_significance_observe(_ALL_CLEAR))
        assert "All **2**" in out

    def test_no_verdicts(self):
        out = "\n".join(_section_significance_observe(
            {"barrier_sizing": {"detail": {"status": "insufficient_data"}}}))
        assert "No significance verdicts" in out


class TestBuildDigestIntegration:
    def test_section_present_when_supplied(self):
        md = build_digest("2026-07-04", significance_observe=_UNDEFENDED)
        assert "Promotion-Gate Significance (observe)" in md
        assert "1 of 2" in md

    def test_section_omitted_when_absent(self):
        md = build_digest("2026-07-04")
        assert "Promotion-Gate Significance" not in md
