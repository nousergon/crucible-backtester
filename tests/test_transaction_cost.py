"""Tests for the W3.4 (L4469) transaction-cost model.

Pins the load-bearing properties: the square-root impact law (monotone in
participation, concave), the ADV-absent fallback (degrade to half-spread +
commission, never error), config wiring, and the dollar-cost arithmetic.
"""
from __future__ import annotations

import math

from analysis.transaction_cost import TransactionCostModel


def _model(**kw):
    base = dict(half_spread_bps=2.5, impact_coef_bps=10.0, commission_bps=0.5, min_cost_bps=0.0)
    base.update(kw)
    return TransactionCostModel(**base)


class TestPerSideBps:
    def test_floor_is_half_spread_plus_commission_when_no_impact(self):
        m = _model()
        # Zero notional ⇒ no impact term ⇒ just half-spread + commission.
        assert m.per_side_bps(0.0, adv_dollar=1e9) == 2.5 + 0.5

    def test_sqrt_law_value_at_known_participation(self):
        m = _model()
        # participation = 1e6 / 1e8 = 0.01 → impact = 10 * sqrt(0.01) = 1.0 bps
        bps = m.per_side_bps(notional=1e6, adv_dollar=1e8)
        assert math.isclose(bps, 2.5 + 1.0 + 0.5, rel_tol=1e-9)

    def test_impact_monotone_increasing_in_participation(self):
        m = _model()
        small = m.per_side_bps(1e5, adv_dollar=1e8)
        big = m.per_side_bps(1e7, adv_dollar=1e8)
        assert big > small  # larger order → larger impact

    def test_impact_is_concave_sqrt_not_linear(self):
        m = _model(half_spread_bps=0.0, commission_bps=0.0)
        # Doubling notional should less-than-double the (pure-impact) cost.
        c1 = m.per_side_bps(1e6, adv_dollar=1e8)
        c2 = m.per_side_bps(2e6, adv_dollar=1e8)
        assert c2 < 2 * c1
        assert math.isclose(c2 / c1, math.sqrt(2), rel_tol=1e-9)

    def test_adv_absent_degrades_to_spread_plus_commission(self):
        m = _model()
        for adv in (None, 0.0, -1.0):
            assert m.per_side_bps(1e6, adv_dollar=adv) == 2.5 + 0.5  # no impact, no error

    def test_min_cost_floor_applied(self):
        m = _model(half_spread_bps=0.0, commission_bps=0.0, min_cost_bps=3.0)
        assert m.per_side_bps(0.0, adv_dollar=1e9) == 3.0


class TestCostForTurnover:
    def test_zero_turnover_zero_cost(self):
        assert _model().cost_for_turnover(0.0, adv_dollar=1e8) == 0.0

    def test_dollar_cost_arithmetic(self):
        m = _model()
        # $1e6 notional, participation 0.01 → 4.0 bps → $1e6 * 4e-4 = $400
        cost = m.cost_for_turnover(1e6, adv_dollar=1e8)
        assert math.isclose(cost, 1e6 * (2.5 + 1.0 + 0.5) / 1e4, rel_tol=1e-9)

    def test_zero_cost_model_yields_zero(self):
        m = _model(half_spread_bps=0.0, impact_coef_bps=0.0, commission_bps=0.0)
        assert m.cost_for_turnover(5e6, adv_dollar=1e8) == 0.0


class TestFromConfig:
    def test_defaults_when_block_absent(self):
        m = TransactionCostModel.from_config({})
        assert m.half_spread_bps == 2.5 and m.impact_coef_bps == 10.0

    def test_none_config_uses_defaults(self):
        m = TransactionCostModel.from_config(None)
        assert m.commission_bps == 0.5

    def test_overrides_applied(self):
        m = TransactionCostModel.from_config(
            {"transaction_cost": {"half_spread_bps": 1.0, "impact_coef_bps": 20.0,
                                  "commission_bps": 0.1, "min_cost_bps": 0.5}}
        )
        assert (m.half_spread_bps, m.impact_coef_bps, m.commission_bps, m.min_cost_bps) == (
            1.0, 20.0, 0.1, 0.5)
