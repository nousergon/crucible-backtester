"""transaction_cost.py — institutional transaction-cost model (W3.4, L4469).

A square-root market-impact model (Almgren-Chriss / Kissell "square-root law")
on top of a half-spread + commission floor. Per-side cost in basis points:

    per_side_bps(notional, adv_dollar) = half_spread_bps
        + impact_coef_bps * sqrt(notional / adv_dollar)   # √-law market impact
        + commission_bps

This turns the backtester's GROSS per-horizon alpha into NET (see
``analysis/horizon_net_alpha.py``) so the horizon-cutover question (target 21d
vs 60/90d?) is judged net-of-cost — longer horizons rebalance less (turnover
~1/h), so they can win net even at similar gross IC. Gross IC alone (the
predictor's leak-free horizon curve) does NOT settle it.

Design:
- **Pure + config-driven.** No I/O, no logging — fully unit-testable; callers
  own coverage logging (e.g. how many names lacked ADV).
- **ADV-absent fallback.** When average-daily-dollar-volume is missing/≤0 the
  impact term drops to 0 (half-spread + commission only) rather than erroring —
  the conservative degrade, not a silent zero.
- **Defaults** calibrated for liquid large-cap US equities on an IBKR-paper
  book (overridable via the ``transaction_cost`` config block): half-spread
  ~2.5bps (≈5bps full spread), impact ~10bps at 100% participation,
  commission ~0.5bps ($0.005/share ≈ <1bp on large-caps).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Institutional defaults — large-cap US equities / IBKR-paper calibration.
_DEFAULT_HALF_SPREAD_BPS = 2.5
_DEFAULT_IMPACT_COEF_BPS = 10.0
_DEFAULT_COMMISSION_BPS = 0.5
_DEFAULT_MIN_COST_BPS = 0.0


@dataclass(frozen=True)
class TransactionCostModel:
    """Per-side equity transaction cost via the square-root impact law."""

    half_spread_bps: float = _DEFAULT_HALF_SPREAD_BPS
    impact_coef_bps: float = _DEFAULT_IMPACT_COEF_BPS
    commission_bps: float = _DEFAULT_COMMISSION_BPS
    min_cost_bps: float = _DEFAULT_MIN_COST_BPS

    @classmethod
    def from_config(cls, config: dict | None) -> "TransactionCostModel":
        """Build from the optional ``transaction_cost`` block of the backtester
        config; absent keys fall back to the institutional defaults."""
        cfg = ((config or {}).get("transaction_cost") or {}) if config else {}
        return cls(
            half_spread_bps=float(cfg.get("half_spread_bps", _DEFAULT_HALF_SPREAD_BPS)),
            impact_coef_bps=float(cfg.get("impact_coef_bps", _DEFAULT_IMPACT_COEF_BPS)),
            commission_bps=float(cfg.get("commission_bps", _DEFAULT_COMMISSION_BPS)),
            min_cost_bps=float(cfg.get("min_cost_bps", _DEFAULT_MIN_COST_BPS)),
        )

    def per_side_bps(self, notional: float, adv_dollar: float | None) -> float:
        """Per-side cost (bps) to trade ``notional`` dollars of a name whose
        average daily dollar volume is ``adv_dollar``. ADV missing/≤0 → the
        √-impact term drops to 0 (half-spread + commission only)."""
        notional = abs(float(notional))
        impact_bps = 0.0
        if adv_dollar is not None and adv_dollar > 0 and notional > 0:
            participation = notional / float(adv_dollar)
            impact_bps = self.impact_coef_bps * math.sqrt(participation)
        bps = self.half_spread_bps + impact_bps + self.commission_bps
        return max(bps, self.min_cost_bps)

    def cost_for_turnover(
        self, turnover_notional: float, adv_dollar: float | None
    ) -> float:
        """Dollar cost of trading ``turnover_notional`` dollars (ONE side) of a
        name. Each rebalance's per-name |Δweight|·book_notional IS one side, so
        the caller applies this per (rebalance, name); a full buy-then-sell
        cycle is naturally two applications (the +Δw at entry and the −Δw at
        exit)."""
        notional = abs(float(turnover_notional))
        if notional <= 0:
            return 0.0
        return notional * self.per_side_bps(notional, adv_dollar) / 1e4
