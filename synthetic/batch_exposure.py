"""Same-batch exposure caps for the scalar simulator (alpha-engine-config-I11504).

Root cause
----------
``executor.deciders.decide_entries`` checks every ENTER candidate with
``executor.risk_guard.check_order`` against ``current_positions`` — the book
as it stood BEFORE the batch. ``current_positions`` is never updated as
candidates are approved, so the max-equity and max-sector rules each compare
ONE candidate against the pre-batch book, and a batch whose orders are
individually under the cap is approved in full even when their SUM crosses
it. Measured in the 2026-09-23 weekly SF rehearsal
(``rehearsal-2026-09-23-2``): day-1 entries summed to ~71% of NAV, day-2 to
~25%, and the book sat at 96-101% against a 90% cap — the 62
``BLOCKED … Total equity exposure 100.2% would exceed max 90.0%`` lines were
the cap firing only once the breach was already in the book.

The deciders are the live executor's code, imported from
``executor_paths``; this repo does not change them. The simulator owns the
step that turns the decider's plan into fills (``_simulate_single_date``),
so that is where the batch is re-walked: in the decider's own approval
order, each order is re-checked against the pre-batch book PLUS the notional
of every order already kept from the same batch. An order that no longer
fits is dropped — exactly as ``check_order`` would have blocked it had it
seen the pending orders — and later, smaller orders may still fit.

The quantities mirror ``check_order`` so the two can be compared line for
line: the order's dollar size is ``position_pct × portfolio_nav_at_order``
(the ``dollar_size`` the decider passed to ``check_order``), existing
exposure is the sum of ``market_value`` over ``current_positions``, the
comparison is strict ``>``, and the config keys and defaults are the same
(``max_equity_pct`` 0.90, ``max_sector_pct`` 0.25; a candidate with no
sector is ``"Technology"``, as in ``decide_entries``).

The vectorized sweep carries the same accumulation as gate 14 of
``synthetic.vectorized_entries.compute_vectorized_entries``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEFAULT_MAX_EQUITY_PCT = 0.90
DEFAULT_MAX_SECTOR_PCT = 0.25
#: ``decide_entries`` reads ``sig.get("sector", "Technology")``.
DEFAULT_SECTOR = "Technology"


def _order_dollars(order: dict) -> float:
    """The dollar size ``check_order`` saw for this order."""
    pct = order.get("position_pct")
    nav = order.get("portfolio_nav_at_order")
    if pct is not None and nav is not None:
        return float(pct) * float(nav)
    return float(order.get("shares") or 0) * float(order.get("price_at_order") or 0.0)


def enforce_batch_exposure_caps(
    orders: list[dict],
    *,
    current_positions: dict[str, dict],
    portfolio_nav: float,
    config: dict,
) -> tuple[list[dict], list[dict]]:
    """Drop the ENTER orders a batch-aware exposure check would have blocked.

    Args:
        orders: ``EntryPlan.orders`` in the decider's approval order.
        current_positions: the enriched pre-batch book the decider checked
            against (``{ticker: {"market_value", "sector", ...}}``).
        portfolio_nav: the NAV the decider sized against.
        config: the merged executor config the decider was given.

    Returns:
        ``(kept, blocked)``. ``kept`` preserves order. Each ``blocked`` entry
        carries ``ticker``, ``rule`` (``max_sector`` | ``max_equity``),
        ``reason``, ``value``, ``threshold`` and ``pending_dollars`` — the
        same-batch notional the decider did not count. Non-ENTER orders pass
        through untouched.
    """
    if not orders or not portfolio_nav or portfolio_nav <= 0:
        return list(orders), []

    max_equity = float(config.get("max_equity_pct", DEFAULT_MAX_EQUITY_PCT))
    max_sector = float(config.get("max_sector_pct", DEFAULT_MAX_SECTOR_PCT))

    existing_equity = sum(
        float(p.get("market_value") or 0.0) for p in current_positions.values()
    )
    existing_sector: dict[str, float] = {}
    for p in current_positions.values():
        sec = p.get("sector")
        existing_sector[sec] = existing_sector.get(sec, 0.0) + float(
            p.get("market_value") or 0.0
        )

    pending_equity = 0.0
    pending_sector: dict[str, float] = {}
    kept: list[dict] = []
    blocked: list[dict] = []

    for order in orders:
        if order.get("action") != "ENTER":
            kept.append(order)
            continue
        ticker = order.get("ticker")
        sector = order.get("sector") or DEFAULT_SECTOR
        dollars = _order_dollars(order)

        # Rule order matches check_order: sector (6) before equity (7).
        sec_pending = pending_sector.get(sector, 0.0)
        sector_pct = (existing_sector.get(sector, 0.0) + sec_pending + dollars) / portfolio_nav
        if sector_pct > max_sector:
            blocked.append({
                "ticker": ticker,
                "rule": "max_sector",
                "reason": (
                    f"Sector exposure {sector_pct:.1%} would exceed max "
                    f"{max_sector:.1%} for {sector} counting same-batch "
                    f"pending ${sec_pending:,.0f} (config-I11504)"
                ),
                "value": sector_pct,
                "threshold": max_sector,
                "pending_dollars": sec_pending,
            })
            continue

        equity_pct = (existing_equity + pending_equity + dollars) / portfolio_nav
        if equity_pct > max_equity:
            blocked.append({
                "ticker": ticker,
                "rule": "max_equity",
                "reason": (
                    f"Total equity exposure {equity_pct:.1%} would exceed max "
                    f"{max_equity:.1%} counting same-batch pending "
                    f"${pending_equity:,.0f} (config-I11504)"
                ),
                "value": equity_pct,
                "threshold": max_equity,
                "pending_dollars": pending_equity,
            })
            continue

        pending_equity += dollars
        pending_sector[sector] = sec_pending + dollars
        kept.append(order)

    for b in blocked:
        logger.info("BLOCKED %s — %s", b["ticker"], b["reason"])
    return kept, blocked
