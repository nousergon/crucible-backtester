"""model_version_net_alpha.py — per-model-version NET alpha under a FIXED policy (L4488b).

Champion/challenger Phase 2 completion (model-rotation scaffolding arc, L4488).
The leaderboard's rank-IC / hit-rate (L4469 Phase 2/3) answer "does the version
RANK well?" — but the decision metric is **realized net-of-cost alpha**, and it
must be **execution-isolated**: score every registered version's predictions
through ONE fixed, frozen execution policy so the comparison measures the MODEL,
not model×execution.

Mechanism: reuse ``horizon_net_alpha._one_horizon`` — the same top-N
equal-weight, fixed-cadence long book + √-impact transaction-cost model the
executor approximates — but run it **per model version** at the canonical
horizon, ranking by each version's own predicted score (``p_up``, monotonic in
predicted alpha). Identical policy across versions ⇒ the only thing that varies
is the model.

Plus the multiple-testing control: picking the best of N challengers rewards
luck, so we report a **Deflated-Sharpe best-of-N** (López de Prado) — the
probability the leader's net Sharpe beats the expected maximum of N null trials.

OBSERVE-only: emitted to ``backtest/{date}/model_version_net_alpha.json``; gates
nothing. The operator reads it (with the leaderboard) before promoting a
challenger to champion (switch-GATE 2).
"""
from __future__ import annotations

import logging
import math
import sqlite3
from pathlib import Path
from statistics import NormalDist

import numpy as np

from analysis.horizon_net_alpha import _one_horizon
from analysis.transaction_cost import TransactionCostModel

logger = logging.getLogger(__name__)

_GAMMA = 0.5772156649015329  # Euler–Mascheroni
_FIXED_HORIZON_DEFAULT = 21
_DEFAULT_TOP_N = 20
_norm = NormalDist()


def load_version_predictions(research_db_path: str | None) -> dict:
    """``{version_label: {date_str: {ticker: p_up}}}`` from the outcome tables.

    Champion rows come from ``predictor_outcomes`` (grouped by ``model_version``;
    legacy NULL → ``champion-legacy``); challengers from
    ``predictor_outcomes_shadow``. The rank signal is ``p_up`` (monotonic in the
    predicted alpha). A missing table/column is skipped (the shadow table only
    exists once Phase-2 seeding ran). Realized returns come from prices in the
    book-sim, NOT from these rows — so only the predicted ranking is needed here.
    """
    out: dict = {}
    if not research_db_path or not Path(research_db_path).exists():
        return out
    conn = sqlite3.connect(research_db_path)
    try:
        for table in ("predictor_outcomes", "predictor_outcomes_shadow"):
            try:
                rows = conn.execute(
                    f"SELECT model_version, prediction_date, symbol, p_up "
                    f"FROM {table} WHERE p_up IS NOT NULL"
                ).fetchall()
            except sqlite3.OperationalError:
                continue  # table or model_version/p_up column absent
            for mv, d, sym, pu in rows:
                if table == "predictor_outcomes":
                    label = mv or "champion-legacy"
                else:
                    label = mv or "challenger-unknown"
                if pu is None or sym is None or d is None:
                    continue
                out.setdefault(label, {}).setdefault(str(d), {})[sym] = float(pu)
    finally:
        conn.close()
    return out


def _deflated_sharpe_best_of_n(sharpes: list, n_obs: int | None) -> float | None:
    """López de Prado Deflated-Sharpe for the BEST of N trials: P(leader's net
    Sharpe > the expected maximum of N null (zero-skill) trials). Controls for
    having shopped across N challengers. None when <2 finite Sharpes or too few
    observations to be meaningful. Normal-return approximation (skew=0, kurt=3)."""
    finite = [float(s) for s in sharpes if s is not None and np.isfinite(s)]
    n = len(finite)
    if n < 2 or n_obs is None or n_obs < 3:
        return None
    sr_star = max(finite)
    sr_std = float(np.std(finite, ddof=1))
    if sr_std <= 1e-12:
        return None
    # Expected maximum Sharpe of N null trials (LdP "expected max").
    e_max = (1.0 - _GAMMA) * _norm.inv_cdf(1.0 - 1.0 / n) + _GAMMA * _norm.inv_cdf(
        1.0 - 1.0 / (n * math.e)
    )
    sr_benchmark = sr_std * e_max
    denom = math.sqrt(1.0 + 0.5 * sr_star * sr_star)  # normal-return variance term
    dsr = _norm.cdf((sr_star - sr_benchmark) * math.sqrt(n_obs - 1) / denom)
    return round(float(dsr), 4)


def compute_model_version_net_alpha(
    version_preds: dict,
    price_matrix,
    spy_prices,
    *,
    cost_model: TransactionCostModel | None = None,
    horizon: int = _FIXED_HORIZON_DEFAULT,
    top_n: int = _DEFAULT_TOP_N,
    init_cash: float = 1_000_000.0,
    adv_dollar_by_ticker: dict | None = None,
) -> dict:
    """Per-version execution-isolated net-of-cost alpha (L4488b, OBSERVE).

    Runs the SAME fixed policy (``_one_horizon`` at ``horizon``, top-N
    equal-weight, cost model) on each version's predictions. Returns per-version
    net/gross alpha + turnover + cost + net Sharpe, the net-alpha leader, and the
    Deflated-Sharpe best-of-N selection control.
    """
    cost_model = cost_model or TransactionCostModel()
    per_version: dict = {}
    for version, preds_by_date in version_preds.items():
        if not preds_by_date:
            per_version[version] = {"status": "no_predictions"}
            continue
        try:
            per_version[version] = _one_horizon(
                int(horizon), preds_by_date, price_matrix, spy_prices,
                cost_model, top_n, init_cash, adv_dollar_by_ticker,
            )
        except Exception as exc:  # observe-only — never fail the backtest
            logger.warning(
                "model_version_net_alpha: version %s failed (non-fatal): %s",
                version, exc,
            )
            per_version[version] = {"status": "error", "error": str(exc)}

    finite = {
        v: r["net_alpha_ann"] for v, r in per_version.items()
        if isinstance(r.get("net_alpha_ann"), (int, float)) and np.isfinite(r["net_alpha_ann"])
    }
    leader = max(finite, key=finite.get) if finite else None
    sharpes = [r.get("net_sharpe") for r in per_version.values()]
    n_obs_leader = per_version.get(leader, {}).get("n_rebalances") if leader else None
    dsr = _deflated_sharpe_best_of_n(sharpes, n_obs_leader)

    if leader:
        logger.info(
            "model_version_net_alpha (OBSERVE, NOT gated): net-alpha by version=%s "
            "| leader=%s | DSR-best-of-%d=%s. NET-of-cost under a FIXED policy is "
            "the promotion judge (switch-GATE 2); rank-IC alone is not.",
            {v: round(float(a), 4) for v, a in finite.items()}, leader,
            len(finite), dsr,
        )

    return {
        "status": "ok",
        "fixed_policy": {"horizon": int(horizon), "top_n": top_n},
        "versions": per_version,
        "net_alpha_leader": leader,
        "deflated_sharpe_best_of_n": dsr,
        "cost_model": {
            "half_spread_bps": cost_model.half_spread_bps,
            "impact_coef_bps": cost_model.impact_coef_bps,
            "commission_bps": cost_model.commission_bps,
        },
        "note": "OBSERVE-only; gates nothing. Execution-isolated (one fixed top-N "
                "policy across all versions) net-of-cost alpha — measures the MODEL, "
                "not model×execution. DSR deflates best-of-N challenger selection.",
    }
