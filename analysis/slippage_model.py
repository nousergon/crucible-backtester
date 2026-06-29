"""Realistic slippage model (config#919).

The backtester currently charges a single flat ``slippage_bps`` constant per
side (``vectorbt_bridge`` / ``reference_sim``). That is wrong in opposite
directions across the book: it overcharges large, liquid, low-vol names and
undercharges small, thin, volatile ones. This module replaces the constant with
a parametric model

    actual_slippage_bps = f(market_cap, dollar_volume, order_size, volatility,
                            trigger_type)

fit on realized ``slippage_vs_signal`` from live ENTER trades.

Design choices
--------------
* **Closed-form, numpy-only.** No sklearn/statsmodels dependency — the model is
  a log-linear OLS the simulator can evaluate per-order with a dict of coeffs.
* **Log features.** Market-impact literature is multiplicative: impact scales
  ~with (order_size / ADV) and volatility, and shrinks with liquidity. We
  regress slippage_bps on log1p-transformed size/liquidity features + a
  per-trigger intercept (one-hot), so coefficients are interpretable and the
  fit is convex with a unique solution (ridge-regularized for stability on
  small live samples).
* **Pure compute.** ``fit_slippage_model`` takes a list of observation dicts
  (already joined to features upstream) and returns coefficients + diagnostics;
  ``predict_slippage_bps`` evaluates one order. No I/O, so both are trivially
  testable and reusable from a Lambda, a notebook, or the sim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Continuous features, in the order the design matrix builds them. Each is
# log1p-transformed (all are non-negative magnitudes); the model learns one
# coefficient per feature plus a per-trigger intercept.
_CONTINUOUS_FEATURES = (
    "market_cap",        # USD market capitalization (liquidity proxy, ↓ slippage)
    "dollar_volume",     # avg daily $ volume / ADV (liquidity proxy, ↓ slippage)
    "order_notional",    # this order's $ size (↑ slippage)
    "volatility",        # realized/implied vol, fractional e.g. 0.02 (↑ slippage)
)

# Fallback flat slippage (bps) when the model is unavailable / under-sampled.
DEFAULT_FLAT_SLIPPAGE_BPS = 10.0

_MIN_OBSERVATIONS = 30  # below this, refuse to fit — fall back to flat
_RIDGE_LAMBDA = 1.0     # L2 on continuous coefs (not the trigger intercepts)


@dataclass(frozen=True)
class SlippageModel:
    """A fitted log-linear slippage model.

    ``continuous_coefs`` maps each feature in ``_CONTINUOUS_FEATURES`` to its
    coefficient (applied to ``log1p(feature)``). ``trigger_intercepts`` maps a
    normalized trigger label to its intercept (bps); ``default_intercept`` is
    used for unseen triggers. ``predict`` returns slippage in bps, floored at 0.
    """

    continuous_coefs: dict[str, float]
    trigger_intercepts: dict[str, float]
    default_intercept: float
    n_obs: int
    r_squared: float
    rmse_bps: float
    diagnostics: dict = field(default_factory=dict)

    def predict(
        self,
        *,
        market_cap: float,
        dollar_volume: float,
        order_notional: float,
        volatility: float,
        trigger_type: str | None,
    ) -> float:
        """Predict per-order slippage in basis points (>= 0)."""
        feats = {
            "market_cap": market_cap,
            "dollar_volume": dollar_volume,
            "order_notional": order_notional,
            "volatility": volatility,
        }
        intercept = self.trigger_intercepts.get(
            normalize_trigger(trigger_type), self.default_intercept
        )
        val = intercept
        for name in _CONTINUOUS_FEATURES:
            val += self.continuous_coefs.get(name, 0.0) * _log1p_nonneg(feats[name])
        return max(0.0, float(val))

    def to_dict(self) -> dict:
        return {
            "continuous_coefs": dict(self.continuous_coefs),
            "trigger_intercepts": dict(self.trigger_intercepts),
            "default_intercept": self.default_intercept,
            "n_obs": self.n_obs,
            "r_squared": self.r_squared,
            "rmse_bps": self.rmse_bps,
            "diagnostics": dict(self.diagnostics),
        }


def normalize_trigger(trigger_type: str | None) -> str:
    """Normalize a free-text trigger label to a stable bucket key."""
    if not trigger_type:
        return "unspecified"
    return str(trigger_type).strip().lower()


def _log1p_nonneg(x) -> float:
    """log1p of a non-negative magnitude; clamps NaN/neg to 0."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v) or v < 0:
        return 0.0
    return math.log1p(v)


def predict_slippage_bps(
    model: SlippageModel | None,
    *,
    market_cap: float,
    dollar_volume: float,
    order_notional: float,
    volatility: float,
    trigger_type: str | None,
    fallback_bps: float = DEFAULT_FLAT_SLIPPAGE_BPS,
) -> float:
    """Predict slippage for one order, falling back to a flat bps when unfit."""
    if model is None:
        return fallback_bps
    return model.predict(
        market_cap=market_cap,
        dollar_volume=dollar_volume,
        order_notional=order_notional,
        volatility=volatility,
        trigger_type=trigger_type,
    )


def fit_slippage_model(
    observations: list[dict],
    *,
    min_observations: int = _MIN_OBSERVATIONS,
    ridge_lambda: float = _RIDGE_LAMBDA,
) -> SlippageModel | None:
    """Fit the log-linear slippage model from realized live trades (config#919).

    Args:
        observations: each dict must carry the target ``slippage_bps`` (realized
            ``slippage_vs_signal`` expressed in basis points) plus the five
            features ``market_cap, dollar_volume, order_notional, volatility,
            trigger_type``. Rows missing the target or with a non-finite target
            are dropped.
        min_observations: refuse to fit (return ``None``) below this many usable
            rows — the caller falls back to the flat constant.
        ridge_lambda: L2 penalty on the continuous coefficients (the per-trigger
            intercepts are left unpenalized).

    Returns:
        A ``SlippageModel`` or ``None`` when under-sampled.
    """
    rows = [o for o in observations if _usable(o)]
    if len(rows) < min_observations:
        return None

    triggers = sorted({normalize_trigger(o.get("trigger_type")) for o in rows})
    trig_index = {t: i for i, t in enumerate(triggers)}
    n_trig = len(triggers)
    n_cont = len(_CONTINUOUS_FEATURES)
    n_cols = n_trig + n_cont  # one-hot trigger intercepts + continuous slopes

    X = np.zeros((len(rows), n_cols), dtype=float)
    y = np.zeros(len(rows), dtype=float)
    for r, o in enumerate(rows):
        X[r, trig_index[normalize_trigger(o.get("trigger_type"))]] = 1.0
        for j, name in enumerate(_CONTINUOUS_FEATURES):
            X[r, n_trig + j] = _log1p_nonneg(o.get(name))
        y[r] = float(o["slippage_bps"])

    # Ridge on continuous coefs only: penalty matrix zeros out the intercept
    # block so per-trigger means are unbiased.
    penalty = np.zeros((n_cols, n_cols), dtype=float)
    for j in range(n_trig, n_cols):
        penalty[j, j] = ridge_lambda
    xtx = X.T @ X + penalty
    try:
        beta = np.linalg.solve(xtx, X.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

    trigger_intercepts = {t: float(beta[trig_index[t]]) for t in triggers}
    continuous_coefs = {
        name: float(beta[n_trig + j]) for j, name in enumerate(_CONTINUOUS_FEATURES)
    }
    default_intercept = float(np.mean(list(trigger_intercepts.values())))

    y_hat = X @ beta
    resid = y - y_hat
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = math.sqrt(ss_res / len(rows))

    return SlippageModel(
        continuous_coefs=continuous_coefs,
        trigger_intercepts=trigger_intercepts,
        default_intercept=default_intercept,
        n_obs=len(rows),
        r_squared=round(r_squared, 4),
        rmse_bps=round(rmse, 4),
        diagnostics={
            "n_triggers": n_trig,
            "triggers": triggers,
            "mean_slippage_bps": round(float(y.mean()), 4),
            "ridge_lambda": ridge_lambda,
        },
    )


def build_observations_from_trades(
    entries: list[dict],
    feature_lookup,
) -> list[dict]:
    """Join live ENTER trades to per-(ticker, date) features for fitting.

    Args:
        entries: ENTER trade rows. Each must carry ``ticker``, ``date``,
            ``trigger_type`` and a realized slippage in basis points. Slippage
            is read from ``slippage_bps`` if present, else converted from
            ``slippage_vs_signal`` (a fractional return, ``0.001`` → ``10`` bps).
        feature_lookup: callable ``(ticker, date) -> dict | None`` returning
            ``{market_cap, dollar_volume, order_notional, volatility}`` for that
            order (the executor/feature-store join the caller owns). Rows whose
            lookup returns ``None`` are skipped.

    Returns observation dicts ready for :func:`fit_slippage_model`. Kept thin and
    dependency-free so the live wiring (ArcticDB / feature-store reads) lives in
    the caller and this stays unit-testable with an injected lookup.
    """
    obs: list[dict] = []
    for e in entries:
        slip_bps = e.get("slippage_bps")
        if slip_bps is None and e.get("slippage_vs_signal") is not None:
            try:
                slip_bps = float(e["slippage_vs_signal"]) * 10_000.0
            except (TypeError, ValueError):
                slip_bps = None
        if slip_bps is None:
            continue
        feats = feature_lookup(e.get("ticker"), e.get("date"))
        if not feats:
            continue
        obs.append({
            "slippage_bps": slip_bps,
            "trigger_type": e.get("trigger_type"),
            "market_cap": feats.get("market_cap", 0.0),
            "dollar_volume": feats.get("dollar_volume", 0.0),
            "order_notional": feats.get("order_notional", 0.0),
            "volatility": feats.get("volatility", 0.0),
        })
    return obs


def _usable(o: dict) -> bool:
    """A row is usable iff it has a finite target slippage_bps."""
    if "slippage_bps" not in o:
        return False
    try:
        v = float(o["slippage_bps"])
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)
