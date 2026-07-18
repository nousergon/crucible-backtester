"""
score_calibrator.py — isotonic calibration of composite research scores → P(beat SPY).

Why this module exists
----------------------
``composite_scoring``'s 0-100 output (``crucible-research/scoring/composite.py``)
is a *ranking* heuristic (weighted quant+qual+factor blend). It was never fit to
historical hit-rates, yet ``compute_portfolio_calibration`` fed ``score / 100``
into the ECE-based ``calibration_diagnostics`` gate as if it were ``P(beats SPY)``.
Grading an uncalibrated ranking score against a probability red-line produced the
0.24-vs-0.15 RED (config#2304) — the gate was measuring the wrong thing.

Operator ruling (config#2304, 2026-07-12, **Option A**): give ``composite_scoring``
an isotonic calibration layer fit against the ``score_performance`` corpus so its
output can *legitimately* be read as a probability, mirroring the predictor's
existing ``ResearchCalibrator`` pattern (``crucible-predictor/model/research_calibrator.py``).
This module is the isotonic upgrade of that pattern: monotone, non-parametric
(``sklearn.isotonic.IsotonicRegression``), with a Platt/logistic fallback.

Circularity guard (the config#2304 Delta)
------------------------------------------
The training corpus (``score_performance``) is the SAME corpus the ECE gate
scores. Fitting on the full corpus and then measuring ECE on it is circular — an
isotonic fit is calibrated *by construction* on its own training data, so the ECE
would collapse to ~0 and the gate would become a rubber-stamp that can never
detect real miscalibration. ``out_of_fold_calibrated_probabilities`` therefore
K-fold-splits the corpus, fits on the train folds, and predicts each held-out
fold — every calibrated probability is out-of-sample, so the resulting ECE
honestly measures whether the score's monotone signal *generalizes*.

Pure-compute. ``save``/``load`` persist a full-corpus fit for downstream consumers
(the funnel / predictor) that want the canonical score→probability mapping — the
gate itself never needs a persisted artifact (it re-derives OOF each grading run).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_PRIOR = 0.50          # neutral prior when unfit / no data
_MIN_FIT_SAMPLES = 20         # below this the fit is meaningless — stay at prior
_MIN_PER_FOLD = 10            # each OOF fold needs enough held-out points to be informative
_DEFAULT_N_FOLDS = 5
_OOF_SEED = 0                 # fixed → deterministic OOF probabilities (grading must be reproducible)

_VALID_METHODS = ("isotonic", "platt")


class ScoreProbabilityCalibrator:
    """Map a composite research score (0-100) to a calibrated ``P(beat SPY)``.

    Isotonic (default) learns a monotone non-decreasing step function from the
    empirical (score → beat_spy) corpus — the SOTA non-parametric calibrator
    when a large corpus and a monotone score→outcome relationship are expected.
    Platt (``method="platt"``) fits a 1-D logistic instead, preferable on thin
    corpora where isotonic's step function overfits.
    """

    def __init__(self, method: str = "isotonic"):
        if method not in _VALID_METHODS:
            raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")
        self.method = method
        self._model = None
        self._fitted = False
        self._n_samples = 0
        self._overall_hit_rate: float | None = None

    def fit(self, scores: np.ndarray, beat_spy: np.ndarray) -> "ScoreProbabilityCalibrator":
        """Fit the calibrator from historical (score, beat_spy) pairs.

        ``scores`` are composite scores (0-100); ``beat_spy`` is a binary array
        (1 if the pick beat SPY over the primary horizon, else 0). NaN in either
        drops that pair.
        """
        scores = np.asarray(scores, dtype=np.float64).ravel()
        beat_spy = np.asarray(beat_spy, dtype=np.float64).ravel()
        if len(scores) != len(beat_spy):
            raise ValueError(f"Length mismatch: {len(scores)} scores vs {len(beat_spy)} labels")

        valid = np.isfinite(scores) & np.isfinite(beat_spy)
        scores = scores[valid]
        beat_spy = beat_spy[valid]
        self._n_samples = int(scores.size)

        if self._n_samples < _MIN_FIT_SAMPLES:
            log.warning(
                "ScoreProbabilityCalibrator: only %d valid samples (< %d) — staying at prior",
                self._n_samples, _MIN_FIT_SAMPLES,
            )
            self._fitted = False
            return self

        self._overall_hit_rate = float(beat_spy.mean())

        # Degenerate corpora: a constant score column or a single-class outcome
        # can't yield a monotone mapping — collapse to the base rate. This is the
        # honest calibrated answer (P = base rate everywhere), not a failure.
        if scores.min() == scores.max() or beat_spy.min() == beat_spy.max():
            self._model = None  # predict() falls back to overall_hit_rate
            self._fitted = True
            log.info(
                "ScoreProbabilityCalibrator: degenerate corpus (constant score or outcome) "
                "— calibrated P = base rate %.4f", self._overall_hit_rate,
            )
            return self

        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            model.fit(scores, beat_spy)
        else:  # platt
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression()
            model.fit(scores.reshape(-1, 1), beat_spy.astype(int))

        self._model = model
        self._fitted = True
        log.info(
            "ScoreProbabilityCalibrator fitted: method=%s n=%d overall_hit_rate=%.2f%%",
            self.method, self._n_samples, self._overall_hit_rate * 100,
        )
        return self

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def predict(self, score: float) -> float:
        """Return calibrated ``P(beat SPY)`` for a single composite score."""
        return float(self.predict_batch(np.asarray([score], dtype=np.float64))[0])

    def predict_batch(self, scores: np.ndarray) -> np.ndarray:
        """Vectorized calibrated probabilities for an array of composite scores."""
        scores = np.asarray(scores, dtype=np.float64).ravel()
        if not self._fitted:
            return np.full(scores.shape, DEFAULT_PRIOR, dtype=np.float64)
        if self._model is None:
            # Degenerate-corpus fit: calibrated P is the base rate everywhere.
            fill = self._overall_hit_rate if self._overall_hit_rate is not None else DEFAULT_PRIOR
            return np.full(scores.shape, float(fill), dtype=np.float64)
        if self.method == "isotonic":
            preds = self._model.predict(scores)
        else:
            preds = self._model.predict_proba(scores.reshape(-1, 1))[:, 1]
        return np.clip(np.asarray(preds, dtype=np.float64), 0.0, 1.0)

    def metrics(self) -> dict:
        return {
            "type": f"score_probability_calibrator_{self.method}",
            "fitted": self._fitted,
            "n_samples": self._n_samples,
            "overall_hit_rate": (
                round(self._overall_hit_rate, 4) if self._overall_hit_rate is not None else None
            ),
        }

    # ── persistence (for downstream consumers; the gate re-derives OOF) ──────
    def save(self, path: str | Path) -> None:
        """Persist the fitted mapping as a plain-JSON lookup grid (no pickle).

        Stores the calibrated probability sampled on an integer 0-100 grid so
        the artifact is inspectable and framework-independent, mirroring the
        predictor's ``isotonic_calibrator.meta.json`` convention.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        grid = np.arange(0, 101, dtype=np.float64)
        probs = self.predict_batch(grid) if self._fitted else np.full(grid.shape, DEFAULT_PRIOR)
        data = {
            "type": f"score_probability_calibrator_{self.method}",
            "method": self.method,
            "fitted": self._fitted,
            "n_samples": self._n_samples,
            "overall_hit_rate": self._overall_hit_rate,
            "grid": {str(int(s)): round(float(p), 6) for s, p in zip(grid, probs)},
        }
        path.write_text(json.dumps(data, indent=2))
        log.info("ScoreProbabilityCalibrator saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ScoreProbabilityCalibrator":
        """Load a calibrator from a saved lookup grid (interpolating predict)."""
        path = Path(path)
        data = json.loads(path.read_text())
        rc = cls(method=data.get("method", "isotonic"))
        rc._n_samples = int(data.get("n_samples", 0))
        rc._overall_hit_rate = data.get("overall_hit_rate")
        rc._fitted = bool(data.get("fitted"))
        grid = data.get("grid", {})
        if grid:
            xs = np.array(sorted(int(k) for k in grid), dtype=np.float64)
            ys = np.array([grid[str(int(x))] for x in xs], dtype=np.float64)

            class _GridModel:
                def predict(self, scores):
                    return np.interp(np.asarray(scores, dtype=np.float64), xs, ys)

            rc._model = _GridModel()
            rc.method = "isotonic"  # grid interpolation reproduces the monotone map
        log.info("ScoreProbabilityCalibrator loaded from %s (n=%d)", path, rc._n_samples)
        return rc


def out_of_fold_calibrated_probabilities(
    scores: np.ndarray,
    beat_spy: np.ndarray,
    n_folds: int = _DEFAULT_N_FOLDS,
    method: str = "isotonic",
    seed: int = _OOF_SEED,
) -> np.ndarray:
    """Honest, non-circular calibrated probabilities via K-fold out-of-fold fit.

    Every returned probability is predicted by a calibrator that did NOT see that
    sample during fitting, so downstream ECE measures out-of-sample calibration
    (the config#2304 circularity guard). Returns an array the same length as the
    input; positions that could not be scored (empty train fold) are NaN, which
    ``compute_calibration`` drops.

    ``n_folds`` is clamped down when the corpus is too small to give each fold
    ``_MIN_PER_FOLD`` held-out points; below ``_MIN_FIT_SAMPLES`` total, every
    output is NaN (caller degrades to insufficient_data).
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    beat_spy = np.asarray(beat_spy, dtype=np.float64).ravel()
    if len(scores) != len(beat_spy):
        raise ValueError(f"Length mismatch: {len(scores)} scores vs {len(beat_spy)} labels")

    n = scores.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n < _MIN_FIT_SAMPLES:
        return out

    # Clamp folds so each held-out fold keeps ~>= _MIN_PER_FOLD points, and the
    # train side keeps >= _MIN_FIT_SAMPLES. At least 2 folds (OOF needs a split).
    k = min(n_folds, max(2, n // _MIN_PER_FOLD))
    k = max(2, min(k, n - _MIN_FIT_SAMPLES + 1))
    if k < 2:
        return out

    from sklearn.model_selection import KFold

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(scores):
        if train_idx.size < _MIN_FIT_SAMPLES:
            continue
        cal = ScoreProbabilityCalibrator(method=method)
        cal.fit(scores[train_idx], beat_spy[train_idx])
        if not cal.is_fitted:
            continue
        out[test_idx] = cal.predict_batch(scores[test_idx])
    return out
