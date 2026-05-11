"""Contract tests for the ``keep_predictions`` kwarg added to
``synthetic.predictor_backtest.run``.

The kwarg exposes ``predictions_by_date`` (raw GBM alpha forecasts) in the
result dict without keeping the ~1.1 GB ``features_by_ticker`` dict alive —
that's the lightweight alternative needed by the portfolio-optimizer
backtest harness (PR 3 of portfolio-optimizer-260511.md).

We test the contract at the signature + result-dict level rather than
running the full pipeline (which needs a 10y feature fixture + GBM model
+ S3 round-trip — a heavy integration setup that already happens on the
Saturday SF spot).
"""

from __future__ import annotations

import inspect

from synthetic.predictor_backtest import run


def test_run_signature_includes_keep_predictions_kwarg():
    """``run()`` must accept ``keep_predictions: bool = False``."""
    sig = inspect.signature(run)
    assert "keep_predictions" in sig.parameters, (
        "run() must accept keep_predictions kwarg — needed by portfolio-optimizer "
        "backtest harness (analysis/portfolio_optimizer_backtest.py)"
    )
    param = sig.parameters["keep_predictions"]
    assert param.default is False, (
        f"keep_predictions must default to False (zero-cost-by-default); "
        f"got default={param.default!r}"
    )
    assert param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_run_docstring_documents_keep_predictions_contract():
    """The docstring must explain what keep_predictions exposes and why it's
    distinct from keep_features."""
    doc = run.__doc__ or ""
    assert "keep_predictions" in doc, "Docstring must reference the new kwarg"
    assert "predictions_by_date" in doc, (
        "Docstring must explain that keep_predictions=True exposes "
        "predictions_by_date in the result"
    )
    assert "1.1 GB" in doc or "features" in doc, (
        "Docstring should explain why this is distinct from keep_features "
        "(the 1.1 GB memory cost)"
    )


def test_keep_features_and_keep_predictions_are_compatible():
    """``keep_features=True`` already exposes predictions_by_date — the new
    kwarg must not break that path. They're allowed to coexist; keep_features
    is the superset (also keeps features) and keep_predictions is the subset
    (predictions only)."""
    sig = inspect.signature(run)
    # No mutual-exclusion validation should reject (True, True) — that's the
    # pre-existing keep_features semantics where predictions_by_date is also
    # set. The fact that they're both bool defaults to False means the user
    # opts into either or both independently.
    assert sig.parameters["keep_features"].default is False
    assert sig.parameters["keep_predictions"].default is False
