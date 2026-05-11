"""Tests for the stance taxonomy arc PR 4 (backtester wiring).

Three load-bearing pins:

  1. Stance threshold params are in ``executor_optimizer.SAFE_PARAMS`` —
     otherwise the auto-tune path skips them when reading/writing
     config/executor_params.json.
  2. Stance threshold params have ``FACTORY_DEFAULTS`` entries — drift
     monitoring + cold-start fallback rely on this.
  3. Stance threshold params are in ``param_sweep.EXTENDED_GRID`` —
     otherwise the sweep never proposes alternatives, leaving the
     cold-start defaults frozen.
  4. ``signal_quality.compute_accuracy`` emits ``by_stance`` when the
     ``stance`` column is present in score_performance, and skips it
     when absent (graceful degrade during the data-layer transition).
"""

from __future__ import annotations

import pandas as pd
import pytest


class TestExecutorOptimizerStanceParams:
    """Stance threshold params must be wired into the auto-tune path."""

    def test_value_drawdown_min_in_safe_params(self):
        from optimizer.executor_optimizer import SAFE_PARAMS
        assert "value_stance_drawdown_min" in SAFE_PARAMS, (
            "value_stance_drawdown_min must be in SAFE_PARAMS for the "
            "auto-tune path to read/write it from "
            "config/executor_params.json"
        )

    def test_quality_threshold_in_safe_params(self):
        from optimizer.executor_optimizer import SAFE_PARAMS
        assert "quality_stance_momentum_threshold" in SAFE_PARAMS

    def test_value_drawdown_min_factory_default(self):
        """Factory default matches the executor-side ``_plan_entries``
        fallback (-0.05). Drift between the two means the executor and
        backtester disagree on default behavior — surfaces in
        regression-monitor drift alerts."""
        from optimizer.executor_optimizer import FACTORY_DEFAULTS
        assert FACTORY_DEFAULTS["value_stance_drawdown_min"] == pytest.approx(-0.05)

    def test_quality_threshold_factory_default(self):
        from optimizer.executor_optimizer import FACTORY_DEFAULTS
        assert FACTORY_DEFAULTS["quality_stance_momentum_threshold"] == pytest.approx(-15.0)


class TestParamSweepStanceRanges:
    """Sweep search space must include stance thresholds so the
    optimizer can move them in either direction once data exists."""

    def test_value_drawdown_in_extended_grid(self):
        from analysis.param_sweep import EXTENDED_GRID
        assert "value_stance_drawdown_min" in EXTENDED_GRID
        candidates = EXTENDED_GRID["value_stance_drawdown_min"]
        # Cold-start default must be in the range so the sweep at least
        # re-validates it; bracket ranges (tighter + looser candidates)
        # let the optimizer move in either direction.
        assert -0.05 in candidates
        assert any(v < -0.05 for v in candidates), (
            "Sweep range must include a TIGHTER (more negative) value "
            "candidate so optimizer can move toward stricter drawdown "
            "requirement"
        )
        assert any(v > -0.05 for v in candidates), (
            "Sweep range must include a LOOSER (less negative) value "
            "candidate so optimizer can move toward easier qualification"
        )

    def test_quality_threshold_in_extended_grid(self):
        from analysis.param_sweep import EXTENDED_GRID
        assert "quality_stance_momentum_threshold" in EXTENDED_GRID
        candidates = EXTENDED_GRID["quality_stance_momentum_threshold"]
        assert -15.0 in candidates
        assert any(v < -15.0 for v in candidates)
        assert any(v > -15.0 for v in candidates)


class TestSignalQualityStanceAttribution:
    """``compute_accuracy`` emits ``by_stance`` cohort split when the
    stance column is present in score_performance — graceful degrade
    when it isn't."""

    def _make_df(self, n: int, with_stance: bool) -> pd.DataFrame:
        """Build a synthetic score_performance frame with N rows.

        Schema mirrors what ``compute_accuracy``'s slice-metric helper
        actually reads: ``beat_spy_{5,10,30}d`` for accuracy +
        ``return_{5,10,30}d`` / ``spy_{5,10,30}d_return`` for the
        avg-alpha computation. Extra fields (conviction, stance) drive
        the per-field attribution branches.
        """
        data = {
            "score": [70 + i % 20 for i in range(n)],
            "beat_spy_5d": [(i % 2) for i in range(n)],
            "beat_spy_10d": [(i % 2) for i in range(n)],
            "beat_spy_30d": [(i % 2) for i in range(n)],
            "return_5d": [0.01 if i % 2 else -0.01 for i in range(n)],
            "return_10d": [0.02 if i % 2 else -0.02 for i in range(n)],
            "return_30d": [0.03 if i % 2 else -0.03 for i in range(n)],
            "spy_5d_return": [0.005] * n,
            "spy_10d_return": [0.010] * n,
            "spy_30d_return": [0.015] * n,
            "conviction": ["rising" if i % 3 == 0 else "stable" for i in range(n)],
        }
        if with_stance:
            stances = ["momentum", "value", "quality", "catalyst"]
            data["stance"] = [stances[i % 4] for i in range(n)]
        return pd.DataFrame(data)

    def test_by_stance_emitted_when_stance_column_present(self):
        from analysis.signal_quality import compute_accuracy

        df = self._make_df(n=100, with_stance=True)
        result = compute_accuracy(df, min_samples=10)
        assert result["status"] == "ok"
        assert "by_stance" in result, (
            "by_stance cohort split must appear when score_performance "
            "carries the stance column"
        )
        stance_rows = result["by_stance"]
        # 4 stances each with ~25 rows in a 100-row frame
        stance_labels = {r["stance"] for r in stance_rows}
        assert stance_labels == {"momentum", "value", "quality", "catalyst"}

    def test_by_stance_skipped_when_stance_column_absent(self):
        """During the data-layer transition (predictor emits stance
        from 2026-05-11; research.db score_performance hasn't been
        migrated to join in stance yet), the column is absent.
        ``compute_accuracy`` must skip per-stance attribution without
        crashing — graceful degrade."""
        from analysis.signal_quality import compute_accuracy

        df = self._make_df(n=100, with_stance=False)
        result = compute_accuracy(df, min_samples=10)
        assert result["status"] == "ok"
        assert "by_stance" not in result, (
            "by_stance must NOT appear when stance column is absent — "
            "this is the data-layer transition's graceful-degrade case"
        )
        # Other attributions still flow through
        assert "by_conviction" in result
        assert "overall" in result

    def test_by_stance_skipped_when_all_stance_values_null(self):
        """Stance column exists but every row is NULL (e.g., partial
        data migration in progress). Skip per-stance attribution rather
        than emitting a misleading single-cohort table."""
        from analysis.signal_quality import compute_accuracy

        df = self._make_df(n=100, with_stance=True)
        df["stance"] = pd.NA
        result = compute_accuracy(df, min_samples=10)
        assert result["status"] == "ok"
        assert "by_stance" not in result


class TestReporterStanceSection:
    """The reporter must render a ``### By stance`` markdown table when
    ``by_stance`` is present, and skip it when absent."""

    def test_renders_by_stance_section_when_present(self):
        from reporter import _section_signal_quality

        sq = {
            "status": "ok",
            "rows_5d_populated": 50, "rows_10d_populated": 50, "rows_30d_populated": 50,
            "overall": {"accuracy_5d": 0.55, "accuracy_10d": 0.58, "accuracy_30d": 0.62,
                        "avg_alpha_5d": 0.005, "avg_alpha_10d": 0.012,
                        "avg_alpha_30d": 0.018, "n": 50},
            "by_stance": [
                {"stance": "momentum", "accuracy_5d": 0.60, "accuracy_10d": 0.65,
                 "accuracy_30d": 0.70, "avg_alpha_10d": 0.020, "n_10d": 15},
                {"stance": "value", "accuracy_5d": 0.50, "accuracy_10d": 0.52,
                 "accuracy_30d": 0.55, "avg_alpha_10d": 0.005, "n_10d": 15},
                {"stance": "quality", "accuracy_5d": 0.55, "accuracy_10d": 0.58,
                 "accuracy_30d": 0.60, "avg_alpha_10d": 0.010, "n_10d": 12},
                {"stance": "catalyst", "accuracy_5d": 0.62, "accuracy_10d": 0.68,
                 "accuracy_30d": 0.72, "avg_alpha_10d": 0.025, "n_10d": 8},
            ],
        }
        lines = _section_signal_quality(sq)
        rendered = "\n".join(lines)
        assert "### By stance" in rendered
        assert "momentum" in rendered
        assert "value" in rendered
        assert "quality" in rendered
        assert "catalyst" in rendered

    def test_skips_by_stance_section_when_absent(self):
        from reporter import _section_signal_quality

        sq = {
            "status": "ok",
            "rows_5d_populated": 50, "rows_10d_populated": 50, "rows_30d_populated": 50,
            "overall": {"accuracy_5d": 0.55, "accuracy_10d": 0.58, "accuracy_30d": 0.62,
                        "avg_alpha_5d": 0.005, "avg_alpha_10d": 0.012,
                        "avg_alpha_30d": 0.018, "n": 50},
            # No by_stance key
        }
        lines = _section_signal_quality(sq)
        rendered = "\n".join(lines)
        assert "### By stance" not in rendered, (
            "Stance section must not render when by_stance is absent"
        )
