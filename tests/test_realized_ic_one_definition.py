"""alpha-engine-config-I8701 — the two realized-IC producers answer the same
question off the same table, so they must scope the same rows.

`analysis/production_health.py::compute_production_health` gained the horizon
and post-cutover filters on 2026-05-15, after the false-positive
`ic_degradation` retrain alert of 2026-05-11 (rolling = -0.1005 against a
training IC of 0.4634). `pipeline_common.py::push_predictor_rolling_metrics`
answers the same question off the same table and NEVER received them — the
`fix-not-propagated-to-analogous-sites` class.

Measured 2026-08-26, off one weekly run, the two published values disagreed in
SIGN:

    predictor/metrics/latest.json           ic_30d          = -0.0768  (N=547)
    predictor/metrics/production_health.json rolling_30d_ic = +0.0321  (N=25)

and -0.0768 was the number rendered on the daily Alpha Engine Brief.

These tests are written to FAIL against the pre-fix `push_predictor_rolling_metrics`
(champion-challenger-policy §7.4 — a guard that cannot fail is worse than no
guard, because it reads as coverage).
"""
from __future__ import annotations

import inspect
import sqlite3

import pytest

import pipeline_common
from pipeline_common import (
    CANONICAL_CUTOVER_DATE,
    CURRENT_HORIZON_FILTER_SQL,
    POST_CUTOVER_FILTER_SQL,
)

_COLS = (
    "symbol, prediction_date, predicted_direction, prediction_confidence, "
    "p_up, p_flat, p_down, score_modifier_applied, actual_5d_return, "
    "correct_5d, actual_log_alpha, horizon_days, correct"
)


def _make_db(tmp_path, rows: list[tuple]) -> str:
    db = tmp_path / "research.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE predictor_outcomes ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, prediction_date TEXT, "
        "predicted_direction TEXT, prediction_confidence REAL, p_up REAL, "
        "p_flat REAL, p_down REAL, score_modifier_applied REAL, "
        "actual_5d_return REAL, correct_5d INTEGER, actual_log_alpha REAL, "
        "horizon_days REAL, correct INTEGER)"
    )
    conn.executemany(
        f"INSERT INTO predictor_outcomes ({_COLS}) VALUES ({','.join('?' * 13)})",
        rows,
    )
    conn.commit()
    conn.close()
    return str(db)


def _row(i: int, pred_date: str, horizon, *, p_up: float, alpha: float) -> tuple:
    return (
        f"T{i}", pred_date, "UP", 0.6, p_up, 0.1, 1.0 - p_up - 0.1, 0.0,
        alpha * 100.0, 1, alpha, horizon, 1,
    )


# ── The structural guard: both call sites carry both filters ─────────────────


def test_rolling_metrics_query_applies_the_horizon_and_cutover_filters():
    """The pre-fix source of `push_predictor_rolling_metrics` contained neither
    filter. Asserting on the source is deliberate: the filters are SQL string
    constants spliced into a query, so there is no runtime object to inspect,
    and this is the exact drift that went unnoticed for three months."""
    src = inspect.getsource(pipeline_common.push_predictor_rolling_metrics)
    assert "CURRENT_HORIZON_FILTER_SQL" in src, (
        "push_predictor_rolling_metrics does not scope to the active horizon — "
        "it will blend pre-cutover 5d-arithmetic rows into the realized IC. "
        "See alpha-engine-config-I8701."
    )
    assert "POST_CUTOVER_FILTER_SQL" in src, (
        "push_predictor_rolling_metrics does not scope to post-cutover "
        "prediction dates — predictions MADE by the pre-cutover model but "
        "GRADED after the migration carry horizon_days=21 and would be pooled. "
        "See alpha-engine-config-I8701."
    )


def test_both_producers_scope_the_same_rows():
    """Whatever the filters are, the two functions must use the SAME ones."""
    rolling = inspect.getsource(pipeline_common.push_predictor_rolling_metrics)
    from analysis import production_health

    health = inspect.getsource(production_health.compute_production_health)
    for const in ("CURRENT_HORIZON_FILTER_SQL", "POST_CUTOVER_FILTER_SQL"):
        assert (const in rolling) == (const in health), (
            f"{const} is applied by exactly one of the two realized-IC "
            "producers. They read the same table to answer the same question; "
            "a filter on one and not the other is how they came to disagree in "
            "sign (alpha-engine-config-I8701)."
        )


# ── The behavioural guard: contaminating rows change the sign ────────────────


@pytest.fixture()
def _no_s3(monkeypatch):
    """Capture the merged metrics dict instead of writing it to S3."""
    captured: dict = {}

    class _FakeS3:
        class exceptions:  # noqa: N801 — mirrors botocore's attribute shape
            class NoSuchKey(Exception):
                pass

        def get_object(self, **_kw):
            raise _FakeS3.exceptions.NoSuchKey()

        def put_object(self, **kw):
            import json as _json

            captured.update(_json.loads(kw["Body"].decode("utf-8")))

    monkeypatch.setattr(
        pipeline_common.boto3, "client", lambda *_a, **_k: _FakeS3()
    )
    return captured


def test_off_horizon_rows_cannot_flip_the_published_sign(tmp_path, _no_s3, monkeypatch):
    """Twelve on-horizon rows with a positive relationship, plus twelve legacy
    NULL-horizon rows with the opposite relationship, all inside the lookback
    window. Pre-fix the pooled Pearson r is negative; post-fix the legacy rows
    are excluded and it is positive.

    This is the 2026-08-26 shape in miniature. The horizon filter is the half
    that admits contamination on a live pool today — the cutover filter's own
    rows sit outside the ~60-day lookback and are exercised structurally by
    `test_rolling_metrics_query_applies_the_horizon_and_cutover_filters`.
    """
    monkeypatch.setattr(pipeline_common, "_load_active_horizon_days", lambda: 21)

    import datetime as _dt

    recent = (_dt.date.today() - _dt.timedelta(days=5)).strftime("%Y-%m-%d")
    rows = []
    for i in range(12):
        # active horizon: high p_up goes with high realized alpha (positive IC)
        rows.append(_row(i, recent, 21.0, p_up=0.5 + i * 0.02, alpha=0.001 * i))
    for i in range(12):
        # legacy NULL-horizon 5d-arithmetic row: high p_up goes with LOW alpha
        rows.append(_row(100 + i, recent, None, p_up=0.5 + i * 0.02, alpha=-0.02 * i))

    db = _make_db(tmp_path, rows)
    pipeline_common.push_predictor_rolling_metrics(
        {"signals_bucket": "test-bucket"}, db
    )

    assert _no_s3, "producer wrote nothing"
    ic = _no_s3.get("ic_30d")
    assert ic is not None, f"IC not computed: {_no_s3.get('ic_null_reason')}"
    assert ic > 0, (
        f"published IC {ic} is negative — off-horizon legacy rows with an "
        "inverted sign reached the pool. That is exactly "
        "alpha-engine-config-I8701."
    )
    assert _no_s3["rolling_n"] == 12, (
        f"rolling_n={_no_s3['rolling_n']}, expected the 12 on-horizon rows only"
    )


# ── The honesty guard: the artifact describes its own estimator ──────────────


def _recent_date() -> str:
    """A prediction date inside the 60-day lookback at ANY run date. These two
    tests hardcoded 2026-08-20, which aged out of the window on 2026-10-20 and
    turned the first red and the second vacuous (rows filtered, IC None)."""
    import datetime as _dt
    return (_dt.date.today() - _dt.timedelta(days=5)).strftime("%Y-%m-%d")


def test_realized_ic_block_reports_effective_n_not_just_rows(tmp_path, _no_s3, monkeypatch):
    """`rolling_n` counts (ticker, date) rows over overlapping 21-day label
    windows. The brief rendered the IC beside that count as if it were the N
    behind the estimate. The artifact must carry the non-overlapping count too.
    """
    monkeypatch.setattr(pipeline_common, "_load_active_horizon_days", lambda: 21)

    rows = [
        _row(i, _recent_date(), 21.0, p_up=0.5 + i * 0.01, alpha=0.001 * i)
        for i in range(20)
    ]
    db = _make_db(tmp_path, rows)
    pipeline_common.push_predictor_rolling_metrics(
        {"signals_bucket": "test-bucket"}, db
    )

    block = _no_s3.get("realized_ic")
    assert block, "realized_ic descriptor block missing from the artifact"
    assert block["n_rows"] == 20
    # 20 rows, ONE prediction date, 21-day horizon → one label window, not 20.
    assert block["n_prediction_dates"] == 1
    assert block["n_effective"] == 1, (
        f"n_effective={block['n_effective']} — 20 rows sharing one prediction "
        "date span one non-overlapping label window, not twenty."
    )
    assert block["window_days"] != 30, (
        "window_days must state the ACTUAL lookback; `ic_30d` is a misnomer "
        "retained only for back-compat."
    )
    assert block["signal_leg"] == "p_up_minus_p_down"
    assert block["horizon_filtered"] is True
    assert block["post_cutover_filtered"] is True


def test_ic_ir_is_never_a_bare_nan(tmp_path, _no_s3, monkeypatch):
    """A constant-signal chunk makes pearsonr return NaN, which serialised into
    the artifact as the bare literal `NaN` — invalid JSON. Measured live in
    latest.json on 2026-08-26.
    """
    monkeypatch.setattr(pipeline_common, "_load_active_horizon_days", lambda: 21)

    # Every row identical → zero variance in both legs → NaN correlations.
    rows = [
        _row(i, _recent_date(), 21.0, p_up=0.6, alpha=0.01)
        for i in range(20)
    ]
    db = _make_db(tmp_path, rows)
    pipeline_common.push_predictor_rolling_metrics(
        {"signals_bucket": "test-bucket"}, db
    )

    ir = _no_s3.get("ic_ir_30d")
    assert ir is None or (isinstance(ir, float) and ir == ir), (
        f"ic_ir_30d={ir!r} is a non-finite float; it must be reported as null "
        "so the artifact stays valid JSON (alpha-engine-config-I8701)."
    )


def test_cutover_constant_unchanged():
    assert CANONICAL_CUTOVER_DATE == "2026-05-09"
    assert POST_CUTOVER_FILTER_SQL.endswith(f"'{CANONICAL_CUTOVER_DATE}'")
    assert CURRENT_HORIZON_FILTER_SQL.startswith("horizon_days = ")
