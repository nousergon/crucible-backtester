"""Tests for the research-free meta-ensemble backfill producer (config#1405,
build items 1+2) — ``analysis/scanner_predictor_research_free_backfill.py``.

Mirrors ``tests/test_scanner_then_predictor.py``'s approach: a synthetic
sqlite fixture (no ArcticDB / S3 / real model artifact needed) exercising the
producer's logic — idempotency (skip-if-cached), the (ticker, eval_date)
pending-universe query against ``scanner_evaluations``, table creation /
schema, and the research-free feature-zeroing contract
(``_assemble_research_free_features``). The ArcticDB-backed feature
computation (``run_backfill`` end-to-end) is exercised in the PR description
against the LIVE production store instead — not reproducible hermetically
here, per the issue's own testing section ("the meta-ensemble backfill
validates on the spot run").
"""

from __future__ import annotations

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.scanner_predictor_research_free_backfill import (
    RESEARCH_META_FEATURES,
    TABLE_NAME,
    _assemble_research_free_features,
    _ensure_table,
    _existing_keys,
    _pending_universe,
)


def _scanner_db(tmp_path, *, prefill_predictions=None):
    conn = sqlite3.connect(str(tmp_path / "r.db"))
    conn.execute(
        "CREATE TABLE scanner_evaluations (ticker TEXT, eval_date TEXT, quant_filter_pass INTEGER)"
    )
    dates = ["2026-04-12", "2026-04-20"]
    for d in dates:
        for i in range(5):
            conn.execute(
                "INSERT INTO scanner_evaluations VALUES (?,?,?)",
                (f"T{i}", d, 1 if i < 3 else 0),  # only T0/T1/T2 pass per date
            )
    if prefill_predictions:
        conn.execute(
            f"CREATE TABLE {TABLE_NAME} (ticker TEXT, prediction_date TEXT, "
            "predicted_alpha REAL, n_research_features_missing INTEGER)"
        )
        for ticker, d, alpha in prefill_predictions:
            conn.execute(
                f"INSERT INTO {TABLE_NAME} VALUES (?,?,?,?)", (ticker, d, alpha, 4)
            )
    conn.commit()
    return conn


# ── _ensure_table / schema ───────────────────────────────────────────────────


def test_ensure_table_creates_expected_schema(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "e.db"))
    _ensure_table(conn)
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({TABLE_NAME})")}
    assert cols == {"ticker", "prediction_date", "predicted_alpha", "n_research_features_missing"}, cols


def test_ensure_table_is_idempotent_call(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "e.db"))
    _ensure_table(conn)
    _ensure_table(conn)  # must not raise on a second call
    n = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    assert n == 0


def test_table_matches_consumer_contract_from_end_to_end_test():
    """The schema this producer writes must exactly match the frozen contract
    ``tests/test_scanner_then_predictor.py`` builds and
    ``analysis/end_to_end.py::_scanner_then_predictor_topN`` reads — same
    table name, same column set, same join-key semantics
    (``prediction_date`` == the scanner's ``eval_date``)."""
    import sqlite3 as _sq

    conn = _sq.connect(":memory:")
    _ensure_table(conn)
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({TABLE_NAME})")]
    assert TABLE_NAME == "predictor_outcomes_research_free"
    assert cols == ["ticker", "prediction_date", "predicted_alpha", "n_research_features_missing"]


# ── _pending_universe / idempotency (skip-if-cached) ────────────────────────


def test_pending_universe_returns_all_passing_rows_when_table_empty(tmp_path):
    conn = _scanner_db(tmp_path)
    _ensure_table(conn)
    pending = _pending_universe(conn)
    # 3 passing tickers x 2 dates = 6 rows, none cached yet
    assert len(pending) == 6, pending
    assert set(pending["ticker"]) == {"T0", "T1", "T2"}
    assert set(pending["eval_date"]) == {"2026-04-12", "2026-04-20"}


def test_pending_universe_excludes_already_cached_rows(tmp_path):
    conn = _scanner_db(
        tmp_path,
        prefill_predictions=[("T0", "2026-04-12", 0.01), ("T1", "2026-04-12", -0.02)],
    )
    pending = _pending_universe(conn)
    # 6 total - 2 cached = 4 remaining
    assert len(pending) == 4, pending
    pairs = set(zip(pending["ticker"], pending["eval_date"]))
    assert ("T0", "2026-04-12") not in pairs
    assert ("T1", "2026-04-12") not in pairs
    assert ("T2", "2026-04-12") in pairs
    assert ("T0", "2026-04-20") in pairs


def test_pending_universe_empty_when_fully_cached(tmp_path):
    prefill = [
        (f"T{i}", d, 0.0)
        for d in ("2026-04-12", "2026-04-20")
        for i in range(3)
    ]
    conn = _scanner_db(tmp_path, prefill_predictions=prefill)
    pending = _pending_universe(conn)
    assert pending.empty, pending


def test_pending_universe_raises_without_quant_filter_pass_column(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "bad.db"))
    conn.execute("CREATE TABLE scanner_evaluations (ticker TEXT, eval_date TEXT)")
    conn.commit()
    try:
        _pending_universe(conn)
        assert False, "expected sqlite3.OperationalError"
    except sqlite3.OperationalError as e:
        assert "quant_filter_pass" in str(e)


def test_existing_keys_reads_ticker_prediction_date_pairs(tmp_path):
    conn = _scanner_db(
        tmp_path, prefill_predictions=[("T0", "2026-04-12", 0.01)]
    )
    keys = _existing_keys(conn)
    assert keys == {("T0", "2026-04-12")}


def test_existing_keys_empty_when_table_absent(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "e.db"))
    assert _existing_keys(conn) == set()


# ── _assemble_research_free_features — the research-free contract ───────────


def test_research_features_always_zeroed_even_if_available():
    """The 4 research meta-features are zeroed unconditionally — the whole
    point of the arm is "what if research never ran," not "zero only when
    missing." A ticker with a value sitting in momentum/macro dicts for a
    research-feature NAME must still be zeroed (defense: no legacy scorer
    dict accidentally supplies a 'research_*' key that leaks through)."""
    feat_names = [
        "research_calibrator_prob", "momentum_score", "expected_move",
        "research_composite_score", "research_conviction", "sector_macro_modifier",
        "macro_spy_20d_return", "regime_intensity_z",
    ]
    feats = _assemble_research_free_features(
        "AAPL", feat_names,
        momentum_scores={"AAPL": 0.4},
        resid_scores={},
        expected_moves={"AAPL": 0.02},
        macro_row={"macro_spy_20d_return": 0.03, "regime_intensity_z": 0.5},
    )
    for f in RESEARCH_META_FEATURES:
        if f in feat_names:
            assert feats[f] == 0.0, (f, feats)
    assert feats["momentum_score"] == 0.4
    assert feats["expected_move"] == 0.02
    assert feats["macro_spy_20d_return"] == 0.03
    assert feats["regime_intensity_z"] == 0.5


def test_residual_momentum_variant_schema():
    """The live deployed model may use residual_momentum_score instead of
    momentum_score/expected_move (observed in production — see module
    docstring); the assembler must read from ``resid_scores`` for that name
    without requiring momentum_score/expected_move to also be present."""
    feat_names = ["research_calibrator_prob", "residual_momentum_score", "macro_vix_level"]
    feats = _assemble_research_free_features(
        "MSFT", feat_names,
        momentum_scores={},
        resid_scores={"MSFT": -0.15},
        expected_moves={},
        macro_row={"macro_vix_level": 1.2},
    )
    assert feats["research_calibrator_prob"] == 0.0
    assert feats["residual_momentum_score"] == -0.15
    assert feats["macro_vix_level"] == 1.2


def test_unknown_feature_name_degrades_to_zero_not_crash():
    """A feature name this producer has no computer registered for (future
    model-schema drift) degrades to 0.0 rather than raising — matches
    MetaModel.predict_single's own .get(f, 0.0) missing-key contract."""
    feat_names = ["momentum_score", "some_future_feature_v9"]
    feats = _assemble_research_free_features(
        "GOOG", feat_names,
        momentum_scores={"GOOG": 0.1},
        resid_scores={},
        expected_moves={},
        macro_row={},
    )
    assert feats["some_future_feature_v9"] == 0.0
    assert feats["momentum_score"] == 0.1


def test_missing_ticker_in_component_dicts_degrades_to_zero():
    """A ticker absent from a component dict (e.g. the vol scorer failed for
    just this ticker) degrades that single feature to 0.0 rather than
    KeyError — the per-ticker graceful-degrade contract."""
    feat_names = ["momentum_score", "expected_move"]
    feats = _assemble_research_free_features(
        "ZZZZ", feat_names,
        momentum_scores={},  # ZZZZ absent
        resid_scores={},
        expected_moves={},  # ZZZZ absent
        macro_row={},
    )
    assert feats == {"momentum_score": 0.0, "expected_move": 0.0}


def test_n_research_features_missing_is_a_count_of_four_by_construction():
    """The RESEARCH_META_FEATURES set is exactly the 4 the issue names —
    a guard against silent drift in this module's constant."""
    assert RESEARCH_META_FEATURES == {
        "research_calibrator_prob",
        "research_composite_score",
        "research_conviction",
        "sector_macro_modifier",
    }
    assert len(RESEARCH_META_FEATURES) == 4


# ── Idempotent insert semantics (INSERT OR REPLACE on the PK) ───────────────


def test_insert_or_replace_on_primary_key_is_idempotent(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "e.db"))
    _ensure_table(conn)
    conn.execute(
        f"INSERT OR REPLACE INTO {TABLE_NAME} VALUES (?,?,?,?)",
        ("T0", "2026-04-12", 0.01, 4),
    )
    conn.execute(
        f"INSERT OR REPLACE INTO {TABLE_NAME} VALUES (?,?,?,?)",
        ("T0", "2026-04-12", 0.05, 4),  # re-run with a different value
    )
    conn.commit()
    rows = conn.execute(f"SELECT * FROM {TABLE_NAME}").fetchall()
    assert len(rows) == 1, rows  # PK collision replaced, not duplicated
    assert rows[0][2] == 0.05


# ── S3 artifact seam: materialize_from_s3 / _export_artifact ────────────────
#
# The producer (PredictorBacktest box) and consumer (Evaluator box) each pull
# their OWN throwaway research.db copy from S3 and nothing pushes it back —
# the parquet at ARTIFACT_KEY is the only wire between them. These tests
# exercise both directions of that seam hermetically via a fake s3 client
# (mirrors tests/test_reporter.py's injected s3_client idiom).


class _FakeS3:
    """download_file/upload_file backed by a local dict of key -> filepath."""

    def __init__(self, tmp_path):
        self._tmp = tmp_path
        self._objects: dict[str, str] = {}
        self.upload_calls: list[tuple[str, str, str]] = []

    def put_local(self, key: str, local_path: str) -> None:
        self._objects[key] = local_path

    def download_file(self, bucket, key, dest):
        import shutil

        from botocore.exceptions import ClientError

        if key not in self._objects:
            raise ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
            )
        shutil.copyfile(self._objects[key], dest)

    def upload_file(self, src, bucket, key):
        import shutil

        stored = str(self._tmp / f"stored_{key.replace('/', '_')}")
        shutil.copyfile(src, stored)
        self._objects[key] = stored
        self.upload_calls.append((src, bucket, key))


def test_materialize_from_s3_returns_zero_when_artifact_absent(tmp_path):
    from analysis.scanner_predictor_research_free_backfill import materialize_from_s3

    conn = sqlite3.connect(str(tmp_path / "m.db"))
    n = materialize_from_s3(conn, "any-bucket", s3_client=_FakeS3(tmp_path))
    assert n == 0
    # honest empty state: table exists (or is creatable) with zero rows
    _ensure_table(conn)
    assert conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0] == 0


def test_export_then_materialize_roundtrip(tmp_path):
    """Producer-side export -> consumer-side materialize must reproduce the
    exact table contents on a second, independent connection (the two-box
    seam in miniature)."""
    import pandas as pd

    from analysis.scanner_predictor_research_free_backfill import (
        ARTIFACT_KEY,
        _export_artifact,
        materialize_from_s3,
    )

    s3 = _FakeS3(tmp_path)
    producer = sqlite3.connect(str(tmp_path / "producer.db"))
    _ensure_table(producer)
    rows = [("T0", "2026-04-12", 0.013, 4), ("T1", "2026-04-20", -0.021, 4)]
    producer.executemany(f"INSERT INTO {TABLE_NAME} VALUES (?,?,?,?)", rows)
    producer.commit()

    key = _export_artifact(producer, "any-bucket", s3_client=s3)
    assert key == ARTIFACT_KEY
    assert [c[2] for c in s3.upload_calls] == [ARTIFACT_KEY]

    consumer = sqlite3.connect(str(tmp_path / "consumer.db"))
    n = materialize_from_s3(consumer, "any-bucket", s3_client=s3)
    assert n == 2
    got = sorted(consumer.execute(f"SELECT * FROM {TABLE_NAME}").fetchall())
    assert got == sorted(rows)

    # re-materializing is idempotent (INSERT OR REPLACE on the PK)
    n2 = materialize_from_s3(consumer, "any-bucket", s3_client=s3)
    assert n2 == 2
    assert consumer.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0] == 2


def test_materialize_seeds_pending_universe_idempotency(tmp_path):
    """run_backfill's idempotency depends on seeding the fresh local pull from
    the artifact — after materializing, _pending_universe must exclude the
    already-computed keys."""
    from analysis.scanner_predictor_research_free_backfill import (
        _export_artifact,
        materialize_from_s3,
    )

    s3 = _FakeS3(tmp_path)
    prior = sqlite3.connect(str(tmp_path / "prior.db"))
    _ensure_table(prior)
    prior.executemany(
        f"INSERT INTO {TABLE_NAME} VALUES (?,?,?,?)",
        [("T0", "2026-04-12", 0.01, 4), ("T1", "2026-04-12", 0.02, 4)],
    )
    prior.commit()
    _export_artifact(prior, "any-bucket", s3_client=s3)

    fresh = _scanner_db(tmp_path)  # a brand-new pull: no backfill table at all
    materialize_from_s3(fresh, "any-bucket", s3_client=s3)
    pending = _pending_universe(fresh)
    pairs = set(zip(pending["ticker"], pending["eval_date"]))
    assert ("T0", "2026-04-12") not in pairs
    assert ("T1", "2026-04-12") not in pairs
    assert len(pending) == 4  # 6 passing - 2 seeded


def test_materialize_raises_on_non_404_download_error(tmp_path):
    """A corrupt/unreachable artifact must raise, never silently demote the
    counterfactual back to 'skipped' (fail-loud doctrine)."""
    from botocore.exceptions import ClientError

    from analysis.scanner_predictor_research_free_backfill import materialize_from_s3

    class _Denied(_FakeS3):
        def download_file(self, bucket, key, dest):
            raise ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "no"}}, "GetObject"
            )

    conn = sqlite3.connect(str(tmp_path / "d.db"))
    try:
        materialize_from_s3(conn, "any-bucket", s3_client=_Denied(tmp_path))
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "AccessDenied" in str(e)
