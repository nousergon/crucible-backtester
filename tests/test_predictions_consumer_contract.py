"""Cross-repo predictions.json consumer-contract test.

Pins the contract between the **predictor** (producer of
``predictor/predictions/{date}.json``) and the **backtester** measurement-
coverage consumer (``analysis.measurement_coverage._read_prediction_tickers``).

Producer schema (alpha-engine-predictor/inference/stages/write_output.py): a
metadata envelope whose ``predictions`` field is a **LIST of per-ticker
records**, each ``{"ticker": ..., "predicted_alpha": ..., "predicted_direction":
..., ...}``::

    {
      "date": "...", "model_version": "...", "model_hit_rate_30d": ...,
      "n_predictions": N, "n_high_confidence": ...,
      "output_distribution_gate": {...}, "level_neutralization": {...},
      "predictions": [ {"ticker": "AAA", "predicted_alpha": 0.11, ...}, ... ]
    }

Background — the false-0%-coverage bug (config#909, 2026-06-27). The consumer
originally assumed a flat ``{ticker: alpha}`` map (or a ``{"predictions":
{ticker: alpha}}`` envelope) and fell through to ``set(payload.keys())``, so it
read the envelope's METADATA keys (``date``/``model_version``/...) as if they
were tickers. The signal∩prediction intersection was therefore empty and the
producer emitted a fabricated ``predicted_of_signal = 0%`` — the exact dashboard
false-alarm the measurement-coverage panel exists to avoid. Every prior unit
test used the flat-map form, so they encoded the wrong schema and passed; only a
run against live S3 surfaced it.

This test fails LOUDLY if a future change:
  - Reverts the consumer to flat-map-only parsing (re-introducing the
    metadata-keys-as-tickers bug against the real producer envelope).
  - The producer's per-record ``ticker`` key is dropped on the consumer path.

The M0 slot-boundary discipline pins predictions.json to ONE shared *versioned
JSON Schema* (``nousergon_lib.contracts`` ``predictions`` slot, v1) validated in
BOTH predictor and backtester CI. The producer half lives in crucible-predictor's
``test_predictions_schema_conformance.py`` (validates the real assembled
write_predictions envelope). THIS file is the consumer half (config#1321): the
envelope the backtester's measurement-coverage reader consumes is validated
against the SAME shared lib schema — not a hand-built local schema — so producer
and consumer can never silently drift onto two different shapes.

See: ``~/Development/CLAUDE.md`` M0 contract discipline (predictions.json is a
named slot-boundary artifact); sibling ``test_scanner_consumer_contract.py``.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.measurement_coverage import _read_prediction_tickers  # noqa: E402


# ── Fixture: a faithful copy of the production predictions.json envelope ──────


def _production_predictions_envelope(tickers: list[str]) -> dict:
    """Mirror the alpha-engine-predictor write_output.py envelope verbatim in
    structure: top-level metadata + a ``predictions`` LIST of per-ticker dicts.
    The metadata keys are the trap the original consumer mistook for tickers.

    Each per-ticker record carries the schema-REQUIRED core fields the producer
    always emits (``ticker``/``predicted_direction``/``prediction_confidence``/
    ``predicted_alpha``/``combined_rank``/``gbm_veto``/``momentum_veto``) plus a
    common optional field, so this fixture conforms to the shared Slot M contract
    (``nousergon_lib.contracts`` ``predictions`` v1) — the same schema the
    producer-side conformance test validates its assembled envelope against. The
    fixture and the schema are therefore pinned together: drift on either side
    turns ``test_consumer_fixture_conforms_to_shared_slot_contract`` red."""
    return {
        "date": "2026-06-26",
        "model_version": "meta-v3",
        "model_hit_rate_30d": 0.55,
        "n_predictions": len(tickers),
        "n_high_confidence": 1,
        "output_distribution_gate": {"passed": True, "blocking": False},
        "level_neutralization": {"applied": False},
        "predictions": [
            {
                "ticker": t,
                "predicted_direction": "UP",
                "prediction_confidence": 0.41,
                "predicted_alpha": 0.10 + i * 0.01,
                "combined_rank": i + 1,
                "gbm_veto": False,
                "momentum_veto": False,
                "p_up": 0.70,
            }
            for i, t in enumerate(tickers)
        ],
    }


class _StubBody:
    def __init__(self, raw: bytes):
        self._raw = raw

    def read(self) -> bytes:
        return self._raw


class _StubS3:
    """Returns the registered JSON payload for the predictions key."""

    def __init__(self, payload: dict):
        import json

        self._raw = json.dumps(payload).encode()

    def get_object(self, *, Bucket, Key):  # noqa: N803 (boto3 kwarg names)
        return {"Body": _StubBody(self._raw)}


# ── The contract ─────────────────────────────────────────────────────────────


def test_consumer_extracts_tickers_from_production_list_envelope():
    """The consumer MUST return the per-record tickers from the production
    list envelope — NOT the envelope's metadata keys (the false-0% bug)."""
    tickers = ["AAA", "BBB", "CCC"]
    s3 = _StubS3(_production_predictions_envelope(tickers))
    notes: list[str] = []

    got = _read_prediction_tickers(s3, bucket="b", date="2026-06-26", notes=notes)

    assert got == set(tickers)
    # The metadata keys must NOT leak in as tickers.
    for meta_key in (
        "date", "model_version", "model_hit_rate_30d", "n_predictions",
        "n_high_confidence", "output_distribution_gate", "level_neutralization",
        "predictions",
    ):
        assert meta_key not in got, f"metadata key {meta_key!r} leaked as a ticker"
    assert notes == []


def test_consumer_handles_empty_production_envelope():
    """An empty predictions list is a measured-empty set (denominator zero),
    not an error and not the metadata keys."""
    s3 = _StubS3(_production_predictions_envelope([]))
    notes: list[str] = []
    got = _read_prediction_tickers(s3, bucket="b", date="2026-06-26", notes=notes)
    assert got == set()


# ── Shared-schema conformance (config#1321, M0 consumer half) ─────────────────
#
# The two tests above pin the consumer's PARSING against the production envelope
# SHAPE. These pin that same envelope against the ONE shared versioned schema in
# ``nousergon_lib.contracts`` — the identical schema the producer-side
# conformance test validates against — so the fixture this consumer is tested
# with cannot drift onto a shape the producer never emits (or vice versa).


def test_consumer_fixture_conforms_to_shared_slot_contract():
    """The exact envelope the backtester consumes MUST validate against the
    SHARED Slot M contract (``predictions`` v1), not a hand-built local schema.
    This is the consumer half of the cross-repo schema pin (config#1321):
    producer CI validates its assembled output and this asserts the consumer's
    contract fixture is the same shape, against the same source-of-truth schema."""
    contracts = pytest.importorskip(
        "nousergon_lib.contracts",
        reason="needs nousergon-lib[contracts] (jsonschema) for the shared slot schema",
    )

    envelope = _production_predictions_envelope(["AAA", "BBB", "CCC"])

    # Fail-loud: raises ContractViolation listing every conformance error on drift.
    contracts.validate("predictions", envelope)
    assert contracts.conformance_errors("predictions", envelope) == []

    # Empty-run envelope conforms too (denominator-zero is a valid measured state).
    contracts.validate("predictions", _production_predictions_envelope([]))

    # We are pinned to the versioned contract, v1.
    assert contracts.SCHEMA_VERSIONS["predictions"] == 1


def test_shared_contract_gate_fires_on_dropped_load_bearing_field():
    """Red-fixture demo: dropping a load-bearing per-record field the producer
    always emits MUST surface a non-empty conformance error against the shared
    schema — proving the gate actually fires rather than rubber-stamping."""
    contracts = pytest.importorskip(
        "nousergon_lib.contracts",
        reason="needs nousergon-lib[contracts] (jsonschema) for the shared slot schema",
    )

    envelope = _production_predictions_envelope(["AAA"])
    del envelope["predictions"][0]["gbm_veto"]

    errors = contracts.conformance_errors("predictions", envelope)
    assert errors and "gbm_veto" in " ".join(errors)
    with pytest.raises(contracts.ContractViolation):
        contracts.validate("predictions", envelope)
