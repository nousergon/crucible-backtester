"""consecutive_blocked_weeks counts distinct ISO weeks, not audit writes
(alpha-engine-config-I10064 item 3, Brian's ruling #18).

The live defect: the executor_params streak read 16 "weeks" over 16 audits
that fell in 12 ISO weeks, because re-runs on 08-03/04/07, 08-13/14 and
09-23/25 each added 1. These tests pin (1) the forward rule, (2) the
legacy-prior fallback, and (3) the one-time rebase from the dated history.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from optimizer.apply_audit import (
    LOOPS,
    build_audit,
    emit_apply_audit,
    iso_week,
    rebase_prior_from_history,
    replay_history,
)


def _weight_blocked():
    return {
        "status": "ok",
        "updated": False,
        "current_weights": {"quant": 0.5, "qual": 0.5},
        "suggested_weights": {"quant": 0.6, "qual": 0.4},
        "apply_result": {
            "applied": False,
            "reason": "Change too small (0.010 < 0.02)",
            "blocked_by": ["min_meaningful_change"],
        },
    }


def _weight_promoted():
    r = _weight_blocked()
    r["updated"] = True
    r["apply_result"] = {"applied": True, "reason": "applied"}
    return r


def _results(weight_result):
    return {"weight_result": weight_result}


def _rec(outcome, n, week="__absent__"):
    rec = {"outcome": outcome, "consecutive_blocked_weeks": n}
    if week != "__absent__":
        rec["last_blocked_iso_week"] = week
    return rec


def _prior(as_of, rec):
    return {"schema_version": 1, "as_of": as_of, "loops": {"scoring_weights": rec}}


def _sw(audit):
    return audit["loops"]["scoring_weights"]


class TestIsoWeek:

    def test_labels(self):
        assert iso_week("2026-10-09") == "2026-W41"
        assert iso_week("2026-07-24") == "2026-W30"
        # ISO year differs from calendar year at the boundary.
        assert iso_week("2027-01-01") == "2026-W53"
        assert iso_week("2025-12-29") == "2026-W01"

    def test_malformed_raises(self):
        with pytest.raises(ValueError):
            iso_week("2026-13-01")


class TestForwardRule:

    def test_absent_prior_blocked_counts_its_week(self):
        rec = _sw(build_audit("2026-10-09", _results(_weight_blocked())))
        assert rec["consecutive_blocked_weeks"] == 1
        assert rec["last_blocked_iso_week"] == "2026-W41"

    def test_new_week_increments(self):
        prior = _prior("2026-10-02", _rec("blocked", 5, "2026-W40"))
        rec = _sw(build_audit("2026-10-09", _results(_weight_blocked()), prior=prior))
        assert rec["consecutive_blocked_weeks"] == 6
        assert rec["last_blocked_iso_week"] == "2026-W41"

    def test_rerun_on_another_date_in_same_week_does_not_increment(self):
        """The 09-23 / 09-25 shape: two audits, one ISO week (W39)."""
        prior = _prior("2026-09-23", _rec("blocked", 5, "2026-W39"))
        rec = _sw(build_audit("2026-09-25", _results(_weight_blocked()), prior=prior))
        assert rec["consecutive_blocked_weeks"] == 5
        assert rec["last_blocked_iso_week"] == "2026-W39"

    def test_same_as_of_rerun_carries_both_fields(self):
        prior = _prior("2026-10-09", _rec("blocked", 5, "2026-W41"))
        rec = _sw(build_audit("2026-10-09", _results(_weight_blocked()), prior=prior))
        assert rec["consecutive_blocked_weeks"] == 5
        assert rec["last_blocked_iso_week"] == "2026-W41"

    def test_promoted_resets_both(self):
        prior = _prior("2026-10-02", _rec("blocked", 5, "2026-W40"))
        rec = _sw(build_audit("2026-10-09", _results(_weight_promoted()), prior=prior))
        assert rec["consecutive_blocked_weeks"] == 0
        assert rec["last_blocked_iso_week"] is None

    def test_blocked_after_promotion_in_same_week_restarts_at_one(self):
        prior = _prior("2026-10-06", _rec("promoted", 0, None))
        rec = _sw(build_audit("2026-10-09", _results(_weight_blocked()), prior=prior))
        assert rec["consecutive_blocked_weeks"] == 1

    def test_error_carries_week_so_a_later_same_week_block_is_not_double_counted(self):
        """blocked (W41) -> error (W41) -> blocked (W41) is still one week."""
        a1 = build_audit("2026-10-05", _results(_weight_blocked()),
                         prior=_prior("2026-09-28", _rec("blocked", 3, "2026-W40")))
        assert _sw(a1)["consecutive_blocked_weeks"] == 4
        a2 = build_audit("2026-10-07", _results({"status": "error", "error": "boom"}), prior=a1)
        assert _sw(a2)["outcome"] == "error"
        assert _sw(a2)["consecutive_blocked_weeks"] == 4
        assert _sw(a2)["last_blocked_iso_week"] == "2026-W41"
        a3 = build_audit("2026-10-09", _results(_weight_blocked()), prior=a2)
        assert _sw(a3)["consecutive_blocked_weeks"] == 4

    def test_artifact_still_conforms_to_frozen_schema(self):
        jsonschema = pytest.importorskip("jsonschema")
        from nousergon_lib import contracts

        prior = _prior("2026-10-02", _rec("blocked", 5, "2026-W40"))
        audit = build_audit("2026-10-09", _results(_weight_blocked()), prior=prior,
                            counter_rebase={"source": "history_replay", "n_audits": 2,
                                            "from": {}, "to": {}})
        jsonschema.validate(instance=audit, schema=contracts.load_schema("apply_audit"))
        assert set(audit["loops"]) == set(LOOPS)


class TestLegacyPrior:
    """A prior written before the rule has no last_blocked_iso_week."""

    def test_legacy_blocked_prior_same_week_is_not_recounted(self):
        prior = _prior("2026-09-23", _rec("blocked", 14))
        rec = _sw(build_audit("2026-09-25", _results(_weight_blocked()), prior=prior))
        assert rec["consecutive_blocked_weeks"] == 14
        assert rec["last_blocked_iso_week"] == "2026-W39"

    def test_legacy_blocked_prior_earlier_week_increments(self):
        prior = _prior("2026-10-09", _rec("blocked", 16))
        rec = _sw(build_audit("2026-10-16", _results(_weight_blocked()), prior=prior))
        assert rec["consecutive_blocked_weeks"] == 17
        assert rec["last_blocked_iso_week"] == "2026-W42"

    def test_legacy_error_prior_has_no_known_week(self):
        prior = _prior("2026-10-09", _rec("error", 6))
        rec = _sw(build_audit("2026-10-09", _results({"status": "error", "error": "x"}), prior=prior))
        assert rec["last_blocked_iso_week"] is None


# ── one-time rebase from the dated history ───────────────────────────────────

# The live executor_params shape from the 10-12 classification: 16 blocked
# audits in 12 ISO weeks (W30..W41), counted 16 by the per-write rule.
_LIVE_DATES = [
    "2026-07-24", "2026-07-31", "2026-08-03", "2026-08-04", "2026-08-07",
    "2026-08-13", "2026-08-14", "2026-08-21", "2026-08-28", "2026-09-04",
    "2026-09-11", "2026-09-18", "2026-09-23", "2026-09-25", "2026-10-02",
    "2026-10-09",
]


def _legacy_audit(as_of, n, outcome="blocked"):
    return {
        "schema_version": 1,
        "as_of": as_of,
        "loops": {"executor_params": {"outcome": outcome, "blocked_by": ["alpha_floor"],
                                      "consecutive_blocked_weeks": n, "detail": ""}},
    }


def test_live_history_is_twelve_weeks_not_sixteen():
    assert len({iso_week(d) for d in _LIVE_DATES}) == 12
    audits = [_legacy_audit(d, i + 1) for i, d in enumerate(_LIVE_DATES)]
    replayed = replay_history(audits)
    rec = replayed["loops"]["executor_params"]
    assert replayed["as_of"] == "2026-10-09"
    assert rec["consecutive_blocked_weeks"] == 12
    assert rec["last_blocked_iso_week"] == "2026-W41"


def test_replay_resets_on_promotion_and_carries_through_error():
    audits = [
        _legacy_audit("2026-08-07", 1),
        _legacy_audit("2026-08-14", 0, outcome="promoted"),
        _legacy_audit("2026-08-21", 1),
        _legacy_audit("2026-08-28", 1, outcome="error"),
        _legacy_audit("2026-08-29", 2),  # same week as 08-28's error, new week vs 08-21
        _legacy_audit("2026-09-04", 3),
    ]
    rec = replay_history(audits)["loops"]["executor_params"]
    assert rec["consecutive_blocked_weeks"] == 3


class _FakeS3:
    """Minimal list/get fake over an in-memory prefix."""

    def __init__(self, objects: dict[str, dict], page_size: int = 1000):
        self.objects = objects
        self.page_size = page_size
        self.puts: list[dict] = []

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = keys[start:start + self.page_size]
        resp = {"Contents": [{"Key": k} for k in page]}
        if start + self.page_size < len(keys):
            resp["IsTruncated"] = True
            resp["NextContinuationToken"] = str(start + self.page_size)
        return resp

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        data = json.dumps(self.objects[Key]).encode()
        body = MagicMock()
        body.read.return_value = data
        return {"Body": body}

    def put_object(self, **kw):
        self.puts.append(kw)


def _live_bucket(page_size=1000):
    objects = {
        f"config/apply_audit/{d}.json": _legacy_audit(d, i + 1)
        for i, d in enumerate(_LIVE_DATES)
    }
    objects["config/apply_audit/latest.json"] = objects["config/apply_audit/2026-10-09.json"]
    return _FakeS3(objects, page_size=page_size)


def _executor_blocked():
    return {
        "status": "ok",
        "recommended_params": {"min_score": 65},
        "baseline_params": {"min_score": 70},
        "apply_result": {"applied": False, "reason": "alpha below floor",
                         "blocked_by": ["alpha_floor"]},
    }


class TestEmitRebase:

    def test_first_run_after_rule_rebases_from_history(self):
        s3 = _live_bucket(page_size=5)  # forces pagination
        audit = emit_apply_audit(
            bucket="b", run_date="2026-10-16",
            opt_results={"executor_rec": _executor_blocked()},
            upload=True, s3_client=s3,
        )
        rec = audit["loops"]["executor_params"]
        assert rec["outcome"] == "blocked"
        # 12 historical ISO weeks + W42, not 16 + 1.
        assert rec["consecutive_blocked_weeks"] == 13
        assert rec["last_blocked_iso_week"] == "2026-W42"
        rebase = audit["counter_rebase"]
        assert rebase["source"] == "history_replay"
        assert rebase["n_audits"] == 16
        assert rebase["from"]["executor_params"] == 16
        assert rebase["to"]["executor_params"] == 12
        written = json.loads(s3.puts[-1]["Body"])
        assert written["loops"]["executor_params"]["consecutive_blocked_weeks"] == 13

    def test_rule_era_prior_does_not_replay(self):
        s3 = _live_bucket()
        latest = {"schema_version": 1, "as_of": "2026-10-16",
                  "loops": {loop: _rec("blocked", 13, "2026-W42") for loop in LOOPS}}
        s3.objects["config/apply_audit/latest.json"] = latest
        s3.list_objects_v2 = MagicMock(side_effect=AssertionError("must not list"))
        audit = emit_apply_audit(
            bucket="b", run_date="2026-10-23",
            opt_results={"executor_rec": _executor_blocked()},
            upload=True, s3_client=s3,
        )
        assert audit["loops"]["executor_params"]["consecutive_blocked_weeks"] == 14
        assert "counter_rebase" not in audit

    def test_history_not_ending_at_latest_falls_back_to_legacy_counter(self):
        s3 = _live_bucket()
        del s3.objects["config/apply_audit/2026-10-09.json"]
        audit = emit_apply_audit(
            bucket="b", run_date="2026-10-16",
            opt_results={"executor_rec": _executor_blocked()},
            upload=True, s3_client=s3,
        )
        assert audit["loops"]["executor_params"]["consecutive_blocked_weeks"] == 17
        assert "counter_rebase" not in audit

    def test_unreadable_history_falls_back_to_legacy_counter(self):
        s3 = _live_bucket()
        s3.objects["config/apply_audit/2026-08-21.json"] = {"as_of": "2026-08-21", "loops": "garbage"}
        prior = s3.objects["config/apply_audit/latest.json"]
        replayed, rebase = rebase_prior_from_history("b", prior, s3_client=s3)
        assert replayed is prior
        assert rebase is None

    def test_list_denied_falls_back_to_legacy_counter(self):
        s3 = _live_bucket()
        s3.list_objects_v2 = MagicMock(
            side_effect=ClientError({"Error": {"Code": "AccessDenied"}}, "ListObjectsV2"),
        )
        audit = emit_apply_audit(
            bucket="b", run_date="2026-10-16",
            opt_results={"executor_rec": _executor_blocked()},
            upload=True, s3_client=s3,
        )
        assert audit["loops"]["executor_params"]["consecutive_blocked_weeks"] == 17

    def test_no_upload_never_reads_history(self):
        s3 = MagicMock()
        emit_apply_audit(bucket="b", run_date="2026-10-16",
                         opt_results={"executor_rec": _executor_blocked()},
                         upload=False, s3_client=s3)
        assert s3.list_objects_v2.call_args_list == []
