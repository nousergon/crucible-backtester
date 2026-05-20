"""Tests for optimizer.tech_weight_ablation.

PR-C of the 2026-05-09 sector-team diagnostic arc. Given persisted
sub-scores (PR-B's research v15 migration), find the per-sector weight
config that minimizes (most-negative) corr(rank, return_5d). Surfaced
from the post-mortem on quant rank inversion in healthcare/industrials/
tech.

Locked behavior:

- WeightConfig validates weights sum to 1.0
- Synthetic score = weighted sum of sub-scores
- Re-rank within (team, eval_date), then corr(rank, ret) across team
- Recommendation gates: ≥30 rows/team, best must beat current by 0.10+
- Recommendation-only — applied=False with explanatory note
- Schema-missing graceful degradation (status=no_data, no crash)
- min_weeks gate produces insufficient_data status
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

import pytest

from optimizer.tech_weight_ablation import (
    DEFAULT_GRID,
    S3_LIVE_KEY,
    S3_SHADOW_PREFIX,
    WeightConfig,
    _MIN_CONSECUTIVE_WEEKS,
    _MIN_IMPROVEMENT,
    _MIN_ROWS_PER_TEAM,
    _build_per_sector_payload,
    _check_reproduction_gate,
    _evaluate_team_under_config,
    _load_live_composite_weights_per_sector,
    apply,
    compute_tech_weight_ablation,
    init_config,
)


@pytest.fixture
def conn():
    """Build an in-memory DB with the v15 schema (sub-score columns)."""
    c = sqlite3.connect(":memory:")
    c.executescript("""
        CREATE TABLE team_candidates (
            id INTEGER PRIMARY KEY,
            ticker TEXT, eval_date TEXT, team_id TEXT,
            quant_rank INTEGER, quant_score REAL, qual_score REAL,
            team_recommended INTEGER DEFAULT 0,
            rsi_sub_score REAL, macd_sub_score REAL,
            ma50_sub_score REAL, ma200_sub_score REAL,
            momentum_sub_score REAL
        );
        CREATE TABLE universe_returns (
            id INTEGER PRIMARY KEY,
            ticker TEXT, eval_date TEXT, return_5d REAL, beat_spy_5d INTEGER
        );
    """)
    yield c
    c.close()


def _seed(conn, team_id: str, eval_date: str, picks: list[tuple]):
    """Insert (ticker, rsi, macd, ma50, ma200, momentum, return_5d) rows."""
    for i, (ticker, rsi, macd, ma50, ma200, mom, ret) in enumerate(picks, 1):
        conn.execute(
            "INSERT INTO team_candidates "
            "(ticker, eval_date, team_id, quant_rank, quant_score, "
            "rsi_sub_score, macd_sub_score, ma50_sub_score, "
            "ma200_sub_score, momentum_sub_score) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (ticker, eval_date, team_id, i, 0.0,
             rsi, macd, ma50, ma200, mom),
        )
        conn.execute(
            "INSERT INTO universe_returns (ticker, eval_date, return_5d) "
            "VALUES (?,?,?)",
            (ticker, eval_date, ret),
        )


# ── WeightConfig ────────────────────────────────────────────────────────────


class TestWeightConfig:
    def test_validates_sum_to_one(self):
        with pytest.raises(ValueError, match="weights sum"):
            WeightConfig("bad", rsi=0.5, macd=0.5, ma50=0.5,
                         ma200=0.0, momentum=0.0)

    def test_synthetic_score_weighted_sum(self):
        c = WeightConfig("test", rsi=0.5, macd=0.5,
                         ma50=0.0, ma200=0.0, momentum=0.0)
        # 0.5*100 + 0.5*60 = 80
        assert c.synthetic_score(100, 60, 0, 0, 0) == 80.0


def test_default_grid_includes_current():
    names = {c.name for c in DEFAULT_GRID}
    assert "current_default" in names
    assert "rsi_only" in names
    assert "momentum_only" in names


def test_default_grid_all_valid():
    """Each named config must sum to 1.0 (validated at construction)."""
    for c in DEFAULT_GRID:
        s = c.rsi + c.macd + c.ma50 + c.ma200 + c.momentum
        assert abs(s - 1.0) < 1e-6


# ── _evaluate_team_under_config ─────────────────────────────────────────────


class TestEvaluateTeamUnderConfig:
    def test_returns_none_with_single_row_per_date(self):
        # Each date has only 1 row → rank is undefined
        rows = [
            ("2026-04-25", 50, 50, 50, 50, 50, 0.05),
            ("2026-05-02", 50, 50, 50, 50, 50, 0.03),
        ]
        cfg = next(c for c in DEFAULT_GRID if c.name == "current_default")
        assert _evaluate_team_under_config(rows, cfg) is None

    def test_perfect_inverse_rank_correlation(self):
        """If high synthetic-score rows have HIGHEST returns, the
        re-rank will produce strongly negative correlation (rank 1 →
        best return = skilled scorer)."""
        # All on one date, 5 picks. Assign sub-scores so that rsi_only
        # config produces a ranking that perfectly matches return
        # ordering (highest rsi → highest return).
        rows = [
            ("2026-04-25", 90, 50, 50, 50, 50, 0.05),  # high rsi → high return
            ("2026-04-25", 70, 50, 50, 50, 50, 0.03),
            ("2026-04-25", 50, 50, 50, 50, 50, 0.01),
            ("2026-04-25", 30, 50, 50, 50, 50, -0.02),
            ("2026-04-25", 10, 50, 50, 50, 50, -0.04),  # low rsi → low return
        ]
        rsi_only = next(c for c in DEFAULT_GRID if c.name == "rsi_only")
        corr = _evaluate_team_under_config(rows, rsi_only)
        assert corr is not None
        assert corr < -0.9  # near-perfect negative — skilled

    def test_anti_skilled_under_wrong_weight(self):
        """If we re-rank by momentum_only when momentum is anti-correlated
        with returns, we should see strongly POSITIVE rank correlation."""
        rows = [
            ("2026-04-25", 50, 50, 50, 50, 90, -0.04),  # high momentum → loss
            ("2026-04-25", 50, 50, 50, 50, 70, -0.02),
            ("2026-04-25", 50, 50, 50, 50, 50, 0.01),
            ("2026-04-25", 50, 50, 50, 50, 30, 0.03),
            ("2026-04-25", 50, 50, 50, 50, 10, 0.05),  # low momentum → win
        ]
        mom_only = next(c for c in DEFAULT_GRID if c.name == "momentum_only")
        corr = _evaluate_team_under_config(rows, mom_only)
        assert corr is not None
        assert corr > 0.9  # near-perfect positive — anti-skill


# ── compute_tech_weight_ablation ────────────────────────────────────────────


class TestComputeTechWeightAblation:
    def test_must_provide_db(self):
        result = compute_tech_weight_ablation()
        assert result["status"] == "error"

    def test_invalid_run_date(self, conn):
        result = compute_tech_weight_ablation(
            db_conn=conn, run_date="not-a-date"
        )
        assert result["status"] == "error"

    def test_missing_team_candidates_table(self):
        c = sqlite3.connect(":memory:")
        result = compute_tech_weight_ablation(
            db_conn=c, run_date="2026-05-09",
        )
        assert result["status"] == "no_data"
        assert "team_candidates" in result["reason"]
        c.close()

    def test_missing_sub_score_columns(self):
        """Pre-v15 schema (just quant_rank/quant_score) must surface
        clearly so operator knows the producer-side migration hasn't
        rolled out yet, not crash."""
        c = sqlite3.connect(":memory:")
        c.executescript("""
            CREATE TABLE team_candidates (
                id INTEGER PRIMARY KEY, ticker TEXT, eval_date TEXT,
                team_id TEXT, quant_rank INTEGER, quant_score REAL
            );
            CREATE TABLE universe_returns (
                id INTEGER PRIMARY KEY, ticker TEXT, eval_date TEXT,
                return_5d REAL
            );
        """)
        result = compute_tech_weight_ablation(
            db_conn=c, run_date="2026-05-09",
        )
        assert result["status"] == "no_data"
        assert "sub-score columns" in result["reason"]
        c.close()

    def test_insufficient_rows_per_team(self, conn):
        """Each team needs ≥ _MIN_ROWS_PER_TEAM. With 5 rows for one
        team and nothing else, status must be insufficient_data."""
        _seed(conn, "technology", "2026-05-02", [
            ("A", 80, 80, 80, 80, 80, 0.05),
            ("B", 70, 70, 70, 70, 70, 0.03),
            ("C", 60, 60, 60, 60, 60, 0.01),
            ("D", 50, 50, 50, 50, 50, -0.02),
            ("E", 40, 40, 40, 40, 40, -0.04),
        ])
        result = compute_tech_weight_ablation(
            db_conn=conn, run_date="2026-05-09",
        )
        assert result["status"] == "insufficient_data"
        # Per-team status reflects the floor
        tech = next(t for t in result["per_team"]
                    if t["team_id"] == "technology")
        assert tech["status"] == "insufficient_data"
        assert tech["min_required"] == _MIN_ROWS_PER_TEAM

    def test_recommendation_keep_current_when_close(self, conn):
        """If best ablation config beats current_default by less than
        _MIN_IMPROVEMENT, recommendation must be 'keep_current'."""
        # Seed exactly 30 rows where current_default is already
        # near-optimal (no big improvement possible).
        for i in range(30):
            date = f"2026-04-{(i % 8) + 1:02d}"  # 8 dates, ~4 picks each
            score = 50 + (i % 10)
            ret = score / 1000.0  # weak positive correlation with all sub-scores
            _seed(conn, "financials", date, [
                (f"T{i}", score, score, score, score, score, ret),
            ])
        # Need ≥2 rows per date for ranking — re-seed with 4 per date
        conn.execute("DELETE FROM team_candidates")
        conn.execute("DELETE FROM universe_returns")
        for d in range(8):
            date = f"2026-04-{d+1:02d}"
            picks = [
                (f"T{d}{i}", 50 + i*5, 50 + i*5, 50 + i*5, 50 + i*5, 50 + i*5,
                 (i+1) * 0.005)
                for i in range(4)
            ]
            _seed(conn, "financials", date, picks)
        result = compute_tech_weight_ablation(
            db_conn=conn, run_date="2026-04-30",
        )
        assert result["status"] == "ok"
        fin = next(t for t in result["per_team"]
                   if t["team_id"] == "financials")
        assert fin["status"] == "ok"
        # All sub-scores identical per ticker (i=0..3), so all configs
        # produce identical ranking → 0 improvement → keep_current.
        assert fin["recommendation"] == "keep_current"

    def test_recommendation_switch_when_alternative_clears_gate(self, conn):
        """Anti-skill with current_default but clean signal under
        rsi_only: recommend the switch."""
        # 8 dates × 5 picks each. current_default weights have momentum
        # at 0.25, but momentum is anti-correlated with returns.
        # rsi_only picks the right names.
        for d in range(8):
            date = f"2026-04-{d+1:02d}"
            # rsi-skilled order: rsi descending matches return descending
            # momentum-anti-skilled: momentum descending matches return ASCENDING
            picks = [
                (f"H{d}A", 90, 50, 50, 50, 10, 0.05),
                (f"H{d}B", 80, 50, 50, 50, 30, 0.03),
                (f"H{d}C", 60, 50, 50, 50, 50, 0.01),
                (f"H{d}D", 40, 50, 50, 50, 70, -0.02),
                (f"H{d}E", 20, 50, 50, 50, 90, -0.04),
            ]
            _seed(conn, "healthcare", date, picks)
        result = compute_tech_weight_ablation(
            db_conn=conn, run_date="2026-04-30",
        )
        assert result["status"] == "ok"
        hc = next(t for t in result["per_team"]
                  if t["team_id"] == "healthcare")
        assert hc["status"] == "ok"
        assert hc["recommendation"].startswith("switch_to_")
        # rsi_only should win (or at least beat current_default by gate)
        assert hc["best_corr"] < hc["current_corr"]
        assert (hc["current_corr"] - hc["best_corr"]) >= _MIN_IMPROVEMENT

    def test_recommendation_only_no_apply(self, conn):
        for d in range(8):
            date = f"2026-04-{d+1:02d}"
            picks = [
                (f"T{d}A", 90, 50, 50, 50, 10, 0.05),
                (f"T{d}B", 80, 50, 50, 50, 30, 0.03),
                (f"T{d}C", 60, 50, 50, 50, 50, 0.01),
                (f"T{d}D", 40, 50, 50, 50, 70, -0.02),
                (f"T{d}E", 20, 50, 50, 50, 90, -0.04),
            ]
            _seed(conn, "technology", date, picks)
        result = compute_tech_weight_ablation(
            db_conn=conn, run_date="2026-04-30",
        )
        assert result["status"] == "ok"
        # compute_tech_weight_ablation() itself NEVER auto-applies;
        # apply() is a separate call gated on two flags + the
        # reproduction guard (ROADMAP L2553 auto-apply cutover).
        assert result["applied"] is False
        assert "apply()" in result["apply_note"]
        assert "use_tech_ablation_target" in result["apply_note"]

    def test_window_filtering(self, conn):
        # Old rows outside window must be excluded
        for d in range(8):
            date = f"2026-04-{d+1:02d}"
            picks = [
                (f"T{d}A", 90, 50, 50, 50, 10, 0.05),
                (f"T{d}B", 80, 50, 50, 50, 30, 0.03),
                (f"T{d}C", 60, 50, 50, 50, 50, 0.01),
                (f"T{d}D", 40, 50, 50, 50, 70, -0.02),
                (f"T{d}E", 20, 50, 50, 50, 90, -0.04),
            ]
            _seed(conn, "technology", date, picks)
        # Add ancient rows that should be excluded
        _seed(conn, "technology", "2024-01-01", [
            ("ANCIENT", 99, 99, 99, 99, 99, 0.99),
        ])
        result = compute_tech_weight_ablation(
            db_conn=conn, run_date="2026-04-30", lookback_weeks=8,
        )
        tech = next(t for t in result["per_team"]
                    if t["team_id"] == "technology")
        # 8 dates × 5 picks = 40 rows in window; ancient must not appear
        assert tech["n_rows"] == 40


# ── Live composite_weights_per_sector reader ────────────────────────────────


class TestLiveBaselineWeightsReader:
    """Lock the consumer-side awareness of the L1374 schema addition.

    The ablation module reads alpha-engine-config/research/scoring.yaml's
    technical.composite_weights_per_sector and surfaces the per-team
    override (if present) on every team result. Gate semantics are
    unchanged — that's the L2202 cutover's scope.
    """

    def test_loader_missing_file_returns_empty(self, tmp_path):
        missing = tmp_path / "does_not_exist" / "scoring.yaml"
        assert _load_live_composite_weights_per_sector([missing]) == {}

    def test_loader_reads_override_block(self, tmp_path):
        yaml_path = tmp_path / "scoring.yaml"
        yaml_path.write_text(
            "technical:\n"
            "  composite_weights:\n"
            "    rsi: 0.25\n"
            "    macd: 0.20\n"
            "    ma50: 0.15\n"
            "    ma200: 0.15\n"
            "    momentum: 0.25\n"
            "  composite_weights_per_sector:\n"
            "    healthcare:\n"
            "      rsi: 0.50\n"
            "      macd: 0.125\n"
            "      ma50: 0.125\n"
            "      ma200: 0.125\n"
            "      momentum: 0.125\n"
        )
        result = _load_live_composite_weights_per_sector([yaml_path])
        assert "healthcare" in result
        assert result["healthcare"]["rsi"] == 0.50

    def test_loader_missing_block_returns_empty(self, tmp_path):
        yaml_path = tmp_path / "scoring.yaml"
        yaml_path.write_text(
            "technical:\n"
            "  composite_weights:\n"
            "    rsi: 0.25\n"
        )
        assert _load_live_composite_weights_per_sector([yaml_path]) == {}

    def test_loader_empty_block_returns_empty(self, tmp_path):
        yaml_path = tmp_path / "scoring.yaml"
        yaml_path.write_text(
            "technical:\n"
            "  composite_weights_per_sector: {}\n"
        )
        assert _load_live_composite_weights_per_sector([yaml_path]) == {}

    def test_per_team_output_includes_live_baseline_weights_field(
        self, conn, monkeypatch,
    ):
        """Every per-team result must carry `live_baseline_weights` (None when no override)."""
        monkeypatch.setattr(
            "optimizer.tech_weight_ablation._load_live_composite_weights_per_sector",
            lambda: {
                "healthcare": {
                    "rsi": 0.50, "macd": 0.125,
                    "ma50": 0.125, "ma200": 0.125, "momentum": 0.125,
                },
            },
        )
        for d in range(8):
            date = f"2026-04-{d+1:02d}"
            picks = [
                (f"H{d}A", 90, 50, 50, 50, 10, 0.05),
                (f"H{d}B", 80, 50, 50, 50, 30, 0.03),
                (f"H{d}C", 60, 50, 50, 50, 50, 0.01),
                (f"H{d}D", 40, 50, 50, 50, 70, -0.02),
                (f"H{d}E", 20, 50, 50, 50, 90, -0.04),
            ]
            _seed(conn, "healthcare", date, picks)
        result = compute_tech_weight_ablation(
            db_conn=conn, run_date="2026-04-30",
        )
        hc = next(t for t in result["per_team"] if t["team_id"] == "healthcare")
        assert hc["live_baseline_weights"] == {
            "rsi": 0.50, "macd": 0.125,
            "ma50": 0.125, "ma200": 0.125, "momentum": 0.125,
        }
        tech = next(t for t in result["per_team"] if t["team_id"] == "technology")
        # No override for technology → field present but None
        assert "live_baseline_weights" in tech
        assert tech["live_baseline_weights"] is None


# ── Auto-apply path (ROADMAP L2553) ─────────────────────────────────────────


class _StubS3:
    """In-memory S3 stub for apply()/_check_reproduction_gate() tests.

    Mirrors the shape the boto3 client returns: put_object stores a
    body; get_object raises ClientError-shaped NoSuchKey on miss;
    list_objects_v2 walks keys by prefix.
    """

    class _ClientError(Exception):
        def __init__(self, code: str = "NoSuchKey") -> None:
            self.response = {"Error": {"Code": code}}

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket, Key, Body, ContentType=None, **kw):
        body = Body if isinstance(Body, (bytes, bytearray)) else str(Body).encode()
        self.store[(Bucket, Key)] = bytes(body)
        return {"ETag": "stub"}

    def get_object(self, *, Bucket, Key):
        from io import BytesIO
        if (Bucket, Key) not in self.store:
            raise self._ClientError("NoSuchKey")
        return {"Body": BytesIO(self.store[(Bucket, Key)])}

    def list_objects_v2(self, *, Bucket, Prefix, **kw):
        return {
            "Contents": [
                {"Key": k} for (b, k) in self.store
                if b == Bucket and k.startswith(Prefix)
            ],
        }


@pytest.fixture(autouse=True)
def _reset_cfg():
    """Each test runs against a fresh tech_weight_ablation config block."""
    init_config({})
    yield
    init_config({})


def _ok_result(recommendations: dict[str, str]) -> dict:
    """Synthesize a compute_tech_weight_ablation 'ok' result with
    the given recommendations dict."""
    return {
        "status": "ok",
        "run_date": "2026-05-19",
        "min_improvement": _MIN_IMPROVEMENT,
        "recommendations": recommendations,
        "per_team": [
            {"team_id": tid, "status": "ok", "best_config": cfg}
            for tid, cfg in recommendations.items()
        ],
    }


class TestBuildPerSectorPayload:
    def test_maps_recommendation_names_to_weight_dicts(self):
        result = _ok_result({"healthcare": "rsi_only", "technology": "trend_only"})
        payload = _build_per_sector_payload(result)
        assert set(payload.keys()) == {"healthcare", "technology"}
        # rsi_only: 100% RSI
        assert payload["healthcare"] == {
            "rsi": 1.0, "macd": 0.0, "ma50": 0.0, "ma200": 0.0, "momentum": 0.0,
        }
        # trend_only: 25/25/25/25 (no RSI)
        assert payload["technology"] == {
            "rsi": 0.0, "macd": 0.25, "ma50": 0.25, "ma200": 0.25, "momentum": 0.25,
        }

    def test_unknown_config_name_is_dropped(self):
        result = _ok_result({"healthcare": "definitely_not_in_grid"})
        assert _build_per_sector_payload(result) == {}

    def test_empty_recommendations_yields_empty_payload(self):
        assert _build_per_sector_payload(_ok_result({})) == {}


class TestApplyShadowGating:
    def test_flag_off_skips_all_writes(self):
        s3 = _StubS3()
        # default _cfg is empty → use_tech_ablation_target=False
        out = apply(
            _ok_result({"healthcare": "rsi_only"}), "alpha-engine-research",
        )
        assert out["applied"] is False
        assert out["reason"] == "use_tech_ablation_target=False"
        # No S3 side effects at all.
        assert len(s3.store) == 0

    def test_status_not_ok_skips(self):
        init_config({"tech_weight_ablation": {"use_tech_ablation_target": True}})
        out = apply({"status": "insufficient_data"}, "alpha-engine-research")
        assert out["applied"] is False
        assert "status=insufficient_data" in out["reason"]

    def test_empty_recommendations_skips(self):
        init_config({"tech_weight_ablation": {"use_tech_ablation_target": True}})
        out = apply(_ok_result({}), "alpha-engine-research")
        assert out["applied"] is False
        assert "no per-sector recommendation" in out["reason"]

    def test_shadow_only_writes_archive_not_live(self, monkeypatch):
        """flag on, enforce off → shadow archive written, live key untouched."""
        s3 = _StubS3()
        monkeypatch.setattr(
            "boto3.client", lambda svc, *a, **kw: s3,
        )
        init_config({"tech_weight_ablation": {
            "use_tech_ablation_target": True,
            "enforce_tech_ablation": False,
        }})
        out = apply(
            _ok_result({"healthcare": "rsi_only"}),
            "alpha-engine-research",
        )
        assert out["applied"] is False
        assert "shadow mode" in out["reason"]
        assert "shadow_key" in out
        # Shadow archive present
        assert any(
            k.startswith(S3_SHADOW_PREFIX + "/") and k.endswith(".json")
            for (_, k) in s3.store
        )
        # Live key NOT written
        assert ("alpha-engine-research", S3_LIVE_KEY) not in s3.store


class TestReproductionGate:
    def test_insufficient_history_fails_gate(self):
        s3 = _StubS3()
        # Empty bucket → no prior archives
        current = {"healthcare": {"rsi": 1.0, "macd": 0, "ma50": 0, "ma200": 0, "momentum": 0}}
        out = _check_reproduction_gate(s3, "alpha-engine-research", current)
        assert out["passed"] is False
        assert "only 0 prior shadow archive" in out["reason"]
        assert out["n_consecutive"] == 0

    def test_exact_match_across_min_consecutive_passes(self, monkeypatch):
        s3 = _StubS3()
        current = {"healthcare": {"rsi": 1.0, "macd": 0, "ma50": 0, "ma200": 0, "momentum": 0}}
        # Seed _MIN_CONSECUTIVE_WEEKS prior archives, all matching
        import json as _json
        for i in range(_MIN_CONSECUTIVE_WEEKS):
            # Lexicographically-sorted YYMMDDHHMM-style keys
            key = f"{S3_SHADOW_PREFIX}/26051{i}1234_result.json"
            s3.store[("alpha-engine-research", key)] = _json.dumps({
                "per_sector": current,
            }).encode()
        out = _check_reproduction_gate(s3, "alpha-engine-research", current)
        assert out["passed"] is True
        assert out["n_consecutive"] == _MIN_CONSECUTIVE_WEEKS

    def test_one_drift_breaks_streak(self, monkeypatch):
        s3 = _StubS3()
        current = {"healthcare": {"rsi": 1.0, "macd": 0, "ma50": 0, "ma200": 0, "momentum": 0}}
        drift = {"healthcare": {"rsi": 0.5, "macd": 0.5, "ma50": 0, "ma200": 0, "momentum": 0}}
        # 3 matches then 1 drift — sorted-desc means drift is in position [3]
        # (oldest of the 4 we read); should fail at the last archive.
        import json as _json
        # YYMMDDHHMM keys, sorted desc means biggest first (most recent first)
        payloads = [current, current, current, drift]
        for i, payload in enumerate(payloads):
            # newer keys = bigger lex prefix
            key = f"{S3_SHADOW_PREFIX}/2605{9 - i}01234_result.json"
            s3.store[("alpha-engine-research", key)] = _json.dumps({
                "per_sector": payload,
            }).encode()
        out = _check_reproduction_gate(s3, "alpha-engine-research", current)
        assert out["passed"] is False
        assert "reproduction gate broken at archive[-4]" in out["reason"]


class TestApplyLiveGating:
    def _seed_matching_history(self, s3, current_payload, n: int):
        """Plant n shadow archives that all match current_payload."""
        import json as _json
        for i in range(n):
            key = f"{S3_SHADOW_PREFIX}/26050{i}1234_result.json"
            s3.store[("alpha-engine-research", key)] = _json.dumps({
                "per_sector": current_payload,
            }).encode()

    def test_enforce_with_insufficient_history_writes_shadow_not_live(
        self, monkeypatch,
    ):
        """enforce=True but reproduction gate fails → shadow yes, live no."""
        s3 = _StubS3()
        monkeypatch.setattr("boto3.client", lambda svc, *a, **kw: s3)
        init_config({"tech_weight_ablation": {
            "use_tech_ablation_target": True,
            "enforce_tech_ablation": True,
        }})
        out = apply(
            _ok_result({"healthcare": "rsi_only"}),
            "alpha-engine-research",
        )
        assert out["applied"] is False
        assert "reproduction gate" in out["reason"]
        # Shadow archive written
        assert any(
            k.startswith(S3_SHADOW_PREFIX + "/")
            for (_, k) in s3.store
        )
        # Live NOT written
        assert ("alpha-engine-research", S3_LIVE_KEY) not in s3.store

    def test_enforce_with_full_reproduction_writes_live(self, monkeypatch):
        s3 = _StubS3()
        monkeypatch.setattr("boto3.client", lambda svc, *a, **kw: s3)
        # Pre-seed: _MIN_CONSECUTIVE_WEEKS - 1 prior shadow archives all
        # matching this week's payload. apply() writes this week's shadow
        # first (making it the {n}th in the streak), then checks the gate.
        rsi_only = {
            "rsi": 1.0, "macd": 0.0, "ma50": 0.0,
            "ma200": 0.0, "momentum": 0.0,
        }
        current = {"healthcare": rsi_only}
        self._seed_matching_history(
            s3, current, n=_MIN_CONSECUTIVE_WEEKS - 1,
        )
        init_config({"tech_weight_ablation": {
            "use_tech_ablation_target": True,
            "enforce_tech_ablation": True,
        }})
        out = apply(
            _ok_result({"healthcare": "rsi_only"}),
            "alpha-engine-research",
        )
        assert out["applied"] is True, out
        assert out["live_key"] == S3_LIVE_KEY
        assert out["per_sector"] == current
        # Live key written
        assert ("alpha-engine-research", S3_LIVE_KEY) in s3.store
        # Shadow archive also present
        assert any(
            k.startswith(S3_SHADOW_PREFIX + "/")
            and k.endswith(".json")
            and not k.endswith("/latest.json")
            for (_, k) in s3.store
        )

    def test_drift_in_history_blocks_live_write(self, monkeypatch):
        s3 = _StubS3()
        monkeypatch.setattr("boto3.client", lambda svc, *a, **kw: s3)
        rsi_only = {
            "rsi": 1.0, "macd": 0.0, "ma50": 0.0,
            "ma200": 0.0, "momentum": 0.0,
        }
        # Pre-seed 2 matching + 1 drift → reproduction breaks at the drift.
        # After apply() writes this week's archive, that's 3 in a row of
        # matching + 1 drift before; gate requires 4-in-a-row so blocks.
        drift_payload = {"healthcare": {
            "rsi": 0.5, "macd": 0.5, "ma50": 0.0, "ma200": 0.0, "momentum": 0.0,
        }}
        import json as _json
        # Older drift archive (smaller lex prefix), then 2 matching newer.
        s3.store[("alpha-engine-research",
                  f"{S3_SHADOW_PREFIX}/2605011234_result.json")] = (
            _json.dumps({"per_sector": drift_payload}).encode()
        )
        s3.store[("alpha-engine-research",
                  f"{S3_SHADOW_PREFIX}/2605051234_result.json")] = (
            _json.dumps({"per_sector": {"healthcare": rsi_only}}).encode()
        )
        s3.store[("alpha-engine-research",
                  f"{S3_SHADOW_PREFIX}/2605091234_result.json")] = (
            _json.dumps({"per_sector": {"healthcare": rsi_only}}).encode()
        )
        init_config({"tech_weight_ablation": {
            "use_tech_ablation_target": True,
            "enforce_tech_ablation": True,
        }})
        out = apply(
            _ok_result({"healthcare": "rsi_only"}),
            "alpha-engine-research",
        )
        assert out["applied"] is False
        assert "reproduction gate broken" in out["reason"]
        assert ("alpha-engine-research", S3_LIVE_KEY) not in s3.store
