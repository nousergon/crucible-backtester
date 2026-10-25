"""The deployed-strategy headline reaches the PUBLISHED weekly report
(alpha-engine-config-I12658).

backtest.py computes the deployed strategy's backtest (production signals ->
MVO optimizer, config#1053), and evaluate.py writes ``backtest/<date>/report.md``
last, at the same key. Until this fix evaluate.py built that report without
``production_stats``, so every published weekly report from 2026-09-25 to
2026-10-23 said "the deployed-strategy backtest did not run", while S3's
version history shows backtest.py's own report carrying an ``ok`` headline
three times on 2026-10-24.

The contract tested here:
  * backtest.py persists the result as ``production_strategy.json`` via
    ``reporter.save``;
  * evaluate.py reads it (``_load_production_strategy``) and passes it to
    ``build_report``;
  * a missing or unreadable file renders a banner naming THAT gap, never the
    false "did not run".
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
from botocore.exceptions import ClientError

import backtest as B
import evaluate as E
from reporter import _section_deployed_strategy, build_report, save

REPO = Path(__file__).resolve().parents[1]
DATE = "2026-10-23"
PREFIX = f"backtest/{DATE}"

# A production_stats shaped like run_production_strategy_backtest's ok return.
OK_STATS = {
    "status": "ok",
    "metrics": {
        "total_return": 0.187,
        "sharpe_ratio": 0.51,
        "sortino_ratio": 0.82,
        "max_drawdown": -0.046,
        "psr": 0.95,
        "total_alpha": 0.037,
    },
    "risk_matched": None,
    "input_fingerprint": {"predictions_by_date": "abc"},
    "production_window": ["2026-03-12", DATE],
    "n_production_dates": 45,
    "n_rebalances": 9,
    "n_solver_failures": 0,
}

DID_NOT_RUN = "the deployed-strategy backtest did not run"


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class DictS3:
    def __init__(self, objects: dict[str, bytes] | None = None, error_code: str | None = None):
        self.objects = objects or {}
        self.error_code = error_code

    def get_object(self, Bucket: str, Key: str):  # noqa: N803 — boto3 kwargs
        if self.error_code:
            raise ClientError({"Error": {"Code": self.error_code, "Message": "x"}}, "GetObject")
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject")
        return {"Body": _Body(self.objects[Key])}


def _kwarg_names_of_call(path: Path, func_name: str, assigned_to: str) -> list[set[str]]:
    """Keyword names of each ``<assigned_to> = <func_name>(...)`` in ``path``."""
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == func_name
            and any(isinstance(t, ast.Name) and t.id == assigned_to for t in node.targets)
        ):
            found.append({k.arg for k in node.value.keywords if k.arg})
    return found


# ── the defect, end to end ─────────────────────────────────────────────────


def test_published_report_renders_backtest_ok_headline_not_did_not_run(tmp_path):
    """The failing shape: backtest.py produced an ok headline, and the report
    evaluate.py publishes must show it. On the pre-fix evaluate.py this fails:
    the report call passed no production_stats and _load_production_strategy
    did not exist."""
    # backtest.py side: persist the ok result through reporter.save.
    record = B._production_strategy_record(OK_STATS, run_date=DATE, mode="all")
    out_dir = save(
        report_md="# backtest.py report",
        signal_quality={"status": "skipped"},
        score_analysis=[],
        run_date=DATE,
        results_dir=str(tmp_path),
        production_strategy=record,
    )
    body = (out_dir / "production_strategy.json").read_bytes()

    # evaluate.py side: read it back from the same key and build the report.
    loaded = E._load_production_strategy(DictS3({f"{PREFIX}/production_strategy.json": body}), "b", PREFIX)
    md = build_report(
        run_date=DATE,
        signal_quality={"status": "skipped"},
        regime_analysis=[],
        score_analysis=[],
        attribution={"status": "skipped"},
        production_stats=loaded,
    )

    assert "DEPLOYED-STRATEGY BACKTEST UNAVAILABLE" not in md
    assert DID_NOT_RUN not in md
    assert "45 production dates" in md


def test_evaluate_report_call_passes_production_stats():
    calls = _kwarg_names_of_call(REPO / "evaluate.py", "build_report", "report_md")
    assert calls, "evaluate.py has no `report_md = build_report(...)` call"
    assert all("production_stats" in kw for kw in calls), (
        "evaluate.py builds the published report.md without production_stats, "
        "so it always renders 'the deployed-strategy backtest did not run'"
    )


def test_backtest_save_call_persists_production_strategy():
    calls = _kwarg_names_of_call(REPO / "backtest.py", "save", "out_dir")
    assert calls
    assert any("production_strategy" in kw for kw in calls)


def test_init_data_sources_loads_production_strategy_into_config():
    src = (REPO / "evaluate.py").read_text()
    assert 'config["_production_stats"] = _load_production_strategy(' in src


# ── evaluate.py loader ─────────────────────────────────────────────────────


def test_loader_returns_the_persisted_record():
    body = json.dumps(OK_STATS).encode()
    loaded = E._load_production_strategy(DictS3({f"{PREFIX}/production_strategy.json": body}), "b", PREFIX)
    assert loaded == OK_STATS


def test_missing_file_banners_the_gap_not_did_not_run():
    loaded = E._load_production_strategy(DictS3(), "bkt", PREFIX)
    assert loaded["status"] == "not_persisted"
    banner = "\n".join(_section_deployed_strategy(loaded))
    assert "DEPLOYED-STRATEGY BACKTEST UNAVAILABLE" in banner
    assert DID_NOT_RUN not in banner
    assert f"s3://bkt/{PREFIX}/production_strategy.json" in banner


def test_unreadable_file_never_raises():
    assert E._load_production_strategy(DictS3(error_code="AccessDenied"), "b", PREFIX)["status"] == "unreadable"
    bad = DictS3({f"{PREFIX}/production_strategy.json": b"not json"})
    assert E._load_production_strategy(bad, "b", PREFIX)["status"] == "unreadable"
    not_obj = DictS3({f"{PREFIX}/production_strategy.json": b"[1, 2]"})
    assert E._load_production_strategy(not_obj, "b", PREFIX)["status"] == "unreadable"


def test_a_persisted_failure_renders_its_own_reason():
    record = B._production_strategy_record(
        {"status": "error", "error": "executor_paths not found on disk"}, run_date=DATE, mode="all",
    )
    loaded = E._load_production_strategy(
        DictS3({f"{PREFIX}/production_strategy.json": json.dumps(record).encode()}), "b", PREFIX,
    )
    banner = "\n".join(_section_deployed_strategy(loaded))
    assert "executor_paths not found on disk" in banner


# ── backtest.py record + fingerprint ───────────────────────────────────────


def test_record_names_its_writer_and_keeps_the_result():
    record = B._production_strategy_record(OK_STATS, run_date=DATE, mode="portfolio")
    assert record["status"] == "ok"
    assert record["metrics"] == OK_STATS["metrics"]
    assert record["run_date"] == DATE
    assert record["writer"] == "backtest.py"
    assert record["mode"] == "portfolio"
    assert record["written_at"]
    assert B._production_strategy_record(None, run_date=DATE, mode="all")["status"] == "error"


def _inputs(close=100.0):
    idx = pd.to_datetime(["2026-10-21", "2026-10-22"])
    return {
        "predictions_by_date": {"2026-10-21": {"AAA": 0.01}},
        "alpha_uncertainty_by_date": {},
        "price_matrix": pd.DataFrame({"AAA": [close, 101.0]}, index=idx),
        "spy_prices": pd.Series([500.0, 501.0], index=idx),
        "sector_map": {"AAA": "Tech"},
    }


def test_fingerprint_is_stable_for_equal_inputs_and_moves_when_an_input_moves():
    a, b = B._input_fingerprint(_inputs()), B._input_fingerprint(_inputs())
    assert a == b
    assert set(a) == {
        "predictions_by_date", "alpha_uncertainty_by_date", "price_matrix", "spy_prices", "sector_map",
    }
    moved = B._input_fingerprint(_inputs(close=99.0))
    assert moved["price_matrix"] != a["price_matrix"]
    assert moved["predictions_by_date"] == a["predictions_by_date"]


def test_fingerprint_never_raises():
    class Unhashable:
        def __repr__(self):
            raise RuntimeError("no")

    fp = B._input_fingerprint({"sector_map": Unhashable()})
    assert fp["sector_map"].startswith("unhashable")
