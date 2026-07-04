"""L4520 slice 4 — producer contracts for the three LEGACY direct-write
auto-tuned configs (cross-repo).

Companion to PIPELINE_CONTRACT.yaml boundaries `scoring_weights` /
`research_params` / `predictor_params` and to the executor_params slice
(#314). These three predate the recommendation-artifact/assembler refactor
and write their live keys directly:

- config/scoring_weights.json   ← optimizer/weight_optimizer.py::apply_weights
- config/research_params.json   ← optimizer/research_optimizer.py::apply
- config/predictor_params.json  ← analysis/veto_analysis.py::apply (veto leg)
                                  + optimizer/predictor_optimizer.py::
                                    apply_recommendations (Phase 4 leg,
                                    merges {**existing, **updates})

Each consumer filters to the keys it knows (research: `_RP_DEFAULTS`;
predictor: explicit gets) — an undeclared new producer key is a tuned param
that silently never applies. Key sets are extracted from the producer SOURCE
(AST) where they are dict/subscript literals, and live-imported where they
are module constants, so ADDING a key fails this test until the contract +
consumer are updated. The declared sets mirror PIPELINE_CONTRACT.yaml (the
human SoT; per-repo CI can't import the config repo's YAML).
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer import research_optimizer, weight_optimizer

_REPO = Path(__file__).resolve().parent.parent


def _func_node(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def _literal_keys(fn: ast.FunctionDef, var: str) -> set[str]:
    """String keys assigned into ``var`` within ``fn`` — both dict-literal
    assignments (``var = {"k": ...}``, ``**spread`` entries ignored) and
    subscript assignments (``var["k"] = ...``)."""
    keys: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (isinstance(tgt, ast.Name) and tgt.id == var
                        and isinstance(node.value, ast.Dict)):
                    keys |= {k.value for k in node.value.keys
                             if isinstance(k, ast.Constant) and isinstance(k.value, str)}
                if (isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == var
                        and isinstance(tgt.slice, ast.Constant)
                        and isinstance(tgt.slice.value, str)):
                    keys.add(tgt.slice.value)
    return keys


# ── declared vocabularies (mirror PIPELINE_CONTRACT.yaml) ────────────────────

SCORING_WEIGHTS_DECLARED = {
    "quant", "qual",
    "updated_at", "n_samples", "confidence", "fit_target",
}
RESEARCH_PARAMS_DECLARED = {
    "short_interest_buy_threshold_pct", "short_interest_high_threshold_pct",
    "short_interest_buy_boost", "short_interest_high_boost",
    "institutional_min_funds", "institutional_boost",
    "consistency_bullish_dominance", "consistency_bearish_dominance",
    "consistency_low_score", "consistency_high_score",
    "updated_at", "n_samples", "correlations",
}
PREDICTOR_PARAMS_DECLARED = {
    # veto_analysis.apply payload
    "veto_confidence", "fit_target", "precision", "n_vetoes",
    "updated_at", "recommendation_reason",
    # predictor_optimizer.apply_recommendations updates
    "preferred_ensemble_mode", "ensemble_eval_date", "ensemble_eval_reason",
    "recommended_signal_threshold", "signal_threshold_eval_date",
    "signal_threshold_eval_reason",
    "prune_features", "pruning_eval_date", "pruning_eval_reason",
    # operator-written flip keys (consumer-read; no producer code path here)
    "regime_veto_enabled", "regime_veto_scale", "regime_veto_cap",
    "regime_forced_bear_enabled", "drawdown_regime_enabled",
}
_OPERATOR_ONLY = {
    "regime_veto_enabled", "regime_veto_scale", "regime_veto_cap",
    "regime_forced_bear_enabled", "drawdown_regime_enabled",
}


def test_scoring_weights_vocabulary():
    # weights = the SUB_SCORES the optimizer rebalances; envelope = the
    # payload literal in apply_weights.
    emitted = set(weight_optimizer.SUB_SCORES) | _literal_keys(
        _func_node(_REPO / "optimizer" / "weight_optimizer.py", "apply_weights"),
        "payload",
    )
    assert emitted == SCORING_WEIGHTS_DECLARED, (
        f"scoring_weights producer/contract drift: emitted-not-declared="
        f"{sorted(emitted - SCORING_WEIGHTS_DECLARED)} declared-not-emitted="
        f"{sorted(SCORING_WEIGHTS_DECLARED - emitted)} — update "
        f"PIPELINE_CONTRACT.yaml + the research consumer together."
    )


def test_research_params_vocabulary():
    emitted = set(research_optimizer.SAFE_PARAMS) | _literal_keys(
        _func_node(_REPO / "optimizer" / "research_optimizer.py", "apply"),
        "payload",
    )
    assert emitted == RESEARCH_PARAMS_DECLARED, (
        f"research_params producer/contract drift: emitted-not-declared="
        f"{sorted(emitted - RESEARCH_PARAMS_DECLARED)} declared-not-emitted="
        f"{sorted(RESEARCH_PARAMS_DECLARED - emitted)} — a key the research "
        f"config._RP_DEFAULTS filter doesn't know is SILENTLY DROPPED in "
        f"live scoring; update contract + consumer together."
    )


def test_predictor_params_vocabulary():
    veto = _literal_keys(
        _func_node(_REPO / "analysis" / "veto_analysis.py", "apply"), "payload",
    )
    phase4 = _literal_keys(
        _func_node(
            _REPO / "optimizer" / "predictor_optimizer.py",
            "apply_recommendations",
        ),
        "updates",
    )
    emitted = veto | phase4
    assert veto and phase4, "AST extraction found an empty writer leg — helper broke"
    undeclared = emitted - PREDICTOR_PARAMS_DECLARED
    orphans = PREDICTOR_PARAMS_DECLARED - emitted - _OPERATOR_ONLY
    assert not undeclared, (
        f"predictor_params writer(s) emit undeclared key(s) {sorted(undeclared)} "
        f"— declare in PIPELINE_CONTRACT.yaml AND confirm the predictor "
        f"write_output consumer handles them (it reads explicit keys only)."
    )
    assert not orphans, (
        f"PIPELINE_CONTRACT.yaml declares predictor_params key(s) with no "
        f"producer path: {sorted(orphans)} — stale contract, prune both sides."
    )


def test_dual_writers_merge_not_overwrite():
    # The two predictor_params writers MUST merge over the existing key —
    # a plain overwrite by one leg would silently erase the other leg's
    # tuned values (veto_confidence erased by a Phase 4 run or vice versa).
    src = (_REPO / "optimizer" / "predictor_optimizer.py").read_text()
    assert "{**existing, **updates}" in src, (
        "predictor_optimizer.apply_recommendations no longer read-merge-writes "
        "config/predictor_params.json — veto_analysis's veto_confidence would "
        "be silently erased on every Phase 4 apply."
    )


# ── Sole-writer guard for config/research_params.json (config#1719) ───────────
# The retired ``pipeline_optimizer.apply_cio_mode`` was a SECOND, undeclared
# writer into the shared config/research_params.json key — it wrote a
# ``cio_mode`` field NO consumer read (research ``config._RP_DEFAULTS`` drops
# it), a 63-day dead write. Guard the bug class structurally: research_params
# must have exactly ONE producer module. A new writer fails here until it is
# declared in PIPELINE_CONTRACT.yaml AND wired to a research-side consumer.

_RESEARCH_PARAMS_KEY = "config/research_params.json"
_ALLOWED_RESEARCH_PARAMS_WRITERS = {"research_optimizer"}


def _module_const_strings(tree: ast.Module) -> dict[str, str]:
    """Module-level ``NAME = "literal"`` string constants."""
    out: dict[str, str] = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    out[tgt.id] = node.value.value
    return out


def _put_object_keys(tree: ast.Module) -> set[str]:
    """Every ``Key=`` value passed to an ``s3.put_object(...)`` call, resolving
    module-level string-constant references. Ignores docstrings/comments — only
    ACTUAL write targets count (that is the whole point of the guard)."""
    consts = _module_const_strings(tree)
    keys: set[str] = set()
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "put_object"):
            for kw in n.keywords:
                if kw.arg != "Key":
                    continue
                v = kw.value
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    keys.add(v.value)
                elif isinstance(v, ast.Name) and v.id in consts:
                    keys.add(consts[v.id])
    return keys


def _modules_writing_key(key: str, subdirs=("optimizer", "analysis")) -> set[str]:
    writers: set[str] = set()
    for sub in subdirs:
        d = _REPO / sub
        if not d.exists():
            continue
        for path in d.glob("*.py"):
            if key in _put_object_keys(ast.parse(path.read_text())):
                writers.add(path.stem)
    return writers


def test_research_params_has_single_declared_writer():
    """config#1719 regression: only research_optimizer writes research_params.json.

    Catches the cio_mode bug class — a second module silently writing a field
    into the shared config key that no consumer reads.
    """
    writers = _modules_writing_key(_RESEARCH_PARAMS_KEY)
    assert writers == _ALLOWED_RESEARCH_PARAMS_WRITERS, (
        f"unexpected writer(s) into {_RESEARCH_PARAMS_KEY}: "
        f"{sorted(writers - _ALLOWED_RESEARCH_PARAMS_WRITERS)} — a new producer "
        f"of the shared research-params config must be declared in "
        f"PIPELINE_CONTRACT.yaml AND wired to a research-side consumer "
        f"(config#1719); an undeclared field is a silent dead write."
    )


def test_apply_cio_mode_retired():
    """config#1719: the dead cio_mode write path is gone (measurement retained)."""
    from optimizer import pipeline_optimizer
    assert not hasattr(pipeline_optimizer, "apply_cio_mode"), (
        "apply_cio_mode was retired (config#1719 — dead write, no consumer). Do "
        "not reintroduce a cio_mode actuation without a research-side consumer "
        "and a bidirectional gate (scoped under config#1060)."
    )
    assert hasattr(pipeline_optimizer, "analyze_cio_performance"), (
        "analyze_cio_performance (the retained CIO-vs-ranking measurement) must "
        "survive — its recommendation feeds the eval report + config#1060."
    )
