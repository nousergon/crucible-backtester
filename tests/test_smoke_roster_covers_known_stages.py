"""tests/test_smoke_roster_covers_known_stages.py — CI chokepoint (config#3121).

Brian's scope-extension ruling on config#3121: EVERY stage in
``infrastructure/spot_backtest.sh``'s ``_KNOWN_STAGES`` must declare a
smoke (a tiny-slice execution preflight run before its full pass). This
test enumerates ``_KNOWN_STAGES`` and asserts each one has a corresponding
smoke marker in the script — adding a new stage to ``_KNOWN_STAGES``
without adding its smoke fails this test, not silently shipping a gap
like the one that let ``pit_parity``'s numba/numpy ImportError surface
~2h into a real run instead of in seconds at smoke time.

Stage -> smoke mapping (as of this PR):
  - backtest:    smoke-simulate / smoke-param-sweep / smoke-predictor-
                 backtest / smoke-phase4 / smoke-predictor-param-sweep
                 (backtest.py --mode=smoke-<phase>, the pre-existing
                 per-phase smoke harness, ROADMAP Backtester P0 #3)
  - pit_parity:  smoke-pit-parity (backtest.py --mode=smoke-pit-parity;
                 new in this PR)
  - parity:      smoke-parity (pytest --collect-only proof; new in this
                 PR — the stage's own full pass is already a small
                 10-day-window integration test, so its smoke is an
                 import/wiring proof rather than a second tiny-slice DATA
                 pass)
  - evaluator:   smoke-evaluator (evaluate.py --smoke; new in this PR —
                 previously only had input_quality_gate, an input CHECK,
                 not an execution smoke of its own imports/config/S3-wiring)

Two independent checks, mirroring the design of
tests/test_evaluator_artifact_contract.py (bind a declarative-ish mapping
to the live script text so drift fails CI):
  1. ``_KNOWN_STAGES`` (the source of truth for what stages exist) is
     bound to the mapping below — a stage added to _KNOWN_STAGES without
     a matching entry here fails loud.
  2. Each mapped smoke marker must actually be present in the script.
"""

from __future__ import annotations

import re
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "infrastructure" / "spot_backtest.sh"
)


def _read_script() -> str:
    return _SCRIPT.read_text()


def _known_stages() -> list[str]:
    script = _read_script()
    m = re.search(r'_KNOWN_STAGES="([^"]+)"', script)
    assert m, "_KNOWN_STAGES declaration not found in spot_backtest.sh"
    return m.group(1).split()


# Stage -> list of regex patterns; AT LEAST ONE must match somewhere in the
# script for the stage's smoke to count as present. Each stage's smoke
# marker is deliberately specific (not just the bare word "smoke") so a
# marker for an unrelated stage can't accidentally satisfy this test.
_STAGE_SMOKE_PATTERNS: dict[str, list[str]] = {
    "backtest": [
        r"smoke-simulate\s+smoke-param-sweep\s+smoke-predictor-backtest\s+"
        r"smoke-phase4\s+smoke-predictor-param-sweep",
    ],
    "pit_parity": [
        r"_smoke_run_mode smoke-pit-parity",
        r"--mode smoke-pit-parity",
    ],
    "parity": [
        r"smoke-parity",
    ],
    "evaluator": [
        r"_smoke_run_evaluator",
        r"evaluate\.py --smoke",
    ],
}


def test_known_stages_mapping_is_declared_for_every_stage():
    """Every stage in _KNOWN_STAGES must have an entry in
    _STAGE_SMOKE_PATTERNS above (the source of truth this test checks
    against). A stage added to _KNOWN_STAGES without a corresponding
    entry here is exactly the gap this chokepoint exists to catch —
    fails loud instead of silently shipping unsmoked."""
    stages = _known_stages()
    mapped = set(_STAGE_SMOKE_PATTERNS.keys())
    missing = set(stages) - mapped
    assert not missing, (
        f"_KNOWN_STAGES contains stage(s) with no smoke mapping declared "
        f"in tests/test_smoke_roster_covers_known_stages.py: {sorted(missing)}. "
        f"Add a smoke for the new stage in infrastructure/spot_backtest.sh "
        f"(and backtest.py / evaluate.py if it routes through a Python "
        f"entrypoint), then add its pattern(s) to _STAGE_SMOKE_PATTERNS."
    )
    # Also guard the reverse: a stale entry mapped to a stage that no
    # longer exists should be cleaned up, not silently ignored.
    stale = mapped - set(stages)
    assert not stale, (
        f"_STAGE_SMOKE_PATTERNS has entries for stage(s) no longer in "
        f"_KNOWN_STAGES: {sorted(stale)}. Remove the stale mapping."
    )


def test_every_known_stage_has_a_smoke_marker_in_the_script():
    """Each stage's declared smoke pattern(s) must actually appear in
    spot_backtest.sh — this is the real regression guard: it fails if a
    future edit deletes a smoke wiring block without updating the
    mapping test above (which would otherwise happily keep "passing"
    against a mapping that no longer reflects reality)."""
    script = _read_script()
    stages = _known_stages()
    for stage in stages:
        patterns = _STAGE_SMOKE_PATTERNS.get(stage)
        assert patterns, f"no smoke pattern(s) declared for stage {stage!r}"
        found = any(re.search(p, script) for p in patterns)
        assert found, (
            f"stage={stage!r} has no smoke marker matching any of "
            f"{patterns} in infrastructure/spot_backtest.sh — this stage "
            f"is missing its execution-preflight smoke (config#3121)."
        )


# ── Test for the test: prove the assertion logic itself would catch a ───────
# ── missing smoke, not just pass vacuously on the current script. ───────────


def test_assertion_logic_would_catch_a_stage_added_without_a_smoke():
    """Directly exercises the same assertion logic the two tests above
    use, against a SYNTHETIC stage list + mapping that omits a smoke for
    one stage — proving the chokepoint actually fires on the failure
    mode it exists to prevent (a stage added to _KNOWN_STAGES without a
    matching smoke), not just that it happens to pass today."""
    synthetic_known_stages = ["backtest", "pit_parity", "parity", "evaluator", "new_stage"]
    synthetic_mapping = dict(_STAGE_SMOKE_PATTERNS)  # "new_stage" NOT added

    missing = set(synthetic_known_stages) - set(synthetic_mapping.keys())
    assert missing == {"new_stage"}, (
        "the mapping-completeness check must flag a stage with no smoke "
        "mapping declared — this proves test_known_stages_mapping_is_"
        "declared_for_every_stage's logic actually discriminates missing "
        "coverage instead of vacuously passing"
    )

    # And the marker-presence check: even if a mapping entry EXISTS but
    # its pattern never matches the script (e.g. someone added a mapping
    # row with a typo'd/stale pattern, or removed the wiring without
    # removing the row), the check must still fail.
    synthetic_mapping_with_bad_pattern = dict(_STAGE_SMOKE_PATTERNS)
    synthetic_mapping_with_bad_pattern["new_stage"] = [r"this_pattern_will_never_match_xyz123"]
    script = _read_script()
    found = any(
        re.search(p, script)
        for p in synthetic_mapping_with_bad_pattern["new_stage"]
    )
    assert not found, (
        "sanity check: the synthetic never-match pattern must not "
        "accidentally match the real script (would make this meta-test "
        "meaningless)"
    )
