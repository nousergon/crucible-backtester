"""Pins the config-I3112 --eval-half vocabulary in infrastructure/spot_backtest.sh.

The weekly SF's single bundled Evaluator state is decomposed into two
sequential states (EvaluatorDiagnostics → EvaluatorOptimize) that reuse this
same spot box: the diagnostics half runs `--eval-half=diagnostics`
(evaluate.py --mode diagnostics) and the optimize half `--eval-half=optimize`
(--mode optimize, reading the S3 diagnostics snapshot the diagnostics half
wrote). This test pins:

  * the flag parses and validates against {all, diagnostics, optimize}
    (unknown values hard-fail per no-silent-fails — a typo must not silently
    run the FULL evaluator);
  * the value is baked into the spot heredoc (same mechanism as RUN_DATE) so
    the runtime evaluator-stage gate resolves it on the spot instance;
  * the evaluator stage invokes evaluate.py with --mode "${EVAL_HALF}".

Regex-based on the script text, mirroring test_spot_backtest_pit_parity_
stage_gate.py.
"""

from __future__ import annotations

import re
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "infrastructure" / "spot_backtest.sh"
)


def _read_script() -> str:
    return _SCRIPT.read_text()


def test_default_is_all():
    """Manual runs without the flag keep the bundled behavior — the
    invocation must be byte-identical to the pre-split script."""
    script = _read_script()
    m = re.search(r'EVAL_HALF="\$\{EVAL_HALF:-([^}]+)\}"', script)
    assert m, "EVAL_HALF default declaration not found"
    assert m.group(1) == "all", f"default must be 'all', got {m.group(1)!r}"


def test_flag_is_parsed_both_forms():
    script = _read_script()
    assert '--eval-half) EVAL_HALF="$2"; shift 2 ;;' in script
    assert '--eval-half=*) EVAL_HALF="${1#*=}"; shift ;;' in script


def test_known_vocabulary_validated():
    """The typo guard must accept exactly {all, diagnostics, optimize} and
    hard-fail anything else (no-silent-fails — a typo like
    --eval-half=diagnostic would otherwise silently run the full evaluator)."""
    script = _read_script()
    m = re.search(
        r'case "\$EVAL_HALF" in\s*\n\s*(all\|diagnostics\|optimize)\s*\)\s*;;',
        script,
    )
    assert m, "EVAL_HALF vocabulary case statement not found"
    assert m.group(1) == "all|diagnostics|optimize"


def test_unknown_value_hard_fails():
    script = _read_script()
    assert "ERROR: unknown --eval-half=" in script
    assert "exit 1" in script


def test_eval_half_baked_into_spot_heredoc():
    """Same bake mechanism as RUN_DATE/SKIP_STAGES — the dispatcher-side
    value is interpolated at heredoc-generation time so the runtime gate
    resolves it on the spot instance."""
    script = _read_script()
    assert 'EVAL_HALF="${EVAL_HALF}"' in script, (
        "EVAL_HALF must be baked into the spot heredoc next to RUN_DATE"
    )


def test_evaluator_stage_invokes_mode_eval_half():
    """The evaluator stage must pass evaluate.py's mode through from the
    baked EVAL_HALF value (--mode all keeps the bundled behavior)."""
    script = _read_script()
    m = re.search(
        r'evaluate\.py --mode "\\\$\{EVAL_HALF\}" --upload --date "\\\$\{RUN_DATE\}"',
        script,
    )
    assert m, (
        "evaluate.py must be invoked as --mode \"${EVAL_HALF}\" "
        "--upload --date \"${RUN_DATE}\""
    )
