"""Pins spot_backtest.sh `--preflight-only` to the Friday shell_run dry-path
hard invariant: boot + deps + the EXISTING bootstrap-class smoke harness
(backtest.py --mode=smoke = BacktesterPreflight + _runtime_smoke), then
`exit 0` BEFORE the per-phase smoke modes, the evaluate.py S3-probe
diagnostics, AND the full-backtest heredoc — with NO param sweep, NO
portfolio sim, NO parity, NO evaluator, NO config/*.json optimizer
auto-apply, ZERO external API calls, and ZERO S3/config writes.

Owed-item #3 of ROADMAP "Friday shell-run — per-module dry-path
activation" (P1). Static-analysis test (mirrors
test_spot_backtest_aws_region.py) — the spot_backtest.sh SSM/EC2 path
cannot be exercised in CI; these assertions guard the structural
invariant against a future edit that would let preflight-only fall
through into the sweep / parity / evaluator / config-auto-apply.

Cross-script consistency: the flag name is `--preflight-only`, verbatim
identical to the data (spot_data_weekly.sh #259) and predictor
(spot_train.sh #175) siblings, because the Friday shell_run SF keystone
follow-on dispatches the same flag name to every module.
"""

from __future__ import annotations

from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "infrastructure" / "spot_backtest.sh"
)


def _text() -> str:
    return _SCRIPT.read_text()


def test_spot_backtest_exists():
    assert _SCRIPT.is_file()


def test_preflight_only_flag_parses():
    text = _text()
    assert "--preflight-only) PREFLIGHT_ONLY=1; shift ;;" in text, (
        "--preflight-only flag not wired into the flag parser"
    )


def test_preflight_only_is_an_orthogonal_modifier_default_off():
    """PREFLIGHT_ONLY is a MODIFIER (default 0), orthogonal to RUN_MODE —
    matching the data/predictor siblings. A default of 0 means a normal
    Saturday SF run (no flag) is completely unaffected."""
    text = _text()
    assert "PREFLIGHT_ONLY=0" in text, (
        "PREFLIGHT_ONLY must default to 0 so the unflagged Saturday run "
        "is unaffected"
    )


def test_preflight_only_branch_exists_and_exits_zero():
    text = _text()
    assert 'if [ "$PREFLIGHT_ONLY" = "1" ]; then' in text, (
        "no dedicated preflight-only branch found"
    )
    # The branch must terminate with `exit 0` (clean dispatcher exit;
    # trap cleanup still terminates the spot instance).
    branch = text.split('if [ "$PREFLIGHT_ONLY" = "1" ]; then', 1)[1]
    branch = branch.split("# ── Smoke test", 1)[0]
    assert "exit 0" in branch, "preflight-only branch must exit 0"


def test_preflight_only_runs_before_smoke_only_block_and_full_backtest():
    """The exit 0 must short-circuit BEFORE both the --smoke-only body
    (which runs the heavy per-phase smoke modes + evaluate.py S3-probe
    diagnostics) and the full-backtest heredoc (sweep/parity/evaluator/
    config auto-apply)."""
    text = _text()
    i_branch = text.index('if [ "$PREFLIGHT_ONLY" = "1" ]; then')
    i_smoke_only = text.index('if [ "$RUN_MODE" = "smoke-only" ]; then')
    i_full = text.index("# ── Full backtest")
    assert i_branch < i_smoke_only < i_full, (
        "preflight-only branch must precede the smoke-only block and the "
        "full-backtest heredoc so its exit 0 short-circuits before any "
        "sweep / parity / evaluator / config auto-apply"
    )


def test_preflight_only_body_only_runs_mode_smoke_and_no_writers():
    """The preflight-only SSM heredoc must invoke ONLY backtest.py
    --mode=smoke (the read-only bootstrap harness) and must NOT reference
    any sweep / sim / parity / evaluator / --upload / optimizer
    auto-apply token — those are the param-sweep, portfolio-sim, and
    config/*.json S3 writers."""
    text = _text()
    start = text.index('if [ "$PREFLIGHT_ONLY" = "1" ]; then')
    # The preflight heredoc payload ends at its terminator.
    end = text.index("\nPREFLIGHT\n", start)
    payload = text[start:end]

    # Strip comment + echo lines so the human-readable proof text
    # ("NO sweep / sim / parity") does not false-positive against the
    # forbidden-token scan.
    code_lines = [
        ln
        for ln in payload.splitlines()
        if not ln.lstrip().startswith(("#", "echo "))
    ]
    code = "\n".join(code_lines)

    assert "backtest.py --mode=smoke" in code, (
        "preflight-only must reuse the existing backtest.py --mode=smoke "
        "bootstrap harness (do not rebuild a parallel preflight)"
    )

    forbidden = [
        "--mode $BACKTEST_MODE",      # full backtest
        "--mode smoke-",              # per-phase heavy smoke modes
        "--mode=smoke-",
        "evaluate.py",                # evaluator + S3-probe diagnostics
        "--upload",                   # the optimizer config/*.json auto-apply
        "--pit-parity",               # pit_parity extra predictor sim
        "param_sweep",
        "test_parity_replay",         # parity stage
        "put-metric-data",            # CloudWatch heartbeat
        "aws s3 cp",                  # any S3 upload
        "aws s3 sync",
    ]
    for token in forbidden:
        assert token not in code, (
            f"preflight-only body must NOT reference {token!r} — it would "
            f"break the no-sweep/no-parity/no-evaluator/no-auto-apply/"
            f"no-write invariant"
        )


def test_preflight_only_step_keeps_aws_region_export():
    """Same #247 regression guard as test_spot_backtest_aws_region.py —
    the preflight heredoc sources ENV_SOURCE, which must export
    AWS_REGION/AWS_DEFAULT_REGION (BacktesterPreflight + boto3 require
    it; no .env post-deprecation)."""
    text = _text()
    start = text.index('if [ "$PREFLIGHT_ONLY" = "1" ]; then')
    end = text.index("\nPREFLIGHT\n", start)
    payload = text[start:end]
    assert "${ENV_SOURCE}" in payload, (
        "preflight-only heredoc must source ${ENV_SOURCE} so AWS_REGION / "
        "AWS_DEFAULT_REGION (and the .env runtime config) are exported — "
        "without it BacktesterPreflight / boto3 fail (#247 class)."
    )
