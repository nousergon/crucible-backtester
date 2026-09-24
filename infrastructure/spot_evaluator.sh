#!/usr/bin/env bash
# infrastructure/spot_evaluator.sh — Evaluator stage (evaluate.py against
# today's backtest artifacts: per-signal grading email + optimizer
# config/*.json auto-apply) on a spot EC2 instance.
#
# alpha-engine-config-I4442: one of five scripts split out of the
# spot_backtest.sh monolith, mapped 1:1 to the Saturday SF's Evaluator
# state, which currently calls:
#
#   spot_backtest.sh --no-pit-parity --skip-stages=backtest,parity{preflight_args}
#
# Reads s3://{bucket}/backtest/{RUN_DATE}/ artifacts produced by the prior
# Backtester-family states and runs evaluate.py --mode all --upload. Since
# backtest is ALWAYS skipped in this invocation (that stage now lives in
# spot_backtester.sh / spot_predictor_backtest.sh /
# spot_portfolio_optimizer_backtest.sh), --skip-backtester is always passed
# to evaluate.py (config#2887 — distinguishes intentional absence from an
# unexpected failure).
#
# This PR does not wire the SF to this script (separate nousergon-data PR,
# sequenced after); spot_backtest.sh stays the currently-wired path,
# unchanged.
#
# Usage:
#   ./infrastructure/spot_evaluator.sh
#   ./infrastructure/spot_evaluator.sh --preflight-only
#   ./infrastructure/spot_evaluator.sh --freeze-evaluator   # diagnostic-only, no config writes
#   ./infrastructure/spot_evaluator.sh --run-date 2026-08-08
#   ./infrastructure/spot_evaluator.sh --help

set -euo pipefail
export HOME="${HOME:-/home/ec2-user}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPOT_STAGE_NAME="evaluator"

# shellcheck source=./_spot_common.sh
source "$SCRIPT_DIR/_spot_common.sh"
spot_common_init_defaults
_ORIG_ARGS=("$@")

usage() {
    cat <<'EOF'
spot_evaluator.sh — Evaluator stage (evaluate.py: per-signal grading email +
optimizer config/*.json auto-apply) on a spot EC2 instance. Equivalent to:
  spot_backtest.sh --no-pit-parity --skip-stages=backtest,parity

Reads backtest/{RUN_DATE}/ artifacts from the prior backtest-family stages.
Missing artifacts degrade gracefully (evaluate.py logs a WARNING and
continues) — the stage preflight below warns, non-blocking, matching that
existing posture.

Flags:
  --preflight-only        Boot + deps + smoke harness only, exit 0, zero spend
  --run-date DATE         Override RUN_DATE (default: today, normalized to NYSE trading day)
  --branch BRANCH         Git branch the spot clones (default: main)
  --instance-type TYPE    Override instance-type rotation
  --freeze-evaluator       Diagnostic-only: suppress optimizer config writes
  --eval-half {all|diagnostics|optimize}  (config-I3112; SF still passes none -> "all")
  --help                  Print this and exit 0 (no AWS calls made)
EOF
}

spot_common_parse_flags "$@"
if [ "$SHOW_HELP" = "1" ]; then
    usage
    exit 0
fi

# SF state name this script asserts against (config-I7214). Unlike every
# other split launcher, this ONE script serves TWO SF states
# (EvaluatorDiagnostics / EvaluatorOptimize) distinguished only by the
# --eval-half flag the SF sends — so the stage name is derived from that
# flag, not hardcoded, or one half's assertion would file under the
# other's name. --eval-half=all (the pre-I3112 default, manual/ad-hoc runs
# only — no live SF state ever sends it) maps to no single SF state, so
# the assertion below is skipped for that invocation.
case "$EVAL_HALF" in
    diagnostics) _COVERAGE_STAGE="EvaluatorDiagnostics" ;;
    optimize)    _COVERAGE_STAGE="EvaluatorOptimize" ;;
    *)           _COVERAGE_STAGE="" ;;
esac

echo "═══════════════════════════════════════════════════════════════"
echo "  Evaluator Spot Run (stage=evaluator, eval-half=$EVAL_HALF) — $(spot_common_stage_run_date)"
echo "═══════════════════════════════════════════════════════════════"
echo "  Branch        : $BRANCH"
echo "  Preflight-only: $PREFLIGHT_ONLY"
echo "  Freeze eval   : $FREEZE_EVALUATOR"
echo "  Spot attempt  : $SPOT_ATTEMPT/$MAX_SPOT_ATTEMPTS"
echo ""

spot_common_normalize_run_date

# ── Preflight checks ─────────────────────────────────────────────────────────
if [ "$PREFLIGHT_ONLY" != "1" ] && [ ! -f "$REPO_ROOT/config.yaml" ]; then
    echo "ERROR: config.yaml not found — copy from config.yaml.example"
    exit 1
fi
spot_common_resolve_executor_config
# evaluate.py does not run predictor_pipeline directly — predictor.yaml
# stays a soft-skip, matching the monolith's uniform behavior.
spot_common_resolve_predictor_config 0

# ── Stage-specific preflight (weekly-sf-policy.md §2.2) ──────────────────────
# Evaluator reads backtest/{RUN_DATE}/ artifacts, but evaluate.py already
# degrades gracefully on missing input (logs a WARNING and continues —
# verified against the 2026-05-07 v1 validation run). This preflight
# mirrors that posture: WARN loudly, non-blocking, rather than inventing a
# stricter contract this stage has never actually enforced.
preflight_evaluator() {
    echo "==> Stage preflight: evaluator"
    if ! spot_common_s3_prefix_nonempty "backtest/${RUN_DATE}/"; then
        echo "WARNING: s3://${S3_BUCKET}/backtest/${RUN_DATE}/ is empty or unreachable." >&2
        echo "         evaluate.py will run without same-cohort backtest artifacts (existing graceful-degrade behavior) — continuing." >&2
    else
        echo "  stage preflight OK (backtest/${RUN_DATE}/ has upstream artifacts)."
    fi
}
preflight_evaluator

echo "==> Dispatcher pre-launch preflight (fail-fast before provisioning spot)..."
spot_common_pre_launch_preflight \
    "$REPO_ROOT/evaluate.py" \
    "$REPO_ROOT/preflight.py" \
    "$REPO_ROOT/pipeline_common.py"

# evaluate.py materialises the full ArcticDB universe before the diagnostics
# phase, so it needs the floor. The premise this comment used to carry — "does
# not run predictor_pipeline, so the cheap rotation is fine" — was falsified in
# production on 2026-08-13: the evaluator-diagnostics stage was OOM-killed
# (SIGKILL, `26756 Killed`) on a 4 GB c5.large immediately after logging
# `Load complete in 163.0s: 921 price tickers, 903 feature tickers`.
# It is the universe read that costs the memory, not the GBM tensor.
spot_common_apply_large_universe_ram_floor
spot_common_collapse_instance_type
echo "  Instance types: $INSTANCE_TYPES"

spot_common_launch_instance
spot_common_install_cleanup_trap

echo "==> Waiting for instance to enter running state..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$AWS_REGION"

spot_common_stage_configs
spot_common_wait_for_ssm_agent
spot_common_bootstrap
spot_common_install_deps
spot_common_fetch_predictor_cache
spot_common_build_env_source
spot_common_run_preflight_only_and_maybe_exit

# ── Stage: evaluator ──────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  EVALUATOR (--mode $EVAL_HALF)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

run_ssm "evaluator" "$MAX_RUNTIME_SECONDS" <<EVALUATOR
set -eo pipefail
cd /home/ec2-user/alpha-engine-backtester
${ENV_SOURCE}

RUN_DATE="${RUN_DATE}"
EVAL_HALF="${EVAL_HALF}"

# config#3121: smoke-evaluator runs immediately before the real evaluator
# pass, in the same invocation that executes it. Cheap preflight (imports +
# config load + S3 reachability) via evaluate.py --smoke. Non-fatal — loud
# WARNING, spot run continues; the real pass below surfaces the same break
# if it's real.
# alpha-engine-config-I11475: --date is this cycle's RUN_DATE. Without it
# evaluate.py defaulted --date to the box's UTC day, and the 2026-09-23
# rehearsal's smoke read backtest/2026-09-24/attestation.json (NoSuchKey).
echo "▶ stage=smoke-evaluator START at \$(date -u +%H:%M:%S)"
if ! $REMOTE_PYTHON -u evaluate.py --smoke --date "\${RUN_DATE}" --log-level INFO 2>&1; then
    echo "WARNING: smoke-evaluator FAILED — evaluate.py's import/config/S3 wiring is broken. Continuing but the real pass below will likely fail the same way." >&2
fi
echo "▶ stage=smoke-evaluator END at \$(date -u +%H:%M:%S)"

echo "▶ stage=evaluator START at \$(date -u +%H:%M:%S) freeze=${FREEZE_EVALUATOR}"
_EVAL_FREEZE=""
if [ "${FREEZE_EVALUATOR}" = "true" ]; then
    _EVAL_FREEZE="--freeze"
fi
# --skip-backtester is ALWAYS passed: this script IS the standalone
# Evaluator SF state, which always skips the backtest stage (that stage now
# lives in the three separate spot_*backtest*.sh scripts). Distinguishes
# intentional absence from an unexpected failure (config#2887).
_STAGE_TAIL_LOG="\$(mktemp /tmp/evaluator-stage-XXXXXX.log)"
_EVAL_RC=0
$REMOTE_PYTHON -u evaluate.py --mode "\${EVAL_HALF}" --upload --date "\${RUN_DATE}" \$_EVAL_FREEZE --skip-backtester --log-level INFO 2>&1 | tee "\$_STAGE_TAIL_LOG" || _EVAL_RC=\$?
if [ "\$_EVAL_RC" -ne 0 ]; then
    # config-I7258: preserve the REAL exit code — a bare \`exit 1\` here
    # laundered rc=137 (OOM SIGKILL) into a generic 1 before ssm_dispatcher
    # ever got to read it, which is why an OOM kill reached the operator as
    # an indistinguishable "evaluate.py failed". \$_EVAL_RC is what SSM's
    # get_command_invocation ResponseCode reports back, and krepis.ssm_dispatcher
    # (>=0.60.0) classifies 137/-9 as RESOURCE KILL (OOM) and 124/-14/143 as
    # RESOURCE KILL (TIMEOUT) directly from that field.
    case "\$_EVAL_RC" in
        137|-9) echo "ERROR: evaluate.py SIGKILLed (rc=\$_EVAL_RC) — likely OOM on \$(hostname) (\$(curl -s -m 2 http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo unknown-instance-type)). Spot run marked FAILED." >&2 ;;
        124|-14|143) echo "ERROR: evaluate.py timed out (rc=\$_EVAL_RC) on \$(hostname). Spot run marked FAILED." >&2 ;;
        *) echo "ERROR: evaluate.py failed (rc=\$_EVAL_RC). Spot run marked FAILED." >&2 ;;
    esac
    # config-I7396 / config-I7399: SSM's GetCommandInvocation returns only the
    # FIRST 24 KB of stdout, and this stage's stdout is thousands of INFO
    # lines, so a traceback at the END of the run is structurally unreachable
    # there -- no amount of bounding the chatter helps, because truncation is
    # from the front. STDERR comes back intact and is what the Step Function
    # copies verbatim into its failure cause and its alert, so the real cause
    # is republished here. Without this the operator reads
    # "evaluate.py failed (rc=1)" and nothing else, which is exactly what
    # happened on 2026-08-15 while the traceback sat unread in S3.
    echo "--- tail of evaluator stage output (the real cause; stdout is truncated by SSM at 24 KB) ---" >&2
    tail -n 60 "\$_STAGE_TAIL_LOG" 2>/dev/null >&2 || echo "(no stage log captured)" >&2
    echo "--- end of evaluator stage tail ---" >&2
    exit "\$_EVAL_RC"
fi
echo "▶ stage=evaluator END at \$(date -u +%H:%M:%S)"

echo ""
echo "Evaluator stage complete at \$(date)"
EVALUATOR

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Evaluator complete. Instance will be terminated."
echo "═══════════════════════════════════════════════════════════════"

spot_common_emit_heartbeat evaluator

# Per-stage output assertion (config-I7214, sf-pipeline-policy.md §2.1):
# assert THIS stage wrote what it declared, at the boundary where the fact
# becomes knowable. OBSERVE MODE — it can never fail the stage. Skipped for
# --eval-half=all (no single SF state name to assert against — see the
# _COVERAGE_STAGE derivation above).
if [ -n "$_COVERAGE_STAGE" ]; then
    "$LIB_PYTHON" -m krepis.stage_coverage assert --stage "$_COVERAGE_STAGE" --window-start "$_STAGE_WINDOW_START" --run-date "$EXECUTION_RUN_DATE" || echo "WARNING: stage-coverage assertion did not run for $_COVERAGE_STAGE (rc=$?) — observe mode, stage NOT failed (config-I7214)" >&2
else
    echo "NOTE: stage-coverage assertion skipped — --eval-half=all is not a single SF state (config-I7214)." >&2
fi
