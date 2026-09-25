#!/usr/bin/env bash
# infrastructure/spot_portfolio_optimizer_backtest.sh — Portfolio-optimizer
# stage (portfolio_optimizer_gate + cov_estimator_sweep + gamma_sweep) on a
# spot EC2 instance.
#
# alpha-engine-config-I4442: one of five scripts split out of the
# spot_backtest.sh monolith, mapped 1:1 to the Saturday SF's
# PortfolioOptimizerBacktest state, which currently calls:
#
#   spot_backtest.sh --mode=portfolio-optimizer-backtest --no-pit-parity \
#       --skip-stages=parity,evaluator{preflight_args}
#
# Runs AFTER the PredictorBacktest state. Each of its three phases re-runs
# the predictor pipeline (GBM inference + 10y ArcticDB read) internally.
# Reads/writes the same backtest/{RUN_DATE}/ prefix. This PR does not wire
# the SF to this script (separate nousergon-data PR, sequenced after);
# spot_backtest.sh stays the currently-wired path, unchanged.
#
# Usage:
#   ./infrastructure/spot_portfolio_optimizer_backtest.sh
#   ./infrastructure/spot_portfolio_optimizer_backtest.sh --preflight-only
#   ./infrastructure/spot_portfolio_optimizer_backtest.sh --run-date 2026-08-08
#   ./infrastructure/spot_portfolio_optimizer_backtest.sh --help

set -euo pipefail
export HOME="${HOME:-/home/ec2-user}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPOT_STAGE_NAME="portfolio-optimizer"
# SF state name this script asserts against (config-I7214) — this script IS
# the PortfolioOptimizerBacktest SF state, hardcoded 1:1, no flag-derived
# mapping needed.
_COVERAGE_STAGE="PortfolioOptimizerBacktest"

# shellcheck source=./_spot_common.sh
source "$SCRIPT_DIR/_spot_common.sh"
spot_common_init_defaults
_ORIG_ARGS=("$@")

usage() {
    cat <<'EOF'
spot_portfolio_optimizer_backtest.sh — Portfolio-optimizer stage
(portfolio_optimizer_gate + cov_estimator_sweep + gamma_sweep) on a spot EC2
instance. Equivalent to:
  spot_backtest.sh --mode=portfolio-optimizer-backtest --no-pit-parity --skip-stages=parity,evaluator

Depends on the PredictorBacktest stage's backtest/{RUN_DATE}/ predictor
sweep output already existing in S3 — the stage preflight asserts this
before spending on a spot.

Flags:
  --preflight-only        Boot + deps + smoke harness only, exit 0, zero spend
  --run-date DATE         Override RUN_DATE (default: today, normalized to NYSE trading day)
  --branch BRANCH         Git branch the spot clones (default: main)
  --instance-type TYPE    Override instance-type rotation (default: >=16GB RAM floor)
  --dry-run
  --skip-phases LIST / --only-phases LIST / --force / --force-phases LIST
  --skip-phase4-evaluations / --use-vectorized-sweep
  --help                  Print this and exit 0 (no AWS calls made)
EOF
}

spot_common_parse_flags "$@"
if [ "$SHOW_HELP" = "1" ]; then
    usage
    exit 0
fi
spot_common_compute_phase_flags

echo "═══════════════════════════════════════════════════════════════"
echo "  Portfolio-Optimizer-Backtest Spot Run (stage=backtest, mode=portfolio-optimizer-backtest) — $(spot_common_stage_run_date)"
echo "═══════════════════════════════════════════════════════════════"
echo "  Branch        : $BRANCH"
echo "  Preflight-only: $PREFLIGHT_ONLY"
echo "  Spot attempt  : $SPOT_ATTEMPT/$MAX_SPOT_ATTEMPTS"
echo ""

spot_common_normalize_run_date

# ── Preflight checks ─────────────────────────────────────────────────────────
if [ "$PREFLIGHT_ONLY" != "1" ] && [ ! -f "$REPO_ROOT/config.yaml" ]; then
    echo "ERROR: config.yaml not found — copy from config.yaml.example"
    exit 1
fi
spot_common_resolve_executor_config
# This stage re-runs predictor_pipeline internally in each of its three
# phases — predictor.yaml is REQUIRED, same reasoning as
# spot_predictor_backtest.sh.
spot_common_resolve_predictor_config 1

# ── Stage-specific preflight (weekly-sf-policy.md §2.2) ──────────────────────
# PortfolioOptimizerBacktest runs after PredictorBacktest — it reads the
# predictor sweep artifacts that stage just wrote. Assert that upstream
# output exists BEFORE provisioning a spot.
preflight_portfolio_optimizer_backtest() {
    echo "==> Stage preflight: portfolio-optimizer-backtest"
    if ! spot_common_s3_prefix_nonempty "backtest/${RUN_DATE}/"; then
        echo "ERROR: s3://${S3_BUCKET}/backtest/${RUN_DATE}/ is empty or unreachable." >&2
        echo "       This stage reads the PredictorBacktest stage's predictor-sweep output from that prefix — it must run and succeed first." >&2
        echo "       Failing before spend rather than discovering this mid-run." >&2
        exit 1
    fi
    echo "  stage preflight OK (backtest/${RUN_DATE}/ has upstream PredictorBacktest output)."
}
preflight_portfolio_optimizer_backtest

echo "==> Dispatcher pre-launch preflight (fail-fast before provisioning spot)..."
spot_common_pre_launch_preflight \
    "$REPO_ROOT/backtest.py" \
    "$REPO_ROOT/preflight.py" \
    "$REPO_ROOT/pipeline_common.py" \
    "$REPO_ROOT/synthetic/predictor_backtest.py"

# This mode re-runs predictor_pipeline internally — apply the >=16 GB
# instance floor (I3280) unless the operator passed an explicit --instance-type.
spot_common_apply_large_universe_ram_floor
echo "  Instance types: $INSTANCE_TYPES"

spot_common_launch_instance   # arms the cleanup EXIT trap before it launches (alpha-engine-config-I11573)

echo "==> Waiting for instance to enter running state..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$AWS_REGION"

spot_common_stage_configs
spot_common_wait_for_ssm_agent
spot_common_bootstrap
spot_common_install_deps
spot_common_fetch_predictor_cache
spot_common_build_env_source
spot_common_run_preflight_only_and_maybe_exit

# ── Stage: backtest (mode=portfolio-optimizer-backtest) ─────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  PORTFOLIO-OPTIMIZER-BACKTEST (--mode portfolio-optimizer-backtest)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

run_ssm "portfolio-optimizer" "$MAX_RUNTIME_SECONDS" <<BACKTEST
set -eo pipefail
cd /home/ec2-user/alpha-engine-backtester
${ENV_SOURCE}

RUN_DATE="${RUN_DATE}"

echo "▶ stage=backtest START at \$(date -u +%H:%M:%S)"
_STAGE_TAIL_LOG="\$(mktemp /tmp/portfolio-optimizer-backtest-stage-XXXXXX.log)"
_BT_RC=0
$REMOTE_PYTHON -u backtest.py --mode portfolio-optimizer-backtest --date "\${RUN_DATE}" --upload --log-level INFO $BACKTEST_SKIP_PHASE4_FLAG $BACKTEST_PHASE_FLAGS 2>&1 | tee "\$_STAGE_TAIL_LOG" || _BT_RC=\$?
if [ "\$_BT_RC" -ne 0 ]; then
    # config-I7258: preserve the REAL exit code (a bare \`exit 1\` here
    # launders rc=137/OOM into a generic 1 before krepis.ssm_dispatcher
    # (>=0.60.0) can classify it via SSM's ResponseCode).
    case "\$_BT_RC" in
        137|-9) echo "ERROR: backtest.py SIGKILLed (rc=\$_BT_RC) — likely OOM on \$(hostname) (\$(curl -s -m 2 http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo unknown-instance-type)). Spot run marked FAILED — check flow-doctor alerts." >&2 ;;
        124|-14|143) echo "ERROR: backtest.py timed out (rc=\$_BT_RC) on \$(hostname). Spot run marked FAILED — check flow-doctor alerts." >&2 ;;
        *) echo "ERROR: backtest.py failed (rc=\$_BT_RC). Spot run marked FAILED — check flow-doctor alerts." >&2 ;;
    esac
    # config-I7396 / config-I7399: SSM's GetCommandInvocation returns only the
    # FIRST 24 KB of stdout, and this stage's stdout is thousands of INFO
    # lines, so a traceback at the END of the run is structurally unreachable
    # there -- no amount of bounding the chatter helps, because truncation is
    # from the front. STDERR comes back intact and is what the Step Function
    # copies verbatim into its failure cause and its alert, so the real cause
    # is republished here. Without this the operator reads
    # "backtest.py failed (rc=1)" and nothing else, which is exactly what
    # happened on 2026-08-15 while the traceback sat unread in S3.
    echo "--- tail of portfolio-optimizer-backtest stage output (the real cause; stdout is truncated by SSM at 24 KB) ---" >&2
    tail -n 60 "\$_STAGE_TAIL_LOG" 2>/dev/null >&2 || echo "(no stage log captured)" >&2
    echo "--- end of portfolio-optimizer-backtest stage tail ---" >&2
    exit "\$_BT_RC"
fi
echo "▶ stage=backtest END at \$(date -u +%H:%M:%S)"

echo ""
echo "Portfolio-optimizer-backtest stage complete at \$(date)"
BACKTEST

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Portfolio-optimizer-backtest complete. Instance will be terminated."
echo "═══════════════════════════════════════════════════════════════"

# No CloudWatch heartbeat emitted here — see spot_predictor_backtest.sh's
# identical comment. Tracked: alpha-engine-config-I6710.

# Per-stage output assertion (config-I7214, sf-pipeline-policy.md §2.1):
# assert THIS stage wrote what it declared, at the boundary where the fact
# becomes knowable. OBSERVE MODE — it can never fail the stage.
"$LIB_PYTHON" -m krepis.stage_coverage assert --stage "$_COVERAGE_STAGE" --window-start "$_STAGE_WINDOW_START" --run-date "$EXECUTION_RUN_DATE" || echo "WARNING: stage-coverage assertion did not run for $_COVERAGE_STAGE (rc=$?) — observe mode, stage NOT failed (config-I7214)" >&2
