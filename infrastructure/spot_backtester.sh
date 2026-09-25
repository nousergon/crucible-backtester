#!/usr/bin/env bash
# infrastructure/spot_backtester.sh — Backtester stage (simulation-only:
# 10y simulate + param sweep) on a spot EC2 instance.
#
# alpha-engine-config-I4442: one of five scripts split out of the
# spot_backtest.sh monolith, mapped 1:1 to the Saturday SF's Backtester
# state. That state currently calls (nousergon-data infrastructure/step_function.json):
#
#   spot_backtest.sh --mode=param-sweep --no-pit-parity \
#       --skip-stages=parity,evaluator{preflight_args}
#
# This script reproduces that exact semantics as its ONLY behavior — no
# --mode / --skip-stages / --pit-parity-enabled flags exist here, because
# this script IS the backtest stage. `{preflight_args}` becomes
# --preflight-only below (the only value the SF ever substitutes there).
#
# This PR is additive: the SF is NOT yet wired to this script (that is a
# separate nousergon-data PR, sequenced after this one lands). spot_backtest.sh
# stays the currently-wired path, unchanged.
#
# Usage:
#   ./infrastructure/spot_backtester.sh
#   ./infrastructure/spot_backtester.sh --preflight-only   # Friday shell_run dry path
#   ./infrastructure/spot_backtester.sh --run-date 2026-08-08
#   ./infrastructure/spot_backtester.sh --instance-type c5.xlarge
#   ./infrastructure/spot_backtester.sh --dry-run
#   ./infrastructure/spot_backtester.sh --help

set -euo pipefail
export HOME="${HOME:-/home/ec2-user}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPOT_STAGE_NAME="backtester"
# SF state name this script asserts against (config-I7214) — this script IS
# the Backtester SF state, hardcoded 1:1, no flag-derived mapping needed.
_COVERAGE_STAGE="Backtester"

# shellcheck source=./_spot_common.sh
source "$SCRIPT_DIR/_spot_common.sh"
spot_common_init_defaults
_ORIG_ARGS=("$@")  # captured pre-parse for the #883 relaunch exec

usage() {
    cat <<'EOF'
spot_backtester.sh — Backtester stage (simulation-only: 10y simulate + param
sweep) on a spot EC2 instance. Equivalent to:
  spot_backtest.sh --mode=param-sweep --no-pit-parity --skip-stages=parity,evaluator

Flags:
  --preflight-only        Boot + deps + smoke harness only, exit 0, zero spend
  --run-date DATE         Override RUN_DATE (default: today, normalized to NYSE trading day)
  --branch BRANCH         Git branch the spot clones (default: main)
  --instance-type TYPE    Override instance-type rotation
  --dry-run               Full-universe exercise without production S3 writes
  --skip-phases LIST      Comma-separated phase names to skip
  --only-phases LIST      Comma-separated phase names to run exclusively
  --force / --force-phases LIST
  --skip-phase4-evaluations
  --use-vectorized-sweep
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
echo "  Backtester Spot Run (stage=backtest, mode=param-sweep) — $(spot_common_stage_run_date)"
echo "═══════════════════════════════════════════════════════════════"
echo "  Branch        : $BRANCH"
echo "  Preflight-only: $PREFLIGHT_ONLY"
echo "  Dry-run       : $DRY_RUN"
echo "  Spot attempt  : $SPOT_ATTEMPT/$MAX_SPOT_ATTEMPTS"
echo ""

spot_common_normalize_run_date

# ── Preflight checks ─────────────────────────────────────────────────────────
if [ "$PREFLIGHT_ONLY" != "1" ] && [ ! -f "$REPO_ROOT/config.yaml" ]; then
    echo "ERROR: config.yaml not found — copy from config.yaml.example"
    exit 1
fi
spot_common_resolve_executor_config
# param-sweep mode does not load the predictor tensor — predictor.yaml is
# optional here (soft-skip, matching the monolith's uniform behavior for
# this mode).
spot_common_resolve_predictor_config 0

# ── Stage-specific preflight (weekly-sf-policy.md §2.2) ──────────────────────
# Backtester is the FIRST stage in the chain — it has no upstream S3
# artifact to assert. Assert the substrate it DOES depend on: the output
# bucket is reachable before we pay for a 60-100 min spot boot+sim.
preflight_backtester() {
    echo "==> Stage preflight: backtester"
    if ! aws s3 ls "s3://${S3_BUCKET}/" --region "$AWS_REGION" >/dev/null 2>&1; then
        echo "ERROR: S3 bucket s3://${S3_BUCKET}/ is not reachable — cannot upload backtest artifacts. Failing before spend." >&2
        exit 1
    fi
    echo "  stage preflight OK (S3 bucket reachable)."
}
preflight_backtester

echo "==> Dispatcher pre-launch preflight (fail-fast before provisioning spot)..."
spot_common_pre_launch_preflight \
    "$REPO_ROOT/backtest.py" \
    "$REPO_ROOT/preflight.py" \
    "$REPO_ROOT/pipeline_common.py"

# alpha-engine-config-I7216 (2026-08-13): param-sweep DOES need a RAM floor.
#
# This comment used to argue that because param-sweep does not run
# predictor_pipeline, it could stay on the cheap default rotation and skip the
# floor. The premise is about the predictor tensor and is true; the conclusion
# does not follow, and was falsified in production:
#
#   bash: line 16: 26748 Killed  python -u backtest.py --mode param-sweep ...
#
# A kernel OOM kill on a 4 GB c5.large (2026-08-13, instance
# i-0eba8344f3a124b3c). param-sweep reads the same ArcticDB feature store over
# ~900 tickers; not loading the GBM tensor does not make it cheap.
#
# The blast radius is not one backtest. THIS stage gates PredictorBacktest,
# which writes predictor/research_free_backfill/predictor_outcomes_research_free
# .parquet — the LIVE ENTRY FEED while scanner_predictor_direct is champion.
# With this stage dying the cohort froze at prediction_date 2026-08-07 and every
# trading day drew its entry candidates from that stale pool; distinct names
# newly entered fell from ~20/month to 3.
#
# It also failed INTERMITTENTLY, which is why it read as flakiness: the default
# rotation (_spot_common.sh) is c5.large,m5.large,c6i.large,c5a.large — 4 GB,
# 8 GB, 4 GB, 4 GB. For a memory-bound job the effective budget became a
# function of which spot capacity pool answered.
#
# Reusing the predictor floor rather than inventing a smaller one is an
# asymmetry call, not a capacity estimate: an OOM-killed process reports no
# peak, so the true requirement is UNKNOWN. Guessing low costs another failed
# nightly run and another day of trading on a frozen cohort; guessing high
# costs a few cents of spot. Right-sizing DOWN from a measured peak RSS on a
# surviving run is tracked on alpha-engine-config-I7216 — a cap derived from
# another cap is not a budget.
spot_common_apply_large_universe_ram_floor
spot_common_collapse_instance_type
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

# ── Stage: backtest (mode=param-sweep) ───────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  BACKTESTER (--mode param-sweep)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

run_ssm "backtest" "$MAX_RUNTIME_SECONDS" <<BACKTEST
set -eo pipefail
cd /home/ec2-user/alpha-engine-backtester
${ENV_SOURCE}

RUN_DATE="${RUN_DATE}"

# If backtest.py fails we exit non-zero so the run is marked FAILED — fail
# loud so the heartbeat metric is not emitted and the Step Function catches
# it (replaces the pre-2026 || WARNING swallow that let evaluator run
# against invalid sweep results and was the root cause of multiple
# undetected param oscillations).
echo "▶ stage=backtest START at \$(date -u +%H:%M:%S)"
_STAGE_TAIL_LOG="\$(mktemp /tmp/backtester-stage-XXXXXX.log)"
_BT_RC=0
$REMOTE_PYTHON -u backtest.py --mode param-sweep --date "\${RUN_DATE}" --upload --log-level INFO $BACKTEST_SKIP_PHASE4_FLAG $BACKTEST_PHASE_FLAGS 2>&1 | tee "\$_STAGE_TAIL_LOG" || _BT_RC=\$?
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
    echo "--- tail of backtester stage output (the real cause; stdout is truncated by SSM at 24 KB) ---" >&2
    tail -n 60 "\$_STAGE_TAIL_LOG" 2>/dev/null >&2 || echo "(no stage log captured)" >&2
    echo "--- end of backtester stage tail ---" >&2
    exit "\$_BT_RC"
fi
echo "▶ stage=backtest END at \$(date -u +%H:%M:%S)"

echo ""
echo "Backtester stage complete at \$(date)"
BACKTEST

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Backtester complete. Instance will be terminated."
echo "═══════════════════════════════════════════════════════════════"

spot_common_emit_heartbeat backtester

# Per-stage output assertion (config-I7214, sf-pipeline-policy.md §2.1):
# assert THIS stage wrote what it declared, at the boundary where the fact
# becomes knowable. OBSERVE MODE — it can never fail the stage.
"$LIB_PYTHON" -m krepis.stage_coverage assert --stage "$_COVERAGE_STAGE" --window-start "$_STAGE_WINDOW_START" --run-date "$EXECUTION_RUN_DATE" || echo "WARNING: stage-coverage assertion did not run for $_COVERAGE_STAGE (rc=$?) — observe mode, stage NOT failed (config-I7214)" >&2
