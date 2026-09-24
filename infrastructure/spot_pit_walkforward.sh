#!/usr/bin/env bash
# infrastructure/spot_pit_walkforward.sh — PitParityWalkforward stage: run the
# pit_parity WALKFORWARD predictor pass on its own spot EC2 instance and
# publish its stats artifact to
# s3://{bucket}/parity/{run_date}/pit_stats_walkforward.json
# (contracts/pit_stats_pass.schema.json).
#
# alpha-engine-config#6030 (sf-pipeline-policy §2.1 — atomic): the weekly
# SF's Parity state bundled THREE logical stages (pit_parity lookahead pass
# + walkforward pass + parity replay) behind one script. This script IS one
# stage: exactly one pass, one artifact, one timeout, one failure route.
# The sibling stages: spot_pit_lookahead.sh, spot_parity_replay.sh,
# spot_parity_compare.sh (the join). spot_parity.sh (the bundled
# pit_parity+replay stage) remains the currently-wired SF path until the
# nousergon-data SF-cutover PR lands; this script deploys inert until then.
#
# Failure posture: this script FAILS LOUD (non-zero) when the pass or the
# artifact upload fails — the SF branch absorbs it fail-open (branch
# degraded marker, siblings unaffected) and the PitParityCompare stage
# emits verdict UNKNOWN (§2.3a).
#
# Usage:
#   ./infrastructure/spot_pit_walkforward.sh
#   ./infrastructure/spot_pit_walkforward.sh --preflight-only
#   ./infrastructure/spot_pit_walkforward.sh --run-date 2026-08-08
#   ./infrastructure/spot_pit_walkforward.sh --help

set -euo pipefail
export HOME="${HOME:-/home/ec2-user}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPOT_STAGE_NAME="pit-walkforward"
PIT_PASS="walkforward"
# SF state name this script asserts against (config-I7214) — this script IS
# the PitParityWalkforward SF state, hardcoded 1:1, no flag-derived mapping
# needed.
_COVERAGE_STAGE="PitParityWalkforward"

# Per-stage runtime budget (sf-pipeline-policy §4 — sized to THIS stage, not
# the old bundle): healthy walkforward pass ≈ 10 min measured
# (sf_budgets.py calibration, SSM logs 2026-07-03/07-11) + ~15 min spot
# boot/deps. Workload run_ssm timeout 3600 < watchdog 4200 < SF
# executionTimeout 5400 < SF state TimeoutSeconds 5460 (the DataPhase2
# budget-chain shape; the watchdog covers boot+deps+workload).
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-4200}"
PASS_WORKLOAD_TIMEOUT="${PASS_WORKLOAD_TIMEOUT:-3600}"

# shellcheck source=./_spot_common.sh
source "$SCRIPT_DIR/_spot_common.sh"
spot_common_init_defaults
_ORIG_ARGS=("$@")

usage() {
    cat <<EOF
spot_pit_${PIT_PASS}.sh — pit_parity ${PIT_PASS} pass on a spot EC2 instance.
Publishes parity/{run_date}/pit_stats_${PIT_PASS}.json (schema
pit_stats_pass-1.0.0) for the PitParityCompare join stage.

Flags:
  --preflight-only        Boot + deps + smoke harness only, exit 0, zero spend
  --run-date DATE         Override RUN_DATE (default: today, normalized to NYSE trading day)
  --branch BRANCH         Git branch the spot clones (default: main)
  --instance-type TYPE    Override instance-type rotation
  --help                  Print this and exit 0 (no AWS calls made)
EOF
}

spot_common_parse_flags "$@"
if [ "$SHOW_HELP" = "1" ]; then
    usage
    exit 0
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  PitParity ${PIT_PASS} Spot Run — $(spot_common_stage_run_date)"
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
# The pass RUNS predictor_pipeline — predictor.yaml is a hard requirement
# here (required=1), unlike the bundled spot_parity.sh's soft-skip: a pass
# with no predictor config can never publish real stats, so failing before
# spend is strictly better than a spot boot that degrades 15 minutes in.
spot_common_resolve_predictor_config 1

# ── Stage-specific preflight (sf-pipeline-policy §2.2) ───────────────────────
# Assert THIS stage's actual inputs before provisioning a spot: the Layer-1A
# GBM weights the predictor pass loads (the on-spot backtest.py preflight
# re-asserts them, but that is ~15 min of boot+deps later).
preflight_pit_pass() {
    echo "==> Stage preflight: pit_parity ${PIT_PASS} pass"
    if ! spot_common_s3_key_exists "predictor/weights/meta/momentum_model.txt"; then
        echo "ERROR: s3://${S3_BUCKET}/predictor/weights/meta/momentum_model.txt does not exist or is unreachable." >&2
        echo "       The ${PIT_PASS} pass cannot run GBM inference without the Layer-1A weights. Failing before spend (§2.2)." >&2
        exit 1
    fi
    echo "  stage preflight OK (predictor weights reachable)."
}
preflight_pit_pass

echo "==> Dispatcher pre-launch preflight (fail-fast before provisioning spot)..."
spot_common_pre_launch_preflight \
    "$REPO_ROOT/backtest.py" \
    "$REPO_ROOT/preflight.py" \
    "$REPO_ROOT/pipeline_common.py" \
    "$REPO_ROOT/analysis/pit_stats_artifact.py" \
    "$REPO_ROOT/synthetic/predictor_backtest.py"

# The pass runs predictor_pipeline — universal >=16 GB floor (I3280).
spot_common_apply_large_universe_ram_floor
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

# ── Stage: pit_parity ${PIT_PASS} pass ───────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  PIT PARITY ${PIT_PASS} PASS (stage=pit_${PIT_PASS})"
echo "═══════════════════════════════════════════════════════════════"
echo ""

run_ssm "pit-${PIT_PASS}" "$PASS_WORKLOAD_TIMEOUT" <<PASS
set -eo pipefail
cd /home/ec2-user/alpha-engine-backtester
${ENV_SOURCE}

RUN_DATE="${RUN_DATE}"

# Tiny-slice wiring smoke (config#3121 pattern): proves the import/config/
# subprocess chain in ~1 min before the real pass. FATAL here — this stage's
# whole product is the pass artifact, so a broken wiring should fail at
# minute ~16, not ~30 (§2.2 fail-fast; contrast spot_parity.sh where
# pit_parity was a non-blocking co-tenant and the smoke only warned).
echo "▶ stage=smoke-pit-parity START at \$(date -u +%H:%M:%S)"
$REMOTE_PYTHON -u backtest.py --mode smoke-pit-parity --date "\${RUN_DATE}" --log-level INFO 2>&1
echo "▶ stage=smoke-pit-parity END at \$(date -u +%H:%M:%S)"

echo "▶ stage=pit_${PIT_PASS} START at \$(date -u +%H:%M:%S)"
# Fail-loud: non-zero exit marks this SSM command Failed; the SF branch
# records DEGRADED and the compare emits verdict UNKNOWN. The always-emit
# contract inside publish_pass_artifact still writes a status=failed
# artifact + pages a warning before exiting non-zero.
#
# config#6032: the PredictorBacktest phase (earlier in this SF, same
# RUN_DATE, its own stage/box) already ran the SAME walk-forward (PIT)
# inference over the same config and wrote
# backtest/{RUN_DATE}/predictor_stats.json — bake that key in so
# publish_pass_artifact can reuse it and skip this pass's own
# full-predictor-pipeline subprocess (~25 min saved). No-op for the
# lookahead sibling script (predictor-stats-key is ignored there — the
# lookahead pass forces legacy single-pass mode no phase artifact
# reproduces). Best-effort: a missing/unreadable artifact falls back to
# the subprocess inside publish_pass_artifact (never fails this stage).
$REMOTE_PYTHON -u backtest.py --pit-parity-pass-publish ${PIT_PASS} \\
    --predictor-stats-key "backtest/\${RUN_DATE}/predictor_stats.json" \\
    --date "\${RUN_DATE}" --log-level INFO 2>&1
echo "▶ stage=pit_${PIT_PASS} END at \$(date -u +%H:%M:%S)"

echo ""
echo "pit_parity ${PIT_PASS} pass complete at \$(date)"
PASS

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  pit_parity ${PIT_PASS} pass complete. Instance will be terminated."
echo "═══════════════════════════════════════════════════════════════"

# No CloudWatch heartbeat emitted here — see spot_predictor_backtest.sh's
# identical comment. Tracked: alpha-engine-config-I6710.

# Per-stage output assertion (config-I7214, sf-pipeline-policy.md §2.1):
# assert THIS stage wrote what it declared, at the boundary where the fact
# becomes knowable. OBSERVE MODE — it can never fail the stage.
"$LIB_PYTHON" -m krepis.stage_coverage assert --stage "$_COVERAGE_STAGE" --window-start "$_STAGE_WINDOW_START" --run-date "$EXECUTION_RUN_DATE" || echo "WARNING: stage-coverage assertion did not run for $_COVERAGE_STAGE (rc=$?) — observe mode, stage NOT failed (config-I7214)" >&2
