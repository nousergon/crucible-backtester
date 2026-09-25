#!/usr/bin/env bash
# infrastructure/spot_parity_compare.sh — PitParityCompare stage: the JOIN of
# the split pit_parity passes (alpha-engine-config#6030). Reads
# parity/{run_date}/pit_stats_{lookahead,walkforward}.json, computes the
# contamination delta report (delta_pit_minus_current: ΔSortino/ΔPSR/ΔCVaR/
# Δmax-DD/PBO + materiality + analysis/parity_alarms.py), and writes
# backtest/{run_date}/pit_parity.json where today's consumers (evaluate.py
# Report Card pipeline_health, the freshness monitor) already read it.
#
# §2.3a join semantics: a missing/unparseable/non-ok pass artifact does NOT
# abort this stage — the compare still runs and emits the parity verdict as
# UNKNOWN (never pass). Uniform architecture: its own spot quartet like the
# sibling stages (no new Lambda — Lambda bootstrap is operator-gated).
# Siblings: spot_pit_lookahead.sh, spot_pit_walkforward.sh,
# spot_parity_replay.sh. spot_parity.sh (the bundled stage) remains the
# currently-wired SF path until the nousergon-data SF-cutover PR lands.
#
# Usage:
#   ./infrastructure/spot_parity_compare.sh
#   ./infrastructure/spot_parity_compare.sh --preflight-only
#   ./infrastructure/spot_parity_compare.sh --run-date 2026-08-08
#   ./infrastructure/spot_parity_compare.sh --help

set -euo pipefail
export HOME="${HOME:-/home/ec2-user}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPOT_STAGE_NAME="parity-compare"
# SF state name this script asserts against (config-I7214) — this script IS
# the PitParityCompare SF state, hardcoded 1:1, no flag-derived mapping
# needed.
_COVERAGE_STAGE="PitParityCompare"

# Per-stage runtime budget (sf-pipeline-policy §4): the compare itself is
# seconds of compute; the spot boot/deps (~15 min) dominates. Workload
# run_ssm timeout 1800 < watchdog 2400 < SF executionTimeout 2700 < SF
# state TimeoutSeconds 2760.
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-2400}"
COMPARE_WORKLOAD_TIMEOUT="${COMPARE_WORKLOAD_TIMEOUT:-1800}"

# shellcheck source=./_spot_common.sh
source "$SCRIPT_DIR/_spot_common.sh"
spot_common_init_defaults
_ORIG_ARGS=("$@")

usage() {
    cat <<'EOF'
spot_parity_compare.sh — pit_parity compare/join stage on a spot EC2
instance. Reads both pass artifacts (parity/{run_date}/pit_stats_*.json),
writes backtest/{run_date}/pit_parity.json. Missing pass artifact =>
verdict UNKNOWN, never pass (sf-pipeline-policy §2.3a).

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
echo "  PitParity Compare Spot Run (stage=pit_parity_compare) — $(spot_common_stage_run_date)"
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
# The compare never runs predictor_pipeline — soft-skip.
spot_common_resolve_predictor_config 0

# ── Stage-specific preflight (sf-pipeline-policy §2.2) ───────────────────────
# Deliberately does NOT require the pass artifacts to exist: §2.3a mandates
# the compare RUN when a pass artifact is absent (that is the UNKNOWN-verdict
# path, this stage's own contract). What must hold before spend is S3
# reachability of the parity prefix — an S3/permission outage would make even
# the UNKNOWN report unwritable, and THAT fails before a spot boots.
preflight_parity_compare() {
    echo "==> Stage preflight: pit_parity compare"
    if ! aws s3 ls "s3://${S3_BUCKET}/parity/" --region "$AWS_REGION" >/dev/null 2>&1; then
        # `aws s3 ls` on an empty-but-reachable prefix exits 1 with no
        # output; only treat a FAILING credentials/endpoint probe on the
        # bucket root as breakage.
        if ! aws s3 ls "s3://${S3_BUCKET}/" --region "$AWS_REGION" >/dev/null 2>&1; then
            echo "ERROR: s3://${S3_BUCKET}/ is unreachable — the compare could not read pass artifacts NOR write the UNKNOWN report. Failing before spend (§2.2)." >&2
            exit 1
        fi
        echo "  note: parity/ prefix empty or absent — compare will emit verdict UNKNOWN (§2.3a), which is a legal run."
    fi
    echo "  stage preflight OK (bucket reachable)."
}
preflight_parity_compare

echo "==> Dispatcher pre-launch preflight (fail-fast before provisioning spot)..."
spot_common_pre_launch_preflight \
    "$REPO_ROOT/backtest.py" \
    "$REPO_ROOT/preflight.py" \
    "$REPO_ROOT/pipeline_common.py" \
    "$REPO_ROOT/analysis/pit_stats_artifact.py" \
    "$REPO_ROOT/analysis/pit_parity.py" \
    "$REPO_ROOT/analysis/parity_alarms.py"

# No predictor RAM floor: the compare reads two small JSONs and computes
# numpy stats — the cheap 4-8 GB rotation is the right size (§4 right-sizing).
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

# ── Stage: pit_parity compare ────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  PIT PARITY COMPARE (join of the split passes)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

run_ssm "parity-compare" "$COMPARE_WORKLOAD_TIMEOUT" <<COMPARE
set -eo pipefail
cd /home/ec2-user/alpha-engine-backtester
${ENV_SOURCE}

RUN_DATE="${RUN_DATE}"

# Import smoke (config#3121 pattern, FATAL): proves the compare's import
# chain (pit_stats_artifact -> pit_parity -> parity_alarms -> pbo/dsr) in
# seconds before the real invocation. Deliberately NOT smoke-pit-parity —
# that mode runs the tiny predictor passes, which this stage never does.
echo "▶ stage=smoke-parity-compare START at \$(date -u +%H:%M:%S)"
$REMOTE_PYTHON -c "from analysis.pit_stats_artifact import run_compare_and_publish, load_pass_artifact, build_unknown_report; from analysis.parity_alarms import evaluate_parity_alarms; print('compare import chain OK')"
echo "▶ stage=smoke-parity-compare END at \$(date -u +%H:%M:%S)"

echo "▶ stage=pit_parity_compare START at \$(date -u +%H:%M:%S)"
# Exit 0 even on verdict UNKNOWN — emitting the honest UNKNOWN verdict IS
# this stage succeeding (§2.3a); the failed pass's own SF branch already
# carries the degraded flag. A crash (S3 unreachable, report unwritable)
# exits non-zero and the SF degrades THIS stage.
$REMOTE_PYTHON -u backtest.py --pit-parity-compare --date "\${RUN_DATE}" --log-level INFO 2>&1
echo "▶ stage=pit_parity_compare END at \$(date -u +%H:%M:%S)"

echo ""
echo "pit_parity compare complete at \$(date)"
COMPARE

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  pit_parity compare complete. Instance will be terminated."
echo "═══════════════════════════════════════════════════════════════"

# No CloudWatch heartbeat emitted here — see spot_predictor_backtest.sh's
# identical comment. Tracked: alpha-engine-config-I6710.

# Per-stage output assertion (config-I7214, sf-pipeline-policy.md §2.1):
# assert THIS stage wrote what it declared, at the boundary where the fact
# becomes knowable. OBSERVE MODE — it can never fail the stage.
"$LIB_PYTHON" -m krepis.stage_coverage assert --stage "$_COVERAGE_STAGE" --window-start "$_STAGE_WINDOW_START" --run-date "$EXECUTION_RUN_DATE" || echo "WARNING: stage-coverage assertion did not run for $_COVERAGE_STAGE (rc=$?) — observe mode, stage NOT failed (config-I7214)" >&2
