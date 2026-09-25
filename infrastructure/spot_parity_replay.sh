#!/usr/bin/env bash
# infrastructure/spot_parity_replay.sh — ParityReplay stage: the
# backtester<->executor parity replay (10-day-window integration test
# against live trades_latest.db) on its own spot EC2 instance.
#
# alpha-engine-config#6030 (sf-pipeline-policy §2.1 — atomic): the weekly
# SF's Parity state bundled the replay with the two pit_parity passes; the
# replay consumes NEITHER pass's output and neither pass consumes the
# replay's, so all three died together for no reason. This script IS the
# replay alone. Siblings: spot_pit_lookahead.sh, spot_pit_walkforward.sh,
# spot_parity_compare.sh. spot_parity.sh (the bundled stage) remains the
# currently-wired SF path until the nousergon-data SF-cutover PR lands;
# this script deploys inert until then.
#
# Usage:
#   ./infrastructure/spot_parity_replay.sh
#   ./infrastructure/spot_parity_replay.sh --preflight-only
#   ./infrastructure/spot_parity_replay.sh --run-date 2026-08-08
#   ./infrastructure/spot_parity_replay.sh --help

set -euo pipefail
export HOME="${HOME:-/home/ec2-user}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPOT_STAGE_NAME="parity-replay"
# SF state name this script asserts against (config-I7214) — this script IS
# the ParityReplay SF state, hardcoded 1:1, no flag-derived mapping needed.
_COVERAGE_STAGE="ParityReplay"

# Per-stage runtime budget (sf-pipeline-policy §4): healthy replay ≈ 2.5 min
# measured (sf_budgets.py calibration, SSM logs 2026-07-03/07-11) + ~15 min
# spot boot/deps. Workload run_ssm timeout 2100 < watchdog 2400 < SF
# executionTimeout 2700 < SF state TimeoutSeconds 2760.
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-2400}"
REPLAY_WORKLOAD_TIMEOUT="${REPLAY_WORKLOAD_TIMEOUT:-2100}"

# shellcheck source=./_spot_common.sh
source "$SCRIPT_DIR/_spot_common.sh"
spot_common_init_defaults
_ORIG_ARGS=("$@")

usage() {
    cat <<'EOF'
spot_parity_replay.sh — backtester<->executor parity replay on a spot EC2
instance. Produces backtest/{run_date}/parity_report.json + appends
backtest/parity_metrics.csv. Independent of the pit_parity passes and the
compare join (alpha-engine-config#6030).

Depends on trades/trades_latest.db existing in S3 — the stage preflight
asserts this before spending on a spot (the 2026-07-25 incident class).

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
echo "  Parity Replay Spot Run (stage=parity_replay) — $(spot_common_stage_run_date)"
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
# The replay does not run predictor_pipeline — predictor.yaml stays a
# soft-skip (the executor-sim path never loads it).
spot_common_resolve_predictor_config 0

# ── Stage-specific preflight (sf-pipeline-policy §2.2) ───────────────────────
# THE motivating incident for I4442: on 2026-07-25 the Parity SF state
# failed 37 minutes in because trades_latest.db didn't exist in S3.
preflight_parity_replay() {
    echo "==> Stage preflight: parity replay"
    if ! spot_common_s3_key_exists "trades/trades_latest.db"; then
        echo "ERROR: s3://${S3_BUCKET}/trades/trades_latest.db does not exist or is unreachable." >&2
        echo "       The parity replay cannot run without it. Failing before spend — this is the exact" >&2
        echo "       class of miss that cost 37 minutes of spot boot on 2026-07-25 (alpha-engine-config-I4442)." >&2
        exit 1
    fi
    echo "  stage preflight OK (trades/trades_latest.db reachable)."
}
preflight_parity_replay

echo "==> Dispatcher pre-launch preflight (fail-fast before provisioning spot)..."
spot_common_pre_launch_preflight \
    "$REPO_ROOT/backtest.py" \
    "$REPO_ROOT/preflight.py" \
    "$REPO_ROOT/pipeline_common.py"

# No predictor RAM floor: the replay never loads the GBM tensor — the cheap
# 4-8 GB rotation is the right size (sf-pipeline-policy §4 right-sizing).
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

# ── Stage: parity replay ─────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  PARITY REPLAY (backtester<->executor, observability not a gate)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

run_ssm "parity-replay" "$REPLAY_WORKLOAD_TIMEOUT" <<REPLAY
set -eo pipefail
cd /home/ec2-user/alpha-engine-backtester
${ENV_SOURCE}

RUN_DATE="${RUN_DATE}"
BUCKET="\$(\$PYTHON_BIN -c 'import yaml,sys; print((yaml.safe_load(open(\"config.yaml\")) or {}).get(\"output_bucket\") or \"alpha-engine-research\")' 2>/dev/null || echo alpha-engine-research)"

# Import/collect smoke (config#3121 pattern). FATAL here — unlike the
# bundled spot_parity.sh, the replay is this stage's whole product, so a
# collect failure (broken import) fails fast instead of warning past the
# same failure two minutes later (§2.2 fail-fast).
echo "▶ stage=smoke-parity START at \$(date -u +%H:%M:%S)"
$REMOTE_PYTHON -m pytest tests/test_parity_replay.py -m parity --collect-only -q 2>&1
echo "▶ stage=smoke-parity END at \$(date -u +%H:%M:%S)"

# Each run produces parity_report.json (per-run drill-down) +
# parity_metrics.csv (time-series append). The run does NOT fail the SF on
# parity DIVERGENCE — 0% historical parity is structurally unreachable for
# a system with weekly auto-tuned configs and evolving executor code (see
# tests/test_parity_replay.py module docstring). Setup-level failures
# (missing trades.db, ArcticDB unreachable) ARE fatal — real infrastructure
# breakage, not "expected drift".
echo "▶ stage=parity_replay START at \$(date -u +%H:%M:%S)"
PARITY_TRADES_DB="/tmp/trades_latest.db"
PARITY_REPORT_DIR="/tmp/parity_report"
mkdir -p "\$PARITY_REPORT_DIR"

if ! aws s3 cp "s3://\${BUCKET}/trades/trades_latest.db" "\$PARITY_TRADES_DB" --quiet; then
    echo "ERROR: could not download trades_latest.db from S3 — parity replay cannot run" >&2
    echo "       This is infrastructure breakage (not divergence) — failing spot." >&2
    exit 1
fi

PARITY_EXIT=0
TRADES_DB_PATH="\$PARITY_TRADES_DB" \\
SIGNALS_BUCKET="\${BUCKET}" \\
PARITY_REPORT_DIR="\$PARITY_REPORT_DIR" \\
PARITY_RUN_DATE="\${RUN_DATE}" \\
USE_REAL_ARCTICDB=1 \\
$REMOTE_PYTHON -m pytest tests/test_parity_replay.py -m parity -v 2>&1 || PARITY_EXIT=\$?

if [ -f "\$PARITY_REPORT_DIR/parity_report.json" ]; then
    aws s3 cp "\$PARITY_REPORT_DIR/parity_report.json" \\
        "s3://\${BUCKET}/backtest/\${RUN_DATE}/parity_report.json" --quiet \\
        && echo "Uploaded parity_report.json to s3://\${BUCKET}/backtest/\${RUN_DATE}/" \\
        || echo "WARNING: failed to upload parity_report.json (non-fatal)"
else
    # alpha-engine-config#7199 — THE ABSENCE IS THE DEFECT. Until now a missing
    # parity_report.json was indistinguishable from a skipped upload: the guard
    # had no else-branch, the non-zero pytest exit degraded to a WARNING on
    # stderr, and the stage returned 0. That is how parity_report.json went
    # unwritten from 2026-07-24 to 2026-08-13 while the sibling report.md in the
    # same prefix kept updating weekly and nothing noticed. (Root cause: PR #550
    # stopped co-installing the predictor requirements, which is where pytest
    # came from; "No module named pytest", exit in 1 second, SF success path.)
    #
    # An absent artifact now becomes a PRESENT artifact that says it is absent,
    # so every downstream reader renders FAILED rather than a silently stale
    # ABSENT row. sf-pipeline-policy 2.3a rule 2: a missing verdict is never a
    # pass.
    echo "ERROR: the parity replay produced NO parity_report.json (producer exit=\$PARITY_EXIT)." >&2
    echo "       The backtester-to-executor fill-parity claim is UNPROVEN for \${RUN_DATE}." >&2
    printf '{"schema":"parity_report-0.0.0","run_date":"%s","status":"failed","verdict":"FAIL","verdict_reason":"the parity replay produced no report this run (producer exit=%s) - the backtester-to-executor fill-parity claim is unproven","producer_exit":%s}\\n' \\
        "\${RUN_DATE}" "\$PARITY_EXIT" "\$PARITY_EXIT" > /tmp/parity_report_missing.json
    aws s3 cp /tmp/parity_report_missing.json \\
        "s3://\${BUCKET}/backtest/\${RUN_DATE}/parity_report.json" --quiet \\
        && echo "       Recorded the absence at s3://\${BUCKET}/backtest/\${RUN_DATE}/parity_report.json" \\
        || echo "       WARNING: could not record the absence marker either." >&2
    # This branch is fail-open at the SF layer: every failure path converges on
    # ParityReplayDegraded and the sibling parity branches are untouched
    # (sf-pipeline-policy 2.1 blast radius). So failing here costs the branch
    # and nothing else, and the artifact IS this stage's product.
    exit 1
fi

if [ "\$PARITY_EXIT" != "0" ]; then
    echo "WARNING: parity pytest exited \$PARITY_EXIT (likely setup-side error)." >&2
    echo "         See s3://\${BUCKET}/backtest/\${RUN_DATE}/parity_report.json (if present)." >&2
    echo "         Continuing spot run — parity divergence is observability, not a gate." >&2
fi
echo "▶ stage=parity_replay END at \$(date -u +%H:%M:%S)"

echo ""
echo "Parity replay complete at \$(date)"
REPLAY

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Parity replay complete. Instance will be terminated."
echo "═══════════════════════════════════════════════════════════════"

# No CloudWatch heartbeat emitted here — see spot_predictor_backtest.sh's
# identical comment. Tracked: alpha-engine-config-I6710.

# Per-stage output assertion (config-I7214, sf-pipeline-policy.md §2.1):
# assert THIS stage wrote what it declared, at the boundary where the fact
# becomes knowable. OBSERVE MODE — it can never fail the stage.
"$LIB_PYTHON" -m krepis.stage_coverage assert --stage "$_COVERAGE_STAGE" --window-start "$_STAGE_WINDOW_START" --run-date "$EXECUTION_RUN_DATE" || echo "WARNING: stage-coverage assertion did not run for $_COVERAGE_STAGE (rc=$?) — observe mode, stage NOT failed (config-I7214)" >&2
