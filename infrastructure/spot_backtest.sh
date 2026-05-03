#!/usr/bin/env bash
# infrastructure/spot_backtest.sh — Run weekly backtest on a spot EC2 instance.
#
# Launches a c5.large spot instance (~$0.03/hr), clones the backtester +
# predictor + executor repos, runs the full backtest pipeline with 10y of
# price data, uploads results to S3, and self-terminates.
#
# Usage:
#   ./infrastructure/spot_backtest.sh                   # full run (--mode all)
#   ./infrastructure/spot_backtest.sh --smoke-only      # quick validation, then terminate
#   ./infrastructure/spot_backtest.sh --mode simulate   # override backtest mode
#   ./infrastructure/spot_backtest.sh --instance-type c5.xlarge  # override instance type
#   ./infrastructure/spot_backtest.sh --dry-run         # full-universe exercise without
#                                                       #   production S3 pollution:
#                                                       #   markers + artifacts + reports
#                                                       #   go to .dry-run/{date}/, no
#                                                       #   optimizer config writes, no
#                                                       #   reporter upload. Safe to run
#                                                       #   concurrently with scheduled SF.
#   ./infrastructure/spot_backtest.sh --use-vectorized-sweep  # run predictor_param_sweep
#                                                       #   through the matrix-axis vectorized
#                                                       #   engine (Tier 4). Default off until
#                                                       #   v14 spot validation confirms parity.
#
# Prerequisites:
#   - AWS CLI configured (uses alpha-engine-executor-profile for S3/SES access)
#   - SSH key at ~/.ssh/alpha-engine-key.pem
#   - Code committed and pushed to origin (instance clones from GitHub)
#   - .env file with EMAIL_SENDER, EMAIL_RECIPIENTS, GMAIL_APP_PASSWORD
#   - config.yaml (gitignored — SCP'd to EC2 by this script)
#
# For scheduled weekly runs, call this script from the always-on EC2 cron
# or from an EventBridge → Lambda trigger:
#
#   0 8 * * 1  cd ~/alpha-engine-backtester && bash infrastructure/spot_backtest.sh >> /var/log/backtester-spot.log 2>&1

set -euo pipefail

# ── Ensure HOME is set (SSM RunCommand does not set it) ──────────────────────
export HOME="${HOME:-/home/ec2-user}"

# ── Load .env ────────────────────────────────────────────────────────────────
# Master .env lives in alpha-engine-data; fall back to ~/.alpha-engine.env
# (Step Functions SSM), then local .env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ENV_FILE="$(dirname "$REPO_ROOT")/alpha-engine-data/.env"
if [ ! -f "$ENV_FILE" ]; then
    ENV_FILE="$HOME/.alpha-engine.env"
fi
if [ ! -f "$ENV_FILE" ]; then
    ENV_FILE="$REPO_ROOT/.env"
fi
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    echo "Loaded .env from $ENV_FILE"
else
    echo "WARNING: No .env file found"
fi

# ── Configuration ──────────────────────────────────────────────────────────────
AWS_REGION="${AWS_REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET:-alpha-engine-research}"
BRANCH="${BRANCH:-main}"
INSTANCE_TYPE="c5.large"            # 2 vCPU, 4GB RAM — keep tight; larger instance
                                    # would enable sloppy memory usage. The 2026-04-23
                                    # predictor_data_prep OOM is being fixed structurally
                                    # via the ohlcv_by_ticker → DataFrame refactor
                                    # (P2 in SYSTEM_STATE backtester section).
AMI_ID="ami-0c421724a94bba6d6"      # Amazon Linux 2023 x86_64
# Spot-side watchdog budget: backtester's 10y simulate + param sweep
# historically runs 60-100 min. 120 min with headroom. Bump (don't
# silently rely on the orphan reaper) if a run legitimately needs more.
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-7200}"
KEY_NAME="alpha-engine-key"
KEY_FILE="$HOME/.ssh/alpha-engine-key.pem"
SECURITY_GROUP="sg-03cd3c4bd91e610b0"
SUBNET_ID="subnet-e07166ec"
IAM_PROFILE="alpha-engine-executor-profile"
BACKTEST_MODE="all"

# ── Parse flags ──────────────────────────────────────────────────────────────
RUN_MODE="full"  # full | smoke-only
# All PhaseRegistry-adjacent flags are also routable from the
# Saturday SF input via env vars. When set they pass through as
# CLI args to backtest.py.
SKIP_PHASE4="${SKIP_PHASE4_EVALUATIONS:-false}"
SKIP_PHASES="${SKIP_PHASES:-}"            # comma-separated phase names
ONLY_PHASES="${ONLY_PHASES:-}"            # comma-separated phase names
FORCE_ALL="${FORCE_ALL:-false}"           # true → --force
FORCE_PHASES="${FORCE_PHASES:-}"          # comma-separated phase names
DRY_RUN="${DRY_RUN:-false}"               # true → --dry-run
# Pipeline-level stage control: comma-separated subset of {backtest, parity,
# evaluator}. All three stages run by default on the spot. Used for fast
# iteration against a single stage (e.g. parity-only when debugging a cred
# divergence).
SKIP_STAGES="${SKIP_STAGES:-}"
# Freeze the evaluator (passes --freeze to evaluate.py → suppresses per-
# optimizer S3 config writes; report artifacts + email still upload). Use
# for off-cycle test runs so mid-week sweeps don't auto-promote weights/
# params/thresholds against Monday trading. Replaces the retired SF
# CheckEvaluatorFreeze Choice state (evaluator consolidated into spot
# 2026-04-24); the freeze_evaluator SF input param is no longer honored.
FREEZE_EVALUATOR="${FREEZE_EVALUATOR:-false}"
USE_VECTORIZED_SWEEP="${USE_VECTORIZED_SWEEP:-false}"
# Accept both --flag value and --flag=value forms for every value-taking
# flag. The equals form is GNU-getopt-style muscle memory and it's cheap to
# support — each value flag gets a companion `--foo=*` case that splits on
# `=`. Boolean flags (--smoke-only, --force, --dry-run, etc.) accept no
# value and don't need the companion case.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-only) RUN_MODE="smoke-only"; shift ;;
        --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
        --instance-type=*) INSTANCE_TYPE="${1#*=}"; shift ;;
        --mode) BACKTEST_MODE="$2"; shift 2 ;;
        --mode=*) BACKTEST_MODE="${1#*=}"; shift ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --branch=*) BRANCH="${1#*=}"; shift ;;
        --skip-phase4-evaluations) SKIP_PHASE4="true"; shift ;;
        --skip-phases) SKIP_PHASES="$2"; shift 2 ;;
        --skip-phases=*) SKIP_PHASES="${1#*=}"; shift ;;
        --only-phases) ONLY_PHASES="$2"; shift 2 ;;
        --only-phases=*) ONLY_PHASES="${1#*=}"; shift ;;
        --force) FORCE_ALL="true"; shift ;;
        --force-phases) FORCE_PHASES="$2"; shift 2 ;;
        --force-phases=*) FORCE_PHASES="${1#*=}"; shift ;;
        --dry-run) DRY_RUN="true"; shift ;;
        --skip-stages) SKIP_STAGES="$2"; shift 2 ;;
        --skip-stages=*) SKIP_STAGES="${1#*=}"; shift ;;
        --freeze-evaluator) FREEZE_EVALUATOR="true"; shift ;;
        --use-vectorized-sweep) USE_VECTORIZED_SWEEP="true"; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ── Validate --skip-stages against the known stage vocabulary ────────────────
# Hard-fail on unknown names per no-silent-fails: a typo like
# --skip-stages=evaulator would silently run evaluator (no match) and mislead
# the operator into thinking the pipeline respected their request.
_KNOWN_STAGES="backtest parity evaluator"
if [ -n "$SKIP_STAGES" ]; then
    IFS=',' read -ra _SKIP_ARR <<< "$SKIP_STAGES"
    for _s in "${_SKIP_ARR[@]}"; do
        _s_trim="$(echo "$_s" | tr -d '[:space:]')"
        case " $_KNOWN_STAGES " in
            *" $_s_trim "*) ;;
            *)
                echo "ERROR: unknown stage '$_s_trim' in --skip-stages=$SKIP_STAGES" >&2
                echo "       Valid stages: $_KNOWN_STAGES" >&2
                exit 1
                ;;
        esac
    done
fi

# Convert each flag to a backtest.py CLI arg suffix (empty string when
# disabled, so we don't pass an invalid empty arg through the heredoc).
if [ "$SKIP_PHASE4" = "true" ]; then
    BACKTEST_SKIP_PHASE4_FLAG="--skip-phase4-evaluations"
else
    BACKTEST_SKIP_PHASE4_FLAG=""
fi

BACKTEST_PHASE_FLAGS=""
if [ -n "$SKIP_PHASES" ]; then
    BACKTEST_PHASE_FLAGS="$BACKTEST_PHASE_FLAGS --skip-phases=$SKIP_PHASES"
fi
if [ -n "$ONLY_PHASES" ]; then
    BACKTEST_PHASE_FLAGS="$BACKTEST_PHASE_FLAGS --only-phases=$ONLY_PHASES"
fi
if [ "$FORCE_ALL" = "true" ]; then
    BACKTEST_PHASE_FLAGS="$BACKTEST_PHASE_FLAGS --force"
fi
if [ -n "$FORCE_PHASES" ]; then
    BACKTEST_PHASE_FLAGS="$BACKTEST_PHASE_FLAGS --force-phases=$FORCE_PHASES"
fi
if [ "$DRY_RUN" = "true" ]; then
    BACKTEST_PHASE_FLAGS="$BACKTEST_PHASE_FLAGS --dry-run"
fi
if [ "$USE_VECTORIZED_SWEEP" = "true" ]; then
    BACKTEST_PHASE_FLAGS="$BACKTEST_PHASE_FLAGS --use-vectorized-sweep"
fi

# Smoke-safe subset of BACKTEST_PHASE_FLAGS. Smoke modes set their own
# only-/skip-phases via `_apply_smoke_fixture`, so propagating the
# operator's --skip-phases / --only-phases / --force-phases would
# conflict with the fixture's narrowing semantics. Only flags that
# affect compute behavior (not phase selection) flow through. Currently
# just --use-vectorized-sweep — added for Tier 4 Layer 2 smoke
# validation (ROADMAP P0 2026-04-27). Without this, the host parses
# --use-vectorized-sweep but no smoke command ever sees the flag, so
# `smoke-predictor-param-sweep` would silently exercise the scalar path.
SMOKE_PHASE_FLAGS=""
if [ "$USE_VECTORIZED_SWEEP" = "true" ]; then
    SMOKE_PHASE_FLAGS="$SMOKE_PHASE_FLAGS --use-vectorized-sweep"
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  Backtester Spot Run — $(date +%Y-%m-%d)"
echo "═══════════════════════════════════════════════════════════════"
echo "  Instance type : $INSTANCE_TYPE"
echo "  AMI           : $AMI_ID"
echo "  Region        : $AWS_REGION"
echo "  Branch        : $BRANCH"
echo "  Backtest mode : $BACKTEST_MODE"
echo "  Run mode      : $RUN_MODE"
echo "  Skip phase 4  : $SKIP_PHASE4"
echo "  Skip phases   : ${SKIP_PHASES:-(none)}"
echo "  Only phases   : ${ONLY_PHASES:-(none)}"
echo "  Force all     : $FORCE_ALL"
echo "  Force phases  : ${FORCE_PHASES:-(none)}"
echo "  Dry-run       : $DRY_RUN"
echo "  Skip stages   : ${SKIP_STAGES:-(none)}"
echo "  Freeze eval   : $FREEZE_EVALUATOR"
echo "  Vectorized sw : $USE_VECTORIZED_SWEEP"
echo "  S3 bucket     : $S3_BUCKET"
echo ""

# ── Preflight checks ──────────────────────────────────────────────────────────
if [ ! -f "$KEY_FILE" ]; then
    echo "ERROR: SSH key not found at $KEY_FILE"
    exit 1
fi

if [ ! -f "$REPO_ROOT/config.yaml" ]; then
    echo "ERROR: config.yaml not found — copy from config.yaml.example"
    exit 1
fi

# ── Launch spot instance ──────────────────────────────────────────────────────
echo "==> Requesting spot instance ($INSTANCE_TYPE)..."

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SECURITY_GROUP" \
    --subnet-id "$SUBNET_ID" \
    --iam-instance-profile Name="$IAM_PROFILE" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=alpha-engine-backtest-$(date +%Y%m%d)}]" \
    --region "$AWS_REGION" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "  Instance ID: $INSTANCE_ID"

# Cleanup function — always terminate the instance
cleanup() {
    echo ""
    echo "==> Terminating spot instance $INSTANCE_ID..."
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$AWS_REGION" --output text > /dev/null 2>&1 || true
    echo "  Instance terminated."
}
trap cleanup EXIT

# Wait for instance to be running
echo "==> Waiting for instance to enter running state..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$AWS_REGION"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text \
    --region "$AWS_REGION")

if [ "$PUBLIC_IP" = "None" ] || [ -z "$PUBLIC_IP" ]; then
    echo "ERROR: Instance has no public IP. Check subnet/VPC configuration."
    exit 1
fi

echo "  Public IP: $PUBLIC_IP"

# ── Wait for SSH ──────────────────────────────────────────────────────────────
echo "==> Waiting for SSH to become available..."
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o LogLevel=ERROR"

for i in $(seq 1 30); do
    if ssh $SSH_OPTS -i "$KEY_FILE" ec2-user@"$PUBLIC_IP" "echo ok" 2>/dev/null; then
        echo "  SSH ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: SSH not available after 150s"
        exit 1
    fi
    sleep 5
done

# Helper: run command on EC2
run_remote() {
    ssh $SSH_OPTS -i "$KEY_FILE" ec2-user@"$PUBLIC_IP" "$@"
}

# ── Spot-side watchdog ──────────────────────────────────────────────────────
# Dispatcher-side `trap cleanup EXIT` only fires when THIS bash script exits
# cleanly. If the dispatcher SSM command is cancelled, the dispatcher EC2
# is stopped mid-run, or the shell gets SIGKILLed, the trap never runs and
# the spot orphans until manually terminated — hit 3 times in April 2026.
# Transient systemd timer on the spot fires shutdown -h now after
# MAX_RUNTIME_SECONDS regardless of dispatcher state.
echo "==> Installing spot-side watchdog (${MAX_RUNTIME_SECONDS}s = $((MAX_RUNTIME_SECONDS / 60)) min)..."
run_remote "sudo systemd-run --on-active=${MAX_RUNTIME_SECONDS} --unit=alpha-engine-watchdog --description='alpha-engine spot hard-timeout' /sbin/shutdown -h now"

# ── Bootstrap environment ──────────────────────────────────────────────────────
echo "==> Bootstrapping EC2 environment..."
run_remote bash -s <<'BOOTSTRAP'
set -euo pipefail

# Amazon Linux 2023: install Python 3.12, git, gcc
sudo dnf install -y -q python3.12 python3.12-pip python3.12-devel git gcc 2>/dev/null || \
    sudo dnf install -y -q python3 python3-pip python3-devel git gcc

if command -v python3.12 &>/dev/null; then
    PYTHON=python3.12
else
    PYTHON=python3
fi
echo "Using: $($PYTHON --version)"

# SSH for GitHub
mkdir -p ~/.ssh
ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null
BOOTSTRAP

echo "==> Cloning repositories (branch: $BRANCH)..."
# flow-doctor is now pulled in via alpha-engine-lib[flow_doctor] from
# requirements.txt — no bundled editable install needed.
for REPO in alpha-engine-backtester alpha-engine alpha-engine-predictor; do
    echo "  Cloning $REPO..."
    ssh -A $SSH_OPTS -i "$KEY_FILE" ec2-user@"$PUBLIC_IP" \
        "git clone --depth 1 --branch $BRANCH git@github.com:cipher813/$REPO.git /home/ec2-user/$REPO" 2>/dev/null || {
        HTTPS_URL="https://github.com/cipher813/$REPO.git"
        run_remote "git clone --depth 1 --branch $BRANCH $HTTPS_URL /home/ec2-user/$REPO"
    }
done

# ── Upload .env BEFORE pip install ─────────────────────────────────────────────
# .env carries non-secret runtime config (EMAIL_*, S3_BUCKET, etc.) that the
# workload sources before running. alpha-engine-lib was flipped public
# 2026-05-03, so the spot installs it directly from git+https with no auth —
# earlier versions of this script fetched a PAT from SSM /alpha-engine/lib-token.
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: .env not found (checked alpha-engine-data/.env, ~/.alpha-engine.env, ./.env)"
    exit 1
fi
echo "==> Uploading .env to spot instance..."
scp $SSH_OPTS -i "$KEY_FILE" \
    "$ENV_FILE" \
    ec2-user@"$PUBLIC_IP":/home/ec2-user/alpha-engine-backtester/.env

echo "==> Installing Python dependencies..."
run_remote bash -s <<'DEPS'
set -euo pipefail
cd /home/ec2-user/alpha-engine-backtester

# Source .env for non-secret runtime vars (EMAIL_*, S3_BUCKET, etc.).
# alpha-engine-lib is public; pip resolves the git+https URL in
# requirements.txt without auth.
set -a
# shellcheck disable=SC1091
source /home/ec2-user/alpha-engine-backtester/.env
set +a

if command -v python3.12 &>/dev/null; then
    PIP="python3.12 -m pip"
else
    PIP="python3 -m pip"
fi

$PIP install --upgrade pip -q
$PIP install -q -r requirements.txt

# Also install predictor deps (needed for GBM inference + feature computation)
cd /home/ec2-user/alpha-engine-predictor
if [ -f requirements.txt ]; then
    $PIP install -q -r requirements.txt 2>/dev/null || true
fi

# Force numpy<2 after all deps (pyarrow compiled against numpy 1.x)
$PIP install -q 'numpy<2'

echo "Dependencies installed."
DEPS

# ── Copy remaining config files ─────────────────────────────────────────────────
echo "==> Uploading config.yaml..."
scp $SSH_OPTS -i "$KEY_FILE" \
    "$REPO_ROOT/config.yaml" \
    ec2-user@"$PUBLIC_IP":/home/ec2-user/alpha-engine-backtester/config.yaml

# Copy executor config (needed for simulation).
# Prod path is alpha-engine-config/executor/risk.yaml — the private config
# repo pulled daily on ae-dashboard by boot-pull. Legacy alpha-engine/config/
# path is kept for local dev fallback but has not been populated on
# ae-dashboard since the config-repo split (2026-04-07). Hit 2026-04-20:
# spot silently fell back to risk.yaml.example, executor read placeholder
# signals_bucket="your-research-bucket-name", ArcticDB KeyNotFound on a
# nonexistent bucket.
EXECUTOR_CONFIG=""
for candidate in \
    "$HOME/alpha-engine-config/executor/risk.yaml" \
    "$HOME/Development/alpha-engine-config/executor/risk.yaml" \
    "$HOME/alpha-engine/config/risk.yaml" \
    "$HOME/Development/alpha-engine/config/risk.yaml"; do
    if [ -f "$candidate" ]; then
        EXECUTOR_CONFIG="$candidate"
        break
    fi
done

if [ -z "$EXECUTOR_CONFIG" ]; then
    echo "ERROR: executor risk.yaml not found in any search path:" >&2
    echo "  ~/alpha-engine-config/executor/risk.yaml" >&2
    echo "  ~/Development/alpha-engine-config/executor/risk.yaml" >&2
    echo "  ~/alpha-engine/config/risk.yaml (legacy)" >&2
    echo "  ~/Development/alpha-engine/config/risk.yaml (legacy)" >&2
    echo "Backtester simulation cannot run without the executor config — silently" >&2
    echo "falling back to risk.yaml.example produces all-placeholder bucket names" >&2
    echo "and ArcticDB KeyNotFoundException deep in the executor-sim run." >&2
    exit 1
fi
echo "  Uploading risk.yaml from $EXECUTOR_CONFIG"
run_remote "mkdir -p /home/ec2-user/alpha-engine/config"
scp $SSH_OPTS -i "$KEY_FILE" \
    "$EXECUTOR_CONFIG" \
    ec2-user@"$PUBLIC_IP":/home/ec2-user/alpha-engine/config/risk.yaml

# Copy predictor config (needed for predictor backtest).
PREDICTOR_CONFIG=""
for candidate in \
    "$HOME/alpha-engine-predictor/config/predictor.yaml" \
    "$HOME/Development/alpha-engine-predictor/config/predictor.yaml"; do
    if [ -f "$candidate" ]; then
        PREDICTOR_CONFIG="$candidate"
        break
    fi
done

if [ -n "$PREDICTOR_CONFIG" ]; then
    echo "  Uploading predictor.yaml from $PREDICTOR_CONFIG"
    run_remote "mkdir -p /home/ec2-user/alpha-engine-predictor/config"
    scp $SSH_OPTS -i "$KEY_FILE" \
        "$PREDICTOR_CONFIG" \
        ec2-user@"$PUBLIC_IP":/home/ec2-user/alpha-engine-predictor/config/predictor.yaml
else
    echo "  WARNING: predictor.yaml not found — predictor backtest will be skipped"
fi

# Bootstrap predictor data cache (slim cache parquets + sector_map required for backtest)
echo "==> Downloading predictor slim cache from S3 (~25 MB)..."
run_remote bash -s <<'CACHE'
set -euo pipefail
CACHE_DIR="/home/ec2-user/alpha-engine-predictor/data/cache"
mkdir -p "$CACHE_DIR"
if command -v aws &>/dev/null; then
    aws s3 cp s3://alpha-engine-research/predictor/price_cache/sector_map.json "$CACHE_DIR/sector_map.json" 2>/dev/null || true
    aws s3 sync s3://alpha-engine-research/predictor/price_cache_slim/ "$CACHE_DIR/" --quiet 2>/dev/null || true
fi
echo "Predictor cache dir: $(ls "$CACHE_DIR"/*.parquet 2>/dev/null | wc -l) parquet files"
CACHE

# ── Build env export command ─────────────────────────────────────────────────
# PYTHONUNBUFFERED=1: line-buffering stdout/stderr so SSM ships log lines as
# they're emitted. Without this, stdout is block-buffered when the agent
# captures it to CloudWatch — the 2026-04-22 4th Saturday SF dry-run lost
# ~16 minutes of in-flight output when the SSM agent died mid-run and
# buffered lines never reached the log. Combined with the phase markers
# in pipeline_common.phase (which explicit-flush after each START/END),
# this closes the "silent 110-minute phase" blind spot. Paired with
# `python -u` on each backtest.py invocation below as belt-and-suspenders.
ENV_SOURCE='set -a; [ -f /home/ec2-user/alpha-engine-backtester/.env ] && source /home/ec2-user/alpha-engine-backtester/.env; set +a; export XDG_CACHE_HOME=/tmp; export PYTHONUNBUFFERED=1;'

# Determine python binary on remote
REMOTE_PYTHON=$(run_remote "command -v python3.12 || command -v python3")

# ── Smoke test ────────────────────────────────────────────────────────────────
if [ "$RUN_MODE" = "smoke-only" ]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  SMOKE TEST"
    echo "═══════════════════════════════════════════════════════════════"

    # backtest.py --mode=smoke runs BacktesterPreflight + runtime smoke
    # (end-to-end with minimal data: universe symbols, per-ticker Arctic
    # read, recent signals.json load, Layer-1A GBM load + predict) and
    # exits 0. Keeps the smoke path in lockstep with what full modes do
    # at startup — no drift between bash-driven smoke and in-process
    # pipeline validation. Evaluate-mode smoke follows: artifact-read
    # path + BacktesterPreflight(mode="evaluate").
    run_remote bash -s <<SMOKE
set -euo pipefail
cd /home/ec2-user/alpha-engine-backtester
${ENV_SOURCE}

BUCKET="\${OUTPUT_BUCKET:-alpha-engine-research}"

# Per-mode smoke summary — collected throughout the run and printed as
# a single table at the end. Each entry: "name|status|duration|budget|usage".
# Populated regardless of pass/fail so partial runs still show which
# modes completed before the failure.
declare -a _SMOKE_SUMMARY=()

_smoke_record() {
    # args: name, status ("ok" | "FAIL"), duration_s, budget_s (may be ""), usage_pct (may be "")
    _SMOKE_SUMMARY+=("\$1|\$2|\$3|\$4|\$5")
}

_smoke_extract_budget() {
    # Pull "N.Ns <= N.Ns (N% of budget)" from a log file's last
    # budget-check line. Emits "budget_s<TAB>usage_pct" or empty.
    local log_file="\$1"
    local line
    line="\$(grep -oE 'budget check: [0-9.]+s <= [0-9.]+s \([0-9]+% of budget\)' "\$log_file" | tail -1 || true)"
    [ -z "\$line" ] && return
    local budget usage
    budget="\$(echo "\$line" | grep -oE '<= [0-9.]+s' | grep -oE '[0-9.]+s')"
    usage="\$(echo "\$line" | grep -oE '\([0-9]+%' | tr -d '(%')%"
    printf '%s\t%s' "\$budget" "\$usage"
}

_smoke_run_mode() {
    # Run one backtest.py --mode=X, tee output, record to summary.
    # Returns non-zero on Python failure so caller can decide to break.
    local mode="\$1"
    local log_file="/tmp/smoke_\${mode//\//_}.log"
    local start=\$SECONDS
    local status="ok"

    echo ""
    echo "==> Smoke: backtest.py --mode=\$mode $SMOKE_PHASE_FLAGS"
    if ! $REMOTE_PYTHON -u backtest.py --mode=\$mode --log-level INFO $SMOKE_PHASE_FLAGS 2>&1 | tee "\$log_file"; then
        status="FAIL"
    fi
    local dur=\$((SECONDS - start))

    local budget="" usage=""
    local extracted
    extracted="\$(_smoke_extract_budget "\$log_file")"
    if [ -n "\$extracted" ]; then
        budget="\${extracted%%\$'\t'*}"
        usage="\${extracted##*\$'\t'}"
    fi

    _smoke_record "\$mode" "\$status" "\${dur}s" "\$budget" "\$usage"
    [ "\$status" = "ok" ]
}

_smoke_print_summary() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  SMOKE SUMMARY"
    echo "═══════════════════════════════════════════════════════════════"
    printf "  %-28s %-8s %-10s %-10s %-8s\n" "Mode" "Status" "Duration" "Budget" "Usage"
    printf "  %s\n" "─────────────────────────────────────────────────────────────────────"
    local any_fail=0
    for entry in "\${_SMOKE_SUMMARY[@]}"; do
        IFS='|' read -r name status dur budget usage <<< "\$entry"
        printf "  %-28s %-8s %-10s %-10s %-8s\n" "\$name" "\$status" "\$dur" "\${budget:-–}" "\${usage:-–}"
        [ "\$status" = "FAIL" ] && any_fail=1
    done
    echo "═══════════════════════════════════════════════════════════════"
    if [ "\$any_fail" = "1" ]; then
        echo "  RESULT: FAIL (one or more modes did not pass)"
    else
        echo "  RESULT: PASS (all \${#_SMOKE_SUMMARY[@]} modes ok)"
    fi
    echo "═══════════════════════════════════════════════════════════════"
}

# Always print summary, even if a mode aborts mid-run.
trap '_smoke_print_summary' EXIT

# backtest.py --mode=smoke: preflight + runtime smoke (universe symbols,
# per-ticker Arctic read, recent signals.json load, Layer-1A GBM load +
# predict). Keeps smoke in lockstep with what full modes do at startup.
if ! _smoke_run_mode smoke; then
    echo "ERROR: smoke preflight FAILED — aborting"
    exit 1
fi

# Per-phase smoke harness — exercise each pipeline phase-family with a
# tiny fixture (few dates, tiny param grid, short GBM lookback) and
# enforce per-mode wall-clock budgets from timing_budget.yaml. Ordered
# fastest → slowest so a failure in an earlier mode short-circuits
# the harder ones. ROADMAP Backtester P0 #3.
for SMOKE_PHASE_MODE in smoke-simulate smoke-param-sweep smoke-predictor-backtest smoke-phase4 smoke-predictor-param-sweep; do
    if ! _smoke_run_mode "\$SMOKE_PHASE_MODE"; then
        echo "ERROR: smoke phase \$SMOKE_PHASE_MODE FAILED — aborting smoke-only run"
        exit 1
    fi
done

echo ""
echo "==> Resolving most recent backtest artifact date from s3://\${BUCKET}/backtest/..."
# Pick the most-recent date that ALSO has portfolio_stats.json on S3.
# The plain "sort | tail -1" approach picked stale empty prefixes
# created by prior half-complete runs (observed 2026-04-24 smoke: a
# 2026-04-24/ prefix existed but had no artifacts, causing evaluate.py
# to hard-fail with "All critical simulation artifacts missing").
# Excluding hidden prefixes (.smoke/, .dry-run/) keeps the probe
# pointing at production dates.
LATEST_DATE=""
while IFS= read -r candidate; do
    [ -z "\$candidate" ] && continue
    case "\$candidate" in .*) continue ;; esac
    if aws s3api head-object --bucket "\${BUCKET}" --key "backtest/\$candidate/portfolio_stats.json" >/dev/null 2>&1; then
        LATEST_DATE="\$candidate"
        break
    fi
done < <(aws s3 ls "s3://\${BUCKET}/backtest/" | awk '/PRE / {print \$2}' | tr -d '/' | sort -r)
if [ -z "\$LATEST_DATE" ]; then
    echo "ERROR: no backtest/{date}/ prefix with portfolio_stats.json found in s3://\${BUCKET}/backtest/"
    _smoke_record "evaluate-diagnostics" "FAIL" "0s" "" ""
    exit 1
fi
echo "Using backtest date: \$LATEST_DATE"

echo ""
echo "==> Smoke: evaluate.py --mode diagnostics --freeze --date \$LATEST_DATE"
_EVAL_START=\$SECONDS
_EVAL_STATUS="ok"
if ! $REMOTE_PYTHON -u evaluate.py --mode diagnostics --freeze --date "\$LATEST_DATE" --log-level INFO 2>&1 | tail -30; then
    _EVAL_STATUS="FAIL"
fi
_EVAL_DUR=\$((SECONDS - _EVAL_START))
_smoke_record "evaluate-diagnostics" "\$_EVAL_STATUS" "\${_EVAL_DUR}s" "" ""

echo ""
echo "Smoke test complete."
# Summary prints via trap on exit
SMOKE

    echo "==> Smoke-only mode — instance will be terminated."
    exit 0
fi

# ── Full backtest ─────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  FULL BACKTEST (--mode $BACKTEST_MODE)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

run_remote bash -s <<BACKTEST
set -euo pipefail
cd /home/ec2-user/alpha-engine-backtester
${ENV_SOURCE}

# BUCKET used across all three stages. OUTPUT_BUCKET is set by .env (sourced
# above) but fall back to the default so \`\`set -u\`\` doesn't blow up on an
# environment without the .env line. Matches the smoke-only heredoc's line.
BUCKET="\${OUTPUT_BUCKET:-alpha-engine-research}"
# SKIP_STAGES baked in from the dispatcher's --skip-stages flag. Stages in
# this CSV are skipped with a loud ⊘ echo; everything else runs.
SKIP_STAGES="${SKIP_STAGES}"
# Shared RUN_DATE used by parity + evaluator uploads so they land under the
# same backtest/{date}/ prefix.
RUN_DATE=\$(date -u +%Y-%m-%d)

_stage_skipped() {
    case ",\${SKIP_STAGES}," in
        *",\$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

# ── Stage: backtest ─────────────────────────────────────────────────────────
# If backtest.py fails we exit non-zero so parity + evaluator never run
# against stale or missing artifacts — the evaluator would otherwise
# auto-promote garbage params to S3. Fail loud so the spot run is marked
# failed, the heartbeat metric is not emitted, and the Step Function catches
# it. Replaces the previous || { echo WARNING } swallow that silently let
# evaluator run against invalid sweep results and was the root cause of
# multiple undetected param oscillations.
if _stage_skipped backtest; then
    echo "⊘ stage=backtest SKIPPED (--skip-stages=\${SKIP_STAGES})"
else
    echo "▶ stage=backtest START at \$(date -u +%H:%M:%S)"
    if ! $REMOTE_PYTHON -u backtest.py --mode $BACKTEST_MODE --upload --log-level INFO $BACKTEST_SKIP_PHASE4_FLAG $BACKTEST_PHASE_FLAGS 2>&1; then
        echo "ERROR: backtest.py failed. Spot run marked FAILED — check" >&2
        echo "       flow-doctor alerts. Parity + evaluator stages skipped" >&2
        echo "       to prevent auto-promotion of unvalidated configs." >&2
        exit 1
    fi
    echo "▶ stage=backtest END at \$(date -u +%H:%M:%S)"
fi

# ── Stage: parity ───────────────────────────────────────────────────────────
# Parity is OBSERVABILITY, not a gate. Each Saturday SF run produces:
#   * parity_report.json — per-run drill-down (count + ticker-set + field
#     divergence breakdowns), uploaded to s3://{bucket}/backtest/{date}/
#   * parity_metrics.csv — append one row per run with capture_rate,
#     ticker_jaccard_avg, count_divergence_rms, field_diff_rate,
#     n_lifecycle_skipped. Time series at
#     s3://{bucket}/backtest/parity_metrics.csv. The metric trend is the
#     load-bearing signal; step-changes trigger investigation.
# The pytest assertion was removed (test always passes — its job is to
# generate the artifacts). The spot run does NOT fail the SF on parity
# divergence: 0% historical parity is structurally unreachable for a
# system with weekly auto-tuned configs and evolving executor code.
# See tests/test_parity_replay.py module docstring for the full rationale.
# Setup-level failures (missing trades.db, ArcticDB unreachable) are still
# fatal here — those are real infrastructure breakage, not "expected drift".
if _stage_skipped parity; then
    echo "⊘ stage=parity SKIPPED (--skip-stages=\${SKIP_STAGES})"
else
    echo "▶ stage=parity START at \$(date -u +%H:%M:%S)"
    PARITY_TRADES_DB="/tmp/trades_latest.db"
    PARITY_REPORT_DIR="/tmp/parity_report"
    mkdir -p "\$PARITY_REPORT_DIR"

    if ! aws s3 cp "s3://\${BUCKET}/trades/trades_latest.db" "\$PARITY_TRADES_DB" --quiet; then
        echo "ERROR: could not download trades_latest.db from S3 — parity cannot run" >&2
        echo "       This is infrastructure breakage (not divergence) — failing spot." >&2
        exit 1
    fi

    PARITY_EXIT=0
    # USE_REAL_ARCTICDB=1 tells tests/conftest.py to skip the default
    # MagicMock stub so the integration test hits real ArcticDB.
    # PARITY_RUN_DATE pins the time-series CSV's run_date column to
    # today's RUN_DATE so re-runs of a single Saturday cohort overwrite
    # idempotently rather than producing duplicate rows.
    TRADES_DB_PATH="\$PARITY_TRADES_DB" \\
    SIGNALS_BUCKET="\${BUCKET}" \\
    PARITY_REPORT_DIR="\$PARITY_REPORT_DIR" \\
    PARITY_RUN_DATE="\${RUN_DATE}" \\
    USE_REAL_ARCTICDB=1 \\
    $REMOTE_PYTHON -m pytest tests/test_parity_replay.py -m parity -v 2>&1 || PARITY_EXIT=\$?

    # Upload the per-run report. The time-series CSV is appended by the
    # test itself (see append_parity_metrics_row) — best-effort, errors
    # WARN-not-FAIL since the per-run report is the authoritative artifact.
    if [ -f "\$PARITY_REPORT_DIR/parity_report.json" ]; then
        aws s3 cp "\$PARITY_REPORT_DIR/parity_report.json" \\
            "s3://\${BUCKET}/backtest/\${RUN_DATE}/parity_report.json" --quiet \\
            && echo "Uploaded parity_report.json to s3://\${BUCKET}/backtest/\${RUN_DATE}/" \\
            || echo "WARNING: failed to upload parity_report.json (non-fatal)"
    fi

    # Pytest exit codes:
    #   0 = test ran (always-pass, since divergence is observability not gate)
    #   non-zero with parity_report.json present = setup-level error inside the
    #     test body (e.g. ArcticDB read failure on integration path); flag a
    #     WARNING but don't fail the spot — operator can still inspect the
    #     report. The SF alarm should fire on real infrastructure breakage
    #     (the s3 cp failure above), not on observability test signaling.
    if [ "\$PARITY_EXIT" != "0" ]; then
        echo "WARNING: parity pytest exited \$PARITY_EXIT (likely setup-side error)." >&2
        echo "         See s3://\${BUCKET}/backtest/\${RUN_DATE}/parity_report.json (if present)." >&2
        echo "         Continuing spot run — parity is observability, not a gate." >&2
    fi
    echo "▶ stage=parity END at \$(date -u +%H:%M:%S)"
fi

# ── Stage: evaluator ────────────────────────────────────────────────────────
# Runs evaluate.py against today's backtest artifacts in S3. Consolidated
# into the spot step 2026-04-24 — the SF's dedicated Evaluator states
# (CheckSkipEvaluator, CheckEvaluatorFreeze, Evaluator, EvaluatorFrozen,
# WaitForEvaluator, CheckEvaluatorStatus, EvaluatorWait, ExtractEvaluatorError)
# were retired. --freeze-evaluator controls config-promotion (freeze =
# diagnostic-only, no config writes). Default is live-apply for the Sat SF;
# manual iteration runs should pass --freeze-evaluator.
if _stage_skipped evaluator; then
    echo "⊘ stage=evaluator SKIPPED (--skip-stages=\${SKIP_STAGES})"
else
    echo "▶ stage=evaluator START at \$(date -u +%H:%M:%S) freeze=${FREEZE_EVALUATOR}"
    _EVAL_FREEZE=""
    if [ "${FREEZE_EVALUATOR}" = "true" ]; then
        _EVAL_FREEZE="--freeze"
    fi
    if ! $REMOTE_PYTHON -u evaluate.py --mode all --upload \$_EVAL_FREEZE --log-level INFO 2>&1; then
        echo "ERROR: evaluate.py failed. Spot run marked FAILED." >&2
        exit 1
    fi
    echo "▶ stage=evaluator END at \$(date -u +%H:%M:%S)"
fi

echo ""
echo "All requested stages complete at \$(date)"
BACKTEST

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Backtest complete. Instance will be terminated."
echo "═══════════════════════════════════════════════════════════════"

# Emit CloudWatch heartbeat on successful completion
aws cloudwatch put-metric-data \
  --namespace "AlphaEngine" \
  --metric-name "Heartbeat" \
  --dimensions "Process=backtester" \
  --value 1 --unit "Count" \
  --region "${AWS_REGION:-us-east-1}" 2>/dev/null \
  && echo "Heartbeat emitted: backtester" \
  || echo "WARNING: Failed to emit heartbeat (non-fatal)"
