#!/usr/bin/env bash
# infrastructure/deploy_health.sh — Build and deploy the daily health check Lambda.
#
# Lightweight container: numpy/pandas/scipy/boto3 only (~120MB).
# Runs daily predictor health monitoring (Phases 2a, 2b, 5).
#
# Prerequisites:
#   - Docker installed and running
#   - AWS CLI configured
#   - ECR repo 'alpha-engine-health-check' exists in your account
#   - Lambda function 'alpha-engine-predictor-health-check' already created
#
# Usage:
#   ./infrastructure/deploy_health.sh                # full deploy
#   ./infrastructure/deploy_health.sh --dry-run      # build image only, skip AWS push
#
# Environment variables (auto-detected if not set):
#   AWS_ACCOUNT_ID   — 12-digit AWS account ID (auto-detected via aws sts)
#   AWS_REGION       — defaults to us-east-1

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
ECR_REPO="alpha-engine-health-check"
LAMBDA_FUNCTION="alpha-engine-predictor-health-check"
IMAGE_TAG="latest"
DRY_RUN=false

# Parse flags
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# ── Resolve AWS identity ─────────────────────────────────────────────────────
AWS_REGION="${AWS_REGION:-us-east-1}"
if [ -z "${AWS_ACCOUNT_ID:-}" ] && [ "$DRY_RUN" = false ]; then
  AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$AWS_REGION" 2>/dev/null) || { echo "ERROR: Could not auto-detect AWS_ACCOUNT_ID. Set it manually or configure AWS CLI."; exit 1; }
  echo "Auto-detected AWS_ACCOUNT_ID: $AWS_ACCOUNT_ID"
fi

# Move to repo root (script may be called from any directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
echo "Working directory: $REPO_ROOT"

# alpha-engine-lib is installed inside lambda_health/Dockerfile via pip
# from public git+https (lib was flipped public 2026-05-03). No vendor
# staging needed.

# ── Step 1: Build Docker image ────────────────────────────────────────────────
echo ""
echo "==> Building Docker image..."
# alpha-engine-config-I12122: CI exports GITHUB_SHA; a local run stamps HEAD.
GIT_SHA="${GITHUB_SHA:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
docker build --platform linux/amd64 --provenance=false --build-arg GIT_SHA="${GIT_SHA}" --tag "${ECR_REPO}:${IMAGE_TAG}" --file lambda_health/Dockerfile .

echo "  Image built: ${ECR_REPO}:${IMAGE_TAG}"

if [ "$DRY_RUN" = true ]; then
  echo ""
  echo "==> DRY RUN: Skipping ECR push and Lambda update."
  echo "    Image built successfully as ${ECR_REPO}:${IMAGE_TAG}"
  exit 0
fi

# ── Step 2: Authenticate to ECR ───────────────────────────────────────────────
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_IMAGE="${ECR_REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"

echo ""
echo "==> Authenticating to ECR (${ECR_REGISTRY})..."
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# ── Step 3: Tag and push image ────────────────────────────────────────────────
echo ""
echo "==> Tagging image: ${ECR_IMAGE}"
docker tag "${ECR_REPO}:${IMAGE_TAG}" "${ECR_IMAGE}"

echo "==> Pushing to ECR..."
docker push "${ECR_IMAGE}"
echo "  Pushed: ${ECR_IMAGE}"

# ── Step 4: Update Lambda function code ──────────────────────────────────────
echo ""
echo "==> Updating Lambda function: ${LAMBDA_FUNCTION}"
aws lambda update-function-code --function-name "${LAMBDA_FUNCTION}" --image-uri "${ECR_IMAGE}" --region "${AWS_REGION}" --output json | python3 -c "import sys,json; d=json.load(sys.stdin); print('  FunctionArn:', d.get('FunctionArn','?')); print('  LastModified:', d.get('LastModified','?'))"

# ── Step 5: Wait for update to complete ──────────────────────────────────────
echo ""
echo "==> Waiting for Lambda update to complete..."
aws lambda wait function-updated --function-name "${LAMBDA_FUNCTION}" --region "${AWS_REGION}"

# ── Step 5b: Ephemeral storage headroom ──────────────────────────────────────
# alpha-engine-config-I8704. This Lambda downloads research.db into /tmp to
# grade production health. On 2026-08-26 that database was 356 MB against
# Lambda's DEFAULT 512 MB /tmp — already 70% full, and it grows every week.
# The measured failure was `[Errno 28] No space left on device` on a warm
# container (fixed in handler.py too: the canary no longer downloads it, and a
# stale copy is removed before re-downloading). Those two make the CURRENT size
# work; this is the headroom so the next 150 MB of database growth is not
# another outage. Declared in the deploy script rather than set by hand so it
# survives a function replacement — a console-only change would not.
echo ""
echo "==> Ensuring 2048 MB ephemeral storage (research.db is ~356 MB and growing)"
aws lambda update-function-configuration --function-name "${LAMBDA_FUNCTION}" --ephemeral-storage '{"Size": 2048}' --region "${AWS_REGION}" --query 'EphemeralStorage.Size' --output text
aws lambda wait function-updated --function-name "${LAMBDA_FUNCTION}" --region "${AWS_REGION}"

# ── Step 6: Publish version (do NOT promote 'live' yet) ──────────────────────
echo ""
echo "==> Publishing Lambda version..."
VERSION=$(aws lambda publish-version --function-name "${LAMBDA_FUNCTION}" --query "Version" --output text --region "${AWS_REGION}")
echo "  Published version: ${VERSION}"

# ── Step 7: Canary invocation against the NEW VERSION ────────────────────────
# Canary runs BEFORE promoting 'live' so a canary failure leaves the live
# alias pointing at the prior good version — no manual rollback owed.
# Pre-2026-05-22 these scripts promoted live first, ran canary second
# (filed in alpha-engine-config PR #272 as the L221-audit follow-up
# "Backtester Lambda deploy scripts move canary BEFORE the live-alias
# promote"). Sibling research/predictor/data deploy scripts already
# follow the canary-first pattern.
echo ""
echo "==> Running canary invocation against :${VERSION} (dry_run=true)..."
CANARY_OUT=$(mktemp)
aws lambda invoke --function-name "${LAMBDA_FUNCTION}:${VERSION}" --payload '{"dry_run": true}' --cli-binary-format raw-in-base64-out --cli-read-timeout 300 --region "${AWS_REGION}" "$CANARY_OUT" > /dev/null

CANARY_STATUS=$(python3 -c "import json; d=json.load(open('$CANARY_OUT')); print(d.get('statusCode', 0))" 2>/dev/null || echo "0")
rm -f "$CANARY_OUT"

if [ "$CANARY_STATUS" != "200" ]; then
  echo ""
  echo "ERROR: Canary returned status $CANARY_STATUS"
  echo "  Check CloudWatch Logs: /aws/lambda/${LAMBDA_FUNCTION}"
  echo "  'live' alias UNCHANGED — still points at prior good version."
  # ROADMAP L221 — independent-channel surveillance. dedup_key collapses
  # an image-wide rebuild that breaks N Lambdas' canaries within the hour
  # into one alert per (Lambda, version). Best-effort; trailing || true
  # never overrides exit 1.
  # krepis.alerts (config#1339/config#1545) — this is the ONLY deploy
  # script in the repo wired into GHA CI (deploy.yml); the workflow now
  # installs krepis on the runner so this call actually publishes instead
  # of silently ModuleNotFoundError-ing behind the trailing `|| true`.
  python3 -m krepis.alerts publish \
    --severity error \
    --source "alpha-engine-backtester/infrastructure/deploy_health.sh" \
    --dedup-key "canary-fail-${LAMBDA_FUNCTION}-v${VERSION}" \
    --message "Canary failed: ${LAMBDA_FUNCTION}:${VERSION} canary returned statusCode=${CANARY_STATUS}. 'live' alias is UNCHANGED (still on prior good version) — no manual rollback owed; investigate the canary failure and re-deploy. See CloudWatch /aws/lambda/${LAMBDA_FUNCTION}." \
    || true
  exit 1
fi
echo "  Canary passed (status=$CANARY_STATUS)"

# ── Step 8: Promote 'live' alias only on canary success ──────────────────────
echo ""
echo "==> Promoting 'live' alias → version ${VERSION}"
aws lambda update-alias --function-name "${LAMBDA_FUNCTION}" --name live --function-version "${VERSION}" --region "${AWS_REGION}" 2>/dev/null || aws lambda create-alias --function-name "${LAMBDA_FUNCTION}" --name live --function-version "${VERSION}" --region "${AWS_REGION}"

echo ""
echo "==> Deploy complete!"
echo "    Function : ${LAMBDA_FUNCTION}"
echo "    Version  : ${VERSION}"
echo "    Alias    : live → ${VERSION}"
echo "    Image    : ${ECR_IMAGE}"
