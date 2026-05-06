#!/usr/bin/env bash
# infrastructure/deploy_concordance.sh — Build and deploy the weekly
# cheap-model concordance Lambda.
#
# Lightweight container: alpha-engine-lib + langchain_anthropic + boto3
# (~150MB). Runs the trailing-window replay pipeline that emits
# agent_cheap_model_concordance to CloudWatch.
#
# Prerequisites:
#   - Docker installed and running
#   - AWS CLI configured
#   - ECR repo 'alpha-engine-replay-concordance' (created lazily on first push)
#   - Lambda function 'alpha-engine-replay-concordance' (created lazily on first deploy)
#
# Usage:
#   ./infrastructure/deploy_concordance.sh                # full deploy
#   ./infrastructure/deploy_concordance.sh --dry-run      # build image only
#
# Environment variables (auto-detected if not set):
#   AWS_ACCOUNT_ID — 12-digit AWS account ID (auto-detected via aws sts)
#   AWS_REGION     — defaults to us-east-1

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
ECR_REPO="alpha-engine-replay-concordance"
LAMBDA_FUNCTION="alpha-engine-replay-concordance"
IMAGE_TAG="latest"
DRY_RUN=false

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
echo "Working directory: $REPO_ROOT"

# ── Step 1: Build Docker image ───────────────────────────────────────────────
echo ""
echo "==> Building Docker image..."
docker build --platform linux/amd64 --provenance=false --tag "${ECR_REPO}:${IMAGE_TAG}" --file lambda_concordance/Dockerfile .

echo "  Image built: ${ECR_REPO}:${IMAGE_TAG}"

if [ "$DRY_RUN" = true ]; then
  echo ""
  echo "==> DRY RUN: Skipping ECR push and Lambda update."
  echo "    Image built successfully as ${ECR_REPO}:${IMAGE_TAG}"
  exit 0
fi

# ── Step 2: Authenticate to ECR ──────────────────────────────────────────────
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_IMAGE="${ECR_REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"

echo ""
echo "==> Authenticating to ECR (${ECR_REGISTRY})..."
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# Ensure the ECR repo exists (idempotent — first deploy creates).
aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${AWS_REGION}" &>/dev/null || \
  aws ecr create-repository --repository-name "${ECR_REPO}" --region "${AWS_REGION}" > /dev/null

# ── Step 3: Tag and push image ───────────────────────────────────────────────
echo ""
echo "==> Tagging image: ${ECR_IMAGE}"
docker tag "${ECR_REPO}:${IMAGE_TAG}" "${ECR_IMAGE}"

echo "==> Pushing to ECR..."
docker push "${ECR_IMAGE}"
echo "  Pushed: ${ECR_IMAGE}"

# ── Step 4: Create or update Lambda ──────────────────────────────────────────
echo ""
ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/alpha-engine-research-role"

if aws lambda get-function --function-name "${LAMBDA_FUNCTION}" --region "${AWS_REGION}" &>/dev/null; then
  echo "==> Updating Lambda function: ${LAMBDA_FUNCTION}"
  aws lambda update-function-code \
    --function-name "${LAMBDA_FUNCTION}" \
    --image-uri "${ECR_IMAGE}" \
    --region "${AWS_REGION}" \
    --output json | python3 -c "import sys,json; d=json.load(sys.stdin); print('  FunctionArn:', d.get('FunctionArn','?')); print('  LastModified:', d.get('LastModified','?'))"
else
  echo "==> Creating Lambda function: ${LAMBDA_FUNCTION}"
  aws lambda create-function \
    --function-name "${LAMBDA_FUNCTION}" \
    --package-type Image \
    --code "ImageUri=${ECR_IMAGE}" \
    --role "${ROLE_ARN}" \
    --timeout 900 \
    --memory-size 1024 \
    --region "${AWS_REGION}" \
    --output json | python3 -c "import sys,json; d=json.load(sys.stdin); print('  FunctionArn:', d.get('FunctionArn','?'))"
fi

# ── Step 4b: Sync env vars from master .env ─────────────────────────────────
LAMBDA_ENV_FILE="$(dirname "$REPO_ROOT")/alpha-engine-data/.env"
if [ ! -f "$LAMBDA_ENV_FILE" ]; then
  LAMBDA_ENV_FILE="$REPO_ROOT/.env"
fi
if [ -f "$LAMBDA_ENV_FILE" ]; then
  LAMBDA_ENV_JSON=$(python3 -c "
import json
env = {}
with open('$LAMBDA_ENV_FILE') as f:
    for line in f:
        line = line.strip()
        if line == '# LAMBDA_SKIP':
            break
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            continue
        key, val = line.split('=', 1)
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('\"', \"'\"):
            val = val[1:-1]
        if key and val:
            env[key] = val
if env:
    print(json.dumps({'Variables': env}))
else:
    print('')
")
  if [ -n "$LAMBDA_ENV_JSON" ]; then
    echo ""
    echo "==> Syncing env vars from $LAMBDA_ENV_FILE"
    aws lambda wait function-updated --function-name "${LAMBDA_FUNCTION}" --region "${AWS_REGION}" 2>/dev/null || sleep 5
    aws lambda update-function-configuration \
      --function-name "${LAMBDA_FUNCTION}" \
      --environment "$LAMBDA_ENV_JSON" \
      --region "${AWS_REGION}" > /dev/null
  fi
else
  echo "  WARNING: No .env file found — Lambda env vars not updated"
fi

# ── Step 5: Wait for update to complete ──────────────────────────────────────
echo ""
echo "==> Waiting for Lambda update to complete..."
aws lambda wait function-updated --function-name "${LAMBDA_FUNCTION}" --region "${AWS_REGION}"

# ── Step 6: Publish version and update 'live' alias ──────────────────────────
echo ""
echo "==> Publishing Lambda version..."
VERSION=$(aws lambda publish-version --function-name "${LAMBDA_FUNCTION}" --query "Version" --output text --region "${AWS_REGION}")
echo "  Published version: ${VERSION}"

echo "==> Updating 'live' alias → version ${VERSION}"
aws lambda update-alias --function-name "${LAMBDA_FUNCTION}" --name live --function-version "${VERSION}" --region "${AWS_REGION}" 2>/dev/null || \
aws lambda create-alias --function-name "${LAMBDA_FUNCTION}" --name live --function-version "${VERSION}" --region "${AWS_REGION}"

# ── Step 7: Canary invocation ───────────────────────────────────────────────
# dry_run=true skips Anthropic + CloudWatch + S3 puts; just lists candidate
# artifacts. Should complete in seconds.
echo ""
echo "==> Running canary invocation (dry_run=true, window_days=14)..."
CANARY_OUT=$(mktemp)
aws lambda invoke \
  --function-name "${LAMBDA_FUNCTION}:live" \
  --payload '{"dry_run": true, "window_days": 14}' \
  --cli-binary-format raw-in-base64-out \
  --cli-read-timeout 60 \
  --region "${AWS_REGION}" \
  "$CANARY_OUT" > /dev/null

CANARY_STATUS=$(python3 -c "import json; d=json.load(open('$CANARY_OUT')); print(d.get('status', 'ERROR'))" 2>/dev/null || echo "ERROR")
rm -f "$CANARY_OUT"

if [ "$CANARY_STATUS" != "OK" ] && [ "$CANARY_STATUS" != "PARTIAL" ]; then
  echo ""
  echo "ERROR: Canary returned status $CANARY_STATUS"
  echo "  Check CloudWatch Logs: /aws/lambda/${LAMBDA_FUNCTION}"
  exit 1
fi
echo "  Canary passed (status=$CANARY_STATUS)"

echo ""
echo "==> Deploy complete!"
echo "    Function : ${LAMBDA_FUNCTION}"
echo "    Version  : ${VERSION}"
echo "    Alias    : live → ${VERSION}"
echo "    Image    : ${ECR_IMAGE}"
