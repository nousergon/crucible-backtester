#!/usr/bin/env bash
# infrastructure/deploy_concordance.sh — Build and deploy the weekly
# cheap-model concordance Lambda.
#
# Lightweight container: nousergon-lib + langchain_anthropic + boto3
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

# ── Step 5: Wait for update to complete ──────────────────────────────────────
echo ""
echo "==> Waiting for Lambda update to complete..."
aws lambda wait function-updated --function-name "${LAMBDA_FUNCTION}" --region "${AWS_REGION}"

# ── Step 6: Publish version (do NOT promote 'live' yet) ──────────────────────
echo ""
echo "==> Publishing Lambda version..."
VERSION=$(aws lambda publish-version --function-name "${LAMBDA_FUNCTION}" --query "Version" --output text --region "${AWS_REGION}")
echo "  Published version: ${VERSION}"

# ── Step 7: Canary invocation against the NEW VERSION ────────────────────────
# Canary runs BEFORE promoting 'live' so a canary failure leaves the live
# alias pointing at the prior good version — no manual rollback owed.
# Pre-2026-05-22 this script promoted live first, ran canary second
# (filed in alpha-engine-config PR #272 as the L221-audit follow-up).
# Sibling research/predictor/data deploys already follow canary-first.
#
# dry_run=true skips Anthropic + CloudWatch + S3 puts; just lists candidate
# artifacts. Should complete in seconds.
echo ""
echo "==> Running canary invocation against :${VERSION} (dry_run=true, window_days=14)..."
CANARY_OUT=$(mktemp)
aws lambda invoke \
  --function-name "${LAMBDA_FUNCTION}:${VERSION}" \
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
  echo "  'live' alias UNCHANGED — still points at prior good version."
  # ROADMAP L221 — independent-channel surveillance. dedup_key collapses
  # an image-wide rebuild that breaks N Lambdas' canaries within the
  # hour into one alert per (Lambda, version). Best-effort; trailing
  # || true never overrides exit 1.
  # krepis.alerts (config#1339/config#1545) — this script is operator-run
  # only (not wired into GHA CI), so the operator's local venv must have
  # krepis installed (already a repo dependency, requirements.txt) for
  # this alert to fire; no separate runner-install step applies here.
  python3 -m krepis.alerts publish \
    --severity error \
    --source "alpha-engine-backtester/infrastructure/deploy_concordance.sh" \
    --dedup-key "canary-fail-${LAMBDA_FUNCTION}-v${VERSION}" \
    --message "Canary failed: ${LAMBDA_FUNCTION}:${VERSION} canary returned status='${CANARY_STATUS}'. 'live' alias is UNCHANGED (still on prior good version) — no manual rollback owed; investigate and re-deploy. See CloudWatch /aws/lambda/${LAMBDA_FUNCTION}." \
    || true
  exit 1
fi
echo "  Canary passed (status=$CANARY_STATUS)"

# ── Step 8: Promote 'live' alias only on canary success ──────────────────────
echo ""
echo "==> Promoting 'live' alias → version ${VERSION}"
aws lambda update-alias --function-name "${LAMBDA_FUNCTION}" --name live --function-version "${VERSION}" --region "${AWS_REGION}" 2>/dev/null || \
aws lambda create-alias --function-name "${LAMBDA_FUNCTION}" --name live --function-version "${VERSION}" --region "${AWS_REGION}"

echo ""
echo "==> Deploy complete!"
echo "    Function : ${LAMBDA_FUNCTION}"
echo "    Version  : ${VERSION}"
echo "    Alias    : live → ${VERSION}"
echo "    Image    : ${ECR_IMAGE}"
