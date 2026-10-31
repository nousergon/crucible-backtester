#!/usr/bin/env bash
# infrastructure/deploy_counterfactual.sh — Build and deploy the weekly
# counterfactual rule fit Lambda.
#
# Lighter than concordance: nousergon-lib + sklearn + boto3 (~250MB
# — sklearn is the bulk). No LLM calls, no langchain.
#
# Prerequisites:
#   - Docker installed and running
#   - AWS CLI configured
#   - ECR repo 'alpha-engine-replay-counterfactual' (created lazily)
#   - Lambda function 'alpha-engine-replay-counterfactual' (created lazily)
#
# Usage:
#   ./infrastructure/deploy_counterfactual.sh                # full deploy
#   ./infrastructure/deploy_counterfactual.sh --dry-run      # build image only

set -euo pipefail

ECR_REPO="alpha-engine-replay-counterfactual"
LAMBDA_FUNCTION="alpha-engine-replay-counterfactual"
IMAGE_TAG="latest"
DRY_RUN=false

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

AWS_REGION="${AWS_REGION:-us-east-1}"
if [ -z "${AWS_ACCOUNT_ID:-}" ] && [ "$DRY_RUN" = false ]; then
  AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$AWS_REGION" 2>/dev/null) || { echo "ERROR: Could not auto-detect AWS_ACCOUNT_ID. Set it manually or configure AWS CLI."; exit 1; }
  echo "Auto-detected AWS_ACCOUNT_ID: $AWS_ACCOUNT_ID"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
echo "Working directory: $REPO_ROOT"

echo ""
echo "==> Building Docker image..."
# alpha-engine-config-I12122: CI exports GITHUB_SHA; a local run stamps HEAD.
GIT_SHA="${GITHUB_SHA:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
docker build --platform linux/amd64 --provenance=false --build-arg GIT_SHA="${GIT_SHA}" --tag "${ECR_REPO}:${IMAGE_TAG}" --file lambda_counterfactual/Dockerfile .
echo "  Image built: ${ECR_REPO}:${IMAGE_TAG}"

if [ "$DRY_RUN" = true ]; then
  echo ""
  echo "==> DRY RUN: Skipping ECR push and Lambda update."
  echo "    Image built successfully as ${ECR_REPO}:${IMAGE_TAG}"
  exit 0
fi

ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_IMAGE="${ECR_REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"

echo ""
echo "==> Authenticating to ECR (${ECR_REGISTRY})..."
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${AWS_REGION}" &>/dev/null || \
  aws ecr create-repository --repository-name "${ECR_REPO}" --region "${AWS_REGION}" > /dev/null

echo ""
echo "==> Tagging image: ${ECR_IMAGE}"
docker tag "${ECR_REPO}:${IMAGE_TAG}" "${ECR_IMAGE}"

echo "==> Pushing to ECR..."
docker push "${ECR_IMAGE}"
echo "  Pushed: ${ECR_IMAGE}"

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
    --timeout 600 \
    --memory-size 512 \
    --region "${AWS_REGION}" \
    --output json | python3 -c "import sys,json; d=json.load(sys.stdin); print('  FunctionArn:', d.get('FunctionArn','?'))"
fi


echo ""
echo "==> Waiting for Lambda update to complete..."
aws lambda wait function-updated --function-name "${LAMBDA_FUNCTION}" --region "${AWS_REGION}"

# ── Converge the Lambda environment (alpha-engine-config-I7925) ─────────────
# ${LAMBDA_FUNCTION} was one of eleven fleet-wide Lambdas carrying a
# GITHUB_TOKEN set by hand and refreshed by nothing — no repo, IaC file or
# script anywhere wrote this function's environment. The environment
# carried a STALE COPY of the credential — set from an older SSM
# parameter version and never re-derived on deploy — that GitHub rejected
# while the SSM parameter's own value remained valid the whole time
# (alpha-engine-config-I7968 tracks the mis-attribution). On 2026-08-21
# a first-party dependency picked that stale copy up out of site-packages,
# sent it to GitHub, got a 401, and halted the preopen
# trading pipeline (alpha-engine-config-I7924). alpha-engine-predictor-
# inference (crucible-predictor) was the one of eleven that broke and was
# fixed first (crucible-predictor-PR535) — this mirrors that convergence.
#
# DENY-LIST, deliberately, not an allow-list: this function carries
# operator-set flags codified nowhere, and asserting a complete key set
# would delete them. The allow-list end state is tracked separately
# (alpha-engine-config-I7958).
#
# Read-modify-write via the shared krepis.aws remove-lambda-env CLI — a
# bare aws lambda update-function-configuration --environment REPLACES the
# whole variable map.
#
# --defer-publish: this function is alias-pinned (below, 'live' is promoted
# only on canary success) and publish-version immediately below snapshots
# $LATEST into the published version — so the removal must land on
# $LATEST before that publish. Placed after publish-version it would
# mutate $LATEST only and be a silent no-op on the alias (L4497).
# --missing-ok: every deploy after the first finds the key already gone.
LAMBDA_ENV_DENIED_KEYS=(GITHUB_TOKEN)

echo ""
echo "==> Converging Lambda environment on ${LAMBDA_FUNCTION} (removing denied keys)..."
python3 -m krepis.aws remove-lambda-env \
  --function-name "${LAMBDA_FUNCTION}" --region "${AWS_REGION}" \
  --defer-publish --missing-ok \
  "${LAMBDA_ENV_DENIED_KEYS[@]/#/--unset=}"

echo ""
echo "==> Publishing Lambda version (do NOT promote 'live' yet)..."
VERSION=$(aws lambda publish-version --function-name "${LAMBDA_FUNCTION}" --query "Version" --output text --region "${AWS_REGION}")
echo "  Published version: ${VERSION}"

# Canary runs BEFORE promoting 'live' so a canary failure leaves the live
# alias pointing at the prior good version — no manual rollback owed.
# Pre-2026-05-22 these scripts promoted live first, ran canary second
# (filed in alpha-engine-config PR #272 as the L221-audit follow-up).
# Sibling research/predictor/data deploys already follow canary-first.
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
    --source "alpha-engine-backtester/infrastructure/deploy_counterfactual.sh" \
    --dedup-key "canary-fail-${LAMBDA_FUNCTION}-v${VERSION}" \
    --message "Canary failed: ${LAMBDA_FUNCTION}:${VERSION} canary returned status='${CANARY_STATUS}'. 'live' alias is UNCHANGED (still on prior good version) — no manual rollback owed; investigate and re-deploy. See CloudWatch /aws/lambda/${LAMBDA_FUNCTION}." \
    || true
  exit 1
fi
echo "  Canary passed (status=$CANARY_STATUS)"

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
