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

# ── Step 4b: Apply router addressing (merge, not replace) ───────────────────
# Mirrors crucible-research/infrastructure/deploy.sh::_apply_router_env
# (alpha-engine-config-I6373). Read-modify-write because
# update-function-configuration --environment REPLACES the whole map, and
# this Lambda's other env vars (FMP_API_KEY, RAG_DATABASE_URL, etc. — see
# alpha-engine-config-I6377, the same uncodified-env gap on the sibling
# research Lambda) are not owned by this script.
_apply_router_env() {
  local fn="$1"
  echo "  Applying router addressing to $fn (merge, not replace)..."

  # Lambda serializes updates per function; a configuration update issued
  # while LastUpdateStatus=InProgress fails ResourceConflictException, which
  # aborts the whole deploy under set -euo pipefail. Wait first.
  aws lambda wait function-updated --function-name "$fn" --region "${AWS_REGION}" 2>/dev/null || sleep 5

  local tmp_cur tmp_new
  tmp_cur="$(mktemp)"; tmp_new="$(mktemp)"
  # shellcheck disable=SC2064
  trap "rm -f '$tmp_cur' '$tmp_new'" RETURN

  aws lambda get-function-configuration \
    --function-name "$fn" --region "${AWS_REGION}" \
    --query "Environment.Variables" --output json > "$tmp_cur" 2>/dev/null \
    || echo '{}' > "$tmp_cur"

  ROUTER_URL="${ROUTER_URL:-https://router.nousergon.ai:8443}" \
  ROUTER_CREDENTIAL_SECRET="${ROUTER_CREDENTIAL_SECRET:-ROUTER_CONSUMER_REPLAY}" \
  python3 - "$tmp_cur" "$tmp_new" <<'PYEOF'
import json, os, sys

cur_path, new_path = sys.argv[1], sys.argv[2]
with open(cur_path) as fh:
    variables = json.load(fh) or {}

# exec_context names WHERE CODE RUNS (model-router-policy R28). The registry
# declares `lambda` on NO model entry, deliberately -- a Lambda has no local
# egress proxy and no private-network peer, so the router is its only path
# and this call site FAILS CLOSED rather than reaching a provider unscanned.
variables.update({
    "KREPIS_EXEC_CONTEXT": "lambda",
    "KREPIS_LITELLM_PROXY_URL": os.environ["ROUTER_URL"],
    # Its OWN secret, not LITELLM_MASTER_KEY: the edge identifies a consumer
    # BY its credential VALUE and krepis.secrets resolves SSM BEFORE
    # os.environ, so a shared name collapses this Lambda into another
    # consumer's identity at the edge however the environment is set.
    "KREPIS_ROUTER_CREDENTIAL_SECRET": os.environ["ROUTER_CREDENTIAL_SECRET"],
    # crucible-backtester is PUBLIC, so private-docs/LLM_MODEL_REGISTRY.yaml
    # is correctly absent from the image. All three are required: krepis'
    # AppConfig path is opt-in and SWALLOWS its own errors, falling through
    # to a filesystem walk that finds nothing here.
    "KREPIS_APPCONFIG_APPLICATION": "alpha-engine",
    "KREPIS_APPCONFIG_CONFIG_PROFILE": "llm-model-registry",
    "KREPIS_APPCONFIG_ENVIRONMENT": "production",
})

with open(new_path, "w") as fh:
    json.dump({"Variables": variables}, fh)
print(f"    merged 5 router variables into {len(variables)} total (values not shown)")
PYEOF

  aws lambda update-function-configuration \
    --function-name "$fn" \
    --environment "file://$tmp_new" \
    --region "${AWS_REGION}" > /dev/null
  aws lambda wait function-updated --function-name "$fn" --region "${AWS_REGION}" 2>/dev/null || sleep 5
  echo "    router addressing applied."
}

_apply_router_env "${LAMBDA_FUNCTION}"

# ── Step 5: Wait for update to complete ──────────────────────────────────────
echo ""
echo "==> Waiting for Lambda update to complete..."
aws lambda wait function-updated --function-name "${LAMBDA_FUNCTION}" --region "${AWS_REGION}"

# ── Step 5b: Merge cost-sink environment onto the function ──────────────────
# replay/runner.py's _invoke_target_model builds krepis.llm.LLMClient with no
# cost_sink (alpha-engine-config-I7179) — every DeepSeek call this Lambda
# makes was landing on no cost record. The fix is NOT a per-call-site
# cost_sink= at the runner.py call — that reproduces the gap for the next
# call site added. krepis>=0.57.0 (krepis-PR140) makes LLMClient resolve a
# default sink from these two env vars when cost_sink is not supplied.
# merge-lambda-env is read-modify-write: it preserves every var already on
# the live function. Must run BEFORE publish-version so the version this
# deploy promotes actually carries the sink config, not the version after it.
echo ""
echo "==> Merging cost-sink environment onto ${LAMBDA_FUNCTION}..."
python3 -m krepis.aws merge-lambda-env --function-name alpha-engine-replay-concordance --set KREPIS_COST_SINK_BUCKET=alpha-engine-research --set KREPIS_COST_SINK_PREFIX=decision_artifacts/_cost_raw --region "${AWS_REGION}"

# ── Step 5c: Converge the Lambda environment (alpha-engine-config-I7925) ────
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
# operator-set flags and provider keys codified nowhere (KREPIS_ROUTER_
# CREDENTIAL_SECRET's resolved value, etc.), and asserting a complete key
# set would delete them. The allow-list end state is tracked separately
# (alpha-engine-config-I7958).
#
# Read-modify-write via the shared krepis.aws remove-lambda-env CLI — a
# bare aws lambda update-function-configuration --environment REPLACES the
# whole variable map.
#
# --defer-publish: this function is alias-pinned (Step 8 below promotes
# 'live' only on canary success) and Step 6 immediately below snapshots
# $LATEST into the published version — so the removal must land on
# $LATEST before that publish, same as the cost-sink merge just above.
# Placed after publish-version it would mutate $LATEST only and be a
# silent no-op on the alias (L4497). --missing-ok: every deploy after the
# first finds the key already gone.
#
# ANTHROPIC_API_KEY (alpha-engine-config-I6789): a provider credential is a
# router bypass (llm-egress-proxy-policy §2a). Nothing this function runs
# reads it: lambda_concordance/handler.py lists no provider key, and the
# target-model call authenticates to the router edge with this function's
# own KREPIS_ROUTER_CREDENTIAL_SECRET (I2997, I7878). The key survived only
# because _apply_router_env above carries every live variable forward.
LAMBDA_ENV_DENIED_KEYS=(GITHUB_TOKEN ANTHROPIC_API_KEY)

echo ""
echo "==> Converging Lambda environment on ${LAMBDA_FUNCTION} (removing denied keys)..."
python3 -m krepis.aws remove-lambda-env \
  --function-name "${LAMBDA_FUNCTION}" --region "${AWS_REGION}" \
  --defer-publish --missing-ok \
  "${LAMBDA_ENV_DENIED_KEYS[@]/#/--unset=}"

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
