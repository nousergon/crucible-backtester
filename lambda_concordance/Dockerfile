# Dockerfile — Lambda container image for weekly cheap-model concordance.
#
# Lightweight relative to the spot backtester: alpha-engine-lib +
# langchain_anthropic + boto3, no LightGBM / vectorbt / ArcticDB.
#
# Build (from repo root):
#   docker build --platform linux/amd64 -f lambda_concordance/Dockerfile -t alpha-engine-replay-concordance .
#
# Run locally (simulates Lambda):
#   docker run -p 9000:8080 alpha-engine-replay-concordance
#   curl -X POST http://localhost:9000/2015-03-31/functions/function/invocations \
#        -d '{"dry_run": true, "window_days": 14}'

FROM public.ecr.aws/lambda/python:3.12

# Install git — required for `pip install git+https://...` of
# alpha-engine-lib below. Same workaround the research Lambda uses
# (Lambda Python 3.12 base image does not include git; surfaced in
# the data-collector Lambda 2026-05-04 when the lib-public flip
# moved off vendored copies). Image-size impact: ~25MB.
RUN microdnf install -y git && microdnf clean all

# Install dependencies. alpha-engine-lib v0.4.0 ships agent_schemas
# (the canonical Pydantic contract surface that replay.runner
# validates against). [flow_doctor] extra plumbs flow-doctor's logging
# integration. No [arcticdb] / [rag] — concordance reads decision
# artifacts from S3 only.
#
# krepis[openai] (alpha-engine-config-I2997, 2026-07-19): replay/runner.py's
# target-model invocation migrated off langchain_anthropic.ChatAnthropic to
# krepis.llm.LLMClient's OpenRouter transport (DeepSeek V4 Flash) — the
# fleet's off-direct-Anthropic migration. Floor bumped past v0.12.2, the
# earliest tag carrying krepis.llm's OpenRouter unresolved/leaked-tool-call
# safety fix. Floor raised again to >=0.59.4 (alpha-engine-config-I7214): the
# first published krepis carrying krepis.stage_coverage, which handler.py
# imports to self-assert ReplayConcordance's declared output. The import is
# ImportError-guarded and observe-mode, so an older krepis in the image would
# make the assertion silently emit nothing rather than fail the build
# (config-I7334) — the floor is what keeps it measurable.
#
# It also carries the >=0.57.0 requirement of alpha-engine-config-I7179
# (krepis-PR140): LLMClient resolves a default cost_sink from
# KREPIS_COST_SINK_BUCKET / KREPIS_COST_SINK_PREFIX (wired by
# deploy_concordance.sh's merge-lambda-env step) when the caller passes
# none — this callsite never has, so every DeepSeek call this Lambda makes
# was previously landing on no cost record.
COPY lambda_concordance/requirements-concordance.txt requirements-concordance.txt
RUN pip install --no-cache-dir "nousergon-lib[flow-doctor] @ git+https://github.com/nousergon/nousergon-lib@v0.124.99" "krepis[openai]>=0.59.8" && \
    pip install --no-cache-dir -r requirements-concordance.txt && \
    rm -rf /root/.cache/pip

# nous-ergon-ops-I738: krepis-PR95 (v0.59.15/0.59.16, merged 2026-08-18)
# added an in-process DLP scan hook to krepis.llm.LLMClient that shells
# out to `gitleaks` and fails closed (LLMError DLP_SCAN_ERROR) when the
# binary is missing from PATH. The `krepis[openai]>=0.59.8` floor above
# has no ceiling, so a fresh image build resolves to 0.59.16 — and
# handler.py's `replay.batch` path calls replay/runner.py's
# `_invoke_target_with_schema`, which invokes the real
# krepis.llm.LLMClient.structured() LIVE against OpenRouter every weekly
# concordance run. Without gitleaks on PATH, every live production
# invocation would fail closed. Same pinned version/checksum as krepis's
# own CI step (krepis .github/workflows/test.yml, commit 3488b0dc); no
# `sudo` since this runs as root in the build.
RUN microdnf install -y tar gzip && microdnf clean all && \
    GITLEAKS_VERSION="8.30.1" && \
    GITLEAKS_SHA256="551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb" && \
    curl -sSL -o /tmp/gitleaks.tar.gz "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz" && \
    echo "${GITLEAKS_SHA256}  /tmp/gitleaks.tar.gz" | sha256sum -c - && \
    tar -xzf /tmp/gitleaks.tar.gz -C /tmp gitleaks && \
    chmod +x /tmp/gitleaks && \
    mv /tmp/gitleaks /usr/local/bin/gitleaks && \
    rm -f /tmp/gitleaks.tar.gz && \
    gitleaks version && \
    mkdir -p /opt/groom-llm-routing && \
    printf '[extend]\nuseDefault = true\n' > /opt/groom-llm-routing/gitleaks-egress.toml

# Application code — mirror lambda_health's per-file COPY pattern.
# Lambda Lambda task layout flattens lambda_concordance/handler.py to
# /var/task/handler.py (handler resolves LAMBDA_TASK_ROOT). replay/
# imports stay nested.
COPY preflight.py .
# stage_coverage_run_date.py — repo-ROOT single-file module resolving the
# partition key the stage-coverage verdict is written under
# (alpha-engine-config-I10171). handler.py imports it; omitting the COPY
# makes the whole coverage block ModuleNotFoundError in the deployed
# image — on the observer that exists to notice absences.
COPY stage_coverage_run_date.py .
# stage_substatus.py — repo-ROOT single-file module deriving the stage's
# status from its own sub-results (alpha-engine-config-I10198).
COPY stage_substatus.py .
COPY replay/ replay/

# flow-doctor.yaml ships alongside the handler; handler.py resolves
# the absolute path via LAMBDA_TASK_ROOT and passes it to
# alpha_engine_lib.logging.setup_logging().
COPY flow-doctor.yaml .

COPY lambda_concordance/handler.py handler.py

# Lambda handler entry point
CMD ["handler.handler"]
