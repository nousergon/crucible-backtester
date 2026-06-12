# Dockerfile — Lambda container image for weekly counterfactual rule fit.
#
# Lighter than the concordance Lambda — no langchain, no Anthropic. Pure
# sklearn + boto3 + alpha-engine-lib + replay/counterfactual.py.
#
# Build (from repo root):
#   docker build --platform linux/amd64 -f lambda_counterfactual/Dockerfile -t alpha-engine-replay-counterfactual .
#
# Run locally (simulates Lambda):
#   docker run -p 9000:8080 alpha-engine-replay-counterfactual
#   curl -X POST http://localhost:9000/2015-03-31/functions/function/invocations \
#        -d '{"dry_run": true, "window_days": 14}'

FROM public.ecr.aws/lambda/python:3.12

# Install git — required for `pip install git+https://...` of
# alpha-engine-lib below. Same workaround the rest of the Lambda fleet
# uses (Lambda Python 3.12 base image does not include git).
RUN microdnf install -y git && microdnf clean all

# Install dependencies. alpha-engine-lib v0.4.0 ships agent_schemas
# (used by replay/runner.py — counterfactual doesn't import it but
# replay/__init__.py shares the surface). [flow_doctor] extra plumbs
# flow-doctor's logging integration. No [arcticdb] / [rag] — counter-
# factual reads decision artifacts from S3 only.
COPY lambda_counterfactual/requirements-counterfactual.txt requirements-counterfactual.txt
RUN pip install --no-cache-dir "alpha-engine-lib[flow_doctor] @ git+https://github.com/cipher813/alpha-engine-lib@v0.59.3" && \
    pip install --no-cache-dir -r requirements-counterfactual.txt && \
    rm -rf /root/.cache/pip

# Application code — mirror lambda_concordance pattern. Lambda task
# layout flattens lambda_counterfactual/handler.py to /var/task/handler.py
# (handler resolves LAMBDA_TASK_ROOT). replay/ imports stay nested.
COPY preflight.py .
COPY replay/ replay/

# flow-doctor.yaml ships alongside the handler; handler.py resolves
# the absolute path via LAMBDA_TASK_ROOT and passes it to
# alpha_engine_lib.logging.setup_logging().
COPY flow-doctor.yaml .

COPY lambda_counterfactual/handler.py handler.py

# Lambda handler entry point
CMD ["handler.handler"]
