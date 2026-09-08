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
#
# krepis floor raised >=0.3.0 -> >=0.59.4 (alpha-engine-config-I7214): the
# first published krepis carrying krepis.stage_coverage, which handler.py
# imports to self-assert Counterfactual's declared output before returning.
# That import is ImportError-guarded and observe-mode by design, so an image
# built against the old floor would emit NOTHING and look exactly like an
# assertion that found nothing wrong (config-I7334). The floor is the only
# thing making the assertion measurable in the deployed artifact.
COPY lambda_counterfactual/requirements-counterfactual.txt requirements-counterfactual.txt
RUN pip install --no-cache-dir "nousergon-lib[flow-doctor] @ git+https://github.com/nousergon/nousergon-lib@v0.124.99" "krepis>=0.59.4" && \
    pip install --no-cache-dir -r requirements-counterfactual.txt && \
    rm -rf /root/.cache/pip

# Application code — mirror lambda_concordance pattern. Lambda task
# layout flattens lambda_counterfactual/handler.py to /var/task/handler.py
# (handler resolves LAMBDA_TASK_ROOT). replay/ imports stay nested.
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

COPY lambda_counterfactual/handler.py handler.py

# Lambda handler entry point
CMD ["handler.handler"]
