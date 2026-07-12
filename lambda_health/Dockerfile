# Dockerfile — Lambda container image for daily predictor health check.
#
# Lightweight: numpy/pandas/scipy/boto3 + alpha-engine-lib (~130MB).
# No LightGBM, CatBoost, ArcticDB, or vectorbt.
#
# Build (from repo root):
#   docker build --platform linux/amd64 -f lambda_health/Dockerfile -t alpha-engine-health-check .
#
# Run locally (simulates Lambda):
#   docker run -p 9000:8080 alpha-engine-health-check
#   curl -X POST http://localhost:9000/2015-03-31/functions/function/invocations \
#        -d '{"dry_run": true}'

FROM public.ecr.aws/lambda/python:3.12

# Copy and install Python requirements first for better layer caching.
# alpha-engine-lib is installed from public git+https with [flow_doctor]
# extra (lib was flipped public 2026-05-03; previous versions vendored
# a local copy via deploy_health.sh staging). No [arcticdb] — health
# Lambda reads S3 only.
COPY requirements-health.txt .
# `git` is required for the git+https install below and is NOT present in
# the AL2023-based Lambda python base image. Omitting it silently broke
# every `deploy_health.sh` build after the 2026-05-03 lib-public flip
# switched this line from a vendored local copy to git+https — which is
# why the health Lambda image stayed frozen at its 2026-04-08 build and
# re-raised the already-fixed 2026-05-11 false-positive retrain alert.
RUN microdnf install -y git && microdnf clean all
RUN pip install --no-cache-dir "nousergon-lib[flow_doctor] @ git+https://github.com/nousergon/nousergon-lib@v0.96.0" "krepis>=0.3.0" && \
    pip install --no-cache-dir -r requirements-health.txt && \
    rm -rf /root/.cache/pip

# Copy application code
COPY preflight.py .
COPY analysis/__init__.py analysis/
COPY analysis/production_health.py analysis/
COPY analysis/retrain_alert.py analysis/
# production_health.py imports the canonical COALESCE / horizon-filter SQL
# constants + ACTIVE_HORIZON_DAYS from pipeline_common (introduced by the
# #178/#180 canonical-alpha analytics arc). The Dockerfile was never
# updated to ship it, so the first post-#178 build raised
# ModuleNotFoundError: pipeline_common at runtime. Slim-safe: its
# module-level imports are stdlib + boto3/pandas/yaml only.
COPY pipeline_common.py .
COPY lambda_health/handler.py handler.py

# flow-doctor.yaml ships alongside the handler; handler.py resolves
# the absolute path via LAMBDA_TASK_ROOT and passes it to
# alpha_engine_lib.logging.setup_logging().
COPY flow-doctor.yaml .

# Lambda handler entry point
CMD ["handler.handler"]
