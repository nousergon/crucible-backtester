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
RUN pip install --no-cache-dir "alpha-engine-lib[flow_doctor] @ git+https://github.com/cipher813/alpha-engine-lib@v0.16.0" && \
    pip install --no-cache-dir -r requirements-health.txt && \
    rm -rf /root/.cache/pip

# Copy application code
COPY ssm_secrets.py .
COPY health_status.py .
COPY preflight.py .
COPY analysis/__init__.py analysis/
COPY analysis/production_health.py analysis/
COPY analysis/retrain_alert.py analysis/
COPY lambda_health/handler.py handler.py

# flow-doctor.yaml ships alongside the handler; handler.py resolves
# the absolute path via LAMBDA_TASK_ROOT and passes it to
# alpha_engine_lib.logging.setup_logging().
COPY flow-doctor.yaml .

# Lambda handler entry point
CMD ["handler.handler"]
