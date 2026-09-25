#!/usr/bin/env bash
# infrastructure/spot_parity.sh — Parity stage (pit_parity observational
# pass + backtester<->executor parity check) on a spot EC2 instance.
#
# alpha-engine-config-I4442: one of five scripts split out of the
# spot_backtest.sh monolith, mapped 1:1 to the Saturday SF's Parity state,
# which currently calls:
#
#   spot_backtest.sh --pit-parity-enabled=1 --skip-stages=backtest,evaluator{preflight_args}
#
# --skip-stages=backtest,evaluator does NOT include pit_parity, so BOTH the
# pit_parity pass (predictor-backtest mode, --pit-parity) and the parity
# pytest run in this SF state — the only one where pit_parity is enabled
# (L4486: gated on the `pit_parity` token so it runs in a fresh process with
# full RAM headroom, not stacked inside PredictorBacktest). This is the
# exact motivating incident for alpha-engine-config-I4442: on 2026-07-25
# Parity failed after 37 minutes because its trades_latest.db dependency
# wasn't checked until deep into the monolith's shared code path. This
# script's stage preflight asserts that dependency BEFORE spending on a
# spot boot.
#
# This PR does not wire the SF to this script (separate nousergon-data PR,
# sequenced after); spot_backtest.sh stays the currently-wired path,
# unchanged.
#
# Usage:
#   ./infrastructure/spot_parity.sh
#   ./infrastructure/spot_parity.sh --preflight-only
#   ./infrastructure/spot_parity.sh --run-date 2026-08-08
#   ./infrastructure/spot_parity.sh --help

set -euo pipefail
export HOME="${HOME:-/home/ec2-user}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPOT_STAGE_NAME="parity"

# shellcheck source=./_spot_common.sh
source "$SCRIPT_DIR/_spot_common.sh"
spot_common_init_defaults
_ORIG_ARGS=("$@")

usage() {
    cat <<'EOF'
spot_parity.sh — Parity stage (pit_parity observational pass +
backtester<->executor parity check) on a spot EC2 instance. Equivalent to:
  spot_backtest.sh --pit-parity-enabled=1 --skip-stages=backtest,evaluator

Depends on trades/trades_latest.db existing in S3 — the stage preflight
asserts this before spending on a spot (the 2026-07-25 incident this issue
is named for: Parity failed after 37 minutes on exactly this missing
dependency, unexercised until deep into the shared monolith).

Flags:
  --preflight-only        Boot + deps + smoke harness only, exit 0, zero spend
  --run-date DATE         Override RUN_DATE (default: today, normalized to NYSE trading day)
  --branch BRANCH         Git branch the spot clones (default: main)
  --instance-type TYPE    Override instance-type rotation
  --help                  Print this and exit 0 (no AWS calls made)
EOF
}

spot_common_parse_flags "$@"
if [ "$SHOW_HELP" = "1" ]; then
    usage
    exit 0
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  Parity Spot Run (stage=pit_parity+parity) — $(spot_common_stage_run_date)"
echo "═══════════════════════════════════════════════════════════════"
echo "  Branch        : $BRANCH"
echo "  Preflight-only: $PREFLIGHT_ONLY"
echo "  Spot attempt  : $SPOT_ATTEMPT/$MAX_SPOT_ATTEMPTS"
echo ""

spot_common_normalize_run_date

# ── Preflight checks ─────────────────────────────────────────────────────────
if [ "$PREFLIGHT_ONLY" != "1" ] && [ ! -f "$REPO_ROOT/config.yaml" ]; then
    echo "ERROR: config.yaml not found — copy from config.yaml.example"
    exit 1
fi
spot_common_resolve_executor_config
# pit_parity is observational and non-blocking by design (never fails the
# spot run) — predictor.yaml stays a soft-skip here, matching that posture.
# A missing predictor.yaml degrades pit_parity's own WARNING-and-continue
# path; it does not block the parity pytest, which is independent.
spot_common_resolve_predictor_config 0

# ── Stage-specific preflight (weekly-sf-policy.md §2.2) ──────────────────────
# THE motivating incident for I4442: on 2026-07-25 the Parity SF state
# failed 37 minutes in because trades_latest.db didn't exist in S3 — a
# check that existed in the monolith but only ran AFTER boot+deps+bootstrap.
# Assert it here, dispatcher-side, before provisioning a spot at all.
preflight_parity() {
    echo "==> Stage preflight: parity"
    if ! spot_common_s3_key_exists "trades/trades_latest.db"; then
        echo "ERROR: s3://${S3_BUCKET}/trades/trades_latest.db does not exist or is unreachable." >&2
        echo "       The parity pytest cannot run without it. Failing before spend — this is the exact" >&2
        echo "       class of miss that cost 37 minutes of spot boot on 2026-07-25 (alpha-engine-config-I4442)." >&2
        exit 1
    fi
    echo "  stage preflight OK (trades/trades_latest.db reachable)."
}
preflight_parity

echo "==> Dispatcher pre-launch preflight (fail-fast before provisioning spot)..."
spot_common_pre_launch_preflight \
    "$REPO_ROOT/backtest.py" \
    "$REPO_ROOT/preflight.py" \
    "$REPO_ROOT/pipeline_common.py" \
    "$REPO_ROOT/synthetic/predictor_backtest.py"

# pit_parity's predictor pass shares the universal >=16 GB floor (I3280;
# subprocess isolation bounds its per-pass footprint to ~2.8 GB, same
# guard as the other predictor-bearing stages) unless the operator passed
# an explicit --instance-type.
spot_common_apply_large_universe_ram_floor
echo "  Instance types: $INSTANCE_TYPES"

spot_common_launch_instance   # arms the cleanup EXIT trap before it launches (alpha-engine-config-I11573)

echo "==> Waiting for instance to enter running state..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$AWS_REGION"

spot_common_stage_configs
spot_common_wait_for_ssm_agent
spot_common_bootstrap
spot_common_install_deps
spot_common_fetch_predictor_cache
spot_common_build_env_source
spot_common_run_preflight_only_and_maybe_exit

# ── Stage: pit_parity + parity ───────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  PARITY (pit_parity + backtester<->executor parity)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

run_ssm "parity" "$MAX_RUNTIME_SECONDS" <<PARITY
set -eo pipefail
cd /home/ec2-user/alpha-engine-backtester
${ENV_SOURCE}

RUN_DATE="${RUN_DATE}"
BUCKET="\$(\$PYTHON_BIN -c 'import yaml,sys; print((yaml.safe_load(open(\"config.yaml\")) or {}).get(\"output_bucket\") or \"alpha-engine-research\")' 2>/dev/null || echo alpha-engine-research)"

# ── pit_parity (observational, NEVER fails the spot run) ────────────────────
# Runs unconditionally in this script — this IS the Parity state, the one
# place pit_parity fires (L4486). A tiny-slice smoke (config#3121) proves
# imports/config/subprocess wiring in well under a minute before the real
# pass; an ImportError here fails fast instead of ~2h into the real run.
echo "▶ stage=smoke-pit-parity START at \$(date -u +%H:%M:%S)"
if ! $REMOTE_PYTHON -u backtest.py --mode smoke-pit-parity --date "\${RUN_DATE}" --log-level INFO 2>&1; then
    echo "WARNING: smoke-pit-parity FAILED — pit_parity's import/config/subprocess wiring is broken. Continuing (pit_parity is non-blocking) but the real pass below will likely fail the same way." >&2
fi
echo "▶ stage=smoke-pit-parity END at \$(date -u +%H:%M:%S)"

echo "▶ stage=pit_parity START at \$(date -u +%H:%M:%S) (observational, non-blocking)"
# Swallow on non-zero exit per feedback_no_silent_fails secondary-
# observability carve-out: (a) failure mode swallowed = pit_parity
# exception path; (b) primary deliverable survives = the parity pytest
# below is independent of pit_parity; (c) concrete recording surfaces =
# (1) S3 artifact at backtest/{date}/pit_parity.json with status=failed,
# always emitted by backtest.py::main's pit_parity branch; (2) Telegram +
# SNS alert via alpha_engine_lib.alerts (sev=warning, dedup-keyed on run_date).
$REMOTE_PYTHON -u backtest.py --mode predictor-backtest --pit-parity \\
    --date "\${RUN_DATE}" --log-level INFO 2>&1 \\
    || echo "WARNING: pit_parity stage failed (observational — spot run continues; failure-artifact + Telegram alert published by the inner Python)"
echo "▶ stage=pit_parity END at \$(date -u +%H:%M:%S)"

# ── parity (observability, not a gate) ───────────────────────────────────────
# Each run produces parity_report.json (per-run drill-down) +
# parity_metrics.csv (time-series append). The spot run does NOT fail the SF
# on parity divergence — 0% historical parity is structurally unreachable
# for a system with weekly auto-tuned configs and evolving executor code
# (see tests/test_parity_replay.py module docstring). Setup-level failures
# (missing trades.db, ArcticDB unreachable) ARE still fatal here — real
# infrastructure breakage, not "expected drift".
echo "▶ stage=smoke-parity START at \$(date -u +%H:%M:%S)"
if ! $REMOTE_PYTHON -m pytest tests/test_parity_replay.py -m parity --collect-only -q 2>&1; then
    echo "WARNING: smoke-parity FAILED — test_parity_replay.py failed to import/collect. Continuing (parity is non-blocking) but the real pass below will likely fail the same way." >&2
fi
echo "▶ stage=smoke-parity END at \$(date -u +%H:%M:%S)"

echo "▶ stage=parity START at \$(date -u +%H:%M:%S)"
PARITY_TRADES_DB="/tmp/trades_latest.db"
PARITY_REPORT_DIR="/tmp/parity_report"
mkdir -p "\$PARITY_REPORT_DIR"

if ! aws s3 cp "s3://\${BUCKET}/trades/trades_latest.db" "\$PARITY_TRADES_DB" --quiet; then
    echo "ERROR: could not download trades_latest.db from S3 — parity cannot run" >&2
    echo "       This is infrastructure breakage (not divergence) — failing spot." >&2
    exit 1
fi

PARITY_EXIT=0
TRADES_DB_PATH="\$PARITY_TRADES_DB" \\
SIGNALS_BUCKET="\${BUCKET}" \\
PARITY_REPORT_DIR="\$PARITY_REPORT_DIR" \\
PARITY_RUN_DATE="\${RUN_DATE}" \\
USE_REAL_ARCTICDB=1 \\
$REMOTE_PYTHON -m pytest tests/test_parity_replay.py -m parity -v 2>&1 || PARITY_EXIT=\$?

if [ -f "\$PARITY_REPORT_DIR/parity_report.json" ]; then
    aws s3 cp "\$PARITY_REPORT_DIR/parity_report.json" \\
        "s3://\${BUCKET}/backtest/\${RUN_DATE}/parity_report.json" --quiet \\
        && echo "Uploaded parity_report.json to s3://\${BUCKET}/backtest/\${RUN_DATE}/" \\
        || echo "WARNING: failed to upload parity_report.json (non-fatal)"
else
    # alpha-engine-config#7199 — THE ABSENCE IS THE DEFECT. Until now a missing
    # parity_report.json was indistinguishable from a skipped upload: the guard
    # had no else-branch, the non-zero pytest exit degraded to a WARNING on
    # stderr, and the stage returned 0. That is how parity_report.json went
    # unwritten from 2026-07-24 to 2026-08-13 while the sibling report.md in the
    # same prefix kept updating weekly and nothing noticed. (Root cause: PR #550
    # stopped co-installing the predictor requirements, which is where pytest
    # came from; "No module named pytest", exit in 1 second, SF success path.)
    #
    # An absent artifact now becomes a PRESENT artifact that says it is absent,
    # so every downstream reader renders FAILED rather than a silently stale
    # ABSENT row. sf-pipeline-policy 2.3a rule 2: a missing verdict is never a
    # pass.
    echo "ERROR: the parity replay produced NO parity_report.json (producer exit=\$PARITY_EXIT)." >&2
    echo "       The backtester-to-executor fill-parity claim is UNPROVEN for \${RUN_DATE}." >&2
    printf '{"schema":"parity_report-0.0.0","run_date":"%s","status":"failed","verdict":"FAIL","verdict_reason":"the parity replay produced no report this run (producer exit=%s) - the backtester-to-executor fill-parity claim is unproven","producer_exit":%s}\\n' \\
        "\${RUN_DATE}" "\$PARITY_EXIT" "\$PARITY_EXIT" > /tmp/parity_report_missing.json
    aws s3 cp /tmp/parity_report_missing.json \\
        "s3://\${BUCKET}/backtest/\${RUN_DATE}/parity_report.json" --quiet \\
        && echo "       Recorded the absence at s3://\${BUCKET}/backtest/\${RUN_DATE}/parity_report.json" \\
        || echo "       WARNING: could not record the absence marker either." >&2
    # Deliberately NOT exiting non-zero here: this launcher still runs co-tenant
    # stages behind one script, and killing stages that do not consume the
    # parity artifact would trade one blindness for an outage
    # (sf-pipeline-policy 2.1). The recorded absence above is the surface; the
    # split ParityReplay branch (spot_parity_replay.sh) is where this fails the
    # stage, because there it costs only that branch.
fi

if [ "\$PARITY_EXIT" != "0" ]; then
    echo "WARNING: parity pytest exited \$PARITY_EXIT (likely setup-side error)." >&2
    echo "         See s3://\${BUCKET}/backtest/\${RUN_DATE}/parity_report.json (if present)." >&2
    echo "         Continuing spot run — parity is observability, not a gate." >&2
fi
echo "▶ stage=parity END at \$(date -u +%H:%M:%S)"

echo ""
echo "Parity stage complete at \$(date)"
PARITY

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Parity complete. Instance will be terminated."
echo "═══════════════════════════════════════════════════════════════"

# No CloudWatch heartbeat emitted here — see spot_predictor_backtest.sh's
# identical comment. Tracked: alpha-engine-config-I6710.
