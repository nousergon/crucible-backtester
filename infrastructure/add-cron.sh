#!/bin/bash
# Register the weekly backtester cron job.
# Safe to run multiple times — replaces existing entry.
#
# The backtester runs on a spot instance launched from the always-on EC2.
# The spot_backtest.sh script handles the full lifecycle:
#   launch spot → clone repos → install deps → run backtest → terminate
#
# Schedule: Saturdays at 08:00 UTC (2 hours after research pipeline starts at 06:00)
# Saturday gives a weekend buffer to fix pipeline issues before Monday trading.
#
# Secrets sourced from ~/.alpha-engine.env (shared with executor).
#
# Usage:
#   bash infrastructure/add-cron.sh

set -euo pipefail

ENV_FILE="/home/ec2-user/.alpha-engine.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: ${ENV_FILE} not found."
    echo "Create it with GMAIL_APP_PASSWORD, then chmod 600."
    exit 1
fi

SOURCE_ENV=". ${ENV_FILE} &&"

# Path to the alpha-engine-config checkout on the dispatcher. The backtester's
# config.yaml is a symlink into this checkout and spot_backtest.sh also stages
# executor/risk.yaml from here (see spot_backtest.sh), so config edits only
# reach the running system if this checkout is pulled each run (config#822).
CONFIG_REPO="/home/ec2-user/alpha-engine-config"

# Launch spot instance for the full backtest (10y data, param sweep, upload results).
# Pull alpha-engine-config FIRST so config edits (backtester guardrails, risk.yaml,
# predictor.yaml, the symlinked config.yaml) propagate on every run — without this
# the dispatcher silently runs whatever stale config it last had (config#822). The
# pull is fail-LOUD but non-blocking: a failed pull logs a warning and the backtest
# still runs on the last-good config (better than skipping the weekly run entirely),
# rather than silently lagging.
CRON_LINE="0 8 * * 6  { git -C ${CONFIG_REPO} pull --ff-only || echo 'WARN(add-cron): alpha-engine-config pull failed — backtest may run on STALE config (config#822)'; } >> /var/log/backtester.log 2>&1 && cd /home/ec2-user/alpha-engine-backtester && git pull --ff-only >> /var/log/backtester.log 2>&1 && ${SOURCE_ENV} bash infrastructure/spot_backtest.sh >> /var/log/backtester.log 2>&1"

# Remove existing backtester entry, then add new one
EXISTING=$(crontab -l 2>/dev/null || true)
FILTERED=$(echo "$EXISTING" | grep -v "alpha-engine-backtester" || true)

{
    echo "$FILTERED"
    echo "$CRON_LINE"
} | crontab -

echo "Backtester cron job registered: Saturdays 08:00 UTC"
echo "  Mode: spot instance (launched from always-on EC2)"
echo "  Secrets: sourced from ${ENV_FILE}"
echo ""
echo "Current crontab:"
crontab -l
