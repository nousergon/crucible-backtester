#!/bin/bash
# Post-clone setup: venv, deps, log file, cron.
# Run after git clone.
#
# Usage (from EC2, after cloning):
#   bash ~/alpha-engine-backtester/infrastructure/setup-ec2.sh

set -euo pipefail

REPO_DIR="/home/ec2-user/alpha-engine-backtester"

echo "=== Alpha Engine Backtester — EC2 setup ==="

cd "$REPO_DIR"

# ── 1. Virtualenv ─────────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "Creating virtualenv..."
    python3.11 -m venv .venv
fi

echo "Installing dependencies..."
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

# ── 2. Log file ───────────────────────────────────────────────────────────────
sudo touch /var/log/backtester.log
sudo chown ec2-user:ec2-user /var/log/backtester.log

# ── 3. Cron ───────────────────────────────────────────────────────────────────
bash "$REPO_DIR/infrastructure/add-cron.sh"

echo ""
echo "=== Setup complete ==="
echo "Test: cd $REPO_DIR && .venv/bin/python evaluate.py --mode diagnostics --freeze"
echo "Logs: tail -f /var/log/backtester.log"
