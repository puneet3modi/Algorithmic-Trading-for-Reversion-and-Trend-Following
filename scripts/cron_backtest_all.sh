#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root (cron often starts elsewhere)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Activate venv (explicit path inside repo)
if [[ ! -f ".venv/bin/activate" ]]; then
    echo "ERROR: .venv not found at $REPO_ROOT/.venv"
    echo "Create it first: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi
source ".venv/bin/activate"

# Ensure logs directory exists
mkdir -p logs

# Log everything (both stdout + stderr) with timestamped file
LOG_FILE="logs/cron_backtest_all_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== START cron_backtest_all.sh (UTC) ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"
echo "Repo: $REPO_ROOT"
python --version
echo "========================================"

python scripts/run_qa.py
python scripts/compute_indicators.py

python scripts/compute_positions_macd.py
python scripts/compute_positions_ema_ratio.py
python scripts/compute_positions_reversion_vwap.py

python scripts/run_backtest_macd.py
python scripts/run_backtest_ema_ratio.py
python scripts/run_backtest_reversion_vwap.py

# ML optional: comment out if you don't want it scheduled every run
python scripts/train_classifier_and_backtest.py

python scripts/compare_strategies.py

# Risk dashboard/report summary (only if script exists)
if [[ -f "scripts/report_risk_dashboard.py" ]]; then
    python scripts/report_risk_dashboard.py --all
else
    echo "NOTE: scripts/report_risk_dashboard.py not found; skipping."
fi

echo "=== DONE: full offline pipeline completed (UTC) ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"
echo "Log: $LOG_FILE"
