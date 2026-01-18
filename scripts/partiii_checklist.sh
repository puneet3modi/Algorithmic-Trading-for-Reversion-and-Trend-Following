#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

echo "1) Smoketest"
python scripts/testnet_smoketest.py

echo "2) Live loop once (no babysitting)"
python scripts/live_loop_testnet.py --once

echo "3) Risk dashboard"
python scripts/report_risk_dashboard.py --all

echo "OK: Part III checklist passed"
