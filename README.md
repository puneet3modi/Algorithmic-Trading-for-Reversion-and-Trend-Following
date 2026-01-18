Here is a clean, submission-ready README.md tailored exactly to what you built and what CQF expects.
You can copy-paste this as-is into README.md.

⸻

Algorithmic Trading for Trend-Following and Mean-Reversion

CQF Final Project

Overview

This repository implements a professional-grade algorithmic trading research and execution framework covering:
	•	Strategy design & backtesting (trend-following, mean-reversion, ML classifier)
	•	Broker integration via REST API (Binance Spot Testnet)
	•	Live-style order loop with full reconciliation and risk controls
	•	Risk reporting and validation checklist

The project is structured to mirror institutional quant workflows, separating research, execution, and risk control layers.

⚠️ No real trading is performed.
All live components run against Binance Spot Testnet with far-from-market limit orders to avoid execution.

⸻

Project Structure

.
├── config/
│   └── base.yaml                 # Central configuration
│
├── data/
│   ├── raw/                      # Raw market data
│   ├── processed/                # Indicators, positions, backtests
│   └── live/                     # Live-loop logs & reconciliation
│
├── scripts/
│   ├── fetch_data.py
│   ├── run_qa.py
│   ├── compute_indicators.py
│   ├── compute_positions_*.py
│   ├── run_backtest_*.py
│   ├── compare_strategies.py
│   ├── report_risk_dashboard.py
│   ├── live_loop_testnet.py
│   ├── testnet_smoketest.py
│   └── reconcile_once_testnet.py
│
├── src/
│   ├── strategies/               # Strategy logic
│   ├── broker/                   # REST, order intent, reconciliation
│   ├── risk/                     # Risk metrics & reporting
│   └── common/                   # Logging & utilities
│
├── tests/                        # Unit tests
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── README.md


⸻

Implemented Strategies

1. Trend-Following
	•	MACD + Regime Filter
	•	EMA Ratio (price / slow EMA)
	•	Uses hysteresis, confirmation bars, and regime gating

2. Mean-Reversion
	•	VWAP Reversion (locked configuration)
	•	Explicit holding limits and volatility gating

3. Machine Learning (Optional Extension)
	•	Supervised classifier using engineered features
	•	Trained offline, evaluated via backtest

All strategies are evaluated on 15-minute BTCUSDT data with full cost accounting.

⸻

Broker & Execution (Part II)
	•	Broker: Binance Spot Testnet
	•	API Type: REST (HTTP)
	•	Order Type:
	•	LIMIT orders placed far from market (safety by design)
	•	Execution Safeguards:
	•	Quantized price/quantity using exchange filters
	•	MinQty & MinNotional enforcement
	•	Max open orders constraint
	•	Automatic stale order cancellation

No FIX connectivity is used; FIX is discussed conceptually in the report.

⸻

Live Loop & Reconciliation (Part III)

The live loop (live_loop_testnet.py) implements:
	•	Shadow position inference from account balances
	•	Desired vs shadow position reconciliation
	•	Broker-truth reconciliation:
	•	Open orders vs expected state
	•	Partial fills detection
	•	Unexpected trades detection
	•	Automatic safety actions:
	•	Cancel unexpected orders
	•	Log anomalies
	•	Skip trading when unsafe

All events are logged to CSV for auditability.

⸻

Risk Reporting

report_risk_dashboard.py --all generates:
	•	Gross & net:
	•	Sharpe
	•	Volatility
	•	VaR / Expected Shortfall
	•	Max drawdown
	•	Final equity
	•	Turnover & time-in-market
	•	Strategy-level comparison table

Outputs:

data/processed/BTCUSDT_15m_risk_dashboard.csv


⸻

Installation

1. Environment

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

2. Configuration

Create .env:

BINANCE_TESTNET_API_KEY=your_key
BINANCE_TESTNET_API_SECRET=your_secret


⸻

Usage

Run Full Research Pipeline

make pipeline

Run Unit Tests

make test

Test Broker Connectivity

make smoketest

Run Live Loop (single safe iteration)

python scripts/live_loop_testnet.py --once

Generate Risk Dashboard

python scripts/report_risk_dashboard.py --all

Full Validation Checklist

make checklist


⸻

Docker (Optional)

make docker-build
make docker-up
make docker-run CMD="make checklist"
make docker-down


⸻

Safety & Disclaimer
	•	No real funds are used
	•	All orders are intentionally non-executable
	•	This project is for academic and educational purposes only
	•	Not investment advice

⸻

Final Notes

This repository demonstrates:
	•	End-to-end quant research
	•	Realistic broker interaction
	•	Professional risk controls
	•	Reproducible experimentation

It is designed to be readable, auditable, and defensible in a CQF assessment setting.
