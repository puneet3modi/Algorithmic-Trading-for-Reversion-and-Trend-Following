# Algorithmic Trading for Trend-Following and Mean-Reversion
### CQF Final Project

---

## Overview

This repository implements a professional-grade **algorithmic trading research and execution framework** covering:

- Strategy design and backtesting
- Broker integration via REST APIs
- Live-style order loops with reconciliation
- Risk evaluation and validation

The architecture mirrors **institutional quantitative trading workflows**, with clear separation between:

- Research
- Execution
- Risk control

> ⚠️ **No real trading is performed.**  
> All live components run on **Binance Spot Testnet** using **far-from-market LIMIT orders** to avoid execution.

---

## Project Structure
```text
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
```

---

## Implemented Strategies

### 1. Trend-Following Strategies

#### MACD + Regime Filter

- Uses MACD histogram with normalization
- A regime filter suppresses trades in low-trend environments
- Incorporates:
  - Entry and exit hysteresis
  - Confirmation bars
  - Cooldown logic to prevent overtrading

This design allows the same MACD indicator to be used both as a trend detector and as a regime classifier, depending on the market state.

#### EMA Ratio Strategy

- Computes the ratio of price to a slow EMA
- Interpretation:
  - Ratio > 1 → Uptrend
  - Ratio < 1 → Downtrend
- Includes:
  - Multi-bar confirmation
  - Explicit flat regime
  - Clean long / flat mapping suitable for spot trading

This strategy follows the CQF-recommended ratio-based trend framework and emphasizes temporal stability.

---

### 2. Mean-Reversion Strategy

#### VWAP Reversion (Locked Configuration)

- Uses deviation from intraday VWAP
- Explicit volatility gating
- Strict holding-period limits
- Configuration is locked to ensure reproducibility and prevent overfitting

This strategy is only evaluated in regimes where mean reversion is empirically reasonable over the chosen horizon.

---

### 3. Machine Learning Strategy (Optional Extension)

- Supervised classifier trained offline
- Uses engineered features derived from technical indicators
- Evaluated strictly via historical backtesting
- No online learning or live inference

The ML model is treated as an auxiliary encoder rather than a primary trading engine.

---

## Data & Backtesting

- **Asset:** BTCUSDT
- **Frequency:** 15-minute bars
- **Data source:** Broker-compatible historical data

**Costs accounted for:**
- Trading fees
- Turnover
- Slippage proxy via far-order execution logic

All backtests are fully reproducible end-to-end using the research pipeline.

---

## Broker Integration (Part II)

**Broker:**
- Binance Spot Testnet

**API Type:**
- REST (HTTP)

**Order Types:**
- LIMIT orders only
- Prices placed far from the market to avoid execution

**Execution Safeguards:**
- Exchange filter enforcement:
  - `LOT_SIZE`
  - `PRICE_FILTER`
  - `MIN_NOTIONAL`
- Maximum open-orders constraint
- Automatic stale-order cancellation
- Strict position reconciliation before order placement

FIX connectivity is discussed conceptually in the written report but is not implemented.

---

## Live Loop & Reconciliation (Part III)

The live execution loop (`live_loop_testnet.py`) implements:

### Position Control

- Shadow position inferred from account balances
- Mapping of strategy signals to spot-feasible target positions
- Explicit handling of `-1 → flat` for spot markets

### Broker Reconciliation

- Open orders vs expected order IDs
- Detection of:
  - Partial fills
  - Unexpected fills
  - Unexpected open orders
- Safety actions:
  - Cancel unexpected orders
  - Skip trading when unsafe
  - Log all anomalies

### Audit Logging

- CSV-based logs for:
  - Order lifecycle events
  - Reconciliation snapshots
  - Errors and safety actions

---

## Risk Reporting

Running:
```bash
python scripts/report_risk_dashboard.py --all
```

Generates consolidated strategy-level metrics including:

- Sharpe ratio (gross and net)
- Volatility
- Value-at-Risk (VaR)
- Expected Shortfall (ES)
- Maximum drawdown
- Final equity
- Turnover
- Time-in-market

**Output file:**
```text
data/processed/BTCUSDT_15m_risk_dashboard.csv
```

---

## Installation

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Create a `.env` file:
```env
BINANCE_TESTNET_API_KEY=your_key
BINANCE_TESTNET_API_SECRET=your_secret
```

---

## Usage

### Run Full Research Pipeline
```bash
make pipeline
```

### Run Unit Tests
```bash
make test
```

### Test Broker Connectivity
```bash
make smoketest
```

### Run Live Loop (Single Safe Iteration)
```bash
python scripts/live_loop_testnet.py --once
```

### Generate Risk Dashboard
```bash
python scripts/report_risk_dashboard.py --all
```

### Full Validation Checklist
```bash
make checklist
```

---

## Docker (Optional)
```bash
make docker-build
make docker-up
make docker-run CMD="make checklist"
make docker-down
```

---

## Safety & Disclaimer

- No real funds are used
- All orders are intentionally non-executable
- This project is for academic and educational purposes only
- Not investment advice

---

## Final Notes

This repository demonstrates:

- End-to-end quantitative research
- Realistic broker interaction
- Robust order handling and reconciliation
- Professional risk controls
- Reproducible experimentation

---