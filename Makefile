.PHONY: help venv test lint fmt strategies ml live smoketest docker-build docker-up docker-run docker-down pipeline pipeline-docker live-testnet

help:
	@echo "Targets:"
	@echo "  pipeline        Fetch+QA+Indicators+Positions+Backtests+Compare"
	@echo "  strategies      Run all non-ML strategies"
	@echo "  ml              Train ML classifier + backtest"
	@echo "  live            Run testnet live loop (paper-style far orders)"
	@echo "  smoketest       Testnet REST connectivity checks"
	@echo "  test            Run unit tests"
	@echo "  docker-build    Build Docker image"
	@echo "  docker-up       Start container (compose)"
	@echo "  docker-run CMD=...   Run a command in the container"
	@echo "  docker-down     Stop container"

pipeline:
	python scripts/run_pipeline.py

strategies:
	python scripts/fetch_data.py
	python scripts/run_qa.py
	python scripts/compute_indicators.py
	python scripts/compute_positions_macd.py
	python scripts/run_backtest_macd.py
	python scripts/compute_positions_ema_ratio.py
	python scripts/run_backtest_ema_ratio.py
	python scripts/compute_positions_reversion_vwap.py
	python scripts/run_backtest_reversion_vwap.py
	python scripts/compare_strategies.py

ml:
	python scripts/train_classifier_and_backtest.py

live:
	python scripts/live_loop_testnet.py

smoketest:
	python scripts/testnet_smoketest.py

test:
	python -m pytest -q

docker-build:
	docker build -t cqf-algo:latest .

docker-up:
	docker compose up -d --build

docker-run:
	docker compose run --rm cqf bash -lc "$(CMD)"

docker-down:
	docker compose down

pipeline-docker:
	docker compose run --rm cqf make pipeline

live-testnet:
	python scripts/live_loop_testnet.py

checklist:
	python scripts/testnet_smoketest.py
	python scripts/reconcile_once_testnet.py
	python scripts/live_loop_testnet.py --once
	python scripts/report_risk_dashboard.py --all
	