from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import yaml

sys.path.append(os.path.abspath("."))

from src.common.logging import setup_logger
from src.backtest.simple_backtest import BacktestParams, run_backtest, annualized_sharpe, max_drawdown

COST_PER_TURNOVER = 0.0002   # 2 bps
EXECUTION_LAG = 1            # trade next bar
BARS_PER_YEAR = 365 * 24 * 4 # 15m bars


def main() -> None:
    logger = setup_logger("run_backtest_reversion_vwap")

    with open("config/base.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    symbol = cfg["symbol"]
    interval = cfg["timebars"]["interval"]
    processed_dir = cfg["paths"]["processed_dir"]

    pos_path = f"{processed_dir}/{symbol}_{interval}_reversion_vwap_positions.csv"
    out_path = f"{processed_dir}/{symbol}_{interval}_backtest_reversion_vwap.csv"

    df = pd.read_csv(pos_path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").set_index("open_time")

    # Backtest expects 'close' and 'position'
    bt_df = pd.DataFrame(index=df.index)
    bt_df["close"] = df["close"].astype(float)
    bt_df["position"] = df["position_reversion"].astype(int)

    bt = run_backtest(bt_df, BacktestParams(cost_per_turnover=COST_PER_TURNOVER, execution_lag=EXECUTION_LAG))

    sharpe_g = annualized_sharpe(bt["strat_ret_gross"], BARS_PER_YEAR)
    sharpe_n = annualized_sharpe(bt["strat_ret_net"], BARS_PER_YEAR)
    mdd_g = max_drawdown(bt["equity_gross"])
    mdd_n = max_drawdown(bt["equity_net"])
    turnover = float(bt["turnover"].sum())
    pct_mkt = float((bt["pos_exec"].abs() > 0).mean())

    logger.info(f"Sharpe gross: {sharpe_g:.3f} | Sharpe net: {sharpe_n:.3f}")
    logger.info(f"Max DD gross: {mdd_g:.3%} | Max DD net: {mdd_n:.3%}")
    logger.info(f"Total turnover: {turnover:.0f}")
    logger.info(f"Pct time in market: {pct_mkt:.3f}")

    bt_out = bt[["equity_gross", "equity_net", "pos_exec", "turnover"]].copy()
    bt_out.to_csv(out_path)

    logger.info(f"Saved backtest output: {out_path}")
    logger.info("Tail equity:")
    logger.info(bt_out[["equity_gross", "equity_net"]].tail().to_string())


if __name__ == "__main__":
    main()
